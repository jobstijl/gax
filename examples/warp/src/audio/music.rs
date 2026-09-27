//! Generative music, synthesized live: restrained, textural, slowly evolving.
//!
//! * A modal pitch set per run (from the seed), a fixed tempo, and a sample-accurate sequencer
//!   (it counts samples inside the audio thread; no frame timers).
//! * A drone/pad of detuned saws through a slowly moving low-pass carries the harmony: chords of
//!   stacked thirds with 9ths and 11ths, changing every 8 or 16 bars, with voice leading that
//!   moves each voice to the nearest tone of the next chord (and glides there).
//! * A sub/pluck bass on a sparse Euclidean pattern, and soft percussion (kick, hats) whose
//!   density follows the intensity.
//! * The game steers it: intensity (filter, density, layers, how much dissonance the voicing
//!   allows), a death (everything drops to the drone, then rebuilds over 8 bars), a bomb (a
//!   filter sweep and a duck), a nearby singularity (pitch drifts down, the tone darkens). The
//!   tempo never changes.

use super::dsp::{Env, Noise, Osc, Smooth, Svf, db, mtof, soft};
use super::sfx::Scale;

const PAD_VOICES: usize = 5;

#[derive(Clone, Copy, Debug, Default)]
struct PadVoice {
    osc: [Osc; 3],
    note: Smooth,
    target: f32,
    gain: Smooth,
    on: f32,
}

/// Control input from the game.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Controls {
    /// Intensity, `0..1`.
    pub intensity: f32,
    /// How close a singularity is, `0..1`.
    pub darkness: f32,
    /// Whether a run is being played (off: the title's calm).
    pub playing: bool,
}

/// The music engine.
pub struct Music {
    sr: f32,
    /// Beats per minute.
    pub tempo: f32,
    /// The pitch material.
    pub scale: Scale,
    rng: Noise,
    clock: f32,
    /// Sixteenths since the start.
    pub step: u64,
    chord_root: i32,
    next_change: u64,
    pad: [PadVoice; PAD_VOICES],
    pad_filter: [Svf; 2],
    lfo: Osc,
    bass_osc: Osc,
    bass_sub: Osc,
    bass_env: Env,
    bass_filter_env: Env,
    bass_svf: Svf,
    bass_note: f32,
    kick_osc: Osc,
    kick_env: Env,
    kick_pitch: Env,
    hat_env: Env,
    hat_svf: Svf,
    click_env: Env,
    noise: Noise,
    controls: Controls,
    intensity: Smooth,
    darkness: Smooth,
    rebuild: f32,
    duck: Smooth,
    sweep: f32,
    bass_rot: u32,
    hat_rot: u32,
}

/// `E(k, n)`: is step `i` a hit of the Euclidean rhythm with `k` hits in `n` steps, rotated by
/// `rot`?
pub fn euclid(i: u32, k: u32, n: u32, rot: u32) -> bool {
    k > 0 && (((i + rot) % n) * k) % n < k
}

impl Music {
    /// Music for a run's seed at sample rate `sr`.
    pub fn new(seed: u64, sr: f32) -> Music {
        let scale = Scale::from_seed(seed);
        let mut m = Music {
            sr,
            tempo: 124.0,
            scale,
            rng: Noise((seed as u32) | 1),
            clock: 0.0,
            step: 0,
            chord_root: 0,
            next_change: 16 * 8,
            pad: [PadVoice::default(); PAD_VOICES],
            pad_filter: [Svf::default(); 2],
            lfo: Osc::default(),
            bass_osc: Osc::default(),
            bass_sub: Osc::default(),
            bass_env: Env::default(),
            bass_filter_env: Env::default(),
            bass_svf: Svf::default(),
            bass_note: 0.0,
            kick_osc: Osc::default(),
            kick_env: Env::default(),
            kick_pitch: Env::default(),
            hat_env: Env::default(),
            hat_svf: Svf::default(),
            click_env: Env::default(),
            noise: Noise(0x9e37_79b9 ^ seed as u32),
            controls: Controls {
                intensity: 0.1,
                darkness: 0.0,
                playing: false,
            },
            intensity: Smooth { value: 0.1 },
            darkness: Smooth::default(),
            rebuild: 1.0,
            duck: Smooth { value: 1.0 },
            sweep: 0.0,
            bass_rot: (seed % 16) as u32,
            hat_rot: ((seed / 16) % 16) as u32,
        };
        let voicing = m.voicing(0, 0.3);
        for (v, n) in m.pad.iter_mut().zip(voicing) {
            v.note.value = n;
            v.target = n;
        }
        m
    }

    /// Set the sample rate (a new stream).
    pub fn set_sample_rate(&mut self, sr: f32) {
        self.sr = sr;
    }

    /// Steer the music.
    pub fn set(&mut self, c: Controls) {
        self.controls = c;
    }

    /// The ship was destroyed: drop to the drone, rebuild over 8 bars.
    pub fn death(&mut self) {
        self.rebuild = 0.0;
        self.sweep = -1.0;
    }

    /// A bomb: a filter sweep and a short duck.
    pub fn bomb(&mut self) {
        self.sweep = 1.0;
        self.duck.value = 0.35;
    }

    /// The chord on scale degree `root` as pad notes: root, fifth, seventh, ninth, and the
    /// eleventh when the intensity allows the extra dissonance.
    fn voicing(&self, root: i32, intensity: f32) -> [f32; PAD_VOICES] {
        let top = if intensity > 0.55 { 10 } else { 7 };
        let degrees = [root, root + 4, root + 6, root + 8, root + top];
        degrees.map(|d| self.scale.note(d) + 12.0)
    }

    /// Move each pad voice to the nearest tone of the new chord (any octave, within reach),
    /// so that the harmony changes by the smallest steps.
    fn lead_voices(&mut self, root: i32) {
        let chord = self.voicing(root, self.intensity.value);
        let classes: [i32; PAD_VOICES] = chord.map(|n| (n as i32).rem_euclid(12));
        let mut used = [false; 12];
        for v in &mut self.pad {
            let from = v.target;
            let mut best = (f32::MAX, from);
            for &pc in &classes {
                for octave in 2..7 {
                    let n = (octave * 12 + pc) as f32;
                    let d = (n - from).abs() + if used[pc as usize] { 3.0 } else { 0.0 };
                    if d < best.0 && n > 40.0 && n < 84.0 {
                        best = (d, n);
                    }
                }
            }
            used[(best.1 as i32).rem_euclid(12) as usize] = true;
            v.target = best.1;
        }
    }

    fn on_step(&mut self) {
        let step = self.step;
        let i16 = (step % 16) as u32;
        let bar = step / 16;
        let int = self.intensity.value * self.rebuild;
        if step.is_multiple_of(16) {
            // Rebuild after a death over 8 bars.
            self.rebuild = (self.rebuild + 1.0 / 8.0).min(1.0);
        }
        // Slow harmonic rhythm.
        if step >= self.next_change {
            let choices = [0, 5, 3, 6, 4, 2];
            let mut next = self.chord_root;
            while next == self.chord_root {
                next = choices[(self.rng.unit() * choices.len() as f32) as usize % choices.len()];
            }
            self.chord_root = next;
            self.lead_voices(next);
            let bars = if self.rng.unit() < 0.5 { 8 } else { 16 };
            self.next_change = step + 16 * bars;
        }
        // Bass: sparse, denser with intensity; the chord's root, sometimes its fifth.
        if self.controls.playing && int > 0.12 {
            let k = 2 + (int * 5.0) as u32;
            if euclid(i16, k, 16, self.bass_rot) {
                let fifth = self.rng.unit() < 0.2;
                let d = self.chord_root + if fifth { 4 } else { 0 };
                self.bass_note = self.scale.note(d - 7).max(26.0);
                self.bass_env
                    .start(0.003, 0.18 + 0.25 * (1.0 - int), self.sr);
                self.bass_filter_env.start(0.0, 0.12, self.sr);
            }
        }
        // Kick: from half-time to four on the floor.
        if self.controls.playing && int > 0.35 {
            let k = if int > 0.65 { 4 } else { 2 };
            if euclid(i16, k, 16, 0) {
                self.kick_env.start(0.001, 0.32, self.sr);
                self.kick_pitch.start(0.0, 0.04, self.sr);
            }
        }
        // Hats and clicks: Euclidean, denser with intensity.
        if self.controls.playing && int > 0.5 {
            let k = 3 + ((int - 0.5) * 16.0) as u32;
            if euclid(i16, k.min(11), 16, self.hat_rot) {
                self.hat_env
                    .start(0.001, 0.035 + 0.03 * self.rng.unit(), self.sr);
            }
            if i16 % 8 == 6 && self.rng.unit() < 0.4 {
                self.click_env.start(0.0, 0.012, self.sr);
            }
        }
        if bar > 0 && step.is_multiple_of(16 * 4) {
            // Every four bars, let the patterns drift a little.
            if self.rng.unit() < 0.3 {
                self.bass_rot = (self.bass_rot + 3) % 16;
            }
            if self.rng.unit() < 0.3 {
                self.hat_rot = (self.hat_rot + 5) % 16;
            }
        }
    }

    /// Render into `l` and `r` (overwriting).
    pub fn render(&mut self, l: &mut [f32], r: &mut [f32]) {
        let sr = self.sr;
        let inv = 1.0 / sr;
        let samples_per_step = sr * 60.0 / self.tempo / 4.0;
        let k_slow = 1.0 - (-1.0 / (0.8 * sr)).exp();
        let k_glide = 1.0 - (-1.0 / (1.2 * sr)).exp();
        let playing = self.controls.playing;
        let target_int = if playing {
            self.controls.intensity
        } else {
            0.08
        };
        for (ol, or) in l.iter_mut().zip(r.iter_mut()) {
            self.clock += 1.0;
            if self.clock >= samples_per_step {
                self.clock -= samples_per_step;
                self.step += 1;
                self.on_step();
            }
            let int = self.intensity.to(target_int, k_slow * 0.3);
            let dark = self.darkness.to(self.controls.darkness, k_slow);
            let duck = self.duck.to(1.0, k_slow * 2.0);
            self.sweep *= 1.0 - 1.5 * inv;
            // Pitch drifts down near a singularity: up to half a semitone.
            let bend = -0.5 * dark;
            let lfo = self.lfo.sine(0.05 * inv);
            // Pad.
            let cutoff = (220.0 + 2600.0 * int.powf(1.4) * self.rebuild.max(0.3))
                * (1.0 + 0.25 * lfo)
                * (1.0 - 0.55 * dark)
                * (1.0 + 1.5 * self.sweep).max(0.15);
            let mut pl = 0.0;
            let mut pr = 0.0;
            for (k, v) in self.pad.iter_mut().enumerate() {
                let n = v.note.to(v.target, k_glide);
                let on = if k == 4 && int < 0.5 { 0.0 } else { 1.0 };
                let g = v.gain.to(on, k_slow);
                if g < 1e-4 {
                    continue;
                }
                let f = mtof(n + bend);
                let a = v.osc[0].saw(f * inv);
                let b = v.osc[1].saw(f * 1.0035 * inv);
                let c = v.osc[2].saw(f * 0.9968 * inv);
                let spread = (k as f32 / (PAD_VOICES - 1) as f32) * 2.0 - 1.0;
                let (gl, gr) = super::dsp::pan(spread * 0.6);
                let s = (a + b * 0.8 + c * 0.8) * g;
                pl += s * gl;
                pr += s * gr;
                v.on = on;
            }
            let pl = self.pad_filter[0].tick(pl, cutoff, 0.9, sr).lp;
            let pr = self.pad_filter[1].tick(pr, cutoff * 1.03, 0.9, sr).lp;
            let pad_gain = db(-24.0);
            let mut mix_l = pl * pad_gain;
            let mut mix_r = pr * pad_gain;
            // Bass.
            let be = self.bass_env.tick();
            if be > 1e-5 {
                let f = mtof(self.bass_note + bend);
                let fe = self.bass_filter_env.tick();
                let saw = self.bass_osc.saw(f * inv);
                let sub = self.bass_sub.sine(f * 0.5 * inv);
                let fc = (90.0 + 900.0 * fe * (0.4 + int)) * (1.0 - 0.4 * dark);
                let x = self.bass_svf.tick(saw, fc, 1.3, sr).lp * 0.6 + sub * 0.8;
                let y = soft(x * 1.5) * be * db(-16.0);
                mix_l += y;
                mix_r += y;
            }
            // Kick.
            let ke = self.kick_env.tick();
            if ke > 1e-5 {
                let kp = self.kick_pitch.tick();
                let f = 46.0 * (1.0 + 3.0 * kp);
                let y = soft(self.kick_osc.sine(f * inv) * 1.4) * ke * db(-15.0);
                mix_l += y;
                mix_r += y;
            }
            // Hats and clicks.
            let he = self.hat_env.tick();
            let ce = self.click_env.tick();
            if he > 1e-5 || ce > 1e-5 {
                let n = self.noise.next();
                let h = self
                    .hat_svf
                    .tick(n, 7500.0 * (1.0 - 0.3 * dark), 1.2, sr)
                    .hp;
                let y = h * he * db(-30.0) + n * ce * db(-34.0);
                mix_l += y * 0.8;
                mix_r += y;
            }
            *ol = mix_l * duck;
            *or = mix_r * duck;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn euclidean_rhythms_have_k_hits() {
        for k in 0..=16 {
            let hits = (0..16).filter(|&i| euclid(i, k, 16, 3)).count();
            assert_eq!(hits, k as usize);
        }
    }

    #[test]
    fn voice_leading_moves_little() {
        let mut m = Music::new(3, 48000.0);
        let before: Vec<f32> = m.pad.iter().map(|v| v.target).collect();
        m.lead_voices(3);
        let after: Vec<f32> = m.pad.iter().map(|v| v.target).collect();
        let moved: f32 = before.iter().zip(&after).map(|(a, b)| (a - b).abs()).sum();
        assert!(moved <= 5.0 * 4.0, "{before:?} -> {after:?}");
    }

    #[test]
    fn renders_finite_and_bounded() {
        let mut m = Music::new(11, 48000.0);
        m.set(Controls {
            intensity: 1.0,
            darkness: 0.5,
            playing: true,
        });
        let (mut l, mut r) = (vec![0.0f32; 480], vec![0.0f32; 480]);
        let mut peak = 0.0f32;
        for _ in 0..1000 {
            m.render(&mut l, &mut r);
            peak = l.iter().chain(&r).fold(peak, |p, x| p.max(x.abs()));
            assert!(l.iter().all(|x| x.is_finite()));
        }
        assert!(peak > 0.01 && peak < 1.5, "peak {peak}");
    }
}
