//! Sound effects, synthesized: a preallocated pool of voices, each a small patch (a tone or
//! FM pair, filtered noise, pitch and filter envelopes). Every event family has a recipe and a
//! voice limit; a full family steals its oldest-quietest voice.

use super::dsp::{Env, Noise, Osc, Svf, db, mtof, pan, soft};

/// Sound families.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Family {
    /// A shot.
    Shot = 0,
    /// A hit that did not kill.
    Hit,
    /// A small kill.
    Kill,
    /// A singularity dies (or bursts).
    KillBig,
    /// The ship is destroyed.
    Death,
    /// A bomb.
    Bomb,
    /// A shard collected.
    Pickup,
    /// A spawn announced.
    Warn,
    /// A spawn landed.
    Spawn,
    /// A singularity eats.
    Absorb,
    /// An extra life or bomb.
    Extra,
    /// A shot hits the wall.
    Wall,
}

const FAMILIES: usize = 12;

impl Family {
    fn from_u8(x: u8) -> Option<Family> {
        use Family::*;
        [
            Shot, Hit, Kill, KillBig, Death, Bomb, Pickup, Warn, Spawn, Absorb, Extra, Wall,
        ]
        .get(x as usize)
        .copied()
    }

    /// Voices this family may hold at once.
    fn limit(self) -> usize {
        match self {
            Family::Shot => 4,
            Family::Hit => 6,
            Family::Kill => 10,
            Family::Pickup => 6,
            Family::Warn => 4,
            Family::Spawn => 4,
            Family::Wall => 3,
            _ => 3,
        }
    }
}

/// A trigger: what to play, where and how loud. Fits in a Firewheel `CustomBytes` event.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Trigger {
    /// The family.
    pub family: Family,
    /// The pitch, as a MIDI note (fractional allowed).
    pub note: f32,
    /// Stereo position in `[-1, 1]`.
    pub pan: f32,
    /// Gain (linear, on top of the family's level).
    pub gain: f32,
    /// A free parameter per family (size, brightness).
    pub variant: f32,
}

impl Trigger {
    /// Encode into the 36 bytes of a `CustomBytes` event.
    pub fn to_bytes(self) -> [u8; 36] {
        let mut b = [0u8; 36];
        b[0] = self.family as u8;
        b[4..8].copy_from_slice(&self.note.to_le_bytes());
        b[8..12].copy_from_slice(&self.pan.to_le_bytes());
        b[12..16].copy_from_slice(&self.gain.to_le_bytes());
        b[16..20].copy_from_slice(&self.variant.to_le_bytes());
        b
    }

    /// Decode.
    pub fn from_bytes(b: &[u8; 36]) -> Option<Trigger> {
        let f = |i: usize| f32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]]);
        Some(Trigger {
            family: Family::from_u8(b[0])?,
            note: f(4),
            pan: f(8),
            gain: f(12),
            variant: f(16),
        })
    }
}

/// The waveform of a voice's tone.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
enum Wave {
    #[default]
    Sine,
    Tri,
    Saw,
    Fm,
}

/// A voice's patch, filled in by the recipes.
#[derive(Clone, Copy, Debug, Default)]
struct Patch {
    wave: Wave,
    /// Base frequency (Hz).
    f0: f32,
    /// Pitch envelope depth (semitones at the start, decaying to 0).
    sweep: f32,
    sweep_time: f32,
    attack: f32,
    decay: f32,
    tone: f32,
    noise: f32,
    /// Noise filter: base cutoff (Hz), envelope depth (octaves), time, resonance, band-pass
    /// (else low-pass).
    cutoff: f32,
    cut_depth: f32,
    cut_time: f32,
    q: f32,
    band: bool,
    fm_ratio: f32,
    fm_index: f32,
    /// A second partial (ratio to f0) and its level.
    partial: f32,
    partial_gain: f32,
    gain: f32,
}

#[derive(Clone, Copy, Debug, Default)]
struct Voice {
    active: bool,
    family: u8,
    age: u32,
    patch: Patch,
    osc: Osc,
    mod_osc: Osc,
    part: Osc,
    amp: Env,
    pitch: Env,
    filt: Env,
    svf: Svf,
    tone_svf: Svf,
    pan: (f32, f32),
}

/// The sound-effect engine.
pub struct Sfx {
    sr: f32,
    voices: [Voice; 48],
    noise: Noise,
}

impl Sfx {
    /// An engine at sample rate `sr`.
    pub fn new(sr: f32) -> Sfx {
        Sfx {
            sr,
            voices: [Voice::default(); 48],
            noise: Noise(0x1234_5678),
        }
    }

    /// Set the sample rate (a new stream).
    pub fn set_sample_rate(&mut self, sr: f32) {
        self.sr = sr;
    }

    fn recipe(t: &Trigger) -> Patch {
        let f0 = mtof(t.note);
        let base = Patch {
            f0,
            q: 0.8,
            ..Patch::default()
        };
        match t.family {
            Family::Shot => Patch {
                wave: Wave::Tri,
                sweep: 12.0,
                sweep_time: 0.025,
                attack: 0.001,
                decay: 0.075,
                tone: 1.0,
                noise: 0.2,
                cutoff: 5000.0,
                cut_depth: 1.0,
                cut_time: 0.02,
                band: true,
                gain: db(-27.0),
                ..base
            },
            Family::Hit => Patch {
                wave: Wave::Fm,
                fm_ratio: 3.51,
                fm_index: 2.5,
                sweep: 5.0,
                sweep_time: 0.02,
                attack: 0.001,
                decay: 0.12,
                tone: 0.9,
                noise: 0.25,
                cutoff: 3000.0,
                cut_depth: 1.5,
                cut_time: 0.04,
                band: true,
                gain: db(-22.0),
                ..base
            },
            Family::Kill => Patch {
                wave: Wave::Saw,
                sweep: -5.0,
                sweep_time: 0.15,
                attack: 0.002,
                decay: 0.35 + 0.2 * t.variant,
                tone: 0.55,
                noise: 0.9,
                cutoff: 450.0,
                cut_depth: 3.2,
                cut_time: 0.12,
                q: 1.6,
                band: true,
                partial: 1.5,
                partial_gain: 0.3,
                gain: db(-15.0),
                ..base
            },
            Family::KillBig => Patch {
                wave: Wave::Sine,
                f0: 42.0,
                sweep: 26.0,
                sweep_time: 0.25,
                attack: 0.004,
                decay: 1.4,
                tone: 1.0,
                noise: 0.8,
                cutoff: 200.0,
                cut_depth: 4.5,
                cut_time: 0.5,
                q: 0.9,
                partial: 7.13,
                partial_gain: 0.12,
                gain: db(-8.0),
                ..base
            },
            Family::Death => Patch {
                wave: Wave::Fm,
                fm_ratio: 1.41,
                fm_index: 4.0,
                sweep: 24.0,
                sweep_time: 1.1,
                attack: 0.003,
                decay: 2.2,
                tone: 0.7,
                noise: 1.0,
                cutoff: 150.0,
                cut_depth: 5.0,
                cut_time: 0.9,
                q: 0.9,
                gain: db(-6.0),
                ..base
            },
            Family::Bomb => Patch {
                wave: Wave::Sine,
                f0: 38.0,
                sweep: 30.0,
                sweep_time: 0.35,
                attack: 0.01,
                decay: 2.0,
                tone: 1.0,
                noise: 1.0,
                cutoff: 120.0,
                cut_depth: 6.0,
                cut_time: 1.0,
                q: 0.7,
                gain: db(-6.0),
                ..base
            },
            Family::Pickup => Patch {
                wave: Wave::Sine,
                sweep: 0.3,
                sweep_time: 0.01,
                attack: 0.002,
                decay: 0.22,
                tone: 1.0,
                partial: 2.0,
                partial_gain: 0.35,
                gain: db(-21.0),
                ..base
            },
            Family::Warn => Patch {
                wave: Wave::Sine,
                sweep: -12.0,
                sweep_time: 0.6,
                attack: 0.35,
                decay: 0.25,
                tone: 0.3,
                noise: 0.7,
                cutoff: 3500.0,
                cut_depth: -2.5,
                cut_time: 0.5,
                q: 2.5,
                band: true,
                gain: db(-30.0),
                ..base
            },
            Family::Spawn => Patch {
                wave: Wave::Tri,
                sweep: 3.0,
                sweep_time: 0.03,
                attack: 0.002,
                decay: 0.18,
                tone: 1.0,
                gain: db(-26.0),
                ..base
            },
            Family::Absorb => Patch {
                wave: Wave::Sine,
                sweep: -14.0,
                sweep_time: 0.3,
                attack: 0.01,
                decay: 0.3,
                tone: 1.0,
                noise: 0.2,
                cutoff: 400.0,
                cut_depth: 1.0,
                cut_time: 0.1,
                gain: db(-17.0),
                ..base
            },
            Family::Extra => Patch {
                wave: Wave::Sine,
                attack: 0.005,
                decay: 1.1,
                tone: 1.0,
                partial: 1.5,
                partial_gain: 0.6,
                gain: db(-15.0),
                ..base
            },
            Family::Wall => Patch {
                wave: Wave::Tri,
                sweep: 7.0,
                sweep_time: 0.01,
                attack: 0.001,
                decay: 0.04,
                tone: 0.5,
                noise: 0.5,
                cutoff: 6000.0,
                cut_depth: 0.0,
                cut_time: 0.01,
                band: true,
                gain: db(-33.0),
                ..base
            },
        }
    }

    /// Start a voice.
    pub fn trigger(&mut self, t: &Trigger) {
        let fam = t.family as u8;
        let limit = t.family.limit();
        let in_family = self
            .voices
            .iter()
            .filter(|v| v.active && v.family == fam)
            .count();
        // Oldest-quietest: the lowest level, older first among equals.
        let score = |v: &Voice| v.amp.level - v.age as f32 * 1e-7;
        let slot = if in_family >= limit {
            self.voices
                .iter()
                .enumerate()
                .filter(|(_, v)| v.active && v.family == fam)
                .min_by(|a, b| score(a.1).total_cmp(&score(b.1)))
                .map(|(i, _)| i)
        } else {
            self.voices.iter().position(|v| !v.active).or_else(|| {
                self.voices
                    .iter()
                    .enumerate()
                    .min_by(|a, b| score(a.1).total_cmp(&score(b.1)))
                    .map(|(i, _)| i)
            })
        };
        let Some(i) = slot else { return };
        let patch = Sfx::recipe(t);
        let sr = self.sr;
        let v = &mut self.voices[i];
        *v = Voice {
            active: true,
            family: fam,
            age: 0,
            patch: Patch {
                gain: patch.gain * t.gain,
                ..patch
            },
            osc: Osc { phase: 0.0 },
            mod_osc: Osc::default(),
            part: Osc::default(),
            amp: Env::default(),
            pitch: Env::default(),
            filt: Env::default(),
            svf: Svf::default(),
            tone_svf: Svf::default(),
            pan: pan(t.pan),
        };
        v.amp.start(patch.attack, patch.decay, sr);
        v.pitch.start(0.0, patch.sweep_time.max(1e-3), sr);
        v.filt.start(0.0, patch.cut_time.max(1e-3), sr);
    }

    /// Voices sounding (for tests and the debug overlay).
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn active(&self) -> usize {
        self.voices.iter().filter(|v| v.active).count()
    }

    /// Voices of a family sounding.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn active_in(&self, f: Family) -> usize {
        self.voices
            .iter()
            .filter(|v| v.active && v.family == f as u8)
            .count()
    }

    /// Add the voices into `l` and `r`. Returns whether anything sounded.
    pub fn render(&mut self, l: &mut [f32], r: &mut [f32]) -> bool {
        let sr = self.sr;
        let inv = 1.0 / sr;
        let mut any = false;
        for v in &mut self.voices {
            if !v.active {
                continue;
            }
            any = true;
            let p = v.patch;
            for (ol, or) in l.iter_mut().zip(r.iter_mut()) {
                let a = v.amp.tick();
                let pe = v.pitch.tick();
                let fe = v.filt.tick();
                let f = p.f0 * (p.sweep * pe / 12.0).exp2();
                let inc = f * inv;
                let tone = match p.wave {
                    Wave::Sine => v.osc.sine(inc),
                    Wave::Tri => v.osc.tri(inc),
                    Wave::Saw => {
                        let s = v.osc.saw(inc);
                        v.tone_svf
                            .tick(s, (f * 6.0 * (0.3 + a)).min(12000.0), 0.7, sr)
                            .lp
                    }
                    Wave::Fm => {
                        let m = v.mod_osc.sine(inc * p.fm_ratio);
                        v.osc.phase = (v.osc.phase + m * p.fm_index * a * inc).rem_euclid(1.0);
                        v.osc.sine(inc)
                    }
                };
                let part = if p.partial_gain > 0.0 {
                    v.part.sine(inc * p.partial) * p.partial_gain
                } else {
                    0.0
                };
                let mut x = (tone + part) * p.tone;
                if p.noise > 0.0 {
                    let n = self.noise.next();
                    let fc = (p.cutoff * (p.cut_depth * fe).exp2()).clamp(30.0, 16000.0);
                    let o = v.svf.tick(n, fc, p.q, sr);
                    x += p.noise * if p.band { o.bp * 1.5 } else { o.lp };
                }
                let y = soft(x) * a * p.gain;
                *ol += y * v.pan.0;
                *or += y * v.pan.1;
            }
            v.age = v.age.saturating_add(l.len() as u32);
            if v.amp.done() {
                v.active = false;
            }
        }
        any
    }
}

/// The run's pitch material, from its seed: a mode and a root, shared by the music and the
/// effects so that the effects play with the music.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Scale {
    /// Semitone offsets of the seven degrees.
    pub steps: [i32; 7],
    /// The root as a MIDI note (in the second octave).
    pub root: i32,
    /// The mode's name.
    pub name: &'static str,
}

impl Scale {
    /// The mode for a seed: Dorian, Aeolian, Phrygian or Lydian, on a root from D to A.
    pub fn from_seed(seed: u64) -> Scale {
        let modes: [(&str, [i32; 7]); 4] = [
            ("dorian", [0, 2, 3, 5, 7, 9, 10]),
            ("aeolian", [0, 2, 3, 5, 7, 8, 10]),
            ("phrygian", [0, 1, 3, 5, 7, 8, 10]),
            ("lydian", [0, 2, 4, 6, 7, 9, 11]),
        ];
        let (name, steps) = modes[(seed % 4) as usize];
        let roots = [38, 40, 41, 43, 45];
        Scale {
            steps,
            root: roots[((seed / 4) % 5) as usize],
            name,
        }
    }

    /// The MIDI note of scale degree `d` (any integer; 7 is an octave up).
    pub fn note(&self, d: i32) -> f32 {
        let o = d.div_euclid(7);
        let i = d.rem_euclid(7) as usize;
        (self.root + 12 * o + self.steps[i]) as f32
    }
}

const _: () = assert!(FAMILIES == 12);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn triggers_round_trip_through_bytes() {
        let t = Trigger {
            family: Family::Pickup,
            note: 79.5,
            pan: -0.3,
            gain: 0.8,
            variant: 2.0,
        };
        assert_eq!(Trigger::from_bytes(&t.to_bytes()), Some(t));
    }

    #[test]
    fn families_keep_their_voice_limits_and_steal() {
        let mut s = Sfx::new(48000.0);
        for k in 0..20 {
            s.trigger(&Trigger {
                family: Family::Shot,
                note: 80.0 + k as f32,
                pan: 0.0,
                gain: 1.0,
                variant: 0.0,
            });
        }
        assert_eq!(s.active_in(Family::Shot), Family::Shot.limit());
        // Others still get voices.
        s.trigger(&Trigger {
            family: Family::Bomb,
            note: 40.0,
            pan: 0.0,
            gain: 1.0,
            variant: 0.0,
        });
        assert_eq!(s.active_in(Family::Bomb), 1);
        // Everything dies away.
        let (mut l, mut r) = (vec![0.0f32; 512], vec![0.0f32; 512]);
        for _ in 0..48000 * 4 / 512 {
            l.fill(0.0);
            r.fill(0.0);
            s.render(&mut l, &mut r);
            assert!(l.iter().chain(&r).all(|x| x.is_finite() && x.abs() < 4.0));
        }
        assert_eq!(s.active(), 0);
    }

    #[test]
    fn scales_are_modal() {
        let s = Scale::from_seed(0);
        assert_eq!(s.name, "dorian");
        assert_eq!(s.note(7) - s.note(0), 12.0);
    }
}
