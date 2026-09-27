//! DSP building blocks, independent of the audio engine: band-limited oscillators, envelopes,
//! a state-variable filter, noise and a small reverb. Nothing here allocates while processing.

use core::f32::consts::{PI, TAU};

/// MIDI note number to frequency.
pub fn mtof(m: f32) -> f32 {
    440.0 * (2.0f32).powf((m - 69.0) / 12.0)
}

/// Decibels to linear gain.
pub fn db(x: f32) -> f32 {
    (10.0f32).powf(x / 20.0)
}

/// A cheap soft clipper.
pub fn soft(x: f32) -> f32 {
    let x = x.clamp(-3.0, 3.0);
    x * (27.0 + x * x) / (27.0 + 9.0 * x * x)
}

/// White noise (xorshift).
#[derive(Clone, Copy, Debug)]
pub struct Noise(pub u32);

impl Noise {
    /// Uniform in `[-1, 1)`.
    #[inline]
    pub fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        (self.0 >> 8) as f32 / (1u32 << 23) as f32 - 1.0
    }

    /// Uniform in `[0, 1)`.
    #[inline]
    pub fn unit(&mut self) -> f32 {
        0.5 * (self.next() + 1.0)
    }
}

#[inline]
fn poly_blep(t: f32, dt: f32) -> f32 {
    if t < dt {
        let t = t / dt;
        t + t - t * t - 1.0
    } else if t > 1.0 - dt {
        let t = (t - 1.0) / dt;
        t * t + t + t + 1.0
    } else {
        0.0
    }
}

/// A phase accumulator with band-limited waveforms (PolyBLEP).
#[derive(Clone, Copy, Debug, Default)]
pub struct Osc {
    /// Phase in `[0, 1)`.
    pub phase: f32,
}

impl Osc {
    #[inline]
    fn advance(&mut self, inc: f32) {
        self.phase += inc;
        if self.phase >= 1.0 {
            self.phase -= 1.0;
        }
    }

    /// Sine, at `inc` cycles per sample.
    #[inline]
    pub fn sine(&mut self, inc: f32) -> f32 {
        let y = (self.phase * TAU).sin();
        self.advance(inc);
        y
    }

    /// Band-limited saw.
    #[inline]
    pub fn saw(&mut self, inc: f32) -> f32 {
        let y = 2.0 * self.phase - 1.0 - poly_blep(self.phase, inc);
        self.advance(inc);
        y
    }

    /// Band-limited pulse of `width` in `(0, 1)`.
    #[inline]
    #[allow(dead_code)] // for the fragment and texture layers (M3)
    pub fn pulse(&mut self, inc: f32, width: f32) -> f32 {
        let mut y = if self.phase < width { 1.0 } else { -1.0 };
        y += poly_blep(self.phase, inc);
        let t2 = (self.phase + 1.0 - width) % 1.0;
        y -= poly_blep(t2, inc);
        self.advance(inc);
        y
    }

    /// Triangle (integrated square, cheap and soft).
    #[inline]
    pub fn tri(&mut self, inc: f32) -> f32 {
        let y = 1.0 - 4.0 * (self.phase - 0.5).abs();
        self.advance(inc);
        y
    }
}

/// A state-variable filter (Simper's trapezoidal form): stable under fast modulation.
#[derive(Clone, Copy, Debug, Default)]
pub struct Svf {
    ic1: f32,
    ic2: f32,
}

/// Outputs of the filter.
#[derive(Clone, Copy, Debug)]
pub struct SvfOut {
    /// Low-pass.
    pub lp: f32,
    /// Band-pass.
    pub bp: f32,
    /// High-pass.
    pub hp: f32,
}

impl Svf {
    /// Filter one sample at cutoff `fc` (Hz) and resonance `q` (0.5 soft .. 10 sharp).
    #[inline]
    pub fn tick(&mut self, x: f32, fc: f32, q: f32, sr: f32) -> SvfOut {
        let g = (PI * (fc / sr).clamp(1e-5, 0.49)).tan();
        let k = 1.0 / q.max(0.05);
        let a1 = 1.0 / (1.0 + g * (g + k));
        let a2 = g * a1;
        let a3 = g * a2;
        let v3 = x - self.ic2;
        let v1 = a1 * self.ic1 + a2 * v3;
        let v2 = self.ic2 + a2 * self.ic1 + a3 * v3;
        self.ic1 = 2.0 * v1 - self.ic1;
        self.ic2 = 2.0 * v2 - self.ic2;
        SvfOut {
            lp: v2,
            bp: v1,
            hp: x - k * v1 - v2,
        }
    }

    /// Clear the state.
    #[allow(dead_code)] // for the fragment and texture layers (M3)
    pub fn reset(&mut self) {
        *self = Svf::default();
    }
}

/// An attack-decay envelope with exponential decay.
#[derive(Clone, Copy, Debug, Default)]
pub struct Env {
    /// The current level.
    pub level: f32,
    attack_inc: f32,
    decay_mul: f32,
    attacking: bool,
}

impl Env {
    /// Start: rise to 1 over `attack` seconds, then fall by 60 dB over `decay` seconds.
    pub fn start(&mut self, attack: f32, decay: f32, sr: f32) {
        self.attack_inc = 1.0 / (attack * sr).max(1.0);
        self.decay_mul = (-6.9 / (decay * sr).max(1.0)).exp();
        self.attacking = true;
        if attack <= 0.0 {
            self.level = 1.0;
            self.attacking = false;
        }
    }

    /// The next level.
    #[inline]
    pub fn tick(&mut self) -> f32 {
        if self.attacking {
            self.level += self.attack_inc;
            if self.level >= 1.0 {
                self.level = 1.0;
                self.attacking = false;
            }
        } else {
            self.level *= self.decay_mul;
        }
        self.level
    }

    /// Whether it has died away.
    pub fn done(&self) -> bool {
        !self.attacking && self.level < 1e-4
    }
}

/// A one-pole smoother for parameters.
#[derive(Clone, Copy, Debug, Default)]
pub struct Smooth {
    /// Current value.
    pub value: f32,
}

impl Smooth {
    /// Move towards `target` with time constant `k` per sample (0..1).
    #[inline]
    pub fn to(&mut self, target: f32, k: f32) -> f32 {
        self.value += (target - self.value) * k;
        self.value
    }
}

/// A comb filter with damping (Freeverb).
struct Comb<const N: usize> {
    buf: [f32; N],
    i: usize,
    len: usize,
    store: f32,
}

impl<const N: usize> Comb<N> {
    fn new(len: usize) -> Self {
        Comb {
            buf: [0.0; N],
            i: 0,
            len: len.min(N),
            store: 0.0,
        }
    }
    #[inline]
    fn tick(&mut self, x: f32, feedback: f32, damp: f32) -> f32 {
        let y = self.buf[self.i];
        self.store = y * (1.0 - damp) + self.store * damp;
        self.buf[self.i] = x + self.store * feedback;
        self.i = (self.i + 1) % self.len;
        y
    }
}

struct Allpass<const N: usize> {
    buf: [f32; N],
    i: usize,
    len: usize,
}

impl<const N: usize> Allpass<N> {
    fn new(len: usize) -> Self {
        Allpass {
            buf: [0.0; N],
            i: 0,
            len: len.min(N),
        }
    }
    #[inline]
    fn tick(&mut self, x: f32) -> f32 {
        let b = self.buf[self.i];
        let y = -x + b;
        self.buf[self.i] = x + b * 0.5;
        self.i = (self.i + 1) % self.len;
        y
    }
}

/// A Freeverb-style stereo reverb (used by the offline renders; the game routes to Firewheel's
/// `FreeverbNode`, the same design).
pub struct Reverb {
    combs: [[Comb<2048>; 8]; 2],
    alls: [[Allpass<1024>; 4]; 2],
    /// Room size (0..1).
    pub room: f32,
    /// Damping (0..1).
    pub damp: f32,
}

impl Reverb {
    /// A reverb for sample rate `sr` (boxed: the buffers are large).
    pub fn new(sr: f32) -> Box<Reverb> {
        let k = sr / 44100.0;
        let c = [1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617];
        let a = [556, 441, 341, 225];
        let comb = |n: usize, s: usize| Comb::new(((n + s) as f32 * k) as usize);
        let all = |n: usize, s: usize| Allpass::new(((n + s) as f32 * k) as usize);
        Box::new(Reverb {
            combs: [0, 23].map(|s| c.map(|n| comb(n, s))),
            alls: [0, 23].map(|s| a.map(|n| all(n, s))),
            room: 0.85,
            damp: 0.4,
        })
    }

    /// One stereo sample in, the wet signal out.
    #[inline]
    pub fn tick(&mut self, l: f32, r: f32) -> (f32, f32) {
        let input = (l + r) * 0.015;
        let fb = 0.7 + 0.28 * self.room;
        let mut out = [0.0f32; 2];
        for ((o, combs), alls) in out.iter_mut().zip(&mut self.combs).zip(&mut self.alls) {
            let mut s = 0.0;
            for c in combs {
                s += c.tick(input, fb, self.damp);
            }
            for a in alls {
                s = a.tick(s);
            }
            *o = s;
        }
        (out[0], out[1])
    }
}

/// Equal-power pan gains for `pan` in `[-1, 1]`.
#[inline]
pub fn pan(pan: f32) -> (f32, f32) {
    let a = (pan.clamp(-1.0, 1.0) + 1.0) * core::f32::consts::FRAC_PI_4;
    (a.cos(), a.sin())
}
