//! DSP building blocks, independent of the audio engine: band-limited oscillators, envelopes,
//! a state-variable filter, noise and a small reverb. Nothing here allocates while processing.
//!
//! Oscillation is rotation: a sine is the height of a phasor, a unit direction that a rotation
//! motor turns a little every sample (phase modulation turns it further). A resonance is a
//! damped rotation: the phasor turned and shrunk every sample, with the input added
//! (`Resonator`). The state-variable filter's undamped core is a rotation too, in its
//! trapezoidal (Cayley) form, and its coefficient is read off the rotor of its turn. An
//! equal-power pan is the rotor that turns "left" towards the source: its two coefficients,
//! the cosine and sine of half the angle, are the two gains.

use core::f32::consts::TAU;
use gax::Unit;
use gax::pga2d::{Motor, Point};

use crate::geom::{ORIGIN, phasor};

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

/// A phase accumulator with band-limited waveforms (PolyBLEP), and a phasor for the sine.
#[derive(Clone, Copy, Debug)]
pub struct Osc {
    /// Phase in `[0, 1)` (saw, pulse, triangle).
    pub phase: f32,
    /// The sine's phasor: a unit direction going round the origin.
    turning: Point<(), f32>,
    /// The turn per sample, for the increment it was made for.
    step: Unit<Motor<(), f32>>,
    step_inc: f32,
    samples: u32,
}

impl Default for Osc {
    fn default() -> Osc {
        Osc {
            phase: 0.0,
            turning: Point::direction(1.0, 0.0),
            step: Motor::rotation(ORIGIN, 0.0),
            step_inc: 0.0,
            samples: 0,
        }
    }
}

impl Osc {
    #[inline]
    fn advance(&mut self, inc: f32) {
        self.phase += inc;
        if self.phase >= 1.0 {
            self.phase -= 1.0;
        }
    }

    /// Sine, at `inc` cycles per sample: the phasor's height, then one turn of `inc` cycles
    /// (the motor is kept while the frequency holds).
    #[inline]
    pub fn sine(&mut self, inc: f32) -> f32 {
        let y = self.turning.e01();
        if inc != self.step_inc {
            self.step = Motor::rotation(ORIGIN, TAU * inc);
            self.step_inc = inc;
        }
        self.turning = self.step >> self.turning;
        // Rounding slowly changes its length: back to the unit circle now and then.
        self.samples = self.samples.wrapping_add(1);
        if self.samples.is_multiple_of(1024) {
            self.turning = self.turning * (1.0 / self.turning.ideal_norm());
        }
        self.advance(inc);
        y
    }

    /// Phase modulation: turn the phasor further by `turns` of a cycle.
    #[inline]
    pub fn nudge(&mut self, turns: f32) {
        self.turning = Motor::rotation(ORIGIN, TAU * turns) >> self.turning;
        self.phase = (self.phase + turns).rem_euclid(1.0);
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
///
/// Undamped (`k = 0`), its state turns: trapezoidal integration of a harmonic oscillator is the
/// Cayley transform `(1 + gJ)/(1 - gJ)` of its generator, a rotation by `2 atan g` per sample.
/// Prewarping makes that exactly the cutoff's turn `θ = 2π fc / sr`, so `g = tan(θ/2)`: the
/// ratio of the bivector and scalar parts of the rotor that turns by `θ`, which is where the
/// filter takes it from. The damping `k` shrinks one axis of the turn.
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
        // The prewarped gain: the rotor of the turn per sample, `cos(θ/2) - sin(θ/2) e12`, its
        // bivector part over its scalar part.
        let rotor = Motor::rotation(ORIGIN, TAU * (fc / sr).clamp(1e-5, 0.49)).into_inner();
        let g = -rotor.e12() / rotor.s();
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

/// A resonance as a damped rotation (Mathews and Smith's phasor filter): every sample the state,
/// a direction, is turned by the centre frequency's rotation and shrunk by `r`, and the input
/// is added along `x`. It rings at the turn's frequency and dies away at the shrink's rate, so
/// the turn is the pitch and the shrink the bandwidth, `r = exp(-π bw / sr)`. The output is
/// the state's height, scaled to a peak gain of about one.
#[derive(Clone, Copy, Debug)]
pub struct Resonator {
    state: Point<(), f32>,
    step: Unit<Motor<(), f32>>,
    step_fc: f32,
}

impl Default for Resonator {
    fn default() -> Resonator {
        Resonator {
            state: Point::direction(0.0, 0.0),
            step: Motor::rotation(ORIGIN, 0.0),
            step_fc: 0.0,
        }
    }
}

impl Resonator {
    /// Ring `x` at centre `fc` (Hz) with quality `q` (the centre over the bandwidth).
    #[inline]
    pub fn tick(&mut self, x: f32, fc: f32, q: f32, sr: f32) -> f32 {
        if fc != self.step_fc {
            self.step = Motor::rotation(ORIGIN, TAU * (fc / sr).clamp(1e-5, 0.49));
            self.step_fc = fc;
        }
        let r = (-core::f32::consts::PI * fc / (q.max(0.1) * sr)).exp();
        // A real input is half on the turning side: `2 (1 - r)` gives the peak a gain of one.
        self.state = (self.step >> self.state) * r + Point::direction(x * 2.0 * (1.0 - r), 0.0);
        self.state.e01()
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

/// Equal-power pan gains for `pan` in `[-1, 1]`, hard left to hard right: the source's
/// direction at that fraction of the half turn from left (through ahead) to right, as `toward`
/// hears it.
#[inline]
pub fn pan(pan: f32) -> (f32, f32) {
    toward(phasor(
        (1.0 - pan.clamp(-1.0, 1.0)) * core::f32::consts::FRAC_PI_2,
    ))
}

/// Equal-power gains for a source in direction `d` from the listener (`x` to the right, `y`
/// ahead): the rotor that turns "left" towards `d` is `cos(β/2) + sin(β/2) e12`, and its two
/// coefficients are the gains. Left is `(1, 0)`, ahead the even split, right `(0, 1)`; their
/// squares always sum to one. A source behind is heard as its mirror in front.
#[inline]
pub fn toward(d: Point<(), f32>) -> (f32, f32) {
    let ahead = Point::direction(d.e20(), d.e01().abs());
    let rotor = Motor::rotation_between(Point::direction(-1.0, 0.0), ahead).into_inner();
    (rotor.s().abs(), rotor.e12().abs())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The phasor's height is the sine, and stays on it over a minute of samples (the turns
    /// are exact rotations; rounding is renormalized away).
    #[test]
    #[allow(clippy::disallowed_methods)] // the reference it is checked against
    fn a_turning_phasor_is_a_sine() {
        let mut o = Osc::default();
        let inc = 440.0 / 48000.0;
        let mut worst = 0.0f32;
        for n in 0..48000 * 60 {
            let y = o.sine(inc);
            let t = (f64::from(n) * f64::from(inc)).fract() * core::f64::consts::TAU;
            worst = worst.max((y - t.sin() as f32).abs());
        }
        assert!(worst < 2e-3, "{worst}");
        // Phase modulation by a quarter cycle turns sine into cosine.
        let mut o = Osc::default();
        o.nudge(0.25);
        assert!((o.sine(0.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn panning_keeps_the_power() {
        for k in 0..=20 {
            let (l, r) = pan(k as f32 / 10.0 - 1.0);
            assert!((l * l + r * r - 1.0).abs() < 1e-5);
        }
        let (l, r) = pan(-1.0);
        assert!((l - 1.0).abs() < 1e-6 && r.abs() < 1e-6, "{l} {r}");
        let (l, r) = pan(1.0);
        assert!(l.abs() < 1e-6 && (r - 1.0).abs() < 1e-6, "{l} {r}");
        let (l, r) = pan(0.0);
        assert!((l - r).abs() < 1e-6);
        // By direction: right, ahead-right, and behind-right heard as ahead-right.
        let (l, r) = toward(Point::direction(5.0, 0.0));
        assert!(l.abs() < 1e-6 && (r - 1.0).abs() < 1e-6);
        let (a, b) = toward(Point::direction(1.0, 1.0));
        let (c, d) = toward(Point::direction(1.0, -1.0));
        assert!(b > a && (a - c).abs() < 1e-6 && (b - d).abs() < 1e-6);
    }

    /// The undamped filter's state turns by the cutoff's angle per sample: an impulse into it
    /// rings at the cutoff.
    #[test]
    fn the_filter_core_is_a_rotation_at_the_cutoff() {
        let sr = 48000.0;
        let fc = 1000.0;
        let mut f = Svf::default();
        // `q` large: nearly undamped. Count the zero crossings of the ringing.
        let mut last = f.tick(1.0, fc, 1e4, sr).bp;
        let mut crossings = 0;
        for _ in 0..48000 {
            let y = f.tick(0.0, fc, 1e4, sr).bp;
            crossings += usize::from((y > 0.0) != (last > 0.0));
            last = y;
        }
        assert!((crossings as f32 / 2.0 - fc).abs() < 2.0, "{crossings}");
    }

    /// A damped rotation rings at its turn's frequency: its response peaks there, with about
    /// unit gain, and falls away off the centre.
    #[test]
    fn a_damped_rotation_resonates() {
        let sr = 48000.0;
        let gain = |f: f32| {
            let mut r = Resonator::default();
            let mut o = Osc::default();
            let mut peak = 0.0f32;
            for n in 0..48000 {
                let y = r.tick(o.sine(f / sr), 2000.0, 8.0, sr);
                if n > 24000 {
                    peak = peak.max(y.abs());
                }
            }
            peak
        };
        let centre = gain(2000.0);
        assert!((0.5..2.0).contains(&centre), "{centre}");
        assert!(gain(1000.0) < 0.3 * centre && gain(4000.0) < 0.3 * centre);
    }
}
