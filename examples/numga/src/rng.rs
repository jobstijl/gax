//! A small seeded random generator (xorshift64) with uniform and Gaussian draws: numpy's streams
//! cannot be reproduced, so the examples use this one and keep their checks robust to the draw.
//!
//! The Gaussian draw (Box and Muller) and the normalization of a random direction are sampling
//! formulas, not geometry: the one place in the shared code with float square roots and
//! trigonometry.
#![allow(clippy::disallowed_methods)]

/// The generator's state: `Rng(seed)` starts from the seed as given, `Rng::new` scrambles it.
#[derive(Clone, Debug)]
pub struct Rng(pub u64);

impl Rng {
    /// A generator from a scrambled seed (any value).
    pub fn new(seed: u64) -> Rng {
        Rng(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1)
    }

    fn bits(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// Uniform in `(0, 1)`: the midpoints of `2^53` bins, so neither end (`ln` and division
    /// are safe).
    pub fn uniform(&mut self) -> f64 {
        ((self.bits() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    /// Uniform in `(lo, hi)`.
    pub fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.uniform()
    }

    /// A standard normal draw (Box and Muller).
    pub fn normal(&mut self) -> f64 {
        let (u, v) = (self.uniform(), self.uniform());
        (-2.0 * u.ln()).sqrt() * (core::f64::consts::TAU * v).cos()
    }

    /// A direction uniform on the unit sphere in `n` dimensions.
    pub fn direction<const N: usize>(&mut self) -> [f64; N] {
        let v: [f64; N] = core::array::from_fn(|_| self.normal());
        let n = v.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-300);
        v.map(|x| x / n)
    }
}
