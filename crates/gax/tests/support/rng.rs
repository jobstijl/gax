//! The tests' random numbers: a small xorshift generator, so failures reproduce. `Rng(seed)`
//! starts from the seed as given (the streams the tests were written with); `Rng::new` scrambles
//! it first.
#![allow(dead_code)]

use gax::{Coef, Extensor, Kind, Slots};

/// A xorshift64 generator.
pub struct Rng(pub u64);

impl Rng {
    /// A generator from a scrambled seed (any value, zero included).
    pub fn new(seed: u64) -> Rng {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    /// The next 64 random bits.
    pub fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// Uniform in `[-1, 1)`, from 53 bits.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }

    /// Uniform in `[-1, 1)`, from 24 bits (exact in `f32`).
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }

    /// A value or map with every coefficient uniform in `[-scale, scale)`, drawn in
    /// coefficient order.
    pub fn value<M: Extensor>(&mut self, scale: f64) -> M {
        M::from_coeffs(<M::Kind as Kind>::arr_from_fn(|_| {
            <M::Slots as Slots>::from_flat(&mut |_| M::Coef::from_f64(scale * self.next_f64()), 0)
        }))
    }

    /// `n` of [`Rng::value`].
    pub fn values<M: Extensor>(&mut self, n: usize, scale: f64) -> Vec<M> {
        (0..n).map(|_| self.value(scale)).collect()
    }
}
