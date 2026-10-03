//! The tests' random numbers, from `rand`, seeded so that failures reproduce.
#![allow(dead_code)]

use gax::{Coef, Extensor, Kind, Slots};
use rand::SeedableRng;

/// The generator.
pub type Rng = rand::rngs::StdRng;

/// A generator from a seed.
pub fn rng(seed: u64) -> Rng {
    Rng::seed_from_u64(seed)
}

/// Draws the tests need, for any generator.
pub trait Draw: rand::Rng {
    /// Uniform in `[-1, 1)`.
    fn next_f64(&mut self) -> f64 {
        self.random_range(-1.0..1.0)
    }

    /// Uniform in `[-1, 1)`.
    fn next_f32(&mut self) -> f32 {
        self.random_range(-1.0..1.0)
    }

    /// A value or map with every coefficient uniform in `[-scale, scale)`, drawn in
    /// coefficient order.
    fn value<M: Extensor>(&mut self, scale: f64) -> M {
        M::from_coeffs(<M::Kind as Kind>::arr_from_fn(|_| {
            <M::Slots as Slots>::from_flat(&mut |_| M::Coef::from_f64(scale * self.next_f64()), 0)
        }))
    }

    /// `n` of [`Draw::value`].
    fn values<M: Extensor>(&mut self, n: usize, scale: f64) -> Vec<M> {
        (0..n).map(|_| self.value(scale)).collect()
    }
}

impl<R: rand::Rng> Draw for R {}
