//! Random numbers for the examples: `rand`'s seeded generator, with the draws they use
//! (numpy's streams cannot be reproduced, so the checks hold for any draw).

use rand::SeedableRng;

/// The generator.
pub type Rng = rand::rngs::StdRng;

/// A generator from a seed.
pub fn rng(seed: u64) -> Rng {
    Rng::seed_from_u64(seed)
}

/// The draws the examples use, for any generator.
pub trait Draw: rand::Rng {
    /// Uniform in `[0, 1)`.
    fn uniform(&mut self) -> f64 {
        self.random()
    }

    /// Uniform in `[lo, hi)`.
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        self.random_range(lo..hi)
    }

    /// A standard normal draw.
    fn normal(&mut self) -> f64 {
        self.sample(rand_distr::StandardNormal)
    }

    /// A value of unit norm, uniform over the directions of its kind (a unit vector of any
    /// algebra, uniform on its sphere): normal coefficients, normalized.
    fn direction<V>(&mut self) -> V
    where
        V: gax::Extensor<Slots = (), Coef = f64> + gax::Normalize,
    {
        let c = <V::Kind as gax::Kind>::arr_from_fn(|_| self.normal());
        V::from_coeffs(c).normalized().into_inner()
    }
}

impl<R: rand::Rng + ?Sized> Draw for R {}
