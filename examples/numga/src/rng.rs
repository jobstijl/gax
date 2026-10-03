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

    /// A direction uniform on the unit sphere in `n` dimensions (`n` normals, normalized).
    fn direction<const N: usize>(&mut self) -> [f64; N] {
        let v: [f64; N] = core::array::from_fn(|_| self.normal());
        #[allow(clippy::disallowed_methods)] // sampling, not geometry
        let n = v.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-300);
        v.map(|x| x / n)
    }
}

impl<R: rand::Rng + ?Sized> Draw for R {}
