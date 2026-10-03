//! A seeded generator, so that a run is reproducible from its seed: `rand_pcg`'s PCG32
//! (XSH RR), seeded as the PCG reference does (the stream the game had before it used the
//! crate, so old replays still play).

use rand::{Rng as _, RngCore};
use rand_pcg::Pcg32;

/// The simulation's generator.
#[derive(Clone, Debug)]
pub struct Rng(Pcg32);

impl Rng {
    /// A generator from a seed.
    pub fn new(seed: u64) -> Rng {
        Rng(Pcg32::new(seed ^ 0x853c_49e6_748f_ea9b, seed))
    }

    /// A fingerprint of the generator's state, for state hashes: its next draw, from a copy.
    pub fn state(&self) -> u64 {
        self.0.clone().next_u64()
    }

    /// Uniform in `[0, 1)` (24 bits).
    pub fn unit(&mut self) -> f32 {
        self.0.random()
    }

    /// Uniform in `[lo, hi)`.
    pub fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.unit()
    }

    /// True with probability `p`.
    pub fn chance(&mut self, p: f32) -> bool {
        self.unit() < p
    }

    /// An angle in `[0, 2π)`.
    pub fn angle(&mut self) -> f32 {
        self.range(0.0, core::f32::consts::TAU)
    }
}
