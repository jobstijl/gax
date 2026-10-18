//! The particle update, written once: `build.rs` traces it into a fused Rust kernel (with a
//! SIMD batch form) and a WGSL function, so the CPU and the GPU run the same verified program.

use gax::pga2d::{Motor, Twist};
use gax::{Real, Unit};

/// One step of a particle moving with a constant rate: `m exp(dt B)`, where `B` is a PGA2D
/// twist, an extrinsic bivector (a turn about a point, or at infinity a translation), followed
/// by one Newton step so that `m` stays a unit motor however long it flies.
///
/// `m` is taken as a plain motor, not a `Unit`: tracing a `Unit` would let the simplifier
/// assume `m ~m = 1`, and the renormalization would simplify away.
pub fn particle_step<T: Real>(m: Motor<(), T>, rate: Twist<(), T>, dt: T) -> Motor<(), T> {
    (Unit::new_unchecked(m) * rate.gp(dt).exp())
        .renormalize_fast()
        .into_inner()
}
