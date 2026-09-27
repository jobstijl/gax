//! Kernels written once and traced at build time (`build.rs`) into fused Rust functions (with
//! SIMD batch forms) and WGSL functions: the GPU runs them in compute shaders, and the CPU runs
//! the same programs as the fallback and as the test oracle.
//!
//! Points are PGA2D points: a finite point is `(x, y, 1)`, a direction (a velocity, a force)
//! a point at infinity `(x, y, 0)`. The difference of two finite points is the direction
//! between them, and the norm of their join is their distance.

use gax::Real;
use gax::pga2d::Point;

/// One step of a lattice node of warped space.
///
/// * `p`, `v`: the node's position (a finite point) and velocity (a direction);
/// * `rest`: where it rests;
/// * `n`: its four neighbours' positions (a missing neighbour is `p` itself, which exerts no
///   force);
/// * `force`: the external acceleration (explosions, wells), a direction;
/// * `k`: `[spring stiffness, rest length, anchor stiffness, damping, dt]`.
///
/// Returns the new position and velocity.
pub fn grid_node<T: Real>(
    p: Point<(), T>,
    v: Point<(), T>,
    rest: Point<(), T>,
    n: [Point<(), T>; 4],
    force: Point<(), T>,
    k: [T; 5],
) -> (Point<(), T>, Point<(), T>) {
    let [spring, len, anchor, damping, dt] = k;
    let eps = T::from_f64(1e-4);
    let mut acc = force + (rest - p).gp(anchor);
    for q in n {
        // Hooke along each spring: the join's norm is the distance.
        let dist = (p & q).norm();
        acc += (q - p).gp(spring * (dist - len) / (dist + eps));
    }
    let v = (v + acc.gp(dt)).gp(T::one() - damping * dt);
    (p + v.gp(dt), v)
}

/// The acceleration a source at `s` gives a body at `p`: `strength (s - p) / (|s - p|² + r²)`,
/// towards the source for a positive strength (a well), away for a negative one (a blast).
/// `k` is `[strength, r²]`.
pub fn source_force<T: Real>(p: Point<(), T>, s: Point<(), T>, k: [T; 2]) -> Point<(), T> {
    let [strength, r2] = k;
    let d2 = (p & s).norm_squared();
    (s - p).gp(strength / (d2 + r2))
}

/// One step of a particle: velocity from the external acceleration, drag, then motion.
/// `k` is `[drag, dt]`.
pub fn particle_step<T: Real>(
    p: Point<(), T>,
    v: Point<(), T>,
    force: Point<(), T>,
    k: [T; 2],
) -> (Point<(), T>, Point<(), T>) {
    let [drag, dt] = k;
    let v = (v + force.gp(dt)).gp(T::one() - drag * dt);
    (p + v.gp(dt), v)
}
