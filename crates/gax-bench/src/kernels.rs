//! Kernels shared by the benchmarks and by `build.rs`, which traces them into fused versions.

use gax::Real;
use gax::pga3d::{Line, Motor};

/// Principal inertia of a rigid body in its own frame (at its centre of mass), as a map from
/// twists to forques: `I[B] = J(D B)`, with `D` the masses and moments and `J` the dual
/// (Dorst & De Keninck, "May the Forque Be with You").
#[inline(always)]
pub fn inertia<T: Real>(b: Line<(), T>, mass: T, moments: [T; 3]) -> Line<(), T> {
    Line::new(
        mass * b.e01(),
        mass * b.e02(),
        mass * b.e03(),
        moments[0] * b.e23(),
        moments[1] * b.e31(),
        moments[2] * b.e12(),
    )
}

/// The inverse of [`inertia`].
#[inline(always)]
pub fn inertia_inv<T: Real>(f: Line<(), T>, mass: T, moments: [T; 3]) -> Line<(), T> {
    let r = mass.recip();
    Line::new(
        f.e01() / moments[0],
        f.e02() / moments[1],
        f.e03() / moments[2],
        f.e23() * r,
        f.e31() * r,
        f.e12() * r,
    )
}

/// One explicit Euler step of a free rigid body under a constant forque (in the body frame):
/// `Ṁ = -½ M B`, `Ḃ = I⁻¹[B × I[B] + F]`, followed by renormalization of the motor.
#[inline(always)]
pub fn rigid_step<T: Real>(
    m: Motor<(), T>,
    b: Line<(), T>,
    f: Line<(), T>,
    dt: T,
    mass: T,
    moments: [T; 3],
) -> (Motor<(), T>, Line<(), T>) {
    let db = inertia_inv(b.commutator(inertia(b, mass, moments)) + f, mass, moments);
    let dm: Motor<(), T> = (m * b).gp(T::from_f64(-0.5));
    let m2 = (m + dm.gp(dt)).normalized().into_inner();
    (m2, b + db.gp(dt))
}

/// [`rigid_step`] with the body's constants fixed, as a game would trace it: gravity along -z
/// in the body frame is not constant, so it stays an argument; the time step and the body's
/// mass and moments are compile-time constants here.
#[inline(always)]
pub fn rigid_step_fixed<T: Real>(
    m: Motor<(), T>,
    b: Line<(), T>,
    f: Line<(), T>,
) -> (Motor<(), T>, Line<(), T>) {
    let k = T::from_f64;
    rigid_step(m, b, f, k(1.0 / 60.0), k(2.0), [k(0.5), k(0.75), k(1.0)])
}
