//! Kernels written once as plain generic functions. `build.rs` includes this same file and
//! traces the functions with symbolic coefficients to emit fused versions.

use gax::pga3d::{Line, Motor, Plane, Point};
use gax::{Real, Slots, Unit};

/// Move a point by a unit motor, then project it onto a plane from a light point:
/// a chain of a sandwich, a join and a meet.
pub fn shadow_of_moved<S: Slots, T: Real>(
    m: Unit<Motor<(), T>>,
    light: Point<(), T>,
    ground: Plane<(), T>,
    p: Point<S, T>,
) -> Point<S, T> {
    let moved = m >> p;
    let ray: Line<S, T> = light & moved;
    ray ^ ground
}

/// Compose two unit motors and apply the result to a point.
pub fn compose_apply<T: Real>(
    a: Unit<Motor<(), T>>,
    b: Unit<Motor<(), T>>,
    p: Point<(), T>,
) -> Point<(), T> {
    Unit::new_unchecked(a.into_inner() * b.into_inner()) >> p
}

/// Normalize a point to unit weight (a division: the reciprocal is computed once).
pub fn euclidean<T: Real>(p: Point<(), T>) -> Point<(), T> {
    let w = p.e123().recip();
    Point::new(p.e032() * w, p.e013() * w, p.e021() * w, T::one())
}

/// The same chain onto the floor plane `z = 0`, a constant: its zero coefficients fold away
/// in the fused kernel, which a call of the generic code cannot do.
pub fn shadow_on_floor<S: Slots, T: Real>(
    m: Unit<Motor<(), T>>,
    light: Point<(), T>,
    p: Point<S, T>,
) -> Point<S, T> {
    let floor = Plane::new(T::zero(), T::zero(), T::one(), T::zero());
    shadow_of_moved(m, light, floor, p)
}

/// Move a point along the screw motion `exp(b)` (sines and cosines inside).
pub fn screw_apply<T: Real>(b: Line<(), T>, p: Point<(), T>) -> Point<(), T> {
    b.exp() >> p
}
