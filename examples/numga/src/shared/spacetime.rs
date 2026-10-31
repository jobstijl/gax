//! numga's spacetime algebra, `STA = Algebra("x+y+z+t-")`, which is gax's own [`gax::sta`]
//! (ADR-055): space `x`, `y`, `z` (`e1, e2, e3`) square to one, time `t` (`e4`) to minus one,
//! and the kinds hold numga's blades in numga's order (`xt` is `e14`, `xyzt` is `e1234`). Shared
//! by the spacetime ports (optical rotation, black sandwiches, spherical synchronization, the
//! magnetic bottle, Thomas precession, Terrell rotation); each uses part of it.
//!
//! numga models events, velocities, rays and directions in spacetime as antivectors, intrinsic:
//! here the intrinsic trivectors, [`Direction`] (ADR-051). A vector is a slice of spacetime (a
//! cut, a hyperplane through the origin), which a direction crosses by their pairing `v & d`:
//! `t & d` is how far the direction runs in the lab's time, `x & d` how far along `x`. Fields
//! and the generators of boosts and turns are extrinsic bivectors; the even kind holds the
//! rotors, whose `exp`, `log` and `sqrt` gax generates in closed form.
//!
//! numga's `dual` is the metric (Hodge) dual, `~a I` with `I = xyzt`: gax's `hodge()`, which
//! squares to minus one on even grades and to one on odd ones (so numga's `dual_inverse` is
//! `-b.hodge()` on bivectors, `d.hodge()` on vectors and trivectors). gax's metric-free `dual()`
//! differs from it in the sign of the blades with `t`, and is not used here.

pub use gax::sta::*;

/// An event, a velocity, a ray or a direction in spacetime: an intrinsic trivector, numga's
/// antivector.
pub type Direction = intrinsic::Trivector;

use crate::{Dir3, Point3};
use gax::Orientation;

/// The slice `x = 0`, read as a coordinate: `x() & d` is how far `d` runs along `x`.
pub fn x() -> Vector {
    Vector::new(1.0, 0.0, 0.0, 0.0)
}

/// The slice `y = 0`.
pub fn y() -> Vector {
    Vector::new(0.0, 1.0, 0.0, 0.0)
}

/// The slice `z = 0`.
pub fn z() -> Vector {
    Vector::new(0.0, 0.0, 1.0, 0.0)
}

/// The lab's slice of space, `t = 0`: `t() & d` is how far `d` runs in the lab's time.
pub fn t() -> Vector {
    Vector::new(0.0, 0.0, 0.0, 1.0)
}

/// One unit along the lab's time, numga's `-mv.t.dual()`: the direction that pairs to one with
/// `t()` (the dual carries the minus of `t`'s square).
pub fn time() -> Direction {
    -t().hodge()
}

/// The unit directions of space, numga's `stack([mv.x, mv.y, mv.z]).dual()`.
pub fn space() -> [Direction; 3] {
    [x(), y(), z()].map(|v| v.hodge())
}

/// The direction with lab coordinates `x`, `y`, `z` and time `w`.
pub fn direction(x: f64, y: f64, z: f64, w: f64) -> Direction {
    let [ax, ay, az] = space();
    ax.gp(x) + ay.gp(y) + az.gp(z) + time().gp(w)
}

/// The part of a direction or an event in the lab's slice of space, its time taken away.
pub fn in_space(d: Direction) -> Direction {
    d - time().gp((t() & d).s())
}

/// A direction's lab coordinates: its pairings with the slices `x`, `y`, `z` and `t`.
pub fn coordinates<O: Orientation>(d: oriented::Trivector<(), f64, O>) -> [f64; 4] {
    [x(), y(), z(), t()].map(|v| (v & d).s())
}

// The spatial part of a direction or an event, drawn as a direction of the drawing's space.
impl<O: Orientation> Dir3 for oriented::Trivector<(), f64, O> {
    fn dir3(self) -> Point3 {
        let [x, y, z, _] = coordinates(self).map(|c| c as f32);
        Point3::direction(x, y, z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gax::ApproxEq;

    /// numga's dual, `~a xyzt`: its table on the basis blades, and minus one on the
    /// pseudoscalar.
    #[test]
    fn the_hodge_dual_is_numgas_dual() {
        assert!(
            x().hodge()
                .approx_eq(&Direction::new(0.0, 1.0, 0.0, 0.0), 1e-15)
        );
        assert!(
            t().hodge()
                .approx_eq(&Direction::new(1.0, 0.0, 0.0, 0.0), 1e-15)
        );
        let xy = Bivector::new(0.0, 0.0, 1.0, 0.0, 0.0, 0.0);
        let xt = Bivector::new(0.0, 0.0, 0.0, 1.0, 0.0, 0.0);
        let dual = |b: [f64; 6]| oriented::Bivector::<(), f64, gax::Intrinsic>::from_coeffs(b);
        assert!(
            xy.hodge()
                .approx_eq(&dual([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]), 1e-15)
        );
        assert!(
            xt.hodge()
                .approx_eq(&dual([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 1e-15)
        );
        let xyz = Trivector::new(1.0, 0.0, 0.0, 0.0);
        assert!((xyz.hodge() - t().with_orientation()).norm() < 1e-15);
        let unit: Pseudoscalar = Pseudoscalar::new(1.0);
        assert!((unit.hodge().s() + 1.0).abs() < 1e-15);
    }

    /// numga's pairings: time pairs to one with `t`, `x` with its dual; the plane `x` sweeps in
    /// time is `yz`, the field along it `xt`; the products of directions, and their commutator
    /// (halved, as numga's).
    #[test]
    fn pairings_are_numgas() {
        let [ax, ay, _] = space();
        assert!(time().approx_eq(&Direction::new(-1.0, 0.0, 0.0, 0.0), 1e-15));
        assert!(((t() & time()).s() - 1.0).abs() < 1e-15);
        assert!(((x() & ax).s() - 1.0).abs() < 1e-15);
        let swept = ax & time();
        let yz = oriented::Bivector::<(), f64, gax::Intrinsic>::new(1.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        assert!(swept.approx_eq(&yz, 1e-15));
        assert!(
            swept
                .hodge()
                .approx_eq(&Bivector::new(0.0, 0.0, 0.0, 1.0, 0.0, 0.0), 1e-15)
        );
        let zt = oriented::Bivector::<(), f64, gax::Intrinsic>::new(0.0, 0.0, 0.0, 0.0, 0.0, 1.0);
        assert!((ax & ay).approx_eq(&zt, 1e-15));
        assert!(
            (ax * time()).approx_eq(&Even::new(0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0), 1e-15)
        );
        assert!(((time() * time()).s() + 1.0).abs() < 1e-15);
        let xt = Bivector::new(0.0, 0.0, 0.0, 1.0, 0.0, 0.0);
        assert!(ax.commutator(time()).approx_eq(&-xt, 1e-15));
    }

    /// numga's reflections in `x` of time, of `x`'s and `y`'s directions, and of fields.
    #[test]
    fn reflections_are_numgas() {
        let [ax, ay, _] = space();
        let m = x();
        assert!((m >> time()).approx_eq(&time(), 1e-15));
        assert!((m >> ax).approx_eq(&-ax, 1e-15));
        assert!((m >> ay).approx_eq(&ay, 1e-15));
        let xy = Bivector::new(0.0, 0.0, 1.0, 0.0, 0.0, 0.0);
        assert!((m >> xy).approx_eq(&-xy, 1e-15));
    }

    /// numga's rotor `exp(0.3 xt + 0.2 yz)`, its logarithm and its square root.
    #[test]
    fn rotors_are_numgas() {
        let generator = Bivector::new(0.2, 0.0, 0.0, 0.3, 0.0, 0.0);
        let unit = generator.exp();
        let rotor = unit.into_inner();
        let numga = Even::new(
            1.02450134, 0.2076767, 0.0, 0.0, 0.29845016, 0.0, 0.0, 0.06049884,
        );
        assert!(rotor.abs_diff_eq(&numga, 1e-8), "{rotor:?}");
        assert!(unit.log().approx_eq(&generator, 1e-14));
        let root = rotor.sqrt().into_inner();
        let numga = Even::new(
            1.00621897, 0.10095865, 0.0, 0.0, 0.14981095, 0.0, 0.0, 0.01503123,
        );
        assert!(root.abs_diff_eq(&numga, 1e-8), "{root:?}");
    }
}
