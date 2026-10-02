//! Geometry of PGA3D in floating point, as property tests: unit motors built by `exp` preserve
//! weights and norms and compose as products, incidence holds, and the induced map on planes
//! pulls back. The algebraic laws are proved exactly, on symbolic coefficients, in
//! `laws_pga3d.rs`.
//!
//! Inputs are coefficients in [-1, 1]. The products involved have at most a few dozen terms, so
//! floating-point error is a few ulps times the magnitude of the result; a relative tolerance
//! of 1e-12 is about 1000 times that, and any real error is O(1).

#![cfg(feature = "pga3d")]

use gax::pga3d::*;
use gax::{Extensor, Unit};
use proptest::prelude::*;

fn close<M: Extensor<Slots = (), Coef = f64>>(a: &M, b: &M) -> bool {
    let (a, b) = (a.coeffs().as_ref(), b.coeffs().as_ref());
    let scale = a.iter().chain(b).fold(1.0f64, |m, x| m.max(x.abs()));
    a.iter().zip(b).all(|(x, y)| (x - y).abs() <= 1e-12 * scale)
}

fn point() -> impl Strategy<Value = Point<(), f64>> {
    prop::array::uniform4(-1.0f64..1.0).prop_map(Point::from_coeffs)
}
fn plane() -> impl Strategy<Value = Plane<(), f64>> {
    prop::array::uniform4(-1.0f64..1.0).prop_map(Plane::from_coeffs)
}
fn line() -> impl Strategy<Value = Line<(), f64>> {
    prop::array::uniform6(-1.0f64..1.0).prop_map(Line::from_coeffs)
}
/// A unit motor: the exponential of a bivector.
fn unit_motor() -> impl Strategy<Value = Unit<Motor<(), f64>>> {
    line().prop_map(Line::exp)
}

proptest! {
    #[test]
    fn unit_sandwich_preserves_grade_and_norm(m in unit_motor(), p in point(), l in line()) {
        // The output types are the input kinds (grade preserved, by type).
        let p2: Point<(), f64> = m >> p;
        let l2: Line<(), f64> = m >> l;
        // Weight of a point and the Euclidean norm of a line are invariant.
        prop_assert!((p2.e123() - p.e123()).abs() < 1e-12);
        prop_assert!((l2.norm_squared() - l.norm_squared()).abs() < 1e-12 * (1.0 + l.norm_squared()));
    }

    #[test]
    fn sandwich_composes_as_product(a in unit_motor(), b in unit_motor(), p in point()) {
        let ab = a * b;
        prop_assert!(close(&(ab >> p), &(a >> (b >> p))));
        prop_assert!(close(&(a << (a >> p)), &p));
    }

    #[test]
    fn join_and_meet_are_dual(p in point(), q in point(), r in point(), e in plane()) {
        // Incidence: the plane through p, q, r contains each of them: (p & q & r) & p == 0.
        let pl: Plane<(), f64> = (p & q) & r;
        prop_assert!((pl & p).s().abs() < 1e-12);
        // Meet of that plane with another is a line lying in both.
        let l: Line<(), f64> = pl ^ e;
        let on: Point<(), f64> = l ^ pl;
        prop_assert!(on.coeffs().iter().all(|c| c.abs() < 1e-12));
    }

    #[test]
    fn map_adjoint_by_pairing(m in unit_motor(), l in plane(), p in point()) {
        // The plane map induced by the motor's point map satisfies induced(l) & p == l & t(p).
        let t: Point<(Point,), f64> = m >> Point::slot();
        let induced: Plane<(Plane,), f64> = (Plane::slot() & Point::slot()).solve(Plane::slot() & t);
        prop_assert!(((induced.of(l) & p).s() - (l & t.of(p)).s()).abs() < 1e-10);
        // For a rigid motion it pulls planes back: (m >> l) & (m >> p) == l & p, so the
        // induced map is m << l.
        prop_assert!(close(&induced.of(l), &(m << l)));
    }
}

/// `Motor::look_at` from eyes ever closer to straight above the target, `+z` up, in `f32` and
/// `f64`: the camera's `+z` points at the target and its `+y` is orthogonal to it. (In `f32`
/// the roll was once the shortest rotation between two opposite directions, whose axis is
/// rounding noise, and turned the camera round.)
#[test]
fn look_at_near_the_up_direction() {
    fn check<T: gax::Real + Into<f64>>(tol: f64) {
        for k in 0..12 {
            let d = 10f64.powi(-k);
            for (dx, dy) in [(0.0, d), (d, 0.0), (-d, d)] {
                let p = |x: f64, y: f64, z: f64| {
                    Point::<(), T>::xyz(T::from_f64(x), T::from_f64(y), T::from_f64(z))
                };
                let m = Motor::look_at(
                    p(dx, dy, 10.0),
                    p(0.0, 0.0, 0.0),
                    Point::direction(T::zero(), T::zero(), T::one()),
                );
                let f = m >> Point::direction(T::zero(), T::zero(), T::one());
                let f: [f64; 3] = [f.e032().into(), f.e013().into(), f.e021().into()];
                let want = [-dx, -dy, -10.0];
                let n = (dx * dx + dy * dy + 100.0).sqrt();
                for i in 0..3 {
                    assert!((f[i] - want[i] / n).abs() < tol, "d {d:e}: forward {f:?}");
                }
                let u = m >> Point::direction(T::zero(), T::one(), T::zero());
                let u: [f64; 3] = [u.e032().into(), u.e013().into(), u.e021().into()];
                let dot = u[0] * want[0] / n + u[1] * want[1] / n + u[2] * want[2] / n;
                assert!(dot.abs() < tol, "d {d:e}: up {u:?} along forward");
            }
        }
    }
    check::<f32>(1e-5);
    check::<f64>(1e-12);
}
