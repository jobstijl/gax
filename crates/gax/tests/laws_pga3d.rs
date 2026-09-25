//! Algebraic laws of the generated PGA3D types, as property tests.
//!
//! Inputs are coefficients in [-1, 1]. The products involved have at most a few dozen terms, so
//! floating-point error is a few ulps times the magnitude of the result; a relative tolerance
//! of 1e-12 is about 1000 times that, and any real error is O(1).

#![cfg(feature = "pga3d")]

use gax::pga3d::*;
use gax::{Extensor, Kind, Unit};
use proptest::prelude::*;

fn close<M: Extensor<Slots = (), Coef = f64>>(a: &M, b: &M) -> bool {
    let (a, b) = (a.coeffs().as_ref(), b.coeffs().as_ref());
    let scale = a.iter().chain(b).fold(1.0f64, |m, x| m.max(x.abs()));
    a.iter().zip(b).all(|(x, y)| (x - y).abs() <= 1e-12 * scale)
}

fn motor() -> impl Strategy<Value = Motor<(), f64>> {
    prop::array::uniform8(-1.0f64..1.0).prop_map(Motor::from_coeffs)
}
fn mv() -> impl Strategy<Value = Multivector<(), f64>> {
    prop::array::uniform16(-1.0f64..1.0).prop_map(Multivector::from_coeffs)
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
    line().prop_map(|b| b.exp())
}

proptest! {
    #[test]
    fn geometric_product_is_associative(a in mv(), b in mv(), c in mv()) {
        prop_assert!(close(&((a * b) * c), &(a * (b * c))));
    }

    #[test]
    fn geometric_product_distributes(a in mv(), b in mv(), c in mv()) {
        prop_assert!(close(&(a * (b + c)), &(a * b + a * c)));
        prop_assert!(close(&((a + b) * c), &(a * c + b * c)));
    }

    #[test]
    fn reverse_is_an_anti_automorphism(a in mv(), b in mv()) {
        prop_assert!(close(&(a * b).reverse(), &(b.reverse() * a.reverse())));
    }

    #[test]
    fn motor_products_stay_motors(a in motor(), b in motor(), c in motor()) {
        // Even * even is even: the product type is Motor, and associativity holds there too.
        let ab: Motor<(), f64> = a * b;
        prop_assert!(close(&(ab * c), &(a * (b * c))));
    }

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
    fn binding_order_does_not_matter(a in point(), b in point(), c in point()) {
        let form: Scalar<(Point, Point, Point), f64> = ((Point::slot() & Point::slot()) & Point::slot()) & Point::new(0.3, -0.2, 0.7, 1.0);
        let v1 = form.of(a).of(b).of(c);
        let v2 = form.at::<1>().of(b).of(a).of(c);
        let v3 = form.at::<2>().of(c).at::<1>().of(b).of(a);
        prop_assert!(close(&v1, &v2) && close(&v1, &v3));
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

#[test]
fn kinds_have_the_declared_sizes() {
    assert_eq!(<Point as Kind>::N, 4);
    assert_eq!(<Line as Kind>::N, 6);
    assert_eq!(<Motor as Kind>::N, 8);
    assert_eq!(<Multivector as Kind>::N, 16);
}
