//! Division, embeddings, and the motor between two elements (`sqrt(b / a)`), in PGA3D and PGA2D.
//!
//! Inputs keep away from the degenerate cases (nearly coincident points, nearly opposite
//! elements, where the answer is ill conditioned), so a relative tolerance of 1e-9 is far above
//! rounding and any real error is O(1).

#![cfg(all(feature = "pga2d", feature = "pga3d"))]

use gax::pga3d::{Line, Motor, Plane, Point, Rotor, Translator};
use gax::{Extensor, Unit, pga2d};
use proptest::prelude::*;

fn close<M: Extensor<Slots = (), Coef = f64>>(a: &M, b: &M, tol: f64) -> bool {
    let (a, b) = (a.coeffs().as_ref(), b.coeffs().as_ref());
    let scale = a.iter().chain(b).fold(1.0f64, |m, x| m.max(x.abs()));
    a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tol * scale)
}

fn xyz() -> impl Strategy<Value = [f64; 3]> {
    prop::array::uniform3(-2.0f64..2.0)
}
/// A point with a weight away from zero, of either sign.
fn point() -> impl Strategy<Value = Point<(), f64>> {
    (xyz(), 0.3f64..2.0, any::<bool>()).prop_map(|([x, y, z], w, neg)| {
        let w = if neg { -w } else { w };
        Point::new(x * w, y * w, z * w, w)
    })
}
fn plane() -> impl Strategy<Value = Plane<(), f64>> {
    prop::array::uniform4(-1.0f64..1.0)
        .prop_filter("a normal", |c| {
            c[0] * c[0] + c[1] * c[1] + c[2] * c[2] > 0.1
        })
        .prop_map(Plane::from_coeffs)
}
/// A line through two points apart: a simple bivector, with any scale.
fn line() -> impl Strategy<Value = Line<(), f64>> {
    (point(), point())
        .prop_filter("apart", |(p, q)| {
            let (a, b) = (p.to_euclidean(), q.to_euclidean());
            (0..3).map(|i| (a[i] - b[i]).powi(2)).sum::<f64>() > 0.1
        })
        .prop_map(|(p, q)| p & q)
}

/// The ratio of two elements, normalized: its scalar part is the cosine of the angle between
/// them. Nearly `-1` is nearly opposite: ill conditioned, so kept out.
fn not_opposite<X: core::ops::Div<X, Output = Motor<(), f64>>>(a: X, b: X) -> bool {
    (b / a).normalized().s() > -0.9
}

proptest! {
    #[test]
    fn between_planes(a in plane(), b in plane()) {
        prop_assume!(not_opposite(a, b));
        let m = Motor::between(a, b);
        prop_assert!(close(&(m >> a).normalized().into_inner(), &b.normalized().into_inner(), 1e-9));
    }

    #[test]
    fn between_lines(a in line(), b in line()) {
        prop_assume!(not_opposite(a, b));
        let m = Motor::between(a, b);
        prop_assert!(close(&(m >> a).normalized().into_inner(), &b.normalized().into_inner(), 1e-9));
    }

    /// Points of either weight: the translation between them, a pure translation.
    #[test]
    fn between_points(a in point(), b in point()) {
        let m = Motor::between(a, b);
        prop_assert!(close(&(m >> a).unitized(), &b.unitized(), 1e-9));
        let r = m.into_inner();
        prop_assert!(close(&Rotor::new(r.s(), r.e12(), r.e31(), r.e23()), &Rotor::new(1.0, 0.0, 0.0, 0.0), 1e-12));
    }

    /// For unit elements the literal formula, `(b / a).sqrt()`, is the same motor.
    #[test]
    fn between_is_the_square_root_of_the_ratio(a in line(), b in line()) {
        prop_assume!(not_opposite(a, b));
        let (a, b) = (a.normalized().into_inner(), b.normalized().into_inner());
        let root = (b / a).sqrt();
        prop_assert!(close(&root.into_inner(), &Motor::between(a, b).into_inner(), 1e-9));
        prop_assert!(close(&(root * root).into_inner(), &(b / a), 1e-9));
    }

    #[test]
    fn between_in_the_plane(a in prop::array::uniform3(-1.0f64..1.0), b in prop::array::uniform3(-1.0f64..1.0)) {
        let (a, b) = (pga2d::Line::from_coeffs(a), pga2d::Line::from_coeffs(b));
        prop_assume!(a.e1().hypot(a.e2()) > 0.3 && b.e1().hypot(b.e2()) > 0.3);
        prop_assume!((b / a).normalized().s() > -0.9);
        let m = pga2d::Motor::between(a, b);
        prop_assert!(close(&(m >> a).normalized().into_inner(), &b.normalized().into_inner(), 1e-9));
        let (p, q) = (pga2d::Point::xy(a.e1(), a.e2()), pga2d::Point::new(b.e0(), b.e1(), -0.5));
        let t = pga2d::Motor::between(p, q);
        prop_assert!(close(&(t >> p).unitized(), &q.unitized(), 1e-9));
    }

    /// Division is the product with the inverse, and undoes it; the dividend may be a map.
    #[test]
    fn division_undoes_the_product(a in line(), b in plane(), p in point()) {
        prop_assert!(close(&((a * b) / b), &Motor::from(a), 1e-9));
        let pb: Motor<(), f64> = p * b; // grades 2 and 4
        prop_assert!(close(&(pb / b), &gax::pga3d::Flector::from(p), 1e-9));
        let map: Motor<(Plane,), f64> = Plane::slot() / b;
        prop_assert!(close(&map.of(b), &Motor::from(gax::pga3d::Scalar::new(1.0)), 1e-12));
        prop_assert!(close(&(p / 2.0), &(p * 0.5), 0.0));
        // By a unit versor: the reverse, the same as the inverse.
        let u = (Line::new(0.3, -0.2, 0.5, 0.1, 0.7, -0.4) * 0.8).exp();
        prop_assert!(close(&(a / u), &(a / u.into_inner()), 1e-12));
        prop_assert!(close(&((p / u) * u.into_inner()), &gax::pga3d::Flector::from(p), 1e-12));
    }
}

/// Embeddings keep the multivector: a rotor or a translator acts the same as a motor.
#[test]
fn embeddings_keep_the_action() {
    let r = Unit::new_unchecked(Rotor::<(), f64>::new(0.6, 0.0, 0.8, 0.0));
    let t: Unit<Translator<(), f64>> =
        (Point::xyz(1.0, -2.0, 0.5) / Point::xyz(0.0, 0.0, 0.0)).sqrt();
    let p = Point::xyz(0.3, 0.4, -1.0);
    assert!(close(&(r.widen::<Motor<(), f64>>() >> p), &(r >> p), 1e-15));
    assert!(close(&(t.widen::<Motor<(), f64>>() >> p), &(t >> p), 1e-15));
    // Blades stored with another orientation change sign: e31 in a rotor, e13 nowhere; the
    // multivector is the same, so the products agree.
    let m: Motor<(), f64> = r.into_inner().into();
    assert!(close(
        &(m * m),
        &Motor::from(r.into_inner() * r.into_inner()),
        1e-15
    ));
}

/// Near a half turn the motor between two elements stays as precise in `f32` as the geometry
/// allows. Nearly opposite planes meet far away (at about `1/δ`), so an angle error `ε` moves
/// the result by `ε/δ`: the best possible error grows like `ulp/δ`. The literal
/// `normalize(1 + b / a)` loses the angle itself to cancellation, an error like `ulp/δ²`,
/// which is O(1) by `δ = 1e-4`.
#[test]
fn between_nearly_opposite_is_precise_in_f32() {
    for k in 1..=5 {
        let delta = 10f64.powi(-k);
        let th = core::f64::consts::PI - delta;
        let (c, s) = (th.cos() as f32, th.sin() as f32);
        let a = Plane::<(), f32>::new(1.0, 0.0, 0.0, 0.3);
        let b = Plane::<(), f32>::new(c, s, 0.0, -0.2);
        let err = (Motor::between(a, b) >> a)
            .c
            .iter()
            .zip(b.c)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(
            f64::from(err) < 2e-7 / delta,
            "planes {delta:e} from a half turn: off by {err:e}"
        );
        let a = (Point::<(), f32>::xyz(0.0, 0.0, 1.0) & Point::xyz(1.0, 0.0, 1.0))
            .normalized()
            .into_inner();
        let b = (Point::<(), f32>::xyz(0.0, 0.5, 0.0) & Point::xyz(c, 0.5 + s, 0.0))
            .normalized()
            .into_inner();
        let err = (Motor::between(a, b) >> a)
            .c
            .iter()
            .zip(b.c)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(
            f64::from(err) < 1e-6 / delta,
            "lines {delta:e} from a half turn: off by {err:e}"
        );
    }
}
