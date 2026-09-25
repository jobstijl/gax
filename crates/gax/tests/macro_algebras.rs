//! User algebras declared with `algebra!`: STAP `R(3,1,1)` and CSTA `R(4,2)`, the signatures
//! the standard algebras do not cover.

#![cfg(feature = "macros")]

gax::algebra! {
    algebra stap "Spacetime algebra with a projective (null) dimension, R(3,1,1).";
    basis ep = 0, e0 = 1, e1 = -1, e2 = -1, e3 = -1;
    kind Scalar = [1];
    versor Vector = [ep, e0, e1, e2, e3];
    kind Bivector = [ep0, ep1, ep2, ep3, e01, e02, e03, e12, e31, e23];
    versor Even = [1, ep0, ep1, ep2, ep3, e01, e02, e03, e12, e31, e23, ep012, ep031, ep023, ep123, e0123];
    kind Multivector = [1, ep, e0, e1, e2, e3, ep0, ep1, ep2, ep3, e01, e02, e03, e12, e31, e23, ep01, ep02, ep03, ep12, ep31, ep23, e012, e031, e023, e123, ep012, ep031, ep023, ep123, e0123, ep0123];
}

gax::algebra! {
    algebra csta "Conformal spacetime algebra R(4,2).";
    basis e0 = 1, e1 = -1, e2 = -1, e3 = -1, ep = 1, em = -1;
    kind Scalar = [1];
    versor Vector = [e0, e1, e2, e3, ep, em];
    kind Bivector = [e01, e02, e03, e12, e31, e23, e0p, e1p, e2p, e3p, e0m, e1m, e2m, e3m, epm];
    versor Even = [1, e01, e02, e03, e12, e31, e23, e0p, e1p, e2p, e3p, e0m, e1m, e2m, e3m, epm,
        e0123, e012p, e031p, e023p, e123p, e012m, e031m, e023m, e123m, e01pm, e02pm, e03pm, e12pm, e31pm, e23pm, e0123pm];
}

#[test]
fn stap_products_and_sandwich() {
    use stap::*;
    let v = Vector::<(), f64>::new(0.5, 2.0, 0.3, -0.2, 0.1);
    let w = Vector::<(), f64>::new(-1.0, 1.0, 0.7, 0.4, -0.5);
    // v w = v . w + v ^ w, and v . w uses the metric: e_p² = 0, e0² = 1, e_i² = -1.
    let vw: Even<(), f64> = v * w;
    let dot = 2.0 * 1.0 - 0.3 * 0.7 - (-0.2) * 0.4 - 0.1 * (-0.5);
    assert!((vw.s() - dot).abs() < 1e-12);
    // A reflection keeps the metric: (v >> w)² == v⁴ w².
    let r: Vector<(), f64> = v >> w;
    let (vv, ww, rr) = ((v * v).s(), (w * w).s(), (r * r).s());
    assert!((rr - vv * vv * ww).abs() < 1e-10);
}

#[test]
fn csta_is_six_dimensional() {
    use csta::*;
    let a = Vector::<(), f64>::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0);
    // (e0 + ep)² = 1 + 1 = 2
    assert!(((a | a).s() - 2.0).abs() < 1e-12);
    let b: Bivector<(Vector,), f64> = a ^ Vector::slot();
    assert_eq!(b.of(a), Bivector::zero());
}

#[test]
fn stap_bivector_exp_log_round_trip() {
    use stap::*;
    // A 5D bivector: its square has a scalar and a 4-vector part, handled by the general
    // Study-number functions.
    let b = Bivector::<(), f64>::new(0.1, -0.2, 0.05, 0.3, 0.2, 0.1, -0.15, 0.4, 0.25, -0.3);
    let r = b.exp();
    let back: Bivector<(), f64> = r.log();
    for (x, y) in back.c.iter().zip(b.c.iter()) {
        assert!((x - y).abs() < 1e-10, "{back:?} vs {b:?}");
    }
    // exp(B) is a unit versor: R ~R = 1.
    let n: Even<(), f64> = r.into_inner() * r.into_inner().reverse();
    assert!((n.s() - 1.0).abs() < 1e-12);
}

#[test]
fn csta_exp_by_scaling_and_squaring() {
    use csta::*;
    // A rotation in the e12 plane (e1² = e2² = -1, so e12² = -1): exp(θ e12) = cos θ + sin θ e12.
    let th = 0.9f64;
    let b = Bivector::<(), f64>::new(
        0.0, 0.0, 0.0, th, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    );
    let r = b.exp().into_inner();
    assert!(
        (r.s() - th.cos()).abs() < 1e-12 && (r.e12() - th.sin()).abs() < 1e-12,
        "{r:?}"
    );
    // A general bivector: exp(B) exp(-B) = 1 and exp(B) is a unit versor.
    let g = Bivector::<(), f64>::new(
        0.1, -0.2, 0.3, 0.2, 0.1, -0.3, 0.05, 0.1, 0.2, -0.1, 0.3, 0.2, -0.15, 0.1, 0.25,
    );
    let (p, m) = (g.exp().into_inner(), (-g).exp().into_inner());
    let id: Even<(), f64> = p * m;
    assert!(
        (id.s() - 1.0).abs() < 1e-11 && id.c[1..].iter().all(|c| c.abs() < 1e-11),
        "{id:?}"
    );
    let n: Even<(), f64> = p * p.reverse();
    assert!((n.s() - 1.0).abs() < 1e-11 && n.c[1..].iter().all(|c| c.abs() < 1e-11));
}

gax::algebra! {
    algebra pga4d "Plane-based PGA of 4D Euclidean space, R(4,0,1).";
    basis e0 = 0, e1 = 1, e2 = 1, e3 = 1, e4 = 1;
    kind Hyperplane = [e1, e2, e3, e4, e0];
    kind Bivector = [e12, e13, e14, e23, e24, e34, e01, e02, e03, e04];
    kind Point = [e0234, e0134, e0124, e0123, e1234];
    versor Motor = [1, e12, e13, e14, e23, e24, e34, e01, e02, e03, e04,
        e1234, e0123, e0124, e0134, e0234];
}

#[test]
fn pga4d_motors_move_points_rigidly() {
    use pga4d::*;
    // A rotation in two orthogonal planes plus a translation: a 5D bivector with a 4-vector
    // part in its square, so exp and log take the general Study-number path.
    let b = Bivector::<(), f64>::new(0.3, 0.0, 0.0, 0.0, 0.0, 0.5, 0.1, -0.2, 0.3, 0.4);
    let m = b.exp();
    let back: Bivector<(), f64> = m.log();
    for (x, y) in back.c.iter().zip(b.c.iter()) {
        assert!((x - y).abs() < 1e-10, "{back:?} vs {b:?}");
    }
    // Rigid: the weight of a point and the distance between two points are preserved.
    let p = Point::<(), f64>::new(1.0, 2.0, 3.0, 4.0, 1.0);
    let q = Point::<(), f64>::new(-1.0, 0.5, 2.0, 0.0, 1.0);
    let (p2, q2) = (m >> p, m >> q);
    assert!((p2.e1234() - 1.0).abs() < 1e-12);
    let d = |a: Point<(), f64>, b: Point<(), f64>| {
        (0..4)
            .map(|i| (a.c[i] / a.c[4] - b.c[i] / b.c[4]).powi(2))
            .sum::<f64>()
    };
    assert!((d(p, q) - d(p2, q2)).abs() < 1e-10);
}
