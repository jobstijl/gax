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
