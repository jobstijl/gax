//! The Hodge dual `x.hodge() = ~x I`, with the metric (numga's `dual`), against the metric-free
//! complement `dual`: equal where every basis vector squares to `+1`, opposite on blades with an
//! odd number of negative squares, and zero on blades containing a null basis vector.

#![cfg(all(
    feature = "vga2d",
    feature = "vga3d",
    feature = "sta",
    feature = "pga3d",
    feature = "macros"
))]

#[test]
fn euclidean_hodge_is_the_complement() {
    use gax::vga3d::{Bivector, Vector};
    let v = Vector::<(), f64>::new(1.0, -2.0, 3.0);
    assert_eq!(v.hodge().c, v.dual().c);
    let b = Bivector::<(), f64>::new(0.5, 0.25, -1.0);
    assert_eq!(b.hodge().c, b.dual().c);
}

#[test]
fn spacetime_hodge_is_the_reverse_times_the_pseudoscalar() {
    use gax::sta::{Bivector, Pseudoscalar, Vector};
    let i = Pseudoscalar::<(), f64>::new(1.0);
    let v = Vector::<(), f64>::new(1.0, -2.0, 3.0, 0.5);
    let want = v.reverse() * i;
    assert_eq!(v.hodge().c, want.c);
    // A blade times its Hodge dual is its squared norm times I (a scalar multiple of I).
    let b = Bivector::<(), f64>::new(0.3, -0.7, 1.1, 0.2, 0.4, -0.9);
    let h = b.hodge();
    assert_eq!(h.c, (b.reverse() * i).c);
    // Twice: x ↦ x I² up to the reverse's sign, so ±x (I² = −1 in STA).
    let back = h.hodge();
    for (a, x) in back.c.iter().zip(b.c) {
        assert!((a.abs() - x.abs()).abs() < 1e-15);
    }
    // It differs from the metric-free complement on some blade.
    assert_ne!(v.hodge().c, v.dual().c);
}

#[test]
fn projective_hodge_vanishes_on_the_null_vector() {
    use gax::pga3d::Plane;
    // The plane at infinity e0 squares to zero: its Hodge dual vanishes, its complement does not.
    let inf = Plane::<(), f64>::new(0.0, 0.0, 0.0, 1.0);
    assert!(inf.hodge().c.iter().all(|x| *x == 0.0));
    assert!(inf.dual().c.iter().any(|x| *x != 0.0));
}

#[test]
fn hodge_applies_to_maps() {
    use gax::vga3d::Vector;
    // On a map it acts on the output: the identity map's Hodge dual sends v to v.hodge().
    let m = Vector::<(), f64>::slot().hodge();
    let v = Vector::<(), f64>::new(1.0, 2.0, 3.0);
    assert_eq!(m.of(v).c, v.hodge().c);
}

#[test]
fn hodge_inverse_undoes_hodge() {
    use gax::sta::{Bivector, Trivector, Vector};
    // In STA (`I² = −1`) the inverse is `-x.hodge()` on the even grades and `x.hodge()` on the
    // odd ones: `(−1)^(g(n−g)) / (I ~I)` with `I ~I = −1`.
    let v = Vector::<(), f64>::new(1.0, -2.0, 3.0, 0.5);
    assert_eq!(v.hodge_inverse().c, v.hodge().c);
    assert_eq!(v.hodge().hodge_inverse().c, v.c);
    let b = Bivector::<(), f64>::new(0.3, -0.7, 1.1, 0.2, 0.4, -0.9);
    assert_eq!(b.hodge_inverse().c, (-b.hodge()).c);
    assert_eq!(b.hodge_inverse().hodge().c, b.c);
    let t = Trivector::<(), f64>::new(0.1, 0.2, -0.3, 0.4);
    assert_eq!(t.hodge().hodge_inverse().c, t.c);
    // In the Euclidean plane, minus the dual on vectors (a quarter turn back).
    let w = gax::vga2d::Vector::<(), f64>::new(2.0, 1.0);
    assert_eq!(w.hodge_inverse().c, (-w.hodge()).c);
    // In VGA3D, odd dimension, the same as the dual.
    let u = gax::vga3d::Bivector::<(), f64>::new(0.5, 0.25, -1.0);
    assert_eq!(u.hodge_inverse().c, u.hodge().c);
}

#[test]
fn metric_is_numgas_dot() {
    use gax::vga3d::Bivector;
    // `⟨a ~b⟩₀`: positive on Euclidean bivectors, where the inner product `|` is negative.
    let b = Bivector::<(), f64>::new(0.0, 0.0, 2.0);
    assert_eq!(b.metric(b).s(), 4.0);
    assert_eq!((b | b).s(), -4.0);
    assert_eq!(b.metric(b).s(), b.norm_squared());
    // In STA it reads the metric on every grade: a timelike vector and an electric plane are
    // negative, a magnetic plane positive.
    let t = gax::sta::Vector::<(), f64>::new(0.0, 0.0, 0.0, 1.0);
    assert_eq!(t.metric(t).s(), -1.0);
    let e = gax::sta::Bivector::<(), f64>::new(0.0, 0.0, 0.0, 1.0, 0.0, 0.0);
    let m = gax::sta::Bivector::<(), f64>::new(1.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    assert_eq!((e.metric(e).s(), m.metric(m).s()), (-1.0, 1.0));
    // And a join with a Hodge dual (law S).
    let x = gax::sta::Bivector::<(), f64>::new(0.3, -0.7, 1.1, 0.2, 0.4, -0.9);
    assert!(((x & e.hodge()).s() - x.metric(e).s()).abs() < 1e-15);
}

gax::algebra! {
    algebra cl21 "A Lorentzian plane with no pseudoscalar kind declared.";
    basis e1 = 1, e2 = 1, e3 = -1;
    kind Scalar = [1];
    kind Vector = [e1, e2, e3];
    kind Bivector = [e12, e13, e23];
}

#[test]
fn hodge_needs_no_pseudoscalar_kind() {
    use cl21::{Bivector, Vector};
    let v = Vector::<(), f64>::new(1.0, 2.0, 3.0);
    // `~v I` with `I = e123`: `e1 e123 = e23`, `e2 e123 = −e13`, `e3 e123 = e12 e3² = −e12`.
    let h: cl21::intrinsic::Bivector = v.hodge();
    assert_eq!(h.c, [-3.0, -2.0, 1.0]);
    assert_eq!(h.hodge_inverse().c, v.c);
    let b = Bivector::<(), f64>::new(0.5, -1.0, 2.0);
    assert_eq!(b.hodge().hodge_inverse().c, b.c);
    // With no kind for `e123` of its own, a scalar's dual lands in the general multivector.
    let s: cl21::intrinsic::Multivector = cl21::Scalar::<(), f64>::new(2.0).hodge();
    let nonzero: Vec<f64> = s.c.into_iter().filter(|c| *c != 0.0).collect();
    assert_eq!(nonzero, [2.0]);
}
