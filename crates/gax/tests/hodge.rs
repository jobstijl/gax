//! The Hodge dual `x.hodge() = ~x I`, with the metric (numga's `dual`), against the metric-free
//! complement `dual`: equal where every basis vector squares to `+1`, opposite on blades with an
//! odd number of negative squares, and zero on blades containing a null basis vector.

#![cfg(all(feature = "vga3d", feature = "sta", feature = "pga3d"))]

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
