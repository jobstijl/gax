//! The algebra of the scalars alone, Cl(0,0): `basis;` declares no vectors.

#![cfg(feature = "macros")]

gax::algebra! {
    algebra scalars "The real numbers as a geometric algebra, Cl(0,0).";
    basis;
    kind Scalar = [1];
}

use scalars::Scalar;

#[test]
fn the_scalars_are_an_algebra() {
    let (a, b) = (Scalar::<(), f64>::new(3.0), Scalar::new(-2.0));
    assert_eq!((a * b).s(), -6.0);
    assert_eq!((a + b).s(), 1.0);
    assert_eq!(a.reverse(), a);
    // A map on the scalars is a number, and binding it scales.
    let twice = Scalar::<(), f64>::slot().gp(2.0);
    assert_eq!(twice.of(a).s(), 6.0);
}
