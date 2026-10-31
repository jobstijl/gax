//! A scalar's square root is its number's. As a unit versor every positive scalar normalizes to
//! 1, so the versors' `normalize(1 + R)` gave 1 for all of them (four numga ports were caught by
//! it); scalar kinds have their own `sqrt`, NaN below zero as `f64::sqrt`.

#![cfg(all(feature = "vga3d", feature = "pga3d", feature = "macros"))]

use gax::Sqrt;

gax::algebra! {
    algebra cl20 "The Euclidean plane, declared here.";
    basis e1 = 1, e2 = 1;
    kind Scalar = [1];
    versor Vector = [e1, e2];
    versor Rotor = [1, e12];
}

#[test]
fn a_scalar_sqrt_is_the_numbers() {
    assert_eq!(gax::vga3d::Scalar::new(2.0).sqrt().s(), 2.0f64.sqrt());
    assert_eq!(gax::pga3d::Scalar::new(9.0).sqrt().s(), 3.0);
    assert_eq!(cl20::Scalar::new(0.25).sqrt().s(), 0.5);
    assert!(gax::vga3d::Scalar::new(-1.0f64).sqrt().s().is_nan());
    // Through the trait, as generic code calls it.
    assert_eq!(Sqrt::sqrt(gax::vga3d::Scalar::new(16.0)).s(), 4.0);
}

#[test]
fn a_rotor_sqrt_is_still_the_versors() {
    // `e12`, the rotor of a half turn; its square root squares back to it.
    let r = cl20::Rotor::<(), f64>::new(0.0, 1.0).unit();
    let h = r.sqrt().into_inner();
    let back = h * h;
    assert!(back.s().abs() < 1e-15 && (back.e12() - 1.0).abs() < 1e-15);
}
