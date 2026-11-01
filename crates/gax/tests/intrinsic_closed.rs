//! Kinds declared `intrinsic` that are closed under the geometric product (ADR-051): their
//! values multiply to extrinsic ones (`I·I = E`), so the generated methods that multiply a kind
//! by itself (the general inverse's recursion, `exp`, `log`) work in the extrinsic orientation
//! and relabel at the end. These algebras did not compile before (the twistors port declared
//! its twistor kind extrinsic for it).

#![cfg(feature = "macros")]

use gax::ApproxEq;

gax::algebra! {
    algebra cl30i "Euclidean space with an intrinsic multivector (the general inverse).";
    basis e1 = 1, e2 = 1, e3 = 1;
    kind Scalar = [1];
    kind Vector = [e1, e2, e3];
    kind Bivector = [e23, e31, e12];
    kind Multivector = [1, e1, e2, e3, e23, e31, e12, e123] intrinsic;
}

gax::algebra! {
    algebra cl201i "The projective plane with an intrinsic multivector (the general inverse with a null direction).";
    basis e0 = 0, e1 = 1, e2 = 1;
    kind Scalar = [1];
    kind Vector = [e0, e1, e2];
    kind Multivector = [1, e0, e1, e2, e01, e02, e12, e012] intrinsic;
}

gax::algebra! {
    algebra cl20i "The plane with intrinsic rotors (the closed forms).";
    basis e1 = 1, e2 = 1;
    kind Scalar = [1];
    kind Vector = [e1, e2];
    kind Bivector = [e12];
    versor Rotor = [1, e12] intrinsic;
}

#[test]
fn the_general_inverse_of_an_intrinsic_kind() {
    let x = cl30i::Multivector::new(1.0, 0.2, -0.3, 0.4, 0.5, -0.6, 0.7, 0.8);
    // An intrinsic value's inverse is intrinsic: inversion keeps the orientation.
    let y: cl30i::Multivector = x.inverse();
    // Their product is extrinsic, and one.
    let one: cl30i::extrinsic::Multivector = x * y;
    let mut expected = cl30i::extrinsic::Multivector::zero();
    expected.c[0] = 1.0;
    assert!(one.approx_eq(&expected, 1e-12), "{one:?}");
    assert!(x.try_inverse().is_some());
    // The same coefficients as the extrinsic kind's inverse.
    let e: cl30i::extrinsic::Multivector = x.with_orientation();
    assert_eq!(y.c, e.inverse().c);
}

#[test]
fn the_general_inverse_with_a_null_direction() {
    let x = cl201i::Multivector::new(1.5, 0.2, -0.3, 0.4, 0.5, -0.6, 0.7, 0.8);
    let y: cl201i::Multivector = x.inverse();
    let one: cl201i::extrinsic::Multivector = y * x;
    let mut expected = cl201i::extrinsic::Multivector::zero();
    expected.c[0] = 1.0;
    assert!(one.approx_eq(&expected, 1e-12), "{one:?}");
}

#[test]
fn closed_forms_of_an_intrinsic_versor_kind() {
    let r = cl20i::Rotor::new(0.6, 0.8);
    let y: cl20i::Rotor = r.inverse();
    let one: cl20i::extrinsic::Rotor = r * y;
    assert!((one.s() - 1.0).abs() < 1e-15 && one.e12().abs() < 1e-15);
    // `exp` takes extrinsic values to extrinsic ones whatever the kinds' defaults.
    let b = cl20i::Bivector::new(0.5);
    let u: gax::Unit<cl20i::extrinsic::Rotor> = b.exp();
    let u = u.into_inner();
    assert!((u.s() - 0.5f64.cos()).abs() < 1e-15 && (u.e12() - 0.5f64.sin()).abs() < 1e-15);
    let back: cl20i::Bivector = u.unit().log();
    assert!((back.e12() - 0.5).abs() < 1e-15);
}
