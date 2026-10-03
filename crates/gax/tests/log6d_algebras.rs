//! The closed-form logarithm of 6D even versors (docs/log6d.md) on 6D algebras other than CSTA,
//! declared with `algebra!`: Euclidean `R(6,0)` (rotations only), split `R(3,3)` (rotations,
//! boosts and loxodromic planes) and degenerate `R(5,0,1)` (5D PGA: rotations and
//! translations). The generator emits the closed form for any 6D algebra's full even kind; here
//! it is checked against `exp`, near and past half turns too.

#![cfg(feature = "macros")]

#[path = "support/rng.rs"]
mod rng;
use rng::{Draw, Rng, rng};

gax::algebra! {
    algebra r60 "Euclidean 6D, R(6,0).";
    basis e1 = 1, e2 = 1, e3 = 1, e4 = 1, e5 = 1, e6 = 1;
    kind Scalar = [1];
    versor Vector = [e1, e2, e3, e4, e5, e6];
    kind Bivector = [e12, e13, e14, e15, e16, e23, e24, e25, e26, e34, e35, e36, e45, e46, e56];
    versor Even = [1, e12, e13, e14, e15, e16, e23, e24, e25, e26, e34, e35, e36, e45, e46, e56, e1234, e1235, e1236, e1245, e1246, e1256, e1345, e1346, e1356, e1456, e2345, e2346, e2356, e2456, e3456, e123456];
}

gax::algebra! {
    algebra r33 "Split signature, R(3,3).";
    basis e1 = 1, e2 = 1, e3 = 1, e4 = -1, e5 = -1, e6 = -1;
    kind Scalar = [1];
    versor Vector = [e1, e2, e3, e4, e5, e6];
    kind Bivector = [e12, e13, e14, e15, e16, e23, e24, e25, e26, e34, e35, e36, e45, e46, e56];
    versor Even = [1, e12, e13, e14, e15, e16, e23, e24, e25, e26, e34, e35, e36, e45, e46, e56, e1234, e1235, e1236, e1245, e1246, e1256, e1345, e1346, e1356, e1456, e2345, e2346, e2356, e2456, e3456, e123456];
}

gax::algebra! {
    algebra pga5d "Plane-based PGA of 5D Euclidean space, R(5,0,1).";
    basis e0 = 0, e1 = 1, e2 = 1, e3 = 1, e4 = 1, e5 = 1;
    kind Scalar = [1];
    versor Vector = [e0, e1, e2, e3, e4, e5];
    kind Bivector = [e01, e02, e03, e04, e05, e12, e13, e14, e15, e23, e24, e25, e34, e35, e45];
    versor Even = [1, e01, e02, e03, e04, e05, e12, e13, e14, e15, e23, e24, e25, e34, e35, e45, e0123, e0124, e0125, e0134, e0135, e0145, e0234, e0235, e0245, e0345, e1234, e1235, e1245, e1345, e2345, e012345];
}

fn close(a: &[f64], b: &[f64], tol: f64) -> bool {
    let scale = a.iter().chain(b).fold(1.0f64, |m, x| m.max(x.abs()));
    a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tol * scale)
}

/// The same checks on each algebra, whose kinds share names; `$planes` are three disjoint basis
/// planes (positions in `Bivector`), rotations first.
macro_rules! log6d_checks {
    ($alg:ident, $checks:ident, $planes:expr) => {
        mod $checks {
            use super::$alg::{Bivector, Even};
            use super::{Draw, Rng, close, rng};
            use gax::Unit;

            fn bivector(rng: &mut Rng, size: f64) -> Bivector<(), f64> {
                Bivector::from_coeffs(core::array::from_fn(|_| size * rng.next_f64()))
            }

            /// Small enough that every plane is well below a half turn: `log(exp B) = B`.
            #[test]
            fn log_of_exp_is_the_bivector() {
                let mut rng = rng(0x6d_1065);
                for size in [0.01, 0.1, 0.25] {
                    for _ in 0..40 {
                        let b = bivector(&mut rng, size);
                        let r = b.exp();
                        let back: Bivector<(), f64> = r.log();
                        assert!(
                            close(&back.c, &b.c, 1e-11),
                            "size {size}: {back:?} vs {b:?}"
                        );
                    }
                }
            }

            /// Larger versors, planes near and past half turns included (`⟨R⟩₀` of either sign):
            /// the log is a log of `R` and a fixed point of `log ∘ exp`.
            #[test]
            fn exp_of_log_is_the_versor() {
                let mut rng = rng(0x6d_e7b0);
                let mut checked = 0;
                for _ in 0..200 {
                    let r = bivector(&mut rng, 0.8).exp();
                    let b: Bivector<(), f64> = r.log();
                    let again = b.exp();
                    assert!(
                        close(&again.into_inner().c, &r.into_inner().c, 1e-10),
                        "exp(log R) differs from R"
                    );
                    let b2: Bivector<(), f64> = again.log();
                    assert!(
                        close(&b2.c, &b.c, 1e-10),
                        "log(exp(log R)) differs from log R"
                    );
                    checked += 1;
                }
                assert_eq!(checked, 200);
            }

            /// Single planes (two or three coinciding invariants), each basis bivector alone,
            /// and where it is a rotation, towards and past a half turn (turned first).
            #[test]
            fn single_planes() {
                for i in 0..15 {
                    let mut c = [0.0f64; 15];
                    c[i] = 0.7;
                    let b = Bivector::from_coeffs(c);
                    let back: Bivector<(), f64> = b.exp().log();
                    assert!(close(&back.c, &b.c, 1e-13), "plane {i}: {back:?}");
                    for angle in [
                        core::f64::consts::FRAC_PI_2 - 1e-6,
                        core::f64::consts::FRAC_PI_2 + 0.4,
                    ] {
                        c[i] = angle;
                        let b = Bivector::from_coeffs(c);
                        let r = b.exp();
                        if r.into_inner().c[0].abs() < 0.5 {
                            let back: Bivector<(), f64> = r.log();
                            assert!(
                                close(&back.c, &b.c, 1e-12),
                                "plane {i} at {angle}: {back:?}"
                            );
                        }
                    }
                }
            }

            /// Two rotation planes within `delta` of a half turn together and a third plane:
            /// the pair's invariants are tiny, and its product is taken from `p3` (from `p2`
            /// it would be lost to rounding, and the turned planes' sum with it). The planes
            /// themselves are fixed by `R` only to about `ε/δ`, and turning them costs
            /// `ε/δ²` (docs/log6d.md §5).
            #[test]
            fn two_planes_near_a_half_turn() {
                let planes: [usize; 3] = $planes;
                for delta in [1e-2, 1e-4, 1e-6] {
                    let tol = 1e-9 + 4e-15 / (delta * delta);
                    for third in [0.3, 1.1] {
                        let mut c = [0.0f64; 15];
                        c[planes[0]] = core::f64::consts::FRAC_PI_2 - delta;
                        c[planes[1]] = core::f64::consts::FRAC_PI_2 - delta;
                        c[planes[2]] = third;
                        let b = Bivector::from_coeffs(c);
                        let r = b.exp();
                        let got: Bivector<(), f64> = r.log();
                        let again = got.exp();
                        assert!(
                            close(&again.into_inner().c, &r.into_inner().c, tol),
                            "delta {delta}, third {third}: {got:?}"
                        );
                    }
                }
            }

            /// Three planes at one angle (a triple invariant) with `⟨R⟩₀ < 1/16`: the turning
            /// step must not split the triple (its isolated root's Newton step divided by a
            /// vanishing derivative and made the triple look apart).
            #[test]
            fn equal_planes_below_a_sixteenth() {
                let planes: [usize; 3] = $planes;
                for angle in [1.2, 1.228, 1.3, 1.45] {
                    let mut c = [0.0f64; 15];
                    for &i in &planes[..2] {
                        c[i] = angle;
                    }
                    // The third plane is a rotation in R(6,0) only; elsewhere it stays as given.
                    c[planes[2]] = if stringify!($alg) == "r60" {
                        angle
                    } else {
                        0.2
                    };
                    let b = Bivector::from_coeffs(c);
                    let r = b.exp();
                    let got: Bivector<(), f64> = r.log();
                    assert!(close(&got.c, &b.c, 1e-9), "angle {angle}: {got:?}");
                }
            }

            #[test]
            fn in_f32() {
                let mut rng = rng(0x6d_f32f);
                for _ in 0..50 {
                    let b = bivector(&mut rng, 0.4);
                    let r = b.exp().into_inner();
                    let r32 = Even::<(), f32>::from_coeffs(core::array::from_fn(|i| r.c[i] as f32));
                    let got: Bivector<(), f32> = Unit::new_unchecked(r32).log();
                    let got: [f64; 15] = core::array::from_fn(|i| f64::from(got.c[i]));
                    assert!(close(&got, &b.c, 2e-4), "{got:?} vs {:?}", b.c);
                }
            }
        }
    };
}

// Planes: R(6,0) e12, e34, e56; R(3,3) e12 (positive), e45 (negative), e36 (a boost);
// R(5,0,1) e12, e34 and e05 (a translation).
log6d_checks!(r60, r60_log, [0, 9, 14]);
log6d_checks!(r33, r33_log, [0, 12, 11]);
log6d_checks!(pga5d, pga5d_log, [5, 12, 4]);
