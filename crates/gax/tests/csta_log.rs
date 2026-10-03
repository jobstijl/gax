//! The logarithm of CSTA's even versors (the full 6D conformal group), in closed form, turning
//! planes near a half turn first (docs/log6d.md): `exp(log R) = R` for versors built from random
//! bivectors mixing rotations, boosts and dilations, and `log(exp B) = B` for bivectors on the
//! principal branch.

#![cfg(feature = "csta")]

#[path = "support/rng.rs"]
mod rng;
use rng::Rng;

use gax::Unit;
use gax::csta::{Bivector, Even};

fn close<const N: usize>(a: &[f64; N], b: &[f64; N], tol: f64) -> bool {
    let scale = a.iter().chain(b).fold(1.0f64, |m, x| m.max(x.abs()));
    a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tol * scale)
}

#[test]
fn exp_of_log_is_the_versor() {
    let mut rng = Rng(0x0c57_a106);
    for size in [0.01, 0.1, 0.3, 0.6] {
        for _ in 0..40 {
            let r: Unit<Even<(), f64>> = rng.value::<Bivector<(), f64>>(size).exp();
            let b: Bivector<(), f64> = r.log();
            let back = b.exp();
            assert!(
                close(&back.into_inner().c, &r.into_inner().c, 1e-9),
                "size {size}: exp(log R) differs from R"
            );
        }
    }
}

#[test]
fn log_of_exp_is_the_bivector_on_the_principal_branch() {
    let mut rng = Rng(0x005e_ed6d);
    // Small enough that every rotation part is well below a half turn.
    for size in [0.01, 0.1, 0.25] {
        for _ in 0..40 {
            let b = rng.value::<Bivector<(), f64>>(size);
            let back: Bivector<(), f64> = b.exp().log();
            assert!(
                close(&back.c, &b.c, 1e-9),
                "size {size}: log(exp B) differs from B"
            );
        }
    }
}

/// Boosts (`e41`, `e42`, `e43`) and dilations (`eoi`) of large rapidity, alone and with a small
/// rotation.
#[test]
fn large_boosts_and_dilations() {
    let mut rng = Rng(0x0b00_57ed);
    for size in [1.0, 2.0, 3.0] {
        for _ in 0..20 {
            let mut c = [0.0f64; 15];
            for k in [3, 4, 5, 14] {
                c[k] = size * rng.next_f64();
            }
            let pure = Bivector::from_coeffs(c);
            let back: Bivector<(), f64> = pure.exp().log();
            assert!(
                close(&back.c, &pure.c, 1e-9),
                "size {size}: log(exp B) for a boost"
            );
            for k in [0, 1, 2] {
                c[k] = 0.2 * rng.next_f64();
            }
            let r = Bivector::from_coeffs(c).exp();
            let again = r.log().exp();
            assert!(
                close(&again.into_inner().c, &r.into_inner().c, 1e-9),
                "size {size}: exp(log R) with a rotation"
            );
        }
    }
}

fn plane(parts: &[(&str, f64)]) -> Bivector<(), f64> {
    use gax::Kind;
    let names = <Bivector as Kind>::BLADES;
    Bivector::from_coeffs(core::array::from_fn(|i| {
        parts
            .iter()
            .find(|(n, _)| names[i] == *n)
            .map_or(0.0, |(_, v)| *v)
    }))
}

/// Coinciding invariants, where a formula through the individual roots would divide by their
/// differences: one plane (two roots at 1), a translation (null: all three at 1), isoclinic
/// planes, and near the identity. The closed form takes them through a series at the roots'
/// mean, with no division.
#[test]
fn coinciding_invariants() {
    for (name, parts) in [
        ("one plane", vec![("e12", 0.7)]),
        ("one boost", vec![("e41", 0.9)]),
        ("translation", vec![("e1i", 0.8)]),
        ("transversion", vec![("e2o", 0.6)]),
        ("rotation and translation", vec![("e12", 0.5), ("e3i", 0.4)]),
        ("isoclinic", vec![("e23", 0.5), ("e1i", 0.3), ("e1o", -0.3)]),
        ("near the identity", vec![("e12", 1e-7), ("e41", 2e-7)]),
    ] {
        let b = plane(&parts);
        let back: Bivector<(), f64> = b.exp().log();
        assert!(close(&back.c, &b.c, 1e-13), "{name}: {back:?} vs {b:?}");
    }
}

/// Towards a half turn the closed form would lose `ε/⟨R⟩₀`; below `⟨R⟩₀ = 1/16` the log turns
/// that plane by a quarter turn first, and stays accurate.
#[test]
fn towards_a_half_turn() {
    for delta in [0.2, 1e-2, 1e-4, 1e-6, 1e-8] {
        let b = plane(&[("e12", core::f64::consts::FRAC_PI_2 - delta), ("e3i", 0.3)]);
        let back: Bivector<(), f64> = b.exp().log();
        assert!(close(&back.c, &b.c, 1e-12), "half turn - {delta}");
    }
}

/// Past a half turn (`⟨R⟩₀ < 0`), and towards one with boosts and dilations, where `1 + R` has
/// no real square root in some channel: `log(exp B) = B`.
#[test]
fn past_a_half_turn_and_with_boosts() {
    use core::f64::consts::FRAC_PI_2;
    for (name, parts) in [
        (
            "past, with a translation",
            vec![("e12", FRAC_PI_2 + 0.3), ("e3i", 0.4)],
        ),
        ("far past", vec![("e23", FRAC_PI_2 + 1.2), ("e1o", 0.2)]),
        (
            "towards, with a boost",
            vec![("e12", FRAC_PI_2 - 1e-4), ("e43", 1.5)],
        ),
        (
            "past, with a boost",
            vec![("e12", FRAC_PI_2 + 0.2), ("e43", 2.0)],
        ),
        (
            "towards, with a dilation",
            vec![("e31", FRAC_PI_2 - 1e-6), ("eoi", 2.0)],
        ),
    ] {
        let b = plane(&parts);
        let back: Bivector<(), f64> = b.exp().log();
        assert!(close(&back.c, &b.c, 1e-12), "{name}: {back:?} vs {b:?}");
    }
}

/// Large random versors, many near or past half turns in several planes: every log is a log of
/// `R` (where inverse scaling and squaring, the fallback before turning, missed by a central
/// element or found no square root).
#[test]
fn every_versor() {
    let mut rng = Rng(0x00e7_e3ee);
    for size in [1.2, 1.6] {
        for _ in 0..400 {
            let r = rng.value::<Bivector<(), f64>>(size).exp();
            let b: Bivector<(), f64> = r.log();
            let again = b.exp();
            assert!(
                close(&again.into_inner().c, &r.into_inner().c, 1e-8),
                "size {size}: exp(log R) differs from R: {:?} vs {:?}",
                again.into_inner().c,
                r.into_inner().c
            );
        }
    }
}

/// Larger random versors, rotations and boosts coupled (loxodromic planes), `⟨R⟩₀` of either
/// sign. The log is a log of `R`, and a fixed point: `log(exp(log R)) = log R`, as a principal
/// log is.
#[test]
fn larger_versors() {
    let mut rng = Rng(0x0006_d106);
    let mut checked = 0;
    for _ in 0..400 {
        let r = rng.value::<Bivector<(), f64>>(1.0).exp();
        let b: Bivector<(), f64> = r.log();
        let again = b.exp();
        assert!(
            close(&again.into_inner().c, &r.into_inner().c, 1e-11),
            "exp(log R) differs from R"
        );
        let b2: Bivector<(), f64> = again.log();
        assert!(
            close(&b2.c, &b.c, 1e-11),
            "log(exp(log R)) differs from log R"
        );
        checked += 1;
    }
    assert_eq!(checked, 400);
}

/// On SIMD lanes, a lane near a half turn is turned and the others take the closed form
/// directly, each equal to its scalar result.
#[cfg(feature = "batch")]
#[test]
fn lanes_mix_the_closed_form_and_turning() {
    type L = gax::batch::Lanes<f64, 4>;
    let bs = [
        plane(&[("e12", 0.4), ("e41", 0.3)]),
        plane(&[("e12", core::f64::consts::FRAC_PI_2 - 1e-5)]),
        plane(&[("e1i", 0.8)]),
        plane(&[("e23", 0.5), ("e1i", 0.3), ("e1o", -0.3)]),
    ];
    let rs: Vec<Even<(), f64>> = bs.iter().map(|b| b.exp().into_inner()).collect();
    let lanes = Even::<(), L>::from_coeffs(core::array::from_fn(|i| {
        L::new([rs[0].c[i], rs[1].c[i], rs[2].c[i], rs[3].c[i]])
    }));
    let got: Bivector<(), L> = Unit::new_unchecked(lanes).log();
    for (l, b) in bs.iter().enumerate() {
        let lane: [f64; 15] = core::array::from_fn(|i| got.c[i].v[l]);
        assert!(close(&lane, &b.c, 1e-11), "lane {l}");
    }
}

/// In `f32`: within `f32`'s accuracy of the `f64` log, on random versors and the degenerate
/// cases (the `f64` guards of the closed form underflow to zero in `f32`; only discarded
/// branches meet them).
#[test]
fn in_f32() {
    let mut rng = Rng(0x00f3_2f32);
    let mut cases: Vec<Bivector<(), f64>> = (0..100)
        .map(|_| rng.value::<Bivector<(), f64>>(0.5))
        .collect();
    cases.push(plane(&[("e1i", 0.8)]));
    cases.push(plane(&[("e12", 0.7)]));
    cases.push(plane(&[("e23", 0.5), ("e1i", 0.3), ("e1o", -0.3)]));
    cases.push(plane(&[("e12", core::f64::consts::FRAC_PI_2 - 1e-3)]));
    cases.push(plane(&[
        ("e12", core::f64::consts::FRAC_PI_2 + 0.3),
        ("e43", 1.0),
    ]));
    for b in cases {
        let r = b.exp().into_inner();
        let r32 = Even::<(), f32>::from_coeffs(core::array::from_fn(|i| r.c[i] as f32));
        let got: Bivector<(), f32> = Unit::new_unchecked(r32).log();
        let got: [f64; 15] = core::array::from_fn(|i| f64::from(got.c[i]));
        assert!(close(&got, &b.c, 2e-4), "{got:?} vs {:?}", b.c);
    }
}
