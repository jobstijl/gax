//! The 6D logarithm (CSTA's even versors, `docs/log6d.md`) in `f32`: finite, and within
//! `f32`'s precision of the `f64` result, for rotations up to near a half turn, boosts and
//! their mixtures. The Study functions' guards must hold in `f32`, where `1e-300` is zero.

#![cfg(feature = "csta")]

#[path = "support/rng.rs"]
mod rng;
use rng::{Draw, Rng, rng};

use gax::Unit;
use gax::csta::{Bivector, Even};

/// `B` with rotation angles and boost rapidities up to `size` in each of its planes.
fn bivector(rng: &mut Rng, rotation: f64, boost: f64) -> Bivector<(), f64> {
    // Coefficients `[e23, e31, e12, e41, e42, e43, ...]`: rotations, boosts, then the rest.
    Bivector::from_coeffs(core::array::from_fn(|i| match i {
        0..=2 => rotation * rng.next_f64(),
        3..=5 => boost * rng.next_f64(),
        _ => 0.3 * rng.next_f64(),
    }))
}

#[test]
fn log_in_f32_matches_f64() {
    let mut rng = rng(0x0f32_10c6);
    let cases = [
        (0.01, 0.0),
        (0.5, 0.0),
        (1.7, 0.0),
        (0.0, 0.5),
        (0.0, 2.5),
        (1.0, 1.0),
        (1.5, 2.0),
    ];
    for (rotation, boost) in cases {
        for _ in 0..200 {
            let r64: Unit<Even<(), f64>> = bivector(&mut rng, rotation, boost).exp();
            let b64: Bivector<(), f64> = r64.log();
            let r32 = Unit::new_unchecked(r64.into_inner().map_coefs(|x| x as f32));
            let b32: Bivector<(), f32> = r32.log();
            let scale = b64.c.iter().fold(1.0f64, |m, x| m.max(x.abs()));
            for (x, y) in b32.c.iter().zip(b64.c) {
                assert!(x.is_finite(), "{rotation} {boost}: {b32:?}");
                assert!(
                    (f64::from(*x) - y).abs() < 2e-3 * scale,
                    "{rotation} {boost}: {b32:?} vs {b64:?}"
                );
            }
        }
    }
}
