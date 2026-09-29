//! The logarithm of CSTA's even versors (the full 6D conformal group), by inverse scaling and
//! squaring: `exp(log R) = R` for versors built from random bivectors mixing rotations, boosts
//! and dilations, and `log(exp B) = B` for bivectors on the principal branch.

#![cfg(feature = "csta")]

use gax::Unit;
use gax::csta::{Bivector, Even};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }
    fn bivector(&mut self, size: f64) -> Bivector<(), f64> {
        Bivector::from_coeffs(core::array::from_fn(|_| size * self.next()))
    }
}

fn close<const N: usize>(a: &[f64; N], b: &[f64; N], tol: f64) -> bool {
    let scale = a.iter().chain(b).fold(1.0f64, |m, x| m.max(x.abs()));
    a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tol * scale)
}

#[test]
fn exp_of_log_is_the_versor() {
    let mut rng = Rng(0x0c57_a106);
    for size in [0.01, 0.1, 0.3, 0.6] {
        for _ in 0..40 {
            let r: Unit<Even<(), f64>> = rng.bivector(size).exp();
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
            let b = rng.bivector(size);
            let back: Bivector<(), f64> = b.exp().log();
            assert!(
                close(&back.c, &b.c, 1e-9),
                "size {size}: log(exp B) differs from B"
            );
        }
    }
}

/// Boosts (`e41`, `e42`, `e43`) and dilations (`eoi`) of large rapidity, alone and with a small
/// rotation: the square roots' Newton steps start from parts scaled to at most 1, so they
/// converge however unequal the parts are.
#[test]
fn large_boosts_and_dilations() {
    let mut rng = Rng(0x0b00_57ed);
    for size in [1.0, 2.0, 3.0] {
        for _ in 0..20 {
            let mut c = [0.0f64; 15];
            for k in [3, 4, 5, 14] {
                c[k] = size * rng.next();
            }
            let pure = Bivector::from_coeffs(c);
            let back: Bivector<(), f64> = pure.exp().log();
            assert!(
                close(&back.c, &pure.c, 1e-9),
                "size {size}: log(exp B) for a boost"
            );
            for k in [0, 1, 2] {
                c[k] = 0.2 * rng.next();
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
