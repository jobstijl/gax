//! The solvers on the native SIMD lanes of every level this CPU runs (SSE2 to AVX-512, NEON),
//! against the scalar solvers: `crates/gax-core/tests/solver_lanes.rs` checks the portable
//! lanes; here they run inside `batch::map`, where each level's lane type is the native one.
//!
//! * LU (`inverse`, `det`, `solve`) uses the same pivot rule and operation order on lanes as
//!   in scalar code, so it is bit-identical per element, on random and nearly singular maps.
//! * The Jacobi methods (`eigh`, `svd`) iterate until every lane has converged, so an element
//!   may get extra tiny rotations: the values agree within `c · n · ε · ‖A‖`.
#![cfg(all(feature = "batch", feature = "pga3d"))]

#[path = "support/rng.rs"]
mod rng;
use rng::Rng;

use gax::Real;
use gax::batch::{self, Map};
use gax::pga3d::{Multivector, Point, Scalar};

/// The 4x4 map on points held in a multivector's 16 coefficients.
fn matrix<T: Real>(x: Multivector<(), T>) -> Point<(Point,), T> {
    Point::from_coeffs(core::array::from_fn(|o| {
        core::array::from_fn(|i| x.c[4 * o + i])
    }))
}

/// The inverse's 16 entries.
struct Inverse;
impl Map for Inverse {
    type X = Multivector;
    type Y = Multivector;
    #[inline(always)]
    fn call<T: Real>(&self, x: Multivector<(), T>) -> Multivector<(), T> {
        let inv = matrix(x).inverse();
        Multivector::from_coeffs(core::array::from_fn(|k| inv.c[k / 4][k % 4]))
    }
}

/// `solve(A, b)` for a fixed `b`, and `det(A)`, in a point's four coefficients and the next.
struct SolveDet;
impl Map for SolveDet {
    type X = Multivector;
    type Y = Multivector;
    #[inline(always)]
    fn call<T: Real>(&self, x: Multivector<(), T>) -> Multivector<(), T> {
        let a = matrix(x);
        let b = Point::new(
            T::one(),
            T::from_f64(-2.0),
            T::from_f64(0.5),
            T::from_f64(3.0),
        );
        let s = a.solve(b);
        let d = a.det();
        Multivector::from_coeffs(core::array::from_fn(|k| match k {
            0..=3 => s.c[k],
            4 => d,
            _ => T::zero(),
        }))
    }
}

/// The singular values, and the eigenvalues of the symmetric part.
struct Spectra;
impl Map for Spectra {
    type X = Multivector;
    type Y = Multivector;
    #[inline(always)]
    fn call<T: Real>(&self, x: Multivector<(), T>) -> Multivector<(), T> {
        let a = matrix(x);
        let (_, sigma, _) = a.svd();
        let form = Scalar::<(Point, Point), T>::from_coeffs([core::array::from_fn(|i| {
            core::array::from_fn(|j| x.c[4 * i + j] + x.c[4 * j + i])
        })]);
        let (lambda, _) = form.eigh();
        Multivector::from_coeffs(core::array::from_fn(|k| match k {
            0..=3 => sigma[k],
            4..=7 => lambda[k - 4],
            _ => T::zero(),
        }))
    }
}

/// The projection onto the range, `A A⁺`, by the pseudo-inverse with the cutoff `1e-8`.
struct Projection;
impl Map for Projection {
    type X = Multivector;
    type Y = Multivector;
    #[inline(always)]
    fn call<T: Real>(&self, x: Multivector<(), T>) -> Multivector<(), T> {
        let a = matrix(x);
        let p = a.pinv_with(T::from_f64(1e-8));
        let ap = a.of(p);
        Multivector::from_coeffs(core::array::from_fn(|k| ap.c[k / 4][k % 4]))
    }
}

/// Random maps, and nearly singular ones (the last row a combination of the others plus
/// `delta` noise, `delta` from 1e-3 down to 1e-12), some with a zero leading entry.
fn inputs(n: usize) -> Vec<Multivector<(), f64>> {
    let mut rng = Rng(0x1ee7_c0de);
    (0..n)
        .map(|k| {
            let mut a: [f64; 16] = core::array::from_fn(|_| rng.next_f64());
            if k % 3 != 0 {
                let w = [rng.next_f64(), rng.next_f64(), rng.next_f64()];
                let delta = 10f64.powi(-3 - (k % 10) as i32);
                for j in 0..4 {
                    a[12 + j] =
                        (0..3).map(|i| a[4 * i + j] * w[i]).sum::<f64>() + delta * rng.next_f64();
                }
            }
            if k % 5 == 0 {
                a[0] = 0.0;
            }
            Multivector::from_coeffs(a)
        })
        .collect()
}

#[test]
fn lu_is_bit_identical_on_every_level() {
    let xs = inputs(101);
    let want_inv: Vec<_> = xs.iter().map(|&x| Inverse.call(x)).collect();
    let want_sd: Vec<_> = xs.iter().map(|&x| SolveDet.call(x)).collect();
    for level in batch::levels() {
        batch::with_level(level, || {
            let mut got = vec![Multivector::zero(); xs.len()];
            batch::map(&Inverse, &xs, &mut got);
            for (i, (g, w)) in got.iter().zip(&want_inv).enumerate() {
                assert!(
                    g.c.iter()
                        .zip(w.c)
                        .all(|(a, b)| a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())),
                    "inverse, {}: element {i} differs",
                    batch::level_name(batch::level())
                );
            }
            batch::map(&SolveDet, &xs, &mut got);
            for (i, (g, w)) in got.iter().zip(&want_sd).enumerate() {
                assert!(
                    g.c.iter()
                        .zip(w.c)
                        .all(|(a, b)| a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())),
                    "solve and det, {}: element {i} differs",
                    batch::level_name(batch::level())
                );
            }
        });
    }
}

#[test]
fn spectra_agree_on_every_level() {
    let xs = inputs(101);
    let want: Vec<_> = xs.iter().map(|&x| Spectra.call(x)).collect();
    for level in batch::levels() {
        batch::with_level(level, || {
            let mut got = vec![Multivector::zero(); xs.len()];
            batch::map(&Spectra, &xs, &mut got);
            for (i, ((g, w), x)) in got.iter().zip(&want).zip(&xs).enumerate() {
                let norm = x.c.iter().map(|v| v * v).sum::<f64>().sqrt();
                let tol = 64.0 * 4.0 * f64::EPSILON * norm.max(1.0);
                for k in 0..8 {
                    assert!(
                        (g.c[k] - w.c[k]).abs() <= tol,
                        "{} element {i}, value {k}: {} vs {}",
                        batch::level_name(batch::level()),
                        g.c[k],
                        w.c[k]
                    );
                }
            }
        });
    }
}

/// The pseudo-inverse on every level: random maps, and maps of rank exactly 3 (their last row a
/// combination of the others), far from the cutoff either way, so every lane keeps the same
/// singular values. `A A⁺` (the identity, or the projection onto the range) agrees with the
/// scalar path within rounding.
#[test]
fn pseudo_inverse_agrees_on_every_level() {
    let mut rng = Rng(0x9e17);
    let xs: Vec<Multivector<(), f64>> = (0..101)
        .map(|k| {
            let mut a: [f64; 16] = core::array::from_fn(|_| rng.next_f64());
            if k % 2 == 1 {
                let w = [rng.next_f64(), rng.next_f64(), rng.next_f64()];
                for j in 0..4 {
                    a[12 + j] = (0..3).map(|i| a[4 * i + j] * w[i]).sum::<f64>();
                }
            }
            Multivector::from_coeffs(a)
        })
        .collect();
    let want: Vec<_> = xs.iter().map(|&x| Projection.call(x)).collect();
    for level in batch::levels() {
        batch::with_level(level, || {
            let mut got = vec![Multivector::zero(); xs.len()];
            batch::map(&Projection, &xs, &mut got);
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                for k in 0..16 {
                    assert!(
                        (g.c[k] - w.c[k]).abs() <= 1e-9,
                        "{} element {i}, entry {k}: {} vs {}",
                        batch::level_name(batch::level()),
                        g.c[k],
                        w.c[k]
                    );
                }
            }
        });
    }
}
