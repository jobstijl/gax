//! The scalar solvers (branching pivots) and the lane solvers (select-based pivots) agree on
//! nearly singular inputs (docs/numerics.md, "Solvers").
//!
//! LU uses the same pivot rule in both (the first row with the strictly largest magnitude) and
//! the same operation order, so `det`, `inverse` and `solve` are bit-identical per lane, even
//! where the pivot is ill-determined. Their accuracy is then a property of the one algorithm:
//! the forward error of `solve` is within `c · n · ε · κ∞(A)`. The Jacobi methods iterate until
//! every lane has converged, so a lane may get extra (tiny) rotations; eigenvalues and singular
//! values agree within `c · n · ε · ‖A‖`, and vectors within that divided by the gap to the
//! nearest other value, with the same order and the same signs.
#![cfg(feature = "batch")]
#![allow(clippy::needless_range_loop, clippy::float_cmp)] // exact comparisons are the point

use gax_core::batch::Lanes;
use gax_core::linalg::*;
use proptest::prelude::*;

type L = Lanes<f64, 4>;
const EPS: f64 = f64::EPSILON;

fn lanes<const N: usize>(m: &[[[f64; N]; N]; 4]) -> [[L; N]; N] {
    core::array::from_fn(|i| core::array::from_fn(|j| L::new(core::array::from_fn(|l| m[l][i][j]))))
}
fn inf_norm<const N: usize>(a: &[[f64; N]; N]) -> f64 {
    a.iter()
        .map(|r| r.iter().map(|x| x.abs()).sum::<f64>())
        .fold(0.0, f64::max)
}
fn same_bits(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

/// A matrix whose last row is a combination of the others plus `delta` times noise, so that
/// `κ(A)` is about `1/delta`; with `swap` the leading entry is zero, forcing a pivot.
fn near_singular<const N: usize>(v: &[f64], delta: f64, swap: bool) -> [[f64; N]; N] {
    let mut a: [[f64; N]; N] = core::array::from_fn(|i| core::array::from_fn(|j| v[i * N + j]));
    for j in 0..N {
        let comb: f64 = (0..N - 1).map(|i| a[i][j] * v[40 + i]).sum();
        a[N - 1][j] = comb + delta * v[50 + j];
    }
    if swap {
        a[0][0] = 0.0;
    }
    a
}

/// A symmetric matrix with eigenvalues `1, 1 + gap, 2, 3, ...` in a random orthonormal basis
/// (the Q of a Gram–Schmidt of random rows).
fn clustered<const N: usize>(v: &[f64], gap: f64) -> [[f64; N]; N] {
    let mut q: [[f64; N]; N] = core::array::from_fn(|i| core::array::from_fn(|j| v[i * N + j]));
    for i in 0..N {
        for k in 0..i {
            let d: f64 = (0..N).map(|j| q[i][j] * q[k][j]).sum();
            for j in 0..N {
                q[i][j] -= d * q[k][j];
            }
        }
        let n = (0..N).map(|j| q[i][j] * q[i][j]).sum::<f64>().sqrt();
        for j in 0..N {
            q[i][j] /= n;
        }
    }
    let vals: [f64; N] = core::array::from_fn(|k| match k {
        0 => 1.0,
        1 => 1.0 + gap,
        _ => k as f64,
    });
    core::array::from_fn(|i| {
        core::array::from_fn(|j| (0..N).map(|k| q[k][i] * vals[k] * q[k][j]).sum())
    })
}

fn check_lu<const N: usize>(v: &[f64], deltas: [f64; 4]) {
    let mats: [[[f64; N]; N]; 4] =
        core::array::from_fn(|l| near_singular(&v[l..], deltas[l], l % 2 == 1));
    let a = lanes(&mats);
    let x_true: [f64; N] = core::array::from_fn(|i| v[60 + i]);
    let bs: [[f64; N]; 4] = core::array::from_fn(|l| matvec(&mats[l], &x_true));
    let b: [L; N] = core::array::from_fn(|i| L::new(core::array::from_fn(|l| bs[l][i])));
    let (d, inv, x) = (det(&a), inverse(&a), lu_solve(&lu(&a), &b));
    for l in 0..4 {
        let m = &mats[l];
        assert!(
            same_bits(d.v[l], det(m)),
            "det lane {l}: {} vs {}",
            d.v[l],
            det(m)
        );
        let si = inverse(m);
        let sx = lu_solve(&lu(m), &bs[l]);
        for i in 0..N {
            assert!(same_bits(x[i].v[l], sx[i]), "solve lane {l}");
            for j in 0..N {
                assert!(same_bits(inv[i][j].v[l], si[i][j]), "inverse lane {l}");
            }
        }
        // Accuracy of the (shared) algorithm, scaled by the condition number.
        let kappa = inf_norm(m) * inf_norm(&si);
        if kappa.is_finite() && kappa < 1e15 {
            let err = (0..N)
                .map(|i| (sx[i] - x_true[i]).abs())
                .fold(0.0, f64::max);
            let size = x_true.iter().fold(0.0f64, |s, x| s.max(x.abs()));
            assert!(
                err <= 64.0 * N as f64 * EPS * kappa * size,
                "n={N} κ={kappa:e}: forward error {err:e}"
            );
        }
    }
}

fn check_eigh<const N: usize>(v: &[f64], gaps: [f64; 4]) {
    let mats: [[[f64; N]; N]; 4] = core::array::from_fn(|l| clustered(&v[l..], gaps[l]));
    let (vals, vecs) = eigh(&lanes(&mats), 12);
    for l in 0..4 {
        let (sv, svec) = eigh(&mats[l], 12);
        let scale = inf_norm(&mats[l]);
        for k in 0..N {
            let dv = (vals[k].v[l] - sv[k]).abs();
            assert!(
                dv <= 16.0 * N as f64 * EPS * scale,
                "eigenvalue lane {l}: {dv:e}"
            );
            // The gap from λ_k to the nearest other eigenvalue.
            let gap = (0..N)
                .filter(|&j| j != k)
                .map(|j| (sv[j] - sv[k]).abs())
                .fold(f64::INFINITY, f64::min);
            for i in 0..N {
                let dx = (vecs[k][i].v[l] - svec[k][i]).abs();
                assert!(
                    dx <= 16.0 * N as f64 * EPS * scale / gap.min(1.0),
                    "eigenvector lane {l} (gap {gap:e}): {dx:e}, same sign expected"
                );
            }
        }
    }
}

fn check_svd<const N: usize>(v: &[f64], deltas: [f64; 4]) {
    let mats: [[[f64; N]; N]; 4] =
        core::array::from_fn(|l| near_singular(&v[l..], deltas[l], false));
    let (u, s, w) = svd(&lanes(&mats), 14);
    for l in 0..4 {
        let (su, ss, sw) = svd(&mats[l], 14);
        let scale = inf_norm(&mats[l]).max(1e-300);
        for k in 0..N {
            let ds = (s[k].v[l] - ss[k]).abs();
            assert!(ds <= 16.0 * N as f64 * EPS * scale, "σ lane {l}: {ds:e}");
            if k > 0 {
                assert!(s[k - 1].v[l] >= s[k].v[l], "descending, lane {l}");
            }
            let gap = (0..N)
                .filter(|&j| j != k)
                .map(|j| (ss[j] - ss[k]).abs())
                .fold(f64::INFINITY, f64::min);
            // A singular value within rounding of zero has an arbitrary u; skip its vectors.
            if ss[k] <= 64.0 * N as f64 * EPS * scale {
                continue;
            }
            for i in 0..N {
                let tol = 16.0 * N as f64 * EPS * scale / gap.min(ss[k]).min(1.0);
                assert!((u[k][i].v[l] - su[k][i]).abs() <= tol, "u lane {l}");
                assert!((w[k][i].v[l] - sw[k][i]).abs() <= tol, "v lane {l}");
            }
        }
    }
}

fn exps() -> impl Strategy<Value = [f64; 4]> {
    prop::array::uniform4(-14.0f64..0.0).prop_map(|e| e.map(|x| 10f64.powf(x)))
}

proptest! {
    #[test]
    fn lu_is_bit_identical_on_nearly_singular_maps(
        v in prop::collection::vec(-1.0f64..1.0, 72), d in exps(),
    ) {
        check_lu::<2>(&v, d); check_lu::<3>(&v, d); check_lu::<4>(&v, d);
        check_lu::<5>(&v, d); check_lu::<8>(&v, d);
    }

    #[test]
    fn eigh_agrees_on_clustered_spectra(
        v in prop::collection::vec(-1.0f64..1.0, 72), g in exps(),
    ) {
        check_eigh::<2>(&v, g); check_eigh::<3>(&v, g); check_eigh::<4>(&v, g);
        check_eigh::<6>(&v, g);
    }

    #[test]
    fn svd_agrees_on_nearly_singular_maps(
        v in prop::collection::vec(-1.0f64..1.0, 72), d in exps(),
    ) {
        check_svd::<2>(&v, d); check_svd::<3>(&v, d); check_svd::<4>(&v, d);
        check_svd::<6>(&v, d);
    }
}

/// Ties: equal magnitudes in the pivot column pick the first row in both paths, so a matrix
/// that needs no pivoting by value is not reordered in either.
#[test]
fn pivot_ties_pick_the_first_row() {
    // Every pivot column has equal magnitudes: |1| = |-1| = |1|, then |1| = |-1|.
    let m = [[1.0, 0.0, 0.0], [-1.0, 1.0, 0.0], [1.0, -1.0, 1.0]];
    let f = lu(&m);
    let fl = lu(&lanes(&[m; 4]));
    for i in 0..3 {
        assert_eq!(f.perm[i], i as f64);
        assert_eq!(fl.perm[i].v, [i as f64; 4]);
    }
}
