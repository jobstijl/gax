//! Property tests of the small dense solvers.
//!
//! Tolerances follow backward-error bounds. LU with partial pivoting and Cholesky are
//! backward stable: `‖A x − b‖ ≤ c · n · ε · ‖A‖ · ‖x‖` with a modest `c` (the growth factor is
//! small in practice for these well-conditioned random matrices). Jacobi eigen and SVD methods
//! have residuals `‖A v − λ v‖ ≤ c · n² · ε · ‖A‖` after convergence. With `n ≤ 8` and
//! `ε = 2.2e-16` these are about 1e-14 relative; we assert 1e-11, which leaves a margin of about
//! 1000 for growth and accumulated rounding while still catching any real error (which is O(1)).
#![allow(clippy::needless_range_loop)]

use gax_core::linalg::*;
use proptest::prelude::*;

const TOL: f64 = 1e-11;

fn norm<const N: usize>(a: &[[f64; N]; N]) -> f64 {
    a.iter()
        .flatten()
        .fold(0.0f64, |m, x| m.max(x.abs()))
        .max(1e-300)
}

fn vmax(x: &[f64]) -> f64 {
    x.iter().fold(0.0f64, |m, v| m.max(v.abs()))
}

fn mat<const N: usize>(v: &[f64]) -> [[f64; N]; N] {
    core::array::from_fn(|i| core::array::from_fn(|j| v[i * N + j]))
}

fn spd<const N: usize>(v: &[f64]) -> [[f64; N]; N] {
    // A Aᵀ + N I is symmetric positive definite and well conditioned.
    let a: [[f64; N]; N] = mat(v);
    let mut s = matmul(&a, &transpose(&a));
    for i in 0..N {
        s[i][i] += N as f64;
    }
    s
}

macro_rules! for_sizes {
    ($name:ident, $body:ident) => {
        proptest! {
            #[test]
            fn $name(v in prop::collection::vec(-1.0f64..1.0, 64), b in prop::collection::vec(-1.0f64..1.0, 8)) {
                $body::<1>(&v, &b); $body::<2>(&v, &b); $body::<3>(&v, &b); $body::<4>(&v, &b);
                $body::<5>(&v, &b); $body::<6>(&v, &b); $body::<8>(&v, &b);
            }
        }
    };
}

fn check_lu<const N: usize>(v: &[f64], b: &[f64]) {
    let mut a: [[f64; N]; N] = spd::<N>(v);
    a[0][N - 1] += 0.5; // break the symmetry
    let b: [f64; N] = core::array::from_fn(|i| b[i]);
    let x = lu_solve(&lu(&a), &b);
    let r = matvec(&a, &x);
    for i in 0..N {
        assert!(
            (r[i] - b[i]).abs() <= TOL * norm(&a) * vmax(&x).max(1.0),
            "LU residual n={N}"
        );
    }
    let inv = inverse(&a);
    let id = matmul(&a, &inv);
    for i in 0..N {
        for j in 0..N {
            let want = if i == j { 1.0 } else { 0.0 };
            assert!(
                (id[i][j] - want).abs() <= TOL * norm(&a) * norm(&inv),
                "A A⁻¹ n={N}"
            );
        }
    }
}

fn check_cholesky<const N: usize>(v: &[f64], b: &[f64]) {
    let a: [[f64; N]; N] = spd::<N>(v);
    let l = cholesky(&a);
    let llt = matmul(&l, &transpose(&l));
    for i in 0..N {
        for j in 0..N {
            assert!((llt[i][j] - a[i][j]).abs() <= TOL * norm(&a), "L Lᵀ n={N}");
        }
    }
    let b: [f64; N] = core::array::from_fn(|i| b[i]);
    let x = cholesky_solve(&l, &b);
    let r = matvec(&a, &x);
    for i in 0..N {
        assert!(
            (r[i] - b[i]).abs() <= TOL * norm(&a) * vmax(&x).max(1.0),
            "Cholesky solve n={N}"
        );
    }
}

fn check_eigh<const N: usize>(v: &[f64], _b: &[f64]) {
    let a: [[f64; N]; N] = spd::<N>(v);
    let (vals, vecs) = eigh(&a, 12);
    for k in 0..N {
        let av = matvec(&a, &vecs[k]);
        for i in 0..N {
            assert!(
                (av[i] - vals[k] * vecs[k][i]).abs() <= TOL * norm(&a),
                "A v = λ v n={N}"
            );
        }
        for l in 0..N {
            let d: f64 = (0..N).map(|i| vecs[k][i] * vecs[l][i]).sum();
            assert!(
                (d - if k == l { 1.0 } else { 0.0 }).abs() <= TOL,
                "orthonormal n={N}"
            );
        }
    }
}

fn check_generalized<const N: usize>(v: &[f64], b: &[f64]) {
    let a: [[f64; N]; N] = spd::<N>(v);
    let mut w = v.to_vec();
    w.rotate_left(7);
    w[0] += b[0];
    let m: [[f64; N]; N] = spd::<N>(&w);
    let (vals, xs) = eigh_generalized(&a, &m, 12);
    for k in 0..N {
        let ax = matvec(&a, &xs[k]);
        let mx = matvec(&m, &xs[k]);
        for i in 0..N {
            assert!(
                (ax[i] - vals[k] * mx[i]).abs() <= TOL * norm(&a) * norm(&m),
                "A x = λ B x n={N}"
            );
        }
        if k > 0 {
            assert!(vals[k - 1] <= vals[k] + TOL, "sorted");
        }
    }
}

fn check_svd<const N: usize>(v: &[f64], _b: &[f64]) {
    let a: [[f64; N]; N] = mat(v);
    let (u, s, vv) = svd(&a, 14);
    for k in 0..N {
        let av = matvec(&a, &vv[k]);
        for i in 0..N {
            assert!(
                (av[i] - s[k] * u[k][i]).abs() <= TOL * norm(&a).max(1.0),
                "A v = σ u n={N}"
            );
        }
        assert!(s[k] >= -TOL);
        if k > 0 {
            assert!(s[k - 1] + TOL >= s[k], "descending");
        }
    }
}

for_sizes!(lu_solves_and_inverts, check_lu);
for_sizes!(cholesky_reconstructs, check_cholesky);
for_sizes!(jacobi_eigh_is_an_eigenbasis, check_eigh);
for_sizes!(generalized_eigh_solves_the_pencil, check_generalized);
for_sizes!(svd_reconstructs, check_svd);

#[cfg(feature = "wide")]
#[test]
fn lanes_match_scalar() {
    use gax_core::simd::wide::f64x4;
    let mats: [[[f64; 3]; 3]; 4] = [
        [[4.0, 1.0, 0.5], [1.0, 3.0, 0.2], [0.5, 0.2, 1.0]],
        [[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 5.0]],
        [[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 3.0]],
        [[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]],
    ];
    let lanes: [[f64x4; 3]; 3] = core::array::from_fn(|i| {
        core::array::from_fn(|j| f64x4::new(core::array::from_fn(|l| mats[l][i][j])))
    });
    let d = det(&lanes).to_array();
    let inv = inverse(&lanes);
    let (vals, _) = eigh(&lanes, 12);
    for l in 0..4 {
        assert!((d[l] - det(&mats[l])).abs() < 1e-12, "det lane {l}");
        let si = inverse(&mats[l]);
        let (sv, _) = eigh(&mats[l], 12);
        for i in 0..3 {
            assert!(
                (vals[i].to_array()[l] - sv[i]).abs() < 1e-12,
                "eigh lane {l}"
            );
            for j in 0..3 {
                assert!(
                    (inv[i][j].to_array()[l] - si[i][j]).abs() < 1e-10,
                    "inverse lane {l}"
                );
            }
        }
    }
}
