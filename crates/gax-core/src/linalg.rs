//! Small dense linear algebra, generic over the coefficient type and the size.
//!
//! Extensor maps and forms are small (at most 32 × 32 for the algebras shipped) and their
//! size is known at compile time, so the algorithms here are written over
//! [`SquareArr`], implemented for `[[T; N]; N]`: `N` is only a loop bound, which LLVM unrolls,
//! and no `generic_const_exprs` is needed. Every algorithm is branch free, using
//! [`Real::select_lt`] where a scalar implementation would branch, so the same code runs
//! lane-wise on SIMD coefficients, one independent problem per lane.
//!
//! * [`lu`]: LU factorization with partial pivoting; [`lu_solve`], [`inverse`], [`det`].
//! * [`cholesky`]: `A = L Lᵀ` for symmetric positive definite `A`; [`cholesky_solve`].
//! * [`eigh`]: cyclic Jacobi eigen-decomposition of a symmetric matrix.
//! * [`eigh_generalized`]: `A x = λ B x` with `B` positive definite, by Cholesky reduction.
//! * [`svd`]: one-sided (Hestenes) Jacobi singular value decomposition.
//!
//! Jacobi methods are chosen over QR for their accuracy on small symmetric problems (Demmel &
//! Veselić) and because a fixed number of sweeps makes them branch free.
//!
//! ```
//! use gax_core::linalg::{eigh, inverse, matmul};
//! let a: [[f64; 2]; 2] = [[2.0, 1.0], [1.0, 2.0]];
//! let (values, _vectors) = eigh(&a, 8);
//! assert!((values[0] * values[1] - 3.0).abs() < 1e-12); // the determinant
//! let i = matmul(&a, &inverse(&a));
//! assert!((i[0][0] - 1.0).abs() < 1e-12 && i[0][1].abs() < 1e-12);
//! ```

// Matrix code indexes rows and columns explicitly; iterators would obscure the algorithms.
#![allow(clippy::needless_range_loop)]

use crate::coef::Real;
use core::ops::{Index, IndexMut};

/// A square matrix stored as an array of rows: `m[i][j]` is row `i`, column `j`.
#[diagnostic::on_unimplemented(
    message = "the map is not square: `{Self}`",
    note = "inverse, det, solve and svd need a map between kinds with the same number of coefficients"
)]
pub trait SquareArr<T: Real>: Copy + Index<usize, Output = Self::Vector> + IndexMut<usize> {
    /// Dimension.
    const N: usize;
    /// A vector of length `N`.
    type Vector: Copy + Index<usize, Output = T> + IndexMut<usize>;
    /// The zero matrix.
    fn zero() -> Self;
    /// The zero vector.
    fn zero_vector() -> Self::Vector;
}

impl<T: Real, const N: usize> SquareArr<T> for [[T; N]; N] {
    const N: usize = N;
    type Vector = [T; N];
    #[inline(always)]
    fn zero() -> Self {
        [[T::zero(); N]; N]
    }
    #[inline(always)]
    fn zero_vector() -> [T; N] {
        [T::zero(); N]
    }
}

/// The identity matrix.
#[inline]
pub fn identity<T: Real, M: SquareArr<T>>() -> M {
    let mut m = M::zero();
    for i in 0..M::N {
        m[i][i] = T::one();
    }
    m
}

/// Transpose.
#[inline]
pub fn transpose<T: Real, M: SquareArr<T>>(a: &M) -> M {
    let mut t = M::zero();
    for i in 0..M::N {
        for j in 0..M::N {
            t[j][i] = a[i][j];
        }
    }
    t
}

/// Matrix product.
#[inline]
pub fn matmul<T: Real, M: SquareArr<T>>(a: &M, b: &M) -> M {
    let mut c = M::zero();
    for i in 0..M::N {
        for j in 0..M::N {
            let mut s = a[i][0] * b[0][j];
            for k in 1..M::N {
                s = s + a[i][k] * b[k][j];
            }
            c[i][j] = s;
        }
    }
    c
}

/// Matrix times vector.
#[inline]
pub fn matvec<T: Real, M: SquareArr<T>>(a: &M, x: &M::Vector) -> M::Vector {
    let mut y = M::zero_vector();
    for i in 0..M::N {
        let mut s = a[i][0] * x[0];
        for k in 1..M::N {
            s = s + a[i][k] * x[k];
        }
        y[i] = s;
    }
    y
}

/// An LU factorization `P A = L U`, stored compactly.
#[derive(Clone, Copy, Debug)]
pub struct Lu<M, V> {
    /// `L` (unit lower triangle, below the diagonal) and `U` (on and above the diagonal).
    pub lu: M,
    /// Row permutation: row `i` of `P A` is row `perm[i]` of `A` (stored as coefficients so
    /// that it can differ per SIMD lane).
    pub perm: V,
    /// `±1`: the sign of the permutation.
    pub sign: V,
}

/// LU factorization with partial pivoting, branch free.
///
/// Pivot choice and row swaps are done with lane-wise selects, so each SIMD lane pivots on
/// its own data. A singular matrix gives zero (or tiny) pivots and non-finite results
/// downstream, as for the scalar algorithm.
#[inline]
pub fn lu<T: Real, M: SquareArr<T>>(a: &M) -> Lu<M, M::Vector> {
    let n = M::N;
    let mut m = *a;
    let mut perm = M::zero_vector();
    for i in 0..n {
        perm[i] = T::from_i64(i as i64);
    }
    let mut sign = M::zero_vector();
    sign[0] = T::one();
    for k in 0..n {
        // Find the pivot row lane-wise: the largest |m[r][k]| for r >= k.
        let mut best = m[k][k].abs();
        let mut best_row = T::from_i64(k as i64);
        for r in k + 1..n {
            let v = m[r][k].abs();
            best_row = T::select_lt(best, v, T::from_i64(r as i64), best_row);
            best = best.max(v);
        }
        // Swap row k with the pivot row, lane-wise; skipped when no lane needs a swap.
        let half = T::from_f64(0.5);
        let no_swap = T::all_lt((best_row - T::from_i64(k as i64)).abs(), half);
        for r in (k + 1..n).filter(|_| !no_swap) {
            let rr = T::from_i64(r as i64);
            // is_pivot: best_row == r, as a select on (best_row < r+0.5) and (r-0.5 < best_row).
            let lo = rr - half;
            let hi = rr + half;
            let sel = |x: T, y: T| T::select_lt(best_row, hi, T::select_lt(lo, best_row, x, y), y);
            for j in 0..n {
                let (a_k, a_r) = (m[k][j], m[r][j]);
                m[k][j] = sel(a_r, a_k);
                m[r][j] = sel(a_k, a_r);
            }
            let (p_k, p_r) = (perm[k], perm[r]);
            perm[k] = sel(p_r, p_k);
            perm[r] = sel(p_k, p_r);
            sign[0] = sel(-sign[0], sign[0]);
        }
        let inv = m[k][k].recip();
        for r in k + 1..n {
            let f = m[r][k] * inv;
            m[r][k] = f;
            for j in k + 1..n {
                m[r][j] = m[r][j] - f * m[k][j];
            }
        }
    }
    Lu { lu: m, perm, sign }
}

/// Solve `A x = b` given the LU factorization of `A`.
#[inline]
pub fn lu_solve<T: Real, M: SquareArr<T>>(f: &Lu<M, M::Vector>, b: &M::Vector) -> M::Vector {
    let n = M::N;
    // Apply the permutation lane-wise: pb[i] = b[perm[i]]; skipped when it is the identity in
    // every lane.
    let half = T::from_f64(0.5);
    let mut displaced = T::zero();
    for i in 0..n {
        displaced = displaced + (f.perm[i] - T::from_i64(i as i64)).abs();
    }
    let mut pb = *b;
    if !T::all_lt(displaced, half) {
        for i in 0..n {
            let mut v = b[0];
            for r in 1..n {
                let rr = T::from_i64(r as i64);
                v = T::select_lt(rr - half, f.perm[i], b[r], v);
            }
            pb[i] = v;
        }
    }
    // Forward substitution with unit L.
    let mut y = pb;
    for i in 0..n {
        for k in 0..i {
            y[i] = y[i] - f.lu[i][k] * y[k];
        }
    }
    // Back substitution with U.
    let mut x = y;
    for i in (0..n).rev() {
        for k in i + 1..n {
            x[i] = x[i] - f.lu[i][k] * x[k];
        }
        x[i] = x[i] / f.lu[i][i];
    }
    x
}

/// Determinant.
#[inline]
pub fn det<T: Real, M: SquareArr<T>>(a: &M) -> T {
    let f = lu(a);
    let mut d = f.sign[0];
    for i in 0..M::N {
        d = d * f.lu[i][i];
    }
    d
}

/// Inverse, by LU factorization.
#[inline]
pub fn inverse<T: Real, M: SquareArr<T>>(a: &M) -> M {
    let f = lu(a);
    let mut inv = M::zero();
    for j in 0..M::N {
        let mut e = M::zero_vector();
        e[j] = T::one();
        let col = lu_solve(&f, &e);
        for i in 0..M::N {
            inv[i][j] = col[i];
        }
    }
    inv
}

/// Cholesky factorization `A = L Lᵀ` of a symmetric positive definite matrix (lower `L`).
///
/// Only the lower triangle of `a` is read. A matrix that is not positive definite produces
/// NaN in `L` (the square root of a negative number).
#[inline]
pub fn cholesky<T: Real, M: SquareArr<T>>(a: &M) -> M {
    let n = M::N;
    let mut l = M::zero();
    for j in 0..n {
        let mut d = a[j][j];
        for k in 0..j {
            d = d - l[j][k] * l[j][k];
        }
        let ljj = d.sqrt();
        l[j][j] = ljj;
        let inv = ljj.recip();
        for i in j + 1..n {
            let mut s = a[i][j];
            for k in 0..j {
                s = s - l[i][k] * l[j][k];
            }
            l[i][j] = s * inv;
        }
    }
    l
}

/// Solve `L y = b` for lower triangular `L`.
#[inline]
pub fn solve_lower<T: Real, M: SquareArr<T>>(l: &M, b: &M::Vector) -> M::Vector {
    let mut y = *b;
    for i in 0..M::N {
        for k in 0..i {
            y[i] = y[i] - l[i][k] * y[k];
        }
        y[i] = y[i] / l[i][i];
    }
    y
}

/// Solve `Lᵀ x = y` for lower triangular `L`.
#[inline]
pub fn solve_lower_transposed<T: Real, M: SquareArr<T>>(l: &M, y: &M::Vector) -> M::Vector {
    let mut x = *y;
    for i in (0..M::N).rev() {
        for k in i + 1..M::N {
            x[i] = x[i] - l[k][i] * x[k];
        }
        x[i] = x[i] / l[i][i];
    }
    x
}

/// Solve `A x = b` with `A = L Lᵀ` given by [`cholesky`].
#[inline]
pub fn cholesky_solve<T: Real, M: SquareArr<T>>(l: &M, b: &M::Vector) -> M::Vector {
    solve_lower_transposed(l, &solve_lower(l, b))
}

/// The Jacobi rotation `(c, s)` that annihilates `a[p][q]` of a symmetric 2×2 block
/// `[[app, apq], [apq, aqq]]`, computed without branches. When `apq = 0` it is the identity.
#[inline(always)]
fn jacobi_rotation<T: Real>(app: T, aqq: T, apq: T) -> (T, T) {
    // theta = (aqq - app) / (2 apq); t = sign(theta) / (|theta| + sqrt(theta² + 1)).
    // Written as t = 2 apq sign(d) / (|d| + sqrt(d² + 4 apq²)) with d = aqq - app, which is
    // finite for apq = 0 (t = 0) unless d = apq = 0, handled by the select below.
    let zero = T::zero();
    let two = T::from_i64(2);
    let d = aqq - app;
    let root = (d * d + two * two * apq * apq).sqrt();
    let denom = d.abs() + root;
    let sign = T::select_lt(d, zero, -T::one(), T::one());
    let tiny = T::select_lt(denom, T::epsilon() * T::epsilon(), T::one(), zero);
    // When denom is ~0 the block is already diagonal: use t = 0.
    let safe = denom + tiny;
    let t = T::select_lt(zero, tiny, zero, two * apq * sign / safe);
    let c = (T::one() + t * t).sqrt().recip();
    (c, t * c)
}

/// Eigen-decomposition of a symmetric matrix by cyclic Jacobi: `A = V diag(λ) Vᵀ`.
///
/// Returns the eigenvalues (unsorted) and the eigenvectors as the *rows* of the second
/// matrix: `vectors[k]` is the eigenvector of `values[k]`. Runs a fixed number of sweeps
/// (`sweeps`), enough for full precision at these sizes (quadratic convergence); 8 is a good
/// default for `N ≤ 8`, 12 for `N ≤ 32`.
#[inline]
pub fn eigh<T: Real, M: SquareArr<T>>(a: &M, sweeps: usize) -> (M::Vector, M) {
    let n = M::N;
    let mut a = *a;
    let mut v: M = identity();
    // Converged when the off-diagonal part is negligible against the diagonal, in every lane.
    let tol = T::epsilon() * T::epsilon();
    for _ in 0..sweeps {
        let mut off = T::zero();
        let mut diag = T::zero();
        for p in 0..n {
            diag = diag + a[p][p] * a[p][p];
            for q in p + 1..n {
                off = off + a[p][q] * a[p][q];
            }
        }
        if T::all_lt(off, tol * diag) {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                let (c, s) = jacobi_rotation(a[p][p], a[q][q], a[p][q]);
                // A <- Jᵀ A J on rows/columns p, q.
                for k in 0..n {
                    let (akp, akq) = (a[k][p], a[k][q]);
                    a[k][p] = c * akp - s * akq;
                    a[k][q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let (apk, aqk) = (a[p][k], a[q][k]);
                    a[p][k] = c * apk - s * aqk;
                    a[q][k] = s * apk + c * aqk;
                }
                // Accumulate V <- V J (stored transposed: rows are eigenvectors).
                for k in 0..n {
                    let (vpk, vqk) = (v[p][k], v[q][k]);
                    v[p][k] = c * vpk - s * vqk;
                    v[q][k] = s * vpk + c * vqk;
                }
            }
        }
    }
    let mut values = M::zero_vector();
    for i in 0..n {
        values[i] = a[i][i];
    }
    (values, v)
}

/// Sort eigenpairs (or singular pairs) by ascending value. Branch free (a sorting network of
/// compare-exchanges), so it also sorts per SIMD lane.
#[inline]
pub fn sort_pairs<T: Real, M: SquareArr<T>>(values: &mut M::Vector, vectors: &mut M) {
    let n = M::N;
    for i in 0..n {
        for j in 0..n - 1 - i {
            let (a, b) = (values[j], values[j + 1]);
            // swap if b < a
            values[j] = T::select_lt(b, a, b, a);
            values[j + 1] = T::select_lt(b, a, a, b);
            for k in 0..n {
                let (x, y) = (vectors[j][k], vectors[j + 1][k]);
                vectors[j][k] = T::select_lt(b, a, y, x);
                vectors[j + 1][k] = T::select_lt(b, a, x, y);
            }
        }
    }
}

/// Generalized symmetric-definite eigenproblem `A x = λ B x`, with `B` positive definite.
///
/// Reduces to a standard problem with the Cholesky factor `B = L Lᵀ`:
/// `C = L⁻¹ A L⁻ᵀ`, `C y = λ y`, `x = L⁻ᵀ y`. The eigenvectors are `B`-orthonormal
/// (`xᵢᵀ B xⱼ = δᵢⱼ`) and returned as rows, sorted by ascending eigenvalue.
#[inline]
pub fn eigh_generalized<T: Real, M: SquareArr<T>>(a: &M, b: &M, sweeps: usize) -> (M::Vector, M) {
    let n = M::N;
    let l = cholesky(b);
    // C = L⁻¹ A L⁻ᵀ: first W = L⁻¹ A (column by column), then C = L⁻¹ Wᵀ (W symmetric-ish).
    let mut w = M::zero();
    for j in 0..n {
        let mut col = M::zero_vector();
        for i in 0..n {
            col[i] = a[i][j];
        }
        let y = solve_lower(&l, &col);
        for i in 0..n {
            w[i][j] = y[i];
        }
    }
    let mut c = M::zero();
    for i in 0..n {
        let mut row = M::zero_vector();
        for j in 0..n {
            row[j] = w[i][j];
        }
        let y = solve_lower(&l, &row);
        for j in 0..n {
            c[i][j] = y[j];
        }
    }
    let (mut values, ys) = eigh(&c, sweeps);
    let mut xs = M::zero();
    for k in 0..n {
        let x = solve_lower_transposed(&l, &ys[k]);
        xs[k] = x;
    }
    sort_pairs(&mut values, &mut xs);
    (values, xs)
}

/// Singular value decomposition `A = U diag(σ) Vᵀ` by one-sided (Hestenes) Jacobi.
///
/// Returns `(u, sigma, v)` with the singular vectors as *rows*: `A vᵢ = σᵢ uᵢ`. Singular
/// values are non-negative and sorted in descending order. For a zero singular value the
/// corresponding `u` row is zero.
#[inline]
pub fn svd<T: Real, M: SquareArr<T>>(a: &M, sweeps: usize) -> (M, M::Vector, M) {
    let n = M::N;
    // Work on the columns of A: orthogonalize them pairwise with right rotations.
    let mut u = transpose(a); // rows of u = columns of A
    let mut v: M = identity(); // rows of v accumulate the right rotations
    let tol = T::epsilon() * T::epsilon();
    for _ in 0..sweeps {
        // Converged when every pair of columns is orthogonal to working precision, in every lane.
        let mut worst = T::zero();
        for p in 0..n {
            for q in p + 1..n {
                let mut alpha = u[p][0] * u[p][0];
                let mut beta = u[q][0] * u[q][0];
                let mut gamma = u[p][0] * u[q][0];
                for k in 1..n {
                    alpha = alpha + u[p][k] * u[p][k];
                    beta = beta + u[q][k] * u[q][k];
                    gamma = gamma + u[p][k] * u[q][k];
                }
                // gamma² / (alpha beta), guarded against zero columns
                let r = gamma * gamma - tol * alpha * beta;
                worst = worst.max(r);
            }
        }
        if T::all_lt(worst, T::epsilon() * T::epsilon() * T::epsilon()) {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                let mut alpha = u[p][0] * u[p][0];
                let mut beta = u[q][0] * u[q][0];
                let mut gamma = u[p][0] * u[q][0];
                for k in 1..n {
                    alpha = alpha + u[p][k] * u[p][k];
                    beta = beta + u[q][k] * u[q][k];
                    gamma = gamma + u[p][k] * u[q][k];
                }
                let (c, s) = jacobi_rotation(alpha, beta, gamma);
                for k in 0..n {
                    let (x, y) = (u[p][k], u[q][k]);
                    u[p][k] = c * x - s * y;
                    u[q][k] = s * x + c * y;
                    let (x, y) = (v[p][k], v[q][k]);
                    v[p][k] = c * x - s * y;
                    v[q][k] = s * x + c * y;
                }
            }
        }
    }
    let mut sigma = M::zero_vector();
    for i in 0..n {
        let mut s = u[i][0] * u[i][0];
        for k in 1..n {
            s = s + u[i][k] * u[i][k];
        }
        let norm = s.sqrt();
        sigma[i] = norm;
        let inv = T::select_lt(T::zero(), norm, norm.recip(), T::zero());
        for k in 0..n {
            u[i][k] = u[i][k] * inv;
        }
    }
    // Sort descending: sort ascending on -sigma, permuting u and v together.
    let mut neg = M::zero_vector();
    for i in 0..n {
        neg[i] = -sigma[i];
    }
    let mut uv_sort_u = u;
    let mut neg_u = neg;
    sort_pairs(&mut neg_u, &mut uv_sort_u);
    let mut vv = v;
    sort_pairs(&mut neg, &mut vv);
    for i in 0..n {
        sigma[i] = -neg[i];
    }
    (uv_sort_u, sigma, vv)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-10
    }

    #[test]
    fn lu_inverse_det() {
        let a = [[4.0, 3.0, 2.0], [2.0, 1.0, 3.0], [3.0, 2.0, 1.0]];
        let inv = inverse(&a);
        let id = matmul(&a, &inv);
        for i in 0..3 {
            for j in 0..3 {
                assert!(close(id[i][j], if i == j { 1.0 } else { 0.0 }));
            }
        }
        // det by cofactors: 4(1-6) - 3(2-9) + 2(4-3) = -20 + 21 + 2 = 3
        assert!(close(det(&a), 3.0));
        // pivoting is needed here
        let b = [[0.0, 1.0], [1.0, 0.0]];
        assert!(close(det(&b), -1.0));
        let x = lu_solve(&lu(&b), &[2.0, 3.0]);
        assert!(close(x[0], 3.0) && close(x[1], 2.0));
    }

    #[test]
    fn jacobi_eigh_reconstructs() {
        let a = [[4.0, 1.0, 0.5], [1.0, 3.0, 0.2], [0.5, 0.2, 1.0]];
        let (vals, vecs) = eigh(&a, 10);
        for k in 0..3 {
            let av = matvec(&a, &vecs[k]);
            for i in 0..3 {
                assert!(close(av[i], vals[k] * vecs[k][i]));
            }
        }
    }

    #[test]
    fn svd_reconstructs() {
        let a = [[1.0, 2.0, 0.0], [0.0, 1.0, 3.0], [1.0, 0.0, 1.0]];
        let (u, s, v) = svd(&a, 12);
        for k in 0..3 {
            let av = matvec(&a, &v[k]);
            for i in 0..3 {
                assert!(close(av[i], s[k] * u[k][i]));
            }
        }
        assert!(s[0] >= s[1] && s[1] >= s[2]);
    }
}
