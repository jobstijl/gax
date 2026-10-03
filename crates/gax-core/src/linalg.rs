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
//! * [`orthogonalize`] and [`pinv_apply`]: least squares and the pseudo-inverse of a matrix of
//!   any shape, by the same one-sided Jacobi on its columns ([`Column`]).
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
    T::vectorize(
        #[inline(always)]
        || {
            let n = M::N;
            let mut m = *a;
            let mut perm = M::zero_vector();
            for i in 0..n {
                perm[i] = T::from_i64(i as i64);
            }
            let mut sign = M::zero_vector();
            sign[0] = T::one();
            for k in 0..n {
                if T::SCALAR {
                    // One number: pick the pivot row and swap rows with branches.
                    let (mut p, mut best) = (k, m[k][k].abs());
                    for r in k + 1..n {
                        let v = m[r][k].abs();
                        if T::all_lt(best, v) {
                            (p, best) = (r, v);
                        }
                    }
                    if p != k {
                        for j in 0..n {
                            let t = m[k][j];
                            m[k][j] = m[p][j];
                            m[p][j] = t;
                        }
                        let t = perm[k];
                        perm[k] = perm[p];
                        perm[p] = t;
                        sign[0] = -sign[0];
                    }
                    let inv = m[k][k].recip();
                    for r in k + 1..n {
                        let f = m[r][k] * inv;
                        m[r][k] = f;
                        for j in k + 1..n {
                            m[r][j] = m[r][j] - f * m[k][j];
                        }
                    }
                    continue;
                }
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
                    let sel = |x: T, y: T| {
                        T::select_lt(best_row, hi, T::select_lt(lo, best_row, x, y), y)
                    };
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
        },
    )
}

/// Solve `A x = b` given the LU factorization of `A`.
#[inline]
pub fn lu_solve<T: Real, M: SquareArr<T>>(f: &Lu<M, M::Vector>, b: &M::Vector) -> M::Vector {
    T::vectorize(
        #[inline(always)]
        || {
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
        },
    )
}

/// Determinant.
#[inline]
pub fn det<T: Real, M: SquareArr<T>>(a: &M) -> T {
    T::vectorize(
        #[inline(always)]
        || {
            let f = lu(a);
            let mut d = f.sign[0];
            for i in 0..M::N {
                d = d * f.lu[i][i];
            }
            d
        },
    )
}

/// Inverse, by LU factorization.
///
/// The `n` columns share the factorization and the reciprocals of the pivots, so the only
/// divisions are those `n` reciprocals.
#[inline]
pub fn inverse<T: Real, M: SquareArr<T>>(a: &M) -> M {
    T::vectorize(
        #[inline(always)]
        || {
            let n = M::N;
            let f = lu(a);
            let mut rdiag = M::zero_vector();
            for i in 0..n {
                rdiag[i] = f.lu[i][i].recip();
            }
            let half = T::from_f64(0.5);
            let mut inv = M::zero();
            for j in 0..n {
                // P e_j: y[i] = 1 where perm[i] == j (lane-wise for SIMD lanes).
                let mut y = M::zero_vector();
                let jj = T::from_i64(j as i64);
                for i in 0..n {
                    y[i] = T::select_lt(
                        f.perm[i],
                        jj + half,
                        T::select_lt(jj - half, f.perm[i], T::one(), T::zero()),
                        T::zero(),
                    );
                }
                // Forward substitution with unit L, then back substitution with U.
                for i in 0..n {
                    for k in 0..i {
                        y[i] = y[i] - f.lu[i][k] * y[k];
                    }
                }
                for i in (0..n).rev() {
                    for k in i + 1..n {
                        y[i] = y[i] - f.lu[i][k] * y[k];
                    }
                    y[i] = y[i] * rdiag[i];
                }
                for i in 0..n {
                    inv[i][j] = y[i];
                }
            }
            inv
        },
    )
}

/// Cholesky factorization `A = L Lᵀ` of a symmetric positive definite matrix (lower `L`).
///
/// Only the lower triangle of `a` is read. A matrix that is not positive definite produces
/// NaN in `L` (the square root of a negative number).
#[inline]
pub fn cholesky<T: Real, M: SquareArr<T>>(a: &M) -> M {
    T::vectorize(
        #[inline(always)]
        || {
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
        },
    )
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
    T::vectorize(
        #[inline(always)]
        || solve_lower_transposed(l, &solve_lower(l, b)),
    )
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
    T::vectorize(
        #[inline(always)]
        || {
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
        },
    )
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
    T::vectorize(
        #[inline(always)]
        || {
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
        },
    )
}

/// Singular value decomposition `A = U diag(σ) Vᵀ` by one-sided (Hestenes) Jacobi.
///
/// Returns `(u, sigma, v)` with the singular vectors as *rows*: `A vᵢ = σᵢ uᵢ`. Singular
/// values are non-negative and sorted in descending order. For a zero singular value the
/// corresponding `u` row is zero.
#[inline]
pub fn svd<T: Real, M: SquareArr<T>>(a: &M, sweeps: usize) -> (M, M::Vector, M) {
    T::vectorize(
        #[inline(always)]
        || {
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
        },
    )
}

/// A column of a least-squares problem: a vector with the operations one-sided Jacobi needs.
/// Implemented for `[T; N]`; gax implements it for an extensor's coefficients, so the
/// columns of a map with several slots need not be flattened into one array.
pub trait Column<T: Real>: Copy {
    /// The dot product (the Euclidean one of the coefficients).
    fn dot(&self, other: &Self) -> T;
    /// The plane rotation `(a, b) ← (c a − s b, s a + c b)`.
    fn rotate(a: &mut Self, b: &mut Self, c: T, s: T);
}

impl<T: Real, const N: usize> Column<T> for [T; N] {
    #[inline(always)]
    fn dot(&self, other: &Self) -> T {
        let mut s = T::zero();
        for k in 0..N {
            s = self[k].mul_add(other[k], s);
        }
        s
    }
    #[inline(always)]
    fn rotate(a: &mut Self, b: &mut Self, c: T, s: T) {
        for k in 0..N {
            let (x, y) = (a[k], b[k]);
            a[k] = c * x - s * y;
            b[k] = s * x + c * y;
        }
    }
}

/// The columns of a least-squares problem, addressed by one index: any array or slice of
/// [`Column`]s, or (in gax) the columns of a map's first two slots, kept nested.
pub trait Columns<C> {
    /// Column `i`.
    fn col(&self, i: usize) -> &C;
    /// Columns `p` and `q`, `p < q`, both mutable.
    fn pair_mut(&mut self, p: usize, q: usize) -> (&mut C, &mut C);
}

impl<C, A: AsRef<[C]> + AsMut<[C]> + ?Sized> Columns<C> for A {
    #[inline(always)]
    fn col(&self, i: usize) -> &C {
        &self.as_ref()[i]
    }
    #[inline(always)]
    fn pair_mut(&mut self, p: usize, q: usize) -> (&mut C, &mut C) {
        let (lo, hi) = self.as_mut().split_at_mut(q);
        (&mut lo[p], &mut hi[0])
    }
}

/// One-sided (Hestenes) Jacobi on the columns `a` of a matrix of any shape (one column per
/// row of `M`, so `M::N` columns): rotated pairwise until orthogonal, `W = A V`. Returns the
/// rotated columns `W` and `V` with its columns as *rows* (`v[i]` is the `i`-th right
/// singular vector, `‖W[i]‖` its singular value). Branch free but for the convergence test,
/// which stops once every lane has converged.
#[inline]
pub fn orthogonalize<T, C, A, M>(a: &A, sweeps: usize) -> (A, M)
where
    T: Real,
    C: Column<T>,
    A: Copy + Columns<C>,
    M: SquareArr<T>,
{
    T::vectorize(
        #[inline(always)]
        || {
            let n = M::N;
            let mut w = *a;
            let mut v: M = identity();
            let tol = T::epsilon() * T::epsilon();
            let cols = &mut w;
            for _ in 0..sweeps {
                // Converged when every pair is orthogonal relative to its own lengths,
                // `γ² ≤ ε² α β`: a test on sums would let large columns hide a small pair that is
                // not (singular values from 10⁶ down to 10⁻² left `A⁺ A` wrong by 10⁻³). A zero
                // column counts as orthogonal to everything.
                let mut open = T::zero();
                for p in 0..n {
                    for q in p + 1..n {
                        let (cp, cq) = (cols.col(p), cols.col(q));
                        let g = cp.dot(cq);
                        let ab = cp.dot(cp) * cq.dot(cq);
                        open = open + T::select_lt(tol * ab, g * g, T::one(), T::zero());
                    }
                }
                if T::all_lt(open, T::from_f64(0.5)) {
                    break;
                }
                for p in 0..n {
                    for q in p + 1..n {
                        let (cp, cq) = cols.pair_mut(p, q);
                        let alpha = cp.dot(cp);
                        let beta = cq.dot(cq);
                        let gamma = cp.dot(cq);
                        let (c, s) = jacobi_rotation(alpha, beta, gamma);
                        C::rotate(cp, cq, c, s);
                        for k in 0..n {
                            let (x, y) = (v[p][k], v[q][k]);
                            v[p][k] = c * x - s * y;
                            v[q][k] = s * x + c * y;
                        }
                    }
                }
            }
            (w, v)
        },
    )
}

/// The weights `1/σᵢ²` of the columns `W` of [`orthogonalize`], zero where
/// `σᵢ ≤ rcond · σ_max` (the pseudo-inverse's cutoff, `rcond` as in `numpy.linalg`).
#[inline]
pub fn pinv_weights<T, C, A, M>(w: &A, rcond: T) -> M::Vector
where
    T: Real,
    C: Column<T>,
    A: Columns<C> + ?Sized,
    M: SquareArr<T>,
{
    let mut sq = M::zero_vector();
    let mut largest = T::zero();
    for i in 0..M::N {
        sq[i] = w.col(i).dot(w.col(i));
        largest = largest.max(sq[i]);
    }
    let cut = rcond * rcond * largest;
    let mut weights = M::zero_vector();
    for i in 0..M::N {
        // A zero column (and with it the zero matrix) gets weight zero: `cut < 0` is false.
        let safe = T::select_lt(cut, sq[i], sq[i], T::one());
        weights[i] = T::select_lt(cut, sq[i], safe.recip(), T::zero());
    }
    weights
}

/// The minimum-norm least-squares solution `x = A⁺ b` for one right-hand side `b`, from
/// [`orthogonalize`]'s `W` and `V` and [`pinv_weights`]: `x = Σᵢ vᵢ (wᵢ · b) / σᵢ²`.
#[inline]
pub fn pinv_apply<T, C, A, M>(w: &A, v: &M, weights: &M::Vector, b: &C) -> M::Vector
where
    T: Real,
    C: Column<T>,
    A: Columns<C> + ?Sized,
    M: SquareArr<T>,
{
    let mut x = M::zero_vector();
    for i in 0..M::N {
        let t = w.col(i).dot(b) * weights[i];
        for h in 0..M::N {
            x[h] = v[i][h].mul_add(t, x[h]);
        }
    }
    x
}

/// The eigenvalues of a general (non-symmetric) real matrix, as `(re, im)`, complex pairs
/// conjugate and adjacent, sorted by real part, then imaginary part. Householder reduction to
/// Hessenberg form, then the double-shift QR iteration (EISPACK's `hqr`, as in JAMA), with its
/// exceptional shifts at the 10th and 30th iteration. Eigenvalues that do not converge in
/// `30 N` iterations are NaN. For scalar coefficients: the iteration branches on the data.
#[allow(clippy::many_single_char_names, clippy::too_many_lines)]
pub fn eigvals<T: Real, M: SquareArr<T>>(a: &M) -> (M::Vector, M::Vector) {
    let n = M::N;
    let mut h = *a;
    let (zero, one) = (T::zero(), T::one());
    let lt = |x: T, y: T| T::all_lt(x, y);
    // Householder reduction to upper Hessenberg form (an orthogonal similarity).
    for k in 0..n.saturating_sub(2) {
        let mut v = M::zero_vector();
        let mut alpha2 = zero;
        for i in k + 1..n {
            v[i] = h[i][k];
            alpha2 = alpha2 + v[i] * v[i];
        }
        let norm = alpha2.sqrt();
        if !lt(T::from_f64(1e-300), norm) {
            continue;
        }
        let sign = if lt(v[k + 1], zero) { -one } else { one };
        v[k + 1] = v[k + 1] + sign * norm;
        let mut vv = zero;
        for i in k + 1..n {
            vv = vv + v[i] * v[i];
        }
        let two_over = T::from_i64(2) / vv;
        for j in 0..n {
            let mut s = zero;
            for i in k + 1..n {
                s = s + v[i] * h[i][j];
            }
            let s = s * two_over;
            for i in k + 1..n {
                h[i][j] = h[i][j] - s * v[i];
            }
        }
        for i in 0..n {
            let mut s = zero;
            for j in k + 1..n {
                s = s + h[i][j] * v[j];
            }
            let s = s * two_over;
            for j in k + 1..n {
                h[i][j] = h[i][j] - s * v[j];
            }
        }
    }
    let (mut d, mut e) = (M::zero_vector(), M::zero_vector());
    if n == 0 {
        return (d, e);
    }
    let eps = T::epsilon();
    let mut norm = zero;
    for i in 0..n {
        for j in i.saturating_sub(1)..n {
            norm = norm + h[i][j].abs();
        }
    }
    let (mut p, mut q, mut r) = (zero, zero, zero);
    let (mut s, mut z): (T, T);
    let mut exshift = zero;
    let mut top = n as isize - 1;
    let mut iter = 0usize;
    let low = 0isize;
    let at = |h: &M, i: isize, j: isize| h[i as usize][j as usize];
    while top >= low {
        let nn = top;
        // A small subdiagonal element splits the matrix.
        let mut l = nn;
        while l > low {
            let mut s0 = at(&h, l - 1, l - 1).abs() + at(&h, l, l).abs();
            if s0 == zero {
                s0 = norm;
            }
            if lt(at(&h, l, l - 1).abs(), eps * s0) {
                break;
            }
            l -= 1;
        }
        if l == nn {
            // One root.
            d[nn as usize] = at(&h, nn, nn) + exshift;
            e[nn as usize] = zero;
            top -= 1;
            iter = 0;
        } else if l == nn - 1 {
            // Two roots: a real pair or a conjugate pair.
            let w = at(&h, nn, nn - 1) * at(&h, nn - 1, nn);
            p = (at(&h, nn - 1, nn - 1) - at(&h, nn, nn)) * T::from_f64(0.5);
            q = p * p + w;
            z = q.abs().sqrt();
            let x = at(&h, nn, nn) + exshift;
            let (i1, i0) = ((nn - 1) as usize, nn as usize);
            if lt(q, zero) {
                d[i1] = x + p;
                d[i0] = x + p;
                e[i1] = z;
                e[i0] = -z;
            } else {
                z = if lt(p, zero) { p - z } else { p + z };
                d[i1] = x + z;
                d[i0] = if z == zero { d[i1] } else { x - w / z };
                e[i1] = zero;
                e[i0] = zero;
            }
            top -= 2;
            iter = 0;
        } else {
            if iter > 30 * n {
                for k in low..=nn {
                    d[k as usize] = T::from_f64(f64::NAN);
                    e[k as usize] = T::from_f64(f64::NAN);
                }
                break;
            }
            let mut x = at(&h, nn, nn);
            let mut y = zero;
            let mut w = zero;
            if l < nn {
                y = at(&h, nn - 1, nn - 1);
                w = at(&h, nn, nn - 1) * at(&h, nn - 1, nn);
            }
            // Wilkinson's exceptional shift.
            if iter == 10 {
                exshift = exshift + x;
                for i in low..=nn {
                    h[i as usize][i as usize] = h[i as usize][i as usize] - x;
                }
                let s0 = at(&h, nn, nn - 1).abs() + at(&h, nn - 1, nn - 2).abs();
                x = T::from_f64(0.75) * s0;
                y = x;
                w = T::from_f64(-0.4375) * s0 * s0;
            }
            // MATLAB's exceptional shift.
            if iter == 30 {
                let mut s0 = (y - x) * T::from_f64(0.5);
                s0 = s0 * s0 + w;
                if lt(zero, s0) {
                    s0 = s0.sqrt();
                    if lt(y, x) {
                        s0 = -s0;
                    }
                    s0 = x - w / ((y - x) * T::from_f64(0.5) + s0);
                    for i in low..=nn {
                        h[i as usize][i as usize] = h[i as usize][i as usize] - s0;
                    }
                    exshift = exshift + s0;
                    x = T::from_f64(0.964);
                    y = x;
                    w = x;
                }
            }
            iter += 1;
            // Two consecutive small subdiagonal elements.
            let mut m = nn - 2;
            while m >= l {
                z = at(&h, m, m);
                r = x - z;
                s = y - z;
                p = (r * s - w) / at(&h, m + 1, m) + at(&h, m, m + 1);
                q = at(&h, m + 1, m + 1) - z - r - s;
                r = at(&h, m + 2, m + 1);
                s = p.abs() + q.abs() + r.abs();
                p = p / s;
                q = q / s;
                r = r / s;
                if m == l {
                    break;
                }
                let lhs = at(&h, m, m - 1).abs() * (q.abs() + r.abs());
                let rhs = eps
                    * (p.abs()
                        * (at(&h, m - 1, m - 1).abs() + z.abs() + at(&h, m + 1, m + 1).abs()));
                if lt(lhs, rhs) {
                    break;
                }
                m -= 1;
            }
            for i in m + 2..=nn {
                h[i as usize][(i - 2) as usize] = zero;
                if i > m + 2 {
                    h[i as usize][(i - 3) as usize] = zero;
                }
            }
            // The double QR step on rows l..=nn and columns m..=nn.
            let mut k = m;
            while k < nn {
                let notlast = k != nn - 1;
                if k != m {
                    p = at(&h, k, k - 1);
                    q = at(&h, k + 1, k - 1);
                    r = if notlast { at(&h, k + 2, k - 1) } else { zero };
                    x = p.abs() + q.abs() + r.abs();
                    if x == zero {
                        k += 1;
                        continue;
                    }
                    p = p / x;
                    q = q / x;
                    r = r / x;
                }
                s = (p * p + q * q + r * r).sqrt();
                if lt(p, zero) {
                    s = -s;
                }
                if s != zero {
                    if k != m {
                        h[k as usize][(k - 1) as usize] = -s * x;
                    } else if l != m {
                        h[k as usize][(k - 1) as usize] = -at(&h, k, k - 1);
                    }
                    p = p + s;
                    x = p / s;
                    y = q / s;
                    z = r / s;
                    q = q / p;
                    r = r / p;
                    for j in k..=nn {
                        let (k0, j0) = (k as usize, j as usize);
                        let mut pp = h[k0][j0] + q * h[k0 + 1][j0];
                        if notlast {
                            pp = pp + r * h[k0 + 2][j0];
                            h[k0 + 2][j0] = h[k0 + 2][j0] - pp * z;
                        }
                        h[k0][j0] = h[k0][j0] - pp * x;
                        h[k0 + 1][j0] = h[k0 + 1][j0] - pp * y;
                    }
                    let last = nn.min(k + 3);
                    for i in l..=last {
                        let (i0, k0) = (i as usize, k as usize);
                        let mut pp = x * h[i0][k0] + y * h[i0][k0 + 1];
                        if notlast {
                            pp = pp + z * h[i0][k0 + 2];
                            h[i0][k0 + 2] = h[i0][k0 + 2] - pp * r;
                        }
                        h[i0][k0] = h[i0][k0] - pp;
                        h[i0][k0 + 1] = h[i0][k0 + 1] - pp * q;
                    }
                }
                k += 1;
            }
        }
    }
    // Sort by real part, then imaginary part (insertion sort: N is small).
    for i in 1..n {
        let mut j = i;
        while j > 0 && (lt(d[j], d[j - 1]) || (d[j] == d[j - 1] && lt(e[j], e[j - 1]))) {
            let (a, b) = (d[j], e[j]);
            d[j] = d[j - 1];
            e[j] = e[j - 1];
            d[j - 1] = a;
            e[j - 1] = b;
            j -= 1;
        }
    }
    (d, e)
}

/// An eigenvector of `a` for the eigenvalue `re + i im`, as `(re, im)` parts, unit length, by
/// two steps of inverse iteration on `a − λ I` (complex Gaussian elimination with partial
/// pivoting, the shift nudged off the eigenvalue by a few ulps of the matrix's norm).
#[allow(clippy::many_single_char_names, clippy::too_many_lines)]
pub fn eigvector<T: Real, M: SquareArr<T>>(a: &M, re: T, im: T) -> (M::Vector, M::Vector) {
    let n = M::N;
    let zero = T::zero();
    let lt = |x: T, y: T| T::all_lt(x, y);
    let mut norm = zero;
    for i in 0..n {
        for j in 0..n {
            norm = norm + a[i][j].abs();
        }
    }
    let nudge = T::epsilon() * T::from_i64(64) * norm.max(T::one());
    let (lr, li) = (re + nudge, im);
    // B = A − λ I, as real and imaginary parts.
    let (mut br, mut bi) = (*a, M::zero());
    for i in 0..n {
        br[i][i] = br[i][i] - lr;
        bi[i][i] = bi[i][i] - li;
    }
    // LU with partial pivoting, in place; the permutation matrix records the row swaps.
    let mut perm = M::zero();
    for i in 0..n {
        perm[i][i] = T::one();
    }
    for k in 0..n {
        let mut piv = k;
        let mut best = br[k][k] * br[k][k] + bi[k][k] * bi[k][k];
        for i in k + 1..n {
            let v = br[i][k] * br[i][k] + bi[i][k] * bi[i][k];
            if lt(best, v) {
                best = v;
                piv = i;
            }
        }
        if piv != k {
            let (r0, i0) = (br[k], bi[k]);
            br[k] = br[piv];
            bi[k] = bi[piv];
            br[piv] = r0;
            bi[piv] = i0;
            let t = perm[k];
            perm[k] = perm[piv];
            perm[piv] = t;
        }
        let (pr, pi) = (br[k][k], bi[k][k]);
        let mut d2 = pr * pr + pi * pi;
        if !lt(zero, d2) {
            // A zero pivot: the iteration only needs a tiny one.
            br[k][k] = nudge;
            d2 = nudge * nudge;
        }
        let (pr, pi) = (br[k][k], bi[k][k]);
        for i in k + 1..n {
            // factor = B[i][k] / pivot.
            let (xr, xi) = (br[i][k], bi[i][k]);
            let fr = (xr * pr + xi * pi) / d2;
            let fi = (xi * pr - xr * pi) / d2;
            br[i][k] = fr;
            bi[i][k] = fi;
            for j in k + 1..n {
                let (ur, ui) = (br[k][j], bi[k][j]);
                br[i][j] = br[i][j] - (fr * ur - fi * ui);
                bi[i][j] = bi[i][j] - (fr * ui + fi * ur);
            }
        }
    }
    let (mut xr, mut xi) = (M::zero_vector(), M::zero_vector());
    for i in 0..n {
        // A start with every component, so no eigenvector is orthogonal to it.
        xr[i] = T::one() + T::from_f64(0.1) * T::from_i64(i as i64);
    }
    for _ in 0..3 {
        // Permute, then forward and back substitution.
        let (mut yr, mut yi) = (M::zero_vector(), M::zero_vector());
        for i in 0..n {
            for j in 0..n {
                yr[i] = yr[i] + perm[i][j] * xr[j];
                yi[i] = yi[i] + perm[i][j] * xi[j];
            }
        }
        for i in 0..n {
            for k in 0..i {
                let (fr, fi) = (br[i][k], bi[i][k]);
                yr[i] = yr[i] - (fr * yr[k] - fi * yi[k]);
                yi[i] = yi[i] - (fr * yi[k] + fi * yr[k]);
            }
        }
        for i in (0..n).rev() {
            let (mut sr, mut si) = (yr[i], yi[i]);
            for k in i + 1..n {
                let (ur, ui) = (br[i][k], bi[i][k]);
                sr = sr - (ur * yr[k] - ui * yi[k]);
                si = si - (ur * yi[k] + ui * yr[k]);
            }
            let (pr, pi) = (br[i][i], bi[i][i]);
            let d2 = pr * pr + pi * pi;
            yr[i] = (sr * pr + si * pi) / d2;
            yi[i] = (si * pr - sr * pi) / d2;
        }
        let mut len = zero;
        for i in 0..n {
            len = len + yr[i] * yr[i] + yi[i] * yi[i];
        }
        let inv = len.sqrt().recip();
        for i in 0..n {
            xr[i] = yr[i] * inv;
            xi[i] = yi[i] * inv;
        }
    }
    (xr, xi)
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

    /// Least squares through the columns: a tall full-rank system against the normal
    /// equations, and a rank-deficient one against its minimum-norm solution.
    #[test]
    fn least_squares() {
        // 4 equations, 2 unknowns: columns (1,1,1,1) and (0,1,2,3), b = (1,2,2,4).
        let a: [[f64; 4]; 2] = [[1.0, 1.0, 1.0, 1.0], [0.0, 1.0, 2.0, 3.0]];
        let (w, v): (_, [[f64; 2]; 2]) = orthogonalize(&a, 10);
        let weights = pinv_weights::<f64, [f64; 4], _, [[f64; 2]; 2]>(&w, 1e-15);
        let x = pinv_apply(&w, &v, &weights, &[1.0, 2.0, 2.0, 4.0]);
        // Normal equations: [[4, 6], [6, 14]] x = [9, 18] → x = (0.9, 0.9).
        assert!(close(x[0], 0.9) && close(x[1], 0.9), "{x:?}");
        // Two equal columns: the minimum-norm solution splits the weight evenly.
        let a: [[f64; 3]; 2] = [[1.0, 2.0, 2.0], [1.0, 2.0, 2.0]];
        let (w, v): (_, [[f64; 2]; 2]) = orthogonalize(&a, 10);
        let weights = pinv_weights::<f64, [f64; 3], _, [[f64; 2]; 2]>(&w, 1e-12);
        let x = pinv_apply(&w, &v, &weights, &[3.0, 6.0, 6.0]);
        assert!(close(x[0], 1.5) && close(x[1], 1.5), "{x:?}");
        // The zero matrix: the zero solution, no NaN.
        let a = [[0.0f64; 3]; 2];
        let (w, v): (_, [[f64; 2]; 2]) = orthogonalize(&a, 10);
        let weights = pinv_weights::<f64, [f64; 3], _, [[f64; 2]; 2]>(&w, 1e-12);
        let x = pinv_apply(&w, &v, &weights, &[1.0, 2.0, 3.0]);
        assert!(x.iter().all(|v| v.abs() < f64::MIN_POSITIVE), "{x:?}");
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
