//! Solving a square field map of square map cells: conjugate gradients, and LU on the
//! assembled block matrix.

extern crate std;

use super::{Field, SparseMap};
use crate::coef::Real;
use crate::kind::{Extensor, Kind};
use std::vec;
use std::vec::Vec;

/// The result of an iterative solve ([`SparseMap::cg`]).
#[derive(Clone, Debug, PartialEq)]
pub struct Solved<X: Extensor> {
    /// The solution, or the last iterate if the solve did not converge.
    pub x: Field<X>,
    /// The number of iterations taken.
    pub iterations: usize,
    /// The relative residual `‖b − A x‖ / ‖b‖` (coefficient norms) of the recurrence, at the
    /// end.
    pub residual: X::Coef,
    /// Whether the residual reached the tolerance.
    pub converged: bool,
}

impl<C, K, T> SparseMap<C>
where
    C: Extensor<Kind = K, Slots = (K,), Coef = T>,
    K: Kind,
    T: Real,
{
    /// `y = A x` on flattened coefficients (`site * K::N + blade`), each row's couplings in
    /// order: the block matrix-vector product, with the rounding of `apply`.
    fn apply_flat(&self, x: &[T], y: &mut [T]) {
        let n = K::N;
        for r in 0..self.shape.0 {
            let out = &mut y[r * n..(r + 1) * n];
            out.fill(T::zero());
            for k in self.starts[r]..self.starts[r + 1] {
                let xs = &x[self.cols[k] * n..(self.cols[k] + 1) * n];
                let cell = self.cells[k].coeffs().as_ref();
                for (o, row) in cell.iter().enumerate() {
                    let row = row.as_ref();
                    let mut acc = xs[0] * row[0];
                    for i in 1..n {
                        acc = acc + xs[i] * row[i];
                    }
                    out[o] = out[o] + acc;
                }
            }
        }
    }

    /// The diagonal of the block matrix: the diagonal coefficients of the cells on the site
    /// diagonal, summed.
    fn diagonal_flat(&self) -> Vec<T> {
        let n = K::N;
        let mut d = vec![T::zero(); self.shape.0 * n];
        for r in 0..self.shape.0 {
            for k in self.starts[r]..self.starts[r + 1] {
                if self.cols[k] == r {
                    let cell = self.cells[k].coeffs().as_ref();
                    for o in 0..n {
                        d[r * n + o] = d[r * n + o] + cell[o].as_ref()[o];
                    }
                }
            }
        }
        d
    }

    #[track_caller]
    fn check_square(&self, b: usize) {
        assert!(
            self.shape.0 == self.shape.1,
            "solving needs a square field map, not {} <- {} sites",
            self.shape.0,
            self.shape.1
        );
        assert!(
            b == self.shape.0,
            "a field map over {} sites solved against a field over {b} sites",
            self.shape.0
        );
    }

    /// Solve `self.apply(x) == b` by conjugate gradients with a Jacobi (diagonal)
    /// preconditioner, the whole coupled system at once, starting from zero.
    ///
    /// The couplings must be symmetric positive definite under the coefficient inner product
    /// ([`Field::dot`]), as a stiffness, a mass matrix or a Laplacian with anchors are. The
    /// iteration stops when the relative residual `‖b − A x‖ / ‖b‖` reaches `tolerance`, after
    /// `max_iterations`, or when it meets a direction of non-positive curvature (the map is then
    /// not positive definite, and `converged` is false). For scalar coefficients.
    ///
    /// # Panics
    /// If the map is not square, or `b` is not over its sites.
    #[must_use]
    #[track_caller]
    pub fn cg(
        &self,
        b: &Field<K::Mv<(), T>>,
        tolerance: T,
        max_iterations: usize,
    ) -> Solved<K::Mv<(), T>> {
        self.check_square(b.len());
        let flat = flatten(b);
        let diag = self.diagonal_flat();
        let (x, iterations, residual, converged) = pcg(
            |x, y| self.apply_flat(x, y),
            &diag,
            &flat,
            tolerance,
            max_iterations,
        );
        Solved {
            x: unflatten(&x),
            iterations,
            residual,
            converged,
        }
    }

    /// Solve `self.apply(x) == b` for `x`: [`cg`](Self::cg) to a relative residual of `16 ε`,
    /// for at most four iterations per unknown.
    ///
    /// For symmetric positive definite couplings. For any other nonsingular map,
    /// [`solve_dense`](Self::solve_dense) solves exactly by LU.
    ///
    /// # Panics
    /// If the map is not square, `b` is not over its sites, or the iteration fails: its
    /// residual stays above `√ε` (the map is not positive definite, or too ill-conditioned).
    #[must_use]
    #[track_caller]
    pub fn solve(&self, b: &Field<K::Mv<(), T>>) -> Field<K::Mv<(), T>> {
        let unknowns = b.len() * K::N;
        let eps = T::epsilon();
        let s = self.cg(b, eps * T::from_i64(16), 4 * unknowns + 20);
        assert!(
            s.converged || T::all_lt(s.residual, eps.sqrt()),
            "SparseMap::solve: conjugate gradients did not converge ({} iterations, relative residual {:?}); the couplings must be symmetric positive definite, else use solve_dense",
            s.iterations,
            s.residual
        );
        s.x
    }

    /// Solve `self.apply(x) == b` exactly, by LU with partial pivoting on the assembled block
    /// matrix of `(sites · K::N)²` coefficients: any nonsingular square field map, at the cost
    /// of a dense solve. A singular map gives non-finite coefficients. For scalar coefficients.
    ///
    /// # Panics
    /// If the map is not square, or `b` is not over its sites.
    #[must_use]
    #[track_caller]
    pub fn solve_dense(&self, b: &Field<K::Mv<(), T>>) -> Field<K::Mv<(), T>> {
        self.check_square(b.len());
        let n = K::N;
        let m = self.shape.0 * n;
        let mut a = vec![T::zero(); m * m];
        for (r, c, cell) in self.couplings() {
            let cell = cell.coeffs().as_ref();
            for o in 0..n {
                for i in 0..n {
                    let e = &mut a[(r * n + o) * m + c * n + i];
                    *e = *e + cell[o].as_ref()[i];
                }
            }
        }
        let mut x = flatten(b);
        lu_solve(&mut a, &mut x, m);
        unflatten(&x)
    }
}

/// The coefficients of a field of values, site by site.
fn flatten<X: Extensor<Slots = ()>>(f: &Field<X>) -> Vec<X::Coef> {
    f.sites()
        .iter()
        .flat_map(|v| v.coeffs().as_ref().iter().copied())
        .collect()
}

/// A field of values from its coefficients, site by site.
fn unflatten<X: Extensor<Slots = ()>>(c: &[X::Coef]) -> Field<X> {
    let n = <X::Kind as Kind>::N;
    Field::from_fn(c.len() / n, |s| {
        X::from_coeffs(<X::Kind as Kind>::arr_from_fn(|i| c[s * n + i]))
    })
}

/// `Σ aᵢ bᵢ` with four independent accumulators, so that the multiply-adds overlap instead of
/// waiting on one another.
fn dot<T: Real>(a: &[T], b: &[T]) -> T {
    let mut s = [T::zero(); 4];
    let (ca, ra) = a.as_chunks::<4>();
    let (cb, rb) = b.as_chunks::<4>();
    for (x, y) in ca.iter().zip(cb) {
        for l in 0..4 {
            s[l] = x[l].mul_add(y[l], s[l]);
        }
    }
    let mut t = (s[0] + s[1]) + (s[2] + s[3]);
    for (x, y) in ra.iter().zip(rb) {
        t = x.mul_add(*y, t);
    }
    t
}

/// Preconditioned conjugate gradients from zero: `(x, iterations, relative residual,
/// converged)`. The preconditioner is the inverse of `diag` where it is positive, else 1.
fn pcg<T: Real>(
    mut apply: impl FnMut(&[T], &mut [T]),
    diag: &[T],
    b: &[T],
    tolerance: T,
    max_iterations: usize,
) -> (Vec<T>, usize, T, bool) {
    let m = b.len();
    let inv: Vec<T> = diag
        .iter()
        .map(|&d| T::select_lt(T::zero(), d, d.recip(), T::one()))
        .collect();
    let bb = dot(b, b);
    let mut x = vec![T::zero(); m];
    if bb == T::zero() {
        return (x, 0, T::zero(), true);
    }
    let target = tolerance * tolerance * bb;
    let mut r = b.to_vec();
    let mut z: Vec<T> = r.iter().zip(&inv).map(|(r, i)| *r * *i).collect();
    let mut p = z.clone();
    let mut rz = dot(&r, &z);
    let mut ap = vec![T::zero(); m];
    let mut rr = bb;
    for it in 0..max_iterations {
        if T::all_lt(rr, target) || rr == target {
            return (x, it, (rr / bb).sqrt(), true);
        }
        apply(&p, &mut ap);
        let pap = dot(&p, &ap);
        if !T::all_lt(T::zero(), pap) {
            // Non-positive curvature: not positive definite.
            return (x, it, (rr / bb).sqrt(), false);
        }
        let alpha = rz / pap;
        for k in 0..m {
            x[k] = alpha.mul_add(p[k], x[k]);
            r[k] = (-alpha).mul_add(ap[k], r[k]);
            z[k] = r[k] * inv[k];
        }
        rr = dot(&r, &r);
        let rz_new = dot(&r, &z);
        let beta = rz_new / rz;
        rz = rz_new;
        for k in 0..m {
            p[k] = beta.mul_add(p[k], z[k]);
        }
    }
    let converged = T::all_lt(rr, target) || rr == target;
    (x, max_iterations, (rr / bb).sqrt(), converged)
}

/// Solve `a x = b` in place (`a` is `m × m`, row-major) by LU with partial pivoting.
fn lu_solve<T: Real>(a: &mut [T], b: &mut [T], m: usize) {
    for k in 0..m {
        // The pivot: the largest magnitude in column k at or below the diagonal.
        let mut p = k;
        for i in k + 1..m {
            if T::all_lt(a[p * m + k].abs(), a[i * m + k].abs()) {
                p = i;
            }
        }
        if p != k {
            for j in 0..m {
                a.swap(k * m + j, p * m + j);
            }
            b.swap(k, p);
        }
        let inv = a[k * m + k].recip();
        for i in k + 1..m {
            let f = a[i * m + k] * inv;
            if f == T::zero() {
                continue;
            }
            for j in k + 1..m {
                a[i * m + j] = (-f).mul_add(a[k * m + j], a[i * m + j]);
            }
            b[i] = (-f).mul_add(b[k], b[i]);
        }
    }
    for k in (0..m).rev() {
        let mut s = b[k];
        for j in k + 1..m {
            s = (-a[k * m + j]).mul_add(b[j], s);
        }
        b[k] = s / a[k * m + k];
    }
}
