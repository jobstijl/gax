//! Least squares on field maps of any shape: LSQR.

extern crate std;

use super::solve::{flatten, unflatten};
use super::{Field, Solved, SparseMap};
use crate::coef::Real;
use crate::kind::{Extensor, Kind};
use std::vec;

impl<C, A, T> SparseMap<C>
where
    C: Extensor<Slots = (A,), Coef = T>,
    A: Kind,
    T: Real,
{
    /// `y = A x` on flattened coefficients, for cells of any shape `K <- A`.
    pub(super) fn apply_rect(&self, x: &[T], y: &mut [T]) {
        let (nk, na) = (<C::Kind as Kind>::N, A::N);
        y.fill(T::zero());
        for r in 0..self.shape.0 {
            for k in self.starts[r]..self.starts[r + 1] {
                let xs = &x[self.cols[k] * na..(self.cols[k] + 1) * na];
                for (o, row) in self.cells[k].coeffs().as_ref().iter().enumerate() {
                    let row = row.as_ref();
                    let mut acc = y[r * nk + o];
                    for i in 0..na {
                        acc = row[i].mul_add(xs[i], acc);
                    }
                    y[r * nk + o] = acc;
                }
            }
        }
    }

    /// `x = Aᵀ y` on flattened coefficients: the transpose under the coefficient inner product.
    pub(super) fn apply_transpose_rect(&self, y: &[T], x: &mut [T]) {
        let (nk, na) = (<C::Kind as Kind>::N, A::N);
        x.fill(T::zero());
        for r in 0..self.shape.0 {
            let ys = &y[r * nk..(r + 1) * nk];
            for k in self.starts[r]..self.starts[r + 1] {
                let c = self.cols[k];
                for (o, row) in self.cells[k].coeffs().as_ref().iter().enumerate() {
                    for (i, &a) in row.as_ref().iter().enumerate() {
                        x[c * na + i] = a.mul_add(ys[o], x[c * na + i]);
                    }
                }
            }
        }
    }

    /// The least-squares solution of `self.apply(x) ≈ b` of least norm: the `x` over the input
    /// sites that minimizes `‖A x − b‖`, and among those the smallest, both in coefficient
    /// norms. A singular system's gauge (a translation of a free mesh, a constant potential)
    /// is left at zero. Maps of any shape: more output sites than input ones (fitting), fewer
    /// (under-determined), or cells between kinds of different sizes.
    ///
    /// It runs LSQR (Paige and Saunders) to [`lsqr`](Self::lsqr)'s defaults: a relative
    /// residual of the normal equations of `64 ε`, at most ten iterations per unknown. For
    /// scalar coefficients.
    ///
    /// ```
    /// use gax::field::{Field, SparseMap};
    /// use gax::pga3d::{Plane, Scalar};
    ///
    /// // Three sites, two differences: the steps are fixed, the level is free.
    /// let s = Scalar::<(), f64>::new;
    /// let steps = SparseMap::new(vec![s(-1.0), s(1.0), s(-1.0), s(1.0)], vec![0, 0, 1, 1], vec![0, 1, 1, 2], (2, 3));
    /// let map = &steps * Plane::slot();
    /// let rises = Field::from_fn(2, |_| Plane::new(1.0, 0.0, 0.0, 0.0));
    /// let x = map.lstsq(&rises);
    /// // The heights rise by one per site, centred on zero: the least-norm choice of level.
    /// let heights: Vec<f64> = x.iter().map(|p| p.e1()).collect();
    /// assert!(heights.iter().zip([-1.0, 0.0, 1.0]).all(|(h, w)| (h - w).abs() < 1e-12));
    /// ```
    ///
    /// # Panics
    /// If `b` is not over the map's output sites.
    #[must_use]
    #[track_caller]
    pub fn lstsq(&self, b: &Field<<C::Kind as Kind>::Mv<(), T>>) -> Field<A::Mv<(), T>> {
        let unknowns = self.shape.1 * A::N;
        self.lsqr(b, T::epsilon() * T::from_i64(64), 10 * unknowns + 10)
            .x
    }

    /// LSQR (Paige and Saunders, 1982) from zero, which converges to the least-norm
    /// least-squares solution: it stops when the normal equations' residual `‖Aᵀ r‖`, relative
    /// to `‖A‖ ‖r‖` (estimated as it goes), or the residual `‖r‖` relative to `‖b‖`, reaches
    /// `tolerance`, or after `max_iterations`. `residual` in the result is the first of these.
    ///
    /// # Panics
    /// If `b` is not over the map's output sites.
    #[must_use]
    #[track_caller]
    pub fn lsqr(
        &self,
        b: &Field<<C::Kind as Kind>::Mv<(), T>>,
        tolerance: T,
        max_iterations: usize,
    ) -> Solved<A::Mv<(), T>> {
        assert!(
            b.len() == self.shape.0,
            "a field map with {} output sites fitted to a field over {} sites",
            self.shape.0,
            b.len()
        );
        let (m, n) = (self.shape.0 * <C::Kind as Kind>::N, self.shape.1 * A::N);
        let norm = |v: &[T]| v.iter().fold(T::zero(), |s, x| x.mul_add(*x, s)).sqrt();
        let scale = |v: &mut [T], f: T| v.iter_mut().for_each(|x| *x = *x * f);
        let mut x = vec![T::zero(); n];
        let mut u = flatten(b);
        let mut beta = norm(&u);
        let zero = T::zero();
        if beta == zero {
            return Solved {
                x: unflatten(&x),
                iterations: 0,
                residual: zero,
                converged: true,
            };
        }
        let b_norm = beta;
        scale(&mut u, beta.recip());
        let mut v = vec![zero; n];
        self.apply_transpose_rect(&u, &mut v);
        let mut alpha = norm(&v);
        if alpha == zero {
            // b is orthogonal to the range: x = 0 is the least-norm solution.
            return Solved {
                x: unflatten(&x),
                iterations: 0,
                residual: zero,
                converged: true,
            };
        }
        scale(&mut v, alpha.recip());
        let mut w = v.clone();
        let (mut phi_bar, mut rho_bar) = (beta, alpha);
        let mut a_norm2 = alpha * alpha;
        let (mut av, mut atu) = (vec![zero; m], vec![zero; n]);
        let mut relative = T::one();
        for it in 1..=max_iterations {
            // Bidiagonalization: β u = A v − α u, α v = Aᵀ u − β v.
            self.apply_rect(&v, &mut av);
            for (ui, ai) in u.iter_mut().zip(&av) {
                *ui = (-alpha).mul_add(*ui, *ai);
            }
            beta = norm(&u);
            if beta == zero {
                alpha = zero;
            } else {
                scale(&mut u, beta.recip());
                self.apply_transpose_rect(&u, &mut atu);
                for (vi, ai) in v.iter_mut().zip(&atu) {
                    *vi = (-beta).mul_add(*vi, *ai);
                }
                alpha = norm(&v);
                if alpha != zero {
                    scale(&mut v, alpha.recip());
                }
            }
            a_norm2 = a_norm2 + alpha * alpha + beta * beta;
            // The plane rotation that eliminates β.
            let rho = (rho_bar * rho_bar + beta * beta).sqrt();
            let (c, s) = (rho_bar / rho, beta / rho);
            let theta = s * alpha;
            rho_bar = -(c * alpha);
            let phi = c * phi_bar;
            phi_bar = s * phi_bar;
            let (step, turn) = (phi / rho, theta / rho);
            for k in 0..n {
                x[k] = step.mul_add(w[k], x[k]);
                w[k] = (-turn).mul_add(w[k], v[k]);
            }
            // ‖r‖ = φ̄, ‖Aᵀ r‖ = φ̄ α |c|.
            let normal = phi_bar * alpha * c.abs();
            let normal_rel = normal / (a_norm2.sqrt() * phi_bar.max(T::epsilon() * b_norm));
            let residual_rel = phi_bar / b_norm;
            relative = normal_rel.min(residual_rel);
            if T::all_lt(relative, tolerance) || relative == zero || alpha == zero {
                return Solved {
                    x: unflatten(&x),
                    iterations: it,
                    residual: relative,
                    converged: true,
                };
            }
        }
        Solved {
            x: unflatten(&x),
            iterations: max_iterations,
            residual: relative,
            converged: false,
        }
    }
}
