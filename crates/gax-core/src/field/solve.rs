//! Solving a square field map iteratively: conjugate gradients, in the cells' pairing.

extern crate std;

use super::{CellPairing, Field, SparseMap, axpy, dot};
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

/// The signed permutation that pairs a cell's output coefficients with its input's (see
/// [`CellPairing`]): row `i` of the paired system is output row `rows[i].0`, negated where
/// `rows[i].1`.
#[derive(Clone, Debug)]
pub(super) struct Paired {
    rows: Vec<(usize, bool)>,
}

impl Paired {
    pub(super) fn of<C: CellPairing>(n: usize) -> Self {
        Paired {
            rows: (0..n).map(C::pair).collect(),
        }
    }

    /// The output blade that input blade `i` pairs with, and whether negated.
    pub(super) fn row(&self, i: usize) -> (usize, bool) {
        self.rows[i]
    }

    /// `−P`.
    pub(super) fn negated(self) -> Self {
        Paired {
            rows: self.rows.into_iter().map(|(j, n)| (j, !n)).collect(),
        }
    }

    /// `P y`, site by site: the output coefficients `y` as the input's.
    pub(super) fn apply<T: Real>(&self, y: &[T], out: &mut [T]) {
        let n = self.rows.len();
        for (ys, os) in y.chunks_exact(n).zip(out.chunks_exact_mut(n)) {
            for (o, &(j, negative)) in os.iter_mut().zip(&self.rows) {
                *o = if negative { -ys[j] } else { ys[j] };
            }
        }
    }

    /// `Pᵀ x`, site by site: the input coefficients `x` back as the output's.
    pub(super) fn apply_transpose<T: Real>(&self, x: &[T], out: &mut [T]) {
        let n = self.rows.len();
        for (xs, os) in x.chunks_exact(n).zip(out.chunks_exact_mut(n)) {
            for (&xi, &(j, negative)) in xs.iter().zip(&self.rows) {
                os[j] = if negative { -xi } else { xi };
            }
        }
    }
}

impl<C, A, B, T> SparseMap<C>
where
    C: Extensor<Kind = B, Slots = (A,), Coef = T> + CellPairing,
    A: Kind,
    B: Kind,
    T: Real,
{
    /// The diagonal of the paired block matrix `P A` (see [`CellPairing`]).
    pub(super) fn paired_diagonal(&self, paired: &Paired) -> Vec<T> {
        let n = A::N;
        let mut d = vec![T::zero(); self.shape.0 * n];
        for r in 0..self.shape.0 {
            for k in self.starts[r]..self.starts[r + 1] {
                if self.cols[k] == r {
                    let cell = self.cells[k].coeffs().as_ref();
                    for (i, &(j, negative)) in paired.rows.iter().enumerate() {
                        let v = cell[j].as_ref()[i];
                        d[r * n + i] = d[r * n + i] + if negative { -v } else { v };
                    }
                }
            }
        }
        d
    }

    #[track_caller]
    fn check_square(&self, b: usize) {
        assert!(
            A::N == B::N && self.shape.0 == self.shape.1,
            "solving needs a square field map, not {} <- {} sites of {} <- {} coefficients",
            self.shape.0,
            self.shape.1,
            B::N,
            A::N
        );
        assert!(
            b == self.shape.0,
            "a field map over {} sites solved against a field over {b} sites",
            self.shape.0
        );
    }

    /// Solve `self.of(x) == b` by conjugate gradients with a Jacobi (diagonal)
    /// preconditioner, the whole coupled system at once, starting from zero.
    ///
    /// The couplings must be symmetric and definite in their cells' pairing ([`CellPairing`]):
    /// a map of a kind to itself under the coefficient inner product ([`Field::dot`]), as a
    /// stiffness, a mass matrix or a Laplacian with anchors are; a correlation, from a kind to
    /// its complement, under the regressive product, as a conductance from potentials to
    /// amounts is. Either sign: a negative definite map (a conductance whose pairing puts the
    /// current before the gradient) is solved as its negative. The iteration stops when the
    /// relative residual `‖b − A x‖ / ‖b‖` reaches `tolerance`, after `max_iterations`, or when
    /// it meets a direction of the wrong curvature (the map is then not definite, and
    /// `converged` is false). For scalar coefficients.
    ///
    /// # Panics
    /// If the map is not square, or `b` is not over its sites.
    #[must_use]
    #[track_caller]
    pub fn cg(
        &self,
        b: &Field<B::Mv<(), T>>,
        tolerance: T,
        max_iterations: usize,
    ) -> Solved<A::Mv<(), T>> {
        self.check_square(b.len());
        let paired = Paired::of::<C>(A::N);
        let mut diag = self.paired_diagonal(&paired);
        // The sign of the definite map: its diagonal's.
        let trace = diag.iter().fold(T::zero(), |s, d| s + *d);
        let negative = T::all_lt(trace, T::zero());
        if negative {
            for d in &mut diag {
                *d = -*d;
            }
        }
        let flat = flatten(b);
        let mut rhs = vec![T::zero(); flat.len()];
        paired.apply(&flat, &mut rhs);
        if negative {
            for x in &mut rhs {
                *x = -*x;
            }
        }
        let mut image = vec![T::zero(); flat.len()];
        let (x, iterations, residual, converged) = pcg(
            |x, y| {
                self.apply_rect(x, &mut image);
                paired.apply(&image, y);
                if negative {
                    for v in y.iter_mut() {
                        *v = -*v;
                    }
                }
            },
            &diag,
            &rhs,
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

    /// Solve `self.of(x) == b` for `x`: [`cg`](Self::cg) to a relative residual of `16 ε`,
    /// for at most four iterations per unknown.
    ///
    /// For couplings symmetric and definite in their cells' pairing ([`CellPairing`]): maps of
    /// a kind to itself, and correlations, from a kind to its complement (cells `Amount <-
    /// Scalar`, a field of amounts in, a field of potentials out). For any other nonsingular
    /// map, [`solve_direct`](Self::solve_direct) solves exactly by sparse LU.
    ///
    /// ```
    /// use gax::field::{Field, SparseMap};
    /// use gax::pga2d::{Direction, OriginLine};
    /// // A correlation: forces (lines through the origin) to the velocities (directions) they
    /// // cause, two beads pushing each other a little. Symmetric and definite in the pairing of
    /// // a force with a velocity, its power.
    /// let along = OriginLine::<(), f64>::slot().dual().cast::<Direction>();
    /// let mobility = SparseMap::from_triples(
    ///     [(0, 0, along * 2.0), (0, 1, along * 0.5), (1, 0, along * 0.5), (1, 1, along * 2.0)],
    ///     (2, 2),
    /// );
    /// let velocities = Field::from_fn(2, |i| Direction::new(1.0, i as f64));
    /// let forces: Field<OriginLine<(), f64>> = mobility.solve(&velocities);
    /// let r = &mobility.of(&forces) - &velocities;
    /// assert!(r.norm() < 1e-14);
    /// ```
    ///
    /// # Panics
    /// If the map is not square, `b` is not over its sites, or the iteration fails (see
    /// [`try_solve`](Self::try_solve)).
    #[must_use]
    #[track_caller]
    pub fn solve(&self, b: &Field<B::Mv<(), T>>) -> Field<A::Mv<(), T>> {
        let s = self.solve_cg(b);
        assert!(
            solved(&s),
            "SparseMap::solve: conjugate gradients did not converge ({} iterations, relative residual {:?}); the couplings must be symmetric and definite, else use solve_direct",
            s.iterations,
            s.residual
        );
        s.x
    }

    /// [`solve`](Self::solve), or `None` where the iteration fails: its residual stays above
    /// `√ε` (the map is not definite, or too ill-conditioned, or `b` is not finite).
    ///
    /// ```
    /// use gax::field::{Field, SparseMap};
    /// use gax::vga3d::{Scalar, Vector};
    /// let s = Scalar::<(), f64>::new;
    /// let b = Field::from_fn(2, |_| Vector::new(1.0, 0.0, 0.0));
    /// let definite = &SparseMap::from_diagonal(&Field::from_fn(2, |_| s(2.0))) * Vector::slot();
    /// assert!(definite.try_solve(&b).is_some());
    /// // Negative definite is definite too.
    /// assert!((-&definite).try_solve(&b).is_some());
    /// // Conjugate gradients needs a definite map, and diag(1, -1) is not.
    /// let signs = Field::from_fn(2, |i| s(if i == 0 { 1.0 } else { -1.0 }));
    /// let indefinite = &SparseMap::from_diagonal(&signs) * Vector::slot();
    /// assert!(indefinite.try_solve(&b).is_none());
    /// ```
    ///
    /// # Panics
    /// If the map is not square, or `b` is not over its sites.
    #[must_use]
    #[track_caller]
    pub fn try_solve(&self, b: &Field<B::Mv<(), T>>) -> Option<Field<A::Mv<(), T>>> {
        let s = self.solve_cg(b);
        solved(&s).then_some(s.x)
    }

    /// [`cg`](Self::cg) with the tolerance and iterations of [`solve`](Self::solve).
    #[track_caller]
    fn solve_cg(&self, b: &Field<B::Mv<(), T>>) -> Solved<A::Mv<(), T>> {
        let unknowns = b.len() * B::N;
        self.cg(b, T::epsilon() * T::from_i64(16), 4 * unknowns + 20)
    }
}

/// Whether an iteration of [`SparseMap::solve`] is accepted: converged, or with a residual
/// below `√ε`.
fn solved<X: Extensor<Coef: Real>>(s: &Solved<X>) -> bool {
    s.converged || X::Coef::all_lt(s.residual, X::Coef::epsilon().sqrt())
}

/// The coefficients of a field of values, site by site.
pub(super) fn flatten<X: Extensor<Slots = ()>>(f: &Field<X>) -> Vec<X::Coef> {
    f.sites()
        .iter()
        .flat_map(|v| v.coeffs().as_ref().iter().copied())
        .collect()
}

/// A field of values from its coefficients, site by site.
pub(super) fn unflatten<X: Extensor<Slots = ()>>(c: &[X::Coef]) -> Field<X> {
    let n = <X::Kind as Kind>::N;
    Field::from_fn(c.len() / n, |s| {
        X::from_coeffs(<X::Kind as Kind>::arr_from_fn(|i| c[s * n + i]))
    })
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
        axpy(alpha, &p, &mut x);
        axpy(-alpha, &ap, &mut r);
        for ((z, r), i) in z.iter_mut().zip(&r).zip(&inv) {
            *z = *r * *i;
        }
        rr = dot(&r, &r);
        let rz_new = dot(&r, &z);
        let beta = rz_new / rz;
        rz = rz_new;
        for (p, z) in p.iter_mut().zip(&z) {
            *p = beta.mul_add(*p, *z);
        }
    }
    let converged = T::all_lt(rr, target) || rr == target;
    (x, max_iterations, (rr / bb).sqrt(), converged)
}
