//! Sparse generalized eigenproblems: shift-invert block Lanczos on the sparse LU, with the
//! Rayleigh–Ritz step on the pencil itself.

extern crate std;

use super::dense::symmetric_eigen;
use super::solve::{Paired, flatten, unflatten};
use super::{CellPairing, Field, SparseMap, axpy, dot};
use crate::coef::Real;
use crate::kind::{Extensor, Kind};
use core::cmp::Ordering;
use core::ops::Add;
use std::vec;
use std::vec::Vec;

impl<C, A, B, T> SparseMap<C>
where
    C: Extensor<Kind = B, Slots = (A,), Coef = T> + CellPairing + Add<Output = C>,
    A: Kind,
    B: Kind,
    T: Real,
{
    /// The `count` eigenpairs nearest zero of `self(x) = λ metric(x)`: values ascending, and
    /// fields orthonormal under the metric (`xᵢ · metric(xⱼ) = δᵢⱼ` in the cells' pairing,
    /// [`CellPairing`]). `self` must be symmetric and `metric` symmetric definite (a stiffness
    /// and a mass, a Laplacian and an area form), both maps of a kind to itself, paired by the
    /// coefficients, or both correlations, from a kind to its complement, paired by the
    /// regressive product (energy forms between primal and dual cochains); a negative definite
    /// metric (a correlation whose pairing comes out negative) is taken as its negative. `self`
    /// may be semidefinite: its null space (the rigid motions of a free mesh, a constant
    /// potential) then comes first. It may also be
    /// indefinite (a Hamiltonian), the pairs nearest zero then lying on both sides. numga's
    /// sparse `eigh`.
    ///
    /// The search space is built by block Lanczos on `(self − σ metric)⁻¹ metric`, the spectrum
    /// inverted about a shift `σ` just below zero (`−√ε` relative to the pencil's scale), so that
    /// the eigenvectors nearest zero dominate it; each step solves with one sparse LU of
    /// `self − σ metric`. The eigenpairs are then taken from the pencil itself on that space
    /// (Rayleigh–Ritz with `self` and `metric`, which need no solves; for an indefinite pencil
    /// or one with a null space, chosen first on the inverted operator, as ARPACK does), and the
    /// space grows until each pair's backward error, its true residual `‖self(x) − λ metric(x)‖` relative to
    /// `‖self‖ + |λ| ‖metric‖`, is within `ε^¾`.
    /// So the solves near the shift, which lose half their digits where the null space is
    /// amplified, shape the space but do not limit the accuracy. Two steps of block inverse
    /// iteration on the converged pairs' own vectors then sharpen the eigenvectors, whose error
    /// is the residual over the gap to the next eigenvalue.
    ///
    /// The blocks have `count + 2` vectors, at most `2 K::N + 2`. A block finds an eigenvalue
    /// with as many copies as it has vectors, and no value is wanted more than `count` times,
    /// so every copy asked for is found; past `count` the bound covers an eigenvalue repeated on
    /// every coefficient of the kind (a field map of isotropic cells repeats each one `K::N`
    /// times) and doubled by a symmetry of the sites ([`eigh_block`](Self::eigh_block) for
    /// more). A larger block than needed only grows the space faster than the pairs converge:
    /// a connection Laplacian's two parallel fields took 121 ms with blocks of 8, 62 ms with 2.
    ///
    /// ```
    /// use gax::field::{Field, SparseMap};
    /// use gax::pga3d::{Plane, Scalar};
    ///
    /// // A free string of five beads, displaced in the four coefficients of a plane: its
    /// // Laplacian, identity mass. The rigid shift is a mode on each coefficient (λ = 0, four
    /// // times), then the half wave, 2 − 2 cos(π / 5), four times.
    /// let n = 5;
    /// let s = Scalar::<(), f64>::new;
    /// let ends = SparseMap::new(
    ///     (0..n - 1).flat_map(|_| [s(-1.0), s(1.0)]).collect(),
    ///     (0..n - 1).flat_map(|e| [e, e]).collect(),
    ///     (0..n - 1).flat_map(|e| [e, e + 1]).collect(),
    ///     (n - 1, n),
    /// );
    /// let laplacian = &(&ends.reverse() * &ends) * Plane::slot();
    /// let mass = SparseMap::from_diagonal(&Field::from_fn(n, |_| Plane::<(), f64>::slot()));
    /// let (values, modes) = laplacian.eigh(&mass, 6);
    /// assert!(values[..4].iter().all(|v| v.abs() < 1e-12));
    /// let half_wave = 2.0 - 2.0 * (std::f64::consts::PI / 5.0).cos();
    /// assert!(values[4..].iter().all(|v| (v - half_wave).abs() < 1e-12));
    /// assert_eq!(modes.len(), 6);
    /// ```
    ///
    /// # Panics
    /// If the maps are not square over the same sites, `count` is zero or exceeds the
    /// unknowns, or `self − σ metric` is singular (the metric is not positive definite).
    #[must_use]
    #[track_caller]
    #[allow(clippy::type_complexity)]
    pub fn eigh(&self, metric: &SparseMap<C>, count: usize) -> (Vec<T>, Vec<Field<A::Mv<(), T>>>) {
        self.eigh_with(metric, count, default_block::<A>(count), &[])
    }

    /// [`eigh`](Self::eigh) with Lanczos blocks of `block` vectors. A block finds an
    /// eigenvalue with up to `block` copies at once; the default, `count + 2` and at most
    /// `2 K::N + 2`, finds every copy asked for, and past that covers a value repeated on every
    /// coefficient of the kind and doubled by a symmetry of the sites (a square grid's modes
    /// along its two axes). Use a larger block for more symmetric meshes.
    ///
    /// # Panics
    /// As [`eigh`](Self::eigh), and if `block` is zero.
    #[must_use]
    #[track_caller]
    #[allow(clippy::type_complexity)]
    pub fn eigh_block(
        &self,
        metric: &SparseMap<C>,
        count: usize,
        block: usize,
    ) -> (Vec<T>, Vec<Field<A::Mv<(), T>>>) {
        self.eigh_with(metric, count, block, &[])
    }

    /// [`eigh`](Self::eigh) from a start: fields near the wanted eigenvectors, such as the last
    /// step's modes in a flow that solves a slowly changing pencil step after step. They make
    /// the first block of the search space (topped up with the usual pseudo-random fields), so
    /// a good start needs fewer vectors and checks; any start gives the same pairs, to the same
    /// tolerance.
    ///
    /// ```
    /// use gax::field::{Field, SparseMap};
    /// use gax::vga3d::{Scalar, Vector};
    /// // A string of beads anchored at its ends, and the same string a little stiffer in the
    /// // middle: the first one's lowest modes start the second's search.
    /// let (n, s) = (200, Scalar::<(), f64>::new);
    /// let ends = SparseMap::new(
    ///     (0..n - 1).flat_map(|_| [s(-1.0), s(1.0)]).collect(),
    ///     (0..n - 1).flat_map(|e| [e, e]).collect(),
    ///     (0..n - 1).flat_map(|e| [e, e + 1]).collect(),
    ///     (n - 1, n),
    /// );
    /// let anchors = SparseMap::from_diagonal(&Field::from_fn(n, |i| s(if i == 0 || i == n - 1 { 1.0 } else { 0.0 })));
    /// let string = |middle: f64| {
    ///     let springs = SparseMap::from_diagonal(&Field::from_fn(n - 1, |e| s(if e == n / 2 { middle } else { 1.0 })));
    ///     &(&(&ends.reverse() * &(&springs * &ends)) + &anchors) * Vector::slot()
    /// };
    /// let mass = SparseMap::from_diagonal(&Field::from_fn(n, |_| Vector::<(), f64>::slot()));
    /// let (_, before) = string(1.0).eigh(&mass, 3);
    /// let (warm, _) = string(1.1).eigh_from(&mass, 3, &before);
    /// let (cold, _) = string(1.1).eigh(&mass, 3);
    /// assert!(warm.iter().zip(&cold).all(|(w, c)| (w - c).abs() < 1e-12 * c));
    /// ```
    ///
    /// # Panics
    /// As [`eigh`](Self::eigh), and if a start field is not over the map's sites.
    #[must_use]
    #[track_caller]
    #[allow(clippy::type_complexity)]
    pub fn eigh_from(
        &self,
        metric: &SparseMap<C>,
        count: usize,
        start: &[Field<A::Mv<(), T>>],
    ) -> (Vec<T>, Vec<Field<A::Mv<(), T>>>) {
        for s in start {
            self.check_input(s.len());
        }
        self.eigh_with(metric, count, default_block::<A>(count), start)
    }

    #[track_caller]
    #[allow(clippy::type_complexity)]
    fn eigh_with(
        &self,
        metric: &SparseMap<C>,
        count: usize,
        block: usize,
        start: &[Field<A::Mv<(), T>>],
    ) -> (Vec<T>, Vec<Field<A::Mv<(), T>>>) {
        assert!(block > 0, "eigh: a Lanczos block of zero vectors");
        assert!(
            self.shape.0 == self.shape.1 && metric.shape == self.shape && A::N == B::N,
            "eigh needs square field maps over the same sites: {:?} and {:?}",
            self.shape,
            metric.shape
        );
        let n = self.shape.0 * A::N;
        assert!(
            count > 0 && count <= n,
            "eigh: {count} eigenpairs asked of {n} unknowns"
        );
        // The pairing, signed so that the metric is positive: `P M` and `P A` are symmetric,
        // `P M` positive definite.
        let mut paired = Paired::of::<C>(A::N);
        let metric_diagonal = metric.paired_diagonal(&paired);
        if T::all_lt(
            metric_diagonal.iter().fold(T::zero(), |s, d| s + *d),
            T::zero(),
        ) {
            paired = paired.negated();
        }
        // The scales: the ∞-norms, which bound the spectral radii. (Not the diagonals: a
        // Hamiltonian of hops alone has none.)
        let (a_scale, m_scale) = (self.row_sum_norm(), metric.row_sum_norm());
        // The shift, just below zero relative to the pencil's scale.
        let mut sigma = -(T::epsilon().sqrt() * a_scale / m_scale);
        let first_shift = sigma;
        // The metric's diagonal, inverted: the eigenvalue error bounds measure residuals in
        // `M⁻¹`, which it stands for (exactly, for a lumped mass).
        let inverse_diagonal: Vec<T> = metric
            .paired_diagonal(&paired)
            .into_iter()
            .map(|d| T::select_lt(T::zero(), d, d.recip(), m_scale.recip()))
            .collect();
        let mut lu = factor_shifted(self, metric, sigma);
        let p = block.min(n);
        let mut image = vec![T::zero(); n];
        let mut image2 = vec![T::zero(); n];
        let mut space = Space::new(
            n,
            sigma,
            |x: &[T], y: &mut [T]| {
                metric.apply_rect(x, &mut image);
                paired.apply(&image, y);
            },
            |x: &[T], y: &mut [T]| {
                self.apply_rect(x, &mut image2);
                paired.apply(&image2, y);
            },
        );
        // (A − σM)⁻¹ Pᵀ y: the images are taken of paired vectors.
        let solve = |lu: &super::SparseLu<A, T, B>, y: &[T]| {
            let mut unpaired = vec![T::zero(); y.len()];
            paired.apply_transpose(y, &mut unpaired);
            lu.solve_flat(&unpaired)
        };
        // A start of at least `count` fields is first iterated on its own (block inverse
        // iteration); it seeds the space if that does not converge.
        let mut start: Vec<Vec<T>> = start.iter().map(flatten).collect();
        if start.len() >= count {
            let tolerances = Progress::<T>::new().tolerances();
            match space.iterate_start(start, count, (a_scale, m_scale), tolerances, |x| {
                solve(&lu, x)
            }) {
                Ok((values, vectors)) => {
                    return (values, vectors.iter().map(|x| unflatten(x)).collect());
                }
                Err(iterated) => start = iterated,
            }
        }
        space.extend_from(start.into_iter(), p);
        let mut progress = Progress::new();
        let mut reshifts = 0;
        // The space is checked when it has grown by a quarter since the last check: each check
        // costs a dense eigenproblem of the space's size, so the checks then cost a fixed
        // multiple of the last one.
        let mut next_check = count;
        loop {
            // The next directions: (A − σM)⁻¹ M q for each vector of the last block.
            space.image_last_block(|x| solve(&lu, x));
            // Past a re-shift the wanted cluster lies above the shift, not around it: a block
            // more of candidates keeps a value between zero and the shift in view.
            let extra = if reshifts > 0 { p } else { 0 };
            let m = space.len();
            if m >= next_check || m >= n {
                next_check = m + m / 4;
                let pairs = space.ritz(count, extra, |x| solve(&lu, x));
                if m >= n
                    || progress.done(&pairs, |value, coefficients| {
                        // The backward error: the residual relative to the pencil at λ.
                        space.residual(value, coefficients) / (a_scale + value.abs() * m_scale)
                    })
                {
                    let (values, vectors) =
                        space.refine(&pairs, (a_scale, m_scale), |x| solve(&lu, x));
                    return (values, vectors.iter().map(|x| unflatten(x)).collect());
                }
                // A spectrum that starts well above zero: shift to just below the wanted
                // cluster, so that it dominates the inverted operator. The space is kept and
                // its images are taken again with the new shift. Each new shift is at least
                // twice as close to the cluster as the last, so a factorization is spent only
                // when the estimates have moved enough. The estimates must be trusted first:
                // * not on the first check, whose Ritz values are the start's, averages over the
                //   spectrum (a free torus's tangent fields, all its pairs at zero, showed 461
                //   and moved there, and its space grew to half the unknowns);
                // * with their error bound counted in the spread: the residual in `M⁻¹`, within
                //   which some eigenvalue lies, which need not be the lowest (Spot's spin
                //   transformation estimated 8.3 for a lowest value of 0.0045). The diagonal of
                //   `M` stands for `M` (exactly, for a lumped mass); the plain residual
                //   understated the bound by `1/√m` for a small mass `m`.
                // Nor does it move when the space is settling, the strict tolerance two checks
                // away at the last check's rate: a move costs a factorization and the images of
                // the whole space again, more than those checks (Spot's spin transformation, its
                // error falling a thousandfold a check, keeps its first factorization). And a
                // move that has left a wanted value below the shift is taken back, to just below
                // that value. (The pairs are ordered by distance from zero, not by value; an
                // indefinite pencil, with wanted values on both sides of zero, is never
                // re-shifted.)
                let shifts = (sigma, first_shift, reshifts);
                if let Some(target) = space.next_shift(&pairs, shifts, &progress, &inverse_diagonal)
                {
                    sigma = target;
                    lu = factor_shifted(self, metric, sigma);
                    space.shift = sigma;
                    space.images_stale = true;
                    reshifts += 1;
                }
            }
            if !space.grow(p) {
                // The space is the whole space: the pairs are exact.
                let extra = if reshifts > 0 { p } else { 0 };
                let pairs = space.ritz(count, extra, |x| solve(&lu, x));
                return space.finish::<A>(&pairs);
            }
        }
    }
}

/// The default Lanczos block: `count + 2` vectors, at most `2 K::N + 2`.
fn default_block<K: Kind>(count: usize) -> usize {
    (count + 2).min(2 * K::N + 2)
}

/// The sparse LU of `A − σ M`.
fn factor_shifted<C, A, B, T>(
    a: &SparseMap<C>,
    metric: &SparseMap<C>,
    sigma: T,
) -> super::SparseLu<A, T, B>
where
    C: Extensor<Kind = B, Slots = (A,), Coef = T> + Add<Output = C>,
    A: Kind,
    B: Kind,
    T: Real,
{
    let shifted = a + &metric.map_cells(|c| super::scale(c, -sigma));
    match shifted.factor() {
        Ok(lu) => lu,
        Err(e) => panic!(
            "SparseMap::eigh: the shifted pencil is singular ({e}); the metric must be positive definite"
        ),
    }
}

impl<C, A, T> SparseMap<C>
where
    C: Extensor<Slots = (A,), Coef = T>,
    A: Kind,
    T: Real,
{
    /// The ∞-norm of the block matrix: its largest sum of absolute values along a row.
    pub(super) fn row_sum_norm(&self) -> T {
        let mut largest = T::zero();
        for r in 0..self.shape.0 {
            let cells = &self.cells[self.starts[r]..self.starts[r + 1]];
            for o in 0..<C::Kind as Kind>::N {
                let sum = cells.iter().fold(T::zero(), |s, cell| {
                    let row = cell.coeffs().as_ref()[o];
                    row.as_ref().iter().fold(s, |s, x| s + x.abs())
                });
                largest = largest.max(sum);
            }
        }
        largest
    }
}

/// When to stop growing the space: the worst backward error, `‖A x − λ M x‖ / (‖A‖ + |λ| ‖M‖)`
/// in ∞-norms, within `ε^¾`, or within `√ε` once it has stopped improving (no halving over two
/// checks). The measure is the residual relative to the pencil
/// at `λ`, so it does not change when the pencil is scaled (a stiffness of `10¹¹`). The
/// residuals have a floor of rounding, near `10⁴ ε` to `10⁵ ε`; past it, the space would grow
/// to the whole space for nothing.
struct Progress<T> {
    strict: T,
    loose: T,
    best: T,
    stalled: usize,
    /// The worst backward error of the last check, and of the one before.
    worst: T,
    previous: Option<T>,
    checked: bool,
}

impl<T: Real> Progress<T> {
    fn new() -> Self {
        let eps = T::epsilon();
        Progress {
            strict: eps.sqrt() * eps.sqrt().sqrt(),
            loose: eps.sqrt(),
            best: T::from_f64(f64::MAX),
            stalled: 0,
            worst: T::from_f64(f64::MAX),
            previous: None,
            checked: false,
        }
    }

    /// The strict and the loose tolerance on the backward error.
    fn tolerances(&self) -> (T, T) {
        (self.strict, self.loose)
    }

    /// Whether the last check's estimates come from a space that has taken a step: the first
    /// check sees the start vectors alone, whose Ritz values are averages over the spectrum.
    fn trusted(&self) -> bool {
        self.previous.is_some()
    }

    /// Whether the space is settling: at the rate of the last check, the strict tolerance is at
    /// most two checks away.
    fn settling(&self) -> bool {
        self.previous.is_some_and(|previous| {
            let rate = self.worst / previous;
            T::all_lt(rate, T::one()) && T::all_lt(self.worst * rate * rate, self.strict)
        })
    }

    /// Whether the pairs (value, coefficients) have converged, by `residual`. They are checked
    /// from the farthest from the shift, the slowest to converge, and the check stops at the
    /// first that has not: the worst residual seen is the one that decides.
    fn done(&mut self, pairs: &[(T, Vec<T>)], mut residual: impl FnMut(T, &[T]) -> T) -> bool {
        let mut worst = T::zero();
        for (value, coefficients) in pairs.iter().rev() {
            let r = residual(*value, coefficients);
            worst = worst.max(r);
            if T::all_lt(self.loose, r) {
                break; // not even near: no need to look further
            }
        }
        self.previous = self.checked.then_some(self.worst);
        self.checked = true;
        self.worst = worst;
        if !T::all_lt(self.strict, worst) {
            return true;
        }
        if T::all_lt(worst, self.best * T::from_f64(0.5)) {
            self.best = worst;
            self.stalled = 0;
        } else {
            self.stalled += 1;
        }
        self.stalled >= 2 && !T::all_lt(self.loose, worst)
    }
}

/// A search space orthonormal in a metric, grown by blocks, with the images of its vectors
/// under the metric and under the operator `A`.
struct Space<T, M, A> {
    n: usize,
    metric: M,
    operator: A,
    /// The basis, orthonormal in the metric.
    basis: Vec<Vec<T>>,
    /// `M q` for each basis vector.
    metric_images: Vec<Vec<T>>,
    /// `A q` for each basis vector.
    operator_images: Vec<Vec<T>>,
    /// `Qᵀ A Q`, grown with the basis: `projected[i][j] = qᵢ · A qⱼ` for `j ≤ i`.
    projected: Vec<Vec<T>>,
    /// `Qᵀ M (A − σM)⁻¹ M Q`, the inverted operator on the space, grown as the images are
    /// taken: `inverted[j][i] = M qᵢ · wⱼ`, `wⱼ` the image of `qⱼ`, for the `qᵢ` in the basis
    /// when it was taken.
    inverted: Vec<Vec<T>>,
    /// The shift `σ` of the images; whether it has moved since `inverted` was taken.
    shift: T,
    images_stale: bool,
    /// Whether the pairs are chosen on the inverted operator: the pencil has shown a negative
    /// value on the space, or one that is zero to rounding (a null space). A growing space only
    /// moves its Ritz values outwards to the spectrum's ends (by interlacing), so once either
    /// shows it stays.
    choose_inverted: bool,
    /// The directions that start the next block.
    remainder: Vec<Vec<T>>,
    /// The first vector of the last block.
    block_start: usize,
    seed: u64,
}

impl<T: Real, M: FnMut(&[T], &mut [T]), A: FnMut(&[T], &mut [T])> Space<T, M, A> {
    fn new(n: usize, shift: T, metric: M, operator: A) -> Self {
        Space {
            n,
            metric,
            operator,
            basis: Vec::new(),
            metric_images: Vec::new(),
            operator_images: Vec::new(),
            projected: Vec::new(),
            inverted: Vec::new(),
            shift,
            images_stale: false,
            choose_inverted: false,
            remainder: Vec::new(),
            block_start: 0,
            seed: 0,
        }
    }

    fn len(&self) -> usize {
        self.basis.len()
    }

    fn apply_metric(&mut self, x: &[T]) -> Vec<T> {
        let mut y = vec![T::zero(); self.n];
        (self.metric)(x, &mut y);
        y
    }

    /// `v` with its components along the basis removed in the metric, twice (full
    /// reorthogonalization).
    fn orthogonalize(&mut self, v: &mut [T]) {
        // qᵢ · M v = (M qᵢ) · v: the stored images give the coefficients without a product.
        for _ in 0..2 {
            for (q, mq) in self.basis.iter().zip(&self.metric_images) {
                axpy(-dot(mq, v), q, v);
            }
        }
    }

    /// Append `v`, orthonormalized against the basis, if it has a component outside it.
    fn push(&mut self, mut v: Vec<T>) {
        let before = dot(&v, &self.apply_metric(&v)).sqrt();
        self.orthogonalize(&mut v);
        let mv = self.apply_metric(&v);
        let length = dot(&v, &mv).sqrt();
        if !T::all_lt(T::epsilon() * T::from_i64(1000) * before, length) {
            return; // dependent on the basis
        }
        let inv = length.recip();
        let q: Vec<T> = v.iter().map(|x| *x * inv).collect();
        let mut aq = vec![T::zero(); self.n];
        (self.operator)(&q, &mut aq);
        // The new row of Qᵀ A Q: q · A qᵢ (A is symmetric, so one side serves).
        let row: Vec<T> = self
            .operator_images
            .iter()
            .map(|aqi| dot(&q, aqi))
            .chain(core::iter::once(dot(&q, &aq)))
            .collect();
        self.projected.push(row);
        self.basis.push(q);
        self.metric_images
            .push(mv.iter().map(|x| *x * inv).collect());
        self.operator_images.push(aq);
    }

    /// Pseudo-random vectors until the block since `block_start` has `p` vectors (or the space
    /// is full).
    fn top_up(&mut self, p: usize) {
        let mut tries = 0;
        while self.basis.len() - self.block_start < p && self.basis.len() < self.n && tries < 4 * p
        {
            let v = start_vector::<T>(self.n, self.seed);
            self.seed += 1;
            tries += 1;
            self.push(v);
        }
    }

    /// Start a block with the given vectors, topped up to `p` with pseudo-random ones.
    fn extend_from(&mut self, start: impl Iterator<Item = Vec<T>>, p: usize) {
        self.block_start = self.basis.len();
        for v in start {
            if self.basis.len() >= self.n {
                break;
            }
            self.push(v);
        }
        self.top_up(p);
    }

    /// The image `w = (A − σM)⁻¹ M q` of each vector of the last block, by `solve`, recorded
    /// in the inverted operator and kept as the directions that start the next block.
    fn image_last_block(&mut self, mut solve: impl FnMut(&[T]) -> Vec<T>) {
        // After a new shift the earlier rows are stale; they are taken again only if the
        // inverted operator is needed (`ritz`).
        debug_assert_eq!(self.inverted.len(), self.block_start);
        self.remainder = (self.block_start..self.basis.len())
            .map(|j| self.image(j, &mut solve))
            .collect();
    }

    /// Every basis vector's image again, after a new shift.
    fn reimage(&mut self, mut solve: impl FnMut(&[T]) -> Vec<T>) {
        self.inverted.clear();
        for j in 0..self.basis.len() {
            self.image(j, &mut solve);
        }
        self.images_stale = false;
    }

    /// The image of basis vector `j`, its pairings with the basis recorded.
    fn image(&mut self, j: usize, solve: &mut impl FnMut(&[T]) -> Vec<T>) -> Vec<T> {
        let w = solve(&self.metric_images[j]);
        let row = self.metric_images.iter().map(|mq| dot(mq, &w)).collect();
        self.inverted.push(row);
        w
    }

    /// The next block: the remainder orthonormalized against the basis, topped up with
    /// pseudo-random vectors where it is dependent. Whether the space grew.
    fn grow(&mut self, p: usize) -> bool {
        self.block_start = self.basis.len();
        for w in core::mem::take(&mut self.remainder) {
            if self.basis.len() >= self.n {
                break;
            }
            self.push(w);
        }
        self.top_up(p);
        self.basis.len() > self.block_start
    }

    /// The `count` Ritz pairs nearest zero of the pencil on the space, as `(λ, x)` with `x`
    /// the coefficients on the basis, nearest first.
    ///
    /// Where the pencil is definite on the space, the values nearest zero are its lowest, and
    /// Rayleigh–Ritz with `A` (the basis is orthonormal in the metric, so `Qᵀ M Q = I`) finds
    /// them best. It does not in two cases, where the pairs are chosen on the inverted operator
    /// instead, as ARPACK's shift-invert mode does:
    /// * inside an indefinite spectrum (a Hamiltonian's levels around its gap), where it finds
    ///   spurious values near zero, mixtures of eigenvectors on both sides;
    /// * next to a null space (a free mesh's rigid motions, a constant field), where its values
    ///   are zero to rounding: a Rayleigh quotient is quadratic in the error, so a null vector
    ///   polluted to `√ε` looks exact to it, while on the inverted operator a null vector stands
    ///   out by the shift's inverse.
    ///
    /// There each vector of the inverted operator has its eigenvalue estimate `σ + 1/θ`; the
    /// `count + extra` estimates nearest zero (for any shift: past a re-shift the wanted values
    /// lie below it) are rotated by Rayleigh–Ritz with `A` on their span, whose values are exact
    /// to rounding however the images were rounded.
    fn ritz(
        &mut self,
        count: usize,
        extra: usize,
        solve: impl FnMut(&[T]) -> Vec<T>,
    ) -> Vec<(T, Vec<T>)> {
        let m = self.basis.len();
        let projected = self.projected_matrix();
        if !self.choose_inverted {
            let (values, vectors) = symmetric_eigen(&projected, m);
            let largest = values.iter().fold(T::zero(), |l, v| l.max(v.abs()));
            let lowest = values.iter().fold(T::zero(), |l, v| l.min(*v));
            let nearest_zero = values.iter().fold(largest, |l, v| l.min(v.abs()));
            let rounding = T::epsilon().sqrt() * largest;
            self.choose_inverted =
                T::all_lt(lowest, -rounding) || T::all_lt(nearest_zero, rounding);
            if !self.choose_inverted {
                let mut nearest: Vec<usize> = (0..m).collect();
                nearest.sort_by(|&i, &j| order(values[i].abs(), values[j].abs()));
                return nearest
                    .into_iter()
                    .take(count)
                    .map(|i| (values[i], vectors[i].clone()))
                    .collect();
            }
        }
        if self.images_stale {
            self.reimage(solve);
        }
        let (inverted, vectors) = symmetric_eigen(&self.inverted_matrix(), m);
        // The eigenvalue each vector of the inverted operator estimates; a zero there is no
        // estimate (an infinite one).
        let estimate = |theta: T| {
            if theta == T::zero() {
                T::from_f64(f64::MAX)
            } else {
                (self.shift + theta.recip()).abs()
            }
        };
        let mut nearest: Vec<usize> = (0..m).collect();
        nearest.sort_by(|&i, &j| order(estimate(inverted[i]), estimate(inverted[j])));
        let chosen: Vec<&Vec<T>> = nearest
            .iter()
            .take((count + extra).min(m))
            .map(|&i| &vectors[i])
            .collect();
        let k = chosen.len();
        let mut on_span = vec![T::zero(); k * k];
        for (a, ya) in chosen.iter().enumerate() {
            let pya = mat_vec(&projected, ya, m);
            for (b, yb) in chosen.iter().enumerate().take(a + 1) {
                let v = dot(yb, &pya);
                on_span[a * k + b] = v;
                on_span[b * k + a] = v;
            }
        }
        let (values, rotations) = symmetric_eigen(&on_span, k);
        let mut pairs: Vec<(T, Vec<T>)> = values
            .into_iter()
            .zip(rotations)
            .map(|(value, z)| {
                let mut c = vec![T::zero(); m];
                for (y, &zi) in chosen.iter().zip(&z) {
                    axpy(zi, y, &mut c);
                }
                (value, c)
            })
            .collect();
        pairs.sort_by(|a, b| order(a.0.abs(), b.0.abs()));
        pairs.truncate(count);
        pairs
    }

    /// `Qᵀ A Q`, square.
    fn projected_matrix(&self) -> Vec<T> {
        let m = self.basis.len();
        let mut p = vec![T::zero(); m * m];
        for (i, row) in self.projected.iter().enumerate() {
            for (j, &v) in row.iter().enumerate() {
                p[i * m + j] = v;
                p[j * m + i] = v;
            }
        }
        p
    }

    /// `Qᵀ M (A − σM)⁻¹ M Q`, square and symmetric: each entry from the image taken when both
    /// vectors were in the basis, the mean of the two where both were.
    fn inverted_matrix(&self) -> Vec<T> {
        let m = self.basis.len();
        let half = T::from_f64(0.5);
        let mut h = vec![T::zero(); m * m];
        for i in 0..m {
            for j in 0..=i {
                let (a, b) = (self.inverted[j].get(i), self.inverted[i].get(j));
                let v = match (a, b) {
                    (Some(&a), Some(&b)) => (a + b) * half,
                    (Some(&v), None) | (None, Some(&v)) => v,
                    (None, None) => unreachable!("every pair of basis vectors has an image"),
                };
                h[i * m + j] = v;
                h[j * m + i] = v;
            }
        }
        h
    }

    /// The vector `x = Σ cᵢ qᵢ` of a pair's coefficients.
    fn combine(&self, coefficients: &[T]) -> Vec<T> {
        let mut x = vec![T::zero(); self.n];
        for (q, &c) in self.basis.iter().zip(coefficients) {
            axpy(c, q, &mut x);
        }
        x
    }

    /// Block inverse iteration on the converged pairs' own vectors, outside the search space:
    /// `Y = (A − σM)⁻¹ M X`, made orthonormal in the metric, and Rayleigh–Ritz with `A` on `Y`,
    /// twice. The stopping residual bounds an eigenvector's error only by the residual over the
    /// gap to the next eigenvalue, which on a fine mesh is a small part of the scale, and where
    /// the pairs are chosen on the inverted operator they carry its rounding; each step shrinks
    /// an error component by `|λ − σ| / |λⱼ − σ|`, the ratio of the wanted eigenvalue's distance
    /// from the shift to the other's, which next to a null space is about `√ε`. (Adding the
    /// images to the Lanczos space instead, as the first version did, gained nothing there: the
    /// corrections lie in the space already, below the rounding of the extraction, and a
    /// connection Laplacian's parallel field came back with an energy of `10⁻¹²` for an exact
    /// zero.) The refined pairs are kept unless their backward error is worse (`scales` are
    /// the ∞-norms of `A` and `M`): the values, and the vectors, ascending.
    #[allow(clippy::type_complexity)]
    fn refine(
        &mut self,
        pairs: &[(T, Vec<T>)],
        scales: (T, T),
        mut solve: impl FnMut(&[T]) -> Vec<T>,
    ) -> (Vec<T>, Vec<Vec<T>>) {
        let start_values: Vec<T> = pairs.iter().map(|p| p.0).collect();
        let start: Vec<Vec<T>> = pairs.iter().map(|(_, c)| self.combine(c)).collect();
        let before = self.backward_error(&start_values, &start, scales);
        let mut refined = (start_values.clone(), start.clone());
        for _ in 0..2 {
            match self.inverse_step(&refined.1, &mut solve) {
                Some(step) => refined = step,
                None => return finish_vectors(start_values, start),
            }
        }
        let after = self.backward_error(&refined.0, &refined.1, scales);
        if T::all_lt(before * T::from_i64(2), after) {
            finish_vectors(start_values, start)
        } else {
            finish_vectors(refined.0, refined.1)
        }
    }

    /// The worst backward error of the pairs `(values, x)`, `‖A x − λ M x‖ / (‖A‖ + |λ| ‖M‖)`,
    /// `x` of unit length in the metric and `scales` the ∞-norms of `A` and `M`.
    fn backward_error(&mut self, values: &[T], x: &[Vec<T>], scales: (T, T)) -> T {
        values.iter().zip(x).fold(T::zero(), |worst, (&value, xi)| {
            let mx = self.apply_metric(xi);
            let mut ax = vec![T::zero(); self.n];
            (self.operator)(xi, &mut ax);
            let r = ax
                .iter()
                .zip(&mx)
                .fold(T::zero(), |s, (a, m)| {
                    let r = (-value).mul_add(*m, *a);
                    r.mul_add(r, s)
                })
                .sqrt();
            worst.max(r / (scales.0 + value.abs() * scales.1))
        })
    }

    /// One step of block inverse iteration: `Y = (A − σM)⁻¹ M X`, orthonormal in the metric,
    /// and Rayleigh–Ritz with `A` on `Y`: the values ascending and their vectors, or `None`
    /// where `Y` loses a dimension.
    #[allow(clippy::type_complexity)]
    fn inverse_step(
        &mut self,
        x: &[Vec<T>],
        solve: &mut impl FnMut(&[T]) -> Vec<T>,
    ) -> Option<(Vec<T>, Vec<Vec<T>>)> {
        let k = x.len();
        let (mut y, mut my): (Vec<Vec<T>>, Vec<Vec<T>>) =
            (Vec::with_capacity(k), Vec::with_capacity(k));
        for xi in x {
            let mut v = solve(&self.apply_metric(xi));
            let size = dot(&v, &self.apply_metric(&v)).sqrt();
            for _ in 0..2 {
                for (yj, myj) in y.iter().zip(&my) {
                    axpy(-dot(myj, &v), yj, &mut v);
                }
            }
            let mv = self.apply_metric(&v);
            let length = dot(&v, &mv).sqrt();
            if !T::all_lt(T::epsilon() * T::from_i64(1000) * size, length) {
                return None;
            }
            let inv = length.recip();
            y.push(v.iter().map(|x| *x * inv).collect::<Vec<T>>());
            my.push(mv.iter().map(|x| *x * inv).collect::<Vec<T>>());
        }
        let ay: Vec<Vec<T>> = y
            .iter()
            .map(|v| {
                let mut av = vec![T::zero(); self.n];
                (self.operator)(v, &mut av);
                av
            })
            .collect();
        let mut g = vec![T::zero(); k * k];
        for i in 0..k {
            for j in 0..=i {
                let v = (dot(&y[i], &ay[j]) + dot(&y[j], &ay[i])) * T::from_f64(0.5);
                g[i * k + j] = v;
                g[j * k + i] = v;
            }
        }
        let (theta, rotations) = symmetric_eigen(&g, k);
        let x = rotations
            .iter()
            .map(|z| {
                let mut xi = vec![T::zero(); self.n];
                for (yj, &zj) in y.iter().zip(z) {
                    axpy(zj, yj, &mut xi);
                }
                xi
            })
            .collect();
        Some((theta, x))
    }

    /// Block inverse iteration from a start of at least `count` vectors, while each step cuts
    /// the backward error of the `count` pairs nearest zero tenfold (above the loose tolerance)
    /// or in half (below it): the pairs, if they reach the strict tolerance or stall below the
    /// loose one (the floor of rounding, where [`Progress`] stops too), else the last block, to
    /// start the Lanczos space with. A good start
    /// (the last step's modes in a flow) converges in a few solves of its own vectors, where
    /// the space would grow by whole blocks; a poor one, or a cluster the start does not
    /// separate, falls through after a step or two.
    #[allow(clippy::type_complexity)]
    fn iterate_start(
        &mut self,
        start: Vec<Vec<T>>,
        count: usize,
        scales: (T, T),
        (strict, loose): (T, T),
        mut solve: impl FnMut(&[T]) -> Vec<T>,
    ) -> Result<(Vec<T>, Vec<Vec<T>>), Vec<Vec<T>>> {
        let mut x = start;
        let mut last = T::from_f64(f64::MAX);
        for _ in 0..32 {
            let Some((values, vectors)) = self.inverse_step(&x, &mut solve) else {
                return Err(x);
            };
            let mut nearest: Vec<usize> = (0..values.len()).collect();
            nearest.sort_by(|&i, &j| order(values[i].abs(), values[j].abs()));
            nearest.truncate(count);
            let wanted: Vec<T> = nearest.iter().map(|&i| values[i]).collect();
            let wanted_x: Vec<Vec<T>> = nearest.iter().map(|&i| vectors[i].clone()).collect();
            let worst = self.backward_error(&wanted, &wanted_x, scales);
            x = vectors;
            if !T::all_lt(strict, worst) {
                return Ok(finish_vectors(wanted, wanted_x));
            }
            // Above the loose tolerance a step must cut the error tenfold, or the Lanczos space
            // would do better; below it, a halving keeps it going, and less is the floor of
            // rounding.
            if T::all_lt(loose, worst) {
                if !T::all_lt(worst * T::from_i64(10), last) {
                    return Err(x);
                }
            } else if !T::all_lt(worst * T::from_i64(2), last) {
                return Ok(finish_vectors(wanted, wanted_x));
            }
            last = worst;
        }
        Err(x)
    }

    /// The pairs as fields, in ascending order of their values.
    #[allow(clippy::type_complexity)]
    fn finish<K: Kind>(&self, pairs: &[(T, Vec<T>)]) -> (Vec<T>, Vec<Field<K::Mv<(), T>>>) {
        let (values, vectors) = finish_vectors(
            pairs.iter().map(|p| p.0).collect(),
            pairs.iter().map(|(_, c)| self.combine(c)).collect(),
        );
        (values, vectors.iter().map(|x| unflatten(x)).collect())
    }

    /// The next shift, if the rule of [`SparseMap::eigh`] moves it (see there): `shifts` is the
    /// current shift, the first one and the number of moves so far.
    fn next_shift(
        &self,
        pairs: &[(T, Vec<T>)],
        (sigma, first_shift, reshifts): (T, T, usize),
        progress: &Progress<T>,
        inverse_diagonal: &[T],
    ) -> Option<T> {
        let low = pairs.iter().fold(pairs[0].0, |m, p| m.min(p.0));
        let high = pairs.iter().fold(pairs[0].0, |m, p| m.max(p.0));
        let below = reshifts > 0 && T::all_lt(low, sigma);
        if reshifts >= 16 || !progress.trusted() || (!below && progress.settling()) {
            return None;
        }
        let bound = pairs
            .iter()
            .map(|(value, c)| self.eigenvalue_bound(*value, c, inverse_diagonal))
            .fold(T::zero(), T::max);
        let spread = (high - low).max(bound).max(T::epsilon() * high.abs());
        let target = low - spread;
        if below {
            Some(first_shift.max(target))
        } else {
            (T::all_lt(T::zero(), target)
                && T::all_lt(sigma, target)
                && T::all_lt(spread, (low - sigma) * T::from_f64(0.5)))
            .then_some(target)
        }
    }

    /// A bound on the distance from `value` to the nearest eigenvalue: the residual of
    /// `x = Σ cᵢ qᵢ` in the norm of `M⁻¹`, with `M`'s inverted diagonal standing for `M⁻¹`
    /// (`x` has unit length in `M`). Some eigenvalue lies within it, which need not be the lowest.
    fn eigenvalue_bound(&self, value: T, coefficients: &[T], inverse_diagonal: &[T]) -> T {
        let mut ax = vec![T::zero(); self.n];
        let mut mx = vec![T::zero(); self.n];
        for (&c, (aq, mq)) in coefficients
            .iter()
            .zip(self.operator_images.iter().zip(&self.metric_images))
        {
            axpy(c, aq, &mut ax);
            axpy(c, mq, &mut mx);
        }
        ax.iter()
            .zip(&mx)
            .zip(inverse_diagonal)
            .fold(T::zero(), |s, ((a, m), d)| {
                let r = (-value).mul_add(*m, *a);
                (r * r).mul_add(*d, s)
            })
            .sqrt()
    }

    /// `‖A x − λ M x‖` for `x = Σ cᵢ qᵢ`, from the basis' images (no products).
    fn residual(&self, value: T, coefficients: &[T]) -> T {
        let mut ax = vec![T::zero(); self.n];
        let mut mx = vec![T::zero(); self.n];
        for (&c, (aq, mq)) in coefficients
            .iter()
            .zip(self.operator_images.iter().zip(&self.metric_images))
        {
            axpy(c, aq, &mut ax);
            axpy(c, mq, &mut mx);
        }
        ax.iter()
            .zip(&mx)
            .fold(T::zero(), |s, (a, m)| {
                let r = (-value).mul_add(*m, *a);
                r.mul_add(r, s)
            })
            .sqrt()
    }
}

/// Values and their vectors, in ascending order of the values.
#[allow(clippy::type_complexity)]
fn finish_vectors<T: Real>(values: Vec<T>, vectors: Vec<Vec<T>>) -> (Vec<T>, Vec<Vec<T>>) {
    let mut pairs: Vec<(T, Vec<T>)> = values.into_iter().zip(vectors).collect();
    pairs.sort_by(|a, b| order(a.0, b.0));
    pairs.into_iter().unzip()
}

fn order<T: Real>(a: T, b: T) -> Ordering {
    if T::all_lt(a, b) {
        Ordering::Less
    } else if T::all_lt(b, a) {
        Ordering::Greater
    } else {
        Ordering::Equal
    }
}

/// `P y` for a square `m × m` matrix `P`, row by row.
fn mat_vec<T: Real>(p: &[T], y: &[T], m: usize) -> Vec<T> {
    p.chunks(m).map(|row| dot(row, y)).collect()
}

/// A deterministic pseudo-random vector with every coefficient in `[0.5, 1.5)`, different for
/// each `seed`.
fn start_vector<T: Real>(n: usize, seed: u64) -> Vec<T> {
    let mut state =
        0x9e37_79b9_7f4a_7c15u64 ^ seed.wrapping_add(1).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            T::from_f64(0.5 + (state >> 11) as f64 / (1u64 << 53) as f64)
        })
        .collect()
}
