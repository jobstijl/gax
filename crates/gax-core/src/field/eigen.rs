//! Sparse generalized eigenproblems: shift-invert block Lanczos on the sparse LU, with the
//! Rayleigh–Ritz step on the pencil itself.

extern crate std;

use super::solve::unflatten;
use super::{Field, SparseMap};
use crate::coef::Real;
use crate::kind::{Extensor, Kind};
use core::cmp::Ordering;
use core::ops::Add;
use std::vec;
use std::vec::Vec;

/// How [`SparseMap::eigh_with`] searches; [`Default`] is what [`SparseMap::eigh`] uses.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EighOptions<T> {
    /// Vectors per Lanczos block (`None`: `2 K::N + 2`). A block finds an eigenvalue with up to
    /// `block` copies at once; the default covers a value repeated on every coefficient of the
    /// kind and doubled by a symmetry of the sites (a square grid's modes along its two axes).
    /// Use a larger block for more symmetric meshes.
    pub block: Option<usize>,
    /// The backward error each pair must reach, `‖A x − λ M x‖ / (‖A‖ + |λ| ‖M‖)` (`None`:
    /// `ε^¾`). An eigenvector's error is about its backward error over the gap to the next
    /// eigenvalue (both relative to the pencil's scale), so a mesh whose gaps are a small part
    /// of the scale may ask for less; a looser tolerance stops sooner. The residuals have a floor
    /// of rounding (near `10⁴ ε` to `10⁵ ε`): past it, the search stops where they stop
    /// improving.
    pub tolerance: Option<T>,
}

impl<T> Default for EighOptions<T> {
    fn default() -> Self {
        EighOptions {
            block: None,
            tolerance: None,
        }
    }
}

impl<C, K, T> SparseMap<C>
where
    C: Extensor<Kind = K, Slots = (K,), Coef = T> + Add<Output = C>,
    K: Kind,
    T: Real,
{
    /// The `count` eigenpairs nearest zero of `self(x) = λ metric(x)`: values ascending, and
    /// fields orthonormal under the metric (`xᵢ · metric(xⱼ) = δᵢⱼ`, the coefficient inner
    /// product). `self` must be symmetric and `metric` symmetric positive definite (a stiffness
    /// and a mass, a Laplacian and an area form). `self` may be semidefinite: its null space
    /// (the rigid motions of a free mesh, a constant potential) then comes first. It may also be
    /// indefinite (a Hamiltonian), the pairs nearest zero then lying on both sides. numga's
    /// sparse `eigh`.
    ///
    /// The search space is built by block Lanczos on `(self − σ metric)⁻¹ metric`, the spectrum
    /// inverted about a shift `σ` just below zero (`−√ε` relative to the pencil's scale), so that
    /// the eigenvectors nearest zero dominate it; each step solves with one sparse LU of
    /// `self − σ metric`. The eigenpairs are then taken from the pencil itself on that space
    /// (Rayleigh–Ritz with `self` and `metric`, which need no solves; for an indefinite pencil,
    /// chosen first on the inverted operator, where the wanted values are the largest), and the space grows until
    /// each pair's backward error, its true residual `‖self(x) − λ metric(x)‖` relative to
    /// `‖self‖ + |λ| ‖metric‖`, is within `ε^¾`.
    /// So the solves near the shift, which lose half their digits where the null space is
    /// amplified, shape the space but do not limit the accuracy. A last step of inverse iteration
    /// on the converged pairs themselves then sharpens the eigenvectors, whose error is the
    /// residual over the gap to the next eigenvalue: on a fine mesh, where that gap is a small
    /// part of the scale, it gains about a digit.
    ///
    /// The blocks have `2 K::N + 2` vectors, so an eigenvalue repeated on every coefficient of
    /// the kind (a field map of isotropic cells repeats each one `K::N` times), and doubled by a
    /// symmetry of the sites, is found with its whole eigenspace
    /// ([`eigh_with`](Self::eigh_with) for another block or tolerance).
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
    pub fn eigh(&self, metric: &SparseMap<C>, count: usize) -> (Vec<T>, Vec<Field<K::Mv<(), T>>>) {
        self.eigh_with(metric, count, EighOptions::default())
    }

    /// [`eigh`](Self::eigh) with the search's [`EighOptions`]: the Lanczos block's size, the
    /// backward error to reach.
    ///
    /// ```
    /// use gax::field::{EighOptions, Field, SparseMap};
    /// use gax::pga3d::Scalar;
    /// let s = Scalar::<(), f64>::new;
    /// let k = SparseMap::new(vec![s(2.0), s(-1.0), s(-1.0), s(2.0)], vec![0, 0, 1, 1], vec![0, 1, 0, 1], (2, 2));
    /// let k = &k * Scalar::slot(); // map cells
    /// let mass = SparseMap::from_diagonal(&Field::from_fn(2, |_| Scalar::<(), f64>::slot()));
    /// let options = EighOptions { block: Some(2), tolerance: Some(64.0 * f64::EPSILON) };
    /// let (values, _) = k.eigh_with(&mass, 2, options);
    /// assert!((values[0] - 1.0).abs() < 1e-14 && (values[1] - 3.0).abs() < 1e-14);
    /// ```
    ///
    /// # Panics
    /// As [`eigh`](Self::eigh), and if the block is zero vectors or the tolerance not positive.
    #[must_use]
    #[track_caller]
    #[allow(clippy::type_complexity)]
    pub fn eigh_with(
        &self,
        metric: &SparseMap<C>,
        count: usize,
        options: EighOptions<T>,
    ) -> (Vec<T>, Vec<Field<K::Mv<(), T>>>) {
        let block = options.block.unwrap_or(2 * K::N + 2);
        let tolerance = options
            .tolerance
            .unwrap_or_else(Progress::default_tolerance);
        assert!(
            T::all_lt(T::zero(), tolerance),
            "eigh: the tolerance must be positive"
        );
        assert!(block > 0, "eigh: a Lanczos block of zero vectors");
        assert!(
            self.shape.0 == self.shape.1 && metric.shape == self.shape,
            "eigh needs square field maps over the same sites: {:?} and {:?}",
            self.shape,
            metric.shape
        );
        let n = self.shape.0 * K::N;
        assert!(
            count > 0 && count <= n,
            "eigh: {count} eigenpairs asked of {n} unknowns"
        );
        // The scales: the ∞-norms, which bound the spectral radii. (Not the diagonals: a
        // Hamiltonian of hops alone has none.)
        let (a_scale, m_scale) = (self.row_sum_norm(), metric.row_sum_norm());
        // The shift, just below zero relative to the pencil's scale.
        let mut sigma = -(T::epsilon().sqrt() * a_scale / m_scale);
        let mut lu = factor_shifted(self, metric, sigma);
        let p = block.min(n);
        let mut space = Space::new(
            n,
            |x: &[T], y: &mut [T]| metric.apply_flat(x, y),
            |x: &[T], y: &mut [T]| self.apply_flat(x, y),
        );
        space.extend_random(p);
        let mut progress = Progress::new(tolerance);
        let mut reshifts = 0;
        // The space is checked when it has grown by a quarter since the last check: each check
        // costs a dense eigenproblem of the space's size, so the checks then cost a fixed
        // multiple of the last one.
        let mut next_check = count;
        loop {
            // The next directions: (A − σM)⁻¹ M q for each vector of the last block.
            space.image_last_block(|x| lu.solve_flat(x));
            // Past a re-shift the wanted cluster lies above the shift, not around it: a block
            // more of candidates keeps a value between zero and the shift in view.
            let extra = if reshifts > 0 { p } else { 0 };
            let m = space.len();
            if m >= next_check || m >= n {
                next_check = m + m / 4;
                let pairs = space.ritz(count, extra, |x| lu.solve_flat(x));
                if m >= n
                    || progress.done(&pairs, |value, coefficients| {
                        // The backward error: the residual relative to the pencil at λ.
                        space.residual(value, coefficients) / (a_scale + value.abs() * m_scale)
                    })
                {
                    let pairs = space.polish(&pairs, count, extra, |x| lu.solve_flat(x));
                    return space.finish::<K>(pairs);
                }
                // A spectrum that starts well above zero: shift to just below the wanted
                // cluster, so that it dominates the inverted operator. The space is kept and
                // its images are taken again with the new shift. Each new shift is at least
                // twice as close to the cluster as the last, so a factorization is spent only
                // when the estimates have moved enough. (The pairs are ordered by distance from
                // zero, not by value; an indefinite pencil, with wanted values on both sides of
                // zero, is never re-shifted.)
                let low = pairs.iter().fold(pairs[0].0, |m, p| m.min(p.0));
                let high = pairs.iter().fold(pairs[0].0, |m, p| m.max(p.0));
                let spread = (high - low).max(T::epsilon() * high.abs());
                let target = low - spread;
                if reshifts < 16
                    && T::all_lt(T::zero(), target)
                    && T::all_lt(sigma, target)
                    && T::all_lt(spread, (low - sigma) * T::from_f64(0.5))
                {
                    sigma = target;
                    lu = factor_shifted(self, metric, sigma);
                    space.images_stale = true;
                    reshifts += 1;
                }
            }
            if !space.grow(p) {
                // The space is the whole space: the pairs are exact.
                let extra = if reshifts > 0 { p } else { 0 };
                let pairs = space.ritz(count, extra, |x| lu.solve_flat(x));
                return space.finish::<K>(pairs);
            }
        }
    }
}

/// The sparse LU of `A − σ M`.
fn factor_shifted<C, K, T>(
    a: &SparseMap<C>,
    metric: &SparseMap<C>,
    sigma: T,
) -> super::SparseLu<K, T>
where
    C: Extensor<Kind = K, Slots = (K,), Coef = T> + Add<Output = C>,
    K: Kind,
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

impl<C, K, T> SparseMap<C>
where
    C: Extensor<Kind = K, Slots = (K,), Coef = T>,
    K: Kind,
    T: Real,
{
    /// The ∞-norm of the block matrix: its largest sum of absolute values along a row.
    fn row_sum_norm(&self) -> T {
        let mut largest = T::zero();
        for r in 0..self.shape.0 {
            let cells = &self.cells[self.starts[r]..self.starts[r + 1]];
            for o in 0..K::N {
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
/// in ∞-norms, within the tolerance (by default `ε^¾`), or within `√ε` once it has stopped
/// improving (no halving over two checks). The measure is the residual relative to the pencil
/// at `λ`, so it does not change when the pencil is scaled (a stiffness of `10¹¹`). The
/// residuals have a floor of rounding, near `10⁴ ε` to `10⁵ ε`; past it, the space would grow
/// to the whole space for nothing.
struct Progress<T> {
    strict: T,
    loose: T,
    best: T,
    stalled: usize,
}

impl<T: Real> Progress<T> {
    /// `ε^¾`.
    fn default_tolerance() -> T {
        let eps = T::epsilon();
        eps.sqrt() * eps.sqrt().sqrt()
    }

    /// Stop at `tolerance`, or at `√ε` (or the tolerance, if looser) once stalled.
    fn new(tolerance: T) -> Self {
        Progress {
            strict: tolerance,
            loose: T::epsilon().sqrt().max(tolerance),
            best: T::from_f64(f64::MAX),
            stalled: 0,
        }
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
    /// Whether the shift has moved since `inverted` was taken.
    images_stale: bool,
    /// Whether the pencil has shown a negative value on the space. A growing space only lowers
    /// its lowest Ritz value (by interlacing), so once indefinite it stays so.
    indefinite: bool,
    /// The directions that start the next block.
    remainder: Vec<Vec<T>>,
    /// The first vector of the last block.
    block_start: usize,
    seed: u64,
}

impl<T: Real, M: FnMut(&[T], &mut [T]), A: FnMut(&[T], &mut [T])> Space<T, M, A> {
    fn new(n: usize, metric: M, operator: A) -> Self {
        Space {
            n,
            metric,
            operator,
            basis: Vec::new(),
            metric_images: Vec::new(),
            operator_images: Vec::new(),
            projected: Vec::new(),
            inverted: Vec::new(),
            images_stale: false,
            indefinite: false,
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
                let c = dot(mq, v);
                for (vi, qi) in v.iter_mut().zip(q) {
                    *vi = (-c).mul_add(*qi, *vi);
                }
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

    /// Start a block of `p` pseudo-random vectors.
    fn extend_random(&mut self, p: usize) {
        self.block_start = self.basis.len();
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
    /// Where the pencil is semidefinite on the space, the values nearest zero are its lowest,
    /// and Rayleigh–Ritz with `A` (the basis is orthonormal in the metric, so `Qᵀ M Q = I`)
    /// finds them best. Inside an indefinite spectrum (a Hamiltonian's levels around its gap)
    /// it does not: it finds spurious values near zero, mixtures of eigenvectors on both sides.
    /// There the pairs are chosen on the inverted operator, where the wanted eigenvalues are the
    /// largest, and the `count + extra` vectors of its largest values are rotated by
    /// Rayleigh–Ritz with `A` on their span, whose values are exact to rounding however the
    /// images were rounded.
    fn ritz(
        &mut self,
        count: usize,
        extra: usize,
        solve: impl FnMut(&[T]) -> Vec<T>,
    ) -> Vec<(T, Vec<T>)> {
        let m = self.basis.len();
        let projected = self.projected_matrix();
        if !self.indefinite {
            let (values, vectors) = symmetric_eigen(&projected, m);
            let largest = values.iter().fold(T::zero(), |l, v| l.max(v.abs()));
            let lowest = values.iter().fold(T::zero(), |l, v| l.min(*v));
            self.indefinite = T::all_lt(lowest, -(T::epsilon().sqrt() * largest));
            if !self.indefinite {
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
        let mut largest: Vec<usize> = (0..m).collect();
        largest.sort_by(|&i, &j| order(inverted[j].abs(), inverted[i].abs()));
        let chosen: Vec<&Vec<T>> = largest
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
                    for (ci, yi) in c.iter_mut().zip(y.iter()) {
                        *ci = zi.mul_add(*yi, *ci);
                    }
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
            for (xi, qi) in x.iter_mut().zip(q) {
                *xi = c.mul_add(*qi, *xi);
            }
        }
        x
    }

    /// One step of inverse iteration on the converged pairs themselves: the image
    /// `(A − σM)⁻¹ M x` of each pair's vector joins the space as a block (imaged in turn), and
    /// the pairs are taken again. The stopping residual bounds an eigenvector's error only by
    /// the residual over the gap to the next eigenvalue, which on a fine mesh is a small part
    /// of the scale; the step shrinks each error component by `|λ − σ| / |λⱼ − σ|`, the ratio
    /// of the wanted eigenvalue's distance from the shift to the other's.
    fn polish(
        &mut self,
        pairs: &[(T, Vec<T>)],
        count: usize,
        extra: usize,
        mut solve: impl FnMut(&[T]) -> Vec<T>,
    ) -> Vec<(T, Vec<T>)> {
        let images: Vec<Vec<T>> = pairs
            .iter()
            .map(|(_, c)| {
                let x = self.combine(c);
                solve(&self.apply_metric(&x))
            })
            .collect();
        self.block_start = self.basis.len();
        for y in images {
            if self.basis.len() >= self.n {
                break;
            }
            self.push(y);
        }
        self.image_last_block(&mut solve);
        self.ritz(count, extra, solve)
    }

    /// The pairs as fields, in ascending order of their values.
    #[allow(clippy::type_complexity)]
    fn finish<K: Kind>(&self, mut pairs: Vec<(T, Vec<T>)>) -> (Vec<T>, Vec<Field<K::Mv<(), T>>>) {
        pairs.sort_by(|a, b| order(a.0, b.0));
        let values = pairs.iter().map(|p| p.0).collect();
        let fields = pairs
            .iter()
            .map(|(_, c)| unflatten(&self.combine(c)))
            .collect();
        (values, fields)
    }

    /// `‖A x − λ M x‖` for `x = Σ cᵢ qᵢ`, from the basis' images (no products).
    fn residual(&self, value: T, coefficients: &[T]) -> T {
        let mut ax = vec![T::zero(); self.n];
        let mut mx = vec![T::zero(); self.n];
        for (&c, (aq, mq)) in coefficients
            .iter()
            .zip(self.operator_images.iter().zip(&self.metric_images))
        {
            for ((a, m), (aqi, mqi)) in ax.iter_mut().zip(mx.iter_mut()).zip(aq.iter().zip(mq)) {
                *a = c.mul_add(*aqi, *a);
                *m = c.mul_add(*mqi, *m);
            }
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

fn dot<T: Real>(a: &[T], b: &[T]) -> T {
    a.iter()
        .zip(b)
        .fold(T::zero(), |s, (x, y)| x.mul_add(*y, s))
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

/// The eigenvalues (ascending) and eigenvectors of the symmetric `m × m` matrix `a` (row-major):
/// `(θ, s)` with `s[j]` the unit eigenvector of `θ[j]`. Householder reduction to tridiagonal form
/// and the implicit QL method, after Numerical Recipes' `tred2` and `tqli`, kept in their
/// 1-based indexing.
fn symmetric_eigen<T: Real>(a: &[T], m: usize) -> (Vec<T>, Vec<Vec<T>>) {
    let zero = T::zero();
    // 1-based copies: a1[i][j] for i, j in 1..=m.
    let mut z: Vec<Vec<T>> = (0..=m)
        .map(|i| {
            (0..=m)
                .map(|j| {
                    if i == 0 || j == 0 {
                        zero
                    } else {
                        a[(i - 1) * m + j - 1]
                    }
                })
                .collect()
        })
        .collect();
    let mut d = vec![zero; m + 1];
    let mut e = vec![zero; m + 1];
    tred2(&mut z, &mut d, &mut e, m);
    tqli(&mut d, &mut e, &mut z, m);
    let mut idx: Vec<usize> = (1..=m).collect();
    idx.sort_by(|&i, &j| order(d[i], d[j]));
    let theta = idx.iter().map(|&i| d[i]).collect();
    let vectors = idx
        .iter()
        .map(|&i| (1..=m).map(|row| z[row][i]).collect())
        .collect();
    (theta, vectors)
}

/// Householder reduction of the symmetric `a` (1-based, `n × n`) to tridiagonal form: the
/// diagonal in `d`, the subdiagonal in `e[2..=n]`, and the orthogonal transformation in `a`.
#[allow(clippy::many_single_char_names, clippy::needless_range_loop)]
fn tred2<T: Real>(a: &mut [Vec<T>], d: &mut [T], e: &mut [T], n: usize) {
    let zero = T::zero();
    for i in (2..=n).rev() {
        let l = i - 1;
        let mut h = zero;
        if l > 1 {
            let mut scale = zero;
            for k in 1..=l {
                scale = scale + a[i][k].abs();
            }
            if scale == zero {
                e[i] = a[i][l];
            } else {
                for k in 1..=l {
                    a[i][k] = a[i][k] / scale;
                    h = a[i][k].mul_add(a[i][k], h);
                }
                let f = a[i][l];
                let g = if T::all_lt(f, zero) {
                    h.sqrt()
                } else {
                    -h.sqrt()
                };
                e[i] = scale * g;
                h = h - f * g;
                a[i][l] = f - g;
                let mut f = zero;
                for j in 1..=l {
                    a[j][i] = a[i][j] / h;
                    let mut g = zero;
                    for k in 1..=j {
                        g = a[j][k].mul_add(a[i][k], g);
                    }
                    for k in j + 1..=l {
                        g = a[k][j].mul_add(a[i][k], g);
                    }
                    e[j] = g / h;
                    f = e[j].mul_add(a[i][j], f);
                }
                let hh = f / (h + h);
                for j in 1..=l {
                    let f = a[i][j];
                    let g = e[j] - hh * f;
                    e[j] = g;
                    for k in 1..=j {
                        a[j][k] = a[j][k] - (f * e[k] + g * a[i][k]);
                    }
                }
            }
        } else {
            e[i] = a[i][l];
        }
        d[i] = h;
    }
    d[1] = zero;
    e[1] = zero;
    for i in 1..=n {
        let l = i - 1;
        if d[i] != zero {
            for j in 1..=l {
                let mut g = zero;
                for k in 1..=l {
                    g = a[i][k].mul_add(a[k][j], g);
                }
                for k in 1..=l {
                    a[k][j] = (-g).mul_add(a[k][i], a[k][j]);
                }
            }
        }
        d[i] = a[i][i];
        a[i][i] = T::one();
        for j in 1..=l {
            a[j][i] = zero;
            a[i][j] = zero;
        }
    }
}

/// The implicit QL method on the tridiagonal matrix of `tred2` (1-based): the eigenvalues in
/// `d`, and the eigenvectors in the columns of `z` (which starts as `tred2`'s transformation).
#[allow(clippy::many_single_char_names, clippy::needless_range_loop)]
fn tqli<T: Real>(d: &mut [T], e: &mut [T], z: &mut [Vec<T>], n: usize) {
    let zero = T::zero();
    let two = T::from_i64(2);
    for i in 2..=n {
        e[i - 1] = e[i];
    }
    e[n] = zero;
    for l in 1..=n {
        let mut iterations = 0;
        loop {
            let mut m = l;
            while m < n {
                let dd = d[m].abs() + d[m + 1].abs();
                if !T::all_lt(T::epsilon() * dd, e[m].abs()) {
                    break;
                }
                m += 1;
            }
            if m == l {
                break;
            }
            iterations += 1;
            if iterations > 60 {
                break;
            }
            let mut g = (d[l + 1] - d[l]) / (two * e[l]);
            let mut r = g.mul_add(g, T::one()).sqrt();
            let signed = if T::all_lt(g, zero) { -r } else { r };
            g = d[m] - d[l] + e[l] / (g + signed);
            let (mut s, mut c, mut p) = (T::one(), T::one(), zero);
            let mut i = m;
            let mut split = false;
            while i > l {
                i -= 1;
                let f = s * e[i];
                let b = c * e[i];
                r = f.mul_add(f, g * g).sqrt();
                e[i + 1] = r;
                if r == zero {
                    d[i + 1] = d[i + 1] - p;
                    e[m] = zero;
                    split = true;
                    break;
                }
                s = f / r;
                c = g / r;
                g = d[i + 1] - p;
                r = (d[i] - g) * s + two * c * b;
                p = s * r;
                d[i + 1] = g + p;
                g = c * r - b;
                for k in 1..=n {
                    let f = z[k][i + 1];
                    z[k][i + 1] = s * z[k][i] + c * f;
                    z[k][i] = c * z[k][i] - s * f;
                }
            }
            if split {
                continue;
            }
            d[l] = d[l] - p;
            e[l] = g;
            e[m] = zero;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::symmetric_eigen;
    extern crate std;
    use std::vec::Vec;

    /// Householder plus QL against the definition: `A v = θ v`, orthonormal, ascending.
    #[test]
    fn symmetric_eigen_is_an_eigendecomposition() {
        for m in [1usize, 2, 3, 6, 11] {
            let a: Vec<f64> = (0..m * m)
                .map(|k| {
                    let (i, j) = (k / m, k % m);
                    let (i, j) = (i.min(j), i.max(j));
                    ((i * 7 + j * 3) as f64 * 0.37).sin()
                })
                .collect();
            let (theta, s) = symmetric_eigen(&a, m);
            assert!(theta.windows(2).all(|w| w[0] <= w[1]));
            for (t, v) in theta.iter().zip(&s) {
                for i in 0..m {
                    let av: f64 = (0..m).map(|j| a[i * m + j] * v[j]).sum();
                    assert!((av - t * v[i]).abs() < 1e-12, "m {m}: row {i}");
                }
                let len: f64 = v.iter().map(|x| x * x).sum();
                assert!((len - 1.0).abs() < 1e-12);
            }
        }
    }
}
