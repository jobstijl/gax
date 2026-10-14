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
    /// (the rigid motions of a free mesh, a constant potential) then comes first. numga's sparse
    /// `eigh`.
    ///
    /// The search space is built by block Lanczos on `(self − σ metric)⁻¹ metric`, the spectrum
    /// inverted about a shift `σ` just below zero (`−√ε` relative to the pencil's scale), so that
    /// the eigenvectors nearest zero dominate it; each step solves with one sparse LU of
    /// `self − σ metric`. The eigenpairs are then taken from the pencil itself on that space
    /// (Rayleigh–Ritz with `self` and `metric`, which need no solves), and the space grows until
    /// each pair's true residual `‖self(x) − λ metric(x)‖` is within `ε^¾` of the pencil's scale.
    /// So the solves near the shift, which lose half their digits where the null space is
    /// amplified, shape the space but do not limit the accuracy. A last step of inverse iteration
    /// on the converged pairs themselves then sharpens the eigenvectors, whose error is the
    /// residual over the gap to the next eigenvalue: on a fine mesh, where that gap is a small
    /// part of the scale, it gains about a digit.
    ///
    /// The blocks have `2 K::N + 2` vectors, so an eigenvalue repeated on every coefficient of
    /// the kind (a field map of isotropic cells repeats each one `K::N` times), and doubled by a
    /// symmetry of the sites, is found with its whole eigenspace
    /// ([`eigh_with_block`](Self::eigh_with_block) for more).
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
        self.eigh_with_block(metric, count, 2 * K::N + 2)
    }

    /// [`eigh`](Self::eigh) with Lanczos blocks of `block` vectors. A block finds an eigenvalue
    /// with up to `block` copies at once; the default, `2 K::N + 2`, covers a value repeated on
    /// every coefficient of the kind and doubled by a symmetry of the sites (a square grid's
    /// modes along its two axes). Use a larger block for more symmetric meshes.
    ///
    /// # Panics
    /// As [`eigh`](Self::eigh), and if `block` is zero.
    #[must_use]
    #[track_caller]
    #[allow(clippy::type_complexity)]
    pub fn eigh_with_block(
        &self,
        metric: &SparseMap<C>,
        count: usize,
        block: usize,
    ) -> (Vec<T>, Vec<Field<K::Mv<(), T>>>) {
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
        let zero = T::zero();
        let largest = |d: Vec<T>| d.into_iter().fold(zero, |m, x| m.max(x.abs()));
        let (a_scale, m_scale) = (
            largest(self.diagonal_flat()),
            largest(metric.diagonal_flat()),
        );
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
        let mut progress = Progress::new(a_scale + m_scale);
        let mut reshifts = 0;
        loop {
            let (start, end) = space.last_block();
            // The next directions: (A − σM)⁻¹ M q for each vector of the last block.
            let images: Vec<Vec<T>> = (start..end)
                .map(|j| lu.solve_flat(&space.metric_images[j]))
                .collect();
            space.set_remainder(images);
            let m = space.len();
            if m >= count {
                let pairs = space.ritz(count);
                if m >= n
                    || progress.done(&pairs, |value, coefficients| {
                        space.residual(value, coefficients) / (T::one() + value.abs())
                    })
                {
                    let pairs = space.polish(&pairs, count, |x: &[T]| lu.solve_flat(x));
                    return space.finish::<K>(pairs);
                }
                // A spectrum that starts well above zero: shift to just below the wanted
                // cluster, so that it dominates the inverted operator. The space is kept, since
                // the pairs come from the pencil whatever shift built it. Each new shift is at
                // least twice as close to the cluster as the last, so a factorization is spent
                // only when the estimates have moved enough.
                let (low, high) = (pairs[0].0, pairs[pairs.len() - 1].0);
                let spread = (high - low).max(T::epsilon() * high.abs());
                let target = low - spread;
                if reshifts < 16
                    && T::all_lt(T::zero(), target)
                    && T::all_lt(sigma, target)
                    && T::all_lt(spread, (low - sigma) * T::from_f64(0.5))
                {
                    sigma = target;
                    lu = factor_shifted(self, metric, sigma);
                    reshifts += 1;
                }
            }
            if !space.grow(p) {
                // The space is the whole space: the pairs are exact.
                let pairs = space.ritz(count);
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

/// When to stop growing the space: the worst residual within `ε^¾` of the pencil's scale, or
/// within `√ε` once it has stopped improving (no halving over two checks). The residuals have
/// a floor of rounding, near `10⁵ ε` of the scale on small dense spectra; past it, the space
/// would grow to the whole space for nothing.
struct Progress<T> {
    strict: T,
    loose: T,
    best: T,
    stalled: usize,
}

impl<T: Real> Progress<T> {
    fn new(scale: T) -> Self {
        let eps = T::epsilon();
        Progress {
            strict: eps.sqrt() * eps.sqrt().sqrt() * scale,
            loose: eps.sqrt() * scale,
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
            remainder: Vec::new(),
            block_start: 0,
            seed: 0,
        }
    }

    fn len(&self) -> usize {
        self.basis.len()
    }

    fn last_block(&self) -> (usize, usize) {
        (self.block_start, self.basis.len())
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

    fn set_remainder(&mut self, images: Vec<Vec<T>>) {
        self.remainder = images;
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

    /// The `count` Ritz pairs nearest zero of the pencil on the space: the eigenpairs of
    /// `Qᵀ A Q` (the basis is orthonormal in the metric, so `Qᵀ M Q = I`), as `(λ, x)`, nearest
    /// first.
    fn ritz(&self, count: usize) -> Vec<(T, Vec<T>)> {
        let m = self.basis.len();
        let mut k = vec![T::zero(); m * m];
        for (i, row) in self.projected.iter().enumerate() {
            for (j, &v) in row.iter().enumerate() {
                k[i * m + j] = v;
                k[j * m + i] = v;
            }
        }
        let (values, vectors) = symmetric_eigen(&k, m);
        let mut nearest: Vec<usize> = (0..m).collect();
        nearest.sort_by(|&i, &j| order(values[i].abs(), values[j].abs()));
        nearest
            .into_iter()
            .take(count)
            .map(|i| (values[i], vectors[i].clone()))
            .collect()
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
    /// `(A − σM)⁻¹ M x` of each pair's vector joins the space, and the pairs are taken again.
    /// The stopping residual bounds an eigenvector's error only by the residual over the gap to
    /// the next eigenvalue, which on a fine mesh is a small part of the scale; the step shrinks
    /// each error component by `|λ − σ| / |λⱼ − σ|`, the ratio of the wanted eigenvalue's
    /// distance from the shift to the other's. The Rayleigh–Ritz over the larger space cannot
    /// make a pair worse.
    fn polish(
        &mut self,
        pairs: &[(T, Vec<T>)],
        count: usize,
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
        self.ritz(count)
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
    let one = T::one();
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
    let _ = one;
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
            for w in theta.windows(2) {
                assert!(w[0] <= w[1]);
            }
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
