//! Dense symmetric eigenproblems of run-time size: the whole spectrum of a field map, against a
//! metric that may be semidefinite.

extern crate std;

use super::solve::{Paired, unflatten};
use super::{CellPairing, Field, SparseMap};
use crate::coef::Real;
use crate::kind::{Extensor, Kind};
use core::cmp::Ordering;
use std::vec;
use std::vec::Vec;

/// The eigenvalues (ascending) and eigenvectors of the symmetric `m × m` matrix `a` (row-major):
/// `(θ, s)` with `s[j]` the unit eigenvector of `θ[j]`. Householder reduction to tridiagonal form
/// and the implicit QL method, after Numerical Recipes' `tred2` and `tqli`, kept in their
/// 1-based indexing.
pub(super) fn symmetric_eigen<T: Real>(a: &[T], m: usize) -> (Vec<T>, Vec<Vec<T>>) {
    if m == 0 {
        return (Vec::new(), Vec::new());
    }
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

fn order<T: Real>(a: T, b: T) -> Ordering {
    if T::all_lt(a, b) {
        Ordering::Less
    } else if T::all_lt(b, a) {
        Ordering::Greater
    } else {
        Ordering::Equal
    }
}

impl<C, A, B, T> SparseMap<C>
where
    C: Extensor<Kind = B, Slots = (A,), Coef = T> + CellPairing,
    A: Kind,
    B: Kind,
    T: Real,
{
    /// Every eigenpair of `self(x) = λ metric(x)`, by a dense solve: the whole spectrum,
    /// values ascending, fields orthonormal under the metric (in the cells' pairing,
    /// [`CellPairing`], as [`eigh`](Self::eigh)). For a few hundred to a few thousand unknowns:
    /// it assembles both maps as dense matrices (`n²` numbers each) and takes `O(n³)` steps;
    /// [`eigh`](Self::eigh) finds a few pairs nearest zero of a large sparse pencil.
    ///
    /// `self` must be symmetric and `metric` symmetric and semidefinite, of either sign. Where
    /// the metric is definite these are all the pairs. Where it is only semidefinite (an energy
    /// blind to gradients, a weight blind to directions), these are the finite pairs, as many as
    /// the metric's rank: on a direction the metric does not measure (its eigenvalues within
    /// `√ε` of its largest) the pencil has an infinite eigenvalue, deflated by solving `self`'s
    /// rows there for that direction's part of each eigenvector, or, where `self` vanishes there
    /// too (gradients carry no helicity), every value, and that direction is left out. numga's
    /// `eigh` on dense extensors.
    ///
    /// The method: the metric's own eigenvectors `v` with their values `μ`, the kept ones
    /// scaled to `v / √μ` (orthonormal in the metric), `self` projected on them less its
    /// coupling through the null space (a Schur complement), and that matrix's eigenvectors
    /// carried back; each dense step is Householder's reduction to a tridiagonal matrix and the
    /// implicit QL method.
    ///
    /// ```
    /// use gax::field::{Field, SparseMap};
    /// use gax::vga3d::{Scalar, Vector};
    /// // A free string of four beads displaced along a vector: every mode, the rigid shift
    /// // three times at zero, then 2 − 2 cos(kπ/4) for k = 1, 2, 3, each three times.
    /// let s = Scalar::<(), f64>::new;
    /// let ends = SparseMap::new(
    ///     (0..3).flat_map(|_| [s(-1.0), s(1.0)]).collect(),
    ///     (0..3).flat_map(|e| [e, e]).collect(),
    ///     (0..3).flat_map(|e| [e, e + 1]).collect(),
    ///     (3, 4),
    /// );
    /// let laplacian = &(&ends.reverse() * &ends) * Vector::slot();
    /// let mass = SparseMap::from_diagonal(&Field::from_fn(4, |_| Vector::<(), f64>::slot()));
    /// let (values, modes) = laplacian.eigh_all(&mass);
    /// assert_eq!((values.len(), modes.len()), (12, 12));
    /// for (i, v) in values.iter().enumerate() {
    ///     let k = (i / 3) as f64;
    ///     assert!((v - (2.0 - 2.0 * (k * std::f64::consts::FRAC_PI_4).cos())).abs() < 1e-14);
    /// }
    /// // The Laplacian as a metric: blind to the rigid shift, so nine pairs, the mass's
    /// // values over the Laplacian's.
    /// let (inverse, _) = mass.eigh_all(&laplacian);
    /// assert_eq!(inverse.len(), 9);
    /// assert!((inverse[8] - 1.0 / (2.0 - 2.0 * std::f64::consts::FRAC_PI_4.cos())).abs() < 1e-12);
    /// ```
    ///
    /// # Panics
    /// If the maps are not square over the same sites.
    #[must_use]
    #[track_caller]
    #[allow(clippy::type_complexity)]
    pub fn eigh_all(&self, metric: &SparseMap<C>) -> (Vec<T>, Vec<Field<A::Mv<(), T>>>) {
        assert!(
            self.shape.0 == self.shape.1 && metric.shape == self.shape && A::N == B::N,
            "eigh_all needs square field maps over the same sites: {:?} and {:?}",
            self.shape,
            metric.shape
        );
        let n = self.shape.0 * A::N;
        let mut paired = Paired::of::<C>(A::N);
        let trace = metric
            .paired_diagonal(&paired)
            .iter()
            .fold(T::zero(), |s, d| s + *d);
        if T::all_lt(trace, T::zero()) {
            paired = paired.negated();
        }
        let a = self.dense_paired(&paired);
        let m = metric.dense_paired(&paired);
        let (values, vectors) = eigh_on_range(&a, &m, n);
        let fields = vectors.iter().map(|x| unflatten(x)).collect();
        (values, fields)
    }

    /// `P A` as a dense symmetric matrix (its symmetric part), row-major.
    fn dense_paired(&self, paired: &Paired) -> Vec<T> {
        let na = A::N;
        let n = self.shape.0 * na;
        let mut d = vec![T::zero(); n * n];
        for (r, c, cell) in self.couplings() {
            let cell = cell.coeffs().as_ref();
            for i in 0..na {
                let (j, negative) = paired.row(i);
                for (k, &v) in cell[j].as_ref().iter().enumerate() {
                    let entry = &mut d[(r * na + i) * n + c * na + k];
                    *entry = if negative { *entry - v } else { *entry + v };
                }
            }
        }
        let half = T::from_f64(0.5);
        for i in 0..n {
            for j in 0..i {
                let v = (d[i * n + j] + d[j * n + i]) * half;
                d[i * n + j] = v;
                d[j * n + i] = v;
            }
        }
        d
    }
}

/// The finite eigenpairs of the symmetric `a x = λ m x` (row-major `n × n`, `m` semidefinite):
/// values ascending, vectors of unit length in `m`. `m`'s null directions are deflated: with `m`'s
/// range spanned by `B` (orthonormal in `m`) and its null space by `V₀`, `x = B y + V₀ z`, the
/// null rows `V₀ᵀ a (B y + V₀ z) = 0` give `z = −a₀₀⁺ V₀ᵀ a B y`, and `y` solves the Schur
/// complement `(Bᵀ a B − Bᵀ a V₀ a₀₀⁺ V₀ᵀ a B) y = λ y`. Where `a` vanishes on part of the null
/// space too (the pencil is singular there), the pseudo-inverse leaves that part out.
pub(super) fn eigh_on_range<T: Real>(a: &[T], m: &[T], n: usize) -> (Vec<T>, Vec<Vec<T>>) {
    let half = T::from_f64(0.5);
    let (mu, v) = symmetric_eigen(m, n);
    let largest = mu.iter().fold(T::zero(), |l, x| l.max(x.abs()));
    let tolerance = T::epsilon().sqrt() * largest;
    // The basis of the range, orthonormal in m (v / √μ), and the null space.
    let (mut basis, mut null) = (Vec::new(), Vec::new());
    for (mu, v) in mu.iter().zip(&v) {
        if T::all_lt(tolerance, *mu) {
            let scale = mu.sqrt().recip();
            basis.push(v.iter().map(|x| *x * scale).collect::<Vec<T>>());
        } else {
            null.push(v.clone());
        }
    }
    let apply = |x: &[T]| -> Vec<T> { a.chunks(n).map(|row| super::dot(row, x)).collect() };
    let gram = |left: &[Vec<T>], right: &[Vec<T>]| -> Vec<Vec<T>> {
        left.iter()
            .map(|l| right.iter().map(|r| super::dot(l, r)).collect())
            .collect()
    };
    let (r, k) = (basis.len(), null.len());
    let images: Vec<Vec<T>> = basis.iter().map(|b| apply(b)).collect();
    let null_images: Vec<Vec<T>> = null.iter().map(|z| apply(z)).collect();
    // Bᵀ a B, V₀ᵀ a B (k × r), and V₀ᵀ a V₀.
    let a11 = gram(&basis, &images);
    let a01 = gram(&null, &images);
    let a00 = gram(&null, &null_images);
    // a₀₀⁺ through its eigenvectors, the values within √ε of a's scale dropped.
    let flat00: Vec<T> = (0..k * k)
        .map(|i| (a00[i / k][i % k] + a00[i % k][i / k]) * half)
        .collect();
    let (theta, w) = symmetric_eigen(&flat00, k);
    let scale = a.chunks(n).fold(T::zero(), |l, row| {
        l.max(row.iter().fold(T::zero(), |s, x| s + x.abs()))
    });
    let kept: Vec<(T, &Vec<T>)> = theta
        .iter()
        .zip(&w)
        .filter(|(t, _)| T::all_lt(T::epsilon().sqrt() * scale, t.abs()))
        .map(|(t, w)| (*t, w))
        .collect();
    // c = W₊ᵀ (V₀ᵀ a B), so that a₀₁ᵀ a₀₀⁺ a₀₁ = cᵀ diag(1/θ) c.
    let c: Vec<Vec<T>> = kept
        .iter()
        .map(|(_, w)| {
            (0..r)
                .map(|j| (0..k).fold(T::zero(), |s, i| w[i].mul_add(a01[i][j], s)))
                .collect()
        })
        .collect();
    let mut reduced = vec![T::zero(); r * r];
    for i in 0..r {
        for j in 0..=i {
            let schur = kept
                .iter()
                .zip(&c)
                .fold(T::zero(), |s, ((t, _), c)| s + c[i] * c[j] / *t);
            let v = (a11[i][j] + a11[j][i]) * half - schur;
            reduced[i * r + j] = v;
            reduced[j * r + i] = v;
        }
    }
    let (values, y) = symmetric_eigen(&reduced, r);
    let vectors = y
        .iter()
        .map(|y| {
            let mut x = vec![T::zero(); n];
            for (b, &yi) in basis.iter().zip(y) {
                super::axpy(yi, b, &mut x);
            }
            // z = −V₀ W₊ diag(1/θ) c y.
            for ((t, w), c) in kept.iter().zip(&c) {
                let coefficient = -super::dot(c, y) / *t;
                for (wi, z) in w.iter().zip(&null) {
                    super::axpy(coefficient * *wi, z, &mut x);
                }
            }
            x
        })
        .collect();
    (values, vectors)
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
