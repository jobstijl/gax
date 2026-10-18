//! Fields: one object spread over sites, and sparse field maps that couple the sites (ADR-046).
//!
//! A batch holds independent copies; a [`Field`] is one object with a value at each site: the
//! vertices of a mesh, the beads of a string, the atoms of a lattice. A [`SparseMap`] couples
//! the sites. It stores only the couplings that are there, each a typed cell from one input
//! site (its column) to one output site (its row), as numga's `SparseExtensor`. The cell type is
//! checked at compile time: a map cell `Vector <- Vector` takes a field of vectors, nothing
//! else. The site counts are checked at run time, with a panic that names both.
//!
//! * **Multivector cells multiply.** `&ends * &positions` takes each cell's product with the
//!   value at its column and sums at the rows; `&ends.reverse() * &ends` multiplies the cells
//!   along every path through the shared sites (a mesh Laplacian); `&cells * Vector::slot()`
//!   turns the cells into maps.
//! * **Maps are built** from couplings ([`SparseMap::from_triples`], [`SparseMap::new`]), from the input sites of each row
//!   ([`SparseMap::from_columns`]: a mesh's edges or faces), or on the diagonal
//!   ([`SparseMap::from_diagonal`]).
//! * **Map cells apply** ([`SparseMap::of`], or [`SparseMap::of_into`] a kept field, with
//!   no allocation), compose ([`SparseMap::compose`]), transpose ([`SparseMap::transpose`]) and
//!   solve: [`SparseMap::solve`] (preconditioned conjugate gradients, for symmetric positive
//!   definite couplings), [`SparseMap::cg`] (the same with a report), [`SparseMap::solve_direct`]
//!   and [`SparseMap::factor`] (sparse LU after a minimum-degree ordering, for any nonsingular
//!   map; the factors solve again for every right-hand side), [`SparseMap::lstsq`] (least
//!   squares, LSQR, any shape) and [`SparseMap::eigh_with`] (the eigenpairs nearest zero of a
//!   pencil).
//! * **Batch kernels** (feature `batch`): `SparseMap::packed` lays the cells out for SIMD
//!   once, so that a map applied many times (a stiffness inside an iteration) runs on lanes.
//! * **Derivatives** with respect to a whole field: [`gradient`] and [`hessian`] for any scalar
//!   function of a field; for energies that are sums of local terms, the chain rule through a
//!   sparse map is far cheaper (see [`hessian`]).
//!
//! ```
//! use gax::field::{Field, SparseMap};
//! use gax::pga3d::{Point, Scalar};
//!
//! // The differences of neighbouring points on a line of three: two couplings per edge.
//! let s = |x: f64| Scalar::<(), f64>::new(x);
//! let ends = SparseMap::new(vec![s(-1.0), s(1.0), s(-1.0), s(1.0)], vec![0, 0, 1, 1], vec![0, 1, 1, 2], (2, 3));
//! let points: Field<_> = [0.0, 1.0, 3.0].iter().map(|&x| Point::xyz(x, 0.0, 0.0)).collect();
//! let steps = &ends * &points;
//! assert_eq!(steps.site(1), Point::xyz(3.0, 0.0, 0.0) - Point::xyz(1.0, 0.0, 0.0));
//! ```

extern crate std;

mod derivative;
mod direct;
mod eigen;
mod lsqr;
#[cfg(feature = "batch")]
mod packed;
mod solve;
mod sparse;

pub use derivative::{gradient, hessian};
pub use direct::{Singular, SparseLu};
#[cfg(feature = "batch")]
pub use packed::PackedSparseMap;
pub use solve::Solved;
pub use sparse::SparseMap;

use crate::coef::{Coef, Real};
use crate::kind::Extensor;
use crate::slots::fold_all_pairs;
use core::ops::{Add, Neg, Sub};
use std::vec::Vec;

pub(crate) use crate::kind::{scale, zero};

/// One object spread over sites: a value (or map, or form) of type `M` at each site.
///
/// Operations on elements act site by site through [`map`](Self::map) and
/// [`zip_with`](Self::zip_with); `+`, `-` and unary `-` act on fields over the same sites.
/// [`sum`](Self::sum) and [`mean`](Self::mean) reduce over the sites.
///
/// ```
/// use gax::field::Field;
/// use gax::pga3d::{Motor, Point};
/// let beads = Field::from_fn(4, |i| Point::<(), f64>::xyz(i as f64, 0.0, 0.0));
/// let turn = Motor::rotation_about(0.0, 0.0, 1.0, 0.5);
/// let turned = beads.map(|p| turn >> p); // the whole string, turned
/// assert_eq!(turned.len(), 4);
/// assert_eq!(beads.mean().to_euclidean(), [1.5, 0.0, 0.0]);
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct Field<M> {
    sites: Vec<M>,
}

impl<M: Extensor> Field<M> {
    /// A field from its values, one per site.
    #[must_use]
    pub fn new(sites: Vec<M>) -> Self {
        Field { sites }
    }

    /// A field over `n` sites, `f(i)` at site `i`.
    pub fn from_fn(n: usize, f: impl FnMut(usize) -> M) -> Self {
        Field {
            sites: (0..n).map(f).collect(),
        }
    }

    /// Zero at each of `n` sites.
    #[must_use]
    pub fn zeros(n: usize) -> Self {
        Self::from_fn(n, |_| zero())
    }

    /// The number of sites.
    #[must_use]
    pub fn len(&self) -> usize {
        self.sites.len()
    }

    /// Whether the field has no sites.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.sites.is_empty()
    }

    /// The values, site by site.
    #[must_use]
    pub fn sites(&self) -> &[M] {
        &self.sites
    }

    /// The values, site by site, to change in place.
    #[must_use]
    pub fn sites_mut(&mut self) -> &mut [M] {
        &mut self.sites
    }

    /// The values, site by site, as a `Vec`.
    #[must_use]
    pub fn into_sites(self) -> Vec<M> {
        self.sites
    }

    /// The value at site `i`.
    ///
    /// # Panics
    /// If `i` is not a site.
    #[must_use]
    pub fn site(&self, i: usize) -> M {
        self.sites[i]
    }

    /// The values, site by site.
    pub fn iter(&self) -> impl Iterator<Item = M> + '_ {
        self.sites.iter().copied()
    }

    /// `f` applied at every site: an operation on elements acts on a field site by site.
    pub fn map<N: Extensor>(&self, f: impl FnMut(M) -> N) -> Field<N> {
        Field {
            sites: self.sites.iter().copied().map(f).collect(),
        }
    }

    /// `f` applied to the values of two fields over the same sites, site by site.
    ///
    /// # Panics
    /// If the fields have different numbers of sites.
    pub fn zip_with<N: Extensor, P: Extensor>(
        &self,
        other: &Field<N>,
        mut f: impl FnMut(M, N) -> P,
    ) -> Field<P> {
        same_sites(self.len(), other.len());
        Field {
            sites: self
                .sites
                .iter()
                .zip(&other.sites)
                .map(|(a, b)| f(*a, *b))
                .collect(),
        }
    }

    /// The sum over the sites (numga's `.sites.sum()`): zero for no sites.
    #[must_use]
    pub fn sum(&self) -> M
    where
        M: Add<Output = M>,
    {
        self.sites.iter().fold(zero(), |a, b| a + *b)
    }

    /// The mean over the sites (numga's `.sites.mean()`): the sum scaled by `1 / len`.
    ///
    /// # Panics
    /// If the field has no sites.
    #[must_use]
    pub fn mean(&self) -> M
    where
        M: Add<Output = M>,
        M::Coef: Real,
    {
        assert!(!self.is_empty(), "the mean of a field with no sites");
        let n = M::Coef::from_f64(self.len() as f64);
        scale(self.sum(), n.recip())
    }

    /// Every coefficient at every site scaled by `c`.
    #[must_use]
    pub fn scaled(&self, c: M::Coef) -> Self {
        self.map(|m| scale(m, c))
    }

    /// The coefficient inner product summed over the sites, `Σ_sites Σ_k a_k b_k`: the
    /// Euclidean inner product of the fields' coefficient vectors.
    ///
    /// # Panics
    /// If the fields have different numbers of sites.
    #[must_use]
    pub fn dot(&self, other: &Self) -> M::Coef {
        same_sites(self.len(), other.len());
        self.sites
            .iter()
            .zip(&other.sites)
            .fold(M::Coef::zero(), |sum, (a, b)| {
                let (a, b) = (a.coeffs().as_ref(), b.coeffs().as_ref());
                fold_all_pairs::<M::Slots, _, _>(a, b, sum, |sum, x, y| x.mul_add(y, sum))
            })
    }
}

impl<M: Extensor<Coef: Real>> Field<M> {
    /// The norm of the fields' coefficient vector, `√(self · self)`: the size [`dot`](Self::dot)
    /// measures.
    ///
    /// ```
    /// use gax::field::Field;
    /// use gax::pga3d::Plane;
    /// let f = Field::new(vec![Plane::<(), f64>::new(3.0, 0.0, 0.0, 0.0), Plane::new(0.0, 4.0, 0.0, 0.0)]);
    /// assert_eq!(f.norm(), 5.0);
    /// ```
    #[must_use]
    pub fn norm(&self) -> M::Coef {
        self.dot(self).sqrt()
    }
}

/// `Σ aᵢ bᵢ` of two coefficient vectors, with four independent accumulators, so that the
/// multiply-adds overlap instead of waiting on one another.
pub(crate) fn dot<T: Real>(a: &[T], b: &[T]) -> T {
    let mut s = [T::zero(); 4];
    let (ca, ra) = a.as_chunks::<4>();
    let (cb, rb) = b.as_chunks::<4>();
    for (x, y) in ca.iter().zip(cb) {
        for ((s, x), y) in s.iter_mut().zip(x).zip(y) {
            *s = x.mul_add(*y, *s);
        }
    }
    let t = (s[0] + s[1]) + (s[2] + s[3]);
    ra.iter().zip(rb).fold(t, |t, (x, y)| x.mul_add(*y, t))
}

/// `y ← a x + y` on coefficient vectors.
pub(crate) fn axpy<T: Real>(a: T, x: &[T], y: &mut [T]) {
    for (yi, xi) in y.iter_mut().zip(x) {
        *yi = a.mul_add(*xi, *yi);
    }
}

#[track_caller]
pub(crate) fn same_sites(a: usize, b: usize) {
    assert!(
        a == b,
        "fields over different sites: {a} sites against {b}; a site-by-site operation needs the same sites"
    );
}

impl<M: Extensor> FromIterator<M> for Field<M> {
    /// A field with the values in order, one per site.
    fn from_iter<I: IntoIterator<Item = M>>(iter: I) -> Self {
        Field {
            sites: iter.into_iter().collect(),
        }
    }
}

/// `+` and `-` site by site, for every combination of owned and borrowed fields.
macro_rules! site_by_site {
    ($Op:ident, $op:ident) => {
        impl<M: Extensor + $Op<Output = M>> $Op<&Field<M>> for &Field<M> {
            type Output = Field<M>;
            /// Site by site.
            ///
            /// # Panics
            /// If the fields have different numbers of sites.
            #[track_caller]
            fn $op(self, o: &Field<M>) -> Field<M> {
                self.zip_with(o, |a, b| a.$op(b))
            }
        }
        impl<M: Extensor + $Op<Output = M>> $Op<Field<M>> for &Field<M> {
            type Output = Field<M>;
            #[track_caller]
            fn $op(self, o: Field<M>) -> Field<M> {
                self.$op(&o)
            }
        }
        impl<M: Extensor + $Op<Output = M>> $Op<&Field<M>> for Field<M> {
            type Output = Field<M>;
            #[track_caller]
            fn $op(self, o: &Field<M>) -> Field<M> {
                (&self).$op(o)
            }
        }
        impl<M: Extensor + $Op<Output = M>> $Op<Field<M>> for Field<M> {
            type Output = Field<M>;
            #[track_caller]
            fn $op(self, o: Field<M>) -> Field<M> {
                (&self).$op(&o)
            }
        }
    };
}
site_by_site!(Add, add);
site_by_site!(Sub, sub);

impl<M: Extensor + Neg<Output = M>> Neg for &Field<M> {
    type Output = Field<M>;
    /// Site by site.
    fn neg(self) -> Field<M> {
        self.map(|a| -a)
    }
}

impl<M: Extensor + Neg<Output = M>> Neg for Field<M> {
    type Output = Field<M>;
    /// Site by site.
    fn neg(self) -> Field<M> {
        -&self
    }
}
