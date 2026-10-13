//! The sparse field map: couplings stored by row (compressed sparse rows).

extern crate std;

use super::{Field, zero};
use crate::bind::Of;
use crate::kind::{Extensor, Kind};
use crate::ops::Reverse;
use core::ops::{Add, Mul};
use std::vec::Vec;

/// A field map stored as its couplings: each a cell from one input site (its column) to one
/// output site (its row). Repeated couplings add. The cells are multivectors (they multiply the
/// field's values) or maps (they apply to them).
///
/// The couplings are kept by row (compressed sparse rows), in the order given within a row, so
/// that applying the map sums each row's contributions in that order.
///
/// ```
/// use gax::field::{Field, SparseMap};
/// use gax::pga3d::{Plane, Scalar};
///
/// // A string of three beads: its springs as the differences of neighbouring beads, and the
/// // stiffness `~ends * ends` with each bead also tied to its rest position. The beads'
/// // displacements are any kind with an identity slot; here planes' coefficients.
/// let s = Scalar::<(), f64>::new;
/// let ends = SparseMap::new(vec![s(-1.0), s(1.0), s(-1.0), s(1.0)], vec![0, 0, 1, 1], vec![0, 1, 1, 2], (2, 3));
/// let anchors = SparseMap::from_diagonal(&Field::from_fn(3, |_| s(1.0)));
/// let stiffness = &(&(&ends.reverse() * &ends) + &anchors) * Plane::slot();
/// // The sag under a unit load on the middle bead.
/// let load = Field::from_fn(3, |i| Plane::new(0.0, 0.0, if i == 1 { -1.0 } else { 0.0 }, 0.0));
/// let sag = stiffness.solve(&load);
/// assert!((sag.site(1).e3() + 0.5).abs() < 1e-14);
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct SparseMap<C> {
    /// The cells, row by row.
    pub(super) cells: Vec<C>,
    /// The column of each cell.
    pub(super) cols: Vec<usize>,
    /// Row `r`'s cells are `cells[starts[r]..starts[r + 1]]`.
    pub(super) starts: Vec<usize>,
    /// The numbers of output and input sites.
    pub(super) shape: (usize, usize),
}

impl<C: Copy> SparseMap<C> {
    /// Couplings `cells[k]` from input site `cols[k]` to output site `rows[k]`, `shape` the
    /// numbers of output and input sites. The couplings may come in any order and may repeat.
    ///
    /// # Panics
    /// If the three lists differ in length, or an index is outside `shape`.
    #[must_use]
    #[track_caller]
    pub fn new(cells: Vec<C>, rows: Vec<usize>, cols: Vec<usize>, shape: (usize, usize)) -> Self {
        assert!(
            cells.len() == rows.len() && cells.len() == cols.len(),
            "SparseMap::new: {} cells, {} rows, {} columns",
            cells.len(),
            rows.len(),
            cols.len()
        );
        if let Some(r) = rows.iter().find(|&&r| r >= shape.0) {
            panic!("SparseMap::new: row {r} outside {} output sites", shape.0);
        }
        if let Some(c) = cols.iter().find(|&&c| c >= shape.1) {
            panic!("SparseMap::new: column {c} outside {} input sites", shape.1);
        }
        Self::from_triples(
            cells
                .into_iter()
                .zip(rows)
                .zip(cols)
                .map(|((c, r), k)| (r, k, c)),
            shape,
        )
    }

    /// From `(row, column, cell)` triples within `shape`, kept in order within each row.
    pub(super) fn from_triples(
        triples: impl IntoIterator<Item = (usize, usize, C)>,
        shape: (usize, usize),
    ) -> Self {
        let mut triples: Vec<(usize, usize, C)> = triples.into_iter().collect();
        // Stable: the order within a row is kept.
        triples.sort_by_key(|t| t.0);
        let mut starts = Vec::with_capacity(shape.0 + 1);
        let mut k = 0;
        for r in 0..=shape.0 {
            while k < triples.len() && triples[k].0 < r {
                k += 1;
            }
            starts.push(k);
        }
        SparseMap {
            cols: triples.iter().map(|t| t.1).collect(),
            cells: triples.into_iter().map(|t| t.2).collect(),
            starts,
            shape,
        }
    }

    /// Each value of a field on its own site, as a cell: weights such as masses or areas,
    /// applied where they belong.
    #[must_use]
    pub fn from_diagonal(field: &Field<C>) -> Self
    where
        C: Extensor,
    {
        let n = field.len();
        SparseMap {
            cells: field.sites().to_vec(),
            cols: (0..n).collect(),
            starts: (0..=n).collect(),
            shape: (n, n),
        }
    }

    /// The numbers of output and input sites.
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        self.shape
    }

    /// The number of stored couplings.
    #[must_use]
    pub fn couplings_len(&self) -> usize {
        self.cells.len()
    }

    /// The stored couplings, row by row: `(row, column, cell)`.
    pub fn couplings(&self) -> impl Iterator<Item = (usize, usize, C)> + '_ {
        (0..self.shape.0).flat_map(move |r| {
            (self.starts[r]..self.starts[r + 1]).map(move |k| (r, self.cols[k], self.cells[k]))
        })
    }

    /// `f` applied to every cell, couplings kept.
    #[must_use]
    pub fn map_cells<D: Copy>(&self, f: impl FnMut(C) -> D) -> SparseMap<D> {
        SparseMap {
            cells: self.cells.iter().copied().map(f).collect(),
            cols: self.cols.clone(),
            starts: self.starts.clone(),
            shape: self.shape,
        }
    }

    /// Every coupling run the other way, from its row to its column, with `f` on its cell.
    #[must_use]
    pub fn transpose_with<D: Copy>(&self, mut f: impl FnMut(C) -> D) -> SparseMap<D> {
        SparseMap::from_triples(
            self.couplings().map(|(r, c, cell)| (c, r, f(cell))),
            (self.shape.1, self.shape.0),
        )
    }

    /// The reverse: the couplings run back and each cell reversed, so that
    /// `~(ends * positions) == ~positions * ~ends` (numga's `~` on a sparse extensor).
    #[must_use]
    pub fn reverse(&self) -> SparseMap<C>
    where
        C: Reverse<Output = C>,
    {
        self.transpose_with(Reverse::reverse)
    }

    /// The map cells applied to a field: each cell to the value at its column, summed at the
    /// rows (the block matrix-vector product).
    ///
    /// # Panics
    /// If `x` is not over the map's input sites.
    #[must_use]
    #[track_caller]
    pub fn apply<X, Y>(&self, x: &Field<X>) -> Field<Y>
    where
        C: Of<X, Output = Y>,
        X: Extensor,
        Y: Extensor + Add<Output = Y>,
    {
        self.combine(x, Of::of)
    }

    /// Each cell combined with the value at its column, summed at the rows.
    #[track_caller]
    fn combine<X: Extensor, Y: Extensor + Add<Output = Y>>(
        &self,
        x: &Field<X>,
        mut f: impl FnMut(C, X) -> Y,
    ) -> Field<Y> {
        self.check_input(x.len());
        let x = x.sites();
        Field::from_fn(self.shape.0, |r| {
            let mut acc = zero::<Y>();
            for k in self.starts[r]..self.starts[r + 1] {
                acc = acc + f(self.cells[k], x[self.cols[k]]);
            }
            acc
        })
    }

    #[track_caller]
    pub(super) fn check_input(&self, n: usize) {
        assert!(
            n == self.shape.1,
            "a field map with {} input sites applied to a field over {n} sites",
            self.shape.1,
        );
    }

    /// Couplings along every path through the shared sites, `f` on each pair of cells: one
    /// cell per pair (`self` from site `k` to `r`, `other` from `c` to `k`) at `(r, c)`, the
    /// pairs that meet at the same `(r, c)` summed.
    #[track_caller]
    fn chain<D: Copy, E: Extensor + Add<Output = E>>(
        &self,
        other: &SparseMap<D>,
        mut f: impl FnMut(C, D) -> E,
    ) -> SparseMap<E> {
        assert!(
            self.shape.1 == other.shape.0,
            "composing field maps over different sites: {} input sites against {} output sites",
            self.shape.1,
            other.shape.0
        );
        let mut cells = Vec::new();
        let mut cols = Vec::new();
        let mut starts = Vec::with_capacity(self.shape.0 + 1);
        // One dense accumulator row over the output columns, with the columns touched so far.
        let mut acc: Vec<Option<E>> = (0..other.shape.1).map(|_| None).collect();
        let mut touched = Vec::new();
        for r in 0..self.shape.0 {
            starts.push(cells.len());
            for k in self.starts[r]..self.starts[r + 1] {
                let s = self.cols[k];
                for j in other.starts[s]..other.starts[s + 1] {
                    let c = other.cols[j];
                    let e = f(self.cells[k], other.cells[j]);
                    acc[c] = Some(if let Some(a) = acc[c] {
                        a + e
                    } else {
                        touched.push(c);
                        e
                    });
                }
            }
            touched.sort_unstable();
            for &c in &touched {
                cells.push(acc[c].take().expect("touched"));
                cols.push(c);
            }
            touched.clear();
        }
        starts.push(cells.len());
        SparseMap {
            cells,
            cols,
            starts,
            shape: (self.shape.0, other.shape.1),
        }
    }

    /// The map cells composed with another field map's along the paths through the shared
    /// sites: `self ∘ other`, one cell per pair of output and input sites that a path joins.
    ///
    /// # Panics
    /// If `other`'s output sites are not `self`'s input sites.
    #[must_use]
    #[track_caller]
    pub fn compose<D: Copy, E>(&self, other: &SparseMap<D>) -> SparseMap<E>
    where
        C: Of<D, Output = E>,
        E: Extensor + Add<Output = E>,
    {
        self.chain(other, Of::of)
    }
}

impl<C, A> SparseMap<C>
where
    C: Extensor<Slots = (A,)>,
    A: Kind,
{
    /// The transpose: the couplings run back and each map cell `K <- A` transposed to
    /// `A <- K` (its coefficient matrix transposed). It is the adjoint under the coefficient
    /// inner product of the fields ([`Field::dot`]): `y.dot(m.apply(x)) == m.transpose().apply(y).dot(x)`.
    #[must_use]
    pub fn transpose(&self) -> SparseMap<A::Mv<(C::Kind,), C::Coef>> {
        self.transpose_with(|cell| {
            let c = cell.coeffs().as_ref();
            Extensor::from_coeffs(A::arr_from_fn(|i| {
                <C::Kind as Kind>::arr_from_fn(|o| c[o].as_ref()[i])
            }))
        })
    }
}

impl<C: Copy + Add<Output = C>> Add for &SparseMap<C> {
    type Output = SparseMap<C>;
    /// The couplings of both: they add where they meet.
    ///
    /// # Panics
    /// If the shapes differ.
    #[track_caller]
    fn add(self, o: Self) -> SparseMap<C> {
        assert!(
            self.shape == o.shape,
            "adding field maps of different shapes: {:?} and {:?}",
            self.shape,
            o.shape
        );
        SparseMap::from_triples(self.couplings().chain(o.couplings()), self.shape)
    }
}

/// Multivector cells times a field: each cell's product with the value at its column, summed
/// at the rows.
impl<C, X, Y> Mul<&Field<X>> for &SparseMap<C>
where
    C: Copy + Mul<X, Output = Y>,
    X: Extensor,
    Y: Extensor + Add<Output = Y>,
{
    type Output = Field<Y>;
    #[track_caller]
    fn mul(self, x: &Field<X>) -> Field<Y> {
        self.combine(x, |c, v| c * v)
    }
}

/// Multivector cells times multivector cells, along the paths through the shared sites.
impl<C, D, E> Mul<&SparseMap<D>> for &SparseMap<C>
where
    C: Copy + Mul<D, Output = E>,
    D: Copy,
    E: Extensor + Add<Output = E>,
{
    type Output = SparseMap<E>;
    #[track_caller]
    fn mul(self, o: &SparseMap<D>) -> SparseMap<E> {
        self.chain(o, |a, b| a * b)
    }
}

/// Every cell times an extensor on the right: with an open slot, `&cells * Vector::slot()`,
/// the cells become maps.
impl<C, M, E> Mul<M> for &SparseMap<C>
where
    C: Copy + Mul<M, Output = E>,
    M: Extensor,
    E: Copy,
{
    type Output = SparseMap<E>;
    fn mul(self, m: M) -> SparseMap<E> {
        self.map_cells(|c| c * m)
    }
}
