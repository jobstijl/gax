//! The sparse field map: couplings stored by row (compressed sparse rows).

extern crate std;

use super::{Field, zero};
use crate::bind::Of;
use crate::kind::{Extensor, Kind};
use crate::ops::Reverse;
use crate::transpose::{Adjoint, Adjugate};
use core::ops::{Add, Mul, Neg, Sub};
use std::vec;
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
    /// The couplings `(row, column, cell)`, each from its input site (column) to its output site
    /// (row), `shape` the numbers of output and input sites. The couplings may come in any order
    /// and may repeat; within a row they keep their order.
    ///
    /// ```
    /// use gax::field::SparseMap;
    /// use gax::pga3d::Scalar;
    /// let s = Scalar::<(), f64>::new;
    /// let ends = SparseMap::from_triples([(0, 0, s(-1.0)), (0, 1, s(1.0)), (1, 1, s(-1.0)), (1, 2, s(1.0))], (2, 3));
    /// assert_eq!(ends.couplings_len(), 4);
    /// ```
    ///
    /// # Panics
    /// If a row or column is outside `shape`.
    #[must_use]
    #[track_caller]
    pub fn from_triples(
        triples: impl IntoIterator<Item = (usize, usize, C)>,
        shape: (usize, usize),
    ) -> Self {
        let mut triples: Vec<(usize, usize, C)> = triples.into_iter().collect();
        for &(r, c, _) in &triples {
            assert!(
                r < shape.0,
                "SparseMap: row {r} outside {} output sites",
                shape.0
            );
            assert!(
                c < shape.1,
                "SparseMap: column {c} outside {} input sites",
                shape.1
            );
        }
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

    /// Couplings `cells[k]` from input site `cols[k]` to output site `rows[k]`:
    /// [`from_triples`](Self::from_triples) with the three parts in their own lists.
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
        let triples = rows.into_iter().zip(cols).zip(cells);
        Self::from_triples(triples.map(|((r, c), cell)| (r, c, cell)), shape)
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

    /// The selection map reading a field over `len` sites at the given ones, in that order,
    /// `[indices] <- [len]`, cells one of a scalar kind (numga's `mesh.reading`): the free
    /// vertices of a boundary-value problem, the boundary edges. Its transpose puts values at
    /// those sites back onto all of them, zero elsewhere.
    ///
    /// ```
    /// use gax::field::{Field, SparseMap};
    /// use gax::vga3d::Scalar;
    /// let f: Field<_> = [5.0, 6.0, 7.0, 8.0].into_iter().map(Scalar::<(), f64>::new).collect();
    /// let pick = SparseMap::<Scalar<(), f64>>::selection(&[3, 1], 4);
    /// assert_eq!((&pick * &f).site(0).s(), 8.0);
    /// let back = &pick.transpose_with(|c| c) * &(&pick * &f);
    /// assert_eq!(back.iter().map(|x| x.s()).collect::<Vec<_>>(), [0.0, 6.0, 0.0, 8.0]);
    /// ```
    ///
    /// # Panics
    /// If an index is not below `len`.
    #[must_use]
    #[track_caller]
    pub fn selection(indices: &[usize], len: usize) -> Self
    where
        C: Extensor<Slots = ()>,
    {
        const {
            assert!(
                <C::Kind as Kind>::N == 1,
                "the cells of a selection map are scalars"
            );
        }
        let one = C::from_coeffs(<C::Kind as Kind>::arr_from_fn(|_| {
            <C::Coef as crate::coef::Coef>::one()
        }));
        let triples = indices.iter().enumerate().map(|(r, &i)| (r, i, one));
        Self::from_triples(triples, (indices.len(), len))
    }

    /// The field map with no couplings, `shape` the numbers of output and input sites: the
    /// zero block of a saddle-point system ([`BlockMap`](super::BlockMap)).
    #[must_use]
    pub fn zeros(shape: (usize, usize)) -> Self {
        SparseMap {
            cells: Vec::new(),
            cols: Vec::new(),
            starts: vec![0; shape.0 + 1],
            shape,
        }
    }

    /// The field map whose row `r` couples to the input sites `columns[r]`, its `k`-th through
    /// the cell `cell(r, k)`, over `inputs` input sites: numga's `SparseExtensor.from_columns`.
    /// A mesh's incidence maps are built this way, from its edges or faces.
    ///
    /// ```
    /// use gax::field::SparseMap;
    /// use gax::pga3d::Scalar;
    /// // Each edge's head less its tail, for the edges 0-1 and 1-2 of a chain.
    /// let edges = [[0, 1], [1, 2]];
    /// let d = SparseMap::from_columns(&edges, 3, |_, end| Scalar::<(), f64>::new([-1.0, 1.0][end]));
    /// assert_eq!(d.shape(), (2, 3));
    /// assert_eq!(d.couplings().count(), 4);
    /// ```
    ///
    /// # Panics
    /// If a column is not below `inputs`.
    #[must_use]
    #[track_caller]
    pub fn from_columns<const N: usize>(
        columns: &[[usize; N]],
        inputs: usize,
        mut cell: impl FnMut(usize, usize) -> C,
    ) -> Self {
        let triples = columns
            .iter()
            .enumerate()
            .flat_map(|(r, cols)| cols.iter().enumerate().map(move |(k, &c)| (r, k, c)))
            .map(|(r, k, c)| (r, c, cell(r, k)));
        SparseMap::from_triples(triples, (columns.len(), inputs))
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

    /// Every cell's coefficients scaled by `c`, couplings kept: `&map * c` for any coefficient
    /// type.
    #[must_use]
    pub fn scaled(&self, c: C::Coef) -> SparseMap<C>
    where
        C: Extensor,
    {
        self.map_cells(|cell| super::scale(cell, c))
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

    /// The adjugate: the couplings run back and each map cell's adjugate taken, so that
    /// `(m.adjugate().of(c) & x)₀` summed over the sites equals `(c & m.of(x))₀` summed
    /// (numga's `adjugate` on a sparse extensor; see [`Adjugate`]). It moves a field map on
    /// one kind to the complement kinds without a metric: the boundary of a mesh's points to
    /// its planes, a strain to its stress.
    ///
    /// ```
    /// use gax::field::{Field, SparseMap};
    /// use gax::pga3d::{Plane, Point, Scalar};
    /// let s = Scalar::<(), f64>::new;
    /// // Differences of neighbouring points along a chain, and the planes that pair with them.
    /// let ends = &SparseMap::from_triples([(0, 0, s(-1.0)), (0, 1, s(1.0)), (1, 1, s(-1.0)), (1, 2, s(1.0))], (2, 3)) * Point::slot();
    /// let back = ends.adjugate();
    /// let x = Field::from_fn(3, |i| Point::xyz(i as f64, 1.0, -2.0));
    /// let c = Field::from_fn(2, |i| Plane::new(1.0, i as f64, 0.5, -1.0));
    /// let pair = |a: &Field<Plane<(), f64>>, b: &Field<Point<(), f64>>| a.iter().zip(b.iter()).map(|(p, q)| (p & q).s()).sum::<f64>();
    /// assert!((pair(&back.of(&c), &x) - pair(&c, &ends.of(&x))).abs() < 1e-12);
    /// ```
    #[must_use]
    pub fn adjugate(&self) -> SparseMap<<C as Adjugate>::Output>
    where
        C: Adjugate,
    {
        self.transpose_with(Adjugate::adjugate)
    }

    /// The adjoint through the scalar product: the couplings run back and each map cell's
    /// adjoint taken, so that `m.adjoint().of(b)` paired with `x` by `scalar_product`, summed
    /// over the sites, equals `b` paired with `m.of(x)` (numga's `adjoint` on a sparse
    /// extensor; see [`Adjoint`]). `m.adjoint() * w * m` is the Laplacian of a connection
    /// `m` with edge weights `w`.
    #[must_use]
    pub fn adjoint(&self) -> SparseMap<<C as Adjoint>::Output>
    where
        C: Adjoint,
    {
        self.transpose_with(Adjoint::adjoint)
    }

    /// The map cells applied to a field: each cell to the value at its column, summed at the
    /// rows (the block matrix-vector product).
    ///
    /// # Panics
    /// If `x` is not over the map's input sites.
    #[must_use]
    #[track_caller]
    pub fn of<X, Y>(&self, x: &Field<X>) -> Field<Y>
    where
        C: Of<X, Output = Y>,
        X: Extensor,
        Y: Extensor + Add<Output = Y>,
    {
        self.combine(x, Of::of)
    }

    /// [`of`](Self::of) into an existing field over the map's output sites, which it
    /// overwrites: no allocation, for loops that step a field every tick.
    ///
    /// ```
    /// use gax::field::{Field, SparseMap};
    /// use gax::pga3d::{Plane, Scalar};
    /// let s = Scalar::<(), f64>::new;
    /// // Differences along a chain of three sites, as map cells on planes.
    /// let ends = SparseMap::new(vec![s(-1.0), s(1.0), s(-1.0), s(1.0)], vec![0, 0, 1, 1], vec![0, 1, 1, 2], (2, 3));
    /// let ends = &ends * Plane::slot();
    /// let x = Field::from_fn(3, |i| Plane::new(i as f64, 0.0, 0.0, 0.0));
    /// let mut steps = Field::zeros(2);
    /// ends.of_into(&x, &mut steps);
    /// assert_eq!(steps, ends.of(&x));
    /// ```
    ///
    /// # Panics
    /// If `x` is not over the map's input sites or `y` over its output sites.
    #[track_caller]
    pub fn of_into<X, Y>(&self, x: &Field<X>, y: &mut Field<Y>)
    where
        C: Of<X, Output = Y>,
        X: Extensor,
        Y: Extensor + Add<Output = Y>,
    {
        self.combine_into(x, y, Of::of);
    }

    /// The product of multivector cells with a field (`&self * x`) into an existing field over
    /// the map's output sites, which it overwrites: no allocation.
    ///
    /// # Panics
    /// If `x` is not over the map's input sites or `y` over its output sites.
    #[track_caller]
    pub fn mul_into<X, Y>(&self, x: &Field<X>, y: &mut Field<Y>)
    where
        C: Mul<X, Output = Y>,
        X: Extensor,
        Y: Extensor + Add<Output = Y>,
    {
        self.combine_into(x, y, |c, v| c * v);
    }

    /// Scalar cells as weights on a field of any kind: each row's sum of the values at its
    /// columns, each scaled by its cell's coefficient, in the field's own kind. `&self * x`
    /// multiplies instead, and its kind is the product's, which for a part is the full kind (a
    /// scalar times a PGA3D `Direction` is a `Point`); this keeps the part (numga's scalar map
    /// on a field of directions gives directions).
    ///
    /// ```
    /// use gax::field::{Field, SparseMap};
    /// use gax::pga3d::{Direction, Scalar};
    /// let s = Scalar::<(), f64>::new;
    /// let mean = SparseMap::from_triples([(0, 0, s(0.5)), (0, 1, s(0.5))], (1, 2));
    /// let pulls = Field::new(vec![Direction::new(1.0, 0.0, 0.0), Direction::new(0.0, 2.0, 0.0)]);
    /// let pull: Field<Direction<(), f64>> = mean.weigh(&pulls);
    /// assert_eq!(pull.site(0), Direction::new(0.5, 1.0, 0.0));
    /// ```
    ///
    /// # Panics
    /// If `x` is not over the map's input sites.
    #[must_use]
    #[track_caller]
    pub fn weigh<X>(&self, x: &Field<X>) -> Field<X>
    where
        C: Extensor<Slots = ()>,
        X: Extensor<Coef = C::Coef> + Add<Output = X>,
    {
        const {
            assert!(
                <C::Kind as Kind>::N == 1,
                "the cells of a weighing are scalars"
            );
        }
        self.combine(x, |c, v| super::scale(v, c.coeffs().as_ref()[0]))
    }

    /// Each cell combined with the value at its column, summed at the rows.
    #[track_caller]
    fn combine<X: Extensor, Y: Extensor + Add<Output = Y>>(
        &self,
        x: &Field<X>,
        f: impl FnMut(C, X) -> Y,
    ) -> Field<Y> {
        let mut y = Field::zeros(self.shape.0);
        self.combine_into(x, &mut y, f);
        y
    }

    /// [`combine`](Self::combine) into `y`, overwriting it.
    #[track_caller]
    fn combine_into<X: Extensor, Y: Extensor + Add<Output = Y>>(
        &self,
        x: &Field<X>,
        y: &mut Field<Y>,
        mut f: impl FnMut(C, X) -> Y,
    ) {
        self.check_input(x.len());
        assert!(
            y.len() == self.shape.0,
            "a field map with {} output sites written into a field over {} sites",
            self.shape.0,
            y.len()
        );
        let x = x.sites();
        for (r, out) in y.sites_mut().iter_mut().enumerate() {
            let mut acc = zero::<Y>();
            for k in self.starts[r]..self.starts[r + 1] {
                acc = acc + f(self.cells[k], x[self.cols[k]]);
            }
            *out = acc;
        }
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
    /// inner product of the fields ([`Field::dot`]): `y.dot(m.of(x)) == m.transpose().of(y).dot(x)`.
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

impl<C: Copy + Add<Output = C> + Neg<Output = C>> Sub for &SparseMap<C> {
    type Output = SparseMap<C>;
    /// The couplings of `self` and the negated couplings of `o`: they add where they meet.
    ///
    /// # Panics
    /// If the shapes differ.
    #[track_caller]
    fn sub(self, o: Self) -> SparseMap<C> {
        assert!(
            self.shape == o.shape,
            "subtracting field maps of different shapes: {:?} and {:?}",
            self.shape,
            o.shape
        );
        let negated = o.couplings().map(|(r, c, cell)| (r, c, -cell));
        SparseMap::from_triples(self.couplings().chain(negated), self.shape)
    }
}

impl<C: Copy + Neg<Output = C>> Neg for &SparseMap<C> {
    type Output = SparseMap<C>;
    /// Every cell negated, couplings kept.
    fn neg(self) -> SparseMap<C> {
        self.map_cells(|c| -c)
    }
}

impl<C: Copy + Neg<Output = C>> Neg for SparseMap<C> {
    type Output = SparseMap<C>;
    /// Every cell negated, couplings kept.
    fn neg(mut self) -> SparseMap<C> {
        for c in &mut self.cells {
            *c = -*c;
        }
        self
    }
}

/// A field map times a number: every cell's coefficients scaled (numga's `S * 0.5`). For the
/// coefficient types `f32` and `f64`; [`SparseMap::scaled`] takes any.
macro_rules! scalar_mul {
    ($($t:ty),*) => {$(
        impl<C: Extensor<Coef = $t>> Mul<$t> for &SparseMap<C> {
            type Output = SparseMap<C>;
            fn mul(self, c: $t) -> SparseMap<C> {
                self.scaled(c)
            }
        }

        impl<C: Extensor<Coef = $t>> Mul<$t> for SparseMap<C> {
            type Output = SparseMap<C>;
            fn mul(mut self, c: $t) -> SparseMap<C> {
                for cell in &mut self.cells {
                    *cell = super::scale(*cell, c);
                }
                self
            }
        }

        impl<C: Extensor<Coef = $t>> Mul<&SparseMap<C>> for $t {
            type Output = SparseMap<C>;
            fn mul(self, m: &SparseMap<C>) -> SparseMap<C> {
                m.scaled(self)
            }
        }
    )*};
}
scalar_mul!(f32, f64);
