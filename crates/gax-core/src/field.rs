//! Fields: one object spread over sites, and sparse field maps that couple the sites (ADR-046,
//! a spike of the run-time tier).
//!
//! A batch holds independent copies; a [`Field`] is one object with a value at each site (the
//! vertices of a mesh, the beads of a string). A [`SparseMap`] couples the sites: it stores only
//! the couplings that are there, each a typed cell from one input site (its column) to one
//! output site (its row), as numga's `SparseExtensor`. The cell type is checked at compile time
//! (a map cell `Vector <- Vector` takes a field of vectors, nothing else); the site counts at
//! run time, with a panic that names both.
//!
//! * Multivector cells multiply: `&ends * &positions` takes each cell's product with the value
//!   at its column and sums at the rows; `&ends.reverse() * &ends` multiplies cells along the
//!   paths through shared sites; `&cells * Vector::slot()` turns them into map cells.
//! * Map cells apply: `stiffness.apply(&loads)`, `a.compose(&b)`, and
//!   `stiffness.solve(&loads)` (conjugate gradients: symmetric positive definite couplings).
//!
//! ```
//! use gax::field::{Field, SparseMap};
//! use gax::pga3d::{Point, Scalar};
//!
//! // The differences of neighbouring points on a line of three: two couplings per edge.
//! let s = |x: f64| Scalar::<(), f64>::new(x);
//! let ends = SparseMap::new(vec![s(-1.0), s(1.0), s(-1.0), s(1.0)], vec![0, 0, 1, 1], vec![0, 1, 1, 2], (2, 3));
//! let points = Field::new(vec![Point::xyz(0.0, 0.0, 0.0), Point::xyz(1.0, 0.0, 0.0), Point::xyz(3.0, 0.0, 0.0)]);
//! let steps = &ends * &points;
//! assert_eq!(steps.site(1), Point::xyz(3.0, 0.0, 0.0) - Point::xyz(1.0, 0.0, 0.0));
//! ```

extern crate std;

use crate::bind::Of;
use crate::coef::{Coef, Real};
use crate::kind::{Extensor, Kind};
use crate::ops::Reverse;
use crate::slots::Slots;
use core::ops::{Add, Mul, Sub};
use std::vec::Vec;

/// The zero value, map or form of `M`'s type.
#[inline(always)]
fn zero<M: Extensor>() -> M {
    M::from_coeffs(<M::Kind as Kind>::arr_from_fn(|_| {
        <M::Slots as Slots>::from_flat(&mut |_| M::Coef::zero(), 0)
    }))
}

/// One object spread over sites: a value (or map, or form) of type `M` at each site.
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

    /// The value at site `i`.
    #[must_use]
    pub fn site(&self, i: usize) -> M {
        self.sites[i]
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

    /// The sum over the sites: a reduction of the field's own sites (numga's `.sites.sum()`).
    #[must_use]
    pub fn sum(&self) -> M
    where
        M: Add<Output = M>,
    {
        self.sites.iter().fold(zero(), |a, b| a + *b)
    }

    /// Every coefficient scaled by `c`.
    #[must_use]
    pub fn scaled(&self, c: M::Coef) -> Self
    where
        M: crate::kind::Extensor,
    {
        Field {
            sites: self
                .sites
                .iter()
                .map(|m| {
                    M::from_coeffs(<M::Kind as Kind>::arr_map(m.coeffs(), |col| {
                        <M::Slots as Slots>::map(col, &mut |x| *x * c)
                    }))
                })
                .collect(),
        }
    }

    /// The coefficient inner product summed over the sites, `Σ_sites Σ_k a_k b_k`.
    ///
    /// # Panics
    /// If the fields have different numbers of sites.
    #[must_use]
    pub fn dot(&self, other: &Self) -> M::Coef {
        same_sites(self.len(), other.len());
        let mut sum = M::Coef::zero();
        for (a, b) in self.sites.iter().zip(&other.sites) {
            for (ca, cb) in a.coeffs().as_ref().iter().zip(b.coeffs().as_ref()) {
                let n = <M::Slots as Slots>::SIZE;
                for k in 0..n {
                    sum = <M::Slots as Slots>::get_flat(ca, k)
                        .mul_add(<M::Slots as Slots>::get_flat(cb, k), sum);
                }
            }
        }
        sum
    }
}

fn same_sites(a: usize, b: usize) {
    assert!(
        a == b,
        "fields over different sites: {a} sites against {b}; a field map's input or a site-by-site operation needs the same sites"
    );
}

impl<M: Extensor + Add<Output = M>> Add for &Field<M> {
    type Output = Field<M>;
    fn add(self, o: Self) -> Field<M> {
        self.zip_with(o, |a, b| a + b)
    }
}

impl<M: Extensor + Sub<Output = M>> Sub for &Field<M> {
    type Output = Field<M>;
    fn sub(self, o: Self) -> Field<M> {
        self.zip_with(o, |a, b| a - b)
    }
}

/// A field map stored as its couplings: cell `k` couples input site `cols[k]` to output site
/// `rows[k]`. Couplings may repeat; they add. The cells are multivectors (they multiply the
/// field's values) or maps (they apply to them).
#[derive(Clone, Debug, PartialEq)]
pub struct SparseMap<C> {
    cells: Vec<C>,
    rows: Vec<usize>,
    cols: Vec<usize>,
    shape: (usize, usize),
}

impl<C: Copy> SparseMap<C> {
    /// Couplings `cells[k]` from input site `cols[k]` to output site `rows[k]`, `shape` the
    /// numbers of output and input sites.
    ///
    /// # Panics
    /// If the three lists differ in length, or an index is outside `shape`.
    #[must_use]
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
        SparseMap {
            cells,
            rows,
            cols,
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
        SparseMap::new(
            field.sites.clone(),
            (0..n).collect(),
            (0..n).collect(),
            (n, n),
        )
    }

    /// The numbers of output and input sites.
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        self.shape
    }

    /// The stored couplings: `(row, column, cell)`.
    pub fn couplings(&self) -> impl Iterator<Item = (usize, usize, C)> + '_ {
        (0..self.cells.len()).map(|k| (self.rows[k], self.cols[k], self.cells[k]))
    }

    /// `f` applied to every cell, couplings kept.
    #[must_use]
    pub fn map_cells<D: Copy>(&self, f: impl FnMut(C) -> D) -> SparseMap<D> {
        SparseMap {
            cells: self.cells.iter().copied().map(f).collect(),
            rows: self.rows.clone(),
            cols: self.cols.clone(),
            shape: self.shape,
        }
    }

    /// Every coupling run the other way, from its row to its column, with `f` on its cell.
    #[must_use]
    pub fn transpose_with<D: Copy>(&self, f: impl FnMut(C) -> D) -> SparseMap<D> {
        SparseMap {
            cells: self.cells.iter().copied().map(f).collect(),
            rows: self.cols.clone(),
            cols: self.rows.clone(),
            shape: (self.shape.1, self.shape.0),
        }
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
    pub fn apply<X, Y>(&self, x: &Field<X>) -> Field<Y>
    where
        C: Of<X, Output = Y>,
        X: Extensor,
        Y: Extensor + Add<Output = Y>,
    {
        self.combine(x, Of::of)
    }

    /// Each cell combined with the value at its column, summed at the rows.
    fn combine<X: Extensor, Y: Extensor + Add<Output = Y>>(
        &self,
        x: &Field<X>,
        mut f: impl FnMut(C, X) -> Y,
    ) -> Field<Y> {
        assert!(
            x.len() == self.shape.1,
            "a field map with {} input sites applied to a field over {} sites",
            self.shape.1,
            x.len()
        );
        let mut out: Vec<Y> = (0..self.shape.0).map(|_| zero()).collect();
        for k in 0..self.cells.len() {
            let r = self.rows[k];
            out[r] = out[r] + f(self.cells[k], x.sites[self.cols[k]]);
        }
        Field { sites: out }
    }

    /// Couplings along every path through the shared sites, `f` on each pair of cells: the
    /// output has one cell per pair (`self` from site `k` to `r`, `other` from `c` to `k`),
    /// at `(r, c)`, duplicates summed.
    fn chain<D: Copy, E: Extensor + Add<Output = E>>(
        &self,
        other: &SparseMap<D>,
        mut f: impl FnMut(C, D) -> E,
    ) -> SparseMap<E> {
        assert!(
            self.shape.1 == other.shape.0,
            "field maps over different sites: {} input sites against {} output sites",
            self.shape.1,
            other.shape.0
        );
        // The other map's couplings by row.
        let mut by_row: Vec<Vec<usize>> = (0..other.shape.0).map(|_| Vec::new()).collect();
        for (k, &r) in other.rows.iter().enumerate() {
            by_row[r].push(k);
        }
        let mut entries: Vec<(usize, usize, E)> = Vec::new();
        for i in 0..self.cells.len() {
            for &j in &by_row[self.cols[i]] {
                entries.push((
                    self.rows[i],
                    other.cols[j],
                    f(self.cells[i], other.cells[j]),
                ));
            }
        }
        entries.sort_by_key(|e| (e.0, e.1));
        let (mut cells, mut rows, mut cols) = (Vec::new(), Vec::new(), Vec::new());
        for (r, c, e) in entries {
            if rows.last() == Some(&r) && cols.last() == Some(&c) {
                let last = cells.len() - 1;
                cells[last] = cells[last] + e;
            } else {
                cells.push(e);
                rows.push(r);
                cols.push(c);
            }
        }
        SparseMap {
            cells,
            rows,
            cols,
            shape: (self.shape.0, other.shape.1),
        }
    }

    /// The map cells composed with another field map's, along the paths through the shared
    /// sites: `self ∘ other`.
    #[must_use]
    pub fn compose<D: Copy, E>(&self, other: &SparseMap<D>) -> SparseMap<E>
    where
        C: Of<D, Output = E>,
        E: Extensor + Add<Output = E>,
    {
        self.chain(other, Of::of)
    }

    /// Solve `self.apply(x) == b` for `x` by conjugate gradients, the whole coupled system at
    /// once. The couplings must be symmetric positive definite under the coefficient inner
    /// product (a stiffness, a Laplacian with anchors); iterates until the residual is
    /// `ε`-small relative to `b`, or for twice the number of unknowns.
    ///
    /// # Panics
    /// If `b` is not over the map's output sites or the map is not square.
    #[must_use]
    pub fn solve<X>(&self, b: &Field<X>) -> Field<X>
    where
        C: Of<X, Output = X>,
        X: Extensor<Coef: Real> + Add<Output = X> + Sub<Output = X>,
    {
        assert!(
            self.shape.0 == self.shape.1,
            "solve needs a square field map, not {} <- {} sites",
            self.shape.0,
            self.shape.1
        );
        let n = b.len() * <X::Kind as Kind>::N * <X::Slots as Slots>::SIZE;
        let mut x = Field::<X>::zeros(b.len());
        let mut r = b.clone();
        let mut p = r.clone();
        let mut rs = r.dot(&r);
        let tol = rs * X::Coef::epsilon() * X::Coef::epsilon();
        for _ in 0..2 * n {
            if X::Coef::all_lt(rs, tol) || rs == X::Coef::zero() {
                break;
            }
            let ap = self.apply(&p);
            let alpha = rs / p.dot(&ap);
            x = &x + &p.scaled(alpha);
            r = &r - &ap.scaled(alpha);
            let rs_new = r.dot(&r);
            p = &r + &p.scaled(rs_new / rs);
            rs = rs_new;
        }
        x
    }
}

impl<C: Copy + Add<Output = C>> Add for &SparseMap<C> {
    type Output = SparseMap<C>;
    /// The couplings of both: they add where they meet.
    fn add(self, o: Self) -> SparseMap<C> {
        assert!(
            self.shape == o.shape,
            "adding field maps of different shapes: {:?} and {:?}",
            self.shape,
            o.shape
        );
        let mut s = self.clone();
        s.cells.extend_from_slice(&o.cells);
        s.rows.extend_from_slice(&o.rows);
        s.cols.extend_from_slice(&o.cols);
        s
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

#[cfg(feature = "batch")]
impl<C: Extensor> SparseMap<C> {
    /// [`apply`](Self::apply) on SIMD lanes of the current batch level: `L` couplings at a time,
    /// their cells and the values at their columns gathered into lanes, the products scattered
    /// into the rows one by one (rows repeat, so the sums stay serial). The batch kernels use
    /// fused multiply-adds, so the last bits can differ from `apply`.
    ///
    /// # Panics
    /// If `x` is not over the map's input sites.
    #[must_use]
    pub fn apply_batch<A: Kind>(
        &self,
        x: &Field<A::Mv<(), C::Coef>>,
    ) -> Field<<C::Kind as Kind>::Mv<(), C::Coef>>
    where
        C: Extensor<Slots = (A,), Coef: crate::batch::LaneElem>,
        <C::Kind as Kind>::Mv<(), C::Coef>: Add<Output = <C::Kind as Kind>::Mv<(), C::Coef>>,
    {
        use crate::batch::{Batch, Kernel, LaneElem, Mv, run};
        struct Run<'a, Y: Kind, X: Kind, E: LaneElem, M> {
            map: &'a SparseMap<M>,
            x: &'a [Mv<X, E>],
            out: &'a mut [Mv<Y, E>],
        }
        impl<Y: Kind, X: Kind, E: LaneElem, M: Extensor<Kind = Y, Slots = (X,), Coef = E>> Kernel<E>
            for Run<'_, Y, X, E, M>
        where
            Mv<Y, E>: Add<Output = Mv<Y, E>>,
        {
            type Output = ();
            #[inline(always)]
            fn run<L: Batch<Elem = E>>(self) {
                let Run { map, x, out } = self;
                let n = map.cells.len();
                let full = n - n % L::LANES;
                let mut k = 0;
                while k < full {
                    let cells = &map.cells[k..k + L::LANES];
                    let cols = &map.cols[k..k + L::LANES];
                    let m: Y::Arr<X::Arr<L>> = Y::arr_from_fn(
                        #[inline(always)]
                        |o| {
                            X::arr_from_fn(
                                #[inline(always)]
                                |i| L::from_fn(|l| cells[l].coeffs().as_ref()[o].as_ref()[i]),
                            )
                        },
                    );
                    let xs: Mv<X, L> = Extensor::from_coeffs(X::arr_from_fn(
                        #[inline(always)]
                        |i| L::from_fn(|l| x[cols[l]].coeffs().as_ref()[i]),
                    ));
                    let y = crate::batch::kernels::apply_rows::<Y, X, L>(&m, &xs);
                    let lanes = Y::arr_map(y.coeffs(), |c| crate::batch::to_array(*c));
                    for l in 0..L::LANES {
                        let v: Mv<Y, E> = Extensor::from_coeffs(Y::arr_map(&lanes, |c| c[l]));
                        let r = map.rows[k + l];
                        out[r] = out[r] + v;
                    }
                    k += L::LANES;
                }
                // The remainder one coupling at a time, with the lanes' arithmetic.
                for k in full..n {
                    let r = map.rows[k];
                    let y = crate::batch::kernels::apply_rows::<Y, X, E>(
                        map.cells[k].coeffs(),
                        &x[map.cols[k]],
                    );
                    out[r] = out[r] + y;
                }
            }
        }
        assert!(
            x.len() == self.shape.1,
            "a field map with {} input sites applied to a field over {} sites",
            self.shape.1,
            x.len()
        );
        let mut out: Vec<Mv<C::Kind, C::Coef>> = (0..self.shape.0).map(|_| zero()).collect();
        run(Run::<C::Kind, A, C::Coef, C> {
            map: self,
            x: &x.sites,
            out: &mut out,
        });
        Field { sites: out }
    }
}

/// A [`SparseMap`] of map cells packed for the batch kernels once: its cells in blocks of
/// [`BLOCK`](crate::batch::BLOCK), coefficient by coefficient, as a `Soa` stores values, so that
/// a lane of a cell coefficient is one vector load. Build it once for a map applied many times
/// (a stiffness in an iteration), as `BatchOf` splats its single matrix once.
#[cfg(feature = "batch")]
#[derive(Clone, Debug)]
pub struct PackedSparseMap<C: Extensor> {
    blocks: Vec<C::Coef>,
    rows: Vec<usize>,
    cols: Vec<usize>,
    shape: (usize, usize),
}

#[cfg(feature = "batch")]
impl<C: Extensor> SparseMap<C> {
    /// The cells packed for [`PackedSparseMap::apply`].
    #[must_use]
    pub fn packed(&self) -> PackedSparseMap<C> {
        use crate::batch::BLOCK;
        let q = <C::Kind as Kind>::N * <C::Slots as Slots>::SIZE;
        let n = self.cells.len();
        let nb = n.div_ceil(BLOCK);
        let mut blocks = std::vec![C::Coef::zero(); nb * q * BLOCK];
        for (k, cell) in self.cells.iter().enumerate() {
            let (b, l) = (k / BLOCK, k % BLOCK);
            for (o, col) in cell.coeffs().as_ref().iter().enumerate() {
                for i in 0..<C::Slots as Slots>::SIZE {
                    let flat = o * <C::Slots as Slots>::SIZE + i;
                    blocks[(b * q + flat) * BLOCK + l] = <C::Slots as Slots>::get_flat(col, i);
                }
            }
        }
        PackedSparseMap {
            blocks,
            rows: self.rows.clone(),
            cols: self.cols.clone(),
            shape: self.shape,
        }
    }
}

#[cfg(feature = "batch")]
impl<C: Extensor> PackedSparseMap<C> {
    /// [`SparseMap::apply`] from the packed cells: per lane batch, the cell coefficients by
    /// vector loads, the values at the columns gathered, the products scattered into the rows.
    ///
    /// # Panics
    /// If `x` is not over the map's input sites.
    #[must_use]
    pub fn apply<A: Kind>(
        &self,
        x: &Field<A::Mv<(), C::Coef>>,
    ) -> Field<<C::Kind as Kind>::Mv<(), C::Coef>>
    where
        C: Extensor<Slots = (A,), Coef: crate::batch::LaneElem>,
        <C::Kind as Kind>::Mv<(), C::Coef>: Add<Output = <C::Kind as Kind>::Mv<(), C::Coef>>,
    {
        use crate::batch::{BLOCK, Batch, Kernel, LaneElem, Mv, run};
        struct Run<'a, Y: Kind, X: Kind, E: LaneElem> {
            blocks: &'a [E],
            rows: &'a [usize],
            cols: &'a [usize],
            x: &'a [Mv<X, E>],
            out: &'a mut [Mv<Y, E>],
        }
        impl<Y: Kind, X: Kind, E: LaneElem> Kernel<E> for Run<'_, Y, X, E>
        where
            Mv<Y, E>: Add<Output = Mv<Y, E>>,
        {
            type Output = ();
            #[inline(always)]
            fn run<L: Batch<Elem = E>>(self) {
                let Run {
                    blocks,
                    rows,
                    cols,
                    x,
                    out,
                } = self;
                let q = Y::N * X::N;
                let n = rows.len();
                let mut k = 0;
                while k < n {
                    let (b, off) = (k / BLOCK, k % BLOCK);
                    let block = &blocks[b * q * BLOCK..(b + 1) * q * BLOCK];
                    let m: Y::Arr<X::Arr<L>> = Y::arr_from_fn(
                        #[inline(always)]
                        |o| {
                            X::arr_from_fn(
                                #[inline(always)]
                                |i| L::load(&block[(o * X::N + i) * BLOCK + off..]),
                            )
                        },
                    );
                    let last = n - 1;
                    let xs: Mv<X, L> = Extensor::from_coeffs(X::arr_from_fn(
                        #[inline(always)]
                        |i| L::from_fn(|l| x[cols[(k + l).min(last)]].coeffs().as_ref()[i]),
                    ));
                    let y = crate::batch::kernels::apply_rows::<Y, X, L>(&m, &xs);
                    let lanes = Y::arr_map(y.coeffs(), |c| crate::batch::to_array(*c));
                    for l in 0..L::LANES.min(n - k) {
                        let v: Mv<Y, E> = Extensor::from_coeffs(Y::arr_map(&lanes, |c| c[l]));
                        let r = rows[k + l];
                        out[r] = out[r] + v;
                    }
                    k += L::LANES;
                }
            }
        }
        assert!(
            x.len() == self.shape.1,
            "a field map with {} input sites applied to a field over {} sites",
            self.shape.1,
            x.len()
        );
        let mut out: Vec<Mv<C::Kind, C::Coef>> = (0..self.shape.0).map(|_| zero()).collect();
        run(Run::<C::Kind, A, C::Coef> {
            blocks: &self.blocks,
            rows: &self.rows,
            cols: &self.cols,
            x: &x.sites,
            out: &mut out,
        });
        Field { sites: out }
    }
}
