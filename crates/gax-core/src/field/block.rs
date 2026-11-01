//! Block field maps over two fields: saddle-point systems, solved by one sparse LU.

use super::direct::{Assembly, Factors};
use super::solve::{flatten, unflatten};
use super::{Field, Singular, SparseMap};
use crate::coef::Real;
use crate::kind::{Extensor, Kind};
use core::marker::PhantomData;
use core::ops::Add;

/// A field map over two fields, as four blocks `[[A, B], [C, D]]`: `(x, y) ↦ (A x + B y,
/// C x + D y)`, with `x` a field of one kind and `y` of another, over sites of their own. The
/// saddle-point systems of constraints are of this shape: Stokes flow's viscous stiffness and
/// divergence, `[[K, −Dᵀ], [−D, 0]]` over fluxes and pressures; piezoelectricity's elastic,
/// coupling and dielectric blocks over displacements and potentials. Each block keeps its own
/// cell type, so nothing is flattened into numbers.
///
/// The blocks' kinds must chain: `A: Y₁ <- X₁`, `B: Y₁ <- X₂`, `C: Y₂ <- X₁`, `D: Y₂ <- X₂`,
/// with `A` and `D` square (each field's sites and coefficients as many as its equations'). The
/// system is solved directly, by one sparse LU of the whole block matrix
/// ([`solve_direct`](Self::solve_direct), [`factor`](Self::factor)), with the sites of both
/// fields ordered together by minimum degree; a zero `D` (a saddle point) is no obstacle to it.
///
/// ```
/// use gax::field::{BlockMap, Field, SparseMap};
/// use gax::vga3d::Vector;
///
/// // Three springs tied to their rest positions, of stiffnesses 1, 2 and 4, pulled so that
/// // their displacements sum to `c`: the least energy under a constraint, with the constraint's
/// // force (a Lagrange multiplier) on one site of its own.
/// let one = Vector::<(), f64>::slot();
/// let stiffness = SparseMap::from_diagonal(&Field::from_fn(3, |i| one * f64::from(1 << i)));
/// let spread = SparseMap::from_triples((0..3).map(|i| (i, 0, one)), (3, 1));
/// let sum = SparseMap::from_triples((0..3).map(|i| (0, i, one)), (1, 3));
/// let system = BlockMap::saddle(stiffness, spread, sum);
/// let c = Vector::new(7.0, 0.0, -3.5);
/// let (x, force) = system.solve_direct(&Field::zeros(3), &Field::new(vec![c]));
/// // The displacements share the pull in proportion to the springs' compliances.
/// assert!((x.sum() - c).norm() < 1e-14);
/// assert!((x.site(0) - c * (4.0 / 7.0)).norm() < 1e-14);
/// assert!((force.site(0) + c * (4.0 / 7.0)).norm() < 1e-14);
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct BlockMap<A, B, C, D> {
    a: SparseMap<A>,
    b: SparseMap<B>,
    c: SparseMap<C>,
    d: SparseMap<D>,
}

/// The sparse LU of a [`BlockMap`]: solves for a field of `X₁` and one of `X₂` from right-hand
/// sides of `Y₁` and `Y₂`.
#[derive(Clone, Debug)]
pub struct BlockLu<X1: Kind, X2: Kind, T, Y1: Kind = X1, Y2: Kind = X2> {
    sites: (usize, usize),
    factors: Factors<T>,
    marker: PhantomData<(X1, X2, Y1, Y2)>,
}

impl<X1: Kind, X2: Kind, T: Real, Y1: Kind, Y2: Kind> BlockLu<X1, X2, T, Y1, Y2> {
    /// The numbers of sites of the two fields.
    #[must_use]
    pub fn sites(&self) -> (usize, usize) {
        self.sites
    }

    /// The number of stored entries of `L` and `U` together.
    #[must_use]
    pub fn fill(&self) -> usize {
        self.factors.fill()
    }

    /// Solve the block system for the right-hand sides `f` and `g`.
    ///
    /// # Panics
    /// If `f` or `g` is not over its block's sites.
    #[must_use]
    #[track_caller]
    #[allow(clippy::type_complexity)]
    pub fn solve(
        &self,
        f: &Field<Y1::Mv<(), T>>,
        g: &Field<Y2::Mv<(), T>>,
    ) -> (Field<X1::Mv<(), T>>, Field<X2::Mv<(), T>>) {
        assert!(
            f.len() == self.sites.0 && g.len() == self.sites.1,
            "a block field map over {:?} sites solved against fields over ({}, {}) sites",
            self.sites,
            f.len(),
            g.len()
        );
        let mut b = flatten(f);
        b.extend(flatten(g));
        let x = self.factors.solve(&b);
        let split = self.sites.0 * X1::N;
        (unflatten(&x[..split]), unflatten(&x[split..]))
    }
}

impl<A, B, C, D, X1, X2, Y1, Y2, T> BlockMap<A, B, C, D>
where
    A: Extensor<Kind = Y1, Slots = (X1,), Coef = T>,
    B: Extensor<Kind = Y1, Slots = (X2,), Coef = T>,
    C: Extensor<Kind = Y2, Slots = (X1,), Coef = T>,
    D: Extensor<Kind = Y2, Slots = (X2,), Coef = T>,
    X1: Kind,
    X2: Kind,
    Y1: Kind,
    Y2: Kind,
    T: Real,
{
    /// The block map `[[a, b], [c, d]]`.
    ///
    /// # Panics
    /// If the blocks' sites do not chain (`a` and `d` square, `b` and `c` across), or `a`'s or
    /// `d`'s cells are not square.
    #[must_use]
    #[track_caller]
    pub fn new((a, b): (SparseMap<A>, SparseMap<B>), (c, d): (SparseMap<C>, SparseMap<D>)) -> Self {
        let (n1, n2) = (a.shape.0, d.shape.0);
        assert!(
            a.shape == (n1, n1)
                && d.shape == (n2, n2)
                && b.shape == (n1, n2)
                && c.shape == (n2, n1),
            "BlockMap: the blocks' sites do not chain: {:?}, {:?}; {:?}, {:?}",
            a.shape,
            b.shape,
            c.shape,
            d.shape
        );
        assert!(
            X1::N == Y1::N && X2::N == Y2::N,
            "BlockMap: the diagonal blocks' cells are not square: {} <- {} and {} <- {} coefficients",
            Y1::N,
            X1::N,
            Y2::N,
            X2::N
        );
        BlockMap { a, b, c, d }
    }

    /// The numbers of sites of the two fields.
    #[must_use]
    pub fn sites(&self) -> (usize, usize) {
        (self.a.shape.0, self.d.shape.0)
    }

    /// The blocks `((a, b), (c, d))`.
    #[must_use]
    #[allow(clippy::type_complexity)]
    pub fn blocks(
        &self,
    ) -> (
        (&SparseMap<A>, &SparseMap<B>),
        (&SparseMap<C>, &SparseMap<D>),
    ) {
        ((&self.a, &self.b), (&self.c, &self.d))
    }

    /// The map applied to the pair of fields: `(a x + b y, c x + d y)`.
    ///
    /// # Panics
    /// If `x` or `y` is not over its block's sites.
    #[must_use]
    #[track_caller]
    #[allow(clippy::type_complexity)]
    pub fn of(
        &self,
        x: &Field<X1::Mv<(), T>>,
        y: &Field<X2::Mv<(), T>>,
    ) -> (Field<Y1::Mv<(), T>>, Field<Y2::Mv<(), T>>)
    where
        A: crate::bind::Of<X1::Mv<(), T>, Output = Y1::Mv<(), T>>,
        B: crate::bind::Of<X2::Mv<(), T>, Output = Y1::Mv<(), T>>,
        C: crate::bind::Of<X1::Mv<(), T>, Output = Y2::Mv<(), T>>,
        D: crate::bind::Of<X2::Mv<(), T>, Output = Y2::Mv<(), T>>,
        Y1::Mv<(), T>: Add<Output = Y1::Mv<(), T>>,
        Y2::Mv<(), T>: Add<Output = Y2::Mv<(), T>>,
    {
        let first = self.a.of(x).zip_with(&self.b.of(y), |p, q| p + q);
        let second = self.c.of(x).zip_with(&self.d.of(y), |p, q| p + q);
        (first, second)
    }

    /// The sparse LU factorization of the whole block matrix, the sites of both fields in one
    /// minimum-degree order, for [`BlockLu::solve`].
    ///
    /// # Errors
    /// [`Singular`] if a column has no nonzero pivot: the system is singular (a pressure's
    /// gauge left free, a constraint repeated). Its site counts the first field's sites first,
    /// then the second's.
    pub fn factor(&self) -> Result<BlockLu<X1, X2, T, Y1, Y2>, Singular> {
        let (n1, n2) = self.sites();
        let mut assembly = Assembly::new(&[(n1, X1::N), (n2, X2::N)]);
        assembly.add(&self.a, 0, 0);
        assembly.add(&self.b, 0, n1);
        assembly.add(&self.c, n1, 0);
        assembly.add(&self.d, n1, n1);
        Ok(BlockLu {
            sites: (n1, n2),
            factors: assembly.factor()?,
            marker: PhantomData,
        })
    }

    /// Solve `self.of(x, y) == (f, g)` exactly, by [`factor`](Self::factor) and
    /// [`BlockLu::solve`].
    ///
    /// # Panics
    /// If `f` or `g` is not over its block's sites, or the system is singular.
    #[must_use]
    #[track_caller]
    #[allow(clippy::type_complexity)]
    pub fn solve_direct(
        &self,
        f: &Field<Y1::Mv<(), T>>,
        g: &Field<Y2::Mv<(), T>>,
    ) -> (Field<X1::Mv<(), T>>, Field<X2::Mv<(), T>>) {
        match self.factor() {
            Ok(lu) => lu.solve(f, g),
            Err(e) => panic!("BlockMap::solve_direct: {e}"),
        }
    }

    /// [`solve_direct`](Self::solve_direct), or `None` where the system is singular or the
    /// solution is not finite.
    ///
    /// # Panics
    /// If `f` or `g` is not over its block's sites.
    #[must_use]
    #[track_caller]
    #[allow(clippy::type_complexity)]
    pub fn try_solve_direct(
        &self,
        f: &Field<Y1::Mv<(), T>>,
        g: &Field<Y2::Mv<(), T>>,
    ) -> Option<(Field<X1::Mv<(), T>>, Field<X2::Mv<(), T>>)> {
        let (x, y) = self.factor().ok()?.solve(f, g);
        let finite =
            x.sites().iter().all(Extensor::is_finite) && y.sites().iter().all(Extensor::is_finite);
        finite.then_some((x, y))
    }
}

impl<A, B, C, X1, X2, Y1, Y2, T> BlockMap<A, B, C, Y2::Mv<(X2,), T>>
where
    A: Extensor<Kind = Y1, Slots = (X1,), Coef = T>,
    B: Extensor<Kind = Y1, Slots = (X2,), Coef = T>,
    C: Extensor<Kind = Y2, Slots = (X1,), Coef = T>,
    X1: Kind,
    X2: Kind,
    Y1: Kind,
    Y2: Kind,
    T: Real,
{
    /// The saddle-point map `[[a, b], [c, 0]]`: the second field a multiplier with no equation
    /// of its own (a pressure, a constraint's force).
    ///
    /// # Panics
    /// As [`new`](Self::new).
    #[must_use]
    #[track_caller]
    pub fn saddle(a: SparseMap<A>, b: SparseMap<B>, c: SparseMap<C>) -> Self {
        let n2 = c.shape.0;
        Self::new((a, b), (c, SparseMap::zeros((n2, n2))))
    }
}
