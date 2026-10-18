//! Fields over a fixed number of sites, as kinds (ADR-045).
//!
//! `Sites<K, N>` is a kind: the field of `K` over `N` sites, numga's `K[N]`, with `N · K::N`
//! coefficients. Being a kind, it fills slots, so a field map `K[N] <- K[N]` is an ordinary
//! extensor, [`SiteField<K, N, (Sites<K, N>,), T>`]: binding sums over the paired sites,
//! composition over the shared ones, and `solve`, `inverse`, `det`, `eigh` and `svd` see the
//! whole coupled block system (as [`SquareMap`] does for any map). A slot of `Sites<K, 4>` takes a field over four sites and
//! nothing else, checked at compile time.
//!
//! The products act site by site: `turn >> positions` turns every site, `a * field` multiplies
//! every site by `a`, and two fields over the same sites multiply site with site. For fields
//! whose site counts are known only at run time (meshes), see [`field`](crate::field).
//!
//! ```
//! use gax::sites::{SiteField, Sites};
//! use gax::pga3d::Plane;
//!
//! // Three planes, each tied to its neighbours with the weights `w`: one map over all three.
//! type P3 = Sites<Plane, 3>;
//! let w = [[2.0, -1.0, 0.0], [-1.0, 3.0, -1.0], [0.0, -1.0, 2.0]];
//! let k: SiteField<Plane, 3, (P3,), f64> = SiteField::from_blocks(|i, j| Plane::slot().gp(w[i][j]));
//! let load = SiteField::<Plane, 3, (), f64>::from_sites([Plane::new(0.0, 0.0, -1.0, 0.0); 3]);
//! let x = k.solve(load); // the coupled 12 × 12 system
//! assert!((k.of(x) - load).c.as_ref().iter().all(|v| v.abs() < 1e-14));
//! ```

use crate::bind::Of;
use crate::cast::{Cast, SubKind};
use crate::coef::{Coef, Elem, Real};
use crate::extensor::SquareMap;
use crate::kind::{Extensor, FlatKind, Kind, scale, zero};
use crate::ops::{
    Anticommutator, Commutator, Dot, Gp, Lc, Rc, ScalarProduct, Transform, TransformInv, Vee, Wedge,
};
use crate::slots::Slots;
use crate::unit::Unit;
use core::fmt;
use core::marker::PhantomData;
use core::ops::{Add, Index, IndexMut, Neg, Sub};

/// The field of `K` over `N` sites, as a kind: `N · K::N` coefficients, site by site. numga's
/// `K[N]`.
pub struct Sites<K, const N: usize>(PhantomData<K>);

impl<K, const N: usize> Clone for Sites<K, N> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<K, const N: usize> Copy for Sites<K, N> {}
impl<K, const N: usize> PartialEq for Sites<K, N> {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}
impl<K, const N: usize> fmt::Debug for Sites<K, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Sites<{}, {N}>", core::any::type_name::<K>())
    }
}

/// The coefficient array of [`Sites<K, N>`]: one array of `K` per site, viewed as one flat
/// slice of `N · K::N`.
#[derive(Clone, Copy)]
pub struct SiteArr<K: Kind, X: Elem, const N: usize>(pub [K::Arr<X>; N]);

impl<K: Kind, X: Elem, const N: usize> PartialEq for SiteArr<K, X, N> {
    fn eq(&self, o: &Self) -> bool {
        self.0 == o.0
    }
}
impl<K: Kind, X: Elem, const N: usize> fmt::Debug for SiteArr<K, X, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl<K: FlatKind, X: Elem, const N: usize> AsRef<[X]> for SiteArr<K, X, N> {
    #[inline(always)]
    fn as_ref(&self) -> &[X] {
        // SAFETY: `K::Arr<X>` is `[X; K::N]` (FlatKind), and an array of `N` of them is
        // `N · K::N` contiguous `X`s with no padding between them.
        unsafe { core::slice::from_raw_parts(self.0.as_ptr().cast::<X>(), K::N * N) }
    }
}

impl<K: FlatKind, X: Elem, const N: usize> AsMut<[X]> for SiteArr<K, X, N> {
    #[inline(always)]
    fn as_mut(&mut self) -> &mut [X] {
        // SAFETY: as in `as_ref`.
        unsafe { core::slice::from_raw_parts_mut(self.0.as_mut_ptr().cast::<X>(), K::N * N) }
    }
}

impl<K: FlatKind, X: Elem, const N: usize> Index<usize> for SiteArr<K, X, N> {
    type Output = X;
    #[inline(always)]
    fn index(&self, i: usize) -> &X {
        &self.as_ref()[i]
    }
}

impl<K: FlatKind, X: Elem, const N: usize> IndexMut<usize> for SiteArr<K, X, N> {
    #[inline(always)]
    fn index_mut(&mut self, i: usize) -> &mut X {
        &mut self.as_mut()[i]
    }
}

impl<K: FlatKind, const N: usize> Kind for Sites<K, N> {
    const N: usize = K::N * N;
    const NAME: &'static str = "Sites";
    const MODULE: &'static str = K::MODULE;
    /// A field's coefficients are its sites' blades, which have no single name.
    const BLADES: &'static [&'static str] = &[];
    type Arr<X: Elem> = SiteArr<K, X, N>;
    type Mv<S: Slots, T: Coef> = SiteField<K, N, S, T>;
    type Scalar = K::Scalar;
    type Orientation = K::Orientation;
    const DEFAULT_INTRINSIC: bool = K::DEFAULT_INTRINSIC;

    #[inline(always)]
    fn arr_from_fn<X: Elem>(mut f: impl FnMut(usize) -> X) -> SiteArr<K, X, N> {
        SiteArr(core::array::from_fn(|s| {
            K::arr_from_fn(|i| f(s * K::N + i))
        }))
    }
    #[inline(always)]
    fn arr_map<X: Elem, Y: Elem>(
        a: &SiteArr<K, X, N>,
        mut f: impl FnMut(&X) -> Y,
    ) -> SiteArr<K, Y, N> {
        SiteArr(core::array::from_fn(|s| K::arr_map(&a.0[s], &mut f)))
    }
    #[inline(always)]
    fn arr_zip<X: Elem, Y: Elem, Z: Elem>(
        a: &SiteArr<K, X, N>,
        b: &SiteArr<K, Y, N>,
        mut f: impl FnMut(&X, &Y) -> Z,
    ) -> SiteArr<K, Z, N> {
        SiteArr(core::array::from_fn(|s| {
            K::arr_zip(&a.0[s], &b.0[s], &mut f)
        }))
    }
}

/// `(i, i, false)` for every index: the identity's blade table, sliced to a field kind's size.
static IDENTITY: [(usize, usize, bool); 4096] = {
    let mut t = [(0, 0, false); 4096];
    let mut i = 0;
    while i < 4096 {
        t[i] = (i, i, false);
        i += 1;
    }
    t
};

impl<K: FlatKind, const N: usize> Cast<Sites<K, N>> for Sites<K, N> {
    /// A field kind shares every blade with itself (fields of up to 4096 coefficients).
    const SHARED: &'static [(usize, usize, bool)] = IDENTITY.split_at(K::N * N).0;
}

impl<K: FlatKind, const N: usize> SubKind<Sites<K, N>> for Sites<K, N> {}

/// The square block matrix of a field map `Sites<K, N> <- Sites<K, N>`, for the solvers.
impl<K: FlatKind, T: Real, const N: usize> crate::linalg::SquareArr<T>
    for SiteArr<K, SiteArr<K, T, N>, N>
{
    const N: usize = K::N * N;
    type Vector = SiteArr<K, T, N>;
    #[inline]
    fn zero() -> Self {
        <Sites<K, N> as Kind>::arr_from_fn(|_| Self::zero_vector())
    }
    #[inline]
    fn zero_vector() -> SiteArr<K, T, N> {
        <Sites<K, N> as Kind>::arr_from_fn(|_| T::zero())
    }
}

/// A value, map or form of the field kind [`Sites<K, N>`]: `K` at each of `N` sites, with open
/// slots `S` and coefficients `T`.
#[derive(Clone, Copy)]
pub struct SiteField<K: FlatKind, const N: usize, S: Slots = (), T: Coef = f32> {
    /// The coefficients, site by site: each site's output-first coefficients of `K` with slots
    /// `S`.
    pub c: SiteArr<K, S::Arr<T>, N>,
}

impl<K: FlatKind, const N: usize, S: Slots, T: Coef> PartialEq for SiteField<K, N, S, T> {
    fn eq(&self, o: &Self) -> bool {
        self.c == o.c
    }
}
impl<K: FlatKind, const N: usize, S: Slots, T: Coef> fmt::Debug for SiteField<K, N, S, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.c.0.iter()).finish()
    }
}

impl<K: FlatKind, const N: usize, S: Slots, T: Coef> Extensor for SiteField<K, N, S, T> {
    type Kind = Sites<K, N>;
    type Slots = S;
    type Coef = T;
    #[inline(always)]
    fn from_coeffs(c: SiteArr<K, S::Arr<T>, N>) -> Self {
        SiteField { c }
    }
    #[inline(always)]
    fn coeffs(&self) -> &SiteArr<K, S::Arr<T>, N> {
        &self.c
    }
}

impl<K: FlatKind, const N: usize, S: Slots, T: Coef> SiteField<K, N, S, T> {
    /// The field from its sites' values (or maps).
    #[must_use]
    pub fn from_sites(sites: [K::Mv<S, T>; N]) -> Self {
        from_values(sites)
    }

    /// Zero at every site.
    #[must_use]
    pub fn zero() -> Self {
        zero()
    }

    /// The value (or map) at site `i`.
    ///
    /// # Panics
    /// If `i` is not below `N`.
    #[must_use]
    pub fn site(&self, i: usize) -> K::Mv<S, T> {
        Extensor::from_coeffs(self.c.0[i])
    }

    /// `f` applied at every site.
    #[must_use]
    pub fn map_sites<M: Extensor<Coef = T>>(
        &self,
        mut f: impl FnMut(K::Mv<S, T>) -> M,
    ) -> Sited<M, N>
    where
        M::Kind: FlatKind,
    {
        from_values(core::array::from_fn(|i| f(self.site(i))))
    }

    /// The sum over the sites.
    #[must_use]
    pub fn sum(&self) -> K::Mv<S, T>
    where
        K::Mv<S, T>: Add<Output = K::Mv<S, T>>,
    {
        (1..N).fold(self.site(0), |a, i| a + self.site(i))
    }

    /// Every coefficient scaled by `c`.
    #[must_use]
    pub fn scaled(self, c: T) -> Self {
        scale(self, c)
    }

    /// Fill the first open slot with a value, or compose a map into it (see [`Of`]).
    #[inline]
    pub fn of<X>(self, x: X) -> <Self as Of<X>>::Output
    where
        Self: Of<X>,
    {
        Of::of(self, x)
    }
}

#[cfg(feature = "std")]
impl<K: FlatKind, const N: usize, T: Coef> SiteField<K, N, (), T> {
    /// The field from a run-time [`Field`](crate::field::Field) over exactly `N` sites.
    ///
    /// # Panics
    /// If the field is not over `N` sites.
    #[must_use]
    #[track_caller]
    pub fn from_field(field: &crate::field::Field<K::Mv<(), T>>) -> Self {
        assert!(
            field.len() == N,
            "a field over {} sites read as a field over {N}",
            field.len()
        );
        Self::from_sites(core::array::from_fn(|i| field.site(i)))
    }

    /// The field as a run-time [`Field`](crate::field::Field).
    #[must_use]
    pub fn to_field(&self) -> crate::field::Field<K::Mv<(), T>> {
        crate::field::Field::from_fn(N, |i| self.site(i))
    }
}

impl<K: FlatKind, const N: usize, T: Coef> SiteField<K, N, (Sites<K, N>,), T> {
    /// The field map `K[N] <- K[N]` whose block from site `j` to site `i` is `block(i, j)`, a
    /// map `K <- K`.
    #[must_use]
    pub fn from_blocks(mut block: impl FnMut(usize, usize) -> K::Mv<(K,), T>) -> Self {
        Self::from_coeffs(SiteArr(core::array::from_fn(|i| {
            let cells: [K::Mv<(K,), T>; N] = core::array::from_fn(|j| block(i, j));
            K::arr_from_fn(|o| SiteArr(cells.map(|cell| cell.coeffs().as_ref()[o])))
        })))
    }

    /// The block from site `j` to site `i`: a map `K <- K`.
    #[must_use]
    pub fn block(&self, i: usize, j: usize) -> K::Mv<(K,), T> {
        let rows = self.c.0[i].as_ref();
        Extensor::from_coeffs(K::arr_from_fn(|o| rows[o].0[j]))
    }

    /// The field map from a run-time [`SparseMap`](crate::field::SparseMap) over exactly `N`
    /// sites, its couplings summed into dense blocks.
    ///
    /// # Panics
    /// If the map is not over `N` sites, in and out.
    #[cfg(feature = "std")]
    #[must_use]
    #[track_caller]
    pub fn from_sparse<C>(map: &crate::field::SparseMap<C>) -> Self
    where
        C: Extensor<Kind = K, Slots = (K,), Coef = T>,
        K::Mv<(K,), T>: Add<Output = K::Mv<(K,), T>>,
    {
        assert!(
            map.shape() == (N, N),
            "a field map over {:?} sites read as one over {N}",
            map.shape()
        );
        let mut blocks = [[zero::<K::Mv<(K,), T>>(); N]; N];
        for (r, c, cell) in map.couplings() {
            blocks[r][c] = blocks[r][c] + Extensor::from_coeffs(*cell.coeffs());
        }
        Self::from_blocks(|i, j| blocks[i][j])
    }
}

impl<K: FlatKind, const N: usize, T: Real> SiteField<K, N, (Sites<K, N>,), T> {
    /// Solve `self.of(x) == b` for `x`: the coupled system over every site and blade, by LU.
    #[must_use]
    pub fn solve<X>(self, b: X) -> SiteField<K, N, X::Slots, T>
    where
        X: Extensor<Kind = Sites<K, N>, Coef = T>,
    {
        SquareMap::solve(self, b)
    }

    /// The inverse field map.
    #[must_use]
    pub fn inverse(self) -> Self {
        SquareMap::inverse(self)
    }

    /// The determinant of the block matrix.
    #[must_use]
    pub fn det(self) -> T {
        SquareMap::det(self)
    }

    /// The eigenvalues (ascending) and eigenvectors (fields) of the block matrix taken as
    /// symmetric (see [`SquareMap::eigh`]).
    #[must_use]
    #[allow(clippy::type_complexity)]
    pub fn eigh(self) -> (SiteArr<K, T, N>, SiteArr<K, SiteField<K, N, (), T>, N>) {
        SquareMap::eigh(self)
    }
}

/// The field over `N` sites whose site values have `M`'s type.
pub type Sited<M, const N: usize> =
    SiteField<<M as Extensor>::Kind, N, <M as Extensor>::Slots, <M as Extensor>::Coef>;

/// A field from its sites' values of any type of a flat kind.
#[inline]
fn from_values<M: Extensor, const N: usize>(values: [M; N]) -> Sited<M, N>
where
    M::Kind: FlatKind,
{
    SiteField {
        c: SiteArr(values.map(|v| *v.coeffs())),
    }
}

impl<K: FlatKind, const N: usize, S: Slots, T: Coef> Add for SiteField<K, N, S, T> {
    type Output = Self;
    #[inline]
    fn add(self, o: Self) -> Self {
        Self::from_coeffs(<Sites<K, N> as Kind>::arr_zip(&self.c, &o.c, |a, b| {
            S::zip(a, b, &mut |x, y| *x + *y)
        }))
    }
}

impl<K: FlatKind, const N: usize, S: Slots, T: Coef> Sub for SiteField<K, N, S, T> {
    type Output = Self;
    #[inline]
    fn sub(self, o: Self) -> Self {
        Self::from_coeffs(<Sites<K, N> as Kind>::arr_zip(&self.c, &o.c, |a, b| {
            S::zip(a, b, &mut |x, y| *x - *y)
        }))
    }
}

impl<K: FlatKind, const N: usize, S: Slots, T: Coef> Neg for SiteField<K, N, S, T> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self::from_coeffs(<Sites<K, N> as Kind>::arr_map(&self.c, |a| {
            S::map(a, &mut |x| -*x)
        }))
    }
}

/// The product `l · self` of a value `l` with a site value `self` on its right, asked of the
/// right operand: `P` names the product ([`GpOp`] for [`Gp`], and so on).
///
/// The site-by-site products of a value with a field are bounded by this rather than by
/// `L: Gp<X>` directly. A bound on `L` makes every probe of `l.gp(?)` with the operand's type
/// still unknown recurse into fields of fields of …; asked of `X`, the compiler waits until
/// `X` is known. Implemented for every value of a flat kind whose product with `L` exists.
pub trait RightOf<L, P> {
    /// `l · self`.
    type Output;
    /// `l · self`.
    fn right_of(self, l: L) -> Self::Output;
}

/// The products site by site: a field with a value of a flat kind (on either side, unit
/// versors on the left included), and two fields over the same sites.
macro_rules! site_products {
    ($(($Trait:ident, $method:ident, $Op:ident)),* $(,)?) => {$(
        #[doc = concat!("[`", stringify!($Trait), "`], as the product named in [`RightOf`].")]
        #[derive(Clone, Copy, Debug)]
        pub struct $Op;

        impl<X, L> RightOf<L, $Op> for X
        where
            X: Extensor<Kind: FlatKind>,
            L: $Trait<X>,
        {
            type Output = <L as $Trait<X>>::Output;
            #[inline(always)]
            fn right_of(self, l: L) -> Self::Output {
                l.$method(self)
            }
        }

        impl<K, const N: usize, S, T, R> $Trait<R> for SiteField<K, N, S, T>
        where
            K: FlatKind,
            S: Slots,
            T: Coef,
            R: Extensor<Coef = T, Kind: FlatKind>,
            K::Mv<S, T>: $Trait<R, Output: Extensor<Coef = T, Kind: FlatKind>>,
        {
            type Output = Sited<<K::Mv<S, T> as $Trait<R>>::Output, N>;
            #[inline]
            fn $method(self, rhs: R) -> Self::Output {
                from_values(core::array::from_fn(|i| self.site(i).$method(rhs)))
            }
        }

        impl<L, K, const N: usize, S, T> $Trait<SiteField<K, N, S, T>> for L
        where
            L: Extensor<Coef = T, Kind: FlatKind>,
            K: FlatKind,
            S: Slots,
            T: Coef,
            K::Mv<S, T>: RightOf<L, $Op, Output: Extensor<Coef = T, Kind: FlatKind>>,
        {
            type Output = Sited<<K::Mv<S, T> as RightOf<L, $Op>>::Output, N>;
            #[inline]
            fn $method(self, rhs: SiteField<K, N, S, T>) -> Self::Output {
                from_values(core::array::from_fn(|i| rhs.site(i).right_of(self)))
            }
        }

        impl<M, K, const N: usize, S, T> $Trait<SiteField<K, N, S, T>> for Unit<M>
        where
            M: Copy,
            K: FlatKind,
            S: Slots,
            T: Coef,
            K::Mv<S, T>: RightOf<Unit<M>, $Op, Output: Extensor<Coef = T, Kind: FlatKind>>,
        {
            type Output = Sited<<K::Mv<S, T> as RightOf<Unit<M>, $Op>>::Output, N>;
            #[inline]
            fn $method(self, rhs: SiteField<K, N, S, T>) -> Self::Output {
                from_values(core::array::from_fn(|i| rhs.site(i).right_of(self)))
            }
        }

        impl<A, B, const N: usize, S1, S2, T> $Trait<SiteField<B, N, S2, T>> for SiteField<A, N, S1, T>
        where
            A: FlatKind,
            B: FlatKind,
            S1: Slots,
            S2: Slots,
            T: Coef,
            A::Mv<S1, T>: $Trait<B::Mv<S2, T>, Output: Extensor<Coef = T, Kind: FlatKind>>,
        {
            type Output = Sited<<A::Mv<S1, T> as $Trait<B::Mv<S2, T>>>::Output, N>;
            #[inline]
            fn $method(self, rhs: SiteField<B, N, S2, T>) -> Self::Output {
                from_values(core::array::from_fn(|i| self.site(i).$method(rhs.site(i))))
            }
        }
    )*};
}

site_products!(
    (Gp, gp, GpOp),
    (Wedge, wedge, WedgeOp),
    (Vee, vee, VeeOp),
    (Dot, dot, DotOp),
    (Lc, lc, LcOp),
    (Rc, rc, RcOp),
    (ScalarProduct, scalar_product, ScalarProductOp),
    (Commutator, commutator, CommutatorOp),
    (Anticommutator, anticommutator, AnticommutatorOp),
    (Transform, transform, TransformOp),
    (TransformInv, transform_inv, TransformInvOp),
);

/// The operators of a field on the left forward to the products, as for every generated type.
macro_rules! site_operators {
    ($(($Op:ident, $op:ident, $Trait:ident, $method:ident)),* $(,)?) => {$(
        impl<K: FlatKind, const N: usize, S: Slots, T: Coef, R> core::ops::$Op<R> for SiteField<K, N, S, T>
        where
            Self: $Trait<R>,
        {
            type Output = <Self as $Trait<R>>::Output;
            #[inline]
            fn $op(self, rhs: R) -> Self::Output {
                <Self as $Trait<R>>::$method(self, rhs)
            }
        }
    )*};
}

site_operators!(
    (Mul, mul, Gp, gp),
    (BitXor, bitxor, Wedge, wedge),
    (BitAnd, bitand, Vee, vee),
    (BitOr, bitor, Dot, dot),
    (Shr, shr, Transform, transform),
    (Shl, shl, TransformInv, transform_inv),
);

#[cfg(test)]
mod tests {
    use super::*;

    /// A stand-in flat kind of three coefficients.
    #[derive(Clone, Copy, Debug, PartialEq)]
    struct Three;

    #[derive(Clone, Copy)]
    struct Value<S: Slots, T: Coef> {
        c: [S::Arr<T>; 3],
    }

    impl<S: Slots, T: Coef> PartialEq for Value<S, T> {
        fn eq(&self, o: &Self) -> bool {
            self.c == o.c
        }
    }
    impl<S: Slots, T: Coef> fmt::Debug for Value<S, T> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            self.c.fmt(f)
        }
    }

    impl<S: Slots, T: Coef> Extensor for Value<S, T> {
        type Kind = Three;
        type Slots = S;
        type Coef = T;
        fn from_coeffs(c: [S::Arr<T>; 3]) -> Self {
            Value { c }
        }
        fn coeffs(&self) -> &[S::Arr<T>; 3] {
            &self.c
        }
    }

    impl Kind for Three {
        const N: usize = 3;
        const NAME: &'static str = "Three";
        const MODULE: &'static str = module_path!();
        const BLADES: &'static [&'static str] = &["a", "b", "c"];
        type Arr<X: Elem> = [X; 3];
        type Mv<S: Slots, T: Coef> = Value<S, T>;
        type Scalar = Three;
        type Orientation = crate::Extrinsic;
        const DEFAULT_INTRINSIC: bool = false;
        fn arr_from_fn<X: Elem>(f: impl FnMut(usize) -> X) -> [X; 3] {
            core::array::from_fn(f)
        }
        fn arr_map<X: Elem, Y: Elem>(a: &[X; 3], mut f: impl FnMut(&X) -> Y) -> [Y; 3] {
            core::array::from_fn(|i| f(&a[i]))
        }
        fn arr_zip<X: Elem, Y: Elem, Z: Elem>(
            a: &[X; 3],
            b: &[Y; 3],
            mut f: impl FnMut(&X, &Y) -> Z,
        ) -> [Z; 3] {
            core::array::from_fn(|i| f(&a[i], &b[i]))
        }
    }

    // SAFETY: `Arr<X>` is `[X; 3]`.
    unsafe impl FlatKind for Three {}

    /// The flat view reads and writes every site's coefficients in order (run under miri in
    /// CI, which checks the `unsafe` view).
    #[test]
    fn flat_view() {
        let mut a: SiteArr<Three, f64, 4> = <Sites<Three, 4> as Kind>::arr_from_fn(|i| i as f64);
        assert_eq!(a.as_ref().len(), 12);
        assert_eq!(a.0[2][1], 7.0);
        a.as_mut()[11] = -1.0;
        assert_eq!(a.0[3][2], -1.0);
        assert_eq!(<Sites<Three, 4> as Cast<Sites<Three, 4>>>::SHARED.len(), 12);
    }
}
