//! Prepared versor actions: a versor's map on one kind, with the structural zeros removed.
//!
//! `m >> Point::slot()` gives the dense `Point<(Point,)>` of a motor, and applying it multiplies
//! every coefficient, zeros included (IEEE rules forbid removing `0 * x`). `m.prepare::<Point>()`
//! keeps only the entries that are not structurally zero or constant, in a type that knows the
//! pattern, and `prepared >> p` applies them sparsely. Use it to transform many objects
//! with one versor.

use core::marker::PhantomData;

/// The action of versor `V` on kind `X`, stored as the `N` non-constant entries of its matrix.
///
/// ```
/// use gax::pga3d::{Motor, Point};
/// let m = Motor::<(), f64>::rotation_about(1.0, 0.0, 0.0, 0.5);
/// let t = m.prepare::<Point>();            // 12 entries: the zeros are in the type
/// let p = Point::xyz(1.0, 2.0, 3.0);
/// let (a, b) = ((t >> p).to_euclidean(), (m >> p).to_euclidean());
/// assert!(a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-12)); // same result
/// let dense: Point<(Point,), f64> = t.to_map();
/// # let _ = dense;
/// ```
///
/// `V` is the versor's kind, wrapped in `Unit` when the versor is certified (the entries then
/// come from the simplified formulas).
pub struct Prepared<V, X, T, const N: usize> {
    /// The non-constant matrix entries, in the order the generated code expects.
    pub m: [T; N],
    marker: PhantomData<fn() -> (V, X)>,
}

impl<V, X, T, const N: usize> Prepared<V, X, T, N> {
    /// Wrap entries computed by generated code.
    #[inline(always)]
    pub const fn from_entries(m: [T; N]) -> Self {
        Prepared {
            m,
            marker: PhantomData,
        }
    }
}

impl<V, X, T: Copy, const N: usize> Clone for Prepared<V, X, T, N> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}
impl<V, X, T: Copy, const N: usize> Copy for Prepared<V, X, T, N> {}

impl<V, X, T: core::fmt::Debug, const N: usize> core::fmt::Debug for Prepared<V, X, T, N> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Prepared").field("m", &self.m).finish()
    }
}

/// Prepare the action of a versor on kind `X` (implemented by the generated algebras).
pub trait Prepare<X> {
    /// The prepared action.
    type Output;
    /// Compute the non-constant entries of the versor's matrix on `X`.
    fn prepare(self) -> Self::Output;
}

impl<V, X, T, const N: usize, R> core::ops::Shr<R> for Prepared<V, X, T, N>
where
    Self: crate::ops::Transform<R>,
{
    type Output = <Self as crate::ops::Transform<R>>::Output;
    #[inline(always)]
    fn shr(self, rhs: R) -> Self::Output {
        crate::ops::Transform::transform(self, rhs)
    }
}

impl<M> crate::unit::Unit<M> {
    /// Prepare this versor's action on kind `X`, for applying it to many objects:
    /// `let t = m.prepare::<Point>(); for p in points { t >> p }`.
    ///
    /// ```
    /// use gax::ApproxEq;
    /// use gax::pga3d::{Motor, Point};
    /// let m = Motor::<(), f64>::rotation_about(1.0, 1.0, 0.0, 0.7);
    /// let t = m.prepare::<Point>();
    /// for p in [Point::xyz(1.0, 0.0, 0.0), Point::xyz(0.0, 2.0, -1.0)] {
    ///     assert!((t >> p).approx_eq(&(m >> p), 1e-12));
    /// }
    /// ```
    #[inline(always)]
    pub fn prepare<X>(self) -> <Self as Prepare<X>>::Output
    where
        Self: Prepare<X>,
    {
        Prepare::prepare(self)
    }
}

impl<V, X, T, const N: usize> Prepared<V, X, T, N> {
    /// The dense map of this action (for composing it with other maps): the entries written in
    /// place, without multiplying through an identity map.
    ///
    /// ```
    /// use gax::ApproxEq;
    /// use gax::pga3d::{Motor, Point};
    /// let m = Motor::<(), f64>::rotation_about(0.0, 0.0, 1.0, 0.3);
    /// let map: Point<(Point,), f64> = m.prepare::<Point>().to_map();
    /// assert!(map.approx_eq(&(m >> Point::slot()), 1e-12));
    /// ```
    #[inline(always)]
    pub fn to_map<M>(self) -> M
    where
        M: From<Self>,
    {
        M::from(self)
    }
}
