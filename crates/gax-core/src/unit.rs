//! Certified unit versors.

use core::ops::Deref;

/// A versor certified to satisfy `x ~x = 1`.
///
/// ```
/// use gax::pga3d::{Line, Motor, Point};
/// let m: gax::Unit<Motor<(), f64>> = Line::new(0.0, 0.0, 0.3, 1.0, 0.0, 0.0).exp();
/// let p = Point::xyz(1.0, 0.0, 0.0);
/// let q = m >> p;                           // simplified sandwich
/// let back = m.inverse() >> q;              // the inverse is the reverse: no arithmetic
/// assert!((back.e032() - 1.0).abs() < 1e-12);
/// ```
///
/// The certificate is a promise made at construction, as with `nalgebra::Unit`: operations
/// on a `Unit` use it to take cheaper paths (the inverse is the reverse, sandwiches use the
/// simplified formulas). Obtain one from `normalized()`, from products of unit versors, or
/// assert it with [`Unit::new_unchecked`].
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(transparent)]
pub struct Unit<M>(M);

impl<M> Unit<M> {
    /// Certify `m` as a unit versor without checking. If `m ~m` is not 1, results of
    /// operations that rely on the certificate are unspecified (but memory safe).
    #[inline(always)]
    pub const fn new_unchecked(m: M) -> Self {
        Unit(m)
    }

    /// The underlying versor.
    #[inline(always)]
    pub fn into_inner(self) -> M {
        self.0
    }

    /// The same versor as a kind that contains it, such as a translator or a rotor as a
    /// motor, or its image under a homomorphism between algebras (a PGA2D motor as a PGA3D
    /// one). Only conversions that keep `x ~x = 1` qualify ([`Widen`]).
    ///
    /// ```
    /// use gax::pga3d::{Motor, Point, Translator};
    /// use gax::Unit;
    /// let t: Unit<Translator<(), f64>> = (Point::xyz(1.0, 0.0, 0.0) / Point::xyz(0.0, 0.0, 0.0)).sqrt();
    /// let m: Unit<Motor<(), f64>> = t.widen();
    /// assert_eq!((m >> Point::xyz(0.0, 0.0, 0.0)).to_euclidean(), [1.0, 0.0, 0.0]);
    /// ```
    #[inline(always)]
    pub fn widen<N: From<M>>(self) -> Unit<N>
    where
        M: Widen<N>,
    {
        Unit(N::from(self.0))
    }
}

/// A conversion `M -> N` that keeps the unit condition, so [`Unit::widen`] may carry the
/// certificate across. The generated algebras implement it for embeddings between kinds and
/// for the homomorphisms between algebras that commute with the reverse (`φ(~a) = ~φ(a)`, proved
/// by the generator): not, for instance, the spacetime split on vectors, which sends a unit
/// vector to a bivector `B` with `B ~B = -1`.
pub trait Widen<N> {}

/// One Newton step towards the unit condition, `x (3 − x ~x) / 2` (implemented by the
/// generated algebras for the kinds whose norm is a Study number). For `x ~x = 1 + e` the result
/// is off by `O(e²)`, with no square root. See [`Unit::renormalize_fast`].
pub trait NewtonStep {
    /// `x (3 − x ~x) / 2`.
    fn newton_step(self) -> Self;
}

impl<M: NewtonStep> Unit<M> {
    /// Pull a drifted unit versor back towards `x ~x = 1` with one Newton step,
    /// `x (3 − x ~x) / 2`: an error `e` in the norm becomes `O(e²)`, for a few multiplications and
    /// no square root. `normalized()` is the exact path.
    ///
    /// A policy that keeps the certificate honest: call it after every integration step, or
    /// after every few products of unit versors.
    ///
    /// ```
    /// use gax::pga3d::Line;
    /// use gax::Unit;
    /// let u = Line::<(), f64>::new(0.4, -0.3, 0.8, 0.9, -0.5, 0.2).exp();
    /// let drifted = Unit::new_unchecked(u.into_inner().gp(1.0 + 1e-4));
    /// let fixed = drifted.renormalize_fast();
    /// let err = |m: Unit<gax::pga3d::Motor<(), f64>>| (m.into_inner().norm_squared() - 1.0).abs();
    /// assert!(err(drifted) > 1e-4 && err(fixed) < 1e-7);
    /// ```
    #[inline(always)]
    #[must_use]
    pub fn renormalize_fast(self) -> Self {
        Unit(self.0.newton_step())
    }
}

impl<M: NewtonStep + core::ops::Mul<Output = M>> Unit<M> {
    /// The product of two unit versors, renormalized with one Newton step
    /// ([`renormalize_fast`](Unit::renormalize_fast)). `*` keeps the certificate without
    /// renormalizing, like nalgebra and glam; use this in long chains of compositions.
    #[inline(always)]
    #[must_use]
    pub fn mul_renormalized(self, other: Self) -> Self {
        Unit((self.0 * other.0).newton_step())
    }
}

impl<M> Deref for Unit<M> {
    type Target = M;
    #[inline(always)]
    fn deref(&self) -> &M {
        &self.0
    }
}

impl<M, R> core::ops::Shr<R> for Unit<M>
where
    Unit<M>: crate::ops::Transform<R>,
{
    type Output = <Unit<M> as crate::ops::Transform<R>>::Output;
    #[inline(always)]
    fn shr(self, rhs: R) -> Self::Output {
        crate::ops::Transform::transform(self, rhs)
    }
}

impl<M, R> core::ops::Shl<R> for Unit<M>
where
    Unit<M>: crate::ops::TransformInv<R>,
{
    type Output = <Unit<M> as crate::ops::TransformInv<R>>::Output;
    #[inline(always)]
    fn shl(self, rhs: R) -> Self::Output {
        crate::ops::TransformInv::transform_inv(self, rhs)
    }
}

impl<M: crate::ops::Reverse<Output = M>> Unit<M> {
    /// The inverse of a unit versor: its reverse (no arithmetic).
    #[inline(always)]
    pub fn inverse(self) -> Unit<M> {
        Unit(self.0.reverse())
    }

    /// The reverse, which for a unit versor is also its inverse.
    #[inline(always)]
    pub fn reverse(self) -> Unit<M> {
        Unit(self.0.reverse())
    }
}

impl<M> Unit<M> {
    /// The logarithm, a bivector `B` with `B.exp() == self`.
    #[inline(always)]
    pub fn log<B>(self) -> B
    where
        Self: crate::ops::Log<B>,
    {
        crate::ops::Log::log(self)
    }
}

/// The product of unit versors is a unit versor.
impl<M: crate::ops::Gp<N>, N> core::ops::Mul<Unit<N>> for Unit<M> {
    type Output = Unit<M::Output>;
    #[inline(always)]
    fn mul(self, rhs: Unit<N>) -> Self::Output {
        Unit(self.0.gp(rhs.0))
    }
}
