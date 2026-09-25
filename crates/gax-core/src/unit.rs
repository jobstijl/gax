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
