//! Certified unit versors.

use core::ops::Deref;

/// A versor certified to satisfy `x ~x = 1`.
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
