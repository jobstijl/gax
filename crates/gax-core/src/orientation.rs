//! Orientation types (ADR-051): every value and map carries one of two orientations, as a type
//! parameter.
//!
//! An element is oriented either within itself, *intrinsic* (a point by its weight, a line by a
//! direction along it, a plane by the winding of a triangle in it), or by the space around it,
//! *extrinsic* (a plane by its positive side, a line by a turn about it, a point by a handedness
//! around it). A mirror reverses the space around things and leaves what lies within them, so the
//! two kinds reflect with opposite signs. With the sign chosen by the orientation, every versor,
//! odd ones included, commutes with every product: the join of reflected points is the reflected
//! join, and a reflected momentum keeps its kinetic energy.
//!
//! Orientations combine as signs multiply, extrinsic as `+` and intrinsic as `−`: the geometric,
//! outer and inner products give the product of their factors' signs, the regressive product its
//! negative (it is the outer product of the duals, dualized), the dual flips. The names, and the
//! reading of projective duality as the change of orientation, are Leo Dorst's (*Projective
//! duality encodes complementary orientations in geometric algebras*, 2024); numga carries the same
//! rules.
//!
//! ```
//! use gax::pga3d::{Line, Plane, Point};
//! use gax::{Extrinsic, Intrinsic};
//!
//! // Points are intrinsic, and so is the line joining them; planes are extrinsic, and so is
//! // their meet.
//! let (p, q) = (Point::<(), f64>::xyz(0.0, 0.0, 0.0), Point::xyz(0.0, 0.0, 1.0));
//! let joined: Line<(), f64, Intrinsic> = p & q;
//! let met: Line<(), f64, Extrinsic> = Plane::new(1.0, 0.0, 0.0, 0.0) ^ Plane::new(0.0, 1.0, 0.0, 0.0);
//! // The mirror x = 0 contains the z axis: it keeps the joined line and reverses the turn about it.
//! let mirror = Plane::<(), f64>::new(1.0, 0.0, 0.0, 0.0);
//! assert_eq!(mirror >> joined, joined);
//! assert_eq!(mirror >> met, -met);
//! ```

use core::fmt::Debug;
use core::hash::Hash;

mod sealed {
    pub trait Sealed {}
    impl Sealed for super::Extrinsic {}
    impl Sealed for super::Intrinsic {}
}

/// One of the two orientations: [`Extrinsic`] or [`Intrinsic`].
pub trait Orientation:
    Copy
    + Default
    + Debug
    + PartialEq
    + Eq
    + Hash
    + Send
    + Sync
    + 'static
    + sealed::Sealed
    + IsExtrinsic
    + IsIntrinsic
{
    /// `+1` for extrinsic, `−1` for intrinsic: the sign the orientation gives a reflection.
    const SIGN: i8;
    /// `""` for extrinsic, `"intrinsic "` for intrinsic: how a type's name is printed.
    const PREFIX: &'static str;
    /// The other orientation: what the dual makes of this one.
    type Flip: Orientation;
    /// The orientation of a product with an element of orientation `B`: extrinsic for equal
    /// orientations, intrinsic for different ones.
    type Times<B: Orientation>: Orientation;
    /// Whether `B` is this orientation, as a type-level boolean.
    type Same<B: Orientation>: crate::fill::Bit;
}

/// Oriented by the space around it: a plane by its side, a line by a turn about it. Vectors and
/// everything built from them by products are extrinsic.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Extrinsic;

/// Oriented within itself: a point by its weight, a line by a direction along it, a plane by a
/// winding in it. Duals of extrinsic elements, PGA's points, and joins of points are intrinsic.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Intrinsic;

impl Orientation for Extrinsic {
    const SIGN: i8 = 1;
    const PREFIX: &'static str = "";
    type Flip = Intrinsic;
    type Times<B: Orientation> = B;
    type Same<B: Orientation> = <B as IsExtrinsic>::Out;
}

impl Orientation for Intrinsic {
    const SIGN: i8 = -1;
    const PREFIX: &'static str = "intrinsic ";
    type Flip = Extrinsic;
    type Times<B: Orientation> = B::Flip;
    type Same<B: Orientation> = <B as IsIntrinsic>::Out;
}

/// Whether an orientation is extrinsic, as a type-level boolean (for [`Orientation::Same`]).
pub trait IsExtrinsic {
    /// `True` for [`Extrinsic`].
    type Out: crate::fill::Bit;
}
impl IsExtrinsic for Extrinsic {
    type Out = crate::fill::True;
}
impl IsExtrinsic for Intrinsic {
    type Out = crate::fill::False;
}

/// Whether an orientation is intrinsic, as a type-level boolean (for [`Orientation::Same`]).
pub trait IsIntrinsic {
    /// `True` for [`Intrinsic`].
    type Out: crate::fill::Bit;
}
impl IsIntrinsic for Extrinsic {
    type Out = crate::fill::False;
}
impl IsIntrinsic for Intrinsic {
    type Out = crate::fill::True;
}

/// The orientation of a product of elements of orientations `A` and `B`.
pub type Times<A, B> = <A as Orientation>::Times<B>;

/// The orientation of a regressive product (a join) of elements of orientations `A` and `B`: the
/// product's, flipped.
pub type Joined<A, B> = <<A as Orientation>::Times<B> as Orientation>::Flip;

/// The other orientation.
pub type Flip<A> = <A as Orientation>::Flip;

/// The same value, map or kind with orientation `O`: the coefficients kept, their orientation
/// declared anew. Generated for every kind; [`cast`](crate::cast) and the dual use it to keep or
/// flip an orientation, and values call it as `with_orientation::<O>()`.
pub trait Reorient<O: Orientation> {
    /// The same type with orientation `O`.
    type Out;
    /// The same coefficients, as `Self::Out`.
    fn reorient(self) -> Self::Out;
}

/// The sign an odd versor's sandwich takes on its result, by orientation (ADR-051): the plain
/// sandwich `m x ~m`, on each blade of grade `k`, times `(−1)ᵏ` for an extrinsic `x` and
/// `−(−1)ᵏ` for an intrinsic one. `odd[i]` says whether blade `i` has odd grade. The generated
/// reflections call it with constant arguments, so it folds to a negation of some coefficients.
#[inline(always)]
#[must_use]
pub fn reflect<O: Orientation, S: crate::slots::Slots, T: crate::coef::Coef, const N: usize>(
    mut c: [S::Arr<T>; N],
    odd: [bool; N],
) -> [S::Arr<T>; N] {
    for (x, &odd) in c.iter_mut().zip(&odd) {
        if odd == (O::SIGN > 0) {
            *x = (-crate::slots::SlotArr::<S, T>(*x)).0;
        }
    }
    c
}
