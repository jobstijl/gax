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
//! use gax::pga3d::{Line, Plane, Point, intrinsic};
//!
//! // Points are intrinsic, and so is the line joining them; planes are extrinsic, and so is
//! // their meet. An algebra's root names each kind in its default orientation, `intrinsic` and
//! // `extrinsic` in either (ADR-052).
//! let (p, q) = (Point::xyz(0.0, 0.0, 0.0), Point::xyz(0.0, 0.0, 1.0));
//! let joined: intrinsic::Line = p & q;
//! let met: Line = Plane::new(1.0, 0.0, 0.0, 0.0) ^ Plane::new(0.0, 1.0, 0.0, 0.0);
//! // The mirror x = 0 contains the z axis: it keeps the joined line and reverses the turn about it.
//! let mirror = Plane::new(1.0, 0.0, 0.0, 0.0);
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

/// The arithmetic of orientations: the flip and the product. [`Orientation`] states its laws, so
/// that they hold for a generic orientation too (ADR-052).
pub trait Signs:
    Copy + Default + Debug + PartialEq + Eq + Hash + Send + Sync + 'static + sealed::Sealed
{
    /// The other orientation: what the dual makes of this one.
    type Flip: Orientation;
    /// The orientation of a product with an element of orientation `B`: extrinsic for equal
    /// orientations, intrinsic for different ones.
    type Times<B: Orientation>: Orientation;
    /// Whether this is [`Extrinsic`], as a type-level boolean.
    type Ext: crate::fill::Bit;
}

/// One of the two orientations: [`Extrinsic`] or [`Intrinsic`].
///
/// The supertraits are the sign laws, `o·(+) = o`, `−(−o) = o`, `o·(−) = −o` and `o·o = (+)`: with
/// them a product of a generic orientation `O` with a known one reduces (`Times<Times<O, I>, I>`
/// is `O`), so code generic over `O` typechecks. One law per bound, as Rust 1.89 requires.
pub trait Orientation:
    Signs<Times<Extrinsic> = Self, Flip: Signs<Flip = Self>>
    + Signs<Times<Intrinsic> = <Self as Signs>::Flip>
    + Signs<Times<Self> = Extrinsic>
{
    /// `+1` for extrinsic, `−1` for intrinsic: the sign the orientation gives a reflection.
    const SIGN: i8;
    /// `""` for extrinsic, `"intrinsic "` for intrinsic: how a type's name is printed.
    const PREFIX: &'static str;
}

/// Oriented by the space around it: a plane by its side, a line by a turn about it. Vectors and
/// everything built from them by products are extrinsic.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Extrinsic;

/// Oriented within itself: a point by its weight, a line by a direction along it, a plane by a
/// winding in it. Duals of extrinsic elements, PGA's points, and joins of points are intrinsic.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Intrinsic;

impl Signs for Extrinsic {
    type Flip = Intrinsic;
    type Times<B: Orientation> = B;
    type Ext = crate::fill::True;
}

impl Signs for Intrinsic {
    type Flip = Extrinsic;
    type Times<B: Orientation> = <B as Signs>::Flip;
    type Ext = crate::fill::False;
}

impl Orientation for Extrinsic {
    const SIGN: i8 = 1;
    const PREFIX: &'static str = "";
}

impl Orientation for Intrinsic {
    const SIGN: i8 = -1;
    const PREFIX: &'static str = "intrinsic ";
}

/// The orientation of a product of elements of orientations `A` and `B`.
pub type Times<A, B> = <A as Signs>::Times<B>;

/// The orientation of a regressive product (a join) of elements of orientations `A` and `B`: the
/// product's, flipped.
pub type Joined<A, B> = Flip<Times<A, B>>;

/// The other orientation.
pub type Flip<A> = <A as Signs>::Flip;

/// Whether `A` and `B` are the same orientation, as a type-level boolean: whether their product
/// is extrinsic.
pub type Same<A, B> = <Times<A, B> as Signs>::Ext;

/// The kind `K` in orientation `O`: `Oriented<Line, Intrinsic>` is `Line<(), f64, Intrinsic>`,
/// for naming a slot kind generically (ADR-052; an algebra's `intrinsic` and `extrinsic` modules
/// name each kind directly).
pub type Oriented<K, O> = <K as Reorient<O>>::Out;

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
