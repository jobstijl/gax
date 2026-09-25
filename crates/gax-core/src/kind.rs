//! Kinds (typed subspaces) and the [`Extensor`] trait shared by all generated types.

use crate::coef::{Coef, Elem};
use crate::slots::Slots;
use core::fmt::Debug;

/// A typed subspace of an algebra, such as `Point` or `Motor`: an ordered list of oriented
/// basis blades.
///
/// A kind is written as its multivector type with default parameters, so `Point` is both
/// the type of a point value and the marker for an open point slot in `Line<(Point,)>`.
pub trait Kind: Copy + Debug + PartialEq + 'static {
    /// Number of coefficients.
    const N: usize;
    /// Name of the kind.
    const NAME: &'static str;
    /// Blade names in layout order, e.g. `["e032", "e013", "e021", "e123"]`.
    const BLADES: &'static [&'static str];
    /// Coefficient array of this kind: `[X; N]`.
    type Arr<X: Elem>: Elem + AsRef<[X]> + AsMut<[X]>;
    /// The multivector type of this kind with slots `S` and coefficients `T`.
    type Mv<S: Slots, T: Coef>: Extensor<Kind = Self, Slots = S, Coef = T>;
    /// The scalar kind of the same algebra.
    type Scalar: Kind;

    /// Build an array from its index.
    fn arr_from_fn<X: Elem>(f: impl FnMut(usize) -> X) -> Self::Arr<X>;
    /// Elementwise map.
    fn arr_map<X: Elem, Y: Elem>(a: &Self::Arr<X>, f: impl FnMut(&X) -> Y) -> Self::Arr<Y>;
    /// Elementwise combination.
    fn arr_zip<X: Elem, Y: Elem, Z: Elem>(
        a: &Self::Arr<X>,
        b: &Self::Arr<Y>,
        f: impl FnMut(&X, &Y) -> Z,
    ) -> Self::Arr<Z>;
}

/// The coefficient storage of a multivector type: one slot array per output coefficient.
pub type Coeffs<M> = <<M as Extensor>::Kind as Kind>::Arr<
    <<M as Extensor>::Slots as Slots>::Arr<<M as Extensor>::Coef>,
>;

/// A multivector, map or form of some kind: the common interface of all generated types.
pub trait Extensor: Copy {
    /// The output kind.
    type Kind: Kind;
    /// The open slots.
    type Slots: Slots;
    /// The coefficient type.
    type Coef: Coef;

    /// Construct from output-first coefficients.
    fn from_coeffs(c: Coeffs<Self>) -> Self;
    /// The output-first coefficients.
    fn coeffs(&self) -> &Coeffs<Self>;
}

/// Same kind, other slots and coefficients.
pub type Retype<M, S, T> = <<M as Extensor>::Kind as Kind>::Mv<S, T>;
