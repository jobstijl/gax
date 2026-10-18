//! Maps in any representation: dense extensors, and structured forms that store fewer numbers
//! (ADR-044).
//!
//! Every generated type stores its coefficients densely and implements [`Extensor`]. A
//! structured map, such as `pga3d::PrincipalInertia` (a `Line <- Line` stored as four numbers),
//! has no stored dense coefficients to lend out, so it is not an `Extensor`. It implements
//! [`LinearMap`] instead, which gives the dense coefficients by value, and the operations take
//! a `LinearMap` where a structured map should be accepted:
//!
//! * binding: `dense.of(structured)` composes (through the dense form), as does a form bound
//!   with it;
//! * the map's own action, solve and inverse: [`SquareLinearMap`], which every dense square map
//!   implements through [`SquareMap`], and a structured map implements with its fast paths.
//!
//! Stable Rust has no specialization, so a fast path cannot override a blanket impl. It does
//! not need to: a structured map is its own type, so its impls of these traits are its own,
//! beside the blanket impls for dense extensors.

use crate::coef::Real;
use crate::extensor::SquareMap;
use crate::kind::{Extensor, Kind};
use crate::slots::Slots;

/// The dense type of a [`LinearMap`]: the generated type with its kind, slots and coefficients.
pub type Dense<M> =
    <<M as LinearMap>::Kind as Kind>::Mv<<M as LinearMap>::Slots, <M as LinearMap>::Coef>;

/// A value, map or form in any representation, with its dense coefficients by value. Every
/// [`Extensor`] is one (its dense form is itself).
///
/// ```
/// use gax::map::LinearMap;
/// use gax::pga3d::{Forque, Line, PrincipalInertia};
/// let inertia = PrincipalInertia::new(2.0, [0.5, 0.75, 1.0]);
/// // From twists to momenta (forques, intrinsic lines).
/// let dense: Forque<(Line,), f64> = inertia.to_dense();
/// let b = Line::new(0.1, 0.2, 0.3, 1.0, -1.0, 0.5);
/// assert_eq!(dense.of(b), inertia.of(b));
/// ```
pub trait LinearMap: Copy {
    /// The output kind.
    type Kind: Kind;
    /// The open slots.
    type Slots: Slots;
    /// The coefficient type.
    type Coef: crate::coef::Coef;
    /// The dense coefficients, as the generated type.
    fn to_dense(self) -> Dense<Self>;
}

impl<M: Extensor> LinearMap for M {
    type Kind = M::Kind;
    type Slots = M::Slots;
    type Coef = M::Coef;
    #[inline(always)]
    fn to_dense(self) -> Dense<Self> {
        Extensor::from_coeffs(*self.coeffs())
    }
}

/// A map `Kind <- (Input,)` between kinds of one size, in any representation: what generic code
/// over maps is written against. Dense maps implement it through [`SquareMap`] (LU); a
/// structured map with its own paths.
///
/// ```
/// use gax::ApproxEq;
/// use gax::map::SquareLinearMap;
/// use gax::pga3d::{Forque, Line, PrincipalInertia};
///
/// // Written once, for any representation of an inertia (twists to momenta).
/// fn twist_of<I: SquareLinearMap<Kind = Forque, Input = Line, Coef = f64>>(
///     inertia: I,
///     momentum: Forque<(), f64>,
/// ) -> Line<(), f64> {
///     inertia.solve(momentum)
/// }
/// let compact = PrincipalInertia::new(2.0, [0.5, 0.75, 1.0]);
/// let f = Forque::from_coeffs([1.0, 0.0, -1.0, 0.5, 0.25, 2.0]);
/// assert!(twist_of(compact, f).approx_eq(&twist_of(compact.to_dense(), f), 1e-15));
/// ```
pub trait SquareLinearMap: LinearMap<Coef: Real> {
    /// The input kind.
    type Input: Kind;
    /// The inverse map's representation.
    type Inverse: LinearMap<Kind = Self::Input, Slots = (Self::Kind,), Coef = Self::Coef>;
    /// The image of a value.
    fn of(
        self,
        x: <Self::Input as Kind>::Mv<(), Self::Coef>,
    ) -> <Self::Kind as Kind>::Mv<(), Self::Coef>;
    /// Solve `self.of(x) == b` for `x`.
    fn solve(
        self,
        b: <Self::Kind as Kind>::Mv<(), Self::Coef>,
    ) -> <Self::Input as Kind>::Mv<(), Self::Coef>;
    /// The inverse map.
    fn inverse(self) -> Self::Inverse;
}

impl<M> SquareLinearMap for M
where
    M: SquareMap + Extensor<Slots = (<M as SquareMap>::Input,)>,
{
    type Input = <M as SquareMap>::Input;
    type Inverse = <M as SquareMap>::Inverse;

    #[inline]
    fn of(self, x: <Self::Input as Kind>::Mv<(), M::Coef>) -> <M::Kind as Kind>::Mv<(), M::Coef> {
        // Σᵢ xᵢ colᵢ per output coefficient, in the order and rounding of `of`.
        let xc = x.coeffs().as_ref();
        Extensor::from_coeffs(<M::Kind as Kind>::arr_map(self.coeffs(), |col| {
            let col = col.as_ref();
            (1..col.len()).fold(xc[0] * col[0], |acc, i| acc + xc[i] * col[i])
        }))
    }

    #[inline]
    fn solve(
        self,
        b: <M::Kind as Kind>::Mv<(), M::Coef>,
    ) -> <Self::Input as Kind>::Mv<(), M::Coef> {
        SquareMap::solve(self, b)
    }

    #[inline]
    fn inverse(self) -> Self::Inverse {
        SquareMap::inverse(self)
    }
}
