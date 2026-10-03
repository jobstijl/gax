//! Code written once over several algebras: [`Motions`], the rigid motions (or rotations, or
//! the Lorentz and Poincaré groups) of an algebra with their Lie algebra, its dual and points.
//!
//! The kinds of one algebra are distinct types, so a function over "the motor of PGA2D or
//! PGA3D" needs the kinds as parameters, and every operation it uses as a bound. [`Motions`]
//! collects those: a marker type per algebra names the kinds that play each role (rigid
//! motions in [`Pga2d`], [`Pga3d`], [`Cga2d`] and [`Cga3d`], rotations in [`Vga2d`] and
//! [`Vga3d`], the Lorentz group in [`Sta`], the Poincaré group in [`Stap`] and [`Csta`]), and
//! the operations a rigid-body or pose computation
//! needs come with them, as operators on the associated types (which a generic function gets
//! without writing any bounds) or as the trait's functions.
//!
//! ```
//! use gax::motions::{Linear, Motions, Pga2d, Pga3d};
//! use gax::{Extensor, Of};
//!
//! /// The kinetic energy of point masses moving with `rate`, in any algebra.
//! fn energy<G: Motions<f64>>(points: &[G::Coords], rate: G::Twist) -> f64 {
//!     let mut inertia = G::Inertia::zero();
//!     for c in points {
//!         inertia += G::point_inertia(G::point(*c));
//!     }
//!     G::pair(inertia.of(rate), rate) * 0.5
//! }
//!
//! let spin2 = gax::pga2d::Point::<(), f64>::new(0.0, 0.0, 1.0); // a turn about the origin
//! let spin3 = gax::pga3d::Line::<(), f64>::new(0.0, 0.0, 1.0, 0.0, 0.0, 0.0); // about z
//! let e2 = energy::<Pga2d>(&[[1.0, 0.0], [0.0, 2.0]], spin2);
//! let e3 = energy::<Pga3d>(&[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], spin3);
//! assert!((e2 - e3).abs() < 1e-12);
//! ```
//!
//! An algebra declared with `gax::algebra!` gets an implementation from [`motions!`](crate::motions!).

use crate::{Coef, Extensor, Kind, Of, Real, Slots};
use core::ops::{Add, AddAssign, Mul, Neg, Shl, Shr, Sub, SubAssign};

/// Values that add, subtract, scale by a coefficient and have a zero: the linear structure of
/// every kind, maps included (implemented for every such type).
pub trait Linear<T>:
    Extensor<Coef = T>
    + Copy
    + Add<Output = Self>
    + Sub<Output = Self>
    + Neg<Output = Self>
    + Mul<T, Output = Self>
    + AddAssign
    + SubAssign
{
    /// Zero.
    fn zero() -> Self;
}

impl<T: Coef, X> Linear<T> for X
where
    X: Extensor<Coef = T>
        + Copy
        + Add<Output = X>
        + Sub<Output = X>
        + Neg<Output = X>
        + Mul<T, Output = X>
        + AddAssign
        + SubAssign,
{
    #[inline]
    fn zero() -> Self {
        X::from_coeffs(<X::Kind as Kind>::arr_from_fn(|_| {
            <X::Slots as Slots>::from_flat(&mut |_| T::zero(), 0)
        }))
    }
}

/// A value (no open slots) with the linear structure of [`Linear`].
pub trait Value<T>: Linear<T> + Extensor<Slots = ()> {}

impl<T, X: Linear<T> + Extensor<Slots = ()>> Value<T> for X {}

/// The rigid motions (or, in a vector algebra, the rotations) of an algebra, implemented by a
/// marker type per algebra.
///
/// * [`Motor`](Motions::Motor): a pose, a unit even versor. Composed by `*`, and it moves
///   twists, forques and points by `>>` (and back by `<<`).
/// * [`Twist`](Motions::Twist): its Lie algebra, the bivectors: a rate (angular and, in PGA,
///   linear velocity), or the generator of a motion.
/// * [`Forque`](Motions::Forque): the dual of twists, paired with them by
///   [`pair`](Motions::pair): a momentum, a force line.
/// * [`Point`](Motions::Point): a mass point, at the Euclidean [`Coords`](Motions::Coords).
/// * The maps between them: [`Inertia`](Motions::Inertia) (twist to forque),
///   [`Mobility`](Motions::Mobility) (forque to twist), and [`TwistMap`](Motions::TwistMap).
///
/// gax's commutator is half the Lie bracket, as numga's.
pub trait Motions<T: Real>: Copy + 'static {
    /// A pose: a unit even versor.
    type Motor: Copy
        + core::fmt::Debug
        + Mul<Output = Self::Motor>
        + Shr<Self::Twist, Output = Self::Twist>
        + Shl<Self::Twist, Output = Self::Twist>
        + Shr<Self::Forque, Output = Self::Forque>
        + Shl<Self::Forque, Output = Self::Forque>
        + Shr<Self::Point, Output = Self::Point>
        + Shl<Self::Point, Output = Self::Point>;
    /// A twist: a bivector, the Lie algebra of [`Motor`](Motions::Motor).
    type Twist: Value<T>;
    /// A forque: the dual of twists.
    type Forque: Value<T>;
    /// A point.
    type Point: Value<T>;
    /// A map from twists to forques, such as an inertia.
    type Inertia: Linear<T> + Of<Self::Twist, Output = Self::Forque>;
    /// A map from forques to twists, such as an inverse inertia.
    type Mobility: Linear<T> + Of<Self::Forque, Output = Self::Twist>;
    /// A linear map on twists, such as `ad`.
    type TwistMap: Linear<T>
        + Of<Self::Twist, Output = Self::Twist>
        + Of<Self::TwistMap, Output = Self::TwistMap>;
    /// The Euclidean coordinates of a point, `[T; n]`.
    type Coords: Copy + core::fmt::Debug + AsRef<[T]> + AsMut<[T]>;
    /// The dimension of the space: the length of [`Coords`](Motions::Coords).
    const DIM: usize;

    /// The identity motion.
    fn identity() -> Self::Motor;
    /// The motion generated by a twist, `exp(b)`.
    fn exp(b: Self::Twist) -> Self::Motor;
    /// The twist generating a motion, `log(m)`.
    fn log(m: Self::Motor) -> Self::Twist;
    /// The inverse motion, `~m`.
    fn reverse(m: Self::Motor) -> Self::Motor;
    /// One Newton step back to `m ~m = 1` after drift (`Unit::renormalize_fast`).
    fn renormalize(m: Self::Motor) -> Self::Motor;
    /// The commutator of twists, `[a, b]` (zero where motions commute).
    fn commutator(a: Self::Twist, b: Self::Twist) -> Self::Twist;
    /// The commutator of a forque with a twist (the gyroscopic term `[p, r]`).
    fn coadjoint(f: Self::Forque, b: Self::Twist) -> Self::Forque;
    /// The pairing of a forque with a twist, `f & b` (power, or twice a kinetic energy).
    fn pair(f: Self::Forque, b: Self::Twist) -> T;
    /// The commutator with `h` as a map, `[h, ·]`.
    fn ad(h: Self::Twist) -> Self::TwistMap;
    /// The identity map on twists.
    fn identity_map() -> Self::TwistMap;
    /// The inertia of a unit mass at `p`: the twist to the forque `p & [p, b]`.
    fn point_inertia(p: Self::Point) -> Self::Inertia;
    /// The inverse of an inertia.
    fn mobility(inertia: Self::Inertia) -> Self::Mobility;
    /// The point at Euclidean coordinates `c`.
    fn point(c: Self::Coords) -> Self::Point;
    /// The Euclidean coordinates of a point.
    fn coords(p: Self::Point) -> Self::Coords;
    /// Coordinates from a function of the axis.
    fn coords_from_fn(f: impl FnMut(usize) -> T) -> Self::Coords;
}

/// Implement [`Motions`] for a marker type, from an algebra module's kinds.
///
/// ```ignore
/// gax::algebra! { vga4d: basis e1 = 1, e2 = 1, e3 = 1, e4 = 1; /* ... */ }
/// gax::motions! {
///     /// The rotations of 4D space.
///     pub struct Vga4d in crate::vga4d {
///         Motor = Even, Twist = Bivector, Forque = Bivector, Point = Trivector,
///         Coords = [T; 4],
///         point(c) = Vector::<(), T>::from_coeffs(c).dual(),
///         coords(p) = p.undual().c,
///     }
/// }
/// ```
///
/// Inside `point` and `coords` the module's kinds are in scope and `T` is the coefficient
/// type. Where motions commute (the rotations of the plane, whose bivectors have no
/// commutator), write `commutative struct` instead of `struct`. An optional last entry,
/// `mobility(i) = …`, replaces the inverse of the inertia `i` (the conformal algebras invert it
/// on the forques proper, see [`Cga3d`]).
#[macro_export]
macro_rules! motions {
    (
        $(#[$meta:meta])*
        $vis:vis struct $name:ident in $($m:ident)::+ {
            Motor = $Motor:ident, Twist = $Twist:ident, Forque = $Forque:ident, Point = $Point:ident,
            Coords = [T; $n:expr],
            point($c:ident) = $point:expr,
            coords($p:ident) = $coords:expr
            $(, mobility($mi:ident) = $mob:expr)? $(,)?
        }
    ) => {
        $crate::motions! { @impl ($(#[$meta])*) $vis $name ($($m)::+)
            $Motor $Twist $Forque $Point $n $c $point $p $coords [$($mi = $mob)?]
            commutator(a, b) = a.commutator(b);
            coadjoint(f, b) = f.commutator(b);
            ad(h) = h.commutator($Twist::slot());
        }
    };
    (
        $(#[$meta:meta])*
        $vis:vis commutative struct $name:ident in $($m:ident)::+ {
            Motor = $Motor:ident, Twist = $Twist:ident, Forque = $Forque:ident, Point = $Point:ident,
            Coords = [T; $n:expr],
            point($c:ident) = $point:expr,
            coords($p:ident) = $coords:expr
            $(, mobility($mi:ident) = $mob:expr)? $(,)?
        }
    ) => {
        $crate::motions! { @impl ($(#[$meta])*) $vis $name ($($m)::+)
            $Motor $Twist $Forque $Point $n $c $point $p $coords [$($mi = $mob)?]
            commutator(a, b) = { let _ = (a, b); $Twist::zero() };
            coadjoint(f, b) = { let _ = (f, b); $Forque::zero() };
            ad(h) = { let _ = h; $Twist::<($Twist,), T>::zero() };
        }
    };
    (@mobility $i:ident []) => {
        $i.inverse()
    };
    (@mobility $i:ident [$mi:ident = $mob:expr]) => {{
        let $mi = $i;
        $mob
    }};
    (
        @impl ($(#[$meta:meta])*) $vis:vis $name:ident ($($m:ident)::+)
        $Motor:ident $Twist:ident $Forque:ident $Point:ident $n:tt
        $c:ident $point:tt $p:ident $coords:tt $mobility:tt
        commutator($a:ident, $b:ident) = $comm:expr;
        coadjoint($f:ident, $fb:ident) = $coad:expr;
        ad($h:ident) = $ad:expr;
    ) => {
        $(#[$meta])*
        #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
        $vis struct $name;

        const _: () = {
            #[allow(unused_imports)]
            use $($m)::+::*;

            impl<T: $crate::Real> $crate::motions::Motions<T> for $name {
                type Motor = $crate::Unit<$Motor<(), T>>;
                type Twist = $Twist<(), T>;
                type Forque = $Forque<(), T>;
                type Point = $Point<(), T>;
                type Inertia = $Forque<($Twist,), T>;
                type Mobility = $Twist<($Forque,), T>;
                type TwistMap = $Twist<($Twist,), T>;
                type Coords = [T; $n];
                const DIM: usize = $n;

                #[inline]
                fn identity() -> Self::Motor {
                    $Twist::<(), T>::zero().exp()
                }
                #[inline]
                fn exp(b: Self::Twist) -> Self::Motor {
                    b.exp()
                }
                #[inline]
                fn log(m: Self::Motor) -> Self::Twist {
                    m.log()
                }
                #[inline]
                fn reverse(m: Self::Motor) -> Self::Motor {
                    m.reverse()
                }
                #[inline]
                fn renormalize(m: Self::Motor) -> Self::Motor {
                    m.renormalize_fast()
                }
                #[inline]
                fn commutator($a: Self::Twist, $b: Self::Twist) -> Self::Twist {
                    $comm
                }
                #[inline]
                fn coadjoint($f: Self::Forque, $fb: Self::Twist) -> Self::Forque {
                    $coad
                }
                #[inline]
                fn pair(f: Self::Forque, b: Self::Twist) -> T {
                    (f & b).s()
                }
                #[inline]
                fn ad($h: Self::Twist) -> Self::TwistMap {
                    $ad
                }
                #[inline]
                fn identity_map() -> Self::TwistMap {
                    $Twist::slot()
                }
                #[inline]
                fn point_inertia(p: Self::Point) -> Self::Inertia {
                    p & p.commutator($Twist::slot())
                }
                #[inline]
                fn mobility(inertia: Self::Inertia) -> Self::Mobility {
                    $crate::motions!(@mobility inertia $mobility)
                }
                #[inline]
                fn point($c: Self::Coords) -> Self::Point {
                    $point
                }
                #[inline]
                fn coords($p: Self::Point) -> Self::Coords {
                    $coords
                }
                #[inline]
                fn coords_from_fn(f: impl FnMut(usize) -> T) -> Self::Coords {
                    core::array::from_fn(f)
                }
            }
        };
    };
}

#[cfg(feature = "pga2d")]
motions! {
    /// The rigid motions of the plane, in PGA2D: motors, twists (points), forques (lines).
    pub struct Pga2d in crate::pga2d {
        Motor = Motor, Twist = Point, Forque = Line, Point = Point,
        Coords = [T; 2],
        point(c) = Point::xy(c[0], c[1]),
        coords(p) = p.to_euclidean(),
    }
}

#[cfg(feature = "pga3d")]
motions! {
    /// The rigid motions of space, in PGA3D: motors, twists and forques (lines), points.
    pub struct Pga3d in crate::pga3d {
        Motor = Motor, Twist = Line, Forque = Line, Point = Point,
        Coords = [T; 3],
        point(c) = Point::xyz(c[0], c[1], c[2]),
        coords(p) = p.to_euclidean(),
    }
}

#[cfg(feature = "vga2d")]
motions! {
    /// The rotations of the plane, in VGA2D: rotors, bivectors, scalars (their dual), and
    /// points as vectors (the dual of a vector, in 2D another vector). Rotations commute.
    pub commutative struct Vga2d in crate::vga2d {
        Motor = Rotor, Twist = Bivector, Forque = Scalar, Point = Vector,
        Coords = [T; 2],
        point(c) = Vector::<(), T>::from_coeffs(c).dual(),
        coords(p) = p.undual().c,
    }
}

#[cfg(feature = "vga3d")]
motions! {
    /// The rotations of space, in VGA3D: rotors, bivectors, vectors (their dual), and points
    /// as bivectors (the dual of a vector).
    pub struct Vga3d in crate::vga3d {
        Motor = Rotor, Twist = Bivector, Forque = Vector, Point = Bivector,
        Coords = [T; 3],
        point(c) = Vector::<(), T>::from_coeffs(c).dual(),
        coords(p) = p.undual().c,
    }
}

#[cfg(feature = "sta")]
motions! {
    /// The Lorentz group, in STA: rotors, bivectors (boosts and rotations) and their dual,
    /// bivectors again, and events as trivectors (the dual of a vector), at coordinates
    /// `(t, x, y, z)`. The metric is indefinite: an inertia's pairing is not positive.
    pub struct Sta in crate::sta {
        Motor = Even, Twist = Bivector, Forque = Bivector, Point = Trivector,
        Coords = [T; 4],
        point(c) = Vector::<(), T>::from_coeffs(c).dual(),
        coords(p) = p.undual().c,
    }
}

#[cfg(feature = "stap")]
motions! {
    /// The Poincaré group, in STAP (the projective algebra of spacetime): motors, bivectors
    /// and their dual, trivectors (lines), and events (quadvectors) at coordinates
    /// `(x, y, z, t)`. The metric is indefinite: an inertia's pairing is not positive.
    pub struct Stap in crate::stap {
        Motor = Motor, Twist = Bivector, Forque = Trivector, Point = Quadvector,
        Coords = [T; 4],
        point(c) = Quadvector::new(c[0], c[1], c[2], c[3], T::one()),
        coords(p) = {
            let w = p.c[4].recip();
            [p.c[0] * w, p.c[1] * w, p.c[2] * w, p.c[3] * w]
        },
    }
}

#[cfg(feature = "cga2d")]
motions! {
    /// The rigid motions of the plane, in CGA2D: motors, twists, bivectors as forques, and
    /// points as the dual of `up(x)`. Translations carry a forque into bivectors that pair with
    /// no twist, so a forque is a bivector up to those, and the mobility inverts the inertia
    /// on the forques proper, as in [`Cga3d`].
    pub struct Cga2d in crate::cga2d {
        Motor = Motor, Twist = Twist, Forque = Bivector, Point = Trivector,
        Coords = [T; 2],
        point(c) = Vector::<(), T>::up(c[0], c[1]).dual(),
        coords(p) = p.undual().down(),
        mobility(i) = {
            // Dualized, the forques proper are the twists; the rest leaves the twist kind.
            let proper: Twist<(Bivector,), T> = Bivector::slot().dual().cast::<Twist>();
            i.dual().cast::<Twist>().inverse().of(proper)
        },
    }
}

#[cfg(feature = "cga3d")]
motions! {
    /// The rigid motions of space, in CGA3D: motors, twists, trivectors as forques, and points
    /// as the dual of `up(x)` (quadvectors). A rotation's twist `e23` pairs with `e1oi`, a
    /// translation's `e1i` with `e23o`; the other four trivectors pair with no twist, and a
    /// translation carries a forque into them, so a forque is a trivector up to those (a
    /// quotient). Momenta keep those parts, which do no work; the mobility inverts the inertia
    /// on the forques proper and ignores them. Under the (metric-free) dual the forques proper
    /// are exactly the twists' blades and the rest leave the twist kind, so the projection is
    /// `f.dual().cast::<Twist>()`.
    pub struct Cga3d in crate::cga3d {
        Motor = Motor, Twist = Twist, Forque = Trivector, Point = Quadvector,
        Coords = [T; 3],
        point(c) = Vector::<(), T>::up(c[0], c[1], c[2]).dual(),
        coords(p) = p.undual().down(),
        mobility(i) = {
            // Dualized, the forques proper are the twists; the rest leaves the twist kind.
            let proper: Twist<(Trivector,), T> = Trivector::slot().dual().cast::<Twist>();
            i.dual().cast::<Twist>().inverse().of(proper)
        },
    }
}

#[cfg(feature = "csta")]
motions! {
    /// The Poincaré group, in CSTA: motors, twists, quadvectors as forques, and events as the
    /// dual of `up(x) = x + ½x² ei + eo` (with the Minkowski square), at coordinates
    /// `(x, y, z, t)`. As in [`Cga3d`], a forque is a quadvector up to the parts that pair with
    /// no twist, and the mobility inverts the inertia on the forques proper.
    pub struct Csta in crate::csta {
        Motor = Motor, Twist = Twist, Forque = Quadvector, Point = Quintvector,
        Coords = [T; 4],
        point(c) = {
            let square = c[0] * c[0] + c[1] * c[1] + c[2] * c[2] - c[3] * c[3];
            Vector::<(), T>::new(c[0], c[1], c[2], c[3], T::one(), square * T::from_f64(0.5)).dual()
        },
        coords(p) = {
            let v = p.undual();
            let w = v.c[4].recip();
            [v.c[0] * w, v.c[1] * w, v.c[2] * w, v.c[3] * w]
        },
        mobility(i) = {
            // Dualized, the forques proper are the twists; the rest leaves the twist kind.
            let proper: Twist<(Quadvector,), T> = Quadvector::slot().dual().cast::<Twist>();
            i.dual().cast::<Twist>().inverse().of(proper)
        },
    }
}
