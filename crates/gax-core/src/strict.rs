//! Deterministic coefficients: [`Strict<T>`], the same bits on every target and SIMD level.
//!
//! `Strict<f32>` and `Strict<f64>` compute what `f32` and `f64` do, with two differences: a
//! multiply-add is never fused, and the elementary functions come from pure Rust (`gax::math`
//! for `f32`, the `libm` crate for `f64`), not from the platform. So a computation on `Strict`
//! coefficients gives the same bits on every machine, every build and, through the batch
//! kernels, every SIMD level, while the rest of the program keeps plain `f32` with fused
//! multiply-adds. Lockstep simulations and replays use `Strict`; rendering need not.
//!
//! This is the per-value form of the `deterministic` feature, which makes plain `f32` and `f64`
//! behave like this everywhere (a cargo feature is on for the whole build, so it cannot be
//! chosen per use). `Strict` is `repr(transparent)`, and [`Strict::wrap`] and
//! [`Strict::unwrap`] convert whole values and maps without copying coefficient by
//! coefficient by hand.
//!
//! ```
//! use gax::strict::Strict;
//! use gax::pga3d::{Line, Point};
//!
//! type S = Strict<f32>;
//! let b = Line::<(), f32>::new(0.1, 0.2, 0.3, 0.4, 0.5, 0.6);
//! let m = Strict::wrap(b).exp();               // a Unit<Motor<(), Strict<f32>>>
//! let p = m >> Point::<(), S>::xyz(S::new(1.0), S::new(2.0), S::new(3.0));
//! let back: Point<(), f32> = Strict::unwrap(p); // to plain f32 for rendering
//! # let _ = back;
//! ```

use crate::coef::{Coef, Real};
use crate::kind::{Extensor, Retype};
use core::ops::{Add, Div, Mul, Neg, Sub};

/// A coefficient that computes with the same bits everywhere (see the [module](self)).
#[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct Strict<T>(pub T);

impl<T> Strict<T> {
    /// Wrap a number.
    #[inline(always)]
    pub const fn new(x: T) -> Self {
        Strict(x)
    }

    /// The number inside.
    #[inline(always)]
    pub fn get(self) -> T {
        self.0
    }
}

impl<T: Coef> Strict<T> {
    /// A value, map or form with `T` coefficients, as one with `Strict<T>` coefficients.
    #[inline]
    pub fn wrap<M: Extensor<Coef = T>>(m: M) -> Retype<M, M::Slots, Strict<T>>
    where
        Strict<T>: Coef,
    {
        m.map_coefs(Strict)
    }

    /// The reverse of [`Strict::wrap`]: `T` coefficients again.
    #[inline]
    pub fn unwrap<M: Extensor<Coef = Strict<T>>>(m: M) -> Retype<M, M::Slots, T>
    where
        Strict<T>: Coef,
    {
        m.map_coefs(|x| x.0)
    }
}

impl<T> From<T> for Strict<T> {
    #[inline(always)]
    fn from(x: T) -> Self {
        Strict(x)
    }
}

macro_rules! op {
    ($Tr:ident, $m:ident) => {
        impl<T: $Tr<Output = T>> $Tr for Strict<T> {
            type Output = Self;
            #[inline(always)]
            fn $m(self, o: Self) -> Self {
                Strict(self.0.$m(o.0))
            }
        }
    };
}
op!(Add, add);
op!(Sub, sub);
op!(Mul, mul);
op!(Div, div);

impl<T: Neg<Output = T>> Neg for Strict<T> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Strict(-self.0)
    }
}

impl<T: Coef> Coef for Strict<T> {
    #[inline(always)]
    fn zero() -> Self {
        Strict(T::zero())
    }
    #[inline(always)]
    fn one() -> Self {
        Strict(T::one())
    }
    #[inline(always)]
    fn from_i64(i: i64) -> Self {
        Strict(T::from_i64(i))
    }
    #[inline(always)]
    fn from_f64(f: f64) -> Self {
        Strict(T::from_f64(f))
    }
    #[inline(always)]
    fn from_ratio(num: i64, den: i64) -> Self {
        Strict(T::from_ratio(num, den))
    }
    #[inline(always)]
    fn check_unit(deviations: &[Self]) {
        for d in deviations {
            T::check_unit(&[d.0]);
        }
    }
    #[inline(always)]
    fn note_renormalize() {
        T::note_renormalize();
    }
    /// Never fused: `self * a + b`, rounded twice, on every target.
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        Strict(self.0 * a.0 + b.0)
    }
    #[inline(always)]
    fn vectorize<R>(f: impl FnOnce() -> R) -> R {
        T::vectorize(f)
    }
}

/// Number types with elementary functions that give the same bits everywhere: pure Rust, and
/// for lane types lane for lane the scalar's. Implemented for `f32`, `f64` and the lane types;
/// what [`Strict`] computes its elementary functions with.
pub trait StrictElementary: Real {
    /// Sine and cosine.
    fn strict_sin_cos(self) -> (Self, Self);
    /// Hyperbolic sine.
    fn strict_sinh(self) -> Self;
    /// Hyperbolic cosine.
    fn strict_cosh(self) -> Self;
    /// Four-quadrant arctangent of `self / x`.
    fn strict_atan2(self, x: Self) -> Self;
    /// Natural logarithm.
    fn strict_ln(self) -> Self;
    /// `e^x`.
    fn strict_exp(self) -> Self;
}

impl StrictElementary for f32 {
    #[inline(always)]
    fn strict_sin_cos(self) -> (Self, Self) {
        crate::math::sin_cos(self)
    }
    #[inline(always)]
    fn strict_sinh(self) -> Self {
        crate::math::sinh(self)
    }
    #[inline(always)]
    fn strict_cosh(self) -> Self {
        crate::math::cosh(self)
    }
    #[inline(always)]
    fn strict_atan2(self, x: Self) -> Self {
        crate::math::atan2(self, x)
    }
    #[inline(always)]
    fn strict_ln(self) -> Self {
        crate::math::ln(self)
    }
    #[inline(always)]
    fn strict_exp(self) -> Self {
        crate::math::exp(self)
    }
}

impl StrictElementary for f64 {
    #[inline(always)]
    fn strict_sin_cos(self) -> (Self, Self) {
        (libm::sin(self), libm::cos(self))
    }
    #[inline(always)]
    fn strict_sinh(self) -> Self {
        libm::sinh(self)
    }
    #[inline(always)]
    fn strict_cosh(self) -> Self {
        libm::cosh(self)
    }
    #[inline(always)]
    fn strict_atan2(self, x: Self) -> Self {
        libm::atan2(self, x)
    }
    #[inline(always)]
    fn strict_ln(self) -> Self {
        libm::log(self)
    }
    #[inline(always)]
    fn strict_exp(self) -> Self {
        libm::exp(self)
    }
}

impl<T: StrictElementary> Real for Strict<T> {
    const SCALAR: bool = T::SCALAR;
    #[inline(always)]
    fn sqrt(self) -> Self {
        Strict(self.0.sqrt())
    }
    #[inline(always)]
    fn recip(self) -> Self {
        Strict(T::one() / self.0)
    }
    #[inline(always)]
    fn abs(self) -> Self {
        Strict(self.0.abs())
    }
    #[inline(always)]
    fn sin(self) -> Self {
        Strict(self.0.strict_sin_cos().0)
    }
    #[inline(always)]
    fn cos(self) -> Self {
        Strict(self.0.strict_sin_cos().1)
    }
    #[inline(always)]
    fn sin_cos(self) -> (Self, Self) {
        let (s, c) = self.0.strict_sin_cos();
        (Strict(s), Strict(c))
    }
    #[inline(always)]
    fn sinh(self) -> Self {
        Strict(self.0.strict_sinh())
    }
    #[inline(always)]
    fn cosh(self) -> Self {
        Strict(self.0.strict_cosh())
    }
    #[inline(always)]
    fn atan2(self, x: Self) -> Self {
        Strict(self.0.strict_atan2(x.0))
    }
    #[inline(always)]
    fn ln(self) -> Self {
        Strict(self.0.strict_ln())
    }
    #[inline(always)]
    fn exp(self) -> Self {
        Strict(self.0.strict_exp())
    }
    #[inline(always)]
    fn select_lt(a: Self, b: Self, x: Self, y: Self) -> Self {
        Strict(T::select_lt(a.0, b.0, x.0, y.0))
    }
    #[inline(always)]
    fn all_lt(a: Self, b: Self) -> bool {
        T::all_lt(a.0, b.0)
    }
    #[inline(always)]
    fn max(self, o: Self) -> Self {
        Strict(self.0.max(o.0))
    }
    #[inline(always)]
    fn min(self, o: Self) -> Self {
        Strict(self.0.min(o.0))
    }
    #[inline(always)]
    fn epsilon() -> Self {
        Strict(T::epsilon())
    }
}

/// `StrictElementary` for a lane type of `f32` or `f64`, lane by lane with the scalar's
/// functions, through `to_array` and a constructor from an array.
#[cfg(any(feature = "wide", feature = "batch"))]
macro_rules! per_lane {
    ($(@[$($g:tt)*])? $ty:ty, $e:ty, $to:expr, $from:expr) => {
        impl$(<$($g)*>)? StrictElementary for $ty {
            #[inline(always)]
            fn strict_sin_cos(self) -> (Self, Self) {
                let sc = $to(self).map(<$e>::strict_sin_cos);
                ($from(sc.map(|p| p.0)), $from(sc.map(|p| p.1)))
            }
            #[inline(always)]
            fn strict_sinh(self) -> Self {
                $from($to(self).map(<$e>::strict_sinh))
            }
            #[inline(always)]
            fn strict_cosh(self) -> Self {
                $from($to(self).map(<$e>::strict_cosh))
            }
            #[inline(always)]
            fn strict_atan2(self, x: Self) -> Self {
                let (a, b) = ($to(self), $to(x));
                $from(core::array::from_fn(|i| a[i].strict_atan2(b[i])))
            }
            #[inline(always)]
            fn strict_ln(self) -> Self {
                $from($to(self).map(<$e>::strict_ln))
            }
            #[inline(always)]
            fn strict_exp(self) -> Self {
                $from($to(self).map(<$e>::strict_exp))
            }
        }
    };
}

#[cfg(feature = "wide")]
mod wide_lanes {
    use super::StrictElementary;
    use wide::{f32x4, f32x8, f64x2, f64x4};
    per_lane!(f32x4, f32, |x: f32x4| x.to_array(), f32x4::new);
    per_lane!(f32x8, f32, |x: f32x8| x.to_array(), f32x8::new);
    per_lane!(f64x2, f64, |x: f64x2| x.to_array(), f64x2::new);
    per_lane!(f64x4, f64, |x: f64x4| x.to_array(), f64x4::new);
}

#[cfg(feature = "batch")]
mod batch_lanes {
    use super::StrictElementary;
    use crate::batch::Lanes;

    per_lane!(@[const N: usize] Lanes<f32, N>, f32, |x: Lanes<f32, N>| x.v, Lanes::new);
    per_lane!(@[const N: usize] Lanes<f64, N>, f64, |x: Lanes<f64, N>| x.v, Lanes::new);
}
