//! SIMD lane coefficients from the `wide` crate (feature `wide`).
//!
//! A multivector with lane coefficients is a batch in struct-of-arrays form: a
//! `Point<(), f32x8>` holds eight points, and every generated kernel, map and solver runs on
//! all lanes at once without change.

use crate::coef::{Coef, Real};
use wide::{f32x4, f32x8, f64x2, f64x4};

macro_rules! lanes {
    ($t:ident, $s:ident) => {
        impl Coef for $t {
            #[inline(always)]
            fn zero() -> Self {
                $t::ZERO
            }
            #[inline(always)]
            fn one() -> Self {
                $t::ONE
            }
            #[inline(always)]
            fn from_i64(i: i64) -> Self {
                $t::splat(i as $s)
            }
            #[inline(always)]
            fn from_f64(f: f64) -> Self {
                $t::splat(f as $s)
            }
        }

        impl Real for $t {
            #[inline(always)]
            fn sqrt(self) -> Self {
                $t::sqrt(self)
            }
            /// Exact `1 / x` (the `wide` method of the same name is an approximation).
            #[inline(always)]
            fn recip(self) -> Self {
                $t::ONE / self
            }
            #[inline(always)]
            fn abs(self) -> Self {
                $t::abs(self)
            }
            #[inline(always)]
            fn sin(self) -> Self {
                $t::sin_cos(self).0
            }
            #[inline(always)]
            fn cos(self) -> Self {
                $t::sin_cos(self).1
            }
            #[inline(always)]
            fn sin_cos(self) -> (Self, Self) {
                $t::sin_cos(self)
            }
            #[inline(always)]
            fn sinh(self) -> Self {
                $t::sinh(self)
            }
            #[inline(always)]
            fn cosh(self) -> Self {
                $t::cosh(self)
            }
            #[inline(always)]
            fn atan2(self, x: Self) -> Self {
                $t::atan2(self, x)
            }
            #[inline(always)]
            fn ln(self) -> Self {
                $t::ln(self)
            }
            #[inline(always)]
            fn select_lt(a: Self, b: Self, x: Self, y: Self) -> Self {
                a.simd_lt(b).select(x, y)
            }
            #[inline(always)]
            fn all_lt(a: Self, b: Self) -> bool {
                a.simd_lt(b).all()
            }
            #[inline(always)]
            fn max(self, o: Self) -> Self {
                $t::max(self, o)
            }
            #[inline(always)]
            fn min(self, o: Self) -> Self {
                $t::min(self, o)
            }
            #[inline(always)]
            fn epsilon() -> Self {
                $t::splat(<$s>::EPSILON)
            }
            #[inline(always)]
            fn mul_add(self, a: Self, b: Self) -> Self {
                $t::mul_add(self, a, b)
            }
        }
    };
}

lanes!(f32x4, f32);
lanes!(f32x8, f32);
lanes!(f64x2, f64);
lanes!(f64x4, f64);

pub use wide;
