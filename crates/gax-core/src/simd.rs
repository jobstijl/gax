//! SIMD lane coefficients from the `wide` crate (feature `wide`).
//!
//! A multivector with lane coefficients is a batch in struct-of-arrays form: a
//! `Point<(), f32x8>` holds eight points, and every generated kernel, map and solver runs on
//! all lanes at once without change.

use crate::coef::{Coef, Real};
#[cfg(feature = "deterministic")]
use crate::strict::StrictElementary;
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
            #[inline(always)]
            fn mul_add(self, a: Self, b: Self) -> Self {
                #[cfg(not(feature = "deterministic"))]
                {
                    $t::mul_add(self, a, b)
                }
                #[cfg(feature = "deterministic")]
                {
                    self * a + b
                }
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
                self.sin_cos().0
            }
            #[inline(always)]
            fn cos(self) -> Self {
                self.sin_cos().1
            }
            #[cfg(not(feature = "deterministic"))]
            #[inline(always)]
            fn sin_cos(self) -> (Self, Self) {
                $t::sin_cos(self)
            }
            #[cfg(not(feature = "deterministic"))]
            #[inline(always)]
            fn sinh(self) -> Self {
                $t::sinh(self)
            }
            #[cfg(not(feature = "deterministic"))]
            #[inline(always)]
            fn cosh(self) -> Self {
                $t::cosh(self)
            }
            #[cfg(not(feature = "deterministic"))]
            #[inline(always)]
            fn atan2(self, x: Self) -> Self {
                $t::atan2(self, x)
            }
            #[cfg(not(feature = "deterministic"))]
            #[inline(always)]
            fn ln(self) -> Self {
                $t::ln(self)
            }
            // Deterministic: lane by lane, with the scalar type's functions.
            #[cfg(feature = "deterministic")]
            #[inline(always)]
            fn sin_cos(self) -> (Self, Self) {
                StrictElementary::strict_sin_cos(self)
            }
            #[cfg(feature = "deterministic")]
            #[inline(always)]
            fn sinh(self) -> Self {
                StrictElementary::strict_sinh(self)
            }
            #[cfg(feature = "deterministic")]
            #[inline(always)]
            fn cosh(self) -> Self {
                StrictElementary::strict_cosh(self)
            }
            #[cfg(feature = "deterministic")]
            #[inline(always)]
            fn atan2(self, x: Self) -> Self {
                StrictElementary::strict_atan2(self, x)
            }
            #[cfg(feature = "deterministic")]
            #[inline(always)]
            fn ln(self) -> Self {
                StrictElementary::strict_ln(self)
            }
            #[cfg(feature = "deterministic")]
            #[inline(always)]
            fn exp(self) -> Self {
                StrictElementary::strict_exp(self)
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
        }
    };
}

lanes!(f32x4, f32);
lanes!(f32x8, f32);
lanes!(f64x2, f64);
lanes!(f64x4, f64);

pub use wide;

#[cfg(test)]
mod tests {
    use super::*;

    /// The lanes' `exp` is `Real`'s default, built from the hyperbolic functions without
    /// cancellation: right across the range, overflowing to infinity and underflowing to zero
    /// (where `sinh x + cosh x` alone gives `inf - inf` or cancels).
    #[test]
    #[allow(clippy::float_cmp)]
    fn exp_from_the_hyperbolic_functions() {
        for x in [
            -120.0f32, -60.0, -10.0, -1.5, -1e-3, 0.0, 1e-3, 0.7, 10.0, 60.0, 120.0,
        ] {
            let lanes = Real::exp(f32x8::splat(x)).to_array()[0];
            let want = x.exp();
            if want.is_infinite() {
                assert_eq!(lanes, f32::INFINITY);
            } else {
                assert!(
                    (lanes - want).abs() <= 1e-5 * want + 1e-30,
                    "{x}: {lanes} vs {want}"
                );
            }
        }
    }
}
