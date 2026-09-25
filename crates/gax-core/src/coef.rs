//! Coefficient traits.
//!
//! Every multivector, map and form is generic over its coefficient type. [`Coef`] is the
//! ring the generated products need; [`Real`] adds what norms, inverses, solvers and
//! exponentials need. `f32` and `f64` implement both; SIMD lane types (feature `wide`) and
//! the symbolic type used by build-time tracing implement them too.

use core::fmt::Debug;
use core::ops::{Add, Div, Mul, Neg, Sub};

/// Marker bound for everything stored in coefficient arrays.
pub trait Elem: Copy + Debug + PartialEq + 'static {}
impl<X: Copy + Debug + PartialEq + 'static> Elem for X {}

/// A coefficient ring: what the exact operation tables need.
///
/// Implementations must behave like real numbers under `+`, `-`, `*`: the generated code
/// relies on commutativity and associativity only up to floating-point rounding.
pub trait Coef:
    Elem + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Neg<Output = Self>
{
    /// The additive identity.
    fn zero() -> Self;
    /// The multiplicative identity.
    fn one() -> Self;
    /// Convert a small exact integer from an operation table.
    fn from_i64(i: i64) -> Self;
    /// Convert a constant.
    fn from_f64(f: f64) -> Self;
    /// `self * a + b`. Fused (one rounding) where the target has a fused multiply-add, which
    /// generated kernels use for every product that feeds a single sum.
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        self * a + b
    }
}

/// A real field with the elementary functions used by norms, inverses, solvers and exp/log.
///
/// For SIMD lane types every function acts lane-wise, and [`Real::select_lt`] replaces
/// branches, so the solvers built on this trait run unchanged across a batch.
///
/// Code written over `T: Real` runs on scalars, on SIMD lanes (feature `wide`) and, at build
/// time, on symbolic coefficients (`gax::trace`):
///
/// ```
/// use gax::pga3d::Point;
/// use gax::Real;
/// fn midpoint<T: Real>(a: Point<(), T>, b: Point<(), T>) -> Point<(), T> {
///     (a + b).gp(T::from_f64(0.5))
/// }
/// assert_eq!(midpoint(Point::xyz(0.0f32, 0.0, 0.0), Point::xyz(2.0, 0.0, 0.0)).to_euclidean(), [1.0, 0.0, 0.0]);
/// ```
pub trait Real: Coef + Div<Output = Self> {
    /// Square root.
    fn sqrt(self) -> Self;
    /// Reciprocal `1 / self`.
    fn recip(self) -> Self {
        Self::one() / self
    }
    /// Absolute value.
    fn abs(self) -> Self;
    /// Sine.
    fn sin(self) -> Self;
    /// Cosine.
    fn cos(self) -> Self;
    /// Sine and cosine together.
    fn sin_cos(self) -> (Self, Self) {
        (self.sin(), self.cos())
    }
    /// Hyperbolic sine.
    fn sinh(self) -> Self;
    /// Hyperbolic cosine.
    fn cosh(self) -> Self;
    /// Four-quadrant arctangent of `self / x`.
    fn atan2(self, x: Self) -> Self;
    /// Natural logarithm.
    fn ln(self) -> Self;
    /// Lane-wise `if a < b { x } else { y }`.
    fn select_lt(a: Self, b: Self, x: Self, y: Self) -> Self;
    /// Whether `a < b` holds in every lane (for a scalar, whether `a < b`). Iterative solvers
    /// use it to stop once every lane has converged; it is the only branch they take.
    fn all_lt(a: Self, b: Self) -> bool;
    /// Lane-wise maximum.
    fn max(self, o: Self) -> Self {
        Self::select_lt(self, o, o, self)
    }
    /// Lane-wise minimum.
    fn min(self, o: Self) -> Self {
        Self::select_lt(self, o, self, o)
    }
    /// Machine epsilon of the lane type.
    fn epsilon() -> Self;
}

macro_rules! float_impl {
    ($t:ident) => {
        impl Coef for $t {
            #[inline(always)]
            fn zero() -> Self {
                0.0
            }
            #[inline(always)]
            fn one() -> Self {
                1.0
            }
            #[inline(always)]
            fn from_i64(i: i64) -> Self {
                i as $t
            }
            #[inline(always)]
            fn from_f64(f: f64) -> Self {
                f as $t
            }
            #[inline(always)]
            fn mul_add(self, a: Self, b: Self) -> Self {
                // Only with hardware FMA: the software fallback is much slower than `*` and `+`.
                #[cfg(all(target_feature = "fma", feature = "std"))]
                {
                    extern crate std;
                    <$t>::mul_add(self, a, b)
                }
                #[cfg(not(all(target_feature = "fma", feature = "std")))]
                {
                    self * a + b
                }
            }
        }
        impl Real for $t {
            #[inline(always)]
            fn sqrt(self) -> Self {
                libm_shim::$t::sqrt(self)
            }
            #[inline(always)]
            fn abs(self) -> Self {
                libm_shim::$t::abs(self)
            }
            #[inline(always)]
            fn sin(self) -> Self {
                libm_shim::$t::sin(self)
            }
            #[inline(always)]
            fn cos(self) -> Self {
                libm_shim::$t::cos(self)
            }
            #[inline(always)]
            fn sinh(self) -> Self {
                libm_shim::$t::sinh(self)
            }
            #[inline(always)]
            fn cosh(self) -> Self {
                libm_shim::$t::cosh(self)
            }
            #[inline(always)]
            fn atan2(self, x: Self) -> Self {
                libm_shim::$t::atan2(self, x)
            }
            #[inline(always)]
            fn ln(self) -> Self {
                libm_shim::$t::ln(self)
            }
            #[inline(always)]
            fn select_lt(a: Self, b: Self, x: Self, y: Self) -> Self {
                if a < b { x } else { y }
            }
            #[inline(always)]
            fn all_lt(a: Self, b: Self) -> bool {
                a < b
            }
            #[inline(always)]
            fn max(self, o: Self) -> Self {
                <$t>::max(self, o)
            }
            #[inline(always)]
            fn min(self, o: Self) -> Self {
                <$t>::min(self, o)
            }
            #[inline(always)]
            fn epsilon() -> Self {
                <$t>::EPSILON
            }
        }
    };
}

float_impl!(f32);
float_impl!(f64);

/// Elementary functions for `f32`/`f64`: `std` when available, else the `libm` crate.
mod libm_shim {
    macro_rules! shim {
        ($t:ident, $libm:ident, [$($f:ident => $lf:ident ($($a:ident),*)),*]) => {
            pub mod $t {
                $(
                    #[inline(always)]
                    pub fn $f($($a: $t),*) -> $t {
                        #[cfg(feature = "std")]
                        {
                            extern crate std;
                            <$t>::$f($($a),*)
                        }
                        #[cfg(not(feature = "std"))]
                        {
                            libm::Libm::<$t>::$lf($($a),*)
                        }
                    }
                )*
            }
        };
    }
    shim!(f32, Libm, [sqrt => sqrt(x), abs => fabs(x), sin => sin(x), cos => cos(x), sinh => sinh(x),
        cosh => cosh(x), atan2 => atan2(y, x), ln => log(x)]);
    shim!(f64, Libm, [sqrt => sqrt(x), abs => fabs(x), sin => sin(x), cos => cos(x), sinh => sinh(x),
        cosh => cosh(x), atan2 => atan2(y, x), ln => log(x)]);
}
