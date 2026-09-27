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
    /// Check that the parts of `u ~u − 1` of a `Unit` versor are zero within `√ε` (the
    /// `check-units` feature calls this whenever a certified kernel consumes a `Unit`). Does
    /// nothing by default; `f32` and `f64` panic when a part is too large.
    #[inline(always)]
    fn check_unit(_deviations: &[Self]) {}
    //// The rational constant `num / den`. Generated code writes non-integer constants this way,
    /// so that exact coefficient types (the symbolic `Sym`) get them exactly; for floating
    /// point it is `from_f64(num / den)`, folded at compile time.
    #[inline(always)]
    fn from_ratio(num: i64, den: i64) -> Self {
        Self::from_f64(num as f64 / den as f64)
    }
    /// `self * a + b`. Fused (one rounding) where the target has a fused multiply-add, which
    /// generated kernels use for every product that feeds a single sum.
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        self * a + b
    }
    /// Run `f` where operations on this type compile to its SIMD instructions: for the lanes
    /// of a detected level (`gax::batch`), inside a function compiled for that level; for
    /// every other type, just `f()`. Large generic functions wrap their bodies in it, so that
    /// they stay vectorized when the compiler does not inline them into a batch kernel.
    #[inline(always)]
    fn vectorize<R>(f: impl FnOnce() -> R) -> R {
        f()
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
    /// Whether this is a single number (`f32`, `f64`) rather than a batch of lanes. Solvers
    /// branch on scalars where lane types must select.
    const SCALAR: bool = false;
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
    /// `e^x`. The default builds it from the hyperbolic functions without cancellation:
    /// `sinh x + cosh x` for `x ≥ 0` (overflowing to infinity), its reciprocal at `-x` below
    /// (underflowing to zero), so it is correct wherever those are.
    fn exp(self) -> Self {
        let neg = -self;
        let up = self.sinh() + self.cosh();
        let down = (neg.sinh() + neg.cosh()).recip();
        Self::select_lt(self, Self::zero(), down, up)
    }
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
            fn check_unit(deviations: &[Self]) {
                let tol = libm_shim::$t::sqrt(<$t>::EPSILON);
                for (i, d) in deviations.iter().enumerate() {
                    assert!(
                        libm_shim::$t::abs(*d) <= tol,
                        "check-units: a Unit versor is not unit (part {i} of u ~u - 1 is {d}, tolerance {tol}); renormalize it (Unit::renormalize_fast or normalized)"
                    );
                }
            }
            #[inline(always)]
            fn mul_add(self, a: Self, b: Self) -> Self {
                // Only with hardware FMA: the software fallback is much slower than `*` and `+`.
                // With `deterministic`, never fused: results must not depend on the target.
                #[cfg(all(target_feature = "fma", feature = "std", not(feature = "deterministic")))]
                {
                    extern crate std;
                    <$t>::mul_add(self, a, b)
                }
                #[cfg(not(all(target_feature = "fma", feature = "std", not(feature = "deterministic"))))]
                {
                    self * a + b
                }
            }
        }
        impl Real for $t {
            const SCALAR: bool = true;
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
                elementary::$t::sin(self)
            }
            #[inline(always)]
            fn cos(self) -> Self {
                elementary::$t::cos(self)
            }
            #[inline(always)]
            fn sinh(self) -> Self {
                elementary::$t::sinh(self)
            }
            #[inline(always)]
            fn cosh(self) -> Self {
                elementary::$t::cosh(self)
            }
            #[inline(always)]
            fn atan2(self, x: Self) -> Self {
                elementary::$t::atan2(self, x)
            }
            #[inline(always)]
            fn ln(self) -> Self {
                elementary::$t::ln(self)
            }
            #[inline(always)]
            fn exp(self) -> Self {
                elementary::$t::exp(self)
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
/// The elementary functions of `f32` and `f64`: the platform's (through `std`) by default; with
/// the `deterministic` feature, pure Rust everywhere: `crate::math` for `f32` (the functions the
/// SIMD lanes use) and the `libm` crate for `f64`.
pub(crate) mod elementary {
    pub mod f32 {
        #[cfg(not(feature = "deterministic"))]
        pub use super::super::libm_shim::f32::{atan2, cos, cosh, exp, ln, sin, sinh};
        #[cfg(feature = "deterministic")]
        mod det {
            use crate::math;
            #[inline(always)]
            pub fn sin(x: f32) -> f32 {
                math::sin_cos(x).0
            }
            #[inline(always)]
            pub fn cos(x: f32) -> f32 {
                math::sin_cos(x).1
            }
            #[inline(always)]
            pub fn sinh(x: f32) -> f32 {
                math::sinh(x)
            }
            #[inline(always)]
            pub fn cosh(x: f32) -> f32 {
                math::cosh(x)
            }
            #[inline(always)]
            pub fn atan2(y: f32, x: f32) -> f32 {
                math::atan2(y, x)
            }
            #[inline(always)]
            pub fn ln(x: f32) -> f32 {
                math::ln(x)
            }
            #[inline(always)]
            pub fn exp(x: f32) -> f32 {
                math::exp(x)
            }
        }
        #[cfg(feature = "deterministic")]
        pub use det::{atan2, cos, cosh, exp, ln, sin, sinh};
    }
    pub mod f64 {
        #[cfg(not(feature = "deterministic"))]
        pub use super::super::libm_shim::f64::{atan2, cos, cosh, exp, ln, sin, sinh};
        #[cfg(feature = "deterministic")]
        mod det {
            macro_rules! libm_fns {
                ($($f:ident => $lf:ident ($($a:ident),*)),*) => {$(
                    #[inline(always)]
                    pub fn $f($($a: f64),*) -> f64 {
                        libm::Libm::<f64>::$lf($($a),*)
                    }
                )*};
            }
            libm_fns!(sin => sin(x), cos => cos(x), sinh => sinh(x), cosh => cosh(x), atan2 => atan2(y, x), ln => log(x), exp => exp(x));
        }
        #[cfg(feature = "deterministic")]
        pub use det::{atan2, cos, cosh, exp, ln, sin, sinh};
    }
}

#[cfg_attr(feature = "deterministic", allow(dead_code))]
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
        cosh => cosh(x), atan2 => atan2(y, x), ln => log(x), exp => exp(x)]);
    shim!(f64, Libm, [sqrt => sqrt(x), abs => fabs(x), sin => sin(x), cos => cos(x), sinh => sinh(x),
        cosh => cosh(x), atan2 => atan2(y, x), ln => log(x), exp => exp(x)]);
}
