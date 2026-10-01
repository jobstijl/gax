//! Portable lanes: `N` values in an array, combined lane by lane.

use super::Batch;
use crate::coef::{Coef, Real};
use crate::math;
use core::fmt;
use core::ops::{Add, Div, Mul, Neg, Sub};

/// `N` values processed together without a detected SIMD level (the [`Portable`] lanes).
///
/// The lanes are a plain array and every operation a loop over them, which the compiler
/// vectorizes as far as the target allows. Kernels on detected levels use `fearless_simd`
/// vectors instead.
#[derive(Clone, Copy, PartialEq)]
#[repr(transparent)]
pub struct Lanes<E, const N: usize> {
    /// The lanes.
    pub v: [E; N],
}

/// The level marker for code that runs without a detected SIMD level.
#[derive(Clone, Copy, Debug)]
pub enum Portable {}

impl<E: Copy, const N: usize> Lanes<E, N> {
    /// Wrap an array of lanes.
    #[inline(always)]
    pub const fn new(v: [E; N]) -> Self {
        Lanes { v }
    }
}

impl<E: fmt::Debug, const N: usize> fmt::Debug for Lanes<E, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.v.fmt(f)
    }
}

macro_rules! binop {
    ($e:ident, $Tr:ident, $f:ident, $op:tt) => {
        impl<const N: usize> $Tr for Lanes<$e, N> {
            type Output = Self;
            #[inline(always)]
            fn $f(mut self, o: Self) -> Self {
                for i in 0..N {
                    self.v[i] $op o.v[i];
                }
                self
            }
        }
    };
}

macro_rules! lane_map {
    ($name:ident, $f:expr) => {
        #[inline(always)]
        fn $name(mut self) -> Self {
            for i in 0..N {
                self.v[i] = $f(self.v[i]);
            }
            self
        }
    };
}

macro_rules! lanes {
    ($e:ident, $sin_cos:expr, $sinh:expr, $cosh:expr, $atan2:expr, $ln:expr, $exp:expr) => {
        binop!($e, Add, add, +=);
        binop!($e, Sub, sub, -=);
        binop!($e, Mul, mul, *=);
        binop!($e, Div, div, /=);

        impl<const N: usize> Neg for Lanes<$e, N> {
            type Output = Self;
            #[inline(always)]
            fn neg(mut self) -> Self {
                for i in 0..N {
                    self.v[i] = -self.v[i];
                }
                self
            }
        }

        impl<const N: usize> Coef for Lanes<$e, N> {
            #[inline(always)]
            fn zero() -> Self {
                Self::new([0.0; N])
            }
            #[inline(always)]
            fn one() -> Self {
                Self::new([1.0; N])
            }
            #[inline(always)]
            fn from_i64(i: i64) -> Self {
                Self::new([i as $e; N])
            }
            #[inline(always)]
            fn from_f64(f: f64) -> Self {
                Self::new([f as $e; N])
            }
            #[inline(always)]
            fn mul_add(mut self, a: Self, b: Self) -> Self {
                for i in 0..N {
                    self.v[i] = Coef::mul_add(self.v[i], a.v[i], b.v[i]);
                }
                self
            }
        }

        impl<const N: usize> Real for Lanes<$e, N> {
            lane_map!(sqrt, $e::sqrt);
            lane_map!(abs, $e::abs);
            lane_map!(sinh, $sinh);
            lane_map!(cosh, $cosh);
            lane_map!(ln, $ln);
            // The scalar type's `exp`, so a lane gives a scalar's bits.
            lane_map!(exp, $exp);
            #[inline(always)]
            fn recip(self) -> Self {
                Self::one() / self
            }
            #[inline(always)]
            fn sin(self) -> Self {
                self.sin_cos().0
            }
            #[inline(always)]
            fn cos(self) -> Self {
                self.sin_cos().1
            }
            #[inline(always)]
            fn sin_cos(self) -> (Self, Self) {
                let (mut s, mut c) = (self, self);
                for i in 0..N {
                    (s.v[i], c.v[i]) = $sin_cos(self.v[i]);
                }
                (s, c)
            }
            #[inline(always)]
            fn atan2(mut self, x: Self) -> Self {
                for i in 0..N {
                    self.v[i] = $atan2(self.v[i], x.v[i]);
                }
                self
            }
            #[inline(always)]
            fn select_lt(a: Self, b: Self, mut x: Self, y: Self) -> Self {
                for i in 0..N {
                    x.v[i] = if a.v[i] < b.v[i] { x.v[i] } else { y.v[i] };
                }
                x
            }
            #[inline(always)]
            fn all_lt(a: Self, b: Self) -> bool {
                let mut all = true;
                for i in 0..N {
                    all &= a.v[i] < b.v[i];
                }
                all
            }
            #[inline(always)]
            fn epsilon() -> Self {
                Self::new([$e::EPSILON; N])
            }
        }

        impl<const N: usize> Batch for Lanes<$e, N> {
            type Elem = $e;
            const LANES: usize = N;
            #[inline(always)]
            fn splat(e: $e) -> Self {
                Self::new([e; N])
            }
            #[inline(always)]
            fn from_fn(f: impl FnMut(usize) -> $e) -> Self {
                Self::new(core::array::from_fn(f))
            }
            #[inline(always)]
            fn lane(&self, i: usize) -> $e {
                self.v[i]
            }
            #[inline(always)]
            fn load(src: &[$e]) -> Self {
                let mut v = [0.0; N];
                v.copy_from_slice(&src[..N]);
                Self::new(v)
            }
            #[inline(always)]
            fn store(self, dst: &mut [$e]) {
                dst[..N].copy_from_slice(&self.v);
            }
        }
    };
}

lanes!(
    f32,
    math::sin_cos,
    math::sinh,
    math::cosh,
    math::atan2,
    math::ln,
    crate::coef::elementary::f32::exp
);
lanes!(
    f64,
    |x| (
        crate::coef::elementary::f64::sin(x),
        crate::coef::elementary::f64::cos(x)
    ),
    crate::coef::elementary::f64::sinh,
    crate::coef::elementary::f64::cosh,
    crate::coef::elementary::f64::atan2,
    crate::coef::elementary::f64::ln,
    crate::coef::elementary::f64::exp
);
