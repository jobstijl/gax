//! Lanes at a detected SIMD level: `fearless_simd` vectors.
//!
//! # Safety argument
//!
//! A `fearless_simd` vector carries a token proving that the CPU supports its level, and
//! every constructor needs one. The [`Coef`] and [`Batch`] constructors (`zero`, `splat`, ...)
//! take no arguments, so these lane types obtain the token with `assume_supported`. That is
//! sound because the level is part of the type, as `Proof<S>`, and `Proof` is private to this
//! crate: code outside it cannot name the lane types, and the only place that instantiates
//! them is [`run`](super::run), in the branch where `Level::as_*` has returned the token. A
//! lane value or a function over these types can therefore only come into existence after
//! the process has detected the level. (The detection is per process: a CPU does not lose
//! instruction sets while a program runs.)

use super::{Batch, math_simd};
use crate::coef::{Coef, Real};
use core::fmt;
use core::marker::PhantomData;
use core::ops::{Add, Div, Mul, Neg, Sub};
use fearless_simd::{Select, Simd, SimdBase, SimdFloat, SimdMask, f32x8, f64x4};

/// The proof that level `S` was detected (private: see the module documentation).
pub struct Proof<S>(PhantomData<fn() -> S>);

/// A detected level's token.
pub trait Proven: 'static {
    /// The token type.
    type S: Simd;
    /// The token.
    fn token() -> Self::S;
}

macro_rules! proven {
    ($($path:path),*) => {$(
        impl Proven for Proof<$path> {
            type S = $path;
            #[inline(always)]
            fn token() -> $path {
                // SAFETY: `Proof<$path>` is only instantiated by `run` after detecting the
                // level (see the module documentation).
                unsafe { <$path>::assume_supported() }
            }
        }
    )*};
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
proven!(
    fearless_simd::x86::Sse2,
    fearless_simd::x86::Sse4_2,
    fearless_simd::x86::Avx2,
    fearless_simd::x86::Avx512
);
#[cfg(target_arch = "aarch64")]
proven!(fearless_simd::aarch64::Neon);
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
impl Proven for Proof<fearless_simd::wasm32::WasmSimd128> {
    type S = fearless_simd::wasm32::WasmSimd128;
    #[inline(always)]
    fn token() -> Self::S {
        fearless_simd::wasm32::WasmSimd128::assume_supported()
    }
}

/// Eight `f32` lanes at a detected SIMD level.
#[repr(transparent)]
pub struct F32x8<P: Proven>(pub(crate) f32x8<P::S>);

/// Four `f64` lanes at a detected SIMD level.
#[repr(transparent)]
pub struct F64x4<P: Proven>(pub(crate) f64x4<P::S>);

macro_rules! common {
    ($T:ident, $v:ident, $e:ident, $n:literal) => {
        impl<P: Proven> Clone for $T<P> {
            #[inline(always)]
            fn clone(&self) -> Self {
                *self
            }
        }
        impl<P: Proven> Copy for $T<P> {}
        impl<P: Proven> PartialEq for $T<P> {
            #[inline]
            fn eq(&self, o: &Self) -> bool {
                self.0.as_slice() == o.0.as_slice()
            }
        }
        impl<P: Proven> fmt::Debug for $T<P> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.0.as_slice().fmt(f)
            }
        }
        impl<P: Proven> Add for $T<P> {
            type Output = Self;
            #[inline(always)]
            fn add(self, o: Self) -> Self {
                $T(self.0 + o.0)
            }
        }
        impl<P: Proven> Sub for $T<P> {
            type Output = Self;
            #[inline(always)]
            fn sub(self, o: Self) -> Self {
                $T(self.0 - o.0)
            }
        }
        impl<P: Proven> Mul for $T<P> {
            type Output = Self;
            #[inline(always)]
            fn mul(self, o: Self) -> Self {
                $T(self.0 * o.0)
            }
        }
        impl<P: Proven> Div for $T<P> {
            type Output = Self;
            #[inline(always)]
            fn div(self, o: Self) -> Self {
                $T(self.0 / o.0)
            }
        }
        impl<P: Proven> Neg for $T<P> {
            type Output = Self;
            #[inline(always)]
            fn neg(self) -> Self {
                $T(-self.0)
            }
        }
        impl<P: Proven> Coef for $T<P> {
            #[inline(always)]
            fn zero() -> Self {
                Self::splat(0.0)
            }
            #[inline(always)]
            fn one() -> Self {
                Self::splat(1.0)
            }
            #[inline(always)]
            fn from_i64(i: i64) -> Self {
                Self::splat(i as $e)
            }
            #[inline(always)]
            fn from_f64(f: f64) -> Self {
                Self::splat(f as $e)
            }
            #[inline(always)]
            fn vectorize<R>(f: impl FnOnce() -> R) -> R {
                P::token().vectorize(f)
            }
            /// Fused where the level has FMA (AVX2, AVX-512, NEON), `self * a + b` otherwise.
            #[inline(always)]
            fn mul_add(self, a: Self, b: Self) -> Self {
                $T(self.0.mul_add(a.0, b.0))
            }
        }
        impl<P: Proven> Batch for $T<P> {
            type Elem = $e;
            const LANES: usize = $n;
            #[inline(always)]
            fn splat(e: $e) -> Self {
                $T($v::splat(P::token(), e))
            }
            #[inline(always)]
            fn from_fn(f: impl FnMut(usize) -> $e) -> Self {
                $T($v::from_fn(P::token(), f))
            }
            #[inline(always)]
            fn lane(&self, i: usize) -> $e {
                self.0.as_slice()[i]
            }
            #[inline(always)]
            fn load(src: &[$e]) -> Self {
                $T($v::from_slice(P::token(), &src[..$n]))
            }
            #[inline(always)]
            fn store(self, dst: &mut [$e]) {
                self.0.store_slice(&mut dst[..$n]);
            }
        }
    };
}

common!(F32x8, f32x8, f32, 8);
common!(F64x4, f64x4, f64, 4);

macro_rules! real_common {
    () => {
        #[inline(always)]
        fn sqrt(self) -> Self {
            Self(self.0.sqrt())
        }
        #[inline(always)]
        fn abs(self) -> Self {
            Self(self.0.abs())
        }
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
        fn select_lt(a: Self, b: Self, x: Self, y: Self) -> Self {
            Self(a.0.simd_lt(b.0).select(x.0, y.0))
        }
        #[inline(always)]
        fn all_lt(a: Self, b: Self) -> bool {
            a.0.simd_lt(b.0).all_true()
        }
    };
}

impl<P: Proven> Real for F32x8<P> {
    real_common!();
    #[inline(always)]
    fn sin_cos(self) -> (Self, Self) {
        let (s, c) = math_simd::sin_cos(self.0);
        (F32x8(s), F32x8(c))
    }
    #[inline(always)]
    fn sinh(self) -> Self {
        F32x8(math_simd::sinh(self.0))
    }
    #[inline(always)]
    fn cosh(self) -> Self {
        F32x8(math_simd::cosh(self.0))
    }
    #[inline(always)]
    fn atan2(self, x: Self) -> Self {
        F32x8(math_simd::atan2(self.0, x.0))
    }
    #[inline(always)]
    fn ln(self) -> Self {
        F32x8(math_simd::ln(self.0))
    }
    #[inline(always)]
    fn epsilon() -> Self {
        Self::splat(f32::EPSILON)
    }
}

/// The `f64` elementary functions run lane by lane on the scalar implementation.
macro_rules! per_lane {
    ($self:ident, $f:expr) => {{
        let a = $self.0;
        F64x4(f64x4::from_fn(P::token(), |i| $f(a.as_slice()[i])))
    }};
}

impl<P: Proven> Real for F64x4<P> {
    real_common!();
    #[inline(always)]
    fn sin_cos(self) -> (Self, Self) {
        (per_lane!(self, f64::sin), per_lane!(self, f64::cos))
    }
    #[inline(always)]
    fn sinh(self) -> Self {
        per_lane!(self, f64::sinh)
    }
    #[inline(always)]
    fn cosh(self) -> Self {
        per_lane!(self, f64::cosh)
    }
    #[inline(always)]
    fn atan2(self, x: Self) -> Self {
        let (a, b) = (self.0, x.0);
        F64x4(f64x4::from_fn(P::token(), |i| {
            a.as_slice()[i].atan2(b.as_slice()[i])
        }))
    }
    #[inline(always)]
    fn ln(self) -> Self {
        per_lane!(self, f64::ln)
    }
    #[inline(always)]
    fn epsilon() -> Self {
        Self::splat(f64::EPSILON)
    }
}
