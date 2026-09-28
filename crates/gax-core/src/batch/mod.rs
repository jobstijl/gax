//! Runtime-dispatched batch kernels (feature `batch`).
//!
//! Every generated kernel is generic over its coefficient type. This module supplies a lane
//! type, [`Lanes`], that runs a kernel on 8 `f32` (or 4 `f64`) values at once, and a
//! dispatcher, [`run`], that detects the CPU's SIMD level with [`fearless_simd`] and runs the
//! kernel compiled for that level (SSE2, SSE4.2, AVX2 with FMA, or AVX-512 on x86; NEON on
//! Arm; SIMD128 on WebAssembly). One binary thus uses AVX2 where the CPU has it, without
//! `-C target-cpu`.
//!
//! On top of that, the sandwich of every versor on every kind comes in batch form through
//! [`BatchTransform`], on array-of-structs slices and on struct-of-arrays [`Soa`] storage
//! (with `gax`'s `batch` feature; `docs/batch.md` has the tested version of this example):
//!
//! ```ignore
//! use gax::batch::{BatchTransform, Soa};
//! use gax::pga3d::{Motor, Point};
//! let m = Motor::<(), f32>::rotation_about(0.0, 0.0, 1.0, 0.5).normalized();
//! let points: Vec<Point> = (0..100).map(|i| Point::xyz(i as f32, 1.0, 0.0)).collect();
//! let mut moved = vec![Point::zero(); points.len()];
//! m.transform_slice(&points, &mut moved); // one prepared map, applied 8 points at a time
//! let soa: Soa<Point> = points.iter().copied().collect();
//! let mut out = Soa::new();
//! m.transform_soa(&soa, &mut out); // the same on struct-of-arrays storage
//! ```
//!
//! [`Lanes`] implements [`Real`] with vectorized `sin`, `cos`, `sinh`, `cosh`,
//! `atan2`, and `ln` for `f32` (see [`math`]), so kernels with exponentials and logarithms,
//! and kernels traced at build time, run batched too. For `f64` lanes the elementary
//! functions are evaluated lane by lane with the scalar implementation.
//!
//! Your own kernels run batched by implementing [`Map`] (one input kind) and calling
//! [`map`] or [`map_soa`], or, for any shape, by implementing [`Kernel`] and calling [`run`] (with
//! [`chunks`], [`gather`] and [`scatter`] for the array-of-structs plumbing).

mod kernels;
mod lanes;
mod math_simd;
mod simd_lanes;
mod soa;

/// The elementary functions of one `f32` lane (the same as [`crate::math`]).
pub use crate::math;
pub use fearless_simd;
pub use fearless_simd::Level;
pub use kernels::{
    BatchOf, BatchTransform, Certified, MAX_LANES, Map, OutOf, OutputOf, Plain, SandwichKernel,
    VersorType, chunks, chunks2, column, gather, map, map_soa, scatter, splat, to_array,
};
pub use lanes::{Lanes, Portable};
use simd_lanes::Proof;
pub use soa::{BLOCK, Soa, load_block, soa_map, soa_map2, store_block};

extern crate std;
use crate::coef::Real;
use crate::kind::Kind;
use core::cell::Cell;
use std::vec::Vec;

/// The value type of kind `K` with coefficients `T`.
pub type Mv<K, T> = <K as Kind>::Mv<(), T>;

/// A lane type: `LANES` values of `Elem` processed together.
pub trait Batch: Real {
    /// The scalar type of one lane.
    type Elem: LaneElem;
    /// The number of lanes (a divisor of [`BLOCK`]).
    const LANES: usize;
    /// Every lane set to `e`.
    fn splat(e: Self::Elem) -> Self;
    /// Lane `i` set to `f(i)`.
    fn from_fn(f: impl FnMut(usize) -> Self::Elem) -> Self;
    /// Lane `i`.
    fn lane(&self, i: usize) -> Self::Elem;
    /// The first `LANES` values of `src`.
    fn load(src: &[Self::Elem]) -> Self;
    /// Write the lanes to the first `LANES` values of `dst`.
    fn store(self, dst: &mut [Self::Elem]);
}

/// A SIMD level as a type: [`Portable`], or (unnameable outside this crate) a level the
/// dispatcher has detected. Sealed.
pub trait LevelMarker: sealed::Sealed + 'static {
    /// The `f32` lanes of the level.
    type F32: Batch<Elem = f32>;
    /// The `f64` lanes of the level.
    type F64: Batch<Elem = f64>;
}

mod sealed {
    pub trait Sealed {}
    impl Sealed for super::Portable {}
    impl<S> Sealed for super::Proof<S> {}
}

impl LevelMarker for Portable {
    type F32 = Lanes<f32, 8>;
    type F64 = Lanes<f64, 4>;
}

impl<S: 'static> LevelMarker for Proof<S>
where
    Proof<S>: simd_lanes::Proven,
{
    type F32 = simd_lanes::F32x8<Proof<S>>;
    type F64 = simd_lanes::F64x4<Proof<S>>;
}

/// A scalar type with a lane type for each SIMD level: `f32` (8 lanes) and `f64` (4 lanes).
pub trait LaneElem: Real {
    /// The lanes of this type at level `P`.
    type Lanes<P: LevelMarker>: Batch<Elem = Self>;
}
impl LaneElem for f32 {
    type Lanes<P: LevelMarker> = P::F32;
}
impl LaneElem for f64 {
    type Lanes<P: LevelMarker> = P::F64;
}

/// A computation that is generic over its lane type, for [`run`].
///
/// ```
/// use gax::batch::{self, Batch, Kernel};
/// struct Sum<'a>(&'a [f32]);
/// impl Kernel<f32> for Sum<'_> {
///     type Output = f32;
///     #[inline(always)]
///     fn run<L: Batch<Elem = f32>>(self) -> f32 {
///         let mut acc = L::splat(0.0);
///         let chunks = self.0.chunks_exact(L::LANES);
///         let tail: f32 = chunks.remainder().iter().sum();
///         for c in chunks {
///             acc = acc + L::load(c);
///         }
///         (0..L::LANES).map(|i| acc.lane(i)).sum::<f32>() + tail
///     }
/// }
/// let xs: Vec<f32> = (0..100).map(|i| i as f32).collect();
/// assert_eq!(batch::run(Sum(&xs)), 4950.0);
/// ```
pub trait Kernel<E: LaneElem> {
    /// The result.
    type Output;
    /// Run on lanes of type `L`. Mark implementations `#[inline(always)]`: the body is
    /// compiled into the dispatcher's per-level function only when it is inlined.
    fn run<L: Batch<Elem = E>>(self) -> Self::Output;
}

/// The level forced on a thread by [`with_level`].
#[derive(Clone, Copy)]
enum Forced {
    Auto,
    To(Option<Level>),
}

std::thread_local! {
    static FORCED: Cell<Forced> = const { Cell::new(Forced::Auto) };
}

/// The SIMD level [`run`] uses on this thread: the one set by [`with_level`], or the best the
/// CPU supports. `None` means portable code without a detected level.
#[must_use]
pub fn level() -> Option<Level> {
    match FORCED.get() {
        Forced::Auto => Some(Level::new()),
        Forced::To(level) => level,
    }
}

/// Run `f` with [`run`] forced to `level` on this thread (`None`: portable code). For tests and
/// benchmarks of the individual levels; see [`levels`].
pub fn with_level<R>(level: Option<Level>, f: impl FnOnce() -> R) -> R {
    struct Restore(Forced);
    impl Drop for Restore {
        fn drop(&mut self) {
            FORCED.set(self.0);
        }
    }
    let _restore = Restore(FORCED.replace(Forced::To(level)));
    f()
}

/// Every level this CPU can run, lowest first, starting with `None` (portable code).
#[must_use]
pub fn levels() -> Vec<Option<Level>> {
    let best = Level::new();
    #[allow(unused_mut)]
    let mut out = std::vec![None];
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        out.extend(best.as_sse2().map(|t| Some(Level::Sse2(t))));
        out.extend(best.as_sse4_2().map(|t| Some(Level::Sse4_2(t))));
        out.extend(best.as_avx2().map(|t| Some(Level::Avx2(t))));
        out.extend(best.as_avx512().map(|t| Some(Level::Avx512(t))));
    }
    #[cfg(target_arch = "aarch64")]
    out.extend(best.as_neon().map(|t| Some(Level::Neon(t))));
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    out.extend(best.as_wasm_simd128().map(|t| Some(Level::WasmSimd128(t))));
    let _ = best;
    out
}

/// A short name of a level, for reports.
#[must_use]
pub fn level_name(level: Option<Level>) -> &'static str {
    #[allow(unreachable_patterns)]
    match level {
        None => "portable",
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        Some(Level::Sse2(_)) => "sse2",
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        Some(Level::Sse4_2(_)) => "sse4.2",
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        Some(Level::Avx2(_)) => "avx2",
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        Some(Level::Avx512(_)) => "avx512",
        #[cfg(target_arch = "aarch64")]
        Some(Level::Neon(_)) => "neon",
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        Some(Level::WasmSimd128(_)) => "simd128",
        Some(_) => "other",
    }
}

/// Run `k` on the lanes of the current [`level`], inside a function compiled for it.
#[inline]
pub fn run<E: LaneElem, K: Kernel<E>>(k: K) -> K::Output {
    #[allow(unused_imports)]
    use fearless_simd::Simd;
    let Some(level) = level() else {
        return k.run::<E::Lanes<Portable>>();
    };
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        use fearless_simd::x86::{Avx2, Avx512, Sse2, Sse4_2};
        if let Some(t) = level.as_avx512() {
            return t.vectorize(
                #[inline(always)]
                || k.run::<E::Lanes<Proof<Avx512>>>(),
            );
        }
        if let Some(t) = level.as_avx2() {
            return t.vectorize(
                #[inline(always)]
                || k.run::<E::Lanes<Proof<Avx2>>>(),
            );
        }
        if let Some(t) = level.as_sse4_2() {
            return t.vectorize(
                #[inline(always)]
                || k.run::<E::Lanes<Proof<Sse4_2>>>(),
            );
        }
        if let Some(t) = level.as_sse2() {
            return t.vectorize(
                #[inline(always)]
                || k.run::<E::Lanes<Proof<Sse2>>>(),
            );
        }
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = level.as_neon() {
        return t.vectorize(
            #[inline(always)]
            || k.run::<E::Lanes<Proof<fearless_simd::aarch64::Neon>>>(),
        );
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = level.as_wasm_simd128() {
        return t.vectorize(
            #[inline(always)]
            || k.run::<E::Lanes<Proof<fearless_simd::wasm32::WasmSimd128>>>(),
        );
    }
    let _ = level;
    k.run::<E::Lanes<Portable>>()
}
