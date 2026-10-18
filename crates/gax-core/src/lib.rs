//! Core traits of `gax`: slot lists, kinds, coefficients, binding, and the math core.
//!
//! Most users use these through the `gax` crate, which re-exports them together with the
//! generated algebras. The pieces:
//!
//! * [`Slots`] (tuples of kinds) and [`Kind`]: the type-level structure of values, maps and forms.
//! * [`Extensor`]: the interface every generated type implements; [`Of`] for binding and
//!   composition; [`Cast`], [`SubKind`] and [`GradePart`] for projections, embeddings and grade
//!   parts between kinds.
//! * [`Coef`] and [`Real`]: coefficient types (`f32`, `f64`, SIMD lanes, the symbolic `Sym`,
//!   [`fp::Fp`], a prime field for exact randomized identity checks, [`jet::Jet`], values with
//!   their derivatives for forward-mode differentiation (with respect to a kind, or
//!   [`jet::Dual`] for loose parameters), and [`strict::Strict`], coefficients that give the
//!   same bits on every target and SIMD level).
//! * [`SquareMap`], [`Endomorphism`], [`Form`], [`FormAsMap`], [`Pairing`], [`OfBoth`],
//!   [`TraceFirst`], [`SingularValues`], [`PseudoInverse`], [`LeastSquares`],
//!   [`LeastSquaresDyad`] and [`OfDyad`]: solving, eigenproblems (real, and [`Complex`] for
//!   general maps), singular values, pairings and least squares on maps and forms.
//! * [`Unit`] and [`Prepared`]: certified unit versors and prepared versor actions.
//!
//! ```
//! use gax::pga3d::{Motor, Point};
//! use gax::Slots;
//!
//! // Written once, generic over the open slots.
//! fn lift<S: Slots>(p: Point<S, f64>) -> Point<S, f64> {
//!     Motor::translation(0.0, 0.0, 1.0) >> p
//! }
//! let value = lift(Point::xyz(1.0, 2.0, 3.0));
//! let map: Point<(Point,), f64> = lift(Point::slot());
//! assert_eq!(map.of(Point::xyz(1.0, 2.0, 3.0)).to_euclidean(), value.to_euclidean());
//! ```
#![no_std]

#[cfg(feature = "batch")]
pub mod batch;
pub mod bind;
pub mod calculus;
pub mod cast;
pub mod coef;
pub mod complex;
#[cfg(feature = "std")]
pub mod dec;
pub mod extensor;
#[cfg(feature = "std")]
pub mod field;
mod fill;
pub mod fp;
pub mod gpu;
pub mod jet;
pub mod kind;
mod linalg;
pub mod map;
pub mod math;
#[cfg(feature = "std")]
pub mod mesh;
pub mod ops;
pub mod orientation;
mod permute;
pub mod prepared;
#[cfg(feature = "wide")]
pub mod simd;
pub mod simplex;
pub mod sites;
pub mod slots;
pub mod strict;
mod study;
pub mod trace;
pub mod unit;

pub use bind::Of;
pub use calculus::Reciprocal;
pub use cast::{Cast, GradePart, SubKind};
pub use coef::{Coef, Elem, Real};
pub use complex::Complex;
pub use extensor::{
    Endomorphism, Form, FormAsMap, LeastSquares, LeastSquaresDyad, OfBoth, OfDyad, Pairing,
    PseudoInverse, SingularValues, SquareMap, Svd, TraceFirst,
};
pub use gpu::GpuMat;
pub use kind::{ApproxEq, Coeffs, Extensor, FlatKind, Kind, Retype, Slot, select_lt};
pub use map::{LinearMap, SquareLinearMap};
pub use ops::{
    Anticommutator, Commutator, Conjugate, DivBy, Dot, Dual, Exp, Gp, Inverse, Involute, Lc, Log,
    Norm, Normalize, Outermorphism, Rc, Reverse, ScalarProduct, Sqrt, Transform, TransformInv,
    Undual, Vee, Wedge,
};
pub use orientation::{Extrinsic, Intrinsic, Orientation, Oriented};
pub use prepared::{Prepare, Prepared};
pub use slots::{Cat, Slots, SplitFirst};
pub use unit::{Unit, Widen};

/// Items that only generated code names: type-level machinery, the dense kernels and the
/// functions of Study numbers. Not part of the API; they may change in any release.
#[doc(hidden)]
pub mod __private {
    pub use crate::fill::{Bit, False, FillList, FillStep, KindEq, SplitLast, True};
    pub use crate::orientation::{Reorient, reflect};
    pub use crate::permute::MoveToFront;
    pub use crate::prepared::PreparedDense;
    pub use crate::slots::{HasCat, SlotArr, by_entries};
    pub use crate::unit::{LogNeedsUnit, NewtonStep};

    /// The dense kernels: LU, Jacobi eigen and singular value decompositions, QR iteration.
    pub mod linalg {
        pub use crate::linalg::*;
    }

    /// Functions of Study numbers, behind exp, log, normalization and inverses.
    pub mod study {
        pub use crate::study::*;
    }
}

#[cfg(feature = "bytemuck")]
pub use bytemuck;
