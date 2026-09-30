//! Core traits of `gax`: slot lists, kinds, coefficients, binding, and the math core.
//!
//! Most users use these through the `gax` crate, which re-exports them together with the
//! generated algebras. The pieces:
//!
//! * [`Slots`] (tuples of kinds) and [`Kind`]: the type-level structure of values, maps and forms.
//! * [`Extensor`]: the interface every generated type implements; [`Of`], [`MoveToFront`],
//!   [`FillList`] for binding and composition.
//! * [`Coef`] and [`Real`]: coefficient types (`f32`, `f64`, SIMD lanes, the symbolic `Sym`, and
//!   [`fp::Fp`], a prime field for exact randomized identity checks).
//! * [`SquareMap`], [`Endomorphism`], [`Form`], [`Pairing`]: solving, eigenproblems and pairings on
//!   maps and forms, built on [`linalg`].
//! * [`Unit`] and [`Prepared`]: certified unit versors and prepared versor actions.
//! * [`study`]: functions of Study numbers, behind exp, log, normalization and inverses.
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
pub mod coef;
pub mod extensor;
pub mod fill;
pub mod fp;
pub mod gpu;
pub mod kind;
pub mod linalg;
pub mod math;
pub mod ops;
pub mod permute;
pub mod prepared;
#[cfg(feature = "wide")]
pub mod simd;
pub mod slots;
pub mod study;
pub mod trace;
pub mod unit;

pub use bind::Of;
pub use coef::{Coef, Elem, Real};
pub use extensor::{Endomorphism, Form, Pairing, SquareMap, TraceFirst};
pub use fill::{False, FillList, KindEq, SplitLast, True};
pub use gpu::GpuMat;
pub use kind::{ApproxEq, Coeffs, Extensor, Kind, Retype, select_lt};
pub use ops::{
    Anticommutator, Commutator, Conjugate, DivBy, Dot, Dual, Gp, Involute, Lc, Log, Outermorphism,
    Rc, Reverse, ScalarProduct, Transform, TransformInv, Undual, Vee, Wedge,
};
pub use permute::MoveToFront;
pub use prepared::{Prepare, Prepared};
pub use slots::{Cat, HasCat, SlotArr, Slots, SplitFirst};
pub use trace::Traceable;
pub use unit::{NewtonStep, Unit, Widen};

#[cfg(feature = "bytemuck")]
pub use bytemuck;
