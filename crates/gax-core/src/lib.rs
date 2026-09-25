//! Core traits of `gax`: slot lists, kinds, coefficients, binding, and the math core.
//!
//! Most users use these through the `gax` crate, which re-exports them together with the
//! generated algebras.
#![no_std]

pub mod bind;
pub mod coef;
pub mod kind;
pub mod ops;
#[cfg(feature = "wide")]
pub mod simd;
pub mod slots;
pub mod trace;
pub mod unit;

pub use bind::Of;
pub use coef::{Coef, Elem, Real};
pub use kind::{Coeffs, Extensor, Kind, Retype};
pub use ops::{
    Anticommutator, Commutator, Conjugate, Dot, Dual, Gp, Involute, Lc, Rc, Reverse, ScalarProduct,
    Transform, TransformInv, Undual, Vee, Wedge,
};
pub use slots::{Cat, HasCat, SlotArr, Slots, SplitFirst};
pub use trace::Traceable;
pub use unit::Unit;
