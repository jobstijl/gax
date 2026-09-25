#![doc = include_str!("../README.md")]
#![no_std]
#![cfg_attr(docsrs, feature(doc_cfg))]

pub use gax_core::*;

mod extras;

/// Declare a geometric algebra of any signature (see the crate documentation).
#[cfg(feature = "macros")]
pub use gax_macros::algebra;

/// Build-time tracing: run generic kernels on symbolic coefficients and emit fused code.
#[cfg(feature = "trace")]
pub use gax_gen::trace;

/// Plane-based projective geometric algebra of the Euclidean plane.
#[cfg(feature = "pga2d")]
#[allow(
    missing_docs,
    unused_variables,
    clippy::all,
    clippy::pedantic,
    unused_parens
)]
#[path = "algebras/pga2d.rs"]
pub mod pga2d;

/// Plane-based projective geometric algebra of Euclidean space.
#[cfg(feature = "pga3d")]
#[allow(
    missing_docs,
    unused_variables,
    clippy::all,
    clippy::pedantic,
    unused_parens
)]
#[path = "algebras/pga3d.rs"]
pub mod pga3d;

/// Vector geometric algebra of the Euclidean plane.
#[cfg(feature = "vga2d")]
#[allow(
    missing_docs,
    unused_variables,
    clippy::all,
    clippy::pedantic,
    unused_parens
)]
#[path = "algebras/vga2d.rs"]
pub mod vga2d;

/// Vector geometric algebra of Euclidean space.
#[cfg(feature = "vga3d")]
#[allow(
    missing_docs,
    unused_variables,
    clippy::all,
    clippy::pedantic,
    unused_parens
)]
#[path = "algebras/vga3d.rs"]
pub mod vga3d;

/// Spacetime algebra.
#[cfg(feature = "sta")]
#[allow(
    missing_docs,
    unused_variables,
    clippy::all,
    clippy::pedantic,
    unused_parens
)]
#[path = "algebras/sta.rs"]
pub mod sta;

/// Conformal geometric algebra of Euclidean space (null basis).
#[cfg(feature = "cga3d")]
#[allow(
    missing_docs,
    unused_variables,
    clippy::all,
    clippy::pedantic,
    unused_parens
)]
#[path = "algebras/cga3d.rs"]
pub mod cga3d;

/// The code blocks of `docs/guide.md` and the README, compiled and run as doctests.
#[cfg(all(doctest, feature = "pga3d"))]
#[doc = include_str!("../../../docs/guide.md")]
pub struct GuideDoctests;
