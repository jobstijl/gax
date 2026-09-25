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

/// Plane-based projective geometric algebra of the Euclidean plane, `R(2,0,1)`.
///
/// The types, products and methods are generated from `specs/pga2d.gax` by `gax-regen`.
#[cfg(feature = "pga2d")]
#[allow(missing_docs, unused_variables, unused_parens, clippy::all, clippy::pedantic)]
#[rustfmt::skip]
pub mod pga2d {
    include!("algebras/pga2d.rs");
}

/// Plane-based projective geometric algebra of Euclidean space, `R(3,0,1)`.
///
/// The types, products and methods are generated from `specs/pga3d.gax` by `gax-regen`.
#[cfg(feature = "pga3d")]
#[allow(missing_docs, unused_variables, unused_parens, clippy::all, clippy::pedantic)]
#[rustfmt::skip]
pub mod pga3d {
    include!("algebras/pga3d.rs");
    pub use crate::extras::PrincipalInertia;
}

/// Vector geometric algebra of the Euclidean plane, `R(2,0,0)`.
///
/// The types, products and methods are generated from `specs/vga2d.gax` by `gax-regen`.
#[cfg(feature = "vga2d")]
#[allow(missing_docs, unused_variables, unused_parens, clippy::all, clippy::pedantic)]
#[rustfmt::skip]
pub mod vga2d {
    include!("algebras/vga2d.rs");
}

/// Vector geometric algebra of Euclidean space, `R(3,0,0)`.
///
/// The types, products and methods are generated from `specs/vga3d.gax` by `gax-regen`.
#[cfg(feature = "vga3d")]
#[allow(missing_docs, unused_variables, unused_parens, clippy::all, clippy::pedantic)]
#[rustfmt::skip]
pub mod vga3d {
    include!("algebras/vga3d.rs");
}

/// Spacetime algebra, `R(1,3,0)`.
///
/// The types, products and methods are generated from `specs/sta.gax` by `gax-regen`.
#[cfg(feature = "sta")]
#[allow(missing_docs, unused_variables, unused_parens, clippy::all, clippy::pedantic)]
#[rustfmt::skip]
pub mod sta {
    include!("algebras/sta.rs");
}

/// Conformal geometric algebra of Euclidean space, `R(4,1,0)`, in the null basis `eo`, `ei`.
///
/// The types, products and methods are generated from `specs/cga3d.gax` by `gax-regen`.
#[cfg(feature = "cga3d")]
#[allow(missing_docs, unused_variables, unused_parens, clippy::all, clippy::pedantic)]
#[rustfmt::skip]
pub mod cga3d {
    include!("algebras/cga3d.rs");
}

/// The code blocks of `docs/guide.md` and the README, compiled and run as doctests.
#[cfg(all(doctest, feature = "pga3d"))]
#[doc = include_str!("../../../docs/guide.md")]
pub struct GuideDoctests;
