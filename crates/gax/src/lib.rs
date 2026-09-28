#![doc = include_str!("../README.md")]
#![no_std]
#![cfg_attr(docsrs, feature(doc_cfg))]

pub use gax_core::*;

mod extras;

/// The homomorphisms between the standard algebras, as `From` impls (generated from
/// `gax_gen::emit_homs::HOMS`; see the guide, "Between algebras").
#[allow(unused_imports, unused_parens, clippy::all, clippy::pedantic)]
#[rustfmt::skip]
mod homs {
    include!("algebras/homs.rs");
}

#[cfg(feature = "std")]
extern crate std;

#[cfg(feature = "wgsl")]
pub mod wgsl;

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
    pub use crate::extras::Between;
}

/// Plane-based projective geometric algebra of Euclidean space, `R(3,0,1)`.
///
/// The types, products and methods are generated from `specs/pga3d.gax` by `gax-regen`.
#[cfg(feature = "pga3d")]
#[allow(missing_docs, unused_variables, unused_parens, clippy::all, clippy::pedantic)]
#[rustfmt::skip]
pub mod pga3d {
    include!("algebras/pga3d.rs");
    pub use crate::extras::{Between, PrincipalInertia};
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

/// Conformal geometric algebra of the Euclidean plane, `R(3,1,0)`, in the null basis `eo`, `ei`:
/// points, circles and lines of the plane, and their conformal transformations.
///
/// The types, products and methods are generated from `specs/cga2d.gax` by `gax-regen`.
#[cfg(feature = "cga2d")]
#[allow(missing_docs, unused_variables, unused_parens, clippy::all, clippy::pedantic)]
#[rustfmt::skip]
pub mod cga2d {
    include!("algebras/cga2d.rs");
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

/// Projective spacetime algebra `R(3,1,1)`: space `e1, e2, e3`, time `e4` (square `-1`) and
/// the degenerate `e0`. Vectors are hyperplanes, quadvectors are events, and `Motor` is a
/// Poincaré motion (feature `stap`).
///
/// The types, products and methods are generated from `specs/stap.gax` by `gax-regen`.
#[cfg(feature = "stap")]
#[allow(missing_docs, unused_variables, unused_parens, clippy::all, clippy::pedantic)]
#[rustfmt::skip]
pub mod stap {
    include!("algebras/stap.rs");
}

/// Conformal spacetime algebra `R(4,2)`: space `e1, e2, e3`, time `e4` (square `-1`) and the
/// null basis `eo`, `ei` (feature `csta`).
///
/// The types, products and methods are generated from `specs/csta.gax` by `gax-regen`.
#[cfg(feature = "csta")]
#[allow(missing_docs, unused_variables, unused_parens, clippy::all, clippy::pedantic)]
#[rustfmt::skip]
pub mod csta {
    include!("algebras/csta.rs");
}

/// The code blocks of `docs/guide.md` and the README, compiled and run as doctests.
#[cfg(all(doctest, feature = "pga3d"))]
#[doc = include_str!("../../../docs/guide.md")]
pub struct GuideDoctests;

/// The code blocks of `docs/batch.md`, compiled and run as doctests.
#[cfg(all(doctest, feature = "batch", feature = "pga3d"))]
#[doc = include_str!("../../../docs/batch.md")]
pub struct BatchDoctests;
