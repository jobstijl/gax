#![doc = include_str!("../README.md")]
#![no_std]
#![cfg_attr(docsrs, feature(doc_cfg))]

pub use gax_core::*;

mod extras;
#[cfg(any(feature = "pga2d", feature = "pga3d"))]
mod interop;
#[cfg(any(feature = "pga2d", feature = "pga3d"))]
mod moments;
pub mod motions;

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

/// The algebra modules, with their attributes, the code `gax-regen` generates from
/// `specs/{name}.gax` and the extras listed.
macro_rules! algebras {
    ($($(#[$attr:meta])* $name:ident { $($extra:item)* })*) => {$(
        $(#[$attr])*
        #[doc = ""]
        #[doc = concat!(
            "The types, products and methods are generated from `specs/",
            stringify!($name),
            ".gax` by `gax-regen`."
        )]
        #[allow(missing_docs, unused_variables, unused_parens, clippy::all, clippy::pedantic)]
        #[rustfmt::skip]
        pub mod $name {
            include!(concat!("algebras/", stringify!($name), ".rs"));
            $($extra)*
        }
    )*};
}

algebras! {
    /// Plane-based projective geometric algebra of the Euclidean plane, `R(2,0,1)`.
    #[cfg(feature = "pga2d")]
    pga2d {
        pub use crate::extras::Between;
        pub use crate::moments::Moments2 as Moments;
    }

    /// Plane-based projective geometric algebra of Euclidean space, `R(3,0,1)`.
    #[cfg(feature = "pga3d")]
    pga3d {
        pub use crate::extras::{Between, PrincipalInertia};
        pub use crate::moments::Moments3 as Moments;
    }

    /// Vector geometric algebra of the Euclidean plane, `R(2,0,0)`.
    #[cfg(feature = "vga2d")]
    vga2d {}

    /// Vector geometric algebra of Euclidean space, `R(3,0,0)`.
    #[cfg(feature = "vga3d")]
    vga3d {}

    /// Spacetime algebra, `R(1,3,0)`.
    #[cfg(feature = "sta")]
    sta {}

    /// Conformal geometric algebra of the Euclidean plane, `R(3,1,0)`, in the null basis `eo`,
    /// `ei`: points, circles and lines of the plane, and their conformal transformations.
    #[cfg(feature = "cga2d")]
    cga2d {}

    /// Conformal geometric algebra of Euclidean space, `R(4,1,0)`, in the null basis `eo`, `ei`.
    #[cfg(feature = "cga3d")]
    cga3d {}

    /// Projective spacetime algebra `R(3,1,1)`: space `e1, e2, e3`, time `e4` (square `-1`) and
    /// the degenerate `e0`. Vectors are hyperplanes, quadvectors are events, and `Motor` is a
    /// Poincaré motion (feature `stap`).
    #[cfg(feature = "stap")]
    stap {}

    /// Conformal spacetime algebra `R(4,2)`: space `e1, e2, e3`, time `e4` (square `-1`) and
    /// the null basis `eo`, `ei` (feature `csta`).
    #[cfg(feature = "csta")]
    csta {}
}

/// The code blocks of `docs/guide.md` and the README, compiled and run as doctests.
#[cfg(all(doctest, feature = "pga3d"))]
#[doc = include_str!("../guide.md")]
pub struct GuideDoctests;

/// The code blocks of `docs/batch.md`, compiled and run as doctests.
#[cfg(all(doctest, feature = "batch", feature = "pga3d"))]
#[doc = include_str!("../batch.md")]
pub struct BatchDoctests;
