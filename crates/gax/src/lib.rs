//! Geometric algebra with extensors.
//!
//! Every multivector type is generic over its open slots: `Point` is a point, `Point<(Point,)>`
//! a linear map on points, `Scalar<(Twist, Twist)>` a bilinear form on twists. Products work
//! on all of them alike, so the same generic function evaluates a value or builds a map.
#![no_std]
#![cfg_attr(docsrs, feature(doc_cfg))]

pub use gax_core::*;

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
