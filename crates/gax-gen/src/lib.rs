//! Generator library for `gax`.
//!
//! Turns an algebra description into exact operation tables, typed subspaces and Rust
//! source. It is used by the `algebra!` proc macro, by build-time tracing, and by the tool
//! that regenerates the standard algebras shipped in the `gax` crate.

pub mod algebra;
pub mod cse;
pub mod emit;
pub mod groebner;
pub mod poly;
pub mod slp;
pub mod spec;
pub mod sym;
pub mod symbolic;
pub mod table;
pub mod trace;
