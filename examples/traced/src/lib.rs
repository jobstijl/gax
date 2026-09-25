//! Build-time traced kernels.
//!
//! `kernels` holds plain generic functions; `build.rs` traces them with symbolic
//! coefficients and emits the fused `*_fused` functions included below.

pub mod kernels;

include!(concat!(env!("OUT_DIR"), "/fused.rs"));
