//! Tests of gax's WGSL modules through the `wesl` crate (docs/shaders.md, test layers 2 and 4):
//! WESL imports across packages with stripping, the traced kernels' module, and CPU
//! evaluation of every kernel against its exact value. The tests are in `tests/`; this crate
//! has no code of its own.
