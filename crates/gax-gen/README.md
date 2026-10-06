# gax-gen

The code generator of [gax](https://crates.io/crates/gax), geometric algebra with extensors:
exact blade products for any metric, operation tables, symbolic simplification (common
subexpressions, Gröbner-basis verification) and the emission of Rust and WGSL.

Applications use it in two ways, both through `gax`:

* `gax::trace` (feature `trace`) traces a user's kernel from `build.rs` into one fused,
  verified straight-line program, in Rust and WGSL;
* the `algebra!` macro (crate `gax-macros`) runs it at compile time to declare an algebra of
  any signature.

Licensed under either of Apache License 2.0 or MIT, at your option.
