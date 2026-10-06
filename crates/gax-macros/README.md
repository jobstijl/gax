# gax-macros

The `algebra!` macro of [gax](https://crates.io/crates/gax), geometric algebra with extensors:
declare a geometric algebra of any signature `Cl(p, q, r)`, degenerate and non-diagonal metrics
included, and get the same typed API as gax's standard algebras. Use it as `gax::algebra!`.

The macro runs gax's generator (crate `gax-gen`) at compile time. For algebras of dimension 5
and up, optimize build scripts and proc macros in the using crate's `Cargo.toml`:

```toml
[profile.dev.build-override]
opt-level = 3
```

Licensed under either of Apache License 2.0 or MIT, at your option.
