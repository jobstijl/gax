# gax: geometric algebra with extensors, in Rust

`gax` is a geometric algebra library whose types carry *open slots*. A `Point` is a value, a
`Point<(Point,)>` a linear map, and a `Scalar<(Twist, Twist)>` a bilinear form. The products of the
algebra work on all of them, so a plain generic function evaluates values and builds maps alike.
Its design follows [numga](https://github.com/EelcoHoogendoorn/numga)'s extensors, brought into
Rust's type system. It generates code from exact tables and symbolic simplification, aiming at
the speed of hand-written graphics math.

```rust
use gax::pga3d::{Plane, Point};
use gax::Slots;

fn shadow<S: Slots>(light: Point, ground: Plane, p: Point<S>) -> Point<S> {
    (light & p) ^ ground
}

let light = Point::xyz(0.0, 0.0, 10.0);
let ground = Plane::from_normal([0.0, 0.0, 1.0], 0.0);
let s: Point = shadow(light, ground, Point::xyz(1.0, 2.0, 3.0)); // a value
let projection: Point<(Point,)> = shadow(light, ground, Point::slot()); // the map
```

## Workspace

| crate | what |
|---|---|
| [`crates/gax`](crates/gax) | the library: standard algebras (PGA2D/3D, VGA2D/3D, STA, CGA3D) behind features, `algebra!`, tracing |
| [`crates/gax-core`](crates/gax-core) | slot lists, kinds, coefficient traits, binding, solvers, Study-number functions (`no_std`) |
| [`crates/gax-gen`](crates/gax-gen) | generator: exact tables, symbolic polynomials, simplifier, emitter, `gax-regen` |
| [`crates/gax-macros`](crates/gax-macros) | the `algebra!` proc macro |
| [`crates/gax-bench`](crates/gax-bench) | benchmarks against glam, ultraviolet, nalgebra (not published) |
| [`examples/traced`](examples/traced) | build-time traced kernels, end to end |
| [`fuzz`](fuzz) | fuzzing of the algebra declaration parser |

## Documentation

* [Guide](docs/guide.md): extensors as a composition language, in plain terms.
* [Design record](docs/design.md): architecture decisions, and the verdict on each design
  hypothesis.
* [Research log](docs/research.md): prior art, and what was taken from it.
* [Performance](docs/performance.md): benchmarks, assembly findings, compile times.
* [TODO](TODO.md): status and open work.

## Development

```sh
cargo test --workspace                                         # tests (default algebras)
cargo test -p gax --features all-algebras,wide                 # every algebra, SIMD lanes
cargo run --release -p gax-gen --bin gax-regen                 # regenerate the standard algebras
cargo run --release -p gax-gen --bin gax-regen -- --check      # CI: are they up to date?
RUSTFLAGS="-C target-cpu=native" cargo bench -p gax-bench      # benchmarks
```

The minimum supported Rust version is 1.89. The crates are not yet published on crates.io.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT)
at your option. Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in this crate by you, as defined in the Apache-2.0 license, shall be dual licensed as
above, without any additional terms or conditions.
