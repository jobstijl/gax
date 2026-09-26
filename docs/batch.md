# Batch kernels (`batch` feature)

With the `batch` feature, `gax::batch` runs gax's kernels over slices: 8 `f32` (or 4 `f64`)
values per instruction, using the best SIMD level the CPU has. One binary picks it at run
time, so a default build with no `-C target-cpu` flag still uses AVX2 and FMA on a machine
that has them. Level detection and per-level compilation come from
[`fearless_simd`](https://docs.rs/fearless_simd) 1.0. The feature is independent of `wide`,
and it enables `std`.

```toml
[dependencies]
gax = { version = "…", features = ["batch"] }
```

## Sandwiches

`BatchTransform` gives every versor and `Unit` versor four batch forms of `v >> x`. They
work for every pair of kinds that the algebra's `>>` supports:

| method | input layout | versors |
|---|---|---|
| `v.transform_slice(&xs, &mut out)` | array of structs (`&[Point]`) | one, prepared once |
| `v.transform_soa(&xs, &mut out)` | struct of arrays (`Soa<Point>`) | one, prepared once |
| `V::transform_each(&vs, &xs, &mut out)` | array of structs | one per element |
| `V::transform_each_soa(&vs, &xs, &mut out)` | struct of arrays | one per element |

```rust
use gax::batch::{BatchTransform, Soa};
use gax::pga3d::{Motor, Point};
use gax::Unit;

let m = Motor::<(), f32>::rotation_about(0.0, 0.0, 1.0, 0.5).normalized();
let points: Vec<Point> = (0..100).map(|i| Point::xyz(i as f32, 1.0, 0.0)).collect();

// Array of structs: the motor's action is prepared once, then applied 8 points at a time.
let mut moved = vec![Point::zero(); points.len()];
m.transform_slice(&points, &mut moved);
let d = moved[7].to_euclidean()[0] - (m >> points[7]).to_euclidean()[0];
assert!(d.abs() < 1e-5);

// Struct of arrays: no transposes, so this is the fastest layout.
let soa: Soa<Point> = points.iter().copied().collect();
let mut out: Soa<Point> = Soa::new();
m.transform_soa(&soa, &mut out);
assert_eq!(out.len(), 100);
assert!((out.get(7).to_euclidean()[0] - moved[7].to_euclidean()[0]).abs() < 1e-6);

// One motor per point.
let motors: Vec<Unit<Motor>> = (0..100)
    .map(|i| Motor::rotation_about(0.0, 0.0, 1.0, 0.01 * i as f32).normalized())
    .collect();
Unit::transform_each(&motors, &points, &mut moved);
```

`Soa<K>` stores values in blocks of 16: for each block, coefficient 0 of its 16 values comes
first, then coefficient 1, and so on. Inside a block every load is a single vector load at an
offset the compiler knows. Each generated algebra has an alias per kind, such as
`pga3d::PointSoa`.

## Your own kernels

To run any function that is generic over `T: Real` on batches, implement `Map` (one input
kind, one output kind):

```rust
use gax::batch::{self, Map};
use gax::pga3d::{Line, Motor};
use gax::Real;

struct Exp;
impl Map for Exp {
    type X = Line;
    type Y = Motor;
    #[inline(always)]
    fn call<T: Real>(&self, b: Line<(), T>) -> Motor<(), T> {
        b.exp().into_inner()
    }
}

let twists: Vec<Line> = (0..50).map(|i| Line::new(0.0, 0.0, 0.1 * i as f32, 1.0, 0.0, 0.0)).collect();
let mut motors = vec![Motor::zero(); twists.len()];
batch::map(&Exp, &twists, &mut motors);
```

For any other shape, implement `Kernel` and call `batch::run`. The `chunks`, `gather`,
`scatter`, `column` and `to_array` helpers do the array-of-structs plumbing. Mark kernel
methods `#[inline(always)]`: a function body is compiled for the detected level only when it
is inlined into the dispatcher.

Kernels traced at build time get batch forms too. Call `Tracer::batch(true)` in `build.rs`,
and every traced `name` gets a `name_batch(&a0, &a1, ..., &mut out)`. An argument slice of
length 1 is broadcast to every element (see `examples/traced`).

## Levels

`batch::run` dispatches to the best level the CPU supports, from lowest to highest:

| target | levels |
|---|---|
| x86 / x86-64 | SSE2, SSE4.2, AVX2 (with FMA), AVX-512 |
| aarch64 | NEON |
| wasm32 | SIMD128 (when compiled with `+simd128`) |
| everything | portable code (lane arrays, no detected level) |

`batch::levels()` lists every level this CPU can run. `batch::with_level(level, f)` forces
one level on the current thread, which is what the tests and benchmarks use:

```rust
use gax::batch;
for level in batch::levels() {
    batch::with_level(level, || {
        assert_eq!(batch::level_name(batch::level()), batch::level_name(level));
    });
}
```

The lane types of the detected levels are `fearless_simd` vectors. Their constructors need
the level's token, and `Coef::zero()` takes no arguments. They are sound anyway because the
level is part of the lane type through a type the crate does not export, and only the
dispatcher instantiates it, after detection succeeds. The argument is spelled out in the
source (`gax-core/src/batch/simd_lanes.rs`). The portable lanes are plain arrays.

## Elementary functions

`f32` lanes evaluate `sin`, `cos`, `sinh`, `cosh`, `atan2` and `ln` with vectorized
polynomials (Cephes' single-precision fits with Cody–Waite reduction). `batch::math` has the
same functions for one `f32`, and the tests check that the vector and scalar versions agree
bit for bit on every level. Against the standard library they are within 2 ulp (3 for `sinh`
and `cosh`), with `sin` and `cos` accurate for `|x| ≤ 8192`. `f64` lanes call the scalar
functions lane by lane.

Lanes use fused multiply-add where the level has it (AVX2, AVX-512, NEON). Scalar `f32` code
uses it only when compiled with FMA enabled, so batch results can differ from scalar results
in the last bits.

## Performance

The numbers are in [performance.md](performance.md#batch-kernels). In short, with 1024
elements on AVX2:

* SoA is 3 to 4 times faster than glam for one motor on many points, and 4 to 6 times faster
  than scalar gax.
* The AoS forms pay for transposes but still beat scalar loops.
* `exp` of twists (vectorized `sin` and `cos`) runs 8 times faster than scalar code.
