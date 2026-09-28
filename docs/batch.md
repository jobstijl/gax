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

## Maps

A map built once, such as a motor composed with a projection or a camera, applies to many
values with `BatchOf`. It is the batch form of `m.of(x)`, for any extensor with one open slot:

| method | input layout |
|---|---|
| `map.of_slice(&xs, &mut out)` | array of structs |
| `map.of_soa(&xs, &mut out)` | struct of arrays |

```rust
use gax::batch::BatchOf;
use gax::pga3d::{Motor, Plane, Point};

let cam = Motor::<(), f32>::translation(0.0, 1.0, -5.0);
let screen = Plane::<(), f32>::from_normal([0.0, 0.0, 1.0], 1.0);
let eye = Point::<(), f32>::xyz(0.0, 0.0, 0.0);
// World to camera, then the central projection onto the screen: one 4x4.
let view: Point<(Point,), f32> = (eye & (cam << Point::slot())) ^ screen;
let points: Vec<Point> = (0..100).map(|i| Point::xyz(i as f32 * 0.1, 0.0, 10.0)).collect();
let mut out = vec![Point::zero(); points.len()];
view.of_slice(&points, &mut out);
assert!((out[3].to_euclidean()[2] - 1.0).abs() < 1e-5);
```

It is as fast as the prepared action of a single motor ([performance.md](performance.md)).

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

`batch::map_soa` does the same on `Soa` storage, without transposes. For any other shape,
implement `Kernel` and call `batch::run`. The `chunks`, `gather`,
`scatter`, `column` and `to_array` helpers do the array-of-structs plumbing. Mark kernel
methods `#[inline(always)]`: a function body is compiled for the detected level only when it
is inlined into the dispatcher.

Kernels traced at build time get batch forms too. Call `Tracer::batch(true)` in `build.rs`,
and every traced `name` gets:

* `name_batch(&a0, &a1, ..., &mut out)` on slices;
* `name_batch_soa`, when the result is a value of one kind. It takes `Soa` storage for the
  arguments of one kind and slices for the others (scalars, arrays), and writes a `Soa`.

In both forms an argument of length 1 is broadcast to every element. `examples/traced` shows
them.

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

The numbers are in [performance.md](performance.md) (section "Batch kernels"). With 1024 elements
in a default build (no `target-cpu`), where the dispatcher picks AVX2:

* One motor on many points in SoA form takes 0.27 µs, about 3x faster than glam's
  `Affine3A` loop and 4x faster than scalar gax.
* `exp` of twists in SoA form is 19x faster than scalar code, thanks to the vectorized `sin` and
  `cos`.
* The array-of-structs forms pay for transposes: they beat scalar loops but not glam's
  single-point transform.
