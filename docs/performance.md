# Performance

All measurements are on an AMD Ryzen 7 5800X with rustc 1.98.1. They use `--release`, and
`-C target-cpu=native` (AVX2 and FMA) where stated. Assembly counts come from the
`#[inline(never)]` probes in `crates/gax-bench/src/lib.rs`
(`cargo rustc -p gax-bench --release --lib -- --emit asm`).

Reproduce the timings with:

```sh
RUSTFLAGS="-C target-cpu=native" cargo bench -p gax-bench --bench transform
```

## Common PGA3D operations against glam (f32, `-C target-cpu=native`)

Run with `RUSTFLAGS="-C target-cpu=native" cargo bench -p gax-bench --bench transform`.

| operation | gax | glam 0.30 |
|---|---|---|
| transform one point: `Unit<Motor> >> Point` (fused) | 6.3 ns | `Affine3A::transform_point3a`: 2.2 ns; `Quat * Vec3A` (rotation only): 3.2 ns |
| transform one point: prepared sparse map `m.prepare::<Point>() >> p` | 4.2 ns | |
| transform one point: dense map `Point<(Point,)>::of` | 3.9 ns | |
| transform 1024 points, AoS: direct / prepared / dense map | 1.87 / 1.39 / 1.48 µs | `Affine3A` loop: 1.18 µs |
| transform 1024 points, SoA `f32x8`: direct / **prepared** | 0.66 / **0.43 µs** | (2.7x faster than glam) |
| compose motors `Unit<Motor> * Unit<Motor>` | 7.6 ns | `Affine3A * Affine3A`: 5.3 ns; `Quat * Quat`: 2.0 ns |
| invert a unit motor (the reverse) | **1.9 ns** | `Affine3A::inverse`: 8.6–13.7 ns |
| normalize a motor (Study-number `rsqrt`) | 5.8 ns | `Quat::normalize`: 2.4 ns |
| motor exponential `Line::exp` | 24 ns | `Quat::from_axis_angle`: 3.2–5.6 ns |
| motor logarithm `Unit<Motor>::log` | 24 ns | — |
| build the point map: `m.prepare::<Point>().to_map()` / `m >> Point::slot()` | 7.1 / 11.9 ns | `Affine3A::from_rotation_translation`: 2.9–4.1 ns |

Ranges show run-to-run variation, measured in separate runs of the suite.

### Where gax is faster

* **Batches in struct-of-arrays form.** A `Point<(), f32x8>` holds eight points, and every generated
  kernel runs on it unchanged. The prepared action of a unit motor needs 12 multiplications per
  eight points, with no structural zeros. That transforms 1024 points 2.7x faster than a glam
  `Affine3A` loop (hypotheses 6 and 9).
* **Inverting a unit motor** costs only negations: the certificate in `Unit` makes the inverse the
  reverse.

### Where gax is slower, and why

* **Single operations in array-of-structs form** take 1.4–3x as long as glam:
  * gax's kernels are generic over the coefficient type, so they are scalar code, and LLVM only partly
    recovers SIMD from them;
  * glam writes the same operations by hand with SSE shuffles on 4-lane registers, and stores
    matrices column-major for broadcasts.
* **A motor holds a rotation and a translation in 8 numbers.** Composing, normalizing and
  exponentiating it do more work than the quaternion alone, which is what glam's `Quat` timings
  measure. `Affine3A` is the fair comparison for rigid motions: gax is at 1.4x for composition and
  faster for inversion.
* **exp and log** go through closed forms with a series near zero (ADR-019). They take 24 ns, against
  glam's 3–6 ns for building a quaternion from an axis and angle. A motor's exponential also produces
  its translation part and needs the square root of the bivector's norm.

**Changes that closed part of the gap:**

* balanced summation trees (floating-point sums cannot be reassociated by the compiler), which took
  motor composition from 12.6 ns to 7.6 ns;
* real-trigonometric fast paths when the generator proves the square of a bivector is non-positive
  (a rotation), which took `exp` from 174 ns to 24 ns, `log` from 61 ns to 24 ns and `normalized`
  from 20 ns to 5.8 ns;
* prepared sparse maps.

**Still open:** emitting `mul_add` when FMA is available, and SIMD-friendly layouts for single
values.

## Against nalgebra, ultraviolet and `geometric_algebra` (`benches/compare.rs`)

| operation | gax | others |
|---|---|---|
| transform a point by a rigid motion (f32) | **6.3 ns** | `geometric_algebra` 0.3 `Motor::transformation`: 10.9 ns; nalgebra `Isometry3 * Point3`: 13.5 ns |
| eight points at once (SoA `f32x8`) | 8.4 ns (rotation and translation) | ultraviolet `Rotor3x8 * Vec3x8` (rotation only): 8.0 ns |
| compose a chain of 5 rigid motions | 43 ns | glam `Affine3A`: 13 ns; nalgebra `Isometry3`: 26 ns |
| 6x6 map inverse (f64) | 465 ns | nalgebra `Matrix6::try_inverse`: 301 ns |
| 6x6 solve | 217 ns | nalgebra LU solve: 190 ns |
| 6x6 generalized eigenproblem (vibration modes) | 5.6 µs | nalgebra Cholesky + `SymmetricEigen`: 2.1 µs |
| 4x4 SVD | **1.11 µs** | nalgebra `Matrix4::svd`: 1.35 µs |
| CGA3D `Unit<Motor> >> point` (f32) | 6.96 ns | — |
| CGA3D general even versor `>> point` | 38.7 ns | — |
| CGA3D `Twist::exp` | 15.2 ns | — |
| CSTA (6D) vector product | 13.2 ns | — |

**The solvers are branch free** (ADR-017): fixed Jacobi sweeps and select-based pivoting, so they
run unchanged on SIMD lanes. A scalar, branching implementation such as nalgebra's is faster for
one matrix at a time: 1.5x for the inverse and 2.7x for the generalized eigenproblem. Early exit for
scalar coefficients is planned (TODO). The SVD is already faster than nalgebra's.

**A motor chain costs more than an `Affine3A` chain.** A motor product is 48 scalar mul and 40 add,
while glam's affine product is a SIMD 3x3 product plus a translation.

## Build-time tracing: arithmetic against wall time (`benches/compare.rs`, rigid-body step)

An explicit Euler step of a free rigid body in PGA3D (motor, twist, principal inertia,
renormalization), with the body's constants known at build time:

| | arithmetic | one body (f32) | eight bodies (`f32x8` lanes) |
|---|---|---|---|
| generic code | 113 mul, 70 add, 5 div | **20.7 ns** | 43.6 ns |
| fused at build time | 98 mul, 70 add, 1 div | 28.0 ns | **32.0 ns** |

* **Batches.** In SoA lanes the arithmetic count is what runs, and the fused kernel is 26% faster.
* **Single values.** Here LLVM's SLP vectorizer turns the *more regular* generic code into better SIMD
  than the leaner but irregular fused program: 169 against 191 instructions, with more shuffles in the
  fused one. The fused kernel is slower.
* **What it means:**
  * tracing pays off for batches and for scalar-only targets, and whenever constants remove work;
  * for single values on a SIMD target, measure first.

  Emitting fused code in a shape the SLP vectorizer handles well is open work (TODO).

## Fused sandwich kernels (op counts from the generator)

| kernel | gax | reference |
|---|---|---|
| `Unit<Rotor> >> Point` (PGA3D) | 18 mul, 12 add | the quaternion formula: 18 mul, 12 add |
| `Unit<Motor> >> Point` (PGA3D, general weight) | 25 mul, 18 add | GAmphetamine, weight fixed at 1: 21 mul, 18 add |
| `Motor >> Point` (not unit) | 38 mul, 32 add | GAmphetamine, weight fixed at 1: 28 mul, 21 add |
| `Unit<Motor> >> Plane` | 28 mul, 18 add | — |
| `Unit<Motor> >> Line` | 58 mul, 45 add | open: the line kernel is not yet as good as the point kernel |
| `Unit<Motor> >> Point` (PGA2D) | 15 mul, 7 add | hand-derived: 16 mul, 7 add |

Run `cargo run -p gax-gen --bin gax-regen -- --verbose` to list every kernel's cost.

## Build-time traced kernels

Examples from `examples/traced`, as reported by the tracer. Each fused kernel is checked against the
generic code's cost at run time, with every operation performed and no constant folded:

| kernel | fused | generic code at run time |
|---|---|---|
| move a point, join with a light, meet a given plane | 49 mul, 32 add | 49 mul, 32 add |
| the same onto the floor `z = 0` (a constant) | **31 mul, 21 add** | 46 mul, 32 add |
| compose two unit motors, then apply | 73 mul, 58 add | 73 mul, 58 add |
| normalize a point's weight | 3 mul, 1 div | 3 mul, 1 div |

## IEEE semantics (hypothesis 7)

These results come from the assembly, at x86-64-v3:

* `x + 0.0` compiles to `vaddss` with a zeroed register;
* `x + (-0.0)` compiles to nothing;
* `0.0 * x` compiles to `vmulss`.

Structural zeros therefore never reach runtime arithmetic in gax:

* products are emitted term by term, each accumulator starting from its first term;
* fused kernels drop zero matrix entries;
* traced constants fold.

The one exception is a product whose result is rounded up to a larger kind (ADR-021): the extra
coefficients are zeros.

## Constant-table loops are not unrolled (hypothesis 5)

* A loop over a constant table of sparse terms unrolls fully at 16 terms (32 FP ops, no branches).
* At 20 terms or more it stays a runtime loop with five jumps per call.

Most GA products have more terms than that, so the generator emits straight-line code instead
(ADR-006).

## Compile time and code size

These are debug builds of the library alone; generic code is compiled only when used.

| algebra | generated lines | product impls | sandwich kernels | debug build |
|---|---|---|---|---|
| VGA2D | 6k | 190 | 20 | 0.7 s |
| VGA3D | 13k | 363 | 42 | 1.5 s |
| PGA2D | 20k | 602 | 108 | 2.2 s |
| STA | 24k | 470 | 48 | 3.7 s |
| PGA3D | 35k | 733 | 140 | 4.8 s |
| CGA3D | 73k | 739 | 80 | 14.4 s |

A release build with all six algebras takes 27 s. Regenerating all standard algebras takes 47 s in
release, most of it spent simplifying CGA3D's 16-component versors.

* **Why generated code costs nothing unused.** Every generated function is either generic (over slots
  and coefficients) or `#[inline]`, so machine code exists only for what a program uses.
* **What does cost.** The cost is in parsing and type-checking the generated impls. That is why the
  algebras are separate cargo features.
