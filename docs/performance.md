# Performance

All measurements are on an AMD Ryzen 7 5800X with rustc 1.98.1. They were taken on an otherwise idle machine, and any outlier was re-run in isolation. They use `--release`, and
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
| transform one point: `Unit<Motor> >> Point` (fused, FMA) | 5.3 ns | `Affine3A::transform_point3a`: 2.2 ns; `Quat * Vec3A` (rotation only): 3.2 ns |
| transform one point: prepared sparse map `m.prepare::<Point>() >> p` | 4.2 ns | |
| transform one point: dense map `Point<(Point,)>::of` | 3.9 ns | |
| transform 1024 points, AoS: direct / prepared / dense map | 1.87 / 1.39 / 1.48 µs | `Affine3A` loop: 1.18 µs |
| transform 1024 points, SoA `f32x8`: direct / **prepared** | 0.51 / **0.43 µs** | (2.7x faster than glam) |
| compose motors `Unit<Motor> * Unit<Motor>` | 7.6 ns | `Affine3A * Affine3A`: 5.3 ns; `Quat * Quat`: 2.0 ns |
| invert a unit motor (the reverse) | **1.9 ns** | `Affine3A::inverse`: 8.6–13.7 ns |
| normalize a motor (Study-number `rsqrt`) | 5.8 ns | `Quat::normalize`: 2.4 ns |
| motor exponential `Line::exp` (rotation and translation) | 24 ns | `Quat::from_scaled_axis` (rotation only, the same work of finding the angle): 9.5 ns; `Quat::from_axis_angle` (angle given): 5.6 ns |
| motor logarithm `Unit<Motor>::log` | 24 ns | — |
| build the point map: `m.prepare::<Point>().to_map()` / `m >> Point::slot()` | 7.1 / 11.9 ns | `Affine3A::from_rotation_translation`: 2.9–4.1 ns |

These timings predate two changes: the drift-tolerant `Unit` kernels (8 more multiplications on
motor-on-point; see "Fused sandwich kernels" below), and the longer series near the boundaries of
`exp` and `log` (numerics.md). Re-measuring them is open (TODO).

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

Fused kernels emit `mul_add` for every product that feeds a single sum. It is a hardware fused
multiply-add when the target has FMA (for example with `-C target-cpu=native`), and `a*b + c`
otherwise. It took the single fused sandwich from 6.3 ns to 5.3 ns and the SoA direct batch from
0.66 µs to 0.51 µs.

**Still open:** SIMD-friendly layouts for single values.

## Against nalgebra, ultraviolet and `geometric_algebra` (`benches/compare.rs`)

| operation | gax | others |
|---|---|---|
| transform a point by a rigid motion (f32) | **6.3 ns** | `geometric_algebra` 0.3 `Motor::transformation`: 10.9 ns; nalgebra `Isometry3 * Point3`: 13.5 ns |
| eight points at once (SoA `f32x8`) | 8.4 ns (rotation and translation) | ultraviolet `Rotor3x8 * Vec3x8` (rotation only): 8.0 ns |
| compose a chain of 5 rigid motions | 43 ns | glam `Affine3A`: 13 ns; nalgebra `Isometry3`: 26 ns |
| 6x6 map inverse (f64) | **118 ns** | nalgebra `Matrix6::try_inverse`: 202 ns |
| 6x6 solve | **109 ns** | nalgebra LU solve: 149 ns |
| 6x6 generalized eigenproblem (vibration modes) | **1.18 µs** | nalgebra Cholesky + `SymmetricEigen`: 1.30 µs |
| 4x4 SVD | **0.62 µs** | nalgebra `Matrix4::svd`: 0.91 µs |
| CGA3D `Unit<Motor> >> point` (f32) | 6.96 ns | — |
| CGA3D general even versor `>> point` | 38.7 ns | — |
| CGA3D `Twist::exp` | 15.2 ns | — |
| CSTA (6D) vector product | 13.2 ns | — |

**The solvers are branch free per lane** (ADR-017), so they run unchanged on SIMD lanes. Two
lane-wide exits, `Real::all_lt`, keep them competitive for single matrices:

* **Jacobi sweeps** stop once every lane has converged. This took the 6x6 generalized eigenproblem
  from 5.6 µs to 1.5 µs and the 4x4 SVD from 1.11 µs to 0.72 µs.
* **LU skips work** it does not need: pivot swaps when no lane needs one, and the permutation gather
  when it is the identity in every lane.

* **Scalars pivot with branches.** `Real::SCALAR` tells the LU that it has a single number, which
  then swaps rows the ordinary way instead of with lane-wise selects. The inverse also shares the
  pivot reciprocals across its columns, so its only divisions are those `n` reciprocals. Together
  these took the 6x6 inverse from 458 ns (1.4x nalgebra) to 118 ns (0.6x).

(The solver rows were measured again for this change: the earlier eigh and SVD numbers were 1.5 µs
and 0.72 µs against nalgebra's 2.1 µs and 1.35 µs, on a busier machine.)

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
| `Unit<Rotor> >> Point` (PGA3D) | 26 mul, 15 add (18 mul, 12 add before) | the quaternion formula (not homogeneous): 18 mul, 12 add |
| `Unit<Motor> >> Point` (PGA3D, general weight) | 33 mul, 21 add (25 mul, 18 add before) | GAmphetamine, weight fixed at 1: 21 mul, 18 add |
| `Motor >> Point` (not unit) | 38 mul, 32 add | GAmphetamine, weight fixed at 1: 28 mul, 21 add |
| `Unit<Motor> >> Plane` | 36 mul, 21 add (28 mul, 18 add before) | — |
| `Unit<Motor> >> Line` | 58 mul, 45 add | open: the line kernel is not yet as good as the point kernel |
| `Unit<Motor> >> Point` (PGA2D) | 16 mul, 8 add (15 mul, 7 add before) | hand-derived: 16 mul, 7 add |

**The `Unit` kernels are drift-tolerant** (ADR-020, [numerics.md](numerics.md)). The simplifier
reduces a `Unit` kernel modulo `u ~u = 1`, which makes it cheaper but no longer homogeneous in `u`,
so a drifted versor (`u ~u = (1 + δ)²`) distorted shapes by about `2δ`. The kernels are now made
homogeneous again: a pass over the simplified program multiplies by `‖u‖²` where the reduction
had substituted 1 (`cse::repair_degree`), and the generator verifies the result exactly. The
"before" numbers above are the non-homogeneous kernels.

* **Cost.** Over all 314 `Unit` sandwich kernels, multiplications went from 19,895 to 21,146
  (+6%); the plain kernels total 22,703.
* **Why not free.** In the quadratic slice, the only homogeneous representatives of a `Unit`
  kernel are the plain kernel plus multiples of the Study condition. So re-expanding and
  simplifying afresh gives back the plain kernel's cost (38 mul for motor on point). Repairing
  degrees in place keeps most of the reduced structure (33 mul).

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

## Batch kernels (`benches/batch.rs`)

1024 elements, f32, Ryzen 7 5800X. The default build targets baseline x86-64, and
`batch::run` dispatches to AVX2 at run time. The native build adds `-C target-cpu=native`, which
gives `wide`, glam and scalar gax AVX2 and FMA too. "wide" is the same generic kernel on
`wide::f32x8` lanes, with the inputs packed beforehand, so it pays no transposes.

| operation | build | scalar gax | wide f32x8 (SoA) | glam | batch AoS | batch SoA |
|---|---|---|---|---|---|---|
| one motor, 1024 points | default | 1.20 µs | 0.54 µs | 0.74 µs | 0.91 µs | **0.27 µs** |
| | native | 1.09 µs | 0.26 µs | 0.69 µs | 0.95 µs | **0.26 µs** |
| 1024 motor-point pairs | default | 2.80 µs | 1.11 µs | 0.79 µs¹ | 1.98 µs | **0.54 µs** |
| | native | 2.24 µs | **0.46 µs** | 0.82 µs¹ | 1.94 µs | 0.49 µs |
| exp of 1024 twists | default | 29.8 µs | 4.03 µs | 26.3 µs² | 3.64 µs | **1.58 µs** |
| | native | 13.5 µs | 1.83 µs | 7.15 µs² | 3.75 µs | **1.63 µs** |
| rigid-body step, traced (1024 bodies) | default | 17.3 µs | 7.24 µs | — | 7.59 µs | — |
| | native | 15.0 µs | **2.63 µs** | — | 7.21 µs | — |
| rigid-body rate, traced (1024 bodies)³ | default | 4.97 µs | 2.09 µs | — | 3.20 µs | **1.13 µs** |
| | native | 5.00 µs | **0.68 µs** | — | 3.03 µs | 0.84 µs |

¹ `Affine3A::transform_point3a` per pair, with the affines built beforehand (9 multiply-adds,
against the motor sandwich's 24 multiplies and 14 adds per point).
² `Quat::from_scaled_axis`: rotations only, where the motor exponential includes translations.
³ The velocity half of the step, `Line` to `Line`, which has a struct-of-arrays form (`name_batch_soa`);
the full step returns a tuple and is batched in AoS form only.

Per level, in the default build, SoA with one motor takes 0.47 µs portable, 0.50 µs on SSE2 and
SSE4.2, and 0.27 µs on AVX2.

* **In a default build, the dispatched kernels are the fastest option.** They use AVX2 and FMA,
  which statically compiled code cannot assume.
* **With `target-cpu=native`, `wide` catches up** on the same generic kernels, and it wins where
  the batch form works on arrays of structs. The traced rigid step is batched only in AoS form,
  and five transposes per batch cost more than the kernel. The SoA form of the traced rate kernel
  comes within 25% of `wide`, and it beats `wide` by 1.85x in the default build.
* **The vectorized elementary functions are fast.** `exp` on SoA is 19x faster than scalar gax and
  2.5x faster than `wide` in the default build, and it still beats `wide` natively.
* **Arrays of structs cost transposes.** The AoS forms beat scalar loops but not glam's
  single-point affine transform, whose data layout is its SIMD layout. Keep hot data in `Soa`.

How the kernels stay vectorized (inlining, `Coef::vectorize`, the block layout) is in ADR-023 of
the [design record](design.md).

## Law-based rewrites in the tracer

The laws license rewrites (docs/laws.md §4), such as folding a versor chain
`a >> (b >> x)` into `(a * b) >> x`, or reassociating a composition. At run time Rust executes
the expression as written, so a rewrite would have to happen in the tracer. Expanded polynomials
do not depend on bracketing, so the only place a rewrite can matter is the low expansion limits
of the portfolio (ADR-010), where the traced DAG keeps the shape as written.

`crates/gax/tests/trace_rewrites.rs` traces each pair of equivalent kernels and prints the cost at
every limit. The kept kernel is the cheapest; "—" means the strategy was skipped (too many
terms) or overflowed. PGA3D, `Unit` motors unless marked plain:

| kernel | kept | limit 0 | limit 8 | unbounded |
|---|---|---|---|---|
| `a >> (b >> (c >> p))` | **99 mul, 63 add** | 99m 63a | — | 1167m 870a |
| `(a * b * c) >> p` | 129 mul, 101 add | 129m 101a | 375m 291a | 1167m 870a |
| `a >> (b >> p)`, plain | **76 mul, 64 add** | 76m 64a | 118m 72a | 390m 312a |
| `(a * b) >> p`, plain | 86 mul, 72 add | 86m 72a | 113m 89a | 390m 312a |
| `((a * b) >> Point::slot()).of(p)` | 92 mul, 72 add | 92m 72a | 116m 84a | 172m 129a |
| `f.of(g.of(h.of(p)))`, maps of motors | **132 mul, 96 add** | 132m 96a | — | 1167m 870a |
| `f.of(g).of(h).of(p)` | 186 mul, 132 add | 186m 132a | — | 1167m 870a |
| `(m * b) · s` | 42 mul, 28 add | 44m 28a | 42m 28a | 42m 28a |
| `m * (b · s)` | 42 mul, 28 add | 42m 28a | 42m 28a | 42m 28a |

**The result is negative, so no rewrite pass was built.**

* **Folding a versor chain costs more for a single object.** The product of the versors (48 mul
  per motor product) and one sandwich cost more than two sandwiches. Folding pays only when the
  product is reused across many objects, which is visible to the user (prepare it once, as the
  batch kernels do) and not to a trace of one application.
* **Composing maps before applying them** costs more for the same reason.
* **Pulling scalars through products** is found by the portfolio at higher limits anyway.
* **Trace cyclicity and outermorphism fusion** were not tried: no benchmark motivates them, and
  `emit_outer` already derives the minors symbolically.

These numbers are for the drift-tolerant `Unit` kernels. With the earlier kernels the nested unit
chain was 75 mul, and the conclusions were the same.

## Compile time and code size

These are debug builds of the library alone; generic code is compiled only when used.

| algebra | generated lines | product impls | sandwich kernels | debug build |
|---|---|---|---|---|
| VGA2D | 6k | 190 | 20 | 0.7 s |
| VGA3D | 13k | 363 | 42 | 1.5 s |
| PGA2D | 20k | 602 | 108 | 2.2 s |
| STA | 24k | 470 | 48 | 3.7 s |
| PGA3D | 35k | 733 | 140 | 4.8 s |
| CGA2D | 40k | 739 | 80 | 5.9 s¹ |
| CGA3D | 73k | 739 | 80 | 14.4 s |
| STAP | 58k | 571 | 54 | 7.0 s |
| CSTA | 111k | 1044 | 48 | 31.7 s |

¹ Measured on a busy machine, where CGA3D took 18 s instead of 14.4 s.

A release build with the first six algebras takes 27 s. Regenerating all eight takes about two
minutes in release. Most of it goes to CSTA (100 s) and to simplifying CGA3D's 16-component
versors. The line counts above predate the batch kernels (about 5% more per algebra now).

* **Why generated code costs nothing unused.** Every generated function is either generic (over slots
  and coefficients) or `#[inline]`, so machine code exists only for what a program uses.
* **What does cost.** The cost is in parsing and type-checking the generated impls. That is why the
  algebras are separate cargo features.
