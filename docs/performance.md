# Performance

All measurements are on an AMD Ryzen 7 5800X with rustc 1.99.0, taken again on 2026-10-05 (the
transform, compare and batch benches; the 7D to 9D rows earlier) on an otherwise idle
machine (load average below 0.5 before the runs, 1 to 2 during them: the
benchmarks themselves). Outliers were re-run in isolation. A single call of a few nanoseconds
moves by 10–25% between runs and between compiler versions (code placement and inlining in the
benchmark loop), so compare such rows within one run. They use `--release`, and
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
| transform one point: `Unit<Motor> >> Point` (fused, FMA) | 4.0 ns (4.7 in a second run) | `Affine3A::transform_point3a`: 1.3 ns; `Quat * Vec3A` (rotation only): 2.4 ns |
| transform one point: prepared sparse map `m.prepare::<Point>() >> p`, by value / read from memory | 7.7 / 3.1 ns | |
| transform one point: dense map `Point<(Point,)>::of`, by value / read from memory | 3.2 / 2.4 ns (5.7 by value in one run) | |
| transform 1024 points, AoS: direct / prepared / dense map | 1.50 / 1.10 / 1.14 µs | `Affine3A` loop: 0.72 µs |
| transform 1024 points, SoA `f32x8`: direct / **prepared** | 0.47 / **0.25 µs** | (2.9x faster than glam) |
| compose motors `Unit<Motor> * Unit<Motor>` | 5.2 ns | `Affine3A * Affine3A`: 3.2 ns; `Quat * Quat`: 1.2 ns |
| invert a unit motor (the reverse) | **1.2–1.5 ns** | `Affine3A::inverse`: 8.4 ns |
| normalize a motor (Study-number `rsqrt`) | 4.4 ns | `Quat::normalize`: 1.8 ns |
| motor exponential `Line::exp` (rotation and translation): one call / per argument over 256 | 18 / 9.2 ns | `Quat::from_scaled_axis` (rotation only, the same work of finding the angle): 5.0 / 4.6 ns; `Quat::from_axis_angle` (angle given): 2.8 ns |
| motor logarithm `Unit<Motor>::log`: one call / per argument over 256 | 17 / 16 ns | — |
| build the point map: `m.prepare::<Point>().to_map()` / `m >> Point::slot()` | 4.6 / 9.7 ns | `Affine3A::from_rotation_translation`: 5.0 ns |

**The prepared map passed by value** takes 7.7 ns, and 3.1 ns read from memory (from an array,
or through a reference). Its 13 entries (52 bytes) are passed through memory, and the kernel,
which LLVM vectorizes by columns, reads them with loads that straddle the stores that just wrote
them (offsets 12, 28 and 44), so the processor cannot forward the stored values and waits for
them to reach the cache (store-forwarding stalls). Only a map stored right before each use pays
this: a non-inlined call taking a `Prepared` by value, or this benchmark, which passes it through
`black_box` on every iteration. In loops and in batches the prepared action is the fastest form
(1.03 µs per 1024 points in AoS form, 0.24 µs in SoA form). A layout the column loads could
forward from would be the padded 4×4 matrix, which is the dense map. (With rustc 1.98.1 the by-value
row measured 3.5 ns: how the compiler copies the map decides whether the stall happens.)

### Where gax is faster

* **Batches in struct-of-arrays form.** A `Point<(), f32x8>` holds eight points, and every generated
  kernel runs on it unchanged. The prepared action of a unit motor needs 12 multiplications per
  eight points, with no structural zeros. That transforms 1024 points 2.8x faster than a glam
  `Affine3A` loop (hypotheses 6 and 9).
* **Inverting a unit motor** costs only negations: the certificate in `Unit` makes the inverse the
  reverse.

### Where gax is slower, and why

* **Single operations in array-of-structs form** take 2–6x as long as glam (3x for the motor
  sandwich on a point):
  * gax's kernels are generic over the coefficient type, so they are scalar code, and LLVM only partly
    recovers SIMD from them;
  * glam writes the same operations by hand with SSE shuffles on 4-lane registers, and stores
    matrices column-major for broadcasts.
* **A motor holds a rotation and a translation in 8 numbers.** Composing, normalizing and
  exponentiating it do more work than the quaternion alone, which is what glam's `Quat` timings
  measure. `Affine3A` is the fair comparison for rigid motions: gax is at 1.6x for composition and
  5.6x faster for inversion.
* **exp and log** go through closed forms with a series near zero (ADR-019). They take 18 and 17 ns
  for one call, 9.2 and 16 ns per argument over 256 arguments, against glam's 3–5 ns for building
  a quaternion from an axis and angle. A motor's exponential also produces its translation part
  and needs the square root of the bivector's norm.
* **The single-call exp moves with code placement.** It took 15.2 ns on 2026-10-02 and takes
  18.0 ns now, but measured commit by commit in between it reads 15.4, 15.9, 17.0, 17.2, 16.0,
  17.0, 16.8 and 18.0 ns, mostly at commits that do not touch its code. The one that does (scalar
  branches in exp and log, 4521125) costs it 0.5 ns and takes log from 27.9 to 16.8 ns. The
  256-argument rows are steadier, so compare those.

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

**SIMD-friendly layouts for single values: investigated, not adopted (2026-09-30).** glam's
kernels are written per operation on 4-lane registers. The generator's analogue would emit a
product "lane-grouped": each group of four outputs sums the same number of isomorphic terms, a
broadcast of one left coefficient times a shuffle of the right, as elementwise code for LLVM's
SLP vectorizer. The PGA3D motor product has that shape exactly (outputs 0–3 take one term per
left coefficient 0–3, outputs 4–7 one per 0–7, each slot one shuffle of one register), so it is
the best case:

| `Motor * Motor` (f32) | default target | `-C target-cpu=native` |
|---|---|---|
| generated (balanced sums, as now) | 12.2 ns | 6.6 ns |
| lane-grouped, signs as a factor | 18.6 ns | **5.3 ns** |
| lane-grouped, signs as add/sub | 17.2 ns | 14.7 ns |
| glam `Affine3A * Affine3A` | 4.3 ns | 4.2 ns |

With FMA and AVX2 it saves a fifth; without them LLVM does not pack it well and it costs half as
much again. Adopting it would take per-target code paths (`cfg(target_feature)`) for products,
while the kernels that matter most for single values, the simplified sandwiches, are
straight-line programs without this structure. glam's remaining lead comes from layouts chosen
per type (a 3×4 matrix, whose product is broadcast-shaped) and hand-written shuffles; the batch
kernels (struct of arrays) are where gax is faster instead.

## Against nalgebra, ultraviolet and `geometric_algebra` (`benches/compare.rs`)

| operation | gax | others |
|---|---|---|
| transform a point by a rigid motion (f32) | 4.0 ns | `geometric_algebra` 0.3 `Motor::transformation`: 7.4 ns; nalgebra `Isometry3 * Point3`: **3.2 ns** |
| eight points at once (SoA `f32x8`) | 4.9 ns (rotation and translation; prepared: 5.8 ns) | ultraviolet `Rotor3x8 * Vec3x8` (rotation only): 3.7 ns |
| compose a chain of 5 rigid motions | 28 ns | glam `Affine3A`: 9.5 ns; nalgebra `Isometry3`: 19 ns |
| 6x6 map inverse (f64) | **101 ns** | nalgebra `Matrix6::try_inverse`: 211 ns |
| 6x6 solve | **101 ns** | nalgebra LU solve: 125 ns |
| 6x6 generalized eigenproblem (vibration modes) | **1.15 µs** | nalgebra Cholesky + `SymmetricEigen`: 1.30 µs |
| 4x4 SVD | **0.60 µs** | nalgebra `Matrix4::svd`: 0.88 µs |
| CGA3D `Unit<Motor> >> point` (f32) | 5.7 ns | — |
| CGA3D general even versor `>> point` | 25.9 ns | — |
| CGA3D `Twist::exp` | 10.5 ns | — |
| CSTA (6D) vector product | 12.9 ns | — |
| CSTA `Unit<Even>::log` (f64): closed form (log6d.md) / near a half turn (turned) / inverse scaling and squaring (before, not measured again) | **2.7 µs** / 6.6 µs / 50.7 µs | — |
| 7D / 8D / 9D `Unit<Even>::log` (f64, `R(4,3)`, `R(4,4)`, `R(5,4)` declared with `algebra!`): closed form / turned | 2.9 / 3.0 / 4.0 µs; 8.0 / 10 / 27 µs | — |
| CSTA `Bivector::exp` (f64): closed form (log6d.md §12) / beyond a quarter turn (turned) / scaling and squaring (before) | **3.4 µs** / 4.2 µs / 4.1 µs | — |
| general `inverse` (Shirokov, ADR-037; CGA3D `Even` in closed form): PGA3D `Multivector` / CGA3D `Even` / CGA3D `Multivector` / CSTA `Even` / CSTA `Multivector` | 0.26 / **0.061** / 2.2 / 2.7 / 15 µs | — |
| `pinv` of a 6 × 4 map (one-sided Jacobi, ADR-038) / `lstsq` against a value / LU `inverse` of a 6 × 6 map for scale | 1.01 / 1.01 / 0.13 µs | — |
| `Unit<Motor> >> Point` and `Line::exp`, `f64` / with six derivatives (`Dual<f64, 6>`, ADR-039) | 4.5 / 41 ns; 19 / 48 ns | — |
| 7D / 8D / 9D `Bivector::exp` (f64): closed form / scaling and squaring (before) | **4.3 / 6.7 / 10 µs**; 35 µs / 0.24 ms / 0.95 ms | — |
| 7D / 8D / 9D even product (4096 terms unrolled; 16384 and 65536 as table loops, ADR-034) | 1.7 / 11 / 43 µs | — |

**nalgebra's `Isometry3 * Point3`** took 11.5 ns in the earlier measurement (rustc 1.98.1) and
takes 3.2 ns with rustc 1.99.0, faster than gax's motor sandwich for a single point. The 7D to
9D rows come from a separate harness with `algebra!` declarations (building 9D algebras is too
heavy for the benchmark crate); the rest from `benches/compare.rs`. CGA3D's general `Even::inverse`
went from 373 to 61 ns when 5D even elements got their inverse in closed form (2afe3a2).

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

| | arithmetic | one body (f32), two runs | eight bodies (`f32x8` lanes) |
|---|---|---|---|
| generic code | 113 mul, 70 add, 5 div | 21.1 / 21.3 ns | 28.6 ns |
| fused at build time | 98 mul, 70 add, 1 div | 22.9 / 17.9 ns | 24.1 ns |
| fused, without `mul_add` (`Tracer::fma(false)`) | the same | 22.0 / 23.2 ns | **22.4 ns** |

* **Batches.** In SoA lanes the arithmetic count is what runs: the fused kernel is 16% faster than
  the generic code, and 22% without `mul_add`.
* **Single values.** The three variants are within the spread between runs (18 to 24 ns). An
  earlier measurement (rustc 1.98.1, 2026-09-29) found the fused kernel without `mul_add` 35%
  faster than the generic code (15.5 ns against 20.8 ns): with FMA each single-use product
  feeding a sum became a scalar `vfmadd`, which LLVM's SLP vectorizer did not pack, while it
  packed the plain products. With rustc 1.99.0 that no longer reproduces.
* **Loops.** In a scalar loop of 1024 steps (`benches/batch.rs`) the fused kernel takes 18.2 µs
  and the generic code 16.2 µs; on `f32x8` lanes the fused kernel takes 2.56 µs.
* **Why it moves with the compiler.** For single values, LLVM's SLP vectorizer decides whether a
  kernel runs packed or scalar, and it changed its mind between versions. The probes
  `probe_rigid_fused` and `probe_rigid_generic` (`--emit asm`, `target-cpu=native`): with rustc
  1.98.1 the fused kernel is 147 instructions, packed (`vmulps`, `vfmadd213ps`) with 2 stack
  accesses, and the generic code 189, mostly scalar with 13 (spills); with 1.99.0 it is the other
  way round, the fused kernel 160 instructions, mostly scalar (`vfmadd231ss`) with 14 stack
  accesses, and the generic code 165, packed with 2. Only explicit SIMD per target would pin
  this down, and lanes and batches, where gax's speed comes from, do not depend on it.
* **What it means:**
  * `mul_add` stays the default: it rounds each multiply-add once, and the variants are within
    10% of each other on lanes;
  * for a hot single-value kernel, measure `Tracer::fma(false)` against the default on the target;
  * tracing pays off whenever constants remove work, and a fused kernel's operation count is what
    runs on lanes.
* **Two experiments, measured and not merged** (2026-10-03, idle machine, each against its own
  base commit, `benches/transform.rs` twice and `benches/batch.rs` once):
  * *Explicit SIMD for the PGA3D motor sandwich and product* (`std::arch` per target): a single
    `Unit<Motor> >> Point` on the default target went from 6.5 to 4.7 ns, but with
    `target-cpu=native` it was within noise (4.6 against 4.7 ns), and the loop over 1024 points
    went from 1.42 to 3.81 µs (native) and 1.70 to 4.84 µs (default): the explicit kernel keeps
    LLVM from vectorizing across points, where the scalar kernel is vectorized.
  * *16-byte alignment of the kind structs*: no change on 69 batch rows (median +0.1%), the SoA
    point transform 11–12% slower, single values within noise.
  * Noise: the same binary measured twice moved single-value rows by up to 50% (and an inverse
    from 1.2 to 3.0 ns), which is larger than most effects under test; rows over 1 µs agree to
    within a few percent.

## Fused sandwich kernels (op counts from the generator)

The full list, for every algebra and both paths, is generated into
[kernel-costs.md](kernel-costs.md) and checked by CI.

**Why the line sandwich stays at 58 mul.** A unit PGA3D motor factors as `m = T r`: its rotor
part `r` (the blades without `e0`) is itself unit, and `T = m ~r` is a translator with scalar
part exactly 1. Rotating a line and then translating it would cost about 52 mul: 34 to rotate
(a rotation matrix shared by the direction and the moment), 12 to extract `T`, 6 for the cross
product. But a `Unit` kernel must be homogeneous of degree 2 in the versor, so that a drifted
motor scales results uniformly instead of distorting them (ADR-020). The factored form is of
degree 4 once reduced, and restoring degree 2 costs a division or about 10 multiplications for
the norm, which is the whole saving. Plain motors cannot factor at all without dividing by
`r ~r`. The generator's single-expression kernel is therefore kept (investigated 2026-09-29).

| kernel | gax | reference |
|---|---|---|
| `Unit<Rotor> >> Point` (PGA3D) | 26 mul, 15 add (18 mul, 12 add before) | the quaternion formula (not homogeneous): 18 mul, 12 add |
| `Unit<Motor> >> Point` (PGA3D, general weight) | 33 mul, 21 add (25 mul, 18 add before) | GAmphetamine, weight fixed at 1: 21 mul, 18 add |
| `Motor >> Point` (not unit) | 38 mul, 32 add | GAmphetamine, weight fixed at 1: 28 mul, 21 add |
| `Unit<Motor> >> Plane` | 36 mul, 21 add (28 mul, 18 add before) | — |
| `Unit<Motor> >> Line` | 58 mul, 45 add | rotating, then translating: about 52 mul, but not homogeneous (above) |
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
| one motor, 1024 points | default | 1.12 µs | 0.49 µs | 0.73 µs | 0.93 µs | **0.32 µs** |
| | native | 1.05 µs | **0.26 µs** | 0.71 µs | 0.96 µs | 0.28 µs |
| 1024 motor-point pairs | default | 2.76 µs | 1.28 µs | 0.74 µs¹ | 2.25 µs | **0.53 µs** |
| | native | 2.49 µs | **0.48 µs** | 0.78 µs¹ | 2.34 µs | 0.57 µs |
| exp of 1024 twists | default | 28.5 µs | 4.26 µs | 24.9 µs² | 3.51 µs | **1.83 µs** |
| | native | 11.6 µs | 1.94 µs | 6.95 µs² | 3.54 µs | **1.83 µs** |
| rigid-body step, traced (1024 bodies) | default | 18.6 µs | 6.70 µs | — | 6.99 µs | — |
| | native | 16.6 µs | **2.60 µs** | — | 7.05 µs | — |
| rigid-body rate, traced (1024 bodies)³ | default | 4.79 µs | 1.96 µs | — | 2.95 µs | **0.99 µs** |
| | native | 4.64 µs | **0.67 µs** | — | 2.90 µs | 0.85 µs |

¹ `Affine3A::transform_point3a` per pair, with the affines built beforehand (9 multiply-adds,
against the motor sandwich's 24 multiplies and 14 adds per point).
² `Quat::from_scaled_axis`: rotations only, where the motor exponential includes translations.
³ The velocity half of the step, `Line` to `Line`, which has a struct-of-arrays form (`name_batch_soa`);
the full step returns a tuple and is batched in AoS form only.

The default build was measured on 2026-10-02, the native one again on 2026-10-05. The native
scalar rigid-body step took 14.0 µs with rustc 1.98.1, 18.2 µs with 1.99.0 on 2026-10-02 and
16.6 µs now. In a
native build the "portable" batch path is compiled for AVX2 and FMA too, so it matches the AVX2
dispatch (SoA with one motor: 0.23 µs portable, 0.25 µs AVX2, 0.41 µs on SSE2 and SSE4.2).

A built map applied to many values (`BatchOf`) runs at the same speed as the prepared
motor action: the dense 4x4 of `m >> Point::slot()` takes 1.01 µs in AoS form and 0.31 µs in SoA
form, against 0.93 µs and 0.32 µs for `transform_slice` and `transform_soa` measured in the same
run (default build, AVX2). At this size, both are bound by memory, not arithmetic. So any composed
map (a camera and a projection, a lens) gets the SIMD path that a single motor gets.

Per level, in the default build, SoA with one motor takes 0.47 µs portable, 0.48 µs on SSE2 and
SSE4.2, and 0.32 µs on AVX2.

* **In a default build, the dispatched kernels are the fastest option.** They use AVX2 and FMA,
  which statically compiled code cannot assume.
* **With `target-cpu=native`, `wide` catches up** on the same generic kernels, and it wins where
  the batch form works on arrays of structs. The traced rigid step is batched only in AoS form,
  and five transposes per batch cost more than the kernel. The SoA form of the traced rate kernel
  comes within 15% of `wide`, and it beats `wide` by 2x in the default build.
* **The vectorized elementary functions are fast.** `exp` on SoA is 16x faster than scalar gax and
  2.3x faster than `wide` in the default build, and it still beats `wide` natively.
* **Arrays of structs cost transposes.** The AoS forms beat scalar loops but not glam's
  single-point affine transform, whose data layout is its SIMD layout. Keep hot data in `Soa`.

How the kernels stay vectorized (inlining, `Coef::vectorize`, the block layout) is in ADR-023 of
the [design record](design.md).

## GPU (`examples/wgpu -- --bench`)

These were measured on an AMD RX 6900 XT (RADV, Vulkan), with `2²⁰` instances or particles, on
an idle machine (measured again on 2026-10-02). See [shaders.md](shaders.md).

**Instancing: a motor per instance against a matrix per instance.** `2²⁰` PGA2D triangles are
drawn into a 1920x1080 target, each placed either by a unit motor (the vertex shader runs
`unit_motor_sandwich_point`) or by the motor's matrix `m >> Point::slot()` (a `mat3x3`,
computed on the CPU). Times are per frame:

| per instance | bytes (with colour) | CPU preparation | upload + draw | draw only |
|---|---|---|---|---|
| unit motor | 32 | 3.7 ms | 7.3 ms | 5.7 ms |
| matrix | 64 | 7.9 ms | 10.4 ms | 5.7 ms |

* The draw costs the same either way; the sandwich in the vertex shader is free here, since
  the frame is bound by rasterization.
* The motor saves half the upload and most of the CPU work: the matrix has to be built from
  the motor for every instance.
* A PGA2D motor is one `vec4`; a 2D affine matrix needs three in WGSL's layout.

**The particle step on the GPU and on the CPU.** One step of `particle_step` (`m exp(dt B)`
followed by a Newton renormalization, traced from `src/kernels.rs`) for `2²⁰` particles:

| where | per step |
|---|---|
| CPU, the fused kernel in a scalar loop | 5.9 ms |
| CPU, its batch form (SIMD, one thread) | 4.1 ms |
| GPU, the same kernel in a compute shader (pipelined) | 0.12 ms |
| GPU, submit and wait | 0.16 ms |

It is the same program on both sides, so moving an effect to the GPU changes nothing in what
it computes. `--check` compares GPU particles with their CPU twins after up to two seconds of
flight: they agree to `7·10⁻⁶`.

**CSTA's two bivector exponentials on the GPU** (`crates/gax-gpu-tests`, `csta_exp_timing`,
ignored by default; `2¹⁸` invocations of 32 dependent exponentials each, bivectors with entries
up to 1):

| | per exponential | within, of the Rust `f64` exp: entries up to 1 / up to 3 / a half turn and a boost of 12 |
|---|---|---|
| `bivector_exp` (scaling and squaring in the even kind) | **1.8 ns** | `2·10⁻⁵` / `8·10⁻⁵` / `3·10⁻⁵` |
| `bivector_exp_closed` (the closed form, log6d.md §12) | 10 ns | `2·10⁻⁶` / `4·10⁻⁶` / `3·10⁻⁵` |

On the CPU, in `f64`, the closed form is the faster one (3.4 µs against 4.1 µs, and 35 µs to
4.3 µs in 7D). On a GPU it is not: its series (16 terms) and tables (24) are local arrays, which
a GPU keeps in memory rather than registers, while scaling and squaring is uniform multiply-adds.
Two changes brought it from 40 ns to 10 ns: `bivector_exp_from` called once instead of in both
branches of the turning test (invocations of one wavefront that differ in turning ran both;
19 ns), and tables of 24 terms instead of 40, which `f32` does not need (the shift to a centre
is the helper's main cost). Computing the invariants once instead of twice would not pay: the
kernels that compute only them (`bivector_exp_reach`, `bivector_exp_turning`) take under 0.1 ns.
In the `f16` module, `bivector_exp` is the closed form, within `4·10⁻³` of the Rust exp where
scaling and squaring in `f16` is off by up to `4·10⁻²` (`f16_exponentials_on_the_gpu`).

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
| VGA2D | 10k | 190 | 20 | 0.4 s |
| VGA3D | 21k | 363 | 42 | 0.9 s |
| PGA2D | 34k | 602 | 108 | 1.4 s |
| STA | 34k | 470 | 48 | 1.6 s |
| PGA3D | 54k | 733 | 140 | 2.5 s |
| CGA2D | 47k | 739 | 80 | 2.3 s |
| CGA3D | 112k | 889 | 88 | 6.3 s |
| STAP | 65k | 571 | 54 | 3.7 s |
| CSTA | 120k | 1044 | 48 | 11.6 s |

Measured on 2026-10-02 with rustc 1.99.0, rebuilding `gax` alone with one algebra (its
dependencies built). With rustc 1.98.1 the same builds took 2 to 3 times as long (CSTA 31.7 s,
CGA3D 14.4 s), with fewer generated lines.

A release build with the first six algebras takes 10.4 s (27 s with rustc 1.98.1).

**Regenerating** all nine standard algebras (`gax-regen`) takes 48 s on 16 cores (34 s when
the generator was first parallelized, before the kind tables, the general inverse and the
closed-form WGSL kernels). It took
about 8 minutes when it ran on one core: CSTA 205 s, CGA3D 173 s, STAP 63 s, STA 23 s,
CGA2D 13 s, and the rest a few seconds. The generator's jobs are pure functions of the
algebra (ADR-030), so it now runs them in parallel and writes their output in order:

* the sandwich kernels, heaviest first, and in each the value path beside the map path;
* the outermorphisms, beside the sandwiches;
* the value methods of each kind;
* the law derivations, one per versor and product;
* the strategies of the compiler's portfolio, for programs over 2000 terms.

Its output is byte for byte the sequential one: `gax-regen --check` passes. What remains is
the longest single jobs, 13 s each: CGA3D's unit sandwiches on a whole multivector and CSTA's
outermorphism from vectors to quadvectors. `gax-regen --verbose --only <algebra>` times one
algebra's phases, and `GAX_GEN_PROFILE=1` lists every job over half a second.

**Building the library** is type checking, not code generation (the code is generic). With
every algebra, rustc 1.98.1 spent 39 s: 20.7 s type checking, 10.3 s borrow checking, 2.3 s writing
metadata, 1.9 s on coherence (`-Z time-passes`). Binding each product's term operator once,
as a closure (ADR-031), cut CSTA alone from 16.3 s to 13.9 s, with the release code
unchanged. Line tables instead of full debug info in the dev profile halve a test binary
(45 MB to 22 MB for `ops_pga3d`).

* **Why generated code costs nothing unused.** Every generated function is either generic (over slots
  and coefficients) or `#[inline]`, so machine code exists only for what a program uses.
* **What does cost.** The cost is in parsing and type-checking the generated impls. That is why the
  algebras are separate cargo features.
