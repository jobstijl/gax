# Performance

All measurements are on an AMD Ryzen 7 5800X with rustc 1.98.1. They use `--release`, and
`-C target-cpu=native` (AVX2 and FMA) where stated. Assembly counts come from the
`#[inline(never)]` probes in `crates/gax-bench/src/lib.rs`
(`cargo rustc -p gax-bench --release --lib -- --emit asm`).

Reproduce the timings with:

```sh
RUSTFLAGS="-C target-cpu=native" cargo bench -p gax-bench --bench transform
```

## Rigid transforms of points (PGA3D, f32)

| operation | gax | glam 0.30 |
|---|---|---|
| one point, direct fused sandwich `Unit<Motor> >> Point` | 6.6 ns | quaternion rotation only (`Quat * Vec3A`): 3.2 ns |
| one point, precomputed map `Point<(Point,)>::of` | 4.0 ns | `Affine3A::transform_point3a`: 2.2 ns |
| 1024 points, direct sandwich (AoS) | 1.88 µs | — |
| 1024 points, map built once, then applied (AoS) | 1.44 µs | `Affine3A` loop: 1.11 µs |
| 1024 points, direct sandwich (SoA `f32x8`) | 0.72 µs | — |
| 1024 points, map built once (SoA `f32x8`) | **0.50 µs** | — |

The `-C target-cpu=native` figures are the ones above. On the baseline x86-64 target, where `f32x8`
is emulated with two SSE registers:

* the SoA map path takes 1.12 µs, against 1.48 µs for glam;
* the direct single transform takes 7.9 ns, against glam's 3.0 ns (affine) and 3.5 ns (quaternion).

### Where gax is faster

**Batches in struct-of-arrays form.** A `Point<(), f32x8>` holds eight points, and every generated
kernel runs on it unchanged. Transforming 1024 points through the motor's map takes 16 lane FMAs per
eight points, 2.2x faster than a glam `Affine3A` loop. This is hypothesis 9, and the reason to batch.

### Where gax is slower, and why

**Single transforms in array-of-structs form are 1.8–3x slower than glam.**

* The fused sandwich of a unit motor on a general point is 25 mul and 18 add, all scalar: the 43
  arithmetic instructions in the assembly. glam's quaternion rotation is 27 instructions *because it
  is written with SSE shuffles on a 4-lane `Vec3A`*. glam's affine transform is 9 instructions for the
  same reason, and because its matrix is stored column-major for broadcasts.
* gax's kernels are generic over the coefficient type, so they are scalar code, and LLVM's SLP
  vectorizer only partly recovers SIMD from them.
* The output-first (row-major) storage of a map makes each output a dot product. That suits SoA lanes
  and not single-point SIMD.

**Planned improvements**, tracked in `TODO.md`:

* a prepared sparse map type for versor-to-kind transforms, which skips the structural zeros of the
  motor's point map: 12 mul and 9 add instead of 16 and 12;
* optional `mul_add` emission when FMA is available;
* for users who need single-point speed, tracing a kernel with the point's weight fixed at 1 (see
  below).

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
