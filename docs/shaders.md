# Shaders: gax in WGSL

gax generates a WGSL module for each standard algebra. The kernels in it are printed from the
same verified straight-line programs as the Rust kernels, so a motor sandwich on the GPU
computes the same polynomial as `m >> p` on the CPU. The design is in ADR-028 of the
[design record](design.md).

* **Features.**
  * `wgsl` gives the module sources, `gax::wgsl::{PGA2D, PGA3D, …}`, and their `f16`
    versions, `gax::wgsl::{PGA2D_F16, …}`.
  * `bytemuck` gives the matching Rust types, `{Kind}Gpu`, `{Kind}Gpu16` and `GpuMat`, with
    `bytemuck::Pod`.
* **No runtime dependencies.** The modules are text, so gax does not depend on `wesl`, `naga`
  or `wgpu`, and cannot conflict with your engine's versions of them.

## What a module contains

**Kinds.** A kind with N coefficients is a struct of `ceil(N/4)` `vec4<f32>` fields
`c0, c1, …`. The coefficients are in the Rust blade order, zero-padded; each struct lists its
blades in a comment.

| kind (PGA3D) | coefficients | WGSL | bytes |
|---|---|---|---|
| `Point` | 4 | `struct Point { c0: vec4<f32> }` | 16 |
| `Plane` | 4 | `struct Plane { c0: vec4<f32> }` | 16 |
| `Motor` | 8 | `struct Motor { c0: vec4<f32>, c1: vec4<f32> }` | 32 |
| `Line` | 6 | `struct Line { c0: vec4<f32>, c1: vec4<f32> }` | 32 |

A PGA2D motor or point is one `vec4`. The layout works in uniform, storage and vertex buffers:
in a vertex buffer, each field is one attribute location.

**Functions**, for every kind:

| name | what |
|---|---|
| `point_new(e032, e013, e021, e123)` | a value from its coefficients |
| `motor_from_rotor(r)` | an embedding between versor kinds |
| `motor_reverse(m)` | `~m` |
| `motor_mul_motor(a, b)` | the geometric product of versor kinds (composition) |
| `motor_sandwich_point(m, p)` | `m p ~m`, for any motor |
| `unit_motor_sandwich_point(m, p)` | the same for `m ~m = 1`: the cheaper, drift-tolerant `Unit` kernel |
| `motor_matrix_point(m)`, `unit_motor_matrix_point(m)` | the sandwich's matrix (`mat4x4<f32>`), applied as `M * p.c0` |
| `motor_normalized(m)` | `(m ~m)^(-1/2) m` |
| `motor_renormalize_fast(m)` | one Newton step towards `m ~m = 1`, with no square root |
| `motor_norm_squared(m)` | the scalar part of `m ~m` |
| `line_exp(b)` | the exponential of a bivector: a unit motor |
| `unit_motor_log(m)` | its inverse, with the rotation half-angle in `[0, π]` |

The names are `snake_case` kind names joined by the operation. `exp` and `log` exist wherever
the Rust code has them, in every algebra: rotations and motions through real trigonometry, boosts
and general versors (STA, CGA, STAP, CSTA) through the general Study functions, CSTA's
bivector `exp` by scaling and squaring, and the log of the full even kind in closed form for CSTA
and every 6D and 7D algebra (turning planes near a half turn first), as in Rust (log6d.md §8).
Kinds of more than 64 coefficients (from 8D on) have no WGSL kernels.

## Using a module

**With the `wesl` crate (0.5 and later)** register the modules and import from them. In
`wesl` 0.5, an import from another package needs the declarations marked `public`, and
`Module::wesl_public()` adds that:

```rust,ignore
use wesl::{CompileOptions, resolver::VirtualResolver};

let mut r = VirtualResolver::new();
for m in gax::wgsl::ALL {
    r.add_module(m.path.parse()?, m.wesl_public().into());
}
r.add_module("package::main".parse()?, my_shader.into());
let wgsl = wesl::compile(&"package::main".parse()?, &CompileOptions { strip: true, ..Default::default() }, &r)?
    .syntax
    .to_string();
```

```wgsl
import gax::pga3d::{Motor, Point, unit_motor_sandwich_point};
```

Stripping keeps what the shader uses. In the tests, a shader using two kernels keeps 8 of the
542 functions of the two modules it imports.

**With an engine's WESL resolver**, register [`Module::source`] under [`Module::path`] as a
shader library.
* Bevy on `wesl` 0.4 does not parse `public`, so use `source` there.
* Nothing in gax is engine-specific.

**With plain WGSL**, prepend a module's source to your shader. It is self-contained. Don't
prepend two algebras to one shader: both define `Motor`. WESL paths keep them apart.

**The Rust side.** `MotorGpu::from(motor)`, `PointGpu::from(point)` and back are lossless.
`bytemuck::cast_slice(&motors)` gives the bytes to upload. Their layout is asserted at compile
time and checked against naga in the tests.

**Matrices.** WGSL matrices are column-major, while gax maps store output rows. So
`GpuMat::<4>::from(m >> Point::slot())` transposes: each column is the image of one input
coefficient, and the shader applies it as `M * x`.

## Traced kernels

`Tracer::wgsl(true)` in `build.rs` also emits each traced kernel as a WGSL function, printed
from the same verified program as the Rust function:
* it writes `fused.wesl` next to `fused.rs`;
* `fused.rs` gets `pub const FUSED_WESL: &str`.

Arguments and results can be kinds (WESL paths such as `gax::pga3d::Point`), `Unit`s of them,
scalars (`f32`), and arrays and tuples of these. Register `FUSED_WESL` as a module of your
package, next to gax's modules.

**Algebras of your own.** With gax's `wgsl` feature, `algebra!` also generates the algebra's
modules, `WGSL_MODULE` and `WGSL_MODULE_F16` (the same functions as the standard modules), with
the paths `package::{algebra}` and `package::{algebra}_f16`. A kernel traced over its kinds names
them `package::{algebra}::Point`, so register the module under its path:

```rust,ignore
gax::algebra! { algebra plane "…"; basis e0 = 0, e1 = 1, e2 = 1; /* kinds */ }

r.add_module(plane::WGSL_MODULE.path.parse()?, plane::WGSL_MODULE.source.into());
r.add_module("package::fused".parse()?, FUSED_WESL.into());
``` Game logic on the CPU and
effects on the GPU then run one kernel, and cannot drift apart.

A traced kernel sees a `Unit` argument as certified. The simplifier then uses `m ~m = 1`, and a
renormalization inside the kernel would simplify away. The tracer notices: such a kernel gets
a warning in its report, in its generated docs, and as a cargo warning from `build.rs`. To
renormalize in a kernel, take the plain kind and wrap it yourself, as `examples/wgpu` does:

```rust,ignore
pub fn particle_step<T: Real>(m: Motor<(), T>, rate: Point<(), T>, dt: T) -> Motor<(), T> {
    (Unit::new_unchecked(m) * rate.gp(dt).exp()).renormalize_fast().into_inner()
}
```

## Numerics on the GPU

* **`f32`, or `f16`.** WGSL has no `f64`. Positions far from the origin lose accuracy as they
  do on the CPU (see [numerics.md](numerics.md)).
* **The `f16` modules** (`gax::pga3d_f16`, device feature `shader-f16`) have the same functions
  on structs of `vec4<f16>`, half the memory and bandwidth. The Rust types are `{Kind}Gpu16`,
  converted with correct rounding (`gax::gpu::f16_bits`, tested on every `f16`).
  * Straight-line kernels compute in `f16`, within their computed error bound for `f16`'s unit
    roundoff `2⁻¹¹` (on the RX 6900 XT, at most 0.4 of the allowance, which adds `2⁻¹³`
    absolute for flushed subnormals).
  * `exp`, `log` and `normalized` compute their norms and Study functions in `f32`, and only
    the final combination in `f16`: within `1.2·10⁻³` relative, about one `f16` ulp. A sum of
    squares in `f16` would underflow below coefficients of `1/128` and overflow above 256.
  * `f16` holds about three decimal digits and values up to 65504: positions in metres are
    good to millimetres only within a few metres of the origin. Keep positions in `f32`, and
    use `f16` for directions, rotations, normals and colours.
* **Arithmetic.** `+`, `-` and `*` are correctly rounded, and whether `fma` fuses is up to the
  implementation. Either way, a straight-line kernel is within its computed forward error
  bound. On an RX 6900 XT, the worst output over every kernel of five algebras reached 0.98
  of it.
* **Elementary functions.** WGSL specifies loose accuracy for them: `sin` and `cos` to an
  absolute `2⁻¹¹`. Kernels that use them (`exp`, `log`, `normalized`) agreed with their exact
  values within `1.5·10⁻⁵` on that GPU. The test allows `2⁻¹⁰`.
* **Determinism.** Cross-GPU determinism is out of scope. Lockstep logic belongs on the CPU,
  with gax's `deterministic` feature.
* **Drift.** For motors integrated on the GPU, renormalize every step or frame:
  `motor_renormalize_fast`, or `renormalize_fast` inside a traced kernel. The `unit_` kernels
  are drift-tolerant, so a drifted motor scales points uniformly instead of distorting shapes.

## How it is tested

1. **Printer.** Golden strings cover every instruction, the reversed `select`, `fma` and
   constants (`gax-gen`).
2. **Validation, with no GPU.**
   * naga parses and validates every module (`gax/tests/wgsl.rs`).
   * `wesl` compiles cross-package imports with stripping, and the traced module
     (`crates/gax-wesl-tests`).
3. **Layout, with no GPU.** naga's size, alignment and member offsets for every struct and
   matrix equal Rust's (`gax/tests/wgsl.rs`).
4. **CPU evaluation.** `wesl`'s evaluator runs every kernel of PGA2D, PGA3D, VGA2D, VGA3D and
   STA on random inputs (`crates/gax-wesl-tests/tests/eval.rs`).
   * Each output must be within the kernel's computed `f32` error bound of its exact value; the
     worst reached 0.96.
   * The evaluator implements neither `fma` nor non-`@const` calls, so the test evaluates `fma`
     as `a * b + c`. It does not test fusion.
5. **GPU execution** (`crates/gax-gpu-tests`, run in CI on Mesa's lavapipe):
   * every kernel of five algebras against its exact value;
   * the matrix orientation (`M * x` equals `m >> x`);
   * a layout round trip;
   * a traced kernel against its Rust twin.

   Without an adapter the tests pass with a message. With `GAX_REQUIRE_GPU` set, as in CI, they
   fail.

## The example

`examples/wgpu` is a plain wgpu program:
* ships are instanced triangles placed by unit motors in the vertex shader;
* explosions are particles stepped in a compute shader by a traced kernel, the same program
  the CPU uses for the ships;
* `build.rs` links the shaders against `gax::wgsl` with the `wesl` crate, so the program ships
  plain WGSL.

Run it with:
* `cargo run --release` for the window;
* `-- --check` to compare GPU particles with CPU twins, headless;
* `-- --bench` for the numbers in [performance.md](performance.md).

## Not included yet

Nothing that the Rust side has: every kind, product, sandwich, `exp` and `log` of every algebra,
standard or declared, has a WGSL form in `f32` and `f16`. Traced kernels are `f32`.

[`Module::source`]: https://docs.rs/gax/latest/gax/wgsl/struct.Module.html
[`Module::path`]: https://docs.rs/gax/latest/gax/wgsl/struct.Module.html
