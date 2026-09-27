# What warp verifies about gax

The game's second purpose is to validate gax under real load. This file lists every gax feature
it uses, where, and what was found. It covers milestone M2 (the Plane vertical slice) and grows
with the game. The **friction log** at the end is the most valuable part: each entry is a gax
issue or fix.

## Features used

| gax feature | where | found |
|---|---|---|
| `Unit<Motor>` poses, `exp` of twists, `renormalize_fast` every tick | `sim/body.rs`: every body, `pose ← exp(dt B) pose` | Over a 10-minute integration, `m ~m` stays within `1e-5` (a test); in play, the worst drift per session is below `4e-7` (F3 shows it) |
| Twists add: translation plus rotation | `Body::twist` | Works as intended, once the twist constructors existed (friction 1) |
| Motor interpolation, `a exp(t log(~a b))` | drawing between ticks (`Body::lerp`) | Every entity, every frame; `log` of nearly-identity motors is on the series path (fixed in the numerics work) |
| `log` for a camera spring | `fx.rs`: critically damped spring on `log(target ~cam)` | Smooth; shake is `exp(ε B) cam` |
| Join, signed distance, perpendicular | `sim/collide.rs`: swept bullets (`a & b`, `l & c`, `l \| c`) | Catches tunnelling (a test); the sign conventions of the hull and facing tests were right the first time |
| Distance as the norm of a join | `body::distance`, the lattice kernel | `(p & q).norm()` |
| Maps: `cam << Point::slot()`, `proj.of(view)`, `GpuMat` | `render/scene.rs`: the camera | The view-projection is a gax map, uploaded in the WGSL layout |
| `{Kind}Gpu` Pod types | line instances (`MotorGpu`), lattice nodes and particles (`PointGpu`) | Layouts asserted at compile time; no hand-written byte layouts in the game |
| Generated WGSL modules (`gax::wgsl`) | every shader imports `gax::pga2d` through `wesl` at build time | `unit_motor_sandwich_point` places every shape segment in the vertex shader |
| Traced kernels, CPU and GPU (`Tracer::wgsl`) | the lattice (`grid_node`, `source_force`) and particles (`particle_step`) | See "CPU and GPU" |
| Batch SoA path as the CPU twin | `render/cpu_grid.rs`: `grid_node_batch`, `source_force_batch` | The oracle for the GPU (a test) |
| The deterministic mode | not used yet | Replays (M3) will state same-build determinism, or use `deterministic` |

## Numbers (M2)

* **CPU and GPU**, the same traced programs (`headless.rs` tests; RX 6900 XT, RADV):
  * particles after 3 steps: `2e-7`, relative;
  * the lattice after 1 step: `1.5e-7`, which is ulps;
  * the lattice after 240 steps with blasts and a well: `4.4e-4`. The rounding differences
    between the fused CPU form and the GPU's `fma` accumulate through the coupled springs, as
    expected for an iterated coupled system; they do not grow into a different shape.
* **Traced kernel costs:**
  * `grid_node`: 57 mul, 57 add, 4 div, 8 calls (the same as the generic code: there is nothing
    to fuse in a sum of spring forces);
  * `source_force`: 10 mul, 7 add, 1 div;
  * `particle_step`: 10 mul, 7 add.
* **Frame rate:** the startup test (`--smoke`, a bot playing for 8 s) held vsync, 60 Hz on this
  display, with a lattice of 9 417 nodes and a pool of 262 144 particles.

## Friction log

1. **PGA2D twist conventions were easy to get wrong.** A translation at velocity `(vx, vy)` is
   the bivector `(vy/2, −vx/2, 0)` in `[e20, e01, e12]`.
   * `examples/asteroids` wrote its own helper, and `examples/wgpu` got it wrong: it used
     `(v/2, 0, ω/2)`, which moves along the local y axis, not forward along x. It now uses
     the new constructors.
   * **Fixed in gax:** `Point::translation_twist(vx, vy)` and `Point::rotation_twist(centre, ω)`
     in PGA2D, and `Line::translation_twist` and `Line::rotation_twist` in PGA3D, each tested
     against `Motor::translation` and `Motor::rotation`.
2. **Tracing a `Unit` argument simplifies renormalization away.** The tracer assumes `m ~m = 1`
   for a `Unit`, so `renormalize_fast` inside a kernel becomes the identity. Kernels that must
   renormalize take the plain kind. This is documented in `docs/shaders.md`; a lint or warning
   in the tracer would be better (open).
3. **Numeric literals don't pin the coefficient type.** `Point::translation_twist(3.0, -1.0)`
   in a doctest failed to infer `T`. It's a known Rust limitation, documented in the guide's
   pitfalls; annotate `Point::<(), f64>`.
4. **An "ideal norm" of a direction is missing.** The Euclidean length of a point at infinity
   (a velocity) has no method; `norm()` of a PGA2D point is its weight, which is 0 there. The
   game uses `sqrt(e20² + e01²)` (`body::length`). Worth a gax method (open).
5. **Scalar multiplication is `gp`.** `v.gp(dt)` reads oddly next to `+`. `Mul<T>` for
   coefficients would be friendlier, and might conflict with `*` meaning the geometric product
   (open question).
6. **The WGSL modules are large.** PGA2D is 120 KB of source. `wesl`'s stripping keeps only
   what the shaders use, so it costs nothing at run time, but build scripts parse all of it.
