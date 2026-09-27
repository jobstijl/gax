# What warp verifies about gax

The game's second purpose is to validate gax under real load. This file lists every gax feature
it uses, where, and what was found. It covers milestones M2 (the Plane vertical slice) and M3 so far
(the enemy roster), and grows with the game. The **friction log** at the end is the most valuable part: each entry is a gax
issue or fix.

## Features used

| gax feature | where | found |
|---|---|---|
| `Unit<Motor>` poses, `exp` of twists, `renormalize_fast` every tick | `sim/body.rs`: every body, `pose ← exp(dt B) pose` | Over a 10-minute integration, `m ~m` stays within `1e-5` (a test); in play, the worst drift per session is below `4e-7` (F3 shows it) |
| Twists add: translation plus rotation | `Body::twist` | Works as intended, once the twist constructors existed (friction 1) |
| Motor interpolation, `a exp(t log(~a b))` | drawing between ticks (`Body::lerp`) | Every entity, every frame; `log` of nearly-identity motors is on the series path (fixed in the numerics work) |
| `log` for a camera spring | `fx.rs`: critically damped spring on `log(target ~cam)` | Smooth; shake is `exp(ε B) cam` |
| Join, signed distance, perpendicular | `sim/collide.rs`: swept bullets (`a & b`, `l & c`, `l \| c`) | Catches tunnelling (a test); the sign conventions of the hull and facing tests were right the first time |
| Lines of flight | `sim/mod.rs`: evaders take every shot as the line `p & (p + v)` and step along its normal when the signed distance is small | One join and one join-with-point per shot and evader; a test fires at an evader and it survives |
| Reflection by a line (odd versor) | `collide::reflect`: a warden's shield is the line through its centre along its local y axis, and a shot's velocity is sandwiched by it | Needs a sign for directions (friction 7) |
| Motor interpolation as a follow law | serpent segments: each pose moves toward the one ahead by `interpolate(a, b, k)` with `k = 1 − exp(−30 dt)` | The chain keeps its spacing within tolerance through turns (a test) |
| Distance as the norm of a join | `body::distance`, the lattice kernel | `(p & q).norm()` |
| Maps: `cam << Point::slot()`, `proj.of(view)`, `GpuMat` | `render/scene.rs`: the camera | The view-projection is a gax map, uploaded in the WGSL layout |
| `{Kind}Gpu` Pod types | line instances (`MotorGpu`), lattice nodes and particles (`PointGpu`) | Layouts asserted at compile time; no hand-written byte layouts in the game |
| Generated WGSL modules (`gax::wgsl`) | every shader imports `gax::pga2d` through `wesl` at build time | `unit_motor_sandwich_point` places every shape segment in the vertex shader |
| Traced kernels, CPU and GPU (`Tracer::wgsl`) | the lattice (`grid_node`, `source_force`) and particles (`particle_step`) | See "CPU and GPU" |
| Batch SoA path as the CPU twin | `render/cpu_grid.rs`: `grid_node_batch`, `source_force_batch` | The oracle for the GPU (a test) |
| The deterministic mode | not used: replays promise same-build determinism | A state hash every second over all poses, velocities and the generator; runs of 75 s on three seeds replay bit for bit, also through the game loop with hit-stops and uneven frames (tests). The build id includes a hash of gax's sources, since any change in gax's arithmetic may change a run |

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
7. **Reflecting a direction by a line needs a sign.** The sandwich `l x l⁻¹` of an odd versor
   (a line) applied to a point gives the reflected point up to sign. For a proper point the sign
   is harmless, since `−p` and `p` are the same point. For a direction (a velocity), `−v` is the
   opposite velocity: the first warden shield sent shots *through* itself, mirrored across the
   shield's normal instead of back. The game negates (`collide::reflect`, with a test that
   `(1, 0.5)` mirrors to `(−1, 0.5)`). A gax `reflect` that applies the grade-dependent sign of
   the odd sandwich would remove the trap (open).
