# What warp verifies about gax

The game's second purpose is to validate gax under real load. This file lists every gax feature
it uses, where, and what was found. It covers milestones M2 (the Plane vertical slice), M3 (the
Plane complete) and M4 (the Tunnel slice), and grows with the game. The **friction log** at the
end is the most valuable part: each entry is a gax issue or fix.

**The contract.** All of the game's geometry is gax, and the build enforces it. `clippy.toml`
bans glam's types and Bevy's transforms, and also `sqrt`, `sin`, `cos`, `tan`, `atan2`, `asin`,
`acos`, `sin_cos` and `hypot` on floats, everywhere:
* lengths are ideal norms and joins;
* angles come from the rotation between two directions and its logarithm;
* turns are rotation motors, and oscillations are phasors, in the game and in the audio.

The only exception is a test that checks a phasor against `sin`.

## Features used

### Motion and collision (PGA2D, the Plane)

| gax feature | where | found |
|---|---|---|
| `Unit<Motor>` poses, `exp` of twists, `renormalize_fast` every tick | `sim/body.rs`: every body, `pose ← exp(dt B) pose` | Over a 10-minute integration, `m ~m` stays within `1e-5` (a test); in play, the worst drift per session is below `4e-7` (F3 shows it) |
| Twists add: translation plus rotation | `Body::twist` | Works as intended, once the twist constructors existed (friction 1) |
| Motor interpolation, `a exp(t log(~a b))` | drawing between ticks (`Body::lerp`); serpent segments following the one ahead | Every entity, every frame; the serpent keeps its spacing through turns (a test) |
| The rotation between two directions, and its angle | `body::turn`: steering (chasers, wardens, the ship's nose), `body::angle_of` for the replays' aim | Replaces every `atan2` and angle wrap: the rotation is the relative angle, already in `[-π, π]` (friction 12 on precision near a half turn) |
| Rotation motors on directions | `body::heading`, `turned`: spawns, particle bursts, rings of motes, the barrels' offset | Every `cos`/`sin` pair became a turn |
| Walls as lines | `collide::walls`: the arena's four edges, inside on the positive side | Bounces reflect position and velocity in the wall (`Line::reflect`); the ship stops at its foot `(l \| p) ^ l` and keeps the mean of its velocity and the reflection, the motion along the wall; a shot's wall hit is the meet of its path with the wall |
| Joins and perpendiculars | `collide`: swept bullets (`a & b`, `l & c`, `l \| c`); `body::distance` | Catches tunnelling (a test) |
| Lines of flight | evaders take every shot as the line `p & (p + v)`, and step away from their foot on it | A test fires at an evader and it survives |
| Reflection in a line | wardens' shields; wall bounces | `Line::reflect`, fixed in gax (friction 7) |
| Maps: `cam << Point::slot()`, `proj.of(view)`, `GpuMat` | `render/scene.rs`: the view map; screen effects place points through it (`to_uv`) | The view-projection is a gax map, uploaded in the WGSL layout |

### Drawing and light

| gax feature | where | found |
|---|---|---|
| Points as shapes, weights as scale | `render/scene.rs`: every shape is constant PGA2D points; `(x, y, 1/s)` is the point scaled by `s` | No coordinate arithmetic to draw at a size |
| Light as homogeneous points (Grassmann) | `light.rs`: a colour is a PGA3D point in RGB space with its intensity as the weight | Adding is additive mixing (a test); fading scales the weight; whitening moves towards white; the shaders emit the first three coordinates, which are radiance |
| Rotation about a line, projection onto a line | `light::hue_shift` (the grey axis), `light::desaturate` (towards `(l \| axis) ^ axis`) | The Plane's lattice turns violet with the intensity; menus desaturate the world behind them |
| Traced kernels, CPU and GPU (`Tracer::wgsl`) | the lattice (`grid_node`, `source_force`), particles (`particle_step`), the line renderer (`segment_corner`, `segment_distance`), the lattice's heat (`edge_heat`), streaks, light mixing, whitening, fading | The shaders do no geometry by hand; each kernel is tested against its source |
| Post-processing as kernels | `post.wesl`: luminance is a light's pairing with a plane of Rec. 709 weights; AgX's inset and outset are gax maps on light points and its look moves away from the grey of the same luminance; the shock ripple pushes points along their radial direction; chromatic aberration scales about the screen's centre point; the vignette is a radial distance. `stars.wesl`: the distance to a star is the norm of a join, the twinkle a phasor | The frames match the hand-written shaders to within one level of 255 (a pixel comparison of two scenes) |
| Distance to a segment by joins | `kernels::segment_distance`, per fragment | Inside the strip between the perpendiculars at the ends (`l \| a`, `l \| b`) it is the distance to `l`, outside to the nearer end; 35 mul, 3 sqrt |
| `{Kind}Gpu` Pod types | line instances (`MotorGpu`, lights as PGA3D `PointGpu`), lattice nodes and particles (`PointGpu`) | Layouts asserted at compile time; no hand-written byte layouts in the game |
| Generated WGSL modules (`gax::wgsl`) | every shader imports `gax::pga2d` and `gax::pga3d` through `wesl` at build time | `unit_motor_sandwich_point` places every shape segment in the vertex shader |
| Batch SoA path as the CPU twin | `render/cpu_grid.rs`: `grid_node_batch`, `source_force_batch` | The oracle for the GPU (a test) |

### The Tunnel (PGA3D)

| gax feature | where | found |
|---|---|---|
| PGA3D screw motions | `tunnel/track.rs`: the track is `K_i exp((s - s_i) T_i)`, each `T_i` a sum of `Line` twists (forward translation, pitch, yaw, roll about the frame's own axes, so it multiplies on the right) | Arc length is exact along the axis and the frame is continuous across joins (a test over 2 km); a level-keeping term uses `K << up` to find the world's up in the frame |
| Placing and straightening | `Track::place` (`frame(s) >> p`), `Track::straighten` (`frame(s) << p`, corrected twice) | They invert each other (a test) |
| The axis as a line | `tunnel::axis`: the distance from it is a join, the pull back inside goes towards the foot on it, rings and wall placements are turns about it | |
| PGA3D lattice | `tunnel/lattice.rs`: rest positions are turns about the axis; displacements and velocities are directions | Ripples travel and settle (a test) |
| Joins in 3D | `tunnel::segment_hits_sphere`: the distance to a segment's line is the weight of `a & b & c` over that of `a & b`; the ends by the planes `l \| a`, `l \| b` | Right the first time (a test) |
| Reflection in a plane | mines bounce off the lane's edge (`Plane::reflect`) | |
| Meets | the ship's shadow (the light-to-ship line meets the wall's tangent plane, `wall point \| radial line`); near-plane clipping; the reticle's ray meeting the tunnel's cross-section | On the wall, ahead of the ship (a test) |
| A camera as a motor | `Motor::look_at` (fixed in gax, friction 8); `cam << p` into its frame; the projection is homogeneous (the depth is the 2D point's weight) | The level camera does not roll (a test); the cross-section fills the screen (a test) |
| Aiming through the camera, backwards | `View::aim`, `View::across`: a screen point's ray (`eye & direction`) meets the cross-section, straightened back | A screen direction moves the ship that way on screen, and a ray comes back as the point it was cast through (tests); the reticle locks the enemy under it (a test) |
| Rotation about a line | the barrel roll: `Motor::rotation(axis, θ) >> ship`, eased over 0.4 s | Keeps the radius exactly (a test) |
| PGA3D motor interpolation | `track::interpolate`: `M(s)` between keyframes, and the camera's spring | For a screw between keyframes the interpolation is the screw itself |

### Sound

| gax feature | where | found |
|---|---|---|
| Phasors | `audio/dsp.rs`: a sine is the height of a unit direction turned by a rotation motor every sample; phase modulation turns it further; the filter's `tan θ` is a phasor's height over its width; equal-power pan is `(1, 0)` turned by up to a quarter | Stays on the sine over a minute of samples (a test: `2e-3`, renormalized every 1024 samples); four one-minute offline renders take 4.6 s |
| Positional audio | Plane: `cam << p` in 2D; Tunnel: the camera motor in 3D | Pan from across, distance from the offset's ideal norm |

### Replays

| gax feature | where | found |
|---|---|---|
| The deterministic mode | not used: replays promise same-build determinism | A state hash every second over all poses, velocities and the generator; runs of 75 s on three seeds replay bit for bit, also through the game loop with hit-stops and uneven frames (tests). The build id includes a hash of gax's sources, since any change in gax's arithmetic may change a run |

## Numbers

* **CPU and GPU**, the same traced programs (`headless.rs` tests; RX 6900 XT, RADV):
  * particles after 3 steps: `2e-7`, relative;
  * the lattice after 1 step: `1.5e-7`, which is ulps;
  * the lattice after 240 steps with blasts and a well: `4.4e-4`. The rounding differences
    between the fused CPU form and the GPU's `fma` accumulate through the coupled springs, as
    expected for an iterated coupled system; they do not grow into a different shape.
* **Traced kernel costs** (and the generic code's):
  * `grid_node`: 57 mul, 57 add, 4 div, 8 calls (the same: there is nothing to fuse in a sum of
    spring forces);
  * `source_force`: 10 mul, 7 add, 1 div; `particle_step`: 10 mul, 7 add;
  * `segment_corner`: 15 mul, 16 add, 1 div, 5 calls (generic: 41 mul, 23 add, 3 div,
    12 calls; the quarter-turn motor folds away, friction 9);
  * `segment_distance`: 35 mul, 19 add, 1 div, 9 calls;
  * `edge_heat`: 13 mul, 7 add, 1 div, 7 calls;
  * the light kernels: 4 to 8 mul; `luma`: 3 mul, 2 add;
  * `agx`: 72 mul, 41 add, 1 div, 21 calls (generic: 83 mul, 54 add); `ripple`: 9 mul, 8 add,
    1 div, 3 calls.
* **GPU time per frame** at 1600×900 in the Plane: compute 0.01 ms, scene 0.13 ms, bloom
  0.08 ms, post 0.04 ms. The join-based distance per fragment and the traced tonemapper cost
  little.
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
   pitfalls; annotate `Point::<(), f64>`. It came up again in the new doctests.
4. **An "ideal norm" of a direction was missing.** `norm()` of a point is its weight, which is
   0 for a direction. **Fixed in gax:** `Point::ideal_norm` in PGA2D and PGA3D.
5. **Scalar multiplication is `gp`.** `v.gp(dt)` reads oddly next to `+`. `Mul<T>` for
   coefficients would be friendlier, and might conflict with `*` meaning the geometric product
   (open question).
6. **The WGSL modules are large.** PGA2D is 120 KB of source. `wesl`'s stripping keeps only
   what the shaders use, so it costs nothing at run time, but build scripts parse all of it.
   They also have no addition or scaling functions, so anything compound (a light's mix, a
   streak's tail) is a traced kernel, which is the better design anyway.
7. **Reflecting a direction by a line needs a sign.** In PGA2D the sandwich `l x l⁻¹` of a line
   (odd) with a point (even) comes out negated. That is harmless for a proper point but wrong
   for a direction: the first warden shield sent shots *through* itself. **Fixed in gax:**
   `pga2d::Line::reflect` applies the sign, and `pga3d::Plane::reflect` needs none (tested with
   directions).
8. **There was no motor from a frame.** The Tunnel's level camera needs "the motor whose
   forward is `f` and whose right is horizontal" (a look-at); the game composed a yaw and a
   pitch from `atan2` and `asin`. **Fixed in gax:** `pga3d::Motor::rotation_between` and
   `Motor::look_at`, and `pga2d::Motor::rotation_between` with `Motor::angle`. All are built
   from `normalize(1 + B A)` of planes (or lines) and branch-free, so they trace.
9. **The tracer did not fold calls on constants.** A motor built from a constant angle inside a
   kernel (a quarter turn) cost `sqrt`, `sin` and `cos` per run, and a shader carried
   `abs(1.0)`. **Fixed in gax-gen:** calls on constants are evaluated at trace time, through
   named constants too, and arithmetic on named constants folds. Large inexact values, like the
   nearest `f64` to `π/4`, become named constants instead of exact dyadics that overflow the
   `i128` arithmetic. The quarter turn of a direction went from 17 mul and 7 calls to 2 mul.
10. **The tracer compared constants by structure.** `Rational` derived `Ord`, which compares
    numerator then denominator, so a `select` on two constants decided `1/2 < 1/3`. **Fixed:**
    a by-value comparison for selects. The structural order stays for term ordering, since the
    generated code depends on it.
11. **Relation reduction could reach back into a stage.** With everything else folded, the
    reduction rewrote a square root's own argument with its relation `s² = x`, compiling
    `sqrt(s s)`. The verifier could not see it, because it reads temporaries as their variables.
    **Fixed:** a reduced stage argument that uses anything not yet defined is kept as written.
12. **`rotation_between` loses precision near a half turn.** `1 + B A` is small there, so the
    result carries about the square root of the rounding error. Steering does not mind; the
    replays' aim quantization did, off by one bin near 180°. The game measures the angle from
    the nearest axis direction instead (`body::angle_of`); gax documents the limit.
13. **Meets and reflections return points of any weight, even negative.** `normalized()` keeps
    the sign, so `p − q` of a reflected point and a normal one is not a direction. The game
    divides by the weight (`body::unit_weight`). A gax method for "the same point with weight 1"
    would remove the trap (open).
14. **`Real` had no `exp`.** The shaders' tonemapper and ripple need it. Writing it as
    `sinh x + cosh x` is wrong in both directions: `inf − inf = NaN` for large arguments (the
    ripple's ring would have blanked the screen whenever a shock played), and catastrophic
    cancellation for negative ones (the tonemapper's gamma on dark pixels). **Fixed in gax:**
    `Real::exp`, direct for `f32` and `f64` (gax's own `exp` in the deterministic mode), traced
    as WGSL `exp`. Its default, used by the SIMD lanes, uses the sum for `x ≥ 0` and the
    reciprocal of the sum at `−x` below, which never cancels (a test across ±120).
