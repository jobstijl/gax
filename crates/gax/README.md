# gax: geometric algebra with extensors

`gax` is a geometric algebra library for Rust in which every type can carry *open slots*.

* `Point` is a point.
* `Point<(Point,)>` is a linear map from points to points.
* `Scalar<(Twist, Twist)>` is a bilinear form on twists.

The products of the algebra work on all of them alike. So one plain generic function evaluates a
value when it is given a point, and returns the whole transformation as a map when it is given an
open slot.

```rust
use gax::pga3d::{Motor, Plane, Point};
use gax::Slots;

/// The shadow of a point on the ground, cast from a point light. Written once.
fn shadow<S: Slots>(light: Point, ground: Plane, p: Point<S>) -> Point<S> {
    (light & p) ^ ground // the line from the light through p, met with the ground
}

let light = Point::xyz(0.0, 0.0, 10.0);
let ground = Plane::from_normal([0.0, 0.0, 1.0], 0.0);
let p = Point::xyz(1.0, 2.0, 3.0);

// Given a point, it returns a point...
let s: Point = shadow(light, ground, p);

// ...and given an open slot, it returns the projection as a map, built once.
let projection: Point<(Point,)> = shadow(light, ground, Point::slot());
let [x, y, z] = projection.of(p).to_euclidean();
let [sx, sy, sz] = s.to_euclidean();
assert!((x - sx).abs() < 1e-5 && (y - sy).abs() < 1e-5 && (z - sz).abs() < 1e-5);

// Maps compose. Spin the scene about the z axis before casting the shadow:
let spin = Motor::rotation_about(0.0, 0.0, 1.0, 0.5);
let spun: Point<(Point,)> = projection.of(spin >> Point::slot());
```

## What is in the box

* **Algebras of any signature** `Cl(p,q,r)`, including degenerate metrics (plane-based PGA) and
  non-diagonal ones (the CGA null basis `eo`, `ei`), declared with the [`algebra!`] macro.
* **Standard algebras, pre-generated behind cargo features**, so using them compiles no proc macro:
  `pga2d`, `pga3d` (default), `vga2d`, `vga3d`, `sta`, `cga2d`, `cga3d`, and the larger `stap` (projective
  spacetime, `R(3,1,1)`) and `csta` (conformal spacetime, `R(4,2)`).
* **Products for values, maps and forms alike**, each with the slot bookkeeping done by the type
  system: `*`, `^`, `&`, `|`, versor transport `>>` and `<<`, contractions and commutators.
* **Binding and composition:**
  * `m.of(x)` fills the first slot, or composes a map into it;
  * `m.at::<I>()` moves a slot to the front;
  * `m.fill(x)` fills every slot of `x`'s kind;
  * `form.swap()` exchanges a form's two slots.
* **Linear algebra on maps and forms, with typed results:**
  * `inverse`, `det`, `solve`, `svd` and `trace` on maps;
  * `eigh_with(metric)` on forms, whose modes come back as values of the slot kind, such as twists;
  * pairing solves for induced maps.
* **Closed forms on values:** `inverse`, `normalized` (to a certified `Unit` versor), `exp`, `log` and
  `sqrt`, derived symbolically per type.
* **Performance:**
  * operations are generated as straight-line code with exact integer tables, and the sandwich kernels
    are simplified symbolically;
  * SIMD lanes (`f32x8`, feature `wide`) batch any kernel in struct-of-arrays form;
  * [batch kernels](https://github.com/jobstijl/gax/blob/main/docs/batch.md) (feature `batch`) run
    sandwiches, your own generic functions and traced kernels over slices, on the best SIMD level
    the CPU has, chosen at run time;
  * [build-time tracing](https://github.com/jobstijl/gax/blob/main/docs/guide.md#build-time-tracing) fuses your
    own generic functions.
* **No macros in user code** except `algebra!`. Stable Rust (MSRV 1.89), `no_std`.

## Learn more

* [The guide](https://github.com/jobstijl/gax/blob/main/docs/guide.md): extensors as a composition
  language, in plain terms.
* Examples:
  * `scene_graph`: a robot arm and a camera, composed into one map per part;
  * `rigid_body_modes`: vibration modes that come out as twists;
  * `cga_sta`: conformal and spacetime algebra;
  * `spacetime`: boosts and light cones in STAP and CSTA;
  * `ik_chain`: inverse kinematics of a robot arm with motors;
  * `batch_particles`: a particle swarm, scalar against the batch kernels;
  * a small windowed game, [`examples/asteroids`](https://github.com/jobstijl/gax/tree/main/examples/asteroids).
* The [design record](https://github.com/jobstijl/gax/blob/main/docs/design.md) and
  [performance notes](https://github.com/jobstijl/gax/blob/main/docs/performance.md).

The library is inspired by [numga](https://github.com/EelcoHoogendoorn/numga),
whose extensor model it brings to Rust's type system.
