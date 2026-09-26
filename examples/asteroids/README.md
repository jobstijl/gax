# gax asteroids

A small Asteroids game on plane-based geometric algebra, as a worked example of
[gax](../../README.md) outside the library's own workspace (the windowing crate,
[macroquad](https://macroquad.rs), is a dependency of this crate only).

```sh
cd examples/asteroids
cargo run --release
```

Arrows or A/D turn, Up or W thrusts, Space shoots, Enter restarts, Esc quits.

What gax does here (all in [`src/game.rs`](src/game.rs)):

* Every ship and rock is a PGA2D **motor** (`Unit<Motor>`), moved each step by the
  exponentials of two **twists**: `exp(dt · T(v))` translates in the world and
  `exp(dt · R(ω))` spins about the body's own origin, so `pose ← travel · pose · turn`.
  Wrapping around the screen edge is one more translation motor.
* Shapes are polygons of **points** in body coordinates, placed in the world with the batch
  sandwich kernels (`pose.transform_slice(&points, &mut placed)`, feature `batch`).
* A hit test is a sign test: a point is inside a convex polygon when it lies on the same
  side of every edge line as the polygon's center, and the side of a point `p` relative to
  the line `a & b` through two vertices is the sign of `(a & b) ^ p`.

`cargo test` checks the twist conventions, the wrapping, the polygon test, and plays a long
headless game.
