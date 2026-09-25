# A guide to gax

This guide explains the ideas behind `gax` and how to use them. It assumes you know Rust and some
linear algebra; the geometric algebra it needs is introduced on the way. The code uses plane-based
projective geometric algebra (PGA), where planes, lines and points are the vectors, bivectors and
trivectors, and rigid motions are *motors*. The same API works in every algebra.

## 1. The idea: expressions with holes

In geometry you often write the same expression in two ways:

* about a particular point, as in "the shadow of *this* point";
* about all points at once, as in "the map that sends every point to its shadow".

The second is a function. In matrix code you would build its matrix by hand, while the first you
evaluate directly.

In `gax` both are one expression. Every type can carry *open slots*, holes in the expression that
are filled in later:

```rust
use gax::pga3d::{Line, Point};

let p: Point = Point::xyz(1.0, 2.0, 3.0); // a point
let open: Point<(Point,)> = Point::slot(); // a hole where a point goes: the identity map
let q: Point = Point::xyz(0.0, 0.0, 5.0);

let line_through_q_and_p: Line = q & p; // a value
let lines_through_q: Line<(Point,)> = q & open; // a map: point -> the line through q and it
assert_eq!(lines_through_q.of(p), line_through_q_and_p);
```

Read `Line<(Point,)>` as "a `Line` that is still waiting for a `Point`". Types with open slots are
called *extensors*, the name Hestenes and Sobczyk use for multilinear functions of multivectors:

| type | meaning |
|---|---|
| `Point` (= `Point<()>`) | a point: a value |
| `Point<(Point,)>` | a linear map from points to points |
| `Line<(Point, Point)>` | a bilinear map: two points in, a line out |
| `Scalar<(Line, Line)>` | a bilinear form on lines (twists), such as an energy |

The last parameter is the coefficient type (default `f32`): `Point<(), f64>`, or
`Point<(), f32x8>` for eight points at once (section 7).

## 2. Write a function once

Because the operators work on any slots, a plain generic function serves both purposes. It needs
only the bound `S: Slots`:

```rust
use gax::pga3d::{Plane, Point};
use gax::Slots;

fn shadow<S: Slots>(light: Point, ground: Plane, p: Point<S>) -> Point<S> {
    (light & p) ^ ground
}

let (light, ground) = (Point::xyz(0.0, 0.0, 10.0), Plane::from_normal([0.0, 0.0, 1.0], 0.0));
let s: Point = shadow(light, ground, Point::xyz(1.0, 2.0, 3.0)); // evaluate
let map: Point<(Point,)> = shadow(light, ground, Point::slot()); // or build the map
```

The output slots of an operation are the input slots, concatenated in order. So `light & p` has
the slots of `p`, and `(…) ^ ground` keeps them. The compiler checks all of this; with generic
`S`, `gax` proves that combining an `S`-slotted value with a plain value gives `S` again.

## 3. Filling slots

| call | effect |
|---|---|
| `m.of(x)` | fill the first slot with `x`; if `x` is itself a map, compose it in: its slots take the place of the filled one |
| `m.at::<I>()` | move slot `I` to the front, so `m.at::<1>().of(x)` fills the second slot |
| `form.swap()` | exchange the two slots of a two-slot extensor |
| `m.fill(x)` | fill *every* slot of `x`'s kind with `x` |

Composition is filling a slot with a map:

```rust
use gax::pga3d::{Motor, Point};

let t = Motor::translation(1.0, 0.0, 0.0);
let r = Motor::rotation_about(0.0, 0.0, 1.0, 0.3);
let move_then_turn: Point<(Point,)> = (r >> Point::slot()).of(t >> Point::slot());
let p = Point::xyz(0.0, 1.0, 0.0);
let [a, b, c] = move_then_turn.of(p).to_euclidean();
let [x, y, z] = (r >> (t >> p)).to_euclidean();
assert!((a - x).abs() < 1e-6 && (b - y).abs() < 1e-6 && (c - z).abs() < 1e-6);
```

`fill` matches numga's *equality groups*: slots that receive the same value. A sandwich built with
the motor left open has two motor slots, and filling them gives the transformed point:

```rust
use gax::pga3d::{Flector, Motor, Point};

let p: Point<(), f64> = Point::xyz(1.0, 0.0, 0.0);
let open: Flector<(Motor, Motor), f64> = (Motor::slot() * p) * Motor::slot().reverse();
let m = Motor::<(), f64>::translation(0.0, 2.0, 0.0).into_inner();
let moved = open.fill(m); // the same as m >> p, as a flector whose plane part is zero
```

## 4. Products

| operator | method | meaning in PGA |
|---|---|---|
| `a * b` | `gp` | geometric product: composes motors, reflections |
| `a ^ b` | `wedge` | outer product: *meet* (intersection) of planes and lines |
| `a & b` | `vee` | regressive product: *join* of points and lines, and the plane–point pairing |
| `a \| b` | `dot` | inner product (metric) |
| `v >> x` | `transform` | versor transport `v x ~v`: move `x` by the motor `v` |
| `v << x` | `transform_inv` | the inverse transport `~v x v` |
| | `lc`, `rc`, `scalar_product`, `commutator`, `anticommutator` | contractions, and the (anti)commutator |
| | `reverse`, `involute`, `conjugate`, `dual`, `undual` | involutions and complements |

Every product exists between every pair of kinds whose product is nonzero, and the result's kind
follows from the algebra. A product that is identically zero, such as `point ^ point` in 3D, does
not exist, and using it is a compile error with an explanation.

## 5. Versors, `Unit`, and matrices

Rigid motions are *motors*, and `m >> x` moves any object `x` by `m`: points, planes, lines, other
motors, and maps. `Unit<Motor>` certifies that `m ~m = 1`, and certified versors take cheaper paths:

* `unit.inverse()` is the reverse, with no arithmetic;
* sandwiches use formulas simplified with the unit condition (a unit motor on a point takes
  25 mul and 18 add);
* the product of units is a unit.

Get a unit with `normalized()`, `exp()`, or constructors such as `Motor::translation`.

The matrix of a motor is not a separate concept. It is the motor applied to an open slot:

```rust
use gax::pga3d::{Motor, Point};

let m = Motor::rotation_about(1.0, 0.0, 0.0, 0.7);
let matrix: Point<(Point,)> = m >> Point::slot(); // built once
let p = Point::xyz(1.0, 2.0, 3.0);
let _ = matrix.of(p); // applied as often as needed
```

To transform one point, use `m >> p`, which runs a fused formula. To transform many, build the
map once and apply it: for 1024 points that is about 25% faster, see
[performance.md](performance.md).

## 6. Solving, eigenproblems, and pairings

Maps between kinds of the same size have `inverse`, `det`, `solve`, `svd` and, for maps from a kind
to itself, `trace`. A map between kinds of different sizes has no `inverse`, and the compiler says
so. Right-hand sides keep their slots: solving against a map returns a map.

Forms (`Scalar<(A, A)>`) have `eigh_with(metric)` for the generalized eigenproblem between two
forms. The eigenvectors come back as values of the slot kind. For vibration modes of a rigid body
they are twists, each one a rotation about a point
(example `rigid_body_modes`):

```rust,ignore
let kinetic: Scalar<(Point, Point), f64> = Point::slot() & inertia;     // forms from maps: pair with an open twist
let potential: Scalar<(Point, Point), f64> = Point::slot() & stiffness;
let (values, modes) = potential.eigh_with(kinetic);                      // modes: [Point; 3], twists
```

**There is no transpose.** A transpose identifies a space with its dual by coefficient index, which is
right only for a Euclidean orthonormal basis. As in numga, pairings are explicit:

* `&` pairs a plane with a point without using the metric;
* `|` uses the metric.

An induced map is a pairing solve. It exists even when the map has no inverse, such as a central
projection:

```rust
use gax::pga3d::{Plane, Point, Scalar};

let eye = Point::<(), f64>::xyz(0.0, 0.0, 5.0);
let screen = Plane::<(), f64>::from_normal([0.0, 0.0, 1.0], 0.0);
let projection: Point<(Point,), f64> = (eye & Point::slot()) ^ screen;
let pairing: Scalar<(Plane, Point), f64> = Plane::slot() & Point::slot();
let on_planes: Plane<(Plane,), f64> = pairing.solve(Plane::slot() & projection);
// on_planes.of(l) & p == l & projection.of(p) for every plane l and point p
```

Values have closed-form methods where their type allows them:

* `inverse`;
* `normalized`;
* `norm`;
* `exp` for bivectors, returning `Unit` versors;
* `log` on `Unit` versors;
* `sqrt`.

A method is generated only for the kinds with the structure it needs. So `Multivector` has no
`inverse()`, and a degenerate pseudoscalar has none either.

## 7. Performance: three tiers, and batching

1. **Every product is straight-line code**, generated per pair of kinds. There are no loops over
   tables and no multiplications by structural zeros: IEEE rules forbid removing `0 * x`, so zeros must
   never reach the arithmetic. The same code serves values (scalar arithmetic) and maps (loops over
   the slots).
2. **Sandwiches and the value methods are simplified symbolically** when the algebra is generated.
   The generator uses common subexpressions, the unit condition and shared reciprocals, and verifies
   every simplification exactly.
3. **Your own functions can be fused at build time** (next section).

For throughput, use SIMD lanes as coefficients. With the `wide` feature, `Point<(), f32x8>` is
eight points in struct-of-arrays form, and every kernel runs on all eight at once. Transforming
1024 points this way is 2.2 times faster than a glam `Affine3A` loop. Single transforms in
array-of-structs form are slower than glam's hand-tuned SIMD; [performance.md](performance.md) says
where and why.

## 8. Build-time tracing

`gax` can run your generic function on *symbolic* coefficients at build time, simplify the result,
and emit a fused function. This pays off when the function has constants, such as a fixed plane or a
point with weight 1, and for chains of operations. The fused kernel is never slower than the
original: the tracer compares against the original's operation count and keeps the cheaper.

1. Put the kernels in their own file, generic over the coefficient type:

   ```rust,ignore
   // src/kernels.rs
   use gax::pga3d::{Motor, Plane, Point};
   use gax::{Real, Slots, Unit};

   pub fn shadow_on_floor<S: Slots, T: Real>(m: Unit<Motor<(), T>>, light: Point<(), T>, p: Point<S, T>) -> Point<S, T> {
       let floor = Plane::new(T::zero(), T::zero(), T::one(), T::zero());
       (light & (m >> p)) ^ floor
   }
   ```

2. Trace them in `build.rs` (add `gax` with feature `trace` under `[build-dependencies]`):

   ```rust,ignore
   #[path = "src/kernels.rs"]
   mod kernels;
   use gax::pga3d::{Motor, Point};
   use gax::trace::{Sym, Tracer};
   use gax::Unit;

   fn main() {
       let mut t = Tracer::new();
       t.kernel("shadow_on_floor_fused", |m: Unit<Motor<(), Sym>>, l: Point<(), Sym>, p: Point<(), Sym>| {
           kernels::shadow_on_floor(m, l, p)
       });
       t.write_out_dir("fused.rs");
       println!("cargo:rerun-if-changed=src/kernels.rs");
   }
   ```

3. Include the result:

   ```rust,ignore
   // src/lib.rs
   pub mod kernels;
   include!(concat!(env!("OUT_DIR"), "/fused.rs"));
   ```

The fused function is `shadow_on_floor_fused<T: Real>(m, light, p)`, with the same argument types.
In this example it takes 31 mul and 21 add, where the generic code takes 46 and 32.

* **Unit conditions.** `Unit` arguments bring their unit condition into the simplification.
* **Branches.** Traced code must not branch on coefficient values (`select_lt` panics during tracing).

`examples/traced` in the repository is a complete crate.

## 9. Your own algebra

```rust,ignore
gax::algebra! {
    algebra stap "Spacetime algebra with a projective dimension, R(3,1,1).";
    basis ep = 0, e0 = 1, e1 = -1, e2 = -1, e3 = -1;
    kind Scalar = [1];
    versor Vector = [ep, e0, e1, e2, e3];
    kind Bivector = [ep0, ep1, ep2, ep3, e01, e02, e03, e12, e31, e23];
}
```

* **The basis.** Basis vectors are `e` plus one character, given with their squares. `metric eo ei = -1;`
  sets an off-diagonal entry, for null bases.
* **Kinds.** A `kind` lists its blades in the order and orientation you want (`e31` is `−e13`).
  `versor` also generates fused sandwiches with that kind as the versor. `Scalar` and a full
  `Multivector` are added if you leave them out.
* **Build time.** The macro runs the generator at compile time. Add `[profile.dev.build-override]`
  with `opt-level = 3` to your `Cargo.toml`, or large algebras expand slowly.

## 10. Conventions and pitfalls

* **PGA layouts** follow the bivector.net cheat sheets:
  * a PGA3D point is `x e032 + y e013 + z e021 + w e123` (`Point::xyz`);
  * a plane is `a e1 + b e2 + c e3 + d e0`, meaning `ax + by + cz + d = 0`.
* **Signed distance:** `plane & point` is the signed distance of a unit point from a unit plane.
* **The regressive product** is `J⁻¹(J a ∧ J b)`, with `J` the right complement. Its sign for some pairs
  differs from ganja.js, and in PGA3D it differs from numga by the orientation of the pseudoscalar
  (`e0123` here); see ADR-009 in the [design record](design.md).
* **Float literals.** Type parameter defaults do not drive inference in Rust. `Point::xyz(1.0, 2.0, 3.0)`
  alone is `f64` by the literal fallback; annotate (`let p: Point = …`) for the `f32` default.
* **Degenerate metrics.** In PGA the metric pairing `|` of lines ignores the moment part, so energy
  forms built with it are singular. Build them with the regressive pairing `&`, as the modes example
  does.
