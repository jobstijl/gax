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
| `m.of(x)` | fill the first slot with `x`; if `x` is itself a map, compose it in: its slots take the place of the filled one. `x` may be of a smaller kind whose blades all lie in the slot's (a rotor in a motor slot): it is embedded |
| `m.at::<I>()` | move slot `I` to the front, so `m.at::<1>().of(x)` fills the second slot |
| `form.swap()` | exchange the two slots of a two-slot extensor |
| `m.fill(x)` | fill *every* slot of `x`'s kind with `x` |
| `m.trace_at::<I>()` | contract the output with slot `I` (of the output's kind): a scalar with the other slots |
| `t.outermorphism::<B>()` | extend a map on vectors (by `^`) or on antivectors such as PGA points (by `&`) to the kind `B` |
| `x.cast::<K>()` | the blades `x` shares with `K`, as a `K` (other blades zero): a projection, an embedding, or both; on a map, of its output |
| `x.grade::<G>()` | the grade-`G` part, as the declared kind that holds it |

Composition is filling a slot with a map:

```rust
use gax::pga3d::{Motor, Point};

let t = Motor::translation(1.0, 0.0, 0.0);
let axis = Point::xyz(1.0, 0.0, 0.0) & Point::xyz(1.0, 0.0, 1.0); // a vertical line through (1, 0, 0)
let r = Motor::rotation(axis, 0.3);
let move_then_turn: Point<(Point,)> = (r >> Point::slot()).of(t >> Point::slot());
let p = Point::xyz(0.0, 1.0, 0.0);
let [a, b, c] = move_then_turn.of(p).to_euclidean();
let [x, y, z] = (r >> (t >> p)).to_euclidean();
assert!((a - x).abs() < 1e-6 && (b - y).abs() < 1e-6 && (c - z).abs() < 1e-6);
```

Binding a smaller kind embeds it, and composing a smaller kind's slot into a larger slot narrows
the slot:

```rust
use gax::pga3d::{Flector, Line, Motor, Point, Rotor, Scalar};

let act: Flector<(Motor,), f64> = Motor::slot() * Point::xyz(1.0, 2.0, 3.0);
let r = Rotor::<(), f64>::from_coeffs([0.9, 0.1, 0.2, 0.3]);
assert_eq!(act.of(r), act.of(r.cast::<Motor>())); // a rotor fills a motor slot
let on_rotors: Flector<(Rotor,), f64> = act.of(Rotor::slot()); // the slot, narrowed
assert_eq!(on_rotors.of(r), act.of(r));
let m = Motor::<(), f64>::rotation_about(0.0, 0.0, 1.0, 0.5).into_inner();
let (s, b): (Scalar<(), f64>, Line<(), f64>) = (m.grade::<0>(), m.grade::<2>());
assert_eq!(s.cast::<Motor>() + b.cast::<Motor>() + m.grade::<4>().cast::<Motor>(), m);
```

Kinds are a closed family per algebra, so these are tables the generator emits (`Cast`,
`SubKind`, `GradePart`), not a search at run time; a cast between kinds with no blade in common,
or a grade a kind lacks, is a compile error. (numga's types are any set of blades, computed at run
time. gax rounds a result up to the smallest declared kind that holds it, so declare a kind for a
set of blades you want as a type of its own.)

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
| `a * t`, `t * a`, `a / t` | `gp` with a coefficient | scaling: the geometric product with a scalar (`v * dt`) |
| `a / b` | `div_by` | division `a b⁻¹`: `b / a` is the motion from `a` to `b`, twice (section 5) |
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
  33 mul and 21 add, against 38 and 32);
* the product of units is a unit.

Get a unit with `normalized()`, `exp()`, or constructors such as `Motor::translation`.

**Drift.** A product of units is certified without renormalizing, so a long chain drifts slightly
from `m ~m = 1`. The simplified formulas are built so that drift only scales the result
uniformly. In PGA that cancels: shapes stay rigid. Still, renormalize now and then:

* `m.renormalize_fast()` is one Newton step, with no square root;
* `a.mul_renormalized(b)` multiplies and renormalizes in one call;
* the `check-units` feature asserts, in debug runs, that certified kernels only see units.

See [numerics.md](numerics.md).

**The motor between two elements.** Two planes make a rotation about the line where they meet
(or a translation, if they are parallel), by twice the angle between them: the ratio `b / a` is
the reflection in `a` followed by the reflection in `b`. Its square root is the motion that
carries `a` onto `b`, and the same holds for lines and points:

```rust
use gax::pga3d::{Motor, Point};

let (p, q, r) = (Point::xyz(0.0, 0.0, 0.0), Point::xyz(1.0, 0.0, 0.0), Point::xyz(0.0, 1.0, 0.0));
let m = Motor::between(p & q, p & r); // the x axis onto the y axis: a quarter turn about z
let t = Motor::between(p, q);         // the translation from p to q
let [x, y, _]: [f64; 3] = (m >> q).to_euclidean();
assert!(x.abs() < 1e-12 && (y - 1.0).abs() < 1e-12);
assert_eq!((t >> p).to_euclidean(), [1.0, 0.0, 0.0]);
```

For unit elements, `(b / a).sqrt()` is the same motor, as the formula reads. `Motor::between`
takes any scale and stays precise near a half turn, where the plain square root loses the
angle to cancellation. A `Unit<Translator>` or `Unit<Rotor>` becomes a motor with `widen()`,
and any kind becomes a larger one that contains it with `From` (`Motor::from(rotor)`).

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

Maps of any shape, singular ones included, have `pinv` (the Moore–Penrose pseudo-inverse) and
`lstsq` (the least-norm least-squares solution), by one-sided Jacobi. On a map with several slots,
`lstsq` solves for the first slot against a right-hand side on the others (`at::<I>()` picks
another): the least-squares form of a pairing's `solve`. As in numga and NumPy, the norms are
those of the coefficients, and singular values below `rcond` times the largest count as zero
(`lstsq_with`, `pinv_with`; the default is machine epsilon times the larger dimension):

```rust
use gax::ApproxEq;
use gax::pga3d::{Line, Plane};

// Planes meeting the floor in a given line: 6 equations, 4 unknowns, the floor in the kernel.
let floor = Plane::<(), f64>::new(0.0, 0.0, 1.0, 0.0);
let meet: Line<(Plane,), f64> = Plane::slot() ^ floor;
let l = Plane::new(1.0, 0.0, 0.0, -2.0) ^ floor;
let x: Plane<(), f64> = meet.lstsq(l); // the least-norm one: the plane x = 2, upright
assert!((x ^ floor).approx_eq(&l, 1e-12));
let back: Plane<(Line,), f64> = meet.pinv();
```

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

A method is generated only for the kinds with the structure it needs. `inverse` has a general
form for the other kinds (CGA's and CSTA's `Even`, every `Multivector`): Shirokov's method, the
inverse as a polynomial in `x` from the scalar parts of its powers, refined by a Newton step
(up to 9 products in the kind's product closure in the standard algebras; ADR-037). A kind none of whose values is
invertible, such as a degenerate pseudoscalar, has no `inverse()`.

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

The `batch` feature does the same without choosing an instruction set at compile time.
`m.transform_slice(&points, &mut out)` and `m.transform_soa(&soa, &mut out)` run on the best
SIMD level the CPU has, detected at run time, and so do your own generic functions (`batch::map`)
and traced kernels (`name_batch`). See [batch.md](batch.md).

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
* **Constants fold.** Whatever does not depend on the inputs is computed at trace time, calls
  included: a motor built from a constant angle inside a kernel (`Motor::rotation(c, π/2)`)
  costs no `sqrt`, `sin` or `cos` at run time, only the products of applying it. Small exact
  values stay exact; others become named constants.
* **Branches.** `T::select_lt` is data flow, not a branch: it becomes a `select` in the kernel
  (and folds when both compared values are constants). Traced code must not otherwise branch
  on coefficient values. Iterative solvers (`all_lt`) run at trace time on constants, so the
  `inverse()` of a fixed matrix folds to its constants, but they cannot be traced on inputs.

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
* **Size.** Up to 9 dimensions. From 7D on the kinds are large (a 9D even versor has 256
  coefficients): sandwiches are plain `(v x) ~v` rather than simplified, the largest products
  are loops over tables, and compiling takes 30 to 60 s and 2.5 to 4.5 GB (ADR-034). The full
  even kind of a 6D to 9D algebra has its `log`, and its bivector its `exp`, in closed form
  ([log6d.md](log6d.md)).
* **Shaders.** With gax's `wgsl` feature, the algebra also has WGSL modules, `WGSL_MODULE` and
  `WGSL_MODULE_F16`, and kernels traced over its kinds link against them (section 11).

## 10. Laws you can rely on

Every rule below is proved exactly for every standard algebra, on symbolic coefficients
([laws.md](laws.md) has the full list and how).

* **Filling and composing don't depend on the order you do them in.** `f.of(g.of(h))` is
  `f.of(g).of(h)`. Binding slot 0 and then slot 1 gives the same as binding slot 1 first
  (`at::<1>`). `K::slot()` is the identity.
* **Casts are projections by blade.** `x.cast::<K>()` keeps exactly the blades `x` shares with
  `K`, with their orientations converted. From a kind into one that holds all its blades and back
  is the identity, and binding such a value is binding its embedding. `x.grade::<G>()` is `x` with
  its other grades zeroed.
* **Maps are linear.** `f.of(x + y) == f.of(x) + f.of(y)`, and scalars pull through.
* **Operations work the same on maps and values.** Combining two maps and then filling them is
  the same as filling them and then combining: `(f ^ g).of(x).of(y) == f.of(x) ^ g.of(y)`.
* **Versor chains fold.** `a >> (b >> x) == (a * b) >> x`, and the matrix of `a * b` is the
  composition of the two matrices.
* **Versors respect products, up to a known scale.** `(m >> a) * (m >> b)` equals
  `m >> (a * b)` times `‖m‖²` (the scalar part of `m ~m`). For a `Unit` versor that factor is 1.
  Products that go through the complement (`&`) can pick up a sign under reflections.
  [law-factors.md](law-factors.md) has the factor for every versor and product.
* **Moving a map** (its input and its output) by a versor is `(m >> f).of(m << K::slot())`,
  which is how you would move an inertia tensor with a body. For a `Unit` versor it satisfies
  `moved.of(m >> x) == m >> f.of(x)`.
* **`fill` is the diagonal.** `form.fill(x)` binds `x` into every slot of its kind, so it is a
  quadratic (or higher) form in `x`.

## 11. On the GPU

gax ships a WGSL module per algebra (feature `wgsl`) and matching Rust types (feature
`bytemuck`). A motor on the GPU is the same 8 numbers as on the CPU, in `ceil(N/4)` `vec4`s. The
shader functions are printed from the same verified programs as the Rust kernels:

```rust,ignore
use gax::pga3d::{Line, MotorGpu, Point};

let m = Line::<(), f32>::new(0.1, 0.2, 0.3, 1.0, 0.0, 0.0).exp();
let gpu: MotorGpu = m.into(); // 32 bytes, bytemuck::Pod
let matrix: gax::GpuMat<4> = (m >> Point::slot()).into(); // a WGSL mat4x4, applied as M * x
```

```wgsl
import gax::pga3d::{Motor, Point, unit_motor_sandwich_point};

fn place(m: Motor, p: Point) -> Point {
    return unit_motor_sandwich_point(m, p);
}
```

Register `gax::wgsl::PGA3D.source` under its path, `gax::pga3d`, with your WESL resolver, or
prepend it to a plain WGSL shader. Traced kernels get a WGSL form too, with
`Tracer::wgsl(true)`: the same kernel then runs on the CPU and the GPU.

See [shaders.md](shaders.md) for the list of functions, the layout, how everything is tested,
and GPU numerics. `examples/wgpu` puts it together: instanced motors and a traced particle
kernel, with plain wgpu.

## 12. Between algebras

The algebras are related, and gax converts between them with `From` wherever one algebra is a
part of another. Each conversion is an algebra homomorphism: it sends every basis vector to a
vector of the same square, so it keeps every product, `φ(a b) = φ(a) φ(b)`, and with it
sandwiches, meets and joins. The generator proves this exactly for every pair of kinds.

| from | to | what it is |
|---|---|---|
| `vga2d` | `vga3d`, `pga2d`, `cga2d` | the plane in space; vectors as lines through the origin; the Euclidean part of CGA |
| `vga3d` | `pga3d`, `cga3d` | vectors as planes through the origin, rotors as rotations about it |
| `pga2d` | `pga3d` | the `xy` plane in space: lines as vertical planes, points as vertical lines |
| `pga2d`, `pga3d` | `cga2d`, `cga3d` | plane-based PGA in CGA (`e0` to `-ei`): planes to dual planes, motors to motors |
| `pga3d` | `stap` | space in spacetime |
| `cga2d`, `cga3d` | `cga3d`, `csta` | the plane in space; space in spacetime |
| `vga3d` | `sta` | the spacetime split for the observer `e0`: vectors as relative vectors `eₖ e0` |

A conversion keeps the meaning that the homomorphism gives: a 2D point is the vertical line
through it, because that is what the product structure says. A unit versor converts with
`widen()`, where the conversion keeps `x ~x = 1`:

```rust
use gax::{pga2d, pga3d};
use gax::Unit;

let m2 = pga2d::Motor::<(), f64>::rotation(pga2d::Point::xy(1.0, 0.0), 0.5);
let m3: Unit<pga3d::Motor<(), f64>> = m2.widen(); // a turn about the vertical line through (1, 0)
let p = pga2d::Point::xy(0.0, 2.0);
// Move, then convert; or convert, then move: the same vertical line.
let a: pga3d::Line<(), f64> = (m2 >> p).into();
let b = m3 >> pga3d::Line::from(p);
assert!(a.c.iter().zip(b.c).all(|(x, y)| (x - y).abs() < 1e-12));
```

The impls are generic over slots, so a map converts too, and `pga3d::Line::from(pga2d::Point::slot())`
is the conversion itself as a map. Maps may also go between algebras of your choice:
`from_images` builds one from the images of its input's basis blades. A projection that drops
`z`, from PGA3D points to PGA2D points, composed with a motion of space, is one matrix:

```rust
use gax::{pga2d, pga3d};

let drop_z = pga2d::Point::<(pga3d::Point,), f64>::from_images([
    pga2d::Point::new(1.0, 0.0, 0.0), // e032 (x) to x
    pga2d::Point::new(0.0, 1.0, 0.0), // e013 (y) to y
    pga2d::Point::new(0.0, 0.0, 0.0), // e021 (z) to nothing
    pga2d::Point::new(0.0, 0.0, 1.0), // e123 (the weight) to the weight
]);
let m = pga3d::Motor::<(), f64>::translation(1.0, 2.0, 3.0);
let view: pga2d::Point<(pga3d::Point,), f64> = drop_z.of(m >> pga3d::Point::slot());
let [x, y] = view.of(pga3d::Point::xyz(0.5, 0.5, 9.0)).to_euclidean();
assert!((x - 1.5).abs() < 1e-12 && (y - 2.5).abs() < 1e-12);
```

What is not a homomorphism is spelled out as a function: a Euclidean point as a CGA round
point (`cga3d::Vector::up`, which is quadratic) and back (`down`).

## 13. Generic over the kind

The functions of section 2 are generic over the slots. They can be generic over the kind as
well: every product is a trait (`Vee` for `&`, `Wedge` for `^`, `Dot`, `Gp`, `Transform`, …)
whose output kind is a table entry, so a function can take any operand the expression accepts
and let the tables give the result. A shadow is a join with the light and a
meet with the ground, for points and lines alike:

```rust
use gax::pga3d::{Line, Plane, Point};
use gax::{Vee, Wedge};

fn shadow<X>(light: Point<(), f64>, ground: Plane<(), f64>, x: X)
    -> <<Point<(), f64> as Vee<X>>::Output as Wedge<Plane<(), f64>>>::Output
where
    Point<(), f64>: Vee<X>,
    <Point<(), f64> as Vee<X>>::Output: Wedge<Plane<(), f64>>,
{
    light.vee(x).wedge(ground)
}

let (light, ground) = (Point::xyz(0.0, 0.0, 10.0), Plane::from_normal([0.0, 0.0, 1.0], 0.0));
let pole = Point::xyz(1.0, 0.0, 0.0) & Point::xyz(1.0, 0.0, 5.0);
let p: Point<(), f64> = shadow(light, ground, Point::xyz(1.0, 2.0, 5.0)); // a point's: a point
let l: Line<(), f64> = shadow(light, ground, pole);                       // a line's: a line
let on_lines: Line<(Line,), f64> = shadow(light, ground, Line::slot());   // and maps, still
assert_eq!(on_lines.of(pole), l);
assert_eq!(p.to_euclidean(), [2.0, 4.0, 0.0]);
```

`X` covers the kind and the slots at once. The `where` clause retraces the expression, since a
generic signature states what its body needs; this pays where one formula serves several kinds,
and where the kinds are fixed, concrete types read better.

This is also how a narrower operand gives a narrower result: evaluate the expression on it.
Binding it into a map already built keeps the map's output kind, since a map's type does not
record which of its coefficients are structurally zero.

## 14. Mass properties

`pga3d::Moments` and `pga2d::Moments` compute the size, the centre of mass and the inertia of a
solid from its boundary mesh (of a region from its boundary polygon), after De Keninck, Roelfs,
Dorst and Eelbode, *Clean up your Mesh!* (2025). A triangle is a join of three points, and the
body's moments are one form on planes, `M(P, Q) = ∫ (P & x)(Q & x) dV`, summed exactly over
the cones from an apex to the boundary triangles:

```rust
use gax::pga3d::{Moments, Point};

// A unit cube as twelve outward-facing triangles.
let v: Vec<Point<(), f64>> = (0..8)
    .map(|i| Point::xyz(f64::from(i & 1), f64::from(i >> 1 & 1), f64::from(i >> 2 & 1)))
    .collect();
let faces = [[0, 2, 1], [1, 2, 3], [4, 5, 6], [5, 7, 6], [0, 1, 4], [1, 5, 4],
             [2, 6, 3], [3, 6, 7], [0, 4, 2], [2, 4, 6], [1, 3, 5], [3, 7, 5]];
let m = Moments::of_mesh(&v, &faces);
assert!((m.volume() - 1.0).abs() < 1e-12);
let (inertia, frame) = m.inertia(1.0); // a PrincipalInertia, and the motor to its axes
# let _ = (inertia, frame);
```

Because the moments are one extensor, those of parts add (`+`), and `m.moved(motor)` moves them
with the body by composing the form's slots, with no new sum over the mesh. A mesh cut by a plane
needs no cap: put the apex on the plane (`Moments::of_triangles(triangles_below, apex)`), as for
the remaining fuel in a tank. In 2D, `Moments::of_polygon` gives the area, the centroid and the
polar moment; the asteroids example uses it to break rocks into the pieces of their outlines,
with momentum conserved.

## 15. Derivatives

`gax::dual::Dual<T, N>` is a coefficient type carrying `N` derivatives along with each value
(forward-mode automatic differentiation). Every product, sandwich, map, solver, `exp` and `log`
runs on it unchanged, so the derivative of a whole geometric computation comes out exact to
rounding in one pass. `derivative`, `gradient` and `jacobian` set up the variables:

```rust
use gax::dual::{Dual, gradient};
use gax::pga3d::{Motor, Point};

type D = Dual<f64, 2>;
let c = D::constant;
// The squared distance from a moved point to a target, and its gradient in the motion's
// parameters: an angle about z, and a shift along x.
let (d2, grad) = gradient(
    |[a, tx]: [D; 2]| {
        let m = Motor::translation(tx, c(0.0), c(0.0))
            * Motor::rotation_about(c(0.0), c(0.0), c(1.0), a);
        let [x, y, z] = (m >> Point::xyz(c(1.0), c(0.0), c(0.0))).to_euclidean();
        (x - c(1.0)) * (x - c(1.0)) + (y - c(1.0)) * (y - c(1.0)) + z * z
    },
    [0.0, 0.0],
);
assert!((d2 - 1.0).abs() < 1e-12 && (grad[0] + 2.0).abs() < 1e-12 && grad[1].abs() < 1e-12);
```

Branch-free code differentiates the branch it selects (one-sided at a switch), and a derivative
that is exactly zero stays zero through a function that is singular at the value (`√x` at `0`
inside `cos √x`, as a rotation's closed form computes its angle), so `exp` of a zero bivector has
the right derivative where plain forward mode gives `NaN`. `Dual<T, N>` over SIMD lanes
differentiates a batch at once; over `gax::fp::Fp` the derivatives of polynomial kernels are
exact.

## 16. Conventions and pitfalls

* **PGA layouts** follow the bivector.net cheat sheets:
  * a PGA3D point is `x e032 + y e013 + z e021 + w e123` (`Point::xyz`);
  * a plane is `a e1 + b e2 + c e3 + d e0`, meaning `ax + by + cz + d = 0`.
* **Signed distance:** `plane & point` is the signed distance of a unit point from a unit plane.
* **The regressive product** is `J⁻¹(J a ∧ J b)`, with `J` the right complement. The join and
  the meet agree on the lines they share: the join of the origin and a point on `+x` is `+e23`,
  the meet of the planes `y = 0` and `z = 0`, and a positive rotation about it is right handed
  about `+x`. Its sign for some pairs differs from ganja.js, and in PGA3D it differs from numga
  by the orientation of the pseudoscalar (`e0123` here); see ADR-009 in the
  [design record](design.md).
* **Float literals.** Type parameter defaults do not drive inference in Rust. `Point::xyz(1.0, 2.0, 3.0)`
  alone is `f64` by the literal fallback; annotate (`let p: Point = …`) for the `f32` default.
* **Floating point.** See [numerics.md](numerics.md) for the details:
  * In `f32`, positions far from the origin lose accuracy (about 2 ulp of the distance, so
    `10⁻³` at `10⁴`). Use `f64` or a floating origin.
  * `log` of a motor with a negative scalar part is the long way round; negate it for the
    shortest motion.
  * Results can differ in the last bits between builds and between scalar and batch code. For
    the same bits everywhere (lockstep simulations, replays), use `gax::strict::Strict<f32>`
    coefficients for the parts that need them, `Strict::wrap(x)` and `Strict::unwrap(x)` at the
    boundary; the rest keeps plain `f32` with fused multiply-adds. The `deterministic` feature
    does the same for every `f32` and `f64` in the build.
* **Degenerate metrics.** In PGA the metric pairing `|` of lines ignores the moment part, so energy
  forms built with it are singular. Build them with the regressive pairing `&`, as the modes example
  does.
