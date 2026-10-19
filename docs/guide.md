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
use gax::pga3d::{Line, Point, intrinsic};

let p: Point = Point::xyz(1.0, 2.0, 3.0); // a point
let open: Point<(Point,)> = Point::slot(); // a hole where a point goes: the identity map
let q: Point = Point::xyz(0.0, 0.0, 5.0);

let line_through_q_and_p = q & p; // a value: a line
let lines_through_q = q & open; // a map: point -> the line through q and it
assert_eq!(lines_through_q.of(p), line_through_q_and_p);
let _: intrinsic::Line<(Point,)> = lines_through_q; // a join is intrinsic (§5)
```

Read `Line<(Point,)>` as "a `Line` that is still waiting for a `Point`". Types with open slots are
called *extensors*, the name Hestenes and Sobczyk use for multilinear functions of multivectors:

| type | meaning |
|---|---|
| `Point` (= `Point<()>`) | a point: a value |
| `Point<(Point,)>` | a linear map from points to points |
| `Line<(Point, Point)>` | a bilinear map: two points in, a line out |
| `Scalar<(Line, Line)>` | a bilinear form on lines (twists), such as an energy |

The last parameter is the coefficient type (default `f64`): `Point<(), f32>` for the GPU, or
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

let (light, ground) = (Point::xyz(0.0, 0.0, 10.0), Plane::with_normal(Point::direction(0.0, 0.0, 1.0), 0.0));
let s: Point = shadow(light, ground, Point::xyz(1.0, 2.0, 3.0)); // evaluate
let map: Point<(Point,)> = shadow(light, ground, Point::slot()); // or build the map
```

The output slots of an operation are the input slots, concatenated in order. So `light & p` has
the slots of `p`, and `(…) ^ ground` keeps them. The compiler checks all of this; with generic
`S`, `gax` proves that combining an `S`-slotted value with a plain value gives `S` again.

With several generic operands, one thing is not automatic: how the slot lists are bracketed.
`(a * b) * c` has the slots `Cat<Cat<A, B>, C>`, and the compiler cannot see that this is the
same list as `Cat<A, Cat<B, C>>`. If the signature brackets the other way, convert with
`gax::slots::reassoc`, which costs nothing at run time:

```rust
use gax::pga3d::Point;
use gax::{Cat, Slots};

fn triple<A: Slots, B: Slots, C: Slots>(a: Point<A>, b: Point<B>, c: Point<C>)
    -> Point<Cat<A, Cat<B, C>>>
{
    gax::slots::reassoc::<A, B, C, _>((a * b) * c)
}
```

Without `reassoc` the error is a type mismatch between the two bracketings, and rustc suggests
the bound `A: Slots<Cat<B> = A>`. Don't add it: it only holds when `B` is `()`.

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

let t = Motor::translation(Point::direction(1.0, 0.0, 0.0));
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
use gax::Intrinsic;
use gax::pga3d::{Flector, Line, Motor, Point, Rotor, Scalar, intrinsic};

// A point is intrinsic (§5), and so is its product with a motor.
let act: intrinsic::Flector<(Motor,), f64> = Motor::slot() * Point::xyz(1.0, 2.0, 3.0);
let r = Rotor::<(), f64>::from_coeffs([0.9, 0.1, 0.2, 0.3]);
assert_eq!(act.of(r), act.of(r.cast::<Motor>())); // a rotor fills a motor slot
let on_rotors: intrinsic::Flector<(Rotor,), f64> = act.of(Rotor::slot()); // the slot, narrowed
assert_eq!(on_rotors.of(r), act.of(r));
let m = Motor::<(), f64>::rotation_about(Point::direction(0.0, 0.0, 1.0), 0.5).into_inner();
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
use gax::pga3d::{Flector, Motor, Point, intrinsic};

let p: Point<(), f64> = Point::xyz(1.0, 0.0, 0.0);
let open: intrinsic::Flector<(Motor, Motor), f64> = (Motor::slot() * p) * Motor::slot().reverse();
let m = Motor::<(), f64>::translation(Point::direction(0.0, 2.0, 0.0)).into_inner();
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

Get a unit with `unit()` (`normalized()` scales the same way and returns a plain value), `exp()`,
or constructors such as `Motor::translation`.

**Drift.** A product of units is certified without renormalizing, so a long chain drifts slightly
from `m ~m = 1`. The simplified formulas are built so that drift only scales the result
uniformly. In PGA that cancels: shapes stay rigid. Still, renormalize now and then:

* `m.renormalize_fast()` is one Newton step, with no square root;
* `a.mul_renormalized(b)` multiplies and renormalizes in one call;
* the `check-units` feature asserts, in debug runs, that certified kernels only see units.

See [numerics.md](numerics.md).

**Orientation.** Every value and map carries one of two orientations as its last type
parameter, `Extrinsic` or `Intrinsic` (ADR-051). An extrinsic element is oriented by the space
around it: a plane by its positive side, a line by a turn about it. An intrinsic one is oriented
within itself: a point by its weight, a line by a direction along it. Vectors and their products
are extrinsic; PGA's points and directions are intrinsic, and so is a join of them. A product
multiplies the orientations as signs (extrinsic `+`, intrinsic `−`), `&` flips the result once
more, and `dual()` flips. So `p & q` is an intrinsic line, the meet of two planes an extrinsic one,
and `Forque` names PGA's intrinsic line, the kind of a momentum, as `Twist` names the extrinsic
one in PGA2D.

A mirror reverses what surrounds an element and keeps what lies in it, and `>>` follows: an odd
versor gives extrinsic and intrinsic elements opposite signs. With that, every versor commutes
with every product, the join included, so a reflected momentum keeps its energy and a point keeps
its side of a reflected plane, with no sign to write:

```rust
use gax::pga3d::{Plane, Point};

let mirror = Plane::<(), f64>::new(1.0, 0.0, 0.0, -0.5);
let water = Plane::new(0.0, 0.0, 1.0, 0.0);
let (duck, peer) = (Point::xyz(2.0, 1.0, 1.0), Point::xyz(2.0, 3.0, 2.0));
assert_eq!(mirror >> duck, Point::xyz(-1.0, 1.0, 1.0));
assert_eq!((mirror >> water) & (mirror >> duck), water & duck); // still above the water
assert_eq!((mirror >> duck) & (mirror >> peer), mirror >> (duck & peer));
```

The two orientations are different types: a twist and a momentum cannot be added, and a map on
twists does not take a join. Each kind is named in its default orientation at the algebra's root
(`Line`, `Point`), and in either in the modules `intrinsic` and `extrinsic`: `intrinsic::Line` is a
join, `extrinsic::Point` a meet of planes, `Forque` is `intrinsic::Line` (ADR-052). Each name has its
constructors, `intrinsic::Line::new(..)`, `Forque::slot()`, and a slot list names kinds the same way,
`Line<(intrinsic::Line,)>`. Code generic over the orientation names the kind
`oriented::Line<S, T, O>`, takes its slot from `gax::Slot`, and names a kind in another orientation
`gax::Oriented<K, O>`. Where coefficients mean the other orientation, say so with
`x.with_orientation::<Intrinsic>()`. A meet of joins (two joined lines in PGA2D) is an extrinsic
point, its sign a handedness; where it is used as a place, `p.position()` is the intrinsic point of
weight one there. A mirror's own orientation does not matter (it enters twice), so `wall.reflect(p)`
takes a joined wall too. Exponentials, logarithms and square roots are of extrinsic elements only.

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

**Angles.** The angle between two directions is `atan2` of their outer product's size (the
sine) and their inner product (the cosine), scaled alike, so nothing needs normalizing and the
angle keeps its digits near 0 and a half turn. `angle_between` is unsigned, in `[0, π]`, in
every dimension; `angle_to` is signed, counterclockwise, in the plane, where an orientation is
given; and `from_angle` (VGA2D) or `direction_from_angle` (PGA2D) is the unit direction at an
angle, the inverse of `angle_to` from the x axis:

```rust
use gax::pga2d::Point;
use gax::pga3d;

let east = Point::<(), f64>::direction(1.0, 0.0);
let d = Point::direction_from_angle(-2.0);
assert!((east.angle_to(d) + 2.0).abs() < 1e-15);
assert!((east.angle_between(d) - 2.0).abs() < 1e-15);
let (x, y) = (pga3d::Point::<(), f64>::direction(1.0, 0.0, 0.0), pga3d::Point::direction(0.0, 5.0, 0.0));
assert!((x.angle_between(y) - std::f64::consts::FRAC_PI_2).abs() < 1e-15);
```

**Constructors take gax values.** Coordinates enter through the plain constructors
(`Point::xyz`, `Point::direction`, `new`, CGA's `up`); everything with a geometric meaning takes
points, directions and lines, of either orientation. A translation is by a direction (the
difference of two points is one), a rotation about a line or, through the origin, along a
direction, a plane has a normal, and a twist (the rate of a motion) is named `Twist` in every
algebra, as `Forque`, `Even` and `Odd` are:

```rust
use gax::pga3d::{Motor, Plane, Point, Twist};
use gax::Unit;

let (p, q) = (Point::<(), f64>::xyz(1.0, 2.0, 3.0), Point::xyz(1.0, 2.0, 5.0));
let up = Point::direction(0.0, 0.0, 1.0);
let lift = Motor::translation(q - p);            // from p to q
let turn = Motor::rotation_about(up, 0.5);       // about the z axis
let floor = Plane::with_normal(up, 0.0);         // z = 0, its normal `up`
assert_eq!(lift >> p, q);
assert_eq!(floor.normal(), up);
assert_eq!((p & q).direction(), q - p);          // a join points from p to q
assert_eq!(p.distance(q), 2.0);
let rate = Twist::translation(up) + Twist::rotation(p & q, 0.5); // a screw, per second
let _ = (turn, rate.exp(), Unit::<Motor>::identity());
```

The plane has the same helpers (`Line::with_normal`, `Line::normal`, `Line::direction`,
`Line::through(p, d)` for the line through a point along a direction), and the vector algebras
their rotors: `vga3d::Rotor::rotation(axis, angle)` and `rotation_in(plane, angle)` (the same
rotation, named by the plane it turns in: `a ^ b` turns `a` towards `b`, and
`rotation_in(axis.dual(), θ)` is `rotation(axis, θ)`), `vga2d::Rotor::rotation(angle)` and
`angle()`, `Rotor::interpolate`, and `Rotor::between` for vectors (and VGA3D's planes), as
`Motor::between` for flats. `Unit::identity()` (also `Default`) is the identity of any versor
kind with a scalar part.

**Named values as constants.** The named values are functions, generic over the coefficient
type: `Point::origin()`, `Plane::infinity()` (PGA3D), `Line::infinity()` (PGA2D),
`Unit::identity()`. The symbolic coefficients of the tracer and the jets build their zero at
run time, so these cannot be `const`. For the coefficient types whose zero and one are
constants (`gax::ConstCoef`: `f32`, `f64` and the `wide` lanes), the points and flats are also
associated consts, `Point::ORIGIN`, `Plane::INFINITY` and `Line::INFINITY`, and every plain
constructor (`new`) is `const`:

```rust
use gax::pga3d::{Plane, Point};

const EYE: Point<(), f32> = Point::ORIGIN;
const FLOOR: Plane<(), f32> = Plane::new(0.0, 0.0, 1.0, 0.0);
assert_eq!(FLOOR & EYE, gax::pga3d::Scalar::new(0.0));
```

A versor's identity has no const: it would need one per versor kind, and
`Unit::new_unchecked(Motor::new(1.0, 0.0, ..))` is `const` where a `static` needs one.

**Orbits.** `m.orbit(start, n)`, on a unit versor, yields the first `n` steps of repeating it
from `start`, as the pairs `(x, m >> x)`: the edges of a regular polygon, the arms of a
reticle, the gates along a screw. Each step is one sandwich of the last.

```rust
use gax::pga2d::{Motor, Point};

let sixth = Motor::<(), f64>::rotation(Point::origin(), std::f64::consts::TAU / 6.0);
let hexagon: Vec<_> = sixth.orbit(Point::xy(1.0, 0.0), 6).collect(); // six edges
assert_eq!(hexagon[1].0, hexagon[0].1);
```

The matrix of a motor is not a separate concept. It is the motor applied to an open slot:

```rust
use gax::pga3d::{Motor, Point};

let m = Motor::rotation_about(Point::direction(1.0, 0.0, 0.0), 0.7);
let matrix: Point<(Point,)> = m >> Point::slot(); // built once
let p = Point::xyz(1.0, 2.0, 3.0);
let _ = matrix.of(p); // applied as often as needed
```

To transform one point, use `m >> p`, which runs a fused formula. To transform many, build the
map once and apply it: for 1024 points that is about 25% faster, see
[performance.md](performance.md).

## 6. Solving, eigenproblems, and pairings

Maps between kinds of the same size have `inverse`, `det`, `solve` and, for maps from a kind to
itself, `trace`. A map between kinds of different sizes has no `inverse`, and the compiler says
so. Right-hand sides keep their slots: solving against a map returns a map. The right-hand side
has the map's output kind, orientation included (§5): an inertia, from twists to momenta, solves
against a momentum, and a pairing `forques & twists` (an extrinsic scalar) against an extrinsic
reading.

Maps of any shape have their singular values: `svdvals` (descending) and `svd`, a `gax::Svd`
that adds the right singular vectors as `inputs` (values of the input kind; those of zero
singular values span the null space) and the left ones as `outputs`, with
`m.of(inputs[i]) == values[i] * outputs[i]`. A form `Scalar<(A, B)>` is the map `A <- (B,)` with
the same coefficients through `as_map`, for `svd`, `pinv` or `det`. `of_both(p, q)` fills both
slots of a two-slot map with maps at once (`M <- (A, B)` with `p: A <- (X,)`, `q: B <- (Y,)` gives
`M <- (X, Y)`): a cost form pulled back through a step's dynamics and actuation.

A form's `eigh` needs a positive definite metric. Against one that is only semidefinite, such as
PGA's point weight (which does not see ideal points), `eigh_semidefinite` lists the finite
eigenvalues first and the directions the metric cannot measure as `+∞`: fitting a point to
samples is the smallest finite mode of the misfit against the weight.

Maps from a kind to itself also have eigenvalues. `eigh` takes the coefficient matrix as
symmetric (numga's `eigh` on a map: right for maps self-adjoint under the coefficient inner
product, such as stretches of Euclidean vectors); `eigvals` and `eig` take it as it is, so the
eigenvalues come out complex, as `gax::Complex<T>`, and `eig`'s eigenvectors are values with
complex coefficients. `Complex<f64>` is a coefficient type like any other, so complex
multivectors and maps use the same products; `map_coefs` converts a value or map from one
coefficient type to another:

```rust
use gax::Complex;
use gax::pga3d::{Motor, Plane, Point};

let r = Motor::<(), f64>::rotation_about(Point::direction(0.0, 0.0, 1.0), 0.7); // a rotation by 0.7 about z
let m = r >> Plane::slot();
let (values, vectors) = m.eig(); // e^{-0.7i}, 1, 1, e^{0.7i}
let complex = m.map_coefs(Complex::real);
for (l, v) in values.iter().zip(vectors) {
    let image = complex.of(v); // m v = λ v, with complex coefficients
    assert!(image.c.iter().zip(v.c).all(|(a, b)| (*a - *l * b).abs() < 1e-12));
}
```

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

When the unknown itself has two slots (a dyad, a covariance, a second moment), `lstsq_dyad`
solves for both of the map's first two slots at once, as numga's `lstsq` does when it unbinds
two slots, and `of_dyad` binds such a value back in: for `m: K <- (A, B, R…)` the result is the
least-norm `x: A<(B,)>` with `m.of_dyad(x) ≈ rhs`. The examples `geometry_pose_diffusion`
(a Lyapunov equation) and `mechanics_inertia` (a second moment from its inertia) use it:

```rust
use gax::ApproxEq;
use gax::pga3d::{Line, Point, intrinsic};

// A dyad on points whose joins sum to a given line: the least-norm one is antisymmetric.
let (p, q) = (Point::<(), f64>::xyz(1.0, 0.0, 0.0), Point::xyz(0.0, 2.0, 1.0));
let join: intrinsic::Line<(Point, Point), f64> = Point::slot() & Point::slot();
let x: Point<(Point,), f64> = join.lstsq_dyad(p & q);
assert!(join.of_dyad(x).approx_eq(&(p & q), 1e-12));
```

Forms (`Scalar<(A, A)>`) have `eigh(metric)` for the generalized eigenproblem between two
forms. A form's eigenproblem always names its metric, because its eigenvalues against the
identity depend on the basis; where the coefficient basis is orthonormal (Euclidean vectors,
rotors), `form.eigh(Scalar::identity())` is the plain symmetric eigenproblem of its coefficients.
The eigenvectors come back as values of the slot kind. For vibration modes of a rigid body
they are twists, each one a rotation about a point
(example `rigid_body_modes`):

```rust,ignore
let kinetic: Scalar<(Point, Point), f64> = Point::slot() & inertia;     // forms from maps: pair with an open twist
let potential: Scalar<(Point, Point), f64> = Point::slot() & stiffness;
let (values, modes) = potential.eigh(kinetic);                           // modes: [Point; 3], twists
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
let screen = Plane::<(), f64>::with_normal(Point::direction(0.0, 0.0, 1.0), 0.0);
let projection: Point<(Point,), f64> = (eye & Point::slot()) ^ screen;
let pairing: Scalar<(Plane, Point), f64> = Plane::slot() & Point::slot();
let on_planes: Plane<(Plane,), f64> = pairing.solve(Plane::slot() & projection);
// on_planes.of(l) & p == l & projection.of(p) for every plane l and point p
```

**When a solve fails.** The plain methods do not check. A singular dense map's `inverse` and
`solve`, and a singular pairing's `solve`, return non-finite coefficients (NaN or infinities),
and a form's `eigh` returns NaN against a metric that is not positive definite. That keeps
them branch free, so they run unchanged on SIMD lanes and in
traced kernels. The field maps' `solve` and `solve_direct` panic instead, with the reason,
because a field is too large to inspect for NaN afterwards.

Each solve has a checked twin that returns an `Option`:

* `try_inverse()` and `try_solve(rhs)` on square maps;
* `try_solve(rhs)` on pairings;
* `try_solve(b)` (conjugate gradients) and `try_solve_direct(b)` (sparse LU) on field maps;
* `try_inverse()` on values (below).

The twin runs the same factorization and returns `None` if the map is singular or the result is
not finite. Singular means one test everywhere: a pivot of the LU factorization at most
`n ε max|aᵢⱼ|` in magnitude, with `n` the number of unknowns and `aᵢⱼ` the coefficients. That
is NumPy's `matrix_rank` tolerance, applied to the pivots instead of the singular values.
The sparse `factor` reports `Err(Singular)` only for a column with no nonzero pivot left, with
no relative tolerance, since shift-and-invert eigensolvers factor nearly singular maps on
purpose; `try_solve_direct` adds the finiteness test. Conjugate gradients fails when its residual
stays above `√ε`. On SIMD lanes, `None` means that some lane failed.

```rust
use gax::pga3d::{Plane, Point};

let eye = Point::<(), f64>::xyz(0.0, 0.0, 5.0);
let floor = Plane::<(), f64>::new(0.0, 0.0, 1.0, 0.0);
let projection: Point<(Point,), f64> = (eye & Point::slot()) ^ floor;
assert_eq!(projection.try_inverse(), None); // it sends the eye to zero
let p = Point::xyz(1.0, 2.0, 3.0);
assert_eq!(projection.try_solve(projection.of(p)), None); // the ray from the eye shares the shadow
```

Values have closed-form methods where their type allows them:

* `inverse`;
* `normalized`;
* `norm`;
* `exp` for bivectors, returning `Unit` versors;
* `log` on `Unit` versors;
* `sqrt`.

A method is generated only for the kinds with the structure it needs. `inverse` has a general
form for the other kinds (CSTA's `Even`, every `Multivector`): Shirokov's method, the
inverse as a polynomial in `x` from the scalar parts of its powers, refined by a Newton step
(up to 9 products in the kind's product closure in the standard algebras; ADR-037). A kind none of whose values is
invertible, such as a degenerate pseudoscalar, has no `inverse()`.

A value's `inverse()` does not check either, and its twin `try_inverse()` (also a method of
`gax::Inverse`) returns `None` where there is none to working precision. The closed form
`~x (x ~x)⁻¹` divides by the Study number `N = x ~x`, so the test is on `N`, scaled like the
LU test: `None` where `N` is at most `n ε ‖x‖²` in size (its own rounding, `n` the number of
coefficients and `‖x‖²` the sum of their squares), or where its norm `N conj(N)` cancels to at
most `n ε |N|²` (a zero divisor such as `1 + E` with `E² = 1`). The general inverse has no
Study number; its twin tests the residual `x x⁻¹ − 1` against `√ε`. Like a map's test, this one
is relative: a plane whose coefficients are all `10⁻¹⁷` is a plane, while one whose normal
rounding left at `10⁻¹⁷` next to an offset of 1 is the plane at infinity.

```rust
use gax::pga3d::Plane;

let wall = Plane::<(), f64>::new(1.0, 0.0, 0.0, -2.0);
assert_eq!(wall.try_inverse(), Some(wall.inverse()));
let horizon = Plane::<(), f64>::new(1e-17, 0.0, 0.0, 1.0);
assert!(horizon.inverse().e1() > 1e16); // finite, and meaningless
assert_eq!(horizon.try_inverse(), None);
```

The maps above couple the coefficients of one value. A map that couples the values at many
sites, such as the beads of a string or the vertices of a mesh, is a *field map*: `gax::field`
stores it sparsely, as typed cells from site to site, and solves the whole coupled system at once.
When the number of sites is fixed, `gax::sites` makes the field a kind, `Sites<K, N>`, and a field
map an ordinary map whose slot takes exactly `N` sites. See [fields.md](fields.md).

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
  sets an off-diagonal entry, for null bases. `basis;` with no vectors declares Cl(0,0), the
  real numbers.
* **Kinds.** A `kind` lists its blades in the order and orientation you want (`e31` is `−e13`).
  `versor` also generates fused sandwiches with that kind as the versor. A `part` is a kind
  that operations never take as their result type: it is reached only by name, through casts,
  constructors and slots, so declaring one changes no existing type (PGA2D's translators log to a
  `Point`, though their support also fits the `Direction` part). `Scalar` and a full
  `Multivector` are added if you leave them out.
* **Parts in the standard algebras.** PGA2D and PGA3D have `Direction`, the ideal points
  (`point.cast::<Direction>()` drops the weight, `Direction::slot()` is a map that takes only
  directions); VGA3D has `Paravector` (`1 + r`, the Pauli algebra's states). STA's `Phasor`
  (`a + b e0123`, a complex number commuting with the even subalgebra) is a full kind instead:
  phasors multiply, invert and sandwich to phasors, the pseudoscalar's `exp` lands in one, and
  so does the anticommutator of two bivectors (a field's invariants, `F∘F = E² − B² + 2 E·B I`).
* **Build time.** The macro runs the generator at compile time. Add `[profile.dev.build-override]`
  and `[profile.release.build-override]` with `opt-level = 3` to your `Cargo.toml` (both
  profiles build proc macros unoptimized by default), or large algebras expand slowly.
* **Size.** Up to 9 dimensions. From 7D on the kinds are large (a 9D even versor has 256
  coefficients): sandwiches are plain `(v x) ~v` rather than simplified, the largest products
  are loops over tables, and compiling takes 30 to 60 s and 2.5 to 4.5 GB (ADR-034). The full
  even kind of a 6D to 9D algebra has its `log`, and its bivector its `exp`, in closed form
  ([log6d.md](log6d.md)).
* **Outermorphisms, up to a size.** A kind holding every vector (or every antivector) gets
  `t.outermorphism::<B>()` to each kind `B` holding every blade of one grade, whose matrix is
  derived as minors: for blades of `k` factors in `n` dimensions, `C(n, k)²` minors of `k!`
  terms. It is generated up to 16 384 terms: every extension up to 6 dimensions; in 7, those to
  2, 3 and 7 factors (bivectors, trivectors and the determinant, for a map on vectors); in 8
  and 9, to 2. Larger ones are left out (the 4-vectors of an 8D algebra, 117 600 terms, take
  gigabytes to compile; ADR-034). Build such a map from its images instead: the image of a
  basis blade `e_a ∧ e_b ∧ …` of `B` is `t.of(e_a) ^ t.of(e_b) ^ …`, cast to `B`, and
  `B::<(B,), f64>::from_images` takes them in `B`'s layout order.
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
  [law-factors.md](law-factors.md) has the factor for every versor and product.
* **Versors respect every product, the join included.** The factor is `‖m‖²` for every versor
  and product, because `>>` takes its sign from the orientation (§5): the join of reflected
  points is the reflected join, a reflected momentum `p & v` keeps its energy, and the side test
  `plane & point` survives a mirror. This is Dorst's rule (*Projective duality encodes
  complementary orientations in geometric algebras*, 2024, §6.3 and §8), checked in
  `tests/reflections.rs`.
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
// Move, then convert; or convert, then move: the same vertical line, a join (intrinsic) as the
// point is.
let a: pga3d::intrinsic::Line<(), f64> = (m2 >> p).into();
let b = m3 >> pga3d::intrinsic::Line::from(p);
assert!(a.c.iter().zip(b.c).all(|(x, y)| (x - y).abs() < 1e-12));
```

The impls are generic over slots, so a map converts too, and `pga3d::intrinsic::Line::from(pga2d::Point::slot())`
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
let m = pga3d::Motor::<(), f64>::translation(pga3d::Point::direction(1.0, 2.0, 3.0));
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
use gax::pga3d::{Line, Plane, Point, intrinsic};
use gax::{Intrinsic, Vee, Wedge};

fn shadow<X>(light: Point<(), f64>, ground: Plane<(), f64>, x: X)
    -> <<Point<(), f64> as Vee<X>>::Output as Wedge<Plane<(), f64>>>::Output
where
    Point<(), f64>: Vee<X>,
    <Point<(), f64> as Vee<X>>::Output: Wedge<Plane<(), f64>>,
{
    light.vee(x).wedge(ground)
}

let (light, ground) = (Point::xyz(0.0, 0.0, 10.0), Plane::with_normal(Point::direction(0.0, 0.0, 1.0), 0.0));
let pole = Point::xyz(1.0, 0.0, 0.0) & Point::xyz(1.0, 0.0, 5.0);
let p: Point<(), f64> = shadow(light, ground, Point::xyz(1.0, 2.0, 5.0)); // a point's: a point
// The pole is a join, so intrinsic, and so is its shadow; the map takes intrinsic lines.
let l: intrinsic::Line<(), f64> = shadow(light, ground, pole);           // a line's: a line
let on_lines = shadow(light, ground, intrinsic::Line::<(), f64>::slot()); // and maps, still
assert_eq!(on_lines.of(pole), l);
assert_eq!(p.to_euclidean(), [2.0, 4.0, 0.0]);
```

`X` covers the kind and the slots at once. The `where` clause retraces the expression, since a
generic signature states what its body needs; this pays where one formula serves several kinds,
and where the kinds are fixed, concrete types read better.

This is also how a narrower operand gives a narrower result: evaluate the expression on it.
Binding it into a map already built keeps the map's output kind, since a map's type does not
record which of its coefficients are structurally zero.

Generic over the *algebra* is the next step: one function for the plane and for space, or for
rotations in any dimension. `gax::motions::Motions` names the kinds that play each role in
rigid motion, a marker type per algebra: rigid motions in `Pga2d`, `Pga3d`, `Cga2d` and
`Cga3d`, rotations in `Vga2d` and `Vga3d`, the Lorentz group in `Sta` and the Poincaré group in
`Stap` and `Csta` (a spacetime's motions, with an indefinite metric). The roles are the `Motor`
(a unit even versor), its Lie algebra the `Twist`, their dual the `Forque`, the `Point`, and
the maps `Inertia`, `Mobility` and `TwistMap`. Each algebra module names the roles as types
too, `Twist`, `Forque`, `Even` and `Odd` (`vga3d::Twist` is the `Bivector`, `vga3d::Forque` the
intrinsic `Vector`, `pga3d::Even` the `Motor`). The operations come with them, as operators on
the associated types (composition, `>>` and `<<`, `+`, `-`, scaling, `.of`) which a generic
function uses without writing a bound, and as the trait's functions (`exp`, `log`,
`commutator`, `pair`, `point_inertia`, `mobility`, `point`, …). `Linear` gives `zero()`:

```rust
use gax::motions::{Linear, Motions, Pga2d, Pga3d};
use gax::Of;

/// Kinetic energy of unit point masses moving with `rate`, in any algebra.
fn energy<G: Motions<f64>>(points: &[G::Coords], rate: G::Twist) -> f64 {
    let mut inertia = G::Inertia::zero();
    for c in points {
        inertia += G::point_inertia(G::point(*c));
    }
    G::pair(inertia.of(rate), rate) * 0.5
}

let spin2 = gax::pga2d::Twist::<(), f64>::from_coeffs([0.0, 0.0, 1.0]); // a turn about the origin
let spin3 = gax::pga3d::Line::<(), f64>::new(0.0, 0.0, 1.0, 0.0, 0.0, 0.0); // about z
let e2 = energy::<Pga2d>(&[[1.0, 0.0], [0.0, 2.0]], spin2);
let e3 = energy::<Pga3d>(&[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], spin3);
assert!((e2 - e3).abs() < 1e-12);
```

In the conformal algebras a translation carries a forque into parts that pair with no twist
(a CGA3D forque is a trivector up to those), so their mobility inverts the inertia on the
forques proper. One law holds for every implementation, and `tests/motions.rs` checks it: an
inertia pairs a twist `b` to the same multiple of `⟨v, v⟩` (the metric's, `v` the points'
velocities under `exp(t b)`), twice the kinetic energy. An algebra declared with
`gax::algebra!` joins in with `gax::motions!`, which implements the trait from the module's
kinds (`commutative struct` where rotations commute, and an optional `mobility`). The numga
examples' Lie-group steppers are written once this way (`shared/mechanics_lie.rs`), and the
tennis racket runs them on VGA2D and VGA3D and on its own VGA4D and VGA5D.

Smaller roles have a trait of their own. `gax::IdealNorm` is the length of a direction (a
point at infinity: a velocity, an offset) in both PGAs, so code on directions is written once
for the plane and for space: `ideal_norm()` and `ideal_norm_squared()`, `with_ideal_norm(len)`
and `clamp_ideal_norm(max)`, branch free (a zero direction stays zero):

```rust
use gax::{IdealNorm, pga2d, pga3d};

/// An inverse-square pull along the offset `d` to its source, with a softened core.
fn pull<D: IdealNorm<Coef = f64>>(d: D, strength: f64, soft: f64) -> D {
    let r2 = d.ideal_norm_squared() + soft;
    d.gp(strength / (r2 * r2.sqrt()))
}
let flat = pull(pga2d::Point::direction(3.0, 4.0), 125.0, 0.0);
let deep = pull(pga3d::Point::direction(0.0, 3.0, 4.0), 125.0, 0.0);
assert_eq!(flat, pga2d::Point::direction(3.0, 4.0));
assert_eq!(deep.ideal_norm(), 5.0);
```

## 14. Mass properties

`pga3d::Moments` and `pga2d::Moments` compute the size, the centre of mass and the inertia of a
solid from its boundary mesh (of a region from its boundary polygon), after De Keninck, Roelfs,
Dorst and Eelbode, *Clean up your Mesh!* (2025). A triangle is a join of three points, and the
body's moments are one form on planes, `M(P, Q) = ∫ (P & x)(Q & x) dV`, summed exactly over
the cones from an apex to the boundary triangles:

```rust
use gax::mesh::Mesh;
use gax::pga3d::{Moments, Point};

// A unit cube as twelve outward-facing triangles.
let v = (0..8).map(|i| Point::<(), f64>::xyz(f64::from(i & 1), f64::from(i >> 1 & 1), f64::from(i >> 2 & 1)));
let faces = [[0, 2, 1], [1, 2, 3], [4, 5, 6], [5, 7, 6], [0, 1, 4], [1, 5, 4],
             [2, 6, 3], [3, 6, 7], [0, 4, 2], [2, 4, 6], [1, 3, 5], [3, 7, 5]];
let cube = Mesh::new(v.collect(), &faces).unwrap();
let m = Moments::of_mesh(&cube); // or Moments::of_faces(&vertices, &faces) from slices
assert!((m.volume() - 1.0).abs() < 1e-12);
let (inertia, frame) = m.inertia(1.0); // a PrincipalInertia, and the motor to its axes
# let _ = (inertia, frame);
```

Because the moments are one extensor, those of parts add (`+`), and `motor >> m` moves them
with the body by composing the form's slots, with no new sum over the mesh. A mesh cut by a plane
needs no cap: put the apex on the plane (`Moments::of_triangles(triangles_below, apex)`), as for
the remaining fuel in a tank. In 2D, `Moments::of_polygon` gives the area, the centroid and the
polar moment; the asteroids example uses it to break rocks into the pieces of their outlines,
with momentum conserved.

The inertia comes back as a `PrincipalInertia`: the map `Line <- Line` from twists to momenta in
the body's principal frame, stored as four numbers (the mass and three moments) rather than 36.
It is a map like the dense ones. It applies, solves and inverts by its four numbers, binds into
dense maps and forms, and composes either way round; every other operation goes through its
dense form, `to_dense()`. Code written against `gax::SquareLinearMap` takes either
representation and uses the fast path where there is one (ADR-044):

```rust
use gax::pga3d::{Forque, Line, PrincipalInertia};
use gax::SquareLinearMap;

/// The twist a momentum gives the body, for any representation of its inertia: from momenta
/// (intrinsic lines, `Forque`) to twists (extrinsic, `Line`).
fn twist<I: SquareLinearMap<Kind = Forque, Input = Line, Coef = f64>>(i: I, f: Forque<(), f64>) -> Line<(), f64> {
    i.solve(f)
}

let body = PrincipalInertia::new(2.0, [0.5, 0.75, 1.0]);
let f = Forque::from_coeffs([1.0, 0.0, -1.0, 0.5, 0.25, 2.0]);
let fast = twist(body, f); // four reciprocals and six products
let dense = twist(body.to_dense(), f); // LU on the 6 × 6
assert!((fast - dense).c.iter().all(|x| x.abs() < 1e-15));

// It binds into dense maps and forms as they are, with no `to_dense()`:
let energy = Line::slot() & body.of(Line::slot()); // the kinetic energy form, Scalar <- (Line, Line)
assert!((energy.of(fast).of(fast).s() - (fast & body.of(fast)).s()).abs() < 1e-15);
```

Products take its dense form: `Line::slot() & body` does not compile, and
`Line::slot() & body.to_dense()` does. A general inertia, about axes that are not principal, is
a dense `Line<(Line,)>`, and that is the type `motions::Motions::Inertia` keeps.

## 15. Derivatives

The coefficients of a gax value can carry derivatives along with their values
(forward-mode automatic differentiation). Every product, sandwich, map, solver, `exp` and `log`
runs on such coefficients unchanged, so the derivative of a whole geometric computation comes
out exact to rounding, in one pass.

The derivative with respect to a gax value is itself an extensor, as in numga:

* `gax::jet::jacobian(f, x)` of `f: A -> B` returns `f(x)` and the map `B <- (A,)`. Binding a
  small step gives the first-order change: `f(x + dx) ≈ f(x) + df.of(dx)`. A function that
  returns a map `B <- (S,)` gets `A`'s slot after its own: `B <- (S, A)`.
* `gradient(f, x)` of a scalar function returns the linear form `Scalar <- (A,)`, not an element
  of `A`: it sends a step to the change of `f`.
* `hessian(f, x)` returns the value, the gradient and the second derivative, the symmetric form
  `Scalar <- (A, A)`.

They are ordinary maps and forms, so `of`, `solve`, `lstsq` and `eigh` apply to them. A
Newton step is `x - hessian.solve(gradient)`. Here it recovers a rigid motion from where it
took three points, iterating on the motion's twist `b` (the motion is `exp(b)`):

```rust
use gax::jet::hessian;
use gax::pga3d::{Line, Point};
use gax::{ApproxEq, Real};

/// The squared distances from the moved points to their targets, summed. It is generic over the
/// coefficient type, so the same function gives the value, the gradient and the Hessian.
fn misfit<T: Real>(b: Line<(), T>, pairs: &[(Point<(), f64>, Point<(), f64>)]) -> T {
    let m = b.exp();
    let mut sum = T::zero();
    for (p, q) in pairs {
        let [x, y, z] = (m >> p.map_coefs(T::from_f64)).to_euclidean();
        let [u, v, w] = q.to_euclidean().map(T::from_f64);
        sum = sum + (x - u) * (x - u) + (y - v) * (y - v) + (z - w) * (z - w);
    }
    sum
}

let truth = Line::<(), f64>::new(0.1, -0.2, 0.3, 0.5, 0.0, -0.4);
let pairs = [Point::xyz(0.0, 0.0, 0.0), Point::xyz(1.0, 0.0, 0.0), Point::xyz(0.0, 2.0, 0.0)]
    .map(|p| (p, truth.exp() >> p));

let mut b = Line::<(), f64>::zero();
for _ in 0..8 {
    // gradient: Scalar <- Line; curvature: Scalar <- (Line, Line)
    let (_, gradient, curvature) = hessian(|b| misfit(b, &pairs), b);
    b = b - curvature.solve(gradient); // the Newton step, a twist
}
assert!(b.approx_eq(&truth, 1e-12));
```

On a quadratic energy one Newton step lands on the minimizer. The value `f` sees has jets as its
coefficients (`gax::jet::Jet<T, A>`: a value and one derivative per coefficient of `A`). For the
Hessian they are jets of jets. A function written over `T: Real` accepts both; constants enter
with `map_coefs(T::from_f64)`.

For derivatives along loose parameters rather than a gax value, `Dual<T, N>` carries `N` of them,
and `derivative`, `gradient_array` and `jacobian_array` set them up:

```rust
use gax::jet::{Dual, gradient_array};
use gax::pga3d::{Motor, Point};

type D = Dual<f64, 2>;
let c = D::constant;
// The squared distance from a moved point to a target, and its gradient in the motion's
// parameters: an angle about z, and a shift along x.
let (d2, grad) = gradient_array(
    |[a, tx]: [D; 2]| {
        let m = Motor::translation(Point::direction(tx, c(0.0), c(0.0)))
            * Motor::rotation_about(Point::direction(c(0.0), c(0.0), c(1.0)), a);
        let [x, y, z] = (m >> Point::xyz(c(1.0), c(0.0), c(0.0))).to_euclidean();
        (x - c(1.0)) * (x - c(1.0)) + (y - c(1.0)) * (y - c(1.0)) + z * z
    },
    [0.0, 0.0],
);
assert!((d2 - 1.0).abs() < 1e-12 && (grad[0] + 2.0).abs() < 1e-12 && grad[1].abs() < 1e-12);
```

Branch-free code differentiates the branch it selects (one-sided at a switch). A derivative that
is exactly zero stays zero through a function that is singular at the value (`√x` at `0` inside
`cos √x`, as a rotation's closed form computes its angle), so `exp` of a zero bivector has the
right derivative where plain forward mode gives `NaN`. Jets over SIMD lanes differentiate a batch
at once; over `gax::fp::Fp` the derivatives of polynomial kernels are exact. The cost is about
`1 + N` times the arithmetic for `N` directions, and `(1 + N)²` for the Hessian (49 numbers per
coefficient for a twist).

**Derivatives as elements.** The gradient is a form because turning it into an element of `A`
takes the metric. Where the metric allows it, `gax::calculus` does that: `nabla` is the
derivative with respect to a multivector, `∂_X F = Σ e^J ∂_J F` (for a vector the `∇` of geometric
calculus), with `divergence`, `curl` and `laplacian` beside it. A kind holding PGA's null `e0`
has no such derivative. See [calculus.md](calculus.md).

## 16. Other math libraries

Engines and physics libraries speak quaternions and matrices. gax converts to and from plain
arrays, so it depends on none of them: quaternions are `[x, y, z, w]` (glam's `to_array`,
nalgebra's `coords`), translations `[x, y, z]`, and matrices column-major `[[T; 4]; 4]` (glam's
`from_cols_array_2d`, nalgebra's storage, WGSL's `mat4x4`):

```rust
use gax::pga3d::{Motor, Point, Rotor};

let h = std::f64::consts::FRAC_PI_4;
let q = [0.0, 0.0, h.sin(), h.cos()]; // a quarter turn about z
let m = Motor::from_rotation_translation(q, [1.0, 2.0, 3.0]); // rotate, then translate
let [x, y, _] = (m >> Point::xyz(1.0, 0.0, 0.0)).to_euclidean();
assert!((x - 1.0).abs() < 1e-12 && (y - 3.0).abs() < 1e-12);

let (q2, t) = m.to_rotation_translation(); // back (q up to its sign)
let cols = m.to_matrix(); // cols[3] is the translation
assert_eq!(cols[3][..3], t[..]);
let again = Motor::from_matrix(cols); // a rigid matrix back to a motor
let r = Rotor::from_quaternion(q2); // only the rotation
# let _ = (again, r);
```

With glam, `Motor::from_rotation_translation(quat.to_array(), t.to_array())` and
`Mat4::from_cols_array_2d(&m.to_matrix())`; with nalgebra, `UnitQuaternion` coordinates and
`Isometry3::to_homogeneous`. PGA2D has `Motor::to_matrix` and `Motor::from_matrix` for the
3×3 homogeneous matrices of the plane. The conventions are checked against glam and nalgebra
themselves (`crates/gax-bench/tests/interop.rs`).

## 17. Conventions and pitfalls

* **PGA layouts** follow the bivector.net cheat sheets:
  * a PGA3D point is `x e032 + y e013 + z e021 + w e123` (`Point::xyz`);
  * a plane is `a e1 + b e2 + c e3 + d e0`, meaning `ax + by + cz + d = 0`.
* **Signed distance:** `plane & point` is the signed distance of a unit point from a unit plane.
* **Directions and normals.** A plane's `normal()` points to the side where `plane & x` is
  positive; for a join `p & q & r` that is the side from which the points turn
  counterclockwise. A line's `direction()` is `q - p` for the join `p & q` of points of weight
  one, and `a × b` (of the normals) for the meet `a ^ b`; in PGA2D a line's normal points to the
  left of its direction. Rotations are right-handed about their axis (counterclockwise seen
  from where it points), counterclockwise in the plane.
* **The regressive product** is `J⁻¹(J a ∧ J b)`, with `J` the right complement. The join and
  the meet agree on the lines they share: the join of the origin and a point on `+x` is `+e23`,
  the meet of the planes `y = 0` and `z = 0`, and a positive rotation about it is right handed
  about `+x`. Its sign for some pairs differs from ganja.js, and in PGA3D it differs from numga
  by the orientation of the pseudoscalar (`e0123` here); see ADR-009 in the
  [design record](design.md).
* **Two duals.** `dual()` is the metric-free complement (it works in PGA, where the metric is
  degenerate); `hodge()` is `~x I` with the metric, numga's `dual`. In STA and other algebras
  with negative squares they differ in sign on some blades: physics written with the Hodge dual
  (Maxwell, constitutive relations) needs `hodge()`.
* **Coming from numga.** Since numga 93c2ae3 (October 2026), its PGA3D has gax's basis: `w`
  first, the blades `wx wy wz wzy wxz wyx xyz wxyz` (gax's `e01 e02 e03 e032 e013 e021 e123
  e0123`). Joins, meets and `dual` agree in sign: the origin joined with a point on `+x` is
  `+yz`, and the dual of the origin is `−w` in both. Before that commit, numga ordered the basis
  `x y z w`, so `x∧y∧z∧w = −e0123`, and its PGA3D points, `&` and `dual` were the negatives of
  gax's. `examples/numga` ports numga's examples from before the change and notes those sign
  differences where they arise. PGA2D always agreed.
* **Clippy and the operators.** `clippy::eq_op` reads `a | a` and `b ^ b` as bitwise operations
  on equal operands; write `a.dot(a)`, `a.norm_squared()` or `b.wedge(b)`. And `>>` binds looser
  than `*` but clippy's `precedence` lint asks for parentheses in `a * b >> c`: write `(a * b) >> c`.
* **Orientation in messages.** A mismatch reads ``expected `Line<…, Extrinsic>`, found
  `Line<…, Intrinsic>` ``: a join where a turn was expected (§5). `K::from(x)` converts into `K`'s
  default orientation; `x.cast::<K>()` and `x.into()` keep `x`'s.
* **`f32`.** The default coefficient type is `f64`, which float literals infer too; GPU and
  game code that wants `f32` names it (`Point<(), f32>`), and slot lists name kinds bare
  (`Point<(Point,), f32>` is an `f32` map on points).
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
* **Comparing values.** `approx_eq(&b, tol)` is relative to the larger of 1 and the largest
  coefficient, which suits results of any size; `abs_diff_eq(&b, tol)` is absolute, for a
  tolerance in the units of the coefficients (a pixel, a field strength), and
  `max_abs_diff(&b)` is the number both read. A NaN is never close.
* **Degenerate metrics.** In PGA the metric pairing `|` of lines ignores the moment part, so energy
  forms built with it are singular. Build them with the regressive pairing `&`, as the modes example
  does.
