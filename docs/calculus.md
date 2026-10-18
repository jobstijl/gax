# Geometric calculus

Geometric calculus differentiates and integrates with the geometric product. Its derivative with
respect to a vector, `∇`, holds the divergence and the curl in one operator, and it differentiates
with respect to any multivector just as well. This page covers `gax::calculus`. The design is in
ADR-047 of the [design record](design.md), and the laws it satisfies are in [laws.md](laws.md),
section O.

## The derivative with respect to a multivector

A value `x` of a kind with blades `e_J` has coefficients `x^J`. The derivative of a function `F` with
respect to `x` is

```text
∂_X F = Σ_J e^J ∂_J F
```

where `∂_J` differentiates along `x^J` and `e^J` is the *reciprocal* blade: `e^J ∗ e_K = δ^J_K`,
with `∗` the scalar product. For a vector `x` this is the vector derivative `∇F`. Its two grade
parts are the divergence `∇·F` and the curl `∇∧F`:

```rust
use gax::calculus::{curl, divergence, nabla};
use gax::vga3d::{Bivector, Vector};
use gax::Coef;

// A rotation about e12, x ↦ x ⌋ e12. It is free of divergence, and its curl is twice the
// plane of rotation.
fn swirl<T: Coef>(x: Vector<(), T>) -> Vector<(), T> {
    x.lc(Bivector::new(T::zero(), T::zero(), T::one()))
}
let x = Vector::<(), f64>::new(0.3, -1.0, 2.0);
let (_, d) = nabla(swirl, x);
assert_eq!(divergence(swirl, x).1.s(), 0.0);
assert_eq!(curl(swirl, x).1, Bivector::new(0.0, 0.0, 2.0));
assert_eq!(d.grade::<0>().s(), 0.0);
assert_eq!(d.grade::<2>(), Bivector::new(0.0, 0.0, 2.0));
```

* `nabla(f, x)` returns `f(x)` and `∂_X f`. The derivative's kind is that of the geometric product
  of `x` and `f(x)`: the derivative of a vector field is a scalar plus a bivector.
* `divergence(f, x)` and `curl(f, x)` are its inner (`⌋`) and outer (`∧`) parts.
* `laplacian(f, x)` is `∂_X ∗ ∂_X f`, for a vector `x` the Laplacian `∇²f = ∇·∇f`, of the kind
  of `f`.

The functions differentiate by forward mode, as `jacobian` does
([guide](guide.md), section 15). `f` sees its argument with jet coefficients, so write it generic
over the coefficient type, as `swirl` is, or as a closure over the jet type. A function generic
over `Coef` also runs on the prime field `gax::fp::Fp`, where the derivatives of polynomials are
exact. A function that needs roots or transcendentals is written over `Real`.

## Spacetime: Maxwell's equations

In the spacetime algebra STA (`t² = 1`, `x² = y² = z² = −1`), `∇` is the derivative with respect
to an event, and the electromagnetic field is a bivector `F`. Maxwell's four equations are one,
`∇F = J`. A plane wave of any profile `φ` travelling along a null direction `k`, polarized along
`a ⟂ k`, is `F = (k ∧ a) φ(k · x)`, and it satisfies `∇F = 0` in vacuum:

```rust
use gax::calculus::{laplacian, nabla};
use gax::sta::{Scalar, Vector};

let k = Vector::<(), f64>::new(1.0, 1.0, 0.0, 0.0); // along x at the speed of light
let a = Vector::<(), f64>::new(0.0, 0.0, 1.0, 0.0); // polarized along y
let wave = |x: Vector<(), _>| {
    let (k, a) = (k.map_coefs(Into::into), a.map_coefs(Into::into));
    let phase: Scalar<(), _> = k.scalar_product(x);
    (k ^ a) * (phase * phase * phase) // the profile φ(s) = s³
};
let event = Vector::<(), f64>::new(0.5, 0.2, -0.1, 0.3);
let (f, maxwell) = nabla(wave, event);
assert!(f.c.iter().any(|&c| c != 0.0));
assert!(maxwell.c.iter().all(|&c| c == 0.0));

// The same profile satisfies the wave equation □φ = 0.
// Constants enter a function of jets of jets with `Coef::from_f64`.
use gax::Coef;
let (_, box_phi) = laplacian(|x: Vector<(), _>| {
    let phase: Scalar<(), _> = k.map_coefs(Coef::from_f64).scalar_product(x);
    phase * phase * phase
}, event);
assert_eq!(box_phi.s(), 0.0);
```

## Which kinds have a derivative

The reciprocal blades invert the Gram matrix `G_JK = e_J ∗ e_K` of a kind's blades. gax-gen
inverts it exactly, in rationals, for every kind, and the kinds where it is invertible implement
`gax::calculus::Reciprocal`. Off-diagonal metrics are covered: in CGA, `eo · e∞ = −1`, so the
reciprocal of `eo` is `−e∞`. So are the metrics of `algebra!` declarations.

| algebra | kinds with a reciprocal frame |
|---|---|
| VGA2D, VGA3D, STA | every kind |
| CGA2D, CGA3D, CSTA | every kind holding `eo` and `e∞` together, or neither |
| PGA2D, PGA3D | only the kinds without `e0`: `Scalar` and `Rotor` |
| STAP | only `Scalar` |

A kind that holds a null direction (PGA's `e0`, which squares to zero) has a singular Gram matrix:
no element of the kind pairs to one with `e0`, so a derivative cannot be taken back to an element.
Calling `nabla` on a PGA plane is a compile error that says so. There, the derivative is the form
that `gax::jet::gradient` returns (guide, section 15), which needs no metric.

## Integration over simplices

`gax::simplex` integrates over simplices: segments, triangles, tetrahedra, and their analogues in
any dimension (ADR-048). A `Simplex` is its vertices in order, which orient it.

```rust
use gax::simplex::{Rule, Simplex};
use gax::pga3d::{Point, Scalar};

// A triangle in space, its vertices points of weight one.
let t = Simplex([Point::<(), f64>::xyz(0.0, 0.0, 0.0), Point::xyz(2.0, 0.0, 0.0), Point::xyz(0.0, 1.0, 1.0)]);
let area = t.measure().norm(); // the measure is the plane through it, over 2
assert!((area - 2f64.sqrt()).abs() < 1e-15);
// The mean of z² over it, exact for polynomials up to the rule's degree: 1/6.
let z2 = t.mean(Rule::degree(2), |p| Scalar::new(p.c[1] * p.c[1]));
assert!((z2.s() - 1.0 / 6.0).abs() < 1e-15);
// ∫ z² dA is the mean times the area.
assert!((t.integral(Rule::degree(2), |p| Scalar::new(p.c[1] * p.c[1])).s() - 2f64.sqrt() / 6.0).abs() < 1e-15);
```

(In PGA3D, `p.c[1]` is a point's `e013` coefficient, its `y`, and for this triangle `z = y`.)

* `mean(rule, f)`: the mean of `f` over the simplex, for `f` returning any value or map.
* `measure()`: the *directed measure* `dX`, a blade whose norm is the simplex's volume and whose
  attitude and sign are its span and orientation.
* `directed(rule, f)`: `∫ dX F`, the measure times the mean, as geometric calculus writes it.
* `integral(rule, f)`: `∫ F dV`, the mean times the volume.

**Rules.** `Rule::degree(d)` is the cheapest Grundmann–Möller rule exact on polynomials of degree
`d`, up to 15. Its nodes and weights are rational, so on `gax::fp::Fp` an integral of a polynomial
is exact. In floating point the weights cancel: numerics.md has the error per degree.
`Rule::VerticesAndCentroid` has positive weights and is exact to degree 2.

**Models.** What a position is depends on the model of space, and `simplex::Position` and
`simplex::Measure<N>` say how a model combines positions and measures a simplex of `N` vertices:

| model | positions | measure of a `k`-simplex |
|---|---|---|
| VGA2D, VGA3D, STA | vectors | `(v₁−v₀) ∧ … ∧ (v_k−v₀) / k!` |
| PGA2D, PGA3D | points of weight one | the join `P₀ ∨ … ∨ P_k / k!`: a line, a plane, a volume |
| CGA2D, CGA3D | round points of weight one | the flat `P₀ ∧ … ∧ P_k ∧ e∞ / k!` |

In every model the norm of the measure is the volume, and a simplex of the full dimension has
the same signed volume in all three. A single vertex measures one. For an algebra of your own,
implement the two traits for its position kind.

**The fundamental theorem.** With `∇` from the first part, the theorem of geometric calculus holds
on each simplex: `∫_σ (∇ ⌋ dX) F = Σᵢ (−1)ⁱ ∫_{∂ᵢσ} dS F`. The face `∂ᵢσ` leaves out vertex
`i`, and `(∇ ⌋ dX) F` is `Σⱼ (eʲ ⌋ dX) ∂ⱼ F`, which `gax::calculus::contract` computes. It
contains the theorems of Stokes, Gauss and Green, and laws.md (section P) has it as a law, exact
for polynomial fields.

```rust
use gax::calculus::contract;
use gax::simplex::{Rule, Simplex};
use gax::vga3d::{Scalar, Vector};
use gax::Coef;

// Green's theorem for F = x y² on a triangle: the derivative over the area equals the field
// around its edges.
fn f<T: Coef>(x: Vector<(), T>) -> Scalar<(), T> {
    Scalar::new(x.c[0] * x.c[1] * x.c[1])
}
let v = [Vector::<(), f64>::new(0.0, 0.0, 0.0), Vector::new(1.0, 0.0, 0.0), Vector::new(0.0, 2.0, 0.0)];
let t = Simplex(v);
let dx = t.measure();
let inside = t.mean(Rule::degree(2), |p| contract(f, p, |up, column| up.lc(dx) * column).1);
let around = Simplex([v[1], v[2]]).directed(Rule::degree(3), f)
    - Simplex([v[0], v[2]]).directed(Rule::degree(3), f)
    + Simplex([v[0], v[1]]).directed(Rule::degree(3), f);
assert!((inside - around).c.iter().all(|c| c.abs() < 1e-15));
```

## Meshes

A mesh is a chain of simplices that share faces. `gax::mesh` (feature `std`) holds the topology of
one of any dimension and its boundary maps, the discrete exterior derivative (ADR-049).

* `Topology<N>` is built from its top simplices, `N` vertex indices each (triangles for a
  surface, tetrahedra for a solid). It derives every face: the `k`-simplices for `k` from 0 (the
  vertices) to `N − 1`. The top simplices keep their order, which orients them; below the top a
  simplex's vertices are sorted, and each dimension is in lexicographic order.
* `d(k)` maps fields on the `k`-simplices to fields on the `(k+1)`-simplices, a `SparseMap` with
  cells `±1`: the difference along each edge for `k = 0`, the circulation around each triangle for
  `k = 1`. Each row lists a simplex's faces with the orientations they get from it, so
  `d(k+1) ∘ d(k) = 0`.
* `means(k)` takes the mean over each `k`-simplex's vertices.
* `Mesh<V, N>` is a topology with its vertices, a `Field` of any kind: VGA vectors, PGA points.

The construction returns an error, not a panic, for a vertex index out of range, a simplex that
repeats a vertex, or a top simplex given twice. Non-manifold complexes are valid, and queries tell
them apart:
* `is_pseudomanifold`: no face in more than two top simplices.
* `is_oriented`: neighbours induce opposite orientations on each face they share.
* `boundary`: the faces of one top simplex.
* `euler_characteristic`.
* `betti_numbers`: the dimensions of the homology, from exact ranks of the boundary maps over the
  prime field.

```rust
use gax::mesh::Topology;

// A torus: a 4 × 4 grid of squares, two triangles each, with opposite sides glued.
let id = |i: usize, j: usize| (i % 4) * 4 + j % 4;
let faces: Vec<[usize; 3]> = (0..4)
    .flat_map(|i| (0..4).flat_map(move |j| {
        [[id(i, j), id(i + 1, j), id(i + 1, j + 1)], [id(i, j), id(i + 1, j + 1), id(i, j + 1)]]
    }))
    .collect();
let torus = Topology::new(16, &faces).unwrap();
assert_eq!(torus.betti_numbers(), [1, 2, 1]); // one piece, two loops, one cavity
assert_eq!(torus.euler_characteristic(), 0);
assert!(torus.is_oriented() && torus.boundary().is_empty());
```

`pga3d::Moments::of_mesh` takes a closed surface as a `Mesh<Point, 3>`, and `pga2d::Moments::of_mesh`
a closed curve as a `Mesh<Point, 2>`; `of_faces` and `of_polygon` take slices.

## Discrete exterior calculus

`gax::dec` (feature `std`) adds the metric to a mesh: volumes, Hodge stars, and the maps between
fields and cochains (ADR-050).

**The metric is intrinsic.** A model of space gives the squared distance between two positions
(`dec::Distance`: VGA's `|a − b|²`, PGA's between points of weight one, CGA's `−2 A ∗ B`). Everything
else follows simplex by simplex from the Gram matrix of its edges, `Gᵢⱼ = ½(d₀ᵢ² + d₀ⱼ² − dᵢⱼ²)`:
volumes, circumcentres, the inner products of the barycentric gradients. VGA, PGA and CGA give the
same numbers for the same mesh.

`Geometry::new(&mesh)` computes the primal volumes and the circumcentric dual volumes of every
simplex, and gives:

* `star(k)`, the diagonal Hodge star `⋆ₖ = |σ*| / |σ|` (Hirani 2003). The dual volumes are signed:
  a piece of a dual cell counts negative where a circumcentre falls outside its face, so the star
  can be negative off a well-centred mesh (`is_well_centered`). It matches Hirani's PyDEC to
  `10⁻¹²`.
* `mass(k)`, the Galerkin Hodge star: the mass matrix of the Whitney forms, exact, symmetric and
  positive definite on any mesh.
* `whitney_1(&mesh, cochain, top, at)`, the Whitney map on edges: the vector field of a 1-cochain
  inside a top simplex. It inverts the de Rham map.

`dec::de_rham(&mesh, rule, f)` integrates a `k`-vector field over each `k`-simplex against its
measure, `∫_σ F ∗ ~dX`: a cochain. It commutes with the derivative, `R(∇∧F) = d R(F)`, which is
Stokes's theorem.

```rust
use gax::dec::Geometry;
use gax::field::Field;
use gax::mesh::Mesh;
use gax::vga3d::{Scalar, Vector};

// A regular tetrahedron: one solid, its four faces, six edges.
let corners = [[1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]];
let v: Field<_> = corners.iter().map(|p| Vector::<(), f64>::new(p[0], p[1], p[2])).collect();
let tet = Mesh::new(v, &[[0, 1, 2, 3]]).unwrap();
let g = Geometry::new(&tet);
assert!((g.volumes(3)[0] - 8.0 / 3.0).abs() < 1e-14);
assert!(g.is_well_centered());
// On 0-forms the Galerkin stiffness d₀ᵀ M₁ d₀ is the stiffness of linear finite elements.
let d0 = tet.d(0);
let stiffness = &d0.transpose_with(|c| c) * &(&g.mass::<Scalar<(), f64>>(1) * &d0);
let ones: Field<_> = (0..4).map(|_| Scalar::<(), f64>::new(1.0)).collect();
let image: Field<Scalar<(), f64>> = &stiffness * &ones;
assert!(image.sites().iter().all(|x| x.s().abs() < 1e-14));
```

**The two stars on 0-forms.** On triangles both give the cotangent Laplacian `d₀ᵀ ⋆₁ d₀ = d₀ᵀ M₁ d₀`,
the stiffness of linear finite elements. On tetrahedra only the Galerkin one does: the circumcentric
star agrees there only on symmetric meshes (a regular tetrahedron), and differs by tens of percent
on a jittered grid. Use `mass` where finite-element accuracy matters, `star` where a diagonal matters.

## Operators

With the stars, `Geometry` gives the operators of calculus on cochains (ADR-050):

* `codifferential(k)`: `δ_k = ⋆⁻¹ dᵀ ⋆`, from `k`- to `(k−1)`-cochains, the adjoint of `d` in
  the stars' inner products. The images of `d` and `δ` are orthogonal, which makes the Hodge
  decomposition `ω = dα + δβ + h` orthogonal.
* `laplacian(k)`: the Laplace–de Rham operator `Δ_k = dδ + δd`. Its null space holds the harmonic
  cochains, as many as the `k`-th Betti number.
* `laplacian_form(k)`: the symmetric form `⋆_k Δ_k`, which with `star(k)` makes the pencil whose
  eigenpairs are `Δ`'s (`SparseMap::eigh`, on a well-centred mesh, whose stars are
  positive). On vertices it is the cotangent Laplacian.
* `dirac()`: the Dirac operator `D = d + δ` on cochains of every degree at once, one field over all
  the simplices (`offsets()` says where each degree starts). `D² = Δ` degree by degree: it is
  the discrete counterpart of `∇ = ∇∧ + ∇⌋`.
* `stiffness(k)`: the Galerkin `d_kᵀ M_{k+1} d_k`, which with `mass(k)` is the finite-element pencil
  of the Whitney forms. Its mass is positive definite on any mesh.

The diagonal-star operators need nonzero dual volumes (`codifferential` panics on a circumcentre
that falls on a face, a right triangle's hypotenuse).

```rust
use gax::dec::Geometry;
use gax::field::Field;
use gax::mesh::Mesh;
use gax::vga3d::{Scalar, Vector};

// The octahedron: equilateral faces, so well-centred.
let p = [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]];
let v: Field<_> = p.iter().map(|p| Vector::<(), f64>::new(p[0], p[1], p[2])).collect();
let faces = [[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4], [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]];
let octahedron = Mesh::new(v, &faces).unwrap();
let g = Geometry::new(&octahedron);
assert!(g.is_well_centered());
// The Laplacian on vertices has one zero eigenvalue: the surface is one piece.
let form = &g.laplacian_form::<Scalar<(), f64>>(0) * Scalar::<(), f64>::slot();
let star = &g.star::<Scalar<(), f64>>(0) * Scalar::<(), f64>::slot();
let (values, _) = form.eigh(&star, 3);
assert!(values[0].abs() < 1e-12 && values[1] > 0.5);
// The Dirac operator acts on the cochains of all 26 simplices at once (D² = Δ, laws.md §R).
assert_eq!(g.offsets(), [0, 6, 18, 26]);
assert_eq!(g.dirac::<Scalar<(), f64>>().shape(), (26, 26));
```

## Examples

Three programs in `crates/gax/examples` use the whole chain, and each checks what it shows:

* `heat_tetrahedra`: heat spreading through a cube of tetrahedra by the Galerkin pencil, stepped
  by implicit Euler. The total heat stays what it was to `10⁻¹³`, and the temperatures even out.
* `hodge_torus`: a random field on the edges of a torus split into its gradient, curl and harmonic
  parts. The torus has two harmonic 1-forms, one per loop; the parts are orthogonal to
  `10⁻¹²`, the curl part co-closed, the harmonic part closed and co-closed.
* `cavity_modes`: the transverse-electric modes of a square cavity. In spacetime algebra each
  is one bivector field, and `nabla` checks Maxwell's `∇F = 0`; their frequencies come from the
  discrete Laplacian on a triangulated square, to second order with both stars.

```sh
cargo run --release -p gax --features all-algebras --example cavity_modes
```
