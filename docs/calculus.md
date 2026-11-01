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

In the spacetime algebra STA (`x² = y² = z² = 1`, `t² = −1`), `∇` is the derivative with respect
to an event, and the electromagnetic field is a bivector `F`. Maxwell's four equations are one,
`∇F = J`. A plane wave of any profile `φ` travelling along a null direction `k`, polarized along
`a ⟂ k`, is `F = (k ∧ a) φ(k · x)`, and it satisfies `∇F = 0` in vacuum:

```rust
use gax::calculus::{laplacian, nabla};
use gax::sta::{Scalar, Vector};

let k = Vector::<(), f64>::new(1.0, 0.0, 0.0, 1.0); // along x at the speed of light
let a = Vector::<(), f64>::new(0.0, 1.0, 0.0, 0.0); // polarized along y
let wave = |x: Vector<(), _>| {
    let (k, a) = (k.map_coefs(Into::into), a.map_coefs(Into::into));
    let phase: Scalar<(), _> = k.scalar_product(x);
    (k ^ a) * (phase * phase * phase) // the profile φ(s) = s³
};
let event = Vector::<(), f64>::new(0.2, -0.1, 0.3, 0.5);
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

On an interval, `gax::quadrature::gauss_legendre(n)` (and `gauss_legendre_on(n, a, b)`) is the
classical Gauss–Legendre rule, numpy's `leggauss`: `n` nodes exact to degree `2n − 1`, for
integrals along a parameter rather than over a simplex.

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

**Batches.** numga's per-simplex formulas (a triangle's gradients from the sides facing its
corners, a tetrahedron's edge forms) read the simplices of each simplex in a local order. An
`Incidence` holds, for each `n`-simplex, its `m`-simplices in the order of `mesh::corners(n, m)`
(the corners in order, the facet facing each corner, or the subsets in lexicographic order), each
with a sign:

* `incidence(n, m)`: unsigned;
* `oriented(n, m)`: `+1` where the stored simplex runs as the `n`-simplex's own order runs it;
* `facets(n)`: the facets signed by the orientation the simplex induces on them, an edge's tail
  `−1` and head `+1`, so that summed it is `d(n − 1)`.

Entry `j` is a field map (`map(j)`) or a reading of a field (`read(j, f)`, `pick` unsigned), and
`assemble(|r, j| cell)` weights each entry per simplex and sums them into one field map: the face
gradient of a triangle mesh is `t.incidence(2, 0).assemble(|f, k| side[k][f] * twice[f] * slot)`.

**The boundary and the interior.** `exterior(k)` lists the `k`-simplices on the boundary,
`interior(k)` the others, and `SparseMap::selection(&indices, len)` reads a field at the listed
sites (its transpose puts values back): the free vertices of a boundary-value problem are
`selection(&t.interior(0), n)`.

**Orientation.** `coherent()` turns top simplices so that neighbours agree, each connected piece
keeping the sense of its first simplex, and is `None` for a complex that cannot be oriented (a
Möbius strip). `turned()` reverses every orientation. On a mesh, `outward()` winds a closed
surface (a closed curve in the plane) outward, piece by piece, by the sign of the volume it
encloses in the model.

**Homology.** `cycles()` gives closed loops of vertices whose classes are a basis of `H₁`, `b₁` of
them: each closed by an edge off a breadth-first spanning tree, chosen independent of the
triangles' boundaries by exact elimination (on a surface, Eppstein's tree and cotree).
`circulation(&loops)` sums an edge cochain along each loop, so with `d(1)` it gives a cochain's
periods, the constraints of a trivial connection or a harmonic field.

```rust
use gax::field::{Field, SparseMap};
use gax::mesh::Topology;
use gax::vga3d::Scalar;

// A cylinder: a 6 × 2 grid of squares, two triangles each, `j` wrapping around.
let id = |i: usize, j: usize| i * 6 + j % 6;
let faces: Vec<[usize; 3]> = (0..2)
    .flat_map(|i| (0..6).flat_map(move |j| {
        [[id(i, j), id(i + 1, j), id(i + 1, j + 1)], [id(i, j), id(i + 1, j + 1), id(i, j + 1)]]
    }))
    .collect();
let cylinder = Topology::new(18, &faces).unwrap();
assert_eq!(cylinder.exterior(0).len(), 12); // the two rims
let loops = cylinder.cycles();
assert_eq!(loops.len(), 1); // once around
// A closed cochain that is not a gradient: one turn around, a sixth on each edge along a ring.
let turn: Field<Scalar<(), f64>> = cylinder
    .simplices(1)
    .map(|e| {
        let (a, b) = (e[0] % 6, e[1] % 6);
        let step = if b == (a + 1) % 6 { 1.0 } else if a == (b + 1) % 6 { -1.0 } else { 0.0 };
        Scalar::new(step / 6.0)
    })
    .collect();
assert!((&cylinder.d::<Scalar<(), f64>>(1) * &turn).iter().all(|c| c.s().abs() < 1e-15));
let period = (&cylinder.circulation::<Scalar<(), f64>>(&loops) * &turn).site(0).s();
assert!((period.abs() - 1.0).abs() < 1e-15);
// The interior vertices, read and put back.
let free = SparseMap::<Scalar<(), f64>>::selection(&cylinder.interior(0), 18);
assert_eq!(free.shape(), (6, 18));
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
* `form_inner(top, a, b)`, the pointwise inner product of two forms of one degree through the
  metric; integrated against the Whitney forms it is `mass(k)`.
* `angles()`, `angle_defects()`, `mass_with(&mesh, k, &materials)` and `gradients(&mesh, top)`,
  below.

The Whitney forms of every degree need no metric, so they are the topology's:
`topology.whitney(k, cochain, top, at)` is the form `Σ c_σ W_σ` of a `k`-cochain at a
barycentric point of a top simplex, and `topology.whitney_d(k, cochain, top)` its exterior
derivative, the Whitney form of `d c`. A `dec::Form` holds its coefficients on the products
`dλ_I` of the barycentric differentials, the basis dual to the top simplex's edges, so
`form.on_face(&[a, b, …])` reads it on a face by its local vertices, and a face's integral of a
Whitney form is that value at the face's centroid over `k!`. The cochain may take values in any
kind (a cochain of vectors gives a vector-valued form):

```rust
use gax::field::Field;
use gax::mesh::Topology;
use gax::vga3d::Scalar;

// A tetrahedron's six edges carry the differences of a vertex field: its 1-form is then a
// gradient, so its derivative vanishes, and it integrates back to the values on each edge.
let tet = Topology::new(4, &[[0, 1, 2, 3]]).unwrap();
let f: Field<_> = [0.0, 1.0, 3.0, -2.0].into_iter().map(Scalar::<(), f64>::new).collect();
let df = &tet.d::<Scalar<(), f64>>(0) * &f;
assert!(tet.whitney_d(1, &df, 0).coefficients().iter().all(|c| c.s().abs() < 1e-15));
let w = tet.whitney(1, &df, 0, &[0.0, 0.0, 0.5, 0.5]); // the midpoint of edge (2 3)
assert!((w.on_face(&[2, 3]).s() - (-2.0 - 3.0)).abs() < 1e-15);
```

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

**From edge lengths alone.** `Geometry::from_squared_lengths(&topology, &squared)` takes one
squared length per edge and no positions (Regge calculus): the metric is intrinsic, so volumes,
dual volumes, stars and masses follow as from a mesh, and lengths no embedding has (a curved
surface flattened triangle by triangle) keep their curvature. It refuses lengths that break a
triangle inequality. Lengths taken from positions give `Geometry::new`'s numbers (PyDEC's, to
`10⁻¹²`).

**Angles and curvature.** `angles()` gives each top simplex's interior angles at its
`(N−3)`-faces, laid out as `incidence(N − 1, N − 3)` lists them: a triangle's corner angles, a
tetrahedron's dihedral angles at its six edges. `angle_defects()` is the flat angle less their
sum around each `(N−3)`-simplex, a whole turn inside and half a turn on the boundary: on a
surface the Gaussian curvature integrated over each vertex's cell, which sums to `2π χ` over a
closed surface (Gauss–Bonnet); on a solid, Regge's curvature at each edge.

```rust
use gax::dec::Geometry;
use gax::field::Field;
use gax::mesh::Topology;
use gax::vga3d::Scalar;

// Five unit triangles around a vertex: a pentagonal pyramid's apex, a sixth of a turn short.
let fan: Vec<[usize; 3]> = (0..5).map(|i| [0, 1 + i, 1 + (i + 1) % 5]).collect();
let t = Topology::new(6, &fan).unwrap();
let ones: Field<_> = (0..t.len(1)).map(|_| Scalar::<(), f64>::new(1.0)).collect();
let g = Geometry::from_squared_lengths(&t, &ones);
assert!(g.angles().iter().all(|a| (a - std::f64::consts::FRAC_PI_3).abs() < 1e-15));
assert!((g.angle_defects()[0] - std::f64::consts::TAU / 6.0).abs() < 1e-14);
```

**A material per simplex.** `mass_with(&mesh, k, &materials)` is the Galerkin star with the
Whitney forms paired by a material on each top simplex instead of the metric: a symmetric form on
the model's vectors (`Scalar<(Vector, Vector)>`, a conductivity or permittivity), applied to the
barycentric gradients, and on `k`-forms through its outermorphism. With the model's own inner
product it is `mass(k)`; with a stretched one, the mass of the stretched mesh over the stretch's
volume factor. `gradients(&mesh, top)` gives those gradients, the reciprocal frame of the top
simplex's edges.

**Forms as multivectors.** A `Form`'s coefficients on `dλ_I`, combined with gradients, are the
multivector a model writes: `form.vector(&g)` for a 1-form, `form.bivector(&g)` and
`form.trivector(&g)` for 2- and 3-forms (`Σ c_I ∇λ_{i₁} ∧ ∇λ_{i₂} …`), the output the wedge's type.
The gradients may be the model's (`Geometry::gradients`) or their covectors where positions are
antivectors (a triangle's sides over twice its patch).

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
