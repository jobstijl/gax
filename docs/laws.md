# Laws of the composition language

gax's values, maps and forms obey a small set of algebraic laws. This page states them, says how
each one is checked, and lists the rewrites they license. Numerical behaviour (floating point,
fusion, drift) is in [numerics.md](numerics.md).

Tags: **[sym]** is an exact proof on symbolic coefficients, **[sym/ideal]** is an exact proof modulo
the ideal of a condition, **[gen]** is an exact proof the generator runs before it emits the
code, **[sample]** is exact arithmetic on sampled exact inputs (a randomized identity test, for
what symbolic proofs cannot reach; over a 61-bit prime field it is a proof up to a probability
below `2⁻⁴⁴⁰`, law L), **[prop]** is an f64 property test, **[doc]** is
documentation only. The
proofs are the test functions of the `law_suite!` macro in
[`crates/gax/tests/law_suite/mod.rs`](../crates/gax/tests/law_suite/mod.rs). Each algebra
instantiates it in a generated `crates/gax/tests/laws_{algebra}.rs`.

## 1. The structure

gax's extensors form a **symmetric colored linear operad**: a symmetric multicategory enriched in
vector spaces.

* **Colors** are the kinds (`Point`, `Line`, `Motor`, …).
* **A morphism** `(A₁, …, Aₙ) → B` is a `B<(A₁, …, Aₙ)>`.
* **Values** are nullary morphisms, `B<()>`. Filling a slot with a value is composition too.
* **Substitution:** `f.of(g)` substitutes `g` into slot 0, and `f.at::<I>().of(g)` into slot *i*.
  The argument's slot list is spliced in place of the slot, in order.
* **Symmetric:** `at::<I>` (move slot *i* to the front) and `swap` generate the permutations of
  the slots.
* **Linear:** each hom-set is a vector space, and composition is multilinear.
* **Unary fragment:** the maps `A<(B,)>` form the category of the kinds' coefficient spaces with
  linear maps (matrices). Each `K<(K,)>` is its endomorphism algebra.
* **Not closed.** Slots take only kinds, and `B<(C,)>` is not a kind. Going from `B<(A, C)>` to
  `A → B<(C,)>` is partial application, not an internal hom.
* **No copying or discarding of slots.** `fill` is the one deliberate exit: it restricts a
  multilinear map to the diagonal, which gives a homogeneous polynomial. Polarization goes the
  other way.

## 2. Method

`gax_gen::sym::Sym` implements `Coef` and `Real`. Its values are handles to hash-consed exact
polynomials over ℚ, with unbounded expansion, so `==` on symbolic values is polynomial identity.
The generated code is generic over the coefficient type, so the same code that runs on `f32` runs
on `Sym`.

* A law that holds for fresh symbolic inputs is **proved** for that algebra, those kinds and that
  slot shape.
* **Why the shapes stand for all shapes.** The code is emitted once, generic over the slot list
  `S`. The suites instantiate one-, two- and three-slot shapes with two different slot kinds, so
  every code path is covered.
* **Laws under a condition** (unit versors, or the conditions that make an element a versor) are
  proved modulo the ideal of the condition. `lhs − rhs` is reduced to normal form against a
  Gröbner basis (`gax_gen::groebner`). The generators come from the library's own products on the
  symbolic input: the parts of `m ~m − 1` for `Unit`, and the non-scalar parts of `m ~m` for "is a
  versor".
* **Only polynomial operations are proved.** `sqrt`, `recip` and the elementary functions make
  atoms, and branching cannot be proved this way. Normalization, solvers, exp and log have f64
  property tests instead.
* **What the suites cover.** `gax-regen` writes each suite from what it emitted for the algebra:
  * every product and unary operation among kinds with at most 8 coefficients;
  * every sandwich that keeps its kind, for versors with at most 8 coefficients;
  * every outermorphism of maps on kinds with at most 5 coefficients;
  * every scalar pairing whose matrix is a constant signed permutation.

  Larger instances make the free polynomials explode (see ADR-025). The suites cover the eight
  standard algebras and PGA4D, declared through `algebra!`.
* **Cost.** A default suite runs in 0.02 s (VGA2D) to 9 s (STA) after compiling. The slowest
  laws are marked `#[ignore]`, and CI runs them.
* **Negative controls.** Each algebra's suite checks that a deliberately false law fails: the
  equivariance of a product without its factor, `(m >> a) * (m >> b) == m >> (a * b)`.

## 3. The laws

### A. Multicategory (`multicategory`)

* [sym] Units: `K::slot().of(f) == f`, `f.of(K::slot()) == f`, and in slot 1,
  `f.at::<1>().of(K::slot()) == f.at::<1>()`.
* [sym] Associativity with multi-slot maps: `f.of(g.of(h)) == f.of(g).of(h)`. This also checks
  the order in which slot lists are spliced.
* [sym] Interchange: substitutions into different slots commute, for maps
  (`f.of(g).at::<1>().of(h)` against `f.at::<1>().of(h).at::<1>().of(g)`, up to the resulting slot
  order) and for values.
* [sym] Symmetry: `swap().swap() == id`. For three slots, with `σ = at::<1>` and `τ = at::<2>`:
  `σ² = 1`, `τ³ = 1`, `στσ = τ²` (the relations of the symmetric group).
* [sym] Permuting and then substituting is substituting into the permuted slot:
  `f.at::<1>().of(y) == f.fill(y)`, when `y`'s kind occurs only in slot 1.
* [sym] `gax::slots::reassoc` is the identity on coefficients. For generic lists it converts
  `Cat<Cat<A, B>, C>` to `Cat<A, Cat<B, C>>` (ADR-003). Its doctest brackets a generic function
  both ways, and `compile_fail/generic_bracketing.rs` shows the error without it.

### B. Linearity (`linearity`)

* [sym] `of` is additive and homogeneous in the argument and in the map, `f.of(0) == 0`, and a form
  is linear in each slot.

### C. Lifting is a homomorphism (`lifting`)

* [sym] For every product the kinds support (`*`, `^`, `&`, `|`, `⌋`, `⌊`, the scalar product, the
  commutator and the anticommutator) and every slot shape:
  * `op(f, g).of(x).of(y) == op(f.of(x), g.of(y))`, with slot order `Cat<S1, S2>`;
  * `op(a, g).of(y) == op(a, g.of(y))`;
  * `op(f, b).of(x) == op(f.of(x), b)`.
* [sym] For every unary operation (reverse, involute, conjugate, dual, undual):
  `op(f).of(x) == op(f.of(x))`.

### D. Versors

The matrix of a versor's action is spelled `v >> X::slot()`; there is no separate `to_matrix`
method (ADR-013).

* [sym] Action (`versor_action`): `(a * b) >> x == a >> (b >> x)` and
  `(a * b) >> X::slot() == (a >> X::slot()).of(b >> X::slot())`. This is exact for plain versors,
  where no norm condition is needed.
* [sym/ideal] `u << (u >> x) == x` for `Unit` versors, modulo `u ~u = 1` (`versor_action`).
* [sym/ideal] **Equivariance** (`equivariance`):
  `(m >> a) op (m >> b) == f · (m >> (a op b))`.
  * This holds only modulo the conditions that make `m` a versor (the non-scalar parts of `m ~m`
    vanish). For a free even element it fails, for example in PGA3D, where
    `m ~m = s + p e0123`.
  * The factor `f` depends on the operation and the algebra. The generator derives it per
    versor, product and pair of kinds, and the suite checks the library against it. The table is
    [law-factors.md](law-factors.md).
  * **Pattern:** every product picks up `‖m‖²`, the regressive product `&` included, under
    even and odd versors alike: `>>` takes its sign from the passenger's orientation (ADR-051),
    which is what makes reflections commute with the join. (With the plain sandwich, `&` picked
    up `−‖m‖²` under odd versors in even dimensions.)
  * For `Unit` versors the factor is its sign alone.
* [sym/ideal] **Conjugation** of maps, `conj_m(f) = (m >> f).of(m << K::slot())` (ADR-026):
  * `conj_m(f).of(m >> x) == ‖m‖⁴ · (m >> f.of(x))`, in `conjugation`. The factor is `‖m‖⁴`,
    not `‖m‖²`, because `~m (m x ~m) m = (~m m) x (~m m)`.
  * `conj_m(f) ∘ conj_m(g) == ‖m‖⁴ · conj_m(f ∘ g)`, in `conjugation_slow`.
  * `conj_{ab}(f) == conj_a(conj_b(f))` exactly [sym], in `conjugation_slow`.
  * For `Unit` versors all factors are 1, modulo `u ~u = 1`.
  * The suites prove these where a small versor keeps the slot kind under a plain sandwich and
    is closed under products (PGA2D, PGA3D, VGA2D, VGA3D, STA, CGA2D); elsewhere the list is empty.

### E. Outermorphism (`outermorphism`)

* [sym] It is a functor: `f.of(g).outermorphism::<B>() == f.outermorphism::<B>().of(g.outermorphism::<B>())`,
  and the identity extends to the identity.
* [sym] The determinant, read from the top grade, is multiplicative.
* [sym/ideal] The determinant of a unit versor's map on vectors (or on antivectors, such as PGA
  points) is `±1`, modulo `u ~u = 1`. The generator computes the sign for each versor kind:
  `+1` for even versors, and for odd ones whatever the oriented `x ↦ b >> x` of a unit basis
  blade `b` gives on that kind (ADR-051).

### F. Trace (`trace`)

* [sym] Cyclic: `f.of(g).trace() == g.of(f).trace()` for `f: A<(B,)>` and `g: B<(A,)>`.
* [sym] `K::slot().trace() == K::N`; the trace is linear.
* [sym] `trace_at::<I>` agrees with binding the other slots and taking the trace:
  `m.trace_at::<1>().of(y) == m.of(y).trace()`.

### G. Adjoint through a pairing (`adjoint_laws`)

There is no `adj` method, on purpose (ADR-014): the adjoint is a pairing solve,
`pairing.solve(A::slot() & t)`. Solves branch and divide, so they can't be proved on `Sym`.
Instead, for pairings whose matrix `P` is a constant signed permutation, the suite uses the
closed form `adj(t) = P tᵀ Pᵀ`, and proves:

* [sym] the defining identity, `pair(adj(t).of(l), x) == pair(l, t.of(x))`;
* [sym] it reverses composition, `adj(t ∘ s) == adj(s) ∘ adj(t)`;
* [sym] `adj(id) == id`;
* [sym] with the flipped pairing, applying it twice gives back `t`;
* [prop] `Pairing::solve` agrees with the closed form in f64.

**Which pairing.** In PGA the metric is degenerate (`e0 · e0 = 0`), so the inner product `|` is
singular. The join `&` pairs planes with points non-degenerately (the pairing matrix is a signed
permutation), and it is the pairing there. The suites include every such pairing that the
generator finds.

### H. `fill` is the diagonal (`fill`)

* [sym] Homogeneous of degree *k*, where *k* is the number of slots of `x`'s kind.
* [sym] It equals binding the same `x` into each of those slots.
* [sym] Polarization for *k* = 2: `Q(x + y) − Q(x) − Q(y) == B(x, y) + B(y, x)`, with
  `Q = B.fill(·)`.
* [sym] A form and its symmetrization `(B + B.swap()) / 2` fill to the same thing.

### I. The implementation refines the semantics

* **Already in the generator.** The generator re-expands every fused program and checks it against
  the traced polynomials modulo the type conditions (ADR-011, `gax-gen/tests/symbolic.rs`).
* [sym] **The generated kernels on `Sym`** (`tiers`):
  * the sandwich value path equals the tier-1 products `(v * x) * ~v`, projected to the output
    kind, with the orientation's sign for an odd versor (ADR-051: a blade of grade `k` times
    `(−1)ᵏ` for an extrinsic `x`, `−(−1)ᵏ` for an intrinsic one);
  * the map path `(v >> X::slot()).of(x)` equals it;
  * the prepared action equals it, and its dense form equals `v >> X::slot()`.

  For plain versors these are exact; for `Unit` versors they hold modulo `u ~u = 1`.
* **Batch lanes** run the same generic code, but not on `Sym`, and they are not bit-identical to
  scalar code: lanes use FMA where the level has it. `tests/batch.rs` compares them with scalar
  kernels within a tolerance. Only the elementary functions of `batch::math` are bit-identical
  between lanes and scalar code. With the `deterministic` feature, lanes and scalar code are
  bit-identical everywhere (`tests/determinism.rs`, ADR-027).
* [prop] In f32, tier 1 and the fused tier 2 differ by at most the sum of their computed forward
  error bounds ([`gax-gen/tests/error_bounds.rs`](../crates/gax-gen/tests/error_bounds.rs);
  [numerics.md](numerics.md)).

### J. Non-laws [doc]

* No copying or discarding of slots, except through `fill`.
* No `+` between different slot lists (`compile_fail/add_different_slots.rs`,
  `compile_fail/add_map_to_value.rs`).
* `Cat` is associative only through `reassoc` (`compile_fail/generic_bracketing.rs`).

### K. Division, and maps between algebras

* [sym/ideal] **Division undoes the product** (`division`): `(a / v) v == a` for every invertible
  versor kind `v` and every kind `a` whose quotient comes back to a kind that holds it. It is
  checked on symbolic values modulo the relations of the reciprocals in the inverse (`r N = 1`),
  so it holds wherever `v` is invertible. Dividing by a `Unit` versor multiplies by its reverse
  (`tests/between.rs` checks it against the inverse).
* [gen] **Homomorphisms between algebras keep products.** For each of the twelve declared
  homomorphisms (ADR-032), the generator proves `φ(a b) == φ(a) φ(b)` exactly, on symbolic
  values of every pair of source kinds, before it emits the `From` impls, and proves
  `φ(~a) == ~φ(a)` for the kinds it lets `Unit::widen` convert. `tests/homs.rs` checks the
  emitted code on random values.

### L. Versors with more than 8 coefficients (`sampled_versors`, `sampled_versors_mod_p`)

The free symbolic laws of a 16-coefficient versor (CGA3D's `Even` and `Odd`, STAP's `Motor`, a
5D PGA's `Motor`) are out of reach: the Gröbner bases of their conditions explode. Every
versor is a product of vectors, though, so these laws are proved on products of reflections,
by randomized identity testing:

* **Sampling.** A unit vector is a basis vector `u₀` of square +1 reflected in a vector `w`,
  `u₀ − 2 (u₀·w)/(w·w) w`: unit exactly, in any metric, degenerate and null bases included. The
  product of four (three for an odd versor), formed in the full multivector, is a unit versor,
  and is checked to lie in the versor's kind.
* **Over ℚ** (`sampled_versors`): `w` with small integer coordinates and the passenger
  symbolic, so each law is an identity in the passenger, at three versors (checked to be
  generic: at least half their coefficients nonzero).
* **Over ℤ/p** (`sampled_versors_mod_p`, `p = 2⁶¹ − 1`, the coefficient type `gax::fp::Fp`):
  `w` and the passenger uniform in `ℤ/p`, eight samples per law. The generated code is generic
  over the coefficient ring, so it runs unchanged there, exactly.
* [sample] **The laws:**
  * the plain and `Unit` sandwich kernels equal the projection of `(v x) ~v`, with the
    orientation's sign for an odd versor, the map path equals
    them, and `u << (u >> x) == x`;
  * `Unit` versors compose: `(a b) >> x == a >> (b >> x)`;
  * the single-grade products are equivariant, up to the sign that a unit versor's factor can be.

**Why the samples prove the laws.** Written in the reflections' parameters `w` and the
passenger's coefficients `x`, each law is `N(w, x) = 0` for a polynomial `N` with integer
coefficients (the denominators `w·w` are nonzero at the samples). A versor of four reflections
has coefficients of degree 8 in `w`, a sandwich is quadratic in the versor, and the laws above
have degree below 70 in all. If `N ≢ 0` over ℚ, and `p` does not divide all its coefficients,
then over `ℤ/p` a uniform sample is a root with probability at most `deg N / p < 2⁻⁵⁵` (the
Schwartz–Zippel lemma); eight independent samples all miss with probability below `2⁻⁴⁴⁰`. So a
passing run proves `N ≡ 0` (up to that probability, and to `p` dividing every coefficient of a
nonzero `N`). An identity in the reflections holds for every product of reflections, and, being
polynomial, on their Zariski closure: over ℂ, reflections of `u₀` reach every vector of square 1,
and their products every versor of the kind (Cartan–Dieudonné). The ℚ check adds exactness in
the passenger with no probability at all, at few points.

The same test runs on the 7D, 8D and 9D algebras of `gax-highdim-tests` (`exact_mod_p`), whose
even versors (64 to 256 coefficients) have plain sandwich kernels (ADR-034).

### M. Kind tables (`casts`, `sub_kinds`, `grades`)

For every pair of kinds sharing a blade, every pair where one kind's blades all lie in the
other's, and every grade of every kind (ADR-036), on symbolic coefficients:

* [cast] `x.cast::<B>()` is the projection of `x` onto `B`'s blades by blade name and orientation,
  computed in the test from the blade names alone, not from the generated tables; on a map,
  `m.cast::<B>().of(p) == m.of(p).cast::<B>()`.
* [sub] For `A ⊆ B`: `x.cast::<B>().cast::<A>() == x`, `x.cast::<B>() == B::from(x)` (the
  independently generated embedding), `f.of(x) == f.of(x.cast::<B>())` for a form `f` on `B`, and
  `f.of(A::slot()).of(x) == f.of(x)` (narrowing a slot composes).
* [grade] `x.grade::<G>()`, cast back to `x`'s kind, is `x` with every blade of another grade
  zeroed.

### N. The general inverse (`general_inverses`)

For every kind whose `inverse` is the general one (ADR-037), in `f64` on fifty random values:
`x x⁻¹ = x⁻¹ x = 1` within `10⁻⁹` times the condition `‖x‖ ‖x⁻¹‖`. This one is numerical, not
exact (the method divides by the recursion's scalars), but its failure modes are not subtle: a
characteristic polynomial of the wrong degree, or a null direction handled wrongly, leaves
`x x⁻¹ − 1` of order one. The 7D to 9D algebras of `gax-highdim-tests` check the same on their
even kinds (`general_inverse`).

### O. The multivector derivative (`tests/calculus.rs`)

For VGA2D, VGA3D, STA and CGA3D, on cubic fields with random integer coefficients at random points
of the prime field (law L), exactly (ADR-047):

* [sample] `∇x = n`: the divergence of the identity field is the dimension, its curl zero.
* [sample] `∇(x ∗ x) = 2x`, and `∂_X (X ∗ A) = A` for vector and bivector `X`.
* [sample] `∇∧(∇∧f) = 0` for a scalar field and `∇·(∇·B) = 0` for a bivector field.
* [sample] For a vector field, the grade-0 and grade-2 parts of `∇F` are `∇·F` and `∇∧F`.
* [sample] `∇²f = ∇·(∇f)` (one evaluation on jets of jets against two nested ones).
* [sample] Covariance under the rational unit rotor `(3 + 4 e12)/5` in VGA3D: the derivative of
  `y ↦ R F(~R y R) ~R` at `R x ~R` is `R (∇F)(x) ~R`.
* [sample] STA: a plane wave `(k ∧ a) φ(k · x)`, `k` null and `a ⟂ k`, satisfies `∇F = 0`.
* [sample] An `algebra!` declaration with an off-diagonal metric: `e^i ∗ e_k = δ_ik` for every
  blade of its kinds.
* [prop] VGA3D: the point charge's field `x / |x|³` is free of divergence and curl away from the
  charge, within `10⁻¹²`.
* [gen] gax-gen's own tests: in PGA3D only `Scalar` and `Rotor` have a frame, and in CGA3D the
  reciprocal of `eo` is `−e∞`. A kind without a frame gives the compile error of
  `tests/compile_fail/nabla_degenerate.rs`.

### P. Integration over simplices (`gax-core` `simplex::tests`, `tests/simplices.rs`)

Exactly, on the prime field (law L), unless marked (ADR-048):

* [sample] Every Grundmann–Möller rule to degree 15, and the vertices-and-centroid rule, is exact
  on every barycentric monomial of its degree: the mean of `λ^α` over a `k`-simplex is
  `k! α! / (|α| + k)!`. Checked for every monomial, on simplices of dimension 1 to 4 (1 and 2
  above degree 9); the weights sum to one. One degree higher, every rule fails on some monomial,
  so the degrees claimed are the true ones.
* [sample] The fundamental theorem of geometric calculus on one simplex,
  `∫_σ (∇ ⌋ dX) F = Σᵢ (−1)ⁱ ∫_{∂ᵢσ} dS F` (the derivative contracting the measure from the left,
  the faces oriented by the boundary map), for cubic fields of scalars, vectors and bivectors,
  on edges, triangles and tetrahedra in VGA2D and VGA3D and up to 4-simplices in STA.
* [sample] PGA3D: the measure of a simplex moved by a motor of rational coefficients is the moved
  measure (lines and planes; the volume unchanged), and the mean position moves with it.
* [prop] The same Euclidean simplex in VGA, PGA and CGA (2D and 3D): the norms of the measures of
  its edges and triangles agree, and so do the signed volumes of the full-dimensional simplices,
  within `10⁻¹²`.
* [prop] In `f64`, every rule stays within the rounding bounds of numerics.md.

### Q. Simplicial complexes (`tests/meshes.rs`, the fuzz target `mesh_complex`)

On complexes of edges, triangles, tetrahedra and 4-simplices (ADR-049):

* [sample] `d(k+1) ∘ d(k) = 0`, and each face listed for a simplex holds its other vertices, on
  200 random lists of simplices per dimension and on every fuzzed input. Errors are reported for
  the right reason (the index out of range, the vertex repeated, the two simplices equal as sets).
* [sample] Euler–Poincaré: `Σ (−1)ᵏ bₖ = χ`, from the exact ranks over `ℤ/p` against the counts of
  simplices.
* [doc] The Betti numbers of known spaces: a disk (1, 0, 0), a cylinder and a Möbius strip
  (1, 1, 0), a torus (1, 2, 1), a Klein bottle (1, 1, 0), the octahedron (1, 0, 1), a cube of six
  tetrahedra (1, 0, 0, 0), the boundary of a 4-simplex (1, 0, 0, 1), two disjoint triangles
  (2, 0, 0). The Möbius strip and the Klein bottle are not oriented; one face of the octahedron
  turned over breaks its orientation and not its topology.
* [doc] numga's mesh: the examples' `Mesh` on `gax::mesh` gives the same edges, face orientations
  and couplings of `d0`, `d1` and the means as before, bit for bit
  (`examples/numga/src/mesh.rs`, `gax_mesh_keeps_every_number`).

### R. Discrete exterior calculus (`tests/dec.rs`)

On jittered grids of triangles and tetrahedra, some not well-centred (ADR-050):

* [prop] Hirani's PyDEC (`tests/fixtures/pydec.txt`, `docs/research/experiments/pydec_fixtures.py`):
  every primal volume, signed circumcentric dual volume and diagonal Hodge star, of every
  dimension, within `10⁻¹²` (PyDEC signs top volumes by orientation; gax's metric is
  orientation-free, so those are compared in absolute value).
* [prop] VGA3D, PGA3D and CGA3D give the same volumes and dual volumes for the same mesh.
* [prop] The Galerkin stiffness `d₀ᵀ M₁ d₀` is the stiffness of linear finite elements
  `Σ V ∇λₐ · ∇λ_b`, the gradients computed independently as reciprocal frames of the edges
  (`(e₂ ∧ e₃)(e₁ ∧ e₂ ∧ e₃)⁻¹`), in 2D and 3D. On triangles `d₀ᵀ ⋆₁ d₀` is the same (the
  cotangent formula); on tetrahedra only for a regular one, which is checked. The top mass is
  `1/V`.
* [sample] Stokes for the de Rham map, `R(∇∧F) = d R(F)`, exactly on `ℤ/p`, for cubic fields: the
  gradient theorem (scalars), Green's and Stokes's (vectors), Gauss's (bivectors in 3D).
* [prop] The Whitney map inverts the de Rham map on edges, `R ∘ W = id`, and `W₁ d₀ = ∇ W₀`.
* [prop] `δ` is the adjoint of `d` in the stars' inner products, `⟨d a, b⟩_⋆ = ⟨a, δ b⟩_⋆`, on a
  sphere (so the Hodge decomposition is orthogonal); `D² = Δ` degree by degree.
* [prop] The harmonic cochains, the null space of `Δ_k` found by `eigh` on `(⋆Δ, ⋆)`, have the
  Betti numbers' dimensions on a sphere (1, 0, 1) and a torus (1, 2, 1), both well-centred.
* [prop] On subdivided icosahedra the first 16 eigenvalues of the Laplacian on vertices converge to
  the unit sphere's `ℓ(ℓ+1)` at second order, the error falling more than threefold per
  subdivision, for both stars (16th eigenvalue off by 2.7, 0.87, 0.23 with the diagonal star and
  4.2, 1.0, 0.25 with the Galerkin one, at 1, 2 and 3 subdivisions).
* [prop] The numga ports' hand-built metric: their edge ratios are `⋆₁`, their vertex areas the
  row sums of `M₀`, their reconstruction `W₁` at the centroids, within `10⁻¹²`
  (`examples/numga/src/mesh.rs`, `gax_dec_gives_the_metric`).

## 4. Licensed rewrites

Each law licenses a transformation:

* **Folding versor chains:** `a >> (b >> x)` becomes `(a * b) >> x` (action law).
* **Reassociation** of composition and contraction order (associativity, interchange).
* **Distributing and factoring** by linearity.
* **Outermorphism fusion:** `(f ∘ g)^ = f^ ∘ g^` (functor law).
* **Trace cyclicity.**

At run time, Rust executes the expression as written, so a rewrite there is advice to the user
(or a new API), not an optimizer pass. In the tracer, expanded polynomials do not depend on
bracketing, so rewrites can matter only at the low expansion limits of the ADR-010 portfolio,
where the traced DAG keeps the shape as written. The experiments with folding versor chains and
reassociation in the tracer are in [performance.md](performance.md#law-based-rewrites-in-the-tracer).
