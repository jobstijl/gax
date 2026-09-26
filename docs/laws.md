# Laws of the composition language

gax's values, maps and forms obey a small set of algebraic laws. This page states them, says how
each one is checked, and lists the rewrites they license. Numerical behaviour (floating point,
fusion, drift) is in [numerics.md](numerics.md).

Tags: **[sym]** is an exact proof on symbolic coefficients, **[sym/ideal]** is an exact proof modulo
the ideal of a condition, **[prop]** is an f64 property test, **[doc]** is documentation only. The
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
  * **Pattern:** the product-like operations pick up `‖m‖²`. Operations that go through the
    complement (the regressive product `&`) pick up a sign under odd versors (reflections): for
    example `−‖m‖²` for a CGA vector versor.
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
  `+1` for even versors, and for odd ones whatever `x ↦ b x ~b` gives, since `>>` has no
  grade-involution sign.

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
    kind;
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
