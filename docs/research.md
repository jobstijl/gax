# Research log

This log lists the prior art studied before and during the design of `gax`: what each source
contributes and what we take from it. The detailed notes, with file and line citations,
measured op counts and verbatim formulas, are in the appendices:

* [research/codegen.md](research/codegen.md): GAmphetamine.js, kingdon, ganja.js, bivector.net conventions.
* [research/rust-crates.md](research/rust-crates.md): Rust GA crates, glam/ultraviolet/nalgebra/faer, SIMD options, mathbench.
* [research/papers.md](research/papers.md): papers (extensors, invariant decomposition, PGA dynamics, inverses, eigen/SVD).
* [research/typelevel.md](research/typelevel.md): stable-Rust type-level techniques, with compiled experiments in
  [research/experiments/](research/experiments/).

Licenses were checked for every project we learned from. We copy no code from any of them; where a
formula is taken from a paper it is cited at the implementation site.

## 1. numga (direct inspiration)

* **Source:** <https://github.com/EelcoHoogendoorn/numga>, `main` at `71779db` (2026-09-25). License MIT.
  Read: `docs/extensors.md`, `extensor_syntax.md`, `extensor_advanced.md`, `internals.md`, and the
  `algebra`, `gatype`, `operator` and `extensions` packages.
* **Contributes:**
  * **Extensor model:** an expression with open slots is a multilinear map. Its kernel is an exact integer
    table, output-first (`[batch..., O, I1, ..., In]`). Binding a value contracts a slot; binding a map
    contracts and splices that map's inputs in place of the slot.
  * **Identity maps and dyads:** a bare type is the identity map, a dyad is a product with an open slot,
    and a trace pairs the output with an input by matching blades, with no metric.
  * **Maps versus forms:** the pairings `&` (metric free) and `|` (metric) convert between them, and there is
    deliberately no transpose.
  * **Types carry traits:** a certified unit versor inverts by its reverse, and an orthogonal map by
    transposition. Equality groups (the same object bound into several slots) let tables be symmetrized.
  * **Extension methods** dispatch on types: inverse, solve, lstsq, eigh against a metric, generalized
    eigh between two forms, svd, exp/log, outermorphism.
  * **Specialized representations** with the same interface, such as `PrincipalInertiaPGA3` (4 numbers
    instead of a 6x6 map).
  * **Dense versus sparse execution:** dense contraction wins in NumPy. Unrolled sparse terms win for small
    expressions under JAX. Converting a sandwich to a matrix wins as soon as several objects are
    transformed (CGA trivector: 1600 terms against a 10x10 matrix).
* **We take:** the semantics and naming (slots, binding, lifting, trace, pairings, forms versus maps, typed
  eigenproblems), output-first layout, the exact tables, and traits on types. numga is also our
  cross-check oracle: it runs locally in a venv, and its exact tables and `formula()` output are exported
  as test fixtures.
* **We change:** numga decides slots, equality groups and traits at run time, with dispatch tables keyed
  on types. In Rust all of that moves into the type system, and each operation between two types becomes
  generated straight-line code. See [design.md](design.md).

## 2. Symbolic code generation for GA

### GAmphetamine.js

* **Source:** <https://github.com/enkimute/GAmphetamine.js> at `617ab83`.
* **License is ambiguous:** the MIT template has no holder filled in, and `package.json` says ISC. **We
  copy no code from it** and reimplement its ideas from the description in the appendix.
* **Contributes:** symbolic evaluation of each operator per type pair, as exact sparse rational
  polynomials. Then:
  * **type conditions** (`R~R = 1`), turned into rewrite rules;
  * **fixed coefficients** (for example, `e123 = 1`);
  * **a CSE portfolio:** shared residual sums, square-sum completion, perfect squares, differences, Horner
    isolation, and frequent pairs;
  * **reciprocal hoisting** with power reuse;
  * **lazy per-type-pair compilation.**
* **We take:**
  * the pipeline and the list of optimizations;
  * the cost model (mul ≈ add, div = 4);
  * **its measured op counts as regression targets**, for example:
    * a unit-motor sandwich of a point is 21 mul and 18 add;
    * an `even * even` product is 48 mul and 40 add;
    * a motor inverse under the unit condition is 0 mul.
* **We change:**
  * **Type conditions** become proper ideal reduction (normal forms modulo a Gröbner basis), not a
    non-confluent rewrite loop.
  * **CSE** is verified by construction: every rewrite is checked by polynomial re-expansion.

### kingdon

* **Source:** <https://github.com/tBuLi/kingdon>, MIT.
* **Contributes:**
  * binary blade keys with one-line product predicates;
  * a compile cache keyed by the type and present blades of each argument;
  * types defined by symbolic expressions, with numeric coefficients becoming fixed slots;
  * Hitzer–Sangwine inverses for d ≤ 5 and Shirokov's inverse beyond.
* **We take:** the product predicates for the derived products (wedge, contractions, dot, commutator),
  the rule that the output type is the most specific declared type containing the result, and the
  inverse cascade.

### ganja.js

* **Source:** <https://github.com/enkimute/ganja.js>, MIT.
* **Contributes:**
  * Cayley generation for any `Cl(p,q,r)`, with ordering conventions (null generators first, `e0`);
  * remapping of custom blade names such as `e31` and `e021`, with signs;
  * the bivector.net PGA2D/PGA3D layouts and their sign-free Poincaré dual.
* **We take:** the bivector.net layouts as the standard PGA layouts. The join sign convention is our own
  choice, see [design.md, ADR-009](design.md#adr-009).

### bivector.net cheat sheets and "May the Forque Be with You"

* **Sources:** <https://bivector.net/3DPGA.pdf>, <https://bivector.net/2DPGA.pdf>, and Dorst & De Keninck,
  *May the Forque Be with You* (<https://bivector.net/PGAdyn.pdf>).
* **Contributes:** the PGA2D/3D basis, point, plane and line conventions, and the rigid-body equations
  `Ṁ = −½ M B`, `Ḃ = I⁻¹[B × I[B] + F]`, with the inertia map `I = J·D`, where `J` is the dual. The appendix
  checked these numerically; momentum is conserved to 1e−12.

## 3. Rust prior art

The full survey is in [research/rust-crates.md](research/rust-crates.md). Summary:

| crate | approach | strength | where it stops |
|---|---|---|---|
| `geometric_algebra` (Lichtso) | codegen from signature, `f32` SIMD groups | many algebras | f32 only; the SSE2 build calls an AVX intrinsic out of line; the sandwich is two full products (94 instrs); unmaintained since 2023 |
| `emilk/pga` | symbolic codegen, types inferred from blades | clean idea | unpublished experiment; no sandwich kernels, no SIMD |
| `clifford` 0.3 | generated types, `num_traits::Float` | generic scalars | no CSE; no SIMD |
| `vee` (MPL-2.0) | symbolic reduction, op counting, 3x4 motor matrix | closest in spirit | MPL: ideas only |
| `garust`, `amari`, `ga3`, `wedged`, `tclifford` | dense or heap storage | simple | 10–50x slower than glam |

* **Gap we fill:** a Rust GA crate that is coefficient-generic (f32, f64, SIMD lanes, symbolic), fused by
  symbolic optimization, SoA-batched, and whose maps are first-class typed extensors.
* **Performance bar, from glam 0.33** (arithmetic instructions, AVX2+FMA):
  * `Quat * Vec3A` ≈ 27;
  * `Affine3A::transform_point3a` = 9;
  * `Mat4 * Vec4` = 11;
  * `Quat * Quat` = 19.
* **ultraviolet** `Rotor3x8::rotate_vec` takes about 5 ALU ops per rotation, the SoA target.
* **SIMD:**
  * `std::simd` is still nightly-only (E0658 on 1.98.1).
  * `wide` 1.7.1 (MSRV 1.89, Zlib/Apache/MIT) is the stable choice, behind a cargo feature.
  * `simba`'s `SimdRealField` and `ggmath` show the coefficient-trait pattern we follow in a lighter form.
* **Linear algebra:**
  * `nalgebra`'s small decompositions are scalar-only and iterative.
  * `faer` targets large matrices and says it is not for 4x4 to 6x6.
  * The hard constraint of our own math core is therefore cheap to honour: sizes are ≤ 32 and known per
    algebra.
* **Benchmark methodology:** follow mathbench-rs. It uses Criterion over a ring buffer of 2^13 random
  inputs with a "return self" baseline, at 1x and 100x sizes plus wide variants.

## 4. Papers

Formulas and verification scripts are in [research/papers.md](research/papers.md).

* **Hestenes & Sobczyk, *Clifford Algebra to Geometric Calculus* (1984), ch. 3.**
  * Contributes: extensors as multilinear functions of multivectors, the outermorphism, the adjoint,
    `det f = f(I) I⁻¹`, and `f⁻¹(X) = f̄(X I) I⁻¹ / det f`.
  * We take: the definitions and the vocabulary. With a degenerate metric the adjoint must be defined
    through the complement (`&`), not through `I⁻¹`, which is what numga does.
* **Roelfs & De Keninck, *Graded Symmetry Groups: Plane and Simple* (arXiv:2107.03771).**
  * Contributes: the invariant decomposition of a bivector into commuting simple parts, by radicals for
    k ≤ 4. It gives exp/log in any dimension.
  * We take: the generic exp/log path.
* **De Keninck & Roelfs, *Normalization, Square Roots, and the Exponential and Logarithmic Maps in
  Geometric Algebras of Less than 6D* (arXiv:2206.07496).**
  * Contributes: closed-form normalize, sqrt, exp and log for PGA3D motors, with op counts. All were
    verified in the bivector.net basis.
  * We take: the optimized PGA primitives, and their op counts as targets.
  * Warning: the exp/log listing on the "Look Ma No Matrices" page pairs `e01` with `e12`. As transcribed,
    that gives a wrong exp; use the paper's `e01↔e23` pairing.
* **Hitzer & Sangwine, Appl. Math. Comput. 311 (2017), DOI 10.1016/j.amc.2017.05.027.**
  * Contributes: closed-form inverses for n ≤ 5, via involution products.
* **Shirokov, Faddeev–LeVerrier inverse in Clifford algebras (arXiv:2005.04015).**
  * Contributes: an inverse in any dimension.
  * The appendix verified both of these inverses on degenerate metrics too (PGA3D, R(2,0,1), R(3,1,1)).
* **De Keninck, *Look, Ma, No Matrices!* (SIGGRAPH 2024 talk).**
  * Contributes: op counts for the sandwich against a matrix (motor on point: 21 mul / 18 add against
    16/12 for a 4x4 matrix times a vector).
  * We take: the crossover question of hypothesis 6, which we measure ourselves.
* **Dorst & De Keninck, *May the Forque Be with You* (v2.6, 2023).**
  * Contributes: PGA rigid-body dynamics, and inertia as a map from twists to forques.
  * We take: the inertia example and the compact principal-inertia representation.
* **McAdams et al. 2011, *Computing the SVD of 3x3 matrices with minimal branching*.**
  * Contributes: approximate Givens rotations and branchless Jacobi.
  * We take: the branchless style for SIMD batches.
* **Demmel & Veselić.**
  * Contributes: the relative accuracy of Jacobi.
  * We take: the choice of Jacobi over QR for small symmetric problems.
* **Kopp (2008) and Habera & Zilian (arXiv:2511.00292).**
  * Contributes: closed-form 3x3 symmetric eigenvalues, with failure cases.
  * We take: we use the closed form only where it is safe.
* **Generalized symmetric eigenproblem `K x = λ M x` via Cholesky** (`C = L⁻¹ K L⁻ᵀ`).
  * We take: vibration modes (`eigh(metric)`).
* **Other GA code generators:** GATL (C++ expression templates, compile-time lazy evaluation), Gaalop,
  Garamon, TbGAL, Versor, Gaigen 2, Klein (SSE PGA3D).
  * We take: confirmation that fusing a whole expression before emitting code is where the performance
    lies. GATL's C++ template approach is the closest to type-level fusion, but it pays for it heavily in
    compile time.
* **Fernández, Moya & Rodrigues, *Extensors in Geometric Algebras* (arXiv:math/0501558).**
  * Contributes: the extension operator (outermorphism), which numga follows.

## 5. Rust type-level techniques

Details and compiled experiments: [research/typelevel.md](research/typelevel.md).

* **variadics_please** (Bevy, MIT/Apache): per-arity tuple impls. We use the pattern with our own small
  internal `macro_rules`.
* **frunk** (MIT):
  * Contributes: HList concatenation, and the inferred-index `Selector`/`Plucker` trick. The index type
    parameter makes the impls disjoint and is left to inference; it resolves only for concrete lists.
  * We take: the idea of type-level indices, though we use const generic indices per tuple arity instead.
* **The key open problem is solved on stable:** with only `S: Slots`, `S ++ () = S` is provable. The
  concatenation lives in a GAT on a helper trait, `trait HasCat { type Cat<R: Slots>: Slots; }`, and
  `Slots` states the right-identity law as a supertrait equality: `trait Slots: HasCat<Cat<()> = Self>`.
  The compiler proves the law inductively for every list impl, and generic code gets it for free.
  Formulations that fail, with their errors, are recorded in the appendix.
* **Specialization workarounds:**
  * Impls disjoint only by an associated type value are rejected (E0119).
  * Autoref specialization does not work inside generic functions.
  * `TypeId` comparisons fold at `-O`, but they are not usable in const.
  * What works: case analysis as methods on `Slots` (one impl for `()`, one per non-empty tuple), which we
    use for value/map dispatch.
* **`generic_const_exprs`** is still unstable. Array lengths from associated consts (`[T; K::N]`) do not
  compile in generic code; nested arrays through GATs (`K::Arr<S::Arr<T>>`) avoid the problem.
* **Next trait solver:** the default on nightly since 2026-08, targeting stable 1.100. It changes none of
  the results above.

## 6. Own experiments (evidence for hypotheses 5 and 7)

Source: `docs/research/experiments/` and the notes below. All were run on rustc 1.98.1 with
`-C opt-level=3 -C target-cpu=x86-64-v3`.

1. **LLVM does not fully unroll a loop over a constant sparse term table beyond about 16 terms.**
   * Test: the loop `for t in TERMS { out[t.o] += s * a[t.i] * b[t.j] }`.
   * With 16 terms it becomes 32 FP ops and no branches.
   * With 20, 24, 32, 40, 48 or 64 terms it stays a runtime loop with 5 jumps.
   * A dense 8x8x8 table multiplies every zero in (144 FP ops for 64 nonzero terms).
   * So the "tier 1 = constant tables, LLVM unrolls" hypothesis fails for most GA products. Tier 1 must be
     emitted straight-line code.
2. **IEEE semantics:**
   * `x + 0.0` is not folded (`vaddss` with a zeroed register).
   * `x + (-0.0)` is folded to nothing.
   * `0.0 * x` is not folded (`vmulss`).
   * So structural zeros must never reach runtime arithmetic. They live in the generated code and the
     types. Hypothesis 7 is confirmed.
3. **Straight-line kernels generic over slot lists compile optimally.** Test: a cross-product-shaped kernel
   written once over `S1, S2: Slots` with nested GAT arrays.
   * value × value: 6 FP ops, SLP-vectorized, no loops;
   * map × value: 6 FP ops on 4-wide vectors;
   * map × map: 15 FP ops (vector), no branches.

   So one generated function per (operation, left kind, right kind) serves values, maps and forms alike.
