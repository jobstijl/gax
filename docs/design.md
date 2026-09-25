# Design

This document records the architecture of `gax` as a series of decisions. Each has a status:
*accepted*, *proposed* (planned, not yet backed by an implementation), or *superseded*. The nine
design hypotheses from the project brief are tracked in [§ Hypotheses](#hypotheses) with their
verdict and evidence. The prior art behind each decision is in [research.md](research.md).

> **Name.** `gax` ("geometric algebra, extensors") is a working name. The final crate name is to be
> chosen by the project owner before anything is published.

## Overview

```text
algebra description ──► gax-gen ──► exact tables ──► Rust source ──► gax (standard algebras, committed)
   (text / macro)        │                                         └─► algebra! (user algebras, proc macro)
                         └─► Sym (symbolic coefficient) ──► simplifier ──► fused kernels
                                                                           ├─ tier 2: sandwich, to_matrix, norms, inverse, exp/log
                                                                           └─ tier 3: user kernels traced in build.rs
gax-core: Slots, Kind, Coef/Real, binding machinery, linear-algebra core (no_std)
```

## Crates

| crate | role | depends on |
|---|---|---|
| `gax-core` | slot lists, kinds, coefficient traits, binding and composition machinery, math core; `no_std` | none (`libm` without `std`, `wide` optional) |
| `gax-gen` | algebra → exact tables and typed layouts; symbolic polynomials (`Sym`); simplifier; Rust emission; the regenerate tool | `gax-core` |
| `gax-macros` | the `algebra!` proc macro: parses the declaration and calls `gax-gen` | `gax-gen` |
| `gax` | facade: pre-generated standard algebras behind features, re-exports, the tracing API (feature `trace`) | `gax-core`, `gax-macros`, (`gax-gen` with `trace`) |
| `gax-bench` | criterion benchmarks against glam, ultraviolet, nalgebra; `publish = false` | all |

---

## ADR-001: Workspace of core, generator, macro and facade crates
*Status: accepted.*

This follows the brief's suggested architecture. The generator is a plain library, so three consumers
share one implementation:

1. the proc macro;
2. build-time tracing, which runs in `build.rs`;
3. the regeneration tool that writes the committed standard algebras.

Users of the standard algebras compile no proc macro: the pre-generated algebras are ordinary source
files in `gax/src/algebras/`, behind cargo features.

## ADR-002: MSRV 1.89, edition 2024
*Status: accepted.*

* **Why 1.89:**
  * the design needs GATs (1.65), associated type bounds (1.79) and edition 2024 (1.85);
  * `wide` 1.7, which provides the SIMD lanes, requires 1.89;
  * 1.89 is thirteen months old at the time of writing.
* **Enforcement:** CI builds and tests on 1.89 and on stable.

## ADR-003: Slot lists are tuples, with the right identity stated as a trait law
*Status: accepted. Evidence: [research/typelevel.md](research/typelevel.md) §2.5, §2.8.*

* **Spelling:** a multivector type is generic over its open slots, `Point<S>`:
  * `Point<()>` is a value;
  * `Point<(Point,)>` is a map;
  * `Scalar<(Twist, Twist)>` is a bilinear form.

  Slot lists are tuples because the brief allows no macros in user code, which rules out a
  `slots![..]` type macro.
* **Operators:** a binary operation between `A<S1>` and `B<S2>` returns `C<Cat<S1, S2>>`.
* **The open problem, `S ++ () = S` for generic `S`,** is solved by keeping concatenation in a GAT on a
  helper trait:

  ```rust
  pub trait HasCat { type Cat<R: Slots>: Slots; }
  pub trait Slots: HasCat<Cat<()> = Self> + Copy + 'static { ... }
  ```

  * The supertrait equality lets any generic `S: Slots` normalize `Cat<S, ()>` to `S`.
  * The compiler checks that every tuple impl satisfies it.
  * `Cat<(), S>` normalizes to `S` directly from the `()` impl.
  * So `fn f<S: Slots>(p: Point<S>) -> Point<S> { (LIGHT & p) ^ GROUND }` compiles with no other bound.
* **Non-empty tuples:** these define `Cat` through a `Prepend` GAT. The nested array type then needs a
  `prepend_arr` witness, which is the identity at run time.
* **Maximum length:** tuples have a maximum length of 8. A longer concatenation maps to a sentinel type
  whose witness fails in a `const` block, so it is a compile error only if it is actually used.
* **Associativity** (`Cat<Cat<A,B>,C>` against `Cat<A,Cat<B,C>>`) is not known for generic lists. It only
  matters when a generic signature brackets differently from the expression that produces it. A
  zero-cost `reassoc` witness method covers that case.

## ADR-004: One struct per kind; the bare type name is the slot marker
*Status: accepted.*

* **One struct per kind:** each typed subspace (kind) of an algebra is its own generic struct,
  `pub struct Point<S: Slots = (), T: Coef = f32> { pub c: [S::Arr<T>; 4] }`.
* **Canonical slot marker:** the canonical instantiation `Point` (that is, `Point<(), f32>`) implements
  `Kind`, so the element of a slot list is written exactly as the bare type name: `Point<(Point,), f64>`.
  Writing `Point<(), f64>` in a slot list is rejected with "not a `Kind`", which keeps one spelling per
  slot type.
* **Default coefficient `f32`:** this matches glam and the graphics use case.
* **Why not one generic `Mv<K, S, T>` struct with aliases:** error messages would print
  `Mv<PointKind, (Mv<PointKind, (), f32>,), f32>`. Readable types matter for a type-heavy API.
* **What it costs:** library-internal generic machinery goes from a kind to its struct family through a
  GAT, `Kind::Mv<S, T>`, and from a struct back to its kind through `Multivector::Kind`.
* **Arithmetic:** the standard operators are implemented per struct by the generator, since the orphan
  rule forbids a blanket impl.

## ADR-005: Output-first nested arrays through GATs
*Status: accepted (hypothesis 1, storage part).*

* **Layout:** coefficients are stored output-first. Each output coefficient is an array over the slots,
  `[S::Arr<T>; N]`, where
  * `(A, B)::Arr<T> = A::Arr<B::Arr<T>>`, and
  * each kind sets `Arr<X> = [X; n]` with a literal `n`.

  This needs no `generic_const_exprs`, and concrete types normalize fully: a `Scalar<(Twist, Twist)>`
  stores `[[[f32; 6]; 6]; 1]`.
* **Why output-first:** it makes a map's column for one output coefficient a single slot array. That is
  what the generic outer product in the generated kernels works on.

## ADR-006: Tier 1 is emitted straight-line code, generic over slots
*Status: accepted. Changes hypothesis 5 (tier 1).*

* **What the experiment showed:** LLVM fully unrolls a loop over a constant term table only up to about
  16 terms ([research.md §6](research.md#6-own-experiments-evidence-for-hypotheses-5-and-7)). Most GA
  products are larger, and a dense table multiplies structural zeros in.
* **What the generator does instead:** for each (operation, left kind, right kind) with a nonzero result,
  it emits one function. The function is written with generic slot-array helpers:
  `out[o] = mul(a[i], b[j]) + ... - ...`, where `mul` is the outer product over the two slot lists.
* **One function serves values, maps and forms:**
  * for values it compiles to exactly the scalar products, with no zeros and no loops;
  * for maps it compiles to vector loops over the slot arrays.
* **The accumulator** starts from the first term, never from `0.0`.
* **Code size:** generic functions are compiled only when used, so code size is paid only per use.
  Compile time is dominated by parsing and type-checking the generated impls. It is tracked in
  [performance.md](performance.md).

## ADR-007: Structural zeros live in code and types, never in runtime data
*Status: accepted. Confirms hypothesis 7.*

* **IEEE semantics:** under IEEE rules `x + 0.0` and `0.0 * x` are not folded, and `x + (-0.0)` is. So:
  * no generated kernel ever touches a coefficient that is structurally zero;
  * a kernel that does start from a zero accumulator starts from `-0.0`.
* **Where the zero pattern lives:** it is carried by kinds (the blade set of each type) and by the
  generated code of specialized kernels, such as the sparsity of a motor's point matrix.

## ADR-008: Algebras with any integer metric, via the Chevalley recursion
*Status: accepted. Evidence: `gax-gen/tests/oracle.rs`.*

* **Products:** blade products are computed in the wedge basis by the Chevalley recursion
  `(e_i ∧ A') X = e_i (A' X) − (e_i ⌋ A') X`. This holds for any symmetric integer bilinear form: diagonal,
  degenerate (PGA) or non-diagonal (the CGA null basis `eo·ei = −1`).
* **Every coefficient is an integer,** so all tables are exact.
* **The oracle** is an independent naive implementation: rational congruence diagonalization, then the
  bitmask rule, then an outermorphism back. It checks every blade pair of PGA2D, PGA3D, VGA3D, STA, STAP,
  CGA2D, CGA3D, CSTA and a mixed non-diagonal metric.
* **Naming:** basis vectors are named `e` plus one character (`e0`, `e1`, `eo`, `ei`), and a blade is
  written by listing its factors in any order. `e031` is `−e013`.
* **Layouts:** a kind is a layout, a list of oriented blades, which reproduces the bivector.net layouts
  (`e31`, `e021`, `e032`) exactly.

## ADR-009: Products and duality conventions
*Status: accepted.*

* **Operators:**
  * `*` geometric product;
  * `^` wedge;
  * `&` regressive (vee);
  * `|` symmetric ("fat dot") inner product;
  * `<<` and `>>` versor transport (see ADR-013).
* **Named methods:** `lc`, `rc`, `scalar`, `commutator`, `anticommutator`.
* **The regressive product** is `a ∨ b = J_L(J_R a ∧ J_R b)`, with `J_R` the right complement
  (`a ∧ J_R a = I`) and `J_L` its inverse (Lengyel's antiwedge).
  * It is metric free, so it works in PGA and in null bases.
  * It gives `e123 ∨ e032 = +e23`, like GAmphetamine and kingdon. ganja and bivector.net have the
    opposite sign for this pair.
* **Duality:** `dual()` is `J_R` and `undual()` is `J_L`. Both are metric free.
* **Standard PGA layouts:** these are the bivector.net layouts:
  * PGA3D: `1,e0,e1,e2,e3,e01,e02,e03,e12,e31,e23,e021,e013,e032,e123,e0123`;
  * PGA2D: `1,e0,e1,e2,e01,e20,e12,e012`.

  Types are subsets of these, so a PGA3D point is `[e032, e013, e021, e123]`, as in the cheat sheet.
* **Plane before point:** as numga recommends, pairings are written plane first (`plane & point`), and
  the examples follow this.

## ADR-010: The symbolic coefficient `Sym` is a `Copy` handle to hash-consed polynomials
*Status: accepted.*

* **Why `Copy`:** tracing runs ordinary generic user code with `T = Sym`, so `Sym` must satisfy the same
  `Coef` bound as `f32`, including `Copy`.
* **What it is:** `Sym` is a `u32` handle into a thread-local arena of expanded polynomials with exact
  rational coefficients. The arena is hash-consed, so two handles are equal exactly when their polynomials
  are.
* **Non-polynomial operations** (`recip`, `sqrt`, `sin`, `cos`, ...) create *atoms*: fresh variables that
  carry their definition. Because atoms are hash-consed, a reciprocal of the same denominator is created
  once. That is reciprocal hoisting at the source.
* **Branches cannot be traced.** `select_lt` on `Sym` panics with a message that names the kernel, so a
  traced kernel must be branch free.

## ADR-011: The simplifier: ideal reduction plus a verified CSE portfolio
*Status: proposed (GAmphetamine's pipeline, made sound).*

1. **Type conditions.** A kind can carry conditions: `unit` means `x ~x = 1`, and fixed coefficients such
   as `e123 = 1` can also be declared. The tracer turns the conditions of each input into polynomial
   identities, and every output coefficient is reduced to a normal form modulo their ideal: a reduced
   Gröbner basis, which is tiny for these cases. Square-sum completion is the cost-driven choice between
   equivalent representatives (`a² + b² − c² − d²` against `1 − 2(c² + d²)`).
2. **CSE.** Several strategies run and the cheapest result wins under the cost model
   `mul = add = 1, div = 4`:
   * factoring by the linear "passenger" variables (the matrix-then-apply shape);
   * shared residual sums;
   * Horner isolation;
   * greedy frequent-pair extraction.

   Every rewrite is exact by construction, and the emitted DAG is re-expanded and compared with the input
   polynomials in tests.
3. **Regression targets** are GAmphetamine's measured op counts
   ([research/codegen.md §1.10](research/codegen.md)).

## ADR-012: The value/map tier dispatch is a method on `Slots`
*Status: accepted. Changes the mechanism of hypothesis 5.*

* **What didn't work:** stable Rust has no specialization, and impls that are disjoint only by an
  associated type value are rejected.
* **What the library does instead:** the case analysis is a method on `Slots` with one impl for `()` and
  one per non-empty tuple:
  * `as_value(&S::Arr<X>) -> Option<X>`;
  * `from_value(X) -> Option<S::Arr<X>>`.
* **How a kernel uses it:** it tries the value path first. For `()` the `Some` branch is taken, and for a
  tuple the `None` branch; both fold away at compile time.
* **Scope:** this chooses between the direct sandwich (values) and matrix-then-apply (maps) with no bound
  beyond `S: Slots`.

## ADR-013: Versor transport: `v >> x` pushes forward, `v << x` pulls back
*Status: proposed.*

* **The operators:** `v >> x` is `v x ~v`, following numga, and `v << x` is `~v x v`.
* **Value versor:** when `v` is a value, `>>` dispatches (ADR-012) as follows:
  * a value `x` uses the tier-2 fused sandwich;
  * a map `x` uses the tier-2 `to_matrix(v)` and composes it into the output of `x`.
* **Open versor:** when `v` has open slots, `>>` falls back to tier-1 products.
* **The matrix itself:** `to_matrix()` is `v >> X::slot()`, and is generated directly with its structural
  zeros known. Applying a runtime `Point<(Point,)>` is dense; the crossover against the direct sandwich
  is measured (hypothesis 6).

## ADR-014: Binding and composition API
*Status: proposed.*

* **`m.of(x)`:** binds the first slot. `x` can be a value, which fills the slot, or a map, which composes
  it: the slot is replaced by `x`'s own slots, spliced in place.
* **`m.at::<I>()`:** moves slot `I` to the front. `m.at::<1>().of(x)` binds slot 1.
* **`.swap()`:** exchanges the two slots of a form.
* **`m.trace::<I>()`:** contracts the output against input slot `I`, which must be the output's own kind.
  This is a blade-matching contraction with no metric (numga's `trace(slot)`). A contraction between two
  *inputs* needs a pairing and is written with `&` or `|`. That is why the brief's
  `.trace::<I, J>()` is not provided.
* **`m.adjoint()`:** for a map `B <- A`, returns the map `A* <- B*` that satisfies
  `adjoint(y) & x == y & m(x)`, where `A*` is the complement kind (Plane for Point). For the complement
  pairing this is a signed transpose, so no solve is needed.

## ADR-015: Equal slots are grouped by kind by default
*Status: proposed. Changes hypothesis 3.*

* **Why not labels per call:** Rust cannot create a fresh type for each call of `Motor::slot()`.
  Type-level labels would need either a closed family of label types or type equality, which stable Rust
  lacks for arbitrary types.
* **What the generator provides instead:** algebras are closed families, so the generator emits
  type-level kind equality (`KindEq`) for each pair of kinds.
* **`m.fill(x)`:** binds *every* slot of `x`'s kind. It is the natural equality group, as in
  `Point<(Motor, Point, Motor)>.fill(motor)`.
* **Explicit labels** (`Motor::labelled::<L1>()`, from a small provided family `L1..L8`) refine this when
  two slots of the same kind must stay independent.
* **Where the performance comes from instead:** symmetrizing tables over equal slots is left to tier 2
  and tier 3. There the symbolic tracer sees the repeated variables directly, which subsumes it.

## ADR-016: Build-time tracing through a shared module
*Status: proposed (hypothesis 4).*

* **Where the user writes kernels:** in an ordinary module, `src/kernels.rs`. It holds generic functions
  over `T: Coef` and `S: Slots`.
* **The build script:**
  * includes that same file with `#[path = "src/kernels.rs"] mod kernels;`;
  * instantiates the functions with `T = gax::trace::Sym`;
  * simplifies them;
  * writes `fused.rs` to `OUT_DIR`.
* **The crate** uses `include!(concat!(env!("OUT_DIR"), "/fused.rs"))`.
* **Why this shape:** no user macro is needed, and the traced function is the very function the crate
  compiles, so they cannot drift apart.
* **Type conditions** come from the input kinds (for example, `Motor` is `unit`), with an explicit
  `.assume_unit()` override.

## ADR-017: Own small-matrix math core, generic over `Real`
*Status: proposed (hypothesis 8).*

* **Solvers:** unrolled Cholesky/LDLᵀ, partial-pivot LU, cyclic Jacobi eigen and one-sided Jacobi SVD.
  They are written over `[[T; N]; N]` with const `N`; sizes are small and known per algebra.
* **Branch free:** all use `Real::select_lt` instead of branches, so they batch over SIMD lanes. Jacobi
  runs a fixed number of sweeps.
* **Glue:** each kind gets generated glue that calls them with its literal `N`.
* **Closed forms by type:**
  * a unit versor inverts by its reverse;
  * a versor's matrix inverts by the matrix of the reverse;
  * principal inertia has a compact representation.
* **Generalized symmetric eigenproblems** go through a Cholesky reduction.
* **exp/log:** closed forms for PGA2D/3D (De Keninck & Roelfs 2022), and the invariant decomposition
  elsewhere.

## ADR-018: Batching through SIMD coefficient types
*Status: proposed (hypothesis 9).*

* **Lanes are coefficients:** `wide::f32x8` and `f64x4` implement `Coef` and `Real` (feature `wide`), so
  `Point<(), f32x8>` is eight points in SoA form. Every kernel, including solvers, runs unchanged.
* **Slices:** helpers convert between `&[Point<(), f32>]` and lane chunks.

---

## Hypotheses

| # | hypothesis | verdict | evidence / decision |
|---|---|---|---|
| 1 | Types generic over open slots; ops written once; `S ++ () = S` for generic `S` | **kept** | ADR-003, ADR-005; compiled experiments e11, e14 |
| 2 | Plain generic functions are the composition language | **kept** | ADR-014; API names adjusted (`trace::<I>`, `adjoint` via complement) |
| 3 | Labelled slots record equality groups | **changed** | ADR-015: equality by kind (`fill`), with explicit labels from a closed family |
| 4 | Build-time tracing with a symbolic coefficient type | **kept** | ADR-010, ADR-016 |
| 5 | Three performance tiers | **changed** | Tier 1 is emitted straight-line code, not constant tables (ADR-006); dispatch by `Slots` methods (ADR-012) |
| 6 | Build the map, then apply it | pending | ADR-013; crossover measured in `gax-bench` |
| 7 | IEEE float semantics limit folding | **confirmed** | ADR-007; asm evidence in research.md §6 |
| 8 | Own solvers with closed forms by type | pending | ADR-017 |
| 9 | Batching via SoA and SIMD coefficients | pending | ADR-018 |
