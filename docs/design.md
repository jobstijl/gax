# Design

This document records the architecture of `gax` as a series of decisions. Each has a status:
*accepted*, *proposed* (planned, not yet backed by an implementation), or *superseded*. The nine
design hypotheses from the project brief are tracked in [§ Hypotheses](#hypotheses) with their
verdict and evidence. The prior art behind each decision is in [research.md](research.md).

> **Name.** `gax`: "geometric algebra, extensors". Licensed MIT OR Apache-2.0.

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
* **Pseudoscalar orientation differs from numga.** The PGA3D pseudoscalar is oriented as in
  bivector.net, `e0123`. numga orients it as `x∧y∧z∧w = −e0123`. The regressive product and the
  complement flip sign with that orientation, so in PGA3D `gax`'s `a & b` and `dual` are the negatives
  of numga's; in PGA2D they agree. The numga fixture test (`tests/numga_fixtures.rs`) pins this.

## ADR-010: The symbolic coefficient `Sym` is a `Copy` handle to hash-consed polynomials
*Status: accepted, implemented (`gax-gen/src/sym.rs`).*

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
* **Bounded expansion (added during implementation).** Expanding a whole chain of products into flat
  polynomials destroys its factored structure. The first version made the fused kernel for "move,
  join, meet" cost 163 mul against 49 for the generic code. A sum or product whose expanded result
  exceeds a term *limit* therefore becomes an opaque node variable defined by its operation, so the
  trace keeps its DAG structure.
  * The tracer runs the closure under the limits `0, 1, 8, 32, 128, ∞`, compiles each result, and keeps
    the cheapest verified program.
  * Limit 0 is the computation as written, with constants folded and common subexpressions shared, so
    a fused kernel is never worse than the generic code.
* **Runtime model.** A second trace with every operation a node and every constant opaque gives the
  cost of the generic code *as it runs*: IEEE forbids folding `0 * x`. This is the baseline the tracer
  reports against.

## ADR-011: The simplifier: ideal reduction plus a verified CSE portfolio
*Status: accepted, implemented (`gax-gen/src/cse.rs`, `groebner.rs`).*

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
4. **What was built:**
   * **Kernel extraction.** A Brayton–McMullen style pass looks for sums shared across outputs under
     different co-kernels, up to a rational factor, trying every subset of each residual. It brought
     the unit motor–point sandwich from 39 mul / 30 add to 25 / 18. GAmphetamine's 21 / 18 is for a point
     with the fixed coefficient `e123 = 1`; in `gax`, fixed coefficients come from constants in traced
     code.
   * **Three reduction strategies** compete in the portfolio: none, greedy term reduction (square-sum
     completion), and the normal form modulo a Gröbner basis (Buchberger, grevlex).
   * **Exact verification.** Every winning program is expanded back to polynomials. Its difference from
     the traced polynomials must have Gröbner normal form zero, modulo the type conditions and the atom
     identities `r·x = 1` and `s² = x`.

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
*Status: accepted, implemented.*

* **The operators:** `v >> x` is `v x ~v`, following numga, and `v << x` is `~v x v`.
* **Value versor:** when `v` is a value, `>>` dispatches (ADR-012) as follows:
  * a value `x` uses the tier-2 fused sandwich;
  * a map `x` uses the tier-2 `to_matrix(v)` and composes it into the output of `x`.
* **Open versor:** when `v` has open slots, `>>` falls back to tier-1 products.
* **The matrix itself:** `to_matrix()` is `v >> X::slot()`, and is generated directly with its structural
  zeros known. Applying a runtime `Point<(Point,)>` is dense; the crossover against the direct sandwich
  is measured (hypothesis 6).

## ADR-014: Binding and composition API
*Status: accepted, implemented.*

* **`m.of(x)`:** binds the first slot. `x` can be a value, which fills the slot, or a map, which composes
  it: the slot is replaced by `x`'s own slots, spliced in place.
* **`m.at::<I>()`:** moves slot `I` to the front. `m.at::<1>().of(x)` binds slot 1.
* **`.swap()`:** exchanges the two slots of a form.
* **`m.trace_at::<I>()`:** contracts the output against input slot `I`, which must be the output's own kind.
  For a map from a kind to itself, `trace()` is the matrix trace.
  This is a blade-matching contraction with no metric (numga's `trace(slot)`). A contraction between two
  *inputs* needs a pairing and is written with `&` or `|`. That is why the brief's
  `.trace::<I, J>()` is not provided.
* **Adjoints: no `m.adjoint()` method.** An adjoint is written as numga writes it, as a pairing solve.
  For example, `(Plane::slot() & Point::slot()).solve(Plane::slot() & t)` is the map on planes induced by
  a point map `t`, and it satisfies `induced(l) & p == l & t(p)`. It exists even when `t` is singular.
  * `Pairing::solve` accepts a right-hand side with leading slots, which become slots of the solution.
  * We dropped the dedicated method because the pairing has to be named anyway (`&` or `|`), and the
    solve says which one.
* **Maps (`K<(A,), T>`):** `inverse`, `det`, `solve` (right-hand sides keep their slots), `svd` (typed
  singular vectors) and `trace`.
* **Forms (`Scalar<(A, A)>`):** `eigh_with(metric)`, which returns the modes as values of the slot kind,
  and `eigh`.
* **Outermorphisms:** `t.outermorphism::<B>()` extends a map on vectors by `∧`, or a map on antivectors
  (PGA points) by `∨`, to every homogeneous kind `B`. The generator derives the extended matrices
  symbolically, as minors; on the top grade the result is the determinant. This is numga's extension
  operator, after Fernández, Moya & Rodrigues.
* **Values:** `inverse`, `normalized` (returns `Unit<K>`), `norm`, `exp`, `Unit<K>::log` and `sqrt`.
  Each is emitted only where the kind has the structure its closed form needs (ADR-019).

## ADR-015: Equal slots are grouped by kind by default
*Status: accepted, implemented; explicit labels not implemented. Changes hypothesis 3.*

* **Why not labels per call:** Rust cannot create a fresh type for each call of `Motor::slot()`.
  Type-level labels would need either a closed family of label types or type equality, which stable Rust
  lacks for arbitrary types.
* **What the generator provides instead:** algebras are closed families, so the generator emits
  type-level kind equality (`KindEq`) for each pair of kinds.
* **`m.fill(x)`:** binds *every* slot of `x`'s kind. It is the natural equality group, as in
  `Point<(Motor, Point, Motor)>.fill(motor)`.
* **Explicit labels, not implemented.** The plan was `Motor::labelled::<L1>()` from a small provided
  family `L1..L8`, for when two slots of the same kind must stay independent. Every use case so far is
  served by kinds plus `at::<I>()`, which binds one slot of a kind while leaving the others open. The
  machinery (`KindEq`, `FillList`) extends to a label family unchanged if one is needed.
* **Where the performance comes from instead:** symmetrizing tables over equal slots is left to tier 2
  and tier 3. There the symbolic tracer sees the repeated variables directly, which subsumes it.

## ADR-016: Build-time tracing through a shared module
*Status: accepted, implemented (`examples/traced`).*

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
* **Type conditions.** A `Unit<K>` argument contributes `x ~x = 1`. The tracer computes it by running the
  library's own generated products on symbolic coefficients, so it needs no algebra metadata.
* **Measured.** In the example, "move a point by a motor, cast its shadow on the floor `z = 0`":
  * fused: 31 mul / 21 add;
  * the generic code at run time: 46 / 32.

  Without build-time constants the fused kernels match the generic code, which is the guarantee the
  limit-0 trace gives.
* **Arithmetic is not wall time.** For the rigid-body step with constants traced, the fused kernel
  does 98 mul and 1 div against the generic code's 113 and 5. On eight bodies in `f32x8` lanes it is
  26% faster. On one body in f32 it is 35% *slower*, because LLVM's SLP vectorizer handles the regular
  generic code better. Hypothesis 4 holds for batched and scalar execution, not for single values on
  SIMD targets. performance.md has the details.

## ADR-017: Own small-matrix math core, generic over `Real`
*Status: accepted, implemented (`gax-core/src/linalg.rs`, `extensor.rs`).*

* **Solvers:** unrolled Cholesky/LDLᵀ, partial-pivot LU, cyclic Jacobi eigen and one-sided Jacobi SVD.
  They are written over `[[T; N]; N]` with const `N`; sizes are small and known per algebra.
* **Branch free:** all use `Real::select_lt` instead of branches, so they batch over SIMD lanes. Jacobi
  runs a fixed number of sweeps.
* **No glue needed.** The solvers are generic over `SquareArr`, which is implemented for `[[T; N]; N]`.
  The extensor methods require `Coeffs<Self>: SquareArr<T>`, so a non-square map simply has no
  `inverse()`: a compile-time shape check, with no per-kind glue and no `generic_const_exprs`.
* **Closed forms by type:**
  * a unit versor (`Unit<K>`) inverts by its reverse (implemented);
  * a versor's matrix inverts by the matrix of the reverse, `m << X` (implemented; the inverse test
    checks that it equals the LU inverse);
  * a compact principal-inertia representation (not yet implemented).
* **Generalized symmetric eigenproblems** go through a Cholesky reduction.
* **exp/log:** closed forms for PGA2D/3D (De Keninck & Roelfs 2022), and the invariant decomposition
  elsewhere.

* **Scalars may branch (added later).** `Real::SCALAR` is `true` for `f32` and `f64`, and the LU
  then pivots with ordinary row swaps instead of lane-wise selects. That and shared pivot
  reciprocals make the 6x6 inverse 1.7x faster than nalgebra's (it was 1.4x slower).

## ADR-018: Batching through SIMD coefficient types
*Status: accepted, implemented. See [performance.md](performance.md).*

* **Lanes are coefficients:** `wide::f32x8` and `f64x4` implement `Coef` and `Real` (feature `wide`), so
  `Point<(), f32x8>` is eight points in SoA form. Every kernel, including solvers, runs unchanged.
* **Slices:** helpers convert between `&[Point<(), f32>]` and lane chunks.
* **Runtime dispatch** of the same kernels, independent of `wide`, is ADR-023.

## ADR-019: Exponentials, logarithms, inverses and normalization through Study numbers
*Status: accepted, implemented (`gax-core/src/study.rs`, `gax-gen/src/emit_values.rs`).*

* **The common structure.** In an algebra of dimension at most 4, `x ~x` of a versor and `B²` of a
  bivector are Study numbers `a + bI`, with `I²` in `{0, +1, −1}`. Every analytic function extends to
  them through two channels:
  * real `a ± b` when `I² = 1`;
  * complex `a + ib` when `I² = −1`;
  * dual `f(a) + b f'(a) I` when `I² = 0`.
* **Written once.** `exp` (`C = cosh√x`, `S = sinh√x/√x`), `log` (`2 atanh(t)/(t(1+c))` with
  `t = √u/(1+c)`, stable near the identity), `rsqrt` and the inverse are written once over complex and
  dual-complex numbers. They cover PGA (dual), STA boosts and rotations (complex) and Euclidean rotors
  alike.
* **Generation checks the structure first.** The generator verifies it symbolically per kind and emits
  the method only when it holds, so a kind without it has no method.
* **5D (added later).** In CGA3D and STAP, `B²` is a scalar plus a 4-vector `Q` with several
  components, but `Q² = q` is still a scalar. `gax_core::study::study_q` evaluates functions of such
  a pair through a complex root `w = √q`:
  * `f₀ = (f(a+w) + f(a−w))/2` and `f₁ = (f(a+w) − f(a−w))/(2w)`;
  * duals near `q = 0`.

  exp is `c₀ + c₁Q + (s₀ + s₁Q)B`. log recovers `B² = acosh(C)²` from the versor's scalar-plus-4-vector
  part and divides out `S(B²)`. Both are tested as round trips: CGA3D `Bivector ↔ Even` and
  `Twist ↔ Motor`, and STAP.
* **Rotation fast paths.** When the generator proves the scalar part of `B²` is non-positive (minus a
  sum of squares), it emits real-trigonometric closed forms instead. See performance.md.
* **6D and up (CSTA).** A bivector there splits into three commuting parts, so no closed form is
  generated. `exp` falls back to scaling and squaring in the smallest kind closed under the product:
  a Taylor series of degree 8 on `B / 256`, then eight squarings. It is correct in any algebra, and
  the CSTA test checks it against the exact rotation and for `exp(B)·exp(−B) = 1`.
* **Still open: 6D log.** It needs either the cubic invariant decomposition or square roots of
  versors, and the normalization those need is not closed form in 6D.

## ADR-020: Certification by a wrapper type, `Unit<M>`
*Status: accepted.*

* **The choice.** numga attaches traits such as "certified unit versor" to its runtime types. Here the
  certificate is a wrapper, `Unit<M>`, in the style of `nalgebra::Unit`, rather than a second kind with
  the same blades.
* **What `Unit` provides:**
  * its inverse is its reverse;
  * the product of units is a unit;
  * its sandwiches use kernels simplified with `x ~x = 1`;
  * tracing adds the condition to the ideal.
* **Why a wrapper:** kinds stay plain subspaces, and the number of generated pairs does not double.

## ADR-021: Output kinds
*Status: accepted.*

* **The rule.** The output kind of a product is the declared kind that contains the support of the
  result, chosen as follows:
  1. prefer one that adds no grade the result lacks (a bivector stays a bivector kind);
  2. then the smallest;
  3. then the first declared.
* **Structural zeros.** Rounding a result up to a larger kind introduces zero coefficients. These are
  the only structural zeros that reach runtime data. Fused kernels (tier 2 and 3) avoid them.

## ADR-022: User algebras through `algebra!`, with the generator optimized at build time
*Status: accepted.*

* **One text format.** `algebra!` parses the same format as the committed `.gax` files and emits the
  same code. Missing `Scalar` and `Multivector` kinds are added in canonical order, so a user only
  declares the kinds they want.
* **Build time.** The generator runs inside the proc macro. Unoptimized it is slow, so the workspace sets
  `[profile.dev.build-override] opt-level = 3`, and the macro documents the same setting for users.
  Re-expanding STAP (5D) and CSTA (6D) then takes about 6 s.

## ADR-023: Runtime-dispatched batch kernels (`batch`)
*Status: accepted, implemented (`gax-core/src/batch`). See [batch.md](batch.md).*

* **Goal.** One binary that uses AVX2 and FMA where the CPU has them, without
  `-C target-cpu=native`, for the kernels gax already generates.
* **Dispatch.** `fearless_simd` 1.0 detects the level and compiles each kernel once per level: SSE2,
  SSE4.2, AVX2 or AVX-512 on x86, NEON on Arm, SIMD128 on WebAssembly, and portable code otherwise.
  `batch::run` takes a `Kernel`, whose `run<L>` method is generic over the lane type, and calls it
  inside `Simd::vectorize` for the detected level.
* **Lane types are coefficients.** The generated kernels are generic over `T: Coef`, so nothing
  is generated twice:
  * on detected levels, the lanes are `fearless_simd` vectors (`f32x8`, `f64x4`);
  * the portable lanes are plain arrays.
* **Soundness.** A `fearless_simd` vector needs its level's token, and `Coef::zero()` has no
  argument to take one from. The lane types therefore carry a proof type, `Proof<S>`, that is
  private to gax-core. Only the dispatcher instantiates it, after detection. User code can reach
  the lane types only as the parameter `L` of `Kernel::run`, so a lane value cannot exist before
  the level has been detected in the process. Their constructors then assume the token.
* **Inlining is the whole game.** A function that is not inlined into the dispatched function is
  compiled for the baseline, and every vector operation in it becomes a call. So:
  * everything on the hot path is `#[inline(always)]`;
  * generated code avoids `core::array::from_fn` and `map` there, spelling the arrays out, and uses
    loops in `slots::values`;
  * large functions (exp, log, the solvers) wrap their bodies in `Coef::vectorize`, which re-enters
    the level's target features when the compiler keeps them out of line (`fearless_simd`'s own
    remedy).

  Before these changes, batched `exp` on AVX2 was 8× slower than on SSE2.
* **Layouts.**
  * `Soa<K>` stores blocks of 16 values, coefficient-major within a block (AoSoA), so loads sit at
    offsets known at compile time with no bounds checks. Plain columns were 2× slower on SSE.
  * The array-of-structs forms transpose through small buffers: scalar moves plus one vector load
    per coefficient. Chains of lane inserts were slower than scalar code.
  * Remainders go through padded buffers, so the kernel body is inlined once and always sees full
    batches.
* **Sandwiches.** `SandwichKernel<X, Plain | Certified>` is implemented per pair on the versor's kind
  marker by generated code (the orphan rule rules out a shared marker type). `BatchTransform`
  builds the uniform (prepared once) and per-element forms on it, for both layouts.
* **Traced kernels.** `Tracer::batch(true)` emits two forms. `name_batch(&a0, ..., &mut out)`
  gathers lanes, calls the fused `name::<L>` and scatters the results. `name_batch_soa` loads
  arguments of one kind from `Soa` storage, for kernels whose result is a value of one kind. Both
  broadcast length-1 arguments. Tracing now supports `select_lt` as a data-flow stage, so `exp` can be traced.
* **Elementary functions.** `f32` lanes have vectorized sin, cos, sinh, cosh, atan2 and ln (Cephes
  polynomials, Cody–Waite reduction). They match the scalar `batch::math` versions bit for bit,
  and those are within 2 to 3 ulp of `std`. `f64` lanes evaluate them per lane.
* **Verification.** Every level the CPU runs, and the portable path, is checked against the scalar
  kernels lane for lane with lengths that leave remainders: `gax/tests/batch.rs` and
  `examples/traced/tests/batch.rs`.

## ADR-024: STAP and CSTA are pre-generated, behind their own features
*Status: accepted.*

* **Signatures.** Both follow the signatures as stated, `R(3,1,1)` and `R(4,2)`: space `e1, e2, e3`
  squares to `+1` and time `e4` to `−1`. The STA module keeps its own `(+,−,−,−)` convention with
  `e0` as time.
  * STAP adds the degenerate `e0`: vectors are hyperplanes and quadvectors are events, as in PGA.
    `Motor` is the even subalgebra (Poincaré motions).
  * CSTA uses the null basis `eo`, `ei` with `eo · ei = −1`, as `cga3d` does. `Motor` is the
    16-component Poincaré subgroup, and `Twist` its bivectors.
* **Size.** With 32 blades, STAP generates 2.5 MB of code in 13 s. CSTA has 64 blades. With `Even`
  and `Odd` declared as versors it produced 14.7 MB in 15 minutes, almost all of it fused sandwiches
  of the 32-component versors. As plain kinds (products only; `Motor` and `Vector` keep fused
  sandwiches) it is 6.7 MB in 100 s. Neither algebra is in the default feature set.
* **Tests.** The generated oracle tests found that random unit Poincaré motors cannot be sampled by
  normalizing random elements. The harness now also builds them as products of simple factors
  `1 + c B`.

---

## Hypotheses

| # | hypothesis | verdict | evidence / decision |
|---|---|---|---|
| 1 | Types generic over open slots; ops written once; `S ++ () = S` for generic `S` | **kept** | ADR-003, ADR-005; compiled experiments e11, e14 |
| 2 | Plain generic functions are the composition language | **kept** | ADR-014; API names adjusted (`trace::<I>`, `adjoint` via complement) |
| 3 | Labelled slots record equality groups | **changed** | ADR-015: equality by kind (`fill`, type-level `KindEq`); symmetrization over equal slots left to tracing |
| 4 | Build-time tracing with a symbolic coefficient type | **kept, refined** | ADR-010, ADR-016: bounded expansion; verified; never worse than the generic code |
| 5 | Three performance tiers | **changed** | Tier 1 is emitted straight-line code, not constant tables (ADR-006); dispatch by `Slots` methods (ADR-012) |
| 6 | Build the map, then apply it | **kept** | For 1024 points the map path beats the direct sandwich: 1.44 µs vs 1.88 µs (AoS), 0.50 µs vs 0.72 µs (SoA, AVX2); see performance.md |
| 7 | IEEE float semantics limit folding | **confirmed** | ADR-007; asm evidence in research.md §6 |
| 8 | Own solvers with closed forms by type | **kept** | ADR-017, ADR-019, ADR-020; property-tested, branch free, SIMD lanes match scalar |
| 9 | Batching via SoA and SIMD coefficients | **kept** | 1024 rigid transforms: 0.50 µs (gax f32x8) vs 1.11 µs (glam Affine3A loop) |
