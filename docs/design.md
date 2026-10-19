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
                                                                           ├─ tier 2: sandwich, its matrix, norms, inverse, exp/log
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
files in `gax/src/algebras/`, behind cargo features. Each algebra is an index file
(`algebras/pga3d.rs`) that `include!`s its parts (`algebras/pga3d/kinds.rs`, `products_gp.rs`,
`sandwiches_motor.rs`, …), each at most 400 KB, cut at item boundaries (`gax_gen::split`): the
compiled module is the same as one file, and every part stays small enough for GitHub to
highlight. `gax-regen --check` also reports part files no longer generated.

## ADR-002: MSRV 1.89, edition 2024
*Status: accepted.*

* **Why 1.89:** each requirement below was checked by building with the release before it
  (amended 2026-10-12):

  | needs | what | where |
  |---|---|---|
  | 1.65 | GATs | `Kind::Arr`, `Slots::Arr`, `HasCat::Cat` |
  | 1.79 | associated type bounds | `Extensor<Coef: Real>` |
  | 1.85 | edition 2024 | every crate |
  | 1.87 | `cast_signed`, `cast_unsigned` | `gax-core` (`math.rs`) |
  | 1.88 | let chains | `gax-gen`, so the default feature `macros` and `trace` |
  | 1.89 | `wide` 1.7, `safe_arch` 1.2, `fearless_simd` 1.0 declare it | features `wide` and `batch` |

  Without `wide` and `batch`, the library builds on 1.88. Cargo's `rust-version` is per crate,
  not per feature, so every crate declares 1.89. That is also the only old release CI tests.
* **Enforcement:** CI builds and tests on 1.89 and on stable. The two test crates that need the
  `wesl` crate (`gax-wesl-tests`, `gax-highdim-tests`) declare 1.97.1, as `wesl` does; they are
  never published.

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
* **Maximum length:** tuples have a maximum length of 12 (`slots::MAX_SLOTS`). A longer concatenation maps to a sentinel type
  whose witness fails in a `const` block, so it is a compile error only if it is actually used.
* **Associativity** (`Cat<Cat<A,B>,C>` against `Cat<A,Cat<B,C>>`) is not known for generic lists. It only
  matters when a generic signature brackets differently from the expression that produces it.
  `gax::slots::reassoc::<A, B, C, _>(m)` rebrackets a value, map or form (amended: it now exists).
  * It is the provided method `Slots::reassoc`: both lists hold the same slots in the same order, so
    it copies the coefficients through their row-major flat index. There is no per-tuple impl.
  * A doctest brackets a generic function differently from its expression. A compile-fail test
    shows the error without the witness, and the law suite checks that it is the identity on
    coefficients.

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
  * Why this sign: the join and the meet agree on every line.
    * The join of the origin and a point on `+x` is `+e23`, the same bivector as the meet
      `e2 ∧ e3` of the planes `y = 0` and `z = 0`.
    * The rotation `exp(-θ/2 · p ∨ q)` is right handed about the direction from `p` to `q`.
    * Planes and volumes come out with the right-hand rule: `O ∨ X ∨ Y = +e3`, and
      `O ∨ X ∨ Y ∨ Z = +1`.
    * ganja's join, `UnDual(Dual b ∧ Dual a)`, swaps its operands. The line from the origin to
      `+x` then comes out as `-e23`, the opposite of the meet that makes the same axis. Both
      conventions agree on planes through three points and on `plane ∨ point`, because the two
      sign changes cancel there.
    * The successors of ganja by its own author (GAmphetamine) and kingdon (Roelfs) use the sign
      chosen here.
* **Duality:** `dual()` is `J_R` and `undual()` is `J_L`. Both are metric free.
* **The Hodge dual** `hodge()` is `~x I`, with the metric: numga's `dual`, and what Maxwell's
  equations and constitutive maps in STA need. It agrees with `dual` where every basis vector
  squares to `+1`, flips sign on blades with an odd number of negative squares (STA, R(4,1)), and
  vanishes on blades containing a null vector, where `dual` does not (so PGA keeps `dual`). It
  is generated for every kind of an algebra with a pseudoscalar kind, values and maps alike
  (found porting numga's electromagnetism and cyclide examples, where the metric-free dual gave
  wrong results).
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
  * a map `x` uses the tier-2 matrix of `v`'s action and composes it into the output of `x`.
* **Open versor:** when `v` has open slots, `>>` falls back to tier-1 products.
* **The matrix itself** is spelled `v >> X::slot()` (there is no separate `to_matrix` method, so there is one
  spelling for one thing). It is generated directly with its structural zeros known. Applying a runtime `Point<(Point,)>` is dense; the crossover against the direct sandwich
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
  * **Laws (amended).** For pairings whose matrix `P` is a constant signed permutation (the join of
    planes and points in PGA3D, the inner product of vectors in a non-degenerate metric), the
    adjoint has the closed form `P tᵀ Pᵀ`. The law suite proves exactly that it satisfies the
    defining identity, reverses composition, fixes the identity, and returns `t` when applied twice
    with the flipped pairing. It also checks that `Pairing::solve` agrees in f64. The inner
    product `|` is degenerate in PGA (`e0 · e0 = 0`), so the join `&` is the pairing there.
* **Maps (`K<(A,), T>`):** `inverse`, `det`, `solve` (right-hand sides keep their slots), `svd` (typed
  singular vectors) and `trace`.
* **Forms (`Scalar<(A, A)>`):** `eigh(metric)`, which returns the modes as values of the slot kind
  (`Scalar::identity()` is the coefficient metric, for orthonormal layouts).
* **Outermorphisms:** `t.outermorphism::<B>()` extends a map on vectors by `∧`, or a map on antivectors
  (PGA points) by `∨`, to every homogeneous kind `B`. The generator derives the extended matrices
  symbolically, as minors; on the top grade the result is the determinant. This is numga's extension
  operator, after Fernández, Moya & Rodrigues. It is derived up to a size (ADR-034); above it, the
  map is built from its images.
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
* **6D and up (CSTA).** A bivector there splits into three or four commuting parts. At first `exp`
  fell back to scaling and squaring in the smallest kind closed under the product; for the full
  bivector of a 6D to 9D algebra it is now a closed form (ADR-035). The fallback remains for
  other kinds of bivectors and is correct in any algebra.
  * **Amended.** The first version used a fixed `B / 256`, 8 Taylor terms and no renormalization,
    so its unit error grew with `‖B‖` (to `10⁻¹⁰` at `‖B‖₁ = 64`).
  * Now the number of halvings is chosen from `‖B‖₁` (Higham's scheme), the series has degree 10,
    and Newton steps renormalize before squaring and, where it is well-conditioned, after it.
  * The unit error is a few ε at every norm, except for mixed bivectors where the squarings'
    rounding remains. See [numerics.md](numerics.md).
* **Branch and series (amended).**
  * `log` returns the rotation half-angle in `[0, π]`, so `exp(log R) = R`, and a versor with a
    negative scalar part gets the long way round.
  * At `R = −T` (a translation times −1) the log is not unique, and it returns `log(−R)`, the same
    motion.
  * Edge-case property tests found three defects, now fixed:
    * a wrong second-order term in the `log` series;
    * cancellation near the series thresholds of the translation coupling;
    * a branch switch near a full turn.
  * The series now reach far enough that both branches are within a few ulps at the boundary.
    See [numerics.md](numerics.md), "Transcendental functions".
* **6D log:** at first by inverse scaling and squaring, now in closed form up to 9D (ADR-033).

## ADR-020: Certification by a wrapper type, `Unit<M>`
*Status: accepted.*

* **The choice.** numga attaches traits such as "certified unit versor" to its runtime types. Here the
  certificate is a wrapper, `Unit<M>`, in the style of `nalgebra::Unit`, rather than a second kind with
  the same blades.
* **What `Unit` provides:**
  * its inverse is its reverse;
  * the product of units is a unit;
  * its sandwiches use kernels simplified with `x ~x = 1`;
  * its sandwiches keep the passenger's grades: `Unit<Even> >> Vector` is a `Vector` in CGA3D,
    STAP and CSTA, where the plain sandwich of an even element widens to the odd kind. The
    terms of other grades vanish only on unit versors, and not by the degree-2 relations
    `u ~u = 1` alone: in 5D the pseudoscalar part lies in the radical of their ideal, not in the
    ideal. The generator therefore drops them rather than deriving them. Up to 5D every even
    element with `u ~u = 1` is a versor (the Lipschitz group equals the even Clifford group,
    Lounesto), so the rule holds for every value a `Unit` can carry. CSTA's `Motor` is 6D but
    lies in the even algebra of spacetime and `ei` alone, a degenerate 5D algebra, which rules
    out the 6D counterexample `cos θ + I sin θ` (its sandwich turns vectors into grade 5);
    CSTA's full `Even` is not a versor kind. `tests/unit_versors.rs` checks the rule on random
    elements projected onto `u ~u = 1`;
  * `normalized()`, `inverse()` and `renormalize_fast()` also where `x ~x` is a scalar plus a
    4-vector `X` with `X²` scalar (5D even and odd kinds, CSTA's motor and twist):
    `x⁻¹ = ~x (a − X) / (a² − X²)` and `(a + X)^(-1/2) x` in closed form (De Keninck and Dorst,
    2022), the factor on the left since `X` does not commute with `x`;
  * tracing adds the condition to the ideal.
* **Why a wrapper:** kinds stay plain subspaces, and the number of generated pairs does not double.
* **Drift (amended).** `Unit * Unit` keeps the certificate without renormalizing, as nalgebra and
  glam do, so a long chain drifts to `u ~u = (1 + δ)²`.
  * **The problem.** The simplified `Unit` kernels were reduced modulo `u ~u = 1`, and so were no
    longer homogeneous in `u`. A drifted versor then *distorted* shapes: in PGA3D, a drift of
    `10⁻³` changed pairwise distances between moved points by `2.6·10⁻³`. The plain kernel,
    homogeneous of degree 2, only scales the result, which cancels projectively.
  * **The fix.** Every `Unit` kernel (the value path, the matrix path and the prepared action) is
    made homogeneous of degree 2 again. `cse::repair_degree` multiplies by `‖u‖²` where the
    reduction substituted 1, and falls back to re-expansion when that is cheaper. The generator
    asserts that the result equals the sandwich modulo the unit condition and is homogeneous.
  * **The effect.** Drift is now a uniform scale:
    * points, lines and planes keep their incidences and shapes exactly
      (`tests/numerics_drift.rs`);
    * in non-projective algebras (VGA), all lengths scale by the same factor.
  * **The cost.** `Unit<Motor> >> Point` (PGA3D) went from 25 to 33 multiplications (the plain
    kernel has 38); over all kernels the multiplications went up 6%.
  * **Renormalization.** `Unit::renormalize_fast()` is one Newton step, `(3 − u ~u) u / 2`,
    with no square root. It turns an error `e` in `u ~u` into `O(e²)`: the identity
    `r ~r = n (3 − n)² / 4` is proved on symbolic coefficients, and the convergence measured.
    The factor is on the left, so the identity holds whether or not `n` commutes with `u` (it
    does not for odd kinds in 4D, nor in 5D).
    `Unit::mul_renormalized` composes and renormalizes in one call; `normalized()` stays the
    exact path.
    * Suggested policy: renormalize after every integration step, or after every few products.
  * **Types are unchanged.** `Unit * Unit` still returns a `Unit`. Returning a plain versor would
    lose the cheap inverse and the simplified kernels for every composition chain.
  * **Debug check.** The opt-in `check-units` feature asserts, whenever a certified kernel
    consumes a `Unit`, that `u ~u` is 1 within `√ε` of the coefficient type. With the feature
    off it costs nothing.

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

## ADR-025: The laws are proved on symbolic coefficients
*Status: accepted, implemented. See [laws.md](laws.md).*

* **Method.** `gax_gen::sym::Sym` is a coefficient type whose values are exact polynomials over ℚ,
  hash-consed, with unbounded expansion. On `Sym`, `==` is polynomial identity. So a law that
  holds for fresh symbolic inputs is proved, for that algebra and those kinds and slot shapes.
  * Conditional laws (unit versors, the conditions that make an element a versor) are checked
    modulo a Gröbner basis of the condition. Its generators come from the library's own products
    on the symbolic input.
* **One suite per algebra, generated.** `gax-regen` writes `tests/laws_{alg}.rs` as one
  `law_suite!` invocation. It lists every product, unary operation, sandwich, outermorphism and
  signed-permutation pairing that the emitter produced among the small kinds (at most 8
  coefficients). The suite then covers exactly what exists.
  * The versor laws carry factors (`±‖m‖^2k`). The generator derives them from its own tables and
    emits them as expected values, so the library's kernels are checked against an independent
    derivation. `docs/law-factors.md` publishes them.
  * Every suite has negative controls: the equivariance law without its factor must fail.
* **Scope and cost.**
  * Versor laws use versors with at most 8 coefficients, and outermorphism laws maps on kinds
    with at most 5. A 16-component motor makes the free polynomials explode, and so does the
    determinant of a product of free 6x6 maps.
  * Two conjugation laws are degree 8 in a free versor (or degree 4 in two). They take minutes
    in non-degenerate metrics, so they are `#[ignore]` and a CI job runs them.
  * `gax-gen` is compiled with `opt-level = 3` in dev and test builds, which makes the suites
    about 8x faster.
* **Coverage.** The suites cover the eight standard algebras (STAP and CSTA behind their
  features) and PGA4D declared through `algebra!`. They replace the earlier f64 property tests of
  algebraic laws; the f64 tests left are about floating point: exp and log, normalization,
  solvers and lanes.

## ADR-026: Conjugation of maps is spelled with the existing operators
*Status: accepted.*

* **What it is.** Moving a map `f: K<(K,)>` by a versor means moving both its output and its
  input: `conj_m(f) = (m >> f).of(m << K::slot())`. numga writes
  `motor >> inertia(motor << Bivector)`.
* **Laws.** They are proved in the law suites, and their factors are in `law-factors.md`:
  * `conj_m(f).of(m >> x) == ‖m‖⁴ · (m >> f.of(x))`: it is `‖m‖⁴`, not `‖m‖²`, because
    `~m (m x ~m) m = (~m m) x (~m m)`;
  * `conj_m(f) ∘ conj_m(g) == ‖m‖⁴ · conj_m(f ∘ g)`;
  * `conj_{ab} == conj_a ∘ conj_b` exactly;
  * for `Unit` versors the factors are 1.
* **Decision: no new method.**
  * The expression is already short, and each side of it is a fused tier-2 matrix
    (`m >> f` transports the output, `m << K::slot()` is the inverse action's matrix), so a
    dedicated kernel would not save work.
  * Rust has no free operator left for it.
  * A method name would compete with `Conjugate`, the Clifford conjugation. The guide shows the
    idiom instead.
  * If a benchmark shows a fused conjugation kernel paying off (for example for inertia tensors
    moved every frame), a `Unit`-only method can be added then.

## ADR-027: Determinism is a feature, not the default
*Status: accepted, implemented. See [numerics.md](numerics.md).*

* **The problem.** The same program can give different last bits:
  * scalar `mul_add` is a hardware FMA only when compiled with `target_feature = "fma"`, so the
    default and `target-cpu=native` builds differ;
  * batch lanes pick FMA at run time from the detected level, so the scalar and batch paths
    differ too.

  That is fine for most uses, but it breaks lockstep networking and replays, which games need.
* **Options.**
  * Document "no cross-build determinism", and stop there.
  * An opt-in feature that makes every path compute the same bits.
* **Decision: the feature, `deterministic`.**
  * `Coef::mul_add` is `a * b + c`, never fused, for `f32` and `f64`, in the `wide` lanes and in
    the batch lanes on every level.
  * The elementary functions (`sin`, `cos`, `sinh`, `cosh`, `atan2`, `ln`) come from pure Rust
    everywhere:
    * `gax::math` (the polynomials the batch lanes vectorize) for `f32`;
    * the `libm` crate for `f64`.

    So they no longer depend on the platform's C library either.
  * The summation trees are already fixed: they are the generated straight-line programs, the same
    in scalar and lane code.
* **Why not the default.**
  * FMA is faster (the fused sandwich takes 5.3 ns with it and 6.3 ns without) and slightly more
    accurate.
  * The platform `libm` is faster than the portable functions for `f64`.
  * Most users never compare results across machines.
* **Verification.** `tests/determinism.rs` runs `exp`, the sandwich and `log` through the `Map`
  pipeline, `transform_each` and `transform_slice`, and requires bit equality with the scalar path
  on every SIMD level the machine has. CI runs it.
* **Not covered.** Different compilers may still differ where Rust itself does not specify the
  result:
  * the bits of NaN payloads;
  * `f64` transcendental functions from the standard library when the feature is off.

## ADR-028: WGSL modules by text emission, with plain GPU layouts
*Status: accepted, implemented. See [shaders.md](shaders.md).*

* **Printing, not a second implementation.** Every fused kernel is a `gax_gen::slp::Program`,
  verified exactly before any text is written (ADR-011). A WGSL printer
  (`Target::Wgsl`) renders the same programs:
  * `mul_add` becomes `fma`;
  * `select_lt(a, b, x, y)` becomes `select(y, x, a < b)`, since WGSL takes the false value
    first;
  * exact constants become abstract-float expressions such as `(1.0 / 3.0)`, which the shader
    compiler rounds to `f32` once.

  The emitters record each kernel's programs, and the Study-number helper between them, in a
  language-neutral form (`gax_gen::kernel::Kernel`). The WGSL module only renders those. The
  Rust output is unchanged, byte for byte.
* **Text only, no runtime dependencies.**
  * The modules are `&'static str` in `gax::wgsl` (feature `wgsl`), each with its module path
    (`gax::pga3d`).
  * gax depends on neither `wesl` nor `naga` nor `encase`. So no version conflict with an
    engine is possible: Bevy pins `wesl` 0.4 while 0.5 is current.
  * `wesl` and `naga` are dev-dependencies, used for validation only.
* **Plain WGSL, valid WESL.** Each module is self-contained: the Study helpers it uses are
  included, prefixed `study_`. It is plain WGSL, which can be prepended to a shader, and a WESL
  module, which can be imported with stripping.
  * WESL 0.5 added visibility, and imports across packages need `public` declarations, which
    older parsers reject. The source has none; `Module::wesl_public()` adds them.
* **The layout.**
  * A kind of N coefficients is a struct of `ceil(N/4)` `vec4<f32>` fields `c0, c1, …`, in the
    Rust blade order, zero-padded. It works in uniform, storage and vertex buffers (one
    location per field) without special cases. Only PGA3D `Line` (6 → 8 floats) wastes much.
  * The Rust side is a generated `{Kind}Gpu`: `#[repr(C, align(16))]` with
    `[[f32; 4]; ceil(N/4)]`, lossless `From` conversions both ways, and `bytemuck::Pod`
    (feature `bytemuck`).
  * The generator writes both sides, so it also writes `const` assertions of size, alignment
    and field offset (the idea of `const_shader_layout`, without the dependency). A test compares
    naga's layout of every struct with the Rust types.
* **Maps.**
  * WGSL matrices are column-major, and gax maps are stored output first. So a map between
    kinds of 3 or 4 coefficients converts to `GpuMat<C>` by transposing: column `i` is the
    image of input coefficient `i`.
  * Shaders apply it as `m * x`, and the generated `{v}_matrix_{x}` functions return it in the
    same orientation.
* **What is emitted, and its names.** WGSL has no overloading, and the full product set is
  large, so each algebra gets a curated set of functions, for every kind:
  * `motor_new` and `motor_from_rotor` (embeddings between versor kinds);
  * `motor_reverse`, and `motor_mul_point` (geometric products of versor kinds);
  * `motor_sandwich_point` and `unit_motor_sandwich_point`;
  * `motor_matrix_point` and `unit_motor_matrix_point`;
  * `motor_normalized`, `motor_renormalize_fast` and `motor_norm_squared`;
  * `line_exp` and `unit_motor_log`, on the real-trigonometric paths only.

  Solvers stay on the CPU.
* **Traced kernels.** `Tracer::wgsl(true)` writes `{stem}.wesl` next to the Rust kernels, from
  the same program, and adds a `{STEM}_WESL` constant. One traced function then drives CPU
  gameplay and GPU work, and the two cannot drift apart (the argument of ADR-016, extended).
  Kinds are written as WESL paths (`gax::pga3d::Point`).
* **Numerics.**
  * WGSL is `f32` only.
  * Its elementary functions may be less accurate than a CPU's, and whether `fma` fuses is up to
    the implementation.
  * Cross-GPU determinism is out of scope.
  * The drift-tolerant `Unit` kernels and `renormalize_fast` matter more here: renormalize
    motors integrated on the GPU every frame.
* **Testing, in layers.**
  1. Golden strings from the printer.
  2. naga validates every module with no GPU, and `wesl` compiles cross-package imports with
     stripping.
  3. naga's layouts are checked against Rust's.
  4. `wesl`'s evaluator runs every kernel on the CPU, compared with the exact value within the
     kernel's computed error bound.
  5. A wgpu harness runs every kernel on a GPU (lavapipe in CI), and checks the matrix
     orientation, a layout round trip and a traced kernel.
* **Every `exp` and `log`.** The general Study helpers (`study_exp_split`, `study_log_q`, …)
  are ports of `gax_core::study` over a small complex and dual-number library (`CHANNELS`),
  with their direct forms evaluated at a stand-in argument wherever the series is selected, so
  no discarded branch divides by zero. CSTA's bivector `exp` is scaling and squaring there, a
  loop emitted as text (`fallback_exp`), and the closed form of ADR-035 is `bivector_exp_closed`
  (`closed_exp`: recorded kernels `bivector_exp_reach`, `_from`, `_turning` composed by a loop
  that halves and squares), both tested against the Rust `exp`, within `2⁻¹²` on the CPU
  evaluator.
* **`f16`.** Every module has an `f16` twin (`gax::pga3d_f16`, `{Kind}Gpu16`), printed from
  the same programs with a precision parameter. Its Study helpers and the programs that feed
  them (norms) stay in `f32`, converted at the boundary, because `f16`'s range (2⁻¹⁴ to 65504)
  cannot hold a sum of squares. Tested on the GPU: the straight-line kernels against their
  error bound at unit roundoff `2⁻¹¹`, the others within `2⁻⁷` relative.
* **Declared algebras.** `algebra!` emits the same modules (`WGSL_MODULE`, `WGSL_MODULE_F16`,
  paths `package::{algebra}`), and the tracer names their kinds `package::{algebra}::Kind`
  (`Kind::MODULE` tells a standard algebra from a declared one). Tested by linking a traced
  kernel over a declared algebra and validating it (`gax-wesl-tests/tests/user_algebra.rs`).

## ADR-029: Division, embeddings, and the motor between two elements
*Status: accepted, implemented.*

* **The idiom.** In plane-based GA, the motion that carries an element `a` onto `b` of the same
  kind is `sqrt(b / a)`: for planes, `b / a` is two reflections, a rotation about their meet by
  twice their angle. The same formula works for lines (a screw) and points (a translation). This
  is how De Keninck, Roelfs and Todd teach PGA, and it needs division and a square root.
* **Division.**
  * `a / b` is `a b⁻¹`, right division as in ganja.js.
  * It is the trait `DivBy`, emitted for every coefficient and for every pair where `b` is a
    value of a kind with a closed-form inverse and the product `a b⁻¹` exists. `a` may carry
    slots, `b` may not, because the inverse is not linear.
  * Why a trait per pair, not a blanket `Div<R> where R: Invert`: with a blanket, a float
    literal on the right (`v / 2.0`) could be either `f32` or `f64`, and Rust falls back to
    `f64`. With one impl per right-hand kind, only the coefficient impl accepts a float, so
    inference works as it does for `*`.
* **Embeddings.** `From<A> for B` is emitted whenever every blade of `A` is in `B`, with the sign
  for blades that `B` stores with the other orientation. `Unit::widen` carries the certificate
  across, because an embedding keeps `x ~x = 1`. The WGSL modules already had
  `motor_from_rotor`; now the Rust side has the general form.
* **`Motor::between(a, b)`**, for PGA3D planes, lines and points and PGA2D lines and points:
  * It is computed as `exp(log(normalize(b / a)) / 2)`, not `normalize(1 + b / a)`.
  * Near a half turn, `1 + b / a` cancels, which gives an angle error of about `ulp/δ²` at `δ`
    from the half turn: O(1) in `f32` by `δ = 1e-4`. Going through the logarithm (with
    `atan2`) keeps the error at about `ulp/δ`, which is the conditioning of the problem itself,
    since nearly opposite planes meet far away.
  * `(b / a).sqrt()` stays available as the cheap literal form for unit elements.
  * Points are unitized first. No motor carries `p` to `-p`, because motors keep the sign of the
    weight.
  * `b = -a` is undefined: a half turn about any of infinitely many axes.
* **Verification.** `tests/between.rs` has property tests for every kind (unnormalized inputs,
  both algebras), the square-root identity, division undoing the product (maps included), the
  embeddings, and the `f32` precision near a half turn against the `ulp/δ` bound.

## ADR-030: The generator runs its jobs in parallel
*Status: accepted, implemented (`gax-gen/src/par.rs`).*

* **The problem.** Regenerating the standard algebras took about 8 minutes on one core, and
  the `algebra!` macro runs the same generator at compile time.
* **What is parallel.** Every job is a pure function of the algebra: a sandwich kernel, an
  outermorphism, a kind's value methods, a law. `par::map` runs them on all cores, taking the
  next job from a shared counter, heaviest first where a cost estimate exists, and returns the
  results in order. The emitter then writes them sequentially, so the output is byte for byte
  the sequential one, and `gax-regen --check` guards that.
* **Why no dependency.** `std::thread::scope` and an atomic counter are enough, and the
  generator is also a proc-macro dependency, where every crate adds build time.
* **Nesting.** Inside a job, the value path and the map path compile side by side, and the
  compiler's portfolio of strategies runs in parallel for programs over 2000 terms. Below that
  size, threads cost more than they save. Running whole algebras side by side as well was
  slower (39 s against 34 s), because it only oversubscribes the cores.
* **The tracer's `Sym` is thread-local.** The emitters use `SymMv`, plain data, so they are
  safe to run anywhere. Traced kernels still run on one thread each.
* **Result.** 34 s on 16 cores (ADR measurements in performance.md). The rest is the longest
  single jobs, which only a faster compiler (CSE) can shorten.

## ADR-031: Compile time: one closure per product, and line tables in test builds
*Status: accepted, implemented.*

* **Where the time goes.** gax is all generic code, so building it is parsing and checking,
  with no code generation. With every algebra, rustc spent 20.7 s type checking and 10.3 s
  borrow checking, out of 39 s. Every term of a product, `a[i] * b[j]`, was a separate
  operator to resolve, with the slot list of its result, `Cat<S1, S2>`, to normalize.
* **The change.** Each product impl now binds `let p = move |i, j| a[i] * b[j];` once and
  writes the terms as `p(i, j)`.
  * CSTA alone: 16.3 s to 13.9 s (type checking 9.1 s to 6.4 s, borrow checking 4.3 s to
    5.1 s).
  * A closure that borrows `a` and `b` instead (not `move`) makes borrow checking dearer than
    the saving. Closures for the sums as well (`add`, `sub`) made it slower overall, at 19.3 s.
  * The release code is identical: the product, rigid-body, matrix and motor-to-matrix probes
    of `gax-bench` have the same instruction counts.
* **Rejected: fewer, more generic impls.** A single blanket impl per product family (or
  products through constant tables) would type-check faster, but it gives up either the exact
  output kinds or straight-line code (hypothesis 5).
* **Debug info.** `gax` and `gax-core` build with `debug = "line-tables-only"` in the dev
  profile:
  * a test binary drops from 45 MB to 22 MB, and the target directory by a third;
  * backtraces and `file:line` stay;
  * only a debugger's view of locals inside generated code is lost.

## ADR-032: Homomorphisms between the standard algebras
*Status: accepted, implemented (`gax-gen/src/emit_homs.rs`).*

* **What.** A homomorphism is declared by the image of each basis vector. When the images
  keep the metric, it extends to the whole algebra and keeps every product. The generator:
  * checks the metric;
  * derives every blade's image, by the wedge of the images when vectors go to vectors, and
    by the geometric product otherwise (the spacetime split, on an orthogonal source);
  * **proves** `φ(a b) = φ(a) φ(b)` exactly, for every pair of source kinds;
  * emits `From<src::A<S, T>> for tgt::B<S, T>`, with `B` the smallest target kind that holds
    the image.

  A generated test (`tests/homs.rs`) checks the emitted code on random values.
* **Which.** Twelve natural, injective ones (the table in the guide, section 12).
  * `pga3d` to `cga3d` sends `e0` to `-ei`, so that a plane `ax + by + cz + d = 0` becomes the
    dual plane with `X · π` its signed distance. `e0` to `+ei` is a homomorphism too, but it
    would flip the side.
  * Quotients (PGA to VGA, `e0` to `0`) keep products too, but they lose information, so they
    are not `From`. None is shipped yet.
* **`From`, generic over slots.** A map converts like a value, and `B::from(A::slot())` is the
  homomorphism as a map. Maps between algebras need nothing new: `Of` is generic over the slot
  kind. `from_images` builds any linear map from the images of its input's basis blades.
* **Units.** `Unit::widen` now requires `Widen<N>`, which is emitted only where the generator
  proves `φ(~a) = ~φ(a)`:
  * for every embedding inside an algebra;
  * for grade-preserving homomorphisms;
  * for scalars, bivectors and rotors under the spacetime split, but not its vectors, which go
    to bivectors with `B ~B = -1`.

  The earlier bound, `N: From<M>`, would have certified those.
* **Not homomorphisms, so functions.** The round point `up` (quadratic) and `down` in
  `cga2d`/`cga3d`, spheres, and a round point's PGA point.

## ADR-033: The logarithm of a 6D to 9D even versor in closed form
*Status: accepted, implemented (`gax_core::study::{log_coeffs_6d, log_turn_6d}` for three
invariant planes, `{log_coeffs_8d, log_turn_8d}` for four; derivation in log6d.md).*

* **The problem.** CSTA's even versors had a logarithm only by inverse scaling and squaring
  (50 µs, loops). Robust closed forms were published below six dimensions; in any dimension,
  Roelfs and De Keninck factor a rotor root by root through its tangent decomposition. A 6D
  bivector splits into three commuting parts (a cubic), an 8D or 9D one into four (a quartic).
* **The closed form.** The invariants `uⱼ = cosh² μⱼ` of the planes are the roots of a
  polynomial whose coefficients are the scalar parts of `R`'s grade parts squared.
  `log R = r₀⁻¹ Σ αᵢ Qᵢ`, with bivectors `Q` from products of consecutive grade parts (through
  a matrix of determinant ±1) and `α` interpolating `φ(u) = √u · asinh(√(u−1))/√(u−1)` at the
  roots. The planes are never separated.
* **Why it is robust.** Coinciding roots (one plane, translations, isoclinic planes, the
  identity) are the common case, and divided differences through them would divide by zero.
  The interpolant is built from factors of the polynomial whose roots are apart, each through
  `φ`'s Taylor series at its roots' mean (a symmetric function of them, exact where they
  coincide) or its roots' values, joined by Chinese remaindering. Three planes: the most
  isolated root and the remaining pair. Four: whichever of all four, an isolated root and a
  cubic, two real pairs or two conjugate pairs separates best, from the resolvent cubic with
  Euler's form of the roots, refined by Newton steps in `u` and deflated from the stable side.
  Series are computed in a scaled variable, so they do not overflow `f32` near a half turn.
* **What the generator emits.** The invariants and the bivectors as one straight-line program
  from exact polynomials, the interpolant, and the combination.
* **Near and past a half turn** (`⟨R⟩₀ ≤ 1/16`), where the formula loses `ε/⟨R⟩₀`, or is wrong
  for `⟨R⟩₀ < 0`, the lanes there first turn the planes near a half turn by a quarter turn:
  `log R = log(R E) + (π/2) Z`. `Z` is the sum of those planes' unit bivectors, an interpolant
  applied to the same bivectors. `E = ∏(−b̂)` is a polynomial in `Z` (degree up to 4). The
  planes are chosen by an error estimate among those making `⟨R E⟩₀` positive
  (`log_turn_6d`, `log_turn_8d`). Orienting each plane by its weight in `⟨R⟩₂` makes that
  positive for any odd number of turned planes.
* **Not inverse scaling and squaring**, the earlier fallback. In 6D its channel-principal
  square root can leave the spin group, where rotation angles add up past a half turn; its log
  then misses `R` by a central element (`±1`, `±I`). With boosts it can also find no real root.
* **The branch.** Every invariant plane below a half turn (`cⱼ` with non-negative real part)
  where `⟨R⟩₀ > 0`: the geometric principal logarithm, idempotent under `log ∘ exp`.
* **Everywhere.** The generator emits it for the full even kind of every 6D to 9D algebra
  (tested for CSTA, `R(6,0)`, `R(3,3)`, `R(5,0,1)`, and in `gax-highdim-tests` for three algebras
  each in 7D, 8D and 9D). In WGSL (kinds up to 64 coefficients: 6D and 7D) it is
  `unit_even_log`: recorded kernels with the Study helpers `study_log6` and `study_log6_turn`.
* **Result.** 2.6 µs (6.6 µs turned) in CSTA against 50.7 µs; 2.5 to 3.8 µs (7 to 31 µs
  turned) in 7D to 9D. Right near and past half turns, where the old fallback was wrong or NaN
  for 7% of such versors; on 175,000 random versors in 6D to 9D, no NaN and no log that misses
  `R` (log6d.md §6).

## ADR-034: Algebras of 7 to 9 dimensions
*Status: accepted, implemented (`gax_gen::emit`: `FUSED_MAX`, `PLAIN_MAX_TERMS`, `MAP_MAX`,
`UNROLL_MAX`, `INLINE_MAX`; `gax_gen::slp::LETS_MAX`; `gax_gen::emit_outer::OUTER_MAX_TERMS`;
`algebra::MAX_DIM` = 9).*

The generator's approach (every product unrolled, every sandwich simplified symbolically) was
built for the standard algebras, whose largest kinds have 64 coefficients. A 7D algebra's even
kind has 64, 8D's 128, 9D's 256, and its full multivector 512: the symbolic sandwiches ran out
of memory (16 GB) in 7D, and the generated code overflowed rustc's stack and a debug build's.
Four changes make them work, each above a size no standard algebra reaches, so the standard
algebras' kernels are unchanged:

* **Plain sandwiches for versors over 32 coefficients.** `(v x) ~v` as two products straight
  from the tables, without symbolic simplification (whose cost grows with the cube of the
  versor's size), and the map computed at run time by applying the kernel to the passenger's
  basis. A `Unit` kernel keeps the passenger's grades and is homogeneous of degree 2, so
  drift-tolerant (ADR-020) with nothing to repair. A sandwich is not generated where its kernel
  has more than 2¹⁶ terms or its map more than 2¹⁴ entries (a 9D even versor on another):
  `v * x * v.reverse()` remains.
* **Products over 4096 terms as loops** over a static table of their terms, grouped by output
  and accumulated in a register (8D's even product: 11 µs; unrolled it is megabytes of code).
  The unrolled products over 1024 terms are `#[inline]` rather than `#[inline(always)]`, so a
  debug build does not give every inlined copy its own stack slots.
* **Long straight-line programs in blocks.** Every `let` opens a scope that lasts to the end of
  the function, and rustc's debug info recurses once per level: a few thousand overflowed its
  stack. Programs over 512 statements are written in blocks that hand their live values on as
  one tuple; the expressions and their order are unchanged (this also changed the layout of
  CGA3D's, STAP's and CSTA's largest kernels).
* **GPU layouts only up to 64 coefficients,** the WGSL modules' limit.
* **Products of large maps per pair of slot entries** (added after porting numga's spin groups,
  whose 6D sandwich maps took minutes each to compile). A product is generic over its operands'
  slots, and unrolled it multiplies whole slot arrays term by term: a CSTA `Even` times a
  `Bivector` slot is some 7000 array operations for LLVM. Above `SLOT_UNROLL_MAX` (4096) terms
  times slot entries, the generated product calls `slots::by_entries`, which splits both operands
  into slot arrays of plain values, runs the value product on every pair (`Slots::outer`), and
  reassembles the coefficients. The choice is an `if const` on the slot sizes, so values and
  small maps compile only the unrolled code. A probe with three such maps in CSTA went from
  128 s to 4 s in a release build (5 s with the same products on values).
* **Outermorphisms up to 16 384 terms** (added after porting numga's Hubbard ring, whose 8D
  algebra with a kind of 4-vectors ran out of memory). The extension of a map on `n` vectors
  (or antivectors) to blades of `k` factors has `C(n, k)²` entries, each a minor of `k!`
  terms; it was derived from every kind holding all vectors (or antivectors) to every kind
  of one grade. Measured in those terms, with the generator's time and code for the pair
  alone, and a debug build of a crate declaring the algebra's vectors and that grade, with the
  extension and without:

  | Extension | Terms | Generator | Code | Build with / without |
  |---|---|---|---|---|
  | 7D trivectors (numga's octonions; the largest anywhere in the repository) | 7 350 | 0.7 s | 0.4 MB | 4.0 s, 0.9 GB / 2.7 s, 0.6 GB |
  | 8D trivectors | 18 816 | 2.8 s | 1.1 MB | 10.7 s, 2.9 GB / 5.1 s, 1.0 GB |
  | 7D 4-vectors | 29 400 | 91 s | 1.7 MB | 78 s, 1.8 GB / 2.6 s, 0.6 GB |
  | 9D trivectors | 42 336 | 12 s | 2.7 MB | not built |
  | 8D 4-vectors (the Hubbard ring's states) | 117 600 | 436 s, 2.2 GB | 7.8 MB | killed at 10 GB after 6.5 min / 4.7 s, 1.0 GB |

  After the octonions' trivectors, the largest in the standard algebras, the tests and the
  examples are CSTA's 4-vectors and 5-vectors (6D: 5400 and 4320) and 9D's bivectors (2592). No
  extension of up to 9 dimensions lies between 7350 and 18 816, so every limit in that gap
  admits the same ones; `OUTER_MAX_TERMS` = 2¹⁴ admits every extension declared today
  (generated code unchanged): every one up to 6D; in 7D, those to 2, 3 and 7 factors; in 8D
  and 9D, to 2. Larger ones are not emitted. The `Outermorphism`
  trait's `on_unimplemented` note says so, and names what to do instead: build the map with
  `B::<(B,), T>::from_images`, the image of each basis blade the wedge of its factors' images
  (the Hubbard ring port builds its maps on 4-vectors from images too). The 8D test algebra in
  `macro_algebras.rs` checks that construction against the generated bivector extension.
  The extension to a kind missing some blades of its grade is now skipped before it is derived:
  a general map's extension fills the grade, so it never had an impl (CSTA's
  `Quintvector -> Twist`, 3600 terms, was derived and dropped).

  No switch to raise the limit: the next extension above it already costs two gigabytes, and
  `from_images` builds the same map at run time from `C(n, k)` wedges. No switch to turn
  outermorphisms off either: below the limit they cost seconds (CSTA's, the most, 12 s of
  generator time, beside its sandwiches).

Compiling a 7D, 8D or 9D algebra takes 30 to 60 s and 2.5 to 4.5 GB in a release build. `exp`
there was scaling and squaring (0.24 ms in 8D, 0.95 ms in 9D) until ADR-035.

## ADR-035: The exponential of a 6D to 9D bivector in closed form
*Status: accepted, implemented (`gax_core::study::{exp_weights_6d, exp_turn_6d, exp_reach_6d}`
and their `_8d` twins, `h_weights`; derivation in log6d.md §12).*

* **The problem.** `exp` of a 6D to 9D bivector was scaling and squaring in the even kind: fine
  in CSTA (4.1 µs, 32 coefficients), but 35 µs in 7D, 0.24 ms in 8D and 0.95 ms in 9D, where an
  even product has up to 65,536 terms.
* **The closed form.** `exp B = C (1 + T + T∧T/2 + …)` with `T = Σ tanh(μⱼ) b̂ⱼ` and
  `C = ∏ cosh μⱼ`. The `μⱼ²` are the roots of a polynomial whose coefficients are `⟨Wₘ²⟩₀`,
  `Wₘ = B^∧m/m!`; `T` is an interpolant of `tanh(√λ)/√λ` at the roots applied to bivectors
  `⟨Wₘ Wₘ₋₁⟩₂` (through a triangular matrix with ±1 on its diagonal), and `ln C` the trace of
  an interpolant of `ln cosh √λ` (`∏(1 − tⱼ²)^(−1/2)` cancels for large boosts). The planes are
  never separated.
* **Shared with the log.** The interpolants reuse ADR-033's groupings of the roots, rewritten
  over a trait of analytic data (values, a Taylor series at a centre, the distance to a
  singularity), so `exp` and `log` cannot drift apart in how they treat coinciding roots.
* **Rotations near a half turn**, where `tanh` has a pole: rotation planes beyond a quarter turn
  are turned back by one, `exp B = exp(B − (π/2) Z) · ∏ êⱼ` (ADR-033's polynomial in `Z`), only
  where some rotation is beyond `1.1 π/4`. A rotation beyond `3π/4`, or a loxodromic pair's
  beyond `π/4`, halves `B` and squares the result.
* **Fixes found on the way,** in the log too: a root divided out backward only when it is larger
  than its cofactor (a quotient `0/tiny` made the 8D PGA `exp` wrong by `10⁴⁶`, and an `f32`
  turn count 1.55), and a Newton step on the cubic's root kept only when it lowers the residual
  (at a triple root it created a fake separation: 8D PGA's log was wrong at three planes of
  1.2 rad plus a translation; regression test `equal_planes_below_a_sixteenth`).
* **WGSL** (added later): the helpers `study_exp6`, `study_exp6_turn` and `study_exp6_reach`,
  ports in `f32` next to `study_log6`, whose root finding and series arithmetic they reuse;
  the module's `bivector_exp_closed` halves, turns and squares as the Rust `exp` does. On a GPU
  it is 10 to 20 times as accurate as scaling and squaring for typical bivectors but 6 times
  slower (10 ns against 1.8 ns: local arrays live in memory on a GPU, and dense products are
  uniform multiply-adds), so `bivector_exp` stays scaling and squaring in `f32`. In `f16`,
  where scaling and squaring is off by up to `4·10⁻²`, `bivector_exp` is the closed form.
* **Result.** 3.5 µs in CSTA (4.3 µs turned), 4.0 / 6.8 / 10 µs in 7D / 8D / 9D: 9x to 90x
  faster from 7D on. Within `5·10⁻¹³` of a Taylor series in `f64` on random bivectors of seven
  algebras, no NaN (log6d.md §12).

## ADR-036: Kind tables: casts, grade parts, and binding a smaller kind
*Status: accepted, implemented (`gax_core::cast::{Cast, SubKind, GradePart}`, emitted per
algebra; `Of` accepts sub-kinds; `slots::MAX_SLOTS` = 12).*

* **The problem.** numga's types are arbitrary sets of blades, so it projects onto any subspace
  (`select_grade`, `cast`) and binds a vector into a slot of the full multivector by embedding it.
  gax had `From` between kinds whose blades nest, but no projection, no grade part, and binding
  required the slot's exact kind.
* **Not a `Multivector` trait.** A trait on values ("anything that converts to the full
  multivector") would route every bind through the largest kind and lose the zeros. The relation
  that matters is between kinds, and kinds are a closed family per algebra, so the generator
  emits it as tables: `Cast<B> for A` lists the blades `A` shares with `B` (positions and
  orientation), `SubKind<B> for A` marks `A ⊆ B`, and `GradePart<G> for A` names the declared kind
  holding `A`'s grade-`G` part (a kind of exactly that grade where one is declared). `cast`,
  `grade` and binding are loops over these constant tables; nothing is searched at run time.
* **Binding.** `Of<X>` requires `X::Kind: SubKind<Head>` instead of `X::Kind = Head`. For the
  slot's own kind the table is the identity, decided in a `const` block, and the code is the
  previous contraction unchanged (checked: same arithmetic, same timings, 21.8 ns for a CGA3D
  even-versor slot). A smaller kind is first placed in the slot's layout. Composing a smaller
  kind's slot into a larger one narrows the slot (`act.of(Rotor::slot())`).
* **What stays exact.** A product's result is still the smallest declared kind holding its
  support (ADR-007's zeros live in the code, not the type); declaring a kind gives a set of
  blades a type of its own. Exact result types per expression would need kinds as const
  bitmasks with computed result types (`generic_const_exprs`, unstable).
* **More slots.** Slot lists go up to 12 (was 8; Rust has no variadic generics, so tuples are
  expanded by macros). A dense extensor has the product of its slots' sizes per output
  coefficient, so the practical limit is that product, not the count.
* **Laws** (law M, every standard algebra, exact): a cast is the projection by blade name and
  orientation (computed independently from the blade names), on values and maps; a sub-kind's
  round trip is the identity and equals the generated `From`; binding a sub-kind equals binding
  its embedding, and narrowing a slot composes; a grade part is the value with its other grades
  zeroed. Compile-fail tests cover a cast with no shared blade, a missing grade, and a value
  that does not fit a slot.

## ADR-037: The inverse of any kind, by Shirokov's method
*Status: accepted, implemented (`emit_inverse_general`; `Algebra::signature`).*

* **The problem.** `inverse()` existed only where a kind's `x ~x` is a Study number (ADR-019),
  so CGA's and CSTA's `Even`, every `Multivector`, and most mixed-grade kinds had none.
* **The method.** Shirokov (2021): the Faddeev–LeVerrier recursion on left multiplication,
  `U₁ = x`, `Cₖ = (N/k) ⟨Uₖ⟩₀`, `Uₖ₊₁ = x (Uₖ − Cₖ)`, and `x⁻¹ = (U_{N−1} − C_{N−1}) / C_N`.
  It needs only products and scalar parts, and stays in the smallest kind closed under the
  product (the inverse is a polynomial in `x`). `x` is scaled to its largest coefficient first,
  so the scalars stay near 1.
* **The degree, and null directions.** The recursion needs a faithful representation whose
  trace is its dimension times the scalar part: for `R(p, q)` both half-spinor modules,
  `N = 2^⌈(p+q)/2⌉`. With `r` null directions Shirokov's degree `2^⌈n/2⌉` (counting them) is
  not always enough: checked exactly on random values, it holds for PGA2D, PGA3D and STAP but
  fails for `R(2,0,2)`, which needs 8, not 4. `2^r` times the non-degenerate degree always
  suffices (the Grassmann algebra acting on itself), but there left multiplication has repeated
  eigenvalues in Jordan blocks, which cost the recursion every digit for 5 to 9% of random values
  in 7D and 8D PGA (degree 32). So `x` is split instead: `x = a + n` with `a` free of null directions and `n`
  nilpotent (`(a⁻¹ n)^(r+1) = 0`, a null direction squaring to zero), the recursion runs on `a`
  at degree `2^⌈(p+q)/2⌉`, and `x⁻¹ = Σₖ (−a⁻¹ n)ᵏ a⁻¹` for `k ≤ r` is exact. (Where the null
  directions are not basis vectors the full degree is used.) The signature comes from the
  metric by Sylvester's law of inertia over the rationals (CGA's null basis included).
* **Only where something is invertible.** A kind whose values are all nilpotent (a degenerate
  pseudoscalar) gets no method: the generator evaluates `C_N` at random points modulo
  `2⁶¹ − 1`, and a nonzero residue proves the polynomial nonzero.
* **Refinement.** The recursion still loses digits as its degree grows (`10⁻⁶` to `10⁻⁴` at
  degree 32, 9D), so Newton–Schulz steps `y ← y (2 − x y)` follow, each squaring the residual:
  one up to degree 16, two at 32.
* **Accuracy and cost.** Within `10⁻⁹` times the condition `‖x‖ ‖x⁻¹‖` on random values in every
  standard algebra (law N), and within `4·10⁻¹¹` on 200 random even values in each 7D to 9D test
  algebra. PGA3D's `Multivector` takes 0.36 µs (degree 4, 7 products of 16 coefficients in all),
  CGA3D's `Even` 0.43 µs and `Multivector` 2.2 µs, CSTA's `Even` 2.2 µs and `Multivector` 14 µs
  (f64, Ryzen 7 5800X).
* **Not done.** WGSL modules keep their closed-form inverses only.

## ADR-038: Least squares and the pseudo-inverse on maps
*Status: accepted, implemented (`gax_core::linalg::{orthogonalize, pinv_weights, pinv_apply}`,
`gax_core::extensor::{LeastSquares, PseudoInverse}`).*

* **The problem.** `solve` and `inverse` need square, regular maps. numga has `pinv` and `lstsq`
  on maps and forms, including solving for one slot of a multi-slot map.
* **One algorithm for every shape.** One-sided (Hestenes) Jacobi on the columns, as `svd`
  already did for square maps: it works for tall and wide matrices alike, is accurate, and with
  a fixed number of sweeps branch free. The columns are an extensor's coefficient arrays (the
  `Column` trait), so a map with several slots is never flattened into one array (which would
  need `generic_const_exprs` for its size).
* **Semantics as numga's and NumPy's.** Coefficient norms, not the algebra's metric; singular
  values at most `rcond` times the largest count as zero, by default machine epsilon times the
  larger dimension (`numpy.linalg.lstsq`); `pinv_with`/`lstsq_with` take another. A one-slot
  map's right-hand side may have slots (kept) and be of a smaller kind (embedded, ADR-036); a
  multi-slot map's right-hand side has exactly the remaining slots, and the solution is a value
  of the first slot's kind (`at::<I>()` for another slot).
* **Checks.** The four Penrose conditions on random, rank-deficient and badly scaled square,
  tall and wide maps (`support/solver_checks.rs`, shared with the `solve` fuzz target), per-lane
  agreement with the scalar path, and on every SIMD level. The fuzz target found that a summed
  convergence test stops too early when singular values spread from `10⁶` to `10⁻²` (large
  columns hide a small pair that is not yet orthogonal; `A⁺ A` was off by `10⁻³`): each pair is
  now tested against its own lengths, and the input is a regression test. It later found that rotating such a pair does harm: two orthogonal columns of equal length were turned by an eighth of a turn every sweep, however small their dot product, which on a rank-deficient map undid the progress against longer columns parallel to them; singular values that should be zero only halved per sweep and ended at `2·10⁻⁵` of the largest. Pairs already orthogonal are no longer rotated (as in LAPACK's `dgesvj`), in the least squares and in `svd`, whose convergence test is now the same per-pair one. Cost: 0.93 µs for the pseudo-inverse of a 6 × 4 map (Jacobi sweeps until every column pair is orthogonal), against 0.17 µs for LU on a 6 × 6 map; a map whose columns are already orthogonal stops after the first convergence test.

## ADR-039: Dual numbers for derivatives
*Status: accepted, implemented (`gax_core::dual`).*

* **The problem.** numga differentiates through JAX. gax had no derivatives, though every
  kernel is generic over its coefficient type.
* **Forward mode as a coefficient.** `Dual<T, N>` carries `N` derivatives; it implements `Coef`
  and `Real`, so products, sandwiches, maps, solvers, `exp` and `log` (the 6D–9D closed forms
  included) run on it unchanged. `derivative`, `gradient` and `jacobian` set up the variables
  (amended: the typed derivatives of ADR-043 take the names `gradient` and `jacobian`; the
  array forms are `gradient_array` and `jacobian_array`, and `Dual<T, N>` is `Jet<T, Params<N>>`).
  Forward mode suits gax's sizes: a motor has 8 coefficients, and the costs scale with the
  number of inputs, which is small for rigid motions.
* **Selects and zero tangents.** `select_lt` chooses on the values and carries the chosen
  derivatives, so branch-free piecewise code differentiates the piece it uses. A derivative that
  is exactly zero stays zero through a function singular at the value: the closed forms compute a
  rotation's angle as `√(−B²)`, whose derivative is infinite at `B = 0` while its input's tangent
  is zero there; plain forward mode gives `0 · ∞ = NaN` for the (smooth) `exp` at zero.
* **Exactness.** `Dual<Fp, N>` differentiates polynomial kernels exactly (the product rule
  holds coefficient for coefficient). `Dual<f64x4, N>` differentiates four problems per lane.
* **Cost.** A unit-motor sandwich of a point with six derivatives takes 77 ns (12 ns in `f64`); the exponential of a line with its full 8 × 6 Jacobian 88 ns (16 ns): about `N + 1` times the arithmetic, as forward mode costs.

## ADR-040: Determinism per value: `Strict<T>`
*Status: accepted, implemented (`gax_core::strict`). Refines ADR-027.*

* **The problem.** The `deterministic` feature (ADR-027) makes every `f32` and `f64` computation
  in the build deterministic. Cargo features are additive across the dependency graph, so a
  program cannot keep a deterministic simulation and a fused renderer side by side, and one
  dependency turning the feature on changes every other user's results.
* **Not a second crate.** A `gax-deterministic` with its own generated types would duplicate
  every algebra, and its values would not mix with `gax`'s.
* **A coefficient type.** Everything is generic over the coefficient, as `Dual` (ADR-039)
  already uses: `Strict<T>` wraps `f32`, `f64` or a lane type, never fuses a multiply-add, and
  takes its elementary functions from `StrictElementary`, pure Rust per number type
  (`gax::math` for `f32`, its vectorized twin for the `f32` SIMD lanes, `libm` for `f64`, lane by
  lane). The batch kernels get `Strict` lanes from each level's, so `batch::map` and the batch
  sandwiches run on `Strict` data at full width. `repr(transparent)`, with `wrap`/`unwrap` for
  whole values and maps.
* **The feature stays,** for programs that want everything deterministic without changing a
  type; it and `Strict` compute the same bits (tested).
* **Found on the way.** The batch lanes built `exp` from `sinh` and `cosh` while the scalars
  called `gax::math::exp`: with the feature, kernels calling `exp` directly differed between
  scalar and SIMD in the last bits. Every lane type's `exp` is now its scalar's.

## ADR-041: Mass properties as one form (`Moments`)
*Status: accepted, implemented (`gax::pga3d::Moments`, `gax::pga2d::Moments`).*

* **The source.** De Keninck, Roelfs, Dorst and Eelbode (*Clean up your Mesh! Part 1*, 2025)
  write a simplex as a join of its vertices and a mesh as a sum, and get size and centre of mass
  from the Euclidean and ideal norms of joins with an apex. For the inertia they leave PGA: a
  frame of three vectors, diagonalized by Jacobi rotations of their own.
* **One form.** The second moments are a bilinear form on planes,
  `M(P, Q) = ∫ (P & x)(Q & x) dV`, a `Scalar<(Plane, Plane)>`. It holds the zeroth and first
  moments too (a point's pairing with the plane at infinity is its weight), so volume, centre of
  mass and inertia are its pairings with four fixed planes, and the type is a plain extensor:
  the moments of parts add, and a motion `m` moves them by composing both slots with
  `m << Plane::slot()`.
* **Exact from the boundary.** Over a simplex, the integral of a product of linear functions is
  `V/((k+1)(k+2)) (Σᵢ f(vᵢ) g(vᵢ) + f(Σ vᵢ) g(Σ vᵢ))`, so each cone from the apex to a boundary
  triangle contributes a sum of dyads `(P & x)(Q & x)` of its vertices. Written once
  (`simplex_dyads`) for 2D and 3D; only the joins (two points or three) are per algebra. The
  apex is arbitrary for a closed boundary, and closes a boundary cut by a plane through it.
* **The principal frame** comes from the existing symmetric eigensolver, then a rotation from the
  eigenvector frame (Shepperd's quaternion, then axis and angle into `Motor::rotation_about`),
  into `PrincipalInertia` and the rigid-body dynamics.
* **Checks.** Boxes, the unit tetrahedron and a triangulated sphere against closed forms; the
  apex's independence; additivity; equivariance (`moved` against moving the mesh); a cut mesh;
  the frame diagonalizing the inertia at half turns; polygons against the shoelace formula and
  known polar moments; `f32`.

## ADR-042: Complex coefficients and general eigenvalues
*Status: accepted, implemented (`gax::Complex`, `SquareMap::{eig, eigvals, eigh}`, `map_coefs`).*

* **The need.** Porting numga's examples, nine of them needed eigenvalues of maps that are not
  symmetric (rotations, boosts, stress-energy tensors, generalized problems against singular
  metrics) or complex numbers outright (the complex transversals of four lines, spin groups'
  eigenvalues `±i`); each wrote its own solver.
* **`Complex<T>` is a `Coef`, not a `Real`.** The generated products need only a ring, so
  complex multivectors and maps work unchanged; what needs an ordering (norms, solvers,
  `exp`) stays real, and `Complex` has its own `sqrt`, `exp`, `ln` and division.
* **`eigvals`** reduces the coefficient matrix to Hessenberg form by Householder reflections
  and runs the double-shift QR iteration with EISPACK's exceptional shifts (`hqr`, as in JAMA);
  eigenvalues come sorted, conjugate pairs adjacent. **`eig`** adds an eigenvector per
  eigenvalue by inverse iteration on `A − λI` with complex Gaussian elimination, returned as a
  value of the input kind with complex coefficients. Unlike the symmetric solvers they branch
  on the data, so they are for scalar coefficients, not SIMD lanes.
* **`eigh` on a map** takes the coefficient matrix as symmetric: numga's semantics, for maps
  self-adjoint under the coefficient inner product. A map self-adjoint under another metric is
  paired with it as a form first (`Form::eigh`), as before.
* **`map_coefs`** (on `Extensor`) converts the coefficients of a value or map, to `Complex`,
  `Dual` or `f32`; `Strict`'s wrapping was the private version of it.
* **Checks.** Rotations give `e^{±iθ}` and 1, boosts `e^{±φ}` and 1, and 200 random 6×6 maps
  satisfy `A v = λ v` with their eigenvalues multiplying to the determinant and adding up to the
  trace.

---

## ADR-043: Typed derivatives through jets over a kind
*Status: accepted, implemented (`gax_core::dual`). Amends ADR-039.*

* **The problem.** `jacobian` returned `([T; M], [[T; N]; M])`: plain arrays, where numga returns
  extensors. The gradient of a scalar function of `A` is the form `Scalar <- (A,)`, the
  Jacobian of `f: A -> B` the map `B <- (A,)`, the Hessian the form `Scalar <- (A, A)`, and a
  Newton step is `hessian.solve(gradient)`. The obstacle was `Dual<T, N>`'s const `N`: a
  function generic over `A` needs `Dual<T, { A::N }>`, which takes `generic_const_exprs`
  (ADR-005 keeps gax on stable without it).
* **Options considered.**
  1. *A function per kind, generated.* Every kind of every algebra gets its own
     `jacobian`/`gradient`/`hessian`, with `N` written out. More generated code and compile
     time for every algebra, and generic code over the kind would still have no single
     function to call.
  2. *A GAT on `Kind` for its dual type* (`type Dual<T>: Real` = `Dual<T, 4>` for `Point`). It
     works, but changes the trait every generated kind implements, for one feature, and the
     derivatives arrive as `[T; N]` that must be copied into kind arrays.
  3. *Run `f` once on the slot-generic `A::slot()`.* The derivative of a linear `f` is `f`
     itself on the slot, but a nonlinear `f` is not one map: `x * x` on a slotted `x` has two
     slots, `x * x * x` three, and their sum does not type-check. It covers linear maps only,
     which need no derivative.
  4. **Chosen: index the derivatives by a set of directions, not a count.**
     `Jet<T, D>` holds `re: T` and `du: D::Arr<T>`, for `D: Directions`. Every kind is a
     direction set, with its own coefficient array (`Kind::Arr`, which exists already), so
     `Jet<T, A>` carries one derivative per coefficient of `A` with no const arithmetic.
     `Params<N>` is the direction set of `N` loose parameters, `[X; N]`, and
     `Dual<T, N>` is now an alias for `Jet<T, Params<N>>`: one implementation of the
     arithmetic, and the existing code with `Dual` unchanged.
* **The typed functions.**
  * `jacobian(f, x)` evaluates `f` once at `variables(x)` (each coefficient its own
    direction) and reads `Df` off the coefficients. The derivative's coefficients for output
    `o` and step direction `i` are `y.c[o].du[i]`, which is the storage of `B <- (A,)` as it
    stands (output first). A map-valued `f`, `B <- (S…)`, gets `A`'s slot after its own, as in
    numga's `derivative`.
  * `gradient(f, x)` returns the form `Scalar <- (A,)` (of `A`'s algebra), and accepts a
    coefficient or a scalar-kind value from `f` (`IntoScalar`).
  * `hessian(f, x)` evaluates `f` once on jets of jets, `Jet<Jet<T, A>, A>`, and returns the
    value, the gradient and the form `Scalar <- (A, A)`. Nested forward mode costs
    `(1 + A::N)²` numbers per coefficient (49 for a twist); reverse mode would cost less, but
    gax's kinds are small and the nesting needs no new machinery.
  * The results are ordinary extensors: `of`, `solve`, `inverse`, `lstsq` and `eigh` take
    them. A Newton step is `x - hessian.solve(gradient)`.
* **Writing `f`.** `f` sees `A` with jet coefficients, so it is written generic over
  `T: Real` (the same function then serves values, gradients and Hessians), with constants
  lifted by `map_coefs(T::from_f64)`. Closures work where their argument type is inferred.
  The array forms are renamed `gradient_array` and `jacobian_array` (`derivative` of one
  variable is unchanged).
* **Checks.** `tests/derivatives.rs` compares every typed derivative with central
  differences, the gradient with `gradient_array` (identical numbers), the Hessian with the
  difference of the gradient, a Newton step on a quadratic with its exact minimizer, and the
  gradient, Hessian and Jacobian of `exp` with numga's `derivative` on JAX
  (`docs/research/experiments/numga_derivatives.py`, VGA3D, where numga's blade order and
  orientation are gax's).

## ADR-044: Structured maps beside dense extensors (`LinearMap`)
*Status: accepted (2026-10-13), implemented for `pga3d::PrincipalInertia` and
`PrincipalMobility`.*

* **The problem.** numga's `PrincipalInertiaPGA3` is a `Bivector <- Bivector` stored as four
  numbers, with a fast apply and a fast inverse; it subclasses `Extensor`, so generic code takes
  it as an ordinary map and every other operation rebuilds the dense 6 × 6 on demand. gax's
  `pga3d::PrincipalInertia` had `of`, `inverse_of` and `to_map()`, but it was a separate struct:
  composing, binding and solving needed `to_map()` first. It cannot implement `Extensor`, whose
  `coeffs(&self)` lends out stored dense coefficients that a compact struct does not have, and
  whose `from_coeffs` builds one from any dense coefficients.
* **Options.**
  1. *`Extensor` with its coefficients by value*, and structured maps as `Extensor`s. Every
     blanket method (`solve`, `eigh`, `lstsq`, binding) would accept them through the dense
     form, but `from_coeffs` would have to leave the trait, and no fast path could be added:
     the map methods are blanket impls over `Extensor`, and stable Rust has no specialization.
     Every generated type would change its core trait, and every use of a large map's
     coefficients would copy them in debug builds.
  2. *A representation enum inside the generated types* (dense or structured, chosen at run
     time). Every map pays a branch and the size of its largest variant; it defeats the
     straight-line kernels (ADR-006).
  3. **Chosen: a second trait beside `Extensor`.** `Extensor` stays the dense storage that every
     generated type implements, unchanged. `map::LinearMap` is any representation: kind, slots,
     coefficient type, and `to_dense()`, the dense coefficients by value. Every `Extensor`
     implements it by a blanket impl (`to_dense` is a copy). A structured type implements it
     itself. This is coherent because the structured type is not an `Extensor`. The blanket impl
     in gax-core and the impl for a gax type do not overlap, which a two-crate experiment
     confirmed.
* **Fast paths without specialization.** `map::SquareLinearMap` (`apply`, `solve`, `inverse`,
  with an associated `Inverse` representation) is what generic code over square maps is
  written against. Dense maps implement it through `SquareMap` (LU), and a structured map
  implements it with its own paths. Neither overrides the other: they are different types, so
  each has its own impl, and the choice is made at compile time.
* **Binding.** `Of<X>` now takes any `X: LinearMap` and binds its dense form, so a dense map
  composes with a structured one (`m.of(inertia)`), and forms bind it (`pairing.of(inertia)`).
  For a dense `X` the dense form is the value itself, and the release code is unchanged
  (below). A structured map that is the *outer* map composes by its own `Of` impl, column by
  column: `inertia.of(m)` scales and permutes `m`'s columns and never builds the 6 × 6.
* **The spike.**
  * `PrincipalInertia` (mass and three moments) and `PrincipalMobility`, its inverse in the same
    four numbers, implement `LinearMap`, `SquareLinearMap` and `Of`.
  * `of` and `inverse_of` take maps as well as values.
  * `tests/structured_maps.rs` checks against the dense form:
    * composition either way round;
    * binding into forms;
    * the energy form;
    * solve and inverse;
    * one function written against `SquareLinearMap`, called with either representation.
* **Scope, decided after the spike.**
  * `SquareLinearMap` stays at `apply`, `solve` and `inverse`: the operations where a
    representation can be faster than its dense form. The rest (`eigh`, `svd`, `lstsq`, `det`)
    goes through `to_dense()`, as numga rebuilds the dense map for everything but apply and
    inverse. Generic code with concrete kinds can call them on `m.to_dense()` with no extra
    bound.
  * Operators stay dense-only. Products are generated per pair of dense kinds; an impl per
    product per structured type, or a blanket impl over `LinearMap` that would collide with the
    generated ones, costs more than `.to_dense()` at the call site. The products' error note
    now names `.to_dense()`.
  * `motions::Motions::Inertia` stays a dense `Line <- Line`: a general inertia, about axes that
    are not principal, has all 36 numbers.
  * Concrete signatures: a function that takes `Line<(Line,), T>` still takes only the dense
    type. Rust has no subclassing, so "accepted as an ordinary map" means accepted by code
    written against `LinearMap` and `SquareLinearMap`. These are exported at the crate root.
* **Measurements** (rustc 1.99.0, against the commit before the spike):
  * *Release code of dense maps is unchanged.* The `#[inline(never)]` probes in `gax-bench`
    (`-C target-cpu=native`) have the same instruction counts before and after:
    | probe | instructions |
    |---|---|
    | `Line <- Line` bound to a value | 81 |
    | two such maps composed | 77, with its two callees |
    | a form bound with a map | 280 |
    | `solve` | 423 |
    | point map applied | 37 |
    | motor sandwich | 52 |
    | point map built | 115 |
  * *Debug build time is unchanged.* Builds of `gax` alone, per algebra, interleaved base and
    spike, three of each, have medians within the noise of one another:
    | algebra | base | spike |
    |---|---|---|
    | VGA2D | 0.57 s | 0.55 s |
    | PGA3D | 3.25 s | 3.27 s |
    | CGA3D | 7.06 s | 6.98 s |
    | CSTA | 12.43 s | 12.15 s |

    A first, sequential run had the spike 0.2–0.8 s slower; it ran while the machine was still
    loaded from the release build before it (load average 6.8), and the interleaved runs do
    not reproduce it.
  * *Compile errors.* The eleven existing compile-fail messages are byte for byte the same.
    * A structured map bound to a slot of the wrong kind gets the same message as a dense
      one, naming the map: "a `Line` does not fit a slot of kind `Point`, required for
      `Point<(Point,)>` to implement `Of<PrincipalInertia<f64>>`"
      (`structured_wrong_slot`).
    * A structured map in a product gets the generic "no regressive product (`&`) of
      `Line<(Line,)>` and `PrincipalInertia`", whose note now also names `.to_dense()`
      (`structured_operator`).

## ADR-045: Fields over a fixed number of sites, as kinds
*Status: accepted, implemented (`gax_core::sites`, `docs/fields.md`, `tests/sites.rs`).*

* **The problem.** numga's fields put the sites in the type, `vector[4]`, and a field map
  `vector[4] <- vector[4]` couples them: binding sums over the paired sites, composition over
  the shared ones, and `solve` solves the whole coupled block system. A fixed count fits Rust
  as a const generic, `Sites<K, N>`, stored nested, `[K::Arr<X>; N]`, with no arithmetic on
  `N` in any type. The snag is `Kind::Arr<X>: AsRef<[X]>`: every solver, binding and cast
  reads coefficients through a flat slice, and a flat `[X; K::N * N]` would need
  `generic_const_exprs` (ADR-005).
* **Options, measured.**
  1. *Relax the bound* (`Arr<X>: Elem` only, a flat index through methods). Removing it breaks
     129 places in gax-core alone: `extensor.rs` (the solvers, 124 with `linalg.rs`'s
     `Columns`), `slots.rs` 13, `bind.rs` 4, `dual.rs` 8, `trace.rs` 3, `fill.rs` 2,
     `cast.rs` 2. Each would become an index method call. That is mechanical, but it touches
     every algorithm, and gax would then compile and test only fields through those methods.
  2. *A separate trait beside `Kind`.* Nothing existing breaks, but a field is then not a
     kind. It cannot fill a slot, so `Vector<(Sites<Vector, 4>,)>` does not exist, and field
     maps would need their own types and every map method again: binding, composition,
     solve, eigenproblems.
  3. **Recommended: a field is a kind, with a flat view of its nested array.** `Sites<K, N>`
     implements `Kind` with `N = K::N * N` (an associated const, which needs no const
     generic arithmetic) and `Arr<X> = SiteArr<K, X, N>`, a wrapper around
     `[K::Arr<X>; N]`. The wrapper implements `AsRef<[X]>` by viewing its `N` arrays as one
     slice. That is sound when `K::Arr<X>` is exactly `[X; K::N]`, which an `unsafe trait
     FlatKind` states and the generator emits for its kinds (arrays have no padding between
     elements). The solvers' `SquareArr` gets one impl for `SiteArr<K, SiteArr<K, T, N>, N>`,
     square without naming `K::N * N`.
* **Built as recommended (option 3), and nothing existing changed.**
  * `Sites<K, N>` is a kind, `SiteArr<K, X, N>` its array, `SiteField<K, N, S, T>` its
    extensor. The generator emits `unsafe impl FlatKind` for every kind (its `Arr<X>` is
    `[X; N]`). The flat view is the module's one `unsafe` block; a unit test of it runs under
    miri in CI.
  * A field kind fills slots: `SiteField<Vector, 4, (Sites<Vector, 4>,)>` is a field map. `of`
    binds and composes over the sites through the generic `Of`, and `solve`, `inverse`, `det`,
    `eigh` and `svd` come from the `SquareMap` blanket through one `SquareArr` impl. On
    numga's bead string they agree with the sparse map's `solve_direct` and `eigh` to 10⁻¹⁴.
  * A field over four sites in a slot of `Sites<K, 3>` does not compile, with the binding's own
    message (`compile_fail/sites_wrong_count.rs`).
  * `Cast` of a field kind to itself slices a static identity table, which bounds a field
    kind at 4096 coefficients. That is no loss: the dense block solvers are for small systems,
    and larger fields belong to ADR-046's sparse maps (`from_field`, `to_field` and
    `from_sparse` convert).
* **The products, site by site.** Each of the 11 products has four blanket impls in gax-core,
  none generated: field with value, value with field, unit versor with field
  (`turn >> positions`), and field with field over the same `N`. They never overlap because a
  field kind is not `FlatKind`, and the generated impls are untouched.
  * The first form of value-with-field, `impl<L: Gp<K::Mv<S, T>>> Gp<SiteField<…>> for L`,
    broke existing code: any `x.gp(c)` with `c`'s type not yet inferred (generic `T`
    coefficients in `extras.rs`) made the compiler try `L: Gp<SiteField<?…>>`, then
    `L: Gp<?K::Mv<…>>` with `?K` unknown, recursing to the overflow limit.
  * The fix asks the bound of the right operand: `K::Mv<S, T>: RightOf<L, GpOp>`, implemented
    for every value `X` of a flat kind with `L: Gp<X>`. With `X` unknown the compiler finds
    the goal ambiguous and waits, instead of recursing.
  * The markers are `GpOp`, `WedgeOp` and so on. Markers named `Gp`, `Wedge`, … made rustc
    print every product trait by its full path in unrelated errors, which the compile-fail
    tests caught.
  * `Unit`'s and `Prepared`'s `>>`/`<<` forwarding now names its trait fully
    (`<Self as Transform<R>>::transform`) for the same reason.

## ADR-046: Fields over a run-time number of sites, and sparse field maps
*Status: accepted, implemented (`gax_core::field`, `docs/fields.md`).*

* **The problem.** Meshes have their site counts at run time, and their couplings are sparse: a
  vertex feels its edges. numga's `SparseExtensor` stores typed cells with row and column
  indices; `~ends * ends` is a mesh Laplacian. gax had only the batch side (`Soa`, slice
  kernels), with no sites in any type and nothing that couples them.
* **Types.**
  * `Field<M>`: a value (or map, or form) of type `M` at each site, in a `Vec`.
    * Site by site: `map`, `zip_with`, and `+`, `-` and unary `-` on owned and borrowed fields.
    * Over the sites: `sum`, `mean`, and `dot`, the coefficient inner product.
  * `SparseMap<C>`: cells of type `C`, each with a column, stored by row (compressed sparse
    rows, the order within a row kept), with a shape `(output sites, input sites)`. Repeated
    couplings add.
* **Operations, after numga.**
  * Multivector cells multiply:
    * `&ends * &positions`, each cell with the value at its column, summed at the rows;
    * `&ends.reverse() * &ends`, the cells along every path through the shared sites, summed
      per pair of sites;
    * `&cells * Vector::slot()`, which turns the cells into maps;
    * `&a + &b`, which adds couplings.
  * Map cells:
    * `apply` and `compose` go through `Of`, so the cell type is checked as binding checks it;
    * `transpose` runs the couplings back with the cells transposed: the adjoint under the
      fields' coefficient inner product.
* **Checks.**
  * The cell type is checked at compile time. Map cells `Point <- Point` applied to a field of
    lines fail with the binding's own message.
  * The site counts are checked at run time, with both counts in the message: "a field map with
    4 input sites applied to a field over 3 sites".
* **Solving.** Two methods, for two kinds of system.
  * `solve` is conjugate gradients with a diagonal (Jacobi) preconditioner, iterated to a
    relative residual of `16 ε`. CG is the right method for the symmetric positive definite
    couplings of physics (stiffness, mass, anchored Laplacians): it needs only the product and
    no more memory than the map. If it fails, because it meets non-positive curvature or its
    residual stays above `√ε`, it panics with a message that names `solve_direct`. A silent
    wrong answer would be worse.
  * `cg(b, tolerance, max_iterations)` returns the iterate with its iterations, residual and
    whether it converged.
  * `factor` is a sparse LU of the block matrix, and `solve_direct` is `factor` followed by a
    solve. It covers every nonsingular map: non-symmetric couplings, indefinite ones, and
    saddle-point systems of constraints, where CG fails.
    * The method is the left-looking LU of Gilbert and Peierls, as in Davis's CSparse
      (`cs_lu`). Each column comes from a sparse triangular solve whose pattern a depth-first
      search finds, so the work follows the arithmetic.
    * The columns are taken in a minimum-degree order of the sites: one at a time, the site
      with the fewest neighbours left in the elimination graph, its neighbours then joined
      into a clique. It is exact minimum degree on the explicit graph, which is fine for the
      site graphs of meshes. On a 64 × 64 grid of vectors it leaves 0.42 M stored entries
      against 1.08 M for a reverse Cuthill–McKee order (banded), and factors in about half the
      time; the first version used RCM.
    * The rows are pivoted by threshold partial pivoting. The diagonal is kept while it is
      within a factor of 10 of the column's largest candidate, as in UMFPACK.
    * A column with no nonzero pivot is reported as `Err(Singular { site, blade })`.
    * The factors (`SparseLu`) solve again for each right-hand side.
    * Measured (`f64`):
      * on the string of 1024 sites, factor and solve take 601 µs against CG's 467 µs, and a
        solve with the factors 18 µs;
      * on a 64 × 64 grid of points (16384 unknowns), factor and solve take 53 ms against CG's
        3.9 ms, and a solve with the factors 0.24 ms.
      * A factorization pays after about two solves on a chain and about 14 on a grid.
  * Considered and left out:
    * Approximate minimum degree (AMD) with quotient graphs scales to millions of sites; the
      explicit elimination graph is simpler and fast on meshes up to about 10⁵ sites.
    * A supernodal or multifrontal factorization pays from about 10⁵ unknowns.
  * `lstsq` is LSQR (Paige and Saunders), which converges from zero to the least-norm
    least-squares solution, for maps of any shape, iterating with the map and its transpose:
    numga's `lsqr` with its gauge left at zero. `lsqr` reports iterations and residual.
  * `eigh(metric, count)` gives the `count` eigenpairs nearest zero of `A x = λ M x`, after
    numga's `eigsh(…, sigma = −√ε · scale)`:
    * The search space is grown by block Lanczos on `(A − σM)⁻¹ M`, with one sparse LU of
      `A − σM` and full reorthogonalization in the metric.
    * The pairs are taken from `A` itself on the space (Rayleigh–Ritz on `Qᵀ A Q`, which needs
      no solves), and the space grows until each pair's backward error, its true residual
      `‖A x − λ M x‖` relative to `‖A‖ + |λ| ‖M‖` (∞-norms), is at `ε^¾`. It also stops at
      `√ε` once the residual stops improving: it has a floor of rounding near `10⁵ ε` on small
      dense spectra. (The first version divided the residual by `1 + |λ|` and compared it with
      `ε^¾` times the pencil's scale, scaling twice: a girder of stiffness `10¹¹` stopped with
      its eighth mode `5·10⁻⁴` off. The backward error does not change when the pencil is
      scaled.)
    * A converged eigenvector's error is its residual over the gap to the next eigenvalue,
      which is a small part of the scale on a fine mesh: `(π/n)²` above a free chain's rigid
      shift. So the converged pairs get one more step of inverse iteration, their own images
      `(A − σM)⁻¹ M x` added to the space, and Rayleigh–Ritz once more. On a free chain of 400
      sites, the first wave's distance from its exact eigenspace fell from 1.5·10⁻¹¹ to
      1.1·10⁻¹², for `count` more solves with the factors in hand. (Found by the
      spin-transformations port, which needs the constant field to 10⁻¹¹.)
    * An indefinite pencil (a Hamiltonian, whose levels nearest zero lie around its gap) needs
      the pairs chosen differently. Rayleigh–Ritz with `A` is reliable only at the ends of a
      spectrum; inside it, it finds spurious values near zero, mixtures of eigenvectors on both
      sides, with residuals near 1. The Kane–Mele port's 32 levels of a 620-unknown flake grew
      the space to nearly all of it. Once the space shows a negative Ritz value (and, by
      interlacing, it then stays indefinite), the pairs are chosen on the inverted operator
      `Qᵀ M (A − σM)⁻¹ M Q`, assembled from the images the Lanczos steps take anyway, where the
      wanted values are the largest. Rayleigh–Ritz with `A` on the chosen vectors' span then
      gives the values.
    * A pencil with a null space (a free mesh's rigid motions, a field's constants) is chosen on
      the inverted operator too, as ARPACK's shift-invert mode does. Rayleigh–Ritz with `A` sees
      a vector through its Rayleigh quotient, which is quadratic in the error: a null vector
      polluted at `10⁻⁸` has a quotient near `10⁻¹⁶`, rounding's own size, so `A` cannot tell
      it from an exact one, and the search stalls there. On the inverted operator a null vector
      stands out by `1/σ`. The spin-transformations port measured it: its constant field (an
      exact null space, 4-fold, with the next eigenvalue only `2.5·10⁻⁵` of the energy's scale
      above it) came back `3·10⁻⁹` off with `A`'s choice, `2·10⁻¹⁴` with the inverted operator's;
      numga asks for `10⁻¹¹`. The switch is made once the space shows a Ritz value zero to
      rounding (within `√ε` of the largest).
    * On the inverted operator each candidate's eigenvalue is estimated as `σ + 1/θ`, and the
      candidates nearest zero are chosen. Choosing the largest `θ` instead chooses the nearest to
      the *shift*, which is the same only while the shift is at zero: past a re-shift on a
      spectrum that seemed to start above zero, it picked the bulk modes just above the shift
      over the Kitaev wire's Majorana pairs at `10⁻¹⁴` below it.
    * Tried and dropped: a tolerance option for the backward error (`EighOptions`). Its only
      motivation was the spin-transformations gap, which was the choice of vectors above, not
      the stopping test; with the choice fixed, even a loose tolerance gives eigenvectors to
      `10⁻¹³` on a free chain. Iteratively refined solves in the last inverse-iteration step did
      not help either: the solves were not what limited the vectors.
    * The shift is scaled by the pencil's ∞-norms (the largest absolute row sums, which bound
      the spectral radii), not by its diagonals: a Hamiltonian of hops alone has none, and its
      shift was zero, on a singular pencil.
    * Each check of the space costs a dense eigenproblem of its size, so the space is checked
      only after it has grown by a quarter since the last check. With these three changes the
      Kane–Mele port's tests went from 88 s to 11 s.
    * Why not the Lanczos recurrence's own estimates, as ARPACK uses: with a null space (a free
      mesh) the shift is within `√ε` of the spectrum, a solve's error in every other direction
      is the null space's amplified size times `ε κ`, and the recurrence's residual estimates
      then claim `10⁻¹²` where the true residuals are `10⁻⁵`. That was measured; the
      projection on `A` restores the precision to about `10⁻¹²`.
    * A spectrum that starts well above zero (an anchored mesh) makes the shift below zero a
      poor one: the wanted cluster's inverted values crowd together (1 and 0.9976 on a 64 × 64
      grid), and the space grows slowly. So once the Ritz values show the cluster, the shift
      moves to just below it (one spread below the lowest), and moves again whenever it can
      come at least twice as close. Each move costs one factorization; the space is kept,
      since the pairs come from the pencil whatever shift built it. On the anchored 64 × 64
      grid of points this took 12 modes from 57 s to 1.9 s.
    * The move waits for estimates it can trust: the estimates' error bound, the backward
      error times the pencil's scale at the value, counts in the spread (2026-10-19). A first
      estimate from a small space can be far above the cluster: Spot's spin transformation
      (11720 unknowns) estimated 8.3 for a lowest value of 0.0045 from its first ten vectors,
      moved there, and then grew its space to 240 vectors favouring the values around 8.3;
      with the bound, it moves after the second check and stops at 40 to 50 (2.2 s to 0.9 s a
      solve). The anchored grid and the icosphere's modes are unchanged, within the noise.
    * Single-vector Lanczos finds one vector per eigenspace, and a field map of isotropic
      cells repeats every value `K::N` times. The blocks have `2 K::N + 2` vectors, which also
      covers a doubling by a symmetry of the sites (a square grid); `eigh_block` takes
      more.
    * Measured: on a free 64 × 64 grid of vectors (12288 unknowns), 12 modes in 0.32 s and 30
      in 1.1 s; on an anchored grid of points (16384 unknowns), 12 modes in 1.9 s; residuals
      near `10⁻¹²`.
  * Not here: a sparse SVD, and a non-symmetric sparse eigensolver.
  * An earlier `solve_dense` (LU on the assembled dense block matrix) is replaced by
    `solve_direct`, which is exact on the same maps and sparse.
* **Derivatives with respect to a field** (with ADR-043's jets):
  * `field::gradient` evaluates the function once per site, with jets seeded on that site, and
    gives a field of forms `Scalar <- A`.
  * `field::hessian` evaluates it once per pair of sites, `n (n + 1) / 2` times on jets of
    jets, and gives a sparse map of form cells `Scalar <- (A, A)`. Blocks that come out exactly
    zero are not stored, so the coupling structure is found, not declared.
  * That cost suits small fields. For energies that are sums of local terms, the documented
    path is the chain rule through the sparse map `D` to the terms' arguments: the gradient
    `~D * g` and the Hessian `~D * diag(H) * D`, with ADR-043's `hessian` on each term.
  * Both paths match numga's `derivative` on the bead string exactly.
* **Storage: an array of structs.** A field map reads its input at the columns of its cells, an
  arbitrary order: in a `Vec` each read is one contiguous value, in a `Soa` one strided read
  per coefficient. `Soa` stays the layout of independent copies.
* **Batch kernels: sliced ELLPACK.** The spike's batch apply put lanes on couplings. It gathered
  each cell and each input, then scattered the products into the rows one by one, because rows
  repeat; it was slower than the scalar loop. `SparseMap::packed()` builds a
  `PackedSparseMap` once instead:
  * The rows go in slices of 16, each slice padded to its longest row. A slice's `j`-th
    couplings are stored coefficient by coefficient, 16 rows side by side, as `Soa` stores
    values.
  * A lane works on one row. Each cell coefficient is a vector load, only the values at the
    columns are gathered, and a row's sum stays in its lane, so the result is written once per
    row.
  * Padding couplings are masked by comparing `j` with the row's length, not multiplied by a
    zero cell, so a non-finite value at a padding column cannot leak into a row
    (`packed_padding_is_masked`).
  * `PackedSparseMap` has `apply` only. A packed `cg` and `solve` were built and measured, and
    then removed. In conjugate gradients the vector arithmetic weighs as much as the products,
    and with 4 `f64` lanes the gathers eat the gain: the packed solve was 15–22% slower in
    `f64` and even in `f32`. A fast path that is slower does not belong in the API. The CG's
    dot products use four accumulators instead of one chain, which made every solve about 30%
    faster.
  * Also tried and dropped (2026-10-13): reading the input coefficient by coefficient, so that a
    lane group whose columns are consecutive loads them with one vector load. Columns are
    consecutive in 100% of lane groups on the string and 95% on a naturally ordered grid, but
    the per-apply transpose of the input and the per-coupling check cost more than the loads
    save. `f32` slowed from 5.4 to 6.6 µs and `f64` from 8.2 to 12.1 µs on the string.
* **Performance** (`benches/batch.rs`, groups `fields`; rustc 1.99.0, idle machine): a string
  of 1024 sites.

  | | default target | `target-cpu=native` |
  |---|---|---|
  | `apply`, 3070 couplings, each its own PGA3D `Point <- Point` (`f32`): scalar loop | 7.04 µs | 7.98 µs |
  | the same, packed (sliced ELLPACK) | **5.44 µs** | **5.23 µs** |
  | the dense batch path: one map, `of_slice` over as many values | 2.85 µs | 2.94 µs |
  | one product of the string's stiffness (`f64`): scalar / packed | 8.30 / 8.51 µs | 7.83 / 8.89 µs |
  | `solve` the stiffness, `f64` (33 iterations) | 452 µs | 449 µs |
  | `solve` the stiffness, `f32` | 138 µs | 146 µs |

  The sparse map costs 1.9× the dense batch path per coupling. That is the cost of a distinct
  cell per coupling: the dense path loads its one matrix once. The spike's first batch form put
  lanes on couplings and scattered into the rows; it ran at 10.3 µs, slower than the scalar
  loop.
* **Not here.**
  * A sparse SVD and a non-symmetric sparse eigensolver.
  * Batch products for multivector cells. Through `&cells * K::slot()` they become map cells,
    which `packed()` takes.

## ADR-047: The multivector derivative, through reciprocal frames
*Status: accepted, implemented (`gax_core::calculus`, `docs/calculus.md`). Phase 10, M1.*

* **The problem.** Geometric calculus differentiates with respect to a multivector:
  `∂_X F = Σ_J e^J ∂_J F` over the blades `e_J` of `X`'s kind, with `e^J` the reciprocal blades
  (`e^J ∗ e_K = δ^J_K`, `∗` the scalar product). For a vector `x` it is the vector derivative
  `∇F = ∇·F + ∇∧F`, which holds the divergence and the curl and writes Maxwell's equations as
  `∇F = J`. gax's derivatives (ADR-043) are forms, `Scalar <- (A,)`, never elements of `A`, as in
  numga. Turning a form into an element takes the metric, so it must be explicit, and it is
  impossible where the metric is degenerate.
* **Reciprocal frames, generated.**
  * gax-gen computes, for each kind, the Gram matrix of its blades under the scalar product and
    inverts it exactly, in rationals. Off-diagonal metrics (CGA's `eo`, `ei`) and the metrics of
    `algebra!` declarations are covered the same way.
  * Where the matrix is invertible, the kind implements `Reciprocal`: a table `FRAME` of
    `(coefficient, numerator, denominator)` terms per reciprocal blade, which `reciprocal(i)`
    turns into the value `e^i` with `Coef::from_ratio` (exact on `Fp`). The impls carry
    `#[diagnostic::do_not_recommend]`, so the error does not list them: the list would depend
    on which algebras are enabled. Where it is singular, there is no impl, and `#[diagnostic::on_unimplemented]`
    says why: the kind holds a null direction (PGA's `e0`), so it has no reciprocal frame, and
    `gradient` gives the derivative as the form it is.
  * Every kind of VGA and STA has a reciprocal frame, and so does every kind of CGA and CSTA
    that holds `eo` and `e∞` together (one with only one of them has none). In PGA, no kind
    with `e0` in a blade has one (`Plane`, `Line`, `Point`), and `Rotor` does; STAP likewise.
* **The functions** (`gax_core::calculus`; they read the derivative off the jets, as `jacobian` does).
  * `nabla(f, x)` returns `f(x)` and `∂_X f = Σ_J e^J ∂_J f`, with `e^J` on the left (Hestenes).
    It is `jacobian` contracted with the frame. Its kind is that of `X * F`, which the products
    already give. Called on a vector kind it is `∇`; on any other kind with a frame it is the
    multivector derivative.
  * `divergence(f, x)` and `curl(f, x)`: the same contraction with `⌋` and with `^`. They are the
    two grade parts of `nabla` for vector `x`, and that is a law.
  * `laplacian(f, x)`: `∇·∇f`, from the nested Jacobian (jets of jets, as `hessian`).
  * The functions are generic over the coefficient, so `f` runs on jets of `f32`, `f64` or the
    prime field.
* **Bounds relaxed.** `variables`, `jacobian`, `gradient` and `hessian` are bounded by
  `Real` today, though they never take a root or a transcendental. They are relaxed to `Coef`,
  so the laws below are proved on `Fp` (law L) and not only sampled in `f64`.
* **Laws.** [sample] on `Fp`, for polynomial fields, and [prop] in `f64` for transcendental ones:
  * `∇x = n` and `∇·x = n`, `∇∧x = 0` (the identity field), `∇(x²) = 2x`;
  * `∂_X (X ∗ A) = A` for `A` of `X`'s kind (Hestenes and Sobczyk, ch. 2);
  * `∇∧(∇∧f) = 0` for a scalar field, `∇·(∇·B) = 0` for a bivector field;
  * `nabla == divergence + curl` for vector `x`;
  * covariance under a unit versor `R`: the derivative of `x ↦ R F(~R x R) ~R` at `R x ~R` is
    `R (∂F)(x) ~R`;
  * STA: a plane wave `(k ∧ a) φ(k · x)` satisfies `∇F = 0`; VGA3D: a point charge's field is
    free of divergence and curl away from the charge [prop];
  * the frames of an `algebra!` declaration with an off-diagonal metric;
  * in laws.md, section O.
* **Not covered.** PGA and STAP (degenerate). The Euclidean `∇` inside CGA, where a position is a
  conformal point, not a vector: differentiate with respect to VGA coordinates there. numga has
  no `∇`, so the checks are the laws and central differences (as in ADR-043).

## ADR-048: Integration over simplices
*Status: accepted, implemented (`gax_core::simplex`, `gax/src/positions.rs`,
`docs/calculus.md`). Phase 10, M2.*

* **The problem.** Geometric calculus integrates over oriented chains, with a directed measure
  `dX`, a blade, and its fundamental theorem `∫ dX ∇F = ∮ dS F` contains Stokes, Gauss and
  Green. A mesh is a chain of simplices. gax integrated over simplices only inside `moments`:
  one rule, exact for products of two linear forms.
* **Simplices.** `Simplex<P, N>` holds `N` vertices of a position kind `P` (`N = k + 1` for a
  `k`-simplex; stable Rust has no `[P; K + 1]`). What a position is depends on the model of space,
  so two traits in `gax-core` give each model's operations, implemented in `gax` for the
  standard models (`positions.rs`, the way `Motions` is):
  * `Position`: the affine combination of positions, which forms the rules' nodes. In CGA it
    combines the Euclidean parts and lifts the result, since a combination of null vectors is
    not null.
  * `Measure<N>`: the directed measure of `N` vertices, a blade whose norm is the volume.
    * VGA2D, VGA3D, STA: `(v₁−v₀)∧…∧(v_k−v₀)/k!`.
    * PGA2D, PGA3D: the join of the points of weight one over `k!`: an edge's line, whose norm
      is its length; a triangle's plane in 3D; a scalar for the full dimension.
    * CGA2D, CGA3D: `P₀∧…∧P_k∧e∞` over `k!`.
    * A single vertex measures the scalar one.
    * The full-dimensional measures are signed as VGA's pseudoscalar coefficient in every model
      (which took a minus on CGA3D's tetrahedra, none elsewhere), a law below.
  * An `algebra!` declaration implements the two traits for its own position kind.
* **Rules.** `Rule::GrundmannMoller { s }`, of degree `2s + 1` in any dimension, and
  `Rule::degree(d)`, the cheapest exact on degree `d`.
  * Their nodes are rational barycentric points and their weights rationals, so a rule runs on
    any `Coef` and is exact on `Fp`. Gauss–Jacobi (Stroud) rules have positive weights but
    irrational nodes, which would put every law about integrals out of reach of law L.
  * A weight is built as a product of small factors (`mᵈ`, then `1/j` over the factorials, `4⁻ˢ`),
    each exact in every coefficient type. (The first version folded `mᵈ` into the loop over
    `(d+k−i)!`, which has fewer than `d` factors once `i > k`: the monomial law caught it.)
  * The nodes are visited by a callback over fixed buffers (`for_each_node`, up to
    `MAX_VERTICES = 9`), so the module needs no allocation and stays in `no_std`.
  * The weights alternate in sign from `s = 1`, and rounding grows with `Σ |w|`. The rules are
    offered to degree 15, about `10⁻¹²` relative; numerics.md has the measured table.
  * `Rule::VerticesAndCentroid` has positive weights and is exact to degree 2 (on an edge it is
    Simpson's rule, exact to degree 3).
  * `simplex::dyads` is the same rule in the homogeneous form `moments` uses, the centroid's
    weight folded into a quadratic form's homogeneity (`dyad(Σ vᵢ) = (k+1)² dyad(c)`). `moments`
    calls it, so its results are bit for bit what they were.
* **The integrals.** For an integrand `f: Fn(P) -> F`, with `F` any value or map that adds and
  scales:
  * `mean(rule, f)`, the mean of `f` over the simplex;
  * `measure()`, the directed measure;
  * `directed(rule, f)`, `∫ dX F`, which on a flat simplex is the measure times the mean;
  * `integral(rule, f)`, `∫ F dV`, which takes the measure's norm and so needs `Real`.
* **The fundamental theorem's form.** The operator that pairs with the boundary map
  `∂σ = Σ (−1)ⁱ ∂ᵢσ` is the measure contracted by the derivative from the left,
  `(∇ ⌋ dX) F = Σⱼ (eʲ ⌋ dX) ∂ⱼ F`. With the right contraction `dX ⌊ ∇` the sign alternates
  with the dimension, as `dX ⌊ v = (−1)^{k−1} v ⌋ dX`; the exact test found it on the first
  triangle. `gax::calculus::contract` (now public) computes it: the derivative combined with
  the reciprocal blades by any product.
* **Laws** (laws.md, section P).
  * Every rule is exact on the barycentric monomials up to its degree: `∫ λ^α dV` is
    `k! α! / (|α| + k)!` times the volume, on `Fp`, for every monomial; one degree higher
    each fails.
  * The fundamental theorem on one simplex, `∫_σ (∇ ⌋ dX) F = Σᵢ (−1)ⁱ ∫_{∂ᵢσ} dS F`, exact
    for cubic `F` on `Fp`, in VGA2D, VGA3D and STA.
  * Covariance: the measure and the mean position of a simplex moved by a rational motor are
    the moved ones (PGA3D, on `Fp`).
  * The models agree: VGA, PGA and CGA give the same volumes, and the same signed volumes at
    the full dimension.

## ADR-049: Simplicial complexes and meshes
*Status: accepted, implemented (`gax_core::mesh`, `docs/calculus.md`). Phase 10, M3.*

* **The problem.** The numga ports carry an oriented triangle mesh (`examples/numga/src/mesh.rs`,
  numga's `mesh.py`). It derives edges from faces, and its incidence operators `d0` and `d1`
  are field maps: the discrete exterior derivative. 19 ports use it, 6 its `d0` and `d1`, so its
  API has been tried. It handled triangles only, `f64` only, and lived outside the library.
  `Moments` took raw vertex and face slices.
* **Two types** (`gax_core::mesh`, feature `std`).
  * `Complex<N>`: topology alone, built from top simplices `[usize; N]` over a number of
    vertices (some may be in no simplex).
    * Every `k`-face is derived. Faces below the top are stored with their vertices sorted, in
      lexicographic order. That is numga's edge order, and it is canonical.
    * Top simplices keep the order they were given, which is their orientation.
    * Each simplex stores its faces, opposite each vertex in turn, with the orientation it
      induces on them: `(−1)ⁱ` times the sign of the permutation that sorts the face.
    * `d(k)`, from `k`-cochains to `(k+1)`-cochains, is a `SparseMap` of `±1` cells of a scalar
      kind. A bound `Kind<Scalar = Self>` would say "scalar", but generic code over a vertex
      kind `V` cannot prove it of `V`'s scalar kind, so it is a compile-time
      `const { assert!(N == 1) }` on the cell kind instead (the examples' generic `Mesh<V>`
      needed this).
    * `d∘d = 0` holds by construction, and it is a law.
    * The cells of a row keep numga's order, so the ports keep their bits: an edge lists its
      ends in its order (the first `−1`), a simplex of two or more dimensions the face opposite
      each vertex in turn.
    * `means(k)`: the mean over each `k`-simplex's vertices, in its order.
    * `index_of(k, vertices)`: a simplex by its vertices, by binary search, below the top in
      the stored order and at the top through a sorted index the construction keeps (a linear
      scan there made `Geometry::new` quadratic in the number of top simplices).
  * `Mesh<V, N>`: a complex with its vertices, a `Field<V>` of any kind (VGA vectors, PGA
    points), generic over the coefficient type, with `d`, `means` and `with_vertices`. Metric
    quantities are ADR-050's.
* **Checks.**
  * Construction returns a `MeshError`, not a panic: a vertex index out of range, a simplex
    that repeats a vertex, a top simplex given twice (as a set of vertices).
  * Queries, since non-manifold complexes are valid inputs:
    * `is_pseudomanifold`: each `(N−2)`-face in at most two top simplices. The name is the
      precise one: two surfaces meeting at a vertex pass, since it checks faces, not the stars
      of vertices (the design said `is_manifold`);
    * `is_oriented`: neighbours induce opposite orientations on the face they share;
    * `boundary()`: the `(N−2)`-faces of one top simplex, by index;
    * `coface_counts(k)`: the valences, for `k = 0`;
    * the Euler characteristic, and the Betti numbers.
  * The Betti numbers are `dim C_k − rank d(k) − rank d(k−1)`, the ranks exact on `Fp` by sparse
    elimination of the rows. Over ℤ/p they equal those over ℚ unless the homology has torsion of
    order `p`, impossible for `p = 2⁶¹ − 1` at any mesh size this library handles. (A Klein
    bottle's `ℤ/2` torsion is invisible, as over ℚ.)
* **Moments.** `pga3d::Moments::of_mesh` takes a closed surface as a `&Mesh<Point, 3>`, and
  `pga2d::Moments::of_mesh` a closed curve as a `&Mesh<Point, 2>`. The slice form of 3D is
  renamed `of_faces`; `of_polygon` stays. The mesh forms give the slice forms' numbers bit for
  bit.
* **The numga ports.** The examples' `Mesh` keeps its public fields (19 ports read `edges` and
  `faces` directly) and derives them, and `d0`, `d1` and the means, from a `Complex<3>`. A test
  keeps the old construction as a reference and checks every edge, orientation and coupling bit
  for bit on the three disks and a Loop-refined mesh.
* **What stays in the examples.** Loop subdivision, the disks, the Delaunay triangulation, the
  per-corner couplings: mesh making and processing, not calculus. gax has no mesh generation,
  and the tests use structured meshes.
* **Laws and checks** (laws.md, section Q).
  * `d(k+1) ∘ d(k) = 0` and Euler–Poincaré on random complexes of 2 to 5 vertices per simplex,
    and in the fuzz target `mesh_complex` on arbitrary input (shared checks,
    `tests/support/complex_checks.rs`).
  * The Betti numbers of a disk, a cylinder, a Möbius strip, a torus, a Klein bottle, the
    octahedron, a cube of six tetrahedra, the boundary of a 4-simplex, and two pieces.

## ADR-050: Discrete exterior calculus
*Status: accepted, implemented (`gax_core::dec`, `docs/calculus.md`). Phase 10, M4 and M5.*

* **The problem.** The metric half of calculus on meshes: Hodge stars, the codifferential, the
  Laplace–de Rham operator and the Dirac operator. The numga ports carried a triangle-only
  version (cotangent weights, hat gradients, the Whitney reconstruction, the spin Dirac
  operator of `surfaces_dirac_waves`) assembled by hand.
* **The metric is intrinsic** (changed from the design, which gave each model circumcentres and
  barycentric-gradient covectors of its own).
  * A model gives one thing, the squared distance between two positions (`Distance`): VGA's
    `|a − b|²`, PGA's squared norm of the direction between two points of weight one, CGA's
    `−2 A ∗ B`. STA is left out (its distances are not Euclidean).
  * Everything else follows simplex by simplex from the Gram matrix of its edges from the first
    vertex, `Gᵢⱼ = ½(d₀ᵢ + d₀ⱼ − dᵢⱼ)` in squared distances:
    * volumes, `√det G / k!`;
    * circumcentres in barycentric coordinates, `G x = ½ diag G`, `λ = (1 − Σx, x)`;
    * distances between barycentric points, `|Σ wᵢ vᵢ|² = −½ Σ wᵢ wⱼ dᵢⱼ` for `Σ w = 0`;
    * the inner products of the barycentric gradients, `G⁻¹` (and `dλ₀ = −Σ dλᵢ`).
  * So one implementation serves every model, and a user algebra needs only `Distance`; the
    three models give the same numbers (a law). It is how DEC is usually stated (the metric as
    edge lengths).
* **Two Hodge stars** (`Geometry::new(&mesh)` computes the volumes once).
  * Diagonal (circumcentric): `⋆_k = |σ*| / |σ|`. The dual volumes follow PyDEC's algorithm
    intrinsically: in each top simplex, along every flag of faces down to `σ`, the volume of the
    simplex of the flag's circumcentres, signed by the side of each face its parent's
    circumcentre falls on (the parent's barycentric coordinate at the opposite vertex; PyDEC's
    fix of June 2025). Positive on well-centred meshes (`is_well_centered`); signed elsewhere.
    The top simplices' primal volumes are unsigned (PyDEC signs them by orientation when they
    fill their space).
  * Galerkin (Whitney): `M_k = ∫ Wᵢ · Wⱼ dV`, `W = k! Σ (−1)ⁱ λᵢ dλ₀∧…∧dλ̂ᵢ∧…∧dλ_k` in each
    face's own vertex order (sorted below the top, as given at the top). The products are
    quadratic in the barycentric coordinates with constant gradient parts, so the entries come
    exactly from `∫ λₐ λ_b dV = (1 + δₐ_b) V / ((n+1)(n+2))` and determinants of the gradients'
    inner products.
  * **A finding.** The design claimed both stars give the same stiffness on 0-forms in any
    dimension. On triangles they do (`d₀ᵀ ⋆₁ d₀ = d₀ᵀ M₁ d₀`, the cotangent formula, for any
    triangle). On tetrahedra the circumcentric one agrees with linear finite elements only on
    symmetric ones: `2·10⁻¹⁶` off on a regular tetrahedron, 0.02 on a near-regular one, 0.17 on
    a random one, about 25 % on a jittered grid (PyDEC gives the same circumcentric numbers). The
    Galerkin stiffness is the finite-element one in every dimension, which is the law kept.
* **The maps.**
  * The de Rham map `de_rham(&mesh, rule, F)`: the integral of a `k`-vector field over each
    `k`-simplex against its measure, `∫_σ F ∗ ~dX`, by `gax::simplex`'s rules. It needs no metric,
    so it runs on any `Coef`, and Stokes's theorem is exact on `Fp`.
  * The Whitney map on edges, `Geometry::whitney_1`: `Σ c_e (λₐ ∇λ_b − λ_b ∇λₐ)` in a top
    simplex, the gradients the reciprocal frame of its edges in the model
    (`∇λᵢ = Σⱼ (G⁻¹)ᵢⱼ (vⱼ − v₀)`, vectors in VGA, directions in PGA). Higher degrees need wedges
    of gradients typed by degree, and wait for a use.
* **Operators** (M5).
  * With the diagonal star: `δ_k = ⋆_{k−1}⁻¹ d_{k−1}ᵀ ⋆_k`, `Δ_k = d δ + δ d` (positive
    semidefinite), and the Dirac operator `D = d + δ` on the cochains of every degree at once,
    one field over all simplices. All are `SparseMap`s. Eigenproblems take the symmetric pencil
    `(⋆Δ, ⋆)` to `eigh`.
  * With the Galerkin star: the stiffness `d_kᵀ M_{k+1} d_k` against the mass `M_k`, a
    symmetric pencil for `eigh` and `solve`. The mixed Hodge Laplacian of finite element exterior
    calculus needs `M⁻¹` (a saddle-point system) and is left out until an example needs it.
  * The inverse diagonal star panics on a zero dual volume (a circumcentre on a face: a right
    triangle's hypotenuse), naming the dimension, rather than producing infinities.
  * `Geometry::from_parts(&complex, &vertices)` builds the metric without a `Mesh`: the numga
    ports keep their vertices in their own mesh type, which the ports may change.
  * The ports' `edge_ratio` and `laplacian` are now the library's `⋆₁` and `⋆₀Δ₀`.
* **Laws and checks** (laws.md, section R).
  * Hirani's PyDEC, as fixtures (`docs/research/experiments/pydec_fixtures.py`, its own venv from
    GitHub; the `pydec` on PyPI is another package): every volume, dual volume and star of every
    dimension on jittered 2D and 3D grids, within `10⁻¹²`.
  * The three models agree; the Galerkin stiffness is linear finite elements (gradients from
    GA reciprocal frames, independently); Stokes for the de Rham map exactly on `Fp`;
    `R ∘ W = id` and `W₁ d₀ = ∇ W₀`; the numga ports' hand-built metric is the library's.
  * `δ` adjoint to `d`, so the Hodge decomposition is orthogonal; `D² = Δ`; the harmonic
    dimensions are the Betti numbers on a sphere and a torus; on subdivided icosahedra the
    vertex Laplacian's spectrum converges to `ℓ(ℓ+1)` at second order with both stars.
* **Scope.** Euclidean models only: VGA2D, VGA3D, PGA2D, PGA3D, CGA2D, CGA3D. Meshes in
  spacetime, whose stars are indefinite, are left out. The spin Dirac operator stays an example,
  built from these pieces.

## ADR-051: Orientation types: intrinsic and extrinsic
*Status: implemented (2026-10-18). Phase 11.*

* **The problem.** gax's `>>` is the plain sandwich `v x ~v`. It commutes with the geometric,
  outer and inner products, but under an odd versor (a reflection) not with the join:
  `(m >> a) & (m >> b) == −(m >> (a & b))` in even dimensions (law-factors.md). So a reflected
  momentum, forque or plane comes out negated (ganja.js#180, kingdon#154, the workaround in
  guide §10). Dorst (2024) explains why: every element carries one of two orientations, a sense
  within itself (intrinsic: a point's weight, a join's direction along a line, a triangle's
  winding) or a sense in the space around it (extrinsic: a plane's side, a meet's turn about a
  line). A mirror reverses the space around things, so the two kinds reflect with opposite
  signs, and no single sandwich serves both. numga adopted the distinction (upstream `7dff347`,
  `docs/why_orientation.md`, `docs/orientation.md`): an orientation on every type, propagated by
  the products, checked on sums and bindings, choosing the sign of `>>`.
* **The rules** (numga's, which are Dorst's; reading extrinsic as `+` and intrinsic as `−`):
  * `*`, `^`, `|`, `⌋`, `⌊`, the scalar product, the commutator and anticommutator, `a / b`: the
    product of the signs. `&`: minus the product (it is the outer product of the duals,
    dualized: three flips). `dual`, `undual`, `hodge`: flipped. `~a`, `a.involute()`, `-a`, `a * t`
    for a number `t`, `a.inverse()`, casts: kept. `a + b`: equal orientations required.
  * `versor >> x` keeps `x`'s orientation. An even versor is the sandwich. An odd versor is the
    sandwich of the involute, negated for an intrinsic `x`: `±m x̂ m⁻¹`. Then every versor commutes
    with every product, the join included, and (in a non-degenerate algebra) with the dual. The
    versor's own orientation does not matter (it appears twice).
  * Relative to gax's plain sandwich, the new `>>` of an odd versor multiplies a blade of grade
    `k` by `(−1)ᵏ` for an extrinsic passenger and by `−(−1)ᵏ` for an intrinsic one: a sign per
    blade, decided at compile time. Even versors are unchanged.
  * Which orientation an element has is set by how it is made, not by its meaning. Vectors and
    everything built from them by products are extrinsic; a dual of an extrinsic element is
    intrinsic. In PGA the vectors are planes, so meets are extrinsic, and points, made as the
    complements of coordinates, are intrinsic, and so are their joins. In gax's CGA the points are
    null vectors, so they are extrinsic, and so are the circles wedged from them, which then
    reflect by the twisted sandwich as Hamish Todd argues; their duals, the spheres-meet forms,
    are intrinsic. In numga's plane-based CGA the encodings swap, and the same rules give the
    same geometry.
* **Options for carrying it in Rust.**
  1. *Twin kinds* (`Line` and `Spear`) declared in the specs. Every result needs both twins, and
     the product impls grow up to fourfold; cga3d and csta are already the slowest to compile.
  2. *A run-time bit* on each value (Gunn's "complementarity bit"). Mixing kinds would be a
     run-time panic, and every kernel would branch or multiply.
  3. **Chosen: an orientation type parameter on every kind**, `Line<S, T, O>`, with `O` one of the
     zero-sized markers `Extrinsic` and `Intrinsic`. One generated impl covers both orientations,
     generic over them, with the result's orientation computed by a type-level product
     (`<O1 as Orientation>::Times<O2>`), so compile time and code size barely move. The sign of a
     reflection is a constant of the monomorphized impl. Mixing them is a type error: `twist +
     momentum` does not compile, nor does binding a momentum into a slot that takes a twist, nor
     `momentum.exp()`.
* **Defaults.** numga's types carry no default; Rust needs one for writing types at all. Each
  kind's default is the orientation of its own constructors, declared in the spec
  (`kind Point = [...] intrinsic`): PGA's `Point` and `Direction` are intrinsic (`Point::xyz`,
  `Point::direction`), everything else extrinsic. So `Point<(), f64>` is what `Point::xyz` returns,
  and the meet of three planes is a `Point<(), f64, Extrinsic>`. A wrong default cannot pass
  silently: every mismatch is a type error. `algebra!` declarations take the same keyword; without
  it every kind is extrinsic.
* **Relabelling.** Coefficients alone do not say which kind they hold. `x.with_orientation::<O>()`
  keeps the coefficients and declares their orientation, for data from outside (a file, a GPU
  buffer, a raw `new`).
* **What follows from the types, with no extra code.**
  * Sums need equal orientations (they need equal types already).
  * Binding needs the slot's orientation: a slot is a kind, and the kind includes `O`.
  * `det`, `eigh`, `solve` and the other square-map operations need an endomorphism, output and
    slot of the same kind and orientation: an inertia `Line<(Line,), T, Intrinsic>` (twists to
    momenta) is a correlation and has no eigenvalues of its own, as in numga.
  * `exp`, `log`, `sqrt` exist for extrinsic kinds only: a force generates no motion.
* **The rest of the library.**
  * `Motions`: `Twist` stays `Line`; `Forque` becomes the intrinsic `Line`.
  * `Plane::reflect` and `pga2d::Line::reflect` lose their hand-written signs: they become `>>`.
    `tests/reflections.rs` becomes the orientation laws, and guide §10 the rules above.
  * Batch kernels (`Soa<K, E>`): `K` includes `O`, and the sandwich kernels apply the sign.
  * GPU layouts and WGSL modules carry no orientation: the conversions accept either, and the WGSL
    odd-versor sandwiches stay the plain sandwich, documented, with the CPU's `>>` the reference.
  * The tracer prints `O` where it is not the kind's default (`Kind::DEFAULT_INTRINSIC` says
    which that is): `Line<(), T, ::gax::Intrinsic>` in a fused kernel's signature, bare `Line`
    otherwise, so traced code for default orientations reads as before.
* **Laws.**
  * Every versor law factor becomes `‖m‖²` for every product, `&` included: the `−‖m‖²` entries of
    law-factors.md disappear, proved on symbolic coefficients as now.
  * `m >> a.hodge() == (m >> a).hodge()` for odd and even versors in the non-degenerate algebras
    (`tests/reflections.rs`). The metric-free `dual()` commutes too where it equals the Hodge
    dual up to sign, in an orthonormal Euclidean basis; in CGA's null basis it does not, as it
    is not a metric operation.
  * The issues' examples: a reflected momentum keeps its kinetic energy, a mirrored duck stays on
    its side of the water, with no signs written by hand.
  * Compile-fail tests for each refusal.
* **The numga ports** follow upstream `7dff347`, whose examples were migrated to orientations
  ("meets and joins by meaning"), and port its four orientation notebooks (surface normals,
  mechanical work, a mirror's energy, a circle constructed twice), checked against numga's
  numbers as before.
* **Plan.** The markers and traits in `gax-core`; the generator (structs, products, unary
  operations, casts, kind tables, sandwiches, batch, GPU, tracer) with the spec keyword; the
  library, tests and examples migrated; the laws; then the numga sync.

## ADR-052: API consolidation: orientation by name, `f64` by default, one name per idea
*Status: accepted (2026-10-18). Phase 12.*

A review of the whole public API after ADR-051, with the migration of about 150 programs as
evidence, found it sound in its core (slots, generated products, `Unit`, the laws, the
diagnostics) and uneven around it. The changes below drop backward compatibility.

* **Orientation by name.** The struct of a kind lives in `{alg}::oriented` with no default
  orientation: `oriented::Line<S, T, O>`. The module root names the kind's default orientation by
  an alias, `type Line<S = (), T = f64> = oriented::Line<S, T, Extrinsic>`, and the modules
  `{alg}::intrinsic` and `{alg}::extrinsic` name every kind in that orientation (`Forque` is
  `intrinsic::Line`, PGA2D's `Twist` is `extrinsic::Point`). Constructors (`new`, `xyz`, `slot`,
  `from_images` and the extras) are emitted for both concrete orientations: a path through an alias
  fixes `O`, so `Line::new(..)`, `Forque::new(..)` and `Forque::slot()` all resolve with no
  annotation and stay `const` (verified on 1.89 and stable; a constructor generic over `O`, or
  defined twice on one name with `O` open, is ambiguous, E0034/E0283). Slot kinds read
  `Line<(intrinsic::Line,)>`. Generic code names `oriented::Line<S, T, O>` and gets slots from the
  trait `gx::Slot`; `oriented_slot` goes. `gx::Oriented<K, O>` names a kind in another orientation
  generically. The tracer prints `intrinsic::Line` where it printed `Line<(), T, Intrinsic>`.
  A conversion through an alias converts into that alias's orientation, `pga3d::Line::from(x)`
  into the default; `x.cast::<Line>()` and `x.into()` keep `x`'s. The struct keeps its parameter
  defaults (`oriented::Line` alone is the default orientation), so generated code reads as before;
  the algebra root re-exports `oriented::*` and its aliases shadow the structs' names. Hand-written
  code was migrated by a rewriter (`Line<a, b, Intrinsic>` to `intrinsic::Line<a, b>`, a generic
  `O` to `oriented::Line<a, b, O>`).
* **Orientation arithmetic that reduces for generic `O`.** The products and the flip move into a
  sealed base trait, and `Orientation` states the sign laws as supertrait bindings
  (`O·E = O`, `O·I = Flip<O>`, `O·O = E`, `Flip<Flip<O>> = O`), so helpers generic over `O`
  typecheck (one binding per bound, as 1.89 requires).
* **`f64` by default.** The default coefficient type becomes `f64` (and with it the kind markers,
  `Kind for oriented::Line<(), f64, O>`): callers wrote `<(), f64>` some 16 000 times against about
  100 for `f32`, and `Point::xyz(1.0, ..)` already inferred `f64` while the type `Point` meant
  `f32`. GPU code names `f32` explicitly, as it does its `*Gpu` layouts.
* **One name per idea, and one idea per name.** `normalized()` returns `Self`; the certified unit
  is `unit()`. `Log` has an associated output, so `m.log()` needs no annotation; it is implemented
  on the versor kind (`M::log(unit)`), since `impl Log for Unit<M>` would implement a foreign trait
  for a foreign type. A map is applied
  with `of` everywhere (`SparseMap`, with `of_into` for a kept field, `PackedSparseMap`,
  `SquareLinearMap`, structured maps) and densified with `to_dense`: `Prepared` is a
  `LinearMap` (its output kind comes from a hidden trait on the input kind, which the algebra
  owns; `impl LinearMap for Prepared<..>` in generated code would break the orphan rule), so it
  binds into dense maps, and it and the structured maps keep an inherent `to_dense` that needs
  no import. The tracer's facade re-exports `Traceable`, so `gax::trace` is one module with or
  without the feature. The jet module is `jet` and the study internals private. ADR-049's
  `Complex` is `mesh::Topology` (its accessor `Mesh::topology()`), since it holds the
  combinatorics, the simplices of every dimension with their incidences and boundary maps, while
  `Mesh` adds the positions and `dec::Geometry` the metric; the name `Complex` stays the root's
  complex numbers.
* **Linear algebra results and failures.** `Svd { values, inputs, outputs }` replaces the
  positional tuples of `svd` (`(u, σ, v)`) and `svd_thin` (`(σ, v, u)`): one `svd`, for maps of
  any shape, one triple per input blade with `m.of(inputs[i]) = values[i] outputs[i]` (on a
  square map the same factorization as the square `svd` was). The eigenproblems keep their
  `(values, vectors)` tuples, which agree. There is one name, `eigh`, and the metric is an
  argument: a map's `m.eigh()` is its own eigenproblem, while a form's is always
  `form.eigh(metric)`, because a form's eigenvalues against the identity change with the basis
  (it transforms as `Pᵀ F P`, not `P⁻¹ F P`); the identity-metric form `eigh()` is gone, and
  `form.eigh(Scalar::identity())` names that metric where the coefficient basis is orthonormal.
  The sparse one is `SparseMap::eigh(metric, count)` (`eigh_block` for larger Lanczos blocks);
  `eigh_semidefinite(metric)` keeps its name, since it is a different problem. A pairing's solve
  is `solve` (the trait method too), like a square map's, since the types tell them apart: a
  map's solves `m.of(x) = b`, a pairing's `p(x, ·) = r(l, ·)`. Its dyad binding stays
  `of_dyad` and `lstsq_dyad` (`OfDyad`, `LeastSquaresDyad`), apart from `of` and `lstsq`, which
  bind one slot.
* **One failure policy.** The plain methods keep their behaviour: a singular dense map's
  `inverse` and `solve`, and a pairing's `solve`, return non-finite coefficients, which keeps
  them branch free on lanes and in traced kernels; the field maps' `solve` and `solve_direct`
  panic with the reason. A checked twin returns an `Option`: `try_inverse` and `try_solve` on
  square maps (trait methods of `SquareMap`, with `SiteField`'s forwards), `try_solve` on
  pairings, `try_solve` and `try_solve_direct` on field maps. It runs the same factorization
  (shared helpers, no second code path) and returns `None` when the map is singular or the
  result is not finite. Singular is one test: an LU pivot at most `n ε max|aᵢⱼ|` in magnitude,
  `numpy.linalg.matrix_rank`'s tolerance with the pivots in place of the singular values, in any
  lane. A test on the result alone (non-finite) misses the usual case, a pivot that rounding left
  at `10⁻¹⁷` instead of zero, whose inverse is huge and finite. The sparse `factor` keeps its
  exact test (a column with no nonzero pivot), since shift-and-invert eigensolvers factor nearly
  singular maps on purpose (the relative test refused the numga Dirac spheres' shifted pencils),
  and `try_solve_direct` adds the finiteness test; conjugate gradients fails
  where `solve` panics, a residual above `√ε`. The eigenproblems have no twin: a symmetric
  eigenproblem has no singular case, and a form's `eigh` against a metric that is not positive
  definite returns NaN, as documented.
* **A smaller visible surface.** Items only generated code needs (`True`, `False`, `FillList`,
  `KindEq`, `SplitLast`, `MoveToFront`, `HasCat`, `SlotArr`, `Reorient`, `NewtonStep`,
  `LogNeedsUnit`, `PreparedDense`, `orientation::reflect`, `slots::by_entries`, the `linalg`
  kernels and the `study` functions) move to a hidden `gx::__private`, which the generator
  names; `fill`, `permute`, `linalg` and `study` become private modules, the items that stay in
  public ones are `#[doc(hidden)]`, and `trace::write_kind_path` is private. Methods are emitted
  only where their bounds can hold, as the generator knows from its own tables: a product
  method where the kind has that product with some kind, `hodge` where it has one with the
  pseudoscalar, `transform`, `transform_inv` and `prepare` on the values of a versor kind whose
  sandwiches were generated, `outermorphism` on the maps of a base kind to itself, and the
  methods of forms and pairings (`as_map`, a pairing's `solve`, `eigh(metric)`,
  `eigh_semidefinite`, `identity`) on the scalar kind. The standard algebras' generated code
  shrank by about 6 400 lines; a call that compiled before still does.
* **Geometry in gax types, evenly across algebras.** Constructors take gax values, not float
  tuples; coordinates enter only through the plain constructors (`Point::xyz`,
  `Point::direction`, `new`, CGA's `up`), the boundary to floats, as `to_euclidean` and the
  interop arrays are the way out.
  * Translations take a direction of either orientation, of which only the ideal part is read
    (`Motor::translation(q - p)` in PGA2D and PGA3D); PGA3D's `Motor::rotation_about(d, angle)`
    turns about the axis through the origin along `d`; `Plane::with_normal(n, d)` and its
    inverse `Plane::normal()`, and in PGA2D `Line::with_normal`, `Line::normal` and
    `Line::through(p, d)`; the twist constructors are `Twist::translation(v)` and
    `Twist::rotation(axis, ω)` in both PGAs (on `Line` in PGA3D, on `extrinsic::Point` in
    PGA2D, the kinds `Twist` names); CGA's `sphere(centre, r)` takes a round point.
  * New helpers, each with a doctest that pins its sign: `Line::direction()` (intrinsic, read
    from the Euclidean part: `q - p` for a join `p & q`, `a × b` for a meet `a ^ b`; in PGA2D the
    normal turned a quarter clockwise), `Point::distance(q)`, `Unit::identity()` and `Default`
    for every versor kind with a scalar part (in gax-core, generic: an inherent impl on
    `Unit<pga3d::Motor>` would be one on a foreign type), VGA3D `Rotor::rotation(axis, angle)`,
    VGA2D `Rotor::rotation(angle)` and `angle()`, `Rotor::interpolate` in both, and `Between`
    for VGA vectors (and VGA3D bivectors, planes through the origin), giving `Rotor::between`.
    `Plane::orthogonal_to`, `Motor::rotation_between` (both PGAs), `look_at`'s `up` and CGA's
    `from_point` take either orientation.
  * PGA3D's `Line::moment` was left out: the moment has no gax kind of its own as a vector (it
    is neither a position nor a direction), and the plane through the line and the origin, a
    join, says the same in gax terms.
  * Redundant helpers went: `Motor::from_rotor` (`r.widen()`, or `Motor::from` for a plain
    rotor), `Moments::moved(m)` (now `m >> moments`, an impl of `Transform` for
    `Unit<Motor>`, and `m << moments`). `Motor::rotation_between` stays: it turns one direction
    into another, where `Between` on points translates.
  * Every standard spec names the roles (`Twist`, the rate of a motion, extrinsic; `Forque`, its
    dual, intrinsic; `Even` and `Odd`, the versors) where they are not kinds already, so generic
    code and documents can rely on the names; they are the kinds `Motions` uses.
* **Conformal and Lorentz motions** (2026-10-19), as batch 3 for the algebras it left. CGA3D's
  `Motor::translation(d)` (the Euclidean part of `d`, so `q - p` of round points of weight one),
  `Motor::rotation_about(d, angle)` and `Motor::rotation(axis, angle)` about a direct line
  `p ^ q ^ ei` (right-handed about `q - p`, as PGA3D's `p & q`; the rotation about the parallel
  axis through the origin, conjugated by the translation to the line's foot), CGA2D's
  `translation` and `rotation(centre, angle)`, and in both `Even::dilation(centre, factor)`
  (`exp(½ ln s eo ∧ ei)`, built from `√s`, moved to the centre). A test pins the rigid ones to
  PGA's through the homomorphism, which already carried PGA's motors over with `widen()`; the
  constructors keep CGA code in CGA and add the dilation, which PGA cannot express. STA's
  rotor kind is `Even`: `Even::boost(direction, rapidity)` (`exp(½ζ d̂ e0)`, taking `e0` to
  `cosh ζ e0 + sinh ζ d̂`; rapidities add, and a speed enters as `β.atanh()`) and
  `Even::boost_between(from, to)`, the pure boost `(1 + w u)/√(2(1 + w·u))` between unit
  four-velocities. Rotations about a circle (conformal, the exponential of its dual) stay
  `exp` of the user's bivector.
* **What the example cleanups asked for** (2026-10-19). Five helpers the examples had written
  for themselves, and two refusals:
  * *Constants.* The named values stay functions generic over `T: Real` (`Point::origin()`,
    `Plane::infinity()`, `Line::infinity()`, `Unit::identity()`), since the tracer's `Sym` and
    the jets build their zero at run time. The coefficient types whose zero and one are
    constants implement `ConstCoef` (`ZERO`, `ONE`: `f32`, `f64`, the `wide` lanes), and the
    points and flats are associated consts generic over it, `Point::ORIGIN`, `Plane::INFINITY`,
    PGA2D's `Line::INFINITY`: one impl, so the coefficient type is inferred as for `new` (consts
    on the concrete `Point<(), f32>` and `Point<(), f64>` would be two impls, ambiguous where
    the type is not named, E0034). The constructors `new` were `const` already. Not every
    kind gets one: only these values have names, and a versor identity would need a const per
    versor kind (`Unit` is gax-core's and builds its value through trait methods), while
    `Unit::new_unchecked(Motor::new(1.0, ..))` is `const` today.
  * *`IdealNorm`*, the length of a PGA direction, in both PGAs and either orientation (points
    and `Direction`s): `ideal_norm_squared`, `ideal_norm`, and the scalings `with_ideal_norm`
    and `clamp_ideal_norm`, branch free through `select_lt` (a zero direction stays zero, a NaN
    stays NaN), so warp's `geom::Direction` needs no local trait. The inherent `ideal_norm`
    methods stay, so concrete code needs no import. It is the ideal norm, not a "length" of
    every kind: VGA vectors have `norm`, and a trait over both would name two ideas once.
  * *VGA3D `Rotor::rotation_in(plane, angle)`*, in the plane's sense (`a ^ b` turns `a`
    towards `b`); `rotation(axis, angle)` is `rotation_in` of the plane with the axis's
    coefficients, `axis.dual()`, as its doctest pins.
  * *`ApproxEq::abs_diff_eq(&b, tol)`*, an absolute comparison (`max_abs_diff(&b) <= tol`, the
    `approx` crate's name), beside the relative `approx_eq`: the examples compare positions in
    pixels and fields in their units.
  * *`Unit::orbit(start, n)`*, the `n` steps `(x, m >> x)` of repeating a unit versor, generic
    over what it moves: warp's chords of circles and screws, the arms of reticles.
  * *Forms and pairings refuse a larger output kind.* The blanket impls of `Form` and
    `Pairing` take any extensor with two slots and read coefficient 0; each method now asserts
    at compile time that the output kind has one coefficient, as `FormAsMap::as_map` does (a
    post-monomorphization error, pinned by `compile_fail` doctests, since `cargo check` and so
    the trybuild tests do not reach it).
  * *A value's `try_inverse()`*, generated beside `inverse()` and a method of `Inverse`: the
    one failure policy above, for values. A closed form divides by the Study number
    `N = x ~x`, and the test is on it, scaled as the LU test is: `None` where `|N| ≤ n ε ‖x‖²`
    (`n` coefficients, `‖x‖²` and `|N|²` the sums of the squared coefficients; `N` vanishes to
    its own rounding, an ideal plane or line, a round point) or where its norm cancels,
    `|N conj(N)| ≤ n ε |N|²` (a zero divisor `1 + E` with `E² = 1`, as in CGA3D's `Even`), in
    any lane, and then where the result is not finite. The general inverse (Shirokov's) has no
    Study number, so its twin tests the residual `x x⁻¹ − 1` against `√ε`, as conjugate
    gradients does. Both tests are relative: a value whose coefficients are all at rounding
    size is a scaled value like another, as for maps.
* **Plan.** Three batches, each merged when CI is green: (1) the generator and types (orientation
  by name, the arithmetic, `f64`, `Slot`, `Log`, `normalized`/`unit`); (2) names and linear
  algebra; (3) geometry helpers and uniform roles. Internal cleanups of the generator (output
  byte-identical) and of gax-core go first.

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
