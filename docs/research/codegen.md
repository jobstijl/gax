# How GAmphetamine.js, kingdon and ganja.js generate GA code

Research notes for a Rust GA library built on extensors, with exact integer operation tables and symbolic codegen.

Local clones used (paths are relative to `scratchpad/ref/`):

| project | repo | commit examined | license |
|---|---|---|---|
| GAmphetamine.js | https://github.com/enkimute/GAmphetamine.js | `617ab83` (2026-09-07) | The `LICENSE` file is MIT, but its copyright line is still the template placeholder `Copyright (c) [year] [fullname]`, `package.json` says `"license": "ISC"`, and the README shows an MIT badge. **The license is ambiguous.** Reimplement the algorithms instead of copying code, or ask enkimute (Steven De Keninck) to clarify. |
| kingdon | https://github.com/tBuLi/kingdon | `1084c70` (2026-09-17) | MIT, (c) 2022 Martin Roelfs |
| ganja.js | https://github.com/enkimute/ganja.js | `6e97cb4` (2024-01-14) | MIT, (c) 2017 Steven De Keninck |

I ran the code where I could: GAmphetamine and ganja under `deno run --unstable-unsafe-proto`, and kingdon with Python 3.14 and sympy. The generated-code excerpts below come from those runs. The scripts are `scratchpad/gam_demo.mjs`, `gam_rules.mjs`, `gam_dual.mjs` and `ganja_test.js`.

---

## 0. TL;DR: what to take

1. **Symbolic pipeline (GAmphetamine).** Expand every operator over symbolic coefficients into exact sparse polynomials. Reduce them with *type conditions*, which are polynomial identities such as `R~R = 1`. Run a *portfolio* of CSE heuristics and keep the cheapest result under a mul/add/div cost model, then do string-level cleanups such as reciprocal hoisting. The polynomial side is sound. The CSE side is a large stack of pattern-specific greedy heuristics (≈1500 LOC) that is validated by numeric equivalence tests, not by construction. We should take its *ideas* and *regression targets* (op counts, listed below) and build a cleaner, verifiable pass in Rust. One option is kernel/co-kernel extraction plus Horner factoring, with every rewrite verified by expansion.
2. **Type conditions must be proper ideal reduction.** GAmphetamine's rewrite rules are ad hoc and can oscillate between rules; they stop at a 32-round guard. In Rust, compute a small Gröbner basis (or use its linear-substitution special case) under a chosen monomial order, and normal-form every coefficient. This makes "normalized motor ⇒ inverse is the reverse" and "sandwich with unit rotor drops the `a0²+…` scale" fall out exactly.
3. **Typed layouts with fixed coefficients.** Both GAmphetamine (`fixed:[0,0,0,1]`) and kingdon (layout dict `{key: ... | 1.0}`) treat a known constant coefficient, such as the `e123 = 1` of a normalized point, as a literal during symbolic evaluation. It is not stored and not passed as an argument. The output type is then *inferred* from which coefficients are structurally nonzero and which equal the fixed constants.
4. **Lazy per-(type, type) compilation** with a jump table (GAmphetamine) or a dict cache keyed by `(type, keys)` (kingdon). In Rust this becomes codegen at build time (proc-macro or `build.rs`) for the type pairs we declare, plus an optional runtime-JIT fallback that isn't needed.
5. **Conventions to match bivector.net.** PGA3D basis `1,e0,e1,e2,e3,e01,e02,e03,e12,e31,e23,e021,e013,e032,e123,e0123`, and PGA2D basis `1,e0,e1,e2,e01,e20,e12,e012`. Duality is the *sign-free coefficient reversal* ("Poincaré/J map"). Points are `x e032 + y e013 + z e021 + e123`, and planes are `a e1 + b e2 + c e3 + d e0`. Watch the join sign, which differs between ganja/bivector.net and GAmphetamine/kingdon (§3.5).

---

## 1. GAmphetamine.js (`src/`)

### 1.1 Summary

GAmphetamine is ganja.js's successor. It is a JS algebra generator with a *coefficient-level symbolic engine*. Every operator (`gp, op, ip, lip, rip, dual, undual, reverse, …, sw, inverse, normalized, sqrt`) is written once as a symbolic function over flat `2^n` arrays of rational polynomials (`src/symbolicOperators.js`). For each pair of concrete *types*, the operator is evaluated on symbolic inputs `a[0..]`, `b[0..]`. The resulting polynomials are simplified, CSE'd, formatted and compiled with `new Function`, all lazily on first use (`src/GAmphetamine.js:555-1177`).

File map:
- `polynomial.js` (1507 lines): the sparse polynomial type and the whole CSE pipeline.
- `rationalPolynomial.js` (911 lines): numerator plus a *factored* denominator, sqrt atoms, constraint rules, rational CSE wrapper, and string-level post-processing.
- `symbolicOperators.js` (253 lines): products and involutions over any coefficient type.
- `GAmphetamine.js` (1362 lines): options, basis, Cayley, types, `Element.compile` (codegen), dispatch tables, numeric fallbacks (Shirokov inverse, invariant split, exp/log).

### 1.2 Basis, metric, Cayley (`GAmphetamine.js:50-153`)

- **Metric order.** When built from p, q, r, generators are ordered *null first, then +, then −* (`:97`). `startIndex` is 0 if the first generator is null, otherwise 1 (`:100`).
- **Default basis.** Blades are sorted by grade, then lexically (`:103-105`). A custom `basis` may use any orientation, for example `e31` or `e021`.
- **Natural basis.** `naturalBasis` is each name with its indices sorted, looked up through the hash `nbHash`. `basisPermutation[i]` is the sign of the user blade relative to the sorted one (`:108-114`).
- **`contract(args)`** (`:124-143`) implements the contraction axiom on an index string, for example `'1223'` for `e12*e23`. It bubble-sorts, flipping the sign per transposition, and removes equal pairs times `metric[i]`. It then multiplies by `basisPermutation[idx]`. It returns `[sign ∈ {-1,0,1}, index]`.
- **Cayley table.** `Cayley[i][j] = [sign, idx]` is precomputed when `n ≤ 8` (`:147`). Above that, `contract` runs per pair.
- **Hodge-like dual** (`:117-121`). `dualBasis[i] = [sign(e_comp·e_i), comp(i)]` and `dual(a)[i] = ±a[comp(i)]`. Measured in 3DPGA, it maps `e0→e123, e1→e032, e2→e013, e3→e021, e01→e23, e02→e31, e03→e12, e032→−e1, e013→−e2, e021→−e3, e123→−e0`, so it satisfies `x ∧ dual(x) = I`. `undual` is its inverse. `rp(a,b) = undual(op(dual a, dual b))` (`:244`), which gives `e123 ∨ e032 = +e23`.
- **Named algebras** (`:53-85`):
  - `"3DPGA"` has basis `1,e1,e2,e3,e0,e01,e02,e03,e12,e31,e23,e032,e013,e021,e123,e0123`. This has the **same blade orientations** as bivector.net, but a different memory order: `e0` comes last among the vectors, and the trivectors are ordered `e032,e013,e021,e123`.
  - `"2DPGA"` has basis `1,e1,e2,e0,e20,e01,e12,e012`.
  - STA uses `startIndex 0`. CGA uses a lexical, diagonal metric, with no null basis.

### 1.3 Polynomial representation (`polynomial.js:1-270`)

```
poly  := 0 | [term, ...]              // sorted by `compare`
term  := [coeff, v1, v2, ...]         // coeff: number|bigint; vars: strings, SORTED, repeated for powers
e.g.  x² + xy + 3  =  [[1,'x','x'], [1,'x','y'], [3]]
```

- `compare` (`:37`) is lexicographic on the factor list, with shorter lists first when one is a prefix of the other. The coefficient is ignored.
- `add` (`:52`) is a merge of two sorted lists that sums the coefficients of equal monomials and drops zeros. `mul` (`:77`) forms all pairwise products, merges the sorted factor lists, and accumulates with `add`.
- Zero is the literal `0`, so exact cancellation is structural. That is how sparsity is discovered.
- Helpers:
  - `monomialGCD` (`:134`): the common monomial factor.
  - `divmod`/`divide` (`:185-219`): leading-term division, exact if the remainder is 0.
  - `detectSquare` (`:251`): a sparse square root by peeling the leading term. It takes `q₀ = √lead`, then `qᵢ = lead(rem)/(2·lead(Q))`, and verifies `Q² = P`.

**Rational coefficients** (`rationalPolynomial.js`). An element is `[N, F]`, where `F = [[Q₁,n₁],…]` is a *factored* denominator that is never expanded.
- `mul` merges factor lists by summing exponents (`mulF`).
- `add` cross-multiplies only the non-shared factors (`splitF` plays the role of a gcd by factor identity).
- `cancelF` (`:171`) tries exact polynomial division of N by each factor, then cancels numeric content and common monomials.
- `inv` swaps numerator and denominator.
- **Sqrt atoms.** `sqrt(poly)` becomes an opaque variable named `"(poly)**.5"`, registered in `sqrtRoots` (`:37-69`). Whenever a product contains `atom·atom`, `reduceSqrtPairs` replaces it with the radicand. A denominator `(√R)^{2k}` becomes `R^k` (`reduceSqrtF`). This is why `1 - a.normalized()*~a.normalized()` compiles to `return 0`.

### 1.4 Symbolic operators (`symbolicOperators.js`)

- All products are the same double loop over nonzero `a[i]`, `b[j]`, using `Cayley[i][j] = [metric, idx]` with a grade filter (`:35-115`):

  | product | kept when |
  |---|---|
  | `ip` | `|gᵢ−gⱼ| == g_out` |
  | `lip` | `gⱼ−gᵢ == g_out` |
  | `rip` | `gᵢ−gⱼ == g_out` |
  | `op` | `gᵢ+gⱼ == g_out` |

- Involutions use grade masks: reverse negates `g&2`, involute negates `g&1`, conjugate negates `g%4%3`.
- `type(x)` (`:31`) is the *smallest* declared type whose layout covers every nonzero coefficient and whose `fixed` entries match the formatted coefficient exactly. "Smallest" means fewest non-fixed slots.
- `inv` handles scalars and Study numbers only. In a degenerate 4D algebra, `inv(s + pI) = 1/s − p/s²·I` (`:185-199`).
- `sqrt` (`:211`):
  - For a Study number in PGA: `√s + p/(2√s) I`.
  - For a general Study number: `a = √((s + √(s² − p²I²))/2)`, `b = p/(2a)`.
  - Otherwise: `sqrt(x) = normalized(1+x)`. Before the root is frozen into an atom, the Study square `(1+x)(1+~x)` is first reduced by the active type conditions.

Derived methods (`GAmphetamine.js:237-292`):
- `sw(a,b) = grade(a b ~a + b(1 − ⟨a~a⟩₀), grade b)` for even `a`, and `grade(a b̂ ~a)` for odd `a`. The added `b(1−⟨a~a⟩₀)` term is exact for unit versors and cancels the `|a|²·b` terms, so it is a cheap stand-in for a type condition. kingdon copies this trick (`operators.py:88-113`).
- `inverse`: if `a~a` is a scalar, return `~a/(a~a)`. If `n ≤ 3`, `adj = ~a·involute(a~a)`. In 4D PGA, if `a~a` is a Study number, return `inv(a~a)·~a`. Otherwise use the 4D Hitzer–Sangwine form `adj = ā·(a ā with grades 3,4 negated)`, then `adj·(a·adj)⁻¹`. If none of these apply, return `undefined`, which falls back to the numeric Shirokov inverse (`:423-438`).
- `normalized`: `a · inv(sqrt(a~a))`, with the result *frozen* (see §1.7).

### 1.5 Typed layouts, fixed coefficients, lazy compilation

Types are `{name, layout:[blade names in memory order], fixed?:[0|const,...], condition?: b=>mv}`.

**Default types.** When `types` isn't given, the defaults are the k-vectors, `direction` (only for PGA3D), `study`, `even`, `odd` and `multivector` (`:159-174`). The 3DPGA preset adds (`:55-78`):

```
point      layout [e032,e013,e021,e123]  fixed [0,0,0,1]
dpoint     layout [e1,e2,e3,e0]          fixed [0,0,0,1]
translation layout [e01,e02,e03,1]       fixed [0,0,0,1]
rotation   [1,e23,e31,e12]; ebivector, ibivector, evector, horizon (e0 fixed 1), origin (e123 fixed 1)
```

**Fixed coefficients.**
- Storage size is `layout.length − #nonzero fixed` (`:373`). The fixed entries must be *last* in the layout, because the numeric-to-symbolic conversion indexes `a[i] ?? fixed[i]` (`:311`).
- The symbolic `create` puts the constant in place of the symbol (`symbolicOperators.js:24`), so constants propagate. For example, point ∨ point in 3DPGA compiles to `res[0]=b0-a0; …; res[3]=a1*b2-a2*b1`, with 6 muls.
- On output, fixed slots are neither written nor stored (`:619`). Output type inference only picks a fixed type if the coefficient formats *exactly* to the constant, as in `fitsType` (`:598`).

**Lazy per-type-pair compilation** (`Element.addMethod`, `:1180-1206`).
- For each method, build a table of size `(T+1)×(T+1)`. Each entry starts as a bound thunk `Element.compile(func,[i,j],name,table)`.
- The first call symbolically evaluates `func(symvars[0][i], symvars[1][j])`, generates the JS source, `new Function`s it, *overwrites* `table[i][j]` with the compiled function, and calls it.
- Dispatch is `ta[B.tp](this,B,R)`, where every class prototype has an integer `tp`.
- Slot `T` handles symbolic multivectors.
- `precompile:true` fills the whole table eagerly. For 3DPGA that is ~333 kB of JS in ~700 ms, per the README.
- Compiled functions take an optional preallocated `res`.

**Prefetch** (`:674-704`). Emit `const a0=a[0]` only for input coefficients used more than once. Single-use ones stay `a[4]` inline.

### 1.6 Type conditions (`GAmphetamine.js:190-230, 567-614`; `rationalPolynomial.js:205-288`)

**Declaration.** A type carries `condition: b => <multivector expression that must vanish>`, for example `{name:'motor', layout:[1,e23,e31,e12,e01,e02,e03,e0123], condition: b => 1 - b*~b}`. Types are added with `extraTypes`.

**Turning conditions into rules.**
- At compile time, `condition(symArg)` is evaluated *symbolically* for each input whose type has a condition (`:567-572`). Every nonzero coefficient polynomial `P` of the result is an identity `P = 0`. For a PGA motor this yields `1 − (a0²+a1²+a2²+a3²)` and `2(a0a7 − a1a4 − a2a5 − a3a6)`.
- `constraintRules(P)` (`rationalPolynomial.js:255`) first normalizes P: it divides out the gcd of the coefficients and makes the leading coefficient positive. Then:
  - If P has a constant term `c`, emit:
    1. **Elimination rule.** The first non-constant monomial goes to `−(c + rest)/coef`, e.g. `a0² → 1 − a1² − a2² − a3²`.
    2. **Complement rules.** For each term t, (sum of the others) → `c' − t`, e.g. `a1²+a2²+a3² → 1 − a0²`.
    3. **Whole-sum rule.** `a0²+a1²+a2²+a3² → 1`.
  - If P has no constant term: `P → 0`.

  Dumped for the motor (`scratchpad/gam_rules.mjs`):
  ```
  [a0a0] -> 1 - a1a1 - a2a2 - a3a3
  [a1a1+a2a2+a3a3] -> 1 - a0a0        (and 3 more complements)
  [a0a0+a1a1+a2a2+a3a3] -> 1
  [a0a7 - a1a4 - a2a5 - a3a6] -> 0
  ```

**Applying a rule** (`applyConstraintRule`, `:230`). For each monomial `mt` of the rule's match and each term `pt` of the polynomial:
1. Compute `factor = pt / mt`, a monomial quotient that includes the coefficient.
2. Require every term of `factor·match` to be present in the polynomial with *exactly* equal coefficients.
3. Rewrite `poly ← poly − factor·match + factor·replace`.

`reducePolynomialByRules` applies all rules in sequence and iterates to a fixpoint, with a guard of 32 rounds. Numerators and each denominator factor are reduced, then cancelled.

**Where rules are used.**
- (a) On the final expression `AB` (`:596`).
- (b) On `a~a` inside `norm`, `normalized` and `inverse` (`reduceConstraints`, `:230`), so a motor's `a~a` becomes the scalar `1`. The inverse then becomes `~a`, with 0 muls, and `normalized` becomes a no-op.
- (c) On the Study square inside `sqrt`.

**Output type refinement** (`:605-614`). If a conditioned type has the same layout size and its `condition(result)` reduces to all zeros modulo the input rules, the result gets that type. This is how `normalized(even)` returns a `motor`.

**Freezing** (`freezeNormalized`, `:197-220`). When `normalized` is called *inside* a larger expression, its coefficients are replaced by fresh symbols `_f0.._fk`. Their definitions are kept separately, and the conditioned type's condition, evaluated on these symbols, is added to the rule set. So `a.normalized()*b*~a.normalized()` is simplified *as if* the rotor were a unit motor. The definitions are CSE'd separately and emitted first.

**Soundness caveat.** The rule set isn't confluent. Rule 1 eliminates `a0²` and the complement rule reintroduces it, so the loop can ping-pong until the guard trips. It works in practice for the unit-norm case.

**In Rust, do this properly.** Treat the conditions as an ideal `I = ⟨P₁,…,P_k⟩` over ℚ[vars]. Compute its reduced Gröbner basis under a graded order (tiny for these cases: often the generators themselves, plus S-polynomials), and normal-form every coefficient. For the extensor library, conditions such as `R~R = 1` are declared per type and are polynomial in that type's coefficients, so the basis can be cached per type.

Measured effect in 3DPGA:

| expression | result |
|---|---|
| motor sandwich of a point | 21 muls / 18 adds, and the output is typed `point` |
| same sandwich with an unconditioned `even` | 28 muls / 21 adds, output `trivector` with `e123 = _w0` |
| same, without CSE | 71 muls / 36 adds |
| motor inverse | 0 muls / 6 negations |

### 1.7 The CSE pipeline (`polynomial.cse`, `polynomial.js:1395-1483`)

**Inputs.**
- `expr`: an array of output-component polynomials. They are mutated in place and may become *nested*: a term whose last factor is an array means "factor list × (inner poly)".
- `prot`: protected variables, which are never merged into products. `Element.compile` passes `[]`.
- `iso`: the isolation candidates. `Element.compile` passes `[2, a[0], a[1], …, b[0], …, freeze names]` (`:651-655`).

It returns `[prelude: ["name=body", …], expr]`. Numerators of rational expressions go through `rationalCSE` (§1.8).

**Top-level control.**

```
cse(expr, prot, iso, opts):
  if single component and detectSquare(P)=Q:        # whole output is a perfect square
      pre,Qc = cse([Q], noSquare); emit _sq=Q; return _sq*_sq
  if opts.complete undefined:                          # PORTFOLIO
      A = cse(clone, complete=true)
      if A.completed: B = cse(clone, complete=false); keep argmin opCost(format(A)), opCost(format(B))
  (phases below, order is test-pinned)
```

The cost model is `opCost(s) = 1.0001·#'*' + 4·#'/' + 1·#('+'|'-')` on the formatted text (`:32`). `Element.compile` repeats the same with/without portfolio *after* string post-processing (`GAmphetamine.js:621-669`).

**Phases, in order.**

0. **`completeSquareSums`** (`:399`), only when `complete=true`. For each component, bucket degree-3 terms of the form `c·o·x²` by their outer variable `o`, and degree-2 terms `c·x²` under `o=''`. For a bucket with at least 2 squares of equal `|c|`:
   - Take the majority sign.
   - Let `D = Σx²`.
   - Compute `R = N − c·o·D` *exactly*, and keep it only if `|R| + 2 ≤ |N|`.

   Keep the best candidate per component, and only if the same `D` occurs in at least 2 components. Emit `_wK = D` once, rewrite `N = c·o·_w + R`, and optionally extract `R` as `monomial × shared sum` into `_vK`. This produces the `_w0=a0*a0+a1*a1+a2*a2+a3*a3; res[0]=_w0*b0+2*(…)` shape of the rotor sandwich.

1. **`findSharedSums`** (`:476`), which is the core "sum" CSE. For every (component, iso-var `v`) pair, collect the residual of the terms containing `v`, with `v` removed. For residuals of size 4, also try each 3-subset. Normalize the residual:
   - divide by the gcd of the integer coefficients;
   - sort;
   - make the first coefficient positive, recording the sign.

   The key is the canonical string. Candidates occur in at least 2 *different components*, with `score = 10·#components + #occurrences`. Take them greedily in score order, skipping term slots already used. Emit `tK = normSum`, then replace the occurrence terms by `gcd·sign·v·tK`. For 2-term sums `tK = x − y`, record `x ↦ tK + y` in `sumMap`.

2. **`substituteExtracted`** (`:668`). Substitute `x = tK + y` everywhere and re-normalize, which can cancel terms. Then run `findSharedSums` once more over the new t-vars. This reveals differences such as `(c−a)`.

3. **`detectLinearDeps`** (`:1199`), only if sums fired. Take the heaviest component. If it equals `Σ ±cv·comp_i`, where each `cv` is a variable exclusive to it, emit `_linK` names later.

4. **`findSquareStructure`** (`:723`):
   - `'pair'` mode: `c X² + c Y² ± 2c XY → c (X ± Y)²`, with `X`, `Y` monomials, emitted as `tK`. Candidate squares are found by checking even multiplicities in the sorted factor list. Cross terms are looked up in a map keyed by the sorted factors of `X·Y`, with coefficient `±2c`.
   - `'triangle'` mode: `Σ c tᵢ² + Σ ±2c tᵢtⱼ → c(±tᵢ±tⱼ±tₖ)²`, where edges require the full product `tᵢ·tⱼ` expansion to be present, and the sign product must be +1. Emitted as `uK`.

5. **`findBilinearDiffs`** (`:909-1050`). Expand the u-vars to base variables. A 6-monomial alternating-sign 6-cycle over 6 distinct variables is factored as `(A−B)(C−D) ± (E−F)(G−H)`. Each of the 3 anchors is tried, and the one that reuses the most existing `_dK` differences wins. The result is verified by re-expansion. This targets cross-product-of-differences shapes such as the 4-point join determinant.

6. **`bindSameVarPowers`** (`:634`). An adjacent `x·x` that occurs in at least 2 terms becomes the helper `xx = x*x`.

7. **`isolate`** (`:537`), a greedy multivariate Horner factoring. For each `p` in `isoList = reversed(iso vars) ++ numeric iso (2)`, and in each component:
   - Split the terms into those with and without `p`.
   - Strip `p` from each term that has it, and also strip every factor common to all of them.
   - If all inner terms are negative, hoist a `−1`.
   - Rebuild the component as `terms_without_p ++ [[prefix…, p, inner]]`.

   Only top-level factor lists are searched, so the first isolated variable ends up outermost. Numeric `2` is isolated last, which produces the `2*(…)` wrappers.

8. **Post-isolate passes:**
   - `normalizeTermScalars` folds signs back into the coefficients.
   - `findGroupMerges` (`:1061`): if `Σ v1_k P1_k + Σ v2_k P2_k` satisfies `Σ v2_k P1_k = ±Σ v2_k P2_k` as a polynomial identity, merge it to `Σ (v1_k ± v2_k) P1_k`. Families are recognized by the name regex `a0 | a[0] | a_{0}`.
   - `findNestedBilinearDiffs`.
   - `extractSharedNestedSums` (`:1167`): nested inner polynomials seen more than 2 times become `_SK`.

9. **Linear-dependency application**, then **`findFactoredSquareSums`** (`:837`). Per component and per candidate outer factor `o`, run pair-square matching on the terms divided by `o`: `o(−X²−2XY−Y²) → −o·q², q = X+Y`.

10. **`findSharedProducts`** (`:649`), repeated to a fixpoint. Count every unordered factor pair `(term[i], term[j])` inside each leaf term across all components. Pairs with count > 1 are replaced by the combined atom `a0b1 = a0*b1`. This is plain greedy pairwise product extraction by frequency, with no cost-benefit beyond "occurs ≥2 times". Then `findFactoredSquareSums` runs once more.

11. **`finalizeCSEPrelude`** (`:1309`), which works *on strings*:
    - Repeated `x*y` in prelude bodies become helpers.
    - Atomic product helpers are hoisted to the front and substituted back.
    - A parenthesized substring seen at least twice becomes `_sK`.
    - Iterated `sharedTermPairs`: any `±t_i ± t_j` pair of top-level terms shared by at least 2 entries (sign-canonical, flip-aware) becomes `_pK`, repeated to a fixpoint.

**Takeaway on CSE.** There is no hash-consing and no DAG. CSE is a sequence of *pattern detectors* over the flat sum-of-products form: shared residual sums, perfect squares, differences, bilinear forms, Horner isolation and frequent factor pairs. Every accepted rewrite is exact, either because it is verified by expansion or because it is algebraic by construction. Global choices are made greedily by frequency. The only *global* cost comparison is the complete/no-complete portfolio. Correctness is checked in tests by evaluating at random points (`test/polynomial.test.js:34-60`, `expectEquivalentCSE`).

### 1.8 Rational CSE and reciprocal hoisting

**`rationalCSE`** (`rationalPolynomial.js:395-531`):
1. If `F` contains a squared factor `s`, or a sqrt atom whose radicand is `s`, split `N/(…s…)` into `P/(F/s) + Q/F`, where `N = P·s + Q` by `divmod` or by trying `±v·s` for single variables. The two parts become separate CSE slots.
2. Collect the unique denominators, keyed and sorted by degree, as `D1, D2, …`. Single-variable denominators, such as a sqrt atom, are aliased.
3. Run the polynomial CSE on the numerators only.
4. Emit the denominators factor-wise, reusing names. A power becomes a product of names (`D2=D1*D1*D1`). A factor equal to a sqrt atom's radicand is emitted as `atom*atom`, so `√s·s` becomes `D1*D1*D1`.
5. Recombine the split fractions.

**`postprocessCSE`** (`:540-909`) is also string-level. It:
- rewrites `x*(p − a − 2·√a√b − b)`-style squares to `q*q`;
- flips helper signs when their negated uses dominate (`flipDef`);
- hoists repeated `(…)**.5` and repeated parenthesized sums into `_rK`;
- extracts shared term pairs;
- factors a common atom or numeric coefficient out of sums;
- extracts frequent adjacent products and `2*x` / `2*(product)` helpers;
- eliminates dead entries.

**Reciprocal hoisting** (`GAmphetamine.js:706-897`, on by default with `reciprocalHoist:true`). Steps:
1. Find every `(numerator)/(den)` in the prelude and outputs.
2. A denominator seen more than once becomes `_ivK = 1/(den)`, and the uses are rewritten to `num*_ivK`. Numeric denominators become multiplication by a constant.
3. If `den` is defined as `x*x*x` or `x*x` and `x` already has a reciprocal, emit `_ivK = _ivJ*_ivJ*_ivJ` instead of another division. A prelude body `D*(sum)`, where `sum` is `D`'s radicand, is rewritten as `D*D*D` first.
4. `splitNormalizedCubes` rewrites `(num)*_iv3`, where `num = base·Σroot² ± common·q`, into `base*_iv0 ± common*q*_iv3`.
5. `hoistLateProducts` and `hoistSharedResultFactors` extract repeated products across outputs.
6. `normalizeSigns` greedily flips temporaries' signs to remove leading negations, using the add-count cost.

Result:

| operation | ops | code |
|---|---|---|
| 3DPGA `even.normalized()` | 23 muls / 1 div / 10 adds | `D1=(a0²+a1²+a2²+a3²)**.5; _iv0=1/D1; _iv1=_iv0³; t0=a0a7−a1a4−a2a5−a3a6; res[4]=a4*_iv0+a1*t0*_iv1 …` |
| R₃,₀,₁ bivector inverse | 18 muls / 1 div / 8 adds | — |
| kingdon's equivalent of that inverse | 43 muls / 6 divs | kingdon doesn't hoist reciprocals |

### 1.9 Numeric fallbacks (`GAmphetamine.js:421-547`)

- **Inverse.** For `n ≤ 5`: `cir = ā·â·ã`, `adj = cir·gradeInv₁,₄(x·cir)`, `x⁻¹ = adj/⟨x·adj⟩₀`. Otherwise the **Shirokov** iteration: with `N = 2^⌈n/2⌉`, repeat `U_k = x·(U_{k-1} − (N/k)⟨U_{k-1}⟩₀)` and use the last adjugate.
- **`split()`.** An invariant bivector decomposition: closed form for PGA below 6D, a quadratic for `k < 3`, otherwise via the eigenvalues of the characteristic polynomial.
- **`factorize()`**, **`exp`** (per simple part: cos/cosh/1+B) and **`log`**.

### 1.10 What to take from GAmphetamine

- **Architecture.** A symbolic evaluation of generic operator definitions over typed symbolic inputs. Then exact sparse-polynomial simplification, ideal reduction by type conditions, a CSE portfolio chosen by cost model, and emission. This maps directly onto our extensor tables: an extensor with integer tables applied to symbolic coefficient vectors yields the same polynomial systems.
- **Must-have features:** fixed coefficients; conditioned types, including output-type refinement and freezing of normalized intermediates; a factored rational representation with sqrt atoms; hoisting of shared reciprocals, including power reuse (`1/D³ = (1/D)³`); prefetch-only-if-reused; a cost-model portfolio.
- **Regression targets.** Reproduce these op counts in Rust tests:

  | case | muls | divs | adds |
  |---|---|---|---|
  | even gp even | 48 | 0 | 40 |
  | unit-motor sandwich point | 21 | 0 | 18 |
  | even sandwich point | 28 | 0 | 21 |
  | R3 rotor sandwich vector | 25 | 0 | 15 |
  | even inverse | 23 | 1 | 12 |
  | bivector inverse | 18 | 1 | 8 |
  | even normalized | 23 | 1 | 10 |
  | motor inverse (with condition) | 0 | 0 | 6 |

  Tests pin most of these (`test/GAmphetamine.test.js:700-860`).
- **Don't copy:** the string-regex post-processing or the fixed-point ping-pong rule application. Represent temporaries as a typed expression DAG with an explicit cost, and verify every rewrite by expansion or random evaluation.

---

## 2. kingdon (`kingdon/`)

### 2.1 Summary

kingdon is a Python GA library by Martin Roelfs. It keeps multivectors *sparse by keys*: `mv.keys()` is a tuple of binary blade indices that are present. Every operator is an `OperatorDict` that generates and caches one compiled function per input *signature*. It has **its own polynomial engine**, `kingdon/polynomial.py` (`Polynomial`, `RationalPolynomial`, `poly_cse`). The CSE is explicitly "ported from polynomial.js", an *older, smaller* version of GAmphetamine's pipeline. sympy is used only when a custom printer is requested, or when components have different denominators (`sympy.cse` fallback, `codegen.py:425-428`). It optionally wraps generated code with numba, and supports array or torch-valued coefficients.

### 2.2 Blades, signs, operator tables (`algebra.py`)

- **Signature and start index.** If `r == 1`, the signature is `[0]*r + [1]*p + [-1]*q` and indices start at 0 (PGA). Otherwise it is `[1]*p + [-1]*q + [0]*r`, starting at 1 (`:172-178`).
- **Binary keys.** Bit `j` ↔ the j-th basis vector, in the order given. `canon2bin` and `bin2canon` map names to keys. The order is grade-then-lexical, or taken from a custom `basis` (`:196-213`).
- **Sign table.** `signs[(I,J)]` is a lazily filled dict (`DefaultKeyDict(_compute_sign)`, `:215, 340-351`). `_swap_blades` counts the transpositions needed to bring the strings together (`:749-782`), then multiplies by the signature for each eliminated index. There is also an unused bit-twiddling version (`_swap_blades_bin`, `:685`) identical to ganja's `simplify_bits`.
- **Products via integer filters on binary keys** (`operators.py:46-233`). Every product is `product(x, y, filter_func, sign_func, keyout_func)` over `x.items() × y.items()`, with `key_out = kx ^ ky`. Worth copying:

  | product | kept when |
  |---|---|
  | outer | `k_out == kx + ky`, i.e. the key sets are disjoint |
  | symmetric inner | `k_out == |kx − ky|`, i.e. one key set contains the other |
  | left contraction | `k_out == ky − kx` |
  | right contraction | `k_out == kx − ky` |
  | scalar product | `k_out == 0` |
  | commutator | `signs[kx,ky] − signs[ky,kx] ≠ 0`, i.e. only anticommuting pairs. This equals ½(xy−yx) exactly, with no ½ factor. |
  | anticommutator | `signs[kx,ky] + signs[ky,kx] ≠ 0` |
  | regressive | `keyout = pss − (kx^ky)`, filter `pss == kx + ky − k_out`. The sign is the product of four signs: dual of each input, the wedge, and the undual. |

- **Hodge dual** (`operators.py:489-500`). `hodge(e_I) = sign(e_I, e_{pss−I}) · e_{pss−I}`, so `x ∧ hodge(x) = I`. `unhodge` uses `sign(e_{pss−I}, e_I)`. Polarity is `x·I` (with `−` if `I² = −1`), and unpolarity is `x·I`.
- **`dual(kind='auto')`** (`multivector.py:793-829`) uses **polarity if r = 0, Hodge if r = 1, and raises for r ≥ 2**. Measured in `fromname('3DPGA')`: `e1→e032, e2→e013, e3→e021, e01→e23, e032→−e1, e123→−e0`, which is the same as GAmphetamine. `e123 & e032 = +e23`.
- **`fromname`** (`:263-285`) gives:
  - 3DPGA: `e,e1,e2,e3,e0,e01,e02,e03,e12,e31,e23,e032,e013,e021,e123,e0123` (GAmphetamine's order);
  - 2DPGA: `e,e1,e2,e0,e20,e01,e12,e012`;
  - STAP.

  Custom bases must be grade-sorted (`:199`).
- **Degenerate metric.** Nothing special beyond the zero in the signature and the Hodge dual choice. `large` algebras (d > 6) skip codegen entirely and run `do_operation` directly (`:217-223`).

### 2.3 Codegen flow and caching

1. `OperatorDict.__getitem__(mvs)` (`operator_dict.py:145-153`) builds the key `types_in = tuple((type(mv), mv.keys()) for mv in mvs)`. On a miss, it creates symbolic multivectors for exactly those keys, with coefficient symbols named `a1, a12, …` after the arg name plus the blade (`RationalPolynomial.fromname`). It then calls `algebra.compile(codegen, *symbolic_mvs)` and caches the result. **The cache is per operator, per (type, keys) of every argument, so sparsity is in the cache key.**
2. `do_compile_symbolic` (`codegen.py:108-159`):
   - Run the codegen function, which is the Python GA expression itself (e.g. `ops.gp`), on the symbolic multivectors.
   - Zeros are dropped *structurally* in `dict_to_multivector` (`operators.py:34-39`), so the result's keys are its structurally nonzero blades.
   - Build `res_layout = {key: float | ...}` and run `resolve_layout` (`codegen.py:52-105`) to pick the most specific registered type. A type matches if its fixed values agree, it doesn't fix any blade the result leaves free, and it knows every fixed blade. The cost is `(#free slots that are fixed in result, #free slots outside result)`, lexicographically, with ties broken by registration order.
   - In `full_layout` mode, pad to the type's full layout.
3. `lambdify` (`codegen.py:343-451`):
   - If every expression is a `RationalPolynomial` and all non-unit denominators are *equal*, call `poly_cse` on the numerators plus the common denominator, with `iso=[2]+sorted(all_vars)` and `prot=None`.
   - Emit `[a, a1, …] = A` unpacking, the `name=poly` prelude lines, a single `_d = denom` when more than one output uses it, and `return [num/_d, …]`.
   - The function is compiled with `exec`, and the source is registered in `linecache` so `inspect.getsource` works. The docstring carries `n muls / n divs / n adds`.
4. `poly_cse` (`polynomial.py:431-490`) runs, in order:
   - `_find_shared_sums` (identical to GAmphetamine phase 1, including the 3-of-4 subsets);
   - `_substitute_extracted` plus a second sums round;
   - `_detect_linear_deps`;
   - `_isolate` (Horner, with the numeric iso `2` matched on the coefficient);
   - `_find_shared_products`: a single pass of pairwise factor frequency.

   It has **no** square completion, square/triangle/bilinear detection, reciprocal hoisting or portfolio.

Example: kingdon `sw` of a `Bireflection` on a vector in R₃,₀,₁ gives 22 muls / 15 adds:

```
t0=x*y1+x12*y2+x13*y3; t1=…; t2=…
return [y0+2*(x01*t0+x02*t1+x03*t2), y1+2*(x12*t1+x13*t2), …]
```

### 2.4 Types (`multivector.py:831-1025`; docs `docs/workings.rst`, `docs/types.rst`)

- A type's `layout` is either a dict `{blade: ... | number}` or a **classmethod that evaluates a GA expression symbolically**. Coefficients that come out numeric become fixed constants, and symbolic ones become free slots (`Algebra._bind_layout`, `algebra.py:708-747`). Examples:
  - `Bireflection.layout = gp(p, ~q)` for two symbolic vectors, which gives the even subalgebra in 2D and 3D.
  - `Point.layout = hodge(UPoint)`, where `UPoint = EVector + 1·e0`.
  - `Translation = Point·~Point`.
- For a point in 3DPGA, the layout is `{14:…, 13:…, 11:…, 7: 1.0}`. The `e123` coefficient (key 7) is **not stored and not an argument**; codegen substitutes the constant.
- `full_layout=True` (the deprecated alias is `graded`) makes each multivector carry its full type layout. This cuts the number of (type, keys) combinations at the cost of some sparsity.

### 2.5 Inverse, sqrt, exp (`operators.py:249-470`)

- **`inv(y)`.**
  - If `y~y` is a scalar: `~y/(y~y)`.
  - Else if `d < 6`: **Hitzer–Sangwine**, with `x̄` = Clifford conjugate:
    - d=1: `num = x̂`
    - d=2: `num = x̄`
    - d=3: `num = x̄ · rev(x x̄)`
    - d=4: `num = x̄ · (x x̄ − 2⟨x x̄⟩_{3,4})`
    - d=5: `c = x̄ · rev(x x̄)`, `num = c · (x c − 2⟨x c⟩_{1,4})`

    In every case `denom = ⟨x·num⟩₀` (`hitzer_inv`, `:271-305`).
  - Else **Shirokov** (`:308-339`): with `N = 2^⌊(d+1)/2⌋`, run the Faddeev–LeVerrier-style iteration `x_i = xⁱ − Σ c_j x^{i-j-1}`, `c_i = (N/i)⟨x_i⟩₀`. It stops early when `x_i` becomes scalar. Powers come from `power_supply`, which uses minimal addition chains (`powers.py`).
  - Returned as `Fraction(num, denom)`, so `div` stays one division.
- **`sqrt`** (`:450`) uses the Study-number formula (`cp = √((a + √(a² − (bI)²))/2)`, `√x = cp + bI/(2cp)`), citing De Keninck & Roelfs, "Normalization, square roots, and the exponential and logarithmic maps in geometric algebras of less than 6D", MMAS 2022, doi:10.1002/mma.8639.
- **`exp`** (`multivector.py:728`) handles only *simple* bivectors: `cosh(l) + x·sinhc(l)` with `l = √(−⟨x²⟩₀)`, with sympy or numpy branches. kingdon has **no general log** and no invariant decomposition. There are also `outerexp`, `outersin`, `outercos` and `outertan`.
- `expr_as_matrix` (`matrixreps.py:98`) extracts the matrix of a linear-in-last-argument GA expression, such as `R >> x`. This is the same idea as an extensor's table.

### 2.6 What to take from kingdon

- The **binary-key filter formulation** of all products: a two-line predicate per product, which is ideal for building integer extensor tables.
- The **cache key `(type, present keys)`**, and structural-zero dropping to get output sparsity.
- **Types defined by a symbolic expression**, whose numeric outputs become fixed slots. This is elegant for "motor = product of two reflections" or "point = hodge(e0 + x)".
- The **most-specific-type resolution cost**.
- The **dual choice by metric**: polarity for r=0, Hodge for r=1, error otherwise.
- The **Hitzer–Sangwine formulas for d ≤ 5 and Shirokov beyond**, returned as a fraction, with power chains.

---

## 3. ganja.js (`ganja.js`, `codegen/`)

### 3.1 Summary

ganja.js is a single-file generator (`ganja.js`, 1912 lines) for arbitrary Cl(p,q,r), including custom Cayley tables, dual numbers and subalgebras. It has two storage and codegen modes:
- **Flat** (the default when n ≤ 6): products are unrolled into full-multivector JS via `new Function`, with no sparsity and no symbolic simplification.
- **Graded** (n > 6, or `graded:true`): a multivector is an array of per-grade arrays, and products are interpreted loops using bit tricks.

`codegen/generate.js` turns the flat JS into C++, C#, Python and Rust files through templates. These are the files distributed on bivector.net.

### 3.2 Basis generation and ordering (`ganja.js:109-157`)

- **Generator order: null first (r), then p positive, then q negative.** See the checks in `simplify`: `(s[i]-low) < r` → 0, `≥ p+r` → −1, otherwise +1 (`:145`). When r ≠ 0, indices start at `e0`. For example, `Algebra(3,0,1)` has `e0` null and `e1..e3` positive, while `Algebra(4,1)` has `e1..e4 = +1` and `e5 = −1`.
- **Default basis.** Take the binary indices, map their set bits to digits, sort by grade then lexically ("not cyclic"). For n > 9, digits are two characters wide (`:130-133`).
- **Custom names.** `simplify(s)` bubble-sorts the concatenated index string, flipping the sign per swap and contracting equal pairs with the metric. The `brm` remap table then maps the canonical result (e.g. `e02`) back to the user name with a sign (`-e20`) (`:140-151`). Any orientation, such as `e31`, `e021` or `e20`, therefore works.
- **Bitwise sign** (`simplify_bits`, `:153-155`), used in graded mode. Also a good reference for Rust:

```
simplify_bits(A,B): n=p+q+r; ab=A&B; res=A^B
  if ab & ((1<<r)-1): return [0,0]           # shared null generator
  t=0; while n--: t ^= (A >>= 1)              # bit k of t = parity(A bits above k)
  t &= B                                      # transpositions
  t ^= ab >> (p+r)                            # each shared negative generator flips sign
  t ^= t>>16; t ^= t>>8; t ^= t>>4            # fold parity into low 4 bits
  sign = 1 - 2*((0x6996 >> (t&15)) & 1)       # 27030 == 0x6996 = 4-bit parity LUT
```

- **Cayley and products** (`:167-196`). `mulTable[i][j]` holds strings such as `'-e013'`, from `options.Cayley` or from `simplify`. For each output blade, the source pairs are grouped (`gpo`), and string tables are built per output row `oi` and input column `xi`:
  - `gp[oi][xi] = Σ ±'b[yi]*this[xi]'`;
  - `op`: grade sum;
  - `cp`, which is the *left contraction* LDot: `g_o = g_y − g_x`;
  - `cps`, the symmetric Dot: `g_o = |g_y − g_x|`.

  These are joined into `Mul`, `Wedge`, `LDot`, `Dot` functions via `new Function`. `describe()` prints the basis, metric and Cayley table (`:751`).
- **Non-diagonal metrics and CGA.** ganja does **not** do a basis change. CGA is diagonal (`Algebra(4,1)` or `(3,1)`), and the null vectors are defined in user code. The examples use `ni = 1e4+1e5, no = .5e5-.5e4`, which gives `ni·no = −1`, and `ni = e3+e4, no = ½(e4−e3)` in 2D CGA (`examples/*cga*`). Arbitrary non-diagonal algebras are possible only by passing a full string `Cayley` table (`options.Cayley`, `:102, 159, 167-170`). The WebGL renderer hard-codes `ni = e4+e5, no = ½(e4−e5)` (`:1340`). That has the opposite sign of `no` from the examples, so treat it as internal.
- **Other options:** `dual`, which builds a multi-dual-number algebra with a custom Cayley table (`:117-120`); `even` or a partial `basis` for subalgebras (`:173-178`); `mix`, which accesses components by name (`.e12`), so elements of different subalgebras interoperate.

### 3.3 Duality in ganja (`:161-164, 223-224, 251`)

- **Degenerate (r > 0), flat mode.** `drm` is the complement index map: sort the basis by grade then by the numeric value of the sorted digits, and reverse. `drms[i] = sign(basis[drm[i]] · basis[i])`, so `Dual(x)[i] = x[drm[i]]·drms[i]` and **`x ∧ Dual(x) = +I`**. `UnDual` uses the mirrored sign, `drms[n−1−i]`.
- **Non-degenerate (r = 0).** `Dual = x·I` (multiplication *by* I, not I⁻¹), and `UnDual = x/I`.

Measured in the bivector.net basis for PGA3D:

| x | Dual(x) | UnDual(x) |
|---|---|---|
| e0 | e123 | −e123 |
| e1 | e032 | −e032 |
| e2 | e013 | |
| e3 | e021 | |
| e01 | e23 | e23 |
| e02 | e31 | |
| e03 | e12 | |
| e021 | −e3 | e3 |
| e013 | −e2 | |
| e032 | −e1 | |
| e123 | −e0 | e0 |
| 1 ↔ e0123 | | |

In the PGA2D basis, both are sign-free: `e0↔e12, e1↔e20, e2↔e01`.

**Regressive product.** Flat mode, "Conforms to the new Chapter 11 now", `:251`. I verified numerically (`scratchpad/ganja_test.js`) that **`a & b = UnDual(Dual(b) ∧ Dual(a))`**, with the operands *swapped*. Compared with the J-map join `J(Ja ∧ Jb)`, ganja differs by −1 for grade pairs (2,3), (3,2) and (3,3), and agrees elsewhere. So in ganja and the bivector.net code, **`e123 ∨ e032 = −e23`**, while GAmphetamine and kingdon give `+e23`. Graded mode uses `Vee = Dual(Dual a ∧ Dual b)` (`:420`).

### 3.4 Codegen and flat vs graded

- **Flat.** Every product is a fully unrolled `2^n × 2^n` string, e.g. `res[0]=b[0]*this[0]+…`. There are no zero checks and no types. Performance comes only from the JIT.
- **Graded.** `generator extends Array`, where `this[g]` is the coefficient array of grade g, or undefined. `Mul`, `Wedge`, `LDot` and `Dot` loop over present grades and nonzero coefficients using `simplify_bits`, `bc` (popcount) and `bits_basis` (`:258-430`). Coefficients may be *strings*, in which case the loops build expression strings. This is how `OPNS_GLSL` and `IPNS_GLSL` emit GLSL for implicit-surface ray marching (`:356-400`).
- **`codegen/generate.js`** is used for the bivector.net downloads. It instantiates `Algebra({p,q,r,basis,graded:false})`, grabs `prototype.Mul`, `Wedge`, `Vee`, `Dot`, `Add` and the rest as source text, rewrites `this`→`a`, and pastes the result into templates (`cpp.template.js`, `rs.template.js`, …).
  - **For r == 1 the emitted `Dual` is the sign-free reversal `res[i] = a[2^n−1−i]`**: `[...x.reverse()]`, with minus signs stripped.
  - The emitted `Vee` is ganja's signed Vee above. `Dot` is the symmetric inner product that includes scalars.
  - Pregenerated outputs are in `codegen/{cpp,csharp,python}/pga3d.*`. `pga3d.cpp:9` holds the basis. Its helpers are `point(x,y,z) = e123 + x e032 + y e013 + z e021` and `plane(a,b,c,d) = a e1 + b e2 + c e3 + d e0` (`pga3d.cpp:439-448`), along with `rotor(angle,line) = cos(a/2) + sin(a/2)·line.normalized()` and `translator(d,line) = 1 + d/2·line`.

### 3.5 bivector.net conventions (to match)

Sources: `codegen/generate.js:36`, `codegen/cpp/pga3d.cpp`, and the cheat sheets https://bivector.net/2DPGA.pdf and https://bivector.net/3DPGA.pdf (SIGGRAPH 2019 course notes; text extracted in `scratchpad/pdf/`).

**PGA3D, R*₃,₀,₁**

| index | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| blade | 1 | e0 | e1 | e2 | e3 | e01 | e02 | e03 | e12 | e31 | e23 | e021 | e013 | e032 | e123 | e0123 |
| square | +1 | 0 | +1 | +1 | +1 | 0 | 0 | 0 | −1 | −1 | −1 | 0 | 0 | 0 | −1 | 0 |

- **Plane** `ax+by+cz+d=0` is `a e1 + b e2 + c e3 + d e0`.
- **Point** `(x,y,z)` is `x e032 + y e013 + z e021 + e123`. A direction (ideal point) is the same without `e123`.
- **Line** `ℓ = … + d e12 + e e31 + f e23`, with Euclidean norm `‖ℓ‖ = √(d²+e²+f²)`.
- **Dual (Poincaré / J)** is coefficient reversal with no signs: `1↔e0123, e0↔e123, e1↔e032, e2↔e013, e3↔e021, e01↔e23, e02↔e31, e03↔e12`. This basis is chosen so that J is sign-free and involutive.
- **Join** is `P₁ ∨ P₂`, and meet is `p₁ ∧ p₂`. In the generated code, `a & b` equals ganja's `UnDual(Dual b ∧ Dual a)` (§3.3).
- **Reverse** negates grades 2 and 3.

**PGA2D, R*₂,₀,₁**

| index | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| blade | 1 | e0 | e1 | e2 | e01 | e20 | e12 | e012 |
| square | +1 | 0 | +1 | +1 | 0 | 0 | −1 | 0 |

- **Point** `(x,y)` is `x e20 + y e01 + e12`, and a direction is `x e20 + y e01`.
- **Line** `ax+by+c=0` is `a e1 + b e2 + c e0`.
- **Dual** is reversal: `a + b e0 + c e1 + d e2 + e e01 + f e20 + g e12 + h e012 ↦ h + g e0 + f e1 + e e2 + d e01 + c e20 + b e12 + a e012`.
- Inside ganja itself, `examples/example_projective_2d.html` uses the order `['1','e0','e1','e2','e12','e20','e01','e012']`. That is the same blades in a different order. The bivector.net cheat sheet and codegen order is the one above.

**The ganja default for `Algebra(3,0,1)`** is lexical: `1,e0,e1,e2,e3,e01,e02,e03,e12,e13,e23,e012,e013,e023,e123,e0123`, with signed `Dual` (`e1→−e023`, `e3→−e012`, …).

**Recommendation.** Store blades internally by bitmask, with a per-algebra *presentation basis* (names, order and orientation signs), like ganja's `brm` or GAmphetamine's `basisPermutation`. Ship bivector.net presets whose Dual is the sign-free J. Pick one join sign convention explicitly and document it. I suggest `a ∨ b = J(Ja ∧ Jb)`, which gives `e123 ∨ e032 = +e23` and agrees with GAmphetamine and kingdon. Add a compatibility test against the ganja/bivector.net output if byte-level parity is required.

---

## 4. Rust design notes derived from the three projects

1. **Exact tables.**
   - Represent blades as `u32` bitmasks, with generator metric `m_i ∈ {−1,0,+1}` ordered null, then +, then − by default. The ordering is configurable.
   - Compute each basis product sign with ganja's `simplify_bits` (parity LUT) or the kingdon filter predicates. A basis-permutation sign vector maps storage blades (`e31`, `e021`, …) to canonical bitmasks.
   - An extensor's integer table is `T[out][i][j] ∈ ℤ` with sparse storage, and products are filters on `(kx, ky, kx^ky)`.
2. **Symbolic engine.**
   - Variables are interned `u32`s. A monomial is a sorted `SmallVec<[u32; 4]>`, or an exponent vector for small arities. A polynomial is a sorted `Vec<(Rational or i64, Monomial)>`; GAmphetamine's representation already works.
   - Rationals are a numerator plus a *factored* denominator `Vec<(PolyId, u32)>`, with sqrt atoms as special variables that carry their radicand. This reproduces GAmphetamine's normalize/sqrt cancellations.
3. **Type conditions.** Build the ideal from `condition(symbolic type instance)`, compute a reduced Gröbner basis (Buchberger is fine at these sizes), and apply `normal_form` to every coefficient. The same machinery handles:
   - output-type refinement (the condition reduces to 0);
   - frozen normalized intermediates (fresh variables plus their condition);
   - fixed coefficients (substituting constants).
4. **CSE.** Model the output as a DAG of `Sum`/`Prod` nodes with a cost `mul = 1, add = 1, div = 4` (GAmphetamine uses a 1.0001 mul tiebreak). Useful passes, all exact:
   - shared residual sums across components (GAmphetamine phase 1 or kingdon), which gives the classic `t0=…` intermediates of sandwiches;
   - square-sum completion with verification;
   - Horner isolation;
   - pairwise product extraction by frequency;
   - hoisting of shared denominators as reciprocals, with power reuse.

   Run at least two strategies and keep the cheaper one, like the GAmphetamine portfolio. Verify with random-point evaluation in debug builds, like `expectEquivalentCSE`.
5. **Laziness in Rust** means generating only the declared `(TypeA, TypeB, Op)` impls (proc-macro or `build.rs`), with output types inferred as in kingdon's `resolve_layout` or GAmphetamine's `type()`. If the combinatorics explode, a kingdon-style `full_layout` switch bounds the number of combinations.
6. **Inverse, sqrt, exp, log:**
   - Inverse: blade/versor shortcut, then Study-number shortcut, then Hitzer–Sangwine (d ≤ 5), then Shirokov (any d).
   - Sqrt: the Study-number formula, and `normalized(1+x)` for motors.
   - Exp and log: invariant decomposition. ganja and GAmphetamine have closed forms for PGA below 6D; kingdon only handles simple bivectors.
