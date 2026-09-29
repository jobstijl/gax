# Numerics

[laws.md](laws.md) proves that every tier computes the same *polynomial*. This page is about
what happens in floating point, where the tiers may differ: what gax guarantees, how it is
measured, and what it does not fix. Each claim links to the test that checks it.

## The guarantee per tier

The tiers are not bit-identical, and this is not a goal. They differ in expression shape
(common subexpressions, kernel extraction, balanced trees), in reduction modulo an ideal, and in
where `mul_add` is used. Instead, each of them has a computed error bound.

| tier | what it is | guarantee |
|---|---|---|
| 1 | the products as the tables give them (ADR-006) | each output is a sum of products; forward error `≤ γₙ Σ|terms|` (standard running-error analysis) |
| 2 | fused sandwiches, their matrices, prepared maps (ADR-011) | the same polynomial as tier 1 (proved); error within the bound of its own straight-line program |
| 2, `Unit` | kernels reduced modulo `u ~u = 1`, made homogeneous again | equal to the plain sandwich for a unit `u`; for a drifted `u`, equal up to a uniform scale (see "Drift") |
| 3 | user kernels traced in `build.rs` (ADR-010) | as tier 2, for the program the tracer emitted |

**The bound.** Every emitted kernel is a straight-line program whose DAG the generator has.
`Program::error_bound(mag, u)` propagates a first-order bound through it:

* a sum adds its operands' bounds plus `u` times the magnitude of the result;
* a product adds each operand's bound times the other's magnitude, plus `u` times the result;
* a `mul_add` rounds once;
* a non-dyadic constant such as `1/3` adds its own rounding.

The magnitudes are those of the inputs, so the bound holds wherever the polynomial cancels.

**The test** (`gax-gen/tests/error_bounds.rs`) evaluates, in `f32`, the tier-1 program and the
fused tier-2 program of every plain sandwich kernel of PGA2D, PGA3D, VGA3D, STA and CGA3D on
random inputs. It requires their difference to be within the sum of the two bounds, not within
an ad hoc tolerance. The largest difference observed is 0.49 of the bound. A second test checks
the bound on a known cancellation, `x² − y²` at `x ≈ y`.

**The bound as a tiebreak.** When the simplifier chooses between equivalent programs
(`cse::compile_best`), it takes the cheapest by weighted operation count, unless a candidate
within one operation of it has less than half its error score. No standard kernel changed cost
because of this rule. Square-sum completion is the interesting case:

* `1 − 2(c² + d²)` is more accurate than `a² + b² − c² − d²` near the identity;
* but it is not homogeneous in the versor, which matters once the versor drifts (next section).

gax now prefers homogeneity for `Unit` kernels (ADR-020).

## Drift

`Unit * Unit` keeps the certificate without renormalizing, as nalgebra and glam do. So a long
chain of products drifts to `u ~u = (1 + δ)²`.

**The problem.** The `Unit` kernels had been simplified modulo `u ~u = 1`, and so were not
homogeneous in `u`. Drift then showed up as *distortion* rather than scale. Measured in PGA3D
(f64), by scaling a unit motor by `1 + δ` and moving four points:

| drift δ | `Unit` kernel before | `Unit` kernel now | plain kernel |
|---|---|---|---|
| 1e-3 | 2.6e-3 | < 1e-12 | < 1e-12 |

The numbers are the worst relative change of a pairwise distance. Before the fix the distortion
was proportional to δ; the test now asserts `< 10⁻¹²` for δ = `10⁻³`, `10⁻⁶` and `10⁻⁹`.

**The fix.** Every `Unit` kernel is homogeneous of degree 2 in `u` again. The generator
multiplies by `‖u‖²` where the reduction had substituted 1, and asserts both properties of the
result: it equals the sandwich modulo the unit condition, and it is homogeneous. Drift is now a
uniform scale:

* In projective algebras the scale cancels. Points, lines and planes keep their shapes and
  incidences (`tests/numerics_drift.rs`).
* In VGA it is one length factor for every vector, so angles are kept.

The kernel costs are in [performance.md](performance.md). `Unit<Motor> >> Point` went from 25 to
33 multiplications; the plain kernel has 38.

**Renormalization.**

* **`Unit::renormalize_fast()`** is one Newton step, `u (3 − u ~u) / 2`, with no square root.
  * The identity `r ~r = n (3 − n)² / 4`, with `n = u ~u`, is proved on symbolic coefficients,
    so an error `e` in `n` becomes `¾ e²`. It holds for Study numbers too (the pseudoscalar part
    of a PGA motor's norm).
  * `tests/numerics_renormalize.rs` checks it and measures quadratic convergence.
* **`Unit::mul_renormalized`** multiplies and renormalizes in one call. Over 100 000 products it
  keeps `u ~u − 1` below `10⁻¹³`.
* **`normalized()`** is the exact path, with a square root.
* **Policy:** renormalize after each integration step, or after every few products.
* **`check-units`.** This opt-in feature asserts, in every certified kernel that consumes a
  `Unit`, that `u ~u` is 1 within `√ε`, Study part included. Off, it costs nothing. CI runs the
  test suite with it, and it caught a hand-written "unit" motor with a nonzero Study part in an
  old test.

## Determinism (ADR-027)

By default, `mul_add` is a fused multiply-add where the build has FMA:

* scalar code when compiled with `target_feature = "fma"`;
* batch lanes on the detected level (AVX2, AVX-512, NEON).

So the default build and a `target-cpu=native` build differ in the last bits, and so do the
scalar and batch paths.

With the **`deterministic`** feature:

* `mul_add` is `a * b + c`, never fused, in scalar code, `wide` lanes and batch lanes;
* the elementary functions are pure Rust everywhere: `gax::math` (the batch polynomials) for
  `f32`, and the `libm` crate for `f64`;
* the summation trees are the generated ones in every path.

`tests/determinism.rs` runs exp, sandwich and log through the `Map` pipeline, `transform_each`
and `transform_slice`. The results must be bit-equal to the scalar path on every available
level (portable, SSE2, SSE4.2, AVX2 here), and CI runs it. Use the feature for lockstep
networking and replays. The cost is the lost FMA: the fused sandwich took 6.3 ns without FMA
and 5.3 ns with it ([performance.md](performance.md)).

## Transcendental functions

`exp`, `log`, `normalized` and `sqrt` go through Study numbers (ADR-019). The edge cases are
property tests in `tests/numerics_edges.rs`, sampled log-uniformly near each edge rather than
uniformly.

**Small angles and the series boundaries.** The rotation fast path (PGA) switches to a series
near `a = 0`. The tests check `log(exp B) = B` to 16 ε of `|B|` for half-angles from `10⁻¹²` to
1, with translations up to `10³`. They also check `exp(B) = exp(B/2)²` straddling each boundary,
which compares the two branches. This found three defects, now fixed:

* **`log`'s series was wrong at second order.** It used `(1 + s²/6)/c` for `θ/s`, where the
  correct series is `(1/c)(1 − t²/3 + …)` with `t = s/c`. Below the threshold (`s < 10⁻³`),
  `log(exp B)` was off by up to `3·10⁻⁷`.
* **The `I` parts cancel.** Those are the coupling of rotation and translation: in `exp`,
  `(S − C)/(2a²)`; in `log`, `(θ − cs/n)/(2s³)`; and `atanh(t)/t` in the general `log`. Their
  direct forms lose `ε/a²` near 0, so a series threshold of `10⁻⁴` left about `10⁻¹²` relative
  error. The series now run to about 10 terms and cover `a² < 1/4`, `t² < 1/25` and `|t²| < 1/100`
  respectively, and both branches are within a few ulps at the boundary.
* **Near a full turn, the series silently changed branch** (see below).

**The complex square root cancelled** (found porting the 5D functions to WGSL, 2026-09-29). The
general Study functions evaluate `f(a ± √q)` and take the `I` part from the difference, which
lives in the small imaginary part of `√(a + iw)` when `|w| ≪ |a|`. `Cx::sqrt` computed that part
as `√((|z| − a)/2)`, which cancels there: in `f32` a CSTA twist's `exp` was off by 0.2% (and
`f64` lost about five digits). It now computes the larger part `t = √((|z| + |a|)/2)` and the
other as `b/(2t)`, without cancellation, in Rust and in WGSL alike.

**Pure translations** (`B² = 0`, a degenerate Study number): `exp(B) = 1 + B` and
`log(1 + B) = B` exactly, for any size.

**The branch of `log`.**

* **The general rule.** The rotation part of `log R` has half-angle `θ ∈ [0, π]`, so
  `exp(log R) = R`.
  * A versor with a negative scalar part is a rotation by more than a half turn, and gets the
    long way round.
  * For the shortest motion, negate `R` first (`R` and `−R` are the same motion).
* **Exactly at `R = −T`** (a translation `T` times −1, including `R = −1`), no unique logarithm
  exists: any axis parallel to the translation works. `log` returns `log(−R)`, the same motion,
  and there `exp(log R) = −R`.
* **Near a full turn** (`θ = π − δ`) the axis is the direction of a part of size `sin δ`, and the
  pitch divides by it. So the error of `log R`, and of `exp(log R)`, grows like `ε/δ`. The test
  asserts exactly that bound.
* **At a half turn of the motion** (`θ = π/2`) nothing is special.
* **The general `log`** (STA, CGA3D) is stable up to, but not including, `θ = π`.

**`normalized`.**

* **Scale.** It is scale-invariant down to a norm of `10⁻¹⁵⁰` in f64, where `u ~u` underflows;
  at zero the result is not finite.
* **Large Study part.** The inverse square root of a Study number `s + pI` with `I² = 0` is
  exact in form (`s^(−1/2) − (p/2) s^(−3/2) I`). So normalizing `R (s + pI)` returns `R` with an
  error that grows only with `p/s`, the size of the input (tested up to `p/s = 10⁶`).

**The CSTA exponential.** No closed form covers the full conformal group of CSTA
(`Bivector` to `Even`), so `exp` is scaling and squaring in the product closure:

* It halves `B` `s` times, until `‖B/2^s‖₁ ≤ 1/16`. The count is chosen from the norm, per
  call.
* A Taylor series of degree 10 then gives `exp(B/2^s)` to below `10⁻²⁰`.
* A Newton step renormalizes it while it is near 1.
* `s` squarings follow.
* A second Newton step is applied where the result is small (`‖r‖₁ < 4`). Only there is it
  well-conditioned: for a large boost, `~r r − 1` cancels at the scale of `cosh²`.

The previous version used a fixed `B/256`, 8 terms and no renormalization.
`‖exp(B) exp(B)~ − 1‖∞`, relative to `‖exp B‖²` and worst of 8 directions per row:

| `B` | ‖B‖₁ | adaptive | fixed `1/256` |
|---|---|---|---|
| rotation | 0.001 | 1.1e-16 | 2.8e-14 |
| rotation | 1 | 3.5e-16 | 9.2e-14 |
| rotation | 64 | 2.2e-16 | 1.0e-10 |
| boost | 1 | 1.5e-16 | 5.3e-14 |
| boost | 64 | 2.6e-16 | 1.7e-16 |
| general | 1 | 2.2e-16 | 3.7e-14 |
| general | 4 | 7.7e-14 | 7.7e-14 |
| general | 64 | 1.6e-15 | 9.9e-16 |

On the Poincaré bivectors the result also agrees with the closed form of `Twist::exp`. For a
general bivector of moderate size, where neither Newton step applies at the end, the rounding
of the squarings remains, about `2^s ε`; the test asserts that bound.

## Solvers

The scalar LU pivots with branches (`Real::SCALAR`), and the lanes pivot with selects. Both use
the same rule (the first row with the strictly largest magnitude) and the same operation order.
So `det`, `inverse` and `solve` are **bit-identical per lane**, even on nearly singular maps
where the pivot choice is ill-determined (`gax-core/tests/solver_lanes.rs`, with condition
numbers up to `10¹⁴`, and exact ties). Their accuracy is a property of the one algorithm: the
forward error of `solve` is within `64 n ε κ∞(A)`.

The Jacobi methods (`eigh`, `svd`) iterate until every lane has converged, so a lane can get a
few extra, tiny rotations. So they are not bit-identical, but:

* eigenvalues and singular values agree within `16 n ε ‖A‖`;
* vectors agree within that divided by the gap to the nearest other value;
* order and signs are the same (they are not realigned in the test).

This holds on spectra with gaps down to `10⁻¹⁴`. Singular vectors of a singular value within
rounding of zero are arbitrary, and are not compared.

## Documented, not fixed

**`f32` with large coordinates.** A point's error grows with its distance from the origin,
about 2 ulp of that distance after a rigid motion:

| distance from the origin | 1 | 10² | 10⁴ | 10⁶ |
|---|---|---|---|---|
| absolute error of `m >> p` (f32) | 2.7e-7 | 2.0e-5 | 1.9e-3 | 0.22 |

This is not specific to gax: the coordinates themselves carry that error. Use `f64`, or rebase
the origin near the objects of interest (a floating origin).

**Geometric ill-conditioning.** Some constructions are inherently ill-conditioned, and PGA
degrades gracefully rather than failing:

* **The join of two nearly coincident points** is a line whose direction part has the size of
  their separation. Its direction is accurate to about `ε/separation` (measured: exact at
  `10⁻⁸`, `10⁻⁶` at `10⁻¹¹`). At zero separation it is the zero line.
* **The meet of two nearly parallel planes** is a line, and its meet with a third plane is a
  point whose weight is the sine of the angle. `to_euclidean` divides by it, so the position is
  accurate to about `ε/angle` (measured: `10⁻⁸` at `10⁻⁸`). Exactly parallel planes meet in an
  ideal line, and its points have weight 0: they are directions, and `to_euclidean` is not
  finite.

Check the weight (or the norm of the Euclidean part) before dividing, when inputs can be
degenerate.
