# The logarithm of a 6D even versor, in closed form

gax computes `log R` for CSTA's even versors (the conformal group of spacetime, `R(4,2)`) in
closed form. Below six dimensions a bivector splits into at most two commuting parts, and De
Keninck and Roelfs give robust closed forms for `exp` and `log` (*Normalization, square roots,
and the exponential and logarithmic maps in geometric algebras of less than 6D*, 2022). In any
dimension, Roelfs and De Keninck factor a rotor into commuting simple rotors through its
tangent decomposition (*Graded symmetry groups: plane and simple*, 2021, §8–9). The planes'
tangents follow from `R`'s grade parts and the roots of a polynomial, one root at a time, and
the logarithm is the sum of the factors' logarithms. In 6D a bivector has three parts, and
that polynomial is a cubic.

This note reaches the same cubic but never separates the planes. The logarithm is three
bivectors built from `R`'s grade parts, weighted by an interpolant of one function at the
cubic's roots (§2). That keeps it exact where roots coincide (one plane, translations,
isoclinic planes), which is where individual roots are ill-conditioned and factors are not
unique (§3). It is also branch free, so it runs per SIMD lane and in shaders, and it turns
planes near a half turn out of the way first (§5). The note derives the closed form, shows how
it is evaluated without its singularities, and records how it was validated. The code is `gax_core::study::log_coeffs_6d` and the generated
`Log<Bivector> for Unit<Even>` in `gax::csta` and in every 6D algebra declared with `algebra!`,
and `unit_even_log` in the WGSL modules.

## 1. Invariants of a versor: a cubic

A unit even versor is `R = exp(B)`, and `B = b₁ + b₂ + b₃` with commuting simple bivectors
(the invariant decomposition; Roelfs and De Keninck, *Graded symmetry groups: plane and
simple*, 2021). With `μⱼ² = bⱼ²` (complex in general: imaginary for rotations, real for boosts,
zero for translations) and `b̂ⱼ = bⱼ/μⱼ`, so `b̂ⱼ² = 1`:

```
R = ∏ⱼ (cⱼ + sⱼ b̂ⱼ),   cⱼ = cosh μⱼ,  sⱼ = sinh μⱼ.
```

The `b̂ⱼ` commute and their products are blades of higher grade, so the grade parts of `R` are

```
⟨R⟩₀ = c₁c₂c₃          ⟨R⟩₂ = Σ sⱼ cₖ cₗ b̂ⱼ
⟨R⟩₄ = Σ sᵢ sⱼ cₖ b̂ᵢb̂ⱼ   ⟨R⟩₆ = s₁s₂s₃ b̂₁b̂₂b̂₃
```

(`{i, j, k, l}` ranging over the planes; `k, l` are the others). Square each part and keep the
scalar: with `uⱼ = cⱼ² = cosh² μⱼ` and `sⱼ² = uⱼ − 1`, only symmetric functions of the `uⱼ`
remain. With `p₁, p₂, p₃` the elementary symmetric functions of `u₁, u₂, u₃`:

```
⟨R⟩₀²      = p₃
⟨R₂ R₂⟩₀   = 3p₃ − p₂
⟨R₄ R₄⟩₀   = 3p₃ − 2p₂ + p₁
```

so the invariants are **the roots of a cubic whose coefficients come from `R` alone**:

```
uⱼ = cosh² μⱼ  are the roots of  t³ − p₁t² + p₂t − p₃,
p₃ = ⟨R⟩₀²,   p₂ = 3p₃ − ⟨R₂²⟩₀,   p₁ = ⟨R₄²⟩₀ + 3p₃ − 2⟨R₂²⟩₀.
```

(`⟨R₆²⟩₀ = p₃ − p₂ + p₁ − 1` holds too; with the others it is `R ~R = 1`.)

## 2. Separating the planes

`B = Σ μⱼ b̂ⱼ`, and `⟨R⟩₂ = Σ wⱼ b̂ⱼ` with `wⱼ = sⱼcₖcₗ`. Products of grade parts give two more
bivectors with other weights on the same planes. Using `b̂ᵢb̂ᵢ = 1`, with `r₀ = ⟨R⟩₀`:

```
G₁ = ⟨R⟩₂               weight 1
G₂ = r₀ ⟨R₄ R₂⟩₂         weight uⱼ² − p₁uⱼ + 2p₃
G₃ = r₀ ⟨R₆ R₄⟩₂         weight uⱼ² + (1 − p₁)uⱼ + p₃
```

(each times `wⱼ b̂ⱼ`). Hence `Q₁ = G₃ − G₂ + p₃G₁` weighs plane `j` by `uⱼ`, and
`Q₂ = G₂ + p₁Q₁ − 2p₃G₁` by `uⱼ²`. Since `μⱼ/wⱼ = (μⱼ/sⱼ)(cⱼ/r₀)`,

```
B = r₀⁻¹ (α₂Q₂ + α₁Q₁ + α₀G₁),
```

where `α₂u² + α₁u + α₀` is **the quadratic interpolating** `φ(uⱼ) = cⱼ μⱼ/sⱼ` at the three
roots. As a function of `u`,

```
φ(u) = √u · asinh(√(u−1)) / √(u−1),
```

analytic at `u = 1` (where `φ = 1`) and singular only at `u = 0`, a half turn. The planes are
never separated explicitly: everything is products of `R`'s grade parts and three numbers.

## 3. Evaluating the interpolant without its singularities

Through the roots, the interpolant is a combination of divided differences, and those divide
by differences of roots. Roots coincide in the commonest cases: a rotation in one plane
(roots `cos²θ, 1, 1`), a translation (null planes: `1, 1, 1`), isoclinic planes, the identity.
The individual roots are also ill-conditioned there (a double root moves like the square root
of a perturbation). gax evaluates the interpolant in two regimes, chosen per lane:

* **Close roots: no roots at all.** The interpolant's coefficients are symmetric functions of
  the roots, so they follow from `p₁, p₂, p₃` directly. Expand `φ` in its Taylor series at the
  roots' mean `m = p₁/3`. In `t = u − m` the roots satisfy `t³ + e₂t − e₃ = 0` (`e₂, e₃` from
  the `p`s), and `tᵏ` reduced modulo that cubic is `aₖt² + bₖt + dₖ` by a three-term
  recurrence. The interpolant is `Σ φₖ (aₖt² + bₖt + dₖ)`. Coinciding roots need nothing
  special: the reduction is exact. This is used while the roots' spread (bounded from `e₂` and
  `e₃`) is under a quarter of `m`, the distance to `φ`'s singularity, so 26 terms reach `10⁻¹⁶`.
* **Spread roots.** One real root always exists. Take the most isolated root `r` (Cardano, or
  the trigonometric form for three real roots, then a Newton step), and the remaining pair by
  its sum `S = p₁ − r` and product `P = p₂ − rS`, never the individual pair. The pair's line is
  `φ`'s series at its midpoint reduced modulo `t² − d²` when the pair is close, and the chord
  through its two values otherwise (a conjugate pair's chord is real). "Close" needs a positive
  midpoint: a loxodromic pair can have invariants with a negative real part, and then it
  straddles the branch cut of `√u`, which the series cannot cross. The quadratic adds the
  isolated root: `L(u) + (u² − Su + P)(φ(r) − L(r))/(r² − Sr + P)`.

`φ`'s Taylor coefficients at any centre come without cancellation from `φ(u) = √u · F(u−1)`,
`F(x) = asinh(√x)/√x`. Near `x = 0` (`|x| < 1/4`), `F`'s Maclaurin series (90 terms) is
shifted to the centre by repeated Horner. Elsewhere it is composed from series that are
analytic there: `v = √(x₀ + t)`, `asinh(v)' = v'/√(m + t)`, `F = asinh(v)/v` (or `asin` for
`x₀ < 0`). Real arithmetic throughout, except `φ` at a far conjugate pair.

## 4. Conditioning and the branch

* **Near a half turn** (`r₀ → 0`) the formula divides by `r₀` and loses `ε/r₀`: `6·10⁻¹⁵`
  at `r₀ = 0.01`, `8·10⁻⁹` at `10⁻⁸`.
* **The branch.** The closed form takes `cⱼ = √uⱼ` with a non-negative real part: every
  invariant plane's rotation is below a half turn (for a loxodromic pair, the principal value
  of the pair). That is the geometric principal logarithm, and it is idempotent:
  `log(exp(log R)) = log R`. It is right wherever `r₀ > 0`: then an even number of the `cⱼ`
  are negative, and turning two planes by a half turn each leaves `R` unchanged. Where
  `r₀ < 0` an odd number are, and the formula returns a logarithm of `−R`.

So the formula alone serves `r₀ > 1/16`. Below, gax first turns the planes near a half turn
out of the way (§5).

## 5. Turning planes near a half turn

A rotation plane (`uⱼ < 1`, `cⱼ = cos θⱼ`, `sⱼ = sin θⱼ`) near a half turn has `cⱼ ≈ 0`.
Multiplying `R` by a quarter turn in that plane, `−b̂ⱼ = exp(−(π/2) b̂ⱼ)`, turns its factor
`cⱼ + sⱼ b̂ⱼ` into `sⱼ − cⱼ b̂ⱼ`: its angle drops by `π/2`, and `sⱼ` takes the place of `cⱼ` in
`r₀`. For a set `T` of rotation planes, with `E = ∏_T (−b̂ⱼ)`,

```
R' = R E,   ⟨R'⟩₀ = ∏_T sⱼ ∏_rest cⱼ,   log R = log R' + (π/2) Σ_T b̂ⱼ,
```

since the `b̂ⱼ` commute with `R`'s planes. The closed form then serves `R'`.

* **The sign comes for free.** Orient each `b̂ⱼ` so that its weight `wⱼ = sⱼ cₖ cₗ` in `⟨R⟩₂`
  is positive. Then `sign(sⱼ) = sign(cₖ cₗ)`, and `⟨R'⟩₀` is positive when `T` has one plane or
  three, and has the sign of `r₀` when it has two. So turning one plane repairs `r₀ < 0` too.
* **Which planes.** `|⟨R'⟩₀| = ∏_T √(1 − uⱼ) ∏_rest √uⱼ` follows from the roots, and so do
  the planes' weights `|wⱼ|` in `⟨R⟩₂` (below). A plane's direction is read from its weight,
  so turning it costs `ε/|wⱼ|`, and the closed form on `R'` costs `ε/⟨R'⟩₀`. Among the sets
  with a positive `⟨R'⟩₀`, none included, gax picks the one with the smallest
  `1/⟨R'⟩₀ + Σ_T 1/|wⱼ|` (`gax_core::study::log_turn_6d`). For example, with one plane at a
  half turn and another at `u = 0.26`, turning both gives the larger `⟨R'⟩₀`. But the second
  plane's weight contains the first one's cosine and is `3·10⁻⁴`, so only the first is turned.
* **The planes' sum** `Z = Σ_T b̂ⱼ` is again an interpolant applied to the bivectors of §2:
  `Z = α₂Q₂ + α₁Q₁ + α₀G₁` with `α` interpolating `h(u) = [u ∈ T]/|w(u)|`,
  `|wⱼ| = √((1 − uⱼ) ∏_{k≠j} uₖ)`, at the roots. Coinciding roots must be turned together (a
  cluster); on a cluster, `h` is the smooth `√(u/(1−u))/|r₀|`, taken through its series as in
  §3. The candidates are none, all three through the series at their mean, and unions of the
  isolated root, the pair and each of the pair. Sets that separate the isolated root from the
  pair are preferred where its gap is at least 1/64 (relative).
* **The product `E`** is a polynomial in `Z`: for `n` commuting orthogonal unit rotation
  bivectors, `∏(−b̂ⱼ)` is `−Z`, `1 + Z²/2` or `−(Z³ + 7Z)/6` (`turn_polynomial`).
* **What remains ill-conditioned is so in itself.** Where two or three planes are at a half
  turn together (`R = b̂₁b̂₂(…)`, or `R = ±I`), `R` fixes only their product, and the log is not
  unique. Where a plane is near a full turn (`R ≈ −1` in it), its direction is barely
  determined. Near those points any logarithm moves by `1/δ` per unit change of `R`.

**Why not inverse scaling and squaring.** gax used it as the fallback before, and it fails in
6D in two ways:

* **It can leave the group.** Its square root `normalize(1 + R)` takes the principal root in
  each of the even algebra's four eigen-channels. In `R(6,0)`, `Spin(6) = SU(4)` has channel
  phases `±θ₁ ± θ₂ ± θ₃`, with an even number of minus signs. When the angles add up past a
  half turn, one phase wraps around. The "root" is then unitary with determinant `−1`, outside
  the spin group, and the logarithm misses `R` by a central element, `±1` or the pseudoscalar
  `±I`. Below six dimensions this cannot happen.
* **It can find no root at all.** With boosts, `(1 + R)~(1 + R)` has negative channel values
  near a half turn (`2 − 2 cosh α` for a half turn times a boost), and the Newton iteration
  returns NaN.

At random CSTA bivector entries up to 1.2, one versor in five is below `r₀ = 1/16`. Of those,
inverse scaling and squaring got 7% wrong or NaN.

## 6. Validation and cost

Checked in `f64` against `exp` (itself scaling and squaring, independent of the log):

| case | `max |log(exp B) − B|` |
|---|---|
| random bivectors, entries up to 0.4 / 0.6 | `2·10⁻¹⁵` / `5·10⁻¹⁵` |
| one rotation up to half-angle 1.5 (a half turn is π/2) | `7·10⁻¹⁶` |
| boosts and dilations up to rapidity 3 | `4·10⁻¹⁵` |
| translation (null), rotation and translation, isoclinic, near the identity | `0` to `10⁻¹⁶` |
| towards a half turn, `π/2 − 10⁻⁸` (turned) | `10⁻¹⁶` |
| past a half turn (`r₀ < 0`), and towards one with a boost or dilation (turned) | `10⁻¹²` or better |

Random CSTA versors, 5000 per size, checked for `exp(log R) = R` (relative):

| bivector entries up to | below `r₀ = 1/16` | worst | not fixed points of `log ∘ exp` |
|---|---|---|---|
| 0.5 | 0 | `2·10⁻¹⁴` | 0 |
| 1.0 | 303 | `3·10⁻¹²` | 0 |
| 1.5 | 1991 | `10⁻⁷` | 0 |
| 2.0 | 2987 | `1.4·10⁻⁶` | 1 |
| 2.5 | 3418 | `4·10⁻⁶` | 2 |

No log is NaN, and none misses `R`. The least accurate ones combine boosts of `cosh² μ` in the
hundreds to thousands (`|R|` up to 200) with a plane near a full turn (`u ≈ 1`, `r₀ < 0`),
whose direction is barely determined (§5). The tests (`tests/csta_log.rs`) also cover
portable SIMD lanes (lanes mixing the closed form and turning, each equal to its scalar result)
and `f32` (within `2·10⁻⁴`).

The closed form takes 2.6 µs, and 6.6 µs near a half turn (turned: the closed form twice and
four products). Inverse scaling and squaring took 50.7 µs (f64, Ryzen 7 5800X,
`benches/compare.rs`; the timings of this section on a loaded machine).

## 7. Other 6D algebras

Nothing in the construction is specific to CSTA's signature. The generator emits it for the full
even kind of every 6D algebra, standard or declared with `algebra!`. `tests/log6d_algebras.rs`
declares three more and checks each against `exp`:
- Euclidean `R(6,0)` (rotations only);
- split `R(3,3)` (rotations, boosts and loxodromic planes);
- degenerate `R(5,0,1)` (5D PGA: rotations and translations, null planes).

For each:
- `log(exp B) = B` holds to `10⁻¹¹` for bivector entries up to 0.25.
- Each basis plane alone (two or three coinciding invariants) is exact to `10⁻¹³`, and to
  `10⁻¹²` towards and past a half turn.
- Random versors with entries up to 0.8, `r₀` of either sign, are logs of `R` and fixed points.

## 8. In WGSL

The WGSL modules (`gax::wgsl::CSTA`, and those of declared 6D algebras) have
`unit_even_log(x: Even) -> Bivector`:

* `unit_even_log_closed` is a recorded kernel like the others: `r₀`, the `p`s and the three
  bivectors as one verified straight-line program, the weights from `study_log6`, and their
  combination. `study_log6` ports `log_coeffs_6d` to `f32` with the same two regimes, but
  16 series terms (the regimes keep the ratio at most 1/4, so 16 reach `2·10⁻¹⁰`, below
  `f32`'s precision) and branches instead of selects.
* `unit_even_log_turning` is another: the same program, `study_log6_turn` (a port of
  `log_turn_6d`), and `Z`.
* `unit_even_log` takes the closed form above `⟨x⟩₀ = 1/16`. Below, it recovers the number of
  turned planes as `n = −⟨Z²⟩₀`, forms `E` from `Z`, and returns
  `unit_even_log_closed(x E) + (π/2) Z`.

In the `f16` module the helpers and the programs feeding them compute in `f32`. Checked with
`wesl`'s CPU evaluator: both kernels agree with their `f64` evaluations. `unit_even_log`
agrees with the Rust `log` within `6·10⁻⁵` relative, on unit versors of which a quarter are near
a half turn and a quarter below `⟨x⟩₀ = 1/16`. On a GPU (`gax-gpu-tests`) it agrees within
`3·10⁻⁶`.

## 9. Higher dimensions

The construction carries over to `n` dimensions, where a bivector has `k = ⌊n/2⌋` commuting
planes.

* **The invariants.** `uⱼ = cosh² μⱼ` are the roots of a polynomial of degree `k`, whose
  coefficients are again the scalar parts of `R`'s grade parts squared. There is one per grade
  `0, 2, …`, and `R ~R = 1` makes the last one redundant.
* **The separation.** It takes `k` bivectors from products of grade parts, with weights
  `1, u, …, u^(k−1)` after recombination (`⟨R₈R₆⟩₂` and so on), and the interpolant has degree
  `k − 1`.
* **Close roots need no roots.** The series of `φ` at the mean, reduced modulo the polynomial,
  works in any degree. So one plane, translations and isoclinic planes stay exact.
* **Spread roots need the individual roots.** 7D still has three planes, so the cubic carries
  over unchanged, as long as the grade-part identities hold there (not tested; the generator
  emits the closed form only in 6D). 8D and 9D give a quartic, which still has roots in
  closed form. From 10D on (five planes), Abel–Ruffini rules out a formula in radicals, and the
  spread regime would find roots numerically: closed form except for a polynomial root.
* **Turning** carries over as it is: `⟨R'⟩₀` is positive for an odd number of turned planes,
  and `E` is a polynomial in `Z` of degree up to `k`.

The costs grow quickly: the even kind has `2ⁿ⁻¹` coefficients (128 in 8D), and the bivectors
of the separation are polynomials of degree up to `k + 1` in them.
