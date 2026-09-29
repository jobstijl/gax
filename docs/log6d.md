# The logarithm of a 6D even versor, in closed form

gax computes `log R` for CSTA's even versors (the conformal group of spacetime, `R(4,2)`) in
closed form. Published closed forms for `exp` and `log` stop below six dimensions, where a
bivector splits into at most two commuting parts ("less than 6D": De Keninck and Roelfs,
*Normalization, square roots, and the exponential and logarithmic maps in geometric algebras
of less than 6D*, 2022). In 6D a bivector has three parts, and a cubic enters. This note
derives the closed form, shows how it is evaluated without its singularities, and records how
it was validated. The code is `gax_core::study::log_coeffs_6d` and the generated
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
  through its two values otherwise (a conjugate pair's chord is real). The quadratic adds the
  isolated root: `L(u) + (u² − Su + P)(φ(r) − L(r))/(r² − Sr + P)`.

`φ`'s Taylor coefficients at any centre come without cancellation from `φ(u) = √u · F(u−1)`,
`F(x) = asinh(√x)/√x`. Near `x = 0` (`|x| < 1/4`), `F`'s Maclaurin series (90 terms) is
shifted to the centre by repeated Horner. Elsewhere it is composed from series that are
analytic there: `v = √(x₀ + t)`, `asinh(v)' = v'/√(m + t)`, `F = asinh(v)/v` (or `asin` for
`x₀ < 0`). Real arithmetic throughout, except `φ` at a far conjugate pair.

## 4. Conditioning and the branch

* **Near a half turn** (`r₀ → 0`) the formula divides by `r₀` and loses `ε/r₀`: `6·10⁻¹⁵`
  at `r₀ = 0.01`, `8·10⁻⁹` at `10⁻⁸`. Below `r₀ = 1/16` gax uses inverse scaling and squaring
  instead (`Even::log_by_scaling`, per lane), which stays at machine precision there, because
  its first square roots move away from the half turn.
* **The branch.** The closed form takes `cⱼ = √uⱼ` with a non-negative real part: every
  invariant plane's rotation is below a half turn (for a loxodromic pair, the principal value
  of the pair). That is the geometric principal logarithm, and it is idempotent:
  `log(exp(log R)) = log R`. It is right wherever `⟨R⟩₀ > 0`: then an even number of the
  `cⱼ` are negative, and turning two planes by a half turn each leaves `R` unchanged.
* **The fallback's limit.** Inverse scaling and squaring takes the square root
  `normalize(1 + R)`, which is the principal root in each of the even algebra's four
  eigen-channels (in `R(6,0)`, `Spin(6) = SU(4)`, whose channels have the phases
  `±θ₁ ± θ₂ ± θ₃` with an even number of minus signs). When the rotation angles add up past a
  half turn, one channel's phase wraps around. The "root" is then unitary but leaves the spin
  group (its determinant is `−1`), and the logarithm misses `R` by a central element, `±1` or
  the pseudoscalar `±I`. Below six dimensions this cannot happen; in 6D it does, for
  `⟨R⟩₀ < 0` in `R(6,0)` for example (§6). The closed form is used wherever `⟨R⟩₀ > 1/16`, so
  the limit concerns only the fallback's domain, and there only versors with large rotations in
  several planes. A fix needs the planes of the result, to add the missing quarter or half
  turns (§8).

## 5. Validation and cost

Checked in `f64` against `exp` (itself scaling and squaring, independent of the log):

| case | `max |log(exp B) − B|` |
|---|---|
| random bivectors, entries up to 0.4 / 0.6 | `2·10⁻¹⁵` / `5·10⁻¹⁵` |
| one rotation up to half-angle 1.5 (a half turn is π/2) | `7·10⁻¹⁶` |
| boosts and dilations up to rapidity 3 | `4·10⁻¹⁵` |
| translation (null), rotation and translation, isoclinic, near the identity | `0` to `10⁻¹⁶` |
| towards a half turn, `π/2 − 10⁻⁸` (fallback) | `7·10⁻¹⁶` |

At random entries up to 1, where some planes pass a half turn, every log with `⟨R⟩₀ > 0.1` is a
log of `R` and a fixed point of `log ∘ exp` (`tests/csta_log.rs`). This holds on portable SIMD
lanes (lanes mixing the closed form and the fallback) and in `f32` (within `2·10⁻⁴`).

The closed form takes 2.9 µs against 50.7 µs for inverse scaling and squaring (f64, Ryzen 7
5800X, `benches/compare.rs`).

## 6. Other 6D algebras

Nothing in the construction is specific to CSTA's signature. The generator emits it for the full
even kind of every 6D algebra, standard or declared with `algebra!`. `tests/log6d_algebras.rs`
declares three more and checks each against `exp` and against inverse scaling and squaring:
Euclidean `R(6,0)` (rotations only), split `R(3,3)` (rotations, boosts and loxodromic planes)
and degenerate `R(5,0,1)` (5D PGA: rotations and translations, null planes). `log(exp B) = B`
holds to `10⁻¹¹` for bivector entries up to 0.25, each basis plane alone (two or three
coinciding invariants) to `10⁻¹³`, and larger versors with `⟨R⟩₀ > 0.1` are fixed points. In
`R(6,0)`, versors with `⟨R⟩₀ < 0` (random entries up to 0.8) show the fallback's limit of §4:
the closed form is not used there, and inverse scaling and squaring misses by a central element.

## 7. In WGSL

The WGSL modules (`gax::wgsl::CSTA`, and those of declared 6D algebras) have
`unit_even_log(x: Even) -> Bivector`:

* `unit_even_log_closed` is a recorded kernel like the others: `r₀`, the `p`s and the three
  bivectors as one verified straight-line program, the weights from `study_log6`, and their
  combination. `study_log6` ports `log_coeffs_6d` to `f32` with the same two regimes, but
  16 series terms (the regimes keep the ratio at most 1/4, so 16 reach `2·10⁻¹⁰`, below
  `f32`'s precision) and branches instead of selects.
* Below `⟨x⟩₀ = 1/16`, `unit_even_log` calls `even_log_by_scaling`, a WGSL port of the fallback
  (loops, like the scaling-and-squaring `exp`).

In the `f16` module the helpers and the programs feeding them compute in `f32`. Checked with
`wesl`'s CPU evaluator: `unit_even_log_closed` agrees with its `f64` evaluation, and
`unit_even_log` agrees with the Rust `log` on unit versors, a quarter of them near a half turn,
within `4·10⁻⁵` relative; on a GPU (`gax-gpu-tests`), `unit_even_log` agrees within `3·10⁻⁶`.

## 8. Higher dimensions

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

The costs grow quickly: the even kind has `2ⁿ⁻¹` coefficients (128 in 8D), and the bivectors
of the separation are polynomials of degree up to `k + 1` in them. The fallback's limit (§4)
grows with the dimension too: more channels, whose phases can wrap.

## 9. Open

* **A fallback that stays in the spin group.** The limit of §4: the fallback needs the planes of
  its result, to add the quarter or half turns that the central element takes away. The plane
  projectors are Lagrange polynomials in `Q₁, Q₂` through the roots. Alternatively, the closed
  form could extend to `⟨R⟩₀ < 0`, by interpolating `−√u (π − acos √u)/√(1−u)` at an
  isolated root instead of `φ`. Only near `⟨R⟩₀ = 0` would a fallback remain.
