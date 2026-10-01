# The logarithm and exponential of an even versor in 6 to 9 dimensions, in closed form

A unit even versor `R` in a geometric algebra of dimension 6 to 9 is the exponential of a
bivector, `R = exp(B)`, and `B` splits into three (6D, 7D) or four (8D, 9D) commuting simple
parts. This note computes `log R = B` in closed form, for any signature, degenerate metrics
included. The method never separates the invariant planes. It is exact where planes coincide, it
has no branches (so it runs per SIMD lane and on a GPU), and it handles rotations near and past
a half turn. It is implemented in the Rust library [gax](https://github.com/jobstijl/gax), for
CSTA (the conformal model of spacetime, `R(4,2)`) and for every 6D to 9D algebra a user declares.

**Prior art.** Below six dimensions a bivector splits into at most two commuting parts, and De
Keninck and Roelfs give robust closed forms for `exp` and `log` (*Normalization, square roots,
and the exponential and logarithmic maps in geometric algebras of less than 6D*, 2022). In any
dimension, Roelfs and De Keninck factor a rotor into commuting simple rotors through its
tangent decomposition (*Graded symmetry groups: plane and simple*, 2021, §8–9). There the
planes' tangents follow from `R`'s grade parts and the roots of a polynomial, one root at a
time, and the logarithm is the sum of the factors' logarithms. This note reaches the same
polynomial, the cubic of 6D and 7D and the quartic of 8D and 9D, but takes a different route:

* **The planes are never separated.** The logarithm is `k` bivectors built from products of
  `R`'s grade parts, weighted by an interpolant of one scalar function at the polynomial's roots
  (§2). The weights of the bivectors come from a matrix whose determinant is ±1, so nothing is
  divided by a difference of roots there.
* **Coinciding roots are exact** (one plane, translations, isoclinic planes, the identity), where
  individual roots are ill-conditioned and factors are not unique. The interpolant is built from
  factors of the polynomial whose roots are apart, each through a series or its roots' values,
  joined by Chinese remaindering (§3).
* **Half turns.** Near a half turn in some plane the formula divides by `⟨R⟩₀ → 0`, and past
  one it returns a logarithm of `−R`. Planes near a half turn are first turned by a quarter
  turn (§5), exactly and branch free.

Results (§6, §7): on 175,000 random versors of seven algebras in 6D to 9D, each a logarithm of
`R` and a fixed point of `log ∘ exp`, with no NaN; within `10⁻¹³` of `R` for versors that are not
near a half turn, and within `10⁻⁶` to `10⁻⁹` for large boosts and planes near half turns. A
logarithm takes 2.6 to 3.8 µs (6.6 to 31 µs when planes are turned).

**The exponential** (§12) runs the same construction backwards: the invariants of `B` (the
squared wedge powers) give the planes' `μⱼ²` as the roots of a polynomial of the same degree, and
`exp B` is a polynomial in one bivector `T = Σ tanh(μⱼ) b̂ⱼ`, again an interpolant applied to
bivectors from products of `B`'s wedge powers. It is within `5·10⁻¹³` of a Taylor series on
random bivectors in 6D to 9D, and takes 3.5 to 10 µs, against 35 µs to 0.95 ms for scaling and
squaring in 7D to 9D.

## 1. Invariants of a versor: a polynomial

A unit even versor is `R = exp(B)` with `B = b₁ + … + b_k` commuting simple bivectors, `k = ⌊n/2⌋`
(the invariant decomposition; Roelfs and De Keninck 2021). With `μⱼ² = bⱼ²` (complex in general:
imaginary for rotations, real for boosts, zero for translations) and `b̂ⱼ = bⱼ/μⱼ`, so
`b̂ⱼ² = 1`:

```
R = ∏ⱼ (cⱼ + sⱼ b̂ⱼ),   cⱼ = cosh μⱼ,  sⱼ = sinh μⱼ.
```

The `b̂ⱼ` commute, and a product of distinct ones is a blade of higher grade. So `R`'s grade-`2m`
part collects the products of `m` of them:

```
⟨R⟩₂ₘ = Σ_{|S| = m} ∏_{j∈S} sⱼ ∏_{j∉S} cⱼ · b̂_S,     b̂_S = ∏_{j∈S} b̂ⱼ.
```

Square each part and keep the scalar. With `uⱼ = cⱼ² = cosh² μⱼ` and `sⱼ² = uⱼ − 1`,
`Aₘ = ⟨R₂ₘ R₂ₘ⟩₀ = Σ_{|S|=m} ∏_S (uⱼ − 1) ∏_{S̄} uⱼ`, a symmetric function of the `uⱼ`, and
`A₀ = ⟨R⟩₀²`. Their generating function is `∏ⱼ (uⱼ − y) = Σₘ Aₘ yᵐ (1 − y)^{k−m}`, so **the
invariants are the roots of a polynomial whose coefficients come from `R` alone**:

```
uⱼ = cosh² μⱼ  are the roots of  tᵏ − p₁ tᵏ⁻¹ + p₂ tᵏ⁻² − … ± p_k,
p_{k−j} = Σ_{m=0..j} (−1)ᵐ C(k−m, j−m) Aₘ.
```

For three planes (6D, 7D) and four (8D, 9D):

```
k = 3:  p₃ = A₀,  p₂ = 3A₀ − A₁,  p₁ = 3A₀ − 2A₁ + A₂
k = 4:  p₄ = A₀,  p₃ = 4A₀ − A₁,  p₂ = 6A₀ − 3A₁ + A₂,  p₁ = 4A₀ − 3A₁ + 2A₂ − A₃
```

(`A_k` is redundant: with the others it is `R ~R = 1`.) 7D has the same three planes as 6D and 9D
the same four as 8D; the planes then span a subspace of dimension `2k`, and nothing changes.

## 2. Separating the planes

`B = Σ μⱼ b̂ⱼ`, and `⟨R⟩₂ = Σ wⱼ b̂ⱼ` with `wⱼ = sⱼ ∏_{i≠j} cᵢ`. Products of consecutive grade
parts give more bivectors on the same planes, with other weights. With `r₀ = ⟨R⟩₀`:

```
G₁ = ⟨R⟩₂,   Gₘ = r₀ ⟨R₂ₘ R₂ₘ₋₂⟩₂  (m = 2 … k),
weight of plane j in Gₘ:  wⱼ · uⱼ · Σ_{S ⊆ others, |S| = m−1} ∏_S (uᵢ − 1) ∏_{others∖S} uᵢ.
```

(The grade-2 part of `b̂_S b̂_{S'}` with `|S| = |S'| + 1` is nonzero only for `S = S' ∪ {j}`, where
it is `b̂ⱼ`.) Reduced modulo the polynomial, each weight is `wⱼ` times a polynomial in `uⱼ` of
degree below `k`, with coefficients in the `p`s: rows of a matrix `M` in the basis `1, u, u², …`:

```
k = 3:  G₁: [1, 0, 0]          k = 4:  G₁: [1, 0, 0, 0]
        G₂: [2p₃, −p₁, 1]              G₂: [3p₄, −p₂, p₁, −1]
        G₃: [p₃, 1 − p₁, 1]            G₃: [3p₄, p₁ − 2p₂, 2p₁ − 1, −2]
                                       G₄: [p₄, p₁ − p₂ − 1, p₁ − 1, −1]
```

`det M = ±1` in both cases, so the bivectors `Qᵢ` of weight `wⱼ uⱼⁱ` are integer polynomial
combinations of the `G`s, `Q = M⁻¹ G`:

```
k = 3:  Q₁ = p₃G₁ − G₂ + G₃,   Q₂ = p₃(p₁ − 2)G₁ + (1 − p₁)G₂ + p₁G₃
k = 4:  Q₁ = p₄G₁ − G₂ + G₃ − G₄
        Q₂ = p₄(p₁ − 3)G₁ + (2 − p₁)G₂ + (p₁ − 1)G₃ − p₁G₄
        Q₃ = p₄(p₁² − 3p₁ − p₂ + 3)G₁ + (2p₁ + p₂ − p₁² − 1)G₂ + (p₁² − p₁ − p₂)G₃ + (p₂ − p₁²)G₄
```

Since `μⱼ/wⱼ = (μⱼ/sⱼ)(cⱼ/r₀)`,

```
B = r₀⁻¹ Σᵢ αᵢ Qᵢ,
```

where `Σ αᵢ uⁱ` is **the polynomial of degree `k − 1` interpolating** `φ(uⱼ) = cⱼ μⱼ/sⱼ` at the
roots. As a function of `u`,

```
φ(u) = √u · asinh(√(u−1)) / √(u−1),
```

analytic at `u = 1` (where `φ = 1`), singular only at `u = 0` (a half turn), with its branch
cut on `u ≤ 0`. Everything is products of `R`'s grade parts and `k + 1` numbers; the planes are
never found.

## 3. Evaluating the interpolant without its singularities

Through the roots, an interpolant is a combination of divided differences, which divide by
differences of roots. Roots coincide in the commonest cases: a rotation in one plane (roots
`cos²θ, 1, 1, …`), translations (null planes: `u = 1`), isoclinic planes, the identity. The
individual roots are ill-conditioned there too (a double root moves like the square root of a
perturbation). The interpolant is instead built from **factors of the polynomial whose roots are
apart**: for each factor, the interpolant of `φ` modulo that factor, which depends on the factor's
roots only symmetrically; then the factors are joined by Chinese remaindering, which divides by
the factors' resultants, products of differences between roots that are apart.

**A factor of close roots: `φ`'s series at their mean, reduced modulo the factor.** The
interpolant's coefficients are symmetric functions of the roots, so they follow from the factor's
coefficients alone. In `t = u − m` (`m` the mean), `tᵏ` reduced modulo the factor is a
polynomial of lower degree by a short recurrence, and the interpolant is `Σ φₖ tᵏ` reduced.
Coinciding roots need nothing special. This is used while the roots' spread (bounded from the
coefficients, Fujiwara's bound) is under a quarter of `m`, the distance to `φ`'s singularity, so
26 terms reach `10⁻¹⁶`. The series is computed in `s = t/m`: in `t`, its coefficients grow like
`m⁻ᵏ`, which overflows `f32` for a centre near a half turn.

`φ`'s Taylor coefficients at any centre come without cancellation from `φ(u) = √u · F(u−1)`,
`F(x) = asinh(√x)/√x`: near `x = 0` (`|x| < 1/4`), `F`'s Maclaurin series (90 terms) is shifted
to the centre by repeated Horner; elsewhere it is composed from series that are analytic there:
`v = √(x₀ + t)`, `asinh(v)' = v'/√(m + t)`, `F = asinh(v)/v` (or `asin` for `x₀ < 0`).

**Three planes (6D, 7D).** One real root always exists. Where the roots are not all close, the
most isolated real root `r` (Cardano, or the trigonometric form for three real roots, then a
Newton step) and the remaining pair by its sum `S` and product `P`, never the individual pair.
The pair's line is `φ`'s series at its midpoint reduced modulo `t² − d²` when the pair is close,
and the chord through its two values otherwise (a conjugate pair's chord is real). "Close" needs
a positive midpoint: a loxodromic pair can have invariants with a negative real part, and then it
straddles the branch cut of `√u`, which the series cannot cross. The quadratic adds the isolated
root: `L(u) + (u² − Su + P)(φ(r) − L(r))/(r² − Sr + P)`.

`S` and `P` come from whichever side is stable: forward (`S = p₁ − r`, `P = p₂ − rS`) for a small
`r`, backward (`P = p₃/r`, `S = (p₂ − P)/r`) for a large one, each by its error bound. Two planes
near a half turn and a third, say, have a pair of invariants near `10⁻⁸`, whose product
`p₂ − rS` would be lost to the rounding of `p₂` (whose absolute error is `ε`), and the pair would
look far apart.

**Four planes (8D, 9D).** The roots can group in more ways: all four close; one apart and three
(close or not); two and two; or, with two loxodromic pairs whose invariants nearly coincide
(possible in `R(4,4)` and `R(5,4)`), two complex conjugate clusters. The interpolant takes, per
lane, whichever grouping separates its factors best:

* **All four close:** the series at their mean, reduced modulo the quartic.
* **An isolated real root `r` and a cubic:** the three-plane interpolant of the cubic (above),
  and `r` added as in 6D. The cubic comes from deflation by `r`, each coefficient from the stable
  side.
* **Two real quadratics:** each pair's line (series or chord, as in 6D), joined by
  `La + qa·K`, `K = (Lb − La) qa⁻¹ mod qb`; the inverse of `qa ≡ g₁u + g₀` modulo `qb` is
  `(−g₁u + g₁Sb + g₀)/N` with `N = qa(b₁) qa(b₂)` (the resultant).
* **Two conjugate complex quadratics:** the same in complex arithmetic, with `φ`'s series at a
  complex midpoint; the result is real.

The groupings come from the resolvent cubic of the shifted quartic `t⁴ + a t² + b t + c`,
`y³ + 2a y² + (a² − 4c) y − b² = 0`: one root per way of pairing the four roots, `y = (t₁ + t₂)²`,
real and non-negative for a pairing into real factors, negative for one into conjugate factors.
Approximate roots come from Euler's form `t = ½(±√y₁ ± √y₂ ± √y₃)` with `√y₁√y₂√y₃ = −b`,
which holds up where roots cluster (a multiple resolvent root; the textbook route through one
resolvent root divides by a pair sum that is then zero). Each grouping's gap (the least distance
between its factors' roots) picks the grouping; a conjugate grouping must also pair each root
with the other factor's conjugate. The chosen factors are then refined in `u` (not `t`, where a
small root loses its relative accuracy): a quadratic by three Newton steps on the factorization
(Bairstow's method, quadratically convergent exactly where the factors are apart), an isolated
root by Newton steps kept only where they lower the residual (at a multiple root the derivative
vanishes). The other factor follows by deflation, from the stable side.

## 4. Conditioning and the branch

* **Near a half turn** (`r₀ → 0`) the formula divides by `r₀` and loses `ε/r₀`: `6·10⁻¹⁵`
  at `r₀ = 0.01`, `8·10⁻⁹` at `10⁻⁸`.
* **The branch.** The closed form takes `cⱼ = √uⱼ` with a non-negative real part: every
  invariant plane's rotation is below a half turn (for a loxodromic pair, the principal value
  of the pair). That is the geometric principal logarithm, and it is idempotent:
  `log(exp(log R)) = log R`. It is right wherever `r₀ > 0`: then an even number of the `cⱼ`
  are negative, and turning two planes by a half turn each leaves `R` unchanged. Where
  `r₀ < 0` an odd number are, and the formula returns a logarithm of `−R`.

So the formula alone serves `r₀ > 1/16`. Below, the planes near a half turn are first turned out
of the way (§5).

## 5. Turning planes near a half turn

A rotation plane (`uⱼ < 1`, `cⱼ = cos θⱼ`, `sⱼ = sin θⱼ`) near a half turn has `cⱼ ≈ 0`.
Multiplying `R` by a quarter turn in that plane, `−b̂ⱼ = exp(−(π/2) b̂ⱼ)`, turns its factor
`cⱼ + sⱼ b̂ⱼ` into `sⱼ − cⱼ b̂ⱼ`: its angle drops by `π/2`, and `sⱼ` takes the place of `cⱼ` in
`r₀`. For a set `T` of rotation planes, with `E = ∏_T (−b̂ⱼ)`,

```
R' = R E,   ⟨R'⟩₀ = ∏_T sⱼ ∏_rest cⱼ,   log R = log R' + (π/2) Σ_T b̂ⱼ,
```

since the `b̂ⱼ` commute with `R`'s planes. The closed form then serves `R'`.

* **The sign comes for free.** Orient each `b̂ⱼ` so that its weight `wⱼ` in `⟨R⟩₂` is
  positive. Then `sign(sⱼ) = sign(∏_{i≠j} cᵢ)`, and `sign ⟨R'⟩₀ = sign(r₀)^{|T|+1}`: positive
  when `T` has an odd number of planes, the sign of `r₀` for an even one. So turning one plane
  repairs `r₀ < 0` too, in any dimension.
* **Which planes.** `|⟨R'⟩₀| = ∏_T √(1 − uⱼ) ∏_rest √uⱼ` follows from the roots, and so do the
  planes' weights `|wⱼ| = √((1 − uⱼ) ∏_{i≠j} uᵢ)`. A plane's direction is read from its weight,
  so turning it costs `ε/|wⱼ|`, and the closed form on `R'` costs `ε/⟨R'⟩₀`. Among the sets with a
  positive `⟨R'⟩₀`, none included, the one with the smallest `1/⟨R'⟩₀ + Σ_T 1/|wⱼ|` is turned. For
  example, with one plane at a half turn and another at `u = 0.26`, turning both gives the larger
  `⟨R'⟩₀`; but the second plane's weight contains the first one's cosine and is `3·10⁻⁴`, so only
  the first is turned.
* **The candidates follow the grouping of the roots (§3):** close roots are turned together (a
  cluster), or apart roots one by one. In 6D, none, all three through the series at their mean,
  and unions of the isolated root, the pair and each of the pair; in 8D, the same for the cubic
  with the isolated root in or out, or none, both or either one of each pair, and all four
  together. Sets that separate groups are preferred where the groups are apart (a gap of at least
  1/64, relative). The split is computed even where all roots are close: with `r₀ < 0` only an
  odd set can be turned, and four close but distinct invariants (0.56 to 0.69, say) leave a
  single plane apart enough.
* **The planes' sum** `Z = Σ_T b̂ⱼ` is again an interpolant applied to the bivectors of §2:
  `Z = Σ αᵢ Qᵢ` with `α` interpolating `h(u) = [u ∈ T]/|w(u)|` at the roots, through the same
  factors and joins as `φ`. On a cluster `h` is the smooth `√(u/(1−u))/|r₀|`, taken through its
  series (in `s = t/ρ`, `ρ = min(m, 1 − m)`).
* **The product `E`** is a polynomial in `Z`: for `n` commuting orthogonal unit rotation
  bivectors, the elementary symmetric functions of the `b̂ⱼ` follow from the power sums
  `Σ b̂ⱼᵏ` by Newton's identities, and `∏(−b̂ⱼ)` is `−Z`, `1 + Z²/2`, `−(Z³ + 7Z)/6` or
  `1 + 2Z²/3 + Z⁴/24` for `n = 1 … 4`. `R E` is formed by products with the bivector `Z` only.

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

## 6. Validation

Checked in `f64` against `exp`. The CSTA table was measured against scaling and squaring,
independent of the log; the random versors against the closed-form `exp` of §12, which shares
the root groupings with the log but was itself checked against a Taylor series (§12).

**CSTA** (`R(4,2)`):

| case | `max |log(exp B) − B|` |
|---|---|
| random bivectors, entries up to 0.4 / 0.6 | `2·10⁻¹⁵` / `5·10⁻¹⁵` |
| one rotation up to half-angle 1.5 (a half turn is π/2) | `7·10⁻¹⁶` |
| boosts and dilations up to rapidity 3 | `4·10⁻¹⁵` |
| translation (null), rotation and translation, isoclinic, near the identity | `0` to `10⁻¹⁶` |
| towards a half turn, `π/2 − 10⁻⁸` (turned) | `10⁻¹⁶` |
| past a half turn (`r₀ < 0`), and towards one with a boost or dilation (turned) | `10⁻¹²` or better |

**Random versors** `exp(B)`, 5000 per size and algebra, checked for `exp(log R) = R` (the largest
coefficient difference, relative to the largest coefficient). "Turned" counts the versors with
`⟨R⟩₀ < 1/16`:

| algebra | entries up to | turned | worst `exp(log R) − R` |
|---|---|---|---|
| CSTA `R(4,2)` | 0.5 / 1.0 / 1.5 / 2.0 / 2.5 | 0 / 303 / 1991 / 2987 / 3418 | `1.2·10⁻¹⁵` / `10⁻¹³` / `6·10⁻⁹` / `1.7·10⁻⁶` / `3·10⁻⁹` |
| 7D `R(4,3)` | 0.25 / 0.5 / 0.75 / 1.0 / 1.5 | 0 / 0 / 0 / 52 / 690 | `3·10⁻¹⁶` / `1.7·10⁻¹⁵` / `4·10⁻¹⁴` / `7·10⁻¹⁴` / `1.8·10⁻⁹` |
| 8D `R(8,0)` | same | 0 / 79 / 4929 / 3637 / 2180 | `6·10⁻¹⁶` / `1.3·10⁻¹³` / `7·10⁻¹¹` / `4·10⁻¹¹` / `10⁻¹⁰` |
| 8D `R(4,4)` | same | 0 / 0 / 0 / 25 / 277 | `4·10⁻¹⁶` / `1.4·10⁻¹⁵` / `1.5·10⁻¹³` / `2.4·10⁻¹⁰` / `8·10⁻⁹` |
| 9D `R(9,0)` | same | 0 / 751 / 4966 / 2919 / 4254 | `10⁻¹⁵` / `4·10⁻¹³` / `5·10⁻¹¹` / `1.1·10⁻¹¹` / `5·10⁻⁷` |
| 9D `R(5,4)` | same | 0 / 0 / 6 / 218 / 1024 | `4·10⁻¹⁶` / `3·10⁻¹⁵` / `1.4·10⁻¹³` / `7·10⁻⁹` / `4·10⁻⁷` |
| 9D PGA `R(8,0,1)` | same | 0 / 80 / 4933 / 3651 / 2198 | `7·10⁻¹⁶` / `1.5·10⁻¹³` / `7·10⁻¹¹` / `4·10⁻¹¹` / `3·10⁻¹⁰` |

No log is NaN, none misses `R`, and every one is a fixed point of `log ∘ exp` (within `10⁻⁶`).
The least accurate ones combine large boosts (`cosh² μ` in the hundreds to thousands) with
planes near a half turn or a full turn, where `exp` itself is ill-conditioned.

The four-plane interpolant alone (`p`s from chosen roots, checked against `φ` at every root) was
stressed on 180,000 root sets in `f64` and 27,000 in `f32`: rotations and boosts, clusters of two,
three and four at distances from `0` to `10⁻²`, two pairs, conjugate pairs with real roots, two
conjugate pairs, and two nearly equal conjugate pairs. In `f64` it is within `5·10⁻¹²` of `φ`
except near `u = 0` (a half turn, where `φ ~ √u` and turning takes over). In `f32` it has no NaN,
and is within `2·10⁻⁷` wherever the roots are away from `u = 0` and from `φ`'s branch cut (§11).

The tests: `tests/csta_log.rs` (CSTA, portable SIMD lanes mixing the closed form and turning,
`f32` within `2·10⁻⁴`), `tests/log6d_algebras.rs` (`R(6,0)`, `R(3,3)`, 5D PGA `R(5,0,1)`), and
the crate `gax-highdim-tests`: `R(7,0)`, `R(4,3)`, 6D PGA `R(6,0,1)`, `R(8,0)`, `R(4,4)`, 7D PGA
`R(7,0,1)`, `R(9,0)`, `R(5,4)` and 8D PGA `R(8,0,1)`, each for `log(exp B) = B`, random versors
past half turns, single planes, clusters of two to four coinciding planes near and past half
turns, lanes and `f32`.

## 7. Cost

| algebra | `log` (closed form) | `log` (planes turned) | `exp` (closed form, §12) | `exp` (scaling and squaring, before) |
|---|---|---|---|---|
| CSTA (6D, 32 coefficients) | 2.6 µs | 6.6 µs | 3.5 µs (4.3 µs turned) | 4.1 µs |
| 7D `R(4,3)` (64) | 2.5 µs | 7.0 µs | 4.0 µs | 35 µs |
| 8D `R(4,4)` (128) | 2.9 µs | 10 µs | 6.8 µs | 0.24 ms |
| 9D `R(5,4)` (256) | 3.8 µs | 31 µs | 10 µs | 0.95 ms |

(f64, Ryzen 7 5800X, one core, release builds; the machine was not idle.) The closed form is
dominated by one straight-line program for `r₀`, the `p`s and the `G`s. Turning adds `k`
products by the bivector `Z` and a second closed form. Inverse scaling and squaring took 50.7 µs
in CSTA. Scaling and squaring in CSTA works in the 32-coefficient even kind, where a product is
cheap; from 7D on its products (up to 65,536 terms) dominate, and the closed form, whose cost is
one straight-line program over the bivector, wins by 9x to 90x.

## 8. In WGSL

The WGSL modules take kinds of up to 64 coefficients: CSTA's and every 6D and 7D algebra's
(declared with `algebra!`). They have `unit_even_log(x: Even) -> Bivector`:

* `unit_even_log_closed` is a recorded kernel: `r₀`, the `p`s and the three bivectors as one
  verified straight-line program, the weights from `study_log6`, and their combination.
  `study_log6` ports the three-plane interpolant to `f32` with the same regimes, but 16 series
  terms (the regimes keep the ratio at most 1/4, so 16 reach `2·10⁻¹⁰`, below `f32`'s precision)
  and branches instead of selects.
* `unit_even_log_turning` is another: the same program, `study_log6_turn` (the turning), and `Z`.
* `unit_even_log` takes the closed form above `⟨x⟩₀ = 1/16`. Below, it recovers the number of
  turned planes as `n = −⟨Z²⟩₀`, forms `E` from `Z`, and returns
  `unit_even_log_closed(x E) + (π/2) Z`.

In the `f16` modules the helpers and the programs feeding them compute in `f32`. Checked with
`wesl`'s CPU evaluator: both kernels agree with their `f64` evaluations, and `unit_even_log`
agrees with the Rust `log` within `6·10⁻⁵` relative in CSTA and `4·10⁻⁷` in 7D. On a GPU
(`gax-gpu-tests`) CSTA's agrees within `3·10⁻⁶`.

## 9. Beyond 9D

From 10D on (five planes) the same construction holds (§1, §2), but the polynomial has degree 5
and more, so Abel–Ruffini rules out roots in radicals. The groupings would then come from
numerical roots (and the factors from Newton steps on the factorization, as for the quartic),
and turning carries over as it is (`E` a polynomial of degree up to `k` in `Z`). The cost grows
quickly: the even kind has `2ⁿ⁻¹` coefficients (512 in 10D), and gax's generator stops at 9D.

## 10. Implementation

In gax, `gax_core::study::{log_coeffs_6d, log_turn_6d}` (three planes) and
`gax_core::study::{log_coeffs_8d, log_turn_8d, q_weights_8d, turn_polynomial_8d}` (four planes)
compute the interpolants; the generator emits `Log<Bivector> for Unit<Even>` for the full even
kind of every 6D to 9D algebra: the invariants and the `G`s as one straight-line program,
compiled and verified symbolically, then the weights. `exp` (§12) is
`gax_core::study::{exp_weights_6d, exp_turn_6d, exp_reach_6d}` and their `_8d` twins, with
`h_weights`; the generator emits `Bivector::exp` from them for the same algebras, the wedge
powers and the `H`s as one straight-line program (`exp_invariants`). Algebras from 7D on also
need the generator itself to scale (sandwiches without symbolic simplification, products as loops over
tables of terms); see ADR-034 in [design.md](design.md).

## 11. Limitations

* **Two or three planes near a half turn together.** `R` fixes only their product at a half
  turn, so near one the planes are determined to about `ε/δ` (`δ` the distance to the half turn).
  Turning them loses more: the pair's `Z` needs the slope of `h` across two roots near `u = 0`,
  and `exp(log R)` is within about `2·10⁻¹⁵/δ²` of `R` (`10⁻⁷` at `δ = 10⁻⁴`, `10⁻³` at `10⁻⁶`),
  where an exact method would reach `ε`. Reading their product from `⟨R⟩₄` would avoid this.
* **A loxodromic pair on the branch cut.** A pair of conjugate invariants with a negative real
  part and a small imaginary part (the rotation part of a `(2,2)` block near a half turn) has
  `φ` values across the cut of `√u`: the chord through them divides by the small imaginary part.
  In `f64` this costs little (the one CSTA versor above, within `1.4·10⁻⁶`); in `f32` it can cost
  all digits. Turning covers rotation planes only.
* **Four coinciding invariants with `⟨R⟩₀ < 0`** (three planes at one angle and the fourth at its
  supplement): the planes are not determined by the invariants, the logarithm is not unique, and
  no set can be turned. The closed form then returns a logarithm of `−R`.
* **`exp` in `f32`** is within `10⁻⁴` of the `f64` result for most bivectors, but up to `10⁻³`
  for large boosts (entries up to 2.5 in `R(4,3)`), where `C` and `T` are large and the wedge
  powers of `T` cancel against each other. In `f64` the same cases are within `3·10⁻¹²`.
* **`exp` in WGSL** is still scaling and squaring. A port would reuse `study_log6`'s three-plane
  groupings with `τ` and `ln cosh` as data.

## 12. The exponential

The construction runs backwards for `exp B`. Write `B = Σ μⱼ b̂ⱼ` with commuting simple
`b̂ⱼ`, `b̂ⱼ² = 1` (so `μⱼ` is imaginary for a rotation, real for a boost, complex for a
loxodromic pair, and a null plane has `μⱼ b̂ⱼ` with `λⱼ = 0`), and `λⱼ = μⱼ²`.

**The invariants.** The wedge powers `Wₘ = B^∧m/m!` are the sums over `m` planes of the
products `∏ μⱼ b̂ⱼ`, so `eₘ = ⟨Wₘ²⟩₀` is the `m`-th elementary symmetric function of the
`λ`s: they are the roots of

```text
λᵏ − e₁ λᵏ⁻¹ + e₂ λᵏ⁻² − … ± eₖ = 0,
```

a polynomial of the degree of §1's, in `λ` rather than `u`. The bivectors `Hₘ = ⟨Wₘ Wₘ₋₁⟩₂`
weigh plane `j` by `μⱼ eₘ₋₁(λ without λⱼ) = Σₜ (−1)ᵗ eₘ₋₁₋ₜ λⱼᵗ μⱼ`. In terms of
`Bᵢ = Σ λⱼⁱ μⱼ b̂ⱼ` this is a triangular matrix with `±1` on its diagonal, so every `Bᵢ` is an
integer polynomial combination of `H₁ = B, H₂, …, Hₖ` in the `e`s (`h_weights`), as the `Q`s of
the log are combinations of the `G`s.

**The formula.** The planes commute, so

```text
exp B = ∏ (cosh μⱼ + sinh μⱼ b̂ⱼ) = C ∏ (1 + tⱼ b̂ⱼ) = C (1 + T + T∧T/2 + … + T^∧k/k!),
T = Σ tⱼ b̂ⱼ,   tⱼ = tanh μⱼ,   C = ∏ cosh μⱼ.
```

`T = Σ τ(λⱼ) μⱼ b̂ⱼ` with `τ(λ) = tanh(√λ)/√λ`, which is even in `√λ` and so a function of
`λ`, analytic but for poles at the half turns `λ = −(π/2 + nπ)²`. With `P(λ) = Σ αᵢ λⁱ`
interpolating `τ` at the roots, `T = Σ αᵢ Bᵢ`, a combination of the `H`s. The wedge powers of `T`
come from the same straight-line program as `B`'s. `C` is `exp` of the trace of `ln cosh √λ`
over the roots: `Σⱼ Q(λⱼ) = Σ βᵢ pᵢ` for its interpolant `Q = Σ βᵢ λⁱ`, with the power sums `pᵢ`
from the `e`s by Newton's identities. This avoids `C = ∏ (1 − tⱼ²)^(−1/2)`, which cancels for
large boosts (`tⱼ → 1`) and made `f32` (and, less, `f64`) inaccurate there. Null planes need nothing special
(`τ(0) = 1`, `ln cosh 0 = 0`), nor do coinciding planes: the interpolants come from the groupings
of §2 and §3 (series at a cluster's centre, the isolated root and a pair, the quartic's
groupings from Euler's resolvent, Chinese remaindering), written once for any data with values,
complex values, a Taylor series at a centre, and a distance to its nearest singularity. `τ` and
`ln cosh √λ` have Maclaurin series of radius `π²/4`, shifted to the centre; far out, `ln cosh`
is `z + ln(1 + e⁻²ᶻ) − ln 2`.

**Turning and halving.** `τ`'s pole at a half turn makes the interpolant ill-conditioned for
rotations near one, so rotation planes beyond a quarter turn (`λ < −π²/16`) are turned back by a
quarter turn first, as the log turns its planes (§5):

```text
B = B' + (π/2) Z,   exp B = exp B' · ∏ êⱼ = exp B' · (−1)ⁿ ∏(−êⱼ),
```

with `Z = Σ êⱼ` the sum of the turned planes' unit bivectors (`êⱼ² = −1`, along `B`), an
interpolant of `1/θ` (0 on the other planes) applied to the `H`s, and `n` their number, the trace
of their indicator. `∏(−êⱼ)` is the log's polynomial in `Z` (degree up to `k`). A cluster within
a quarter of its reach of the threshold is turned or not as a whole, by its centre; roots
further apart use a chord, whose slope is bounded. Turning only happens where some rotation is
beyond `1.1 π/4` (`τ` is still accurate a little past `π/4`), so the common case is one
closed form. Turned rotations end within `π/4` of zero, but a rotation beyond `3π/4`, or a
loxodromic pair whose rotation part is beyond `π/4` (turning covers rotation planes only), is
left near a pole: there `B` is halved until it is not, and the result squared as often. So is
a boost of rapidity over 8: the interpolant's terms grow like `λ` while `τ(λ) ~ 1/√λ`, so it
cancels about `|λ|^(3/2)` ulps (at rapidity 37, `exp B` was unit only to `4·10⁻¹²`; halved, to
`2·10⁻¹⁶`).

**Accuracy.** Against a Taylor series of `exp B` (degree 30, after enough halvings, then
squared back), on 1000 to 5000 random bivectors per size, relative to the largest coefficient:

| algebra | entries up to 0.5 / 1.0 / 1.5 / 2.5 | `f32` against `f64` |
|---|---|---|
| CSTA `R(4,2)` | `1.6·10⁻¹⁴` / `4·10⁻¹⁴` / `1.5·10⁻¹³` / `7·10⁻¹⁴` | `7·10⁻⁷` / `2.5·10⁻⁵` / `5·10⁻⁵` / `7·10⁻⁵` |
| 7D `R(4,3)` | `1.4·10⁻¹⁴` / `1.5·10⁻¹³` / `1.1·10⁻¹³` / `3·10⁻¹²` | `3·10⁻⁷` / `10⁻⁴` / `4·10⁻⁴` / `1.2·10⁻³` |
| 8D `R(8,0)` | `3·10⁻¹⁴` / `8·10⁻¹⁴` / `2·10⁻¹³` / `2·10⁻¹³` | `1.7·10⁻⁵` / `9·10⁻⁵` / `1.5·10⁻⁴` / `1.6·10⁻⁴` |
| 8D `R(4,4)` | `8·10⁻¹⁴` / `1.3·10⁻¹³` / `1.7·10⁻¹³` / `3·10⁻¹³` | `2·10⁻⁷` / `9·10⁻⁶` / `3·10⁻⁵` / `1.2·10⁻⁴` |
| 9D `R(9,0)` | `5·10⁻¹⁴` / `1.2·10⁻¹³` / `2·10⁻¹³` / `5·10⁻¹³` | `3·10⁻⁵` / `6·10⁻⁵` / `1.4·10⁻⁴` / `1.8·10⁻⁴` |
| 9D `R(5,4)` | `1.2·10⁻¹³` / `2.6·10⁻¹³` / `2.7·10⁻¹³` / `5·10⁻¹³` | `10⁻⁶` / `2.5·10⁻⁵` / `3.5·10⁻⁵` / `4·10⁻⁵` |
| 9D PGA `R(8,0,1)` | `4·10⁻¹⁴` / `1.5·10⁻¹³` / `4·10⁻¹³` / `6·10⁻¹³` | `1.7·10⁻⁵` / `8·10⁻⁵` / `1.4·10⁻⁴` / `3·10⁻⁴` |

No result is NaN. In `gax-highdim-tests`, every 7D to 9D algebra checks `exp` of sums of basis
planes against the product of their exponentials (rotations within a quarter turn, beyond one,
and beyond three quarters; boosts; null planes), against the series, in SIMD lanes and in `f32`;
every `log` test there and in `tests/csta_log.rs` and `tests/log6d_algebras.rs` goes through
`exp`. Cost: §7.

## Appendix: the algorithm

For a unit even versor `R` with `k` invariant planes (`k = 3` in 6D and 7D, 4 in 8D and 9D):

1. `r₀ = ⟨R⟩₀`, `Aₘ = ⟨R₂ₘ²⟩₀`, the `p`s by §1; `G₁ = ⟨R⟩₂`, `Gₘ = r₀⟨R₂ₘR₂ₘ₋₂⟩₂`.
2. If `r₀ > 1/16`: `α` = the interpolant of `φ` at the roots (§3), and
   `log R = r₀⁻¹ Σ αᵢ Qᵢ` with `Q = M⁻¹G` (§2), folded into `k` weights of the `G`s.
3. Else: the set `T` of planes to turn and the interpolant `α'` of `h` (§5); `Z = Σ α'ᵢ Qᵢ`;
   `n = |T|`; `E = e(n, Z)`; `R' = R E`; `log R = (step 2 on R') + (π/2) Z`.

The interpolant (§3): if the roots are close, the series at their mean modulo the polynomial.
Otherwise, factors that are apart (an isolated root and the rest; two pairs; conjugate pairs),
refined by Newton steps and deflated from the stable side, each factor's part from its series
(close roots) or its roots' values, joined by Chinese remaindering.
