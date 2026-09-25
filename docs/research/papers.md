# Literature research for an extensor-based Rust GA library

Compiled 2026-09-25. Every source below has a citation, a link, a summary, the formulas we need,
and a "what we take" list.

**How to read the tags.**
- **[VERIFIED-NUM]**: I checked the formula numerically in this session with a small Python Clifford
  implementation in the bivector.net PGA3D basis. The scripts are `pga_check.py`, `dyn_check.py`,
  `inv_check.py` and `dec_check.py` in the scratchpad.
- **[VERBATIM]**: transcribed from the source text or code, which I read.
- **UNVERIFIED**: I could not check it against the source text. It is either from memory or
  derived by me.

## 0. Conventions used throughout (bivector.net PGA3D, R(3,0,1))

- **Basis order (16).** `1, e0, e1, e2, e3, e01, e02, e03, e12, e31, e23, e021, e013, e032, e123, e0123`.
  Source: the bivector.net 3DPGA cheat sheet (SIGGRAPH 2019 course notes, https://bivector.net/3DPGA.pdf).
- **Metric.** `e0² = 0` and `e1² = e2² = e3² = 1`. The Euclidean bivectors square to -1
  (`e12² = e31² = e23² = -1`), `e123² = -1`, and the ideal blades square to 0.
- **Dual (bivector.net `!`/`*`).** This is just reversal of the coefficient array, with no sign
  changes: `a+b e0+…+p I ↦ p+o e0+…+a I`. So `e01↔e23`, `e02↔e31`, `e03↔e12`, `e0↔e123`,
  `e1↔e032`, `e2↔e013`, `e3↔e021`.
  - Regressive product: `A ∨ B = ⋆(⋆A ∧ ⋆B)`. Here `⋆` is its own inverse. [VERIFIED-NUM, used consistently in the inertia check]
- **Reverse.** Negates grades 2 and 3.
- **Elements.**
  - Plane: `a e1 + b e2 + c e3 + d e0` (the plane `ax+by+cz+d=0`).
  - Finite point: `x e032 + y e013 + z e021 + e123`.
  - Ideal point: the same with a zero `e123` coefficient.
- **Sandwich.** `X' = M X M̃` for any grade when M is even. The general rule is
  `U[V] = (-1)^{grade U · grade V} U V U⁻¹` (Roelfs & De Keninck).
- **Even subalgebra order used by the papers' code.** `[1, e01, e02, e03, e12, e31, e23, e0123]`,
  with bivector order `[e01, e02, e03, e12, e31, e23]`. This is the bivector.net order restricted to
  even grades, so the code below can be pasted as is.
- **Plücker pairing.** Always `e01↔e23`, `e02↔e31`, `e03↔e12`.
  - `B∧B = 2(b01 b23 + b02 b31 + b03 b12) e0123`
  - `B·B = −(b12² + b31² + b23²)`
  - [VERIFIED-NUM]
- **PGA2D (bivector.net).** Basis `1, e0, e1, e2, e01, e20, e12, e012`. The dual is again coefficient
  reversal. Point: `x e20 + y e01 + e12` (from https://bivector.net/2DPGA.pdf).

---

## 1. Hestenes & Sobczyk, *Clifford Algebra to Geometric Calculus*, ch. 3

**Citation.** D. Hestenes, G. Sobczyk, *Clifford Algebra to Geometric Calculus: A Unified Language for
Mathematics and Physics*, D. Reidel, Dordrecht, 1984 (paperback 1987). DOI 10.1007/978-94-009-6292-7.

**Structure of ch. 3, "Linear and Multilinear Functions".**
- 3-1: Linear Transformations and Outermorphisms
- 3-2: Characteristic Multivectors and the Cayley–Hamilton Theorem
- Later sections: eigenblades, symmetric/skew maps, normal forms, and tensors/extensors
- The exact numbering after 3-2 is UNVERIFIED. I could not open the book. The two section titles
  above come from the Google Books table of contents and search snippets. The formulas below are
  checked against Wikipedia's "Outermorphism" article, which cites H&S pp. 68–70.

### Definitions to implement

- **Extensor.** A multilinear, multivector-valued function of multivector arguments. A
  `(p,q)`-extensor in the Fernández–Moya–Rodrigues terminology (see §7) is a linear map from
  `Λ^p V` to `Λ^q V`.
  - Our core abstraction: a linear map between graded subspaces, stored as a matrix acting on blade
    coefficients.
  - Examples: an outermorphism is block-diagonal by grade, and the rigid-body inertia is a
    6×6 map from bivectors to bivectors.
- **Outermorphism** of a linear `f: V→V`.
  - `f(α) = α` for scalars, and `f(a∧b∧…) = f(a)∧f(b)∧…`.
  - It preserves grade: `f(⟨A⟩_r) = ⟨f(A)⟩_r` (H&S p. 68).
  - The grade-r block of its matrix is the r-th compound matrix: the r×r minors of the vector
    matrix `F`, indexed by blades.
  - Composition: `(f∘g)` extends to `f̲∘g̲`.
- **Adjoint.** `f̄` is defined by `a·f(b) = f̄(a)·b`. It extends to multivectors with the scalar
  product: `f̄(A) * B = A * f(B)`.
  - Practical formula: `f̄(a) = ∂_b (a·f(b)) = Σ_i e^i (a·f(e_i))`, where `e^i` is the reciprocal frame.
  - In matrix terms, `F̄ = G⁻¹ Fᵀ G` with metric Gram matrix G. For an orthonormal basis this is
    `Fᵀ` with signs.
  - Useful H&S identities (sign conventions UNVERIFIED): `A·f̄(B) = f̄[f(A)·B]` for
    grade A ≤ grade B, and `f(A)·B = f[A·f̄(B)]` for grade A ≥ grade B.
- **Determinant.** `f(I) = (det f) I`, so `det f = f(I) I⁻¹` (H&S p. 70). Also `det f̄ = det f` and
  `det(f∘g) = det f · det g`.
  - This uses only the wedge, so it is metric-free: compute `f(e1)∧…∧f(en)` and read off the
    pseudoscalar coefficient.
- **Inverse.**
  - `f⁻¹(X) = f̄(X I) [f̄(I)]⁻¹ = f̄(X I) I⁻¹ / det f`
  - `f̄⁻¹(X) = [f(I)]⁻¹ f(I X)`
  - Sanity check: `f = λ·id` in 3D gives `λ² a / λ³ = a/λ`.
- **Characteristic polynomial and Cayley–Hamilton.**
  - H&S define characteristic multivectors from `f_(r) = f(a1)∧…∧f(ar)` contracted with
    `∂_(r) = ∂_{a_r}∧…∧∂_{a_1}`. The notation and normalization (`1/r!`) are UNVERIFIED.
  - Implementable, metric-free equivalent:
    - `c_r = tr(Λ^r f) = Σ_{|J|=r} coeff_J( f(e_J) )`, i.e. the sum of principal r×r minors.
    - `det(λ − f) = Σ_{r=0}^{n} (−1)^r c_r λ^{n−r}`
    - Cayley–Hamilton: `Σ_{r=0}^{n} (−1)^r c_r f^{n−r} = 0`
  - Consequences: `c_1 = tr f = ∂_a·f(a)` and `c_n = det f`.
  - Inverse via Cayley–Hamilton:
    `f⁻¹ = (1/c_n) Σ_{r=0}^{n−1} (−1)^{n−1−r} c_r f^{n−1−r}`.
    Derived from the identity above; the sign bookkeeping is mine, so test it.

### Degenerate metric caveat (important for PGA)

- The outermorphism, determinant and characteristic coefficients need only ∧ and blade
  coefficients, so they are metric-independent and work in R(3,0,1).
- The **adjoint and the `X I` inverse formula need a non-degenerate inner product.** `I⁻¹` does not
  exist in PGA (`I² = 0`).
- For PGA, use the metric-free versions:
  - adjoint with respect to the dual pairing (the coefficient transpose in an orthonormal basis), or
  - the inverse `f⁻¹(X) = ⋆⁻¹ f̄_⋆(⋆X) / det f`, with the complement `⋆` in place of `·I`.
  - This is my derivation; it is standard exterior-algebra (Laplace/cofactor) theory, but it is
    not in H&S.

**What we take.**
- Types: `Outermorphism<N>` (a vector matrix plus lazily built compound blocks) and
  `Extensor<GradeIn, GradeOut>`.
- Operations: `det` via `f(I)`, `adjoint`, `inverse` (complement form for degenerate metrics), and
  the characteristic coefficients `c_r` by summing principal minors.
- Tests: Cayley–Hamilton as a unit-test oracle.

---

## 2. Roelfs & De Keninck, "Graded Symmetry Groups: Plane and Simple"

**Citation.** M. Roelfs, S. De Keninck, *Graded Symmetry Groups: Plane and Simple*, Adv. Appl.
Clifford Algebras 33, 30 (2023). arXiv:2107.03771. DOI 10.1007/s00006-023-01269-9.
Read from the arXiv PDF.

### Summary

- **Theorem 1 (invariant decomposition).** A product of k reflections factors into ⌈k/2⌉ commuting
  factors, each a product of at most two reflections. It generalizes Cartan–Dieudonné and contains
  Mozzi–Chasles as the 3D Euclidean case.
- Any bivector in dimension n splits into at most `k = ⌊n/2⌋` commuting, orthogonal 2-blades:
  - `B = b1 + … + bk`
  - `b_i b_j = b_i ∧ b_j` for i≠j, so `b_i·b_j = b_i×b_j = 0`
  - `b_i² = λ_i`, which is a scalar and may be complex, e.g. in R(2,2)

### Formulas [VERBATIM from §6, eqs. 10–17]

- **Wedge powers.** `W_m := (1/m!) B^{∧m} = Σ_{i1<…<im} b_{i1}…b_{im}`, with `W_0 = 1` and
  `W_1 = B`. (eq. 12)
- **Eigenvalues λ_i** are the roots of `0 = Σ_{m=0}^{k} ⟨W_m²⟩_0 (−λ)^{k−m}`. (eq. 14)
  This is `∏(b_j² − λ)`, a polynomial of degree k in λ.
- **Blades (eq. 13)**, with `r = ⌊k/2⌋` and `b_i = N_i D_i⁻¹`, valid for simple roots:
  - k even: `N_i = λ_i^r W_0 + λ_i^{r−1} W_2 + … + W_k` and `D_i = λ_i^{r−1} W_1 + λ_i^{r−2} W_3 + … + W_{k−1}`
  - k odd: `N_i = λ_i^r W_1 + λ_i^{r−1} W_3 + … + W_k` and `D_i = λ_i^r W_0 + λ_i^{r−1} W_2 + … + W_{k−1}`
  - Limit: `lim_{λ_i→0} b_i = W_k / W_{k−1}`, which gives the null bivector cheaply.
  - If exactly one λ_i is 0, you can use `b_i = B − Σ_{j≠i} b_j` instead. (eq. 22)
- **n < 6 (k = 2):**
  - `b_i = (λ_i + ½ B∧B) / B` (eq. 15)
  - `λ_i² − λ_i (B·B) + ¼ (B∧B)² = 0`, so `λ_{1,2} = ½ B·B ± ½ √((B·B)² − (B∧B)²)` (eqs. 16, 17)
  - Fails when `Δ = (B·B)² − (B∧B)² = 0`. Then `B³ = 4λB`, and any 2-blade with
    `b B = λ + ½ B∧B` works.
- **Closed form by dimension.**
  - n ≤ 5: quadratic
  - n = 6, 7: cubic (k = 3)
  - n = 8, 9: quartic (k = 4)
  - So closed form by radicals is possible for n ≤ 9; this is my remark. For n ≥ 10 you need
    numerical root finding, e.g. eigenvalues of a companion matrix or Jacobi.
- **Exponential (eq. 23).** `e^B = ∏_i [c(b_i) + s(b_i)]`, where for a simple b with `λ = b²`:
  - `c(b) = cosh√λ` and `s(b) = b · sinch√λ`, with `sinch z = sinh z / z` and `sinch 0 = 1`
  - λ<0: `cos θ + b sin θ/θ` with `θ = √(−λ)`
  - λ=0: `1 + b`
  - λ>0: `cosh + b sinh/…`
- **Logarithm of a simple rotor (eq. 5).**
  `Ln R = ⟨R⟩₂ · { arccosh⟨R⟩ / √(⟨R⟩₂²) if ⟨R⟩₂²>0 ;  1 if =0 ;  arccos⟨R⟩ / √(−⟨R⟩₂²) if <0 }`
  - The acos form loses accuracy near θ≈0 and θ≈π, so implement it as
    `atan2(√(−⟨R⟩₂²), ⟨R⟩)/√(−⟨R⟩₂²)` (my robustness remark).
  - For a general rotor, factorize and use `Ln R = Σ Ln R_i`.
- **Tangent decomposition** (eqs. 26–30). Let `t(b_i) = s(b_i)/c(b_i)`.
  - The t(b_i) follow from the grades of R alone. Eq. 29 has the same structure as eq. 13, with
    `⟨R⟩_{2m}` in place of `W_m`.
  - The λ_i are the roots of `Σ_m ⟨R⟩_{2m}² (−λ)^{k−m} = 0`, where the `⟨R⟩_{2m}²` factor is the
    scalar part.
  - Then `R_i = |c(b_i)| + sign(c(b_i)) s(b_i) ∝ 1 + t(b_i)` (eq. 32). The last factor is
    `R_k = R̃_1…R̃_{k−1} R` (eq. 33), which keeps the sign.
- **n < 6 example 8.1:**
  - `t(b_i) = (λ_i ⟨R⟩ + ⟨R⟩₄) / ⟨R⟩₂`
  - `λ_{1,2} = [⟨R⟩₂·⟨R⟩₂ ± √((⟨R⟩₂·⟨R⟩₂)² − 4⟨R⟩₄²)] / (2⟨R⟩₂)`. The printed denominator is
    garbled in the PDF text, so treat it as UNVERIFIED.
- **PGA3D (Ex. 6.2, Mozzi–Chasles).** `λ1 = B·B ≤ 0` and `λ2 = 0`.
  - `b2 = (B∧B)/(2B)` and `b1 = B − b2`
  - In coefficients, with `l = b12²+b31²+b23²` and `m = b01 b23 + b02 b31 + b03 b12`:
    `b2 = (m/l)(b23 e01 + b31 e02 + b12 e03)` (ideal, pure translation) and `b1 = B − b2`
    (a Euclidean line with `b1∧b1 = 0`) [VERIFIED-NUM].
  - Motor factorization (Ex. 9.1): `T = 1 + ⟨M⟩₄/⟨M⟩₂` and `R = M T̃ = T̃ M`. The two commute.
    [VERIFIED-NUM; bivector inverse via Study number, §2b]
- **Odd-versor factorization (eq. 35).** `P = r R = R r` with `r = ⟨P⟩₁` and `R = P/r`, when
  `⟨P⟩₁ ≠ 0`. Otherwise use `R = x·P` for any x with `x·P ≠ 0`.
- **§10 "Clifford representation."** Build the 2ⁿ×2ⁿ real matrices recursively with Kronecker
  products. The metric is chosen per basis vector:
  - `P=[[0,1],[1,0]]` for `+1`, `Q=[[0,−1],[1,0]]` for `−1`, `R=[[0,1],[0,0]]` for `0`
  - `I' = diag(1,−1)` in the prefix
  - Then permute so that the first column of the matrix of X equals X's coefficients.
  - The result: matrix×matrix and matrix×vector both implement the geometric product. The
    conjugation action is block-diagonal by grade, i.e. the outermorphism blocks.

**What we take.**
- A generic `invariant_decomposition(B)` via eqs. 12–15, with the n≤5 fast path and a numeric root
  path for larger n.
- A generic `exp` and `log` via the decomposition.
- The left-multiplication matrix `L_X` (the Clifford representation) as a universal fallback for
  inverse and solve in any `R(p,q,r)`, including degenerate ones.

## 2b. De Keninck & Roelfs, normalization, square roots, exp/log in GAs of dimension n < 6

**Citation.** S. De Keninck, M. Roelfs, *Normalization, Square Roots, and the Exponential and
Logarithmic Maps in Geometric Algebras of Less than 6D*, Math. Meth. Appl. Sci. (2022) 1–17.
arXiv:2206.07496. DOI 10.1002/mma.8639.
Code: https://enki.ws/ganja.js/examples/coffeeshop.html#NSELGA

**Study numbers.**
- `S = a + bI` with `(bI)² ∈ ℝ`.
- Conjugate `S̆ = a − bI` and norm `‖S‖ = √(a² − (bI)²)`.
- Inverse: `S⁻¹ = (a − bI)/(a² − (bI)²)` (eq. 19).
- Square root (eqs. 21–22): `√S = c + (b/(2c)) I` with `c = √(½(⟨S⟩ ± ‖S‖))`. Use `c₊` by default,
  and `c₋` for the square root of a negative real.
- Inverse square root (eq. 23): `S^{−1/2} = [2c/(4c⁴−(bI)²)]·... − ...`. The printed form is
  garbled. Use `S^{−1/2} = (√S)⁻¹` with eq. 19.

**Renormalization, i.e. polar decomposition `X = S R` with `S² = X X̃` (eq. 24).**
- `R = (X X̃)^{−1/2} X`.
- Always valid for even elements when n < 6, because `X X̃ = ⟨·⟩₀ + ⟨·⟩₄` is a Study number.
- It replaces Gram–Schmidt/SVD re-orthogonalization.

**Principal square root.** `√R = normalize(1 + R)` (eq. 11 and §4). This is valid for rotations,
translations and boosts alike.

**Bivector split for n < 6 (eqs. 33–35).**
- `b± = [B·B + B∧B ± √((B·B)² − (B∧B)²)] / (2B) = ½(1 ± B̆²/‖B²‖) B`, with `λ± = ½B² ± ½‖B²‖`.
- Complex solutions happen only in R(2,2) among n<6 algebras. In R(3,0,1), R(3,1), R4 and R(4,1)
  they are always real.

**Logarithm (eqs. 40–44).** `B = S·s(B)` with the Study number `S = α + βI`:
- `α = [sin θ₋ cos θ₊ θ₋ − sin θ₊ cos θ₋ θ₊] / ‖s²(B)‖`
- `βI = [sinc θ₊ cos θ₋ − sinc θ₋ cos θ₊] b₊b₋ / ‖s²(B)‖`
- `θ₋ = atan2(σ₋, ⟨R⟩)` with `σ± = √(s²(b±) c²(b∓))`. The printed placement of `√` is UNVERIFIED.

**PGA3D code, Listing 2 [VERBATIM; all four VERIFIED-NUM against series exp, log∘exp, R R̃ = 1, and (√M)² = M].**
Basis `X=[1,e01,e02,e03,e12,e31,e23,e0123]`, bivector `B=[e01,e02,e03,e12,e31,e23]`:
```js
function Normalize(X){ // 23 mul, 10 add, 1 sqrt, 1 div
 var A = 1/(X[0]*X[0] + X[4]*X[4] + X[5]*X[5] + X[6]*X[6])**0.5;
 var B = (X[7]*X[0] - (X[1]*X[6] + X[2]*X[5] + X[3]*X[4]))*A*A*A;
 return rotor(A*X[0], A*X[1]+B*X[6], A*X[2]+B*X[5], A*X[3]+B*X[4],
              A*X[4], A*X[5], A*X[6], A*X[7]-B*X[0]); }
function sqrt(R){ return Normalize(1 + R); }
function log(R){ // 14 mul, 5 add, 1 div, 1 acos, 1 sqrt
 if (R[0]==1) return bivector(R[1],R[2],R[3],0,0,0);
 var a = 1/(1 - R[0]*R[0]), b = acos(R[0])*sqrt(a), c = a*R[7]*(1 - R[0]*b);
 return bivector(c*R[6]+b*R[1], c*R[5]+b*R[2], c*R[4]+b*R[3], b*R[4], b*R[5], b*R[6]); }
function exp(B){ // 17 mul, 8 add, 2 div, 1 sincos, 1 sqrt
 var l = B[3]*B[3] + B[4]*B[4] + B[5]*B[5];
 if (l==0) return rotor(1, B[0], B[1], B[2], 0, 0, 0, 0);
 var m = B[0]*B[5] + B[1]*B[4] + B[2]*B[3], a = sqrt(l), c = cos(a), s = sin(a)/a, t = m/l*(c-s);
 return rotor(c, s*B[0]+t*B[5], s*B[1]+t*B[4], s*B[2]+t*B[3], s*B[3], s*B[4], s*B[5], m*s); }
```

In formula form:
- **Normalize:** `A = 1/√(s²+b12²+b31²+b23²)` and `B = A³ (s·p − (b01 b23 + b02 b31 + b03 b12))`,
  where p is the `e0123` coefficient. Then `M̂ = A·M + B·(b23 e01 + b31 e02 + b12 e03 − s e0123)`.
- **exp:** this is `e^B`, not `e^{B/2}`. A rotation by α about a normalized line ℓ is
  `e^{(α/2)ℓ} = cos(α/2) + sin(α/2) ℓ`, and a translator is `e^{(d/2)ℓ∞} = 1 + (d/2)ℓ∞`
  (cheat-sheet convention).

**Robustness fixes (mine).**
- `log`: exact `R[0]==1` tests fail for nearly-identity motors, and `acos` is ill-conditioned near
  ±1. Use `θ = atan2(√(b12²+b31²+b23²), s)` and a series for `θ/sin θ` and for
  `(1 − s·θ/sin θ)/sin²θ` when θ is small.
- `exp`: use Taylor series for `sin(a)/a` and `(c − s)/l` when l is small, since `(c−s)/l → −1/3`.
- Branchless SIMD version: evaluate both branches and select.

**Other listings.** R(3,1) (STA), R4 and R(4,1) (CGA) normalization, i.e. `sqrt` via
`Normalize(1+R)` [VERBATIM in the paper]. For R(3,1), with `S = ⟨XX̃⟩₀` and `T = ⟨XX̃⟩₄`
(the `e1234` coefficient, 2·(...)):
- `N = √(√(S²+T²)+S)` and `M = √2·N/(N⁴+T²)`
- `A = N²M` and `B = −T·M`
- `R = A X + B I X`, with the coefficient signs as in the listing.

**PGA2D (derived, VERIFIED by hand).** Every bivector `B = a e01 + b e20 + c e12` is simple with
`B² = −c²`.
- `exp B = cos c + (sin c / c) B`, which is `1 + B` when c = 0.
- For a normalized `M = s + a e01 + b e20 + c e12` with s²+c²=1:
  `log M = (atan2(|c|, s)/|c|)·⟨M⟩₂` (with `c→0`: `⟨M⟩₂`).

**What we take.**
- Exactly these PGA3D kernels: `normalize`, `sqrt`, `exp`, `log`, and `T = 1 + ⟨M⟩₄/⟨M⟩₂`.
- A `Study<T>` number type (`a + bI` with `I² ∈ {−1, 0, +1}`) providing inverse, sqrt and rsqrt.
- The generic polar-decomposition renormalizer for n ≤ 5.

---

## 3. PGA rigid-body dynamics: "May the Forque Be with You" and Gunn's SIGGRAPH 2019 notes

**Citation, correcting the brief: the authors are Dorst and De Keninck, not Gunn.** L. Dorst,
S. De Keninck, *May the Forque Be with You — Dynamics in PGA*, v2.6, 15 Aug 2023, 99 pp.
- PDF: https://bivector.net/PGAdyn.pdf
- Page: https://bivector.net/PGADYN.html
- Runnable code: https://enki.ws/ganja.js/examples/pga_dyn.html

**Citation.** C. G. Gunn, *Geometric Algebra for Computer Graphics*, course notes, SIGGRAPH '19 Courses
(the course is by Gunn and De Keninck). DOI 10.1145/3305366.3328099.
PDF: https://bivector.net/PROJECTIVE_GEOMETRIC_ALGEBRA.pdf. Rigid-body details:
C. Gunn, PhD thesis, TU Berlin 2011, ch. 9, DOI 10.14279/depositonce-3058.

### Dorst & De Keninck conventions [VERBATIM §2.1–2.5]

- **Rate bivector.** `B = −2 Ṁ M̃` (eq. 2.3). For constant rate, `M = exp(−Bt/2)`.
  - World frame: `Ṁ = −½ B_w M` (2.6).
  - Body frame: `Ṁ = −½ M B_b` (2.7), with `B_w = M B_b M̃`.
  - [VERIFIED-NUM: `d/dt exp(−tB/2) = −½ B M`]
- **Moving element.** `Ẋ_w = X_w × B_w` with commutator `P×Q = ½(PQ − QP)` (2.5, 2.8).
- **Momentum of a point mass.** `P = m X ∨ Ẋ`.
- **Inertia map.** `I_w[B] = Σ_i m_i X_i ∨ (X_i × B)` (2.13) and `I_b[B_b] = M̃ I_w[M B_b M̃] M` (2.14).
  It is additive over bodies, so no parallel-axis theorem is needed.
- **Principal (eigen) body frame (2.20–2.21).** With ij ∈ (23, 31, 12) paired with k = (1, 2, 3):
  - `I_b[B] = Σ_ij i_ij [B]_ij e_0k + Σ_k m [B]_0k e_ij`
  - `I_b⁻¹[B] = Σ_ij (1/i_ij) [B]_0k e_ij + Σ_k (1/m) [B]_ij e_0k`
  - This is a weighted Hodge dual; with unit mass and unit inertia it equals the dual `⋆`.
- **Equations of motion (Table 2.1, eqs. 2.28–2.29).**
  - Newton/Euler: `Ṗ = F`
  - Dynamics: `Ḃ_b = I_b⁻¹[ B_b × I_b[B_b] + F_b ]`, where F_b is the body-frame forque (wrench)
  - Kinematics: `Ṁ = −½ M B_b`
  - With uniform unit inertia: `Ḃ = ⋆⁻¹(B × ⋆B)`
  - ganja code: `[-0.5*M*B, (F - 0.5*(B.Dual*B - B*B.Dual)).UnDual]`
- **Forques.**
  - Gravity: `F_g = ⋆(M̃ (−9.81 e02) M)`, a line through the centre of mass.
  - Hooke spring: `F_H = k (P_b ∨ M̃ A_w M)`.
  - Damping: `F_d = ⋆(−α B)`.
- **Optimized uniform-inertia rate update** in coefficients (Dorst §2.5.8):
  `Ḃ = (b02 b12 − b03 b31) e01 + (−b01 b12 − b03 b23) e02 + (b01 b31 − b02 b23) e03`
  (6 mul, 3 add).
- **Contact impulse (§2.6).** At contact Q with normal line `N = n·Q`, the local velocity is
  `V± = Q ∨ (Q × B±)`. After contact, `B± ∓ j I±⁻¹[N]`, with j solved from the restitution ρ.

**My numerical verification in the bivector.net basis [VERIFIED-NUM].**
- Setup: `∨ = ⋆(⋆A∧⋆B)` with `⋆` = coefficient reversal, and a symmetric point cloud (8 unit masses
  at (±1,±2,±3)). The point-cloud definition gives
  ```
  I[e01] = m e23,  I[e02] = m e31,  I[e03] = m e12,
  I[e23] = Ixx e01, I[e31] = Iyy e02, I[e12] = Izz e03      (all + signs)
  ```
  with `Ixx = Σ m(y²+z²)`, and so on.
- As a matrix on `b = (b01,b02,b03,b12,b31,b23)`: **`I = J · D`**, where J is the anti-diagonal
  swap (the dual) and `D = diag(m, m, m, Izz, Iyy, Ixx)`.
  - The quadratic form `⟨B ∧ I[B]⟩_{e0123} = bᵀ D b` is symmetric positive-definite. Kinetic
    energy is `½ bᵀ D b`.
  - So "the inertia is a symmetric 6×6 map" means D, or more generally `G = J⁻¹ I`, is symmetric.
  - In a non-principal or off-centre frame, G is the full spatial-inertia matrix. Build it by
    evaluating the point-cloud formula on the six basis bivectors, or as `M̃ I[M·M̃] M`.
- **Momentum conservation check.** RK4 with `Ṁ = −½ M B` and `Ḃ = I⁻¹[B × I[B]]` conserves the world
  momentum `M I[B] M̃` to 1e−12. With the opposite sign `I⁻¹[I[B] × B]` it drifts by O(1). So the
  sign in the brief is correct.
- **Caveat about ganja's example.** The cuboid "C" bivector writes `12e13` in a basis containing
  e31, and the PDF text pairs the w/h/d labels ambiguously. Do not copy it; derive from `I = J·D`.

**Gunn's notes (§8.2) use a different normalization [VERBATIM].**
- `ġ = g Ω_c` (no −½), with `A` the "6D symmetric bilinear form" and `Π_c = J⁻¹(A(Ω_c))`.
- `Ω̇_c = A⁻¹(Φ_c + 2 A(Ω_c) × Ω_c)`, and kinetic energy `E = Ω_c ∧ Π_c`.
- Substituting `Ω = −½B` maps this onto Dorst's form, up to force scaling.
- Why PGA integrates well: 14 ODE dimensions against 12 true ones. Renormalizing the motor projects
  back onto the solution manifold, so no Lagrange multipliers are needed.

**What we take.**
- `RigidBody { M: Motor, B: Bivector }` with inertia as an `Extensor<2,2>` stored as `J·D`
  (principal) or `J·G` (general 6×6 symmetric G).
- `inertia_from_points`, `forque_gravity`, `hooke`, and `damping`.
- RK4 or a Lie-group integrator (`M ← M·exp(−½hB)`) followed by `normalize`.
- **Vibration modes:** `G ẍ + K x = 0` on bivector coordinates, a symmetric-definite generalized
  eigenproblem (§6).

---

## 4. Multivector inverses (Hitzer–Sangwine, Acus–Dargys, Shirokov)

**Citations.**
- E. Hitzer, S. J. Sangwine, *Multivector and multivector matrix inverses in real Clifford algebras*,
  Appl. Math. Comput. 311 (2017) 375–389. DOI 10.1016/j.amc.2017.05.027. There is no arXiv version;
  a tech report CES-534 is at https://repository.essex.ac.uk/17282/1/TechReport_CES-534.pdf, and I read it.
- E. Hitzer, S. J. Sangwine, *Construction of Multivector Inverse for Clifford Algebras over
  2m+1-Dimensional Vector Spaces from Multivector Inverse for Clifford Algebras over 2m-Dimensional
  Vector Spaces*, Adv. Appl. Clifford Algebras 29, 29 (2019). DOI 10.1007/s00006-019-0942-7.
  - The article number is UNVERIFIED, and I could not read the full text. It gives explicit results
    for n' = 2, 4, 6 lifted to n = 3, 5, 7.
- A. Acus, A. Dargys, *The Inverse of a Multivector: Beyond the Threshold p+q=5*, Adv. Appl. Clifford
  Algebras 28 (2018). arXiv:1712.05204. DOI 10.1007/s00006-018-0885-4. It gives n = 6 formulas as
  linear combinations of grade-negation products.
- D. S. Shirokov, *On computing the determinant, other characteristic polynomial coefficients, and
  inverse in Clifford algebras of arbitrary dimension*, Comput. Appl. Math. 40, 173 (2021).
  arXiv:2005.04015. DOI 10.1007/s40314-021-01536-0.
- Follow-up: K. Abdulkhaev, D. Shirokov, *Explicit Formula for Inverse and Determinant in Geometric
  Algebras over Odd-dimensional Vector Spaces* (n = 7), arXiv:2606.23259 (June 2026).

**Notation.**
- `x̂` = grade involution `(−1)^k`, `x̃` = reverse `(−1)^{k(k−1)/2}`, and `x̄` = Clifford conjugate
  `(−1)^{k(k+1)/2}`.
- `m_{j̄,k̄}(x)` negates grades j and k.

**Closed forms (Hitzer–Sangwine) [VERBATIM].** The denominators are real scalars. Zero means x is a
zero-divisor. Left and right inverses agree.

| n | right inverse `x⁻¹` |
|---|---|
| 1 | `x̂ / (x x̂)` |
| 2 | `x̄ / (x x̄)` |
| 3 | `x̄ x̂ x̃ / (x x̄ x̂ x̃)` |
| 4 | `x̄ m_{3̄4̄}(x x̄) / (x x̄ m_{3̄4̄}(x x̄))` |
| 5 | `x̄ x̂ x̃ m_{1̄4̄}(x x̄ x̂ x̃) / (x x̄ x̂ x̃ m_{1̄4̄}(x x̄ x̂ x̃))` |

- For n = 5, the three steps are `w = x x̄`, then `y = w ŵ̃`, which equals `x x̄ x̂ x̃`, then
  `z = y m_{1̄4̄}(y)`.
- For n > 5, Hitzer–Sangwine use `Cl(p+1,q+1) ≅ Mat(2, Cl(p,q))` and `Cl(p,q) ≅ Cl(p∓4, q±4)`
  recursively (block-matrix inverse).

**Shirokov, Theorem 4: Faddeev–LeVerrier in GA [VERBATIM].**
- Let `N := 2^{⌊(n+1)/2⌋}`.
- `U_(1) = U`, `C_(k) = (N/k) ⟨U_(k)⟩₀`, and `U_(k+1) = U (U_(k) − C_(k))`.
- `Det(U) = −C_(N)`, `Adj(U) = C_(N−1) − U_(N−1)`, and `U⁻¹ = Adj(U)/Det(U)`.
- The C_(k) are the characteristic polynomial coefficients of U's (complexified) matrix
  representation.
- Alternative: `S_(k) = (−1)^{k−1} N (k−1)! ⟨U^k⟩₀` with complete Bell polynomials.
- Cost: N−1 full geometric products. N is 2, 4 and 8 for n ∈ {1,2}, {3,4} and {5,6}.

**Degenerate metrics (my finding, VERIFIED-NUM).**
- Hitzer's n=1..5 formulas and Shirokov's recursion with N = 2^{⌊(n+1)/2⌋} (n = total dimension
  p+q+r) both give exact inverses (residual ~1e−15) on random multivectors in:
  - R(0,0,1), R(1,0,1), R(2,0,1), R(3,0,1) (PGA3D), R(3,1,1), R(3,0,2)
  - and the non-degenerate controls
- The papers state them only for Cl(p,q). Why they still hold:
  - Each derivation step uses only grade and commutation arguments. For example, in n=4 the r3 and
    r4 parts anticommute and square to scalars, and a PGA pseudoscalar squaring to 0 is still a
    scalar.
  - `Cl(p,q,r)` embeds as a subalgebra of `Cl(p+r, q+r)` via `e0 ↦ e₊+e₋`, the scalar part is
    preserved, and inverses are polynomials in x (Cayley–Hamilton).
- Zero-divisors (e.g. pure ideal elements) give denominator 0; test against a relative ε.

**What we take.**
- `inverse()` dispatch by n: the Hitzer forms for n ≤ 5, generated at compile time as
  involution-product chains; Shirokov's Faddeev–LeVerrier as the generic path; the left-mult
  matrix LU as a numeric fallback.
- Also expose `det_clifford(x)` (Shirokov's Det), which is useful as a condition estimate.
- For versors and motors, use the cheap special cases (`M⁻¹ = M̃` after normalization, and the
  Study-number inverse for bivectors: `B⁻¹ = B̃ (a + b e0123)⁻¹` with `(a+bε)⁻¹ = 1/a − bε/a²`).

---

## 5. "Look, Ma, No Matrices!" and Klein: operation counts

**Citation, correcting the brief's 2019/GAME2020 date.** S. De Keninck, *Look, Ma, No Matrices!*,
ACM SIGGRAPH 2024 Talks. DOI 10.1145/3641233.3665801.
Full article: https://enkimute.github.io/LookMaNoMatrices/

- It builds a glTF PBR renderer with PGA motors only.
- Its layout is `motor = [s,e23,e31,e12 | e01,e02,e03,e0123]` and
  `line = [[e23,e31,e12],[e01,e02,e03]]`. This is a different order from bivector.net.

**Operation counts [VERBATIM].**

| operation | mul | add | vs matrix |
|---|---|---|---|
| motor∘motor (`gp_mm`) | 48 | 40 | 4×4·4×4: 64 mul, 48 add |
| rotor∘rotor `gp_rr` | 16 | 12 | |
| translator∘translator `gp_tt` | 0 | 3 | |
| `gp_rt`/`gp_tr` | 12 | 8 | |
| `gp_rm`/`gp_mr` | 32 | 24 | |
| `gp_tm`/`gp_mt` | 12 | 12 | |
| point sandwich, naive `M p M̃` | 33 | 29 | mat4·vec4: 16 mul, 12 add |
| point sandwich, optimized | **21** | **18** | best known for dual quaternions |
| direction sandwich | 18 | 12 | |
| basis-axis sandwich (output scaled ½) | 6 | 4 | |
| `normalize_m` | 21 | 5 | +1 sqrt, 1 div |
| `sqrt_m` = normalize(1+M) | 21 | 6 | |
| `log_m` | 14 | 5 | +1 div, acos, sqrt |
| `exp_b` | 17 | 8 | +2 div, sqrt, sin, cos |
| full vertex (pos + tangent frame) | 47 | 38 | matrix+normal+tangent: 48 mul, 36 add; 25% fewer floats per vertex |

**The trick for the point sandwich** is to evaluate `p' = M p M̃ + p·(1 − M M̃)`. The second term is
zero for normalized M, but it cancels terms symbolically. GLSL, with M = (a0 = [s,e23,e31,e12],
a1 = [e01,e02,e03,e0123]) and point b = (e032,e013,e021) with e123 = 1:
```glsl
direction t = cross(b, a[0].yzw) - a[1].xyz;
return (a[0].x * t + cross(t, a[0].yzw) - a[0].yzw * a[1].w) * 2. + b;   // 21 mul 18 add
```

**Warning [VERIFIED-NUM].** The page's `exp_b` and `log_m` pair the ideal part with a `.zyx`/`.wzy`
swizzle, i.e. `e01↔e12`. As transcribed, that gives a wrong exp (error 0.2 against the series).
- The same page's `normalize_m` uses the correct `e01↔e23` pairing.
- Use the 2022 paper's pairing (§2b). Possibly a typo on the page, or my transcription of its
  layout; either way, cover it with a unit test.

**When matrices win.**
- LMNM concludes that the best renderer "is most likely a hybrid".
- Motor→3×4 matrix conversion costs roughly 30 mul (my estimate: quaternion→rotation plus the
  translation `t = 2(...)`; UNVERIFIED).
- A 3×4 affine transform costs 9 mul and 9 add per point, against 21/18 for the motor sandwich.
- So for more than ~3 points per motor, convert once and apply the matrix. Examples: batched mesh
  vertices, or an outermorphism applied to many vectors.
- For composition, interpolation (log/exp), normalization and a small number of points, stay with
  motors.
- **Our design:** `Motor::to_outermorphism()` returns the 4×4 (or 3×4) vector block, and a batch API
  picks the path from the batch size.

**Klein (C++/SSE PGA3D).** J. Ong, *Klein*, https://github.com/jeremyong/klein (MIT, 2020). Read
from a clone.
- **Layout.** The multivector is split into four `__m128` partitions:
  - `p0=(e0,e1,e2,e3)`
  - `p1=(1,e23,e31,e12)`
  - `p2=(e0123,e01,e02,e03)`
  - `p3=(e123,e032,e013,e021)`
  - Sandwiches are hand-expanded per partition pair (`sw00`, `sw10`, `sw20`, `sw30`, `sw012`,
    `sw312`, …). They are "NOT implemented in terms of two geometric products", and they exploit
    normalization.
- **llvm-mca numbers** from `docs/perf.md`, per call, 100 iterations:
  - rotor∘rotor: Klein 24 instructions (block RThroughput 8.0); RTM quaternion 23 (7.0); GLM 57 (14.0)
  - motor applied to point: Klein 59 instructions (RThroughput 13.5); GLM `mat3x4_cast(dualquat)*vec4`
    141 (38.5)
- **Lessons for Rust:**
  - Keep partition-aligned layouts, `[1,e23,e31,e12]` and `[e0123,e01,e02,e03]`, so the rotor part
    is one SIMD lane.
  - Use shuffles plus fused multiply-add (FMA) with sign masks via XOR.
  - Provide specialized types (rotor, translator, motor, point, line, plane) so zero coefficients
    are known at compile time. That is exactly the benefit of LMNM's `gp_rt`-style specializations.

---

## 6. Small-matrix symmetric eigensolvers and SVD (batched, branchless)

### 6.1 McAdams et al. 2011: 3×3 SVD with minimal branching

**Citation.** A. McAdams, A. Selle, R. Tamstorf, J. Teran, E. Sifakis, *Computing the Singular Value
Decomposition of 3×3 matrices with minimal branching and elementary floating point operations*,
Tech. Rep. #1690, Univ. of Wisconsin–Madison, May 2011.
https://pages.cs.wisc.edu/~sifakis/papers/SVD_TR1690.pdf

**Algorithm.**
1. Form `S = AᵀA`.
2. Run cyclic Jacobi on S with quaternion-accumulated approximate Givens rotations to get V.
   Typically 4 sweeps; the paper suggests 10–15 iterations.
3. Compute `B = AV`.
4. Sort the columns by norm (with sign flips to keep det V = +1).
5. Take a Givens QR of B, which gives `U` and `Σ = R` (diagonal).

**Approximate Givens (Alg. 2) [VERBATIM].** Constants `γ = 3+2√2`, `c* = cos(π/8)`, `s* = sin(π/8)`.
```
ch = 2(a11 − a22); sh = a12;  b = (γ sh² < ch²)
ω = rsqrt(ch² + sh²);  ch = b ? ω ch : c*;  sh = b ? ω sh : s*
return quaternion (ch, 0, 0, sh)      // rotation in (1,2) plane; permute for other pairs
```
- The unscaled rotation matrix is `[[ch²−sh², −2sh ch, 0],[2sh ch, ch²−sh², 0],[0,0,ch²+sh²]]`.
  Quaternions can be multiplied unnormalized; normalize once at the end.
- **Sort (Alg. 3):** a `CondNegSwap(c, X, Y)` bubble sort on `ρ_i = ‖b_i‖²`, with selects only.
- **QR Givens (Alg. 4):**
  - `ρ = √(a1²+a2²)`, `sh = ρ>ε ? a2 : 0`, and `ch = |a1| + max(ρ, ε)`
  - if `a1<0`, swap(sh, ch)
  - normalize with rsqrt
- **Accuracy (4 sweeps, 16.7M random unit-Frobenius matrices):** mean max off-diagonal 3e−6 and
  worst case 4e−3. Use one Newton step on `rsqrt`. So 4 sweeps are for graphics; use more sweeps in
  double precision for physics.

### 6.2 Jacobi theory and accuracy

- **Two-sided cyclic Jacobi (Golub & Van Loan, *Matrix Computations*, 4th ed., §8.5).**
  - For the pair (p,q): `θ = (a_qq − a_pp)/(2 a_pq)`, `t = sign(θ)/(|θ| + √(θ²+1))`,
    `c = 1/√(1+t²)` and `s = t c`.
  - Updates: `a_pp −= t a_pq`, `a_qq += t a_pq`, `a_pq = 0`, then rotate the other rows and columns.
  - Branchless: `sign` via `copysign`; if `a_pq == 0`, select `t = 0`.
  - A 6×6 sweep is 15 rotations. It converges quadratically; about 5–8 sweeps reach machine
    precision in double (UNVERIFIED rule of thumb).
- **One-sided (Hestenes) Jacobi SVD.** M. R. Hestenes, *Inversion of matrices by biorthogonalization
  and related results*, J. SIAM 6(1):51–90, 1958. DOI 10.1137/0106005.
  - Orthogonalize column pairs: `α=‖a_p‖²`, `β=‖a_q‖²`, `γ=a_p·a_q`, `ζ=(β−α)/(2γ)`,
    `t = sign(ζ)/(|ζ|+√(1+ζ²))`.
  - Rotate columns (and V). The singular values are the final column norms.
  - It never forms AᵀA and is highly parallel.
- **Accuracy.** J. Demmel, K. Veselić, *Jacobi's method is more accurate than QR*, SIAM J. Matrix
  Anal. Appl. 13(4):1204–1245, 1992. DOI 10.1137/0613074.
  - For symmetric positive-definite `H = D A D` (D diagonal), Jacobi computes the eigenvalues with
    relative error governed by κ(A), not κ(H). QR-based methods only guarantee `O(ε‖H‖)` absolute
    error.
  - Stopping criterion: `|a_pq| ≤ tol · √(a_pp a_qq)`.
  - Relevant to mass-scaled problems: inertia ranges over orders of magnitude, but relative accuracy
    survives. Faster, equally accurate variant: Z. Drmač, K. Veselić, *New fast and accurate Jacobi
    SVD algorithm I/II*, SIMAX 29(4), 2008 (citation from memory, UNVERIFIED).
- **Batched on GPUs.**
  - W. Boukaram, G. Turkiyyah, H. Ltaief, D. Keyes, *Batched QR and SVD algorithms on GPUs with
    applications in hierarchical matrix compression*, Parallel Computing 74:19–33 (2018).
    arXiv:1707.05141. One-sided Jacobi for batches.
  - NVIDIA cuSOLVER `cusolverDn<t>syevjBatched`: Jacobi, efficient up to n = 32; the size limit is
    a performance knee, not a validity limit.
  - Our SIMD analogue: struct-of-arrays, with one matrix per lane of `f32x8`/`f64x4`, a fixed sweep
    count, and all branches as selects (`simd_select`/`blend`). Convergence is data-dependent, so
    either run a fixed number of sweeps or loop until `all(lane_converged)`.

### 6.3 Closed-form 3×3 symmetric eigenvalues

- **Smith (1961).** O. K. Smith, *Eigenvalues of a symmetric 3×3 matrix*, Commun. ACM 4(4):168, 1961.
  DOI 10.1145/355578.366316. It was motivated by principal moments of inertia. Standard form
  (UNVERIFIED against the original page; matches the textbook statement):
  ```
  p1 = a12²+a13²+a23²;  q = tr(A)/3
  p2 = (a11−q)²+(a22−q)²+(a33−q)² + 2 p1;   p = √(p2/6)
  Bm = (A − q I)/p;  r = det(Bm)/2;  φ = acos(clamp(r,−1,1))/3
  λ1 = q + 2p cos φ;  λ3 = q + 2p cos(φ + 2π/3);  λ2 = 3q − λ1 − λ3
  ```
- **Kopp (2008).** J. Kopp, *Efficient numerical diagonalization of hermitian 3×3 matrices*, Int. J.
  Mod. Phys. C 19 (2008) 523–548. arXiv:physics/0610206. Code: http://www.mpi-hd.mpg.de/~globes/3x3/.
  - **Cardano form [VERBATIM]:**
    - `c2 = −tr A`, `c1 = a11a22+a11a33+a22a33−|a12|²−|a13|²−|a23|²`,
      `c0 = a11|a23|²+a22|a13|²+a33|a12|²−a11a22a33−2Re(a13* a12 a23)`
    - `p = c2²−3c1` and `q = −27/2 c0 − c2³ + 9/2 c2 c1`
    - `φ = ⅓ atan2( √(27[¼c1²(p−c1) + c0(q + 27/4 c0)]), q )`. This avoids computing `p³−q²`
      directly.
    - `x1 = 2cos φ`, `x2 = −cos φ − √3 sin φ`, `x3 = −cos φ + √3 sin φ`, and `λ_i = (√p/3) x_i − c2/3`
  - Eigenvectors by cross products: `v_i = (A1 − λ_i e1) × (A2 − λ_i e2)`. For a degenerate λ, use
    v1 × a column of `(A − λ1 I)`. Take `v3 = v1 × v2`.
  - **Accuracy caveats [VERBATIM]:**
    - Eigenvalue error is `O(ε λ_max)`, so small eigenvalues lose relative accuracy. Example
      `[[1e40,1e19,1e19],[1e19,1e20,1e9],[1e19,1e9,1]]` gives `1e40, 5e19, −5e19`.
    - Cross-product eigenvectors can be completely wrong for nearly degenerate pairs.
    - The hybrid falls back to QL when `‖v_i‖² ≤ 2⁸ ε Λ²`, with `Λ = max(λ_max², λ_max)`.
  - Benchmarks: Jacobi is the most accurate but slowest; the analytic method is more than 2× faster.
- **Newer: stable closed form.** M. Habera, A. Zilian, *Numerically stable evaluation of closed-form
  expressions for eigenvalues of 3×3 matrices*, Numer. Algorithms (2026). arXiv:2511.00292.
  - Uses the invariants I₁, J₂, J₃ and the discriminant Δ, with an accurate algorithm for J₂.
  - Forward-stable for well-conditioned eigenbases, and about 10× faster than LAPACK.
  - Better than raw Smith/Kopp if we want closed form. Companion: arXiv:2111.02117 (symbolic
    spectral decomposition).

### 6.4 Generalized symmetric-definite problem `A x = λ B x` (vibration modes)

For `M ẍ + K x = 0`, try `x = y e^{iωt}`; this gives `K y = ω² M y`, with M SPD (mass/inertia) and K
symmetric (stiffness). Cholesky reduction (Golub & Van Loan §8.7; LAPACK `sygst`/`sygv`):
```
M = L Lᵀ;   C = L⁻¹ K L⁻ᵀ  (symmetric);   C z = ω² z  (Jacobi);   y = L⁻ᵀ z
```
- The y are M-orthonormal: `yᵢᵀ M yⱼ = δᵢⱼ`.
- **Rigid body in the principal frame.** `G = diag(m,m,m,Izz,Iyy,Ixx)` in the (b01…b23) ordering of
  §3, so `L = diag(√g_i)` and `C_ij = K_ij/√(g_i g_j)`. That is just a symmetric diagonal scaling,
  and it is exactly the Demmel–Veselić "D A D" setting.
- For N bodies, M is block-diagonal of 6×6 blocks. Cholesky is blockwise.
- Caveat: if M is ill-conditioned, the explicit `L⁻¹ K L⁻ᵀ` loses accuracy. Alternatives: a
  simultaneous-diagonalization Jacobi (Veselić), or a QZ-type method. Rigid inertia is usually well
  enough conditioned after unit scaling.
- **GA reading.** K and G are both symmetric maps from twists (bivectors) to wrenches (dual
  bivectors). `G⁻¹K` is a (1,1)-extensor on bivectors, and the eigenbivectors are the modal screws.

**What we take.**
- `sym_eig3` with three paths: branchless Jacobi (default for accuracy), Smith/Habera closed form
  (fast), and the Kopp hybrid.
- `sym_eigN_jacobi<const N>` with a Demmel–Veselić stopping rule and SoA SIMD batching.
- `svd3` following McAdams, plus a one-sided Jacobi `svdN`.
- `gen_sym_eig(K, M)` via Cholesky.
- Principal-inertia extraction feeding `I = J·D`.

---

## 7. Other relevant GA code-generation and optimization work

- **GATL.** L. A. F. Fernandes, *Exploring Lazy Evaluation and Compile-Time Simplifications for
  Efficient Geometric Algebra Computations*. In: *Systems, Patterns and Data Engineering with
  Geometric Calculi*, SEMA SIMAI Springer Series 13, 2021. DOI 10.1007/978-3-030-74486-1_6.
  Code: https://github.com/laffernandes/gatl.
  - C++ template metaprogramming builds expression trees at compile time and simplifies them
    symbolically, doing Gaalop-like optimization without an external tool.
  - It works over arbitrary metrics, and the lazy evaluation avoids temporaries.
  - **Most relevant model** for a Rust design with const-generic or type-level grade/blade masks and
    expression templates via traits.
- **Gaalop.** D. Hildenbrand et al., *Gaalop — High Performance Parallel Computing based on Conformal
  Geometric Algebra* (2010); https://www.gaalop.de.
  - An external precompiler: symbolic GA to optimized C/CUDA/OpenCL, eliminating the zero
    coefficients and expressions that are never needed.
- **Garamon.** S. Breuils, V. Nozick, L. Fuchs, *Garamon: A Geometric Algebra Library Generator*,
  Adv. Appl. Clifford Algebras 29, 69 (2019). DOI 10.1007/s00006-019-0987-7.
  - Generates C++ libraries using a recursive prefix-tree product scheme. Scales to high dimension
    (for example n = 15) with numerically stable implementations.
  - Also: Breuils et al., *Computational aspects of GA products of two homogeneous multivectors*,
    AACA (2022), DOI 10.1007/s00006-022-01249-5.
- **TbGAL.** E. V. Sousa, L. A. F. Fernandes, *TbGAL: A Tensor-Based Library for Geometric Algebra*,
  Adv. Appl. Clifford Algebras 30, 27 (2020). DOI 10.1007/s00006-020-1053-1.
  - Represents GA products as tensor contractions (Eigen); efficient in high dimensions.
- **Versor.** P. Colapinto, *Versor: Spatial Computing with Conformal Geometric Algebra*, MSc thesis,
  UC Santa Barbara, 2011 (UNVERIFIED details); https://github.com/wolftype/versor.
  - C++11 templates that generate product types per blade set at compile time. There is no lazy
    fusion across expressions; GATL added that.
- **Gaigen 2.** D. Fontijne, *Gaigen 2: a geometric algebra implementation generator*, GPCE 2006.
  Specialized types, as in *Geometric Algebra for Computer Science*.
- **Outermorphisms.** A. H. Eid, *A Low-Memory Time-Efficient Implementation of Outermorphisms for
  Higher-Dimensional Geometric Algebras*, arXiv:1909.02408 (2019).
  - Orders of magnitude less memory than storing a full `2ⁿ×2ⁿ` matrix. Directly relevant to our
    `Outermorphism` storage: store the n×n vector matrix and generate grade blocks lazily from minors.
- **Extensor theory.** V. V. Fernández, A. M. Moya, W. A. Rodrigues Jr.:
  - *Extensors*, arXiv:math-ph/0212046
  - *Extensors in geometric algebras*, arXiv:math/0501558
  - *Multivector and extensor fields on smooth manifolds*, arXiv:math/0501559
  - *Duality products of multivectors and multiforms, and extensors*, arXiv:math-ph/0703054
  - Covers (p,q)-extensors, the extension (outermorphism) and generalization operators, the adjoint
    commuting with extension, and metric/gauge extensors ("golden formula": any Clifford algebra
    as a deformation of the Euclidean one).
  - **Naming and theory backbone for our API.** Titles are from arXiv search snippets; the exact
    titles are UNVERIFIED.
- **Kingdon.** M. Roelfs, *The Willing Kingdon Clifford Algebra Library*, arXiv:2503.10451 (2025);
  https://github.com/tBuLi/kingdon.
  - Symbolic optimization of GA operators with sparsity-aware, JIT-compiled codegen, generic over
    the numeric type.
  - Its cache-per-input-sparsity design suits a Rust proc-macro or build.rs generator.
- **Rust prior art.** `geometric_algebra` crate by Lichtso, https://github.com/Lichtso/geometric_algebra
  (MIT).
  - A DSL feeds a multiplication table, an AST, an optimizer, legalization, and finally Rust/GLSL
    emission.
  - SIMD on SSE2/NEON/wasm128; supports 1–16 generators; ships prebuilt PGA 1D–3D.
  - Study it before writing our own codegen.
- **ganja.js / GAmphetamine.** S. De Keninck: https://github.com/enkimute/ganja.js; code generator
  for any `R(p,q,r)`; GAmphetamine (symbolic simplification, used in PGAdyn §2.5.8).
- **Guided tour.** L. Dorst, S. De Keninck, *A Guided Tour to the Plane-Based Geometric Algebra PGA*
  (2022), https://bivector.net/PGA4CS.pdf. The best reference for the PGA3D element table and for
  the meaning of norms and duals.

---

## 8. Implementation checklist (priority order)

1. **Core:** const-generic `R(p,q,r)` using the bivector.net basis table (dual = coefficient reversal) and generated product tables; PGA3D-specialized types in Klein/LMNM SIMD layouts.
2. **Extensors:** an `Extensor<Gin,Gout>` matrix type; an `Outermorphism` built from an n×n matrix with lazy compound blocks; `det = f(I)/I`; adjoint; inverse (the `I`-form when non-degenerate, the complement form for PGA); characteristic coefficients `c_r`, tested against Cayley–Hamilton.
3. **PGA3D kernels (§2b, verified):** normalize, sqrt, exp, log, the Mozzi–Chasles split, `T = 1 + ⟨M⟩₄/⟨M⟩₂`, the bivector inverse, and the 21-mul point sandwich.
4. **Generic n ≤ 5:** a Study-number type, polar renormalization, the `b±` split with exp/log, and the Hitzer inverses (verified to work on degenerate metrics).
5. **Generic, any n:** wedge powers `W_m`, then the λ-polynomial (eq. 14), then the blades (eq. 13); Shirokov's Faddeev–LeVerrier inverse and det; left-multiplication-matrix LU as the fallback.
6. **Numerics (§6):** branchless Jacobi eigen/SVD, McAdams `svd3`, the Smith/Habera closed form with the Kopp fallback, and Cholesky-reduced generalized eigen.
7. **Dynamics (§3):** `I = J·D`, `Ḃ = I⁻¹[B×I[B]+F]`, `Ṁ = −½MB`, then RK4 or a Lie integrator plus normalize; modal analysis via §6.4.
8. **Tests:** the series-exp oracle, `log∘exp`, `RR̃ = 1`, `(√M)² = M`, Cayley–Hamilton, `x·x⁻¹ = 1` for every signature including degenerate ones, and momentum conservation of a free rigid body.
