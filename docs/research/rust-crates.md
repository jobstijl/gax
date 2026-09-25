# Rust ecosystem survey for an extensor-based, coefficient-generic GA library

Research date: 2026-09-25. Toolchain used for measurements: rustc/cargo 1.98.1 (2026-09-01), x86_64 Linux.
Metadata comes from the crates.io API (`/api/v1/crates/<name>`). Instruction counts are my own, taken from `--emit asm`
of `#[no_mangle]` wrappers. The measurement crate is in `scratchpad/asmtest/` (`count.py` / `count2.py`).
"SSE2" means the default x86_64 target. "v3" means `-C target-cpu=x86-64-v3` (AVX2+FMA).

---

## 0. TL;DR

* **No Rust GA crate is both coefficient-generic and glam-fast.** Each one gives up at least one of three things:
  * generic scalars: Lichtso's `geometric_algebra` and `g3` are f32-only, `cga2d` is f64-only.
  * SIMD: `clifford`, `ga3`, `wedged`, `tclifford` and `garust` are scalar and generic over `num_traits::Float`
    or similar.
  * sparsity and inlining: `amari-core` stores a heap `Box<[f64]>`, and `garust` does a dense Cayley-table product
    (74 ns per point transform versus 1.5 ns in nalgebra, by its own benchmark).
* **The closest prior art for our pipeline:**
  * `vee` (qu1x) does symbolic reduction plus op counting. It factors the motor sandwich into a
    quadratic-in-motor 3x4 matrix form and eliminates the normalization conditions.
  * `clifford-codegen` does TOML specs, type discovery (the u·ũ scalar constraint), field constraints,
    and generated `Sandwich` impls. Its output is expanded triple products with no CSE.
  * Lichtso's codegen does SIMD grouping, but it has correctness and performance problems (§1.1).
* **glam's bar (single item, f32):**
  * `Quat * Vec3A` is about 27 ALU ops (v3) or 28 (SSE2), 32 or 39 instructions including load/store.
  * `Mat4 * Vec4` is 11 ALU ops (4 broadcast/shuffle, 4 mul, 3 add).
  * `Affine3A::transform_point3a` is 9 ALU ops.
* **Batched (SoA) is where GA can win clearly.** `ultraviolet::Rotor3x8::rotate_vec` takes 41 AVX ALU ops for 8
  rotations, about 5 per rotation. That is 5x better per item than glam's AoS quat.
* **`std::simd` is still nightly-only.** I checked this on 1.98.1: E0658, tracking issue #86656 is still open.
  The stable options are `wide` 1.7.1, `pulp` 0.22.3, `fearless_simd` 1.0.0 (released 2026-09-22), and raw
  `core::arch`.
* **Suggested MSRV target: 1.89.** That is what `wide`, `safe_arch`, `nalgebra` 0.35 and `fearless_simd` 1.0
  require.

---

## 1. Rust GA crates

### 1.1 `geometric_algebra` (Lichtso)

* **Links:** https://github.com/Lichtso/geometric_algebra · https://crates.io/crates/geometric_algebra
* **Status:** v0.3.0, MIT, 2018 edition, no MSRV declared. The last commit was 2023-09-20. The repo was pushed
  2024-05 but has no newer commits, so it is effectively dormant. About 12k downloads.
* **Approach: a codegen binary**, local clone at `ref/geometric_algebra/codegen`, about 2.3k LOC. It works in
  stages:
  1. A DSL string such as
     `"ppga3d:0,1,1,1;Scalar:1;...;Motor:1,e23,-e13,e12|e0123,e01,e02,e03;Point:e123,-e023,e013,-e012;..."`
     gives the generator squares and the named classes. `|` separates **groups of 4 (one SIMD register)**, and a
     sign prefix on a basis name permutes or negates the stored basis.
  2. It generates Cayley tables, compiles products to an AST, runs a small optimizer and a legalizer, and emits
     **Rust and GLSL**.
  3. The generated code is checked into the published crate. Nine prebuilt algebras ship: elliptic, parabolic and
     hyperbolic PGA in 1D, 2D and 3D.
* **Types and grades:**
  * Each class is a `union { groups: {g0: Simd32x4, ...}, elements: [f32; N] }`.
  * Class-by-class trait impls are generated for Add, Sub, Mul, Div, Into (projection), GeometricProduct,
    RegressiveProduct, OuterProduct, InnerProduct, Left/RightContraction, ScalarProduct, Transformation,
    Dual, Reversal, Automorphism, Conjugation, Inverse, Exp, Ln, Powi, Powf, and so on.
  * An impl is emitted only when the result lands in a registered class.
  * The result class of a product is the smallest registered class covering the output blades.
* **SIMD:**
  * A hand-written `Simd32x4` union covers SSE2, NEON and wasm simd128, with an array fallback.
  * Products are emitted as sums of `splat(a[i]) * swizzle!(b, ...) * [±1, ...]` sign-mask constants.
    Zero lanes are handled by multiplying by `[1,0,0,0]` masks, which wastes multiplies.
* **Findings from compiling the generated `ppga3d.rs` myself:**
  * **Bug: `swizzle!` calls `_mm_permute_ps` (an AVX intrinsic) under `cfg(target_feature="sse2")`.** On the
    default x86_64 target, every swizzle becomes an out-of-line `call core::arch::x86::avx::_mm_permute_ps`.
    `Rotor::transformation(Point)` compiles to 104 instructions with 9 calls, and on a non-AVX CPU it would
    SIGILL.
  * **`Transformation` is not fused.** It is literally `self.geometric_product(other).geometric_product(self.reversal()).into()`,
    that is, two full geometric products and then a projection.
  * With v3, `Motor.transformation(Point)` is 94 instructions (33 `vmulps`, 15 `vaddps`, 21 `vshufps`,
    9 broadcasts). That is about 3x glam's `Quat * Vec3A`, and it only does rotation plus translation.
  * **The generated impls carry no `#[inline]`.** Without LTO, a downstream `m.transformation(p)` is a `jmp` to
    the library function.
  * Everything is f32 only: `polynomial.rs` and the `Zero`/`One` impls are f32 only.
* **Where it stops:** there is no generic scalar type, no f64, no batching (AoS 4-lane only), no algebraic
  simplification beyond constant folding, and no use of constraints such as unit motors or the
  `e123 = 1` point normalization.
* **What we take:** the "group = one SIMD register" idea and a signed/permuted basis ordering chosen to make
  sign masks regular. Also the lessons: fuse sandwiches symbolically, put `#[inline]` on every generated
  kernel, never emit a stronger-ISA intrinsic than the cfg guarantees, and avoid multiply-by-zero masks.

### 1.2 `emilk/pga`

* **Links:** https://github.com/emilk/pga
* **Status:** not published (the crates.io `pga` name belongs to someone else, see below). Dual MIT/Apache.
  A single commit; the last push was 2023-07. The author calls it "an experiment".
* **Approach:** a generator of about 2.7k LOC, with a symbolic `Expr` tree and a `simplify.rs` that merges terms
  with integer coefficients. It follows Lengyel's conventions: `X Y Z W` naming, wedge and anti-wedge,
  anti-geometric product, left/right complements, reverse and anti-reverse.
  * Every blade is a **newtype** (`WX`, `YZW`, …), so assigning an x coefficient to a y field fails to compile.
  * The generator emits Rust structs per named type (Vec3, Vec4, Line3, Plane, Rotor3, Translator3, Motor3,
    Moment3) and **infers output types** by matching output blades against the named types, for example
    `Point3 ^ Point3 -> Line3`.
  * Products whose output does not match a named type are emitted as comments ("Omitted: … (unnamed type)").
  * Markdown multiplication tables are generated as documentation.
* **Where it stops:**
  * Motor3 only gets `geometric`, `dot` and `wedge` against a few types.
  * No sandwich or transform kernels are emitted: `sblade.rs` has a "used for sandwich products" helper, but no
    generated Motor→Point.
  * No SIMD, f32 only, several `todo!()` in type inference, and it depends on `derive_more` 0.99.
* **What we take:**
  * Type inference by blade-set matching, including a Venn-diagram notion where a value can belong to several
    named types.
  * Explicitly listing omitted products, which is good for coverage reporting.
  * Lengyel's naming and complement conventions as an option for the user-facing API.
  * The per-blade newtype idea is interesting for type safety but costly for ergonomics; we would not copy it.

### 1.3 Other GA crates (crates.io and GitHub, as of 2026-09)

| crate | ver / date | license | approach | where it stops | what we take |
|---|---|---|---|---|---|
| [`clifford`](https://github.com/DevonMorris/clifford) + [`clifford-codegen`](https://docs.rs/clifford-codegen) | 0.3.0 / 2026-02-04, MSRV 1.87 (2024 edition), about 16.6k downloads | MIT | Generic dense `Multivector` plus **15 "specialized" algebras** (complex, dual, quaternion, dual quaternion, Euclidean 2/3, PGA 2/3, CGA 2/3, Minkowski, elliptic, hyperbolic) generated from TOML (see below). About 38k LOC generated for PGA3. `T: Float`, which is `num_traits::Float` + approx + `'static`. Has nalgebra conversions, proptest and rerun. | **Scalar only.** The generated sandwich is fully expanded (e.g. `Motor→Point` is 12 triple-product terms per component, no CSE or matrix factoring). `num_traits::Float` rules out SIMD lanes and symbolic types. The claim of "competitive with quaternions" has no published numbers. | TOML spec workflow (`discover` → edit → `generate` → `verify`), `Unit<T>`/`Unitized<T>` wrapper types, and the "discover entities via the u·ũ scalar constraint" algorithm. |
| [`vee`](https://docs.rs/vee) (qu1x) | 0.4.1 / 2026-09-05, MSRV 1.94.1 | **MPL-2.0** | **Symbolic engine**: Multivector/Polynomial/Monomial over symbols with rational coefficients. It reduces expressions uniquely, counts mul/add, emits text/LaTeX/Rust/s-expr/egglog/DOT, and eliminates orthonormalization conditions (`.unit()`, `.pin()`). PGA for D ≤ 8 in all three metrics. | Code generator and research tool, not a runtime library. "Generate SIMD mode" and egglog CSE are on the roadmap only. The MPL license is file-level copyleft. | **The closest to our extensor pipeline.** For example `point << motor.unit()` gives `(1-2yy-2zz)X + 2(vz+xy)Y + ...`, which is exactly the motor-to-3x4-matrix factoring. Reuse the idea of symbolic normalization pins and op counting. |
| [`g3`](https://github.com/wrnrlr/g3) | 0.1.4 / 2023-02, last push 2023-05 | ISC | Klein-inspired PGA3 (plane-based). `Motor { p1: f32x4, p2: f32x4 }` using `std::simd`. Sandwich via `FnMut` call syntax `m(p)`. | **Nightly only** (`portable_simd`, `fn_traits`, `unboxed_closures`, `adt_const_params`). Dormant. | Klein's register layout: the motor as two f32x4 (`p1 = s,e23,e31,e12`; `p2 = e0123,e01,e02,e03`). |
| [`klein-ported-to-rust`](https://github.com/dyaso/klein-ported-to-rust) | not on crates.io, 2020-10 | MIT | Port of Klein's SSE intrinsics (via the C# port). | Self-described learning project, "untried". Dormant. | A reference for Klein's `sw012`/`sw312` sandwich kernels. |
| [`klein`](https://crates.io/crates/klein) (jamen) | 0.1.1 / 2021 | MIT | FFI bindings to C++ Klein. | Dead, and C++-dependent. | A possible benchmark target via C++ Klein itself. |
| [`pga`](https://github.com/Makogan/demiurge) (Makogan) | 0.1.0 / 2026-08-26 | Apache-2.0 | `no_std`, SPIR-V-friendly PGA3: a 16-component `MultiVec3D` plus Rotor, Translator and Motor versors. Sandwich is generic over `linear_isomorphic::InnerSpace`, so it works with nalgebra or its own points. | Brand new (36 downloads). Dense 16-float core. | Generic sandwich over the user's vector type (the operand-side genericity idea), and no_std/SPIR-V as a design constraint. |
| [`rotorlab-ga`](https://github.com/alejandro-soto-franco/rotorlab) | 0.0.4 / 2026-05, MSRV 1.85 | Apache-2.0 | `Algebra` trait const-generic over `(P,Q,R)`, a universal dense `Multivector<A>`, PGA3 Motor/Rotor/Translator newtypes, closed-form exp/log, `bytemuck::Pod`, miri in CI. | Dense 16-float, PGA3 only. | Const-generic signature trait shape. |
| [`garust`](https://github.com/westerngazoo/garust) | not on crates.io, active 2026-09 | Apache-2.0 | `Cl(P,Q,R)` plus generic scalar, compile-time Cayley table, branchless dense GP, sparse sandwich, optional `wide` SoA batch. | Self-reported benchmark: **74 ns per point transform versus 1.5 ns in nalgebra (about 50x); compose is 78 ns versus 2.9 ns**. It recommends `Motor::to_matrix()` for bulk work. | A cautionary data point: dense table-driven GA is 30–50x off. Also, "motor → 3x4 matrix, then bulk mat·vec" is a legitimate fast path that we should generate. |
| [`amari-core`](https://github.com/justinelliottcobb/Amari) | 0.24.1 / 2026-08, MSRV 1.75 | MIT OR Apache-2.0 | `Multivector<P,Q,R>` with `coefficients: Box<[f64]>`. Claims AVX2 and "cache-aligned". Part of a sprawling 20-crate ecosystem (tropical, GPU, holographic, …). | Heap allocation per multivector, f64 only. Largely AI-generated breadth. | Nothing for the hot path. |
| [`wedged`](https://github.com/jsmith628/wedged) | 0.1.1 / 2025-12 | Apache-2.0 | nalgebra-style `Blade<T,N,G>` / `Even` / `Multivector` with `Dim` plus `Alloc*` traits (typenum). Follows Dorst's GA4CS conventions. | Euclidean only, no SIMD. Its "homogeneous" module is still "planned". | The nalgebra-style allocator pattern for grade-sized storage `C(n,g)` when dimensions are generic. |
| [`tclifford`](https://github.com/grumyantsev/tclifford) | 0.1.0 / 2025-01 | MIT | `declare_algebra!(A, [+,+,-,-,0])`, **generic over any field type T**, dense and sparse representations, and an FFT matrix representation for high dimensions. | Early stage, runtime-dense, not performance-oriented for 3D. | Generic-over-field coefficients (symbolic and complex). The FFT/matrix representation is useful for high-dimensional inverses. |
| [`reefer`](https://github.com/kgullion/reefer) | 0.3.0 / 2025-11 | MIT | Proc-macro that generates optimized GA expressions at compile time (it has an mdbook describing the optimization pipeline). | Incomplete, by its own README. | Proc-macro plus optimization-pipeline design, an alternative to a build.rs or checked-in codegen. |
| [`ga2`](https://gitlab.com/porky11/ga2) / [`ga3`](https://gitlab.com/porky11/ga3) | 0.8.0 / 0.3.7, 2026 | MIT OR Apache-2.0 | Small generic `Vector<T>`, `Bivector<T>`, `Rotor<T>`, integrated with `vector-space` traits. | VGA only, scalar only. | Trait integration with ecosystem vector-space traits. |
| [`cga2d`](https://github.com/HactarCE/cga2d) | 3.0.0 / 2025-08, MSRV 1.88 | MIT OR Apache-2.0 | 2D CGA with static blade, rotor and flector types. | f64 only (the author explicitly avoids generics). | Nothing specific. |
| [`geonum`](https://github.com/mxfactorial/geonum) | 0.16.1 / 2026-07 | BSD-3-Clause | Non-standard representation: `{magnitude, angle{blade, t}}` "for O(1) in any dimension". | Not a Clifford-algebra implementation in the usual sense. | Nothing. |
| [`symtropy-math`](https://github.com/luminous-dynamics/symtropy) | 0.2.2 / 2026 | Apache-2.0 OR MIT | Const-generic `Point<D>`, `Bivector<D>` and `Rotor<D>` over `nalgebra::SVector`, plus GJK shapes. | f64 only, VGA only. | Nothing. |
| `deep_causality_multivector`, `quantique`, `geoit`, `cliff64` (AGPL), `meridian-gac-core` (MPL), `simply_2dpga`, `g_2_0_0` (GPL), `algebraic-gen`, `pga2d`, `galgebra`, `versor` | various | various | Niche, dense, or 2D-only libraries; several are AI-generated breadth. | | Nothing specific. |
| Not found or unrelated | | | `gafro` is a C++ robotics GA library, not Rust. `tensorgeo` and `pga3d` do not exist on crates.io. `blades` is a static-site generator. `klein-rs` does not exist. | | |

**Gap we fill:** we would be the only library that combines all of the following:
* symbolic factoring of versor operations into extensors, i.e. linear maps such as the motor to 3x4 matrix;
* generic coefficients (f32, f64, SIMD lanes, symbolic);
* SoA batching;
* inlined, fused kernels.

---

## 2. Rust math and SIMD crates

### 2.1 `glam` — the performance bar

* **Links:** https://github.com/bitshifter/glam-rs · https://crates.io/crates/glam
* **Status:** v0.33.10 (2026-09-24), **MSRV 1.68.2**, MIT OR Apache-2.0, 147M downloads.
* **Storage:**
  * `Vec3A`, `Vec4`, `Quat`, `Mat2`, `Mat3A`, `Mat4`, `Affine2` and `Affine3A` wrap a 128-bit register:
    `#[repr(transparent)] struct Quat(__m128)`, 16-byte aligned.
  * Per-architecture backends live in `src/f32/{sse2,neon,wasm,coresimd,scalar}/`.
  * `Mat4` has 4 `Vec4` columns. `Affine3A` is `Mat3A` (3 x `Vec3A`) plus a `Vec3A` translation, 64 bytes.
  * `Vec3` is 12 bytes and scalar. f64 types (`DQuat`, `DMat4`, …) are **scalar**.
  * The `core-simd` feature is nightly. There are `scalar-math` and `libm` features.
* **Quat × Vec3 formula** (sse2 `mul_vec3a`, and identical in the scalar path):
  `v' = v·(w² − b·b) + b·(2·(v·b)) + (b × v)·(2w)` where `b = (x, y, z)`.
  * `dot3_into_m128` uses mul + 2 shuffles + 2 `addss` + broadcast.
  * `cross` uses the 3-shuffle trick `(a.zxy*b - a*b.zxy).zxy`.
  * nalgebra uses the other common form, `t = 2(q×v); v' = v + w·t + q×t`.
* **Quat × Quat:** based on RTM's `quat_mul`: 4 splats, 3 lane shuffles, and sign-control constant vectors with
  `m128_mul_add`. That is `_mm_fmadd_ps` only if `target_feature="fma"` is enabled at compile time.

**Measured instruction counts** (glam 0.33.10, rustc 1.98.1, non-inlined `#[no_mangle]` wrappers; totals include
the arg loads, result store, `mov rax` and `ret`, about 4–6 instructions):

| op | SSE2 total | SSE2 ALU breakdown | v3 total | v3 ALU breakdown |
|---|---|---|---|---|
| `Quat * Vec3A` (`mul_vec3a`) | 39 | 7 mulps, 4 addps, 4 addss, 2 subps, 10 shufps, 1 movhlps (≈28) | 32 | 6 vmulps, 4 vaddps, 4 vaddss, 1 vsubps, 1 vfnmadd, 6 vshufps, 2 vmovshdup, 2 vbroadcastss, 1 vshufpd (≈27) |
| `Quat * Vec3` (12-byte Vec3) | 44 | same plus pack/unpack | 36 | same plus vinsertps/vextractps |
| `Quat * Quat` | 27 | 7 mulps, 3 addps, 7 shufps | 19 | 5 vmulps, 2 vfmadd, 1 vaddps, 4 vbroadcastss, 3 shuffles |
| `Mat4 * Vec4` | 18 | **4 shufps + 4 mulps + 3 addps = 11** | 14 | **4 vbroadcastss + 4 vmulps + 3 vaddps = 11** (no FMA: glam writes mul then add, and rustc never contracts) |
| `Mat4 * Mat4` | 70 | 16 shuf, 16 mul, 12 add | 54 | 16 bcast, 16 mul, 12 add |
| `Affine3A::transform_point3a` | 15 | 3 shuf, 3 mul, 3 add | 13 | 1 bcast + 2 shuf, 3 mul, 3 add |
| `Affine3A * Affine3A` | 54 | 12 shuf, 12 mul, 9 add | 46 | |
| `Mat4::inverse` | out-of-line (`jmp`) | | | |

* **The 3D PGA targets for us:** a motor applied to a point should approach `Affine3A::transform_point3a`, about
  9–12 ALU ops, via a cached 3x4 extensor. A one-shot motor→point should approach `Quat*Vec3A` plus a
  translation, about 27–30 ops.
* The historical mathbench numbers are in §3.
* **What we take:**
  * Match glam's ABI shapes (`repr(transparent)` over `__m128`, 16-byte alignment, a `Vec3A`-style padded
    type).
  * Take the explicit `m128_mul_add` pattern (FMA only under `cfg(target_feature="fma")`). Rust never fuses
    automatically, so our generated kernels must emit `mul_add` explicitly where fusing is wanted, and keep
    results reproducible otherwise.
  * Provide `From`/`Into` conversions to glam types for interop and benchmarks.

### 2.2 `ultraviolet` — SoA "wide" types

* **Links:** https://github.com/fu5ha/ultraviolet (formerly termhn) · https://crates.io/crates/ultraviolet
* **Status:** v0.10.0 (2025-04-26), MIT OR Apache-2.0 OR Zlib. The last push was 2025-06, so maintenance is
  slow. It depends on **`wide` 0.7**, far behind wide 1.7.
* **Design:**
  * Every type has a scalar version and wide versions (`Vec3x4`/`Vec3x8` over `f32x4`/`f32x8`, and
    `DVec3x2`/`DVec3x4` over f64), generated by `macro_rules!` over `(Type => (Mat, Vec, Bivec, lane_t))` lists.
  * Wide types are pure SoA: `Vec3x8 { x: f32x8, y: f32x8, z: f32x8 }`.
  * It provides `From<[Vec3; 8]>` (gather), `splat`, and `blend(mask, a, b)`.
  * No generics are used at all (a deliberate choice for compile time and readable errors).
  * It uses **rotors** instead of quaternions, with derivations in the repo's `derivations/` folder.
* **`Rotor3::rotate_vec`** is a two-step geometric product (`f = R v`, then `f R~`), written out in 4 + 3 lines.
  * Wide v3 measurement: `Rotor3x8::rotate_vec` is **53 instructions: 24 vmulps + 9 vaddps + 8 vsubps + loads
    and stores**, for **8 rotations**, about 5 ALU ops per rotation. rustc does not fuse these into FMA.
  * On SSE2, `f32x8` is emulated as two `f32x4`, which gives 161 instructions.
  * Scalar `Rotor3::rotate_vec` is 70 (SSE2) or 51 (v3) instructions, worse than glam.
* **What we take:**
  * The **SoA batch type shape** (`Motor3x8 { s: f32x8, e12: f32x8, … }`), AoS↔SoA conversion helpers, and
    mask/blend.
  * Evidence that the same GA formulas become very cheap when vectorized vertically, with no shuffles.
  * **Improvement over ultraviolet:** be generic over the lane type instead of stamping out types with macros
    (see ggmath/simba below), and emit explicit FMAs.

### 2.3 `nalgebra`

* **Links:** https://nalgebra.rs · https://crates.io/crates/nalgebra
* **Status:** v0.35.0 (2026-05-24), **MSRV 1.89**, 2024 edition, Apache-2.0, 91M downloads.
* **Storage:**
  * `Matrix<T, R, C, S>`, with `SMatrix<T, R, C> = Matrix<T, Const<R>, Const<C>, ArrayStorage<T, R, C>>`
    (column-major `[[T; R]; C]`). Dynamic dimensions use `Dyn`.
  * Dimension arithmetic still uses `DimName`/`DimAdd`/`DimMin` traits plus the `Allocator<R, C>` trait bound
    pattern, which is noisy but works on stable.
  * T is not required to be `Copy`: code uses `.clone()` everywhere, so heap-y scalars such as bignums and
    symbolic types work.
* **Fixed-size decompositions** (in `src/linalg/`; all bound on `T: ComplexField`, i.e. scalar and branching,
  **not** `SimdComplexField`):
  * **Cholesky** — standard unpivoted `O(n³/3)`. It returns `Option` on a non-positive pivot, and there is a
    `new_with_substitute` variant.
  * **SymmetricEigen** — Householder tridiagonalization, then implicit QR with Wilkinson shifts. Iterative, with
    `eps` and `max_niter`.
  * **SVD** — general bidiagonalization plus implicit-shift Golub–Kahan. It is **specialized via a `TypeId` check
    for 2x2** (closed form, from IEEE 486688) **and 3x3**: an eigen-decomposition of MᵀM plus a QR step,
    "McAdams et al." inspired, but still using the iterative symmetric eigen.
  * **Inverse** — closed-form specializations for n = 1..4 (`do_inverse4` via cofactors); everything else goes
    through LU.
  * Also available: LU, full-pivot LU, QR, column-pivot QR, Schur, Hessenberg, UDU, LBLᵀ, exp and pow.
* **Performance:** fine but not glam-level for 4x4. In mathbench (2021): `matrix4 inverse` 71.8 ns versus glam
  16.4 ns, `matrix4 determinant` 69 ns versus 6.2 ns. Mat4×Vec4 is on par (my v3 measurement: 11 ALU ops,
  identical to glam). The iterative eigen/SVD have data-dependent branches and loop counts, so they cannot run
  lane-parallel.
* **SIMD genericity:** geometry types (`UnitQuaternion<T>`, `Isometry3<T>`, `Matrix<T,…>` arithmetic) are generic
  over **`T: SimdRealField`**. That allows `UnitQuaternion<simba::WideF32x8>` (AoSoA, see the
  [rustsim blog](https://www.rustsim.org/blog/2020/03/23/simd-aosoa-in-nalgebra/)). Decompositions are not.
* **What we take:**
  * The **split between SIMD-generic "straight-line" algebra and scalar-only "branchy" numerics**
    (decompositions, normalization edge cases).
  * `Clone`-not-`Copy` bounds so symbolic coefficients work.
  * Closed-form small-n specializations selected by type.
  * For extensor spectral work such as the eigen-decomposition of a 3x3 or 4x4 symmetric outermorphism, call
    nalgebra's `SymmetricEigen`/`SVD` in scalar mode rather than writing our own.
  * Add a `convert-nalgebra` feature, as clifford does (with version-suffixed features).

### 2.4 `faer`

* **Links:** https://codeberg.org/sarah-quinones/faer · https://crates.io/crates/faer
* **Status:** v0.24.4 (2026-06-24), MSRV 1.84, MIT, 4.7M downloads.
* **Small matrices:** its own crate docs say it is focused on medium and large dense matrices, and that "its design
  is not well suited for applications that operate mostly on low dimensional vectors and matrices such as computer
  graphics or game development … `nalgebra` and `cgmath` may be better suited". Its types are dynamically sized
  (`Mat`/`MatRef`/`MatMut`) with runtime SIMD dispatch through `pulp`, so per-call overhead dominates at 4x4–6x6.
* **What we take:** do not use it for 4x4–6x6 extensor matrices. It is possibly relevant only for high-dimensional
  algebras (n ≥ 8, 256x256 extensor matrices), or as an optional backend for batched large solves.

### 2.5 `wide`

* **Links:** https://github.com/Lokathor/wide · https://crates.io/crates/wide
* **Status:** **v1.7.1 (2026-09-14), MSRV 1.89, 2024 edition, license `Zlib OR Apache-2.0 OR MIT`**, 75M downloads.
  `no_std`. Its dependencies are `bytemuck` and `safe_arch` 1.2 (x86 only).
* **Types:** `f32x4`, `f32x8`, `f32x16`, `f64x2`, `f64x4`, `f64x8`, and integer lanes up to 512 bits.
  * Implementation is chosen **at compile time** via `pick!{ if #[cfg(target_feature="avx")] … else … }`.
    For example, `f32x8` is `m256` under AVX, otherwise `{a: f32x4, b: f32x4}`; `repr(C, align(32))` either way.
  * There is **no runtime dispatch**.
  * Other targets: NEON on aarch64, simd128 on wasm; everything else falls back to arrays.
* **API:** operators; `splat`; `new([..])`; `mul_add`, `mul_sub`, `mul_neg_add`, `mul_neg_sub` (single rounding
  only with hardware FMA, otherwise two roundings); `sqrt`; `recip`; `recip_sqrt`; `sin_cos`; `exp`; `ln`; `powf`;
  `asin`/`acos`/`atan`; `reduce_add`; comparisons return masks (`CmpEq`/`CmpLt` traits); `select`/`blend`;
  `shuffle(u32xN)` and multi-input `[a, b].shuffle(..)`; and `bytemuck` casts.
* **Genericity:** there is **no public "SimdFloat" trait**. Methods are inherent, generated by `impl_simd_float!`.
  Only small traits exist (`Select`, `ShuffleExt`, `Cmp*`, `AlignTo`).
* **What we take:** use it as the default stable lane backend behind our own `Coeff`/`Lanes` traits (for example,
  impl `Coeff for wide::f32x8`). Enable FMA explicitly via `mul_add`. Batching users should compile with
  `-C target-cpu`/`target-feature`.

### 2.6 `std::simd` status

* **It is still nightly-only as of 2026-09.**
  * Verified: `use std::simd::f32x4` on stable 1.98.1 gives `E0658: use of unstable library feature portable_simd`.
  * The tracking issue [rust-lang/rust#86656](https://github.com/rust-lang/rust/issues/86656) is open, labeled
    `needs-rfc`, last updated 2026-08-12.
  * Some downstream projects run "stabilization canary" CI jobs.
* **What we take:** keep a `nightly-simd` feature at most. The stable path is `wide` or `core::arch`.

### 2.7 `pulp`, `fearless_simd` and friends (runtime dispatch)

* **[`pulp`](https://github.com/sarah-quinones/pulp)** — v0.22.3 (2026-06-20), MIT, no declared MSRV, 26M
  downloads.
  * **Runtime** feature detection: `Arch::new().dispatch(WithSimd impl)`.
  * The user writes `fn with_simd<S: Simd>(self, simd: S)`, and the `S` token provides `splat_f64s`, `mul_f64s`,
    `as_mut_simd_f64s` (head/tail split), and so on.
  * Default features are `x86-v3` and `relaxed-simd`; `x86-v4` adds AVX-512. It is faer's backend.
  * Good for slice kernels; its token-passing style is clumsy for small value types.
* **[`fearless_simd`](https://linebender.org/blog/fearless-simd-1-0/)** — v1.0.0 (2026-09-22), MSRV 1.89,
  Apache-2.0 OR MIT, from Linebender.
  * Uses target-feature 1.1 plus a `kernel!` macro, so intrinsics can be called without `unsafe`.
  * Offers both native-width and fixed-width vectors, and precise versus fast variants.
  * Its `#[simd]` macro does multiversioning, and it promises 3 years of security support.
* **[`multiversion`](https://crates.io/crates/multiversion)** — 0.9.0 (MSRV 1.86): function multiversioning for
  runtime CPU dispatch.
* **[`macerator`](https://crates.io/crates/macerator)** — 0.4.0 (MSRV 1.94): type- and target-generic SIMD.
* **[`safe_arch`](https://crates.io/crates/safe_arch)** — 1.2.0 (MSRV 1.89): safe `core::arch` wrappers behind
  `#[cfg]`.
* **What we take:**
  * Value-type kernels (single motor, SoA batch) should use compile-time `wide`/`core::arch`.
  * Slice-level batch APIs (`transform_points(&[P])`) can optionally offer runtime dispatch through
    `multiversion`, `pulp` or `fearless_simd` behind a feature.
  * The 1.89 MSRV lines up across wide, safe_arch, nalgebra and fearless_simd.

### 2.8 `simba` — the SIMD-generic scalar pattern

* **Links:** https://github.com/dimforge/simba · https://crates.io/crates/simba
* **Status:** v0.10.2 (2026-08-07), Apache-2.0, 2018 edition, 77M downloads (it comes in through nalgebra).
  Optional dependencies: `wide` 1.5, `fixed`, `decimal`, `cordic`, `rkyv`, `libm`. Also has a
  `portable_simd_impl` behind nightly.
* **Trait tower:**
  * `SimdValue { const LANES; type Element; type SimdBool: SimdBool; splat; extract; replace; select(self, cond, other); map_lanes… }`
  * `SimdPartialOrd` (`simd_gt` … returns `SimdBool`)
  * `SimdSigned`
  * `SimdComplexField: SubsetOf<Self> + SupersetOf<f32> + SupersetOf<f64> + Field + Clone + Neg + Send + Sync + Any + 'static` (with `simd_sqrt`, `simd_sin`, …)
  * `SimdRealField: SimdPartialOrd + SimdSigned + SimdComplexField<SimdRealField = Self>` (with `simd_atan2`,
    `simd_copysign`, constants)
* **Blanket impls:** `impl<T: RealField> SimdRealField for T`, so every scalar is a 1-lane SIMD value.
* **Lane wrappers:** newtypes such as `WideF32x8(pub wide::f32x8)`, needed because of the orphan rule.
* **Assessment:**
  * It proves that one code path can serve f32/f64 and lanes.
  * Downsides: it is heavy (Any + 'static + SupersetOf f32/f64 + full transcendental set). The newtype wrappers
    hurt ergonomics. It is `Clone`-based, so generic code is littered with `.clone()`.
  * It gives no way to express "exact/symbolic" coefficients that lack `sqrt` or comparisons.
* **What we take:** the core shape, with a lighter layering:
  1. `Ring`-like `Coeff` (`Clone + Add + Sub + Mul + Neg + zero/one + from_i32` for the ±1/±2 constants;
     optional `mul_add` with a default).
  2. `Field`/`Real` extension (`recip`, `sqrt`) for normalization and inverses.
  3. `Lanes` extension (`LANES`, `splat`, `select`, `Mask`) for SIMD.
  4. Scalar-only `Ordered` for branchy code.
  * Blanket-implement for f32/f64, and implement directly for `wide::f32x8` etc. (our crate owns the trait, so
    no newtype is needed). Consider also implementing for `simba::WideF32x8` for nalgebra interop.

### 2.9 Other notable math crates

| crate | ver | MSRV | license | note | what we take |
|---|---|---|---|---|---|
| [`ggmath`](https://github.com/Noam2Stein/ggmath) | 0.18.1 (2026-09-16) | 1.95 | MIT/Apache | `Vector<N, T, A: Alignment>` whose storage is `<T as VectorBackend<N, A>>::Inner`. Each element type picks its own register type through an associated type. Functions dispatch per N and alignment through a `const`-evaluated fn-pointer `specialize!` macro. SoA via `Vec3<wide::f32x4>`. Row-major, `vector * matrix`. | **Stable "specialization" for coefficient-generic storage.** The pattern `T: CoeffBackend<Layout>` with `type Storage` lets `Motor<f32>` be two `__m128` while `Motor<Sym>` is `[Sym; 8]`. The README's warning about SoA register pressure (a `Mat4<f32x4>` uses 16 registers) applies to us too. |
| [`vek`](https://github.com/yoanlcq/vek) | 0.17.2 (2025-09) | – | MIT/Apache | Generic `Vec3<T>` with `repr_c`/`repr_simd` variants. | Poor mathbench results; avoid its approach. |
| `cgmath` | 0.18.0 (2021) | – | Apache-2.0 | Dormant. | Only as a baseline. |
| [`glamour`](https://github.com/simonask/glamour), [`macaw`](https://github.com/EmbarkStudios/macaw) | 0.18 / 0.30 | – | MIT/Apache | Wrappers over glam (strong typing, extras). | Unit-typed wrappers. |
| [`mint`](https://github.com/kvark/mint) | 0.5.9 | – | MIT | Interoperability types. | Offer `mint` conversions. |
| [`lav`](https://github.com/qu1x/lav) | 0.8.4 | – | MPL-2.0 | Portable SIMD trait via GATs (same author as vee). | Another lane-trait design reference. |

---

## 3. Benchmark suites: mathbench-rs

* **Links:** https://github.com/bitshifter/mathbench-rs (glam's author). Repo pushed 2026-07-26.
  * The dependencies were refreshed then (glam 0.33.2, nalgebra 0.35.0, ultraviolet 0.10.0, cgmath 0.18.0,
    euclid 0.22, vek 0.17.1, simba 0.10, wide 1.5, criterion 0.8.2).
  * **The published tables are still from 2021** (glam 0.20.1, nalgebra 0.29, ultraviolet 0.8.1, rustc 1.56,
    i7-4710HQ). License: MIT/Apache.
* **Method:**
  * Criterion. The categories are "return self" (measures harness overhead), single ops, throughput ops (x100 or
    x256 with the same matrix), and workloads (Euler integration over 10k/80k vectors, ray–sphere intersection).
  * `bench_unop!`/`bench_binop!` pre-generate `1<<13` random inputs with `rand_pcg::Pcg64Mcg` via a
    `BenchValue::random_value` trait, and pre-fill the output vector.
  * Each iteration does `i = (i+1) & (size-1)`, then `get_unchecked(i)`, the op, and a store to the output.
    Inputs and outputs are wrapped in `black_box`.
  * Wide variants run batches of `ceil(size/width)` items per iteration. Glam is included as `glam_f32x1` doing the
    same total work.
  * The `unstable` feature adds `#[inline(never)]` in the ray–sphere test to defeat autovectorization.
  * Libraries are gated by cargo features (`bench!("nalgebra", group, …)` expands to `#[cfg(feature="nalgebra")]`).
  * Default `profile.bench`, meaning **no `target-cpu`**. Wide runs use `RUSTFLAGS='-C target-feature=+avx2'`.
  * `scripts/summary.py` and `tools/summarize` produce the tables. `tools/buildbench` measures clean build times
    (nightly `-Z timings`).
  * `tests/` cross-checks results between libraries.
* **Scalar results** (ns, 2021; lower is better): glam / nalgebra / ultraviolet / cgmath

| op | glam | nalgebra | ultraviolet | cgmath |
|---|---|---|---|---|
| rotation3 mul vector3 x1 | **6.50** | 7.58 | 8.97 | 7.68 |
| rotation3 mul vector3 x100 | **646** | 757 | 942 | 784 |
| rotation3 mul rotation3 | **3.66** | 7.48 | 7.64 | 7.53 |
| matrix4 mul vector4 x1 | **3.03** | 3.41 | 6.25 | 7.74 |
| matrix4 mul vector4 x100 | **614** | 627 | 801 | 968 |
| matrix4 mul matrix4 | **7.77** | 8.65 | 26.8 | 26.7 |
| matrix4 inverse | **16.4** | 71.8 | 41.2 | 47.1 |
| matrix4 determinant | **6.21** | 69.3 | 8.27 | 11.1 |
| transform point3 x1 | **2.96** | 6.10 | 6.84 | 10.7 |
| transform point3 x100 | **610** | 806 | 890 | 1270 |
| transform3 inverse | **11.0** | 71.4 | – | – |
| transform3 mul transform3 | **6.59** | 8.57 | – | – |
| vector3 normalize | **4.05** | 8.41 | 5.84 | 5.84 |
| ray-sphere x10000 (µs) | 56.2 | **15.3** | 50.9 | 55.7 |

* **Wide results** (AVX2, 2021): glam_f32x1 / uv_f32x4 / nalgebra_f32x4 / uv_f32x8
  * rotation3 mul vector3 x16: 130.7 / 36.8 / – / **26.1** ns, so SoA f32x8 is 5x faster than glam.
  * ray–sphere x80000: 568 / 155 / – / **61.5** µs, a 9x speedup.
  * matrix4 mul vector4 x256: 1.58 / **1.54** / – / 1.60 µs, a wash, because glam's AoS mat·vec is already
    efficient.
  * transform point3 x256: 1.57 / **1.40** / – / 1.43 µs.
  * euler 3d x80000: 141 / 97 / 96 / 104 µs.
* **Takeaways for us:**
  1. For **rotation-type** ops (quaternion or rotor sandwich), SoA gives large speedups (4–5x). For **matrix-type**
     ops (mat·vec) it gives almost nothing over glam AoS. So a motor→3x4 extensor applied AoS already matches
     glam, and SoA matters for the sandwich and compose paths.
  2. Our benchmark should reuse this harness: the same `BenchValue` random-input pattern, the `1<<13` ring
     buffer, the "return self" overhead baseline, and x1/x100 plus wide x16/x256 sizes. Add a `glam` column
     (Quat, Affine3A, Mat4) and an `ultraviolet` column (Rotor3x8) as baselines.
  3. Report both default-target and `-C target-cpu=x86-64-v3` results, since glam's FMA paths are
     cfg-gated.
  4. Complement Criterion timings with **instruction counts** from `--emit asm` or `cargo asm`, as done here,
     for regression tests. Also consider `iai-callgrind` for deterministic CI numbers.
* **Other suites:** [ga-benchmark](https://github.com/ga-developers/ga-benchmark) is C++ (Klein, Gaalop,
  Garamon, Versor, TbGAL, …) and follows the same methodology for GA-specific ops (sandwich, meet/join, inverse).
  It is worth mirroring its op list for our PGA/CGA comparisons. `garust` and `clifford` have ad-hoc Criterion
  benches with no cross-library table.

---

## 4. Design implications (summary)

1. **Coefficient trait tower** (see simba §2.8 and ggmath §2.9):
   * `Coeff` (ring ops, Clone, small integer constants, `mul_add` with a default of `a*b+c`)
   * then `Real` (sqrt, recip, used for normalization)
   * then `Lanes` (splat, select, mask; implemented for `wide::f32x4/f32x8/f64x4`)
   * plus scalar-only `Ordered`.

   Generated kernels are straight-line code over `Coeff` only, so they work for f32, f64, lanes and symbolic
   types.
2. **Storage specialization on stable** through an associated backend type, as ggmath does with
   `<T as Backend<Layout>>::Storage`. This allows glam-style `__m128` grouping for AoS f32 (Lichtso's "group of
   4") next to plain arrays for everything else. Keep a `repr(transparent)` path for glam-interop layouts.
3. **Fuse products symbolically before emitting:**
   * sandwich → extensor (matrix) form, e.g. vee's `(1-2yy-2zz)…`;
   * then CSE, as garust and clifford don't;
   * exploit unit/normalization constraints (vee's `.unit()`, `.pin()`; clifford's `Unitized<T>`).

   Emit explicit `mul_add`. Put `#[inline]` on everything.
4. **Two fast paths for "motor acts on many points":**
   * build the 3x4 extensor once, then apply it AoS (target: `Affine3A::transform_point3a`, 9–12 ALU ops);
   * or run the sandwich SoA over `f32x8` (target: at most ultraviolet's roughly 5 ALU ops per rotation, plus
     translation).
5. **Benchmarks:** fork mathbench's harness (§3) and add GA ops. Track asm instruction counts in CI.
6. **Delegate** spectral and decomposition work on 3x3/4x4/6x6 extensor matrices to nalgebra (scalar)
   behind a feature. Do not use faer at these sizes.
7. **MSRV 1.89.** Licenses of candidate dependencies are all permissive (wide Zlib/Apache/MIT; nalgebra and simba
   Apache-2.0; glam MIT/Apache). **Avoid copying code from MPL-2.0 (`vee`) or AGPL/GPL crates**; learn from their
   ideas only.
