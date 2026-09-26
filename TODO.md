# TODO

Status of the work plan. `[x]` done, `[~]` in progress, `[ ]` open.

## Phase 0: research and design
- [x] Research log (`docs/research.md`) with appendices
- [x] Design record (`docs/design.md`)
- [x] Evidence for hypotheses 5 and 7 (unroll threshold, IEEE folding, generic straight-line codegen)

## Phase 1: vertical slice (PGA2D)
- [x] Exact blade products for any integer metric, with an independent oracle
- [x] Exact rational polynomials; layouts and operation tables
- [x] `gax-core`: `Coef`/`Real`, tuple `Slots` with the `HasCat` law, `Kind`, `Extensor`, `SlotArr`
- [x] Spec format and parser; emitter; regeneration tool plus the `--check` mode for CI
- [x] Binding (`of`), identity `slot()`, the generic `S: Slots` function
- [x] Simplifier: CSE, kernel extraction, square-sum completion, Gröbner verification
- [x] Tier-2 fused sandwiches (value and matrix paths, `Unit` variants)
- [x] Oracle tests for every generated kernel; hand-written extensor tests
- [x] First benchmark against glam

## Phase 2: broaden
- [x] PGA3D, VGA2D, VGA3D, STA, CGA3D pre-generated; STAP and CSTA behind their own features; CGA2D
- [x] `Sym`, the tracer with bounded expansion, `examples/traced` (build.rs)
- [x] SIMD lanes (`wide`), SoA benchmark
- [x] Solvers (LU, Cholesky, Jacobi eigh, generalized eigh, SVD), property-tested, branch free
- [x] Map/form methods: inverse, det, solve, svd, trace, eigh_with, eigh, pairing solve
- [x] Value methods: norm, inverse, normalized, exp, log, sqrt (Study-number calculus)
- [x] `at::<I>`, `swap`, `fill` (type-level kind equality)
- [x] `algebra!` proc macro (STAP, CSTA tested)
- [x] exp/log for 5D algebras (CGA3D, STAP): Study functions with a general 4-vector direction
- [x] exp for 6D and up (CSTA): scaling and squaring in the product closure
- [x] exp and log of CSTA Poincaré motors (`Twist` to `Motor`): closed forms, since `Q²` is a scalar there
- [ ] log for the full 6D conformal group (CSTA `Bivector` to `Even`): the cubic invariant decomposition
- [x] Faster PGA3D exp/log: a real-trig path when the scalar part of `B²` is provably ≤ 0
- [x] Compact principal-inertia representation (`pga3d::PrincipalInertia`)
- [x] Prepared sparse sandwich maps; balanced summation trees
- [x] `mul_add` fusion in emitted programs (hardware FMA when available)
- [x] Multi-slot `trace_at::<I>`; outermorphism (wedge and vee, symbolic minors)
- [ ] Better line sandwich kernel (58 mul now; a rotate-then-translate factorization would save about 10)
- [x] Early exit in the Jacobi solvers and cheaper pivoting (eigh and SVD now faster than nalgebra)
- [x] A cheaper map inverse: scalar LU pivots with branches (`Real::SCALAR`), shared pivot reciprocals (now 0.6x nalgebra)
- [ ] Emit fused code in a shape LLVM's SLP vectorizer handles well (fused single-value kernels can lose to generic code)

## Phase 2b: batch kernels
- [x] `batch` feature: runtime dispatch with fearless_simd 1.0 (SSE2, SSE4.2, AVX2, AVX-512, NEON, SIMD128, portable)
- [x] Lane types as `Coef`/`Real`; sound token handling through a private proof type
- [x] Vectorized f32 sin, cos, sinh, cosh, atan2, ln (bit-identical to the scalar `batch::math`)
- [x] Per-kind `Soa` storage (blocks of 16), AoS slice functions, uniform and per-element sandwiches
- [x] `Map`, `Kernel`, and batch forms of traced kernels (`Tracer::batch`), with broadcasting
- [x] Lane-for-lane tests on every level with remainders; benchmarks (default and native builds)

## Phase 3: quality and documentation
- [x] trybuild compile-fail tests (slot mismatch, wrong kind bound, non-square inverse)
- [x] numga fixtures cross-check
- [x] Property tests of algebraic laws on the generated types
- [x] Fuzz the spec parser (cargo-fuzz)
- [x] clippy (pedantic) clean; cargo-deny config; miri on core
- [x] CI workflow: stable plus MSRV on Linux/macOS/Windows, regen check, tests, clippy, deny, miri, fuzz, the example game (green on GitHub Actions)
- [x] README, guide (doctested)
- [ ] Runnable rustdoc examples on more public items
- [x] Examples: scene graph and camera; rigid body inertia and vibration modes; CGA/STA; spacetime (STAP, CSTA); inverse kinematics; batch particles
- [x] A small windowed game (`examples/asteroids`, its own crate)
- [x] `docs/performance.md` (first version)

## Decisions for the project owner
- Publishing to crates.io (later).
