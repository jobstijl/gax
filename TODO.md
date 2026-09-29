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
- [x] Better line sandwich kernel: investigated, kept at 58 mul (the rotate-then-translate factorization breaks the degree-2 homogeneity of drift-tolerant kernels, and restoring it costs the saving; performance.md)
- [x] Early exit in the Jacobi solvers and cheaper pivoting (eigh and SVD now faster than nalgebra)
- [x] A cheaper map inverse: scalar LU pivots with branches (`Real::SCALAR`), shared pivot reciprocals (now 0.6x nalgebra)
- [ ] Emit fused code in a shape LLVM's SLP vectorizer handles well (fused single-value kernels can lose to generic code)
- [x] Division `a / b` (`DivBy`), embeddings between kinds (`From`, `Unit::widen`), `Motor::between` as `sqrt(b / a)` (ADR-029)
- [x] The generator runs its jobs in parallel: regenerating went from about 8 minutes to 34 s (ADR-030)
- [x] Compile time: one closure per product (CSTA 16.3 s to 13.9 s, release code unchanged), line tables in dev builds of gax (ADR-031)
- [x] Fuzzing: the solvers (inverse, det, solve, SVD, eigh) and the tracer (random programs against the generic code), as fuzz targets in CI and as fixed-sample tests
- [x] `Motor::between` on SIMD lanes and in WGSL (traced; the traced example's WGSL now links and validates with naga)
- [x] Tracer bugs found by the fuzzing: a reciprocal of zero made the relations inconsistent (a kernel returned the zero motor and verified); Gröbner bases and expansions without a work bound (a kernel never finished, one took 10 GB); the size cap now covers every strategy
- [x] CI records quick criterion runs on main (not gating)
- [x] Division by a `Unit` versor (its reverse), the division law in every symbolic law suite, `gax::select_lt` for any value or map, `ApproxEq`
- [x] Homomorphisms between the standard algebras as `From` (twelve, proved exactly), `Widen` for the ones that keep units, `from_images` for maps between any kinds, CGA `up`/`down`/`sphere` (ADR-032)

## Phase 2b: batch kernels
- [x] `batch` feature: runtime dispatch with fearless_simd 1.0 (SSE2, SSE4.2, AVX2, AVX-512, NEON, SIMD128, portable)
- [x] Lane types as `Coef`/`Real`; sound token handling through a private proof type
- [x] Vectorized f32 sin, cos, sinh, cosh, atan2, ln (bit-identical to the scalar `batch::math`)
- [x] Per-kind `Soa` storage (blocks of 16), AoS slice functions, uniform and per-element sandwiches
- [x] `Map`, `Kernel`, and batch forms of traced kernels (`Tracer::batch`), with broadcasting
- [x] Lane-for-lane tests on every level with remainders; benchmarks (default and native builds)
- [x] Built maps on batches: `BatchOf::of_slice` and `of_soa` for any one-slot extensor, as fast as a prepared motor

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

## Phase 4: laws
- [x] `reassoc` (ADR-003 amended); the matrix is `v >> X::slot()`, no `to_matrix` (ADR-013 amended)
- [x] `Sym` law harness and `law_suite!`, generated per algebra: multicategory, linearity, lifting, versor action, outermorphism, trace, fill, tiers
- [x] Equivariance modulo the versor ideal, with the scale-factor table (`docs/law-factors.md`) and negative controls per algebra
- [x] Conjugation laws (ADR-026: spelled with existing operators); the slow ones `#[ignore]`d and run in CI
- [x] Adjoint through a signed-permutation pairing matrix; `Pairing::solve` property test
- [x] Suites for all standard algebras, STAP/CSTA behind features, PGA4D through `algebra!` (test-only spec)
- [x] `docs/laws.md` (each law linked to its test or marked [doc]), ADR-025, guide section "Laws you can rely on"
- [x] Law-based rewrites in the tracer measured (performance.md: none pays off)
- [ ] Laws for versors with more than 8 coefficients (the free polynomials explode; needs a structured approach)

## Phase 5: numerics
- [x] `docs/numerics.md`: the guarantee per tier, drift, determinism, transcendental edges, solvers, what is documented rather than fixed
- [x] Drift-tolerant `Unit` kernels (homogeneous again; +6% multiplications), ADR-020 amended
- [x] `Unit::renormalize_fast`, `Unit::mul_renormalized`, `NewtonStep`; the Newton identity proved symbolically
- [x] `check-units` feature (asserts unit inputs to certified kernels); CI runs the tests with it
- [x] Error-bound tier tests (`Program::error_bound`); the bound as a tiebreak in `compile_best`
- [x] `deterministic` feature (no FMA, pure-Rust elementary functions), ADR-027; bit-equality across SIMD levels in CI
- [x] Edge-case property tests for exp, log and normalize; fixed the `log` series (second-order error), the cancellation at the series boundaries, and the branch near a full turn
- [x] CSTA `exp`: squarings chosen from the norm, Newton renormalization; measured across norms
- [x] Solver agreement: LU bit-identical scalar vs lanes on nearly singular maps; eigh/svd within gap-scaled bounds
- [ ] Re-measure the timings in performance.md after the drift-tolerant kernels and the longer series
- [ ] Solver agreement on the native SIMD lane types (the tests use the portable lanes)

## Phase 6: shaders (WGSL/WESL)
- [x] WGSL target for the straight-line-program printer (`fma`, reversed `select`, abstract-float constants), golden tests
- [x] Language-neutral kernel records (`gax_gen::kernel`); WGSL modules per algebra (`gax::wgsl`, feature `wgsl`), plain WGSL and valid WESL; `wesl_public()` for `wesl` 0.5 visibility
- [x] GPU layouts: `{Kind}Gpu` and `GpuMat` with `bytemuck::Pod` (feature `bytemuck`), generated `const` layout assertions, naga layout test
- [x] `Tracer::wgsl`: traced kernels as WGSL (`FUSED_WESL`); `Traceable` WGSL spellings (kinds, units, scalars, arrays, tuples)
- [x] Tests: naga validation, WESL imports with stripping, CPU evaluation within error bounds, wgpu execution (lavapipe in CI)
- [x] `examples/wgpu`: instanced motors, traced particle kernel, GPU/CPU cross-check, benchmarks; ADR-028, docs/shaders.md, guide section
- [ ] Kernels for kinds over 16 coefficients; the 5D Study functions and the scaling-and-squaring `exp` in WGSL
- [ ] WGSL modules for `algebra!` algebras (and WGSL forms of their traced kernels)
- [ ] `f16` variants
- [ ] Re-measure the CPU columns of the GPU benchmark on an idle machine

## Phase 7: warp, the game (examples/warp)
- [x] M0: Bevy as plumbing (dependency contract script, clippy disallowed types), own wgpu surface, fixed-step pure simulation, input boundary, CI job
- [x] M1: renderer core: SDF lines placed by motors in the vertex shader, HDR, bloom mip chain, AgX tonemapping, stroke font, GPU timestamps
- [x] M2: Plane vertical slice: ship, bullets, drifter/chaser/singularity (+ motes), warped-space lattice and particles as traced kernels on the GPU with a CPU twin, synthesized effects, drone/pulse music (plus soft percussion), offline music renders
- [x] M2 gate: the owner played it: fun (2026-09-28)
- [x] M3a: the full roster (evader, splitter, serpent, warden, carrier) with director formations
- [x] M3b: replays (quantized inputs, per-second state hashes), high scores with initials, menus, settings (shake, flashes, colour-blind schemes, volumes, full screen)
- [x] M3c: music layers from the multiplier (fragments, texture, octave doubling), effects on the beat
- [x] M4: Tunnel slice: a track of PGA3D screw motions, a PGA3D lattice on the wall, flight (reticle, barrel roll, throttle), a level camera with an FOV setting, drones, mines and turrets, shadow and lock/lead indicators
- [x] M4 gate: the owner played it: fun (2026-09-28)
- [x] M5: Tunnel complete (serpents, bonus gates, singularities that pinch the tunnel, replays and a table for the Tunnel, its own tempo), polish, attract mode, accessibility
- [x] M6: VERIFY.md complete, performance numbers (the Plane's and the Tunnel's CPU and GPU numbers, friction 1-18, the playtest verdict)
- [x] gax friction fixed from the game: ideal norms, sign-correct reflections, rotations between directions and look-at, the tracer's constant folding and two tracer bugs (VERIFY.md 4, 7, 8-11)
- [x] warp in gax throughout: lights as points, traced line shaders, the Tunnel and the Plane rewritten, phasor audio, a clippy ban on geometry by hand
- [x] The Tunnel renders through camera maps: one 4x4 per object and per wall ring (`Track::placement`), the frame interpolated once per object
- [x] More extensor use in warp: AgX's outset is `inset.inverse()` (folded when traced), OkLab's way out is the inverse maps, one `colour_map`; `across()` is the exact differential of the screen-to-section map
- [x] Tracer: solvers fold on constants (`all_lt` answers when nothing depends on the inputs), and arithmetic on named constants folds before the expansion limit makes it a node
- [x] gax friction: a tracer warning when a `Unit` argument's renormalization simplifies away (VERIFY.md 2); maps between algebras and `select_lt` (VERIFY.md 16, 17)

## Decisions for the project owner
- Publishing to crates.io (later).
