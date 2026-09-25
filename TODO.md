# TODO

Status of the work plan. `[x]` done, `[~]` in progress, `[ ]` open.

## Phase 0: research and design
- [x] Research log (`docs/research.md`) with appendices
- [x] Design record (`docs/design.md`)
- [x] Evidence for hypotheses 5 and 7 (unroll threshold, IEEE folding, generic straight-line codegen)

## Phase 1: vertical slice, PGA2D
- [x] Exact blade products for any integer metric, with an independent oracle
- [x] Exact rational polynomials
- [x] Layouts and operation tables (gp, wedge, vee, contractions, dot, scalar, commutators, involutions, complements)
- [ ] `gax-core`: `Coef`/`Real`, `Slots` (tuples, `HasCat` law), `Kind`, `Multivector`, slot-array helpers
- [ ] Algebra spec format and parser (shared by the macro and the regeneration tool)
- [ ] Emitter: kind structs, per-pair tier-1 kernels, unary ops, arithmetic
- [ ] PGA2D generated into `gax` and committed; regeneration tool plus CI diff check
- [ ] Binding: `of`, identity `slot()`, `at::<I>`, `trace`, `fill`
- [ ] `Sym` coefficient plus the tracing API
- [ ] Simplifier v1: CSE (passenger factoring plus pair extraction), unit conditions
- [ ] One tier-2 primitive: fused rotor/motor sandwich plus `to_matrix` for PGA2D
- [ ] Tests: oracle vs generated for every op and kind pair; proptests; the generic `S: Slots` function compiles
- [ ] One benchmark: PGA2D motor on points, direct against matrix against hand-written

## Phase 2: broaden
- [ ] PGA3D, VGA2D/3D, STA, CGA3D pre-generated
- [ ] Tier-2 primitives: norms, normalize, inverse, exp/log (PGA closed forms, invariant decomposition)
- [ ] Solvers: Cholesky/LDLᵀ, LU, Jacobi eigh, SVD, generalized eigh; closed forms by trait
- [ ] Extensor methods: solve, inverse, eigh, svd, det, outermorphism, adjoint
- [ ] Build-time tracing end to end (example crate using `build.rs`)
- [ ] SIMD coefficients (`wide`), SoA helpers
- [ ] `algebra!` proc macro plus fuzzing of the parser
- [ ] numga fixture cross-checks
- [ ] trybuild compile-fail tests
- [ ] Benchmarks vs glam/ultraviolet/nalgebra/geometric_algebra; `docs/performance.md` with asm findings
- [ ] Examples: scene graph and camera, rigid body inertia and modes, CGA/STA
- [ ] Guide, README, rustdoc everywhere
- [ ] CI: stable plus MSRV on Linux/macOS/Windows, clippy, deny, miri, regen check

## Decisions for the project owner
- Final crate name (working name `gax`).
- Publishing to crates.io.
