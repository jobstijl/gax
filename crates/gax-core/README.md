# gax-core

The core of [gax](https://crates.io/crates/gax), geometric algebra with extensors: the traits
for slots, kinds and coefficients (`Slots`, `Kind`, `Coef`, `Real`), the `Extensor` trait that
values, maps and forms share, and the small-matrix math behind their methods (LU, Cholesky,
Jacobi eigenvalues, the SVD, least squares, general eigenvalues), branch free so that it runs per
SIMD lane. It also has the Study-number functions behind `exp`, `log` and `sqrt`, dual and
complex coefficients, and the runtime-dispatched batch kernels (feature `batch`).

Use it through `gax`, which re-exports what an application needs. Depend on `gax-core` directly
only to write code generic over gax's traits without the generated algebras.

Licensed under either of Apache License 2.0 or MIT, at your option.
