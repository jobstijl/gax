# Geometric calculus

Geometric calculus differentiates and integrates with the geometric product. Its derivative with
respect to a vector, `∇`, holds the divergence and the curl in one operator, and it differentiates
with respect to any multivector just as well. This page covers `gax::calculus`. The design is in
ADR-047 of the [design record](design.md), and the laws it satisfies are in [laws.md](laws.md),
section O.

## The derivative with respect to a multivector

A value `x` of a kind with blades `e_J` has coefficients `x^J`. The derivative of a function `F` with
respect to `x` is

```text
∂_X F = Σ_J e^J ∂_J F
```

where `∂_J` differentiates along `x^J` and `e^J` is the *reciprocal* blade: `e^J ∗ e_K = δ^J_K`,
with `∗` the scalar product. For a vector `x` this is the vector derivative `∇F`. Its two grade
parts are the divergence `∇·F` and the curl `∇∧F`:

```rust
use gax::calculus::{curl, divergence, nabla};
use gax::vga3d::{Bivector, Vector};
use gax::Coef;

// A rotation about e12, x ↦ x ⌋ e12. It is free of divergence, and its curl is twice the
// plane of rotation.
fn swirl<T: Coef>(x: Vector<(), T>) -> Vector<(), T> {
    x.lc(Bivector::new(T::zero(), T::zero(), T::one()))
}
let x = Vector::<(), f64>::new(0.3, -1.0, 2.0);
let (_, d) = nabla(swirl, x);
assert_eq!(divergence(swirl, x).1.s(), 0.0);
assert_eq!(curl(swirl, x).1, Bivector::new(0.0, 0.0, 2.0));
assert_eq!(d.grade::<0>().s(), 0.0);
assert_eq!(d.grade::<2>(), Bivector::new(0.0, 0.0, 2.0));
```

* `nabla(f, x)` returns `f(x)` and `∂_X f`. The derivative's kind is that of the geometric product
  of `x` and `f(x)`: the derivative of a vector field is a scalar plus a bivector.
* `divergence(f, x)` and `curl(f, x)` are its inner (`⌋`) and outer (`∧`) parts.
* `laplacian(f, x)` is `∂_X ∗ ∂_X f`, for a vector `x` the Laplacian `∇²f = ∇·∇f`, of the kind
  of `f`.

The functions differentiate by forward mode, as `jacobian` does
([guide](guide.md), section 15). `f` sees its argument with jet coefficients, so write it generic
over the coefficient type, as `swirl` is, or as a closure over the jet type. A function generic
over `Coef` also runs on the prime field `gax::fp::Fp`, where the derivatives of polynomials are
exact. A function that needs roots or transcendentals is written over `Real`.

## Spacetime: Maxwell's equations

In the spacetime algebra STA (`t² = 1`, `x² = y² = z² = −1`), `∇` is the derivative with respect
to an event, and the electromagnetic field is a bivector `F`. Maxwell's four equations are one,
`∇F = J`. A plane wave of any profile `φ` travelling along a null direction `k`, polarized along
`a ⟂ k`, is `F = (k ∧ a) φ(k · x)`, and it satisfies `∇F = 0` in vacuum:

```rust
use gax::calculus::{laplacian, nabla};
use gax::sta::{Scalar, Vector};

let k = Vector::<(), f64>::new(1.0, 1.0, 0.0, 0.0); // along x at the speed of light
let a = Vector::<(), f64>::new(0.0, 0.0, 1.0, 0.0); // polarized along y
let wave = |x: Vector<(), _>| {
    let (k, a) = (k.map_coefs(Into::into), a.map_coefs(Into::into));
    let phase: Scalar<(), _> = k.scalar_product(x);
    (k ^ a) * (phase * phase * phase) // the profile φ(s) = s³
};
let event = Vector::<(), f64>::new(0.5, 0.2, -0.1, 0.3);
let (f, maxwell) = nabla(wave, event);
assert!(f.c.iter().any(|&c| c != 0.0));
assert!(maxwell.c.iter().all(|&c| c == 0.0));

// The same profile satisfies the wave equation □φ = 0.
// Constants enter a function of jets of jets with `Coef::from_f64`.
use gax::Coef;
let (_, box_phi) = laplacian(|x: Vector<(), _>| {
    let phase: Scalar<(), _> = k.map_coefs(Coef::from_f64).scalar_product(x);
    phase * phase * phase
}, event);
assert_eq!(box_phi.s(), 0.0);
```

## Which kinds have a derivative

The reciprocal blades invert the Gram matrix `G_JK = e_J ∗ e_K` of a kind's blades. gax-gen
inverts it exactly, in rationals, for every kind, and the kinds where it is invertible implement
`gax::calculus::Reciprocal`. Off-diagonal metrics are covered: in CGA, `eo · e∞ = −1`, so the
reciprocal of `eo` is `−e∞`. So are the metrics of `algebra!` declarations.

| algebra | kinds with a reciprocal frame |
|---|---|
| VGA2D, VGA3D, STA | every kind |
| CGA2D, CGA3D, CSTA | every kind holding `eo` and `e∞` together, or neither |
| PGA2D, PGA3D | only the kinds without `e0`: `Scalar` and `Rotor` |
| STAP | only `Scalar` |

A kind that holds a null direction (PGA's `e0`, which squares to zero) has a singular Gram matrix:
no element of the kind pairs to one with `e0`, so a derivative cannot be taken back to an element.
Calling `nabla` on a PGA plane is a compile error that says so. There, the derivative is the form
that `gax::dual::gradient` returns (guide, section 15), which needs no metric.
