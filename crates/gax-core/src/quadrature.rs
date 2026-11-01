//! Quadrature on an interval: Gauss–Legendre rules (numpy's `leggauss`).
//!
//! The rules on simplices of any dimension are in [`simplex`](crate::simplex) (Grundmann and
//! Möller's, rational); these are the classical Gaussian rules on a segment, exact on
//! polynomials of degree `2n − 1` with `n` nodes, for integrals along a parameter (a confocal
//! family's widenings, a path's length, an average over a period).
//!
//! ```
//! use gax::quadrature::{gauss_legendre, gauss_legendre_on};
//! // Five nodes integrate x⁸ over [-1, 1] exactly: 2/9.
//! let (x, w) = gauss_legendre::<f64>(5);
//! let integral: f64 = x.iter().zip(&w).map(|(x, w)| w * x.powi(8)).sum();
//! assert!((integral - 2.0 / 9.0).abs() < 1e-15);
//! // On [0, 2], ∫ t³ dt = 4.
//! let (t, w) = gauss_legendre_on::<f64>(2, 0.0, 2.0);
//! assert!((t.iter().zip(&w).map(|(t, w)| w * t * t * t).sum::<f64>() - 4.0).abs() < 1e-14);
//! ```

extern crate std;

use crate::coef::Real;
use std::vec::Vec;

/// The `n`-point Gauss–Legendre rule on `[−1, 1]`: its nodes, the roots of the Legendre
/// polynomial `Pₙ`, ascending, and its weights `2 / ((1 − x²) Pₙ'(x)²)`. Exact on polynomials of
/// degree `2n − 1`; numpy's `leggauss(n)`.
///
/// Each root is found by Newton's method on the three-term recurrence from Tricomi's estimate
/// `cos(π (k − 1/4) / (n + 1/2))`, and the rule is made exactly symmetric: the nodes come in
/// pairs `±x` with equal weights (and `0` for odd `n`). Against the rule to 50 digits
/// (`tests/quadrature.rs`, `n` up to 100): nodes within `2 ε`, weights within `8 ε` (numpy's
/// weights are off by up to `18 ε`).
///
/// # Panics
/// If `n` is zero.
#[must_use]
pub fn gauss_legendre<T: Real>(n: usize) -> (Vec<T>, Vec<T>) {
    assert!(n > 0, "gauss_legendre: a rule of zero nodes");
    let one = T::one();
    let nf = T::from_f64(n as f64);
    // Pₙ(x) and Pₙ'(x), by the recurrence k Pₖ = (2k − 1) x Pₖ₋₁ − (k − 1) Pₖ₋₂.
    let legendre = |x: T| {
        let (mut p0, mut p1) = (one, x);
        for k in 2..=n {
            let k = T::from_f64(k as f64);
            let next = ((k + k - one) * x * p1 - (k - one) * p0) / k;
            (p0, p1) = (p1, next);
        }
        (p1, nf * (x * p1 - p0) / (x * x - one))
    };
    let half = n / 2;
    // The positive roots, largest first (k = 1 is nearest 1).
    let mut positive: Vec<(T, T)> = Vec::with_capacity(half + 1);
    for k in 1..=half {
        let angle = T::from_f64(core::f64::consts::PI * (k as f64 - 0.25) / (n as f64 + 0.5));
        let mut x = angle.cos();
        for _ in 0..100 {
            let (p, dp) = legendre(x);
            let step = p / dp;
            x = x - step;
            if !T::all_lt(T::epsilon(), step.abs()) {
                break;
            }
        }
        let (_, dp) = legendre(x);
        positive.push((x, (one + one) / ((one - x * x) * dp * dp)));
    }
    let mut nodes = Vec::with_capacity(n);
    let mut weights = Vec::with_capacity(n);
    for &(x, w) in &positive {
        nodes.push(-x);
        weights.push(w);
    }
    if n % 2 == 1 {
        let (_, dp) = legendre(T::zero());
        nodes.push(T::zero());
        weights.push((one + one) / (dp * dp));
    }
    for &(x, w) in positive.iter().rev() {
        nodes.push(x);
        weights.push(w);
    }
    (nodes, weights)
}

/// The `n`-point Gauss–Legendre rule moved to `[a, b]`: nodes `(a + b)/2 + (b − a)/2 · x` and
/// weights scaled by `(b − a)/2`, so that `Σ wᵢ f(tᵢ)` approximates `∫ₐᵇ f`.
///
/// # Panics
/// If `n` is zero.
#[must_use]
pub fn gauss_legendre_on<T: Real>(n: usize, a: T, b: T) -> (Vec<T>, Vec<T>) {
    let (x, w) = gauss_legendre::<T>(n);
    let half = T::from_f64(0.5);
    let (mid, radius) = ((a + b) * half, (b - a) * half);
    (
        x.into_iter().map(|x| radius.mul_add(x, mid)).collect(),
        w.into_iter().map(|w| w * radius).collect(),
    )
}
