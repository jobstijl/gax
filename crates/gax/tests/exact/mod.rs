//! Exact checks over the prime field `ℤ/p` (law L): random points, and polynomial fields with
//! random integer coefficients. An identity between polynomials that holds at random points of
//! `ℤ/p` holds as a polynomial identity with probability `1 − d/p` per sample (Schwartz–Zippel).

#![allow(dead_code)] // each test binary uses part of it

use gax::fp::Fp;
use gax::{Coef, Extensor, Kind};

/// A deterministic stream of samples.
pub struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 3
    }

    /// A point of `ℤ/p`.
    pub fn fp(&mut self) -> Fp {
        Fp::new(self.next())
    }

    /// A value with a random point of `ℤ/p` for each coefficient.
    pub fn point<X: Extensor<Slots = (), Coef = Fp>>(&mut self) -> X {
        X::from_coeffs(<X::Kind as Kind>::arr_from_fn(|_| self.fp()))
    }

    /// The coefficients of a cubic field: random integers below `2⁵⁹` in size.
    pub fn cubic(&mut self) -> Vec<i64> {
        (0..4096)
            .map(|_| (self.next() >> 2) as i64 - (1 << 59))
            .collect()
    }
}

/// The cubic field with coefficients `c`, from `X` to kind `Y`: each output coefficient is
/// `Σ c x̂ᵢ x̂ⱼ x̂ₖ` over `i ≤ j ≤ k`, with `x̂ = (1, x…)`, every monomial of degree at most 3.
pub fn cubic<X, Y>(c: &[i64], x: X) -> Y::Mv<(), X::Coef>
where
    X: Extensor<Slots = ()>,
    Y: Kind,
{
    let one = X::Coef::one();
    let v: Vec<X::Coef> = core::iter::once(one)
        .chain(x.coeffs().as_ref().iter().copied())
        .collect();
    let n = v.len();
    let mut terms = c.iter();
    Extensor::from_coeffs(Y::arr_from_fn(|_| {
        let mut sum = X::Coef::zero();
        for i in 0..n {
            for j in i..n {
                for k in j..n {
                    let ck = X::Coef::from_i64(*terms.next().expect("enough coefficients"));
                    sum = sum + ck * v[i] * v[j] * v[k];
                }
            }
        }
        sum
    }))
}
