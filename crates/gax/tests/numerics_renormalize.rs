//! `Unit::renormalize_fast` is one Newton step: for a unit versor `u` that has drifted to
//! `m ~m = 1 + e`, the step leaves `r ~r − 1 = O(e²)`. The identity behind it is proved on
//! symbolic coefficients, and the convergence measured in f64 (docs/numerics.md, "Drift").

#![cfg(all(feature = "pga3d", feature = "pga2d", feature = "vga3d"))]

mod law_suite;

use gax::{Coef, Extensor, Kind, NewtonStep, Unit};
use law_suite::{Sym, T, fresh, reset};

/// For any `m` whose norm `n = m ~m` is a Study number (it commutes with `m`), the step
/// `r = m (3 − n) / 2` has `r ~r = n (3 − n)² / 4` exactly. With `n = 1 + e` that is
/// `1 − ¾ e² + ¼ e³` (a scalar identity, valid for Study numbers too), so one step turns an
/// error `e` into `O(e²)`.
fn newton_identity<M, N>()
where
    M: Extensor<Slots = (), Coef = Sym>
        + gax::Reverse<Output = M>
        + core::ops::Mul<Output = N>
        + core::ops::Mul<N, Output = M>
        + NewtonStep,
    N: Extensor<Slots = (), Coef = Sym>
        + core::ops::Mul<Output = N>
        + core::ops::Sub<Output = N>
        + Copy,
    N: gax::Gp<Sym, Output = N>,
{
    reset();
    let m: M = fresh();
    let n: N = m * m.reverse();
    let three: N = N::from_coeffs(<N::Kind as Kind>::arr_from_fn(|i| {
        if <N::Kind as Kind>::BLADES[i] == "1" {
            Sym::from_i64(3)
        } else {
            Sym::from_i64(0)
        }
    }));
    let q = (three - n).gp(Sym::from_f64(0.5));
    let r = m.newton_step();
    let mq = m * q;
    law_suite::law(r == mq, "the step is m (3 − n) / 2");
    law_suite::law(r * r.reverse() == n * q * q, "r ~r = n (3 − n)² / 4");
}

#[test]
fn one_newton_step_squares_the_error() {
    newton_identity::<gax::pga3d::Motor<(), T>, gax::pga3d::Motor<(), T>>();
    newton_identity::<gax::pga2d::Motor<(), T>, gax::pga2d::Motor<(), T>>();
    newton_identity::<gax::vga3d::Rotor<(), T>, gax::vga3d::Rotor<(), T>>();
}

#[test]
fn newton_steps_converge_quadratically() {
    use gax::pga3d::{Line, Motor};
    let u = Line::<(), f64>::new(0.4, -0.3, 0.8, 0.9, -0.5, 0.2).exp();
    let err = |m: Motor<(), f64>| {
        let n = m * m.reverse();
        n.c.iter()
            .enumerate()
            .map(|(i, c)| if i == 0 { (c - 1.0).abs() } else { c.abs() })
            .fold(0.0f64, f64::max)
    };
    for delta in [1e-2, 1e-4] {
        // Drift in both Study parts: scale and a pseudoscalar component.
        let drift = Motor::new(1.0 + delta, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, delta);
        let m = Unit::new_unchecked(u.into_inner() * drift);
        let e0 = err(m.into_inner());
        let e1 = err(m.renormalize_fast().into_inner());
        let e2 = err(m.renormalize_fast().renormalize_fast().into_inner());
        assert!(e0 > delta, "drift {e0}");
        assert!(e1 < 4.0 * e0 * e0, "one step: {e0} -> {e1}");
        assert!(e2 < 4.0 * e1 * e1 + 1e-15, "two steps: {e1} -> {e2}");
    }
    // mul_renormalized keeps a chain of compositions on the unit condition.
    let (a, b) = (
        Line::<(), f64>::new(0.1, 0.2, -0.3, 0.2, 0.1, 0.0).exp(),
        Line::<(), f64>::new(-0.2, 0.1, 0.4, 0.0, 0.3, -0.1).exp(),
    );
    let mut plain = a;
    let mut fixed = a;
    for _ in 0..100_000 {
        plain = plain * b;
        fixed = fixed.mul_renormalized(b);
    }
    assert!(
        err(fixed.into_inner()) < 1e-13,
        "renormalized chain: {}",
        err(fixed.into_inner())
    );
    assert!(err(fixed.into_inner()) <= err(plain.into_inner()));
}
