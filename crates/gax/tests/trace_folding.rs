//! The tracer folds what does not depend on the inputs, and keeps staged programs well-founded.

#![cfg(all(feature = "trace", feature = "pga2d"))]

use gax::pga2d::{Motor, Point};
use gax::trace::{Sym, Tracer};
use gax::{Coef, Real};

type P = Point<(), Sym>;

fn select<T: Real>(a: T, b: T, x: Point<(), T>, y: Point<(), T>) -> Point<(), T> {
    Point::from_coeffs(core::array::from_fn(|i| T::select_lt(a, b, x.c[i], y.c[i])))
}

fn quarter() -> gax::Unit<Motor<(), Sym>> {
    Motor::rotation(
        Point::xy(Sym::zero(), Sym::zero()),
        Sym::from_f64(core::f64::consts::FRAC_PI_2),
    )
}

/// A motor from constant angles costs nothing to build: its `exp` (square roots, sine,
/// cosine, the small-angle select) folds at trace time, and turning a direction a quarter
/// is a swap with a sign.
#[test]
fn constant_motors_fold() {
    let mut t = Tracer::new();
    t.kernel("quarter", |u: P| quarter() >> u);
    let c = t.reports()[0].cost;
    assert_eq!((c.calls, c.divs), (0, 0), "{c}");
    assert!(c.muls <= 2, "{c}");
}

/// Relations must not reach back into a stage: `sqrt x` with the relation `s² = x` once
/// compiled into `sqrt(s s)`, reading its own result.
#[test]
fn stages_only_use_what_is_defined_before_them() {
    let mut t = Tracer::new();
    t.kernel("unit_then_turn", |a: P, b: P| {
        let d = b - a;
        let len = d.ideal_norm();
        let u = select(
            len,
            Sym::from_f64(1e-6),
            Point::direction(Sym::one(), Sym::zero()),
            d.gp(Sym::one() / (len + Sym::from_f64(1e-30))),
        );
        quarter() >> u
    });
    let c = t.reports()[0].cost;
    assert!(c.calls <= 4 && c.muls <= 10, "{c}");
    let v: f32 = unit_then_turn_check();
    assert!((v - 1.0).abs() < 1e-6);
}

/// The generic code, for comparison: turning `(3, 0)` gives the unit `(0, 1)` (up to the
/// direction's sign convention); the traced kernel is checked against it by the tracer.
fn unit_then_turn_check() -> f32 {
    let q = Motor::rotation(Point::xy(0.0f32, 0.0), core::f32::consts::FRAC_PI_2);
    let u = q >> Point::direction(1.0f32, 0.0);
    u.e01().abs()
}
