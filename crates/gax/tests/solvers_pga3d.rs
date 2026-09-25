//! Methods of maps and forms: inverse, solve, det, trace, SVD, eigenproblems (PGA3D).

mod common;
use common::{Rng, assert_close, flat, random, random_map};
use gax::Unit;
use gax::pga3d::*;

fn unit_motor(rng: &mut Rng) -> Unit<Motor<(), f64>> {
    // exp of a random bivector, written out: rotation about an axis through the origin,
    // then a translation.
    let (ax, ay, az) = (rng.next_f64(), rng.next_f64(), rng.next_f64());
    let n = (ax * ax + ay * ay + az * az).sqrt().max(1e-3);
    let angle = 2.0 * rng.next_f64();
    let (s, c) = (angle / 2.0).sin_cos();
    let r = Motor::new(c, s * ax / n, s * ay / n, s * az / n, 0.0, 0.0, 0.0, 0.0);
    let t = Motor::new(
        1.0,
        0.0,
        0.0,
        0.0,
        rng.next_f64(),
        rng.next_f64(),
        rng.next_f64(),
        0.0,
    );
    Unit::new_unchecked(t * r)
}

#[test]
fn motor_map_inverse_is_the_reverse_map() {
    let mut rng = Rng::new(21);
    for _ in 0..10 {
        let m = unit_motor(&mut rng);
        let fwd: Point<(Point,), f64> = m >> Point::slot();
        let back: Point<(Point,), f64> = m << Point::slot();
        assert_close(
            &flat(&fwd.inverse()),
            &flat(&back),
            "inverse of the motor's map",
        );
        assert!(
            (fwd.det() - 1.0).abs() < 1e-12,
            "rigid motions preserve volume and weight"
        );
        assert!(
            (fwd.of(back).trace() - 4.0).abs() < 1e-12,
            "trace of the identity"
        );
    }
}

#[test]
fn solve_inverts_application_for_values_and_maps() {
    let mut rng = Rng::new(22);
    for _ in 0..10 {
        let t: Line<(Line,), f64> = random_map(&mut rng);
        let x: Line<(), f64> = random(&mut rng);
        assert_close(&flat(&t.solve(t.of(x))), &flat(&x), "solve(of(x))");
        let y: Line<(Point,), f64> = random_map(&mut rng);
        let sol: Line<(Point,), f64> = t.solve(t.of(y));
        assert_close(
            &flat(&sol),
            &flat(&y),
            "solve keeps the right-hand side's slots",
        );
    }
}

#[test]
fn svd_of_a_map_between_kinds() {
    let mut rng = Rng::new(23);
    let t: Plane<(Point,), f64> = random_map(&mut rng);
    let (u, s, v) = t.svd();
    for i in 0..4 {
        assert_close(&flat(&t.of(v[i])), &flat(&(u[i] * s[i])), "t(v) = s u");
    }
    assert!(s[0] >= s[1] && s[1] >= s[2] && s[2] >= s[3] && s[3] >= 0.0);
}

#[test]
fn generalized_eigenproblem_returns_modes_of_the_slot_kind() {
    let mut rng = Rng::new(24);
    // Two symmetric positive definite forms on twists, built as sums of dyads. The pairing is
    // the regressive product: the metric pairing `|` of PGA lines is degenerate (it ignores
    // the moment), so dyads built with it would be singular.
    let mut stiffness: Scalar<(Line, Line), f64> = Scalar::zero();
    let mut inertia: Scalar<(Line, Line), f64> = Scalar::zero();
    for _ in 0..8 {
        let a: Line<(), f64> = random(&mut rng);
        let b: Line<(), f64> = random(&mut rng);
        let pa: Scalar<(Line,), f64> = a & Line::slot();
        let pb: Scalar<(Line,), f64> = b & Line::slot();
        stiffness += pa * pa;
        inertia += pb * pb;
    }
    let (values, modes) = stiffness.eigh_with(inertia);
    for k in 0..6 {
        // stiffness(x, .) == λ inertia(x, .)
        let lhs: Scalar<(Line,), f64> = stiffness.of(modes[k]);
        let rhs: Scalar<(Line,), f64> = inertia.of(modes[k]).gp(values[k]);
        assert_close(&flat(&lhs), &flat(&rhs), "K x = λ M x");
        let norm = inertia.of(modes[k]).of(modes[k]);
        assert!(
            (norm.s() - 1.0).abs() < 1e-10,
            "modes are inertia-normalized"
        );
    }
    // A form solve: stiffness(x, .) == stiffness(y, .) recovers y.
    let y: Line<(), f64> = random(&mut rng);
    assert_close(
        &flat(&stiffness.solve(stiffness.of(y))),
        &flat(&y),
        "form solve",
    );
}
