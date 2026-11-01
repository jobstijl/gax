//! Correlations, maps between a kind and its complement (ADR-014, amended 2026-11-01): their
//! outermorphisms, each side multiplying by its own product, and their eigenproblems through the
//! regressive product. The exact laws are in the law suites (`outermorphism`, laws.md E).

#![cfg(all(feature = "pga2d", feature = "pga3d", feature = "vga3d"))]

use gax::{ApproxEq, pga2d, pga3d, vga3d};

/// A polarity from points to planes: the pole-polar map of the quadric `x² + 2y² - z² - w² = 0`
/// plus a shear, so that no blade goes to its own complement alone.
fn polarity() -> pga3d::Plane<(pga3d::Point,)> {
    let rows = [
        [1.0, 0.2, 0.0, 0.1],
        [0.2, 2.0, 0.3, 0.0],
        [0.0, 0.3, -1.0, 0.4],
        [0.1, 0.0, 0.4, -1.0],
    ];
    // Plane `a x + b y + c z + d w` from a point's (x, y, z, w).
    let images = core::array::from_fn(|i| {
        let r = rows[i];
        pga3d::Plane::new(r[0], r[1], r[2], r[3])
    });
    pga3d::Plane::from_images(images)
}

#[test]
fn a_polarity_carries_joins_to_meets() {
    let c = polarity();
    let (p, q, r) = (
        pga3d::Point::xyz(1.0, 2.0, 0.5),
        pga3d::Point::xyz(-0.3, 0.7, 2.0),
        pga3d::Point::direction(0.2, -1.0, 0.4),
    );
    // The line through two points goes to the meet of their polar planes.
    let on_lines = c.outermorphism::<pga3d::Line>();
    assert!(on_lines.of(p & q).approx_eq(&(c.of(p) ^ c.of(q)), 1e-12));
    // The plane through three points to the meet of three planes, a point.
    let on_planes = c.outermorphism::<pga3d::Plane>();
    let pole = on_planes.of(p & q & r);
    assert!(pole.approx_eq(&(c.of(p) ^ c.of(q) ^ c.of(r)), 1e-12));
    // The scalar (the join of four points) to the pseudoscalar: the determinant.
    let volume = c.outermorphism::<pga3d::Scalar>();
    let s = pga3d::Point::xyz(0.0, 0.0, 1.0);
    let wedge = c.of(p) ^ c.of(q) ^ c.of(r) ^ c.of(s);
    assert!(volume.of(p & q & r & s).approx_eq(&wedge, 1e-12));
}

#[test]
fn a_map_from_planes_to_points_carries_meets_to_joins() {
    // Back the other way: the inverse of the polarity, from planes to points.
    let c: pga3d::Point<(pga3d::Plane,)> = polarity().inverse();
    let (a, b) = (pga3d::Plane::new(1.0, 0.0, 0.5, -1.0), pga3d::Plane::new(0.0, 1.0, 1.0, 2.0));
    let on_lines = c.outermorphism::<pga3d::Line>();
    assert!(on_lines.of(a ^ b).approx_eq(&(c.of(a) & c.of(b)), 1e-12));
}

/// PGA2D's normal modes (numga's `mechanics/modes`): a stiffness and an inertia from twists to
/// forques, both correlations.
fn stiffness_and_inertia() -> (pga2d::Forque<(pga2d::Twist,)>, pga2d::Forque<(pga2d::Twist,)>) {
    let mut k = pga2d::Forque::<(pga2d::Twist,)>::zero();
    for (l, w) in [
        (pga2d::Forque::new(1.0, 0.0, 0.5), 2.0),
        (pga2d::Forque::new(0.0, 1.0, -0.3), 1.0),
        (pga2d::Forque::new(0.6, 0.8, 0.2), 3.0),
    ] {
        k += (l * (l & pga2d::Twist::slot())).gp(w);
    }
    let points = [pga2d::Point::xy(1.0, 0.0), pga2d::Point::xy(-1.0, 0.5), pga2d::Point::xy(0.0, -1.0)];
    let m = points
        .iter()
        .zip([1.0, 2.0, 0.5])
        .map(|(p, mass)| (*p & p.commutator(pga2d::Twist::slot())).gp(mass))
        .fold(pga2d::Forque::<(pga2d::Twist,)>::zero(), |a, b| a + b);
    (k, m)
}

#[test]
fn a_correlations_eigenproblem_is_the_forms() {
    let (k, m) = stiffness_and_inertia();
    let (values, modes) = k.eigh_in(m);
    // numga's route: the forms `Twist & k` and `Twist & m`.
    let (form_values, _) = (pga2d::Twist::slot() & k).eigh(pga2d::Twist::slot() & m);
    for (a, b) in values.iter().zip(form_values) {
        assert!((a - b).abs() < 1e-12 * (1.0 + b.abs()), "{a} against {b}");
    }
    for (value, mode) in values.iter().zip(modes) {
        assert!(k.of(mode).approx_eq(&(m.of(mode) * *value), 1e-12));
        assert!(((m.of(mode) & mode).s() - 1.0).abs() < 1e-12);
    }
    // A metric of the other sign is taken as its negative: the values change sign.
    let (negated, _) = k.eigh_in(-m);
    for (a, b) in values.iter().rev().zip(negated) {
        assert!((a + b).abs() < 1e-12 * (1.0 + a.abs()));
    }
}

#[test]
fn a_symmetric_map_through_the_hodge_dual() {
    // A stress, symmetric on vectors, read as a correlation to planes through the Hodge dual
    // (numga's `Plane & stress(Plane)` against `Plane | Plane`): its principal values.
    let s = vga3d::Vector::<(vga3d::Vector,)>::from_images([
        vga3d::Vector::new(2.0, 0.5, 0.0),
        vga3d::Vector::new(0.5, -1.0, 0.3),
        vga3d::Vector::new(0.0, 0.3, 0.5),
    ]);
    let correlation = s.hodge();
    let metric = vga3d::Vector::slot().hodge();
    let (values, _) = correlation.eigh_in(metric);
    let (want, _) = s.eigh();
    for (a, b) in values.iter().zip(want) {
        assert!((a - b).abs() < 1e-12, "{a} against {b}");
    }
}
