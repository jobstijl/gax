//! Values, maps and forms through one API (PGA2D).

mod common;
use common::{Rng, assert_close, flat, random, random_map};
use gax::pga2d::*;
use gax::{Slots, Unit};

/// The motivating example: written once, generic over the open slots.
fn shadow<S: Slots>(
    light: Point<(), f64>,
    ground: Line<(), f64>,
    p: Point<S, f64>,
) -> Point<S, f64> {
    (light & p) ^ ground
}

/// The same with the value operands on the other sides.
fn shadow_flipped<S: Slots>(
    light: Point<(), f64>,
    ground: Line<(), f64>,
    p: Point<S, f64>,
) -> Point<S, f64> {
    ground ^ (p & light)
}

#[test]
fn generic_function_evaluates_values_and_builds_maps() {
    let mut rng = Rng::new(7);
    let light: Point<(), f64> = random(&mut rng);
    let ground: Line<(), f64> = random(&mut rng);
    let map: Point<(Point,), f64> = shadow(light, ground, Point::slot());
    let flipped: Point<(Point,), f64> = shadow_flipped(light, ground, Point::slot());
    for _ in 0..10 {
        let p: Point<(), f64> = random(&mut rng);
        assert_close(
            &flat(&map.of(p)),
            &flat(&shadow(light, ground, p)),
            "shadow map vs value",
        );
        assert_close(
            &flat(&flipped.of(p)),
            &flat(&shadow_flipped(light, ground, p)),
            "flipped",
        );
    }
}

#[test]
fn identity_slot_is_the_identity() {
    let mut rng = Rng::new(8);
    let p: Point<(), f64> = random(&mut rng);
    assert_eq!(Point::slot().of(p), p);
    let m: Line<(Point,), f64> = random_map(&mut rng);
    assert_eq!(m.of(Point::slot()), m);
}

#[test]
fn composition_matches_sequential_application() {
    let mut rng = Rng::new(9);
    let f: Line<(Point,), f64> = random_map(&mut rng);
    let g: Point<(Line,), f64> = random_map(&mut rng);
    let h: Line<(Line,), f64> = random_map(&mut rng);
    let fg: Line<(Line,), f64> = f.of(g);
    let fgh = f.of(g.of(h));
    let fg_h = f.of(g).of(h);
    assert_close(&flat(&fgh), &flat(&fg_h), "associativity of composition");
    for _ in 0..10 {
        let x: Line<(), f64> = random(&mut rng);
        assert_close(&flat(&fg.of(x)), &flat(&f.of(g.of(x))), "composition");
    }
}

#[test]
fn products_of_maps_bind_in_any_order() {
    let mut rng = Rng::new(10);
    // A bilinear map: the join of two open points.
    let join: Line<(Point, Point), f64> = Point::slot() & Point::slot();
    let a: Point<(), f64> = random(&mut rng);
    let b: Point<(), f64> = random(&mut rng);
    assert_close(&flat(&join.of(a).of(b)), &flat(&(a & b)), "join(a)(b)");
    // Binding the second slot first: fix b by composing on the right.
    let with_b: Line<(Point,), f64> = Point::slot() & b;
    assert_close(&flat(&with_b.of(a)), &flat(&(a & b)), "join(., b)(a)");
    // A map times a map times a value.
    let f: Motor<(Line,), f64> = random_map(&mut rng);
    let g: Point<(Point,), f64> = random_map(&mut rng);
    let prod: Motor<(Line, Point), f64> = f * g;
    let (x, y): (Line<(), f64>, Point<(), f64>) = (random(&mut rng), random(&mut rng));
    assert_close(
        &flat(&prod.of(x).of(y)),
        &flat(&(f.of(x) * g.of(y))),
        "bilinear product",
    );
}

#[test]
fn sandwich_matrix_path_matches_direct_path() {
    let mut rng = Rng::new(11);
    for _ in 0..10 {
        let m: Motor<(), f64> = random(&mut rng);
        let matrix: Point<(Point,), f64> = m >> Point::slot();
        let p: Point<(), f64> = random(&mut rng);
        assert_close(
            &flat(&matrix.of(p)),
            &flat(&(m >> p)),
            "motor map vs direct",
        );
        // Transporting a map moves its output.
        let f: Point<(Line,), f64> = random_map(&mut rng);
        let l: Line<(), f64> = random(&mut rng);
        assert_close(
            &flat(&(m >> f).of(l)),
            &flat(&(m >> f.of(l))),
            "motor >> map",
        );
        // Unit versors take the simplified kernels.
        let n = (m.s() * m.s() + m.e12() * m.e12()).sqrt();
        let u = Unit::new_unchecked(m / n);
        assert_close(
            &flat(&(u >> Point::slot()).of(p)),
            &flat(&(u >> p)),
            "unit map vs direct",
        );
        assert_close(&flat(&(u << (u >> p))), &flat(&p), "u << (u >> p) == p");
    }
}

#[test]
fn forms_from_pairings() {
    let mut rng = Rng::new(12);
    // A rank-one quadratic form on points, the dyad of a line with itself: q(a, b) = (l & a)(l & b).
    let l: Line<(), f64> = random(&mut rng);
    let lp: Scalar<(Point,), f64> = l & Point::slot();
    let form: Scalar<(Point, Point), f64> = lp * lp;
    let (a, b): (Point<(), f64>, Point<(), f64>) = (random(&mut rng), random(&mut rng));
    let direct = (l & a) * (l & b);
    assert_close(&flat(&form.of(a).of(b)), &flat(&direct), "form(a, b)");
}
