//! Values, maps and forms through one API (PGA2D). The algebraic laws are proved in `laws_pga2d.rs`.

mod common;
use common::{Rng, assert_close, flat, random};
use gax::Slots;
use gax::pga2d::*;

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
