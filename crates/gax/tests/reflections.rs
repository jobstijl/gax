//! Reflections and the join. `>>` is the plain sandwich `v x ~v`, and under an odd versor the
//! join picks up the determinant (Dorst, *Projective duality encodes complementary orientations
//! in geometric algebras*, 2024, §6.3): in PGA3D `(m >> a) & (m >> b) == −(m >> (a & b))` for a
//! unit plane `m`, in PGA2D no sign. These are the two cases of ganja.js issue 180 and kingdon
//! issue 154: a momentum `p & v` reflected by the sandwich against one rebuilt from the
//! reflected point and rate, and a point's side of a plane in a mirrored scene.

#![cfg(all(feature = "pga2d", feature = "pga3d"))]

/// A unit mass at the origin translating along `x`: reflected in the wall `x = 0`, its momentum
/// is minus the sandwich (it is built with one join), and then the kinetic energy is kept.
#[test]
fn pga3d_momentum_is_minus_the_sandwich() {
    use gax::pga3d::{Line, Plane, Point};
    let wall = Plane::<(), f64>::new(1.0, 0.0, 0.0, 0.0);
    let p = Point::<(), f64>::xyz(0.0, 0.0, 0.0);
    let rate = Line::<(), f64>::new(0.0, 0.0, 0.0, 1.0, 0.0, 0.0);
    let momentum = |p: Point<(), f64>, rate: Line<(), f64>| p & p.commutator(rate);
    let energy = |rate: Line<(), f64>, m: Line<(), f64>| 0.5 * (rate & m).s();
    let m = momentum(p, rate);
    assert_eq!(energy(rate, m), 0.5);

    let (p2, rate2) = (wall.reflect(p), wall >> rate);
    let rebuilt = momentum(p2, rate2);
    assert_eq!(rebuilt.c, (-(wall >> m)).c);
    assert_eq!(energy(rate2, rebuilt), 0.5);
    // The plain sandwich of the momentum is the sign the issues report.
    assert_eq!(energy(rate2, wall >> m), -0.5);
}

/// A duck at `(2, 1, 1)` above the water `z = 0`, mirrored in the wall `x = 1/2`: `reflect`
/// keeps the point's weight, and the plane is minus the sandwich (the sandwich alone reverses
/// its orientation), so the duck stays on the same side.
#[test]
fn pga3d_side_of_a_plane_survives_a_mirror() {
    use gax::pga3d::{Plane, Point};
    let wall = Plane::<(), f64>::new(1.0, 0.0, 0.0, -0.5);
    let water = Plane::<(), f64>::new(0.0, 0.0, 1.0, 0.0);
    let duck = Point::<(), f64>::xyz(2.0, 1.0, 1.0);
    let side = (water & duck).s();
    assert_eq!(side, 1.0);

    let duck2 = wall.reflect(duck);
    assert_eq!(duck2.c, Point::xyz(-1.0, 1.0, 1.0).c);
    assert_eq!(((-(wall >> water)) & duck2).s(), side);
    assert_eq!(((wall >> water) & duck2).s(), -side);
}

/// In PGA2D the join keeps its sign under reflections: `Line::reflect` on the point and the
/// direction, and the plain sandwich on the momentum, agree.
#[test]
fn pga2d_momentum_is_the_sandwich() {
    use gax::pga2d::{Line, Point};
    let wall = Line::<(), f64>::new(1.0, 0.0, 0.0);
    let p = Point::<(), f64>::xy(1.0, 2.0);
    let v = Point::<(), f64>::direction(1.0, 0.5);
    let m = p & v;
    let rebuilt = wall.reflect(p) & wall.reflect(v);
    assert_eq!(rebuilt.c, (wall >> m).c);
}
