//! Inclusions by generator names (ADR-057, numga's `from_subalgebra`): what they mean. The
//! laws themselves are proved by the generator on every pair of kinds and checked on random
//! values in `tests/inclusions.rs`.

#![cfg(all(feature = "pga2d", feature = "pga3d"))]

use gax::{ApproxEq, pga2d, pga3d};

#[test]
fn a_point_of_the_plane_is_the_point_of_space_at_z_zero() {
    let p = pga2d::Point::new(2.0, -1.0, 0.5); // (4, -2) with weight 0.5
    let q: pga3d::Point = p.include();
    assert!(q.approx_eq(&pga3d::Point::new(2.0, -1.0, 0.0, 0.5), 0.0));
    // Directions stay directions.
    let d: pga3d::Point = pga2d::Point::direction(1.0, 3.0).include();
    assert!(d.approx_eq(&pga3d::Point::direction(1.0, 3.0, 0.0), 0.0));
    // The homomorphism takes it elsewhere: to the vertical line through it, which is the
    // inclusion joined with the direction up.
    let vertical = pga3d::intrinsic::Line::from(p);
    let up = pga3d::Point::direction(0.0, 0.0, 1.0);
    assert!(vertical.approx_eq(&(q & up), 1e-15) || vertical.approx_eq(&-(q & up), 1e-15));
}

#[test]
fn joins_and_pairings_carry_over() {
    let (p, q) = (pga2d::Point::xy(1.0, 2.0), pga2d::Point::xy(-0.5, 0.25));
    let (p3, q3): (pga3d::Point, pga3d::Point) = (p.include(), q.include());
    // The line through two points of the plane is the line of space through them.
    let line = pga3d::intrinsic::Line::from_subalgebra(p & q);
    assert!(line.approx_eq(&(p3 & q3), 1e-15));
    // A line of the plane is the plane of space with its equation (the vertical plane), and its
    // pairing with a point is the same signed distance, the extrinsic value on the left.
    let l = pga2d::Line::new(0.6, -0.8, 0.5);
    let plane: pga3d::Plane = l.include();
    assert!(plane.approx_eq(&pga3d::Plane::new(0.6, -0.8, 0.0, 0.5), 0.0));
    assert!(((l & p).s() - (plane & p3).s()).abs() < 1e-15);
    // With the intrinsic value on the left the sign follows each algebra's dimension: in the
    // plane `p & l == l & p`, in space `p3 & plane == -(plane & p3)` (plane first, ADR-009).
    assert!(((p & l).s() - (l & p).s()).abs() < 1e-15);
    assert!(((p3 & plane).s() + (plane & p3).s()).abs() < 1e-15);
    // The meet of two lines of the plane, a point (extrinsic), is the meet of their planes: the
    // vertical line through it.
    let m = pga2d::Line::new(1.0, 1.0, -1.0);
    let meet = pga3d::Line::from_subalgebra(l ^ m);
    assert!(meet.approx_eq(&(plane ^ pga3d::Plane::from_subalgebra(m)), 1e-15));
}

#[test]
fn motions_carry_over() {
    let m = pga2d::Motor::rotation(pga2d::Point::xy(1.0, 0.0), 0.7).into_inner();
    let m3: pga3d::Motor = m.include();
    // The turn about a point of the plane is the turn about the vertical line through it: the
    // same as the homomorphism on extrinsic values.
    assert!(m3.approx_eq(&pga3d::Motor::from(m), 0.0));
    let p = pga2d::Point::xy(0.0, 2.0);
    let moved: pga3d::Point = (m >> p).include();
    assert!(moved.approx_eq(&(m3 >> p.include::<pga3d::Point>()), 1e-15));
    // A reflection too (odd versor, intrinsic passenger, ADR-051's `>>`).
    let l = pga2d::Line::new(0.6, 0.8, -1.0);
    let mirrored: pga3d::Point = (l >> p).include();
    assert!(mirrored.approx_eq(&(l.include::<pga3d::Plane>() >> p.include::<pga3d::Point>()), 1e-15));
}

#[test]
fn maps_carry_their_output_and_keep_their_slots() {
    // The inclusion itself, as a map from the plane's points.
    let lift = pga3d::Point::from_subalgebra(pga2d::Point::slot());
    let p = pga2d::Point::xy(0.3, -0.7);
    assert!(lift.of(p).approx_eq(&p.include(), 0.0));
    // A map of the plane, included: its output lands in space.
    let m = pga2d::Motor::translation(pga2d::Point::direction(1.0, 2.0)).into_inner();
    let moved = pga3d::Point::from_subalgebra(m >> pga2d::Point::slot());
    assert!(moved.of(p).approx_eq(&(m >> p).include(), 1e-15));
}

#[test]
fn duals_carry_over() {
    let l = pga2d::Line::new(0.6, -0.8, 0.5);
    let a: pga3d::Point = l.dual().include();
    let b = l.include::<pga3d::Plane>().dual();
    assert!(a.approx_eq(&b, 0.0));
    let p = pga2d::Point::xy(1.0, 2.0);
    let a: pga3d::Plane = p.undual().include();
    let b = p.include::<pga3d::Point>().undual();
    assert!(a.approx_eq(&b, 0.0));
}

#[cfg(feature = "vga3d")]
#[test]
fn space_in_projective_space() {
    use gax::vga3d;
    // numga's points of space are antivectors, offsets from the origin: they include as the
    // projective points at infinity of those offsets, directions.
    let offset = vga3d::Vector::new(1.0, 2.0, 3.0).dual(); // intrinsic bivector
    let d: pga3d::Direction = offset.include();
    assert!(d.approx_eq(&pga3d::Direction::new(1.0, 2.0, 3.0), 0.0));
    // A vector (a plane through the origin, extrinsic) keeps its blades, and a rotor its axis.
    // Each lands in the smallest kind holding its image, parts included.
    let v: pga3d::OriginPlane = vga3d::Vector::new(0.0, 0.0, 1.0).include();
    assert!(pga3d::Plane::from(v).approx_eq(&pga3d::Plane::new(0.0, 0.0, 1.0, 0.0), 0.0));
    let r = vga3d::Rotor::rotation(vga3d::Vector::new(0.0, 0.0, 1.0), 0.4).into_inner();
    let r3: pga3d::Rotor = r.include();
    assert!(r3.approx_eq(&pga3d::Rotor::from(r), 0.0));
    let moved: pga3d::Direction = (r >> offset).include();
    assert!(moved.approx_eq(&(r3 >> d), 1e-15));
}

#[cfg(all(feature = "vga2d", feature = "sta"))]
#[test]
fn the_plane_in_spacetime() {
    use gax::{sta, vga2d};
    // VGA2D's vectors are STA's `x, y` (photoelasticity's cuts through a plate).
    let cuts = sta::Vector::from_subalgebra(vga2d::Vector::slot());
    let v = cuts.of(vga2d::Vector::new(0.6, 0.8));
    assert!(v.approx_eq(&sta::Vector::new(0.6, 0.8, 0.0, 0.0), 0.0));
    // The metric carries over on extrinsic values.
    let w = vga2d::Vector::new(-1.0, 2.0);
    let w4: sta::Vector = w.include();
    assert!(((vga2d::Vector::new(0.6, 0.8) | w).s() - (v | w4).s()).abs() < 1e-15);
}
