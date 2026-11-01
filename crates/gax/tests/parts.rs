//! Parts as results and operands (ADR-054, amended 2026-11-01): a part and a kind holding it
//! add into the kind; the products, sandwiches and unary operations of parts stay in a part
//! where they close; and PGA3D's tilts are slots only, forms and maps on them reached through
//! narrowing a spin's.

#![cfg(all(feature = "pga3d", feature = "vga3d"))]

use gax::pga3d::{Direction, Motor, OriginPlane, Plane, Point, Scalar, Spin, Tilt, TiltMoment};

fn close(a: &[f64], b: &[f64], tol: f64) -> bool {
    a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tol)
}

#[test]
fn a_kind_and_its_part_add_into_the_kind() {
    use gax::vga3d::{Horizontal, Vector};
    let p = Point::xyz(1.0, 2.0, 3.0);
    let d: Direction = Point::direction(0.5, 0.0, -1.0).cast::<Direction>();
    let q: Point = p + d;
    assert_eq!(q.c, [1.5, 2.0, 2.0, 1.0]);
    let r: Point = d - p;
    assert_eq!(r.c, (Point::from(d) - p).c);
    let mut s = p;
    s -= d;
    assert_eq!(s.c, [0.5, 2.0, 4.0, 1.0]);
    // And in VGA3D: a vector from its horizontal part and its height.
    let v: Vector = Horizontal::new(1.0, 2.0) + Vector::new(0.0, 0.0, 3.0);
    assert_eq!(v.c, [1.0, 2.0, 3.0]);
}

#[test]
fn parts_stay_parts_where_they_close() {
    use gax::vga3d::Horizontal;
    // A motor moves a direction to a direction: translations leave it, rotations turn it.
    let m = Motor::translation(Point::direction(5.0, -1.0, 2.0))
        * Motor::rotation_about(
            Point::direction(0.0, 0.0, 1.0),
            core::f64::consts::FRAC_PI_2,
        );
    let d: Direction = Point::direction(1.0, 2.0, 3.0).cast::<Direction>();
    let moved: Direction = m >> d;
    assert!(close(&moved.c, &[-2.0, 1.0, 3.0], 1e-12));
    let back: Direction = m << moved;
    assert!(close(&back.c, &d.c, 1e-12));
    // Two planes through the origin meet in a line through the origin, a spin, and a spin's
    // commutator with one of them is another plane through the origin.
    let (x, y) = (
        OriginPlane::new(1.0, 0.0, 0.0),
        OriginPlane::new(0.0, 1.0, 0.0),
    );
    let s: Spin = x ^ y;
    assert_eq!(s.c, [0.0, 0.0, 1.0]);
    let turned: OriginPlane = s.commutator(x);
    assert_eq!(turned.c.iter().filter(|c| **c != 0.0).count(), 1);
    let r: Spin = s.reverse();
    assert_eq!(r.c, [0.0, 0.0, -1.0]);
    // The complement of a plane through the origin is a direction.
    let n: gax::pga3d::intrinsic::Direction = x.dual();
    assert_eq!(n.c.iter().filter(|c| **c != 0.0).count(), 1);
    // The other kinds keep their results: a plane's complement is a point.
    let p: gax::pga3d::intrinsic::Point = Plane::new(1.0, 0.0, 0.0, 2.0).dual();
    assert_eq!(p.c.len(), 4);
    // VGA3D: the dual of a horizontal vector is a tilt, and back.
    let t: gax::vga3d::intrinsic::Tilt = Horizontal::new(1.0, 2.0).hodge();
    let h: Horizontal = t.hodge_inverse();
    assert_eq!(h.c, [1.0, 2.0]);
}

#[test]
fn tilts_are_slots_of_spin_forms() {
    // A form on spins, narrowed to tilts: the 2 × 2 block numga's floating bodies read.
    let k: Scalar<(Spin, Spin)> =
        Scalar::from_coeffs([[[2.0, 0.5, 0.1], [0.5, 3.0, 0.2], [0.1, 0.2, 7.0]]]);
    let block: Scalar<(Tilt, Tilt)> = k.of(Tilt::slot()).at::<1>().of(Tilt::slot());
    assert_eq!(block.c, [[[2.0, 0.5], [0.5, 3.0]]]);
    let (values, vectors) = block.eigh(Scalar::<(Tilt, Tilt)>::identity());
    let mean = 2.5;
    let spread = (0.25f64 + 0.25).sqrt();
    assert!(close(&values, &[mean - spread, mean + spread], 1e-14));
    assert!((block.of(vectors[0]).of(vectors[0]).s() - values[0]).abs() < 1e-14);
    // A tilt is a spin by its embedding, and binds a spin form directly.
    let t = Tilt::new(1.0, -1.0);
    assert_eq!(k.of(t).of(t).s(), k.of(Spin::from(t)).of(Spin::from(t)).s());
    // Its complement holds the couples that tilt: the adjugate of a map on tilts acts on them,
    // `(t.adjugate().of(c) & a)₀ == (c & t.of(a))₀`, here through the operand kinds.
    let map: Tilt<(Tilt,)> = Tilt::from_coeffs([[1.0, 2.0], [-0.5, 3.0]]);
    let adjugate: TiltMoment<(TiltMoment,)> = map.adjugate();
    let (c, a) = (TiltMoment::new(0.3, -0.7), Tilt::new(1.5, 0.25));
    let lhs = gax::pga3d::Moment::from(adjugate.of(c)) & Spin::from(a);
    let rhs = gax::pga3d::Moment::from(c) & Spin::from(map.of(a));
    assert!((lhs.s() - rhs.s()).abs() < 1e-14);
}
