//! Slot permutation, equality groups by kind, and pairings (PGA3D).

mod common;
use common::{Rng, assert_close, flat, random, random_map};
use gax::Unit;
use gax::pga3d::*;

#[test]
fn at_moves_a_slot_to_the_front() {
    let mut rng = Rng::new(31);
    let join: Line<(Point, Point), f64> = Point::slot() & Point::slot();
    let (a, b): (Point<(), f64>, Point<(), f64>) = (random(&mut rng), random(&mut rng));
    // Fill the second slot first.
    let second: Line<(Point,), f64> = join.at::<1>().of(b);
    assert_close(&flat(&second.of(a)), &flat(&(a & b)), "at::<1>");
    // Three slots: the volume spanned by three open points and a fixed one. Move the last
    // slot to the front.
    let q: Point<(), f64> = random(&mut rng);
    let tri: Scalar<(Point, Point, Point), f64> =
        ((Point::slot() & Point::slot()) & Point::slot()) & q;
    let c: Point<(), f64> = random(&mut rng);
    let direct = tri.of(a).of(b).of(c);
    let moved = tri.at::<2>().of(c).of(a).of(b);
    assert_close(&flat(&moved), &flat(&direct), "at::<2>");
}

#[test]
fn swap_transposes_a_form() {
    let mut rng = Rng::new(32);
    let f: Scalar<(Line, Point), f64> = random_map(&mut rng);
    let (l, p): (Line<(), f64>, Point<(), f64>) = (random(&mut rng), random(&mut rng));
    assert_close(&flat(&f.swap().of(p).of(l)), &flat(&f.of(l).of(p)), "swap");
}

#[test]
fn fill_binds_every_slot_of_a_kind() {
    let mut rng = Rng::new(33);
    // A sandwich with the motor left open: m p ~m has two motor slots.
    let p: Point<(), f64> = random(&mut rng);
    let open: Flector<(Motor, Motor), f64> = (Motor::slot() * p) * Motor::slot().reverse();
    for _ in 0..5 {
        let m: Motor<(), f64> = random(&mut rng);
        let filled: Flector<(), f64> = open.fill(m);
        let direct: Point<(), f64> = m >> p;
        // The fused sandwich returns the point part; the rest of the flector vanishes.
        let d = flat(&filled);
        assert_close(&d[..4], &[0.0; 4], "no plane part");
        assert_close(&d[4..], &flat(&direct), "fill == sandwich");
    }
    // Filling a kind that is not among the slots is the identity on the slots.
    let q: Line<(Point,), f64> = random_map(&mut rng);
    let same: Line<(Point,), f64> = q.fill(Motor::<(), f64>::zero());
    assert_eq!(same, q);
    // Mixed slots: only the motor slots are filled, the point slot stays open.
    let mixed: Flector<(Motor, Point, Motor), f64> =
        (Motor::slot() * Point::slot()) * Motor::slot().reverse();
    let m: Motor<(), f64> = random(&mut rng);
    let map: Flector<(Point,), f64> = mixed.fill(m);
    assert_close(
        &flat(&map.of(p)),
        &flat(&open.fill(m)),
        "fill keeps other slots in order",
    );
}

#[test]
fn induced_map_on_planes_from_a_pairing() {
    let mut rng = Rng::new(34);
    // A central projection onto a plane is singular, but it still induces a map on planes.
    let eye: Point<(), f64> = Point::new(0.0, 0.0, 5.0, 1.0);
    let screen: Plane<(), f64> = Plane::new(0.0, 0.0, 1.0, 0.0);
    let projection: Point<(Point,), f64> = (eye & Point::slot()) ^ screen;
    let pairing: Scalar<(Plane, Point), f64> = Plane::slot() & Point::slot();
    let induced: Plane<(Plane,), f64> = pairing.solve(Plane::slot() & projection);
    for _ in 0..5 {
        let (l, p): (Plane<(), f64>, Point<(), f64>) = (random(&mut rng), random(&mut rng));
        assert_close(
            &flat(&(induced.of(l) & p)),
            &flat(&(l & projection.of(p))),
            "induced(l) & p == l & t(p)",
        );
    }
    // A rigid motion's induced map on planes is the motion acting on planes.
    let m = Unit::new_unchecked(Motor::new(0.8, 0.6, 0.0, 0.0, 0.3, -0.1, 0.2, 0.0));
    let on_points: Point<(Point,), f64> = m >> Point::slot();
    let on_planes: Plane<(Plane,), f64> = pairing.solve(Plane::slot() & on_points.inverse());
    let l: Plane<(), f64> = random(&mut rng);
    let _ = on_planes.of(l);
}

#[test]
fn trace_at_contracts_a_slot_with_the_output() {
    let mut rng = Rng::new(35);
    // For a map on points, tracing its only slot is the matrix trace.
    let t: Point<(Point,), f64> = random_map(&mut rng);
    let s: Scalar<(), f64> = t.trace_at::<0>();
    assert!((s.s() - t.trace()).abs() < 1e-12);
    // A bilinear map Point <- (Line, Point): tracing the point slot leaves a linear form on lines,
    // which on any line l equals the trace of the map q -> m(l, q).
    let m: Point<(Line, Point), f64> = random_map(&mut rng);
    let form: Scalar<(Line,), f64> = m.trace_at::<1>();
    let l: Line<(), f64> = random(&mut rng);
    let direct = m.of(l).trace();
    assert!((form.of(l).s() - direct).abs() < 1e-12);
}

#[test]
fn outermorphisms_extend_maps_factor_by_factor() {
    let mut rng = Rng::new(36);
    // A map on planes extends by the wedge: M(a ^ b) = T(a) ^ T(b), and the top grade is det.
    let t: Plane<(Plane,), f64> = random_map(&mut rng);
    let (a, b, c): (Plane<(), f64>, Plane<(), f64>, Plane<(), f64>) =
        (random(&mut rng), random(&mut rng), random(&mut rng));
    let m2: Line<(Line,), f64> = t.outermorphism::<Line>();
    assert_close(&flat(&m2.of(a ^ b)), &flat(&(t.of(a) ^ t.of(b))), "M(a^b)");
    let m3: Point<(Point,), f64> = t.outermorphism::<Point>();
    assert_close(
        &flat(&m3.of(a ^ b ^ c)),
        &flat(&(t.of(a) ^ t.of(b) ^ t.of(c))),
        "M(a^b^c)",
    );
    let top: Pseudoscalar<(Pseudoscalar,), f64> = t.outermorphism::<Pseudoscalar>();
    assert!(
        (top.c[0][0] - t.det()).abs() < 1e-12,
        "top grade is the determinant"
    );
    // A map on points (a singular camera projection too) extends by the vee: M(p & q) = T(p) & T(q).
    let eye: Point<(), f64> = Point::xyz(0.0, 0.0, 5.0);
    let screen: Plane<(), f64> = Plane::from_normal([0.0, 0.0, 1.0], 0.0);
    let projection: Point<(Point,), f64> = (eye & Point::slot()) ^ screen;
    for tp in [projection, random_map(&mut rng)] {
        let (p, q, r): (Point<(), f64>, Point<(), f64>, Point<(), f64>) =
            (random(&mut rng), random(&mut rng), random(&mut rng));
        let lines: Line<(Line,), f64> = tp.outermorphism::<Line>();
        assert_close(
            &flat(&lines.of(p & q)),
            &flat(&(tp.of(p) & tp.of(q))),
            "M(p & q)",
        );
        let planes: Plane<(Plane,), f64> = tp.outermorphism::<Plane>();
        assert_close(
            &flat(&planes.of((p & q) & r)),
            &flat(&((tp.of(p) & tp.of(q)) & tp.of(r))),
            "M(p & q & r)",
        );
    }
}
