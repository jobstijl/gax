//! Angles of directions and vectors: `from_angle` and the signed `angle_to` in the plane, the
//! unsigned `angle_between` in every dimension, agreeing across VGA and PGA; and
//! `Extensor::is_finite`.

#![cfg(all(
    feature = "pga2d",
    feature = "pga3d",
    feature = "vga2d",
    feature = "vga3d"
))]

use core::f64::consts::{FRAC_PI_2, PI};
use gax::Extensor;
use gax::{pga2d, pga3d, vga2d, vga3d};

/// Angles over a full turn, the half turn and its neighbours included.
fn angles() -> impl Iterator<Item = f64> {
    (-64..=64)
        .map(|k| f64::from(k) * PI / 64.0)
        .chain([PI - 1e-9, -PI + 1e-9, 1e-9, -1e-9])
}

#[test]
fn from_angle_and_angle_to_invert_each_other_in_the_plane() {
    let (e1, x) = (
        vga2d::Vector::<(), f64>::new(1.0, 0.0),
        pga2d::Point::direction(1.0, 0.0),
    );
    for a in angles() {
        let v = vga2d::Vector::from_angle(a);
        let d = pga2d::Point::direction_from_angle(a);
        // The half turn reads as π or -π: the same direction.
        let wrapped = |b: f64| {
            if (b - a).abs() > PI {
                b - 2.0 * PI * b.signum()
            } else {
                b
            }
        };
        assert!((wrapped(e1.angle_to(v)) - a).abs() < 1e-15, "{a}");
        assert!((wrapped(x.angle_to(d)) - a).abs() < 1e-15, "{a}");
        assert!((v.norm() - 1.0).abs() < 1e-15 && (d.ideal_norm() - 1.0).abs() < 1e-15);
    }
}

#[test]
fn the_unsigned_angle_is_the_size_of_the_signed_one_and_the_same_in_every_algebra() {
    for a in angles() {
        // Lengths differ: no normalizing is needed.
        let (p, q) = (
            vga2d::Vector::new(2.0, 0.0),
            vga2d::Vector::from_angle(a).gp(0.5),
        );
        let want = p.angle_to(q).abs();
        assert!((p.angle_between(q) - want).abs() < 1e-15, "{a}");
        let (dp, dq) = (
            pga2d::Point::direction(p.e1(), p.e2()),
            pga2d::Point::direction(q.e1(), q.e2()),
        );
        assert!((dp.angle_between(dq) - want).abs() < 1e-15, "{a}");
        // The same pair in a plane of space, tilted, in VGA and in PGA.
        let tilt =
            |v: vga2d::Vector<(), f64>| vga3d::Vector::new(v.e1(), v.e2() * 0.6, v.e2() * 0.8);
        let (sp, sq) = (tilt(p), tilt(q));
        assert!((sp.angle_between(sq) - want).abs() < 1e-14, "{a}");
        let direction = |v: vga3d::Vector<(), f64>| pga3d::Point::direction(v.e1(), v.e2(), v.e3());
        assert!(
            (direction(sp).angle_between(direction(sq)) - want).abs() < 1e-14,
            "{a}"
        );
    }
}

#[test]
fn small_angles_and_near_half_turns_keep_their_digits() {
    // An arc cosine of the normalized inner product would lose half of them here: near 0 the
    // angle keeps its relative precision, near π its absolute precision (an angle near π is
    // itself only representable to a few ulps of π).
    for tiny in [1e-6, 1e-9, 1e-12] {
        let a = vga3d::Vector::<(), f64>::new(1.0, 0.0, 0.0);
        let b = vga3d::Vector::new(1.0, tiny, 0.0);
        assert!((a.angle_between(b) / tiny - 1.0).abs() < 1e-12, "{tiny}");
        let c = vga3d::Vector::new(-1.0, tiny, 0.0);
        assert!(
            (a.angle_between(c) - (PI - tiny)).abs() < 4.0 * f64::EPSILON,
            "{tiny}"
        );
    }
    let quarter = pga3d::Point::<(), f64>::direction(0.0, 3.0, 0.0);
    let x = pga3d::Point::direction(1.0, 0.0, 0.0);
    assert!((x.angle_between(quarter) - FRAC_PI_2).abs() < 1e-15);
}

#[test]
fn rotation_between_turns_by_angle_to() {
    for a in angles() {
        let from = pga2d::Point::<(), f64>::direction(0.0, 2.0);
        let to = pga2d::Motor::rotation(pga2d::Point::xy(0.0, 0.0), a) >> from;
        let r = pga2d::Motor::rotation_between(from, to);
        assert!((r.angle() - from.angle_to(to)).abs() < 1e-12, "{a}");
        let turned = r >> from;
        assert!((turned - to).ideal_norm() < 1e-12, "{a}");
    }
}

#[test]
fn finite_values_and_maps() {
    assert!(pga3d::Point::<(), f64>::xyz(1.0, -2.0, 3.0).is_finite());
    assert!(pga3d::Point::<(), f64>::slot().is_finite());
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(
            !pga3d::Point::<(), f64>::new(0.0, bad, 0.0, 1.0).is_finite(),
            "{bad}"
        );
        let mut map = pga3d::Point::<(), f64>::slot();
        map.c[2][1] = bad;
        assert!(!map.is_finite(), "{bad}");
    }
    assert!(vga2d::Vector::<(), f32>::new(f32::MAX, f32::MIN).is_finite());
}
