//! numga's tests of the tippe top, its scenario check, and its numbers.

use super::contact::lowest_point;
use super::tippe::*;
use gax::motions::Motions;
use gax::pga3d::{Direction, Line, Motor, Point};
use gax_numga_examples::shared::mechanics_lie::rigid::G;
use std::sync::OnceLock;

/// The body, built once for every test.
fn model() -> &'static Body {
    static MODEL: OnceLock<Body> = OnceLock::new();
    MODEL.get_or_init(body)
}

/// The body as numga builds it: its mass, a point weighted by the volume, and its inertia's
/// pairings with the unit rates.
#[test]
fn the_body_is_numgas() {
    let b = model();
    let numga = [
        0.053_486_132_812_49,
        0.043_059_960_937_49,
        -0.436_993_945_312_51,
        3.224_390_625,
    ];
    assert!(
        b.mass
            .c
            .iter()
            .zip(numga)
            .all(|(a, n)| (a - n).abs() < 1e-11),
        "{:?}",
        b.mass
    );
    let unit = |k: usize| {
        Line::<(), f64>::from_coeffs(core::array::from_fn(|i| f64::from(u8::from(i == k))))
    };
    let pairings: Vec<f64> = (0..6)
        .map(|k| (unit(k) & b.inertia.of(unit(k))).s())
        .collect();
    let numga = [
        1.291_405_454_101_556_7,
        1.438_796_704_101_555_7,
        1.477_293_520_507_801_5,
        3.224_390_625,
        3.224_390_625,
        3.224_390_625,
    ];
    for (a, n) in pairings.iter().zip(numga) {
        assert!((a - n).abs() < 1e-10, "{pairings:?}");
    }
}

/// The contacts lie on their parts with horizontal tangent planes, for normals all round the
/// sphere; the cavity never removes the shell's exposed support, and the stem lies behind the
/// contacts.
#[test]
fn contacts_lie_on_the_parts_and_have_horizontal_tangents() {
    let b = model();
    let [x, y, z] = axes();
    let (n_latitudes, n_longitudes) = (16, 32);
    for i in 0..n_latitudes {
        let latitude = (i as f64 + 0.5) * core::f64::consts::PI / n_latitudes as f64;
        for j in 0..n_longitudes {
            let longitude = j as f64 * core::f64::consts::TAU / n_longitudes as f64;
            let turns = turn(x, y, longitude) * turn(x, z, latitude);
            let normal = (turns >> Point::from(up())).cast::<Direction>();
            let points = b.contacts().map(|q| lowest_point(q, normal));
            for (q, p) in b.contacts().iter().zip(points) {
                assert!((q.of(p) & p).s().abs() < 1e-10);
                let tangent = q.of(p).dual();
                let direction = tangent - origin() * (w() & tangent).s();
                let across = direction & Point::from(normal);
                assert!(across.c.iter().all(|c| c.abs() < 1e-10), "{across:?}");
            }
            let heights = points.map(|p| (Point::from(normal).undual() & p).s());
            let cut = (b.cavity.of(points[0]) & points[0]).s() < 0.0;
            assert!(!(cut && heights[0] < heights[1]));
            let stem = lowest_point(b.stem, normal);
            let lowest = heights[0].min(heights[1]);
            assert!((Point::from(normal).undual() & stem).s() >= lowest - 1e-10);
        }
    }
}

/// Over a frame of steps gravity and friction exchange and dissipate energy: some, not much,
/// is lost; and with the table far below, the centre of mass follows its parabola.
#[test]
fn gravity_and_friction_transfer_and_dissipate_energy() {
    let b = model();
    for (motor, rate) in launch(b, &SPINS, LEAN) {
        let initial = energy(b, motor, rate, GRAVITY);
        let (end_motor, end_rate) =
            advance(b, motor, rate, GRAVITY, FRICTION, STEP, n_steps_per_frame());
        let last = energy(b, end_motor, end_rate, GRAVITY);
        assert!(last < initial && last > 0.9 * initial, "{initial} {last}");
    }
    let (motor, rate) = launch(b, &SPINS, LEAN)[0];
    let (height, duration, n_steps) = (10.0, 0.1, 25);
    let airborne = Motor::translation(Point::from(up() * height)) * motor;
    let (end, _) = advance(
        b,
        airborne,
        rate * 0.0,
        GRAVITY,
        [0.0, 0.0],
        duration / n_steps as f64,
        n_steps,
    );
    let centre = |m: gax::Unit<Motor>| (m >> b.mass).unitized();
    let fall = (axes()[2] & (centre(end) - centre(airborne))).s();
    assert!(
        (fall + GRAVITY * duration * duration / 2.0).abs() < 3e-4,
        "{fall}"
    );
}

/// Mirroring the body and its launch in the plane across `y` mirrors the motion: the centre,
/// the axis and the rate after a frame of steps. (numga also checks its batch against single
/// launches; the port steps each launch alone.)
#[test]
fn mirroring_the_cavity_and_launch_mirrors_the_motion() {
    let b = model();
    let mirror = axes()[1];
    let reflect = |q: super::contact::Quadric| mirror >> q.of(mirror << Point::slot());
    let inertia = mirror >> b.inertia.of(mirror << Line::slot());
    let reflected = Body {
        shell: reflect(b.shell),
        cavity: reflect(b.cavity),
        stem: reflect(b.stem),
        tip: reflect(b.tip),
        mass: mirror >> b.mass,
        inertia,
        inertia_inv: G::mobility(inertia),
    };
    let (motor, rate) = launch(b, &SPINS[..1], LEAN)[0];
    // This lean lies in the mirror plane, so both placements coincide at release.
    let n = n_steps_per_frame();
    let (m0, r0) = advance(b, motor, rate, GRAVITY, FRICTION, STEP, n);
    let (m1, r1) = advance(
        &reflected,
        motor,
        mirror >> rate,
        GRAVITY,
        FRICTION,
        STEP,
        n,
    );
    let close = |a: &[f64], b: &[f64]| a.iter().zip(b).all(|(p, q)| (p - q).abs() < 1e-8);
    assert!(close(&(mirror >> (m0 >> origin())).c, &(m1 >> origin()).c));
    let axis = |m: gax::Unit<Motor>| m >> Point::from(up());
    assert!(close(&(mirror >> axis(m0)).c, &axis(m1).c));
    assert!(close(&(mirror >> r0).c, &r1.c), "{:?} {r1:?}", mirror >> r0);
}

/// numga's scenario check over the 80 s run: the top spun clockwise turns over onto its tip,
/// its axis pointing down, while the one spun counterclockwise stays upright; and the run
/// follows numga's, state by state: to 1e-6 up to the flip, to 1e-3 after it.
#[test]
fn the_top_spun_one_way_turns_over() {
    let runs = evolve(model());
    let vertical = runs
        .each_ref()
        .map(|r| r.iter().map(|(m, _)| vertical(*m)).collect::<Vec<_>>());
    let last = n_states() - 1;
    assert!(vertical[0][last] < -0.8 && vertical[1][last] > 0.8);
    let numga = [
        (10, [0.883_551_234_190_73, 0.905_211_877_073_82]),
        (100, [0.996_915_505_105_21, 0.948_373_428_339_55]),
        (300, [0.374_340_713_117_34, 0.956_767_567_953_2]),
        (400, [-0.961_696_175_797_54, 0.952_541_632_420_71]),
        (500, [-0.938_740_892_584_3, 0.944_036_576_124_77]),
    ];
    for (k, values) in numga {
        for (side, value) in values.iter().enumerate() {
            let ours = vertical[side][k];
            // Through the flip (about state 330) the motion amplifies rounding: the port and
            // numga order their sums differently, and part by a few 1e-5 after it.
            let tolerance = if k <= 300 { 1e-6 } else { 1e-3 };
            assert!(
                (ours - value).abs() < tolerance,
                "state {k}, top {side}: {ours} {value}"
            );
        }
    }
}

#[test]
fn a_frame_draws() {
    let b = *model();
    let starts = launch(&b, &SPINS, LEAN);
    let motors = starts.iter().map(|(m, _)| vec![*m]).collect::<Vec<_>>();
    let scenes = super::Scenes {
        body: b,
        motors: [motors[0].clone(), motors[1].clone()],
        vertical: [vec![1.0], vec![1.0]],
        travel: core::array::from_fn(|_| vec![gax_numga_examples::Point2::xy(0.0, 0.0)]),
    };
    gax_numga_examples::app::assert_draws_with(super::draw, &scenes, 0.0);
}
