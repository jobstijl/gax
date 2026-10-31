//! numga's tests of the rattleback, its scenario checks, and its numbers.

use super::rattleback::*;
use gax::pga3d::{Direction, Line, Motor, OriginPlane, Point, Spin};
use gax::vga3d;
use std::sync::OnceLock;

/// The body, built once for every test.
fn model() -> &'static Rattleback {
    static MODEL: OnceLock<Rattleback> = OnceLock::new();
    MODEL.get_or_init(rattleback)
}

fn small<const N: usize>(c: &[f64; N], tol: f64) -> bool {
    c.iter().all(|x| x.abs() < tol)
}

/// The body as numga builds it: its grid finds the same 4799 points, the shell's centre lies
/// where numga's does from the centre of mass, and its inertia pairs the unit rates as numga's.
#[test]
fn the_body_is_numgas() {
    let r = model();
    assert_eq!(r.samples.len(), 4799);
    let [x, y, z] = mirrors();
    let offset = [x, y, z].map(|m| (m & r.centre).s());
    let numga = [0.001_865_944_64, 0.001_196_526_954, 0.126_921_648_616];
    assert!(
        offset.iter().zip(numga).all(|(a, b)| (a - b).abs() < 1e-11),
        "{offset:?}"
    );
    let unit = |k: usize| {
        Line::<(), f64>::from_coeffs(core::array::from_fn(|i| f64::from(u8::from(i == k))))
    };
    let pairings: Vec<f64> = (0..6)
        .map(|k| (unit(k) & r.inertia.of(unit(k))).s())
        .collect();
    let numga = [
        1.865_203_877_144_599,
        20.687_390_877_721_06,
        21.635_060_639_004_66,
        4.0,
        4.0,
        4.0,
    ];
    for (a, b) in pairings.iter().zip(numga) {
        assert!((a - b).abs() < 1e-11, "{pairings:?}");
    }
}

/// numga's translation `exp(d origin⁻¹ / 2)` is gax's `Motor::translation(d)`.
#[test]
fn the_translation_to_a_point() {
    let d = direction(0.3, -1.2, 0.7);
    let numga = (Point::from(d) * origin().inverse() * 0.5)
        .cast::<Line>()
        .exp();
    let gax = Motor::translation(Point::from(d));
    assert!(small(&(numga.into_inner() - gax.into_inner()).c, 1e-15));
}

/// Rolling keeps the touching point still, and the adjugate of the rolling map sends any
/// reaction at the contact to nothing: it does no work under rolling.
#[test]
fn rolling_annihilates_contact_motion_and_contact_work() {
    let r = model();
    let [x, y, z] = mirrors();
    let normal = (x + y * 2.0 + z * 8.0).normalized().cast::<OriginPlane>();
    #[allow(clippy::disallowed_methods)] // the support function
    let height = (normal & r.shape.of(normal)).s().sqrt();
    let contact = r.centre - r.shape.of(normal) * (1.0 / height);
    let rolling = Motor::translation(Point::from(contact)) >> Spin::slot();
    let point = origin() + Point::from(contact);
    let reaction = point & (x + y * 2.0 + z * 3.0).dual();
    assert!(
        point
            .commutator(rolling)
            .c
            .iter()
            .flatten()
            .all(|c| c.abs() < 1e-11)
    );
    let work = rolling.adjugate().of(reaction);
    assert!(small(&work.c, 1e-11), "{work:?}");
}

/// The reduced balance: the work rate of the acceleration's momentum change is the loss of
/// potential energy, and it matches Newton and Euler's two balances with the contact's reaction
/// eliminated by hand (numga's with numpy arrays; here VGA3D's vectors, cross products the
/// duals of wedges, and the 3 × 3 solve a map's).
#[test]
fn reduced_balance_matches_newton_euler_and_has_zero_energy_derivative() {
    let r = model();
    let [x, y, z] = mirrors();
    let motor = ((x ^ z) * 0.21 + (y ^ z) * 0.12 + (x ^ y) * 0.07).exp();
    let spin = ((y ^ z) * 0.3 - (x ^ z) * -0.7 + (x ^ y) * 1.2).cast::<Spin>();
    let (rate, acceleration) = dynamics(
        motor,
        spin,
        r.shape,
        r.centre,
        r.inertia,
        MASS,
        gravity(),
        up(),
    );
    let Contact {
        normal,
        support,
        height,
        rolling,
    } = contact_geometry(motor, r.shape, r.centre, up());
    let normal_rate = spin.commutator(normal).cast::<OriginPlane>();
    let height_rate = (normal & r.shape.of(normal_rate)).s() / height;
    let contact_rate = r.shape.of(normal_rate) * (-1.0 / height) - support * (height_rate / height);
    let rate_change = rolling.of(acceleration)
        + (Point::from(contact_rate) * origin().inverse())
            .commutator(spin)
            .cast::<Line>();
    let power = (rate_change & r.inertia.of(rate)).s();
    // The centre of mass rises as the shell's centre does, less as the shell's centre turns
    // about it.
    let rise = height_rate - (normal_rate & r.centre).s();
    assert!(
        (power + MASS * GRAVITY * rise).abs() < 1e-10,
        "{power} {rise}"
    );

    // Newton and Euler, by hand: vectors of VGA3D, the cross product the dual of the wedge.
    let v = |a: [f64; 3]| vga3d::Vector::new(a[0], a[1], a[2]);
    // The cross product is an axial vector, intrinsic (ADR-051); numpy's arrays carry no
    // orientation, so the reference relabels it.
    let cross =
        |a: vga3d::Vector, b: vga3d::Vector| (a ^ b).dual().with_orientation::<gax::Extrinsic>();
    let dot = |a: vga3d::Vector, b: vga3d::Vector| (a | b).s();
    let axes = [y ^ z, -(x ^ z), x ^ y];
    let moment = vga3d::Vector::<(vga3d::Vector,)>::from_images(
        axes.map(|j| v(axes.map(|i| (i & r.inertia.of(j)).s()))),
    );
    let offset = v([x, y, z].map(|m| (m & r.centre).s()));
    let n = v(normal.c);
    let omega = v([0.3, -0.7, 1.2]);
    let lengths_squared = SEMI_AXES.map(|a| a * a);
    let stretch = |a: vga3d::Vector| v(core::array::from_fn(|k| lengths_squared[k] * a.c[k]));
    #[allow(clippy::disallowed_methods)] // the support function
    let h = dot(n, stretch(n)).sqrt();
    let lever = offset - stretch(n) * (1.0 / h);
    let normal_change = cross(n, omega);
    let support_array = stretch(n) * (-1.0 / h);
    let lever_change = stretch(normal_change) * (-1.0 / h)
        - support_array * (dot(n, stretch(normal_change)) / (h * h));
    let velocity = cross(lever, omega);
    let contact_moment = moment
        + (vga3d::Vector::slot() * dot(lever, lever) - lever * (lever | vga3d::Vector::slot()))
            * MASS;
    let torque = cross(moment.of(omega), omega) + cross(lever, cross(lever_change, omega)) * MASS
        - cross(lever, cross(velocity, omega)) * MASS
        + cross(lever, n) * (MASS * GRAVITY);
    let expected = contact_moment.solve(torque);
    let actual = Line::from(acceleration);
    let actual = [actual.e23(), actual.e31(), actual.e12()];
    for (a, b) in actual.iter().zip(expected.c) {
        assert!(
            (a - b).abs() <= 1e-11 + 1e-11 * b.abs(),
            "{actual:?} {expected:?}"
        );
    }
}

/// Mirroring every input of the dynamics mirrors its outputs: the rolling rate and the angular
/// acceleration, under gax's oriented sandwich (ADR-051).
#[test]
fn mirroring_the_body_mirrors_its_rolling_dynamics() {
    let r = model();
    let [x, y, z] = mirrors();
    let motor = ((x ^ z) * 0.13 + (y ^ z) * 0.07 + (x ^ y) * 0.2).exp();
    let spin = ((x ^ y) * -1.5 + (y ^ z) * 0.2).cast::<Spin>();
    let (rate, acceleration) = dynamics(
        motor,
        spin,
        r.shape,
        r.centre,
        r.inertia,
        MASS,
        gravity(),
        up(),
    );
    let mirror = (x + y * (1.0 / 3.0) + z * 0.2).normalized();
    let flip = |d: Direction| (mirror >> Point::from(d)).cast::<Direction>();
    let shape = (mirror
        >> r.shape
            .of((mirror << OriginPlane::slot()).cast::<OriginPlane>()))
    .cast::<Direction>();
    let inertia = mirror >> r.inertia.of(mirror << Line::slot());
    let mirrored_motor = gax::Unit::new_unchecked(mirror >> motor.into_inner());
    let mirrored_spin = (mirror >> Line::from(spin)).cast::<Spin>();
    let (mirrored_rate, mirrored_acceleration) = dynamics(
        mirrored_motor,
        mirrored_spin,
        shape,
        flip(r.centre),
        inertia,
        MASS,
        flip(gravity()),
        flip(up()),
    );
    let close = |a: Line, b: Line| {
        a.c.iter()
            .zip(b.c)
            .all(|(p, q)| (p - q).abs() <= 1e-10 + 1e-10 * q.abs())
    };
    assert!(
        close(mirrored_rate, mirror >> rate),
        "{mirrored_rate:?} {:?}",
        mirror >> rate
    );
    assert!(close(
        Line::from(mirrored_acceleration),
        mirror >> Line::from(acceleration)
    ));
}

/// The energy is kept to within the step's error, which falls as the fourth power of the step;
/// the tipped body spins up from rest, the spinning one reverses; and the table only ever
/// touches the painted underside: the touching point lies inside the deck and the cuts and
/// outside the balls, every other quadric of the wood.
#[test]
fn spin_reverses_and_energy_converges_with_the_time_step() {
    let r = model();
    let duration = 10.0;
    let error = |launch: &Launch| {
        let first = launch.energies[0];
        launch
            .energies
            .iter()
            .map(|e| (e / first - 1.0).abs())
            .fold(0.0, f64::max)
    };
    let coarse = launches(r, 0.05, (duration / 0.05) as usize);
    let fine = launches(r, 0.025, (duration / 0.025) as usize);
    for k in 0..2 {
        let (c, f) = (error(&coarse[k]), error(&fine[k]));
        assert!(f < 0.001 && f < c / 4.0, "{k}: {c} {f}");
    }
    let [tipped, spun] = fine.each_ref().map(Launch::spin_about_vertical);
    assert!(
        tipped[0].abs() < 1e-11 && tipped[tipped.len() - 1] > 1.4,
        "{}",
        tipped[tipped.len() - 1]
    );
    assert!(
        spun[0] < -1.4 && spun[spun.len() - 1] > 1.4,
        "{}",
        spun[spun.len() - 1]
    );
    let others = &r.body.wood()[1..];
    for launch in &fine {
        for m in &launch.motors {
            let support = contact_geometry(*m, r.shape, r.centre, up()).support;
            let contact = origin() + Point::from(r.centre + support);
            for q in others {
                assert!((q.of(contact) & contact).s() < 0.0);
            }
        }
    }
}

/// The scene's checks, numga's numbers: the tipped body spins up from rest to numga's spin, the
/// spinning one reverses to numga's, and the energy is kept.
#[test]
fn the_scenes_rock_into_spin_and_reverse() {
    let launches = launches(model(), TIME_STEP, N_STEPS);
    let spins = launches.each_ref().map(Launch::spin_about_vertical);
    assert!(spins[0][0].abs() < 1e-11 && spins[1][0] < -1.4);
    let numga = [1.491_221_036_576, 1.484_589_636_143];
    for (s, n) in spins.iter().zip(numga) {
        assert!((s[N_STEPS] - n).abs() < 1e-6, "{} {n}", s[N_STEPS]);
    }
    let first = [58.476_839_610_683, 58.681_684_347_495];
    for (launch, e) in launches.iter().zip(first) {
        assert!(
            (launch.energies[0] - e).abs() < 1e-9,
            "{}",
            launch.energies[0]
        );
        assert!(launch.energies.iter().all(|x| (x / e - 1.0).abs() < 1e-2));
    }
}

#[test]
fn a_frame_draws() {
    gax_numga_examples::app::assert_draws_with(super::draw, &super::scenes(), 3.0);
}
