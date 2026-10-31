//! Thomas precession's mathematics, numga's `core.py` and `scenarios.py`: boosts as rotors,
//! velocities on the hyperboloid, spins carried by pure boosts, and the frame of a particle with
//! a magnetic moment turned by the Bargmann-Michel-Telegdi bivector.

use super::spacetime::{
    Bivector, Direction, Even, Vector, in_space, intrinsic, space, t, time, x, y, z,
};

/// A velocity, an event or a direction: an intrinsic trivector (numga's antivector).
pub type Velocity = Direction;
/// A spin: the patch of the plane it turns in, an intrinsic bivector (numga's antibivector).
pub type Spin = intrinsic::Bivector;
/// A boost or a turn.
pub type Rotor = Even;

/// The plane `xy`.
pub fn xy() -> Bivector {
    (x() ^ y()).cast::<Bivector>()
}

/// The rotor `exp(b)`.
pub fn exp(b: Bivector) -> Rotor {
    b.exp().into_inner()
}

/// The pure boost by `rapidity` along a unit spatial direction: a turn in the plane of the
/// direction and time, their join, which the Hodge dual turns into a generator.
pub fn boost(direction: Direction, rapidity: f64) -> Rotor {
    exp((direction & time()).hodge().gp(rapidity / 2.0))
}

/// The pure boost carrying the lab's time direction to `velocity`.
pub fn rest_boost(velocity: Velocity) -> Rotor {
    (velocity * time().inverse()).sqrt().into_inner()
}

/// A spin of the frame moving at `velocity`, as the observer at rest reads it: after the pure
/// boost back to rest.
pub fn lab_view(velocity: Velocity, spin: Spin) -> Spin {
    (time() * velocity.inverse()).sqrt() >> spin
}

/// The pure boost from velocity `tail` to velocity `head`, in their plane.
pub fn transport(tail: Velocity, head: Velocity) -> Rotor {
    (head * tail.inverse()).sqrt().into_inner()
}

/// The velocities along the pure boost from `tail` to `head`, at `samples` fractions of the way
/// from none to all.
pub fn geodesic(tail: Velocity, head: Velocity, samples: usize) -> Vec<Velocity> {
    let generator = (head * tail.inverse()).sqrt().log();
    (0..samples)
        .map(|k| exp(generator.gp(k as f64 / (samples - 1) as f64)) >> tail)
        .collect()
}

/// The velocity whose spatial part is `spatial`: the unit direction in time above it.
pub fn on_hyperboloid(spatial: Direction) -> Velocity {
    // The mass shell: the energy √(1 + p²) of the momentum per unit mass.
    #[allow(clippy::disallowed_methods)]
    let height = (1.0 + (spatial * spatial).s()).sqrt();
    spatial + time().gp(height)
}

/// A spin carried along a path of velocities by the pure boost between each velocity and the
/// next, torque-free: the spin at each velocity.
pub fn carried(velocities: &[Velocity], spin: Spin) -> Vec<Spin> {
    let mut spins = vec![spin];
    for w in velocities.windows(2) {
        let last = spins[spins.len() - 1];
        spins.push(transport(w[0], w[1]) >> last);
    }
    spins
}

/// The bivector turning the frame of a particle of magnetic anomaly `anomaly`, `g / 2 - 1`:
/// its velocity by the Lorentz force, and its spin by the Bargmann-Michel-Telegdi equation.
pub fn bmt_generator(field: Bivector, velocity: Velocity, anomaly: f64) -> Bivector {
    let force: Vector = velocity & field;
    // The particle's own slice of space, which reads its velocity as one: minus the velocity's
    // Hodge dual, as for the lab's `time`.
    let rest_slice: Vector = -velocity.hodge();
    // On the velocity the plane of the slice and the force acts as the field does, so the
    // anomaly cancels there.
    field * (1.0 + anomaly) - (rest_slice ^ force) * anomaly
}

/// A storage ring's field at an event: a uniform magnetic field in the orbit's plane, turning
/// against `xy` so that it bends a positive charge counterclockwise, and an electric field of
/// strength `electric` pointing away from the ring's axis.
pub fn ring_field(event: Velocity, magnetic: f64, electric: f64) -> Bivector {
    let outward = in_space(event).normalized();
    xy() * -magnetic + (outward & time()).hodge() * electric
}

/// One midpoint step in proper time of a particle's event and frame in the ring's field.
pub fn bmt_step(
    event: Velocity,
    frame: Rotor,
    ring: (f64, f64),
    anomaly: f64,
    proper_step: f64,
) -> (Velocity, Rotor) {
    let (magnetic, electric) = ring;
    let rate = bmt_generator(
        ring_field(event, magnetic, electric),
        frame >> time(),
        anomaly,
    );
    let halfway_frame = exp(rate.gp(proper_step / 4.0)) * frame;
    let halfway_event = event + (frame >> time()).gp(proper_step / 2.0);
    let halfway_rate = bmt_generator(
        ring_field(halfway_event, magnetic, electric),
        halfway_frame >> time(),
        anomaly,
    );
    (
        event + (halfway_frame >> time()).gp(proper_step),
        exp(halfway_rate.gp(proper_step / 2.0)) * frame,
    )
}

/// The particle's speed at Lorentz factor `gamma`.
#[allow(clippy::disallowed_methods)] // the speed √(1 - 1/γ²) of a Lorentz factor, a law
pub fn speed(gamma: f64) -> f64 {
    (1.0 - 1.0 / (gamma * gamma)).sqrt()
}

/// The radius of the circular orbit at Lorentz factor `gamma`, where the magnetic pull inwards
/// less the electric push outwards turns the momentum.
pub fn ring_radius(gamma: f64, magnetic: f64, electric: f64) -> f64 {
    let speed = speed(gamma);
    gamma * speed * speed / (speed * magnetic - electric)
}

/// The angle in the orbit's plane from a spin along the momentum to the spin, as the observer
/// at rest reads both.
pub fn anomalous_angle(velocity: Velocity, spin: Spin) -> f64 {
    let direction = in_space(velocity).normalized();
    // A spin along the momentum: the patch spanned by the vertical and the direction across the
    // motion, a quarter turn in the orbit's plane.
    let along: Spin = direction.commutator(xy()) & z().hodge();
    // The product of two such patches is the rotor through their angle, read by pairing with
    // the orbit's patch.
    let turn = (along.inverse() * lab_view(velocity, spin)).unit().log();
    (turn & xy().hodge()).s()
}

// --- the scenes -------------------------------------------------------------------------

/// The rocket's two equal boosts, their rapidity swelling and shrinking between two values over
/// the loop, and the samples along each side of the triangle of velocities.
pub const BOOST_FRAMES: usize = 120 / 2 + 1;
pub const MIN_RAPIDITY: f64 = 0.8;
pub const MAX_RAPIDITY: f64 = 2.0;
pub const EDGE_SAMPLES: usize = 32;
/// Lorentz factors of the circular orbits, their steps around, and the steps between frames.
pub const ORBIT_GAMMAS: [f64; 3] = [1.25, 1.5, 1.75];
pub const ORBIT_STEPS: usize = 450;
pub const ORBIT_STRIDE: usize = 4;
/// The box's half width, the gyroscope's momentum per unit mass along each axis, the frames of
/// each flight and the steps of each bounce, and the round trips shown.
pub const BOX_HALF_WIDTH: f64 = 1.0;
pub const BOX_MOMENTUM: f64 = 0.8;
pub const FLIGHT_FRAMES: usize = 13;
pub const BOUNCE_STEPS: usize = 12;
pub const LOOPS: usize = 3;
/// The storage ring: the muon's anomaly exaggerated so the spin laps the momentum in a few
/// turns, the magnetic field, the animation's ring purely magnetic, its turns, steps per turn
/// and between frames.
pub const MUON_ANOMALY: f64 = 0.1;
pub const RING_MAGNETIC: f64 = 1.0;
pub const RING_TURNS: usize = 3;
pub const RING_TURN_STEPS: usize = 300;
pub const RING_STRIDE: usize = 6;
/// The electric fields swept at two Lorentz factors, the magic one last, over a fixed time in
/// the lab.
pub const ELECTRIC_FIELDS: usize = 7;
pub const SWEEP_LAB_DURATION: f64 = 10.0;
pub const SWEEP_STEPS: usize = 200;

/// The gyroscope's spin: the patch of the plane it turns in, `yz`.
pub fn gyroscope_spin() -> Spin {
    (y() ^ z()).cast::<Bivector>().hodge()
}

/// The muon's spin, turning in `zx`, across its first momentum along `y`.
pub fn muon_spin() -> Spin {
    (z() ^ x()).cast::<Bivector>().hodge()
}

/// The magic Lorentz factor of an anomaly, `√(1 + 1/a)`.
#[allow(clippy::disallowed_methods)] // the magic factor's definition, γ² = 1 + 1/a
pub fn magic(anomaly: f64) -> f64 {
    (1.0 + 1.0 / anomaly).sqrt()
}

/// The swell of the rocket's rapidity at each frame.
pub fn rapidities() -> Vec<f64> {
    (0..BOOST_FRAMES)
        .map(|k| core::f64::consts::PI * k as f64 / (BOOST_FRAMES - 1) as f64)
        .map(|a| {
            MIN_RAPIDITY + (MAX_RAPIDITY - MIN_RAPIDITY) * gax_numga_examples::signal::haversine(a)
        })
        .collect()
}

/// A frame of the rocket's two boosts.
pub struct TwoBoosts {
    /// The triangle of velocities: rest to the first boost to the second, and straight back.
    pub triangle: Vec<Velocity>,
    /// Its corners.
    pub corners: [Velocity; 3],
    /// The rocket's `x` axis at each corner, as the observer at rest reads it.
    pub axes: [Direction; 3],
}

/// The rocket boosting along its own `x` and then its own `y` by a rapidity, and straight back
/// to rest.
pub fn two_boosts(rapidity: f64, samples: usize) -> TwoBoosts {
    let [ax, ay, _] = space();
    let first = boost(ax, rapidity);
    let rocket = first * boost(ay, rapidity);
    let (corner, velocity) = (first >> time(), rocket >> time());
    let triangle = [(time(), corner), (corner, velocity), (velocity, time())]
        .into_iter()
        .flat_map(|(a, b)| geodesic(a, b, samples))
        .collect();
    TwoBoosts {
        triangle,
        corners: [time(), corner, velocity],
        axes: [
            ax,
            rest_boost(corner).reverse() >> (first >> ax),
            rest_boost(velocity).reverse() >> (rocket >> ax),
        ],
    }
}

/// The turn a rocket's two equal boosts along its own `x` and `y` leave behind: their product
/// after the straight boost from rest to the same velocity is undone.
pub fn wigner_turn(rapidity: f64) -> Rotor {
    let [ax, ay, _] = space();
    let rocket = boost(ax, rapidity) * boost(ay, rapidity);
    rest_boost(rocket >> time()).reverse() * rocket
}

/// Twice the turn's angle in `xy`: twice its logarithm's scalar product with `xy`.
pub fn wigner_angle(rapidity: f64) -> f64 {
    2.0 * wigner_turn(rapidity).unit().log().scalar_product(xy()).s()
}

/// Velocities around the circular orbit at a Lorentz factor, counterclockwise from moving along
/// `y`, at `steps + 1` phases.
pub fn orbit_velocities(gamma: f64, steps: usize) -> Vec<Velocity> {
    (0..=steps)
        .map(|k| core::f64::consts::TAU * k as f64 / steps as f64)
        .map(|phase| {
            let direction = exp(xy().gp(-phase / 2.0)) >> space()[1];
            boost(direction, gamma.acosh()) >> time()
        })
        .collect()
}

/// The spin at every velocity of an orbit, carried from the gyroscope's spin in the orbit's
/// first frame, as the observer at rest reads it.
pub fn orbit_spins(velocities: &[Velocity]) -> Vec<Spin> {
    let start = rest_boost(velocities[0]) >> gyroscope_spin();
    carried(velocities, start)
        .into_iter()
        .zip(velocities)
        .map(|(s, v)| lab_view(*v, s))
        .collect()
}

/// One round trip of a particle bouncing along a diamond between the midpoints of a square
/// box's walls: its positions and velocities. In flight the velocity holds; at a wall the
/// momentum along the wall holds and the momentum across it reverses.
pub fn box_path(
    half_width: f64,
    momentum: f64,
    flight: usize,
    bounce: usize,
) -> (Vec<Velocity>, Vec<Velocity>) {
    let [ax, ay, _] = space();
    // Quarter turns from one before the first wall to one past the last.
    let quarter = |k: i32| exp(xy().gp(-f64::from(k) * core::f64::consts::FRAC_PI_4));
    let walls: Vec<Direction> = (-1..5).map(|k| quarter(k) >> ax.gp(half_width)).collect();
    let corners: Vec<Direction> = (-1..5)
        .map(|k| quarter(k) >> (ax + ay).gp(momentum))
        .collect();
    let share = |k: usize, n: usize| k as f64 / (n - 1) as f64;
    let (mut positions, mut velocities) = (Vec::new(), Vec::new());
    for wall in 0..4 {
        let (start, end) = (walls[wall], walls[wall + 1]);
        let (before, after) = (corners[wall + 1], corners[wall + 2]);
        for k in 0..flight {
            positions.push(start + (end - start).gp(share(k, flight)));
            velocities.push(on_hyperboloid(before));
        }
        for k in 0..bounce {
            positions.push(end);
            velocities.push(on_hyperboloid(
                before + (after - before).gp(share(k, bounce)),
            ));
        }
    }
    (positions, velocities)
}

/// The spin carried along the box's path, as the observer at rest reads it at each step.
pub fn bouncing(velocities: &[Velocity]) -> Vec<Spin> {
    let start = rest_boost(velocities[0]) >> gyroscope_spin();
    carried(velocities, start)
        .into_iter()
        .zip(velocities)
        .map(|(s, v)| lab_view(*v, s))
        .collect()
}

/// The area the square of momenta of side `2 a` encloses on the hyperboloid: four times the
/// integral of `1 / √(1 + x² + y²)` over a quarter of it.
#[allow(clippy::disallowed_methods)] // the closed form of the area, the reference
pub fn square_area(a: f64) -> f64 {
    4.0 * (2.0 * a * (a / (1.0 + a * a).sqrt()).asinh()
        - (a * a / (1.0 + 2.0 * a * a).sqrt()).atan())
}

/// A particle on the circular orbit at Lorentz factor `gamma`, at `x` and moving along `y`: its
/// event, and its frame, the pure boost from rest, so that the muon's spin, turning in `zx`,
/// lies along the momentum.
pub fn ring_start(gamma: f64, magnetic: f64, electric: f64) -> (Velocity, Rotor) {
    let event = space()[0].gp(ring_radius(gamma, magnetic, electric));
    (event, boost(space()[1], gamma.acosh()))
}

/// A frame of the storage ring: the particle's event, velocity, and spin as the observer at
/// rest reads it.
pub struct RingState {
    pub event: Velocity,
    pub velocity: Velocity,
    pub spin: Spin,
}

/// Every `stride`-th step of a particle's turns around the ring.
pub fn ring(
    gamma: f64,
    ring: (f64, f64),
    anomaly: f64,
    turns: usize,
    turn_steps: usize,
    stride: usize,
) -> Vec<RingState> {
    let (magnetic, electric) = ring;
    let radius = ring_radius(gamma, magnetic, electric);
    let proper_period = core::f64::consts::TAU * radius / (gamma * speed(gamma));
    let (mut event, mut frame) = ring_start(gamma, magnetic, electric);
    let mut states = Vec::new();
    for k in 0..=turns * turn_steps {
        if k % stride == 0 {
            let velocity = frame >> time();
            states.push(RingState {
                event,
                velocity,
                spin: lab_view(velocity, frame >> muon_spin()),
            });
        }
        (event, frame) = bmt_step(
            event,
            frame,
            ring,
            anomaly,
            proper_period / turn_steps as f64,
        );
    }
    states
}

/// The rate at which the spin turns away from the momentum, over the anomaly times the
/// magnetic field, at a Lorentz factor and an electric field: the angle reached after
/// `lab_duration` of the observer's time.
pub fn anomalous_rate(
    gamma: f64,
    ring: (f64, f64),
    anomaly: f64,
    lab_duration: f64,
    steps: usize,
) -> f64 {
    let (magnetic, electric) = ring;
    let (mut event, mut frame) = ring_start(gamma, magnetic, electric);
    let proper_step = lab_duration / (gamma * steps as f64);
    for _ in 0..steps {
        (event, frame) = bmt_step(event, frame, ring, anomaly, proper_step);
    }
    anomalous_angle(frame >> time(), frame >> muon_spin()) / (anomaly * magnetic * lab_duration)
}

/// The electric field of the sweep's `k`-th point, from -0.3 to 0.3.
pub fn electric_field(k: usize) -> f64 {
    -0.3 + 0.6 * k as f64 / (ELECTRIC_FIELDS - 1) as f64
}

/// The sweep's rates at a Lorentz factor of 2 and at the magic one, `[gammas][fields]`.
pub fn magic_sweep() -> [Vec<f64>; 2] {
    [2.0, magic(MUON_ANOMALY)].map(|gamma| {
        (0..ELECTRIC_FIELDS)
            .map(|k| {
                let ring = (RING_MAGNETIC, electric_field(k));
                anomalous_rate(gamma, ring, MUON_ANOMALY, SWEEP_LAB_DURATION, SWEEP_STEPS)
            })
            .collect()
    })
}

/// How far a direction runs along the slices `x` and `y`, its planar coordinates.
pub fn planar(d: Direction) -> [f64; 2] {
    [(x() & d).s(), (y() & d).s()]
}

/// A spin's arrow in the plane: its pairings with the planes `yz` and `zx`, the planes its
/// patch turns in when it points along `x` and along `y`.
pub fn spin_planar(spin: Spin) -> [f64; 2] {
    [y() ^ z(), z() ^ x()].map(|plane| (plane.cast::<Bivector>() & spin).s())
}

/// How far a direction runs in the lab's time.
pub fn lab_time(d: Direction) -> f64 {
    (t() & d).s()
}

#[cfg(test)]
mod tests {
    use super::*;
    use gax::ApproxEq;

    /// Closeness of spins, to an absolute tolerance.
    fn near(a: Spin, b: Spin, tol: f64) -> bool {
        a.abs_diff_eq(&b, tol)
    }

    /// numga's first test, and the scenario's at every frame of the swell: two equal
    /// perpendicular boosts leave a pure turn, by the closed form `-2 atan(tanh²(η/2))`.
    #[test]
    fn two_equal_perpendicular_boosts_leave_the_closed_form_turn() {
        for rapidity in [0.3, 1.0, 2.0].into_iter().chain(rapidities()) {
            let turn = wigner_turn(rapidity);
            assert!((turn >> time()).abs_diff_eq(&time(), 1e-12));
            let expected = -2.0 * (rapidity / 2.0).tanh().powi(2).atan();
            assert!(
                (wigner_angle(rapidity) - expected).abs() < 1e-12,
                "{rapidity}"
            );
        }
    }

    /// numga's scenario check: the turn is the area of the triangle of velocities, half a turn
    /// less twice the angle at rest.
    #[test]
    #[allow(clippy::disallowed_methods)] // the angle at rest, the reference
    fn the_turn_is_the_triangles_area() {
        for rapidity in rapidities() {
            let [ax, ay, _] = space();
            let last = (boost(ax, rapidity) * boost(ay, rapidity)) >> time();
            let [x, y] = planar(last);
            let expected = -(core::f64::consts::FRAC_PI_2 - 2.0 * y.atan2(x));
            assert!((wigner_angle(rapidity) - expected).abs() < 1e-12);
        }
    }

    /// numga's second test, and the scenario's at three speeds: around a circular orbit the
    /// spin turns back by the cap's area, `2π (γ - 1)`.
    #[test]
    fn a_circular_orbit_turns_the_spin_back_by_the_caps_area() {
        for (gammas, steps, tol) in [
            (&ORBIT_GAMMAS[..2], 360, 2e-3),
            (&ORBIT_GAMMAS[..], ORBIT_STEPS, 5e-3),
        ] {
            for gamma in gammas {
                let spins = orbit_spins(&orbit_velocities(*gamma, steps));
                let turn = exp(xy().gp(core::f64::consts::PI * (gamma - 1.0)));
                let expected = turn >> gyroscope_spin();
                assert!(near(spins[steps], expected, tol), "{gamma}");
            }
        }
    }

    /// numga's third test, and the scenario's three round trips: each round trip in the box
    /// turns the spin back by the area the square of momenta encloses on the hyperboloid.
    #[test]
    fn each_round_trip_in_the_box_turns_the_spin_back_by_its_area() {
        let (_, path) = box_path(1.0, BOX_MOMENTUM, 2, 60);
        let spins = bouncing(&path);
        let expected = exp(xy().gp(square_area(BOX_MOMENTUM) / 2.0)) >> gyroscope_spin();
        let last = spins[spins.len() - 1];
        assert!(
            near(last, expected, 2e-3),
            "{last:?} {expected:?} {:?}",
            path[0]
        );

        let (_, path) = box_path(BOX_HALF_WIDTH, BOX_MOMENTUM, FLIGHT_FRAMES, BOUNCE_STEPS);
        let path: Vec<Velocity> = (0..LOOPS).flat_map(|_| path.iter().copied()).collect();
        let spins = bouncing(&path);
        let area = LOOPS as f64 * square_area(BOX_MOMENTUM);
        let expected = exp(xy().gp(area / 2.0)) >> gyroscope_spin();
        assert!(near(spins[spins.len() - 1], expected, 5e-2));
    }

    /// numga's fourth test: at the magic Lorentz factor the electric field drops out of the
    /// anomalous rate; at a Lorentz factor of 2 it enters.
    #[test]
    fn the_magic_lorentz_factor_removes_the_electric_field() {
        for electric in [-0.2, 0.0, 0.2] {
            let rate = |gamma| anomalous_rate(gamma, (1.0, electric), 0.1, 5.0, 100);
            assert!((rate(magic(0.1)) - 1.0).abs() < 1e-3);
            if electric < 0.0 {
                assert!(rate(2.0) < 0.9);
            }
            if electric > 0.0 {
                assert!(rate(2.0) > 1.1);
            }
        }
    }

    /// numga's scenario check of the sweep: at the magic Lorentz factor the rate is one, and
    /// elsewhere the Bargmann-Michel-Telegdi formula's.
    #[test]
    fn the_sweep_follows_the_bmt_formula() {
        let [two, at_magic] = magic_sweep();
        let gamma = 2.0;
        for k in 0..ELECTRIC_FIELDS {
            let e = electric_field(k);
            assert!((at_magic[k] - 1.0).abs() < 1e-3);
            let factor = 1.0 - 1.0 / (MUON_ANOMALY * (gamma * gamma - 1.0));
            let expected = 1.0 - factor * speed(gamma) * e / RING_MAGNETIC;
            assert!(
                (two[k] - expected).abs() < 1e-3,
                "{k} {} {expected}",
                two[k]
            );
        }
    }

    /// The frame's bivector turns the velocity by the Lorentz force whatever the anomaly, and
    /// keeps it a unit velocity.
    #[test]
    fn the_anomaly_cancels_on_the_velocity() {
        let states = ring(2.0, (1.0, 0.1), 0.3, 1, 40, 1);
        for s in &states {
            assert!(((s.velocity * s.velocity).s() + 1.0).abs() < 1e-12);
            let field = ring_field(s.event, 1.0, 0.1);
            let [a, b] = [0.0, 0.3].map(|anomaly| {
                s.velocity
                    .commutator(bmt_generator(field, s.velocity, anomaly))
            });
            assert!(a.abs_diff_eq(&b, 1e-12));
        }
    }
}
