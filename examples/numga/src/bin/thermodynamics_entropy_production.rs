//! numga's `thermodynamics/entropy_production`: heat passing between two finite bodies, as a
//! gradient flow, in an algebra with no metric at all (`a0b0`, declared here with
//! `gax::algebra!`: both generators square to zero).
//!
//! A state holds the two bodies' energies, an intrinsic vector (numga's antivector) spanned by
//! `a.dual()` and `b.dual()`, and each energy is read back from it by the regressive product,
//! `a & state`, which needs no metric. The entropy's gradient, each body's received heat over its
//! temperature, `a / T_a + b / T_b`, is an extrinsic vector: it pairs with a change of state to
//! the entropy the change produces. The mobility is a map from gradients to changes of state, a
//! dyad that sends every gradient along the transfer `b.dual() - a.dual()`, trading one body's
//! energy for the other's, with a positive strength. It sends the entropy's gradient to the heat
//! flow and the total energy's gradient, `a + b`, to nothing: the flow conserves energy and never
//! lowers the entropy. Over a step the temperature difference decays exponentially, so the flow
//! integrates exactly.
//!
//! The animation: the bodies coloured by their gap to the common temperature they approach; the
//! plane of their energies with the entropy's level lines, where the state runs along the line of
//! fixed total energy to where a level line touches it; the entropy along that line, a hill the
//! state climbs, with its tangent; and the two temperatures closing on the common one.

gax::algebra! {
    algebra bodies "Two bodies' energies, `a0b0`: both generators square to zero, so nothing here uses a metric.";
    basis ea = 0, eb = 0;
    kind Scalar = [1];
    kind Vector = [ea, eb];
    kind Pseudoscalar = [eab];
}

use charts::{BOLD, Diagram, Family, MEDIUM};
use gax_numga_examples::points::Point3;
use gax_numga_examples::scene3::panel3;
use gax_numga_examples::shared::thermodynamics_charts as charts;
use gax_numga_examples::{Anim, Canvas, Point2, Rect, caption, run_with};

mod exchange {
    use super::bodies::{Scalar, Vector, intrinsic};
    use gax::Real;

    /// A state: the two bodies' energies, an intrinsic vector (numga's `Antivector-`).
    pub type State<T = f64> = intrinsic::Vector<(), T>;
    /// A gradient, a reading of changes of state (numga's `Vector+`).
    pub type Gradient = Vector<(), f64>;
    /// `State <- Gradient`.
    pub type Mobility = intrinsic::Vector<(Vector,), f64>;
    /// `Scalar <- State`: a reading of states.
    pub type Reading = Scalar<(intrinsic::Vector,), f64>;

    /// Body a's energy's gradient, `a`, and body b's, `b`.
    pub fn a() -> Gradient {
        Vector::new(1.0, 0.0)
    }
    pub fn b() -> Gradient {
        Vector::new(0.0, 1.0)
    }
    /// The state of energies `u_a` and `u_b`: `a.dual() u_a + b.dual() u_b`.
    pub fn state(u_a: f64, u_b: f64) -> State {
        a().dual() * u_a + b().dual() * u_b
    }
    /// Each body's energy, read by pairing.
    pub fn energy_a() -> Reading {
        a() & State::slot()
    }
    pub fn energy_b() -> Reading {
        b() & State::slot()
    }
    /// The total energy's gradient.
    pub fn energy_gradient() -> Gradient {
        a() + b()
    }
    pub fn total_energy() -> Reading {
        energy_gradient() & State::slot()
    }
    /// The direction trading body a's energy for body b's, which the total energy reads as
    /// nothing.
    pub fn transfer() -> State {
        b().dual() - a().dual()
    }
    /// The mobility of unit strength: every gradient sent along the transfer, by the gradient's
    /// reading of it.
    pub fn unit_mobility() -> Mobility {
        transfer() * (Vector::slot() & transfer())
    }

    /// A gradient's reading of a state with coefficients of any type (jets included).
    fn read<T: Real>(g: Gradient, state: State<T>) -> T {
        (g.map_coefs(T::from_f64) & state).s()
    }

    /// The two temperatures.
    pub fn temperatures<T: Real>(state: State<T>, capacity_a: f64, capacity_b: f64) -> [T; 2] {
        [
            read(a(), state) / T::from_f64(capacity_a),
            read(b(), state) / T::from_f64(capacity_b),
        ]
    }

    /// The total entropy, temperatures measured relative to one reference temperature.
    pub fn entropy<T: Real>(state: State<T>, capacity_a: f64, capacity_b: f64) -> T {
        let [t_a, t_b] = temperatures(state, capacity_a, capacity_b);
        T::from_f64(capacity_a) * t_a.ln() + T::from_f64(capacity_b) * t_b.ln()
    }

    /// The entropy's gradient: each body's entropy grows by its received heat over its
    /// temperature.
    pub fn entropy_gradient(state: State, capacity_a: f64, capacity_b: f64) -> Gradient {
        let [t_a, t_b] = temperatures(state, capacity_a, capacity_b);
        a() * t_a.recip() + b() * t_b.recip()
    }

    /// Every response along the energy-conserving direction, with a positive strength.
    pub fn mobility(state: State, capacity_a: f64, capacity_b: f64, conductance: f64) -> Mobility {
        let [t_a, t_b] = temperatures(state, capacity_a, capacity_b);
        unit_mobility().gp(conductance * t_a * t_b)
    }

    pub fn heat_flow(state: State, capacity_a: f64, capacity_b: f64, conductance: f64) -> State {
        mobility(state, capacity_a, capacity_b, conductance)
            .of(entropy_gradient(state, capacity_a, capacity_b))
    }

    /// The state after a nonnegative time, the heat flow integrated exactly: the temperature
    /// difference decays exponentially, which gives the integrated strength of the transfer.
    pub fn advance(
        state: State,
        capacity_a: f64,
        capacity_b: f64,
        conductance: f64,
        dt: f64,
    ) -> State {
        let inverse_capacity = capacity_a.recip() + capacity_b.recip();
        let integrated = -(-conductance * inverse_capacity * dt).exp_m1() / inverse_capacity;
        state
            + mobility(state, capacity_a, capacity_b, integrated)
                .of(entropy_gradient(state, capacity_a, capacity_b))
    }

    // --- the scenario: two insulated bodies at different temperatures -------------------

    pub const CAPACITY_A: f64 = 1.0;
    pub const CAPACITY_B: f64 = 2.0;
    pub const CONDUCTANCE: f64 = 0.6;
    pub const INITIAL: [f64; 2] = [2.8, 0.6];

    pub fn initial_state() -> State {
        state(CAPACITY_A * INITIAL[0], CAPACITY_B * INITIAL[1])
    }
    pub fn relaxation_time() -> f64 {
        (CONDUCTANCE * (CAPACITY_A.recip() + CAPACITY_B.recip())).recip()
    }
    pub fn entropy_of<T: Real>(state: State<T>) -> T {
        entropy(state, CAPACITY_A, CAPACITY_B)
    }
    pub fn temperatures_of(state: State) -> [f64; 2] {
        temperatures(state, CAPACITY_A, CAPACITY_B)
    }

    /// The state, its heat flow and the time at each frame, over a number of relaxation times.
    pub fn exchange(frames: usize, relaxation_times: f64) -> Vec<(State, State, f64)> {
        let dt = relaxation_times * relaxation_time() / (frames - 1) as f64;
        let mut state = initial_state();
        (0..frames)
            .map(|frame| {
                let flow = heat_flow(state, CAPACITY_A, CAPACITY_B, CONDUCTANCE);
                let out = (state, flow, frame as f64 * dt);
                state = advance(state, CAPACITY_A, CAPACITY_B, CONDUCTANCE, dt);
                out
            })
            .collect()
    }

    /// The entropy's change along the transfer at a state: the slope of the hill per unit of
    /// energy moved, the entropy's derivative bound to the transfer.
    pub fn slope_along_transfer(state: State) -> f64 {
        let (_, differential) = gax::jet::gradient(entropy_of, state);
        differential.of(transfer()).s()
    }
}

use exchange::*;

/// The frames and their duration, numga's.
const FRAMES: usize = 135;
const RELAXATION_TIMES: f64 = 5.0;
const FRAME_SECONDS: f32 = 0.04;
const HOLD: f32 = 1.2;

/// The charts' ranges: the two energies, body a's energy against the entropy, time and
/// temperature.
const ENERGY_RANGE: [f64; 2] = [0.0, 4.3];
const ENTROPY_RANGE: [f64; 2] = [-1.6, 1.3];
const TIME_RANGE: [f64; 2] = [0.0, 5.8];
const TEMPERATURE_RANGE: [f64; 2] = [0.0, 3.0];
/// How far an arrow reaches in numga pixels for a unit of flow on the chart, and the tangent's
/// half length in energy; the decades of the gap to the common temperature the colours span.
const ARROW_PER_FLOW: f32 = 40.0;
const TANGENT_HALF: f64 = 0.8;
const DECADES: f64 = 2.5;

/// What is computed once: the exchange, and the charts' fixed lines.
struct Scene {
    steps: Vec<(State, State, f64)>,
    level_lines: Vec<Vec<Point2>>,
    touching: Vec<Vec<Point2>>,
    energy_line: Vec<Point2>,
    hill_line: Vec<Point2>,
    top: usize,
}

/// A state on the chart of the two energies, read by pairing.
fn energy_point(state: State) -> Point2 {
    charts::at(energy_a().of(state).s(), energy_b().of(state).s())
}

/// A state on the chart of the entropy against body a's energy.
fn hill_point(state: State) -> Point2 {
    charts::at(energy_a().of(state).s(), entropy_of(state))
}

fn scene() -> Scene {
    let steps = exchange(FRAMES, RELAXATION_TIMES);
    let first = steps[0].0;
    // The entropy's levels over a grid of energies.
    let grid: Vec<f64> = (0..160)
        .map(|k| 0.04 + (4.3 - 0.04) * k as f64 / 159.0)
        .collect();
    let values: Vec<f64> = grid
        .iter()
        .flat_map(|&u_b| grid.iter().map(move |&u_a| entropy_of(state(u_a, u_b))))
        .collect();
    let level_lines = (0..11)
        .flat_map(|k| charts::level_lines(&grid, &grid, &values, -0.4 + 0.2 * k as f64))
        .collect();
    // The line of fixed total energy through the state, along the transfer, from body a holding
    // almost none of the energy to almost all of it.
    let total = total_energy().of(first).s();
    let line: Vec<State> = (0..240)
        .map(|k| {
            let held = total * (0.02 + 0.96 * k as f64 / 239.0);
            let along = (held - energy_a().of(first).s()) / energy_a().of(transfer()).s();
            first + transfer() * along
        })
        .collect();
    let entropies: Vec<f64> = line.iter().map(|s| entropy_of(*s)).collect();
    let top = (0..line.len())
        .max_by(|&i, &j| entropies[i].total_cmp(&entropies[j]))
        .unwrap_or(0);
    // The entropy's level line through the top of the hill touches the line there.
    let touching = charts::level_lines(&grid, &grid, &values, entropies[top]);
    Scene {
        steps,
        level_lines,
        touching,
        energy_line: line.iter().map(|s| energy_point(*s)).collect(),
        hill_line: line.iter().map(|s| hill_point(*s)).collect(),
        top,
    }
}

/// The two bodies, each as large as its heat capacity and coloured by its gap to the common
/// temperature (the widest gap at the thermal ramp's ends), joined by a bar.
fn bodies(c: &mut Canvas, rect: Rect, temperatures: [f64; 2], widest: f64) {
    let capacities = [CAPACITY_A, CAPACITY_B];
    let common = (capacities[0] * temperatures[0] + capacities[1] * temperatures[1])
        / (capacities[0] + capacities[1]);
    let corners: Vec<Point3> = [-1.75, 1.75]
        .into_iter()
        .flat_map(|x| [-0.75, 0.75].map(|y| (x, y)))
        .flat_map(|(x, y)| [0.0, 1.6].map(|z| Point3::xyz(x, y, z)))
        .collect();
    let square = Rect::around(
        rect.centre(),
        Point2::direction(rect.width() * 0.5, rect.width() * 0.4),
    );
    let cam = charts::apparatus_camera(square, &corners, -16.0, 18.0);
    panel3(c, square, cam, |s| {
        for (k, x) in [-1.0f32, 1.0].into_iter().enumerate() {
            let size = (0.5 * capacities[k].cbrt()) as f32;
            let warmth = charts::relaxing(temperatures[k] - common, widest, DECADES);
            let centre = Point3::xyz(x, 0.0, size);
            charts::cuboid(s, centre, [size; 3], charts::thermal(warmth), 1.0);
        }
        let bar = Point3::xyz(0.0, 0.0, 0.45);
        charts::cuboid(s, bar, [0.5, 0.12, 0.1], charts::GREY, 1.0);
    });
}

fn draw(c: &mut Canvas, sc: &Scene, t: f32) {
    gax_numga_examples::backdrop(c);
    let frame = ((t / FRAME_SECONDS) as usize).min(FRAMES - 1);
    let (state, flow, _) = sc.steps[frame];
    let panels = charts::row(c, &[0.85, 1.0, 1.0, 1.0]);
    // The bodies, coloured by their gaps to the common temperature, the first gap the widest.
    let first = temperatures_of(sc.steps[0].0);
    let common = (CAPACITY_A * first[0] + CAPACITY_B * first[1]) / (CAPACITY_A + CAPACITY_B);
    let widest = (first[0] - common).abs().max((first[1] - common).abs());
    bodies(c, panels[0], temperatures_of(state), widest);

    // The plane of the two energies.
    let names = ["U_A", "U_B"];
    let energies = Diagram::new(panels[1], ENERGY_RANGE, ENERGY_RANGE, [0.0, 0.0], names);
    energies.family(c, &sc.level_lines, Family::Entropy);
    energies.lines(c, &sc.touching, MEDIUM, charts::BOUNDARY);
    energies.polyline(c, &sc.energy_line, MEDIUM, charts::GREY);
    energies.dots(c, &[sc.energy_line[sc.top]], charts::GREEN);
    let candidates = charts::along(&sc.level_lines, 8);
    let entropy_light = Family::Entropy.light();
    energies.float(c, &candidates, "S = CONST", entropy_light, charts::LB);
    let path: Vec<Point2> = sc.steps[..=frame]
        .iter()
        .map(|s| energy_point(s.0))
        .collect();
    energies.polyline(c, &path, BOLD, charts::ORANGE);
    // The flow's arrow: as long as the flow on the chart, times so many pixels.
    let now = energy_point(state);
    let ahead = energy_point(state + flow);
    let size = (ahead - now).ideal_norm();
    if size > 1e-3 {
        energies.arrow(c, now, ahead, ARROW_PER_FLOW * size, MEDIUM, charts::INK);
    }
    energies.dots(c, &[now], charts::ORANGE);
    energies.axes(c);

    // The entropy along the line of fixed energy, with the tangent at the state: the change along
    // the transfer, of body a's energy and of the entropy.
    let crossing = [0.0, ENTROPY_RANGE[0]];
    let hill = Diagram::new(
        panels[2],
        ENERGY_RANGE,
        ENTROPY_RANGE,
        crossing,
        ["U_A", "S"],
    );
    hill.polyline(c, &sc.hill_line, MEDIUM, charts::GREY);
    hill.dots(c, &[sc.hill_line[sc.top]], charts::GREEN);
    let hill_path: Vec<Point2> = sc.steps[..=frame].iter().map(|s| hill_point(s.0)).collect();
    hill.polyline(c, &hill_path, BOLD, charts::ORANGE);
    let tangent = charts::toward(energy_a().of(transfer()).s(), slope_along_transfer(state));
    let u_a = energy_a().of(state).s();
    let here = hill_point(state);
    let reach = [u_a - TANGENT_HALF, u_a + TANGENT_HALF];
    hill.tangent(c, here, tangent, reach, MEDIUM, charts::ORANGE);
    hill.dots(c, &[here], charts::ORANGE);
    hill.axes(c);

    // The two temperatures against time, the hot body red and the cold one blue, over the common
    // temperature, dashed.
    let cooling = Diagram::new(
        panels[3],
        TIME_RANGE,
        TEMPERATURE_RANGE,
        [0.0, 0.0],
        ["TIME", "T"],
    );
    let level = [
        charts::at(TIME_RANGE[0], common),
        charts::at(TIME_RANGE[1], common),
    ];
    cooling.dashed(c, &level, MEDIUM, 4.0, charts::GREY);
    for (body, colour) in [(0, charts::thermal(1.0)), (1, charts::thermal(0.0))] {
        let curve: Vec<Point2> = sc.steps[..=frame]
            .iter()
            .map(|(s, _, time)| charts::at(*time, temperatures_of(*s)[body]))
            .collect();
        cooling.polyline(c, &curve, BOLD, colour.faded(1.6));
        cooling.dots(c, &curve[curve.len() - 1..], colour.faded(1.6));
    }
    let (ink, size) = (charts::INK, charts::TEXT);
    cooling.label(c, level[1], "COMMON", ink, size, charts::RB, [0.0, 3.0]);
    cooling.axes(c);
    let notes = [
        "ENERGIES: S LEVELS, FIXED TOTAL",
        "ENTROPY ALONG THE FIXED TOTAL",
        "THE TEMPERATURES MEET",
    ];
    for (diagram, note) in [energies, hill, cooling].iter().zip(notes) {
        diagram.note(c, note);
    }

    caption(
        c,
        "ENTROPY PRODUCTION: HEAT BETWEEN TWO BODIES",
        "THE MOBILITY, A MAP FROM GRADIENTS TO CHANGES OF STATE, SENDS DS TO THE FLOW AND DU TO NOTHING (A0B0)",
    );
}

fn main() {
    run_with(
        Anim::new("entropy_production", FRAME_SECONDS * FRAMES as f32 + HOLD).size(960, 540),
        scene,
        draw,
    );
}

#[cfg(test)]
mod tests {
    use super::exchange::*;
    use gax::ApproxEq;

    const ROUND_OFF: f64 = 1e-11;

    /// The states of numga's test: three pairs of temperatures.
    fn states(capacity_a: f64, capacity_b: f64) -> [State; 3] {
        let (t_a, t_b) = ([0.4, 1.0, 4.0], [3.0, 1.0, 0.6]);
        core::array::from_fn(|k| state(capacity_a * t_a[k], capacity_b * t_b[k]))
    }

    /// The entropy's gradient is its derivative; the mobility sends it to the heat flow and the
    /// energy's gradient to nothing; the flow conserves energy and produces entropy at
    /// `k (T_a - T_b)² / (T_a T_b)`; and the mobility is symmetric and positive.
    #[test]
    fn differential_and_mobility_generate_physical_heat_flow() {
        let (capacity_a, capacity_b, conductance) = (2.0, 3.0, 0.7);
        let probes_a = [(-3.0, 2.0), (0.0, 7.0), (4.0, -1.0)];
        let probes_b = [(1.0, 4.0), (-1.0, 2.0), (2.0, 3.0)];
        for (k, s) in states(capacity_a, capacity_b).into_iter().enumerate() {
            let [t_a, t_b] = temperatures(s, capacity_a, capacity_b);
            let gradient = entropy_gradient(s, capacity_a, capacity_b);
            let (_, differential) = gax::jet::gradient(|x| entropy(x, capacity_a, capacity_b), s);
            let mobility = mobility(s, capacity_a, capacity_b, conductance);
            let flow = mobility.of(gradient);
            let expected_flow = transfer() * (conductance * (t_a - t_b));
            let production = conductance * (t_a - t_b).powi(2) / (t_a * t_b);
            let reading = gradient & State::slot();
            assert!(
                reading.abs_diff_eq(&differential, ROUND_OFF),
                "{reading:?} {differential:?}"
            );
            assert!(flow.abs_diff_eq(&expected_flow, ROUND_OFF));
            let idle = mobility.of(energy_gradient());
            assert!(idle.abs_diff_eq(&State::zero(), ROUND_OFF));
            assert!(total_energy().of(flow).s().abs() < ROUND_OFF);
            assert!(((gradient & flow).s() - production).abs() < ROUND_OFF);
            let probe = |(x, y): (f64, f64)| a() * x + b() * y;
            let (p, q) = (probe(probes_a[k]), probe(probes_b[k]));
            let symmetric = (q & mobility.of(p)).s() - (p & mobility.of(q)).s();
            assert!(symmetric.abs() < ROUND_OFF);
            assert!((p & mobility.of(p)).s() >= -ROUND_OFF);
        }
    }

    /// Exact steps conserve the energy, never lower the entropy, keep both energies positive,
    /// close the temperature difference exponentially, and reach equilibrium; without
    /// conductance nothing moves.
    #[test]
    fn finite_steps_conserve_energy_increase_entropy_and_reach_equilibrium() {
        let (capacity_a, capacity_b, conductance) = (2.0, 3.0, 0.7);
        let (dt, long_dt, steps) = (0.5, 100.0, 12);
        for initial in states(capacity_a, capacity_b) {
            let mut trajectory = vec![initial];
            for _ in 0..steps {
                let last = trajectory[trajectory.len() - 1];
                trajectory.push(advance(last, capacity_a, capacity_b, conductance, dt));
            }
            let total = total_energy().of(initial).s();
            for pair in trajectory.windows(2) {
                let before = entropy(pair[0], capacity_a, capacity_b);
                let after = entropy(pair[1], capacity_a, capacity_b);
                assert!(after - before >= -ROUND_OFF);
            }
            for s in &trajectory {
                assert!((total_energy().of(*s).s() - total).abs() < ROUND_OFF);
                assert!(energy_a().of(*s).s() > 0.0 && energy_b().of(*s).s() > 0.0);
            }
            let [t_a, t_b] = temperatures(initial, capacity_a, capacity_b);
            let rate = conductance * (capacity_a.recip() + capacity_b.recip());
            let decay = (-rate * dt * steps as f64).exp();
            let [u_a, u_b] = temperatures(trajectory[steps], capacity_a, capacity_b);
            assert!((u_a - u_b - (t_a - t_b) * decay).abs() < ROUND_OFF);
            let equilibrium_temperature = total / (capacity_a + capacity_b);
            let equilibrium = state(
                capacity_a * equilibrium_temperature,
                capacity_b * equilibrium_temperature,
            );
            let relaxed = advance(initial, capacity_a, capacity_b, conductance, long_dt);
            assert!(relaxed.abs_diff_eq(&equilibrium, ROUND_OFF));
            let still = advance(initial, capacity_a, capacity_b, 0.0, long_dt);
            assert!(still.abs_diff_eq(&initial, ROUND_OFF));
        }
    }

    /// The hill's slope along the transfer is the entropy's change per unit of energy moved to
    /// body b, `1/T_b - 1/T_a`.
    #[test]
    fn the_hill_slope_is_the_difference_of_inverse_temperatures() {
        for (s, _, _) in exchange(6, 5.0) {
            let [t_a, t_b] = temperatures_of(s);
            assert!((slope_along_transfer(s) - (t_b.recip() - t_a.recip())).abs() < 1e-12);
        }
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws_with(super::draw, &super::scene(), 1.0);
    }
}
