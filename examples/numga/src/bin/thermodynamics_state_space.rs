//! numga's `thermodynamics/state_space`: heating a gas under a constraint. Equal heat warms a gas
//! held at fixed volume more than one held at fixed pressure, in the metric-free algebra `s0v0`
//! (declared here with `gax::algebra!`).
//!
//! A state of the gas is an intrinsic vector (numga's antivector) spanned by entropy,
//! `s.dual()`, and volume, `v.dual()`, and quantities are read from it by the regressive
//! product, `s & state`. The gas is its energy as a function of the state; the energy's
//! derivative (a jet gradient) is the reading of a change of state that gives the temperature
//! and minus the pressure, and its second derivative (the jet Hessian) how those two change. A
//! constraint is another reading of the change, one that must stay zero. The two readings,
//! the temperature's change and the constraint, are one map from a change of state to the
//! vectors `s` and `v` as a pair of slots (`s dT + v constraint`, where numga stacks them into a
//! field), and solving it for `s`, one degree warmer and no change of the constraint, gives the
//! heating direction. The heat a change takes in is the temperature times its change of entropy:
//! the heat capacity at fixed volume is `c_v`, at fixed pressure `c_v + 1`.
//!
//! The animation: the two cylinders, latched (fixed volume) and loaded (fixed pressure), heated
//! alike; the temperature-entropy chart, where the heat is the area under each path; the
//! pressure-volume chart, where the work is; and each temperature against the heat taken in.
//! Entropy is in nR, volume and temperature in their initial values, energy in nRT₀.

gax::algebra! {
    algebra gas "Entropy and volume, `s0v0`: both generators square to zero.";
    basis es = 0, ev = 0;
    kind Scalar = [1];
    kind Vector = [es, ev];
    kind Pseudoscalar = [esv];
}

use charts::{BOLD, Diagram, Family, Floating, GasLines, MEDIUM};
use gax_colour::Light;
use gax_numga_examples::shared::thermodynamics_charts as charts;
use gax_numga_examples::{Anim, Canvas, Point2, Rect, caption, run_with};

mod heating {
    use super::gas::{Scalar, Vector, intrinsic};
    use gax::Real;
    use gax::jet::hessian;

    /// A state: entropy and volume, an intrinsic vector (numga's `Antivector-`).
    pub type State<T = f64> = intrinsic::Vector<(), T>;
    /// `Scalar <- State`: a reading of changes of state.
    pub type Reading = Scalar<(intrinsic::Vector,), f64>;
    /// Two readings of one change, as a map onto the pair `s`, `v`.
    pub type Readings = Vector<(intrinsic::Vector,), f64>;

    /// The entropy's and the volume's gradients, `s` and `v`.
    pub fn s() -> Vector {
        Vector::new(1.0, 0.0)
    }
    pub fn v() -> Vector {
        Vector::new(0.0, 1.0)
    }
    pub fn state(entropy: f64, volume: f64) -> State {
        s().dual() * entropy + v().dual() * volume
    }
    pub fn entropy() -> Reading {
        s() & State::slot()
    }
    pub fn volume() -> Reading {
        v() & State::slot()
    }

    /// The energy of an ideal gas with a constant heat capacity at fixed volume.
    pub fn energy<T: Real>(state: State<T>, capacity: f64) -> T {
        let read = |g: Vector| (g.map_coefs(T::from_f64) & state).s();
        let c = T::from_f64(capacity);
        c * ((read(s()) - read(v()).ln()) / c).exp()
    }

    /// Temperature and pressure, and the readings of how they change, all differentiated from
    /// the energy.
    pub fn response(state: State, capacity: f64) -> (f64, f64, Reading, Reading) {
        let (_, differential, second) = hessian(|x| energy(x, capacity), state);
        let temperature = differential.of(s().dual()).s();
        let pressure = -differential.of(v().dual()).s();
        (
            temperature,
            pressure,
            second.of(s().dual()),
            -second.of(v().dual()),
        )
    }

    /// The change of state per degree warmer with the constraint's reading held, and the heat
    /// it takes in.
    pub fn heating_direction(state: State, constraint: Reading, capacity: f64) -> (State, f64) {
        let (temperature, _, temperature_change, _) = response(state, capacity);
        // The two readings of one unknown change: the temperature's rises by one, the
        // constraint's stays put.
        let readings: Readings = s() * temperature_change + v() * constraint;
        let direction = readings.solve(s());
        (direction, temperature * entropy().of(direction).s())
    }

    /// What each cylinder holds: the latched one its volume, the loaded one its pressure.
    pub fn held(states: [State; 2], capacity: f64) -> [Reading; 2] {
        let (_, _, _, pressure_change) = response(states[1], capacity);
        [volume(), pressure_change]
    }

    /// The cylinders' states after taking in a little heat each, along their heating
    /// directions, by the midpoint rule.
    pub fn heated(states: [State; 2], heat: f64, capacity: f64) -> [State; 2] {
        let step = |x: [State; 2], scale: f64| {
            let held = held(x, capacity);
            core::array::from_fn(|k| {
                let (direction, capacity) = heating_direction(x[k], held[k], capacity);
                direction * (heat * scale / capacity)
            })
        };
        let half: [State; 2] = step(states, 0.5);
        let halfway = [states[0] + half[0], states[1] + half[1]];
        let full: [State; 2] = step(halfway, 1.0);
        [states[0] + full[0], states[1] + full[1]]
    }

    // --- the scenario: equal heat into a latched and a loaded cylinder ------------------

    pub const CAPACITY: f64 = 1.5;

    pub fn temperature_of(x: State) -> f64 {
        response(x, CAPACITY).0
    }
    #[cfg(test)]
    pub fn pressure_of(x: State) -> f64 {
        response(x, CAPACITY).1
    }

    /// Both cylinders at the reference state: unit volume and temperature, entropy measured from
    /// there.
    pub fn initial() -> [State; 2] {
        [state(0.0, 1.0); 2]
    }

    /// A stage of the heating: the cylinders' states, latched then loaded, their heating
    /// directions, and the heat each has taken in.
    pub struct Stage {
        pub states: [State; 2],
        pub directions: [State; 2],
        pub heat: f64,
    }

    /// The stages as equal slices of heat come in.
    pub fn equal_heating(total_heat: f64, steps: usize) -> Vec<Stage> {
        let mut states = initial();
        (0..=steps)
            .map(|step| {
                let held = held(states, CAPACITY);
                let directions =
                    core::array::from_fn(|k| heating_direction(states[k], held[k], CAPACITY).0);
                let stage = Stage {
                    states,
                    directions,
                    heat: step as f64 * total_heat / steps as f64,
                };
                states = heated(states, total_heat / steps as f64, CAPACITY);
                stage
            })
            .collect()
    }
}

use heating::*;

/// numga's run: 2.4 units of heat in 97 slices, 40 ms a frame.
const TOTAL_HEAT: f64 = 2.4;
const STEPS: usize = 97;
const FRAME_SECONDS: f32 = 0.04;
const HOLD: f32 = 1.2;

/// The charts' ranges: entropy and temperature, volume and pressure, heat.
const ENTROPY_RANGE: [f64; 2] = [-0.4, 2.0];
const TEMPERATURE_RANGE: [f64; 2] = [0.0, 3.0];
const VOLUME_RANGE: [f64; 2] = [0.0, 2.4];
const PRESSURE_RANGE: [f64; 2] = [0.0, 3.0];
const HEAT_RANGE: [f64; 2] = [0.0, 2.6];
/// The levels of the families: volumes, pressures, temperatures and entropies.
const ISOCHORES: [f64; 5] = [0.5, 0.7, 1.0, 1.4, 2.0];
const ISOBARS: [f64; 6] = [0.5, 0.7, 1.0, 1.4, 2.0, 2.8];
const ISOTHERMS: [f64; 5] = [0.5, 1.0, 1.5, 2.0, 2.5];
const ADIABATS: [f64; 7] = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0];
/// The height of a unit of volume in the cylinder; how far along a heating direction its arrow
/// looks, and its length in numga pixels.
const HEIGHT_PER_VOLUME: f64 = 0.9;
const ARROW_STEP: f64 = 0.05;
const ARROW_PIXELS: f32 = 26.0;

/// The cylinders' colours on the charts: latched blue, loaded orange.
const CASES: [Light; 2] = [charts::BLUE, charts::ORANGE];

/// The gas's temperature and pressure at an entropy and a volume, for the charts.
fn gas(entropy: f64, volume: f64) -> [f64; 2] {
    let (temperature, pressure, _, _) = response(state(entropy, volume), CAPACITY);
    [temperature, pressure]
}

/// A state on the two charts.
fn points(x: State) -> [Point2; 2] {
    charts::gas_points(&gas, entropy().of(x).s(), volume().of(x).s())
}

/// What is computed once: the heating, and the gas's lines.
struct Scene {
    stages: Vec<Stage>,
    lines: GasLines,
}

fn scene() -> Scene {
    let entropies: Vec<f64> = (0..120).map(|k| -2.5 + 5.5 * k as f64 / 119.0).collect();
    // Volumes by equal ratios from 0.2 to 4.
    let volumes: Vec<f64> = (0..120)
        .map(|k| 0.2 * 20f64.powf(k as f64 / 119.0))
        .collect();
    let levels = [&ISOCHORES[..], &ISOBARS, &ISOTHERMS, &ADIABATS];
    Scene {
        stages: equal_heating(TOTAL_HEAT, STEPS),
        lines: charts::gas_lines(&gas, &entropies, &volumes, levels),
    }
}

/// One of the two charts of the gas: its families, the area under each cylinder's path down to
/// zero (the heat on the temperature-entropy chart, the work on the pressure-volume chart), the
/// paths, and arrows along the heating directions.
#[allow(clippy::too_many_arguments)]
fn gas_chart(
    c: &mut Canvas,
    d: &Diagram,
    families: [(&[Vec<Point2>], Family, &str); 2],
    sc: &Scene,
    frame: usize,
    which: usize,
    note: &str,
) {
    let paths: Vec<[Point2; 2]> = sc.stages[..=frame]
        .iter()
        .map(|stage| [0, 1].map(|k| points(stage.states[k])[which]))
        .collect();
    if frame > 0 {
        for (k, light) in CASES.into_iter().enumerate() {
            let mut under: Vec<Point2> = paths.iter().map(|p| p[k]).collect();
            under.push(charts::foot(paths[frame][k], 0.0));
            under.push(charts::foot(paths[0][k], 0.0));
            d.area(c, &under, light.faded(0.3), 0.25);
        }
    }
    for (lines, family, _) in families {
        d.family(c, lines, family);
    }
    let floating: Vec<Floating> = families
        .iter()
        .map(|(lines, family, name)| Floating::along(lines, 8, name, family.light()))
        .collect();
    let names = d.place(c, &floating);
    let stage = &sc.stages[frame];
    for (k, light) in CASES.into_iter().enumerate() {
        let path: Vec<Point2> = paths.iter().map(|p| p[k]).collect();
        d.polyline(c, &path, BOLD, light);
        let aim = points(stage.states[k] + stage.directions[k] * ARROW_STEP)[which];
        d.arrow(c, paths[frame][k], aim, ARROW_PIXELS, MEDIUM, charts::INK);
    }
    for (k, light) in CASES.into_iter().enumerate() {
        d.dots(c, &[paths[frame][k]], light);
    }
    d.write_names(c, &names);
    d.axes(c);
    d.note(c, note);
}

/// Each cylinder's temperature against the heat it has taken in: straight lines whose slopes are
/// one over the heat capacities.
fn heating_curves(c: &mut Canvas, rect: Rect, sc: &Scene, frame: usize) {
    let d = Diagram::new(rect, HEAT_RANGE, TEMPERATURE_RANGE, [0.0, 0.0], ["Q", "T"]);
    let mut now = [charts::at(0.0, 0.0); 2];
    for (k, light) in CASES.into_iter().enumerate() {
        let curve: Vec<Point2> = sc.stages[..=frame]
            .iter()
            .map(|stage| charts::at(stage.heat, temperature_of(stage.states[k])))
            .collect();
        d.polyline(c, &curve, BOLD, light);
        now[k] = curve[frame];
    }
    for (k, light) in CASES.into_iter().enumerate() {
        d.dots(c, &[now[k]], light);
    }
    let text = charts::TEXT;
    d.label(
        c,
        now[0],
        "FIXED VOLUME",
        CASES[0],
        text,
        charts::RB,
        [-6.0, 6.0],
    );
    d.label(
        c,
        now[1],
        "FIXED PRESSURE",
        CASES[1],
        text,
        charts::RT,
        [-6.0, -8.0],
    );
    d.axes(c);
    d.note(c, "SLOPES: ONE OVER THE CAPACITIES");
}

fn draw(c: &mut Canvas, sc: &Scene, t: f32) {
    gax_numga_examples::backdrop(c);
    let frame = ((t / FRAME_SECONDS) as usize).min(STEPS);
    let stage = &sc.stages[frame];
    let panels = charts::row(c, &[0.9, 1.0, 1.0, 1.0]);
    // The cylinders, latched then loaded, side by side.
    let maximum = (VOLUME_RANGE[1] * HEIGHT_PER_VOLUME) as f32;
    for k in 0..2 {
        let half = panels[0].part(0.5 * k as f32, 0.2, 0.5 * (k + 1) as f32, 0.8);
        let height = (volume().of(stage.states[k]).s() * HEIGHT_PER_VOLUME) as f32;
        let warmth = temperature_of(stage.states[k]) / TEMPERATURE_RANGE[1];
        charts::gas_cylinder(c, half, height, warmth, k == 0, maximum);
    }
    let lines = &sc.lines;
    let crossing = [ENTROPY_RANGE[0], 0.0];
    let st = Diagram::new(
        panels[1],
        ENTROPY_RANGE,
        TEMPERATURE_RANGE,
        crossing,
        ["S", "T"],
    );
    let families = [
        (
            &lines.isochores.entropy_temperature[..],
            Family::Volume,
            "V = CONST",
        ),
        (
            &lines.isobars.entropy_temperature,
            Family::Pressure,
            "P = CONST",
        ),
    ];
    gas_chart(c, &st, families, sc, frame, 0, "HEAT: THE AREA UNDER T(S)");
    let vp = Diagram::new(
        panels[2],
        VOLUME_RANGE,
        PRESSURE_RANGE,
        [0.0, 0.0],
        ["V", "P"],
    );
    let families = [
        (
            &lines.isotherms.volume_pressure[..],
            Family::Temperature,
            "T = CONST",
        ),
        (
            &lines.adiabats.volume_pressure,
            Family::Entropy,
            "S = CONST",
        ),
    ];
    gas_chart(c, &vp, families, sc, frame, 1, "WORK: THE AREA UNDER P(V)");
    heating_curves(c, panels[3], sc, frame);
    caption(
        c,
        "STATE SPACE: HEATING AT FIXED VOLUME AND AT FIXED PRESSURE",
        "THE HEATING DIRECTION SOLVES TWO READINGS OF ONE CHANGE: DT = 1, CONSTRAINT = 0 (S0V0)",
    );
}

fn main() {
    let seconds = FRAME_SECONDS * (STEPS + 1) as f32 + HOLD;
    run_with(
        Anim::new("state_space", seconds).size(960, 540),
        scene,
        draw,
    );
}

#[cfg(test)]
mod tests {
    use super::heating::*;

    const ROUND_OFF: f64 = 1e-12;

    /// The temperature is the energy over the capacity, the gas is ideal, the Maxwell relation
    /// holds, and the heating directions keep their constraints and give the heat capacities
    /// `c_v` and `c_v + 1`.
    #[test]
    fn energy_derivatives_and_constrained_heat_capacities() {
        for (entropy_value, volume_value) in [(-0.3, 0.7), (0.1, 1.1), (0.8, 1.8)] {
            let x = state(entropy_value, volume_value);
            let (temperature, pressure, temperature_change, pressure_change) =
                response(x, CAPACITY);
            assert!((temperature - energy(x, CAPACITY) / CAPACITY).abs() < ROUND_OFF);
            assert!((pressure * volume().of(x).s() - temperature).abs() < ROUND_OFF);
            // A Maxwell relation: the temperature's change with volume is minus the pressure's
            // with entropy.
            let maxwell =
                temperature_change.of(v().dual()).s() + pressure_change.of(s().dual()).s();
            assert!(maxwell.abs() < ROUND_OFF);
            let constraints = [volume(), pressure_change];
            let expected = [CAPACITY, CAPACITY + 1.0];
            for (constraint, capacity) in constraints.into_iter().zip(expected) {
                let (direction, heat) = heating_direction(x, constraint, CAPACITY);
                assert!((temperature_change.of(direction).s() - 1.0).abs() < ROUND_OFF);
                assert!(constraint.of(direction).s().abs() < ROUND_OFF);
                assert!((heat - capacity).abs() < ROUND_OFF);
            }
        }
    }

    /// Both cylinders take in 2.4 units of heat as energy and work; the latched one keeps its
    /// volume and the loaded one its pressure, and they end at temperatures 2.6 and 1.96.
    #[test]
    fn equal_heating_conserves_energy_under_both_constraints() {
        let total = 2.4;
        let stages = equal_heating(total, 48);
        let (start, end) = (stages[0].states, stages[stages.len() - 1].states);
        let close = |a: f64, b: f64| (a - b).abs() <= 1e-3 * b.abs();
        for k in 0..2 {
            let work =
                pressure_of(start[k]) * (volume().of(end[k]).s() - volume().of(start[k]).s());
            let energy_change = energy(end[k], CAPACITY) - energy(start[k], CAPACITY);
            assert!(
                close(energy_change + work, total),
                "{}",
                energy_change + work
            );
        }
        assert!((volume().of(end[0]).s() - 1.0).abs() < ROUND_OFF);
        assert!(close(pressure_of(end[1]), 1.0));
        assert!(close(temperature_of(end[0]), 2.6));
        assert!(close(temperature_of(end[1]), 1.96));
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws_with(super::draw, &super::scene(), 2.0);
    }
}
