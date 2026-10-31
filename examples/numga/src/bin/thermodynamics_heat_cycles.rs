//! numga's `thermodynamics/heat_cycles`: a Carnot engine. A gas taken around a cycle between a
//! hot and a cold reservoir turns part of the heat it takes in into work, in the metric-free
//! algebra `s0v0` (declared here with `gax::algebra!`).
//!
//! A state of the gas is an intrinsic vector (numga's antivector) spanned by entropy and volume,
//! and the energy's derivative (a jet gradient) is the reading of changes of state that gives the
//! temperature and minus the pressure. The temperature times the entropy's reading is the heat
//! a change takes in, the pressure times the volume's reading the work it does; their difference
//! is the energy's derivative, while neither alone is the derivative of anything, so heat and
//! work depend on the path and their difference does not. The antisymmetrised derivatives of the
//! two readings are forms on pairs of changes (`Scalar <- (State, State)`), reading oriented
//! areas: the heat circulated around an area is the temperature's change wedged with the
//! entropy's, the work the pressure's change wedged with the volume's, and the two are one form.
//! Around a closed cycle the energy returns, and the areas the cycle encloses on the two charts,
//! the net heat and the net work, are equal. The strokes run straight in entropy and
//! temperature; their tangents are the jet derivative of the stroke along its fraction, and the
//! heat and work are summed by the midpoint rule, which is exact here.
//!
//! The animation: the piston, its gas coloured by its temperature; the pressure-volume chart with
//! isotherms and adiabats, and the temperature-entropy chart with isochores and isobars, the
//! cycle's enclosed area shaded on both. Entropy is in nR, volume and temperature in reference
//! values, energy in nRT₀; work is positive when the gas does it.

gax::algebra! {
    algebra gas "Entropy and volume, `s0v0`: both generators square to zero.";
    basis es = 0, ev = 0;
    kind Scalar = [1];
    kind Vector = [es, ev];
    kind Pseudoscalar = [esv];
}

use charts::{BOLD, Diagram, Family, Floating, GasLines, THIN};
use gax_colour::Light;
use gax_numga_examples::shared::thermodynamics_charts as charts;
use gax_numga_examples::{Anim, Canvas, Point2, caption, run_with};

mod engine {
    use super::gas::{Scalar, Vector, intrinsic};
    use gax::Real;
    use gax::jet::{gradient, hessian, jacobian};

    /// A state: entropy and volume, an intrinsic vector (numga's `Antivector-`).
    pub type State<T = f64> = intrinsic::Vector<(), T>;
    /// `Scalar <- State`.
    pub type Reading = Scalar<(intrinsic::Vector,), f64>;
    /// `Scalar <- (State, State)`: a reading of oriented areas.
    #[cfg_attr(not(test), allow(dead_code))]
    pub type AreaReading = Scalar<(intrinsic::Vector, intrinsic::Vector), f64>;

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

    /// The gas's state at an entropy and a temperature: the volume where the energy's slope along
    /// the entropy is the temperature.
    pub fn lifted<T: Real>(entropy: T, temperature: T, capacity: f64) -> State<T> {
        let volume = (entropy - T::from_f64(capacity) * temperature.ln()).exp();
        let (s_dual, v_dual) = (
            s().dual().map_coefs(T::from_f64),
            v().dual().map_coefs(T::from_f64),
        );
        s_dual.gp(entropy) + v_dual.gp(volume)
    }

    /// The state a fraction of the way along the stroke from one corner to the next, straight in
    /// entropy and temperature: at the reservoirs' temperatures the gas takes in or gives up
    /// heat, between them it is insulated. Corners are `[entropy, temperature]`.
    pub fn stroke<T: Real>(fraction: T, from: [f64; 2], to: [f64; 2], capacity: f64) -> State<T> {
        let along = |k: usize| T::from_f64(from[k]) + T::from_f64(to[k] - from[k]) * fraction;
        lifted(along(0), along(1), capacity)
    }

    /// The energy's derivative: the reading giving the temperature and minus the pressure.
    pub fn differential(state: State, capacity: f64) -> Reading {
        gradient(|x| energy(x, capacity), state).1
    }

    /// The heat a change along a tangent takes in and the work it does.
    pub fn exchanges(state: State, tangent: State, capacity: f64) -> (f64, f64) {
        let d = differential(state, capacity);
        let heat = entropy() * d.of(s().dual()).s();
        let work = volume() * -d.of(v().dual()).s();
        (heat.of(tangent).s(), work.of(tangent).s())
    }

    /// The heat and the work circulated around small oriented areas of states, as forms on the
    /// pairs of changes that span them.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn circulations(state: State, capacity: f64) -> (AreaReading, AreaReading) {
        let (_, _, second) = hessian(|x| energy(x, capacity), state);
        let temperature_change: Reading = second.of(s().dual());
        let pressure_change: Reading = -second.of(v().dual());
        let heat_area = temperature_change * entropy() - entropy() * temperature_change;
        let work_area = pressure_change * volume() - volume() * pressure_change;
        (heat_area, work_area)
    }

    // --- the scenario: a reversible engine between two reservoirs -----------------------

    pub const CAPACITY: f64 = 1.5;
    pub const HOT: f64 = 1.5;
    pub const COLD: f64 = 1.0;
    pub const ENTROPY_SPAN: f64 = 1.0;

    /// The cycle's corners in entropy and temperature, the first again at the end: heated at the
    /// hot temperature, expanded insulated, cooled at the cold temperature, compressed
    /// insulated.
    pub const CORNERS: [[f64; 2]; 5] = [
        [0.0, HOT],
        [ENTROPY_SPAN, HOT],
        [ENTROPY_SPAN, COLD],
        [0.0, COLD],
        [0.0, HOT],
    ];

    pub fn temperature_of(x: State) -> f64 {
        differential(x, CAPACITY).of(s().dual()).s()
    }
    pub fn pressure_of(x: State) -> f64 {
        -differential(x, CAPACITY).of(v().dual()).s()
    }

    /// The cycle's states, every stroke in steps, the first state again at the end; and the heat
    /// taken in and the work done up to each, by the midpoint rule along the strokes' tangents.
    pub fn cycle(steps: usize) -> (Vec<State>, Vec<f64>, Vec<f64>) {
        let mut states = vec![lifted(CORNERS[0][0], CORNERS[0][1], CAPACITY)];
        let (mut heat, mut work) = (vec![0.0], vec![0.0]);
        for k in 0..4 {
            let (from, to) = (CORNERS[k], CORNERS[k + 1]);
            for step in 0..steps {
                let middle = (step as f64 + 0.5) / steps as f64;
                // The tangent at the middle: the stroke's derivative there bound to a whole
                // stroke's fraction.
                let (at, slope) = jacobian(
                    |f: Scalar<(), _>| stroke(f.s(), from, to, CAPACITY),
                    Scalar::new(middle),
                );
                let tangent = slope.of(Scalar::new(1.0));
                let (heat_rate, work_rate) = exchanges(at, tangent, CAPACITY);
                let end = (step + 1) as f64 / steps as f64;
                states.push(stroke(end, from, to, CAPACITY));
                heat.push(heat[heat.len() - 1] + heat_rate / steps as f64);
                work.push(work[work.len() - 1] + work_rate / steps as f64);
            }
        }
        (states, heat, work)
    }
}

use engine::*;

/// numga's run: 28 steps a stroke, 40 ms a frame.
const STEPS: usize = 28;
const FRAME_SECONDS: f32 = 0.04;
const HOLD: f32 = 0.6;

/// The charts' ranges: volume and pressure, entropy and temperature.
const VOLUME_RANGE: [f64; 2] = [0.0, 3.2];
const PRESSURE_RANGE: [f64; 2] = [0.0, 3.2];
const ENTROPY_RANGE: [f64; 2] = [-0.6, 1.6];
const TEMPERATURE_RANGE: [f64; 2] = [0.0, 2.0];
/// The levels of the families: volumes, pressures, temperatures and entropies.
const ISOCHORES: [f64; 5] = [0.5, 1.0, 1.5, 2.0, 3.0];
const ISOBARS: [f64; 6] = [0.4, 0.7, 1.0, 1.5, 2.2, 3.0];
const ISOTHERMS: [f64; 5] = [0.5, 1.0, 1.5, 2.0, 2.5];
const ADIABATS: [f64; 7] = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0];
/// The height of a unit of volume in the cylinder.
const HEIGHT_PER_VOLUME: f64 = 0.7;

/// The strokes' colours: heated, insulated, cooled, insulated; the enclosed area's.
const STROKES: [Light; 4] = [charts::ORANGE, charts::GREY, charts::BLUE, charts::GREY];
fn enclosed() -> Light {
    gax_colour::hex(0xc5d9a7, 0.5)
}

/// The gas's temperature and pressure at an entropy and a volume, for the charts.
fn gas(entropy: f64, volume: f64) -> [f64; 2] {
    let x = state(entropy, volume);
    [temperature_of(x), pressure_of(x)]
}

/// What is computed once: the cycle on both charts, and the gas's lines.
struct Scene {
    states: Vec<State>,
    /// The cycle on the temperature-entropy chart and on the pressure-volume chart.
    charted: [Vec<Point2>; 2],
    lines: GasLines,
}

fn scene() -> Scene {
    let (states, _, _) = cycle(STEPS);
    let entropies: Vec<f64> = (0..120).map(|k| -2.5 + 5.5 * k as f64 / 119.0).collect();
    // Volumes by equal ratios from 0.2 to 5.
    let volumes: Vec<f64> = (0..120)
        .map(|k| 0.2 * 25f64.powf(k as f64 / 119.0))
        .collect();
    let levels = [&ISOCHORES[..], &ISOBARS, &ISOTHERMS, &ADIABATS];
    let points: Vec<[Point2; 2]> = (states.iter())
        .map(|x| charts::gas_points(&gas, entropy().of(*x).s(), volume().of(*x).s()))
        .collect();
    Scene {
        charted: [0, 1].map(|k| points.iter().map(|p| p[k]).collect()),
        states,
        lines: charts::gas_lines(&gas, &entropies, &volumes, levels),
    }
}

/// A chart with its lines, the area the cycle encloses, the strokes faint and, as far as
/// travelled, bold, and the work written amid the area (at the mean of the cycle's samples).
fn chart(
    c: &mut Canvas,
    d: &Diagram,
    cycle: &[Point2],
    families: [(&[Vec<Point2>], Family, &str); 2],
    travelled: usize,
) {
    d.area(c, &cycle[..cycle.len() - 1], enclosed(), 0.35);
    for (lines, family, _) in families {
        d.family(c, lines, family);
    }
    let floating: Vec<Floating> = families
        .iter()
        .map(|(lines, family, name)| Floating::along(lines, 8, name, family.light()))
        .collect();
    let names = d.place(c, &floating);
    for (k, light) in STROKES.into_iter().enumerate() {
        let part = &cycle[k * STEPS..=(k + 1) * STEPS];
        d.polyline(c, part, THIN, light.faded(0.6));
        let done = (travelled + 1).saturating_sub(k * STEPS).min(STEPS + 1);
        if done > 1 {
            d.polyline(c, &part[..done], BOLD, light);
        }
    }
    let n = (cycle.len() - 1) as f32;
    let mean = (cycle[..cycle.len() - 1].iter())
        .fold(Point2::direction(0.0, 0.0), |sum, p| {
            sum + (*p - Point2::ORIGIN)
        })
        .gp(1.0 / n)
        + Point2::ORIGIN;
    d.label(
        c,
        mean,
        "W",
        charts::INK,
        charts::TEXT,
        charts::MM,
        [0.0, 0.0],
    );
    d.dots(c, &[cycle[travelled]], charts::INK);
    d.write_names(c, &names);
    d.axes(c);
}

fn draw(c: &mut Canvas, sc: &Scene, t: f32) {
    gax_numga_examples::backdrop(c);
    let last = sc.states.len() - 1;
    let travelled = ((t / FRAME_SECONDS) as usize).min(last);
    let state = sc.states[travelled];
    let panels = charts::row(c, &[0.7, 1.0, 1.0]);
    // The piston, its gas coloured from the cold reservoir's temperature to the hot one's.
    let largest = (sc.states.iter())
        .map(|x| volume().of(*x).s())
        .fold(0.0, f64::max);
    let maximum = (largest * HEIGHT_PER_VOLUME) as f32;
    let height = (volume().of(state).s() * HEIGHT_PER_VOLUME) as f32;
    let warmth = (temperature_of(state) - COLD) / (HOT - COLD);
    let cylinder = panels[0].part(0.1, 0.15, 0.9, 0.85);
    charts::gas_cylinder(c, cylinder, height, warmth, false, maximum);

    let lines = &sc.lines;
    let vp = Diagram::new(
        panels[1],
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
    chart(c, &vp, &sc.charted[1], families, travelled);
    vp.note(c, "THE NET WORK: THE AREA ENCLOSED");

    let crossing = [ENTROPY_RANGE[0], 0.0];
    let st = Diagram::new(
        panels[2],
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
    chart(c, &st, &sc.charted[0], families, travelled);
    // The reservoirs named at the chart's left edge, level with the strokes at their
    // temperatures: the line through a corner along the entropy, met with that edge.
    let corners = &sc.charted[0];
    let hot = charts::level_with(corners[0], ENTROPY_RANGE[0]);
    let cold = charts::level_with(corners[2 * STEPS], ENTROPY_RANGE[0]);
    let text = charts::TEXT;
    st.label(c, hot, "HOT", charts::ORANGE, text, charts::LB, [4.0, 2.0]);
    st.label(c, cold, "COLD", charts::BLUE, text, charts::LT, [4.0, -2.0]);
    st.note(c, "THE NET HEAT: THE SAME AREA");
    caption(
        c,
        "HEAT CYCLES: A CARNOT ENGINE",
        "HEAT AND WORK CIRCULATE AS ONE FORM ON PAIRS OF CHANGES: DT^DS = DP^DV (S0V0)",
    );
}

fn main() {
    let seconds = FRAME_SECONDS * (4 * STEPS + 1) as f32 + HOLD;
    run_with(
        Anim::new("heat_cycles", seconds).size(960, 540),
        scene,
        draw,
    );
}

#[cfg(test)]
mod tests {
    use super::engine::*;
    use gax::ApproxEq;

    const ROUND_OFF: f64 = 1e-12;

    /// The gas is ideal, the energy's derivative is the heat less the work, and the heat and the
    /// work circulate as one form, `T / (c V)` on the unit area.
    #[test]
    fn first_law_and_equal_circulations() {
        for (entropy_value, volume_value) in [(0.0, 0.6), (0.5, 1.2), (1.0, 2.5)] {
            let x = state(entropy_value, volume_value);
            let d = differential(x, CAPACITY);
            let (temperature, pressure) = (d.of(s().dual()).s(), -d.of(v().dual()).s());
            assert!((pressure * volume().of(x).s() - temperature).abs() < ROUND_OFF);
            let first_law = entropy() * temperature - volume() * pressure - d;
            assert!(first_law.abs_diff_eq(&Reading::zero(), ROUND_OFF));
            let (heat_area, work_area) = circulations(x, CAPACITY);
            assert!(heat_area.abs_diff_eq(&work_area, ROUND_OFF));
            let unit = heat_area.of(s().dual()).of(v().dual()).s();
            let expected = temperature / (CAPACITY * volume().of(x).s());
            assert!((unit - expected).abs() < ROUND_OFF, "{unit} {expected}");
        }
    }

    /// The cycle closes; heat less work is the energy's change; the first stroke is at the hot
    /// temperature and the second at the entropy span; the net work is
    /// `(T_hot - T_cold) ΔS`, a third of the heat taken in.
    #[test]
    fn carnot_cycle_closes_and_turns_a_third_of_its_heat_into_work() {
        let steps = 16;
        let (states, heat, work) = cycle(steps);
        let last = states.len() - 1;
        assert!(states[last].abs_diff_eq(&states[0], ROUND_OFF));
        let e0 = energy(states[0], CAPACITY);
        for k in 0..=last {
            let change = energy(states[k], CAPACITY) - e0;
            assert!((heat[k] - work[k] - change).abs() < 1e-10);
        }
        for x in &states[..=steps] {
            assert!((temperature_of(*x) - HOT).abs() < ROUND_OFF);
        }
        for x in &states[steps..=2 * steps] {
            assert!((entropy().of(*x).s() - ENTROPY_SPAN).abs() < ROUND_OFF);
        }
        assert!((work[last] - (HOT - COLD) * ENTROPY_SPAN).abs() < 1e-10);
        assert!((work[last] / heat[steps] - (1.0 - COLD / HOT)).abs() < 1e-10);
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws_with(super::draw, &super::scene(), 2.0);
    }
}
