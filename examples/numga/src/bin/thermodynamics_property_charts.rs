//! numga's `thermodynamics/property_charts`: cooling, drying and reheating moist air at fixed
//! pressure, on its psychrometric chart, in the metric-free algebra `t0w0` (declared here with
//! `gax::algebra!`).
//!
//! A state of the air is an intrinsic vector (numga's antivector) spanned by its temperature,
//! `t.dual()`, and its water content per kilogram of dry air, `w.dual()`, each read back by the
//! regressive product, `w & state`. Relative humidity and enthalpy are two more readings of the
//! same state, functions of it. A cooling coil first holds the water content; once the air is
//! saturated, it holds the relative humidity at one, the water beyond that condensing out; the
//! reheater holds the water content again. A held reading is a constraint on the change of
//! state; with the temperature's reading it makes one map from a change of state onto the pair
//! `t`, `w` (`t temperature + w constraint`, where numga stacks the two into a field), and solving
//! it for `t`, one degree warmer and the held reading unchanged, gives the direction the air moves
//! in. The relative humidity's reading at saturation is its jet gradient.
//!
//! The animation: the duct, with the coil, its tray of collected water, the reheater and the
//! parcel of air passing them; and the chart, temperature across and water content up, with the
//! lines of equal relative humidity, the saturation curve, the lines of equal enthalpy marked
//! with their values, and the arrows of the directions at the inlet and at the coil's outlet.
//! Temperature is in degrees Celsius, pressure in kPa, water content in kg water per kg dry air
//! (grams on the chart), enthalpy in kJ per kg dry air; saturation by the Magnus approximation
//! over liquid water.

gax::algebra! {
    algebra air "Temperature and water content, `t0w0`: both generators square to zero.";
    basis et = 0, ew = 0;
    kind Scalar = [1];
    kind Vector = [et, ew];
    kind Pseudoscalar = [etw];
}

use charts::{BOLD, Diagram, Family, MEDIUM, THIN};
use gax_numga_examples::points::Point3;
use gax_numga_examples::scene3::panel3;
use gax_numga_examples::shared::thermodynamics_charts as charts;
use gax_numga_examples::{Anim, Canvas, Marker, Point2, Rect, caption, run_with};

mod psychrometrics {
    use super::air::{Scalar, Vector, intrinsic};
    use gax::Real;

    /// A state: temperature and water content, an intrinsic vector (numga's `Antivector-`).
    pub type State<T = f64> = intrinsic::Vector<(), T>;
    /// `Scalar <- State`.
    pub type Reading = Scalar<(intrinsic::Vector,), f64>;
    /// Two readings of one change, as a map onto the pair `t`, `w`.
    pub type Readings = Vector<(intrinsic::Vector,), f64>;

    pub fn t() -> Vector {
        Vector::new(1.0, 0.0)
    }
    pub fn w() -> Vector {
        Vector::new(0.0, 1.0)
    }
    pub fn state(temperature: f64, water: f64) -> State {
        t().dual() * temperature + w().dual() * water
    }
    pub fn temperature() -> Reading {
        t() & State::slot()
    }
    pub fn humidity() -> Reading {
        w() & State::slot()
    }

    /// The Magnus approximation of the saturation pressure, in kPa, over a temperature in
    /// degrees Celsius; the ratio of the molar masses of water and dry air; the specific heats
    /// of dry air, of vapour and of liquid water, and the latent heat of vaporisation at zero
    /// degrees, in kJ per kg.
    pub const MAGNUS: [f64; 3] = [0.61078, 17.27, 237.3];
    pub const MOLAR_RATIO: f64 = 0.62198;
    pub const DRY_AIR_HEAT: f64 = 1.006;
    pub const VAPOUR_HEAT: f64 = 1.86;
    pub const LIQUID_HEAT: f64 = 4.186;
    pub const LATENT_HEAT: f64 = 2501.0;

    /// A gradient's reading of a state with coefficients of any type (jets included).
    fn read<T: Real>(g: Vector, state: State<T>) -> T {
        (g.map_coefs(T::from_f64) & state).s()
    }

    pub fn saturation_pressure<T: Real>(state: State<T>) -> T {
        let warmth = read(t(), state);
        let c = T::from_f64;
        c(MAGNUS[0]) * (c(MAGNUS[1]) * warmth / (warmth + c(MAGNUS[2]))).exp()
    }

    /// The most water the air holds as vapour at its temperature.
    pub fn saturation_humidity(state: State, pressure: f64) -> f64 {
        let vapour = saturation_pressure(state);
        MOLAR_RATIO * vapour / (pressure - vapour)
    }

    /// The vapour's partial pressure over the saturation pressure.
    pub fn relative_humidity<T: Real>(state: State<T>, pressure: f64) -> T {
        let water = read(w(), state);
        T::from_f64(pressure) * water
            / (T::from_f64(MOLAR_RATIO) + water)
            / saturation_pressure(state)
    }

    /// Air at a temperature and a relative humidity.
    pub fn air_state(warmth: f64, relative: f64, pressure: f64) -> State {
        let dry = state(warmth, 0.0);
        let vapour = relative * saturation_pressure(dry);
        dry + w().dual() * (MOLAR_RATIO * vapour / (pressure - vapour))
    }

    /// The dry air's heat and the vapour's, latent and sensible, per kg dry air.
    pub fn enthalpy(x: State) -> f64 {
        let (warmth, water) = (temperature().of(x).s(), humidity().of(x).s());
        DRY_AIR_HEAT * warmth + water * (LATENT_HEAT + VAPOUR_HEAT * warmth)
    }

    /// The temperature at which the air's water content saturates it.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn dew_point(x: State, pressure: f64) -> f64 {
        let water = humidity().of(x).s();
        let logarithm = (pressure * water / ((MOLAR_RATIO + water) * MAGNUS[0])).ln();
        MAGNUS[2] * logarithm / (MAGNUS[1] - logarithm)
    }

    /// The air cooled a fraction of the way to the coil's outlet: its water content held until
    /// it saturates, then the water beyond saturation condensed out.
    pub fn cooling_path(
        initial: State,
        outlet: f64,
        pressure: f64,
        fractions: &[f64],
    ) -> Vec<State> {
        let (start, water) = (temperature().of(initial).s(), humidity().of(initial).s());
        fractions
            .iter()
            .map(|f| {
                let uncondensed = state(start * (1.0 - f) + outlet * f, water);
                // The water beyond what saturated air holds at that temperature, none until the
                // air reaches its dew point.
                let condensate = (water - saturation_humidity(uncondensed, pressure)).max(0.0);
                uncondensed - w().dual() * condensate
            })
            .collect()
    }

    /// The air reheated a fraction of the way to the supply temperature, its water content held.
    pub fn reheating_path(initial: State, outlet: f64, fractions: &[f64]) -> Vec<State> {
        let (start, water) = (temperature().of(initial).s(), humidity().of(initial).s());
        (fractions.iter())
            .map(|f| state(start * (1.0 - f) + outlet * f, water))
            .collect()
    }

    /// The heat the coil removes and the water it collects per kg dry air, the water leaving at
    /// the outlet's temperature.
    pub fn coil_balance(initial: State, outlet: State) -> (f64, f64) {
        let condensate = humidity().of(initial).s() - humidity().of(outlet).s();
        let cooling = enthalpy(initial)
            - enthalpy(outlet)
            - condensate * LIQUID_HEAT * temperature().of(outlet).s();
        (cooling, condensate)
    }

    /// The change of state per degree warmer with the constraint's reading held: the two
    /// readings of one unknown change, the temperature's rising by one and the constraint's
    /// staying put.
    pub fn constrained_direction(constraint: Reading) -> State {
        let readings: Readings = t() * temperature() + w() * constraint;
        readings.solve(t())
    }

    // --- the scenario: a coil and a reheater bring warm humid air to a supply condition --

    pub const PRESSURE: f64 = 101.325;
    pub const INLET: [f64; 2] = [30.0, 0.7];
    pub const COIL: f64 = 12.0;
    pub const SUPPLY: f64 = 22.0;

    pub fn relative_humidity_of<T: Real>(x: State<T>) -> T {
        relative_humidity(x, PRESSURE)
    }

    /// The air's path through the coil and the reheater; its states at the inlet and at the
    /// coil's outlet; the directions it moves in there, holding its water content and holding
    /// its relative humidity; and the cooling, the water collected, the reheating and the
    /// supply's relative humidity.
    pub struct Conditioning {
        pub path: Vec<State>,
        pub states: [State; 2],
        pub directions: [State; 2],
        pub cooling: f64,
        pub condensate: f64,
        pub reheating: f64,
        pub supply: f64,
    }

    pub fn conditioning(steps: usize) -> Conditioning {
        let fractions: Vec<f64> = (0..steps).map(|k| k as f64 / (steps - 1) as f64).collect();
        let inlet = air_state(INLET[0], INLET[1], PRESSURE);
        let cooled = cooling_path(inlet, COIL, PRESSURE, &fractions);
        let outlet = cooled[steps - 1];
        let reheated = reheating_path(outlet, SUPPLY, &fractions);
        let (cooling, condensate) = coil_balance(inlet, outlet);
        let supplied = reheated[steps - 1];
        // Saturated air holds its relative humidity at one: the reading it holds is the
        // relative humidity's change.
        let (_, relative_change) = gax::jet::gradient(relative_humidity_of, outlet);
        Conditioning {
            path: [&cooled[..], &reheated[1..]].concat(),
            states: [inlet, outlet],
            directions: [
                constrained_direction(humidity()),
                constrained_direction(relative_change),
            ],
            cooling,
            condensate,
            reheating: enthalpy(supplied) - enthalpy(outlet),
            supply: relative_humidity(supplied, PRESSURE),
        }
    }
}

use psychrometrics::*;

/// numga's run: 52 steps a stage, 40 ms a frame.
const STEPS: usize = 52;
const FRAME_SECONDS: f32 = 0.04;
const HOLD: f32 = 1.2;

/// The chart's ranges, temperature in degrees and water content in grams per kg of dry air, and
/// the values read off its axes.
const TEMPERATURE_RANGE: [f64; 2] = [5.0, 35.0];
const WATER_RANGE: [f64; 2] = [0.0, 25.0];
const TEMPERATURE_TICKS: [f64; 3] = [10.0, 20.0, 30.0];
const WATER_TICKS: [f64; 4] = [5.0, 10.0, 15.0, 20.0];
const GRAMS_PER_KILOGRAM: f64 = 1000.0;
/// The relative humidities and enthalpies of the lines; how far a direction's arrow reaches in
/// numga pixels; where the saturation curve's name is written.
const HUMIDITY_LEVELS: [f64; 9] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
const ENTHALPY_LEVELS: [f64; 9] = [20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0];
const ARROW_PIXELS: f32 = 25.0;
const SATURATION_LABEL: [f64; 2] = [27.5, 22.5];

/// A state on the chart, its water in grams.
fn chart_point(x: State) -> Point2 {
    let water = GRAMS_PER_KILOGRAM * humidity().of(x).s();
    charts::at(temperature().of(x).s(), water)
}

/// What is computed once: the conditioning, and the chart's lines and labels.
struct Scene {
    run: Conditioning,
    humidities: Vec<(f64, Vec<Vec<Point2>>)>,
    enthalpies: Vec<(f64, Vec<Vec<Point2>>)>,
    saturation: Vec<Vec<Point2>>,
    /// The region above the saturation curve, up to the chart's top.
    above: Vec<Point2>,
}

fn scene() -> Scene {
    let run = conditioning(STEPS);
    let sample = |range: [f64; 2]| -> Vec<f64> {
        (0..150)
            .map(|k| range[0] + (range[1] - range[0]) * k as f64 / 149.0)
            .collect()
    };
    let (temperatures, grams) = (sample(TEMPERATURE_RANGE), sample(WATER_RANGE));
    let grid: Vec<State> = grams
        .iter()
        .flat_map(|&g| (temperatures.iter()).map(move |&t| state(t, g / GRAMS_PER_KILOGRAM)))
        .collect();
    let relative: Vec<f64> = grid.iter().map(|x| relative_humidity_of(*x)).collect();
    // The enthalpy only where the air holds its water as vapour.
    let heat: Vec<f64> = (grid.iter().zip(&relative))
        .map(|(x, r)| if *r <= 1.0 { enthalpy(*x) } else { f64::NAN })
        .collect();
    let lines =
        |values: &[f64], level: f64| charts::level_lines(&temperatures, &grams, values, level);
    let humidities = (HUMIDITY_LEVELS.iter())
        .map(|&l| (l, lines(&relative, l)))
        .collect();
    let enthalpies = (ENTHALPY_LEVELS.iter())
        .map(|&l| (l, lines(&heat, l)))
        .collect();
    let saturation = lines(&relative, 1.0);
    // Above the saturation curve: the curve from cold to warm, its samples past the top dropped
    // onto it along the water content, and the chart's top corners back.
    let mut curve: Vec<Point2> = saturation.iter().flatten().copied().collect();
    curve.sort_by(|a, b| a.to_euclidean()[0].total_cmp(&b.to_euclidean()[0]));
    let mut above: Vec<Point2> = (curve.iter())
        .map(|p| {
            if f64::from(p.to_euclidean()[1]) > WATER_RANGE[1] {
                charts::foot(*p, WATER_RANGE[1])
            } else {
                *p
            }
        })
        .collect();
    above.push(charts::at(TEMPERATURE_RANGE[1], WATER_RANGE[1]));
    above.push(charts::at(TEMPERATURE_RANGE[0], WATER_RANGE[1]));
    Scene {
        run,
        humidities,
        enthalpies,
        saturation,
        above,
    }
}

/// The duct: the cooling coil, its tray of collected water, the reheater, and the parcel of air
/// passing them, coloured by how warm it is.
fn apparatus(c: &mut Canvas, rect: Rect, fraction: f32, warmth: f64, liquid: f64) {
    let corners: Vec<Point3> = [-2.5, 2.5]
        .into_iter()
        .flat_map(|x| [-1.0, 0.65].map(|y| (x, y)))
        .flat_map(|(x, y)| [0.0, 1.5].map(|z| Point3::xyz(x, y, z)))
        .collect();
    let square = Rect::around(
        rect.centre(),
        Point2::direction(rect.width() * 0.5, rect.width() * 0.4),
    );
    let cam = charts::apparatus_camera(square, &corners, -10.0, 18.0);
    panel3(c, square, cam, |s| {
        let tray = Point3::xyz(0.0, 0.0, 0.1);
        charts::cuboid(s, tray, [2.35, 0.6, 0.08], charts::GREY.faded(0.6), 1.0);
        for offset in [-0.25, 0.0, 0.25] {
            let plate = Point3::xyz(-0.8 + offset, 0.0, 0.8);
            charts::cuboid(s, plate, [0.045, 0.5, 0.55], charts::BLUE.faded(0.6), 1.0);
        }
        let reheater = Point3::xyz(1.1, 0.0, 0.8);
        charts::cuboid(
            s,
            reheater,
            [0.12, 0.45, 0.45],
            charts::ORANGE.faded(0.6),
            1.0,
        );
        // The collected water, resting on the tray's floor and as deep as it has filled.
        let depth = (0.01 + 0.2 * liquid) as f32;
        let water = Point3::xyz(-0.8, -0.72, 0.15 + 0.5 * depth);
        charts::cuboid(s, water, [0.45, 0.2, 0.5 * depth], charts::BLUE, 1.0);
        // The parcel a fraction of the way along the duct, from its entry to its exit.
        let (entry, exit) = (
            Point3::xyz(-2.0, -0.65, 0.82),
            Point3::xyz(2.0, -0.65, 0.82),
        );
        let parcel = entry + (exit - entry).gp(fraction);
        s.dot(
            parcel,
            Marker::Dot,
            0.1 * square.width(),
            charts::thermal(warmth).faded(1.5),
        );
    });
}

/// The chart: the lines, the region above saturation, the path so far, and the directions.
fn chart(c: &mut Canvas, rect: Rect, sc: &Scene, travelled: usize) {
    let crossing = [TEMPERATURE_RANGE[0], 0.0];
    let d = Diagram::new(rect, TEMPERATURE_RANGE, WATER_RANGE, crossing, ["T", "W"]);
    d.area(c, &sc.above, charts::GREY.faded(0.4), 0.25);
    for (_, lines) in &sc.humidities {
        d.family(c, lines, Family::Humidity);
    }
    for (_, lines) in &sc.enthalpies {
        d.family(c, lines, Family::Enthalpy);
    }
    d.lines(c, &sc.saturation, MEDIUM, charts::BOUNDARY);
    // Each enthalpy line's value at its upper end, where it meets the saturation curve; each
    // humidity line's at its right end; each only well inside the chart.
    let reading = |p: &Point2, k: usize| f64::from(p.to_euclidean()[k]);
    let highest = |line: &Vec<Point2>, k: usize| {
        line.iter()
            .copied()
            .max_by(|a, b| reading(a, k).total_cmp(&reading(b, k)))
    };
    for (level, lines) in &sc.enthalpies {
        for top in lines.iter().filter_map(|l| highest(l, 1)) {
            let on_saturation = relative_humidity_of(state(
                reading(&top, 0),
                reading(&top, 1) / GRAMS_PER_KILOGRAM,
            )) > 0.9;
            let inside =
                reading(&top, 0) > TEMPERATURE_RANGE[0] && reading(&top, 1) < WATER_RANGE[1] - 1.0;
            if inside && on_saturation {
                let light = Family::Enthalpy.light();
                d.label(
                    c,
                    top,
                    &format!("{level}"),
                    light,
                    charts::TICK,
                    charts::RB,
                    [-2.0, 2.0],
                );
            }
        }
    }
    for (level, lines) in &sc.humidities {
        for right in lines.iter().filter_map(|l| highest(l, 0)) {
            if reading(&right, 1) < WATER_RANGE[1] - 1.0 {
                let text = format!("{:.0}%", level * 100.0);
                let light = Family::Humidity.light();
                d.label(
                    c,
                    right,
                    &text,
                    light,
                    charts::TICK,
                    charts::RM,
                    [-2.0, 0.0],
                );
            }
        }
    }
    let name = charts::at(SATURATION_LABEL[0], SATURATION_LABEL[1]);
    d.label(
        c,
        name,
        "SATURATION",
        charts::BOUNDARY,
        charts::TEXT,
        charts::LB,
        [0.0, 2.0],
    );
    let path: Vec<Point2> = sc.run.path.iter().map(|x| chart_point(*x)).collect();
    d.polyline(c, &path, THIN, charts::ORANGE.faded(0.6));
    d.polyline(c, &path[..=travelled], BOLD, charts::ORANGE);
    for (x, direction) in sc.run.states.iter().zip(sc.run.directions) {
        let (from, aim) = (chart_point(*x), chart_point(*x + direction));
        d.arrow(c, from, aim, ARROW_PIXELS, MEDIUM, charts::GREEN);
    }
    d.dots(c, &[path[travelled]], charts::ORANGE);
    d.ticks(c, &TEMPERATURE_TICKS, &WATER_TICKS);
    d.axes(c);
    let run = &sc.run;
    let note = format!(
        "COIL {:.1} KJ/KG, {:.1} G/KG WATER; REHEAT {:.1} KJ/KG; SUPPLY {:.0}%",
        run.cooling,
        run.condensate * GRAMS_PER_KILOGRAM,
        run.reheating,
        run.supply * 100.0
    );
    d.note(c, &note);
}

fn draw(c: &mut Canvas, sc: &Scene, t: f32) {
    gax_numga_examples::backdrop(c);
    let last = sc.run.path.len() - 1;
    let travelled = ((t / FRAME_SECONDS) as usize).min(last);
    let x = sc.run.path[travelled];
    let panels = charts::row(c, &[1.0, 1.5]);
    let warmths: Vec<f64> = sc
        .run
        .path
        .iter()
        .map(|x| temperature().of(*x).s())
        .collect();
    let (cold, hot) = (warmths.iter().copied().fold(f64::MAX, f64::min), warmths[0]);
    let water = |x: State| humidity().of(x).s();
    let liquid =
        (water(sc.run.path[0]) - water(x)) / (water(sc.run.path[0]) - water(sc.run.path[last]));
    let warmth = (warmths[travelled] - cold) / (hot - cold);
    apparatus(c, panels[0], travelled as f32 / last as f32, warmth, liquid);
    chart(c, panels[1], sc, travelled);
    caption(
        c,
        "PROPERTY CHARTS: AIR CONDITIONING ON THE PSYCHROMETRIC CHART",
        "A HELD READING IS A CONSTRAINT; WITH DT IT IS ONE MAP, SOLVED FOR ONE DEGREE WARMER (T0W0)",
    );
}

fn main() {
    let seconds = FRAME_SECONDS * (2 * STEPS - 1) as f32 + HOLD;
    run_with(
        Anim::new("property_charts", seconds).size(960, 540),
        scene,
        draw,
    );
}

#[cfg(test)]
mod tests {
    use super::psychrometrics::*;

    /// Dry air keeps its water as it cools; humid air saturates and then loses water without
    /// passing saturation; the dew point saturates the air.
    #[test]
    fn dry_cooling_and_condensation_follow_their_constraints() {
        let fractions: Vec<f64> = (0..40).map(|k| k as f64 / 39.0).collect();
        for (case, relative) in [0.2, 0.7].into_iter().enumerate() {
            let initial = air_state(30.0, relative, PRESSURE);
            let path = cooling_path(initial, 12.0, PRESSURE, &fractions);
            let water: Vec<f64> = path.iter().map(|x| humidity().of(*x).s()).collect();
            let relatives: Vec<f64> = path
                .iter()
                .map(|x| relative_humidity(*x, PRESSURE))
                .collect();
            if case == 0 {
                assert!(water.iter().all(|w| (w - water[0]).abs() < 1e-12));
            } else {
                assert!(water.windows(2).all(|p| p[1] - p[0] <= 1e-12));
                assert!((relatives[39] - 1.0).abs() < 1e-10);
            }
            assert!(relatives.iter().all(|r| *r <= 1.0 + 1e-10));
            let dew = state(dew_point(initial, PRESSURE), humidity().of(initial).s());
            assert!((relative_humidity(dew, PRESSURE) - 1.0).abs() < 1e-10);
        }
    }

    /// The coil and the reheater close their water and energy balances; the coil removes 40 to
    /// 50 kJ and 9 to 11 g of water per kg of dry air, the supply is at 50 to 60% relative
    /// humidity; and the directions keep their held readings.
    #[test]
    fn coil_and_reheater_close_mass_and_energy_balances() {
        let run = conditioning(32);
        let [initial, cooled] = run.states;
        let end = run.path[run.path.len() - 1];
        let water = |x: State| humidity().of(x).s();
        assert!((water(initial) - water(end) - run.condensate).abs() < 1e-11);
        let condensate_heat = run.condensate * LIQUID_HEAT * temperature().of(cooled).s();
        let balance =
            enthalpy(initial) - enthalpy(end) - run.cooling - condensate_heat + run.reheating;
        assert!(balance.abs() < 1e-10, "{balance}");
        assert!(40.0 < run.cooling && run.cooling < 50.0, "{}", run.cooling);
        assert!(0.009 < run.condensate && run.condensate < 0.011);
        assert!(0.5 < run.supply && run.supply < 0.6);
        for direction in run.directions {
            assert!((temperature().of(direction).s() - 1.0).abs() < 1e-10);
        }
        assert!(humidity().of(run.directions[0]).s().abs() < 1e-10);
        let (_, constraint) = gax::jet::gradient(relative_humidity_of, cooled);
        assert!(constraint.of(run.directions[1]).s().abs() < 1e-10);
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws_with(super::draw, &super::scene(), 2.0);
    }
}
