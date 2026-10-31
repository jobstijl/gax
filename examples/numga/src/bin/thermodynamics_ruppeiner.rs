//! numga's `thermodynamics/ruppeiner`: the fluctuation metric of a van der Waals fluid and its
//! curvature near the critical point, in the metric-free algebra `t0v0` (declared here with
//! `gax::algebra!`; the metric is the fluid's, not the algebra's).
//!
//! A state of the fluid is an intrinsic vector (numga's antivector) spanned by temperature,
//! `t.dual()`, and volume, `v.dual()`. The metric is a map from a variation of the state to the
//! gradient of half its cost (`Vector <- State`), which pairs with the variation to the
//! variation's cost: how improbable a fluctuation that large is. It is built from the Helmholtz
//! energy's second derivative in its own variables (a jet Hessian), the thermal part with its
//! sign turned, both over the temperature; near the critical point the pressure hardly changes
//! with the volume, the mechanical part of the cost vanishes, and fluctuations of the volume grow
//! cheap. The metric has no cross term, and its scalar curvature needs the mechanical entry's
//! first and second changes along the temperature: a Hessian of a function that is itself a
//! Hessian, so jets of jets of jets of jets, fourth derivatives of the Helmholtz energy, in one
//! pass. The curvature grows without bound toward the critical point, as the inverse square of
//! the distance in temperature.
//!
//! The animation: the plane of volume and temperature, the unstable region under the spinodal
//! shaded, the ellipses of equal cost about a lattice of states, the curvature's level lines
//! crowding toward the critical point, and the cooling state's own ellipse stretching along the
//! volume; the isotherms with the state's own and its flattening tangent; and the curvature
//! against the distance above the critical temperature, on logarithmic scales, beside a line of
//! slope -2. Temperature and volume are in their critical values, pressure in `R T_c / V_c`,
//! energy in `R T_c`.

#[path = "../shared/thermodynamics_charts.rs"]
mod charts;

gax::algebra! {
    algebra fluid "Temperature and volume, `t0v0`: both generators square to zero.";
    basis et = 0, ev = 0;
    kind Scalar = [1];
    kind Vector = [et, ev];
    kind Pseudoscalar = [etv];
}

use charts::{BOLD, Diagram, Family, Floating, MEDIUM, THIN};
use gax_numga_examples::points::Pos2;
use gax_numga_examples::{Anim, Canvas, Point2, Rect, caption, plot, run_with};

mod fluctuations {
    use super::fluid::{Scalar, Vector, intrinsic};
    use gax::Real;
    use gax::jet::{gradient, hessian};

    /// A state: temperature and volume, an intrinsic vector (numga's `Antivector-`).
    pub type State<T = f64> = intrinsic::Vector<(), T>;
    /// The metric, `Gradient <- State`.
    pub type Metric = Vector<(intrinsic::Vector,), f64>;
    /// `Scalar <- State`.
    pub type Reading = Scalar<(intrinsic::Vector,), f64>;

    pub fn t() -> Vector {
        Vector::new(1.0, 0.0)
    }
    pub fn v() -> Vector {
        Vector::new(0.0, 1.0)
    }
    pub fn state(temperature: f64, volume: f64) -> State {
        t().dual() * temperature + v().dual() * volume
    }
    /// Regressive pairings read coordinates without assuming a metric.
    pub fn temperature() -> Reading {
        t() & State::slot()
    }
    pub fn volume() -> Reading {
        v() & State::slot()
    }

    /// A gradient's reading of a state with coefficients of any type (jets included).
    fn read<T: Real>(g: Vector, x: State<T>) -> T {
        (g.map_coefs(T::from_f64) & x).s()
    }

    /// The fluid: the strength of its molecules' attraction, the volume they exclude, and its
    /// heat capacity at fixed volume.
    #[derive(Clone, Copy, Debug)]
    pub struct Fluid {
        pub attraction: f64,
        pub exclusion: f64,
        pub capacity: f64,
    }

    impl Fluid {
        /// The van der Waals equation of state.
        pub fn pressure<T: Real>(&self, x: State<T>) -> T {
            let (temperature, volume) = (read(t(), x), read(v(), x));
            temperature / (volume - T::from_f64(self.exclusion))
                - T::from_f64(self.attraction) / (volume * volume)
        }

        pub fn helmholtz_energy<T: Real>(&self, x: State<T>) -> T {
            let (temperature, volume) = (read(t(), x), read(v(), x));
            let c = T::from_f64;
            -c(self.capacity) * temperature * temperature.ln()
                - temperature * (volume - c(self.exclusion)).ln()
                - c(self.attraction) / volume
        }

        /// The metric's two entries, thermal and mechanical: the Helmholtz energy's second
        /// derivative in temperature and volume, its thermal part with the sign turned, both
        /// over the temperature. In these variables the cost has no cross term.
        pub fn entries<T: Real>(&self, x: State<T>) -> [T; 2] {
            let (_, _, second) = hessian(|y| self.helmholtz_energy(y), x);
            let (t_dual, v_dual) = (
                t().dual().map_coefs(T::from_f64),
                v().dual().map_coefs(T::from_f64),
            );
            let temperature = read(t(), x);
            let thermal = -second.of(t_dual).of(t_dual).s() / temperature;
            let mechanical = second.of(v_dual).of(v_dual).s() / temperature;
            [thermal, mechanical]
        }

        /// The cost of variations about a state, as a map from a variation to a gradient.
        pub fn metric(&self, x: State) -> Metric {
            let [thermal, mechanical] = self.entries(x);
            t() * temperature() * thermal + v() * volume() * mechanical
        }

        /// The metric's scalar curvature, twice its Gaussian curvature, positive on a sphere. The
        /// metric has no cross term and its thermal entry does not change with the volume: the
        /// curvature needs only the mechanical entry's changes along the temperature and the
        /// change of the squared area the two entries span.
        pub fn curvature(&self, x: State) -> f64 {
            let t_dual = t().dual();
            let (_, slope, change) = hessian(|y| self.entries(y)[1], x);
            let (slope, slope_change) = (slope.of(t_dual).s(), change.of(t_dual).of(t_dual).s());
            let area_squared = |y: State<_>| {
                let [thermal, mechanical] = self.entries(y);
                thermal * mechanical
            };
            let (area, area_change) = gradient(area_squared, x);
            -slope_change / area + slope * area_change.of(t_dual).s() / (2.0 * area * area)
        }
    }

    // --- the scenario: states approaching the critical point ----------------------------

    pub const FLUID: Fluid = Fluid {
        attraction: 9.0 / 8.0,
        exclusion: 1.0 / 3.0,
        capacity: 1.5,
    };

    /// States at the critical volume cooling through temperatures toward the critical one.
    pub fn approach(temperatures: &[f64]) -> Vec<State> {
        temperatures.iter().map(|&t| state(t, 1.0)).collect()
    }
}

use fluctuations::*;

/// numga's run: 96 frames from 1.8 to 1.02, 40 ms each.
const FRAMES: usize = 96;
const FRAME_SECONDS: f32 = 0.04;
const HOLD: f32 = 1.2;
const WARM: f64 = 1.8;
const COOL: f64 = 1.02;

/// The charts' ranges: volume and temperature, volume and pressure.
const VOLUME_RANGE: [f64; 2] = [0.0, 3.0];
const TEMPERATURE_RANGE: [f64; 2] = [0.7, 2.0];
const PRESSURE_RANGE: [f64; 2] = [0.0, 1.0];
/// The curvature's levels; the lattice of states whose ellipses are drawn, and the cost they are
/// drawn at; the isotherms' temperatures; the tangent's half span in volume.
const CURVATURE_LEVELS: [f64; 6] = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0];
const LATTICE_VOLUMES: [f64; 6] = [0.5, 0.9, 1.3, 1.7, 2.1, 2.5];
const LATTICE_TEMPERATURES: [f64; 6] = [0.95, 1.15, 1.35, 1.55, 1.75, 1.95];
const COST: f64 = 0.0036;
const ISOTHERMS: [f64; 8] = [0.9, 1.0, 1.1, 1.25, 1.4, 1.6, 1.8, 2.0];
const TANGENT_SPAN: f64 = 0.35;
/// The scaling chart's ranges: the logarithms of the distance above the critical temperature
/// and of the curvature.
const DISTANCE_RANGE: [f64; 2] = [-3.0, 0.0];
const LOG_CURVATURE_RANGE: [f64; 2] = [-0.5, 5.5];

/// A state on the chart of temperature up against volume across, read by pairing.
fn plane_point(x: State) -> Point2 {
    charts::at(volume().of(x).s(), temperature().of(x).s())
}

/// A state on the chart of pressure up against volume across.
fn isotherm_point(x: State) -> Point2 {
    charts::at(volume().of(x).s(), FLUID.pressure(x))
}

/// The ellipse of variations of equal cost about a state, on the chart: the cost
/// `mechanical dV² + thermal dT²` at the cost level, `plot::ellipse` of the variances
/// `cost / (4 entry)` along the chart's two axes (it draws twice the root).
fn ellipse(x: State) -> Vec<Point2> {
    // The metric's entries: its image of each unit variation, paired with that variation.
    let form = FLUID.metric(x);
    let entry = |step: State| (form.of(step) & step).s();
    let (thermal, mechanical) = (entry(t().dual()), entry(v().dual()));
    let axes = [
        gax::vga2d::Vector::new(1.0, 0.0),
        gax::vga2d::Vector::new(0.0, 1.0),
    ];
    let variances = [COST / (4.0 * mechanical), COST / (4.0 * thermal)];
    let [vol, temp] = [volume().of(x).s(), temperature().of(x).s()];
    let centre = gax::pga2d::Point::xy(vol, temp);
    (plot::ellipse(centre, variances, axes, 48).into_iter())
        .map(|p| p.point2())
        .collect()
}

/// What is computed once: the states, the plane's regions, lines and ellipses, the isotherms,
/// and the curvature against the distance from the critical temperature.
struct Scene {
    states: Vec<State>,
    spinodal: Vec<Vec<Point2>>,
    curvature_lines: Vec<Vec<Point2>>,
    ellipses: Vec<Vec<Point2>>,
    lattice: Vec<Point2>,
    isotherms: Vec<(f64, Vec<Point2>)>,
    volumes: Vec<f64>,
    scaling: Vec<Point2>,
}

fn sample(range: [f64; 2], n: usize) -> Vec<f64> {
    (0..n)
        .map(|k| range[0] + (range[1] - range[0]) * k as f64 / (n - 1) as f64)
        .collect()
}

fn scene() -> Scene {
    let temperatures = sample([WARM, COOL], FRAMES);
    let (grid_volumes, grid_temperatures) =
        (sample([0.36, 3.0], 160), sample(TEMPERATURE_RANGE, 160));
    let grid: Vec<State> = grid_temperatures
        .iter()
        .flat_map(|&t| grid_volumes.iter().map(move |&v| state(t, v)))
        .collect();
    // The mechanical part of the cost, and the curvature only where it is positive.
    let mechanical: Vec<f64> = grid.iter().map(|x| FLUID.entries(*x)[1]).collect();
    let curvature: Vec<f64> = (grid.iter().zip(&mechanical))
        .map(|(x, m)| {
            if *m > 0.0 {
                FLUID.curvature(*x)
            } else {
                f64::NAN
            }
        })
        .collect();
    let lines = |values: &[f64], level: f64| {
        charts::level_lines(&grid_volumes, &grid_temperatures, values, level)
    };
    // The lattice's states where every variation costs: both entries positive.
    let stable: Vec<State> = LATTICE_TEMPERATURES
        .iter()
        .flat_map(|&t| LATTICE_VOLUMES.iter().map(move |&v| state(t, v)))
        .filter(|x| FLUID.entries(*x).iter().all(|e| *e > 0.0))
        .collect();
    let volumes = sample([0.38, VOLUME_RANGE[1]], 240);
    let isotherms = (ISOTHERMS.iter())
        .map(|&t| {
            (
                t,
                volumes
                    .iter()
                    .map(|&v| isotherm_point(state(t, v)))
                    .collect(),
            )
        })
        .collect();
    // Distances above the critical temperature by equal ratios, along the critical volume.
    let scaling = (0..100)
        .map(|k| {
            let offset = 0.002 * 400f64.powf(k as f64 / 99.0);
            let r = FLUID.curvature(state(1.0 + offset, 1.0));
            charts::at(offset.log10(), r.log10())
        })
        .collect();
    Scene {
        states: approach(&temperatures),
        spinodal: lines(&mechanical, 0.0),
        curvature_lines: CURVATURE_LEVELS
            .iter()
            .flat_map(|&l| lines(&curvature, l))
            .collect(),
        ellipses: stable.iter().map(|x| ellipse(*x)).collect(),
        lattice: stable.iter().map(|x| plane_point(*x)).collect(),
        isotherms,
        volumes,
        scaling,
    }
}

/// The plane of volume and temperature: the unstable region under the spinodal, the curvature's
/// level lines, the lattice's ellipses, the critical point, and the moving state's ellipse.
fn plane(c: &mut Canvas, rect: Rect, sc: &Scene, frame: usize) {
    let crossing = [0.0, TEMPERATURE_RANGE[0]];
    let d = Diagram::new(rect, VOLUME_RANGE, TEMPERATURE_RANGE, crossing, ["V", "T"]);
    for line in &sc.spinodal {
        d.area(c, line, charts::GREY.faded(0.4), 0.3);
    }
    d.lines(c, &sc.curvature_lines, THIN, Family::Level.light());
    d.lines(c, &sc.ellipses, THIN, Family::Entropy.light());
    d.lines(c, &sc.spinodal, MEDIUM, charts::BOUNDARY);
    let critical = charts::at(1.0, 1.0);
    d.dots(c, &[critical], charts::BOUNDARY);
    let names = d.place(
        c,
        &[
            Floating::along(&sc.curvature_lines, 6, "R = CONST", Family::Level.light()),
            Floating {
                anchor: charts::LT,
                ..Floating::along(
                    std::slice::from_ref(&sc.lattice),
                    1,
                    "EQUAL COST",
                    Family::Entropy.light(),
                )
            },
        ],
    );
    let path: Vec<Point2> = sc.states[..=frame]
        .iter()
        .map(|x| plane_point(*x))
        .collect();
    d.polyline(c, &path, BOLD, charts::ORANGE.faded(0.6));
    let own = ellipse(sc.states[frame]);
    d.area(c, &own, charts::ORANGE.faded(0.5), 0.25);
    d.polyline(c, &own, MEDIUM, charts::ORANGE);
    d.dots(c, &path[frame..], charts::ORANGE);
    let text = charts::TEXT;
    d.label(
        c,
        critical,
        "CRITICAL POINT",
        charts::BOUNDARY,
        text,
        charts::LT,
        [6.0, -4.0],
    );
    let unstable = charts::at(0.45, 0.75);
    d.label(
        c,
        unstable,
        "UNSTABLE",
        charts::INK,
        text,
        charts::LB,
        [0.0, 2.0],
    );
    d.write_names(c, &names);
    d.axes(c);
    d.note(c, "FLUCTUATIONS OF EQUAL COST");
}

/// The isotherms, the state's own bold, and its tangent: a unit of volume, and the change of the
/// pressure along it (a jet gradient), whose slope flattens toward the critical point.
fn isotherms(c: &mut Canvas, rect: Rect, sc: &Scene, frame: usize) {
    let d = Diagram::new(rect, VOLUME_RANGE, PRESSURE_RANGE, [0.0, 0.0], ["V", "P"]);
    for (t, line) in &sc.isotherms {
        let light = if *t == 1.0 {
            charts::BOUNDARY
        } else {
            Family::Temperature.light()
        };
        d.polyline(c, line, THIN, light);
    }
    let lines: Vec<Vec<Point2>> = sc.isotherms.iter().map(|(_, l)| l.clone()).collect();
    let light = Family::Temperature.light();
    let names = d.place(c, &[Floating::along(&lines, 8, "T = CONST", light)]);
    let x = sc.states[frame];
    let temp = temperature().of(x).s();
    let own: Vec<Point2> = (sc.volumes.iter())
        .map(|&v| isotherm_point(state(temp, v)))
        .collect();
    d.polyline(c, &own, BOLD, charts::INK);
    let (_, slope) = gax::jet::gradient(|y| FLUID.pressure(y), x);
    let tangent = charts::toward(1.0, slope.of(v().dual()).s());
    let here = isotherm_point(x);
    let v0 = volume().of(x).s();
    let reach = [v0 - TANGENT_SPAN, v0 + TANGENT_SPAN];
    d.tangent(c, here, tangent, reach, MEDIUM, charts::ORANGE);
    d.dots(c, &[here], charts::ORANGE);
    d.write_names(c, &names);
    d.axes(c);
    d.note(c, "THE TANGENT FLATTENS");
}

/// The curvature against the distance above the critical temperature, both on logarithmic
/// scales, beside a dashed line falling as the inverse square.
fn scaling(c: &mut Canvas, rect: Rect, sc: &Scene, frame: usize) {
    let crossing = [DISTANCE_RANGE[0], LOG_CURVATURE_RANGE[0]];
    let names = ["LOG(T - 1)", "LOG R"];
    let d = Diagram::new(rect, DISTANCE_RANGE, LOG_CURVATURE_RANGE, crossing, names);
    // The line falling as the inverse square, through a point a little above the curve's first.
    let first = sc.scaling[0];
    let through = first + Point2::direction(0.0, 0.4);
    let falling = charts::toward(1.0, -2.0);
    let reference = charts::tangent_points(through, falling, &sample(DISTANCE_RANGE, 60));
    d.dashed(c, &reference, MEDIUM, 4.0, charts::GREY);
    d.polyline(c, &sc.scaling, BOLD, charts::BLUE);
    let mark = charts::tangent_points(through, falling, &[-1.0])[0];
    d.label(
        c,
        mark,
        "SLOPE -2",
        charts::GREY,
        charts::TEXT,
        charts::LB,
        [6.0, 2.0],
    );
    // The moving state on the curve.
    let x = sc.states[frame];
    let here = charts::at(
        (temperature().of(x).s() - 1.0).log10(),
        FLUID.curvature(x).log10(),
    );
    d.dots(c, &[here], charts::ORANGE);
    d.axes(c);
    d.note(c, "R GROWS AS (T - 1) TO THE -2");
}

fn draw(c: &mut Canvas, sc: &Scene, t: f32) {
    gax_numga_examples::backdrop(c);
    let frame = ((t / FRAME_SECONDS) as usize).min(FRAMES - 1);
    let panels = charts::row(c, &[1.0, 1.0, 1.0]);
    plane(c, panels[0], sc, frame);
    isotherms(c, panels[1], sc, frame);
    scaling(c, panels[2], sc, frame);
    caption(
        c,
        "RUPPEINER: THE FLUCTUATION METRIC OF A VAN DER WAALS FLUID",
        "THE METRIC, A MAP FROM VARIATIONS TO GRADIENTS; ITS CURVATURE FROM FOURTH-ORDER JETS (T0V0)",
    );
}

fn main() {
    run_with(
        Anim::new("ruppeiner", FRAME_SECONDS * FRAMES as f32 + HOLD).size(960, 540),
        scene,
        draw,
    );
}

#[cfg(test)]
mod tests {
    use super::fluctuations::*;

    /// The metric's thermal entry is the heat capacity over the squared temperature, its
    /// mechanical entry minus the pressure's change with volume over the temperature (positive
    /// at these stable states), and it has no cross term.
    #[test]
    fn metric_matches_heat_capacity_and_pressure_response() {
        for (temp, vol) in [(1.02, 1.0), (1.2, 0.8), (1.8, 1.5)] {
            let x = state(temp, vol);
            let form = FLUID.metric(x);
            let (_, pressure_change) = gax::jet::gradient(|y| FLUID.pressure(y), x);
            let thermal = (form.of(t().dual()) & t().dual()).s();
            let mechanical = (form.of(v().dual()) & v().dual()).s();
            assert!((thermal - 1.5 / (temp * temp)).abs() < 1e-11);
            let expected = -pressure_change.of(v().dual()).s() / temp;
            assert!((mechanical - expected).abs() < 1e-11);
            assert!(mechanical > 0.0);
            assert!((form.of(t().dual()) & v().dual()).s().abs() < 1e-11);
        }
    }

    /// The curvature from fourth-order jets matches an independent closed expression, vanishes
    /// for the ideal gas, and grows as `1 / (3 (T - 1)²)` at the critical volume.
    #[test]
    fn curvature_matches_independent_closed_expression_and_flat_limit() {
        let (a, b, capacity) = (9.0 / 8.0, 1.0 / 3.0, 1.5);
        let ideal = Fluid {
            attraction: 0.0,
            exclusion: 0.0,
            capacity,
        };
        let cases = [(1.002, 1.0), (1.2, 0.8), (1.8, 1.5)];
        for (temp, vol) in cases {
            let x = state(temp, vol);
            let bracket = temp * vol.powi(3) - 2.0 * a * (vol - b).powi(2);
            let expected =
                2.0 * a * (vol - b).powi(2) * (temp * vol.powi(3) - a * (vol - b).powi(2))
                    / (capacity * bracket * bracket);
            let result = FLUID.curvature(x);
            assert!(
                (result - expected).abs() <= 1e-10 + 1e-8 * expected.abs(),
                "{result} {expected}"
            );
            assert!(ideal.curvature(x).abs() < 1e-10);
        }
        let near = FLUID.curvature(state(cases[0].0, cases[0].1));
        assert!((near * 3.0 * (cases[0].0 - 1.0f64).powi(2) - 1.0).abs() < 0.01);
    }

    /// On the spinodal, where the pressure's slope along the volume vanishes, so does the
    /// mechanical cost.
    #[test]
    fn spinodal_is_where_the_mechanical_cost_vanishes() {
        for vol in [0.8, 1.0, 1.5] {
            let temp = 2.0 * 9.0 / 8.0 * (vol - 1.0 / 3.0f64).powi(2) / (vol * vol * vol);
            let form = FLUID.metric(state(temp, vol));
            assert!((form.of(v().dual()) & v().dual()).s().abs() < 1e-11);
        }
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws_with(super::draw, &super::scene(), 2.0);
    }
}
