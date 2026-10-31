//! numga's `thermodynamics/phase_change`: coexisting liquid and vapour of a van der Waals fluid,
//! and boiling between them at fixed temperature and pressure, in the metric-free algebra `l0g0`
//! (declared here with `gax::algebra!`).
//!
//! A state of coexistence holds the two phases' volumes, an intrinsic vector (numga's
//! antivector) spanned by the liquid's, `l.dual()`, and the vapour's, `g.dual()`, read back by
//! the regressive product. Coexistence is two conditions on the two volumes, equal pressure and
//! equal chemical potential, which say together that the Helmholtz energy's tangent lines at the
//! two volumes are one line, the common tangent. Their mismatch is a function from states to
//! states; Newton's method solves it with the map its jet derivative gives at each guess, from a
//! correction to the change of the mismatch (`State <- State`, solved for the correction that
//! cancels it), eight steps from a guess on either side of the critical point. The volume where
//! an isobar reaches a temperature is the same Newton's method on the algebra's scalars. A
//! mixture of the two phases lies on the common tangent, its volume and its enthalpy shared out
//! by the vapour's fraction, and heat taken in at fixed pressure is the enthalpy's change, so a
//! heated fluid walks along its isobar in equal steps of enthalpy.
//!
//! The animation: the cylinder under its loaded piston, liquid at the bottom and vapour above;
//! the pressure-volume chart, the isotherm at the boiling temperature crossing its loop at the
//! saturation pressure and cutting off two equal areas; the temperature-enthalpy chart, the
//! isobars flat across the dome while the fluid boils; and the phase diagram, the saturation
//! curve ending at the critical point. Temperature and volume are in their critical values,
//! energy in `R T_c`, pressure in `R T_c / V_c`.

#[path = "../../shared/thermodynamics_charts.rs"]
mod charts;

gax::algebra! {
    algebra phases "The liquid's and the vapour's volumes, `l0g0`: both generators square to zero.";
    basis el = 0, eg = 0;
    kind Scalar = [1];
    kind Vector = [el, eg];
    kind Pseudoscalar = [elg];
}

use charts::{BOLD, Diagram, Family, Floating, Label, MEDIUM, THIN};
use gax_colour::Light;
use gax_numga_examples::scene3::panel3;
use gax_numga_examples::{Anim, Canvas, Point2, Rect, caption, run_with};

mod coexistence;
use coexistence::*;

/// numga's run: boiling at 0.85 from 0.7 to 0.98, 105 frames of 40 ms.
const BOILING: f64 = 0.85;
const COLDEST: f64 = 0.7;
const HOTTEST: f64 = 0.98;
const FRAMES: usize = 105;
const FRAME_SECONDS: f32 = 0.04;
const HOLD: f32 = 1.2;

/// The charts' ranges and their families' levels.
const VOLUME_RANGE: [f64; 2] = [0.0, 4.4];
const PRESSURE_RANGE: [f64; 2] = [0.0, 0.44];
const ISOTHERMS: [f64; 9] = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1];
const ENTHALPY_RANGE: [f64; 2] = [-1.6, 2.4];
const TEMPERATURE_RANGE: [f64; 2] = [0.6, 1.12];
const PHASE_PRESSURE_RANGE: [f64; 2] = [0.0, 0.55];
const ISOCHORES: [f64; 7] = [0.5, 0.6, 0.8, 1.0, 1.4, 2.0, 3.0];
const ISOBAR_TEMPERATURES: [f64; 6] = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95];
const VAPOUR_FRACTIONS: [f64; 4] = [0.2, 0.4, 0.6, 0.8];
const SAMPLES: usize = 160;
/// The critical point; the cylinder's height of a unit of volume.
const CRITICAL: [f64; 3] = [1.0, 3.0 / 8.0, 1.0];
const HEIGHT_PER_VOLUME: f64 = 0.55;

/// The liquid's colour, and the vapour's.
const LIQUID: Light = charts::BLUE;
fn vapour_light() -> Light {
    gax_colour::hex(0x9ccfe0, 0.9)
}

/// What is computed once: the heating, and every chart's fixed curves.
struct Scene {
    steps: Vec<Heated>,
    saturation: f64,
    isotherms: Vec<(f64, Vec<Point2>)>,
    pressure_dome: Vec<Point2>,
    lobes: [Vec<Point2>; 2],
    heat_dome: Vec<Point2>,
    fractions: Vec<Vec<Point2>>,
    isobars: Vec<(f64, Vec<Point2>)>,
    saturation_curve: Vec<Point2>,
    isochores: Vec<Vec<Point2>>,
}

fn sample(a: f64, b: f64, n: usize) -> Vec<f64> {
    (0..n)
        .map(|k| a + (b - a) * k as f64 / (n - 1) as f64)
        .collect()
}

/// The isobar at a saturation pressure on the temperature-enthalpy chart: the liquid squeezed
/// toward its excluded volume up to boiling, the fluid boiling, and the vapour expanding beyond,
/// each by equal ratios.
fn isobar(coexisting: State, saturation: f64) -> Vec<Point2> {
    let b = FLUID.exclusion;
    let liquid = liquid_volume().of(coexisting).s();
    let vapour = vapour_volume().of(coexisting).s();
    let squeezed =
        (0..SAMPLES).map(|k| b + (liquid - b) * 20f64.powf(k as f64 / (SAMPLES - 1) as f64 - 1.0));
    let expansion = 4.0 * TEMPERATURE_RANGE[1] / (saturation * vapour);
    let expanded = (0..SAMPLES).map(|k| vapour * expansion.powf(k as f64 / (SAMPLES - 1) as f64));
    squeezed
        .chain(expanded)
        .map(|v| {
            let t = FLUID.isobar_temperature(v, saturation);
            charts::at(FLUID.enthalpy(v, t), t)
        })
        .collect()
}

fn scene() -> Scene {
    // The dome: the coexisting volumes up to the critical point.
    let dome_temperatures = sample(0.6, 0.995, 40);
    let domes: Vec<(f64, f64, f64)> = (dome_temperatures.iter())
        .map(|&t| {
            let x = coexistence_at(t);
            (t, liquid_volume().of(x).s(), vapour_volume().of(x).s())
        })
        .collect();
    let [cv, cp, ct] = CRITICAL;
    let level = |(t, liquid, _): &(f64, f64, f64)| FLUID.pressure(*liquid, *t);
    let pressure_dome = (domes.iter().map(|d| charts::at(d.1, level(d))))
        .chain([charts::at(cv, cp)])
        .chain(domes.iter().rev().map(|d| charts::at(d.2, level(d))))
        .collect();
    let liquid_heat: Vec<Point2> = (domes.iter())
        .map(|(t, l, _)| charts::at(FLUID.enthalpy(*l, *t), *t))
        .collect();
    let vapour_heat: Vec<Point2> = (domes.iter())
        .map(|(t, _, g)| charts::at(FLUID.enthalpy(*g, *t), *t))
        .collect();
    let heat_dome = (liquid_heat.iter().copied())
        .chain([charts::at(FLUID.enthalpy(cv, ct), ct)])
        .chain(vapour_heat.iter().rev().copied())
        .collect();
    // The mixtures of equal vapour fraction across the dome: points shared out between the
    // phases' points at each temperature.
    let fractions = (VAPOUR_FRACTIONS.iter())
        .map(|&x| {
            (liquid_heat.iter().zip(&vapour_heat))
                .map(|(a, b)| a.gp(1.0 - x as f32) + b.gp(x as f32))
                .collect()
        })
        .collect();
    let volumes = sample(0.36, VOLUME_RANGE[1], 2 * SAMPLES);
    let isotherms = (ISOTHERMS.iter())
        .map(|&t| {
            (
                t,
                (volumes.iter())
                    .map(|&v| charts::at(v, FLUID.pressure(v, t)))
                    .collect(),
            )
        })
        .collect();
    let saturation_curve = (domes.iter().map(|d| charts::at(d.0, level(d))))
        .chain([charts::at(ct, cp)])
        .collect();
    let isochores = (ISOCHORES.iter())
        .map(|&v| {
            (sample(TEMPERATURE_RANGE[0], TEMPERATURE_RANGE[1], 60).into_iter())
                .map(|t| charts::at(t, FLUID.pressure(v, t)))
                .collect()
        })
        .collect();
    let isobars = (ISOBAR_TEMPERATURES.iter())
        .map(|&t| {
            let x = coexistence_at(t);
            (t, isobar(x, FLUID.pressure(liquid_volume().of(x).s(), t)))
        })
        .collect();
    // The loop of the isotherm at the boiling temperature, cut where it crosses the saturation
    // pressure into its two lobes, each closed back along the flat stretch.
    let coexisting = coexistence_at(BOILING);
    let (liquid, vapour) = (
        liquid_volume().of(coexisting).s(),
        vapour_volume().of(coexisting).s(),
    );
    let saturation = FLUID.pressure(liquid, BOILING);
    let loop_points: Vec<Point2> = (sample(liquid, vapour, 2 * SAMPLES).into_iter())
        .map(|v| charts::at(v, FLUID.pressure(v, BOILING)))
        .collect();
    let above = |p: &Point2| f64::from(p.to_euclidean()[1]) > saturation;
    let n = loop_points.len();
    let middle = (1..n - 2)
        .find(|&k| above(&loop_points[k]) != above(&loop_points[k + 1]))
        .unwrap_or(n / 2);
    // Where the loop crosses the flat stretch: the line through its samples on either side met
    // with it; and the loop's ends on it, the feet of its first and last samples.
    let flat = charts::up_line(saturation);
    let crossing = ((loop_points[middle] & loop_points[middle + 1]) ^ flat).position();
    let (start, end) = (
        charts::foot(loop_points[0], saturation),
        charts::foot(loop_points[n - 1], saturation),
    );
    let lobes = [
        [&[start][..], &loop_points[1..=middle], &[crossing]].concat(),
        [&[crossing][..], &loop_points[middle + 1..n - 1], &[end]].concat(),
    ];
    Scene {
        steps: heating(BOILING, COLDEST, HOTTEST, FRAMES),
        saturation,
        isotherms,
        pressure_dome,
        lobes,
        heat_dome,
        fractions,
        isobars,
        saturation_curve,
        isochores,
    }
}

/// The names of the three regions a chart of the fluid falls into, at values of its two
/// quantities.
fn regions(places: [[f64; 2]; 3]) -> Vec<Label> {
    let names = ["LIQUID", "LIQUID + VAPOUR", "VAPOUR"];
    (places.into_iter().zip(names))
        .map(|([x, y], name)| Label::new(charts::at(x, y), name, charts::INK, charts::MM, [0.0; 2]))
        .collect()
}

/// The path so far, bold, and a dot at its end.
fn marked(c: &mut Canvas, d: &Diagram, path: &[Point2]) {
    if let Some(last) = path.last() {
        d.polyline(c, path, BOLD, charts::ORANGE);
        d.dots(c, &[*last], charts::ORANGE);
    }
}

/// Lines drawn, in two widths: those at the boiling temperature medium, the rest thin; and the
/// two sets apart.
fn split(
    c: &mut Canvas,
    d: &Diagram,
    lines: &[(f64, Vec<Point2>)],
    l: Light,
) -> [Vec<Vec<Point2>>; 2] {
    let (mut own, mut others) = (Vec::new(), Vec::new());
    for (t, line) in lines {
        let width = if *t == BOILING { MEDIUM } else { THIN };
        d.polyline(c, line, width, l);
        if *t == BOILING { &mut own } else { &mut others }.push(line.clone());
    }
    [own, others]
}

/// A floating name along a single line, its top left at the place.
fn along_one(line: &[Point2], every: usize, text: &str, l: Light) -> Floating {
    Floating {
        anchor: charts::LT,
        ..Floating::along(&[line.to_vec()], every, text, l)
    }
}

/// The pressure-volume chart: the isotherms, the boiling one's two lobes of equal area, the dome
/// of coexisting volumes, and the path at the saturation pressure.
fn pressure_volume(c: &mut Canvas, rect: Rect, sc: &Scene, frame: usize) {
    let d = Diagram::new(rect, VOLUME_RANGE, PRESSURE_RANGE, [0.0, 0.0], ["V", "P"]);
    d.area(c, &sc.lobes[0], LIQUID.faded(0.4), 0.3);
    d.area(c, &sc.lobes[1], charts::ORANGE.faded(0.4), 0.3);
    let isotherm = Family::Temperature.light();
    let [own, others] = split(c, &d, &sc.isotherms, isotherm);
    d.polyline(c, &sc.pressure_dome, MEDIUM, charts::BOUNDARY);
    let critical = charts::at(CRITICAL[0], CRITICAL[1]);
    let mut labels = regions([[0.2, 0.06], [2.0, 0.12], [3.3, 0.33]]);
    labels.push(Label::new(
        critical,
        "CRITICAL POINT",
        charts::BOUNDARY,
        charts::LB,
        [6.0, 3.0],
    ));
    let floating = [
        Floating::along(&own, 6, &format!("T = {BOILING}"), isotherm),
        Floating::along(&others, 6, "T = CONST", isotherm),
        along_one(&sc.pressure_dome, 3, "COEXISTENCE", charts::BOUNDARY),
    ];
    let names = d.place_around(c, &floating, &labels);
    let path: Vec<Point2> = (sc.steps[..=frame].iter())
        .map(|s| charts::at(s.volume, sc.saturation))
        .collect();
    marked(c, &d, &path);
    d.labels(c, &labels);
    d.write_names(c, &names);
    d.axes(c);
    d.note(c, "EQUAL AREAS AT SATURATION");
}

/// The temperature-enthalpy chart: the dome, the lines of equal vapour fraction across it, and
/// the isobars, flat across it while the fluid boils.
fn temperature_enthalpy(c: &mut Canvas, rect: Rect, sc: &Scene, frame: usize) {
    let crossing = [ENTHALPY_RANGE[0], TEMPERATURE_RANGE[0]];
    let names = ["H", "T"];
    let d = Diagram::new(rect, ENTHALPY_RANGE, TEMPERATURE_RANGE, crossing, names);
    let level = Family::Level.light();
    d.lines(c, &sc.fractions, THIN, level);
    let isobar = Family::Pressure.light();
    let [own, others] = split(c, &d, &sc.isobars, isobar);
    d.polyline(c, &sc.heat_dome, MEDIUM, charts::BOUNDARY);
    let critical = sc.heat_dome[sc.heat_dome.len() / 2];
    let mut labels = regions([[-1.15, 1.02], [0.45, 0.665], [2.0, 0.8]]);
    labels.push(Label::new(
        critical,
        "CRITICAL POINT",
        charts::BOUNDARY,
        charts::LB,
        [6.0, 3.0],
    ));
    // Each fraction's value at the bottom of the dome.
    for (line, x) in sc.fractions.iter().zip(VAPOUR_FRACTIONS) {
        let text = format!("{x}");
        let size = charts::TICK;
        labels.push(Label {
            size,
            ..Label::new(line[1], &text, level, charts::MT, [0.0, -2.0])
        });
    }
    let floating = [
        Floating::along(&own, 8, &format!("P = {:.2}", sc.saturation), isobar),
        Floating::along(&others, 8, "P = CONST", isobar),
        Floating::along(&sc.fractions, 4, "X = CONST", level),
    ];
    let names = d.place_around(c, &floating, &labels);
    let path: Vec<Point2> = (sc.steps[..=frame].iter())
        .map(|s| charts::at(s.enthalpy, s.temperature))
        .collect();
    marked(c, &d, &path);
    d.labels(c, &labels);
    d.write_names(c, &names);
    d.axes(c);
    d.note(c, "HEAT AT FIXED PRESSURE IS DH");
}

/// The phase diagram: the saturation pressure against the temperature up to the critical point,
/// the lines of equal volume, and the heated fluid crossing the saturation curve.
fn pressure_temperature(c: &mut Canvas, rect: Rect, sc: &Scene, frame: usize) {
    let crossing = [TEMPERATURE_RANGE[0], 0.0];
    let names = ["T", "P"];
    let d = Diagram::new(
        rect,
        TEMPERATURE_RANGE,
        PHASE_PRESSURE_RANGE,
        crossing,
        names,
    );
    let volume = Family::Volume.light();
    d.lines(c, &sc.isochores, THIN, volume);
    d.polyline(c, &sc.saturation_curve, MEDIUM, charts::BOUNDARY);
    let critical = sc.saturation_curve[sc.saturation_curve.len() - 1];
    d.dots(c, &[critical], charts::BOUNDARY);
    let (ink, mm) = (charts::INK, charts::MM);
    let labels = [
        Label::new(charts::at(0.68, 0.33), "LIQUID", ink, mm, [0.0; 2]),
        Label::new(charts::at(0.98, 0.1), "VAPOUR", ink, mm, [0.0; 2]),
        Label::new(
            critical,
            "CRITICAL POINT",
            charts::BOUNDARY,
            charts::RB,
            [-6.0, 3.0],
        ),
    ];
    let floating = [
        Floating::along(&sc.isochores, 4, "V = CONST", volume),
        along_one(&sc.saturation_curve, 3, "SATURATION", charts::BOUNDARY),
    ];
    let names = d.place_around(c, &floating, &labels);
    let path: Vec<Point2> = (sc.steps[..=frame].iter())
        .map(|s| charts::at(s.temperature, sc.saturation))
        .collect();
    marked(c, &d, &path);
    d.labels(c, &labels);
    d.write_names(c, &names);
    d.axes(c);
    d.note(c, "THE SATURATION CURVE");
}

/// The cylinder under its loaded piston, the liquid at the bottom and the vapour above it,
/// coloured by how warm it is.
fn boiler(c: &mut Canvas, rect: Rect, step: &Heated) {
    let height = (step.volume * HEIGHT_PER_VOLUME) as f32;
    let liquid = (step.liquid * HEIGHT_PER_VOLUME) as f32;
    let maximum = (VOLUME_RANGE[1] * HEIGHT_PER_VOLUME) as f32;
    let warmth =
        (step.temperature - TEMPERATURE_RANGE[0]) / (TEMPERATURE_RANGE[1] - TEMPERATURE_RANGE[0]);
    let vapour = charts::thermal(warmth).mix_light(vapour_light(), 0.5);
    let cam = charts::cylinder_camera(rect, maximum);
    panel3(c, rect, cam, |s| {
        charts::cylinder(s, height, maximum);
        charts::load(s, height);
        if liquid > 1e-3 {
            charts::fill(s, 0.0, liquid, LIQUID.faded(0.8), 0.9);
        }
        charts::fill(s, liquid, height, vapour, 0.35);
    });
}

fn draw(c: &mut Canvas, sc: &Scene, t: f32) {
    gax_numga_examples::backdrop(c);
    let frame = ((t / FRAME_SECONDS) as usize).min(sc.steps.len() - 1);
    let panels = charts::row(c, &[0.6, 1.0, 1.0, 1.0]);
    boiler(c, panels[0].part(0.0, 0.1, 1.0, 0.85), &sc.steps[frame]);
    pressure_volume(c, panels[1], sc, frame);
    temperature_enthalpy(c, panels[2], sc, frame);
    pressure_temperature(c, panels[3], sc, frame);
    caption(
        c,
        "PHASE CHANGE: BOILING A VAN DER WAALS FLUID",
        "COEXISTENCE BY NEWTON'S METHOD: THE MISMATCH'S JET DERIVATIVE, A MAP ON STATES, SOLVED (L0G0)",
    );
}

fn main() {
    let steps = heating(BOILING, COLDEST, HOTTEST, FRAMES).len();
    run_with(
        Anim::new("phase_change", FRAME_SECONDS * steps as f32 + HOLD).size(960, 540),
        scene,
        draw,
    );
}

#[cfg(test)]
mod tests;
