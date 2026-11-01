//! numga's `electromagnetism/piezoelectric`: piezoelectric continuum elasticity in a laminated
//! strip, a PZT-5A bimorph used as an actuator and as a sensor.
//!
//! Displacements are intrinsic directions, the complements of the planes of space that read
//! them; electric fields are planes through time, in numga's spacetime of the plane `x+y+t-`
//! (`shared/sta2d.rs`). A material's elastic, piezoelectric and dielectric maps, built from its
//! two material planes, give one electric enthalpy, and pulling their work back through the
//! finite-element readings (each shape gradient bound into the maps, the corners' incidences run
//! back and forth) gives the nodes' equilibrium and their charge balance. The coupling block read
//! backwards is the converse effect: the same map that bends the strip under a voltage reads a
//! charge from its bending. Metal electrodes hold one potential; a floating electrode keeps its
//! total charge while its voltage responds.
//!
//! Top, the actuator: 100 V across the two layers, poled against each other (the arrows), bends
//! the clamped strip, one layer stretching (orange) as the other shortens (sky); the electrodes
//! show their voltage. Below, the sensor: a tip load of 60 mN bends it, and the floating top
//! electrode reads the voltage the bending makes. Both swing through a slow sinusoidal sweep of
//! equilibria, the displacements drawn a hundred times their size.

mod piezo;
mod scenes;
mod sta2d;
#[cfg(test)]
mod tests;

use gax::field::Field;
use gax_numga_examples::shared::simplicial;
use gax_numga_examples::signal::wave;
use gax_numga_examples::{
    Anim, Axes, Canvas, Light, Point2, Rect, backdrop, caption, label_under, palette, run_with,
};
use piezo::*;
use scenes::*;

/// One loop: numga's 90 frames of 40 ms.
const SECONDS: f32 = 3.6;
/// The poling arrows along each layer.
const POLES: usize = 7;

/// A state of the strip at full amplitude.
struct State {
    displacement: Field<Point>,
    potential: Field<Scalar>,
    strain: Vec<f64>,
}

struct Shown {
    strip: Strip,
    actuator: State,
    sensor: State,
}

fn state(
    device: &Device,
    (displacement, potential, _): (Field<Point>, Field<Scalar>, Field<Amount>),
) -> State {
    let strain = axial_strain(&device.strip.mesh, &displacement, &device.gradients);
    State {
        displacement,
        potential,
        strain,
    }
}

fn shown() -> Shown {
    let device = Device::new(LENGTH, THICKNESS, N_LENGTH, N_THICKNESS, WIDTH);
    let actuator = state(&device, device.solve(APPLIED_VOLTAGE, 0.0, false));
    let sensor = state(&device, device.solve(0.0, APPLIED_FORCE, true));
    Shown {
        strip: device.strip,
        actuator,
        sensor,
    }
}

/// A point as drawn.
fn at(p: Point) -> Point2 {
    let [x, y] = coordinates(p);
    Point2::xy(x as f32, y as f32)
}

/// A signed value's light: orange where positive (extension, a higher voltage), sky where
/// negative.
fn diverging(v: f64) -> Light {
    let t = v.clamp(-1.0, 1.0) as f32;
    let accent = if t > 0.0 {
        palette::ORANGE
    } else {
        palette::SKY
    };
    palette::GRID.faded(0.6) + accent.faded(0.06 + 0.5 * t.abs())
}

/// The strip in a state at an amplitude: the clamp, the rest outline, the deformed body by its
/// axial strain, the interface and the tip, the electrodes by their voltage and the poling.
fn draw_strip(
    c: &mut Canvas,
    rect: Rect,
    s: &Strip,
    state: &State,
    amplitude: f64,
    voltage_scale: f64,
    label: &str,
) {
    let half = (3.3 * THICKNESS).max(0.11 * LENGTH) as f32;
    let length = LENGTH as f32;
    let ax = Axes::new(
        rect.part(0.0, 0.0, 1.0, 0.86),
        [-0.045 * length, 1.025 * length],
        [-half, half],
    );
    let t = THICKNESS as f32;
    let clamp = [
        Point2::xy(-0.025 * length, -t),
        Point2::xy(0.0, -t),
        Point2::xy(0.0, t),
        Point2::xy(-0.025 * length, t),
    ];
    ax.fill(c, &clamp, palette::GRID.faded(3.0), 0.6);
    let rest: Vec<Point2> = s
        .bottom
        .iter()
        .chain(&s.tip)
        .chain(s.top.iter().rev())
        .map(|&v| at(s.vertices.site(v)))
        .collect();
    ax.dashed(c, &rest, 0.8, 4.0, palette::INK.faded(0.3));
    let scale = DISPLACEMENT_SCALE * amplitude;
    let deformed: Vec<Point2> = (0..s.vertices.len())
        .map(|v| at(s.vertices.site(v) + state.displacement.site(v) * scale))
        .collect();
    for (f, strain) in state.strain.iter().enumerate() {
        let corners = s
            .mesh
            .simplex(2, f)
            .iter()
            .map(|&v| deformed[v])
            .collect::<Vec<_>>();
        ax.fill(
            c,
            &corners,
            diverging(strain * amplitude / STRAIN_SCALE),
            1.0,
        );
    }
    let line = |vs: &[usize]| -> Vec<Point2> { vs.iter().map(|&v| deformed[v]).collect() };
    ax.polyline(c, &line(&s.interface), 0.9, palette::INK.faded(0.5));
    ax.polyline(c, &line(&s.tip), 1.0, palette::INK.faded(0.5));
    // The electrodes read their voltage difference, whatever the potential's zero.
    let mean = |vs: &[usize]| {
        vs.iter().map(|&v| state.potential.site(v).s()).sum::<f64>() / vs.len() as f64
    };
    let (top, bottom) = (mean(&s.top) * amplitude, mean(&s.bottom) * amplitude);
    let middle = (top + bottom) / 2.0;
    for (vs, v) in [(&s.top, top), (&s.bottom, bottom)] {
        ax.polyline(
            c,
            &line(vs),
            2.2,
            diverging((v - middle) / (voltage_scale / 2.0)),
        );
    }
    // The poling: up in the upper layer, down in the lower, across the strip.
    for k in 1..=POLES {
        let i = k * (s.top.len() - 1) / (POLES + 1);
        let (upper, lower) = (deformed[s.top[i]], deformed[s.bottom[i]]);
        let across = upper - lower;
        let reach = across.gp(0.32 * t / across.ideal_norm().max(1e-6));
        for (centre, d) in [
            ((upper.gp(0.75) + lower.gp(0.25)).unitized(), reach),
            ((upper.gp(0.25) + lower.gp(0.75)).unitized(), -reach),
        ] {
            ax.arrow(
                c,
                centre - d.gp(0.5),
                centre + d.gp(0.5),
                1.0,
                4.0,
                palette::INK.faded(0.6),
            );
        }
    }
    label_under(c, rect, label, 12.0, 0.9);
}

fn draw(c: &mut Canvas, s: &Shown, t: f32) {
    backdrop(c);
    let screen = c.rect();
    let (w, h) = (screen.width(), screen.height());
    let amplitude = f64::from(wave(core::f32::consts::TAU * (t / SECONDS).rem_euclid(1.0)));
    draw_strip(
        c,
        Rect::new(10.0, h * 0.14, w - 10.0, h * 0.56),
        &s.strip,
        &s.actuator,
        amplitude,
        APPLIED_VOLTAGE,
        "ACTUATOR: 100 V ACROSS THE ELECTRODES BENDS THE CLAMPED BIMORPH",
    );
    draw_strip(
        c,
        Rect::new(10.0, h * 0.56, w - 10.0, h * 0.98),
        &s.strip,
        &s.sensor,
        amplitude,
        SENSOR_VOLTAGE,
        "SENSOR: A 60 MN TIP LOAD BENDS IT, AND THE FLOATING TOP ELECTRODE READS A VOLTAGE",
    );
    caption(
        c,
        "PIEZOELECTRIC BIMORPH: ONE ENTHALPY FOR STRAIN, FIELD AND CHARGE",
        "DISPLACEMENTS ARE DIRECTIONS, FIELDS PLANES THROUGH TIME; THE COUPLING READ BACKWARDS IS THE CONVERSE EFFECT (X+Y+T-, PZT-5A, X100)",
    );
}

fn main() {
    run_with(
        Anim::new("piezoelectric", SECONDS).size(960, 540),
        shown,
        draw,
    );
}
