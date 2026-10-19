//! numga's `optics/coating`: optical layers from their constitutive maps in spacetime algebra.
//! A medium is its constitutive map `χ`, from the field bivector `F` to the excitation
//! `G = χ(F)` (numga's antibivectors are bivectors again in four dimensions). In a layer normal
//! to `z`, with a fixed tangential wave covector `p` (its temporal part the unit `t`), the two
//! Maxwell equations `k ∧ F = 0` and `k ∧ G = 0` split by whether they hold a derivative along
//! `z`:
//!
//! * Four do. Their amplitudes are the tangential electric field, the trivectors
//!   `t ∧ (t · (z ∧ F))` (blades `txz tyz`), and the tangential magnetic excitation, the
//!   vectors `t · (t ∧ (z ∧ G)*)` (blades `x y`). These are continuous across an interface.
//! * Two do not. They fix the longitudinal fields: solving them reconstructs the whole field
//!   from the four boundary amplitudes.
//!
//! What remains is the amplitudes' generator `G` along `z`, a 4 x 4 map whose eigenvalues are the
//! normal indices of the layer's waves. A layer carries amplitudes across its thickness `d` by
//! `exp(i k₀ d G)`, stacks compose, and in an isotropic exterior `(1 ± G / n) / 2` split the
//! amplitudes into the incoming and returning waves. Their interface conditions give the
//! reflection and transmission maps on the incident polarization, both polarizations coupled.
//!
//! The scenes: a quarter-wave layer that matches glass to air, a stack of birefringent slices
//! turned slice by slice (a helix) that reflects the circular polarization turning with it and
//! passes the other, and an axion interface (`χ` plus a multiple of the identity) that turns the
//! reflected polarization. A leapfrog on a line of nodes, a gax field stepped by a sparse
//! difference map, sends a pulse through the same helix in time.
//!
//! The animation shows that pulse. Its carrier is linear, half of each circular hand: the hand
//! that turns with the helix comes back, the other passes, and the field's tip traces the hands'
//! loops, coloured by its direction around the line. Below are the reflectance spectra of the
//! coatings and of the helix, and the axion interface's reflection against the jump, with
//! cursors sweeping them.
//!
//! `coating.rs` holds the layers' algebra, `line.rs` the time domain, and `scenes.rs` numga's
//! scenes.

mod coating;
mod line;
mod scenes;

use coating::{Polarization, in_plane, t};
use gax::pga2d::Point;
use gax_numga_examples::{
    Align, Anim, Axes, Canvas, Light, Marker, Rect, backdrop, caption, colormap, palette, run,
    signal::haversine,
};
use scenes::{
    BAND_WAVELENGTH, CELLS, LINEAR, SLAB_END, SLAB_START, helix, in_slab, positions, results,
};

/// One loop: the pulse's run along the line, its frames shown in turn.
const SECONDS: f32 = 9.0;
/// The tangential field's tip is drawn at a depth slanted by its `y`: microns of depth per unit
/// of field (numga's oblique view).
const SLANT: f64 = 1.0;
/// Nanometres per micron, the depth axis's unit.
const NM_PER_MICRON: f64 = 1000.0;
/// The pulse panel's reach above and below the line, in units of the pulse's peak field.
const FIELD_REACH: f64 = 1.25;

/// A tangential vector's tip in the oblique view: its depth along the line (microns), moved by
/// its `y` aslant, and its `x` up.
fn oblique(depth: f64, v: Polarization) -> Point {
    Point::xy(depth / NM_PER_MICRON + SLANT * v.e2(), v.e1())
}

/// The point of the line at `depth` (nm) in the oblique view.
fn on_line(depth: f64) -> Point {
    oblique(depth, Polarization::zero())
}

/// The hue of a tangential vector's direction around the line, a light as bright as the vector
/// is strong.
fn direction_light(v: Polarization) -> Light {
    let east = Polarization::new(1.0, 0.0);
    let hue = east.angle_to(v) / core::f64::consts::TAU;
    let strength = (4.0 * v.norm()).min(1.0);
    colormap::hsv(hue as f32).faded(1.6 * strength as f32)
}

/// The pulse in the oblique view: the slab a band with its optic axes as rods, the field's tip
/// along the line over faint spokes from the line, coloured by its direction. A field that turns
/// traces loops; one that does not stays in its plane.
fn draw_pulse(c: &mut Canvas, rect: Rect, at: f64) {
    let s = results();
    let linear = &s.pulses[LINEAR];
    let count = linear.frames.len();
    let frame = &linear.frames[((at * count as f64) as usize).min(count - 1)];
    let positions = positions();
    let span = positions[CELLS - 1] - positions[0];
    // numga's window: a little of each end cut away.
    let (lo, hi) = (
        positions[0] + 0.08 * span,
        positions[CELLS - 1] - 0.12 * span,
    );
    let ax = Axes::new(
        rect.inset(40.0, 22.0, 14.0, 30.0),
        [(lo / NM_PER_MICRON) as f32, (hi / NM_PER_MICRON) as f32],
        [-FIELD_REACH as f32, FIELD_REACH as f32],
    );
    ax.frame(
        c,
        "A LINEAR PULSE MEETS THE HELIX: ONE HAND RETURNS, THE OTHER PASSES",
        "DEPTH (MICRONS)",
        "",
    );
    let band = [
        (SLAB_START, -FIELD_REACH),
        (SLAB_END, -FIELD_REACH),
        (SLAB_END, FIELD_REACH),
        (SLAB_START, FIELD_REACH),
    ]
    .map(|(d, e)| Point::xy(d / NM_PER_MICRON, e));
    ax.fill(c, &band, palette::BLUE.faded(0.06), 0.4);
    let read = in_plane();
    let rods: Vec<[Point; 2]> = positions
        .iter()
        .filter(|p| in_slab(**p))
        .step_by(3)
        .map(|p| {
            let axis = read.of(helix(p - SLAB_START, 1.0).0) * 0.6;
            [oblique(*p, -axis), oblique(*p, axis)]
        })
        .collect();
    ax.stroke(c, &rods, 1.0, palette::INK.faded(0.15));
    ax.line(c, on_line(lo), on_line(hi), 1.0, palette::GRID);
    let tips: Vec<(Point, Light)> = positions
        .iter()
        .zip(frame.iter())
        .map(|(p, f)| {
            let e = read.of(f | t());
            (oblique(*p, e), direction_light(e))
        })
        .collect();
    for (p, (tip, light)) in positions.iter().zip(&tips).step_by(2) {
        ax.line(c, on_line(*p), *tip, 1.0, light.faded(0.25));
    }
    for segment in tips.windows(2) {
        ax.line(c, segment[0].0, segment[1].0, 1.6, segment[0].1);
    }
    let fraction = |part: f64| 100.0 * part / linear.before;
    let note = format!(
        "AFTER THE RUN: {:.0}% REFLECTED, {:.0}% TRANSMITTED",
        fraction(linear.reflected),
        fraction(linear.transmitted)
    );
    let size = (rect.height() / 30.0).clamp(4.0, 8.0);
    ax.text(
        c,
        ax.at(0.01, 0.92),
        &note,
        size,
        palette::INK.faded(0.7),
        Align::Left,
    );
}

/// Curves over one axis, a cursor at sample `cursor` with a dot on each, and their legend.
fn draw_curves(
    c: &mut Canvas,
    ax: &Axes,
    xs: &[f64],
    curves: &[(&str, Vec<f64>, Light)],
    cursor: usize,
) {
    for (_, ys, light) in curves {
        let pts: Vec<Point> = xs.iter().zip(ys).map(|(x, y)| Point::xy(*x, *y)).collect();
        ax.polyline(c, &pts, 1.6, *light);
        ax.scatter(c, &pts[cursor..=cursor], Marker::Dot, 7.0, *light);
    }
    let (x, [y0, y1]) = (xs[cursor], ax.y.map(f64::from));
    ax.line(
        c,
        Point::xy(x, y0),
        Point::xy(x, y1),
        1.0,
        palette::INK.faded(0.4),
    );
    let names: Vec<(&str, Light)> = curves.iter().map(|(n, _, l)| (*n, *l)).collect();
    ax.legend(c, &names);
}

/// The plot area of a lower panel.
fn plot_area(rect: Rect) -> Rect {
    rect.inset(40.0, 24.0, 12.0, 34.0)
}

/// Reflectances in percent.
fn percent(powers: &[(f64, f64)]) -> Vec<f64> {
    powers.iter().map(|(r, _)| 100.0 * r).collect()
}

fn draw_coatings(c: &mut Canvas, rect: Rect, cursor: usize) {
    let s = results();
    // Room above the curves for the legend.
    let ax = Axes::new(plot_area(rect), [450.0, 850.0], [0.0, 6.5]);
    ax.frame(c, "COATINGS ON GLASS", "WAVELENGTH (NM)", "R (%)");
    let names = ["BARE GLASS", "IDEAL LAYER", "MGF2 LAYER"];
    let curves: Vec<_> = names
        .iter()
        .zip(&s.coatings)
        .enumerate()
        .map(|(i, (n, powers))| (*n, percent(powers), palette::series(i)))
        .collect();
    draw_curves(c, &ax, &s.wavelengths, &curves, cursor);
}

fn draw_helix(c: &mut Canvas, rect: Rect, cursor: usize) {
    let s = results();
    // Room above the full reflection for the legend.
    let ceiling = 125.0;
    let ax = Axes::new(plot_area(rect), [450.0, 850.0], [0.0, ceiling as f32]);
    ax.frame(c, "THE HELIX: CIRCULAR LIGHT", "WAVELENGTH (NM)", "R (%)");
    // The pulse's carrier, at the band's centre.
    let carrier = [
        Point::xy(BAND_WAVELENGTH, 0.0),
        Point::xy(BAND_WAVELENGTH, ceiling),
    ];
    let faint = palette::INK.faded(0.5);
    ax.dashed(c, &carrier, 1.0, 4.0, faint);
    let size = (rect.height() / 30.0).clamp(4.0, 8.0);
    let label = Point::xy(BAND_WAVELENGTH + 8.0, 6.0);
    ax.text(c, label, "PULSE", size, faint, Align::Left);
    let curves = [
        ("X + IY, WITH IT", percent(&s.helix[0][0]), palette::ORANGE),
        ("X - IY, AGAINST", percent(&s.helix[1][0]), palette::SKY),
    ];
    draw_curves(c, &ax, &s.wavelengths, &curves, cursor);
}

fn draw_axion(c: &mut Canvas, rect: Rect, cursor: usize) {
    let s = results();
    // Room above the curves for the legend.
    let ax = Axes::new(plot_area(rect), [-3.0, 3.0], [0.0, 65.0]);
    ax.frame(c, "AN AXION INTERFACE", "AXION JUMP", "R (%)");
    let channel = |k: usize| s.axion.iter().map(|r| 100.0 * r[k]).collect();
    let curves = [
        ("X, AS INCIDENT", channel(0), palette::GREEN),
        ("Y, TURNED", channel(1), palette::PURPLE),
        ("TOTAL", channel(2), palette::YELLOW),
    ];
    draw_curves(c, &ax, &s.jumps, &curves, cursor);
}

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let screen = c.rect();
    let (w, h) = (screen.width(), screen.height());
    let at = f64::from((t / SECONDS).rem_euclid(1.0));
    // Below the caption (its subtitle's baseline three caption sizes down), the pulse above,
    // the spectra below.
    let top = (h / 30.0).clamp(10.0, 22.0) * 3.0 + 4.0;
    let mid = top + (h - top) * 0.5;
    draw_pulse(c, Rect::new(0.0, top, w, mid), at);
    // The cursors sweep the spectra back and forth.
    let sweep = haversine(core::f64::consts::TAU * at);
    let sample = |n: usize| (sweep * (n - 1) as f64).round() as usize;
    let lower = |x0: f32, x1: f32| Rect::new(w * x0, mid, w * x1, h);
    let s = results();
    draw_coatings(c, lower(0.0, 1.0 / 3.0), sample(s.wavelengths.len()));
    draw_helix(c, lower(1.0 / 3.0, 2.0 / 3.0), sample(s.wavelengths.len()));
    draw_axion(c, lower(2.0 / 3.0, 1.0), sample(s.jumps.len()));
    caption(
        c,
        "OPTICAL LAYERS FROM CONSTITUTIVE MAPS",
        "FOUR BOUNDARY AMPLITUDES, CARRIED LAYER BY LAYER BY THEIR GENERATOR (STA)",
    );
}

fn main() {
    run(Anim::new("coating", SECONDS).size(960, 540), draw);
}

#[cfg(test)]
mod tests {
    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws(super::draw, 0.5);
    }
}
