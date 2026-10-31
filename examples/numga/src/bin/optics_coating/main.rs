//! numga's `optics/coating`: optical layers from their constitutive maps in spacetime algebra.
//! A medium is its constitutive map `χ`, from the field bivector `F` to the excitation
//! `G = χ(F)`, an intrinsic bivector (numga's antibivector), a sum of dyads: the planes each
//! axis of space sweeps through time read the electric field and carry the permittivities, the
//! patches of the planes of space read the magnetic field and share one inverse permeability.
//! In a layer normal to `z`, with a fixed tangential wavevector `p` (its part in time at unit
//! frequency, `−t`), the two Maxwell equations `k ∧ F = 0` and `k ∧ G = 0` split by whether
//! they hold a derivative along `z`:
//!
//! * Four do. Their amplitudes are the tangential electric field, the trivectors
//!   `t ∧ (t⁻¹ · (z ∧ F))` (blades `txz tyz`), and the tangential magnetic excitation, the
//!   vectors `t⁻¹ · (t ∧ (z ∧ G)*)` (blades `x y`). These are continuous across an interface.
//! * Two do not. They fix the longitudinal fields: solving them reconstructs the whole field
//!   from the four boundary amplitudes.
//!
//! What remains is the amplitudes' generator `G` along `z`, a 4 x 4 map whose eigenvalues are the
//! normal indices of the layer's waves. A layer carries amplitudes across its thickness `d` by
//! `exp(i k₀ d G)`, stacks compose, and in an isotropic exterior `(1 ± G / n) / 2` split the
//! amplitudes into the incoming and returning waves. Their interface conditions give the
//! reflection and transmission maps on the incident polarization, the tangential electric field
//! (the planes `xt yt`), both polarizations coupled. The power a wave carries through a face is
//! the energy current along the observer's time, read by the face.
//!
//! The scenes: a quarter-wave layer that matches glass to air, a stack of birefringent slices
//! turned slice by slice (a helix) that reflects the circular polarization turning with it and
//! passes the other, and an axion interface (`χ` plus a multiple of the identity) that turns the
//! reflected polarization. A leapfrog on a line of nodes, a gax field stepped by a sparse
//! difference map, sends a pulse through the same helix in time.
//!
//! The animation shows that pulse, seen from aside and a little above, the slab a translucent
//! box with its optic axes as rods. Its carrier is linear, half of each circular hand: the hand
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

use coating::{Direction, Polarization, in_plane};
use gax::pga2d::Point;
use gax::pga3d;
use gax_numga_examples::{
    Align, Anim, Axes, Camera, Canvas, Light, Marker, Rect, Scene3, backdrop, caption, colormap,
    palette, points::bounds, run_with, signal::haversine,
};
use scenes::{
    BAND_WAVELENGTH, CELLS, LINEAR, Results, SLAB_END, SLAB_START, helix, in_slab, positions,
    results,
};

/// One loop: the pulse's run along the line, its frames shown in turn.
const SECONDS: f32 = 9.0;
/// Nanometres per micron, the scene's unit.
const NM_PER_MICRON: f64 = 1000.0;
/// The field drawn this many microns across the line per unit (numga's 3000 nm).
const FIELD_LENGTH: f64 = 3.0;
/// The stretch of line framed: shares of its length cut from its start and its end.
const FRAMED: [f64; 2] = [0.08, 0.12];
/// The view from aside and a little above: its elevation and its turn off the broadside, in
/// radians (numga's 6° and 30°).
const ELEVATION: f32 = 0.105;
const AZIMUTH: f32 = 0.524;
/// The optic axes' half length and the slab's half width across the line, in field units.
const OPTIC_HALF_LENGTH: f64 = 0.6;
const SLAB_HALF_WIDTH: f64 = 1.0;
/// The least strength at which a spoke is drawn.
const SPOKE_FLOOR: f64 = 0.02;

/// The beam's frame turned into the scene's, whose `z` is up: a half turn about the line along
/// `x + z` carries the beam's line, along `z`, onto `x`, across the picture, and the field's `x`
/// onto `z`, up (numga's `upright`).
fn upright() -> gax::Unit<pga3d::Motor> {
    pga3d::Motor::rotation_about(
        pga3d::Point::direction(1.0, 0.0, 1.0),
        core::f64::consts::PI,
    )
}

/// The middle of the framed stretch of line, nm.
fn framed_middle() -> f64 {
    let positions = positions();
    let (first, last) = (positions[0], positions[CELLS - 1]);
    let span = last - first;
    ((first + FRAMED[0] * span) + (last - FRAMED[1] * span)) / 2.0
}

/// The scene's point of the beam's frame at `depth` (nm) along the line, moved across it by the
/// field units `across` (its `x` and `y`).
fn station(depth: f64, across: [f64; 2]) -> pga3d::Point {
    let [x, y] = across.map(|a| a * FIELD_LENGTH);
    let z = (depth - framed_middle()) / NM_PER_MICRON;
    upright() >> pga3d::Point::xyz(x, y, z)
}

/// A direction across the line read in field units: its `x` and `y`.
fn across(d: Direction) -> [f64; 2] {
    [d.e234(), d.e314()]
}

/// The hue of a tangential field's direction around the line, a light as bright as the field
/// is strong, and that strength.
fn direction_light(v: Polarization) -> (Light, f64) {
    let east = Polarization::new(1.0, 0.0);
    let hue = east.angle_to(v) / core::f64::consts::TAU;
    let strength = (4.0 * v.norm()).min(1.0);
    (
        colormap::hsv(hue as f32).faded(1.7 * strength as f32),
        strength,
    )
}

/// The slab: a translucent box about the line, as long as the slab's stretch of it.
fn draw_slab(sc: &mut Scene3) {
    let w = SLAB_HALF_WIDTH;
    let corner = |depth: f64, i: f64, j: f64| station(depth, [i * w, j * w]);
    let ends = [SLAB_START, SLAB_END];
    let light = palette::BLUE.faded(0.35);
    let square = [(-1.0, -1.0), (-1.0, 1.0), (1.0, 1.0), (1.0, -1.0)];
    // The four long faces and the two ends.
    for k in 0..4 {
        let ((a, b), (c, d)) = (square[k], square[(k + 1) % 4]);
        let [s, e] = ends;
        sc.quad(
            corner(s, a, b),
            corner(e, a, b),
            corner(e, c, d),
            corner(s, c, d),
            light,
            0.12,
        );
    }
    for depth in ends {
        let [a, b, c, d] = square.map(|(i, j)| corner(depth, i, j));
        sc.quad(a, b, c, d, light, 0.12);
    }
    let edges: Vec<[pga3d::Point; 2]> = (0..4)
        .flat_map(|k| {
            let ((a, b), (c, d)) = (square[k], square[(k + 1) % 4]);
            ends.map(|depth| [corner(depth, a, b), corner(depth, c, d)])
                .into_iter()
                .chain([[corner(ends[0], a, b), corner(ends[1], a, b)]])
        })
        .collect();
    sc.stroke(&edges, 1.0, palette::BLUE.faded(0.3));
}

/// The pulse seen from aside and a little above: the slab a translucent box with its optic axes
/// as grey rods, the field's tip along the line over faint spokes from the line, coloured by
/// its direction around the line. A field that turns traces loops; one that does not stays in
/// its plane.
fn draw_pulse(c: &mut Canvas, rect: Rect, s: &Results, at: f64) {
    let linear = &s.pulses[LINEAR];
    let count = linear.frames.len();
    let frame = &linear.frames[((at * count as f64) as usize).min(count - 1)];
    let positions = positions();
    let span = positions[CELLS - 1] - positions[0];
    let (lo, hi) = (
        positions[0] + FRAMED[0] * span,
        positions[CELLS - 1] - FRAMED[1] * span,
    );
    // Below the title and the note.
    let size = (rect.height() / 26.0).clamp(4.0, 10.0);
    let view = rect.inset(0.0, 3.2 * size + 6.0, 0.0, 0.0);
    // Looking across the line from `−y`, turned off the broadside, at the scale where the
    // framed line and the slab's box fill the view: measured at one pixel per micron first.
    let camera = |scale: f32| {
        Camera::parallel(
            view,
            scale,
            -core::f32::consts::FRAC_PI_2 + AZIMUTH,
            ELEVATION,
        )
    };
    let unit = camera(1.0);
    let reach = SLAB_HALF_WIDTH;
    let corners = [SLAB_START, SLAB_END].into_iter().flat_map(|depth| {
        [(-1.0, -1.0), (-1.0, 1.0), (1.0, 1.0), (1.0, -1.0)]
            .map(|(i, j)| station(depth, [i * reach, j * reach]))
    });
    let extent = [station(lo, [0.0; 2]), station(hi, [0.0; 2])]
        .into_iter()
        .chain(corners)
        .filter_map(|p| unit.px(p));
    let [low, high] = bounds(extent);
    let scale =
        (0.94 * view.width() / (high[0] - low[0])).min(0.9 * view.height() / (high[1] - low[1]));
    let cam = camera(scale);
    c.clip(rect);
    let mut sc = Scene3::new(cam.viewport(view));
    draw_slab(&mut sc);
    let rods: Vec<[pga3d::Point; 2]> = positions
        .iter()
        .filter(|p| in_slab(**p))
        .step_by(3)
        .map(|p| {
            let [x, y] = across(helix(p - SLAB_START, 1.0).0).map(|a| a * OPTIC_HALF_LENGTH);
            [station(*p, [-x, -y]), station(*p, [x, y])]
        })
        .collect();
    sc.stroke(&rods, 1.5, palette::INK.faded(0.35));
    sc.seg(
        station(lo, [0.0; 2]),
        station(hi, [0.0; 2]),
        1.5,
        palette::INK.faded(0.45),
    );
    let read = in_plane();
    let tips: Vec<(f64, pga3d::Point, Light, f64)> = positions
        .iter()
        .zip(frame.iter())
        .filter(|(p, _)| (lo..=hi).contains(*p))
        .map(|(p, f)| {
            let e = read.of(f);
            let (light, strength) = direction_light(e);
            (*p, station(*p, [e.e1(), e.e2()]), light, strength)
        })
        .collect();
    // Every other node's spoke, where the field shows.
    for (p, tip, light, strength) in tips.iter().step_by(2) {
        if *strength > SPOKE_FLOOR {
            sc.seg(station(*p, [0.0; 2]), *tip, 1.1, light.faded(0.5));
        }
    }
    for pair in tips.windows(2) {
        sc.seg(pair[0].1, pair[1].1, 2.25, pair[0].2);
    }
    sc.draw(c);
    c.unclip();
    let title_at = rect.lo + gax_numga_examples::Point2::direction(12.0, 1.2 * size + 6.0);
    c.text(
        "A LINEAR PULSE MEETS THE HELIX: ONE HAND RETURNS, THE OTHER PASSES",
        title_at,
        size,
        palette::INK,
        Align::Left,
    );
    let fraction = |part: f64| 100.0 * part / linear.before;
    let note = format!(
        "AFTER THE RUN: {:.0}% REFLECTED, {:.0}% TRANSMITTED",
        fraction(linear.reflected),
        fraction(linear.transmitted)
    );
    let note_at = title_at + gax_numga_examples::Point2::direction(0.0, 1.6 * size);
    c.text(
        &note,
        note_at,
        0.8 * size,
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

fn draw_coatings(c: &mut Canvas, rect: Rect, s: &Results, cursor: usize) {
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

fn draw_helix(c: &mut Canvas, rect: Rect, s: &Results, cursor: usize) {
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
        ("X TO Y, WITH IT", percent(&s.helix[0][0]), palette::ORANGE),
        ("Y TO X, AGAINST", percent(&s.helix[1][0]), palette::SKY),
    ];
    draw_curves(c, &ax, &s.wavelengths, &curves, cursor);
}

fn draw_axion(c: &mut Canvas, rect: Rect, s: &Results, cursor: usize) {
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

fn draw(c: &mut Canvas, s: &Results, t: f32) {
    backdrop(c);
    let screen = c.rect();
    let (w, h) = (screen.width(), screen.height());
    let at = f64::from((t / SECONDS).rem_euclid(1.0));
    // Below the caption (its subtitle's baseline three caption sizes down), the pulse above,
    // the spectra below.
    let top = (h / 30.0).clamp(10.0, 22.0) * 3.0 + 4.0;
    let mid = top + (h - top) * 0.5;
    draw_pulse(c, Rect::new(0.0, top, w, mid), s, at);
    // The cursors sweep the spectra back and forth.
    let sweep = haversine(core::f64::consts::TAU * at);
    let sample = |n: usize| (sweep * (n - 1) as f64).round() as usize;
    let lower = |x0: f32, x1: f32| Rect::new(w * x0, mid, w * x1, h);
    draw_coatings(c, lower(0.0, 1.0 / 3.0), s, sample(s.wavelengths.len()));
    draw_helix(
        c,
        lower(1.0 / 3.0, 2.0 / 3.0),
        s,
        sample(s.wavelengths.len()),
    );
    draw_axion(c, lower(2.0 / 3.0, 1.0), s, sample(s.jumps.len()));
    caption(
        c,
        "OPTICAL LAYERS FROM CONSTITUTIVE MAPS",
        "FOUR BOUNDARY AMPLITUDES, CARRIED LAYER BY LAYER BY THEIR GENERATOR (STA)",
    );
}

fn main() {
    run_with(Anim::new("coating", SECONDS).size(960, 540), results, draw);
}

#[cfg(test)]
mod tests {
    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws_with(super::draw, &super::results(), 0.5);
    }
}
