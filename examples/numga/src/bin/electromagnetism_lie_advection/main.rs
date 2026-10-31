//! numga's `electromagnetism/lie_advection`: Lie advection carries magnetic flux across a fixed
//! mesh, in the spacetime of the plane, `x+y+t-`.
//!
//! An edge's reading of a 1-form, its circulation, changes at the rate the field reads on the
//! world sheet the edge sweeps through spacetime: its motion, `time + velocity`, joined with it.
//! That is Cartan's formula, `L_v A = i_v dA + d(i_v A)`: the flux of the 2-form through the
//! sheet, plus the difference of the form's readings on the ends' motions, which cancel around a
//! closed boundary. The magnetic field is the plane `xy` of space and reads only a sheet's spatial
//! part; an electric field is a plane through time. Ideal Ohm's law is the field vanishing on the
//! sheets of edges that move with the conductor, so a fixed edge's reading changes by the
//! magnetic flux through the patch between its own world sheet and the comoving one. A prescribed
//! flow gives one evolution map on the edge readings, and resistivity dissipates the field
//! through the curl's adjugate.
//!
//! Left, two lobes of flux wound up by an elliptical vortex on a disk, ideal and resistive: the
//! flux density on each face, over the flow's streamlines. Right, a triangle whose corners move,
//! the ribbons its sides sweep coloured by the flux through them, with each side's endpoint
//! difference, swept flux and their sum; and the magnetic energy, kept by the ideal flow to
//! `10⁻⁴` and dissipated by the resistive one, while the total flux stays fixed in both.

mod advection;
mod scenes;
#[cfg(test)]
mod tests;

#[path = "../../shared/simplicial.rs"]
mod simplicial;
#[path = "../../shared/sta2d.rs"]
mod sta2d;
#[path = "../../shared/surfaces_plane.rs"]
mod surfaces_plane;

use advection::*;
use gax_numga_examples::signal::frame;
use gax_numga_examples::{
    Anim, Axes, Canvas, Light, Marker, Point2, Rect, backdrop, caption, label_under, palette,
    run_with,
};
use scenes::*;
use simplicial::faces;
use sta2d::{Point, coordinates};
use surfaces_plane::{Flat, arrow};

/// One loop: numga's 90 frames of 40 ms.
const SECONDS: f32 = 3.6;
/// The flow's streamlines drawn.
const STREAMLINES: usize = 9;

struct Shown {
    flat: Flat,
    stream: Vec<f64>,
    /// `densities[case][frame][face]`.
    densities: [Vec<Vec<f64>>; 2],
    density_scale: f64,
    energies: [Vec<f64>; 2],
    corners: [Point; 3],
    forms: [Plane; 3],
    velocities: [Point; 3],
    cartan: Cartan,
}

fn shown() -> Shown {
    let disk = Disk::new(DISK_DIVISIONS);
    let o = &disk.operators;
    let runs = RESISTIVITIES.map(|r| advect(&disk.start, &disk.rate(r), DT, STEPS));
    let densities = [0, 1].map(|case| {
        runs[case]
            .iter()
            .step_by(STEPS_PER_FRAME)
            .map(|a| o.flux_density.of(a).iter().map(|x| x.s()).collect())
            .collect()
    });
    let start: Vec<f64> = o
        .flux_density
        .of(&disk.start)
        .iter()
        .map(|x| x.s())
        .collect();
    let mut sorted: Vec<f64> = start.iter().map(|x| x.abs()).collect();
    sorted.sort_by(f64::total_cmp);
    let density_scale = sorted[(0.98 * (sorted.len() - 1) as f64) as usize];
    let energies = [0, 1].map(|case| runs[case].iter().map(|a| o.energy(a)).collect());
    let points: Vec<[f64; 2]> = o.vertices.iter().map(coordinates).collect();
    let (corners, forms, velocities) = (single_triangle(), corner_forms(), corner_velocities());
    Shown {
        flat: Flat::new(&points, &faces(&o.mesh)),
        stream: disk.stream.iter().map(|s| s.s()).collect(),
        densities,
        density_scale,
        energies,
        corners,
        forms,
        velocities,
        cartan: cartan_decomposition(corners, forms, velocities),
    }
}

/// A point as drawn.
fn at(p: Point) -> Point2 {
    let [x, y] = coordinates(p);
    Point2::xy(x as f32, y as f32)
}

/// A signed value's light: orange where positive, sky where negative.
fn diverging(v: f64) -> Light {
    let t = v.clamp(-1.0, 1.0) as f32;
    let accent = if t > 0.0 {
        palette::ORANGE
    } else {
        palette::SKY
    };
    palette::GRID.faded(0.5) + accent.faded(0.5 * t.abs().powf(0.8))
}

/// The flux density of a case at a frame, over the flow's streamlines.
fn draw_disk(c: &mut Canvas, rect: Rect, s: &Shown, densities: &[f64], label: &str) {
    let ax = Axes::fitting(
        rect.part(0.0, 0.0, 1.0, 0.92),
        s.flat.points.iter().copied(),
        1.04,
    );
    s.flat.shade(c, &ax, 2, |f, _| {
        Some(diverging(densities[f] / s.density_scale))
    });
    let low = s.stream.iter().copied().fold(f64::INFINITY, f64::min);
    let high = s.stream.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    for k in 1..=STREAMLINES {
        let level = low + (high - low) * k as f64 / (STREAMLINES + 1) as f64;
        ax.stroke(
            c,
            &s.flat.level(&s.stream, level),
            0.7,
            palette::INK.faded(0.25),
        );
    }
    ax.stroke(c, &s.flat.rim(), 1.3, palette::INK.faded(0.7));
    label_under(c, rect, label, 20.0, 0.9);
}

/// The moving triangle: the ribbons its sides sweep in a step, coloured by the flux through them,
/// the form across it, the corners' velocities, and the moved triangle.
fn draw_triangle(c: &mut Canvas, rect: Rect, s: &Shown) {
    let displaced = [0, 1, 2].map(|k| s.corners[k] + s.velocities[k] * RIBBON_STEP);
    let all: Vec<Point2> = s.corners.iter().chain(&displaced).map(|&p| at(p)).collect();
    let ax = Axes::fitting(rect.part(0.0, 0.0, 1.0, 0.84), all, 1.12);
    let largest = s
        .cartan
        .swept_fluxes
        .iter()
        .map(|f| f.s().abs())
        .fold(1e-8, f64::max);
    for k in 0..3 {
        let (a, b) = ((k + 1) % 3, (k + 2) % 3);
        let ribbon = [s.corners[a], s.corners[b], displaced[b], displaced[a]].map(at);
        let light = diverging(s.cartan.swept_fluxes[k].s() / largest);
        ax.fill(c, &ribbon, light, 0.6);
        ax.dashed(
            c,
            &[ribbon[0], ribbon[1], ribbon[2], ribbon[3], ribbon[0]],
            0.8,
            3.0,
            palette::INK.faded(0.4),
        );
    }
    let corners = s.corners.map(at);
    ax.fill(c, &corners, palette::BLUE, 0.3);
    ax.polyline(
        c,
        &[corners[0], corners[1], corners[2], corners[0]],
        1.2,
        palette::INK.faded(0.7),
    );
    let moved = displaced.map(at);
    ax.dashed(
        c,
        &[moved[0], moved[1], moved[2], moved[0]],
        1.0,
        4.0,
        palette::INK,
    );
    // The form's dual across the triangle, interpolated by the corners' hats.
    let gradients = hat_gradients(sides(s.corners));
    let longest = s.forms.iter().map(|f| f.norm()).fold(0.0, f64::max);
    for (i, j) in [(1, 1), (2, 1), (1, 2), (3, 1), (2, 2), (1, 3)] {
        let p =
            (s.corners[0] * (5 - i - j) as f64 + s.corners[1] * i as f64 + s.corners[2] * j as f64)
                * 0.2;
        let form = (0..3)
            .map(|k| s.forms[k] * (Scalar::new(1.0) + (gradients[k] & (p - s.corners[k]))))
            .fold(Plane::zero(), |a, b| a + b);
        let d = [form.ex() * 0.12 / longest, form.ey() * 0.12 / longest];
        let tail = at(p) - Point2::direction(0.5 * d[0] as f32, 0.5 * d[1] as f32);
        arrow(c, &ax, tail, d, 1.2, 5.0, palette::SKY);
    }
    for k in 0..3 {
        ax.arrow(c, corners[k], moved[k], 1.8, 8.0, palette::ORANGE);
    }
    ax.scatter(c, &corners, Marker::Dot, 6.0, palette::INK);
    ax.scatter(c, &moved, Marker::Dot, 5.0, palette::INK.faded(0.6));
    label_under(c, rect, "SWEPT RIBBONS OF A MOVING TRIANGLE", 14.0, 0.95);
}

/// Each side's endpoint difference, swept flux and their sum, and the totals.
fn draw_bars(c: &mut Canvas, rect: Rect, s: &Shown) {
    let e = s.cartan.endpoint_differences.map(|x| x.s());
    let f = s.cartan.swept_fluxes.map(|x| x.s());
    let groups: Vec<[f64; 3]> = (0..4)
        .map(|k| {
            if k < 3 {
                [e[k], f[k], e[k] + f[k]]
            } else {
                let (se, sf) = (e.iter().sum::<f64>(), f.iter().sum::<f64>());
                [se, sf, se + sf]
            }
        })
        .collect();
    let span = groups.iter().flatten().map(|v| v.abs()).fold(0.1, f64::max) as f32 * 1.18;
    let ax = Axes::new(rect, [-0.55, 3.55], [-span, span]);
    ax.frame(
        c,
        "CARTAN: ENDS + SWEPT = RATE",
        "SIDE FACING 0, 1, 2; TOTAL",
        "",
    );
    ax.line(
        c,
        Point2::xy(-0.55, 0.0),
        Point2::xy(3.55, 0.0),
        1.0,
        palette::GRID.faded(3.0),
    );
    let lights = [palette::SKY, palette::ORANGE, palette::INK];
    for (k, group) in groups.iter().enumerate() {
        for (j, &v) in group.iter().enumerate() {
            let x = k as f32 + (j as f32 - 1.0) * 0.26;
            let bar = [
                Point2::xy(x - 0.11, 0.0),
                Point2::xy(x + 0.11, 0.0),
                Point2::xy(x + 0.11, v as f32),
                Point2::xy(x - 0.11, v as f32),
            ];
            ax.fill(c, &bar, lights[j], 0.5);
        }
    }
}

/// The magnetic energy of both cases over time, relative to the start, and the total flux.
fn draw_energy(c: &mut Canvas, rect: Rect, s: &Shown, now: usize) {
    let end = (STEPS as f64 * DT) as f32;
    let ax = Axes::new(rect, [0.0, end], [0.0, 1.1]);
    ax.frame(c, "MAGNETIC ENERGY OVER ITS START", "TIME", "");
    for (case, l) in [(0, palette::INK), (1, palette::ORANGE)] {
        let e0 = s.energies[case][0];
        let curve: Vec<Point2> = s.energies[case]
            .iter()
            .enumerate()
            .map(|(k, e)| Point2::xy((k as f64 * DT) as f32, (e / e0) as f32))
            .collect();
        ax.polyline(c, &curve, 1.6, l);
    }
    let t = (now as f64 * DT) as f32;
    ax.line(
        c,
        Point2::xy(t, 0.0),
        Point2::xy(t, 1.1),
        1.0,
        palette::YELLOW.faded(0.6),
    );
    ax.legend(
        c,
        &[("IDEAL", palette::INK), ("RESISTIVE", palette::ORANGE)],
    );
}

fn draw(c: &mut Canvas, s: &Shown, t: f32) {
    backdrop(c);
    let screen = c.rect();
    let (w, h) = (screen.width(), screen.height());
    let below = Rect::new(0.0, h * 0.13, w, h * 0.97);
    let k = frame((t / SECONDS).rem_euclid(1.0), s.densities[0].len());
    let disks = below.part(0.0, 0.0, 0.64, 1.0);
    draw_disk(
        c,
        disks.column(0, 2).inset(6.0, 4.0, 6.0, 4.0),
        s,
        &s.densities[0][k],
        "IDEAL: FLUX WOUND UP, KEPT",
    );
    draw_disk(
        c,
        disks.column(1, 2).inset(6.0, 4.0, 6.0, 4.0),
        s,
        &s.densities[1][k],
        "RESISTIVE: WOUND AND DIFFUSED",
    );
    let right = below.part(0.64, 0.0, 1.0, 1.0);
    draw_triangle(
        c,
        right.part(0.0, 0.0, 1.0, 0.4).inset(4.0, 2.0, 4.0, 4.0),
        s,
    );
    draw_bars(
        c,
        right.part(0.0, 0.4, 1.0, 0.7).inset(34.0, 22.0, 8.0, 30.0),
        s,
    );
    draw_energy(
        c,
        right.part(0.0, 0.7, 1.0, 1.0).inset(34.0, 22.0, 8.0, 30.0),
        s,
        k * STEPS_PER_FRAME,
    );
    caption(
        c,
        "LIE ADVECTION: MAGNETIC FLUX CARRIED ACROSS A FIXED MESH",
        "AN EDGE'S READING CHANGES BY THE FIELD ON THE WORLD SHEET IT SWEEPS: CARTAN'S FORMULA, IN THE SPACETIME OF THE PLANE (X+Y+T-)",
    );
}

fn main() {
    run_with(
        Anim::new("lie advection", SECONDS).size(960, 540),
        shown,
        draw,
    );
}
