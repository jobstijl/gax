//! numga's `relativity/moving_mirror`: a scalar pulse through a triangulated worldsheet with a
//! moving boundary. An advancing mirror compresses the reflected pulse and supplies its extra
//! energy, by the Doppler factor of relativity.
//!
//! The spacetime plane `x+t-` is numga's own algebra (declared here with `gax::algebra!`). A
//! triangle's signed patch, the join of two sides, gives the interpolation by corner hats and
//! edge forms with no metric: a side over twice the patch is the gradient of the hat of the
//! corner it faces. The signature enters once, through the dual that turns a gradient into the
//! current it drives: a null gradient pairs to nothing with its dual, and pulled back through
//! the gradient's adjugate, the dual gives the wave operator, whose characteristics are the null
//! rays. The metric alone does not choose a direction of time; initial data and boundary values
//! do, so the solution marches up the strip one row at a time, as a single sparse direct solve.
//!
//! Left, the cavity's worldsheet coloured by the wave rate, its slice of simultaneity sweeping
//! up while the right mirror swings in; middle, the cavity at that moment, the pulse compressed
//! and amplified as it comes back off the advancing mirror, and the energy between each two rows,
//! kept before and after the reflection and raised across it. Right, a spacetime triangle's hat
//! with its light cone, its gradient's Euclidean normal beside its Lorentzian dual; an edge form
//! circulating about its corner; and the causal wake of a source, inside its forward cone.

mod scenes;
mod spacetime;
#[cfg(test)]
mod tests;

use gax_numga_examples::shared::simplicial;
use gax_numga_examples::shared::surfaces_plane;
use gax_numga_examples::signal::frame;
use gax_numga_examples::{
    Anim, Axes, Canvas, Light, Marker, Point2, Rect, backdrop, caption, label_under, palette,
    run_with,
};
use scenes::*;
use simplicial::faces;
use spacetime::*;
use surfaces_plane::{Flat, arrow};

/// One loop: numga's 66 frames of 40 ms.
const SECONDS: f32 = 2.64;
const FRAMES: usize = 66;
/// The hat's levels.
const LEVELS: usize = 7;

struct Shown {
    triangle: [Point; 3],
    gradients: [Plane; 3],
    samples: Vec<Point>,
    edge_field: Vec<Point>,
    wake: Wake,
    wake_flat: Flat,
    wake_limit: f64,
    cavity: Cavity,
    cavity_flat: Flat,
    rate_limit: f64,
    /// The wave rate at each row and column, over the peak of row 20.
    profiles: Vec<Vec<f64>>,
}

/// The value of `|v|` below which a share `p` of the values lie.
fn percentile(values: impl Iterator<Item = f64>, p: f64) -> f64 {
    let mut v: Vec<f64> = values.map(f64::abs).collect();
    v.sort_by(f64::total_cmp);
    v[(p * (v.len() - 1) as f64).round() as usize].max(1e-12)
}

/// A strip's mesh as drawn.
fn flat(mesh: &gax::mesh::Topology<3>, vertices: &gax::field::Field<Point>) -> Flat {
    let points: Vec<[f64; 2]> = vertices.iter().map(coordinates).collect();
    Flat::new(&points, &faces(mesh))
}

fn shown() -> Shown {
    let (triangle, samples) = single_triangle(TRIANGLE_SUBDIVISIONS);
    let (_, gradients, _) = barycentric_hats(triangle, &samples);
    let edge_field = edge_whitney_forms(triangle, &samples)
        .iter()
        .map(|f| dual().of(f[0]))
        .collect();
    let wake = Wake::new();
    let wake_flat = flat(&wake.mesh, &wake.vertices);
    let wake_limit = percentile(wake.wake.iter().map(|w| w.s()), 0.992);
    let cavity = Cavity::new();
    let cavity_flat = flat(&cavity.mesh, &cavity.vertices);
    let rate_limit = percentile(cavity.wave.rate.iter().map(|w| w.s()), 0.994);
    let rate: Vec<f64> = cavity.wave.rate.iter().map(|r| r.s()).collect();
    let rows: Vec<&[f64]> = rate.chunks(MIRROR_COLUMNS).collect();
    let peak = rows[20].iter().copied().map(f64::abs).fold(0.0, f64::max);
    let profiles = rows
        .iter()
        .map(|row| row.iter().map(|r| r / peak).collect())
        .collect();
    Shown {
        triangle,
        gradients,
        samples,
        edge_field,
        wake,
        wake_flat,
        wake_limit,
        cavity,
        cavity_flat,
        rate_limit,
        profiles,
    }
}

/// An event as drawn, `x` across and `t` up.
fn at(p: Point) -> Point2 {
    let [x, t] = coordinates(p);
    Point2::xy(x as f32, t as f32)
}

/// A wave's light: orange where it rises, sky where it falls.
fn wave_light(v: f64) -> Light {
    let t = v.clamp(-1.0, 1.0) as f32;
    let accent = if t > 0.0 {
        palette::ORANGE
    } else {
        palette::SKY
    };
    palette::GRID.faded(0.5) + accent.faded(0.5 * t.abs())
}

/// The two null rays through an event, `before` into its past and `after` into its future.
fn null_rays(event: Point, before: f64, after: f64) -> [[Point2; 2]; 2] {
    [time() - space(), time() + space()].map(|n| [at(event - n * before), at(event + n * after)])
}

fn cone(c: &mut Canvas, ax: &Axes, rays: [[Point2; 2]; 2]) {
    for ray in rays {
        ax.dashed(c, &ray, 1.2, 5.0, palette::YELLOW.faded(0.7));
    }
}

/// The row of the cavity at frame time `t`.
fn row_at(t: f32) -> usize {
    let k = frame((t / SECONDS).rem_euclid(1.0), FRAMES);
    4 + (k * (MIRROR_ROWS - 9)) / (FRAMES - 1)
}

/// The worldsheet: the wave rate over the strip, the walls, the rows before and after, and the
/// slice of simultaneity.
fn draw_worldsheet(c: &mut Canvas, rect: Rect, s: &Shown, row: usize) {
    let ax = Axes::fitting(rect, s.cavity_flat.points.iter().copied(), 1.04);
    let rate: Vec<f64> = s
        .cavity
        .wave
        .rate
        .iter()
        .map(|r| r.s() / s.rate_limit)
        .collect();
    s.cavity_flat.shade_field(c, &ax, &rate, wave_light);
    let place = |r: usize, col: usize| at(s.cavity.vertices.site(r * MIRROR_COLUMNS + col));
    let wall = |col: usize| -> Vec<Point2> { (0..MIRROR_ROWS).map(|r| place(r, col)).collect() };
    ax.polyline(c, &wall(0), 1.4, palette::INK.faded(0.5));
    ax.polyline(c, &wall(MIRROR_COLUMNS - 1), 2.2, palette::INK);
    for (r, l) in [(BEFORE_ROW, palette::SKY), (AFTER_ROW, palette::ORANGE)] {
        let line = [place(r, 0), place(r, MIRROR_COLUMNS - 1)];
        ax.dashed(c, &line, 1.0, 4.0, l.faded(0.7));
    }
    ax.line(
        c,
        place(row, 0),
        place(row, MIRROR_COLUMNS - 1),
        2.0,
        palette::YELLOW,
    );
    ax.scatter(
        c,
        &[place(row, MIRROR_COLUMNS - 1)],
        Marker::Dot,
        8.0,
        palette::YELLOW,
    );
    label_under(c, rect, "THE WORLDSHEET: X ACROSS, T UP", 22.0, 0.9);
}

/// The cavity at one row: the wave rate across it, between the wall and the mirror.
fn draw_cavity(c: &mut Canvas, rect: Rect, s: &Shown, row: usize) {
    let ax = Axes::new(rect, [0.0, 1.02], [-2.6, 2.6]);
    ax.frame(c, "THE CAVITY NOW", "X", "WAVE RATE");
    let wall = coordinates(
        s.cavity
            .vertices
            .site(row * MIRROR_COLUMNS + MIRROR_COLUMNS - 1),
    )[0] as f32;
    ax.fill(
        c,
        &[
            Point2::xy(wall, -2.6),
            Point2::xy(1.02, -2.6),
            Point2::xy(1.02, 2.6),
            Point2::xy(wall, 2.6),
        ],
        palette::GRID,
        0.5,
    );
    ax.line(
        c,
        Point2::xy(0.0, 0.0),
        Point2::xy(1.02, 0.0),
        1.0,
        palette::GRID.faded(2.0),
    );
    let profile: Vec<Point2> = (0..MIRROR_COLUMNS)
        .map(|col| {
            let [x, _] = coordinates(s.cavity.vertices.site(row * MIRROR_COLUMNS + col));
            Point2::xy(x as f32, s.profiles[row][col] as f32)
        })
        .collect();
    let mut under = profile.clone();
    under.push(Point2::xy(wall, 0.0));
    under.push(Point2::xy(0.0, 0.0));
    ax.fill(c, &under, palette::SKY, 0.25);
    ax.polyline(c, &profile, 1.8, palette::SKY);
    ax.line(
        c,
        Point2::xy(0.0, -2.6),
        Point2::xy(0.0, 2.6),
        2.0,
        palette::INK.faded(0.6),
    );
    ax.line(
        c,
        Point2::xy(wall, -2.6),
        Point2::xy(wall, 2.6),
        2.6,
        palette::INK,
    );
}

/// The energy between each two rows over the energy before the reflection.
fn draw_energy(c: &mut Canvas, rect: Rect, s: &Shown, row: usize) {
    let e = &s.cavity.wave.slab_energies;
    let reference = e[BEFORE_ROW];
    let top = (e[AFTER_ROW] / reference * 1.15) as f32;
    let dt = MIRROR_DURATION / (MIRROR_ROWS - 1) as f64;
    let ax = Axes::new(rect, [0.0, MIRROR_DURATION as f32], [0.0, top]);
    ax.frame(c, "ENERGY IN EACH SLAB", "T", "OVER ITS START");
    ax.dashed(
        c,
        &[
            Point2::xy(0.0, 1.0),
            Point2::xy(MIRROR_DURATION as f32, 1.0),
        ],
        1.0,
        4.0,
        palette::GRID.faded(3.0),
    );
    let curve: Vec<Point2> = e
        .iter()
        .enumerate()
        .map(|(k, v)| Point2::xy(((k as f64 + 0.5) * dt) as f32, (v / reference) as f32))
        .collect();
    ax.polyline(c, &curve, 1.6, palette::INK);
    ax.scatter(
        c,
        &curve[BEFORE_ROW..=BEFORE_ROW],
        Marker::Dot,
        7.0,
        palette::SKY,
    );
    ax.scatter(
        c,
        &curve[AFTER_ROW..=AFTER_ROW],
        Marker::Dot,
        7.0,
        palette::ORANGE,
    );
    let now = curve[row.min(curve.len() - 1)];
    ax.scatter(c, &[now], Marker::Dot, 8.0, palette::YELLOW);
}

/// The triangle with the hat of corner 1, the light cone through its centroid, and its
/// gradient's Euclidean normal (sky) beside its Lorentzian dual (orange).
fn draw_hat(c: &mut Canvas, rect: Rect, s: &Shown) {
    let corners = s.triangle.map(at);
    let ax = Axes::fitting(rect, corners, 1.3);
    ax.fill(c, &corners, palette::BLUE, 0.18);
    let (index, others) = (1, [2, 0]);
    for k in 1..=LEVELS {
        let level = k as f64 / (LEVELS + 1) as f64;
        let ends = others
            .map(|o| at(s.triangle[index] + (s.triangle[o] - s.triangle[index]) * (1.0 - level)));
        ax.line(c, ends[0], ends[1], 0.9, palette::SKY.faded(0.45));
    }
    ax.polyline(
        c,
        &[corners[0], corners[1], corners[2], corners[0]],
        1.0,
        palette::INK.faded(0.5),
    );
    ax.line(c, corners[others[0]], corners[others[1]], 2.4, palette::INK);
    ax.scatter(c, &corners, Marker::Dot, 5.0, palette::INK.faded(0.7));
    ax.scatter(
        c,
        &corners[index..=index],
        Marker::Dot,
        9.0,
        palette::ORANGE,
    );
    let centroid = (s.triangle[0] + s.triangle[1] + s.triangle[2]) * (1.0 / 3.0);
    cone(c, &ax, null_rays(centroid, 0.38, 0.38));
    let lorentz = dual().of(s.gradients[index]);
    let euclidean = -(t_plane() >> lorentz);
    let [lx, lt] = coordinates(lorentz);
    let scale = 0.28 / gax::vga2d::Vector::new(lx, lt).norm();
    for (direction, l) in [(euclidean, palette::SKY), (lorentz, palette::ORANGE)] {
        ax.arrow(
            c,
            at(centroid),
            at(centroid + direction * scale),
            2.0,
            9.0,
            l,
        );
    }
    ax.scatter(c, &[at(centroid)], Marker::Dot, 6.0, palette::INK);
    label_under(c, rect, "A HAT: NORMAL AND LORENTZ DUAL", 14.0, 0.95);
}

/// The first edge's form, as the direction its dual drives, across the triangle.
fn draw_edge(c: &mut Canvas, rect: Rect, s: &Shown) {
    let corners = s.triangle.map(at);
    let ax = Axes::fitting(rect, corners, 1.3);
    ax.fill(c, &corners, palette::BLUE, 0.18);
    ax.polyline(
        c,
        &[corners[0], corners[1], corners[2], corners[0]],
        1.0,
        palette::INK.faded(0.5),
    );
    cone(c, &ax, null_rays(s.triangle[0], 0.42, 0.42));
    let longest = s
        .edge_field
        .iter()
        .map(|&d| {
            let [x, t] = coordinates(d);
            gax::vga2d::Vector::new(x, t).norm()
        })
        .fold(0.0, f64::max);
    for (&p, &d) in s.samples.iter().zip(&s.edge_field) {
        let [x, t] = coordinates(d * (0.11 / longest));
        let tail = at(p) - Point2::direction(0.45 * x as f32, 0.45 * t as f32);
        arrow(c, &ax, tail, [x, t], 1.2, 5.0, palette::SKY);
    }
    ax.line(c, corners[1], corners[2], 2.4, palette::ORANGE);
    ax.scatter(c, &corners[0..1], Marker::Dot, 9.0, palette::SKY);
    label_under(c, rect, "AN EDGE FORM CIRCULATES", 14.0, 0.95);
}

/// The causal wake of a source event, inside its forward light cone.
fn draw_wake(c: &mut Canvas, rect: Rect, s: &Shown) {
    let ax = Axes::fitting(rect, s.wake_flat.points.iter().copied(), 1.04);
    let values: Vec<f64> = s.wake.wake.iter().map(|w| w.s() / s.wake_limit).collect();
    s.wake_flat.shade_field(c, &ax, &values, wave_light);
    cone(c, &ax, null_rays(s.wake.source, 0.0, 0.5));
    ax.scatter(c, &[at(s.wake.source)], Marker::Dot, 9.0, palette::ORANGE);
    ax.stroke(c, &s.wake_flat.rim(), 1.0, palette::INK.faded(0.5));
    label_under(c, rect, "THE CAUSAL WAKE OF A SOURCE", 16.0, 0.95);
}

fn draw(c: &mut Canvas, s: &Shown, t: f32) {
    backdrop(c);
    let screen = c.rect();
    let (w, h) = (screen.width(), screen.height());
    let below = Rect::new(0.0, h * 0.13, w, h * 0.97);
    let row = row_at(t);
    draw_worldsheet(
        c,
        below.part(0.0, 0.0, 0.3, 1.0).inset(8.0, 6.0, 8.0, 18.0),
        s,
        row,
    );
    let middle = below.part(0.3, 0.0, 0.64, 1.0);
    draw_cavity(
        c,
        middle
            .part(0.0, 0.0, 1.0, 0.5)
            .inset(44.0, 26.0, 10.0, 34.0),
        s,
        row,
    );
    draw_energy(
        c,
        middle
            .part(0.0, 0.5, 1.0, 1.0)
            .inset(44.0, 26.0, 10.0, 34.0),
        s,
        row,
    );
    let right = below.part(0.64, 0.0, 1.0, 1.0);
    draw_hat(
        c,
        right.part(0.0, 0.0, 0.5, 0.48).inset(4.0, 4.0, 4.0, 14.0),
        s,
    );
    draw_edge(
        c,
        right.part(0.5, 0.0, 1.0, 0.48).inset(4.0, 4.0, 4.0, 14.0),
        s,
    );
    draw_wake(
        c,
        right.part(0.0, 0.5, 1.0, 1.0).inset(16.0, 4.0, 16.0, 18.0),
        s,
    );
    caption(
        c,
        "A MOVING MIRROR: THE REFLECTED PULSE COMES BACK COMPRESSED, WITH MORE ENERGY",
        "THE LORENTZIAN DUAL PULLED BACK THROUGH THE GRADIENT'S ADJUGATE IS THE WAVE OPERATOR; THE WALLS AND FIRST ROWS CHOOSE THE CAUSAL SOLUTION (X+T-)",
    );
}

fn main() {
    run_with(
        Anim::new("moving mirror", SECONDS).size(960, 540),
        shown,
        draw,
    );
}
