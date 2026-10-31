//! numga's `surfaces/regge_calculus`: a surface's intrinsic geometry from its edge lengths
//! alone, its curvature, and the current it carries (VGA2D).
//!
//! A metric is a correlation from displacements to planes: `metric(edge) & edge` reads an
//! edge's squared length. On a triangle, the symmetric products of its corner gradients (each
//! side over twice the patch) give Regge's basis, one correlation per side that reads one on
//! that side and nothing on the other two, so the squared lengths, summed with them, are the
//! triangle's metric: with the Euclidean lengths, the Euclidean one. The wedge of the metric's
//! readings of two sides carries its area scaling; at each corner the metric inner product of the
//! two sides there with twice the intrinsic area in the bivector is a rotor, and the three rotors
//! multiply to minus one: half a turn. Around a vertex the corner turns fall short of a whole
//! turn by the angle defect, which over the vertex's area is the Gaussian curvature, and the
//! defects add up to a whole turn whatever the lengths (Gauss–Bonnet). Each side's dyad with the
//! corner cotangents is the area-weighted inverse metric, a conductivity, which the face
//! gradient's adjugate assembles into the intrinsic Laplace–Beltrami operator: the cotangent
//! Laplacian of the lengths.
//!
//! The animation shows numga's three figures in turn: the three basis forms of a triangle
//! (hyperbolic, null along the lines parallel to the other two sides) and their sum's ellipses;
//! the curvature of a disk whose lengths are stretched radially or around it, with each sampled
//! face's unit ellipse of the metric beside a circle; and the steady current between two
//! electrodes on the flat and the two stretched disks. Geometry from edge lengths (intrinsic
//! areas, corner turns, defects) is written here: `gax::dec::Geometry` measures from positions.

mod regge;
#[cfg(test)]
mod tests;

#[path = "../../shared/simplicial.rs"]
mod simplicial;
#[path = "../../shared/surfaces_plane.rs"]
mod surfaces_plane;

use gax::field::Field;
use gax_numga_examples::contour;
use gax_numga_examples::{
    Anim, Axes, Canvas, Light, Marker, Point2, Rect, backdrop, caption, label_under, palette,
    run_with,
};
use regge::*;
use simplicial::{exterior, faces};
use surfaces_plane::{Flat, arrow};

/// Each figure's time on screen.
const FIGURE: f32 = 4.0;
const SECONDS: f32 = 3.0 * FIGURE;
/// The potential's levels, the ellipses' radius and samples, and the grid of the level lines on
/// the triangle.
const LEVELS: usize = 7;
const RADIUS: f64 = 0.082;
const ELLIPSE_SAMPLES: usize = 48;
const CONTOUR_GRID: usize = 120;

/// The first figure: the triangle, its Regge basis and the stretched metric.
struct Basis {
    corners: [Point; 3],
    triangle: Flat,
    basis: [Metric; 3],
    metric: Metric,
}

/// The disk and, for each case, its curvature at the vertices (zero on the rim), its metrics,
/// and its potential and vertex currents.
struct Disk {
    mesh: Flat,
    curvatures: [Vec<f64>; 3],
    metrics: [Vec<Metric>; 3],
    ellipse_faces: Vec<usize>,
    potentials: [Vec<f64>; 3],
    currents: [Vec<[f64; 2]>; 3],
}

struct Shown {
    basis: Basis,
    disk: Disk,
}

/// Well-spaced faces from the centre outward: those whose centre lies within `reach` of the
/// origin and farther than `apart` from every face chosen before it.
fn spaced_faces(centres: &[[f64; 2]], reach: f64, apart: f64) -> Vec<usize> {
    let size = |c: [f64; 2]| gax::vga2d::Vector::new(c[0], c[1]).norm();
    let mut order: Vec<usize> = (0..centres.len())
        .filter(|&f| size(centres[f]) <= reach)
        .collect();
    order.sort_by(|&a, &b| size(centres[a]).total_cmp(&size(centres[b])));
    let mut chosen: Vec<usize> = Vec::new();
    for f in order {
        let far = |g: &usize| {
            size([
                centres[f][0] - centres[*g][0],
                centres[f][1] - centres[*g][1],
            ]) > apart
        };
        if chosen.iter().all(far) {
            chosen.push(f);
        }
    }
    chosen
}

fn shown() -> Shown {
    let corners = single_triangle();
    let (basis, _, _) = regge_metric_basis(corners);
    let squared: [f64; 3] =
        core::array::from_fn(|k| sides(corners)[k].norm_squared() * EDGE_STRETCHES[k]);
    let (metric, ..) = triangle_geometry(corners, squared);
    let (mesh, vertices) = disk(DISK_RADIUS, DISK_DIVISIONS);
    let rim = exterior(&mesh, 0);
    let geometry: Vec<Intrinsic> = (0..3)
        .map(|case| mesh_geometry(&mesh, &vertices, &stretched(&mesh, &vertices, case)))
        .collect();
    let (held, prescribed) = electrodes(&mesh, &vertices);
    let solved: Vec<_> = geometry
        .iter()
        .map(|g| intrinsic_conduction(&mesh, &vertices, &g.conductivities, &held, &prescribed))
        .collect();
    let centres: Vec<[f64; 2]> = (&mesh.means::<Scalar>(2) * &vertices)
        .iter()
        .map(coordinates)
        .collect();
    // Area-weighted vertex currents: a continuous drawing of the facewise solution.
    let areas: Vec<f64> = (0..mesh.len(2))
        .map(|f| {
            let [a, b, c] = [0, 1, 2].map(|k| vertices.site(mesh.simplex(2, f)[k]));
            ((b - a) & (c - a)).s().abs() * 0.5
        })
        .collect();
    let vertex_current = |current: &Field<Point>| {
        let mut sum = vec![[0.0; 3]; vertices.len()];
        for (f, &area) in areas.iter().enumerate() {
            let [x, y] = coordinates(current.site(f));
            for &v in mesh.simplex(2, f) {
                sum[v][0] += area * x;
                sum[v][1] += area * y;
                sum[v][2] += area;
            }
        }
        sum.iter().map(|s| [s[0] / s[2], s[1] / s[2]]).collect()
    };
    Shown {
        basis: Basis {
            corners,
            triangle: Flat::new(&corners.map(coordinates), &[[0, 1, 2]]),
            basis,
            metric,
        },
        disk: Disk {
            mesh: Flat::new(
                &vertices.iter().map(coordinates).collect::<Vec<_>>(),
                &faces(&mesh),
            ),
            curvatures: core::array::from_fn(|case| {
                (0..vertices.len())
                    .map(|v| {
                        if rim.contains(&v) {
                            0.0
                        } else {
                            geometry[case].curvatures.site(v).e12()
                        }
                    })
                    .collect()
            }),
            metrics: core::array::from_fn(|case| geometry[case].metrics.clone()),
            ellipse_faces: spaced_faces(&centres, 0.84, 0.24),
            potentials: core::array::from_fn(|case| solved[case].0.iter().map(|s| s.s()).collect()),
            currents: core::array::from_fn(|case| vertex_current(&solved[case].1)),
        },
    }
}

/// A point of the plane as drawn.
fn at(p: Point) -> Point2 {
    let [x, y] = coordinates(p);
    Point2::xy(x as f32, y as f32)
}

/// A diverging light: orange where positive, sky where negative, the grid's faint blue at zero.
fn diverging(t: f64) -> Light {
    let t = t.clamp(-1.0, 1.0) as f32;
    let accent = if t > 0.0 {
        palette::ORANGE
    } else {
        palette::SKY
    };
    palette::GRID.faded(0.6) + accent.faded(0.45 * t.abs())
}

/// The level line `f = level` over the axes, inside the mesh only.
fn level_inside(ax: &Axes, mesh: &Flat, f: impl Fn(Point2) -> f32, level: f32) -> Vec<[Point2; 2]> {
    let segs = contour::of_fn(f, ax.x, ax.y, CONTOUR_GRID, level);
    segs.into_iter()
        .filter(|[a, b]| mesh.locate((*a + *b).unitized()).is_some())
        .collect()
}

/// The quadratic form of a metric at a step from the centroid, at a canvas point.
fn form(metric: Metric, centroid: Point, q: Point2) -> f64 {
    let [x, y] = q.to_euclidean().map(f64::from);
    let step = point(x, y) - centroid;
    (metric.of(step) & step).s()
}

/// The first figure: the three basis forms, then their sum's ellipses.
fn draw_basis(c: &mut Canvas, b: &Basis) {
    let screen = c.rect();
    let below = Rect::new(
        0.0,
        screen.height() * 0.2,
        screen.width(),
        screen.height() * 0.84,
    );
    let points = b.corners.map(at);
    let centroid = (b.corners[0] + b.corners[1] + b.corners[2]) * (1.0 / 3.0);
    let ends = simplicial::corners(2, 1);
    // The forms' bound over the triangle: their largest size at its corners, where a quadratic
    // form on it peaks.
    let bound = b
        .basis
        .iter()
        .flat_map(|m| {
            b.corners
                .iter()
                .map(move |&p| (m.of(p - centroid) & (p - centroid)).s().abs())
        })
        .fold(0.0, f64::max);
    for (k, &metric) in b.basis.iter().enumerate() {
        let rect = below.column(k, 4).inset(6.0, 6.0, 6.0, 6.0);
        let ax = Axes::fitting(rect, points, 1.22);
        b.triangle.shade(c, &ax, 2, |_, w| {
            let q = (0..3).fold(Point2::xy(0.0, 0.0) - Point2::xy(0.0, 0.0), |acc, j| {
                acc + (points[j] - Point2::xy(0.0, 0.0)).gp(w[j])
            });
            Some(diverging(
                form(metric, centroid, Point2::xy(0.0, 0.0) + q) / bound,
            ))
        });
        for level in [-0.75, -0.45, -0.15, 0.15, 0.45, 0.75] {
            let f = |q: Point2| form(metric, centroid, q) as f32;
            let segs = level_inside(&ax, &b.triangle, f, (level * bound) as f32);
            ax.stroke(c, &segs, 0.8, palette::INK.faded(0.35));
        }
        let null = level_inside(&ax, &b.triangle, |q| form(metric, centroid, q) as f32, 0.0);
        ax.stroke(c, &null, 1.5, palette::SKY);
        ax.stroke(c, &b.triangle.wire(), 1.1, palette::INK.faded(0.5));
        ax.line(
            c,
            points[ends[k][0]],
            points[ends[k][1]],
            2.6,
            palette::ORANGE,
        );
        ax.scatter(c, &points, Marker::Dot, 5.0, palette::INK.faded(0.7));
        label_under(
            c,
            rect,
            &format!("THE FORM OF SIDE {k}: ONE ON IT"),
            14.0,
            0.9,
        );
    }
    let rect = below.column(3, 4).inset(6.0, 6.0, 6.0, 6.0);
    let ax = Axes::fitting(rect, points, 1.22);
    let largest = b
        .corners
        .iter()
        .map(|&p| (b.metric.of(p - centroid) & (p - centroid)).s())
        .fold(0.0, f64::max);
    ax.image(c, 2, |q| {
        b.triangle.locate(q)?;
        let value = form(b.metric, centroid, q) / largest;
        Some(palette::SKY.faded(0.03 + 0.4 * value.clamp(0.0, 1.0) as f32))
    });
    for k in 0..4 {
        let level = largest * (0.03 + 0.25 * k as f64 / 3.0);
        let ellipse = level_inside(
            &ax,
            &b.triangle,
            |q| form(b.metric, centroid, q) as f32,
            level as f32,
        );
        ax.stroke(c, &ellipse, 1.8, palette::ORANGE);
        let circle = level_inside(
            &ax,
            &b.triangle,
            |q| form(euclidean(), centroid, q) as f32,
            level as f32,
        );
        ax.stroke(c, &circle, 0.9, palette::INK.faded(0.45));
    }
    ax.stroke(c, &b.triangle.wire(), 1.2, palette::INK.faded(0.5));
    ax.scatter(c, &points, Marker::Dot, 8.0, palette::ORANGE);
    ax.scatter(
        c,
        &[at(centroid)],
        Marker::Dot,
        5.0,
        palette::INK.faded(0.7),
    );
    label_under(c, rect, "THEIR SUM: THE METRIC'S ELLIPSES", 14.0, 0.9);
}

/// The Euclidean metric: the inverse of the dual.
fn euclidean() -> Metric {
    Plane::slot().hodge().inverse()
}

/// The unit ellipse of a metric about a point, `radius` across, as `ELLIPSE_SAMPLES + 1` points:
/// each direction scaled to where the metric reads `radius²`.
fn ellipse(metric: Metric, centre: [f64; 2], radius: f64) -> Vec<Point2> {
    (0..=ELLIPSE_SAMPLES)
        .map(|k| {
            let d = gax::vga2d::Vector::from_angle(
                core::f64::consts::TAU * k as f64 / ELLIPSE_SAMPLES as f64,
            );
            let u = point(d.e1(), d.e2());
            let reach = radius / root((metric.of(u) & u).s());
            Point2::xy(
                (centre[0] + reach * d.e1()) as f32,
                (centre[1] + reach * d.e2()) as f32,
            )
        })
        .collect()
}

/// Three disk panels in a row, each drawn by `panel`.
fn disk_row(
    c: &mut Canvas,
    disk: &Flat,
    labels: [&str; 3],
    mut panel: impl FnMut(&mut Canvas, &Axes, usize),
) {
    let screen = c.rect();
    let below = Rect::new(
        0.0,
        screen.height() * 0.15,
        screen.width(),
        screen.height() * 0.92,
    );
    for (case, label) in labels.iter().enumerate() {
        let rect = below.column(case, 3).inset(8.0, 6.0, 8.0, 6.0);
        let ax = Axes::fitting(rect, disk.points.iter().copied(), 1.06);
        panel(c, &ax, case);
        label_under(c, rect, label, 20.0, 0.9);
    }
}

const CASES: [&str; 3] = ["FLAT", "STRETCHED RADIALLY", "STRETCHED AROUND"];

/// The second figure: the curvature of each case, and its metric's ellipses at sampled faces.
fn draw_curvature(c: &mut Canvas, d: &Disk) {
    let bound = d
        .curvatures
        .iter()
        .flatten()
        .map(|k| k.abs())
        .fold(0.0, f64::max);
    disk_row(c, &d.mesh, CASES, |c, ax, case| {
        d.mesh
            .shade_field(c, ax, &d.curvatures[case], |k| diverging(k / bound));
        ax.stroke(c, &d.mesh.wire(), 0.5, palette::GRID);
        // Each ellipse coloured by how much more the metric stretches radially than around.
        for &f in &d.ellipse_faces {
            let [x, y] = d.mesh.centroid(f).to_euclidean().map(f64::from);
            let metric = d.metrics[case][f];
            let radial = point(x, y);
            let around = Bivector::new(-core::f64::consts::FRAC_PI_4).exp() >> radial;
            let stretch = |u: Point| (metric.of(u) & u).s();
            let ratio = stretch(radial) / stretch(around);
            let tint = if ratio > 1.0 {
                palette::ORANGE
            } else {
                palette::SKY
            };
            let strength = ((ratio - 1.0).abs() * 1.5).min(1.0) as f32;
            let colour = palette::INK.faded(0.6).blend(tint, strength);
            let shape = ellipse(metric, [x, y], RADIUS);
            ax.fill(c, &shape, colour, 0.22);
            ax.polyline(
                c,
                &ellipse(euclidean(), [x, y], RADIUS),
                0.8,
                palette::INK.faded(0.45),
            );
            ax.polyline(c, &shape, 1.5, colour);
        }
        ax.stroke(c, &d.mesh.rim(), 1.3, palette::INK.faded(0.7));
    });
}

/// The third figure: the potential and current between the electrodes, each case.
fn draw_flow(c: &mut Canvas, d: &Disk) {
    disk_row(c, &d.mesh, CASES, |c, ax, case| {
        d.mesh.shade_field(c, ax, &d.potentials[case], diverging);
        for k in 1..=LEVELS {
            let level = -0.85 + 1.7 * k as f64 / (LEVELS + 1) as f64;
            ax.stroke(
                c,
                &d.mesh.level(&d.potentials[case], level),
                0.9,
                palette::INK.faded(0.4),
            );
        }
        ax.stroke(c, &d.mesh.wire(), 0.5, palette::GRID);
        // The arrows to the scale of all three cases' currents: the median speed of the flat
        // case a fixed length, the fastest at most 1.6 times it.
        let mut speeds: Vec<f64> = d.currents[0]
            .iter()
            .map(|v| gax::vga2d::Vector::new(v[0], v[1]).norm())
            .collect();
        speeds.sort_by(f64::total_cmp);
        let reference = speeds[speeds.len() / 2];
        for (p, v) in d.mesh.points.iter().zip(&d.currents[case]) {
            let speed = gax::vga2d::Vector::new(v[0], v[1]).norm();
            let k = 0.08 / reference * (1.6 * reference / speed.max(1e-12)).min(1.0);
            let step = [v[0] * k, v[1] * k];
            let tail = *p - Point2::direction(0.45 * step[0] as f32, 0.45 * step[1] as f32);
            arrow(c, ax, tail, step, 1.2, 5.0, palette::INK.faded(0.9));
        }
        ax.stroke(c, &d.mesh.rim(), 1.3, palette::INK.faded(0.7));
    });
}

fn draw(c: &mut Canvas, s: &Shown, t: f32) {
    backdrop(c);
    match ((t / FIGURE).floor() as usize).min(2) {
        0 => {
            draw_basis(c, &s.basis);
            caption(
                c,
                "REGGE CALCULUS: A METRIC FROM EDGE LENGTHS",
                "SYMMETRIC PRODUCTS OF THE CORNER GRADIENTS READ ONE ON THEIR OWN SIDE; THE LENGTHS SUM THEM INTO A METRIC (VGA2D)",
            );
        }
        1 => {
            draw_curvature(c, &s.disk);
            caption(
                c,
                "REGGE CALCULUS: CURVATURE FROM CORNER TURNS",
                "THE ROTORS AT A VERTEX FALL SHORT OF A WHOLE TURN BY ITS DEFECT; THE DEFECTS ADD UP TO A TURN (VGA2D)",
            );
        }
        _ => {
            draw_flow(c, &s.disk);
            caption(
                c,
                "REGGE CALCULUS: CONDUCTION IN THE INTRINSIC METRIC",
                "THE AREA-WEIGHTED INVERSE METRIC, PULLED BACK BY THE FACE GRADIENT'S ADJUGATE, IS THE COTANGENT LAPLACIAN (VGA2D)",
            );
        }
    }
}

fn main() {
    run_with(
        Anim::new("regge calculus", SECONDS).size(960, 540),
        shown,
        draw,
    );
}
