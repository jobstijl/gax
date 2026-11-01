//! numga's `surfaces/dirac_kahler`: fields of every grade on a triangle mesh of the plane, one
//! geometric derivative that joins a scalar potential and a stream potential into one field,
//! the Hodge decomposition of a velocity, and the waves the derivative carries (VGA2D).
//!
//! On each face the corner gradients (each side over twice the patch, a vector) times a vertex
//! multivector field, summed over the corners, is its geometric derivative there: the
//! divergence of the vector part in grade 0; in grade 1 the scalar's gradient plus the bivector's
//! gradient turned a quarter turn; the curl in grade 2. On the even subalgebra, a scalar and a
//! bivector, it pairs a source-to-sink gradient with a circulating stream gradient into one
//! 1-form. Separately, the Whitney maps reconstruct cochains on the vertices, edges and faces
//! into one multivector at each face, whose top grade is the product's curl.
//!
//! A velocity, an intrinsic antivector, splits into a potential part that the medium drives, a
//! stream part that needs no metric (the stream function's gradient met with the patch runs
//! along its level curves), and a harmonic part circulating around a hole. The adjugate of the
//! even gradient, the transpose through the join, reads a velocity's divergence in grade 0 and
//! its circulation in grade 2 at once, so one mixed-grade Poisson solve on the even subalgebra
//! finds both potentials, and the three parts come out orthogonal in the energy. The same
//! adjugate, over the vertices' areas, is the skew partner of the even gradient: together they
//! carry waves that keep their energy, and a compressional pulse (grade 0 alone) turns partly
//! into vorticity (grade 2) where it meets the rims.
//!
//! The animation shows numga's three: the grades of the derivative of a polynomial field on a
//! disk beside Whitney's reconstruction, the Hodge decomposition of a velocity on the
//! perforated disk, and a pulse spreading there, lifted as reliefs of its scalar part (red) and
//! its bivector part (blue), and of all three grades together (red, green for the 1-form's
//! speed, blue). numga's fourth-order Runge–Kutta steps the pair of fields, the even one on the
//! vertices and the 1-form on the faces, as `shared/mechanics_lie.rs`'s `rk4` steps any state
//! that adds and scales.

mod dirac;
mod scenes;
#[cfg(test)]
mod tests;

use dirac::*;
use gax::field::Field;
use gax_numga_examples::scene3::panel3;
use gax_numga_examples::shared::simplicial;
use gax_numga_examples::shared::surfaces_plane;
use gax_numga_examples::signal::frame;
use gax_numga_examples::{
    Anim, Axes, Camera, Canvas, Lens, Light, ORIGIN3, Point2, Point3, Rect, backdrop, caption,
    label_under, palette, run_with,
};
use scenes::*;
use simplicial::faces;
use surfaces_plane::{Flat, arrow};

/// Each still figure's time on screen, and the waves' (numga's 61 frames of 40 ms, slowed).
const STILL: f32 = 3.5;
const WAVES: f32 = 6.0;
const SECONDS: f32 = 2.0 * STILL + WAVES;
/// The arrows' grid across each panel, and the levels of a potential.
const GRID: usize = 15;
const LEVELS: usize = 6;
/// The reliefs: the height of a unit amplitude, the view (numga's, in radians), and the camera's
/// distance.
const RELIEF_HEIGHT: f64 = 0.85;
const ELEVATION: f32 = 0.681;
const AZIMUTH: f32 = 0.384;
const DISTANCE: f32 = 5.6;

/// A field drawn as arrows: a value per vertex, the faces' values averaged at each.
struct Arrows {
    values: Vec<[f64; 2]>,
    light: Light,
    label: &'static str,
}

/// One figure of four flow panels on one mesh, with a potential's level lines on one of them.
struct Flows {
    mesh: Flat,
    fields: [Arrows; 4],
    levels: (usize, Vec<f64>),
}

/// The waves: the annulus at rest, and at each frame the scalar part, the bivector part and the
/// face field's speed at each vertex, over the first frame's largest amplitude.
struct Waves {
    mesh: Flat,
    frames: Vec<[Vec<f64>; 3]>,
}

struct Shown {
    grades: Flows,
    hodge: Flows,
    waves: Waves,
}

/// A planar mesh to draw on.
fn flat(mesh: &gax::mesh::Topology<3>, vertices: &Field<Point>) -> Flat {
    let points: Vec<[f64; 2]> = vertices.iter().map(coordinates).collect();
    Flat::new(&points, &faces(mesh))
}

/// A face field averaged at each vertex over the faces at it.
fn at_vertices(mesh: &gax::mesh::Topology<3>, field: &[[f64; 2]]) -> Vec<[f64; 2]> {
    let mut sum = vec![[0.0; 3]; mesh.len(0)];
    for (f, v) in field.iter().enumerate() {
        for &corner in mesh.simplex(2, f) {
            sum[corner][0] += v[0];
            sum[corner][1] += v[1];
            sum[corner][2] += 1.0;
        }
    }
    sum.iter().map(|s| [s[0] / s[2], s[1] / s[2]]).collect()
}

fn grade_flows() -> Flows {
    let (mesh, vertices) = disk(DISK_RADIUS, DISK_DIVISIONS);
    let ops = Operators::new(&mesh, &vertices);
    let g = grades(&ops, &mesh, &vertices, &GRADES);
    let plane = |v: Plane| [v.e1(), v.e2()];
    let gradient: Vec<[f64; 2]> = g.scalar_gradient.iter().map(plane).collect();
    let product: Vec<[f64; 2]> = g.face_dirac.iter().map(|m| [m.e1(), m.e2()]).collect();
    let turned: Vec<[f64; 2]> = product
        .iter()
        .zip(&gradient)
        .map(|(p, q)| [p[0] - q[0], p[1] - q[1]])
        .collect();
    let whitney: Vec<[f64; 2]> = g.whitney.iter().map(|m| [m.e1(), m.e2()]).collect();
    let arrows = |field: &[[f64; 2]], light, label| Arrows {
        values: at_vertices(&mesh, field),
        light,
        label,
    };
    Flows {
        mesh: flat(&mesh, &vertices),
        fields: [
            arrows(&gradient, palette::ORANGE, "THE SCALAR'S GRADIENT"),
            arrows(&turned, palette::SKY, "THE BIVECTOR'S, TURNED"),
            arrows(&product, palette::GREEN, "GRADE 1 OF THE PRODUCT"),
            arrows(&whitney, palette::PURPLE, "WHITNEY'S RECONSTRUCTION"),
        ],
        levels: (0, g.potential.iter().map(|s| s.s()).collect()),
    }
}

fn hodge_flows() -> Flows {
    let a = annulus(INNER_RADIUS, OUTER_RADIUS, ANNULUS_RINGS);
    let ops = Operators::new(&a.mesh, &a.vertices);
    let (_, harmonic) = harmonic_circulation(&ops, &a.inner_ring, &a.interior);
    let bubble = |[x, y]: [f64; 2]| {
        let r2 = x * x + y * y;
        (r2 - INNER_RADIUS * INNER_RADIUS) * (OUTER_RADIUS * OUTER_RADIUS - r2)
    };
    let along = |k: [f64; 2]| {
        a.vertices.map(|p| {
            let c = coordinates(p);
            Scalar::new(bubble(c) * (k[0] * c[0] + k[1] * c[1]))
        })
    };
    let total = &(&ops.potential_velocity.of(&along([-2.2, 0.9]))
        + &ops.stream_velocity.of(&along([1.6, 1.9])))
        + &harmonic.scaled(CIRCULATION);
    let parts = hodge_decomposition(&ops, &total, &a.interior);
    let arrows = |field: &Field<Point>, light, label| Arrows {
        values: at_vertices(&a.mesh, &field.iter().map(coordinates).collect::<Vec<_>>()),
        light,
        label,
    };
    Flows {
        mesh: flat(&a.mesh, &a.vertices),
        fields: [
            arrows(&total, palette::INK, "THE VELOCITY"),
            arrows(&parts.potential_flow, palette::ORANGE, "ITS POTENTIAL PART"),
            arrows(&parts.stream_flow, palette::SKY, "ITS STREAM PART"),
            arrows(&parts.harmonic_flow, palette::PURPLE, "ITS HARMONIC PART"),
        ],
        levels: (1, parts.potential.iter().map(|s| s.s()).collect()),
    }
}

fn waves() -> Waves {
    let a = annulus(INNER_RADIUS, OUTER_RADIUS, ANNULUS_RINGS);
    let ops = Operators::new(&a.mesh, &a.vertices);
    let states = pulse(&ops, &a.vertices, PULSE_WIDTH, DT, STEPS);
    let scale = states[0].0.iter().map(|e| e.norm()).fold(0.0, f64::max);
    let frames = states
        .iter()
        .step_by(STRIDE)
        .map(|(even, faces)| {
            let speeds: Vec<[f64; 2]> = faces.iter().map(|f| [f.norm(), 0.0]).collect();
            let speed = at_vertices(&a.mesh, &speeds);
            [
                even.iter().map(|e| e.s() / scale).collect(),
                even.iter().map(|e| e.e12() / scale).collect(),
                speed.iter().map(|s| s[0] / scale).collect(),
            ]
        })
        .collect();
    Waves {
        mesh: flat(&a.mesh, &a.vertices),
        frames,
    }
}

fn shown() -> Shown {
    Shown {
        grades: grade_flows(),
        hodge: hodge_flows(),
        waves: waves(),
    }
}

/// The value of a vertex field at a point of the mesh, if it is on it.
fn sample(mesh: &Flat, values: &[[f64; 2]], q: Point2) -> Option<[f64; 2]> {
    let (f, w) = mesh.locate(q)?;
    let corners = mesh.faces[f];
    let mut out = [0.0; 2];
    for k in 0..3 {
        for (o, v) in out.iter_mut().zip(values[corners[k]]) {
            *o += f64::from(w[k]) * v;
        }
    }
    Some(out)
}

/// A figure of four flow panels in a row.
fn draw_flows(c: &mut Canvas, flows: &Flows) {
    let screen = c.rect();
    let below = Rect::new(
        0.0,
        screen.height() * 0.2,
        screen.width(),
        screen.height() * 0.86,
    );
    let mesh = &flows.mesh;
    for (i, field) in flows.fields.iter().enumerate() {
        let rect = below.column(i, 4).inset(6.0, 6.0, 6.0, 6.0);
        let ax = Axes::fitting(rect, mesh.points.iter().copied(), 1.08);
        for face in &mesh.faces {
            ax.fill(c, &face.map(|v| mesh.points[v]), palette::BLUE, 0.07);
        }
        if flows.levels.0 == i {
            let values = &flows.levels.1;
            let bound = values.iter().map(|v| v.abs()).fold(0.0, f64::max);
            let spacing = 1.6 * bound / (LEVELS - 1) as f64;
            for k in 0..LEVELS {
                let level = spacing * (k as f64 - (LEVELS - 1) as f64 / 2.0);
                ax.stroke(c, &mesh.level(values, level), 0.8, palette::INK.faded(0.4));
            }
        }
        ax.stroke(c, &mesh.rim(), 1.1, palette::INK.faded(0.6));
        // Arrows on a grid across the panel, the 95th percentile of the speeds a fixed length.
        let mut speeds: Vec<f64> = field
            .values
            .iter()
            .map(|v| gax::vga2d::Vector::new(v[0], v[1]).norm())
            .collect();
        speeds.sort_by(f64::total_cmp);
        let reference = speeds[speeds.len() * 3 / 4].max(1e-12);
        let step = 2.0 / GRID as f32;
        for j in 0..=GRID {
            for k in 0..=GRID {
                let q = Point2::xy(-1.0 + step * k as f32, -1.0 + step * j as f32);
                if let Some(v) = sample(mesh, &field.values, q) {
                    let speed = gax::vga2d::Vector::new(v[0], v[1]).norm();
                    let scale = 0.12 / reference * (1.3 * reference / speed.max(1e-12)).min(1.0);
                    let d = [v[0] * scale, v[1] * scale];
                    let tail = q - Point2::direction(0.5 * d[0] as f32, 0.5 * d[1] as f32);
                    arrow(c, &ax, tail, d, 1.3, 5.5, field.light);
                }
            }
        }
        label_under(c, rect, field.label, 14.0, 0.9);
    }
}

/// A grade's light by the sign and size of its value: the accent where it is positive, dimmer
/// where negative, faint where it is small.
fn signed(accent: Light, value: f64) -> Light {
    let strength = (value.abs() * 2.6).min(1.0) as f32;
    let sign = if value >= 0.0 { 1.0 } else { 0.5 };
    palette::GRID + accent.faded(0.02 + 0.4 * strength * strength * sign)
}

/// One relief: the annulus lifted by `heights`, each face in the light `colour` gives its
/// corners, with the rim lifted and the rim at rest.
fn relief(
    c: &mut Canvas,
    rect: Rect,
    mesh: &Flat,
    heights: &[f64],
    colour: impl Fn([usize; 3]) -> Light,
) {
    let cam = Camera::orbit(
        rect,
        ORIGIN3,
        DISTANCE,
        AZIMUTH,
        ELEVATION,
        Lens::Perspective(0.5),
    );
    let lifted = |v: usize| {
        let [x, y] = mesh.points[v].to_euclidean();
        Point3::xyz(x, y, (RELIEF_HEIGHT * heights[v]) as f32)
    };
    let rest = |v: usize| {
        let [x, y] = mesh.points[v].to_euclidean();
        Point3::xyz(x, y, 0.0)
    };
    panel3(c, rect, cam, |s| {
        for &face in &mesh.faces {
            let [a, b, d] = face.map(lifted);
            let lit = s.lit(a, b, d, colour(face));
            s.tri(a, b, d, lit, 1.0);
        }
        let rim = mesh.rim_edges();
        let lifted_rim: Vec<[Point3; 2]> = rim.iter().map(|e| e.map(lifted)).collect();
        let rest_rim: Vec<[Point3; 2]> = rim.iter().map(|e| e.map(rest)).collect();
        s.stroke(&rest_rim, 0.8, palette::GRID.faded(2.0));
        s.stroke(&lifted_rim, 1.2, palette::INK.faded(0.5));
    });
}

/// The waves at `phase`: the scalar part, all three grades, the bivector part.
fn draw_waves(c: &mut Canvas, w: &Waves, phase: f32) {
    let screen = c.rect();
    let below = Rect::new(
        0.0,
        screen.height() * 0.14,
        screen.width(),
        screen.height() * 0.95,
    );
    let [scalar, bivector, speed] = &w.frames[frame(phase, w.frames.len())];
    let mean =
        |values: &[f64], face: [usize; 3]| face.iter().map(|&v| values[v]).sum::<f64>() / 3.0;
    let amplitude: Vec<f64> = (0..scalar.len())
        .map(|v| gax::vga3d::Vector::new(scalar[v], speed[v], bivector[v]).norm())
        .collect();
    let rects = [0, 1, 2].map(|i| below.column(i, 3).inset(4.0, 4.0, 4.0, 4.0));
    relief(c, rects[0], &w.mesh, scalar, |f| {
        signed(palette::RED, mean(scalar, f))
    });
    relief(c, rects[1], &w.mesh, &amplitude, |f| {
        let parts = [
            mean(scalar, f).abs(),
            mean(speed, f),
            mean(bivector, f).abs(),
        ];
        let largest = parts.iter().copied().fold(1e-3, f64::max);
        let size = (mean(&amplitude, f) * 3.2).min(1.0) as f32;
        let glow =
            |accent: Light, part: f64| accent.faded(size * (0.04 + 0.45 * (part / largest) as f32));
        palette::GRID
            + glow(palette::RED, parts[0])
            + glow(palette::GREEN, parts[1])
            + glow(palette::BLUE, parts[2])
    });
    relief(c, rects[2], &w.mesh, bivector, |f| {
        signed(palette::BLUE, mean(bivector, f))
    });
    let labels = [
        "SCALAR PART",
        "ALL THREE GRADES",
        "BIVECTOR PART: VORTICITY",
    ];
    for (rect, label) in rects.iter().zip(labels) {
        label_under(c, *rect, label, 16.0, 0.8);
    }
}

fn draw(c: &mut Canvas, s: &Shown, t: f32) {
    backdrop(c);
    if t < STILL {
        draw_flows(c, &s.grades);
        caption(
            c,
            "DIRAC-KAHLER: ONE PRODUCT, EVERY GRADE",
            "THE CORNER GRADIENTS TIMES A VERTEX MULTIVECTOR: GRADIENT PLUS TURNED BIVECTOR GRADIENT IN GRADE 1 (VGA2D)",
        );
    } else if t < 2.0 * STILL {
        draw_flows(c, &s.hodge);
        caption(
            c,
            "DIRAC-KAHLER: THE HODGE DECOMPOSITION OF A VELOCITY",
            "ONE EVEN POISSON SOLVE FINDS BOTH POTENTIALS; THE REST CIRCULATES AROUND THE HOLE (VGA2D)",
        );
    } else {
        draw_waves(c, &s.waves, (t - 2.0 * STILL) / WAVES);
        caption(
            c,
            "DIRAC-KAHLER WAVES AROUND AN OBSTACLE",
            "THE EVEN GRADIENT AND ITS ADJUGATE KEEP THE ENERGY; A COMPRESSIONAL PULSE TURNS INTO VORTICITY AT THE RIMS (VGA2D)",
        );
    }
}

fn main() {
    run_with(
        Anim::new("dirac kahler", SECONDS).size(960, 540),
        shown,
        draw,
    );
}
