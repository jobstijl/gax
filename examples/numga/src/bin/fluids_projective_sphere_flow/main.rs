//! numga's `fluids/projective_sphere_flow`: steady diffusion on a fixed physical surface,
//! carried by weighted corner hats and edge forms, on the sphere (VGA3D) and on the affine
//! plane (PGA2D).
//!
//! A triangle's corners are points, and points are projective: a point and any multiple of it
//! are the same place. The side facing a corner, the join of the other two, over the join of
//! all three, is that corner's reciprocal plane, and the reciprocals sum to the triangle's
//! section. A point's hat of a corner is the corner's reading over the section's: a quotient of
//! two linear forms, unchanged when the point is scaled, so the hats sum to one at every point
//! of the ray, and their gradients annihilate the radial direction. Weighting the corners
//! changes the section, and with it the interpolation the triangle carries; it does not change
//! the place.
//!
//! On the sphere, the samples of the quadrature and their measures belong to the physical
//! surface, and so do the source (twice the height) and the material. The gradient's adjugate,
//! its transpose through the join, pulls the material's response back to the vertices: the
//! stiffness. Each frame lifts one vertex along its ray and solves the same problem again,
//! under the rational interpolation that the new weights choose. The solution stays the height,
//! the sphere's first eigenfunction, to the mesh's resolution; only the interpolation moves.
//!
//! Left, the lifted vertices in space, their rays from the sphere, the mesh between them and its
//! dual, the faces' sections as points on the sphere joined by great-circle arcs; middle, the
//! potential on the fixed sphere; right, the same quotient on PGA2D's affine slice, with weighted
//! corners, where the material pairs spatial directions only and the current is a direction.

mod flow;
#[cfg(test)]
mod tests;

#[path = "../../shared/simplicial.rs"]
mod simplicial;
#[path = "../../shared/surfaces_plane.rs"]
mod surfaces_plane;

use flow::*;
use gax_numga_examples::measure::roots;
use gax_numga_examples::scene3::{Scene3, panel3};
use gax_numga_examples::signal::{frame, smoothstep};
use gax_numga_examples::{
    Anim, Axes, Camera, Canvas, Lens, Light, Marker, ORIGIN3, Point2, Point3, Rect, backdrop,
    caption, colormap, label_under, on_sphere, palette, reach3, run_with,
};
use simplicial::faces;
use surfaces_plane::{Flat, arrow};

/// One loop: numga's 90 frames of 40 ms.
const SECONDS: f32 = 3.6;
/// The views: numga's elevation and azimuth (radians), and the half heights of the space and
/// of the sphere.
const ELEVATION: f32 = 0.384;
const AZIMUTH: f32 = 0.960;
const SPACE_HALF: f32 = 1.85;
const SPHERE_HALF: f32 = 1.15;
/// The points along each arc of the dual.
const ARC_SEGMENTS: usize = 16;
/// The potential's level lines, a quarter apart.
const LEVEL_STEP: f64 = 0.25;
/// The arcs' lift off the sphere, so that they draw over its shell.
const LIFT: f64 = 1.004;

struct Shown {
    sphere: SphereScene,
    frames: Vec<Frame>,
    planar: Planar,
    flat: Flat,
}

fn shown() -> Shown {
    let sphere = SphereScene::new();
    let phases: Vec<f64> = (0..FRAMES)
        .map(|k| core::f64::consts::TAU * k as f64 / FRAMES as f64)
        .collect();
    let frames = gax_numga_examples::par_map(&phases, |&phase| sphere.frame(&sphere.scales(phase)));
    let planar = Planar::new();
    let coordinates: Vec<[f64; 2]> = planar
        .points
        .iter()
        .map(|p| {
            let [x, y] = p.to_euclidean();
            [x, y]
        })
        .collect();
    let flat = Flat::new(&coordinates, &faces(&planar.grid));
    Shown {
        sphere,
        frames,
        planar,
        flat,
    }
}

/// A point of space as drawn.
fn drawn(p: sphere::Point) -> Point3 {
    let [x, y, z] = coordinates(p).map(|c| c as f32);
    Point3::xyz(x, y, z)
}

/// The potential's light: orange where it is high, sky where low, faint at zero.
fn potential_light(v: f64) -> Light {
    let t = v.clamp(-1.0, 1.0) as f32;
    let accent = if t > 0.0 {
        palette::ORANGE
    } else {
        palette::SKY
    };
    palette::GRID.faded(0.8) + accent.faded(0.1 + 0.4 * t.abs())
}

/// The half height that shows `half` either way in a panel, whichever side is shorter.
fn half_height(rect: Rect, half: f32) -> f32 {
    half * (rect.height() / rect.width()).max(1.0)
}

fn camera(rect: Rect, half: f32) -> Camera {
    Camera::orbit(rect, ORIGIN3, 8.0, AZIMUTH, ELEVATION, Lens::Parallel(half))
}

/// The lifted mesh in space: the shell of the sphere, each vertex's ray from it, the mesh's
/// edges and the dual's arcs.
fn draw_space(c: &mut Canvas, rect: Rect, s: &SphereScene, f: &Frame) {
    panel3(
        c,
        rect,
        camera(rect, half_height(rect, SPACE_HALF)),
        |scene: &mut Scene3| {
            let at = |u: f32, v: f32| {
                use core::f32::consts::{PI, TAU};
                reach3(on_sphere(TAU * u, PI * (v - 0.5)))
            };
            scene.surface(at, 36, 18, |_, _| palette::BLUE.faded(0.25), 0.25, None);
            let edges: Vec<[Point3; 2]> = (0..s.mesh.len(1))
                .map(|e| {
                    let ends = s.mesh.simplex(1, e);
                    [drawn(f.lifted.site(ends[0])), drawn(f.lifted.site(ends[1]))]
                })
                .collect();
            scene.stroke(&edges, 1.3, palette::BLUE);
            // The chord between the poles of the faces beside each edge, pushed out onto the sphere.
            for &[a, b] in &s.beside {
                let (p, q) = (f.poles.site(a), f.poles.site(b));
                let arc: Vec<Point3> = (0..=ARC_SEGMENTS)
                    .map(|k| {
                        let w = k as f64 / ARC_SEGMENTS as f64;
                        drawn((p * (1.0 - w) + q * w).normalized() * LIFT)
                    })
                    .collect();
                scene.polyline(&arc, 1.1, palette::ORANGE.faded(0.8));
            }
            for v in 0..s.vertices.len() {
                let lifted = f.lifted.site(v);
                let on_sphere = drawn(lifted.normalized());
                let place = drawn(lifted);
                if (place - on_sphere).ideal_norm() > 1e-3 {
                    scene.seg(on_sphere, place, 1.0, palette::INK.faded(0.45));
                }
                let (size, colour) = if v == s.chosen {
                    (8.0, palette::YELLOW)
                } else {
                    (4.5, palette::BLUE.whitened(0.3))
                };
                scene.dot(place, Marker::Dot, size, colour);
            }
        },
    );
}

/// The point of the unit sphere a ray from the camera meets first, if it meets it: from the
/// ray's origin `o` along the unit direction `d`, the nearer root of `|o + t d|² = 1`.
fn hit(origin: Point3, direction: Point3) -> Option<sphere::Point> {
    let [x, y, z] = origin.to_euclidean().map(f64::from);
    let o = sphere::Plane::new(x, y, z);
    let d = sphere::Plane::new(
        f64::from(direction.e032()),
        f64::from(direction.e013()),
        f64::from(direction.e021()),
    );
    let (middle, half) = roots(1.0, (o | d).s(), o.norm_squared() - 1.0)?;
    Some((o + d * (middle - half)).hodge())
}

/// The potential on the fixed sphere, as numga's shader draws it: at every pixel the face
/// holding the point, the interpolation's value there, lit by the facing of the surface, and
/// a level line at every quarter.
fn draw_sphere(c: &mut Canvas, rect: Rect, s: &SphereScene, f: &Frame) {
    let cam = camera(rect, half_height(rect, SPHERE_HALF)).viewport(rect);
    let eye = cam.eye().unitized() - ORIGIN3;
    let to_eye = sphere::Plane::new(
        f64::from(eye.e032()),
        f64::from(eye.e013()),
        f64::from(eye.e021()),
    )
    .normalized();
    c.clip(rect);
    c.shade(2, |q| {
        let (origin, direction) = cam.ray(q);
        let p = hit(origin, direction)?;
        let face = s.face_of(p)?;
        let value = f.value(face, p);
        let facing = (to_eye & p).s().max(0.0);
        // The distance to the nearest quarter, in quarters: a line where it is small.
        let quarters = value / LEVEL_STEP;
        let off = (quarters - quarters.round()).abs();
        let line = smoothstep(1.0 - off as f32 / 0.07);
        Some(
            potential_light(value).faded(0.5 + 0.5 * facing as f32)
                + palette::INK.faded(0.55 * line),
        )
    });
    c.unclip();
}

/// The affine slice: the potential shaded with its level lines, and the current as arrows.
fn draw_planar(c: &mut Canvas, rect: Rect, p: &Planar, flat: &Flat) {
    let ax = Axes::fitting(rect, flat.points.iter().copied(), 1.12);
    flat.shade_field(c, &ax, &p.potential, |v| {
        colormap::viridis(v.clamp(0.0, 1.0) as f32).faded(0.5)
    });
    for k in 1..8 {
        let level = f64::from(k) / 8.0;
        ax.stroke(
            c,
            &flat.level(&p.potential, level),
            0.8,
            palette::INK.faded(0.4),
        );
    }
    ax.stroke(c, &flat.rim(), 1.4, palette::INK.faded(0.7));
    let fastest = p.current.iter().map(|d| d.ideal_norm()).fold(0.0, f64::max);
    for (&at, &d) in p.points.iter().zip(&p.current) {
        let [x, y] = at.to_euclidean();
        let k = 0.16 / fastest;
        let reach = [d.e20() * k, d.e01() * k];
        let tail = Point2::xy(x as f32, y as f32)
            - Point2::direction(0.45 * reach[0] as f32, 0.45 * reach[1] as f32);
        arrow(c, &ax, tail, reach, 1.1, 5.0, palette::INK.faded(0.9));
    }
    let corners = PLANAR_CORNERS.map(|[x, y]| Point2::xy(x as f32, y as f32));
    ax.scatter(c, &corners, Marker::Dot, 7.0, palette::ORANGE);
}

fn draw(c: &mut Canvas, s: &Shown, t: f32) {
    backdrop(c);
    let screen = c.rect();
    let (w, h) = (screen.width(), screen.height());
    let below = Rect::new(0.0, h * 0.14, w, h * 0.93);
    let panels = [0, 1, 2].map(|i| below.column(i, 3).inset(8.0, 6.0, 8.0, 6.0));
    let f = &s.frames[frame((t / SECONDS).rem_euclid(1.0), FRAMES)];
    draw_space(c, panels[0], &s.sphere, f);
    label_under(
        c,
        panels[0],
        "LIFTED VERTICES, THEIR MESH AND DUAL",
        22.0,
        0.9,
    );
    draw_sphere(c, panels[1], &s.sphere, f);
    label_under(c, panels[1], "THE POTENTIAL ON THE FIXED SPHERE", 22.0, 0.9);
    draw_planar(c, panels[2], &s.planar, &s.flat);
    label_under(
        c,
        panels[2],
        "WEIGHTED CORNERS ON THE AFFINE SLICE",
        22.0,
        0.9,
    );
    caption(
        c,
        "PROJECTIVE SPHERE FLOW: THE WEIGHTS CHOOSE THE INTERPOLATION, NOT THE PLACE",
        "ONE VERTEX SLIDES ALONG ITS RAY; DIFFUSION ON THE SAME SPHERE STAYS THE HEIGHT (VGA3D), AND THE QUOTIENT READS THE PLANE TOO (PGA2D)",
    );
}

fn main() {
    run_with(
        Anim::new("projective sphere flow", SECONDS).size(960, 540),
        shown,
        draw,
    );
}
