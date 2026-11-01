//! numga's `fluids/floating`: the hydrostatic stability of floating bodies, from small tilts to
//! capsizing (PGA3D).
//!
//! A floating body is a closed surface of triangles, fixed in its own frame. Still water pushes
//! on each wet patch of the surface, square to it and in proportion to its depth: a force along a
//! line through the patch, a forque. The buoyancy is the sum of those lines over the wet surface,
//! one line of force, upward through the centre of the displaced water; for a closed surface it
//! holds the displaced weight (Archimedes), with no volume integral. Each triangle is cut at the
//! water into wet parts, and each part's push integrated exactly by three points halfway between
//! its corners and its centroid (`shared/mechanics_simplex.rs`). The body floats at the height
//! where the buoyancy balances its weight, found by Newton's method on the push's growth with the
//! water's rise. Tilted, the buoyancy line and the weight line (straight down through the centre
//! of gravity) no longer coincide, and their sum is the righting couple.
//!
//! The water is a plane, the centres points, forces forques and turns twists. Over the turns
//! about horizontal lines through the origin, the tilts (`Tilt::slot()`, heel `e23` and trim
//! `e31`), the change of the net force under a small tilt is a map, and paired with the tilt it
//! is the body's stiffness: its eigenvalues over the weight are the metacentric heights.
//!
//! The animation shows numga's four: a Wigley hull heeled in the water, its wet surface darker
//! with depth, with its buoyancy and its weight; the hull released from a heel with a low load,
//! rolling back upright, and with a high one, capsizing to float upside down; their righting arm
//! curves from upright to upside down with the slopes the stiffness gives; and a cube's righting
//! couple over density and heel, with cubes at rest above it, face up only outside a band of
//! densities.

mod hydro;
#[cfg(test)]
mod tests;

use gax::Unit;
use gax::pga3d::{Motor, Plane, Point};
use gax_numga_examples::scene3::{Scene3, panel3};
use gax_numga_examples::{
    Align, Anim, Axes, Camera, Canvas, Light, Marker, Point2, Rect, backdrop, caption, colormap,
    palette, run_with,
};
use hydro::*;

/// The loop: numga's 100 frames of 40 ms.
const SECONDS: f32 = 4.0;
/// The load and heel of the hull drawn whole.
const SHOWN_LOAD: usize = 0;
const SHOWN_HEEL: f64 = 35.0;
/// The depth at which the wet surface is darkest, the water's half extents drawn, and the
/// forces' drawn length.
const DEEPEST: f64 = 0.6;
const SEA: [f64; 2] = [3.0, 1.8];
const FORCE: f64 = 0.9;
/// The heels over which each curve's tangent at upright is drawn.
const TANGENT: f64 = 25.0;

/// Everything computed once.
struct Scenes {
    hull: Body,
    weight: f64,
    centres: Vec<Point>,
    shown: (Unit<Motor>, gax::pga3d::Forque),
    couples: Vec<Vec<gax::pga3d::Forque>>,
    slopes: Vec<f64>,
    releases: [Vec<Frame>; 2],
    cube: Body,
    cube_arms: Vec<Vec<f64>>,
    rests: Vec<(f64, Unit<Motor>)>,
}

fn scenes() -> Scenes {
    let (hull, weight, centres) = ship();
    let triangles = hull.triangles();
    let (couples, stiffnesses) = righting_curves(&triangles, weight, &centres);
    let turn = heeled(SHOWN_HEEL);
    let height = afloat(&triangles, turn, weight, 0.0, FLUID_WEIGHT, NEWTON_STEPS);
    let (push, _) = buoyancy(&triangles, turn << surface_at(height), FLUID_WEIGHT);
    let releases = RELEASED.map(|load| released(&triangles, weight, centres[load]));
    let cube = cuboid([1.0; 3]);
    let cube_triangles = cube.triangles();
    let densities = densities();
    let couples_map = cube_couples(&cube_triangles, &densities);
    let cube_arms = couples_map
        .iter()
        .zip(&densities)
        .map(|(row, d)| row.iter().map(|c| arm(*c, FLUID_WEIGHT * d)).collect())
        .collect();
    let rests = SHOWN_DENSITIES
        .iter()
        .map(|d| {
            let row = cube_couples(&cube_triangles, &[*d]).remove(0);
            let heel = stable_heel(&row);
            (heel, resting(&cube_triangles, *d, heel))
        })
        .collect();
    Scenes {
        shown: (placement(turn, height), push),
        slopes: stiffnesses
            .iter()
            .map(|k| heel_stiffness(*k) / weight)
            .collect(),
        hull,
        weight,
        centres,
        couples,
        releases,
        cube,
        cube_arms,
        rests,
    }
}

/// The light of each load, low to high.
fn load_light(load: usize) -> Light {
    colormap::viridis(0.15 + 0.75 * load as f32 / (CENTRE_HEIGHTS.len() - 1) as f32)
}

/// The lights of the weight and of the buoyancy.
fn weight_light() -> Light {
    palette::ORANGE
}
fn buoyancy_light() -> Light {
    palette::SKY
}

/// The body placed in the world: its triangles lit, dry pale and wet darker with depth.
fn draw_body(s: &mut Scene3, body: &Body, placed: Unit<Motor>, opacity: f32) {
    for f in &body.faces {
        let [a, b, c] = f.map(|v| placed >> body.vertices[v]);
        let middle = (a + b + c) * (1.0 / 3.0);
        let depth = -(water() & middle).s() / middle.e123();
        let light = if depth > 0.0 {
            palette::BLUE
                .faded(0.25)
                .mix_light(palette::BLUE.faded(0.08), (depth / DEEPEST).min(1.0) as f32)
        } else {
            palette::INK.faded(0.3)
        };
        let lit = s.lit(a, b, c, light);
        s.tri(a, b, c, lit, opacity);
    }
}

/// The water around the body: a translucent plate.
fn draw_water(s: &mut Scene3, [x, y]: [f64; 2]) {
    let corner = |sx: f64, sy: f64| Point::xyz(sx * x, sy * y, 0.0);
    let (a, b, c, d) = (
        corner(-1.0, -1.0),
        corner(1.0, -1.0),
        corner(1.0, 1.0),
        corner(-1.0, 1.0),
    );
    s.quad(a, b, c, d, palette::SKY.faded(0.12), 0.3);
    s.polyline(&[a, b, c, d, a], 1.0, palette::SKY.faded(0.5));
}

/// The weight down from the centre of gravity, and the buoyancy up along its line, from where
/// the line meets a plane below the water, each drawn `FORCE` long.
fn draw_forces(
    s: &mut Scene3,
    placed: Unit<Motor>,
    push: gax::pga3d::Forque,
    centre: Point,
    below: f64,
) {
    let centre = placed >> centre;
    let down = Point::direction(0.0, 0.0, -FORCE);
    s.arrow(centre, down, 2.0, 8.0, weight_light());
    s.dot(centre, Marker::Dot, 7.0, weight_light().whitened(0.3));
    let line = placed >> push;
    let foot = (line ^ Plane::new(0.0, 0.0, 1.0, -below)).position();
    s.arrow(
        foot,
        Point::direction(0.0, 0.0, FORCE),
        2.0,
        8.0,
        buoyancy_light(),
    );
}

/// The heeled hull in the water, in space.
fn draw_ship(c: &mut Canvas, rect: Rect, s: &Scenes) {
    let cam = Camera::parallel(
        rect,
        0.5 * rect.height() / 2.0,
        (-55f32).to_radians(),
        18f32.to_radians(),
    );
    let (placed, push) = s.shown;
    panel3(c, rect, cam, |scene| {
        draw_body(scene, &s.hull, placed, 0.97);
        draw_water(scene, SEA);
        draw_forces(scene, placed, push, s.centres[SHOWN_LOAD], -0.6);
    });
}

/// The released hulls at a frame, seen along the axis of heel.
fn draw_release(c: &mut Canvas, rect: Rect, s: &Scenes, frame: usize) {
    for (k, frames) in s.releases.iter().enumerate() {
        let panel = rect.column(k, 2);
        let cam = Camera::parallel(panel, 0.5 * panel.height() / 1.3, 0.0, 4f32.to_radians());
        let f = frames[frame];
        panel3(c, panel, cam, |scene| {
            draw_body(scene, &s.hull, f.placed, 0.97);
            draw_forces(scene, f.placed, f.push, s.centres[RELEASED[k]], -0.9);
        });
        // The waterline, across the panel.
        let ends = [Point::xyz(0.0, -SEA[1], 0.0), Point::xyz(0.0, SEA[1], 0.0)];
        if let (Some(a), Some(b)) = (cam.px(ends[0]), cam.px(ends[1])) {
            c.clip(panel);
            c.line(a, b, 1.4, palette::SKY.faded(0.7));
            c.unclip();
        }
        let label = format!(
            "{} LOAD: {:.0} DEG",
            ["LOW", "HIGH"][k],
            f.roll.to_degrees()
        );
        let at = panel.bottom_middle() - Point2::direction(0.0, 2.0);
        c.text(&label, at, 9.0, load_light(RELEASED[k]), Align::Center);
    }
}

/// The righting arm of each load over the heels, the tangent at upright that the stiffness
/// gives, and the released hulls' heels so far.
fn draw_curves(c: &mut Canvas, rect: Rect, s: &Scenes, frame: usize) {
    let heels = heels();
    let arms: Vec<Vec<f64>> = s
        .couples
        .iter()
        .map(|row| row.iter().map(|x| arm(*x, s.weight)).collect())
        .collect();
    let (lo, hi) = arms
        .iter()
        .flatten()
        .fold((0.0f64, 0.0f64), |(a, b), x| (a.min(*x), b.max(*x)));
    let ax = Axes::new(rect, [0.0, 180.0], [(lo * 1.1) as f32, (hi * 1.1) as f32]);
    ax.line(
        c,
        Point2::xy(0.0, 0.0),
        Point2::xy(180.0, 0.0),
        1.0,
        palette::GRID,
    );
    for load in 0..CENTRE_HEIGHTS.len() {
        let curve: Vec<Point2> = heels
            .iter()
            .zip(&arms)
            .map(|(h, a)| Point2::xy(*h as f32, a[load] as f32))
            .collect();
        ax.polyline(c, &curve, 1.4, load_light(load));
        let rise = (s.slopes[load] * TANGENT.to_radians()) as f32;
        ax.dashed(
            c,
            &[Point2::xy(0.0, 0.0), Point2::xy(TANGENT as f32, rise)],
            1.0,
            4.0,
            load_light(load).faded(0.6),
        );
    }
    for (k, frames) in s.releases.iter().enumerate() {
        let load = RELEASED[k];
        let so_far: Vec<Point2> = frames[..=frame]
            .iter()
            .map(|f| {
                // The heel within a half turn either way; the arm is odd in it.
                let heel = (f.roll.to_degrees() + 180.0).rem_euclid(360.0) - 180.0;
                let at = (heel.abs() / 2.0).round() as usize;
                let a = arms[at.min(heels.len() - 1)][load] * heel.signum();
                Point2::xy(heel.abs() as f32, a as f32)
            })
            .collect();
        ax.scatter(
            c,
            &so_far[so_far.len() - 1..],
            Marker::Dot,
            8.0,
            load_light(load).whitened(0.4),
        );
    }
    ax.frame(
        c,
        "RIGHTING ARMS, LOW TO HIGH LOAD",
        "HEEL (DEG)",
        "ARM (M)",
    );
}

/// The cube's righting couple over density and heel, pushing back upright warm and over cool,
/// its stable rests, and cubes of five densities at rest above.
fn draw_cubes(c: &mut Canvas, rect: Rect, s: &Scenes) {
    let densities = densities();
    let heels = cube_heels();
    let map = rect.part(0.0, 0.4, 1.0, 1.0);
    let ax = Axes::new(
        map,
        [densities[0] as f32, densities[densities.len() - 1] as f32],
        [heels[0] as f32, heels[heels.len() - 1] as f32],
    );
    let largest = s
        .cube_arms
        .iter()
        .flatten()
        .map(|a| a.abs())
        .fold(0.0, f64::max);
    ax.image(c, 1, |q| {
        let [d, h] = q.to_euclidean();
        let i = ((f64::from(d) - densities[0]) / (densities[1] - densities[0])).round() as usize;
        let j = ((f64::from(h) - heels[0]) / (heels[1] - heels[0])).round() as usize;
        let arm = *s.cube_arms.get(i)?.get(j)?;
        // Pushing back upright cool, over warm, brighter the stronger.
        let level = (arm.abs() / largest) as f32;
        let sense = if arm > 0.0 {
            palette::SKY
        } else {
            palette::ORANGE
        };
        Some(palette::BOTTOM.mix_light(sense.faded(0.6), level.powf(0.6)))
    });
    let rests: Vec<Point2> = SHOWN_DENSITIES
        .iter()
        .zip(&s.rests)
        .map(|(d, (h, _))| Point2::xy(*d as f32, *h as f32))
        .collect();
    ax.scatter(c, &rests, Marker::Ring, 9.0, palette::INK);
    ax.frame(c, "", "DENSITY", "HEEL (DEG)");
    // The cubes at rest, seen along the axis of heel, each over its density.
    let strip = rect.part(0.0, 0.0, 1.0, 0.3);
    let across = [densities[0] as f32, densities[densities.len() - 1] as f32];
    let row = Axes::new(strip, across, [-1.0, 1.0]);
    for (d, (_, placed)) in SHOWN_DENSITIES.iter().zip(&s.rests) {
        let corners: Vec<Point2> = [0, 1, 3, 2]
            .iter()
            .map(|k| {
                let p = *placed >> s.cube.vertices[4 + k];
                let [_, y, z] = p.to_euclidean();
                Point2::xy(*d as f32 + 0.06 * y as f32, z as f32 / 1.4)
            })
            .collect();
        row.fill(c, &corners, palette::INK.faded(0.3), 0.9);
        row.polyline(
            c,
            &[corners.clone(), vec![corners[0]]].concat(),
            1.0,
            palette::INK,
        );
        let span = 0.04;
        row.line(
            c,
            Point2::xy(*d as f32 - span, 0.0),
            Point2::xy(*d as f32 + span, 0.0),
            1.2,
            palette::SKY,
        );
    }
}

fn draw(c: &mut Canvas, s: &Scenes, t: f32) {
    backdrop(c);
    let frame = ((t / SECONDS).rem_euclid(1.0) * FRAMES as f32) as usize % FRAMES;
    let screen = c.rect();
    let (w, h) = (screen.width(), screen.height());
    let body = Rect::new(0.0, h * 0.13, w, h);
    let label = |c: &mut Canvas, rect: Rect, text: &str| {
        let at = rect.top_middle() - Point2::direction(0.0, 4.0);
        c.text(text, at, 9.0, palette::INK, Align::Center);
    };
    let ship = body.part(0.0, 0.0, 0.5, 0.5).inset(6.0, 16.0, 6.0, 2.0);
    draw_ship(c, ship, s);
    label(c, ship, "HEELED 35 DEG: BUOYANCY UP, WEIGHT DOWN");
    let release = body.part(0.5, 0.0, 1.0, 0.5).inset(6.0, 16.0, 6.0, 14.0);
    draw_release(c, release, s, frame);
    label(c, release, "RELEASED FROM 30 DEG: RIGHTED, OR CAPSIZED");
    draw_curves(
        c,
        body.part(0.0, 0.5, 0.5, 1.0).inset(46.0, 24.0, 12.0, 30.0),
        s,
        frame,
    );
    let cubes = body.part(0.5, 0.5, 1.0, 1.0).inset(46.0, 22.0, 12.0, 30.0);
    draw_cubes(c, cubes, s);
    label(c, cubes, "A CUBE: FACE UP ONLY OUTSIDE (3 -+ SQRT 3) / 6");
    caption(
        c,
        "FLOATING: BUOYANCY AS A SUM OF PRESSURE LINES",
        "EACH WET PATCH PUSHES ALONG A LINE; THE SUM IS ONE FORQUE, AND ITS CHANGE UNDER A TILT THE STIFFNESS (PGA3D)",
    );
}

fn main() {
    run_with(Anim::new("floating", SECONDS).size(960, 540), scenes, draw);
}
