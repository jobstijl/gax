//! numga's `mechanics/tippe_top`: a hollow tippe top, a shell carved by a turned cavity and
//! joined to a stem with a ball at its end, released with equal and opposite spins (PGA3D).
//!
//! The same Boolean solid of quadrics supplies the visible surfaces and the mass: a grid of
//! samples, each kept where it lies inside the shell and outside the cavity, or inside the stem
//! or the tip, gives the mass (a point at the centre of mass, weighted by the mass) and the
//! inertia about the shell's centre, the sum of each sample's `p & [p, ·]`. The cavity turns
//! the mass distribution relative to the outer shell, so the top is not symmetric about its
//! axis. It steps as the spinning top does: a free rigid step under gravity, then the contacts
//! at the shell and the tip corrected part by part (`shared/mechanics_contact.rs`, numga's
//! spinning-top core). Sliding friction at the shell or the tip couples spinning to rocking:
//! the top spun one way rises onto its stem and comes to rest on its tip, inverted; spun the
//! other way it stays upright, as the turned cavity makes the two spins differ.
//!
//! The animation ray traces both tops over the 80 s run, the shell chequered in blocks fixed to
//! the body so that its spin shows, each view following its top's travel across the table, and
//! plots how far each top's axis points up.

#[cfg(test)]
mod tests;
mod tippe;

use gax::pga3d::{Plane, Point};
use gax_colour::Light;
use gax_numga_examples::shared::mechanics_contact as contact;
use gax_numga_examples::shared::mechanics_lie::rigid::{M, R};
use gax_numga_examples::shared::quadrics_tracing as tracing;
use gax_numga_examples::{
    Anim, Axes, Camera, Canvas, Lens, Marker, Point2, Point3, Rect, backdrop, caption, label_above,
    palette, run_with,
};
use tippe::*;
use tracing::{CLEARANCE, FAR, Quadric, region_hit};

/// The loop: every recorded state, 0.16 s of the run, shown for 40 ms.
const SECONDS: f32 = 501.0 * 0.04;
/// The view: raised 0.24 rad, turned as numga's, at its height above the table and half width,
/// following each top's centre of mass averaged over this many states either side.
const ELEVATION: f32 = 0.24;
const AZIMUTH: f32 = -0.65 - core::f32::consts::FRAC_PI_2;
const VIEW_HEIGHT: f32 = 1.1;
const HALF_EXTENT: f32 = 1.6;
const SMOOTHING: usize = 16;
/// The colours (sRGB): the shell's light and dark blocks, the stem and the tip, the table and
/// its dark squares; the light that falls everywhere; the table's intensity, low under the
/// navy look; the table's squares.
const LIGHT_SHELL: [f32; 3] = [0.93, 0.62, 0.42];
const SHELL: [f32; 3] = [0.78, 0.30, 0.16];
const STEM: [f32; 3] = [0.84, 0.65, 0.32];
const FLOOR: [f32; 3] = [0.80, 0.79, 0.75];
const DARK_SQUARE: [f32; 3] = [0.62, 0.61, 0.57];
const LOOK: tracing::Look = tracing::Look {
    ambient: 0.35,
    checker: 0.0,
};
const INTENSITY: f32 = 0.9;
const FLOOR_INTENSITY: f32 = 0.25;
const SQUARE: f64 = 1.0;

/// The points on a plane's negative side, as a quadric.
fn half_space(plane: Plane) -> Quadric {
    (plane * (w() & Point::slot()) + w() * (plane & Point::slot())) * 0.5
}

/// The solids drawn, each an intersection of quadrics with its colour: the shell carved by the
/// cavity, chequered in blocks by the body's halves either side of a plane through its axis and
/// above and below its equator; the stem and the tip.
fn solids(body: &Body, motor: M) -> Vec<(Vec<Quadric>, [f32; 3])> {
    let [x, _, z] = axes();
    let (one_side, other_side) = (half_space(x), half_space(-x));
    let (lower, upper) = (half_space(z), half_space(-z));
    let shell = |side, height| vec![body.shell, -body.cavity, side, height];
    let parts = vec![
        (shell(one_side, lower), LIGHT_SHELL),
        (shell(other_side, upper), LIGHT_SHELL),
        (shell(one_side, upper), SHELL),
        (shell(other_side, lower), SHELL),
        (vec![body.stem], STEM),
        (vec![body.tip], STEM),
    ];
    let place = |q: Quadric| motor >> q.of(motor << Point::slot());
    parts
        .into_iter()
        .map(|(qs, colour)| (qs.into_iter().map(place).collect(), colour))
        .collect()
}

/// The first surface a ray meets: its distance, tangent plane and colour.
fn first_hit(
    solids: &[(Vec<Quadric>, [f32; 3])],
    start: Point,
    direction: Point,
) -> Option<(f64, Plane, [f32; 3])> {
    let mut hit = None;
    let mut bound = FAR;
    for (quadrics, colour) in solids {
        if let Some((d, tangent, _)) = region_hit(quadrics, start, direction, CLEARANCE, bound) {
            bound = d;
            hit = Some((d, tangent, *colour));
        }
    }
    hit
}

/// What a ray sees: the top, lit by the lamp; or the table's checkerboard, shadowed where the
/// top stands between it and the lamp; or nothing (the backdrop) above the horizon.
fn sample(
    solids: &[(Vec<Quadric>, [f32; 3])],
    lamp: Point,
    start: Point,
    direction: Point,
) -> Option<Light> {
    let [x, y, z] = axes();
    let to_table = -(z & start).s() / (z & direction).s();
    let hit = first_hit(solids, start, direction);
    if to_table > 0.0 && hit.is_none_or(|(d, _, _)| d > to_table) {
        let on_table = start + direction * to_table;
        let square = |p: Plane| ((p & on_table).s() / SQUARE).floor();
        let dark = (square(x) + square(y)).rem_euclid(2.0) == 1.0;
        let [r, g, b] = if dark { DARK_SQUARE } else { FLOOR };
        let shadowed = first_hit(solids, on_table, lamp).is_some();
        let lit = if shadowed {
            0.0
        } else {
            (z & lamp).s().max(0.0)
        };
        return Some(LOOK.surface(Light::from_srgb(r, g, b, FLOOR_INTENSITY), 0.0, lit));
    }
    let (_, tangent, [r, g, b]) = hit?;
    // The tangent plane at unit weight, facing the viewer.
    let sign = if (tangent & direction).s() > 0.0 {
        -1.0
    } else {
        1.0
    };
    let normal = (tangent * sign).normalized();
    let lit = (normal & lamp).s().max(0.0);
    Some(LOOK.surface(Light::from_srgb(r, g, b, INTENSITY), 0.0, lit))
}

/// A top ray traced into a panel, the view following its travel.
fn trace(c: &mut Canvas, rect: Rect, body: &Body, motor: M, travel: Point2) {
    let [tx, ty] = travel.to_euclidean();
    let target = Point3::xyz(tx, ty, VIEW_HEIGHT);
    let cam = Camera::orbit(
        rect,
        target,
        20.0,
        AZIMUTH,
        ELEVATION,
        Lens::Parallel(HALF_EXTENT),
    );
    // The lamp above the viewer's left shoulder: back toward the viewer, up and to the left.
    let to64 = |p: Point3| p.map_coefs(f64::from);
    let toward = to64(cam.pose >> Point3::direction(0.3, 0.8, -1.0));
    let lamp = toward * (1.0 / gax::IdealNorm::ideal_norm(toward));
    let solids = solids(body, motor);
    c.clip(rect);
    c.shade(1, |q| {
        let (origin, direction) = cam.ray(q);
        let direction = to64(direction);
        let direction = direction * (1.0 / gax::IdealNorm::ideal_norm(direction));
        sample(&solids, lamp, to64(origin).unitized(), direction)
    });
    c.unclip();
}

/// How far each top's axis points up, against time, the state shown marked.
fn draw_history(c: &mut Canvas, rect: Rect, s: &Scenes, now: usize) {
    let ax = Axes::new(rect, [0.0, RUN_TIME as f32], [-1.08, 1.08]);
    ax.line(
        c,
        Point2::xy(0.0, 0.0),
        Point2::xy(RUN_TIME as f32, 0.0),
        1.0,
        palette::GRID.faded(2.0),
    );
    for (k, light) in [palette::ORANGE, palette::SKY].iter().enumerate() {
        let curve: Vec<Point2> = s.vertical[k]
            .iter()
            .enumerate()
            .map(|(j, v)| Point2::xy((j as f64 * FRAME_INTERVAL) as f32, *v as f32))
            .collect();
        ax.polyline(c, &curve, 1.6, *light);
        ax.scatter(c, &[curve[now]], Marker::Dot, 6.0, light.whitened(0.3));
    }
    ax.frame(c, "VERTICAL AXIS", "TIME", "");
}

struct Scenes {
    body: Body,
    motors: [Vec<M>; 2],
    vertical: [Vec<f64>; 2],
    /// Each top's travel across the table, smoothed, at every state.
    travel: [Vec<Point2>; 2],
}

/// The centre of mass's travel across the table at every state, averaged over the states
/// either side, weighted down linearly: a few rocking periods, so that the view glides with the
/// drift and ignores the rocking.
fn travel(body: &Body, motors: &[M]) -> Vec<Point2> {
    let [x, y, _] = axes();
    let centres: Vec<[f64; 2]> = motors
        .iter()
        .map(|m| {
            let c = (*m >> body.mass).unitized();
            [(x & c).s(), (y & c).s()]
        })
        .collect();
    let n = SMOOTHING as isize;
    let total = ((n + 1) * (n + 1)) as f64;
    (0..centres.len() as isize)
        .map(|k| {
            let mut sum = [0.0; 2];
            for d in -n..=n {
                let j = (k + d).clamp(0, centres.len() as isize - 1) as usize;
                let weight = (n + 1 - d.abs()) as f64 / total;
                sum = [
                    sum[0] + weight * centres[j][0],
                    sum[1] + weight * centres[j][1],
                ];
            }
            Point2::xy(sum[0] as f32, sum[1] as f32)
        })
        .collect()
}

fn scenes() -> Scenes {
    let body = body();
    let runs = evolve(&body);
    let motors = runs
        .each_ref()
        .map(|r| r.iter().map(|(m, _): &(M, R)| *m).collect::<Vec<_>>());
    let vertical = motors
        .each_ref()
        .map(|ms| ms.iter().map(|m| tippe::vertical(*m)).collect());
    let travel = motors.each_ref().map(|ms| travel(&body, ms));
    Scenes {
        body,
        motors,
        vertical,
        travel,
    }
}

fn draw(c: &mut Canvas, s: &Scenes, t: f32) {
    backdrop(c);
    let now = gax_numga_examples::signal::frame((t / SECONDS).rem_euclid(1.0), n_states());
    let screen = c.rect();
    let (w, h) = (screen.width(), screen.height());
    let top = h * 0.2;
    let side = (h - top - 8.0).min(w * 0.31);
    let names = ["SPUN CLOCKWISE", "SPUN COUNTERCLOCKWISE"];
    for (k, name) in names.iter().enumerate() {
        let x0 = 8.0 + k as f32 * (side + 8.0);
        let rect = Rect::new(x0, top, x0 + side, top + side);
        trace(c, rect, &s.body, s.motors[k][now], s.travel[k][now]);
        label_above(c, rect, name, (h / 45.0).min(12.0), 5.0);
    }
    let chart = Rect::new(2.0 * side + 24.0, top, w, h).inset(40.0, 6.0, 14.0, 60.0);
    draw_history(c, chart, s, now);
    let time = now as f64 * FRAME_INTERVAL;
    caption(
        c,
        "TIPPE TOP: A HOLLOW TOP SPUN BOTH WAYS",
        &format!(
            "ONE SOLID OF QUADRICS GIVES SURFACES AND MASS; CONTACTS AT SHELL AND TIP (PGA3D). T = {time:.1}"
        ),
    );
}

fn main() {
    run_with(
        Anim::new("tippe_top", SECONDS).size(480, 270).scale(2),
        scenes,
        draw,
    );
}
