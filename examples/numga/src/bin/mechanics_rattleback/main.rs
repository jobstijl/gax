//! numga's `mechanics/rattleback`: a rattleback, a body with an ellipsoidal underside rolling
//! without slipping on a table, which turns rocking into spin, and spin of one sense into
//! rocking and back (PGA3D).
//!
//! Its points and directions are antivectors, and its underside a map from planes to
//! directions, the support map (`Direction <- OriginPlane`): the table's plane through the
//! shell's centre, read in the body, goes to the direction from that centre to the point that
//! touches the table, `−shape(normal) / height`, the height the pairing `normal & shape(normal)`
//! under its square root. The body turns about its centre of mass, the origin, and the shell's
//! centre sits off it, so the touching point lies at `centre + support` from the centre of
//! mass. Rolling holds that point still: a spin of the body about its centre of mass (a `Spin`,
//! a line through the origin) becomes the rate that turns it about the contact, the spin carried
//! by the translation to the contact, a map `Twist <- Spin`. Its adjugate, `Moment <- Forque`,
//! pulls forques back onto spins, and does so without the contact's reaction, which does no work
//! under any rolling motion, so one solve gives the body's angular acceleration with the reaction
//! eliminated. As the body rocks, the contact moves across the shell; differentiating the
//! rolling map adds to the acceleration even at a fixed spin, and stays in the balance solved.
//! The steps are fourth-order Runge–Kutta–Munthe-Kaas, `dexpinv` the shared Lie steppers'.
//!
//! The body is a wooden boat, its underside painted, with two cuts across opposite corners of
//! its deck and a steel ball pressed into each side, every part an intersection of quadrics;
//! the same quadrics give its mass (a grid of points weighted by the density of the part each
//! lies in) and are ray traced (numga's WGSL tracer, here on the CPU).
//!
//! The animation shows both launches: tipped from rest, it rocks into spin; spun the
//! unfavourable way, it rocks, stops and spins the other way; and their spins about the
//! vertical against time.

mod rattleback;
#[cfg(test)]
mod tests;
#[path = "../../shared/quadrics_tracing.rs"]
mod tracing;

use gax::pga3d::{Plane, Point};
use gax_colour::Light;
use gax_numga_examples::{
    Anim, Axes, Camera, Canvas, Lens, Marker, Point2, Point3, Rect, backdrop, caption, label_above,
    palette, run_with,
};
use rattleback::*;
use tracing::{CLEARANCE, FAR, Quadric, region_hit};

/// The loop: numga's 250 steps of 40 ms.
const SECONDS: f32 = 10.0;
/// The view: raised 25 degrees, looking along `+y`; the room around the body, every way it may
/// come to face as it spins, for it to wander as it rolls; and the height it looks at.
const ELEVATION: f32 = 25f32.to_radians();
const AZIMUTH: f32 = -core::f32::consts::FRAC_PI_2;
const DRIFT: f64 = 0.3;
const LOOK_AT: f32 = 0.7;

/// A surface's finish: the colour of the surface a quadric bounds (sRGB), its glint, and how
/// much of what lies around it it mirrors.
#[derive(Clone, Copy)]
struct Finish {
    colour: [f32; 3],
    gloss: f64,
    mirroring: f64,
}

/// The painted underside, the bare wood of the deck and the cuts, and the steel.
const PAINT: Finish = Finish {
    colour: [0.19, 0.52, 0.62],
    gloss: 0.08,
    mirroring: 0.0,
};
const DECK: Finish = Finish {
    colour: [0.91, 0.81, 0.60],
    gloss: 0.0,
    mirroring: 0.0,
};
const CUT: Finish = Finish {
    colour: [0.80, 0.63, 0.42],
    gloss: 0.0,
    mirroring: 0.0,
};
/// The steel mirrors less than numga's, all of it: its sky is the dark backdrop, and a quarter
/// of its own lit grey keeps it steel.
const STEEL: Finish = Finish {
    colour: [0.68, 0.68, 0.70],
    gloss: 0.0,
    mirroring: 0.75,
};
/// The light that falls everywhere, the glint's sharpness, the intensity of the body's lights.
const LOOK: tracing::Look = tracing::Look {
    ambient: 0.38,
    checker: 0.0,
};
const SHININESS: i32 = 32;
const INTENSITY: f32 = 0.9;
/// The ground: tiles of warm grey stone, each a little lighter or darker than its neighbours,
/// honed smooth enough to mirror a little, the more the more glancing the view; how much of the
/// lamp's light the body's shadow takes; the tiles' size, and the joints' width and darkness;
/// the stone's intensity, low under the navy look.
const STONE: [f32; 3] = [0.47, 0.44, 0.40];
const TILE_VARIATION: f64 = 0.06;
const STONE_REFLECTANCE: f64 = 0.04;
const SHADOW: f64 = 0.22;
const TILE_SIZE: f64 = 2.0;
const JOINT_WIDTH: f64 = 0.02;
const JOINT_DARKNESS: f64 = 0.45;
const STONE_INTENSITY: f32 = 0.35;
/// The lamp seen in a mirror: how close to its direction, and how bright.
const LAMP_SIZE: f64 = 0.995;

/// The lamp's direction, at unit length.
fn lamp() -> Point {
    let toward = Point::direction(-0.3, -0.6, 0.85);
    toward * (1.0 / gax::IdealNorm::ideal_norm(toward))
}

/// The parts placed: the wood and each ball, each an intersection of quadrics with a finish per
/// quadric.
struct Parts(Vec<(Vec<Quadric>, Vec<Finish>)>);

impl Parts {
    fn of(body: &Body) -> Parts {
        let wood = body.wood();
        // The underside painted, the deck and the cuts bare wood; the pockets lie under the
        // balls.
        let finishes = (0..wood.len())
            .map(|k| [PAINT, DECK][..].get(k).copied().unwrap_or(CUT))
            .collect();
        let mut parts = vec![(wood, finishes)];
        parts.extend(body.balls.iter().map(|b| (vec![*b], vec![STEEL])));
        Parts(parts)
    }

    /// The first surface a ray meets past `nearest`: its distance, tangent plane and finish.
    fn first_hit(
        &self,
        start: Point,
        direction: Point,
        nearest: f64,
    ) -> Option<(f64, Plane, Finish)> {
        let mut hit = None;
        let mut bound = FAR;
        for (quadrics, finishes) in &self.0 {
            if let Some((d, tangent, k)) = region_hit(quadrics, start, direction, nearest, bound) {
                bound = d;
                hit = Some((d, tangent, finishes[k]));
            }
        }
        hit
    }

    /// Whether the body stands between a point and the lamp.
    fn shadowed(&self, point: Point) -> bool {
        self.first_hit(point, lamp(), CLEARANCE).is_some()
    }
}

/// A tile's tone: a hash of its cell, from 0 to 1 (numga's `fract(sin(...))`, here integer
/// arithmetic).
fn hash(i: i64, j: i64) -> f64 {
    let mut h = (i.wrapping_mul(374_761_393) ^ j.wrapping_mul(668_265_263)) as u64;
    h = (h ^ (h >> 13)).wrapping_mul(1_274_126_177);
    ((h ^ (h >> 16)) & 0xffff) as f64 / 65_536.0
}

/// A tangent plane at unit weight, facing a viewer looking along `direction`.
fn facing(tangent: Plane, direction: Point) -> Plane {
    let sign = if (tangent & direction).s() > 0.0 {
        -1.0
    } else {
        1.0
    };
    (tangent * sign).normalized()
}

/// A surface lit by the lamp, with its finish's glint, unless the body stands between them.
fn lit_or_shadowed(
    parts: &Parts,
    point: Point,
    tangent: Plane,
    finish: Finish,
    direction: Point,
) -> Light {
    let normal = facing(tangent, direction);
    let share = if parts.shadowed(point) { 0.0 } else { 1.0 };
    let light = (normal & lamp()).s().max(0.0);
    // The view ray reflected in the tangent plane, against the lamp.
    let glint = (-((normal >> direction).dual() & lamp()).s())
        .max(0.0)
        .powi(SHININESS);
    let [r, g, b] = finish.colour;
    LOOK.surface(Light::from_srgb(r, g, b, INTENSITY), 0.0, light * share)
        + tracing::glint(finish.gloss * share * glint * f64::from(u8::from(light > 0.0)))
}

/// The ground where a ray meets it, tiled and darker where the body stands between it and the
/// lamp; the sky, with the lamp in it, where the ray rises.
fn surroundings(parts: &Parts, start: Point, direction: Point) -> Light {
    let [x, y, z] = mirrors();
    if (z & direction).s() >= 0.0 {
        let toward_lamp = -(direction.dual() & lamp()).s();
        let lamp_disc = ((toward_lamp - LAMP_SIZE) / (1.0 - LAMP_SIZE)).clamp(0.0, 1.0);
        return tracing::sky((z & direction).s()) + tracing::glint(3.0 * lamp_disc);
    }
    let to_ground = -(z & start).s() / (z & direction).s();
    let on_ground = start + direction * to_ground;
    let spot = [
        (x & on_ground).s() / TILE_SIZE,
        (y & on_ground).s() / TILE_SIZE,
    ];
    let off_joint = spot
        .map(|s| (s - s.round()).abs())
        .iter()
        .fold(f64::MAX, |a, b| a.min(*b))
        * TILE_SIZE;
    let joint = (1.0 - off_joint / JOINT_WIDTH).clamp(0.0, 1.0);
    let tone =
        1.0 + TILE_VARIATION * (2.0 * hash(spot[0].floor() as i64, spot[1].floor() as i64) - 1.0);
    let shaded = if parts.shadowed(on_ground) {
        1.0 - SHADOW
    } else {
        1.0
    };
    let [r, g, b] = STONE;
    Light::from_srgb(r, g, b, STONE_INTENSITY)
        .faded((tone * (1.0 - JOINT_DARKNESS * joint) * shaded) as f32)
}

/// What a ray sees, without looking into mirrors.
fn seen(parts: &Parts, start: Point, direction: Point) -> Light {
    match parts.first_hit(start, direction, CLEARANCE) {
        Some((d, tangent, finish)) => {
            lit_or_shadowed(parts, start + direction * d, tangent, finish, direction)
        }
        None => surroundings(parts, start, direction),
    }
}

/// What the ray from a start along a direction sees: the body, its steel mirroring what lies
/// around it; the honed stone, mirroring a little; or nothing, the backdrop, above the horizon.
fn sample(parts: &Parts, start: Point, direction: Point) -> Option<Light> {
    let z = mirrors()[2];
    let hit = parts.first_hit(start, direction, 0.0);
    let to_ground = -(z & start).s() / (z & direction).s();
    let ground_first = to_ground > 0.0 && hit.is_none_or(|(d, _, _)| d > to_ground);
    if ground_first {
        let glancing = (1.0 - (z & direction).s().abs()).powi(5);
        let reflectance = STONE_REFLECTANCE + (1.0 - STONE_REFLECTANCE) * glancing;
        let on_ground = start + direction * to_ground;
        let mirrored = seen(parts, on_ground, z >> direction);
        return Some(surroundings(parts, start, direction).mix_light(mirrored, reflectance as f32));
    }
    let (d, tangent, finish) = hit?;
    let point = start + direction * d;
    let lit = lit_or_shadowed(parts, point, tangent, finish, direction);
    if finish.mirroring == 0.0 {
        return Some(lit);
    }
    // A mirroring surface shows what the ray, reflected in its tangent plane, goes on to see,
    // tinted by its own colour as a metal is, and less tinted the more glancing the ray.
    let normal = tangent.normalized();
    let glancing = (1.0 - (normal & direction).s().abs()).powi(5);
    let tint = f64::from(finish.colour.iter().sum::<f32>() / 3.0);
    let reflectance = tint + (1.0 - tint) * glancing;
    let mirrored = seen(parts, point, normal >> direction).faded(reflectance as f32);
    Some(lit.mix_light(mirrored, finish.mirroring as f32))
}

/// The body placed by a motor, ray traced from the fixed parallel camera into a panel.
fn trace(c: &mut Canvas, rect: Rect, body: &Body, motor: M, reach: f32) {
    // Half the view's height: the reach across the panel, or the reach seen end-on from the
    // view's elevation, with the deck's height, whichever is more.
    let end_on = reach / (1.0 + DRIFT as f32) * gax_numga_examples::signal::wave(ELEVATION) + 0.6;
    let half = (reach * rect.height() / rect.width()).max(end_on);
    let target = Point3::xyz(0.0, 0.0, LOOK_AT);
    let cam = Camera::orbit(rect, target, 40.0, AZIMUTH, ELEVATION, Lens::Parallel(half));
    let parts = Parts::of(&body.placed(motor));
    let to64 = |p: Point3| p.map_coefs(f64::from);
    c.clip(rect);
    c.shade(1, |q| {
        let (origin, direction) = cam.ray(q);
        let direction = to64(direction);
        let direction = direction * (1.0 / gax::IdealNorm::ideal_norm(direction));
        sample(&parts, to64(origin).unitized(), direction)
    });
    c.unclip();
}

/// Both launches' spins about the vertical against time, the step shown marked.
fn draw_spins(c: &mut Canvas, rect: Rect, s: &Scenes, now: usize) {
    let duration = (N_STEPS as f64 * TIME_STEP) as f32;
    let ax = Axes::new(rect, [0.0, duration], [-2.0, 2.0]);
    ax.line(
        c,
        Point2::xy(0.0, 0.0),
        Point2::xy(duration, 0.0),
        1.0,
        palette::GRID.faded(2.0),
    );
    for (k, light) in [palette::SKY, palette::ORANGE].iter().enumerate() {
        let curve: Vec<Point2> = s.spins[k]
            .iter()
            .enumerate()
            .map(|(j, v)| Point2::xy((j as f64 * TIME_STEP) as f32, *v as f32))
            .collect();
        ax.polyline(c, &curve, 1.6, *light);
        ax.scatter(c, &[curve[now]], Marker::Dot, 6.0, light.whitened(0.3));
    }
    ax.frame(c, "VERTICAL SPIN (RAD/S)", "TIME (S)", "");
    ax.legend(c, &[("TIPPED", palette::SKY), ("SPUN", palette::ORANGE)]);
}

struct Scenes {
    rattleback: Rattleback,
    /// The view's reach across from the body's middle: as far as the points its mass is read
    /// from lie from its vertical axis, so that it fits facing any way, with room to wander.
    reach: f32,
    launches: [Launch; 2],
    spins: [Vec<f64>; 2],
}

fn scenes() -> Scenes {
    let rattleback = rattleback();
    let launches = launches(&rattleback, TIME_STEP, N_STEPS);
    let spins = launches.each_ref().map(Launch::spin_about_vertical);
    let up = mirrors()[2];
    let widest = rattleback
        .samples
        .iter()
        .map(|p| (*p & (origin() & Point::from(up.dual().cast::<gax::pga3d::Direction>()))).norm())
        .fold(0.0, f64::max);
    Scenes {
        reach: (widest * (1.0 + DRIFT)) as f32,
        rattleback,
        launches,
        spins,
    }
}

fn draw(c: &mut Canvas, s: &Scenes, t: f32) {
    backdrop(c);
    let now = gax_numga_examples::signal::frame((t / SECONDS).rem_euclid(1.0), N_STEPS + 1);
    let screen = c.rect();
    let (w, h) = (screen.width(), screen.height());
    let names = [
        "TIPPED FROM REST: ROCKING INTO SPIN",
        "SPUN THE WRONG WAY: IT REVERSES",
    ];
    let top = h * 0.17;
    let panel = (h - top) / 2.0;
    for (k, name) in names.iter().enumerate() {
        let y0 = top + panel * k as f32;
        let rect = Rect::new(0.0, y0 + 16.0, w * 0.62, y0 + panel).inset(6.0, 0.0, 6.0, 4.0);
        trace(
            c,
            rect,
            &s.rattleback.body,
            s.launches[k].motors[now],
            s.reach,
        );
        label_above(c, rect, name, (h / 45.0).min(12.0), 4.0);
    }
    let chart = Rect::new(w * 0.62, top, w, h).inset(50.0, 30.0, 16.0, 44.0);
    draw_spins(c, chart, s, now);
    let time = now as f64 * TIME_STEP;
    caption(
        c,
        "RATTLEBACK: ROCKING TURNED INTO SPIN",
        &format!(
            "ROLLING Twist <- Spin; ITS ADJUGATE Moment <- Forque DROPS THE CONTACT'S REACTION (PGA3D). T = {time:.1} S"
        ),
    );
}

fn main() {
    run_with(
        Anim::new("rattleback", SECONDS).size(480, 270).scale(2),
        scenes,
        draw,
    );
}
