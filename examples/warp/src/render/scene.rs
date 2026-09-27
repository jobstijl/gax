//! From the simulation to line instances: shapes placed by interpolated unit motors, bullets,
//! warp-ins, shards and the arena, plus the cameras as gax maps.
//!
//! Shapes are PGA2D points. A shape is drawn at a size by giving its points a weight: the point
//! `(x, y, 1/s)` is `(s x, s y)`, so scaling about the local origin needs no arithmetic on
//! coordinates. Circles, arcs and spirals are points turned by rotation motors. Colours are
//! lights (`light.rs`).

use super::font::{self, Align};
use super::{CameraUniform, LineInstance};
use crate::light::{Light, fade, light, whiten};
use crate::sim::body::{Pose, identity};
use crate::sim::{ARENA, Kind, Phase, World};
use crate::store::Scheme;
use gax::pga2d::{Motor, Point};

/// Colours as lights. One hue per family; the player's is used by nothing else.
pub mod palette {
    use crate::light::{Light, light};
    /// The ship: warm white.
    pub const SHIP: Light = light(1.0, 0.86, 0.62, 3.2);
    /// Shots.
    pub const BULLET: Light = light(1.0, 0.82, 0.5, 3.0);
    /// Drifters: cyan.
    pub const DRIFTER: Light = light(0.15, 0.85, 1.0, 2.6);
    /// Chasers: magenta.
    pub const CHASER: Light = light(1.0, 0.2, 0.62, 2.8);
    /// Motes: violet-pink.
    pub const MOTE: Light = light(0.85, 0.35, 1.0, 2.6);
    /// Singularities: deep violet.
    pub const SINGULARITY: Light = light(0.55, 0.3, 1.0, 1.9);
    /// Evaders: yellow.
    pub const EVADER: Light = light(1.0, 0.86, 0.1, 2.6);
    /// Splitters and their fragments: orange.
    pub const SPLITTER: Light = light(1.0, 0.36, 0.1, 2.7);
    /// Serpents: teal.
    pub const SERPENT: Light = light(0.1, 1.0, 0.72, 2.5);
    /// Wardens: red.
    pub const WARDEN: Light = light(1.0, 0.14, 0.18, 2.8);
    /// Carriers: blue.
    pub const CARRIER: Light = light(0.32, 0.48, 1.0, 2.8);
    /// Shards: lime.
    pub const SHARD: Light = light(0.45, 1.0, 0.3, 2.6);
    /// The arena's border.
    pub const BORDER: Light = light(0.3, 0.5, 1.0, 2.0);
    /// The lattice.
    pub const GRID: Light = light(0.14, 0.24, 0.75, 0.22);
    /// HUD text.
    pub const HUD: Light = light(0.75, 0.85, 1.0, 1.6);
    /// A white-hot core.
    pub const HOT: Light = light(1.0, 0.9, 1.0, 2.2);
    /// An engine flame.
    pub const FLAME: Light = light(1.0, 0.45, 0.15, 3.0);
}

static SCHEME: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// Choose the colour scheme for enemies and shards (a setting).
pub fn set_scheme(s: Scheme) {
    SCHEME.store(s as u8, std::sync::atomic::Ordering::Relaxed);
}

fn scheme() -> Scheme {
    match SCHEME.load(std::sync::atomic::Ordering::Relaxed) {
        1 => Scheme::RedGreen,
        2 => Scheme::BlueYellow,
        _ => Scheme::Standard,
    }
}

/// The light of a family, in the current scheme. Shapes carry the identity of every family;
/// the schemes keep neighbours in the roster apart for colour-blind players (each scheme
/// moves the colours, a family keeps its intensity).
pub fn color(kind: Kind) -> Light {
    let standard = match kind {
        Kind::Drifter => palette::DRIFTER,
        Kind::Chaser => palette::CHASER,
        Kind::Mote => palette::MOTE,
        Kind::Singularity => palette::SINGULARITY,
        Kind::Evader => palette::EVADER,
        Kind::Splitter | Kind::Fragment => palette::SPLITTER,
        Kind::Serpent => palette::SERPENT,
        Kind::Warden => palette::WARDEN,
        Kind::Carrier => palette::CARRIER,
    };
    let [r, g, b] = match (scheme(), kind) {
        (Scheme::Standard, _) => return standard,
        // Blue, orange, yellow, purple and white, after Okabe and Ito; no red against green.
        (Scheme::RedGreen, Kind::Drifter) => [0.35, 0.72, 1.0],
        (Scheme::RedGreen, Kind::Chaser) => [1.0, 0.62, 0.0],
        (Scheme::RedGreen, Kind::Mote) => [0.95, 0.55, 0.85],
        (Scheme::RedGreen, Kind::Singularity) => [0.45, 0.35, 1.0],
        (Scheme::RedGreen, Kind::Evader) => [1.0, 0.95, 0.25],
        (Scheme::RedGreen, Kind::Splitter | Kind::Fragment) => [0.75, 0.45, 1.0],
        (Scheme::RedGreen, Kind::Serpent) => [0.0, 0.8, 0.65],
        (Scheme::RedGreen, Kind::Warden) => [0.85, 0.92, 1.0],
        (Scheme::RedGreen, Kind::Carrier) => [0.15, 0.4, 1.0],
        // Reds against cyans; no blue against green or yellow against violet.
        (Scheme::BlueYellow, Kind::Drifter) => [0.1, 0.9, 1.0],
        (Scheme::BlueYellow, Kind::Chaser) => [1.0, 0.2, 0.62],
        (Scheme::BlueYellow, Kind::Mote) => [1.0, 0.55, 0.75],
        (Scheme::BlueYellow, Kind::Singularity) => [0.55, 0.3, 1.0],
        (Scheme::BlueYellow, Kind::Evader) => [1.0, 0.22, 0.15],
        (Scheme::BlueYellow, Kind::Splitter | Kind::Fragment) => [0.45, 1.0, 0.3],
        (Scheme::BlueYellow, Kind::Serpent) => [0.1, 0.8, 0.8],
        (Scheme::BlueYellow, Kind::Warden) => [0.95, 0.95, 1.0],
        (Scheme::BlueYellow, Kind::Carrier) => [0.3, 0.5, 1.0],
    };
    light(r, g, b, crate::light::intensity(standard))
}

/// The shards' light, in the current scheme.
pub fn shard() -> Light {
    match scheme() {
        Scheme::Standard => palette::SHARD,
        Scheme::RedGreen => light(0.85, 1.0, 1.0, 2.6),
        Scheme::BlueYellow => light(1.0, 0.95, 0.6, 2.6),
    }
}

/// The default stroke: half width, glow radius, glow strength.
pub const THIN: [f32; 4] = [0.032, 0.32, 0.22, 0.0];

/// The point `(x, y)`.
pub const fn pt(x: f32, y: f32) -> Point<(), f32> {
    Point::new(x, y, 1.0)
}

/// The origin of a local frame.
pub const ORIGIN: Point<(), f32> = pt(0.0, 0.0);

/// `p` scaled by `s` about the origin: the same coordinates with weight `w / s`.
pub fn scaled(p: Point<(), f32>, s: f32) -> Point<(), f32> {
    Point::new(p.e20(), p.e01(), p.e12() / s)
}

/// The point at distance `r` from the origin at `angle`: `(r, 0)` turned by a rotation motor.
pub fn polar(r: f32, angle: f32) -> Point<(), f32> {
    Motor::rotation(ORIGIN, angle) >> pt(r, 0.0)
}

/// The segment `a → b`, in the frame of `motor`.
pub fn seg(
    a: Point<(), f32>,
    b: Point<(), f32>,
    color: Light,
    style: [f32; 4],
    motor: Pose,
) -> LineInstance {
    let ([ax, ay], [bx, by]) = (a.to_euclidean(), b.to_euclidean());
    LineInstance {
        ab: [ax, ay, bx, by],
        color: color.into(),
        style,
        motor: motor.into(),
    }
}

/// A closed outline through `pts`, scaled by `size`, in the frame of `m`.
pub fn outline(
    out: &mut Vec<LineInstance>,
    pts: &[Point<(), f32>],
    size: f32,
    color: Light,
    style: [f32; 4],
    m: Pose,
) {
    for (i, &a) in pts.iter().enumerate() {
        let b = pts[(i + 1) % pts.len()];
        out.push(seg(scaled(a, size), scaled(b, size), color, style, m));
    }
}

/// A circle of radius `r` as `n` chords, turned by `phase`: each corner the last one turned.
pub fn circle(
    out: &mut Vec<LineInstance>,
    r: f32,
    n: usize,
    phase: f32,
    color: Light,
    style: [f32; 4],
    m: Pose,
) {
    let step = Motor::rotation(ORIGIN, core::f32::consts::TAU / n as f32);
    let mut a = polar(r, phase);
    for _ in 0..n {
        let b = step >> a;
        out.push(seg(a, b, color, style, m));
        a = b;
    }
}

const DIAMOND: [Point<(), f32>; 4] = [pt(1.0, 0.0), pt(0.0, 1.0), pt(-1.0, 0.0), pt(0.0, -1.0)];
const SQUARE: [Point<(), f32>; 4] = [pt(0.8, 0.8), pt(-0.8, 0.8), pt(-0.8, -0.8), pt(0.8, -0.8)];
const DART: [Point<(), f32>; 3] = [pt(1.0, 0.0), pt(-0.6, 0.7), pt(-0.6, -0.7)];
const BOW: [Point<(), f32>; 3] = [pt(0.0, 0.0), pt(0.95, 0.7), pt(0.95, -0.7)];
const GEM: [Point<(), f32>; 4] = [pt(0.22, 0.0), pt(0.0, 0.16), pt(-0.22, 0.0), pt(0.0, -0.16)];
const SHARP: [Point<(), f32>; 3] = [pt(0.55, 0.0), pt(0.15, 0.22), pt(0.15, -0.22)];
const ARROW: [Point<(), f32>; 4] = [
    pt(1.0, 0.0),
    pt(-0.7, 0.55),
    pt(-0.35, 0.0),
    pt(-0.7, -0.55),
];
const HEAD: [Point<(), f32>; 4] = [pt(1.1, 0.0), pt(-0.3, 0.75), pt(-0.1, 0.0), pt(-0.3, -0.75)];
const TAIL: [Point<(), f32>; 4] = [pt(0.45, 0.0), pt(-0.6, 0.5), pt(-0.35, 0.0), pt(-0.6, -0.5)];
const CARGO: [Point<(), f32>; 3] = [pt(0.14, 0.0), pt(-0.1, 0.1), pt(-0.1, -0.1)];
const HULL: [Point<(), f32>; 4] = [
    pt(0.75, 0.0),
    pt(-0.45, 0.48),
    pt(-0.18, 0.0),
    pt(-0.45, -0.48),
];

/// A regular polygon of `n` corners on the unit circle.
fn regular(n: usize) -> Vec<Point<(), f32>> {
    (0..n)
        .map(|k| polar(1.0, k as f32 * core::f32::consts::TAU / n as f32))
        .collect()
}

/// The shape of a family, drawn at pose `m` (time `t` animates it).
#[allow(clippy::too_many_arguments)]
pub fn draw_enemy(
    out: &mut Vec<LineInstance>,
    kind: Kind,
    m: Pose,
    radius: f32,
    mass: f32,
    flash: f32,
    t: f32,
    bright: f32,
) {
    // A hit flashes towards white, briefly and partly: under fire the colour must stay. A
    // singularity (under fire for seconds) swells instead.
    let swell = flash;
    let flash = if kind == Kind::Singularity {
        0.0
    } else {
        flash
    };
    let c = fade(
        whiten(color(kind), 0.45 * flash),
        (1.0 + 0.45 * flash) * bright,
    );
    match kind {
        Kind::Drifter => {
            outline(out, &DIAMOND, radius, c, THIN, m);
            outline(out, &DIAMOND, radius * 0.45, fade(c, 0.7), THIN, m);
        }
        Kind::Chaser => {
            outline(out, &SQUARE, radius * 0.9, c, THIN, m);
            // A pinwheel inside, turning against the body.
            let inner = m * Motor::rotation(ORIGIN, -3.0 * t);
            for k in 0..4 {
                let tip = polar(radius * 0.7, k as f32 * core::f32::consts::FRAC_PI_2);
                out.push(seg(ORIGIN, tip, fade(c, 0.8), THIN, inner));
            }
        }
        Kind::Mote => outline(out, &DART, radius, c, THIN, m),
        Kind::Evader => {
            // A bow tie along its heading: a triangle and its half-turned twin.
            let half = Motor::rotation(ORIGIN, core::f32::consts::PI);
            outline(out, &BOW, radius, c, THIN, m);
            outline(out, &BOW, radius, c, THIN, m * half);
        }
        Kind::Splitter => {
            outline(out, &SQUARE, radius * 1.06, c, THIN, m);
            // Its three fragments, visible inside.
            for k in 0..3 {
                let r = Motor::rotation(ORIGIN, k as f32 * core::f32::consts::TAU / 3.0);
                outline(out, &SHARP, radius, fade(c, 0.75), THIN, m * r);
            }
        }
        Kind::Fragment => outline(out, &ARROW, radius, c, THIN, m),
        Kind::Serpent => {
            outline(out, &HEAD, radius, c, THIN, m);
            // Eyes: a short stroke and its mirror image across the heading.
            for side in [1.0f32, -1.0] {
                let (a, b) = (pt(0.25, 0.22 * side), pt(0.4, 0.22 * side));
                out.push(seg(a, b, fade(c, 1.4), THIN, m));
            }
        }
        Kind::Warden => {
            // The shield: a bright arc across the front, as chords of a turning radius.
            let shield = [0.06, 0.4, 0.4, 0.0];
            let n = 12;
            let step = Motor::rotation(ORIGIN, 2.6 / n as f32);
            let mut a = polar(1.05 * radius, -1.3);
            for _ in 0..n {
                let b = step >> a;
                out.push(seg(a, b, fade(c, 1.3), shield, m));
                a = b;
            }
            outline(out, &TAIL, radius, c, THIN, m);
        }
        Kind::Carrier => {
            let hex = regular(6);
            outline(out, &hex, radius, c, THIN, m);
            let inner = m * Motor::rotation(ORIGIN, -t * 1.3);
            outline(out, &hex, radius * 0.62, fade(c, 0.7), THIN, inner);
            // Cargo circling inside: each piece a turn about the centre, then out.
            for k in 0..3 {
                let a = t * 2.0 + k as f32 * core::f32::consts::TAU / 3.0;
                let q = Motor::rotation(ORIGIN, a) * Motor::translation(0.35 * radius, 0.0);
                outline(out, &CARGO, 1.0, palette::MOTE, THIN, m * q);
            }
        }
        Kind::Singularity => {
            let pulse = 1.0 + 0.08 * crate::signal::wave(t * 5.0) + 0.02 * mass + 0.06 * swell;
            let style = [0.04, 0.3, 0.3, 0.0];
            circle(out, radius * pulse, 40, t, c, style, m);
            circle(
                out,
                radius * 0.72 * pulse,
                28,
                -t * 1.7,
                fade(c, 0.7),
                THIN,
                m,
            );
            // Spiral arms falling inwards: a direction turned step by step while the radius
            // shrinks (the point's weight grows).
            let turn = Motor::rotation(ORIGIN, 2.4 / 8.0);
            for arm in 0..5 {
                let a0 = arm as f32 * core::f32::consts::TAU / 5.0 + t * 2.2;
                let mut dir = polar(1.0, a0);
                let mut prev = scaled(dir, radius * 1.35);
                for s in 1..9 {
                    let f = s as f32 / 8.0;
                    dir = turn >> dir;
                    let p = scaled(dir, radius * (1.35 - 1.2 * f));
                    out.push(seg(prev, p, fade(c, 0.55 * (1.0 - 0.5 * f)), THIN, m));
                    prev = p;
                }
            }
            // A white-hot core.
            out.push(seg(
                pt(-0.05, 0.0),
                pt(0.05, 0.0),
                fade(palette::HOT, bright),
                [0.1, 0.35, 0.5, 0.0],
                m,
            ));
        }
    }
}

/// The ship.
pub fn draw_ship(out: &mut Vec<LineInstance>, m: Pose, speed: f32, t: f32, alpha: f32) {
    let c = fade(palette::SHIP, alpha);
    let style = [0.04, 0.38, 0.28, 0.0];
    outline(out, &HULL, 1.2, c, style, m);
    for side in [1.0f32, -1.0] {
        out.push(seg(
            pt(0.35, 0.0),
            pt(-0.05, 0.18 * side),
            fade(c, 0.6),
            THIN,
            m,
        ));
    }
    // Engine flame, flickering with speed.
    if speed > 0.5 {
        let flicker = 0.75 + 0.25 * crate::signal::wave(t * 47.0);
        let len = 0.25 + 0.35 * (speed / 11.5).min(1.0) * flicker;
        out.push(seg(
            pt(-0.2, 0.0),
            pt(-0.2 - len, 0.0),
            fade(palette::FLAME, alpha),
            [0.06, 0.35, 0.5, 0.0],
            m,
        ));
    }
}

/// The view of a camera at `cam` showing `half_height` world units above and below its
/// centre, on a target of `size` pixels, as a gax map from world points to clip space.
///
/// The view is gax's inverse action of the camera pose on points, `cam << Point::slot()`; the
/// projection is a gax map scaling to clip space; the whole is `proj.of(view)`.
pub fn view_map(cam: Pose, half_height: f32, size: [u32; 2]) -> Point<(Point,), f32> {
    let aspect = size[0] as f32 / size[1].max(1) as f32;
    let (sx, sy) = (1.0 / (half_height * aspect), 1.0 / half_height);
    let view: Point<(Point,), f32> = cam << Point::slot();
    let proj =
        Point::<(Point,), f32>::from_coeffs([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]]);
    proj.of(view)
}

/// The camera uniform: the view map in the WGSL layout (`GpuMat`), and the viewport.
pub fn camera(cam: Pose, half_height: f32, size: [u32; 2], time: f32) -> CameraUniform {
    CameraUniform {
        view_proj: view_map(cam, half_height, size).into(),
        viewport: [
            size[0] as f32,
            size[1] as f32,
            2.0 * half_height / size[1].max(1) as f32,
            time,
        ],
    }
}

/// Where a world point lands on the target, in `0..1` uv (y down), for screen effects: the
/// view map applied to the point.
pub fn to_uv(view: &Point<(Point,), f32>, p: Point<(), f32>) -> [f32; 2] {
    let [x, y] = view.of(p).to_euclidean();
    [0.5 + 0.5 * x, 0.5 - 0.5 * y]
}

/// All world segments of a frame, `alpha` of the way from the last tick to the current one.
pub fn world_lines(w: &World, alpha: f32, time: f32, out: &mut Vec<LineInstance>) {
    out.clear();
    // The arena's border, breathing with the director's intensity.
    let glow = 0.8 + 0.6 * w.director.intensity;
    let border = [0.07, 0.9 * glow, 0.45, 0.0];
    let (hw, hh) = (ARENA[0], ARENA[1]);
    let corners = [pt(-hw, -hh), pt(hw, -hh), pt(hw, hh), pt(-hw, hh)];
    outline(out, &corners, 1.0, palette::BORDER, border, identity());
    // Warp-ins.
    for p in &w.pending {
        let f = 1.0 - (p.t / p.total).clamp(0.0, 1.0);
        let m = crate::sim::body::place(p.pos, f * 6.0);
        let c = fade(color(p.kind), 0.4 + 0.8 * f);
        let r = p.kind.radius();
        circle(out, r * (3.0 - 2.0 * f), 20, time * 4.0, c, THIN, m);
        draw_enemy(out, p.kind, m, r * f, 0.0, 0.0, time, 0.25 + 0.5 * f);
    }
    for e in &w.enemies {
        // A serpent's body: shrinking diamonds on a spine.
        if !e.chain.is_empty() {
            let c = color(Kind::Serpent);
            let mut prev = e.body.lerp(alpha) >> ORIGIN;
            for (k, seg_pose) in e.chain.iter().enumerate() {
                let p = *seg_pose >> ORIGIN;
                let f = 1.0 - 0.5 * k as f32 / e.chain.len() as f32;
                out.push(seg(prev, p, fade(c, 0.35), THIN, identity()));
                outline(out, &DIAMOND, 0.38 * f, fade(c, 0.9 * f), THIN, *seg_pose);
                prev = p;
            }
        }
        let m = e.body.lerp(alpha);
        let appear = (e.age * 6.0).min(1.0);
        let t = time + e.phase;
        draw_enemy(out, e.kind, m, e.radius * appear, e.mass, e.flash, t, 1.0);
    }
    for s in &w.shards {
        let blink = if s.life < 1.5 && crate::signal::wave(s.life * 10.0) < 0.0 {
            0.25
        } else {
            1.0
        };
        let m = s.body.lerp(alpha);
        outline(out, &GEM, 1.0, fade(shard(), blink), THIN, m);
    }
    for b in &w.bullets {
        // A short streak behind the shot, along its velocity.
        let p = b.prev + (b.pos - b.prev) * alpha;
        let tail = p - b.vel * 0.03;
        let style = [0.035, 0.28, 0.35, 0.0];
        out.push(seg(tail, p, palette::BULLET, style, identity()));
    }
    if w.phase == Phase::Playing {
        let m = w.ship.body.lerp(alpha);
        let blink =
            w.ship.invulnerable > 0.0 && crate::signal::wave(w.ship.invulnerable * 14.0) < 0.0;
        let speed = w.ship.body.vel.ideal_norm();
        draw_ship(out, m, speed, time, if blink { 0.35 } else { 1.0 });
        if w.ship.invulnerable > 0.0 {
            let c = fade(palette::SHIP, 0.35);
            circle(out, 1.1, 32, time * 2.0, c, THIN, m);
        }
    }
}

/// Text into the HUD (HUD units: the screen is 36 units tall, centred).
pub fn text(
    out: &mut Vec<LineInstance>,
    s: &str,
    x: f32,
    y: f32,
    size: f32,
    color: Light,
    align: Align,
) {
    // Stroke width grows with the glyph; the glow does not, beyond a point.
    let style = [0.045 * size / 1.2, (0.3 * size / 1.2).min(0.45), 0.3, 0.0];
    for [ax, ay, bx, by] in font::segments(s, x, y, size, align) {
        out.push(seg(pt(ax, ay), pt(bx, by), color, style, identity()));
    }
}

/// A number with thousands separators.
pub fn grouped(n: u64) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i).is_multiple_of(3) {
            out.push(' ');
        }
        out.push(c);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weights_scale_and_motors_turn() {
        let p = scaled(pt(1.0, -2.0), 3.0).to_euclidean();
        assert!((p[0] - 3.0).abs() < 1e-6 && (p[1] + 6.0).abs() < 1e-6);
        let q = polar(2.0, core::f32::consts::FRAC_PI_2).to_euclidean();
        assert!(q[0].abs() < 1e-6 && (q[1] - 2.0).abs() < 1e-6);
    }
}
