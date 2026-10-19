//! From the simulation to line instances: shapes placed by interpolated unit motors, bullets,
//! warp-ins, shards and the arena, plus the cameras as gax maps.
//!
//! Shapes are PGA2D points. A shape is drawn at a size by giving its points a weight: the point
//! `(x, y, 1/s)` is `(s x, s y)`, so scaling about the local origin needs no arithmetic on
//! coordinates. Circles, arcs and spirals are points turned by rotation motors. Colours are
//! lights (`light.rs`).

use super::font::{self, Align};
use super::{CameraUniform, LineInstance};
use crate::geom::{ORIGIN, lerp, orbit};
use crate::light::{Light, light};
use crate::sim::body::{Pose, pose_at};
use crate::sim::ship::SHIP_SPEED;
use crate::sim::{ARENA, Kind, Phase, World, at};
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
    light(r, g, b, standard.intensity())
}

/// The shards' light, in the current scheme.
pub fn shard() -> Light {
    match scheme() {
        Scheme::Standard => palette::SHARD,
        Scheme::RedGreen => light(0.85, 1.0, 1.0, 2.6),
        Scheme::BlueYellow => light(1.0, 0.95, 0.6, 2.6),
    }
}

/// How a line is drawn, in world units: its half width, its glow's radius and its glow's
/// strength.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Stroke {
    /// Half the line's width.
    pub width: f32,
    /// How far its glow reaches.
    pub glow: f32,
    /// How bright its glow is.
    pub strength: f32,
}

impl Stroke {
    /// The stroke of half width `width` with a glow of radius `glow` and strength `strength`.
    pub const fn new(width: f32, glow: f32, strength: f32) -> Stroke {
        Stroke {
            width,
            glow,
            strength,
        }
    }
}

/// The default stroke.
pub const THIN: Stroke = Stroke::new(0.032, 0.32, 0.22);

/// `p` scaled by `s` about the origin: the same coordinates with weight `w / s`.
pub fn scaled(p: Point<(), f32>, s: f32) -> Point<(), f32> {
    Point::new(p.e20(), p.e01(), p.e12() / s)
}

/// The point at distance `r` from the origin at `angle`: `(r, 0)` turned by a rotation motor.
pub fn polar(r: f32, angle: f32) -> Point<(), f32> {
    Motor::rotation(ORIGIN, angle) >> at(r, 0.0)
}

/// The segment `a → b`, in the frame of `motor`.
pub fn seg(
    a: Point<(), f32>,
    b: Point<(), f32>,
    color: Light,
    style: Stroke,
    motor: Pose,
) -> LineInstance {
    LineInstance {
        a: a.into(),
        b: b.into(),
        color: color.into(),
        style: [style.width, style.glow, style.strength, 0.0],
        motor: motor.into(),
    }
}

/// A closed outline through `pts`, scaled by `size`, in the frame of `m`. An outline of size
/// zero is nothing (its points would be at infinity, of weight `1 / 0`).
pub fn outline(
    out: &mut Vec<LineInstance>,
    pts: &[Point<(), f32>],
    size: f32,
    color: Light,
    style: Stroke,
    m: Pose,
) {
    if size == 0.0 {
        return;
    }
    for (&a, &b) in pts.iter().zip(pts.iter().cycle().skip(1)) {
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
    style: Stroke,
    m: Pose,
) {
    let step = Motor::rotation(ORIGIN, core::f32::consts::TAU / n as f32);
    out.extend(orbit(polar(r, phase), n, step).map(|(a, b)| seg(a, b, color, style, m)));
}

const DIAMOND: [Point<(), f32>; 4] = [at(1.0, 0.0), at(0.0, 1.0), at(-1.0, 0.0), at(0.0, -1.0)];
const SQUARE: [Point<(), f32>; 4] = [at(0.8, 0.8), at(-0.8, 0.8), at(-0.8, -0.8), at(0.8, -0.8)];
const DART: [Point<(), f32>; 3] = [at(1.0, 0.0), at(-0.6, 0.7), at(-0.6, -0.7)];
const BOW: [Point<(), f32>; 3] = [at(0.0, 0.0), at(0.95, 0.7), at(0.95, -0.7)];
const GEM: [Point<(), f32>; 4] = [at(0.22, 0.0), at(0.0, 0.16), at(-0.22, 0.0), at(0.0, -0.16)];
const SHARP: [Point<(), f32>; 3] = [at(0.55, 0.0), at(0.15, 0.22), at(0.15, -0.22)];
const ARROW: [Point<(), f32>; 4] = [
    at(1.0, 0.0),
    at(-0.7, 0.55),
    at(-0.35, 0.0),
    at(-0.7, -0.55),
];
const HEAD: [Point<(), f32>; 4] = [at(1.1, 0.0), at(-0.3, 0.75), at(-0.1, 0.0), at(-0.3, -0.75)];
const TAIL: [Point<(), f32>; 4] = [at(0.45, 0.0), at(-0.6, 0.5), at(-0.35, 0.0), at(-0.6, -0.5)];
const CARGO: [Point<(), f32>; 3] = [at(0.14, 0.0), at(-0.1, 0.1), at(-0.1, -0.1)];
const HULL: [Point<(), f32>; 4] = [
    at(0.75, 0.0),
    at(-0.45, 0.48),
    at(-0.18, 0.0),
    at(-0.45, -0.48),
];

/// The regular hexagon on the unit circle (the carriers'), turned out once.
static HEX: std::sync::LazyLock<[Point<(), f32>; 6]> = std::sync::LazyLock::new(|| {
    core::array::from_fn(|k| polar(1.0, k as f32 * core::f32::consts::TAU / 6.0))
});

/// The main outline of a family at radius 1, for its wreckage (`None` for a singularity, which
/// is drawn as circles and goes up in particles).
pub fn hull(kind: Kind) -> Option<Vec<Point<(), f32>>> {
    let scaled_by = |pts: &[Point<(), f32>], k: f32| pts.iter().map(|&p| scaled(p, k)).collect();
    match kind {
        Kind::Drifter => Some(DIAMOND.to_vec()),
        Kind::Chaser => Some(scaled_by(&SQUARE, 0.9)),
        Kind::Mote => Some(DART.to_vec()),
        Kind::Evader => Some(BOW.to_vec()),
        Kind::Splitter => Some(scaled_by(&SQUARE, 1.06)),
        Kind::Fragment => Some(ARROW.to_vec()),
        Kind::Serpent => Some(HEAD.to_vec()),
        Kind::Warden => Some(TAIL.to_vec()),
        Kind::Carrier => Some(HEX.to_vec()),
        Kind::Singularity => None,
    }
}

/// The shape of a family, drawn at pose `m` (time `t` animates it); nothing at radius zero
/// (an enemy appearing, a warp-in's first frame).
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
    if radius == 0.0 {
        return;
    }
    // A hit flashes towards white, briefly and partly: under fire the colour must stay. A
    // singularity (under fire for seconds) swells instead.
    let swell = flash;
    let flash = if kind == Kind::Singularity {
        0.0
    } else {
        flash
    };
    let c = ((color(kind)).whitened(0.45 * flash)).faded((1.0 + 0.45 * flash) * bright);
    match kind {
        Kind::Drifter => {
            outline(out, &DIAMOND, radius, c, THIN, m);
            outline(out, &DIAMOND, radius * 0.45, c.faded(0.7), THIN, m);
        }
        Kind::Chaser => {
            outline(out, &SQUARE, radius * 0.9, c, THIN, m);
            // A pinwheel inside, turning against the body.
            let inner = m * Motor::rotation(ORIGIN, -3.0 * t);
            for k in 0..4 {
                let tip = polar(radius * 0.7, k as f32 * core::f32::consts::FRAC_PI_2);
                out.push(seg(ORIGIN, tip, c.faded(0.8), THIN, inner));
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
                outline(out, &SHARP, radius, c.faded(0.75), THIN, m * r);
            }
        }
        Kind::Fragment => outline(out, &ARROW, radius, c, THIN, m),
        Kind::Serpent => {
            outline(out, &HEAD, radius, c, THIN, m);
            // Eyes: a short stroke and its mirror image across the heading.
            for side in [1.0f32, -1.0] {
                let (a, b) = (at(0.25, 0.22 * side), at(0.4, 0.22 * side));
                out.push(seg(a, b, c.faded(1.4), THIN, m));
            }
        }
        Kind::Warden => {
            // The shield: a bright arc across the front, as chords of a turning radius.
            let shield = Stroke::new(0.06, 0.4, 0.4);
            let n = 12;
            let step = Motor::rotation(ORIGIN, 2.6 / n as f32);
            let arc = orbit(polar(1.05 * radius, -1.3), n, step);
            out.extend(arc.map(|(a, b)| seg(a, b, c.faded(1.3), shield, m)));
            outline(out, &TAIL, radius, c, THIN, m);
        }
        Kind::Carrier => {
            outline(out, &*HEX, radius, c, THIN, m);
            let inner = m * Motor::rotation(ORIGIN, -t * 1.3);
            outline(out, &*HEX, radius * 0.62, c.faded(0.7), THIN, inner);
            // Cargo circling inside: each piece a turn about the centre, then out.
            for k in 0..3 {
                let a = t * 2.0 + k as f32 * core::f32::consts::TAU / 3.0;
                let q = Motor::rotation(ORIGIN, a)
                    * Motor::translation(Point::direction(0.35 * radius, 0.0));
                outline(out, &CARGO, 1.0, palette::MOTE, THIN, m * q);
            }
        }
        Kind::Singularity => {
            let pulse = 1.0 + 0.08 * crate::signal::wave(t * 5.0) + 0.02 * mass + 0.06 * swell;
            let style = Stroke::new(0.04, 0.3, 0.3);
            circle(out, radius * pulse, 40, t, c, style, m);
            circle(
                out,
                radius * 0.72 * pulse,
                28,
                -t * 1.7,
                c.faded(0.7),
                THIN,
                m,
            );
            // Spiral arms falling inwards: a direction turned step by step while the radius
            // shrinks (the point's weight grows).
            let turn = Motor::rotation(ORIGIN, 2.4 / 8.0);
            let f = |s: usize| s as f32 / 8.0;
            let size = |s: usize| radius * (1.35 - 1.2 * f(s));
            for arm in 0..5 {
                let a0 = arm as f32 * core::f32::consts::TAU / 5.0 + t * 2.2;
                let turns = orbit(polar(1.0, a0), 8, turn).enumerate();
                out.extend(turns.map(|(s, (a, b))| {
                    let (a, b) = (scaled(a, size(s)), scaled(b, size(s + 1)));
                    seg(a, b, c.faded(0.55 * (1.0 - 0.5 * f(s + 1))), THIN, m)
                }));
            }
            // A white-hot core.
            out.push(seg(
                at(-0.05, 0.0),
                at(0.05, 0.0),
                palette::HOT.faded(bright),
                Stroke::new(0.1, 0.35, 0.5),
                m,
            ));
        }
    }
}

/// The ship.
pub fn draw_ship(out: &mut Vec<LineInstance>, m: Pose, speed: f32, t: f32, alpha: f32) {
    let c = palette::SHIP.faded(alpha);
    let style = Stroke::new(0.04, 0.38, 0.28);
    outline(out, &HULL, 1.2, c, style, m);
    for side in [1.0f32, -1.0] {
        out.push(seg(
            at(0.35, 0.0),
            at(-0.05, 0.18 * side),
            c.faded(0.6),
            THIN,
            m,
        ));
    }
    // Engine flame, flickering with speed.
    if speed > 0.5 {
        let flicker = 0.75 + 0.25 * crate::signal::wave(t * 47.0);
        let len = 0.25 + 0.35 * (speed / SHIP_SPEED).min(1.0) * flicker;
        out.push(seg(
            at(-0.2, 0.0),
            at(-0.2 - len, 0.0),
            palette::FLAME.faded(alpha),
            Stroke::new(0.06, 0.35, 0.5),
            m,
        ));
    }
}

/// Half the HUD's height, in HUD units: the screen is `2 HUD_HALF_HEIGHT` units tall, centred,
/// whatever its size in pixels.
pub const HUD_HALF_HEIGHT: f32 = 18.0;

/// The HUD's camera on a target of `size` pixels at `time`: the identity pose, showing
/// [`HUD_HALF_HEIGHT`] above and below the centre.
pub fn hud_camera(size: [u32; 2], time: f32) -> CameraUniform {
    camera(Pose::identity(), HUD_HALF_HEIGHT, size, time)
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
pub fn to_uv(view: &Point<(Point,), f32>, p: Point<(), f32>) -> Point<(), f32> {
    let [x, y] = view.of(p).to_euclidean();
    at(0.5 + 0.5 * x, 0.5 - 0.5 * y)
}

/// All world segments of a frame, `alpha` of the way from the last tick to the current one.
pub fn world_lines(w: &World, alpha: f32, time: f32, out: &mut Vec<LineInstance>) {
    out.clear();
    // Things drawn where they are in the world.
    let still = Pose::identity();
    // The arena's border, breathing with the director's intensity.
    let glow = 0.8 + 0.6 * w.director.intensity;
    let border = Stroke::new(0.07, 0.9 * glow, 0.45);
    let (hw, hh) = (ARENA[0], ARENA[1]);
    let corners = [at(-hw, -hh), at(hw, -hh), at(hw, hh), at(-hw, hh)];
    outline(out, &corners, 1.0, palette::BORDER, border, still);
    // Warp-ins.
    for p in &w.pending {
        let f = 1.0 - (p.t / p.total).clamp(0.0, 1.0);
        let m = pose_at(p.pos, f * 6.0);
        let c = (color(p.kind)).faded(0.4 + 0.8 * f);
        let r = p.kind.stats().radius;
        circle(out, r * (3.0 - 2.0 * f), 20, time * 4.0, c, THIN, m);
        draw_enemy(out, p.kind, m, r * f, 0.0, 0.0, time, 0.25 + 0.5 * f);
    }
    for e in &w.enemies {
        // A serpent's body: shrinking diamonds on a spine.
        if !e.chain.is_empty() {
            let c = color(Kind::Serpent);
            let mut prev = e.body.interpolated(alpha) >> ORIGIN;
            for (k, body) in e.chain.iter().enumerate() {
                let pose = body.interpolated(alpha);
                let p = pose >> ORIGIN;
                let f = 1.0 - 0.5 * k as f32 / e.chain.len() as f32;
                out.push(seg(prev, p, c.faded(0.35), THIN, still));
                outline(out, &DIAMOND, 0.38 * f, c.faded(0.9 * f), THIN, pose);
                prev = p;
            }
        }
        let m = e.body.interpolated(alpha);
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
        let m = s.body.interpolated(alpha);
        outline(out, &GEM, 1.0, shard().faded(blink), THIN, m);
    }
    for b in &w.bullets {
        // A short streak behind the shot, along its velocity.
        let p = lerp(b.prev, b.pos, alpha);
        let tail = p - b.vel * 0.03;
        let style = Stroke::new(0.035, 0.28, 0.35);
        out.push(seg(tail, p, palette::BULLET, style, still));
    }
    if w.phase == Phase::Playing {
        let m = w.ship.body.interpolated(alpha);
        let blink =
            w.ship.invulnerable > 0.0 && crate::signal::wave(w.ship.invulnerable * 14.0) < 0.0;
        let speed = w.ship.body.vel.ideal_norm();
        draw_ship(out, m, speed, time, if blink { 0.35 } else { 1.0 });
        if w.ship.invulnerable > 0.0 {
            let c = palette::SHIP.faded(0.35);
            circle(out, 1.1, 32, time * 2.0, c, THIN, m);
        }
    }
}

/// Text into the HUD (HUD units, [`HUD_HALF_HEIGHT`]), its baseline at `at`.
pub fn text(
    out: &mut Vec<LineInstance>,
    s: &str,
    at: Point<(), f32>,
    size: f32,
    color: Light,
    align: Align,
) {
    // Stroke width grows with the glyph; the glow does not, beyond a point.
    let style = Stroke::new(0.045 * size / 1.2, (0.3 * size / 1.2).min(0.45), 0.3);
    for [a, b] in font::segments(s, at, size, align) {
        out.push(seg(a, b, color, style, Pose::identity()));
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
    use gax::Extensor;

    #[test]
    fn weights_scale_and_motors_turn() {
        let p = scaled(at(1.0, -2.0), 3.0).to_euclidean();
        assert!((p[0] - 3.0).abs() < 1e-6 && (p[1] + 6.0).abs() < 1e-6);
        let q = polar(2.0, core::f32::consts::FRAC_PI_2).to_euclidean();
        assert!(q[0].abs() < 1e-6 && (q[1] - 2.0).abs() < 1e-6);
    }

    /// A frame's lines are finite: a bot plays a run, and every frame's world and HUD lines
    /// (warp-ins and enemies appearing at size zero included) have finite coefficients.
    #[test]
    fn the_games_lines_are_finite() {
        let mut g = crate::game::Game::new(crate::store::Store::new(None));
        let mut sound = crate::audio::Sound::new();
        g.start_seeded(crate::game::Mode::Plane, 4);
        let finite = |l: &LineInstance| {
            let (a, b): (Point<(), f32>, Point<(), f32>) = (l.a.into(), l.b.into());
            let m: gax::pga2d::Motor<(), f32> = l.motor.into();
            let c: gax::pga3d::Point<(), f32> = l.color.into();
            a.is_finite()
                && b.is_finite()
                && m.is_finite()
                && c.is_finite()
                && l.style.iter().all(|s| s.is_finite())
        };
        let mut t = 0.0;
        while t < 40.0 {
            t += 1.0 / 60.0;
            g.time = t;
            let (input, _) = crate::bot::input(&g.run, t);
            crate::frame::advance(
                &mut g,
                input,
                Default::default(),
                1.0 / 60.0,
                16.0 / 9.0,
                &mut sound,
            );
            let world = &g.run.plane().expect("a Plane run").world;
            world_lines(world, g.alpha, g.time, &mut g.world_lines);
            crate::hud::hud(&mut g, [1600, 900], None);
            for (k, l) in g.world_lines.iter().chain(&g.hud_lines).enumerate() {
                assert!(finite(l), "at {t:.2} s, line {k}: {l:?}");
            }
        }
    }
}
