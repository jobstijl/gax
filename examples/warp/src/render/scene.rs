//! From the simulation to line instances: shapes placed by interpolated unit motors, bullets,
//! warp-ins, shards and the arena, plus the cameras as gax maps.

use super::font::{self, Align};
use super::{CameraUniform, LineInstance};
use crate::sim::body::{Pose, identity};
use crate::sim::{ARENA, Kind, Phase, World};
use crate::store::Scheme;
use gax::pga2d::{Motor, Point};

/// Colours: HDR rgb and intensity. One hue per family; the player's is used by nothing else.
pub mod palette {
    /// The ship: warm white.
    pub const SHIP: [f32; 4] = [1.0, 0.86, 0.62, 3.2];
    /// Shots.
    pub const BULLET: [f32; 4] = [1.0, 0.82, 0.5, 3.0];
    /// Drifters: cyan.
    pub const DRIFTER: [f32; 4] = [0.15, 0.85, 1.0, 2.6];
    /// Chasers: magenta.
    pub const CHASER: [f32; 4] = [1.0, 0.2, 0.62, 2.8];
    /// Motes: violet-pink.
    pub const MOTE: [f32; 4] = [0.85, 0.35, 1.0, 2.6];
    /// Singularities: deep violet.
    pub const SINGULARITY: [f32; 4] = [0.55, 0.3, 1.0, 1.9];
    /// Evaders: yellow.
    pub const EVADER: [f32; 4] = [1.0, 0.86, 0.1, 2.6];
    /// Splitters and their fragments: orange.
    pub const SPLITTER: [f32; 4] = [1.0, 0.36, 0.1, 2.7];
    /// Serpents: teal.
    pub const SERPENT: [f32; 4] = [0.1, 1.0, 0.72, 2.5];
    /// Wardens: red.
    pub const WARDEN: [f32; 4] = [1.0, 0.14, 0.18, 2.8];
    /// Carriers: blue.
    pub const CARRIER: [f32; 4] = [0.32, 0.48, 1.0, 2.8];
    /// Shards: lime.
    pub const SHARD: [f32; 4] = [0.45, 1.0, 0.3, 2.6];
    /// The arena's border.
    pub const BORDER: [f32; 4] = [0.3, 0.5, 1.0, 2.0];
    /// The lattice.
    pub const GRID: [f32; 4] = [0.14, 0.24, 0.75, 0.22];
    /// HUD text.
    pub const HUD: [f32; 4] = [0.75, 0.85, 1.0, 1.6];
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

/// The colour of a family, in the current scheme. Shapes carry the identity of every family;
/// the schemes keep neighbours in the roster apart for colour-blind players.
pub fn color(kind: Kind) -> [f32; 4] {
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
    let rgb = match (scheme(), kind) {
        (Scheme::Standard, _) => return standard,
        // Blue, orange, yellow and purple, after Okabe and Ito; no red against green.
        (Scheme::RedGreen, Kind::Drifter) => [0.35, 0.72, 1.0],
        (Scheme::RedGreen, Kind::Chaser) => [1.0, 0.62, 0.0],
        (Scheme::RedGreen, Kind::Mote) => [0.95, 0.55, 0.85],
        (Scheme::RedGreen, Kind::Singularity) => [0.45, 0.35, 1.0],
        (Scheme::RedGreen, Kind::Evader) => [1.0, 0.95, 0.25],
        (Scheme::RedGreen, Kind::Splitter | Kind::Fragment) => [0.85, 0.3, 0.0],
        (Scheme::RedGreen, Kind::Serpent) => [0.0, 0.8, 0.65],
        (Scheme::RedGreen, Kind::Warden) => [1.0, 0.75, 0.9],
        (Scheme::RedGreen, Kind::Carrier) => [0.15, 0.4, 1.0],
        // Reds against cyans; no blue against green or yellow against violet.
        (Scheme::BlueYellow, Kind::Drifter) => [0.1, 0.9, 1.0],
        (Scheme::BlueYellow, Kind::Chaser) => [1.0, 0.2, 0.62],
        (Scheme::BlueYellow, Kind::Mote) => [1.0, 0.55, 0.75],
        (Scheme::BlueYellow, Kind::Singularity) => [0.55, 0.3, 1.0],
        (Scheme::BlueYellow, Kind::Evader) => [1.0, 0.22, 0.15],
        (Scheme::BlueYellow, Kind::Splitter | Kind::Fragment) => [1.0, 0.62, 0.5],
        (Scheme::BlueYellow, Kind::Serpent) => [0.1, 0.8, 0.8],
        (Scheme::BlueYellow, Kind::Warden) => [0.95, 0.95, 1.0],
        (Scheme::BlueYellow, Kind::Carrier) => [0.3, 0.5, 1.0],
    };
    [rgb[0], rgb[1], rgb[2], standard[3]]
}

/// The shards' colour, in the current scheme.
pub fn shard() -> [f32; 4] {
    match scheme() {
        Scheme::Standard | Scheme::BlueYellow => palette::SHARD,
        Scheme::RedGreen => [0.85, 1.0, 1.0, 2.6],
    }
}

const THIN: [f32; 4] = [0.032, 0.32, 0.22, 0.0];

/// A segment in the frame of `motor`.
pub fn seg(ab: [f32; 4], color: [f32; 4], style: [f32; 4], motor: Pose) -> LineInstance {
    LineInstance {
        ab,
        color,
        style,
        motor: motor.into(),
    }
}

/// Segments of a closed polygon.
fn polygon(
    out: &mut Vec<LineInstance>,
    pts: &[[f32; 2]],
    scale: f32,
    color: [f32; 4],
    style: [f32; 4],
    m: Pose,
) {
    for i in 0..pts.len() {
        let (a, b) = (pts[i], pts[(i + 1) % pts.len()]);
        out.push(seg(
            [a[0] * scale, a[1] * scale, b[0] * scale, b[1] * scale],
            color,
            style,
            m,
        ));
    }
}

/// A circle as a polygon.
fn circle(
    out: &mut Vec<LineInstance>,
    r: f32,
    n: usize,
    phase: f32,
    color: [f32; 4],
    style: [f32; 4],
    m: Pose,
) {
    let pts: Vec<[f32; 2]> = (0..n)
        .map(|k| {
            let a = phase + k as f32 * core::f32::consts::TAU / n as f32;
            [r * a.cos(), r * a.sin()]
        })
        .collect();
    polygon(out, &pts, 1.0, color, style, m);
}

fn scale_color(c: [f32; 4], k: f32) -> [f32; 4] {
    [c[0], c[1], c[2], c[3] * k]
}

fn mix_white(c: [f32; 4], t: f32) -> [f32; 4] {
    [
        c[0] + (1.0 - c[0]) * t,
        c[1] + (1.0 - c[1]) * t,
        c[2] + (1.0 - c[2]) * t,
        c[3] * (1.0 + t),
    ]
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
    fade: f32,
) {
    // A hit flashes towards white, briefly and partly: under fire the colour must stay. A
    // singularity (under fire for seconds) swells instead.
    let swell = flash;
    let flash = if kind == Kind::Singularity {
        0.0
    } else {
        flash
    };
    let c = scale_color(mix_white(color(kind), 0.45 * flash), fade);
    match kind {
        Kind::Drifter => {
            polygon(
                out,
                &[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]],
                radius,
                c,
                THIN,
                m,
            );
            polygon(
                out,
                &[[0.45, 0.0], [0.0, 0.45], [-0.45, 0.0], [0.0, -0.45]],
                radius,
                scale_color(c, 0.7),
                THIN,
                m,
            );
        }
        Kind::Chaser => {
            polygon(
                out,
                &[[0.8, 0.8], [-0.8, 0.8], [-0.8, -0.8], [0.8, -0.8]],
                radius * 0.9,
                c,
                THIN,
                m,
            );
            // A pinwheel inside, turning against the body.
            let spin = Motor::rotation(Point::xy(0.0, 0.0), -3.0 * t);
            let inner = m * spin;
            for k in 0..4 {
                let a = k as f32 * core::f32::consts::FRAC_PI_2;
                let (s, co) = a.sin_cos();
                out.push(seg(
                    [0.0, 0.0, co * radius * 0.7, s * radius * 0.7],
                    scale_color(c, 0.8),
                    THIN,
                    inner,
                ));
            }
        }
        Kind::Mote => {
            polygon(
                out,
                &[[1.0, 0.0], [-0.6, 0.7], [-0.6, -0.7]],
                radius,
                c,
                THIN,
                m,
            );
        }
        Kind::Evader => {
            // A bow tie along its heading.
            polygon(
                out,
                &[[0.0, 0.0], [0.95, 0.7], [0.95, -0.7]],
                radius,
                c,
                THIN,
                m,
            );
            polygon(
                out,
                &[[0.0, 0.0], [-0.95, 0.7], [-0.95, -0.7]],
                radius,
                c,
                THIN,
                m,
            );
        }
        Kind::Splitter => {
            polygon(
                out,
                &[[0.85, 0.85], [-0.85, 0.85], [-0.85, -0.85], [0.85, -0.85]],
                radius,
                c,
                THIN,
                m,
            );
            // Its three fragments, visible inside.
            for k in 0..3 {
                let a = k as f32 * core::f32::consts::TAU / 3.0;
                let r = Motor::rotation(Point::xy(0.0, 0.0), a);
                polygon(
                    out,
                    &[[0.55, 0.0], [0.15, 0.22], [0.15, -0.22]],
                    radius,
                    scale_color(c, 0.75),
                    THIN,
                    m * r,
                );
            }
        }
        Kind::Fragment => {
            polygon(
                out,
                &[[1.0, 0.0], [-0.7, 0.55], [-0.35, 0.0], [-0.7, -0.55]],
                radius,
                c,
                THIN,
                m,
            );
        }
        Kind::Serpent => {
            polygon(
                out,
                &[[1.1, 0.0], [-0.3, 0.75], [-0.1, 0.0], [-0.3, -0.75]],
                radius,
                c,
                THIN,
                m,
            );
            // Eyes.
            for side in [-1.0f32, 1.0] {
                out.push(seg(
                    [0.25, 0.22 * side, 0.4, 0.22 * side],
                    scale_color(c, 1.4),
                    THIN,
                    m,
                ));
            }
        }
        Kind::Warden => {
            // The shield: a bright arc across the front.
            let arc: Vec<[f32; 2]> = (0..=12)
                .map(|k| {
                    let a = -1.3 + 2.6 * k as f32 / 12.0;
                    [1.05 * a.cos(), 1.05 * a.sin()]
                })
                .collect();
            let shield = [0.06, 0.4, 0.4, 0.0];
            for w in arc.windows(2) {
                out.push(seg(
                    [
                        w[0][0] * radius,
                        w[0][1] * radius,
                        w[1][0] * radius,
                        w[1][1] * radius,
                    ],
                    scale_color(c, 1.3),
                    shield,
                    m,
                ));
            }
            polygon(
                out,
                &[[0.45, 0.0], [-0.6, 0.5], [-0.35, 0.0], [-0.6, -0.5]],
                radius,
                c,
                THIN,
                m,
            );
        }
        Kind::Carrier => {
            let hex = |r: f32| -> Vec<[f32; 2]> {
                (0..6)
                    .map(|k| {
                        let a = k as f32 * core::f32::consts::FRAC_PI_3;
                        [r * a.cos(), r * a.sin()]
                    })
                    .collect()
            };
            polygon(out, &hex(1.0), radius, c, THIN, m);
            let inner = m * Motor::rotation(Point::xy(0.0, 0.0), -t * 1.3);
            polygon(out, &hex(0.62), radius, scale_color(c, 0.7), THIN, inner);
            // Cargo circling inside.
            for k in 0..3 {
                let a = t * 2.0 + k as f32 * core::f32::consts::TAU / 3.0;
                let p = [0.35 * radius * a.cos(), 0.35 * radius * a.sin()];
                let q = Motor::translation(p[0], p[1]);
                polygon(
                    out,
                    &[[0.14, 0.0], [-0.1, 0.1], [-0.1, -0.1]],
                    1.0,
                    palette::MOTE,
                    THIN,
                    m * q,
                );
            }
        }
        Kind::Singularity => {
            let pulse = 1.0 + 0.08 * (t * 5.0).sin() + 0.02 * mass + 0.06 * swell;
            let style = [0.04, 0.3, 0.3, 0.0];
            circle(out, radius * pulse, 40, t, c, style, m);
            circle(
                out,
                radius * 0.72 * pulse,
                28,
                -t * 1.7,
                scale_color(c, 0.7),
                THIN,
                m,
            );
            // Spiral arms falling inwards.
            for arm in 0..5 {
                let a0 = arm as f32 * core::f32::consts::TAU / 5.0 + t * 2.2;
                let mut prev = [radius * 1.35 * a0.cos(), radius * 1.35 * a0.sin()];
                for s in 1..9 {
                    let f = s as f32 / 8.0;
                    let r = radius * (1.35 - 1.2 * f);
                    let a = a0 + f * 2.4;
                    let p = [r * a.cos(), r * a.sin()];
                    out.push(seg(
                        [prev[0], prev[1], p[0], p[1]],
                        scale_color(c, 0.55 * (1.0 - 0.5 * f)),
                        THIN,
                        m,
                    ));
                    prev = p;
                }
            }
            // A white-hot core.
            out.push(seg(
                [-0.05, 0.0, 0.05, 0.0],
                [1.0, 0.9, 1.0, 2.2 * fade],
                [0.1, 0.35, 0.5, 0.0],
                m,
            ));
        }
    }
}

/// The ship.
pub fn draw_ship(out: &mut Vec<LineInstance>, m: Pose, speed: f32, t: f32, alpha: f32) {
    let c = scale_color(palette::SHIP, alpha);
    let style = [0.04, 0.38, 0.28, 0.0];
    polygon(
        out,
        &[[0.75, 0.0], [-0.45, 0.48], [-0.18, 0.0], [-0.45, -0.48]],
        1.2,
        c,
        style,
        m,
    );
    out.push(seg([0.35, 0.0, -0.05, 0.18], scale_color(c, 0.6), THIN, m));
    out.push(seg([0.35, 0.0, -0.05, -0.18], scale_color(c, 0.6), THIN, m));
    // Engine flame, flickering with speed.
    if speed > 0.5 {
        let len = 0.25 + 0.35 * (speed / 11.5).min(1.0) * (0.75 + 0.25 * (t * 47.0).sin());
        let flame = [1.0, 0.45, 0.15, 3.0 * alpha];
        out.push(seg(
            [-0.2, 0.0, -0.2 - len, 0.0],
            flame,
            [0.06, 0.35, 0.5, 0.0],
            m,
        ));
    }
}

/// The camera uniform for a camera at `cam` showing `half_height` world units above and below
/// its centre, on a target of `size` pixels.
///
/// The view is gax's inverse action of the camera pose on points, `cam << Point::slot()`; the
/// projection is a gax map scaling to clip space; the matrix is `proj.of(view)`, uploaded in
/// the WGSL layout (`GpuMat`).
pub fn camera(cam: Pose, half_height: f32, size: [u32; 2], time: f32) -> CameraUniform {
    let aspect = size[0] as f32 / size[1].max(1) as f32;
    let (sx, sy) = (1.0 / (half_height * aspect), 1.0 / half_height);
    let view: Point<(Point,), f32> = cam << Point::slot();
    let proj =
        Point::<(Point,), f32>::from_coeffs([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]]);
    CameraUniform {
        view_proj: proj.of(view).into(),
        viewport: [
            size[0] as f32,
            size[1] as f32,
            2.0 * half_height / size[1].max(1) as f32,
            time,
        ],
    }
}

/// Where a world point lands on the target, in `0..1` uv (y down), for screen effects.
pub fn to_uv(cam: &CameraUniform, p: [f32; 2]) -> [f32; 2] {
    let m = &cam.view_proj.cols;
    let x = m[0][0] * p[0] + m[1][0] * p[1] + m[2][0];
    let y = m[0][1] * p[0] + m[1][1] * p[1] + m[2][1];
    let w = m[0][2] * p[0] + m[1][2] * p[1] + m[2][2];
    [0.5 + 0.5 * x / w, 0.5 - 0.5 * y / w]
}

/// All world segments of a frame, `alpha` of the way from the last tick to the current one.
pub fn world_lines(w: &World, alpha: f32, time: f32, out: &mut Vec<LineInstance>) {
    out.clear();
    // The arena's border, breathing with the director's intensity.
    let glow = 0.8 + 0.6 * w.director.intensity;
    let border = [0.07, 0.9 * glow, 0.45, 0.0];
    let (hw, hh) = (ARENA[0], ARENA[1]);
    polygon(
        out,
        &[[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]],
        1.0,
        palette::BORDER,
        border,
        identity(),
    );
    // Warp-ins.
    for p in &w.pending {
        let f = 1.0 - (p.t / p.total).clamp(0.0, 1.0);
        let m =
            Motor::translation(p.pos[0], p.pos[1]) * Motor::rotation(Point::xy(0.0, 0.0), f * 6.0);
        let c = scale_color(color(p.kind), 0.4 + 0.8 * f);
        circle(
            out,
            p.kind.radius() * (3.0 - 2.0 * f),
            20,
            time * 4.0,
            c,
            THIN,
            m,
        );
        draw_enemy(
            out,
            p.kind,
            m,
            p.kind.radius() * f,
            0.0,
            0.0,
            time,
            0.25 + 0.5 * f,
        );
    }
    for e in &w.enemies {
        // A serpent's body: shrinking diamonds on a spine.
        if !e.chain.is_empty() {
            let c = color(Kind::Serpent);
            let mut prev = (e.body.lerp(alpha) >> Point::xy(0.0, 0.0)).to_euclidean();
            for (k, seg_pose) in e.chain.iter().enumerate() {
                let p = (*seg_pose >> Point::xy(0.0, 0.0)).to_euclidean();
                let f = 1.0 - 0.5 * k as f32 / e.chain.len() as f32;
                out.push(seg(
                    [prev[0], prev[1], p[0], p[1]],
                    scale_color(c, 0.35),
                    THIN,
                    identity(),
                ));
                let r = 0.38 * f;
                polygon(
                    out,
                    &[[r, 0.0], [0.0, r], [-r, 0.0], [0.0, -r]],
                    1.0,
                    scale_color(c, 0.9 * f),
                    THIN,
                    *seg_pose,
                );
                prev = p;
            }
        }
        let m = e.body.lerp(alpha);
        let appear = (e.age * 6.0).min(1.0);
        draw_enemy(
            out,
            e.kind,
            m,
            e.radius * appear,
            e.mass,
            e.flash,
            time + e.phase,
            1.0,
        );
    }
    for s in &w.shards {
        let blink = if s.life < 1.5 && (s.life * 10.0).sin() < 0.0 {
            0.25
        } else {
            1.0
        };
        let m = s.body.lerp(alpha);
        polygon(
            out,
            &[[0.22, 0.0], [0.0, 0.16], [-0.22, 0.0], [0.0, -0.16]],
            1.0,
            scale_color(shard(), blink),
            THIN,
            m,
        );
    }
    for b in &w.bullets {
        let p = b.prev + (b.pos - b.prev).gp(alpha);
        let [x, y] = p.to_euclidean();
        let (vx, vy) = (b.vel.e20(), b.vel.e01());
        out.push(seg(
            [x - vx * 0.03, y - vy * 0.03, x, y],
            palette::BULLET,
            [0.035, 0.28, 0.35, 0.0],
            identity(),
        ));
    }
    if w.phase == Phase::Playing {
        let m = w.ship.body.lerp(alpha);
        let blink = w.ship.invulnerable > 0.0 && (w.ship.invulnerable * 14.0).sin() < 0.0;
        let speed = crate::sim::body::length(w.ship.body.vel);
        draw_ship(out, m, speed, time, if blink { 0.35 } else { 1.0 });
        if w.ship.invulnerable > 0.0 {
            circle(
                out,
                1.1,
                32,
                time * 2.0,
                scale_color(palette::SHIP, 0.35),
                THIN,
                m,
            );
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
    color: [f32; 4],
    align: Align,
) {
    // Stroke width grows with the glyph; the glow does not, beyond a point.
    let style = [0.045 * size / 1.2, (0.3 * size / 1.2).min(0.45), 0.3, 0.0];
    for ab in font::segments(s, x, y, size, align) {
        out.push(seg(ab, color, style, identity()));
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
