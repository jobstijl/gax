//! Drawing the Tunnel: a camera motor in 3D, perspective by hand, and everything as projected
//! lines in the HUD's screen units (the line renderer is 2D).
//!
//! * The camera is a PGA3D motor: `cam << p` takes a world point into its frame. It follows
//!   the ship on a motor spring (interpolation towards the target), and by default does not
//!   roll with the tunnel (a comfort setting), which is why it is built from a heading and a
//!   pitch rather than taken from the track's frame.
//! * Depth reads through fog that fades into the dark, line widths and glows that scale with
//!   `1/z`, bolts that glow brighter as they come close, the ship's shadow on the wall (a meet
//!   of the light-to-ship line with the wall's tangent plane), and the reticle's lock and lead.

use super::LineInstance;
use super::font::Align;
use super::scene::{self, palette};
use crate::sim::body::identity;
use crate::sim::rng::Rng;
use crate::sim::{DT, Phase};
use crate::tunnel::lattice::{AROUND, RADIUS, RINGS};
use crate::tunnel::track::{Frame, interpolate};
use crate::tunnel::{AIM_DEPTH, Event, Foe, World};
use gax::pga3d::{Motor, Plane, Point};

/// How far the fog lets you see.
const FOG: f32 = 78.0;
/// The wall fades sooner: its lines crowd towards the vanishing point, where enemies come
/// from, and must not glare there.
const WALL_FOG: f32 = 58.0;
const NEAR: f32 = 0.25;

/// A particle in straightened coordinates.
#[derive(Clone, Copy, Debug)]
struct Spark {
    pos: [f32; 3],
    vel: [f32; 3],
    color: [f32; 4],
    life: f32,
    total: f32,
}

/// The Tunnel's view: camera, effects, and the projection.
pub struct View {
    /// The camera's pose in the world.
    pub cam: Frame,
    /// Screen units per unit of `x/z` (from the field of view).
    pub focal: f32,
    /// Follow the tunnel's roll (off by default).
    pub follow_roll: bool,
    /// Screen shake scale (a setting).
    pub shake_scale: f32,
    /// Flash scale (the reduced-flashes setting).
    pub flash_scale: f32,
    shake: f32,
    /// A full-screen flash, decaying.
    pub flash: f32,
    sparks: Vec<Spark>,
    dust: Vec<[f32; 3]>,
    popups: Vec<([f32; 3], String, f32, [f32; 4])>,
    rng: Rng,
    started: bool,
}

fn len3(v: [f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn fade(c: [f32; 4], k: f32) -> [f32; 4] {
    [c[0], c[1], c[2], c[3] * k]
}

impl View {
    /// A view with a field of view in degrees.
    pub fn new(fov_degrees: f32) -> View {
        let mut rng = Rng::new(0xd057);
        let dust = (0..420)
            .map(|_| {
                let a = rng.angle();
                let r = RADIUS * rng.unit().sqrt() * 0.95;
                [r * a.cos(), r * a.sin(), rng.range(-5.0, FOG)]
            })
            .collect();
        View {
            cam: Motor::translation(0.0, 0.0, 0.0),
            focal: View::focal_for(fov_degrees),
            follow_roll: false,
            shake_scale: 1.0,
            flash_scale: 1.0,
            shake: 0.0,
            flash: 0.0,
            sparks: Vec::new(),
            dust,
            popups: Vec::new(),
            rng,
            started: false,
        }
    }

    /// The focal length for a vertical field of view (the HUD is 36 units tall).
    pub fn focal_for(fov_degrees: f32) -> f32 {
        18.0 / (fov_degrees.to_radians() * 0.5).tan()
    }

    /// The camera the view wants: behind and above the ship, looking down the track. Without
    /// roll it is built from a heading and a pitch about the world's axes.
    fn target(&self, w: &World, alpha: f32) -> Frame {
        let ship = lerp3(xyz(w.ship.prev), xyz(w.ship.pos), alpha);
        let [x, y, s] = ship;
        if self.follow_roll {
            let m = w.track.frame(s - 6.5);
            let pitch = Motor::rotation_about(1.0, 0.0, 0.0, 0.12);
            return m * Motor::translation(0.45 * x, 0.45 * y + 1.3, 0.0) * pitch;
        }
        let eye = w.track.point(0.45 * x, 0.45 * y + 1.3, s - 6.5);
        let look = w.track.point(0.2 * x, 0.2 * y, s + 16.0);
        let f = [look[0] - eye[0], look[1] - eye[1], look[2] - eye[2]];
        let l = len3(f);
        let (fx, fy, fz) = (f[0] / l, f[1] / l, f[2] / l);
        let yaw = fx.atan2(fz);
        let pitch = fy.clamp(-1.0, 1.0).asin();
        Motor::translation(eye[0], eye[1], eye[2])
            * Motor::rotation_about(0.0, 1.0, 0.0, yaw)
            * Motor::rotation_about(1.0, 0.0, 0.0, -pitch)
    }

    /// Advance the camera spring and the effects by a frame of `dt`.
    pub fn update(&mut self, w: &World, alpha: f32, dt: f32) {
        let target = self.target(w, alpha);
        if !self.started {
            self.cam = target;
            self.started = true;
        } else {
            self.cam = interpolate(self.cam, target, 1.0 - (-12.0 * dt).exp());
        }
        self.shake = (self.shake - dt * 2.2).max(0.0);
        self.flash = (self.flash - dt * 2.5).max(0.0);
        for p in &mut self.sparks {
            for k in 0..3 {
                p.pos[k] += p.vel[k] * dt;
                p.vel[k] *= 1.0 - 1.8 * dt;
            }
            p.life -= dt;
        }
        self.sparks.retain(|p| p.life > 0.0);
        for p in &mut self.popups {
            p.2 += dt;
        }
        self.popups.retain(|p| p.2 < 1.1);
        // Dust behind the ship comes round again ahead.
        let s = w.ship.s();
        for d in &mut self.dust {
            if d[2] < s - 5.0 {
                d[2] += FOG + 5.0;
            }
        }
    }

    /// The camera for drawing: with shake.
    fn shaken(&mut self) -> Frame {
        let a = self.shake * self.shake * self.shake_scale;
        if a < 1e-4 {
            return self.cam;
        }
        let r = &mut self.rng;
        let jitter = Motor::translation(
            r.range(-1.0, 1.0) * a * 0.3,
            r.range(-1.0, 1.0) * a * 0.3,
            0.0,
        ) * Motor::rotation_about(0.0, 0.0, 1.0, r.range(-0.015, 0.015) * a);
        self.cam * jitter
    }

    fn burst(&mut self, pos: [f32; 3], color: [f32; 4], n: usize, speed: f32, life: f32) {
        for _ in 0..n {
            let a = self.rng.angle();
            let b = self.rng.range(-1.0, 1.0);
            let r = (1.0 - b * b).sqrt();
            let v = self.rng.range(0.3, 1.0) * speed;
            let l = self.rng.range(0.5, 1.0) * life;
            self.sparks.push(Spark {
                pos,
                vel: [r * a.cos() * v, r * a.sin() * v, b * v],
                color,
                life: l,
                total: l,
            });
        }
        if self.sparks.len() > 6000 {
            self.sparks.drain(..self.sparks.len() - 6000);
        }
    }

    /// React to a tick's events.
    pub fn on_events(&mut self, events: &[Event]) {
        for e in events {
            match *e {
                Event::Hit { pos, foe } => {
                    self.burst(pos, scene::color(foe.kind()), 6, 6.0, 0.25);
                }
                Event::Kill { pos, foe, points } => {
                    let c = scene::color(foe.kind());
                    let big = foe == Foe::Turret;
                    self.burst(pos, c, if big { 160 } else { 70 }, 11.0, 0.9);
                    self.burst(pos, [1.0, 1.0, 1.0, 3.0], 12, 16.0, 0.3);
                    self.shake = self.shake.max(if big { 0.5 } else { 0.15 });
                    if points >= 250 {
                        self.popups.push((pos, points.to_string(), 0.0, c));
                    }
                }
                Event::Wall { pos } => {
                    self.burst(pos, palette::BULLET, 4, 4.0, 0.2);
                }
                Event::Bolt { pos } => {
                    self.burst(pos, BOLT, 10, 3.0, 0.3);
                }
                Event::Pickup { pos, mult } => {
                    self.burst(pos, scene::shard(), 8, 4.0, 0.3);
                    if mult % 10 == 0 {
                        self.popups
                            .push((pos, format!("X{mult}"), 0.0, scene::shard()));
                    }
                }
                Event::Death { pos } => {
                    self.burst(pos, palette::SHIP, 900, 22.0, 1.8);
                    self.shake = 1.6;
                    self.flash = 0.7;
                }
                Event::Bomb { pos } => {
                    for k in 0..6 {
                        let p = [0.0, 0.0, pos[2] + 6.0 + 9.0 * k as f32];
                        self.burst(p, [0.55, 0.8, 1.0, 3.0], 220, 26.0, 1.0);
                    }
                    self.shake = 1.2;
                    self.flash = 0.5;
                }
                Event::Roll { .. } => {}
                Event::Warn { .. }
                | Event::Fire { .. }
                | Event::Respawn
                | Event::Extra { .. }
                | Event::GameOver => {}
            }
        }
    }

    /// Post settings for the frame.
    pub fn post(&self) -> super::PostSettings {
        super::PostSettings {
            bloom: 0.34,
            exposure: 1.0 + self.flash * 1.5 * self.flash_scale,
            vignette: 0.45,
            grain: 0.012,
            aberration: 0.007,
            saturation: 1.35,
            shock: None,
        }
    }

    /// Build the frame's lines (screen units, for the HUD camera). `aim` is the reticle.
    pub fn draw(
        &mut self,
        w: &World,
        alpha: f32,
        time: f32,
        aim: [f32; 2],
        out: &mut Vec<LineInstance>,
    ) {
        out.clear();
        let cam = self.shaken();
        let p = Proj {
            cam,
            focal: self.focal,
        };
        let track = &w.track;
        let s_ship = lerp(w.ship.prev.e021(), w.ship.pos.e021(), alpha);
        let world = |q: [f32; 3]| track.point(q[0], q[1], q[2]);

        // The wall: every node into the world once, then the lines between neighbours.
        let lat = &w.lattice;
        let mut nodes = vec![None; RINGS * AROUND];
        for r in 0..RINGS {
            let f = track.frame(lat.ring_s(r));
            for j in 0..AROUND {
                let q = lat.node(r, j);
                let pw = (f >> Point::xyz(q[0], q[1], 0.0)).to_euclidean();
                // The along-track displacement: a small shift along the frame's forward.
                let fw = f >> Point::direction(0.0, 0.0, 1.0);
                let dz = q[2] - lat.ring_s(r);
                let pw = [
                    pw[0] + fw.e032() * dz,
                    pw[1] + fw.e013() * dz,
                    pw[2] + fw.e021() * dz,
                ];
                nodes[r * AROUND + j] = p.camera(pw).map(|c| (c, lat.strain(r, j)));
            }
        }
        let grid = palette::GRID;
        for r in 0..RINGS {
            for j in 0..AROUND {
                let Some((a, sa)) = nodes[r * AROUND + j] else {
                    continue;
                };
                let mut edge = |b: Option<([f32; 3], f32)>, bright: f32| {
                    if let Some((b, sb)) = b {
                        let glow = 1.0 + 4.0 * (sa + sb).min(1.5);
                        let c = [grid[0], grid[1], grid[2], grid[3] * 1.5 * bright * glow];
                        p.segment_in(out, a, b, c, 0.03, 0.22, true);
                    }
                };
                edge(nodes[r * AROUND + (j + 1) % AROUND], 1.0);
                if r + 1 < RINGS {
                    edge(nodes[(r + 1) * AROUND + j], 0.8);
                }
            }
        }

        // Dust, streaked by the speed.
        let streak = 0.012 * w.ship.speed;
        for d in &self.dust {
            if d[2] < s_ship - 2.0 {
                continue;
            }
            let a = p.camera(world(*d));
            let b = p.camera(world([d[0], d[1], d[2] - streak]));
            if let (Some(a), Some(b)) = (a, b) {
                p.segment_in(out, a, b, [0.6, 0.7, 1.0, 0.35], 0.02, 0.08, true);
            }
        }

        // Warp-ins: rings contracting onto where a spawn lands.
        for pd in &w.pending {
            let k = (pd.t / 0.8).clamp(0.0, 1.0);
            let c = fade(scene::color(pd.foe.kind()), 0.8);
            let r = 0.4 + 2.0 * k;
            p.ring(out, &world, pd.pos, r, c, 10);
        }

        // Enemies.
        for e in &w.enemies {
            let q = lerp3(xyz(e.prev), xyz(e.pos), alpha);
            let c = scene::color(e.foe.kind());
            let c = if e.flash > 0.0 {
                let k = e.flash;
                [
                    c[0] + (1.0 - c[0]) * k,
                    c[1] + (1.0 - c[1]) * k,
                    c[2] + (1.0 - c[2]) * k,
                    c[3],
                ]
            } else {
                c
            };
            let spin = e.age * 2.0 + e.id as f32;
            match e.foe {
                Foe::Drone => p.solid(out, &world, q, &OCTAHEDRON, 0.7, spin, c),
                Foe::Mine => {
                    let pulse = 1.0 + 0.15 * (e.age * 7.0).sin();
                    p.solid(out, &world, q, &SPIKES, 0.9 * pulse, spin * 0.5, c);
                    p.solid(out, &world, q, &CUBE, 0.35, -spin, c);
                }
                Foe::Turret => {
                    // A pyramid on the wall, pointing at the axis; it glows before it fires.
                    let charge = ((0.35 - e.timer) / 0.35).clamp(0.0, 1.0);
                    let c = fade(c, 1.0 + 2.0 * charge);
                    let (ca, sa) = (e.angle.cos(), e.angle.sin());
                    let base = |t: f32| {
                        let (x, y) = (q[0] + 0.5 * ca, q[1] + 0.5 * sa);
                        let (tx, ty) = (-sa * t, ca * t);
                        [x + tx, y + ty]
                    };
                    let tip = [q[0] - 1.1 * ca, q[1] - 1.1 * sa, q[2]];
                    let corners = [
                        [base(0.8)[0], base(0.8)[1], q[2] - 0.8],
                        [base(-0.8)[0], base(-0.8)[1], q[2] - 0.8],
                        [base(-0.8)[0], base(-0.8)[1], q[2] + 0.8],
                        [base(0.8)[0], base(0.8)[1], q[2] + 0.8],
                    ];
                    for k in 0..4 {
                        p.line3(out, &world, corners[k], corners[(k + 1) % 4], c, 0.06);
                        p.line3(out, &world, corners[k], tip, c, 0.06);
                    }
                }
            }
        }

        // Bolts: they grow as they come (perspective) and glow brighter when close.
        for b in &w.bolts {
            let q = lerp3(xyz(b.prev), xyz(b.pos), alpha);
            let near = ((22.0 - (q[2] - s_ship)) / 22.0).clamp(0.0, 1.0);
            let c = fade(BOLT, 1.0 + 2.5 * near);
            p.solid(out, &world, q, &OCTAHEDRON, 0.32, time * 9.0, c);
        }

        // Shards.
        for sh in &w.shards {
            let q = xyz(sh.pos);
            p.solid(out, &world, q, &OCTAHEDRON, 0.3, time * 4.0, scene::shard());
        }

        // Shots: streaks along their flight.
        for b in &w.shots {
            let q = lerp3(xyz(b.prev), xyz(b.pos), alpha);
            let v = xyz(b.vel);
            let tail = [
                q[0] - v[0] * 0.018,
                q[1] - v[1] * 0.018,
                q[2] - v[2] * 0.018,
            ];
            p.line3(out, &world, tail, q, palette::BULLET, 0.07);
        }

        // Sparks: streaks along their velocity.
        for sp in &self.sparks {
            let k = sp.life / sp.total;
            let tail = [
                sp.pos[0] - sp.vel[0] * 0.03,
                sp.pos[1] - sp.vel[1] * 0.03,
                sp.pos[2] - sp.vel[2] * 0.03,
            ];
            p.line3(out, &world, tail, sp.pos, fade(sp.color, k), 0.03);
        }

        // The ship, its shadow on the wall, and the reticle.
        if w.phase == Phase::Playing {
            let q = lerp3(xyz(w.ship.prev), xyz(w.ship.pos), alpha);
            let blink = w.ship.invulnerable > 0.0
                && w.ship.roll.is_none()
                && (w.ship.invulnerable * 14.0).sin() < 0.0;
            let c = fade(palette::SHIP, if blink { 0.35 } else { 1.0 });
            let roll = w.ship.roll.map_or(0.0, |r| {
                r.dir * core::f32::consts::TAU * (r.t / 0.4).min(1.0)
            });
            let bank = -0.25 * (q[0] - w.ship.prev.e032()) / DT / 12.0;
            p.ship(out, &world, q, roll + bank, c);
            // The shadow: the line from a light on the axis just behind the ship, through
            // the ship, meets the wall's tangent plane under it.
            let r = (q[0] * q[0] + q[1] * q[1]).sqrt();
            if r > 0.3 {
                let light = Point::xyz(0.0, 0.0, q[2] - 2.0);
                let ray = light & Point::xyz(q[0], q[1], q[2]);
                let wall = Plane::from_normal([q[0] / r, q[1] / r, 0.0], RADIUS);
                let hit = (ray ^ wall).to_euclidean();
                // A small cross on the wall: along the track and around it.
                let (tx, ty) = (-q[1] / r * 0.5, q[0] / r * 0.5);
                let c = [0.6, 0.75, 1.0, 1.2];
                let h = |dx: f32, dy: f32, dz: f32| [hit[0] + dx, hit[1] + dy, hit[2] + dz];
                p.line3(out, &world, h(-tx, -ty, 0.0), h(tx, ty, 0.0), c, 0.05);
                p.line3(out, &world, h(0.0, 0.0, -0.7), h(0.0, 0.0, 0.7), c, 0.05);
            }
            self.reticle(out, &p, w, &world, q, aim, time);
        }

        // Popups, at their depth.
        for (pos, text, age, color) in &self.popups {
            if let Some(c) = p.camera(world(*pos)) {
                let [sx, sy] = p.screen(c);
                let size = (p.focal / c[2]).clamp(0.4, 1.6);
                let k = (1.0 - age / 1.1).max(0.0);
                scene::text(
                    out,
                    text,
                    sx,
                    sy + 1.0 + age * 2.0,
                    size,
                    fade(*color, k),
                    Align::Center,
                );
            }
        }
    }

    /// The reticle at the aim point, a lock bracket on the enemy nearest to it on screen, and
    /// a lead dot where the shots will cross that enemy's depth.
    #[allow(clippy::too_many_arguments)]
    fn reticle(
        &self,
        out: &mut Vec<LineInstance>,
        p: &Proj,
        w: &World,
        world: &dyn Fn([f32; 3]) -> [f32; 3],
        ship: [f32; 3],
        aim: [f32; 2],
        time: f32,
    ) {
        let hud = palette::HUD;
        let Some(c) = p.camera(world([aim[0], aim[1], ship[2] + AIM_DEPTH])) else {
            return;
        };
        let [x, y] = p.screen(c);
        let arm = 0.7;
        for (dx, dy) in [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)] {
            out.push(scene::seg(
                [x + dx * 0.35, y + dy * 0.35, x + dx * arm, y + dy * arm],
                hud,
                [0.05, 0.25, 0.3, 0.0],
                identity(),
            ));
        }
        // Lock: the enemy ahead nearest the reticle on screen.
        let mut best: Option<([f32; 2], f32, [f32; 3])> = None;
        for e in &w.enemies {
            let q = xyz(e.pos);
            if q[2] < ship[2] + 3.0 {
                continue;
            }
            if let Some(ec) = p.camera(world(q)) {
                let [ex, ey] = p.screen(ec);
                let d = ((ex - x).powi(2) + (ey - y).powi(2)).sqrt();
                if d < 3.0 && best.is_none_or(|b| d < b.1) {
                    best = Some(([ex, ey], d, q));
                }
            }
        }
        if let Some(([ex, ey], _, q)) = best {
            let r = 1.0 + 0.1 * (time * 10.0).sin();
            let c = [1.0, 0.9, 0.5, 2.0];
            for (sx, sy) in [(1.0, 1.0), (-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0)] {
                let (cx, cy) = (ex + sx * r, ey + sy * r);
                let st = [0.05, 0.2, 0.3, 0.0];
                out.push(scene::seg([cx, cy, cx - sx * 0.4, cy], c, st, identity()));
                out.push(scene::seg([cx, cy, cx, cy - sy * 0.4], c, st, identity()));
            }
            // Lead: where the shots cross the target's depth.
            let t = ((q[2] - ship[2]) / AIM_DEPTH).max(0.0);
            let lx = ship[0] + (aim[0] - ship[0]) * t;
            let ly = ship[1] + (aim[1] - ship[1]) * t;
            if let Some(lc) = p.camera(world([lx, ly, q[2]])) {
                let [px, py] = p.screen(lc);
                out.push(scene::seg(
                    [px - 0.12, py, px + 0.12, py],
                    c,
                    [0.12, 0.3, 0.4, 0.0],
                    identity(),
                ));
            }
        }
    }
}

/// Bolt colour: hot pink, the colour of danger.
const BOLT: [f32; 4] = [1.0, 0.25, 0.45, 3.0];

const OCTAHEDRON: [[f32; 6]; 12] = {
    const X: [f32; 3] = [1.0, 0.0, 0.0];
    const Y: [f32; 3] = [0.0, 1.0, 0.0];
    const Z: [f32; 3] = [0.0, 0.0, 1.0];
    const fn e(a: [f32; 3], b: [f32; 3], sa: f32, sb: f32) -> [f32; 6] {
        [
            a[0] * sa,
            a[1] * sa,
            a[2] * sa,
            b[0] * sb,
            b[1] * sb,
            b[2] * sb,
        ]
    }
    [
        e(X, Y, 1.0, 1.0),
        e(Y, X, 1.0, -1.0),
        e(X, Y, -1.0, -1.0),
        e(Y, X, -1.0, 1.0),
        e(X, Z, 1.0, 1.0),
        e(Y, Z, 1.0, 1.0),
        e(X, Z, -1.0, 1.0),
        e(Y, Z, -1.0, 1.0),
        e(X, Z, 1.0, -1.0),
        e(Y, Z, 1.0, -1.0),
        e(X, Z, -1.0, -1.0),
        e(Y, Z, -1.0, -1.0),
    ]
};

const SPIKES: [[f32; 6]; 7] = [
    [-1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, -1.0, 0.0, 0.0, 1.0],
    [-0.6, -0.6, -0.6, 0.6, 0.6, 0.6],
    [0.6, -0.6, -0.6, -0.6, 0.6, 0.6],
    [-0.6, 0.6, -0.6, 0.6, -0.6, 0.6],
    [0.6, 0.6, -0.6, -0.6, -0.6, 0.6],
];

const CUBE: [[f32; 6]; 12] = [
    [-1.0, -1.0, -1.0, 1.0, -1.0, -1.0],
    [1.0, -1.0, -1.0, 1.0, 1.0, -1.0],
    [1.0, 1.0, -1.0, -1.0, 1.0, -1.0],
    [-1.0, 1.0, -1.0, -1.0, -1.0, -1.0],
    [-1.0, -1.0, 1.0, 1.0, -1.0, 1.0],
    [1.0, -1.0, 1.0, 1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0, -1.0, 1.0, 1.0],
    [-1.0, 1.0, 1.0, -1.0, -1.0, 1.0],
    [-1.0, -1.0, -1.0, -1.0, -1.0, 1.0],
    [1.0, -1.0, -1.0, 1.0, -1.0, 1.0],
    [1.0, 1.0, -1.0, 1.0, 1.0, 1.0],
    [-1.0, 1.0, -1.0, -1.0, 1.0, 1.0],
];

/// The ship: an arrow into the screen, wings, a fin.
const SHIP: [[f32; 6]; 9] = [
    [0.0, 0.0, 1.3, 0.9, 0.0, -0.6],
    [0.0, 0.0, 1.3, -0.9, 0.0, -0.6],
    [0.9, 0.0, -0.6, 0.3, 0.0, -0.3],
    [-0.9, 0.0, -0.6, -0.3, 0.0, -0.3],
    [0.3, 0.0, -0.3, -0.3, 0.0, -0.3],
    [0.0, 0.0, 1.3, 0.0, 0.35, -0.4],
    [0.0, 0.35, -0.4, 0.3, 0.0, -0.3],
    [0.0, 0.35, -0.4, -0.3, 0.0, -0.3],
    [0.9, 0.0, -0.6, 1.1, 0.1, -0.9],
];

fn xyz(p: gax::pga3d::Point<(), f32>) -> [f32; 3] {
    [p.e032(), p.e013(), p.e021()]
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn lerp3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [
        lerp(a[0], b[0], t),
        lerp(a[1], b[1], t),
        lerp(a[2], b[2], t),
    ]
}

/// The projection: a camera motor and a focal length.
struct Proj {
    cam: Frame,
    focal: f32,
}

impl Proj {
    /// A world point in the camera's frame (`cam << p`), or `None` behind it.
    fn camera(&self, p: [f32; 3]) -> Option<[f32; 3]> {
        let c = (self.cam << Point::xyz(p[0], p[1], p[2])).to_euclidean();
        (c[2] > NEAR).then_some(c)
    }

    /// Screen units. The camera looks along `+z` with `+y` up, so its `+x` is on the left.
    fn screen(&self, c: [f32; 3]) -> [f32; 2] {
        [-c[0] / c[2] * self.focal, c[1] / c[2] * self.focal]
    }

    /// A segment between two camera-frame points: width and glow by depth, faded by fog.
    /// Objects see through more fog than the wall (`wall`), so they read at a distance.
    #[allow(clippy::too_many_arguments)]
    fn segment_in(
        &self,
        out: &mut Vec<LineInstance>,
        a: [f32; 3],
        b: [f32; 3],
        color: [f32; 4],
        width: f32,
        glow: f32,
        wall: bool,
    ) {
        let z = 0.5 * (a[2] + b[2]);
        let fog = if wall {
            let far = (1.0 - z / WALL_FOG).clamp(0.0, 1.0);
            // Right at the camera the wall fades too, so it never floods the screen.
            let near = ((z - 0.8) / 4.0).clamp(0.0, 1.0);
            far * far * far * near
        } else {
            (1.0 - (z - 45.0) / (FOG + 20.0 - 45.0)).clamp(0.0, 1.0)
        };
        if fog <= 0.0 {
            return;
        }
        let k = self.focal / z;
        let [ax, ay] = self.screen(a);
        let [bx, by] = self.screen(b);
        let (wmin, wmax) = if wall { (0.01, 0.1) } else { (0.035, 0.3) };
        out.push(scene::seg(
            [ax, ay, bx, by],
            fade(color, fog),
            [
                (width * k).clamp(wmin, wmax),
                (glow * k).clamp(0.08, 0.6),
                0.3,
                0.0,
            ],
            identity(),
        ));
    }

    fn segment(
        &self,
        out: &mut Vec<LineInstance>,
        a: [f32; 3],
        b: [f32; 3],
        color: [f32; 4],
        width: f32,
        glow: f32,
    ) {
        self.segment_in(out, a, b, color, width, glow, false);
    }

    /// A segment between two straightened points, clipped at the near plane.
    fn line3(
        &self,
        out: &mut Vec<LineInstance>,
        world: &dyn Fn([f32; 3]) -> [f32; 3],
        a: [f32; 3],
        b: [f32; 3],
        color: [f32; 4],
        width: f32,
    ) {
        let ca = (self.cam << world_point(world(a))).to_euclidean();
        let cb = (self.cam << world_point(world(b))).to_euclidean();
        let (ca, cb) = match (ca[2] > NEAR, cb[2] > NEAR) {
            (true, true) => (ca, cb),
            (false, false) => return,
            (true, false) => (ca, clip(ca, cb)),
            (false, true) => (clip(cb, ca), cb),
        };
        self.segment(out, ca, cb, color, width, width * 5.0);
    }

    /// A wireframe solid at `q`, in the track's frame there, spun about its local z.
    #[allow(clippy::too_many_arguments)]
    fn solid(
        &self,
        out: &mut Vec<LineInstance>,
        world: &dyn Fn([f32; 3]) -> [f32; 3],
        q: [f32; 3],
        edges: &[[f32; 6]],
        size: f32,
        spin: f32,
        color: [f32; 4],
    ) {
        let (c, s) = (spin.cos(), spin.sin());
        let (c2, s2) = ((spin * 0.7).cos(), (spin * 0.7).sin());
        let rot = |v: [f32; 3]| {
            // About z, then about x: a tumble.
            let (x, y, z) = (v[0] * c - v[1] * s, v[0] * s + v[1] * c, v[2]);
            let (y, z) = (y * c2 - z * s2, y * s2 + z * c2);
            [q[0] + x * size, q[1] + y * size, q[2] + z * size]
        };
        for e in edges {
            let a = rot([e[0], e[1], e[2]]);
            let b = rot([e[3], e[4], e[5]]);
            self.line3(out, world, a, b, color, 0.05);
        }
    }

    /// A ring across the tunnel (perpendicular to the track) at `q`.
    fn ring(
        &self,
        out: &mut Vec<LineInstance>,
        world: &dyn Fn([f32; 3]) -> [f32; 3],
        q: [f32; 3],
        r: f32,
        color: [f32; 4],
        n: usize,
    ) {
        for k in 0..n {
            let (a, b) = (
                k as f32 * core::f32::consts::TAU / n as f32,
                (k + 1) as f32 * core::f32::consts::TAU / n as f32,
            );
            self.line3(
                out,
                world,
                [q[0] + r * a.cos(), q[1] + r * a.sin(), q[2]],
                [q[0] + r * b.cos(), q[1] + r * b.sin(), q[2]],
                color,
                0.04,
            );
        }
    }

    /// The ship at `q`, rolled about its axis by `roll`.
    fn ship(
        &self,
        out: &mut Vec<LineInstance>,
        world: &dyn Fn([f32; 3]) -> [f32; 3],
        q: [f32; 3],
        roll: f32,
        color: [f32; 4],
    ) {
        let (c, s) = (roll.cos(), roll.sin());
        let place = |v: [f32; 3]| {
            let (x, y) = (v[0] * c - v[1] * s, v[0] * s + v[1] * c);
            [q[0] + x * 0.7, q[1] + y * 0.7, q[2] + v[2] * 0.7]
        };
        for e in &SHIP {
            let a = place([e[0], e[1], e[2]]);
            let b = place([e[3], e[4], e[5]]);
            self.line3(out, world, a, b, color, 0.06);
        }
        // The mirrored wingtip fin.
        let a = place([-0.9, 0.0, -0.6]);
        let b = place([-1.1, 0.1, -0.9]);
        self.line3(out, world, a, b, color, 0.06);
    }
}

fn world_point(p: [f32; 3]) -> Point<(), f32> {
    Point::xyz(p[0], p[1], p[2])
}

/// The point on `a → b` at the near plane (`a` in front, `b` behind).
fn clip(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    let t = (a[2] - NEAR) / (a[2] - b[2]);
    [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, NEAR]
}

/// Map the player's screen-space input into the tunnel's cross-section at the ship: where do
/// the tunnel's `x` and `y` axes point on screen (projected through the camera), and which
/// combination of them is the input? Also used for the mouse reticle at the aim depth.
pub fn screen_to_track(view: &View, w: &World, depth: f32, screen: [f32; 2]) -> [f32; 2] {
    let p = Proj {
        cam: view.cam,
        focal: view.focal,
    };
    let s = w.ship.s() + depth;
    let at = |x: f32, y: f32| {
        p.camera(w.track.point(x, y, s))
            .map(|c| p.screen(c))
            .unwrap_or([0.0, 0.0])
    };
    let o = at(0.0, 0.0);
    let ex = at(1.0, 0.0);
    let ey = at(0.0, 1.0);
    let (ax, ay) = (ex[0] - o[0], ex[1] - o[1]);
    let (bx, by) = (ey[0] - o[0], ey[1] - o[1]);
    let det = ax * by - ay * bx;
    if det.abs() < 1e-6 {
        return [0.0, 0.0];
    }
    let (dx, dy) = (screen[0] - o[0], screen[1] - o[1]);
    [(dx * by - dy * bx) / det, (ax * dy - ay * dx) / det]
}

/// Where a point in straightened coordinates is on screen (for the bot and tests).
pub fn track_to_screen(view: &View, w: &World, q: [f32; 3]) -> Option<[f32; 2]> {
    let p = Proj {
        cam: view.cam,
        focal: view.focal,
    };
    p.camera(w.track.point(q[0], q[1], q[2]))
        .map(|c| p.screen(c))
}

/// A direction on screen (movement) as a direction across the tunnel at the ship.
pub fn screen_dir_to_track(view: &View, w: &World, v: [f32; 2]) -> [f32; 2] {
    let base = screen_to_track(view, w, 0.0, [0.0, 0.0]);
    let tip = screen_to_track(view, w, 0.0, [v[0], v[1]]);
    let (x, y) = (tip[0] - base[0], tip[1] - base[1]);
    let l = (x * x + y * y).sqrt();
    let want = (v[0] * v[0] + v[1] * v[1]).sqrt().min(1.0);
    if l < 1e-6 {
        [0.0, 0.0]
    } else {
        [x / l * want, y / l * want]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The camera without roll looks where the target says, and its up stays in the world's
    /// vertical plane (no roll).
    #[test]
    fn the_camera_looks_ahead_without_rolling() {
        let mut w = World::new(4);
        for _ in 0..120 * 20 {
            w.tick(&crate::tunnel::Input::default());
        }
        let v = View::new(85.0);
        let cam = v.target(&w, 0.0);
        let f = cam >> Point::direction(0.0, 0.0, 1.0);
        let r = cam >> Point::direction(1.0, 0.0, 0.0);
        // Its right axis is horizontal.
        assert!(r.e013().abs() < 1e-4, "rolled: {}", r.e013());
        // It looks at a point down the track, which projects near the screen's centre.
        let s = w.ship.s();
        let ahead = w.track.point(0.0, 0.0, s + 16.0);
        let p = Proj {
            cam,
            focal: v.focal,
        };
        let c = p.camera(ahead).expect("in front");
        let [x, y] = p.screen(c);
        assert!(x.abs() < 3.0 && y.abs() < 4.0, "{x} {y}");
        let _ = f;
    }

    /// Input to the right on screen moves the ship to the right on screen, however the track
    /// has rolled.
    #[test]
    fn screen_directions_map_through_the_camera() {
        let mut w = World::new(8);
        for _ in 0..120 * 30 {
            w.tick(&crate::tunnel::Input::default());
        }
        let mut v = View::new(85.0);
        v.update(&w, 0.0, 1.0);
        let d = screen_dir_to_track(&v, &w, [1.0, 0.0]);
        let s = w.ship.s();
        let [x, y] = w.ship.xy();
        let p = Proj {
            cam: v.cam,
            focal: v.focal,
        };
        let a = p.screen(p.camera(w.track.point(x, y, s)).unwrap());
        let b = p.screen(p.camera(w.track.point(x + d[0], y + d[1], s)).unwrap());
        assert!(
            b[0] - a[0] > 0.1 && (b[1] - a[1]).abs() < 0.05 * (b[0] - a[0]),
            "{a:?} {b:?}"
        );
    }

    #[test]
    fn the_shadow_falls_on_the_wall() {
        // The meet of the light-to-ship line with the wall's tangent plane is on the wall.
        let q = [3.0f32, 1.0, 10.0];
        let r = (q[0] * q[0] + q[1] * q[1]).sqrt();
        let ray = Point::xyz(0.0, 0.0, 8.0) & Point::xyz(q[0], q[1], q[2]);
        let wall = Plane::from_normal([q[0] / r, q[1] / r, 0.0], RADIUS);
        let h = (ray ^ wall).to_euclidean();
        let rh = (h[0] * h[0] + h[1] * h[1]).sqrt();
        assert!((rh - RADIUS).abs() < 1e-3, "{h:?}");
        assert!(h[2] > q[2], "the shadow falls ahead");
    }
}
