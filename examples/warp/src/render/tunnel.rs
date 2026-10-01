//! Drawing the Tunnel: a camera motor in 3D, a projective projection, and everything as lines
//! in the HUD's screen units (the line renderer is 2D).
//!
//! * The camera is `Motor::look_at` from just behind the ship, near the axis, down the track;
//!   by default its up is the world's (it does not roll with the tunnel, a comfort setting).
//!   It follows on a motor spring (screw interpolation towards the target).
//! * `cam << p` takes a world point into the camera's frame. Objects go there as maps: one
//!   4x4 per object, `(~cam placement) >> Point::slot()`, built from the motors once and
//!   applied to every vertex (a wall ring's nodes in a batch, `of_slice`); the track's frame
//!   is interpolated once per object, not per point. The projection to the screen is
//!   projective geometry: the camera-space point becomes the PGA2D point with the depth as its
//!   weight, and `to_euclidean` is the perspective divide. Segments crossing the near plane
//!   are cut by their meet with it.
//! * Depth reads through fog, widths and glows that scale with `1/z`, bolts that glow brighter
//!   as they come close, the ship's shadow on the wall (the meet of the light-to-ship line with
//!   the wall's tangent plane, which is `wall point | radial line`), and the reticle's lock and
//!   lead.
//! * The reticle aims through the same camera, backwards: a screen point is a ray from the eye,
//!   which meets the tunnel's cross-section at the aim depth; `Track::straighten` takes the
//!   meet back into the simulation's coordinates.
//! * A singularity bends light. Everything behind it is lensed as by a point mass: a point
//!   seen at `β` from it on screen appears at the outer root of the lens equation
//!   `θ² - βθ - θ_E² = 0`, a scaling about the singularity's screen point, so what lies behind
//!   it is pushed out into an Einstein ring round a dark shadow. Its accretion disc is beamed
//!   (the side coming at you brighter and bluer, the other dimmer and redder), and light near
//!   it is redshifted: a turn of its hue in OkLab, and dimmer.

use super::LineInstance;
use super::font::Align;
use super::scene::{self, palette};
use crate::light::{self, Light};
use crate::sim::body::identity;
use crate::sim::rng::Rng;
use crate::sim::{DT, Phase};
use crate::tunnel::lattice::{AROUND, RADIUS, RINGS};
use crate::tunnel::track::{Frame, Track, interpolate};
use crate::tunnel::{
    AIM_DEPTH, Event, Foe, GATE_RADIUS, GateState, P, World, about_axis, arc, dir, foot, off_axis,
};
use gax::batch::BatchOf;
use gax::pga2d::Point as Point2;
use gax::pga3d::{Motor, Plane, Point};

/// How far the fog lets objects be seen.
const FOG: f32 = 90.0;
/// The wall fades sooner: its lines crowd towards the vanishing point, where enemies come
/// from, and must not glare there.
const WALL_FOG: f32 = 58.0;
const NEAR: f32 = 0.25;
/// The camera's image plane, through its eye, facing along its view (`z = 0` in its frame).
const IMAGE_PLANE: Plane<(), f32> = Plane::new(0.0, 0.0, 1.0, 0.0);

/// A particle in straightened coordinates.
#[derive(Clone, Copy, Debug)]
struct Spark {
    pos: P,
    vel: P,
    light: Light,
    life: f32,
    total: f32,
}

/// The Tunnel's view: camera, effects, the reticle, and the projection.
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
    /// The reticle on screen (HUD units).
    pub reticle: Point2<(), f32>,
    /// The enemy the reticle has locked onto.
    pub lock: Option<u32>,
    /// The lock-on margin beyond an enemy's apparent size (the aim-assist setting).
    pub assist: f32,
    /// Singularities lens what lies behind them (off with reduced motion).
    pub lensing: bool,
    /// A collapse's shock ripple: where, and how long ago.
    shock: Option<(P, f32)>,
    /// The singularities as the camera sees them, for this frame.
    lenses: Vec<Lens>,
    sparks: Vec<Spark>,
    dust: Vec<P>,
    popups: Vec<(P, String, f32, Light)>,
    rng: Rng,
    started: bool,
}

/// `a` to `b` by `t` (points of weight 1, or directions).
fn lerp(a: P, b: P, t: f32) -> P {
    a + (b - a) * t
}

/// Bolts: hot pink, the colour of danger.
const BOLT: Light = light::light(1.0, 0.25, 0.45, 3.0);
/// Gates: warm gold, a pickup's colour, never an enemy's.
pub const GATE_LIGHT: Light = light::light(1.0, 0.8, 0.32, 2.2);
/// How strongly a singularity lenses (its Einstein radius² per unit of strength, focal length²
/// and inverse depth).
const LENS: f32 = 1.0;

impl View {
    /// A view with a field of view in degrees.
    pub fn new(fov_degrees: f32) -> View {
        let mut rng = Rng::new(0xd057);
        let dust = (0..420)
            .map(|_| {
                let r = RADIUS * gax::Real::sqrt(rng.unit()) * 0.95;
                crate::tunnel::around(r, rng.angle(), rng.range(-5.0, FOG))
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
            reticle: Point2::xy(0.0, 0.0),
            lock: None,
            assist: 1.1,
            lensing: true,
            shock: None,
            lenses: Vec::new(),
            sparks: Vec::new(),
            dust,
            popups: Vec::new(),
            rng,
            started: false,
        }
    }

    /// The focal length for a vertical field of view (the HUD is 36 units tall): the
    /// screen's half height over the tangent of half the angle.
    pub fn focal_for(fov_degrees: f32) -> f32 {
        let half = fov_degrees.to_radians() * 0.5;
        // tan = sin / cos: the phasor at the half angle, height over width.
        let d = crate::geom::phasor(half);
        18.0 * d.e20() / d.e01()
    }

    /// The camera the view wants: just behind the ship, a little above and close to the
    /// axis, looking down the track.
    fn target(&self, w: &World, alpha: f32) -> Frame {
        let ship = lerp(w.ship.prev, w.ship.pos, alpha);
        let (axis_point, across) = (foot(ship), ship - foot(ship));
        let eye = axis_point + across * 0.3 + dir(0.0, 0.9, -7.5);
        let look = axis_point + across * 0.1 + dir(0.0, 0.0, 20.0);
        let up = if self.follow_roll {
            w.track.frame(arc(eye)) >> dir(0.0, 1.0, 0.0)
        } else {
            dir(0.0, 1.0, 0.0)
        };
        Motor::look_at(w.track.place(eye), w.track.place(look), up)
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
            p.pos += p.vel * dt;
            p.vel = p.vel * (1.0 - 1.8 * dt);
            p.life -= dt;
        }
        self.sparks.retain(|p| p.life > 0.0);
        for p in &mut self.popups {
            p.2 += dt;
        }
        self.popups.retain(|p| p.2 < 1.1);
        if let Some((_, t)) = &mut self.shock {
            *t += dt;
            if *t > 0.9 {
                self.shock = None;
            }
        }
        self.lenses = self.lenses_for(self.cam, w);
        // Dust behind the ship comes round again ahead.
        let s = w.ship.s();
        for d in &mut self.dust {
            if arc(*d) < s - 5.0 {
                *d += dir(0.0, 0.0, FOG + 5.0);
            }
        }
    }

    fn proj(&self) -> Proj {
        Proj::new(self.cam, self.focal, self.lenses.clone())
    }

    /// The singularities in front of the camera `cam`, as lenses.
    fn lenses_for(&self, cam: Frame, w: &World) -> Vec<Lens> {
        if !self.lensing {
            return Vec::new();
        }
        let p = Proj::new(cam, self.focal, Vec::new());
        w.wells()
            .filter_map(|(q, strength)| {
                let c = p.camera(w.track.place(q))?;
                let depth = p.depth(c);
                Some(Lens {
                    centre: p.screen(c),
                    depth,
                    k: LENS * strength * self.focal * self.focal / depth.max(2.0),
                })
            })
            .collect()
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

    /// Sparks leaving `pos` in random directions: `(0, 0, v)` turned by a random rotation.
    fn burst(&mut self, pos: P, light: Light, n: usize, speed: f32, life: f32) {
        for _ in 0..n {
            let turn = Motor::rotation_about(
                self.rng.range(-1.0, 1.0),
                self.rng.range(-1.0, 1.0),
                self.rng.range(-1.0, 1.0),
                self.rng.angle(),
            );
            let v = turn >> dir(0.0, 0.0, self.rng.range(0.3, 1.0) * speed);
            let l = self.rng.range(0.5, 1.0) * life;
            self.sparks.push(Spark {
                pos,
                vel: v,
                light,
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
                    let big = matches!(foe, Foe::Turret | Foe::Singularity);
                    self.burst(pos, c, if big { 160 } else { 70 }, 11.0, 0.9);
                    self.burst(pos, light::light(1.0, 1.0, 1.0, 3.0), 12, 16.0, 0.3);
                    self.shake = self.shake.max(if big { 0.5 } else { 0.15 });
                    if points >= 250 {
                        self.popups.push((pos, points.to_string(), 0.0, c));
                    }
                    if foe == Foe::Singularity {
                        // The collapse: a shock ripple from it, a flash, and a storm of light.
                        self.shock = Some((pos, 0.0));
                        self.flash = self.flash.max(0.45);
                        self.shake = self.shake.max(1.0);
                        self.burst(pos, light::whiten(c, 0.5), 700, 30.0, 1.4);
                    }
                }
                Event::Wall { pos } => self.burst(pos, palette::BULLET, 4, 4.0, 0.2),
                Event::Bolt { pos } => self.burst(pos, BOLT, 10, 3.0, 0.3),
                Event::Dive { pos } => {
                    self.burst(pos, scene::color(Foe::Drone.kind()), 14, 5.0, 0.3);
                }
                Event::Pickup { pos, mult } => {
                    self.burst(pos, scene::shard(), 8, 4.0, 0.3);
                    if mult % 10 == 0 {
                        self.popups
                            .push((pos, format!("X{mult}"), 0.0, scene::shard()));
                    }
                }
                Event::Death { pos } => {
                    let k = 0.35 + 0.65 * self.flash_scale;
                    self.burst(pos, light::fade(palette::SHIP, k), 900, 22.0, 1.8);
                    self.shake = 1.6;
                    self.flash = 0.7;
                }
                Event::Bomb { pos } => {
                    let k = 0.35 + 0.65 * self.flash_scale;
                    for step in 0..6 {
                        let p = foot(pos) + dir(0.0, 0.0, 6.0 + 9.0 * step as f32);
                        let c = light::light(0.55, 0.8, 1.0, 3.0 * k);
                        self.burst(p, c, 220, 26.0, 1.0);
                    }
                    self.shake = 1.2;
                    self.flash = 0.5;
                }
                Event::Gate { pos, points, chain } => {
                    self.burst(pos, GATE_LIGHT, 40, 9.0, 0.5);
                    self.popups
                        .push((pos, format!("+{points}"), 0.0, GATE_LIGHT));
                    if chain >= 3 && chain % 3 == 0 {
                        let p = pos + dir(0.0, 1.2, 0.0);
                        self.popups
                            .push((p, format!("CHAIN X{chain}"), 0.0, GATE_LIGHT));
                    }
                }
                Event::GateMiss { pos } => {
                    let grey = light::desaturate(GATE_LIGHT, 1.0);
                    self.burst(pos, light::fade(grey, 0.5), 20, 4.0, 0.4);
                }
                Event::Absorb { pos, .. } => {
                    let c = scene::color(Foe::Singularity.kind());
                    self.burst(pos, light::whiten(c, 0.4), 30, 5.0, 0.3);
                }
                Event::Burst { pos } => {
                    let c = scene::color(Foe::Singularity.kind());
                    self.burst(pos, c, 300, 18.0, 1.0);
                    self.shake = self.shake.max(0.8);
                }
                Event::Slingshot { pos } => {
                    let c = light::whiten(scene::color(Foe::Singularity.kind()), 0.3);
                    self.popups
                        .push((pos + dir(0.0, 1.4, 0.0), "SLINGSHOT".into(), 0.0, c));
                }
                Event::Roll { .. }
                | Event::Warn { .. }
                | Event::Fire { .. }
                | Event::Respawn
                | Event::Extra { .. }
                | Event::GameOver => {}
            }
        }
    }

    /// Post settings for the frame (`aspect`: the screen's width over its height).
    pub fn post(&self, w: &World, aspect: f32) -> super::PostSettings {
        // The collapse's ripple, from where the singularity was on screen.
        let shock = self.shock.filter(|_| self.lensing).and_then(|(q, t)| {
            let [x, y] = self.on_screen(w, q)?.to_euclidean();
            let uv = [x / (36.0 * aspect) + 0.5, 0.5 - y / 36.0];
            let strength = 0.05 * (1.0 - t / 0.9) * (0.4 + 0.6 * self.flash_scale);
            Some((uv, t * 1.1, strength))
        });
        super::PostSettings {
            bloom: 0.34,
            exposure: 1.0 + self.flash * 1.5 * self.flash_scale,
            vignette: 0.45,
            grain: 0.012,
            aberration: 0.007,
            saturation: 1.35,
            shock,
        }
    }

    /// Watching a replay: the reticle over the replay's aim point, locked onto the enemy
    /// there.
    pub fn follow(&mut self, w: &World, aim: P) {
        if let Some(q) = self.on_screen(w, aim) {
            self.reticle = q;
        }
        self.lock = w
            .enemies
            .iter()
            .find(|e| (e.pos & aim).norm() < 0.5)
            .map(|e| e.id);
    }

    /// The reticle at a screen point: lock onto the enemy under it (the nearest on screen
    /// within reach), or aim where the ray through it meets the cross-section at
    /// `AIM_DEPTH`. Returns the aim point for the simulation.
    pub fn aim(&mut self, w: &World, reticle: Point2<(), f32>) -> P {
        self.reticle = reticle;
        let p = self.proj();
        let s = w.ship.s();
        let lock = w
            .enemies
            .iter()
            .filter(|e| arc(e.pos) > s + 3.0)
            .filter_map(|e| {
                let c = p.camera(w.track.place(e.pos))?;
                // Where it is seen (a singularity's lens moves what lies behind it).
                let on_screen = p.lensed(c);
                // Within reach on screen: a margin plus the enemy's own apparent size.
                let reach = self.assist + e.radius() * self.focal / p.depth(c);
                let d = (on_screen & reticle).norm();
                (d < reach).then_some((e, d))
            })
            .min_by(|a, b| a.1.total_cmp(&b.1));
        self.lock = lock.map(|(e, _)| e.id);
        match lock {
            Some((e, _)) => e.pos,
            None => self.through(w, reticle, s + AIM_DEPTH),
        }
    }

    /// Where the ray from the eye through the screen point `at` meets the tunnel's
    /// cross-section at arc length `s`, in straightened coordinates.
    pub fn through(&self, w: &World, at: Point2<(), f32>, s: f32) -> P {
        let p = self.proj();
        let ray = p.ray(at);
        let section = w.track.frame(s) >> Plane::from_normal([0.0, 0.0, 1.0], 0.0);
        let hit = ray ^ section;
        w.track.straighten(hit.unitized(), s)
    }

    /// A direction on screen (movement, length up to 1) as a direction across the tunnel at
    /// the ship: how the ship's cross-section is met by the ray through its screen point, as
    /// that point moves along `v`.
    ///
    /// In the frame of the cross-section (`z = 0` there, `x` and `y` straightened), the ray
    /// along a camera-frame direction meets it at `hit.of(d)`: a projective map. The screen
    /// point `(x, y)` is the direction `(-x/f, y/f, 1)`, so moving it at `v` moves the hit
    /// as `(a + t b) / (w_a + t w_b)`, whose direction at the ship is `b w_a - a w_b`: the
    /// exact differential, from two applications of one map.
    pub fn across(&self, w: &World, v: Point2<(), f32>) -> P {
        let p = self.proj();
        let s = w.ship.s();
        let Some(c) = p.camera(w.track.place(w.ship.pos)) else {
            return dir(0.0, 0.0, 0.0);
        };
        let cam = w.track.frame(s).reverse() * p.cam;
        let eye = cam >> Point::xyz(0.0, 0.0, 0.0);
        let section = Plane::from_normal([0.0, 0.0, 1.0], 0.0);
        let hit: CamMap = (eye & (cam >> Point::slot())) ^ section;
        let z = c.e021();
        let (a, b) = (
            hit.of(dir(c.e032() / z, c.e013() / z, 1.0)),
            hit.of(dir(-v.e20() / self.focal, v.e01() / self.focal, 0.0)),
        );
        let d = b * a.e123() - a * b.e123();
        let d = dir(d.e032(), d.e013(), 0.0);
        let n = d.ideal_norm();
        if n < 1e-6 {
            dir(0.0, 0.0, 0.0)
        } else {
            d * (v.ideal_norm().min(1.0) / n)
        }
    }

    /// Where a world point is for the ears: in the camera's frame (`cam << p`), to the right
    /// as `x` and ahead as `y` (the camera's `+x` is on the left).
    pub fn listen(&self, p: P) -> Point2<(), f32> {
        let c = (self.cam << p).unitized();
        Point2::xy(-c.e032(), c.e021())
    }

    /// Where a straightened point is seen on screen (lensed), if in front of the camera.
    pub fn on_screen(&self, w: &World, q: P) -> Option<Point2<(), f32>> {
        let p = self.proj();
        p.camera(w.track.place(q)).map(|c| p.lensed(c))
    }

    /// The screen point of the ship (for the pad's reticle, which sits ahead of it).
    pub fn ship_on_screen(&self, w: &World) -> Point2<(), f32> {
        let p = self.proj();
        p.camera(w.track.place(w.ship.pos))
            .map_or(Point2::xy(0.0, 0.0), |c| p.screen(c))
    }

    /// Build the frame's lines (screen units, for the HUD camera).
    pub fn draw(&mut self, w: &World, alpha: f32, time: f32, out: &mut Vec<LineInstance>) {
        out.clear();
        let cam = self.shaken();
        let p = Proj::new(cam, self.focal, self.lenses_for(cam, w));
        let track = &w.track;
        let place = |q: P| track.place(q);
        let s_ship = arc(lerp(w.ship.prev, w.ship.pos, alpha));
        // Light from near a singularity is redshifted: by how much, from `q`.
        let wells: Vec<(P, f32)> = w.wells().collect();
        let shift = |q: P| {
            wells
                .iter()
                .map(|&(c, strength)| {
                    let d = (q & c).norm();
                    1.1 * strength / (1.0 + d * d / 4.0)
                })
                .fold(0.0f32, f32::max)
        };
        let tinted = |c: Light, z: f32| if z > 0.03 { light::redshift(c, z) } else { c };

        // The wall: one camera map per ring, from its placement, applied to the ring's nodes
        // in a batch; then the lines between neighbours.
        let lat = &w.lattice;
        let mut nodes: Vec<Option<(P, f32, f32)>> = vec![None; RINGS * AROUND];
        let mut seen = [Point::zero(); AROUND];
        for r in 0..RINGS {
            let view = p.to_camera(track.placement(lat.ring_s(r)));
            let ring: [P; AROUND] = core::array::from_fn(|j| lat.node(r, j));
            view.of_slice(&ring, &mut seen);
            for (j, (&node, &c)) in ring.iter().zip(&seen).enumerate() {
                let z = if wells.is_empty() { 0.0 } else { shift(node) };
                nodes[r * AROUND + j] = (p.depth(c) > NEAR).then(|| (c, lat.strain(r, j), z));
            }
        }
        let grid = palette::GRID;
        for r in 0..RINGS {
            for j in 0..AROUND {
                let Some((a, sa, za)) = nodes[r * AROUND + j] else {
                    continue;
                };
                let mut edge = |b: Option<(P, f32, f32)>, bright: f32| {
                    if let Some((b, sb, zb)) = b {
                        let glow = 1.0 + 4.0 * (sa + sb).min(1.5);
                        let c = tinted(light::fade(grid, 1.5 * bright * glow), 0.5 * (za + zb));
                        p.segment(out, a, b, c, 0.03, 0.22, Fog::Wall);
                    }
                };
                edge(nodes[r * AROUND + (j + 1) % AROUND], 1.0);
                if r + 1 < RINGS {
                    edge(nodes[(r + 1) * AROUND + j], 0.8);
                }
            }
        }

        // Dust, streaked along the track by the speed.
        let streak = dir(0.0, 0.0, 0.012 * w.ship.speed);
        for &d in &self.dust {
            if arc(d) < s_ship - 2.0 {
                continue;
            }
            let dust = light::light(0.6, 0.7, 1.0, 0.35);
            p.streak(out, track, d - streak, d, dust, 0.02, Fog::Wall);
        }

        // Warp-ins: rings contracting onto where a spawn lands.
        for pd in &w.pending {
            let k = (pd.t / 0.8).clamp(0.0, 1.0);
            let c = light::fade(scene::color(pd.foe.kind()), 0.8);
            p.ring(
                out,
                &p.at(track, pd.pos),
                pd.pos,
                0.4 + 2.0 * k,
                c,
                10,
                0.04,
            );
        }

        // Gates: the next one bright and pulsing, the rest of the slalom dimmer and joined by
        // a faint path; flown through, a ring flung outwards; missed, grey and gone.
        let mut open = 0;
        let mut last: Option<P> = None;
        for g in &w.gates {
            match g.state {
                GateState::Open => {
                    let k = if open == 0 {
                        1.2 + 0.4 * crate::signal::wave(time * 6.0)
                    } else {
                        0.55
                    };
                    open += 1;
                    let c = light::fade(GATE_LIGHT, k);
                    let view = p.at(track, g.pos);
                    p.ring(out, &view, g.pos, GATE_RADIUS, c, 32, 0.09);
                    p.ring(
                        out,
                        &view,
                        g.pos,
                        GATE_RADIUS * 0.8,
                        light::fade(c, 0.3),
                        32,
                        0.03,
                    );
                    if let Some(prev) = last {
                        let path = light::fade(GATE_LIGHT, 0.15);
                        p.line(out, place(prev), place(g.pos), path, 0.02, Fog::Objects);
                    }
                    last = Some(g.pos);
                }
                GateState::Passed if g.t < 0.5 => {
                    let c = light::fade(light::whiten(GATE_LIGHT, 0.6), 1.5 * (1.0 - g.t / 0.5));
                    p.ring(
                        out,
                        &p.at(track, g.pos),
                        g.pos,
                        GATE_RADIUS * (1.0 + 3.0 * g.t),
                        c,
                        32,
                        0.07,
                    );
                }
                GateState::Missed if g.t < 0.6 => {
                    let grey = light::desaturate(GATE_LIGHT, 1.0);
                    let c = light::fade(grey, 0.5 * (1.0 - g.t / 0.6));
                    p.ring(out, &p.at(track, g.pos), g.pos, GATE_RADIUS, c, 32, 0.04);
                }
                _ => {}
            }
        }

        // Enemies.
        let eye = foot(w.ship.pos) + dir(0.0, 0.9, -7.5);
        for e in &w.enemies {
            let q = lerp(e.prev, e.pos, alpha);
            let c = scene::color(e.foe.kind());
            let c = light::fade(light::whiten(c, e.flash), 1.0 + e.flash);
            let c = if e.foe == Foe::Singularity {
                c
            } else {
                tinted(c, shift(q))
            };
            let spin = e.age * 2.0 + e.id as f32;
            let tumble = Motor::rotation_about(0.0, 0.0, 1.0, spin)
                * Motor::rotation_about(1.0, 0.0, 0.0, 0.7 * spin);
            match e.foe {
                Foe::Drone => p.solid(out, &p.object(track, q, tumble), &OCTAHEDRON, 0.7, c),
                Foe::Mine => {
                    let pulse = 1.0 + 0.15 * crate::signal::wave(e.age * 7.0);
                    p.solid(out, &p.object(track, q, tumble), &SPIKES, 0.9 * pulse, c);
                    let inner = p.object(track, q, tumble.reverse());
                    p.solid(out, &inner, &CUBE, 0.35, c);
                }
                Foe::Turret => {
                    // A pyramid on the wall pointing at the axis, turned about the axis to its
                    // place; it glows before it fires.
                    let charge = ((0.35 - e.timer) / 0.35).clamp(0.0, 1.0);
                    let c = light::fade(c, 1.0 + 2.0 * charge);
                    let model = p.object(track, q, about_axis(e.angle));
                    p.solid(out, &model, &PYRAMID, 1.0, c);
                }
                Foe::Serpent => {
                    // The head, pointing where it swims; the body, diamonds on the helix it has
                    // just swum, joined by its spine.
                    let pose = Motor::rotation_between(dir(0.0, 0.0, 1.0), e.vel);
                    p.solid(out, &p.object(track, q, pose), &HEAD, 0.95, c);
                    let base = scene::color(Foe::Serpent.kind());
                    let n = e.body.len().max(1) as f32;
                    let mut last = q;
                    for (k, seg) in e.body.iter().enumerate() {
                        let sq = lerp(seg.prev, seg.pos, alpha);
                        let sc = tinted(light::fade(base, 1.0 - 0.45 * k as f32 / n), shift(sq));
                        let spin =
                            Motor::rotation_about(0.0, 0.0, 1.0, e.age * 3.0 + k as f32 * 0.5);
                        p.solid(out, &p.object(track, sq, spin), &OCTAHEDRON, 0.5, sc);
                        p.line(
                            out,
                            place(last),
                            place(sq),
                            light::fade(sc, 0.5),
                            0.03,
                            Fog::Objects,
                        );
                        last = sq;
                    }
                }
                Foe::Singularity => singularity(out, &p, &p.at(track, q), eye, e, q, c, time),
            }
        }

        // Bolts: they grow as they come (perspective) and glow brighter when close.
        for b in &w.bolts {
            let q = lerp(b.prev, b.pos, alpha);
            let bolt = tinted(BOLT, shift(q));
            let near = ((22.0 - (arc(q) - s_ship)) / 22.0).clamp(0.0, 1.0);
            let spin = Motor::rotation_about(0.0, 0.0, 1.0, time * 9.0);
            p.solid(
                out,
                &p.object(track, q, spin),
                &OCTAHEDRON,
                0.32,
                light::fade(bolt, 1.0 + 2.5 * near),
            );
        }

        // Shards.
        for sh in &w.shards {
            let spin = Motor::rotation_about(0.0, 1.0, 0.0, time * 4.0);
            let model = p.object(track, sh.pos, spin);
            p.solid(out, &model, &OCTAHEDRON, 0.3, scene::shard());
        }

        // Shots: streaks along their flight (bent ones curve round a singularity).
        for b in &w.shots {
            let q = lerp(b.prev, b.pos, alpha);
            let c = tinted(palette::BULLET, shift(q));
            p.streak(out, track, q - b.vel * 0.018, q, c, 0.07, Fog::Objects);
        }

        // Sparks: streaks along their velocity.
        for sp in &self.sparks {
            let c = light::fade(sp.light, sp.life / sp.total);
            p.streak(
                out,
                track,
                sp.pos - sp.vel * 0.03,
                sp.pos,
                c,
                0.03,
                Fog::Objects,
            );
        }

        // The ship, its shadow on the wall, and the reticle.
        if w.phase == Phase::Playing {
            let q = lerp(w.ship.prev, w.ship.pos, alpha);
            let blink = w.ship.invulnerable > 0.0
                && w.ship.roll.is_none()
                && crate::signal::wave(w.ship.invulnerable * 14.0) < 0.0;
            // Close to the camera, the ship needs less light than on the Plane.
            let c = light::fade(palette::SHIP, if blink { 0.2 } else { 0.55 });
            let roll = w.ship.roll.map_or(0.0, |r| {
                r.dir * core::f32::consts::TAU * (r.t / 0.4).min(1.0)
            });
            // Banking into the movement across the tunnel.
            let sideways = (w.ship.pos - w.ship.prev).e032() / DT / 12.0;
            let pose = Motor::rotation_about(0.0, 0.0, 1.0, roll - 0.25 * sideways);
            p.solid(out, &p.object(track, q, pose), &SHIP, 0.7, c);
            // The shadow: the line from a light on the axis just behind the ship, through the
            // ship, meets the wall's tangent plane under it (the plane through the wall point
            // orthogonal to the radial line).
            let r = off_axis(q);
            if r > 0.3 {
                let base = foot(q);
                let wall_point = base + (q - base) * (RADIUS / r);
                let radial = base & q;
                let wall = wall_point | radial;
                let light_at = base - dir(0.0, 0.0, 2.0);
                let hit = (light_at & q) ^ wall;
                let hit = hit.unitized();
                // A small cross on the wall: along the track and around it.
                let around = (q - base) * (0.5 / r);
                let around = about_axis(core::f32::consts::FRAC_PI_2) >> around;
                let along = dir(0.0, 0.0, 0.7);
                let shade = light::light(0.6, 0.75, 1.0, 1.2);
                p.line(
                    out,
                    place(hit - around),
                    place(hit + around),
                    shade,
                    0.05,
                    Fog::Objects,
                );
                p.line(
                    out,
                    place(hit - along),
                    place(hit + along),
                    shade,
                    0.05,
                    Fog::Objects,
                );
            }
            self.draw_reticle(out, &p, w, time);
        }

        // Popups, at their depth.
        for (pos, text, age, color) in &self.popups {
            if let Some(c) = p.camera(place(*pos)) {
                let [sx, sy] = p.screen(c).to_euclidean();
                let size = (self.focal / p.depth(c)).clamp(0.4, 1.6);
                let k = (1.0 - age / 1.1).max(0.0);
                let y = sy + 1.0 + age * 2.0;
                scene::text(
                    out,
                    text,
                    sx,
                    y,
                    size,
                    light::fade(*color, k),
                    Align::Center,
                );
            }
        }
    }

    /// The reticle, a lock bracket on the locked enemy, and a lead dot where the shots will
    /// cross its depth.
    fn draw_reticle(&self, out: &mut Vec<LineInstance>, p: &Proj, w: &World, time: f32) {
        let hud = palette::HUD;
        let style = [0.05, 0.25, 0.3, 0.0];
        // Four arms: a short segment turned a quarter at a time about the reticle.
        let o = self.reticle;
        let quarter = gax::pga2d::Motor::rotation(o, core::f32::consts::FRAC_PI_2);
        let (mut a, mut b) = (
            o + Point2::direction(0.35, 0.0),
            o + Point2::direction(0.7, 0.0),
        );
        for _ in 0..4 {
            out.push(scene::seg(a, b, hud, style, identity()));
            (a, b) = (quarter >> a, quarter >> b);
        }
        let Some(e) = self
            .lock
            .and_then(|id| w.enemies.iter().find(|e| e.id == id))
        else {
            return;
        };
        let Some(c) = p.camera(w.track.place(e.pos)) else {
            return;
        };
        let centre = p.screen(c);
        let r = 1.0 + 0.1 * crate::signal::wave(time * 10.0);
        let lock = light::light(1.0, 0.9, 0.5, 2.0);
        let st = [0.05, 0.2, 0.3, 0.0];
        // Four corner brackets, each a turn of the first.
        let quarter = gax::pga2d::Motor::rotation(centre, core::f32::consts::FRAC_PI_2);
        let corner = centre + Point2::direction(r, r);
        let (mut k, mut u, mut v) = (
            corner,
            corner - Point2::direction(0.4, 0.0),
            corner - Point2::direction(0.0, 0.4),
        );
        for _ in 0..4 {
            out.push(scene::seg(k, u, lock, st, identity()));
            out.push(scene::seg(k, v, lock, st, identity()));
            (k, u, v) = (quarter >> k, quarter >> u, quarter >> v);
        }
        // Lead: where the enemy will be when the shots cross its depth.
        let ship = w.ship.pos;
        let closing = (w.ship.speed + 60.0 - e.vel.e021()).max(10.0);
        let t = (arc(e.pos) - arc(ship)) / closing;
        if let Some(lc) = p.camera(w.track.place(e.pos + e.vel * t)) {
            let d = p.screen(lc);
            let (a, b) = (
                d - Point2::direction(0.12, 0.0),
                d + Point2::direction(0.12, 0.0),
            );
            out.push(scene::seg(a, b, lock, [0.12, 0.3, 0.4, 0.0], identity()));
        }
    }
}

/// The cosine of the angle between two directions: the inner product of the planes
/// orthogonal to them, over their norms.
fn cosine(a: P, b: P) -> f32 {
    let (pa, pb) = (Plane::orthogonal_to(a), Plane::orthogonal_to(b));
    (pa | pb).s() / (pa.norm() * pb.norm()).max(1e-9)
}

/// A singularity: an accretion disc, nearly edge on and slowly precessing, whose rings turn
/// faster inside; each piece of it beamed by its orbital velocity against the line of sight
/// to the eye (coming at you, blueshifted and brighter; going away, redshifted and dimmer).
/// The disc's far half lies behind the singularity and is lensed up over its shadow. Round
/// the shadow, the photon ring.
#[allow(clippy::too_many_arguments)]
fn singularity(
    out: &mut Vec<LineInstance>,
    p: &Proj,
    view: &CamMap,
    eye: P,
    e: &crate::tunnel::Enemy,
    q: P,
    c: Light,
    time: f32,
) {
    let grow = 1.0 + 0.1 * e.mass;
    let tilt = Motor::rotation_about(0.0, 0.0, 1.0, 0.15 * time)
        * Motor::rotation_about(1.0, 0.0, 0.0, 1.25);
    let quarter = Motor::rotation_about(0.0, 0.0, 1.0, core::f32::consts::FRAC_PI_2);
    const N: usize = 36;
    let step = Motor::rotation_about(0.0, 0.0, 1.0, core::f32::consts::TAU / N as f32);
    for (k, r) in [1.9f32, 2.5, 3.2].into_iter().enumerate() {
        let heat = 1.0 - 0.3 * k as f32;
        let base = light::fade(light::whiten(c, 0.55 * heat), heat);
        let spin = time * (2.4 - 0.6 * k as f32);
        let mut radial = Motor::rotation_about(0.0, 0.0, 1.0, spin) >> dir(r * grow, 0.0, 0.0);
        for _ in 0..N {
            let next = step >> radial;
            let a = q + (tilt >> radial);
            let b = q + (tilt >> next);
            let orbit = tilt >> (quarter >> radial);
            let z = -0.55 * cosine(orbit, eye - a);
            let (a, b) = (view.of(a), view.of(b));
            p.cam_line(out, a, b, light::redshift(base, z), 0.05, Fog::Objects);
            radial = next;
        }
    }
    let rim = light::fade(light::whiten(c, 0.75), 1.3);
    p.ring(out, view, q, 1.2 * grow, rim, 32, 0.05);
}

const fn v(x: f32, y: f32, z: f32) -> P {
    Point::new(x, y, z, 0.0)
}

const OCTAHEDRON: [(P, P); 12] = [
    (v(1.0, 0.0, 0.0), v(0.0, 1.0, 0.0)),
    (v(0.0, 1.0, 0.0), v(-1.0, 0.0, 0.0)),
    (v(-1.0, 0.0, 0.0), v(0.0, -1.0, 0.0)),
    (v(0.0, -1.0, 0.0), v(1.0, 0.0, 0.0)),
    (v(1.0, 0.0, 0.0), v(0.0, 0.0, 1.0)),
    (v(0.0, 1.0, 0.0), v(0.0, 0.0, 1.0)),
    (v(-1.0, 0.0, 0.0), v(0.0, 0.0, 1.0)),
    (v(0.0, -1.0, 0.0), v(0.0, 0.0, 1.0)),
    (v(1.0, 0.0, 0.0), v(0.0, 0.0, -1.0)),
    (v(0.0, 1.0, 0.0), v(0.0, 0.0, -1.0)),
    (v(-1.0, 0.0, 0.0), v(0.0, 0.0, -1.0)),
    (v(0.0, -1.0, 0.0), v(0.0, 0.0, -1.0)),
];

const SPIKES: [(P, P); 7] = [
    (v(-1.0, 0.0, 0.0), v(1.0, 0.0, 0.0)),
    (v(0.0, -1.0, 0.0), v(0.0, 1.0, 0.0)),
    (v(0.0, 0.0, -1.0), v(0.0, 0.0, 1.0)),
    (v(-0.6, -0.6, -0.6), v(0.6, 0.6, 0.6)),
    (v(0.6, -0.6, -0.6), v(-0.6, 0.6, 0.6)),
    (v(-0.6, 0.6, -0.6), v(0.6, -0.6, 0.6)),
    (v(0.6, 0.6, -0.6), v(-0.6, -0.6, 0.6)),
];

const CUBE: [(P, P); 12] = [
    (v(-1.0, -1.0, -1.0), v(1.0, -1.0, -1.0)),
    (v(1.0, -1.0, -1.0), v(1.0, 1.0, -1.0)),
    (v(1.0, 1.0, -1.0), v(-1.0, 1.0, -1.0)),
    (v(-1.0, 1.0, -1.0), v(-1.0, -1.0, -1.0)),
    (v(-1.0, -1.0, 1.0), v(1.0, -1.0, 1.0)),
    (v(1.0, -1.0, 1.0), v(1.0, 1.0, 1.0)),
    (v(1.0, 1.0, 1.0), v(-1.0, 1.0, 1.0)),
    (v(-1.0, 1.0, 1.0), v(-1.0, -1.0, 1.0)),
    (v(-1.0, -1.0, -1.0), v(-1.0, -1.0, 1.0)),
    (v(1.0, -1.0, -1.0), v(1.0, -1.0, 1.0)),
    (v(1.0, 1.0, -1.0), v(1.0, 1.0, 1.0)),
    (v(-1.0, 1.0, -1.0), v(-1.0, 1.0, 1.0)),
];

/// A serpent's head: a long diamond along `+z`, where it swims.
const HEAD: [(P, P); 12] = [
    (v(0.0, 0.0, 1.8), v(0.6, 0.0, 0.0)),
    (v(0.0, 0.0, 1.8), v(-0.6, 0.0, 0.0)),
    (v(0.0, 0.0, 1.8), v(0.0, 0.6, 0.0)),
    (v(0.0, 0.0, 1.8), v(0.0, -0.6, 0.0)),
    (v(0.6, 0.0, 0.0), v(0.0, 0.6, 0.0)),
    (v(0.0, 0.6, 0.0), v(-0.6, 0.0, 0.0)),
    (v(-0.6, 0.0, 0.0), v(0.0, -0.6, 0.0)),
    (v(0.0, -0.6, 0.0), v(0.6, 0.0, 0.0)),
    (v(0.6, 0.0, 0.0), v(0.0, 0.0, -0.9)),
    (v(-0.6, 0.0, 0.0), v(0.0, 0.0, -0.9)),
    (v(0.0, 0.6, 0.0), v(0.0, 0.0, -0.9)),
    (v(0.0, -0.6, 0.0), v(0.0, 0.0, -0.9)),
];

/// A turret: a square base against the wall (local `+x` is outwards) and a tip towards the
/// axis.
const PYRAMID: [(P, P); 8] = [
    (v(0.5, 0.8, -0.8), v(0.5, -0.8, -0.8)),
    (v(0.5, -0.8, -0.8), v(0.5, -0.8, 0.8)),
    (v(0.5, -0.8, 0.8), v(0.5, 0.8, 0.8)),
    (v(0.5, 0.8, 0.8), v(0.5, 0.8, -0.8)),
    (v(0.5, 0.8, -0.8), v(-1.1, 0.0, 0.0)),
    (v(0.5, -0.8, -0.8), v(-1.1, 0.0, 0.0)),
    (v(0.5, -0.8, 0.8), v(-1.1, 0.0, 0.0)),
    (v(0.5, 0.8, 0.8), v(-1.1, 0.0, 0.0)),
];

/// The ship: an arrow into the screen, wings, a fin, wingtips.
const SHIP: [(P, P); 10] = [
    (v(0.0, 0.0, 1.3), v(0.9, 0.0, -0.6)),
    (v(0.0, 0.0, 1.3), v(-0.9, 0.0, -0.6)),
    (v(0.9, 0.0, -0.6), v(0.3, 0.0, -0.3)),
    (v(-0.9, 0.0, -0.6), v(-0.3, 0.0, -0.3)),
    (v(0.3, 0.0, -0.3), v(-0.3, 0.0, -0.3)),
    (v(0.0, 0.0, 1.3), v(0.0, 0.35, -0.4)),
    (v(0.0, 0.35, -0.4), v(0.3, 0.0, -0.3)),
    (v(0.0, 0.35, -0.4), v(-0.3, 0.0, -0.3)),
    (v(0.9, 0.0, -0.6), v(1.1, 0.1, -0.9)),
    (v(-0.9, 0.0, -0.6), v(-1.1, 0.1, -0.9)),
];

/// Which fog a segment sees.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Fog {
    /// The wall and the dust: short, and fading right at the camera too.
    Wall,
    /// Everything else: seen from further.
    Objects,
}

/// A singularity as the camera sees it: its screen point, its depth, and its lensing
/// strength (the Einstein radius² of what lies far behind it).
#[derive(Clone, Copy, Debug)]
struct Lens {
    centre: Point2<(), f32>,
    depth: f32,
    k: f32,
}

/// The projection: a camera motor, a focal length, and the lenses in view.
struct Proj {
    cam: Frame,
    focal: f32,
    lenses: Vec<Lens>,
    /// The pinhole: camera-frame points to screen points, with the depth as the weight.
    pinhole: Point2<(Point,), f32>,
}

/// A map from points (straightened, or in an object's own frame) to camera-frame points.
type CamMap = Point<(Point,), f32>;

impl Proj {
    fn new(cam: Frame, focal: f32, lenses: Vec<Lens>) -> Self {
        // The images of `x`, `y`, `z` and the weight: the camera looks along `+z`, with `x`
        // mirrored onto the screen, and the depth becomes the weight (the perspective divide).
        let pinhole = Point2::<(Point,), f32>::from_images([
            Point2::new(-focal, 0.0, 0.0),
            Point2::new(0.0, focal, 0.0),
            Point2::new(0.0, 0.0, 1.0),
            Point2::new(0.0, 0.0, 0.0),
        ]);
        Proj {
            cam,
            focal,
            lenses,
            pinhole,
        }
    }

    /// The camera map of a placement: `cam << (placement >> x)`, one 4x4 built from the motor
    /// `~cam placement` and applied to every point placed by it.
    fn to_camera(&self, placement: Frame) -> CamMap {
        (self.cam.reverse() * placement) >> Point::slot()
    }

    /// The camera map for straightened points near `q`, placed rigidly with the frame at its
    /// arc length.
    fn at(&self, track: &Track, q: P) -> CamMap {
        self.to_camera(track.placement(arc(q)))
    }

    /// The camera map of an object at `q` (straightened) turned by `turn`: from the object's
    /// own frame, through the track's frame at `q`, into the camera.
    fn object(&self, track: &Track, q: P, turn: Frame) -> CamMap {
        let [x, y, s] = q.to_euclidean();
        self.to_camera(track.frame(s) * Motor::translation(x, y, 0.0) * turn)
    }

    /// A world point in the camera's frame (`cam << p`), or `None` behind the near plane.
    fn camera(&self, p: P) -> Option<P> {
        let c = self.cam << p;
        (self.depth(c) > NEAR).then_some(c)
    }

    /// The depth of a camera-frame point (of weight 1): its signed distance from the camera's
    /// image plane, `plane & point`.
    fn depth(&self, c: P) -> f32 {
        (IMAGE_PLANE & c).s()
    }

    /// The screen point of a camera-frame point: the PGA2D point with the depth as its
    /// weight (the perspective divide is normalizing it). The camera looks along `+z` with
    /// `+y` up, so its `+x` is on the left.
    fn screen(&self, c: P) -> Point2<(), f32> {
        self.pinhole.of(c).unitized()
    }

    /// Where a camera-frame point is seen: its screen point, lensed by every singularity in
    /// front of it. For a point mass the Einstein radius² of a source at depth `z` behind a
    /// lens at `z_l` is `k (z - z_l) / z`, and a source at `β` from the lens is seen at the
    /// outer root of `θ² - βθ - θ_E² = 0`, `θ = (β + |(β, 2θ_E)|) / 2`: a scaling about the
    /// lens by `θ / β` (the traced `scale_about`).
    fn lensed(&self, c: P) -> Point2<(), f32> {
        let mut q = self.screen(c);
        let z = self.depth(c);
        for l in &self.lenses {
            if z <= l.depth {
                continue;
            }
            let e = gax::Real::sqrt((l.k * (z - l.depth) / z).min(100.0));
            let beta = (q & l.centre).norm().max(1e-3);
            let theta = 0.5 * (beta + Point2::direction(beta, 2.0 * e).ideal_norm());
            q = crate::scale_about(q, l.centre, theta / beta);
        }
        q
    }

    /// The world ray from the eye through a screen point: the camera-frame direction whose
    /// projection it is, taken into the world, joined with the eye.
    fn ray(&self, at: Point2<(), f32>) -> gax::pga3d::Line<(), f32> {
        let [x, y] = at.to_euclidean();
        let d = dir(-x / self.focal, y / self.focal, 1.0);
        let eye = self.cam >> Point::xyz(0.0, 0.0, 0.0);
        eye & (self.cam >> d)
    }

    /// A segment between two camera-frame points: width and glow by depth, faded by fog.
    #[allow(clippy::too_many_arguments)]
    fn segment(
        &self,
        out: &mut Vec<LineInstance>,
        a: P,
        b: P,
        color: Light,
        width: f32,
        glow: f32,
        fog: Fog,
    ) {
        let z = 0.5 * (self.depth(a) + self.depth(b));
        let k = match fog {
            Fog::Wall => {
                let far = (1.0 - z / WALL_FOG).clamp(0.0, 1.0);
                // Right at the camera the wall fades too, so it never floods the screen.
                let near = ((z - 0.8) / 4.0).clamp(0.0, 1.0);
                far * far * far * near
            }
            Fog::Objects => (1.0 - (z - 45.0) / (FOG - 45.0)).clamp(0.0, 1.0),
        };
        if k <= 0.0 {
            return;
        }
        let scale = self.focal / z;
        let (wmin, wmax) = if fog == Fog::Wall {
            (0.01, 0.1)
        } else {
            (0.035, 0.3)
        };
        let style = [
            (width * scale).clamp(wmin, wmax),
            (glow * scale).clamp(0.08, 0.6),
            0.3,
            0.0,
        ];
        out.push(scene::seg(
            self.lensed(a),
            self.lensed(b),
            light::fade(color, k),
            style,
            identity(),
        ));
    }

    /// A segment between two world points, cut at the near plane by its meet with it.
    fn line(&self, out: &mut Vec<LineInstance>, a: P, b: P, color: Light, width: f32, fog: Fog) {
        self.cam_line(out, self.cam << a, self.cam << b, color, width, fog);
    }

    /// A short segment between two straightened points, both placed with the frame at `b`:
    /// streaks, whose ends are too close for the track to bend between them.
    #[allow(clippy::too_many_arguments)]
    fn streak(
        &self,
        out: &mut Vec<LineInstance>,
        track: &Track,
        a: P,
        b: P,
        color: Light,
        width: f32,
        fog: Fog,
    ) {
        let v = self.at(track, b);
        self.cam_line(out, v.of(a), v.of(b), color, width, fog);
    }

    /// A segment between two camera-frame points, cut at the near plane by its meet with it.
    #[allow(clippy::too_many_arguments)]
    fn cam_line(
        &self,
        out: &mut Vec<LineInstance>,
        ca: P,
        cb: P,
        color: Light,
        width: f32,
        fog: Fog,
    ) {
        let (fa, fb) = (self.depth(ca) > NEAR, self.depth(cb) > NEAR);
        let near = Plane::from_normal([0.0, 0.0, 1.0], NEAR);
        let cut = || {
            let m = (ca & cb) ^ near;
            m.unitized()
        };
        let (ca, cb) = match (fa, fb) {
            (true, true) => (ca, cb),
            (false, false) => return,
            (true, false) => (ca, cut()),
            (false, true) => (cut(), cb),
        };
        self.segment(out, ca, cb, color, width, width * 5.0, fog);
    }

    /// A wireframe solid seen through its camera map `model` (see [`Proj::object`]), scaled
    /// by `size`: each vertex is a direction from the centre.
    fn solid(
        &self,
        out: &mut Vec<LineInstance>,
        model: &CamMap,
        edges: &[(P, P)],
        size: f32,
        color: Light,
    ) {
        let o = Point::xyz(0.0, 0.0, 0.0);
        for &(a, b) in edges {
            let (a, b) = (model.of(o + a * size), model.of(o + b * size));
            self.cam_line(out, a, b, color, 0.05, Fog::Objects);
        }
    }

    /// A ring across the tunnel (perpendicular to the track) at `q`: chords of a radius
    /// turned about the line through `q` along the track, seen through `view`, the camera map
    /// at `q` (every point of the ring is at its arc length).
    #[allow(clippy::too_many_arguments)]
    fn ring(
        &self,
        out: &mut Vec<LineInstance>,
        view: &CamMap,
        q: P,
        r: f32,
        color: Light,
        n: usize,
        width: f32,
    ) {
        let spin = Motor::rotation(q & dir(0.0, 0.0, 1.0), core::f32::consts::TAU / n as f32);
        let mut a = q + dir(r, 0.0, 0.0);
        for _ in 0..n {
            let b = spin >> a;
            self.cam_line(out, view.of(a), view.of(b), color, width, Fog::Objects);
            a = b;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tunnel::{Input, at};

    fn flown(seed: u64, seconds: usize) -> World {
        let mut w = World::new(seed);
        w.ship.invulnerable = 1e9;
        for _ in 0..120 * seconds {
            w.tick(&Input::default());
        }
        w
    }

    /// The level camera looks down the track and its right axis is horizontal (no roll).
    #[test]
    fn the_camera_looks_ahead_without_rolling() {
        let w = flown(4, 20);
        let v = View::new(85.0);
        let cam = v.target(&w, 0.0);
        let right = cam >> dir(1.0, 0.0, 0.0);
        assert!(right.e013().abs() < 1e-4, "rolled: {}", right.e013());
        // The axis ahead lands near the screen's centre.
        let p = Proj::new(cam, v.focal, Vec::new());
        let ahead = w.track.place(at(0.0, 0.0, w.ship.s() + 16.0));
        let [x, y] = p.screen(p.camera(ahead).expect("in front")).to_euclidean();
        assert!(x.abs() < 3.0 && y.abs() < 4.0, "{x} {y}");
    }

    /// The ship's cross-section fills the screen: its edge is well out towards the border.
    #[test]
    fn the_cross_section_uses_the_screen() {
        let w = flown(2, 5);
        let mut v = View::new(85.0);
        v.update(&w, 0.0, 1.0);
        let p = v.proj();
        let s = w.ship.s();
        let edge = crate::tunnel::around(crate::tunnel::SHIP_RADIUS, 0.0, s);
        let c = p.camera(w.track.place(edge)).expect("in front");
        let centre = p.camera(w.track.place(at(0.0, 0.0, s))).expect("in front");
        let r = (p.screen(c) & p.screen(centre)).norm();
        assert!(r > 10.0, "the cross-section is {r} of 18 units");
    }

    /// Input to the right on screen moves the ship to the right on screen, however the track
    /// has rolled; and the reticle's ray comes back as the point it was cast through.
    #[test]
    fn screen_directions_and_rays_map_through_the_camera() {
        let w = flown(8, 30);
        let mut v = View::new(85.0);
        v.update(&w, 0.0, 1.0);
        let d = v.across(&w, Point2::direction(1.0, 0.0));
        let p = v.proj();
        let a = p.screen(p.camera(w.track.place(w.ship.pos)).unwrap());
        let b = p.screen(p.camera(w.track.place(w.ship.pos + d)).unwrap());
        let (a, b) = (a.to_euclidean(), b.to_euclidean());
        assert!(
            b[0] - a[0] > 0.1 && (b[1] - a[1]).abs() < 0.1 * (b[0] - a[0]),
            "{a:?} {b:?}"
        );
        // Cast through a screen point at the aim depth, and project back: the same point.
        let s = w.ship.s() + AIM_DEPTH;
        let at_screen = Point2::xy(3.0, -2.0);
        let q = v.through(&w, at_screen, s);
        assert!((arc(q) - s).abs() < 0.05, "{} vs {s}", arc(q));
        let back = p.screen(p.camera(w.track.place(q)).unwrap());
        assert!(
            (back & at_screen).norm() < 0.05,
            "{:?}",
            back.to_euclidean()
        );
    }

    /// The reticle locks onto the enemy under it, at any depth, and the aim is that enemy.
    #[test]
    fn the_reticle_locks_what_is_under_it() {
        let mut w = flown(6, 12);
        w.enemies.retain(|e| e.foe == Foe::Drone);
        let mut v = View::new(85.0);
        v.update(&w, 0.0, 1.0);
        let s = w.ship.s();
        let Some(e) = w.enemies.iter().find(|e| arc(e.pos) > s + 5.0).cloned() else {
            return;
        };
        let p = v.proj();
        let on_screen = p.screen(p.camera(w.track.place(e.pos)).unwrap());
        let aim = v.aim(&w, on_screen + Point2::direction(0.08, -0.05));
        assert_eq!(v.lock, Some(e.id));
        assert!((aim & e.pos).norm() < 1e-4);
    }

    /// The camera never whips round. Its target, `look_at` down the track, flips the sign
    /// of its motor now and then (whenever the track heads back through the world's `-z`),
    /// and a spring that interpolated the long way then spun nearly a full turn in a frame.
    #[test]
    fn the_camera_follows_smoothly_through_sign_flips() {
        let mut flips = 0;
        for seed in 1..4u64 {
            let mut w = World::new(seed);
            w.ship.invulnerable = 1e9;
            let mut v = View::new(85.0);
            v.update(&w, 0.0, DT);
            let mut target = v.target(&w, 0.0);
            for t in 0..120 * 200 {
                let a = t as f32 * 0.013;
                let movement = about_axis(a) >> dir(0.8, 0.0, 0.0);
                w.tick(&Input {
                    movement,
                    ..Input::default()
                });
                let before = v.cam;
                v.update(&w, 0.0, DT);
                let next = v.target(&w, 0.0);
                flips += usize::from((target.reverse() * next).s() < 0.0);
                target = next;
                // The view direction moves by a small angle per frame.
                let f = |m: Frame| m >> dir(0.0, 0.0, 1.0);
                let (a, b) = (f(before), f(v.cam));
                let cos = (a.e032() * b.e032() + a.e013() * b.e013() + a.e021() * b.e021())
                    / (a.ideal_norm() * b.ideal_norm());
                assert!(
                    cos > 0.999,
                    "seed {seed}, tick {t}: the camera swung ({cos})"
                );
            }
        }
        assert!(flips > 0, "no sign flip happened: the test tests nothing");
    }

    #[test]
    fn the_shadow_falls_on_the_wall() {
        let q = at(3.0, 1.0, 10.0);
        let base = foot(q);
        let r = off_axis(q);
        let wall = (base + (q - base) * (RADIUS / r)) | (base & q);
        let hit = ((base - dir(0.0, 0.0, 2.0)) & q) ^ wall;
        let hit = hit.unitized();
        assert!(
            (off_axis(hit) - RADIUS).abs() < 1e-3,
            "{:?}",
            hit.to_euclidean()
        );
        assert!(arc(hit) > arc(q), "the shadow falls ahead");
    }

    #[test]
    fn a_projection_is_a_weight() {
        let p = Proj::new(Motor::translation(0.0, 0.0, 0.0), 10.0, Vec::new());
        let s = p
            .screen(p.camera(at(2.0, 1.0, 4.0)).unwrap())
            .to_euclidean();
        assert!((s[0] + 5.0).abs() < 1e-5 && (s[1] - 2.5).abs() < 1e-5);
        // A segment across the near plane is cut where it meets it.
        let mut out = Vec::new();
        p.line(
            &mut out,
            at(0.0, 0.0, -1.0),
            at(0.0, 1.0, 3.0),
            palette::HUD,
            0.05,
            Fog::Objects,
        );
        assert_eq!(out.len(), 1);
    }
}
