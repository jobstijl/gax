//! Drawing the Tunnel: a camera motor in 3D, a projective projection, and everything as lines
//! in the HUD's screen units (the line renderer is 2D).
//!
//! * The camera is `Motor::look_at` from just behind the ship, near the axis, down the track;
//!   by default its up is the world's (it does not roll with the tunnel, a comfort setting).
//!   It follows on a motor spring (screw interpolation towards the target).
//! * `cam << p` takes a world point into the camera's frame. Objects go there as maps: one
//!   4x4 per object, `(~cam placement) >> Point::slot()`, built from the motors once and
//!   applied to every vertex (a wall ring's nodes in a batch, `of_slice`); the track's frame
//!   is computed once per object, not per point. The projection to the screen is projective
//!   geometry (`proj`): the camera-space point becomes the PGA2D point with the depth as its
//!   weight, and `to_euclidean` is the perspective divide. Segments crossing the near plane
//!   are cut by their meet with it.
//! * Depth reads through fog, widths and glows that scale with `1/z`, bolts that glow brighter
//!   as they come close, the ship's shadow on the wall (the meet of the light-to-ship line with
//!   the wall's tangent plane, which is `wall point | radial line`), and the reticle's lock and
//!   lead.
//! * The reticle aims through the same camera, backwards (`aim`).
//! * A singularity bends light. Everything behind it is lensed as by a point mass: a point
//!   seen at `β` from it on screen appears at the outer root of the lens equation
//!   `θ² - βθ - θ_E² = 0`, a scaling about the singularity's screen point, so what lies behind
//!   it is pushed out into an Einstein ring round a dark shadow. Its accretion disc is beamed
//!   (the side coming at you brighter and bluer, the other dimmer and redder), and light near
//!   it is redshifted: a turn of its hue in OkLab, and dimmer.

mod aim;
mod effects;
mod models;
mod proj;

use super::LineInstance;
use super::font::Align;
use super::scene::{self, Stroke, palette};
use crate::geom::{ORIGIN, lerp};
use crate::light::{self, Light};
use crate::sim::body::Pose;
use crate::sim::rng::Rng;
use crate::sim::{DT, Phase};
use crate::tunnel::lattice::{AROUND, RADIUS, RINGS};
use crate::tunnel::track::Frame;
use crate::tunnel::{
    Enemy, Foe, GATE_RADIUS, GateState, P, ROLL_TIME, Wells, World, about_axis, arc, dir, foot,
    off_axis,
};
use effects::{Popup, Spark};
use gax::batch::BatchOf;
use gax::pga2d::Point as Point2;
use gax::pga3d::{Motor, Point};
use models::{CUBE, HEAD, OCTAHEDRON, PYRAMID, SHIP, SPIKES};
use proj::{CamMap, FOG, Fog, Lens, NEAR, Proj, Seen, cosine};

/// Bolts: hot pink, the colour of danger.
const BOLT: Light = light::light(1.0, 0.25, 0.45, 3.0);
/// Gates: warm gold, a pickup's colour, never an enemy's.
pub const GATE_LIGHT: Light = light::light(1.0, 0.8, 0.32, 2.2);
/// Dust in the tunnel.
const DUST: Light = light::light(0.6, 0.7, 1.0, 0.35);
/// How strongly a singularity lenses (its Einstein radius² per unit of strength, focal length²
/// and inverse depth).
const LENS: f32 = 1.0;

/// The view's settings, from the game's.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ViewSettings {
    /// Screen units per unit of `x/z` (from the field of view, [`View::focal_for`]).
    pub focal: f32,
    /// Follow the tunnel's roll (off by default).
    pub follow_roll: bool,
    /// Screen shake scale.
    pub shake_scale: f32,
    /// Flash scale (the reduced-flashes setting).
    pub flash_scale: f32,
    /// The lock-on margin beyond an enemy's apparent size (the aim-assist setting).
    pub assist: f32,
    /// Singularities lens what lies behind them (off with reduced motion).
    pub lensing: bool,
}

/// A wall node as seen this frame: where, how far it is displaced (for its glow), and how
/// far its light is redshifted.
#[derive(Clone, Copy, Debug)]
struct WallNode {
    seen: Seen,
    strain: f32,
    shift: f32,
}

/// The Tunnel's view: camera, effects, the reticle, and the projection.
pub struct View {
    /// The camera's pose in the world.
    pub cam: Frame,
    /// The settings.
    pub settings: ViewSettings,
    /// The projection through `cam` (without the shake), with the singularities in view as
    /// lenses: for aiming, and for drawing when the camera does not shake.
    proj: Proj,
    shake: f32,
    /// A full-screen flash, decaying.
    pub flash: f32,
    /// The reticle on screen (HUD units).
    pub reticle: Point2<(), f32>,
    /// The enemy the reticle has locked onto.
    pub lock: Option<u32>,
    /// A collapse's shock ripple: where, and how long ago.
    shock: Option<(P, f32)>,
    sparks: Vec<Spark>,
    dust: Vec<P>,
    popups: Vec<Popup>,
    /// The wall's nodes as seen, kept from frame to frame so that drawing allocates nothing.
    wall: Vec<Option<WallNode>>,
    /// The frame's camera maps along the track (kept likewise), see [`CameraMaps`].
    maps: Vec<CamMap>,
    rng: Rng,
    started: bool,
}

/// The camera's eye for the ship at `ship`: just behind it, a little above, and close to the
/// axis.
fn eye(ship: P) -> P {
    let axis_point = foot(ship);
    axis_point + (ship - axis_point) * 0.3 + dir(0.0, 0.9, -7.5)
}

/// A light from where the light is redshifted by `z` (near a singularity).
fn tint(c: Light, z: f32) -> Light {
    if z > 0.03 { light::redshift(c, z) } else { c }
}

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
        let focal = View::focal_for(fov_degrees);
        let cam = Frame::identity();
        View {
            cam,
            settings: ViewSettings {
                focal,
                follow_roll: false,
                shake_scale: 1.0,
                flash_scale: 1.0,
                assist: 1.1,
                lensing: true,
            },
            proj: Proj::new(cam, focal),
            shake: 0.0,
            flash: 0.0,
            reticle: ORIGIN,
            lock: None,
            shock: None,
            sparks: Vec::new(),
            dust,
            popups: Vec::new(),
            wall: Vec::with_capacity(RINGS * AROUND),
            maps: Vec::new(),
            rng,
            started: false,
        }
    }

    /// The focal length for a vertical field of view (the HUD is 36 units tall): the
    /// screen's half height over the tangent of half the angle.
    pub fn focal_for(fov_degrees: f32) -> f32 {
        let half = fov_degrees.to_radians() * 0.5;
        // tan = sin / cos: the direction at the half angle, height over width.
        let d = Point2::direction_from_angle(half);
        scene::HUD_HALF_HEIGHT * d.e20() / d.e01()
    }

    /// The camera the view wants: just behind the ship, a little above and close to the
    /// axis, looking down the track.
    fn target(&self, w: &World, alpha: f32) -> Frame {
        let ship = lerp(w.ship.prev, w.ship.pos, alpha);
        let axis_point = foot(ship);
        let eye = eye(ship);
        let look = axis_point + (ship - axis_point) * 0.1 + dir(0.0, 0.0, 20.0);
        let up = if self.settings.follow_roll {
            w.track.frame(arc(eye)) >> dir(0.0, 1.0, 0.0)
        } else {
            dir(0.0, 1.0, 0.0)
        };
        Motor::look_at(w.track.place(eye), w.track.place(look), up)
    }

    /// Advance the camera spring and the effects by a frame of `dt`.
    pub fn update(&mut self, w: &World, alpha: f32, dt: f32) {
        let target = self.target(w, alpha);
        self.cam = if self.started {
            // The shorter way (`m` and `-m` are the same motion; see `Motor::interpolate`).
            Motor::interpolate(self.cam, target, 1.0 - (-12.0 * dt).exp()).renormalize_fast()
        } else {
            target
        };
        self.started = true;
        self.age_effects(dt);
        self.proj = self.projection(self.cam, w);
        // Dust behind the ship comes round again ahead.
        let s = w.ship.s();
        for d in &mut self.dust {
            if arc(*d) < s - 5.0 {
                *d += dir(0.0, 0.0, FOG + 5.0);
            }
        }
    }

    /// The projection through the camera `cam`, with the singularities in front of it as
    /// lenses (none with lensing off).
    fn projection(&self, cam: Frame, w: &World) -> Proj {
        let focal = self.settings.focal;
        let mut p = Proj::new(cam, focal);
        if self.settings.lensing {
            p.lenses = w
                .wells()
                .iter()
                .filter_map(|(q, strength)| {
                    let c = p.camera(w.track.place(q))?;
                    let depth = p.depth(c);
                    Some(Lens {
                        centre: p.screen(c),
                        depth,
                        k: LENS * strength * focal * focal / depth.max(2.0),
                    })
                })
                .collect();
        }
        p
    }

    /// The camera shaken for drawing, if it shakes.
    fn shaken(&mut self) -> Option<Frame> {
        let a = self.shake * self.shake * self.settings.shake_scale;
        if a < 1e-4 {
            return None;
        }
        let r = &mut self.rng;
        let jitter = Motor::translation(dir(
            r.range(-1.0, 1.0) * a * 0.3,
            r.range(-1.0, 1.0) * a * 0.3,
            0.0,
        )) * about_axis(r.range(-0.015, 0.015) * a);
        Some(self.cam * jitter)
    }

    /// Build the frame's lines (screen units, for the HUD camera).
    pub fn draw(&mut self, w: &World, alpha: f32, time: f32, out: &mut Vec<LineInstance>) {
        out.clear();
        let shaken = self.shaken().map(|cam| self.projection(cam, w));
        let p = shaken.as_ref().unwrap_or(&self.proj);
        let sight = Sight::new(p, w, alpha, time, &mut self.maps);
        sight.draw_wall(&mut self.wall, out);
        sight.draw_dust(&self.dust, out);
        sight.draw_warpins(out);
        sight.draw_gates(out);
        sight.draw_enemies(out);
        sight.draw_bolts_shards_shots(out);
        sight.draw_sparks(&self.sparks, out);
        if w.phase == Phase::Playing {
            sight.draw_ship(out);
            sight.draw_reticle(self.reticle, self.lock, out);
        }
        sight.draw_popups(&self.popups, out);
    }
}

/// The frame's camera maps at every `STEP` of arc length around the ship, for the many small
/// things (sparks, dust, shots, warp-ins): one track frame per step, not one per object. A
/// point is seen through the map of the nearest step, which places it as rigidly from at most
/// half a step away: a turn of at most `0.036 STEP / 2` radians about the track's frame.
struct CameraMaps<'a> {
    maps: &'a [CamMap],
    /// The arc length of the first.
    from: f32,
}

impl<'a> CameraMaps<'a> {
    /// The arc length between maps.
    const STEP: f32 = 0.5;
    /// How far behind the ship they start, and how far beyond the fog they reach.
    const BEHIND: f32 = 12.0;
    const BEYOND: f32 = 12.0;

    /// The maps around the ship at `s` through `p`, into the kept buffer `maps`.
    fn build(maps: &'a mut Vec<CamMap>, p: &Proj, w: &World, s: f32) -> Self {
        let from = s - Self::BEHIND;
        let n = ((Self::BEHIND + FOG + Self::BEYOND) / Self::STEP) as usize + 1;
        maps.clear();
        maps.extend((0..n).map(|k| p.to_camera(w.track.placement(from + k as f32 * Self::STEP))));
        CameraMaps { maps, from }
    }

    /// The map for points near arc length `s`, if within reach.
    fn near(&self, s: f32) -> Option<&CamMap> {
        let k = ((s - self.from) / Self::STEP).round();
        (k >= 0.0).then(|| self.maps.get(k as usize)).flatten()
    }
}

/// One frame as the camera sees it: the projection (shaken), the world between its ticks,
/// the camera maps along the track, and the singularities that redshift light.
struct Sight<'a> {
    p: &'a Proj,
    w: &'a World,
    alpha: f32,
    time: f32,
    maps: CameraMaps<'a>,
    wells: Wells,
    /// The ship between ticks.
    ship: P,
}

impl<'a> Sight<'a> {
    /// The frame through `p` at `alpha` between ticks, its camera maps along the track built
    /// into the kept buffer `maps`.
    fn new(p: &'a Proj, w: &'a World, alpha: f32, time: f32, maps: &'a mut Vec<CamMap>) -> Self {
        let ship = lerp(w.ship.prev, w.ship.pos, alpha);
        Sight {
            p,
            w,
            alpha,
            time,
            maps: CameraMaps::build(maps, p, w, arc(ship)),
            wells: w.wells(),
            ship,
        }
    }

    /// The world point of a straightened point.
    fn place(&self, q: P) -> P {
        self.w.track.place(q)
    }

    /// The camera map for straightened points near `q`, placed rigidly with the frame at its
    /// arc length.
    fn near(&self, q: P) -> CamMap {
        self.p.at(&self.w.track, q)
    }

    /// The camera map for a small thing near `q`: the frame's nearest map along the track
    /// (see [`CameraMaps`]), or its own beyond them.
    fn small(&self, q: P) -> CamMap {
        self.maps
            .near(arc(q))
            .copied()
            .unwrap_or_else(|| self.near(q))
    }

    /// The camera map of an object at `q` turned by `turn`.
    fn object(&self, q: P, turn: Frame) -> CamMap {
        self.p.object(&self.w.track, q, turn)
    }

    /// How far light from `q` is redshifted: the most any singularity shifts it, its strength
    /// over its softened squared distance.
    fn shift(&self, q: P) -> f32 {
        self.wells
            .iter()
            .map(|(c, strength)| {
                let d = q.distance(c);
                1.1 * strength / (1.0 + d * d / 4.0)
            })
            .fold(0.0f32, f32::max)
    }

    /// The wall: one camera map per ring, from its placement, applied to the ring's nodes in a
    /// batch, each node seen once; then the lines between neighbours.
    fn draw_wall(&self, nodes: &mut Vec<Option<WallNode>>, out: &mut Vec<LineInstance>) {
        let (p, lat) = (self.p, &self.w.lattice);
        nodes.clear();
        let mut cam = [Point::zero(); AROUND];
        for r in 0..RINGS {
            let view = p.to_camera(self.w.track.placement(lat.ring_s(r)));
            let ring: [P; AROUND] = core::array::from_fn(|j| lat.node(r, j));
            view.of_slice(&ring, &mut cam);
            nodes.extend(ring.iter().zip(&cam).enumerate().map(|(j, (&node, &c))| {
                (p.depth(c) > NEAR).then(|| WallNode {
                    seen: p.seen(c),
                    strain: lat.strain(r, j),
                    shift: self.shift(node),
                })
            }));
        }
        for r in 0..RINGS {
            for j in 0..AROUND {
                let Some(a) = nodes[r * AROUND + j] else {
                    continue;
                };
                let mut edge = |b: Option<WallNode>, bright: f32| {
                    if let Some(b) = b {
                        let glow = 1.0 + 4.0 * (a.strain + b.strain).min(1.5);
                        let c = palette::GRID.faded(1.5 * bright * glow);
                        let c = tint(c, 0.5 * (a.shift + b.shift));
                        p.segment(out, a.seen, b.seen, c, 0.03, 0.22, Fog::Wall);
                    }
                };
                edge(nodes[r * AROUND + (j + 1) % AROUND], 1.0);
                if r + 1 < RINGS {
                    edge(nodes[(r + 1) * AROUND + j], 0.8);
                }
            }
        }
    }

    /// Dust, streaked along the track by the speed.
    fn draw_dust(&self, dust: &[P], out: &mut Vec<LineInstance>) {
        let streak = dir(0.0, 0.0, 0.012 * self.w.ship.speed);
        let s = arc(self.ship);
        for &d in dust.iter().filter(|&&d| arc(d) >= s - 2.0) {
            let view = self.small(d);
            self.p
                .streak(out, &view, d - streak, d, DUST, 0.02, Fog::Wall);
        }
    }

    /// Warp-ins: rings contracting onto where a spawn lands.
    fn draw_warpins(&self, out: &mut Vec<LineInstance>) {
        for pd in &self.w.pending {
            let k = (pd.t / 0.8).clamp(0.0, 1.0);
            let c = (scene::color(pd.foe.kind())).faded(0.8);
            let view = self.small(pd.pos);
            self.p.ring(out, &view, pd.pos, 0.4 + 2.0 * k, c, 10, 0.04);
        }
    }

    /// Gates: the next one bright and pulsing, the rest of the slalom dimmer and joined by a
    /// faint path; flown through, a ring flung outwards; missed, grey and gone.
    fn draw_gates(&self, out: &mut Vec<LineInstance>) {
        let p = self.p;
        let mut open = 0;
        let mut last: Option<P> = None;
        for g in &self.w.gates {
            let view = self.near(g.pos);
            match g.state {
                GateState::Open => {
                    let k = if open == 0 {
                        1.2 + 0.4 * crate::signal::wave(self.time * 6.0)
                    } else {
                        0.55
                    };
                    open += 1;
                    let c = GATE_LIGHT.faded(k);
                    p.ring(out, &view, g.pos, GATE_RADIUS, c, 32, 0.09);
                    p.ring(out, &view, g.pos, GATE_RADIUS * 0.8, c.faded(0.3), 32, 0.03);
                    if let Some(prev) = last {
                        let path = GATE_LIGHT.faded(0.15);
                        let (a, b) = (self.place(prev), self.place(g.pos));
                        p.line(out, a, b, path, 0.02, Fog::Objects);
                    }
                    last = Some(g.pos);
                }
                GateState::Passed if g.t < 0.5 => {
                    let c = (GATE_LIGHT.whitened(0.6)).faded(1.5 * (1.0 - g.t / 0.5));
                    let r = GATE_RADIUS * (1.0 + 3.0 * g.t);
                    p.ring(out, &view, g.pos, r, c, 32, 0.07);
                }
                GateState::Missed if g.t < 0.6 => {
                    let grey = GATE_LIGHT.perceptually(|c| c.desaturated(1.0));
                    let c = grey.faded(0.5 * (1.0 - g.t / 0.6));
                    p.ring(out, &view, g.pos, GATE_RADIUS, c, 32, 0.04);
                }
                _ => {}
            }
        }
    }

    /// Enemies, each in its own shape, tumbling, flashing when hit, redshifted near a
    /// singularity.
    fn draw_enemies(&self, out: &mut Vec<LineInstance>) {
        let p = self.p;
        for e in &self.w.enemies {
            let q = lerp(e.prev, e.pos, self.alpha);
            let c = scene::color(e.foe.kind());
            let c = (c.whitened(e.flash)).faded(1.0 + e.flash);
            let c = if e.foe == Foe::Singularity {
                c
            } else {
                tint(c, self.shift(q))
            };
            let spin = e.age * 2.0 + e.id as f32;
            let tumble = about_axis(spin)
                * Motor::rotation_about(Point::direction(1.0, 0.0, 0.0), 0.7 * spin);
            match e.foe {
                Foe::Drone => p.solid(out, &self.object(q, tumble), &OCTAHEDRON, 0.7, c),
                Foe::Mine => {
                    let pulse = 1.0 + 0.15 * crate::signal::wave(e.age * 7.0);
                    p.solid(out, &self.object(q, tumble), &SPIKES, 0.9 * pulse, c);
                    p.solid(out, &self.object(q, tumble.reverse()), &CUBE, 0.35, c);
                }
                Foe::Turret => {
                    // A pyramid on the wall pointing at the axis, turned about the axis to its
                    // place; it glows before it fires.
                    let charge = ((0.35 - e.timer) / 0.35).clamp(0.0, 1.0);
                    let c = c.faded(1.0 + 2.0 * charge);
                    p.solid(out, &self.object(q, about_axis(e.angle)), &PYRAMID, 1.0, c);
                }
                Foe::Serpent => self.draw_serpent(e, q, c, out),
                Foe::Singularity => self.draw_singularity(e, q, c, out),
            }
        }
    }

    /// A serpent: the head, pointing where it swims; the body, diamonds on the helix it has
    /// just swum, joined by its spine.
    fn draw_serpent(&self, e: &Enemy, q: P, c: Light, out: &mut Vec<LineInstance>) {
        let p = self.p;
        let pose = Motor::rotation_between(dir(0.0, 0.0, 1.0), e.vel);
        p.solid(out, &self.object(q, pose), &HEAD, 0.95, c);
        let base = scene::color(Foe::Serpent.kind());
        let n = e.body.len().max(1) as f32;
        let mut last = q;
        for (k, seg) in e.body.iter().enumerate() {
            let sq = lerp(seg.prev, seg.pos, self.alpha);
            let sc = tint(base.faded(1.0 - 0.45 * k as f32 / n), self.shift(sq));
            let spin = about_axis(e.age * 3.0 + k as f32 * 0.5);
            p.solid(out, &self.object(sq, spin), &OCTAHEDRON, 0.5, sc);
            let (a, b) = (self.place(last), self.place(sq));
            p.line(out, a, b, sc.faded(0.5), 0.03, Fog::Objects);
            last = sq;
        }
    }

    /// A singularity: an accretion disc, nearly edge on and slowly precessing, whose rings
    /// turn faster inside; each piece of it beamed by its orbital velocity against the line
    /// of sight to the eye (coming at you, blueshifted and brighter; going away, redshifted
    /// and dimmer). The disc's far half lies behind the singularity and is lensed up over its
    /// shadow. Round the shadow, the photon ring.
    fn draw_singularity(&self, e: &Enemy, q: P, c: Light, out: &mut Vec<LineInstance>) {
        const N: usize = 36;
        let (p, time) = (self.p, self.time);
        let view = self.near(q);
        let eye = eye(self.ship);
        let grow = 1.0 + 0.1 * e.mass;
        let tilt =
            about_axis(0.15 * time) * Motor::rotation_about(Point::direction(1.0, 0.0, 0.0), 1.25);
        let quarter = about_axis(core::f32::consts::FRAC_PI_2);
        let step = about_axis(core::f32::consts::TAU / N as f32);
        for (k, r) in [1.9f32, 2.5, 3.2].into_iter().enumerate() {
            let heat = 1.0 - 0.3 * k as f32;
            let base = (c.whitened(0.55 * heat)).faded(heat);
            let spin = time * (2.4 - 0.6 * k as f32);
            let radial = about_axis(spin) >> dir(r * grow, 0.0, 0.0);
            for (radial, next) in step.orbit(radial, N) {
                let a = q + (tilt >> radial);
                let b = q + (tilt >> next);
                let velocity = tilt >> (quarter >> radial);
                let z = -0.55 * cosine(velocity, eye - a);
                let (a, b) = (view.of(a), view.of(b));
                p.cam_line(out, a, b, light::redshift(base, z), 0.05, Fog::Objects);
            }
        }
        let rim = (c.whitened(0.75)).faded(1.3);
        p.ring(out, &view, q, 1.2 * grow, rim, 32, 0.05);
    }

    /// Bolts, which grow as they come (perspective) and glow brighter when close; shards; and
    /// shots, streaks along their flight (bent ones curve round a singularity).
    fn draw_bolts_shards_shots(&self, out: &mut Vec<LineInstance>) {
        let (p, w, time) = (self.p, self.w, self.time);
        let s_ship = arc(self.ship);
        let spin = about_axis(time * 9.0);
        for b in &w.bolts {
            let q = lerp(b.prev, b.pos, self.alpha);
            let bolt = tint(BOLT, self.shift(q));
            let near = ((22.0 - (arc(q) - s_ship)) / 22.0).clamp(0.0, 1.0);
            let c = bolt.faded(1.0 + 2.5 * near);
            p.solid(out, &self.object(q, spin), &OCTAHEDRON, 0.32, c);
        }
        let spin = Motor::rotation_about(Point::direction(0.0, 1.0, 0.0), time * 4.0);
        for sh in &w.shards {
            p.solid(
                out,
                &self.object(sh.pos, spin),
                &OCTAHEDRON,
                0.3,
                scene::shard(),
            );
        }
        for b in &w.shots {
            let q = lerp(b.prev, b.pos, self.alpha);
            let c = tint(palette::BULLET, self.shift(q));
            p.streak(
                out,
                &self.small(q),
                q - b.vel * 0.018,
                q,
                c,
                0.07,
                Fog::Objects,
            );
        }
    }

    /// Sparks: streaks along their velocity.
    fn draw_sparks(&self, sparks: &[Spark], out: &mut Vec<LineInstance>) {
        for sp in sparks {
            let c = sp.light.faded(sp.life / sp.total);
            let tail = sp.pos - sp.vel * 0.03;
            self.p.streak(
                out,
                &self.small(sp.pos),
                tail,
                sp.pos,
                c,
                0.03,
                Fog::Objects,
            );
        }
    }

    /// The ship, banking into its movement and turning through a barrel roll, and its shadow
    /// on the wall.
    fn draw_ship(&self, out: &mut Vec<LineInstance>) {
        let (p, ship) = (self.p, &self.w.ship);
        let q = self.ship;
        let blink = ship.invulnerable > 0.0
            && ship.roll.is_none()
            && crate::signal::wave(ship.invulnerable * 14.0) < 0.0;
        // Close to the camera, the ship needs less light than on the Plane.
        let c = palette::SHIP.faded(if blink { 0.2 } else { 0.55 });
        let roll = ship.roll.map_or(0.0, |r| {
            r.dir * core::f32::consts::TAU * (r.t / ROLL_TIME).min(1.0)
        });
        // Banking into the movement across the tunnel.
        let sideways = (ship.pos - ship.prev).e032() / DT / 12.0;
        let pose = about_axis(roll - 0.25 * sideways);
        p.solid(out, &self.object(q, pose), &SHIP, 0.7, c);
        self.draw_shadow(q, out);
    }

    /// The ship's shadow: the line from a light on the axis just behind the ship, through the
    /// ship at `q`, meets the wall's tangent plane under it (the plane through the wall point
    /// orthogonal to the radial line). A small cross on the wall, along the track and around
    /// it.
    fn draw_shadow(&self, q: P, out: &mut Vec<LineInstance>) {
        let r = off_axis(q);
        if r <= 0.3 {
            return;
        }
        let base = foot(q);
        let wall_point = base + (q - base) * (RADIUS / r);
        let wall = wall_point | (base & q);
        let light_at = base - dir(0.0, 0.0, 2.0);
        let hit = ((light_at & q) ^ wall).unitized();
        let around = about_axis(core::f32::consts::FRAC_PI_2) >> ((q - base) * (0.5 / r));
        let along = dir(0.0, 0.0, 0.7);
        let shade = light::light(0.6, 0.75, 1.0, 1.2);
        for arm in [around, along] {
            let (a, b) = (self.place(hit - arm), self.place(hit + arm));
            self.p.line(out, a, b, shade, 0.05, Fog::Objects);
        }
    }

    /// The reticle, a lock bracket on the locked enemy, and a lead dot where the shots will
    /// cross its depth.
    fn draw_reticle(
        &self,
        reticle: Point2<(), f32>,
        lock: Option<u32>,
        out: &mut Vec<LineInstance>,
    ) {
        let (p, w) = (self.p, self.w);
        let (hud, still) = (palette::HUD, Pose::identity());
        let style = Stroke::new(0.05, 0.25, 0.3);
        // Four arms: a short segment turned a quarter at a time about the reticle.
        let o = reticle;
        let (a, b) = (
            o + Point2::direction(0.35, 0.0),
            o + Point2::direction(0.7, 0.0),
        );
        out.extend(quarters(o).map(|q| scene::seg(q >> a, q >> b, hud, style, still)));
        let Some(e) = lock.and_then(|id| w.enemies.iter().find(|e| e.id == id)) else {
            return;
        };
        let Some(c) = p.camera(self.place(e.pos)) else {
            return;
        };
        let centre = p.screen(c);
        let r = 1.0 + 0.1 * crate::signal::wave(self.time * 10.0);
        let lock = light::light(1.0, 0.9, 0.5, 2.0);
        let st = Stroke::new(0.05, 0.2, 0.3);
        // Four corner brackets, each a turn of the first.
        let k = centre + Point2::direction(r, r);
        let (u, v) = (
            k - Point2::direction(0.4, 0.0),
            k - Point2::direction(0.0, 0.4),
        );
        out.extend(
            quarters(centre)
                .flat_map(|q| [u, v].map(|end| scene::seg(q >> k, q >> end, lock, st, still))),
        );
        // Lead: where the enemy will be when the shots cross its depth.
        if let Some(lc) = p.camera(self.place(w.lead(e))) {
            let d = p.screen(lc);
            let (a, b) = (
                d - Point2::direction(0.12, 0.0),
                d + Point2::direction(0.12, 0.0),
            );
            let dot = Stroke::new(0.12, 0.3, 0.4);
            out.push(scene::seg(a, b, lock, dot, still));
        }
    }

    /// Popups, at their depth, rising as they fade.
    fn draw_popups(&self, popups: &[Popup], out: &mut Vec<LineInstance>) {
        let p = self.p;
        for pop in popups {
            if let Some(c) = p.camera(self.place(pop.pos)) {
                let at = p.screen(c) + Point2::direction(0.0, 1.0 + pop.age * 2.0);
                let size = (p.focal / p.depth(c)).clamp(0.4, 1.6);
                let color = pop.light.faded(pop.fade());
                scene::text(out, &pop.text, at, size, color, Align::Center);
            }
        }
    }
}

/// The four quarter turns about the screen point `o`, from none.
fn quarters(o: Point2<(), f32>) -> impl Iterator<Item = gax::Unit<gax::pga2d::Motor<(), f32>>> {
    (0..4).map(move |k| gax::pga2d::Motor::rotation(o, k as f32 * core::f32::consts::FRAC_PI_2))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::tunnel::{Event, Flight, Input, Shot, at};

    /// A world flown for `seconds` with an invulnerable ship.
    pub(crate) fn flown(seed: u64, seconds: usize) -> World {
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
        let p = Proj::new(cam, v.settings.focal);
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
        let p = &v.proj;
        let s = w.ship.s();
        let edge = crate::tunnel::around(crate::tunnel::SHIP_RADIUS, 0.0, s);
        let c = p.camera(w.track.place(edge)).expect("in front");
        let centre = p.camera(w.track.place(at(0.0, 0.0, s))).expect("in front");
        let r = p.screen(c).distance(p.screen(centre));
        assert!(r > 10.0, "the cross-section is {r} of 18 units");
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
                let cos = cosine(f(before), f(v.cam));
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

    /// A heavy frame on the CPU: a singularity in view collapses as the ship dies, under a
    /// storm of sparks and a shaking camera. Every line is finite, and there are not too many.
    #[test]
    fn a_heavy_frame_draws_finite_lines() {
        let (w, mut v) = collapse_and_death();
        let mut out = Vec::new();
        for frame in 0..4 {
            v.update(&w, 0.5, 1.0 / 60.0);
            v.draw(&w, 0.5, frame as f32 / 60.0, &mut out);
            let floats: &[f32] = bytemuck::cast_slice(&out);
            assert!(floats.iter().all(|x| x.is_finite()), "frame {frame}");
            assert!((1000..40_000).contains(&out.len()), "{} lines", out.len());
        }
    }

    /// A world where a singularity ahead has just collapsed and the ship has just died, and a
    /// view that has seen it.
    fn collapse_and_death() -> (World, View) {
        let mut w = flown(6, 3);
        let s = w.ship.s();
        let i = crate::tunnel::tests::spawn(&mut w, Foe::Singularity, at(1.0, 0.5, s + 22.0));
        w.enemies[i].flight = Flight::Dive;
        let mut v = View::new(85.0);
        for _ in 0..60 {
            w.tick(&Input::default());
            v.update(&w, 0.0, DT);
        }
        // A last shot into the singularity, and a bolt into the ship.
        let well = w
            .enemies
            .iter_mut()
            .find(|e| e.foe == Foe::Singularity)
            .expect("well");
        well.hp = 0.5;
        let at_well = well.pos;
        w.shots.push(Shot::new(
            at_well - dir(0.0, 0.0, 0.3),
            dir(0.0, 0.0, 30.0),
            1.0,
        ));
        w.ship.invulnerable = 0.0;
        let ship = w.ship.pos;
        w.bolts.push(Shot::new(
            ship + dir(0.0, 0.0, 0.5),
            dir(0.0, 0.0, -30.0),
            1.0,
        ));
        w.tick(&Input::default());
        let killed = w.events.iter().any(|e| {
            matches!(
                e,
                Event::Kill {
                    foe: Foe::Singularity,
                    ..
                }
            )
        });
        assert!(killed && w.events.iter().any(|e| matches!(e, Event::Death { .. })));
        v.on_events(&w.events);
        (w, v)
    }
}
