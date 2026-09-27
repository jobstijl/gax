//! Effects that are not the simulation: the camera, screen shake, particle bursts, grid blasts,
//! the bomb's shock ripple. Driven by the simulation's events; purely visual.
//!
//! The camera follows its target with a critically damped spring in the Lie algebra: the
//! error is `log(target ~cam)` (a twist), not an angle; shake is a small random twist
//! `exp(ε B)` composed onto the camera pose. Particles leave along directions turned by
//! rotation motors; their colours are lights.

use crate::light::{self, Light};
use crate::render::scene::{self, palette};
use crate::render::{Particle, PostSettings};
use crate::sim::body::{Pose, heading, pose_at, turned};
use crate::sim::rng::Rng;
use crate::sim::{ARENA, Event, Kind, World};
use gax::pga2d::{Motor, Point};

type P = Point<(), f32>;

struct Blast {
    pos: P,
    strength: f32,
    r2: f32,
    life: f32,
    total: f32,
}

/// Effects state.
pub struct Fx {
    /// The camera pose (without shake).
    pub cam: Pose,
    cam_vel: P,
    shake: f32,
    rng: Rng,
    blasts: Vec<Blast>,
    /// Particles to spawn this frame.
    pub spawn: Vec<Particle>,
    shock: Option<(P, f32)>,
    /// A full-screen flash, decaying.
    pub flash: f32,
    /// Screen shake scale (a setting).
    pub shake_scale: f32,
    /// Flash scale: 1, or less for the reduced-flashes setting (full-screen flashes, big
    /// bursts and the shock ripple).
    pub flash_scale: f32,
    /// Less motion (a setting): no shock ripple.
    pub reduced_motion: bool,
    /// The grid's sources per simulation tick this frame: `[x, y, strength, r²]`, the layout
    /// of the traced `source_force` kernel on the GPU.
    pub grid_steps: Vec<Vec<[f32; 4]>>,
    /// Floating texts: `(position, text, age, light)`.
    pub popups: Vec<(P, String, f32, Light)>,
    /// The view's half height and aspect ratio (for the camera's bounds).
    pub half_height: f32,
    /// Width over height of the view.
    pub aspect: f32,
}

impl Fx {
    /// Effects at rest.
    pub fn new() -> Fx {
        Fx {
            cam: pose_at(0.0, 0.0, 0.0),
            cam_vel: Point::direction(0.0, 0.0),
            shake: 0.0,
            rng: Rng::new(0xf00d),
            blasts: Vec::new(),
            spawn: Vec::new(),
            shock: None,
            flash: 0.0,
            shake_scale: 1.0,
            flash_scale: 1.0,
            reduced_motion: false,
            grid_steps: Vec::new(),
            popups: Vec::new(),
            half_height: 14.0,
            aspect: 16.0 / 9.0,
        }
    }

    fn particle(&mut self, p: P, v: P, color: Light, life: [f32; 4]) {
        self.spawn.push(Particle {
            p: p.into(),
            v: v.into(),
            color: color.into(),
            life,
        });
    }

    fn burst(
        &mut self,
        pos: P,
        color: Light,
        n: usize,
        speed: (f32, f32),
        life: (f32, f32),
        hot: f32,
    ) {
        // Big bursts are the bright ones.
        let (color, hot) = if n >= 200 {
            let k = 0.35 + 0.65 * self.flash_scale;
            (light::fade(color, k), hot * k)
        } else {
            (color, hot)
        };
        for _ in 0..n {
            let v = heading(self.rng.angle(), self.rng.range(speed.0, speed.1));
            // Some sparks leave white: the same light, all the way to white, a little brighter.
            let c = if self.rng.chance(hot) {
                light::fade(light::whiten(color, 1.0), 1.3)
            } else {
                color
            };
            let life = [
                0.0,
                self.rng.range(life.0, life.1),
                self.rng.range(1.2, 3.0),
                self.rng.range(0.025, 0.05),
            ];
            self.particle(pos, v, c, life);
        }
    }

    fn blast(&mut self, pos: P, strength: f32, r2: f32, life: f32) {
        self.blasts.push(Blast {
            pos,
            strength,
            r2,
            life,
            total: life,
        });
    }

    /// React to a tick's events.
    pub fn on_events(&mut self, events: &[Event]) {
        for e in events {
            match *e {
                Event::Fire { pos, dir } => {
                    let muzzle = pos + dir * 0.6;
                    for _ in 0..2 {
                        let d = turned(dir, self.rng.range(-0.4, 0.4));
                        let v = d * self.rng.range(6.0, 14.0);
                        self.particle(muzzle, v, palette::BULLET, [0.0, 0.12, 6.0, 0.02]);
                    }
                }
                Event::Hit { pos, kind } => {
                    // Sparks in the enemy's colour; a singularity's stay violet (it takes many hits).
                    let hot = if kind == Kind::Singularity { 0.0 } else { 0.3 };
                    let n = if kind == Kind::Singularity { 4 } else { 10 };
                    let c = scene::color(kind);
                    self.burst(pos, c, n, (4.0, 12.0), (0.15, 0.4), hot);
                    if kind == Kind::Singularity {
                        self.blast(pos, 30.0, 2.0, 0.06);
                    }
                }
                Event::Kill {
                    pos,
                    kind,
                    size,
                    scored,
                    points,
                } => {
                    let p = pos;
                    let c = scene::color(kind);
                    if points >= 250 {
                        self.popups.push((p, points.to_string(), 0.0, c));
                    }
                    let n = (70.0 * size) as usize + if scored { 20 } else { 0 };
                    let reach = 16.0 * gax::Real::sqrt(size);
                    self.burst(p, c, n, (3.0, reach), (0.5, 1.4), 0.15);
                    self.blast(p, -70.0 * size, 2.5 * size, 0.12);
                    self.shake = self.shake.max(0.12 * size);
                    if kind == Kind::Singularity {
                        let white = light::light(1.0, 1.0, 1.0, 4.0);
                        self.burst(p, white, 300, (10.0, 30.0), (0.6, 1.5), 0.0);
                        self.blast(p, -450.0, 9.0, 0.22);
                        self.shake = self.shake.max(0.8);
                    }
                }
                Event::Deflect { pos, kind } => {
                    let c = scene::color(kind);
                    self.burst(pos, c, 6, (6.0, 14.0), (0.1, 0.25), 0.5);
                }
                Event::Wall { pos } => {
                    self.burst(pos, palette::BULLET, 5, (2.0, 8.0), (0.1, 0.3), 0.2);
                    self.blast(pos, -25.0, 0.5, 0.05);
                }
                Event::Absorb { pos, .. } => {
                    let c = palette::SINGULARITY;
                    self.burst(pos, c, 40, (1.0, 5.0), (0.3, 0.8), 0.3);
                    self.blast(pos, 90.0, 3.0, 0.15);
                }
                Event::Burst { pos } => {
                    self.burst(pos, palette::MOTE, 600, (6.0, 26.0), (0.6, 1.8), 0.2);
                    self.blast(pos, -550.0, 9.0, 0.25);
                    self.shake = self.shake.max(0.9);
                }
                Event::Warn { pos, kind } => {
                    // Particles converging on the spot: from a ring around it, inwards.
                    let c = light::fade(scene::color(kind), 0.6);
                    for _ in 0..24 {
                        let out = heading(self.rng.angle(), self.rng.range(1.5, 3.0));
                        self.particle(pos + out, out * -2.5, c, [0.0, 0.4, 0.5, 0.03]);
                    }
                }
                Event::Spawn { pos, kind } => {
                    let c = scene::color(kind);
                    self.burst(pos, c, 16, (2.0, 6.0), (0.2, 0.5), 0.5);
                    if kind == Kind::Singularity {
                        self.blast(pos, 300.0, 6.0, 0.4);
                    }
                }
                Event::Pickup { pos, mult } => {
                    if mult % 10 == 0 {
                        self.popups
                            .push((pos, format!("X{mult}"), 0.0, scene::shard()));
                    }
                    self.burst(pos, scene::shard(), 6, (2.0, 6.0), (0.2, 0.45), 0.3);
                }
                Event::Death { pos } => {
                    let p = pos;
                    self.burst(p, palette::SHIP, 1500, (4.0, 36.0), (0.8, 2.4), 0.3);
                    self.blast(p, -1300.0, 12.0, 0.3);
                    self.shake = 1.6;
                    self.flash = 0.7;
                    self.shock = Some((p, 0.0));
                }
                Event::Bomb { pos } => {
                    // A ring of light: one spark every 1/2400 of a turn.
                    let p = pos;
                    let c = light::light(0.55, 0.8, 1.0, 3.5 * (0.35 + 0.65 * self.flash_scale));
                    let step =
                        Motor::rotation(Point::xy(0.0, 0.0), core::f32::consts::TAU / 2400.0);
                    let mut dir = Point::direction(1.0, 0.0);
                    for _ in 0..2400 {
                        let v = dir * self.rng.range(28.0, 42.0);
                        let life = [0.0, self.rng.range(0.8, 1.3), 1.2, 0.03];
                        self.particle(p, v, c, life);
                        dir = step >> dir;
                    }
                    self.blast(p, -2600.0, 60.0, 0.35);
                    self.shake = 1.2;
                    self.flash = 0.5;
                    self.shock = Some((p, 0.0));
                }
                Event::Respawn => {
                    let o = Point::xy(0.0, 0.0);
                    self.burst(o, palette::SHIP, 200, (2.0, 10.0), (0.3, 0.8), 0.5);
                    self.blast(o, 450.0, 9.0, 0.25);
                }
                Event::Extra { .. } | Event::GameOver => {}
            }
        }
    }

    /// The grid's sources for one simulation tick (blasts and wells), and age the blasts.
    pub fn grid_tick(&mut self, w: &World, dt: f32) {
        let mut s: Vec<[f32; 4]> = Vec::new();
        let source = |p: P, strength: f32, r2: f32| {
            let [x, y] = p.to_euclidean();
            [x, y, strength, r2]
        };
        for b in &mut self.blasts {
            let k = (b.life / b.total).clamp(0.0, 1.0);
            s.push(source(b.pos, b.strength * k, b.r2));
            b.life -= dt;
        }
        self.blasts.retain(|b| b.life > 0.0);
        for (p, strength) in w.wells() {
            s.push(source(p, strength * 3.0, 12.0));
        }
        // The ship's wake: a light push where it flies.
        if w.phase == crate::sim::Phase::Playing {
            let v = w.ship.body.vel.ideal_norm();
            if v > 1.0 {
                s.push(source(w.ship.body.pos(), -3.0 * v, 0.8));
            }
        }
        self.grid_steps.push(s);
    }

    /// The wells as particle attractors.
    pub fn particle_wells(w: &World) -> Vec<[f32; 4]> {
        w.wells()
            .map(|(p, s)| {
                let [x, y] = p.to_euclidean();
                [x, y, s * 1.2, 4.0]
            })
            .collect()
    }

    /// Advance the camera by a frame of `dt`: towards a point between the arena's centre and
    /// the ship, and shake. Returns the camera pose to draw with.
    pub fn camera(&mut self, w: &World, dt: f32) -> Pose {
        let [sx, sy] = w.ship.body.pos().to_euclidean();
        // Follow the ship, but keep the view inside the arena (with a small margin).
        let (hw, hh) = (self.half_height * self.aspect, self.half_height);
        let room = |arena: f32, half: f32| (arena + 2.0 - half).max(0.0);
        let target = Motor::translation(
            (sx * 0.9).clamp(-room(ARENA[0], hw), room(ARENA[0], hw)),
            (sy * 0.9).clamp(-room(ARENA[1], hh), room(ARENA[1], hh)),
        );
        // Critically damped spring on the twist error log(target ~cam).
        let omega = 5.0;
        let err: P = (target * self.cam.reverse()).log();
        self.cam_vel += (err * (omega * omega) - self.cam_vel * (2.0 * omega)) * dt;
        self.cam = ((self.cam_vel * dt).exp() * self.cam).renormalize_fast();
        for p in &mut self.popups {
            p.2 += dt;
        }
        self.popups.retain(|p| p.2 < 1.1);
        self.shake = (self.shake - dt * 2.2).max(0.0);
        self.flash = (self.flash - dt * 2.5).max(0.0);
        if let Some((_, t)) = &mut self.shock {
            *t += dt;
            if *t > 0.8 {
                self.shock = None;
            }
        }
        let a = self.shake * self.shake * self.shake_scale;
        if a > 1e-4 {
            let jitter = Point::translation_twist(
                self.rng.range(-1.0, 1.0) * a,
                self.rng.range(-1.0, 1.0) * a,
            ) + Point::rotation_twist(
                self.cam >> Point::xy(0.0, 0.0),
                self.rng.range(-0.02, 0.02) * a,
            );
            jitter.exp() * self.cam
        } else {
            self.cam
        }
    }

    /// Post settings for the frame: the shock ripple placed where the blast was, through the
    /// view map.
    pub fn post(&self, view: &Point<(Point,), f32>) -> PostSettings {
        let shock = self.shock.filter(|_| !self.reduced_motion).map(|(p, t)| {
            let uv = scene::to_uv(view, p);
            let strength = 0.035 * (1.0 - t / 0.8) * (0.4 + 0.6 * self.flash_scale);
            (uv, t * 0.9, strength)
        });
        PostSettings {
            bloom: 0.32,
            exposure: 1.0 + self.flash * 1.5 * self.flash_scale,
            vignette: 0.35,
            grain: 0.012,
            aberration: 0.006,
            saturation: 1.35,
            shock,
        }
    }
}
