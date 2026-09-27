//! Effects that are not the simulation: the camera, screen shake, particle bursts, grid blasts,
//! the bomb's shock ripple. Driven by the simulation's events; purely visual.
//!
//! The camera follows its target with a critically damped spring in the Lie algebra: the
//! error is `log(target ~cam)` (a twist), not an angle; shake is a small random twist
//! `exp(ε B)` composed onto the camera pose.

use crate::render::scene::{self, palette};
use crate::render::{Particle, PostSettings};
use crate::sim::body::{Pose, pose_at};
use crate::sim::rng::Rng;
use crate::sim::{ARENA, Event, Kind, World};
use gax::pga2d::Point;

struct Blast {
    pos: [f32; 2],
    strength: f32,
    r2: f32,
    life: f32,
    total: f32,
}

/// Effects state.
pub struct Fx {
    /// The camera pose (without shake).
    pub cam: Pose,
    cam_vel: Point<(), f32>,
    shake: f32,
    rng: Rng,
    blasts: Vec<Blast>,
    /// Particles to spawn this frame.
    pub spawn: Vec<Particle>,
    shock: Option<([f32; 2], f32)>,
    /// A full-screen flash, decaying.
    pub flash: f32,
    /// Screen shake scale (a setting).
    pub shake_scale: f32,
    /// The grid's sources per simulation tick this frame.
    pub grid_steps: Vec<Vec<[f32; 4]>>,
    /// Floating texts: `(position, text, age, colour)`.
    pub popups: Vec<([f32; 2], String, f32, [f32; 4])>,
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
            grid_steps: Vec::new(),
            popups: Vec::new(),
            half_height: 14.0,
            aspect: 16.0 / 9.0,
        }
    }

    fn burst(
        &mut self,
        pos: [f32; 2],
        color: [f32; 4],
        n: usize,
        speed: (f32, f32),
        life: (f32, f32),
        hot: f32,
    ) {
        for _ in 0..n {
            let a = self.rng.angle();
            let s = self.rng.range(speed.0, speed.1);
            let white = self.rng.chance(hot);
            let c = if white {
                [1.0, 1.0, 1.0, color[3] * 1.3]
            } else {
                color
            };
            self.spawn.push(Particle {
                p: Point::xy(pos[0], pos[1]).into(),
                v: Point::direction(a.cos() * s, a.sin() * s).into(),
                color: c,
                life: [
                    0.0,
                    self.rng.range(life.0, life.1),
                    self.rng.range(1.2, 3.0),
                    self.rng.range(0.025, 0.05),
                ],
            });
        }
    }

    fn blast(&mut self, pos: [f32; 2], strength: f32, r2: f32, life: f32) {
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
                Event::Fire { pos, angle } => {
                    let (c, s) = (angle.cos(), angle.sin());
                    for _ in 0..2 {
                        let a = angle + self.rng.range(-0.4, 0.4);
                        let sp = self.rng.range(6.0, 14.0);
                        self.spawn.push(Particle {
                            p: Point::xy(pos[0] + c * 0.6, pos[1] + s * 0.6).into(),
                            v: Point::direction(a.cos() * sp, a.sin() * sp).into(),
                            color: palette::BULLET,
                            life: [0.0, 0.12, 6.0, 0.02],
                        });
                    }
                }
                Event::Hit { pos, kind } => {
                    // Sparks in the enemy's colour; a singularity's stay violet (it takes many hits).
                    let hot = if kind == Kind::Singularity { 0.0 } else { 0.3 };
                    let n = if kind == Kind::Singularity { 4 } else { 10 };
                    self.burst(pos, scene::color(kind), n, (4.0, 12.0), (0.15, 0.4), hot);
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
                    if points >= 250 {
                        self.popups
                            .push((pos, points.to_string(), 0.0, scene::color(kind)));
                    }
                    let n = (70.0 * size) as usize + if scored { 20 } else { 0 };
                    self.burst(
                        pos,
                        scene::color(kind),
                        n,
                        (3.0, 16.0 * size.sqrt()),
                        (0.5, 1.4),
                        0.15,
                    );
                    self.blast(pos, -70.0 * size, 2.5 * size, 0.12);
                    self.shake = self.shake.max(0.12 * size);
                    if kind == Kind::Singularity {
                        self.burst(
                            pos,
                            [1.0, 1.0, 1.0, 4.0],
                            300,
                            (10.0, 30.0),
                            (0.6, 1.5),
                            0.0,
                        );
                        self.blast(pos, -450.0, 9.0, 0.22);
                        self.shake = self.shake.max(0.8);
                    }
                }
                Event::Wall { pos } => {
                    self.burst(pos, palette::BULLET, 5, (2.0, 8.0), (0.1, 0.3), 0.2);
                    self.blast(pos, -25.0, 0.5, 0.05);
                }
                Event::Absorb { pos, .. } => {
                    self.burst(pos, palette::SINGULARITY, 40, (1.0, 5.0), (0.3, 0.8), 0.3);
                    self.blast(pos, 90.0, 3.0, 0.15);
                }
                Event::Burst { pos } => {
                    self.burst(pos, palette::MOTE, 600, (6.0, 26.0), (0.6, 1.8), 0.2);
                    self.blast(pos, -550.0, 9.0, 0.25);
                    self.shake = self.shake.max(0.9);
                }
                Event::Warn { pos, kind } => {
                    // Particles converging on the spot.
                    let c = scene::color(kind);
                    for _ in 0..24 {
                        let a = self.rng.angle();
                        let r = self.rng.range(1.5, 3.0);
                        let p = [pos[0] + a.cos() * r, pos[1] + a.sin() * r];
                        self.spawn.push(Particle {
                            p: Point::xy(p[0], p[1]).into(),
                            v: Point::direction(-a.cos() * r * 2.5, -a.sin() * r * 2.5).into(),
                            color: [c[0], c[1], c[2], c[3] * 0.6],
                            life: [0.0, 0.4, 0.5, 0.03],
                        });
                    }
                }
                Event::Spawn { pos, kind } => {
                    self.burst(pos, scene::color(kind), 16, (2.0, 6.0), (0.2, 0.5), 0.5);
                    if kind == Kind::Singularity {
                        self.blast(pos, 300.0, 6.0, 0.4);
                    }
                }
                Event::Pickup { pos, mult } => {
                    if mult % 10 == 0 {
                        self.popups
                            .push((pos, format!("X{mult}"), 0.0, palette::SHARD));
                    }
                    self.burst(pos, palette::SHARD, 6, (2.0, 6.0), (0.2, 0.45), 0.3);
                }
                Event::Death { pos } => {
                    self.burst(pos, palette::SHIP, 1500, (4.0, 36.0), (0.8, 2.4), 0.3);
                    self.blast(pos, -1300.0, 12.0, 0.3);
                    self.shake = 1.6;
                    self.flash = 0.7;
                    self.shock = Some((pos, 0.0));
                }
                Event::Bomb { pos } => {
                    for k in 0..2400 {
                        let a = k as f32 / 2400.0 * core::f32::consts::TAU;
                        let s = self.rng.range(28.0, 42.0);
                        self.spawn.push(Particle {
                            p: Point::xy(pos[0], pos[1]).into(),
                            v: Point::direction(a.cos() * s, a.sin() * s).into(),
                            color: [0.55, 0.8, 1.0, 3.5],
                            life: [0.0, self.rng.range(0.8, 1.3), 1.2, 0.03],
                        });
                    }
                    self.blast(pos, -2600.0, 60.0, 0.35);
                    self.shake = 1.2;
                    self.flash = 0.5;
                    self.shock = Some((pos, 0.0));
                }
                Event::Respawn => {
                    self.burst([0.0, 0.0], palette::SHIP, 200, (2.0, 10.0), (0.3, 0.8), 0.5);
                    self.blast([0.0, 0.0], 450.0, 9.0, 0.25);
                }
                Event::Extra { .. } | Event::GameOver => {}
            }
        }
    }

    /// The grid's sources for one simulation tick (blasts and wells), and age the blasts.
    pub fn grid_tick(&mut self, w: &World, dt: f32) {
        let mut s: Vec<[f32; 4]> = Vec::new();
        for b in &mut self.blasts {
            let k = (b.life / b.total).clamp(0.0, 1.0);
            s.push([b.pos[0], b.pos[1], b.strength * k, b.r2]);
            b.life -= dt;
        }
        self.blasts.retain(|b| b.life > 0.0);
        for (p, strength) in w.wells() {
            s.push([p[0], p[1], strength * 3.0, 12.0]);
        }
        // The ship's wake: a light push where it flies.
        if w.phase == crate::sim::Phase::Playing {
            let v = crate::sim::body::length(w.ship.body.vel);
            if v > 1.0 {
                let p = w.ship.body.xy();
                s.push([p[0], p[1], -3.0 * v, 0.8]);
            }
        }
        self.grid_steps.push(s);
    }

    /// The wells as particle attractors.
    pub fn particle_wells(w: &World) -> Vec<[f32; 4]> {
        w.wells().map(|(p, s)| [p[0], p[1], s * 1.2, 4.0]).collect()
    }

    /// Advance the camera by a frame of `dt`: towards a point between the arena's centre and
    /// the ship, and shake. Returns the camera pose to draw with.
    pub fn camera(&mut self, w: &World, dt: f32) -> Pose {
        let ship = w.ship.body.xy();
        // Follow the ship, but keep the view inside the arena (with a small margin).
        let (hw, hh) = (self.half_height * self.aspect, self.half_height);
        let room = |arena: f32, half: f32| (arena + 2.0 - half).max(0.0);
        let target = pose_at(
            (ship[0] * 0.9).clamp(-room(ARENA[0], hw), room(ARENA[0], hw)),
            (ship[1] * 0.9).clamp(-room(ARENA[1], hh), room(ARENA[1], hh)),
            0.0,
        );
        // Critically damped spring on the twist error log(target ~cam).
        let omega = 5.0;
        let err: Point<(), f32> = (target * self.cam.reverse()).log();
        self.cam_vel = self.cam_vel + (err.gp(omega * omega) - self.cam_vel.gp(2.0 * omega)).gp(dt);
        self.cam = (self.cam_vel.gp(dt).exp() * self.cam).renormalize_fast();
        for p in &mut self.popups {
            p.2 += dt;
        }
        self.popups.retain(|p| p.2 < 1.1);
        self.shake = (self.shake - dt * 2.2).max(0.0);
        self.flash = (self.flash - dt * 2.5).max(0.0);
        if let Some((p, t)) = &mut self.shock {
            *t += dt;
            if *t > 0.8 {
                self.shock = None;
            } else {
                let _ = p;
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

    /// Post settings for the frame: the shock ripple placed where the blast was.
    pub fn post(&self, cam: &crate::render::CameraUniform) -> PostSettings {
        let shock = self.shock.map(|(p, t)| {
            let uv = scene::to_uv(cam, p);
            (uv, t * 0.9, 0.035 * (1.0 - t / 0.8))
        });
        PostSettings {
            bloom: 0.32,
            exposure: 1.0 + self.flash * 1.5,
            vignette: 0.35,
            grain: 0.012,
            aberration: 0.006,
            saturation: 1.35,
            shock,
        }
    }
}
