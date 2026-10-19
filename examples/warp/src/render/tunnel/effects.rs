//! The view's effects: sparks, score popups, screen shake and flashes, and a collapse's shock
//! ripple, set off by the simulation's events.

use super::super::scene::{self, palette};
use super::{BOLT, GATE_LIGHT, View};
use crate::light::{self, Light};
use crate::tunnel::{Event, Foe, P, dir, foot};
use gax::pga3d::{Motor, Point};

/// The most sparks alive at once (the oldest go first).
const SPARKS: usize = 6000;
/// How long a popup shows, and a shock ripple runs (seconds).
const POPUP_TIME: f32 = 1.1;
const SHOCK_TIME: f32 = 0.9;

/// A particle in straightened coordinates.
#[derive(Clone, Copy, Debug)]
pub(super) struct Spark {
    pub(super) pos: P,
    pub(super) vel: P,
    pub(super) light: Light,
    /// Seconds left, of `total`.
    pub(super) life: f32,
    pub(super) total: f32,
}

/// Text that rises from where something scored, and fades.
#[derive(Clone, Debug)]
pub(super) struct Popup {
    pub(super) pos: P,
    pub(super) text: String,
    /// Seconds shown.
    pub(super) age: f32,
    pub(super) light: Light,
}

impl Popup {
    /// How bright it still is, `1` to `0`.
    pub(super) fn fade(&self) -> f32 {
        (1.0 - self.age / POPUP_TIME).max(0.0)
    }
}

impl View {
    /// Sparks leaving `pos` in random directions: `(0, 0, v)` turned by a random rotation.
    fn burst(&mut self, pos: P, light: Light, n: usize, speed: f32, life: f32) {
        for _ in 0..n {
            let turn = Motor::rotation_about(
                Point::direction(
                    self.rng.range(-1.0, 1.0),
                    self.rng.range(-1.0, 1.0),
                    self.rng.range(-1.0, 1.0),
                ),
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
        if self.sparks.len() > SPARKS {
            self.sparks.drain(..self.sparks.len() - SPARKS);
        }
    }

    /// A popup at `pos`.
    fn popup(&mut self, pos: P, text: String, light: Light) {
        self.popups.push(Popup {
            pos,
            text,
            age: 0.0,
            light,
        });
    }

    /// React to a tick's events.
    pub fn on_events(&mut self, events: &[Event]) {
        let flash_scale = self.settings.flash_scale;
        for e in events {
            match *e {
                Event::Hit { pos, foe } => {
                    self.burst(pos, scene::color(foe.kind()), 6, 6.0, 0.25);
                }
                Event::Kill { pos, foe, points } => self.on_kill(pos, foe, points),
                Event::Wall { pos } => self.burst(pos, palette::BULLET, 4, 4.0, 0.2),
                Event::Bolt { pos } => self.burst(pos, BOLT, 10, 3.0, 0.3),
                Event::Dive { pos } => {
                    self.burst(pos, scene::color(Foe::Drone.kind()), 14, 5.0, 0.3);
                }
                Event::Pickup { pos, mult } => {
                    self.burst(pos, scene::shard(), 8, 4.0, 0.3);
                    if mult % 10 == 0 {
                        self.popup(pos, format!("X{mult}"), scene::shard());
                    }
                }
                Event::Death { pos } => {
                    let k = 0.35 + 0.65 * flash_scale;
                    self.burst(pos, palette::SHIP.faded(k), 900, 22.0, 1.8);
                    self.shake = 1.6;
                    self.flash = 0.7;
                }
                Event::Bomb { pos } => {
                    let k = 0.35 + 0.65 * flash_scale;
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
                    self.popup(pos, format!("+{points}"), GATE_LIGHT);
                    if chain >= 3 && chain % 3 == 0 {
                        let p = pos + dir(0.0, 1.2, 0.0);
                        self.popup(p, format!("CHAIN X{chain}"), GATE_LIGHT);
                    }
                }
                Event::GateMiss { pos } => {
                    let grey = GATE_LIGHT.perceptually(|c| c.desaturated(1.0));
                    self.burst(pos, grey.faded(0.5), 20, 4.0, 0.4);
                }
                Event::Absorb { pos, .. } => {
                    let c = scene::color(Foe::Singularity.kind());
                    self.burst(pos, c.whitened(0.4), 30, 5.0, 0.3);
                }
                Event::Burst { pos } => {
                    let c = scene::color(Foe::Singularity.kind());
                    self.burst(pos, c, 300, 18.0, 1.0);
                    self.shake = self.shake.max(0.8);
                }
                Event::Slingshot { pos } => {
                    let c = (scene::color(Foe::Singularity.kind())).whitened(0.3);
                    self.popup(pos + dir(0.0, 1.4, 0.0), "SLINGSHOT".into(), c);
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

    /// A kill: sparks in its colour and a white flash of them, a shake, its points; a
    /// singularity collapses in a shock ripple, a flash and a storm of light.
    fn on_kill(&mut self, pos: P, foe: Foe, points: u64) {
        let c = scene::color(foe.kind());
        let big = matches!(foe, Foe::Turret | Foe::Singularity);
        self.burst(pos, c, if big { 160 } else { 70 }, 11.0, 0.9);
        self.burst(pos, light::light(1.0, 1.0, 1.0, 3.0), 12, 16.0, 0.3);
        self.shake = self.shake.max(if big { 0.5 } else { 0.15 });
        if points >= 250 {
            self.popup(pos, points.to_string(), c);
        }
        if foe == Foe::Singularity {
            self.shock = Some((pos, 0.0));
            self.flash = self.flash.max(0.45);
            self.shake = self.shake.max(1.0);
            self.burst(pos, c.whitened(0.5), 700, 30.0, 1.4);
        }
    }

    /// The effects a frame of `dt` later: shake and flash decay, sparks fly and slow, popups
    /// rise, the shock ripple spreads.
    pub(super) fn age_effects(&mut self, dt: f32) {
        self.shake = (self.shake - dt * 2.2).max(0.0);
        self.flash = (self.flash - dt * 2.5).max(0.0);
        for p in &mut self.sparks {
            p.pos += p.vel * dt;
            p.vel = p.vel * (1.0 - 1.8 * dt);
            p.life -= dt;
        }
        self.sparks.retain(|p| p.life > 0.0);
        for p in &mut self.popups {
            p.age += dt;
        }
        self.popups.retain(|p| p.age < POPUP_TIME);
        if let Some((_, t)) = &mut self.shock {
            *t += dt;
            if *t > SHOCK_TIME {
                self.shock = None;
            }
        }
    }

    /// Post settings for a frame of `size` pixels.
    pub fn post(&self, w: &crate::tunnel::World, size: [u32; 2]) -> super::super::PostSettings {
        let flash_scale = self.settings.flash_scale;
        // The collapse's ripple, from where the singularity was on screen (in HUD units, which
        // the HUD's view map takes to the target's uv).
        let hud = scene::view_map(
            crate::sim::body::Pose::identity(),
            scene::HUD_HALF_HEIGHT,
            size,
        );
        let shock = self
            .shock
            .filter(|_| self.settings.lensing)
            .and_then(|(q, t)| {
                Some(super::super::Shock {
                    at: scene::to_uv(&hud, self.on_screen(w, q)?),
                    radius: t * 1.1,
                    strength: 0.05 * (1.0 - t / SHOCK_TIME) * (0.4 + 0.6 * flash_scale),
                })
            });
        super::super::PostSettings {
            bloom: 0.34,
            exposure: 1.0 + self.flash * 1.5 * flash_scale,
            vignette: 0.45,
            grain: 0.012,
            aberration: 0.007,
            saturation: 1.35,
            shock,
        }
    }
}
