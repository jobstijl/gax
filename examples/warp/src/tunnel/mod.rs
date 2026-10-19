//! Tunnel: flying into the screen down a twisting tunnel of warped space.
//!
//! A pure simulation like the Plane's (`sim`): a fixed 120 Hz step from a seed. Gameplay lives
//! in straightened coordinates (`space`), a PGA3D space where `x` and `y` run across the tunnel
//! and `z` is the arc length `s` along it; `track` maps it into the world through a chain of
//! screw motions, which only the drawing needs.
//!
//! Everything here is gax: positions are points, velocities directions, the tunnel's axis a
//! line. The distance from the axis is a join with it, the pull back inside moves towards a
//! point's foot on the axis (`(p | axis) ^ axis`), turns about the axis are motors (rings,
//! barrel rolls, wall placements), mines bounce by reflection in a plane, and collisions are
//! swept segments against spheres with joins. More:
//!
//! * A serpent swims a screw motion about the axis: its head is turned by `exp(DT T)` every
//!   tick, and each body segment is the head moved back along the screw by its own fixed
//!   motor `exp(-lag T)`, so the body is always the helix the head has just swum.
//! * A gate is a ring in a cross-section; the ship flies through it when the line of its
//!   last step meets the gate's plane inside the ring. A slalom of gates lies on a screw
//!   motion too, each gate the last one moved by the same motor.
//! * A singularity pulls (an inverse square, softened), pinches the tunnel's wall towards the
//!   axis around it, bends shots (a shot bent far enough before it kills is a slingshot), eats
//!   mines and grows, and bursts when overfed.

mod combat;
mod director;
mod enemies;
mod entities;
pub mod lattice;
pub mod replay;
mod space;
pub mod track;
mod wells;

pub use entities::{Enemy, Event, Flight, Foe, Gate, GateState, Pending, Roll, Shard, Ship, Shot};
pub use space::{P, about_axis, across, arc, around, at, dir, foot, off_axis};
pub use wells::{PINCH, PINCH_REACH2, Wells};

use crate::sim::rng::Rng;
use crate::sim::{DT, Ledger, Phase};
use gax::IdealNorm;
use lattice::{Bulge, Lattice};
use track::Track;

/// The ship stays this close to the axis.
pub const SHIP_RADIUS: f32 = 5.3;
/// The cruising speed (arc length per second); throttle adds or takes up to 45%.
pub const CRUISE: f32 = 17.0;
/// How far ahead an unlocked reticle aims.
pub const AIM_DEPTH: f32 = 34.0;
/// Enemies closer than this ahead of the ship are not locked onto.
pub const LOCK_AHEAD: f32 = 2.0;
/// Spawns land this far ahead (inside the fog).
const SPAWN_AHEAD: f32 = 80.0;
/// A barrel roll's duration, and how far it turns the ship about the axis.
pub const ROLL_TIME: f32 = 0.4;
const ROLL_ANGLE: f32 = 1.3;
/// A gate's radius.
pub const GATE_RADIUS: f32 = 1.7;

/// The player's input for a tick, already in the tunnel's frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Input {
    /// Movement across the tunnel: a direction of length at most 1.
    pub movement: P,
    /// Where the reticle aims: a point in straightened coordinates (a locked enemy, or where
    /// the camera's ray through the reticle meets the tunnel's cross-section at `AIM_DEPTH`).
    pub aim: P,
    /// Fire held.
    pub fire: bool,
    /// Bomb (edge).
    pub bomb: bool,
    /// Barrel roll (edge): `-1` or `1`.
    pub roll: i8,
    /// Brake (`-1`) to boost (`1`).
    pub throttle: f32,
}

impl Default for Input {
    fn default() -> Input {
        Input {
            movement: P::zero(),
            aim: at(0.0, 0.0, AIM_DEPTH),
            fire: false,
            bomb: false,
            roll: 0,
            throttle: 0.0,
        }
    }
}

/// The tunnel's whole state.
pub struct World {
    rng: Rng,
    /// Ticks since the start.
    pub tick: u64,
    /// Seconds since the start.
    pub time: f32,
    /// The track.
    pub track: Track,
    /// The wall.
    pub lattice: Lattice,
    /// The ship.
    pub ship: Ship,
    /// Enemies.
    pub enemies: Vec<Enemy>,
    /// The player's shots.
    pub shots: Vec<Shot>,
    /// Enemy bolts.
    pub bolts: Vec<Shot>,
    /// Shards.
    pub shards: Vec<Shard>,
    /// Bonus gates.
    pub gates: Vec<Gate>,
    /// Gates flown through in a row.
    pub chain: u32,
    /// Spawns on their way.
    pub pending: Vec<Pending>,
    /// The score, multiplier, lives and bombs.
    pub ledger: Ledger,
    /// Where the run is.
    pub phase: Phase,
    /// This tick's events.
    pub events: Vec<Event>,
    /// Bulges that last a while (blasts), with their time left.
    blasts: Vec<(Bulge, f32)>,
    /// Intensity, `0..1`.
    pub intensity: f32,
    /// The director's clock (seconds).
    clock: f32,
    /// Seconds to the next formation.
    spawn_cooldown: f32,
    /// Seconds to the next drone's dive.
    dive_cooldown: f32,
    /// Seconds to the next slalom.
    gate_timer: f32,
    /// Seconds before another singularity may come.
    well_cooldown: f32,
    /// Whether a serpent has come yet (the first comes by a deadline).
    serpent_introduced: bool,
    /// Whether a singularity has come yet (likewise).
    well_introduced: bool,
    next_id: u32,
}

impl World {
    /// A new run.
    pub fn new(seed: u64) -> World {
        let mut track = Track::new(seed);
        track.extend(SPAWN_AHEAD + 60.0, 0.0);
        let start = at(0.0, -2.0, 0.0);
        World {
            rng: Rng::new(seed ^ 0x7e11),
            tick: 0,
            time: 0.0,
            track,
            lattice: Lattice::new(0.0),
            ship: Ship {
                pos: start,
                prev: start,
                speed: CRUISE,
                cooldown: 0.0,
                invulnerable: 2.0,
                roll: None,
                roll_cool: 0.0,
                barrel: false,
            },
            enemies: Vec::new(),
            shots: Vec::new(),
            bolts: Vec::new(),
            shards: Vec::new(),
            gates: Vec::new(),
            chain: 0,
            pending: Vec::new(),
            ledger: Ledger::new(),
            phase: Phase::Playing,
            events: Vec::new(),
            blasts: Vec::new(),
            intensity: 0.1,
            clock: 0.0,
            spawn_cooldown: 2.0,
            dive_cooldown: 2.0,
            gate_timer: 12.0,
            well_cooldown: 0.0,
            serpent_introduced: false,
            well_introduced: false,
            next_id: 1,
        }
    }

    /// The speed bonus on scores: faster is worth more.
    pub fn speed_bonus(&self) -> f32 {
        (self.ship.speed / CRUISE).max(1.0)
    }

    /// How far from the axis the ship may fly at arc length `s`: inside the wall, which a
    /// singularity pinches in.
    pub fn ship_radius(&self, s: f32) -> f32 {
        SHIP_RADIUS.min(self.lattice.radius(s) - 1.3).max(1.0)
    }

    /// The nearest enemy more than `beyond` ahead of the ship, by arc length (what a bot
    /// aims at).
    pub fn nearest_ahead(&self, beyond: f32) -> Option<&Enemy> {
        let s = self.ship.s();
        self.enemies
            .iter()
            .filter(|e| arc(e.pos) > s + beyond)
            .min_by(|a, b| arc(a.pos).total_cmp(&arc(b.pos)))
    }

    /// One tick. The singularities' pull is theirs at the start of the tick, on everything.
    pub fn tick(&mut self, input: &Input) {
        self.events.clear();
        self.tick += 1;
        self.time += DT;
        let alive = self.phase == Phase::Playing;
        let wells = self.wells();
        if self.phase.count_down() {
            self.respawn();
        }
        if self.phase != Phase::Over {
            self.fly(input, alive, &wells);
            if alive {
                self.fire(input);
                if input.bomb {
                    self.bomb();
                }
            }
            self.direct();
            self.land();
        }
        self.step_enemies(&wells);
        self.step_gates();
        self.step_shots(&wells);
        self.step_bolts(&wells);
        self.step_shards();
        if self.phase == Phase::Playing {
            self.collide_ship();
        }
        self.step_lattice();
        let s = self.ship.s();
        self.track.extend(s + SPAWN_AHEAD + 40.0, s - 30.0);
        self.extras();
    }

    /// The ship's tick: throttle, movement across the tunnel and the singularities' pull,
    /// a barrel roll, and the wall (which a singularity pinches) holding it in.
    fn fly(&mut self, input: &Input, alive: bool, wells: &Wells) {
        // A singularity pulls the ship across the tunnel (not along it), capped so that it can
        // always be flown out of; and where it pinches the wall, the ship must stay further in.
        let pull = across(wells.pull(self.ship.pos, 30.0)).clamp_ideal_norm(7.0);
        let limit = self.ship_radius(self.ship.s() + self.ship.speed * DT);
        let ship = &mut self.ship;
        ship.prev = ship.pos;
        let target = CRUISE * (1.0 + 0.45 * input.throttle.clamp(-1.0, 1.0));
        ship.speed += (target - ship.speed) * (1.0 - (-3.0 * DT).exp());
        ship.cooldown -= DT;
        ship.invulnerable = (ship.invulnerable - DT).max(0.0);
        ship.roll_cool = (ship.roll_cool - DT).max(0.0);
        let across = if alive {
            input.movement * 12.0 + pull
        } else {
            P::zero()
        };
        let mut pos = ship.pos + (across + dir(0.0, 0.0, ship.speed)) * DT;
        // A barrel roll: a rotation about the tunnel's axis, spread over its duration.
        if alive && input.roll != 0 && ship.roll.is_none() && ship.roll_cool <= 0.0 {
            let d = f32::from(input.roll.signum());
            ship.roll = Some(Roll { t: 0.0, dir: d });
            ship.invulnerable = ship.invulnerable.max(ROLL_TIME + 0.1);
            self.events.push(Event::Roll { dir: d });
        }
        if let Some(r) = &mut ship.roll {
            // Eased: fast in the middle (half a turn of a phasor, from 0 to 1).
            let ease = |t: f32| {
                let half = core::f32::consts::FRAC_PI_2;
                0.5 + 0.5 * crate::signal::wave(core::f32::consts::PI * t / ROLL_TIME - half)
            };
            let step = ease((r.t + DT).min(ROLL_TIME)) - ease(r.t);
            pos = about_axis(r.dir * ROLL_ANGLE * step) >> pos;
            r.t += DT;
            if r.t >= ROLL_TIME {
                ship.roll = None;
                ship.roll_cool = 0.5;
            }
        }
        // Keep inside the tunnel: pulled back towards the foot on the axis.
        let r = off_axis(pos);
        if r > limit {
            let f = foot(pos);
            pos = f + (pos - f) * (limit / r);
        }
        ship.pos = pos;
    }

    /// Gates: flown through when the line of the ship's last step meets a gate's plane inside
    /// its ring.
    fn step_gates(&mut self) {
        let (a, b) = (self.ship.prev, self.ship.pos);
        let alive = self.phase == Phase::Playing;
        let bonus = self.speed_bonus();
        let mut passed = Vec::new();
        for g in &mut self.gates {
            if g.state != GateState::Open {
                g.t += DT;
                continue;
            }
            if arc(b) < arc(g.pos) {
                continue;
            }
            let meet = (a & b) ^ g.plane();
            let through = alive && meet.distance(g.pos) < GATE_RADIUS;
            g.state = if through {
                GateState::Passed
            } else {
                GateState::Missed
            };
            passed.push((g.pos, through));
        }
        for (pos, through) in passed {
            if through {
                self.chain += 1;
                self.ledger.raise();
                let points = self
                    .ledger
                    .award((250.0 * self.chain as f32 * bonus).round() as u64);
                self.events.push(Event::Gate {
                    pos,
                    chain: self.chain,
                    points,
                });
                self.blast(pos, 160.0, 4.0, 0.1);
            } else {
                if alive && self.chain > 0 {
                    self.events.push(Event::GateMiss { pos });
                }
                self.chain = 0;
            }
        }
        let s = self.ship.s();
        self.gates.retain(|g| arc(g.pos) > s - 20.0);
    }

    /// Shards: drawn to the ship within reach, collected for the multiplier.
    fn step_shards(&mut self) {
        let ship = self.ship.pos;
        let alive = self.phase == Phase::Playing;
        let mut got = 0;
        self.shards.retain_mut(|sh| {
            sh.life -= DT;
            let to = ship - sh.pos;
            let d = to.ideal_norm();
            // Magnetic within reach.
            if alive && d < 16.0 {
                sh.pos += to * (30.0 / d.max(0.5) * DT);
            }
            if alive && d < 1.4 {
                got += 1;
                return false;
            }
            sh.life > 0.0 && arc(sh.pos) > arc(ship) - 4.0
        });
        let pos = self.ship.pos;
        let pickups = (0..got).map(|_| Event::Pickup {
            pos,
            mult: self.ledger.raise(),
        });
        self.events.extend(pickups);
    }

    /// The wall's tick: it scrolls with the ship, and the blasts, the singularities' pinch and
    /// the ship's wake push and pull it.
    fn step_lattice(&mut self) {
        let s = self.ship.s();
        self.lattice.follow(s);
        let mut bulges: Vec<Bulge> = Vec::with_capacity(self.blasts.len() + 2);
        self.blasts.retain_mut(|(bulge, t)| {
            bulges.push(*bulge);
            *t -= DT;
            *t > 0.0
        });
        // Singularities pinch the wall in around them.
        for (pos, strength) in self.wells().iter() {
            bulges.push(Bulge {
                pos,
                push: -PINCH * strength,
                r2: 8.0,
                reach2: PINCH_REACH2,
            });
        }
        // The ship's wake: a gentle push on the wall nearest to it.
        if self.phase == Phase::Playing {
            bulges.push(Bulge {
                pos: self.ship.pos + dir(0.0, 0.0, 1.0),
                push: 6.0 * self.ship.speed / CRUISE,
                r2: 1.5,
                reach2: 0.0,
            });
        }
        self.lattice.step(DT, &bulges);
    }

    /// The extra lives and bombs the score has earned.
    fn extras(&mut self) {
        let events = &mut self.events;
        self.ledger
            .extras(|life| events.push(Event::Extra { life }));
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Whether two points (of weight 1) are within `1e-3`.
    pub(crate) fn close(a: P, b: P) -> bool {
        a.distance(b) < 1e-3
    }

    /// A bot's aim: the nearest enemy more than 5 ahead, or straight down the tunnel.
    pub(crate) fn aim_ahead(w: &World) -> P {
        w.nearest_ahead(5.0)
            .map_or(at(0.0, 0.0, w.ship.s() + AIM_DEPTH), |e| e.pos)
    }

    /// An enemy put down at `pos`, holding there (a serpent with its body): its index.
    pub(crate) fn spawn(w: &mut World, foe: Foe, pos: P) -> usize {
        let id = w.next_id();
        let (swim, body) = w.swim(foe, pos);
        w.enemies.push(Enemy {
            age: 1.0,
            orbit: off_axis(pos),
            depth: arc(pos) - w.ship.s(),
            timer: director::WELL_HOLD,
            flight: Flight::Hold,
            swim,
            body,
            ..Enemy::new(id, foe, pos)
        });
        w.enemies.len() - 1
    }

    #[test]
    fn a_barrel_roll_turns_the_ship_about_the_axis() {
        let mut w = World::new(1);
        w.ship.pos = at(4.0, 0.0, 0.0);
        w.ship.invulnerable = 0.0;
        w.tick(&Input {
            roll: 1,
            ..Input::default()
        });
        for _ in 0..60 {
            w.tick(&Input::default());
        }
        let p = w.ship.pos;
        assert!((off_axis(p) - 4.0).abs() < 1e-3);
        let expect = around(4.0, ROLL_ANGLE, arc(p));
        assert!(close(p, expect), "{:?}", p.to_euclidean());
    }

    /// A bot that aims at the nearest enemy ahead kills things, and the run stays sane.
    #[test]
    fn a_run_plays() {
        let mut w = World::new(3);
        w.ship.invulnerable = 1e9;
        let mut kills = 0;
        let mut warned = 0;
        for t in 0..120 * 60 {
            let a = t as f32 * 0.01;
            let movement = about_axis(a) >> dir(0.5, 0.0, 0.0);
            w.tick(&Input {
                movement,
                aim: aim_ahead(&w),
                fire: true,
                ..Input::default()
            });
            for e in &w.events {
                match e {
                    Event::Kill { .. } => kills += 1,
                    Event::Warn { .. } => warned += 1,
                    _ => {}
                }
            }
            assert!(off_axis(w.ship.pos) <= SHIP_RADIUS + 1e-3);
        }
        assert!(warned > 30, "{warned} warnings");
        assert!(kills > 20, "{kills} kills");
        assert!(w.ship.s() > 60.0 * CRUISE * 0.9);
        assert!(w.ledger.score > 0);
    }

    /// Extra lives come every 100 000 points and extra bombs every 150 000, as many as a
    /// score passes in one tick.
    #[test]
    fn extras_come_at_their_thresholds() {
        let mut w = World::new(1);
        w.spawn_cooldown = 1e9;
        w.gate_timer = 1e9;
        let (lives, bombs) = (w.ledger.lives, w.ledger.bombs);
        // The extra lives and bombs a tick at `score` brings.
        fn extras(w: &mut World, score: u64) -> (usize, usize) {
            w.ledger.score = score;
            w.tick(&Input::default());
            let n = |life: bool| {
                w.events
                    .iter()
                    .filter(|&&e| e == Event::Extra { life })
                    .count()
            };
            (n(true), n(false))
        }
        assert_eq!(extras(&mut w, 99_999), (0, 0));
        assert_eq!(extras(&mut w, 100_000), (1, 0));
        assert_eq!(extras(&mut w, 450_000), (3, 3));
        assert_eq!((w.ledger.lives, w.ledger.bombs), (lives + 4, bombs + 3));
    }

    /// Through a gate: the chain grows and scores; past one: the chain breaks.
    #[test]
    fn gates_are_flown_through_or_missed() {
        let mut w = World::new(4);
        w.spawn_cooldown = 1e9;
        w.gate_timer = 1e9;
        w.ship.invulnerable = 1e9;
        let s = w.ship.s();
        for (k, x) in [0.0f32, 0.0, 4.0].into_iter().enumerate() {
            w.gates.push(Gate {
                pos: at(x, -2.0, s + 10.0 + 10.0 * k as f32),
                state: GateState::Open,
                t: 0.0,
            });
        }
        let mut chains = Vec::new();
        for _ in 0..240 {
            w.tick(&Input::default());
            for e in &w.events {
                match *e {
                    Event::Gate { chain, points, .. } => chains.push((chain, points > 0)),
                    Event::GateMiss { .. } => chains.push((0, false)),
                    _ => {}
                }
            }
        }
        assert_eq!(chains, [(1, true), (2, true), (0, false)]);
        assert_eq!(w.chain, 0);
        assert_eq!(w.ledger.mult, 3);
    }
}
