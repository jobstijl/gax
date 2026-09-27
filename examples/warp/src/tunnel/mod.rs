//! Tunnel: flying into the screen down a twisting tunnel of warped space.
//!
//! A pure simulation like the Plane's (`sim`): a fixed 120 Hz step from a seed. Gameplay lives
//! in straightened coordinates, a PGA3D space where `x` and `y` run across the tunnel and `z`
//! is the arc length `s` along it; `track` maps it into the world through a chain of screw
//! motions, which only the drawing needs.
//!
//! Everything here is gax: positions are points, velocities directions, the tunnel's axis a
//! line. The distance from the axis is a join with it, the pull back inside moves towards a
//! point's foot on the axis (`(p | axis) ^ axis`), turns about the axis are motors (rings,
//! barrel rolls, wall placements), mines bounce by reflection in a plane, and collisions are
//! swept segments against spheres with joins.

pub mod lattice;
pub mod track;

use crate::sim::rng::Rng;
use crate::sim::{DT, Kind, Phase};
use gax::Unit;
use gax::pga3d::{Line, Motor, Plane, Point};
use lattice::{Lattice, RADIUS, Source};
use track::Track;

/// A point or a direction in straightened coordinates.
pub type P = Point<(), f32>;

/// The ship stays this close to the axis.
pub const SHIP_RADIUS: f32 = 5.3;
/// The cruising speed (arc length per second); throttle adds or takes up to 45%.
pub const CRUISE: f32 = 17.0;
/// How far ahead an unlocked reticle aims.
pub const AIM_DEPTH: f32 = 34.0;
/// Spawns land this far ahead (inside the fog).
const SPAWN_AHEAD: f32 = 80.0;
const SHOT_SPEED: f32 = 60.0;
const FIRE_INTERVAL: f32 = 1.0 / 14.0;
const BOLT_SPEED: f32 = 13.0;
const ROLL_TIME: f32 = 0.4;
const ROLL_ANGLE: f32 = 1.3;
/// The ship's hit sphere (shots are more generous: `HIT_MARGIN`).
const SHIP_SIZE: f32 = 0.38;
const HIT_MARGIN: f32 = 0.35;

/// The point `(x, y)` across the tunnel at arc length `s`.
pub fn at(x: f32, y: f32, s: f32) -> P {
    Point::xyz(x, y, s)
}

/// The direction `(x, y, s)`.
pub fn dir(x: f32, y: f32, s: f32) -> P {
    Point::direction(x, y, s)
}

/// The tunnel's axis.
pub fn axis() -> Line<(), f32> {
    at(0.0, 0.0, 0.0) & dir(0.0, 0.0, 1.0)
}

/// The arc length of a point.
pub fn arc(p: P) -> f32 {
    p.e021() / p.e123()
}

/// The foot of `p` on the axis: the meet of the axis with the plane through `p` orthogonal to
/// it (weight 1).
pub fn foot(p: P) -> P {
    let a = axis();
    let f = (p | a) ^ a;
    f.unitized()
}

/// The distance of `p` from the axis: the norm of their join (the axis is a unit line).
pub fn off_axis(p: P) -> f32 {
    (axis() & p).norm()
}

/// The rotation about the tunnel's axis by `angle`.
pub fn about_axis(angle: f32) -> Unit<Motor<(), f32>> {
    Motor::rotation(axis(), angle)
}

/// The point at distance `r` from the axis, at `angle` around it, at arc length `s`: `(r, 0, s)`
/// turned about the axis.
pub fn around(r: f32, angle: f32, s: f32) -> P {
    about_axis(angle) >> at(r, 0.0, s)
}

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
            movement: dir(0.0, 0.0, 0.0),
            aim: at(0.0, 0.0, AIM_DEPTH),
            fire: false,
            bomb: false,
            roll: 0,
            throttle: 0.0,
        }
    }
}

/// The tunnel's enemies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Foe {
    /// Drones in rings that hold ahead of you, circle, and dive one by one.
    Drone,
    /// Mines drifting in the lane.
    Mine,
    /// Turrets on the wall that fire bolts you must read in depth.
    Turret,
}

impl Foe {
    /// The Plane family whose colour and sounds it borrows.
    pub fn kind(self) -> Kind {
        match self {
            Foe::Drone => Kind::Drifter,
            Foe::Mine => Kind::Splitter,
            Foe::Turret => Kind::Warden,
        }
    }

    /// Its size (the radius of its hit sphere).
    pub fn radius(self) -> f32 {
        match self {
            Foe::Drone => 0.75,
            Foe::Mine => 0.9,
            Foe::Turret => 1.0,
        }
    }

    fn hp(self) -> f32 {
        match self {
            Foe::Drone => 1.0,
            Foe::Mine => 2.0,
            Foe::Turret => 4.0,
        }
    }

    fn points(self) -> u64 {
        match self {
            Foe::Drone => 100,
            Foe::Mine => 150,
            Foe::Turret => 400,
        }
    }
}

/// What a drone is doing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Flight {
    /// Coming in to its place in the ring ahead of the ship.
    Approach,
    /// Holding its place, circling with the ring.
    Hold,
    /// About to dive (a warning flash), then diving at the ship.
    Dive,
}

/// An enemy.
#[derive(Clone, Debug)]
pub struct Enemy {
    /// A stable id.
    pub id: u32,
    /// What it is.
    pub foe: Foe,
    /// Where.
    pub pos: P,
    /// Where last tick.
    pub prev: P,
    /// Velocity.
    pub vel: P,
    /// Hit points.
    pub hp: f32,
    /// Seconds alive.
    pub age: f32,
    /// A hit flash, decaying.
    pub flash: f32,
    /// The angle around the axis (drones: in the ring; turrets: on the wall).
    pub angle: f32,
    /// The ring's radius (drones).
    pub ring: f32,
    /// How far ahead of the ship the ring holds (drones).
    pub depth: f32,
    /// Seconds to the next action (turrets: firing, the last 0.35 s a charge-up; drones: the
    /// dive's warning).
    pub timer: f32,
    /// A drone's flight.
    pub flight: Flight,
}

/// A shot (the player's) or a bolt (an enemy's).
#[derive(Clone, Copy, Debug)]
pub struct Shot {
    /// Where.
    pub pos: P,
    /// Where last tick.
    pub prev: P,
    /// Velocity.
    pub vel: P,
    /// Seconds left.
    pub life: f32,
}

/// A shard: collect it for the multiplier.
#[derive(Clone, Copy, Debug)]
pub struct Shard {
    /// Where.
    pub pos: P,
    /// Seconds left.
    pub life: f32,
}

/// A barrel roll in progress.
#[derive(Clone, Copy, Debug)]
pub struct Roll {
    /// Seconds into it.
    pub t: f32,
    /// `-1` or `1`.
    pub dir: f32,
}

/// The ship.
#[derive(Clone, Copy, Debug)]
pub struct Ship {
    /// Where: across the tunnel, at its arc length.
    pub pos: P,
    /// Last tick's position.
    pub prev: P,
    /// Speed along the track.
    pub speed: f32,
    /// Seconds to the next shot.
    pub cooldown: f32,
    /// Seconds of invulnerability left.
    pub invulnerable: f32,
    /// A barrel roll.
    pub roll: Option<Roll>,
    /// Seconds before the next roll.
    pub roll_cool: f32,
    /// Which barrel fires next.
    pub barrel: bool,
}

impl Ship {
    /// The arc length it is at.
    pub fn s(&self) -> f32 {
        arc(self.pos)
    }
}

/// Something that happened this tick (positions in straightened coordinates).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Event {
    /// A shot left the gun.
    Fire { pos: P },
    /// A hit that did not kill.
    Hit { pos: P, foe: Foe },
    /// A kill.
    Kill { pos: P, foe: Foe, points: u64 },
    /// A turret fired.
    Bolt { pos: P },
    /// A drone began its dive.
    Dive { pos: P },
    /// A shot hit the wall.
    Wall { pos: P },
    /// A spawn was announced.
    Warn { pos: P, foe: Foe },
    /// A shard collected.
    Pickup { pos: P, mult: u32 },
    /// A barrel roll.
    Roll { dir: f32 },
    /// The ship was destroyed.
    Death { pos: P },
    /// The ship reappeared.
    Respawn,
    /// A bomb.
    Bomb { pos: P },
    /// An extra life or bomb.
    Extra { life: bool },
    /// The run ended.
    GameOver,
}

/// A spawn on its way in (announced by a warp-in).
#[derive(Clone, Copy, Debug)]
pub struct Pending {
    /// What.
    pub foe: Foe,
    /// Where.
    pub pos: P,
    /// Seconds to landing.
    pub t: f32,
    /// The angle around the axis.
    pub angle: f32,
    /// The ring's radius (drones).
    pub ring: f32,
    /// The ring's depth ahead of the ship (drones).
    pub depth: f32,
}

/// The tunnel's whole state.
pub struct World {
    /// The run's seed.
    pub seed: u64,
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
    /// Spawns on their way.
    pub pending: Vec<Pending>,
    /// Score.
    pub score: u64,
    /// Multiplier.
    pub mult: u32,
    /// Lives left, including the current one.
    pub lives: u32,
    /// Bombs left.
    pub bombs: u32,
    /// Where the run is.
    pub phase: Phase,
    /// This tick's events.
    pub events: Vec<Event>,
    /// Lattice sources that last a while (blasts), with their time left.
    blasts: Vec<(Source, f32)>,
    /// Intensity, `0..1`.
    pub intensity: f32,
    clock: f32,
    cooldown: f32,
    dive_cooldown: f32,
    next_id: u32,
    next_life: u64,
    next_bomb: u64,
}

/// Does the segment `a → b` pass within `r` of `c`? The distance to the segment's line is the
/// weight of the plane `a & b & c` over the weight of the line `a & b`; whether the nearest
/// point lies between the ends is the sign of `c` against the planes through the ends
/// orthogonal to the line (`l | a`, `l | b`).
pub fn segment_hits_sphere(a: P, b: P, c: P, r: f32) -> bool {
    let dist = |p: P, q: P| (p & q).norm();
    if dist(a, c) < r || dist(b, c) < r {
        return true;
    }
    let l: Line<(), f32> = a & b;
    let len = l.norm();
    if len < 1e-6 {
        return false;
    }
    let between = ((l | a) & c).s() * ((l | b) & c).s() < 0.0;
    between && (l & c).norm() / len < r
}

impl World {
    /// A new run.
    pub fn new(seed: u64) -> World {
        let mut track = Track::new(seed);
        track.extend(SPAWN_AHEAD + 60.0, 0.0);
        let start = at(0.0, -2.0, 0.0);
        World {
            seed,
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
            pending: Vec::new(),
            score: 0,
            mult: 1,
            lives: 3,
            bombs: 3,
            phase: Phase::Playing,
            events: Vec::new(),
            blasts: Vec::new(),
            intensity: 0.1,
            clock: 0.0,
            cooldown: 2.0,
            dive_cooldown: 2.0,
            next_id: 1,
            next_life: 100_000,
            next_bomb: 150_000,
        }
    }

    /// The speed bonus on scores: faster is worth more.
    pub fn speed_bonus(&self) -> f32 {
        (self.ship.speed / CRUISE).max(1.0)
    }

    /// One tick.
    pub fn tick(&mut self, input: &Input) {
        self.events.clear();
        self.tick += 1;
        self.time += DT;
        let alive = self.phase == Phase::Playing;
        if let Phase::Dead(t) = self.phase {
            if t - DT <= 0.0 {
                self.respawn();
            } else {
                self.phase = Phase::Dead(t - DT);
            }
        }
        if self.phase != Phase::Over {
            self.fly(input, alive);
            if alive {
                self.fire(input);
                if input.bomb {
                    self.bomb();
                }
            }
            self.direct();
            self.land();
        }
        self.step_enemies();
        self.step_shots();
        self.step_bolts();
        self.step_shards();
        if self.phase == Phase::Playing {
            self.collide_ship();
        }
        self.step_lattice();
        let s = self.ship.s();
        self.track.extend(s + SPAWN_AHEAD + 40.0, s - 30.0);
        self.extras();
    }

    fn fly(&mut self, input: &Input, alive: bool) {
        let ship = &mut self.ship;
        ship.prev = ship.pos;
        let target = CRUISE * (1.0 + 0.45 * input.throttle.clamp(-1.0, 1.0));
        ship.speed += (target - ship.speed) * (1.0 - (-3.0 * DT).exp());
        ship.cooldown -= DT;
        ship.invulnerable = (ship.invulnerable - DT).max(0.0);
        ship.roll_cool = (ship.roll_cool - DT).max(0.0);
        let across = if alive {
            input.movement * 12.0
        } else {
            dir(0.0, 0.0, 0.0)
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
        if r > SHIP_RADIUS {
            let f = foot(pos);
            pos = f + (pos - f) * (SHIP_RADIUS / r);
        }
        ship.pos = pos;
    }

    /// The enemy nearest to `p` within `reach`, if any: a locked target.
    fn nearest(&self, p: P, reach: f32) -> Option<usize> {
        let s = self.ship.s();
        self.enemies
            .iter()
            .enumerate()
            .filter(|(_, e)| arc(e.pos) > s + 2.0)
            .map(|(i, e)| (i, (e.pos & p).norm()))
            .filter(|&(_, d)| d < reach)
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(i, _)| i)
    }

    fn fire(&mut self, input: &Input) {
        if !input.fire || self.ship.cooldown > 0.0 {
            return;
        }
        self.ship.cooldown = FIRE_INTERVAL;
        self.ship.barrel = !self.ship.barrel;
        let side = if self.ship.barrel { 0.35 } else { -0.35 };
        let muzzle = self.ship.pos + dir(side, 0.0, 0.8);
        let forward = self.ship.speed + SHOT_SPEED;
        // At a locked enemy, lead it: where it will be when the shot gets there.
        let target = match self.nearest(input.aim, 2.5) {
            Some(i) => {
                let e = &self.enemies[i];
                let closing = (forward - e.vel.e021()).max(10.0);
                let t = (arc(e.pos) - arc(muzzle)) / closing;
                e.pos + e.vel * t
            }
            None => input.aim,
        };
        // Along the line to the target, at the shots' speed along the track.
        let to = target - muzzle;
        let ahead = to.e021().max(4.0);
        let vel = dir(to.e032(), to.e013(), ahead) * (forward / ahead);
        self.shots.push(Shot {
            pos: muzzle,
            prev: muzzle,
            vel,
            life: 1.6,
        });
        self.events.push(Event::Fire { pos: muzzle });
    }

    fn bomb(&mut self) {
        if self.bombs == 0 {
            return;
        }
        self.bombs -= 1;
        let s = self.ship.s();
        for i in (0..self.enemies.len()).rev() {
            if arc(self.enemies[i].pos) - s < 70.0 {
                self.kill(i, false);
            }
        }
        self.bolts.clear();
        self.pending.retain(|p| arc(p.pos) - s > 70.0);
        self.events.push(Event::Bomb { pos: self.ship.pos });
        for k in 0..6 {
            self.blast(at(0.0, 0.0, s + 4.0 + k as f32 * 9.0), 900.0, 16.0, 0.3);
        }
        self.cooldown = self.cooldown.max(2.0);
    }

    fn blast(&mut self, pos: P, strength: f32, r2: f32, time: f32) {
        self.blasts.push((Source { pos, strength, r2 }, time));
    }

    /// The director: an intensity curve over the run, and formations ahead in the fog.
    fn direct(&mut self) {
        self.clock += DT;
        self.intensity = crate::sim::director::Director::curve(self.clock * 1.3);
        self.cooldown -= DT;
        if self.cooldown > 0.0 || self.enemies.len() + self.pending.len() > 40 {
            return;
        }
        let i = self.intensity;
        let s = self.ship.s() + SPAWN_AHEAD;
        let roll = self.rng.unit();
        let r = &mut self.rng;
        let mut spawn: Vec<Pending> = Vec::new();
        let pend = |foe, pos, angle, ring, depth| Pending {
            foe,
            pos,
            t: 0.8,
            angle,
            ring,
            depth,
        };
        if self.clock > 25.0 && roll < 0.25 + 0.15 * i {
            // Turrets on the wall, spaced along the track and around it.
            let n = 1 + (i * 3.0) as usize;
            let a0 = r.angle();
            for k in 0..n {
                let a = a0 + k as f32 * 2.1;
                let p = around(RADIUS - 0.5, a, s + k as f32 * 7.0);
                spawn.push(pend(Foe::Turret, p, a, 0.0, 0.0));
            }
        } else if self.clock > 10.0 && roll < 0.55 {
            // Mines scattered in the lane.
            let n = 3 + (i * 6.0) as usize;
            for k in 0..n {
                let p = around(r.range(0.0, SHIP_RADIUS), r.angle(), s + k as f32 * 5.0);
                spawn.push(pend(Foe::Mine, p, 0.0, 0.0, 0.0));
            }
        } else {
            // A ring of drones, which will hold in the band ahead.
            let n = 6 + (i * 6.0) as usize;
            let a0 = r.angle();
            // Close and wide: the ring spreads across the screen, not at the vanishing point.
            let ring = r.range(4.0, 5.3);
            let depth = r.range(6.0, 10.0);
            for k in 0..n {
                let a = a0 + k as f32 * core::f32::consts::TAU / n as f32;
                // Nearer than the rest, and quick to come in: a ring is a close-range fight.
                let at = s - SPAWN_AHEAD * 0.45;
                spawn.push(pend(Foe::Drone, around(ring, a, at), a, ring, depth));
            }
        }
        for p in spawn {
            self.events.push(Event::Warn {
                pos: p.pos,
                foe: p.foe,
            });
            self.pending.push(p);
        }
        self.cooldown = 1.8 + 3.0 * (1.0 - i) + self.rng.range(0.0, 1.0);
    }

    fn land(&mut self) {
        let mut landed = Vec::new();
        self.pending.retain_mut(|p| {
            p.t -= DT;
            if p.t > 0.0 {
                return true;
            }
            landed.push(*p);
            false
        });
        for p in landed {
            let id = self.next_id;
            self.next_id += 1;
            let drift = if p.foe == Foe::Mine {
                about_axis(self.rng.angle()) >> dir(0.7, 0.0, 0.0)
            } else {
                dir(0.0, 0.0, 0.0)
            };
            self.enemies.push(Enemy {
                id,
                foe: p.foe,
                pos: p.pos,
                prev: p.pos,
                vel: drift,
                hp: p.foe.hp(),
                age: 0.0,
                flash: 0.0,
                angle: p.angle,
                ring: p.ring,
                depth: p.depth,
                timer: self.rng.range(0.6, 1.8),
                flight: Flight::Approach,
            });
        }
    }

    fn step_enemies(&mut self) {
        let ship = self.ship.pos;
        let (s_ship, speed) = (self.ship.s(), self.ship.speed);
        let alive = self.phase == Phase::Playing;
        let mut fired = Vec::new();
        // One drone dives at a time: the one that has held longest.
        self.dive_cooldown -= DT;
        if alive && self.dive_cooldown <= 0.0 {
            let holding = self
                .enemies
                .iter_mut()
                .filter(|e| e.foe == Foe::Drone && e.flight == Flight::Hold)
                .max_by(|a, b| a.age.total_cmp(&b.age));
            if let Some(e) = holding {
                e.flight = Flight::Dive;
                e.timer = 0.7;
                self.events.push(Event::Dive { pos: e.pos });
                self.dive_cooldown = 1.6 - 0.8 * self.intensity;
            }
        }
        for e in &mut self.enemies {
            e.prev = e.pos;
            e.age += DT;
            e.flash = (e.flash - DT * 6.0).max(0.0);
            let ahead = arc(e.pos) - s_ship;
            match e.foe {
                Foe::Drone => {
                    // Its place in the ring, which turns and holds `depth` ahead of the ship.
                    e.angle += DT * 0.9;
                    let place = around(e.ring, e.angle, s_ship + e.depth);
                    let with_ship = dir(0.0, 0.0, speed);
                    e.vel = match e.flight {
                        Flight::Approach | Flight::Hold => {
                            // Towards its place, moving along with the ship; the approach is
                            // capped so that a ring glides in.
                            let to = (place - e.pos) * 2.5;
                            let n = to.ideal_norm();
                            let to = if n > 26.0 { to * (26.0 / n) } else { to };
                            if e.flight == Flight::Approach && (place & e.pos).norm() < 1.5 {
                                e.flight = Flight::Hold;
                            }
                            to + with_ship
                        }
                        Flight::Dive => {
                            e.timer -= DT;
                            if e.timer > 0.0 {
                                // The warning: it flashes in place.
                                e.flash = e.flash.max(0.6);
                                (place - e.pos) * 2.5 + with_ship
                            } else {
                                // At the ship, at a speed you can dodge.
                                let to = ship - e.pos;
                                to * (10.0 / to.ideal_norm().max(1e-3)) + with_ship * 0.7
                            }
                        }
                    };
                }
                Foe::Mine => {
                    // Drift, bouncing off the tunnel's inner radius: a reflection in the plane
                    // tangent to it there.
                    let outward = off_axis(e.pos + e.vel * DT) > off_axis(e.pos);
                    if off_axis(e.pos) > SHIP_RADIUS + 0.3 && outward {
                        let tangent = Plane::orthogonal_to(e.pos - foot(e.pos));
                        e.vel = tangent.reflect(e.vel);
                    }
                }
                Foe::Turret => {
                    e.timer -= DT;
                    if e.timer <= 0.0 {
                        e.timer = 1.5;
                        if alive && (12.0..62.0).contains(&ahead) {
                            // Aim where the ship will be when the bolt arrives; you read it by
                            // its growth and its colour.
                            let t = ahead / (speed + BOLT_SPEED);
                            let there = ship + dir(0.0, 0.0, speed * t);
                            fired.push((e.pos, (there - e.pos) * (1.0 / t)));
                        }
                    }
                }
            }
            e.pos += e.vel * DT;
        }
        for (pos, vel) in fired {
            self.bolts.push(Shot {
                pos,
                prev: pos,
                vel,
                life: 8.0,
            });
            self.events.push(Event::Bolt { pos });
        }
        // Behind the ship: gone.
        self.enemies.retain(|e| arc(e.pos) > s_ship - 6.0);
    }

    fn step_shots(&mut self) {
        let mut hits: Vec<(usize, usize)> = Vec::new();
        let mut walls = Vec::new();
        for (k, b) in self.shots.iter_mut().enumerate() {
            b.prev = b.pos;
            b.pos += b.vel * DT;
            b.life -= DT;
            let reach = |e: &Enemy| e.foe.radius() + HIT_MARGIN;
            if let Some(i) = self
                .enemies
                .iter()
                .position(|e| segment_hits_sphere(b.prev, b.pos, e.pos, reach(e)))
            {
                hits.push((k, i));
                b.life = -1.0;
            } else if off_axis(b.pos) > RADIUS {
                walls.push(b.pos);
                b.life = -1.0;
            }
        }
        for pos in walls {
            self.events.push(Event::Wall { pos });
        }
        self.shots.retain(|b| b.life > 0.0);
        let mut dead = Vec::new();
        for (_, i) in hits {
            let e = &mut self.enemies[i];
            e.hp -= 1.0;
            e.flash = 1.0;
            if e.hp <= 0.0 {
                if !dead.contains(&i) {
                    dead.push(i);
                }
            } else {
                self.events.push(Event::Hit {
                    pos: e.pos,
                    foe: e.foe,
                });
            }
        }
        dead.sort_unstable();
        for i in dead.into_iter().rev() {
            self.kill(i, true);
        }
    }

    fn kill(&mut self, i: usize, scored: bool) {
        let e = self.enemies.swap_remove(i);
        let points = if scored {
            (e.foe.points() as f32 * self.mult as f32 * self.speed_bonus()).round() as u64
        } else {
            0
        };
        self.score += points;
        self.events.push(Event::Kill {
            pos: e.pos,
            foe: e.foe,
            points,
        });
        let size = if e.foe == Foe::Turret { 700.0 } else { 260.0 };
        self.blast(e.pos, size, 3.0, 0.12);
        if scored {
            self.shards.push(Shard {
                pos: e.pos,
                life: 6.0,
            });
        }
    }

    fn step_bolts(&mut self) {
        let s = self.ship.s();
        for b in &mut self.bolts {
            b.prev = b.pos;
            b.pos += b.vel * DT;
            b.life -= DT;
        }
        self.bolts.retain(|b| b.life > 0.0 && arc(b.pos) > s - 4.0);
    }

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
        for _ in 0..got {
            self.mult += 1;
            self.events.push(Event::Pickup {
                pos: self.ship.pos,
                mult: self.mult,
            });
        }
    }

    fn collide_ship(&mut self) {
        if self.ship.invulnerable > 0.0 {
            return;
        }
        let (p, prev) = (self.ship.pos, self.ship.prev);
        let hit_enemy = self
            .enemies
            .iter()
            .any(|e| segment_hits_sphere(prev, p, e.pos, e.foe.radius() + SHIP_SIZE));
        let hit_bolt = self
            .bolts
            .iter()
            .any(|b| segment_hits_sphere(b.prev, b.pos, p, SHIP_SIZE + 0.15));
        if hit_enemy || hit_bolt {
            self.die();
        }
    }

    fn die(&mut self) {
        let pos = self.ship.pos;
        self.events.push(Event::Death { pos });
        self.blast(pos, 2500.0, 10.0, 0.3);
        self.mult = 1;
        self.lives = self.lives.saturating_sub(1);
        self.ship.roll = None;
        self.shots.clear();
        if self.lives == 0 {
            self.phase = Phase::Over;
            self.events.push(Event::GameOver);
        } else {
            self.phase = Phase::Dead(2.0);
        }
    }

    fn respawn(&mut self) {
        self.phase = Phase::Playing;
        let s = self.ship.s();
        self.ship.pos = at(0.0, -2.0, s);
        self.ship.prev = self.ship.pos;
        self.ship.invulnerable = 2.5;
        self.bolts.clear();
        self.enemies.retain(|e| arc(e.pos) - s > 30.0);
        self.events.push(Event::Respawn);
    }

    fn step_lattice(&mut self) {
        let s = self.ship.s();
        self.lattice.follow(s);
        let mut sources: Vec<Source> = Vec::with_capacity(self.blasts.len() + 1);
        self.blasts.retain_mut(|(src, t)| {
            sources.push(*src);
            *t -= DT;
            *t > 0.0
        });
        // The ship's wake: a gentle push on the wall nearest to it.
        if self.phase == Phase::Playing {
            sources.push(Source {
                pos: self.ship.pos + dir(0.0, 0.0, 1.0),
                strength: 6.0 * self.ship.speed / CRUISE,
                r2: 1.5,
            });
        }
        self.lattice.step(DT, &sources);
    }

    fn extras(&mut self) {
        if self.score >= self.next_life {
            self.next_life += 100_000;
            self.lives += 1;
            self.events.push(Event::Extra { life: true });
        }
        if self.score >= self.next_bomb {
            self.next_bomb += 150_000;
            self.bombs += 1;
            self.events.push(Event::Extra { life: false });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: P, b: P) -> bool {
        (a & b).norm() < 1e-3
    }

    #[test]
    fn segments_hit_spheres_by_the_join() {
        let a = at(0.0, 0.0, 0.0);
        let b = at(0.0, 0.0, 10.0);
        assert!(segment_hits_sphere(a, b, at(0.5, 0.0, 5.0), 0.6));
        assert!(!segment_hits_sphere(a, b, at(0.7, 0.0, 5.0), 0.6));
        // Beyond the ends: only the endpoints count.
        assert!(!segment_hits_sphere(a, b, at(0.3, 0.0, 11.0), 0.6));
        assert!(segment_hits_sphere(a, b, at(0.3, 0.0, 10.4), 0.6));
        // Diagonal.
        let c = at(3.0, 4.0, 12.0);
        assert!(segment_hits_sphere(a, c, at(1.5, 2.0, 6.0), 0.1));
    }

    #[test]
    fn the_axis_the_foot_and_turns_about_it() {
        let p = at(3.0, 4.0, 7.0);
        assert!((off_axis(p) - 5.0).abs() < 1e-5);
        assert!(close(foot(p), at(0.0, 0.0, 7.0)));
        let q = around(2.0, core::f32::consts::FRAC_PI_2, 3.0);
        assert!(close(q, at(0.0, 2.0, 3.0)), "{:?}", q.to_euclidean());
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

    /// Unlocked, shots fly through the aim point.
    #[test]
    fn shots_fly_to_the_aim_point() {
        let mut w = World::new(1);
        w.ship.pos = at(1.0, -1.0, 0.0);
        let aim = at(-3.0, 2.0, 30.0);
        w.tick(&Input {
            fire: true,
            aim,
            ..Input::default()
        });
        let b = w.shots[0];
        let t = (arc(aim) - arc(b.prev)) / b.vel.e021();
        assert!(close(b.prev + b.vel * t, aim));
    }

    /// Locked, shots lead a moving enemy and hit it.
    #[test]
    fn locked_shots_lead_a_moving_enemy() {
        let mut w = World::new(1);
        w.ship.pos = at(0.0, 0.0, 0.0);
        w.enemies.push(Enemy {
            id: 99,
            foe: Foe::Mine,
            pos: at(2.0, 1.0, 25.0),
            prev: at(2.0, 1.0, 25.0),
            vel: dir(3.0, 0.0, 0.0),
            hp: 5.0,
            age: 0.0,
            flash: 0.0,
            angle: 0.0,
            ring: 0.0,
            depth: 0.0,
            timer: 9.0,
            flight: Flight::Hold,
        });
        let mut hit = false;
        for _ in 0..60 {
            let aim = w.enemies.first().map_or(at(0.0, 0.0, 30.0), |e| e.pos);
            w.tick(&Input {
                fire: true,
                aim,
                ..Input::default()
            });
            hit |= w
                .events
                .iter()
                .any(|e| matches!(e, Event::Hit { .. } | Event::Kill { .. }));
        }
        assert!(hit, "never hit the moving mine");
    }

    /// Drones come in, hold a ring ahead of the ship, and dive one at a time.
    #[test]
    fn drones_hold_ahead_and_dive_one_by_one() {
        let mut w = World::new(5);
        w.ship.invulnerable = 1e9;
        let mut held = false;
        let mut dives = Vec::new();
        for t in 0..120 * 20 {
            w.tick(&Input::default());
            let s = w.ship.s();
            held |= w
                .enemies
                .iter()
                .any(|e| e.flight == Flight::Hold && (4.0..14.0).contains(&(arc(e.pos) - s)));
            for e in &w.events {
                if let Event::Dive { .. } = e {
                    dives.push(t);
                }
            }
        }
        assert!(held, "no drone held in the band");
        assert!(dives.len() > 3, "{} dives", dives.len());
        assert!(
            dives.windows(2).all(|w| w[1] - w[0] > 60),
            "dives not one by one"
        );
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
            let s = w.ship.s();
            let aim = w
                .enemies
                .iter()
                .filter(|e| arc(e.pos) > s + 5.0)
                .min_by(|a, b| arc(a.pos).total_cmp(&arc(b.pos)))
                .map_or(at(0.0, 0.0, s + AIM_DEPTH), |e| e.pos);
            let movement = about_axis(a) >> dir(0.5, 0.0, 0.0);
            w.tick(&Input {
                movement,
                aim,
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
        assert!(w.score > 0);
    }
}
