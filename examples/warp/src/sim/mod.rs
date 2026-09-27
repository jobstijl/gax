//! The simulation: a pure, headless core with no Bevy and no wgpu types. It advances at a
//! fixed 120 Hz from a seed and the players' inputs, so a run is reproducible, and reports what
//! happened as [`Event`]s for the renderer and the audio.
//!
//! All geometry is gax PGA2D: poses are unit motors, velocities twists, positions points,
//! and collisions joins and signed distances.

pub mod body;
pub mod collide;
pub mod director;
pub mod rng;

use body::{Body, angle_of, distance, length, pose_at};
use collide::{SpatialHash, segment_hits_circle};
use director::Director;
use gax::pga2d::Point;
use rng::Rng;

/// The fixed timestep.
pub const DT: f32 = 1.0 / 120.0;
/// Half the arena's width and height.
pub const ARENA: [f32; 2] = [32.0, 18.0];
/// No spawn lands closer than this to the player.
pub const SAFE_RADIUS: f32 = 7.0;

const SHIP_SPEED: f32 = 11.5;
const SHIP_RADIUS: f32 = 0.42;
const BULLET_SPEED: f32 = 40.0;
const FIRE_INTERVAL: f32 = 1.0 / 22.0;
const BULLET_LIFE: f32 = 1.4;

/// One tick's input, in gax types (see `input.rs`, the only boundary with Bevy's vectors).
#[derive(Clone, Copy, Debug)]
pub struct Input {
    /// Movement: a direction of length at most 1.
    pub movement: Point<(), f32>,
    /// Aim: a direction (zero for none).
    pub aim: Point<(), f32>,
    /// Fire held.
    pub fire: bool,
    /// Bomb pressed (edge).
    pub bomb: bool,
}

impl Default for Input {
    fn default() -> Input {
        Input {
            movement: Point::direction(0.0, 0.0),
            aim: Point::direction(0.0, 0.0),
            fire: false,
            bomb: false,
        }
    }
}

/// Enemy families.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Kind {
    /// Wanders; harmless alone. Teaches shooting.
    Drifter,
    /// Homes in with a limited turn rate.
    Chaser,
    /// A gravity well: bends bullets, grid and particles, eats enemies and grows, bursts when
    /// overfed.
    Singularity,
    /// A small, fast chaser from a burst singularity.
    Mote,
}

impl Kind {
    /// Base score.
    pub fn value(self) -> u64 {
        match self {
            Kind::Drifter => 50,
            Kind::Chaser => 100,
            Kind::Singularity => 500,
            Kind::Mote => 25,
        }
    }

    /// Collision radius (singularities grow).
    pub fn radius(self) -> f32 {
        match self {
            Kind::Drifter => 0.62,
            Kind::Chaser => 0.58,
            Kind::Singularity => 0.9,
            Kind::Mote => 0.32,
        }
    }
}

/// An enemy.
#[derive(Clone, Debug)]
pub struct Enemy {
    /// Stable id.
    pub id: u32,
    /// Family.
    pub kind: Kind,
    /// Motion.
    pub body: Body,
    /// Hits left.
    pub hp: f32,
    /// Collision radius.
    pub radius: f32,
    /// Mass eaten (singularities).
    pub mass: f32,
    /// Seconds alive.
    pub age: f32,
    /// Flash after a hit, decaying.
    pub flash: f32,
    /// A per-enemy phase for wandering and animation.
    pub phase: f32,
    /// Velocity from the wells' pull, on top of the enemy's own intent; decays slowly.
    pub pull: Point<(), f32>,
}

/// A shot.
#[derive(Clone, Copy, Debug)]
pub struct Bullet {
    /// Position (normalized point).
    pub pos: Point<(), f32>,
    /// Position at the start of the tick.
    pub prev: Point<(), f32>,
    /// Velocity (a direction).
    pub vel: Point<(), f32>,
    /// Seconds left.
    pub life: f32,
}

/// A shard dropped by a kill; collecting it raises the multiplier.
#[derive(Clone, Copy, Debug)]
pub struct Shard {
    /// Motion.
    pub body: Body,
    /// Seconds left.
    pub life: f32,
}

/// A telegraphed spawn.
#[derive(Clone, Copy, Debug)]
pub struct Pending {
    /// What comes.
    pub kind: Kind,
    /// Where.
    pub pos: [f32; 2],
    /// Seconds until it lands.
    pub t: f32,
    /// Its warning time, for drawing the warp-in.
    pub total: f32,
}

/// The ship.
#[derive(Clone, Debug)]
pub struct Ship {
    /// Motion.
    pub body: Body,
    /// Seconds until the next shot.
    pub cooldown: f32,
    /// Seconds of invulnerability left.
    pub invulnerable: f32,
    /// Alive (between death and respawn it is not).
    pub alive: bool,
    /// Alternates the two gun barrels.
    pub barrel: bool,
}

/// What happened in a tick, for the renderer (particles, grid impulses, shake) and the audio.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Event {
    /// A shot left the gun.
    Fire { pos: [f32; 2], angle: f32 },
    /// A bullet hit an enemy that survived.
    Hit { pos: [f32; 2], kind: Kind },
    /// An enemy died.
    Kill {
        pos: [f32; 2],
        kind: Kind,
        size: f32,
        scored: bool,
        /// Points gained (with the multiplier).
        points: u64,
    },
    /// A bullet hit a wall.
    Wall { pos: [f32; 2] },
    /// A singularity ate an enemy.
    Absorb { pos: [f32; 2], mass: f32 },
    /// An overfed singularity burst.
    Burst { pos: [f32; 2] },
    /// A spawn was announced.
    Warn { pos: [f32; 2], kind: Kind },
    /// A spawn landed.
    Spawn { pos: [f32; 2], kind: Kind },
    /// A shard was collected.
    Pickup { pos: [f32; 2], mult: u32 },
    /// The ship was destroyed.
    Death { pos: [f32; 2] },
    /// The ship (re)appeared.
    Respawn,
    /// A bomb went off.
    Bomb { pos: [f32; 2] },
    /// An extra life or bomb was earned.
    Extra { life: bool },
    /// The run ended.
    GameOver,
}

/// Where the run is.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Phase {
    /// Playing.
    Playing,
    /// The ship is destroyed; it respawns after the timer.
    Dead(f32),
    /// No lives left.
    Over,
}

/// The whole simulation state.
pub struct World {
    /// The seed of the run.
    pub seed: u64,
    /// The generator.
    pub rng: Rng,
    /// Ticks since the start.
    pub tick: u64,
    /// Seconds since the start.
    pub time: f32,
    /// The ship.
    pub ship: Ship,
    /// Shots.
    pub bullets: Vec<Bullet>,
    /// Enemies.
    pub enemies: Vec<Enemy>,
    /// Shards.
    pub shards: Vec<Shard>,
    /// Telegraphed spawns.
    pub pending: Vec<Pending>,
    /// Score.
    pub score: u64,
    /// Multiplier.
    pub mult: u32,
    /// Lives left (including the current one).
    pub lives: u32,
    /// Bombs left.
    pub bombs: u32,
    /// Where the run is.
    pub phase: Phase,
    /// The spawn director.
    pub director: Director,
    /// This tick's events.
    pub events: Vec<Event>,
    /// Seconds of hit-stop requested; the app counts it down in real time and skips ticks
    /// meanwhile (it is not simulation state, so replays are unaffected).
    pub hitstop: f32,
    /// The worst `|m ~m - 1|` seen on any pose this run.
    pub worst_drift: f32,
    next_id: u32,
    next_life: u64,
    next_bomb: u64,
    hash: SpatialHash,
    scratch: Vec<u32>,
}

impl World {
    /// A new run.
    pub fn new(seed: u64) -> World {
        World {
            seed,
            rng: Rng::new(seed),
            tick: 0,
            time: 0.0,
            ship: Ship {
                body: Body::new(pose_at(0.0, 0.0, core::f32::consts::FRAC_PI_2)),
                cooldown: 0.0,
                invulnerable: 2.0,
                alive: true,
                barrel: false,
            },
            bullets: Vec::new(),
            enemies: Vec::new(),
            shards: Vec::new(),
            pending: Vec::new(),
            score: 0,
            mult: 1,
            lives: 3,
            bombs: 3,
            phase: Phase::Playing,
            director: Director::new(),
            events: Vec::new(),
            hitstop: 0.0,
            worst_drift: 0.0,
            next_id: 1,
            next_life: 100_000,
            next_bomb: 150_000,
            hash: SpatialHash::new(ARENA[0], ARENA[1], 2.0),
            scratch: Vec::new(),
        }
    }

    /// The gravity wells (singularities): `(position, strength)`.
    pub fn wells(&self) -> impl Iterator<Item = ([f32; 2], f32)> + '_ {
        self.enemies
            .iter()
            .filter(|e| e.kind == Kind::Singularity && e.age > 0.0)
            .map(|e| (e.body.xy(), 18.0 + 6.0 * e.mass))
    }

    /// The acceleration the wells give a body at `p`.
    fn gravity(&self, p: [f32; 2], scale: f32) -> Point<(), f32> {
        let mut acc = Point::direction(0.0, 0.0);
        for (w, s) in self.wells() {
            let (dx, dy) = (w[0] - p[0], w[1] - p[1]);
            let d2 = dx * dx + dy * dy;
            // s d / (|d|² + r²)^(3/2): an inverse square with a softened core.
            let k = scale * s / (d2 + 1.5).powf(1.5);
            acc += Point::direction(dx * k, dy * k);
        }
        acc
    }

    /// Spawn an enemy now.
    pub fn spawn(&mut self, kind: Kind, [x, y]: [f32; 2]) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        let angle = self.rng.angle();
        let mut body = Body::new(pose_at(x, y, angle));
        let speed = match kind {
            Kind::Drifter => 2.2,
            Kind::Mote => 6.0,
            Kind::Singularity => 0.8,
            Kind::Chaser => 0.0,
        };
        let dir = self.rng.angle();
        body.vel = Point::direction(dir.cos() * speed, dir.sin() * speed);
        body.spin = match kind {
            Kind::Drifter => self.rng.range(-2.0, 2.0),
            Kind::Singularity => 0.6,
            _ => 0.0,
        };
        self.enemies.push(Enemy {
            id,
            kind,
            body,
            hp: match kind {
                Kind::Singularity => 25.0,
                _ => 1.0,
            },
            radius: kind.radius(),
            mass: 0.0,
            age: 0.0,
            flash: 0.0,
            phase: self.rng.range(0.0, 100.0),
            pull: Point::direction(0.0, 0.0),
        });
        id
    }

    /// Announce a spawn: it lands after `t` seconds, never within the safe radius.
    pub fn announce(&mut self, kind: Kind, pos: [f32; 2], t: f32) {
        let pos = self.safe_spot(pos);
        self.pending.push(Pending {
            kind,
            pos,
            t,
            total: t,
        });
        self.events.push(Event::Warn { pos, kind });
    }

    /// `pos`, clamped into the arena and pushed out of the safe radius around the ship.
    pub fn safe_spot(&self, pos: [f32; 2]) -> [f32; 2] {
        let clamp = |p: [f32; 2]| {
            [
                p[0].clamp(-ARENA[0] + 1.5, ARENA[0] - 1.5),
                p[1].clamp(-ARENA[1] + 1.5, ARENA[1] - 1.5),
            ]
        };
        let mut p = clamp(pos);
        let s = self.ship.body.xy();
        let (dx, dy) = (p[0] - s[0], p[1] - s[1]);
        let d = (dx * dx + dy * dy).sqrt();
        let need = SAFE_RADIUS + 1.0;
        if d < need {
            // Push out along the ray from the ship; if that leaves the arena, mirror through it.
            let (ux, uy) = if d > 1e-3 {
                (dx / d, dy / d)
            } else {
                (1.0, 0.0)
            };
            let out = clamp([s[0] + ux * need, s[1] + uy * need]);
            let back = clamp([s[0] - ux * need, s[1] - uy * need]);
            let far = |q: [f32; 2]| ((q[0] - s[0]).powi(2) + (q[1] - s[1]).powi(2)).sqrt();
            p = if far(out) >= far(back) { out } else { back };
        }
        p
    }

    /// Advance one tick.
    pub fn tick(&mut self, input: &Input) {
        self.events.clear();
        self.tick += 1;
        self.time += DT;
        match self.phase {
            Phase::Over => {
                self.step_enemies();
                return;
            }
            Phase::Dead(t) => {
                let t = t - DT;
                if t <= 0.0 {
                    self.respawn();
                } else {
                    self.phase = Phase::Dead(t);
                }
            }
            Phase::Playing => {}
        }
        if self.phase == Phase::Playing {
            self.steer_ship(input);
            if input.bomb {
                self.bomb();
            }
        }
        self.step_bullets();
        self.step_enemies();
        self.step_shards();
        self.step_pending();
        let mut d = std::mem::take(&mut self.director);
        d.update(self);
        self.director = d;
        if self.phase == Phase::Playing {
            self.collide_ship();
        }
        self.extras();
        let drift = body::drift(self.ship.body.pose);
        self.worst_drift = self.worst_drift.max(drift);
    }

    fn steer_ship(&mut self, input: &Input) {
        let ship = &mut self.ship;
        ship.invulnerable = (ship.invulnerable - DT).max(0.0);
        let want = input.movement.gp(SHIP_SPEED);
        // Quick, smooth response: approach the wanted velocity exponentially.
        let k = 1.0 - (-14.0 * DT).exp();
        ship.body.vel = ship.body.vel + (want - ship.body.vel).gp(k);
        // Turn the nose towards the motion.
        let speed = length(ship.body.vel);
        ship.body.spin = if speed > 0.5 {
            let diff = wrap_angle(angle_of(ship.body.vel) - angle_of(ship.body.heading()));
            (diff * 16.0).clamp(-24.0, 24.0)
        } else {
            0.0
        };
        ship.body.step(DT);
        // Walls: stay inside, lose the velocity into the wall.
        let [x, y] = ship.body.xy();
        let (hw, hh) = (ARENA[0] - SHIP_RADIUS, ARENA[1] - SHIP_RADIUS);
        let (cx, cy) = (x.clamp(-hw, hw), y.clamp(-hh, hh));
        if cx != x || cy != y {
            ship.body.shift(cx - x, cy - y);
            let (mut vx, mut vy) = (ship.body.vel.e20(), ship.body.vel.e01());
            if cx != x {
                vx = 0.0;
            }
            if cy != y {
                vy = 0.0;
            }
            ship.body.vel = Point::direction(vx, vy);
        }
        // Fire: two alternating barrels, along the aim.
        ship.cooldown -= DT;
        let aim_len = length(input.aim);
        if input.fire && aim_len > 0.2 && ship.cooldown <= 0.0 {
            ship.cooldown += FIRE_INTERVAL;
            let (ux, uy) = (input.aim.e20() / aim_len, input.aim.e01() / aim_len);
            let [x, y] = ship.body.xy();
            // One shot per interval, from alternating barrels.
            let side = if ship.barrel { 1.0f32 } else { -1.0 };
            let off = 0.14 * side;
            let pos = Point::xy(x - uy * off + ux * 0.4, y + ux * off + uy * 0.4);
            self.bullets.push(Bullet {
                pos,
                prev: pos,
                vel: Point::direction(ux * BULLET_SPEED, uy * BULLET_SPEED),
                life: BULLET_LIFE,
            });
            ship.barrel = !ship.barrel;
            self.events.push(Event::Fire {
                pos: [x, y],
                angle: uy.atan2(ux),
            });
        }
        ship.cooldown = ship.cooldown.max(-FIRE_INTERVAL);
    }

    fn step_bullets(&mut self) {
        // Enemies into the hash.
        self.hash.clear();
        for (i, e) in self.enemies.iter().enumerate() {
            if e.age >= 0.0 {
                self.hash.insert(i as u32, e.body.xy(), e.radius);
            }
        }
        let mut bullets = std::mem::take(&mut self.bullets);
        let mut kills: Vec<usize> = Vec::new();
        bullets.retain_mut(|b| {
            b.life -= DT;
            if b.life <= 0.0 {
                return false;
            }
            // Gravity bends the shot.
            let g = self.gravity(b.pos.to_euclidean(), 40.0);
            b.vel += g.gp(DT);
            b.prev = b.pos;
            b.pos += b.vel.gp(DT);
            let (a, p) = (b.prev.to_euclidean(), b.pos.to_euclidean());
            if p[0].abs() > ARENA[0] || p[1].abs() > ARENA[1] {
                self.events.push(Event::Wall {
                    pos: [
                        p[0].clamp(-ARENA[0], ARENA[0]),
                        p[1].clamp(-ARENA[1], ARENA[1]),
                    ],
                });
                return false;
            }
            self.hash.query(a, p, 1.5, &mut self.scratch);
            for &i in &self.scratch {
                let e = &mut self.enemies[i as usize];
                if e.hp <= 0.0 {
                    continue;
                }
                if segment_hits_circle(b.prev, b.pos, e.body.pos(), e.radius + 0.08) {
                    e.hp -= 1.0;
                    e.flash = 1.0;
                    if e.kind == Kind::Singularity {
                        // A hit pushes it back a little.
                        e.body.vel += b.vel.gp(0.004);
                    }
                    if e.hp <= 0.0 {
                        kills.push(i as usize);
                    } else {
                        self.events.push(Event::Hit {
                            pos: e.body.xy(),
                            kind: e.kind,
                        });
                    }
                    return false;
                }
            }
            true
        });
        self.bullets = bullets;
        kills.sort_unstable();
        kills.dedup();
        for &i in kills.iter().rev() {
            self.kill(i, true);
        }
    }

    /// Remove enemy `i`, with score and shards when `scored`.
    fn kill(&mut self, i: usize, scored: bool) {
        let e = self.enemies.swap_remove(i);
        let pos = e.body.xy();
        let size = match e.kind {
            Kind::Singularity => 2.0 + 0.3 * e.mass,
            Kind::Mote => 0.5,
            _ => 1.0,
        };
        let mut points = 0;
        if scored {
            let value = e.kind.value()
                + if e.kind == Kind::Singularity {
                    100 * e.mass as u64
                } else {
                    0
                };
            points = value * u64::from(self.mult);
            self.score += points;
            let shards = match e.kind {
                Kind::Singularity => 8 + e.mass as usize,
                Kind::Mote => usize::from(self.rng.chance(0.3)),
                _ => 1 + usize::from(self.rng.chance(0.35)),
            };
            for _ in 0..shards {
                let a = self.rng.angle();
                let s = self.rng.range(2.0, 7.0);
                let mut body = Body::new(pose_at(pos[0], pos[1], a));
                body.vel = Point::direction(a.cos() * s, a.sin() * s);
                body.spin = self.rng.range(-5.0, 5.0);
                self.shards.push(Shard { body, life: 7.0 });
            }
            if e.kind == Kind::Singularity {
                self.hitstop = self.hitstop.max(0.07);
            }
        }
        self.events.push(Event::Kill {
            pos,
            kind: e.kind,
            size,
            scored,
            points,
        });
    }

    fn step_enemies(&mut self) {
        let ship = self.ship.body.pos();
        let ship_xy = self.ship.body.xy();
        let hunting = self.phase == Phase::Playing;
        let wells: Vec<([f32; 2], f32, u32, f32)> = self
            .enemies
            .iter()
            .filter(|e| e.kind == Kind::Singularity)
            .map(|e| (e.body.xy(), 18.0 + 6.0 * e.mass, e.id, e.radius))
            .collect();
        let time = self.time;
        for e in &mut self.enemies {
            e.age += DT;
            e.flash = (e.flash - DT * 6.0).max(0.0);
            let p = e.body.xy();
            // The enemy's own velocity, without the wells' pull.
            e.body.vel -= e.pull;
            match e.kind {
                Kind::Drifter => {
                    // Wander: the velocity's direction turns slowly with a smooth noise.
                    let turn =
                        ((time * 0.7 + e.phase).sin() + (time * 1.3 + e.phase * 1.7).sin()) * 0.9;
                    let (vx, vy) = (e.body.vel.e20(), e.body.vel.e01());
                    let (c, s) = (turn * DT).sin_cos();
                    let v = Point::direction(vx * s - vy * c, vx * c + vy * s);
                    let len = length(v).max(1e-3);
                    e.body.vel = v.gp(2.2 / len);
                }
                Kind::Chaser | Kind::Mote => {
                    let (speed, turn_rate) = if e.kind == Kind::Chaser {
                        (6.2, 2.6)
                    } else {
                        (8.5, 4.0)
                    };
                    let target = if hunting {
                        ship
                    } else {
                        Point::xy(e.phase.sin() * 20.0, e.phase.cos() * 12.0)
                    };
                    let to = target - e.body.pos();
                    let want = angle_of(to);
                    let have = if length(e.body.vel) > 0.1 {
                        angle_of(e.body.vel)
                    } else {
                        want
                    };
                    let d = wrap_angle(want - have).clamp(-turn_rate * DT, turn_rate * DT);
                    let a = have + d;
                    let ramp = (e.age * 1.5).min(1.0);
                    e.body.vel = Point::direction(a.cos() * speed * ramp, a.sin() * speed * ramp);
                    e.body.spin = if e.kind == Kind::Chaser { 3.0 } else { 9.0 };
                }
                Kind::Singularity => {
                    // Slow drift towards the ship, damped.
                    let to = ship - e.body.pos();
                    let d = length(to).max(1e-3);
                    e.body.vel = (e.body.vel + to.gp(0.25 * DT / d)).gp(1.0 - 0.3 * DT);
                    e.radius = 0.9 + 0.12 * e.mass.sqrt() * 3.0;
                    e.hp = e.hp.min(25.0 + 4.0 * e.mass);
                }
            }
            // Wells pull everything but wells: the pull accumulates, and decays slowly.
            e.pull = e.pull.gp(1.0 - 0.6 * DT);
            if e.kind != Kind::Singularity {
                for &(w, s, _, _) in &wells {
                    let (dx, dy) = (w[0] - p[0], w[1] - p[1]);
                    let k = 2.5 * s / (dx * dx + dy * dy + 1.5).powf(1.5);
                    e.pull += Point::direction(dx * k * DT, dy * k * DT);
                }
            }
            e.body.vel += e.pull;
            e.body.step(DT);
            // Walls: bounce.
            let [x, y] = e.body.xy();
            let (hw, hh) = (ARENA[0] - e.radius, ARENA[1] - e.radius);
            if x.abs() > hw || y.abs() > hh {
                let (cx, cy) = (x.clamp(-hw, hw), y.clamp(-hh, hh));
                e.body.shift(cx - x, cy - y);
                let flip = |v: Point<(), f32>| {
                    let (mut vx, mut vy) = (v.e20(), v.e01());
                    if cx != x {
                        vx = -vx;
                    }
                    if cy != y {
                        vy = -vy;
                    }
                    Point::direction(vx, vy)
                };
                e.body.vel = flip(e.body.vel);
                e.pull = flip(e.pull);
            }
            let _ = ship_xy;
        }
        // Singularities eat what falls in.
        let mut eaten: Vec<(usize, usize)> = Vec::new();
        for (wi, w) in self.enemies.iter().enumerate() {
            if w.kind != Kind::Singularity {
                continue;
            }
            for (i, e) in self.enemies.iter().enumerate() {
                if e.kind != Kind::Singularity
                    && e.age > 0.3
                    && distance(e.body.pos(), w.body.pos()) < w.radius * 0.8
                {
                    eaten.push((wi, i));
                }
            }
        }
        let mut gone: Vec<usize> = Vec::new();
        for (wi, i) in eaten {
            if gone.contains(&i) {
                continue;
            }
            gone.push(i);
            let w = &mut self.enemies[wi];
            w.mass += 1.0;
            w.hp += 4.0;
            let (pos, mass) = (w.body.xy(), w.mass);
            self.events.push(Event::Absorb { pos, mass });
        }
        gone.sort_unstable();
        for &i in gone.iter().rev() {
            self.enemies.swap_remove(i);
        }
        // Overfed singularities burst into motes.
        let mut bursts = Vec::new();
        self.enemies.retain(|e| {
            if e.kind == Kind::Singularity && e.mass >= 10.0 {
                bursts.push(e.body.xy());
                false
            } else {
                true
            }
        });
        for p in bursts {
            self.events.push(Event::Burst { pos: p });
            for _ in 0..14 {
                let a = self.rng.angle();
                let id = self.spawn(Kind::Mote, [p[0] + a.cos() * 1.2, p[1] + a.sin() * 1.2]);
                let _ = id;
            }
        }
    }

    fn step_shards(&mut self) {
        let alive = self.phase == Phase::Playing;
        let ship = self.ship.body.pos();
        let mut picked = 0;
        self.shards.retain_mut(|s| {
            s.life -= DT;
            if s.life <= 0.0 {
                return false;
            }
            let to = ship - s.body.pos();
            let d = length(to);
            if alive && d < 0.9 {
                picked += 1;
                return false;
            }
            // Drag, and a magnet near the ship.
            let mut v = s.body.vel.gp(1.0 - 2.5 * DT);
            if alive && d < 5.0 {
                v += to.gp(60.0 * DT / (d * d + 1.0));
            }
            s.body.vel = v;
            s.body.step(DT);
            let [x, y] = s.body.xy();
            let (cx, cy) = (x.clamp(-ARENA[0], ARENA[0]), y.clamp(-ARENA[1], ARENA[1]));
            if cx != x || cy != y {
                s.body.shift(cx - x, cy - y);
            }
            true
        });
        for _ in 0..picked {
            self.mult += 1;
            self.events.push(Event::Pickup {
                pos: self.ship.body.xy(),
                mult: self.mult,
            });
        }
    }

    fn step_pending(&mut self) {
        let mut landed = Vec::new();
        let ship = self.ship.body.xy();
        let playing = self.phase == Phase::Playing;
        self.pending.retain_mut(|p| {
            p.t -= DT;
            if p.t > 0.0 {
                return true;
            }
            // Never land inside the safe radius: wait until the player has moved away.
            let d = ((p.pos[0] - ship[0]).powi(2) + (p.pos[1] - ship[1]).powi(2)).sqrt();
            if playing && d < SAFE_RADIUS {
                p.t = 0.1;
                return true;
            }
            landed.push((p.kind, p.pos));
            false
        });
        for (kind, pos) in landed {
            self.spawn(kind, pos);
            self.events.push(Event::Spawn { pos, kind });
        }
    }

    fn collide_ship(&mut self) {
        if self.ship.invulnerable > 0.0 {
            return;
        }
        let s = self.ship.body.pos();
        let hit = self
            .enemies
            .iter()
            .any(|e| e.age > 0.25 && distance(e.body.pos(), s) < e.radius * 0.8 + SHIP_RADIUS);
        if hit {
            self.die();
        }
    }

    fn die(&mut self) {
        let pos = self.ship.body.xy();
        self.events.push(Event::Death { pos });
        self.ship.alive = false;
        self.lives -= 1;
        self.mult = 1;
        self.bullets.clear();
        self.pending.clear();
        while !self.enemies.is_empty() {
            self.kill(self.enemies.len() - 1, false);
        }
        self.hitstop = 0.15;
        if self.lives == 0 {
            self.phase = Phase::Over;
            self.events.push(Event::GameOver);
        } else {
            self.phase = Phase::Dead(2.2);
        }
    }

    fn respawn(&mut self) {
        self.phase = Phase::Playing;
        self.ship.alive = true;
        self.ship.invulnerable = 2.5;
        self.ship.body = Body::new(pose_at(0.0, 0.0, core::f32::consts::FRAC_PI_2));
        self.director.after_death();
        self.events.push(Event::Respawn);
    }

    fn bomb(&mut self) {
        if self.bombs == 0 {
            return;
        }
        self.bombs -= 1;
        let pos = self.ship.body.xy();
        self.events.push(Event::Bomb { pos });
        while !self.enemies.is_empty() {
            self.kill(self.enemies.len() - 1, false);
        }
        self.pending.clear();
        self.ship.invulnerable = self.ship.invulnerable.max(1.0);
        self.director.after_bomb();
    }

    fn extras(&mut self) {
        while self.score >= self.next_life {
            self.next_life += 100_000;
            self.lives += 1;
            self.events.push(Event::Extra { life: true });
        }
        while self.score >= self.next_bomb {
            self.next_bomb += 150_000;
            self.bombs += 1;
            self.events.push(Event::Extra { life: false });
        }
    }
}

/// An angle in `(-π, π]`.
pub fn wrap_angle(a: f32) -> f32 {
    use core::f32::consts::{PI, TAU};
    let mut a = a % TAU;
    if a > PI {
        a -= TAU;
    } else if a <= -PI {
        a += TAU;
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    fn aim(x: f32, y: f32) -> Input {
        Input {
            aim: Point::direction(x, y),
            fire: true,
            ..Input::default()
        }
    }

    #[test]
    fn shots_kill_a_drifter_in_their_path() {
        let mut w = World::new(1);
        w.director.enabled = false;
        w.spawn(Kind::Drifter, [8.0, 0.0]);
        w.enemies[0].body.vel = Point::direction(0.0, 0.0);
        w.enemies[0].body.spin = 0.0;
        let mut killed = false;
        for _ in 0..120 {
            w.tick(&aim(1.0, 0.0));
            killed |= w.events.iter().any(|e| {
                matches!(
                    e,
                    Event::Kill {
                        kind: Kind::Drifter,
                        scored: true,
                        ..
                    }
                )
            });
        }
        assert!(killed, "the drifter survived");
        assert_eq!(w.score, 50);
        assert!(!w.shards.is_empty());
    }

    #[test]
    fn a_chaser_catches_a_still_ship() {
        let mut w = World::new(2);
        w.director.enabled = false;
        w.ship.invulnerable = 0.0;
        w.spawn(Kind::Chaser, [12.0, 5.0]);
        let mut died = false;
        for _ in 0..120 * 10 {
            w.tick(&Input::default());
            died |= w.events.iter().any(|e| matches!(e, Event::Death { .. }));
        }
        assert!(died);
        assert_eq!(w.lives, 2);
    }

    #[test]
    fn singularities_bend_shots_and_eat_enemies() {
        let mut w = World::new(3);
        w.director.enabled = false;
        w.spawn(Kind::Singularity, [0.0, 6.0]);
        w.enemies[0].body.vel = Point::direction(0.0, 0.0);
        // A shot passing beside the well curves towards it.
        w.bullets.push(Bullet {
            pos: Point::xy(-20.0, 3.5),
            prev: Point::xy(-20.0, 3.5),
            vel: Point::direction(40.0, 0.0),
            life: 2.0,
        });
        for _ in 0..58 {
            w.tick(&Input::default());
        }
        let b = w.bullets.first().map(|b| b.pos.to_euclidean());
        assert!(
            b.is_none() || b.unwrap()[1] > 3.6,
            "the shot did not bend: {b:?}"
        );
        // A drifter falling in is eaten.
        w.bullets.clear();
        let s = w.enemies[0].body.xy();
        w.spawn(Kind::Drifter, [s[0], s[1] + 1.2]);
        let mut ate = false;
        for _ in 0..240 {
            w.tick(&Input::default());
            ate |= w.events.iter().any(|e| matches!(e, Event::Absorb { .. }));
        }
        assert!(ate);
    }

    #[test]
    fn a_bomb_clears_the_screen() {
        let mut w = World::new(4);
        w.director.enabled = false;
        for i in 0..10 {
            w.spawn(Kind::Drifter, [-20.0 + 4.0 * i as f32, 10.0]);
        }
        w.tick(&Input {
            bomb: true,
            ..Input::default()
        });
        assert!(w.enemies.is_empty());
        assert_eq!(w.bombs, 2);
    }

    #[test]
    fn multiplier_rises_with_shards() {
        let mut w = World::new(5);
        w.director.enabled = false;
        let mut s = Body::new(pose_at(0.5, 0.0, 0.0));
        s.vel = Point::direction(0.0, 0.0);
        w.shards.push(Shard { body: s, life: 5.0 });
        w.tick(&Input::default());
        assert_eq!(w.mult, 2);
    }

    #[test]
    fn same_seed_same_run() {
        let run = |seed| {
            let mut w = World::new(seed);
            let mut h = 0u64;
            for t in 0..120 * 60 {
                let a = t as f32 * 0.01;
                w.tick(&Input {
                    movement: Point::direction(a.cos() * 0.8, (a * 1.3).sin() * 0.8),
                    aim: Point::direction((a * 2.0).cos(), (a * 2.0).sin()),
                    fire: true,
                    bomb: t % 2000 == 1999,
                });
                h = h.wrapping_mul(31).wrapping_add(
                    w.score ^ w.enemies.len() as u64 ^ u64::from(w.ship.body.xy()[0].to_bits()),
                );
            }
            (h, w.score)
        };
        assert_eq!(run(42), run(42));
        assert_ne!(run(42).0, run(43).0);
    }
}
