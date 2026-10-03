//! The simulation: a pure, headless core with no Bevy and no wgpu types. It advances at a
//! fixed 120 Hz from a seed and the players' inputs, so a run is reproducible, and reports what
//! happened as [`Event`]s for the renderer and the audio.
//!
//! All geometry is gax PGA2D: poses are unit motors, velocities twists, positions points,
//! directions points at infinity, the arena's walls lines. Steering turns directions by the
//! rotation between them, bounces reflect in the walls, and collisions are joins and signed
//! distances.

pub mod body;
pub mod collide;
pub mod director;
pub mod replay;
pub mod rng;

use body::{
    Body, ORIGIN, Pose, distance, heading, interpolate, place, pose_at, turn, turned, with_length,
};
use collide::{SpatialHash, in_front, segment_hits_circle, walls};
use director::Director;
use gax::pga2d::{Line, Point};
use rng::Rng;

/// A point or a direction.
pub type P = Point<(), f32>;

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

/// The point `(x, y)`.
pub fn at(x: f32, y: f32) -> P {
    Point::xy(x, y)
}

/// The direction `(x, y)`.
pub fn dir(x: f32, y: f32) -> P {
    Point::direction(x, y)
}

/// One tick's input, in gax types (see `input.rs`, the only boundary with Bevy's vectors).
#[derive(Clone, Copy, Debug)]
pub struct Input {
    /// Movement: a direction of length at most 1.
    pub movement: P,
    /// Aim: a direction (zero for none).
    pub aim: P,
    /// Fire held.
    pub fire: bool,
    /// Bomb pressed (edge).
    pub bomb: bool,
}

impl Default for Input {
    fn default() -> Input {
        Input {
            movement: dir(0.0, 0.0),
            aim: dir(0.0, 0.0),
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
    /// A small, fast chaser from a burst singularity or a carrier.
    Mote,
    /// Approaches, but sidesteps shots: the distance to each shot's line of flight.
    Evader,
    /// Tough; splits into three fragments.
    Splitter,
    /// A fast dart from a splitter.
    Fragment,
    /// A chain of segments following the head by motor interpolation; only the head is
    /// vulnerable, the body blocks shots.
    Serpent,
    /// Shielded in front (shots reflect off the shield line): flank it.
    Warden,
    /// Slow and tough; launches motes, and a ring of them when destroyed.
    Carrier,
}

/// Segments of a serpent's body.
pub const SERPENT_LEN: usize = 12;
/// Distance between serpent segments.
pub const SERPENT_GAP: f32 = 0.55;

impl Kind {
    /// Base score.
    pub fn value(self) -> u64 {
        match self {
            Kind::Drifter => 50,
            Kind::Chaser => 100,
            Kind::Singularity => 500,
            Kind::Mote => 25,
            Kind::Evader => 100,
            Kind::Splitter => 150,
            Kind::Fragment => 50,
            Kind::Serpent => 400,
            Kind::Warden => 250,
            Kind::Carrier => 600,
        }
    }

    /// Collision radius (singularities grow).
    pub fn radius(self) -> f32 {
        match self {
            Kind::Drifter => 0.62,
            Kind::Chaser => 0.58,
            Kind::Singularity => 0.9,
            Kind::Mote => 0.32,
            Kind::Evader => 0.55,
            Kind::Splitter => 0.75,
            Kind::Fragment => 0.34,
            Kind::Serpent => 0.55,
            Kind::Warden => 0.8,
            Kind::Carrier => 1.15,
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
    pub pull: P,
    /// A serpent's body, head first (empty for others).
    pub chain: Vec<Pose>,
    /// A timer for periodic actions (a carrier's launches).
    pub timer: f32,
}

/// A shot.
#[derive(Clone, Copy, Debug)]
pub struct Bullet {
    /// Position (normalized point).
    pub pos: P,
    /// Position at the start of the tick.
    pub prev: P,
    /// Velocity (a direction).
    pub vel: P,
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
    pub pos: P,
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

/// A killed enemy's motion and size: its outline breaks into pieces that carry them on.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Wreck {
    /// Its pose.
    pub pose: Pose,
    /// Its linear velocity (a direction).
    pub vel: P,
    /// Its angular velocity.
    pub spin: f32,
    /// Its drawn radius.
    pub radius: f32,
}

impl Wreck {
    /// The world velocity of its point at `p` (a unit point): its twist's, a translation at
    /// `vel` plus a turn at `spin` about its centre.
    pub fn velocity_at(&self, p: P) -> P {
        let twist = Point::translation_twist(self.vel.e20(), self.vel.e01())
            + Point::rotation_twist(self.pose >> ORIGIN, self.spin);
        body::velocity_at(twist, p)
    }
}

/// What happened in a tick, for the renderer (particles, grid impulses, shake) and the audio.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Event {
    /// A shot left the gun, along `dir` (a unit direction).
    Fire { pos: P, dir: P },
    /// A bullet hit an enemy that survived.
    Hit { pos: P, kind: Kind },
    /// An enemy died.
    Kill {
        pos: P,
        kind: Kind,
        size: f32,
        scored: bool,
        /// Points gained (with the multiplier).
        points: u64,
        /// Its motion and size, for the wreckage of its outline (`None` for a serpent's body
        /// and the tunnel's foes).
        wreck: Option<Wreck>,
    },
    /// A bullet hit a wall.
    Wall { pos: P },
    /// A shot glanced off a shield or a serpent's body.
    Deflect { pos: P, kind: Kind },
    /// A singularity ate an enemy.
    Absorb { pos: P, mass: f32 },
    /// An overfed singularity burst.
    Burst { pos: P },
    /// A spawn was announced.
    Warn { pos: P, kind: Kind },
    /// A spawn landed.
    Spawn { pos: P, kind: Kind },
    /// A shard was collected.
    Pickup { pos: P, mult: u32 },
    /// The ship was destroyed.
    Death { pos: P },
    /// The ship (re)appeared.
    Respawn,
    /// A bomb went off.
    Bomb { pos: P },
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

/// The first wall that `p` is outside of, `margin` in from the arena's edge.
fn outside(p: P, margin: f32) -> Option<Line<(), f32>> {
    walls(ARENA[0] - margin, ARENA[1] - margin)
        .into_iter()
        .find(|w| (*w & p).s() < 0.0)
}

/// The foot of `p` on the line `l`: the meet of `l` with the perpendicular through `p`.
fn foot(l: Line<(), f32>, p: P) -> P {
    crate::geom::foot(p, l).unitized()
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
    pub fn wells(&self) -> impl Iterator<Item = (P, f32)> + '_ {
        self.enemies
            .iter()
            .filter(|e| e.kind == Kind::Singularity && e.age > 0.0)
            .map(|e| (e.body.pos(), 18.0 + 6.0 * e.mass))
    }

    /// The acceleration the wells give a body at `p`.
    fn gravity(&self, p: P, scale: f32) -> P {
        pull(self.wells(), p, scale)
    }

    /// Spawn an enemy now.
    pub fn spawn(&mut self, kind: Kind, at: P) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        let mut body = Body::new(place(at, self.rng.angle()));
        let speed = match kind {
            Kind::Drifter => 2.2,
            Kind::Mote | Kind::Fragment => 6.0,
            Kind::Singularity => 0.8,
            Kind::Carrier => 1.4,
            _ => 0.0,
        };
        body.vel = heading(self.rng.angle(), speed);
        body.spin = match kind {
            Kind::Drifter => self.rng.range(-2.0, 2.0),
            Kind::Singularity => 0.6,
            Kind::Carrier => 0.7,
            Kind::Splitter => 2.0,
            _ => 0.0,
        };
        let chain = if kind == Kind::Serpent {
            (1..=SERPENT_LEN)
                .map(|k| body.pose * gax::pga2d::Motor::translation(-SERPENT_GAP * k as f32, 0.0))
                .collect()
        } else {
            Vec::new()
        };
        self.enemies.push(Enemy {
            id,
            kind,
            body,
            hp: match kind {
                Kind::Singularity => 25.0,
                Kind::Splitter => 2.0,
                Kind::Serpent => 4.0,
                Kind::Carrier => 14.0,
                _ => 1.0,
            },
            radius: kind.radius(),
            mass: 0.0,
            age: 0.0,
            flash: 0.0,
            phase: self.rng.range(0.0, 100.0),
            pull: dir(0.0, 0.0),
            chain,
            timer: 2.0,
        });
        id
    }

    /// Spawn an enemy now, moving at `vel`.
    fn spawn_moving(&mut self, kind: Kind, at: P, vel: P) {
        self.spawn(kind, at);
        let last = self.enemies.len() - 1;
        self.enemies[last].body.vel = vel;
    }

    /// Announce a spawn: it lands after `t` seconds, never within the safe radius.
    pub fn announce(&mut self, kind: Kind, pos: P, t: f32) {
        let pos = self.safe_spot(pos);
        self.pending.push(Pending {
            kind,
            pos,
            t,
            total: t,
        });
        self.events.push(Event::Warn { pos, kind });
    }

    /// `p` clamped into the arena (a box: its coordinates bounded), then pushed out of the safe
    /// radius around the ship along the ray from it (or back through it, if that leaves the
    /// arena).
    pub fn safe_spot(&self, p: P) -> P {
        let inside = |p: P| {
            let [x, y] = p.to_euclidean();
            at(
                x.clamp(-ARENA[0] + 1.5, ARENA[0] - 1.5),
                y.clamp(-ARENA[1] + 1.5, ARENA[1] - 1.5),
            )
        };
        let p = inside(p);
        let s = self.ship.body.pos();
        let need = SAFE_RADIUS + 1.0;
        if distance(p, s) >= need {
            return p;
        }
        let away = p - s;
        let u = if away.ideal_norm() > 1e-3 {
            with_length(away, need)
        } else {
            dir(need, 0.0)
        };
        let (out, back) = (inside(s + u), inside(s - u));
        if distance(out, s) >= distance(back, s) {
            out
        } else {
            back
        }
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
        let want = input.movement * SHIP_SPEED;
        // Quick, smooth response: approach the wanted velocity exponentially.
        let k = 1.0 - (-14.0 * DT).exp();
        ship.body.vel = ship.body.vel + (want - ship.body.vel) * k;
        // Turn the nose towards the motion: by the rotation from heading to velocity.
        ship.body.spin = if ship.body.vel.ideal_norm() > 0.5 {
            (turn(ship.body.heading(), ship.body.vel) * 16.0).clamp(-24.0, 24.0)
        } else {
            0.0
        };
        ship.body.step(DT);
        // Walls: stop at the wall, keeping only the motion along it (the mean of the velocity
        // and its reflection in the wall).
        for _ in 0..2 {
            let Some(wall) = outside(ship.body.pos(), SHIP_RADIUS) else {
                break;
            };
            ship.body.move_to(foot(wall, ship.body.pos()));
            ship.body.vel = (ship.body.vel + wall.reflect(ship.body.vel)) * 0.5;
        }
        // Fire: two alternating barrels, along the aim.
        ship.cooldown -= DT;
        if input.fire && input.aim.ideal_norm() > 0.2 && ship.cooldown <= 0.0 {
            ship.cooldown += FIRE_INTERVAL;
            let u = with_length(input.aim, 1.0);
            // One shot per interval, from alternating barrels: offset across the aim, a
            // quarter turn of it.
            let side = if ship.barrel { 0.14 } else { -0.14 };
            let across = turned(u, core::f32::consts::FRAC_PI_2);
            let o = ship.body.pos();
            let pos = o + u * 0.4 + across * side;
            self.bullets.push(Bullet {
                pos,
                prev: pos,
                vel: u * BULLET_SPEED,
                life: BULLET_LIFE,
            });
            ship.barrel = !ship.barrel;
            self.events.push(Event::Fire { pos: o, dir: u });
        }
        ship.cooldown = ship.cooldown.max(-FIRE_INTERVAL);
    }

    fn step_bullets(&mut self) {
        // Enemies into the hash; a serpent's body segments too (under the serpent's index).
        self.hash.clear();
        for (i, e) in self.enemies.iter().enumerate() {
            if e.age >= 0.0 {
                self.hash.insert(i as u32, e.body.pos(), e.radius);
                for seg in &e.chain {
                    self.hash.insert(i as u32, *seg >> ORIGIN, 0.45);
                }
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
            b.vel += self.gravity(b.pos, 40.0) * DT;
            b.prev = b.pos;
            b.pos += b.vel * DT;
            if let Some(wall) = outside(b.pos, 0.0) {
                // Where the path met the wall.
                let pos = ((b.prev & b.pos) ^ wall).unitized();
                self.events.push(Event::Wall { pos });
                return false;
            }
            self.hash.query(b.prev, b.pos, 1.5, &mut self.scratch);
            for &i in &self.scratch {
                let e = &mut self.enemies[i as usize];
                if e.hp <= 0.0 {
                    continue;
                }
                // A serpent's body blocks the shot.
                if let Some(seg) = e
                    .chain
                    .iter()
                    .find(|s| segment_hits_circle(b.prev, b.pos, **s >> ORIGIN, 0.42))
                {
                    let pos = *seg >> ORIGIN;
                    self.events.push(Event::Deflect { pos, kind: e.kind });
                    return false;
                }
                if !segment_hits_circle(b.prev, b.pos, e.body.pos(), e.radius + 0.08) {
                    continue;
                }
                // A warden's shield reflects shots from the front, in the shield's line.
                if e.kind == Kind::Warden && in_front(e.body.pose, b.prev) {
                    let shield = (e.body.pose >> ORIGIN) & (e.body.pose >> at(0.0, 1.0));
                    b.vel = shield.reflect(b.vel);
                    b.pos = b.prev;
                    self.events.push(Event::Deflect {
                        pos: e.body.pos(),
                        kind: e.kind,
                    });
                    return true;
                }
                e.hp -= 1.0;
                e.flash = 1.0;
                if matches!(e.kind, Kind::Singularity | Kind::Carrier) {
                    // A hit pushes it back a little.
                    e.body.vel += b.vel * 0.004;
                }
                if e.hp <= 0.0 {
                    kills.push(i as usize);
                } else {
                    self.events.push(Event::Hit {
                        pos: e.body.pos(),
                        kind: e.kind,
                    });
                }
                return false;
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
        let pos = e.body.pos();
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
                let mut body = Body::new(place(pos, a));
                body.vel = heading(a, self.rng.range(2.0, 7.0));
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
            wreck: Some(Wreck {
                pose: e.body.pose,
                vel: e.body.vel,
                spin: e.body.spin,
                radius: e.radius,
            }),
        });
        // A serpent's body goes with its head.
        for seg in &e.chain {
            self.events.push(Event::Kill {
                pos: *seg >> ORIGIN,
                kind: Kind::Serpent,
                size: 0.4,
                scored: false,
                points: 0,
                wreck: None,
            });
        }
        if scored {
            match e.kind {
                // Three fragments, flung apart: a direction turned a third at a time.
                Kind::Splitter => {
                    let mut d = heading(self.rng.angle(), 1.0);
                    for _ in 0..3 {
                        self.spawn_moving(Kind::Fragment, pos + d * 0.5, d * 9.0);
                        let last = self.enemies.len() - 1;
                        self.enemies[last].age = 0.25;
                        d = turned(d, core::f32::consts::TAU / 3.0);
                    }
                }
                // A carrier's cargo: a ring of motes.
                Kind::Carrier => {
                    let mut d = dir(1.0, 0.0);
                    for _ in 0..10 {
                        self.spawn_moving(Kind::Mote, pos + d * 1.5, d * 7.0);
                        d = turned(d, core::f32::consts::TAU / 10.0);
                    }
                    self.events.push(Event::Burst { pos });
                }
                _ => {}
            }
        }
    }

    fn step_enemies(&mut self) {
        let ship = self.ship.body.pos();
        let hunting = self.phase == Phase::Playing;
        let wells: Vec<(P, f32)> = self
            .enemies
            .iter()
            .filter(|e| e.kind == Kind::Singularity)
            .map(|e| (e.body.pos(), 18.0 + 6.0 * e.mass))
            .collect();
        let time = self.time;
        // Shots in flight, for evaders: `(line of flight, position, velocity)`.
        let shots: Vec<(Line<(), f32>, P, P)> = self
            .bullets
            .iter()
            .map(|b| {
                let l = (b.pos & b.vel).normalized().into_inner();
                (l, b.pos, b.vel)
            })
            .collect();
        let mut launches: Vec<P> = Vec::new();
        for e in &mut self.enemies {
            e.age += DT;
            e.flash = (e.flash - DT * 6.0).max(0.0);
            let p = e.body.pos();
            // The enemy's own velocity, without the wells' pull.
            e.body.vel -= e.pull;
            match e.kind {
                Kind::Drifter => {
                    // Wander: the velocity turns slowly with a smooth noise.
                    let wave = crate::signal::wave;
                    let noise =
                        (wave(time * 0.7 + e.phase) + wave(time * 1.3 + e.phase * 1.7)) * 0.9;
                    e.body.vel = with_length(turned(e.body.vel, noise * DT), 2.2);
                }
                Kind::Chaser
                | Kind::Mote
                | Kind::Fragment
                | Kind::Evader
                | Kind::Splitter
                | Kind::Serpent => {
                    let (speed, turn_rate) = match e.kind {
                        Kind::Chaser => (6.2, 2.6),
                        Kind::Mote => (8.5, 4.0),
                        Kind::Fragment => (8.0, 3.2),
                        Kind::Evader => (4.6, 3.0),
                        Kind::Splitter => (3.0, 1.4),
                        _ => (5.2, 2.0),
                    };
                    // Idle (no one to hunt): towards a point of its own.
                    let target = if hunting {
                        ship
                    } else {
                        at(0.0, 0.0) + heading(e.phase, 1.0) * 15.0
                    };
                    let to = target - p;
                    let have = if e.body.vel.ideal_norm() > 0.1 {
                        e.body.vel
                    } else {
                        to
                    };
                    // A serpent winds as it comes.
                    let wind = if e.kind == Kind::Serpent {
                        crate::signal::wave(time * 2.6 + e.phase) * 0.9
                    } else {
                        0.0
                    };
                    // Turn towards the target at a limited rate: part of the rotation between.
                    let limit = turn_rate * DT;
                    let d = (turn(have, to) + wind).clamp(-limit, limit);
                    let ramp = (e.age * 1.5).min(1.0);
                    let way = with_length(turned(have, d), 1.0);
                    e.body.vel = way * (speed * ramp);
                    e.body.spin = match e.kind {
                        Kind::Chaser => 3.0,
                        Kind::Splitter => 2.0,
                        Kind::Evader | Kind::Serpent => 0.0,
                        _ => 9.0,
                    };
                    if matches!(e.kind, Kind::Evader | Kind::Serpent) {
                        // Face the way it goes.
                        e.body.spin = turn(e.body.heading(), way) * 10.0;
                    }
                    if e.kind == Kind::Evader {
                        // Sidestep: for each shot coming this way whose line of flight passes
                        // close, move away from its foot on the line.
                        let mut dodge = dir(0.0, 0.0);
                        for &(l, bp, bv) in &shots {
                            // Ahead of the shot: on the side of the perpendicular through it
                            // that the shot moves towards.
                            let across = l | bp;
                            let ahead = (across & p).s() * (across & bv).s() > 0.0;
                            if !ahead || distance(p, bp) > 8.0 {
                                continue;
                            }
                            let away = p - foot(l, p);
                            let off = away.ideal_norm();
                            if off < 1.8 {
                                dodge += with_length(away, (1.8 - off) * 9.0);
                            }
                        }
                        let v = e.body.vel + dodge;
                        e.body.vel = if v.ideal_norm() > 11.0 {
                            with_length(v, 11.0)
                        } else {
                            v
                        };
                    }
                }
                Kind::Warden => {
                    // Turn slowly to face the ship, and come on behind the shield.
                    let d = turn(e.body.heading(), ship - p);
                    e.body.spin = d.clamp(-1.0, 1.0) * 1.3;
                    let ramp = (e.age * 1.5).min(1.0);
                    e.body.vel = e.body.heading() * (2.4 * ramp);
                }
                Kind::Carrier => {
                    // Drift, and launch a pair of motes every few seconds.
                    e.body.vel = with_length(e.body.vel, 1.4);
                    e.timer -= DT;
                    if e.timer <= 0.0 && hunting {
                        e.timer = 4.5;
                        launches.push(p);
                    }
                }
                Kind::Singularity => {
                    // Slow drift towards the ship, damped.
                    let towards = with_length(ship - p, 0.25 * DT);
                    e.body.vel = (e.body.vel + towards) * (1.0 - 0.3 * DT);
                    e.radius = 0.9 + 0.36 * gax::Real::sqrt(e.mass);
                    e.hp = e.hp.min(25.0 + 4.0 * e.mass);
                }
            }
            // Wells pull everything but wells: the pull accumulates, and decays slowly.
            e.pull = e.pull * (1.0 - 0.6 * DT);
            if e.kind != Kind::Singularity {
                e.pull += pull(wells.iter().copied(), p, 2.5 * DT);
            }
            e.body.vel += e.pull;
            e.body.step(DT);
            // The body follows: each segment moves towards the spot just behind the one
            // ahead, by motor interpolation.
            let k = 1.0 - (-30.0 * DT).exp();
            let mut ahead = e.body.pose;
            for seg in &mut e.chain {
                let target = ahead * gax::pga2d::Motor::translation(-SERPENT_GAP, 0.0);
                *seg = interpolate(*seg, target, k);
                ahead = *seg;
            }
            // Walls: bounce, by reflection in the wall (the position mirrored back inside, the
            // velocity and the pull mirrored).
            for _ in 0..2 {
                let Some(wall) = outside(e.body.pos(), e.radius) else {
                    break;
                };
                e.body.move_to(wall.reflect(e.body.pos()));
                e.body.vel = wall.reflect(e.body.vel);
                e.pull = wall.reflect(e.pull);
            }
        }
        for p in launches {
            for side in [-1.0f32, 1.0] {
                let d = heading(self.rng.angle(), 1.3 * side);
                self.spawn(Kind::Mote, p + d);
            }
            self.events.push(Event::Spawn {
                pos: p,
                kind: Kind::Mote,
            });
        }
        // Singularities eat what falls in.
        let mut eaten: Vec<(usize, usize)> = Vec::new();
        for (wi, w) in self.enemies.iter().enumerate() {
            if w.kind != Kind::Singularity {
                continue;
            }
            for (i, e) in self.enemies.iter().enumerate() {
                if !matches!(e.kind, Kind::Singularity | Kind::Serpent | Kind::Carrier)
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
            let (pos, mass) = (w.body.pos(), w.mass);
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
                bursts.push(e.body.pos());
                false
            } else {
                true
            }
        });
        for p in bursts {
            self.events.push(Event::Burst { pos: p });
            for _ in 0..14 {
                let d = heading(self.rng.angle(), 1.2);
                self.spawn(Kind::Mote, p + d);
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
            let d = to.ideal_norm();
            if alive && d < 0.9 {
                picked += 1;
                return false;
            }
            // Drag, and a magnet near the ship.
            let mut v = s.body.vel * (1.0 - 2.5 * DT);
            if alive && d < 5.0 {
                v += to * (60.0 * DT / (d * d + 1.0));
            }
            s.body.vel = v;
            s.body.step(DT);
            // Kept in the arena: onto the wall it crossed.
            for _ in 0..2 {
                let Some(wall) = outside(s.body.pos(), 0.0) else {
                    break;
                };
                s.body.move_to(foot(wall, s.body.pos()));
            }
            true
        });
        for _ in 0..picked {
            self.mult += 1;
            self.events.push(Event::Pickup {
                pos: self.ship.body.pos(),
                mult: self.mult,
            });
        }
    }

    fn step_pending(&mut self) {
        let mut landed = Vec::new();
        let ship = self.ship.body.pos();
        let playing = self.phase == Phase::Playing;
        self.pending.retain_mut(|p| {
            p.t -= DT;
            if p.t > 0.0 {
                return true;
            }
            // Never land inside the safe radius: wait until the player has moved away.
            if playing && distance(p.pos, ship) < SAFE_RADIUS {
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
        let hit = self.enemies.iter().any(|e| {
            e.age > 0.25
                && (distance(e.body.pos(), s) < e.radius * 0.8 + SHIP_RADIUS
                    || e.chain
                        .iter()
                        .any(|seg| distance(*seg >> ORIGIN, s) < 0.35 + SHIP_RADIUS))
        });
        if hit {
            self.die();
        }
    }

    fn die(&mut self) {
        let pos = self.ship.body.pos();
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
        let pos = self.ship.body.pos();
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

/// The pull of wells `(position, strength)` on a body at `p`: `k s d / (|d|² + r²)^(3/2)`
/// towards each, an inverse square with a softened core.
fn pull(wells: impl Iterator<Item = (P, f32)>, p: P, k: f32) -> P {
    wells.fold(dir(0.0, 0.0), |acc, (w, s)| {
        let d = w - p;
        acc + d * (k * s / (d.ideal_norm_squared() + 1.5).powf(1.5))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn aim(x: f32, y: f32) -> Input {
        Input {
            aim: dir(x, y),
            fire: true,
            ..Input::default()
        }
    }

    #[test]
    fn shots_kill_a_drifter_in_their_path() {
        let mut w = World::new(1);
        w.director.enabled = false;
        w.spawn(Kind::Drifter, at(8.0, 0.0));
        w.enemies[0].body.vel = dir(0.0, 0.0);
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
        assert_ne!(w.shards.len(), 0);
    }

    #[test]
    fn a_chaser_catches_a_still_ship() {
        let mut w = World::new(2);
        w.director.enabled = false;
        w.ship.invulnerable = 0.0;
        w.spawn(Kind::Chaser, at(12.0, 5.0));
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
        w.spawn(Kind::Singularity, at(0.0, 6.0));
        w.enemies[0].body.vel = dir(0.0, 0.0);
        // A shot passing beside the well curves towards it.
        w.bullets.push(Bullet {
            pos: at(-20.0, 3.5),
            prev: at(-20.0, 3.5),
            vel: dir(40.0, 0.0),
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
        let s = w.enemies[0].body.pos();
        w.spawn(Kind::Drifter, s + dir(0.0, 1.2));
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
            w.spawn(Kind::Drifter, at(-20.0 + 4.0 * i as f32, 10.0));
        }
        w.tick(&Input {
            bomb: true,
            ..Input::default()
        });
        assert_eq!(w.enemies.len(), 0);
        assert_eq!(w.bombs, 2);
    }

    #[test]
    fn multiplier_rises_with_shards() {
        let mut w = World::new(5);
        w.director.enabled = false;
        let mut s = Body::new(pose_at(0.5, 0.0, 0.0));
        s.vel = dir(0.0, 0.0);
        w.shards.push(Shard { body: s, life: 5.0 });
        w.tick(&Input::default());
        assert_eq!(w.mult, 2);
    }

    /// Bodies stay in the arena: an enemy bounces off a wall by reflection (its velocity
    /// mirrored), the ship stops at it (its motion into the wall gone).
    #[test]
    fn walls_reflect_enemies_and_stop_the_ship() {
        let mut w = World::new(9);
        w.director.enabled = false;
        w.phase = Phase::Over;
        w.spawn(Kind::Carrier, at(ARENA[0] - 2.0, 0.0));
        w.enemies[0].body.vel = dir(1.4, 0.0);
        for _ in 0..240 {
            w.tick(&Input::default());
        }
        let v = w.enemies[0].body.vel;
        assert!(v.e20() < 0.0, "did not bounce: {v:?}");
        let [x, _] = w.enemies[0].body.pos().to_euclidean();
        assert!(x < ARENA[0]);
        let mut w = World::new(9);
        w.director.enabled = false;
        for _ in 0..600 {
            w.tick(&Input {
                movement: dir(1.0, 0.3),
                ..Input::default()
            });
        }
        let [x, _] = w.ship.body.pos().to_euclidean();
        assert!((x - (ARENA[0] - SHIP_RADIUS)).abs() < 1e-3, "{x}");
        assert!(w.ship.body.vel.e20().abs() < 1e-3);
    }

    #[test]
    fn same_seed_same_run() {
        let run = |seed| {
            let mut w = World::new(seed);
            let mut h = 0u64;
            for t in 0..120 * 60 {
                let a = t as f32 * 0.01;
                w.tick(&Input {
                    movement: heading(a, 0.8),
                    aim: heading(a * 2.0, 1.0),
                    fire: true,
                    bomb: t % 2000 == 1999,
                });
                h = h.wrapping_mul(31).wrapping_add(
                    w.score ^ w.enemies.len() as u64 ^ u64::from(w.ship.body.pos().e20().to_bits()),
                );
            }
            (h, w.score)
        };
        assert_eq!(run(42), run(42));
        assert_ne!(run(42).0, run(43).0);
    }

    fn still(w: &mut World, kind: Kind, p: P, angle: f32) -> usize {
        w.spawn(kind, p);
        let i = w.enemies.len() - 1;
        let e = &mut w.enemies[i];
        e.body = Body::new(place(p, angle));
        e.age = 1.0;
        i
    }

    #[test]
    fn evaders_leave_the_line_of_fire() {
        let mut w = World::new(10);
        w.director.enabled = false;
        w.phase = Phase::Over; // no hunting: only the dodge moves it sideways
        still(&mut w, Kind::Evader, at(10.0, 0.3), 0.0);
        w.bullets.push(Bullet {
            pos: at(0.0, 0.0),
            prev: at(0.0, 0.0),
            vel: dir(40.0, 0.0),
            life: 2.0,
        });
        let mut hit = false;
        for _ in 0..60 {
            w.tick(&Input::default());
            hit |= w
                .events
                .iter()
                .any(|e| matches!(e, Event::Kill { .. } | Event::Hit { .. }));
        }
        assert!(!hit, "the evader was hit");
        let y = w.enemies.first().map(|e| e.body.pos().to_euclidean()[1]);
        assert!(
            y.is_some_and(|y| y > 0.7),
            "the evader stayed near the line: {y:?}"
        );
    }

    #[test]
    fn wardens_reflect_from_the_front_and_fall_from_behind() {
        let mut w = World::new(11);
        w.director.enabled = false;
        // Facing -x, towards the shots from the left: they glance off.
        still(&mut w, Kind::Warden, at(8.0, 0.0), core::f32::consts::PI);
        w.ship.body = Body::new(pose_at(0.0, 0.0, 0.0));
        let mut deflected = false;
        for _ in 0..30 {
            w.enemies[0].body.vel = dir(0.0, 0.0);
            w.enemies[0].body.spin = 0.0;
            w.tick(&aim(1.0, 0.0));
            deflected |= w.events.iter().any(|e| {
                matches!(
                    e,
                    Event::Deflect {
                        kind: Kind::Warden,
                        ..
                    }
                )
            });
        }
        assert!(deflected);
        assert!(
            w.enemies.iter().any(|e| e.kind == Kind::Warden),
            "shot through the shield"
        );
        // From behind it falls.
        let mut w = World::new(12);
        w.director.enabled = false;
        still(&mut w, Kind::Warden, at(8.0, 0.0), 0.0);
        let mut killed = false;
        for _ in 0..60 {
            w.enemies.iter_mut().for_each(|e| {
                e.body.vel = dir(0.0, 0.0);
                e.body.spin = 0.0;
            });
            w.tick(&aim(1.0, 0.0));
            killed |= w.events.iter().any(|e| {
                matches!(
                    e,
                    Event::Kill {
                        kind: Kind::Warden,
                        scored: true,
                        ..
                    }
                )
            });
        }
        assert!(killed);
    }

    #[test]
    fn serpents_block_with_the_body_and_die_by_the_head() {
        let mut w = World::new(13);
        w.director.enabled = false;
        // Playing (bullets fly), with the ship far away and untouchable.
        w.ship.body = Body::new(pose_at(-28.0, -15.0, 0.0));
        w.ship.invulnerable = 1e9;
        let i = still(&mut w, Kind::Serpent, at(0.0, 5.0), 0.0);
        // Let the body settle behind the head.
        for _ in 0..120 {
            w.enemies[i].body.vel = dir(0.0, 0.0);
            w.tick(&Input::default());
        }
        let chain: Vec<P> = w.enemies[0].chain.iter().map(|s| *s >> ORIGIN).collect();
        let gap = distance(chain[0], w.enemies[0].body.pos());
        // Moving, each segment lags a little behind its spot (speed / follow rate).
        assert!(
            gap > SERPENT_GAP - 0.05 && gap < SERPENT_GAP + 0.3,
            "gap {gap}"
        );
        // A shot at the tail is blocked.
        let tail = *chain.last().unwrap() - dir(0.0, 0.7);
        w.bullets.push(Bullet {
            pos: tail,
            prev: tail,
            vel: dir(0.0, 40.0),
            life: 1.0,
        });
        let mut blocked = false;
        for _ in 0..20 {
            w.tick(&Input::default());
            blocked |= w.events.iter().any(|e| {
                matches!(
                    e,
                    Event::Deflect {
                        kind: Kind::Serpent,
                        ..
                    }
                )
            });
        }
        assert!(blocked);
        assert_eq!(w.enemies[0].hp, 4.0);
    }

    #[test]
    fn splitters_split_and_carriers_unload() {
        let mut w = World::new(14);
        w.director.enabled = false;
        let i = still(&mut w, Kind::Splitter, at(0.0, 3.0), 0.0);
        w.enemies[i].hp = 1.0;
        let j = still(&mut w, Kind::Carrier, at(10.0, -5.0), 0.0);
        w.enemies[j].hp = 1.0;
        w.kill(j, true);
        w.kill(0, true);
        let count = |k| w.enemies.iter().filter(|e| e.kind == k).count();
        assert_eq!(count(Kind::Fragment), 3);
        assert_eq!(count(Kind::Mote), 10);
    }
}
