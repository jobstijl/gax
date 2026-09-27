//! Tunnel: flying into the screen down a twisting tunnel of warped space.
//!
//! A pure simulation like the Plane's (`sim`): a fixed 120 Hz step from a seed. Gameplay lives
//! in straightened coordinates `(x, y, s)`, across the tunnel and along it, as PGA3D points and
//! directions; `track` maps them into the world through a chain of screw motions, which only
//! the drawing needs. Collisions are swept segments against spheres, with joins.

pub mod lattice;
pub mod track;

use crate::sim::rng::Rng;
use crate::sim::{DT, Kind, Phase};
use gax::pga3d::{Line, Motor, Point};
use lattice::{Lattice, RADIUS, Source};
use track::Track;

/// A point or a direction in straightened coordinates.
pub type P = Point<(), f32>;

/// The ship stays this close to the axis.
pub const SHIP_RADIUS: f32 = 5.3;
/// The cruising speed (arc length per second); throttle adds or takes up to 45%.
pub const CRUISE: f32 = 17.0;
/// Where the reticle's aim point lies ahead of the ship.
pub const AIM_DEPTH: f32 = 34.0;
/// Spawns land this far ahead (inside the fog).
const SPAWN_AHEAD: f32 = 88.0;
const SHOT_SPEED: f32 = 60.0;
const FIRE_INTERVAL: f32 = 1.0 / 14.0;
const BOLT_SPEED: f32 = 13.0;
const ROLL_TIME: f32 = 0.4;
const ROLL_ANGLE: f32 = 1.3;

/// The player's input for a tick, already in the tunnel's frame.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Input {
    /// Movement across the tunnel, length at most 1.
    pub movement: [f32; 2],
    /// The reticle: where across the tunnel the aim point lies, at `AIM_DEPTH` ahead.
    pub aim: [f32; 2],
    /// Fire held.
    pub fire: bool,
    /// Bomb (edge).
    pub bomb: bool,
    /// Barrel roll (edge): `-1` or `1`.
    pub roll: i8,
    /// Brake (`-1`) to boost (`1`).
    pub throttle: f32,
}

/// The tunnel's enemies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Foe {
    /// Drones in ring formations that close in around you.
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

    fn radius(self) -> f32 {
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

/// An enemy.
#[derive(Clone, Debug)]
pub struct Enemy {
    /// A stable id.
    pub id: u32,
    /// What it is.
    pub foe: Foe,
    /// Where, and where it was last tick.
    pub pos: P,
    /// Last tick's position.
    pub prev: P,
    /// Velocity.
    pub vel: P,
    /// Hit points.
    pub hp: f32,
    /// Seconds alive.
    pub age: f32,
    /// A hit flash, decaying.
    pub flash: f32,
    /// Formation parameters: an angle and a radius around the ring's centre (drones), the
    /// angle on the wall (turrets).
    pub angle: f32,
    /// The ring's radius (drones).
    pub ring: f32,
    /// Seconds to the next action (turrets: firing; the last 0.35 s is the charge-up).
    pub timer: f32,
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
    /// Across the tunnel `(x, y)` at arc length `s`: a point in straightened coordinates.
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
        self.pos.e021()
    }

    /// Where across the tunnel.
    pub fn xy(&self) -> [f32; 2] {
        [self.pos.e032(), self.pos.e013()]
    }
}

/// Something that happened this tick (positions in straightened coordinates).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Event {
    /// A shot left the gun.
    Fire { pos: [f32; 3] },
    /// A hit that did not kill.
    Hit { pos: [f32; 3], foe: Foe },
    /// A kill.
    Kill {
        pos: [f32; 3],
        foe: Foe,
        points: u64,
    },
    /// A turret fired.
    Bolt { pos: [f32; 3] },
    /// A shot hit the wall.
    Wall { pos: [f32; 3] },
    /// A spawn was announced.
    Warn { pos: [f32; 3], foe: Foe },
    /// A shard collected.
    Pickup { pos: [f32; 3], mult: u32 },
    /// A barrel roll.
    Roll { dir: f32 },
    /// The ship was destroyed.
    Death { pos: [f32; 3] },
    /// The ship reappeared.
    Respawn,
    /// A bomb.
    Bomb { pos: [f32; 3] },
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
    pub pos: [f32; 3],
    /// Seconds to landing.
    pub t: f32,
    /// Ring parameters (drones).
    pub angle: f32,
    /// Ring radius (drones).
    pub ring: f32,
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
    /// Lattice sources this tick (blasts), and lasting ones.
    blasts: Vec<(Source, f32)>,
    /// Intensity, `0..1`.
    pub intensity: f32,
    clock: f32,
    cooldown: f32,
    next_id: u32,
    next_life: u64,
    next_bomb: u64,
}

fn xyz(p: P) -> [f32; 3] {
    [p.e032(), p.e013(), p.e021()]
}

/// Does the segment `a → b` pass within `r` of `c`? The distance to the segment's line is a
/// join: `|a & b & c| / |a & b|` (plane weight over line weight), clamped to the segment by
/// the endpoints.
pub fn segment_hits_sphere(a: P, b: P, c: P, r: f32) -> bool {
    let d2 = |p: P, q: P| {
        let [x, y, z] = xyz(p - q);
        x * x + y * y + z * z
    };
    if d2(a, c) < r * r || d2(b, c) < r * r {
        return true;
    }
    let l: Line<(), f32> = a & b;
    let len = l.norm();
    if len < 1e-6 {
        return false;
    }
    let dist = (l & c).norm() / len;
    if dist > r {
        return false;
    }
    // Between the endpoints: the projection of `c` falls inside `a..b`.
    let [ux, uy, uz] = xyz(b - a);
    let [wx, wy, wz] = xyz(c - a);
    let t = (ux * wx + uy * wy + uz * wz) / (len * len);
    (0.0..=1.0).contains(&t)
}

/// The rotation about the tunnel's axis by `angle` (a motor on straightened coordinates).
pub fn about_axis(angle: f32) -> gax::Unit<Motor<(), f32>> {
    Motor::rotation(Point::xyz(0.0, 0.0, 0.0) & Point::xyz(0.0, 0.0, 1.0), angle)
}

impl World {
    /// A new run.
    pub fn new(seed: u64) -> World {
        let mut track = Track::new(seed);
        track.extend(SPAWN_AHEAD + 60.0, 0.0);
        World {
            seed,
            rng: Rng::new(seed ^ 0x7e11),
            tick: 0,
            time: 0.0,
            track,
            lattice: Lattice::new(0.0),
            ship: Ship {
                pos: Point::xyz(0.0, -2.0, 0.0),
                prev: Point::xyz(0.0, -2.0, 0.0),
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
        match self.phase {
            Phase::Over => {}
            Phase::Dead(t) => {
                if t - DT <= 0.0 {
                    self.respawn();
                } else {
                    self.phase = Phase::Dead(t - DT);
                }
            }
            Phase::Playing => {}
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
        let [mx, my] = if alive { input.movement } else { [0.0, 0.0] };
        let v = Point::direction(mx * 12.0, my * 12.0, ship.speed);
        let mut pos = ship.pos + v.gp(DT);
        // A barrel roll: a rotation about the tunnel's axis, spread over its duration.
        if alive && input.roll != 0 && ship.roll.is_none() && ship.roll_cool <= 0.0 {
            ship.roll = Some(Roll {
                t: 0.0,
                dir: f32::from(input.roll.signum()),
            });
            ship.invulnerable = ship.invulnerable.max(ROLL_TIME + 0.1);
            self.events.push(Event::Roll {
                dir: f32::from(input.roll.signum()),
            });
        }
        if let Some(r) = &mut ship.roll {
            // Eased: fast in the middle.
            let ease = |t: f32| 0.5 - 0.5 * (core::f32::consts::PI * t / ROLL_TIME).cos();
            let d = ease((r.t + DT).min(ROLL_TIME)) - ease(r.t);
            pos = about_axis(r.dir * ROLL_ANGLE * d) >> pos;
            r.t += DT;
            if r.t >= ROLL_TIME {
                ship.roll = None;
                ship.roll_cool = 0.5;
            }
        }
        // Keep inside the tunnel.
        let [x, y, s] = xyz(pos);
        let r = (x * x + y * y).sqrt();
        let k = if r > SHIP_RADIUS {
            SHIP_RADIUS / r
        } else {
            1.0
        };
        ship.pos = Point::xyz(x * k, y * k, s);
    }

    fn fire(&mut self, input: &Input) {
        let ship = &mut self.ship;
        if !input.fire || ship.cooldown > 0.0 {
            return;
        }
        ship.cooldown = FIRE_INTERVAL;
        ship.barrel = !ship.barrel;
        let [x, y, s] = xyz(ship.pos);
        let side = if ship.barrel { 0.35 } else { -0.35 };
        let from = [x + side, y, s + 0.8];
        // Towards the aim point: across the tunnel by the time it is `AIM_DEPTH` ahead.
        let t = AIM_DEPTH / SHOT_SPEED;
        let vel = Point::direction(
            (input.aim[0] - from[0]) / t,
            (input.aim[1] - from[1]) / t,
            ship.speed + SHOT_SPEED,
        );
        let pos = Point::xyz(from[0], from[1], from[2]);
        self.shots.push(Shot {
            pos,
            prev: pos,
            vel,
            life: 1.6,
        });
        self.events.push(Event::Fire { pos: from });
    }

    fn bomb(&mut self) {
        if self.bombs == 0 {
            return;
        }
        self.bombs -= 1;
        let s = self.ship.s();
        let mut killed = Vec::new();
        for (i, e) in self.enemies.iter().enumerate() {
            if e.pos.e021() - s < 70.0 {
                killed.push(i);
            }
        }
        for i in killed.into_iter().rev() {
            self.kill(i, false);
        }
        self.bolts.clear();
        self.pending.retain(|p| p.pos[2] - s > 70.0);
        let [x, y] = self.ship.xy();
        self.events.push(Event::Bomb { pos: [x, y, s] });
        for k in 0..6 {
            self.blast([0.0, 0.0, s + 4.0 + k as f32 * 9.0], 900.0, 16.0, 0.3);
        }
        self.cooldown = self.cooldown.max(2.0);
    }

    fn blast(&mut self, pos: [f32; 3], strength: f32, r2: f32, time: f32) {
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
        let pend = |foe, pos, angle, ring| Pending {
            foe,
            pos,
            t: 0.8,
            angle,
            ring,
        };
        if self.clock > 25.0 && roll < 0.25 + 0.15 * i {
            // Turrets on the wall, spaced along the track.
            let n = 1 + (i * 3.0) as usize;
            let a0 = r.angle();
            for k in 0..n {
                let a = a0 + k as f32 * 2.1;
                let p = [
                    (RADIUS - 0.5) * a.cos(),
                    (RADIUS - 0.5) * a.sin(),
                    s + k as f32 * 7.0,
                ];
                spawn.push(pend(Foe::Turret, p, a, 0.0));
            }
        } else if self.clock > 10.0 && roll < 0.55 {
            // Mines scattered in the lane.
            let n = 3 + (i * 6.0) as usize;
            for k in 0..n {
                let a = r.angle();
                let d = r.range(0.0, SHIP_RADIUS);
                spawn.push(pend(
                    Foe::Mine,
                    [d * a.cos(), d * a.sin(), s + k as f32 * 5.0],
                    0.0,
                    0.0,
                ));
            }
        } else {
            // A ring of drones.
            let n = 6 + (i * 6.0) as usize;
            let a0 = r.angle();
            for k in 0..n {
                let a = a0 + k as f32 * core::f32::consts::TAU / n as f32;
                let ring = 4.6;
                spawn.push(pend(
                    Foe::Drone,
                    [ring * a.cos(), ring * a.sin(), s],
                    a,
                    ring,
                ));
            }
        }
        for p in spawn {
            self.events.push(Event::Warn {
                pos: p.pos,
                foe: p.foe,
            });
            self.pending.push(p);
        }
        self.cooldown = 1.6 + 2.8 * (1.0 - i) + self.rng.range(0.0, 1.0);
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
            let pos = Point::xyz(p.pos[0], p.pos[1], p.pos[2]);
            let drift = if p.foe == Foe::Mine {
                let a = self.rng.angle();
                Point::direction(a.cos() * 0.7, a.sin() * 0.7, 0.0)
            } else {
                Point::direction(0.0, 0.0, 0.0)
            };
            self.enemies.push(Enemy {
                id,
                foe: p.foe,
                pos,
                prev: pos,
                vel: drift,
                hp: p.foe.hp(),
                age: 0.0,
                flash: 0.0,
                angle: p.angle,
                ring: p.ring,
                timer: self.rng.range(0.6, 1.8),
            });
        }
    }

    fn step_enemies(&mut self) {
        let [sx, sy, ss] = xyz(self.ship.pos);
        let speed = self.ship.speed;
        let alive = self.phase == Phase::Playing;
        let mut fired = Vec::new();
        for e in &mut self.enemies {
            e.prev = e.pos;
            e.age += DT;
            e.flash = (e.flash - DT * 6.0).max(0.0);
            let [x, y, s] = xyz(e.pos);
            let ahead = s - ss;
            match e.foe {
                Foe::Drone => {
                    // The ring turns; within reach it closes in around where you are (fly
                    // through its middle), and drifts towards you.
                    e.angle += DT * 0.9;
                    let close = ((40.0 - ahead) / 30.0).clamp(0.0, 1.0);
                    let r = e.ring * (1.0 - 0.5 * close);
                    let (cx, cy) = (sx * close, sy * close);
                    let (tx, ty) = (cx + r * e.angle.cos(), cy + r * e.angle.sin());
                    e.vel = Point::direction((tx - x) * 3.0, (ty - y) * 3.0, -4.0 * close);
                }
                Foe::Mine => {
                    // Drift, bouncing off the tunnel's inner radius.
                    let rr = (x * x + y * y).sqrt();
                    if rr > SHIP_RADIUS + 0.3 && x * e.vel.e032() + y * e.vel.e013() > 0.0 {
                        let (nx, ny) = (x / rr, y / rr);
                        let dot = e.vel.e032() * nx + e.vel.e013() * ny;
                        e.vel = Point::direction(
                            e.vel.e032() - 2.0 * dot * nx,
                            e.vel.e013() - 2.0 * dot * ny,
                            0.0,
                        );
                    }
                }
                Foe::Turret => {
                    e.timer -= DT;
                    if e.timer <= 0.0 {
                        e.timer = 1.5;
                        if alive && (12.0..62.0).contains(&ahead) {
                            // Aim where the ship will be across the tunnel when the bolt
                            // arrives; you read it by its growth and its colour.
                            let t = ahead / (speed + BOLT_SPEED);
                            let vel = Point::direction((sx - x) / t, (sy - y) / t, -BOLT_SPEED);
                            fired.push((e.pos, vel));
                        }
                    }
                }
            }
            e.pos += e.vel.gp(DT);
        }
        for (pos, vel) in fired {
            self.bolts.push(Shot {
                pos,
                prev: pos,
                vel,
                life: 8.0,
            });
            self.events.push(Event::Bolt { pos: xyz(pos) });
        }
        // Behind the ship: gone.
        self.enemies.retain(|e| e.pos.e021() > ss - 6.0);
    }

    fn step_shots(&mut self) {
        let mut hits: Vec<(usize, usize)> = Vec::new();
        for (k, b) in self.shots.iter_mut().enumerate() {
            b.prev = b.pos;
            b.pos += b.vel.gp(DT);
            b.life -= DT;
            for (i, e) in self.enemies.iter().enumerate() {
                if segment_hits_sphere(b.prev, b.pos, e.pos, e.foe.radius()) {
                    hits.push((k, i));
                    b.life = -1.0;
                    break;
                }
            }
            if b.life > 0.0 {
                let [x, y, _] = xyz(b.pos);
                if x * x + y * y > RADIUS * RADIUS {
                    b.life = -2.0;
                }
            }
        }
        for b in &self.shots {
            if b.life == -2.0 {
                self.events.push(Event::Wall { pos: xyz(b.pos) });
            }
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
                    pos: xyz(e.pos),
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
        let pos = xyz(e.pos);
        let points = if scored {
            (e.foe.points() as f32 * self.mult as f32 * self.speed_bonus()).round() as u64
        } else {
            0
        };
        self.score += points;
        self.events.push(Event::Kill {
            pos,
            foe: e.foe,
            points,
        });
        let size = if e.foe == Foe::Turret { 700.0 } else { 260.0 };
        self.blast(pos, size, 3.0, 0.12);
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
            b.pos += b.vel.gp(DT);
            b.life -= DT;
        }
        self.bolts
            .retain(|b| b.life > 0.0 && b.pos.e021() > s - 4.0);
    }

    fn step_shards(&mut self) {
        let ship = self.ship.pos;
        let alive = self.phase == Phase::Playing;
        let mut got = 0;
        self.shards.retain_mut(|sh| {
            sh.life -= DT;
            let [dx, dy, ds] = xyz(ship - sh.pos);
            let d = (dx * dx + dy * dy + ds * ds).sqrt();
            // Magnetic within reach.
            if alive && d < 16.0 {
                let pull = 30.0 / d.max(0.5);
                sh.pos += Point::direction(dx, dy, ds).gp(pull * DT);
            }
            if alive && d < 1.4 {
                got += 1;
                return false;
            }
            sh.life > 0.0 && ds < 4.0
        });
        for _ in 0..got {
            self.mult += 1;
            let [x, y, s] = xyz(self.ship.pos);
            self.events.push(Event::Pickup {
                pos: [x, y, s],
                mult: self.mult,
            });
        }
    }

    fn collide_ship(&mut self) {
        if self.ship.invulnerable > 0.0 {
            return;
        }
        let p = self.ship.pos;
        let prev = self.ship.prev;
        let hit_enemy = self
            .enemies
            .iter()
            .any(|e| segment_hits_sphere(prev, p, e.pos, e.foe.radius() + 0.45));
        let hit_bolt = self
            .bolts
            .iter()
            .any(|b| segment_hits_sphere(b.prev, b.pos, p, 0.55));
        if hit_enemy || hit_bolt {
            self.die();
        }
    }

    fn die(&mut self) {
        let pos = xyz(self.ship.pos);
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
        self.ship.pos = Point::xyz(0.0, -2.0, s);
        self.ship.prev = self.ship.pos;
        self.ship.invulnerable = 2.5;
        self.bolts.clear();
        self.enemies.retain(|e| e.pos.e021() - s > 30.0);
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
        let [x, y] = self.ship.xy();
        if self.phase == Phase::Playing {
            sources.push(Source {
                pos: [x, y, s + 1.0],
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

    #[test]
    fn segments_hit_spheres_by_the_join() {
        let a = Point::xyz(0.0, 0.0, 0.0);
        let b = Point::xyz(0.0, 0.0, 10.0);
        assert!(segment_hits_sphere(a, b, Point::xyz(0.5, 0.0, 5.0), 0.6));
        assert!(!segment_hits_sphere(a, b, Point::xyz(0.7, 0.0, 5.0), 0.6));
        // Beyond the ends: only the endpoints count.
        assert!(!segment_hits_sphere(a, b, Point::xyz(0.3, 0.0, 11.0), 0.6));
        assert!(segment_hits_sphere(a, b, Point::xyz(0.3, 0.0, 10.4), 0.6));
        // Diagonal.
        let c = Point::xyz(3.0, 4.0, 12.0);
        assert!(segment_hits_sphere(a, c, Point::xyz(1.5, 2.0, 6.0), 0.1));
    }

    #[test]
    fn a_barrel_roll_turns_the_ship_about_the_axis() {
        let mut w = World::new(1);
        w.ship.pos = Point::xyz(4.0, 0.0, 0.0);
        w.ship.invulnerable = 0.0;
        w.tick(&Input {
            roll: 1,
            ..Input::default()
        });
        for _ in 0..60 {
            w.tick(&Input::default());
        }
        let [x, y] = w.ship.xy();
        let a = y.atan2(x);
        assert!((a - ROLL_ANGLE).abs() < 0.02, "{a}");
        assert!(((x * x + y * y).sqrt() - 4.0).abs() < 1e-3);
    }

    /// Shots converge on the reticle's aim point at `AIM_DEPTH` ahead of the ship.
    #[test]
    fn shots_fly_to_the_aim_point() {
        let mut w = World::new(1);
        w.ship.pos = Point::xyz(1.0, -1.0, 0.0);
        let aim = [-3.0, 2.0];
        w.tick(&Input {
            fire: true,
            aim,
            ..Input::default()
        });
        let b = w.shots[0];
        let s0 = w.ship.prev.e021();
        let t = AIM_DEPTH / SHOT_SPEED;
        let at = b.prev + b.vel.gp(t);
        let [x, y, s] = xyz(at);
        assert!((x - aim[0]).abs() < 1e-3 && (y - aim[1]).abs() < 1e-3);
        let ahead = s - (s0 + w.ship.speed * t);
        assert!((ahead - (AIM_DEPTH + 0.8)).abs() < 0.2, "{ahead}");
    }

    /// A bot that holds still and fires down the middle survives a while, kills things, and
    /// the run stays sane.
    #[test]
    fn a_run_plays() {
        let mut w = World::new(3);
        w.ship.invulnerable = 1e9;
        let mut kills = 0;
        let mut warned = 0;
        for t in 0..120 * 60 {
            let a = t as f32 * 0.01;
            let target = w
                .enemies
                .iter()
                .filter(|e| e.pos.e021() > w.ship.s() + 5.0)
                .min_by(|a, b| a.pos.e021().total_cmp(&b.pos.e021()))
                .map(|e| [e.pos.e032(), e.pos.e013()])
                .unwrap_or([0.0, 0.0]);
            w.tick(&Input {
                movement: [a.cos() * 0.5, (a * 0.7).sin() * 0.5],
                aim: target,
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
            let [x, y] = w.ship.xy();
            assert!((x * x + y * y).sqrt() <= SHIP_RADIUS + 1e-3);
        }
        assert!(warned > 30, "{warned} warnings");
        assert!(kills > 20, "{kills} kills");
        assert!(w.ship.s() > 60.0 * CRUISE * 0.9);
        assert!(w.score > 0);
    }
}
