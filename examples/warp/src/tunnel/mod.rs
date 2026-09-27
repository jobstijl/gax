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

pub mod lattice;
pub mod replay;
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
/// A serpent's segments, and the seconds between them along its screw.
const SEGMENTS: usize = 14;
const SEGMENT_LAG: f32 = 0.11;
/// Gates in a slalom, their spacing along the track, and their radius.
const GATES: usize = 6;
const GATE_GAP: f32 = 17.0;
pub const GATE_RADIUS: f32 = 1.7;
/// A singularity's pull (per unit of strength) and how long it holds ahead of the ship.
const WELL_HOLD: f32 = 9.0;
/// A singularity bursts at this mass.
const WELL_BURST: f32 = 5.0;
/// A shot bent by more than this (radians) before it kills is a slingshot.
const SLINGSHOT: f32 = 0.3;
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
    /// A serpent's head (its body is its segments), swimming a helix down the tunnel.
    Serpent,
    /// A singularity: pulls, pinches the tunnel, bends shots, eats mines.
    Singularity,
}

impl Foe {
    /// The Plane family whose colour and sounds it borrows.
    pub fn kind(self) -> Kind {
        match self {
            Foe::Drone => Kind::Drifter,
            Foe::Mine => Kind::Splitter,
            Foe::Turret => Kind::Warden,
            Foe::Serpent => Kind::Serpent,
            Foe::Singularity => Kind::Singularity,
        }
    }

    /// Its size (the radius of its hit sphere).
    pub fn radius(self) -> f32 {
        match self {
            Foe::Drone => 0.75,
            Foe::Mine => 0.9,
            Foe::Turret => 1.0,
            Foe::Serpent => 0.9,
            Foe::Singularity => 1.1,
        }
    }

    fn hp(self) -> f32 {
        match self {
            Foe::Drone => 1.0,
            Foe::Mine => 2.0,
            Foe::Turret => 4.0,
            Foe::Serpent => 8.0,
            Foe::Singularity => 24.0,
        }
    }

    fn points(self) -> u64 {
        match self {
            Foe::Drone => 100,
            Foe::Mine => 150,
            Foe::Turret => 400,
            Foe::Serpent => 800,
            Foe::Singularity => 2000,
        }
    }
}

/// What a drone (or a singularity) is doing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Flight {
    /// Coming in to its place ahead of the ship.
    Approach,
    /// Holding its place (drones circle with the ring).
    Hold,
    /// Drones: about to dive (a warning flash), then diving at the ship. Singularities: let
    /// go, staying where they are in the track while the ship passes.
    Dive,
}

/// A serpent's body segment.
#[derive(Clone, Copy, Debug)]
pub struct Segment {
    /// Where.
    pub pos: P,
    /// Where last tick.
    pub prev: P,
    /// Seconds behind the head along the screw.
    pub lag: f32,
    /// Hit points.
    pub hp: f32,
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
    /// A drone's (or singularity's) flight.
    pub flight: Flight,
    /// A singularity's mass: what it has eaten.
    pub mass: f32,
    /// A serpent's screw (its twist: `exp(t T)` is `t` seconds of swimming) and its body.
    pub twist: Line<(), f32>,
    pub body: Vec<Segment>,
    /// The singularity a drone escorts (its ring turns about the singularity).
    pub anchor: Option<u32>,
}

impl Enemy {
    /// Its hit radius: a singularity grows as it eats.
    pub fn radius(&self) -> f32 {
        self.foe.radius() + 0.12 * self.mass
    }

    /// A singularity's pull strength.
    pub fn strength(&self) -> f32 {
        1.0 + 0.35 * self.mass
    }
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
    /// The velocity it left with (a singularity bends it away from that).
    pub from: P,
    /// Seconds left.
    pub life: f32,
}

impl Shot {
    fn new(pos: P, vel: P, life: f32) -> Shot {
        Shot {
            pos,
            prev: pos,
            vel,
            from: vel,
            life,
        }
    }

    /// How far it has been bent: the angle of the rotation from the velocity it left with to
    /// the one it has (twice its logarithm's norm).
    pub fn bend(&self) -> f32 {
        2.0 * Motor::rotation_between(self.from, self.vel).log().norm()
    }
}

/// What became of a gate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GateState {
    /// Waiting.
    Open,
    /// Flown through.
    Passed,
    /// Flown past.
    Missed,
}

/// A bonus gate: a ring across the tunnel.
#[derive(Clone, Copy, Debug)]
pub struct Gate {
    /// Its centre (its arc length is where its plane is).
    pub pos: P,
    /// What became of it.
    pub state: GateState,
    /// Seconds since it was passed or missed.
    pub t: f32,
}

impl Gate {
    /// Its plane: the cross-section at its arc length, `z = s`.
    pub fn plane(&self) -> Plane<(), f32> {
        Plane::from_normal([0.0, 0.0, 1.0], arc(self.pos))
    }
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
    /// Flown through a gate: the chain it extends, and the points.
    Gate { pos: P, chain: u32, points: u64 },
    /// Flown past a gate: the chain is broken.
    GateMiss { pos: P },
    /// A singularity ate something.
    Absorb { pos: P, mass: f32 },
    /// An overfed singularity burst.
    Burst { pos: P },
    /// A shot bent round a singularity killed: double points.
    Slingshot { pos: P },
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
    /// Drones: escorting the singularity that lands with them.
    pub escort: bool,
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
    /// Bonus gates.
    pub gates: Vec<Gate>,
    /// Gates flown through in a row.
    pub chain: u32,
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
    gate_timer: f32,
    well_cooldown: f32,
    /// The kinds introduced so far (bits: serpent, singularity).
    introduced: u8,
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

/// How strongly a singularity pulls the wall in (a lattice source per unit of strength), and
/// how far along the tunnel (the squared reach).
pub const PINCH: f32 = 480.0;
pub const PINCH_REACH2: f32 = 49.0;

/// The singularities for a tick: where each is and how strong.
pub struct Field {
    wells: Vec<(P, f32)>,
}

impl Field {
    /// The pull at `p`, `k s d / (|d|² + 1)^(3/2)` towards each singularity: an inverse
    /// square with a softened core.
    pub fn pull(&self, p: P, k: f32) -> P {
        let mut acc = dir(0.0, 0.0, 0.0);
        for &(w, strength) in &self.wells {
            let d = w - p;
            let soft = d.ideal_norm() * d.ideal_norm() + 1.0;
            acc += d * (k * strength / soft.powf(1.5));
        }
        acc
    }

    /// Whether there are any.
    pub fn is_empty(&self) -> bool {
        self.wells.is_empty()
    }
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
            gates: Vec::new(),
            chain: 0,
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
            gate_timer: 12.0,
            well_cooldown: 0.0,
            introduced: 0,
            next_id: 1,
            next_life: 100_000,
            next_bomb: 150_000,
        }
    }

    /// The speed bonus on scores: faster is worth more.
    pub fn speed_bonus(&self) -> f32 {
        (self.ship.speed / CRUISE).max(1.0)
    }

    /// The singularities' field this tick.
    pub fn field(&self) -> Field {
        Field {
            wells: self.wells().collect(),
        }
    }

    /// The singularities: where, and how strong.
    pub fn wells(&self) -> impl Iterator<Item = (P, f32)> + '_ {
        self.enemies
            .iter()
            .filter(|e| e.foe == Foe::Singularity)
            .map(|e| (e.pos, e.strength()))
    }

    /// How far from the axis the ship may fly at arc length `s`: inside the wall, which a
    /// singularity pinches in.
    pub fn ship_radius(&self, s: f32) -> f32 {
        SHIP_RADIUS.min(self.lattice.radius(s) - 1.3).max(1.0)
    }

    /// How near the ship is to a singularity, `0..1` (the music darkens and bends with it).
    pub fn darkness(&self) -> f32 {
        let ship = self.ship.pos;
        self.wells()
            .map(|(w, strength)| (1.0 - (w & ship).norm() / (22.0 + 8.0 * strength)).max(0.0))
            .fold(0.0, f32::max)
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
        let field = self.field();
        self.step_enemies(&field);
        self.step_gates();
        self.step_shots(&field);
        self.step_bolts(&field);
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
        // A singularity pulls the ship across the tunnel (not along it), capped so that it can
        // always be flown out of; and where it pinches the wall, the ship must stay further in.
        let pull = self.field().pull(self.ship.pos, 30.0);
        let pull = dir(pull.e032(), pull.e013(), 0.0);
        let n = pull.ideal_norm();
        let pull = if n > 7.0 { pull * (7.0 / n) } else { pull };
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
        if r > limit {
            let f = foot(pos);
            pos = f + (pos - f) * (limit / r);
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
        self.shots.push(Shot::new(muzzle, vel, 1.6));
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
                self.kill(i, false, false);
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
        self.blasts.push((
            Source {
                pos,
                strength,
                r2,
                reach2: 0.0,
            },
            time,
        ));
    }

    /// The director: an intensity curve over the run, formations ahead in the fog, and now
    /// and then a slalom of gates.
    fn direct(&mut self) {
        self.clock += DT;
        self.intensity = crate::sim::director::Director::curve(self.clock * 1.3);
        self.cooldown -= DT;
        self.gate_timer -= DT;
        self.well_cooldown -= DT;
        let has_well = self.enemies.iter().any(|e| e.foe == Foe::Singularity)
            || self.pending.iter().any(|p| p.foe == Foe::Singularity);
        if self.gate_timer <= 0.0 && !has_well {
            if self.gates.iter().all(|g| g.state != GateState::Open) {
                self.slalom();
            }
            self.gate_timer = self.rng.range(22.0, 30.0);
        }
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
            escort: false,
        };
        // Bands of one roll: each kind once it has been introduced, drones otherwise.
        // Each kind is introduced by a deadline, whatever the rolls: serpents by 40 s, the first
        // singularity by 60 s.
        let late_serpent = self.introduced & 1 == 0 && self.clock > 40.0;
        let late_well = self.introduced & 2 == 0 && self.clock > 60.0;
        let well_band = roll < 0.16 || late_well;
        let turret_band = (0.16..0.36 + 0.12 * i).contains(&roll) && !late_serpent;
        let serpent_band = (0.48..0.64).contains(&roll) || late_serpent;
        let mine_band = (0.64..0.82).contains(&roll);
        if well_band && self.clock > 40.0 && !has_well && self.well_cooldown <= 0.0 {
            // A singularity, holding ahead, with a ring of drones orbiting it.
            let a0 = r.angle();
            let depth = r.range(24.0, 30.0);
            let off = r.range(1.0, 2.0);
            spawn.push(pend(Foe::Singularity, around(off, a0, s), a0, off, depth));
            let n = 6 + (i * 4.0) as usize;
            for k in 0..n {
                let a = k as f32 * core::f32::consts::TAU / n as f32;
                let mut p = pend(
                    Foe::Drone,
                    around(off, a0, s) + (around(3.4, a, 0.0) - at(0.0, 0.0, 0.0)),
                    a,
                    3.4,
                    depth,
                );
                p.escort = true;
                spawn.push(p);
            }
            self.well_cooldown = 45.0;
            self.introduced |= 2;
        } else if turret_band && self.clock > 25.0 {
            // Turrets on the wall, spaced along the track and around it.
            let n = 1 + (i * 3.0) as usize;
            let a0 = r.angle();
            for k in 0..n {
                let a = a0 + k as f32 * 2.1;
                let p = around(RADIUS - 0.5, a, s + k as f32 * 7.0);
                spawn.push(pend(Foe::Turret, p, a, 0.0, 0.0));
            }
        } else if serpent_band && self.clock > 30.0 {
            // A serpent, coming out of the fog on its helix.
            let a = r.angle();
            let ring = r.range(3.2, 4.4);
            spawn.push(pend(Foe::Serpent, around(ring, a, s), a, ring, 0.0));
            self.introduced |= 1;
        } else if (turret_band || mine_band) && self.clock > 10.0 {
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

    /// A slalom of gates ahead, on a screw motion about the axis: each gate is the last one
    /// turned about the axis and moved along it by the same motor.
    fn slalom(&mut self) {
        let i = self.intensity;
        let r = &mut self.rng;
        let s0 = self.ship.s() + SPAWN_AHEAD * 0.8;
        let off = r.range(1.4, 2.4 + 1.2 * i);
        let side = if r.unit() < 0.5 { -1.0 } else { 1.0 };
        let turn = side * r.range(0.5, 0.9 + 0.5 * i);
        let screw = about_axis(turn) * Motor::translation(0.0, 0.0, GATE_GAP);
        let mut c = around(off, r.angle(), s0);
        self.gates.retain(|g| g.state == GateState::Open);
        for _ in 0..GATES {
            self.gates.push(Gate {
                pos: c,
                state: GateState::Open,
                t: 0.0,
            });
            c = screw >> c;
        }
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
        // A singularity lands before its escorts (it was announced first).
        let mut well = None;
        for p in landed {
            let id = self.next_id;
            self.next_id += 1;
            let drift = if p.foe == Foe::Mine {
                about_axis(self.rng.angle()) >> dir(0.7, 0.0, 0.0)
            } else {
                dir(0.0, 0.0, 0.0)
            };
            let (twist, body) = if p.foe == Foe::Serpent {
                self.serpent(p.pos)
            } else {
                (Line::translation_twist(0.0, 0.0, 0.0), Vec::new())
            };
            if p.foe == Foe::Singularity {
                well = Some(id);
            }
            let timer = if p.foe == Foe::Singularity {
                WELL_HOLD
            } else {
                self.rng.range(0.6, 1.8)
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
                timer,
                flight: Flight::Approach,
                mass: 0.0,
                twist,
                body,
                anchor: if p.escort { well } else { None },
            });
        }
    }

    /// A serpent's screw (a turn about the axis and a swim along it, per second) and its body
    /// behind a head at `head`: segment `k` is the head moved back along the screw by
    /// `k SEGMENT_LAG` seconds, `exp(-lag T)`.
    fn serpent(&mut self, head: P) -> (Line<(), f32>, Vec<Segment>) {
        let side = if self.rng.unit() < 0.5 { -1.0 } else { 1.0 };
        let twist = Line::rotation_twist(axis(), side * self.rng.range(1.3, 1.8))
            + Line::translation_twist(0.0, 0.0, 0.55 * CRUISE);
        let body = (1..=SEGMENTS)
            .map(|k| {
                let lag = k as f32 * SEGMENT_LAG;
                let pos = (twist * -lag).exp() >> head;
                Segment {
                    pos,
                    prev: pos,
                    lag,
                    hp: 2.0,
                }
            })
            .collect();
        (twist, body)
    }

    fn step_enemies(&mut self, field: &Field) {
        let ship = self.ship.pos;
        let (s_ship, speed) = (self.ship.s(), self.ship.speed);
        let alive = self.phase == Phase::Playing;
        let mut fired = Vec::new();
        // One drone dives at a time: the one that has held longest (escorts stay with their
        // singularity).
        self.dive_cooldown -= DT;
        if alive && self.dive_cooldown <= 0.0 {
            let holding = self
                .enemies
                .iter_mut()
                .filter(|e| e.foe == Foe::Drone && e.flight == Flight::Hold && e.anchor.is_none())
                .max_by(|a, b| a.age.total_cmp(&b.age));
            if let Some(e) = holding {
                e.flight = Flight::Dive;
                e.timer = 0.7;
                self.events.push(Event::Dive { pos: e.pos });
                self.dive_cooldown = 1.6 - 0.8 * self.intensity;
            }
        }
        // The singularities, where they are and how they move (their escorts' rings move with
        // them).
        let wells: Vec<(u32, P, P)> = self
            .enemies
            .iter()
            .filter(|e| e.foe == Foe::Singularity)
            .map(|e| (e.id, e.pos, e.vel))
            .collect();
        let with_ship = dir(0.0, 0.0, speed);
        // Towards `place`, moving along with the ship; capped so that it glides in.
        let glide = |from: P, place: P, cap: f32| {
            let to = (place - from) * 2.5;
            let n = to.ideal_norm();
            if n > cap { to * (cap / n) } else { to }
        };
        for e in &mut self.enemies {
            e.prev = e.pos;
            e.age += DT;
            e.flash = (e.flash - DT * 6.0).max(0.0);
            let ahead = arc(e.pos) - s_ship;
            match e.foe {
                Foe::Drone => {
                    e.angle += DT * 0.9;
                    // An escort's ring turns about its singularity; a lost escort joins the
                    // ship's band.
                    let well = e.anchor.and_then(|id| wells.iter().find(|w| w.0 == id));
                    if e.anchor.is_some() && well.is_none() {
                        e.anchor = None;
                        e.ring = 4.6;
                        e.depth = 8.0;
                        e.flight = Flight::Approach;
                    }
                    let (place, place_vel) = match well {
                        Some(&(_, w, v)) => {
                            (w + (around(e.ring, e.angle, 0.0) - at(0.0, 0.0, 0.0)), v)
                        }
                        // Its place in the ring, which turns and holds `depth` ahead of the ship.
                        None => (around(e.ring, e.angle, s_ship + e.depth), with_ship),
                    };
                    e.vel = match e.flight {
                        Flight::Approach | Flight::Hold => {
                            if e.flight == Flight::Approach && (place & e.pos).norm() < 1.5 {
                                e.flight = Flight::Hold;
                            }
                            glide(e.pos, place, 26.0) + place_vel
                        }
                        Flight::Dive => {
                            e.timer -= DT;
                            if e.timer > 0.0 {
                                // The warning: it flashes in place.
                                e.flash = e.flash.max(0.6);
                                (place - e.pos) * 2.5 + place_vel
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
                    // tangent to it there. Singularities pull.
                    let outward = off_axis(e.pos + e.vel * DT) > off_axis(e.pos);
                    if off_axis(e.pos) > SHIP_RADIUS + 0.3 && outward {
                        let tangent = Plane::orthogonal_to(e.pos - foot(e.pos));
                        e.vel = tangent.reflect(e.vel);
                    }
                    e.vel += field.pull(e.pos, 20.0) * DT;
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
                Foe::Serpent => {
                    // Swim: the head along its screw, and each segment the head moved back by
                    // its own lag.
                    let next = (e.twist * DT).exp() >> e.pos;
                    e.vel = (next - e.pos) * (1.0 / DT);
                    for seg in &mut e.body {
                        seg.prev = seg.pos;
                        seg.pos = (e.twist * -seg.lag).exp() >> next;
                    }
                }
                Foe::Singularity => {
                    // Holding ahead of the ship, turning slowly about the axis; then let go, so
                    // that the ship must fly past it.
                    e.angle += DT * 0.3;
                    let place = around(e.ring, e.angle, s_ship + e.depth);
                    e.vel = match e.flight {
                        Flight::Approach => {
                            if (place & e.pos).norm() < 1.5 {
                                e.flight = Flight::Hold;
                            }
                            glide(e.pos, place, 26.0) + with_ship
                        }
                        Flight::Hold => {
                            e.timer -= DT;
                            if e.timer <= 0.0 {
                                e.flight = Flight::Dive;
                            }
                            glide(e.pos, place, 26.0) + with_ship
                        }
                        Flight::Dive => dir(0.0, 0.0, 0.0),
                    };
                }
            }
            e.pos += e.vel * DT;
        }
        for (pos, vel) in fired {
            self.bolts.push(Shot::new(pos, vel, 8.0));
            self.events.push(Event::Bolt { pos });
        }
        self.feed();
        // Behind the ship: gone.
        self.enemies.retain(|e| arc(e.pos) > s_ship - 6.0);
    }

    /// Singularities eat the mines and loose drones that fall in, and burst when overfed.
    fn feed(&mut self) {
        let mut eaten: Vec<(usize, usize)> = Vec::new();
        for (wi, w) in self.enemies.iter().enumerate() {
            if w.foe != Foe::Singularity {
                continue;
            }
            for (i, e) in self.enemies.iter().enumerate() {
                let food = e.foe == Foe::Mine || (e.foe == Foe::Drone && e.anchor.is_none());
                if food && e.age > 0.3 && (e.pos & w.pos).norm() < w.radius() {
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
            w.hp += 3.0;
            w.flash = 1.0;
            let (pos, mass) = (w.pos, w.mass);
            self.events.push(Event::Absorb { pos, mass });
        }
        gone.sort_unstable();
        for &i in gone.iter().rev() {
            self.enemies.swap_remove(i);
        }
        let Some(i) = self
            .enemies
            .iter()
            .position(|e| e.foe == Foe::Singularity && e.mass >= WELL_BURST)
        else {
            return;
        };
        // Overfed: it bursts into a ring of drones.
        let w = self.enemies.swap_remove(i);
        self.events.push(Event::Burst { pos: w.pos });
        self.blast(w.pos, 1800.0, 6.0, 0.25);
        for k in 0..8 {
            let a = k as f32 * core::f32::consts::TAU / 8.0;
            let pos = w.pos + (around(1.0, a, 0.0) - at(0.0, 0.0, 0.0));
            let id = self.next_id;
            self.next_id += 1;
            self.enemies.push(Enemy {
                id,
                foe: Foe::Drone,
                pos,
                prev: pos,
                vel: dir(0.0, 0.0, 0.0),
                hp: 1.0,
                age: 0.0,
                flash: 1.0,
                angle: a,
                ring: 4.8,
                depth: 9.0,
                timer: 1.0,
                flight: Flight::Approach,
                mass: 0.0,
                twist: Line::translation_twist(0.0, 0.0, 0.0),
                body: Vec::new(),
                anchor: None,
            });
        }
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
            let meet = ((a & b) ^ g.plane()).unitized();
            let through = alive && (meet & g.pos).norm() < GATE_RADIUS;
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
                self.mult += 1;
                let points = (250.0 * self.chain as f32 * bonus).round() as u64;
                self.score += points;
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

    fn step_shots(&mut self, field: &Field) {
        // Hits: (shot, enemy, segment of a serpent's body or its head).
        let mut hits: Vec<(usize, usize, Option<usize>, bool)> = Vec::new();
        let mut walls = Vec::new();
        let lattice = &self.lattice;
        for (k, b) in self.shots.iter_mut().enumerate() {
            b.prev = b.pos;
            if !field.is_empty() {
                b.vel += field.pull(b.pos, 900.0) * DT;
            }
            b.pos += b.vel * DT;
            b.life -= DT;
            let bent = b.bend() > SLINGSHOT;
            let hit = self.enemies.iter().enumerate().find_map(|(i, e)| {
                if segment_hits_sphere(b.prev, b.pos, e.pos, e.radius() + HIT_MARGIN) {
                    return Some((i, None));
                }
                e.body
                    .iter()
                    .position(|seg| segment_hits_sphere(b.prev, b.pos, seg.pos, 0.6 + HIT_MARGIN))
                    .map(|j| (i, Some(j)))
            });
            if let Some((i, seg)) = hit {
                hits.push((k, i, seg, bent));
                b.life = -1.0;
            } else if off_axis(b.pos) > lattice.radius(arc(b.pos)) {
                walls.push(b.pos);
                b.life = -1.0;
            }
        }
        for pos in walls {
            self.events.push(Event::Wall { pos });
        }
        self.shots.retain(|b| b.life > 0.0);
        let mut dead: Vec<(usize, bool)> = Vec::new();
        let mut segments: Vec<(usize, usize)> = Vec::new();
        for (_, i, seg, bent) in hits {
            let e = &mut self.enemies[i];
            match seg {
                Some(j) => {
                    let s = &mut e.body[j];
                    s.hp -= 1.0;
                    if s.hp <= 0.0 && !segments.contains(&(i, j)) {
                        segments.push((i, j));
                    } else {
                        self.events.push(Event::Hit {
                            pos: s.pos,
                            foe: e.foe,
                        });
                    }
                }
                None => {
                    e.hp -= if bent { 2.0 } else { 1.0 };
                    e.flash = 1.0;
                    if e.hp <= 0.0 {
                        if !dead.iter().any(|d| d.0 == i) {
                            dead.push((i, bent));
                        }
                    } else {
                        self.events.push(Event::Hit {
                            pos: e.pos,
                            foe: e.foe,
                        });
                    }
                }
            }
        }
        // Body segments shot away (highest first, so the indices hold).
        segments.sort_unstable();
        for &(i, j) in segments.iter().rev() {
            let s = self.enemies[i].body.remove(j);
            let points = (50.0 * self.mult as f32 * self.speed_bonus()).round() as u64;
            self.score += points;
            self.events.push(Event::Kill {
                pos: s.pos,
                foe: Foe::Serpent,
                points,
            });
            self.blast(s.pos, 160.0, 3.0, 0.08);
        }
        dead.sort_unstable();
        for (i, bent) in dead.into_iter().rev() {
            self.kill(i, true, bent);
        }
    }

    fn kill(&mut self, i: usize, scored: bool, bent: bool) {
        let e = self.enemies.swap_remove(i);
        let sling = if bent { 2.0 } else { 1.0 };
        let points = if scored {
            (e.foe.points() as f32 * (1.0 + e.mass) * self.mult as f32 * self.speed_bonus() * sling)
                .round() as u64
        } else {
            0
        };
        self.score += points;
        self.events.push(Event::Kill {
            pos: e.pos,
            foe: e.foe,
            points,
        });
        if bent && scored {
            self.events.push(Event::Slingshot { pos: e.pos });
        }
        let size = match e.foe {
            Foe::Turret | Foe::Serpent => 700.0,
            Foe::Singularity => 2600.0,
            _ => 260.0,
        };
        self.blast(e.pos, size, 3.0, 0.12);
        match e.foe {
            // The head gone, the body goes up segment by segment.
            Foe::Serpent => {
                for s in &e.body {
                    self.events.push(Event::Kill {
                        pos: s.pos,
                        foe: Foe::Serpent,
                        points: 0,
                    });
                    self.blast(s.pos, 200.0, 3.0, 0.1);
                }
            }
            // The collapse: the wall let go all round, a ring of shock along the tunnel.
            Foe::Singularity => {
                let s = arc(e.pos);
                for k in 0..12 {
                    let a = k as f32 * core::f32::consts::TAU / 12.0;
                    self.blast(around(RADIUS - 1.0, a, s), 1400.0, 6.0, 0.3);
                }
            }
            _ => {}
        }
        if scored {
            let n = match e.foe {
                Foe::Serpent => 3,
                Foe::Singularity => 4 + e.mass as usize,
                _ => 1,
            };
            for k in 0..n {
                let off = if n > 1 {
                    around(1.2, k as f32 * core::f32::consts::TAU / n as f32, 0.0)
                        - at(0.0, 0.0, 0.0)
                } else {
                    dir(0.0, 0.0, 0.0)
                };
                self.shards.push(Shard {
                    pos: e.pos + off,
                    life: 6.0,
                });
            }
        }
    }

    fn step_bolts(&mut self, field: &Field) {
        let s = self.ship.s();
        for b in &mut self.bolts {
            b.prev = b.pos;
            if !field.is_empty() {
                b.vel += field.pull(b.pos, 300.0) * DT;
            }
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
        let hit_enemy = self.enemies.iter().any(|e| {
            segment_hits_sphere(prev, p, e.pos, e.radius() + SHIP_SIZE)
                || e.body
                    .iter()
                    .any(|s| segment_hits_sphere(prev, p, s.pos, 0.6 + SHIP_SIZE))
        });
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
        self.chain = 0;
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
        let mut sources: Vec<Source> = Vec::with_capacity(self.blasts.len() + 2);
        self.blasts.retain_mut(|(src, t)| {
            sources.push(*src);
            *t -= DT;
            *t > 0.0
        });
        // Singularities pinch the wall in around them.
        for (pos, strength) in self.wells() {
            sources.push(Source {
                pos,
                strength: -PINCH * strength,
                r2: 8.0,
                reach2: PINCH_REACH2,
            });
        }
        // The ship's wake: a gentle push on the wall nearest to it.
        if self.phase == Phase::Playing {
            sources.push(Source {
                pos: self.ship.pos + dir(0.0, 0.0, 1.0),
                strength: 6.0 * self.ship.speed / CRUISE,
                r2: 1.5,
                reach2: 0.0,
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
            mass: 0.0,
            twist: Line::translation_twist(0.0, 0.0, 0.0),
            body: Vec::new(),
            anchor: None,
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

    fn enemy(w: &mut World, foe: Foe, pos: P) -> usize {
        let id = w.next_id;
        w.next_id += 1;
        let (twist, body) = if foe == Foe::Serpent {
            w.serpent(pos)
        } else {
            (Line::translation_twist(0.0, 0.0, 0.0), Vec::new())
        };
        w.enemies.push(Enemy {
            id,
            foe,
            pos,
            prev: pos,
            vel: dir(0.0, 0.0, 0.0),
            hp: foe.hp(),
            age: 1.0,
            flash: 0.0,
            angle: 0.0,
            ring: off_axis(pos),
            depth: arc(pos) - w.ship.s(),
            timer: WELL_HOLD,
            flight: Flight::Hold,
            mass: 0.0,
            twist,
            body,
            anchor: None,
        });
        w.enemies.len() - 1
    }

    /// A serpent's body is the helix its head has swum: every segment stays at the head's
    /// distance from the axis, behind it along the track, and where the head was `lag`
    /// seconds ago.
    #[test]
    fn a_serpent_swims_a_screw() {
        let mut w = World::new(2);
        w.cooldown = 1e9;
        w.gate_timer = 1e9;
        let i = enemy(&mut w, Foe::Serpent, around(4.0, 0.3, 60.0));
        let mut trail = vec![w.enemies[i].pos];
        for _ in 0..240 {
            w.tick(&Input::default());
            trail.push(w.enemies[0].pos);
        }
        let e = &w.enemies[0];
        assert_eq!(e.body.len(), SEGMENTS);
        for seg in &e.body {
            assert!(
                (off_axis(seg.pos) - 4.0).abs() < 1e-2,
                "{}",
                off_axis(seg.pos)
            );
            assert!(arc(seg.pos) < arc(e.pos));
            // Where the head was `lag` seconds ago.
            let ticks = (seg.lag / DT).round() as usize;
            let then = trail[trail.len() - 1 - ticks];
            assert!(
                (then & seg.pos).norm() < 0.05,
                "{}",
                (then & seg.pos).norm()
            );
        }
    }

    /// Through a gate: the chain grows and scores; past one: the chain breaks.
    #[test]
    fn gates_are_flown_through_or_missed() {
        let mut w = World::new(4);
        w.cooldown = 1e9;
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
        assert_eq!(w.mult, 3);
    }

    /// A singularity pinches the wall in around it, so the ship must fly further in there;
    /// it bends shots that pass it; it eats mines that fall in, and bursts when overfed.
    #[test]
    fn a_singularity_pinches_bends_eats_and_bursts() {
        let mut w = World::new(6);
        w.cooldown = 1e9;
        w.gate_timer = 1e9;
        w.ship.invulnerable = 1e9;
        let i = enemy(&mut w, Foe::Singularity, at(0.0, 0.0, 30.0));
        // Let go: it stays where it is in the track while the ship comes.
        w.enemies[i].flight = Flight::Dive;
        for _ in 0..120 {
            w.tick(&Input::default());
        }
        let well = w.enemies[0].pos;
        let pinched = w.lattice.radius(arc(well));
        assert!(pinched < RADIUS - 1.5, "{pinched}");
        assert!(w.ship_radius(arc(well)) < SHIP_RADIUS - 1.0);
        // A shot fired past it bends towards it.
        let from = at(2.0, 0.0, arc(well) - 12.0);
        let mut shot = Shot::new(from, dir(0.0, 0.0, 70.0), 1.0);
        let field = w.field();
        for _ in 0..40 {
            shot.vel += field.pull(shot.pos, 900.0) * DT;
            shot.pos += shot.vel * DT;
        }
        assert!(shot.bend() > 0.1, "{}", shot.bend());
        assert!(
            shot.pos.e032() / shot.pos.e123() < 2.0,
            "bent the wrong way"
        );
        // Mines dropped into it are eaten, and it bursts.
        let mut absorbed = 0;
        let mut burst = false;
        for k in 0..6 {
            let w0 = w
                .enemies
                .iter()
                .find(|e| e.foe == Foe::Singularity)
                .map(|e| e.pos);
            let Some(p) = w0 else { break };
            let m = enemy(&mut w, Foe::Mine, p + dir(0.3 * k as f32 * 0.1, 0.2, 0.0));
            w.enemies[m].age = 1.0;
            w.tick(&Input::default());
            for e in &w.events {
                match e {
                    Event::Absorb { .. } => absorbed += 1,
                    Event::Burst { .. } => burst = true,
                    _ => {}
                }
            }
        }
        assert!(absorbed >= 5 && burst, "{absorbed} {burst}");
        assert!(w.enemies.iter().all(|e| e.foe != Foe::Singularity));
        assert!(w.enemies.iter().filter(|e| e.foe == Foe::Drone).count() >= 8);
    }

    /// Over a long run the director brings everything: drones, mines, turrets, serpents, a
    /// singularity with its escort, and slaloms of gates.
    #[test]
    fn the_director_brings_everything() {
        let mut w = World::new(12);
        w.ship.invulnerable = 1e9;
        let mut seen = std::collections::HashSet::new();
        let mut escorts = false;
        let mut gates = false;
        let mut warned = std::collections::HashMap::new();
        for _ in 0..120 * 150 {
            // Shoot at the nearest enemy ahead, so that the tunnel keeps room for more.
            let s = w.ship.s();
            let aim = w
                .enemies
                .iter()
                .filter(|e| arc(e.pos) > s + 5.0)
                .min_by(|a, b| arc(a.pos).total_cmp(&arc(b.pos)))
                .map_or(at(0.0, 0.0, s + AIM_DEPTH), |e| e.pos);
            w.tick(&Input {
                aim,
                fire: true,
                ..Input::default()
            });
            for e in &w.events {
                if let Event::Warn { foe, .. } = e {
                    *warned.entry(format!("{foe:?}")).or_insert(0) += 1;
                }
            }
            for e in &w.enemies {
                seen.insert(format!("{:?}", e.foe));
                escorts |= e.anchor.is_some();
            }
            gates |= !w.gates.is_empty();
        }
        assert_eq!(seen.len(), 5, "{seen:?} {warned:?} {}", w.clock);
        assert!(escorts && gates);
    }
}
