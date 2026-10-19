//! What flies in the tunnel: the ship, its enemies, shots and bolts, gates, shards, spawns on
//! their way, and the events a tick reports.

use super::space::{AXIS, P, arc, segment_hits_sphere};
use super::wells::Wells;
use crate::sim::DT;
use crate::sim::Kind;
use gax::Unit;
use gax::pga3d::{Motor, Plane};

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

    /// Its hit points when it lands.
    fn hp(self) -> f32 {
        match self {
            Foe::Drone => 1.0,
            Foe::Mine => 2.0,
            Foe::Turret => 4.0,
            Foe::Serpent => 8.0,
            Foe::Singularity => 24.0,
        }
    }

    /// What a kill is worth, before the multipliers.
    pub(super) fn points(self) -> u64 {
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
    /// Its place behind the head: the head's screw run back by its lag, `exp(-lag T)`.
    pub behind: Unit<Motor<(), f32>>,
    /// Hit points.
    pub hp: f32,
}

impl Segment {
    /// The radius of a segment's hit sphere.
    pub const RADIUS: f32 = 0.6;
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
    /// The radius it circles at: a drone's ring (about the axis, or about the singularity it
    /// escorts) and a singularity's distance from the axis.
    pub orbit: f32,
    /// How far ahead of the ship the ring holds (drones).
    pub depth: f32,
    /// Seconds to the next action (turrets: firing, the last 0.35 s a charge-up; drones: the
    /// dive's warning).
    pub timer: f32,
    /// A drone's (or singularity's) flight.
    pub flight: Flight,
    /// A singularity's mass: what it has eaten.
    pub mass: f32,
    /// A serpent's swim over one tick: its screw `exp(DT T)`, which turns about the axis and
    /// moves along it.
    pub swim: Unit<Motor<(), f32>>,
    /// A serpent's body, behind its head on the screw.
    pub body: Vec<Segment>,
    /// The singularity a drone escorts (its ring turns about the singularity).
    pub anchor: Option<u32>,
}

impl Enemy {
    /// A new enemy at `pos`: at rest, approaching, with its kind's hit points; whoever spawns
    /// it sets the rest.
    pub fn new(id: u32, foe: Foe, pos: P) -> Enemy {
        Enemy {
            id,
            foe,
            pos,
            prev: pos,
            vel: P::zero(),
            hp: foe.hp(),
            age: 0.0,
            flash: 0.0,
            angle: 0.0,
            orbit: 0.0,
            depth: 0.0,
            timer: 0.0,
            flight: Flight::Approach,
            mass: 0.0,
            swim: Unit::identity(),
            body: Vec::new(),
            anchor: None,
        }
    }

    /// Its hit radius: a singularity grows as it eats.
    pub fn radius(&self) -> f32 {
        self.foe.radius() + 0.12 * self.mass
    }

    /// A singularity's pull strength.
    pub fn strength(&self) -> f32 {
        1.0 + 0.35 * self.mass
    }

    /// What the segment `a → b` passes within `margin` of (beyond the hit spheres): the enemy
    /// itself (`Some(None)`), segment `j` of a serpent's body (`Some(Some(j))`), or nothing.
    pub fn swept(&self, a: P, b: P, margin: f32) -> Option<Option<usize>> {
        if segment_hits_sphere(a, b, self.pos, self.radius() + margin) {
            return Some(None);
        }
        let r = Segment::RADIUS + margin;
        let seg = self
            .body
            .iter()
            .position(|s| segment_hits_sphere(a, b, s.pos, r));
        seg.map(Some)
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
    /// A shot leaving `pos` at `vel`, for `life` seconds.
    pub fn new(pos: P, vel: P, life: f32) -> Shot {
        Shot {
            pos,
            prev: pos,
            vel,
            from: vel,
            life,
        }
    }

    /// One tick of flight, pulled by the singularities `wells` (as strongly as `k` says).
    pub fn advance(&mut self, wells: &Wells, k: f32) {
        self.prev = self.pos;
        if !wells.is_empty() {
            self.vel += wells.pull(self.pos, k) * DT;
        }
        self.pos += self.vel * DT;
        self.life -= DT;
    }

    /// How far it has been bent: the angle between the velocity it left with and the one it
    /// has.
    pub fn bend(&self) -> f32 {
        self.from.angle_between(self.vel)
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
    /// Its plane: the cross-section through its centre, orthogonal to the axis.
    pub fn plane(&self) -> Plane<(), f32> {
        self.pos | AXIS
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
    /// Which barrel fired last (the next shot leaves from the other).
    pub barrel: bool,
}

impl Ship {
    /// The arc length it is at.
    pub fn s(&self) -> f32 {
        arc(self.pos)
    }
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
    /// The radius it will circle at (see [`Enemy::orbit`]).
    pub orbit: f32,
    /// The ring's depth ahead of the ship (drones).
    pub depth: f32,
    /// Drones: escorting the singularity that lands with them.
    pub escort: bool,
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
