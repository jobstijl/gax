//! The simulation: a pure, headless core with no Bevy and no wgpu types. It advances at a
//! fixed 120 Hz from a seed and the players' inputs, so a run is reproducible, and reports what
//! happened as [`Event`]s for the renderer and the audio.
//!
//! All geometry is gax PGA2D: poses are unit motors, velocities twists, positions points,
//! directions points at infinity, the arena's walls lines. Steering turns directions by the
//! rotation between them, bounces reflect in the walls, and collisions are joins and signed
//! distances.
//!
//! The parts: the families' numbers (`kind`), the enemies (`enemy`), the gravity wells
//! (`wells`), the ship and its shards (`ship`), the shots (`shots`), the spawns (`director`).

pub mod body;
pub mod collide;
pub mod director;
pub mod enemy;
pub mod kind;
pub mod ledger;
pub mod replay;
pub mod rng;
pub mod ship;
pub mod shots;
pub mod wells;

pub use director::Pending;
pub use enemy::Enemy;
pub use kind::Kind;
pub use ledger::Ledger;
pub use ship::{Shard, Ship};
pub use shots::{Bullet, Wreck};
pub use wells::Well;

use collide::SpatialHash;
use director::Director;
use gax::pga2d::Point;
use rng::Rng;

/// A point or a direction.
pub type P = Point<(), f32>;

/// The fixed timestep.
pub const DT: f32 = 1.0 / 120.0;
/// The arena's walls: at `x = ±ARENA_X` and `y = ±ARENA_Y`.
pub const ARENA_X: f32 = 32.0;
pub const ARENA_Y: f32 = 18.0;
/// No spawn lands closer than this to the player.
pub const SAFE_RADIUS: f32 = 7.0;

/// The point `(x, y)`.
pub const fn at(x: f32, y: f32) -> P {
    Point::new(x, y, 1.0)
}

/// The direction `(x, y)`.
pub const fn dir(x: f32, y: f32) -> P {
    Point::new(x, y, 0.0)
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
            movement: P::zero(),
            aim: P::zero(),
            fire: false,
            bomb: false,
        }
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

impl Phase {
    /// A tick of a death's countdown: whether it has run out (time to respawn).
    pub fn count_down(&mut self) -> bool {
        match self {
            Phase::Dead(t) => {
                *t -= DT;
                *t <= 0.0
            }
            _ => false,
        }
    }
}

/// The whole simulation state.
pub struct World {
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
    /// The gravity wells, as they stood at the start of the last tick.
    pub wells: Vec<Well>,
    /// The score, multiplier, lives and bombs.
    pub ledger: Ledger,
    /// Where the run is.
    pub phase: Phase,
    /// The spawn director.
    pub director: Director,
    /// This tick's events.
    pub events: Vec<Event>,
    /// The worst `|m ~m - 1|` seen on any pose this run (the ship's, the enemies' and the
    /// serpents' segments').
    pub worst_drift: f32,
    next_id: u32,
    hash: SpatialHash,
    scratch: Vec<u32>,
}

impl World {
    /// A new run.
    pub fn new(seed: u64) -> World {
        World {
            rng: Rng::new(seed),
            tick: 0,
            time: 0.0,
            ship: Ship::new(),
            bullets: Vec::new(),
            enemies: Vec::new(),
            shards: Vec::new(),
            pending: Vec::new(),
            wells: Vec::new(),
            ledger: Ledger::new(),
            phase: Phase::Playing,
            director: Director::new(),
            events: Vec::new(),
            worst_drift: 0.0,
            next_id: 1,
            hash: SpatialHash::new(ARENA_X, ARENA_Y, 2.0),
            scratch: Vec::new(),
        }
    }

    /// Advance one tick.
    pub fn tick(&mut self, input: &Input) {
        self.events.clear();
        self.tick += 1;
        self.time += DT;
        self.find_wells();
        if self.phase == Phase::Over {
            // The run is over: the enemies wander on, the shards drift away.
            self.step_enemies();
            self.step_shards();
            return;
        }
        if self.phase.count_down() {
            self.respawn();
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
        self.worst_drift = self.worst_drift.max(self.drift());
    }

    /// End the run: no lives left, or ended from the pause menu. The shots and the warp-ins
    /// go; the enemies wander on.
    pub fn end(&mut self) {
        self.phase = Phase::Over;
        self.bullets.clear();
        self.pending.clear();
    }

    /// The worst `|m ~m - 1|` of the poses now.
    fn drift(&self) -> f32 {
        let enemies = self
            .enemies
            .iter()
            .flat_map(|e| std::iter::once(&e.body).chain(&e.chain));
        std::iter::once(&self.ship.body)
            .chain(enemies)
            .map(|b| body::drift(b.pose))
            .fold(0.0, f32::max)
    }

    /// A fresh id for a new enemy.
    fn next_id(&mut self) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_same_run() {
        let run = |seed| {
            let mut w = World::new(seed);
            let mut h = 0u64;
            for t in 0..120 * 60 {
                let a = t as f32 * 0.01;
                w.tick(&Input {
                    movement: P::direction_from_angle(a) * 0.8,
                    aim: P::direction_from_angle(a * 2.0),
                    fire: true,
                    bomb: t % 2000 == 1999,
                });
                h = h.wrapping_mul(31).wrapping_add(
                    w.ledger.score
                        ^ w.enemies.len() as u64
                        ^ u64::from(w.ship.body.pos().e20().to_bits()),
                );
            }
            (h, w.ledger.score)
        };
        assert_eq!(run(42), run(42));
        assert_ne!(run(42).0, run(43).0);
    }
}
