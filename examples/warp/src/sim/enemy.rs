//! The enemies: landing, moving by their family's rule, bouncing off the walls, and dying.

use super::body::{Body, interpolate, pose_at, turned};
use super::collide::outside;
use super::kind::{Kind, SERPENT_GAP, SERPENT_LEN};
use super::shots::Wreck;
use super::wells::{Well, pull};
use super::{DT, Event, P, Phase, World, dir};
use crate::geom::{ORIGIN, clamp_length, orbit, with_length};
use gax::pga2d::{Motor, intrinsic};

/// An enemy.
#[derive(Clone, Debug)]
pub struct Enemy {
    /// Stable id.
    pub id: u32,
    /// Family.
    pub kind: Kind,
    /// Motion: its own intent plus the pull, as of its last step.
    pub body: Body,
    /// Its own velocity: the way it means to go.
    pub intent: P,
    /// Velocity from the wells' pull, on top of its intent; decays slowly.
    pub pull: P,
    /// Killed or eaten this tick: removed once the enemies have moved and fed.
    pub dead: bool,
    /// Hits left.
    pub hp: f32,
    /// Collision radius.
    pub radius: f32,
    /// Mass eaten (singularities; zero for the others).
    pub mass: f32,
    /// Seconds alive.
    pub age: f32,
    /// Flash after a hit, decaying.
    pub flash: f32,
    /// A per-enemy phase for wandering and animation.
    pub phase: f32,
    /// A serpent's body, head first (empty for others).
    pub chain: Vec<Body>,
    /// A timer for periodic actions (a carrier's launches).
    pub timer: f32,
}

/// A shot in flight as an evader sees it.
struct Shot {
    /// Its line of flight.
    line: intrinsic::Line<(), f32>,
    /// Where it is.
    pos: P,
    /// Its velocity.
    vel: P,
}

/// What an enemy sees of the world in a tick.
struct View<'a> {
    /// The ship's position.
    ship: P,
    /// Whether the ship is there to hunt.
    hunting: bool,
    /// Seconds since the start.
    time: f32,
    /// The shots in flight.
    shots: &'a [Shot],
    /// The wells.
    wells: &'a [Well],
}

impl World {
    /// Spawn an enemy now, at `at` facing a random way and moving at its family's launch
    /// speed in another.
    pub fn spawn(&mut self, kind: Kind, at: P) -> &mut Enemy {
        let id = self.next_id();
        let stats = kind.stats();
        let mut body = Body::new(pose_at(at, self.rng.angle()));
        let intent = P::direction_from_angle(self.rng.angle()) * stats.launch;
        body.vel = intent;
        let [lo, hi] = stats.spin;
        body.spin = if lo < hi { self.rng.range(lo, hi) } else { lo };
        let chain = if kind == Kind::Serpent {
            (1..=SERPENT_LEN)
                .map(|k| {
                    Body::new(body.pose * Motor::translation(dir(-SERPENT_GAP * k as f32, 0.0)))
                })
                .collect()
        } else {
            Vec::new()
        };
        let phase = self.rng.range(0.0, 100.0);
        self.enemies.push(Enemy {
            id,
            kind,
            body,
            intent,
            pull: P::zero(),
            dead: false,
            hp: stats.hp,
            radius: stats.radius,
            mass: 0.0,
            age: 0.0,
            flash: 0.0,
            phase,
            chain,
            timer: 2.0,
        });
        self.enemies.last_mut().expect("just spawned")
    }

    /// Kill enemy `i` (it is removed once the enemies have moved); when `scored`, with
    /// points, shards and what it carried.
    pub(super) fn kill(&mut self, i: usize, scored: bool) {
        let e = &mut self.enemies[i];
        e.dead = true;
        let (kind, mass, body, radius) = (e.kind, e.mass, e.body, e.radius);
        let pos = body.pos();
        let points = if scored {
            self.reward(kind, mass, pos)
        } else {
            0
        };
        self.events.push(Event::Kill {
            pos,
            kind,
            size: kind.stats().size + 0.3 * mass,
            scored,
            points,
            wreck: Some(Wreck { body, radius }),
        });
        // A serpent's body goes with its head.
        self.events
            .extend(self.enemies[i].chain.iter().map(|seg| Event::Kill {
                pos: seg.pos(),
                kind: Kind::Serpent,
                size: 0.4,
                scored: false,
                points: 0,
                wreck: None,
            }));
        if scored {
            self.unload(kind, pos);
        }
    }

    /// Kill every enemy, unscored, and remove them now (a death, a bomb).
    pub(super) fn kill_all(&mut self) {
        for i in 0..self.enemies.len() {
            if !self.enemies[i].dead {
                self.kill(i, false);
            }
        }
        self.enemies.clear();
    }

    /// Score the kill of a `kind` that ate `mass` at `pos` (a singularity is worth more for
    /// what it ate) and drop its shards; the points gained.
    fn reward(&mut self, kind: Kind, mass: f32, pos: P) -> u64 {
        let stats = kind.stats();
        let points = self
            .ledger
            .award((stats.value + 100 * mass as u64) * u64::from(self.ledger.mult));
        let extra = stats.extra_shard > 0.0 && self.rng.chance(stats.extra_shard);
        let shards = stats.shards + mass as usize + usize::from(extra);
        self.drop_shards(pos, shards);
        points
    }

    /// What a killed enemy leaves behind: a splitter's fragments, a carrier's ring of motes.
    fn unload(&mut self, kind: Kind, pos: P) {
        match kind {
            // Three fragments, flung apart: a direction turned a third at a time.
            Kind::Splitter => {
                let third = Motor::rotation(ORIGIN, core::f32::consts::TAU / 3.0);
                for (d, _) in orbit(P::direction_from_angle(self.rng.angle()), 3, third) {
                    let f = self.spawn(Kind::Fragment, pos + d * 0.5);
                    f.intent = d * 9.0;
                    f.age = 0.25;
                }
            }
            // A carrier's cargo: a ring of motes.
            Kind::Carrier => {
                let tenth = Motor::rotation(ORIGIN, core::f32::consts::TAU / 10.0);
                for (d, _) in orbit(dir(1.0, 0.0), 10, tenth) {
                    self.spawn(Kind::Mote, pos + d * 1.5).intent = d * 7.0;
                }
                self.events.push(Event::Burst { pos });
            }
            _ => {}
        }
    }

    /// Move every living enemy by its family's rule and the wells' pull; carriers launch,
    /// singularities feed, and then the dead are removed.
    pub(super) fn step_enemies(&mut self) {
        let shots: Vec<Shot> = self
            .bullets
            .iter()
            .map(|b| Shot {
                line: (b.pos & b.vel).normalized(),
                pos: b.pos,
                vel: b.vel,
            })
            .collect();
        let view = View {
            ship: self.ship.body.pos(),
            hunting: self.phase == Phase::Playing,
            time: self.time,
            shots: &shots,
            wells: &self.wells,
        };
        let launches: Vec<P> = self
            .enemies
            .iter_mut()
            .filter(|e| !e.dead)
            .filter_map(|e| e.step(&view))
            .collect();
        for p in launches {
            self.launch(p);
        }
        self.feed_singularities();
        self.enemies.retain(|e| !e.dead);
    }

    /// A carrier's launch from `p`: a pair of motes. One launched across a wall starts
    /// reflected in it, where a bounce would have put it.
    fn launch(&mut self, p: P) {
        for side in [-1.0f32, 1.0] {
            let to = p + P::direction_from_angle(self.rng.angle()) * (1.3 * side);
            let margin = Kind::Mote.stats().radius;
            let at = outside(p, to, margin).map_or(to, |wall| wall.reflect(to));
            self.spawn(Kind::Mote, at);
        }
        self.events.push(Event::Spawn {
            pos: p,
            kind: Kind::Mote,
        });
    }
}

impl Enemy {
    /// One tick: its own motion, the wells' pull, the step, its body following, the walls.
    /// Where it launches motes from, if it does.
    fn step(&mut self, view: &View) -> Option<P> {
        self.age += DT;
        self.flash = (self.flash - DT * 6.0).max(0.0);
        let p = self.body.pos();
        let launches = self.act(p, view);
        // Wells pull everything but wells: the pull accumulates, and decays slowly.
        self.pull = self.pull * (1.0 - 0.6 * DT);
        if self.kind != Kind::Singularity {
            self.pull += pull(view.wells, p, 2.5 * DT);
        }
        self.body.vel = self.intent + self.pull;
        self.body.step(DT);
        self.follow_chain();
        self.bounce();
        launches.then_some(p)
    }

    /// Its intent for the tick, by its family's rule, from `p`; whether it launches.
    fn act(&mut self, p: P, view: &View) -> bool {
        let stats = self.kind.stats();
        match self.kind {
            Kind::Drifter => {
                // Wander: the velocity turns slowly with a smooth noise.
                let wave = crate::signal::wave;
                let (t, phase) = (view.time, self.phase);
                let noise = (wave(t * 0.7 + phase) + wave(t * 1.3 + phase * 1.7)) * 0.9;
                self.intent = with_length(turned(self.intent, noise * DT), stats.speed);
            }
            Kind::Chaser
            | Kind::Mote
            | Kind::Fragment
            | Kind::Evader
            | Kind::Splitter
            | Kind::Serpent => {
                let target = if view.hunting {
                    view.ship
                } else {
                    ORIGIN + P::direction_from_angle(self.phase) * 15.0
                };
                self.steer(p, target, view);
            }
            Kind::Warden => {
                // Turn slowly to face the ship, and come on behind the shield.
                let d = self.body.heading().angle_to(view.ship - p);
                self.body.spin = d.clamp(-1.0, 1.0) * stats.turn_rate;
                self.intent = self.body.heading() * (stats.speed * self.ramp());
            }
            Kind::Carrier => {
                // Drift, and launch a pair of motes every few seconds.
                self.intent = with_length(self.intent, stats.speed);
                self.timer -= DT;
                if self.timer <= 0.0 && view.hunting {
                    self.timer = 4.5;
                    return true;
                }
            }
            Kind::Singularity => {
                // Slow drift towards the ship, damped; it grows with what it ate.
                let towards = with_length(view.ship - p, 0.25 * DT);
                self.intent = (self.intent + towards) * (1.0 - 0.3 * DT);
                self.radius = stats.radius + 0.36 * gax::Real::sqrt(self.mass);
                self.hp = self.hp.min(stats.hp + 4.0 * self.mass);
            }
        }
        false
    }

    /// Its speed coming up over its first seconds, `0..1`.
    fn ramp(&self) -> f32 {
        (self.age * 1.5).min(1.0)
    }

    /// A hunter turns towards `target` at its limited rate and keeps its speed (a serpent winds
    /// as it comes, an evader sidesteps the shots coming its way).
    fn steer(&mut self, p: P, target: P, view: &View) {
        let stats = self.kind.stats();
        let to = target - p;
        let have = if self.intent.ideal_norm() > 0.1 {
            self.intent
        } else {
            to
        };
        // A serpent winds as it comes.
        let wind = if self.kind == Kind::Serpent {
            crate::signal::wave(view.time * 2.6 + self.phase) * 0.9
        } else {
            0.0
        };
        // Turn towards the target at a limited rate: part of the rotation between.
        let limit = stats.turn_rate * DT;
        let d = (have.angle_to(to) + wind).clamp(-limit, limit);
        let way = with_length(turned(have, d), 1.0);
        self.intent = way * (stats.speed * self.ramp());
        self.body.spin = if matches!(self.kind, Kind::Evader | Kind::Serpent) {
            // Face the way it goes.
            self.body.heading().angle_to(way) * 10.0
        } else {
            stats.hunting_spin
        };
        if self.kind == Kind::Evader {
            self.intent = clamp_length(self.intent + dodge(p, view.shots), 11.0);
        }
    }

    /// A serpent's body follows: each segment moves towards the spot just behind the one
    /// ahead, by motor interpolation.
    fn follow_chain(&mut self) {
        let k = 1.0 - (-30.0 * DT).exp();
        let mut ahead = self.body.pose;
        for seg in &mut self.chain {
            let target = ahead * Motor::translation(dir(-SERPENT_GAP, 0.0));
            seg.prev = seg.pose;
            seg.pose = interpolate(seg.pose, target, k);
            ahead = seg.pose;
        }
    }

    /// Bounce off the walls, by reflection in the wall it crossed: the position mirrored back
    /// inside, the intent and the pull mirrored.
    fn bounce(&mut self) {
        let (intent, pull) = (&mut self.intent, &mut self.pull);
        self.body.walls_crossed(self.radius, |body, wall| {
            body.move_to(wall.reflect(body.pos()));
            *intent = wall.reflect(*intent);
            *pull = wall.reflect(*pull);
            body.vel = *intent + *pull;
        });
    }
}

/// An evader's sidestep: for each shot coming its way whose line of flight passes close to `p`,
/// away from its foot on the line.
fn dodge(p: P, shots: &[Shot]) -> P {
    shots
        .iter()
        .filter(|s| {
            // Ahead of the shot: on the side of the perpendicular through it that the shot
            // moves towards.
            let across = s.line | s.pos;
            (across & p).s() * (across & s.vel).s() > 0.0 && p.distance(s.pos) <= 8.0
        })
        .map(|s| p - crate::geom::foot(p, s.line).unitized())
        .map(|away| (away, away.ideal_norm()))
        .filter(|&(_, off)| off < 1.8)
        .fold(P::zero(), |dodge, (away, off)| {
            dodge + with_length(away, (1.8 - off) * 9.0)
        })
}

#[cfg(test)]
mod tests {
    use super::super::{ARENA, Bullet, Input, at};
    use super::*;

    #[test]
    fn evaders_leave_the_line_of_fire() {
        let mut w = World::new(10);
        w.director.enabled = false;
        w.phase = Phase::Over; // no hunting: only the dodge moves it sideways
        let e = w.spawn(Kind::Evader, at(10.0, 0.3));
        e.body = Body::new(pose_at(at(10.0, 0.3), 0.0));
        e.age = 1.0;
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

    /// An enemy bounces off a wall by reflection: its velocity mirrored.
    #[test]
    fn walls_reflect_enemies() {
        let mut w = World::new(9);
        w.director.enabled = false;
        w.phase = Phase::Over;
        w.spawn(Kind::Carrier, at(ARENA[0] - 2.0, 0.0)).intent = dir(1.4, 0.0);
        for _ in 0..240 {
            w.tick(&Input::default());
        }
        let v = w.enemies[0].body.vel;
        assert!(v.e20() < 0.0, "did not bounce: {v:?}");
        let [x, _] = w.enemies[0].body.pos().to_euclidean();
        assert!(x < ARENA[0]);
    }

    /// A carrier against a wall launches its motes inside the arena.
    #[test]
    fn motes_launch_inside_the_arena() {
        for seed in 0..64 {
            let mut w = World::new(seed);
            w.director.enabled = false;
            let edge = at(ARENA[0] - 1.15, 0.0);
            w.launch(edge);
            for e in w.enemies.iter().filter(|e| e.kind == Kind::Mote) {
                let [x, y] = e.body.pos().to_euclidean();
                let r = Kind::Mote.stats().radius;
                assert!(
                    x.abs() <= ARENA[0] - r + 1e-4 && y.abs() <= ARENA[1] - r + 1e-4,
                    "{x} {y}"
                );
            }
        }
    }

    /// A carrier launches a pair of motes every 4.5 s (the first after 2 s), and only while
    /// there is a ship to hunt.
    #[test]
    fn carriers_launch_while_hunting() {
        let mut w = World::new(19);
        w.director.enabled = false;
        w.ship.invulnerable = 1e9;
        w.spawn(Kind::Carrier, at(20.0, 10.0));
        let mut launches = Vec::new();
        for _ in 0..120 * 12 {
            w.tick(&Input::default());
            let launched = w.events.iter().filter(|e| {
                matches!(
                    e,
                    Event::Spawn {
                        kind: Kind::Mote,
                        ..
                    }
                )
            });
            launches.extend(launched.map(|_| w.time));
        }
        assert_eq!(launches.len(), 3, "{launches:?}");
        for (t, want) in launches.iter().zip([2.0, 6.5, 11.0]) {
            assert!((t - want).abs() < 2.0 * DT, "{launches:?}");
        }
        w.end();
        for _ in 0..120 * 10 {
            w.tick(&Input::default());
            assert!(!w.events.iter().any(|e| matches!(e, Event::Spawn { .. })));
        }
    }

    /// Over whole runs, every enemy stays inside the arena.
    #[test]
    fn enemies_stay_in_the_arena() {
        for seed in 20..24 {
            let mut w = World::new(seed);
            w.ship.invulnerable = 1e9;
            for t in 0..120 * 150 {
                let a = t as f32 * 0.004 + seed as f32;
                w.tick(&Input {
                    movement: P::direction_from_angle(a * 1.7),
                    aim: P::direction_from_angle(a * 5.0),
                    fire: t % 600 < 300,
                    bomb: false,
                });
                for e in &w.enemies {
                    let [x, y] = e.body.pos().to_euclidean();
                    assert!(
                        x.abs() <= ARENA[0] && y.abs() <= ARENA[1],
                        "seed {seed}, tick {t}: a {:?} at ({x}, {y})",
                        e.kind
                    );
                }
            }
        }
    }

    #[test]
    fn splitters_split_and_carriers_unload() {
        let mut w = World::new(14);
        w.director.enabled = false;
        w.spawn(Kind::Splitter, at(0.0, 3.0));
        w.spawn(Kind::Carrier, at(10.0, -5.0));
        w.kill(1, true);
        w.kill(0, true);
        let count = |k| w.enemies.iter().filter(|e| e.kind == k).count();
        assert_eq!(count(Kind::Fragment), 3);
        assert_eq!(count(Kind::Mote), 10);
    }
}
