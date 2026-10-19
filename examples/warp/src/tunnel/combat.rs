//! Fighting: the ship's shots (aimed, leading a locked enemy, bent by singularities), the
//! turrets' bolts, hits on the ship, bombs and blasts on the wall, death and respawn.

use super::entities::{Enemy, Event, Foe, Shot};
use super::lattice::Bulge;
use super::space::{P, arc, at, dir, off_axis, segment_hits_sphere};
use super::wells::Wells;
use super::{Input, LOCK_AHEAD, World};
use crate::sim::Phase;

/// The shots' speed along the track, relative to the ship.
const SHOT_SPEED: f32 = 60.0;
const FIRE_INTERVAL: f32 = 1.0 / 14.0;
/// A shot bent by more than this (radians) before it kills is a slingshot.
const SLINGSHOT: f32 = 0.3;
/// The ship's hit sphere (shots are more generous: `HIT_MARGIN`).
const SHIP_SIZE: f32 = 0.38;
const HIT_MARGIN: f32 = 0.35;
/// How far a bomb reaches ahead.
const BOMB_REACH: f32 = 70.0;

impl World {
    /// The enemy nearest to `p` within `reach`, if any: a locked target.
    fn nearest(&self, p: P, reach: f32) -> Option<usize> {
        let s = self.ship.s();
        self.enemies
            .iter()
            .enumerate()
            .filter(|(_, e)| arc(e.pos) > s + LOCK_AHEAD)
            .map(|(i, e)| (i, e.pos.distance(p)))
            .filter(|&(_, d)| d < reach)
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(i, _)| i)
    }

    /// Where the next shot leaves: the barrel that did not fire last.
    fn muzzle(&self) -> P {
        let side = if self.ship.barrel { -0.35 } else { 0.35 };
        self.ship.pos + dir(side, 0.0, 0.8)
    }

    /// Where a shot fired now meets the enemy `e`: where `e` will be when the shot crosses its
    /// arc length.
    pub fn lead(&self, e: &Enemy) -> P {
        let closing = (self.ship.speed + SHOT_SPEED - e.vel.e021()).max(10.0);
        let t = (arc(e.pos) - arc(self.muzzle())) / closing;
        e.pos + e.vel * t
    }

    /// Fire, if the trigger is held and the gun is ready: along the line to the aim point, or
    /// to a locked enemy's lead.
    pub(super) fn fire(&mut self, input: &Input) {
        if !input.fire || self.ship.cooldown > 0.0 {
            return;
        }
        let muzzle = self.muzzle();
        let target = match self.nearest(input.aim, 2.5) {
            Some(i) => self.lead(&self.enemies[i]),
            None => input.aim,
        };
        self.ship.cooldown = FIRE_INTERVAL;
        self.ship.barrel = !self.ship.barrel;
        // Along the line to the target, at the shots' speed along the track.
        let forward = self.ship.speed + SHOT_SPEED;
        let to = target - muzzle;
        let ahead = to.e021().max(4.0);
        let vel = dir(to.e032(), to.e013(), ahead) * (forward / ahead);
        self.shots.push(Shot::new(muzzle, vel, 1.6));
        self.events.push(Event::Fire { pos: muzzle });
    }

    /// A bomb: every enemy within reach ahead dies unscored, bolts and near spawns vanish, and
    /// blasts run down the wall.
    pub(super) fn bomb(&mut self) {
        if !self.ledger.take_bomb() {
            return;
        }
        let s = self.ship.s();
        for i in (0..self.enemies.len()).rev() {
            if arc(self.enemies[i].pos) - s < BOMB_REACH {
                self.kill(i, false, false);
            }
        }
        self.bolts.clear();
        self.pending.retain(|p| arc(p.pos) - s > BOMB_REACH);
        self.events.push(Event::Bomb { pos: self.ship.pos });
        for k in 0..6 {
            self.blast(at(0.0, 0.0, s + 4.0 + k as f32 * 9.0), 900.0, 16.0, 0.3);
        }
        self.spawn_cooldown = self.spawn_cooldown.max(2.0);
    }

    /// A blast on the wall at `pos` for `time` seconds: a bulge.
    pub(super) fn blast(&mut self, pos: P, push: f32, r2: f32, time: f32) {
        self.blasts.push((
            Bulge {
                pos,
                push,
                r2,
                reach2: 0.0,
            },
            time,
        ));
    }

    /// The shots' tick: pulled by the singularities, they hit enemies (a serpent's body
    /// shields its head) or the wall.
    pub(super) fn step_shots(&mut self, wells: &Wells) {
        // Hits: the enemy, the segment of a serpent's body (or `None`: the enemy itself), and
        // whether the shot was bent round a singularity.
        let mut hits: Vec<(usize, Option<usize>, bool)> = Vec::new();
        let mut walls = Vec::new();
        let lattice = &self.lattice;
        for b in &mut self.shots {
            b.advance(wells, 900.0);
            let hit = self
                .enemies
                .iter()
                .enumerate()
                .find_map(|(i, e)| e.swept(b.prev, b.pos, HIT_MARGIN).map(|seg| (i, seg)));
            if let Some((i, seg)) = hit {
                hits.push((i, seg, b.bend() > SLINGSHOT));
                b.life = -1.0;
            } else if off_axis(b.pos) > lattice.radius(arc(b.pos)) {
                walls.push(b.pos);
                b.life = -1.0;
            }
        }
        self.events
            .extend(walls.into_iter().map(|pos| Event::Wall { pos }));
        self.shots.retain(|b| b.life > 0.0);
        self.take_hits(hits);
    }

    /// The shots' hits (each the enemy, the segment of a serpent's body or `None`, and whether
    /// the shot was bent): damage, body segments shot away, and kills.
    fn take_hits(&mut self, hits: Vec<(usize, Option<usize>, bool)>) {
        let mut dead: Vec<(usize, bool)> = Vec::new();
        let mut segments: Vec<(usize, usize)> = Vec::new();
        for (i, seg, bent) in hits {
            let e = &mut self.enemies[i];
            match seg {
                // Already shot away this tick.
                Some(j) if segments.contains(&(i, j)) => {}
                Some(j) => {
                    let s = &mut e.body[j];
                    s.hp -= 1.0;
                    if s.hp <= 0.0 {
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
        // Highest first, so that the indices hold.
        segments.sort_unstable();
        for &(i, j) in segments.iter().rev() {
            self.shoot_away(i, j);
        }
        dead.sort_unstable();
        for (i, bent) in dead.into_iter().rev() {
            self.kill(i, true, bent);
        }
    }

    /// Segment `j` of serpent `i`'s body shot away: scored, with a small blast on the wall.
    fn shoot_away(&mut self, i: usize, j: usize) {
        let s = self.enemies[i].body.remove(j);
        let points = (50.0 * self.ledger.mult as f32 * self.speed_bonus()).round() as u64;
        self.ledger.award(points);
        self.events.push(Event::Kill {
            pos: s.pos,
            foe: Foe::Serpent,
            points,
        });
        self.blast(s.pos, 160.0, 3.0, 0.08);
    }

    /// The bolts' tick: pulled by the singularities; gone when spent or behind the ship.
    pub(super) fn step_bolts(&mut self, wells: &Wells) {
        let s = self.ship.s();
        for b in &mut self.bolts {
            b.advance(wells, 300.0);
        }
        self.bolts.retain(|b| b.life > 0.0 && arc(b.pos) > s - 4.0);
    }

    /// The ship's last step against every enemy, every serpent's body and every bolt.
    pub(super) fn collide_ship(&mut self) {
        if self.ship.invulnerable > 0.0 {
            return;
        }
        let (p, prev) = (self.ship.pos, self.ship.prev);
        let hit_enemy = self
            .enemies
            .iter()
            .any(|e| e.swept(prev, p, SHIP_SIZE).is_some());
        let hit_bolt = self
            .bolts
            .iter()
            .any(|b| segment_hits_sphere(b.prev, b.pos, p, SHIP_SIZE + 0.15));
        if hit_enemy || hit_bolt {
            self.die();
        }
    }

    /// The ship destroyed: a life lost, the multiplier and chain reset; the run is over when
    /// no lives are left.
    fn die(&mut self) {
        let pos = self.ship.pos;
        self.events.push(Event::Death { pos });
        self.blast(pos, 2500.0, 10.0, 0.3);
        self.chain = 0;
        let left = self.ledger.lose_life();
        self.ship.roll = None;
        self.shots.clear();
        if left {
            self.phase = Phase::Dead(2.0);
        } else {
            self.phase = Phase::Over;
            self.events.push(Event::GameOver);
        }
    }

    /// The ship back, invulnerable for a while, with room around it.
    pub(super) fn respawn(&mut self) {
        self.phase = Phase::Playing;
        let s = self.ship.s();
        self.ship.pos = at(0.0, -2.0, s);
        self.ship.prev = self.ship.pos;
        self.ship.invulnerable = 2.5;
        self.bolts.clear();
        self.enemies.retain(|e| arc(e.pos) - s > 30.0);
        self.events.push(Event::Respawn);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tunnel::director::SEGMENTS;
    use crate::tunnel::entities::{Flight, Pending};
    use crate::tunnel::tests::{close, spawn};
    use crate::tunnel::{Foe, around};

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
            vel: dir(3.0, 0.0, 0.0),
            hp: 5.0,
            timer: 9.0,
            flight: Flight::Hold,
            ..Enemy::new(99, Foe::Mine, at(2.0, 1.0, 25.0))
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

    /// The lead is where a locked shot meets its enemy: both get there at the same time.
    #[test]
    fn the_lead_is_where_the_shot_meets_the_enemy() {
        let mut w = World::new(1);
        w.ship.pos = at(0.5, -1.0, 0.0);
        let e = Enemy {
            vel: dir(3.0, -1.0, 4.0),
            flight: Flight::Hold,
            ..Enemy::new(99, Foe::Mine, at(2.0, 1.0, 25.0))
        };
        w.enemies.push(e.clone());
        let lead = w.lead(&e);
        w.fire(&Input {
            fire: true,
            aim: e.pos,
            ..Input::default()
        });
        let b = w.shots[0];
        let t = (arc(lead) - arc(b.pos)) / b.vel.e021();
        assert!(close(b.pos + b.vel * t, lead));
        assert!(close(e.pos + e.vel * t, lead));
    }

    /// A bomb kills every enemy within its reach, unscored, with the bolts and the spawns on
    /// their way there; what lies beyond stays.
    #[test]
    fn a_bomb_clears_its_reach() {
        let mut w = World::new(2);
        let s = w.ship.s();
        spawn(&mut w, Foe::Drone, at(1.0, 0.0, s + 30.0));
        spawn(&mut w, Foe::Turret, at(0.0, 6.0, s + 69.0));
        spawn(&mut w, Foe::Mine, at(0.0, 2.0, s + 90.0));
        let pending = |ahead: f32| Pending {
            foe: Foe::Mine,
            pos: at(0.0, 0.0, s + ahead),
            t: 0.8,
            angle: 0.0,
            orbit: 0.0,
            depth: 0.0,
            escort: false,
        };
        w.pending.extend([pending(40.0), pending(100.0)]);
        w.bolts
            .push(Shot::new(at(0.0, 0.0, s + 20.0), dir(0.0, 0.0, -5.0), 8.0));
        w.bomb();
        assert_eq!(w.ledger.bombs, 2);
        let kills: Vec<u64> = w
            .events
            .iter()
            .filter_map(|e| match *e {
                Event::Kill { points, .. } => Some(points),
                _ => None,
            })
            .collect();
        assert_eq!(kills, [0, 0]);
        assert_eq!(w.ledger.score, 0);
        assert_eq!(w.enemies.len(), 1);
        assert_eq!(w.enemies[0].foe, Foe::Mine);
        assert_eq!(w.pending.len(), 1);
        assert!(arc(w.pending[0].pos) > s + BOMB_REACH);
        assert!(w.bolts.is_empty());
    }

    /// A turret's bolt that meets the ship destroys it.
    #[test]
    fn a_bolt_kills_the_ship() {
        let mut w = World::new(3);
        w.spawn_cooldown = 1e9;
        w.ship.invulnerable = 0.0;
        let ship = w.ship.pos;
        w.bolts.push(Shot::new(
            ship + dir(0.0, 0.0, 0.4),
            dir(0.0, 0.0, -40.0),
            8.0,
        ));
        w.tick(&Input::default());
        assert!(w.events.iter().any(|e| matches!(e, Event::Death { .. })));
        assert_eq!(w.ledger.lives, 2);
        assert_eq!(w.phase, Phase::Dead(2.0));
    }

    /// A death resets the multiplier and the ship comes back after a while, invulnerable; the
    /// last death ends the run, and the world no longer flies the ship.
    #[test]
    fn a_death_respawns_and_the_last_ends_the_run() {
        let mut w = World::new(3);
        w.spawn_cooldown = 1e9;
        w.gate_timer = 1e9;
        w.ledger.lives = 2;
        w.ledger.mult = 5;
        w.die();
        assert_eq!(w.phase, Phase::Dead(2.0));
        assert_eq!((w.ledger.lives, w.ledger.mult), (1, 1));
        let mut ticks = 0;
        while !w.events.contains(&Event::Respawn) {
            w.tick(&Input::default());
            ticks += 1;
            assert!(ticks <= 241, "no respawn");
        }
        assert!(ticks >= 239, "{ticks}");
        assert_eq!(w.phase, Phase::Playing);
        assert!(w.ship.invulnerable > 0.0);
        w.die();
        assert_eq!(w.phase, Phase::Over);
        assert!(w.events.contains(&Event::GameOver));
        assert_eq!(w.ledger.lives, 0);
        let s = w.ship.s();
        w.tick(&Input {
            fire: true,
            ..Input::default()
        });
        assert_eq!(w.ship.s(), s);
        assert!(w.shots.is_empty());
    }

    /// A shot that leaves the tunnel hits its wall.
    #[test]
    fn a_shot_hits_the_wall() {
        let mut w = World::new(1);
        w.spawn_cooldown = 1e9;
        let s = w.ship.s();
        w.shots
            .push(Shot::new(at(6.8, 0.0, s + 10.0), dir(60.0, 0.0, 30.0), 1.0));
        w.tick(&Input::default());
        assert!(w.events.iter().any(|e| matches!(e, Event::Wall { .. })));
        assert!(w.shots.is_empty());
    }

    /// A body segment shot away scores 50 times the multiplier, once, however many shots hit
    /// it in the same tick.
    #[test]
    fn a_serpent_segment_shot_away_scores_fifty_times_the_multiplier() {
        let mut w = World::new(2);
        w.spawn_cooldown = 1e9;
        w.gate_timer = 1e9;
        w.ship.invulnerable = 1e9;
        let head = around(4.0, 0.3, w.ship.s() + 30.0);
        let i = spawn(&mut w, Foe::Serpent, head);
        w.ledger.mult = 3;
        w.enemies[i].body[5].hp = 1.0;
        let at_segment = w.enemies[i].body[5].pos;
        for _ in 0..2 {
            let from = at_segment - dir(0.0, 0.0, 0.1);
            w.shots.push(Shot::new(from, dir(0.0, 0.0, 1.0), 1.0));
        }
        w.tick(&Input::default());
        let kills: Vec<u64> = w
            .events
            .iter()
            .filter_map(|e| match *e {
                Event::Kill { points, .. } => Some(points),
                _ => None,
            })
            .collect();
        assert_eq!(kills, [150]);
        assert!(!w.events.iter().any(|e| matches!(e, Event::Hit { .. })));
        assert_eq!(w.ledger.score, 150);
        assert_eq!(w.enemies[i].body.len(), SEGMENTS - 1);
    }
}
