//! Shots: bent by the wells, swept against the enemies over each tick (so none tunnels
//! through), stopped by the walls and a serpent's body, reflected by a warden's shield.

use super::body::{Body, velocity_at};
use super::collide::{in_front, outside, segment_hits_circle, shield};
use super::kind::Kind;
use super::{DT, Event, P, World};

const BULLET_SPEED: f32 = 40.0;
const BULLET_LIFE: f32 = 1.4;

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

impl Bullet {
    /// A shot fired from `pos` along the unit direction `dir`.
    pub fn fired(pos: P, dir: P) -> Bullet {
        Bullet {
            pos,
            prev: pos,
            vel: dir * BULLET_SPEED,
            life: BULLET_LIFE,
        }
    }
}

/// A killed enemy's motion and size: its outline breaks into pieces that carry them on.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Wreck {
    /// Its motion.
    pub body: Body,
    /// Its drawn radius.
    pub radius: f32,
}

impl Wreck {
    /// The world velocity of its point at `p` (a unit point): its twist's.
    pub fn velocity_at(&self, p: P) -> P {
        velocity_at(self.body.twist(), p)
    }
}

/// What a shot did to an enemy on its path.
enum Strike {
    /// Missed it.
    Miss,
    /// Hit it, or its body, and is spent.
    Spent,
    /// Glanced off its shield and flies on.
    Glanced,
}

impl World {
    /// Move the shots, and resolve what they hit; the enemies they destroy die after, in the
    /// order they were hit.
    pub(super) fn step_bullets(&mut self) {
        self.hash_enemies();
        let mut bullets = std::mem::take(&mut self.bullets);
        let mut kills: Vec<usize> = Vec::new();
        bullets.retain_mut(|b| self.fly(b, &mut kills));
        self.bullets = bullets;
        for i in kills {
            self.kill(i, true);
        }
    }

    /// The enemies into the spatial hash, and a serpent's body segments under its index.
    fn hash_enemies(&mut self) {
        self.hash.clear();
        for (i, e) in self.enemies.iter().enumerate() {
            self.hash.insert(i as u32, e.body.pos(), e.radius);
            for seg in &e.chain {
                self.hash.insert(i as u32, seg.pos(), 0.45);
            }
        }
    }

    /// One shot's tick: bent by the wells, moved, and stopped by what it meets on its path;
    /// whether it flies on. Enemies it destroys go into `kills`.
    fn fly(&mut self, b: &mut Bullet, kills: &mut Vec<usize>) -> bool {
        b.life -= DT;
        if b.life <= 0.0 {
            return false;
        }
        b.vel += self.gravity(b.pos, 40.0) * DT;
        b.prev = b.pos;
        b.pos += b.vel * DT;
        if let Some(wall) = outside(b.prev, b.pos, 0.0) {
            // Where the path met the wall: the meet of two joins, a position.
            let pos = ((b.prev & b.pos) ^ wall).position();
            self.events.push(Event::Wall { pos });
            return false;
        }
        self.hash.query(b.prev, b.pos, 1.5, &mut self.scratch);
        for k in 0..self.scratch.len() {
            let i = self.scratch[k] as usize;
            match self.strike(b, i, kills) {
                Strike::Miss => {}
                Strike::Spent => return false,
                Strike::Glanced => return true,
            }
        }
        true
    }

    /// Shot `b` against enemy `i`: a serpent's body blocks it, a warden's shield reflects it
    /// from the front, anything else takes the hit (and is pushed back, if heavy).
    fn strike(&mut self, b: &mut Bullet, i: usize, kills: &mut Vec<usize>) -> Strike {
        let e = &mut self.enemies[i];
        if e.hp <= 0.0 {
            // Destroyed by an earlier shot this tick.
            return Strike::Miss;
        }
        if let Some(seg) = e
            .chain
            .iter()
            .find(|s| segment_hits_circle(b.prev, b.pos, s.pos(), 0.42))
        {
            let pos = seg.pos();
            self.events.push(Event::Deflect { pos, kind: e.kind });
            return Strike::Spent;
        }
        if !segment_hits_circle(b.prev, b.pos, e.body.pos(), e.radius + 0.08) {
            return Strike::Miss;
        }
        if e.kind == Kind::Warden && in_front(e.body.pose, b.prev) {
            b.vel = shield(e.body.pose).reflect(b.vel);
            b.pos = b.prev;
            self.events.push(Event::Deflect {
                pos: e.body.pos(),
                kind: e.kind,
            });
            return Strike::Glanced;
        }
        e.hp -= 1.0;
        e.flash = 1.0;
        e.intent += b.vel * e.kind.stats().knockback;
        if e.hp <= 0.0 {
            kills.push(i);
        } else {
            self.events.push(Event::Hit {
                pos: e.body.pos(),
                kind: e.kind,
            });
        }
        Strike::Spent
    }
}

#[cfg(test)]
mod tests {
    use super::super::body::{Pose, pose_at};
    use super::super::kind::SERPENT_GAP;
    use super::super::{ARENA_X, ARENA_Y, Input, at, dir};
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
        let e = w.spawn(Kind::Drifter, at(8.0, 0.0));
        e.intent = P::zero();
        e.body.spin = 0.0;
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
        assert_eq!(w.ledger.score, 50);
        assert_ne!(w.shards.len(), 0);
    }

    /// A shot's wall hit lies on the arena's border, also where the path crosses near a
    /// corner, nearly along one wall and through the other.
    #[test]
    fn wall_hits_lie_on_the_border() {
        let mut w = World::new(21);
        w.director.enabled = false;
        // Into the top-right corner: through the top first, almost along the right wall, and
        // the other way round; then the bot's shots in every direction.
        for (p, v) in [
            (at(31.9999, 17.99), dir(0.05, 40.0)),
            (at(31.9, 17.999), dir(40.0, 0.5)),
            (at(-31.99, -17.9), dir(-40.0, -39.0)),
        ] {
            w.bullets.push(Bullet {
                pos: p,
                prev: p,
                vel: v,
                life: 1.0,
            });
        }
        let mut hits = 0;
        for t in 0..120 * 20 {
            let a = t as f32 * 0.013;
            w.tick(&Input {
                movement: P::direction_from_angle(a * 0.7),
                aim: P::direction_from_angle(a * 3.0),
                fire: true,
                bomb: false,
            });
            for e in &w.events {
                if let Event::Wall { pos } = e {
                    hits += 1;
                    let [x, y] = pos.to_euclidean();
                    let (dx, dy) = (ARENA_X - x.abs(), ARENA_Y - y.abs());
                    assert!(
                        dx > -1e-3 && dy > -1e-3 && dx.min(dy) < 1e-3,
                        "tick {t}: a wall hit at ({x}, {y})"
                    );
                }
            }
        }
        assert!(hits > 100, "{hits} wall hits");
    }

    #[test]
    fn wardens_reflect_from_the_front_and_fall_from_behind() {
        let mut w = World::new(11);
        w.director.enabled = false;
        // Facing -x, towards the shots from the left: they glance off.
        let e = w.spawn(Kind::Warden, at(8.0, 0.0));
        e.body = Body::new(pose_at(at(8.0, 0.0), core::f32::consts::PI));
        e.age = 1.0;
        w.ship.body = Body::new(Pose::identity());
        let mut deflected = false;
        for _ in 0..30 {
            w.enemies[0].intent = P::zero();
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
        let e = w.spawn(Kind::Warden, at(8.0, 0.0));
        e.body = Body::new(pose_at(at(8.0, 0.0), 0.0));
        e.age = 1.0;
        let mut killed = false;
        for _ in 0..60 {
            w.enemies.iter_mut().for_each(|e| {
                e.intent = P::zero();
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
        w.ship.body = Body::new(pose_at(at(-28.0, -15.0), 0.0));
        w.ship.invulnerable = 1e9;
        let e = w.spawn(Kind::Serpent, at(0.0, 5.0));
        e.body = Body::new(pose_at(at(0.0, 5.0), 0.0));
        e.age = 1.0;
        // Let the body settle behind the head.
        for _ in 0..120 {
            w.enemies[0].intent = P::zero();
            w.tick(&Input::default());
        }
        let chain: Vec<P> = w.enemies[0].chain.iter().map(|s| s.pos()).collect();
        let gap = chain[0].distance(w.enemies[0].body.pos());
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
}
