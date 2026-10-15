//! The ship: flying and firing, dying and coming back, bombs, extra lives and bombs, and the
//! shards it collects.

use super::body::{Body, ORIGIN, distance, place, pose_at, turned, with_length};
use super::collide::outside;
use super::shots::Bullet;
use super::{DT, Event, Input, P, Phase, World};

const SHIP_SPEED: f32 = 11.5;
/// The ship's collision radius.
pub const SHIP_RADIUS: f32 = 0.42;
/// Seconds between shots.
const FIRE_INTERVAL: f32 = 1.0 / 22.0;
/// Seconds the ship is gone after a death.
const RESPAWN_TIME: f32 = 2.2;
/// Seconds of invulnerability after a respawn.
const RESPAWN_SHIELD: f32 = 2.5;

/// The ship.
#[derive(Clone, Debug)]
pub struct Ship {
    /// Motion.
    pub body: Body,
    /// Seconds until the next shot.
    pub cooldown: f32,
    /// Seconds of invulnerability left.
    pub invulnerable: f32,
    /// Alternates the two gun barrels.
    pub barrel: bool,
}

impl Ship {
    /// The ship at the start of a run.
    pub fn new() -> Ship {
        Ship {
            body: at_start(),
            cooldown: 0.0,
            invulnerable: 2.0,
            barrel: false,
        }
    }
}

impl Default for Ship {
    fn default() -> Ship {
        Ship::new()
    }
}

/// A body at rest at the arena's centre, facing up: where the ship starts.
fn at_start() -> Body {
    Body::new(pose_at(0.0, 0.0, core::f32::consts::FRAC_PI_2))
}

/// A shard dropped by a kill; collecting it raises the multiplier.
#[derive(Clone, Copy, Debug)]
pub struct Shard {
    /// Motion.
    pub body: Body,
    /// Seconds left.
    pub life: f32,
}

impl World {
    /// Fly by the input, stop at the walls, and fire.
    pub(super) fn steer_ship(&mut self, input: &Input) {
        let ship = &mut self.ship;
        ship.invulnerable = (ship.invulnerable - DT).max(0.0);
        let want = input.movement * SHIP_SPEED;
        // Quick, smooth response: approach the wanted velocity exponentially.
        let k = 1.0 - (-14.0 * DT).exp();
        ship.body.vel = ship.body.vel + (want - ship.body.vel) * k;
        // Turn the nose towards the motion: by the rotation from heading to velocity.
        ship.body.spin = if ship.body.vel.ideal_norm() > 0.5 {
            (ship.body.heading().angle_to(ship.body.vel) * 16.0).clamp(-24.0, 24.0)
        } else {
            0.0
        };
        ship.body.step(DT);
        // Walls: stop at the wall, keeping only the motion along it (the mean of the velocity
        // and its reflection in the wall).
        for _ in 0..2 {
            let Some(wall) = outside(ship.body.prev >> ORIGIN, ship.body.pos(), SHIP_RADIUS) else {
                break;
            };
            ship.body
                .move_to(crate::geom::foot(ship.body.pos(), wall).unitized());
            ship.body.vel = (ship.body.vel + wall.reflect(ship.body.vel)) * 0.5;
        }
        ship.cooldown -= DT;
        if input.fire && input.aim.ideal_norm() > 0.2 && ship.cooldown <= 0.0 {
            ship.cooldown += FIRE_INTERVAL;
            self.fire(input.aim);
        }
        self.ship.cooldown = self.ship.cooldown.max(-FIRE_INTERVAL);
    }

    /// One shot along `aim`, from the next of the two barrels: offset across the aim, a
    /// quarter turn of it.
    fn fire(&mut self, aim: P) {
        let ship = &mut self.ship;
        let u = with_length(aim, 1.0);
        let side = if ship.barrel { 0.14 } else { -0.14 };
        let across = turned(u, core::f32::consts::FRAC_PI_2);
        let o = ship.body.pos();
        self.bullets
            .push(Bullet::fired(o + u * 0.4 + across * side, u));
        ship.barrel = !ship.barrel;
        self.events.push(Event::Fire { pos: o, dir: u });
    }

    /// The ship dies when an enemy (or a serpent's segment) touches it, unless it is
    /// invulnerable.
    pub(super) fn collide_ship(&mut self) {
        if self.ship.invulnerable > 0.0 {
            return;
        }
        let s = self.ship.body.pos();
        let hit = self.enemies.iter().any(|e| {
            e.age > 0.25
                && (distance(e.body.pos(), s) < e.radius * 0.8 + SHIP_RADIUS
                    || e.chain
                        .iter()
                        .any(|seg| distance(seg.pos(), s) < 0.35 + SHIP_RADIUS))
        });
        if hit {
            self.die();
        }
    }

    /// The ship is destroyed: a life and the multiplier go, and the screen is cleared; the
    /// last life ends the run.
    fn die(&mut self) {
        let pos = self.ship.body.pos();
        self.events.push(Event::Death { pos });
        let left = self.ledger.lose_life();
        self.bullets.clear();
        self.pending.clear();
        self.kill_all();
        if left {
            self.phase = Phase::Dead(RESPAWN_TIME);
        } else {
            self.end();
            self.events.push(Event::GameOver);
        }
    }

    /// The ship comes back at the centre, invulnerable for a while.
    pub(super) fn respawn(&mut self) {
        self.phase = Phase::Playing;
        self.ship.invulnerable = RESPAWN_SHIELD;
        self.ship.body = at_start();
        self.director.after_death();
        self.events.push(Event::Respawn);
    }

    /// A bomb, if one is left: every enemy and warp-in gone, a moment of invulnerability.
    pub(super) fn bomb(&mut self) {
        if !self.ledger.take_bomb() {
            return;
        }
        let pos = self.ship.body.pos();
        self.events.push(Event::Bomb { pos });
        self.kill_all();
        self.pending.clear();
        self.ship.invulnerable = self.ship.invulnerable.max(1.0);
        self.director.after_bomb();
    }

    /// The extra lives and bombs the score has earned.
    pub(super) fn extras(&mut self) {
        let events = &mut self.events;
        self.ledger
            .extras(|life| events.push(Event::Extra { life }));
    }

    /// Scatter `count` shards from `pos`.
    pub(super) fn drop_shards(&mut self, pos: P, count: usize) {
        for _ in 0..count {
            let a = self.rng.angle();
            let mut body = Body::new(place(pos, a));
            body.vel = P::direction_from_angle(a) * self.rng.range(2.0, 7.0);
            body.spin = self.rng.range(-5.0, 5.0);
            self.shards.push(Shard { body, life: 7.0 });
        }
    }

    /// Shards drift and slow down; near a living ship a magnet draws them in, and touching it
    /// collects them.
    pub(super) fn step_shards(&mut self) {
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
                let Some(wall) = outside(s.body.prev >> ORIGIN, s.body.pos(), 0.0) else {
                    break;
                };
                s.body
                    .move_to(crate::geom::foot(s.body.pos(), wall).unitized());
            }
            true
        });
        for _ in 0..picked {
            let mult = self.ledger.raise();
            let pos = self.ship.body.pos();
            self.events.push(Event::Pickup { pos, mult });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::kind::Kind;
    use super::super::ledger::{BOMB_EVERY, LIFE_EVERY};
    use super::super::{ARENA, at, dir};
    use super::*;

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
        assert_eq!(w.ledger.lives, 2);
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
        assert_eq!(w.ledger.bombs, 2);
    }

    #[test]
    fn multiplier_rises_with_shards() {
        let mut w = World::new(5);
        w.director.enabled = false;
        let mut s = Body::new(pose_at(0.5, 0.0, 0.0));
        s.vel = dir(0.0, 0.0);
        w.shards.push(Shard { body: s, life: 5.0 });
        w.tick(&Input::default());
        assert_eq!(w.ledger.mult, 2);
    }

    /// The ship stops at a wall: its motion into the wall gone.
    #[test]
    fn walls_stop_the_ship() {
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

    /// A death takes a life and clears the screen; the ship comes back after 2.2 s,
    /// invulnerable for 2.5 s.
    #[test]
    fn the_ship_respawns_shielded() {
        let mut w = World::new(16);
        w.director.enabled = false;
        w.die();
        assert_eq!((w.ledger.lives, w.phase), (2, Phase::Dead(RESPAWN_TIME)));
        let mut ticks = 0;
        while !w.events.contains(&Event::Respawn) {
            w.tick(&Input::default());
            ticks += 1;
            assert!(ticks < 1000, "no respawn");
        }
        assert!(
            (ticks as f32 * DT - RESPAWN_TIME).abs() <= DT,
            "{ticks} ticks"
        );
        assert_eq!(w.phase, Phase::Playing);
        // Respawned at the start of the tick, then a tick of the shield used up.
        assert!((w.ship.invulnerable - (RESPAWN_SHIELD - DT)).abs() < 1e-6);
    }

    /// The last life ends the run: game over, the shots and warp-ins gone, and the shards
    /// drift on and expire instead of freezing.
    #[test]
    fn the_last_life_ends_the_run() {
        let mut w = World::new(17);
        w.director.enabled = false;
        w.ledger.lives = 1;
        w.drop_shards(at(10.0, 5.0), 5);
        w.announce(Kind::Chaser, at(-12.0, 5.0), 1.0);
        w.tick(&Input {
            aim: dir(1.0, 0.0),
            fire: true,
            ..Input::default()
        });
        assert!(!w.bullets.is_empty());
        w.die();
        assert!(w.events.contains(&Event::GameOver));
        assert_eq!(w.phase, Phase::Over);
        assert!(w.bullets.is_empty() && w.pending.is_empty());
        let start = w.shards[0].body.pos();
        w.tick(&Input::default());
        assert!(w.shards[0].body.pos() != start, "the shards froze");
        for _ in 0..120 * 8 {
            w.tick(&Input::default());
        }
        assert!(w.shards.is_empty(), "{} shards left", w.shards.len());
    }

    /// An extra life at 100 000 points, an extra bomb at 150 000.
    #[test]
    fn extras_at_their_scores() {
        let mut w = World::new(18);
        w.director.enabled = false;
        w.ledger.score = LIFE_EVERY - 1;
        w.tick(&Input::default());
        assert_eq!((w.ledger.lives, w.ledger.bombs), (3, 3));
        w.ledger.score = LIFE_EVERY;
        w.tick(&Input::default());
        assert_eq!((w.ledger.lives, w.ledger.bombs), (4, 3));
        assert!(w.events.contains(&Event::Extra { life: true }));
        w.ledger.score = BOMB_EVERY;
        w.tick(&Input::default());
        assert_eq!((w.ledger.lives, w.ledger.bombs), (4, 4));
        assert!(w.events.contains(&Event::Extra { life: false }));
        w.ledger.score = 2 * LIFE_EVERY;
        w.tick(&Input::default());
        assert_eq!((w.ledger.lives, w.ledger.bombs), (5, 4));
    }
}
