//! Gravity wells: singularities pull shots, enemies, the grid and the particles with an
//! inverse-square law, eat what falls in, and burst when overfed.

use super::kind::Kind;
use super::{Event, P, World};
use crate::geom::inverse_square;

/// A gravity well: a singularity, as the shots, the enemies, the lattice and the particles
/// feel it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Well {
    /// Where it is.
    pub pos: P,
    /// Its strength, growing with the mass it ate.
    pub strength: f32,
}

impl World {
    /// This tick's wells: every singularity as it stands at the start of the tick.
    pub(super) fn find_wells(&mut self) {
        self.wells.clear();
        let singularities = self.enemies.iter().filter(|e| e.kind == Kind::Singularity);
        self.wells.extend(singularities.map(|e| Well {
            pos: e.body.pos(),
            strength: 18.0 + 6.0 * e.mass,
        }));
    }

    /// How near the ship is to a singularity, `0..1` (the music darkens with it).
    pub fn darkness(&self) -> f32 {
        let ship = self.ship.body.pos();
        let near = |well: &Well| (1.0 - well.pos.distance(ship) / 14.0).clamp(0.0, 1.0);
        self.wells.iter().map(near).fold(0.0, f32::max)
    }

    /// The acceleration the wells give a body at `p`, times `scale`.
    pub(super) fn gravity(&self, p: P, scale: f32) -> P {
        pull(&self.wells, p, scale)
    }

    /// Singularities eat what falls in, and burst into motes when overfed. The eaten and the
    /// burst are marked dead.
    pub(super) fn feed_singularities(&mut self) {
        for wi in 0..self.enemies.len() {
            let w = &self.enemies[wi];
            if w.kind == Kind::Singularity && !w.dead {
                self.feed(wi);
            }
        }
        // Overfed singularities burst into motes.
        for i in 0..self.enemies.len() {
            let e = &mut self.enemies[i];
            if e.kind == Kind::Singularity && !e.dead && e.mass >= 10.0 {
                e.dead = true;
                let pos = e.body.pos();
                self.events.push(Event::Burst { pos });
                for _ in 0..14 {
                    let d = P::direction_from_angle(self.rng.angle()) * 1.2;
                    self.spawn(Kind::Mote, pos + d);
                }
            }
        }
    }

    /// Singularity `wi` eats the living, edible enemies (a moment after they land) within
    /// most of its radius.
    fn feed(&mut self, wi: usize) {
        let (at, reach) = (self.enemies[wi].body.pos(), self.enemies[wi].radius * 0.8);
        for i in 0..self.enemies.len() {
            let e = &mut self.enemies[i];
            if e.dead
                || !e.kind.stats().edible
                || e.age <= 0.3
                || e.body.pos().distance(at) >= reach
            {
                continue;
            }
            e.dead = true;
            let w = &mut self.enemies[wi];
            w.mass += 1.0;
            w.hp += 4.0;
            let (pos, mass) = (w.body.pos(), w.mass);
            self.events.push(Event::Absorb { pos, mass });
        }
    }
}

/// The pull of `wells` on a body at `p`: `k s d / (|d|² + r²)^(3/2)` towards each, an
/// inverse square with a softened core.
pub fn pull(wells: &[Well], p: P, k: f32) -> P {
    wells.iter().fold(P::zero(), |acc, w| {
        acc + inverse_square(w.pos - p, k * w.strength, 1.5)
    })
}

#[cfg(test)]
mod tests {
    use super::super::{Bullet, Input, at, dir};
    use super::*;

    #[test]
    fn singularities_bend_shots_and_eat_enemies() {
        let mut w = World::new(3);
        w.director.enabled = false;
        w.spawn(Kind::Singularity, at(0.0, 6.0)).intent = P::zero();
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

    /// A singularity that has eaten ten bursts into fourteen motes.
    #[test]
    fn an_overfed_singularity_bursts_into_motes() {
        let mut w = World::new(15);
        w.director.enabled = false;
        w.spawn(Kind::Singularity, at(0.0, 6.0)).mass = 10.0;
        w.tick(&Input::default());
        assert!(w.events.iter().any(|e| matches!(e, Event::Burst { .. })));
        assert!(w.enemies.iter().all(|e| e.kind == Kind::Mote));
        assert_eq!(w.enemies.len(), 14);
    }
}
