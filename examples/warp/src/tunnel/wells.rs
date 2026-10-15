//! Singularities as a field: their pull on the ship, shots, bolts and mines, their pinch of the
//! wall, and how near the ship is to one (for the music).

use super::World;
use super::entities::Foe;
use super::space::{P, dir};
use crate::geom::length_cubed;

/// How strongly a singularity pulls the wall in (a lattice source per unit of strength), and
/// how far along the tunnel (the squared reach).
pub const PINCH: f32 = 480.0;
pub const PINCH_REACH2: f32 = 49.0;

/// The singularities at one moment: where each is and how strong.
pub struct Wells(Vec<(P, f32)>);

impl Wells {
    /// The pull at `p`, `k s d / (|d|² + 1)^(3/2)` towards each singularity: an inverse
    /// square with a softened core.
    pub fn pull(&self, p: P, k: f32) -> P {
        self.0
            .iter()
            .fold(dir(0.0, 0.0, 0.0), |acc, &(w, strength)| {
                let d = w - p;
                let soft = length_cubed(d.ideal_norm_squared() + 1.0);
                acc + d * (k * strength / soft)
            })
    }

    /// Whether there are any.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Each singularity: where, and how strong.
    pub fn iter(&self) -> impl Iterator<Item = (P, f32)> + '_ {
        self.0.iter().copied()
    }
}

impl World {
    /// The singularities as they are now.
    pub fn wells(&self) -> Wells {
        Wells(
            self.enemies
                .iter()
                .filter(|e| e.foe == Foe::Singularity)
                .map(|e| (e.pos, e.strength()))
                .collect(),
        )
    }

    /// How near the ship is to a singularity, `0..1` (the music darkens and bends with it).
    pub fn darkness(&self) -> f32 {
        let ship = self.ship.pos;
        self.wells()
            .iter()
            .map(|(w, strength)| (1.0 - (w & ship).norm() / (22.0 + 8.0 * strength)).max(0.0))
            .fold(0.0, f32::max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::DT;
    use crate::tunnel::entities::{Event, Flight, Shot};
    use crate::tunnel::lattice::RADIUS;
    use crate::tunnel::space::{arc, at};
    use crate::tunnel::tests::spawn;
    use crate::tunnel::{Input, SHIP_RADIUS};

    /// A singularity pinches the wall in around it, so the ship must fly further in there;
    /// it bends shots that pass it; it eats mines that fall in, and bursts when overfed.
    #[test]
    fn a_singularity_pinches_bends_eats_and_bursts() {
        let mut w = World::new(6);
        w.spawn_cooldown = 1e9;
        w.gate_timer = 1e9;
        w.ship.invulnerable = 1e9;
        let i = spawn(&mut w, Foe::Singularity, at(0.0, 0.0, 30.0));
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
        let wells = w.wells();
        for _ in 0..40 {
            shot.vel += wells.pull(shot.pos, 900.0) * DT;
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
            let m = spawn(&mut w, Foe::Mine, p + dir(0.3 * k as f32 * 0.1, 0.2, 0.0));
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
}
