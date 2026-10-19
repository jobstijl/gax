//! The director: an intensity curve over the run, formations announced ahead in the fog,
//! slaloms of gates, and the spawns landing when their warp-ins close.

use super::entities::{Enemy, Event, Foe, Gate, GateState, Pending, Segment};
use super::lattice::RADIUS;
use super::space::{AXIS, P, about_axis, around, dir, radial};
use super::{CRUISE, SHIP_RADIUS, SPAWN_AHEAD, World};
use crate::sim::DT;
use crate::sim::rng::Rng;
use gax::Unit;
use gax::pga3d::{Motor, Twist};

/// A serpent's segments, and the seconds between them along its screw.
pub(super) const SEGMENTS: usize = 14;
pub(super) const SEGMENT_LAG: f32 = 0.11;
/// Gates in a slalom, and their spacing along the track.
const GATES: usize = 6;
const GATE_GAP: f32 = 17.0;
/// How long a singularity holds ahead of the ship.
pub(super) const WELL_HOLD: f32 = 9.0;

impl World {
    /// The director's tick: the intensity, now and then a slalom of gates, and the next
    /// formation once the spawn cooldown has run out.
    pub(super) fn direct(&mut self) {
        self.clock += DT;
        self.intensity = crate::sim::director::Director::curve(self.clock * 1.3);
        self.spawn_cooldown -= DT;
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
        if self.spawn_cooldown > 0.0 || self.enemies.len() + self.pending.len() > 40 {
            return;
        }
        let spawn = self.formation(has_well);
        for p in spawn {
            self.events.push(Event::Warn {
                pos: p.pos,
                foe: p.foe,
            });
            self.pending.push(p);
        }
        self.spawn_cooldown = 1.8 + 3.0 * (1.0 - self.intensity) + self.rng.range(0.0, 1.0);
    }

    /// The next formation, announced in the fog ahead: by bands of one roll, each kind once
    /// it has been introduced, drones otherwise. Each kind is introduced by a deadline,
    /// whatever the rolls: serpents by 40 s, the first singularity by 60 s (unless one is
    /// about, `has_well`).
    fn formation(&mut self, has_well: bool) -> Vec<Pending> {
        let i = self.intensity;
        let s = self.ship.s() + SPAWN_AHEAD;
        let roll = self.rng.unit();
        let r = &mut self.rng;
        let late_serpent = !self.serpent_introduced && self.clock > 40.0;
        let late_well = !self.well_introduced && self.clock > 60.0;
        let well_band = roll < 0.16 || late_well;
        let turret_band = (0.16..0.36 + 0.12 * i).contains(&roll) && !late_serpent;
        let serpent_band = (0.48..0.64).contains(&roll) || late_serpent;
        let mine_band = (0.64..0.82).contains(&roll);
        if well_band && self.clock > 40.0 && !has_well && self.well_cooldown <= 0.0 {
            self.well_cooldown = 45.0;
            self.well_introduced = true;
            escorted_well(r, s, i)
        } else if turret_band && self.clock > 25.0 {
            turrets(r, s, i)
        } else if serpent_band && self.clock > 30.0 {
            self.serpent_introduced = true;
            serpent(r, s)
        } else if (turret_band || mine_band) && self.clock > 10.0 {
            mines(r, s, i)
        } else {
            drone_ring(r, s, i)
        }
    }

    /// A slalom of gates ahead, on a screw motion about the axis: each gate is the last one
    /// turned about the axis and moved along it by the same motor. Called once every gate of
    /// the last slalom has been passed or missed, which it forgets.
    fn slalom(&mut self) {
        let i = self.intensity;
        let r = &mut self.rng;
        let s0 = self.ship.s() + SPAWN_AHEAD * 0.8;
        let off = r.range(1.4, 2.4 + 1.2 * i);
        let side = if r.unit() < 0.5 { -1.0 } else { 1.0 };
        let turn = side * r.range(0.5, 0.9 + 0.5 * i);
        let screw = about_axis(turn) * Motor::translation(dir(0.0, 0.0, GATE_GAP));
        let first = around(off, r.angle(), s0);
        self.gates = screw
            .orbit(first, GATES)
            .map(|(pos, _)| Gate {
                pos,
                state: GateState::Open,
                t: 0.0,
            })
            .collect();
    }

    /// The spawns whose warp-ins have closed land as enemies.
    pub(super) fn land(&mut self) {
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
            let id = self.next_id();
            let drift = if p.foe == Foe::Mine {
                radial(0.7, self.rng.angle())
            } else {
                P::zero()
            };
            let (swim, body) = self.swim(p.foe, p.pos);
            if p.foe == Foe::Singularity {
                well = Some(id);
            }
            let timer = if p.foe == Foe::Singularity {
                WELL_HOLD
            } else {
                self.rng.range(0.6, 1.8)
            };
            self.enemies.push(Enemy {
                vel: drift,
                angle: p.angle,
                orbit: p.orbit,
                depth: p.depth,
                timer,
                swim,
                body,
                anchor: if p.escort { well } else { None },
                ..Enemy::new(id, p.foe, p.pos)
            });
        }
    }

    /// A fresh id for a new enemy.
    pub(super) fn next_id(&mut self) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    /// A new enemy's swim over one tick and its body, for a `foe` landing at `head`: none but
    /// a serpent's. A serpent swims a screw (a turn about the axis and a swim along it, `T` per
    /// second), and its body is behind its head on it: segment `k` is the head moved back
    /// along the screw by `k SEGMENT_LAG` seconds, `exp(-lag T)`.
    pub(super) fn swim(&mut self, foe: Foe, head: P) -> (Unit<Motor<(), f32>>, Vec<Segment>) {
        if foe != Foe::Serpent {
            return (Unit::identity(), Vec::new());
        }
        let side = if self.rng.unit() < 0.5 { -1.0 } else { 1.0 };
        let twist = Twist::rotation(AXIS, side * self.rng.range(1.3, 1.8))
            + Twist::translation(dir(0.0, 0.0, 0.55 * CRUISE));
        let body = (1..=SEGMENTS)
            .map(|k| {
                let behind = (twist * -(k as f32 * SEGMENT_LAG)).exp();
                let pos = behind >> head;
                Segment {
                    pos,
                    prev: pos,
                    behind,
                    hp: 2.0,
                }
            })
            .collect();
        ((twist * DT).exp(), body)
    }
}

impl Pending {
    /// A `foe` announced at `pos`, at `angle` around the axis, to circle at `orbit` and (a
    /// drone) to hold `depth` ahead of the ship.
    fn announced(foe: Foe, pos: P, angle: f32, orbit: f32, depth: f32) -> Pending {
        Pending {
            foe,
            pos,
            t: 0.8,
            angle,
            orbit,
            depth,
            escort: false,
        }
    }
}

/// A singularity at arc length `s`, holding ahead, with a ring of drones orbiting it (more at
/// intensity `i`).
fn escorted_well(r: &mut Rng, s: f32, i: f32) -> Vec<Pending> {
    let a0 = r.angle();
    let depth = r.range(24.0, 30.0);
    let off = r.range(1.0, 2.0);
    let well = around(off, a0, s);
    let n = 6 + (i * 4.0) as usize;
    let escorts = (0..n).map(|k| {
        let a = k as f32 * core::f32::consts::TAU / n as f32;
        Pending {
            escort: true,
            ..Pending::announced(Foe::Drone, well + radial(3.4, a), a, 3.4, depth)
        }
    });
    core::iter::once(Pending::announced(Foe::Singularity, well, a0, off, depth))
        .chain(escorts)
        .collect()
}

/// Turrets on the wall from arc length `s`, spaced along the track and around it.
fn turrets(r: &mut Rng, s: f32, i: f32) -> Vec<Pending> {
    let n = 1 + (i * 3.0) as usize;
    let a0 = r.angle();
    (0..n)
        .map(|k| {
            let a = a0 + k as f32 * 2.1;
            let p = around(RADIUS - 0.5, a, s + k as f32 * 7.0);
            Pending::announced(Foe::Turret, p, a, 0.0, 0.0)
        })
        .collect()
}

/// A serpent at arc length `s`, coming out of the fog on its helix.
fn serpent(r: &mut Rng, s: f32) -> Vec<Pending> {
    let a = r.angle();
    let orbit = r.range(3.2, 4.4);
    vec![Pending::announced(
        Foe::Serpent,
        around(orbit, a, s),
        a,
        orbit,
        0.0,
    )]
}

/// Mines scattered in the lane from arc length `s`.
fn mines(r: &mut Rng, s: f32, i: f32) -> Vec<Pending> {
    let n = 3 + (i * 6.0) as usize;
    (0..n)
        .map(|k| {
            let p = around(r.range(0.0, SHIP_RADIUS), r.angle(), s + k as f32 * 5.0);
            Pending::announced(Foe::Mine, p, 0.0, 0.0, 0.0)
        })
        .collect()
}

/// A ring of drones, which will hold in the band ahead (announced from arc length `s`).
fn drone_ring(r: &mut Rng, s: f32, i: f32) -> Vec<Pending> {
    let n = 6 + (i * 6.0) as usize;
    let a0 = r.angle();
    // Close and wide: the ring spreads across the screen, not at the vanishing point.
    let orbit = r.range(4.0, 5.3);
    let depth = r.range(6.0, 10.0);
    // Nearer than the rest, and quick to come in: a ring is a close-range fight.
    let at = s - SPAWN_AHEAD * 0.45;
    (0..n)
        .map(|k| {
            let a = a0 + k as f32 * core::f32::consts::TAU / n as f32;
            Pending::announced(Foe::Drone, around(orbit, a, at), a, orbit, depth)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tunnel::Input;
    use crate::tunnel::tests::aim_ahead;

    /// Over a long run the director brings everything: drones, mines, turrets, serpents, a
    /// singularity with its escort, and slaloms of gates.
    #[test]
    fn the_director_brings_everything() {
        let mut w = World::new(12);
        w.ship.invulnerable = 1e9;
        let mut seen = std::collections::BTreeSet::new();
        let mut escorts = false;
        let mut gates = false;
        let mut warned = std::collections::BTreeMap::new();
        for _ in 0..120 * 150 {
            // Shoot at the nearest enemy ahead, so that the tunnel keeps room for more.
            w.tick(&Input {
                aim: aim_ahead(&w),
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
