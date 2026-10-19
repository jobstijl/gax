//! The spawn director: an intensity curve over time with calm stretches between spikes, and
//! formations that are always telegraphed and never land within the safe radius.

use super::rng::Rng;
use super::{ARENA, DT, Event, Kind, P, Phase, SAFE_RADIUS, World, at, dir};
use crate::geom::{ORIGIN, orbit, with_length};
use gax::pga2d::Motor;

/// A telegraphed spawn.
#[derive(Clone, Copy, Debug)]
pub struct Pending {
    /// What comes.
    pub kind: Kind,
    /// Where.
    pub pos: P,
    /// Seconds until it lands.
    pub t: f32,
    /// Its warning time, for drawing the warp-in.
    pub total: f32,
}

impl World {
    /// Announce a spawn: it lands after `t` seconds, never within the safe radius.
    pub fn announce(&mut self, kind: Kind, pos: P, t: f32) {
        let pos = self.safe_spot(pos);
        self.pending.push(Pending {
            kind,
            pos,
            t,
            total: t,
        });
        self.events.push(Event::Warn { pos, kind });
    }

    /// `p` clamped into the arena (a box: its coordinates bounded), then pushed out of the safe
    /// radius around the ship along the ray from it (or back through it, if that leaves the
    /// arena).
    pub fn safe_spot(&self, p: P) -> P {
        let inside = |p: P| {
            let [x, y] = p.to_euclidean();
            at(
                x.clamp(-ARENA[0] + 1.5, ARENA[0] - 1.5),
                y.clamp(-ARENA[1] + 1.5, ARENA[1] - 1.5),
            )
        };
        let p = inside(p);
        let s = self.ship.body.pos();
        let need = SAFE_RADIUS + 1.0;
        if p.distance(s) >= need {
            return p;
        }
        let away = p - s;
        let u = if away.ideal_norm() > 1e-3 {
            with_length(away, need)
        } else {
            dir(need, 0.0)
        };
        let (out, back) = (inside(s + u), inside(s - u));
        if out.distance(s) >= back.distance(s) {
            out
        } else {
            back
        }
    }

    /// Count down the warp-ins, and land the ones whose time has come.
    pub(super) fn step_pending(&mut self) {
        let mut landed = Vec::new();
        let ship = self.ship.body.pos();
        let playing = self.phase == Phase::Playing;
        self.pending.retain_mut(|p| {
            p.t -= DT;
            if p.t > 0.0 {
                return true;
            }
            // Never land inside the safe radius: wait until the player has moved away.
            if playing && p.pos.distance(ship) < SAFE_RADIUS {
                p.t = 0.1;
                return true;
            }
            landed.push((p.kind, p.pos));
            false
        });
        for (kind, pos) in landed {
            self.spawn(kind, pos);
            self.events.push(Event::Spawn { pos, kind });
        }
    }
}

/// The director's state.
pub struct Director {
    /// Whether it spawns (off in tests that place enemies themselves).
    pub enabled: bool,
    /// Seconds into the run (paused while the ship is dead).
    pub clock: f32,
    /// Seconds until the next formation.
    pub cooldown: f32,
    /// Intensity right now, `0..1` (for music and effects too).
    pub intensity: f32,
    /// A dip after a death or bomb, recovering.
    pub(super) relief: f32,
    /// Formations so far.
    pub(super) wave: u32,
}

impl Default for Director {
    fn default() -> Director {
        Director::new()
    }
}

impl Director {
    /// A director at the start of a run.
    pub fn new() -> Director {
        Director {
            enabled: true,
            clock: 0.0,
            cooldown: 1.5,
            intensity: 0.0,
            relief: 0.0,
            wave: 0,
        }
    }

    /// The intensity at run time `t`: a slow rise with breathing waves (calm, then a spike).
    pub fn curve(t: f32) -> f32 {
        let rise = 1.0 - (-t / 150.0).exp();
        let breath = 0.5 + 0.5 * crate::signal::wave(t * core::f32::consts::TAU / 40.0 - 1.2);
        (0.12 + 0.55 * rise + 0.33 * gax::Real::sqrt(rise) * breath * breath).clamp(0.0, 1.0)
    }

    /// Called after the ship is destroyed: a calm stretch to recover.
    pub fn after_death(&mut self) {
        self.relief = 1.0;
        self.cooldown = 2.5;
    }

    /// Called after a bomb.
    pub fn after_bomb(&mut self) {
        self.relief = self.relief.max(0.5);
        self.cooldown = self.cooldown.max(1.5);
    }

    /// Advance and announce spawns into `w`.
    pub fn update(&mut self, w: &mut World) {
        if !self.enabled || w.phase != Phase::Playing {
            return;
        }
        self.clock += DT;
        self.relief = (self.relief - DT / 8.0).max(0.0);
        // A light response to the player's state: fewer lives, slightly calmer.
        let health = match w.ledger.lives {
            1 => 0.9,
            2 => 0.96,
            _ => 1.0,
        };
        self.intensity = Director::curve(self.clock) * (1.0 - 0.6 * self.relief) * health;
        self.cooldown -= DT;
        let alive = w.enemies.len() + w.pending.len();
        let target =
            (3.0 + 55.0 * self.intensity * self.intensity + 10.0 * self.intensity) as usize;
        if self.cooldown > 0.0 || alive >= target {
            return;
        }
        self.wave += 1;
        for (k, p) in self.formation(w) {
            w.announce(k, p, k.stats().warning);
        }
        self.cooldown = 2.2 + 3.5 * (1.0 - self.intensity) + w.rng.range(0.0, 1.5);
    }

    /// The next formation: a singularity, a newcomer from the roster, or a swarm, a ring, a
    /// line or a flank of the basic families; heavier ones as intensity rises.
    fn formation(&self, w: &mut World) -> Vec<(Kind, P)> {
        let i = self.intensity;
        let roll = w.rng.unit();
        let count = |k: Kind| {
            w.enemies.iter().filter(|e| e.kind == k).count()
                + w.pending.iter().filter(|p| p.kind == k).count()
        };
        let singularities = count(Kind::Singularity);
        let features = self.features(count(Kind::Serpent) + count(Kind::Carrier));
        let rng = &mut w.rng;
        let feature_roll = rng.unit();
        let chaser_share = (0.25 + 0.6 * i).min(0.8);
        let kind = |rng: &mut Rng| {
            if rng.chance(chaser_share) {
                Kind::Chaser
            } else {
                Kind::Drifter
            }
        };
        let ship = w.ship.body.pos();
        // Which half of the arena the player is in.
        let right = ship.to_euclidean()[0] > 0.0;
        if self.clock > 35.0 && singularities < 1 + (i * 2.5) as usize && roll < 0.12 + 0.1 * i {
            // A singularity, away from the player.
            let p = at(
                rng.range(-ARENA[0] + 6.0, ARENA[0] - 6.0),
                rng.range(-ARENA[1] + 5.0, ARENA[1] - 5.0),
            );
            vec![(Kind::Singularity, p)]
        } else if !features.is_empty() && feature_roll < 0.3 + 0.25 * i {
            // A newcomer from the roster; the newest ones are a little likelier.
            let pick = ((gax::Real::sqrt(rng.unit()) * features.len() as f32) as usize)
                .min(features.len() - 1);
            feature(features[pick], i, right, rng)
        } else if roll < 0.35 {
            // Corner swarms.
            let n = 2 + (i * 5.0) as usize;
            let mut spawns = Vec::new();
            for (cx, cy) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
                if rng.chance(0.25 + 0.5 * i) || spawns.is_empty() {
                    spawns.extend((0..n).map(|_| {
                        let p = at(
                            cx * (ARENA[0] - 2.0) - cx * rng.range(0.0, 3.0),
                            cy * (ARENA[1] - 2.0) - cy * rng.range(0.0, 3.0),
                        );
                        (kind(rng), p)
                    }));
                }
            }
            spawns
        } else if roll < 0.55 && self.clock > 20.0 {
            // A ring around the player: a radius turned an n-th of a circle at a time.
            let n = 6 + (i * 10.0) as usize;
            let d = P::direction_from_angle(rng.angle()) * 13.0;
            let step = Motor::rotation(ORIGIN, core::f32::consts::TAU / n as f32);
            orbit(d, n, step)
                .map(|(d, _)| (Kind::Chaser, ship + d))
                .collect()
        } else if roll < 0.75 {
            // A line of drifters along a wall.
            let n = 5 + (i * 8.0) as usize;
            let top = rng.chance(0.5);
            let y = if top { ARENA[1] - 2.0 } else { -ARENA[1] + 2.0 };
            let ends = (at(-ARENA[0] + 3.0, y), at(ARENA[0] - 3.0, y));
            spread(ends, n).map(|p| (Kind::Drifter, p)).collect()
        } else {
            // A flank: a column on the side away from the player.
            let n = 3 + (i * 7.0) as usize;
            let x = if right {
                -ARENA[0] + 2.5
            } else {
                ARENA[0] - 2.5
            };
            let ends = (at(x, -ARENA[1] + 3.0), at(x, ARENA[1] - 3.0));
            spread(ends, n).map(|p| (kind(rng), p)).collect()
        }
    }

    /// The newcomers the roster has unlocked by now, each a "feature" formation; the heavy
    /// ones (serpents and carriers, `heavy` of them about) only while there is room.
    fn features(&self, heavy: usize) -> Vec<Kind> {
        let room = 1 + (self.intensity * 2.0) as usize;
        [
            (Kind::Evader, 20.0),
            (Kind::Splitter, 30.0),
            (Kind::Warden, 50.0),
            (Kind::Serpent, 70.0),
            (Kind::Carrier, 90.0),
        ]
        .into_iter()
        .filter(|&(kind, from)| {
            let heavy_room = !matches!(kind, Kind::Serpent | Kind::Carrier) || heavy < room;
            self.clock > from && heavy_room
        })
        .map(|(kind, _)| kind)
        .collect()
    }
}

/// A newcomer's formation at intensity `i`, on the far half of the arena from the player
/// (who is on the `right` or not).
fn feature(kind: Kind, i: f32, right: bool, rng: &mut Rng) -> Vec<(Kind, P)> {
    let far = |rng: &mut Rng| {
        let side = if right { -1.0 } else { 1.0 };
        at(
            side * rng.range(4.0, ARENA[0] - 3.0),
            rng.range(-ARENA[1] + 3.0, ARENA[1] - 3.0),
        )
    };
    match kind {
        Kind::Evader => {
            let c = far(rng);
            (0..2 + (i * 4.0) as usize)
                .map(|_| (kind, c + dir(rng.range(-3.0, 3.0), rng.range(-3.0, 3.0))))
                .collect()
        }
        Kind::Splitter => (0..2 + (i * 3.0) as usize)
            .map(|_| {
                let cx = if rng.chance(0.5) { 1.0 } else { -1.0 };
                let cy = if rng.chance(0.5) { 1.0 } else { -1.0 };
                let p = at(
                    cx * (ARENA[0] - 2.0 - rng.range(0.0, 4.0)),
                    cy * (ARENA[1] - 2.0 - rng.range(0.0, 3.0)),
                );
                (kind, p)
            })
            .collect(),
        Kind::Warden => {
            // A shield arc on the far side, facing the player, 2.5 apart.
            let n = 2 + (i * 2.5) as usize;
            let c = far(rng);
            let half = dir(0.0, (n - 1) as f32 * 1.25);
            spread((c - half, c + half), n).map(|p| (kind, p)).collect()
        }
        Kind::Serpent | Kind::Carrier => vec![(kind, far(rng))],
        _ => Vec::new(),
    }
}

/// `n` points evenly from `a` to `b`, both ends included (one: `a`).
fn spread((a, b): (P, P), n: usize) -> impl Iterator<Item = P> {
    (0..n).map(move |k| a + (b - a) * (k as f32 / (n - 1).max(1) as f32))
}

#[cfg(test)]
mod tests {
    use super::super::{Event, Input, Kind, P, SAFE_RADIUS, World};

    /// Over many seeded runs with a wandering player, no spawn ever lands within the safe
    /// radius of the ship, and the whole roster turns up.
    #[test]
    fn spawns_never_land_near_the_player() {
        let mut spawns = 0;
        let mut seen = std::collections::HashSet::new();
        for seed in 0..60 {
            let mut w = World::new(seed);
            w.ship.invulnerable = 1e9; // keep the run going
            for t in 0..120 * 150 {
                let a = t as f32 * 0.004 + seed as f32;
                w.tick(&Input {
                    movement: P::direction_from_angle(a * 1.7),
                    ..Input::default()
                });
                let ship = w.ship.body.pos();
                for e in &w.events {
                    // Motes launched by a visible carrier are its attack, not a landing.
                    if let Event::Spawn { pos, kind } = e
                        && *kind != Kind::Mote
                    {
                        spawns += 1;
                        seen.insert(*kind);
                        let d = pos.distance(ship);
                        assert!(d >= SAFE_RADIUS, "seed {seed}: a spawn {d} from the ship");
                    }
                }
            }
        }
        assert!(spawns > 1000, "only {spawns} spawns");
        for k in [
            Kind::Drifter,
            Kind::Chaser,
            Kind::Singularity,
            Kind::Evader,
            Kind::Splitter,
            Kind::Warden,
            Kind::Serpent,
            Kind::Carrier,
        ] {
            assert!(seen.contains(&k), "{k:?} never spawned");
        }
    }

    #[test]
    fn intensity_rises_with_calm_stretches() {
        let c = super::Director::curve;
        assert!(c(0.0) < 0.25);
        assert!(c(300.0) > c(30.0));
        // Breathing: within any minute after the start there is a calmer moment.
        for start in [60.0, 120.0, 240.0] {
            let v: Vec<f32> = (0..60).map(|s| c(start + s as f32)).collect();
            let (lo, hi) = v
                .iter()
                .fold((1.0f32, 0.0f32), |(a, b), &x| (a.min(x), b.max(x)));
            assert!(hi - lo > 0.1, "{start}: {lo}..{hi}");
        }
    }
}
