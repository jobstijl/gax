//! The spawn director: an intensity curve over time with calm stretches between spikes, and
//! formations that are always telegraphed and never land within the safe radius.

use super::{ARENA, Kind, Phase, World};

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
    relief: f32,
    wave: u32,
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
        let breath = 0.5 + 0.5 * (t * core::f32::consts::TAU / 40.0 - 1.2).sin();
        (0.12 + 0.55 * rise + 0.33 * rise.sqrt() * breath * breath).clamp(0.0, 1.0)
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
        let dt = super::DT;
        self.clock += dt;
        self.relief = (self.relief - dt / 8.0).max(0.0);
        // A light response to the player's state: fewer lives, slightly calmer.
        let health = match w.lives {
            1 => 0.9,
            2 => 0.96,
            _ => 1.0,
        };
        self.intensity = Director::curve(self.clock) * (1.0 - 0.6 * self.relief) * health;
        self.cooldown -= dt;
        let alive = w.enemies.len() + w.pending.len();
        let target =
            (3.0 + 55.0 * self.intensity * self.intensity + 10.0 * self.intensity) as usize;
        if self.cooldown > 0.0 || alive >= target {
            return;
        }
        self.wave += 1;
        let i = self.intensity;
        let rng = &mut w.rng;
        // Pick a formation; heavier ones as intensity rises.
        let roll = rng.unit();
        let singularities = w
            .enemies
            .iter()
            .filter(|e| e.kind == Kind::Singularity)
            .count()
            + w.pending
                .iter()
                .filter(|p| p.kind == Kind::Singularity)
                .count();
        let chaser_share = (0.25 + 0.6 * i).min(0.8);
        let kind = |rng: &mut super::rng::Rng| {
            if rng.chance(chaser_share) {
                Kind::Chaser
            } else {
                Kind::Drifter
            }
        };
        let ship = w.ship.body.xy();
        let mut spawns: Vec<(Kind, [f32; 2])> = Vec::new();
        if self.clock > 35.0 && singularities < 1 + (i * 2.5) as usize && roll < 0.12 + 0.1 * i {
            // A singularity, away from the player.
            let p = [
                rng.range(-ARENA[0] + 6.0, ARENA[0] - 6.0),
                rng.range(-ARENA[1] + 5.0, ARENA[1] - 5.0),
            ];
            spawns.push((Kind::Singularity, p));
        } else if roll < 0.35 {
            // Corner swarms.
            let n = 2 + (i * 5.0) as usize;
            for (cx, cy) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
                if rng.chance(0.25 + 0.5 * i) || spawns.is_empty() {
                    for _ in 0..n {
                        let p = [
                            cx * (ARENA[0] - 2.0) - cx * rng.range(0.0, 3.0),
                            cy * (ARENA[1] - 2.0) - cy * rng.range(0.0, 3.0),
                        ];
                        spawns.push((kind(rng), p));
                    }
                }
            }
        } else if roll < 0.55 && self.clock > 20.0 {
            // A ring around the player.
            let n = 6 + (i * 10.0) as usize;
            let r = 13.0;
            let a0 = rng.angle();
            for k in 0..n {
                let a = a0 + k as f32 * core::f32::consts::TAU / n as f32;
                spawns.push((Kind::Chaser, [ship[0] + r * a.cos(), ship[1] + r * a.sin()]));
            }
        } else if roll < 0.75 {
            // A line of drifters along a wall.
            let n = 5 + (i * 8.0) as usize;
            let top = rng.chance(0.5);
            let y = if top { ARENA[1] - 2.0 } else { -ARENA[1] + 2.0 };
            for k in 0..n {
                let x = -ARENA[0] + 3.0 + (2.0 * ARENA[0] - 6.0) * k as f32 / (n - 1).max(1) as f32;
                spawns.push((Kind::Drifter, [x, y]));
            }
        } else {
            // A flank: a column on the side away from the player.
            let n = 3 + (i * 7.0) as usize;
            let x = if ship[0] > 0.0 {
                -ARENA[0] + 2.5
            } else {
                ARENA[0] - 2.5
            };
            for k in 0..n {
                let y = -ARENA[1] + 3.0 + (2.0 * ARENA[1] - 6.0) * k as f32 / (n - 1).max(1) as f32;
                spawns.push((kind(rng), [x, y]));
            }
        }
        for (k, p) in spawns {
            let t = if k == Kind::Singularity { 1.4 } else { 0.8 };
            w.announce(k, p, t);
        }
        self.cooldown = 2.2 + 3.5 * (1.0 - i) + w.rng.range(0.0, 1.5);
    }
}

#[cfg(test)]
mod tests {
    use super::super::{Event, Input, SAFE_RADIUS, World};
    use gax::pga2d::Point;

    /// Over many seeded runs with a wandering player, no spawn ever lands within the safe
    /// radius of the ship.
    #[test]
    fn spawns_never_land_near_the_player() {
        let mut spawns = 0;
        for seed in 0..60 {
            let mut w = World::new(seed);
            w.ship.invulnerable = 1e9; // keep the run going
            for t in 0..120 * 90 {
                let a = t as f32 * 0.004 + seed as f32;
                w.tick(&Input {
                    movement: Point::direction(a.cos(), (a * 1.7).sin()),
                    ..Input::default()
                });
                let ship = w.ship.body.xy();
                for e in &w.events {
                    if let Event::Spawn { pos, .. } = e {
                        spawns += 1;
                        let d = ((pos[0] - ship[0]).powi(2) + (pos[1] - ship[1]).powi(2)).sqrt();
                        assert!(d >= SAFE_RADIUS, "seed {seed}: a spawn {d} from the ship");
                    }
                }
            }
        }
        assert!(spawns > 1000, "only {spawns} spawns");
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
