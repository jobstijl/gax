//! The enemies' tick: drones in their rings and dives, mines drifting, turrets firing,
//! serpents swimming their screws, singularities holding and feeding; and kills.

use super::entities::{Enemy, Event, Flight, Foe, Shard, Shot};
use super::lattice::RADIUS;
use super::space::{P, arc, around, dir, foot, off_axis, radial};
use super::wells::Wells;
use super::{SHIP_RADIUS, World};
use crate::geom::clamp_length;
use crate::sim::{DT, Phase};
use gax::pga3d::Plane;

/// A turret's bolts' speed along the track, relative to the ship.
pub(super) const BOLT_SPEED: f32 = 13.0;
/// A singularity bursts at this mass.
const WELL_BURST: f32 = 5.0;
/// How fast a ring of drones turns (radians per second).
const RING_TURN: f32 = 0.9;

/// What an enemy sees of the tunnel in a tick.
struct View<'a> {
    /// The ship's position, its arc length and its speed.
    ship: P,
    s_ship: f32,
    speed: f32,
    /// Whether the ship is there to fight.
    alive: bool,
    /// The singularities: each one's id, where it is and how it moves (their escorts' rings
    /// move with them).
    singularities: &'a [(u32, P, P)],
    /// Their pull.
    wells: &'a Wells,
}

impl View<'_> {
    /// Moving along with the ship.
    fn with_ship(&self) -> P {
        dir(0.0, 0.0, self.speed)
    }
}

/// Towards `place` from `from`, capped so that it glides in (on top of the motion it follows).
fn glide(from: P, place: P, cap: f32) -> P {
    clamp_length((place - from) * 2.5, cap)
}

impl World {
    /// Every enemy's tick, in the singularities' pull `wells`.
    pub(super) fn step_enemies(&mut self, wells: &Wells) {
        let alive = self.phase == Phase::Playing;
        // One drone dives at a time: the one that has held longest (escorts stay with their
        // singularity).
        self.dive_cooldown -= DT;
        if alive && self.dive_cooldown <= 0.0 {
            let holding = self
                .enemies
                .iter_mut()
                .filter(|e| e.foe == Foe::Drone && e.flight == Flight::Hold && e.anchor.is_none())
                .max_by(|a, b| a.age.total_cmp(&b.age));
            if let Some(e) = holding {
                e.flight = Flight::Dive;
                e.timer = 0.7;
                self.events.push(Event::Dive { pos: e.pos });
                self.dive_cooldown = 1.6 - 0.8 * self.intensity;
            }
        }
        let singularities: Vec<(u32, P, P)> = self
            .enemies
            .iter()
            .filter(|e| e.foe == Foe::Singularity)
            .map(|e| (e.id, e.pos, e.vel))
            .collect();
        let view = View {
            ship: self.ship.pos,
            s_ship: self.ship.s(),
            speed: self.ship.speed,
            alive,
            singularities: &singularities,
            wells,
        };
        let fired: Vec<(P, P)> = self
            .enemies
            .iter_mut()
            .filter_map(|e| e.step(&view))
            .collect();
        for (pos, vel) in fired {
            self.bolts.push(Shot::new(pos, vel, 8.0));
            self.events.push(Event::Bolt { pos });
        }
        self.feed();
        // Behind the ship: gone.
        let s = self.ship.s();
        self.enemies.retain(|e| arc(e.pos) > s - 6.0);
    }

    /// Singularities eat the mines and loose drones that fall in, and burst when overfed.
    fn feed(&mut self) {
        let mut eaten: Vec<(usize, usize)> = Vec::new();
        for (wi, w) in self.enemies.iter().enumerate() {
            if w.foe != Foe::Singularity {
                continue;
            }
            for (i, e) in self.enemies.iter().enumerate() {
                let food = e.foe == Foe::Mine || (e.foe == Foe::Drone && e.anchor.is_none());
                if food && e.age > 0.3 && e.pos.distance(w.pos) < w.radius() {
                    eaten.push((wi, i));
                }
            }
        }
        let mut gone: Vec<usize> = Vec::new();
        for (wi, i) in eaten {
            if gone.contains(&i) {
                continue;
            }
            gone.push(i);
            let w = &mut self.enemies[wi];
            w.mass += 1.0;
            w.hp += 3.0;
            w.flash = 1.0;
            let (pos, mass) = (w.pos, w.mass);
            self.events.push(Event::Absorb { pos, mass });
        }
        gone.sort_unstable();
        for &i in gone.iter().rev() {
            self.enemies.swap_remove(i);
        }
        let Some(i) = self
            .enemies
            .iter()
            .position(|e| e.foe == Foe::Singularity && e.mass >= WELL_BURST)
        else {
            return;
        };
        // Overfed: it bursts into a ring of drones.
        let w = self.enemies.swap_remove(i);
        self.events.push(Event::Burst { pos: w.pos });
        self.blast(w.pos, 1800.0, 6.0, 0.25);
        for k in 0..8 {
            let a = k as f32 * core::f32::consts::TAU / 8.0;
            let pos = w.pos + radial(1.0, a);
            let id = self.next_id();
            self.enemies.push(Enemy {
                flash: 1.0,
                angle: a,
                orbit: 4.8,
                depth: 9.0,
                timer: 1.0,
                ..Enemy::new(id, Foe::Drone, pos)
            });
        }
    }

    /// Enemy `i` destroyed: by the player (`scored`, with a shot bent round a singularity if
    /// `bent`) or by a bomb. It blasts the wall, a serpent's body goes up with it, a
    /// singularity collapses, and a scored kill drops shards.
    pub(super) fn kill(&mut self, i: usize, scored: bool, bent: bool) {
        let e = self.enemies.swap_remove(i);
        let sling = if bent { 2.0 } else { 1.0 };
        let points = if scored {
            (e.foe.points() as f32
                * (1.0 + e.mass)
                * self.ledger.mult as f32
                * self.speed_bonus()
                * sling)
                .round() as u64
        } else {
            0
        };
        self.ledger.award(points);
        self.events.push(Event::Kill {
            pos: e.pos,
            foe: e.foe,
            points,
        });
        if bent && scored {
            self.events.push(Event::Slingshot { pos: e.pos });
        }
        let size = match e.foe {
            Foe::Turret | Foe::Serpent => 700.0,
            Foe::Singularity => 2600.0,
            _ => 260.0,
        };
        self.blast(e.pos, size, 3.0, 0.12);
        match e.foe {
            // The head gone, the body goes up segment by segment.
            Foe::Serpent => {
                for s in &e.body {
                    self.events.push(Event::Kill {
                        pos: s.pos,
                        foe: Foe::Serpent,
                        points: 0,
                    });
                    self.blast(s.pos, 200.0, 3.0, 0.1);
                }
            }
            // The collapse: the wall let go all round, a ring of shock along the tunnel.
            Foe::Singularity => {
                let s = arc(e.pos);
                for k in 0..12 {
                    let a = k as f32 * core::f32::consts::TAU / 12.0;
                    self.blast(around(RADIUS - 1.0, a, s), 1400.0, 6.0, 0.3);
                }
            }
            _ => {}
        }
        if scored {
            self.drop_shards(&e);
        }
    }

    /// The shards a scored kill leaves: one, or a ring of them for the big ones.
    fn drop_shards(&mut self, e: &Enemy) {
        let n = match e.foe {
            Foe::Serpent => 3,
            Foe::Singularity => 4 + e.mass as usize,
            _ => 1,
        };
        for k in 0..n {
            let off = if n > 1 {
                radial(1.2, k as f32 * core::f32::consts::TAU / n as f32)
            } else {
                P::zero()
            };
            self.shards.push(Shard {
                pos: e.pos + off,
                life: 6.0,
            });
        }
    }
}

impl Enemy {
    /// One tick: its family's flight, then the step. A turret that fires returns its bolt:
    /// where from, and at what velocity.
    fn step(&mut self, view: &View) -> Option<(P, P)> {
        self.prev = self.pos;
        self.age += DT;
        self.flash = (self.flash - DT * 6.0).max(0.0);
        let bolt = match self.foe {
            Foe::Drone => {
                self.fly_drone(view);
                None
            }
            Foe::Mine => {
                self.drift(view);
                None
            }
            Foe::Turret => self.fire(view),
            Foe::Serpent => {
                self.swim();
                None
            }
            Foe::Singularity => {
                self.hold(view);
                None
            }
        };
        self.pos += self.vel * DT;
        bolt
    }

    /// A drone: to its place in its ring, holding there, and diving at the ship.
    fn fly_drone(&mut self, view: &View) {
        self.angle += DT * RING_TURN;
        // An escort's ring turns about its singularity; a lost escort joins the ship's band.
        let well = self
            .anchor
            .and_then(|id| view.singularities.iter().find(|w| w.0 == id));
        if self.anchor.is_some() && well.is_none() {
            self.anchor = None;
            self.orbit = 4.6;
            self.depth = 8.0;
            self.flight = Flight::Approach;
        }
        // Its place moves with the ship (or the singularity), and turns with the ring: along
        // its radial offset turned a quarter about the axis, at the ring's rate. Gliding with
        // both motions, a drone reaches its place instead of trailing it.
        let quarter = core::f32::consts::FRAC_PI_2;
        let turning = radial(self.orbit, self.angle + quarter) * RING_TURN;
        let (place, place_vel) = match well {
            Some(&(_, w, v)) => (w + radial(self.orbit, self.angle), v + turning),
            // Its place in the ring, which holds `depth` ahead of the ship.
            None => (
                around(self.orbit, self.angle, view.s_ship + self.depth),
                view.with_ship() + turning,
            ),
        };
        self.vel = match self.flight {
            Flight::Approach | Flight::Hold => {
                if self.flight == Flight::Approach && place.distance(self.pos) < 1.5 {
                    self.flight = Flight::Hold;
                }
                glide(self.pos, place, 26.0) + place_vel
            }
            Flight::Dive => {
                self.timer -= DT;
                if self.timer > 0.0 {
                    // The warning: it flashes in place.
                    self.flash = self.flash.max(0.6);
                    (place - self.pos) * 2.5 + place_vel
                } else {
                    // At the ship, at a speed you can dodge.
                    let to = view.ship - self.pos;
                    to * (10.0 / to.ideal_norm().max(1e-3)) + view.with_ship() * 0.7
                }
            }
        };
    }

    /// A mine: drifting, bouncing off the tunnel's inner radius (a reflection in the plane
    /// tangent to it there), pulled by the singularities.
    fn drift(&mut self, view: &View) {
        let outward = off_axis(self.pos + self.vel * DT) > off_axis(self.pos);
        if off_axis(self.pos) > SHIP_RADIUS + 0.3 && outward {
            let tangent = Plane::orthogonal_to(self.pos - foot(self.pos));
            self.vel = tangent.reflect(self.vel);
        }
        self.vel += view.wells.pull(self.pos, 20.0) * DT;
    }

    /// A turret: every 1.5 s, a bolt aimed where the ship will be when it arrives (you read
    /// it by its growth and its colour), if the ship is in range ahead.
    fn fire(&mut self, view: &View) -> Option<(P, P)> {
        self.timer -= DT;
        if self.timer > 0.0 {
            return None;
        }
        self.timer = 1.5;
        let ahead = arc(self.pos) - view.s_ship;
        (view.alive && (12.0..62.0).contains(&ahead)).then(|| {
            let t = ahead / (view.speed + BOLT_SPEED);
            let there = view.ship + dir(0.0, 0.0, view.speed * t);
            (self.pos, (there - self.pos) * (1.0 / t))
        })
    }

    /// A serpent: the head along its screw, and each segment the head moved back by its own
    /// lag.
    fn swim(&mut self) {
        let next = self.swim >> self.pos;
        self.vel = (next - self.pos) * (1.0 / DT);
        for seg in &mut self.body {
            seg.prev = seg.pos;
            seg.pos = seg.behind >> next;
        }
    }

    /// A singularity: holding ahead of the ship, turning slowly about the axis; then let go,
    /// so that the ship must fly past it.
    fn hold(&mut self, view: &View) {
        self.angle += DT * 0.3;
        let place = around(self.orbit, self.angle, view.s_ship + self.depth);
        self.vel = match self.flight {
            Flight::Approach => {
                if place.distance(self.pos) < 1.5 {
                    self.flight = Flight::Hold;
                }
                glide(self.pos, place, 26.0) + view.with_ship()
            }
            Flight::Hold => {
                self.timer -= DT;
                if self.timer <= 0.0 {
                    self.flight = Flight::Dive;
                }
                glide(self.pos, place, 26.0) + view.with_ship()
            }
            Flight::Dive => P::zero(),
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tunnel::director::{SEGMENT_LAG, SEGMENTS};
    use crate::tunnel::tests::spawn;
    use crate::tunnel::{Input, World, at};

    /// Drones come in, hold a ring ahead of the ship, and dive one at a time.
    #[test]
    fn drones_hold_ahead_and_dive_one_by_one() {
        let mut w = World::new(5);
        w.ship.invulnerable = 1e9;
        let mut held = false;
        let mut dives = Vec::new();
        for t in 0..120 * 20 {
            w.tick(&Input::default());
            let s = w.ship.s();
            held |= w
                .enemies
                .iter()
                .any(|e| e.flight == Flight::Hold && (4.0..14.0).contains(&(arc(e.pos) - s)));
            for e in &w.events {
                if let Event::Dive { .. } = e {
                    dives.push(t);
                }
            }
        }
        assert!(held, "no drone held in the band");
        assert!(dives.len() > 3, "{} dives", dives.len());
        assert!(
            dives.windows(2).all(|w| w[1] - w[0] > 60),
            "dives not one by one"
        );
    }

    /// The seconds until drone `i`, approaching its place, holds there and then (the dives
    /// let loose) dives, if within `limit`.
    fn hold_and_dive(w: &mut World, i: usize, limit: f32) -> Option<(f32, f32)> {
        let id = w.enemies[i].id;
        let mut held = None;
        for t in 1..=(limit / DT) as usize {
            w.tick(&Input::default());
            let e = w.enemies.iter().find(|e| e.id == id)?;
            if held.is_none() && e.flight == Flight::Hold {
                held = Some(t as f32 * DT);
                w.dive_cooldown = 0.0;
            }
            if let Some(h) = held
                && w.events.iter().any(|ev| matches!(ev, Event::Dive { .. }))
            {
                return Some((h, t as f32 * DT));
            }
        }
        None
    }

    /// A quiet world for one drone: no spawns, no slaloms, no dives until let loose.
    fn quiet(seed: u64) -> World {
        let mut w = World::new(seed);
        w.spawn_cooldown = 1e9;
        w.gate_timer = 1e9;
        w.dive_cooldown = 1e9;
        w.ship.invulnerable = 1e9;
        w
    }

    /// An escort whose singularity is gone joins the ship's band of drones, holds its place
    /// in the ring there, and dives like the others.
    #[test]
    fn an_escort_that_loses_its_singularity_rejoins_the_band() {
        let mut w = quiet(4);
        let s = w.ship.s();
        let well = spawn(&mut w, Foe::Singularity, at(1.0, 0.0, s + 25.0));
        let id = w.enemies[well].id;
        let escort = spawn(&mut w, Foe::Drone, at(4.4, 0.0, s + 25.0));
        w.enemies[escort].anchor = Some(id);
        w.enemies[escort].orbit = 3.4;
        w.tick(&Input::default());
        assert_eq!(w.enemies[escort].anchor, Some(id));
        w.enemies.retain(|e| e.foe != Foe::Singularity);
        w.tick(&Input::default());
        let e = &w.enemies[0];
        assert_eq!((e.anchor, e.flight), (None, Flight::Approach));
        assert_eq!((e.orbit, e.depth), (4.6, 8.0));
        let (held, dove) = hold_and_dive(&mut w, 0, 6.0).expect("never held and dove");
        assert!(
            held < 3.0 && dove < held + 0.1,
            "held at {held} s, dove at {dove} s"
        );
    }

    /// Drones in rings of every width the director makes come to their places, hold, and
    /// dive.
    #[test]
    fn drones_of_every_ring_width_hold_and_dive() {
        for orbit in [4.0f32, 4.6, 5.3] {
            let mut w = quiet(5);
            let s = w.ship.s();
            let i = spawn(&mut w, Foe::Drone, around(orbit, 0.0, s + 30.0));
            let e = &mut w.enemies[i];
            (e.flight, e.orbit, e.depth, e.angle) = (Flight::Approach, orbit, 8.0, 0.0);
            let (held, _) = hold_and_dive(&mut w, i, 6.0).expect("never held and dove");
            assert!(held < 3.0, "orbit {orbit}: held at {held} s");
        }
    }

    /// A serpent's body is the helix its head has swum: every segment stays at the head's
    /// distance from the axis, behind it along the track, and where the head was `lag`
    /// seconds ago.
    #[test]
    fn a_serpent_swims_a_screw() {
        let mut w = World::new(2);
        w.spawn_cooldown = 1e9;
        w.gate_timer = 1e9;
        let i = spawn(&mut w, Foe::Serpent, around(4.0, 0.3, 60.0));
        let mut trail = vec![w.enemies[i].pos];
        for _ in 0..240 {
            w.tick(&Input::default());
            trail.push(w.enemies[0].pos);
        }
        let e = &w.enemies[0];
        assert_eq!(e.body.len(), SEGMENTS);
        for (k, seg) in e.body.iter().enumerate() {
            assert!(
                (off_axis(seg.pos) - 4.0).abs() < 1e-2,
                "{}",
                off_axis(seg.pos)
            );
            assert!(arc(seg.pos) < arc(e.pos));
            // Where the head was `lag` seconds ago.
            let lag = (k + 1) as f32 * SEGMENT_LAG;
            let then = trail[trail.len() - 1 - (lag / DT).round() as usize];
            assert!(then.distance(seg.pos) < 0.05, "{}", then.distance(seg.pos));
        }
    }
}
