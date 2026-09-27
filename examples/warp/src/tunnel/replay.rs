//! Tunnel replays: the Plane's container (`sim::replay`) with the Tunnel's input record.
//!
//! The aim is a point in the tunnel, which moves with the ship: it is kept relative to the
//! ship's arc length at the tick (across in sixteenths, ahead in halves), so the same record
//! unpacks to the same point in play and in playback.

use super::{Input, Phase, World, arc, at, dir};
use crate::sim::replay::{Desync, Record, Replay};

/// One tick's input, packed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Packed {
    mx: i8,
    my: i8,
    ax: i8,
    ay: i8,
    ahead: u8,
    throttle: i8,
    flags: u8,
}

const FIRE: u8 = 1;
const BOMB: u8 = 2;
const ROLL_LEFT: u8 = 4;
const ROLL_RIGHT: u8 = 8;

impl Packed {
    /// Pack an input at the ship's arc length `s` (lossy: the simulation sees `unpack`).
    pub fn pack(i: &Input, s: f32) -> Packed {
        let q = |x: f32, k: f32| (x * k).round().clamp(-127.0, 127.0) as i8;
        let (mut mx, mut my) = (q(i.movement.e032(), 127.0), q(i.movement.e013(), 127.0));
        // Stay within the unit disc after rounding: step the larger component inward.
        while i32::from(mx).pow(2) + i32::from(my).pow(2) > 127 * 127 {
            if mx.unsigned_abs() >= my.unsigned_abs() {
                mx -= mx.signum();
            } else {
                my -= my.signum();
            }
        }
        let aim = i.aim.unitized();
        Packed {
            mx,
            my,
            ax: q(aim.e032(), 16.0),
            ay: q(aim.e013(), 16.0),
            ahead: ((arc(aim) - s) * 2.0).round().clamp(0.0, 255.0) as u8,
            throttle: q(i.throttle.clamp(-1.0, 1.0), 127.0),
            flags: (u8::from(i.fire) * FIRE)
                | (u8::from(i.bomb) * BOMB)
                | (u8::from(i.roll < 0) * ROLL_LEFT)
                | (u8::from(i.roll > 0) * ROLL_RIGHT),
        }
    }

    /// The input the simulation sees, at the ship's arc length `s`.
    pub fn unpack(self, s: f32) -> Input {
        let f = |x: i8, k: f32| f32::from(x) / k;
        Input {
            movement: dir(f(self.mx, 127.0), f(self.my, 127.0), 0.0),
            aim: at(
                f(self.ax, 16.0),
                f(self.ay, 16.0),
                s + f32::from(self.ahead) * 0.5,
            ),
            fire: self.flags & FIRE != 0,
            bomb: self.flags & BOMB != 0,
            roll: if self.flags & ROLL_LEFT != 0 {
                -1
            } else if self.flags & ROLL_RIGHT != 0 {
                1
            } else {
                0
            },
            throttle: f(self.throttle, 127.0),
        }
    }
}

impl Record for Packed {
    const MAGIC: &'static [u8; 8] = b"WARPTUN1";
    const TABLE: &'static str = "tunnel-scores.txt";
    const EXT: &'static str = "tunnel";
    const SIZE: usize = 7;

    fn write(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&[
            self.mx as u8,
            self.my as u8,
            self.ax as u8,
            self.ay as u8,
            self.ahead,
            self.throttle as u8,
            self.flags,
        ]);
    }

    fn read(b: &[u8]) -> Packed {
        Packed {
            mx: b[0] as i8,
            my: b[1] as i8,
            ax: b[2] as i8,
            ay: b[3] as i8,
            ahead: b[4],
            throttle: b[5] as i8,
            flags: b[6],
        }
    }
}

/// A hash of the Tunnel's state (FNV-1a over the bits that matter).
pub fn hash(w: &World) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut u = |x: u64| {
        for b in x.to_le_bytes() {
            h = (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3);
        }
    };
    u(w.tick);
    u(w.rng.state());
    u(w.score);
    u(u64::from(w.mult) << 32 | u64::from(w.lives) << 16 | u64::from(w.bombs));
    u(u64::from(w.chain));
    u(match w.phase {
        Phase::Playing => 1,
        Phase::Dead(t) => 2 ^ u64::from(t.to_bits()) << 8,
        Phase::Over => 3,
    });
    let mut fs = |xs: &[f32]| {
        for x in xs {
            u(u64::from(x.to_bits()));
        }
    };
    fs(&w.ship.pos.c);
    fs(&[w.ship.speed]);
    for e in &w.enemies {
        fs(&e.pos.c);
        fs(&[e.hp, e.mass, e.timer, e.id as f32]);
        for s in &e.body {
            fs(&s.pos.c);
        }
    }
    for b in w.shots.iter().chain(&w.bolts) {
        fs(&b.pos.c);
        fs(&b.vel.c);
    }
    for g in &w.gates {
        fs(&g.pos.c);
    }
    fs(&[w.lattice.radius(w.ship.s() + 20.0)]);
    h
}

impl Replay<Packed> {
    /// Record a tick: pack the input at the ship's arc length, and return what the world
    /// must see.
    pub fn take(&mut self, input: &Input, w: &World) -> Input {
        let p = Packed::pack(input, w.ship.s());
        self.record(p);
        p.unpack(w.ship.s())
    }

    /// After a recorded tick: keep a hash every second, and the score.
    pub fn after(&mut self, w: &World) {
        self.keep(w.tick, w.score, || hash(w));
    }

    /// Check a world being played back against the recorded hash, if one falls on its tick.
    pub fn check(&self, w: &World) -> Result<(), Desync> {
        self.compare(w.tick, || hash(w))
    }

    /// Play the whole replay headless, checking every hash; returns the final world.
    pub fn verify(&self) -> Result<World, Desync> {
        let mut w = World::new(self.seed);
        for p in &self.inputs {
            let i = p.unpack(w.ship.s());
            w.tick(&i);
            self.check(&w)?;
        }
        self.final_score(w.score)?;
        Ok(w)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tunnel::{AIM_DEPTH, about_axis};

    /// A flight with weaving, aiming at the nearest enemy, fire in bursts, rolls, throttle
    /// and a bomb: recorded, it replays exactly, also through the file form.
    #[test]
    fn tunnel_replays_are_deterministic() {
        let seed = 21;
        let mut w = World::new(seed);
        let mut rec: Replay<Packed> = Replay::new(seed);
        for t in 0..120 * 70u64 {
            let s = w.ship.s();
            let aim = w
                .enemies
                .iter()
                .filter(|e| arc(e.pos) > s + 5.0)
                .min_by(|a, b| arc(a.pos).total_cmp(&arc(b.pos)))
                .map_or(at(0.0, 0.0, s + AIM_DEPTH), |e| e.pos);
            let a = t as f32 * 0.009;
            let input = Input {
                movement: about_axis(a) >> dir(0.7, 0.0, 0.0),
                aim,
                fire: (t / 100) % 3 != 2,
                bomb: t == 5000,
                roll: if t % 900 == 450 { 1 } else { 0 },
                throttle: crate::signal::wave(a),
            };
            let i = rec.take(&input, &w);
            w.tick(&i);
            rec.after(&w);
        }
        assert!(w.score > 0 && rec.hashes.len() == 70);
        let back = Replay::<Packed>::decode(&rec.encode()).unwrap();
        assert_eq!(back, rec);
        let v = back.verify().unwrap_or_else(|d| panic!("{d:?}"));
        assert_eq!(hash(&v), hash(&w));
    }

    /// The aim point survives packing to within the record's resolution.
    #[test]
    fn the_aim_is_kept_relative_to_the_ship() {
        let i = Input {
            aim: at(3.3, -2.1, 140.7),
            ..Input::default()
        };
        let p = Packed::pack(&i, 100.0).unpack(100.0);
        assert!((p.aim & i.aim).norm() < 0.3, "{:?}", p.aim.to_euclidean());
        assert_eq!(Packed::pack(&p, 100.0).unpack(100.0), p);
    }
}
