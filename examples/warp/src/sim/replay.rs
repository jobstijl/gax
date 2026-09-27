//! Replays: a run is its seed and the input of every tick.
//!
//! Inputs are quantized before the simulation sees them (`quantize`), in play as in playback, so
//! a replay reproduces its run bit for bit on the same build. A hash of the world every second
//! of simulation catches any divergence at the second it happens. The container is the same
//! for both games; each has its own input record (`Record`): the Plane's here, the Tunnel's in
//! `tunnel::replay`.

use super::{Input, Phase, World};
use gax::pga2d::Point;

/// Ticks between state hashes (one second).
pub const HASH_EVERY: u64 = 120;

/// A replay's input record: one tick's input as a replay keeps it.
pub trait Record: Copy + Default + PartialEq + core::fmt::Debug {
    /// The file's magic: which game.
    const MAGIC: &'static [u8; 8];
    /// The high-score table's file, and the replays' extension.
    const TABLE: &'static str;
    const EXT: &'static str;
    /// Bytes per record.
    const SIZE: usize;
    /// Append the bytes.
    fn write(&self, out: &mut Vec<u8>);
    /// Read `SIZE` bytes.
    fn read(b: &[u8]) -> Self;
}

impl Record for Packed {
    const MAGIC: &'static [u8; 8] = b"WARPRPL1";
    const TABLE: &'static str = "scores.txt";
    const EXT: &'static str = "warp";
    const SIZE: usize = 5;

    fn write(&self, out: &mut Vec<u8>) {
        out.push(self.mx as u8);
        out.push(self.my as u8);
        out.extend_from_slice(&self.aim.to_le_bytes());
        out.push(self.flags);
    }

    fn read(b: &[u8]) -> Packed {
        Packed {
            mx: b[0] as i8,
            my: b[1] as i8,
            aim: u16::from_le_bytes([b[2], b[3]]),
            flags: b[4],
        }
    }
}

/// The build a replay was recorded on: the game's version and a hash of the simulation's
/// sources (set by `build.rs`). Replays from another build may diverge.
pub fn build_id() -> String {
    format!("{}+{}", env!("CARGO_PKG_VERSION"), env!("WARP_SIM_HASH"))
}

/// One tick's input, packed: movement in 1/127ths, the aim as an angle, and buttons.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Packed {
    mx: i8,
    my: i8,
    aim: u16,
    flags: u8,
}

const FIRE: u8 = 1;
const BOMB: u8 = 2;
const AIMING: u8 = 4;

impl Packed {
    /// Pack an input (lossy: see `quantize`).
    pub fn pack(i: &Input) -> Packed {
        let q = |x: f32| (x.clamp(-1.0, 1.0) * 127.0).round() as i8;
        let (mut mx, mut my) = (q(i.movement.e20()), q(i.movement.e01()));
        // Stay within the unit disc after rounding: step the larger component inward.
        while i32::from(mx).pow(2) + i32::from(my).pow(2) > 127 * 127 {
            if mx.unsigned_abs() >= my.unsigned_abs() {
                mx -= mx.signum();
            } else {
                my -= my.signum();
            }
        }
        // The aim's angle: of the rotation from the x axis to it.
        let aiming = i.aim.ideal_norm() > 0.2;
        let angle = super::body::angle_of(i.aim);
        let turn = (angle / core::f32::consts::TAU).rem_euclid(1.0);
        Packed {
            mx,
            my,
            aim: if aiming {
                (turn * 65536.0).round() as u32 as u16
            } else {
                0
            },
            flags: (u8::from(i.fire) * FIRE)
                | (u8::from(i.bomb) * BOMB)
                | (u8::from(aiming) * AIMING),
        }
    }

    /// The input the simulation sees.
    pub fn unpack(self) -> Input {
        let (mx, my) = (f32::from(self.mx) / 127.0, f32::from(self.my) / 127.0);
        let aim = if self.flags & AIMING != 0 {
            super::body::heading(f32::from(self.aim) / 65536.0 * core::f32::consts::TAU, 1.0)
        } else {
            Point::direction(0.0, 0.0)
        };
        Input {
            movement: Point::direction(mx, my),
            aim,
            fire: self.flags & FIRE != 0,
            bomb: self.flags & BOMB != 0,
        }
    }
}

/// The input as the simulation will see it: what a replay can reproduce exactly.
#[cfg(test)]
pub fn quantize(i: &Input) -> Input {
    Packed::pack(i).unpack()
}

/// A hash of the world's state (FNV-1a over the bits that matter).
pub fn hash(w: &World) -> u64 {
    let mut h = Fnv(0xcbf2_9ce4_8422_2325);
    h.u(w.tick);
    h.u(w.rng.state());
    h.u(w.score);
    h.u(u64::from(w.mult) << 32 | u64::from(w.lives) << 16 | u64::from(w.bombs));
    h.u(match w.phase {
        Phase::Playing => 1,
        Phase::Dead(t) => 2 ^ u64::from(t.to_bits()) << 8,
        Phase::Over => 3,
    });
    h.fs(&w.ship.body.pose.into_inner().c);
    h.fs(&w.ship.body.vel.c);
    for e in &w.enemies {
        h.u(u64::from(e.id) << 8 | e.kind as u64);
        h.fs(&e.body.pose.into_inner().c);
        h.fs(&[e.hp, e.mass, e.timer]);
        for p in &e.chain {
            h.fs(&p.into_inner().c);
        }
    }
    for b in &w.bullets {
        h.fs(&b.pos.c);
        h.fs(&b.vel.c);
    }
    for s in &w.shards {
        h.fs(&s.body.pose.into_inner().c);
    }
    for p in &w.pending {
        h.u(p.kind as u64);
        h.fs(&p.pos.c);
        h.fs(&[p.t]);
    }
    h.0
}

struct Fnv(u64);

impl Fnv {
    fn u(&mut self, x: u64) {
        for b in x.to_le_bytes() {
            self.0 = (self.0 ^ u64::from(b)).wrapping_mul(0x100_0000_01b3);
        }
    }

    fn fs(&mut self, xs: &[f32]) {
        for x in xs {
            self.u(u64::from(x.to_bits()));
        }
    }
}

/// A recorded run.
#[derive(Clone, Debug, PartialEq)]
pub struct Replay<I: Record = Packed> {
    /// The build it was recorded on (`build_id`).
    pub build: String,
    /// The world's seed.
    pub seed: u64,
    /// The final score.
    pub score: u64,
    /// Every tick's input.
    pub inputs: Vec<I>,
    /// The world's hash after every `HASH_EVERY` ticks.
    pub hashes: Vec<u64>,
}

/// Where a playback went wrong.
#[derive(Debug, PartialEq, Eq)]
pub enum Desync {
    /// The state hash differs after this many ticks.
    Hash(u64),
    /// The run ended with another score.
    Score { recorded: u64, replayed: u64 },
}

impl Replay<Packed> {
    /// After a recorded tick: keep a hash every second, and the score.
    pub fn after(&mut self, w: &World) {
        self.keep(w.tick, w.score, || hash(w));
    }

    /// Check a world being played back from this replay against the recorded hash, if one
    /// falls on its current tick.
    pub fn check(&self, w: &World) -> Result<(), Desync> {
        self.compare(w.tick, || hash(w))
    }

    /// Play the whole replay headless, checking every hash; returns the final world.
    pub fn verify(&self) -> Result<World, Desync> {
        let mut w = World::new(self.seed);
        for i in &self.inputs {
            w.tick(&i.unpack());
            self.check(&w)?;
        }
        self.final_score(w.score)?;
        Ok(w)
    }
}

impl<I: Record> Replay<I> {
    /// An empty recording of a run from `seed`.
    pub fn new(seed: u64) -> Replay<I> {
        Replay {
            build: build_id(),
            seed,
            score: 0,
            inputs: Vec::new(),
            hashes: Vec::new(),
        }
    }

    /// Record a tick: call with the packed input just before the world's tick, then `after`.
    pub fn record(&mut self, input: I) {
        self.inputs.push(input);
    }

    /// After tick `tick`: keep the world's hash every second, and the score.
    pub fn keep(&mut self, tick: u64, score: u64, hash: impl FnOnce() -> u64) {
        if tick.is_multiple_of(HASH_EVERY) {
            self.hashes.push(hash());
        }
        self.score = score;
    }

    /// Compare the world's hash at `tick` with the recorded one, if one falls on it.
    pub fn compare(&self, tick: u64, hash: impl FnOnce() -> u64) -> Result<(), Desync> {
        if tick > 0 && tick.is_multiple_of(HASH_EVERY) {
            let k = (tick / HASH_EVERY - 1) as usize;
            if let Some(&h) = self.hashes.get(k)
                && h != hash()
            {
                return Err(Desync::Hash(tick));
            }
        }
        Ok(())
    }

    /// The end of a playback: the score must match.
    pub fn final_score(&self, replayed: u64) -> Result<(), Desync> {
        if replayed == self.score {
            Ok(())
        } else {
            Err(Desync::Score {
                recorded: self.score,
                replayed,
            })
        }
    }

    /// Seconds of simulation.
    pub fn seconds(&self) -> f32 {
        self.inputs.len() as f32 * super::DT
    }

    /// The file form: a header, run-length coded inputs, and the hashes.
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(64 + self.inputs.len() / 4 + self.hashes.len() * 8);
        out.extend_from_slice(I::MAGIC);
        let build = self.build.as_bytes();
        out.extend_from_slice(&(build.len() as u16).to_le_bytes());
        out.extend_from_slice(build);
        out.extend_from_slice(&self.seed.to_le_bytes());
        out.extend_from_slice(&self.score.to_le_bytes());
        let mut runs: Vec<(u16, I)> = Vec::new();
        for &i in &self.inputs {
            match runs.last_mut() {
                Some((n, last)) if *last == i && *n < u16::MAX => *n += 1,
                _ => runs.push((1, i)),
            }
        }
        out.extend_from_slice(&(runs.len() as u32).to_le_bytes());
        for (n, p) in runs {
            out.extend_from_slice(&n.to_le_bytes());
            p.write(&mut out);
        }
        out.extend_from_slice(&(self.hashes.len() as u32).to_le_bytes());
        for h in &self.hashes {
            out.extend_from_slice(&h.to_le_bytes());
        }
        out
    }

    /// Read the file form.
    pub fn decode(bytes: &[u8]) -> Result<Replay<I>, String> {
        let mut r = Reader(bytes);
        if r.take(8)? != I::MAGIC {
            return Err("not a replay of this game".into());
        }
        let n = r.u16()? as usize;
        let build = String::from_utf8(r.take(n)?.to_vec()).map_err(|e| e.to_string())?;
        let seed = r.u64()?;
        let score = r.u64()?;
        let runs = r.u32()?;
        let mut inputs = Vec::new();
        for _ in 0..runs {
            let n = r.u16()?;
            let p = I::read(r.take(I::SIZE)?);
            inputs.extend(std::iter::repeat_n(p, usize::from(n)));
        }
        let n = r.u32()?;
        let hashes = (0..n).map(|_| r.u64()).collect::<Result<_, _>>()?;
        if !r.0.is_empty() {
            return Err("trailing bytes".into());
        }
        Ok(Replay {
            build,
            seed,
            score,
            inputs,
            hashes,
        })
    }
}

struct Reader<'a>(&'a [u8]);

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        if self.0.len() < n {
            return Err("truncated replay".into());
        }
        let (a, b) = self.0.split_at(n);
        self.0 = b;
        Ok(a)
    }
    fn u16(&mut self) -> Result<u16, String> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An input stream with movement, aiming, firing and the odd bomb.
    fn wander(t: u64) -> Input {
        let a = t as f32 * 0.011;
        Input {
            movement: super::super::body::heading(a * 1.7, 0.9),
            aim: super::super::body::heading(a * 3.1, 4.0),
            fire: (t / 90) % 4 != 3,
            bomb: t % 2400 == 2000,
        }
    }

    fn record(seed: u64, ticks: u64) -> (Replay, World) {
        let mut w = World::new(seed);
        let mut rec: Replay = Replay::new(seed);
        for t in 0..ticks {
            let p = Packed::pack(&wander(t));
            rec.record(p);
            w.tick(&p.unpack());
            rec.after(&w);
        }
        (rec, w)
    }

    #[test]
    fn quantizing_is_idempotent() {
        for t in 0..5000 {
            let q = quantize(&wander(t));
            let p = Packed::pack(&q);
            assert_eq!(p, Packed::pack(&wander(t)), "tick {t}");
            let qq = p.unpack();
            assert_eq!(qq.movement.c, q.movement.c);
            assert_eq!(qq.aim.c, q.aim.c);
        }
        // A diagonal from the keyboard stays within the unit disc.
        let d = quantize(&Input {
            movement: Point::direction(1.0, 1.0),
            ..Input::default()
        });
        assert!(d.movement.ideal_norm() <= 1.0 + 1e-6);
    }

    /// The same build replays a run exactly: every per-second hash and the score match, also
    /// after a round trip through the file form.
    #[test]
    fn replays_are_deterministic() {
        for seed in [1, 42, 0xdead_beef] {
            let (rec, live) = record(seed, 120 * 75);
            assert!(rec.hashes.len() == 75);
            let bytes = rec.encode();
            let back = Replay::decode(&bytes).unwrap();
            assert_eq!(back, rec);
            let w = back
                .verify()
                .unwrap_or_else(|d| panic!("seed {seed}: {d:?}"));
            assert_eq!(hash(&w), hash(&live));
            assert_eq!(w.score, live.score);
            assert!(live.score > 0, "seed {seed}: the run did nothing");
        }
    }

    /// A changed input is caught at the next hash.
    #[test]
    fn a_divergence_is_caught_within_a_second() {
        let (mut rec, _) = record(7, 120 * 20);
        rec.inputs[1000].flags ^= BOMB;
        match rec.verify() {
            Err(Desync::Hash(t)) => assert_eq!(t, 1080),
            Err(d) => panic!("{d:?}"),
            Ok(_) => panic!("the change went unnoticed"),
        }
    }

    /// Held inputs cost one run each.
    #[test]
    fn still_stretches_are_run_length_coded() {
        let mut rec: Replay = Replay::new(1);
        for t in 0..1200 {
            rec.record(Packed::pack(&Input {
                fire: t >= 600,
                ..Input::default()
            }));
        }
        assert!(rec.encode().len() < 64, "{} bytes", rec.encode().len());
    }

    #[test]
    fn garbage_is_rejected() {
        assert!(Replay::<Packed>::decode(b"nope").is_err());
        let (rec, _) = record(3, 300);
        let bytes = rec.encode();
        assert!(Replay::<Packed>::decode(&bytes[..bytes.len() - 3]).is_err());
    }
}
