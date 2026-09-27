//! The tunnel's wall: the warped-space lattice on a cylinder, in PGA3D.
//!
//! Nodes sit on rings along the track. Each has a displacement from its rest position and a
//! velocity, both PGA3D directions, and the same physics as the Plane's lattice: an anchor
//! spring to rest, springs to its four neighbours (around the ring, which wraps, and along the
//! track), damping, and forces from sources (blasts bulge the wall, singularities pinch it).
//! The rings scroll with the ship: old ones are dropped behind, new ones start at rest ahead.

use gax::pga3d::Point;

/// A 3D displacement or velocity (a direction).
pub type Dir = Point<(), f32>;

/// Nodes around a ring.
pub const AROUND: usize = 32;
/// Rings kept.
pub const RINGS: usize = 72;
/// Arc length between rings.
pub const GAP: f32 = 1.5;
/// The tunnel's radius.
pub const RADIUS: f32 = 7.0;
/// How far behind the ship the first ring is.
const BEHIND: f32 = 9.0;

const ANCHOR: f32 = 7.0;
const SPRING: f32 = 60.0;
const DAMPING: f32 = 4.0;

/// A force source in straightened coordinates: pushes nodes away (positive strength) or pulls
/// them in, with a softening radius.
#[derive(Clone, Copy, Debug)]
pub struct Source {
    /// Where, `(x, y, s)`.
    pub pos: [f32; 3],
    /// Strength (negative pulls).
    pub strength: f32,
    /// The squared softening radius.
    pub r2: f32,
}

/// The lattice.
pub struct Lattice {
    /// The absolute index of the first ring kept (ring `k` is at `s = k GAP`).
    pub first: i64,
    /// Displacements, ring-major (`ring * AROUND + j`), for rings `first..first + RINGS`.
    pub d: Vec<Dir>,
    v: Vec<Dir>,
    scratch: Vec<Dir>,
}

fn zero() -> Dir {
    Point::direction(0.0, 0.0, 0.0)
}

/// The angle of node `j` around the ring.
pub fn angle(j: usize) -> f32 {
    j as f32 * core::f32::consts::TAU / AROUND as f32
}

/// A node's rest position across the tunnel.
pub fn rest(j: usize) -> [f32; 2] {
    let a = angle(j);
    [RADIUS * a.cos(), RADIUS * a.sin()]
}

/// One node's step: the spring forces as PGA3D direction arithmetic.
#[inline]
fn node(d: Dir, v: Dir, n: [Dir; 4], f: Dir, dt: f32) -> (Dir, Dir) {
    let lap = n[0] + n[1] + n[2] + n[3] - d.gp(4.0);
    let a = lap.gp(SPRING) - d.gp(ANCHOR) - v.gp(DAMPING) + f;
    let v = v + a.gp(dt);
    (d + v.gp(dt), v)
}

impl Lattice {
    /// A lattice at rest around the ship at `s`.
    pub fn new(s: f32) -> Lattice {
        Lattice {
            first: ((s - BEHIND) / GAP).floor() as i64,
            d: vec![zero(); RINGS * AROUND],
            v: vec![zero(); RINGS * AROUND],
            scratch: vec![zero(); RINGS * AROUND],
        }
    }

    /// Arc length of ring `r` (an index into the kept rings).
    pub fn ring_s(&self, r: usize) -> f32 {
        (self.first + r as i64) as f32 * GAP
    }

    /// Scroll with the ship at `s`: rings behind are dropped, new rings start at rest.
    pub fn follow(&mut self, s: f32) {
        let want = ((s - BEHIND) / GAP).floor() as i64;
        let shift = (want - self.first).clamp(0, RINGS as i64) as usize;
        if shift == 0 {
            return;
        }
        let n = shift * AROUND;
        self.d.rotate_left(n);
        self.v.rotate_left(n);
        let len = self.d.len();
        self.d[len - n..].fill(zero());
        self.v[len - n..].fill(zero());
        self.first = want;
    }

    /// One step of `dt` with the given sources.
    pub fn step(&mut self, dt: f32, sources: &[Source]) {
        for r in 0..RINGS {
            let s = self.ring_s(r);
            for j in 0..AROUND {
                let i = r * AROUND + j;
                let [x, y] = rest(j);
                let mut f = zero();
                for src in sources {
                    let (dx, dy, ds) = (x - src.pos[0], y - src.pos[1], s - src.pos[2]);
                    let k = src.strength / (dx * dx + dy * dy + ds * ds + src.r2);
                    f += Point::direction(dx * k, dy * k, ds * k);
                }
                let around = |jj: usize| self.d[r * AROUND + jj % AROUND];
                // Along the track, the end rings see themselves (a free edge).
                let prev = if r > 0 { self.d[i - AROUND] } else { self.d[i] };
                let next = if r + 1 < RINGS {
                    self.d[i + AROUND]
                } else {
                    self.d[i]
                };
                let n = [around(j + 1), around(j + AROUND - 1), prev, next];
                let (d, v) = node(self.d[i], self.v[i], n, f, dt);
                self.scratch[i] = d;
                self.v[i] = v;
            }
        }
        std::mem::swap(&mut self.d, &mut self.scratch);
    }

    /// The displaced node `j` of ring `r`, in straightened coordinates.
    pub fn node(&self, r: usize, j: usize) -> [f32; 3] {
        let [x, y] = rest(j);
        let d = self.d[r * AROUND + j];
        [x + d.e032(), y + d.e013(), self.ring_s(r) + d.e021()]
    }

    /// How far node `j` of ring `r` is displaced (for its glow).
    pub fn strain(&self, r: usize, j: usize) -> f32 {
        let d = self.d[r * AROUND + j];
        (d.e032() * d.e032() + d.e013() * d.e013() + d.e021() * d.e021()).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A blast bulges the wall near it, the bulge travels along the lattice as a ripple, and
    /// everything settles back to rest.
    #[test]
    fn a_blast_ripples_and_settles() {
        let mut l = Lattice::new(0.0);
        let blast = [Source {
            pos: [RADIUS - 1.0, 0.0, 20.0],
            strength: 400.0,
            r2: 2.0,
        }];
        for _ in 0..12 {
            l.step(1.0 / 120.0, &blast);
        }
        let near = (0..RINGS)
            .min_by_key(|&r| ((l.ring_s(r) - 20.0).abs() * 100.0) as i32)
            .unwrap();
        let bulge = l.node(near, 0);
        assert!(bulge[0] > RADIUS + 0.05, "{bulge:?}");
        let far_before = l.strain(near + 12, 0);
        for _ in 0..60 {
            l.step(1.0 / 120.0, &[]);
        }
        assert!(
            l.strain(near + 12, 0) > far_before,
            "the ripple did not travel"
        );
        for _ in 0..1200 {
            l.step(1.0 / 120.0, &[]);
        }
        let worst = (0..RINGS * AROUND)
            .map(|i| l.strain(i / AROUND, i % AROUND))
            .fold(0.0f32, f32::max);
        assert!(worst < 1e-3, "{worst}");
    }

    #[test]
    fn following_keeps_rings_at_their_arc_length() {
        let mut l = Lattice::new(0.0);
        l.d[10 * AROUND] = Point::direction(0.5, 0.0, 0.0);
        let s = l.ring_s(10);
        l.follow(7.5);
        let r = (0..RINGS)
            .find(|&r| (l.ring_s(r) - s).abs() < 1e-3)
            .unwrap();
        assert!((l.node(r, 0)[0] - (RADIUS + 0.5)).abs() < 1e-5);
        assert!(l.ring_s(0) <= 7.5 - BEHIND + GAP);
    }
}
