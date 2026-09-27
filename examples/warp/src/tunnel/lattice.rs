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
    /// Where.
    pub pos: Point<(), f32>,
    /// Strength (negative pulls).
    pub strength: f32,
    /// The squared softening radius.
    pub r2: f32,
}

/// The lattice.
pub struct Lattice {
    /// The absolute index of the first ring kept (ring `k` is at `s = k GAP`).
    pub first: i64,
    /// The rest positions around a ring at arc length 0.
    ring: [Point<(), f32>; AROUND],
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

/// Node `j`'s rest position on the ring at arc length `s`: on the wall, turned about the axis.
pub fn rest(j: usize, s: f32) -> Point<(), f32> {
    crate::tunnel::around(RADIUS, angle(j), s)
}

/// One node's step: the spring forces as PGA3D direction arithmetic.
#[inline]
fn node(d: Dir, v: Dir, n: [Dir; 4], f: Dir, dt: f32) -> (Dir, Dir) {
    let lap = n[0] + n[1] + n[2] + n[3] - d * 4.0;
    let a = lap * SPRING - d * ANCHOR - v * DAMPING + f;
    let v = v + a * dt;
    (d + v * dt, v)
}

impl Lattice {
    /// A lattice at rest around the ship at `s`.
    pub fn new(s: f32) -> Lattice {
        Lattice {
            first: ((s - BEHIND) / GAP).floor() as i64,
            ring: core::array::from_fn(|j| rest(j, 0.0)),
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
            let along = Point::direction(0.0, 0.0, self.ring_s(r));
            for j in 0..AROUND {
                let i = r * AROUND + j;
                let at = self.ring[j] + along;
                let mut f = zero();
                for src in sources {
                    // Away from the source, softened: `strength (p - s) / (|p - s|² + r²)`.
                    let away = at - src.pos;
                    let n = away.ideal_norm();
                    f += away * (src.strength / (n * n + src.r2));
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
    pub fn node(&self, r: usize, j: usize) -> Point<(), f32> {
        self.ring[j] + Point::direction(0.0, 0.0, self.ring_s(r)) + self.d[r * AROUND + j]
    }

    /// How far node `j` of ring `r` is displaced (for its glow).
    pub fn strain(&self, r: usize, j: usize) -> f32 {
        self.d[r * AROUND + j].ideal_norm()
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
            pos: Point::xyz(RADIUS - 1.0, 0.0, 20.0),
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
        assert!(crate::tunnel::off_axis(bulge) > RADIUS + 0.05);
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
        assert!((crate::tunnel::off_axis(l.node(r, 0)) - (RADIUS + 0.5)).abs() < 1e-5);
        assert!(l.ring_s(0) <= 7.5 - BEHIND + GAP);
    }
}
