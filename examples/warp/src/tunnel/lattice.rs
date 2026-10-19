//! The tunnel's wall: the warped-space lattice on a cylinder, in PGA3D.
//!
//! Nodes sit on rings along the track. Each has a displacement from its rest position and a
//! velocity, both PGA3D directions, held as gax fields over the nodes. The forces are linear in
//! the displacements: springs to the four neighbours (around the ring, which wraps, and along
//! the track, where the end rings are free), an anchor spring to rest, damping, and bulges
//! (blasts bulge the wall, singularities pinch it):
//!
//! `a = SPRING L d - ANCHOR d - DAMPING v + f`,
//!
//! where `L d` sums `neighbour - d` over each node's springs: the cylinder grid's graph
//! Laplacian (negated), a sparse field map built once from the lattice's edges. The Plane's
//! lattice has the same law (springs of rest length zero, linear in the positions), with its
//! own constants, as a traced per-node kernel on the GPU (`kernels::grid_node`). The step is symplectic Euler, explicit: an implicit step would damp the ripples
//! that are the look. The rings scroll with the ship: old ones are dropped behind, new ones
//! start at rest ahead; scrolling moves the fields' data, never the map.

use crate::tunnel::{around, dir, off_axis};
use gax::field::{Field, SparseMap};
use gax::pga3d::{Point, Scalar};

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

/// A bulge in the wall (or a pinch): a force source in straightened coordinates, with a
/// softening radius, that pushes nodes away for a positive `push` (a blast bulges the wall)
/// and pulls them in for a negative one (a singularity pinches it). The sign is the opposite of `kernels::source_force`'s strength,
/// which pulls when positive.
#[derive(Clone, Copy, Debug)]
pub struct Bulge {
    /// Where.
    pub pos: Point<(), f32>,
    /// How hard it pushes nodes away (negative: pulls them in).
    pub push: f32,
    /// The squared softening radius.
    pub r2: f32,
    /// The squared reach (`0`: unlimited): beyond it the force falls off faster, by
    /// `reach² / (d² + reach²)`.
    pub reach2: f32,
}

impl Bulge {
    /// Its force on a node at `at`: away from it, softened, `push (p - s) / (|p - s|² + r²)`,
    /// times the reach's fall-off.
    fn force(&self, at: Point<(), f32>) -> Dir {
        let away = at - self.pos;
        let n2 = away.ideal_norm_squared();
        let reach = if self.reach2 > 0.0 {
            self.reach2 / (n2 + self.reach2)
        } else {
            1.0
        };
        away * (self.push * reach / (n2 + self.r2))
    }
}

/// The lattice.
pub struct Lattice {
    /// The absolute index of the first ring kept (ring `k` is at `s = k GAP`).
    first: i64,
    /// The rest positions around a ring at arc length 0.
    ring: [Point<(), f32>; AROUND],
    /// The springs between neighbours as a field map over the nodes: row `i` sums
    /// `neighbour - d_i` over node `i`'s springs, `-~E E` with `E` the springs' ends.
    laplacian: SparseMap<Scalar<(), f32>>,
    /// Displacements and velocities, ring-major (node `r AROUND + j`), for rings
    /// `first..first + RINGS`.
    displacement: Field<Dir>,
    velocity: Field<Dir>,
    /// The springs' pull on each node this step, `L d` (kept, so that a step allocates
    /// nothing).
    pull: Field<Dir>,
    /// Each ring's mean distance from the axis (the wall where a singularity pinches it).
    radii: Vec<f32>,
}

/// Node `j`'s rest position on the ring at arc length `s`: on the wall, turned about the axis.
fn rest(j: usize, s: f32) -> Point<(), f32> {
    around(RADIUS, j as f32 * core::f32::consts::TAU / AROUND as f32, s)
}

/// The springs of the cylinder grid as pairs of nodes: around each ring (which wraps), and
/// along the track between neighbouring rings.
fn springs() -> impl Iterator<Item = [usize; 2]> {
    (0..RINGS).flat_map(|r| {
        (0..AROUND).flat_map(move |j| {
            let i = r * AROUND + j;
            let around = [i, r * AROUND + (j + 1) % AROUND];
            let along = (r + 1 < RINGS).then_some([i, i + AROUND]);
            core::iter::once(around).chain(along)
        })
    })
}

/// The springs' pull as a field map: `-~E E`, with `E` the map from the nodes to the springs
/// that takes each spring's difference of its ends (its row couples to its two ends).
fn laplacian() -> SparseMap<Scalar<(), f32>> {
    let ends: Vec<[usize; 2]> = springs().collect();
    let sign = [-1.0, 1.0];
    let ends = SparseMap::from_columns(&ends, RINGS * AROUND, |_, end| Scalar::new(sign[end]));
    (&ends.reverse() * &ends).map_cells(|c| -c)
}

impl Lattice {
    /// A lattice at rest around the ship at `s`.
    pub fn new(s: f32) -> Lattice {
        let nodes = RINGS * AROUND;
        Lattice {
            first: ((s - BEHIND) / GAP).floor() as i64,
            ring: core::array::from_fn(|j| rest(j, 0.0)),
            laplacian: laplacian(),
            displacement: Field::zeros(nodes),
            velocity: Field::zeros(nodes),
            pull: Field::zeros(nodes),
            radii: vec![RADIUS; RINGS],
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
        for field in [&mut self.displacement, &mut self.velocity] {
            let sites = field.sites_mut();
            sites.rotate_left(n);
            let len = sites.len();
            sites[len - n..].fill(Dir::zero());
        }
        self.radii.rotate_left(shift);
        self.radii[RINGS - shift..].fill(RADIUS);
        self.first = want;
    }

    /// One step of `dt` with the given bulges: the springs' pull from the field map, then
    /// each node's velocity and displacement (symplectic Euler).
    pub fn step(&mut self, dt: f32, bulges: &[Bulge]) {
        self.laplacian.mul_into(&self.displacement, &mut self.pull);
        let pull = self.pull.sites();
        let d = self.displacement.sites_mut();
        let v = self.velocity.sites_mut();
        for r in 0..RINGS {
            let along = dir(0.0, 0.0, (self.first + r as i64) as f32 * GAP);
            for (j, &home) in self.ring.iter().enumerate() {
                let i = r * AROUND + j;
                let at = home + along;
                let f = bulges.iter().fold(Dir::zero(), |f, b| f + b.force(at));
                let a = pull[i] * SPRING - d[i] * ANCHOR - v[i] * DAMPING + f;
                v[i] += a * dt;
                d[i] += v[i] * dt;
            }
        }
        for r in 0..RINGS {
            let sum: f32 = (0..AROUND).map(|j| off_axis(self.node(r, j))).sum();
            self.radii[r] = sum / AROUND as f32;
        }
    }

    /// The wall's radius at arc length `s`: the mean distance of the rings' nodes from the
    /// axis, between the two rings around `s` (at rest, `RADIUS`; less where it is pinched).
    pub fn radius(&self, s: f32) -> f32 {
        let u = s / GAP - self.first as f32;
        if u <= 0.0 || u >= (RINGS - 1) as f32 {
            return RADIUS;
        }
        let (i, t) = (u.floor() as usize, u.fract());
        self.radii[i] + (self.radii[i + 1] - self.radii[i]) * t
    }

    /// Every ring's radius (the mean distance of its nodes from the axis), from the first
    /// kept.
    pub fn radii(&self) -> &[f32] {
        &self.radii
    }

    /// The displaced node `j` of ring `r`, in straightened coordinates.
    pub fn node(&self, r: usize, j: usize) -> Point<(), f32> {
        self.ring[j] + dir(0.0, 0.0, self.ring_s(r)) + self.displacement.site(r * AROUND + j)
    }

    /// How far node `j` of ring `r` is displaced (for its glow).
    pub fn strain(&self, r: usize, j: usize) -> f32 {
        self.displacement.site(r * AROUND + j).ideal_norm()
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
        let blast = [Bulge {
            pos: Point::xyz(RADIUS - 1.0, 0.0, 20.0),
            push: 400.0,
            r2: 2.0,
            reach2: 0.0,
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

    /// A source that pulls pinches the wall in around it, and the wall's radius says so.
    #[test]
    fn a_pull_pinches_the_wall() {
        let mut l = Lattice::new(0.0);
        let well = [Bulge {
            pos: Point::xyz(1.0, 0.0, 30.0),
            push: -crate::tunnel::PINCH,
            r2: 8.0,
            reach2: crate::tunnel::PINCH_REACH2,
        }];
        for _ in 0..240 {
            l.step(1.0 / 120.0, &well);
        }
        let pinched = l.radius(30.0);
        assert!(pinched < RADIUS - 1.5, "{pinched}");
        assert!((l.radius(80.0) - RADIUS).abs() < 0.05);
    }

    #[test]
    fn following_keeps_rings_at_their_arc_length() {
        let mut l = Lattice::new(0.0);
        l.displacement.sites_mut()[10 * AROUND] = Point::direction(0.5, 0.0, 0.0);
        let s = l.ring_s(10);
        l.follow(7.5);
        let r = (0..RINGS)
            .find(|&r| (l.ring_s(r) - s).abs() < 1e-3)
            .unwrap();
        assert!((crate::tunnel::off_axis(l.node(r, 0)) - (RADIUS + 0.5)).abs() < 1e-5);
        assert!(l.ring_s(0) <= 7.5 - BEHIND + GAP);
    }

    /// The springs' map is the cylinder grid's graph Laplacian: symmetric, with the constants
    /// in its null space (moving every node alike stretches no spring), and each node coupled
    /// to its four neighbours, around the ring across the seam, and along the track except
    /// past the end rings.
    #[test]
    fn the_springs_are_the_grids_laplacian() {
        let l = Lattice::new(0.0);
        let cells: std::collections::BTreeMap<(usize, usize), f32> = l
            .laplacian
            .couplings()
            .map(|(r, c, cell)| ((r, c), cell.s()))
            .collect();
        for (&(r, c), &x) in &cells {
            assert_eq!(cells.get(&(c, r)), Some(&x), "not symmetric at {r}, {c}");
        }
        let one = Field::from_fn(RINGS * AROUND, |_| dir(1.0, -2.0, 0.5));
        let mut pull = Field::zeros(RINGS * AROUND);
        l.laplacian.mul_into(&one, &mut pull);
        assert!(pull.iter().all(|p| p.ideal_norm() == 0.0));
        let row = |i: usize| -> Vec<(usize, f32)> {
            cells
                .range((i, 0)..(i + 1, 0))
                .map(|(&(_, c), &x)| (c, x))
                .collect()
        };
        // Inside: four neighbours, the node itself at -4.
        let i = 5 * AROUND + 3;
        let expect = [i - AROUND, i - 1, i, i + 1, i + AROUND];
        assert_eq!(row(i), expect.map(|c| (c, if c == i { -4.0 } else { 1.0 })));
        // At the seam, around the ring to its other end.
        let i = 5 * AROUND;
        let expect = [i - AROUND, i, i + 1, i + AROUND - 1, i + AROUND];
        assert_eq!(row(i), expect.map(|c| (c, if c == i { -4.0 } else { 1.0 })));
        // The end rings are free: three neighbours.
        let i = (RINGS - 1) * AROUND + 7;
        let expect = [i - AROUND, i - 1, i, i + 1];
        assert_eq!(row(i), expect.map(|c| (c, if c == i { -3.0 } else { 1.0 })));
        assert_eq!(row(7).len(), 4);
    }

    /// Scrolling further than the lattice is long starts it all over at rest.
    #[test]
    fn following_far_ahead_starts_at_rest() {
        let mut l = Lattice::new(0.0);
        let blast = [Bulge {
            pos: Point::xyz(RADIUS - 1.0, 0.0, 20.0),
            push: 400.0,
            r2: 2.0,
            reach2: 0.0,
        }];
        for _ in 0..30 {
            l.step(1.0 / 120.0, &blast);
        }
        l.follow(1000.0);
        assert!(l.ring_s(0) <= 1000.0 - BEHIND && l.ring_s(1) > 1000.0 - BEHIND);
        assert!((0..RINGS * AROUND).all(|i| l.strain(i / AROUND, i % AROUND) == 0.0));
        assert!(l.radii().iter().all(|&r| r == RADIUS));
        l.step(1.0 / 120.0, &[]);
        assert!((0..RINGS * AROUND).all(|i| l.strain(i / AROUND, i % AROUND) == 0.0));
    }
}
