//! Collision tests in PGA2D.
//!
//! * A bullet's path over a tick is the join of its old and new positions, `a & b`: a line.
//!   The distance of a centre `c` from it is `|l & c|` for the normalized line, and the hit
//!   lies within the segment when `a` and `b` are on opposite sides of the perpendicular
//!   `l | c` through `c` (or when an endpoint is itself within reach).
//! * A convex hull is a list of edge lines; a point is inside when it is on the inner side of
//!   every edge: the sign of `edge & p`.
//! * A facing test (for shielded enemies) is the side of a point against the line through the
//!   enemy along its heading's normal.

use super::body::{Pose, distance};
use gax::pga2d::{Line, Point};

/// The signed distance of the finite point `p` from the normalized line `l`.
pub fn signed_distance(l: Line<(), f32>, p: Point<(), f32>) -> f32 {
    (l & p.normalized().into_inner()).s()
}

/// Does the segment from `a` to `b` (both finite points) pass within `r` of `c`?
pub fn segment_hits_circle(
    a: Point<(), f32>,
    b: Point<(), f32>,
    c: Point<(), f32>,
    r: f32,
) -> bool {
    if distance(a, c) <= r || distance(b, c) <= r {
        return true;
    }
    let path = a & b;
    let len = path.norm();
    if len < 1e-6 {
        return false;
    }
    let l = path.gp(1.0 / len);
    if signed_distance(l, c).abs() > r {
        return false;
    }
    // The perpendicular to the path through `c` separates `a` from `b` when the closest
    // approach lies between them.
    let perp = l | c.normalized().into_inner();
    (perp & a).s() * (perp & b).s() <= 0.0
}

/// A convex polygon, in its body's local coordinates, counterclockwise.
#[derive(Clone, Debug)]
#[allow(dead_code)] // for the shielded Warden and hulled enemies (M3); tested below
pub struct Hull {
    /// The corners.
    pub corners: Vec<Point<(), f32>>,
}

#[allow(dead_code)]
impl Hull {
    /// A hull from corners `[x, y]`, counterclockwise.
    pub fn new(corners: &[[f32; 2]]) -> Hull {
        Hull {
            corners: corners.iter().map(|c| Point::xy(c[0], c[1])).collect(),
        }
    }

    /// Is the world point `p` inside the hull placed at `pose`?
    pub fn contains(&self, pose: Pose, p: Point<(), f32>) -> bool {
        let n = self.corners.len();
        (0..n).all(|i| {
            let (a, b) = (pose >> self.corners[i], pose >> self.corners[(i + 1) % n]);
            // Counterclockwise edges have the inside on their positive side.
            ((a & b) & p).s() >= 0.0
        })
    }
}

/// Is `p` in front of a body at `pose` (on the side its local x axis points to)?
pub fn in_front(pose: Pose, p: Point<(), f32>) -> bool {
    let (o, up) = (pose >> Point::xy(0.0, 0.0), pose >> Point::xy(0.0, 1.0));
    // The line from the centre along the local y axis; the heading side is negative for this
    // orientation of the join.
    ((o & up) & p).s() < 0.0
}

/// A uniform spatial hash over the arena, rebuilt each tick.
pub struct SpatialHash {
    cell: f32,
    cols: usize,
    rows: usize,
    origin: [f32; 2],
    cells: Vec<Vec<u32>>,
}

impl SpatialHash {
    /// Cells of size `cell` covering `[-hw, hw] x [-hh, hh]` (with a margin).
    pub fn new(hw: f32, hh: f32, cell: f32) -> SpatialHash {
        let cols = ((2.0 * hw) / cell).ceil() as usize + 2;
        let rows = ((2.0 * hh) / cell).ceil() as usize + 2;
        SpatialHash {
            cell,
            cols,
            rows,
            origin: [-hw - cell, -hh - cell],
            cells: vec![Vec::new(); cols * rows],
        }
    }

    fn index(&self, x: f32, y: f32) -> (usize, usize) {
        let c = ((x - self.origin[0]) / self.cell)
            .floor()
            .clamp(0.0, (self.cols - 1) as f32) as usize;
        let r = ((y - self.origin[1]) / self.cell)
            .floor()
            .clamp(0.0, (self.rows - 1) as f32) as usize;
        (c, r)
    }

    /// Empty every cell (keeping their storage).
    pub fn clear(&mut self) {
        for c in &mut self.cells {
            c.clear();
        }
    }

    /// Insert `id` over the disc at `[x, y]` of radius `r`.
    pub fn insert(&mut self, id: u32, [x, y]: [f32; 2], r: f32) {
        let (c0, r0) = self.index(x - r, y - r);
        let (c1, r1) = self.index(x + r, y + r);
        for row in r0..=r1 {
            for col in c0..=c1 {
                self.cells[row * self.cols + col].push(id);
            }
        }
    }

    /// The ids near the box spanned by two points (with a margin `m`), possibly repeated.
    pub fn query(&self, [x0, y0]: [f32; 2], [x1, y1]: [f32; 2], m: f32, out: &mut Vec<u32>) {
        out.clear();
        let (c0, r0) = self.index(x0.min(x1) - m, y0.min(y1) - m);
        let (c1, r1) = self.index(x0.max(x1) + m, y0.max(y1) + m);
        for row in r0..=r1 {
            for col in c0..=c1 {
                out.extend_from_slice(&self.cells[row * self.cols + col]);
            }
        }
        out.sort_unstable();
        out.dedup();
    }
}

#[cfg(test)]
mod tests {
    use super::super::body::pose_at;
    use super::*;

    #[test]
    fn swept_segments_hit_circles_they_pass() {
        let c = Point::xy(5.0, 0.3);
        // Passes by at distance 0.3: a hit for radius 0.5, not for 0.2.
        assert!(segment_hits_circle(
            Point::xy(0.0, 0.0),
            Point::xy(10.0, 0.0),
            c,
            0.5
        ));
        assert!(!segment_hits_circle(
            Point::xy(0.0, 0.0),
            Point::xy(10.0, 0.0),
            c,
            0.2
        ));
        // Stops short of the closest approach: no hit, though the line passes close.
        assert!(!segment_hits_circle(
            Point::xy(0.0, 0.0),
            Point::xy(3.0, 0.0),
            c,
            0.5
        ));
        // Tunnelling: a fast bullet crossing a small target in one tick still hits it.
        assert!(segment_hits_circle(
            Point::xy(-50.0, 0.0),
            Point::xy(50.0, 0.0),
            Point::xy(0.0, 0.1),
            0.15
        ));
        // Degenerate segment.
        assert!(segment_hits_circle(
            Point::xy(5.0, 0.0),
            Point::xy(5.0, 0.0),
            c,
            0.5
        ));
    }

    #[test]
    fn hulls_contain_their_inside_after_any_motion() {
        let square = Hull::new(&[[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]);
        let pose = pose_at(10.0, -3.0, 0.7);
        assert!(square.contains(pose, pose >> Point::xy(0.9, 0.5)));
        assert!(!square.contains(pose, pose >> Point::xy(1.1, 0.5)));
        assert!(!square.contains(pose, Point::xy(0.0, 0.0)));
    }

    #[test]
    fn facing_is_the_side_of_the_heading() {
        let pose = pose_at(2.0, 2.0, core::f32::consts::FRAC_PI_2); // facing +y
        assert!(in_front(pose, Point::xy(2.0, 5.0)));
        assert!(!in_front(pose, Point::xy(2.0, -1.0)));
    }

    #[test]
    fn the_hash_finds_what_it_holds() {
        let mut h = SpatialHash::new(32.0, 18.0, 2.0);
        h.insert(7, [3.0, 3.0], 0.5);
        h.insert(9, [-20.0, 10.0], 3.0);
        let mut out = Vec::new();
        h.query([2.0, 2.0], [4.0, 4.0], 0.0, &mut out);
        assert_eq!(out, vec![7]);
        h.query([-18.0, 9.0], [-18.0, 9.0], 0.1, &mut out);
        assert_eq!(out, vec![9]);
    }
}

/// Reflect the direction `d` in the line `l` (a direction only sees the line's orientation).
///
/// The reflection is the twisted sandwich `-l d ~l`: gax's `>>` computes `l d ~l`, which for
/// an odd versor (a line) on a bivector (a point) carries a sign. For a finite point the sign
/// is harmless (`-p` is the same point), for a direction it is not (VERIFY.md, friction 7).
pub fn reflect(l: gax::pga2d::Line<(), f32>, d: Point<(), f32>) -> Point<(), f32> {
    let n = l.norm().max(1e-9);
    -(gax::Unit::new_unchecked(l.gp(1.0 / n)) >> d)
}

#[cfg(test)]
mod reflect_tests {
    use super::*;

    #[test]
    fn reflection_in_a_line_mirrors_directions() {
        // The vertical line x = 3: a direction (1, 0.5) becomes (-1, 0.5).
        let l = Point::xy(3.0, 0.0) & Point::xy(3.0, 1.0);
        let r = reflect(l, Point::direction(1.0, 0.5));
        assert!(
            (r.e20() + 1.0).abs() < 1e-5 && (r.e01() - 0.5).abs() < 1e-5,
            "{r:?}"
        );
    }
}
