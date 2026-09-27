//! Collision tests in PGA2D.
//!
//! * A bullet's path over a tick is the join of its old and new positions, `a & b`: a line.
//!   The distance of a centre `c` from it is `|l & c|` for the normalized line, and the hit
//!   lies within the segment when `a` and `b` are on opposite sides of the perpendicular
//!   `l | c` through `c` (or when an endpoint is itself within reach).
//! * The arena's walls are lines with the inside on their positive side (`wall & p`); a body
//!   outside bounces by reflection in the wall (`Line::reflect`).
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

/// The walls of the box `[-hw, hw] x [-hh, hh]`, counterclockwise and normalized: each has
/// the inside on its positive side, so `(wall & p).s()` is the signed distance inwards.
pub fn walls(hw: f32, hh: f32) -> [Line<(), f32>; 4] {
    let c = [
        Point::xy(-hw, -hh),
        Point::xy(hw, -hh),
        Point::xy(hw, hh),
        Point::xy(-hw, hh),
    ];
    core::array::from_fn(|i| (c[i] & c[(i + 1) % 4]).normalized().into_inner())
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
    origin: Point<(), f32>,
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
            origin: Point::xy(-hw - cell, -hh - cell),
            cells: vec![Vec::new(); cols * rows],
        }
    }

    /// The cell of a point displaced by `(dx, dy)`: its offset from the grid's origin, in
    /// cells.
    fn index(&self, p: Point<(), f32>, dx: f32, dy: f32) -> (usize, usize) {
        let d = super::body::unit_weight(p) - self.origin + Point::direction(dx, dy);
        let c = (d.e20() / self.cell)
            .floor()
            .clamp(0.0, (self.cols - 1) as f32) as usize;
        let r = (d.e01() / self.cell)
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

    /// Insert `id` over the disc at `p` of radius `r`.
    pub fn insert(&mut self, id: u32, p: Point<(), f32>, r: f32) {
        let (c0, r0) = self.index(p, -r, -r);
        let (c1, r1) = self.index(p, r, r);
        for row in r0..=r1 {
            for col in c0..=c1 {
                self.cells[row * self.cols + col].push(id);
            }
        }
    }

    /// The ids near the segment `a → b` (the cells of its box, with a margin `m`), possibly
    /// repeated.
    pub fn query(&self, a: Point<(), f32>, b: Point<(), f32>, m: f32, out: &mut Vec<u32>) {
        out.clear();
        let (ca, ra) = self.index(a, 0.0, 0.0);
        let (cb, rb) = self.index(b, 0.0, 0.0);
        let k = (m / self.cell).ceil() as usize;
        let (c0, c1) = (
            ca.min(cb).saturating_sub(k),
            (ca.max(cb) + k).min(self.cols - 1),
        );
        let (r0, r1) = (
            ra.min(rb).saturating_sub(k),
            (ra.max(rb) + k).min(self.rows - 1),
        );
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
    fn walls_face_inwards() {
        for w in walls(10.0, 5.0) {
            assert!((w & Point::xy(0.0, 0.0)).s() > 0.0);
            assert!(
                (w & Point::xy(30.0, 30.0)).s() < 0.0 || (w & Point::xy(-30.0, -30.0)).s() < 0.0
            );
        }
        let right = walls(10.0, 5.0)[1];
        assert!(((right & Point::xy(9.0, 0.0)).s() - 1.0).abs() < 1e-5);
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
        h.insert(7, Point::xy(3.0, 3.0), 0.5);
        h.insert(9, Point::xy(-20.0, 10.0), 3.0);
        let mut out = Vec::new();
        h.query(Point::xy(2.5, 2.5), Point::xy(3.5, 3.5), 0.0, &mut out);
        assert_eq!(out, vec![7]);
        h.query(Point::xy(-18.0, 9.0), Point::xy(-18.0, 9.0), 0.1, &mut out);
        assert_eq!(out, vec![9]);
    }
}
