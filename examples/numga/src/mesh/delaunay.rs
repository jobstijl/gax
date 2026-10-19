//! The Delaunay triangulation of points of the plane, by Bowyer and Watson's insertion, its two
//! predicates in gax. Three points turn counter-clockwise when their join, the PGA2D scalar
//! `a & b & c`, is positive. Lifted to null vectors of CGA2D, three points span the circle
//! through them, their outer product; a fourth lies on that circle when its wedge with it
//! vanishes, and inside when the wedge has the sign it has at the circle's centre.
//!
//! Each point goes into the triangulation of the points before it, all inside one large
//! enclosing triangle: the triangles whose circumcircles hold the point form a cavity, found by
//! a breadth-first search from the triangle that contains the point, and the cavity is refilled
//! with a fan from the point to its rim. Points on a common circle, as the rings of a disk, put
//! the in-circle test at the level of rounding, where its answers need not agree; the search
//! keeps the cavity star-shaped as seen from the point, leaving out every triangle whose rim edge
//! the point does not see, so each triangle of the fan turns counter-clockwise.

use gax::{cga2d, pga2d};
use std::collections::{BTreeSet, HashMap, VecDeque};

/// A point of the plane.
pub type Point = pga2d::Point;

/// How far the enclosing triangle reaches, in units of the points' extent: far enough that its
/// circumcircles through two points of the hull do not reach any other point.
const ENCLOSING: f64 = 20.0;

/// The Delaunay triangles of `points`, as indices into them, counter-clockwise. Every point is a
/// vertex, and the triangles cover the points' convex hull.
pub fn triangulate(points: &[Point]) -> Vec<[usize; 3]> {
    let mut mesh = Triangulation::enclosing(points);
    for i in 0..points.len() {
        mesh.insert(i);
    }
    mesh.triangles
        .into_iter()
        .flatten()
        .filter(|t| t.iter().all(|&v| v < points.len()))
        .collect()
}

/// Positive when `a`, `b` and `c` turn counter-clockwise: their join.
pub fn turn(a: Point, b: Point, c: Point) -> f64 {
    (a & b & c).s()
}

/// Positive when `d` lies inside the circle through `a`, `b` and `c`, in counter-clockwise
/// order: the wedge of the four points lifted to null vectors of CGA2D, `eo + x + ½ x² ei`,
/// negative at the centre of the circle.
pub fn in_circle(a: Point, b: Point, c: Point, d: Point) -> f64 {
    let [a, b, c, d] = [a, b, c, d].map(cga2d::Vector::from_point);
    -(a ^ b ^ c ^ d).e12oi()
}

/// A triangulation under construction: the points with the enclosing triangle's three corners
/// after them, the triangles (removed ones `None`), and the triangle on the left of each
/// directed edge.
struct Triangulation {
    points: Vec<Point>,
    triangles: Vec<Option<[usize; 3]>>,
    left_of: HashMap<(usize, usize), usize>,
    /// The latest triangle, where the search for the next point starts.
    latest: usize,
}

impl Triangulation {
    /// The enclosing triangle of `points`, about the centre of their bounding box.
    fn enclosing(points: &[Point]) -> Self {
        let coordinates: Vec<[f64; 2]> = points.iter().map(|p| p.to_euclidean()).collect();
        let bound = |axis: usize, pick: fn(f64, f64) -> f64, from: f64| {
            coordinates.iter().map(|c| c[axis]).fold(from, pick)
        };
        let low = [0, 1].map(|axis| bound(axis, f64::min, f64::INFINITY));
        let high = [0, 1].map(|axis| bound(axis, f64::max, f64::NEG_INFINITY));
        let [cx, cy] = [0, 1].map(|axis| (low[axis] + high[axis]) / 2.0);
        let reach = ENCLOSING * (high[0] - low[0]).max(high[1] - low[1]).max(1.0);
        let n = points.len();
        let mut all = points.to_vec();
        all.extend([
            Point::xy(cx - reach, cy - reach),
            Point::xy(cx + reach, cy - reach),
            Point::xy(cx, cy + reach),
        ]);
        let mut mesh = Triangulation {
            points: all,
            triangles: Vec::new(),
            left_of: HashMap::new(),
            latest: 0,
        };
        mesh.add([n, n + 1, n + 2]);
        mesh
    }

    fn add(&mut self, t: [usize; 3]) {
        let index = self.triangles.len();
        for (a, b) in sides(t) {
            self.left_of.insert((a, b), index);
        }
        self.triangles.push(Some(t));
        self.latest = index;
    }

    fn remove(&mut self, index: usize) {
        let t = self.triangles[index].take().expect("a live triangle");
        for side in sides(t) {
            self.left_of.remove(&side);
        }
    }

    fn corners(&self, index: usize) -> [Point; 3] {
        self.triangles[index]
            .expect("a live triangle")
            .map(|v| self.points[v])
    }

    /// The triangle across the edge `a b` from the triangle on its left.
    fn across(&self, (a, b): (usize, usize)) -> Option<usize> {
        self.left_of.get(&(b, a)).copied()
    }

    /// The triangle containing the point: a walk from the latest triangle, across any edge the
    /// point lies beyond.
    fn locate(&self, p: Point) -> usize {
        let mut at = self.latest;
        loop {
            let t = self.triangles[at].expect("a live triangle");
            let beyond = sides(t)
                .into_iter()
                .find(|&(a, b)| turn(self.points[a], self.points[b], p) < 0.0);
            match beyond {
                Some(side) => at = self.across(side).expect("the point is enclosed"),
                None => return at,
            }
        }
    }

    /// The triangles whose circumcircles hold the point, connected to `start` through each
    /// other, without the `excluded` ones.
    fn cavity(&self, p: Point, start: usize, excluded: &BTreeSet<usize>) -> BTreeSet<usize> {
        let mut cavity = BTreeSet::from([start]);
        let mut queue = VecDeque::from([start]);
        while let Some(at) = queue.pop_front() {
            for side in sides(self.triangles[at].expect("a live triangle")) {
                let Some(next) = self.across(side) else {
                    continue;
                };
                if cavity.contains(&next) || excluded.contains(&next) {
                    continue;
                }
                let [a, b, c] = self.corners(next);
                if in_circle(a, b, c, p) > 0.0 {
                    cavity.insert(next);
                    queue.push_back(next);
                }
            }
        }
        cavity
    }

    /// The cavity's rim: the sides of its triangles that face a triangle outside it, each with
    /// its triangle.
    fn rim(&self, cavity: &BTreeSet<usize>) -> Vec<((usize, usize), usize)> {
        let mut rim = Vec::new();
        for &at in cavity {
            for side in sides(self.triangles[at].expect("a live triangle")) {
                if !self.across(side).is_some_and(|t| cavity.contains(&t)) {
                    rim.push((side, at));
                }
            }
        }
        rim
    }

    /// Insert point `i`: empty its cavity, star-shaped as seen from it, and fan out to the rim.
    fn insert(&mut self, i: usize) {
        let p = self.points[i];
        let start = self.locate(p);
        let mut excluded = BTreeSet::new();
        let (cavity, rim) = loop {
            let cavity = self.cavity(p, start, &excluded);
            let rim = self.rim(&cavity);
            let hidden = rim.iter().find(|&&((a, b), at)| {
                at != start && turn(self.points[a], self.points[b], p) <= 0.0
            });
            match hidden {
                Some(&(_, at)) => excluded.insert(at),
                None => break (cavity, rim),
            };
        };
        for at in cavity {
            self.remove(at);
        }
        for ((a, b), _) in rim {
            self.add([a, b, i]);
        }
    }
}

/// The directed sides of a counter-clockwise triangle.
fn sides([a, b, c]: [usize; 3]) -> [(usize, usize); 3] {
    [(a, b), (b, c), (c, a)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_counter_clockwise_turn_is_positive() {
        let (o, x, y) = (Point::origin(), Point::xy(1.0, 0.0), Point::xy(0.0, 1.0));
        assert!(turn(o, x, y) > 0.0);
        assert!(turn(o, y, x) < 0.0);
    }

    #[test]
    fn the_centre_is_inside_its_circle_and_a_far_point_outside() {
        let (a, b, c) = (
            Point::xy(1.0, 0.0),
            Point::xy(0.0, 1.0),
            Point::xy(-1.0, 0.0),
        );
        assert!(in_circle(a, b, c, Point::origin()) > 0.0);
        assert!(in_circle(a, b, c, Point::xy(0.0, 2.0)) < 0.0);
        assert!(in_circle(a, b, c, Point::xy(0.0, -1.0)).abs() < 1e-15);
    }

    #[test]
    fn a_square_with_its_centre_is_four_triangles() {
        let points = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.5, 0.5)]
            .map(|(x, y)| Point::xy(x, y));
        let triangles = triangulate(&points);
        assert_eq!(triangles.len(), 4);
        for [a, b, c] in triangles {
            assert!(turn(points[a], points[b], points[c]) > 0.0);
            assert!([a, b, c].contains(&4));
        }
    }
}
