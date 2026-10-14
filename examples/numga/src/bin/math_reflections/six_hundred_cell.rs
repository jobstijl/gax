//! numga's `six_hundred_cell` notebook: the group the icosahedron's rotors make, VGA3D.
//!
//! The turns of the icosahedron, the products of its mirror vectors in pairs, generate a group
//! of unit spinors. Starting from one, each round multiplies everything so far by one or by each
//! of the three turns; after eight rounds the 65536 products repeat each other, closed up into
//! 120 different spinors, the binary icosahedral group. As points of the three-sphere they are
//! the vertices of the 600-cell: each has twelve nearest neighbours, at one over the golden
//! ratio. Multiplying every spinor by a unit spinor on the left and another on the right turns
//! the three-sphere in two planes at once and the polytope into itself, moved.

use super::icosahedron::{R, mirrors, turns};
use gax::vga3d::{Bivector, Rotor, Vector};
use gax_numga_examples::{Axes, Canvas, Light, Point2};
use std::collections::HashSet;
use std::sync::OnceLock;

/// The longest products of turns taken: enough to reach every element of the group.
const LENGTH: usize = 8;
/// Spinors closer than this, coefficient by coefficient, are the same.
const SAME: f64 = 1e-6;
/// Where the perspective into three dimensions looks from, along the scalar axis.
const EYE: f64 = 2.4;
/// How far the view tips the projected polytope about `x`.
const TILT: f64 = 0.45;

/// Every product of `LENGTH` steps, each step one or a turn.
pub fn products() -> Vec<R> {
    let steps: Vec<R> = [Rotor::new(1.0, 0.0, 0.0, 0.0)]
        .into_iter()
        .chain(turns(mirrors()).map(|(turn, _)| turn))
        .collect();
    (1..LENGTH).fold(steps.clone(), |products, _| {
        products
            .iter()
            .flat_map(|p| steps.iter().map(move |s| *p * *s))
            .collect()
    })
}

/// The distinct spinors among many, up to round-off: the first of each set equal to six
/// decimals, in order.
pub fn distinct(spinors: &[R]) -> Vec<R> {
    let mut seen = HashSet::new();
    let key = |r: &R| r.c.map(|x| (x / SAME).round() as i64);
    spinors
        .iter()
        .filter(|r| seen.insert(key(r)))
        .copied()
        .collect()
}

/// The group's 120 spinors, found once.
pub fn vertices() -> &'static [R] {
    static VERTICES: OnceLock<Vec<R>> = OnceLock::new();
    VERTICES.get_or_init(|| distinct(&products()))
}

/// The pairs of vertices at the nearest distance, the 600-cell's edges, found once.
pub fn edges() -> &'static [(usize, usize)] {
    static EDGES: OnceLock<Vec<(usize, usize)>> = OnceLock::new();
    EDGES.get_or_init(|| {
        let v = vertices();
        let gap = |i: usize, j: usize| (v[i] - v[j]).norm();
        let pairs = (0..v.len()).flat_map(|i| (i + 1..v.len()).map(move |j| (i, j)));
        let nearest = pairs
            .clone()
            .map(|(i, j)| gap(i, j))
            .fold(f64::INFINITY, f64::min);
        pairs
            .filter(|&(i, j)| (gap(i, j) - nearest).abs() < SAME)
            .collect()
    })
}

/// A spinor as a point of the drawing: the three-sphere in perspective from beyond the scalar
/// axis into space, tipped towards the viewer; its place across and up, and its depth.
fn project(spinor: R) -> (Point2, f64) {
    let space = Vector::new(spinor.e23(), spinor.e31(), spinor.e12()).gp(1.0 / (EYE - spinor.s()));
    let tip = Bivector::new(1.0, 0.0, 0.0).gp(-TILT / 2.0).exp();
    let seen = tip >> space;
    (Point2::xy(seen.e1() as f32, seen.e2() as f32), seen.e3())
}

/// The polytope at time `t` of the act: every spinor turned on the left in the `xy` plane and
/// on the right in the `yz` plane, the right turn at a fifth of the rate; edges nearer the
/// viewer brighter, wider and cream, farther ones fainter and blue.
pub fn draw(c: &mut Canvas, t: f32) {
    let angle = core::f64::consts::TAU * f64::from(t / super::ACT);
    let left = Bivector::new(0.0, 0.0, angle / 2.0).exp().into_inner();
    let right = Bivector::new(angle / 10.0, 0.0, 0.0).exp().into_inner();
    let points: Vec<(Point2, f64)> = vertices()
        .iter()
        .map(|v| project(left * *v * right))
        .collect();
    let (near, far) = (
        Light::from_srgb(0.93, 0.83, 0.62, 1.4),
        Light::from_srgb(0.35, 0.55, 0.72, 1.6),
    );
    let rect = c.rect().part(0.2, 0.12, 0.8, 1.0);
    let ax = Axes::equal(rect, Point2::xy(0.0, 0.0), 0.62);
    let depths = edges()
        .iter()
        .map(|&(i, j)| 0.5 * (points[i].1 + points[j].1));
    let (low, high) = depths.fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), d| {
        (lo.min(d), hi.max(d))
    });
    for &(i, j) in edges() {
        let shade = ((0.5 * (points[i].1 + points[j].1) - low) / (high - low)) as f32;
        let colour = far.mix_light(near, shade).faded(0.35 + 0.65 * shade);
        ax.line(c, points[i].0, points[j].0, 0.6 + 1.2 * shade, colour);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gax_numga_examples::measure::roots;

    /// numga's checks: the products are unit spinors, 120 different ones; the product of any
    /// two of them is one of them; each has twelve nearest neighbours, at the edge length of the
    /// 600-cell, one over the golden ratio (the positive root of `x² + x - 1`).
    #[test]
    fn the_turns_close_into_the_six_hundred_cell() {
        let v = vertices();
        assert_eq!(v.len(), 120);
        for r in v {
            assert!((r.norm() - 1.0).abs() < 1e-12);
        }
        let pairs: Vec<R> = v
            .iter()
            .flat_map(|a| v.iter().map(move |b| *a * *b))
            .collect();
        let closed = distinct(&pairs);
        assert_eq!(closed.len(), 120);
        for p in &closed {
            let nearest = v
                .iter()
                .map(|r| (*r - *p).norm())
                .fold(f64::INFINITY, f64::min);
            assert!(nearest < 1e-9);
        }
        let (middle, half) = roots(1.0, 0.5, -1.0).expect("real roots");
        let edge = middle + half;
        for a in v {
            let neighbours = v.iter().filter(|b| ((*a - **b).norm() - edge).abs() < 1e-9);
            assert_eq!(neighbours.count(), 12);
        }
        assert_eq!(edges().len(), 120 * 12 / 2);
    }
}
