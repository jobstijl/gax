//! A `Unit` versor that has drifted (`m ~m = (1 + δ)²`) must still move objects rigidly: the
//! drift may scale results uniformly, which projective and homogeneous uses cancel, but must not
//! distort them (docs/numerics.md, "Drift").

#![cfg(all(feature = "pga3d", feature = "pga2d", feature = "vga3d"))]
// Deliberately drifted units: `check-units` would reject them.
#![cfg(not(feature = "check-units"))]

use gax::Unit;

/// The worst relative change of pairwise distances between the images of `pts`.
fn distortion(pts: &[[f64; 3]], moved: &[[f64; 3]]) -> f64 {
    let d = |a: [f64; 3], b: [f64; 3]| {
        ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
    };
    let mut worst = 0.0f64;
    for i in 0..pts.len() {
        for j in i + 1..pts.len() {
            let (before, after) = (d(pts[i], pts[j]), d(moved[i], moved[j]));
            worst = worst.max((after - before).abs() / before);
        }
    }
    worst
}

const DRIFTS: [f64; 3] = [1e-3, 1e-6, 1e-9];

#[test]
fn drifted_unit_motors_stay_rigid_on_points() {
    use gax::pga3d::{Line, Point};
    let pts = [
        [0.3, -1.2, 0.7],
        [1.5, 0.2, -0.4],
        [-0.8, 0.9, 2.0],
        [0.0, 0.0, 0.0],
    ];
    let u = Line::<(), f64>::new(0.4, -0.3, 0.8, 0.9, -0.5, 0.2).exp();
    for delta in DRIFTS {
        let drifted = Unit::new_unchecked(u.into_inner().gp(1.0 + delta));
        let moved: Vec<[f64; 3]> = pts
            .iter()
            .map(|p| (drifted >> Point::xyz(p[0], p[1], p[2])).to_euclidean())
            .collect();
        let exact: Vec<[f64; 3]> = pts
            .iter()
            .map(|p| (u >> Point::xyz(p[0], p[1], p[2])).to_euclidean())
            .collect();
        let dist = distortion(&exact, &moved);
        // A uniform scale of the homogeneous result leaves the point where it is.
        assert!(dist < 1e-12, "δ = {delta}: distortion {dist}");
        // The matrix path and the prepared path agree with the value path.
        let m = drifted >> Point::slot();
        let prep = drifted.prepare::<Point>();
        for p in &pts {
            let p = Point::xyz(p[0], p[1], p[2]);
            let (a, b, c) = (
                (drifted >> p).to_euclidean(),
                m.of(p).to_euclidean(),
                (prep >> p).to_euclidean(),
            );
            for k in 0..3 {
                assert!(
                    (a[k] - b[k]).abs() < 1e-12 && (a[k] - c[k]).abs() < 1e-12,
                    "paths differ"
                );
            }
        }
    }
}

#[test]
fn drifted_unit_motors_move_lines_and_planes_consistently() {
    use gax::pga3d::{Line, Plane, Point};
    // A line through two points and a plane through three: their images must be the join of
    // the images (up to scale), however the motor has drifted.
    let u = Line::<(), f64>::new(0.4, -0.3, 0.8, 0.9, -0.5, 0.2).exp();
    let (p, q, r) = (
        Point::xyz(0.3, -1.2, 0.7),
        Point::xyz(1.5, 0.2, -0.4),
        Point::xyz(-0.8, 0.9, 2.0),
    );
    for delta in DRIFTS {
        let m = Unit::new_unchecked(u.into_inner().gp(1.0 + delta));
        let line: Line<(), f64> = m >> (p & q);
        let joined: Line<(), f64> = (m >> p) & (m >> q);
        let plane: Plane<(), f64> = m >> ((p & q) & r);
        let plane_joined: Plane<(), f64> = ((m >> p) & (m >> q)) & (m >> r);
        // Parallel up to scale: every 2x2 minor of the coefficient pair vanishes.
        let parallel = |a: &[f64], b: &[f64]| {
            let scale = a.iter().chain(b).fold(0.0f64, |s, x| s.max(x.abs()));
            (0..a.len()).all(|i| {
                (0..a.len()).all(|j| (a[i] * b[j] - a[j] * b[i]).abs() < 1e-12 * scale * scale)
            })
        };
        assert!(parallel(&line.c, &joined.c), "δ = {delta}: line");
        assert!(parallel(&plane.c, &plane_joined.c), "δ = {delta}: plane");
    }
}

#[test]
fn drifted_unit_rotors_scale_vectors_uniformly() {
    use gax::vga3d::{Bivector, Vector};
    // Not projective: drift is a length error, and it must be the same factor for every vector
    // (so angles are kept).
    let u = Bivector::<(), f64>::new(0.4, -0.3, 0.8).exp();
    let vs = [
        Vector::<(), f64>::new(1.0, 0.0, 0.0),
        Vector::new(0.3, 2.0, -1.0),
        Vector::new(-0.5, 0.1, 0.9),
    ];
    for delta in DRIFTS {
        let m = Unit::new_unchecked(u.into_inner().gp(1.0 + delta));
        let ratios: Vec<f64> = vs.iter().map(|v| (m >> *v).norm() / v.norm()).collect();
        let spread = ratios
            .iter()
            .fold(0.0f64, |s, r| s.max((r - ratios[0]).abs()));
        assert!(spread < 1e-12, "δ = {delta}: length ratios {ratios:?}");
    }
}

#[test]
fn drifted_unit_motors_in_the_plane() {
    use gax::pga2d::{Motor, Point};
    let u = Motor::<(), f64>::rotation(Point::xy(0.5, -0.2), 0.9).into_inner()
        * Motor::translation(1.0, 2.0).into_inner();
    let u = u.normalized();
    let pts = [[0.3, -1.2, 0.0], [1.5, 0.2, 0.0], [-0.8, 0.9, 0.0]];
    for delta in DRIFTS {
        let m = Unit::new_unchecked(u.into_inner().gp(1.0 + delta));
        let moved: Vec<[f64; 3]> = pts
            .iter()
            .map(|p| {
                let e = (m >> Point::xy(p[0], p[1])).to_euclidean();
                [e[0], e[1], 0.0]
            })
            .collect();
        let exact: Vec<[f64; 3]> = pts
            .iter()
            .map(|p| {
                let e = (u >> Point::xy(p[0], p[1])).to_euclidean();
                [e[0], e[1], 0.0]
            })
            .collect();
        let dist = distortion(&exact, &moved);
        assert!(dist < 1e-12, "δ = {delta}: distortion {dist}");
    }
}
