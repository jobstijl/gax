//! numga's `test_origami.py`, and the scene against numga's numbers.

use super::kabuto::*;
use super::paper::*;
use super::picture::{Picture, View};
use gax::pga3d::{Line, Point};
use gax::{ApproxEq, vga2d};
use gax_numga_examples::Light;
use std::f64::consts::PI;

/// The paper's area: per face, half the norm of the sum over its edges of the meets of the
/// corners' duals (whose Euclidean part is the cross product of the corners), summed.
fn area(paper: &Paper) -> f64 {
    let next = paper.following();
    paper
        .faces()
        .into_iter()
        .map(|f| {
            let edges = f.map(|i| paper.points[i].dual() ^ paper.points[next[i]].dual());
            0.5 * edges.fold(Line::zero(), |sum, e| sum + e).norm()
        })
        .sum()
}

/// The squared length of each edge, from each corner to the next.
fn edge_lengths(paper: &Paper) -> Vec<f64> {
    let next = paper.following();
    (0..paper.points.len())
        .map(|i| (paper.points[i] & paper.points[next[i]]).norm_squared())
        .collect()
}

/// The largest difference between two lists of numbers.
fn largest_gap(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

/// The paper moved rigidly.
fn moved(paper: &Paper, frame: M) -> Paper {
    Paper {
        points: paper.points.iter().map(|p| frame >> *p).collect(),
        counts: paper.counts.clone(),
        offsets: paper.offsets.iter().map(|o| frame >> *o).collect(),
        sheet: frame >> paper.sheet,
    }
}

/// Every split keeps all the paper, and the motors neither stretch it nor move a crease; at the
/// end, the two horn tips, the line-bisector construction's landmarks, are corners of the
/// paper, and all of it lies on the sheet.
#[test]
fn kabuto_preserves_area_and_facet_lengths_through_every_fold() {
    let (mut paper, planes) = kabuto();
    let initial = area(&paper);
    let (mut area_error, mut length_error, mut hinge_error) = (0.0f64, 0.0f64, 0.0f64);
    for k in 0..planes.len() {
        let (split, turning, _) = paper.split(planes[k], SELECTIONS[k], CREASE_TOLERANCE);
        let hinge = planes[k] ^ paper.sheet;
        let lengths = edge_lengths(&split);
        for angle in [0.0, PI / 2.0, PI * ENDS[k]] {
            let folded = split.folded(hinge, &turning, angle);
            area_error = area_error.max((area(&folded) - initial).abs());
            length_error = length_error.max(largest_gap(&edge_lengths(&folded), &lengths));
            for (before, after) in split.points.iter().zip(&folded.points) {
                if (planes[k] & *before).s().abs() < CREASE_TOLERANCE {
                    hinge_error = hinge_error.max(before.max_abs_diff(after));
                }
            }
        }
        paper = split.folded(hinge, &turning, PI * ENDS[k]);
    }
    assert!(area_error < 1e-10, "{area_error}");
    assert!(length_error < 1e-10, "{length_error}");
    assert!(hinge_error < 1e-11, "{hinge_error}");
    // The diagonal of the unit square.
    let root_two = vga2d::Vector::<(), f64>::new(1.0, 1.0).norm();
    for tip in [
        Point::xyz(-1.0, 1.0 - root_two, 0.0),
        Point::xyz(root_two - 1.0, 1.0, 0.0),
    ] {
        let nearest = paper
            .points
            .iter()
            .map(|p| (*p - tip).ideal_norm())
            .fold(f64::INFINITY, f64::min);
        assert!(nearest * nearest < 1e-20, "{nearest}");
    }
    for p in &paper.points {
        assert!((paper.sheet & *p).s().abs() < 1e-11);
    }
}

/// The whole sequence, run on the sheet moved off the origin and tilted, is the sequence moved.
#[test]
fn full_folding_sequence_follows_a_moved_sheet() {
    let (mut paper, planes) = kabuto();
    // numga's `exp(0.4 xw + 0.6 zw) exp(0.45 xz)`, with `xw = -e01`, `zw = -e03`, `xz = -e31`.
    let frame = Line::new(0.0, 0.0, 0.0, -0.4, 0.0, -0.6).exp()
        * Line::new(0.0, -0.45, 0.0, 0.0, 0.0, 0.0).exp();
    let mut moving = moved(&paper, frame);
    for k in 0..planes.len() {
        let (split, turning, _) = paper.split(planes[k], SELECTIONS[k], CREASE_TOLERANCE);
        let moved_plane = frame >> planes[k];
        let (moved_split, moved_turning, _) =
            moving.split(moved_plane, SELECTIONS[k], CREASE_TOLERANCE);
        paper = split.folded(planes[k] ^ paper.sheet, &turning, PI * ENDS[k]);
        moving = moved_split.folded(moved_plane ^ moving.sheet, &moved_turning, PI * ENDS[k]);
        let expected = moved(&paper, frame);
        for (p, q) in moving.points.iter().zip(&expected.points) {
            assert!(p.max_abs_diff(q) < 1e-11);
        }
        for (p, q) in moving.offsets.iter().zip(&expected.offsets) {
            assert!(p.max_abs_diff(q) < 1e-10);
        }
    }
}

/// Points of adjacent faces that are one point of the original sheet (found by undoing each
/// face's accumulated motor) stay together throughout every fold.
#[test]
fn folds_keep_shared_material_points_joined() {
    let (mut paper, planes) = kabuto();
    let mut placements: Vec<M> = vec![Line::zero().exp()];
    for k in 0..planes.len() {
        let (split, turning, parents) = paper.split(planes[k], SELECTIONS[k], CREASE_TOLERANCE);
        placements = parents.iter().map(|p| placements[*p]).collect();
        let faces = split.face_of_corners();
        let material: Vec<P> = split
            .points
            .iter()
            .zip(&faces)
            .map(|(p, f)| placements[*f].reverse() >> *p)
            .collect();
        let n = material.len();
        let shared: Vec<(usize, usize)> = (0..n)
            .flat_map(|i| (0..n).map(move |j| (i, j)))
            .filter(|&(i, j)| (material[i] & material[j]).norm_squared() < 1e-18)
            .collect();
        let hinge = (planes[k] ^ paper.sheet).normalized();
        let (peak, end) = (PI * PEAKS[k], PI * ENDS[k]);
        for angle in [peak / 2.0, peak, (peak + end) / 2.0, end] {
            let folded = split.folded(hinge, &turning, angle);
            for &(i, j) in &shared {
                assert!((folded.points[i] & folded.points[j]).norm_squared() < 1e-22);
            }
        }
        let turns = turns(hinge, &turning, end);
        placements = turns
            .iter()
            .zip(&placements)
            .map(|(t, p)| *t * *p)
            .collect();
        paper = split.folded(hinge, &turning, end);
    }
}

/// Folding about a point bisector lays one point on the other; folding about a line bisector
/// lays one line on the other, reversed.
#[test]
fn point_and_line_coincidence_crease_constructions() {
    let (paper, _) = kabuto();
    let [first, second, third] = [0, 1, 2].map(|i| paper.points[i]);
    let half_turn = |plane: Pl| (plane ^ paper.sheet).normalized().gp(PI / 2.0).exp();
    let motion = half_turn(point_bisector(first, third));
    assert!((motion >> first).max_abs_diff(&third) < 1e-12);
    let (first_line, second_line) = (first & second, first & third);
    let turn = half_turn(line_bisector(first_line, second_line, first));
    let matched = turn >> first_line.normalized();
    let target = second_line.normalized();
    assert!((matched + target).approx_eq(&Edge::zero(), 1e-12));
}

/// The picture at every pixel centre of a `size` square, rows from the top.
fn raster(picture: &Picture, size: usize) -> Vec<Option<Light>> {
    (0..size * size)
        .map(|i| picture.at((i % size) as f64 + 0.5, (i / size) as f64 + 0.5))
        .collect()
}

/// The square's screen points over the triangle's (or another square's), depths along a slope.
fn screen(corners: &[[f64; 2]], lift: impl Fn(usize, f64) -> f64) -> Vec<P> {
    corners
        .iter()
        .enumerate()
        .map(|(i, [x, y])| Point::xyz(*x, *y, 0.2 * x + 0.1 * y + lift(i, *x)))
        .collect()
}

/// A buried triangular crease leaves no mark, whatever the drawing order, and where two planes
/// cross, the nearer one shows on each side: no order of whole polygons could draw this.
#[test]
fn depth_test_hides_covered_creases_and_resolves_intersecting_planes() {
    let square = [[8.0, 8.0], [56.0, 8.0], [56.0, 56.0], [8.0, 56.0]];
    let triangle = [[16.0, 16.0], [48.0, 16.0], [32.0, 48.0]];
    let colours = [
        Light::from_srgb(0.9, 0.3, 0.1, 1.0),
        Light::from_srgb(0.1, 0.4, 0.9, 1.0),
    ];
    let pixel = |image: &[Option<Light>], x: usize, y: usize| image[y * 64 + x];
    // The square lies a little above the triangle.
    let points = screen(&[&square[..], &triangle[..]].concat(), |i, _| {
        if i < 4 { 1e-3 } else { 0.0 }
    });
    let image = raster(&Picture::new(&points, &[0..4, 4..7], &colours), 64);
    let reversed = raster(
        &Picture::new(&points, &[4..7, 0..4], &[colours[1], colours[0]]),
        64,
    );
    for y in 16..48 {
        for x in 16..48 {
            assert_eq!(pixel(&image, x, y), Some(colours[0]), "{x} {y}");
        }
    }
    assert!(image == reversed);
    // The second square tilts through the first along x.
    let points = screen(&[square, square].concat(), |i, x| {
        if i < 4 { 0.0 } else { (x - 32.0) * 0.1 }
    });
    let image = raster(&Picture::new(&points, &[0..4, 4..8], &colours), 64);
    for y in 20..40 {
        for x in 16..28 {
            assert_eq!(pixel(&image, x, y), Some(colours[0]));
        }
        for x in 36..48 {
            assert_eq!(pixel(&image, x, y), Some(colours[1]));
        }
    }
}

/// The share of pixels that change between two pictures: a pixel changes where paper appears
/// or goes, or where a colour channel moves by more than about five steps in 255 (numga
/// compares 8-bit sRGB pixels; these are linear lights, compared at 0.02).
fn changed(before: &[Option<Light>], after: &[Option<Light>]) -> f64 {
    let differs = |a: &Option<Light>, b: &Option<Light>| match (a, b) {
        (Some(a), Some(b)) => {
            let (a, b) = (a.to_premultiplied(), b.to_premultiplied());
            (0..3).any(|k| (a[k] - b[k]).abs() > 0.02)
        }
        (None, None) => false,
        _ => true,
    };
    let count = before
        .iter()
        .zip(after)
        .filter(|(a, b)| differs(a, b))
        .count();
    count as f64 / before.len() as f64
}

/// An infinitesimal motion at the start or the end of each fold may change which pixels an
/// edge covers, but cannot recolour a flap, from four views above and below.
#[test]
fn paper_shading_has_no_colour_jump_at_fold_boundaries() {
    const PIXELS: usize = 120;
    // Resolves a visible displacement beyond the infinitesimal layer gaps.
    const STEP: f64 = 1e-3;
    for (elevation, azimuth) in [(35.0, 135.0), (-35.0, 135.0), (45.0, 45.0), (-45.0, 225.0)] {
        let (mut paper, planes) = kabuto();
        for k in 0..planes.len() {
            let (split, turning, _) = paper.split(planes[k], SELECTIONS[k], CREASE_TOLERANCE);
            let hinge = planes[k] ^ paper.sheet;
            let step = STEP.copysign(PEAKS[k]);
            let picture =
                |p: &Paper, view: &View| raster(&Picture::of(p, view, PIXELS as f64), PIXELS);
            let view = View::new(elevation, azimuth, &split.points);
            let starting = split.folded(hinge, &turning, step);
            assert!(changed(&picture(&split, &view), &picture(&starting, &view)) < 0.02);
            let displacement = starting
                .offsets
                .iter()
                .zip(&split.offsets)
                .map(|(a, b)| (*a - *b).ideal_norm())
                .fold(0.0, f64::max);
            assert!(displacement < STEP * 100.0);
            let end = PI * ENDS[k];
            let approach = if end != 0.0 { end - step } else { step };
            let approaching = split.folded(hinge, &turning, approach);
            paper = split.folded(hinge, &turning, end);
            assert!(
                changed(&picture(&approaching, &view), &picture(&paper, &view)) < 0.02,
                "{k}"
            );
        }
    }
}

/// Where a turning face and a stationary one overlap in their interiors (no separating axis
/// among their edges' normals; faces that merely share a crease are separated), a positive turn
/// lifts out of the top and a negative one exits below: the turning face lies on that side.
#[test]
fn folds_do_not_pull_buried_layers_through_stationary_paper() {
    let (mut paper, planes) = kabuto();
    for k in 0..planes.len() {
        let (split, turning, _) = paper.split(planes[k], SELECTIONS[k], CREASE_TOLERANCE);
        let faces = split.faces();
        let flat = |f: usize| -> Vec<vga2d::Vector<(), f64>> {
            let corners = split.points[faces[f].clone()].iter();
            corners
                .map(|p| p.to_euclidean())
                .map(|[x, y, _]| vga2d::Vector::new(x, y))
                .collect()
        };
        let heights: Vec<f64> = split
            .offsets
            .iter()
            .map(|o| (split.sheet & *o).s())
            .collect();
        let sign = PEAKS[k].signum();
        for moving in (0..faces.len()).filter(|f| turning[*f]) {
            for fixed in (0..faces.len()).filter(|f| !turning[*f]) {
                if overlap(&flat(moving), &flat(fixed)) {
                    assert!((heights[moving] - heights[fixed]) * sign >= -1e-10);
                }
            }
        }
        paper = split.folded(planes[k] ^ paper.sheet, &turning, PI * ENDS[k]);
    }
}

/// Whether two convex polygons overlap in their interiors: on every normal of their edges (the
/// edge's dual) their shadows overlap by more than a hair.
fn overlap(first: &[vga2d::Vector<(), f64>], second: &[vga2d::Vector<(), f64>]) -> bool {
    let mut normals = [first, second].into_iter().flat_map(|polygon| {
        (0..polygon.len()).map(|i| {
            let edge = polygon[(i + 1) % polygon.len()] - polygon[i];
            edge.dual().normalized()
        })
    });
    let shadow = |polygon: &[vga2d::Vector<(), f64>], axis: vga2d::intrinsic::Vector<(), f64>| {
        let along = polygon.iter().map(|p| (*p | axis).s());
        (
            along.clone().fold(f64::INFINITY, f64::min),
            along.fold(f64::NEG_INFINITY, f64::max),
        )
    };
    normals.all(|axis| {
        let ((low1, high1), (low2, high2)) = (shadow(first, axis), shadow(second, axis));
        high1.min(high2) - low1.max(low2) > 1e-8
    })
}

/// The crease planes and the finished helmet against numga's: the planes' coefficients, the
/// faces' corner counts, the first corners, and the faces' layer heights.
#[test]
fn the_scene_agrees_with_numga() {
    let (_, planes) = kabuto();
    let (r, root_two) = (core::f64::consts::FRAC_1_SQRT_2, core::f64::consts::SQRT_2);
    let numga = [
        [-r, r, 0.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [r, -r, root_two],
        [0.0, 1.0, -1.0],
        [-1.0, 0.0, -1.0],
        [-0.923879532511287, -0.38268343236509, -0.541196100146197],
        [0.38268343236509, 0.923879532511287, -0.541196100146197],
        [r, -r, 1.25 * root_two],
        [r, -r, root_two],
        [r, -r, root_two],
    ];
    for (plane, [x, y, w]) in planes.iter().zip(numga) {
        assert!(
            plane.approx_eq(&gax::pga3d::Plane::new(x, y, 0.0, w), 1e-12),
            "{plane:?}"
        );
    }
    let stages = stages();
    let last = stages.last().expect("eleven folds");
    let helmet = last.at(last.end);
    let counts = [
        3, 3, 3, 3, 3, 3, 3, 3, 4, 3, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    ];
    assert_eq!(helmet.counts, counts);
    let corners = [
        [-1.0, -0.414213562373095],
        [-0.585786437626905, 0.0],
        [-1.0, 1.0],
    ];
    for (p, [x, y]) in helmet.points.iter().zip(corners) {
        assert!(p.max_abs_diff(&Point::xyz(x, y, 0.0)) < 1e-12, "{p:?}");
    }
    let layers = [
        10, 9, 9, 2, 12, 7, 7, 4, 15, 14, 28, 1, 11, 8, 8, 3, 13, 6, 6, 5, -1, 0,
    ];
    for (o, layer) in helmet.offsets.iter().zip(layers) {
        assert!(((helmet.sheet & *o).s() - f64::from(layer)).abs() < 1e-12);
    }
}

#[test]
fn a_frame_draws() {
    gax_numga_examples::app::assert_draws(super::draw, 6.0);
}
