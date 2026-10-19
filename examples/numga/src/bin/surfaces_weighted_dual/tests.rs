//! numga's tests of `surfaces/weighted_dual`, and the checks of its scenes.

use super::dual::*;
use super::program::{unit_rows, widest};
use super::scenes::*;
use super::surfaces_hull::{POINTS, beside, directions, hull};
use gax::field::Field;
use gax_numga_examples::mesh::{Mesh, Scalar3, Vector3};
use gax_numga_examples::rng::{Draw, rng};

/// numga's frames of one breath, and of the way to the rescue.
const FRAMES: usize = 48;

/// Each point's direction from the centre.
fn images(points: &Field<Vector3>) -> Field<Vector3> {
    points.map(|p| p.normalized())
}

#[test]
fn poles_have_inner_product_one_with_their_faces_vertices_on_a_fixed_mesh() {
    let breath = Breath::new();
    for frame in 0..3 {
        let points = breath.points(core::f64::consts::TAU * frame as f64 / FRAMES as f64);
        let poles = poles(&points, &breath.mesh.faces);
        for (f, corners) in breath.mesh.faces.iter().enumerate() {
            let pairings = corners.map(|v| (poles.site(f) | points.site(v)).s());
            assert!(
                pairings.iter().all(|p| (p - 1.0).abs() < 1e-10),
                "{pairings:?}"
            );
        }
    }
}

#[test]
fn points_on_the_sphere_give_its_delaunay_mesh_with_poles_as_circumcentres() {
    let points = directions(POINTS);
    let faces = hull(&points);
    let centres = images(&poles(&points, &faces));
    // Each pole's image is equally far from the three corners of its face: their circumcentre.
    for (f, corners) in faces.iter().enumerate() {
        let distances = corners.map(|v| (points.site(v) - centres.site(f)).norm());
        assert!(distances.iter().all(|d| (d - distances[0]).abs() < 1e-10));
    }
}

#[test]
fn the_figure_and_two_frames_draw() {
    let shown = super::shown();
    for t in [0.0, 0.08] {
        gax_numga_examples::app::assert_draws_with(super::draw, &shown, t);
    }
}

#[test]
fn the_nearest_rescue_unfolds_the_mesh_steadily() {
    let rescue = Rescue::new();
    let mesh = &rescue.mesh;
    let frames: Vec<Field<Vector3>> = (0..FRAMES)
        .map(|k| rescue.points(k as f64 / (FRAMES - 1) as f64))
        .collect();
    // The crossed dual edges are exactly the folded ones.
    let crossings = |points: &Field<Vector3>| -> Vec<bool> {
        dual_lengths(points, mesh)
            .iter()
            .map(|l| l.s() < 0.0)
            .collect()
    };
    let ones = Field::from_fn(POINTS, |_| Scalar3::new(1.0));
    let folded: Vec<bool> = (&fold_map(&mesh.vertices, mesh) * &ones)
        .iter()
        .map(|f| f.s() < 0.0)
        .collect();
    assert_eq!(crossings(&frames[0]), folded);
    let crossed: Vec<usize> = frames
        .iter()
        .map(|p| crossings(p).iter().filter(|&&c| c).count())
        .collect();
    assert!(crossed[0] > 0 && crossed[FRAMES - 1] == 0, "{crossed:?}");
    assert!(
        crossed.windows(2).all(|pair| pair[1] <= pair[0]),
        "{crossed:?}"
    );
    // Every image on the sphere is held.
    let held = images(&frames[0]).zip_with(&images(&frames[FRAMES - 1]), |a, b| a - b);
    assert!(held.iter().all(|d| d.norm() < 1e-12));
    // numga's further check: at the rescue every pole lies inside its face.
    let last = &frames[FRAMES - 1];
    let sides = sides(last, &mesh.faces, &poles(last, &mesh.faces));
    assert!(sides.iter().all(|side| side.iter().all(|s| s.s() > 0.0)));
}

/// The mesh with one edge turned to join the other two corners of the faces beside it.
fn flip(mesh: &Mesh, edge: usize) -> Mesh {
    let [first, second] = beside(mesh)[edge];
    let [mut a, mut b] = mesh.edges[edge];
    let (c, d) = (
        opposite(mesh.faces[first], [a, b]),
        opposite(mesh.faces[second], [a, b]),
    );
    // The first face runs from `a` to `b`, counter-clockwise.
    let face = mesh.faces[first];
    let at = face
        .iter()
        .position(|&v| v == a)
        .expect("an end of the edge");
    if face[(at + 1) % 3] != b {
        (a, b) = (b, a);
    }
    let mut faces = mesh.faces.clone();
    faces[first] = [a, d, c];
    faces[second] = [d, b, c];
    Mesh::new(mesh.vertices.clone(), faces)
}

/// numga asserts a negative margin for its own draw of eight flips, which lands one flipped edge on
/// an edge already there (a mesh with four faces at one edge, which `Mesh` cannot hold), and
/// scipy's `linprog` finds -0.5. The flips drawn here leave a closed mesh of 90 distinct edges
/// that no weights within their bounds make convex: -0.0032, which `linprog` finds for this mesh
/// too (to 2e-11). On numga's rescue mesh the barrier and `linprog` agree on the widest margin to
/// 1e-10, and the rescue and SLSQP's to 1e-9.
#[test]
fn a_mesh_that_is_no_hull_at_any_distances_has_no_rescue() {
    // The sphere's mesh with so many random edges flipped, from this seed.
    let (flips, seed) = (8, 6);
    let mut draws = rng(seed);
    let points = directions(POINTS);
    let mut mesh = Mesh::new(points.clone(), hull(&points));
    for _ in 0..flips {
        let edge = (draws.uniform() * mesh.edges.len() as f64) as usize;
        mesh = flip(&mesh, edge);
    }
    let (_, margin) = widest(&unit_rows(&fold_map(&points, &mesh)));
    assert!(margin < 0.0, "{margin}");
}
