//! What numga's `surfaces/koebe` shares with `surfaces/weighted_dual`: directions spread over the
//! sphere, the convex hull of points around the centre (scipy's `ConvexHull` in numga, written
//! here with gax's orientation predicate), the faces either side of each edge of a closed mesh,
//! and the panels both draw about the unit sphere.
//!
//! Four points `a b c p` span the volume `(b - a) ^ (c - a) ^ (p - a)`, a trivector, positive when
//! `p` lies on the side the face `a b c` turns counter-clockwise towards. The hull grows one point
//! at a time: the faces a new point sees are removed, and the edges of the hole they leave (the
//! horizon) are joined to it.

use gax::field::Field;
use gax_numga_examples::measure::{GOLDEN, lift};
use gax_numga_examples::mesh::{Mesh, Vector3, direction};
use gax_numga_examples::rng::{Draw, rng};
use gax_numga_examples::{Camera, Canvas, Point3, Rect, palette};
use std::collections::BTreeSet;

/// The number of points, how far their directions are nudged off an even spread, and the seed
/// of the nudges.
pub const POINTS: usize = 32;
const JITTER: f64 = 0.03;
const DIRECTION_SEED: u64 = 0;
/// The spread and seed of the random distances whose hull gives an uneven triangulation, every
/// point on it.
const MESH_SPREAD: f64 = 0.06;
const MESH_SEED: u64 = 57;

/// Directions spread evenly over the sphere along a Fibonacci spiral, each nudged a little at
/// random: the `k`-th at the height `1 - (2k + 1) / count`, on the meridian turned by `2π φ`
/// per point.
pub fn directions(count: usize) -> Field<Vector3> {
    let mut nudges = rng(DIRECTION_SEED);
    let up = Vector3::new(0.0, 0.0, 1.0);
    Field::from_fn(count, |k| {
        let turns = k as f64 + 0.5;
        let height = 1.0 - 2.0 * turns / count as f64;
        // The meridian's radius at that height: the sphere's height over the point there.
        let radius = lift(height, 0.0).expect("a height on the sphere").e3();
        let azimuth = core::f64::consts::TAU * GOLDEN * turns;
        let spiral = direction(azimuth) * radius + up * height;
        let nudge = Vector3::new(nudges.normal(), nudges.normal(), nudges.normal());
        (spiral + nudge * JITTER).normalized().into_inner()
    })
}

/// The faces of the hull of the directions at random distances along them: a triangulation of
/// the sphere with triangles of uneven shape.
pub fn uneven_faces(directions: &Field<Vector3>) -> Vec<[usize; 3]> {
    let mut distances = rng(MESH_SEED);
    hull(&directions.map(|d| d * (MESH_SPREAD * distances.normal()).exp()))
}

/// How far `p` lies on the outer side of the face `a b c`: the volume of the four points.
fn outward(points: &Field<Vector3>, [a, b, c]: [usize; 3], p: usize) -> f64 {
    let at = |v: usize| points.site(v) - points.site(a);
    (at(b) ^ at(c) ^ at(p)).e123()
}

/// The faces of the convex hull of points around the centre, counter-clockwise from outside.
/// Points inside the hull are left out of it.
///
/// # Panics
/// If the points all lie in one plane.
pub fn hull(points: &Field<Vector3>) -> Vec<[usize; 3]> {
    let apex = (3..points.len())
        .find(|&p| outward(points, [0, 1, 2], p) != 0.0)
        .expect("points that span space");
    // The first tetrahedron, each face turned so that its fourth corner lies inside.
    let corners = [0, 1, 2, apex];
    let mut faces: Vec<[usize; 3]> = (0..4)
        .map(|left_out| {
            let [a, b, c] = [1, 2, 3].map(|k| corners[(left_out + k) % 4]);
            let inside = corners[left_out];
            if outward(points, [a, b, c], inside) > 0.0 {
                [a, c, b]
            } else {
                [a, b, c]
            }
        })
        .collect();
    for p in (3..points.len()).filter(|&p| p != apex) {
        let (seen, kept): (Vec<[usize; 3]>, Vec<[usize; 3]>) = faces
            .into_iter()
            .partition(|&f| outward(points, f, p) > 0.0);
        faces = kept;
        // The horizon: the edges of the seen faces whose other face is not seen, joined to p.
        let edges: BTreeSet<[usize; 2]> = seen
            .iter()
            .flat_map(|&[a, b, c]| [[a, b], [b, c], [c, a]])
            .collect();
        let horizon = edges.iter().filter(|&&[a, b]| !edges.contains(&[b, a]));
        faces.extend(horizon.map(|&[a, b]| [a, b, p]));
    }
    faces
}

/// The two faces either side of each edge of a closed mesh, the lower first.
pub fn beside<V>(mesh: &Mesh<V>) -> Vec<[usize; 2]>
where
    V: gax::Extensor<Slots = (), Coef = f64>,
{
    let mut faces = vec![Vec::with_capacity(2); mesh.edges.len()];
    for (f, edges) in mesh.face_edges.iter().enumerate() {
        for &e in edges {
            faces[e].push(f);
        }
    }
    faces
        .into_iter()
        .map(|pair| <[usize; 2]>::try_from(pair).expect("a closed mesh: two faces at each edge"))
        .collect()
}

/// The view of both examples: numga's elevation and azimuth, and how far each panel reaches
/// about the centre, in sphere radii.
const ELEVATION: f32 = 0.35;
const AZIMUTH: f32 = -1.05;
const REACH: f32 = 1.35;

/// Canvas units per sphere radius: the sphere's reach fitting the panel.
fn scale(rect: Rect) -> f32 {
    rect.width().min(rect.height()) / (2.0 * REACH)
}

/// A parallel camera on the panel, the sphere's reach fitting it.
pub fn camera(rect: Rect) -> Camera {
    Camera::parallel(rect, scale(rect), AZIMUTH, ELEVATION)
}

/// The direction towards the viewer.
pub fn facing(cam: &Camera) -> Vector3 {
    let [x, y, z] = cam.eye().to_euclidean().map(f64::from);
    Vector3::new(x, y, z).normalized().into_inner()
}

/// A point of space, as drawn.
pub fn drawn(v: Vector3) -> Point3 {
    Point3::xyz(v.e1() as f32, v.e2() as f32, v.e3() as f32)
}

/// A point of the unit sphere as drawn on a stroke, or a break in the stroke where the point
/// faces away from the viewer, behind the sphere.
pub fn near_side(p: Vector3, toward: Vector3) -> Point3 {
    if (p | toward).s() > 0.0 {
        drawn(p)
    } else {
        Point3::xyz(f32::NAN, 0.0, 0.0)
    }
}

/// The sphere as a faint disk with a rim, behind everything in the panel.
pub fn ghost(c: &mut Canvas, cam: &Camera, rect: Rect) {
    let centre = cam
        .px(Point3::xyz(0.0, 0.0, 0.0))
        .expect("the centre in view");
    c.clip(rect);
    c.disk(centre, scale(rect), palette::grid().faded(0.6));
    c.ring(centre, scale(rect), 1.0, palette::grid());
    c.unclip();
}
