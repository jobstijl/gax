//! What numga's `surfaces/projective_loop` shares with `surfaces/motor_loop`: Loop's subdivision
//! of a field of PGA3D elements (points with their weights, or motors), its steps composed into
//! one map, and the refined surfaces drawn shaded under their control nets.

use core::ops::{Add, Mul, Sub};
use gax::Extensor;
use gax::field::{Field, SparseMap};
use gax::pga3d::Point;
use gax_numga_examples::mesh::{Mesh, ScalarOf, scalar};
use gax_numga_examples::{Camera, Lens, Light, Rect, Scene3, palette};

/// So many Loop steps from a mesh, the `pinned` vertices staying where they are, each step's
/// averages taken back into their kind by `settle` (a motor normalized; a point kept as it is):
/// the steps composed into one map, from the mesh's vertices to the refined mesh's, and the
/// refined mesh.
pub fn refine<V>(
    mesh: &Mesh<V>,
    pinned: &[bool],
    depth: usize,
    settle: impl Fn(V) -> V,
) -> (SparseMap<ScalarOf<V>>, Mesh<V>)
where
    V: Extensor<Slots = (), Coef = f64> + Add<Output = V> + Sub<Output = V>,
    ScalarOf<V>: Mul<V, Output = V> + Mul<Output = ScalarOf<V>> + Add<Output = ScalarOf<V>>,
{
    let ones = Field::from_fn(mesh.vertices.len(), |_| scalar::<V>(1.0));
    let mut composed = SparseMap::from_diagonal(&ones);
    let (mut mesh, mut pinned) = (mesh.clone(), pinned.to_vec());
    for _ in 0..depth {
        let step = mesh.loop_step(&pinned);
        let vertices = (&step * &mesh.vertices).map(&settle);
        composed = &step * &composed;
        pinned = mesh.split_pins(&pinned);
        mesh = Mesh::new(vertices, mesh.split_faces());
    }
    (composed, mesh)
}

/// A point of space with `f64` coordinates, of any weight.
pub type Point64 = Point<(), f64>;

/// The largest distance between where two fields put the same site, each point divided by its
/// weight: the length of the direction between them.
pub fn separation(first: &Field<Point64>, second: &Field<Point64>) -> f64 {
    let gap = first.zip_with(second, |a, b| a.unitized() - b.unitized());
    gap.iter().map(Point64::ideal_norm).fold(0.0, f64::max)
}

/// A surface of points in the scene, each face in its own light and lit, covering what is
/// behind it.
pub fn surface(
    s: &mut Scene3,
    points: &Field<Point64>,
    faces: &[[usize; 3]],
    light: impl Fn(usize) -> Light,
) {
    let at = |v: usize| points.site(v).unitized();
    for (f, &[a, b, c]) in faces.iter().enumerate() {
        let shade = s.lit(at(a), at(b), at(c), light(f));
        s.tri(at(a), at(b), at(c), shade, 1.0);
    }
}

/// The width of a control net's strokes.
const NET_WIDTH: f32 = 1.0;

/// A control net: the edges between points, as one thin stroke.
pub fn net(s: &mut Scene3, points: &Field<Point64>, edges: &[[usize; 2]]) {
    let segments: Vec<[Point64; 2]> = edges
        .iter()
        .map(|&[a, b]| [a, b].map(|v| points.site(v).unitized()))
        .collect();
    s.stroke(&segments, NET_WIDTH, palette::ink().faded(0.35));
}

/// The camera's vertical field of view; how far it first stands from the points' centre, in
/// units of their reach from it; and the share of the panel's half-width and half-height the
/// points may fill.
const FIELD_OF_VIEW: f32 = 0.6;
const FRAMING: f32 = 4.4;
const FILL: f32 = 0.85;

/// A camera on the panel looking at the middle of the points' bounding box from the azimuth
/// and elevation, far enough to fit them: first at their reach, then moved back if they
/// overflow the panel's narrower side (their size on the canvas falls about as one over the
/// distance).
pub fn framing(rect: Rect, points: &[Point64], azimuth: f32, elevation: f32) -> Camera {
    let coordinates: Vec<[f64; 3]> = points.iter().map(|p| p.to_euclidean()).collect();
    let bound = |axis: usize, pick: fn(f64, f64) -> f64, from: f64| {
        coordinates.iter().map(|c| c[axis]).fold(from, pick)
    };
    let middle = [0, 1, 2].map(|axis| {
        (bound(axis, f64::min, f64::INFINITY) + bound(axis, f64::max, f64::NEG_INFINITY)) / 2.0
    });
    let centre = Point64::xyz(middle[0], middle[1], middle[2]);
    let reach = points
        .iter()
        .map(|&p| (p.unitized() - centre).ideal_norm())
        .fold(0.0, f64::max);
    let lens = Lens::Perspective(FIELD_OF_VIEW);
    let aim = |distance: f32| Camera::orbit(rect, centre, distance, azimuth, elevation, lens);
    let first = FRAMING * reach as f32;
    let camera = aim(first);
    // How far each point lies from the panel's centre, in its half-width or half-height.
    let filled = |&p: &Point64| {
        let offset = camera.px(p)? - rect.centre();
        let [x, y] = [offset.e20() / rect.width(), offset.e01() / rect.height()];
        Some(2.0 * x.abs().max(y.abs()))
    };
    let spread = points.iter().filter_map(filled).fold(0.0, f32::max);
    aim(first * (spread / FILL).max(1.0))
}
