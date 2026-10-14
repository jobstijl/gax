//! numga's `surfaces/reciprocal`: a planar mesh lifted by heights at its vertices, the stresses
//! its creases carry, and its reciprocal diagram, in VGA3D.
//!
//! Heights at the vertices of a planar triangle mesh lift it to a polyhedral terrain, a piecewise
//! linear Airy stress function. Each face is a plane, `height = slope | (point + e3)`, and the
//! slope is linear in the heights: the corners' heights times the reciprocal frame of the corners
//! lifted to unit height, the bivector of two corners over the trivector of all three. Across an
//! interior edge the terrain creases, and the crease is a bar of the planar truss: the jump in
//! slope over the edge's length is its stress, positive in tension where the terrain is convex
//! (the trivector of the bar, the jump and `e3`, in units of the unit volume). The slopes of the
//! faces are the vertices of the reciprocal diagram, and the jump across each edge is its
//! reciprocal edge, perpendicular to the bar and as long as the bar's force. Every interior
//! vertex is in equilibrium; the boundary carries the loads. Lifted onto the paraboloid, the
//! reciprocal diagram is the Voronoi diagram and the stresses are the cotangent weights.
//!
//! The animation lets the heights of a jittered grid breathe about the paraboloid, each vertex
//! with its own phase: on the left the terrain with its creases, on the right the truss and its
//! reciprocal diagram. Bars in tension glow blue, in compression orange.

use gax::field::Field;
use gax_numga_examples::mesh::{Mesh, Scalar3, Vector3};
use gax_numga_examples::scene3::panel3;
use gax_numga_examples::{
    Anim, Axes, Camera, Canvas, Lens, Light, Marker, Point2, Point3, Rect, Scene3, backdrop,
    caption, label_under, palette, run,
};
use std::sync::OnceLock;

mod reciprocal {
    use gax::field::Field;
    use gax::vga3d::Pseudoscalar;
    use gax_numga_examples::mesh::{Mesh, Scalar3, Vector3, from_columns};

    /// Up: a point of the plane lifted to unit height is the point plus `e3`.
    pub const UP: Vector3 = Vector3::new(0.0, 0.0, 1.0);

    /// The interior edges, and the faces on their left and on their right, seen along each edge
    /// from its lower vertex to its higher.
    pub struct Beside {
        pub inner: Vec<usize>,
        pub left: Vec<usize>,
        pub right: Vec<usize>,
    }

    pub fn beside(mesh: &Mesh) -> Beside {
        let mut left = vec![None; mesh.edges.len()];
        let mut right = vec![None; mesh.edges.len()];
        for (f, edges) in mesh.face_edges.iter().enumerate() {
            for (k, &e) in edges.iter().enumerate() {
                // A counter-clockwise face runs along the edges it has on its left.
                let along = mesh.face_edge_orientation[f][k] > 0.0;
                let side = if along { &mut left } else { &mut right };
                side[e] = Some(f);
            }
        }
        let mut beside = Beside {
            inner: Vec::new(),
            left: Vec::new(),
            right: Vec::new(),
        };
        for e in 0..mesh.edges.len() {
            if let (Some(l), Some(r)) = (left[e], right[e]) {
                beside.inner.push(e);
                beside.left.push(l);
                beside.right.push(r);
            }
        }
        beside
    }

    /// The plane of each face of the terrain, through its corners at their heights, as the
    /// vector whose inner product with a point lifted to unit height is the height there: the
    /// slope across the plane and the height at the origin. The corners' heights times the
    /// reciprocal frame of the corners lifted to unit height, the three vectors with inner
    /// product one with their own corner and zero with the other two: linear in the heights.
    pub fn slopes(mesh: &Mesh, heights: &Field<Scalar3>) -> Field<Vector3> {
        let lifted = mesh.vertices.map(|v| v + UP);
        let planes = mesh.faces.iter().map(|&corners| {
            let [a, b, c] = corners.map(|v| lifted.site(v));
            let volume = (a ^ b ^ c).inverse();
            let frame = [(b ^ c) * volume, (c ^ a) * volume, (a ^ b) * volume];
            let [ha, hb, hc] = corners.map(|v| heights.site(v).s());
            frame[0] * ha + frame[1] * hb + frame[2] * hc
        });
        planes.collect()
    }

    /// The vertices of the reciprocal diagram: the slope of each face, its height at the origin
    /// left out (the plane's vector rejected from `e3`).
    pub fn reciprocal(mesh: &Mesh, heights: &Field<Scalar3>) -> Field<Vector3> {
        slopes(mesh, heights).map(|p| (p ^ UP) | UP)
    }

    /// The stress of each interior edge: the jump in slope from the face on its left to the face
    /// on its right, over the edge's length, positive in tension, where the terrain bends convex.
    pub fn stresses(mesh: &Mesh, heights: &Field<Scalar3>) -> Field<Scalar3> {
        let Beside { inner, left, right } = beside(mesh);
        let (planes, bars) = (slopes(mesh, heights), mesh.edge_vectors());
        let unit_volume = Pseudoscalar::<(), f64>::new(1.0).reverse();
        let stress = |i: usize| {
            let jump = planes.site(left[i]) - planes.site(right[i]);
            let bar = bars.site(inner[i]);
            (bar ^ jump ^ UP) * unit_volume * (1.0 / bar.norm_squared())
        };
        (0..inner.len()).map(stress).collect()
    }

    /// The largest net force on an interior vertex from the bars meeting there, each pulling its
    /// ends together along itself by its stress times its length.
    pub fn unbalanced(mesh: &Mesh, stresses: &Field<Scalar3>) -> f64 {
        let inner = beside(mesh).inner;
        let bars = mesh.edge_vectors();
        let pulls: Field<Vector3> = inner
            .iter()
            .zip(stresses.iter())
            .map(|(&e, s)| bars.site(e) * s.s())
            .collect();
        // A bar pulls its lower end towards its higher, and the higher back.
        let ends: Vec<[usize; 2]> = inner.iter().map(|&e| mesh.edges[e]).collect();
        let pulled = from_columns(&ends, mesh.vertices.len(), |_, end| {
            Scalar3::new([1.0, -1.0][end])
        });
        let net = &pulled.reverse() * &pulls;
        let rim = mesh.boundary_vertices();
        let interior = (0..net.len()).filter(|&v| !rim[v]);
        interior.map(|v| net.site(v).norm()).fold(0.0, f64::max)
    }
}

mod scenes {
    use gax::field::Field;
    use gax_numga_examples::mesh::{Mesh, Scalar3, Vector3};
    use gax_numga_examples::rng::{Draw, rng};
    use gax_numga_examples::signal::wave;

    /// The grid: its vertices per side across the square from -1 to 1, how far its interior
    /// vertices are nudged, and the seed of the nudges.
    pub const SIDE: usize = 5;
    pub const JITTER: f64 = 0.08;
    pub const GRID_SEED: u64 = 11;
    /// How far each height breathes about the paraboloid, and the seed of the phases.
    pub const AMPLITUDE: f64 = 0.02;
    pub const PHASE_SEED: u64 = 3;

    /// The jittered grid's mesh, and its points' heights on the paraboloid, half their squared
    /// distance from the centre.
    pub fn grid() -> (Mesh, Field<Scalar3>) {
        let mut nudges = rng(GRID_SEED);
        let across = |i: usize| -1.0 + 2.0 * i as f64 / (SIDE - 1) as f64;
        let mut points = Vec::new();
        for i in 0..SIDE {
            for j in 0..SIDE {
                let point = Vector3::new(across(i), across(j), 0.0);
                let inside = i > 0 && j > 0 && i < SIDE - 1 && j < SIDE - 1;
                let nudge = Vector3::new(nudges.normal(), nudges.normal(), 0.0) * JITTER;
                points.push(if inside { point + nudge } else { point });
            }
        }
        let mesh = Mesh::triangulate(Field::new(points));
        let heights = mesh.vertices.map(|v| Scalar3::new(v.norm_squared() / 2.0));
        (mesh, heights)
    }

    /// The grid, its heights on the paraboloid, and a phase for each height's breath.
    pub struct Breathing {
        pub mesh: Mesh,
        pub rest: Field<Scalar3>,
        pub phases: Vec<f64>,
    }

    impl Breathing {
        pub fn new() -> Breathing {
            let (mesh, rest) = grid();
            let mut draws = rng(PHASE_SEED);
            let phases = (0..rest.len())
                .map(|_| draws.range(0.0, core::f64::consts::TAU))
                .collect();
            Breathing { mesh, rest, phases }
        }

        /// The heights at a point of the breath, `angle` from its start.
        pub fn heights(&self, angle: f64) -> Field<Scalar3> {
            let breath = |v: usize| AMPLITUDE * wave(angle + self.phases[v]);
            Field::from_fn(self.rest.len(), |v| {
                self.rest.site(v) + Scalar3::new(breath(v))
            })
        }
    }
}

use reciprocal::*;
use scenes::*;

fn scene() -> &'static Breathing {
    static SCENE: OnceLock<Breathing> = OnceLock::new();
    SCENE.get_or_init(Breathing::new)
}

/// One breath: numga's 48 frames of 80 ms.
const SECONDS: f32 = 3.84;
/// The stresses at which the bars' light saturates, compression far weaker than tension, and the
/// light of a bar at no stress.
const TENSION: f64 = 1.5;
const COMPRESSION: f64 = 0.3;
const UNSTRESSED: f32 = 0.15;
/// The widths of the bars and of the rim.
const BAR_WIDTH: f32 = 2.2;
const RIM_WIDTH: f32 = 1.0;
/// The view of the terrain, how far it stands, and how far the truss's panel reaches about the
/// centre.
const ELEVATION: f32 = 0.49;
const AZIMUTH: f32 = -1.05;
const DISTANCE: f32 = 6.6;
const REACH: f32 = 1.45;

/// A bar's light: blue in tension, orange in compression, brighter as the stress nears its scale.
fn stress_light(stress: f64) -> Light {
    let (base, scale) = if stress > 0.0 {
        (palette::sky(), TENSION)
    } else {
        (palette::orange(), COMPRESSION)
    };
    let strength = (stress.abs() / scale).min(1.0) as f32;
    base.faded(UNSTRESSED + (1.0 - UNSTRESSED) * strength)
}

/// A point of the plane, as drawn.
fn planar(v: Vector3) -> Point2 {
    Point2::xy(v.e1() as f32, v.e2() as f32)
}

/// A vertex lifted to its height, as drawn.
fn lifted(v: Vector3, height: Scalar3) -> gax::pga3d::Point<(), f64> {
    gax::pga3d::Point::xyz(v.e1(), v.e2(), height.s())
}

/// The terrain the heights lift the mesh to, its creases coloured by stress.
fn terrain(
    c: &mut Canvas,
    rect: Rect,
    mesh: &Mesh,
    heights: &Field<Scalar3>,
    stresses: &Field<Scalar3>,
) {
    let target = Point3::xyz(0.0, 0.0, 0.4);
    let cam = Camera::orbit(
        rect,
        target,
        DISTANCE,
        AZIMUTH,
        ELEVATION,
        Lens::Perspective(0.45),
    );
    let at = |v: usize| lifted(mesh.vertices.site(v), heights.site(v));
    panel3(c, rect, cam, |s: &mut Scene3| {
        for &[a, b, c] in &mesh.faces {
            let shade = s.lit(at(a), at(b), at(c), palette::blue().faded(0.5));
            s.tri(at(a), at(b), at(c), shade, 0.35);
        }
        for (&e, stress) in beside(mesh).inner.iter().zip(stresses.iter()) {
            let [a, b] = mesh.edges[e];
            s.seg(at(a), at(b), BAR_WIDTH, stress_light(stress.s()));
        }
        for e in mesh.boundary_edges() {
            let [a, b] = mesh.edges[e];
            s.seg(at(a), at(b), RIM_WIDTH, palette::grid());
        }
    });
}

/// The planar truss, its bars coloured by stress, and the reciprocal diagram over it: a dual
/// vertex at each face's slope and an edge across every bar.
fn truss(
    c: &mut Canvas,
    rect: Rect,
    mesh: &Mesh,
    stresses: &Field<Scalar3>,
    duals: &Field<Vector3>,
) {
    let ax = Axes::equal(rect, Point2::xy(0.0, 0.0), REACH);
    let Beside { inner, left, right } = beside(mesh);
    let point = |v: usize| planar(mesh.vertices.site(v));
    for (&e, stress) in inner.iter().zip(stresses.iter()) {
        let [a, b] = mesh.edges[e];
        ax.line(c, point(a), point(b), BAR_WIDTH, stress_light(stress.s()));
    }
    for e in mesh.boundary_edges() {
        let [a, b] = mesh.edges[e];
        ax.line(c, point(a), point(b), RIM_WIDTH, palette::grid());
    }
    let dual = |f: usize| planar(duals.site(f));
    for (&l, &r) in left.iter().zip(&right) {
        ax.line(c, dual(l), dual(r), 1.0, palette::ink().faded(0.7));
    }
    let vertices: Vec<Point2> = (0..duals.len()).map(dual).collect();
    ax.scatter(c, &vertices, Marker::Dot, 4.0, palette::ink());
    ax.legend(
        c,
        &[
            ("TENSION", palette::sky()),
            ("COMPRESSION", palette::orange()),
            ("RECIPROCAL", palette::ink()),
        ],
    );
}

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let scene = scene();
    let angle = core::f64::consts::TAU * f64::from(t / SECONDS);
    let heights = scene.heights(angle);
    let stresses = stresses(&scene.mesh, &heights);
    let duals = reciprocal(&scene.mesh, &heights);
    let screen = c.rect();
    let below = Rect::new(0.0, screen.height() * 0.12, screen.width(), screen.height());
    let [left, right] = [0, 1].map(|i| below.column(i, 2).inset(8.0, 8.0, 8.0, 8.0));
    terrain(c, left, &scene.mesh, &heights, &stresses);
    truss(c, right, &scene.mesh, &stresses, &duals);
    let net = unbalanced(&scene.mesh, &stresses);
    label_under(
        c,
        left,
        "THE LIFTED TERRAIN: ITS CREASES ARE BARS",
        30.0,
        0.8,
    );
    let truss_label = format!("THE TRUSS AND ITS RECIPROCAL, NET FORCE {net:.0E}");
    label_under(c, right, &truss_label, 30.0, 0.8);
    caption(
        c,
        "RECIPROCAL DIAGRAMS: A LIFTED MESH AND THE STRESSES OF ITS CREASES",
        "FACE SLOPES FROM THE RECIPROCAL FRAME OF LIFTED CORNERS; STRESS: THE TRIVECTOR OF BAR, JUMP AND E3 (VGA3D)",
    );
}

fn main() {
    run(Anim::new("reciprocal", SECONDS).size(960, 540), draw);
}

#[cfg(test)]
mod tests {
    use super::reciprocal::*;
    use super::scenes::*;
    use gax::field::Field;
    use gax_numga_examples::mesh::{Mesh, Scalar3};

    /// The frames of one breath numga checks.
    const FRAMES: usize = 48;

    /// The heights at each of numga's frames of one breath.
    fn frames(breathing: &Breathing) -> Vec<Field<Scalar3>> {
        let angle = |k: usize| core::f64::consts::TAU * k as f64 / FRAMES as f64;
        (0..FRAMES).map(|k| breathing.heights(angle(k))).collect()
    }

    /// How far the corner across each interior edge on its right lies above the plane of the
    /// face on its left: positive where the terrain bends convex.
    fn folds(mesh: &Mesh, heights: &Field<Scalar3>) -> Field<Scalar3> {
        let Beside { inner, left, right } = beside(mesh);
        let planes = slopes(mesh, heights);
        let fold = |i: usize| {
            let [a, b] = mesh.edges[inner[i]];
            let far = mesh.faces[right[i]].into_iter().find(|&v| v != a && v != b);
            let far = far.expect("a triangle's third corner");
            heights.site(far) - (planes.site(left[i]) | (mesh.vertices.site(far) + UP))
        };
        (0..inner.len()).map(fold).collect()
    }

    #[test]
    fn bars_are_in_tension_at_convex_creases_and_every_interior_vertex_balances() {
        let breathing = Breathing::new();
        let mesh = &breathing.mesh;
        let Beside { inner, left, right } = beside(mesh);
        let bars = mesh.edge_vectors();
        let mut compressed = false;
        for heights in frames(&breathing) {
            let stresses = stresses(mesh, &heights);
            let folds = folds(mesh, &heights);
            // Every bar is in tension exactly where the terrain bends convex.
            for (stress, fold) in stresses.iter().zip(folds.iter()) {
                assert_eq!(stress.s() > 0.0, fold.s() > 0.0);
            }
            compressed |= stresses.iter().any(|s| s.s() < 0.0);
            assert!(unbalanced(mesh, &stresses) < 1e-10);
            // Every reciprocal edge is perpendicular to its bar.
            let duals = reciprocal(mesh, &heights);
            for (i, &e) in inner.iter().enumerate() {
                let across = duals.site(left[i]) - duals.site(right[i]);
                assert!((across | bars.site(e)).s().abs() < 1e-10);
            }
        }
        // Some bars go into compression.
        assert!(compressed);
    }

    #[test]
    fn on_the_paraboloid_the_reciprocal_is_the_voronoi_diagram_and_the_stresses_the_cotangent_weights()
     {
        let (mesh, heights) = grid();
        let inner = beside(&mesh).inner;
        let ratio = mesh.edge_ratio();
        for (&e, stress) in inner.iter().zip(stresses(&mesh, &heights).iter()) {
            assert!((stress - ratio.site(e)).s().abs() < 1e-10);
        }
        // The reciprocal vertices are the circumcentres: equally far from each corner.
        let centres = reciprocal(&mesh, &heights);
        let corners = mesh.corners();
        for f in 0..mesh.faces.len() {
            let reach = |k: usize| {
                let d = centres.site(f) - corners[k].site(f);
                d.norm_squared()
            };
            assert!((reach(0) - reach(1)).abs() < 1e-10 && (reach(0) - reach(2)).abs() < 1e-10);
        }
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws(super::draw, 1.0);
    }
}
