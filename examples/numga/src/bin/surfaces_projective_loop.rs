//! numga's `surfaces/projective_loop`: Loop's subdivision of weighted points of projective space,
//! in PGA3D.
//!
//! A Loop step splits every triangle of a closed mesh in four. Each vertex moves to an average of
//! itself and its neighbours, and each edge gets a new vertex, an average of its two ends and the
//! two vertices across from it. The averages are a field map on the vertex sites
//! ([`Mesh::loop_step`]), with scalar cells. Here the sites hold PGA3D points, each with its own
//! weight, and the step adds and scales the points as they are: a heavy vertex pulls the surface
//! towards itself, a light one lets it go. A projective map acts on the points at the sites and
//! the step on the sites themselves, so the two commute, whatever the weights; averaging
//! positions instead commutes with affine maps only.
//!
//! The perspective is a central collineation, a map `Point <- Point` built on an open point
//! (`Point::slot()`): each point moves along its line through the centre by the centre times
//! its pairing with the axis plane, `x + centre (axis & x) strength`. Points on the axis stay.
//!
//! The animation sweeps the weight of the octahedron's top vertex from light to heavy and back
//! (left), and shows the stretched octahedron refined after and before the perspective, as points
//! with their weights (middle) and as positions (right), each face coloured by the gap between
//! the two orders: none for the weighted points, plain to see for the positions.

use gax_numga_examples::shared::surfaces_loop;

use gax::field::{Field, SparseMap};
use gax::pga3d::Scalar;
use gax_numga_examples::label_under;
use gax_numga_examples::mesh::Mesh;
use gax_numga_examples::plot::panel_row;
use gax_numga_examples::scene3::panel3;
use gax_numga_examples::{Anim, Canvas, Light, Rect, backdrop, caption, palette, run};
use std::sync::OnceLock;
use surfaces_loop::{Point64, framing, net, separation, surface};

mod projective {
    use super::surfaces_loop::{Point64, refine};
    use gax::field::{Field, SparseMap};
    use gax::pga3d::{Plane, Point, Scalar};
    use gax_numga_examples::mesh::Mesh;

    /// A map on points, `Point <- Point`.
    pub type PointMap = Point<(Point,)>;

    /// So many Loop steps from a closed mesh, nothing pinned, composed into one map from its
    /// vertices to those of the refined mesh, and the refined mesh.
    pub fn subdivision(mesh: &Mesh<Point64>, depth: usize) -> (SparseMap<Scalar>, Mesh<Point64>) {
        let pinned = vec![false; mesh.vertices.len()];
        refine(mesh, &pinned, depth, |p| p)
    }

    /// The points divided by their weights: their positions, at unit weight.
    pub fn unit(points: &Field<Point64>) -> Field<Point64> {
        points.map(Point64::unitized)
    }

    /// The central collineation of a centre and an axis plane, as a map on points: each point
    /// moves along its line through the centre, by the centre times its pairing with the axis;
    /// points on the axis stay where they are.
    pub fn collineation(centre: Point64, axis: Plane, strength: f64) -> PointMap {
        let open = Point64::slot();
        open + centre * (axis & open) * strength
    }
}

mod scenes {
    use super::projective::*;
    use super::surfaces_loop::Point64;
    use gax::field::{Field, SparseMap};
    use gax::pga3d::{Plane, Scalar};
    use gax_numga_examples::mesh::Mesh;
    use gax_numga_examples::signal::phasor;

    /// The octahedron: its vertices, the one at `TOP` on top, and its faces, counter-clockwise
    /// from outside.
    pub const OCTAHEDRON: [[f64; 3]; 6] = [
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ];
    pub const TOP: usize = 4;
    pub const FACES: [[usize; 3]; 8] = [
        [0, 2, 4],
        [2, 1, 4],
        [1, 3, 4],
        [3, 0, 4],
        [2, 0, 5],
        [1, 2, 5],
        [3, 1, 5],
        [0, 3, 5],
    ];
    /// Loop steps taken.
    pub const DEPTH: usize = 3;
    /// The octahedron stretched and set in front of the perspective, and the perspective: its
    /// centre at the origin, its axis the plane `z = 0`, and its strength, one over the distance
    /// at which it halves sizes.
    pub const STRETCH: [f64; 3] = [1.0, 1.0, 1.5];
    pub const OFFSET: [f64; 3] = [0.0, 0.0, 1.6];
    pub const STRENGTH: f64 = 1.0;
    /// The top vertex's weight swept from light to heavy and back.
    pub const SWEEP: (f64, f64) = (0.25, 4.0);

    /// The octahedron's vertices at the weights.
    pub fn octahedron(weights: &[f64; 6]) -> Field<Point64> {
        let at = |v: usize| {
            let [x, y, z] = OCTAHEDRON[v];
            Point64::xyz(x, y, z) * weights[v]
        };
        Field::from_fn(OCTAHEDRON.len(), at)
    }

    /// The octahedron at unit weights, as a mesh.
    pub fn control() -> Mesh<Point64> {
        Mesh::new(octahedron(&[1.0; 6]), FACES.to_vec())
    }

    /// The octahedron's refinement: the composed map and the refined mesh.
    pub fn refinement() -> (SparseMap<Scalar>, Mesh<Point64>) {
        subdivision(&control(), DEPTH)
    }

    /// The top vertex's weight at a phase of the sweep: out and back along a cosine of its
    /// logarithm, lightest at the start.
    pub fn top_weight(phase: f64) -> f64 {
        let (light, heavy) = SWEEP;
        let middle = (light.ln() + heavy.ln()) / 2.0;
        (middle - (heavy / light).ln() / 2.0 * phasor(phase).e20()).exp()
    }

    /// The octahedron with its top vertex at a phase of the sweep.
    pub fn swept(phase: f64) -> Field<Point64> {
        let mut weights = [1.0; 6];
        weights[TOP] = top_weight(phase);
        octahedron(&weights)
    }

    /// The octahedron after the perspective, and the surface refined after and before it, as
    /// points with their weights and as positions, each point divided by its weight before the
    /// refinement.
    pub struct Perspective {
        pub seen: Field<Point64>,
        pub weighted: [Field<Point64>; 2],
        pub positional: [Field<Point64>; 2],
    }

    pub fn perspective() -> Perspective {
        let (subdivide, _) = refinement();
        let shape = Field::from_fn(OCTAHEDRON.len(), |v| {
            let [x, y, z] =
                [0, 1, 2].map(|axis| OCTAHEDRON[v][axis] * STRETCH[axis] + OFFSET[axis]);
            Point64::xyz(x, y, z)
        });
        let origin = Point64::origin();
        let floor = Plane::new(0.0, 0.0, 1.0, 0.0);
        let lens = collineation(origin, floor, STRENGTH);
        let seen = |points: &Field<Point64>| points.map(|p| lens.of(p));
        let refined = |points: &Field<Point64>| &subdivide * points;
        Perspective {
            seen: seen(&shape),
            weighted: [refined(&seen(&shape)), seen(&refined(&shape))],
            positional: [refined(&unit(&seen(&shape))), seen(&refined(&unit(&shape)))],
        }
    }
}

use scenes::*;

/// Everything shown, computed once: the refined mesh and its map, and the perspective's orders.
struct Shown {
    control: Mesh<Point64>,
    refined: Mesh<Point64>,
    subdivide: SparseMap<Scalar>,
    perspective: Perspective,
}

fn shown() -> &'static Shown {
    static SHOWN: OnceLock<Shown> = OnceLock::new();
    SHOWN.get_or_init(|| {
        let (subdivide, refined) = refinement();
        Shown {
            control: control(),
            refined,
            subdivide,
            perspective: perspective(),
        }
    })
}

/// One sweep: numga's 48 frames of 60 ms.
const SECONDS: f32 = 2.88;
/// The view: numga's elevation and azimuth.
const ELEVATION: f32 = 0.31;
const AZIMUTH: f32 = -1.05;

/// The light of a face where the two orders agree, and where they are furthest apart.
fn agreed() -> Light {
    palette::SKY.faded(0.55)
}
fn apart() -> Light {
    palette::ORANGE.faded(0.8)
}

/// A refined surface over its control net, its faces in the given lights.
fn shaded(
    c: &mut Canvas,
    rect: Rect,
    control: &Field<Point64>,
    surface_points: &Field<Point64>,
    light: impl Fn(usize) -> Light,
) {
    let s = shown();
    let cam = framing(rect, control.sites(), AZIMUTH, ELEVATION);
    panel3(c, rect, cam, |scene| {
        surface(scene, surface_points, &s.refined.faces, light);
        net(scene, control, &s.control.edges);
    });
}

/// Each face's gap between two refinements: the mean distance between where they put its
/// corners.
fn face_gaps(orders: &[Field<Point64>; 2], faces: &[[usize; 3]]) -> Vec<f64> {
    let gaps = orders[0].zip_with(&orders[1], |a, b| a.unitized() - b.unitized());
    let gap = |v: usize| gaps.site(v).ideal_norm();
    faces
        .iter()
        .map(|f| f.iter().map(|&v| gap(v)).sum::<f64>() / 3.0)
        .collect()
}

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let s = shown();
    let [left, middle, right] = panel_row(c, 6.0);

    // The sweep: the top vertex's weight, out and back.
    let phase = core::f64::consts::TAU * f64::from(t / SECONDS);
    let control = swept(phase);
    let refined = &s.subdivide * &control;
    shaded(c, left, &control, &refined, |_| agreed());
    let label = format!("TOP VERTEX OF WEIGHT {:.2}", top_weight(phase));
    label_under(c, left, &label, 30.0, 0.9);

    // The two orders after the perspective, on one scale of gaps.
    let p = &s.perspective;
    let cases = [
        (&p.weighted, "POINTS WITH WEIGHTS"),
        (&p.positional, "POSITIONS"),
    ];
    let gaps = cases.map(|(orders, _)| face_gaps(orders, &s.refined.faces));
    let largest = gaps.iter().flatten().copied().fold(0.0, f64::max);
    for (((orders, name), gaps), rect) in cases.into_iter().zip(&gaps).zip([middle, right]) {
        let light = |f: usize| agreed().blend(apart(), (gaps[f] / largest) as f32);
        shaded(c, rect, &p.seen, &orders[0], light);
        let gap = separation(&orders[0], &orders[1]);
        label_under(c, rect, &format!("{name}: GAP {gap:.0E}"), 30.0, 0.9);
    }
    caption(
        c,
        "PROJECTIVE LOOP: SUBDIVIDING WEIGHTED POINTS",
        "LOOP STEPS AS SPARSE MAPS ON PGA3D POINTS; RIGHT: REFINED AFTER VS BEFORE A PERSPECTIVE, ORANGE WHERE THEY PART",
    );
}

fn main() {
    run(Anim::new("projective loop", SECONDS).size(960, 540), draw);
}

#[cfg(test)]
mod tests {
    use super::projective::*;
    use super::scenes::*;
    use super::surfaces_loop::{Point64, separation};
    use gax::field::Field;
    use std::collections::BTreeSet;

    /// The weights of the vertices per case: all alike, the top one heavy, the top one light.
    const WEIGHTS: [[f64; 6]; 3] = [
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 4.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 0.25, 1.0],
    ];

    /// The weighted octahedra of each case and the surfaces they refine to.
    fn weighted() -> Vec<(Field<Point64>, Field<Point64>)> {
        let (subdivide, _) = refinement();
        let case = |weights: &[f64; 6]| {
            let control = octahedron(weights);
            let surface = &subdivide * &control;
            (control, surface)
        };
        WEIGHTS.iter().map(case).collect()
    }

    #[test]
    fn refining_commutes_with_perspective_for_weighted_points_only() {
        let Perspective {
            weighted,
            positional,
            ..
        } = perspective();
        assert!(separation(&weighted[0], &weighted[1]) < 1e-12);
        assert!(separation(&positional[0], &positional[1]) > 1e-2);
        gax_numga_examples::app::assert_draws(super::draw, 1.0);
    }

    #[test]
    fn a_heavy_vertex_pulls_the_surface_and_a_light_one_releases_it() {
        let cases = weighted();
        let top = |surface: &Field<Point64>| {
            surface
                .iter()
                .map(|p| p.to_euclidean()[2])
                .fold(f64::NEG_INFINITY, f64::max)
        };
        let tops: Vec<f64> = cases.iter().map(|(_, surface)| top(surface)).collect();
        assert!(tops[1] > tops[0] && tops[0] > tops[2], "{tops:?}");
        // With all weights alike the surface is Loop subdivision of the positions.
        let alike = &cases[0].1;
        assert!(separation(alike, &unit(alike)) < 1e-12);
        // The sweep draws at two of its frames.
        for t in [0.0, 0.06] {
            gax_numga_examples::app::assert_draws(super::draw, t);
        }
    }

    #[test]
    fn a_loop_step_keeps_unit_weights_and_splits_every_face_in_four() {
        let control = control();
        let finer = control.subdivide_loop(&[false; 6]);
        assert!(
            finer
                .vertices
                .iter()
                .all(|p| (p.e123() - 1.0).abs() < 1e-12)
        );
        assert_eq!(finer.faces.len(), 4 * FACES.len());
        // Every edge of the finer mesh lies between two of its faces, once each way round.
        let sides: Vec<[usize; 2]> = finer
            .faces
            .iter()
            .flat_map(|&[a, b, c]| [[a, b], [b, c], [c, a]])
            .collect();
        let distinct: BTreeSet<[usize; 2]> = sides.iter().copied().collect();
        assert!(distinct.len() == sides.len() && sides.len() == 2 * finer.edges.len());
    }
}
