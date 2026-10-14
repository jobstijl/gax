//! numga's `surfaces/weighted_dual`: a convex polyhedron, its polar dual about the unit sphere,
//! and both projected onto the sphere, in VGA3D.
//!
//! Points anywhere in space around the centre span a convex hull (`shared/surfaces_hull.rs`, by
//! gax's orientation predicate, where numga calls scipy). Each face of the hull has a pole with
//! respect to the unit sphere, the point whose inner product with every point of the face is
//! one: the face's edges joined to the centre, summed, over the volume of its corners, a
//! bivector over a trivector. The poles are the vertices of the polar dual: a vertex lies on a
//! face exactly when the face's pole has inner product one with the vertex. Seen from the centre,
//! each point has an image on the sphere; the images of the hull are a triangle mesh on the
//! sphere, and the images of the poles its dual vertices.
//!
//! How far a point lies from the centre does not move its image, but it tilts the faces around
//! it, and with them the dual vertices. Along a fixed direction a point's weight is one over its
//! distance, and every edge's fold (a sum of volumes of four directions times their weights) is
//! linear in the weights, as is every pole's place in its face: field maps from the weights
//! (`dual.rs`). The weights that keep every dual edge positive form a convex cone, and the
//! rescue of a mesh whose dual edges cross over is the nearest weights inside it, with a margin:
//! a linear program for the widest margin, then a quadratic one for the nearest weights
//! (`program.rs`, a log barrier on gax's sparse solves, where numga calls scipy).
//!
//! The animation shows, left, the points breathing in and out along their rays on the sphere's
//! mesh, held, with the poles of its faces; right, a mesh from points at uneven distances, its
//! points put back on the sphere, where some dual edges cross over, carried along the straight
//! line to the rescue. Each in space (the hull blue, its dual orange, each point coloured by its
//! distance from the sphere) and on the sphere.

mod dual;
mod program;
#[path = "../../shared/surfaces_hull.rs"]
mod surfaces_hull;

use dual::{dual_lengths, poles, sides};
use gax::field::Field;
use gax_numga_examples::label_under;
use gax_numga_examples::mesh::{Mesh, Vector3};
use gax_numga_examples::plot::panel_row;
use gax_numga_examples::scene3::panel3;
use gax_numga_examples::{Anim, Canvas, Light, Marker, Point3, Rect, Scene3, backdrop, caption};
use gax_numga_examples::{colormap, palette, run};
use std::sync::OnceLock;
use surfaces_hull::{beside, camera, drawn, facing, ghost, near_side};

mod scenes {
    use super::dual::{fold_map, inside_map, weighted};
    use super::program::{Numbers, collect, nearest, unit_rows, widest};
    use super::surfaces_hull::{POINTS, directions, hull, uneven_faces};
    use gax::field::{Field, SparseMap};
    use gax_numga_examples::mesh::{Mesh, Scalar3, Vector3};
    use gax_numga_examples::rng::{Draw, rng};
    use gax_numga_examples::signal::wave;

    /// How far the points breathe about the sphere, and the seed of their phases.
    pub const AMPLITUDE: f64 = 0.06;
    pub const PHASE_SEED: u64 = 3;
    /// The share of the widest margin of its conditions the rescue keeps.
    pub const MARGIN_SHARE: f64 = 0.3;

    /// The breath: the mesh of the directions' hull on the sphere, held throughout, and a phase
    /// for each point.
    pub struct Breath {
        pub mesh: Mesh,
        pub phases: Vec<f64>,
    }

    impl Breath {
        pub fn new() -> Breath {
            let directions = directions(POINTS);
            let faces = hull(&directions);
            let mut draws = rng(PHASE_SEED);
            let phases = (0..POINTS)
                .map(|_| draws.range(0.0, core::f64::consts::TAU))
                .collect();
            Breath {
                mesh: Mesh::new(directions, faces),
                phases,
            }
        }

        /// The points at a phase of the breath, each at its own distance along its direction.
        pub fn points(&self, phase: f64) -> Field<Vector3> {
            let radius = |v: usize| 1.0 + AMPLITUDE * wave(phase + self.phases[v]);
            let directions = &self.mesh.vertices;
            Field::from_fn(directions.len(), |v| directions.site(v) * radius(v))
        }
    }

    /// Every fold outward and every pole inside its face, as conditions on the weights, each
    /// scaled to unit size: `A w > 0`.
    pub fn conditions(mesh: &Mesh) -> SparseMap<Scalar3> {
        let folds = fold_map(&mesh.vertices, mesh);
        let insides = inside_map(&mesh.vertices, &mesh.faces);
        let below = folds.shape().0;
        let shifted = insides.couplings().map(|(r, c, cell)| (below + r, c, cell));
        let rows = below + insides.shape().0;
        unit_rows(&collect(
            folds.couplings().chain(shifted).collect(),
            (rows, POINTS),
        ))
    }

    /// The rescue: the mesh of the hull of points at uneven distances, its vertices the
    /// directions, held; the even weights it starts from, with the points on the sphere, and
    /// the nearest weights that unfold it with every pole inside its face.
    pub struct Rescue {
        pub mesh: Mesh,
        pub start: Numbers,
        pub rescue: Numbers,
    }

    impl Rescue {
        pub fn new() -> Rescue {
            let directions = directions(POINTS);
            let mesh = Mesh::new(directions.clone(), uneven_faces(&directions));
            let conditions = conditions(&mesh);
            let start = Field::from_fn(POINTS, |_| Scalar3::new(1.0));
            let (inside, margin) = widest(&conditions);
            let rescue = nearest(&conditions, MARGIN_SHARE * margin, &start, inside);
            Rescue {
                mesh,
                start,
                rescue,
            }
        }

        /// The points at a share of the way through the rescue, along the straight line of
        /// the weights.
        pub fn points(&self, share: f64) -> Field<Vector3> {
            let change = (&self.rescue - &self.start).scaled(timing(share));
            weighted(&self.mesh.vertices, &(&self.start + &change))
        }
    }

    /// How far along the way each share of the time is: at a steady pace for three quarters,
    /// then closing in on the end over the last quarter, the way left shrinking by so many
    /// e-folds, the pace continuous where it starts closing in.
    pub fn timing(share: f64) -> f64 {
        const CLOSING: f64 = 0.75;
        const FOLDS: f64 = 3.0;
        let rate = FOLDS / (1.0 - CLOSING);
        // The way left where closing in starts.
        let left = 1.0 / (1.0 + CLOSING * rate);
        if share < CLOSING {
            share * left * rate
        } else {
            1.0 - left * (-(share - CLOSING) * rate).exp()
        }
    }
}

use scenes::*;

/// Both scenes, computed once.
struct Shown {
    breath: Breath,
    rescue: Rescue,
}

fn shown() -> &'static Shown {
    static SHOWN: OnceLock<Shown> = OnceLock::new();
    SHOWN.get_or_init(|| Shown {
        breath: Breath::new(),
        rescue: Rescue::new(),
    })
}

/// One breath, and the way to the rescue: numga's 48 frames of 80 ms.
const SECONDS: f32 = 3.84;
/// The samples along each great arc, how far each ray is drawn either side of the sphere, and
/// the distance from the sphere at either end of the points' colours.
const ARC_SAMPLES: usize = 16;
const RAY_SPAN: f64 = 0.15;
const DISTANCE_RANGE: f64 = 0.1;
/// The opacity of the hull's faces; the widths of its edges, of its dual's, of the rays and of
/// the arcs on the sphere; and the size of the points.
const FACE_OPACITY: f32 = 0.15;
const EDGE_WIDTH: f32 = 1.0;
const DUAL_WIDTH: f32 = 1.2;
const RAY_WIDTH: f32 = 0.8;
const ARC_WIDTH: f32 = 1.1;
const POINT_SIZE: f32 = 4.0;

fn primal() -> Light {
    palette::sky()
}
fn dual_light() -> Light {
    palette::orange()
}

/// In space: the hull's faces that face the viewer, its polar dual's edges in front (where
/// either of the two dual faces, the polar planes of an edge's ends, faces the viewer), and each
/// point in front on a stretch of its ray, coloured by its distance from the sphere, blue inside
/// and red outside.
fn space(c: &mut Canvas, rect: Rect, mesh: &Mesh, points: &Field<Vector3>) {
    let cam = camera(rect);
    ghost(c, &cam, rect);
    let toward = facing(&cam);
    let poles = poles(points, &mesh.faces);
    let in_front = |v: Vector3| (v | toward).s() > 0.0;
    panel3(c, rect, cam, |s: &mut Scene3| {
        for &[a, b, d] in &mesh.faces {
            let [a, b, d] = [a, b, d].map(|v| points.site(v));
            if ((b - a) ^ (d - a) ^ toward).e123() > 0.0 {
                s.tri(drawn(a), drawn(b), drawn(d), primal(), FACE_OPACITY);
                s.polyline(&[a, b, d, a].map(drawn), EDGE_WIDTH, primal());
            }
        }
        for (&[a, b], [f, g]) in mesh.edges.iter().zip(beside(mesh)) {
            if in_front(points.site(a)) || in_front(points.site(b)) {
                let [from, to] = [f, g].map(|f| drawn(poles.site(f)));
                s.seg(from, to, DUAL_WIDTH, dual_light());
            }
        }
        for point in points.iter().filter(|&p| in_front(p)) {
            let (distance, ray) = (point.norm(), point.normalized().into_inner());
            let ends = [1.0 - RAY_SPAN, 1.0 + RAY_SPAN].map(|r| drawn(ray * r));
            s.seg(ends[0], ends[1], RAY_WIDTH, palette::ink().faded(0.4));
            let shade = 0.5 + 0.5 * ((distance - 1.0) / DISTANCE_RANGE).clamp(-1.0, 1.0);
            let light = colormap::coolwarm(shade as f32);
            s.dot(drawn(point), Marker::Dot, POINT_SIZE, light);
        }
    });
}

/// A great arc on the unit sphere between the directions of two points, through the normalized
/// points of the chord between them; points behind the sphere, seen from the viewer, break it.
fn arc(a: Vector3, b: Vector3, toward: Vector3) -> Vec<Point3> {
    let at = |k: usize| {
        let share = k as f64 / (ARC_SAMPLES - 1) as f64;
        let p = (a * (1.0 - share) + b * share).normalized().into_inner();
        near_side(p, toward)
    };
    (0..ARC_SAMPLES).map(at).collect()
}

/// On the sphere: the half facing the viewer of the hull's edges and of its dual's, projected
/// from the centre.
fn on_sphere(c: &mut Canvas, rect: Rect, mesh: &Mesh, points: &Field<Vector3>) {
    let cam = camera(rect);
    ghost(c, &cam, rect);
    let toward = facing(&cam);
    let poles = poles(points, &mesh.faces);
    panel3(c, rect, cam, |s: &mut Scene3| {
        for (&[a, b], [f, g]) in mesh.edges.iter().zip(beside(mesh)) {
            let edge = arc(points.site(a), points.site(b), toward);
            s.polyline(&edge, ARC_WIDTH, primal());
            let dual_edge = arc(poles.site(f), poles.site(g), toward);
            s.polyline(&dual_edge, ARC_WIDTH, dual_light());
        }
    });
}

/// The number of dual edges crossed over, and of poles outside their faces.
fn crossed(points: &Field<Vector3>, mesh: &Mesh) -> (usize, usize) {
    let edges = dual_lengths(points, mesh)
        .iter()
        .filter(|l| l.s() < 0.0)
        .count();
    let sides = sides(points, &mesh.faces, &poles(points, &mesh.faces));
    let outside =
        (0..mesh.faces.len()).filter(|&f| sides.iter().any(|side| side.site(f).s() <= 0.0));
    (edges, outside.count())
}

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let s = shown();
    let share = f64::from((t / SECONDS).rem_euclid(1.0));
    let [breath_space, breath_sphere, rescue_space, rescue_sphere] = panel_row(c, 4.0);

    let breathing = s.breath.points(core::f64::consts::TAU * share);
    space(c, breath_space, &s.breath.mesh, &breathing);
    on_sphere(c, breath_sphere, &s.breath.mesh, &breathing);
    label_under(c, breath_space, "BREATHING: HULL AND POLAR DUAL", 30.0, 0.9);
    label_under(c, breath_sphere, "BREATHING: ON THE SPHERE", 30.0, 0.9);

    let rescuing = s.rescue.points(share);
    space(c, rescue_space, &s.rescue.mesh, &rescuing);
    on_sphere(c, rescue_sphere, &s.rescue.mesh, &rescuing);
    let (edges, outside) = crossed(&rescuing, &s.rescue.mesh);
    let label = format!("RESCUE: {edges} CROSSED, {outside} POLES OUT");
    label_under(c, rescue_space, &label, 30.0, 0.9);
    label_under(c, rescue_sphere, "RESCUE: ON THE SPHERE", 30.0, 0.9);
    caption(
        c,
        "WEIGHTED DUALS: A HULL, ITS POLES, AND THE SPHERE",
        "POLES: FACE BIVECTORS OVER THEIR TRIVECTOR (VGA3D); FOLDS LINEAR IN THE WEIGHTS, RESCUED BY CONVEX PROGRAMS",
    );
}

fn main() {
    run(Anim::new("weighted dual", SECONDS).size(960, 540), draw);
}

#[cfg(test)]
mod tests;
