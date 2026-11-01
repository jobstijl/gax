//! numga's `surfaces/qem`: Garland and Heckbert's quadric error metrics in PGA3D. A face's
//! supporting plane `P` measures a point's signed distance as `P & X`, so the open dyad
//! `P (P & Point)` is a map from points to planes whose pairing with the point is the squared
//! distance. Summing the dyads of a vertex's faces gives its error quadric, a polarity
//! `Plane <- Point`; contracting an edge adds the quadrics of its two ends. The best position
//! for the merged vertex has no spatial gradient: only the Euclidean part of its polar plane
//! must vanish, so the correction from the edge's midpoint is a direction, the least-norm
//! solve of the quadric's map from directions to planes through the origin; a flat patch or a
//! crease leaves directions free, and the vertex stays as near the midpoint as they allow.
//!
//! The simplifier repeats this over a closed mesh: each pass takes the cheapest contractions
//! with disjoint neighbourhoods that keep the surface a manifold and every moved face's sense
//! and area, the merged vertices keeping the sum of their initial quadrics. The animation turns
//! the scene and contracts the ridge edge of a small patch into its vertex, the two faces
//! flanking the edge shrinking away, each quadric's error ellipsoid drawn around its vertex;
//! below, the Stanford bunny at numga's face targets.

use gax::pga3d::{Direction, Motor, OriginPlane, Plane, Point, intrinsic};
use gax_numga_examples::measure::roots;
use gax_numga_examples::signal::{haversine, wave};
use gax_numga_examples::{
    Align, Anim, Camera, Canvas, Lens, Light, Marker, ORIGIN3, Point2, Scene3, backdrop, caption,
    palette, run_with,
};

mod qem {
    use super::*;

    /// A face's supporting plane: the join of its corners, oriented by their winding
    /// (intrinsic).
    pub type Pl = intrinsic::Plane;
    /// A quadric: a polarity, each point to its polar plane.
    pub type Quadric = Plane<(Point,)>;

    /// A surface patch with a sharp ridge ending in an apex: six vertices and six triangles.
    pub fn ridge_patch() -> ([Point; 6], [[usize; 3]; 6]) {
        let vertices = [
            Point::xyz(0.35, 0.0, 0.25),   // 0: vertex a, the apex of a sharp corner
            Point::xyz(-0.35, 0.0, 0.25),  // 1: vertex b, where the crease continues
            Point::xyz(0.0, 0.55, -0.15),  // 2: left base vertex
            Point::xyz(0.0, -0.55, -0.15), // 3: right base vertex
            Point::xyz(0.75, 0.0, -0.15),  // 4: front tip, a steep corner at a
            Point::xyz(-0.85, 0.0, -0.15), // 5: back tip, a gentle ramp at b
        ];
        let faces = [
            [0, 1, 2], // 0: left flank, on the edge a-b
            [1, 0, 3], // 1: right flank, on the edge a-b
            [0, 2, 4], // 2: front-left corner, at a only
            [0, 4, 3], // 3: front-right corner, at a only
            [1, 5, 2], // 4: back-left ramp, at b only
            [1, 3, 5], // 5: back-right ramp, at b only
        ];
        (vertices, faces)
    }

    /// The faces' supporting planes: the join of their three vertices, normalized.
    pub fn face_planes(vertices: &[Point; 6], faces: &[[usize; 3]; 6]) -> [Pl; 6] {
        faces.map(|[i, j, k]| (vertices[i] & vertices[j] & vertices[k]).normalized())
    }

    /// The dyad of a plane: `P (P & Point)`, its squared distance to a point as a quadric.
    pub fn dyad(plane: Pl) -> Quadric {
        plane * (plane & Point::slot())
    }

    /// The quadric of a set of planes: the sum of their dyads.
    pub fn quadric(planes: &[Pl]) -> Quadric {
        planes.iter().map(|p| dyad(*p)).sum()
    }

    /// A quadric's error at a point: the point paired with its polar plane.
    pub fn error(q: Quadric, x: Point) -> f64 {
        (q.of(x) & x).s()
    }

    /// Contract an edge `(a, b)`, given the planes of the faces at each end (the faces flanking
    /// the edge in both sets): the two quadrics, their sum, and the merged vertex, whose polar
    /// plane is the plane at infinity.
    pub fn edge_collapse(planes_a: &[Pl], planes_b: &[Pl]) -> (Quadric, Quadric, Quadric, Point) {
        let (qa, qb) = (quadric(planes_a), quadric(planes_b));
        let q_edge = qa + qb;
        let (a, b) = (ridge_patch().0[0], ridge_patch().0[1]);
        let v_edge = collapse_position(q_edge, (a + b).gp(0.5));
        (qa, qb, q_edge, v_edge)
    }

    /// The collapse position: the unit-weight minimum of a quadric, the one nearest the
    /// midpoint where flat directions leave a choice. A direction moves a point without changing
    /// its weight, so only the spatial part of the polar plane must vanish (its constant part is
    /// the residual error): the least-norm correction from the midpoint solves the quadric's
    /// map from directions to planes through the origin against the midpoint's polar.
    pub fn collapse_position(q: Quadric, midpoint: Point) -> Point {
        let gradient: OriginPlane<(Direction,)> = q.of(Direction::slot()).cast::<OriginPlane>();
        let residual: OriginPlane = q.of(midpoint).cast::<OriginPlane>();
        let correction: Direction = gradient.lstsq(-residual);
        midpoint + correction.cast::<Point>()
    }

    /// A closed triangle mesh: its faces, wound outward, and its vertices.
    #[derive(Clone, Debug)]
    pub struct Surface {
        pub faces: Vec<[usize; 3]>,
        pub vertices: Vec<Point>,
    }

    /// The edges of a set of faces, each by its two ends, the lower first, in order.
    pub fn edges(faces: &[[usize; 3]]) -> Vec<[usize; 2]> {
        let mut edges: Vec<[usize; 2]> = faces
            .iter()
            .flat_map(|f| {
                [0, 1, 2].map(|i| {
                    let (a, b) = (f[i], f[(i + 1) % 3]);
                    [a.min(b), a.max(b)]
                })
            })
            .collect();
        edges.sort_unstable();
        edges.dedup();
        edges
    }

    /// The plane of a face, the join of its corners in its winding: not normalized.
    fn face_plane(vertices: &[Point], [a, b, c]: [usize; 3]) -> Pl {
        vertices[a] & vertices[b] & vertices[c]
    }

    /// The lowest-cost legal contractions whose neighbourhoods are disjoint, at most `most` of
    /// them: a contraction keeps the closed surface a manifold where the edge's ends share no
    /// neighbours but the two across the edge (their neighbourhoods, themselves included, share
    /// four vertices).
    fn select_edges(
        edges: &[[usize; 2]],
        neighbourhoods: &[Vec<usize>],
        errors: &[f64],
        legal: &[bool],
        most: usize,
    ) -> Vec<usize> {
        let mut order: Vec<usize> = (0..edges.len()).filter(|e| legal[*e]).collect();
        order.sort_by(|a, b| errors[*a].total_cmp(&errors[*b]));
        let mut locked = vec![false; neighbourhoods.len()];
        let mut chosen = Vec::new();
        for e in order {
            let [first, second] = edges[e];
            if locked[first] || locked[second] {
                continue;
            }
            let shared = neighbourhoods[first]
                .iter()
                .filter(|v| neighbourhoods[second].binary_search(v).is_ok())
                .count();
            if shared != 4 {
                continue;
            }
            chosen.push(e);
            for v in neighbourhoods[first].iter().chain(&neighbourhoods[second]) {
                locked[*v] = true;
            }
            if chosen.len() == most {
                break;
            }
        }
        chosen
    }

    /// Simplify a closed triangle mesh by quadric error metrics: the mesh at each of the
    /// descending face-count targets, after the mesh as given. The initial face planes remain
    /// the reference: every merged vertex keeps the sum of its initial vertices' quadrics. Each
    /// pass contracts separate neighbourhoods, the least errors first, at most half the faces
    /// still to go, and only where every surviving face around the edge keeps its sense and a
    /// nonzero area with the edge contracted.
    pub fn simplify(surface: &Surface, targets: &[usize]) -> Vec<Surface> {
        let mut faces = surface.faces.clone();
        let mut vertices = surface.vertices.clone();
        // Each vertex's quadric sums the dyads of its faces' unit planes.
        let mut quadrics = vec![Quadric::zero(); vertices.len()];
        for f in &faces {
            let d = dyad(face_plane(&vertices, *f).normalized());
            for v in f {
                quadrics[*v] += d;
            }
        }
        let mut snapshots = vec![surface.clone()];
        for &target in targets {
            while faces.len() > target {
                let edges = edges(&faces);
                let mut neighbourhoods: Vec<Vec<usize>> =
                    (0..vertices.len()).map(|v| vec![v]).collect();
                for [a, b] in &edges {
                    neighbourhoods[*a].push(*b);
                    neighbourhoods[*b].push(*a);
                }
                for n in &mut neighbourhoods {
                    n.sort_unstable();
                }
                // The edges at each vertex.
                let mut at_vertex: Vec<Vec<usize>> = vec![Vec::new(); vertices.len()];
                for (e, [a, b]) in edges.iter().enumerate() {
                    at_vertex[*a].push(e);
                    at_vertex[*b].push(e);
                }
                let edge_quadrics: Vec<Quadric> = edges
                    .iter()
                    .map(|[a, b]| quadrics[*a] + quadrics[*b])
                    .collect();
                let optima: Vec<Point> = (edges.iter().zip(&edge_quadrics))
                    .map(|([a, b], q)| collapse_position(*q, (vertices[*a] + vertices[*b]).gp(0.5)))
                    .collect();
                let errors: Vec<f64> = (edge_quadrics.iter().zip(&optima))
                    .map(|(q, x)| error(*q, *x))
                    .collect();
                // Every face at one end of an edge, but not beside it, moves with the edge
                // contracted: its corner at that end goes to the optimum. The side facing the
                // corner, running with the face, joined with the optimum is its new plane.
                let mut legal = vec![true; edges.len()];
                for f in &faces {
                    let old = face_plane(&vertices, *f);
                    for i in 0..3 {
                        let facing = vertices[f[(i + 1) % 3]] & vertices[f[(i + 2) % 3]];
                        for &e in &at_vertex[f[i]] {
                            let [a, b] = edges[e];
                            let other = if a == f[i] { b } else { a };
                            if f.contains(&other) {
                                continue;
                            }
                            let new = facing & optima[e];
                            let keeps_sense = old.scalar_product(new).s() > 0.0;
                            let keeps_area = new.norm_squared() > old.norm_squared() * 1e-12;
                            if !(keeps_sense && keeps_area) {
                                legal[e] = false;
                            }
                        }
                    }
                }
                let most = ((faces.len() - target) / 2).max(1);
                let chosen = select_edges(&edges, &neighbourhoods, &errors, &legal, most);
                assert!(
                    !chosen.is_empty(),
                    "no legal contraction remains at {} faces before {target}",
                    faces.len()
                );
                // The ends of each chosen edge identified, the faces beside it gone, the
                // vertices renumbered in order; the merged vertex at its edge's optimum.
                let mut mapping: Vec<usize> = (0..vertices.len()).collect();
                let mut merged: Vec<Option<usize>> = vec![None; vertices.len()];
                for &e in &chosen {
                    let [first, second] = edges[e];
                    mapping[second] = first;
                    merged[first] = Some(e);
                }
                let surviving: Vec<[usize; 3]> = faces
                    .iter()
                    .map(|f| f.map(|v| mapping[v]))
                    .filter(|[a, b, c]| a != b && b != c && c != a)
                    .collect();
                let mut kept: Vec<usize> = surviving.iter().flatten().copied().collect();
                kept.sort_unstable();
                kept.dedup();
                let mut index = vec![usize::MAX; vertices.len()];
                for (new, old) in kept.iter().enumerate() {
                    index[*old] = new;
                }
                faces = surviving.iter().map(|f| f.map(|v| index[v])).collect();
                (vertices, quadrics) = kept
                    .iter()
                    .map(|v| match merged[*v] {
                        Some(e) => (optima[e], edge_quadrics[e]),
                        None => (vertices[*v], quadrics[*v]),
                    })
                    .unzip();
            }
            snapshots.push(Surface {
                faces: faces.clone(),
                vertices: vertices.clone(),
            });
        }
        snapshots
    }

    /// numga's targets for the bunny, and the bunny's size, the longest side of its box.
    pub const FACE_TARGETS: [usize; 5] = [3500, 1800, 900, 450, 250];
    pub const MESH_SIZE: f64 = 2.0;

    /// The Stanford bunny (as decimated in libigl's tutorial data, built into the binary), its
    /// faces wound outward, centred in a box `MESH_SIZE` across and stood upright: numga's
    /// `(yz * -π/4).exp()`, a quarter turn about x taking y onto z.
    pub fn bunny() -> Surface {
        const BUNNY: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/models/bunny.obj"));
        let model = gax_numga_examples::mesh::Mesh::from_obj(BUNNY, MESH_SIZE);
        let faces = model.closed().faces;
        let upright = Motor::rotation_about(
            Point::direction(1.0, 0.0, 0.0),
            core::f64::consts::FRAC_PI_2,
        );
        let vertices = (model.vertices.iter())
            .map(|v| upright >> Point::xyz(v.c[0], v.c[1], v.c[2]))
            .collect();
        Surface { faces, vertices }
    }

    pub const INCIDENT_A: [usize; 4] = [0, 1, 2, 3];
    pub const INCIDENT_B: [usize; 4] = [0, 1, 4, 5];

    /// The scene: the patch, its faces' planes, and the collapse of the ridge edge.
    pub struct Collapse {
        pub vertices: [Point; 6],
        pub faces: [[usize; 3]; 6],
        pub qa: Quadric,
        pub qb: Quadric,
        pub q_edge: Quadric,
        pub v_edge: Point,
    }

    pub fn collapse() -> Collapse {
        let (vertices, faces) = ridge_patch();
        let planes = face_planes(&vertices, &faces);
        let (qa, qb, q_edge, v_edge) = edge_collapse(
            &INCIDENT_A.map(|i| planes[i]),
            &INCIDENT_B.map(|i| planes[i]),
        );
        Collapse {
            vertices,
            faces,
            qa,
            qb,
            q_edge,
            v_edge,
        }
    }

    /// The radius of a quadric's error ellipsoid `error = epsilon²` along a unit direction,
    /// capped at `max_radius` where the error stays flat.
    pub fn ellipsoid_radius(q: Quadric, dir: Point, epsilon: f64, max_radius: f64) -> f64 {
        // On a direction (an ideal point) the quadric reads its quadratic part alone, the
        // error's growth along that direction; the radius is the root of
        // `growth r² - epsilon² = 0`, half the distance between its two roots `±r`.
        let growth = (q.of(dir) & dir).s().max((epsilon / max_radius).powi(2));
        roots(growth, 0.0, -epsilon * epsilon).map_or(max_radius, |(_, radius)| radius)
    }
}

use qem::*;

/// The point `t` of the way from `a` to `b` (unit points).
fn lerp(a: Point, b: Point, t: f64) -> Point {
    a + (b - a).gp(t)
}

/// The face colours: blue for the flanking faces, red for the corner at a, purple for the ramp
/// at b.
fn face_colour(f: usize) -> Light {
    let hex = [0x0ea5e9, 0x0284c7, 0xf43f5e, 0xe11d48, 0x8b5cf6, 0x7c3aed][f];
    let ch = |s: u32| ((hex >> s) & 0xff) as f32 / 255.0;
    Light::from_srgb(ch(16), ch(8), ch(0), 1.6)
}

/// A quadric's error ellipsoid about `centre`, as a wireframe: along each direction (the pole
/// `z` tilted by `theta` towards `x`, then turned by `phi` about `z`), the radius at which the
/// error reaches its level.
fn ellipsoid(s: &mut Scene3, q: Quadric, centre: Point, colour: Light) {
    let (n_theta, n_phi) = (17, 25);
    let pt = |i: usize, j: usize| {
        let theta = core::f64::consts::PI * i as f64 / (n_theta - 1) as f64;
        let phi = core::f64::consts::TAU * j as f64 / (n_phi - 1) as f64;
        let turn = Motor::rotation_about(Point::direction(0.0, 0.0, 1.0), phi)
            * Motor::rotation_about(Point::direction(0.0, 1.0, 0.0), theta);
        let d = turn >> Point::direction(0.0, 0.0, 1.0);
        centre + d.gp(ellipsoid_radius(q, d, 0.12, 0.4))
    };
    for i in 0..n_theta {
        let ring: Vec<Point> = (0..n_phi).map(|j| pt(i, j)).collect();
        s.polyline(&ring, 1.0, colour.faded(0.45));
    }
    for j in 0..n_phi {
        let meridian: Vec<Point> = (0..n_theta).map(|i| pt(i, j)).collect();
        s.polyline(&meridian, 1.0, colour.faded(0.45));
    }
}

fn camera(c: &Canvas, azimuth: f32) -> Camera {
    Camera::orbit(
        c.rect(),
        ORIGIN3,
        2.9,
        azimuth,
        0.42,
        Lens::Perspective(0.55),
    )
}

/// Triangles covering what is behind by `opacity`, with their edges as one faint stroke each.
fn mesh(s: &mut Scene3, pts: &[Point], faces: &[(usize, [usize; 3])], opacity: f32) {
    for &(f, [i, j, k]) in faces {
        let (a, b, c) = (pts[i], pts[j], pts[k]);
        let col = s.lit(a, b, c, face_colour(f));
        s.tri(a, b, c, col, opacity);
        s.polyline(&[a, b, c, a], 1.0, palette::GRID.faded(0.8));
    }
}

/// A vertex's neighbourhood: its faces, its error ellipsoid, and the vertex.
fn vertex_panel(c: &mut Canvas, scene: &Collapse, which: usize, azimuth: f32, title: &str) {
    backdrop(c);
    let mut s = Scene3::new(camera(c, azimuth));
    let pts = scene.vertices;
    let (incident, q, colour) = if which == 0 {
        (INCIDENT_A, scene.qa, face_colour(2))
    } else {
        (INCIDENT_B, scene.qb, face_colour(4))
    };
    let faces: Vec<_> = incident.iter().map(|&f| (f, scene.faces[f])).collect();
    mesh(&mut s, &pts, &faces, 0.5);
    ellipsoid(&mut s, q, pts[which], colour);
    s.dot(pts[which], Marker::Dot, 8.0, palette::INK);
    s.draw(c);
    label(c, title);
}

/// The contraction, `progress` of the way: a and b slide into the merged vertex, the flanking
/// faces shrinking to nothing.
fn collapse_panel(c: &mut Canvas, scene: &Collapse, progress: f64, azimuth: f32) {
    backdrop(c);
    let mut s = Scene3::new(camera(c, azimuth));
    let target = scene.v_edge;
    let mut pts = scene.vertices;
    let (a, b) = (pts[0], pts[1]);
    pts[0] = lerp(a, target, progress);
    pts[1] = lerp(b, target, progress);
    let faces: Vec<_> = scene.faces.iter().copied().enumerate().collect();
    mesh(&mut s, &pts, &faces, 0.55);
    s.seg(a, b, 1.5, palette::RED.faded(0.5));
    s.dot(a, Marker::Dot, 6.0, palette::INK.faded(0.4));
    s.dot(b, Marker::Dot, 6.0, palette::INK.faded(0.4));
    ellipsoid(&mut s, scene.q_edge, target, palette::GREEN);
    s.dot(target, Marker::Star, 14.0, palette::GREEN);
    s.draw(c);
    label(c, "3. MERGED: V MINIMIZES QA + QB");
    let err = |p: Point| error(scene.q_edge, p);
    let text = format!(
        "ERROR AT A {:.4}  AT B {:.4}  AT V {:.4}",
        err(scene.vertices[0]),
        err(scene.vertices[1]),
        err(scene.v_edge)
    );
    centred(c, &text, 0.93);
}

fn label(c: &mut Canvas, title: &str) {
    centred(c, title, 0.2);
}

/// A line of text centred across the panel at `height` of the way down, small enough to fit.
fn centred(c: &mut Canvas, text: &str, height: f32) {
    let panel = c.rect();
    let size = (panel.height() / 32.0).clamp(8.0, 12.0);
    let fit = panel.width() * 0.94 / gax_numga_examples::font::width(text, 1.0);
    let top_middle = panel.top_middle();
    let at = top_middle + Point2::direction(0.0, height * panel.height());
    c.text(text, at, size.min(fit), palette::INK, Align::Center);
}

const SECONDS: f32 = 8.0;

/// The bunny at numga's face targets, after the bunny as read.
struct Bunnies {
    stages: Vec<Surface>,
}

fn scene() -> Bunnies {
    Bunnies {
        stages: simplify(&bunny(), &FACE_TARGETS),
    }
}

/// A simplified bunny, lit, its faces outlined, turning with the azimuth, its face count below.
fn bunny_panel(c: &mut Canvas, surface: &Surface, azimuth: f32) {
    backdrop(c);
    let camera = Camera::orbit(
        c.rect(),
        Point::xyz(0.0, 0.0, -0.05),
        9.0,
        azimuth,
        0.25,
        Lens::Perspective(0.4),
    );
    let mut s = Scene3::new(camera);
    let pts = &surface.vertices;
    let outline = (60.0 / surface.faces.len() as f32).clamp(0.05, 0.6);
    for [i, j, k] in &surface.faces {
        let (a, b, cc) = (pts[*i], pts[*j], pts[*k]);
        let col = s.lit(a, b, cc, palette::SKY.faded(0.55));
        s.tri(a, b, cc, col, 0.95);
        s.polyline(&[a, b, cc, a], 0.6, palette::GRID.faded(outline));
    }
    s.draw(c);
    centred(c, &format!("{} FACES", surface.faces.len()), 0.9);
}

fn draw(c: &mut Canvas, bunnies: &Bunnies, t: f32) {
    let phase = t / SECONDS * core::f32::consts::TAU;
    let azimuth = 48f32.to_radians() + 0.5 * wave(phase);
    // Contract, hold, and open again: the phasor's reach across, from 1 down to -1 and back.
    let progress = f64::from((1.5 * haversine(phase)).min(1.0));
    let scene = collapse();
    let screen = c.rect();
    let top = screen.part(0.0, 0.1, 1.0, 0.64);
    for i in 0..3 {
        let place = top.column(i, 3);
        let mut sub = c.sub(place);
        match i {
            0 => vertex_panel(&mut sub, &scene, 0, azimuth, "1. VERTEX A: A SHARP CORNER"),
            1 => vertex_panel(&mut sub, &scene, 1, azimuth, "2. VERTEX B: A CREASE"),
            _ => collapse_panel(&mut sub, &scene, progress, azimuth),
        }
        c.blit(&sub, place.lo);
    }
    // The bunny simplified by the same collapses, pass after pass, at numga's targets.
    let bottom = screen.part(0.0, 0.64, 1.0, 1.0);
    let turning = -1.4 + 0.4 * wave(phase);
    for (i, surface) in bunnies.stages.iter().enumerate() {
        let place = bottom.column(i, bunnies.stages.len());
        let mut sub = c.sub(place);
        bunny_panel(&mut sub, surface, turning);
        c.blit(&sub, place.lo);
    }
    caption(
        c,
        "QUADRIC ERROR METRICS: SUMS OF PLANE DYADS",
        "THE MERGED VERTEX: THE LEAST CORRECTION FROM THE MIDPOINT TO A FLAT POLAR (PGA3D)",
    );
}

fn main() {
    run_with(Anim::new("qem", SECONDS).size(960, 540), scene, draw);
}

#[cfg(test)]
mod tests {
    use super::qem::*;
    use gax::pga3d::{Plane, Point};
    use gax::{ApproxEq, Extensor};
    use gax_numga_examples::signal::wave;

    /// A plane's dyad evaluates to the squared perpendicular distance.
    #[test]
    fn plane_dyad_evaluates_to_squared_perpendicular_distance() {
        // The plane y = 0.5, and its dyad.
        let p = Plane::new(0.0, 1.0, 0.0, -0.5);
        let q: Quadric = p * (p & Point::slot());
        assert!((error(q, Point::xyz(1.0, 2.5, -0.5)) - 4.0).abs() < 1e-12);
        assert!((error(q, Point::xyz(0.0, -1.5, 3.0)) - 4.0).abs() < 1e-12);
        assert!(error(q, Point::xyz(1.0, 0.5, -0.5)).abs() < 1e-12);
    }

    /// The edge quadric is the sum of its ends', its optimum stays on the ridge, leans towards
    /// the sharp corner a, and undercuts the error of both ends.
    #[test]
    fn edge_collapse_minimizes_joint_error() {
        let (verts, faces) = ridge_patch();
        let planes = face_planes(&verts, &faces);
        // The flanking planes pass through both ends of the edge.
        for p in &planes[..2] {
            assert!((*p & verts[0]).s().abs() < 1e-12);
            assert!((*p & verts[1]).s().abs() < 1e-12);
        }
        let (qa, qb, q_edge, v_edge) = edge_collapse(
            &INCIDENT_A.map(|i| planes[i]),
            &INCIDENT_B.map(|i| planes[i]),
        );
        let test = Point::xyz(3.0, 4.0, 5.0);
        assert!((error(q_edge, test) - error(qa, test) - error(qb, test)).abs() < 1e-12);
        let [x, y, _] = v_edge.to_euclidean();
        assert!(y.abs() < 1e-12);
        assert!(0.0 < x && x < verts[0].to_euclidean()[0]);
        let e = error(q_edge, v_edge);
        assert!(e < error(q_edge, verts[0]) && e < error(q_edge, verts[1]));
    }

    /// numga's scenario checks: the optimum's polar plane is the plane at infinity (no
    /// Euclidean part), and its joint error undercuts both ends.
    #[test]
    fn the_optimum_s_polar_plane_is_at_infinity() {
        let scene = collapse();
        assert!(scene.q_edge.of(scene.v_edge).norm() < 1e-9);
        for end in &scene.vertices[..2] {
            assert!(error(scene.q_edge, scene.v_edge) < error(scene.q_edge, *end));
        }
        // And it is the minimum: any nearby point has a larger error.
        for [x, y, z] in [[0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, -0.01]] {
            let near = scene.v_edge + Point::direction(x, y, z);
            assert!(error(scene.q_edge, near) > error(scene.q_edge, scene.v_edge));
        }
    }

    /// A collapse against a single plane keeps the point's weight and moves it straight onto
    /// the plane: the least-norm correction leaves the free directions alone.
    #[test]
    fn planar_collapse_preserves_affine_weight_and_unconstrained_position() {
        for height in [0.0, 1.0] {
            let plane = Plane::new(0.0, 0.0, 1.0, -height);
            let q: Quadric = plane * (plane & Point::slot());
            let optimum = collapse_position(q, Point::xyz(2.0, -3.0, height + 1.4));
            assert!((optimum.e123() - 1.0).abs() < 1e-12);
            assert!(optimum.max_abs_diff(&Point::xyz(2.0, -3.0, height)) < 1e-12);
            assert!((plane & optimum).s().abs() < 1e-12);
        }
    }

    /// Against two planes crossing in a crease, the collapse lands on the crease at the
    /// midpoint's place along it.
    #[test]
    fn crease_collapse_preserves_position_along_the_free_direction() {
        let planes = [
            Plane::new(1.0, 0.0, 0.0, -2.0),
            Plane::new(0.0, 0.0, 1.0, -1.0),
        ];
        let q: Quadric = planes.iter().map(|p| *p * (*p & Point::slot())).sum();
        let optimum = collapse_position(q, Point::xyz(4.0, -3.0, 5.0));
        assert!((optimum.e123() - 1.0).abs() < 1e-12);
        assert!(optimum.max_abs_diff(&Point::xyz(2.0, -3.0, 1.0)) < 1e-12);
        for p in planes {
            assert!((p & optimum).s().abs() < 1e-12);
        }
    }

    /// numga's fixture: an icosahedron stretched and perturbed, so that no two edges cost the
    /// same.
    fn icosahedron() -> Surface {
        #[allow(clippy::disallowed_methods)] // the golden ratio
        let golden = 0.5 * (1.0 + 5f64.sqrt());
        let positions = [
            [-1.0, golden, 0.0],
            [1.0, golden, 0.0],
            [-1.0, -golden, 0.0],
            [1.0, -golden, 0.0],
            [0.0, -1.0, golden],
            [0.0, 1.0, golden],
            [0.0, -1.0, -golden],
            [0.0, 1.0, -golden],
            [golden, 0.0, -1.0],
            [golden, 0.0, 1.0],
            [-golden, 0.0, -1.0],
            [-golden, 0.0, 1.0],
        ];
        // numga's `positions *= [1, 1.3, 0.7]; positions += 0.03 sin(arange(36))`.
        let vertices = (positions.iter().enumerate())
            .map(|(i, p)| {
                let c: [f64; 3] = core::array::from_fn(|k| {
                    p[k] * [1.0, 1.3, 0.7][k] + 0.03 * wave((3 * i + k) as f64)
                });
                Point::xyz(c[0], c[1], c[2])
            })
            .collect();
        let faces = vec![
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ];
        Surface { faces, vertices }
    }

    /// numga's repeated collapses: from 20 faces to 18, 16, 14 and 12, each step one
    /// contraction that keeps the surface a closed sphere of unit-weight vertices, every
    /// surviving face's sense, and places the merged vertex at the minimum of all the initial
    /// planes its cluster inherits (each initial face counted once per corner in the cluster),
    /// which rebuilding the quadrics from the current faces would lose.
    #[test]
    fn repeated_collapses_preserve_closed_mesh_and_original_plane_error() {
        let start = icosahedron();
        let initial_planes: Vec<Pl> = (start.faces.iter())
            .map(|[a, b, c]| {
                (start.vertices[*a] & start.vertices[*b] & start.vertices[*c]).normalized()
            })
            .collect();
        let targets = [18, 16, 14, 12];
        let snapshots = simplify(&start, &targets);
        let counts: Vec<usize> = snapshots.iter().map(|s| s.faces.len()).collect();
        assert_eq!(counts, [20, 18, 16, 14, 12]);
        let mut clusters: Vec<Vec<usize>> = (0..start.vertices.len()).map(|v| vec![v]).collect();
        for pair in snapshots.windows(2) {
            let (old, new) = (&pair[0], &pair[1]);
            for v in &new.vertices {
                assert!((v.e123() - 1.0).abs() < 1e-12 && v.is_finite());
            }
            let mut used: Vec<usize> = new.faces.iter().flatten().copied().collect();
            used.sort_unstable();
            used.dedup();
            assert_eq!(used, (0..new.vertices.len()).collect::<Vec<_>>());
            let mut sorted: Vec<[usize; 3]> = (new.faces.iter())
                .map(|f| {
                    let mut f = *f;
                    f.sort_unstable();
                    f
                })
                .collect();
            assert!(sorted.iter().all(|[a, b, c]| a < b && b < c));
            sorted.sort_unstable();
            sorted.dedup();
            assert_eq!(sorted.len(), new.faces.len());
            // Every edge lies between two faces, and the surface is a sphere.
            let edges = edges(&new.faces);
            for [a, b] in &edges {
                let beside = (new.faces.iter())
                    .filter(|f| f.contains(a) && f.contains(b))
                    .count();
                assert_eq!(beside, 2);
            }
            let euler = new.vertices.len() as i64 - edges.len() as i64 + new.faces.len() as i64;
            assert_eq!(euler, 2);
            // Unmoved vertices identify the two merged clusters after the renumbering.
            let same = |p: &Point, q: &Point| p.c == q.c;
            let removed: Vec<usize> = (0..old.vertices.len())
                .filter(|i| !new.vertices.iter().any(|q| same(&old.vertices[*i], q)))
                .collect();
            let added: Vec<usize> = (0..new.vertices.len())
                .filter(|j| !old.vertices.iter().any(|p| same(p, &new.vertices[*j])))
                .collect();
            assert_eq!((removed.len(), added.len()), (2, 1));
            let old_to_new: Vec<usize> = (0..old.vertices.len())
                .map(|i| {
                    (new.vertices.iter().position(|q| same(&old.vertices[i], q)))
                        .unwrap_or(added[0])
                })
                .collect();
            let mut merged_cluster =
                [clusters[removed[0]].clone(), clusters[removed[1]].clone()].concat();
            merged_cluster.sort_unstable();
            let mut next = vec![Vec::new(); new.vertices.len()];
            for (i, j) in old_to_new.iter().enumerate() {
                next[*j] = clusters[i].clone();
            }
            next[added[0]] = merged_cluster.clone();
            // Every surviving face keeps its winding and a nonzero area.
            for f in &old.faces {
                let mapped = f.map(|v| old_to_new[v]);
                let [a, b, c] = mapped;
                if a == b || b == c || c == a {
                    continue;
                }
                let mut key = mapped;
                key.sort_unstable();
                assert!(sorted.binary_search(&key).is_ok());
                let before = old.vertices[f[0]] & old.vertices[f[1]] & old.vertices[f[2]];
                let after = new.vertices[a] & new.vertices[b] & new.vertices[c];
                assert!(before.scalar_product(after).s() > 0.0);
            }
            // The merged vertex minimizes the initial planes its cluster inherits: the error's
            // gradient vanishes, by central differences.
            let multiplicity: Vec<f64> = (start.faces.iter())
                .map(|f| f.iter().filter(|v| merged_cluster.contains(v)).count() as f64)
                .collect();
            let optimum = new.vertices[added[0]];
            let cost = |x: Point| -> f64 {
                (initial_planes.iter().zip(&multiplicity))
                    .map(|(p, m)| m * (*p & x).s().powi(2))
                    .sum()
            };
            let step = 1e-4;
            for d in [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] {
                let along = Point::direction(d[0], d[1], d[2]).gp(step);
                let slope = (cost(optimum + along) - cost(optimum - along)) / (2.0 * step);
                assert!(slope.abs() < 1e-9, "{slope}");
            }
            clusters = next;
        }
    }

    /// numga's bunny, simplified to its targets, stays a closed sphere at each.
    #[test]
    fn the_bunny_simplifies_to_its_targets() {
        let snapshots = super::scene().stages;
        let counts: Vec<usize> = snapshots.iter().map(|s| s.faces.len()).collect();
        assert_eq!(counts[1..], FACE_TARGETS);
        for s in &snapshots {
            let euler =
                s.vertices.len() as i64 - edges(&s.faces).len() as i64 + s.faces.len() as i64;
            assert_eq!(euler, 2);
        }
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws_with(super::draw, &super::scene(), 0.5);
    }
}
