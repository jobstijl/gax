//! numga's `geometry/qem`: Garland and Heckbert's quadric error metrics in PGA3D. A face's
//! supporting plane `P` measures a point's signed distance as `P & X`, so the open dyad
//! `P (P & Point)` is a map from points to planes whose pairing with the point is the squared
//! distance. Summing the dyads of a vertex's faces gives its error quadric, a polarity
//! `Plane <- Point`; contracting an edge adds the quadrics of its two ends. The best position
//! for the merged vertex has no spatial gradient, so its polar plane is the plane at infinity:
//! one least-squares solve, `q.lstsq(e0)`. The animation turns the scene and contracts the
//! ridge edge of a small patch into that vertex, the two faces flanking the edge shrinking away,
//! with each quadric's error ellipsoid drawn around its vertex.

use gax::pga3d::{Motor, Plane, Point};
use gax_numga_examples::{
    Align, Anim, Camera, Canvas, Lens, Light, Marker, Point2, Scene3, backdrop, caption, palette,
    run,
};

mod qem {
    use super::*;

    pub type P = Point<(), f64>;
    pub type Pl = Plane<(), f64>;
    /// A quadric: a polarity, each point to its polar plane.
    pub type Quadric = Plane<(Point,), f64>;

    /// A surface patch with a sharp ridge ending in an apex: six vertices and six triangles.
    pub fn ridge_patch() -> ([P; 6], [[usize; 3]; 6]) {
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
    pub fn face_planes(vertices: &[P; 6], faces: &[[usize; 3]; 6]) -> [Pl; 6] {
        faces.map(|[i, j, k]| {
            (vertices[i] & vertices[j] & vertices[k])
                .normalized()
                .into_inner()
        })
    }

    /// The dyad of a plane: `P (P & Point)`, its squared distance to a point as a quadric.
    pub fn dyad(plane: Pl) -> Quadric {
        plane * (plane & Point::slot())
    }

    /// The quadric of a set of planes: the sum of their dyads.
    pub fn quadric(planes: &[Pl]) -> Quadric {
        planes.iter().fold(Plane::zero(), |q, p| q + dyad(*p))
    }

    /// A quadric's error at a point: the point paired with its polar plane.
    pub fn error(q: Quadric, x: P) -> f64 {
        (q.of(x) & x).s()
    }

    /// Contract an edge `(a, b)`, given the planes of the faces at each end (the faces flanking
    /// the edge in both sets): the two quadrics, their sum, and the merged vertex, whose polar
    /// plane is the plane at infinity.
    pub fn edge_collapse(planes_a: &[Pl], planes_b: &[Pl]) -> (Quadric, Quadric, Quadric, P) {
        let (qa, qb) = (quadric(planes_a), quadric(planes_b));
        let q_edge = qa + qb;
        let infinity = Plane::new(0.0, 0.0, 0.0, 1.0);
        let v_edge: P = q_edge.lstsq(infinity).unitized();
        (qa, qb, q_edge, v_edge)
    }

    pub const INCIDENT_A: [usize; 4] = [0, 1, 2, 3];
    pub const INCIDENT_B: [usize; 4] = [0, 1, 4, 5];

    /// The scene: the patch, its faces' planes, and the collapse of the ridge edge.
    pub struct Collapse {
        pub vertices: [P; 6],
        pub faces: [[usize; 3]; 6],
        pub qa: Quadric,
        pub qb: Quadric,
        pub q_edge: Quadric,
        pub v_edge: P,
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
    pub fn ellipsoid_radius(q: Quadric, dir: P, epsilon: f64, max_radius: f64) -> f64 {
        // On a direction (an ideal point) the quadric reads its quadratic part alone, the
        // error's growth along that direction.
        let growth = (q.of(dir) & dir).s();
        epsilon / growth.max((epsilon / max_radius).powi(2)).sqrt()
    }
}

use qem::*;

/// The point `t` of the way from `a` to `b` (unit points).
fn lerp(a: P, b: P, t: f64) -> P {
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
fn ellipsoid(s: &mut Scene3, q: Quadric, centre: P, colour: Light) {
    let (n_theta, n_phi) = (17, 25);
    let pt = |i: usize, j: usize| {
        let theta = core::f64::consts::PI * i as f64 / (n_theta - 1) as f64;
        let phi = core::f64::consts::TAU * j as f64 / (n_phi - 1) as f64;
        let turn =
            Motor::rotation_about(0.0, 0.0, 1.0, phi) * Motor::rotation_about(0.0, 1.0, 0.0, theta);
        let d = turn >> Point::direction(0.0, 0.0, 1.0);
        centre + d.gp(ellipsoid_radius(q, d, 0.12, 0.4))
    };
    for i in 0..n_theta {
        let ring: Vec<P> = (0..n_phi).map(|j| pt(i, j)).collect();
        s.polyline(&ring, 1.0, colour.faded(0.45));
    }
    for j in 0..n_phi {
        let meridian: Vec<P> = (0..n_theta).map(|i| pt(i, j)).collect();
        s.polyline(&meridian, 1.0, colour.faded(0.45));
    }
}

fn camera(c: &Canvas, azimuth: f32) -> Camera {
    Camera::orbit(
        c.width,
        c.height,
        Point::xyz(0.0, 0.0, 0.0),
        2.9,
        azimuth,
        0.42,
        Lens::Perspective(0.55),
    )
}

/// Triangles covering what is behind by `opacity`, with their edges as one faint stroke each.
fn mesh(s: &mut Scene3, pts: &[P], faces: &[(usize, [usize; 3])], opacity: f32) {
    for &(f, [i, j, k]) in faces {
        let (a, b, c) = (pts[i], pts[j], pts[k]);
        let col = s.lit(a, b, c, face_colour(f));
        s.tri(a, b, c, col, opacity);
        s.polyline(&[a, b, c, a], 1.0, palette::grid().faded(0.8));
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
    s.dot(pts[which], Marker::Dot, 8.0, palette::ink());
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
    s.seg(a, b, 1.5, palette::red().faded(0.5));
    s.dot(a, Marker::Dot, 6.0, palette::ink().faded(0.4));
    s.dot(b, Marker::Dot, 6.0, palette::ink().faded(0.4));
    ellipsoid(&mut s, scene.q_edge, target, palette::green());
    s.dot(target, Marker::Star, 14.0, palette::green());
    s.draw(c);
    label(c, "3. CONTRACTION: V MINIMIZES QA + QB");
    let err = |p: P| error(scene.q_edge, p);
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
    c.text(text, at, size.min(fit), palette::ink(), Align::Center);
}

const SECONDS: f32 = 8.0;

fn draw(c: &mut Canvas, t: f32) {
    let phase = t / SECONDS * core::f32::consts::TAU;
    let azimuth = 48f32.to_radians() + 0.5 * phase.sin();
    // Contract, hold, and open again.
    let progress = f64::from((1.5 * (0.5 - 0.5 * phase.cos())).min(1.0));
    let scene = collapse();
    let w = c.width / 3;
    for i in 0..3 {
        let mut sub = Canvas::new(w, c.height);
        match i {
            0 => vertex_panel(&mut sub, &scene, 0, azimuth, "1. VERTEX A: A SHARP CORNER"),
            1 => vertex_panel(&mut sub, &scene, 1, azimuth, "2. VERTEX B: A CREASE"),
            _ => collapse_panel(&mut sub, &scene, progress, azimuth),
        }
        c.blit(&sub, i * w, 0);
    }
    caption(
        c,
        "QUADRIC ERROR METRICS: EACH VERTEX A SUM OF PLANE DYADS",
        "THE MERGED VERTEX SOLVES Q(V) = E0 (PGA3D)",
    );
}

fn main() {
    run(Anim::new("qem", SECONDS).size(960, 400), draw);
}

#[cfg(test)]
mod tests {
    use super::qem::*;
    use gax::pga3d::{Plane, Point};

    /// A plane's dyad evaluates to the squared perpendicular distance.
    #[test]
    fn plane_dyad_evaluates_to_squared_perpendicular_distance() {
        // The plane y = 0.5.
        let q = dyad(Plane::new(0.0, 1.0, 0.0, -0.5));
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

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws(super::draw, 0.5);
    }
}
