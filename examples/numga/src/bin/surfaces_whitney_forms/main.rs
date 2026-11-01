//! numga's `surfaces/whitney_forms`: Whitney forms, which interpolate values on the vertices,
//! edges and faces of a mesh into fields across it, on triangles of the plane (VGA2D) and on a
//! tetrahedron of space (VGA3D).
//!
//! A triangle's patch, the join of two of its sides, holds everything: each side over twice the
//! patch is the gradient of the hat of the corner it faces, which falls from one at that corner
//! to nothing on the side; and the step from a corner to a point, over twice the patch, is the
//! Whitney 1-form of the side facing it, which circulates around the corner and reads one along
//! its own side and nothing along the other two. The forms of a mesh's edges, summed with
//! a cochain's values, keep their part along each edge from face to face while the part across
//! it jumps. The derivative of an edge's form is twice the wedge of its ends' gradients, the
//! constant 2-form of the face, so reconstruction commutes with the exterior derivative.
//!
//! Points are antivectors, intrinsic, and gradients vectors (planes), which pair with steps by
//! the join and need no metric. A material is a map from gradients to the currents they drive,
//! and pulled back through the face gradients by their adjugate, the transpose through the join,
//! it is the conductance of the vertices; through the Whitney forms at the three side
//! midpoints, which integrate quadratics over a triangle exactly, it is the conductance of the
//! edges, positive definite even where circumcentric DEC's diagonal weight goes negative on an
//! obtuse pair. One dimension up, a tetrahedron's volume, the join of three edges, does the same
//! for its corner gradients, edge forms and face forms, and pairing each edge's form with each
//! edge's curl through the volume links only opposite edges, by a sixth.
//!
//! The animation shows numga's four figures in turn: the three hats of a triangle and the
//! pyramid of a vertex of a disk, with their gradients; the three edge forms and the two faces'
//! forms of a shared edge; the forms of a tetrahedron, turning; and the steady potential and
//! current between two electrodes on a disk of isotropic and of fibred material. gax's own
//! Whitney maps (`Topology::whitney`, `whitney_d`, `Geometry::whitney_1`, `form_inner` and the
//! Galerkin star) give the same forms, as the tests check.

mod scenes;
#[cfg(test)]
mod tests;
mod whitney;

use gax::field::Field;
use gax::mesh::corners as local_corners;
use gax_numga_examples::measure::roots;
use gax_numga_examples::scene3::{Scene3, panel3};
use gax_numga_examples::shared::simplicial;
use gax_numga_examples::shared::surfaces_plane;
use gax_numga_examples::signal::wave;
use gax_numga_examples::{
    Anim, Axes, Camera, Canvas, Lens, Light, Marker, ORIGIN3, Point2, Point3, Rect, backdrop,
    caption, label_under, palette, run_with,
};
use scenes::*;
use simplicial::{faces, sides as edge_sides};
use surfaces_plane::{Flat, arrow};
use whitney::*;

/// Each figure's time on screen; the loop shows the four.
const FIGURE: f32 = 4.0;
const SECONDS: f32 = 4.0 * FIGURE;
/// The hats' levels, at even steps between nothing and one.
const LEVELS: usize = 7;
/// The lattice of the edge forms' arrows, finer than numga's samples.
const ARROW_SUBDIVISIONS: usize = 11;
/// The tetrahedron's view: numga's elevation and azimuth (radians), the swing of the azimuth
/// over the figure, and the camera's distance.
const TET_ELEVATION: f32 = 0.401;
const TET_AZIMUTH: f32 = 0.663;
const TET_SWING: f32 = 0.6;
const TET_DISTANCE: f32 = 3.5;
/// The rings about the edge across the first edge, and the samples around each.
const RING_RADII: [f64; 3] = [0.20, 0.31, 0.42];
const RING_SAMPLES: usize = 65;

/// The first figure: a triangle with its corner gradients, and a vertex's hat on a small disk.
struct Hats {
    corners: [Point; 3],
    gradients: [Plane; 3],
    triangle: Flat,
    star: Flat,
    /// The disk's faces' gradients of the centre vertex's hat.
    star_gradients: Field<Plane>,
}

/// The second figure: the edge forms on a lattice of the triangle, and the forms of a shared
/// edge on each face of a diamond.
struct Edges {
    samples: Vec<Point>,
    forms: Vec<[Plane; 3]>,
    diamond: Flat,
    /// The samples of each face and the shared edge's form at each.
    shared_forms: Vec<Vec<(Point, Plane)>>,
    shared: [Point; 2],
    apexes: [Point; 2],
}

/// The third figure: the tetrahedron about its centroid, its samples spread from it, and the
/// fields of its first edge and first face there.
struct Tet {
    corners: [Point3; 4],
    samples: Vec<Point3>,
    edge_field: Vec<[f64; 3]>,
    face_field: Vec<[f64; 3]>,
    /// The first edge, the edge across it (each from tail to head), and the corner the first
    /// face form radiates from.
    edge: [usize; 2],
    across: [usize; 2],
    apex: usize,
}

/// The fourth figure: the disk, the potential and the vertex currents of each material.
struct Flow {
    disk: Flat,
    potentials: [Vec<f64>; 2],
    currents: [Vec<[f64; 2]>; 2],
    fibre: [f64; 2],
}

struct Shown {
    hats: Hats,
    edges: Edges,
    tet: Tet,
    flow: Flow,
}

/// A planar mesh to draw on.
fn flat(mesh: &gax::mesh::Topology<3>, vertices: &Field<Point>) -> Flat {
    let points: Vec<[f64; 2]> = vertices.iter().map(coordinates).collect();
    Flat::new(&points, &faces(mesh))
}

fn hats() -> Hats {
    let (corners, _) = single_triangle(TRIANGLE_SUBDIVISIONS);
    let (_, gradients) = barycentric_hats(corners, &corners);
    let triangle = Flat::new(&corners.map(coordinates), &[[0, 1, 2]]);
    let (mesh, vertices) = disk(DISK_RADIUS, STAR_DIVISIONS);
    let ops = Operators::new(&mesh, &vertices);
    let hat = Field::from_fn(vertices.len(), |v| {
        Scalar::new(if v == 0 { 1.0 } else { 0.0 })
    });
    Hats {
        corners,
        gradients,
        triangle,
        star: flat(&mesh, &vertices),
        star_gradients: ops.gradient.of(&hat),
    }
}

fn edges() -> Edges {
    let (corners, _) = single_triangle(TRIANGLE_SUBDIVISIONS);
    let samples = triangle_samples(corners, ARROW_SUBDIVISIONS);
    let forms = edge_whitney_forms(corners, &samples);
    let (mesh, vertices, _) = edge_diamond(ARROW_SUBDIVISIONS);
    let ops = Operators::new(&mesh, &vertices);
    let rim = mesh.exterior(1);
    let shared = (0..mesh.len(1))
        .find(|e| !rim.contains(e))
        .expect("a shared edge");
    let circulation = Field::from_fn(mesh.len(1), |e| {
        Scalar::new(if e == shared { 1.0 } else { 0.0 })
    });
    let shared_forms = (0..2)
        .map(|f| {
            let corners = [0, 1, 2].map(|k| ops.corners[k].site(f));
            triangle_samples(corners, ARROW_SUBDIVISIONS)
                .into_iter()
                .map(|p| {
                    let at = Field::from_fn(2, |_| p);
                    (p, ops.whitney1_at(&at).of(&circulation).site(f))
                })
                .collect()
        })
        .collect();
    let ends = mesh.simplex(1, shared);
    let sides = edge_sides(&mesh);
    Edges {
        samples,
        forms,
        diamond: flat(&mesh, &vertices),
        shared_forms,
        shared: [vertices.site(ends[0]), vertices.site(ends[1])],
        apexes: sides[0].across.map(|v| vertices.site(v)),
    }
}

/// A point of space as drawn.
fn drawn(p: SpacePoint) -> Point3 {
    let [x, y, z] = p.c.map(|c| c as f32);
    Point3::xyz(x, y, z)
}

fn tet() -> Tet {
    let (mesh, vertices, points) = single_tetrahedron(TET_SUBDIVISIONS);
    let tet = tetrahedron(&mesh, &vertices, 0);
    let centroid = tet
        .corners
        .iter()
        .copied()
        .reduce(|a, b| a + b)
        .expect("corners")
        * 0.25;
    let corners = tet.corners.map(|c| drawn(c - centroid));
    let samples = points
        .iter()
        .map(|&p| drawn((p - centroid) * 1.22))
        .collect();
    // A face form times the volume's sense is the direction its flux crosses.
    let sense = tet.volume.s().signum();
    let edge_field = points.iter().map(|&p| tet.edge_forms(p)[0].c).collect();
    let face_field = points
        .iter()
        .map(|&p| (tet.face_forms(p)[0] * sense).c)
        .collect();
    let (first, across) = (&local_corners(3, 1)[0], &complements(3, 1)[0]);
    Tet {
        corners,
        samples,
        edge_field,
        face_field,
        edge: [first[0], first[1]],
        across: [across[0], across[1]],
        apex: 0,
    }
}

fn flow() -> Flow {
    let (mesh, vertices) = disk(DISK_RADIUS, DISK_DIVISIONS);
    let ops = Operators::new(&mesh, &vertices);
    let (electrodes, prescribed) = electrodes(&mesh, &vertices);
    let solved = conductivities().map(|k| ops.conduction(k, &electrodes, &prescribed));
    // Area-weighted vertex currents: a continuous drawing of the facewise solution.
    let mut weight = vec![0.0; vertices.len()];
    for f in 0..mesh.len(2) {
        for &v in mesh.simplex(2, f) {
            weight[v] += ops.areas.site(f).s();
        }
    }
    let currents = [0, 1].map(|case| {
        let mut sum = vec![[0.0; 2]; vertices.len()];
        for f in 0..mesh.len(2) {
            let [x, y] = coordinates(solved[case].1.site(f));
            for &v in mesh.simplex(2, f) {
                let a = ops.areas.site(f).s() / weight[v];
                sum[v][0] += a * x;
                sum[v][1] += a * y;
            }
        }
        sum
    });
    Flow {
        disk: flat(&mesh, &vertices),
        potentials: [0, 1].map(|case| solved[case].0.iter().map(|s| s.s()).collect()),
        currents,
        fibre: coordinates(fibre()),
    }
}

fn shown() -> Shown {
    Shown {
        hats: hats(),
        edges: edges(),
        tet: tet(),
        flow: flow(),
    }
}

/// A point of the plane as drawn.
fn at(p: Point) -> Point2 {
    let [x, y] = coordinates(p);
    Point2::xy(x as f32, y as f32)
}

/// The hats' light: the sky's, brighter as the hat rises.
fn hat_light(h: f64) -> Light {
    let h = h.clamp(0.0, 1.0) as f32;
    palette::SKY.faded(0.03 + 0.55 * h * h)
}

/// A field's level lines at even steps between nothing and one.
fn hat_levels(c: &mut Canvas, ax: &Axes, mesh: &Flat, values: &[f64]) {
    for k in 1..=LEVELS {
        let segs = mesh.level(values, k as f64 / (LEVELS + 1) as f64);
        ax.stroke(c, &segs, 0.7, palette::SKY.faded(0.35));
    }
}

/// The equal axes of a panel around the points.
fn fit(rect: Rect, points: impl IntoIterator<Item = Point2>, margin: f32) -> Axes {
    Axes::fitting(rect, points, margin)
}

/// The panels of a row of four under the caption.
fn row(c: &Canvas) -> [Rect; 4] {
    let screen = c.rect();
    let below = Rect::new(
        0.0,
        screen.height() * 0.2,
        screen.width(),
        screen.height() * 0.84,
    );
    [0, 1, 2, 3].map(|i| below.column(i, 4).inset(6.0, 6.0, 6.0, 6.0))
}

/// The first figure: each corner's hat on the triangle, then the hat of a disk's centre.
fn draw_hats(c: &mut Canvas, h: &Hats) {
    let panels = row(c);
    let points = h.corners.map(at);
    let centroid = (points[0] + points[1] + points[2]).unitized();
    let longest = h.gradients.iter().map(|g| g.norm()).fold(0.0, f64::max);
    for (k, &rect) in panels.iter().take(3).enumerate() {
        let ax = fit(rect, points, 1.22);
        let values: Vec<f64> = (0..3).map(|j| if j == k { 1.0 } else { 0.0 }).collect();
        h.triangle.shade_field(c, &ax, &values, hat_light);
        hat_levels(c, &ax, &h.triangle, &values);
        ax.stroke(c, &h.triangle.wire(), 1.2, palette::INK.faded(0.5));
        ax.line(
            c,
            points[(k + 1) % 3],
            points[(k + 2) % 3],
            2.4,
            palette::INK,
        );
        ax.scatter(c, &points, Marker::Dot, 5.0, palette::INK.faded(0.7));
        ax.scatter(c, &points[k..=k], Marker::Dot, 9.0, palette::ORANGE);
        // The gradient through the centroid, where the hat rises.
        let g = h.gradients[k] * (0.24 / longest);
        let reach = Point2::direction(g.e1() as f32, g.e2() as f32);
        ax.arrow(
            c,
            centroid - reach,
            centroid + reach,
            2.0,
            9.0,
            palette::ORANGE,
        );
        label_under(c, rect, &format!("THE HAT OF CORNER {k}"), 14.0, 0.9);
    }
    let rect = panels[3];
    let ax = fit(rect, h.star.points.iter().copied(), 1.08);
    let values: Vec<f64> = (0..h.star.points.len())
        .map(|v| if v == 0 { 1.0 } else { 0.0 })
        .collect();
    h.star.shade_field(c, &ax, &values, hat_light);
    hat_levels(c, &ax, &h.star, &values);
    ax.stroke(c, &h.star.wire(), 0.6, palette::INK.faded(0.2));
    ax.scatter(c, &h.star.points[..1], Marker::Dot, 9.0, palette::ORANGE);
    let longest = h
        .star_gradients
        .iter()
        .map(|g| g.norm())
        .fold(0.0, f64::max);
    for (f, g) in h.star_gradients.iter().enumerate() {
        if g.norm() > 1e-8 {
            let g = g * (0.16 / longest);
            let reach = Point2::direction(g.e1() as f32, g.e2() as f32);
            let centre = h.star.centroid(f);
            ax.arrow(c, centre - reach, centre + reach, 1.6, 7.0, palette::ORANGE);
        }
    }
    label_under(c, rect, "A VERTEX'S HAT AND ITS GRADIENTS", 14.0, 0.9);
}

/// Arrows of a field of forms at points, the longest `reach` long.
fn arrows(c: &mut Canvas, ax: &Axes, field: &[(Point, Plane)], reach: f64, l: Light) {
    let longest = field.iter().map(|(_, w)| w.norm()).fold(0.0, f64::max);
    for &(p, w) in field {
        let d = w * (reach / longest);
        let tail = at(p) - Point2::direction(0.45 * d.e1() as f32, 0.45 * d.e2() as f32);
        arrow(c, ax, tail, [d.e1(), d.e2()], 1.4, 6.0, l);
    }
}

/// An edge with an arrow about its middle along it.
fn running(c: &mut Canvas, ax: &Axes, from: Point2, to: Point2, l: Light) {
    ax.line(c, from, to, 2.4, l);
    let middle = (from + to).unitized();
    let along = (to - from).gp(0.2);
    ax.arrow(c, middle - along, middle + along, 2.0, 10.0, l);
}

/// The second figure: each edge's form on the triangle, then a shared edge's form on both faces
/// of a diamond.
fn draw_edges(c: &mut Canvas, e: &Edges, h: &Hats) {
    let panels = row(c);
    let points = h.corners.map(at);
    let around = complements(2, 0);
    for (k, &rect) in panels.iter().take(3).enumerate() {
        let ax = fit(rect, points, 1.22);
        ax.fill(c, &points, palette::BLUE, 0.18);
        ax.stroke(c, &h.triangle.wire(), 1.1, palette::INK.faded(0.5));
        let field: Vec<(Point, Plane)> = e
            .samples
            .iter()
            .zip(&e.forms)
            .map(|(&p, w)| (p, w[k]))
            .collect();
        arrows(c, &ax, &field, 0.15, palette::SKY);
        running(
            c,
            &ax,
            points[around[k][0]],
            points[around[k][1]],
            palette::ORANGE,
        );
        ax.scatter(c, &points[k..=k], Marker::Dot, 8.0, palette::SKY);
        label_under(
            c,
            rect,
            &format!("THE FORM OF THE EDGE FACING {k}"),
            14.0,
            0.9,
        );
    }
    let rect = panels[3];
    let ax = fit(rect, e.diamond.points.iter().copied(), 1.08);
    for f in 0..2 {
        let corners = e.diamond.faces[f].map(|v| e.diamond.points[v]);
        ax.fill(c, &corners, palette::BLUE, 0.18);
    }
    ax.stroke(c, &e.diamond.wire(), 1.1, palette::INK.faded(0.5));
    for field in &e.shared_forms {
        arrows(c, &ax, field, 0.12, palette::SKY);
    }
    running(c, &ax, at(e.shared[0]), at(e.shared[1]), palette::ORANGE);
    ax.scatter(c, &e.apexes.map(at), Marker::Dot, 8.0, palette::SKY);
    label_under(
        c,
        rect,
        "ONE EDGE: ALONG IT EQUAL, ACROSS IT NOT",
        14.0,
        0.9,
    );
}

/// The third figure: three views of the tetrahedron, its azimuth swinging with `phase`.
fn draw_tet(c: &mut Canvas, t: &Tet, phase: f32) {
    let screen = c.rect();
    let below = Rect::new(
        0.0,
        screen.height() * 0.14,
        screen.width(),
        screen.height() * 0.93,
    );
    let swing = TET_SWING * wave(core::f32::consts::TAU * phase);
    let cam = |rect: Rect| {
        Camera::orbit(
            rect,
            ORIGIN3,
            TET_DISTANCE,
            TET_AZIMUTH + swing,
            TET_ELEVATION,
            Lens::Perspective(0.5),
        )
    };
    let p = &t.corners;
    let shell = |s: &mut Scene3, tint: Option<(usize, Light)>| {
        for k in 0..4 {
            let [a, b, d] = [1, 2, 3].map(|j| p[(k + j) % 4]);
            let colour = match tint {
                Some((face, l)) if face == k => l,
                _ => palette::BLUE.faded(0.4),
            };
            let lit = s.lit(a, b, d, colour);
            s.tri(a, b, d, lit, 0.22);
        }
        for i in 0..4 {
            for j in i + 1..4 {
                s.seg(p[i], p[j], 1.0, palette::GRID.faded(3.0));
            }
        }
    };
    let rod = |s: &mut Scene3, [a, b]: [usize; 2], l: Light| {
        s.seg(p[a], p[b], 3.0, l);
        let (from, to) = (p[a], p[b]);
        let middle = (from + to).unitized();
        s.arrow(
            middle - (to - from).gp(0.18),
            (to - from).gp(0.4),
            2.2,
            10.0,
            l,
        );
    };
    let glyphs = |s: &mut Scene3, field: &[[f64; 3]], l: Light| {
        let longest = field
            .iter()
            .map(|d| gax::vga3d::Vector::new(d[0], d[1], d[2]).norm())
            .fold(0.0, f64::max);
        for (&q, d) in t.samples.iter().zip(field) {
            let k = (0.30 / longest) as f32;
            let reach = Point3::direction(d[0] as f32 * k, d[1] as f32 * k, d[2] as f32 * k);
            s.arrow(q - reach.gp(0.45), reach, 1.4, 6.0, l);
        }
    };
    let panels = [0, 1, 2].map(|i| below.column(i, 3).inset(6.0, 6.0, 6.0, 6.0));
    panel3(c, panels[0], cam(panels[0]), |s| {
        shell(s, None);
        glyphs(s, &t.edge_field, palette::ORANGE);
        rod(s, t.edge, palette::ORANGE);
        s.seg(p[t.across[0]], p[t.across[1]], 2.5, palette::SKY);
    });
    label_under(
        c,
        panels[0],
        "THE FIRST EDGE'S 1-FORM CIRCULATES",
        16.0,
        0.9,
    );
    panel3(c, panels[1], cam(panels[1]), |s| {
        shell(s, Some((t.apex, palette::ORANGE.faded(0.6))));
        glyphs(s, &t.face_field, palette::SKY);
        s.dot(p[t.apex], Marker::Dot, 10.0, palette::ORANGE);
    });
    label_under(c, panels[1], "THE FIRST FACE'S 2-FORM RADIATES", 16.0, 0.9);
    panel3(c, panels[2], cam(panels[2]), |s| {
        shell(s, None);
        rod(s, t.edge, palette::ORANGE);
        rod(s, t.across, palette::SKY);
        for (radius, l) in RING_RADII
            .iter()
            .zip([palette::YELLOW, palette::ORANGE, palette::GREEN])
        {
            s.polyline(&ring(t, *radius), 2.0, l);
        }
    });
    label_under(
        c,
        panels[2],
        "IT THREADS THE EDGE ACROSS: LINKED BY 1/6",
        16.0,
        0.9,
    );
}

/// A ring about the edge across the first edge, near its middle, square to it, starting
/// towards the first edge.
fn ring(t: &Tet, radius: f64) -> Vec<Point3> {
    use gax::vga3d::Vector;
    let v = |q: Point3| {
        let [x, y, z] = q.to_euclidean().map(f64::from);
        Vector::new(x, y, z)
    };
    let [a, b] = t.across.map(|k| v(t.corners[k]));
    let [e, f] = t.edge.map(|k| v(t.corners[k]));
    let middle = (a + b) * 0.5;
    let axis = (b - a).normalized();
    let towards = (e + f) * 0.5 - middle;
    let radial = (towards - axis * (towards | axis).s()).normalized();
    // The plane square to the axis, which a turn about it turns in.
    let plane = axis * gax::vga3d::Pseudoscalar::new(1.0);
    (0..RING_SAMPLES)
        .map(|k| {
            let angle = core::f64::consts::TAU * k as f64 / (RING_SAMPLES - 1) as f64;
            let turn = (plane * (-angle / 2.0)).exp();
            let q = middle * 0.62 + (turn >> radial) * radius;
            Point3::xyz(q.e1() as f32, q.e2() as f32, q.e3() as f32)
        })
        .collect()
}

/// A potential's light: orange where it is high, sky where low.
fn potential_light(v: f64) -> Light {
    let t = v.clamp(-1.0, 1.0) as f32;
    if t > 0.0 {
        palette::ORANGE.faded(0.04 + 0.42 * t)
    } else {
        palette::SKY.faded(0.04 - 0.42 * t)
    }
}

/// The fourth figure: the potential and current of each material on the disk.
fn draw_flow(c: &mut Canvas, f: &Flow) {
    let screen = c.rect();
    let below = Rect::new(
        0.0,
        screen.height() * 0.14,
        screen.width(),
        screen.height() * 0.93,
    );
    let labels = ["ISOTROPIC", "TEN TIMES AS CONDUCTIVE ALONG THE FIBRE"];
    for (case, label) in labels.iter().enumerate() {
        let rect = below.column(case, 2).inset(10.0, 6.0, 10.0, 6.0);
        let ax = fit(rect, f.disk.points.iter().copied(), 1.1);
        f.disk
            .shade_field(c, &ax, &f.potentials[case], potential_light);
        for k in 1..=LEVELS {
            let level = -0.85 + 1.7 * k as f64 / (LEVELS + 1) as f64;
            ax.stroke(
                c,
                &f.disk.level(&f.potentials[case], level),
                0.9,
                palette::INK.faded(0.4),
            );
        }
        ax.stroke(c, &f.disk.wire(), 0.5, palette::GRID);
        if case == 1 {
            // Dotted chords along the fibre, at even offsets across it.
            let along = Point2::direction(f.fibre[0] as f32, f.fibre[1] as f32);
            let across = Point2::direction(-f.fibre[1] as f32, f.fibre[0] as f32);
            for k in 0..9 {
                let offset = -0.85 + 1.7 * k as f32 / 8.0;
                // The half chord: half the distance between the roots of `t² + offset² − 1`.
                let (_, half) =
                    roots(1.0, 0.0, f64::from(offset * offset) - 1.0).unwrap_or_default();
                let half = half as f32;
                let centre = Point2::xy(0.0, 0.0) + across.gp(offset);
                ax.dashed(
                    c,
                    &[centre - along.gp(half), centre + along.gp(half)],
                    0.8,
                    3.0,
                    palette::INK.faded(0.4),
                );
            }
        }
        // Each panel's arrows to the scale of its own currents (the fibred material carries ten
        // times as much): the median speed a fixed length, the fastest at most 1.6 times it.
        let mut speeds: Vec<f64> = f.currents[case]
            .iter()
            .map(|d| gax::vga2d::Vector::new(d[0], d[1]).norm())
            .collect();
        speeds.sort_by(f64::total_cmp);
        let reference = speeds[speeds.len() / 2];
        for (p, d) in f.disk.points.iter().zip(&f.currents[case]) {
            let speed = gax::vga2d::Vector::new(d[0], d[1]).norm();
            let k = 0.08 / reference * (1.6 * reference / speed.max(1e-12)).min(1.0);
            let d = [d[0] * k, d[1] * k];
            let tail = *p - Point2::direction(0.45 * d[0] as f32, 0.45 * d[1] as f32);
            arrow(c, &ax, tail, d, 1.2, 5.0, palette::INK.faded(0.9));
        }
        ax.stroke(c, &f.disk.rim(), 1.3, palette::INK.faded(0.7));
        label_under(c, rect, label, 22.0, 0.8);
    }
}

fn draw(c: &mut Canvas, s: &Shown, t: f32) {
    backdrop(c);
    let figure = ((t / FIGURE).floor() as usize).min(3);
    let phase = (t - figure as f32 * FIGURE) / FIGURE;
    match figure {
        0 => {
            draw_hats(c, &s.hats);
            caption(
                c,
                "WHITNEY FORMS: THE HATS OF A TRIANGLE'S CORNERS",
                "EACH SIDE OVER TWICE THE PATCH IS A CORNER'S GRADIENT; ON A DISK THE HATS MAKE A VERTEX'S PYRAMID (VGA2D)",
            );
        }
        1 => {
            draw_edges(c, &s.edges, &s.hats);
            caption(
                c,
                "WHITNEY FORMS: THE 1-FORMS OF THE EDGES",
                "THE STEP FROM A CORNER OVER TWICE THE PATCH READS ONE ALONG THE EDGE FACING IT (VGA2D)",
            );
        }
        2 => {
            draw_tet(c, &s.tet, phase);
            caption(
                c,
                "WHITNEY FORMS ON A TETRAHEDRON",
                "THE VOLUME, A JOIN OF THREE EDGES, GIVES THE GRADIENTS, THE EDGE AND FACE FORMS, AND THEIR LINKING (VGA3D)",
            );
        }
        _ => {
            draw_flow(c, &s.flow);
            caption(
                c,
                "WHITNEY FORMS: CONDUCTION BETWEEN TWO ELECTRODES",
                "THE MATERIAL PULLED BACK THROUGH THE FACE GRADIENTS BY THEIR ADJUGATE IS THE CONDUCTANCE (VGA2D)",
            );
        }
    }
}

fn main() {
    run_with(
        Anim::new("whitney forms", SECONDS).size(960, 540),
        shown,
        draw,
    );
}
