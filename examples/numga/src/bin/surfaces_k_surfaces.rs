//! numga's `surfaces/k_surfaces`: discrete surfaces of constant negative Gaussian curvature
//! (K-surfaces) grown from compatible reflections of their unit normals (VGA3D).
//!
//! A K-surface's net is an asymptotic one: along each grid line the tangent plane turns about
//! the line itself, so the edge between two neighbours lies in both their tangent planes. With
//! unit normals `n` and `m` at the two ends, it is the commutator `m × n`, the plane of the two
//! normals, read as a point (a displacement) through the oriented unit volume: each normal gives
//! a map from a neighbouring normal to the edge, `Vector <- Vector`. Around a quadrilateral the
//! four edges close exactly when the missing normal is the opposite one reflected in the plane
//! bisecting its two neighbours (the normals' difference): that is the completion rule, and two
//! crossing curves of normals, great-circle arcs here, grow the whole net from the middle out,
//! one diagonal at a time. Bending the angle between the two arcs bends the saddle while every
//! edge keeps its length.
//!
//! The same rule run along a third direction gives a Bäcklund partner: from one starting normal,
//! a second normal field over the grid, and a second surface whose points are joined to the
//! first's by segments of one length that lie in the tangent planes of both. Two partners
//! completed once more give one surface in either order (Bianchi's permutability). Everything
//! is reflections, commutators and sums; a mirror of the input frame mirrors the saddle and its
//! partner.
//!
//! The animation shows numga's two: the saddle bending (its sheet light on the side its normals
//! point to, dark orange on the other), and a saddle with a Bäcklund partner seen from a
//! swinging view, joined by nine of their tangent segments.

use gax::pga3d::Plane;
use gax_numga_examples::scene3::{Scene3, panel3};
use gax_numga_examples::{
    Align, Anim, Camera, Canvas, Light, Marker, ORIGIN3, Point2, Point3, Rect, backdrop, caption,
    palette, reach3, run_with, signal::haversine,
};

mod k_surfaces {
    use core::f64::consts::PI;
    use gax::vga3d::{Bivector, Pseudoscalar, Vector, intrinsic};

    /// A unit normal: a plane through the origin, the tangent plane's direction.
    pub type Normal = Vector;
    /// A point of the surface: an antivector, the plane of directions dual to it.
    pub type Point = intrinsic::Bivector;
    /// The length scale, oriented by the ordered frame: the volume's metric dual.
    pub type Radius = intrinsic::Scalar;

    /// A grid of values, row by row.
    pub type Grid<T> = Vec<Vec<T>>;

    /// The unit length, oriented by the frame `x y z`: numga's `xyz.dual()`.
    pub fn unit_radius() -> Radius {
        Pseudoscalar::new(1.0).hodge()
    }

    /// The missing normal of a quadrilateral: its opposite normal reflected in the plane
    /// bisecting its two neighbours.
    pub fn complete(base: Normal, left: Normal, right: Normal) -> Normal {
        (left - right).normalized() >> base
    }

    /// The map from a neighbouring normal to the edge between the two points, the line where
    /// their tangent planes cross: the commutator with the normal, scaled by the radius.
    pub fn edge_map(normal: Normal, radius: Radius) -> intrinsic::Bivector<(Vector,)> {
        radius * Vector::slot().commutator(normal)
    }

    /// The edge from the point of normal `from` to that of its neighbour `to`.
    pub fn edge(from: Normal, to: Normal, radius: Radius) -> Point {
        edge_map(from, radius).of(to)
    }

    /// The edges along the columns (from each row to the next) and along the rows.
    pub fn edges(normals: &Grid<Normal>, radius: Radius) -> (Grid<Point>, Grid<Point>) {
        let rows = normals.len();
        let along_u = (0..rows - 1)
            .map(|i| {
                let (now, next) = (&normals[i], &normals[i + 1]);
                now.iter()
                    .zip(next)
                    .map(|(n, m)| edge(*n, *m, radius))
                    .collect()
            })
            .collect();
        let along_v = normals
            .iter()
            .map(|row| row.windows(2).map(|w| edge(w[0], w[1], radius)).collect())
            .collect();
        (along_u, along_v)
    }

    /// The surface's points: sums of the edges between neighbours from the first grid point at
    /// `origin`, down the first column, then along each row.
    pub fn positions(normals: &Grid<Normal>, radius: Radius, origin: Point) -> Grid<Point> {
        let (along_u, along_v) = edges(normals, radius);
        let mut start = origin;
        (0..normals.len())
            .map(|i| {
                if i > 0 {
                    start += along_u[i - 1][0];
                }
                let mut point = start;
                let mut row = vec![point];
                for e in &along_v[i] {
                    point += *e;
                    row.push(point);
                }
                row
            })
            .collect()
    }

    /// Two crossing odd-length normal curves, down the middle column and along the middle row,
    /// extended over their rectangular grid: each diagonal out from the crossing depends only on
    /// the two before it, in all four quadrants.
    pub fn normal_net(mid_u: &[Normal], mid_v: &[Normal]) -> Grid<Normal> {
        let (half_u, half_v) = (mid_u.len() / 2, mid_v.len() / 2);
        let mut normals = vec![vec![mid_u[half_u]; mid_v.len()]; mid_u.len()];
        for (row, n) in normals.iter_mut().zip(mid_u) {
            row[half_v] = *n;
        }
        normals[half_u] = mid_v.to_vec();
        for diagonal in 2..=half_u + half_v {
            for du in diagonal.saturating_sub(half_v).max(1)..=half_u.min(diagonal - 1) {
                let dv = diagonal - du;
                // Steps from the crossing in each quadrant, inside the grid by the ranges.
                let step = |from: usize, by: isize| (from as isize + by) as usize;
                for (su, sv) in [(1, 1), (1, -1), (-1, 1), (-1, -1)] {
                    let row = step(half_u, su * du as isize);
                    let column = step(half_v, sv * dv as isize);
                    let (previous_row, previous_column) = (step(row, -su), step(column, -sv));
                    normals[row][column] = complete(
                        normals[previous_row][previous_column],
                        normals[row][previous_column],
                        normals[previous_row][column],
                    );
                }
            }
        }
        normals
    }

    /// A Bäcklund normal field, propagated from its value at the first grid point down the
    /// first column, then along every row.
    pub fn partner(normals: &Grid<Normal>, start: Normal) -> Grid<Normal> {
        let mut first = vec![start];
        for i in 1..normals.len() {
            first.push(complete(normals[i - 1][0], normals[i][0], first[i - 1]));
        }
        normals
            .iter()
            .zip(first)
            .map(|(row, start)| {
                let mut result = vec![start];
                for j in 1..row.len() {
                    result.push(complete(row[j - 1], row[j], result[j - 1]));
                }
                result
            })
            .collect()
    }

    /// The partner's points: each of the surface's moved along its tangent segment, the edge
    /// between the two normals.
    pub fn partner_points(
        points: &Grid<Point>,
        normals: &Grid<Normal>,
        partner: &Grid<Normal>,
        radius: Radius,
    ) -> Grid<Point> {
        let rows = points.iter().zip(normals).zip(partner);
        rows.map(|((p, n), m)| {
            let each = p.iter().zip(n).zip(m);
            each.map(|((p, n), m)| *p + edge(*n, *m, radius)).collect()
        })
        .collect()
    }

    /// `count` values from `-extent` to `extent`.
    fn linspace(extent: f64, count: usize) -> Vec<f64> {
        gax_numga_examples::signal::linspace(-extent, extent, count)
    }

    /// The two crossing curves of a centred patch: great-circle arcs of normals through `z`,
    /// the first turning about `x`, the second about the plane `yz` turned by the corner angle
    /// about `z`; the second's normals alternate in sign, so one completion rule serves both
    /// directions.
    pub fn crossing_normals(half: usize, extent: f64, corner: f64) -> (Vec<Normal>, Vec<Normal>) {
        let count = 2 * half + 1;
        let reference = Vector::new(0.0, 0.0, 1.0);
        let (xy, yz) = (Bivector::new(0.0, 0.0, 1.0), Bivector::new(1.0, 0.0, 0.0));
        let second_plane = (xy * (-corner / 2.0)).exp() >> yz;
        let angles = linspace(extent, count);
        let mid_u = angles.iter().map(|a| (yz * (-a / 2.0)).exp() >> reference);
        let mid_v = angles.iter().enumerate().map(|(k, a)| {
            let alternating = if (k + half).is_multiple_of(2) {
                1.0
            } else {
                -1.0
            };
            ((second_plane * (-a / 2.0)).exp() >> reference) * alternating
        });
        (mid_u.collect(), mid_v.collect())
    }

    /// A saddle grown from the crossing arcs: its normals and its points, centred on the
    /// crossing.
    pub fn saddle(half: usize, extent: f64, corner: f64) -> (Grid<Normal>, Grid<Point>) {
        let (mid_u, mid_v) = crossing_normals(half, extent, corner);
        let normals = normal_net(&mid_u, &mid_v);
        let points = positions(&normals, unit_radius(), Point::new(0.0, 0.0, 0.0));
        let centre = points[half][half];
        let centred = points
            .iter()
            .map(|r| r.iter().map(|p| *p - centre).collect());
        (normals, centred.collect())
    }

    // --- scenarios

    /// The saddle's half size in vertices and the arcs' extent; the bend about a right corner.
    pub const HALF: usize = 12;
    pub const EXTENT: f64 = 0.8;
    pub const NEUTRAL: f64 = PI / 2.0;
    pub const BEND: f64 = PI / 6.0;
    /// The Bäcklund partner's start: the corner normal turned off the plane through it and
    /// `x`, by a turn of the seed plane about the corner normal and then about the seed plane.
    pub const SEED_AZIMUTH: f64 = 1.6;
    pub const SEED_TURN: f64 = 0.65;

    /// The saddle at a corner angle `phase` of the way between the bend's extremes (0 and 1).
    pub fn bent(phase: f64) -> Grid<Point> {
        saddle(HALF, EXTENT, NEUTRAL + BEND * (2.0 * phase - 1.0)).1
    }

    /// The neutral saddle and its Bäcklund partner: the saddle's normals and points, the
    /// partner's normals and points.
    pub struct Pair {
        #[cfg_attr(not(test), allow(dead_code))] // numga's check of the tangent segments
        pub normals: Grid<Normal>,
        pub points: Grid<Point>,
        #[cfg_attr(not(test), allow(dead_code))]
        pub partner_normals: Grid<Normal>,
        pub partner_points: Grid<Point>,
    }

    pub fn pair() -> Pair {
        let (normals, points) = saddle(HALF, EXTENT, NEUTRAL);
        let corner = normals[0][0];
        let x = Vector::new(1.0, 0.0, 0.0);
        let reference_plane = (corner ^ x).normalized();
        // The plane perpendicular to the corner normal, which turns about it.
        let spin_plane = corner.hodge().with_orientation();
        let seed_plane = (spin_plane * (-SEED_AZIMUTH / 2.0)).exp() >> reference_plane;
        let start = (seed_plane * (-SEED_TURN / 2.0)).exp() >> corner;
        let partner_normals = partner(&normals, start);
        let partner_points = partner_points(&points, &normals, &partner_normals, unit_radius());
        Pair {
            normals,
            points,
            partner_normals,
            partner_points,
        }
    }
}

use k_surfaces::*;

/// The loop: numga's orbit, 180 frames of 40 ms out and back; the saddle bends out and back
/// twice in it (numga's 84 frames).
const SECONDS: f32 = 7.2;
/// The views (elevation, azimuth from `x`) in degrees: the saddle's numga's; the orbit swings
/// 50 degrees each way about -70.
const SADDLE_VIEW: (f32, f32) = (15.0, -15.0);
const ORBIT_ELEVATION: f32 = 26.0;
const ORBIT_AZIMUTH: f32 = -70.0;
const ORBIT_SWING: f32 = 50.0;
/// The rows and columns of the tangent segments drawn.
const CONNECTORS: usize = 3;

/// The data, computed once.
struct Scenes {
    pair: Pair,
}

fn scenes() -> Scenes {
    Scenes { pair: pair() }
}

/// A point drawn: where its dual direction reaches.
fn place(p: Point) -> Point3 {
    reach3(p.hodge())
}

/// A parallel camera at (elevation, azimuth) in degrees, `reach` units from the centre to the
/// panel's top.
fn camera(rect: Rect, (elevation, azimuth): (f32, f32), reach: f32) -> Camera {
    let scale = 0.5 * rect.height() / reach;
    Camera::parallel(rect, scale, azimuth.to_radians(), elevation.to_radians())
}

/// The direction towards the viewer of a parallel camera.
fn towards_viewer(cam: &Camera) -> Point3 {
    cam.eye().unitized() - ORIGIN3
}

/// A sheet: each quad lit, `front` where the viewer sees the side its corners turn
/// counterclockwise from (the grid's own side: the normals alternate in sign along the rows) and
/// `back` where it sees the other, every `spacing`-th grid line drawn in `line`, and the
/// boundary.
fn sheet(
    s: &mut Scene3,
    points: &Grid<Point>,
    (front, back): (Light, Light),
    opacity: f32,
    (spacing, line): (usize, Light),
) {
    let viewer = towards_viewer(&s.cam);
    let at = |i: usize, j: usize| place(points[i][j]);
    let (rows, columns) = (points.len(), points[0].len());
    for i in 0..rows - 1 {
        for j in 0..columns - 1 {
            let (a, b, c, d) = (at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1));
            // The quad's plane, against the plane facing the viewer.
            let facing = ((a & b & c) | Plane::orthogonal_to(viewer)).s();
            let colour = if facing > 0.0 { front } else { back };
            let lit = s.lit(a, b, c, colour);
            s.quad(a, b, c, d, lit, opacity);
        }
    }
    for i in (0..rows).filter(|i| i % spacing == 0 || *i == rows - 1) {
        let row: Vec<Point3> = (0..columns).map(|j| at(i, j)).collect();
        s.polyline(&row, 0.8, line);
    }
    for j in (0..columns).filter(|j| j % spacing == 0 || *j == columns - 1) {
        let column: Vec<Point3> = (0..rows).map(|i| at(i, j)).collect();
        s.polyline(&column, 0.8, line);
    }
}

/// The largest distance of a sheet's points from the origin.
fn reach_of(points: &Grid<Point>) -> f32 {
    let norms = points.iter().flatten().map(|p| p.norm() as f32);
    norms.fold(0.0, f32::max)
}

/// The saddle bending, at `phase` between its extremes.
fn draw_saddle(c: &mut Canvas, rect: Rect, phase: f64) {
    let points = bent(phase);
    // One scale for the whole bend: the reach of its extremes.
    let reach = 0.9 * reach_of(&bent(0.0)).max(reach_of(&bent(1.0)));
    let cam = camera(rect, SADDLE_VIEW, reach);
    let colours = (palette::SKY.faded(0.35), palette::ORANGE.faded(0.3));
    let line = palette::INK.faded(0.3);
    panel3(c, rect, cam, |s| {
        sheet(s, &points, colours, 0.95, (2, line))
    });
}

/// The saddle and its partner from a swinging view: two translucent sheets, and nine of the
/// segments between corresponding points.
fn draw_pair(c: &mut Canvas, rect: Rect, pair: &Pair, swing: f32) {
    let reach = 0.85 * reach_of(&pair.points).max(reach_of(&pair.partner_points));
    let azimuth = ORBIT_AZIMUTH + ORBIT_SWING * swing;
    let cam = camera(rect, (ORBIT_ELEVATION, azimuth), reach);
    let count = pair.points.len();
    let margin = count / 6;
    let picks: Vec<usize> = (0..CONNECTORS)
        .map(|k| margin + k * (count - 2 * margin - 1) / (CONNECTORS - 1))
        .collect();
    panel3(c, rect, cam, |s| {
        let source = (palette::SKY.faded(0.3), palette::SKY.faded(0.3));
        let partner = (palette::ORANGE.faded(0.3), palette::ORANGE.faded(0.3));
        let lines = |l: Light| (1, l.whitened(0.3).faded(0.25));
        sheet(s, &pair.points, source, 0.6, lines(palette::SKY));
        let points = &pair.partner_points;
        sheet(s, points, partner, 0.6, lines(palette::ORANGE));
        for i in &picks {
            for j in &picks {
                let (a, b) = (place(pair.points[*i][*j]), place(points[*i][*j]));
                s.seg(a, b, 1.8, palette::INK);
                s.dot(a, Marker::Dot, 5.0, palette::SKY.whitened(0.4));
                s.dot(b, Marker::Dot, 5.0, palette::ORANGE.whitened(0.4));
            }
        }
    });
}

fn draw(c: &mut Canvas, s: &Scenes, t: f32) {
    backdrop(c);
    let turn = core::f32::consts::TAU * (t / SECONDS).rem_euclid(1.0);
    let screen = c.rect();
    let (w, h) = (screen.width(), screen.height());
    let body = Rect::new(0.0, h * 0.13, w, h);
    let left = body.column(0, 2).inset(8.0, 18.0, 8.0, 8.0);
    let right = body.column(1, 2).inset(8.0, 18.0, 8.0, 8.0);
    // The bend out and back twice a loop, the orbit once.
    draw_saddle(c, left, f64::from(haversine(2.0 * turn)));
    draw_pair(c, right, &s.pair, 1.0 - 2.0 * haversine(turn));
    let label = |c: &mut Canvas, rect: Rect, text: &str| {
        let at = rect.top_middle() - Point2::direction(0.0, 4.0);
        c.text(text, at, 10.0, palette::INK, Align::Center);
    };
    label(c, left, "BENDING: EVERY EDGE KEEPS ITS LENGTH");
    label(c, right, "A BACKLUND PARTNER, JOINED BY TANGENTS");
    caption(
        c,
        "K-SURFACES FROM REFLECTED NORMALS",
        "EDGE = R (M X N); THE MISSING NORMAL IS THE OPPOSITE ONE REFLECTED IN M - N (VGA3D)",
    );
}

fn main() {
    run_with(
        Anim::new("k_surfaces", SECONDS).size(960, 540),
        scenes,
        draw,
    );
}

#[cfg(test)]
mod tests {
    use super::k_surfaces::*;
    use gax::vga3d::{Bivector, Vector};

    const ROUND_OFF: f64 = 1e-10;

    /// The largest coefficient of a list of differences.
    fn largest<const N: usize>(values: impl IntoIterator<Item = [f64; N]>) -> f64 {
        let each = values.into_iter().flatten().map(f64::abs);
        each.fold(0.0, f64::max)
    }

    /// The largest coefficient of the difference of two grids of points.
    fn apart(a: &Grid<Point>, b: &Grid<Point>) -> f64 {
        largest(
            a.iter()
                .flatten()
                .zip(b.iter().flatten())
                .map(|(p, q)| (*p - *q).c),
        )
    }

    /// A net from numga's crossing arcs, at the origin.
    fn net(half: usize, extent: f64, corner: f64, origin: Point) -> (Grid<Normal>, Grid<Point>) {
        let (mid_u, mid_v) = crossing_normals(half, extent, corner);
        let normals = normal_net(&mid_u, &mid_v);
        let points = positions(&normals, unit_radius(), origin);
        (normals, points)
    }

    #[test]
    fn saddle_deformation_preserves_edge_lengths_and_planar_stars() {
        let extent = 0.75;
        let corners = [
            core::f64::consts::PI / 3.0,
            core::f64::consts::PI / 2.0,
            2.0 * core::f64::consts::PI / 3.0,
        ];
        for half in [7, 12] {
            #[allow(clippy::disallowed_methods)] // the reference: the chord of the arcs' step
            let expected_length = (extent / half as f64).sin();
            let mut diagonals = Vec::new();
            for corner in corners {
                let (normals, points) = net(half, extent, corner, Point::new(0.0, 0.0, 0.0));
                let n = normals.len();
                assert!(
                    normals
                        .iter()
                        .flatten()
                        .all(|v| (v.norm_squared() - 1.0).abs() <= ROUND_OFF)
                );
                let (along_u, along_v) = edges(&normals, unit_radius());
                // Around every quadrilateral the edges close.
                for i in 0..n - 1 {
                    for j in 0..n - 1 {
                        let closure =
                            along_u[i][j] + along_v[i + 1][j] - along_u[i][j + 1] - along_v[i][j];
                        assert!(largest([closure.c]) <= ROUND_OFF, "{i} {j}");
                    }
                }
                // The four neighbours of every inner point lie in its tangent plane.
                for i in 1..n - 1 {
                    for j in 1..n - 1 {
                        let centre = points[i][j];
                        let star = [
                            points[i + 1][j],
                            points[i - 1][j],
                            points[i][j + 1],
                            points[i][j - 1],
                        ];
                        for p in star {
                            assert!((normals[i][j] & (p - centre)).s().abs() <= ROUND_OFF);
                        }
                    }
                }
                let lengths = along_u.iter().chain(&along_v).flatten();
                assert!(
                    lengths
                        .clone()
                        .all(|e| (e.norm() - expected_length).abs() <= ROUND_OFF)
                );
                // The points sum the edges.
                for i in 0..n - 1 {
                    for j in 0..n {
                        let error = points[i + 1][j] - points[i][j] - along_u[i][j];
                        assert!(largest([error.c]) <= ROUND_OFF);
                    }
                }
                for i in 0..n {
                    for j in 0..n - 1 {
                        let error = points[i][j + 1] - points[i][j] - along_v[i][j];
                        assert!(largest([error.c]) <= ROUND_OFF);
                    }
                }
                diagonals.push((points[n - 1][n - 1] - points[0][0]).norm());
            }
            let spread = diagonals.iter().copied().fold(f64::MIN, f64::max)
                - diagonals.iter().copied().fold(f64::MAX, f64::min);
            assert!(spread > 0.5, "{spread}");
        }
    }

    #[test]
    fn backlund_connectors_and_bianchi_completion_integrate_to_the_same_surface() {
        let radius = unit_radius();
        let turn_a = (Bivector::new(1.0, 0.0, 0.0) * 0.4).exp();
        // `xy + xz`: `xz` is minus numga's `zx`.
        let turn_b = (Bivector::new(0.0, -1.0, 1.0) * 0.35).exp();
        for corner in [1.15, 1.55] {
            let (normals, points) = net(12, 0.65, corner, Point::new(0.0, 0.0, 0.0));
            let [(normal_a, point_a), (normal_b, point_b)] = [turn_a, turn_b].map(|turn| {
                let partners = partner(&normals, turn >> normals[0][0]);
                let partner_points = partner_points(&points, &normals, &partners, radius);
                let connectors = |i: usize, j: usize| partner_points[i][j] - points[i][j];
                let length = connectors(0, 0).norm();
                for (i, row) in partners.iter().enumerate() {
                    for (j, m) in row.iter().enumerate() {
                        assert!((m.norm_squared() - 1.0).abs() <= ROUND_OFF);
                        assert!((connectors(i, j).norm() - length).abs() <= ROUND_OFF);
                        assert!((normals[i][j] & connectors(i, j)).s().abs() <= ROUND_OFF);
                        assert!((*m & connectors(i, j)).s().abs() <= ROUND_OFF);
                    }
                }
                let independent = positions(&partners, radius, partner_points[0][0]);
                assert!(apart(&independent, &partner_points) <= ROUND_OFF);
                (partners, partner_points)
            });
            // The fourth surface: the two partners completed once more, reached from either.
            let normal_ab: Grid<Normal> = (0..normals.len())
                .map(|i| {
                    let row = 0..normals[i].len();
                    row.map(|j| complete(normals[i][j], normal_a[i][j], normal_b[i][j]))
                        .collect()
                })
                .collect();
            let point_ab_a = partner_points(&point_a, &normal_a, &normal_ab, radius);
            let point_ab_b = partner_points(&point_b, &normal_b, &normal_ab, radius);
            assert!(apart(&point_ab_a, &point_ab_b) <= ROUND_OFF);
            let independent_ab = positions(&normal_ab, radius, point_ab_a[0][0]);
            assert!(apart(&independent_ab, &point_ab_a) <= ROUND_OFF);
            for from in [&normal_a, &normal_b] {
                let propagated = partner(from, normal_ab[0][0]);
                let differences = propagated.iter().flatten().zip(normal_ab.iter().flatten());
                assert!(largest(differences.map(|(p, q)| (*p - *q).c)) <= ROUND_OFF);
            }
        }
    }

    /// Reflecting the input frame reflects the saddle and its partner: the mirror commutes with
    /// every reflection and commutator of the construction, the radius's orientation included.
    #[test]
    fn reflecting_the_input_frame_reflects_the_saddle_and_its_partner() {
        let radius = unit_radius();
        let origin: Point = Vector::new(0.2, -0.1, 0.3).hodge();
        let mirror = Vector::new(2.0, -1.0, 3.0).normalized();
        for corner in [1.2, 1.6] {
            let (mid_u, mid_v) = crossing_normals(12, 0.65, corner);
            let normals = normal_net(&mid_u, &mid_v);
            let points = positions(&normals, radius, origin);
            let start = (Bivector::new(1.0, 0.0, 0.0) * 0.4).exp() >> normals[0][0];
            let partners = partner(&normals, start);
            let moved = partner_points(&points, &normals, &partners, radius);

            let reflect =
                |v: &[Normal]| -> Vec<Normal> { v.iter().map(|n| mirror >> *n).collect() };
            let reflected_normals = normal_net(&reflect(&mid_u), &reflect(&mid_v));
            let (mirrored_radius, mirrored_origin) = (mirror >> radius, mirror >> origin);
            let reflected_points = positions(&reflected_normals, mirrored_radius, mirrored_origin);
            let reflected_partner = partner(&reflected_normals, mirror >> start);
            let reflected_partner_points = partner_points(
                &reflected_points,
                &reflected_normals,
                &reflected_partner,
                mirrored_radius,
            );
            let mirrored = |g: &Grid<Point>| -> Grid<Point> {
                g.iter()
                    .map(|r| r.iter().map(|p| mirror >> *p).collect())
                    .collect()
            };
            let differences = reflected_normals
                .iter()
                .flatten()
                .zip(normals.iter().flatten());
            assert!(largest(differences.map(|(r, n)| (*r - (mirror >> *n)).c)) <= ROUND_OFF);
            assert!(apart(&reflected_points, &mirrored(&points)) <= ROUND_OFF);
            assert!(apart(&reflected_partner_points, &mirrored(&moved)) <= ROUND_OFF);
            // The mirror reverses the radius: the reflected frame's orientation.
            assert!((mirrored_radius.s() + radius.s()).abs() <= ROUND_OFF);
        }
    }

    /// numga's scenario checks: bending keeps every edge's length, and the segments to the
    /// partner all have one length and lie in the tangent planes of both sheets.
    #[test]
    fn the_scenarios_bend_rigidly_and_the_partner_is_tangent() {
        let lengths: Vec<f64> = [bent(0.0), bent(1.0)]
            .iter()
            .flat_map(|p| {
                let rows = p.windows(2).flat_map(|w| w[0].iter().zip(&w[1]));
                rows.map(|(a, b)| (*b - *a).norm()).collect::<Vec<_>>()
            })
            .collect();
        assert!(lengths.iter().all(|l| (l - lengths[0]).abs() <= 1e-10));
        let pair = pair();
        let connectors = |i: usize, j: usize| pair.partner_points[i][j] - pair.points[i][j];
        let length = connectors(0, 0).norm();
        for i in 0..pair.points.len() {
            for j in 0..pair.points[i].len() {
                assert!((connectors(i, j).norm() - length).abs() <= 1e-10);
                assert!((pair.normals[i][j] & connectors(i, j)).s().abs() <= 1e-10);
                assert!((pair.partner_normals[i][j] & connectors(i, j)).s().abs() <= 1e-10);
            }
        }
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws_with(super::draw, &super::scenes(), 0.5);
    }
}
