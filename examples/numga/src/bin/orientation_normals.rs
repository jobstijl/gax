//! numga's `orientation/normals`: surface normals, lines and their mirror images, in PGA3D.
//!
//! A surface's normal means one of two things. A chosen side is a plane given as a boundary,
//! extrinsic (ADR-051): oriented by the space around it. A winding is the join of three ordered
//! points, `a & b & c`, an intrinsic plane: oriented by the order of the points within it.
//! The coefficients can agree; the mirror tells them apart. gax's `>>` gives the two
//! orientations opposite signs under a reflection, so a vertical mirror keeps the side above
//! the reflected triangle and reverses the winding, as reflecting the vertices reverses their
//! order. Either way, reflecting the joined plane is joining the reflected points (the guide's
//! "Versors respect every product, the join included").
//!
//! The same holds for lines. A line through two points lying in the mirror is its own mirror
//! image: `mirror >> line` leaves it as it is, while the plain sandwich `m l m⁻¹`, which gives
//! every element of a grade the same sign, reverses it though neither point moved. Built
//! instead as the meet of two planes that are each other's mirror images, the line comes out of
//! the mirror negated: the mirror swaps the planes, and the meet of the swapped pair is minus
//! the meet. A meet is extrinsic, a join intrinsic.
//!
//! The animation shows numga's three figures in turn, each seen from a slowly swinging camera:
//! the side and the winding of a triangle and its mirror image; the line in the mirror, its
//! mirror image and its plain sandwich; and two points and two planes on one line, before and
//! after the mirror.

use gax::pga3d::{Plane, Point};
use gax_numga_examples::scene3::{Scene3, panel3};
use gax_numga_examples::signal::wave;
use gax_numga_examples::{
    Anim, Camera, Canvas, Lens, Light, Marker, Rect, backdrop, caption, label_under, palette, run,
};

mod normals {
    use gax::pga3d::{Line, Plane, Point, intrinsic};

    /// The mirror: the vertical plane `x = 0`.
    pub fn mirror() -> Plane {
        Plane::new(1.0, 0.0, 0.0, 0.0)
    }

    /// numga's triangle, in the horizontal plane, counterclockwise seen from above.
    pub fn triangle() -> [Point; 3] {
        [
            Point::xyz(-2.4, -0.65, 0.0),
            Point::xyz(-0.9, -0.45, 0.0),
            Point::xyz(-1.8, 0.8, 0.0),
        ]
    }

    /// The triangle's chosen side: the plane `z = 0`, positive above.
    pub fn side() -> Plane {
        Plane::new(0.0, 0.0, 1.0, 0.0)
    }

    /// The plane of a triangle oriented by the order of its vertices.
    pub fn winding(t: [Point; 3]) -> intrinsic::Plane {
        (t[0] & t[1] & t[2]).normalized()
    }

    /// A triangle and its two planes, as they are and reflected (by the same `>>`).
    pub struct Normals {
        pub vertices: [[Point; 3]; 2],
        pub sides: [Plane; 2],
        pub windings: [intrinsic::Plane; 2],
    }

    pub fn reflect_normals(triangle: [Point; 3], side: Plane, mirror: Plane) -> Normals {
        let winding = winding(triangle);
        Normals {
            vertices: [triangle, triangle.map(|p| mirror >> p)],
            sides: [side, mirror >> side],
            windings: [winding, mirror >> winding],
        }
    }

    /// numga's two points in the mirror.
    pub fn ends_in_mirror() -> [Point; 2] {
        [Point::xyz(0.0, -1.0, 0.2), Point::xyz(0.0, 1.0, 0.8)]
    }

    /// The line through two points; reflected by `>>`; and by the plain sandwich.
    pub fn reflect_line(ends: [Point; 2], mirror: Plane) -> [intrinsic::Line; 3] {
        let line = ends[0] & ends[1];
        let plain = (mirror * line * mirror.inverse()).cast::<Line>();
        [line, mirror >> line, plain]
    }

    /// numga's crossing: two planes at right angles, each at half a right angle to the mirror,
    /// meeting in the vertical line through the origin, and two points on that line.
    pub fn crossing() -> ([Plane; 2], [Point; 2]) {
        let planes = [
            Plane::new(1.0, 1.0, 0.0, 0.0),
            Plane::new(-1.0, 1.0, 0.0, 0.0),
        ];
        let ends = [Point::xyz(0.0, 0.0, -0.5), Point::xyz(0.0, 0.0, 0.5)];
        (planes, ends)
    }

    /// Two planes and two points on their line, before and after the mirror, with the meet of
    /// the planes and the join of the points built again from each pair.
    pub struct Crossing {
        pub planes: [[Plane; 2]; 2],
        pub ends: [[Point; 2]; 2],
        pub meets: [Line; 2],
        pub joins: [intrinsic::Line; 2],
    }

    pub fn reflect_crossing(planes: [Plane; 2], ends: [Point; 2], mirror: Plane) -> Crossing {
        let planes = [planes, planes.map(|p| mirror >> p)];
        let ends = [ends, ends.map(|p| mirror >> p)];
        Crossing {
            planes,
            ends,
            meets: planes.map(|[a, b]| a ^ b),
            joins: ends.map(|[p, q]| p & q),
        }
    }
}

use normals::*;

/// The seconds each figure shows.
const ACT: f32 = 6.0;
/// A figure: its title and its caption.
type Figure = (&'static str, &'static str);
/// The figures, in turn.
const ACTS: [Figure; 3] = [
    (
        "SURFACE NORMALS IN A MIRROR",
        "A CHOSEN SIDE IS EXTRINSIC, THE WINDING A & B & C INTRINSIC: THE MIRROR KEEPS ONE AND \
         REVERSES THE OTHER (PGA3D)",
    ),
    (
        "A LINE LYING IN THE MIRROR",
        "ITS OWN MIRROR IMAGE: >> KEEPS THE JOIN P & Q; THE PLAIN SANDWICH REVERSES IT",
    ),
    (
        "A JOIN AND A MEET IN THE MIRROR",
        "THE POINTS STAY, SO THEIR JOIN DOES; THE PLANES SWAP, SO THEIR MEET IS NEGATED",
    ),
];
const MIRROR: Light = palette::GRID;
const FACE: Light = palette::INK;
const SIDE: Light = palette::SKY;
const WINDING: Light = palette::ORANGE;
const LINE: Light = palette::INK;
/// The two points and the two planes of the crossing.
const PAIR: [Light; 2] = [palette::RED, palette::BLUE];

/// A direction at unit length.
fn unit(d: Point) -> Point {
    d.gp(d.ideal_norm().recip())
}

/// The middle of points: their sum, whose weight is their count, at weight one.
fn middle(points: &[Point]) -> Point {
    points
        .iter()
        .fold(Point::new(0.0, 0.0, 0.0, 0.0), |sum, p| sum + *p)
        .unitized()
}

/// A square piece of the mirror `2 half` wide about the foot of `centre`, which is halfway
/// from `centre` to its mirror image; its sides along two directions in the mirror, each the
/// direction of the mirror's meet with a plane through the origin.
fn sheet(mirror: Plane, centre: Point, half: f64) -> [Point; 4] {
    let foot = (centre + mirror.reflect(centre)).unitized();
    // Across: the meet with the plane normal to the axis least like the mirror's normal (the
    // inner product of two planes is that of their normals).
    let axes = [
        Point::direction(1.0, 0.0, 0.0),
        Point::direction(0.0, 1.0, 0.0),
        Point::direction(0.0, 0.0, 1.0),
    ];
    let likeness = |axis: Point| (Plane::orthogonal_to(axis) | mirror).s().abs();
    let reference = axes
        .into_iter()
        .min_by(|a, b| likeness(*a).total_cmp(&likeness(*b)))
        .expect("three axes");
    let across = unit((mirror ^ Plane::orthogonal_to(reference)).direction());
    let up = unit((mirror ^ Plane::orthogonal_to(across)).direction());
    [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]
        .map(|(a, b)| foot + across.gp(a * half) + up.gp(b * half))
}

/// The mirror's sheet, faint, with its edge.
fn draw_sheet(s: &mut Scene3, corners: [Point; 4]) {
    let [a, b, c, d] = corners;
    s.quad(a, b, c, d, MIRROR, 0.5);
    s.polyline(&[a, b, c, d, a], 1.0, MIRROR.faded(3.0));
}

/// A camera on `centre` at numga's elevation, its azimuth swinging slowly about numga's
/// through the act, far enough to show a ball of `radius`.
fn camera(rect: Rect, centre: Point, radius: f64, azimuth: f32, elevation: f32, at: f32) -> Camera {
    let swing = 14f32.to_radians() * wave(core::f32::consts::TAU * at / ACT);
    let distance = radius as f32 / 0.17;
    let (azimuth, elevation) = (azimuth.to_radians() + swing, elevation.to_radians());
    Camera::orbit(
        rect,
        centre,
        distance,
        azimuth,
        elevation,
        Lens::Perspective(0.4),
    )
}

/// numga's first figure: the triangle and its mirror image, with the chosen side's normals on
/// the left and the winding's on the right, where arrows follow the vertices' order.
fn draw_normals(c: &mut Canvas, rects: &[Rect], at: f32) {
    let n = reflect_normals(triangle(), side(), mirror());
    let all: Vec<Point> = n.vertices.iter().flatten().copied().collect();
    let centre = middle(&all);
    let span = 4.8;
    let sheet = sheet(mirror(), centre, span * 0.32);
    let normals = [n.sides.map(|p| p.normal()), n.windings.map(|p| p.normal())];
    let labels = [
        "THE SIDE, A PLANE: KEPT",
        "THE WINDING, A & B & C: REVERSED",
    ];
    for (k, (rect, colour)) in rects.iter().zip([SIDE, WINDING]).enumerate() {
        let cam = camera(*rect, centre, span * 0.5, -68.0, 38.0, at);
        panel3(c, *rect, cam, |s| {
            draw_sheet(s, sheet);
            for (triangle, normal) in n.vertices.iter().zip(normals[k]) {
                let [a, b, d] = *triangle;
                s.tri(a, b, d, FACE.faded(0.35), 0.75);
                s.polyline(&[a, b, d, a], 1.4, FACE.faded(0.6));
                s.arrow(
                    middle(triangle),
                    unit(normal).gp(span * 0.22),
                    3.0,
                    10.0,
                    colour,
                );
                // The order of the vertices, each edge a short arrow along it.
                if k == 1 {
                    for (p, q) in gax_numga_examples::points::sides(triangle) {
                        s.arrow(p + (q - p).gp(0.25), (q - p).gp(0.5), 2.0, 7.0, WINDING);
                    }
                }
            }
        });
        label_under(c, *rect, labels[k], 18.0, 0.9);
    }
}

/// numga's second figure: the two points in the mirror and the line through them, with its
/// direction: as it is, reflected by `>>`, and reflected by the plain sandwich.
fn draw_line_in_mirror(c: &mut Canvas, rects: &[Rect], at: f32) {
    let ends = ends_in_mirror();
    let lines = reflect_line(ends, mirror());
    let centre = middle(&ends);
    let span = ends[0].distance(ends[1]);
    let sheet = sheet(mirror(), centre, span * 0.7);
    let labels = [
        "THE LINE P & Q",
        "MIRROR >> LINE: THE SAME",
        "M * L * M^-1: REVERSED",
    ];
    for ((rect, line), label) in rects.iter().zip(lines).zip(labels) {
        let cam = camera(*rect, centre, span * 1.05, -28.0, 24.0, at);
        let direction = unit(line.direction());
        panel3(c, *rect, cam, |s| {
            draw_sheet(s, sheet);
            s.seg(ends[0], ends[1], 2.0, LINE);
            for p in ends {
                s.dot(p, Marker::Dot, 9.0, LINE);
            }
            let start = centre - direction.gp(span * 0.2);
            s.arrow(start, direction.gp(span * 0.4), 3.0, 11.0, WINDING);
        });
        label_under(c, *rect, label, 18.0, 0.9);
    }
}

/// numga's third figure: two points on a line in the mirror (top) and two planes crossing in
/// it (bottom), each coloured by which one it is, before the mirror (left) and after (right);
/// above them on the line, the direction of the join of the points (top) and of the meet of
/// the planes (bottom).
fn draw_pairs_in_mirror(c: &mut Canvas, rects: &[Rect], at: f32) {
    let (planes, ends) = crossing();
    let x = reflect_crossing(planes, ends, mirror());
    let centre = middle(&x.ends[0]);
    let span = x.ends[0][0].distance(x.ends[0][1]);
    let along = unit(x.ends[0][1] - x.ends[0][0]);
    let sheet = sheet(mirror(), centre, span * 0.95);
    let labels = [
        "TWO POINTS ON THE LINE",
        "MIRRORED: THEY STAY, THE JOIN TOO",
        "TWO PLANES THROUGH IT",
        "MIRRORED: SWAPPED, THE MEET NEGATED",
    ];
    for (i, rect) in rects.iter().enumerate() {
        let (row, panel) = (i / 2, i % 2);
        let cam = camera(*rect, centre, span * 1.4, -62.0, 22.0, at);
        panel3(c, *rect, cam, |s| {
            draw_sheet(s, sheet);
            s.seg(x.ends[panel][0], x.ends[panel][1], 2.5, LINE);
            // The line's direction as built from the pair: the join's on top, the meet's below.
            let built = [x.joins[panel].direction(), x.meets[panel].direction()][row];
            // Above the patches, where nothing hides it.
            let arrow = unit(built).gp(0.4 * span);
            let above = centre + along.gp(0.85 * span);
            s.seg(
                above - along.gp(0.25 * span),
                above + along.gp(0.25 * span),
                1.0,
                LINE.faded(0.4),
            );
            s.arrow(above - arrow.gp(0.5), arrow, 3.0, 10.0, WINDING);
            if row == 0 {
                for (p, colour) in x.ends[panel].iter().zip(PAIR) {
                    s.dot(*p, Marker::Dot, 11.0, colour);
                }
                return;
            }
            // Each plane as a square patch about the line: along it, and across it within the
            // plane (the direction of the plane's meet with the plane normal to the line).
            for (plane, colour) in x.planes[panel].iter().zip(PAIR) {
                let across = unit((*plane ^ Plane::orthogonal_to(along)).direction());
                let corner =
                    |a: f64, b: f64| centre + across.gp(a * span * 0.6) + along.gp(b * span * 0.6);
                let patch = [
                    corner(-1.0, -1.0),
                    corner(1.0, -1.0),
                    corner(1.0, 1.0),
                    corner(-1.0, 1.0),
                ];
                let [a, b, d, e] = patch;
                s.quad(a, b, d, e, colour.faded(0.6), 0.22);
                s.polyline(&[a, b, d, e, a], 1.4, colour);
            }
        });
        label_under(c, *rect, labels[i], 14.0, 0.95);
    }
}

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let screen = c.rect();
    let act = ((t / ACT) as usize).min(ACTS.len() - 1);
    let at = t - act as f32 * ACT;
    match act {
        0 => {
            let rects =
                [0, 1].map(|k| screen.part(0.5 * k as f32, 0.12, 0.5 * (k + 1) as f32, 0.94));
            draw_normals(c, &rects, at);
        }
        1 => {
            let third = 1.0 / 3.0;
            let rects = [0, 1, 2]
                .map(|k| screen.part(third * k as f32, 0.12, third * (k + 1) as f32, 0.94));
            draw_line_in_mirror(c, &rects, at);
        }
        _ => {
            let rects = [0, 1, 2, 3].map(|i| {
                let (row, col) = ((i / 2) as f32, (i % 2) as f32);
                screen.part(
                    0.1 + 0.4 * col,
                    0.12 + 0.43 * row,
                    0.5 + 0.4 * col,
                    0.52 + 0.43 * row,
                )
            });
            draw_pairs_in_mirror(c, &rects, at);
        }
    }
    let (title, sub) = ACTS[act];
    caption(c, title, sub);
}

fn main() {
    run(
        Anim::new("normals", ACT * ACTS.len() as f32).size(960, 540),
        draw,
    );
}

#[cfg(test)]
mod tests {
    use super::normals::*;
    use gax::pga3d::{Line, Plane, Point, intrinsic};
    use gax_numga_examples::testing::{close, near};

    /// numga's first test: both normals start upwards, numga's `+z`, and only the winding
    /// reverses in the vertical mirror. A point above the triangle is on the positive side of
    /// the plane, and its mirror image on the positive side of the reflected plane, while its
    /// pairing with the winding reverses. Reflecting the joined plane is joining the reflected
    /// vertices.
    #[test]
    fn mirrored_normals_keep_their_geometric_meanings() {
        let n = reflect_normals(triangle(), side(), mirror());
        let up = Plane::new(0.0, 0.0, 1.0, 0.0);
        assert!(near(n.sides[0], up, 1e-12));
        assert!(
            near(n.windings[0], up.with_orientation(), 1e-12),
            "{:?}",
            n.windings[0]
        );
        assert!(near(n.sides[1], n.sides[0], 1e-12));
        assert!(near(n.windings[1], -n.windings[0], 1e-12));
        // numga's witness: the triangle's middle, one up.
        let [a, b, c] = n.vertices[0];
        let witness = Point::xyz(-1.7, -0.1, 1.0);
        assert!(near(
            (a + b + c).unitized() + Point::direction(0.0, 0.0, 1.0),
            witness,
            1e-12
        ));
        let image = mirror() >> witness;
        assert!(close((n.sides[0] & witness).s(), 1.0, 1e-12));
        assert!(close((n.sides[1] & image).s(), 1.0, 1e-12));
        assert!(close((n.windings[0] & witness).s(), 1.0, 1e-12));
        assert!(close((n.windings[1] & image).s(), -1.0, 1e-12));
        let [p, q, r] = n.vertices[1];
        assert!(near((p & q & r).normalized(), n.windings[1], 1e-12));
        // numga's reflected vertices: the triangle across the mirror.
        assert!(near(p, Point::xyz(2.4, -0.65, 0.0), 1e-12));
    }

    /// numga's second test: the points lie in the mirror, so they and the line through them are
    /// their own mirror images; the plain sandwich keeps the points and reverses the line.
    /// numga's line, `yz zx xy wx wy wz` = `e23 e31 e12 e01 e02 e03`, is `(0, 2, 0.6, -1, 0, 0)`.
    #[test]
    fn a_line_lying_in_the_mirror_is_its_own_mirror_image() {
        let ends = ends_in_mirror();
        let [line, image, plain] = reflect_line(ends, mirror());
        for p in ends {
            assert!(near(mirror() >> p, p, 1e-12));
        }
        let numga = intrinsic::Line::new(0.0, 2.0, 0.6, -1.0, 0.0, 0.0);
        assert!(near(line, numga, 1e-12), "{line:?}");
        assert!(near(image, line, 1e-12));
        assert!(near(plain, -line, 1e-12));
    }

    /// numga's third test: the mirror swaps the two planes and leaves the points, so the meet of
    /// the planes is negated and the join of the points kept; numga's meet is `2 xy`, its join
    /// `xy`.
    #[test]
    fn planes_crossing_in_the_mirror_swap_while_points_on_their_line_stay() {
        let (planes, ends) = crossing();
        let x = reflect_crossing(planes, ends, mirror());
        assert!(near(x.planes[1][0], x.planes[0][1], 1e-12));
        assert!(near(x.planes[1][1], x.planes[0][0], 1e-12));
        assert!(near(x.ends[1][0], x.ends[0][0], 1e-12) && near(x.ends[1][1], x.ends[0][1], 1e-12));
        assert!(
            near(x.meets[0], Line::new(0.0, 0.0, 2.0, 0.0, 0.0, 0.0), 1e-12),
            "{:?}",
            x.meets[0]
        );
        assert!(near(x.meets[1], -x.meets[0], 1e-12));
        let join = intrinsic::Line::new(0.0, 0.0, 1.0, 0.0, 0.0, 0.0);
        assert!(near(x.joins[0], join, 1e-12) && near(x.joins[1], join, 1e-12));
    }

    #[test]
    fn every_figure_draws() {
        for act in 0..super::ACTS.len() {
            let t = (act as f32 + 0.4) * super::ACT;
            gax_numga_examples::app::assert_draws(super::draw, t);
        }
    }
}
