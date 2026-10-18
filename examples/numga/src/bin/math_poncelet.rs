//! numga's `math/poncelet`: Poncelet's porism in PGA2D. A polygon inscribed in one conic and
//! circumscribed about another either closes from every starting point or from none.
//!
//! Two conics, an outer and an inner one. From a point on the outer conic draw a tangent to the
//! inner one, follow it to where it meets the outer conic again, draw the other tangent from
//! there, and go on. If the path closes after some number of sides for one start, it closes
//! after the same number for every start.
//!
//! The outer conic is a map from points to lines, `Conic = Line <- Point`, zero on its points
//! where `conic(point) & point` vanishes. The inner conic is held by its tangent lines, a map the
//! other way, `Envelope = Point <- Line`, zero on the lines that touch it. Everything about one
//! has a twin about the other with points and lines exchanged, and a step of the path is one
//! such pair of twins: a line meets the outer conic twice, and knowing one crossing the other is
//! rational in the conic; through a point pass two tangents of the inner conic, and knowing one
//! the other is rational in the envelope.
//!
//! Five lines fix a conic they touch, as five points fix a conic through them: with the five
//! sides of a pentagon inscribed in the outer conic, the inner conic closes one pentagon, and by
//! the porism every other one started on the outer conic.
//!
//! The animation: a start runs once around the ellipse; about the conic touching the inscribed
//! pentagon the path of five sides closes from every start, about the conic of the same pentagon
//! shrunk a little towards the centre it misses.

use gax::pga2d::{Line, Motor, Point, extrinsic, intrinsic};
use gax_numga_examples::{
    Align, Anim, Axes, Canvas, Marker, Point2, backdrop, caption, measure, palette, plot, run,
};

mod poncelet {
    use super::*;
    use gax::Unit;

    // Every line here is the join of two points and every point the meet of two lines. A join
    // is oriented along itself and a meet by what turns around it, so the lines are intrinsic
    // and the points extrinsic (ADR-051).
    pub type L = intrinsic::Line<(), f64>;
    pub type P = extrinsic::Point<(), f64>;
    /// The kinds of the slots of the maps below: a line and a point of the construction.
    type LineSlot = intrinsic::Line;
    type PointSlot = extrinsic::Point;
    /// A conic: points to their polar lines.
    pub type Conic = intrinsic::Line<(PointSlot,), f64>;
    /// A conic held by its tangents: lines to their poles.
    pub type Envelope = extrinsic::Point<(LineSlot,), f64>;

    /// The point at `(x, y)`, made as the construction makes its points: the meet of the lines
    /// `x = a` and `y = b`.
    pub fn crossing(x: f64, y: f64) -> P {
        Line::new(1.0, 0.0, -x) ^ Line::new(0.0, 1.0, -y)
    }

    /// The points at infinity along x and along y, where the lines `y = 0` and `x = 0` meet
    /// the line at infinity.
    pub fn headings() -> [P; 2] {
        let w = Line::new(0.0, 0.0, 1.0);
        [Line::new(0.0, 1.0, 0.0) ^ w, w ^ Line::new(1.0, 0.0, 0.0)]
    }

    /// The line at infinity, as the join of the two headings.
    pub fn infinity() -> L {
        let [x, y] = headings();
        x & y
    }

    /// Two points as one envelope: zero on the lines through either, its pairing the product
    /// of the line's pairings with the two points.
    pub fn pair(first: P, second: P) -> Envelope {
        let l = L::slot();
        (first * (second & l) + second * (first & l)).gp(0.5)
    }

    /// The pairing of an envelope with a line, zero on its tangents.
    pub fn touches(inner: Envelope, l: L) -> f64 {
        (inner.of(l) & l).s()
    }

    /// The pairing of a conic with a point, zero on the conic.
    pub fn on(outer: Conic, p: P) -> f64 {
        (outer.of(p) & p).s()
    }

    /// The envelope touching five lines: the combination of the two pairs of opposite corners
    /// of the first four that vanishes on the fifth.
    pub fn envelope(sides: [L; 5]) -> Envelope {
        let [first, second, third, fourth, fifth] = sides;
        let corners = pair(first ^ second, third ^ fourth);
        let diagonals = pair(first ^ third, second ^ fourth);
        corners.gp(touches(diagonals, fifth)) - diagonals.gp(touches(corners, fifth))
    }

    /// The two tangents of an envelope through a point: its zeros along the lines through the
    /// point. Along the combinations `first t + second` of two lines through it the envelope is
    /// the quadratic `a t² + 2 b t + c`, whose roots `measure::roots` gives as their midpoint
    /// and half width; each tangent is weighted by `a`, one root per sign, as
    /// `first (± √(b² - a c) - b) + second a`.
    pub fn tangents(inner: Envelope, point: P) -> [L; 2] {
        let [first, second] = headings().map(|h| point & h);
        let at = |x: L, y: L| (inner.of(x) & y).s();
        let a = at(first, first);
        let (middle, half) = measure::roots(a, at(first, second), at(second, second))
            .expect("a point outside the envelope");
        [1.0, -1.0]
            .map(|sign| (first.gp(a * middle + sign * a.abs() * half) + second.gp(a)).normalized())
    }

    /// Where the line from a point on the conic towards a heading meets the conic again. Along
    /// `vertex + heading t` the pairing is `2 t (outer(vertex) & heading) + t² (outer(heading)
    /// & heading)`, zero at the vertex and at the other root.
    pub fn other_crossing(outer: Conic, vertex: P, heading: P) -> P {
        vertex.gp(on(outer, heading)) - heading.gp(2.0 * (outer.of(vertex) & heading).s())
    }

    /// The other tangent through the point where a tangent meets another line: the twin of
    /// `other_crossing`, with lines for points.
    pub fn other_tangent(inner: Envelope, side: L, turned: L) -> L {
        side.gp(touches(inner, turned)) - turned.gp(2.0 * (inner.of(side) & turned).s())
    }

    /// The vertices of the path on the outer conic, from a start on it along a tangent of the
    /// inner conic through it: at each vertex the other tangent through it, and where that
    /// meets the outer conic again.
    pub fn path(outer: Conic, inner: Envelope, start: P, side: L, steps: usize) -> Vec<P> {
        let mut vertices = vec![start];
        let mut side = side;
        for _ in 0..steps {
            let last = *vertices.last().expect("a vertex");
            side = other_tangent(inner, side, side | last).normalized();
            let vertex = other_crossing(outer, last, side ^ infinity());
            vertices.push(vertex.unitized());
        }
        vertices
    }

    /// The semi-axes of the ellipse.
    pub const SEMI: [f64; 2] = [2.0, 1.2];
    /// The corners of the pentagon, as angles around the ellipse, bunched to one side.
    pub const CORNERS: [f64; 5] = [0.1, 0.8, 1.9, 3.0, 5.2];

    /// Where the ellipse sits: its centre at (0.4, -0.2), its first axis turned from x by 0.5.
    pub fn placement() -> Unit<Motor<(), f64>> {
        Motor::translation(Point::direction(0.4, -0.2)) * Motor::rotation(Point::xy(0.0, 0.0), 0.5)
    }

    /// The ellipse: the lines through its centre across each of its axes (the joins of the
    /// centre with the headings of the other axis, placed), each weighted by the inverse square
    /// of the semi-axis, less the line at infinity twice.
    pub fn ellipse() -> Conic {
        let p = P::slot();
        let mut conic = -(infinity() * (infinity() & p));
        let [along_x, along_y] = headings();
        for (heading, semi) in [along_y, along_x].into_iter().zip(SEMI) {
            let through = placement() >> (crossing(0.0, 0.0) & heading);
            conic += (through * (through & p)).gp(1.0 / (semi * semi));
        }
        conic
    }

    /// The point at an angle on the ellipse, drawn in towards its centre by the scale: the
    /// point of the circle of that radius at the angle, stretched along the axes and placed.
    pub fn on_ellipse(angle: f64, scale: f64) -> P {
        let phasor = Point::<(), f64>::direction_from_angle(angle);
        let [x, y] = [phasor.e20(), phasor.e01()];
        placement() >> crossing(scale * SEMI[0] * x, scale * SEMI[1] * y)
    }

    /// The two inner conics: touching the inscribed pentagon, and touching it shrunk.
    pub fn inners() -> [Envelope; 2] {
        [1.0, 0.97].map(|scale| {
            let corners = CORNERS.map(|a| on_ellipse(a, scale));
            envelope([0, 1, 2, 3, 4].map(|i| corners[i] & corners[(i + 1) % 5]))
        })
    }

    /// From the point of the ellipse at an angle, the path of the given number of sides about
    /// each inner conic.
    pub fn paths(angle: f64, steps: usize) -> [Vec<P>; 2] {
        let outer = ellipse();
        let start = on_ellipse(angle, 1.0);
        inners().map(|inner| path(outer, inner, start, tangents(inner, start)[0], steps))
    }

    /// The distance between two points at unit weight: the gap of a path.
    pub fn gap(a: P, b: P) -> f64 {
        (a & b).norm()
    }
}

use poncelet::*;

const SECONDS: f32 = 6.0;
const TITLES: [&str; 2] = ["CLOSES FROM EVERY START", "MISSES FROM EVERY START"];

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let angle = core::f64::consts::TAU * f64::from(t / SECONDS) + 0.05;
    let outer = ellipse();
    let inners = inners();
    let paths = paths(angle, 5);
    for (k, (inner, corners)) in inners.into_iter().zip(&paths).enumerate() {
        let rect = plot::panel(c, k, 2).inset(14.0, 90.0, 14.0, 14.0);
        let ax = Axes::equal(rect, Point2::xy(0.4, -0.25), 2.2);
        let conic = inner.inverse();
        let level = |f: Conic| {
            move |p: Point2| {
                let [x, y] = p.to_euclidean().map(f64::from);
                on(f, crossing(x, y)) as f32
            }
        };
        ax.contour(c, level(outer), 260, 0.0, 2.0, palette::ink());
        ax.contour(c, level(conic), 260, 0.0, 1.8, palette::sky());
        ax.polyline(c, corners, 1.8, palette::red());
        ax.scatter(c, &corners[1..5], Marker::Dot, 6.0, palette::red());
        ax.scatter(c, &corners[..1], Marker::Dot, 12.0, palette::orange());
        ax.scatter(c, &corners[5..], Marker::Ring, 18.0, palette::red());
        let colour = if k == 0 {
            palette::green()
        } else {
            palette::yellow()
        };
        // The notes sit a tenth of a unit in from the panel's top left and bottom left corners.
        let unit = ax.scale();
        let title = rect.lo + Point2::direction(0.1, 0.1).gp(unit);
        c.text(TITLES[k], title, 12.0, colour, Align::Left);
        let miss = gap(corners[5], corners[0]);
        let note = rect.bottom_left() + Point2::direction(0.1, -0.15).gp(unit);
        let text = format!("GAP AFTER FIVE SIDES: {miss:.1E}");
        c.text(&text, note, 10.0, palette::grid(), Align::Left);
    }
    caption(
        c,
        "PONCELET'S PORISM",
        "A PENTAGON BETWEEN TWO CONICS (PGA2D): IT CLOSES FROM EVERY START OR NONE",
    );
}

fn main() {
    run(Anim::new("poncelet", SECONDS).size(960, 540), draw);
}

#[cfg(test)]
mod tests {
    use super::poncelet::*;

    /// numga's scenario checks, over five starts as numga's test runs: every vertex lies on the
    /// ellipse, every side touches its inner conic; the path about the first inner conic closes
    /// after five sides from every start, the path about the second does not.
    #[test]
    fn scenes_pass_their_checks() {
        let outer = ellipse();
        let inners = inners();
        let frames = 5;
        for k in 0..frames {
            let angle = core::f64::consts::TAU * k as f64 / frames as f64 + 0.05;
            let paths = paths(angle, 5);
            for (inner, vertices) in inners.iter().zip(&paths) {
                let scale = inner.c.iter().flatten().fold(0.0f64, |m, v| m.max(v.abs()));
                for v in vertices {
                    assert!(on(outer, *v).abs() < 1e-12, "{}", on(outer, *v));
                }
                for pair in vertices.windows(2) {
                    let side = (pair[0] & pair[1]).normalized();
                    let touch = touches(*inner, side) / scale;
                    assert!(touch.abs() < 1e-10, "{touch}");
                }
            }
            assert!(gap(paths[0][5], paths[0][0]) < 1e-11);
            assert!(gap(paths[1][5], paths[1][0]) > 1e-2);
        }
    }

    /// Both tangents through a point of the ellipse touch the inner conic.
    #[test]
    fn tangents_touch() {
        let inner = inners()[0];
        let scale = inner.c.iter().flatten().fold(0.0f64, |m, v| m.max(v.abs()));
        for l in tangents(inner, on_ellipse(2.5, 1.0)) {
            assert!(touches(inner, l).abs() / scale < 1e-12);
        }
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws(super::draw, 0.5);
    }
}
