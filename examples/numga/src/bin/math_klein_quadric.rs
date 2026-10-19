//! numga's `math/klein_quadric`: four lines in space, and the two lines that meet all four.
//!
//! A line in space is a bivector of PGA3D, but most bivectors are not lines: a bivector is a
//! line when its wedge with itself vanishes, and two lines meet when their wedge vanishes. Both
//! are one form, `KLEIN = Line ^ Line`, a quadratic form on the six-dimensional space of
//! bivectors. The lines are its zeros, the Klein quadric, and two lines meet when the form pairs
//! them to zero.
//!
//! Meeting a given line is one linear condition on a bivector. Four lines leave a pencil of
//! bivectors that meet all four: the combinations of two of them. Along the pencil the Klein
//! form is a quadratic with two zeros, so two bivectors of the pencil are lines, the two
//! transversals. For real lines they are either both real or a complex-conjugate pair; the
//! transversals here have complex coefficients (`gax::Complex` as the coefficient type), so
//! that the pair is there in every case.
//!
//! In space the same count reads differently. The lines that meet three of the four sweep out a
//! ruled quadric, here a hyperboloid; the fourth line crosses it at two points, and through each
//! passes the line of the sweep that meets the fourth line too.
//!
//! The animation: three lines on the hyperboloid `x² + y² = 1 + z²` and a steep fourth line
//! swinging across it, through its waist and out the other side, while the view turns. Outside,
//! the fourth line crosses the hyperboloid twice and both transversals are real (red); through
//! the waist it misses the hyperboloid and the transversals are a complex-conjugate pair, still
//! lines meeting all four, but not drawn in real space. On the right, the discriminant of the
//! Klein form along the pencil changes sign as they do.

use gax::Complex;
use gax::pga3d::{Line, Motor, Plane, Point, intrinsic};

use gax_numga_examples::scene3::panel3;
use gax_numga_examples::signal::phasor;
use gax_numga_examples::{
    Align, Anim, Axes, Camera, Canvas, Lens, Light, Marker, ORIGIN3, Point2, Rect, backdrop,
    caption, measure, palette, run,
};

mod klein {
    use super::*;
    use gax::pga3d::Pseudoscalar;

    /// A complex number.
    pub type C = Complex<f64>;

    /// A line with complex coefficients.
    pub type CL = Line<(), C>;
    /// A point with complex coefficients.
    pub type CP = Point<(), C>;
    /// A quadric: a map from points to planes, zero on its points. A point joined with a line
    /// and met with others gives a plane through points, oriented within itself: intrinsic.
    pub type Quadric = intrinsic::Plane<(Point,)>;

    /// The Klein form: zero on one bivector twice exactly when it is a line, zero on two lines
    /// exactly when they meet.
    pub fn klein() -> Pseudoscalar<(Line, Line)> {
        Line::slot() ^ Line::slot()
    }

    /// The Klein form on two bivectors, as a number.
    pub fn pairing(a: Line, b: Line) -> f64 {
        klein().of(a).of(b).dual().s()
    }

    /// The plane at infinity as the join of the three directions, whose meet with a line is
    /// the line's point at infinity.
    pub fn horizon() -> intrinsic::Plane {
        Point::direction(1.0, 0.0, 0.0)
            & Point::direction(0.0, 1.0, 0.0)
            & Point::direction(0.0, 0.0, 1.0)
    }

    /// The two roots of the quadratic `a + 2 b t + c t²` along the combinations of two ends, as
    /// the weights of the ends, one root per sign: `first (sign root - b) + second a`, with
    /// `root² = b² - a c`.
    pub fn roots(a: f64, b: f64, c: f64) -> [[C; 2]; 2] {
        let root = C::real(b * b - a * c).sqrt();
        [1.0, -1.0].map(|sign| [root.scale(sign) - C::real(b), C::real(a)])
    }

    /// The pencil of bivectors that meet all four lines: each line weighted by the Klein form
    /// between it and a bivector is zero exactly on the bivectors that meet all four, which are
    /// the right singular vectors of zero singular value. Returns the pencil's two bivectors.
    pub fn pencil(lines: &[Line; 4]) -> [Line; 2] {
        // The Klein form of two axes is a signed volume, intrinsic, and so are the misses.
        let misses: intrinsic::Line<(Line,)> =
            lines.iter().map(|l| *l * klein().of(*l).dual()).sum();
        let right = misses.svd().inputs;
        [right[4], right[5]]
    }

    /// The discriminant `b² - a c` of the Klein form along the pencil, over the size of the form
    /// there, `a² + 2 b² + c²`: positive when the transversals are real, negative when they are
    /// complex. The pencil's two bivectors are any orthonormal pair of the null space, and both
    /// the discriminant and this size are the same for every such pair.
    pub fn discriminant(lines: &[Line; 4]) -> f64 {
        let [first, second] = pencil(lines);
        let (a, b, c) = (
            pairing(first, first),
            pairing(first, second),
            pairing(second, second),
        );
        (b * b - a * c) / (a * a + 2.0 * b * b + c * c)
    }

    /// A complex line divided by the square root of the sum of the squares of its direction
    /// coefficients: unit norm, as numga normalizes complex bivectors.
    pub fn normalized(l: CL) -> CL {
        let [a, b, c] = [l.c[0], l.c[1], l.c[2]];
        l.gp((a * a + b * b + c * c).sqrt().recip())
    }

    /// The two lines that meet all four lines, at unit norm: the two zeros of the Klein form
    /// along the pencil.
    pub fn transversals(lines: &[Line; 4]) -> [CL; 2] {
        let [first, second] = pencil(lines);
        let weights = roots(
            pairing(first, first),
            pairing(first, second),
            pairing(second, second),
        );
        let (f, s) = (first.map_coefs(C::real), second.map_coefs(C::real));
        weights.map(|[u, v]| normalized(f.gp(u) + s.gp(v)))
    }

    /// The quadric swept by the lines that meet three lines, as a map from points to planes.
    /// The plane joining a point to the first line meets the third line in a point; the plane
    /// joining that point to the second line holds the first point exactly when a line through
    /// it meets all three.
    pub fn ruled_quadric(first: Line, second: Line, third: Line) -> Quadric {
        ((Point::slot() & first) ^ third) & second
    }

    /// The two points where a line crosses a quadric, at unit weight.
    pub fn crossings(surface: Quadric, line: Line) -> [CP; 2] {
        // The line's point nearest the origin and its point at infinity span it.
        let (first, second) = frame(line);
        // The quadric's map need not be symmetric: the cross term is the mean of both orders.
        let at = |x: Point, y: Point| (surface.of(x) & y).s();
        let cross = (at(first, second) + at(second, first)) / 2.0;
        let (f, s) = (first.map_coefs(C::real), second.map_coefs(C::real));
        roots(at(first, first), cross, at(second, second)).map(|[u, v]| {
            let p = f.gp(u) + s.gp(v);
            // At unit weight: the weight is the pairing with the plane at infinity.
            let weight = (Plane::infinity().map_coefs(C::real) & p).s();
            p.gp(weight.recip())
        })
    }

    /// The line through a point that meets two lines: the meet of the planes joining the point
    /// to each line.
    pub fn through(point: Point, second: Line, third: Line) -> Line {
        (point & second) ^ (point & third)
    }

    /// The point of a line nearest the origin, at unit weight, and its unit direction (its
    /// point at infinity).
    pub fn frame(line: Line) -> (Point, Point) {
        let unit = line.normalized();
        (
            ((Point::origin() | unit) ^ unit).unitized(),
            unit ^ horizon(),
        )
    }

    /// Points spread over a whole line, its point at infinity included: the nearest point
    /// turned towards the unit direction by angles spaced evenly over half a turn.
    pub fn spread(line: Line, count: usize) -> Vec<Point> {
        let (nearest, heading) = frame(line);
        (0..count)
            .map(|k| {
                let turned = phasor(core::f64::consts::PI * k as f64 / count as f64);
                nearest.gp(turned.e20()) + heading.gp(turned.e01())
            })
            .collect()
    }

    /// Lines of the hyperboloid `x² + y² = 1 + z²` through the points of its waist at the given
    /// angles, all leaning the same way: the line through (1, 0, 0) leaning along (0, 1, 1), the
    /// meet of the planes `x = 1` and `y = z`, turned about z.
    pub fn waist_lines(angles: [f64; 3]) -> [Line; 3] {
        let line = Plane::new(1.0, 0.0, 0.0, -1.0) ^ Plane::new(0.0, 1.0, -1.0, 0.0);
        angles.map(|a| Motor::rotation_about(Point::direction(0.0, 0.0, 1.0), a) >> line)
    }

    /// A steep line crossing the plane `z = 0` at an offset along x, beside the x axis: the
    /// meet of the two planes through it along x and along y.
    pub fn steep(offset: f64) -> Line {
        let base =
            Motor::translation(Point::direction(offset, 0.0, 0.0)) >> Point::xyz(0.0, 0.4, 0.0);
        let along = base & Point::direction(0.3, -0.2, 1.0);
        (along & Point::direction(1.0, 0.0, 0.0)) ^ (along & Point::direction(0.0, 1.0, 0.0))
    }

    /// The three lines of the scenes.
    pub fn three() -> [Line; 3] {
        waist_lines([0.3, 2.3, 4.4])
    }

    /// One scene: the four lines, both families of lines on the quadric of the first three (the
    /// lines meeting the three, through points spread over the first; and the lines meeting
    /// three of those), the two transversals of all four and the two points where the fourth
    /// line crosses the quadric.
    pub struct Scene {
        pub lines: [Line; 4],
        pub rulings: [Vec<Line>; 2],
        pub across: [CL; 2],
        pub crossing: [CP; 2],
    }

    pub fn scene(three: [Line; 3], fourth: Line, count: usize) -> Scene {
        let [first, second, third] = three;
        let lines = [first, second, third, fourth];
        let surface = ruled_quadric(first, second, third);
        let sweep: Vec<Line> = spread(first, count)
            .into_iter()
            .map(|p| through(p, second, third))
            .collect();
        let family: Vec<Line> = spread(sweep[0], count)
            .into_iter()
            .map(|p| through(p, sweep[count / 3], sweep[2 * count / 3]))
            .collect();
        Scene {
            lines,
            rulings: [sweep, family],
            across: transversals(&lines),
            crossing: crossings(surface, fourth),
        }
    }

    /// The offset of the fourth line at a phase of the loop.
    pub fn offset(phase: f64) -> f64 {
        1.5 * phasor(core::f64::consts::TAU * phase).e20()
    }

    /// Whether a complex value is real up to round-off, by its coefficients.
    pub fn real(c: &[C]) -> bool {
        let big = c.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        c.iter().all(|v| v.im.abs() <= 1e-6 * big)
    }

    /// A complex line's real part.
    pub fn real_line(l: CL) -> Line {
        l.map_coefs(|v| v.re)
    }

    /// A complex point's real part.
    pub fn real_point(p: CP) -> Point {
        p.map_coefs(|v| v.re)
    }
}

use klein::*;

const SECONDS: f32 = 9.6;
/// Lines are drawn inside this ball.
const RADIUS: f64 = 3.0;

/// The two ends of a line's chord of the ball, or `None` when the line misses it: from the
/// line's point nearest the centre, half the chord each way along it. Along the line the ball
/// is the quadratic `s² + d² - R²` in the distance `s` from the nearest point, `d` off the
/// centre: its roots are `±` the half chord.
fn chord(line: Line) -> Option<[Point; 2]> {
    let (nearest, heading) = frame(line);
    let off_centre = (Point::origin() & nearest).norm_squared();
    let (_, half_chord) = measure::roots(1.0, 0.0, off_centre - RADIUS * RADIUS)?;
    let half = half_chord / heading.ideal_norm();
    Some([-half, half].map(|s| nearest + heading.gp(s)))
}

/// The lines in space, turning once around the hyperboloid over the loop: both rulings, the
/// four lines, the transversals and where they cross when they are real, and the verdict under
/// them. Returns the verdict's colour.
fn draw_lines(c: &mut Canvas, left: Rect, s: &Scene, phase: f64) -> Light {
    let up = Point2::direction(0.0, -1.0);
    let cam = Camera::orbit(
        left,
        ORIGIN3,
        15.0,
        (-60.0f32).to_radians() + core::f32::consts::TAU * phase as f32,
        18.0f32.to_radians(),
        Lens::Perspective(0.44),
    );
    let visible = s.across.iter().all(|l| real(&l.c));
    panel3(c, left, cam, |scene3| {
        let mut segment = |l: Line, width: f32, colour| {
            if let Some([a, b]) = chord(l) {
                scene3.seg(a, b, width, colour);
            }
        };
        for l in &s.rulings[0] {
            segment(*l, 0.9, palette::SKY.faded(0.55));
        }
        for l in &s.rulings[1] {
            segment(*l, 0.9, palette::GRID.faded(0.9));
        }
        for l in &s.lines[..3] {
            segment(*l, 2.6, palette::BLUE);
        }
        segment(s.lines[3], 2.6, palette::ORANGE);
        if visible {
            for l in &s.across {
                segment(real_line(*l), 2.8, palette::RED);
            }
            for p in &s.crossing {
                scene3.dot(real_point(*p), Marker::Dot, 10.0, palette::RED);
            }
        }
    });
    let (verdict, tone) = if visible {
        ("TWO REAL TRANSVERSALS", palette::RED)
    } else {
        ("TWO COMPLEX-CONJUGATE TRANSVERSALS", palette::YELLOW)
    };
    let bottom_middle = left.bottom_middle();
    c.text(
        verdict,
        bottom_middle + up.gp(18.0),
        13.0,
        tone,
        Align::Center,
    );
    tone
}

/// The discriminant along the pencil over the swing, the swing's phase now marked in the
/// verdict's colour, and notes under it.
fn draw_discriminant(c: &mut Canvas, right: Rect, s: &Scene, phase: f64, tone: Light) {
    let h = c.rect().height();
    let down = Point2::direction(0.0, 1.0);
    // The graph's points.
    let graph = |phase: f64, value: f64| gax::pga2d::Point::xy(phase, value);
    let values: Vec<f64> = (0..=120)
        .map(|k| {
            let [a, b, c] = three();
            discriminant(&[a, b, c, steep(offset(k as f64 / 120.0))])
        })
        .collect();
    let curve: Vec<_> = values
        .iter()
        .enumerate()
        .map(|(k, &v)| graph(k as f64 / 120.0, v))
        .collect();
    let (lo, hi) = values
        .iter()
        .fold((0.0f64, 0.0f64), |(lo, hi), &v| (lo.min(v), hi.max(v)));
    let ax = Axes::new(
        right.inset(50.0, h * 0.2, 20.0, h * 0.28),
        [0.0, 1.0],
        [lo as f32 * 1.15, hi as f32 * 1.15],
    );
    ax.frame(c, "DISCRIMINANT ON THE PENCIL", "PHASE OF THE SWING", "");
    ax.line(c, graph(0.0, 0.0), graph(1.0, 0.0), 1.0, palette::GRID);
    ax.polyline(c, &curve, 1.6, palette::SKY);
    let now = graph(phase, discriminant(&s.lines));
    ax.scatter(c, &[now], Marker::Dot, 10.0, tone);
    let notes = [
        "ABOVE ZERO: THE FOURTH LINE CUTS",
        "THE HYPERBOLOID, TWO REAL LINES;",
        "BELOW ZERO: IT MISSES THE WAIST,",
        "A COMPLEX-CONJUGATE PAIR",
    ];
    // The notes stand under the graph, a little out to its left, one line apart.
    let first = ax.rect.bottom_left() + Point2::direction(-30.0, h * 0.13);
    for (k, note) in notes.iter().enumerate() {
        let at = first + down.gp(k as f32 * h * 0.028);
        c.text(note, at, 10.0, palette::INK, Align::Left);
    }
}

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let screen = c.rect();
    let phase = f64::from(t / SECONDS);
    let s = scene(three(), steep(offset(phase)), 36);
    let tone = draw_lines(c, screen.part(0.0, 0.0, 0.62, 1.0), &s, phase);
    draw_discriminant(c, screen.part(0.62, 0.0, 1.0, 1.0), &s, phase, tone);
    caption(
        c,
        "THE KLEIN QUADRIC: LINES MEETING FOUR LINES",
        "ZEROS OF THE KLEIN FORM LINE ^ LINE ON A PENCIL, COMPLEX COEFFICIENTS",
    );
}

fn main() {
    run(Anim::new("klein quadric", SECONDS).size(960, 540), draw);
}

#[cfg(test)]
mod tests {
    use super::klein::*;
    use gax::pga3d::Line;

    fn small(c: &[C], tol: f64) -> bool {
        c.iter().all(|v| v.abs() <= tol)
    }

    fn complex_line(l: Line) -> CL {
        l.map_coefs(C::real)
    }

    /// numga's scenario checks: the transversals are lines and meet all four lines; the
    /// crossings lie on the fourth line and on the quadric, each on one of the transversals;
    /// both families lie on the quadric.
    fn check(s: &Scene) {
        let surface = ruled_quadric(s.lines[0], s.lines[1], s.lines[2]);
        for a in &s.across {
            let twice = *a;
            assert!(small(&(*a ^ twice).c, 1e-12));
            for l in &s.lines {
                assert!(
                    small(&(complex_line(*l) ^ *a).c, 1e-12),
                    "{:?}",
                    (complex_line(*l) ^ *a)
                );
            }
        }
        for p in &s.crossing {
            assert!(small(&(*p & complex_line(s.lines[3])).c, 1e-10));
            assert!(small(
                &[(surface.map_coefs(C::real).of(*p) & *p).s()],
                1e-10
            ));
            let off = s
                .across
                .iter()
                .map(|a| (*p & *a).c.iter().fold(0.0f64, |m, v| m.max(v.abs())))
                .fold(f64::INFINITY, f64::min);
            assert!(off < 1e-10, "{off}");
        }
        for family in &s.rulings {
            for l in family {
                for e in spread(*l, 3) {
                    assert!((surface.of(e) & e).s().abs() < 1e-10);
                }
            }
        }
    }

    /// numga's first test: a steep line outside the waist crosses the hyperboloid and has two
    /// real transversals; one through the waist misses it, and its transversals are a
    /// complex-conjugate pair, still lines meeting all four. The conjugate of one is a multiple
    /// of the other: numga checks that the stacked pair has a zero singular value; here, that
    /// every 2 x 2 minor of the pair vanishes.
    #[test]
    fn both_transversals_exist_whether_the_fourth_line_crosses_the_hyperboloid_or_not() {
        let outside = scene(three(), steep(1.4), 12);
        let waist = scene(three(), steep(0.0), 12);
        assert!(outside.across.iter().all(|l| real(&l.c)));
        assert!(waist.across.iter().all(|l| !real(&l.c)));
        let (a, b) = (waist.across[0].c.map(C::conj), waist.across[1].c);
        for i in 0..6 {
            for j in 0..6 {
                assert!((a[i] * b[j] - a[j] * b[i]).abs() < 1e-12);
            }
        }
        check(&outside);
        check(&waist);
    }

    /// numga's second test: four frames of the swing pass their checks.
    #[test]
    fn scenes_pass_their_checks() {
        for k in 0..4 {
            check(&scene(three(), steep(offset(k as f64 / 4.0)), 12));
        }
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws(super::draw, 0.5);
    }
}
