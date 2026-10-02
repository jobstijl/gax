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
//! transversals here have complex coefficients (a local complex type serves as gax's
//! coefficient), so that the pair is there in every case.
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

use gax::pga3d::{Line, Motor, Plane, Point};
use gax_numga_examples::{
    Align, Anim, Axes, Camera, Canvas, Lens, Marker, Scene3, backdrop, caption, palette, plot, run,
};

/// Complex numbers, as gax coefficients: the products, meets and joins of the transversals.
mod complex {
    use core::ops::{Add, Div, Mul, Neg, Sub};

    /// A complex number.
    #[derive(Clone, Copy, Debug, PartialEq, Default)]
    pub struct C64 {
        pub re: f64,
        pub im: f64,
    }

    impl C64 {
        pub const fn new(re: f64, im: f64) -> C64 {
            C64 { re, im }
        }
        pub fn abs(self) -> f64 {
            self.re.hypot(self.im)
        }
        /// The principal square root.
        pub fn sqrt(self) -> C64 {
            let r = self.abs();
            let re = ((r + self.re) * 0.5).max(0.0).sqrt();
            let im = ((r - self.re) * 0.5).max(0.0).sqrt();
            C64::new(re, if self.im < 0.0 { -im } else { im })
        }
    }

    impl From<f64> for C64 {
        fn from(re: f64) -> C64 {
            C64::new(re, 0.0)
        }
    }

    impl Add for C64 {
        type Output = C64;
        fn add(self, o: C64) -> C64 {
            C64::new(self.re + o.re, self.im + o.im)
        }
    }

    impl Sub for C64 {
        type Output = C64;
        fn sub(self, o: C64) -> C64 {
            C64::new(self.re - o.re, self.im - o.im)
        }
    }

    impl Mul for C64 {
        type Output = C64;
        fn mul(self, o: C64) -> C64 {
            C64::new(
                self.re * o.re - self.im * o.im,
                self.re * o.im + self.im * o.re,
            )
        }
    }

    impl Div for C64 {
        type Output = C64;
        fn div(self, o: C64) -> C64 {
            let d = o.re * o.re + o.im * o.im;
            C64::new(
                (self.re * o.re + self.im * o.im) / d,
                (self.im * o.re - self.re * o.im) / d,
            )
        }
    }

    impl Neg for C64 {
        type Output = C64;
        fn neg(self) -> C64 {
            C64::new(-self.re, -self.im)
        }
    }

    impl gax::Coef for C64 {
        fn zero() -> C64 {
            C64::new(0.0, 0.0)
        }
        fn one() -> C64 {
            C64::new(1.0, 0.0)
        }
        fn from_i64(i: i64) -> C64 {
            C64::new(i as f64, 0.0)
        }
        fn from_f64(f: f64) -> C64 {
            C64::new(f, 0.0)
        }
    }
}

mod klein {
    use super::complex::C64;
    use super::*;
    use gax::pga3d::Pseudoscalar;

    pub type L = Line<(), f64>;
    pub type P = Point<(), f64>;
    /// A line with complex coefficients.
    pub type CL = Line<(), C64>;
    /// A point with complex coefficients.
    pub type CP = Point<(), C64>;
    /// A quadric: a map from points to planes, zero on its points.
    pub type Quadric = Plane<(Point,), f64>;

    /// The Klein form: zero on one bivector twice exactly when it is a line, zero on two lines
    /// exactly when they meet.
    pub fn klein() -> Pseudoscalar<(Line, Line), f64> {
        Line::slot() ^ Line::slot()
    }

    /// The Klein form on two bivectors, as a number.
    pub fn pairing(a: L, b: L) -> f64 {
        klein().of(a).of(b).dual().s()
    }

    /// The origin.
    pub fn origin() -> P {
        Point::xyz(0.0, 0.0, 0.0)
    }

    /// The plane at infinity.
    pub fn infinity() -> Plane<(), f64> {
        Plane::new(0.0, 0.0, 0.0, 1.0)
    }

    /// A real value with complex coefficients.
    pub fn complexify<const N: usize>(c: [f64; N]) -> [C64; N] {
        c.map(C64::from)
    }

    /// The two roots of the quadratic `a + 2 b t + c t²` along the combinations of two ends, as
    /// the weights of the ends, one root per sign: `first (sign root - b) + second a`, with
    /// `root² = b² - a c`.
    pub fn roots(a: f64, b: f64, c: f64) -> [[C64; 2]; 2] {
        let root = C64::from(b * b - a * c).sqrt();
        [1.0, -1.0].map(|sign| [C64::new(sign * root.re - b, sign * root.im), C64::from(a)])
    }

    /// The pencil of bivectors that meet all four lines: each line weighted by the Klein form
    /// between it and a bivector is zero exactly on the bivectors that meet all four, which are
    /// the right singular vectors of zero singular value. Returns the pencil's two bivectors.
    pub fn pencil(lines: &[L; 4]) -> [L; 2] {
        let mut misses: Line<(Line,), f64> = Line::zero();
        for l in lines {
            misses += *l * klein().of(*l).dual();
        }
        let (_, _, right) = misses.svd();
        [right[4], right[5]]
    }

    /// The discriminant `b² - a c` of the Klein form along the pencil, over the size of its
    /// coefficients: positive when the transversals are real, negative when they are complex.
    pub fn discriminant(lines: &[L; 4]) -> f64 {
        let [first, second] = pencil(lines);
        let (a, b, c) = (
            pairing(first, first),
            pairing(first, second),
            pairing(second, second),
        );
        (b * b - a * c) / (a * a + b * b + c * c)
    }

    /// A complex line divided by the square root of the sum of the squares of its direction
    /// coefficients: unit norm, as numga normalizes complex bivectors.
    pub fn normalized(l: CL) -> CL {
        let [a, b, c] = [l.c[0], l.c[1], l.c[2]];
        let norm = (a * a + b * b + c * c).sqrt();
        CL::from_coeffs(l.c.map(|v| v / norm))
    }

    /// The two lines that meet all four lines, at unit norm: the two zeros of the Klein form
    /// along the pencil.
    pub fn transversals(lines: &[L; 4]) -> [CL; 2] {
        let [first, second] = pencil(lines);
        let weights = roots(
            pairing(first, first),
            pairing(first, second),
            pairing(second, second),
        );
        let (f, s) = (
            CL::from_coeffs(complexify(first.c)),
            CL::from_coeffs(complexify(second.c)),
        );
        weights.map(|[u, v]| {
            normalized(CL::from_coeffs(core::array::from_fn(|i| {
                f.c[i] * u + s.c[i] * v
            })))
        })
    }

    /// The quadric swept by the lines that meet three lines, as a map from points to planes.
    /// The plane joining a point to the first line meets the third line in a point; the plane
    /// joining that point to the second line holds the first point exactly when a line through
    /// it meets all three.
    pub fn ruled_quadric(first: L, second: L, third: L) -> Quadric {
        ((Point::slot() & first) ^ third) & second
    }

    /// The two points where a line crosses a quadric, at unit weight.
    pub fn crossings(surface: Quadric, line: L) -> [CP; 2] {
        // The line's point nearest the origin and its point at infinity span it.
        let first = (origin() | line) ^ line;
        let second = line ^ infinity();
        // The quadric's map need not be symmetric: the cross term is the mean of both orders.
        let at = |x: P, y: P| (surface.of(x) & y).s();
        let cross = (at(first, second) + at(second, first)) / 2.0;
        let (f, s) = (complexify(first.c), complexify(second.c));
        roots(at(first, first), cross, at(second, second)).map(|[u, v]| {
            let p: [C64; 4] = core::array::from_fn(|i| f[i] * u + s[i] * v);
            // At unit weight: the weight is the pairing with the plane at infinity.
            let weight =
                (Plane::<(), C64>::from_coeffs(complexify(infinity().c)) & CP::from_coeffs(p)).s();
            CP::from_coeffs(p.map(|v| v / weight))
        })
    }

    /// The line through a point that meets two lines: the meet of the planes joining the point
    /// to each line.
    pub fn through(point: P, second: L, third: L) -> L {
        (point & second) ^ (point & third)
    }

    /// The point of a line nearest the origin, at unit weight, and its unit direction (its
    /// point at infinity), for a line at unit norm.
    pub fn frame(line: L) -> (P, P) {
        let unit = line.normalized().into_inner();
        let nearest = (origin() | unit) ^ unit;
        (nearest.gp(1.0 / nearest.e123()), unit ^ infinity())
    }

    /// Points spread over a whole line, its point at infinity included: the nearest point
    /// turned towards the unit direction by angles spaced evenly over half a turn.
    pub fn spread(line: L, count: usize) -> Vec<P> {
        let (nearest, heading) = frame(line);
        (0..count)
            .map(|k| {
                let a = core::f64::consts::PI * k as f64 / count as f64;
                nearest.gp(a.cos()) + heading.gp(a.sin())
            })
            .collect()
    }

    /// Lines of the hyperboloid `x² + y² = 1 + z²` through the points of its waist at the given
    /// angles, all leaning the same way: the line through (1, 0, 0) leaning along (0, 1, 1),
    /// turned about z.
    pub fn waist_lines(angles: [f64; 3]) -> [L; 3] {
        let line = Point::xyz(1.0, 0.0, 0.0) & Point::direction(0.0, 1.0, 1.0);
        angles.map(|a| Motor::rotation_about(0.0, 0.0, 1.0, a) >> line)
    }

    /// A steep line crossing the plane `z = 0` at an offset along x, beside the x axis.
    pub fn steep(offset: f64) -> L {
        let base = Motor::translation(offset, 0.0, 0.0) >> Point::xyz(0.0, 0.4, 0.0);
        base & Point::direction(0.3, -0.2, 1.0)
    }

    /// The three lines of the scenes.
    pub fn three() -> [L; 3] {
        waist_lines([0.3, 2.3, 4.4])
    }

    /// One scene: the four lines, both families of lines on the quadric of the first three (the
    /// lines meeting the three, through points spread over the first; and the lines meeting
    /// three of those), the two transversals of all four and the two points where the fourth
    /// line crosses the quadric.
    pub struct Scene {
        pub lines: [L; 4],
        pub rulings: [Vec<L>; 2],
        pub across: [CL; 2],
        pub crossing: [CP; 2],
    }

    pub fn scene(three: [L; 3], fourth: L, count: usize) -> Scene {
        let [first, second, third] = three;
        let lines = [first, second, third, fourth];
        let surface = ruled_quadric(first, second, third);
        let sweep: Vec<L> = spread(first, count)
            .into_iter()
            .map(|p| through(p, second, third))
            .collect();
        let family: Vec<L> = spread(sweep[0], count)
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
        1.5 * (core::f64::consts::TAU * phase).cos()
    }

    /// Whether a complex value is real up to round-off, by its coefficients.
    pub fn real(c: &[C64]) -> bool {
        let big = c.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        c.iter().all(|v| v.im.abs() <= 1e-6 * big)
    }

    /// A complex line's real part.
    pub fn real_line(l: CL) -> L {
        Line::from_coeffs(l.c.map(|v| v.re))
    }

    /// A complex point's real part.
    pub fn real_point(p: CP) -> P {
        Point::from_coeffs(p.c.map(|v| v.re))
    }
}

use klein::*;

const SECONDS: f32 = 9.6;
/// Lines are drawn inside this ball.
const RADIUS: f64 = 3.0;

/// The two ends of a line's chord of the ball, or `None` when the line misses it.
fn chord(line: L) -> Option<[[f32; 3]; 2]> {
    let (nearest, heading) = frame(line);
    let n = nearest.to_euclidean();
    let d = [heading.e032(), heading.e013(), heading.e021()];
    let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    let half = (RADIUS * RADIUS - (n[0] * n[0] + n[1] * n[1] + n[2] * n[2])).sqrt();
    if half.is_nan() {
        return None;
    }
    let end = |s: f64| core::array::from_fn(|i| (n[i] + s * half * d[i] / len) as f32);
    Some([end(-1.0), end(1.0)])
}

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let (w, h) = (c.width as f32, c.height as f32);
    let phase = f64::from(t / SECONDS);
    let s = scene(three(), steep(offset(phase)), 36);

    // The lines in space, turning once around the hyperboloid over the loop.
    let wide = (w * 0.62) as usize;
    let cam = Camera::orbit(
        wide,
        c.height,
        [0.0; 3],
        15.0,
        (-60.0f32).to_radians() + core::f32::consts::TAU * phase as f32,
        18.0f32.to_radians(),
        Lens::Perspective(0.44),
    );
    let mut scene3 = Scene3::new(cam);
    let mut segment = |l: L, width: f32, colour, alpha| {
        if let Some([a, b]) = chord(l) {
            scene3.seg(a, b, width, colour, alpha);
        }
    };
    for l in &s.rulings[0] {
        segment(*l, 0.9, palette::sky(), 0.55);
    }
    for l in &s.rulings[1] {
        segment(*l, 0.9, palette::grid(), 0.9);
    }
    for l in &s.lines[..3] {
        segment(*l, 2.6, palette::blue(), 1.0);
    }
    segment(s.lines[3], 2.6, palette::orange(), 1.0);
    let visible = s.across.iter().all(|l| real(&l.c));
    if visible {
        for l in &s.across {
            segment(real_line(*l), 2.8, palette::red(), 1.0);
        }
        for p in &s.crossing {
            let [x, y, z] = real_point(*p).to_euclidean();
            scene3.dot(
                [x as f32, y as f32, z as f32],
                Marker::Dot,
                10.0,
                palette::red(),
            );
        }
    }
    scene3.draw(c);
    c.text(
        if visible {
            "TWO REAL TRANSVERSALS"
        } else {
            "TWO COMPLEX-CONJUGATE TRANSVERSALS"
        },
        w * 0.31,
        h - 18.0,
        13.0,
        if visible {
            palette::red()
        } else {
            palette::yellow()
        },
        Align::Center,
    );

    // The discriminant along the pencil over the swing.
    let curve: Vec<[f32; 2]> = (0..=120)
        .map(|k| {
            let p = k as f64 / 120.0;
            let [a, b, c] = three();
            [p as f32, discriminant(&[a, b, c, steep(offset(p))]) as f32]
        })
        .collect();
    let (lo, hi) = curve
        .iter()
        .fold((0.0f32, 0.0f32), |(lo, hi), p| (lo.min(p[1]), hi.max(p[1])));
    let ax = Axes::new(
        plot::inset([w * 0.62, 0.0, w, h], 50.0, h * 0.2, 20.0, h * 0.28),
        [0.0, 1.0],
        [lo * 1.15, hi * 1.15],
    );
    ax.frame(c, "DISCRIMINANT ALONG THE PENCIL", "PHASE OF THE SWING", "");
    ax.line(c, [0.0, 0.0], [1.0, 0.0], 1.0, palette::grid(), 1.0);
    ax.polyline(c, &curve, 1.6, palette::sky(), 1.0);
    let now = [phase as f32, discriminant(&s.lines) as f32];
    ax.scatter(
        c,
        &[now],
        Marker::Dot,
        10.0,
        if visible {
            palette::red()
        } else {
            palette::yellow()
        },
        1.0,
    );
    let notes = [
        "ABOVE ZERO: THE FOURTH LINE CROSSES",
        "THE HYPERBOLOID, TWO REAL TRANSVERSALS",
        "BELOW ZERO: IT MISSES THE WAIST, A",
        "COMPLEX-CONJUGATE PAIR OF LINES",
    ];
    for (k, note) in notes.iter().enumerate() {
        c.text(
            note,
            ax.rect[0] - 30.0,
            ax.rect[3] + h * 0.13 + k as f32 * h * 0.028,
            10.0,
            palette::ink(),
            Align::Left,
        );
    }

    caption(
        c,
        "THE KLEIN QUADRIC: LINES MEETING FOUR LINES",
        "ZEROS OF THE KLEIN FORM LINE ^ LINE ALONG A PENCIL (PGA3D, COMPLEX COEFFICIENTS)",
    );
}

fn main() {
    run(Anim::new("klein quadric", SECONDS).size(960, 540), draw);
}

#[cfg(test)]
mod tests {
    use super::complex::C64;
    use super::klein::*;

    /// The pairing of a quadric with a point, both made complex.
    fn on(surface: Quadric, p: CP) -> gax::pga3d::Plane<(), C64> {
        let s =
            gax::pga3d::Plane::<(gax::pga3d::Point,), C64>::from_coeffs(surface.c.map(complexify));
        s.of(p)
    }

    /// A real scalar of a complex value: unused imaginary parts are checked by callers.
    fn scalar(s: gax::pga3d::Scalar<(), C64>) -> C64 {
        s.s()
    }

    fn small(c: &[C64], tol: f64) -> bool {
        c.iter().all(|v| v.abs() <= tol)
    }

    fn complex_line(l: L) -> CL {
        CL::from_coeffs(complexify(l.c))
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
            assert!(small(&[scalar(on(surface, *p) & *p)], 1e-10));
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
        let (a, b) = (
            waist.across[0].c.map(|v| C64::new(v.re, -v.im)),
            waist.across[1].c,
        );
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
        let mut draw = super::draw;
        let c = gax_numga_examples::app::frame(
            &gax_numga_examples::Anim::new("t", 1.0).size(320, 180),
            0.5,
            &mut draw,
        );
        assert!(c.mean()[0] > 0.0);
    }
}
