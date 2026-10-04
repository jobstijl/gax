//! numga's `math/invariant_decomposition`: a rotation of six-dimensional space is three
//! independent turns.
//!
//! A bivector of six dimensions generates a rotation. It is the sum of three bivectors that are
//! each a single plane and commute with one another, so the rotation is the product of three
//! turns, one in each plane, at each plane's own rate. Two ways find the planes.
//!
//! From the wedge powers of the bivector, `pair = (B ^ B) / 2` and `triple = (B ^ pair) / 3`:
//! the squares of the three planes are the roots of a cubic whose coefficients are the scalar
//! parts of `B ~B`, `pair ~pair` and `triple ~triple`, and at each root `square` the plane is
//! `(triple + square B) (pair + square)⁻¹`. That inverse is of a scalar plus a quadvector, whose
//! powers stay scalar plus quadvector: it satisfies a quartic, so the inverse is a cubic in it.
//!
//! From the map `(Vector | B) | B`, with the vector left open: its eigenvalues are the same
//! squares, each twice, and each pair of eigenvectors spans one plane. For a unit direction `u`
//! in a plane, `u | B` sees only that plane, so `u ^ (u | B)` is the plane itself.
//!
//! The same holds for rigid motions, in the projective algebra of six-dimensional space with a
//! seventh direction e0 that squares to zero: a motion's bivector is three commuting planes, now
//! placed in space. Their directions are the spectral planes of its Euclidean part; each plane's
//! placement, its part that involves e0, is fixed by two conditions linear in it (the plane
//! commutes with the bivector, and it stays simple), met together by one least-squares solve.
//!
//! Once the planes are known, the rotation over any time is the product of three plane rotors,
//! each `cos(angle / 2) + part sin(angle / 2) / rate`. The two algebras are declared here with
//! `gax::algebra!`: R(6,0,0) and R(6,0,1), with only the kinds the example needs (a 7D algebra
//! with every grade would take far longer to compile).
//!
//! The animation: a point turned by a random six-dimensional rotation traces a tangle in its
//! first three directions (left, the view turning); seen in each of the three planes it turns on
//! a circle, at that plane's rate.

use gax_numga_examples::scene3::panel3;
use gax_numga_examples::{
    Align, Anim, Axes, Camera, Canvas, Lens, Light, Marker, ORIGIN3, Point2, Rect, backdrop,
    caption, palette, run,
};

gax::algebra! {
    algebra r6 "Euclidean six-dimensional space, R(6,0,0): rotations.";
    basis e1 = 1, e2 = 1, e3 = 1, e4 = 1, e5 = 1, e6 = 1;
    kind Scalar = [1];
    kind Vector = [e1, e2, e3, e4, e5, e6];
    kind Bivector = [e12, e13, e14, e15, e16, e23, e24, e25, e26, e34, e35, e36, e45, e46, e56];
    kind Quadvector = [e1234, e1235, e1236, e1245, e1246, e1256, e1345, e1346, e1356, e1456, e2345, e2346, e2356, e2456, e3456];
    kind Pseudoscalar = [e123456];
    kind ScalarQuadvector = [1, e1234, e1235, e1236, e1245, e1246, e1256, e1345, e1346, e1356, e1456, e2345, e2346, e2356, e2456, e3456];
    kind Even = [1, e12, e13, e14, e15, e16, e23, e24, e25, e26, e34, e35, e36, e45, e46, e56, e1234, e1235, e1236, e1245, e1246, e1256, e1345, e1346, e1356, e1456, e2345, e2346, e2356, e2456, e3456, e123456];
}

gax::algebra! {
    algebra p6 "Plane-based projective geometric algebra of six-dimensional space, R(6,0,1): rigid motions.";
    basis e0 = 0, e1 = 1, e2 = 1, e3 = 1, e4 = 1, e5 = 1, e6 = 1;
    kind Scalar = [1];
    kind EuclideanVector = [e1, e2, e3, e4, e5, e6];
    kind EuclideanBivector = [e12, e13, e14, e15, e16, e23, e24, e25, e26, e34, e35, e36, e45, e46, e56];
    kind NullBivector = [e01, e02, e03, e04, e05, e06];
    kind Bivector = [e01, e02, e03, e04, e05, e06, e12, e13, e14, e15, e16, e23, e24, e25, e26, e34, e35, e36, e45, e46, e56];
    kind Conditions = [e01, e02, e03, e04, e05, e06, e12, e13, e14, e15, e16, e23, e24, e25, e26, e34, e35, e36, e45, e46, e56, e0123, e0124, e0125, e0126, e0134, e0135, e0136, e0145, e0146, e0156, e0234, e0235, e0236, e0245, e0246, e0256, e0345, e0346, e0356, e0456, e1234, e1235, e1236, e1245, e1246, e1256, e1345, e1346, e1356, e1456, e2345, e2346, e2356, e2456, e3456];
}

mod decomposition {
    use super::{p6, r6};
    use gax_numga_examples::rng::{Draw, rng};
    use r6::{Bivector, Even, Pseudoscalar, Quadvector, Scalar, ScalarQuadvector, Vector};

    pub type B = Bivector<(), f64>;
    pub type V = Vector<(), f64>;
    pub type E = Even<(), f64>;
    pub type SQ = ScalarQuadvector<(), f64>;

    /// A scalar of the 6D algebra.
    fn scalar(s: f64) -> Scalar<(), f64> {
        Scalar::from_coeffs([s])
    }

    /// The squares of the three commuting planes of a bivector, and the planes, from its wedge
    /// powers.
    // `b ^ b` is the bivector's wedge square, not a mistake: clippy reads it as one.
    #[allow(clippy::eq_op)]
    pub fn from_wedge_powers(b: B) -> ([f64; 3], [B; 3]) {
        let pair: Quadvector<(), f64> = (b ^ b).gp(0.5);
        let triple: Pseudoscalar<(), f64> = (b ^ pair).gp(1.0 / 3.0);
        // The squares of the planes are the roots of the cubic with these coefficients after
        // the leading one.
        let squares = cubic_roots([
            b.scalar_product(b.reverse()).s(),
            pair.scalar_product(pair.reverse()).s(),
            triple.scalar_product(triple.reverse()).s(),
        ]);
        let parts = squares.map(|square| {
            let top = triple.cast::<Even>() + b.cast::<Even>().gp(square);
            let bottom =
                pair.cast::<ScalarQuadvector>() + scalar(square).cast::<ScalarQuadvector>();
            (top * inverse(bottom)).cast::<Bivector>()
        });
        (squares, parts)
    }

    /// The inverse of a scalar plus a quadvector of six dimensions, as a cubic in it. Its powers
    /// stay scalar plus quadvector, and it satisfies a quartic: four times the scalar parts of
    /// its first four powers are the power sums of the quartic's roots, from which Newton's
    /// identities give the coefficients, so `value (e3 - e2 value + e1 square - cube) = e4`.
    pub fn inverse(value: SQ) -> SQ {
        let square = (value * value).cast::<ScalarQuadvector>();
        let cube = (square * value).cast::<ScalarQuadvector>();
        let fourth = (square * square).cast::<ScalarQuadvector>();
        let sums = [value, square, cube, fourth].map(|p| 4.0 * p.c[0]);
        let e1 = sums[0];
        let e2 = (e1 * sums[0] - sums[1]) / 2.0;
        let e3 = (e2 * sums[0] - e1 * sums[1] + sums[2]) / 3.0;
        let e4 = (e3 * sums[0] - e2 * sums[1] + e1 * sums[2] - sums[3]) / 4.0;
        (scalar(e3).cast::<ScalarQuadvector>() - value.gp(e2) + square.gp(e1) - cube).gp(1.0 / e4)
    }

    /// The squares of the three commuting planes of a bivector, and the planes, from the
    /// eigenvalues and eigenvectors of `(Vector | B) | B`: the squares come each twice, and one
    /// direction of each pair gives its plane. A point turned by the rotation moves with
    /// velocity `point | B`: this is its acceleration, paired with a second open vector to make
    /// the symmetric form that `eigh` takes.
    pub fn from_spectrum(b: B) -> ([f64; 3], [B; 3]) {
        let acceleration = (Vector::slot() | b) | b;
        let (values, directions) = (Vector::slot() | acceleration).eigh();
        // The eigenvalues come in equal pairs, in ascending order: one direction of each pair.
        let spanning = [directions[0], directions[2], directions[4]];
        (
            [values[0], values[2], values[4]],
            spanning.map(|u| u ^ (u | b)),
        )
    }

    /// For each time and plane: the cosine of half the plane's angle, and its sine over the
    /// plane's rate, the square root of minus its square.
    fn half_turns(squares: [f64; 3], time: f64) -> [(f64, f64); 3] {
        squares.map(|s| {
            let rate = (-s).sqrt();
            let angle = time * rate / 2.0;
            (angle.cos(), angle.sin() / rate)
        })
    }

    /// The rotor the bivector generates over a time, as the product of its planes' rotors, each
    /// `cos(angle / 2) + part sin(angle / 2) / rate`.
    pub fn rotor(squares: [f64; 3], parts: [B; 3], time: f64) -> E {
        let turns = half_turns(squares, time);
        (0..3)
            .map(|k| scalar(turns[k].0).cast::<Even>() + parts[k].cast::<Even>().gp(turns[k].1))
            .reduce(|a, b| a * b)
            .expect("three planes")
    }

    /// The start turned by the rotation, at a time.
    pub fn orbit(squares: [f64; 3], parts: [B; 3], start: V, time: f64) -> V {
        let r = rotor(squares, parts, time);
        (r * start * r.reverse()).cast::<Vector>()
    }

    /// A point projected into each plane: `(point | part) part⁻¹`, a vector in the plane.
    pub fn projected(point: V, parts: [B; 3]) -> [V; 3] {
        parts.map(|part| {
            let inverse = part.gp(1.0 / (part * part).s());
            ((point | part) * inverse).cast::<Vector>()
        })
    }

    /// The commuting planes of a bivector of the projective algebra, placed in space, and the
    /// part left over. The planes' directions are the spectral planes of its Euclidean part;
    /// each plane's placement is the null bivector that makes it commute with the motion and
    /// keeps it simple: both conditions are linear in the placement, one a bivector and the
    /// other a quadvector, and one least-squares solve with the placement open meets both.
    pub fn placed(
        motion: p6::Bivector<(), f64>,
    ) -> ([p6::Bivector<(), f64>; 3], p6::Bivector<(), f64>) {
        use p6::{Conditions, EuclideanBivector, EuclideanVector, NullBivector};
        let euclidean = motion.cast::<EuclideanBivector>();
        let acceleration = (EuclideanVector::slot() | euclidean) | euclidean;
        let (_, directions) = (EuclideanVector::slot() | acceleration).eigh();
        let planes = [directions[0], directions[2], directions[4]].map(|u| {
            let turning = u ^ (u | euclidean);
            let open = NullBivector::slot();
            let conditions = open.commutator(motion).cast::<Conditions>()
                + (turning ^ open).cast::<Conditions>();
            let rhs = -turning.commutator(motion).cast::<Conditions>();
            let placement = conditions.lstsq(rhs);
            turning.cast::<p6::Bivector>() + placement.cast::<p6::Bivector>()
        });
        let leftover = motion - planes[0] - planes[1] - planes[2];
        (planes, leftover)
    }

    /// The roots of the cubic `x³ + c[0] x² + c[1] x + c[2]`, ascending; real here (the squares
    /// of the planes), from Viète's trigonometric form.
    pub fn cubic_roots(c: [f64; 3]) -> [f64; 3] {
        let [a, b, d] = c;
        let p = b - a * a / 3.0;
        let q = 2.0 * a * a * a / 27.0 - a * b / 3.0 + d;
        let m = 2.0 * (-p / 3.0).max(0.0).sqrt();
        let arg = if m == 0.0 {
            0.0
        } else {
            (3.0 * q / (p * m)).clamp(-1.0, 1.0)
        };
        let theta = arg.acos() / 3.0;
        let mut roots = [0, 1, 2]
            .map(|k| m * (theta - core::f64::consts::TAU * k as f64 / 3.0).cos() - a / 3.0);
        roots.sort_by(f64::total_cmp);
        roots
    }

    /// The example: a random bivector and unit start, the planes by both ways, the times of
    /// one turn of the slowest plane, and a random motion of the projective algebra.
    pub struct Example {
        pub bivector: B,
        pub start: V,
        pub squares: [f64; 3],
        pub parts: [B; 3],
        pub wedge: ([f64; 3], [B; 3]),
        pub period: f64,
        pub motion: p6::Bivector<(), f64>,
    }

    pub fn example(seed: u64) -> Example {
        let mut rng = rng(seed);
        let bivector = B::from_coeffs(core::array::from_fn(|_| rng.normal()));
        let start = V::from_coeffs(rng.direction());
        let (squares, parts) = from_spectrum(bivector);
        let wedge = from_wedge_powers(bivector);
        let slowest = squares.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let period = core::f64::consts::TAU / (-slowest).sqrt();
        let motion = p6::Bivector::from_coeffs(core::array::from_fn(|_| rng.normal()));
        Example {
            bivector,
            start,
            squares,
            parts,
            wedge,
            period,
            motion,
        }
    }
}

use decomposition::*;
use gax::ApproxEq;
use gax::motions::Linear;

/// The largest coefficient of a value: its difference from zero.
fn size<X: Linear<f64> + ApproxEq>(x: X) -> f64 {
    x.max_abs_diff(&X::zero())
}

const SECONDS: f32 = 12.0;
const SEED: u64 = 0;
const SAMPLES: usize = 600;

/// A point in one of the planes, in the plane's own frame.
type Flat = gax::pga2d::Point<(), f64>;
/// A point of the first three directions.
type Space = gax::pga3d::Point<(), f64>;

fn plane_colour(k: usize) -> Light {
    [palette::red(), palette::sky(), palette::green()][k]
}

/// The orbit at the sample times, and its projections into each plane as points of the plane's
/// own frame: along the direction of the first projected point, and a quarter turn on from it.
fn tracks(ex: &Example) -> (Vec<V>, [Vec<Flat>; 3]) {
    let points: Vec<V> = (0..SAMPLES)
        .map(|k| {
            let time = ex.period * k as f64 / (SAMPLES - 1) as f64;
            orbit(ex.squares, ex.parts, ex.start, time)
        })
        .collect();
    let first = projected(points[0], ex.parts);
    let flat = core::array::from_fn(|k| {
        let across = first[k].normalized().into_inner();
        let along = (across | ex.parts[k]).normalized().into_inner();
        points
            .iter()
            .map(|p| {
                let q = projected(*p, ex.parts)[k];
                Flat::xy((q | across).s(), (q | along).s())
            })
            .collect()
    });
    (points, flat)
}

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let screen = c.rect();
    let (w, h) = (screen.width(), screen.height());
    let down = Point2::direction(0.0, 1.0);
    let ex = example(SEED);
    let (points, flat) = tracks(&ex);
    let index = ((t / SECONDS * SAMPLES as f32) as usize).min(SAMPLES - 1);

    // The tangle in the first three directions.
    let left = Rect::new(0.0, 0.0, 0.4 * w, h);
    let cam = Camera::orbit(
        left.width() as usize,
        left.height() as usize,
        ORIGIN3,
        6.5,
        -0.9 + core::f32::consts::TAU * t / SECONDS,
        0.45,
        Lens::Perspective(0.42),
    );
    // The orbit's shadow in the first three directions.
    let path: Vec<Space> = points
        .iter()
        .map(|p| Space::xyz(p.c[0], p.c[1], p.c[2]))
        .collect();
    panel3(c, left, cam, |scene| {
        scene.polyline(&path, 0.8, palette::grid().faded(0.9));
        scene.polyline(&path[..=index], 1.6, palette::purple());
        scene.dot(path[index], Marker::Dot, 9.0, palette::purple());
    });
    let label = Point2::xy(0.2 * w, 0.88 * h);
    c.text("X, Y AND Z", label, 12.0, palette::ink(), Align::Center);

    // Each plane: a circle at the plane's own rate, in a square panel of its own.
    let side = 0.18 * w;
    let centre = Flat::xy(0.0, 0.0);
    for (k, track) in flat.iter().enumerate() {
        let lo = Point2::xy(0.42 * w + k as f32 * 0.19 * w, 0.25 * h);
        let rect = Rect {
            lo,
            hi: lo + Point2::direction(side, side),
        }
        .inset(4.0, 4.0, 4.0, 4.0);
        // The circle's radius, with a margin.
        let radius = track
            .iter()
            .fold(0.0f64, |m, p| m.max((centre & *p).norm()));
        let extent = (radius * 1.2) as f32;
        let ax = Axes::equal(rect, centre, extent);
        ax.polyline(c, track, 0.9, palette::grid());
        ax.polyline(c, &track[..=index], 1.8, plane_colour(k));
        ax.scatter(c, &track[index..=index], Marker::Dot, 9.0, plane_colour(k));
        // The plane's rate: the norm of its part, whose square is minus the rate's.
        let rate = ex.parts[k].norm();
        let name = format!("PLANE {}", k + 1);
        let above = Point2::xy(0.0, extent * 1.15);
        ax.text(c, above, &name, 12.0, palette::ink(), Align::Center);
        let tone = (plane_colour(k)).mix_light(palette::ink(), 0.4);
        let below = Point2::xy(0.0, -extent * 1.3);
        ax.text(
            c,
            below,
            &format!("RATE {rate:.2}"),
            10.0,
            tone,
            Align::Center,
        );
    }
    // The checks, live: both ways agree, the rotors make up the exponential, and the motion of
    // the projective algebra is three placed planes.
    let ways = (0..3)
        .map(|k| ex.wedge.1[k].max_abs_diff(&ex.parts[k]))
        .fold(0.0, f64::max);
    let exp = ex.bivector.gp(0.5).exp().into_inner();
    let product = rotor(ex.squares, ex.parts, 1.0).max_abs_diff(&exp);
    let (planes, leftover) = placed(ex.motion);
    let simple = planes.iter().map(|p| size(p.wedge(*p))).fold(0.0, f64::max);
    let lines = [
        format!("WEDGE POWERS AGAINST SPECTRUM: {ways:.0E}"),
        format!("PRODUCT OF PLANE ROTORS AGAINST EXP(B/2): {product:.0E}"),
        format!(
            "R(6,0,1) MOTION LESS ITS THREE PLACED PLANES: {:.0E}",
            size(leftover)
        ),
        format!("PLACED PLANES SIMPLE, P ^ P: {simple:.0E}"),
    ];
    let first = Point2::xy(0.43 * w, 0.78 * h);
    for (k, line) in lines.iter().enumerate() {
        let at = first + down.gp(k as f32 * 0.04 * h);
        c.text(line, at, 10.0, palette::grid(), Align::Left);
    }
    caption(
        c,
        "INVARIANT DECOMPOSITION: A 6D ROTATION IS THREE TURNS",
        "A RANDOM BIVECTOR OF R(6,0,0) SPLIT INTO THREE COMMUTING PLANES",
    );
}

fn main() {
    run(
        Anim::new("invariant decomposition", SECONDS).size(960, 540),
        draw,
    );
}

#[cfg(test)]
mod tests {
    use super::decomposition::*;
    use super::{p6, r6, size};
    use gax::ApproxEq;

    /// numga's checks in `main`, on the test's seed: the planes sum to the bivector, commute,
    /// and each squares to a scalar; the two ways give the same squares and the same planes.
    /// The product of the planes' rotors is gax's closed-form exponential of half the
    /// bivector, the point keeps its length, and in each plane it stays on a circle.
    #[test]
    fn the_planes_decompose_the_rotation() {
        let ex = example(1);
        let (squares, parts) = (ex.squares, ex.parts);
        let sum = parts[0] + parts[1] + parts[2];
        assert!(sum.max_abs_diff(&ex.bivector) < 1e-11, "{sum:?}");
        for a in &parts {
            for b in &parts {
                assert!(size(a.commutator(*b)) < 1e-11);
            }
            // A single plane squares to a scalar.
            let square = *a * *a;
            assert!(square.max_abs_diff(&square.grade::<0>().cast::<r6::Even>()) < 1e-10);
        }
        let (wedge_squares, wedge_parts) = ex.wedge;
        for k in 0..3 {
            assert!(
                (wedge_squares[k] - squares[k]).abs() < 1e-11,
                "{wedge_squares:?} {squares:?}"
            );
            assert!(wedge_parts[k].max_abs_diff(&parts[k]) < 1e-11);
        }
        let exp = ex.bivector.gp(0.5).exp().into_inner();
        assert!(rotor(squares, parts, 1.0).max_abs_diff(&exp) < 1e-6);
        let first = projected(ex.start, parts);
        for k in 0..60 {
            let time = ex.period * k as f64 / 59.0;
            let p = orbit(squares, parts, ex.start, time);
            assert!(((p | p).s() - 1.0).abs() < 1e-6);
            for (q, q0) in projected(p, parts).iter().zip(&first) {
                assert!(((*q | *q).s() - (*q0 | *q0).s()).abs() < 1e-6);
            }
        }
    }

    /// The inverse of a scalar plus a quadvector, as a cubic in it, is an inverse.
    #[test]
    fn the_cubic_is_an_inverse() {
        let b = example(2).bivector;
        let value = (b ^ b).gp(0.5).cast::<r6::ScalarQuadvector>()
            + r6::Scalar::<(), f64>::from_coeffs([0.7]).cast::<r6::ScalarQuadvector>();
        let one = (value * inverse(value)).cast::<r6::ScalarQuadvector>();
        let unit = r6::Scalar::<(), f64>::from_coeffs([1.0]).cast::<r6::ScalarQuadvector>();
        assert!(one.max_abs_diff(&unit) < 1e-12);
    }

    /// numga's checks on the motion: the placed planes are simple and commute with one
    /// another, and in six dimensions they add up to the whole motion.
    #[test]
    fn the_placed_planes_make_up_the_motion() {
        let ex = example(1);
        let (planes, leftover) = placed(ex.motion);
        for a in &planes {
            assert!(size(a.wedge(*a)) < 1e-11, "{:?}", a.wedge(*a));
            for c in &planes {
                assert!(size(a.commutator(*c)) < 1e-11);
            }
        }
        assert!(size(leftover) < 1e-10, "{leftover:?}");
        let _: p6::Bivector<(), f64> = leftover;
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws(super::draw, 0.5);
    }
}
