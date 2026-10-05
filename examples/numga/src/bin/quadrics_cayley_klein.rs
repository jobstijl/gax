//! numga's `quadrics/cayley_klein`: Cayley-Klein geometry in PGA2D, where the metric is a conic
//! you choose. Projective geometry has joins and meets but no distances; one conic, the
//! absolute, given as a polarity `C` from points to lines, supplies them all. Distances and
//! angles are invariants of pairs against the absolute (`C` for points, its inverse `Q` for
//! lines), read off the rotors between vectors of the algebra whose metric is the absolute
//! (`R(2,1)` for the hyperbolic plane); the perpendiculars to a line all pass through its pole,
//! reflection in a line is the harmonic homology about its pole, and a circle is the quadric of
//! points at a fixed invariant from its centre. With the unit circle as the absolute this is the
//! Beltrami-Klein disk of hyperbolic geometry; flipping one sign gives elliptic geometry with the
//! same lines.
//!
//! The animation moves the triangle's apex and the point P, so the angle sum, the area by
//! Gauss-Bonnet, the perpendicular through the pole and the reflection all follow, and grows
//! two families of circles about their centres.

use gax::pga2d::{Line, Point};
use gax_numga_examples::{
    Align, Anim, Axes, Canvas, Light, Marker, Point2, backdrop, caption, palette, run,
};

mod cayley_klein {
    use gax::pga2d::{Line, Point};
    use gax_numga_examples::measure;
    use gax_numga_examples::signal::phasor;

    pub type P = Point<(), f64>;
    pub type L = Line<(), f64>;
    /// A polarity, points to lines (`Line <- Point`).
    pub type Polarity = Line<(Point,), f64>;

    /// The absolute from three line dyads, `x (x ∨ P) + y (y ∨ P) ± w (w ∨ P)`: the conic
    /// `x² + y² ± w² = 0`. With `-` it is the unit circle (hyperbolic), with `+` a conic with no
    /// real points (elliptic).
    pub fn absolute(sign: f64) -> Polarity {
        let dyad = |l: L| l * (l & Point::slot());
        dyad(Line::new(1.0, 0.0, 0.0))
            + dyad(Line::new(0.0, 1.0, 0.0))
            + dyad(Line::new(0.0, 0.0, 1.0)).gp(sign)
    }

    /// The hyperbolic absolute, the unit circle.
    pub fn hyperbolic() -> Polarity {
        absolute(-1.0)
    }

    /// The elliptic absolute.
    pub fn elliptic() -> Polarity {
        absolute(1.0)
    }

    /// The hyperbolic absolute as a metric: the lines `a x + b y + c = 0` and the points
    /// `(x, y, w)` of PGA2D as the vectors `a e1 + b e2 + c e0` and `x e1 + y e2 + w e0` of
    /// `R(2,1)`, whose inner product is the absolute's pairing, `l ∨ Q(m)` for lines and
    /// `A ∨ C(B)` for points. Lines through the disk are spacelike, and the rotor between two of
    /// them turns by their angle; points inside are timelike, and the rotor between two of them
    /// is a boost by their distance.
    pub mod metric {
        gax::algebra! {
            algebra r21 "The hyperbolic plane's absolute as a metric, R(2,1).";
            basis e1 = 1, e2 = 1, e0 = -1;
            kind Scalar = [1];
            versor Vector = [e1, e2, e0];
            kind Bivector = [e12, e20, e01];
            versor Rotor = [1, e12, e20, e01];
            kind Pseudoscalar = [e120];
        }
        pub use r21::{Bivector, Vector};
    }

    /// A Cayley-Klein geometry: its absolute, and the angles and distances it measures, each the
    /// norm of the logarithm of the rotor between two vectors of the algebra whose metric is the
    /// absolute.
    pub trait Geometry {
        /// The absolute.
        fn absolute() -> Polarity;
        /// The angle between two lines, from 0 to π.
        fn angle(l: L, m: L) -> f64;
        /// The distance between two points.
        fn distance(a: P, b: P) -> f64;
    }

    /// The hyperbolic plane: the unit circle as the absolute.
    pub struct Hyperbolic;

    /// The elliptic plane: the absolute with no real points.
    pub struct Elliptic;

    impl Geometry for Hyperbolic {
        fn absolute() -> Polarity {
            hyperbolic()
        }

        fn angle(l: L, m: L) -> f64 {
            let line = |l: L| metric::Vector::<(), f64>::new(l.e1(), l.e2(), l.e0());
            let log: metric::Bivector<(), f64> = (line(m) * line(l)).normalized().log();
            log.norm()
        }

        /// Inside the disk `A ∨ C(B)` is negative, `-cosh` of the distance, so the boost from
        /// `A` to `B` is the rotor `-B A`.
        fn distance(a: P, b: P) -> f64 {
            let point = |p: P| metric::Vector::<(), f64>::new(p.e20(), p.e01(), p.e12());
            let log: metric::Bivector<(), f64> = (-(point(b) * point(a))).normalized().log();
            log.norm()
        }
    }

    /// The elliptic absolute is the Euclidean metric of the lines' and the points'
    /// coefficients: angles and distances are angles between vectors of space.
    impl Geometry for Elliptic {
        fn absolute() -> Polarity {
            elliptic()
        }

        fn angle(l: L, m: L) -> f64 {
            let line = |l: L| gax::vga3d::Vector::<(), f64>::new(l.e1(), l.e2(), l.e0());
            measure::angle(line(l), line(m))
        }

        fn distance(a: P, b: P) -> f64 {
            let point = |p: P| gax::vga3d::Vector::<(), f64>::new(p.e20(), p.e01(), p.e12());
            measure::angle(point(a), point(b))
        }
    }

    /// A triangle against the absolute.
    pub struct Triangle {
        /// The side from each vertex to the next.
        pub sides: [L; 3],
        /// The angle at each vertex.
        pub angles: [f64; 3],
        /// The side lengths.
        pub lengths: [f64; 3],
        /// The area, by Gauss-Bonnet: the angle defect.
        pub area: f64,
    }

    /// The sides, angles, side lengths and area of a triangle, all from the absolute.
    pub fn triangle<G: Geometry>(v: [P; 3]) -> Triangle {
        // The sides are joins of consecutive vertices; the angle at a vertex is between the
        // two sides leaving it.
        let to_next: [L; 3] = core::array::from_fn(|i| v[i] & v[(i + 1) % 3]);
        let to_prev: [L; 3] = core::array::from_fn(|i| v[i] & v[(i + 2) % 3]);
        let angles = core::array::from_fn(|i| G::angle(to_next[i], to_prev[i]));
        let lengths = core::array::from_fn(|i| G::distance(v[i], v[(i + 1) % 3]));
        let area = core::f64::consts::PI - angles.iter().sum::<f64>();
        Triangle {
            sides: to_next,
            angles,
            lengths,
            area,
        }
    }

    /// The pole of a line, the perpendicular to it from `p`, its foot, and `p` reflected in
    /// the line. Every perpendicular to `l` passes through its pole, so the perpendicular from
    /// `p` is the join `p ∨ pole` and the foot its meet with `l`. Reflection in `l` is the
    /// harmonic homology about the pole, and it keeps the distance to the foot.
    pub fn perpendicular(c: Polarity, side: L, p: P) -> (P, L, P, P) {
        let pole: P = c.solve(side);
        let normal = p & pole;
        let foot = side ^ normal;
        let reflected = p - pole * (2.0 * (p & side).s() / (pole & side).s());
        (pole, normal, foot, reflected)
    }

    /// The circle of radius `r` about `centre`, as a quadric: the dyad of the centre's polar
    /// less `cosh(r)²` times the centre's self-pairing times the absolute. It is the squared
    /// invariant equation `(P ∨ C(c))² = cosh(r)² (c ∨ C(c)) (P ∨ C(P))` with the root cleared.
    pub fn circle(c: Polarity, centre: P, r: f64) -> Polarity {
        let polar = c.of(centre);
        polar * (polar & Point::slot()) - c.gp((centre & polar).s() * r.cosh().powi(2))
    }

    /// The scene at phase `t` (radians).
    pub struct Scene {
        pub c: Polarity,
        pub vertices: [P; 3],
        pub triangle: Triangle,
        pub p: P,
        pub pole: P,
        pub normal: L,
        pub foot: P,
        pub reflected: P,
        pub centres: [P; 2],
        pub radii: [f64; 4],
    }

    /// numga's scene with the apex wandering, P circling and the circles' radii growing.
    pub fn hyperbolic_plane(t: f64) -> Scene {
        let c = Hyperbolic::absolute();
        // The apex and P go round on turned directions, stretched into small loops.
        let (once, twice) = (phasor(t), phasor(2.0 * t));
        let vertices = [
            Point::xy(0.0, 0.0),
            Point::xy(0.65, 0.0),
            Point::xy(0.2 + 0.25 * once.e01(), 0.55 + 0.2 * twice.e20()),
        ];
        let triangle = triangle::<Hyperbolic>(vertices);
        let p = Point::xy(-0.15 + 0.2 * once.e20(), 0.15 + 0.15 * once.e01());
        let (pole, normal, foot, reflected) = perpendicular(c, triangle.sides[1], p);
        let grow = (t / core::f64::consts::TAU).rem_euclid(1.0);
        Scene {
            c,
            vertices,
            triangle,
            p,
            pole,
            normal,
            foot,
            reflected,
            centres: [Point::xy(0.0, 0.0), Point::xy(0.45, 0.25)],
            radii: [0.25, 0.55, 0.9, 1.3].map(|r| r + 0.4 * grow),
        }
    }
}

use cayley_klein::*;

/// A line, drawn through its point nearest the origin (its meet with the perpendicular from
/// the origin) along its direction (its meet with the line at infinity).
fn line(ax: &Axes, c: &mut Canvas, l: L, width: f32, color: Light) {
    let foot = l ^ (l | Point::xy(0.0, 0.0));
    ax.axline(c, foot, l ^ Line::new(0.0, 0.0, 1.0), width, color);
}

/// The locus `P ∨ quadric(P) = 0`.
fn level_set(ax: &Axes, c: &mut Canvas, quadric: Polarity, width: f32, color: Light) {
    ax.contour(
        c,
        |p| {
            let p: P = p.map_coefs(f64::from);
            (p & quadric.of(p)).s() as f32
        },
        260,
        0.0,
        width,
        color,
    );
}

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let phase = f64::from(t) / 12.0 * core::f64::consts::TAU;
    let s = hyperbolic_plane(phase);
    let screen = c.rect();
    let rect = |i: usize| screen.column(i, 2).inset(20.0, 80.0, 20.0, 24.0);
    // The notes go under the panels, centred.
    let below = Point2::direction(0.0, 16.0);
    let box_centre = Point::xy(0.35, 0.2);

    // The triangle, its perpendicular and the reflection, on the left.
    let ax = Axes::equal(rect(0), box_centre, 1.4);
    ax.clip(c);
    level_set(&ax, c, s.c, 2.0, palette::ink());
    for side in s.triangle.sides {
        line(&ax, c, side, 1.6, palette::sky());
    }
    line(&ax, c, s.normal, 1.4, palette::red());
    ax.fill(c, &s.vertices, palette::sky(), 0.18);
    ax.scatter(c, &s.vertices, Marker::Dot, 8.0, palette::sky());
    ax.scatter(c, &[s.p], Marker::Dot, 9.0, palette::red());
    ax.scatter(c, &[s.foot], Marker::Square, 8.0, palette::red());
    ax.scatter(c, &[s.reflected], Marker::Dot, 9.0, palette::orange());
    ax.scatter(c, &[s.pole], Marker::Triangle, 10.0, palette::purple());
    // The angles beside their vertices, the lengths beside the midpoints of the sides.
    for (v, angle) in s.vertices.iter().zip(s.triangle.angles) {
        ax.text(
            c,
            *v + Point::direction(0.04, 0.06),
            &format!("{:.1}", angle.to_degrees()),
            11.0,
            palette::ink(),
            Align::Left,
        );
    }
    for i in 0..3 {
        let midpoint = (s.vertices[i] + s.vertices[(i + 1) % 3]).gp(0.5);
        ax.text(
            c,
            midpoint + Point::direction(0.03, -0.07),
            &format!("{:.2}", s.triangle.lengths[i]),
            10.0,
            palette::sky(),
            Align::Left,
        );
    }
    c.unclip();
    ax.legend(
        c,
        &[
            ("P", palette::red()),
            ("REFLECTION", palette::orange()),
            ("POLE OF BC", palette::purple()),
        ],
    );
    let r = rect(0);
    c.text(
        &format!(
            "ANGLE SUM {:.1} (ELLIPTIC {:.1}) AREA {:.3}",
            s.triangle.angles.iter().sum::<f64>().to_degrees(),
            triangle::<Elliptic>(s.vertices)
                .angles
                .iter()
                .sum::<f64>()
                .to_degrees(),
            s.triangle.area
        ),
        r.bottom_middle() + below,
        10.0,
        palette::ink(),
        Align::Center,
    );

    // Circles as level sets of quadrics, on the right.
    let ax = Axes::equal(rect(1), box_centre, 1.4);
    ax.clip(c);
    level_set(&ax, c, s.c, 2.0, palette::ink());
    for (centre, colour) in s.centres.iter().zip([palette::sky(), palette::orange()]) {
        for r in s.radii {
            level_set(&ax, c, circle(s.c, *centre, r), 1.4, colour);
        }
    }
    ax.scatter(c, &s.centres, Marker::Dot, 7.0, palette::ink());
    c.unclip();
    let r = rect(1);
    c.text(
        &format!(
            "RADII {:.2} TO {:.2}: P & CIRCLE(P) = 0",
            s.radii[0], s.radii[3]
        ),
        r.bottom_middle() + below,
        10.0,
        palette::ink(),
        Align::Center,
    );
    caption(
        c,
        "CAYLEY-KLEIN: THE HYPERBOLIC PLANE FROM ITS ABSOLUTE",
        "C = X(X & P) + Y(Y & P) - W(W & P): DISTANCES, ANGLES, CIRCLES (PGA2D)",
    );
}

fn main() {
    run(Anim::new("cayley klein", 12.0).size(960, 540), draw);
}

#[cfg(test)]
mod tests {
    use super::cayley_klein::*;
    use gax::ApproxEq;
    use gax::pga2d::{Line, Point};

    /// A pole map, lines to points (`Point <- Line`).
    type Pole = Point<(Line,), f64>;

    fn assert_close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol, "{a} vs {b}");
    }

    /// The invariant of two points against an absolute `C`: `A ∨ C(B)` over the roots.
    #[allow(clippy::disallowed_methods)] // the reference it is checked against
    fn point_invariant(c: Polarity, a: P, b: P) -> f64 {
        let pair = |a: P, b: P| (a & c.of(b)).s();
        pair(a, b) / (pair(a, a) * pair(b, b)).sqrt()
    }

    /// The invariant of two lines against the absolute's inverse `Q`: `l ∨ Q(m)` over the roots.
    #[allow(clippy::disallowed_methods)] // the reference it is checked against
    fn line_invariant(c: Polarity, l: L, m: L) -> f64 {
        let q: Pole = c.inverse();
        let pair = |l: L, m: L| (l & q.of(m)).s();
        pair(l, m) / (pair(l, l) * pair(m, m)).sqrt()
    }

    /// The angles and lengths from the rotors of the absolute's algebra are the arccosines (and
    /// for hyperbolic lengths the inverse hyperbolic cosines) of the invariants.
    #[test]
    #[allow(clippy::disallowed_methods)] // the references they are checked against
    fn rotor_measures_are_the_invariants_measures() {
        for k in 0..16 {
            let s = hyperbolic_plane(0.4 * k as f64);
            let v = s.vertices;
            let sides: [L; 3] = core::array::from_fn(|i| v[i] & v[(i + 1) % 3]);
            let back: [L; 3] = core::array::from_fn(|i| v[i] & v[(i + 2) % 3]);
            for (c, t) in [
                (hyperbolic(), triangle::<Hyperbolic>(v)),
                (elliptic(), triangle::<Elliptic>(v)),
            ] {
                for i in 0..3 {
                    let angle = line_invariant(c, sides[i], back[i]).clamp(-1.0, 1.0).acos();
                    assert_close(t.angles[i], angle, 1e-9);
                }
            }
            for i in 0..3 {
                let (a, b) = (v[i], v[(i + 1) % 3]);
                let h = (-point_invariant(hyperbolic(), a, b)).max(1.0).acosh();
                assert_close(Hyperbolic::distance(a, b), h, 1e-9);
                let e = point_invariant(elliptic(), a, b).clamp(-1.0, 1.0).acos();
                assert_close(Elliptic::distance(a, b), e, 1e-9);
            }
        }
    }

    #[test]
    fn absolute_from_line_dyads_is_the_diagonal_polarity() {
        let diag = |d: [f64; 3]| -> [[f64; 3]; 3] {
            core::array::from_fn(|i| core::array::from_fn(|j| if i == j { d[i] } else { 0.0 }))
        };
        assert_eq!(hyperbolic().c, diag([1.0, 1.0, -1.0]));
        let q: Pole = hyperbolic().inverse();
        assert!(q.max_abs_diff(&Pole::from_coeffs(diag([1.0, 1.0, -1.0]))) < 1e-14);
        assert_eq!(elliptic().c, diag([1.0, 1.0, 1.0]));
    }

    #[test]
    fn hyperbolic_distance_along_a_diameter_is_arctanh() {
        let origin = Point::xy(0.0, 0.0);
        for x in [0.2, 0.5, 0.75] {
            let d = Hyperbolic::distance(origin, Point::xy(x, 0.0));
            assert_close(d, f64::atanh(x), 1e-12);
        }
    }

    /// On the gnomonic chart of the sphere, the distance from the origin is the arctangent of
    /// the radius.
    #[test]
    fn elliptic_distance_along_a_diameter_is_arctan() {
        let origin = Point::xy(0.0, 0.0);
        for x in [0.2, 0.5, 2.0] {
            let d = Elliptic::distance(origin, Point::xy(x, 0.0));
            assert_close(d, x.atan(), 1e-12);
        }
    }

    #[test]
    fn perpendicular_through_the_pole_and_reflection_in_both_geometries() {
        for c in [hyperbolic(), elliptic()] {
            let q: Pole = c.inverse();
            let side = Point::xy(0.65, 0.0) & Point::xy(0.2, 0.55);
            let p = Point::xy(-0.15, 0.15);
            let pole = q.of(side);
            let perpendicular = p & pole;
            let foot = side ^ perpendicular;
            let reflected = p - pole * (2.0 * (p & side).s() / (pole & side).s());
            assert!((side & q.of(perpendicular)).s().abs() < 1e-14);
            assert!((foot & side).s().abs() < 1e-14);
            assert_close(
                point_invariant(c, p, foot),
                point_invariant(c, reflected, foot),
                1e-12,
            );
            assert!((reflected & perpendicular).s().abs() < 1e-14);
            // The scenario's construction, through `C.solve`, is the same.
            let (pole2, normal, foot2, reflected2) = super::cayley_klein::perpendicular(c, side, p);
            for (a, b) in [(pole, pole2), (foot, foot2), (reflected, reflected2)] {
                assert!(a.max_abs_diff(&b) < 1e-12);
            }
            assert!((side & q.of(normal)).s().abs() < 1e-14);
        }
    }

    #[test]
    fn triangle_angle_sum_is_below_pi_hyperbolic_and_above_pi_elliptic() {
        let vertices = [
            Point::xy(0.0, 0.0),
            Point::xy(0.65, 0.0),
            Point::xy(0.2, 0.55),
        ];
        let hyperbolic = triangle::<Hyperbolic>(vertices).angles.iter().sum::<f64>();
        let elliptic = triangle::<Elliptic>(vertices).angles.iter().sum::<f64>();
        assert!(hyperbolic < core::f64::consts::PI);
        assert!(core::f64::consts::PI < elliptic);
    }

    #[test]
    fn circle_quadric_contains_the_points_at_its_radius() {
        let c = hyperbolic();
        let (centre, r) = (Point::xy(0.45, 0.25), 0.9);
        let ring = circle(c, centre, r);
        let (mut lo, mut hi) = (0.45, 0.999);
        for _ in 0..60 {
            let mid = 0.5 * (lo + hi);
            let d = Hyperbolic::distance(centre, Point::xy(mid, 0.25));
            if d < r { lo = mid } else { hi = mid }
        }
        let on = Point::xy(lo, 0.25);
        assert!((on & ring.of(on)).s().abs() < 1e-12);
    }

    /// The scenario's checks, at every phase: a positive angle defect, the perpendicular
    /// through the pole, and a reflection that keeps the distance to the foot.
    #[test]
    fn scenario_checks() {
        for k in 0..16 {
            let s = hyperbolic_plane(0.4 * k as f64);
            assert!(s.triangle.area > 0.0);
            let pole_of_normal: P = s.c.solve(s.normal);
            assert!((s.triangle.sides[1] & pole_of_normal).s().abs() < 1e-12);
            assert_close(
                Hyperbolic::distance(s.p, s.foot),
                Hyperbolic::distance(s.reflected, s.foot),
                1e-12,
            );
        }
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws(super::draw, 0.5);
    }
}
