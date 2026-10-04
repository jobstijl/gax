//! numga's `quadrics/quadric_collision`: two ellipses colliding, read off the blend of their dual
//! quadrics in PGA2D. A dual ellipse `Q` maps lines to points: a line `L` is tangent where
//! `L ∨ Q(L) = 0`, and `Q(L)` is then its point of contact. The blend `Q1 (1 - λ) + Q2 λ` keeps
//! the common tangents of the two, and its determinant is a cubic in `λ` that peaks between 0 and
//! 1: above zero the ellipses are apart, at zero they touch, below zero they overlap. At contact
//! the blend at the peak is singular, and its null line is the shared tangent. The animation
//! slides the second ellipse along the contact normal, in and out of the first; the right hand
//! side shows the cubic of the current pose and the peak as a function of the offset.

use gax::pga2d::{Line, Motor, Point};

use gax_numga_examples::measure;
use gax_numga_examples::signal::phasor;
use gax_numga_examples::{
    Align, Anim, Axes, Canvas, Light, Marker, Point2, Rect, backdrop, caption, palette, run,
};

mod collision {
    use super::*;

    pub type L = Line<(), f64>;
    pub type P = Point<(), f64>;
    /// A dual quadric: a map from lines to points.
    pub type Quadric = Point<(Line,), f64>;
    /// A primal quadric: a map from points to their polar lines.
    pub type Polarity = Line<(Point,), f64>;
    pub type M = gax::Unit<Motor<(), f64>>;

    /// The line at infinity.
    pub fn infinity() -> L {
        Line::new(0.0, 0.0, 1.0)
    }

    /// The origin.
    pub fn origin() -> P {
        Point::xy(0.0, 0.0)
    }

    /// The dual ellipse with semi-axes `rx` and `ry` along x and y, centred at the origin: its
    /// principal directions, the ideal points along x and y, each paired with itself and weighted
    /// by its squared semi-axis, less the origin paired with itself.
    pub fn ellipse(rx: f64, ry: f64) -> Quadric {
        let l = Line::slot();
        let (along_x, along_y) = (Point::direction(1.0, 0.0), Point::direction(0.0, 1.0));
        (l & along_x) * along_x.gp(rx * rx) + (l & along_y) * along_y.gp(ry * ry)
            - (l & origin()) * origin()
    }

    /// The rigid motion rotating by `angle` about the origin, then translating by `(tx, ty)`.
    pub fn motor(tx: f64, ty: f64, angle: f64) -> M {
        Motor::translation(tx, ty) * Motor::rotation(origin(), angle)
    }

    /// A quadric moved by a motor: its input pulled back, its output pushed forward.
    pub fn moved(m: M, q: Quadric) -> Quadric {
        m >> q.of(m << Line::slot())
    }

    /// The tangent line of `q` with outward normal direction `normal`, whatever the scale or sign
    /// of `q`. The lines with normal `n` are `n - ∞ offset`; tangency, `L ∨ Q(L) = 0`, is quadratic
    /// in the offset, `a offset² - 2 b offset + c = 0` for `a, b, c = ∞ ∨ Q(∞), n ∨ Q(∞), n ∨ Q(n)`.
    /// Its two roots lie either side of the centre's offset `b / a`, the outward one beyond it
    /// along `n`. Rescaling `q` rescales all three alike, so neither root moves.
    pub fn tangent_line(q: Quadric, normal: L) -> L {
        let n = normal.normalized().into_inner();
        let inf = infinity();
        let (a, b, c) = (
            (inf & q.of(inf)).s(),
            (n & q.of(inf)).s(),
            (n & q.of(n)).s(),
        );
        // The outward root of the quadratic in the offset ([`measure::roots`]), the support
        // distance: the midpoint `b / a` plus half the roots' distance.
        let support = measure::roots(a, -b, c).map_or(f64::NAN, |(middle, half)| middle + half);
        n - inf.gp(support)
    }

    /// The blend `q1 (1 - λ) + q2 λ` of two dual quadrics.
    pub fn blend(q1: Quadric, q2: Quadric, lambda: f64) -> Quadric {
        q1.gp(1.0 - lambda) + q2.gp(lambda)
    }

    /// The determinant of a blend, a cubic in `λ`: the determinant of its dual, a map from lines
    /// to lines.
    pub fn det(q1: Quadric, q2: Quadric, lambda: f64) -> f64 {
        blend(q1, q2, lambda).dual().det()
    }

    /// The maximum of a cubic sampled at 0, 1, 2 and -1: `(where, value)`. The cubic is concave
    /// between 0 and 1, so its maximum is the root of the derivative `3 c3 λ² + 2 c2 λ + c1`
    /// where the second derivative is negative. In the reciprocal `μ = 1 / λ` the derivative
    /// reads `c1 μ² + 2 c2 μ + 3 c3`, and that root is `μ = (√(c2² - 3 c3 c1) - c2) / c1`, the
    /// midpoint `-c2 / c1` plus half the roots' distance along `c1` ([`measure::roots`]): it
    /// stays finite as `c3` vanishes.
    pub fn cubic_peak([y0, y1, y2, y3]: [f64; 4]) -> (f64, f64) {
        let c3 = (3.0 * y0 - 3.0 * y1 + y2 - y3) / 6.0;
        let c2 = -y0 + 0.5 * y1 + 0.5 * y3;
        let c1 = -0.5 * y0 + y1 - y2 / 6.0 - y3 / 3.0;
        let c0 = y0;
        let peak = measure::roots(c1, c2, 3.0 * c3).map_or(f64::NAN, |(middle, half)| {
            1.0 / (middle + half.copysign(c1))
        });
        (peak, ((c3 * peak + c2) * peak + c1) * peak + c0)
    }

    /// Four determinant samples fix the cubic exactly; only locating its peak leaves the algebra.
    pub fn peak(q1: Quadric, q2: Quadric) -> (f64, f64) {
        cubic_peak([0.0, 1.0, 2.0, -1.0].map(|l| det(q1, q2, l)))
    }

    /// The scene: the first ellipse, the second one's shape and the motor that makes it touch
    /// the first along `normal`.
    pub struct Scene {
        pub q1: Quadric,
        pub shape: Quadric,
        pub touch: M,
        pub normal: L,
    }

    /// The first ellipse, semi-axes 2 and 1, turned 25° and centred at (-1.2, 0); the second,
    /// semi-axes 1.6 and 0.9 turned -35°, translated so that its tangent point with the
    /// opposite normal lands on the first's tangent point with a 22° normal.
    pub fn scene() -> Scene {
        let q1 = moved(motor(-1.2, 0.0, 25f64.to_radians()), ellipse(2.0, 1.0));
        let normal = Motor::rotation(origin(), 22f64.to_radians()) >> Line::new(1.0, 0.0, 0.0);
        let shape = ellipse(1.6, 0.9);
        let turn = motor(0.0, 0.0, (-35f64).to_radians());
        let turned = moved(turn, shape);
        let target = q1.of(tangent_line(q1, normal));
        let start = turned.of(tangent_line(turned, -normal));
        // The translation between the two contact points, at unit weight.
        let touch = Motor::between(start, target) * turn;
        Scene {
            q1,
            shape,
            touch,
            normal,
        }
    }

    impl Scene {
        /// The second ellipse moved `offset` along the normal from touching: apart when
        /// positive, overlapping when negative.
        pub fn second(&self, offset: f64) -> Quadric {
            let n = self.normal;
            moved(
                Motor::translation(n.e1() * offset, n.e2() * offset) * self.touch,
                self.shape,
            )
        }
    }

    /// The pose at contact: the parameter of the peak, the shared tangent (the blend's null
    /// line, its last singular vector) and the contact point.
    pub fn contact(q1: Quadric, q2: Quadric) -> (f64, L, P) {
        let (lambda, _) = peak(q1, q2);
        let (_, _, lines) = blend(q1, q2, lambda).dual().svd();
        let line = lines[2].normalized().into_inner();
        (lambda, line, q1.of(line).normalized().into_inner())
    }

    /// The blend at the peak of a pose that is apart, as a primal quadric: a hyperbola
    /// separating the two ellipses.
    pub fn separating(q1: Quadric, q2: Quadric) -> Polarity {
        blend(q1, q2, peak(q1, q2).0).inverse()
    }
}

use collision::*;

/// The outline of an ellipse: its contact points over the tangent normals, the line `x = 0`
/// turned about the origin.
fn outline(q: Quadric) -> Vec<P> {
    (0..150)
        .map(|k| {
            let turn = Motor::rotation(origin(), core::f64::consts::TAU * k as f64 / 150.0);
            q.of(tangent_line(q, turn >> Line::new(1.0, 0.0, 0.0)))
        })
        .collect()
}

/// An infinite line, drawn through its point nearest the origin (its meet with the
/// perpendicular from the origin) along its direction (its meet with the line at infinity).
fn draw_line(ax: &Axes, c: &mut Canvas, l: L, width: f32, color: Light) {
    let foot = l ^ (l | origin());
    ax.axline(c, foot, l ^ infinity(), width, color);
}

/// An ellipse filled (a faint cover of its colour), outlined, and its centre `Q(infinity)`
/// marked.
fn ellipse_fill(ax: &Axes, c: &mut Canvas, q: Quadric, color: Light) {
    let pts = outline(q);
    ax.fill(c, &pts, color.faded(0.3), 0.35);
    ax.polyline(c, &[pts.clone(), vec![pts[0]]].concat(), 2.2, color);
    ax.scatter(c, &[q.of(infinity())], Marker::Dot, 6.0, color);
}

/// The offset of the second ellipse at time `t`: from 0.8 apart to 0.6 inside and back.
fn offset_at(t: f32) -> f64 {
    let phase = f64::from(t) / 8.0 * core::f64::consts::TAU;
    0.1 + 0.7 * phasor(phase).e20()
}

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let screen = c.rect();
    let (w, h) = (screen.width(), screen.height());
    // Lengths in pixels at 960 by 540, scaled with the canvas.
    let unit = c.unit();
    let s = scene();
    let offset = offset_at(t);
    let q1 = s.q1;
    let q2 = s.second(offset);
    let (lambda, top) = peak(q1, q2);
    let state = if top > 1e-3 {
        "SEPARATED: MAX DET > 0"
    } else if top < -1e-3 {
        "OVERLAPPING: MAX DET < 0"
    } else {
        "TOUCHING: MAX DET = 0"
    };
    caption(
        c,
        "QUADRIC COLLISION: THE DETERMINANT OF A BLEND",
        "DUAL ELLIPSES IN PGA2D, Q(L) = Q1 (1-L) + Q2 L",
    );

    // The ellipses, on the left; the lowest tick a little above the corner.
    let ax = Axes::equal(
        Rect::new(0.0, 64.0 * unit, w * 0.58, h).inset(
            34.0 * unit,
            30.0 * unit,
            10.0 * unit,
            24.0 * unit,
        ),
        Point::xy(-0.4, 0.0),
        3.3,
    );
    ax.frame(c, state, "", "");
    if top > 0.0 {
        // The blend at the peak separates them: the hyperbola where its form vanishes.
        let field = separating(q1, q2);
        ax.contour(
            c,
            |p| {
                let p: P = p.map_coefs(f64::from);
                (p & field.of(p)).s() as f32
            },
            160,
            0.0,
            1.5,
            palette::green(),
        );
    } else {
        // The blend at the peak bridges the overlap.
        let pts = outline(blend(q1, q2, lambda));
        ax.fill(c, &pts, palette::purple().faded(0.3), 0.25);
    }
    ellipse_fill(&ax, c, q1, palette::sky());
    ellipse_fill(&ax, c, q2, palette::orange());
    // The tangent lines along the normal and their midline, and the closest (or deepest) points.
    let first = tangent_line(q1, s.normal);
    let second = tangent_line(q2, -s.normal);
    let mid = (first - second).gp(0.5);
    draw_line(&ax, c, first, 1.0, palette::sky().faded(0.7));
    draw_line(&ax, c, second, 1.0, palette::orange().faded(0.7));
    let mid_colour = if top > 0.0 {
        palette::green()
    } else {
        palette::purple()
    };
    draw_line(&ax, c, mid, 2.0, mid_colour);
    let (p1, p2) = (q1.of(first), q2.of(second));
    ax.dashed(c, &[p1, p2], 1.5, 4.0, palette::ink());
    ax.scatter(c, &[p1, p2], Marker::Dot, 7.0, palette::ink());
    if top.abs() < 0.08 {
        // Near contact: the blend's null line and the contact point.
        let (_, line, point) = contact(q1, q2);
        draw_line(&ax, c, line, 2.2, palette::red());
        ax.scatter(c, &[point], Marker::Star, 13.0, palette::yellow());
    }

    // The cubic of the current pose, against the three reference poses.
    // The right side, split into an upper and a lower half.
    let right = Rect::new(w * 0.6, 60.0 * unit, w - 16.0 * unit, h - 10.0 * unit);
    let half = Point2::direction(0.0, right.height() / 2.0);
    let top_rect = Rect {
        lo: right.lo,
        hi: right.hi - half,
    }
    .inset(40.0 * unit, 20.0 * unit, 0.0, 30.0 * unit);
    // The lowest tick a little above the corner, so its label clears the axis's.
    let ax = Axes::new(top_rect, [0.0, 1.0], [-3.5, 3.0]);
    ax.frame(c, "DET Q(L) ALONG THE BLEND", "L", "");
    ax.line(
        c,
        Point2::xy(0.0, 0.0),
        Point2::xy(1.0, 0.0),
        1.0,
        palette::grid(),
    );
    let curve = |q2: Quadric| -> Vec<Point2> {
        (0..=100)
            .map(|k| {
                let l = 0.001 + 0.998 * k as f64 / 100.0;
                Point2::xy(l as f32, det(q1, q2, l) as f32)
            })
            .collect()
    };
    for (k, o) in [0.8, 0.0, -0.6].into_iter().enumerate() {
        let colour = [palette::green(), palette::orange(), palette::red()][k];
        ax.polyline(c, &curve(s.second(o)), 1.0, colour.faded(0.35));
    }
    ax.polyline(c, &curve(q2), 2.4, palette::ink());
    ax.scatter(
        c,
        &[Point2::xy(lambda as f32, top as f32)],
        Marker::Dot,
        8.0,
        palette::yellow(),
    );

    // The peak against the offset: it crosses zero where they touch.
    let bottom_rect = Rect {
        lo: right.lo + half,
        hi: right.hi,
    }
    .inset(40.0 * unit, 20.0 * unit, 0.0, 30.0 * unit);
    let ax = Axes::new(bottom_rect, [-0.6, 0.8], [-1.5, 2.5]);
    ax.frame(c, "MAX DET AGAINST THE NORMAL OFFSET", "", "");
    ax.line(
        c,
        Point2::xy(-0.6, 0.0),
        Point2::xy(0.8, 0.0),
        1.0,
        palette::grid(),
    );
    let sweep: Vec<Point2> = (0..=70)
        .map(|k| {
            let o = -0.6 + 1.4 * k as f64 / 70.0;
            Point2::xy(o as f32, peak(q1, s.second(o)).1 as f32)
        })
        .collect();
    ax.polyline(c, &sweep, 2.0, palette::sky());
    let marker = if top > 0.0 {
        palette::green()
    } else {
        palette::red()
    };
    ax.scatter(
        c,
        &[Point2::xy(offset as f32, top as f32)],
        Marker::Dot,
        9.0,
        marker,
    );
    ax.text(
        c,
        Point2::xy(-0.55, 2.2),
        &format!("MAX = {top:+.3}"),
        11.0 * unit,
        palette::ink().mix_light(marker, 0.3),
        Align::Left,
    );
}

fn main() {
    run(Anim::new("quadric collision", 8.0).size(960, 540), draw);
}

#[cfg(test)]
mod tests {
    use super::collision::*;
    use gax::ApproxEq;
    use gax::pga2d::{Line, Point};

    fn placed() -> Quadric {
        moved(motor(1.5, -0.8, 35f64.to_radians()), ellipse(2.0, 1.2))
    }

    fn normals() -> Vec<L> {
        [0.0f64, 45.0, 90.0, 135.0, 210.0, 315.0]
            .iter()
            .map(|d| motor(0.0, 0.0, d.to_radians()) >> Line::new(1.0, 0.0, 0.0))
            .collect()
    }

    /// The motor rotates counterclockwise and then translates, as numga's.
    #[test]
    #[allow(clippy::disallowed_methods)] // the references it is checked against
    fn motor_conventions() {
        let m = motor(1.0, 2.0, 0.3);
        // The origin lands on (1, 2): the line joining them has no length.
        assert!(((m >> origin()) & Point::xy(1.0, 2.0)).norm() < 1e-14);
        let d = motor(0.0, 0.0, 0.3) >> Point::direction(1.0, 0.0);
        assert!((d.e20() - 0.3f64.cos()).abs() < 1e-14 && (d.e01() - 0.3f64.sin()).abs() < 1e-14);
    }

    /// Tangent lines from the support formula satisfy `L ∨ Q(L) = 0` for every normal.
    #[test]
    fn dual_ellipse_tangency() {
        let q = placed();
        for n in normals() {
            let l = tangent_line(q, n);
            assert!((l & q.of(l)).s().abs() < 1e-10);
        }
    }

    /// `Q` and any nonzero multiple of it, negative included, are the same conic with the same
    /// tangents.
    #[test]
    fn tangent_line_ignores_the_scale_of_the_quadric() {
        let q = placed();
        for n in normals() {
            let (a, b) = (tangent_line(q.gp(-2.5), n), tangent_line(q, n));
            assert!(a.max_abs_diff(&b) < 1e-12);
        }
    }

    /// The closed-form peak of the blend's determinant agrees with a sampled maximum.
    #[test]
    fn cubic_peak_matches_a_dense_sweep() {
        let q1 = ellipse(2.0, 1.0);
        let q2 = moved(motor(2.5, 0.7, (-20f64).to_radians()), ellipse(1.0, 0.5));
        let (parameter, maximum) = peak(q1, q2);
        let (mut best, mut at) = (f64::NEG_INFINITY, 0.0);
        for k in 0..=20000 {
            let l = k as f64 / 20000.0;
            let v = det(q1, q2, l);
            if v > best {
                (best, at) = (v, l);
            }
        }
        assert!((parameter - at).abs() < 0.001);
        assert!((maximum - best).abs() < 1e-8);
    }

    /// The scenario's checks: the three poses are apart, touching and overlapping; the contact
    /// line is a common tangent, and both ellipses map it to one contact point. The peaks and
    /// their parameters agree with numga's.
    #[test]
    fn the_three_poses_classify() {
        let s = scene();
        let q2: Vec<Quadric> = [0.8, 0.0, -0.6].iter().map(|o| s.second(*o)).collect();
        let peaks: Vec<(f64, f64)> = q2.iter().map(|q| peak(s.q1, *q)).collect();
        assert!(peaks[0].1 > 0.0);
        assert!(peaks[1].1.abs() < 1e-8);
        assert!(peaks[2].1 < 0.0);
        // numga's values (its exponential of the translation is good to 1e-11).
        let numga = [
            (0.606868521999339, 2.106763619465154),
            (0.634104483621456, 0.0),
            (0.685785050185595, -1.222_671_242_766_22),
        ];
        for ((l, m), (nl, nm)) in peaks.iter().zip(numga) {
            assert!((l - nl).abs() < 1e-9 && (m - nm).abs() < 1e-9, "{l} {m}");
        }
        let (_, line, point) = contact(s.q1, q2[1]);
        assert!((line & s.q1.of(line)).s().abs() < 1e-6);
        let other = q2[1].of(line).normalized().into_inner();
        // The two contact points are one: their join vanishes.
        let join = other & point;
        assert!(join.max_abs_diff(&Line::zero()) < 1e-5);
        // numga's contact point, at no distance from ours.
        let numga = Point::xy(0.623064025936332, 0.821205844662909);
        assert!((point & numga).norm() < 1e-6);
    }

    /// The blend at the peak of the separated pose is a hyperbola: its form takes both signs on
    /// the points at infinity (it has asymptotes), and both centres lie on one side of it.
    #[test]
    fn the_separating_blend_is_a_hyperbola() {
        let s = scene();
        let q2 = s.second(0.8);
        let sep = separating(s.q1, q2);
        let side = |p: P| (p & sep.of(p)).s();
        let (c1, c2) = (s.q1.of(infinity()), q2.of(infinity()));
        assert!(side(c1) * side(c2) > 0.0);
        let at_infinity: Vec<f64> = (0..36)
            .map(|k| {
                let turn = motor(0.0, 0.0, core::f64::consts::PI * k as f64 / 36.0);
                side(turn >> Point::direction(1.0, 0.0))
            })
            .collect();
        assert!(at_infinity.iter().any(|v| *v > 0.0) && at_infinity.iter().any(|v| *v < 0.0));
    }

    #[test]
    fn a_frame_draws() {
        let mut draw = super::draw;
        for t in [0.0, 2.0, 4.0] {
            let c = gax_numga_examples::app::frame(
                &gax_numga_examples::Anim::new("t", 1.0).size(320, 180),
                t,
                &mut draw,
            );
            assert!(c.mean().luma() > 0.0);
        }
    }
}
