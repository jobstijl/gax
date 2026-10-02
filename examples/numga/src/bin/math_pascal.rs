//! numga's `math/pascal`: the conic through five points, and Pascal's theorem, in PGA2D.
//!
//! A conic is a map from points to lines, `Conic = Line <- Point`: the line it sends a point to
//! is the point's polar, and the conic is the set of points on their own polar, where
//! `conic(point) & point` vanishes. The pairing is linear in the conic, so every point it must
//! pass through is one linear condition on it. Two lines make a conic, the pair of them; through
//! four points pass two such pairs (opposite sides of the quadrilateral, and its diagonals), and
//! every conic through the four is a combination of those two. A fifth point picks one.
//!
//! A sixth point follows from any line through the first: along the line the pairing is a
//! quadratic with one root known, so the other root is rational in the conic.
//!
//! Pascal's theorem: for six points on a conic taken as a hexagon, the three pairs of opposite
//! sides meet in three points on one line. With five points fixed, the condition that those
//! three crossings lie on one line is a form quadratic in the sixth point, and it is the conic
//! through the five.
//!
//! The animation: a sixth point runs once around the conic as the second crossing of a line
//! turning about the first point; the hexagon goes convex and crossed, and the three crossings
//! of its opposite sides stay on one line, which turns with it. On the right, Pascal's
//! condition drawn as a curve of its own lies on the conic, and the join of the three crossings
//! stays at round-off over the whole turn.

use gax::pga2d::{Line, Point, Scalar};
use gax_numga_examples::{
    Align, Anim, Axes, Canvas, Marker, Rgb, backdrop, caption, palette, plot, run,
};

mod pascal {
    use super::*;

    pub type L = Line<(), f64>;
    pub type P = Point<(), f64>;
    /// A conic: a map from points to their polar lines.
    pub type Conic = Line<(Point,), f64>;
    /// A form with a point open in two places.
    pub type Form = Scalar<(Point, Point), f64>;

    /// The next vertex of each vertex of a hexagon.
    pub const NEXT: [usize; 6] = [1, 2, 3, 4, 5, 0];

    /// Two lines as one conic: zero on the points of either line, its pairing the product of
    /// the two lines' pairings with the point.
    pub fn pair(first: L, second: L) -> Conic {
        let p = Point::slot();
        (first * (second & p) + second * (first & p)).gp(0.5)
    }

    /// The pairing of a conic with a point, zero on the conic.
    pub fn on(shape: Conic, p: P) -> f64 {
        (shape.of(p) & p).s()
    }

    /// The conic through five points: the combination of the two line pairs through the first
    /// four that vanishes on the fifth.
    pub fn conic(points: [P; 5]) -> Conic {
        let [first, second, third, fourth, fifth] = points;
        let sides = pair(first & second, third & fourth);
        let diagonals = pair(first & third, second & fourth);
        sides.gp(on(diagonals, fifth)) - diagonals.gp(on(sides, fifth))
    }

    /// Where the line from a point on a conic, towards a point at infinity, meets the conic
    /// again. Along `start + heading t` the pairing is
    /// `2 t (shape(start) & heading) + t² (shape(heading) & heading)`, zero at the start and at
    /// the other root.
    pub fn second_crossing(shape: Conic, start: P, heading: P) -> P {
        start.gp(on(shape, heading)) - heading.gp(2.0 * (shape.of(start) & heading).s())
    }

    /// The three points where opposite sides of a hexagon meet.
    pub fn crossings(hexagon: &[P; 6]) -> [P; 3] {
        let sides: Vec<L> = (0..6).map(|i| hexagon[i] & hexagon[NEXT[i]]).collect();
        [0, 1, 2].map(|i| sides[i] ^ sides[i + 3])
    }

    /// Pascal's condition on a sixth point after five, as a form with the sixth point open in
    /// both places it appears: the join of the three crossings of opposite sides, zero exactly
    /// when they lie on one line.
    pub fn pascal(five: [P; 5]) -> Form {
        let [first, second, third, fourth, fifth] = five;
        let open = Point::slot();
        ((first & second) ^ (fourth & fifth))
            & ((second & third) ^ (fifth & open))
            & ((third & fourth) ^ (open & first))
    }

    /// A point at unit weight.
    pub fn point(x: f64, y: f64) -> P {
        Point::xy(x, y)
    }

    /// The point at infinity in the direction at an angle: the direction along x turned about
    /// the origin.
    pub fn heading(angle: f64) -> P {
        gax::pga2d::Motor::rotation(point(0.0, 0.0), angle) >> Point::direction(1.0, 0.0)
    }

    /// Points at the given angles on an ellipse, off the origin.
    pub fn ellipse_point(angle: f64) -> P {
        point(1.6 * angle.cos() + 0.3, angle.sin() - 0.1)
    }

    /// The five fixed points.
    pub fn five() -> [P; 5] {
        [0.4, 1.44, 4.12, 4.77, 6.08].map(ellipse_point)
    }

    /// The conic through the five points, and for the line through the first at an angle the
    /// hexagon of the five and the line's second crossing with the conic, with the crossings of
    /// its opposite sides.
    pub fn scene(angle: f64) -> (Conic, [P; 6], [P; 3]) {
        let five = five();
        let shape = conic(five);
        let sixth = second_crossing(shape, five[0], heading(angle));
        let sixth = sixth.gp(1.0 / sixth.e12());
        let hexagon = [five[0], five[1], five[2], five[3], five[4], sixth];
        (shape, hexagon, crossings(&hexagon))
    }

    /// The join of the three crossings over the product of their sizes: zero when they lie on
    /// one line, measured against their size since nearly parallel opposite sides meet far out.
    pub fn collinearity(crossing: &[P; 3]) -> f64 {
        let join = (crossing[0] & crossing[1] & crossing[2]).s();
        let size: f64 = crossing
            .iter()
            .map(|p| p.c.iter().map(|v| v * v).sum::<f64>().sqrt())
            .product();
        join / size
    }

    /// The angle of the turning line in a frame: half a turn over the loop (a line is back
    /// after half a turn), started off the first point's tangent.
    pub fn angle(phase: f64, frames: f64) -> f64 {
        core::f64::consts::PI * phase + core::f64::consts::PI / (2.0 * frames)
    }
}

use pascal::*;

const SECONDS: f32 = 8.0;
/// Colours of the three pairs of opposite sides and their crossings.
fn pairs(i: usize) -> Rgb {
    [palette::sky(), palette::green(), palette::purple()][i % 3]
}

fn xy(p: P) -> [f32; 2] {
    let [x, y] = p.to_euclidean();
    [x as f32, y as f32]
}

/// A line drawn across the axes.
fn across(ax: &Axes, c: &mut Canvas, l: L, width: f32, colour: Rgb, alpha: f32) {
    let (a, b, d) = (l.e1(), l.e2(), l.e0());
    let n2 = a * a + b * b;
    // The point of the line nearest the origin, and its direction.
    let foot = [(-a * d / n2) as f32, (-b * d / n2) as f32];
    ax.axline(c, foot, [-b as f32, a as f32], width, colour, alpha);
}

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let (w, h) = (c.width as f32, c.height as f32);
    let phase = f64::from(t / SECONDS);
    let (shape, hexagon, crossing) = scene(angle(phase, 120.0));

    // The hexagon, its sides, the conic and Pascal's line, on the left.
    let ax = Axes::equal(
        plot::inset([0.0, 0.0, w * 0.58, h], 16.0, 64.0, 8.0, 12.0),
        [1.0, 1.5],
        4.0,
    );
    let corners: Vec<[f32; 2]> = hexagon.iter().map(|p| xy(*p)).collect();
    ax.fill(c, &corners, palette::grid(), 0.35);
    for i in 0..6 {
        across(&ax, c, hexagon[i] & hexagon[NEXT[i]], 1.0, pairs(i), 0.55);
    }
    let conic_level = |x: f32, y: f32| on(shape, point(f64::from(x), f64::from(y))) as f32;
    ax.contour(c, conic_level, 240, 0.0, 2.0, palette::ink());
    across(&ax, c, crossing[0] & crossing[2], 2.4, palette::red(), 1.0);
    ax.scatter(c, &corners[..5], Marker::Dot, 7.0, palette::ink(), 1.0);
    ax.scatter(c, &corners[5..], Marker::Dot, 11.0, palette::orange(), 1.0);
    for (i, p) in crossing.iter().enumerate() {
        ax.scatter(c, &[xy(*p)], Marker::Dot, 11.0, palette::red(), 1.0);
        ax.scatter(c, &[xy(*p)], Marker::Dot, 7.0, pairs(i), 1.0);
    }

    // Pascal's condition as a curve of its own: the form on the sixth point, twice.
    let right = w * 0.6;
    let ax2 = Axes::equal(
        plot::inset([right, 40.0, w, h * 0.58], 30.0, 28.0, 16.0, 8.0),
        [0.3, -0.1],
        1.35,
    );
    let five = five();
    let form = pascal(five);
    ax2.contour(c, conic_level, 160, 0.0, 5.0, palette::grid());
    ax2.contour(
        c,
        |x, y| {
            let p = point(f64::from(x), f64::from(y));
            form.of(p).of(p).s() as f32
        },
        160,
        0.0,
        1.6,
        palette::yellow(),
    );
    let fixed: Vec<[f32; 2]> = five.iter().map(|p| xy(*p)).collect();
    ax2.scatter(c, &fixed, Marker::Dot, 7.0, palette::ink(), 1.0);
    ax2.text(
        c,
        [ax2.x[0] + 0.05, ax2.y[1] - 0.2],
        "PASCAL'S CONDITION IS THE CONIC",
        10.0,
        palette::yellow(),
        Align::Left,
    );

    // The join of the three crossings over the turn, at round-off.
    let ax3 = Axes::new(
        plot::inset([right, h * 0.58, w, h], 52.0, 30.0, 16.0, 40.0),
        [0.0, 1.0],
        [1e-18, 1e-12],
    )
    .log_y();
    ax3.frame(c, "JOIN OF THE CROSSINGS / SIZE", "TURN", "");
    let samples: Vec<[f32; 2]> = (0..=96)
        .map(|k| {
            let s = k as f64 / 96.0;
            let (_, _, x) = scene(angle(s, 120.0));
            [s as f32, collinearity(&x).abs().max(1e-18) as f32]
        })
        .collect();
    ax3.polyline(c, &samples, 1.4, palette::sky(), 1.0);
    let now = collinearity(&crossing).abs().max(1e-18) as f32;
    ax3.scatter(
        c,
        &[[phase as f32, now]],
        Marker::Dot,
        8.0,
        palette::orange(),
        1.0,
    );

    caption(
        c,
        "PASCAL: THE CONIC THROUGH FIVE POINTS",
        "OPPOSITE SIDES OF A HEXAGON ON A CONIC MEET ON ONE LINE (PGA2D)",
    );
}

fn main() {
    run(Anim::new("pascal", SECONDS).size(960, 540), draw);
}

#[cfg(test)]
mod tests {
    use super::pascal::*;

    /// numga's scenario checks: the conic passes through the five points and every sixth; the
    /// three crossings lie on one line; Pascal's condition on the sixth point is the conic up
    /// to scale. Six frames, as numga's test runs.
    #[test]
    fn scenes_pass_their_checks() {
        let frames = 6;
        let five = five();
        for k in 0..frames {
            let (shape, hexagon, crossing) = scene(angle(k as f64 / frames as f64, frames as f64));
            for p in &five {
                assert!(on(shape, *p).abs() < 1e-11);
            }
            assert!(
                on(shape, hexagon[5]).abs() < 1e-11,
                "{}",
                on(shape, hexagon[5])
            );
            assert!(collinearity(&crossing).abs() < 1e-12);
        }
        // Pascal's condition against the conic on probe points: one ratio. numga's probes are
        // normal samples; these come from a small xorshift (numga's stream cannot be
        // reproduced), and the ratio does not depend on them.
        let shape = conic(five);
        let form = pascal(five);
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut normal = || {
            let mut u = || {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state >> 11) as f64 / (1u64 << 53) as f64
            };
            let (a, b) = (u().max(1e-300), u());
            (-2.0 * a.ln()).sqrt() * (core::f64::consts::TAU * b).cos()
        };
        let ratios: Vec<f64> = (0..8)
            .map(|_| {
                let p = point(normal(), normal());
                form.of(p).of(p).s() / on(shape, p)
            })
            .collect();
        for r in &ratios {
            assert!(
                (r - ratios[0]).abs() <= 1e-11 * ratios[0].abs(),
                "{ratios:?}"
            );
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
