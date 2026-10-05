//! numga's `quadrics/gaussian`: a Gaussian and its 1σ conic straight from homogeneous point
//! moments in PGA2D. Each point `p` gives the rank-one map `L ↦ p (L ∨ p)` from lines to points;
//! their mean, the second moment, holds both the cloud's location and its spread, with no
//! centring. Its inverse, the precision, maps points to lines, and on unit-weight points
//! `precision(p) ∨ p` is one plus the squared Mahalanobis distance. Subtracting the dyad of the
//! weight line twice turns the precision into a polarity whose zero locus is the 1σ ellipse.
//! The animation turns, slides and stretches the cloud and refits it every frame.

use gax_colour::Light;
use gax_numga_examples::{
    Align, Anim, Axes, Canvas, Marker, Point2, Rect, backdrop, caption, palette, run,
};

mod gaussian {
    use gax::pga2d::{Line, Motor, Point};

    pub type P = Point<(), f64>;
    /// A polarity: points to lines.
    pub type Polarity = Line<(Point,), f64>;

    use gax_numga_examples::rng::{Draw, rng};
    use gax_numga_examples::signal::{phasor, wave};

    /// The fit: the precision, and the 1σ polarity.
    pub struct Fit {
        pub precision: Polarity,
        pub quadric: Polarity,
    }

    /// Fit unit-weight points: the precision (the inverse second moment) and the 1σ polarity.
    pub fn fit(points: &[P]) -> Fit {
        // Leave the line slot open: each point contributes a rank-one map from lines to
        // points. Their mean is the second moment.
        let n = points.len() as f64;
        let moment = points
            .iter()
            .fold(Point::<(Line,), f64>::zero(), |m, p| {
                m + *p * (Line::slot() & *p)
            })
            .gp(1.0 / n);
        let precision: Polarity = moment.inverse();
        // The weight line, from the data: `weight ∨ p == 1` on every unit-weight point. Its
        // dyad evaluates to that constant squared; subtracting it twice makes Mahalanobis
        // distance one the zero locus.
        let mean = points.iter().fold(Point::zero(), |m, p| m + *p).gp(1.0 / n);
        let weight = precision.of(mean);
        let quadric = precision - weight * (weight & Point::slot()).gp(2.0);
        Fit { precision, quadric }
    }

    impl Fit {
        /// The density, with peak one: `exp(-d² / 2)` for the squared Mahalanobis distance d².
        pub fn density(&self, p: P) -> f64 {
            let squared_distance = (self.precision.of(p) & p).s() - 1.0;
            (-0.5 * squared_distance).exp()
        }

        /// The 1σ level, zero on the ellipse: `quadric(p) ∨ p`.
        pub fn level(&self, p: P) -> f64 {
            (self.quadric.of(p) & p).s()
        }
    }

    /// A cloud of `n` points about the origin, with the given standard deviations along x and
    /// y.
    pub fn cloud(seed: u64, n: usize, sd: [f64; 2]) -> Vec<P> {
        let mut rng = rng(seed);
        (0..n)
            .map(|_| Point::xy(rng.normal() * sd[0], rng.normal() * sd[1]))
            .collect()
    }

    /// The cloud at phase `t` (radians): breathing in its aspect, turned and slid around a
    /// small loop (an ellipse, the turned direction stretched by 0.8 across and 0.6 up). The
    /// stretch and the motor compose into one map, applied to every point.
    pub fn placed(base: &[P], t: f64) -> Vec<P> {
        let stretch = 1.0 + 0.35 * wave(2.0 * t);
        let stretch: Point<(Point,), f64> = Point::from_images([
            Point::new(stretch, 0.0, 0.0),
            Point::new(0.0, 1.0 / stretch, 0.0),
            Point::new(0.0, 0.0, 1.0),
        ]);
        let round = phasor(t);
        let placement = Motor::translation(0.6 + 0.8 * round.e20(), -0.4 + 0.6 * round.e01())
            * Motor::rotation(Point::xy(0.0, 0.0), 0.6 + t);
        let map = (placement >> Point::slot()).of(stretch);
        base.iter().map(|p| map.of(*p)).collect()
    }
}

use gaussian::*;

/// The density's colour: from the backdrop through blue to sky blue at the peak, faint enough
/// for the points and the conic to shine over it.
fn shade(d: f32) -> Light {
    let d = d.clamp(0.0, 1.0);
    let (blue, sky) = (palette::blue().faded(0.2), palette::sky().faded(0.45));
    if d < 0.5 {
        palette::bottom().blend(blue, d * 2.0)
    } else {
        blue.blend(sky, d * 2.0 - 1.0)
    }
}

const X: [f32; 2] = [-5.0, 6.0];
const Y: [f32; 2] = [-5.0, 4.0];

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let phase = f64::from(t) / 10.0 * core::f64::consts::TAU;
    let base = cloud(4, 400, [1.5, 0.5]);
    let points = placed(&base, phase);
    let fit = fit(&points);

    let screen = c.rect();
    let rect =
        Rect::new(0.0, 0.0, screen.width() * 0.86, screen.height()).inset(60.0, 76.0, 10.0, 40.0);
    // Equal scales about the middle of the data box, widened to the rectangle's aspect.
    let ax = {
        let half = ((Y[1] - Y[0]) / 2.0).max((X[1] - X[0]) / 2.0 * rect.height() / rect.width());
        let middle = (Point2::xy(X[0], Y[0]) + Point2::xy(X[1], Y[1])).unitized();
        Axes::equal(rect, middle, half)
    };
    ax.image(c, 1, |p| {
        let d = fit.density(p.map_coefs(f64::from)) as f32;
        Some(shade(d))
    });
    let cloud = palette::ink().faded(0.35);
    ax.scatter(c, &points, Marker::Dot, 3.5, cloud);
    ax.contour(
        c,
        |p| fit.level(p.map_coefs(f64::from)) as f32,
        220,
        0.0,
        2.4,
        palette::orange(),
    );
    // The y axis named at its middle, left of the ticks: above the frame it would meet the
    // caption.
    ax.frame(c, "", "X", "");
    let [_, middle] = rect.left_middle().to_euclidean();
    let side = Point2::xy(14.0, middle);
    c.text("Y", side, 12.0, palette::ink(), Align::Left);
    ax.legend(
        c,
        &[
            ("POINT CLOUD", cloud),
            ("1 SIGMA: Q(P) & P = 0", palette::orange()),
        ],
    );
    // A colour bar for the density: the point a fraction `x` across and `f` up the bar.
    let bar = Rect::new(
        screen.width() * 0.89,
        76.0,
        screen.width() * 0.91,
        screen.height() - 40.0,
    );
    let at =
        |x: f32, f: f32| bar.bottom_left() + Point2::direction(x * bar.width(), -f * bar.height());
    let steps = 64;
    for i in 0..steps {
        let (a, b) = (i as f32 / steps as f32, (i + 1) as f32 / steps as f32);
        c.fill(
            &[at(0.0, a), at(1.0, a), at(1.0, b), at(0.0, b)],
            shade(a),
            1.0,
        );
    }
    for (v, s) in [(0.0, "0"), (0.5, "0.5"), (1.0, "1")] {
        let label = at(1.0, v) + Point2::direction(6.0, 4.0);
        c.text(s, label, 11.0, palette::ink(), Align::Left);
    }
    let title = at(0.5, 1.0) + Point2::direction(0.0, -8.0);
    c.text("DENSITY", title, 10.0, palette::grid(), Align::Center);
    caption(
        c,
        "A GAUSSIAN AND ITS 1 SIGMA CONIC FROM POINT MOMENTS",
        "PRECISION = (MEAN OF P (L & P))^-1, NO CENTRING (PGA2D)",
    );
}

fn main() {
    run(Anim::new("gaussian", 10.0).size(960, 540), draw);
}

#[cfg(test)]
mod tests {
    use super::gaussian::*;
    use gax::pga2d::{Motor, Point};

    /// The scenario's check: on unit-weight points the level is the squared distance less one
    /// and the density `exp(-d² / 2)`, so the 1σ conic is the contour at `exp(-1/2)`.
    #[test]
    fn level_is_minus_two_log_density_less_one() {
        for k in 0..4 {
            let points = placed(&cloud(4, 400, [1.5, 0.5]), 1.3 * k as f64);
            let fit = fit(&points);
            for i in 0..40 {
                for j in 0..40 {
                    let p = Point::xy(-5.0 + 11.0 * i as f64 / 39.0, -5.0 + 9.0 * j as f64 / 39.0);
                    let (d, l) = (fit.density(p), fit.level(p));
                    assert!((l - (-2.0 * d.ln() - 1.0)).abs() < 1e-8, "{l} {d}");
                }
            }
        }
    }

    /// The density is one at the sample mean, and the level is the squared Mahalanobis
    /// distance (from the biased covariance) less one.
    #[test]
    fn density_peaks_at_the_mean_and_level_matches_the_covariance() {
        let shift = Motor::translation(1.0, -1.0);
        let points: Vec<P> = cloud(1, 500, [2.0, 0.5])
            .into_iter()
            .map(|p| shift >> p)
            .collect();
        let fit = fit(&points);
        // The moments by hand, from the coordinates.
        let xy: Vec<[f64; 2]> = points.iter().map(|p| p.to_euclidean()).collect();
        let n = xy.len() as f64;
        let mean = [0, 1].map(|k| xy.iter().map(|p| p[k]).sum::<f64>() / n);
        let peak = fit.density(Point::xy(mean[0], mean[1]));
        assert!((peak - 1.0).abs() < 1e-12, "{peak}");
        let cov = |a: usize, b: usize| {
            xy.iter()
                .map(|p| (p[a] - mean[a]) * (p[b] - mean[b]))
                .sum::<f64>()
                / n
        };
        let (sxx, sxy, syy) = (cov(0, 0), cov(0, 1), cov(1, 1));
        let det = sxx * syy - sxy * sxy;
        for q in [[0.0, 0.0], [2.0, 1.0], [-1.0, -3.0]] {
            let (dx, dy) = (q[0] - mean[0], q[1] - mean[1]);
            let mahalanobis = (syy * dx * dx - 2.0 * sxy * dx * dy + sxx * dy * dy) / det;
            let level = fit.level(Point::xy(q[0], q[1]));
            assert!(
                (level - (mahalanobis - 1.0)).abs() < 1e-9,
                "{level} {mahalanobis}"
            );
        }
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws(super::draw, 0.5);
    }
}
