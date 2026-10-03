//! numga's `quadrics/gaussian`: a Gaussian and its 1σ conic straight from homogeneous point
//! moments in PGA2D. Each point `p` gives the rank-one map `L ↦ p (L ∨ p)` from lines to points;
//! their mean, the second moment, holds both the cloud's location and its spread, with no
//! centring. Its inverse, the precision, maps points to lines, and on unit-weight points
//! `precision(p) ∨ p` is one plus the squared Mahalanobis distance. Subtracting the dyad of the
//! weight line twice turns the precision into a polarity whose zero locus is the 1σ ellipse.
//! The animation turns, slides and stretches the cloud and refits it every frame.

use gax::pga2d::Point;
use gax_numga_examples::canvas::{mix, srgb};
use gax_numga_examples::{
    Align, Anim, Axes, Canvas, Marker, backdrop, caption, palette, plot, run,
};

mod gaussian {
    use gax::pga2d::{Line, Motor, Point};

    pub type P = Point<(), f64>;
    /// A polarity: points to lines.
    pub type Polarity = Line<(Point,), f64>;

    pub use gax_numga_examples::rng::Rng;

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

    /// A stretched cloud of `n` points with the given standard deviations, before placement.
    pub fn cloud(seed: u64, n: usize, sd: [f64; 2]) -> Vec<[f64; 2]> {
        // Xorshift's state must not be zero.
        let mut rng = Rng(seed.max(1));
        (0..n)
            .map(|_| [rng.normal() * sd[0], rng.normal() * sd[1]])
            .collect()
    }

    /// The cloud at phase `t` (radians): turned, slid around a small loop and breathing in
    /// its aspect, moved as a batch by one motor.
    pub fn placed(base: &[[f64; 2]], t: f64) -> Vec<P> {
        let stretch = 1.0 + 0.35 * (2.0 * t).sin();
        let placement = Motor::translation(0.6 + 0.8 * t.cos(), -0.4 + 0.6 * t.sin())
            * Motor::rotation(Point::xy(0.0, 0.0), 0.6 + t);
        base.iter()
            .map(|&[x, y]| placement >> Point::xy(x * stretch, y / stretch))
            .collect()
    }
}

use gaussian::*;

/// The density's colour: from the backdrop through blue to pale sky blue at the peak.
fn shade(d: f32) -> gax_numga_examples::Rgb {
    let d = d.clamp(0.0, 1.0);
    if d < 0.5 {
        mix(palette::bottom(), palette::blue(), d * 2.0)
    } else {
        mix(palette::blue(), palette::sky(), d * 2.0 - 1.0)
    }
}

const X: [f32; 2] = [-5.0, 6.0];
const Y: [f32; 2] = [-5.0, 4.0];

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let (w, h) = (c.width as f32, c.height as f32);
    let phase = f64::from(t) / 10.0 * core::f64::consts::TAU;
    let base = cloud(4, 400, [1.5, 0.5]);
    let points = placed(&base, phase);
    let fit = fit(&points);

    let rect = plot::inset([0.0, 0.0, w * 0.86, h], 60.0, 64.0, 10.0, 40.0);
    // Equal scales: the data box widened to the rectangle's aspect.
    let ax = {
        let (rw, rh) = (rect[2] - rect[0], rect[3] - rect[1]);
        let half = ((Y[1] - Y[0]) / 2.0).max((X[1] - X[0]) / 2.0 * rh / rw);
        Axes::equal(rect, [(X[0] + X[1]) / 2.0, (Y[0] + Y[1]) / 2.0], half)
    };
    ax.image(c, 1, |x, y| {
        let d = fit.density(Point::xy(f64::from(x), f64::from(y))) as f32;
        Some(shade(d))
    });
    let xy: Vec<[f32; 2]> = points
        .iter()
        .map(|p| {
            let [x, y] = p.to_euclidean();
            [x as f32, y as f32]
        })
        .collect();
    ax.scatter(c, &xy, Marker::Dot, 3.5, srgb(0.15, 0.21, 0.29), 0.8);
    ax.contour(
        c,
        |x, y| fit.level(Point::xy(f64::from(x), f64::from(y))) as f32,
        220,
        0.0,
        2.4,
        palette::orange(),
    );
    ax.frame(c, "", "X", "Y");
    ax.legend(
        c,
        &[
            ("POINT CLOUD", srgb(0.15, 0.21, 0.29)),
            ("1 SIGMA: Q(P) & P = 0", palette::orange()),
        ],
    );
    // A colour bar for the density.
    let bar = [w * 0.89, 64.0, w * 0.91, h - 40.0];
    let steps = 64;
    for i in 0..steps {
        let (a, b) = (i as f32 / steps as f32, (i + 1) as f32 / steps as f32);
        let y0 = bar[3] - a * (bar[3] - bar[1]);
        let y1 = bar[3] - b * (bar[3] - bar[1]);
        c.fill(
            &[[bar[0], y0], [bar[2], y0], [bar[2], y1], [bar[0], y1]],
            shade(a),
            1.0,
        );
    }
    for (v, s) in [(0.0, "0"), (0.5, "0.5"), (1.0, "1")] {
        let y = bar[3] - v * (bar[3] - bar[1]);
        c.text(s, bar[2] + 6.0, y + 4.0, 11.0, palette::ink(), Align::Left);
    }
    c.text(
        "DENSITY",
        (bar[0] + bar[2]) / 2.0,
        bar[1] - 8.0,
        10.0,
        palette::grid(),
        Align::Center,
    );
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
    use gax::pga2d::Point;

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
        let xy: Vec<[f64; 2]> = cloud(1, 500, [2.0, 0.5])
            .into_iter()
            .map(|[x, y]| [x + 1.0, y - 1.0])
            .collect();
        let points: Vec<P> = xy.iter().map(|&[x, y]| Point::xy(x, y)).collect();
        let fit = fit(&points);
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
