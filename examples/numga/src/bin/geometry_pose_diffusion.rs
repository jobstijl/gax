//! numga's `geometry/pose_diffusion`: how far a station-keeping vessel wanders from its set
//! point, in PGA2D. Gusts push the vessel off; the controller pushes it back in proportion to the
//! error. The error is a twist, and its covariance a map from lines to twists,
//! `Covariance = Twist <- Line`: for a line `l`, `l & covariance(l)` is the variance of the
//! error's reading by `l`. The covariance grows from zero as gusts accumulate and levels off where
//! the controller balances them; the level is found both by integrating the growth and by solving
//! directly for the map at which growth stops (the continuous Lyapunov equation, written with the
//! covariance as a dyad left open in three slots).
//!
//! The animation shows two thousand simulated vessels spreading about the set point, still on
//! the left and turning on the right, with the predicted 2σ ellipse (solid) growing to the
//! settled one (dashed). The gusts hit the side much harder than the bow; turning spreads them
//! over every direction.

use gax::Unit;
use gax::pga2d::{Line, Motor, Point, Scalar};
use gax_numga_examples::{Align, Anim, Axes, Canvas, backdrop, caption, palette, plot, run};
use std::sync::OnceLock;

mod station {
    use super::*;

    /// A position error: the small motion from the set point to the pose, a bivector.
    pub type Twist = Point<(), f64>;
    /// A line reads out a twist, `line & twist`.
    pub type Readout = Line<(), f64>;
    /// A measurement line to the correlated twist.
    pub type Covariance = Point<(Line,), f64>;
    /// The rate of change of a position error.
    pub type Dynamics = Point<(Point,), f64>;
    /// The same rate of change, acting on measurement lines.
    pub type Readouts = Line<(Line,), f64>;
    /// The covariance of two position measurements.
    pub type Spread = Scalar<(Line, Line), f64>;
    /// White noise to a gust's push on the vessel.
    pub type Kicks = Point<(Line,), f64>;
    pub type M = Unit<Motor<(), f64>>;

    /// The controller gain, per second.
    pub const RELAXATION: f64 = 0.6;

    /// The set point; in PGA2D a point is a bivector, like a twist.
    pub fn origin() -> Twist {
        Point::xy(0.0, 0.0)
    }

    /// numga's PGA2D basis twists by name: `yw = e20`, `wx = e01`, `xy = e12`, which are gax's
    /// `Point` coefficients in order. A `yw` twist slides the set point along `y`, sideways; a
    /// `wx` twist along `x`, along the bow; `xy` turns it.
    pub fn yw(a: f64) -> Twist {
        Point::new(a, 0.0, 0.0)
    }
    pub fn wx(a: f64) -> Twist {
        Point::new(0.0, a, 0.0)
    }
    pub fn xy(a: f64) -> Twist {
        Point::new(0.0, 0.0, a)
    }

    /// The turning rate of each scene: still, and turning at two radians per second.
    pub fn rates() -> [(&'static str, Twist); 2] {
        [("STILL", xy(0.0)), ("TURNING", xy(2.0))]
    }

    /// The rate of change of a position error. Turning on the spot turns the error with it,
    /// the commutator with the turning rate; the controller removes the relaxation's share of
    /// the error per second.
    pub fn drift(rate: Twist, relaxation: f64) -> Dynamics {
        let t = Point::slot();
        t.commutator(rate) - t.gp(relaxation)
    }

    /// The rate of change as a map on measurement lines: reading the changed error with `l` is
    /// reading the error itself with `on_readouts(l)`. Solved from the incidence pairing against
    /// `Line & dynamics`; in matrix notation, the transpose.
    pub fn on_readouts(dynamics: Dynamics) -> Readouts {
        (Line::slot() & Point::slot()).solve(Line::slot() & dynamics)
    }

    /// The rate of change of the covariance: the dynamics on its twists, on its lines, and the
    /// gusts' covariance per second.
    pub fn growth(dynamics: Dynamics, covariance: Covariance, noise: Covariance) -> Covariance {
        dynamics.of(covariance) + covariance.of(on_readouts(dynamics)) + noise
    }

    /// The basis lines.
    pub fn basis() -> [Readout; 3] {
        [
            Line::new(1.0, 0.0, 0.0),
            Line::new(0.0, 1.0, 0.0),
            Line::new(0.0, 0.0, 1.0),
        ]
    }

    /// The gusts' covariance per second: white noise has unit variance on each basis line and
    /// none between them, so the sum of the dyads of each basis line's kick.
    pub fn covariance(kicks: Kicks) -> Covariance {
        basis()
            .iter()
            .map(|b| kicks.of(*b) * (kicks.of(*b) & Line::slot()))
            .fold(Point::zero(), |a, b| a + b)
    }

    /// The map from white noise to gust twists, from the standard deviations per square root of
    /// a second of the sideways, forward and turning gusts, in the body frame.
    pub fn shaping(sideways: f64, forward: f64, turning: f64) -> Kicks {
        let l = Line::slot();
        yw(sideways) * (yw(1.0) & l) + wx(forward) * (wx(1.0) & l) + xy(turning) * (xy(1.0) & l)
    }

    /// Mostly sideways.
    pub fn kicks() -> Kicks {
        shaping(0.35, 0.08, 0.15)
    }

    /// One explicit Euler step of length `dt` for an error; the kick is already scaled by
    /// `sqrt(dt)`, as white noise integrated over the step.
    pub fn step_error(dynamics: Dynamics, error: Twist, kick: Twist, dt: f64) -> Twist {
        error + dynamics.of(error).gp(dt) + kick
    }

    /// The same step for the predicted covariance.
    pub fn step_covariance(
        dynamics: Dynamics,
        noise: Covariance,
        covariance: Covariance,
        dt: f64,
    ) -> Covariance {
        covariance + growth(dynamics, covariance, noise).gp(dt)
    }

    /// The covariance at which growth stops. Write it as a sum of dyads `t (s & l)` and leave
    /// all three open: growth with zero noise becomes a map on `(t, s, l)`, linear in the dyad.
    /// Matching its line input against the noise and solving for the coefficients on the two
    /// twist inputs gives the settled covariance, `sum x_ij e_i (e_j & l)`.
    ///
    /// Solved for both twist slots at once with `lstsq_pair`, as numga's `lstsq` does: the
    /// dyad coefficients `x` with `lyapunov.of_pair(x) = −noise`, put back together by the map
    /// that builds the dyads.
    pub fn settled(dynamics: Dynamics, noise: Covariance) -> Covariance {
        let t = Point::slot();
        // Twist <- (Twist, Twist, Line): the dynamics applied to t, and applied to s.
        let lyapunov = dynamics.of(t) * (t & Line::slot()) + t * (dynamics.of(t) & Line::slot());
        let x: Point<(Point,), f64> = lyapunov.lstsq_pair(-noise);
        let dyads = Point::slot() * (Point::slot() & Line::slot());
        dyads.of_pair(x)
    }

    /// The covariance of readings of the vessel's position. An error moves the set point by its
    /// commutator with it; reading that displacement with a line is reading the twist with
    /// another line, solved from the incidence pairing. Pairing two such readings through the
    /// covariance gives a symmetric form on lines, whose eigenpairs are the ellipse's axes.
    pub fn position_spread(covariance: Covariance) -> Spread {
        let shift = Point::slot().commutator(origin());
        let readout = (Line::slot() & Point::slot()).solve(Line::slot() & shift);
        readout & covariance.of(readout)
    }

    /// The 2σ ellipse of the position's readings, as a ring of points about the set point: the
    /// spread's eigenpairs in the coefficient basis, the ones that read a position (the offset
    /// `e0` reads none).
    pub fn ellipse(spread: Spread, n: usize) -> Vec<[f64; 2]> {
        let (values, modes) = spread.eigh();
        let axes: Vec<([f64; 2], f64)> = modes
            .iter()
            .zip(values)
            .filter(|(m, _)| m.e1().hypot(m.e2()) > 0.5)
            .map(|(m, v)| {
                let k = m.e1().hypot(m.e2());
                ([m.e1() / k, m.e2() / k], 2.0 * v.max(0.0).sqrt())
            })
            .collect();
        (0..=n)
            .map(|i| {
                let a = core::f64::consts::TAU * i as f64 / n as f64;
                let (c, s) = (a.cos(), a.sin());
                let mut p = [0.0, 0.0];
                for (k, (normal, r)) in axes.iter().enumerate() {
                    let w = if k == 0 { c } else { s } * r;
                    p[0] += normal[0] * w;
                    p[1] += normal[1] * w;
                }
                p
            })
            .collect()
    }

    pub use gax_numga_examples::rng::Rng;

    /// A run: snapshots of the bodies' errors and of the predicted covariance every few steps,
    /// and the settled covariance.
    pub struct Diffusion {
        pub errors: Vec<Vec<Twist>>,
        pub predicted: Vec<Covariance>,
        pub limit: Covariance,
    }

    /// The pose of a body: the exponential of half its error twist, a motor.
    pub fn pose(error: Twist) -> M {
        error.gp(0.5).exp()
    }

    /// Simulate many bodies holding the commanded pose, and integrate their predicted
    /// covariance. All start at the commanded pose with zero covariance; every step each body
    /// gets an independent gust and the prediction advances with the same dynamics.
    pub fn diffuse(
        rate: Twist,
        kicks: Kicks,
        seconds: f64,
        dt: f64,
        bodies: usize,
        every: usize,
        seed: u64,
    ) -> Diffusion {
        let (dynamics, noise) = (drift(rate, RELAXATION), covariance(kicks));
        let limit = settled(dynamics, noise);
        let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
        let mut errors = vec![Point::zero(); bodies];
        let mut predicted = Point::zero();
        let (mut snapshots, mut covariances) = (Vec::new(), Vec::new());
        let root = dt.sqrt();
        for count in 0..(seconds / dt).round() as usize {
            if count % every == 0 {
                snapshots.push(errors.clone());
                covariances.push(predicted);
            }
            for e in &mut errors {
                let white = Line::new(rng.normal(), rng.normal(), rng.normal()).gp(root);
                *e = step_error(dynamics, *e, kicks.of(white), dt);
            }
            predicted = step_covariance(dynamics, noise, predicted, dt);
        }
        snapshots.push(errors);
        covariances.push(predicted);
        Diffusion {
            errors: snapshots,
            predicted: covariances,
            limit,
        }
    }

    /// The sample covariance of the bodies' errors.
    pub fn empirical(errors: &[Twist]) -> Covariance {
        let sum = errors
            .iter()
            .map(|e| *e * (*e & Line::slot()))
            .fold(Point::zero(), |a, b| a + b);
        sum.gp(1.0 / errors.len() as f64)
    }

    /// The largest coefficient of a covariance.
    #[cfg(test)]
    pub fn max_abs(c: Covariance) -> f64 {
        c.c.iter().flatten().fold(0.0f64, |m, v| m.max(v.abs()))
    }

    /// The largest coefficient difference of two covariances.
    #[cfg(test)]
    pub fn max_diff(a: Covariance, b: Covariance) -> f64 {
        max_abs(a - b)
    }
}

use station::*;

const SECONDS: f64 = 8.0;
const DT: f64 = 0.01;
const EVERY: usize = 5;
const EXTENT: f32 = 1.0;

/// Both runs, simulated once.
fn runs() -> &'static [(String, Diffusion); 2] {
    static RUNS: OnceLock<[(String, Diffusion); 2]> = OnceLock::new();
    RUNS.get_or_init(|| {
        rates().map(|(name, rate)| {
            (
                name.to_string(),
                diffuse(rate, kicks(), SECONDS, DT, 2000, EVERY, 0),
            )
        })
    })
}

fn f32s(p: [f64; 2]) -> [f32; 2] {
    [p[0] as f32, p[1] as f32]
}

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let (w, h) = (c.width as f32, c.height as f32);
    let runs = runs();
    let frames = runs[0].1.errors.len();
    let index = ((t / (DT as f32 * EVERY as f32)) as usize).min(frames - 1);
    let seconds = index as f32 * DT as f32 * EVERY as f32;
    let top = (h / 30.0).clamp(10.0, 22.0) * 3.8;
    for (k, (name, run)) in runs.iter().enumerate() {
        let rect = plot::inset(
            [k as f32 * w / 2.0, top, (k + 1) as f32 * w / 2.0, h],
            w * 0.06,
            h * 0.05,
            w * 0.02,
            h * 0.09,
        );
        let ax = Axes::equal(rect, [0.0, 0.0], EXTENT);
        ax.frame(
            c,
            &format!("{name}, T = {seconds:.1} S"),
            "FORWARD (M)",
            "SIDEWAYS (M)",
        );
        // Each vessel as a short stroke through its position along its heading: the set point
        // and a point a little ahead of it, moved by the vessel's pose.
        let ahead = Point::xy(0.08 * f64::from(EXTENT), 0.0);
        for error in &run.errors[index] {
            let pose = pose(*error);
            let here = (pose >> origin()).to_euclidean();
            let tip = (pose >> ahead).to_euclidean();
            let d = [(tip[0] - here[0]) * 0.5, (tip[1] - here[1]) * 0.5];
            ax.line(
                c,
                f32s([here[0] - d[0], here[1] - d[1]]),
                f32s([here[0] + d[0], here[1] + d[1]]),
                1.0,
                palette::sky(),
                0.35,
            );
        }
        let ring = |cov: Covariance| -> Vec<[f32; 2]> {
            ellipse(position_spread(cov), 96)
                .into_iter()
                .map(f32s)
                .collect()
        };
        ax.dashed(c, &ring(run.limit), 1.8, 6.0, palette::red(), 1.0);
        ax.polyline(
            c,
            &ring(empirical(&run.errors[index])),
            1.4,
            palette::yellow(),
            0.9,
        );
        ax.polyline(c, &ring(run.predicted[index]), 2.2, palette::red(), 1.0);
        if k == 0 {
            ax.legend(
                c,
                &[
                    ("VESSELS", palette::sky()),
                    ("PREDICTED 2 SIGMA", palette::red()),
                    ("SAMPLE 2 SIGMA", palette::yellow()),
                ],
            );
        }
    }
    caption(
        c,
        "POSE DIFFUSION: A STATION-KEEPING VESSEL",
        "COVARIANCE AS A MAP FROM LINES TO TWISTS (PGA2D); DASHED: THE SETTLED ONE",
    );
    c.text(
        "GUSTS MOSTLY SIDEWAYS; TURNING SPREADS THEM",
        w - 12.0,
        h - 10.0,
        11.0,
        palette::grid(),
        Align::Right,
    );
}

fn main() {
    run(Anim::new("pose diffusion", 9.0).size(960, 540), draw);
}

#[cfg(test)]
mod tests {
    use super::station::*;

    /// The settled covariance zeroes the growth, in every scene.
    #[test]
    fn settled_covariance_zeroes_the_growth() {
        for (_, rate) in rates() {
            let dynamics = drift(rate, RELAXATION);
            let noise = covariance(kicks());
            let limit = settled(dynamics, noise);
            assert!(max_abs(growth(dynamics, limit, noise)) < 1e-10);
            assert!(max_abs(limit) > 1e-3);
        }
    }

    /// A short run passes the scenario's checks: the integrated prediction converges to the
    /// settled covariance, and the bodies' sample covariance matches it within sampling error.
    #[test]
    fn a_short_run_settles() {
        for (_, rate) in rates() {
            let run = diffuse(rate, kicks(), 8.0, 0.02, 800, 100, 1);
            let (dynamics, noise) = (drift(rate, RELAXATION), covariance(kicks()));
            assert!(max_abs(growth(dynamics, run.limit, noise)) < 1e-10);
            let scale = max_abs(run.limit);
            let last = *run.predicted.last().expect("a snapshot");
            assert!(max_diff(last, run.limit) < 0.05 * scale);
            let errors = run.errors.last().expect("a snapshot");
            assert!(max_diff(empirical(errors), run.limit) < 0.25 * scale);
            assert_eq!(run.errors.len(), run.predicted.len());
        }
    }

    /// The scenario's own run, at its full size.
    #[test]
    fn the_scenario_run_settles() {
        for (k, (_, run)) in super::runs().iter().enumerate() {
            let scale = max_abs(run.limit);
            let last = *run.predicted.last().expect("a snapshot");
            assert!(max_diff(last, run.limit) < 0.05 * scale, "scene {k}");
            let errors = run.errors.last().expect("a snapshot");
            assert!(max_diff(empirical(errors), run.limit) < 0.25 * scale);
        }
    }

    /// Still, the vessel wanders mostly sideways: the ellipse is longer along y than along x.
    /// Turning, it spreads more evenly.
    #[test]
    fn still_is_sideways_and_turning_spreads() {
        let ratio = |rate| {
            let limit = settled(drift(rate, RELAXATION), covariance(kicks()));
            let ring = ellipse(position_spread(limit), 64);
            let (mut x, mut y) = (0.0f64, 0.0f64);
            for p in ring {
                x = x.max(p[0].abs());
                y = y.max(p[1].abs());
            }
            y / x
        };
        let [(_, still), (_, turning)] = rates();
        let (a, b) = (ratio(still), ratio(turning));
        assert!(a > 2.0, "{a}");
        assert!(b < a * 0.6, "{a} {b}");
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws(super::draw, 2.5);
    }
}
