//! numga's `geometry/kalman`: an extended Kalman filter on PGA2D motors. The state is a motor,
//! its error a right perturbation `estimate exp(δ / 2)` by a bivector `δ`, and the covariance a
//! map from readouts to bivectors: a linear readout of a bivector is a line, so the covariance is
//! `Bivector <- Line`, the twist correlated with each readout. Prediction moves it like any map,
//! pulling the readout through the step and pushing the twist back (`step << Σ(step >> Line)`);
//! the update solves with the map inverse. The Jacobians of the matrix EKF are never written
//! down. The animation drives a meandering path with sparse noisy pose readings: the filter's
//! 2σ position ellipse grows while it dead-reckons and shrinks at each reading, and the error
//! plot fills in beside it.
//!
//! The position ellipse comes from a form on readout lines against the line metric, which
//! measures a line's normal and not its offset; numga solves that singular pencil with a general
//! eigensolver. Both forms vanish on the offset, so here they are restricted to the lines
//! through the origin, the image of VGA2D's vectors in PGA2D, where the line metric is the
//! Euclidean one: a symmetric eigenproblem (`eigh`), with no infinite mode to discard.

use gax::pga2d::{Line, Motor, Point, Scalar};
use gax::{Unit, vga2d};
use gax_numga_examples::canvas::mix;
use gax_numga_examples::rng::{Draw, Rng, rng};
use gax_numga_examples::{Anim, Axes, Canvas, Marker, backdrop, caption, palette, plot, run};

mod kalman {
    use super::*;

    pub type P = Point<(), f64>;
    pub type M = Unit<Motor<(), f64>>;
    /// A covariance: each readout line to the twist correlated with it.
    pub type Covariance = Point<(Line,), f64>;

    /// One state of the filter: after a prediction step, or after an update at a reading.
    #[derive(Clone, Copy, Debug)]
    pub struct State {
        pub estimate: M,
        pub sigma: Covariance,
        pub updated: bool,
    }

    /// The Kalman filter on motors, driven by body-frame steps and noisy full-pose readings, each
    /// paired with the steps since the previous one. Every state is returned: the predictions
    /// and, after each group, the corrected pose and covariance.
    pub fn kalman_filter(
        mut estimate: M,
        mut sigma: Covariance,
        steps: &[Vec<M>],
        measurements: &[M],
        motion_noise: Covariance,
        measurement_noise: Covariance,
    ) -> Vec<State> {
        let mut states = Vec::new();
        for (group, measured) in steps.iter().zip(measurements) {
            for step in group {
                // A perturbation on the right is carried through the step by `step << δ`, so the
                // covariance moves like any map: pull the readout through the step, push the
                // twist back. Then the step's own noise adds. (numga writes
                // `step << sigma(step >> Line) + motion_noise`, which Python, like Rust, reads as
                // `step << (… + motion_noise)`: the noise conjugated on one side only, and the
                // covariance no longer symmetric.)
                estimate = estimate * *step;
                sigma = (*step << sigma.of(*step >> Line::slot())) + motion_noise;
                states.push(State {
                    estimate,
                    sigma,
                    updated: false,
                });
            }
            // The innovation is the log of the relative motor, the gain a ratio of covariance
            // maps, and the correction is exponentiated.
            let innovation: P = (estimate.inverse() * *measured).log();
            let innovation = innovation.gp(2.0);
            let gain = sigma.of((sigma + measurement_noise).inverse());
            estimate = estimate * gain.of(innovation).gp(0.5).exp();
            sigma = sigma - gain.of(sigma);
            states.push(State {
                estimate,
                sigma,
                updated: true,
            });
        }
        states
    }

    /// The origin of the body frame.
    pub fn origin() -> P {
        Point::xy(0.0, 0.0)
    }

    /// The form of position readouts at the estimate: a readout of position along a line `l`,
    /// `l & shift(δ)`, is a readout of the twist through the incidence pairing; the covariance of
    /// those readouts is a form on lines.
    pub fn position_form(estimate: M, sigma: Covariance) -> Scalar<(Line, Line), f64> {
        let here = estimate >> origin();
        // The velocity of `here` under a body-frame twist.
        let shift = Point::slot().commutator(here).of(estimate >> Point::slot());
        let pairing = Line::slot() & Point::slot();
        let readout: Line<(Line,), f64> = pairing.solve(Line::slot() & shift);
        readout & sigma.of(readout)
    }

    /// The estimated position and its principal variances and axes (unit vectors), ascending.
    /// The offset of a readout line has no variance, so the form is read on the lines through
    /// the origin (VGA2D's vectors), where the line metric is the Euclidean one.
    pub fn position_ellipse(
        estimate: M,
        sigma: Covariance,
    ) -> (P, [f64; 2], [vga2d::Vector<(), f64>; 2]) {
        let through: Line<(vga2d::Vector,), f64> = Line::from(vga2d::Vector::slot());
        let form = position_form(estimate, sigma)
            .of(through)
            .at::<1>()
            .of(through);
        let (variances, axes) = form.eigh();
        (estimate >> origin(), variances, axes)
    }

    /// The 2σ position ellipse of a state, as a ring of `n` points: the unit circle, turned out
    /// step by step, through the map that stretches each axis to twice its deviation (a sum of
    /// dyads), about the estimate.
    pub fn ellipse(state: &State, n: usize) -> Vec<P> {
        let (centre, variances, axes) = position_ellipse(state.estimate, state.sigma);
        gax_numga_examples::plot::ellipse(centre, variances, axes, n)
    }

    /// Body-frame noise: isotropic translation, and rotation about the body origin.
    pub fn covariance(translation_std: f64, rotation_std: f64) -> Covariance {
        let dyad = |b: P| b * (b & Line::slot());
        (dyad(Point::new(1.0, 0.0, 0.0)) + dyad(Point::new(0.0, 1.0, 0.0)))
            .gp(translation_std * translation_std)
            + dyad(Point::new(0.0, 0.0, 1.0)).gp(rotation_std * rotation_std)
    }

    /// One bivector drawn from a covariance: readouts orthonormal in the covariance's own form
    /// carry independent unit normal draws, and the twists they select have covariance `cov`.
    pub fn sample(cov: Covariance, rng: &mut Rng) -> P {
        let readouts = Line::slot() & cov;
        let (_, lines) = readouts.eigh_with(readouts);
        lines
            .iter()
            .fold(Point::zero(), |acc, l| acc + cov.of(*l).gp(rng.normal()))
    }

    /// Simulate noisy motion and a pose measurement after each group of increments: the true
    /// and dead-reckoned poses at every step, and the measurements.
    pub fn simulate_motion(
        initial: M,
        increments: &[Vec<P>],
        motion_noise: Covariance,
        measurement_noise: Covariance,
        rng: &mut Rng,
    ) -> (Vec<M>, Vec<M>, Vec<M>) {
        let (mut truth, mut dead) = (initial, initial);
        let (mut true_path, mut dead_path, mut measurements) = (vec![], vec![], vec![]);
        for group in increments {
            for inc in group {
                truth = truth * (*inc + sample(motion_noise, rng)).gp(0.5).exp();
                dead = dead * inc.gp(0.5).exp();
                true_path.push(truth);
                dead_path.push(dead);
            }
            measurements.push(truth * sample(measurement_noise, rng).gp(0.5).exp());
        }
        (true_path, dead_path, measurements)
    }

    pub const DT: f64 = 0.1;
    pub const READINGS: usize = 12;
    pub const STEPS_PER_READING: usize = 25;

    /// The tracking scene.
    pub struct Tracking {
        pub truth: Vec<M>,
        pub dead: Vec<M>,
        pub measurements: Vec<M>,
        pub states: Vec<State>,
    }

    impl Tracking {
        /// The states after each reading.
        pub fn updates(&self) -> Vec<State> {
            self.states.iter().copied().filter(|s| s.updated).collect()
        }
        /// The position errors at the readings, dead reckoning and filtered: the distance
        /// between two unit points is the norm of the line joining them.
        pub fn errors(&self) -> (Vec<f64>, Vec<f64>) {
            let at = |k: usize| (k + 1) * STEPS_PER_READING - 1;
            let ups = self.updates();
            (0..READINGS)
                .map(|k| {
                    let t = self.truth[at(k)] >> origin();
                    (
                        (t & (self.dead[at(k)] >> origin())).norm(),
                        (t & (ups[k].estimate >> origin())).norm(),
                    )
                })
                .unzip()
        }
    }

    /// The seed of the scene's noise.
    pub const SEED: u64 = 1;

    /// Drive at 1 m/s along the body's x axis while the turn rate wanders, so that the path
    /// meanders; read the pose every 25 steps. The noise is drawn from `seed`.
    pub fn tracking(seed: u64) -> Tracking {
        let mut rng = rng(seed);
        let increments: Vec<Vec<P>> = (0..READINGS)
            .map(|r| {
                (0..STEPS_PER_READING)
                    .map(|s| {
                        let k = (r * STEPS_PER_READING + s) as f64;
                        let turn = 0.9 * (0.2 * k * DT).sin();
                        // The twist whose half exponential is the step.
                        (Point::translation_twist(1.0, 0.0) + Point::rotation_twist(origin(), turn))
                            .gp(2.0 * DT)
                    })
                    .collect()
            })
            .collect();
        let motion_noise = covariance(0.05 * DT.sqrt(), 0.12 * DT.sqrt());
        let measurement_noise = covariance(0.3, 0.1);
        let initial = Motor::translation(0.0, 0.0);
        let (truth, dead, measurements) = simulate_motion(
            initial,
            &increments,
            motion_noise,
            measurement_noise,
            &mut rng,
        );
        let steps: Vec<Vec<M>> = increments
            .iter()
            .map(|g| g.iter().map(|b| b.gp(0.5).exp()).collect())
            .collect();
        let states = kalman_filter(
            initial,
            covariance(0.0, 0.0),
            &steps,
            &measurements,
            motion_noise,
            measurement_noise,
        );
        Tracking {
            truth,
            dead,
            measurements,
            states,
        }
    }
}

use kalman::*;

const SECONDS: f32 = 12.0;

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let (w, h) = (c.width as f32, c.height as f32);
    let scene = tracking(SEED);
    let n = scene.truth.len();
    // The cursor runs over the drive in the first 85% of the loop, then holds.
    let shown = (((t / SECONDS) / 0.85).min(1.0) * n as f32).ceil().max(1.0) as usize;
    let path = |ms: &[M]| -> Vec<P> { ms.iter().map(|m| *m >> origin()).collect() };
    let (truth, dead) = (path(&scene.truth), path(&scene.dead));
    let filtered: Vec<P> = scene
        .states
        .iter()
        .map(|s| s.estimate >> origin())
        .collect();
    // The axes hold every path at every time.
    let (mut lo, mut hi) = ([f32::MAX; 2], [f32::MIN; 2]);
    for p in truth.iter().chain(&dead).chain(&filtered) {
        for (i, v) in p.to_euclidean().into_iter().enumerate() {
            lo[i] = lo[i].min(v as f32);
            hi[i] = hi[i].max(v as f32);
        }
    }
    let left = plot::inset([0.0, 0.0, w * 0.58, h], 50.0, 104.0, 16.0, 46.0);
    let half = ((hi[1] - lo[1]) * 0.5)
        .max((hi[0] - lo[0]) * 0.5 * (left[3] - left[1]) / (left[2] - left[0]))
        * 1.12;
    let ax = Axes::equal(left, [(lo[0] + hi[0]) * 0.5, (lo[1] + hi[1]) * 0.5], half);
    ax.frame(c, "PATHS AND 2 SIGMA ELLIPSES", "X", "Y");
    // The states up to the cursor: a prediction per step, plus an update at each reading.
    let readings_seen = shown / STEPS_PER_READING;
    let states_seen = shown + readings_seen;
    for s in scene.states[..states_seen].iter().filter(|s| s.updated) {
        ax.polyline(c, &ellipse(s, 40), 1.0, palette::sky(), 0.6);
    }
    ax.polyline(c, &dead[..shown], 1.3, palette::red(), 0.9);
    ax.polyline(c, &truth[..shown], 2.0, palette::ink(), 1.0);
    ax.polyline(c, &filtered[..states_seen], 1.3, palette::sky(), 1.0);
    let measured = path(&scene.measurements[..readings_seen]);
    ax.scatter(c, &measured, Marker::Cross, 8.0, palette::green(), 0.8);
    // The live ellipse at the cursor.
    let now = &scene.states[states_seen - 1];
    ax.polyline(c, &ellipse(now, 40), 2.0, palette::yellow(), 1.0);
    ax.scatter(
        c,
        &[filtered[states_seen - 1]],
        Marker::Dot,
        7.0,
        palette::yellow(),
        1.0,
    );
    ax.legend(
        c,
        &[
            ("TRUTH", palette::ink()),
            ("DEAD RECKONING", palette::red()),
            ("FILTERED", palette::sky()),
            ("POSE READINGS", palette::green()),
        ],
    );
    // The errors at the readings so far.
    let (dead_err, filt_err) = scene.errors();
    let updates = scene.updates();
    let sigma2: Vec<f32> = updates
        .iter()
        .map(|s| 2.0 * position_ellipse(s.estimate, s.sigma).1[1].sqrt() as f32)
        .collect();
    let times: Vec<f32> = (1..=READINGS)
        .map(|k| (k * STEPS_PER_READING) as f32 * DT as f32)
        .collect();
    let top = dead_err.iter().fold(0.0f64, |m, v| m.max(*v)) as f32 * 1.1;
    let right = plot::inset([w * 0.58, 0.0, w, h], 50.0, 104.0, 16.0, 46.0);
    let ex = Axes::new(right, [0.0, times[READINGS - 1] + 1.0], [0.0, top]);
    ex.frame(c, "ERROR AT THE READINGS", "TIME", "ERROR");
    let series = |v: &[f64]| -> Vec<[f32; 2]> {
        times
            .iter()
            .zip(v)
            .take(readings_seen)
            .map(|(t, e)| [*t, *e as f32])
            .collect()
    };
    let (d, f) = (series(&dead_err), series(&filt_err));
    let s2: Vec<[f32; 2]> = times
        .iter()
        .zip(&sigma2)
        .take(readings_seen)
        .map(|(t, e)| [*t, *e])
        .collect();
    ex.polyline(c, &d, 1.5, palette::red(), 1.0);
    ex.scatter(c, &d, Marker::Dot, 5.0, palette::red(), 1.0);
    ex.polyline(c, &f, 1.5, palette::sky(), 1.0);
    ex.scatter(c, &f, Marker::Dot, 5.0, palette::sky(), 1.0);
    ex.dashed(
        c,
        &s2,
        1.2,
        4.0,
        mix(palette::sky(), palette::ink(), 0.4),
        1.0,
    );
    let now_t = shown as f32 * DT as f32;
    ex.line(c, [now_t, 0.0], [now_t, top], 1.0, palette::grid(), 1.0);
    ex.legend(
        c,
        &[
            ("DEAD RECKONING", palette::red()),
            ("FILTERED", palette::sky()),
            (
                "FILTER'S OWN 2 SIGMA",
                mix(palette::sky(), palette::ink(), 0.4),
            ),
        ],
    );
    caption(
        c,
        "KALMAN FILTER ON MOTORS: COVARIANCE, LINE TO TWIST",
        "PREDICT BY CONJUGATING THE MAP, UPDATE BY ITS INVERSE (PGA2D)",
    );
}

fn main() {
    run(Anim::new("kalman", SECONDS).size(960, 540), draw);
}

#[cfg(test)]
mod tests {
    use super::kalman::*;
    use gax::ApproxEq;
    use gax::pga2d::Line;

    /// numga's test: sparse noisy pose readings keep the filtered path far closer to the truth
    /// than the dead-reckoned one, and closer than the readings themselves. Pooled over a
    /// number of draws, as the dead reckoning's drift varies much from one draw to the next.
    #[test]
    fn filter_beats_dead_reckoning_and_the_readings() {
        let (mut dead, mut filtered, mut read) = (0.0, 0.0, 0.0);
        for seed in 0..16 {
            let scene = tracking(seed);
            let (d, f) = scene.errors();
            dead += d.iter().sum::<f64>();
            filtered += f.iter().sum::<f64>();
            for (k, m) in scene.measurements.iter().enumerate() {
                let truth = scene.truth[(k + 1) * STEPS_PER_READING - 1] >> origin();
                read += (truth & (*m >> origin())).norm();
            }
        }
        assert!(filtered < 0.25 * dead, "{filtered} vs {dead}");
        assert!(filtered < read, "{filtered} vs {read}");
    }

    /// The covariance is symmetric as a form on readouts, and stays so through prediction and
    /// update; the position form vanishes on the readout line's offset, which is why the ellipse
    /// may be read on the lines through the origin.
    #[test]
    fn covariances_stay_symmetric_and_offsets_carry_no_variance() {
        let scene = tracking(SEED);
        for s in &scene.states {
            let form = Line::slot() & s.sigma;
            assert!(form.approx_eq(&form.swap(), 1e-9), "{form:?}");
            let offset = Line::new(0.0, 0.0, 1.0);
            let p = position_form(s.estimate, s.sigma).of(offset);
            assert!(p.max_abs_diff(&gax::pga2d::Scalar::zero()) < 1e-12);
            let (_, values, _) = position_ellipse(s.estimate, s.sigma);
            assert!(values[0] >= -1e-12 && values[0] <= values[1]);
        }
    }

    /// The drawn ellipse reaches across each line through the estimate as far as twice the
    /// deviation of the position's reading by that line (its support function).
    #[test]
    fn the_ellipse_is_two_sigma_along_every_line() {
        let scene = tracking(SEED);
        for s in scene.states.iter().step_by(37) {
            let ring = ellipse(s, 720);
            let here = s.estimate >> origin();
            let form = position_form(s.estimate, s.sigma);
            for k in 0..12 {
                let turn = gax::pga2d::Motor::rotation(here, 0.5 * k as f64);
                // A unit line through the estimate, and the same direction through the origin
                // (the form reads no offset).
                let line = turn >> (here & (here + gax::pga2d::Point::direction(0.0, 1.0)));
                let reach = ring
                    .iter()
                    .fold(0.0f64, |m, p| m.max((line & *p).s().abs()));
                let sigma = form.of(line).of(line).s().sqrt();
                assert!(
                    (reach - 2.0 * sigma).abs() < 1e-4 * sigma,
                    "{reach} {sigma}"
                );
            }
        }
    }

    /// A covariance built from noise levels reads them back: the translation readouts (lines
    /// through the origin, offset 0) have the translation variance and the rotation readout the
    /// rotation variance.
    #[test]
    fn covariance_reads_back_its_variances() {
        let cov = covariance(0.3, 0.1);
        let form = Line::slot() & cov;
        let along = |l: Line<(), f64>| form.of(l).of(l).s();
        assert!((along(Line::new(1.0, 0.0, 0.0)) - 0.09).abs() < 1e-12);
        assert!((along(Line::new(0.0, 1.0, 0.0)) - 0.09).abs() < 1e-12);
        assert!((along(Line::new(0.0, 0.0, 1.0)) - 0.01).abs() < 1e-12);
        // At the start the position ellipse is a circle of the translation variance.
        let start = gax::pga2d::Motor::translation(0.0, 0.0);
        let (here, values, _) = position_ellipse(start, cov);
        assert!(here.to_euclidean().iter().all(|v| v.abs() < 1e-12));
        assert!(
            values.iter().all(|v| (v - 0.09).abs() < 1e-12),
            "{values:?}"
        );
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws(super::draw, 0.5);
    }
}
