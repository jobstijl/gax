//! numga's `mechanics/riccati`: finite-horizon feedback for docking a vessel, with every cost a
//! quadratic form (PGA2D). The remaining displacement and turn is a twist (a point in PGA2D),
//! with pose `exp(-error / 2)`; the thrusters' total force and torque is a forque (a line).
//! Strong drag makes a push a velocity, so one step adds the mobility of the push to the error.
//! A cost is a scalar with two open twist slots, and filling both with maps pulls it back through
//! them: `value.of_both(dynamics, actuation)` is the future cost seen from the present error and
//! push, with no transposes. The Riccati recursion pulls the future cost back one step at a time and
//! minimizes over the push; its solve returns a feedback map from twists to forques. The
//! animation docks the vessel with a cheap and an expensive effort, the thrusters' commands as
//! arrows, and the level set of the remaining cost (heading zero) shrinking to the dock as the
//! deadline nears.

use gax::Unit;
use gax::pga2d::{Line, Motor, Point, Scalar};
use gax_numga_examples::{
    Align, Anim, Axes, Canvas, Marker, Point2, backdrop, canvas, caption, palette, plot, run,
};
use std::sync::OnceLock;

mod riccati {
    use super::*;

    pub type P = Point<(), f64>;
    pub type L = Line<(), f64>;
    /// `Scalar <- (Twist, Twist)`.
    pub type StateCost = Scalar<(Point, Point), f64>;
    /// `Scalar <- (Forque, Forque)`.
    pub type EffortCost = Scalar<(Line, Line), f64>;
    /// `Twist <- Twist`.
    pub type Dynamics = Point<(Point,), f64>;
    /// `Twist <- Forque`.
    pub type Actuation = Point<(Line,), f64>;
    /// `Forque <- Twist`.
    pub type Feedback = Line<(Point,), f64>;

    /// The cost still to pay and the feedback, one more step back from `value` each time.
    pub fn riccati(
        mut value: StateCost,
        dynamics: Dynamics,
        actuation: Actuation,
        state_cost: StateCost,
        effort_cost: EffortCost,
        steps: usize,
    ) -> Vec<(StateCost, Feedback)> {
        (0..steps)
            .map(|_| {
                // A push costs effort now and moves the error whose cost is paid next.
                let control_cost: EffortCost = effort_cost + value.of_both(actuation, actuation);
                // The push that cancels the cost's derivative, for every error at once.
                let cross: Scalar<(Point, Line), f64> = value.of_both(dynamics, actuation);
                let feedback: Feedback = -control_cost.solve(cross);
                // The error's cost now, and its future cost through the step that feedback takes.
                let closed: Dynamics = dynamics + actuation.of(feedback);
                value = state_cost + value.of_both(dynamics, closed);
                (value, feedback)
            })
            .collect()
    }

    /// The initial error and its evolution under the feedbacks.
    pub fn rollout(
        initial: P,
        dynamics: Dynamics,
        actuation: Actuation,
        feedbacks: &[Feedback],
    ) -> Vec<P> {
        let mut error = initial;
        let mut out = vec![error];
        for f in feedbacks {
            error = dynamics.of(error) + actuation.of(f.of(error));
            out.push(error);
        }
        out
    }

    pub const EFFORT_SCALES: [f64; 2] = [0.05, 5.0];
    pub const THRUSTER_WEIGHTS: [f64; 3] = [1.0, 2.0, 3.0];
    /// Drag along the forward and sideways directions through the centre, and against turning.
    pub const DRAG_WEIGHTS: [f64; 3] = [2.0, 3.0, 1.0];
    /// Rows: bow and stern; columns: forward and sideways displacement.
    pub const TRACKING_WEIGHTS: [[f64; 2]; 2] = [[4.0, 8.0], [1.0, 2.0]];
    pub const DT: f64 = 0.1;
    pub const STEPS: usize = 72;
    /// How much stiffer the pose is priced at the deadline than along the way.
    pub const LANDING: f64 = 1e6;

    pub fn hull() -> [P; 5] {
        [
            Point::xy(-0.45, -0.23),
            Point::xy(0.28, -0.23),
            Point::xy(0.55, 0.0),
            Point::xy(0.28, 0.23),
            Point::xy(-0.45, 0.23),
        ]
    }
    pub fn centre() -> P {
        Point::xy(0.0, 0.0)
    }
    /// Forward and sideways, points at infinity.
    pub fn axes() -> [P; 2] {
        [Point::direction(1.0, 0.0), Point::direction(0.0, 1.0)]
    }
    pub fn mounts() -> [P; 3] {
        [
            Point::xy(-0.3, 0.2),
            Point::xy(-0.3, -0.2),
            Point::xy(0.35, 0.0),
        ]
    }
    pub fn directions() -> [P; 3] {
        [
            Point::direction(1.0, 0.0),
            Point::direction(1.0, 0.0),
            Point::direction(0.0, 1.0),
        ]
    }
    pub fn tracking_points() -> [P; 2] {
        [Point::xy(0.45, 0.0), Point::xy(-0.35, 0.0)]
    }
    /// numga's `0.4 xw + 0.5 yw + 0.35 xy`: with `w = e0`, `xw = -e01` and `yw = e20`.
    pub fn initial() -> P {
        Point::new(0.5, -0.4, 0.35)
    }

    /// The thrusters' lines of force: a mount joined with its ideal direction, lever arm
    /// included.
    pub fn thrusters() -> [L; 3] {
        let (m, d) = (mounts(), directions());
        core::array::from_fn(|k| m[k] & d[k])
    }

    /// The tracking lines: each reads one displacement of a hull point.
    pub fn tracking_lines() -> [[L; 2]; 2] {
        let (p, a) = (tracking_points(), axes());
        core::array::from_fn(|i| core::array::from_fn(|j| p[i] & a[j]))
    }

    /// The vessel and its costs.
    pub struct Model {
        /// The thrusters' resistance: a forque to the twist of least weighted effort.
        pub resistance: Point<(Line,), f64>,
        pub state_cost: StateCost,
        /// The effort cost of each case.
        pub effort_cost: [EffortCost; 2],
        pub dynamics: Dynamics,
        pub actuation: Actuation,
    }

    pub fn model() -> Model {
        let thrusters = thrusters();
        let authority = thrusters
            .iter()
            .zip(THRUSTER_WEIGHTS)
            .fold(Line::zero(), |s: Feedback, (t, w)| {
                s + (*t * (*t & Point::slot())).gp(1.0 / w)
            });
        let resistance = authority.inverse();
        let effort: EffortCost = Line::slot() & resistance.of(Line::slot());
        let effort_cost = EFFORT_SCALES.map(|s| effort.gp(s * DT));

        // Each tracking line's reading squared is that displacement's cost.
        let mut state_cost: StateCost = Scalar::zero();
        for (row, weights) in tracking_lines().iter().zip(TRACKING_WEIGHTS) {
            for (l, w) in row.iter().zip(weights) {
                let readout = *l & Point::slot();
                state_cost += (readout * readout).gp(w * DT);
            }
        }

        // Drag resists moving along either axis through the centre, and turning (the ideal
        // line); its inverse turns a push into a velocity twist.
        let [x, y] = axes();
        let drag_lines = [centre() & x, centre() & y, x & y];
        let drag = drag_lines
            .iter()
            .zip(DRAG_WEIGHTS)
            .fold(Line::zero(), |s: Feedback, (l, w)| {
                s + (*l * (*l & Point::slot())).gp(w)
            });
        // Without a push the vessel stays put.
        Model {
            resistance,
            state_cost,
            effort_cost,
            dynamics: Point::slot(),
            actuation: drag.inverse().gp(DT),
        }
    }

    /// One case's docking: the costs for one through `STEPS` remaining actions, the policy in
    /// time order, the errors, and the thrusters' commands.
    pub struct Docking {
        pub values: Vec<StateCost>,
        pub feedbacks: Vec<Feedback>,
        pub errors: Vec<P>,
        pub commands: Vec<[f64; 3]>,
    }

    pub fn docking() -> [Docking; 2] {
        let Model {
            resistance,
            state_cost,
            effort_cost,
            dynamics,
            actuation,
        } = model();
        let thrusters = thrusters();
        effort_cost.map(|effort| {
            // From the deadline back, where whatever error is left is priced LANDING times
            // as stiffly.
            let (values, mut feedbacks): (Vec<_>, Vec<_>) = riccati(
                state_cost.gp(LANDING),
                dynamics,
                actuation,
                state_cost,
                effort,
                STEPS,
            )
            .into_iter()
            .unzip();
            feedbacks.reverse();
            let errors = rollout(initial(), dynamics, actuation, &feedbacks);
            // Each requested push allocated to the thrusters at least weighted squared command.
            let commands = feedbacks
                .iter()
                .zip(&errors)
                .map(|(f, e)| {
                    let push = resistance.of(f.of(*e));
                    core::array::from_fn(|j| (thrusters[j] & push).s() / THRUSTER_WEIGHTS[j])
                })
                .collect();
            Docking {
                values,
                feedbacks,
                errors,
                commands,
            }
        })
    }

    /// The pose of an error: `exp(-error / 2)`.
    pub fn pose(error: P) -> Unit<Motor<(), f64>> {
        error.gp(-0.5).exp()
    }

    /// The twist whose pose is the translation carrying the dock's centre to `to`.
    pub fn offset(to: P) -> P {
        let d = to.unitized() - centre();
        Point::translation_twist(d.e20(), d.e01()).gp(-2.0)
    }
}

use riccati::*;

fn cases() -> &'static [Docking; 2] {
    static D: OnceLock<[Docking; 2]> = OnceLock::new();
    D.get_or_init(docking)
}

/// Seconds per step, and the pause at the dock before the loop restarts.
const STEP: f32 = 0.08;
const HOLD: f32 = 1.4;

/// The lower left and upper right corners of a box around every hull position of both
/// approaches and the dock.
fn limits() -> (P, P) {
    static L: OnceLock<(P, P)> = OnceLock::new();
    *L.get_or_init(|| {
        let mut lo = [f64::MAX; 2];
        let mut hi = [f64::MIN; 2];
        let hull = hull();
        for case in cases() {
            for e in &case.errors {
                for p in &hull {
                    let q = (pose(*e) >> *p).to_euclidean();
                    for i in 0..2 {
                        lo[i] = lo[i].min(q[i]);
                        hi[i] = hi[i].max(q[i]);
                    }
                }
            }
        }
        (Point::xy(lo[0], lo[1]), Point::xy(hi[0], hi[1]))
    })
}

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let (w, h) = (c.width as f32, c.height as f32);
    let seconds = STEP * STEPS as f32 + HOLD;
    let s = (t.rem_euclid(seconds) / STEP).min(STEPS as f32);
    let k = (s.floor() as usize).min(STEPS - 1);
    let frac = f64::from(s - k as f32);
    let hull = hull();
    let closed = |pts: &[P]| [pts, &pts[..1]].concat();
    let [x, y] = axes();
    let horizon = x & y;
    let labels = ["LOWER EFFORT PENALTY", "HIGHER EFFORT PENALTY"];
    let (mounts, directions) = (mounts(), directions());
    // Shared limits: a square around every pose of both approaches and the dock.
    let (lo, hi) = limits();
    for (i, (case, label)) in cases().iter().zip(labels).enumerate() {
        let rect = plot::inset(
            [w * i as f32 / 2.0, 0.0, w * (i + 1) as f32 / 2.0, h],
            w * 0.02,
            h * 0.17,
            w * 0.02,
            h * 0.04,
        );
        let ax = Axes::fitting(rect, [lo, hi], 1.1);
        // The level set of the cost still to pay, over translations of the hull with heading
        // zero: everywhere from which the rest of the approach costs 0.15.
        let remaining = if s >= STEPS as f32 { 1 } else { STEPS - k };
        let value = case.values[remaining - 1];
        ax.contour(
            c,
            |p| {
                let q = offset(p.map_coefs(f64::from));
                value.of(q).of(q).s() as f32
            },
            90,
            0.15,
            1.5,
            palette::green(),
        );
        ax.dashed(c, &closed(&hull), 1.2, 5.0, palette::grid(), 1.0);
        let path: Vec<P> = case.errors.iter().map(|e| pose(*e) >> centre()).collect();
        ax.polyline(c, &path, 1.0, palette::sky(), 0.35);
        // The pose between steps: along the screw from one to the next.
        let m = Motor::interpolate(
            pose(case.errors[k]),
            pose(case.errors[(k + 1).min(STEPS)]),
            if s >= STEPS as f32 { 1.0 } else { frac },
        );
        let body: Vec<P> = hull.iter().map(|p| m >> *p).collect();
        ax.fill(c, &body, canvas::scale(palette::blue(), 0.9), 0.5);
        ax.polyline(c, &closed(&body), 1.6, palette::sky(), 1.0);
        // The total push the feedback asks for, a forque: its line of action, through the foot
        // of the perpendicular from the dock (the meet of the forque with the perpendicular),
        // along its point at infinity (its meet with the horizon).
        if s < STEPS as f32 {
            let f = case.feedbacks[k].of(case.errors[k]);
            if f.norm() > 1e-6 {
                let foot = (f | centre()) ^ f;
                ax.axline(c, foot, f ^ horizon, 1.0, palette::orange(), 0.35);
            }
        }
        // The thrusters' commands: a signed command reverses its ideal direction before the
        // pose carries it into the world.
        let command = if s >= STEPS as f32 {
            [0.0; 3]
        } else {
            case.commands[k]
        };
        for j in 0..3 {
            let base = m >> mounts[j];
            let force = m >> directions[j].gp(command[j] * 0.04);
            ax.scatter(c, &[base], Marker::Square, 6.0, palette::ink(), 1.0);
            ax.arrow(c, base, base + force, 2.0, 8.0, palette::orange());
        }
        ax.text(
            c,
            Point2::xy(ax.x[0] + 0.05, ax.y[1] - 0.1),
            label,
            12.0,
            palette::ink(),
            Align::Left,
        );
        ax.text(
            c,
            Point2::xy(ax.x[0] + 0.05, ax.y[1] - 0.22),
            &format!("{remaining} STEPS LEFT: COST 0.15 LEVEL"),
            10.0,
            palette::green(),
            Align::Left,
        );
    }
    caption(
        c,
        "RICCATI: DOCKING WITH COSTS AS QUADRATIC FORMS",
        "FEEDBACK FROM TWISTS TO FORQUES, SOLVED BACKWARDS. ARROWS: THRUST (PGA2D)",
    );
}

fn main() {
    run(
        Anim::new("riccati", STEP * STEPS as f32 + HOLD).size(960, 540),
        draw,
    );
}

#[cfg(test)]
mod tests {
    use super::riccati::*;
    use gax::ApproxEq;
    use gax::pga2d::{Line, Point, Scalar};

    /// The recursion's policy is the best over the whole horizon, with general dynamics and
    /// actuation. The total cost is convex in the pushes (the effort cost is positive definite),
    /// so it is least where its derivative with respect to every push vanishes. That derivative
    /// is the effort form of the push plus the costate pulled back through the actuation, where
    /// the costate (the derivative of the cost still to come with respect to the error) is
    /// carried back step by step through the dynamics. And what the rollout pays is what the
    /// recursion's value predicts. Forms here are matrices on coefficients, and maps rows of
    /// outputs (`c[o][i]`).
    #[test]
    fn riccati_policy_is_optimal_over_the_horizon() {
        let steps = 9;
        let dynamics = [
            [[1.03, 0.14, -0.04], [0.0, 0.96, 0.12], [0.03, 0.0, 1.01]],
            [[0.93, -0.18, 0.05], [0.09, 1.04, -0.02], [0.0, 0.07, 0.91]],
        ];
        let actuation = [
            [[0.22, 0.05, 0.0], [-0.03, 0.19, 0.04], [0.01, 0.02, 0.16]],
            [[0.17, -0.03, 0.02], [0.04, 0.23, 0.0], [0.01, -0.04, 0.18]],
        ];
        let state_cost = [
            [[2.0, 0.3, -0.2], [0.3, 1.4, 0.1], [-0.2, 0.1, 0.9]],
            [[1.1, -0.1, 0.2], [-0.1, 2.3, -0.3], [0.2, -0.3, 1.7]],
        ];
        let effort_cost = [
            [[0.8, 0.1, 0.0], [0.1, 0.6, -0.1], [0.0, -0.1, 0.5]],
            [[0.5, -0.05, 0.1], [-0.05, 0.7, 0.0], [0.1, 0.0, 1.0]],
        ];
        for case in 0..2 {
            let d: Dynamics = Point::from_coeffs(dynamics[case]);
            let a: Actuation = Point::from_coeffs(actuation[case]);
            let q: StateCost = Scalar::from_coeffs([state_cost[case]]);
            let r: EffortCost = Scalar::from_coeffs([effort_cost[case]]);
            let terminal = q.gp(3.0);
            let (values, mut gains): (Vec<_>, Vec<_>) =
                riccati(terminal, d, a, q, r, steps).into_iter().unzip();
            gains.reverse();
            // Six initial errors fix the value's quadratic form; each tests the whole policy.
            let starts: [P; 6] = [
                Point::new(1.0, 0.0, 0.0),
                Point::new(0.0, 1.0, 0.0),
                Point::new(0.0, 0.0, 1.0),
                Point::new(1.0, 1.0, 0.0),
                Point::new(0.0, 1.0, 1.0),
                Point::new(1.0, 0.0, 1.0),
            ];
            for start in starts {
                let errors = rollout(start, d, a, &gains);
                let pushes: Vec<L> = gains.iter().zip(&errors).map(|(g, x)| g.of(*x)).collect();
                let paid = pushes
                    .iter()
                    .zip(&errors)
                    .map(|(u, x)| q.fill(*x).s() + r.fill(*u).s())
                    .sum::<f64>()
                    + terminal.fill(errors[steps]).s();
                let predicted = values[steps - 1].fill(start).s();
                assert!(
                    (paid - predicted).abs() < 1e-10 * predicted,
                    "{paid} {predicted}"
                );
                // Half the derivatives of the cost (the forms are symmetric).
                let mut costate: Scalar<(Point,), f64> = terminal.of(errors[steps]);
                for step in (0..steps).rev() {
                    let gradient: Scalar<(Line,), f64> = r.of(pushes[step]) + costate.of(a);
                    let size = r.of(pushes[step]).max_abs_diff(&Scalar::zero());
                    assert!(
                        gradient.max_abs_diff(&Scalar::zero()) < 1e-10 * size.max(1.0),
                        "{case} {step} {gradient:?}"
                    );
                    costate = q.of(errors[step]) + costate.of(d);
                }
            }
        }
    }

    /// The docking realizes its Bellman cost in tracking displacements and thruster commands,
    /// step by step, and both approaches reach the dock.
    #[test]
    fn docking_realizes_its_bellman_cost() {
        let lines = tracking_lines();
        for (case, d) in docking().iter().enumerate() {
            let tracking: Vec<f64> = d
                .errors
                .iter()
                .map(|e| {
                    let mut sum = 0.0;
                    for (row, weights) in lines.iter().zip(TRACKING_WEIGHTS) {
                        for (l, w) in row.iter().zip(weights) {
                            sum += (*l & *e).s().powi(2) * w;
                        }
                    }
                    sum * DT
                })
                .collect();
            let running: Vec<f64> = (0..STEPS)
                .map(|s| {
                    let effort: f64 = d.commands[s]
                        .iter()
                        .zip(THRUSTER_WEIGHTS)
                        .map(|(c, w)| c * c * w)
                        .sum();
                    tracking[s] + effort * EFFORT_SCALES[case] * DT
                })
                .collect();
            let landing = tracking[STEPS] * LANDING;
            let cost = |v: &StateCost, e: P| v.of(e).of(e).s();
            let predicted = cost(&d.values[STEPS - 1], d.errors[0]);
            let total: f64 = running.iter().sum::<f64>() + landing;
            assert!((total - predicted).abs() < 1e-9, "{total} {predicted}");
            // What is left to pay falls by exactly each step's running cost.
            let remaining: Vec<f64> = (0..STEPS)
                .map(|s| cost(&d.values[STEPS - 1 - s], d.errors[s]))
                .collect();
            for s in 0..STEPS - 1 {
                assert!((remaining[s] - remaining[s + 1] - running[s]).abs() < 1e-10);
            }
            assert!((remaining[STEPS - 1] - running[STEPS - 1] - landing).abs() < 1e-10);
            // Within a millimetre of the dock at the deadline.
            assert!(d.errors[STEPS].c.iter().all(|v| v.abs() < 1e-3));
        }
    }

    /// The commands add up to the requested forque, and a thruster's work on a twist is its
    /// force dotted into the velocity the twist gives its mount: the motor convention agrees
    /// with the thrust arrows.
    #[test]
    fn thruster_commands_reproduce_the_feedback_forque_and_work() {
        let thrusters = thrusters();
        for d in docking() {
            for s in 0..STEPS {
                let supplied = thrusters
                    .iter()
                    .zip(d.commands[s])
                    .fold(Line::zero(), |sum: L, (t, c)| sum + t.gp(c));
                let requested = d.feedbacks[s].of(d.errors[s]);
                assert!(supplied.max_abs_diff(&requested) < 1e-10);
            }
        }
        for trial in [Point::new(0.2, -0.3, 0.15), Point::new(-0.1, 0.25, -0.12)] {
            for ((m, dir), t) in mounts().iter().zip(directions()).zip(thrusters) {
                let v = m.commutator(trial);
                let power = dir.e20() * v.e20() + dir.e01() * v.e01();
                assert!((power - (t & trial).s()).abs() < 1e-10, "{power}");
            }
        }
    }

    /// The pose of an offset twist is that translation.
    #[test]
    fn offsets_translate() {
        let to = Point::xy(0.3, -0.2);
        let p = pose(offset(to)) >> centre();
        assert!((p.unitized() & to).norm() < 1e-12);
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws(super::draw, 0.5);
    }
}
