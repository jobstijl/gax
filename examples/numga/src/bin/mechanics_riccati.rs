//! numga's `mechanics/riccati`: finite-horizon feedback for docking a vessel, with every cost a
//! quadratic form (PGA2D). The remaining displacement and turn is a twist (a point in PGA2D),
//! with pose `exp(-error / 2)`; the thrusters' total force and torque is a forque (a line).
//! Strong drag makes a push a velocity, so one step adds the mobility of the push to the error.
//! A cost is a scalar with two open twist slots, and filling both with maps pulls it back through
//! them: `value(dynamics, actuation)` is the future cost seen from the present error and push,
//! with no transposes. The Riccati recursion pulls the future cost back one step at a time and
//! minimizes over the push; its solve returns a feedback map from twists to forques. The
//! animation docks the vessel with a cheap and an expensive effort, the thrusters' commands as
//! arrows, and the level set of the remaining cost (heading zero) shrinking to the dock as the
//! deadline nears.

use gax::Unit;
use gax::pga2d::{Line, Motor, Point, Scalar};
use gax_numga_examples::{
    Align, Anim, Axes, Canvas, Marker, backdrop, canvas, caption, palette, plot, run,
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

    /// A two-slot form pulled back through a map in each slot: numga's `form(a, b)`. gax's `of`
    /// fills the first slot only, so the second is brought to the front, filled, and the slots
    /// put back in order. (Values fill both slots with `form.of(a).of(b)`.)
    macro_rules! pull {
        ($form:expr, $a:expr, $b:expr) => {
            $form.of($a).at::<1>().of($b).swap()
        };
    }

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
                let control_cost: EffortCost = effort_cost + pull!(value, actuation, actuation);
                // The push that cancels the cost's derivative, for every error at once.
                let cross: Scalar<(Point, Line), f64> = pull!(value, dynamics, actuation);
                let feedback: Feedback = -control_cost.solve(cross);
                // The error's cost now, and its future cost through the step that feedback takes.
                let closed: Dynamics = dynamics + actuation.of(feedback);
                value = state_cost + pull!(value, dynamics, closed);
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

    /// The model: the thrusters' resistance (a forque to the twist of least weighted effort),
    /// the state cost, the effort cost of each case, the dynamics and the actuation.
    #[allow(clippy::type_complexity)]
    pub fn model() -> (
        Point<(Line,), f64>,
        StateCost,
        [EffortCost; 2],
        Dynamics,
        Actuation,
    ) {
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
        let dynamics = Point::slot();
        let actuation = drag.inverse().gp(DT);
        (resistance, state_cost, effort_cost, dynamics, actuation)
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
        let (resistance, state_cost, effort_cost, dynamics, actuation) = model();
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

    /// The twist whose pose is the translation by `(x, y)`.
    pub fn offset(x: f64, y: f64) -> P {
        Point::translation_twist(x, y).gp(-2.0)
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

fn xy(p: P) -> [f32; 2] {
    let [x, y] = p.to_euclidean();
    [x as f32, y as f32]
}

/// The corners of a box around every hull position of both approaches and the dock.
fn limits() -> ([f32; 2], [f32; 2]) {
    static L: OnceLock<([f32; 2], [f32; 2])> = OnceLock::new();
    *L.get_or_init(|| {
        let mut lo = [f32::MAX; 2];
        let mut hi = [f32::MIN; 2];
        let hull = hull();
        for case in cases() {
            for e in &case.errors {
                for p in &hull {
                    let q = xy(pose(*e) >> *p);
                    for i in 0..2 {
                        lo[i] = lo[i].min(q[i]);
                        hi[i] = hi[i].max(q[i]);
                    }
                }
            }
        }
        (lo, hi)
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
    let target: Vec<[f32; 2]> = hull.iter().map(|p| xy(*p)).collect();
    let mut closed_target = target.clone();
    closed_target.push(target[0]);
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
        let ax = Axes::equal(
            rect,
            [(lo[0] + hi[0]) * 0.5, (lo[1] + hi[1]) * 0.5],
            (hi[1] - lo[1]).max((hi[0] - lo[0]) * (rect[3] - rect[1]) / (rect[2] - rect[0]))
                * 0.5
                * 1.1,
        );
        // The level set of the cost still to pay, over translations of the hull with heading
        // zero: everywhere from which the rest of the approach costs 0.15.
        let remaining = if s >= STEPS as f32 { 1 } else { STEPS - k };
        let value = case.values[remaining - 1];
        ax.contour(
            c,
            |x, y| {
                let q = offset(f64::from(x), f64::from(y));
                value.of(q).of(q).s() as f32
            },
            90,
            0.15,
            1.5,
            palette::green(),
        );
        ax.dashed(c, &closed_target, 1.2, 5.0, palette::grid(), 1.0);
        let path: Vec<[f32; 2]> = case
            .errors
            .iter()
            .map(|e| xy(pose(*e) >> centre()))
            .collect();
        ax.polyline(c, &path, 1.0, palette::sky(), 0.35);
        // The pose between steps: along the screw from one to the next.
        let m = Motor::interpolate(
            pose(case.errors[k]),
            pose(case.errors[(k + 1).min(STEPS)]),
            if s >= STEPS as f32 { 1.0 } else { frac },
        );
        let body: Vec<[f32; 2]> = hull.iter().map(|p| xy(m >> *p)).collect();
        ax.fill(c, &body, canvas::scale(palette::blue(), 0.9), 0.5);
        let mut outline = body.clone();
        outline.push(body[0]);
        ax.polyline(c, &outline, 1.6, palette::sky(), 1.0);
        // The total push the feedback asks for, a forque: its line of action.
        if s < STEPS as f32 {
            let f = case.feedbacks[k].of(case.errors[k]);
            let (a, b, c0) = (f.e1(), f.e2(), f.e0());
            let n2 = a * a + b * b;
            if n2 > 1e-12 {
                let foot = [(-c0 * a / n2) as f32, (-c0 * b / n2) as f32];
                ax.axline(c, foot, [-b as f32, a as f32], 1.0, palette::orange(), 0.35);
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
            let a = xy(base);
            let b = [a[0] + force.e20() as f32, a[1] + force.e01() as f32];
            ax.scatter(c, &[a], Marker::Square, 6.0, palette::ink(), 1.0);
            ax.arrow(c, a, b, 2.0, 8.0, palette::orange());
        }
        ax.text(
            c,
            [ax.x[0] + 0.05, ax.y[1] - 0.1],
            label,
            12.0,
            palette::ink(),
            Align::Left,
        );
        ax.text(
            c,
            [ax.x[0] + 0.05, ax.y[1] - 0.22],
            &format!("{remaining} STEPS LEFT: COST 0.15 LEVEL"),
            10.0,
            palette::green(),
            Align::Left,
        );
    }
    caption(
        c,
        "RICCATI: DOCKING WITH COSTS AS QUADRATIC FORMS",
        "FEEDBACK FROM TWISTS TO FORQUES BY THE BACKWARD RECURSION; ARROWS: THRUSTER COMMANDS (PGA2D)",
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
    use gax::pga2d::{Line, Point, Scalar};

    type Mat = Vec<Vec<f64>>;

    fn zeros(r: usize, c: usize) -> Mat {
        vec![vec![0.0; c]; r]
    }
    fn eye(n: usize) -> Mat {
        let mut m = zeros(n, n);
        for (i, row) in m.iter_mut().enumerate() {
            row[i] = 1.0;
        }
        m
    }
    fn mul(a: &Mat, b: &Mat) -> Mat {
        let mut m = zeros(a.len(), b[0].len());
        for i in 0..a.len() {
            for k in 0..b.len() {
                for j in 0..b[0].len() {
                    m[i][j] += a[i][k] * b[k][j];
                }
            }
        }
        m
    }
    fn tr(a: &Mat) -> Mat {
        (0..a[0].len())
            .map(|j| a.iter().map(|r| r[j]).collect())
            .collect()
    }
    fn add(a: &Mat, b: &Mat) -> Mat {
        a.iter()
            .zip(b)
            .map(|(x, y)| x.iter().zip(y).map(|(u, v)| u + v).collect())
            .collect()
    }
    /// Solve `a x = b` by Gaussian elimination with partial pivoting.
    fn solve(a: &Mat, b: &Mat) -> Mat {
        let n = a.len();
        let mut m: Mat = a
            .iter()
            .zip(b)
            .map(|(r, s)| [r.clone(), s.clone()].concat())
            .collect();
        for col in 0..n {
            let p = (col..n)
                .max_by(|&i, &j| m[i][col].abs().total_cmp(&m[j][col].abs()))
                .unwrap_or(col);
            m.swap(col, p);
            for r in 0..n {
                if r != col {
                    let f = m[r][col] / m[col][col];
                    let pivot = m[col].clone();
                    for (x, y) in m[r].iter_mut().zip(pivot) {
                        *x -= f * y;
                    }
                }
            }
        }
        m.iter()
            .enumerate()
            .map(|(i, r)| r[n..].iter().map(|v| v / r[i]).collect())
            .collect()
    }

    /// Minimize over every control at once, independently of the Riccati recursion: the optimal
    /// controls per step (`[steps][controls][dimension]`, for each unit initial error) and the
    /// optimal cost's matrix.
    fn condensed_control(
        dynamics: &Mat,
        actuation: &Mat,
        state_cost: &Mat,
        effort_cost: &Mat,
        terminal: &Mat,
        steps: usize,
    ) -> (Vec<Mat>, Mat) {
        let (dim, controls) = (actuation.len(), actuation[0].len());
        let mut initial_map = eye(dim);
        let mut control_map = zeros(dim, steps * controls);
        let mut hessian = zeros(steps * controls, steps * controls);
        for s in 0..steps {
            for i in 0..controls {
                for j in 0..controls {
                    hessian[s * controls + i][s * controls + j] = effort_cost[i][j];
                }
            }
        }
        let mut cross = zeros(steps * controls, dim);
        let mut initial_cost = zeros(dim, dim);
        for step in 0..steps {
            hessian = add(
                &hessian,
                &mul(&tr(&control_map), &mul(state_cost, &control_map)),
            );
            cross = add(
                &cross,
                &mul(&tr(&control_map), &mul(state_cost, &initial_map)),
            );
            initial_cost = add(
                &initial_cost,
                &mul(&tr(&initial_map), &mul(state_cost, &initial_map)),
            );
            initial_map = mul(dynamics, &initial_map);
            control_map = mul(dynamics, &control_map);
            for r in 0..dim {
                for c in 0..controls {
                    control_map[r][step * controls + c] += actuation[r][c];
                }
            }
        }
        hessian = add(
            &hessian,
            &mul(&tr(&control_map), &mul(terminal, &control_map)),
        );
        cross = add(
            &cross,
            &mul(&tr(&control_map), &mul(terminal, &initial_map)),
        );
        initial_cost = add(
            &initial_cost,
            &mul(&tr(&initial_map), &mul(terminal, &initial_map)),
        );
        let optimum: Mat = solve(&hessian, &cross)
            .into_iter()
            .map(|r| r.into_iter().map(|v| -v).collect())
            .collect();
        let value = add(&initial_cost, &mul(&tr(&cross), &optimum));
        let actions = (0..steps)
            .map(|s| optimum[s * controls..(s + 1) * controls].to_vec())
            .collect();
        (actions, value)
    }

    fn arr3(m: &Mat) -> [[f64; 3]; 3] {
        core::array::from_fn(|i| core::array::from_fn(|j| m[i][j]))
    }

    /// The recursion matches a single dense optimization over every control, with general
    /// dynamics and actuation. A map's coefficients are rows of outputs (`c[o][i]`), a form's
    /// its matrix on coefficients.
    #[test]
    fn riccati_matches_a_single_dense_optimization() {
        let steps = 9;
        let dynamics = [
            vec![
                vec![1.03, 0.14, -0.04],
                vec![0.0, 0.96, 0.12],
                vec![0.03, 0.0, 1.01],
            ],
            vec![
                vec![0.93, -0.18, 0.05],
                vec![0.09, 1.04, -0.02],
                vec![0.0, 0.07, 0.91],
            ],
        ];
        let actuation = [
            vec![
                vec![0.22, 0.05, 0.0],
                vec![-0.03, 0.19, 0.04],
                vec![0.01, 0.02, 0.16],
            ],
            vec![
                vec![0.17, -0.03, 0.02],
                vec![0.04, 0.23, 0.0],
                vec![0.01, -0.04, 0.18],
            ],
        ];
        let state_cost = [
            vec![
                vec![2.0, 0.3, -0.2],
                vec![0.3, 1.4, 0.1],
                vec![-0.2, 0.1, 0.9],
            ],
            vec![
                vec![1.1, -0.1, 0.2],
                vec![-0.1, 2.3, -0.3],
                vec![0.2, -0.3, 1.7],
            ],
        ];
        let effort_cost = [
            vec![
                vec![0.8, 0.1, 0.0],
                vec![0.1, 0.6, -0.1],
                vec![0.0, -0.1, 0.5],
            ],
            vec![
                vec![0.5, -0.05, 0.1],
                vec![-0.05, 0.7, 0.0],
                vec![0.1, 0.0, 1.0],
            ],
        ];
        for case in 0..2 {
            let terminal: Mat = state_cost[case]
                .iter()
                .map(|r| r.iter().map(|v| 3.0 * v).collect())
                .collect();
            let d: Dynamics = Point::from_coeffs(arr3(&dynamics[case]));
            let a: Actuation = Point::from_coeffs(arr3(&actuation[case]));
            let q: StateCost = Scalar::from_coeffs([arr3(&state_cost[case])]);
            let r: EffortCost = Scalar::from_coeffs([arr3(&effort_cost[case])]);
            let qf: StateCost = Scalar::from_coeffs([arr3(&terminal)]);
            let (values, mut gains): (Vec<_>, Vec<_>) =
                riccati(qf, d, a, q, r, steps).into_iter().unzip();
            gains.reverse();
            let (actions, value) = condensed_control(
                &dynamics[case],
                &actuation[case],
                &state_cost[case],
                &effort_cost[case],
                &terminal,
                steps,
            );
            let last = values[steps - 1].c[0];
            for i in 0..3 {
                for j in 0..3 {
                    assert!((last[i][j] - value[i][j]).abs() < 1e-10);
                }
            }
            // Three unit initial errors test the policy on the whole state space.
            for e in 0..3 {
                let mut initial = [0.0; 3];
                initial[e] = 1.0;
                let errors = rollout(Point::from_coeffs(initial), d, a, &gains);
                for (s, (g, x)) in gains.iter().zip(&errors).enumerate() {
                    let push = g.of(*x);
                    for (p, row) in push.c.iter().zip(&actions[s]) {
                        assert!((p - row[e]).abs() < 1e-10);
                    }
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
                for (a, b) in supplied.c.iter().zip(requested.c) {
                    assert!((a - b).abs() < 1e-10);
                }
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
        let p = pose(offset(0.3, -0.2)) >> centre();
        let [x, y] = p.to_euclidean();
        assert!((x - 0.3).abs() < 1e-12 && (y + 0.2).abs() < 1e-12);
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
