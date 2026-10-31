//! numga's `mechanics/xpbd`: extended position-based dynamics (XPBD) of a chain of rigid
//! links, in PGA3D.
//!
//! The projection of a rigid-body distance constraint is coordinate-free geometry:
//!
//! 1. the world anchors are the body-frame anchors moved by the motors, `m >> a`;
//! 2. their join is the constraint line, `a0 & a1`, whose norm is the Euclidean distance and
//!    whose direction is the line of action of the correcting forque;
//! 3. pulled back into each body's frame, `m << line`, the inverse inertia maps that forque to
//!    a twist step, and pairing the step with the forque gives the body's compliance along the
//!    line (its generalized inverse mass);
//! 4. the XPBD multiplier is the distance over the total compliance, and each body moves by the
//!    exponential of its share, `m (step λ (-1/2)).exp()`.
//!
//! The chain hangs from a pin, a body of infinite mass: its inverse inertia is zero, so nothing
//! moves it. Each link is six unit masses with the inertia of a solid ellipsoid. The links are
//! stepped by the shared Lie integrators (RK4 on the rate), relaxed in red-black partitions of
//! independent joints, their rates recovered from the relaxed motors, and their relative anchor
//! velocities cancelled.
//!
//! The animation shows numga's ten links released from a horizontal arc in weak gravity: the
//! chain uncurls, swings down and over while its joints stay closed to a few millimetres,
//! within numga's bound of a tenth of the spacing (the plot on the right).

use gax::pga3d::{Forque, Line, Point};

use gax_numga_examples::points::on_sphere;
use gax_numga_examples::{
    Align, Anim, Axes, Camera, Canvas, Lens, Light, Marker, Rect, Scene3, backdrop, caption,
    palette, run_with, signal,
};

use gax_numga_examples::shared::mechanics_lie as lie;

mod xpbd {
    use super::*;
    use crate::lie::Lie;
    pub use crate::lie::rigid::{F, G, Inertia, InertiaInv, M, P, R};

    /// The batched state of the chain's rigid bodies.
    #[derive(Clone)]
    pub struct Chain {
        pub motor: Vec<M>,
        /// The rates, in the body frames.
        pub rate: Vec<R>,
        /// The mass-weighted centre, in the body frame (the same for every link).
        pub first_moment: P,
        pub inertia: Inertia,
        /// Zero for a fixed body.
        pub inertia_inv: Vec<InertiaInv>,
        pub damping: f64,
        /// An ideal point: the gravitational acceleration.
        pub gravity: P,
    }

    /// A disjoint set of pairwise joints that can be relaxed simultaneously.
    #[derive(Clone)]
    pub struct Joints {
        /// The joined bodies.
        pub bodies: Vec<[usize; 2]>,
        /// The anchor points, each in its own body's frame (the same for every joint).
        pub anchors: [P; 2],
        /// Inverse stiffness, alpha.
        pub compliance: f64,
    }

    /// The forque along a unit line for each body of a pair, pulled into its frame, the twist
    /// step it causes, and the body's compliance along the line, `step & forque`.
    fn responses(motors: [M; 2], inv: [InertiaInv; 2], direction: F) -> ([R; 2], f64) {
        let mut total = 0.0;
        let steps = [0, 1].map(|i| {
            let local = motors[i] << direction;
            let step = inv[i].of(local);
            total += (step & local).s();
            step
        });
        (steps, total)
    }

    /// Close the gap between the anchors of a body pair.
    pub fn project_distance_constraint(
        motors: [M; 2],
        anchors: [P; 2],
        inv: [InertiaInv; 2],
        compliance: f64,
        dt: f64,
    ) -> [M; 2] {
        // The join of the world anchors: its norm is their distance. A closed joint has a
        // vanishing line; the small offsets keep its direction and multiplier at zero.
        let line = (motors[0] >> anchors[0]) & (motors[1] >> anchors[1]);
        let magnitude = line.norm();
        let direction = line / (magnitude + 1e-24);
        let (steps, inertial) = responses(motors, inv, direction);
        let total = compliance / (dt * dt) + inertial + 1e-24;
        // Equal and opposite multipliers, and the corrective motion by the exponential map.
        let multiplier = magnitude / total;
        [
            motors[0] * (steps[0] * (multiplier * -0.5)).exp(),
            motors[1] * (steps[1] * (-multiplier * -0.5)).exp(),
        ]
    }

    /// The line joining an anchor with its velocity under an open rate: rate to world-frame
    /// velocity line (a join, so a forque), before the motor.
    pub fn anchor_velocity(anchor: P) -> Forque<(Line,)> {
        anchor & anchor.commutator(Line::slot())
    }

    /// Cancel the relative velocity of the anchors of a body pair with an impulse along it.
    pub fn project_velocity_constraint(
        motors: [M; 2],
        rates: [R; 2],
        anchors: [P; 2],
        inv: [InertiaInv; 2],
    ) -> [R; 2] {
        let velocity = |i: usize| motors[i] >> anchor_velocity(anchors[i]).of(rates[i]);
        let forque = velocity(1) - velocity(0);
        let magnitude = forque.norm();
        let direction = forque / (magnitude + 1e-24);
        let (steps, inertial) = responses(motors, inv, direction);
        let multiplier = magnitude / (inertial + 1e-24);
        [
            rates[0] + steps[0] * multiplier,
            rates[1] + steps[1] * -multiplier,
        ]
    }

    /// The external forque in the body frame: the weight, the line through the centre of mass
    /// along gravity (their join), and the damping.
    pub fn external_forque(motor: M, rate: R, first_moment: P, gravity: P, damping: f64) -> F {
        (first_moment & (motor << gravity)) - (rate * damping).dual()
    }

    /// One XPBD step of the chain.
    pub fn step(chain: &Chain, partitions: &[Joints], dt: f64) -> Chain {
        let forque =
            |m: M, r: R| external_forque(m, r, chain.first_moment, chain.gravity, chain.damping);
        // 1. Unconstrained inertial pre-integration.
        let (mut motor, _): (Vec<M>, Vec<R>) = (0..chain.motor.len())
            .map(|b| {
                G::explicit_rk4(
                    chain.motor[b],
                    chain.rate[b],
                    chain.inertia,
                    chain.inertia_inv[b],
                    dt,
                    &forque,
                )
            })
            .unzip();
        // 2. Relax the position constraints, Gauss-Seidel over the red-black partitions.
        for joints in partitions {
            for &[a, b] in &joints.bodies {
                let [ma, mb] = project_distance_constraint(
                    [motor[a], motor[b]],
                    joints.anchors,
                    [chain.inertia_inv[a], chain.inertia_inv[b]],
                    joints.compliance,
                    dt,
                );
                (motor[a], motor[b]) = (ma, mb);
            }
        }
        // 3. Post-integration: the rates from the motor displacements.
        let motor: Vec<M> = motor.into_iter().map(|m| m.renormalize_fast()).collect();
        let mut rate: Vec<R> = chain
            .motor
            .iter()
            .zip(&motor)
            .map(|(before, after)| G::rate_between(*before, *after, dt))
            .collect();
        // 4. Resolve the velocity constraints.
        for joints in partitions {
            for &[a, b] in &joints.bodies {
                let [ra, rb] = project_velocity_constraint(
                    [motor[a], motor[b]],
                    [rate[a], rate[b]],
                    joints.anchors,
                    [chain.inertia_inv[a], chain.inertia_inv[b]],
                );
                (rate[a], rate[b]) = (ra, rb);
            }
        }
        Chain {
            motor,
            rate,
            ..chain.clone()
        }
    }

    /// Advance the chain by `dt` in equal substeps.
    pub fn advance(chain: &Chain, partitions: &[Joints], substeps: usize, dt: f64) -> Chain {
        let mut chain = chain.clone();
        for _ in 0..substeps {
            chain = step(&chain, partitions, dt / substeps as f64);
        }
        chain
    }

    /// The Euclidean separation of the two anchors of every joint: the norm of their join.
    pub fn joint_gaps(motor: &[M], partitions: &[Joints]) -> Vec<f64> {
        partitions
            .iter()
            .flat_map(|joints| {
                joints.bodies.iter().map(|&[a, b]| {
                    ((motor[a] >> joints.anchors[0]) & (motor[b] >> joints.anchors[1])).norm()
                })
            })
            .collect()
    }

    /// A link's anchors, halfway to its neighbours: on +x, then on -x.
    pub fn anchors(spacing: f64) -> [P; 2] {
        let (origin, x) = (Point::origin(), axes()[0]);
        [origin + x * (0.5 * spacing), origin - x * (0.5 * spacing)]
    }

    /// numga's chain: its links, the spacing of their joints, the semi-axes of each link's
    /// solid ellipsoid, the turn between neighbours along its starting arc, the joints'
    /// compliance, the damping and gravity.
    pub const N_LINKS: usize = 10;
    pub const LINK_SPACING: f64 = 9e-2;
    pub const SEMI_AXES: [f64; 3] = [
        0.62 * LINK_SPACING,
        0.38 * LINK_SPACING,
        0.22 * LINK_SPACING,
    ];
    pub const CURL: f64 = 0.3;
    pub const COMPLIANCE: f64 = 1e-9;
    pub const DAMPING: f64 = 1e-3;
    pub const GRAVITY: f64 = 0.5;

    /// The link's axes, x, y and z.
    pub fn axes() -> [P; 3] {
        [
            Point::direction(1.0, 0.0, 0.0),
            Point::direction(0.0, 1.0, 0.0),
            Point::direction(0.0, 0.0, 1.0),
        ]
    }

    /// The mass points of one body: six unit masses on its axes, at `semi_axes * sqrt(3 / 5)`
    /// on either side, which gives it the inertia of a solid ellipsoid with those semi-axes,
    /// three different moments. The first three are on the positive side.
    pub fn cloud(semi_axes: [f64; 3]) -> Vec<P> {
        // A solid ellipsoid's radius of gyration along an axis, over its semi-axis.
        #[allow(clippy::disallowed_methods)]
        let gyration = (3.0f64 / 5.0).sqrt();
        let origin = Point::origin();
        [1.0, -1.0]
            .into_iter()
            .flat_map(|sign| {
                (semi_axes.into_iter().zip(axes()))
                    .map(move |(a, axis)| origin + axis * (sign * a * gyration))
            })
            .collect()
    }

    /// A chain of `n_links` solid ellipsoidal links hanging from a pin at the origin, in
    /// gravity along -z. The pin is body 0, with zero inverse inertia, an infinite mass, so
    /// nothing moves it; the links follow. Neighbours are joined at their ends, half the
    /// spacing along x from their centres, and the chain starts on a horizontal arc: each body
    /// is its neighbour carried half the spacing along x, turned by `curl` about z (numga's
    /// `exp(xy curl / 2)`), and carried on; the bodies are the powers of that motor. The joints
    /// are split red and black: the even joints connect 0-1, 2-3, ... and the odd ones 1-2,
    /// 3-4, ..., so the bodies within one partition are independent.
    #[allow(clippy::too_many_arguments)]
    pub fn chain(
        n_links: usize,
        spacing: f64,
        semi_axes: [f64; 3],
        curl: f64,
        compliance: f64,
        damping: f64,
        gravity: f64,
    ) -> (Chain, Vec<Joints>) {
        let n_bodies = n_links + 1;
        let points = cloud(semi_axes);
        let (inertia, inertia_inv) = G::inertia_from_points(&points);
        let first_moment = points.iter().copied().sum();
        // From one body to the next along the arc: numga's `exp(xw spacing / 4)`, half the
        // spacing along x (`xw = -e01`), around the turn `exp(xy curl / 2)` (`xy = e12`).
        let half_shift = Line::new(0.0, 0.0, 0.0, -0.25 * spacing, 0.0, 0.0).exp();
        let turn = Line::new(0.0, 0.0, 0.5 * curl, 0.0, 0.0, 0.0).exp();
        let neighbour = (half_shift * turn * half_shift).log();
        let state = Chain {
            motor: (0..n_bodies)
                .map(|k| neighbour.gp(k as f64).exp())
                .collect(),
            rate: vec![Line::zero(); n_bodies],
            first_moment,
            inertia,
            // The pin has infinite mass: zero inverse inertia.
            inertia_inv: (0..n_bodies)
                .map(|i| {
                    if i == 0 {
                        InertiaInv::zero()
                    } else {
                        inertia_inv
                    }
                })
                .collect(),
            damping,
            gravity: Point::direction(0.0, 0.0, -gravity),
        };
        // Joint j connects the +x anchor of body j to the -x anchor of body j + 1.
        let anchors = anchors(spacing);
        let partitions = [0, 1]
            .map(|parity| Joints {
                bodies: (parity..n_links).step_by(2).map(|j| [j, j + 1]).collect(),
                anchors,
                compliance,
            })
            .to_vec();
        (state, partitions)
    }

    /// The motors over time and the gap at every joint, numga's chain stepped `steps` times
    /// by `dt` in `substeps`.
    pub fn swinging_chain(
        n_links: usize,
        steps: usize,
        substeps: usize,
        dt: f64,
    ) -> (Vec<Vec<M>>, Vec<Vec<f64>>) {
        let (mut state, partitions) = chain(
            n_links,
            LINK_SPACING,
            SEMI_AXES,
            CURL,
            COMPLIANCE,
            DAMPING,
            GRAVITY,
        );
        let mut motors = vec![state.motor.clone()];
        let mut gaps = vec![joint_gaps(&state.motor, &partitions)];
        for _ in 0..steps {
            state = advance(&state, &partitions, substeps, dt);
            gaps.push(joint_gaps(&state.motor, &partitions));
            motors.push(state.motor.clone());
        }
        (motors, gaps)
    }

    /// The largest joint gap over a run.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn largest_gap(gaps: &[Vec<f64>]) -> f64 {
        gaps.iter().flatten().fold(0.0, |a: f64, &b| a.max(b))
    }

    /// How far the pin strays from the origin over a run: the norm of the join of the two.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn pin_drift(motors: &[Vec<M>]) -> f64 {
        let origin = Point::origin();
        (motors.iter())
            .map(|m| ((m[0] >> origin) & origin).norm())
            .fold(0.0, f64::max)
    }
}

use xpbd::*;

/// numga's run: 4 s in 150 steps of 4 substeps.
const SECONDS: f32 = 4.0;
const STEPS: usize = 150;
const SUBSTEPS: usize = 4;
const DT: f64 = 4.0 / STEPS as f64;

/// The motors and joint gaps per step.
type History = (Vec<Vec<M>>, Vec<Vec<f64>>);

fn scene() -> History {
    swinging_chain(N_LINKS, STEPS, SUBSTEPS, DT)
}

/// A solid ellipsoid with the link's semi-axes, placed by its motor, lit.
fn ellipsoid(s: &mut Scene3, m: M, colour: Light) {
    let origin = Point::origin();
    let [a, b, c] = SEMI_AXES;
    s.surface(
        |u, v| {
            let d = on_sphere(
                core::f64::consts::TAU * f64::from(u),
                core::f64::consts::PI * (f64::from(v) - 0.5),
            );
            m >> (origin + Point::direction(a * d.e1(), b * d.e2(), c * d.e3()))
        },
        16,
        8,
        |_, _| colour,
        0.92,
        None,
    );
}

fn draw(c: &mut Canvas, (motors, gaps): &History, t: f32) {
    backdrop(c);
    let screen = c.rect();
    let (w, h) = (screen.width(), screen.height());
    let k = ((f64::from(t.rem_euclid(SECONDS)) / DT) as usize).min(motors.len() - 1);
    caption(
        c,
        "XPBD: A CHAIN OF SOLID LINKS",
        "CONSTRAINTS AS LINES, PROJECTED BY MOTORS (PGA3D)",
    );
    // The chain, in 3D, from a camera turning slowly about it: its azimuth swings with a wave,
    // once per run.
    let left = w * 0.6;
    let azimuth = -1.0 + 0.25 * signal::wave(core::f32::consts::TAU * t / SECONDS);
    let cam = Camera::orbit(
        Rect::new(0.0, 60.0, left, h),
        Point::xyz(0.05, -0.05, -0.36),
        3.3,
        azimuth,
        0.3,
        Lens::Perspective(0.42),
    );
    let mut s = Scene3::new(cam);
    // The floor a spacing below the lowest the chain can hang.
    let depth = -(N_LINKS as f64 + 1.0) * LINK_SPACING;
    let floor = |x: f64, y: f64| Point::xyz(x, y, depth);
    for i in -6..=8 {
        let x = f64::from(i) * 0.1;
        s.seg(floor(x, -0.7), floor(x, 0.5), 1.0, palette::GRID.faded(0.5));
    }
    for j in -7..=5 {
        let y = f64::from(j) * 0.1;
        s.seg(floor(-0.6, y), floor(0.8, y), 1.0, palette::GRID.faded(0.5));
    }
    // The free end's path so far.
    let [end, _] = anchors(LINK_SPACING);
    let trail: Vec<P> = motors[..=k].iter().map(|m| m[N_LINKS] >> end).collect();
    s.polyline(&trail, 1.5, palette::YELLOW.faded(0.6));
    // The links in two alternating colours, the pin dark.
    let colours = [palette::ORANGE, palette::SKY];
    for (i, &m) in motors[k].iter().enumerate().skip(1) {
        ellipsoid(&mut s, m, colours[i % 2]);
    }
    s.dot(Point::<()>::origin(), Marker::Square, 9.0, palette::RED);
    s.draw(c);
    // The joint gaps over time, as numga's figure: the largest and the mean.
    let right = Rect::new(left, 0.0, w, h);
    let ax = Axes::new(
        right.inset(54.0, 80.0, 20.0, 60.0),
        [0.0, SECONDS],
        [1e-8, 3e-2],
    )
    .log_y();
    ax.frame(c, "JOINT GAPS (METRES)", "TIME (S)", "");
    // The chart's points: (time, gap).
    let at = gax::pga2d::Point::xy;
    let series = |f: &dyn Fn(&[f64]) -> f64| -> Vec<_> {
        gaps[1..=k.max(1)]
            .iter()
            .enumerate()
            .map(|(j, g)| at((j + 1) as f64 * DT, f(g).max(1e-30)))
            .collect()
    };
    let largest = series(&|g| g.iter().fold(0.0, |a: f64, &b| a.max(b)));
    let mean = series(&|g| g.iter().sum::<f64>() / g.len() as f64);
    ax.polyline(c, &largest, 1.6, palette::RED);
    ax.dashed(c, &mean, 1.6, 5.0, palette::SKY);
    // numga's bound: the joints stay closed to within a tenth of the spacing, hidden in the
    // links' overlapping ends.
    let bound = 0.1 * LINK_SPACING;
    let level = [at(0.0, bound), at(f64::from(SECONDS), bound)];
    ax.dashed(c, &level, 1.0, 3.0, palette::GRID);
    let label = at(0.1, bound * 1.4);
    ax.text(
        c,
        label,
        "10% OF THE SPACING",
        9.0,
        palette::GRID,
        Align::Left,
    );
    ax.legend(c, &[("LARGEST", palette::RED), ("MEAN", palette::SKY)]);
}

fn main() {
    run_with(Anim::new("xpbd", SECONDS).size(960, 540), scene, draw);
}

#[cfg(test)]
mod tests {
    use super::lie::Lie;
    use super::xpbd::*;
    use gax::pga3d::{Line, Motor, Point};

    /// Repeated projection closes the gap between two displaced anchors.
    #[test]
    fn distance_projection_closes_a_joint() {
        let (chain, partitions) = chain(1, 0.1, SEMI_AXES, 0.0, 0.0, 1e-3, 0.5);
        let joints = &partitions[0];
        assert!(joint_gaps(&chain.motor, &partitions)[0] < 1e-12);
        // Displace link 1 by (2, -1, 3) mm.
        let mut motor = chain.motor.clone();
        motor[1] = motor[1] * Motor::translation(Point::direction(0.002, -0.001, 0.003));
        assert!(joint_gaps(&motor, &partitions)[0] > 0.003);
        // The anchor does not respond parallel to the gap, so each projection removes a
        // fraction of it.
        let mut pair = [motor[0], motor[1]];
        let inv = [chain.inertia_inv[0], chain.inertia_inv[1]];
        for _ in 0..30 {
            pair = project_distance_constraint(pair, joints.anchors, inv, joints.compliance, 0.01);
        }
        assert!(((pair[0] >> joints.anchors[0]) & (pair[1] >> joints.anchors[1])).norm() < 1e-4);
        // The pin has infinite mass: only link 1 moved.
        assert_eq!(pair[0].c, motor[0].c);
    }

    #[test]
    fn velocity_projection_cancels_relative_anchor_velocity() {
        let (chain, partitions) = chain(1, 0.1, SEMI_AXES, 0.0, 1e-9, 1e-3, 0.5);
        let joints = &partitions[0];
        let motors = [chain.motor[0], chain.motor[1]];
        // numga adds `xw * 0.5 + yz * 0.3` to link 1: a shift along x and a turn about x.
        let rates = [
            chain.rate[0],
            chain.rate[1] + Line::new(0.3, 0.0, 0.0, -0.5, 0.0, 0.0),
        ];
        let relative = |r: [R; 2]| {
            let v = |i: usize| motors[i] >> anchor_velocity(joints.anchors[i]).of(r[i]);
            (v(0) - v(1)).norm()
        };
        assert!(relative(rates) > 0.01);
        let inv = [chain.inertia_inv[0], chain.inertia_inv[1]];
        let resolved = project_velocity_constraint(motors, rates, joints.anchors, inv);
        assert!(relative(resolved) < 1e-6, "{}", relative(resolved));
    }

    /// The joints stay closed (numga's 1% of the spacing for this short run) and the pin
    /// fixed; and the chain falls: the free end has dropped.
    #[test]
    fn numpy_chain_keeps_its_joints_closed() {
        let (motors, gaps) = swinging_chain(4, 20, 2, 0.02);
        assert_eq!((motors.len(), gaps[0].len()), (21, 4));
        assert!(
            largest_gap(&gaps) < 1e-2 * LINK_SPACING,
            "{}",
            largest_gap(&gaps)
        );
        assert!(pin_drift(&motors) < 1e-6);
        let end = (motors[20][4] >> Point::origin()).to_euclidean();
        assert!(end[2] < -1e-3, "{end:?}");
    }

    /// numga's scenario checks: through the whole swing the joints stay closed to within a
    /// tenth of the spacing, hidden in the links' overlapping ends, and the pin never leaves
    /// the origin.
    #[test]
    fn the_scene_keeps_its_joints_closed_and_its_pin_fixed() {
        let (motors, gaps) = super::scene();
        assert!(
            largest_gap(&gaps) < 0.1 * LINK_SPACING,
            "{}",
            largest_gap(&gaps)
        );
        assert!(pin_drift(&motors) < 1e-6);
        // The swing reproduces numga's: the centres of links 1, 5 and 10 halfway through and at
        // the end, and the largest gap (2.363e-3), to numga's printed digits.
        let numga = [
            (
                75,
                [
                    [0.04002, -0.01042, -0.04424],
                    [0.02227, -0.10324, -0.39476],
                    [-0.01957, 0.00946, -0.79664],
                ],
            ),
            (
                150,
                [
                    [0.02283, 0.02153, -0.03294],
                    [-0.16257, 0.27232, -0.19116],
                    [-0.04154, 0.62859, -0.09752],
                ],
            ),
        ];
        for (k, centres) in numga {
            for (b, expected) in [1, 5, 10].into_iter().zip(centres) {
                let c = (motors[k][b] >> Point::origin()).to_euclidean();
                assert!(
                    (0..3).all(|i| (c[i] - expected[i]).abs() < 1e-5),
                    "{k} {b} {c:?}"
                );
            }
        }
        assert!((largest_gap(&gaps) - 2.363007e-3).abs() < 1e-9);
    }

    /// The chain starts on numga's arc: each body its neighbour moved half the spacing along
    /// x, turned by the curl about z, and moved on, so neighbours' joints coincide.
    #[test]
    fn the_chain_starts_closed_on_an_arc() {
        let (state, partitions) = chain(N_LINKS, LINK_SPACING, SEMI_AXES, CURL, 0.0, 0.0, 0.5);
        assert!(
            joint_gaps(&state.motor, &partitions)
                .iter()
                .all(|g| *g < 1e-15)
        );
        // Every body is as far from the next: the chord of the arc.
        let centre = |k: usize| state.motor[k] >> Point::origin();
        let chord = (centre(0) & centre(1)).norm();
        for k in 1..N_LINKS {
            assert!(((centre(k) & centre(k + 1)).norm() - chord).abs() < 1e-14);
        }
    }

    /// Damping dissipates: without gravity a spinning chain loses kinetic energy every step.
    #[test]
    fn damping_dissipates() {
        let (mut state, partitions) = chain(2, 0.09, SEMI_AXES, 0.0, 1e-9, 1e-2, 0.0);
        state.rate[2] = Line::new(0.0, 0.0, 2.0, 0.0, 0.0, 0.0);
        let energy = |s: &Chain| {
            s.rate
                .iter()
                .map(|r| G::kinetic_energy(*r, s.inertia))
                .sum::<f64>()
        };
        let mut last = energy(&state);
        assert!(last > 0.0);
        for _ in 0..20 {
            state = advance(&state, &partitions, 2, 0.02);
            let e = energy(&state);
            assert!(e < last, "{e} {last}");
            last = e;
        }
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws_with(super::draw, &super::scene(), 0.5);
    }
}
