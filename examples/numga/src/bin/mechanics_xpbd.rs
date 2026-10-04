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
//! A fixed body has infinite mass: its inverse inertia is zero, so nothing moves it. The links
//! are stepped by the shared Lie integrators (RK4 on the rate), relaxed in red-black
//! partitions of independent joints, their rates recovered from the relaxed motors, and their
//! relative anchor velocities cancelled.
//!
//! The animation shows a chain of six links released horizontally in weak gravity: it swings
//! down and back while its joints stay closed to a fraction of a millimetre, well within
//! numga's bound of 1% of the spacing (the plot on the right).

use gax::pga3d::{Line, Motor, Point};
use gax_light::fade;
use gax_numga_examples::{
    Align, Anim, Axes, Camera, Canvas, Lens, Marker, Rect, Scene3, backdrop, caption, palette, run,
};

#[path = "../shared/mechanics_lie.rs"]
mod lie;

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
    /// velocity line, before the motor.
    pub fn anchor_velocity(anchor: P) -> Line<(Line,), f64> {
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
    pub fn anchors(distance: f64) -> [P; 2] {
        let (origin, x) = (Point::xyz(0.0, 0.0, 0.0), axes()[0]);
        [origin + x * (0.5 * distance), origin - x * (0.5 * distance)]
    }

    /// The links' spacing in numga's scene.
    pub const LINK_SPACING: f64 = 9e-2;

    /// The link's axes, x, y and z.
    pub fn axes() -> [P; 3] {
        [
            Point::direction(1.0, 0.0, 0.0),
            Point::direction(0.0, 1.0, 0.0),
            Point::direction(0.0, 0.0, 1.0),
        ]
    }

    /// The mass points of one link: six unit masses on its axes, at `size / 3`, `2 size / 3`
    /// and `size` along x, y and z on either side, so its three moments of inertia differ.
    /// The first three are on the positive side.
    pub fn cloud(size: f64) -> Vec<P> {
        let origin = Point::xyz(0.0, 0.0, 0.0);
        [1.0, -1.0]
            .into_iter()
            .flat_map(|sign| {
                (1..=3)
                    .zip(axes())
                    .map(move |(k, axis)| origin + axis * (sign * f64::from(k) * size / 3.0))
            })
            .collect()
    }

    /// A chain of links along x, the first fixed at the origin, hanging in gravity along -z.
    /// Adjacent links are joined halfway between their centres. The joints are split red and
    /// black: the even joints connect 0-1, 2-3, ... and the odd ones 1-2, 3-4, ..., so the
    /// bodies within one partition are independent.
    pub fn chain(
        bodies: usize,
        distance: f64,
        size: f64,
        compliance: f64,
        damping: f64,
        gravity: f64,
    ) -> (Chain, Vec<Joints>) {
        let points = cloud(size);
        let (inertia, inertia_inv) = G::inertia_from_points(&points);
        let first_moment = points.iter().copied().sum();
        let state = Chain {
            motor: (0..bodies)
                .map(|i| Motor::translation(i as f64 * distance, 0.0, 0.0))
                .collect(),
            rate: vec![Line::zero(); bodies],
            first_moment,
            inertia,
            // The fixed link has infinite mass: zero inverse inertia.
            inertia_inv: (0..bodies)
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
        // Joint j connects the +x anchor of link j to the -x anchor of link j + 1.
        let anchors = anchors(distance);
        let partitions = [0, 1]
            .map(|parity| Joints {
                bodies: (parity..bodies - 1)
                    .step_by(2)
                    .map(|j| [j, j + 1])
                    .collect(),
                anchors,
                compliance,
            })
            .to_vec();
        (state, partitions)
    }

    /// The motors over time and the gap at every joint, from numga's scene: links 5 cm big and
    /// 9 cm apart, stiff joints, light damping, gravity 0.5. Checks, as numga's scenario does,
    /// that the joints stay closed to within 1% of the spacing and the fixed link never moves.
    pub fn swinging_chain(
        links: usize,
        steps: usize,
        substeps: usize,
        dt: f64,
    ) -> (Vec<Vec<M>>, Vec<Vec<f64>>) {
        let (mut state, partitions) = chain(links, LINK_SPACING, 5e-2, 1e-9, 1e-3, 0.5);
        let mut motors = vec![state.motor.clone()];
        let mut gaps = vec![joint_gaps(&state.motor, &partitions)];
        for _ in 0..steps {
            state = advance(&state, &partitions, substeps, dt);
            gaps.push(joint_gaps(&state.motor, &partitions));
            motors.push(state.motor.clone());
        }
        // --- checks
        let origin = Point::xyz(0.0, 0.0, 0.0);
        let worst = gaps.iter().flatten().fold(0.0, |a: f64, &b| a.max(b));
        assert!(worst < 1e-2 * LINK_SPACING, "{worst}");
        for m in &motors {
            assert!(((m[0] >> origin) & origin).norm() < 1e-6);
        }
        (motors, gaps)
    }
}

use xpbd::*;

/// numga's scene (six links, 0.02 per step in five substeps) stepped for 6 s instead of 1.2,
/// so that the chain swings down and back.
const LINKS: usize = 6;
const DT: f64 = 0.02;
const SECONDS: f32 = 6.0;

/// The motors and joint gaps per step.
type History = (Vec<Vec<M>>, Vec<Vec<f64>>);

fn scene() -> &'static History {
    static SCENE: std::sync::OnceLock<History> = std::sync::OnceLock::new();
    SCENE.get_or_init(|| swinging_chain(LINKS, (f64::from(SECONDS) / DT).round() as usize, 5, DT))
}

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let (motors, gaps) = scene();
    let screen = c.rect();
    let (w, h) = (screen.width(), screen.height());
    let u = h / 540.0;
    let k = ((f64::from(t.rem_euclid(SECONDS)) / DT) as usize).min(motors.len() - 1);
    caption(
        c,
        "XPBD: A CHAIN OF RIGID LINKS",
        "CONSTRAINTS AS LINES, PROJECTED BY MOTORS (PGA3D)",
    );
    // The chain, in 3D, from a camera turning slowly about it.
    let left = w * 0.6;
    let azimuth = -1.25 + 0.25 * (core::f32::consts::TAU * t / SECONDS).sin();
    let cam = Camera::orbit(
        left as usize,
        c.height,
        Point::xyz(0.0, 0.0, -0.2),
        3.0,
        azimuth,
        0.3,
        Lens::Perspective(0.42),
    );
    let mut s = Scene3::new(cam);
    // The ground grid under the pivot.
    let floor = |x: f64, y: f64| Point::xyz(x, y, -0.62);
    for i in -4..=6 {
        let x = f64::from(i) * 0.1;
        s.seg(
            floor(x, -0.3),
            floor(x, 0.3),
            1.0,
            fade(palette::grid(), 0.6),
        );
    }
    for j in -3..=3 {
        let y = f64::from(j) * 0.1;
        s.seg(
            floor(-0.4, y),
            floor(0.6, y),
            1.0,
            fade(palette::grid(), 0.6),
        );
    }
    // The free end's path so far.
    let [end, start] = anchors(LINK_SPACING);
    let trail: Vec<P> = motors[..=k].iter().map(|m| m[LINKS - 1] >> end).collect();
    s.polyline(&trail, 1.5, fade(palette::yellow(), 0.6));
    let cloud = cloud(5e-2);
    let colours = [palette::orange(), palette::sky(), palette::green()];
    let origin = Point::xyz(0.0, 0.0, 0.0);
    for (i, &m) in motors[k].iter().enumerate() {
        // Each link's mass points, on its three axes, as three bars.
        for axis in 0..3 {
            s.seg(
                m >> cloud[axis],
                m >> cloud[axis + 3],
                3.0 * u,
                colours[axis],
            );
        }
        // The link's anchors and the rod between them.
        s.seg(m >> start, m >> end, 1.5 * u, fade(palette::ink(), 0.8));
        let col = if i == 0 {
            palette::red()
        } else {
            palette::ink()
        };
        s.dot(m >> origin, Marker::Dot, 5.0 * u, col);
    }
    s.dot(origin, Marker::Square, 9.0 * u, palette::red());
    s.draw(c);
    // The joint gaps over time, as numga's figure: the largest and the mean.
    let right = Rect::new(left, 0.0, w, h);
    let ax = Axes::new(
        right.inset(54.0 * u, 80.0 * u, 20.0 * u, 60.0 * u),
        [0.0, SECONDS],
        [1e-7, 1e-2],
    )
    .log_y();
    ax.frame(c, "JOINT GAPS", "TIME (S)", "METRES");
    // The chart's points: (time, gap).
    let at = gax::pga2d::Point::<(), f64>::xy;
    let series = |f: &dyn Fn(&[f64]) -> f64| -> Vec<_> {
        gaps[1..=k.max(1)]
            .iter()
            .enumerate()
            .map(|(j, g)| at((j + 1) as f64 * DT, f(g).max(1e-30)))
            .collect()
    };
    let largest = series(&|g| g.iter().fold(0.0, |a: f64, &b| a.max(b)));
    let mean = series(&|g| g.iter().sum::<f64>() / g.len() as f64);
    ax.polyline(c, &largest, 1.6, palette::red());
    ax.dashed(c, &mean, 1.6, 5.0, palette::sky());
    // numga's bound: the joints stay closed to within 1% of the spacing.
    let bound = 1e-2 * LINK_SPACING;
    let level = [at(0.0, bound), at(f64::from(SECONDS), bound)];
    ax.dashed(c, &level, 1.0, 3.0, palette::grid());
    let label = at(0.1, bound * 1.4);
    ax.text(
        c,
        label,
        "1% OF THE SPACING",
        9.0 * u,
        palette::grid(),
        Align::Left,
    );
    ax.legend(c, &[("LARGEST", palette::red()), ("MEAN", palette::sky())]);
}

fn main() {
    run(Anim::new("xpbd", SECONDS).size(960, 540), draw);
}

#[cfg(test)]
mod tests {
    use super::lie::Lie;
    use super::xpbd::*;
    use gax::pga3d::{Line, Motor, Point};

    /// Repeated projection closes the gap between two displaced anchors.
    #[test]
    fn distance_projection_closes_a_joint() {
        let (chain, partitions) = chain(2, 0.1, 5e-2, 0.0, 1e-3, 0.5);
        let joints = &partitions[0];
        assert!(joint_gaps(&chain.motor, &partitions)[0] < 1e-12);
        // Displace link 1 by (2, -1, 3) mm.
        let mut motor = chain.motor.clone();
        motor[1] = motor[1] * Motor::translation(0.002, -0.001, 0.003);
        assert!(joint_gaps(&motor, &partitions)[0] > 0.003);
        // The anchor does not respond parallel to the gap, so each projection removes a
        // fraction of it.
        let mut pair = [motor[0], motor[1]];
        let inv = [chain.inertia_inv[0], chain.inertia_inv[1]];
        for _ in 0..30 {
            pair = project_distance_constraint(pair, joints.anchors, inv, joints.compliance, 0.01);
        }
        assert!(((pair[0] >> joints.anchors[0]) & (pair[1] >> joints.anchors[1])).norm() < 1e-4);
        // The fixed link has infinite mass: only link 1 moved.
        assert_eq!(pair[0].c, motor[0].c);
    }

    #[test]
    fn velocity_projection_cancels_relative_anchor_velocity() {
        let (chain, partitions) = chain(2, 0.1, 5e-2, 1e-9, 1e-3, 0.5);
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

    /// The scenario checks closed joints and a fixed first link.
    #[test]
    fn chain_swings_with_closed_joints() {
        let (motors, gaps) = swinging_chain(4, 20, 2, 0.02);
        assert_eq!((motors.len(), gaps[0].len()), (21, 3));
        // And it falls: the free end has dropped.
        let end = (motors[20][3] >> Point::xyz(0.0, 0.0, 0.0)).to_euclidean();
        assert!(end[2] < -1e-3, "{end:?}");
    }

    /// Damping dissipates: without gravity a spinning chain loses kinetic energy every step.
    #[test]
    fn damping_dissipates() {
        let (mut state, partitions) = chain(3, 0.09, 5e-2, 1e-9, 1e-2, 0.0);
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
        gax_numga_examples::app::assert_draws(super::draw, 0.5);
    }
}
