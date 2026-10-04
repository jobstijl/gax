//! numga's `mechanics/robot_arm`: a three-joint arm in PGA3D, where joints, rates and the
//! Jacobian are all lines. A joint's configuration is its axis line scaled by the angle, a
//! bivector, and so is its rate. Forward kinematics exponentiates those bivectors in order; the
//! frame in which each joint acts is the product of the joints before it. Carrying the axis lines
//! into those frames gives the Jacobian, so it is never derived: the tip velocity of a twist is
//! its commutator with the tip, and the generalized force a forque exerts on a joint is the
//! pairing of the carried axis with that forque. The animation shows the arm following a tilted
//! loop by least-squares steps, two per target, with the joint angles plotted alongside.

use gax::Unit;
use gax::motions::{Motions, Pga3d};
use gax::pga3d::{Line, Motor, Plane, Point};
use gax_numga_examples::{
    Align, Anim, Axes, Camera, Canvas, Lens, Marker, Point2, Rgb, Scene3, backdrop, caption,
    palette, plot, run,
};
use std::sync::OnceLock;

mod arm {
    use super::*;

    pub type L = Line<(), f64>;
    pub type P = Point<(), f64>;
    pub type M = Unit<Motor<(), f64>>;

    /// The number of targets along the loop, one per frame of numga's animation.
    pub const TARGETS: usize = 72;

    /// The tip pose and the frame before each joint's rotation: the product of the joints
    /// before it.
    pub fn forward_kinematics(joints: &[L; 3]) -> (M, [M; 3]) {
        let mut pose = Pga3d::identity();
        let mut frames = [pose; 3];
        for (frame, joint) in frames.iter_mut().zip(joints) {
            *frame = pose;
            // numga writes `exp(joint / 2)`; gax's join orients the axis lines the other way
            // (the pseudoscalar's orientation, ADR-009), so the same right-handed turn is
            // `exp(-joint / 2)`, as in `Motor::rotation`.
            pose = pose * joint.gp(-0.5).exp();
        }
        (pose, frames)
    }

    /// The tip velocity under joint rates, and the work per unit rate a forque does on each
    /// joint (the joint torques).
    pub fn mechanics(
        joints: &[L; 3],
        axis: &[L; 3],
        rates: &[L; 3],
        tip_home: P,
        forque: L,
    ) -> (P, [f64; 3]) {
        let (pose, frames) = forward_kinematics(joints);
        let tip = pose >> tip_home;
        // Each joint's rate carried into its frame, summed: the world twist of the tip link.
        // Its commutator with the tip is the tip's velocity (the tip first: the motor is
        // `exp(-twist / 2)`, so the derivative of its sandwich is `tip × twist`).
        let twist = frames
            .iter()
            .zip(rates)
            .fold(Line::zero(), |sum: L, (f, r)| sum + (*f >> *r));
        let velocity = tip.commutator(twist);
        // Each carried axis paired with the forque: the work per unit joint rate.
        let torques = core::array::from_fn(|k| ((frames[k] >> axis[k]) & forque).s());
        (velocity, torques)
    }

    /// Least-squares steps towards `target`: the joints, the frame after each joint (the link
    /// motors) and the tip.
    pub fn inverse_kinematics(
        mut joints: [L; 3],
        axis: &[L; 3],
        tip_home: P,
        target: P,
        iterations: usize,
    ) -> ([L; 3], [M; 3], P) {
        for _ in 0..iterations {
            let (pose, frames) = forward_kinematics(&joints);
            let tip = pose >> tip_home;
            // The tip velocity per unit rate of each joint: the Jacobian's columns, as points
            // at infinity. The three joint rates are the components of a `vga3d::Vector`, so
            // the Jacobian is the map from it whose images of the basis are the columns.
            let columns: [P; 3] = core::array::from_fn(|k| tip.commutator(frames[k] >> axis[k]));
            let jacobian = Point::<(gax::vga3d::Vector,), f64>::from_images(columns);
            // The joint increments whose tip velocities best sum to the remaining error, applied
            // along the home-pose axes.
            let w = jacobian.lstsq(target - tip);
            for (k, j) in joints.iter_mut().enumerate() {
                *j += axis[k].gp(w.c[k]);
            }
        }
        let (pose, frames) = forward_kinematics(&joints);
        (joints, [frames[1], frames[2], pose], pose >> tip_home)
    }

    /// The joint axes in the home pose (a yaw about `z` at the base, then two pitch joints about
    /// `y`), the tip in the home pose, and a rest configuration (axes times angles).
    pub fn arm() -> ([L; 3], P, [L; 3]) {
        let origin = Point::xyz(0.0, 0.0, 0.0);
        let elbow = Point::xyz(0.0, 0.0, 1.0);
        let wrist = Point::xyz(0.0, 0.0, 2.0);
        // Each axis is a line: the join of a point with a direction.
        let axis = [
            origin & Point::direction(0.0, 0.0, 1.0),
            elbow & Point::direction(0.0, 1.0, 0.0),
            wrist & Point::direction(0.0, 1.0, 0.0),
        ];
        let tip_home = Point::xyz(0.0, 0.0, 3.0);
        let rest = [axis[0].gp(0.3), axis[1].gp(0.5), axis[2].gp(0.8)];
        (axis, tip_home, rest)
    }

    /// Target `k` along a loop in front of the arm: a circle of radius 0.5 about the vertical
    /// through `(1, 1)`, traced clockwise from above by turns about that axis, lifted onto the
    /// saddle `z = 1.2 + 2.4 (x - 1) (y - 1)` (so its height swings twice per loop, by 0.3).
    pub fn loop_target(k: usize) -> P {
        let centre = Point::xyz(1.0, 1.0, 0.0);
        let turn = Motor::rotation(
            centre & Point::direction(0.0, 0.0, 1.0),
            -core::f64::consts::TAU * k as f64 / TARGETS as f64,
        );
        let below = turn >> Point::xyz(1.0, 1.5, 0.0);
        // `(x - 1) (y - 1)`: its signed distances from the planes x = 1 and y = 1, its joins.
        let across = Plane::from_normal([1.0, 0.0, 0.0], 1.0) & below;
        let along = Plane::from_normal([0.0, 1.0, 0.0], 1.0) & below;
        let height = 1.2 + 2.4 * across.s() * along.s();
        below + Point::direction(0.0, 0.0, height)
    }

    /// The corners of slender boxes along `z`, one per unit link, in the home pose; corner
    /// `4 i + 2 j + k` is at `(±width, ±depth, k)` (minus first).
    pub fn link_boxes(width: f64, depth: f64) -> [[P; 8]; 3] {
        core::array::from_fn(|link| {
            core::array::from_fn(|c| {
                let x = if c & 4 == 0 { -width } else { width };
                let y = if c & 2 == 0 { -depth } else { depth };
                let z = (c & 1) as f64 + link as f64;
                Point::xyz(x, y, z)
            })
        })
    }

    /// The tip velocity and the joint torques at rest, the velocity checked against a finite
    /// difference of forward kinematics.
    pub fn statics() -> (P, [f64; 3]) {
        let (axis, tip_home, rest) = arm();
        let rates = [axis[0].gp(0.5), axis[1].gp(-1.0), axis[2].gp(0.25)];
        // A force at a point is a forque: the join of the point with the weighted direction.
        let forque = Point::xyz(1.0, 0.0, 3.0) & Point::direction(0.0, 2.0, -1.0);
        let (velocity, torques) = mechanics(&rest, &axis, &rates, tip_home, forque);

        // The tip velocity is the derivative of forward kinematics along the joint rates.
        let step = 1e-6;
        let ahead: [L; 3] = core::array::from_fn(|k| rest[k] + rates[k].gp(step));
        let ahead = forward_kinematics(&ahead).0 >> tip_home;
        let here = forward_kinematics(&rest).0 >> tip_home;
        let predicted = here + velocity.gp(step);
        assert!((ahead & predicted).norm() < 1e-9);
        (velocity, torques)
    }

    /// The angle of each joint: the joint projected on its axis, `(joint | axis) / (axis | axis)`.
    /// A Euclidean line squares to minus its squared norm, `axis | axis == -axis ~axis`.
    pub fn angles(joints: &[L; 3], axis: &[L; 3]) -> [f64; 3] {
        core::array::from_fn(|k| -(joints[k] | axis[k]).s() / axis[k].norm_squared())
    }

    /// From rest onto the start of the loop: the remaining tip error and the joint angles.
    pub fn homing() -> (f64, [f64; 3]) {
        let (axis, tip_home, rest) = arm();
        let target = loop_target(0);
        let (joints, _, tip) = inverse_kinematics(rest, &axis, tip_home, target, 8);
        // The distance is the norm of the join of the two points.
        let error = (target & tip).norm();
        assert!(error < 1e-8);
        (error, angles(&joints, &axis))
    }

    /// One tracked state: the joints, the link motors, the target and the tip.
    #[derive(Clone, Copy, Debug)]
    pub struct State {
        pub joints: [L; 3],
        pub links: [M; 3],
        pub target: P,
        pub tip: P,
    }

    /// Home onto the loop, then follow it with two steps per target.
    pub fn tracking() -> Vec<State> {
        let (axis, tip_home, rest) = arm();
        let (mut joints, _, _) = inverse_kinematics(rest, &axis, tip_home, loop_target(0), 8);
        (0..TARGETS)
            .map(|k| {
                let target = loop_target(k);
                let (j, links, tip) = inverse_kinematics(joints, &axis, tip_home, target, 2);
                joints = j;
                State {
                    joints,
                    links,
                    target,
                    tip,
                }
            })
            .collect()
    }
}

use arm::*;

/// The loop's length in seconds: a tenth of a second per target.
const SECONDS: f32 = 7.2;

fn states() -> &'static [State] {
    static S: OnceLock<Vec<State>> = OnceLock::new();
    S.get_or_init(tracking)
}

/// The box faces by corner index (`4 i + 2 j + k`).
const FACES: [[usize; 4]; 6] = [
    [0, 1, 3, 2],
    [4, 6, 7, 5],
    [0, 4, 5, 1],
    [2, 3, 7, 6],
    [0, 2, 6, 4],
    [1, 5, 7, 3],
];

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let (w, h) = (c.width as f32, c.height as f32);
    let states = states();
    let n = states.len();
    let s = (t / SECONDS).rem_euclid(1.0) * n as f32;
    let i = (s.floor() as usize).min(n - 1);
    let frac = f64::from(s - i as f32);
    let (a, b) = (states[i], states[(i + 1) % n]);
    // Between solves, each link moves along the screw between its two solved poses.
    let links: [M; 3] = core::array::from_fn(|k| Motor::interpolate(a.links[k], b.links[k], frac));
    let target = a.target + (b.target - a.target).gp(frac);
    let (axis, tip_home, _) = arm();

    // The arm in 3D on the left, the camera swinging gently.
    let scene_w = (w * 0.62) as usize;
    let phase = core::f32::consts::TAU * t / SECONDS;
    let cam = Camera::orbit(
        scene_w,
        c.height,
        Point::xyz(0.6, 0.2, 1.2),
        6.2,
        -0.87 + 0.35 * phase.sin(),
        0.38,
        Lens::Perspective(0.62),
    );
    let mut sc = Scene3::new(cam);
    // The floor: a grid on z = 0.
    let floor = |x: f64, y: f64| Point::xyz(x, y, 0.0);
    for k in 0..=8 {
        let v = f64::from(k) * 0.375;
        let (x, y) = (v - 1.0, v - 1.5);
        sc.seg(floor(x, -1.5), floor(x, 1.5), 1.0, palette::grid(), 0.6);
        sc.seg(floor(-1.0, y), floor(2.0, y), 1.0, palette::grid(), 0.6);
    }
    // The loop of targets, faint.
    let lp: Vec<P> = (0..=n).map(|k| loop_target(k % n)).collect();
    sc.polyline(&lp, 1.0, palette::red(), 0.35);
    // The tip's trail over the last half loop, fading.
    let trail = n / 2;
    for back in 0..trail {
        let k1 = (i + n - back) % n;
        let k0 = (k1 + n - 1) % n;
        let alpha = 0.9 * (1.0 - back as f32 / trail as f32);
        sc.seg(
            states[k0].tip,
            states[k1].tip,
            2.0,
            palette::yellow(),
            alpha,
        );
    }
    // The links, as lit translucent boxes with their edges.
    let boxes = link_boxes(0.15, 0.06);
    let colours = [palette::sky(), palette::blue(), palette::purple()];
    for ((corners, m), colour) in boxes.iter().zip(links).zip(colours) {
        let p: Vec<P> = corners.iter().map(|q| m >> *q).collect();
        for f in FACES {
            let col = sc.lit(p[f[0]], p[f[1]], p[f[2]], colour);
            sc.quad(p[f[0]], p[f[1]], p[f[2]], p[f[3]], col, 0.8);
            for e in 0..4 {
                sc.seg(p[f[e]], p[f[(e + 1) % 4]], 1.0, palette::ink(), 0.35);
            }
        }
    }
    // The joints: the home joint positions carried by the frame of the link they start.
    for (k, m) in [Pga3d::identity(), links[0], links[1]].iter().enumerate() {
        let joint = *m >> Point::xyz(0.0, 0.0, k as f64);
        sc.dot(joint, Marker::Dot, 8.0, palette::ink());
    }
    sc.dot(links[2] >> tip_home, Marker::Dot, 7.0, palette::yellow());
    sc.dot(target, Marker::Star, 14.0, palette::red());
    sc.draw(c);

    caption(
        c,
        "ROBOT ARM: THE JACOBIAN IS THE JOINT AXES",
        "THREE JOINTS FOLLOW A LOOP BY LEAST SQUARES (PGA3D)",
    );

    // The joint angles over the loop, with the current ones marked.
    let rect = plot::inset(
        [w * 0.62, 0.0, w, h * 0.62],
        w * 0.05,
        h * 0.15,
        w * 0.02,
        h * 0.075,
    );
    let all: Vec<[f64; 3]> = states.iter().map(|st| angles(&st.joints, &axis)).collect();
    let (lo, hi) = all
        .iter()
        .flatten()
        .fold((f64::MAX, f64::MIN), |(a, b), v| (a.min(*v), b.max(*v)));
    let ax = Axes::new(rect, [0.0, 1.0], [(lo - 0.2) as f32, (hi + 0.2) as f32]);
    ax.frame(c, "JOINT ANGLES", "FRACTION OF THE LOOP", "RAD");
    let names = ["YAW", "PITCH 1", "PITCH 2"];
    for (k, colour) in colours.iter().enumerate() {
        let pts: Vec<Point2> = all
            .iter()
            .enumerate()
            .map(|(j, a)| Point2::xy(j as f32 / n as f32, a[k] as f32))
            .collect();
        ax.polyline(c, &pts, 1.8, *colour, 0.9);
        let now = all[i][k] + (all[(i + 1) % n][k] - all[i][k]) * frac;
        ax.scatter(
            c,
            &[Point2::xy(s / n as f32, now as f32)],
            Marker::Dot,
            8.0,
            *colour,
            1.0,
        );
    }
    let legend: Vec<(&str, Rgb)> = names.iter().copied().zip(colours).collect();
    ax.legend(c, &legend);

    // The statics at rest, in text.
    let (velocity, torques) = statics();
    let (error, _) = homing();
    let x0 = w * 0.6;
    let lines = [
        "AT REST, FORCE (0,2,-1) AT (1,0,3):".to_string(),
        format!(
            "JOINT TORQUES  {:+.3} {:+.3} {:+.3}",
            torques[0], torques[1], torques[2]
        ),
        format!(
            "TIP VELOCITY   {:+.3} {:+.3} {:+.3}",
            velocity.e032(),
            velocity.e013(),
            velocity.e021()
        ),
        format!("HOMING ERROR   {error:.1E}"),
        format!("TRACKING ERROR {:.1E}", (a.target & a.tip).norm()),
    ];
    for (k, l) in lines.iter().enumerate() {
        c.text(
            l,
            x0,
            h * 0.70 + k as f32 * 20.0,
            10.0,
            if k == 0 {
                palette::grid()
            } else {
                palette::ink()
            },
            Align::Left,
        );
    }
}

fn main() {
    run(Anim::new("robot arm", SECONDS).size(960, 540), draw);
}

#[cfg(test)]
mod tests {
    use super::arm::*;

    /// numga's `test_statics_and_homing`: `statics` checks the tip velocity against a finite
    /// difference, `homing` that the tip reaches the target, and the torques are numga's.
    #[test]
    fn statics_and_homing() {
        let (_, torques) = statics();
        // numga's values are `[-2.0, -2.1374, -1.1393]`: the joins of the axes and of the
        // forque each flip with the pseudoscalar's orientation, and their pairing once more, so
        // gax's torques are the negatives (and right-handed: the force's moment about +z is +2).
        let expected = [-2.0, -2.1374, -1.1393];
        for (a, b) in torques.iter().zip(expected) {
            assert!((a + b).abs() < 0.01, "{torques:?}");
        }
        let (error, _) = homing();
        assert!(error < 1e-8);
    }

    /// The joint angles read back from the joints: the rest configuration's.
    #[test]
    fn angles_read_back() {
        let (axis, _, rest) = arm();
        let a = angles(&rest, &axis);
        for (x, y) in a.iter().zip([0.3, 0.5, 0.8]) {
            assert!((x - y).abs() < 1e-12, "{a:?}");
        }
    }

    /// The loop is numga's, `(1 + 0.5 sin s, 1 + 0.5 cos s, 1.2 + 0.3 sin 2s)`.
    #[test]
    fn the_loop_is_numgas() {
        for k in 0..TARGETS {
            let s = core::f64::consts::TAU * k as f64 / TARGETS as f64;
            let expected = [
                1.0 + 0.5 * s.sin(),
                1.0 + 0.5 * s.cos(),
                1.2 + 0.3 * (2.0 * s).sin(),
            ];
            let p = loop_target(k).to_euclidean();
            assert!(
                p.iter().zip(expected).all(|(a, b)| (a - b).abs() < 1e-12),
                "{p:?}"
            );
        }
    }

    /// Tracking keeps the tip on the loop: two steps per target suffice.
    #[test]
    fn tracking_follows_the_loop() {
        for s in tracking() {
            assert!((s.target & s.tip).norm() < 1e-3);
        }
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws(super::draw, 0.5);
    }
}
