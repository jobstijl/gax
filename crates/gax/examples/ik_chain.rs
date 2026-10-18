//! Inverse kinematics of a robot arm with motors (PGA3D).
//!
//! The arm is a chain of revolute joints. Forward kinematics is a product of motors: each
//! joint turns about its axis line and each link translates. Cyclic coordinate descent
//! (CCD) then turns one joint at a time, from the hand back to the base, so that the hand
//! swings as close to the target as that joint allows: the best angle is the one between
//! the hand and the target seen along the joint's axis.
//!
//! Run with `cargo run --example ik_chain`.

use gax::Unit;
use gax::pga3d::{Motor, Plane, Point};

/// A joint: an axis direction in the joint's frame, and the link to the next joint (a
/// direction too: a displacement).
struct Joint {
    axis: Point<(), f64>,
    link: Point<(), f64>,
}

type Pose = Unit<Motor<(), f64>>;

/// The pose of every joint (the motor from its frame to the world), and of the hand.
fn forward(joints: &[Joint], angles: &[f64]) -> (Vec<Pose>, Pose) {
    let mut pose: Pose = Unit::identity();
    let mut poses = Vec::new();
    for (j, &a) in joints.iter().zip(angles) {
        pose = pose * Motor::rotation_about(j.axis, a);
        poses.push(pose);
        pose = pose * Motor::translation(j.link);
    }
    (poses, pose)
}

/// The inner product of two directions: of the planes through the origin orthogonal to them.
fn dot(a: Point<(), f64>, b: Point<(), f64>) -> f64 {
    (Plane::orthogonal_to(a) | Plane::orthogonal_to(b)).s()
}

/// The cross product of two directions: the direction of the meet of those planes.
fn cross(a: Point<(), f64>, b: Point<(), f64>) -> Point<(), f64> {
    (Plane::orthogonal_to(a) ^ Plane::orthogonal_to(b)).direction()
}

fn main() {
    let z = |l: f64| Point::direction(0.0, 0.0, l);
    let joints = [
        Joint {
            axis: z(1.0),
            link: z(1.0),
        }, // base yaw
        Joint {
            axis: Point::direction(0.0, 1.0, 0.0),
            link: z(1.5),
        }, // shoulder
        Joint {
            axis: Point::direction(0.0, 1.0, 0.0),
            link: z(1.2),
        }, // elbow
        Joint {
            axis: Point::direction(1.0, 0.0, 0.0),
            link: z(0.5),
        }, // wrist
    ];
    let target = Point::<(), f64>::xyz(1.2, 0.8, 1.9);
    let mut angles = [0.0; 4];
    let origin = Point::xyz(0.0, 0.0, 0.0);

    for sweep in 0..50 {
        for k in (0..joints.len()).rev() {
            let (poses, hand) = forward(&joints, &angles);
            // The joint's pivot and axis direction in the world: the motor moves the origin
            // and the axis direction (a point at infinity) out of the joint's frame.
            let pivot = poses[k] >> origin;
            let d = poses[k] >> joints[k].axis;
            // Hand and target relative to the pivot, projected onto the plane of rotation.
            let project = |p: Point<(), f64>| {
                let v = p - pivot;
                v - d * dot(v, d)
            };
            let h = project(hand >> origin);
            let t = project(target);
            let turn = dot(cross(h, t), d).atan2(dot(h, t));
            // `rotation_about` turns counterclockwise about its axis (right-handed), as
            // `atan2` measures the angle from the hand to the target.
            angles[k] += turn;
        }
        let hand = forward(&joints, &angles).1 >> origin;
        let err = hand.distance(target);
        if sweep % 5 == 0 || err < 1e-9 {
            let [x, y, z] = hand.to_euclidean();
            println!("sweep {sweep:2}: hand at ({x:.4}, {y:.4}, {z:.4}), error {err:.2e}");
        }
        if err < 1e-9 {
            break;
        }
    }
    let err = (forward(&joints, &angles).1 >> origin).distance(target);
    println!(
        "joint angles: {:?}",
        angles.map(|a| (a.to_degrees() * 10.0).round() / 10.0)
    );
    assert!(
        err < 1e-6,
        "the target is in reach, so CCD should get there (error {err})"
    );
}
