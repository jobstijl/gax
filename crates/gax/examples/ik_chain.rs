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
use gax::pga3d::{Motor, Point};

/// A joint: an axis direction in the joint's frame, and the link to the next joint.
struct Joint {
    axis: [f64; 3],
    link: [f64; 3],
}

type Pose = Unit<Motor<(), f64>>;

/// The pose of every joint (the motor from its frame to the world), and of the hand.
fn forward(joints: &[Joint], angles: &[f64]) -> (Vec<Pose>, Pose) {
    let mut pose = Motor::translation(0.0, 0.0, 0.0);
    let mut poses = Vec::new();
    for (j, &a) in joints.iter().zip(angles) {
        let [x, y, z] = j.axis;
        pose = pose * Motor::rotation_about(x, y, z, a);
        poses.push(pose);
        let [dx, dy, dz] = j.link;
        pose = pose * Motor::translation(dx, dy, dz);
    }
    (poses, pose)
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn main() {
    let joints = [
        Joint {
            axis: [0.0, 0.0, 1.0],
            link: [0.0, 0.0, 1.0],
        }, // base yaw
        Joint {
            axis: [0.0, 1.0, 0.0],
            link: [0.0, 0.0, 1.5],
        }, // shoulder
        Joint {
            axis: [0.0, 1.0, 0.0],
            link: [0.0, 0.0, 1.2],
        }, // elbow
        Joint {
            axis: [1.0, 0.0, 0.0],
            link: [0.0, 0.0, 0.5],
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
            let [x, y, z] = joints[k].axis;
            let pivot = (poses[k] >> origin).to_euclidean();
            let dir = poses[k] >> Point::direction(x, y, z);
            let d = [dir.e032(), dir.e013(), dir.e021()];
            // Hand and target relative to the pivot, projected onto the plane of rotation.
            let project = |p: [f64; 3]| {
                let v = sub(p, pivot);
                let along = dot(v, d);
                [
                    v[0] - along * d[0],
                    v[1] - along * d[1],
                    v[2] - along * d[2],
                ]
            };
            let h = project((hand >> origin).to_euclidean());
            let t = project(target.to_euclidean());
            let turn = dot(cross(h, t), d).atan2(dot(h, t));
            // `rotation_about` turns counterclockwise about its axis (right-handed), as
            // `atan2` measures the angle from the hand to the target.
            angles[k] += turn;
        }
        let hand = (forward(&joints, &angles).1 >> origin).to_euclidean();
        let err = dot(
            sub(hand, target.to_euclidean()),
            sub(hand, target.to_euclidean()),
        )
        .sqrt();
        if sweep % 5 == 0 || err < 1e-9 {
            println!(
                "sweep {sweep:2}: hand at ({:.4}, {:.4}, {:.4}), error {err:.2e}",
                hand[0], hand[1], hand[2]
            );
        }
        if err < 1e-9 {
            break;
        }
    }
    let hand = (forward(&joints, &angles).1 >> origin).to_euclidean();
    let err = dot(
        sub(hand, target.to_euclidean()),
        sub(hand, target.to_euclidean()),
    )
    .sqrt();
    println!(
        "joint angles: {:?}",
        angles.map(|a| (a.to_degrees() * 10.0).round() / 10.0)
    );
    assert!(
        err < 1e-6,
        "the target is in reach, so CCD should get there (error {err})"
    );
}
