//! A scene graph and camera pipeline (PGA3D).
//!
//! A robot arm is a chain of motors, and a pinhole camera projects points onto its sensor
//! plane. The projection is a plain generic function: given a point it returns a point,
//! and given `Point::slot()` it returns the projection *map*. Each arm part's chain
//! composes into one map from part coordinates to image coordinates, and is then applied to the
//! corners of the part.
//!
//! Run with `cargo run --example scene_graph`.

use gax::pga3d::{Motor, Plane, Point};
use gax::{Slots, Unit};
use std::f64::consts::FRAC_PI_2;

/// Central projection through `eye` onto `screen`: the line from the eye through the point,
/// met with the screen. Generic over the slots, so it works on points and on point maps.
fn project<S: Slots>(
    eye: Point<(), f64>,
    screen: Plane<(), f64>,
    p: Point<S, f64>,
) -> Point<S, f64> {
    (eye & p) ^ screen
}

fn main() {
    // The scene graph: each joint is a motor relative to its parent.
    let base: Unit<Motor<(), f64>> = Motor::translation(0.0, 0.0, 0.5);
    let shoulder = base * Motor::rotation_about(0.0, 1.0, 0.0, 0.6);
    let upper_arm = shoulder * Motor::translation(0.0, 0.0, 1.0);
    let elbow = upper_arm * Motor::rotation_about(0.0, 1.0, 0.0, -1.1);
    let forearm = elbow * Motor::translation(0.0, 0.0, 0.4);

    // The camera: 6 units back along -y, looking along +y. In its own frame the eye is at the
    // origin and the sensor is the plane z = 1 (focal length 1).
    let camera_pose =
        Motor::translation(0.0, -6.0, 1.0) * Motor::rotation_about(1.0, 0.0, 0.0, -FRAC_PI_2);
    let eye = Point::xyz(0.0, 0.0, 0.0);
    let sensor = Plane::from_normal([0.0, 0.0, 1.0], 1.0);

    // Maps, built once: camera coordinates -> image, world -> camera, world -> image.
    let lens: Point<(Point,), f64> = project(eye, sensor, Point::slot());
    let world_to_camera: Point<(Point,), f64> = camera_pose << Point::slot();
    let world_to_image = lens.of(world_to_camera);

    // One map per part: part coordinates -> image.
    let parts = [("upper arm", upper_arm), ("forearm", forearm)];
    let corners: Vec<Point<(), f64>> = (0..8)
        .map(|i| {
            let s = |bit: usize, a: f64| if i & bit == 0 { -a } else { a };
            Point::xyz(s(1, 0.1), s(2, 0.1), s(4, 0.2))
        })
        .collect();
    for (name, pose) in parts {
        let part_to_image: Point<(Point,), f64> = world_to_image.of(pose >> Point::slot());
        println!("{name}: image corners");
        for &c in &corners {
            let img = part_to_image.of(c).to_euclidean();
            // The same, step by step, through the value path of every operation.
            let direct = project(eye, sensor, camera_pose << (pose >> c)).to_euclidean();
            assert!(img.iter().zip(direct).all(|(a, b)| (a - b).abs() < 1e-9));
            println!("  ({:+.4}, {:+.4})", img[0], img[1]);
        }
    }
}
