//! numga's `mechanics/tricycle`: a tricycle steered by its front wheel, in the plane (PGA2D) and
//! in space (PGA3D), by the same code.
//!
//! Each axle is a hyperplane of the algebra, a vector: a line in the plane, a vertical plane in
//! space. The rear axle lies along y at the origin; the front axle, at rest, a wheelbase further
//! along x. Steering turns the front axle about the steering axis, the meet of the front axle at
//! rest with the hyperplane `y = 0` down the middle of the frame: a point in the plane, a
//! vertical line in space.
//!
//! The two axles meet in the turning centre, `rear ^ front`: a point in the plane, a vertical
//! line in space. Every wheel rolls around it, since every axle passes through it, and the
//! tricycle moves by its exponential: a rotation about it, or, with the front wheel straight
//! and the axles parallel, a translation, the meet then lying at infinity. The meet's weight is
//! the sine of the steering angle and its distance from the front wheel the wheelbase over that
//! sine, so the front wheel, which the pedals turn, moves at the same speed at every steering
//! angle. Over a drive each step moves the tricycle in its own frame, so the steps multiply on
//! the right of the pose.
//!
//! The algebra is a type parameter: [`Ground`] adds the hyperplanes to gax's
//! [`Motions`](gax::motions::Motions) (whose twist is the meet of two hyperplanes, and whose
//! points are the wheels' contacts), and the same functions drive the tricycle in the plane and
//! in space. The construction follows the tricycle example of ganja.js by Steven De Keninck.
//! In the notation of vehicle dynamics this is Ackermann steering: the instantaneous centre of
//! rotation lies on the extension of every axle.
//!
//! Left the plane from above, right space in perspective, both following the tricycle: its frame
//! and wheels, each axle drawn on to the turning centre, and the trail of the middle of the rear
//! axle.

use gax::motions::{Pga2d, Pga3d};
use gax::{pga2d, pga3d};
use gax_numga_examples::scene3::panel3;
use gax_numga_examples::{
    Align, Anim, Axes, Camera, Canvas, Lens, Light, Marker, Point2, Point3, Pos2, Pos3, Rect,
    Scene3, backdrop, caption, palette, run,
};
use std::sync::OnceLock;

mod tricycle {
    use gax::motions::{Motions, Pga2d, Pga3d, Value};
    use gax::{pga2d, pga3d};
    use gax_numga_examples::signal::wave;

    /// An algebra the tricycle drives in: its rigid motions, and its hyperplanes.
    pub trait Ground: Motions<f64> {
        /// A hyperplane: a line in the plane, a plane in space.
        type Plane: Value<f64>;
        /// The hyperplanes `x = 0` and `y = 0`, and the one at infinity.
        const X: Self::Plane;
        const Y: Self::Plane;
        const W: Self::Plane;

        /// The point on the ground at `(x, y)`.
        fn ground(x: f64, y: f64) -> Self::Point;
        /// Where two hyperplanes meet, `a ^ b`.
        fn meet(a: Self::Plane, b: Self::Plane) -> Self::Twist;
        /// A meet scaled to unit weight.
        fn unit(t: Self::Twist) -> Self::Twist;
        /// A hyperplane moved by a motor.
        fn moved(m: Self::Motor, a: Self::Plane) -> Self::Plane;
        /// The pairing of a hyperplane with a point, `a & p`.
        fn pairing(a: Self::Plane, p: Self::Point) -> f64;
        /// The norm of the join of three points scaled to unit weight: how far they are from
        /// lying on one line.
        #[cfg_attr(not(test), allow(dead_code))] // numga's check that no wheel slides
        fn collinearity(a: Self::Point, b: Self::Point, c: Self::Point) -> f64;
        /// The norm of a meet's part off the hyperplane at infinity, `w ^ t`: zero for a meet at
        /// infinity.
        #[cfg_attr(not(test), allow(dead_code))] // numga's check of the straight wheel
        fn finite(t: Self::Twist) -> f64;
    }

    impl Ground for Pga2d {
        type Plane = pga2d::Line<(), f64>;
        const X: Self::Plane = pga2d::Line::new(1.0, 0.0, 0.0);
        const Y: Self::Plane = pga2d::Line::new(0.0, 1.0, 0.0);
        const W: Self::Plane = pga2d::Line::new(0.0, 0.0, 1.0);

        fn ground(x: f64, y: f64) -> Self::Point {
            pga2d::Point::xy(x, y)
        }
        fn meet(a: Self::Plane, b: Self::Plane) -> Self::Twist {
            a ^ b
        }
        fn unit(t: Self::Twist) -> Self::Twist {
            t.normalized()
        }
        fn moved(m: Self::Motor, a: Self::Plane) -> Self::Plane {
            m >> a
        }
        fn pairing(a: Self::Plane, p: Self::Point) -> f64 {
            (a & p).s()
        }
        fn collinearity(a: Self::Point, b: Self::Point, c: Self::Point) -> f64 {
            (a.unitized() & b.unitized() & c.unitized()).s().abs()
        }
        fn finite(t: Self::Twist) -> f64 {
            (Self::W ^ t).e012().abs()
        }
    }

    impl Ground for Pga3d {
        type Plane = pga3d::Plane<(), f64>;
        const X: Self::Plane = pga3d::Plane::new(1.0, 0.0, 0.0, 0.0);
        const Y: Self::Plane = pga3d::Plane::new(0.0, 1.0, 0.0, 0.0);
        const W: Self::Plane = pga3d::Plane::new(0.0, 0.0, 0.0, 1.0);

        fn ground(x: f64, y: f64) -> Self::Point {
            pga3d::Point::xyz(x, y, 0.0)
        }
        fn meet(a: Self::Plane, b: Self::Plane) -> Self::Twist {
            a ^ b
        }
        fn unit(t: Self::Twist) -> Self::Twist {
            t.normalized()
        }
        fn moved(m: Self::Motor, a: Self::Plane) -> Self::Plane {
            m >> a
        }
        fn pairing(a: Self::Plane, p: Self::Point) -> f64 {
            (a & p).s()
        }
        fn collinearity(a: Self::Point, b: Self::Point, c: Self::Point) -> f64 {
            (a.unitized() & b.unitized() & c.unitized()).norm()
        }
        fn finite(t: Self::Twist) -> f64 {
            (Self::W ^ t).norm()
        }
    }

    /// The rear axle, the front axle turned by the steering angle, and the steering motor that
    /// turns it, about the meet of the front axle at rest with the middle of the frame.
    pub fn axles<G: Ground>(wheelbase: f64, steering: f64) -> (G::Plane, G::Plane, G::Motor) {
        let rest = G::X - G::W * wheelbase;
        let axis = G::unit(G::meet(rest, G::Y));
        let steer = G::exp(axis * (-steering / 2.0));
        (G::X, G::moved(steer, rest), steer)
    }

    /// The motion of one step: about the turning centre of its axles, in the tricycle's frame.
    pub fn step<G: Ground>(rear: G::Plane, front: G::Plane, length: f64) -> G::Motor {
        G::exp(G::meet(rear, front) * (-length / 2.0))
    }

    /// Where each wheel touches the ground, the rear wheels and then the front wheel, and a point
    /// a little ahead of each along the way it rolls, the front one turned by the steering, in
    /// the tricycle's own frame.
    pub fn wheels<G: Ground>(wheelbase: f64, track: f64, steer: G::Motor) -> [[G::Point; 3]; 2] {
        let ahead = 0.1 * wheelbase;
        let contacts = [
            G::ground(0.0, -track / 2.0),
            G::ground(0.0, track / 2.0),
            G::ground(wheelbase, 0.0),
        ];
        let forward = [
            G::ground(ahead, -track / 2.0),
            G::ground(ahead, track / 2.0),
            steer >> G::ground(wheelbase + ahead, 0.0),
        ];
        [contacts, forward]
    }

    /// The coordinates along x and y of a point on the ground: its pairings with the hyperplanes
    /// `x = 0` and `y = 0` over its weight, the pairing with the one at infinity.
    pub fn coordinates<G: Ground>(p: G::Point) -> [f64; 2] {
        let weight = G::pairing(G::W, p);
        [G::pairing(G::X, p) / weight, G::pairing(G::Y, p) / weight]
    }

    // --- scenarios

    pub const WHEELBASE: f64 = 1.2;
    pub const TRACK: f64 = 0.8;
    /// The distance the front wheel rolls per step, over the wheelbase, and the steps.
    pub const STEP: f64 = 0.04;
    pub const STEPS: usize = 600;
    /// The steering angle held, how far it swings about that, and how many times it swings over
    /// the drive.
    pub const BIAS: f64 = 0.35;
    pub const SWING: f64 = 0.3;
    pub const SWINGS: f64 = 3.0;

    /// The steering angle at a step.
    pub fn steering(step: usize) -> f64 {
        let phase = core::f64::consts::TAU * SWINGS * step as f64 / STEPS as f64;
        BIAS + SWING * wave(phase)
    }

    /// The drive, per step, in the ground's frame: the wheels' contacts with the ground and the
    /// points ahead of them, the turning centre, and the middle of the rear axle.
    #[derive(Clone, Debug)]
    pub struct Drive<G: Ground> {
        pub contacts: Vec<[G::Point; 3]>,
        pub forward: Vec<[G::Point; 3]>,
        pub centres: Vec<G::Twist>,
        pub middle: Vec<G::Point>,
    }

    /// The drive in an algebra: each step's motion multiplied onto the pose on the right.
    pub fn drive<G: Ground>() -> Drive<G> {
        let mut pose = G::identity();
        let mut drive = Drive {
            contacts: Vec::with_capacity(STEPS),
            forward: Vec::with_capacity(STEPS),
            centres: Vec::with_capacity(STEPS),
            middle: Vec::with_capacity(STEPS),
        };
        for k in 0..STEPS {
            let (rear, front, steer) = axles::<G>(WHEELBASE, steering(k));
            pose = pose * step::<G>(rear, front, STEP);
            let [contacts, forward] = wheels::<G>(WHEELBASE, TRACK, steer);
            drive.contacts.push(contacts.map(|p| pose >> p));
            drive.forward.push(forward.map(|p| pose >> p));
            drive.centres.push(pose >> G::meet(rear, front));
            drive.middle.push(pose >> G::ground(0.0, 0.0));
        }
        drive
    }
}

use tricycle::*;

/// numga's 150 frames of 50 ms over the drive.
const SECONDS: f32 = 7.5;
/// The wheels' radius.
const WHEEL_RADIUS: f64 = 0.25;
/// Half the width of the view, which follows the middle of the rear axle.
const VIEW: f32 = 3.0;
/// Points around a wheel's rim, in space.
const RIM: usize = 32;
/// The ground grid's lines either side of the tricycle, a metre apart.
const GRID: i32 = 5;
/// The camera in space: numga's view, 65 degrees round from x clockwise and 35 up, and its
/// distance from the tricycle.
const AZIMUTH: f32 = -65.0;
const ELEVATION: f32 = 35.0;
const DISTANCE: f32 = 7.5;
/// The height of the turning centre's vertical line as drawn, in wheel radii.
const CENTRE_HEIGHT: f64 = 4.0;

/// Both drives, computed once.
struct Drives {
    plane: Drive<Pga2d>,
    space: Drive<Pga3d>,
}

fn drives() -> &'static Drives {
    static S: OnceLock<Drives> = OnceLock::new();
    S.get_or_init(|| Drives {
        plane: drive(),
        space: drive(),
    })
}

/// The frame's light.
fn frame_light() -> Light {
    palette::ink()
}

/// The wheels' light.
fn wheel_light() -> Light {
    palette::yellow()
}

/// The light of the axles drawn on to the turning centre.
fn axle_light() -> Light {
    palette::red().faded(0.6)
}

/// The light of the trail of the middle of the rear axle.
fn trail_light() -> Light {
    palette::sky().faded(0.6)
}

/// A point on the ground, as drawn in the plane.
fn flat(p: pga2d::Point<(), f64>) -> Point2 {
    let [x, y] = coordinates::<Pga2d>(p);
    Point2::xy(x as f32, y as f32)
}

/// A step `length` long from a point towards another, in the drawing.
fn towards(from: Point2, to: Point2, length: f32) -> Point2 {
    let d = to.unitized() - from.unitized();
    d.gp(length / d.ideal_norm())
}

/// The plane from above, around the tricycle at a step.
fn draw_plane(c: &mut Canvas, rect: Rect, drive: &Drive<Pga2d>, k: usize) {
    let wheels = drive.contacts[k].map(flat);
    let ahead = drive.forward[k].map(flat);
    let ax = Axes::equal(
        rect,
        flat(drive.middle[k]),
        VIEW * rect.height() / rect.width(),
    );
    let trail: Vec<Point2> = drive.middle[..=k].iter().map(|p| flat(*p)).collect();
    ax.polyline(c, &trail, 1.2, trail_light());
    let centre = drive.centres[k];
    if centre.e12() != 0.0 {
        // The turning centre is a meet, an extrinsic point; drawn, it is where it is.
        let at = centre.point2().unitized();
        for w in wheels {
            ax.dashed(c, &[w, at], 1.0, 5.0, axle_light());
        }
        ax.scatter(c, &[at], Marker::Dot, 8.0, palette::red());
    }
    let between = (wheels[0] + wheels[1]).unitized();
    ax.line(c, wheels[0], wheels[1], 3.0, frame_light());
    ax.line(c, between, wheels[2], 3.0, frame_light());
    let scale = ax.scale();
    for (w, a) in wheels.iter().zip(ahead) {
        let along = towards(ax.px(*w), ax.px(a), WHEEL_RADIUS as f32 * scale);
        let at = ax.px(*w);
        c.line(at - along, at + along, 4.0, wheel_light());
    }
    ax.frame(c, "THE PLANE (PGA2D)", "", "");
}

/// A point raised to the wheels' hubs: a contact, or the foot of the turning centre (a meet).
fn hub<O: gax::Orientation>(
    p: pga3d::oriented::Point<(), f64, O>,
) -> pga3d::oriented::Point<(), f64, O> {
    pga3d::Motor::translation(0.0, 0.0, WHEEL_RADIUS) >> p
}

/// A wheel in space: its contact turned about its axle, the line through its hub across the
/// way it rolls (the heading turned a quarter turn about the vertical).
fn wheel(s: &mut Scene3, contact: pga3d::Point<(), f64>, ahead: pga3d::Point<(), f64>) {
    let quarter = pga3d::Motor::rotation_about(0.0, 0.0, 1.0, core::f64::consts::FRAC_PI_2);
    let across = quarter >> (ahead.unitized() - contact.unitized());
    let centre = hub(contact);
    let axle = centre & (centre + across);
    let rim: Vec<Point3> = (0..=RIM)
        .map(|k| {
            let turn = pga3d::Motor::rotation(axle, core::f64::consts::TAU * k as f64 / RIM as f64);
            (turn >> contact).point3()
        })
        .collect();
    s.polyline(&rim, 2.0, wheel_light());
}

/// A faint grid on the ground, a metre apart, around a point.
fn floor(s: &mut Scene3, around: pga3d::Point<(), f64>) {
    let [x, y, _] = around.to_euclidean().map(f64::round);
    let reach = f64::from(GRID);
    for k in -GRID..=GRID {
        let k = f64::from(k);
        let along_x = [-reach, reach].map(|d| pga3d::Point::xyz(x + d, y + k, 0.0).point3());
        let along_y = [-reach, reach].map(|d| pga3d::Point::xyz(x + k, y + d, 0.0).point3());
        s.seg(along_x[0], along_x[1], 1.0, palette::grid().faded(0.7));
        s.seg(along_y[0], along_y[1], 1.0, palette::grid().faded(0.7));
    }
}

/// Space in perspective around the tricycle at a step, the turning centre a vertical line.
fn draw_space(c: &mut Canvas, rect: Rect, drive: &Drive<Pga3d>, k: usize) {
    let middle = drive.middle[k];
    let cam = Camera::orbit(
        rect,
        middle.point3(),
        DISTANCE,
        AZIMUTH.to_radians(),
        ELEVATION.to_radians(),
        Lens::Perspective(0.62),
    );
    panel3(c, rect, cam, |s| {
        floor(s, middle);
        let trail: Vec<Point3> = drive.middle[..=k].iter().map(|p| p.point3()).collect();
        s.polyline(&trail, 1.2, trail_light());
        let wheels = drive.contacts[k];
        let centre = drive.centres[k];
        let ground = pga3d::Plane::new(0.0, 0.0, 1.0, 0.0);
        let foot = centre ^ ground;
        if foot.e123() != 0.0 {
            let up = pga3d::Motor::translation(0.0, 0.0, CENTRE_HEIGHT * WHEEL_RADIUS);
            s.seg(foot.point3(), (up >> foot).point3(), 2.0, palette::red());
            for w in wheels {
                s.seg(hub(w).point3(), hub(foot).point3(), 1.0, axle_light());
            }
        }
        let between = hub((wheels[0] + wheels[1]).unitized());
        s.seg(
            hub(wheels[0]).point3(),
            hub(wheels[1]).point3(),
            3.0,
            frame_light(),
        );
        s.seg(
            between.point3(),
            hub(wheels[2]).point3(),
            3.0,
            frame_light(),
        );
        for (w, a) in wheels.iter().zip(drive.forward[k]) {
            wheel(s, *w, a);
        }
    });
    let corners = [rect.lo, rect.top_right(), rect.hi, rect.bottom_left()];
    c.polyline(&corners, 1.0, palette::grid(), true);
    let title = rect.top_middle() - Point2::direction(0.0, 8.0);
    c.text("SPACE (PGA3D)", title, 13.0, palette::ink(), Align::Center);
}

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let d = drives();
    let k = ((t / SECONDS).rem_euclid(1.0) * (STEPS - 1) as f32).round() as usize;
    let body = c
        .rect()
        .part(0.0, 0.16, 1.0, 1.0)
        .inset(36.0, 26.0, 16.0, 30.0);
    draw_plane(
        c,
        body.part(0.0, 0.0, 0.5, 1.0).inset(0.0, 0.0, 8.0, 0.0),
        &d.plane,
        k,
    );
    draw_space(
        c,
        body.part(0.5, 0.0, 1.0, 1.0).inset(8.0, 0.0, 0.0, 0.0),
        &d.space,
        k,
    );
    let note = format!("STEERING {:.0} DEGREES", steering(k).to_degrees());
    let at = body.bottom_left() + Point2::direction(10.0, -10.0);
    c.text(&note, at, 11.0, palette::ink().faded(0.6), Align::Left);
    caption(
        c,
        "TRICYCLE: ACKERMANN STEERING BY THE SAME CODE IN TWO ALGEBRAS",
        "THE AXLES MEET IN THE TURNING CENTRE REAR ^ FRONT; EACH STEP IS ITS EXPONENTIAL",
    );
}

fn main() {
    run(Anim::new("tricycle", SECONDS).size(960, 540), draw);
}

#[cfg(test)]
mod tests {
    use super::tricycle::*;
    use gax::motions::{Pga2d, Pga3d};
    use gax::vga2d::Vector;

    /// numga's checks of a drive. No wheel slides sideways: each wheel's contact, the point
    /// ahead of it and its next contact join to nothing, but for the step's curvature (a chord
    /// leaves its tangent by its length squared over the turning radius, under the step itself).
    /// The front wheel rolls the same distance every step. With the front wheel straight the
    /// axles meet at infinity.
    fn drives_without_sliding<G: Ground>() -> Drive<G> {
        let drive = drive::<G>();
        for k in 0..STEPS - 1 {
            for w in 0..3 {
                let [contact, ahead, next] = [
                    drive.contacts[k][w],
                    drive.forward[k][w],
                    drive.contacts[k + 1][w],
                ];
                let rolling = G::collinearity(contact, ahead, next);
                assert!(
                    rolling < 0.1 * WHEELBASE * (STEP * WHEELBASE).powi(2),
                    "{rolling}"
                );
            }
            let [before, after] = [k, k + 1]
                .map(|j| Vector::<(), f64>::from_coeffs(coordinates::<G>(drive.contacts[j][2])));
            let length = (after - before).norm();
            assert!(
                (length - STEP * WHEELBASE).abs() <= 1e-3 * STEP * WHEELBASE,
                "{length}"
            );
        }
        let (rear, front, _) = axles::<G>(WHEELBASE, 0.0);
        assert!(G::finite(G::meet(rear, front)) <= 1e-12);
        drive
    }

    #[test]
    fn the_plane_and_space_drive_the_same_track() {
        let (plane, space) = (
            drives_without_sliding::<Pga2d>(),
            drives_without_sliding::<Pga3d>(),
        );
        for (a, b) in plane.middle.iter().zip(&space.middle) {
            let (p, q) = (coordinates::<Pga2d>(*a), coordinates::<Pga3d>(*b));
            assert!(
                (p[0] - q[0]).abs() <= 1e-9 && (p[1] - q[1]).abs() <= 1e-9,
                "{p:?} {q:?}"
            );
        }
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws(super::draw, 0.0);
        gax_numga_examples::app::assert_draws(super::draw, 7.4);
    }
}
