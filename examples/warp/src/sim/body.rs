//! Kinematics in gax terms: a pose is a unit PGA2D motor, a velocity is a twist (a PGA2D
//! bivector, which is a `Point`), and integration is `pose ← exp(dt B) pose`, renormalized with
//! one Newton step every tick (docs/numerics.md: drift policy).

use gax::Unit;
use gax::pga2d::{Motor, Point};

/// A pose.
pub type Pose = Unit<Motor<(), f32>>;

/// The identity pose.
pub fn identity() -> Pose {
    Motor::translation(0.0, 0.0)
}

/// The pose at `(x, y)` turned by `angle` (counterclockwise from the x axis).
pub fn pose_at(x: f32, y: f32, angle: f32) -> Pose {
    Motor::translation(x, y) * Motor::rotation(Point::xy(0.0, 0.0), angle)
}

/// A moving rigid body.
#[derive(Clone, Copy, Debug)]
pub struct Body {
    /// Where it is now.
    pub pose: Pose,
    /// Where it was at the start of the last tick (for interpolated drawing).
    pub prev: Pose,
    /// Its linear velocity (a direction), in world coordinates.
    pub vel: Point<(), f32>,
    /// Its angular velocity, counterclockwise, in radians per second.
    pub spin: f32,
}

impl Body {
    /// A body at rest.
    pub fn new(pose: Pose) -> Body {
        Body {
            pose,
            prev: pose,
            vel: Point::direction(0.0, 0.0),
            spin: 0.0,
        }
    }

    /// Its position.
    pub fn pos(&self) -> Point<(), f32> {
        self.pose >> Point::xy(0.0, 0.0)
    }

    /// Its position as `[x, y]`.
    pub fn xy(&self) -> [f32; 2] {
        self.pos().to_euclidean()
    }

    /// The direction its local x axis points to.
    pub fn heading(&self) -> Point<(), f32> {
        self.pose >> Point::direction(1.0, 0.0)
    }

    /// The world-frame twist of its motion: translation at `vel` plus rotation at `spin`
    /// about its centre. Twists add.
    pub fn twist(&self) -> Point<(), f32> {
        let [vx, vy] = [self.vel.e20(), self.vel.e01()];
        Point::translation_twist(vx, vy) + Point::rotation_twist(self.pos(), self.spin)
    }

    /// Advance by `dt`: `pose ← exp(dt B) pose`, then one Newton step towards `m ~m = 1`.
    pub fn step(&mut self, dt: f32) {
        self.prev = self.pose;
        self.pose = (self.twist().gp(dt).exp() * self.pose).renormalize_fast();
    }

    /// Move by `(dx, dy)` in world coordinates (a wall bounce, a respawn).
    pub fn shift(&mut self, dx: f32, dy: f32) {
        self.pose = Motor::translation(dx, dy) * self.pose;
    }

    /// The drawn pose between the last two ticks: `prev exp(t log(~prev pose))`.
    pub fn lerp(&self, t: f32) -> Pose {
        interpolate(self.prev, self.pose, t)
    }
}

/// Motor interpolation: `a exp(t log(~a b))`, a screw motion from `a` to `b`.
pub fn interpolate(a: Pose, b: Pose, t: f32) -> Pose {
    let rel = a.reverse() * b;
    let log: Point<(), f32> = rel.log();
    a * log.gp(t).exp()
}

/// `|m ~m - 1|`: how far a pose has drifted from a unit motor (logged per session).
pub fn drift(m: Pose) -> f32 {
    let x = m.into_inner();
    let n = x * x.reverse();
    (n.c[0] - 1.0)
        .abs()
        .max(n.c[1].abs())
        .max(n.c[2].abs())
        .max(n.c[3].abs())
}

/// The angle of a direction (a point at infinity), counterclockwise from the x axis.
pub fn angle_of(d: Point<(), f32>) -> f32 {
    d.e01().atan2(d.e20())
}

/// The length of a direction's Euclidean part.
pub fn length(d: Point<(), f32>) -> f32 {
    (d.e20() * d.e20() + d.e01() * d.e01()).sqrt()
}

/// The Euclidean distance between two finite points, as the norm of their join.
pub fn distance(p: Point<(), f32>, q: Point<(), f32>) -> f32 {
    (p.normalized().into_inner() & q.normalized().into_inner()).norm()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: [f32; 2], b: [f32; 2]) -> bool {
        (a[0] - b[0]).abs() < 1e-4 && (a[1] - b[1]).abs() < 1e-4
    }

    #[test]
    fn a_body_moves_and_turns_as_its_twist_says() {
        let mut b = Body::new(pose_at(1.0, 2.0, 0.0));
        b.vel = Point::direction(3.0, -1.0);
        for _ in 0..120 {
            b.step(1.0 / 120.0);
        }
        assert!(close(b.xy(), [4.0, 1.0]), "{:?}", b.xy());
        // Spinning in place: the heading turns, the centre stays.
        let mut s = Body::new(pose_at(1.0, 2.0, 0.0));
        s.spin = core::f32::consts::FRAC_PI_2;
        for _ in 0..120 {
            s.step(1.0 / 120.0);
        }
        assert!(close(s.xy(), [1.0, 2.0]), "{:?}", s.xy());
        let h = s.heading();
        assert!(close([h.e20(), h.e01()], [0.0, 1.0]), "{h:?}");
    }

    #[test]
    fn interpolation_is_a_screw_between_the_ends() {
        let a = pose_at(0.0, 0.0, 0.0);
        let b = pose_at(2.0, 0.0, 1.0);
        assert!(close(
            (interpolate(a, b, 0.0) >> Point::xy(0.0, 0.0)).to_euclidean(),
            [0.0, 0.0]
        ));
        assert!(close(
            (interpolate(a, b, 1.0) >> Point::xy(0.0, 0.0)).to_euclidean(),
            [2.0, 0.0]
        ));
        let mid = interpolate(a, b, 0.5);
        assert!(drift(mid) < 1e-5);
    }

    #[test]
    fn renormalization_keeps_long_runs_on_the_unit_condition() {
        let mut b = Body::new(pose_at(0.0, 0.0, 0.3));
        b.vel = Point::direction(1.7, -0.4);
        b.spin = 2.3;
        for _ in 0..120 * 600 {
            b.step(1.0 / 120.0);
        }
        assert!(
            drift(b.pose) < 1e-5,
            "drift after 10 minutes: {}",
            drift(b.pose)
        );
    }
}
