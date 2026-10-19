//! Kinematics in gax terms: a pose is a unit PGA2D motor, a velocity is a twist (a PGA2D
//! bivector, which is a `Point`), and integration is `pose ← exp(dt B) pose`, renormalized with
//! one Newton step every tick (docs/numerics.md: drift policy).

use super::collide::{Wall, outside};
use super::dir;
use crate::geom::ORIGIN;
use gax::Unit;
use gax::pga2d::{Motor, Point, Twist};

/// A pose.
pub type Pose = Unit<Motor<(), f32>>;

/// The pose at the point `p`, turned by `angle` (counterclockwise from the x axis).
pub fn pose_at(p: Point<(), f32>, angle: f32) -> Pose {
    Motor::translation(p.position() - ORIGIN) * Motor::rotation(ORIGIN, angle)
}

/// A moving rigid body.
#[derive(Clone, Copy, Debug, PartialEq)]
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
            vel: Point::zero(),
            spin: 0.0,
        }
    }

    /// Its position.
    pub fn pos(&self) -> Point<(), f32> {
        self.pose >> ORIGIN
    }

    /// Its position at the start of the last tick.
    pub fn prev_pos(&self) -> Point<(), f32> {
        self.prev >> ORIGIN
    }

    /// The direction its local x axis points to.
    pub fn heading(&self) -> Point<(), f32> {
        self.pose >> dir(1.0, 0.0)
    }

    /// The world-frame twist of its motion: translation at `vel` plus rotation at `spin`
    /// about its centre. Twists add.
    pub fn twist(&self) -> Twist<(), f32> {
        Twist::translation(self.vel) + Twist::rotation(self.pos(), self.spin)
    }

    /// Advance by `dt`: `pose ← exp(dt B) pose`, then one Newton step towards `m ~m = 1`.
    pub fn step(&mut self, dt: f32) {
        self.prev = self.pose;
        self.pose = (self.twist() * dt).exp().mul_renormalized(self.pose);
    }

    /// Move by the direction `d` in world coordinates.
    pub fn shift(&mut self, d: Point<(), f32>) {
        self.pose = Motor::translation(d) * self.pose;
    }

    /// Move its position to the point `p`, keeping its orientation (a wall bounce).
    pub fn move_to(&mut self, p: Point<(), f32>) {
        self.shift(p.unitized() - self.pos());
    }

    /// Back inside the arena after a step out of it: each wall the last step crossed (`margin`
    /// in from the edge; two at a corner) handed to `back`, which moves the body in.
    pub fn walls_crossed(&mut self, margin: f32, mut back: impl FnMut(&mut Body, Wall)) {
        for _ in 0..2 {
            let Some(wall) = outside(self.prev_pos(), self.pos(), margin) else {
                break;
            };
            back(self, wall);
        }
    }

    /// Onto the wall `wall`: to its position's foot there.
    pub fn stop_at(&mut self, wall: Wall) {
        self.move_to(crate::geom::foot(self.pos(), wall).unitized());
    }

    /// The drawn pose between the last two ticks: `prev exp(t log(~prev pose))`.
    pub fn interpolated(&self, t: f32) -> Pose {
        interpolate(self.prev, self.pose, t)
    }
}

/// Motor interpolation: `a exp(t log(~a b))`, a screw motion from `a` to `b` the shorter way,
/// renormalized (the serpent's chain feeds it its own output every tick).
pub fn interpolate(a: Pose, b: Pose, t: f32) -> Pose {
    Motor::interpolate(a, b, t).renormalize_fast()
}

/// `|m ~m - 1|`: how far a pose has drifted from a unit motor (logged per session).
pub fn drift(m: Pose) -> f32 {
    // A PGA2D motor's `m ~m` is a scalar.
    (m.into_inner().norm_squared() - 1.0).abs()
}

/// The velocity of the point `p` (a unit point) of a body moving with the world twist `twist`:
/// the rate of `exp(t B) p exp(-t B)`, `B p - p B`, twice gax's commutator (which is half the
/// bracket).
pub fn velocity_at(twist: Twist<(), f32>, p: Point<(), f32>) -> Point<(), f32> {
    twist.commutator(p) * 2.0
}

/// The direction `d` turned by `angle`.
pub fn turned(d: Point<(), f32>, angle: f32) -> Point<(), f32> {
    Motor::rotation(ORIGIN, angle) >> d
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::at;

    /// Two points within rounding of each other.
    fn close(a: Point<(), f32>, b: Point<(), f32>) -> bool {
        a.distance(b) < 1e-4
    }

    #[test]
    fn a_body_moves_and_turns_as_its_twist_says() {
        let mut b = Body::new(pose_at(at(1.0, 2.0), 0.0));
        b.vel = dir(3.0, -1.0);
        for _ in 0..120 {
            b.step(1.0 / 120.0);
        }
        assert!(close(b.pos(), at(4.0, 1.0)), "{:?}", b.pos());
        // Spinning in place: the heading turns, the centre stays.
        let mut s = Body::new(pose_at(at(1.0, 2.0), 0.0));
        s.spin = core::f32::consts::FRAC_PI_2;
        for _ in 0..120 {
            s.step(1.0 / 120.0);
        }
        assert!(close(s.pos(), at(1.0, 2.0)), "{:?}", s.pos());
        // Directions: their difference has no length.
        let h = s.heading();
        assert!((h - dir(0.0, 1.0)).ideal_norm() < 1e-4, "{h:?}");
    }

    #[test]
    fn interpolation_is_a_screw_between_the_ends() {
        let a = Pose::identity();
        let b = pose_at(at(2.0, 0.0), 1.0);
        assert!(close(interpolate(a, b, 0.0) >> ORIGIN, ORIGIN));
        assert!(close(interpolate(a, b, 1.0) >> ORIGIN, at(2.0, 0.0)));
        let mid = interpolate(a, b, 0.5);
        assert!(drift(mid) < 1e-5);
    }

    #[test]
    fn renormalization_keeps_long_runs_on_the_unit_condition() {
        let mut b = Body::new(pose_at(ORIGIN, 0.3));
        b.vel = dir(1.7, -0.4);
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

    #[test]
    fn a_twist_moves_a_point_with_its_velocity_and_spin() {
        // v + ω × r, with r the arm from the centre: the formula the commutator replaces.
        let mut b = Body::new(Motor::translation(dir(1.0, 2.0)));
        b.vel = dir(3.0, -1.0);
        b.spin = 0.5;
        let p = at(4.0, 6.0);
        let v = velocity_at(b.twist(), p);
        let want = [3.0 - 0.5 * 4.0, -1.0 + 0.5 * 3.0];
        assert!(
            (v.e20() - want[0]).abs() < 1e-5 && (v.e01() - want[1]).abs() < 1e-5,
            "{v:?}"
        );
    }
}
