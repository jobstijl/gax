//! numga's `derived.py`: the couplings found by differentiation. The gaps are written once, as
//! a function of how far each body moves and each mode deforms over the step, and the couplings
//! are its derivatives (`gax::jet::jacobian`, forward-mode jets through the motor's
//! exponential and sandwich), in place of the commutators and mode shapes built by hand.

use crate::girder::P;
use crate::xpbd::{Bodies, Constraints, Coupling, M, SIDES, per_side};
use gax::Real;
use gax::jet::jacobian;
use gax::pga2d::{Direction, Point, Scalar, Twist};

/// A side's anchor in the world: its point in the body's frame moved by the motor after a twist
/// of the body over the step. Generic over the coefficients, for the jets that carry the
/// derivative.
fn placed<T: Real>(motor: M, twist: Twist<(), T>, local: Point<(), T>) -> Direction<(), T> {
    let step = (twist * T::from_f64(-0.5)).exp().into_inner();
    let moved = motor.into_inner().map_coefs(T::from_f64) * step;
    (moved >> local).cast::<Direction>()
}

/// A point of the plane with jets for coefficients.
fn lift<T: Real>(p: P) -> Point<(), T> {
    p.map_coefs(T::from_f64)
}

/// A side's anchor in the world at amplitude `a` of a mode whose shape at the anchor is
/// `shape`, from the anchor as it is, at amplitude `now`.
fn deformed<T: Real>(motor: M, local: P, shape: P, now: f64, a: Scalar<(), T>) -> Direction<(), T> {
    let offset = lift::<T>(shape) * (a.s() - T::from_f64(now));
    placed(motor, Twist::zero(), lift(local) + offset)
}

/// The couplings as the derivatives of each side's signed anchor: by its body's twist, at no
/// twist, and by its body's amplitude of each mode, at the amplitude it has.
pub fn coupling(bodies: &Bodies, constraints: &Constraints) -> Coupling {
    let count = bodies.motor.len();
    let side = |c: usize, side: usize| {
        let body = constraints.bodies[c][side];
        (
            bodies.motor[body],
            constraints.local_anchor(bodies, c, side),
            body,
        )
    };
    let rigid = per_side(constraints, count, |c, s| {
        let (motor, local, _) = side(c, s);
        let (_, change) = jacobian(|twist| placed(motor, twist, lift(local)), Twist::zero());
        change * SIDES[s]
    });
    let modal = (0..constraints.modes.len())
        .map(|m| {
            per_side(constraints, count, |c, s| {
                let (motor, local, body) = side(c, s);
                let shape = constraints.modes[m][c][s].cast::<Point>();
                let now = bodies.amplitudes[m].site(body);
                let at = |a| deformed(motor, local, shape, now.s(), a);
                let (_, change) = jacobian(at, now);
                change * SIDES[s]
            })
        })
        .collect();
    Coupling {
        rigid,
        modal,
        gap: constraints.gaps(bodies),
    }
}
