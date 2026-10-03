//! numga's `mechanics/lie_integrators.py`: Lie-group integrators for rigid bodies, on motors
//! and rates.
//!
//! The state of a body is a motor in the Lie group, its pose, and a rate in the Lie algebra, a
//! bivector of angular (and, in PGA, linear) velocity in the body frame. Inertia maps a rate to
//! its momentum, a forque; its inverse maps a forque back to a rate. External forques come from
//! a function of the motor and the rate.
//!
//! The steppers are written once, as the default methods of [`Lie`], over any algebra's
//! [`gax::motions::Motions`]: the rotors of any dimension and the motors of PGA, as numga's one
//! module reads its algebra from the arguments. `Vga3d::explicit_rk4(...)` steps a rotating
//! body in space, `Pga3d::explicit_rk4(...)` a rigid one; an algebra declared with
//! `gax::algebra!` joins in with `gax::motions!`. Where rotations commute (the plane), the
//! commutators are zero and the gyroscopic term and the dexpinv correction drop out by
//! themselves.
//!
//! Methods: explicit symplectic Verlet (Lie-Euler with the rate recovered from the step),
//! explicit RK4 on Lie groups, and explicit 4th-order Munthe-Kaas (RKMK4) with polynomial
//! dexpinv. numga's implicit Lie-Newmark and variational Lie-Verlet steppers are not ported.
//!
//! References: Hairer, Lubich, Wanner, *Geometric Numerical Integration*; Krysl, *Explicit
//! Newmark-type integrators on Lie groups*.

#![allow(dead_code)]

use core::ops::{Add, Mul};
use gax::Of;
use gax::motions::{Linear, Motions};

/// The classical 4th-order Runge-Kutta step of `y' = f(y)`.
pub fn rk4<Y>(f: impl Fn(Y) -> Y, y: Y, h: f64) -> Y
where
    Y: Copy + Add<Output = Y> + Mul<f64, Output = Y>,
{
    let k1 = f(y);
    let k2 = f(y + k1 * (0.5 * h));
    let k3 = f(y + k2 * (0.5 * h));
    let k4 = f(y + k3 * h);
    y + (k2 + k3 + (k1 + k4) * 0.5) * (h / 3.0)
}

/// Two values stepped together by [`rk4`] (numga stacks them into one extensor).
#[derive(Clone, Copy, Debug)]
pub struct Pair<A>(pub A, pub A);

impl<A: Add<Output = A>> Add for Pair<A> {
    type Output = Self;
    fn add(self, o: Self) -> Self {
        Pair(self.0 + o.0, self.1 + o.1)
    }
}

impl<A: Mul<f64, Output = A>> Mul<f64> for Pair<A> {
    type Output = Self;
    fn mul(self, s: f64) -> Self {
        Pair(self.0 * s, self.1 * s)
    }
}

/// A pose: a unit motor (or rotor).
pub type M<G> = <G as Motions<f64>>::Motor;
/// A rate in the body frame.
pub type R<G> = <G as Motions<f64>>::Twist;
/// A forque (a momentum, or a force line).
pub type F<G> = <G as Motions<f64>>::Forque;
/// A mass point.
pub type P<G> = <G as Motions<f64>>::Point;
/// Rate to momentum.
pub type Inertia<G> = <G as Motions<f64>>::Inertia;
/// Momentum to rate.
pub type InertiaInv<G> = <G as Motions<f64>>::Mobility;
/// A linear map on rates.
pub type RateMap<G> = <G as Motions<f64>>::TwistMap;
/// A stepper: the motor and rate after one step of `dt` under the external forque.
pub type Step<G> =
    fn(M<G>, R<G>, Inertia<G>, InertiaInv<G>, f64, &dyn Fn(M<G>, R<G>) -> F<G>) -> (M<G>, R<G>);

/// The rigid bodies of space, in PGA3D: the algebra and its kinds by role. Rates and forques
/// are both lines, and an inertia and its inverse both maps of lines.
pub mod rigid {
    use gax::pga3d::{Line, Motor, Point};

    /// The algebra.
    pub type G = gax::motions::Pga3d;
    /// A pose.
    pub type M = gax::Unit<Motor<(), f64>>;
    /// A rate, in the body frame.
    pub type R = Line<(), f64>;
    /// A forque.
    pub type F = Line<(), f64>;
    /// A mass point.
    pub type P = Point<(), f64>;
    /// Rate to momentum.
    pub type Inertia = Line<(Line,), f64>;
    /// Momentum to rate.
    pub type InertiaInv = Line<(Line,), f64>;
}

/// The Lie-group steppers, for every algebra's [`Motions`].
pub trait Lie: Motions<f64> {
    /// The gyroscopic forque of a body with momentum `p` turning at rate `r`: `[p, r]`.
    fn gyroscopic(p: F<Self>, r: R<Self>) -> F<Self> {
        Self::coadjoint(p, r)
    }

    /// `dexpinv` at `h`, a map on rates: a polynomial in the adjoint `ad = 2 [h, ·]` (gax's
    /// commutator, like numga's, is half the bracket), to the order an RK4 step needs:
    /// `1 + ad / 2 + ad² / 12`.
    fn dexpinv(h: R<Self>) -> RateMap<Self> {
        let ad = Self::ad(h) * 2.0;
        Self::identity_map() + ad * 0.5 + ad.of(ad) * (1.0 / 12.0)
    }

    /// The inertia of point masses and its inverse. Each point contributes the
    /// rate-to-momentum map `p & [p, ·]`: the join of the point with its own velocity under an
    /// open rate.
    fn inertia_of(points: &[P<Self>], masses: &[f64]) -> (Inertia<Self>, InertiaInv<Self>) {
        let mut inertia = Inertia::<Self>::zero();
        for (p, m) in points.iter().zip(masses) {
            inertia += Self::point_inertia(*p) * *m;
        }
        (inertia, Self::mobility(inertia))
    }

    /// The inertia of a point cloud of unit masses, and its inverse.
    fn inertia_from_points(points: &[P<Self>]) -> (Inertia<Self>, InertiaInv<Self>) {
        Self::inertia_of(points, &vec![1.0; points.len()])
    }

    /// The kinetic energy `(I(r) & r) / 2`.
    fn kinetic_energy(rate: R<Self>, inertia: Inertia<Self>) -> f64 {
        Self::pair(inertia.of(rate), rate) * 0.5
    }

    /// No external forque: a free body.
    fn free(_motor: M<Self>, _rate: R<Self>) -> F<Self> {
        F::<Self>::zero()
    }

    /// The external forque less the gyroscopic one, `forque(m, r) - [I(r), r]`.
    fn net_forque(
        forque: &dyn Fn(M<Self>, R<Self>) -> F<Self>,
        inertia: Inertia<Self>,
        motor: M<Self>,
        rate: R<Self>,
    ) -> F<Self> {
        forque(motor, rate) - Self::gyroscopic(inertia.of(rate), rate)
    }

    /// The rate a step from `before` to `after` implies over `dt`:
    /// `log(~before after) (-2 / dt)`.
    fn rate_between(before: M<Self>, after: M<Self>, dt: f64) -> R<Self> {
        Self::log(Self::reverse(before) * after) * (-2.0 / dt)
    }

    /// Verlet pre- and post-integration (unconstrained): RK4 on the rate, a motor step along
    /// the predicted rate, renormalized as numga does (one Newton step towards `m ~m = 1`,
    /// exact to rounding for a product of two units), and the rate recovered from the step.
    fn explicit_verlet(
        motor: M<Self>,
        rate: R<Self>,
        inertia: Inertia<Self>,
        inertia_inv: InertiaInv<Self>,
        dt: f64,
        forque: &dyn Fn(M<Self>, R<Self>) -> F<Self>,
    ) -> (M<Self>, R<Self>) {
        let dr = |r: R<Self>| inertia_inv.of(Self::net_forque(forque, inertia, motor, r));
        let predicted = rk4(dr, rate, dt);
        let next = Self::renormalize(motor * Self::exp(predicted * (-dt / 2.0)));
        (next, Self::rate_between(motor, next, dt))
    }

    /// RK4 on the rate, then a motor step with the stepped rate.
    fn explicit_rk4(
        motor: M<Self>,
        rate: R<Self>,
        inertia: Inertia<Self>,
        inertia_inv: InertiaInv<Self>,
        dt: f64,
        forque: &dyn Fn(M<Self>, R<Self>) -> F<Self>,
    ) -> (M<Self>, R<Self>) {
        let dr = |r: R<Self>| inertia_inv.of(Self::net_forque(forque, inertia, motor, r));
        let next = rk4(dr, rate, dt);
        (motor * Self::exp(next * (-dt / 2.0)), next)
    }

    /// Explicit 4th-order Munthe-Kaas: the motor step is `motor h.exp()`, with the bivector
    /// `h` integrated in the Lie algebra by classical RK4 alongside the rate. The time
    /// derivative of `motor h.exp()` must equal that motor times `r (-1/2)`; inverting the
    /// derivative of the exponential gives the rate of `h`, `dexpinv(h)(r (-1/2))`. Nothing
    /// here depends on the dimension.
    fn explicit_rkmk4(
        motor: M<Self>,
        rate: R<Self>,
        inertia: Inertia<Self>,
        inertia_inv: InertiaInv<Self>,
        dt: f64,
        forque: &dyn Fn(M<Self>, R<Self>) -> F<Self>,
    ) -> (M<Self>, R<Self>) {
        let dr = |m: M<Self>, r: R<Self>| inertia_inv.of(Self::net_forque(forque, inertia, m, r));
        let dh = |h: R<Self>, r: R<Self>| Self::dexpinv(h).of(r * -0.5);
        // The state (h, rate) is a pair of bivectors, stepped by the plain RK4.
        let derivative = |s: Pair<R<Self>>| Pair(dh(s.0, s.1), dr(motor * Self::exp(s.0), s.1));
        let Pair(h, next) = rk4(derivative, Pair(rate * 0.0, rate), dt);
        (motor * Self::exp(h), next)
    }
}

impl<G: Motions<f64>> Lie for G {}
