//! numga's `mechanics/lie_integrators.py`: Lie-group integrators for rigid bodies, on motors
//! and rates.
//!
//! The state of a body is a motor in the Lie group, its pose, and a rate in the Lie algebra, a
//! bivector of angular (and, in PGA, linear) velocity in the body frame. Inertia maps a rate to
//! its momentum, a forque; its inverse maps a forque back to a rate. External forques come from
//! a function of the motor and the rate.
//!
//! The steppers are written once, in [`integrators!`], and instantiated per algebra: a module
//! imports its kinds under the names `Motor` (the even versor), `Rate` (the bivectors), `Forque`
//! (the antibivectors: the join of two points) and `Point` (the antivectors), then invokes the
//! macro, which expands to the same lines over those kinds. So one text integrates rotors of
//! any dimension and the motors of PGA, as numga's one module does by reading its algebra from
//! the arguments:
//!
//! ```ignore
//! mod spin3 {
//!     use gax::vga3d::{Bivector as Rate, Bivector as Point, Rotor as Motor, Vector as Forque};
//!     crate::lie::integrators!();
//! }
//! ```
//!
//! In two dimensions rotations commute and gax has no commutator of a rotation's bivector with
//! anything rotational (the product is identically zero, so it does not exist): there the
//! invocation is `integrators!(commutative)`, which drops the gyroscopic term and the dexpinv
//! correction.
//!
//! Methods: explicit symplectic Verlet (Lie-Euler with the rate recovered from the step),
//! explicit RK4 on Lie groups, and explicit 4th-order Munthe-Kaas (RKMK4) with polynomial
//! dexpinv. numga's implicit Lie-Newmark and variational Lie-Verlet steppers are not ported.
//!
//! References: Hairer, Lubich, Wanner, *Geometric Numerical Integration*; Krysl, *Explicit
//! Newmark-type integrators on Lie groups*.

#![allow(dead_code)]

use core::ops::{Add, Mul};

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

/// The Lie-group steppers over the kinds `Motor`, `Rate`, `Forque` and `Point` in scope at the
/// invocation (module docs). Every item allows dead code: each example uses a few of them. `integrators!()` for a non-commutative group, the rotors of three
/// or more dimensions and the motors of PGA; `integrators!(commutative)` for the rotations of
/// the plane.
macro_rules! integrators {
    () => {
        /// The gyroscopic forque of a body with momentum `p` turning at rate `r`: `[p, r]`.
        #[allow(dead_code)]
        pub fn gyroscopic(p: F, r: R) -> F {
            p.commutator(r)
        }

        /// `dexpinv` at `h`, a map on rates: a polynomial in the adjoint `ad = 2 [h, ·]`, the
        /// commutator with an open bivector slot (gax's commutator, like numga's, is half the
        /// bracket), to the order an RK4 step needs: `1 + ad / 2 + ad² / 12`.
        #[allow(dead_code)]
        pub fn dexpinv(h: R) -> RateMap {
            let ad = h.commutator(Rate::slot()) * 2.0;
            Rate::slot() + ad * 0.5 + ad.of(ad) * (1.0 / 12.0)
        }

        crate::lie::steppers!();
    };
    (commutative) => {
        /// In the plane rotations commute: no gyroscopic forque.
        #[allow(dead_code)]
        pub fn gyroscopic(_p: F, _r: R) -> F {
            Forque::zero()
        }

        /// In the plane rotations commute: `dexpinv` is the identity.
        #[allow(dead_code)]
        pub fn dexpinv(_h: R) -> RateMap {
            Rate::slot()
        }

        crate::lie::steppers!();
    };
}

/// The part of [`integrators!`] shared by both groups.
macro_rules! steppers {
    () => {
        /// A pose: a unit motor (or rotor).
        #[allow(dead_code)]
        pub type M = gax::Unit<Motor<(), f64>>;
        /// A rate in the body frame.
        #[allow(dead_code)]
        pub type R = Rate<(), f64>;
        /// A forque (a momentum, or a force line).
        #[allow(dead_code)]
        pub type F = Forque<(), f64>;
        /// A mass point.
        #[allow(dead_code)]
        pub type P = Point<(), f64>;
        /// Rate to momentum.
        #[allow(dead_code)]
        pub type Inertia = Forque<(Rate,), f64>;
        /// Momentum to rate.
        #[allow(dead_code)]
        pub type InertiaInv = Rate<(Forque,), f64>;
        /// A linear map on rates.
        #[allow(dead_code)]
        pub type RateMap = Rate<(Rate,), f64>;
        /// A stepper: the motor and rate after one step of `dt` under the external forque.
        #[allow(dead_code)]
        pub type Step = fn(M, R, Inertia, InertiaInv, f64, &dyn Fn(M, R) -> F) -> (M, R);

        /// The inertia of point masses and its inverse. Each point contributes the
        /// rate-to-momentum map `p & [p, ·]`: the join of the point with its own velocity
        /// under an open rate, the regressive product of the point with its commutator with an
        /// open bivector.
        #[allow(dead_code)]
        pub fn inertia_of(points: &[P], masses: &[f64]) -> (Inertia, InertiaInv) {
            let mut inertia = Inertia::zero();
            for (p, m) in points.iter().zip(masses) {
                inertia += (*p & p.commutator(Rate::slot())) * *m;
            }
            (inertia, inertia.inverse())
        }

        /// The inertia of a point cloud of unit masses, and its inverse.
        #[allow(dead_code)]
        pub fn inertia_from_points(points: &[P]) -> (Inertia, InertiaInv) {
            inertia_of(points, &vec![1.0; points.len()])
        }

        /// The kinetic energy `(I(r) & r) / 2`.
        #[allow(dead_code)]
        pub fn kinetic_energy(rate: R, inertia: Inertia) -> f64 {
            (inertia.of(rate) & rate).s() * 0.5
        }

        /// No external forque: a free body.
        #[allow(dead_code)]
        pub fn free(_motor: M, _rate: R) -> F {
            Forque::zero()
        }

        /// The external forque less the gyroscopic one, `forque(m, r) - [I(r), r]`.
        #[allow(dead_code)]
        pub fn net_forque(forque: &dyn Fn(M, R) -> F, inertia: Inertia, motor: M, rate: R) -> F {
            forque(motor, rate) - gyroscopic(inertia.of(rate), rate)
        }

        /// The rate a step from `before` to `after` implies over `dt`:
        /// `log(~before after) (-2 / dt)`.
        #[allow(dead_code)]
        pub fn rate_between(before: M, after: M, dt: f64) -> R {
            let log: R = (before.reverse() * after).log();
            log * (-2.0 / dt)
        }

        /// Verlet pre- and post-integration (unconstrained): RK4 on the rate, a motor step along
        /// the predicted rate, and the rate recovered from the step. numga normalizes the
        /// stepped motor; gax has no closed-form `normalized` for the even versors of five
        /// dimensions and up (`x ~x` has a 4-vector part there), nor `renormalize_fast`, so this
        /// takes the Newton step `x (3 - ~x x) / 2` towards `x ~x = 1` itself, which for a
        /// product of two units is exact to rounding.
        #[allow(dead_code)]
        pub fn explicit_verlet(
            motor: M,
            rate: R,
            inertia: Inertia,
            inertia_inv: InertiaInv,
            dt: f64,
            forque: &dyn Fn(M, R) -> F,
        ) -> (M, R) {
            let dr = |r: R| inertia_inv.of(net_forque(forque, inertia, motor, r));
            let predicted = crate::lie::rk4(dr, rate, dt);
            let next = (motor * (predicted * (-dt / 2.0)).exp()).into_inner();
            let next = gax::Unit::new_unchecked(next * 1.5 - next * (next.reverse() * next) * 0.5);
            (next, rate_between(motor, next, dt))
        }

        /// RK4 on the rate, then a motor step with the stepped rate.
        #[allow(dead_code)]
        pub fn explicit_rk4(
            motor: M,
            rate: R,
            inertia: Inertia,
            inertia_inv: InertiaInv,
            dt: f64,
            forque: &dyn Fn(M, R) -> F,
        ) -> (M, R) {
            let dr = |r: R| inertia_inv.of(net_forque(forque, inertia, motor, r));
            let next = crate::lie::rk4(dr, rate, dt);
            (motor * (next * (-dt / 2.0)).exp(), next)
        }

        /// Explicit 4th-order Munthe-Kaas: the motor step is `motor h.exp()`, with the bivector
        /// `h` integrated in the Lie algebra by classical RK4 alongside the rate. The time
        /// derivative of `motor h.exp()` must equal that motor times `r (-1/2)`; inverting the
        /// derivative of the exponential gives the rate of `h`, `dexpinv(h)(r (-1/2))`. Nothing
        /// here depends on the dimension.
        #[allow(dead_code)]
        pub fn explicit_rkmk4(
            motor: M,
            rate: R,
            inertia: Inertia,
            inertia_inv: InertiaInv,
            dt: f64,
            forque: &dyn Fn(M, R) -> F,
        ) -> (M, R) {
            let dr = |m: M, r: R| inertia_inv.of(net_forque(forque, inertia, m, r));
            let dh = |h: R, r: R| dexpinv(h).of(r * -0.5);
            // The state (h, rate) is a pair of bivectors, stepped by the plain RK4.
            let derivative =
                |s: crate::lie::Pair<R>| crate::lie::Pair(dh(s.0, s.1), dr(motor * s.0.exp(), s.1));
            let crate::lie::Pair(h, next) =
                crate::lie::rk4(derivative, crate::lie::Pair(rate * 0.0, rate), dt);
            (motor * h.exp(), next)
        }
    };
}

pub(crate) use integrators;
pub(crate) use steppers;
