//! Typed derivatives (`gax::jet::{jacobian, gradient, hessian}`): the derivative of a function
//! of a gax value as an extensor, against central differences, against the loose-parameter
//! forms, and against numga's `derivative` (JAX).

#![cfg(feature = "pga3d")]
// Index loops read like the formulas: the Hessian's rows and columns, the fixture's flat kernels.
#![allow(clippy::needless_range_loop)]
// The results' types are spelled out: what the derivatives are is what these tests check.
#![allow(clippy::type_complexity)]

#[path = "support/rng.rs"]
mod rng;
use rng::{Draw, rng};

use gax::jet::{gradient, gradient_array, hessian, jacobian};
use gax::pga3d::{Line, Motor, Plane, Point, Scalar};
use gax::{ApproxEq, Extensor, Kind, Real, Retype};

/// The central difference of `f` at `x` along the coefficient `i` of `x`.
fn central<X, Y>(f: impl Fn(X) -> Y, x: X, i: usize) -> Retype<Y, Y::Slots, f64>
where
    X: Extensor<Slots = (), Coef = f64>,
    Y: Extensor<Coef = f64> + core::ops::Sub<Output = Y>,
{
    let h = 1e-6;
    let step = |s: f64| {
        let mut c = *x.coeffs();
        c.as_mut()[i] += s;
        X::from_coeffs(c)
    };
    (f(step(h)) - f(step(-h))).map_coefs(|v| v / (2.0 * h))
}

/// The unit basis value `i` of a kind.
fn basis<K: Kind>(i: usize) -> K::Mv<(), f64> {
    Extensor::from_coeffs(K::arr_from_fn(|k| if k == i { 1.0 } else { 0.0 }))
}

fn random_line(rng: &mut impl Draw, scale: f64) -> Line<(), f64> {
    Line::from_coeffs(core::array::from_fn(|_| scale * rng.next_f64()))
}

/// The motor of a twist: `jacobian` gives a `Motor <- Line` whose action on each basis twist
/// is the central difference along it.
#[test]
fn jacobian_of_exp_is_a_map() {
    let mut rng = rng(0x1ac0);
    for _ in 0..100 {
        let b = random_line(&mut rng, 0.8);
        let (m, dm): (Motor<(), f64>, Motor<(Line,), f64>) = jacobian(|b| b.exp().into_inner(), b);
        assert!(m.approx_eq(&b.exp().into_inner(), 1e-15));
        for i in 0..6 {
            let fd = central(|b: Line<(), f64>| b.exp().into_inner(), b, i);
            let ad = dm.of(basis::<Line>(i));
            assert!(ad.approx_eq(&fd, 1e-8), "{i}: {ad:?} vs {fd:?}");
        }
    }
}

/// A map-valued function: the derivative keeps the map's own slot first and appends the step's,
/// numga's convention. `b ↦ exp(b) >> Point::slot()` is a `Point <- Point`, its derivative a
/// `Point <- (Point, Line)`.
#[test]
fn jacobian_of_a_map_appends_the_slot() {
    let mut rng = rng(0x51a7);
    for _ in 0..50 {
        let b = random_line(&mut rng, 0.8);
        let f = |b: Line<(), f64>| b.exp() >> Point::slot();
        let (t, dt): (Point<(Point,), f64>, Point<(Point, Line), f64>) =
            jacobian(|b| b.exp() >> Point::slot(), b);
        assert!(t.approx_eq(&f(b), 1e-14));
        let p = Point::xyz(rng.next_f64(), rng.next_f64(), rng.next_f64());
        for i in 0..6 {
            // Binding the point first leaves the derivative of the moved point, a `Point <- Line`.
            let ad = dt.of(p).of(basis::<Line>(i));
            let fd = central(|b: Line<(), f64>| f(b).of(p), b, i);
            assert!(ad.approx_eq(&fd, 1e-8), "{i}: {ad:?} vs {fd:?}");
        }
    }
}

/// The squared distance from `p` moved by `exp(b)` to `q`.
fn misfit<T: Real>(b: Line<(), T>, p: Point<(), f64>, q: Point<(), f64>) -> T {
    let r = (b.exp() >> p.map_coefs(T::from_f64)).to_euclidean();
    let q = q.map_coefs(T::from_f64).to_euclidean();
    (0..3)
        .map(|i| (r[i] - q[i]) * (r[i] - q[i]))
        .fold(T::zero(), |a, d| a + d)
}

/// The gradient is the linear form `Scalar <- Line`, the same numbers as the loose-parameter
/// gradient over the six coefficients.
#[test]
fn gradient_is_a_form() {
    let mut rng = rng(0x9fad);
    let p = Point::xyz(0.3, -0.4, 1.2);
    let q = Point::xyz(1.0, 0.5, -0.25);
    for _ in 0..100 {
        let b = random_line(&mut rng, 0.8);
        let (v, g): (f64, Scalar<(Line,), f64>) = gradient(|b| misfit(b, p, q), b);
        let (v2, g2) = gradient_array(|c| misfit(Line::from_coeffs(c), p, q), *b.coeffs());
        assert_eq!(v, v2);
        assert_eq!(g.c[0], g2);
        for i in 0..6 {
            let fd = central(|b| Scalar::new(misfit(b, p, q)), b, i).s();
            assert!((g.of(basis::<Line>(i)).s() - fd).abs() < 1e-7);
        }
    }
}

/// A function returning a scalar-kind value works as one returning the coefficient.
#[test]
fn gradient_of_a_scalar_value() {
    let x = Plane::<(), f64>::new(1.0, 2.0, 3.0, 4.0);
    let (v, g) = gradient(|x| x.dot(x), x);
    let (w, h) = gradient(Plane::norm_squared, x);
    assert_eq!((v, g), (w, h));
    assert_eq!(v, 14.0);
    // The metric ignores e0: no slope along it.
    assert_eq!(g.c[0], [2.0, 4.0, 6.0, 0.0]);
}

/// A scalar of a moved point, with every kind of term the closed forms meet.
fn energy<T: Real>(b: Line<(), T>) -> T {
    let m = b.exp();
    let p = m >> Point::xyz(1.0, 0.5, -0.5).map_coefs(T::from_f64);
    let [x, y, z] = p.to_euclidean();
    x * x * y + (z * x).sin() + y * y * y * z
}

/// The Hessian is the symmetric form `Scalar <- (Line, Line)`, the central difference of the
/// gradient; its value and gradient agree with `gradient`'s.
#[test]
fn hessian_is_the_derivative_of_the_gradient() {
    let mut rng = rng(0x4e55);
    for _ in 0..50 {
        let b = random_line(&mut rng, 0.8);
        let (v, g, h): (f64, Scalar<(Line,), f64>, Scalar<(Line, Line), f64>) = hessian(energy, b);
        let (v1, g1) = gradient(energy, b);
        assert!((v - v1).abs() < 1e-14 && g.approx_eq(&g1, 1e-13));
        for i in 0..6 {
            for j in 0..6 {
                assert!((h.c[0][i][j] - h.c[0][j][i]).abs() < 1e-12);
            }
            let fd = central(|b| gradient(energy, b).1, b, i);
            // Binding the first slot leaves the change of the gradient along that direction.
            let ad = h.of(basis::<Line>(i));
            assert!(ad.approx_eq(&fd, 1e-6), "{i}: {ad:?} vs {fd:?}");
        }
    }
}

/// Newton on a quadratic lands in one step: `-hessian.solve(gradient)` is the minimizer.
#[test]
fn newton_step_on_a_quadratic() {
    // ½ xᵀ A x − bᵀ x on planes, with A symmetric positive definite.
    fn energy<T: Real>(x: Plane<(), T>) -> T {
        let c = T::from_f64;
        let a = [
            [4.0, 1.0, 0.0, 0.5],
            [1.0, 3.0, -1.0, 0.0],
            [0.0, -1.0, 5.0, 1.0],
            [0.5, 0.0, 1.0, 2.0],
        ];
        let b = [1.0, -2.0, 0.5, 3.0];
        let v = *x.coeffs();
        let mut e = T::zero();
        for i in 0..4 {
            for j in 0..4 {
                e = e + c(0.5 * a[i][j]) * v[i] * v[j];
            }
            e = e - c(b[i]) * v[i];
        }
        e
    }
    let x0 = Plane::<(), f64>::new(0.3, 0.1, -0.7, 2.0);
    let (_, g, h) = hessian(energy, x0);
    let x = x0 - h.solve(g);
    let (_, g_at, _) = hessian(energy, x);
    assert!(g_at.c[0].iter().all(|v| v.abs() < 1e-12), "{g_at:?}");
}

/// numga's `derivative` (JAX, VGA3D), exported by
/// `docs/research/experiments/numga_derivatives.py`: the gradient and Hessian of a scalar
/// energy of a vector, and the Jacobian of `exp` from bivectors to rotors.
#[cfg(feature = "vga3d")]
mod numga {
    use super::*;
    use gax::vga3d::{Bivector, Rotor, Vector};

    fn energy<T: Real>(x: Vector<(), T>) -> T {
        let c = T::from_f64;
        let a = Vector::new(c(0.4), c(-1.1), c(0.7));
        let b = Vector::new(c(0.9), c(0.2), c(-0.5));
        let n = (x - a).norm();
        let xb = x.scalar_product(b).s();
        n * n * n / c(3.0) + xb * xb
    }

    /// The numbers of a fixture line: after `|` if it has one, else after its first word.
    fn numbers(line: &str) -> Vec<f64> {
        let tail = match line.split_once('|') {
            Some((_, t)) => t,
            None => line.split_once(' ').unwrap().1,
        };
        tail.split_whitespace()
            .map(|w| w.parse().unwrap())
            .collect()
    }

    #[test]
    fn against_numga() {
        let text = include_str!("fixtures/numga_derivatives.txt");
        let mut lines = text.lines().filter(|l| !l.starts_with('#'));
        let (mut energies, mut rotors) = (0, 0);
        while let Some(head) = lines.next() {
            if head.starts_with("x ") {
                let x0 = numbers(head);
                let (e, g, h) = (
                    numbers(lines.next().unwrap()),
                    numbers(lines.next().unwrap()),
                    numbers(lines.next().unwrap()),
                );
                let x = Vector::<(), f64>::new(x0[0], x0[1], x0[2]);
                let (v, grad, hess) = hessian(energy, x);
                assert!((v - e[0]).abs() < 1e-14 * e[0].abs().max(1.0));
                for i in 0..3 {
                    assert!((grad.c[0][i] - g[i]).abs() < 1e-13, "gradient {i}");
                    for j in 0..3 {
                        assert!(
                            (hess.c[0][i][j] - h[3 * i + j]).abs() < 1e-13,
                            "hessian {i} {j}"
                        );
                    }
                }
                energies += 1;
            } else if head.starts_with("bivector ") {
                let b0 = numbers(head);
                let (r, jac) = (
                    numbers(lines.next().unwrap()),
                    numbers(lines.next().unwrap()),
                );
                let b = Bivector::<(), f64>::new(b0[0], b0[1], b0[2]);
                let (rotor, d): (Rotor<(), f64>, Rotor<(Bivector,), f64>) =
                    jacobian(|b| b.exp().into_inner(), b);
                for o in 0..4 {
                    assert!((rotor.c[o] - r[o]).abs() < 1e-15, "exp {o}");
                    for i in 0..3 {
                        assert!(
                            (d.c[o][i] - jac[3 * o + i]).abs() < 1e-14,
                            "jacobian {o} {i}"
                        );
                    }
                }
                rotors += 1;
            }
        }
        assert_eq!((energies, rotors), (2, 2));
    }
}

/// The distance from the origin of a plane's first two coefficients, `√(a² + b²)`.
fn radius<T: Real>(p: Plane<(), T>) -> T {
    (p.c[0] * p.c[0] + p.c[1] * p.c[1]).sqrt()
}

/// Second derivatives through `√` where a first derivative is exactly zero: at `(1, 0)` the
/// radius does not change along `b` to first order, but it curves, `∂²r/∂b² = 1/r`. (The chain
/// rule's guard against `0 · ∞` once dropped such a derivative's own derivative.)
#[test]
fn second_derivative_where_the_first_is_zero() {
    let (r, g, h) = hessian(radius, Plane::<(), f64>::new(1.0, 0.0, 0.0, 0.0));
    assert_eq!(r, 1.0);
    assert_eq!(g.c[0], [1.0, 0.0, 0.0, 0.0]);
    assert_eq!(h.c[0][1][1], 1.0);
    assert_eq!(h.c[0][0][0], 0.0);
}
