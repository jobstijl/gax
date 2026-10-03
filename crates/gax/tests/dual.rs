//! Forward-mode derivatives (`gax::dual`): through products, sandwiches, `exp`, `log`,
//! normalization and the solvers, against analytic derivatives and central differences, and
//! exactly over the prime field for polynomial kernels.

#![cfg(feature = "pga3d")]

#[path = "support/rng.rs"]
mod rng;
use rng::Rng;

use gax::dual::{Dual, derivative, gradient, jacobian};
use gax::fp::Fp;
use gax::pga3d::{Line, Motor, Plane, Point};
use gax::{Coef, Real};

type D1 = Dual<f64, 1>;

/// The central difference of `f` at `x`.
fn central(f: impl Fn(f64) -> f64, x: f64) -> f64 {
    let h = 1e-6;
    (f(x + h) - f(x - h)) / (2.0 * h)
}

fn line<T: Real>(c: [T; 6]) -> Line<(), T> {
    Line::from_coeffs(c)
}

/// `d/dt exp(t B)` is `B exp(t B)` (they commute), for random lines `B`: rotations, screws and
/// translations, through gax's closed forms.
#[test]
fn exp_along_a_line() {
    let mut rng = Rng(0xd0a1);
    for k in 0..200 {
        let mut b: [f64; 6] = core::array::from_fn(|_| rng.next_f64());
        if k % 3 == 1 {
            b[..3].fill(0.0); // a translation
        }
        let t0 = rng.next_f64();
        for i in 0..8 {
            let (_, d) = derivative(
                |t: D1| line(b.map(D1::constant)).gp(t).exp().into_inner().c[i],
                t0,
            );
            let m = line(b).gp(t0).exp().into_inner();
            let want = (line(b) * m).c[i];
            assert!(
                (d - want).abs() < 1e-12,
                "{k}, coefficient {i}: {d} vs {want}"
            );
        }
    }
}

/// `log(exp(t B))` has derivative `B` (inside the principal branch), and the derivative of a
/// motor's log agrees with central differences.
#[test]
fn log_of_exp() {
    let mut rng = Rng(0x10_9e);
    for _ in 0..200 {
        let b: [f64; 6] = core::array::from_fn(|_| 0.5 * rng.next_f64());
        let t0 = 0.3 + 0.5 * rng.next_f64().abs();
        for i in 0..6 {
            let (_, d) = derivative(
                |t: D1| {
                    let l: Line<(), D1> = line(b.map(D1::constant)).gp(t).exp().log();
                    l.c[i]
                },
                t0,
            );
            assert!((d - b[i]).abs() < 1e-10, "coefficient {i}: {d} vs {}", b[i]);
        }
    }
}

/// The gradient of a cost through a sandwich, `to_euclidean` and a product of motors, against
/// central differences in each variable.
#[test]
fn gradient_through_motions() {
    let cost = |x: [f64; 3]| {
        let m = Motor::translation(x[0], x[1], 0.0) * Motor::rotation_about(0.0, 0.0, 1.0, x[2]);
        let [a, b, c] = (m >> Point::xyz(1.0, 2.0, 0.5)).to_euclidean();
        (a - 0.3).powi(2) + (b + 1.0).powi(2) + c * c
    };
    let mut rng = Rng(0x9ad);
    for _ in 0..100 {
        let x0: [f64; 3] = core::array::from_fn(|_| rng.next_f64());
        let (v, g) = gradient(
            |[x, y, a]: [Dual<f64, 3>; 3]| {
                let c = Dual::constant;
                let m = Motor::translation(x, y, c(0.0))
                    * Motor::rotation_about(c(0.0), c(0.0), c(1.0), a);
                let [p, q, r] = (m >> Point::xyz(c(1.0), c(2.0), c(0.5))).to_euclidean();
                let (dp, dq) = (p - c(0.3), q + c(1.0));
                dp * dp + dq * dq + r * r
            },
            x0,
        );
        assert!((v - cost(x0)).abs() < 1e-12);
        for i in 0..3 {
            let fd = central(
                |h| {
                    let mut x = x0;
                    x[i] = h;
                    cost(x)
                },
                x0[i],
            );
            assert!((g[i] - fd).abs() < 1e-6, "∂{i}: {} vs {fd}", g[i]);
        }
    }
}

/// The Jacobian of the plane through three moving points (`p & q & r`, normalized) against
/// central differences: the join, normalization and a square root on dual numbers.
#[test]
#[allow(clippy::needless_range_loop)] // Jacobian entries by row and column
fn jacobian_of_a_join() {
    let plane = |x: [f64; 3]| {
        let p =
            Point::xyz(x[0], 0.0, 0.0) & Point::xyz(0.0, x[1], 0.0) & Point::xyz(0.0, 0.0, x[2]);
        let n = (p.e1() * p.e1() + p.e2() * p.e2() + p.e3() * p.e3()).sqrt();
        [p.e1() / n, p.e2() / n, p.e3() / n, p.e0() / n]
    };
    let x0 = [1.0, 2.0, 0.5];
    let (v, j) = jacobian(
        |x: [Dual<f64, 3>; 3]| {
            let c = Dual::constant;
            let p: Plane<(), Dual<f64, 3>> = Point::xyz(x[0], c(0.0), c(0.0))
                & Point::xyz(c(0.0), x[1], c(0.0))
                & Point::xyz(c(0.0), c(0.0), x[2]);
            let n = (p.e1() * p.e1() + p.e2() * p.e2() + p.e3() * p.e3()).sqrt();
            [p.e1() / n, p.e2() / n, p.e3() / n, p.e0() / n]
        },
        x0,
    );
    for (got, want) in v.iter().zip(plane(x0)) {
        assert!((got - want).abs() < 1e-12);
    }
    for m in 0..4 {
        for i in 0..3 {
            let fd = central(
                |h| {
                    let mut x = x0;
                    x[i] = h;
                    plane(x)[m]
                },
                x0[i],
            );
            assert!(
                (j[m][i] - fd).abs() < 1e-7,
                "J[{m}][{i}]: {} vs {fd}",
                j[m][i]
            );
        }
    }
}

/// Derivatives through the linear algebra on maps: `d/dt solve(A + t E, b)` is
/// `−A⁻¹ E A⁻¹ b` at `t = 0`.
#[test]
fn derivative_of_a_solve() {
    let mut rng = Rng(0x501e);
    for _ in 0..50 {
        let a: [[f64; 4]; 4] = core::array::from_fn(|_| core::array::from_fn(|_| rng.next_f64()));
        let e: [[f64; 4]; 4] = core::array::from_fn(|_| core::array::from_fn(|_| rng.next_f64()));
        let b: [f64; 4] = core::array::from_fn(|_| rng.next_f64());
        let am = Point::<(Point,), f64>::from_coeffs(a);
        let em = Point::<(Point,), f64>::from_coeffs(e);
        let x: Point<(), f64> = am.solve(Point::<(), f64>::from_coeffs(b));
        let want: Point<(), f64> = am.solve(em.of(x)).gp(-1.0);
        for i in 0..4 {
            let (_, d) = derivative(
                |t: D1| {
                    let m = Point::<(Point,), D1>::from_coeffs(core::array::from_fn(|o| {
                        core::array::from_fn(|k| D1::constant(a[o][k]) + t * D1::constant(e[o][k]))
                    }));
                    m.solve(Point::<(), D1>::from_coeffs(b.map(D1::constant))).c[i]
                },
                0.0,
            );
            let scale = want.c.iter().fold(1.0f64, |m, v: &f64| m.max(v.abs()));
            assert!((d - want.c[i]).abs() < 1e-9 * scale, "{d} vs {}", want.c[i]);
        }
    }
}

/// Exact derivatives of a polynomial kernel over the prime field: the derivative of
/// `t ↦ (a + t a') (b + t b')` is `a' b + a b'`, coefficient by coefficient, for motors.
#[test]
fn product_rule_exactly() {
    type F = Dual<Fp, 1>;
    let mut s = 0x1234_5678u64;
    let mut next = || {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        Fp::new(s >> 3)
    };
    for _ in 0..20 {
        let (a, da, b, db): ([Fp; 8], [Fp; 8], [Fp; 8], [Fp; 8]) = (
            core::array::from_fn(|_| next()),
            core::array::from_fn(|_| next()),
            core::array::from_fn(|_| next()),
            core::array::from_fn(|_| next()),
        );
        let x = Motor::<(), F>::from_coeffs(core::array::from_fn(|i| Dual::new(a[i], [da[i]])));
        let y = Motor::<(), F>::from_coeffs(core::array::from_fn(|i| Dual::new(b[i], [db[i]])));
        let got = x * y;
        let (ma, mda) = (
            Motor::<(), Fp>::from_coeffs(a),
            Motor::<(), Fp>::from_coeffs(da),
        );
        let (mb, mdb) = (
            Motor::<(), Fp>::from_coeffs(b),
            Motor::<(), Fp>::from_coeffs(db),
        );
        let want = mda * mb + ma * mdb;
        for i in 0..8 {
            assert_eq!(got.c[i].re, (ma * mb).c[i]);
            assert_eq!(got.c[i].du[0], want.c[i]);
        }
    }
    let _ = Fp::zero();
}

/// Through CSTA's closed-form `exp` (log6d.md §12: interpolants, series switching, turning
/// rotations past a quarter turn, halving): `d/dt exp(t B) = B exp(t B)`, for small bivectors,
/// large rotations and large boosts.
#[cfg(feature = "csta")]
#[test]
fn csta_exp_along_a_bivector() {
    use gax::csta::{Bivector, Even};
    let mut rng = Rng(0xc57a);
    for k in 0..60 {
        let size = [0.3, 1.0, 2.5][k % 3];
        let b: [f64; 15] = core::array::from_fn(|_| size * rng.next_f64());
        let t0 = 0.5 + 0.5 * rng.next_f64().abs();
        let bv = Bivector::<(), f64>::from_coeffs(b);
        let m = bv.gp(t0).exp().into_inner();
        let want: Even<(), f64> = bv * m;
        let scale = want.c.iter().fold(1.0f64, |s, v| s.max(v.abs()));
        let (_, d) = gradient(
            |[t]: [Dual<f64, 1>; 1]| {
                let e = Bivector::<(), Dual<f64, 1>>::from_coeffs(b.map(Dual::constant))
                    .gp(t)
                    .exp();
                // One fixed linear functional of all 32 coefficients checks every derivative
                // at once (an error would have to cancel against the weights).
                let probe: [f64; 32] = core::array::from_fn(|i| ((i * 7 + 3) % 11) as f64 - 5.0);
                e.into_inner()
                    .c
                    .iter()
                    .zip(probe)
                    .fold(Dual::constant(0.0), |s, (c, p)| s + *c * Dual::constant(p))
            },
            [t0],
        );
        let probe: [f64; 32] = core::array::from_fn(|i| ((i * 7 + 3) % 11) as f64 - 5.0);
        let want_d: f64 = want.c.iter().zip(probe).map(|(c, p)| c * p).sum();
        assert!(
            (d[0] - want_d).abs() < 1e-8 * scale * 50.0,
            "size {size}: {} vs {want_d}",
            d[0]
        );
    }
}
