//! Least squares and the pseudo-inverse on maps (`lstsq`, `pinv`): the same answers as the
//! inverse where that exists, right-hand sides with slots, several slots, a smaller kind as the
//! right-hand side, and `f32`. The Penrose conditions on random, singular and badly scaled maps
//! are in `support/solver_checks.rs` (also fuzzed); per-lane agreement in gax-core's
//! `solver_lanes`.

#![cfg(feature = "pga3d")]

#[path = "support/rng.rs"]
mod rng;
use rng::Rng;

use gax::ApproxEq;
use gax::pga3d::{Line, Motor, Plane, Point, Rotor, Scalar};

/// On a square map of full rank the pseudo-inverse is the inverse, and least squares solves.
#[test]
fn square_full_rank_is_the_inverse() {
    let mut rng = Rng(0x0015_0eab);
    for _ in 0..200 {
        let a = rng.value::<Line<(Line,), f64>>(1.0);
        let b = Line::<(), f64>::from_coeffs(core::array::from_fn(|_| rng.next_f64()));
        assert!(a.pinv().approx_eq(&a.inverse(), 1e-9));
        assert!(a.lstsq(b).approx_eq(&a.solve(b), 1e-9));
    }
}

/// A tall map of full column rank (points to lines) is undone by least squares, for values and
/// for maps on the right (their slots kept): `a.lstsq(a.of(y)) == y`.
#[test]
fn tall_full_rank_recovers_values_and_maps() {
    let mut rng = Rng(0x7a11);
    for _ in 0..200 {
        let a = rng.value::<Line<(Point,), f64>>(1.0);
        let x = Point::<(), f64>::from_coeffs(core::array::from_fn(|_| rng.next_f64()));
        assert!(a.lstsq(a.of(x)).approx_eq(&x, 1e-9));
        let y = rng.value::<Point<(Plane,), f64>>(1.0);
        let back: Point<(Plane,), f64> = a.lstsq(a.of(y));
        assert!(back.approx_eq(&y, 1e-9));
    }
}

/// The least-squares residual is orthogonal to the map's range (the normal equations), and of
/// all the solutions of a rank-deficient map, the pseudo-inverse's has the least norm.
#[test]
#[allow(clippy::needless_range_loop)] // columns of a row-major matrix
fn normal_equations_and_least_norm() {
    let mut rng = Rng(0xbead);
    for _ in 0..200 {
        // A wide map (lines to points) of rank 3: its last column a combination of the others.
        let mut c: [[f64; 6]; 4] =
            core::array::from_fn(|_| core::array::from_fn(|_| rng.next_f64()));
        for row in &mut c {
            row[5] = 0.5 * row[0] - 2.0 * row[3];
        }
        let a = Point::<(Line,), f64>::from_coeffs(c);
        let b = Point::<(), f64>::from_coeffs(core::array::from_fn(|_| rng.next_f64()));
        let x = a.lstsq(b);
        let r = a.of(x) - b;
        for j in 0..6 {
            let col: f64 = (0..4).map(|i| c[i][j] * r.c[i]).sum();
            assert!(col.abs() < 1e-9, "Aᵀ r ≠ 0: {col:e}");
        }
        // Adding a kernel vector changes nothing in the image but adds norm.
        let mut k = [0.0; 6];
        (k[0], k[3], k[5]) = (0.5, -2.0, -1.0);
        let other = x + Line::from_coeffs(k).gp(0.3);
        assert!(a.of(other).approx_eq(&a.of(x), 1e-9));
        let norm = |l: Line<(), f64>| l.c.iter().map(|v| v * v).sum::<f64>();
        assert!(norm(x) < norm(other));
    }
}

/// Several slots: least squares for the first, against a right-hand side on the others. The
/// plane `x` with `x & p == t & p` for all points `p` is `t`, and with a third slot carried
/// along, the same.
#[test]
fn several_slots_solve_for_the_first() {
    let mut rng = Rng(0x5107);
    for _ in 0..100 {
        let t = Plane::<(), f64>::from_coeffs(core::array::from_fn(|_| rng.next_f64()));
        let l = Plane::<(), f64>::from_coeffs(core::array::from_fn(|_| rng.next_f64()));
        let pairing: Scalar<(Plane, Point), f64> = Plane::slot() & Point::slot();
        let x: Plane<(), f64> = pairing.lstsq(t & Point::slot());
        assert!(x.approx_eq(&t, 1e-9));
        let three: Scalar<(Plane, Point, Point), f64> =
            (Plane::slot() & Point::slot()) * (l & Point::slot());
        let rhs: Scalar<(Point, Point), f64> = (t & Point::slot()) * (l & Point::slot());
        let x: Plane<(), f64> = three.lstsq(rhs);
        assert!(x.approx_eq(&t, 1e-9));
        // Another slot first with `at`: the point `p` with `t' & p == r` for every plane `t'`.
        let p = Point::<(), f64>::from_coeffs(core::array::from_fn(|_| rng.next_f64()));
        let q: Point<(), f64> = pairing.at::<1>().lstsq(Plane::slot() & p);
        assert!(q.approx_eq(&p, 1e-9));
    }
}

/// A right-hand side of a smaller kind is its embedding: a rotor against a map into motors.
#[test]
fn smaller_kind_on_the_right() {
    let mut rng = Rng(0xe4b);
    let a = rng.value::<Motor<(Rotor,), f64>>(1.0);
    let r = Rotor::<(), f64>::from_coeffs(core::array::from_fn(|_| rng.next_f64()));
    assert_eq!(a.lstsq(r), a.lstsq(r.cast::<Motor>()));
}

/// The zero map has the zero pseudo-inverse, and least squares returns zero, not NaN.
#[test]
fn zero_map() {
    let a = Line::<(Point,), f64>::zero();
    assert_eq!(a.pinv(), Point::<(Line,), f64>::zero());
    assert_eq!(
        a.lstsq(Line::from_coeffs([1.0; 6])),
        Point::<(), f64>::zero()
    );
}

/// `f32`, with its own default cutoff: the pseudo-inverse of a well-conditioned tall map.
#[test]
fn single_precision() {
    let mut rng = Rng(0xf32);
    for _ in 0..100 {
        let c: [[f64; 4]; 6] = core::array::from_fn(|_| core::array::from_fn(|_| rng.next_f64()));
        let a = Line::<(Point,), f64>::from_coeffs(c);
        let a32 = Line::<(Point,), f32>::from_coeffs(c.map(|r| r.map(|v| v as f32)));
        let (p, p32) = (a.pinv(), a32.pinv());
        let scale = p.c.iter().flatten().fold(1.0f64, |m, v| m.max(v.abs()));
        let worst =
            p.c.iter()
                .flatten()
                .zip(p32.c.iter().flatten())
                .fold(0.0f64, |m, (x, y)| m.max((x - f64::from(*y)).abs()));
        // Well conditioned in most draws; the error scales with the condition.
        let cond = scale * c.iter().flatten().fold(0.0f64, |m, v| m.max(v.abs()));
        assert!(worst <= 1e-5 * cond * scale, "{worst:e} (cond {cond:.1e})");
    }
}
