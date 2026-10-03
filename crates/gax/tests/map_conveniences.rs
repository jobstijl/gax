//! Conveniences on maps from numga's ports: singular values of maps of any shape (`svdvals`,
//! `svd_thin`) against the eigenvalues of `AᵀA`, `A v = σ u` for every pair, null vectors where
//! the value is zero; a form's `as_map`; and `of_both`, both slots filled with maps at once.

#![cfg(all(feature = "pga3d", feature = "sta"))]

#[path = "support/rng.rs"]
mod rng;
use rng::Rng;

use gax::pga3d::{Line, Plane, Point, Scalar};

/// `A v = σ u` with unit `u`, `v`, for a map `K <- (H,)` given as rows.
fn check_pairs<const K: usize, const H: usize>(
    rows: [[f64; H]; K],
    sigma: &[f64],
    right: &[[f64; H]],
    left: &[[f64; K]],
) {
    for ((s, v), u) in sigma.iter().zip(right).zip(left) {
        let av: Vec<f64> = rows
            .iter()
            .map(|r| r.iter().zip(v).map(|(a, b)| a * b).sum())
            .collect();
        for (x, y) in av.iter().zip(u) {
            assert!((x - s * y).abs() < 1e-11, "{s}: {av:?} vs {u:?}");
        }
        let vn: f64 = v.iter().map(|x| x * x).sum();
        assert!((vn - 1.0).abs() < 1e-12);
    }
    assert!(sigma.windows(2).all(|w| w[0] >= w[1]), "{sigma:?}");
}

#[test]
fn tall_and_wide_maps() {
    let mut rng = Rng(0x0057_d5a1);
    for _ in 0..50 {
        // Tall: points (4) to lines (6).
        let tall = Line::<(Point,), f64>::from_coeffs(core::array::from_fn(|_| {
            core::array::from_fn(|_| rng.next_f64())
        }));
        let (s, right, left) = tall.svd_thin();
        check_pairs(tall.c, &s, &right.map(|v| v.c), &left.map(|u| u.c));
        // The squares are the eigenvalues of AᵀA (a symmetric form on points).
        let ata = tall.c.iter().fold([[0.0; 4]; 4], |mut m, r| {
            for (row, ri) in m.iter_mut().zip(r) {
                for (x, rj) in row.iter_mut().zip(r) {
                    *x += ri * rj;
                }
            }
            m
        });
        let form = Scalar::<(Point, Point), f64>::from_coeffs([ata]);
        let (eig, _) = form.eigh();
        for (sv, e) in s.iter().rev().zip(eig) {
            assert!((sv * sv - e).abs() < 1e-10 * (1.0 + e.abs()));
        }
        // Wide: lines (6) to points (4): two singular values are zero, their vectors null.
        let wide = Point::<(Line,), f64>::from_coeffs(core::array::from_fn(|_| {
            core::array::from_fn(|_| rng.next_f64())
        }));
        let (s, right, left) = wide.svd_thin();
        check_pairs(wide.c, &s, &right.map(|v| v.c), &left.map(|u| u.c));
        assert!(s[4].abs() < 1e-12 && s[5].abs() < 1e-12 && s[3] > 1e-8);
        for v in &right[4..] {
            assert!(wide.of(*v).c.iter().all(|x| x.abs() < 1e-12));
        }
    }
}

/// An 8 x 6 mixed-grade map (the vacuum part of constitutive's wave map, `Odd <- Bivector` in
/// STA): for a null `k`, `k ^ F` vanishes exactly on the bivectors `k ^ a`, a three-dimensional
/// space (four vectors `a`, less `k` itself).
#[test]
fn a_mixed_grade_map_s_null_space() {
    use gax::sta::{Bivector, Odd, Vector};
    let k = Vector::<(), f64>::new(1.0, 0.0, 0.0, 1.0);
    let wave: Odd<(Bivector,), f64> = (k ^ Bivector::<(), f64>::slot()).cast::<Odd>();
    let s = wave.svdvals();
    assert!(s[2] > 1e-8, "{s:?}");
    assert!(s[3..].iter().all(|x| x.abs() < 1e-12), "{s:?}");
    let (_, right, _) = wave.svd_thin();
    for v in &right[3..] {
        // Each null bivector is k ^ a: it wedges to zero with k.
        assert!((k ^ *v).c.iter().all(|x| x.abs() < 1e-12));
    }
}

#[test]
fn a_form_as_a_map() {
    // The incidence of planes and points: rank 4 as a map from points to planes' duals.
    let form: Scalar<(Plane, Point), f64> = Plane::slot() & Point::slot();
    let map = form.as_map();
    let p = Point::<(), f64>::xyz(1.0, 2.0, 3.0);
    let pl = Plane::<(), f64>::new(0.5, -1.0, 2.0, 0.25);
    assert!(
        (map.of(p)
            .c
            .iter()
            .zip(pl.c)
            .map(|(a, b)| a * b)
            .sum::<f64>()
            - form.of(pl).of(p).s())
        .abs()
            < 1e-12
    );
    assert!(map.svdvals().iter().all(|x| (x - 1.0).abs() < 1e-12));
}

/// `of_both(p, q)` is `of(p).at::<1>().of(q).swap()`: a form pulled back through two different
/// maps, slots in order.
#[test]
fn of_both_is_the_chain() {
    use gax::pga3d::Motor;
    let mut rng = Rng(0x00b0_7400);
    let form = Scalar::<(Line, Line), f64>::from_coeffs([core::array::from_fn(|_| {
        core::array::from_fn(|_| rng.next_f64())
    })]);
    let p = Line::<(Point,), f64>::from_coeffs(core::array::from_fn(|_| {
        core::array::from_fn(|_| rng.next_f64())
    }));
    let q = Line::<(Motor,), f64>::from_coeffs(core::array::from_fn(|_| {
        core::array::from_fn(|_| rng.next_f64())
    }));
    let both: Scalar<(Point, Motor), f64> = form.of_both(p, q);
    let chain: Scalar<(Point, Motor), f64> = form.of(p).at::<1>().of(q).swap();
    for (a, b) in both.c[0].iter().flatten().zip(chain.c[0].iter().flatten()) {
        assert!((a - b).abs() < 1e-12);
    }
    let (x, m) = (
        Point::<(), f64>::xyz(1.0, -2.0, 0.5),
        Motor::<(), f64>::translation(0.3, 0.1, 2.0).into_inner(),
    );
    assert!((both.of(x).of(m).s() - form.of(p.of(x)).of(q.of(m)).s()).abs() < 1e-12);
}

/// Fitting a point to noisy points as numga does: the smallest eigenvector of the misfit
/// against the point metric, which measures only the weight (semidefinite). The finite mode is
/// the centroid; the three directions of the ideal points are infinite.
#[test]
fn a_semidefinite_metric_s_finite_modes() {
    let mut rng = Rng(0x0f17_7e45);
    let pts: Vec<Point<(), f64>> = (0..40)
        .map(|_| {
            Point::xyz(
                1.0 + 0.1 * rng.next_f64(),
                -2.0 + 0.1 * rng.next_f64(),
                0.5 + 0.1 * rng.next_f64(),
            )
        })
        .collect();
    // misfit(X, X) = Σ |p ∨ X|², the PGA norm of the lines from the samples to X (their
    // Euclidean part e23, e31, e12: for unit weights, the squared distance); the metric: X's
    // squared weight.
    let mut misfit = Scalar::<(Point, Point), f64>::zero();
    for p in &pts {
        let l = *p & Point::slot();
        for k in 0..3 {
            // The k-th coefficient of the join as a form on X, squared.
            let row = Point::<(), f64>::from_coeffs(core::array::from_fn(|i| l.c[k][i]));
            misfit += Scalar::from_coeffs([core::array::from_fn(|i| {
                core::array::from_fn(|j| row.c[i] * row.c[j])
            })]);
        }
    }
    let w = Point::<(), f64>::new(0.0, 0.0, 0.0, 1.0);
    let metric = Scalar::<(Point, Point), f64>::from_coeffs([core::array::from_fn(|i| {
        core::array::from_fn(|j| w.c[i] * w.c[j])
    })]);
    let (vals, vecs) = misfit.eigh_semidefinite(metric);
    assert!(
        vals[0].is_finite() && vals[1..].iter().all(|v| v.is_infinite()),
        "{vals:?}"
    );
    let fit = vecs[0];
    // Normalized to unit weight (metric(x, x) = 1), up to sign: the centroid.
    let n = pts.len() as f64;
    let centroid = pts.iter().fold([0.0; 3], |a, p| {
        let e = p.to_euclidean();
        [a[0] + e[0] / n, a[1] + e[1] / n, a[2] + e[2] / n]
    });
    let got = fit.to_euclidean();
    assert!((fit.c[3].abs() - 1.0).abs() < 1e-10);
    for (a, b) in got.iter().zip(centroid) {
        assert!((a - b).abs() < 1e-9, "{got:?} vs {centroid:?}");
    }
}
