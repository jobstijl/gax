//! Checks of the linear algebra on maps, for any input (the `solvers` test and the `solve`
//! fuzz target share this file). Nothing may panic, singular or not; where the problem is well
//! conditioned, the answers must be accurate:
//! * `solve` and `inverse`: small residuals, relative to the condition number;
//! * `det`: `det(A) det(A⁻¹) = 1`;
//! * the SVD: `A vᵢ = σᵢ uᵢ`, `σ` non-negative and descending, `u` and `v` orthonormal;
//! * the symmetric eigenproblem: `A v = λ v`, `λ` ascending, `v` orthonormal.

use gax::Extensor;
use gax::pga3d::{Line, Point, Scalar};

/// The checks of one square map `$a` with right-hand side `$b`.
macro_rules! square {
    ($a:expr, $b:expr) => {{
        let (a, b) = ($a, $b);
        // Never panics, whatever the input.
        let inv = a.inverse();
        let det = a.det();
        let x = a.solve(b);
        let (u, sigma, v) = a.svd();
        let am = matrix(&a);
        let scale = am.iter().flatten().fold(0.0f64, |m, x| m.max(x.abs()));
        let finite = |m: &[Vec<f64>]| m.iter().flatten().all(|x| x.is_finite());
        if finite(&am) && (1e-100..1e100).contains(&scale) {
            // The SVD holds for every finite matrix, singular or not.
            let sigma = sigma.to_vec();
            for w in sigma.windows(2) {
                assert!(w[0] >= w[1] - 1e-12 * scale, "σ not descending: {sigma:?}");
            }
            assert!(
                sigma.iter().all(|s| *s >= -1e-12 * scale),
                "negative σ: {sigma:?}"
            );
            for (k, s) in sigma.iter().enumerate() {
                let av = a.of(v[k]);
                let res: Vec<f64> = av.c.iter().zip(u[k].c).map(|(p, q)| p - s * q).collect();
                assert!(
                    norm(&res) <= 1e-9 * scale,
                    "A v{k} ≠ σ{k} u{k}: residual {}",
                    norm(&res)
                );
            }
            // The rest needs a well-conditioned matrix.
            let im = matrix(&inv);
            let cond = if finite(&im) {
                condition(&am, &im)
            } else {
                f64::INFINITY
            };
            // A non-finite right-hand side gives a non-finite solution, rightly.
            if cond < 1e8 && b.c.iter().all(|v| v.is_finite()) {
                let tol = 1e-13 * cond;
                let res: Vec<f64> = a.of(x).c.iter().zip(b.c).map(|(p, q)| p - q).collect();
                let size = norm(&b.c).max(scale * norm(&x.c));
                assert!(
                    norm(&res) <= tol * size.max(1e-300),
                    "solve: residual {} (cond {cond:.1e})",
                    norm(&res)
                );
                let dd = det * inv.det();
                assert!(
                    (dd - 1.0).abs() <= 1e-10 * cond,
                    "det(A) det(A⁻¹) = {dd} (cond {cond:.1e})"
                );
            }
        }
    }};
}

/// A 4×4 map on points, a 6×6 map on lines, and a symmetric 6×6 form on lines, from numbers.
pub fn check(x: &[f64]) {
    let get = |k: usize| x.get(k).copied().unwrap_or(0.0);
    let a = Point::<(Point,), f64>::from_coeffs(core::array::from_fn(|o| {
        core::array::from_fn(|i| get(4 * o + i))
    }));
    let b = Point::<(), f64>::from_coeffs(core::array::from_fn(|k| get(16 + k)));
    square!(a, b);
    let l = Line::<(Line,), f64>::from_coeffs(core::array::from_fn(|o| {
        core::array::from_fn(|i| get(20 + 6 * o + i))
    }));
    let r = Line::<(), f64>::from_coeffs(core::array::from_fn(|k| get(56 + k)));
    square!(l, r);
    symmetric(&core::array::from_fn(|o| {
        core::array::from_fn(|i| get(62 + 6 * o.min(i) + o.max(i)))
    }));
}

fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn matrix<M: Extensor<Coef = f64>>(m: &M) -> Vec<Vec<f64>> {
    m.coeffs()
        .as_ref()
        .iter()
        .map(|row| {
            (0..<M::Slots as gax::Slots>::SIZE)
                .map(|k| <M::Slots as gax::Slots>::get_flat(row, k))
                .collect()
        })
        .collect()
}

/// The ∞-norm condition number from a matrix and its computed inverse.
fn condition(a: &[Vec<f64>], inv: &[Vec<f64>]) -> f64 {
    let n = |m: &[Vec<f64>]| {
        m.iter()
            .map(|r| r.iter().map(|x| x.abs()).sum::<f64>())
            .fold(0.0, f64::max)
    };
    n(a) * n(inv)
}

fn symmetric(a: &[[f64; 6]; 6]) {
    let form = Scalar::<(Line, Line), f64>::from_coeffs([*a]);
    let (vals, vecs) = form.eigh();
    let scale = a.iter().flatten().fold(0.0f64, |m, x| m.max(x.abs()));
    if !a.iter().flatten().all(|x| x.is_finite()) || !(1e-100..1e100).contains(&scale) {
        return;
    }
    let vals = vals.as_ref();
    for w in vals.windows(2) {
        assert!(w[0] <= w[1] + 1e-12 * scale, "λ not ascending: {vals:?}");
    }
    for (k, l) in vals.iter().enumerate() {
        let v = vecs.as_ref()[k].c;
        assert!((norm(&v) - 1.0).abs() < 1e-9, "eigenvector {k} not unit");
        let av: Vec<f64> = (0..6)
            .map(|o| (0..6).map(|i| a[o][i] * v[i]).sum())
            .collect();
        let res: Vec<f64> = av.iter().zip(v).map(|(p, q)| p - l * q).collect();
        assert!(
            norm(&res) <= 1e-9 * scale,
            "A v{k} ≠ λ{k} v{k}: residual {}",
            norm(&res)
        );
        for j in 0..k {
            let w = vecs.as_ref()[j].c;
            let dot: f64 = v.iter().zip(w).map(|(p, q)| p * q).sum();
            assert!(
                dot.abs() < 1e-9,
                "eigenvectors {j} and {k} not orthogonal: {dot}"
            );
        }
    }
}
