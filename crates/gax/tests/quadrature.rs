//! Gauss–Legendre rules against the exact ones and numpy's `leggauss`
//! (`docs/research/experiments/numpy_leggauss.py`), and against exact integrals of polynomials.

use gax::quadrature::{gauss_legendre, gauss_legendre_on};

/// A rule of the fixture: `n`, numpy's nodes and weights, and the exact ones (to 50 digits,
/// rounded).
struct Fixture {
    n: usize,
    numpy: (Vec<f64>, Vec<f64>),
    exact: (Vec<f64>, Vec<f64>),
}

fn fixtures() -> Vec<Fixture> {
    let text = include_str!("fixtures/numpy_leggauss.txt");
    let mut lines = text.lines();
    let mut out = Vec::new();
    while let Some(head) = lines.next() {
        let n = head
            .strip_prefix("n ")
            .expect("an `n` line")
            .parse()
            .unwrap();
        let mut numbers = || -> Vec<f64> {
            lines
                .next()
                .unwrap()
                .split_whitespace()
                .map(|w| w.parse().unwrap())
                .collect()
        };
        let numpy = (numbers(), numbers());
        let exact = (numbers(), numbers());
        out.push(Fixture { n, numpy, exact });
    }
    out
}

/// Within a few rounding units of the exact rule, as numpy's is (whose weights are off by up to
/// `18 ε` at `n = 48`).
#[test]
fn the_rules_are_exact_to_rounding_and_numpys() {
    for f in fixtures() {
        let (nodes, weights) = gauss_legendre::<f64>(f.n);
        assert_eq!(nodes.len(), f.n);
        let close = |a: &[f64], b: &[f64], units: f64, what: &str| {
            for (x, y) in a.iter().zip(b) {
                assert!(
                    (x - y).abs() <= units * f64::EPSILON,
                    "n {}: {what} {x} against {y}",
                    f.n
                );
            }
        };
        close(&nodes, &f.exact.0, 2.0, "node");
        close(&weights, &f.exact.1, 8.0, "weight");
        close(&nodes, &f.numpy.0, 4.0, "numpy's node");
        close(&weights, &f.numpy.1, 32.0, "numpy's weight");
    }
}

#[test]
fn the_rules_are_symmetric_and_sum_to_the_length() {
    for n in 1..60 {
        let (x, w) = gauss_legendre::<f64>(n);
        assert!(x.windows(2).all(|p| p[0] < p[1]), "n {n}: ascending");
        for k in 0..n {
            assert_eq!(x[k], -x[n - 1 - k]);
            assert_eq!(w[k], w[n - 1 - k]);
        }
        let total: f64 = w.iter().sum();
        assert!((total - 2.0).abs() < 32.0 * f64::EPSILON, "n {n}: {total}");
    }
}

/// `n` nodes integrate every monomial of degree up to `2n − 1` exactly (and, for few nodes,
/// visibly not `x^(2n)`).
#[test]
fn n_nodes_are_exact_to_degree_two_n_less_one() {
    for n in [1, 2, 3, 7, 12, 25] {
        let (x, w) = gauss_legendre::<f64>(n);
        let integral = |d: i32| -> f64 { x.iter().zip(&w).map(|(x, w)| w * x.powi(d)).sum() };
        for d in 0..2 * n as i32 {
            let exact = if d % 2 == 0 {
                2.0 / f64::from(d + 1)
            } else {
                0.0
            };
            let error = (integral(d) - exact).abs();
            assert!(error < 1e-13, "n {n}, degree {d}: {error}");
        }
        if n <= 3 {
            let d = 2 * n as i32;
            assert!((integral(d) - 2.0 / f64::from(d + 1)).abs() > 1e-3, "n {n}");
        }
    }
}

#[test]
fn a_rule_moves_to_any_interval_and_runs_in_f32() {
    let (t, w) = gauss_legendre_on::<f64>(4, 1.0, 3.0);
    // ∫₁³ t⁷ dt = (3⁸ − 1) / 8 = 820.
    let integral: f64 = t.iter().zip(&w).map(|(t, w)| w * t.powi(7)).sum();
    assert!((integral - 820.0).abs() < 1e-11);
    let (x, w) = gauss_legendre::<f32>(6);
    let (x64, w64) = gauss_legendre::<f64>(6);
    for k in 0..6 {
        assert!((f64::from(x[k]) - x64[k]).abs() < 1e-6);
        assert!((f64::from(w[k]) - w64[k]).abs() < 1e-6);
    }
}
