//! The linear algebra on maps never panics and is accurate where the problem is well
//! conditioned: random matrices, singular ones, and badly scaled ones (`support/solver_checks.rs`;
//! the `solve` fuzz target runs the same checks on arbitrary bytes).

#![cfg(feature = "pga3d")]

#[path = "support/rng.rs"]
mod rng;
use rand::RngCore;
use rng::{Draw, rng};

#[path = "support/solver_checks.rs"]
mod solver_checks;

#[test]
fn random_singular_and_badly_scaled_matrices() {
    let mut rng = rng(0xdead_beef_1234);
    for k in 0..3000 {
        let mut x: Vec<f64> = (0..83).map(|_| rng.next_f64()).collect();
        match k % 5 {
            // Rank-deficient: repeat rows.
            1 => {
                for i in 0..4 {
                    x[4 + i] = x[i];
                    x[26 + i] = x[20 + i];
                }
            }
            // Rows of wildly different scales.
            2 => {
                for (i, v) in x.iter_mut().enumerate().take(56) {
                    *v *= 10f64.powi((i as i32 % 7) * 4 - 12);
                }
            }
            // Mostly zeros.
            3 => {
                for v in &mut x {
                    if !rng.next_u64().is_multiple_of(3) {
                        *v = 0.0;
                    }
                }
            }
            // Non-finite entries: nothing may panic.
            4 => {
                x[(rng.next_u64() % 83) as usize] =
                    [f64::NAN, f64::INFINITY, -f64::INFINITY][k % 3];
            }
            _ => {}
        }
        solver_checks::check(&x);
    }
}

/// Found by the `solve` fuzz target: singular values from `10⁶` down to `3·10⁻²`. The Jacobi
/// least squares stopped on a summed convergence test while the smallest pair of columns was
/// not yet orthogonal, and `A⁺ A` missed the identity by `10⁻³` (now by rounding: `10⁻⁹`).
#[test]
fn fuzz_regression_widely_spread_singular_values() {
    let data: &[u8] = &[
        0x44, 0x80, 0xff, 0x29, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xef,
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xfd, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x33,
        0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0x33,
        0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0xff, 0xc6,
        0xfe, 0xff, 0xff, 0xff, 0x00, 0x00, 0xfa, 0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        0x0d, 0xff, 0xff, 0xff, 0x44, 0x80, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        0xff, 0xff, 0xef, 0xff, 0xff, 0xff, 0xff, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0xff, 0xff, 0xff, 0xfd, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xc6, 0xff, 0xff, 0xff,
        0xff, 0x00, 0x00, 0xfa, 0x7e, 0x00, 0x00, 0xbd, 0x3d, 0x2b, 0x0a, 0x96, 0xff, 0xfe,
    ];
    solver_checks::check(&numbers(data));
}

/// Found by the `solve` fuzz target: a rank-3 map on lines whose first two rows hold four
/// columns, two short and orthogonal, `(-78125, 0)` and `(0, -78125)`, each parallel to a long one
/// (`4687500`). Of equal length, the short pair was turned by an eighth of a turn every sweep
/// although orthogonal, which undid the progress against the long ones: two singular values that
/// should be zero halved once per sweep and ended near 100, and `A P A` missed `A`. Pairs that
/// are already orthogonal are no longer rotated.
#[test]
fn fuzz_regression_orthogonal_pairs_of_equal_length() {
    let data: &[u8] = &[
        0x0a, 0x6a, 0x68, 0x4f, 0xb4, 0x4f, 0x4b, 0x4f, 0x4f, 0x75, 0x41, 0x75, 0x0a, 0xff, 0xff,
        0xff, 0xff, 0xff, 0x2e, 0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x01, 0x00,
        0xff, 0xff, 0xff, 0x2e, 0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x01, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x3c, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3c, 0xff,
        0xff, 0xff, 0x00, 0x00, 0x00, 0x06, 0x6e, 0x6a, 0xff, 0x6a,
    ];
    solver_checks::check(&numbers(data));
    // The singular values are those of numpy: two of 4688151, one of 85.94, three zeros.
    let x = numbers(data);
    let l = gax::pga3d::Line::<(gax::pga3d::Line,), f64>::from_coeffs(core::array::from_fn(|o| {
        core::array::from_fn(|i| x.get(20 + 6 * o + i).copied().unwrap_or(0.0))
    }));
    let sigma = l.svdvals();
    let big = 4_688_150.996_461_718;
    let want = [big, big, 85.941_051_062_996_14, 0.0, 0.0, 0.0];
    for (s, w) in sigma.iter().zip(want) {
        assert!((s - w).abs() <= 1e-9 * big, "{sigma:?}");
    }
}

/// The fuzz target's numbers: from two bytes each, a mantissa in `[-1, 1)` and a decimal
/// exponent in `[-8, 8)`.
fn numbers(data: &[u8]) -> Vec<f64> {
    data.as_chunks::<2>()
        .0
        .iter()
        .take(83)
        .map(|c| f64::from(c[0] as i8) / 128.0 * 10f64.powi(i32::from(c[1] % 16) - 8))
        .collect()
}
