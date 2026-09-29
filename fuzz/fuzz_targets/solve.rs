//! The linear algebra on maps (inverse, determinant, solve, SVD, symmetric eigenproblem) on
//! arbitrary matrices: never a panic, and accurate where the problem is well conditioned
//! (`crates/gax/tests/solvers.rs` runs the same checks on a fixed sample).
#![no_main]

#[path = "../../crates/gax/tests/support/solver_checks.rs"]
mod solver_checks;

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Each number from two bytes: a mantissa in [-1, 1) and a decimal exponent in [-8, 8).
    let x: Vec<f64> = data
        .chunks_exact(2)
        .take(83)
        .map(|c| f64::from(c[0] as i8) / 128.0 * 10f64.powi(i32::from(c[1] % 16) - 8))
        .collect();
    solver_checks::check(&x);
});
