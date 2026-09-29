//! The linear algebra on maps never panics and is accurate where the problem is well
//! conditioned: random matrices, singular ones, and badly scaled ones (`support/solver_checks.rs`;
//! the `solve` fuzz target runs the same checks on arbitrary bytes).

#![cfg(feature = "pga3d")]

#[path = "support/solver_checks.rs"]
mod solver_checks;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }
}

#[test]
fn random_singular_and_badly_scaled_matrices() {
    let mut rng = Rng(0xdead_beef_1234);
    for k in 0..3000 {
        let mut x: Vec<f64> = (0..83).map(|_| rng.unit()).collect();
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
                    if !rng.next().is_multiple_of(3) {
                        *v = 0.0;
                    }
                }
            }
            // Non-finite entries: nothing may panic.
            4 => x[(rng.next() % 83) as usize] = [f64::NAN, f64::INFINITY, -f64::INFINITY][k % 3],
            _ => {}
        }
        solver_checks::check(&x);
    }
}
