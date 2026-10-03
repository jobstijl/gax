//! A `Unit` versor's sandwich keeps the passenger's grade (the generator types it so, ADR-020):
//! checked on elements projected onto `u ~u = 1` by Newton's method, not built as products of
//! vectors, for the versor kinds whose sandwich keeps its grades only on that variety (5D even
//! and odd elements, CSTA's motors). The typed kernel equals the passenger's grade part of the
//! full product `u x ~u`, and the other grades of the full product vanish.

#![cfg(all(feature = "cga3d", feature = "stap", feature = "csta"))]

#[path = "support/rng.rs"]
mod rng;
use rng::Rng;

/// Newton's method with minimum-norm steps on `r(u) = 0`, `r` the components of `u ~u − 1`
/// (Jacobian by central differences, `(J Jᵀ + 10⁻¹⁴) y = r` since the relations' rows can be
/// dependent); `None` if it does not converge.
fn project<const N: usize>(mut u: [f64; N], r: impl Fn(&[f64; N]) -> Vec<f64>) -> Option<[f64; N]> {
    for _ in 0..80 {
        let f = r(&u);
        if f.iter().all(|x| x.abs() < 1e-14) {
            return Some(u);
        }
        let h = 1e-7;
        let mut jac = vec![[0.0f64; N]; f.len()];
        for c in 0..N {
            let (mut a, mut b) = (u, u);
            a[c] += h;
            b[c] -= h;
            let (fa, fb) = (r(&a), r(&b));
            for (row, j) in jac.iter_mut().enumerate() {
                j[c] = (fa[row] - fb[row]) / (2.0 * h);
            }
        }
        let rows: Vec<usize> = (0..f.len())
            .filter(|&i| jac[i].iter().any(|x| x.abs() > 1e-12))
            .collect();
        let k = rows.len();
        let mut a = vec![vec![0.0f64; k + 1]; k];
        for (i, &ri) in rows.iter().enumerate() {
            for (j, &rj) in rows.iter().enumerate() {
                let dot: f64 = (0..N).map(|c| jac[ri][c] * jac[rj][c]).sum();
                a[i][j] = dot + if i == j { 1e-14 } else { 0.0 };
            }
            a[i][k] = f[ri];
        }
        for col in 0..k {
            let piv = (col..k).max_by(|&x, &y| a[x][col].abs().total_cmp(&a[y][col].abs()))?;
            a.swap(col, piv);
            let pivot = a[col].clone();
            for (row, r) in a.iter_mut().enumerate() {
                if row != col {
                    let m = r[col] / pivot[col];
                    for (x, p) in r.iter_mut().zip(&pivot).skip(col) {
                        *x -= m * p;
                    }
                }
            }
        }
        let y: Vec<f64> = (0..k).map(|i| a[i][k] / a[i][i]).collect();
        for (c, uc) in u.iter_mut().enumerate() {
            *uc -= rows
                .iter()
                .enumerate()
                .map(|(i, &ri)| jac[ri][c] * y[i])
                .sum::<f64>();
        }
    }
    r(&u).iter().all(|x| x.abs() < 1e-12).then_some(u)
}

macro_rules! check {
    ($name:ident, $alg:ident, $versor:ident, $x:ident, $seed:expr) => {
        #[test]
        fn $name() {
            use gax::$alg::{Multivector, $versor, $x};
            let mut rng = Rng($seed);
            let (mut checked, mut worst) = (0, 0.0f64);
            for _ in 0..60 {
                let start: [f64; <$versor as gax::Kind>::N] =
                    core::array::from_fn(|_| rng.next_f64());
                let resid = |c: &[f64; <$versor as gax::Kind>::N]| -> Vec<f64> {
                    let u = $versor::<(), f64>::from_coeffs(*c);
                    let m: Multivector<(), f64> = (u * u.reverse()).cast::<Multivector>();
                    let mut r = m.c.to_vec();
                    r[0] -= 1.0;
                    r
                };
                let Some(c) = project(start, resid) else {
                    continue;
                };
                let u = gax::Unit::new_unchecked($versor::<(), f64>::from_coeffs(c));
                for _ in 0..3 {
                    let x = $x::<(), f64>::from_coeffs(core::array::from_fn(|_| rng.next_f64()));
                    let typed: $x<(), f64> = u >> x;
                    let full: Multivector<(), f64> =
                        (u.into_inner() * x * u.reverse().into_inner()).cast::<Multivector>();
                    let scale = full.c.iter().fold(1.0f64, |m, v| m.max(v.abs()));
                    let part: Multivector<(), f64> = typed.cast::<Multivector>();
                    for (a, b) in part.c.iter().zip(full.c) {
                        worst = worst.max((a - b).abs() / scale);
                    }
                }
                checked += 1;
            }
            assert!(checked > 20, "only {checked} projections converged");
            assert!(worst < 1e-10, "the sandwich left its grade by {worst:e}");
        }
    };
}

check!(cga3d_even_on_vectors, cga3d, Even, Vector, 0x00c6_a3e1);
check!(
    cga3d_even_on_quadvectors,
    cga3d,
    Even,
    Quadvector,
    0x00c6_a3e2
);
check!(cga3d_odd_on_vectors, cga3d, Odd, Vector, 0x00c6_a3e3);
check!(stap_motor_on_vectors, stap, Motor, Vector, 0x0057_a901);
check!(stap_odd_on_vectors, stap, Odd, Vector, 0x0057_a902);
check!(csta_motor_on_vectors, csta, Motor, Vector, 0x00c5_7a01);
