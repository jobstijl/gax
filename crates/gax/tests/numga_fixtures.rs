//! Cross-check against numga (<https://github.com/EelcoHoogendoorn/numga>): the generated
//! operations on full multivectors reproduce numga's results for the same inputs.
//!
//! The fixtures come from `docs/research/experiments/numga_fixtures.py`. It maps numga's
//! basis `x, y, (z), w` (with `w` degenerate) to gax's `e1, e2, (e3), e0`, including the
//! permutation and orientation signs, and writes dense coefficients in gax's canonical order.
//!
//! One convention differs, on purpose. gax follows bivector.net and orients the PGA3D
//! pseudoscalar as `e0123 = w∧x∧y∧z`, while numga uses `x∧y∧z∧w = −e0123`. The regressive
//! product and the complement are defined relative to the pseudoscalar, so in PGA3D they
//! differ from numga's by the relative orientation `σ = −1`. In PGA2D, `e012 = w∧x∧y` and
//! numga's `x∧y∧w` agree (`σ = +1`).

mod common;
use common::{Oracle, assert_close};
use gax::Extensor;

fn check<M>(o: &Oracle, fixture: &str, sigma: f64)
where
    M: Extensor<Slots = (), Coef = f64>
        + gax::Gp<M, Output = M>
        + gax::Wedge<M, Output = M>
        + gax::Vee<M, Output = M>
        + gax::Dot<M, Output = M>
        + gax::Commutator<M, Output = M>
        + gax::Reverse<Output = M>
        + gax::Dual<Output = M>,
{
    let mut checked = std::collections::BTreeMap::<String, usize>::new();
    for line in fixture
        .lines()
        .filter(|l| !l.starts_with('#') && !l.is_empty())
    {
        let mut it = line.split_whitespace();
        let op = it.next().unwrap();
        let v: Vec<f64> = it.map(|x| x.parse().unwrap()).collect();
        let n = v.len() / 3;
        let (a, b, want) = (&v[..n], &v[n..2 * n], &v[2 * n..]);
        let (ma, mb): (M, M) = (o.from_dense(&a.to_vec()), o.from_dense(&b.to_vec()));
        let got = match op {
            "gp" => ma.gp(mb),
            "wedge" => ma.wedge(mb),
            "vee" => ma.vee(mb),
            "dot" => ma.dot(mb),
            "commutator" => ma.commutator(mb),
            "reverse" => ma.reverse(),
            "dual" => ma.dual(),
            other => panic!("unknown op {other}"),
        };
        // vee and dual are defined relative to the pseudoscalar's orientation.
        let s = if op == "vee" || op == "dual" {
            sigma
        } else {
            1.0
        };
        let want: Vec<f64> = want.iter().map(|x| s * x).collect();
        assert_close(&o.dense(&got), &want, &format!("numga {op}"));
        *checked.entry(op.to_string()).or_default() += 1;
    }
    assert_eq!(checked.len(), 7, "every operation was checked: {checked:?}");
}

#[test]
#[cfg(feature = "pga2d")]
fn pga2d_matches_numga() {
    let o = Oracle::from_spec(include_str!("../specs/pga2d.gax"));
    check::<gax::pga2d::Multivector<(), f64>>(&o, include_str!("fixtures/numga_pga2d.txt"), 1.0);
}

#[test]
#[cfg(feature = "pga3d")]
fn pga3d_matches_numga() {
    let o = Oracle::from_spec(include_str!("../specs/pga3d.gax"));
    check::<gax::pga3d::Multivector<(), f64>>(&o, include_str!("fixtures/numga_pga3d.txt"), -1.0);
}
