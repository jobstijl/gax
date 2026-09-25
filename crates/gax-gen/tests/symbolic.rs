//! The simplified kernels are algebraically identical to the unsimplified ones.
//!
//! For every fused sandwich of PGA2D and PGA3D (plain and `Unit`), the simplifier's program
//! is expanded back into polynomials and compared with the traced polynomials: exactly for
//! plain versors, and modulo a Gröbner basis of the unit conditions for `Unit` ones (which is
//! the identity the simplification is allowed to use). The matrix paths are checked the same
//! way, entry by entry.

use gax_gen::cse;
use gax_gen::groebner::{groebner, normal_form};
use gax_gen::poly::{Poly, Var};
use gax_gen::spec::AlgebraSpec;
use gax_gen::symbolic;
use gax_gen::table::{BinOp, UnOp};
use std::collections::BTreeSet;

fn check_algebra(src: &str) -> usize {
    let spec = AlgebraSpec::parse(src).unwrap();
    let alg = &spec.algebra;
    let mut checked = 0;
    for vk in spec.kinds.iter().filter(|k| k.versor) {
        for xk in &spec.kinds {
            let nv = vk.layout.len() as Var;
            let v = symbolic::variables(&vk.layout, 0);
            let x = symbolic::variables(&xk.layout, nv);
            let res = symbolic::binop(
                alg,
                BinOp::Gp,
                &symbolic::binop(alg, BinOp::Gp, &v, &x),
                &symbolic::unop(alg, UnOp::Reverse, &v),
            );
            let outputs: Vec<Poly> = res.values().cloned().collect();
            let passengers: BTreeSet<Var> = (nv..nv + xk.layout.len() as Var).collect();
            for unit in [false, true] {
                let relations = if unit {
                    symbolic::unit_relations(alg, &v)
                } else {
                    Vec::new()
                };
                if unit && relations.is_empty() {
                    continue;
                }
                let basis = groebner(&relations, 2000).expect("small ideal");
                let prog = cse::compile_best(&outputs, &passengers, &relations);
                let got = prog.to_polys();
                for (g, w) in got.iter().zip(&outputs) {
                    let d = g - w;
                    assert!(
                        normal_form(&d, &basis).is_zero(),
                        "{}{} >> {}: simplified output differs from the product",
                        if unit { "Unit " } else { "" },
                        vk.name,
                        xk.name
                    );
                }
                checked += 1;
            }
        }
    }
    checked
}

#[test]
fn pga2d_kernels_are_identical_to_the_products() {
    let n = check_algebra(include_str!("../../gax/specs/pga2d.gax"));
    assert!(n > 50, "checked {n}");
}

#[test]
fn pga3d_kernels_are_identical_to_the_products() {
    let n = check_algebra(include_str!("../../gax/specs/pga3d.gax"));
    assert!(n > 80, "checked {n}");
}

#[test]
fn cse_preserves_random_polynomial_systems() {
    // Random sparse polynomial systems in 6 variables: every strategy's program expands back
    // to the input exactly.
    let mut seed = 12345u64;
    let mut rnd = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    for _ in 0..50 {
        let polys: Vec<Poly> = (0..4)
            .map(|_| {
                let mut p = Poly::zero();
                for _ in 0..(rnd() % 8 + 1) {
                    let deg = rnd() % 4;
                    let m: Vec<Var> = {
                        let mut v: Vec<Var> = (0..deg).map(|_| (rnd() % 6) as Var).collect();
                        v.sort_unstable();
                        v
                    };
                    let c = (rnd() % 7) as i128 - 3;
                    p.add_term(gax_gen::poly::Monomial(m), gax_gen::poly::Rational::int(c));
                }
                p
            })
            .collect();
        for passengers in [BTreeSet::new(), BTreeSet::from([4, 5])] {
            let prog = cse::compile_best(&polys, &passengers, &[]);
            assert_eq!(prog.to_polys(), polys);
        }
    }
}
