//! The tiers agree in `f32` within computed error bounds (docs/numerics.md).
//!
//! For every plain sandwich kernel of the standard algebras, the tier-1 program (the products
//! as the tables give them, no simplification) and the fused tier-2 program are evaluated in
//! `f32` on random inputs. They are different programs, so they are not bit-identical; their
//! results must differ by at most the sum of their a-priori forward error bounds
//! (`Program::error_bound`), not by an ad-hoc tolerance.

use gax_gen::cse::{self, Options};
use gax_gen::poly::{Poly, Var};
use gax_gen::spec::AlgebraSpec;
use gax_gen::symbolic;
use gax_gen::table::{BinOp, UnOp};
use std::collections::BTreeSet;

const U32: f64 = 1.0 / (1u64 << 24) as f64;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

/// (kernels checked, the largest observed difference as a fraction of the bound)
fn check(src: &str) -> (usize, f64) {
    let spec = AlgebraSpec::parse(src).unwrap();
    let alg = &spec.algebra;
    let mut rng = Rng(0x1234_5678);
    let (mut count, mut worst) = (0, 0.0f64);
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
            if outputs.is_empty() {
                continue;
            }
            let passengers: BTreeSet<Var> = (nv..nv + xk.layout.len() as Var).collect();
            let tier1 = cse::compile(
                &outputs,
                &Options {
                    passengers: BTreeSet::new(),
                    relations: Vec::new(),
                    kernels: false,
                },
            );
            let fused = cse::compile_best(&outputs, &passengers, &[]);
            let n = nv as usize + xk.layout.len();
            for _ in 0..20 {
                let inputs: Vec<f32> = (0..n).map(|_| rng.next()).collect();
                let get32 = |i: Var| inputs[i as usize];
                let mag = |i: Var| f64::from(inputs[i as usize]).abs();
                let (a, b) = (tier1.eval_f32(&get32), fused.eval_f32(&get32));
                let (ba, bb) = (tier1.error_bound(&mag, U32), fused.error_bound(&mag, U32));
                for k in 0..a.len() {
                    let diff = f64::from((a[k] - b[k]).abs());
                    let bound = ba[k] + bb[k];
                    assert!(
                        diff <= bound,
                        "{} {} >> {}: output {k} differs by {diff:e}, bound {bound:e}",
                        spec.name,
                        vk.name,
                        xk.name
                    );
                    if bound > 0.0 {
                        worst = worst.max(diff / bound);
                    }
                }
            }
            count += 1;
        }
    }
    (count, worst)
}

#[test]
fn tiers_agree_within_their_error_bounds() {
    for (name, src) in [
        ("pga2d", include_str!("../../gax/specs/pga2d.gax")),
        ("pga3d", include_str!("../../gax/specs/pga3d.gax")),
        ("vga3d", include_str!("../../gax/specs/vga3d.gax")),
        ("sta", include_str!("../../gax/specs/sta.gax")),
        ("cga3d", include_str!("../../gax/specs/cga3d.gax")),
    ] {
        let (n, worst) = check(src);
        println!("{name}: {n} kernels, largest difference {worst:.3} of the bound");
        assert!(n > 10);
    }
}

#[test]
fn the_bound_holds_for_a_known_cancellation() {
    // x*x - y*y at x = y + tiny: the bound is proportional to the magnitudes, not the result.
    let p = Poly::var(0);
    let q = Poly::var(1);
    let prog = cse::compile(
        &[&(&p * &p) - &(&q * &q)],
        &Options {
            passengers: BTreeSet::new(),
            relations: Vec::new(),
            kernels: false,
        },
    );
    let (x, y) = (1.000_001f32, 1.0f32);
    let computed = prog.eval_f32(&|v| if v == 0 { x } else { y })[0];
    let exact = f64::from(x) * f64::from(x) - f64::from(y) * f64::from(y);
    let bound = prog.error_bound(&|v| if v == 0 { f64::from(x) } else { f64::from(y) }, U32)[0];
    assert!((f64::from(computed) - exact).abs() <= bound);
    assert!(bound < 4.0 * U32 * 2.0);
}
