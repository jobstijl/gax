//! Inclusions between the standard algebras by generator names (numga's `from_subalgebra`,
//! ADR-057), as `FromSubalgebra` impls.
//!
//! An algebra contains another when every generator of the smaller one is among its own, by
//! name, with the same inner products (`pga2d`'s `e0, e1, e2` are `pga3d`'s). A value of the
//! smaller algebra then has a place in the larger one, and which place follows from its
//! orientation (ADR-051):
//!
//! * an extrinsic value keeps its blades, `ι(b) = b` by name (a line `a e1 + b e2 + c e0` of the
//!   plane is the vertical plane of the same equation, a rotor the rotor about the same axis);
//! * an intrinsic value keeps its complement, `ι(x) = J(ι(J⁻¹ x))` with `J` the right
//!   complement (`dual`) of each algebra: a point of the plane is the point of space at `z = 0`,
//!   with its weight, and the join of two points of the plane the line of space through them.
//!
//! Both send each blade to one blade with a sign, so an inclusion is a signed reordering of
//! coefficients. Unlike a homomorphism (`emit_homs`, guide §12) it keeps the joins and the
//! pairings where the orientations say so: the generator proves, exactly and for every pair of
//! kinds, the laws listed in [`LAWS`] before it emits anything.

use crate::poly::Rational;
use crate::spec::{AlgebraSpec, KindSpec};
use crate::symbolic::{self, SymMv};
use crate::table::{BinOp, UnOp, blade_unop};
use std::collections::BTreeSet;
use std::fmt::Write as _;

/// How the generators of a source algebra sit among a target's.
pub struct Inclusion<'a> {
    /// The smaller algebra.
    pub src: &'a AlgebraSpec,
    /// The algebra holding its generators.
    pub tgt: &'a AlgebraSpec,
    /// For each source generator, its index in the target.
    index: Vec<usize>,
}

/// Per target coefficient, the source coefficient it reads and the sign (`None` where nothing
/// lands).
pub type Reads = Vec<Option<(usize, i64)>>;

/// Per source kind and orientation (`true` intrinsic), the target kind of its inclusion.
pub type KindImages = Vec<(String, bool, String)>;

/// The inclusion of `src` in `tgt`, where every generator of `src` is one of `tgt`'s by name,
/// with the same inner products with the others.
pub fn inclusion<'a>(src: &'a AlgebraSpec, tgt: &'a AlgebraSpec) -> Option<Inclusion<'a>> {
    let (sa, ta) = (&src.algebra, &tgt.algebra);
    if sa.dim() >= ta.dim() {
        return None;
    }
    let names = ta.basis_suffixes();
    let index: Vec<usize> = sa
        .basis_suffixes()
        .iter()
        .map(|c| names.iter().position(|d| d == c))
        .collect::<Option<_>>()?;
    let keeps_metric = (0..sa.dim())
        .all(|i| (0..sa.dim()).all(|j| sa.metric()[i][j] == ta.metric()[index[i]][index[j]]));
    keeps_metric.then_some(Inclusion { src, tgt, index })
}

/// The products an inclusion keeps, `ι(a ∘ b) == ι(a) ∘ ι(b)`, by the operands' orientations
/// (`true` intrinsic), the result taking the orientation ADR-051's rules give it. Proved by
/// [`emit_inclusion`] for every pair of source kinds, with the sandwiches and duals of
/// [`Inclusion::keeps_sandwich`] and [`Inclusion::keeps_dual`].
pub const LAWS: &[(BinOp, bool, bool)] = &[
    // Two extrinsic values: every product but the join (the extrinsic inclusion is the
    // homomorphism of the generators it shares).
    (BinOp::Gp, false, false),
    (BinOp::Wedge, false, false),
    (BinOp::Lc, false, false),
    (BinOp::Rc, false, false),
    (BinOp::Dot, false, false),
    (BinOp::Scalar, false, false),
    (BinOp::Commutator, false, false),
    (BinOp::Anticommutator, false, false),
    // An extrinsic value with an intrinsic one on its right: an intrinsic inclusion is the
    // extrinsic one times a fixed blade on the right.
    (BinOp::Gp, false, true),
    (BinOp::Wedge, false, true),
    (BinOp::Vee, false, true),
    (BinOp::Lc, false, true),
    // Two intrinsic values: the join.
    (BinOp::Vee, true, true),
];

/// The orientation of `a ∘ b` (ADR-051): the product of the signs, the join flipped.
fn product_orientation(op: BinOp, left: bool, right: bool) -> bool {
    (left != right) != (op == BinOp::Vee)
}

impl Inclusion<'_> {
    /// The target blade and sign of a canonical source blade, extrinsic: the same generators,
    /// reordered into the target's canonical order.
    fn extrinsic_blade(&self, mask: u32) -> (u32, i64) {
        let positions: Vec<usize> = (0..self.src.algebra.dim())
            .filter(|i| mask & (1 << i) != 0)
            .map(|i| self.index[i])
            .collect();
        let inversions: usize = positions
            .iter()
            .enumerate()
            .map(|(a, &i)| positions[a + 1..].iter().filter(|&&j| i > j).count())
            .sum();
        let m = positions.iter().fold(0u32, |m, &i| m | (1 << i));
        (m, if inversions.is_multiple_of(2) { 1 } else { -1 })
    }

    /// The target blade and sign of a canonical source blade in either orientation: the same
    /// blade for an extrinsic value; for an intrinsic one `J(ι(J⁻¹ b))`.
    pub fn blade(&self, mask: u32, intrinsic: bool) -> (u32, i64) {
        if !intrinsic {
            return self.extrinsic_blade(mask);
        }
        let (c, s1) = blade_unop(&self.src.algebra, UnOp::Undual, mask);
        let (e, s2) = self.extrinsic_blade(c);
        let (d, s3) = blade_unop(&self.tgt.algebra, UnOp::Dual, e);
        (d, s1 * s2 * s3)
    }

    /// The image of a symbolic multivector.
    pub fn image(&self, mv: &SymMv, intrinsic: bool) -> SymMv {
        mv.iter()
            .map(|(&m, p)| {
                let (t, s) = self.blade(m, intrinsic);
                (t, p.scale(Rational::int(i128::from(s))))
            })
            .collect()
    }

    /// Whether `ι(a ∘ b) == ι(a) ∘ ι(b)` holds for symbolic values of kinds `a` and `b`.
    pub fn keeps(&self, op: BinOp, a: &KindSpec, b: &KindSpec, left: bool, right: bool) -> bool {
        let (sa, ta) = (&self.src.algebra, &self.tgt.algebra);
        let va = symbolic::variables(&a.layout, 0);
        let vb = symbolic::variables(&b.layout, a.layout.len() as crate::poly::Var);
        let lhs = self.image(
            &symbolic::binop(sa, op, &va, &vb),
            product_orientation(op, left, right),
        );
        let rhs = symbolic::binop(ta, op, &self.image(&va, left), &self.image(&vb, right));
        lhs == rhs
    }

    /// Whether `ι(v >> x) == ι(v) >> ι(x)` for an extrinsic versor kind `v` and a passenger of
    /// either orientation (ADR-051's `>>`: `v x ~v`, of the involute and negated for an
    /// intrinsic passenger where `v` is odd).
    pub fn keeps_sandwich(&self, v: &KindSpec, x: &KindSpec, intrinsic: bool) -> bool {
        let sandwich = |alg: &crate::algebra::Algebra, vv: &SymMv, xx: &SymMv| -> SymMv {
            let odd = vv.keys().all(|m| m.count_ones() % 2 == 1);
            let even = vv.keys().all(|m| m.count_ones() % 2 == 0);
            assert!(odd || even, "a versor has one parity");
            let passenger = if odd {
                symbolic::unop(alg, UnOp::Involute, xx)
            } else {
                xx.clone()
            };
            let r = symbolic::binop(
                alg,
                BinOp::Gp,
                &symbolic::binop(alg, BinOp::Gp, vv, &passenger),
                &symbolic::unop(alg, UnOp::Reverse, vv),
            );
            if odd && intrinsic {
                symbolic::scale_mv(&r, &crate::poly::Poly::constant(Rational::int(-1)))
            } else {
                r
            }
        };
        let (sa, ta) = (&self.src.algebra, &self.tgt.algebra);
        let vv = symbolic::variables(&v.layout, 0);
        let vx = symbolic::variables(&x.layout, v.layout.len() as crate::poly::Var);
        let lhs = self.image(&sandwich(sa, &vv, &vx), intrinsic);
        let rhs = sandwich(ta, &self.image(&vv, false), &self.image(&vx, intrinsic));
        lhs == rhs
    }

    /// Whether `ι(x.dual()) == ι(x).dual()` for extrinsic `x` and `ι(y.undual()) ==
    /// ι(y).undual()` for intrinsic `y`.
    pub fn keeps_dual(&self, x: &KindSpec) -> bool {
        let (sa, ta) = (&self.src.algebra, &self.tgt.algebra);
        let v = symbolic::variables(&x.layout, 0);
        self.image(&symbolic::unop(sa, UnOp::Dual, &v), true)
            == symbolic::unop(ta, UnOp::Dual, &self.image(&v, false))
            && self.image(&symbolic::unop(sa, UnOp::Undual, &v), false)
                == symbolic::unop(ta, UnOp::Undual, &self.image(&v, true))
    }

    /// The target kind of source kind `a` in an orientation, and per target coefficient the
    /// source coefficient and sign it reads (`None` where nothing lands).
    pub fn kind_image(&self, a: &KindSpec, intrinsic: bool) -> Option<(&KindSpec, Reads)> {
        let images: Vec<(u32, i64)> = a
            .layout
            .blades
            .iter()
            .map(|&(m, s)| {
                let (t, ts) = self.blade(m, intrinsic);
                (t, s * ts)
            })
            .collect();
        let support: BTreeSet<u32> = images.iter().map(|&(t, _)| t).collect();
        // The smallest kind holding the image, parts included: an inclusion's image is a
        // subspace, which a part names exactly (a VGA3D offset is a PGA3D `Direction`). Not a
        // slot declaration, which takes part in no product.
        let grades: BTreeSet<u32> = support.iter().map(|m| m.count_ones()).collect();
        let b = self
            .tgt
            .kinds
            .iter()
            .filter(|k| k.operand && support.is_subset(&k.layout.support()))
            .min_by_key(|k| (!k.layout.grades().is_subset(&grades), k.layout.len()))?;
        let reads = b
            .layout
            .blades
            .iter()
            .map(|&(tm, ts)| {
                images
                    .iter()
                    .position(|&(t, _)| t == tm)
                    .map(|k| (k, images[k].1 * ts))
            })
            .collect();
        Some((b, reads))
    }
}

/// Emit the `FromSubalgebra` impls of one inclusion, gated on both algebras' features, after
/// proving [`LAWS`], the sandwiches and the duals for every pair of source kinds.
///
/// Returns the code and, per source kind and orientation (`true` intrinsic), the target kind.
///
/// # Errors
/// If a law fails (a bug in the derivation: the laws hold for every inclusion).
pub fn emit_inclusion(inc: &Inclusion<'_>) -> Result<(String, KindImages), String> {
    let (src, tgt) = (inc.src, inc.tgt);
    let (sm, tm) = (src.name.as_str(), tgt.name.as_str());
    for &(op, left, right) in LAWS {
        for a in &src.kinds {
            for b in &src.kinds {
                if !inc.keeps(op, a, b, left, right) {
                    return Err(format!(
                        "{sm} -> {tm}: the inclusion does not keep {op:?} of {} ({}) and {} ({})",
                        a.name,
                        crate::spec::orientation_name(left),
                        b.name,
                        crate::spec::orientation_name(right)
                    ));
                }
            }
        }
    }
    for v in src.kinds.iter().filter(|k| k.versor) {
        for x in &src.kinds {
            for intrinsic in [false, true] {
                if !inc.keeps_sandwich(v, x, intrinsic) {
                    return Err(format!(
                        "{sm} -> {tm}: the inclusion does not keep {} >> {}",
                        v.name, x.name
                    ));
                }
            }
        }
    }
    if let Some(x) = src.kinds.iter().find(|x| !inc.keeps_dual(x)) {
        return Err(format!(
            "{sm} -> {tm}: the inclusion does not keep the dual of {}",
            x.name
        ));
    }
    let mut out = String::new();
    let _ = writeln!(
        out,
        "/// The inclusion of `{sm}` in `{tm}` by generator names.\n#[cfg(all(feature = \"{sm}\", feature = \"{tm}\"))]\nmod {sm}_in_{tm} {{\n    use crate::{{Coef, Slots, __private::SlotArr}};\n"
    );
    let mut kinds = Vec::new();
    for a in &src.kinds {
        for intrinsic in [false, true] {
            let Some((b, reads)) = inc.kind_image(a, intrinsic) else {
                continue;
            };
            let exprs: Vec<String> = reads
                .iter()
                .map(|r| {
                    let terms: Vec<(i128, usize)> =
                        r.iter().map(|&(k, s)| (i128::from(s), k)).collect();
                    crate::emit_homs::combination(&terms)
                })
                .collect();
            let o = if intrinsic { "Intrinsic" } else { "Extrinsic" };
            let (an, bn) = (&a.name, &b.name);
            let what = if intrinsic {
                "an intrinsic value keeps its complement"
            } else {
                "an extrinsic value keeps its blades"
            };
            let _ = write!(
                out,
                "    impl<S: Slots, T: Coef> crate::FromSubalgebra<crate::{sm}::oriented::{an}<S, T, crate::{o}>> for crate::{tm}::oriented::{bn}<S, T, crate::{o}> {{\n        /// The inclusion of `{sm}` in `{tm}` by generator names: {what} (ADR-057).\n        #[inline(always)]\n        fn from_subalgebra(x: crate::{sm}::oriented::{an}<S, T, crate::{o}>) -> Self {{\n            let a = x.c.map(SlotArr::<S, T>);\n            crate::{tm}::oriented::{bn}::from_coeffs([{}])\n        }}\n    }}\n\n",
                exprs.join(", ")
            );
            kinds.push((an.clone(), intrinsic, bn.clone()));
        }
    }
    out.push_str("}\n\n");
    Ok((out, kinds))
}

/// The inclusions among `specs`: every ordered pair where one holds the other's generators.
pub fn inclusions(specs: &[&AlgebraSpec]) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    for (i, s) in specs.iter().enumerate() {
        for (j, t) in specs.iter().enumerate() {
            if i != j && inclusion(s, t).is_some() {
                pairs.push((i, j));
            }
        }
    }
    pairs
}

/// A test of the emitted inclusions on random values: the dual laws on every kind in both
/// orientations; the geometric product of an extrinsic versor with an extrinsic single-grade
/// kind; and on the single-grade kinds the join of intrinsic values and the pairing of an
/// extrinsic value with an intrinsic one; compared as target multivectors. The generator
/// proves every law on every pair of kinds; this checks the code that renders them.
pub fn emit_inclusion_test(
    inc: &Inclusion<'_>,
    kinds: &KindImages,
    products: &[(BinOp, String, String, String)],
) -> String {
    let (src, tgt) = (inc.src, inc.tgt);
    let (sm, tm) = (src.name.as_str(), tgt.name.as_str());
    let image = |k: &str, i: bool| {
        kinds
            .iter()
            .find(|(a, o, _)| a == k && *o == i)
            .map(|(_, _, b)| b.as_str())
    };
    let ty = |alg: &str, k: &str, i: bool| {
        let o = if i { "Intrinsic" } else { "Extrinsic" };
        format!("{alg}::oriented::{k}<(), f64, gax::{o}>")
    };
    let kind = |name: &str| src.kinds.iter().find(|k| k.name == name);
    let single = |name: &str| kind(name).is_some_and(|k| k.result && k.layout.grades().len() == 1);
    let versor = |name: &str| kind(name).is_some_and(|k| k.result && k.versor);
    let mut fns = String::new();
    let mut calls = String::new();
    let mut n = 0;
    let mut check = |fns: &mut String, body: String| {
        let _ = writeln!(
            fns,
            "    #[inline(never)]\n    fn c{n}(rng: &mut Rng) {{\n{body}    }}"
        );
        let _ = writeln!(calls, "        c{n}(&mut rng);");
        n += 1;
    };
    // The duals: `ι(x.dual()) == ι(x).dual()` for extrinsic x, `ι(y.undual()) ==
    // ι(y).undual()` for intrinsic y.
    for a in &src.kinds {
        let va = symbolic::variables(&a.layout, 0);
        for (intrinsic, op, method) in [(false, UnOp::Dual, "dual"), (true, UnOp::Undual, "undual")]
        {
            let support = symbolic::support(&symbolic::unop(&src.algebra, op, &va));
            let Some(d) = src.result_kind(&support, &[a]) else {
                continue;
            };
            let (Some(ia), Some(id)) = (image(&a.name, intrinsic), image(&d.name, !intrinsic))
            else {
                continue;
            };
            check(
                &mut fns,
                format!(
                    "        let x: {} = rng.value();\n        let lhs = {tm}::oriented::Multivector::from({}::from_subalgebra(x.{method}()));\n        let rhs = {tm}::oriented::Multivector::from({}::from_subalgebra(x).{method}());\n        assert!(close(&lhs, &rhs), \"{} {method}\");\n",
                    ty(sm, &a.name, intrinsic),
                    ty(tm, id, !intrinsic).replace("<(), f64, ", "::<(), f64, "),
                    ty(tm, ia, intrinsic).replace("<(), f64, ", "::<(), f64, "),
                    a.name,
                ),
            );
        }
    }
    // The products.
    for (op, left, right, method) in [
        (BinOp::Gp, false, false, "gp"),
        (BinOp::Vee, true, true, "vee"),
        (BinOp::Vee, false, true, "vee"),
    ] {
        for (pop, a, b, c) in products {
            let first = if op == BinOp::Gp {
                versor(a)
            } else {
                single(a)
            };
            if *pop != op || !first || !single(b) {
                continue;
            }
            let out = product_orientation(op, left, right);
            let (Some(ia), Some(ib)) = (image(a, left), image(b, right)) else {
                continue;
            };
            let lhs = match image(c, out) {
                Some(ic) => format!(
                    "{tm}::oriented::Multivector::from({}::from_subalgebra(a.{method}(b)))",
                    ty(tm, ic, out).replace("<(), f64, ", "::<(), f64, ")
                ),
                None => format!(
                    "{tm}::oriented::Multivector::<(), f64, gax::{}>::zero()",
                    if out { "Intrinsic" } else { "Extrinsic" }
                ),
            };
            check(
                &mut fns,
                format!(
                    "        let (a, b): ({}, {}) = (rng.value(), rng.value());\n        let rhs = {tm}::oriented::Multivector::from({}::from_subalgebra(a).{method}({}::from_subalgebra(b)));\n        assert!(close(&{lhs}, &rhs), \"{a} {method} {b}\");\n",
                    ty(sm, a, left),
                    ty(sm, b, right),
                    ty(tm, ia, left).replace("<(), f64, ", "::<(), f64, "),
                    ty(tm, ib, right).replace("<(), f64, ", "::<(), f64, "),
                ),
            );
        }
    }
    format!(
        "/// The inclusion of `{sm}` in `{tm}`.\n#[cfg(all(feature = \"{sm}\", feature = \"{tm}\"))]\n#[test]\nfn {sm}_in_{tm}() {{\n    use gax::{{{sm}, {tm}}};\n{fns}    let mut rng = Rng(rand::SeedableRng::seed_from_u64(0x{seed:x}));\n    for _ in 0..4 {{\n{calls}    }}\n}}\n\n",
        seed = sm.len() * 7919 + tm.len() * 104_729 + 3,
    )
}

/// The head of the generated test file.
pub const TEST_HEAD: &str = "//! @generated by gax-regen. Do not edit by hand.
//!
//! The inclusions between the standard algebras by generator names (ADR-057) keep the duals,
//! the geometric product of extrinsic values, the join of intrinsic ones and the pairing of an
//! extrinsic value with an intrinsic one, compared as multivectors of the target on random
//! values (the generator proves these and the other laws exactly on every pair of kinds).

#![cfg_attr(rustfmt, rustfmt::skip)]
// The helpers go unused when no pair of algebras here is enabled.
#![allow(dead_code, clippy::pedantic, clippy::too_many_lines, clippy::useless_conversion)]

use gax::Extensor;

struct Rng(rand::rngs::StdRng);
impl Rng {
    fn value<M: Extensor<Slots = (), Coef = f64>>(&mut self) -> M {
        M::from_coeffs(<M::Kind as gax::Kind>::arr_from_fn(|_| rand::Rng::random_range(&mut self.0, -1.0..1.0)))
    }
}

fn close<M: Extensor<Slots = (), Coef = f64>>(a: &M, b: &M) -> bool {
    let (a, b) = (a.coeffs().as_ref(), b.coeffs().as_ref());
    a.iter().zip(b).all(|(x, y)| (x - y).abs() <= 1e-12 * (1.0 + x.abs().max(y.abs())))
}

";

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(name: &str) -> AlgebraSpec {
        let path = format!("{}/../gax/specs/{name}.gax", env!("CARGO_MANIFEST_DIR"));
        AlgebraSpec::parse(&std::fs::read_to_string(path).unwrap()).unwrap()
    }

    /// The pairs found by generator names, and the point of the plane at `z = 0`.
    #[test]
    fn pairs_and_points() {
        let (pga2d, pga3d, cga2d) = (spec("pga2d"), spec("pga3d"), spec("cga2d"));
        assert!(
            inclusion(&pga3d, &pga2d).is_none(),
            "the larger is never included"
        );
        assert!(
            inclusion(&pga2d, &cga2d).is_none(),
            "e0 is not a CGA generator"
        );
        let inc = inclusion(&pga2d, &pga3d).expect("pga2d's generators are pga3d's");
        let point = pga2d.kinds.iter().find(|k| k.name == "Point").unwrap();
        let (b, reads) = inc.kind_image(point, true).unwrap();
        // x e20 + y e01 + w e12 to x e032 + y e013 + 0 e021 + w e123, signs included.
        assert_eq!(b.name, "Point");
        assert_eq!(reads, [Some((0, 1)), Some((1, 1)), None, Some((2, 1))]);
    }

    /// What is left out of [`LAWS`] fails in some pair of algebras, as laws.md says: the join of
    /// extrinsic values, the meet of intrinsic ones, an intrinsic value's products on the left.
    #[test]
    fn the_left_out_laws_fail_somewhere() {
        let (pga2d, pga3d) = (spec("pga2d"), spec("pga3d"));
        let inc = inclusion(&pga2d, &pga3d).unwrap();
        for op in [BinOp::Gp, BinOp::Wedge, BinOp::Vee, BinOp::Lc] {
            for (l, r) in [(false, false), (false, true), (true, false), (true, true)] {
                let all = pga2d
                    .kinds
                    .iter()
                    .all(|a| pga2d.kinds.iter().all(|b| inc.keeps(op, a, b, l, r)));
                assert_eq!(all, LAWS.contains(&(op, l, r)), "{op:?} {l} {r}");
            }
        }
    }
}
