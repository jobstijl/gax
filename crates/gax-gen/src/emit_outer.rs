//! Outermorphisms: the extension of a linear map on vectors (by `∧`) or antivectors (by `∨`)
//! to every homogeneous kind, derived symbolically as minors of the map's matrix.

use crate::cse;
use crate::poly::{Poly, Rational, Var};
use crate::slp::render;
use crate::spec::{AlgebraSpec, KindSpec};
use crate::symbolic::{self, SymMv};
use crate::table::{BinOp, UnOp, blade_unop};
use std::collections::BTreeSet;
use std::fmt::Write as _;

fn single_grade(k: &KindSpec) -> Option<u32> {
    let g: BTreeSet<u32> = k
        .layout
        .blades
        .iter()
        .map(|(m, _)| m.count_ones())
        .collect();
    (g.len() == 1).then(|| *g.iter().next().expect("one"))
}

/// Emit `Outermorphism<B>` impls for maps on the grade-1 and grade-(n-1) kinds.
pub fn outermorphisms(spec: &AlgebraSpec) -> String {
    let alg = &spec.algebra;
    let n = alg.dim() as u32;
    let mut out = String::new();
    for v in &spec.kinds {
        let Some(gv) = single_grade(v) else { continue };
        // Only kinds holding *all* blades of their grade can be the base of an outermorphism.
        let all = (0..alg.blade_count() as u32)
            .filter(|m| m.count_ones() == gv)
            .count();
        if v.layout.len() != all || !(gv == 1 || (gv + 1 == n && n > 2)) {
            continue;
        }
        let wedge = gv == 1;
        let nv = v.layout.len();
        // The map's entries t[o][i] as variables o * nv + i.
        let image = |layout_pos: usize| -> SymMv {
            let coeffs: Vec<Poly> = (0..nv)
                .map(|o| Poly::var((o * nv + layout_pos) as Var))
                .collect();
            symbolic::from_coeffs(&v.layout, &coeffs)
        };
        // T applied to the canonical basis element with mask m (orientation from V's layout).
        let apply = |m: u32| -> Option<SymMv> {
            let (p, s) = v.layout.position(m)?;
            Some(
                image(p)
                    .into_iter()
                    .map(|(b, c)| (b, c.scale(Rational::int(i128::from(s)))))
                    .collect(),
            )
        };
        for b in &spec.kinds {
            let Some(gb) = single_grade(b) else { continue };
            if b.name == v.name || (wedge && gb < 1) {
                continue;
            }
            // The number of vector factors of a blade of B.
            let k = if wedge { gb } else { n - gb };
            if k < 2 && !(k == 0) {
                continue;
            }
            let mut columns: Vec<Vec<Poly>> = Vec::new();
            let mut ok = true;
            for &(ma, sa) in &b.layout.blades {
                // Factor e_A into basis (anti)vectors.
                let (factors, r) = if wedge {
                    (ma, 1i64)
                } else {
                    let (c, r) = alg.right_complement(ma);
                    (c, r)
                };
                let mut col: Option<SymMv> = None;
                for i in 0..n {
                    if factors & (1 << i) == 0 {
                        continue;
                    }
                    let m = if wedge {
                        1 << i
                    } else {
                        blade_unop(alg, UnOp::Undual, 1 << i).0
                    };
                    let sign = if wedge {
                        1
                    } else {
                        blade_unop(alg, UnOp::Undual, 1 << i).1
                    };
                    let Some(mut img) = apply(m) else {
                        ok = false;
                        break;
                    };
                    if sign < 0 {
                        img = img.into_iter().map(|(bb, c)| (bb, -&c)).collect();
                    }
                    col = Some(match col {
                        None => img,
                        Some(acc) => symbolic::binop(
                            alg,
                            if wedge { BinOp::Wedge } else { BinOp::Vee },
                            &acc,
                            &img,
                        ),
                    });
                }
                if !ok {
                    break;
                }
                let col = col.unwrap_or_else(|| symbolic::scalar(Rational::ONE));
                let scaled: SymMv = col
                    .into_iter()
                    .map(|(bb, c)| (bb, c.scale(Rational::int(i128::from(sa * r)))))
                    .collect();
                match symbolic::to_coeffs(&b.layout, &scaled) {
                    Some(c) => columns.push(c),
                    None => {
                        ok = false;
                        break;
                    }
                }
            }
            if !ok {
                continue;
            }
            let nb = b.layout.len();
            let polys: Vec<Poly> = (0..nb)
                .flat_map(|o| columns.iter().map(move |c| c[o].clone()))
                .collect();
            let prog = cse::compile_best(&polys, &BTreeSet::new(), &[]);
            let name = |var: Var| format!("t[{}][{}]", var as usize / nv, var as usize % nv);
            let mut lets = String::new();
            prog.emit_lets(&name, "m", &mut lets);
            let rows: Vec<String> = (0..nb)
                .map(|o| {
                    let row: Vec<String> = (0..nb)
                        .map(|a| render(&prog.outputs[o * nb + a], &name, "m"))
                        .collect();
                    format!("[{}]", row.join(", "))
                })
                .collect();
            let (vn, bn) = (&v.name, &b.name);
            let how = if wedge { "∧" } else { "∨" };
            let _ = write!(
                out,
                "impl<T: gx::Coef> gx::Outermorphism<{bn}> for {vn}<({vn},), T> {{\n    type Output = {bn}<({bn},), T>;\n    /// The extension of a map on `{vn}` to `{bn}`, factor by factor with `{how}`.\n    #[inline]\n    fn outermorphism(self) -> {bn}<({bn},), T> {{\n        let t = self.c;\n{lets}        {bn} {{ c: [{}] }}\n    }}\n}}\n\n",
                rows.join(", ")
            );
        }
    }
    out
}
