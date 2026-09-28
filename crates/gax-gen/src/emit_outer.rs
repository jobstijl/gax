//! Outermorphisms: the extension of a linear map on vectors (by `∧`) or antivectors (by `∨`)
//! to every homogeneous kind, derived symbolically as minors of the map's matrix.

use crate::cse;
use crate::poly::{Poly, Rational, Var};
use crate::slp::render;
use crate::spec::{AlgebraSpec, KindSpec};
use crate::symbolic::{self, SymMv};
use crate::table::{BinOp, UnOp, blade_unop};
use std::collections::BTreeSet;

fn single_grade(k: &KindSpec) -> Option<u32> {
    let g: BTreeSet<u32> = k
        .layout
        .blades
        .iter()
        .map(|(m, _)| m.count_ones())
        .collect();
    (g.len() == 1).then(|| *g.iter().next().expect("one"))
}

/// Emit `Outermorphism<B>` impls for maps on the grade-1 and grade-(n-1) kinds, and list
/// them as `(base kind, extended kind, whether the extended kind is the top)`. The pairs are
/// independent, so they are derived in parallel and written in order.
pub fn outermorphisms(spec: &AlgebraSpec) -> (String, Vec<(String, String, bool)>) {
    let alg = &spec.algebra;
    let n = alg.dim() as u32;
    let bases = spec.kinds.iter().filter(|v| {
        let Some(gv) = single_grade(v) else {
            return false;
        };
        // Only kinds holding *all* blades of their grade can be the base of an outermorphism.
        let all = (0..alg.blade_count() as u32)
            .filter(|m| m.count_ones() == gv)
            .count();
        v.layout.len() == all && (gv == 1 || (gv + 1 == n && n > 2))
    });
    let jobs: Vec<(&KindSpec, &KindSpec)> = bases
        .flat_map(|v| spec.kinds.iter().map(move |b| (v, b)))
        .collect();
    let done = crate::par::map_heaviest_first(
        &jobs,
        |(v, b)| v.layout.len() * b.layout.len(),
        |&(v, b)| {
            crate::par::timed(
                || format!("{} outermorphism {} -> {}", spec.name, v.name, b.name),
                || outermorphism(spec, v, b),
            )
        },
    );
    let mut out = String::new();
    let mut pairs = Vec::new();
    for (code, pair) in done.into_iter().flatten() {
        out.push_str(&code);
        pairs.push(pair);
    }
    (out, pairs)
}

/// The outermorphism from maps on the base kind `v` to the kind `b`, if `b` extends it.
#[allow(clippy::too_many_lines)]
fn outermorphism(
    spec: &AlgebraSpec,
    v: &KindSpec,
    b: &KindSpec,
) -> Option<(String, (String, String, bool))> {
    let alg = &spec.algebra;
    let n = alg.dim() as u32;
    let gv = single_grade(v)?;
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
    let gb = single_grade(b)?;
    if b.name == v.name || (wedge && gb < 1) {
        return None;
    }
    // The number of vector factors of a blade of B.
    let k = if wedge { gb } else { n - gb };
    // One factor: B is another kind of V's grade, not an extension.
    if k == 1 {
        return None;
    }
    let mut columns: Vec<Vec<Poly>> = Vec::new();
    for &(ma, sa) in &b.layout.blades {
        // Factor e_A into basis (anti)vectors.
        let (factors, r) = if wedge {
            (ma, 1i64)
        } else {
            alg.right_complement(ma)
        };
        let mut col: Option<SymMv> = None;
        for i in 0..n {
            if factors & (1 << i) == 0 {
                continue;
            }
            let (m, sign) = if wedge {
                (1 << i, 1)
            } else {
                blade_unop(alg, UnOp::Undual, 1 << i)
            };
            let mut img = apply(m)?;
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
        let col = col.unwrap_or_else(|| symbolic::scalar(Rational::ONE));
        let scaled: SymMv = col
            .into_iter()
            .map(|(bb, c)| (bb, c.scale(Rational::int(i128::from(sa * r)))))
            .collect();
        columns.push(symbolic::to_coeffs(&b.layout, &scaled)?);
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
    let code = format!(
        "impl<T: gx::Coef> gx::Outermorphism<{bn}> for {vn}<({vn},), T> {{\n    type Output = {bn}<({bn},), T>;\n    /// The extension of a map on `{vn}` to `{bn}`, factor by factor with `{how}`.\n    #[inline]\n    fn outermorphism(self) -> {bn}<({bn},), T> {{\n        let t = self.c;\n{lets}        {bn} {{ c: [{}] }}\n    }}\n}}\n\n",
        rows.join(", ")
    );
    Some((code, (vn.clone(), bn.clone(), k == n)))
}
