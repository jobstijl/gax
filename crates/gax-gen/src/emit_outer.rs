//! Outermorphisms: the extension of a linear map on vectors (by `∧`) or antivectors (by `∨`)
//! to the kinds of one grade, derived symbolically as minors of the map's matrix, up to a size.

use crate::cse;
use crate::poly::{Poly, Rational, Var};
use crate::spec::{AlgebraSpec, KindSpec};
use crate::symbolic::{self, SymMv};
use crate::table::{BinOp, UnOp, blade_unop};
use std::collections::BTreeSet;

/// The largest outermorphism derived, in terms of its minors ([`terms`]). Every outermorphism
/// of the standard algebras and of the algebras gax's tests and examples declare is at most
/// 7350 terms (7D trivectors), and none of up to 9 dimensions lies between that and 18 816
/// (8D trivectors, which add 2 GB to a build); 8D's 4-vectors, 117 600 terms, ran a build out
/// of 10 GB (ADR-034). Larger extensions are not emitted: such a map is built from its images
/// instead (`from_images`).
pub const OUTER_MAX_TERMS: usize = 1 << 14;

/// The terms of the outermorphism of a map on `n` (anti)vectors to the blades of `k` factors:
/// `C(n, k)²` entries, each a minor of size `k` with `k!` terms.
#[must_use]
pub fn terms(n: u32, k: u32) -> usize {
    let blades = (0..k).fold(1, |c, i| c * (n - i) as usize / (i + 1) as usize);
    let minor: usize = (1..=k as usize).product();
    blades * blades * minor
}

/// The grade of a kind whose blades are all of one grade.
fn single_grade(k: &KindSpec) -> Option<u32> {
    let g: BTreeSet<u32> = k.layout.grades();
    g.iter().next().copied().filter(|_| g.len() == 1)
}

/// The number of `k`-subsets of `n` things.
fn choose(n: usize, k: usize) -> usize {
    (0..k).fold(1, |c, i| c * (n - i) / (i + 1))
}

/// Emit `Outermorphism<B>` impls for maps on the kinds of grade 1 and grade n-1, up to
/// [`OUTER_MAX_TERMS`], and list them as `(base kind, extended kind, whether the extended kind
/// is the top)`. A base holds all (anti)vectors, or some of them, as a `part` does (PGA3D's
/// `OriginPlane` and `Direction`): its extensions are then the kinds of the subalgebra those
/// factors span. The pairs are independent, so they are derived in parallel and written in
/// order.
pub fn outermorphisms(spec: &AlgebraSpec) -> (String, Vec<(String, String, bool)>) {
    let n = spec.algebra.dim() as u32;
    let bases = spec.kinds.iter().filter(|v| {
        v.layout.len() >= 2 && single_grade(v).is_some_and(|gv| gv == 1 || (gv + 1 == n && n > 2))
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
    done.into_iter().flatten().unzip()
}

/// The outermorphism from maps on the base kind `v` to the kind `b`, if `b` extends it (holds
/// every product of that many of `v`'s factors, as every extension of a general map fills
/// it) within [`OUTER_MAX_TERMS`].
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
    if b.name == v.name || (wedge && gb < 1) || (!wedge && gb > n - 1) {
        return None;
    }
    // The number of vector factors of a blade of B.
    let k = if wedge { gb } else { n - gb };
    // One factor: B is another kind of V's grade, not an extension. And B holds every product
    // of k of V's factors (each of its blades factoring into V's is checked below).
    if k == 1 || b.layout.len() != choose(nv, k as usize) || terms(nv as u32, k) > OUTER_MAX_TERMS {
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
        for i in (0..n).filter(|i| factors & (1 << i) != 0) {
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
    let prog = cse::compile_plain(&polys);
    let name = |var: Var| format!("t[{}][{}]", var as usize / nv, var as usize % nv);
    let (lets, outs) = prog.lets_and_outputs(&name, "m");
    let rows: Vec<String> = outs
        .chunks(nb)
        .map(|row| format!("[{}]", row.join(", ")))
        .collect();
    let (vn, bn) = (&v.name, &b.name);
    let how = if wedge { "∧" } else { "∨" };
    // The orientation of `k` factors of orientation `O` (ADR-051): their wedge `O^k`; their
    // join flips once per `&`, `O^k (−1)^(k−1)`. `B`'s own orientation in the trait is ignored,
    // as a cast's is: `t.outermorphism::<Point>()` names the kind.
    let oo = match (wedge, k % 2 == 0) {
        (true, true) => "gx::Extrinsic",
        (false, true) => "gx::Intrinsic",
        (_, false) => "O",
    };
    let code = format!(
        "impl<T: gx::Coef, O: gx::Orientation, X: gx::Orientation> gx::Outermorphism<{bn}<(), f64, X>> for {vn}<({vn}<(), f64, O>,), T, O> {{\n    type Output = {bn}<({bn}<(), f64, {oo}>,), T, {oo}>;\n    /// The extension of a map on `{vn}` to `{bn}`, factor by factor with `{how}`.\n    #[inline]\n    fn outermorphism(self) -> Self::Output {{\n        let t = self.c;\n{lets}        {}\n    }}\n}}\n\n",
        crate::emit::kind_literal(bn, &format!("[{}]", rows.join(", ")))
    );
    Some((code, (vn.clone(), bn.clone(), k as usize == nv)))
}

#[cfg(test)]
mod tests {
    use super::{OUTER_MAX_TERMS, terms};

    #[test]
    fn the_limit_falls_between_the_extensions_declared_and_the_next() {
        // 7D trivectors, the largest declared; 8D trivectors and 4-vectors, the next and the
        // Hubbard ring's.
        assert_eq!(terms(7, 3), 7350);
        assert_eq!(terms(8, 3), 18_816);
        assert_eq!(terms(8, 4), 117_600);
        // Every extension of up to 9 dimensions is on one side of the gap.
        for n in 2..=9 {
            for k in 2..=n {
                let t = terms(n, k);
                assert!(t <= 7350 || t >= 18_816, "{n}D, {k} factors: {t}");
            }
        }
        assert!((7350..18_816).contains(&OUTER_MAX_TERMS));
    }
}
