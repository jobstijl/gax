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

/// The largest dimension whose correlations (maps between vectors and antivectors) get
/// outermorphisms. numga's uses are in three and four dimensions (polarities, PGA2D's
/// stiffnesses, fresnel's `volumes` in STA); CSTA's (6D) would add 19 k generated lines (15 %) for
/// none, and an `algebra!` of up to 9 dimensions more. Above it, the extension is built from its
/// images (`from_images`).
pub const CORRELATION_MAX_DIM: usize = 5;

/// The terms of the outermorphism of a map on `n` (anti)vectors to the blades of `k` factors:
/// `C(n, k)²` entries, each a minor of size `k` with `k!` terms.
#[must_use]
pub fn terms(n: u32, k: u32) -> usize {
    if k > n {
        return 0;
    }
    let blades = (0..k).fold(1, |c, i| c * (n - i) as usize / (i + 1) as usize);
    let minor: usize = (1..=k as usize).product();
    blades * blades * minor
}

/// The grade of a kind whose blades are all of one grade.
fn single_grade(k: &KindSpec) -> Option<u32> {
    let g: BTreeSet<u32> = k.layout.grades();
    g.iter().next().copied().filter(|_| g.len() == 1)
}

/// The number of `k`-subsets of `n` things (none when `k > n`: a part of fewer factors than a
/// grade has no blades of that grade).
fn choose(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    (0..k).fold(1, |c, i| c * (n - i) / (i + 1))
}

/// One derived extension: `(input base, output base, extended input kind, extended output
/// kind, whether the extended input kind is the top)`.
type Derived = (String, String, String, String, bool);

/// An extension of a correlation, a map between vectors and antivectors (either way): `(input
/// base, output base, extended input kind, extended output kind)`.
pub type CorrelationOuter = (String, String, String, String);

/// What [`outermorphisms`] derives: the impls, the extensions of maps on one base kind (`(base
/// kind, extended kind, whether the extended kind is the top)`), and those of correlations.
pub type Outers = (String, Vec<(String, String, bool)>, Vec<CorrelationOuter>);

/// Emit `Outermorphism<B>` impls for maps on the kinds of grade 1 and grade n-1, up to
/// [`OUTER_MAX_TERMS`], and for correlations between them (maps from vectors to antivectors or
/// back, numga's `C(p & q) = C(p) ^ C(q)`). A base holds all (anti)vectors, or some of them, as
/// a `part` does (PGA3D's `OriginPlane` and `Direction`): its extensions are then the kinds of
/// the subalgebra those factors span. The pairs are independent, so they are derived in
/// parallel and written in order.
pub fn outermorphisms(spec: &AlgebraSpec) -> Outers {
    let n = spec.algebra.dim() as u32;
    let bases: Vec<&KindSpec> = spec
        .kinds
        .iter()
        .filter(|v| {
            v.layout.len() >= 2
                && single_grade(v).is_some_and(|gv| gv == 1 || (gv + 1 == n && n > 2))
        })
        .collect();
    // A map on one base, or a correlation between bases of the two grades (up to
    // CORRELATION_MAX_DIM dimensions).
    let maps: Vec<(&KindSpec, &KindSpec)> = bases
        .iter()
        .flat_map(|&v| bases.iter().map(move |&w| (v, w)))
        .filter(|(v, w)| {
            v.name == w.name
                || (single_grade(v) != single_grade(w) && n as usize <= CORRELATION_MAX_DIM)
        })
        .collect();
    let jobs: Vec<(&KindSpec, &KindSpec, &KindSpec)> = maps
        .iter()
        .flat_map(|&(v, w)| spec.kinds.iter().map(move |b| (v, w, b)))
        .collect();
    let done = crate::par::map_heaviest_first(
        &jobs,
        |(v, w, b)| v.layout.len() * w.layout.len() * b.layout.len(),
        |&(v, w, b)| {
            crate::par::timed(
                || {
                    format!(
                        "{} outermorphism {} -> {} on {}",
                        spec.name, v.name, w.name, b.name
                    )
                },
                || outermorphism(spec, v, w, b),
            )
        },
    );
    let mut code = String::new();
    let (mut endo, mut corr) = (Vec::new(), Vec::new());
    for (c, (v, w, b, bout, top)) in done.into_iter().flatten() {
        code.push_str(&c);
        if v == w {
            endo.push((v, b, top));
        } else {
            corr.push((v, w, b, bout));
        }
    }
    (code, endo, corr)
}

/// The number of factors `k` of an extension of maps `w <- v` to `b`, if `b` extends `v` (holds
/// every product of `k ≥ 2` of its factors) within [`OUTER_MAX_TERMS`].
fn outermorphism_shape(
    spec: &AlgebraSpec,
    v: &KindSpec,
    w: &KindSpec,
    b: &KindSpec,
) -> Option<u32> {
    let n = spec.algebra.dim() as u32;
    let (gv, _) = (single_grade(v)?, single_grade(w)?);
    let wedge = gv == 1;
    let gb = single_grade(b)?;
    if b.name == v.name || (wedge && gb < 1) || (!wedge && gb > n - 1) {
        return None;
    }
    // The number of vector factors of a blade of B.
    let k = if wedge { gb } else { n - gb };
    // One factor: B is another kind of V's grade, not an extension. And B holds every product
    // of k of V's factors (each of its blades factoring into V's is checked below).
    let size = v.layout.len().max(w.layout.len()) as u32;
    if k == 1
        || b.layout.len() != choose(v.layout.len(), k as usize)
        || k as usize > w.layout.len()
        || terms(size, k) > OUTER_MAX_TERMS
    {
        return None;
    }
    Some(k)
}

/// The orientation of the product of `k` factors of orientation `o` (ADR-051): their wedge
/// `O^k`; their join flips once per `&`, `O^k (−1)^(k−1)`.
fn factors_orientation(wedge: bool, k: u32, o: &str) -> String {
    match (wedge, k.is_multiple_of(2)) {
        (true, true) => "gx::Extrinsic".into(),
        (false, true) => "gx::Intrinsic".into(),
        (_, false) => o.into(),
    }
}

/// The outermorphism from maps `w <- v` to the kind `b`, if `b` extends `v` (holds every product
/// of that many of `v`'s factors, as every extension of a general map fills it) within
/// [`OUTER_MAX_TERMS`], and the images' products of `k` factors of `w` fill a declared kind.
/// Each side multiplies by its own exterior product, `∧` for vectors and `∨` for antivectors:
/// for `w == v` the extension of a map on one kind, else of a correlation. Returns the impl and
/// `(v, w, b, the extended output kind, whether b is the top)`.
#[allow(clippy::too_many_lines)]
fn outermorphism(
    spec: &AlgebraSpec,
    v: &KindSpec,
    w: &KindSpec,
    b: &KindSpec,
) -> Option<(String, Derived)> {
    let alg = &spec.algebra;
    let n = alg.dim() as u32;
    let k = outermorphism_shape(spec, v, w, b)?;
    let wedge = single_grade(v)? == 1;
    let wedge_out = single_grade(w)? == 1;
    let (nv, nw) = (v.layout.len(), w.layout.len());
    // The map's entries t[o][i] as variables o * nv + i.
    let image = |layout_pos: usize| -> SymMv {
        let coeffs: Vec<Poly> = (0..nw)
            .map(|o| Poly::var((o * nv + layout_pos) as Var))
            .collect();
        symbolic::from_coeffs(&w.layout, &coeffs)
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
    let mut columns: Vec<SymMv> = Vec::new();
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
                    if wedge_out { BinOp::Wedge } else { BinOp::Vee },
                    &acc,
                    &img,
                ),
            });
        }
        let col = col.unwrap_or_else(|| symbolic::scalar(Rational::ONE));
        columns.push(
            col.into_iter()
                .map(|(bb, c)| (bb, c.scale(Rational::int(i128::from(sa * r)))))
                .collect(),
        );
    }
    // The extended output kind: the declared kind of one grade holding every product of k of
    // W's factors (B itself for a map on one kind).
    let bout = if v.name == w.name {
        b
    } else {
        let support: BTreeSet<u32> = columns.iter().flat_map(|c| c.keys().copied()).collect();
        let grade = if wedge_out { k } else { n - k };
        spec.kinds.iter().find(|c| {
            single_grade(c) == Some(grade)
                && c.layout.len() == choose(nw, k as usize)
                && support.is_subset(&c.layout.support())
        })?
    };
    let columns: Vec<Vec<Poly>> = columns
        .iter()
        .map(|c| symbolic::to_coeffs(&bout.layout, c))
        .collect::<Option<_>>()?;
    let nb = bout.layout.len();
    let polys: Vec<Poly> = (0..nb)
        .flat_map(|o| columns.iter().map(move |c| c[o].clone()))
        .collect();
    let prog = cse::compile_plain(&polys);
    let name = |var: Var| format!("t[{}][{}]", var as usize / nv, var as usize % nv);
    let (lets, outs) = prog.lets_and_outputs(&name, "m");
    let rows: Vec<String> = outs
        .chunks(b.layout.len())
        .map(|row| format!("[{}]", row.join(", ")))
        .collect();
    let (vn, wn, bn, on) = (&v.name, &w.name, &b.name, &bout.name);
    let how = |wedge: bool| if wedge { "∧" } else { "∨" };
    // `B`'s own orientation in the trait is ignored, as a cast's is: `t.outermorphism::<Point>()`
    // names the kind.
    let body = format!(
        "    #[inline]\n    fn outermorphism(self) -> Self::Output {{\n        let t = self.c;\n{lets}        {}\n    }}\n}}\n\n",
        crate::emit::kind_literal(on, &format!("[{}]", rows.join(", ")))
    );
    let code = if v.name == w.name {
        let oo = factors_orientation(wedge, k, "O");
        format!(
            "impl<T: gx::Coef, O: gx::Orientation, X: gx::Orientation> gx::Outermorphism<{bn}<(), f64, X>> for {vn}<({vn}<(), f64, O>,), T, O> {{\n    type Output = {bn}<({bn}<(), f64, {oo}>,), T, {oo}>;\n    /// The extension of a map on `{vn}` to `{bn}`, factor by factor with `{}`.\n{body}",
            how(wedge)
        )
    } else {
        let oi = factors_orientation(wedge, k, "OV");
        let oo = factors_orientation(wedge_out, k, "OW");
        format!(
            "impl<T: gx::Coef, OV: gx::Orientation, OW: gx::Orientation, X: gx::Orientation> gx::Outermorphism<{bn}<(), f64, X>> for {wn}<({vn}<(), f64, OV>,), T, OW> {{\n    type Output = {on}<({bn}<(), f64, {oi}>,), T, {oo}>;\n    /// The extension of a correlation `{wn} <- {vn}` to `{bn}`: the input's factors by `{}`, their\n    /// images by `{}`.\n{body}",
            how(wedge),
            how(wedge_out)
        )
    };
    Some((
        code,
        (
            vn.clone(),
            wn.clone(),
            bn.clone(),
            on.clone(),
            k as usize == nv,
        ),
    ))
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

    #[test]
    fn a_part_of_fewer_factors_than_a_grade_has_no_extension_to_it() {
        // A 5D part of two antivectors against the trivectors: no blades, no terms (this
        // subtracted past zero, a panic in debug builds of the proc macro).
        assert_eq!(super::choose(2, 3), 0);
        assert_eq!(terms(2, 3), 0);
        assert_eq!(super::choose(5, 2), 10);
    }
}
