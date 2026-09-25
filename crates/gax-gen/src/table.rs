//! Typed layouts and exact operation tables.
//!
//! A [`Layout`] is an ordered list of basis blades, each with an orientation sign, so
//! `e31` can be stored as `-e13`. An operation table between layouts lists the nonzero
//! terms `out[o] += c * a[i] * b[j]` with exact integer `c`, in output-first order.

use crate::algebra::{Algebra, Sparse};
use std::collections::BTreeSet;

/// An ordered list of oriented basis blades: coefficient `k` multiplies `sign_k * e_{mask_k}`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Layout {
    /// `(canonical blade mask, orientation sign)` per coefficient.
    pub blades: Vec<(u32, i64)>,
}

impl Layout {
    /// A layout from blade names such as `["e032", "e013", "e021", "e123"]`.
    ///
    /// # Errors
    /// For an invalid or repeated blade.
    pub fn parse(alg: &Algebra, names: &[&str]) -> Result<Layout, crate::algebra::AlgebraError> {
        let blades = names
            .iter()
            .map(|n| alg.parse_blade(n))
            .collect::<Result<Vec<_>, _>>()?;
        let mut seen = BTreeSet::new();
        for (m, _) in &blades {
            if !seen.insert(*m) {
                return Err(crate::algebra::AlgebraError(format!(
                    "layout repeats blade {}",
                    alg.blade_name(*m)
                )));
            }
        }
        Ok(Layout { blades })
    }

    /// Number of coefficients.
    pub fn len(&self) -> usize {
        self.blades.len()
    }

    /// Whether the layout is empty.
    pub fn is_empty(&self) -> bool {
        self.blades.is_empty()
    }

    /// The set of blade masks.
    pub fn support(&self) -> BTreeSet<u32> {
        self.blades.iter().map(|b| b.0).collect()
    }

    /// Position and orientation of a blade in this layout.
    pub fn position(&self, mask: u32) -> Option<(usize, i64)> {
        self.blades
            .iter()
            .position(|b| b.0 == mask)
            .map(|i| (i, self.blades[i].1))
    }

    /// Whether all coefficients have the given grade parity (for even/odd subalgebras).
    pub fn grades(&self) -> BTreeSet<u32> {
        self.blades.iter().map(|b| b.0.count_ones()).collect()
    }
}

/// The binary products the generator knows.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum BinOp {
    /// Geometric product.
    Gp,
    /// Outer (wedge) product, metric free.
    Wedge,
    /// Regressive (vee) product `J⁻¹(J(a) ^ J(b))` with `J` the right complement; metric free.
    Vee,
    /// Left contraction.
    Lc,
    /// Right contraction.
    Rc,
    /// Symmetric ("fat") inner product: grade `|ga - gb|` part of the geometric product.
    Dot,
    /// Scalar product: grade 0 part of the geometric product.
    Scalar,
    /// Commutator product `(ab - ba) / 2`.
    Commutator,
    /// Anticommutator product `(ab + ba) / 2`.
    Anticommutator,
}

impl BinOp {
    /// All binary operations.
    pub const ALL: [BinOp; 9] = [
        BinOp::Gp,
        BinOp::Wedge,
        BinOp::Vee,
        BinOp::Lc,
        BinOp::Rc,
        BinOp::Dot,
        BinOp::Scalar,
        BinOp::Commutator,
        BinOp::Anticommutator,
    ];

    /// Snake-case name used in generated code.
    pub fn name(self) -> &'static str {
        match self {
            BinOp::Gp => "gp",
            BinOp::Wedge => "wedge",
            BinOp::Vee => "vee",
            BinOp::Lc => "lc",
            BinOp::Rc => "rc",
            BinOp::Dot => "dot",
            BinOp::Scalar => "scalar",
            BinOp::Commutator => "commutator",
            BinOp::Anticommutator => "anticommutator",
        }
    }
}

/// The unary linear operations the generator knows.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum UnOp {
    /// Reverse `~a`.
    Reverse,
    /// Grade involution.
    Involute,
    /// Clifford conjugate.
    Conjugate,
    /// Right complement `J` (metric free); the dual used by the regressive product.
    Dual,
    /// Left complement, the inverse of [`UnOp::Dual`].
    Undual,
}

impl UnOp {
    /// All unary operations.
    pub const ALL: [UnOp; 5] = [
        UnOp::Reverse,
        UnOp::Involute,
        UnOp::Conjugate,
        UnOp::Dual,
        UnOp::Undual,
    ];

    /// Snake-case name used in generated code.
    pub fn name(self) -> &'static str {
        match self {
            UnOp::Reverse => "reverse",
            UnOp::Involute => "involute",
            UnOp::Conjugate => "conjugate",
            UnOp::Dual => "dual",
            UnOp::Undual => "undual",
        }
    }
}

/// Blade-level definition of a binary op: the sparse result of `op(e_a, e_b)`.
pub fn blade_binop(alg: &Algebra, op: BinOp, a: u32, b: u32) -> Sparse {
    let ga = a.count_ones();
    let gb = b.count_ones();
    let gp = || -> Sparse { alg.blade_product(a, b).iter().copied().collect() };
    let grade_part = |g: u32| -> Sparse {
        gp().into_iter()
            .filter(|(m, _)| m.count_ones() == g)
            .collect()
    };
    match op {
        BinOp::Gp => gp(),
        BinOp::Wedge => {
            let s = Algebra::wedge_sign(a, b);
            if s == 0 {
                Sparse::new()
            } else {
                Sparse::from([(a | b, s)])
            }
        }
        BinOp::Vee => {
            let (ja, sa) = alg.right_complement(a);
            let (jb, sb) = alg.right_complement(b);
            let s = Algebra::wedge_sign(ja, jb);
            if s == 0 {
                return Sparse::new();
            }
            let (m, sm) = alg.left_complement(ja | jb);
            Sparse::from([(m, sa * sb * s * sm)])
        }
        BinOp::Lc => {
            if ga > gb {
                Sparse::new()
            } else {
                grade_part(gb - ga)
            }
        }
        BinOp::Rc => {
            if gb > ga {
                Sparse::new()
            } else {
                grade_part(ga - gb)
            }
        }
        BinOp::Dot => grade_part(ga.abs_diff(gb)),
        BinOp::Scalar => grade_part(0),
        BinOp::Commutator | BinOp::Anticommutator => {
            let sign = if op == BinOp::Commutator { -1 } else { 1 };
            let mut out = gp();
            for &(m, c) in alg.blade_product(b, a) {
                *out.entry(m).or_default() += sign * c;
            }
            out.retain(|_, v| *v != 0);
            for v in out.values_mut() {
                assert!(*v % 2 == 0, "commutator coefficient not even");
                *v /= 2;
            }
            out
        }
    }
}

/// Blade-level definition of a unary op.
pub fn blade_unop(alg: &Algebra, op: UnOp, a: u32) -> (u32, i64) {
    let g = a.count_ones();
    match op {
        UnOp::Reverse => (a, Algebra::reverse_sign(g)),
        UnOp::Involute => (a, Algebra::involute_sign(g)),
        UnOp::Conjugate => (a, Algebra::conjugate_sign(g)),
        UnOp::Dual => alg.right_complement(a),
        UnOp::Undual => alg.left_complement(a),
    }
}

/// One term of a binary table: `out[o] += coef * a[i] * b[j]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Term2 {
    /// Output coefficient index.
    pub o: usize,
    /// Left input coefficient index.
    pub i: usize,
    /// Right input coefficient index.
    pub j: usize,
    /// Exact coefficient.
    pub coef: i64,
}

/// The support (set of output blades) of a binary op between two layouts.
pub fn binop_support(alg: &Algebra, op: BinOp, a: &Layout, b: &Layout) -> BTreeSet<u32> {
    let mut support = BTreeSet::new();
    // Accumulate per output blade over all input pairs so cancellations are respected:
    // a blade is in the support if some coefficient pair contributes to it. Contributions
    // from different pairs multiply different coefficient products, so they never cancel.
    for &(ma, _) in &a.blades {
        for &(mb, _) in &b.blades {
            for (m, c) in blade_binop(alg, op, ma, mb) {
                if c != 0 {
                    support.insert(m);
                }
            }
        }
    }
    support
}

/// The table of a binary op into a given output layout, which must contain the support.
pub fn binop_table(alg: &Algebra, op: BinOp, a: &Layout, b: &Layout, out: &Layout) -> Vec<Term2> {
    let mut terms = Vec::new();
    for (i, &(ma, sa)) in a.blades.iter().enumerate() {
        for (j, &(mb, sb)) in b.blades.iter().enumerate() {
            for (m, c) in blade_binop(alg, op, ma, mb) {
                let (o, so) = out
                    .position(m)
                    .unwrap_or_else(|| panic!("output layout lacks blade {}", alg.blade_name(m)));
                // a = sa e_ma, b = sb e_mb, out coefficient multiplies so e_m:
                // coef on e_m is sa sb c, so the coefficient of `out[o]` is sa sb c so.
                terms.push(Term2 {
                    o,
                    i,
                    j,
                    coef: sa * sb * c * so,
                });
            }
        }
    }
    terms.sort();
    terms
}

/// One term of a unary table: `out[o] += coef * a[i]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Term1 {
    /// Output coefficient index.
    pub o: usize,
    /// Input coefficient index.
    pub i: usize,
    /// Exact coefficient.
    pub coef: i64,
}

/// Support of a unary op on a layout.
pub fn unop_support(alg: &Algebra, op: UnOp, a: &Layout) -> BTreeSet<u32> {
    a.blades
        .iter()
        .map(|&(m, _)| blade_unop(alg, op, m).0)
        .collect()
}

/// Table of a unary op into an output layout containing its support.
pub fn unop_table(alg: &Algebra, op: UnOp, a: &Layout, out: &Layout) -> Vec<Term1> {
    let mut terms: Vec<Term1> = a
        .blades
        .iter()
        .enumerate()
        .map(|(i, &(ma, sa))| {
            let (m, c) = blade_unop(alg, op, ma);
            let (o, so) = out.position(m).expect("output layout lacks blade");
            Term1 {
                o,
                i,
                coef: sa * c * so,
            }
        })
        .collect();
    terms.sort();
    terms
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pga3d() -> Algebra {
        Algebra::diagonal("0123", &[0, 1, 1, 1]).unwrap()
    }

    #[test]
    fn vee_of_points_is_a_line_and_anticommutes_with_plane() {
        let a = pga3d();
        let e = |s: &str| a.parse_blade(s).unwrap().0;
        // e123 v e032 is a (ideal-free) line through origin and an ideal point: grade 2
        let r = blade_binop(&a, BinOp::Vee, e("e123"), e("e023"));
        assert_eq!(
            r.keys().map(|m| m.count_ones()).collect::<Vec<_>>(),
            vec![2]
        );
        // plane v point = -(point v plane) in 4 dimensions
        for p in [e("e1"), e("e0")] {
            for q in [e("e123"), e("e012"), e("e023")] {
                let x = blade_binop(&a, BinOp::Vee, p, q);
                let y = blade_binop(&a, BinOp::Vee, q, p);
                let neg: Sparse = y.into_iter().map(|(m, c)| (m, -c)).collect();
                assert_eq!(x, neg);
            }
        }
    }

    #[test]
    fn dual_undual_roundtrip() {
        let a = pga3d();
        for m in 0..16 {
            let (d, s) = blade_unop(&a, UnOp::Dual, m);
            let (u, t) = blade_unop(&a, UnOp::Undual, d);
            assert_eq!((u, s * t), (m, 1));
        }
    }

    #[test]
    fn layout_orientation_signs_enter_tables() {
        let a = pga3d();
        let lines = Layout::parse(&a, &["e01", "e02", "e03", "e12", "e31", "e23"]).unwrap();
        let scalar = Layout::parse(&a, &["1"]).unwrap();
        // e31 * e31 = -1: the orientation sign squares away
        let t = binop_table(&a, BinOp::Scalar, &lines, &lines, &scalar);
        let e31 = t.iter().find(|t| t.i == 4 && t.j == 4).unwrap();
        assert_eq!(e31.coef, -1);
    }
}
