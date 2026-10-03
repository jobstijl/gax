//! Multivectors with polynomial coefficients, for deriving fused kernels.

use crate::algebra::Algebra;
use crate::poly::{Poly, Rational, Var};
use crate::table::{BinOp, Layout, UnOp, blade_binop, blade_unop};
use std::collections::{BTreeMap, BTreeSet};

/// A multivector with polynomial coefficients, keyed by canonical blade mask.
pub type SymMv = BTreeMap<u32, Poly>;

/// A multivector of the given layout whose coefficients are the variables `first..first+n`.
pub fn variables(layout: &Layout, first: Var) -> SymMv {
    let coeffs: Vec<Poly> = (0..layout.len())
        .map(|i| Poly::var(first + i as Var))
        .collect();
    from_coeffs(layout, &coeffs)
}

/// A multivector from coefficients in a layout (applying orientation signs).
pub fn from_coeffs(layout: &Layout, coeffs: &[Poly]) -> SymMv {
    let mut mv = SymMv::new();
    for (&(m, s), c) in layout.blades.iter().zip(coeffs) {
        if !c.is_zero() {
            mv.insert(m, c.scale(Rational::int(i128::from(s))));
        }
    }
    mv
}

/// Coefficients of a multivector in a layout, or `None` if its support is not contained in it.
pub fn to_coeffs(layout: &Layout, mv: &SymMv) -> Option<Vec<Poly>> {
    for (m, p) in mv {
        if !p.is_zero() && layout.position(*m).is_none() {
            return None;
        }
    }
    Some(
        layout
            .blades
            .iter()
            .map(|&(m, s)| {
                mv.get(&m)
                    .map_or_else(Poly::zero, |p| p.scale(Rational::int(i128::from(s))))
            })
            .collect(),
    )
}

/// The blades with nonzero coefficients.
pub fn support(mv: &SymMv) -> BTreeSet<u32> {
    mv.iter()
        .filter(|(_, p)| !p.is_zero())
        .map(|(m, _)| *m)
        .collect()
}

/// A binary product.
pub fn binop(alg: &Algebra, op: BinOp, a: &SymMv, b: &SymMv) -> SymMv {
    let mut out = SymMv::new();
    for (&ma, pa) in a {
        for (&mb, pb) in b {
            let prod = pa * pb;
            for (m, c) in blade_binop(alg, op, ma, mb) {
                let e = out.entry(m).or_default();
                *e = &*e + &prod.scale(Rational::int(i128::from(c)));
            }
        }
    }
    out.retain(|_, p| !p.is_zero());
    out
}

/// A unary operation.
pub fn unop(alg: &Algebra, op: UnOp, a: &SymMv) -> SymMv {
    let mut out = SymMv::new();
    for (&m, p) in a {
        let (mm, s) = blade_unop(alg, op, m);
        out.insert(mm, p.scale(Rational::int(i128::from(s))));
    }
    out
}

/// Sum of two multivectors.
pub fn add(a: &SymMv, b: &SymMv) -> SymMv {
    let mut out = a.clone();
    for (&m, p) in b {
        let e = out.entry(m).or_default();
        *e = &*e + p;
    }
    out.retain(|_, p| !p.is_zero());
    out
}

/// A scalar multivector.
pub fn scalar(c: Rational) -> SymMv {
    let mut mv = SymMv::new();
    if !c.is_zero() {
        mv.insert(0, Poly::constant(c));
    }
    mv
}

/// The identities `x ~x = 1` of a unit versor, as polynomials that vanish.
pub fn unit_relations(alg: &Algebra, x: &SymMv) -> Vec<Poly> {
    let norm = binop(alg, BinOp::Gp, x, &unop(alg, UnOp::Reverse, x));
    let diff = add(&norm, &scalar(-Rational::ONE));
    diff.into_values().filter(|p| !p.is_zero()).collect()
}

/// The coefficient of `blade` (zero if absent).
pub fn coef(mv: &SymMv, blade: u32) -> Poly {
    mv.get(&blade).cloned().unwrap_or_default()
}

/// The multivector `p` times `blade` (empty for a zero `p`).
pub fn blade_mv(blade: u32, p: Poly) -> SymMv {
    let mut m = SymMv::new();
    if !p.is_zero() {
        m.insert(blade, p);
    }
    m
}

/// The scalar `p`.
pub fn scalar_mv(p: Poly) -> SymMv {
    blade_mv(0, p)
}

/// `mv` times the polynomial `p`.
pub fn scale_mv(mv: &SymMv, p: &Poly) -> SymMv {
    let mut out: SymMv = mv.iter().map(|(b, c)| (*b, c * p)).collect();
    out.retain(|_, c| !c.is_zero());
    out
}

/// The grade-`g` part of `mv`.
pub fn grade_part(mv: &SymMv, g: u32) -> SymMv {
    mv.iter()
        .filter(|(m, _)| m.count_ones() == g)
        .map(|(m, c)| (*m, c.clone()))
        .collect()
}

/// The square of a basis blade when it is `0` or `±1` (the Study structures), else `None`.
pub fn blade_square(alg: &Algebra, b: u32) -> Option<i8> {
    match alg.blade_product(b, b) {
        [] => Some(0),
        [(0, c)] if c.abs() == 1 => Some(*c as i8),
        _ => None,
    }
}
