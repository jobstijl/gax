//! Tier-2 methods of values, derived symbolically per kind: norms, inverse, normalization,
//! exponential and logarithm, square root.
//!
//! Each method is emitted only when the kind has the structure its closed form needs, which
//! the generator checks symbolically: for instance `x ~x` must be a Study number `a + b I`
//! (a scalar plus one blade whose square is a scalar) for the inverse and the
//! normalization. A kind without that structure simply has no such method, a compile-time
//! error rather than a wrong result.

use crate::algebra::Algebra;
use crate::cse::{self, Stage, StageOp};
use crate::poly::{Poly, Rational, Var};
use crate::slp::{Func, render};
use crate::spec::{AlgebraSpec, KindSpec};
use crate::symbolic::{self, SymMv};
use crate::table::{BinOp, UnOp};
use std::collections::BTreeSet;
use std::fmt::Write as _;

/// A Study number structure `a + b B` of a symbolic multivector: `B` and `B² = isq`.
struct Study {
    blade: Option<u32>,
    isq: i8,
}

/// If `mv` lies in `span{1, B}` for a single blade `B` whose square is a scalar, return it.
fn study_structure(alg: &Algebra, mv: &SymMv) -> Option<Study> {
    let others: Vec<u32> = symbolic::support(mv)
        .into_iter()
        .filter(|&m| m != 0)
        .collect();
    match others.as_slice() {
        [] => Some(Study {
            blade: None,
            isq: -1,
        }),
        [b] => {
            let sq = alg.blade_product(*b, *b);
            match sq {
                [] => Some(Study {
                    blade: Some(*b),
                    isq: 0,
                }),
                [(0, c)] if c.abs() == 1 => Some(Study {
                    blade: Some(*b),
                    isq: *c as i8,
                }),
                _ => None,
            }
        }
        _ => None,
    }
}

/// Whether `p` is minus a sum of squares of single variables, hence never positive.
fn nonpositive(p: &Poly) -> bool {
    !p.is_zero()
        && p.0
            .iter()
            .all(|(m, c)| *c < Rational::ZERO && m.0.len() == 2 && m.0[0] == m.0[1])
}

fn coef(mv: &SymMv, blade: u32) -> Poly {
    mv.get(&blade).cloned().unwrap_or_default()
}

fn blade_mv(blade: u32, p: Poly) -> SymMv {
    let mut m = SymMv::new();
    if !p.is_zero() {
        m.insert(blade, p);
    }
    m
}

fn scale_mv(mv: &SymMv, p: &Poly) -> SymMv {
    let mut out: SymMv = mv.iter().map(|(b, c)| (*b, c * p)).collect();
    out.retain(|_, c| !c.is_zero());
    out
}

/// Which value methods were emitted for a kind, and their output kinds.
#[derive(Clone, Debug, Default)]
pub struct ValueMethods {
    /// Kind name.
    pub kind: String,
    /// `norm_squared` and `norm`.
    pub norm: bool,
    /// `inverse`, with its output kind.
    pub inverse: Option<String>,
    /// `normalized`.
    pub normalized: bool,
    /// `exp`, with its output kind.
    pub exp: Option<String>,
    /// `Unit::log`, with its output kind.
    pub log: Option<String>,
    /// `sqrt`.
    pub sqrt: bool,
}

/// Emit the value methods of kind `k`, returning the code of an inherent impl block (and
/// trait impls) for `K<(), T>`, and what was emitted.
pub fn value_methods(spec: &AlgebraSpec, k: &KindSpec) -> (String, ValueMethods) {
    let mut meta = ValueMethods {
        kind: k.name.clone(),
        ..ValueMethods::default()
    };
    let alg = &spec.algebra;
    let n = k.layout.len();
    let nv = n as Var;
    let name = &k.name;
    let x = symbolic::variables(&k.layout, 0);
    let rev = symbolic::unop(alg, UnOp::Reverse, &x);
    let norm = symbolic::binop(alg, BinOp::Gp, &x, &rev);
    let xvar = |v: Var| format!("x[{v}]");
    let mut body = String::new();
    let mut traits = String::new();

    // norm_squared / norm: the scalar part of x ~x.
    let n0 = coef(&norm, 0);
    meta.norm = !n0.is_zero();
    if !n0.is_zero() {
        let prog = cse::compile_best(std::slice::from_ref(&n0), &BTreeSet::new(), &[]);
        let mut lets = String::new();
        prog.emit_lets(&xvar, "t", &mut lets);
        let _ = write!(
            body,
            "    /// The squared norm: the scalar part of `x ~x`.\n    #[inline]\n    pub fn norm_squared(self) -> T {{\n        let x = self.c;\n{lets}        {}\n    }}\n\n    /// The norm, `sqrt(|norm_squared|)`.\n    #[inline]\n    pub fn norm(self) -> T {{\n        self.norm_squared().abs().sqrt()\n    }}\n\n",
            render(&prog.outputs[0], &xvar, "t")
        );
    }

    if let Some(study) = study_structure(alg, &norm)
        && !n0.is_zero()
    {
        meta.inverse = emit_inverse(spec, k, &rev, &norm, &study, &mut body);
        meta.normalized = emit_normalized(spec, k, &x, &norm, &study, &mut body);
    }

    // exp: for kinds made of bivectors, when B² is a Study number.
    let all_grade2 = k.layout.blades.iter().all(|(m, _)| m.count_ones() == 2);
    if all_grade2 {
        let bsq = symbolic::binop(alg, BinOp::Gp, &x, &x);
        if let Some(study) = study_structure(alg, &bsq) {
            let (c0, c1, s0, s1) = (nv, nv + 1, nv + 2, nv + 3);
            let one = scalar_mv(Poly::var(c0));
            let mut out = symbolic::add(&one, &scale_mv(&x, &Poly::var(s0)));
            if let Some(b) = study.blade {
                let bmv = blade_mv(b, Poly::var(c1));
                out = symbolic::add(&out, &bmv);
                let bx = symbolic::binop(alg, BinOp::Gp, &blade_mv(b, Poly::var(s1)), &x);
                out = symbolic::add(&out, &bx);
            }
            if let Some(ok) = spec.kind_for_support(&symbolic::support(&out)) {
                let out_kind = ok.clone();
                let coeffs = symbolic::to_coeffs(&out_kind.layout, &out).expect("fits");
                let lam = coef(&bsq, 0);
                let mu = study.blade.map_or_else(Poly::zero, |b| coef(&bsq, b));
                let pre = cse::compile_best(&[lam, mu], &BTreeSet::new(), &[]);
                let prog = cse::compile_best(&coeffs, &BTreeSet::new(), &[]);
                let names = move |v: Var| match v {
                    v if v < nv => format!("x[{v}]"),
                    v if v == c0 => "c0".into(),
                    v if v == c1 => "c1".into(),
                    v if v == s0 => "s0".into(),
                    _ => "s1".into(),
                };
                let mut lets = String::new();
                pre.emit_lets(&xvar, "p", &mut lets);
                let lam_e = render(&pre.outputs[0], &xvar, "p");
                let mu_e = render(&pre.outputs[1], &xvar, "p");
                if nonpositive(&coef(&bsq, 0)) && (study.isq == 0 || study.blade.is_none()) {
                    // A rotation (the scalar part of B² is minus a sum of squares): real trig.
                    let _ = writeln!(
                        lets,
                        "        let [c0, c1, s0, s1] = gx::study::exp_coeffs_rotation({lam_e}, {mu_e});"
                    );
                } else {
                    let _ = writeln!(
                        lets,
                        "        let [c0, c1, s0, s1] = gx::study::exp_coeffs({}, {lam_e}, {mu_e});",
                        study.isq
                    );
                }
                prog.emit_lets(&names, "t", &mut lets);
                let outs: Vec<String> = prog
                    .outputs
                    .iter()
                    .map(|o| render(o, &names, "t"))
                    .collect();
                let on = &out_kind.name;
                meta.exp = Some(on.clone());
                let _ = write!(
                    body,
                    "    /// The exponential, a unit versor: `exp(B) = C(B²) + S(B²) B` with `B²` a Study number.\n    #[inline]\n    #[allow(unused_variables)]\n    pub fn exp(self) -> gx::Unit<{on}<(), T>> {{\n        let x = self.c;\n{lets}        gx::Unit::new_unchecked({on}::from_coeffs([{}]))\n    }}\n\n",
                    outs.join(", ")
                );
            }
        }
    }

    // exp in 5D: B² = λ + Q with a 4-vector Q (several blades) whose square is a scalar.
    if all_grade2 && meta.exp.is_none() {
        meta.exp = emit_exp_general(spec, k, &x, &mut body);
    }

    // log: for unit versors whose parts are a Study number and a bivector.
    meta.log = emit_log(spec, k, &x, &mut traits);
    if meta.log.is_none() {
        meta.log = emit_log_general(spec, k, &x, &mut traits);
    }

    // sqrt of a unit versor: normalize(1 + R).
    let scalar_idx = k.layout.position(0);
    if let (Some((si, ss)), true) = (scalar_idx, meta.normalized) {
        meta.sqrt = true;
        let _ = write!(
            body,
            "    /// The principal square root of a unit versor, `normalize(1 + R)` (not defined for `R = -1`).\n    #[inline]\n    pub fn sqrt(self) -> gx::Unit<Self> {{\n        let mut c = self.c;\n        c[{si}] = c[{si}] {} T::one();\n        {name}::from_coeffs(c).normalized()\n    }}\n\n",
            if ss > 0 { "+" } else { "-" }
        );
    }

    let mut out = String::new();
    if !body.is_empty() {
        let _ = write!(out, "impl<T: gx::Real> {name}<(), T> {{\n{body}}}\n\n");
    }
    out.push_str(&traits);
    (out, meta)
}

fn scalar_mv(p: Poly) -> SymMv {
    blade_mv(0, p)
}

fn emit_inverse(
    spec: &AlgebraSpec,
    k: &KindSpec,
    rev: &SymMv,
    norm: &SymMv,
    study: &Study,
    body: &mut String,
) -> Option<String> {
    let alg = &spec.algebra;
    let nv = k.layout.len() as Var;
    let n0 = coef(norm, 0);
    let nb = study.blade.map_or_else(Poly::zero, |b| coef(norm, b));
    // (a + bB)⁻¹ = (a - bB) / (a² - isq b²)
    let mut conj = scalar_mv(n0.clone());
    if let Some(b) = study.blade {
        conj = symbolic::add(&conj, &blade_mv(b, -&nb));
    }
    let num = symbolic::binop(alg, BinOp::Gp, rev, &conj);
    let out_kind = spec.kind_for_support(&symbolic::support(&num))?.clone();
    let d = &(&n0 * &n0) - &(&nb * &nb).scale(Rational::int(i128::from(study.isq)));
    let r = nv;
    let coeffs: Vec<Poly> = symbolic::to_coeffs(&out_kind.layout, &num)
        .expect("fits")
        .iter()
        .map(|c| c * &Poly::var(r))
        .collect();
    let stages = vec![Stage {
        var: r,
        op: StageOp::Call(Func::Recip),
        args: vec![d.clone()],
    }];
    let relations = vec![&(&d * &Poly::var(r)) - &Poly::constant(Rational::ONE)];
    let (prog, _) = cse::compile_staged_best(&coeffs, &stages, &relations);
    let xvar = |v: Var| format!("x[{v}]");
    let mut lets = String::new();
    prog.emit_lets(&xvar, "t", &mut lets);
    let outs: Vec<String> = prog.outputs.iter().map(|o| render(o, &xvar, "t")).collect();
    let on = &out_kind.name;
    let _ = write!(
        body,
        "    /// The inverse under the geometric product, `~x (x ~x)⁻¹` ({}).\n    #[inline]\n    pub fn inverse(self) -> {on}<(), T> {{\n        let x = self.c;\n{lets}        {on}::from_coeffs([{}])\n    }}\n\n",
        prog.cost(),
        outs.join(", ")
    );
    Some(on.clone())
}

fn emit_normalized(
    spec: &AlgebraSpec,
    k: &KindSpec,
    x: &SymMv,
    norm: &SymMv,
    study: &Study,
    body: &mut String,
) -> bool {
    let alg = &spec.algebra;
    let nv = k.layout.len() as Var;
    let (s0, s1) = (nv, nv + 1);
    // (x ~x)^(-1/2) x, with the Study factor on the left.
    let mut s = scalar_mv(Poly::var(s0));
    if let Some(b) = study.blade {
        s = symbolic::add(&s, &blade_mv(b, Poly::var(s1)));
    }
    let out = symbolic::binop(alg, BinOp::Gp, &s, x);
    let Some(coeffs) = symbolic::to_coeffs(&k.layout, &out) else {
        return false;
    };
    let n0 = coef(norm, 0);
    let nb = study.blade.map_or_else(Poly::zero, |b| coef(norm, b));
    let pre = cse::compile_best(&[n0, nb], &BTreeSet::new(), &[]);
    let prog = cse::compile_best(&coeffs, &BTreeSet::new(), &[]);
    let xvar = |v: Var| format!("x[{v}]");
    let names = move |v: Var| match v {
        v if v < nv => format!("x[{v}]"),
        v if v == s0 => "s0".into(),
        _ => "s1".into(),
    };
    let mut lets = String::new();
    pre.emit_lets(&xvar, "p", &mut lets);
    let a = render(&pre.outputs[0], &xvar, "p");
    if study.blade.is_some() {
        let b = render(&pre.outputs[1], &xvar, "p");
        if study.isq == 0 {
            let _ = writeln!(
                lets,
                "        let [s0, s1] = gx::study::rsqrt_nil({a}, {b});"
            );
        } else {
            let _ = writeln!(
                lets,
                "        let [s0, s1] = gx::study::rsqrt({}, {a}, {b});",
                study.isq
            );
        }
    } else {
        let _ = writeln!(lets, "        let s0 = ({a}).abs().sqrt().recip();");
    }
    prog.emit_lets(&names, "t", &mut lets);
    let outs: Vec<String> = prog
        .outputs
        .iter()
        .map(|o| render(o, &names, "t"))
        .collect();
    let name = &k.name;
    let _ = write!(
        body,
        "    /// Scaled to a unit versor, `(x ~x)^(-1/2) x`, so that `x ~x = 1` (`±1` when the norm is negative).\n    #[inline]\n    pub fn normalized(self) -> gx::Unit<Self> {{\n        let x = self.c;\n{lets}        gx::Unit::new_unchecked({name}::from_coeffs([{}]))\n    }}\n\n",
        outs.join(", ")
    );
    true
}

fn emit_log(spec: &AlgebraSpec, k: &KindSpec, x: &SymMv, traits: &mut String) -> Option<String> {
    let alg = &spec.algebra;
    let nv = k.layout.len() as Var;
    // Parts: scalar, bivector P, and at most one grade-4 blade.
    k.layout.position(0)?;
    let grades: BTreeSet<u32> = k
        .layout
        .blades
        .iter()
        .map(|(m, _)| m.count_ones())
        .collect();
    if !grades.contains(&2) || grades.iter().any(|g| ![0, 2, 4].contains(g)) {
        return None;
    }
    let four: Vec<u32> = k
        .layout
        .blades
        .iter()
        .map(|(m, _)| *m)
        .filter(|m| m.count_ones() == 4)
        .collect();
    if four.len() > 1 {
        return None;
    }
    let p: SymMv = x
        .iter()
        .filter(|(m, _)| m.count_ones() == 2)
        .map(|(m, c)| (*m, c.clone()))
        .collect();
    let u = symbolic::binop(alg, BinOp::Gp, &p, &p);
    let study = study_structure(alg, &u)?;
    if let (Some(b), Some(&f)) = (study.blade, four.first())
        && b != f
    {
        return None;
    }
    let blade = study.blade.or(four.first().copied());
    let isq = match blade {
        Some(b) => match alg.blade_product(b, b) {
            [] => 0,
            [(0, c)] if c.abs() == 1 => *c as i8,
            _ => return None,
        },
        None => -1,
    };
    let (h0, h1) = (nv, nv + 1);
    let mut out = scale_mv(&p, &Poly::var(h0));
    if let Some(b) = blade {
        out = symbolic::add(
            &out,
            &symbolic::binop(alg, BinOp::Gp, &blade_mv(b, Poly::var(h1)), &p),
        );
    }
    let out_kind = spec.kind_for_support(&symbolic::support(&out))?.clone();
    let coeffs = symbolic::to_coeffs(&out_kind.layout, &out).expect("fits");
    let c0 = coef(x, 0);
    let cb = blade.map_or_else(Poly::zero, |b| coef(x, b));
    let u0 = coef(&u, 0);
    let ub = blade.map_or_else(Poly::zero, |b| coef(&u, b));
    let pre = cse::compile_best(&[c0, cb, u0, ub], &BTreeSet::new(), &[]);
    let prog = cse::compile_best(&coeffs, &BTreeSet::new(), &[]);
    let xvar = |v: Var| format!("x[{v}]");
    let names = move |v: Var| match v {
        v if v < nv => format!("x[{v}]"),
        v if v == h0 => "h0".into(),
        _ => "h1".into(),
    };
    let mut lets = String::new();
    pre.emit_lets(&xvar, "p", &mut lets);
    let r: Vec<String> = pre.outputs.iter().map(|o| render(o, &xvar, "p")).collect();
    if nonpositive(&coef(&u, 0)) && (isq == 0 || blade.is_none()) {
        let _ = writeln!(
            lets,
            "        let [h0, h1] = gx::study::log_coeffs_rotation(({}, {}), ({}, {}));",
            r[0], r[1], r[2], r[3]
        );
    } else {
        let _ = writeln!(
            lets,
            "        let [h0, h1] = gx::study::log_coeffs({isq}, ({}, {}), ({}, {}));",
            r[0], r[1], r[2], r[3]
        );
    }
    prog.emit_lets(&names, "t", &mut lets);
    let outs: Vec<String> = prog
        .outputs
        .iter()
        .map(|o| render(o, &names, "t"))
        .collect();
    let (name, on) = (&k.name, &out_kind.name);
    let _ = write!(
        traits,
        "impl<T: gx::Real> gx::Log<{on}<(), T>> for gx::Unit<{name}<(), T>> {{\n    /// The logarithm of a unit versor: the bivector `B` with `B.exp() == self`.\n    #[inline]\n    #[allow(unused_variables)]\n    fn log(self) -> {on}<(), T> {{\n        let x = self.into_inner().c;\n{lets}        {on}::from_coeffs([{}])\n    }}\n}}\n\n",
        outs.join(", ")
    );
    Some(on.clone())
}

/// The grade-4 part of `mv` if everything else is scalar, it has several blades, and its
/// square is a scalar: the structure of `B²` in 5D algebras.
fn general_four(alg: &Algebra, mv: &SymMv) -> Option<(SymMv, Poly)> {
    let q: SymMv = mv
        .iter()
        .filter(|(m, _)| **m != 0)
        .map(|(m, c)| (*m, c.clone()))
        .collect();
    if q.is_empty() || q.keys().any(|m| m.count_ones() != 4) || q.len() < 2 {
        return None;
    }
    let qq = symbolic::binop(alg, BinOp::Gp, &q, &q);
    if symbolic::support(&qq).iter().any(|&m| m != 0) {
        return None;
    }
    Some((q, coef(&qq, 0)))
}

fn emit_exp_general(
    spec: &AlgebraSpec,
    k: &KindSpec,
    x: &SymMv,
    body: &mut String,
) -> Option<String> {
    let alg = &spec.algebra;
    let nv = k.layout.len() as Var;
    let bsq = symbolic::binop(alg, BinOp::Gp, x, x);
    let (q, qpoly) = general_four(alg, &bsq)?;
    let (c0, c1, s0, s1) = (nv, nv + 1, nv + 2, nv + 3);
    let mut out = symbolic::add(&scalar_mv(Poly::var(c0)), &scale_mv(x, &Poly::var(s0)));
    out = symbolic::add(&out, &scale_mv(&q, &Poly::var(c1)));
    out = symbolic::add(
        &out,
        &symbolic::binop(alg, BinOp::Gp, &scale_mv(&q, &Poly::var(s1)), x),
    );
    let out_kind = spec.kind_for_support(&symbolic::support(&out))?.clone();
    let coeffs = symbolic::to_coeffs(&out_kind.layout, &out).expect("fits");
    let pre = cse::compile_best(&[coef(&bsq, 0), qpoly], &BTreeSet::new(), &[]);
    let prog = cse::compile_best(&coeffs, &BTreeSet::new(), &[]);
    let xvar = |v: Var| format!("x[{v}]");
    let names = move |v: Var| match v {
        v if v < nv => format!("x[{v}]"),
        v if v == c0 => "c0".into(),
        v if v == c1 => "c1".into(),
        v if v == s0 => "s0".into(),
        _ => "s1".into(),
    };
    let mut lets = String::new();
    pre.emit_lets(&xvar, "p", &mut lets);
    let (lam, qq) = (
        render(&pre.outputs[0], &xvar, "p"),
        render(&pre.outputs[1], &xvar, "p"),
    );
    let _ = writeln!(
        lets,
        "        let [c0, c1, s0, s1] = gx::study::exp_coeffs_q({lam}, {qq});"
    );
    prog.emit_lets(&names, "t", &mut lets);
    let outs: Vec<String> = prog
        .outputs
        .iter()
        .map(|o| render(o, &names, "t"))
        .collect();
    let on = &out_kind.name;
    let _ = write!(
        body,
        "    /// The exponential, a unit versor: `exp(B) = C(B²) + S(B²) B`, with `B² = λ + Q` and `Q² = q` a scalar.\n    #[inline]\n    #[allow(unused_variables)]\n    pub fn exp(self) -> gx::Unit<{on}<(), T>> {{\n        let x = self.c;\n{lets}        gx::Unit::new_unchecked({on}::from_coeffs([{}]))\n    }}\n\n",
        outs.join(", ")
    );
    Some(on.clone())
}

fn emit_log_general(
    spec: &AlgebraSpec,
    k: &KindSpec,
    x: &SymMv,
    traits: &mut String,
) -> Option<String> {
    let alg = &spec.algebra;
    let nv = k.layout.len() as Var;
    k.layout.position(0)?;
    let grades: BTreeSet<u32> = k
        .layout
        .blades
        .iter()
        .map(|(m, _)| m.count_ones())
        .collect();
    if !grades.contains(&2) || !grades.contains(&4) || grades.iter().any(|g| ![0, 2, 4].contains(g))
    {
        return None;
    }
    let c4: SymMv = x
        .iter()
        .filter(|(m, _)| m.count_ones() == 4)
        .map(|(m, c)| (*m, c.clone()))
        .collect();
    let p: SymMv = x
        .iter()
        .filter(|(m, _)| m.count_ones() == 2)
        .map(|(m, c)| (*m, c.clone()))
        .collect();
    let cc = symbolic::binop(alg, BinOp::Gp, &c4, &c4);
    if symbolic::support(&cc).iter().any(|&m| m != 0) {
        return None;
    }
    let (h0, h1) = (nv, nv + 1);
    let full = symbolic::add(
        &scale_mv(&p, &Poly::var(h0)),
        &symbolic::binop(alg, BinOp::Gp, &scale_mv(&c4, &Poly::var(h1)), &p),
    );
    // For R = exp(B), C4 is a function of B∧B, which commutes with B, so C4 P is a bivector:
    // its grade-4 part vanishes. Keep the bivector part.
    let out: SymMv = full
        .into_iter()
        .filter(|(m, _)| m.count_ones() == 2)
        .collect();
    let out_kind = spec.kind_for_support(&symbolic::support(&out))?.clone();
    let coeffs = symbolic::to_coeffs(&out_kind.layout, &out).expect("fits");
    let pre = cse::compile_best(&[coef(x, 0), coef(&cc, 0)], &BTreeSet::new(), &[]);
    let prog = cse::compile_best(&coeffs, &BTreeSet::new(), &[]);
    let xvar = |v: Var| format!("x[{v}]");
    let names = move |v: Var| match v {
        v if v < nv => format!("x[{v}]"),
        v if v == h0 => "h0".into(),
        _ => "h1".into(),
    };
    let mut lets = String::new();
    pre.emit_lets(&xvar, "p", &mut lets);
    let (c0, qc) = (
        render(&pre.outputs[0], &xvar, "p"),
        render(&pre.outputs[1], &xvar, "p"),
    );
    let _ = writeln!(
        lets,
        "        let [h0, h1] = gx::study::log_coeffs_q({c0}, {qc});"
    );
    prog.emit_lets(&names, "t", &mut lets);
    let outs: Vec<String> = prog
        .outputs
        .iter()
        .map(|o| render(o, &names, "t"))
        .collect();
    let (name, on) = (&k.name, &out_kind.name);
    let _ = write!(
        traits,
        "impl<T: gx::Real> gx::Log<{on}<(), T>> for gx::Unit<{name}<(), T>> {{\n    /// The logarithm of a unit versor: the bivector `B` with `B.exp() == self`.\n    #[inline]\n    #[allow(unused_variables)]\n    fn log(self) -> {on}<(), T> {{\n        let x = self.into_inner().c;\n{lets}        {on}::from_coeffs([{}])\n    }}\n}}\n\n",
        outs.join(", ")
    );
    Some(on.clone())
}
