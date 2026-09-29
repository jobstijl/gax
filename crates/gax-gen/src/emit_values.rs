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
use crate::kernel::{Kernel, Source, Step, StudyFn, Ty, snake};
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
    /// The methods that have a WGSL form, in language-neutral form (see [`crate::kernel`]).
    pub kernels: Vec<Kernel>,
}

/// The sources of a kind parameter's coefficients `0..n` as program variables `0..n`.
fn arg_vars(n: usize) -> Vec<(Var, Source)> {
    (0..n)
        .map(|i| (i as Var, Source::Arg { param: 0, index: i }))
        .collect()
}

/// A one-parameter kernel on kind `k`, or `None` when it is too large for the WGSL modules.
fn value_kernel(
    k: &KindSpec,
    name: &str,
    doc: &str,
    result: Ty,
    steps: Vec<Step>,
) -> Option<Kernel> {
    (k.layout.len() <= crate::emit::WGSL_MAX).then(|| Kernel {
        name: name.to_string(),
        doc: doc.to_string(),
        params: vec![("x".into(), Ty::Kind(k.name.clone()))],
        result,
        steps,
        entries: None,
    })
}

/// Emit the value methods of kind `k`, returning the code of an inherent impl block (and
/// trait impls) for `K<(), T>`, and what was emitted.
#[allow(clippy::too_many_lines)]
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
    let ks = snake(name);
    if !n0.is_zero() {
        let prog = cse::compile_best(std::slice::from_ref(&n0), &BTreeSet::new(), &[]);
        meta.kernels.extend(value_kernel(
            k,
            &format!("{ks}_norm_squared"),
            "The squared norm: the scalar part of `x ~x`.",
            Ty::Scalar,
            vec![Step::Lets {
                prog: prog.clone(),
                prefix: "t".into(),
                vars: arg_vars(n),
            }],
        ));
        let mut lets = String::new();
        prog.emit_lets(&xvar, "t", &mut lets);
        let _ = write!(
            body,
            "    /// The squared norm: the scalar part of `x ~x`.\n    #[inline(always)]\n    pub fn norm_squared(self) -> T {{\n        let x = self.c;\n{lets}        {}\n    }}\n\n    /// The norm, `sqrt(|norm_squared|)`.\n    #[inline(always)]\n    pub fn norm(self) -> T {{\n        self.norm_squared().abs().sqrt()\n    }}\n\n",
            render(&prog.outputs[0], &xvar, "t")
        );
    }

    if let Some(study) = study_structure(alg, &norm)
        && !n0.is_zero()
    {
        meta.inverse = emit_inverse(spec, k, &rev, &norm, &study, &mut body);
        meta.normalized = emit_normalized(spec, k, &x, &norm, &study, &mut body, &mut meta.kernels);
        emit_newton_step(spec, k, &x, &norm, &mut traits, &mut meta.kernels);
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
                    let mut post_vars = arg_vars(n);
                    for (i, l) in ["c0", "c1", "s0", "s1"].iter().enumerate() {
                        post_vars.push((nv + i as Var, Source::Local((*l).into())));
                    }
                    if out_kind.layout.len() <= crate::emit::WGSL_MAX {
                        meta.kernels.extend(value_kernel(
                            k,
                            &format!("{ks}_exp"),
                            &format!(
                                "The exponential, a unit `{}`: `exp(B) = C(B²) + S(B²) B`.",
                                out_kind.name
                            ),
                            Ty::Kind(out_kind.name.clone()),
                            vec![
                                Step::Lets {
                                    prog: pre.clone(),
                                    prefix: "p".into(),
                                    vars: arg_vars(n),
                                },
                                Step::Study {
                                    func: StudyFn::ExpRotation,
                                    args: vec![(0, 0), (0, 1)],
                                    outs: ["c0", "c1", "s0", "s1"].map(String::from).to_vec(),
                                },
                                Step::Lets {
                                    prog: prog.clone(),
                                    prefix: "t".into(),
                                    vars: post_vars,
                                },
                            ],
                        ));
                    }
                } else {
                    let _ = writeln!(
                        lets,
                        "        let [c0, c1, s0, s1] = gx::study::exp_coeffs({}, {lam_e}, {mu_e});",
                        study.isq
                    );
                    let mut post_vars = arg_vars(n);
                    for (i, l) in ["c0", "c1", "s0", "s1"].iter().enumerate() {
                        post_vars.push((nv + i as Var, Source::Local((*l).into())));
                    }
                    meta.kernels.extend(value_kernel(
                        k,
                        &format!("{ks}_exp"),
                        &format!(
                            "The exponential, a unit `{}`: `exp(B) = C(B²) + S(B²) B`.",
                            out_kind.name
                        ),
                        Ty::Kind(out_kind.name.clone()),
                        vec![
                            Step::Lets {
                                prog: pre.clone(),
                                prefix: "p".into(),
                                vars: arg_vars(n),
                            },
                            Step::Study {
                                func: StudyFn::Exp(study.isq),
                                args: vec![(0, 0), (0, 1)],
                                outs: ["c0", "c1", "s0", "s1"].map(String::from).to_vec(),
                            },
                            Step::Lets {
                                prog: prog.clone(),
                                prefix: "t".into(),
                                vars: post_vars,
                            },
                        ],
                    ));
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
                    "    /// The exponential, a unit versor: `exp(B) = C(B²) + S(B²) B` with `B²` a Study number.\n    #[inline]\n    #[allow(unused_variables)]\n    pub fn exp(self) -> gx::Unit<{on}<(), T>> {{\n        T::vectorize(#[inline(always)] move || {{\n        let x = self.c;\n{lets}        gx::Unit::new_unchecked({on}::from_coeffs([{}]))\n        }})\n    }}\n\n",
                    outs.join(", ")
                );
            }
        }
    }

    // exp in 5D: B² = λ + Q with a 4-vector Q (several blades) whose square is a scalar.
    if all_grade2 && meta.exp.is_none() {
        meta.exp = emit_exp_general(spec, k, &x, &mut body, &mut meta.kernels);
    }
    // exp in any dimension: scaling and squaring in the product closure (6D and up).
    if all_grade2 && meta.exp.is_none() {
        meta.exp = emit_exp_fallback(spec, k, &mut body);
    }

    // log: for unit versors whose parts are a Study number and a bivector.
    meta.log = emit_log(spec, k, &x, &mut traits, &mut meta.kernels);
    if meta.log.is_none() {
        meta.log = emit_log_general(spec, k, &x, &mut traits, &mut meta.kernels);
    }
    if meta.log.is_none() {
        meta.log = emit_log_fallback(spec, k, &mut traits, &mut meta.kernels);
    }

    // sqrt of a unit versor: normalize(1 + R).
    let scalar_idx = k.layout.position(0);
    if let (Some((si, ss)), true) = (scalar_idx, meta.normalized) {
        meta.sqrt = true;
        let _ = write!(
            body,
            "    /// The principal square root of a unit versor, `normalize(1 + R)` (not defined for `R = -1`).\n    #[inline(always)]\n    pub fn sqrt(self) -> gx::Unit<Self> {{\n        let mut c = self.c;\n        c[{si}] = c[{si}] {} T::one();\n        {name}::from_coeffs(c).normalized()\n    }}\n\n",
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
        "    /// The inverse under the geometric product, `~x (x ~x)⁻¹` ({}).\n    #[inline(always)]\n    pub fn inverse(self) -> {on}<(), T> {{\n        let x = self.c;\n{lets}        {on}::from_coeffs([{}])\n    }}\n\n",
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
    kernels: &mut Vec<Kernel>,
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
    let func = match (study.blade, study.isq) {
        (None, _) => StudyFn::RsqrtAbs,
        (Some(_), 0) => StudyFn::RsqrtNil,
        (Some(_), isq) => StudyFn::Rsqrt(isq),
    };
    let (args, outs) = if study.blade.is_some() {
        (
            vec![(0, 0), (0, 1)],
            vec!["s0".to_string(), "s1".to_string()],
        )
    } else {
        (vec![(0, 0)], vec!["s0".to_string()])
    };
    let mut post_vars = arg_vars(nv as usize);
    post_vars.push((s0, Source::Local("s0".into())));
    post_vars.push((s1, Source::Local("s1".into())));
    kernels.extend(value_kernel(
        k,
        &format!("{}_normalized", snake(&k.name)),
        "Scaled to a unit versor, `(x ~x)^(-1/2) x`, so that `x ~x = 1`.",
        Ty::Kind(k.name.clone()),
        vec![
            Step::Lets {
                prog: pre.clone(),
                prefix: "p".into(),
                vars: arg_vars(nv as usize),
            },
            Step::Study { func, args, outs },
            Step::Lets {
                prog: prog.clone(),
                prefix: "t".into(),
                vars: post_vars,
            },
        ],
    ));
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
        "    /// Scaled to a unit versor, `(x ~x)^(-1/2) x`, so that `x ~x = 1` (`±1` when the norm is negative).\n    #[inline(always)]\n    pub fn normalized(self) -> gx::Unit<Self> {{\n        let x = self.c;\n{lets}        gx::Unit::new_unchecked({name}::from_coeffs([{}]))\n    }}\n\n",
        outs.join(", ")
    );
    true
}

/// `NewtonStep`: `x (3 − x ~x) / 2`, one Newton step towards `x ~x = 1` without a square root.
/// For `x ~x = 1 + e` it leaves an error of order `e²`.
fn emit_newton_step(
    spec: &AlgebraSpec,
    k: &KindSpec,
    x: &SymMv,
    norm: &SymMv,
    traits: &mut String,
    kernels: &mut Vec<Kernel>,
) {
    let alg = &spec.algebra;
    let half = Rational::new(1, 2);
    let q: SymMv = symbolic::add(
        &scalar_mv(Poly::constant(Rational::new(3, 2))),
        &norm.iter().map(|(m, p)| (*m, p.scale(-half))).collect(),
    );
    let out = symbolic::binop(alg, BinOp::Gp, x, &q);
    let Some(coeffs) = symbolic::to_coeffs(&k.layout, &out) else {
        return;
    };
    let prog = cse::compile_best(&coeffs, &BTreeSet::new(), &[]);
    kernels.extend(value_kernel(
        k,
        &format!("{}_renormalize_fast", snake(&k.name)),
        "One Newton step towards `x ~x = 1`, `x (3 - x ~x) / 2`, without a square root.",
        Ty::Kind(k.name.clone()),
        vec![Step::Lets {
            prog: prog.clone(),
            prefix: "t".into(),
            vars: arg_vars(k.layout.len()),
        }],
    ));
    let xvar = |v: Var| format!("x[{v}]");
    let mut lets = String::new();
    prog.emit_lets(&xvar, "t", &mut lets);
    let outs: Vec<String> = prog.outputs.iter().map(|o| render(o, &xvar, "t")).collect();
    let name = &k.name;
    let _ = write!(
        traits,
        "impl<T: gx::Coef> gx::NewtonStep for {name}<(), T> {{\n    /// `x (3 − x ~x) / 2`: one Newton step towards `x ~x = 1`, without a square root.\n    #[inline(always)]\n    fn newton_step(self) -> Self {{\n        let x = self.c;\n{lets}        {name}::from_coeffs([{}])\n    }}\n\n    #[inline(always)]\n    fn note_renormalize() {{\n        T::note_renormalize();\n    }}\n}}\n\n",
        outs.join(", ")
    );
}

#[allow(clippy::too_many_lines)]
fn emit_log(
    spec: &AlgebraSpec,
    k: &KindSpec,
    x: &SymMv,
    traits: &mut String,
    kernels: &mut Vec<Kernel>,
) -> Option<String> {
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
        let mut post_vars = arg_vars(nv as usize);
        post_vars.push((h0, Source::Local("h0".into())));
        post_vars.push((h1, Source::Local("h1".into())));
        if out_kind.layout.len() <= crate::emit::WGSL_MAX {
            kernels.extend(value_kernel(
                k,
                &format!("unit_{}_log", snake(&k.name)),
                &format!(
                    "The logarithm of a unit `{}`: the `{}` B with `exp(B) = x` (rotation half-angle in [0, pi]).",
                    k.name, out_kind.name
                ),
                Ty::Kind(out_kind.name.clone()),
                vec![
                    Step::Lets { prog: pre.clone(), prefix: "p".into(), vars: arg_vars(nv as usize) },
                    Step::Study {
                        func: StudyFn::LogRotation,
                        args: vec![(0, 0), (0, 1), (0, 2), (0, 3)],
                        outs: vec!["h0".into(), "h1".into()],
                    },
                    Step::Lets { prog: prog.clone(), prefix: "t".into(), vars: post_vars },
                ],
            ));
        }
    } else {
        let _ = writeln!(
            lets,
            "        let [h0, h1] = gx::study::log_coeffs({isq}, ({}, {}), ({}, {}));",
            r[0], r[1], r[2], r[3]
        );
        let mut post_vars = arg_vars(nv as usize);
        post_vars.push((h0, Source::Local("h0".into())));
        post_vars.push((h1, Source::Local("h1".into())));
        kernels.extend(value_kernel(
            k,
            &format!("unit_{}_log", snake(&k.name)),
            &format!(
                "The logarithm of a unit `{}`: the `{}` B with `exp(B) = x`.",
                k.name, out_kind.name
            ),
            Ty::Kind(out_kind.name.clone()),
            vec![
                Step::Lets {
                    prog: pre.clone(),
                    prefix: "p".into(),
                    vars: arg_vars(nv as usize),
                },
                Step::Study {
                    func: StudyFn::Log(isq),
                    args: vec![(0, 0), (0, 1), (0, 2), (0, 3)],
                    outs: vec!["h0".into(), "h1".into()],
                },
                Step::Lets {
                    prog: prog.clone(),
                    prefix: "t".into(),
                    vars: post_vars,
                },
            ],
        ));
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
        "impl<T: gx::Real> gx::Log<{on}<(), T>> for gx::Unit<{name}<(), T>> {{\n    /// The logarithm of a unit versor: the bivector `B` with `B.exp() == self`.\n    #[inline]\n    #[allow(unused_variables)]\n    fn log(self) -> {on}<(), T> {{\n        T::vectorize(#[inline(always)] move || {{\n        let x = self.into_inner().c;\n{lets}        {on}::from_coeffs([{}])\n        }})\n    }}\n}}\n\n",
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
    kernels: &mut Vec<Kernel>,
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
    let n = nv as usize;
    let mut post_vars = arg_vars(n);
    for (i, l) in ["c0", "c1", "s0", "s1"].iter().enumerate() {
        post_vars.push((nv + i as Var, Source::Local((*l).into())));
    }
    kernels.extend(value_kernel(
        k,
        &format!("{}_exp", snake(&k.name)),
        &format!(
            "The exponential, a unit `{}`: `exp(B) = C(B²) + S(B²) B` with `B² = λ + Q`, `Q²` a scalar.",
            out_kind.name
        ),
        Ty::Kind(out_kind.name.clone()),
        vec![
            Step::Lets { prog: pre.clone(), prefix: "p".into(), vars: arg_vars(n) },
            Step::Study {
                func: StudyFn::ExpQ,
                args: vec![(0, 0), (0, 1)],
                outs: ["c0", "c1", "s0", "s1"].map(String::from).to_vec(),
            },
            Step::Lets { prog: prog.clone(), prefix: "t".into(), vars: post_vars },
        ],
    ));
    prog.emit_lets(&names, "t", &mut lets);
    let outs: Vec<String> = prog
        .outputs
        .iter()
        .map(|o| render(o, &names, "t"))
        .collect();
    let on = &out_kind.name;
    let _ = write!(
        body,
        "    /// The exponential, a unit versor: `exp(B) = C(B²) + S(B²) B`, with `B² = λ + Q` and `Q² = q` a scalar.\n    #[inline]\n    #[allow(unused_variables)]\n    pub fn exp(self) -> gx::Unit<{on}<(), T>> {{\n        T::vectorize(#[inline(always)] move || {{\n        let x = self.c;\n{lets}        gx::Unit::new_unchecked({on}::from_coeffs([{}]))\n        }})\n    }}\n\n",
        outs.join(", ")
    );
    Some(on.clone())
}

fn emit_log_general(
    spec: &AlgebraSpec,
    k: &KindSpec,
    x: &SymMv,
    traits: &mut String,
    kernels: &mut Vec<Kernel>,
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
    let n = nv as usize;
    let mut post_vars = arg_vars(n);
    post_vars.push((h0, Source::Local("h0".into())));
    post_vars.push((h1, Source::Local("h1".into())));
    kernels.extend(value_kernel(
        k,
        &format!("unit_{}_log", snake(&k.name)),
        &format!(
            "The logarithm of a unit `{}`: the `{}` B with `exp(B) = x`.",
            k.name, out_kind.name
        ),
        Ty::Kind(out_kind.name.clone()),
        vec![
            Step::Lets {
                prog: pre.clone(),
                prefix: "p".into(),
                vars: arg_vars(n),
            },
            Step::Study {
                func: StudyFn::LogQ,
                args: vec![(0, 0), (0, 1)],
                outs: vec!["h0".into(), "h1".into()],
            },
            Step::Lets {
                prog: prog.clone(),
                prefix: "t".into(),
                vars: post_vars,
            },
        ],
    ));
    prog.emit_lets(&names, "t", &mut lets);
    let outs: Vec<String> = prog
        .outputs
        .iter()
        .map(|o| render(o, &names, "t"))
        .collect();
    let (name, on) = (&k.name, &out_kind.name);
    let _ = write!(
        traits,
        "impl<T: gx::Real> gx::Log<{on}<(), T>> for gx::Unit<{name}<(), T>> {{\n    /// The logarithm of a unit versor: the bivector `B` with `B.exp() == self`.\n    #[inline]\n    #[allow(unused_variables)]\n    fn log(self) -> {on}<(), T> {{\n        T::vectorize(#[inline(always)] move || {{\n        let x = self.into_inner().c;\n{lets}        {on}::from_coeffs([{}])\n        }})\n    }}\n}}\n\n",
        outs.join(", ")
    );
    Some(on.clone())
}

/// The logarithm of a unit versor by inverse scaling and squaring, for even kinds without a
/// closed form (CSTA's `Even`, whose bivectors split into three commuting parts): the
/// counterpart of [`emit_exp_fallback`]. Emitted for a kind of even grades, with a scalar and
/// the whole grade 2, closed under the product, whose bivector part is a kind of its own.
#[allow(clippy::too_many_lines)]
fn emit_log_fallback(
    spec: &AlgebraSpec,
    k: &KindSpec,
    traits: &mut String,
    kernels: &mut Vec<Kernel>,
) -> Option<String> {
    let alg = &spec.algebra;
    let blades: BTreeSet<u32> = k.layout.blades.iter().map(|(m, _)| *m).collect();
    if !blades.contains(&0) || blades.iter().any(|m| m.count_ones() % 2 == 1) {
        return None;
    }
    let grade2: BTreeSet<u32> = (0..alg.blade_count() as u32)
        .filter(|m| m.count_ones() == 2)
        .collect();
    if !grade2.is_subset(&blades) {
        return None;
    }
    for &a in &blades {
        for &b in &blades {
            if alg
                .blade_product(a, b)
                .iter()
                .any(|(m, _)| !blades.contains(m))
            {
                return None;
            }
        }
    }
    let bv = spec.kind_for_support(&grade2)?;
    if bv.layout.len() != grade2.len() {
        return None;
    }
    let (en, bn) = (&k.name, &bv.name);
    let (one_pos, one_sign) = k.layout.position(0)?;
    let one = if one_sign > 0 {
        "T::one()"
    } else {
        "-T::one()"
    };
    let project: Vec<String> = bv
        .layout
        .blades
        .iter()
        .map(|&(m, sb)| {
            let (pos, se) = k.layout.position(m).expect("grade 2 is in the kind");
            let sign = if sb * se > 0 { "" } else { "-" };
            format!("{sign}l.c[{pos}] * f")
        })
        .collect();
    // In 6D the closed form is the logarithm, and this is its fallback near a half turn.
    let six = alg.dim() == 6;
    let (head, start) = if six {
        (
            format!(
                "impl<T: gx::Real> {en}<(), T> {{\n    /// The logarithm of a unit `{en}` by inverse scaling and squaring (the fallback of\n    /// `log` near a half turn, where the closed form loses `ε/⟨R⟩₀`): square roots until\n    /// `R` is within 1/16 of the identity, a series of `log(1 + z)`, the scaling undone.\n    #[doc(hidden)]\n    #[inline]\n    pub fn log_by_scaling(self) -> {bn}<(), T> {{"
            ),
            "self",
        )
    } else {
        (
            format!(
                "impl<T: gx::Real> gx::Log<{bn}<(), T>> for gx::Unit<{en}<(), T>> {{\n    /// The logarithm of a unit versor, by inverse scaling and squaring (no closed form is\n    /// generated for `{en}` in this algebra): square roots until `R` is within 1/16 of the\n    /// identity (1-norm), a series of `log(1 + z)` to degree 16, and the scaling undone. Each\n    /// square root is `(1 + R)` scaled so that every invariant part is at most 1, then made\n    /// unit by Newton steps `y (3 - ~y y) / 2` (a polar decomposition), until `~y y` is 1 to\n    /// within 64 ε on every lane. The principal logarithm: rotations below a half turn in each\n    /// invariant plane, and boosts and dilations of large rapidity (tested up to 5).\n    #[inline]\n    fn log(self) -> {bn}<(), T> {{"
            ),
            "self.into_inner()",
        )
    };
    let _ = write!(
        traits,
        "{head}
        T::vectorize(#[inline(always)] move || {{
        let mut one = {en}::<(), T>::zero();
        one.c[{one_pos}] = {one};
        let (half, three) = (T::from_ratio(1, 2), one.gp(T::from_i64(3)));
        let dist = |x: {en}<(), T>| {{
            let mut d = T::zero();
            for (a, b) in x.c.iter().zip(one.c) {{
                d = d + (*a - b).abs();
            }}
            d
        }};
        let tol = T::epsilon() * T::from_i64(64);
        let mut x = {start};
        let mut s = 0u32;
        while s < 64 && !T::all_lt(dist(x), T::from_ratio(1, 16)) {{
            let mut y = one + x;
            // Scale so that every invariant part of y ~y is at most 1 (they are positive, and
            // there are at most four distinct ones, averaging to the scalar part).
            let n = {one_sign_mul}(y.reverse() * y).c[{one_pos}];
            y = y.gp((n * T::from_i64(4)).sqrt().recip());
            let mut k = 0;
            while k < 200 && !T::all_lt(dist(y.reverse() * y), tol) {{
                y = y * (three - y.reverse() * y).gp(half);
                k += 1;
            }}
            x = y;
            s += 1;
        }}
        // log(1 + z) = z (1 - z/2 + z²/3 - ...), by Horner, for |z| <= 1/16.
        let z = x - one;
        let mut q = one.gp(T::from_ratio(-1, 16));
        for k in (1..16).rev() {{
            let a = if k % 2 == 1 {{ T::from_ratio(1, k) }} else {{ T::from_ratio(-1, k) }};
            q = one.gp(a) + z * q;
        }}
        let l = z * q;
        let mut f = T::one();
        for _ in 0..s {{
            f = f + f;
        }}
        {bn}::from_coeffs([{}])
        }})
    }}
}}

",
        project.join(", "),
        one_sign_mul = if one_sign > 0 { "" } else { "-" },
    );
    if six {
        emit_log_6d(spec, k, bv, traits, kernels);
    }
    Some(bn.clone())
}

/// The closed-form logarithm of a 6D even versor (docs/log6d.md): the invariants and the three
/// bivectors `G1 = ⟨R⟩₂`, `G2 = r0 ⟨R₄ R₂⟩₂`, `G3 = r0 ⟨R₆ R₄⟩₂` as one straight-line program,
/// the interpolant from `gx::study::log_coeffs_6d`, and `log_by_scaling` near a half turn.
#[allow(clippy::too_many_lines)]
fn emit_log_6d(
    spec: &AlgebraSpec,
    k: &KindSpec,
    bv: &KindSpec,
    traits: &mut String,
    kernels: &mut Vec<Kernel>,
) {
    let alg = &spec.algebra;
    let x = symbolic::variables(&k.layout, 0);
    let grade = |g: u32| -> SymMv {
        x.iter()
            .filter(|(m, _)| m.count_ones() == g)
            .map(|(m, c)| (*m, c.clone()))
            .collect()
    };
    let (r2, r4, r6) = (grade(2), grade(4), grade(6));
    let r0 = coef(&x, 0);
    let a2 = coef(&symbolic::binop(alg, BinOp::Gp, &r2, &r2), 0);
    let a4 = coef(&symbolic::binop(alg, BinOp::Gp, &r4, &r4), 0);
    let three = Poly::constant(Rational::int(3));
    let p3 = &r0 * &r0;
    let p2 = &(&three * &p3) - &a2;
    let p1 = &(&a4 + &(&three * &p3)) - &a2.scale(Rational::int(2));
    let two = |mv: &SymMv| -> SymMv {
        mv.iter()
            .filter(|(m, _)| m.count_ones() == 2)
            .map(|(m, c)| (*m, c.clone()))
            .collect()
    };
    let g1 = symbolic::to_coeffs(&bv.layout, &r2).expect("the bivectors");
    let g2 = symbolic::to_coeffs(
        &bv.layout,
        &scale_mv(&two(&symbolic::binop(alg, BinOp::Gp, &r4, &r2)), &r0),
    )
    .expect("the bivectors");
    let g3 = symbolic::to_coeffs(
        &bv.layout,
        &scale_mv(&two(&symbolic::binop(alg, BinOp::Gp, &r6, &r4)), &r0),
    )
    .expect("the bivectors");
    let n = bv.layout.len();
    let mut outs = vec![r0, p1, p2, p3];
    outs.extend(g1);
    outs.extend(g2);
    outs.extend(g3);
    let prog = cse::compile_best(&outs, &BTreeSet::new(), &[]);
    // The WGSL kernel: this program, the weights (`study_log6`), and their combination, which
    // reads the `G`s as outputs of the first program. The module's `unit_{even}_log` calls it
    // above `⟨R⟩₀ = 1/16` (`emit_wgsl::fallback_log`).
    let base = k.layout.len() as Var;
    let w = |j: usize| Poly::var(base + j as Var);
    let g = |j: usize, i: usize| Poly::var(base + 3 + (j * n + i) as Var);
    let combined: Vec<Poly> = (0..n)
        .map(|i| &(&(&w(0) * &g(0, i)) + &(&w(1) * &g(1, i))) + &(&w(2) * &g(2, i)))
        .collect();
    let post = cse::compile_best(&combined, &BTreeSet::new(), &[]);
    let mut post_vars: Vec<(Var, Source)> = (0..3)
        .map(|j| (base + j, Source::Local(format!("w{}", j + 1))))
        .collect();
    post_vars.extend((0..3 * n).map(|o| {
        (
            base + 3 + o as Var,
            Source::Output {
                step: 0,
                index: 4 + o,
            },
        )
    }));
    kernels.extend(value_kernel(
        k,
        &format!("unit_{}_log_closed", snake(&k.name)),
        &format!(
            "The logarithm of a unit `{}` in closed form (docs/log6d.md), for `<x>_0 > 1/16`: `unit_{}_log` calls it there.",
            k.name,
            snake(&k.name)
        ),
        Ty::Kind(bv.name.clone()),
        vec![
            Step::Lets {
                prog: prog.clone(),
                prefix: "p".into(),
                vars: arg_vars(k.layout.len()),
            },
            Step::Study {
                func: StudyFn::Log6,
                args: vec![(0, 1), (0, 2), (0, 3), (0, 0)],
                outs: vec!["w1".into(), "w2".into(), "w3".into()],
            },
            Step::Lets {
                prog: post,
                prefix: "t".into(),
                vars: post_vars,
            },
        ],
    ));
    let xvar = |v: Var| format!("x[{v}]");
    let mut lets = String::new();
    prog.emit_lets(&xvar, "p", &mut lets);
    let o: Vec<String> = prog
        .outputs
        .iter()
        .map(|op| render(op, &xvar, "p"))
        .collect();
    let list = |from: usize| o[from..from + n].join(", ");
    let (en, bn) = (&k.name, &bv.name);
    let _ = write!(
        traits,
        "impl<T: gx::Real> gx::Log<{bn}<(), T>> for gx::Unit<{en}<(), T>> {{
    /// The logarithm of a unit versor, in closed form through the invariant decomposition
    /// (docs/log6d.md): `u_j = cosh²(μ_j)` of the three commuting planes are the roots of a
    /// cubic in the scalar parts of `R`'s grade parts squared, and `log R` is
    /// `r0⁻¹ (α2 Q2 + α1 Q1 + α0 ⟨R⟩₂)` for bivectors `Q` from `R`'s grade parts and the
    /// quadratic `α` interpolating `φ(u) = √u asinh(√(u−1))/√(u−1)` at the roots
    /// (`gx::study::log_coeffs_6d`, folded into three weights by `log_weights_6d`). Near a half
    /// turn (`⟨R⟩₀ < 1/16`), where it loses `ε/⟨R⟩₀`, the lanes there use inverse scaling and
    /// squaring instead. The principal logarithm:
    /// rotations below a half turn in each invariant plane, boosts and dilations of any size.
    #[inline]
    #[allow(unused_variables)]
    fn log(self) -> {bn}<(), T> {{
        T::vectorize(#[inline(always)] move || {{
        let x = self.into_inner().c;
{lets}        let (r0, p1, p2, p3) = ({}, {}, {}, {});
        let g1: [T; {n}] = [{}];
        let g2: [T; {n}] = [{}];
        let g3: [T; {n}] = [{}];
        let [w1, w2, w3] = gx::study::log_weights_6d(p1, p2, p3, r0);
        let closed = {bn}::from_coeffs(core::array::from_fn(|i| w1 * g1[i] + w2 * g2[i] + w3 * g3[i]));
        let limit = T::from_ratio(1, 16);
        if T::all_lt(limit, r0) {{
            return closed;
        }}
        let numeric = self.into_inner().log_by_scaling();
        {bn}::from_coeffs(core::array::from_fn(|i| T::select_lt(limit, r0, closed.c[i], numeric.c[i])))
        }})
    }}
}}

",
        o[0],
        o[1],
        o[2],
        o[3],
        list(4),
        list(4 + n),
        list(4 + 2 * n),
    );
}

/// The exponential by scaling and squaring, for bivector kinds without a closed form (the
/// invariant decomposition of a 6D bivector has three parts): a Taylor series of `B / 2^8` in
/// the smallest kind closed under the product, then eight squarings. Works in any algebra.
fn emit_exp_fallback(spec: &AlgebraSpec, k: &KindSpec, body: &mut String) -> Option<String> {
    let alg = &spec.algebra;
    // The support closure of {1} and B's blades under the geometric product.
    let mut closure: BTreeSet<u32> = k.layout.blades.iter().map(|(m, _)| *m).collect();
    closure.insert(0);
    loop {
        let mut next = closure.clone();
        for &a in &closure {
            for &b in &closure {
                for &(m, _) in alg.blade_product(a, b) {
                    next.insert(m);
                }
            }
        }
        if next == closure {
            break;
        }
        closure = next;
    }
    let e = spec.kind_for_support(&closure)?.clone();
    let (en, name) = (&e.name, &k.name);
    let mut embed = String::new();
    for (i, &(m, sb)) in k.layout.blades.iter().enumerate() {
        let (pos, se) = e.layout.position(m)?;
        let sign = if sb * se > 0 { "" } else { "-" };
        let _ = writeln!(embed, "        x.c[{pos}] = {sign}self.c[{i}] * h;");
    }
    let (one_pos, one_sign) = e.layout.position(0)?;
    let one = if one_sign > 0 {
        "T::one()"
    } else {
        "-T::one()"
    };
    let _ = write!(
        body,
        "    /// The exponential, a unit versor, by scaling and squaring in `{en}`: `B` is halved `s` times\n    /// until `‖B / 2^s‖₁ ≤ 1/16` (the sum of absolute coefficients), a Taylor series of degree 10\n    /// gives `exp(B / 2^s)` to below `10⁻²⁰` relative, a Newton step renormalizes it while it is\n    /// near 1, `s` squarings undo the scaling, and a second Newton step renormalizes the result\n    /// where it is small (a large boost is left as squared: there `~r r − 1` cancels). (No closed form is generated for\n    /// `{name}` in this algebra.)\n    #[inline]\n    pub fn exp(self) -> gx::Unit<{en}<(), T>> {{\n        T::vectorize(#[inline(always)] move || {{\n        let mut norm = T::zero();\n        for c in self.c {{\n            norm = norm + c.abs();\n        }}\n        let (limit, half) = (T::from_ratio(1, 16), T::from_ratio(1, 2));\n        let (mut h, mut s) = (T::one(), 0u32);\n        while s < 64 && !T::all_lt(norm * h, limit) {{\n            h = h * half;\n            s += 1;\n        }}\n        let mut x = {en}::<(), T>::zero();\n{embed}        let mut one = {en}::<(), T>::zero();\n        one.c[{one_pos}] = {one};\n        // Horner: 1 + x (1 + x/2 (1 + x/3 (... (1 + x/10))))\n        let mut r = one;\n        for k in (1..=10).rev() {{\n            r = one + (x * r).gp(T::from_ratio(1, k));\n        }}\n        // A Newton step r (3 - ~r r) / 2 while r is near 1, where ~r r - 1 has no cancellation.\n        let three = one.gp(T::from_i64(3));\n        let m = r.reverse() * r;\n        r = r * (three - m).gp(half);\n        for _ in 0..s {{\n            r = r * r;\n        }}\n        // Another after squaring, lane by lane, only where r is small (‖r‖₁ < 4, compact\n        // motions): for a large boost ~r r - 1 cancels at the scale of ‖r‖², and the step\n        // would add error rather than remove it.\n        let mut size = T::zero();\n        for c in r.c {{\n            size = size + c.abs();\n        }}\n        let m = r.reverse() * r;\n        let fixed = r * (three - m).gp(half);\n        for (c, f) in r.c.iter_mut().zip(fixed.c) {{\n            *c = T::select_lt(size, T::from_i64(4), f, *c);\n        }}\n        gx::Unit::new_unchecked(r)\n        }})\n    }}\n\n"
    );
    Some(en.clone())
}
