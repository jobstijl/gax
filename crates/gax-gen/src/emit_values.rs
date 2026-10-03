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
use std::collections::BTreeMap;
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

/// `exp` of a kind spanning `1` and one blade `B` whose square is a constant (`±1` or `0`), or
/// just `B`: for `x = a + b B`, `exp(x) = e^a (C + S B)` with `(C, S)` = `(cos b, sin b)`,
/// `(cosh b, sinh b)` or `(1, b)` by the sign of `B²`. Emitted where the algebra has a kind with
/// exactly the support `{1, B}` for the result (STA's `Phasor`); a plain value, since
/// `x ~x` need not be 1 (in STA `(cos b + sin b I)^~ = cos b + sin b I`).
fn emit_exp_single_blade(spec: &AlgebraSpec, k: &KindSpec, body: &mut String) {
    let alg = &spec.algebra;
    let blades: Vec<(u32, i8)> = k.layout.blades.iter().map(|&(m, s)| (m, s as i8)).collect();
    let others: Vec<(u32, i8)> = blades.iter().copied().filter(|b| b.0 != 0).collect();
    let [(b, sb)] = others.as_slice() else {
        return;
    };
    let isq = match alg.blade_product(*b, *b) {
        [] => 0,
        [(0, c)] if c.abs() == 1 => *c,
        _ => return,
    };
    let support: BTreeSet<u32> = [0, *b].into_iter().collect();
    let Some(out) = spec.kinds.iter().find(|o| {
        o.layout.blades.len() == 2
            && o.layout.blades.iter().map(|x| x.0).collect::<BTreeSet<_>>() == support
    }) else {
        return;
    };
    let pos = |layout: &crate::table::Layout, m: u32| layout.position(m);
    // The input's coefficients of 1 and B, signs applied.
    let scalar = match pos(&k.layout, 0) {
        Some((i, s)) => format!("{}self.c[{i}]", if s > 0 { "" } else { "-" }),
        None => "T::zero()".into(),
    };
    let (bi, _) = pos(&k.layout, *b).expect("the blade");
    let bcoef = format!("{}self.c[{bi}]", if *sb > 0 { "" } else { "-" });
    let (c, sn) = match isq {
        -1 => ("cos", "sin"),
        1 => ("cosh", "sinh"),
        _ => ("", ""),
    };
    let (cs, ss) = if isq == 0 {
        ("T::one()".to_string(), "b".to_string())
    } else {
        (format!("b.{c}()"), format!("b.{sn}()"))
    };
    let (o0, so0) = pos(&out.layout, 0).expect("a scalar");
    let (ob, sob) = pos(&out.layout, *b).expect("the blade");
    let mut coeffs = vec![String::new(); 2];
    coeffs[o0] = format!("{}e * ({cs})", if so0 > 0 { "" } else { "-" });
    coeffs[ob] = format!("{}e * ({ss})", if sob > 0 { "" } else { "-" });
    let (kind_sq, name) = (
        match isq {
            -1 => "-1",
            1 => "+1",
            _ => "0",
        },
        &out.name,
    );
    let _ = write!(
        body,
        "    /// The exponential: for `x = a + b B` with `B² = {kind_sq}`, `e^a (C(b) + S(b) B)` with `C, S` the
    /// cosine and sine (`B² = -1`), the hyperbolic ones (`+1`) or `1, b` (`0`). A `{name}`, not a
    /// `Unit`: `x ~x` need not be 1.
    #[inline]
    pub fn exp(self) -> {name}<(), T> {{
        let (a, b) = ({scalar}, {bcoef});
        let e = a.exp();
        {name}::from_coeffs([{}])
    }}

",
        coeffs.join(", ")
    );
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
#[allow(clippy::struct_excessive_bools)] // one flag per optional method, not a state machine
pub struct ValueMethods {
    /// Kind name.
    pub kind: String,
    /// `norm_squared` and `norm`.
    pub norm: bool,
    /// `inverse`, with its output kind.
    pub inverse: Option<String>,
    /// Whether `inverse` is the general one (Shirokov's, ADR-037) rather than a closed form.
    pub inverse_general: bool,
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
    // Every other kind: the inverse as a polynomial in x (Shirokov), in its product closure.
    if meta.inverse.is_none() && !k.layout.blades.is_empty() {
        meta.inverse = emit_inverse_general(spec, k, &mut body);
        meta.inverse_general = meta.inverse.is_some();
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
    // exp in closed form from 6D to 9D (three or four invariant planes, docs/log6d.md §12).
    if all_grade2 && meta.exp.is_none() {
        meta.exp = emit_exp_closed(spec, k, &mut body, &mut meta.kernels);
    }
    // exp in any dimension: scaling and squaring in the product closure.
    if all_grade2 && meta.exp.is_none() {
        meta.exp = emit_exp_fallback(spec, k, &mut body);
    }

    // exp of a scalar plus one blade whose square is a constant (STA's pseudoscalar, a phasor).
    if meta.exp.is_none() && !all_grade2 {
        emit_exp_single_blade(spec, k, &mut body);
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

/// The inverse of a kind without a closed form (no Study structure), by Shirokov's method
/// (*On computing the determinant, other characteristic polynomial coefficients, and inverse in
/// Clifford algebras of arbitrary dimension*, 2021): the Faddeev–LeVerrier recursion on left
/// multiplication, which needs only products and scalar parts, in the smallest kind closed under
/// the product. Its degree is the dimension of a faithful representation in which the trace is
/// that dimension times the scalar part: `2^⌈(p+q)/2⌉` for `R(p, q)` (both half-spinor modules
/// in odd dimension). Null directions can raise it (to at most `2^r` times; `R(2,0,2)` needs 8,
/// not Shirokov's 4) and give the recursion repeated eigenvalues it cannot resolve, so where
/// they are basis vectors the recursion runs on the part of `x` free of them, and a finite
/// series adds the nilpotent rest. Newton–Schulz steps
/// then restore the digits the recursion loses at high degree (ADR-037).
fn emit_inverse_general(spec: &AlgebraSpec, k: &KindSpec, body: &mut String) -> Option<String> {
    let alg = &spec.algebra;
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
    let (p, q, r) = alg.signature();
    if !some_invertible(alg, k, 1u32 << ((p + q).div_ceil(2) + r)) {
        return None;
    }
    // The null basis vectors (zero rows of the metric). Where they span the metric's radical,
    // `x = a + n` with `a` free of them and `n` nilpotent, and the recursion runs on `a` alone,
    // at the non-degenerate degree: repeated eigenvalues in Jordan blocks, which a null
    // direction gives left multiplication, cost it all its digits in 7D and 8D PGA.
    let null: u32 = alg
        .metric()
        .iter()
        .enumerate()
        .filter(|(_, row)| row.iter().all(|&v| v == 0))
        .fold(0, |m, (i, _)| m | (1 << i));
    let split = null.count_ones() as usize == r && r > 0;
    let degree = if split {
        1u32 << (p + q).div_ceil(2)
    } else {
        1u32 << ((p + q).div_ceil(2) + r)
    };
    let (en, name) = (&e.name, &k.name);
    let mut embed = String::new();
    for (i, &(m, sb)) in k.layout.blades.iter().enumerate() {
        let (pos, se) = e.layout.position(m)?;
        let sign = if sb * se > 0 { "" } else { "-" };
        let _ = writeln!(embed, "        x.c[{pos}] = {sign}self.c[{i}] * scale;");
    }
    let (one, one_sign) = e.layout.position(0)?;
    // `⟨u⟩₀` and `u − c` in `e`'s layout, whose scalar blade may be stored negated.
    let (get, put) = if one_sign > 0 { ("", "-") } else { ("-", "+") };
    let two = if one_sign > 0 {
        "+ T::from_i64(2)"
    } else {
        "- T::from_i64(2)"
    };
    // Refinement steps: one where the recursion is short, two at degree 32 (9D).
    let steps = if degree <= 16 { 1 } else { 2 };
    let (body_of, neumann) = if split {
        let zeroed: Vec<String> = e
            .layout
            .blades
            .iter()
            .enumerate()
            .filter(|(_, (m, _))| m & null != 0)
            .map(|(i, _)| format!("a.c[{i}] = T::zero();"))
            .collect();
        (
            format!(
                "        // The body `a` (no null direction) and the nilpotent rest `n = x − a`.\n        let mut a = x;\n        {}\n        let n = x - a;\n",
                zeroed.join(" ")
            ),
            format!(
                "        // x⁻¹ = Σₖ (−a⁻¹ n)ᵏ a⁻¹, k = 0..={r}: (a⁻¹ n)^{} = 0, every term holding a null\n        // direction twice.\n        let m = -(y * n);\n        let mut term = y;\n        for _ in 0..{r} {{\n            term = m * term;\n            y = y + term;\n        }}\n",
                r + 1
            ),
        )
    } else {
        ("        let a = x;\n".to_string(), String::new())
    };
    let what = if split {
        format!(
            "on the part of `x` free of the null direction(s), then a finite series for the\n    /// nilpotent rest ({r} term(s)), then {steps} Newton–Schulz step(s)"
        )
    } else {
        format!("then {steps} Newton–Schulz step(s)")
    };
    let _ = write!(
        body,
        "    /// The inverse under the geometric product, by Shirokov's method in `{en}` (no closed form\n    /// for `{name}` here): the inverse as a polynomial of degree {d} whose coefficients come\n    /// from the scalar parts of powers (the Faddeev–LeVerrier recursion on left multiplication,\n    /// {d} products in `{en}`), {what} (docs/design.md, ADR-037). `x` is scaled to its largest\n    /// coefficient first. Not finite where `x` has no inverse.\n    #[inline]\n    pub fn inverse(self) -> {en}<(), T>\n    where\n        T: Real,\n    {{\n        T::vectorize(#[inline(always)] move || {{\n        let mut size = T::zero();\n        for c in self.c {{\n            size = size.max(c.abs());\n        }}\n        let scale = size.recip();\n        let mut x = {en}::<(), T>::zero();\n{embed}{body_of}        // U₁ = a, Cₖ = (N/k) ⟨Uₖ⟩₀, Uₖ₊₁ = a (Uₖ − Cₖ); then a⁻¹ = (U_{{N−1}} − C_{{N−1}}) / C_N.\n        let mut prev = a;\n        let c = {get}a.c[{one}] * T::from_i64({n});\n        prev.c[{one}] = prev.c[{one}] {put} c;\n        for k in 2..{n}i64 {{\n            let u = a * prev;\n            let c = {get}u.c[{one}] * T::from_ratio({n}, k);\n            prev = u;\n            prev.c[{one}] = prev.c[{one}] {put} c;\n        }}\n        let det = {get}(a * prev).c[{one}];\n        let mut y = prev.gp(det.recip());\n{neumann}        // Newton–Schulz, y ← y (2 − x y), squares the residual: the recursion loses digits as\n        // its degree grows (to 10⁻⁴ at degree 32), and the step(s) restore them.\n        for _ in 0..{steps} {{\n            let mut t = -(x * y);\n            t.c[{one}] = t.c[{one}] {two};\n            y = y * t;\n        }}\n        y.gp(scale)\n        }})\n    }}\n\n",
        d = degree - 1,
        n = degree,
    );
    Some(en.clone())
}

/// Whether some value of the kind has an inverse: Shirokov's last coefficient `C_N`, a
/// polynomial in the coefficients that vanishes exactly where there is none, is nonzero at a
/// random point (computed modulo `2⁶¹ − 1`; a nonzero residue proves `C_N ≢ 0`). It is
/// identically zero for kinds of nilpotent values, such as a PGA pseudoscalar.
fn some_invertible(alg: &Algebra, k: &KindSpec, degree: u32) -> bool {
    const P: u128 = (1 << 61) - 1;
    let modp = |v: i128| v.rem_euclid(P as i128) as u128;
    let inv = |a: u128| {
        // a^(P−2) mod P
        let (mut base, mut e, mut acc) = (a % P, P - 2, 1u128);
        while e > 0 {
            if e & 1 == 1 {
                acc = acc * base % P;
            }
            base = base * base % P;
            e >>= 1;
        }
        acc
    };
    let product = |x: &BTreeMap<u32, u128>, y: &BTreeMap<u32, u128>| {
        let mut out: BTreeMap<u32, u128> = BTreeMap::new();
        for (&a, &ca) in x {
            for (&b, &cb) in y {
                for &(m, v) in alg.blade_product(a, b) {
                    let t = ca * cb % P * modp(i128::from(v)) % P;
                    let e = out.entry(m).or_default();
                    *e = (*e + t) % P;
                }
            }
        }
        out
    };
    let mut state = 0x5eed_1e55_u64;
    (0..2).any(|_| {
        let x: BTreeMap<u32, u128> = k
            .layout
            .blades
            .iter()
            .map(|(m, _)| {
                state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
                let mut z = state;
                z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                (*m, u128::from(z ^ (z >> 31)) % P)
            })
            .collect();
        let n = u128::from(degree);
        let scalar = |u: &BTreeMap<u32, u128>| u.get(&0).copied().unwrap_or(0);
        let mut prev = x.clone();
        let mut c = scalar(&x) * n % P;
        *prev.entry(0).or_default() = (scalar(&prev) + P - c) % P;
        for kk in 2..n {
            let u = product(&x, &prev);
            c = scalar(&u) * n % P * inv(kk) % P;
            prev = u;
            *prev.entry(0).or_default() = (scalar(&prev) + P - c) % P;
        }
        scalar(&product(&x, &prev)) != 0
    })
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
    // From 6D to 9D the logarithm is in closed form, turning planes near a half turn
    // (log6d.md): three invariant planes in 6D and 7D, four in 8D and 9D.
    match alg.dim() {
        6 | 7 => {
            emit_log_6d(spec, k, bv, traits, kernels);
            return Some(bv.name.clone());
        }
        8 | 9 => {
            emit_log_8d(spec, k, bv, traits);
            return Some(bv.name.clone());
        }
        _ => {}
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
    let head = format!(
        "impl<T: gx::Real> gx::Log<{bn}<(), T>> for gx::Unit<{en}<(), T>> {{\n    /// The logarithm of a unit versor, by inverse scaling and squaring (no closed form is\n    /// generated for `{en}` in this algebra): square roots until `R` is within 1/16 of the\n    /// identity (1-norm), a series of `log(1 + z)` to degree 16, and the scaling undone. Each\n    /// square root is `(1 + R)` scaled so that every invariant part is at most 1, then made\n    /// unit by Newton steps `y (3 - ~y y) / 2` (a polar decomposition), until `~y y` is 1 to\n    /// within 64 ε on every lane. The principal logarithm: rotations below a half turn in each\n    /// invariant plane, and boosts and dilations of large rapidity (tested up to 5).\n    #[inline]\n    fn log(self) -> {bn}<(), T> {{"
    );
    let start = "self.into_inner()";
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
    Some(bn.clone())
}

/// The closed-form logarithm of a 6D even versor (docs/log6d.md): the invariants and the three
/// bivectors `G1 = ⟨R⟩₂`, `G2 = r0 ⟨R₄ R₂⟩₂`, `G3 = r0 ⟨R₆ R₄⟩₂` as one straight-line program,
/// the interpolant from `gx::study::log_coeffs_6d`, and near a half turn the planes turned by
/// `gx::study::log_turn_6d` first. Also the WGSL kernels `unit_{e}_log_closed` and
/// `unit_{e}_log_turning`.
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
    // The planes to turn near a half turn: `Z = α2 Q2 + α1 Q1 + α0 G1` from `study_log6_turn`,
    // with `Q1 = G3 − G2 + p3 G1`, `Q2 = G2 + p1 Q1 − 2 p3 G1`.
    let (p1v, p3v) = (
        Poly::var(base + 3 + (3 * n) as Var),
        Poly::var(base + 4 + (3 * n) as Var),
    );
    let two = Poly::constant(Rational::int(2));
    let zs: Vec<Poly> = (0..n)
        .map(|i| {
            let q1 = &(&g(2, i) - &g(1, i)) + &(&p3v * &g(0, i));
            let q2 = &(&g(1, i) + &(&p1v * &q1)) - &(&(&two * &p3v) * &g(0, i));
            &(&(&w(2) * &q2) + &(&w(1) * &q1)) + &(&w(0) * &g(0, i))
        })
        .collect();
    let turn = cse::compile_best(&zs, &BTreeSet::new(), &[]);
    let mut turn_vars: Vec<(Var, Source)> = (0..3)
        .map(|j| (base + j, Source::Local(format!("a{j}"))))
        .collect();
    turn_vars.extend((0..3 * n).map(|o| {
        (
            base + 3 + o as Var,
            Source::Output {
                step: 0,
                index: 4 + o,
            },
        )
    }));
    turn_vars.push((
        base + 3 + (3 * n) as Var,
        Source::Output { step: 0, index: 1 },
    ));
    turn_vars.push((
        base + 4 + (3 * n) as Var,
        Source::Output { step: 0, index: 3 },
    ));
    kernels.extend(value_kernel(
        k,
        &format!("unit_{}_log_turning", snake(&k.name)),
        &format!(
            "The sum `Z` of the unit bivectors of the planes `unit_{}_log` turns by a quarter turn (docs/log6d.md); `-<Z Z>_0` is their number.",
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
                func: StudyFn::Log6Turn,
                args: vec![(0, 1), (0, 2), (0, 3), (0, 0)],
                outs: vec!["a0".into(), "a1".into(), "a2".into(), "n".into()],
            },
            Step::Lets {
                prog: turn,
                prefix: "t".into(),
                vars: turn_vars,
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
    // `Z` (a bivector) as an `{en}`, and the position of the scalar.
    let embed: Vec<String> = (0..k.layout.len())
        .map(|pos| {
            let m = k.layout.blades[pos].0;
            match bv.layout.position(m) {
                Some((i, sb)) => {
                    let se = k.layout.blades[pos].1;
                    if sb * se > 0 {
                        format!("z[{i}]")
                    } else {
                        format!("-z[{i}]")
                    }
                }
                None => "T::zero()".into(),
            }
        })
        .collect();
    let (one_pos, one_sign) = k.layout.position(0).expect("a scalar");
    let e0 = if one_sign > 0 { "e0" } else { "-e0" };
    let _ = write!(
        traits,
        "impl<T: gx::Real> {en}<(), T> {{
    /// The invariants `[r0, p1, p2, p3]` and bivectors `[G1, G2, G3]` of the closed-form
    /// logarithm (docs/log6d.md), as one straight-line program.
    #[doc(hidden)]
    #[inline]
    #[allow(unused_variables)]
    pub fn log_invariants(self) -> ([T; 4], [[T; {n}]; 3]) {{
        let x = self.c;
{lets}        (
            [{}, {}, {}, {}],
            [[{}], [{}], [{}]],
        )
    }}

    /// The logarithm of a unit `{en}` in closed form (docs/log6d.md), right where `⟨R⟩₀ > 0`
    /// and accurate to `ε/⟨R⟩₀`.
    #[doc(hidden)]
    #[inline]
    pub fn log_closed(self) -> {bn}<(), T> {{
        let ([r0, p1, p2, p3], [g1, g2, g3]) = self.log_invariants();
        let [w1, w2, w3] = gx::study::log_weights_6d(p1, p2, p3, r0);
        {bn}::from_coeffs(core::array::from_fn(|i| w1 * g1[i] + w2 * g2[i] + w3 * g3[i]))
    }}
}}

impl<T: gx::Real> gx::Log<{bn}<(), T>> for gx::Unit<{en}<(), T>> {{
    /// The logarithm of a unit versor, in closed form through the invariant decomposition
    /// (docs/log6d.md): `u_j = cosh²(μ_j)` of the three commuting planes are the roots of a
    /// cubic in the scalar parts of `R`'s grade parts squared, and `log R` is
    /// `r0⁻¹ (α2 Q2 + α1 Q1 + α0 ⟨R⟩₂)` for bivectors `Q` from `R`'s grade parts and the
    /// quadratic `α` interpolating `φ(u) = √u asinh(√(u−1))/√(u−1)` at the roots
    /// (`gx::study::log_coeffs_6d`, folded into three weights by `log_weights_6d`). Below
    /// `⟨R⟩₀ = 1/16` (near or past a half turn in some plane), the lanes there first turn the
    /// planes near a half turn by a quarter turn (`gx::study::log_turn_6d`): `log R =
    /// log(R E) + (π/2) Z`, with `Z` the sum of their unit bivectors and `E = ∏(−b̂)`. The
    /// principal logarithm: rotations below a half turn in each invariant plane where
    /// `⟨R⟩₀ > 0`, boosts and dilations of any size.
    #[inline]
    fn log(self) -> {bn}<(), T> {{
        T::vectorize(#[inline(always)] move || {{
        let x = self.into_inner();
        let ([r0, p1, p2, p3], [g1, g2, g3]) = x.log_invariants();
        let [w1, w2, w3] = gx::study::log_weights_6d(p1, p2, p3, r0);
        let closed: [T; {n}] = core::array::from_fn(|i| w1 * g1[i] + w2 * g2[i] + w3 * g3[i]);
        let limit = T::from_ratio(1, 16);
        if T::all_lt(limit, r0) {{
            return {bn}::from_coeffs(closed);
        }}
        let [a0, a1, a2, n] = gx::study::log_turn_6d(p1, p2, p3, r0);
        let z: [T; {n}] = core::array::from_fn(|i| {{
            let q1 = g3[i] - g2[i] + p3 * g1[i];
            let q2 = g2[i] + p1 * q1 - (p3 + p3) * g1[i];
            a2 * q2 + a1 * q1 + a0 * g1[i]
        }});
        let [e0, e1, e2, e3] = gx::study::turn_polynomial(n);
        let ze = {en}::<(), T>::from_coeffs([{}]);
        let z2 = ze * ze;
        let mut e = ze.gp(e1) + z2.gp(e2) + (z2 * ze).gp(e3);
        e.c[{one_pos}] = e.c[{one_pos}] + {e0};
        let turned = (x * e).log_closed();
        let quarter = T::from_f64(core::f64::consts::FRAC_PI_2);
        {bn}::from_coeffs(core::array::from_fn(|i| {{
            T::select_lt(limit, r0, closed[i], turned.c[i] + quarter * z[i])
        }}))
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
        embed.join(", "),
    );
}

/// The closed-form logarithm of an 8D or 9D even versor (four invariant planes;
/// docs/log6d.md §9): `log R = Σ wₘ Gₘ` with `G1 = ⟨R⟩₂` and `Gₘ = r0 ⟨R₂ₘ R₂ₘ₋₂⟩₂`, the
/// weights from `gx::study::log_weights_8d`, and planes near a half turn turned first
/// (`log_turn_8d`). Rust only: the even kind has more coefficients than the WGSL modules take.
#[allow(clippy::too_many_lines)]
fn emit_log_8d(spec: &AlgebraSpec, k: &KindSpec, bv: &KindSpec, traits: &mut String) {
    let alg = &spec.algebra;
    let x = symbolic::variables(&k.layout, 0);
    let grade = |g: u32| -> SymMv {
        x.iter()
            .filter(|(m, _)| m.count_ones() == g)
            .map(|(m, c)| (*m, c.clone()))
            .collect()
    };
    let parts = [grade(2), grade(4), grade(6), grade(8)];
    let r0 = coef(&x, 0);
    let square = |mv: &SymMv| coef(&symbolic::binop(alg, BinOp::Gp, mv, mv), 0);
    let (a1, a2, a3) = (square(&parts[0]), square(&parts[1]), square(&parts[2]));
    let int = |i: i128| Poly::constant(Rational::int(i));
    // p₄₋ⱼ = Σₘ₌₀..ⱼ (−1)ᵐ C(4−m, j−m) Aₘ with Aₘ = ⟨R₂ₘ²⟩₀ and A₀ = r0².
    let p4 = &r0 * &r0;
    let p3 = &(&int(4) * &p4) - &a1;
    let p2 = &(&(&int(6) * &p4) - &(&int(3) * &a1)) + &a2;
    let p1 = &(&(&(&int(4) * &p4) - &(&int(3) * &a1)) + &(&int(2) * &a2)) - &a3;
    let bivectors = |mv: &SymMv| -> Vec<Poly> {
        let two: SymMv = mv
            .iter()
            .filter(|(m, _)| m.count_ones() == 2)
            .map(|(m, c)| (*m, c.clone()))
            .collect();
        symbolic::to_coeffs(&bv.layout, &two).expect("the bivectors")
    };
    let mut outs = vec![r0.clone(), p1, p2, p3, p4];
    outs.extend(bivectors(&parts[0]));
    for m in 1..4 {
        let prod = symbolic::binop(alg, BinOp::Gp, &parts[m], &parts[m - 1]);
        outs.extend(bivectors(&scale_mv(&prod, &r0)));
    }
    let prog = cse::compile_best(&outs, &BTreeSet::new(), &[]);
    let xvar = |v: Var| format!("x[{v}]");
    let mut lets = String::new();
    prog.emit_lets(&xvar, "p", &mut lets);
    let o: Vec<String> = prog
        .outputs
        .iter()
        .map(|op| render(op, &xvar, "p"))
        .collect();
    let nb = bv.layout.len();
    let list = |j: usize| o[5 + j * nb..5 + (j + 1) * nb].join(", ");
    let (en, bn) = (&k.name, &bv.name);
    let sum = |w: &str| {
        (0..4)
            .map(|j| format!("{w}[{j}] * g[{j}][i]"))
            .collect::<Vec<_>>()
            .join(" + ")
    };
    let (closed_sum, z_sum) = (sum("w"), sum("wz"));
    let _ = write!(
        traits,
        "impl<T: gx::Real> {en}<(), T> {{
    /// The invariants `[r0, p1, p2, p3, p4]` and bivectors `[G1, G2, G3, G4]` of the
    /// closed-form logarithm (docs/log6d.md §9), as one straight-line program.
    #[doc(hidden)]
    #[inline]
    #[allow(unused_variables)]
    pub fn log_invariants(self) -> ([T; 5], [[T; {nb}]; 4]) {{
        let x = self.c;
{lets}        (
            [{}, {}, {}, {}, {}],
            [[{}], [{}], [{}], [{}]],
        )
    }}

    /// The logarithm of a unit `{en}` in closed form (docs/log6d.md §9), right where
    /// `⟨R⟩₀ > 0` and accurate to `ε/⟨R⟩₀`.
    #[doc(hidden)]
    #[inline]
    pub fn log_closed(self) -> {bn}<(), T> {{
        let ([r0, p1, p2, p3, p4], g) = self.log_invariants();
        let w = gx::study::log_weights_8d([p1, p2, p3, p4], r0);
        {bn}::from_coeffs(core::array::from_fn(|i| {closed_sum}))
    }}
}}

impl<T: gx::Real> gx::Log<{bn}<(), T>> for gx::Unit<{en}<(), T>> {{
    /// The logarithm of a unit versor, in closed form through the invariant decomposition
    /// (docs/log6d.md §9): `u_j = cosh²(μ_j)` of the four commuting planes are the roots of a
    /// quartic in the scalar parts of `R`'s grade parts squared, and `log R` is
    /// `r0⁻¹ Σ αᵢ Qᵢ` for bivectors `Q` from `R`'s grade parts and the cubic `α` interpolating
    /// `φ(u) = √u asinh(√(u−1))/√(u−1)` at the roots (`gx::study::log_coeffs_8d`, folded into
    /// four weights by `log_weights_8d`). Below `⟨R⟩₀ = 1/16` (near or past a half turn in some
    /// plane), the lanes there first turn the planes near a half turn by a quarter turn
    /// (`gx::study::log_turn_8d`): `log R = log(R E) + (π/2) Z`, with `Z` the sum of their unit
    /// bivectors and `E = ∏(−b̂)`. The principal logarithm: rotations below a half turn in each
    /// invariant plane where `⟨R⟩₀ > 0`, boosts and dilations of any size.
    #[inline]
    fn log(self) -> {bn}<(), T> {{
        T::vectorize(#[inline(always)] move || {{
        let x = self.into_inner();
        let ([r0, p1, p2, p3, p4], g) = x.log_invariants();
        let p = [p1, p2, p3, p4];
        let w = gx::study::log_weights_8d(p, r0);
        let closed: [T; {nb}] = core::array::from_fn(|i| {closed_sum});
        let limit = T::from_ratio(1, 16);
        if T::all_lt(limit, r0) {{
            return {bn}::from_coeffs(closed);
        }}
        let [a0, a1, a2, a3, n] = gx::study::log_turn_8d(p, r0);
        let wz = gx::study::q_weights_8d([a0, a1, a2, a3], p, T::one());
        let z: [T; {nb}] = core::array::from_fn(|i| {z_sum});
        let zb = {bn}::<(), T>::from_coeffs(z);
        // R E = Σ eₖ R Zᵏ, E = ∏(−b̂) as a polynomial in Z: products by the bivector only.
        let e = gx::study::turn_polynomial_8d(n);
        let mut power = x;
        let mut turned = x.gp(e[0]);
        for ek in &e[1..] {{
            power = power * zb;
            turned = turned + power.gp(*ek);
        }}
        let turned = turned.log_closed();
        let quarter = T::from_f64(core::f64::consts::FRAC_PI_2);
        {bn}::from_coeffs(core::array::from_fn(|i| {{
            T::select_lt(limit, r0, closed[i], turned.c[i] + quarter * z[i])
        }}))
        }})
    }}
}}

",
        o[0],
        o[1],
        o[2],
        o[3],
        o[4],
        list(0),
        list(1),
        list(2),
        list(3),
    );
}

/// The exponential of a bivector in closed form from 6D to 9D (three or four invariant planes;
/// docs/log6d.md §12), for the full grade-2 kind of an algebra with a full even kind. The
/// invariants are one straight-line program from the product tables: the wedge powers
/// `Wₘ = B^∧m/m!`, `eₘ = ⟨Wₘ²⟩₀` and `Hₘ = ⟨Wₘ Wₘ₋₁⟩₂`. Then `exp B = C (1 + T + W₂(T) + …)`
/// with `T = Σ wₘ Hₘ` (`gax::study::exp_weights_6d`), turning rotations beyond a quarter turn
/// back by one and halving beyond three quarters. For three planes, also the WGSL kernels
/// `{k}_exp_reach`, `{k}_exp_from` and `{k}_exp_turning`, which the module's `{k}_exp` composes
/// (`emit_wgsl::closed_exp`).
#[allow(clippy::too_many_lines)]
fn emit_exp_closed(
    spec: &AlgebraSpec,
    k: &KindSpec,
    body: &mut String,
    kernels: &mut Vec<Kernel>,
) -> Option<String> {
    use crate::slp::{Operand, Program};
    use crate::table::{Layout, blade_binop};
    let alg = &spec.algebra;
    let n = alg.dim();
    if !(6..=9).contains(&n) {
        return None;
    }
    let planes = n / 2;
    let grade = |g: u32| -> Layout {
        Layout {
            blades: (0..alg.blade_count() as u32)
                .filter(|m| m.count_ones() == g)
                .map(|m| (m, 1))
                .collect(),
        }
    };
    let even_blades: BTreeSet<u32> = (0..alg.blade_count() as u32)
        .filter(|m| m.count_ones() % 2 == 0)
        .collect();
    let e = spec.kind_for_support(&even_blades)?.clone();
    if e.layout.len() != even_blades.len() {
        return None;
    }
    let nb = k.layout.len();
    if nb != grade(2).blades.len() {
        return None;
    }
    let mut b = cse::Builder::default();
    let int = |c: i64, d: i64| Rational::new(i128::from(c), i128::from(d));
    // W₁ = B on the grade-2 blades in mask order.
    let l2 = grade(2);
    let w1: Vec<Operand> = l2
        .blades
        .iter()
        .map(|&(m, _)| {
            let (pos, sign) = k.layout.position(m).expect("a bivector blade");
            b.mul(Operand::Const(int(sign, 1)), Operand::Var(pos as Var))
        })
        .collect();
    // A product of two graded values, kept on the blades of grade `out` (in mask order).
    let product = |b: &mut cse::Builder,
                   op: BinOp,
                   (la, xa): (&Layout, &[Operand]),
                   (lb, xb): (&Layout, &[Operand]),
                   out: &Layout,
                   scale: Rational|
     -> Vec<Operand> {
        let mut sums: Vec<Vec<(Rational, Operand)>> = vec![Vec::new(); out.blades.len()];
        for (i, &(ma, _)) in la.blades.iter().enumerate() {
            for (j, &(mb, _)) in lb.blades.iter().enumerate() {
                for (m, c) in blade_binop(alg, op, ma, mb) {
                    if c == 0 {
                        continue;
                    }
                    if let Some((o, _)) = out.position(m) {
                        let t = b.mul(xa[i], xb[j]);
                        sums[o].push((scale * int(c, 1), t));
                    }
                }
            }
        }
        sums.iter().map(|s| cse::emit_sum(b, s)).collect()
    };
    let mut layouts = vec![grade(0), l2.clone()];
    let mut w = vec![vec![Operand::Const(Rational::ONE)], w1.clone()];
    for m in 2..=planes {
        let lm = grade(2 * m as u32);
        let wm = product(
            &mut b,
            BinOp::Wedge,
            (&layouts[m - 1], &w[m - 1]),
            (&l2, &w1),
            &lm,
            int(1, m as i64),
        );
        layouts.push(lm);
        w.push(wm);
    }
    let l0 = grade(0);
    let mut outs: Vec<Operand> = Vec::new();
    for m in 1..=planes {
        let sq = product(
            &mut b,
            BinOp::Gp,
            (&layouts[m], &w[m]),
            (&layouts[m], &w[m]),
            &l0,
            Rational::ONE,
        );
        outs.push(sq[0]);
    }
    for m in 1..=planes {
        let h = if m == 1 {
            w1.clone()
        } else {
            product(
                &mut b,
                BinOp::Gp,
                (&layouts[m], &w[m]),
                (&layouts[m - 1], &w[m - 1]),
                &l2,
                Rational::ONE,
            )
        };
        // In the bivector kind's layout.
        for &(mb, sb) in &k.layout.blades {
            let (pos, _) = l2.position(mb).expect("a grade-2 blade");
            outs.push(b.mul(Operand::Const(int(sb, 1)), h[pos]));
        }
    }
    for wm in &w[2..=planes] {
        outs.extend(wm.iter().copied());
    }
    let mut prog = b.prog;
    prog.outputs = outs;
    if planes == 3 {
        // The WGSL kernels: the invariants `e` and `H` (step 0), a Study helper, and a program
        // reading the helper's results as locals and the `H`s as outputs of step 0: its
        // variables are the helper's results (`0..4`), then the `H`s.
        let ks = snake(&k.name);
        let mut pre = prog.clone();
        pre.outputs.truncate(planes + planes * nb);
        pre.compact();
        let first = Step::Lets {
            prog: pre,
            prefix: "p".into(),
            vars: arg_vars(nb),
        };
        let study = |func: StudyFn, outs: [&str; 4]| Step::Study {
            func,
            args: (0..planes).map(|m| (0, m)).collect(),
            outs: outs
                .iter()
                .filter(|o| !o.is_empty())
                .map(|o| (*o).to_string())
                .collect(),
        };
        let vars = |names: [&str; 4]| -> Vec<(Var, Source)> {
            let mut v: Vec<(Var, Source)> = names
                .iter()
                .enumerate()
                .filter(|(_, n)| !n.is_empty())
                .map(|(i, n)| (i as Var, Source::Local((*n).to_string())))
                .collect();
            v.extend((0..planes * nb).map(|o| {
                (
                    (4 + o) as Var,
                    Source::Output {
                        step: 0,
                        index: planes + o,
                    },
                )
            }));
            v
        };
        // `Σ wₘ Hₘ` in the bivector kind's layout, the weights in variables `0..3`.
        let weighted = |b: &mut cse::Builder| -> Vec<Operand> {
            (0..nb)
                .map(|i| {
                    let terms: Vec<(Rational, Operand)> = (0..planes)
                        .map(|m| {
                            let h = Operand::Var((4 + m * nb + i) as Var);
                            (Rational::ONE, b.mul(Operand::Var(m as Var), h))
                        })
                        .collect();
                    cse::emit_sum(b, &terms)
                })
                .collect()
        };
        // `C (1 + T + W₂(T) + W₃(T))` in the even kind, `T = Σ wₘ Hₘ`.
        let mut bf = cse::Builder::default();
        let t = weighted(&mut bf);
        let t1: Vec<Operand> = l2
            .blades
            .iter()
            .map(|&(m, _)| {
                let (pos, sign) = k.layout.position(m).expect("a bivector blade");
                bf.mul(Operand::Const(int(sign, 1)), t[pos])
            })
            .collect();
        let mut tw = vec![vec![Operand::Const(Rational::ONE)], t1.clone()];
        for m in 2..=planes {
            let wm = product(
                &mut bf,
                BinOp::Wedge,
                (&layouts[m - 1], &tw[m - 1]),
                (&l2, &t1),
                &layouts[m],
                int(1, m as i64),
            );
            tw.push(wm);
        }
        let c = Operand::Var(3);
        let assembled: Vec<Operand> = e
            .layout
            .blades
            .iter()
            .map(|&(m, sign)| {
                let g = m.count_ones() as usize / 2;
                let (pos, _) = layouts[g].position(m).expect("a blade of that grade");
                let v = bf.mul(c, tw[g][pos]);
                bf.mul(Operand::Const(int(sign, 1)), v)
            })
            .collect();
        let mut from = bf.prog;
        from.outputs = assembled;
        let mut bz = cse::Builder::default();
        let z = weighted(&mut bz);
        let mut turning = bz.prog;
        turning.outputs = z;
        let pick = Program {
            outputs: vec![Operand::Var(0), Operand::Var(1)],
            ..Program::default()
        };
        let lets = |prog: Program, names: [&str; 4]| Step::Lets {
            prog,
            prefix: "t".into(),
            vars: vars(names),
        };
        kernels.extend(value_kernel(
            k,
            &format!("{ks}_exp_reach"),
            &format!(
                "`[reach, turn]` of the closed-form exponential (docs/log6d.md §12): above 1, `reach` asks `{ks}_exp` to halve `x`, and `turn` to turn rotations beyond a quarter turn back first."
            ),
            Ty::Vec(2),
            vec![
                first.clone(),
                study(StudyFn::Exp6Reach, ["reach", "turn", "", ""]),
                lets(pick, ["reach", "turn", "", ""]),
            ],
        ));
        kernels.extend(value_kernel(
            k,
            &format!("{ks}_exp_from"),
            &format!(
                "The exponential in closed form (docs/log6d.md §12), for rotations within about a quarter turn and rapidities up to 8: `{ks}_exp` calls it."
            ),
            Ty::Kind(e.name.clone()),
            vec![
                first.clone(),
                study(StudyFn::Exp6, ["w1", "w2", "w3", "c"]),
                lets(from, ["w1", "w2", "w3", "c"]),
            ],
        ));
        kernels.extend(value_kernel(
            k,
            &format!("{ks}_exp_turning"),
            &format!(
                "The sum `Z` of the unit bivectors of the rotations beyond a quarter turn, which `{ks}_exp` turns back by one (docs/log6d.md §12); `-<Z Z>_0` is their number."
            ),
            Ty::Kind(k.name.clone()),
            vec![
                first,
                study(StudyFn::Exp6Turn, ["z1", "z2", "z3", "n"]),
                lets(turning, ["z1", "z2", "z3", ""]),
            ],
        ));
    }
    let xvar = |v: Var| format!("x[{v}]");
    let mut lets = String::new();
    prog.emit_lets(&xvar, "p", &mut lets);
    let o: Vec<String> = prog
        .outputs
        .iter()
        .map(|op| render(op, &xvar, "p"))
        .collect();
    let mut at = 0;
    let mut take = |len: usize| {
        let v = o[at..at + len].join(", ");
        at += len;
        v
    };
    let e_list = take(planes);
    let h_list: Vec<String> = (0..planes).map(|_| format!("[{}]", take(nb))).collect();
    let w_lens: Vec<usize> = (2..=planes).map(|m| layouts[m].blades.len()).collect();
    let w_lists: Vec<String> = w_lens
        .iter()
        .map(|&len| format!("[{}]", take(len)))
        .collect();
    let w_types: Vec<String> = w_lens.iter().map(|len| format!("[T; {len}]")).collect();
    let w_names: Vec<String> = (2..=planes).map(|m| format!("w{m}")).collect();
    // Even from the grade parts: 1, T, W₂(T), …, scaled by C.
    let assemble: Vec<String> = e
        .layout
        .blades
        .iter()
        .map(|&(m, sign)| {
            let g = m.count_ones() as usize / 2;
            let neg = if sign < 0 { "-" } else { "" };
            match g {
                0 => format!("{neg}c"),
                1 => {
                    let (pos, sb) = k.layout.position(m).expect("a bivector blade");
                    let neg = if sign * sb < 0 { "-" } else { "" };
                    format!("{neg}c * t.c[{pos}]")
                }
                _ => {
                    let (pos, _) = layouts[g].position(m).expect("a blade of that grade");
                    format!("{neg}c * w{g}[{pos}]")
                }
            }
        })
        .collect();
    let (en, bn) = (&e.name, &k.name);
    let suffix = if planes == 3 { "6d" } else { "8d" };
    let tp = if planes == 3 {
        "gx::study::turn_polynomial(n)"
    } else {
        "gx::study::turn_polynomial_8d(n)"
    };
    let sum_h = |w: &str| {
        (0..planes)
            .map(|m| format!("{w}[{m}] * h[{m}][i]"))
            .collect::<Vec<_>>()
            .join(" + ")
    };
    let (t_sum, z_sum) = (sum_h("wt"), sum_h("wz"));
    let zw = (0..planes)
        .map(|m| format!("tz[{m}]"))
        .collect::<Vec<_>>()
        .join(", ");
    let _ = write!(
        body,
        "    /// The invariants of the closed-form exponential (docs/log6d.md §12): `[e1, …]` with
    /// `eₘ = ⟨Wₘ²⟩₀`, the bivectors `Hₘ = ⟨Wₘ Wₘ₋₁⟩₂`, and the wedge powers `Wₘ = B^∧m/m!`
    /// (`m ≥ 2`, the blades of grade `2m` in mask order), as one straight-line program.
    #[doc(hidden)]
    #[inline]
    #[allow(unused_variables, clippy::type_complexity)]
    pub fn exp_invariants(self) -> ([T; {planes}], [[T; {nb}]; {planes}], {}) {{
        let x = self.c;
{lets}        ([{e_list}], [{}], {})
    }}

    /// `exp` in closed form from this bivector's invariants (`exp_invariants`), for rotations
    /// within about a quarter turn: `C (1 + T + W₂(T) + …)` with `T = Σ tanh(μⱼ) b̂ⱼ`.
    #[doc(hidden)]
    #[inline]
    pub fn exp_from(e: [T; {planes}], h: [[T; {nb}]; {planes}]) -> {en}<(), T> {{
        // T and C = ∏ cosh μⱼ (from the trace of ln cosh √λ: no cancellation for boosts).
        let wt = gx::study::exp_weights_{suffix}(e);
        let c = wt[{planes}];
        let t = {bn}::<(), T>::from_coeffs(core::array::from_fn(|i| {t_sum}));
        let (_, _, {}) = t.exp_invariants();
        {en}::from_coeffs([{}])
    }}

    /// The exponential, a unit versor, in closed form (docs/log6d.md §12): the invariants
    /// `λⱼ = μⱼ²` of the {planes} commuting planes are the roots of a polynomial whose
    /// coefficients are `⟨Wₘ²⟩₀`, `Wₘ = B^∧m/m!`, and `exp B = C (1 + T + T∧T/2 + …)` with
    /// `T = Σ tanh(μⱼ) b̂ⱼ`, an interpolant of `tanh(√λ)/√λ` at the roots applied to
    /// bivectors from `B`'s wedge powers (`gax::study::exp_weights_{suffix}`), and
    /// `C = ∏ cosh μⱼ` from the trace of `ln cosh √λ`. Rotations beyond a quarter turn are
    /// turned back by one first (`exp_turn_{suffix}`: `exp B = exp B' · ∏ êⱼ`, a polynomial in
    /// their sum), and beyond three quarters `B` is halved and the result squared.
    #[inline]
    pub fn exp(self) -> gx::Unit<{en}<(), T>> {{
        T::vectorize(#[inline(always)] move || {{
        let half = T::from_ratio(1, 2);
        let (e, h, ..) = self.exp_invariants();
        let [reach, turn] = gx::study::exp_reach_{suffix}(e);
        let (mut scale, mut s) = (T::one(), 0u32);
        while s < 64 && !T::all_lt(reach * scale, T::one()) {{
            scale = scale * half;
            s += 1;
        }}
        let (b, e, h) = if s == 0 {{
            (self, e, h)
        }} else {{
            let b = self.gp(scale);
            let (e, h, ..) = b.exp_invariants();
            (b, e, h)
        }};
        let mut r = if T::all_lt(turn * scale, T::one()) {{
            Self::exp_from(e, h)
        }} else {{
            let tz = gx::study::exp_turn_{suffix}(e);
            let n = tz[{planes}];
            let wz = [{zw}];
            let z = {bn}::<(), T>::from_coeffs(core::array::from_fn(|i| {z_sum}));
            let quarter = T::from_f64(core::f64::consts::FRAC_PI_2);
            let (e, h, ..) = (b - z.gp(quarter)).exp_invariants();
            let r = Self::exp_from(e, h);
            // ∏ êⱼ = (−1)ⁿ ∏(−êⱼ), a polynomial in Z, applied by products with Z.
            let one = T::one();
            let odd = T::select_lt(n, half, one, T::select_lt(n, T::from_f64(1.5), -one, T::select_lt(n, T::from_f64(2.5), one, T::select_lt(n, T::from_f64(3.5), -one, one))));
            let coeffs = {tp};
            let mut power = r;
            let mut out = r.gp(coeffs[0] * odd);
            for ck in &coeffs[1..] {{
                power = power * z;
                out = out + power.gp(*ck * odd);
            }}
            out
        }};
        for _ in 0..s {{
            r = r * r;
        }}
        gx::Unit::new_unchecked(r)
        }})
    }}

",
        w_types.join(", "),
        h_list.join(", "),
        w_lists.join(", "),
        w_names.join(", "),
        assemble.join(", "),
    );
    Some(en.clone())
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
