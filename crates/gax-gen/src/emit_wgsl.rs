//! WGSL modules of the standard algebras (ADR-028).
//!
//! Each module is plain WGSL, self-contained (the Study-number helpers it uses are included),
//! and therefore also a valid WESL module. It holds:
//!
//! * a struct per kind, `ceil(N/4)` fields `c0, c1, …` of `vec4<f32>` with the coefficients in
//!   the Rust blade order, zero-padded;
//! * constructors (`motor_new`), embeddings between versor kinds (`motor_from_rotor`),
//!   `reverse`, and geometric products of versor kinds (`motor_mul_motor`);
//! * the recorded kernels: sandwiches and their matrices, in plain and `unit_` form, norms,
//!   `normalized`, `renormalize_fast`, and the real-trigonometric `exp` and `log`.
//!
//! Every function body is printed from the same verified [`Program`]s as the Rust code.

use crate::cse;
use crate::emit::{Stats, WGSL_MAX};
use crate::kernel::{Kernel, Source, Step, StudyFn, Ty, snake, vec4s, wgsl_coeff, wgsl_construct};
use crate::poly::{Poly, Var};
use crate::slp::Program;
use crate::spec::{AlgebraSpec, KindSpec};
use crate::symbolic;
use crate::table::{BinOp, UnOp};
use std::collections::BTreeSet;
use std::fmt::Write as _;

/// The WGSL source of a Study-number helper (ports of `gax_core::study`, in `f32`).
#[allow(clippy::too_many_lines)]
pub fn study_source(f: StudyFn) -> String {
    match f {
        StudyFn::ExpRotation => {
            // S' = Σ (-1)^k (k+1)/(2k+3)! a^{2k}, k = 0..=7, by Horner (highest first).
            let ds = [
                "(-8.0 / 355687428096000.0)",
                "(7.0 / 1307674368000.0)",
                "(-6.0 / 6227020800.0)",
                "(5.0 / 39916800.0)",
                "(-4.0 / 362880.0)",
                "(3.0 / 5040.0)",
                "(-2.0 / 120.0)",
                "(1.0 / 6.0)",
            ];
            let mut horner = String::new();
            for (i, c) in ds.iter().enumerate() {
                if i == 0 {
                    let _ = writeln!(horner, "    var ds = {c};");
                } else {
                    let _ = writeln!(horner, "    ds = ds * a2 + {c};");
                }
            }
            format!(
                "// `exp(B) = C + S B` for `B² = lambda + mu I` with `I² = 0` and `lambda <= 0`, as
// `[C, C_I, S, S_I]`: `C = cos a`, `S = sin a / a` with `a = sqrt(-lambda)`, and the `I` parts
// `mu S / 2` and `mu S'`, `S' = (S - C) / (2 a²)` (series near `a = 0`, see gax_core::study).
fn study_exp_rotation(lambda: f32, mu: f32) -> vec4<f32> {{
    let a2 = max(-lambda, 0.0);
    let a = sqrt(a2);
    let sn = sin(a);
    let cs = cos(a);
    let inv = 1.0 / a;
    let s = select(sn * inv, 1.0 + a2 * ((-1.0 / 6.0) + a2 * ((1.0 / 120.0) + a2 * (-1.0 / 5040.0))), a2 < 1e-4);
{horner}    let d = select((s - cs) * inv * inv * 0.5, ds, a2 < 0.25);
    return vec4<f32>(cs, mu * s * 0.5, s, mu * d);
}}
"
            )
        }
        StudyFn::LogRotation => {
            // ∂h/∂u series: (-1)^j (j+1)/(2j+3), j = 0..=11, highest first.
            let mut horner = String::new();
            for j in (0..12u8).rev() {
                let (n, d) = (u32::from(j) + 1, 2 * u32::from(j) + 3);
                let c = if j % 2 == 0 {
                    format!("({n}.0 / {d}.0)")
                } else {
                    format!("(-{n}.0 / {d}.0)")
                };
                if j == 11 {
                    let _ = writeln!(horner, "    var g = {c};");
                } else {
                    let _ = writeln!(horner, "    g = g * t2 + {c};");
                }
            }
            format!(
                "// `log R = h0 P + h1 I P` for a unit versor `R = c + P` with `c = c0 + c1 I`,
// `P² = u0 + u1 I`, `I² = 0`, `u0 <= 0`, as `[h0, h1]`. The rotation half-angle is in
// `[0, pi]`; at `R = -T` (a translation times -1) the result is `log(-R)` (see gax_core::study).
fn study_log_rotation(c0: f32, c1: f32, u0: f32, u1: f32) -> vec2<f32> {{
    let s2 = max(-u0, 0.0);
    let s = sqrt(s2);
    let theta = atan2(s, c0);
    let n = c0 * c0 + s2;
    let ic = 1.0 / c0;
    let t2 = s2 * ic * ic;
    let key = select(select(0.0, 1.0, 0.0 < s2), t2, 0.0 < c0);
    let hs = ic * (1.0 + t2 * ((-1.0 / 3.0) + t2 * ((1.0 / 5.0) + t2 * (-1.0 / 7.0))));
    let h0 = select(theta / s, hs, key < 1e-6);
{horner}    let gs = select((theta - c0 * s / n) / (s2 * s * 2.0), g * ic * ic * ic, key < (1.0 / 25.0));
    return vec2<f32>(h0, -c1 / n + u1 * gs);
}}
"
            )
        }
        StudyFn::RsqrtNil => "// `(a + b I)^(-1/2)` with `I² = 0`, as `[r0, r1]`.
fn study_rsqrt_nil(a: f32, b: f32) -> vec2<f32> {
    let r = 1.0 / sqrt(a);
    return vec2<f32>(r, -(b * r * r * r) * 0.5);
}
"
        .into(),
        StudyFn::Rsqrt(1) => {
            "// `(a + b I)^(-1/2)` with `I² = 1`, as `[r0, r1]`, through the channels `a ± b`.
fn study_rsqrt_split(a: f32, b: f32) -> vec2<f32> {
    let rp = 1.0 / sqrt(a + b);
    let rm = 1.0 / sqrt(a - b);
    return vec2<f32>((rp + rm) * 0.5, (rp - rm) * 0.5);
}
"
            .into()
        }
        StudyFn::Rsqrt(_) => {
            "// `(a + b I)^(-1/2)` with `I² = -1` (principal branch), as `[r0, r1]`.
fn study_rsqrt_complex(a: f32, b: f32) -> vec2<f32> {
    let m = sqrt(a * a + b * b);
    let p = sqrt(0.5 * (m + abs(a)));
    let q = b / (2.0 * p);
    let sb = select(1.0, -1.0, b < 0.0);
    let wr = select(abs(q), p, a >= 0.0);
    let wi = select(sb * p, q, a >= 0.0);
    return vec2<f32>(wr / m, -wi / m);
}
"
            .into()
        }
        StudyFn::RsqrtAbs => "// `1 / sqrt(|a|)`.
fn study_rsqrt_abs(a: f32) -> f32 {
    return 1.0 / sqrt(abs(a));
}
"
        .into(),
    }
}

/// Coefficient expressions of kind `to` taken from a value `x` of kind `from` by blade (with
/// orientation signs), zero where `from` lacks the blade; `None` if `from` has a blade that
/// `to` lacks.
fn embedding(from: &KindSpec, to: &KindSpec, x: &str) -> Option<Vec<String>> {
    if from
        .layout
        .blades
        .iter()
        .any(|(m, _)| to.layout.position(*m).is_none())
    {
        return None;
    }
    Some(
        to.layout
            .blades
            .iter()
            .map(|(m, s)| match from.layout.position(*m) {
                Some((i, fs)) => {
                    let e = wgsl_coeff(x, i);
                    if fs * s > 0 { e } else { format!("-{e}") }
                }
                None => "0.0".into(),
            })
            .collect(),
    )
}

/// A two-parameter product kernel `a op b`, or `None` when the result is zero or has no kind.
fn product(
    spec: &AlgebraSpec,
    op: BinOp,
    a: &KindSpec,
    b: &KindSpec,
    name: &str,
    doc: &str,
) -> Option<Kernel> {
    let alg = &spec.algebra;
    let na = a.layout.len();
    let x = symbolic::variables(&a.layout, 0);
    let y = symbolic::variables(&b.layout, na as Var);
    let res = symbolic::binop(alg, op, &x, &y);
    let support = symbolic::support(&res);
    if support.is_empty() {
        return None;
    }
    let out = spec.kind_for_support(&support)?;
    if out.layout.len() > WGSL_MAX {
        return None;
    }
    let coeffs = symbolic::to_coeffs(&out.layout, &res)?;
    let prog = cse::compile_best(&coeffs, &BTreeSet::new(), &[]);
    let vars = (0..na + b.layout.len())
        .map(|i| {
            let src = if i < na {
                Source::Arg { param: 0, index: i }
            } else {
                Source::Arg {
                    param: 1,
                    index: i - na,
                }
            };
            (i as Var, src)
        })
        .collect();
    Some(Kernel {
        name: name.into(),
        doc: doc.into(),
        params: vec![
            ("a".into(), Ty::Kind(a.name.clone())),
            ("b".into(), Ty::Kind(b.name.clone())),
        ],
        result: Ty::Kind(out.name.clone()),
        steps: vec![Step::Lets {
            prog,
            prefix: "t".into(),
            vars,
        }],
        entries: None,
    })
}

/// A one-parameter kernel from coefficient polynomials of kind `out` in the input's variables.
fn unary(k: &KindSpec, out: &KindSpec, coeffs: &[Poly], name: &str, doc: &str) -> Kernel {
    let prog: Program = cse::compile_best(coeffs, &BTreeSet::new(), &[]);
    Kernel {
        name: name.into(),
        doc: doc.into(),
        params: vec![("x".into(), Ty::Kind(k.name.clone()))],
        result: Ty::Kind(out.name.clone()),
        steps: vec![Step::Lets {
            prog,
            prefix: "t".into(),
            vars: (0..k.layout.len())
                .map(|i| (i as Var, Source::Arg { param: 0, index: i }))
                .collect(),
        }],
        entries: None,
    }
}

/// The WGSL module of an algebra, from its spec and the kernels the Rust emitter recorded.
///
/// `fma` fuses single-use products into `fma(a, b, c)`.
#[allow(clippy::too_many_lines)]
pub fn module(spec: &AlgebraSpec, stats: &Stats, fma: bool) -> String {
    let name = &spec.name;
    let mut s = String::new();
    let _ = write!(
        s,
        "// @generated by gax-regen from the `{name}` spec. Do not edit by hand.
//
// gax WGSL module `gax::{name}` (plain WGSL, also valid WESL; see gax's docs/shaders.md).
//
// * Each kind is a struct of `ceil(N/4)` `vec4<f32>` fields `c0, c1, ...` holding its N
//   coefficients in the Rust blade order (listed per struct), zero-padded. The Rust side has
//   the same layout (`{{Kind}}Gpu`, feature `bytemuck`).
// * Function names: `motor_new`, `motor_from_rotor`, `motor_reverse`, `motor_mul_point`
//   (geometric product), `motor_sandwich_point` (`v x ~v`), `unit_motor_sandwich_point` (for
//   `v ~v = 1`), `motor_matrix_point` (the matrix of the sandwich, applied as `m * x`),
//   `motor_normalized`, `motor_renormalize_fast`, `line_exp`, `unit_motor_log`,
//   `motor_norm_squared`.
// * Everything is `f32`. Kernels are the Rust kernels' verified straight-line programs.

"
    );
    let small: Vec<&KindSpec> = spec
        .kinds
        .iter()
        .filter(|k| k.layout.len() <= WGSL_MAX)
        .collect();
    for k in &spec.kinds {
        let n = k.layout.len();
        let _ = writeln!(s, "// `{}`: [{}]", k.name, k.blades.join(", "));
        let _ = writeln!(s, "struct {} {{", k.name);
        for f in 0..vec4s(n) {
            let _ = writeln!(s, "    c{f}: vec4<f32>,");
        }
        let _ = writeln!(s, "}}\n");
    }
    for (alias, kind) in &spec.aliases {
        let _ = writeln!(s, "alias {alias} = {kind};\n");
    }
    // Constructors.
    for k in &small {
        let params: Vec<String> = k
            .blades
            .iter()
            .map(|b| format!("{}: f32", if b == "1" { "s" } else { b }))
            .collect();
        let parts: Vec<String> = k
            .blades
            .iter()
            .map(|b| if b == "1" { "s".into() } else { b.clone() })
            .collect();
        let _ = writeln!(
            s,
            "// A `{}` from its coefficients.\nfn {}_new({}) -> {} {{\n    return {};\n}}\n",
            k.name,
            snake(&k.name),
            params.join(", "),
            k.name,
            wgsl_construct(&k.name, &parts)
        );
    }
    // Embeddings between versor kinds.
    for from in small.iter().filter(|k| k.versor) {
        for to in small.iter().filter(|k| k.versor && k.name != from.name) {
            if let Some(parts) = embedding(from, to, "x") {
                let _ = writeln!(
                    s,
                    "// A `{}` as a `{}`.\nfn {}_from_{}(x: {}) -> {} {{\n    return {};\n}}\n",
                    from.name,
                    to.name,
                    snake(&to.name),
                    snake(&from.name),
                    from.name,
                    to.name,
                    wgsl_construct(&to.name, &parts)
                );
            }
        }
    }
    let kernels = kernels(spec, stats);
    let mut used: BTreeSet<&'static str> = BTreeSet::new();
    let mut helpers = Vec::new();
    for k in &kernels {
        let _ = writeln!(s, "{}", k.wgsl(fma, ""));
        for st in &k.steps {
            if let Step::Study { func, .. } = st
                && used.insert(func.wgsl_name())
            {
                helpers.push(*func);
            }
        }
    }
    helpers.sort_by_key(|f| f.wgsl_name());
    for f in helpers {
        let _ = writeln!(s, "{}", study_source(f));
    }
    s
}

/// Every kernel of an algebra's WGSL module, in module order: `reverse`, the products of
/// versor kinds, the value methods and the sandwiches (constructors and embeddings, which are
/// not programs, are not included).
pub fn kernels(spec: &AlgebraSpec, stats: &Stats) -> Vec<Kernel> {
    let alg = &spec.algebra;
    let small: Vec<&KindSpec> = spec
        .kinds
        .iter()
        .filter(|k| k.layout.len() <= WGSL_MAX)
        .collect();
    let mut kernels: Vec<Kernel> = Vec::new();
    for k in &small {
        let x = symbolic::variables(&k.layout, 0);
        let r = symbolic::unop(alg, UnOp::Reverse, &x);
        if let Some(coeffs) = symbolic::to_coeffs(&k.layout, &r) {
            kernels.push(unary(
                k,
                k,
                &coeffs,
                &format!("{}_reverse", snake(&k.name)),
                "The reverse `~x`.",
            ));
        }
    }
    for a in small.iter().filter(|k| k.versor) {
        for b in small.iter().filter(|k| k.versor) {
            kernels.extend(product(
                spec,
                BinOp::Gp,
                a,
                b,
                &format!("{}_mul_{}", snake(&a.name), snake(&b.name)),
                &format!(
                    "The geometric product of a `{}` and a `{}` (composition).",
                    a.name, b.name
                ),
            ));
        }
    }
    kernels.extend(stats.values.iter().flat_map(|v| v.kernels.iter().cloned()));
    kernels.extend(stats.kernels.iter().cloned());
    kernels
}

/// The WGSL text with `public` before each top-level declaration, for WESL resolvers that
/// implement visibility (the `wesl` crate from 0.5): imports across packages need it.
pub fn with_public(src: &str) -> String {
    let mut out = String::with_capacity(src.len() + src.len() / 16);
    for line in src.lines() {
        if line.starts_with("fn ") || line.starts_with("struct ") || line.starts_with("alias ") {
            out.push_str("public ");
        }
        out.push_str(line);
        out.push('\n');
    }
    out
}
