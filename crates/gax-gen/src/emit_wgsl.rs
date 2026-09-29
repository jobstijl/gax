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
use crate::kernel::{
    Kernel, Precision, Source, Step, StudyFn, Ty, snake, vec4s, wgsl_coeff, wgsl_construct_in,
};
use crate::poly::{Poly, Var};
use crate::slp::Program;
use crate::spec::{AlgebraSpec, KindSpec};
use crate::symbolic;
use crate::table::{BinOp, UnOp};
use std::collections::BTreeSet;
use std::fmt::Write as _;

/// Complex numbers (`vec2<f32>`, re and im) and dual numbers over them (a value and a
/// derivative), with the elementary functions the general Study helpers need: a port of
/// `gax_core::study`'s `Cx`, `Dual` and `Channel`. Included once when a module uses a helper
/// that [`StudyFn::needs_channels`].
pub const CHANNELS: &str = "// Complex numbers as `vec2<f32>` (re, im) and dual numbers over them (value `p`, derivative
// `d`): the channel arithmetic of gax_core::study, for the general Study helpers below.
struct StudyDual {
    p: vec2<f32>,
    d: vec2<f32>,
}

fn cx_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

fn cx_div(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let d = 1.0 / (b.x * b.x + b.y * b.y);
    return vec2<f32>((a.x * b.x + a.y * b.y) * d, (a.y * b.x - a.x * b.y) * d);
}

// The principal root without cancellation: the larger part `t = sqrt((|z| + |a|) / 2)`, the
// other `b / (2t)` (the form of gax_core::study::Cx::sqrt).
fn cx_sqrt(z: vec2<f32>) -> vec2<f32> {
    let r = sqrt(z.x * z.x + z.y * z.y);
    let t = sqrt(max((r + abs(z.x)) * 0.5, 0.0));
    let other = z.y / (2.0 * select(1.0, t, 0.0 < t));
    let signed = select(t, -t, z.y < 0.0);
    return select(vec2<f32>(t, other), vec2<f32>(abs(other), signed), z.x < 0.0);
}

fn cx_ln(z: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(log(z.x * z.x + z.y * z.y) * 0.5, atan2(z.y, z.x));
}

fn cx_sinh(z: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(sinh(z.x) * cos(z.y), cosh(z.x) * sin(z.y));
}

fn cx_cosh(z: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(cosh(z.x) * cos(z.y), sinh(z.x) * sin(z.y));
}

fn sd_real(x: f32) -> StudyDual {
    return StudyDual(vec2<f32>(x, 0.0), vec2<f32>(0.0, 0.0));
}

fn sd_add(a: StudyDual, b: StudyDual) -> StudyDual {
    return StudyDual(a.p + b.p, a.d + b.d);
}

fn sd_sub(a: StudyDual, b: StudyDual) -> StudyDual {
    return StudyDual(a.p - b.p, a.d - b.d);
}

fn sd_scale(a: StudyDual, k: f32) -> StudyDual {
    return StudyDual(a.p * k, a.d * k);
}

fn sd_mul(a: StudyDual, b: StudyDual) -> StudyDual {
    return StudyDual(cx_mul(a.p, b.p), cx_mul(a.p, b.d) + cx_mul(a.d, b.p));
}

fn sd_div(a: StudyDual, b: StudyDual) -> StudyDual {
    let q = cx_div(a.p, b.p);
    return StudyDual(q, cx_div(a.d - cx_mul(q, b.d), b.p));
}

fn sd_sqrt(a: StudyDual) -> StudyDual {
    let r = cx_sqrt(a.p);
    return StudyDual(r, cx_div(a.d, r + r));
}

fn sd_ln(a: StudyDual) -> StudyDual {
    return StudyDual(cx_ln(a.p), cx_div(a.d, a.p));
}

fn sd_sinh(a: StudyDual) -> StudyDual {
    return StudyDual(cx_sinh(a.p), cx_mul(a.d, cx_cosh(a.p)));
}

fn sd_cosh(a: StudyDual) -> StudyDual {
    return StudyDual(cx_cosh(a.p), cx_mul(a.d, cx_sinh(a.p)));
}

// `if |x|² < t { a } else { b }` on the value's squared magnitude.
fn sd_select_small(x: StudyDual, t: f32, a: StudyDual, b: StudyDual) -> StudyDual {
    let small = dot(x.p, x.p) < t;
    return StudyDual(select(b.p, a.p, small), select(b.d, a.d, small));
}

// Every function below computes its direct form at a stand-in argument wherever its series is
// selected, so a discarded branch never divides by zero (`sqrt` has an infinite derivative at
// 0): the result is the same, and there are no non-finite intermediates to rely on discarding.

// `C(x) = cosh(sqrt x)`, with its series where `|x|² < 1/100` (thresholds for `f32`).
fn study_exp_c(x: StudyDual) -> StudyDual {
    let c = sd_cosh(sd_sqrt(sd_select_small(x, 0.01, sd_real(1.0), x)));
    let x2 = sd_mul(x, x);
    let x3 = sd_mul(x2, x);
    var cs = sd_add(sd_real(1.0), sd_scale(x, 1.0 / 2.0));
    cs = sd_add(cs, sd_scale(x2, 1.0 / 24.0));
    cs = sd_add(cs, sd_scale(x3, 1.0 / 720.0));
    cs = sd_add(cs, sd_scale(sd_mul(x3, x), 1.0 / 40320.0));
    return sd_select_small(x, 0.01, cs, c);
}

// `S(x) = sinh(sqrt x) / sqrt x`, with its series where `|x|² < 1/100`.
fn study_exp_s(x: StudyDual) -> StudyDual {
    let r = sd_sqrt(sd_select_small(x, 0.01, sd_real(1.0), x));
    let s = sd_div(sd_sinh(r), r);
    let x2 = sd_mul(x, x);
    let x3 = sd_mul(x2, x);
    var ss = sd_add(sd_real(1.0), sd_scale(x, 1.0 / 6.0));
    ss = sd_add(ss, sd_scale(x2, 1.0 / 120.0));
    ss = sd_add(ss, sd_scale(x3, 1.0 / 5040.0));
    ss = sd_add(ss, sd_scale(sd_mul(x3, x), 1.0 / 362880.0));
    return sd_select_small(x, 0.01, ss, s);
}

// The factor `H` of `log(R) = H <R>_2` for a versor `R = c + P` with `u = P²`:
// `2 atanh(t) / (t (1 + c))`, `t = sqrt(u) / (1 + c)` (see gax_core::study::log_factor).
fn study_log_factor(c: StudyDual, u: StudyDual) -> StudyDual {
    let opc = sd_add(sd_real(1.0), c);
    // `t² = u / (1 + c)²` for the series, without the root; the direct form at `t = 1/2`
    // where the series is selected.
    let t2 = sd_div(u, sd_mul(opc, opc));
    let safe = sd_select_small(t2, 1e-4, sd_scale(sd_mul(opc, opc), 0.25), u);
    let t = sd_div(sd_sqrt(safe), opc);
    let direct = sd_div(sd_scale(sd_ln(sd_div(sd_add(sd_real(1.0), t), sd_sub(sd_real(1.0), t))), 0.5), t);
    var series = sd_real(1.0 / 15.0);
    series = sd_add(sd_mul(series, t2), sd_real(1.0 / 13.0));
    series = sd_add(sd_mul(series, t2), sd_real(1.0 / 11.0));
    series = sd_add(sd_mul(series, t2), sd_real(1.0 / 9.0));
    series = sd_add(sd_mul(series, t2), sd_real(1.0 / 7.0));
    series = sd_add(sd_mul(series, t2), sd_real(1.0 / 5.0));
    series = sd_add(sd_mul(series, t2), sd_real(1.0 / 3.0));
    series = sd_add(sd_mul(series, t2), sd_real(1.0));
    let ratio = sd_select_small(t2, 1e-4, series, direct);
    return sd_div(sd_scale(ratio, 2.0), opc);
}

// `acosh(y)²`, with its series near `y = 1` (`|y - 1|² < 1e-8`).
fn study_acosh_sq(y: StudyDual) -> StudyDual {
    let t = sd_sub(y, sd_real(1.0));
    let ys = sd_select_small(t, 1e-8, sd_real(2.0), y);
    let w = sd_ln(sd_add(ys, sd_sqrt(sd_mul(sd_sub(ys, sd_real(1.0)), sd_add(ys, sd_real(1.0))))));
    let direct = sd_mul(w, w);
    let tt = sd_mul(t, t);
    var series = sd_scale(t, 2.0);
    series = sd_sub(series, sd_scale(tt, 1.0 / 3.0));
    series = sd_add(series, sd_scale(sd_mul(tt, t), 4.0 / 45.0));
    return sd_select_small(t, 1e-8, series, direct);
}
";

/// The WGSL helper `study_log6` and its series functions (a port of
/// `gax_core::study::log_weights_6d`).
const LOG6: &str = r"// The closed-form logarithm of a 6D even versor (docs/log6d.md), a port of
// gax_core::study::log_weights_6d in f32: the invariants cosh²(μ_j) are the roots of
// t³ − p1 t² + p2 t − p3, and φ(u) = √u asinh(√(u−1))/√(u−1) is interpolated at them through its
// Taylor series at their mean (close roots) or an isolated real root and the remaining pair
// (spread roots). Series have 16 terms: the regimes keep the nodes' spread below a quarter of
// their distance to φ's singularity, so 4^-16 ≈ 2e-10. Returns `[w1, w2, w3, 0]` with
// `log R = w1 G1 + w2 G2 + w3 G3`.
fn log6_mul(a_in: array<f32, 16>, b_in: array<f32, 16>) -> array<f32, 16> {
    var a = a_in;
    var b = b_in;
    var out: array<f32, 16>;
    for (var i = 0; i < 16; i++) {
        for (var j = 0; j < 16 - i; j++) {
            out[i + j] += a[i] * b[j];
        }
    }
    return out;
}

fn log6_div(a_in: array<f32, 16>, b_in: array<f32, 16>) -> array<f32, 16> {
    var a = a_in;
    var b = b_in;
    var q: array<f32, 16>;
    let inv = 1.0 / b[0];
    for (var k = 0; k < 16; k++) {
        var s = a[k];
        for (var j = 1; j <= k; j++) {
            s -= b[j] * q[k - j];
        }
        q[k] = s * inv;
    }
    return q;
}

// `(c + sigma t)^e` as a series in `t`, `e = ±1/2`, `c > 0`.
fn log6_pow(c: f32, sigma: f32, e: f32) -> array<f32, 16> {
    var out: array<f32, 16>;
    var binom = select(sqrt(c), inverseSqrt(c), e < 0.0);
    let step = sigma / c;
    for (var k = 0; k < 16; k++) {
        out[k] = binom;
        binom *= (e - f32(k)) / (f32(k) + 1.0) * step;
    }
    return out;
}

// `F(x) = asinh(√x)/√x` (or `asin(√−x)/√−x`) as a series in `t` at `x0 = s x`, `s = ±1`,
// `x >= 1/4`, from `v = √(x0 + t)` and `asinh(v)' = v'/√(1 + x0 + t)`.
fn log6_far(x: f32, s: f32) -> array<f32, 16> {
    let m = max(1.0 + s * x, 1e-30);
    var v = log6_pow(x, s, 0.5);
    let r = log6_pow(m, 1.0, -0.5);
    var dv: array<f32, 16>;
    for (var k = 1; k < 16; k++) {
        dv[k - 1] = v[k] * f32(k);
    }
    var d = log6_mul(dv, r);
    var a: array<f32, 16>;
    a[0] = select(atan2(v[0], sqrt(m)), log(v[0] + sqrt(v[0] * v[0] + 1.0)), s > 0.0);
    for (var k = 1; k < 16; k++) {
        a[k] = d[k - 1] / f32(k);
    }
    return log6_div(a, v);
}

// The Taylor series of `φ(u) = √u F(u − 1)` at `u = m`.
fn log6_phi_series(m: f32) -> array<f32, 16> {
    let x0 = m - 1.0;
    var f: array<f32, 16>;
    if abs(x0) < 0.25 {
        // F's Maclaurin series (48 terms) shifted to x0 by repeated Horner.
        var c: array<f32, 48>;
        var b = 1.0;
        for (var n = 0; n < 48; n++) {
            c[n] = select(-b, b, n % 2 == 0) / (2.0 * f32(n) + 1.0);
            b *= (2.0 * f32(n) + 1.0) / (2.0 * f32(n) + 2.0);
        }
        for (var i = 0; i < 16; i++) {
            for (var j = 46; j >= i; j--) {
                c[j] += x0 * c[j + 1];
            }
            f[i] = c[i];
        }
    } else if x0 > 0.0 {
        f = log6_far(x0, 1.0);
    } else {
        f = log6_far(-x0, -1.0);
    }
    return log6_mul(log6_pow(m, 1.0, 0.5), f);
}

// `φ(u)` at a real `u > 0`.
fn log6_phi(u: f32) -> f32 {
    let x = u - 1.0;
    let ax = abs(x);
    let r = sqrt(ax);
    var f: f32;
    if ax < 0.05 {
        // The Maclaurin series of F, 8 terms (0.05^8 < 1e-10).
        f = ((((((-143.0 / 10240.0 * x + 231.0 / 13312.0) * x - 63.0 / 2816.0) * x + 35.0 / 1152.0) * x - 5.0 / 112.0) * x + 3.0 / 40.0) * x - 1.0 / 6.0) * x + 1.0;
    } else if x > 0.0 {
        f = log(r + sqrt(ax + 1.0)) / r;
    } else {
        f = atan2(r, sqrt(max(1.0 - ax, 0.0))) / r;
    }
    return sqrt(max(u, 0.0)) * f;
}

// `φ(u)` at a complex `u` (a conjugate pair of roots far apart).
fn log6_phi_cx(u: vec2<f32>) -> vec2<f32> {
    let s = cx_sqrt(u - vec2<f32>(1.0, 0.0));
    let ash = cx_ln(s + cx_sqrt(cx_mul(s, s) + vec2<f32>(1.0, 0.0)));
    return cx_div(cx_mul(cx_sqrt(u), ash), s);
}

// The interpolant `[α0, α1, α2]` of a series at centre `c`, reduced modulo `t³ + e2 t − e3`.
fn log6_reduce3(ph_in: array<f32, 16>, c: f32, e2: f32, e3: f32) -> vec3<f32> {
    var ph = ph_in;
    var q = vec3<f32>(0.0, 0.0, 1.0);
    var sum = vec3<f32>(0.0, 0.0, 0.0);
    for (var k = 0; k < 16; k++) {
        sum += ph[k] * q;
        q = vec3<f32>(q.y, q.z - q.x * e2, q.x * e3);
    }
    return vec3<f32>(sum.x * c * c - sum.y * c + sum.z, sum.y - 2.0 * sum.x * c, sum.x);
}

// The line `[l0, l1]` of a series at centre `c`, reduced modulo `t² − d2`.
fn log6_reduce2(ph_in: array<f32, 16>, c: f32, d2: f32) -> vec2<f32> {
    var ph = ph_in;
    var q = vec2<f32>(0.0, 1.0);
    var sum = vec2<f32>(0.0, 0.0);
    for (var k = 0; k < 16; k++) {
        sum += ph[k] * q;
        q = vec2<f32>(q.y, q.x * d2);
    }
    return vec2<f32>(sum.y - sum.x * c, sum.x);
}

// `sign(x) |x|^(1/3)`.
fn log6_cbrt(x: f32) -> f32 {
    let a = abs(x);
    if a < 1e-30 {
        return 0.0;
    }
    return sign(x) * exp(log(a) / 3.0);
}

fn study_log6(p1: f32, p2: f32, p3: f32, r0: f32) -> vec4<f32> {
    let m = p1 / 3.0;
    let e2 = p2 - 2.0 * m * p1 + 3.0 * m * m;
    let e3 = p3 - m * p2 + m * m * p1 - m * m * m;
    let bound = 2.0 * max(sqrt(abs(e2)), log6_cbrt(abs(e3)));
    var al: vec3<f32>;
    if bound < 0.25 * m {
        al = log6_reduce3(log6_phi_series(m), m, e2, e3);
    } else {
        // An isolated real root r (Cardano, or the most isolated of three), a Newton step.
        let disc = e3 * e3 * 0.25 + e2 * e2 * e2 / 27.0;
        var r: f32;
        if disc >= 0.0 {
            let sq = sqrt(disc);
            r = log6_cbrt(e3 * 0.5 + sq) + log6_cbrt(e3 * 0.5 - sq);
        } else {
            let rad = sqrt(max(-e2 / 3.0, 0.0));
            let pn = min(e2, -1e-30);
            let arg = clamp(-e3 * 1.5 / pn * sqrt(-3.0 / pn), -1.0, 1.0);
            let ang = atan2(sqrt(max(1.0 - arg * arg, 0.0)), arg) / 3.0;
            let third = 2.0943951;
            let s = 2.0 * rad * vec3<f32>(cos(ang), cos(ang - third), cos(ang - 2.0 * third));
            let g = vec3<f32>(
                min(abs(s.x - s.y), abs(s.x - s.z)),
                min(abs(s.y - s.z), abs(s.y - s.x)),
                min(abs(s.z - s.x), abs(s.z - s.y)),
            );
            r = select(select(s.x, s.y, g.x < g.y), s.z, max(g.x, g.y) < g.z);
        }
        r += m;
        let f = ((r - p1) * r + p2) * r - p3;
        let df = (3.0 * r - 2.0 * p1) * r + p2;
        if abs(df) >= 1e-30 {
            r -= f / df;
        }
        // The remaining pair by its sum and product; its line through its series at the
        // midpoint (close) or its two values (real, or a conjugate pair).
        let sum = p1 - r;
        let prod = p2 - r * sum;
        let mid = sum * 0.5;
        let d2 = mid * mid - prod;
        let d = sqrt(abs(d2));
        var pair: vec2<f32>;
        if d < 0.25 * abs(mid) {
            pair = log6_reduce2(log6_phi_series(mid), mid, d2);
        } else if d2 < 0.0 {
            let fc = log6_phi_cx(vec2<f32>(mid, d));
            let slope = fc.y / max(d, 1e-30);
            pair = vec2<f32>(fc.x - slope * mid, slope);
        } else {
            let a = mid + d;
            let b = mid - d;
            let fa = log6_phi(a);
            let slope = (fa - log6_phi(b)) / max(a - b, 1e-30);
            pair = vec2<f32>(fa - slope * a, slope);
        }
        let kk = (log6_phi(r) - (pair.y * r + pair.x)) / ((r - sum) * r + prod);
        al = vec3<f32>(pair.x + prod * kk, pair.y - sum * kk, kk);
    }
    // log R = r0⁻¹ (α2 Q2 + α1 Q1 + α0 G1) = w1 G1 + w2 G2 + w3 G3.
    let b = al.z * p1 + al.y;
    let rinv = 1.0 / r0;
    return vec4<f32>((al.x - 2.0 * p3 * al.z + b * p3) * rinv, (al.z - b) * rinv, b * rinv, 0.0);
}
";

/// A WGSL function `f(a, q) -> vec2<f32>`: the Study function `inner` of `a + X` with `X² = q`
/// (`gax_core::study::study_q`), as `(f0, f1)` with `f(a + X) = f0 + f1 X`.
fn study_q_source(name: &str, inner: &str) -> String {
    format!(
        "// `{inner}` of `a + X` with `X² = q`, as `[f0, f1]` (gax_core::study::study_q).
fn {name}(a: f32, q: f32) -> vec2<f32> {{
    let small = abs(q) < 1e-8;
    let w = cx_sqrt(vec2<f32>(select(q, 1.0, small), 0.0));
    let plus = {inner}(StudyDual(vec2<f32>(a, 0.0) + w, vec2<f32>(0.0, 0.0))).p;
    let minus = {inner}(StudyDual(vec2<f32>(a, 0.0) - w, vec2<f32>(0.0, 0.0))).p;
    let f0 = (plus.x + minus.x) * 0.5;
    let diff = cx_div(plus - minus, w + w);
    let d = {inner}(StudyDual(vec2<f32>(a, 0.0), vec2<f32>(1.0, 0.0)));
    return vec2<f32>(select(f0, d.p.x, small), select(diff.x, d.d.x, small));
}}
"
    )
}

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
        StudyFn::Exp(isq) => {
            let (name, body) = match isq {
                0 => (
                    "study_exp_nil",
                    "    let x = StudyDual(vec2<f32>(lambda, 0.0), vec2<f32>(mu, 0.0));
    let c = study_exp_c(x);
    let s = study_exp_s(x);
    return vec4<f32>(c.p.x, c.d.x, s.p.x, s.d.x);",
                ),
                1 => (
                    "study_exp_split",
                    "    let p = sd_real(lambda + mu);
    let m = sd_real(lambda - mu);
    let cp = study_exp_c(p).p.x;
    let cm = study_exp_c(m).p.x;
    let sp = study_exp_s(p).p.x;
    let sm = study_exp_s(m).p.x;
    return vec4<f32>((cp + cm) * 0.5, (cp - cm) * 0.5, (sp + sm) * 0.5, (sp - sm) * 0.5);",
                ),
                _ => (
                    "study_exp_complex",
                    "    let x = StudyDual(vec2<f32>(lambda, mu), vec2<f32>(0.0, 0.0));
    let c = study_exp_c(x).p;
    let s = study_exp_s(x).p;
    return vec4<f32>(c.x, c.y, s.x, s.y);",
                ),
            };
            format!(
                "// `exp(B) = c0 + c1 I + (s0 + s1 I) B` for `B² = lambda + mu I`, `I² = {isq}`, as
// `[c0, c1, s0, s1]` (gax_core::study::exp_coeffs).
fn {name}(lambda: f32, mu: f32) -> vec4<f32> {{
{body}
}}
"
            )
        }
        StudyFn::Log(isq) => {
            let (name, body) = match isq {
                0 => (
                    "study_log_nil",
                    "    let r = study_log_factor(StudyDual(vec2<f32>(c0, 0.0), vec2<f32>(c1, 0.0)), StudyDual(vec2<f32>(u0, 0.0), vec2<f32>(u1, 0.0)));
    return vec2<f32>(r.p.x, r.d.x);",
                ),
                1 => (
                    "study_log_split",
                    "    let fp = study_log_factor(sd_real(c0 + c1), sd_real(u0 + u1)).p.x;
    let fm = study_log_factor(sd_real(c0 - c1), sd_real(u0 - u1)).p.x;
    return vec2<f32>((fp + fm) * 0.5, (fp - fm) * 0.5);",
                ),
                _ => (
                    "study_log_complex",
                    "    let r = study_log_factor(StudyDual(vec2<f32>(c0, c1), vec2<f32>(0.0, 0.0)), StudyDual(vec2<f32>(u0, u1), vec2<f32>(0.0, 0.0))).p;
    return vec2<f32>(r.x, r.y);",
                ),
            };
            format!(
                "// `log R = (h0 + h1 I) <R>_2` for a unit versor with scalar-pseudoscalar part `c0 + c1 I`
// and bivector part squaring to `u0 + u1 I`, `I² = {isq}`, as `[h0, h1]`
// (gax_core::study::log_coeffs).
fn {name}(c0: f32, c1: f32, u0: f32, u1: f32) -> vec2<f32> {{
{body}
}}
"
            )
        }
        StudyFn::ExpQ => format!(
            "{}{}// `exp(B) = c0 + c1 Q + (s0 + s1 Q) B` for `B² = lambda + Q` with `Q² = q` (5D algebras), as
// `[c0, c1, s0, s1]` (gax_core::study::exp_coeffs_q).
fn study_exp_q(lambda: f32, q: f32) -> vec4<f32> {{
    let c = study_q_exp_c(lambda, q);
    let s = study_q_exp_s(lambda, q);
    return vec4<f32>(c.x, c.y, s.x, s.y);
}}
",
            study_q_source("study_q_exp_c", "study_exp_c"),
            study_q_source("study_q_exp_s", "study_exp_s")
        ),
        StudyFn::LogQ => format!(
            "{}// `log R = h0 P + h1 C4 P` for a unit versor `R = c0 + C4 + P` with `C4² = qc` (5D
// algebras), as `[h0, h1]` (gax_core::study::log_coeffs_q).
fn study_log_q(c0: f32, qc: f32) -> vec2<f32> {{
    let g = study_q_acosh_sq(c0, qc);
    let qx = g.y * g.y * qc;
    let s = study_q_exp_s(g.x, qx);
    let d = 1.0 / (s.x * s.x - s.y * s.y * qx);
    return vec2<f32>(s.x * d, -(s.y * g.y) * d);
}}
",
            study_q_source("study_q_acosh_sq", "study_acosh_sq")
        ),
        StudyFn::Log6 => LOG6.into(),
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

/// The componentwise helpers of the loops ([`fallback_exp`], [`fallback_log`]) for kind `e`:
/// `{e}_add`, `{e}_sub`, `{e}_scale`.
pub fn arithmetic(e: &KindSpec, prec: Precision) -> String {
    let t = prec.scalar();
    let (es, en) = (snake(&e.name), &e.name);
    let fields = vec4s(e.layout.len());
    let each = |f: &dyn Fn(usize) -> String| -> String {
        (0..fields).map(f).collect::<Vec<_>>().join(", ")
    };
    format!(
        "// `a + b` for `{en}`.
fn {es}_add(a: {en}, b: {en}) -> {en} {{
    return {en}({});
}}

// `a - b` for `{en}`.
fn {es}_sub(a: {en}, b: {en}) -> {en} {{
    return {en}({});
}}

// `a k` for `{en}` and a scalar `k`.
fn {es}_scale(a: {en}, k: {t}) -> {en} {{
    return {en}({});
}}
",
        each(&|f| format!("a.c{f} + b.c{f}")),
        each(&|f| format!("a.c{f} - b.c{f}")),
        each(&|f| format!("a.c{f} * k")),
    )
}

/// The exponential of bivector kind `k` by scaling and squaring in kind `e` (the Rust
/// fallback of `emit_values`, for kinds without a closed form: CSTA's bivectors), as WGSL text.
/// It loops, so it is not a straight-line [`Kernel`]; it uses the module's `{e}_mul_{e}`,
/// `{e}_reverse` and [`arithmetic`]. `None` if `k` does not embed in `e`.
pub fn fallback_exp(k: &KindSpec, e: &KindSpec, prec: Precision) -> Option<String> {
    let t = prec.scalar();
    let (ks, es) = (snake(&k.name), snake(&e.name));
    let (kn, en) = (&k.name, &e.name);
    let fields = vec4s(e.layout.len());
    let each = |f: &dyn Fn(usize) -> String| -> String {
        (0..fields).map(f).collect::<Vec<_>>().join(", ")
    };
    let abs_sum = |x: &str, n: usize| -> String {
        (0..n)
            .map(|i| format!("abs({})", wgsl_coeff(x, i)))
            .collect::<Vec<_>>()
            .join(" + ")
    };
    let scaled: Vec<String> = embedding(k, e, "x")?
        .into_iter()
        .map(|c| if c == "0.0" { c } else { format!("({c}) * h") })
        .collect();
    let (one_pos, one_sign) = e.layout.position(0)?;
    let mut one = vec!["0.0".to_string(); e.layout.len()];
    one[one_pos] = if one_sign > 0 { "1.0" } else { "-1.0" }.into();
    let newton = format!(
        "{es}_mul_{es}(r, {es}_scale({es}_sub(three, {es}_mul_{es}({es}_reverse(r), r)), 0.5))"
    );
    Some(format!(
        "// The exponential of a `{kn}`, a unit `{en}`, by scaling and squaring (gax's Rust `exp` for this
// kind): `B` is halved `s` times until its 1-norm is at most 1/16, a Taylor series of degree 10
// gives `exp(B / 2^s)`, a Newton step renormalizes it, `s` squarings undo the scaling, and a
// second Newton step renormalizes the result where it is small (1-norm below 4).
fn {ks}_exp(x: {kn}) -> {en} {{
    let norm = {};
    var h: {t} = 1.0;
    var s = 0u;
    loop {{
        if s >= 64u || norm * h < 0.0625 {{
            break;
        }}
        h = h * 0.5;
        s = s + 1u;
    }}
    let e = {};
    let one = {};
    var r = one;
    for (var k = 10; k >= 1; k = k - 1) {{
        r = {es}_add(one, {es}_scale({es}_mul_{es}(e, r), 1.0 / {t}(k)));
    }}
    let three = {es}_scale(one, 3.0);
    r = {newton};
    for (var i = 0u; i < s; i = i + 1u) {{
        r = {es}_mul_{es}(r, r);
    }}
    let size = {};
    let fixed = {newton};
    return {en}({});
}}
",
        abs_sum("x", k.layout.len()),
        wgsl_construct_in(prec, en, &scaled),
        wgsl_construct_in(prec, en, &one),
        abs_sum("r", e.layout.len()),
        each(&|f| format!("select(r.c{f}, fixed.c{f}, size < 4.0)")),
    ))
}

/// The logarithm of a unit `e` (a 6D algebra's full even kind) into bivector kind `k`, as WGSL
/// text: `unit_{e}_log` takes the recorded closed form `unit_{e}_log_closed` for `⟨x⟩₀ > 1/16`
/// and otherwise `{e}_log_by_scaling`, inverse scaling and squaring (the Rust fallback of
/// `emit_values`: square roots `normalize(1 + x)` by Newton steps of the polar decomposition
/// until `x` is within 1/16 of the identity, a series of `log(1 + z)`, the scaling undone). It
/// uses the module's `{e}_mul_{e}`, `{e}_reverse` and [`arithmetic`]. `None` if `e` has no
/// scalar or lacks one of `k`'s blades.
pub fn fallback_log(k: &KindSpec, e: &KindSpec, prec: Precision) -> Option<String> {
    let t = prec.scalar();
    let es = snake(&e.name);
    let (kn, en) = (&k.name, &e.name);
    let (one_pos, one_sign) = e.layout.position(0)?;
    let scalar = |x: &str| {
        let c = wgsl_coeff(x, one_pos);
        if one_sign > 0 { c } else { format!("-{c}") }
    };
    let mut one = vec!["0.0".to_string(); e.layout.len()];
    one[one_pos] = if one_sign > 0 { "1.0" } else { "-1.0" }.into();
    let dist = |x: &str| -> String {
        (0..e.layout.len())
            .map(|i| {
                let c = wgsl_coeff(x, i);
                if i == one_pos {
                    format!("abs({c} - ({}))", one[one_pos])
                } else {
                    format!("abs({c})")
                }
            })
            .collect::<Vec<_>>()
            .join(" + ")
    };
    let project: Vec<String> = k
        .layout
        .blades
        .iter()
        .map(|&(m, sk)| {
            let (pos, se) = e.layout.position(m)?;
            let sign = if sk * se > 0 { "" } else { "-" };
            Some(format!("{sign}{} * f", wgsl_coeff("l", pos)))
        })
        .collect::<Option<_>>()?;
    // Newton stops at 64 ε of the precision (8 ε in `f16`, whose 64 ε is 1/16).
    let tol = match prec {
        Precision::F32 => "7.6e-6",
        Precision::F16 => "0.008",
    };
    Some(format!(
        "// The logarithm of a unit `{en}` by inverse scaling and squaring (the fallback of
// `unit_{es}_log` near a half turn, where the closed form loses `ε/<x>_0`): square roots
// `normalize(1 + x)` until `x` is within 1/16 of the identity, each `1 + x` scaled so that its
// invariant parts are at most 1 and made unit by Newton steps `y (3 - ~y y) / 2`, then a series
// of `log(1 + z)` to degree 16, and the scaling undone.
fn {es}_log_by_scaling(r: {en}) -> {kn} {{
    let one = {};
    let three = {es}_scale(one, 3.0);
    var x = r;
    var s = 0u;
    loop {{
        if s >= 64u || {} < 0.0625 {{
            break;
        }}
        var y = {es}_add(one, x);
        let yy = {es}_mul_{es}({es}_reverse(y), y);
        y = {es}_scale(y, 1.0 / sqrt(4.0 * {}));
        for (var k = 0; k < 200; k++) {{
            let n = {es}_mul_{es}({es}_reverse(y), y);
            if {} < {tol} {{
                break;
            }}
            y = {es}_mul_{es}(y, {es}_scale({es}_sub(three, n), 0.5));
        }}
        x = y;
        s = s + 1u;
    }}
    let z = {es}_sub(x, one);
    var q = {es}_scale(one, -1.0 / 16.0);
    for (var k = 15; k >= 1; k--) {{
        q = {es}_add({es}_scale(one, select({t}(-1.0), {t}(1.0), k % 2 == 1) / {t}(k)), {es}_mul_{es}(z, q));
    }}
    let l = {es}_mul_{es}(z, q);
    let f = exp2({t}(s));
    return {};
}}

// The logarithm of a unit `{en}`: the `{kn}` B with `exp(B) = x`, principal (every invariant
// plane below a half turn). In closed form (`unit_{es}_log_closed`, docs/log6d.md) where
// `<x>_0 > 1/16`, else by inverse scaling and squaring.
fn unit_{es}_log(x: {en}) -> {kn} {{
    if {} > 0.0625 {{
        return unit_{es}_log_closed(x);
    }}
    return {es}_log_by_scaling(x);
}}
",
        wgsl_construct_in(prec, en, &one),
        dist("x"),
        scalar("yy"),
        dist("n"),
        wgsl_construct_in(prec, kn, &project),
        scalar("x"),
    ))
}

/// The even kinds whose `log` is the closed form with a fallback ([`fallback_log`]), with their
/// bivector kind: those with a recorded `unit_{e}_log_closed` kernel.
pub fn fallback_logs<'a>(
    spec: &'a AlgebraSpec,
    stats: &Stats,
) -> Vec<(&'a KindSpec, &'a KindSpec)> {
    stats
        .values
        .iter()
        .filter_map(|v| {
            let name = format!("unit_{}_log_closed", snake(&v.kind));
            v.kernels.iter().find(|k| k.name == name)?;
            let e = spec.kinds.iter().find(|x| x.name == v.kind)?;
            let b = spec
                .kinds
                .iter()
                .find(|x| Some(&x.name) == v.log.as_ref())?;
            Some((b, e))
        })
        .collect()
}

/// The kinds whose `exp` is the scaling-and-squaring fallback, with the kind it lands in: those
/// with an `exp` in Rust but no straight-line WGSL kernel.
pub fn fallback_exps<'a>(
    spec: &'a AlgebraSpec,
    stats: &Stats,
) -> Vec<(&'a KindSpec, &'a KindSpec)> {
    stats
        .values
        .iter()
        .filter_map(|v| {
            let e = v.exp.as_ref()?;
            let ks = snake(&v.kind);
            if v.kernels.iter().any(|k| k.name == format!("{ks}_exp")) {
                return None;
            }
            let k = spec.kinds.iter().find(|x| x.name == v.kind)?;
            let e = spec.kinds.iter().find(|x| &x.name == e)?;
            Some((k, e))
        })
        .collect()
}

/// The WGSL module of an algebra, from its spec and the kernels the Rust emitter recorded.
///
/// `fma` fuses single-use products into `fma(a, b, c)`.
pub fn module(spec: &AlgebraSpec, stats: &Stats, fma: bool) -> String {
    module_in(spec, stats, fma, Precision::F32)
}

/// [`module`] in the given precision: the `f16` module (`gax::{name}_f16`) has the same
/// functions, on structs of `vec4<f16>`, and starts with `enable f16;`.
#[allow(clippy::too_many_lines)]
pub fn module_in(spec: &AlgebraSpec, stats: &Stats, fma: bool, prec: Precision) -> String {
    let name = &spec.name;
    let t = prec.scalar();
    let (path, enable, note) = match prec {
        Precision::F32 => (name.clone(), String::new(), "Everything is `f32`"),
        Precision::F16 => (
            format!("{name}_f16"),
            "enable f16;\n\n".to_string(),
            "Everything is `f16` (the device needs `shader-f16`), except inside the Study helpers,\n//   which compute in `f32`",
        ),
    };
    let mut s = String::new();
    let _ = write!(
        s,
        "{enable}// @generated by gax-regen from the `{name}` spec. Do not edit by hand.
//
// gax WGSL module `gax::{path}` (plain WGSL, also valid WESL; see gax's docs/shaders.md).
//
// * Each kind is a struct of `ceil(N/4)` `vec4<{t}>` fields `c0, c1, ...` holding its N
//   coefficients in the Rust blade order (listed per struct), zero-padded. The Rust side has
//   the same layout (`{{Kind}}Gpu`, feature `bytemuck`).
// * Function names: `motor_new`, `motor_from_rotor`, `motor_reverse`, `motor_mul_point`
//   (geometric product), `motor_sandwich_point` (`v x ~v`), `unit_motor_sandwich_point` (for
//   `v ~v = 1`), `motor_matrix_point` (the matrix of the sandwich, applied as `m * x`),
//   `motor_normalized`, `motor_renormalize_fast`, `line_exp`, `unit_motor_log`,
//   `motor_norm_squared`.
// * {note}. Kernels are the Rust kernels' verified straight-line programs.

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
            let _ = writeln!(s, "    c{f}: vec4<{t}>,");
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
            .map(|b| format!("{}: {t}", if b == "1" { "s" } else { b }))
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
            wgsl_construct_in(prec, &k.name, &parts)
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
                    wgsl_construct_in(prec, &to.name, &parts)
                );
            }
        }
    }
    let kernels = kernels(spec, stats);
    let fallbacks = fallback_exps(spec, stats);
    let mut used: BTreeSet<&'static str> = BTreeSet::new();
    let mut helpers = Vec::new();
    for k in &kernels {
        let _ = writeln!(s, "{}", k.wgsl_in(prec, fma, ""));
        for st in &k.steps {
            if let Step::Study { func, .. } = st
                && used.insert(func.wgsl_name())
            {
                helpers.push(*func);
            }
        }
    }
    let logs = fallback_logs(spec, stats);
    let mut arith: Vec<&str> = Vec::new();
    for (_, e) in fallbacks.iter().chain(&logs) {
        if !arith.contains(&e.name.as_str()) {
            arith.push(&e.name);
            let _ = writeln!(s, "{}", arithmetic(e, prec));
        }
    }
    for (k, e) in &fallbacks {
        if let Some(text) = fallback_exp(k, e, prec) {
            let _ = writeln!(s, "{text}");
        }
    }
    for (k, e) in &logs {
        if let Some(text) = fallback_log(k, e, prec) {
            let _ = writeln!(s, "{text}");
        }
    }
    // `study_log_q` calls `study_q_exp_s`, which `study_exp_q` defines.
    if helpers.contains(&StudyFn::LogQ) && !helpers.contains(&StudyFn::ExpQ) {
        let _ = writeln!(s, "{}", study_q_source("study_q_exp_s", "study_exp_s"));
    }
    helpers.sort_by_key(|f| f.wgsl_name());
    if helpers.iter().any(|f| f.needs_channels()) {
        let _ = writeln!(s, "{CHANNELS}");
    }
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
    // The scaling-and-squaring loops (`fallback_exp`, `fallback_log`) need their kind's product.
    for (_, e) in fallback_exps(spec, stats)
        .into_iter()
        .chain(fallback_logs(spec, stats))
    {
        let name = format!("{}_mul_{}", snake(&e.name), snake(&e.name));
        if !kernels.iter().any(|k| k.name == name) {
            kernels.extend(product(
                spec,
                BinOp::Gp,
                e,
                e,
                &name,
                &format!("The geometric product of two `{}`.", e.name),
            ));
        }
    }
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
