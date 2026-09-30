//! The logarithm of an even versor with four invariant planes: 8D and 9D (docs/log6d.md §9).
//!
//! As in 6D ([`super::log_coeffs_6d`]), a unit versor `R = exp(B)` with `B = b₁ + b₂ + b₃ + b₄`
//! (commuting simple bivectors) has `uⱼ = cosh²μⱼ` as the roots of a polynomial whose
//! coefficients come from the scalar parts of `R`'s grade parts squared, now the quartic
//! `t⁴ − p1 t³ + p2 t² − p3 t + p4`, and `log R = r0⁻¹ Σ αᵢ Qᵢ` for bivectors `Qᵢ` of weights
//! `uⱼⁱ` and the cubic `α` interpolating `φ(u) = √u asinh(√(u−1))/√(u−1)` at the roots.
//!
//! The interpolant never divides by the difference of two close roots. The roots are grouped
//! into factors that are apart from each other, whichever grouping separates them best:
//! all four as one (φ's series at their mean, reduced modulo the quartic), an isolated real
//! root and a cubic (the 6D interpolant), two real quadratics (a pair each: its series or its
//! chord), or two complex conjugate quadratics (two loxodromic pairs with nearly the same
//! invariants, as in `R(4,4)`). A factor's part depends on its roots only symmetrically, and the
//! parts are joined by Chinese remaindering, which divides by the factors' resultants: products
//! of differences between roots that are apart.

use super::{
    Channel, Cx, LOG6_TERMS, Real, Series, cbrt, cubic6, g_series, log_coeffs_6d, phi_complex,
    phi_pair, phi_real, phi_series, reduce2, split6, turn_alpha3,
};
use core::ops::{Add, Div, Mul, Neg, Sub};

/// The quartic in `t = u − m` (`m = p1/4`, the roots' mean): `t⁴ + a t² + b t + c`, and its
/// regime: close roots where `bound < limit` (their spread, bounded from `a`, `b` and `c`,
/// against a quarter of their distance to `u = 0`).
pub(super) struct Quartic8<T> {
    m: T,
    a: T,
    b: T,
    c: T,
    bound: T,
    limit: T,
}

fn quartic8<T: Real>(p: [T; 4]) -> Quartic8<T> {
    let [p1, p2, p3, p4] = p;
    let (two, three, four, six) = (
        T::from_i64(2),
        T::from_i64(3),
        T::from_i64(4),
        T::from_i64(6),
    );
    let m = p1 * T::from_f64(0.25);
    // Taylor coefficients of the quartic at m (its t³ term vanishes).
    let a = (six * m - three * p1) * m + p2;
    let b = ((four * m - three * p1) * m + two * p2) * m - p3;
    let c = (((m - p1) * m + p2) * m - p3) * m + p4;
    // Fujiwara's bound on the roots of t⁴ + a t² + b t + c.
    let bound = two
        * a.abs()
            .sqrt()
            .max(cbrt(b.abs()))
            .max((c.abs() * T::from_f64(0.5)).sqrt().sqrt());
    Quartic8 {
        m,
        a,
        b,
        c,
        bound,
        limit: m * T::from_f64(0.25),
    }
}

/// `Σ cᵢ tⁱ` with `t = u − m`, as coefficients in `u` (a Taylor shift).
fn shift<T: Real, const N: usize>(mut c: [T; N], m: T) -> [T; N] {
    for i in 0..N - 1 {
        for j in (i..N - 1).rev() {
            c[j] = c[j] - m * c[j + 1];
        }
    }
    c
}

/// The interpolant `[α0, α1, α2, α3]` (in `u`) of a series at the quartic's centre `m` in
/// `s = t/ρ`, reduced modulo `t⁴ + a t² + b t + c`.
fn reduce4<T: Real>(ph: &Series<T>, q: &Quartic8<T>, rho: T) -> [T; 4] {
    let (zero, one) = (T::zero(), T::one());
    let r2 = rho * rho;
    let (qa, qb, qc) = (q.a / r2, q.b / (r2 * rho), q.c / (r2 * r2));
    // tⁿ = x3 t³ + x2 t² + x1 t + x0 modulo the quartic.
    let (mut x3, mut x2, mut x1, mut x0) = (zero, zero, zero, one);
    let mut s = [zero; 4];
    for p in ph {
        s[0] = s[0] + *p * x0;
        s[1] = s[1] + *p * x1;
        s[2] = s[2] + *p * x2;
        s[3] = s[3] + *p * x3;
        // t (x3 t³ + x2 t² + x1 t + x0), with t⁴ = −a t² − b t − c.
        let (n3, n2, n1, n0) = (x2, x1 - x3 * qa, x0 - x3 * qb, -(x3 * qc));
        x3 = n3;
        x2 = n2;
        x1 = n1;
        x0 = n0;
    }
    let s = [s[0], s[1] / rho, s[2] / r2, s[3] / (r2 * rho)];
    shift(s, q.m)
}

/// Arithmetic shared by real and complex numbers, for the parts used in both.
pub(super) trait Field:
    Copy
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
{
}
impl<F> Field for F where
    F: Copy
        + Add<Output = F>
        + Sub<Output = F>
        + Mul<Output = F>
        + Div<Output = F>
        + Neg<Output = F>
{
}

/// The cubic through the line `la` at the roots of `u² − sa u + pa` and the line `lb` at the
/// roots of `u² − sb u + pb` (Chinese remaindering: `la + qa K`, `K = (lb − la) qa⁻¹` modulo
/// `qb`). It divides by the resultant `qa(b₁) qa(b₂)`, the product of the four differences
/// between the two pairs' roots.
fn crt22<F: Field>(la: [F; 2], sa: F, pa: F, lb: [F; 2], sb: F, pb: F) -> [F; 4] {
    // qa ≡ g1 u + g0 modulo qb, and its inverse (−g1 u + g1 sb + g0) / n.
    let (g1, g0) = (sb - sa, pa - pb);
    let n = g0 * g0 + g0 * g1 * sb + g1 * g1 * pb;
    let (h1, h0) = (-g1 / n, (g1 * sb + g0) / n);
    let (d1, d0) = (lb[1] - la[1], lb[0] - la[0]);
    let k1 = d1 * h1 * sb + d1 * h0 + d0 * h1;
    let k0 = d0 * h0 - d1 * h1 * pb;
    [la[0] + pa * k0, la[1] + pa * k1 - sa * k0, k0 - sa * k1, k1]
}

/// The cubic through the quadratic `l3` at the roots of `u³ − q1 u² + q2 u − q3` and the value
/// `fr` at `r`.
fn add_root<T: Real>(l3: [T; 3], q: [T; 3], fr: T, r: T) -> [T; 4] {
    let [q1, q2, q3] = q;
    let at = (l3[2] * r + l3[1]) * r + l3[0];
    let qr = ((r - q1) * r + q2) * r - q3;
    let k = (fr - at) / qr;
    [l3[0] - q3 * k, l3[1] + q2 * k, l3[2] - q1 * k, k]
}

/// The roots of `t² + u t + v` without cancellation (a conjugate pair when complex).
fn quad_roots<T: Real>(u: T, v: T) -> [Cx<T>; 2] {
    let (zero, half) = (T::zero(), T::from_f64(0.5));
    let disc = u * u - T::from_i64(4) * v;
    let sq = disc.abs().sqrt();
    let q = -(u + T::select_lt(u, zero, -sq, sq)) * half;
    let tiny = T::from_f64(1e-30);
    let q_safe = T::select_lt(q.abs(), tiny, T::one(), q);
    let other = T::select_lt(q.abs(), tiny, zero, v / q_safe);
    let (re, im) = (-u * half, sq * half);
    let real = |x: T| Cx { re: x, im: zero };
    [
        Cx::select(disc, zero, Cx { re, im }, real(q)),
        Cx::select(disc, zero, Cx { re, im: -im }, real(other)),
    ]
}

/// One Newton step on the factorization of the monic quartic with lower coefficients
/// `[a3, a2, a1, a0]` by `x² + u x + v` (Bairstow's method): the remainder `r1 x + r0` of the
/// division, and its derivatives from the remainder `s1 x + s0` of the quotient. Converges
/// quadratically where the factor's roots are apart from the cofactor's. Returns the step, the
/// Jacobian's determinant and the quotient `x² + q1 x + q0`.
fn bairstow<F: Field>(a: [F; 4], u: F, v: F) -> (F, F, F, F, F) {
    let [a3, a2, a1, a0] = a;
    let q1 = a3 - u;
    let q0 = a2 - u * q1 - v;
    let r1 = a1 - u * q0 - v * q1;
    let r0 = a0 - v * q0;
    let (s1, s0) = (q1 - u, q0 - v);
    // J = [[−(s0 − u s1), −s1], [v s1, −s0]], J (du, dv) = −(r1, r0).
    let (j11, j12, j21, j22) = (u * s1 - s0, -s1, v * s1, -s0);
    let det = j11 * j22 - j12 * j21;
    let du = (j12 * r0 - j22 * r1) / det;
    let dv = (j21 * r1 - j11 * r0) / det;
    (du, dv, det, q1, q0)
}

/// The roots `[y₁, y₂, y₃]` of the resolvent cubic `y³ + b2 y² + b1 y + b0`, `y₁` real: three
/// real roots from the trigonometric form (`y₁` the largest), or one from Cardano's formula and
/// the conjugate pair of the quadratic that remains. The real ones are polished by Newton steps.
fn resolvent<T: Real>(b2: T, b1: T, b0: T) -> [Cx<T>; 3] {
    let (zero, one) = (T::zero(), T::one());
    let tiny = T::from_f64(1e-30);
    let three = T::from_i64(3);
    let sh = b2 * three.recip();
    let pp = b1 - b2 * sh;
    let qq = (T::from_f64(2.0 / 27.0) * b2 * b2 - b1 * three.recip()) * b2 + b0;
    let disc = qq * qq * T::from_f64(0.25) + pp * pp * pp * T::from_f64(1.0 / 27.0);
    // A Newton step, kept only where it lowers the residual (at a multiple root the derivative
    // vanishes and the step would throw the root off).
    let f = |y: T| ((y + b2) * y + b1) * y + b0;
    let polish = |y: T| {
        let df = (three * y + b2 + b2) * y + b1;
        let next = y - f(y) / T::select_lt(df.abs(), tiny, one, df);
        T::select_lt(f(next).abs(), f(y).abs(), next, y)
    };
    // One real root: Cardano without cancellation, w = ∛(−q/2 ∓ √disc), x = w − p/(3w).
    let sq = disc.max(zero).sqrt();
    let w = cbrt(-(qq * T::from_f64(0.5) + T::select_lt(qq, zero, -sq, sq)));
    let w_safe = T::select_lt(w.abs(), tiny, one, w);
    let cardano = T::select_lt(w.abs(), tiny, zero, w - pp / (three * w_safe));
    let y1 = polish(polish(cardano - sh));
    // The others: y² + e1 y + e0, with e0 (their product) from whichever side is stable.
    let e1 = b2 + y1;
    let e0f = b1 + y1 * e1;
    let e0 = T::select_lt(
        e0f.abs(),
        y1 * y1,
        -b0 / T::select_lt(y1.abs(), tiny, one, y1),
        e0f,
    );
    let [y2, y3] = quad_roots(e1, e0);
    // Three real roots: the trigonometric form.
    let rad = (-pp * three.recip()).max(zero).sqrt();
    let neg = pp.min(-tiny);
    let arg = (qq * T::from_f64(1.5) / neg * (-three / neg).sqrt())
        .max(-one)
        .min(one);
    let ang = (one - arg * arg).max(zero).sqrt().atan2(arg) * three.recip();
    let third = T::from_f64(2.0 * core::f64::consts::PI / 3.0);
    let trig = [zero, one, T::from_i64(2)]
        .map(|k| Cx::real(polish(polish((rad + rad) * (ang - third * k).cos() - sh))));
    let one_real = [Cx::real(y1), y2, y3];
    core::array::from_fn(|k| Cx::select(disc, zero, trig[k], one_real[k]))
}

/// The groupings of a quartic's roots (see the module documentation), each with how far apart
/// its factors are (`gap`, 0 where it does not exist).
pub(super) struct Split8<T> {
    /// An isolated real root `r` (in `u`) and the cubic of the others, `[q1, q2, q3]`.
    r: T,
    cubic: [T; 3],
    gap_r: T,
    /// Two real quadratics `u² − s u + p`: `(s, p)` each.
    pair_a: (T, T),
    pair_b: (T, T),
    gap_pairs: T,
    /// A complex quadratic (with its conjugate): `(s, p)`.
    conj: (Cx<T>, Cx<T>),
    gap_conj: T,
    /// 1 where all four roots are (nearly) real.
    real: T,
}

#[allow(clippy::too_many_lines)]
fn split8<T: Real>(q: &Quartic8<T>, p: [T; 4]) -> Split8<T> {
    let (zero, one, half) = (T::zero(), T::one(), T::from_f64(0.5));
    // Guards that hold in f32 as well (1e-300 would underflow to zero there).
    let (tiny, big) = (T::from_f64(1e-30), T::from_f64(1e30));
    let (a, b, c, m) = (q.a, q.b, q.c, q.m);
    let safe = |x: T| T::select_lt(x.abs(), tiny, one, x);
    let finite = |x: T| T::select_lt(x.abs(), big, x, zero);
    // The resolvent y³ + 2a y² + (a² − 4c) y − b² has one root per way of pairing the roots,
    // (t₁ + t₂)²: real and non-negative for a pairing into real factors, negative for one into
    // complex conjugate factors. Euler: the roots are (±√y₁ ± √y₂ ± √y₃)/2 with
    // √y₁ √y₂ √y₃ = −b, which holds up where roots coincide (a multiple resolvent root).
    let y = resolvent(a + a, a * a - T::from_i64(4) * c, -(b * b));
    let mut s = y.map(Channel::sqrt);
    let prod = s[0] * s[1] * s[2];
    s[0] = Cx::select(prod.re * b, zero, s[0], -s[0]);
    let h = Cx::real(half);
    let t = [
        (s[0] + s[1] + s[2]) * h,
        (s[0] - s[1] - s[2]) * h,
        (s[1] - s[0] - s[2]) * h,
        (s[2] - s[0] - s[1]) * h,
    ];
    let mc = Cx::real(m);
    let roots = t.map(|x| x + mc);
    let dist = |x: Cx<T>, y: Cx<T>| {
        let (dr, di) = (x.re - y.re, x.im - y.im);
        (dr * dr + di * di).sqrt()
    };
    let [p1, p2, p3, p4] = p;

    // Pairing k: {t₀, tₖ₊₁} (sum √yₖ) and the other two. The best of each kind is refined in
    // u itself (not t = u − m, where a small root loses its relative accuracy).
    let others = [[2, 3], [1, 3], [1, 2]];
    let tol = T::epsilon() * T::from_f64(1e4) * (one + a.abs());
    let (mut gap_pairs, mut u, mut v) = (zero, zero, zero);
    let (mut gap_conj, mut cu, mut cv) = (zero, Cx::real(zero), Cx::real(zero));
    for k in 0..3 {
        let [i, j] = others[k];
        let pair = [roots[0], roots[k + 1]];
        let gap = finite(
            dist(pair[0], roots[i])
                .min(dist(pair[0], roots[j]))
                .min(dist(pair[1], roots[i]))
                .min(dist(pair[1], roots[j])),
        );
        let is_real = T::select_lt(y[k].im.abs(), tiny, one, zero);
        let real = is_real * T::select_lt(-tol, y[k].re, one, zero);
        // Conjugate factors: the other two roots are the pair's conjugates (a rounding error
        // in the resolvent's sign would pair real roots as if they were).
        let bar = |z: Cx<T>| Cx {
            re: z.re,
            im: -z.im,
        };
        let (i, j) = (roots[i], roots[j]);
        let mismatch = (dist(bar(pair[0]), i) + dist(bar(pair[1]), j))
            .min(dist(bar(pair[0]), j) + dist(bar(pair[1]), i));
        let conj = is_real * (one - real) * T::select_lt(mismatch, gap * half, one, zero);
        let (sum, prd) = (pair[0] + pair[1], pair[0] * pair[1]);
        let better = T::select_lt(gap_pairs, gap * real, one, zero);
        u = T::select_lt(half, better, -sum.re, u);
        v = T::select_lt(half, better, prd.re, v);
        gap_pairs = gap_pairs.max(gap * real);
        let better = T::select_lt(gap_conj, gap * conj, one, zero);
        cu = Cx::select(better, half, cu, -sum);
        cv = Cx::select(better, half, cv, prd);
        gap_conj = gap_conj.max(gap * conj);
    }
    // Newton steps on the factorization, which converge where the factors are apart.
    let coef = [-p1, p2, -p3, p4];
    let mut quot = (zero, zero);
    for _ in 0..3 {
        let (du, dv, det, q1, q0) = bairstow(coef, u, v);
        let ok = T::select_lt(det.abs(), T::from_f64(1e-35), zero, one);
        u = u + finite(du) * ok;
        v = v + finite(dv) * ok;
        quot = (q1, q0);
    }
    // The cofactor x² − sb x + pb, from the quotient, or (where this factor has the larger
    // product) backward from p4 and p3: pb = p4 / v, sb = (p3 + u pb) / v.
    let backward = T::select_lt(p4.abs().sqrt(), v.abs(), one, zero);
    let pb = T::select_lt(half, backward, p4 / safe(v), quot.1);
    let sb = T::select_lt(half, backward, (p3 + u * pb) / safe(v), -quot.0);
    let coef_cx = coef.map(Cx::real);
    for _ in 0..3 {
        let (du, dv, det, _, _) = bairstow(coef_cx, cu, cv);
        let ok = T::select_lt(det.norm2(), T::from_f64(1e-35), zero, one);
        let ok = ok * T::select_lt(du.norm2() + dv.norm2(), big, one, zero);
        cu = Cx::select(ok, half, cu, cu + du);
        cv = Cx::select(ok, half, cv, cv + dv);
    }

    // The most isolated (nearly) real root, polished by Newton steps on the quartic in u.
    let mut gap_r = zero;
    let mut best = zero;
    let mut all_real = one;
    for j in 0..4 {
        let mut iso = big;
        for (i, ti) in t.iter().enumerate() {
            if i != j {
                iso = iso.min(dist(*ti, t[j]));
            }
        }
        let real = T::select_lt(
            t[j].im.abs(),
            T::epsilon().sqrt() * (one + t[j].re.abs()),
            one,
            zero,
        );
        all_real = all_real * real;
        let iso = finite(iso) * real;
        best = T::select_lt(gap_r, iso, roots[j].re, best);
        gap_r = gap_r.max(iso);
    }
    let quartic = |r: T| (((r - p1) * r + p2) * r - p3) * r + p4;
    let mut r = best;
    for _ in 0..3 {
        let df = ((T::from_i64(4) * r - T::from_i64(3) * p1) * r + p2 + p2) * r - p3;
        let next = r - quartic(r) / safe(df);
        r = T::select_lt(quartic(next).abs(), quartic(r).abs(), next, r);
    }
    // Deflation, each coefficient from the side with the smaller error: forward from p1
    // (stable for a small r) or backward from p4 (for a large one).
    let ar = r.abs();
    let q1f = p1 - r;
    let q2f = p2 - r * q1f;
    let q3f = p3 - r * q2f;
    let e1f = p1.abs() + ar;
    let e2f = p2.abs() + ar * (q1f.abs() + e1f);
    let e3f = p3.abs() + ar * (q2f.abs() + e2f);
    let rinv = safe(r).recip();
    let q3b = p4 * rinv;
    let q2b = (p3 - q3b) * rinv;
    let q1b = (p2 - q2b) * rinv;
    let e3b = T::select_lt(ar, tiny, big, q3b.abs());
    let e2b = T::select_lt(ar, tiny, big, (p3.abs() + q3b.abs() + e3b) * rinv.abs());
    let e1b = T::select_lt(ar, tiny, big, (p2.abs() + q2b.abs() + e2b) * rinv.abs());
    let cubic = [
        T::select_lt(e1b, e1f, q1b, q1f),
        T::select_lt(e2b, e2f, q2b, q2f),
        T::select_lt(e3b, e3f, q3b, q3f),
    ];

    Split8 {
        r,
        cubic,
        gap_r,
        pair_a: (-u, v),
        pair_b: (sb, pb),
        gap_pairs,
        conj: (-cu, cv),
        gap_conj,
        real: all_real,
    }
}

/// Which grouping each lane uses: `(isolated root, two real pairs, two conjugate pairs)` as
/// 1 or 0 (none: all four as one, where they are close).
fn grouping<T: Real>(q: &Quartic8<T>, sp: &Split8<T>) -> (T, T, T) {
    let apart = T::select_lt(q.bound, q.limit, T::zero(), T::one());
    let (r, pairs, conj) = grouping_apart(sp);
    (apart * r, apart * pairs, apart * conj)
}

/// The grouping whose factors are farthest apart, whether or not the roots are close.
fn grouping_apart<T: Real>(sp: &Split8<T>) -> (T, T, T) {
    let (zero, one) = (T::zero(), T::one());
    let conj = T::select_lt(sp.gap_r.max(sp.gap_pairs), sp.gap_conj, one, zero);
    let pairs = (one - conj) * T::select_lt(sp.gap_r, sp.gap_pairs, one, zero);
    ((one - conj) * (one - pairs), pairs, conj)
}

// ---------------------------------------------------------------------------------------------
// Complex series, for a pair of close complex roots.

type CxSeries<T> = [Cx<T>; LOG6_TERMS];

fn cx_mul<T: Real>(a: &CxSeries<T>, b: &CxSeries<T>) -> CxSeries<T> {
    let mut out = [Cx::real(T::zero()); LOG6_TERMS];
    for i in 0..LOG6_TERMS {
        for j in 0..LOG6_TERMS - i {
            out[i + j] = out[i + j] + a[i] * b[j];
        }
    }
    out
}

fn cx_div<T: Real>(a: &CxSeries<T>, b: &CxSeries<T>) -> CxSeries<T> {
    let mut q = [Cx::real(T::zero()); LOG6_TERMS];
    let inv = Cx::real(T::one()) / b[0];
    for k in 0..LOG6_TERMS {
        let mut s = a[k];
        for j in 1..=k {
            s = s - b[j] * q[k - j];
        }
        q[k] = s * inv;
    }
    q
}

/// `(c + ρ s)^(1/2)` or `(c + ρ s)^(−1/2)` as a series in `s`, principal at `c`.
fn cx_sqrt<T: Real>(c: Cx<T>, inverse: bool, rho: T) -> CxSeries<T> {
    let e = if inverse { -0.5 } else { 0.5 };
    let root = c.sqrt();
    let one = Cx::real(T::one());
    let lead = if inverse { one / root } else { root };
    let step = Cx::real(rho) / c;
    let mut out = [Cx::real(T::zero()); LOG6_TERMS];
    let (mut binom, mut power) = (1.0f64, one);
    for (k, o) in out.iter_mut().enumerate() {
        *o = lead * Cx::real(T::from_f64(binom)) * power;
        binom *= (e - k as f64) / (k as f64 + 1.0);
        power = power * step;
    }
    out
}

/// The Taylor series of `φ(u) = √u · F(u − 1)` at a complex `u = m` (principal `√u`) in
/// `s = (u − m)/ρ`, as [`phi_series`] does at a real one: `F`'s Maclaurin series shifted to
/// `x0 = m − 1` where `|x0| < 1/4`, else composed from `v = √(x0 + t)`,
/// `asinh(v)' = v'/√(m + t)`, `F = asinh(v)/v`.
fn phi_series_cx<T: Real>(m: Cx<T>, rho: T) -> CxSeries<T> {
    let one = Cx::real(T::one());
    let zero = [Cx::real(T::zero()); LOG6_TERMS];
    let x0 = m - one;
    let ax = x0.norm2().sqrt();
    let quarter = T::from_f64(0.25);
    let near = if T::all_lt(quarter, ax) {
        zero
    } else {
        let mut c = [Cx::real(T::zero()); 90];
        let mut b = 1.0f64;
        for (n, f) in c.iter_mut().enumerate() {
            let sign = if n % 2 == 0 { 1.0 } else { -1.0 };
            *f = Cx::real(T::from_f64(sign * b / (2.0 * n as f64 + 1.0)));
            b *= (2.0 * n as f64 + 1.0) / (2.0 * n as f64 + 2.0);
        }
        let x = Cx::select(ax, quarter, x0, Cx::real(T::zero()));
        let mut out = zero;
        let mut power = T::one();
        for (i, o) in out.iter_mut().enumerate() {
            for j in (i..c.len() - 1).rev() {
                c[j] = c[j] + x * c[j + 1];
            }
            *o = c[i] * Cx::real(power);
            power = power * rho;
        }
        out
    };
    let far = if T::all_lt(ax, quarter) {
        zero
    } else {
        let x = Cx::select(ax, quarter, Cx::real(quarter), x0);
        let mm = x + one;
        let v = cx_sqrt(x, false, rho);
        let r = cx_sqrt(mm, true, rho);
        let mut dv = zero;
        for k in 1..LOG6_TERMS {
            dv[k - 1] = v[k] * Cx::real(T::from_i64(k as i64));
        }
        let d = cx_mul(&dv, &r);
        let mut a = zero;
        a[0] = (v[0] + mm.sqrt()).ln();
        for k in 1..LOG6_TERMS {
            a[k] = d[k - 1] * Cx::real(T::from_i64(k as i64).recip());
        }
        cx_div(&a, &v)
    };
    let f: CxSeries<T> = core::array::from_fn(|k| Cx::select(ax, quarter, near[k], far[k]));
    cx_mul(&cx_sqrt(m, false, rho), &f)
}

/// The line `[l0, l1]` through `φ` at the roots of `u² − s u + p` (complex): its series at the
/// midpoint where the roots are within a quarter of its distance to φ's branch cut `u ≤ 0`,
/// else the chord.
fn phi_pair_cx<T: Real>(s: Cx<T>, p: Cx<T>) -> [Cx<T>; 2] {
    let half = Cx::real(T::from_f64(0.5));
    let mid = s * half;
    let d2 = mid * mid - p;
    let d = d2.sqrt();
    let ad = d.norm2().sqrt();
    let reach =
        T::select_lt(mid.re, T::zero(), mid.im.abs(), mid.norm2().sqrt()) * T::from_f64(0.25);
    let close = if T::all_lt(reach, ad) {
        [Cx::real(T::zero()); 2]
    } else {
        // In s = t/ρ, ρ the distance to the cut.
        let rho = (reach * T::from_i64(4)).max(T::from_f64(1e-30));
        let ph = phi_series_cx(mid, rho);
        let d2s = d2 / Cx::real(rho * rho);
        // Σ φₖ sᵏ modulo s² − d2/ρ²: s (b s + e) = b d2/ρ² + e s.
        let (mut b, mut e) = (Cx::real(T::zero()), Cx::real(T::one()));
        let (mut sb, mut se) = (b, Cx::real(T::zero()));
        for f in &ph {
            sb = sb + *f * b;
            se = se + *f * e;
            let (nb, ne) = (e, b * d2s);
            b = nb;
            e = ne;
        }
        let sb = sb / Cx::real(rho);
        [se - sb * mid, sb]
    };
    let (z, w) = (mid + d, mid - d);
    let (fz, fw) = (phi_complex(z), phi_complex(w));
    let d_safe = Cx::select(ad, T::from_f64(1e-30), Cx::real(T::one()), d + d);
    let slope = (fz - fw) / d_safe;
    let chord = [fz - slope * z, slope];
    [
        Cx::select(ad, reach, close[0], chord[0]),
        Cx::select(ad, reach, close[1], chord[1]),
    ]
}

/// The coefficients `[α0, α1, α2, α3]` of the cubic interpolating
/// `φ(u) = √u · asinh(√(u−1))/√(u−1)` at the roots of `t⁴ − p1 t³ + p2 t² − p3 t + p4`
/// (`p = [p1, p2, p3, p4]`), the four invariants `cosh²(μⱼ)` of an 8D or 9D even versor:
/// `log R = r0⁻¹ Σ αᵢ Qᵢ` (docs/log6d.md §9). Branch free, and without dividing by the
/// difference of close roots (see the module documentation).
///
/// ```
/// // Rotations in two planes by half-angles 0.4 and 0.9: roots cos²0.4, cos²0.9, 1, 1.
/// let (c1, c2) = (0.4f64.cos().powi(2), 0.9f64.cos().powi(2));
/// let p = [c1 + c2 + 2.0, c1 * c2 + 2.0 * (c1 + c2) + 1.0, 2.0 * c1 * c2 + c1 + c2, c1 * c2];
/// let [a0, a1, a2, a3] = gax_core::study::log_coeffs_8d(p);
/// let at = |u: f64| ((a3 * u + a2) * u + a1) * u + a0;
/// // At u = cos²θ the interpolant is φ = cos θ · θ / sin θ, and 1 at u = 1.
/// assert!((at(c1) - 0.4f64.cos() * 0.4 / 0.4f64.sin()).abs() < 1e-13);
/// assert!((at(1.0) - 1.0).abs() < 1e-13);
/// ```
#[inline]
pub fn log_coeffs_8d<T: Real>(p: [T; 4]) -> [T; 4] {
    let q = quartic8(p);
    let zero = T::zero();
    // All four close: the series at their mean (skipped when no lane has them).
    let jet = if T::all_lt(q.limit, q.bound) {
        [zero; 4]
    } else {
        let rho = q.m.max(T::from_f64(1e-30));
        reduce4(&phi_series(q.m, rho), &q, rho)
    };
    if T::all_lt(q.bound, q.limit) {
        return jet;
    }
    let sp = split8(&q, p);
    let (use_r, use_pairs, use_conj) = grouping(&q, &sp);
    let half = T::from_f64(0.5);
    // An isolated real root and the cubic of the others.
    let isolated = if T::all_lt(use_r, half) {
        [zero; 4]
    } else {
        let [q1, q2, q3] = sp.cubic;
        add_root(log_coeffs_6d(q1, q2, q3), sp.cubic, phi_real(sp.r), sp.r)
    };
    // Two real pairs.
    let pairs = if T::all_lt(use_pairs, half) {
        [zero; 4]
    } else {
        let line = |(s, p): (T, T)| {
            let mid = s * half;
            let d2 = mid * mid - p;
            phi_pair(mid, d2, d2.abs().sqrt())
        };
        let (sa, pa) = sp.pair_a;
        let (sb, pb) = sp.pair_b;
        crt22(line(sp.pair_a), sa, pa, line(sp.pair_b), sb, pb)
    };
    // Two conjugate complex pairs.
    let conj = if T::all_lt(use_conj, half) {
        [zero; 4]
    } else {
        let (s, p) = sp.conj;
        let bar = |x: Cx<T>| Cx {
            re: x.re,
            im: -x.im,
        };
        let l = phi_pair_cx(s, p);
        crt22(l, s, p, [bar(l[0]), bar(l[1])], bar(s), bar(p)).map(|x| x.re)
    };
    core::array::from_fn(|k| {
        let spread = T::select_lt(
            half,
            use_conj,
            conj[k],
            T::select_lt(half, use_pairs, pairs[k], isolated[k]),
        );
        T::select_lt(q.bound, q.limit, jet[k], spread)
    })
}

/// The weights `[w1, w2, w3, w4]` of `G1 = ⟨R⟩₂` and `Gₘ = r0 ⟨R₂ₘ R₂ₘ₋₂⟩₂` (m = 2, 3, 4) in
/// `Σ αᵢ Qᵢ`, for the interpolant `α` at the roots of `t⁴ − p1 t³ + p2 t² − p3 t + p4`: the
/// bivector of weights `uⱼⁱ` is `Qᵢ = Σₘ Mᵢₘ Gₘ`, with the integer polynomial matrix `M`
/// inverse to the weights of the `G`s (docs/log6d.md §9), times `scale`.
#[inline]
pub fn q_weights_8d<T: Real>(alpha: [T; 4], p: [T; 4], scale: T) -> [T; 4] {
    let [p1, p2, _, p4] = p;
    let [a0, a1, a2, a3] = alpha;
    let (one, two, three) = (T::one(), T::from_i64(2), T::from_i64(3));
    // Q0 = G1, Q1 = p4 G1 − G2 + G3 − G4, Q2 = p4 (p1 − 3) G1 + (2 − p1) G2 + (p1 − 1) G3 − p1 G4,
    // Q3 = p4 (p1² − 3 p1 − p2 + 3) G1 + (2 p1 + p2 − p1² − 1) G2 + (p1² − p1 − p2) G3
    //      + (p2 − p1²) G4.
    let sq = p1 * p1;
    let w1 = a0 + p4 * (a1 + a2 * (p1 - three) + a3 * (sq - three * p1 - p2 + three));
    let w2 = a2 * (two - p1) - a1 + a3 * (two * p1 + p2 - sq - one);
    let w3 = a1 + a2 * (p1 - one) + a3 * (sq - p1 - p2);
    let w4 = a3 * (p2 - sq) - a1 - a2 * p1;
    [w1 * scale, w2 * scale, w3 * scale, w4 * scale]
}

/// The weights `[w1, w2, w3, w4]` with `log R = Σ wₘ Gₘ` for an 8D or 9D even versor, from its
/// invariants `p = [p1, p2, p3, p4]` and `r0 = ⟨R⟩₀` ([`log_coeffs_8d`], [`q_weights_8d`]).
#[inline]
pub fn log_weights_8d<T: Real>(p: [T; 4], r0: T) -> [T; 4] {
    q_weights_8d(log_coeffs_8d(p), p, r0.recip())
}

/// `[e0, e1, e2, e3, e4]` with `∏(−b̂) = Σ eₖ Zᵏ` for `Z` the sum of `n` (0 to 4) commuting
/// orthogonal unit rotation bivectors `b̂` (`b̂² = −1`), from Newton's identities:
/// `1`, `−Z`, `1 + Z²/2`, `−(Z³ + 7Z)/6`, `1 + 2Z²/3 + Z⁴/24`.
#[inline]
pub fn turn_polynomial_8d<T: Real>(n: T) -> [T; 5] {
    let pick = |v: [f64; 5]| {
        let at = |k: usize| T::from_f64(v[k]);
        T::select_lt(
            n,
            T::from_f64(0.5),
            at(0),
            T::select_lt(
                n,
                T::from_f64(1.5),
                at(1),
                T::select_lt(
                    n,
                    T::from_f64(2.5),
                    at(2),
                    T::select_lt(n, T::from_f64(3.5), at(3), at(4)),
                ),
            ),
        )
    };
    [
        pick([1.0, 0.0, 1.0, 0.0, 1.0]),
        pick([0.0, -1.0, 0.0, -7.0 / 6.0, 0.0]),
        pick([0.0, 0.0, 0.5, 0.0, 2.0 / 3.0]),
        pick([0.0, 0.0, 0.0, -1.0 / 6.0, 0.0]),
        pick([0.0, 0.0, 0.0, 0.0, 1.0 / 24.0]),
    ]
}

/// A candidate set of planes to turn: its cost (`1/⟨R'⟩₀ + Σ 1/|wⱼ|`, or `invalid`), whether
/// it is preferred (the groups it separates are apart), and its flags.
#[derive(Clone, Copy)]
struct Cand<T, const N: usize> {
    cost: T,
    strict: T,
    flags: [T; N],
}

/// The cheapest candidate, preferring the strict ones (any, where none of those is valid).
fn cheapest<T: Real, const N: usize>(cands: &[Cand<T, N>], invalid: T) -> [T; N] {
    let half = T::from_f64(0.5);
    let pick = |strict: bool| {
        let mut best = (invalid, [T::zero(); N]);
        for c in cands {
            let cost = if strict {
                T::select_lt(half, c.strict, c.cost, invalid)
            } else {
                c.cost
            };
            let better = T::select_lt(cost, best.0, T::one(), T::zero());
            best.0 = best.0.min(cost);
            best.1 = core::array::from_fn(|k| T::select_lt(half, better, c.flags[k], best.1[k]));
        }
        best
    };
    let (strict, loose) = (pick(true), pick(false));
    let use_strict = T::select_lt(strict.0, invalid * half, T::one(), T::zero());
    core::array::from_fn(|k| T::select_lt(half, use_strict, strict.1[k], loose.1[k]))
}

/// Which planes of an 8D or 9D even versor to turn by a quarter turn before
/// [`log_weights_8d`], as [`super::log_turn_6d`] does for 6D (docs/log6d.md §5, §9):
/// `[α0, α1, α2, α3, n]` with `Z = Σ αᵢ Qᵢ` (weights [`q_weights_8d`] with scale 1) the sum of
/// the `n` turned planes' unit bivectors, `R' = R E` with `E = ∏(−b̂)` ([`turn_polynomial_8d`])
/// and `log R = log R' + (π/2) Z`. Among the sets of rotation planes with a positive
/// `⟨R'⟩₀` (one or three planes, or two or four where `⟨R⟩₀ > 0`), including none, the one with
/// the smallest `1/⟨R'⟩₀ + Σ_T 1/|wⱼ|`; the sets follow the grouping of the roots, turning
/// close roots together (then through the series of `√(u/(1−u))/|r₀|`).
///
/// Where four coinciding invariants have `⟨R⟩₀ < 0` (three planes at one angle and the fourth at
/// its supplement), no set is valid: the planes are not determined by the invariants, and the
/// logarithm is not unique.
#[inline]
#[allow(clippy::too_many_lines)]
pub fn log_turn_8d<T: Real>(p: [T; 4], r0: T) -> [T; 5] {
    let (zero, one, half) = (T::zero(), T::one(), T::from_f64(0.5));
    let quarter = T::from_f64(0.25);
    let tiny = T::from_f64(1e-30);
    let invalid = T::from_f64(1e30);
    let apart = T::from_f64(1.0 / 64.0);
    let ar0 = r0.abs().max(tiny);
    let positive = |x: T| T::select_lt(zero, x, one, zero);
    let inv = |x: T| x.max(tiny).recip();
    let pos_r0 = positive(r0);
    // A set's cost where it is valid (and ⟨R'⟩₀, of magnitude `mag`, positive: always for an
    // odd number of planes, where ⟨R⟩₀ > 0 for an even one), else `invalid`.
    let cost = |ok: T, count: usize, mag: T, extra: T| {
        let sign = if count % 2 == 1 { one } else { pos_r0 };
        T::select_lt(zero, ok * sign * positive(mag), inv(mag) + extra, invalid)
    };
    // A cluster's `1/|w| = √(u/(1−u))/|r₀|` at its centre.
    let cluster = |u: T| (u.max(zero) / (one - u).max(tiny)).sqrt() / ar0;
    let sqrt0 = |x: T| x.max(zero).sqrt();
    let q = quartic8(p);
    let [p1, p2, p3, p4] = p;

    // All four as one cluster, through the series at their mean: a candidate beside the sets
    // of the split below (which is computed even where the roots are close: a set that turns
    // some of them, apart enough, may be the only one with a positive ⟨R'⟩₀).
    let sp = split8(&q, p);
    let prod1m = one - p1 + p2 - p3 + p4;
    let series_ok = positive(q.m.min(one - q.m) * quarter - q.bound) * positive(prod1m) * sp.real;
    let all4 = Cand {
        cost: cost(series_ok, 4, sqrt0(prod1m), T::from_i64(4) * cluster(q.m)),
        strict: one,
        flags: [zero; 1],
    };
    let (use_r, use_pairs, _) = grouping_apart(&sp);

    // An isolated root r and the cubic of the others, split as in 6D: r2 and the pair a, b.
    let (isolated, n_isolated, t4_isolated) = if T::all_lt(use_r, half) {
        ([zero; 4], zero, zero)
    } else {
        let r = sp.r;
        let [c1, c2, c3] = sp.cubic;
        let c6 = cubic6(c1, c2, c3);
        let s6 = split6(&c6, c1, c2, c3);
        let (r2, mid, d2, d) = (s6.r, s6.mid, s6.d2, s6.d);
        let (a, b) = (mid + d, (mid - d).max(zero));
        let real = positive(d2 + T::epsilon() * T::from_i64(64) * (one + mid * mid));
        let close = positive(mid * quarter - d);
        let close_g = positive((one - mid) * quarter - d) * close;
        let gap2 = T::select_lt(
            half,
            real,
            (r2 - a).abs().min((r2 - b).abs()),
            ((r2 - mid) * (r2 - mid) + d * d).sqrt(),
        );
        let sep2 = positive(gap2 - apart * (one + r2.abs()));
        let sep_pair = positive(d + d - apart * (one + mid.abs()));
        let sep_r = positive(sp.gap_r - apart * (one + r.abs()));
        let (in_r, out_r) = (sqrt0(one - r), sqrt0(r));
        let (in_r2, out_r2) = (sqrt0(one - r2), sqrt0(r2));
        let (in_a, out_a) = (sqrt0(one - a), sqrt0(a));
        let (in_b, out_b) = (sqrt0(one - b), sqrt0(b));
        let in_pair = sqrt0(one - s6.sum + s6.prod);
        let out_pair = sqrt0(s6.prod);
        let prod1m3 = one - c1 + c2 - c3;
        let series3 = positive(c6.m.min(one - c6.m) * quarter - c6.bound) * positive(prod1m3);
        // 1/|w| of each root: w is its weight in ⟨R⟩₂, √((1 − u) ∏ of the others).
        let iw_r = inv(sqrt0((one - r) * c3));
        let iw_r2 = inv(sqrt0((one - r2) * s6.prod * r));
        let iw_a = inv(sqrt0((one - a) * b * r2 * r));
        let iw_b = inv(sqrt0((one - b) * a * r2 * r));
        let iw_pair = T::select_lt(half, close, cluster(mid) + cluster(mid), iw_a + iw_b);
        let (ok_r, ok_r2) = (positive(one - r), positive(one - r2));
        let ok_pair = real * positive(one - a) * (one - close + close_g);
        let ok_a = real * positive(d) * positive(one - a);
        let ok_b = real * positive(d) * positive(one - b);
        // The cubic's sets (as in 6D): (magnitude of its part of ⟨R'⟩₀, planes, valid,
        // Σ 1/|w|, preferred, flags [t_r2, t_a, t_b, t_all]).
        let cubic_sets = [
            (sqrt0(c3), 0, one, zero, one, [zero, zero, zero, zero]),
            (
                sqrt0(prod1m3),
                3,
                series3,
                T::from_i64(3) * cluster(c6.m),
                one,
                [one, one, one, one],
            ),
            (
                in_r2 * out_pair,
                1,
                ok_r2,
                iw_r2,
                sep2,
                [one, zero, zero, zero],
            ),
            (
                out_r2 * in_pair,
                2,
                ok_pair,
                iw_pair,
                sep2,
                [zero, one, one, zero],
            ),
            (
                in_r2 * in_pair,
                3,
                ok_r2 * ok_pair,
                iw_r2 + iw_pair,
                sep2,
                [one, one, one, zero],
            ),
            (
                out_r2 * in_a * out_b,
                1,
                ok_a,
                iw_a,
                sep2 * sep_pair,
                [zero, one, zero, zero],
            ),
            (
                out_r2 * out_a * in_b,
                1,
                ok_b,
                iw_b,
                sep2 * sep_pair,
                [zero, zero, one, zero],
            ),
            (
                in_r2 * in_a * out_b,
                2,
                ok_r2 * ok_a,
                iw_r2 + iw_a,
                sep2 * sep_pair,
                [one, one, zero, zero],
            ),
            (
                in_r2 * out_a * in_b,
                2,
                ok_r2 * ok_b,
                iw_r2 + iw_b,
                sep2 * sep_pair,
                [one, zero, one, zero],
            ),
        ];
        let mut cands = [Cand {
            cost: all4.cost,
            strict: one,
            flags: [zero, zero, zero, zero, zero, T::from_i64(4), one],
        }; 19];
        for (k, &(mag, count, ok, extra, pref, f)) in cubic_sets.iter().enumerate() {
            for turn_r in [false, true] {
                let (t_r, m_r, ok_rr, iw) = if turn_r {
                    (one, in_r, ok_r, iw_r)
                } else {
                    (zero, out_r, one, zero)
                };
                // Every set but none divides by r's distance to the cubic's roots.
                let needs_r = if k == 0 && !turn_r { one } else { sep_r };
                cands[2 * k + usize::from(turn_r)] = Cand {
                    cost: cost(
                        ok * ok_rr,
                        count + usize::from(turn_r),
                        mag * m_r,
                        extra + iw,
                    ),
                    strict: pref * needs_r,
                    flags: [
                        t_r,
                        f[0],
                        f[1],
                        f[2],
                        f[3],
                        T::from_i64(count as i64) + t_r,
                        zero,
                    ],
                };
            }
        }
        let [t_r, t_r2, t_a, t_b, t_all, n, t4] = cheapest(&cands, invalid);
        let h = [t_r2 * iw_r2, t_a * iw_a, t_b * iw_b];
        let a3 = turn_alpha3(&c6, &s6, close, [t_r2, t_a, t_b, t_all], h, ar0);
        (add_root(a3, sp.cubic, t_r * iw_r, r), n, t4)
    };

    // Two real pairs: each none, both (through the series when close), or either one.
    let (pairs, n_pairs, t4_pairs) = if T::all_lt(use_pairs, half) {
        ([zero; 4], zero, zero)
    } else {
        let side = |(s, pp): (T, T), other: T| {
            let mid = s * half;
            let d2 = mid * mid - pp;
            let d = d2.abs().sqrt();
            let (a, b) = (mid + d, (mid - d).max(zero));
            let real = positive(d2 + T::epsilon() * T::from_i64(64) * (one + mid * mid));
            let close = positive(mid * quarter - d);
            let close_g = positive((one - mid) * quarter - d) * close;
            let iw_a = inv(sqrt0((one - a) * b * other));
            let iw_b = inv(sqrt0((one - b) * a * other));
            let iw_pair = T::select_lt(half, close, cluster(mid) + cluster(mid), iw_a + iw_b);
            let sep = positive(d + d - apart * (one + mid.abs()));
            let ok_pair = real * positive(one - a) * (one - close + close_g);
            let ok_a = real * positive(d) * positive(one - a);
            let ok_b = real * positive(d) * positive(one - b);
            let (in_a, out_a, in_b, out_b) = (sqrt0(one - a), sqrt0(a), sqrt0(one - b), sqrt0(b));
            let sets = [
                (sqrt0(pp), 0, one, zero, one, [zero, zero]),
                (sqrt0(one - s + pp), 2, ok_pair, iw_pair, one, [one, one]),
                (in_a * out_b, 1, ok_a, iw_a, sep, [one, zero]),
                (out_a * in_b, 1, ok_b, iw_b, sep, [zero, one]),
            ];
            (sets, (mid, d2, a, b, close, iw_a, iw_b))
        };
        let (sa, pa) = sp.pair_a;
        let (sb, pb) = sp.pair_b;
        let (sets_a, geo_a) = side(sp.pair_a, pb);
        let (sets_b, geo_b) = side(sp.pair_b, pa);
        let sep_ab =
            positive(sp.gap_pairs - apart * (one + (sa * half).abs().max((sb * half).abs())));
        let mut cands = [Cand {
            cost: all4.cost,
            strict: one,
            flags: [zero, zero, zero, zero, T::from_i64(4), one],
        }; 17];
        for (i, &(ma, ca, oka, xa, pra, fa)) in sets_a.iter().enumerate() {
            for (j, &(mb, cb, okb, xb, prb, fb)) in sets_b.iter().enumerate() {
                let needs = if i == 0 && j == 0 { one } else { sep_ab };
                cands[4 * i + j] = Cand {
                    cost: cost(oka * okb, ca + cb, ma * mb, xa + xb),
                    strict: pra * prb * needs,
                    flags: [
                        fa[0],
                        fa[1],
                        fb[0],
                        fb[1],
                        T::from_i64((ca + cb) as i64),
                        zero,
                    ],
                };
            }
        }
        let [t_a1, t_a2, t_b1, t_b2, n, t4] = cheapest(&cands, invalid);
        let line = |(mid, d2, a, b, close, iw_a, iw_b): (T, T, T, T, T, T, T), t1: T, t2: T| {
            let use_series = close * t1 * t2;
            let series = if T::all_lt(use_series, half) {
                [zero; 2]
            } else {
                let (g, rho) = g_series(mid);
                reduce2(&g, mid, rho, d2).map(|x| x / ar0)
            };
            let (ha, hb) = (t1 * iw_a, t2 * iw_b);
            let slope = (ha - hb) / (a - b).max(tiny);
            let chord = [ha - slope * a, slope];
            [
                T::select_lt(half, use_series, series[0], chord[0]),
                T::select_lt(half, use_series, series[1], chord[1]),
            ]
        };
        let la = line(geo_a, t_a1, t_a2);
        let lb = line(geo_b, t_b1, t_b2);
        (crt22(la, sa, pa, lb, sb, pb), n, t4)
    };

    // Two conjugate complex pairs: no rotation planes, nothing to turn.
    let by_grouping = |pairs: T, isolated: T| {
        T::select_lt(
            half,
            use_pairs,
            pairs,
            T::select_lt(half, use_r, isolated, zero),
        )
    };
    let t4 = by_grouping(t4_pairs, t4_isolated);
    let as_one = if T::all_lt(t4, half) {
        [zero; 4]
    } else {
        let (g, rho) = g_series(q.m);
        reduce4(&g, &q, rho).map(|x| x / ar0)
    };
    let alpha: [T; 4] = core::array::from_fn(|k| {
        T::select_lt(half, t4, as_one[k], by_grouping(pairs[k], isolated[k]))
    });
    let n = by_grouping(n_pairs, n_isolated);
    [alpha[0], alpha[1], alpha[2], alpha[3], n]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The elementary symmetric functions `[p1, p2, p3, p4]` of four (complex) roots.
    fn coeffs(roots: [Cx<f64>; 4]) -> [f64; 4] {
        let mut e = [
            Cx::real(1.0),
            Cx::real(0.0),
            Cx::real(0.0),
            Cx::real(0.0),
            Cx::real(0.0),
        ];
        for x in roots {
            for k in (1..5).rev() {
                e[k] = e[k] + e[k - 1] * x;
            }
        }
        [e[1].re, e[2].re, e[3].re, e[4].re]
    }

    fn at(a: [f64; 4], u: Cx<f64>) -> Cx<f64> {
        ((Cx::real(a[3]) * u + Cx::real(a[2])) * u + Cx::real(a[1])) * u + Cx::real(a[0])
    }

    /// The interpolant matches φ at every root, relative to its size.
    fn check(roots: [Cx<f64>; 4], tol: f64) {
        let a = log_coeffs_8d(coeffs(roots));
        for &u in &roots {
            // φ at a real root through `phi_real`, which takes u = 1 (a 0/0 in the formula).
            let want = if u.im == 0.0 {
                Cx::real(phi_real(u.re))
            } else {
                phi_complex(u)
            };
            let got = at(a, u);
            let err = (got - want).norm2().sqrt() / (1.0 + want.norm2().sqrt());
            assert!(err <= tol, "roots {roots:?}: at {u:?} {got:?} vs {want:?}");
        }
    }

    fn real(r: [f64; 4]) -> [Cx<f64>; 4] {
        r.map(Cx::real)
    }

    #[test]
    fn real_roots_in_every_grouping() {
        // Spread, one isolated, two pairs, a triple, all four close, repeated, and with boosts.
        for roots in [
            [0.1, 0.35, 0.7, 0.95],
            [0.2, 0.9, 1.0, 1.0],
            [0.5, 0.5, 1.0, 1.0],
            [0.3, 0.3, 0.3, 0.9],
            [0.8, 0.82, 0.85, 0.9],
            [1.0, 1.0, 1.0, 1.0],
            [0.6, 0.6, 0.6, 0.6],
            [0.05, 1.0, 3.0, 40.0],
            [0.5, 0.5, 30.0, 30.0],
            [1.0, 1.0, 2.5, 2.5],
            [0.2, 0.2 + 1e-9, 0.7, 0.7 + 1e-7],
            [0.075, 0.075, 0.075, 96.7],
        ] {
            check(real(roots), 1e-11);
        }
    }

    #[test]
    fn loxodromic_pairs() {
        let z = |re: f64, im: f64| Cx { re, im };
        let bar = |x: Cx<f64>| Cx {
            re: x.re,
            im: -x.im,
        };
        for (a, b) in [
            (z(1.6, 2.2), z(0.4, 0.0)),
            (z(-0.9, 0.003), z(1.1, 0.7)),
            (z(1.9, 0.4), z(1.9, 0.4)),
            (z(1.93, 0.42), z(1.92, 0.43)),
            (z(0.7, 1.6), z(0.7, 1.6 + 1e-9)),
        ] {
            // A conjugate pair with two real roots, or two conjugate pairs.
            let roots = if b.im == 0.0 {
                [a, bar(a), b, z(0.02, 0.0)]
            } else {
                [a, bar(a), b, bar(b)]
            };
            check(roots, 1e-10);
        }
    }

    /// In `f32`, away from half turns: no underflowing guard, no overflowing series.
    #[test]
    fn in_f32() {
        for roots in [
            [0.3, 0.55, 0.7, 0.95],
            [0.5, 0.5, 1.0, 1.0],
            [0.3, 0.3, 0.3, 0.9],
            [0.8, 0.8, 0.8, 0.8],
            [0.6, 1.0, 2.0, 9.0],
        ] {
            let p = coeffs(real(roots)).map(|x| x as f32);
            let a = log_coeffs_8d(p).map(f64::from);
            for u in roots {
                let (got, want) = (at(a, Cx::real(u)).re, phi_real(u));
                assert!(
                    (got - want).abs() < 1e-4 * (1.0 + want),
                    "{roots:?}: {got} vs {want}"
                );
            }
        }
    }

    /// Turning: `Z` interpolates `1/|w|` at the turned roots and 0 at the others, and a plane
    /// past a half turn (`⟨R⟩₀ < 0`) is turned alone.
    #[test]
    fn turning_interpolates_the_turned_weights() {
        let roots = [1e-3, 0.4, 0.8, 1.0];
        let p = coeffs(real(roots));
        let r0 = -p[3].sqrt();
        let t = log_turn_8d(p, r0);
        assert!((t[4] - 1.0).abs() < 1e-12, "one plane turned: {t:?}");
        let alpha = [t[0], t[1], t[2], t[3]];
        let others = |j: usize| {
            (0..4)
                .filter(|&i| i != j)
                .map(|i| roots[i])
                .product::<f64>()
        };
        for (j, &u) in roots.iter().enumerate() {
            let want = if j == 0 {
                1.0 / ((1.0 - u) * others(j)).sqrt()
            } else {
                0.0
            };
            let got = at(alpha, Cx::real(u)).re;
            assert!(
                (got - want).abs() < 1e-9 * (1.0 + want),
                "root {u}: {got} vs {want}"
            );
        }
        let e = turn_polynomial_8d(4.0);
        let want = [1.0, 0.0, 2.0 / 3.0, 0.0, 1.0 / 24.0];
        assert!(
            e.iter().zip(want).all(|(a, b)| (a - b).abs() < 1e-15),
            "{e:?}"
        );
    }
}
