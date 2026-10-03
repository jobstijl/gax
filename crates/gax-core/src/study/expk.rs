//! The exponential of a bivector with three or four invariant planes: 6D to 9D (docs/log6d.md
//! §12).
//!
//! A bivector `B = Σ μⱼ b̂ⱼ` (commuting simple parts, `b̂ⱼ² = 1`) has invariants `λⱼ = μⱼ²`, the
//! roots of `λᵏ − e₁λᵏ⁻¹ + e₂λᵏ⁻² − …` with `eₘ = ⟨Wₘ²⟩₀`, `Wₘ = B^∧m/m!`. The bivectors
//! `Hₘ = ⟨Wₘ Wₘ₋₁⟩₂` weigh plane `j` by `μⱼ` times a polynomial in `λⱼ`, through a triangular
//! matrix with ±1 on its diagonal, so `Bᵢ` of weights `λⱼⁱ μⱼ` are integer polynomial
//! combinations of them ([`h_weights`]). Then
//!
//! ```text
//! exp B = ∏ (cⱼ + sⱼ b̂ⱼ) = C (1 + T + T∧T/2 + T^∧3/6 + T^∧4/24),
//! T = Σ tanh(μⱼ) b̂ⱼ = Σ αᵢ Bᵢ,   C = ∏ cⱼ = ∏(1 − tanh² μⱼ)^(−1/2),
//! ```
//!
//! with `α` interpolating `τ(λ) = tanh(√λ)/√λ` at the roots. `τ` is meromorphic, with poles at
//! the half turns `λ = −(π/2 + nπ)²`, so rotation planes beyond a quarter turn are first turned
//! back by a quarter turn ([`exp_turn_6d`]), and those beyond three quarters halved (by the
//! caller, squaring afterwards; [`exp_reach_6d`]). The interpolants come from the same groupings
//! of the roots as the logarithm's (`log8`), for any analytic data ([`Data`]).

use super::log8::{
    CxSeries, Quartic8, add_root, crt22, cx_div, cx_sqrt, euler, grouping_apart, quartic8,
    reduce_pair_cx, reduce4, split8,
};
use super::{
    Channel, Cx, LOG6_TERMS, Real, Series, add_isolated, cubic6, derivative, integral, reduce2,
    reduce3, series_div, series_sqrt, split6, taylor_shift,
};

/// Terms of `τ`'s and `ln cosh √x`'s Maclaurin series (radius `π²/4`), shifted to centres within
/// 1 of zero: the ratio is at most `0.41`, so 80 terms reach `10⁻³¹`.
const TAU_TERMS: usize = 80;

/// The Maclaurin coefficients of `τ(x) = tanh(√x)/√x`, the quotient of
/// `sinh(√x)/√x = Σ xⁿ/(2n+1)!` by `cosh √x = Σ xⁿ/(2n)!`.
const fn tau_maclaurin() -> [f64; TAU_TERMS] {
    let mut s = [0.0; TAU_TERMS];
    let mut c = [0.0; TAU_TERMS];
    s[0] = 1.0;
    c[0] = 1.0;
    let mut n = 1;
    while n < TAU_TERMS {
        let k = n as f64;
        c[n] = c[n - 1] / ((2.0 * k - 1.0) * (2.0 * k));
        s[n] = s[n - 1] / ((2.0 * k) * (2.0 * k + 1.0));
        n += 1;
    }
    let mut q = [0.0; TAU_TERMS];
    let mut k = 0;
    while k < TAU_TERMS {
        let mut v = s[k];
        let mut j = 1;
        while j <= k {
            v -= c[j] * q[k - j];
            j += 1;
        }
        q[k] = v;
        k += 1;
    }
    q
}

const TAU: [f64; TAU_TERMS] = tau_maclaurin();

/// `π²/4`: the first pole of `τ` is at `−π²/4`.
const POLE: f64 = core::f64::consts::PI * core::f64::consts::PI / 4.0;

/// Data to interpolate at the roots: its values, its Taylor series at a centre (in
/// `s = (x − c)/ρ`), and its distance from a centre to its nearest singularity.
pub(super) trait Data<T: Real> {
    fn at(&self, x: T) -> T;
    fn at_cx(&self, x: Cx<T>) -> Cx<T>;
    fn series(&self, c: T, rho: T) -> Series<T>;
    fn series_cx(&self, c: Cx<T>, rho: T) -> CxSeries<T>;
    fn reach(&self, c: T) -> T;
    fn reach_cx(&self, c: Cx<T>) -> T;
}

/// `e^z` for a complex `z`.
fn cx_exp<T: Real>(z: Cx<T>) -> Cx<T> {
    let (s, c) = z.im.sin_cos();
    let r = z.re.exp();
    Cx {
        re: r * c,
        im: r * s,
    }
}

/// The exponential of a series with constant term `g₀`: `e^g₀ · Σ` by `E' = g' E`.
fn cx_series_exp<T: Real>(g: &CxSeries<T>) -> CxSeries<T> {
    let mut e = [Cx::real(T::zero()); LOG6_TERMS];
    e[0] = cx_exp(g[0]);
    for n in 1..LOG6_TERMS {
        let mut acc = Cx::real(T::zero());
        for k in 1..=n {
            acc = acc + Cx::real(T::from_i64(k as i64)) * g[k] * e[n - k];
        }
        e[n] = acc * Cx::real(T::from_i64(n as i64).recip());
    }
    e
}

/// The first 16 terms of a Maclaurin series `table` at `x`, by Horner.
fn horner16<F: Copy + core::ops::Add<Output = F> + core::ops::Mul<Output = F>>(
    table: &[f64; TAU_TERMS],
    x: F,
    k: impl Fn(f64) -> F,
) -> F {
    let mut acc = k(table[15]);
    for i in (0..15).rev() {
        acc = acc * x + k(table[i]);
    }
    acc
}

/// `τ(x) = tanh(√x)/√x`: the weight of `T` on `B`'s planes.
pub(super) struct Tau;

impl Tau {
    /// `τ` by its Maclaurin series, for small `|x|`.
    fn small<F: Copy + core::ops::Add<Output = F> + core::ops::Mul<Output = F>>(
        x: F,
        k: impl Fn(f64) -> F,
    ) -> F {
        horner16(&TAU, x, k)
    }

    /// The Maclaurin series shifted to `x0` (`|x0| ≤ 1`), in `s = (x − x0)/ρ`.
    fn near<T: Real, F>(
        x0: F,
        rho: T,
        zero: F,
        k: impl Fn(f64) -> F,
        real: impl Fn(T) -> F,
    ) -> [F; LOG6_TERMS]
    where
        F: Copy + core::ops::Add<Output = F> + core::ops::Mul<Output = F>,
    {
        taylor_shift(TAU.map(k), x0, rho, zero, real)
    }
}

impl<T: Real> Data<T> for Tau {
    fn at(&self, x: T) -> T {
        let (zero, one, two) = (T::zero(), T::one(), T::from_i64(2));
        let z = x.abs().max(T::from_f64(1e-30)).sqrt();
        // x > 0: tanh(z)/z, tanh z = 1 − 2/(e^2z + 1) (no overflow); x < 0: tan(z)/z.
        let above = (one - two / ((z + z).exp() + one)) / z;
        let (s, c) = z.sin_cos();
        let below = s / (c * z);
        let series = Tau::small(x, |v| T::from_f64(v));
        let far = T::select_lt(x, zero, below, above);
        T::select_lt(x.abs(), T::from_f64(0.05), series, far)
    }

    fn at_cx(&self, x: Cx<T>) -> Cx<T> {
        let one = Cx::real(T::one());
        let z = x.sqrt();
        // tanh z = (1 − e^−2z)/(1 + e^−2z), with Re z ≥ 0 (the principal root).
        let w = cx_exp(Cx::real(T::from_i64(-2)) * z);
        let far = (one - w) / ((one + w) * z);
        let series = Tau::small(x, |v| Cx::real(T::from_f64(v)));
        Cx::select(x.norm2(), T::from_f64(0.0025), series, far)
    }

    fn series(&self, c: T, rho: T) -> Series<T> {
        let one = T::one();
        let zero = [T::zero(); LOG6_TERMS];
        let near = if T::all_lt(one, c) {
            zero
        } else {
            let x0 = c.min(one);
            Tau::near(x0, rho, T::zero(), |v| T::from_f64(v), |r| r)
        };
        let far = if T::all_lt(c, one) {
            zero
        } else {
            // z = √(c + ρ s), w = e^(−2z), tanh z = (1 − w)/(1 + w), τ = tanh(z)/z.
            let cc = c.max(one);
            let z = series_sqrt(cc, 1.0, false, rho);
            let g: CxSeries<T> = core::array::from_fn(|k| Cx::real(T::from_i64(-2) * z[k]));
            let w = cx_series_exp(&g);
            let num: Series<T> =
                core::array::from_fn(|k| if k == 0 { one - w[0].re } else { -w[k].re });
            let den: Series<T> =
                core::array::from_fn(|k| if k == 0 { one + w[0].re } else { w[k].re });
            series_div(&series_div(&num, &den), &z)
        };
        core::array::from_fn(|k| T::select_lt(one, c, far[k], near[k]))
    }

    fn series_cx(&self, c: Cx<T>, rho: T) -> CxSeries<T> {
        let one = Cx::real(T::one());
        let zero = [Cx::real(T::zero()); LOG6_TERMS];
        let ac = c.norm2().sqrt();
        let near = if T::all_lt(T::one(), ac) {
            zero
        } else {
            Tau::near(
                c,
                rho,
                Cx::real(T::zero()),
                |v| Cx::real(T::from_f64(v)),
                Cx::real,
            )
        };
        let far = if T::all_lt(ac, T::one()) {
            zero
        } else {
            let cc = Cx::select(ac, T::one(), one, c);
            let z = cx_sqrt(cc, false, rho);
            let g: CxSeries<T> = core::array::from_fn(|k| Cx::real(T::from_i64(-2)) * z[k]);
            let w = cx_series_exp(&g);
            let num: CxSeries<T> =
                core::array::from_fn(|k| if k == 0 { one - w[0] } else { -w[k] });
            let den: CxSeries<T> = core::array::from_fn(|k| if k == 0 { one + w[0] } else { w[k] });
            cx_div(&cx_div(&num, &den), &z)
        };
        core::array::from_fn(|k| Cx::select(T::one(), ac, far[k], near[k]))
    }

    fn reach(&self, c: T) -> T {
        (c + T::from_f64(POLE)).abs()
    }

    fn reach_cx(&self, c: Cx<T>) -> T {
        (c + Cx::real(T::from_f64(POLE))).norm2().sqrt()
    }
}

/// The Maclaurin coefficients of `ln cosh √x`: the logarithm of `Σ xⁿ/(2n)!` (`L' = C'/C`).
const fn lncosh_maclaurin() -> [f64; TAU_TERMS] {
    let mut c = [0.0; TAU_TERMS];
    c[0] = 1.0;
    let mut n = 1;
    while n < TAU_TERMS {
        let k = n as f64;
        c[n] = c[n - 1] / ((2.0 * k - 1.0) * (2.0 * k));
        n += 1;
    }
    // L = Σ lₙ xⁿ with n lₙ = n cₙ − Σ_{k=1}^{n−1} k lₖ c_{n−k} (c₀ = 1).
    let mut l = [0.0; TAU_TERMS];
    let mut n = 1;
    while n < TAU_TERMS {
        let mut v = n as f64 * c[n];
        let mut k = 1;
        while k < n {
            v -= k as f64 * l[k] * c[n - k];
            k += 1;
        }
        l[n] = v / n as f64;
        n += 1;
    }
    l
}

const LNCOSH: [f64; TAU_TERMS] = lncosh_maclaurin();

/// `ln cosh √x`: its trace over the roots is `ln C = Σ ln cosh μⱼ`, without the cancellation
/// of `∏(1 − tanh² μⱼ)` for large boosts.
pub(super) struct LnCosh;

/// The series of `ln(1 + w)` for a series `w` (`|w₀| < 1`): `L' = w'/(1 + w)`.
fn cx_series_log1p<T: Real>(w: &CxSeries<T>) -> CxSeries<T> {
    let (one, zero) = (Cx::real(T::one()), Cx::real(T::zero()));
    let mut den = *w;
    den[0] = den[0] + one;
    let d = cx_div(&derivative(w, zero, Cx::real), &den);
    integral(&d, (one + w[0]).ln(), zero, Cx::real)
}

impl LnCosh {
    fn small<F: Copy + core::ops::Add<Output = F> + core::ops::Mul<Output = F>>(
        x: F,
        k: impl Fn(f64) -> F,
    ) -> F {
        horner16(&LNCOSH, x, k)
    }

    /// `z + ln(1 + e^−2z) − ln 2` as a series, `z = √(c + ρ s)` (`Re z > 0`).
    fn far<T: Real>(z: &CxSeries<T>) -> CxSeries<T> {
        let g: CxSeries<T> = core::array::from_fn(|k| Cx::real(T::from_i64(-2)) * z[k]);
        let w = cx_series_exp(&g);
        let l = cx_series_log1p(&w);
        let ln2 = Cx::real(T::from_f64(core::f64::consts::LN_2));
        core::array::from_fn(|k| {
            if k == 0 {
                z[0] + l[0] - ln2
            } else {
                z[k] + l[k]
            }
        })
    }
}

impl<T: Real> Data<T> for LnCosh {
    fn at(&self, x: T) -> T {
        let one = T::one();
        let z = x.abs().max(T::from_f64(1e-30)).sqrt();
        let above = z + (one + (-(z + z)).exp()).ln() - T::from_f64(core::f64::consts::LN_2);
        let below = z.cos().ln();
        let series = LnCosh::small(x, |v| T::from_f64(v));
        let far = T::select_lt(x, T::zero(), below, above);
        T::select_lt(x.abs(), T::from_f64(0.05), series, far)
    }

    fn at_cx(&self, x: Cx<T>) -> Cx<T> {
        let one = Cx::real(T::one());
        let z = x.sqrt();
        let w = cx_exp(Cx::real(T::from_i64(-2)) * z);
        let far = z + (one + w).ln() - Cx::real(T::from_f64(core::f64::consts::LN_2));
        let series = LnCosh::small(x, |v| Cx::real(T::from_f64(v)));
        Cx::select(x.norm2(), T::from_f64(0.0025), series, far)
    }

    fn series(&self, c: T, rho: T) -> Series<T> {
        let one = T::one();
        let zero = [T::zero(); LOG6_TERMS];
        let near = if T::all_lt(one, c) {
            zero
        } else {
            taylor_shift(LNCOSH.map(T::from_f64), c.min(one), rho, T::zero(), |p| p)
        };
        let far = if T::all_lt(c, one) {
            zero
        } else {
            let z = series_sqrt(c.max(one), 1.0, false, rho).map(Cx::real);
            LnCosh::far(&z).map(|x| x.re)
        };
        core::array::from_fn(|k| T::select_lt(one, c, far[k], near[k]))
    }

    fn series_cx(&self, c: Cx<T>, rho: T) -> CxSeries<T> {
        let one = Cx::real(T::one());
        let zero = [Cx::real(T::zero()); LOG6_TERMS];
        let ac = c.norm2().sqrt();
        let near = if T::all_lt(T::one(), ac) {
            zero
        } else {
            let m = LNCOSH.map(|v| Cx::real(T::from_f64(v)));
            taylor_shift(m, c, rho, Cx::real(T::zero()), Cx::real)
        };
        let far = if T::all_lt(ac, T::one()) {
            zero
        } else {
            let cc = Cx::select(ac, T::one(), one, c);
            LnCosh::far(&cx_sqrt(cc, false, rho))
        };
        core::array::from_fn(|k| Cx::select(T::one(), ac, far[k], near[k]))
    }

    fn reach(&self, c: T) -> T {
        (c + T::from_f64(POLE)).abs()
    }

    fn reach_cx(&self, c: Cx<T>) -> T {
        (c + Cx::real(T::from_f64(POLE))).norm2().sqrt()
    }
}

/// The rotation planes beyond a quarter turn (`λ < −π²/16`): `1/θ` there (`inverse`, the weight
/// of their unit bivectors' sum on `B`), or 1 (their number, as a trace), and 0 elsewhere.
pub(super) struct Beyond {
    inverse: bool,
}

/// `−π²/16`: a quarter turn.
const QUARTER: f64 = -core::f64::consts::PI * core::f64::consts::PI / 16.0;

impl<T: Real> Data<T> for Beyond {
    fn at(&self, x: T) -> T {
        let v = if self.inverse {
            (-x).max(T::from_f64(1e-30)).sqrt().recip()
        } else {
            T::one()
        };
        T::select_lt(x, T::from_f64(QUARTER), v, T::zero())
    }

    fn at_cx(&self, _: Cx<T>) -> Cx<T> {
        Cx::real(T::zero())
    }

    fn series(&self, c: T, rho: T) -> Series<T> {
        let tiny = T::from_f64(1e-30);
        let s = if self.inverse {
            series_sqrt((-c).max(tiny), -1.0, true, rho)
        } else {
            core::array::from_fn(|k| if k == 0 { T::one() } else { T::zero() })
        };
        core::array::from_fn(|k| T::select_lt(c, T::from_f64(QUARTER), s[k], T::zero()))
    }

    fn series_cx(&self, _: Cx<T>, _: T) -> CxSeries<T> {
        [Cx::real(T::zero()); LOG6_TERMS]
    }

    /// The distance to `0`, where `1/θ` is singular, or, near it, to the quarter turn: the data
    /// vanish between the two. (A cluster across the quarter turn is turned or not as a whole,
    /// by its centre.)
    fn reach(&self, c: T) -> T {
        c.abs().max(c - T::from_f64(QUARTER))
    }

    fn reach_cx(&self, c: Cx<T>) -> T {
        c.norm2().sqrt()
    }
}

/// The line through the data at a pair of roots with midpoint `mid` and half-difference `d`
/// (`d2 = d²`, negative for a conjugate pair): its series at the midpoint where the pair is
/// within a quarter of its reach, else the chord.
fn pair_line<T: Real, D: Data<T>>(data: &D, mid: T, d2: T, d: T) -> [T; 2] {
    let tiny = T::from_f64(1e-30);
    let reach = data.reach(mid);
    let quarter = reach * T::from_f64(0.25);
    let close = if T::all_lt(quarter, d) {
        [T::zero(); 2]
    } else {
        let rho = reach.max(tiny);
        reduce2(&data.series(mid, rho), mid, rho, d2)
    };
    let (a, b) = (mid + d, mid - d);
    let (fa, fb) = (data.at(a), data.at(b));
    let slope = (fa - fb) / (a - b).max(tiny);
    let real_far = [fa - slope * a, slope];
    let f = data.at_cx(Cx { re: mid, im: d });
    let slope = f.im / d.max(tiny);
    let conj_far = [f.re - slope * mid, slope];
    core::array::from_fn(|k| {
        let far = T::select_lt(d2, T::zero(), conj_far[k], real_far[k]);
        T::select_lt(d, quarter, close[k], far)
    })
}

/// The line through the data at the roots of `u² − s u + p` (complex).
fn pair_line_cx<T: Real, D: Data<T>>(data: &D, s: Cx<T>, p: Cx<T>) -> [Cx<T>; 2] {
    let half = Cx::real(T::from_f64(0.5));
    let mid = s * half;
    let d2 = mid * mid - p;
    let d = d2.sqrt();
    let ad = d.norm2().sqrt();
    let reach = data.reach_cx(mid);
    let quarter = reach * T::from_f64(0.25);
    let close = if T::all_lt(quarter, ad) {
        [Cx::real(T::zero()); 2]
    } else {
        let rho = reach.max(T::from_f64(1e-30));
        reduce_pair_cx(&data.series_cx(mid, rho), mid, rho, d2)
    };
    let (z, w) = (mid + d, mid - d);
    let (fz, fw) = (data.at_cx(z), data.at_cx(w));
    let d_safe = Cx::select(ad, T::from_f64(1e-30), Cx::real(T::one()), d + d);
    let slope = (fz - fw) / d_safe;
    let chord = [fz - slope * z, slope];
    [
        Cx::select(ad, quarter, close[0], chord[0]),
        Cx::select(ad, quarter, close[1], chord[1]),
    ]
}

/// The quadratic interpolating the data at the roots of `t³ − p₁t² + p₂t − p₃`: their series at
/// the mean where they are close, else the most isolated real root and the remaining pair.
pub(super) fn interp3<T: Real, D: Data<T>>(p: [T; 3], data: &D) -> [T; 3] {
    let [p1, p2, p3] = p;
    let c = cubic6(p1, p2, p3);
    let reach = data.reach(c.m);
    let limit = reach * T::from_f64(0.25);
    let jet = if T::all_lt(limit, c.bound) {
        [T::zero(); 3]
    } else {
        let rho = reach.max(T::from_f64(1e-30));
        reduce3(&data.series(c.m, rho), c.m, rho, c.e2, c.e3)
    };
    if T::all_lt(c.bound, limit) {
        return jet;
    }
    let sp = split6(&c, p1, p2, p3);
    let pair = pair_line(data, sp.mid, sp.d2, sp.d);
    let spread = add_isolated(pair, data.at(sp.r), sp.r, sp.sum, sp.prod);
    core::array::from_fn(|k| T::select_lt(c.bound, limit, jet[k], spread[k]))
}

/// The cubic interpolating the data at the roots of `t⁴ − p₁t³ + p₂t² − p₃t + p₄`, through the
/// grouping of the roots that separates them best (as `log_coeffs_8d`).
pub(super) fn interp4<T: Real, D: Data<T>>(p: [T; 4], data: &D) -> [T; 4] {
    let zero = T::zero();
    let half = T::from_f64(0.5);
    let q = quartic8(p);
    let reach = data.reach(q.m);
    let limit = reach * T::from_f64(0.25);
    let jet = if T::all_lt(limit, q.bound) {
        [zero; 4]
    } else {
        let rho = reach.max(T::from_f64(1e-30));
        reduce4(&data.series(q.m, rho), &q, rho)
    };
    if T::all_lt(q.bound, limit) {
        return jet;
    }
    let sp = split8(&q, p);
    let (use_r, use_pairs, use_conj) = grouping_apart(&sp);
    let isolated = if T::all_lt(use_r, half) {
        [zero; 4]
    } else {
        add_root(interp3(sp.cubic, data), sp.cubic, data.at(sp.r), sp.r)
    };
    let pairs = if T::all_lt(use_pairs, half) {
        [zero; 4]
    } else {
        let line = |(s, p): (T, T)| {
            let mid = s * half;
            let d2 = mid * mid - p;
            pair_line(data, mid, d2, d2.abs().sqrt())
        };
        let (sa, pa) = sp.pair_a;
        let (sb, pb) = sp.pair_b;
        crt22(line(sp.pair_a), sa, pa, line(sp.pair_b), sb, pb)
    };
    let conj = if T::all_lt(use_conj, half) {
        [zero; 4]
    } else {
        let (s, p) = sp.conj;
        let bar = |x: Cx<T>| Cx {
            re: x.re,
            im: -x.im,
        };
        let l = pair_line_cx(data, s, p);
        crt22(l, s, p, [bar(l[0]), bar(l[1])], bar(s), bar(p)).map(|x| x.re)
    };
    core::array::from_fn(|k| {
        let spread = T::select_lt(
            half,
            use_conj,
            conj[k],
            T::select_lt(half, use_pairs, pairs[k], isolated[k]),
        );
        T::select_lt(q.bound, limit, jet[k], spread)
    })
}

/// The weights `[w₁, …, w_k]` of `H₁ = B` and `Hₘ = ⟨Wₘ Wₘ₋₁⟩₂` in `Σ αᵢ Bᵢ`, `Bᵢ` the bivector
/// of plane weights `λⱼⁱ μⱼ`: from `Hₘ = Σₜ (−1)ᵗ eₘ₋₁₋ₜ Bₜ` (`B₀ = B`).
#[inline]
pub fn h_weights<T: Real, const K: usize>(alpha: [T; K], e: [T; K]) -> [T; K] {
    let zero = T::zero();
    let get = |a: &[T; K], i: usize| if i < K { a[i] } else { zero };
    let (a0, a1, a2, a3) = (
        get(&alpha, 0),
        get(&alpha, 1),
        get(&alpha, 2),
        get(&alpha, 3),
    );
    let (e1, e2, e3) = (get(&e, 0), get(&e, 1), get(&e, 2));
    let two = T::from_i64(2);
    let w = [
        a0 + a1 * e1 + a2 * (e1 * e1 - e2) + a3 * ((e1 * e1 - two * e2) * e1 + e3),
        a3 * (e2 - e1 * e1) - a1 - a2 * e1,
        a2 + a3 * e1,
        -a3,
    ];
    core::array::from_fn(|i| w[i])
}

/// `Σⱼ P(λⱼ)` for the interpolant `P = Σ αᵢ λⁱ`: the power sums `Σ λⱼⁱ` from the `e`s
/// (Newton's identities).
fn trace<T: Real, const K: usize>(alpha: [T; K], e: [T; K]) -> T {
    let mut p = [T::zero(); K];
    p[0] = T::from_i64(K as i64);
    // pᵢ = e₁pᵢ₋₁ − e₂pᵢ₋₂ + … + (−1)ⁱ⁻¹ i eᵢ.
    for i in 1..K {
        let mut v = T::zero();
        for j in 1..i {
            let t = e[j - 1] * p[i - j];
            v = if j % 2 == 1 { v + t } else { v - t };
        }
        let t = T::from_i64(i as i64) * e[i - 1];
        p[i] = if i % 2 == 1 { v + t } else { v - t };
    }
    (0..K).fold(T::zero(), |acc, i| acc + alpha[i] * p[i])
}

/// The weights `[w₁, w₂, w₃]` with `T = Σ wₘ Hₘ = Σ tanh(μⱼ) b̂ⱼ` for a bivector with three
/// invariant planes (6D, 7D), from `e = [e₁, e₂, e₃]`, and `C = ∏ cosh μⱼ`:
/// `exp B = C (1 + T + T∧T/2 + T^∧3/6)`. Right where every rotation plane is within a quarter
/// turn, or nearly ([`exp_turn_6d`]).
#[inline]
pub fn exp_weights_6d<T: Real>(e: [T; 3]) -> [T; 4] {
    let w = h_weights(interp3(e, &Tau), e);
    let c = trace(interp3(e, &LnCosh), e).exp();
    [w[0], w[1], w[2], c]
}

/// [`exp_weights_6d`] for four invariant planes (8D, 9D).
#[inline]
pub fn exp_weights_8d<T: Real>(e: [T; 4]) -> [T; 5] {
    let w = h_weights(interp4(e, &Tau), e);
    let c = trace(interp4(e, &LnCosh), e).exp();
    [w[0], w[1], w[2], w[3], c]
}

/// The rotation planes beyond a quarter turn, to turn back by one: `[w₁, w₂, w₃, n]` with
/// `Z = Σ wₘ Hₘ` the sum of their unit bivectors `êⱼ` (`êⱼ² = −1`, oriented with `B`) and `n`
/// their number. Then `B = B' + (π/2) Z` with every rotation of `B'` within a quarter turn, and
/// `exp B = exp B' · ∏ êⱼ = exp B' · (−1)ⁿ ∏(−êⱼ)`, a polynomial in `Z` (`turn_polynomial`).
#[inline]
pub fn exp_turn_6d<T: Real>(e: [T; 3]) -> [T; 4] {
    let z = h_weights(interp3(e, &Beyond { inverse: true }), e);
    let n = trace(interp3(e, &Beyond { inverse: false }), e);
    [z[0], z[1], z[2], n]
}

/// [`exp_turn_6d`] for four invariant planes.
#[inline]
pub fn exp_turn_8d<T: Real>(e: [T; 4]) -> [T; 5] {
    let z = h_weights(interp4(e, &Beyond { inverse: true }), e);
    let n = trace(interp4(e, &Beyond { inverse: false }), e);
    [z[0], z[1], z[2], z[3], n]
}

/// The largest rapidity left to the closed form (larger ones are halved and squared back).
const BOOST: f64 = 8.0;

/// How far the planes are from what turning handles, `[reach, turn]`: the largest of a
/// rotation's angle over `3π/4` and a loxodromic pair's rotation over `π/4` (above 1, the caller
/// halves `B` and squares the result), and the largest rotation over `1.1 π/4` (above 1, some
/// plane needs turning).
fn reach_of<T: Real>(roots: &[Cx<T>]) -> [T; 2] {
    let tiny = T::from_f64(1e-12);
    let (mut worst, mut turn) = (T::zero(), T::zero());
    for &l in roots {
        let mu = l.sqrt();
        let real = T::select_lt(
            l.im.abs(),
            tiny * (T::one() + l.re.abs()),
            T::one(),
            T::zero(),
        );
        let limit = T::select_lt(
            T::from_f64(0.5),
            real,
            T::from_f64(0.75 * core::f64::consts::PI),
            T::from_f64(0.25 * core::f64::consts::PI),
        );
        worst = worst.max(mu.im.abs() / limit);
        // A large boost: the interpolant cancels about `|λ|^(3/2)` ulps (its slope at the other
        // roots times `λ`, against `τ(λ) ~ 1/√λ`), so it is halved too.
        worst = worst.max(mu.re.abs() / T::from_f64(BOOST));
        // A rotation clearly beyond a quarter turn (τ is still accurate a little beyond).
        turn = turn.max(real * mu.im.abs() / T::from_f64(0.275 * core::f64::consts::PI));
    }
    [worst, turn]
}

/// How far the roots of `λ³ − e₁λ² + e₂λ − e₃` are from what the closed form handles,
/// `[reach, turn]`: above 1, `reach` asks the caller to halve `B` (a rotation beyond `3π/4`, a
/// loxodromic pair's rotation beyond `π/4`, a rapidity over 8), and `turn` to turn rotations
/// beyond a quarter turn back first ([`exp_turn_6d`]).
#[inline]
pub fn exp_reach_6d<T: Real>(e: [T; 3]) -> [T; 2] {
    let [e1, e2, e3] = e;
    let c = cubic6(e1, e2, e3);
    let sp = split6(&c, e1, e2, e3);
    let complex = T::select_lt(sp.d2, T::zero(), T::one(), T::zero());
    let a = Cx {
        re: sp.mid + sp.d * (T::one() - complex),
        im: sp.d * complex,
    };
    let b = Cx {
        re: sp.mid - sp.d * (T::one() - complex),
        im: -(sp.d * complex),
    };
    reach_of(&[Cx::real(sp.r), a, b])
}

/// [`exp_reach_6d`] for the roots of `λ⁴ − e₁λ³ + e₂λ² − e₃λ + e₄`.
#[inline]
pub fn exp_reach_8d<T: Real>(e: [T; 4]) -> [T; 2] {
    let q: Quartic8<T> = quartic8(e);
    let (_, t) = euler(&q);
    let m = Cx::real(q.m);
    reach_of(&t.map(|x| x + m))
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate std;
    use std::{vec, vec::Vec};

    fn tau(mu: Cx<f64>) -> Cx<f64> {
        // tanh(μ)/μ.
        let e = cx_exp(Cx::real(-2.0) * mu);
        (Cx::real(1.0) - e) / ((Cx::real(1.0) + e) * mu)
    }

    fn e_of(l: &[Cx<f64>]) -> Vec<f64> {
        let mut e = vec![Cx::real(1.0); 1];
        e.resize(l.len() + 1, Cx::real(0.0));
        for x in l {
            for k in (1..=l.len()).rev() {
                e[k] = e[k] + e[k - 1] * *x;
            }
        }
        e[1..].iter().map(|z| z.re).collect()
    }

    fn check(l: &[Cx<f64>]) {
        let e = e_of(l);
        let alpha: Vec<f64> = if l.len() == 3 {
            interp3([e[0], e[1], e[2]], &Tau).to_vec()
        } else {
            interp4([e[0], e[1], e[2], e[3]], &Tau).to_vec()
        };
        for &x in l {
            let mut at = Cx::real(0.0);
            for a in alpha.iter().rev() {
                at = at * x + Cx::real(*a);
            }
            let want = if x.norm2() < 1e-20 {
                Cx::real(1.0)
            } else {
                tau(x.sqrt())
            };
            let err = (at - want).norm2().sqrt();
            assert!(
                err < 1e-11 * (1.0 + want.norm2().sqrt()),
                "{l:?} at {x:?}: {at:?} vs {want:?}"
            );
        }
    }

    #[test]
    fn interpolates_tau() {
        let r = Cx::real;
        let z = |re, im| Cx { re, im };
        // Rotations within a quarter turn, boosts, null planes, clusters, conjugate pairs.
        for l in [
            vec![r(-0.3), r(-0.1), r(0.0)],
            vec![r(-0.6), r(0.0), r(0.0)],
            vec![r(0.0), r(0.0), r(0.0)],
            vec![r(-0.2), r(4.0), r(25.0)],
            vec![z(0.3, 0.4), z(0.3, -0.4), r(-0.5)],
            vec![r(-0.6), r(-0.3), r(0.0), r(9.0)],
            vec![r(-0.4), r(-0.4), r(1.0), r(1.0)],
            vec![z(0.2, 0.5), z(0.2, -0.5), z(1.5, 0.3), z(1.5, -0.3)],
            vec![r(-0.5), r(-0.5 + 1e-9), r(-0.5 - 1e-9), r(0.0)],
            vec![r(0.0), r(-0.15), r(-0.4), r(-0.7)],
            vec![r(0.0), r(-0.3), r(-0.3), r(-0.65)],
            vec![r(0.0), r(-0.159_688_25), r(-0.507_727_35), r(-0.584_687_6)],
        ] {
            check(&l);
        }
    }

    #[test]
    fn lncosh_maclaurin_matches() {
        for x in [-1.0f64, -0.3, 0.2, 0.9] {
            let want = if x < 0.0 {
                (-x).sqrt().cos().ln()
            } else {
                x.sqrt().cosh().ln()
            };
            let mut s = 0.0;
            for a in LNCOSH.iter().rev() {
                s = s * x + a;
            }
            assert!((s - want).abs() < 1e-14, "{x}: {s} vs {want}");
        }
        // ln C over the roots, against Σ ln cosh μ.
        let l = [-0.5f64, 0.3, 4.0, 30.0];
        let e = [
            l.iter().sum::<f64>(),
            l[0] * l[1] + l[0] * l[2] + l[0] * l[3] + l[1] * l[2] + l[1] * l[3] + l[2] * l[3],
            l[0] * l[1] * l[2] + l[0] * l[1] * l[3] + l[0] * l[2] * l[3] + l[1] * l[2] * l[3],
            l[0] * l[1] * l[2] * l[3],
        ];
        let got = trace(interp4(e, &LnCosh), e);
        let want: f64 = l.iter().map(|&x| LnCosh.at(x)).sum();
        assert!(
            (got - want).abs() < 1e-12 * (1.0 + want.abs()),
            "{got} vs {want}"
        );
    }

    #[test]
    fn tau_maclaurin_matches() {
        for x in [-1.0f64, -0.3, 0.2, 0.9] {
            let want = if x < 0.0 {
                (-x).sqrt().tan() / (-x).sqrt()
            } else {
                x.sqrt().tanh() / x.sqrt()
            };
            let mut s = 0.0;
            for a in TAU.iter().rev() {
                s = s * x + a;
            }
            assert!((s - want).abs() < 1e-14, "{x}: {s} vs {want}");
        }
    }
}
