//! The functions of [`math`](crate::math) on `fearless_simd` vectors: the same polynomials
//! and reductions, with lane masks for the selects.

use crate::math::cephes::{
    ATAN, COS, EXP, LN, LN2_HI, LN2_LO, MAGIC, MAGIC_BITS, PIO2, SIN, SINH, TAN_PI_8,
};
use fearless_simd::{Bytes, Select, Simd, SimdBase, SimdFloat, SimdInt, f32x8, i32x8};

#[inline(always)]
fn horner<S: Simd>(x: f32x8<S>, c: &[f32]) -> f32x8<S> {
    c[1..]
        .iter()
        .fold(f32x8::splat(x.simd, c[0]), |acc, &k| acc * x + k)
}

#[inline(always)]
fn round_i<S: Simd>(x: f32x8<S>) -> (f32x8<S>, i32x8<S>) {
    let q = x + MAGIC;
    (q - MAGIC, q.bitcast::<i32x8<S>>() - MAGIC_BITS)
}

#[inline(always)]
fn pow2i<S: Simd>(n: i32x8<S>) -> f32x8<S> {
    ((n + 127) << 23).bitcast::<f32x8<S>>()
}

#[inline(always)]
fn ints<S: Simd>(t: S, v: i32) -> i32x8<S> {
    i32x8::splat(t, v)
}

/// Sine and cosine (see [`math::sin_cos`](crate::math::sin_cos)).
#[inline(always)]
pub fn sin_cos<S: Simd>(x: f32x8<S>) -> (f32x8<S>, f32x8<S>) {
    let t = x.simd;
    let (q, qi) = round_i(x * core::f32::consts::FRAC_2_PI);
    let r = ((x - q * PIO2[0]) - q * PIO2[1]) - q * PIO2[2];
    let z = r * r;
    let s = horner(z, &SIN) * z * r + r;
    let c = horner(z, &COS) * z * z - z * 0.5 + 1.0;
    let zero = ints(t, 0);
    let swap = (qi & ints(t, 1)).simd_eq(zero);
    let (s0, c0) = (swap.select(s, c), swap.select(c, s));
    let sin = (qi & ints(t, 2)).simd_eq(zero).select(s0, -s0);
    let cos = ((qi + 1) & ints(t, 2)).simd_eq(zero).select(c0, -c0);
    (x.simd_eq(0.0).select(x, sin), cos)
}

/// `e^x` (see [`math::exp`](crate::math::exp)).
#[inline(always)]
pub fn exp<S: Simd>(x: f32x8<S>) -> f32x8<S> {
    let xc = x.max_precise(-104.0).min_precise(89.0);
    let (n, ni) = round_i(xc * core::f32::consts::LOG2_E);
    let r = xc - n * LN2_HI - n * LN2_LO;
    let z = r * r;
    let y = horner(r, &EXP) * z + r + 1.0;
    let n1 = ni >> 1;
    let e = y * pow2i(n1) * pow2i(ni - n1);
    x.simd_eq(x).select(e, x)
}

/// Natural logarithm (see [`math::ln`](crate::math::ln)).
#[inline(always)]
pub fn ln<S: Simd>(x: f32x8<S>) -> f32x8<S> {
    let t = x.simd;
    let sub = x.simd_lt(f32::MIN_POSITIVE);
    let xs = sub.select(x * 8_388_608.0, x);
    let bits = xs.bitcast::<i32x8<S>>();
    let e = ((bits >> 23) & ints(t, 0xff)) - 126 - sub.select(ints(t, 23), ints(t, 0));
    let m = ((bits & ints(t, 0x007f_ffff)) | ints(t, 0x3f00_0000)).bitcast::<f32x8<S>>();
    let small = m.simd_lt(core::f32::consts::FRAC_1_SQRT_2);
    let e: f32x8<S> = (e - small.select(ints(t, 1), ints(t, 0))).to_float();
    let m = small.select(m + m - 1.0, m - 1.0);
    let z = m * m;
    let p = horner(m, &LN);
    let y = p * m * z + e * LN2_LO - z * 0.5;
    let r = m + y + e * LN2_HI;
    let r = x.simd_eq(f32::INFINITY).select(x, r);
    let r = x.simd_eq(0.0).select(f32x8::splat(t, f32::NEG_INFINITY), r);
    // x >= 0 is false for negative x and for NaN.
    x.simd_ge(0.0).select(r, f32x8::splat(t, f32::NAN))
}

/// Four-quadrant arctangent of `y / x` (see [`math::atan2`](crate::math::atan2)).
#[inline(always)]
pub fn atan2<S: Simd>(y: f32x8<S>, x: f32x8<S>) -> f32x8<S> {
    use core::f32::consts::{FRAC_PI_2, FRAC_PI_4, PI};
    let t = x.simd;
    let (ax, ay) = (x.abs(), y.abs());
    let ylarger = ax.simd_lt(ay);
    let (mx, mn) = (ylarger.select(ay, ax), ylarger.select(ax, ay));
    let one = f32x8::splat(t, 1.0);
    let ratio = mn.simd_eq(f32::INFINITY).select(one, mn / mx);
    let tt = mx.simd_eq(0.0).select(f32x8::splat(t, 0.0), ratio);
    let big = tt.simd_gt(TAN_PI_8);
    let u = big.select((tt - 1.0) / (tt + 1.0), tt);
    let z = u * u;
    let a = horner(z, &ATAN) * z * u + u;
    let a = big.select(a + FRAC_PI_4, a);
    let a = ylarger.select(f32x8::splat(t, FRAC_PI_2) - a, a);
    let xneg = one.copysign(x).simd_lt(0.0);
    let a = xneg.select(f32x8::splat(t, PI) - a, a);
    let a = a.copysign(y);
    let nan = x.simd_eq(x) & y.simd_eq(y);
    nan.select(a, f32x8::splat(t, f32::NAN))
}

/// Hyperbolic sine.
#[inline(always)]
pub fn sinh<S: Simd>(x: f32x8<S>) -> f32x8<S> {
    let a = x.abs();
    let z = x * x;
    let small = horner(z, &SINH) * z * x + x;
    let e = exp(a);
    let big = ((e - f32x8::splat(x.simd, 1.0) / e) * 0.5).copysign(x);
    a.simd_lt(1.0).select(small, big)
}

/// Hyperbolic cosine.
#[inline(always)]
pub fn cosh<S: Simd>(x: f32x8<S>) -> f32x8<S> {
    let e = exp(x.abs());
    (e + f32x8::splat(x.simd, 1.0) / e) * 0.5
}
