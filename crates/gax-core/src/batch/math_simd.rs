//! The functions of [`math`](super::math) on `fearless_simd` vectors: the same polynomials
//! and reductions, with lane masks for the selects.

use fearless_simd::{Bytes, Select, Simd, SimdBase, SimdFloat, SimdInt, f32x8, i32x8};

const MAGIC: f32 = 12_582_912.0;
const MAGIC_BITS: i32 = 0x4B40_0000;

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

/// Sine and cosine (see [`math::sin_cos`](super::math::sin_cos)).
#[inline(always)]
#[allow(clippy::excessive_precision)]
pub fn sin_cos<S: Simd>(x: f32x8<S>) -> (f32x8<S>, f32x8<S>) {
    const PIO2_1: f32 = 1.570_312_5;
    const PIO2_2: f32 = 4.837_512_969_970_703_125e-4;
    const PIO2_3: f32 = 7.549_789_954_891_882_16e-8;
    let t = x.simd;
    let (q, qi) = round_i(x * core::f32::consts::FRAC_2_PI);
    let r = ((x - q * PIO2_1) - q * PIO2_2) - q * PIO2_3;
    let z = r * r;
    let s = horner(
        z,
        &[-1.951_529_589_1e-4, 8.332_160_873_6e-3, -1.666_665_461_1e-1],
    ) * z
        * r
        + r;
    let c = horner(
        z,
        &[
            2.443_315_711_809_948e-5,
            -1.388_731_625_493_765e-3,
            4.166_664_568_298_827e-2,
        ],
    ) * z
        * z
        - z * 0.5
        + 1.0;
    let zero = ints(t, 0);
    let swap = (qi & ints(t, 1)).simd_eq(zero);
    let (s0, c0) = (swap.select(s, c), swap.select(c, s));
    let sin = (qi & ints(t, 2)).simd_eq(zero).select(s0, -s0);
    let cos = ((qi + 1) & ints(t, 2)).simd_eq(zero).select(c0, -c0);
    (x.simd_eq(0.0).select(x, sin), cos)
}

/// `e^x` (see [`math::exp`](super::math::exp)).
#[inline(always)]
#[allow(clippy::excessive_precision)]
pub fn exp<S: Simd>(x: f32x8<S>) -> f32x8<S> {
    let xc = x.max_precise(-104.0).min_precise(89.0);
    let (n, ni) = round_i(xc * core::f32::consts::LOG2_E);
    let r = xc - n * 0.693_359_375 - n * -2.121_944_400_546_905_827_68e-4;
    let z = r * r;
    let p = horner(
        r,
        &[
            1.987_569_150_0e-4,
            1.398_199_950_7e-3,
            8.333_451_907_3e-3,
            4.166_579_589_4e-2,
            1.666_666_545_9e-1,
            5.000_000_120_1e-1,
        ],
    );
    let y = p * z + r + 1.0;
    let n1 = ni >> 1;
    let e = y * pow2i(n1) * pow2i(ni - n1);
    x.simd_eq(x).select(e, x)
}

/// Natural logarithm (see [`math::ln`](super::math::ln)).
#[inline(always)]
#[allow(clippy::excessive_precision)]
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
    let p = horner(
        m,
        &[
            7.037_683_629_2e-2,
            -1.151_461_031_0e-1,
            1.167_699_874_0e-1,
            -1.242_014_084_6e-1,
            1.424_932_278_7e-1,
            -1.666_805_766_5e-1,
            2.000_071_476_5e-1,
            -2.499_999_399_3e-1,
            3.333_333_117_4e-1,
        ],
    );
    let y = p * m * z - e * 2.121_944_400_546_905_827_68e-4 - z * 0.5;
    let r = m + y + e * 0.693_359_375;
    let r = x.simd_eq(f32::INFINITY).select(x, r);
    let r = x.simd_eq(0.0).select(f32x8::splat(t, f32::NEG_INFINITY), r);
    // x >= 0 is false for negative x and for NaN.
    x.simd_ge(0.0).select(r, f32x8::splat(t, f32::NAN))
}

/// Four-quadrant arctangent of `y / x` (see [`math::atan2`](super::math::atan2)).
#[inline(always)]
#[allow(clippy::excessive_precision)]
pub fn atan2<S: Simd>(y: f32x8<S>, x: f32x8<S>) -> f32x8<S> {
    use core::f32::consts::{FRAC_PI_2, FRAC_PI_4, PI};
    let t = x.simd;
    let (ax, ay) = (x.abs(), y.abs());
    let ylarger = ax.simd_lt(ay);
    let (mx, mn) = (ylarger.select(ay, ax), ylarger.select(ax, ay));
    let one = f32x8::splat(t, 1.0);
    let ratio = mn.simd_eq(f32::INFINITY).select(one, mn / mx);
    let tt = mx.simd_eq(0.0).select(f32x8::splat(t, 0.0), ratio);
    let big = tt.simd_gt(0.414_213_562_373_095_05);
    let u = big.select((tt - 1.0) / (tt + 1.0), tt);
    let z = u * u;
    let a = horner(
        z,
        &[
            8.053_744_495_38e-2,
            -1.387_768_560_32e-1,
            1.997_771_064_78e-1,
            -3.333_294_915_39e-1,
        ],
    ) * z
        * u
        + u;
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
#[allow(clippy::excessive_precision)]
pub fn sinh<S: Simd>(x: f32x8<S>) -> f32x8<S> {
    let a = x.abs();
    let z = x * x;
    let small = horner(
        z,
        &[
            2.037_219_129_45e-4,
            8.330_283_762_39e-3,
            1.666_671_602_11e-1,
        ],
    ) * z
        * x
        + x;
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
