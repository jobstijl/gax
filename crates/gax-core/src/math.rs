//! Branch-free `f32` elementary functions: the ones SIMD lanes use (`batch`), and the ones every
//! `f32` uses with the `deterministic` feature.
//!
//! Each function is written for one lane with selects instead of branches, so that inside a
//! lane loop the compiler vectorizes it at whatever SIMD level the dispatcher selected. The
//! polynomials are the single-precision minimax fits from Cephes (S. L. Moshier), with
//! Cody–Waite argument reduction. Accuracy is within a few ulp of the correctly rounded result
//! for the ranges documented on each function; outside them results degrade gracefully.

use core::f32::consts::{FRAC_PI_2, FRAC_PI_4, PI};

/// `1.5 * 2^23`: adding it rounds an `f32` of magnitude below `2^22` to an integer, whose
/// two's-complement value is then in the low mantissa bits.
const MAGIC: f32 = 12_582_912.0;
const MAGIC_BITS: i32 = 0x4B40_0000;

#[inline(always)]
fn select(c: bool, a: f32, b: f32) -> f32 {
    if c { a } else { b }
}

/// `c[0] x^(n-1) + ... + c[n-1]`.
#[inline(always)]
fn horner(x: f32, c: &[f32]) -> f32 {
    c[1..].iter().fold(c[0], |acc, &k| acc * x + k)
}

/// Round to the nearest integer; returns the rounded value and the integer.
#[inline(always)]
fn round_i(x: f32) -> (f32, i32) {
    let q = x + MAGIC;
    (
        q - MAGIC,
        q.to_bits().cast_signed().wrapping_sub(MAGIC_BITS),
    )
}

/// `2^n` for `-126 <= n <= 127`.
#[inline(always)]
fn pow2i(n: i32) -> f32 {
    f32::from_bits((n + 127).cast_unsigned() << 23)
}

/// Sine and cosine. Accurate for `|x| <= 8192`; the reduction loses accuracy above that.
#[inline(always)]
#[allow(clippy::excessive_precision)]
pub fn sin_cos(x: f32) -> (f32, f32) {
    const PIO2_1: f32 = 1.570_312_5;
    const PIO2_2: f32 = 4.837_512_969_970_703_125e-4;
    const PIO2_3: f32 = 7.549_789_954_891_882_16e-8;
    let (q, qi) = round_i(x * core::f32::consts::FRAC_2_PI);
    let r = ((x - q * PIO2_1) - q * PIO2_2) - q * PIO2_3;
    let z = r * r;
    let s = ((-1.951_529_589_1e-4 * z + 8.332_160_873_6e-3) * z - 1.666_665_461_1e-1) * z * r + r;
    let c = ((2.443_315_711_809_948e-5 * z - 1.388_731_625_493_765e-3) * z
        + 4.166_664_568_298_827e-2)
        * z
        * z
        - 0.5 * z
        + 1.0;
    let swap = qi & 1 != 0;
    let (s0, c0) = (select(swap, c, s), select(swap, s, c));
    let sin = f32::from_bits(s0.to_bits() ^ ((qi & 2).cast_unsigned() << 30));
    let cos = f32::from_bits(c0.to_bits() ^ (((qi + 1) & 2).cast_unsigned() << 30));
    // sin(±0) = ±0.
    (select(x == 0.0, x, sin), cos)
}

/// `e^x`, with overflow to infinity and gradual underflow.
#[inline(always)]
#[allow(clippy::excessive_precision)]
pub fn exp(x: f32) -> f32 {
    let xc = x.clamp(-104.0, 89.0);
    let (n, ni) = round_i(xc * core::f32::consts::LOG2_E);
    let r = xc - n * 0.693_359_375 - n * -2.121_944_400_546_905_827_68e-4;
    let z = r * r;
    let p = ((((1.987_569_150_0e-4 * r + 1.398_199_950_7e-3) * r + 8.333_451_907_3e-3) * r
        + 4.166_579_589_4e-2)
        * r
        + 1.666_666_545_9e-1)
        * r
        + 5.000_000_120_1e-1;
    let y = p * z + r + 1.0;
    // Scale in two steps so that 2^n itself never overflows or underflows.
    let n1 = ni >> 1;
    let e = y * pow2i(n1) * pow2i(ni - n1);
    select(x.is_nan(), x, e)
}

/// Natural logarithm: `-inf` at zero, NaN below zero, and subnormal inputs handled.
#[inline(always)]
#[allow(clippy::excessive_precision)]
pub fn ln(x: f32) -> f32 {
    let sub = x < f32::MIN_POSITIVE;
    let xs = select(sub, x * 8_388_608.0, x);
    let bits = xs.to_bits().cast_signed();
    let e = ((bits >> 23) & 0xff) - 126 - select(sub, 23.0, 0.0) as i32;
    let m = f32::from_bits(((bits & 0x007f_ffff) | 0x3f00_0000).cast_unsigned());
    let small = m < core::f32::consts::FRAC_1_SQRT_2;
    let e = (e - i32::from(small)) as f32;
    let m = select(small, m + m - 1.0, m - 1.0);
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
    let y = p * m * z - 2.121_944_400_546_905_827_68e-4 * e - 0.5 * z;
    let r = m + y + 0.693_359_375 * e;
    let r = select(x == f32::INFINITY, x, r);
    let r = select(x == 0.0, f32::NEG_INFINITY, r);
    select(x < 0.0 || x.is_nan(), f32::NAN, r)
}

/// Four-quadrant arctangent of `y / x`, with the signed-zero conventions of `f32::atan2`.
#[inline(always)]
#[allow(clippy::excessive_precision)]
pub fn atan2(y: f32, x: f32) -> f32 {
    let (ax, ay) = (x.abs(), y.abs());
    let (mx, mn) = (select(ax < ay, ay, ax), select(ax < ay, ax, ay));
    // t = min / max in [0, 1]; 0/0 is 0 and inf/inf is 1.
    let t = select(mx == 0.0, 0.0, select(mn == f32::INFINITY, 1.0, mn / mx));
    let big = t > 0.414_213_562_373_095_05;
    let u = select(big, (t - 1.0) / (t + 1.0), t);
    let z = u * u;
    let a = (((8.053_744_495_38e-2 * z - 1.387_768_560_32e-1) * z + 1.997_771_064_78e-1) * z
        - 3.333_294_915_39e-1)
        * z
        * u
        + u;
    let a = select(big, a + FRAC_PI_4, a);
    let a = select(ay > ax, FRAC_PI_2 - a, a);
    let a = select(x.is_sign_negative(), PI - a, a);
    let a = f32::from_bits(a.to_bits() | (y.to_bits() & 0x8000_0000));
    select(x.is_nan() || y.is_nan(), f32::NAN, a)
}

/// Hyperbolic sine.
#[inline(always)]
#[allow(clippy::excessive_precision)]
pub fn sinh(x: f32) -> f32 {
    let a = x.abs();
    let z = x * x;
    let small =
        ((2.037_219_129_45e-4 * z + 8.330_283_762_39e-3) * z + 1.666_671_602_11e-1) * z * x + x;
    let e = exp(a);
    let big = f32::from_bits((0.5 * (e - 1.0 / e)).to_bits() | (x.to_bits() & 0x8000_0000));
    select(a < 1.0, small, big)
}

/// Hyperbolic cosine.
#[inline(always)]
#[allow(clippy::manual_midpoint)] // the same rounding as the vector version
pub fn cosh(x: f32) -> f32 {
    let e = exp(x.abs());
    0.5 * (e + 1.0 / e)
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    extern crate std;
    use super::*;

    fn ulps(a: f32, b: f32) -> u32 {
        if a == b || (a.is_nan() && b.is_nan()) {
            return 0;
        }
        (a.to_bits().cast_signed() - b.to_bits().cast_signed()).unsigned_abs()
    }

    fn close(got: f32, want: f32, ulp: u32, abs: f32, what: &str, x: f32) {
        assert!(
            ulps(got, want) <= ulp || (got - want).abs() <= abs,
            "{what}({x}): {got} vs {want} ({} ulp)",
            ulps(got, want)
        );
    }

    fn sweep(lo: f32, hi: f32, n: u32) -> impl Iterator<Item = f32> {
        (0..=n).map(move |i| lo + (hi - lo) * (i as f32 / n as f32))
    }

    #[test]
    #[cfg_attr(
        miri,
        ignore = "miri perturbs std's float functions by a few ulp on purpose"
    )]
    fn sin_cos_match_std() {
        for x in sweep(-100.0, 100.0, 200_000) {
            let (s, c) = sin_cos(x);
            close(s, x.sin(), 2, 1e-7, "sin", x);
            close(c, x.cos(), 2, 1e-7, "cos", x);
        }
        assert_eq!(sin_cos(0.0).0.to_bits(), 0.0f32.to_bits());
        assert_eq!(sin_cos(-0.0).0.to_bits(), (-0.0f32).to_bits());
    }

    #[test]
    #[cfg_attr(
        miri,
        ignore = "miri perturbs std's float functions by a few ulp on purpose"
    )]
    fn exp_ln_match_std() {
        for x in sweep(-103.0, 88.7, 200_000) {
            close(exp(x), x.exp(), 2, 0.0, "exp", x);
        }
        assert_eq!(exp(100.0), f32::INFINITY);
        assert_eq!(exp(-200.0), 0.0);
        assert!(exp(f32::NAN).is_nan());
        for x in sweep(-30.0, 30.0, 100_000) {
            let x = x.exp2();
            close(ln(x), x.ln(), 2, 1e-7, "ln", x);
        }
        for x in [1e-40f32, 1e-45, f32::MIN_POSITIVE, 1.0, f32::MAX] {
            close(ln(x), x.ln(), 2, 0.0, "ln", x);
        }
        assert_eq!(ln(0.0), f32::NEG_INFINITY);
        assert_eq!(ln(f32::INFINITY), f32::INFINITY);
        assert!(ln(-1.0).is_nan());
    }

    #[test]
    #[cfg_attr(
        miri,
        ignore = "miri perturbs std's float functions by a few ulp on purpose"
    )]
    fn atan2_matches_std() {
        for y in sweep(-3.0, 3.0, 300) {
            for x in sweep(-3.0, 3.0, 301) {
                close(atan2(y, x), y.atan2(x), 2, 1e-7, "atan2", y);
            }
        }
        for (y, x) in [
            (0.0f32, 0.0f32),
            (0.0, -0.0),
            (-0.0, -0.0),
            (-0.0, 1.0),
            (1.0, 0.0),
            (-1.0, -0.0),
        ] {
            // The sign conventions exactly; the value within an ulp (Apple's libm returns the
            // float just below π for atan2(0, -0), glibc the nearest one).
            let (got, want) = (atan2(y, x), y.atan2(x));
            assert!(
                got.is_sign_negative() == want.is_sign_negative() && ulps(got, want) <= 1,
                "atan2({y}, {x}): {got} vs {want}"
            );
        }
        close(
            atan2(f32::INFINITY, f32::INFINITY),
            f32::INFINITY.atan2(f32::INFINITY),
            1,
            0.0,
            "atan2(inf, inf)",
            f32::INFINITY,
        );
    }

    #[test]
    #[cfg_attr(
        miri,
        ignore = "miri perturbs std's float functions by a few ulp on purpose"
    )]
    fn sinh_cosh_match_std() {
        for x in sweep(-80.0, 80.0, 100_000) {
            close(sinh(x), x.sinh(), 3, 1e-7, "sinh", x);
            close(cosh(x), x.cosh(), 3, 0.0, "cosh", x);
        }
    }
}
