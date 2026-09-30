//! A prime field, `ℤ/p` with the Mersenne prime `p = 2⁶¹ − 1`: exact coefficients for testing
//! identities of generated code at random points.
//!
//! Every generated kernel is generic over [`Coef`], so it runs unchanged on [`Fp`]. An identity
//! between two such computations is a polynomial identity in their inputs; checked at a point
//! drawn uniformly from `ℤ/p`, a nonzero polynomial of degree `d` vanishes with probability at
//! most `d/p` (the Schwartz–Zippel lemma). With `p ≈ 2.3·10¹⁸`, each such sample of a law of
//! degree 64 misses a failure with probability below `2⁻⁵⁵`: a randomized proof, for laws whose
//! symbolic proof is out of reach (docs/laws.md, law L).
//!
//! ```
//! use gax_core::fp::Fp;
//! use gax_core::Coef;
//! let (a, b) = (Fp::new(3), Fp::from_ratio(1, 3));
//! assert_eq!(a * b, Fp::one());
//! assert_eq!(Fp::from_f64(0.75) * Fp::from_i64(4), Fp::from_i64(3));
//! ```

use crate::coef::Coef;
use core::ops::{Add, Div, Mul, Neg, Sub};

/// The Mersenne prime `2⁶¹ − 1`.
pub const P: u64 = (1 << 61) - 1;

/// An element of `ℤ/p`, `p = 2⁶¹ − 1`, kept reduced to `0..p`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct Fp(u64);

/// `x mod p` for `x < 2¹²⁸` whose high part is below `2⁶⁴` (a product of two reduced values).
#[inline(always)]
const fn reduce(x: u128) -> u64 {
    // x = hi 2⁶¹ + lo and 2⁶¹ ≡ 1: x ≡ hi + lo.
    let r = (x as u64 & P) + (x >> 61) as u64;
    let r = (r & P) + (r >> 61);
    if r >= P { r - P } else { r }
}

impl Fp {
    /// `x mod p`.
    #[inline]
    pub const fn new(x: u64) -> Fp {
        Fp(reduce(x as u128))
    }

    /// The representative in `0..p`.
    #[inline]
    pub const fn get(self) -> u64 {
        self.0
    }

    /// `self` to the power `e`.
    pub fn pow(self, mut e: u64) -> Fp {
        let (mut base, mut acc) = (self, Fp(1));
        while e > 0 {
            if e & 1 == 1 {
                acc = acc * base;
            }
            base = base * base;
            e >>= 1;
        }
        acc
    }

    /// The inverse (Fermat: `self^(p−2)`). Zero has none.
    ///
    /// # Panics
    /// For zero.
    pub fn inv(self) -> Fp {
        assert!(self.0 != 0, "zero has no inverse in ℤ/p");
        self.pow(P - 2)
    }
}

impl Add for Fp {
    type Output = Fp;
    #[inline]
    fn add(self, o: Fp) -> Fp {
        let s = self.0 + o.0;
        Fp(if s >= P { s - P } else { s })
    }
}

impl Sub for Fp {
    type Output = Fp;
    #[inline]
    fn sub(self, o: Fp) -> Fp {
        Fp(if self.0 >= o.0 {
            self.0 - o.0
        } else {
            self.0 + P - o.0
        })
    }
}

impl Neg for Fp {
    type Output = Fp;
    #[inline]
    fn neg(self) -> Fp {
        Fp(if self.0 == 0 { 0 } else { P - self.0 })
    }
}

impl Mul for Fp {
    type Output = Fp;
    #[inline]
    fn mul(self, o: Fp) -> Fp {
        Fp(reduce(u128::from(self.0) * u128::from(o.0)))
    }
}

impl Div for Fp {
    type Output = Fp;
    /// Multiplication by the inverse.
    #[inline]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, o: Fp) -> Fp {
        self * o.inv()
    }
}

impl Coef for Fp {
    #[inline]
    fn zero() -> Fp {
        Fp(0)
    }
    #[inline]
    fn one() -> Fp {
        Fp(1)
    }
    #[inline]
    fn from_i64(i: i64) -> Fp {
        let m = Fp::new(i.unsigned_abs());
        if i < 0 { -m } else { m }
    }
    /// A finite `f64` exactly: it is `m · 2^e` for an integer `m`.
    ///
    /// # Panics
    /// For an infinite or NaN `f`.
    fn from_f64(f: f64) -> Fp {
        assert!(f.is_finite(), "{f} is not a number of ℤ/p");
        if f == 0.0 {
            return Fp(0);
        }
        let bits = f.to_bits();
        let exponent = ((bits >> 52) & 0x7ff) as i64;
        let fraction = bits & ((1 << 52) - 1);
        // Normal: (2⁵² + fraction) 2^(exponent − 1075); subnormal: fraction 2^(−1074).
        let (mantissa, e) = if exponent == 0 {
            (fraction, -1074)
        } else {
            (fraction | (1 << 52), exponent - 1075)
        };
        let two = Fp(2);
        let scale = if e >= 0 {
            two.pow(e as u64)
        } else {
            two.pow(e.unsigned_abs()).inv()
        };
        let m = Fp::new(mantissa) * scale;
        if f < 0.0 { -m } else { m }
    }
    #[inline]
    fn from_ratio(num: i64, den: i64) -> Fp {
        Fp::from_i64(num) * Fp::from_i64(den).inv()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn field_arithmetic() {
        let a = Fp::new(P - 5);
        let b = Fp::new(1_234_567_890_123);
        assert_eq!(a + Fp::new(5), Fp::zero());
        assert_eq!((a * b) / b, a);
        assert_eq!(a - b + b, a);
        assert_eq!(-(-a), a);
        assert_eq!(b.inv() * b, Fp::one());
        assert_eq!(Fp::new(u64::MAX), Fp::new(u64::MAX % P));
        assert_eq!(Fp::from_i64(-3) + Fp::from_i64(3), Fp::zero());
        assert_eq!(Fp::from_ratio(-2, 6) * Fp::from_i64(3), Fp::from_i64(-1));
        // f64 constants exactly, including fractions and large and tiny exponents.
        assert_eq!(Fp::from_f64(0.5) + Fp::from_f64(0.5), Fp::one());
        assert_eq!(Fp::from_f64(-1.25) * Fp::from_i64(4), Fp::from_i64(-5));
        // Powers of two from their bits (Miri perturbs `powi`).
        let two_to = |e: i64| f64::from_bits(((1023 + e) as u64) << 52);
        assert_eq!(Fp::from_f64(two_to(70)), Fp(2).pow(70));
        assert_eq!(Fp::from_f64(two_to(-60)) * Fp(2).pow(60), Fp::one());
    }
}
