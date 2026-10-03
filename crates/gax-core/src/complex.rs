//! Complex numbers as coefficients: `Complex<f64>` is a [`Coef`], so multivectors, maps and
//! forms with complex coefficients use the generated products unchanged (the complex
//! transversals of four lines, the complex eigenvectors of a map). It is not [`Real`]: there is
//! no ordering, so methods that need one (norms, exponentials, solvers) stay with real
//! coefficients; [`Complex`] has its own `sqrt`, `exp`, `ln` and division.

use crate::coef::{Coef, Real};
use core::ops::{Add, Div, Mul, Neg, Sub};

/// A complex number `re + i im`.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Complex<T> {
    /// The real part.
    pub re: T,
    /// The imaginary part.
    pub im: T,
}

impl<T: Real> Complex<T> {
    /// `re + i im`.
    #[inline(always)]
    pub fn new(re: T, im: T) -> Self {
        Complex { re, im }
    }

    /// A real number.
    #[inline(always)]
    pub fn real(re: T) -> Self {
        Complex { re, im: T::zero() }
    }

    /// The imaginary unit.
    #[inline(always)]
    pub fn i() -> Self {
        Complex {
            re: T::zero(),
            im: T::one(),
        }
    }

    /// `r (cos θ + i sin θ)`.
    #[inline]
    pub fn from_polar(r: T, theta: T) -> Self {
        let (s, c) = theta.sin_cos();
        Complex {
            re: r * c,
            im: r * s,
        }
    }

    /// The conjugate `re − i im`.
    #[inline(always)]
    pub fn conj(self) -> Self {
        Complex {
            re: self.re,
            im: -self.im,
        }
    }

    /// `|z|²`.
    #[inline(always)]
    pub fn norm_sqr(self) -> T {
        self.re * self.re + self.im * self.im
    }

    /// `|z|`, without overflow for large parts.
    #[inline]
    pub fn abs(self) -> T {
        let (a, b) = (self.re.abs(), self.im.abs());
        let big = a.max(b);
        let small = a.min(b);
        // Zero stays zero (in any precision: a guard such as 1e-300 is zero in f32).
        let zero = T::zero();
        let r = small / T::select_lt(zero, big, big, T::one());
        T::select_lt(zero, big, big * (T::one() + r * r).sqrt(), zero)
    }

    /// The argument, in `(−π, π]`.
    #[inline]
    pub fn arg(self) -> T {
        self.im.atan2(self.re)
    }

    /// `z` times a real `k`.
    #[inline(always)]
    pub fn scale(self, k: T) -> Self {
        Complex {
            re: self.re * k,
            im: self.im * k,
        }
    }

    /// `1 / z`.
    #[inline]
    pub fn recip(self) -> Self {
        self.conj().scale(self.norm_sqr().recip())
    }

    /// The principal square root (the one with `re ≥ 0`), without cancellation.
    #[inline]
    pub fn sqrt(self) -> Self {
        let half = T::from_f64(0.5);
        let r = self.abs();
        // The larger of the two parts first, the other from `im = 2 re' im'`.
        let a = ((r + self.re.abs()) * half).sqrt();
        let zero = T::zero();
        let b = self.im * half / T::select_lt(zero, a, a, T::one());
        let pos = Complex { re: a, im: b };
        // re < 0: the root is `(|b|, sign(im) a)`.
        let sign = T::select_lt(self.im, T::zero(), -T::one(), T::one());
        let neg = Complex {
            re: b.abs(),
            im: sign * a,
        };
        let root = Complex {
            re: T::select_lt(self.re, zero, neg.re, pos.re),
            im: T::select_lt(self.re, zero, neg.im, pos.im),
        };
        Complex {
            re: T::select_lt(zero, a, root.re, zero),
            im: T::select_lt(zero, a, root.im, zero),
        }
    }

    /// `e^z`.
    #[inline]
    pub fn exp(self) -> Self {
        Self::from_polar(self.re.exp(), self.im)
    }

    /// The principal logarithm.
    #[inline]
    pub fn ln(self) -> Self {
        Complex {
            re: self.abs().ln(),
            im: self.arg(),
        }
    }
}

impl<T: Real> Add for Complex<T> {
    type Output = Self;
    #[inline(always)]
    fn add(self, o: Self) -> Self {
        Complex {
            re: self.re + o.re,
            im: self.im + o.im,
        }
    }
}

impl<T: Real> Sub for Complex<T> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, o: Self) -> Self {
        Complex {
            re: self.re - o.re,
            im: self.im - o.im,
        }
    }
}

impl<T: Real> Mul for Complex<T> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, o: Self) -> Self {
        Complex {
            re: self.re * o.re - self.im * o.im,
            im: self.re * o.im + self.im * o.re,
        }
    }
}

impl<T: Real> Div for Complex<T> {
    type Output = Self;
    #[inline]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, o: Self) -> Self {
        self * o.recip()
    }
}

impl<T: Real> Neg for Complex<T> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Complex {
            re: -self.re,
            im: -self.im,
        }
    }
}

impl<T: Real> Coef for Complex<T> {
    #[inline(always)]
    fn zero() -> Self {
        Complex::real(T::zero())
    }
    #[inline(always)]
    fn one() -> Self {
        Complex::real(T::one())
    }
    #[inline(always)]
    fn from_i64(i: i64) -> Self {
        Complex::real(T::from_i64(i))
    }
    #[inline(always)]
    fn from_f64(f: f64) -> Self {
        Complex::real(T::from_f64(f))
    }
}

#[cfg(test)]
mod tests {
    use super::Complex;

    fn close(a: Complex<f64>, b: Complex<f64>) -> bool {
        (a - b).abs() < 1e-12
    }

    #[test]
    fn arithmetic_and_roots() {
        let z = Complex::new(3.0, -4.0);
        assert_eq!(z.abs(), 5.0);
        assert!(close(z * z.recip(), Complex::real(1.0)));
        for w in [
            z,
            Complex::new(-3.0, 4.0),
            Complex::new(-2.0, 0.0),
            Complex::new(0.0, -1e-3),
        ] {
            let r = w.sqrt();
            assert!(close(r * r, w), "{w:?}: {r:?}");
            assert!(r.re >= 0.0);
        }
        assert!(close(
            Complex::new(0.0, core::f64::consts::PI).exp(),
            Complex::real(-1.0)
        ));
        assert!(close(z.ln().exp(), z));
    }

    #[test]
    fn zero_in_f32() {
        // Guards must hold in f32 too, where 1e-300 underflows to zero.
        let z = Complex::<f32>::new(0.0, 0.0);
        assert_eq!(z.abs(), 0.0);
        assert_eq!(z.sqrt(), z);
        assert_eq!(Complex::<f32>::new(0.0, -2.0).abs(), 2.0);
    }
}
