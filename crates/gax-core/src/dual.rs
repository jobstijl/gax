//! Forward-mode automatic differentiation: dual numbers as a coefficient type.
//!
//! [`Dual<T, N>`] is a value with its derivatives along `N` directions,
//! `re + Σᵢ duᵢ εᵢ` with `εᵢ εⱼ = 0`. It implements [`Coef`] and [`Real`], so every generated
//! product, sandwich, map, solver, `exp` and `log` runs on it unchanged and carries derivatives
//! through: the derivative of a whole computation, exact to rounding, in one pass.
//!
//! * [`derivative`]: `f'(x)` of a function of one variable.
//! * [`gradient`]: `∇f` of a scalar function of `N` variables, in one pass.
//! * [`jacobian`]: the `M × N` Jacobian of a function with `M` outputs.
//!
//! `T` can be any [`Real`], SIMD lanes included (`Dual<f64x4, N>` differentiates four problems
//! at once). Branch-free code differentiates the branch it selects: [`Real::select_lt`] chooses
//! on the values and passes the chosen derivatives through, so piecewise definitions are
//! differentiated piece by piece (one-sided at a switch point).
//!
//! ```
//! use gax::dual::{Dual, gradient};
//! use gax::pga3d::{Motor, Point};
//!
//! // How the squared distance from a moved point to a target changes with the motion's
//! // parameters: a rotation by `a` about the z axis, then a translation by `tx` along x.
//! type D = Dual<f64, 2>;
//! let c = D::constant;
//! let (d2, grad) = gradient(
//!     |[a, tx]: [D; 2]| {
//!         let m = Motor::translation(tx, c(0.0), c(0.0))
//!             * Motor::rotation_about(c(0.0), c(0.0), c(1.0), a);
//!         let [x, y, z] = (m >> Point::xyz(c(1.0), c(0.0), c(0.0))).to_euclidean();
//!         let (dx, dy) = (x - c(1.0), y - c(1.0));
//!         dx * dx + dy * dy + z * z
//!     },
//!     [0.0, 0.0],
//! );
//! assert!((d2 - 1.0).abs() < 1e-12);
//! // d/da [(cos a + tx − 1)² + (sin a − 1)²] = −2 and d/dtx = 0 at the origin.
//! assert!((grad[0] + 2.0).abs() < 1e-12 && grad[1].abs() < 1e-12);
//! ```

use crate::coef::{Coef, Real};
use core::ops::{Add, Div, Mul, Neg, Sub};

/// A value `re` with its derivatives `du` along `N` directions: `re + Σᵢ du[i] εᵢ`,
/// `εᵢ εⱼ = 0`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dual<T, const N: usize = 1> {
    /// The value.
    pub re: T,
    /// The derivatives along each direction.
    pub du: [T; N],
}

impl<T: Coef, const N: usize> Dual<T, N> {
    /// A value with the given derivatives.
    #[inline(always)]
    pub const fn new(re: T, du: [T; N]) -> Self {
        Dual { re, du }
    }

    /// A constant: all derivatives zero.
    #[inline(always)]
    pub fn constant(re: T) -> Self {
        Dual {
            re,
            du: [T::zero(); N],
        }
    }

    /// The `i`-th of `N` variables at `re`: derivative 1 along direction `i`, 0 along the
    /// others.
    #[inline(always)]
    pub fn variable(re: T, i: usize) -> Self {
        Dual {
            re,
            du: core::array::from_fn(|k| if k == i { T::one() } else { T::zero() }),
        }
    }
}

impl<T: Real, const N: usize> Dual<T, N> {
    /// `self` with the derivatives scaled by `f'` at the value: the chain rule for a function
    /// with value `f` and derivative `df` at `re`. A derivative that is exactly zero stays zero
    /// even where `df` is infinite: `√x` at `x = 0` inside `cos √x`, as the closed forms
    /// compute a rotation angle, has an infinite derivative but a zero tangent there, and the
    /// composition is smooth (plain forward mode would give `0 · ∞ = NaN`).
    #[inline(always)]
    fn chain(self, f: T, df: T) -> Self {
        Dual {
            re: f,
            du: self
                .du
                .map(|d| T::select_lt(T::zero(), d.abs(), d * df, T::zero())),
        }
    }
}

impl<T: Coef, const N: usize> Add for Dual<T, N> {
    type Output = Self;
    #[inline(always)]
    fn add(self, o: Self) -> Self {
        Dual {
            re: self.re + o.re,
            du: core::array::from_fn(|i| self.du[i] + o.du[i]),
        }
    }
}

impl<T: Coef, const N: usize> Sub for Dual<T, N> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, o: Self) -> Self {
        Dual {
            re: self.re - o.re,
            du: core::array::from_fn(|i| self.du[i] - o.du[i]),
        }
    }
}

impl<T: Coef, const N: usize> Mul for Dual<T, N> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, o: Self) -> Self {
        Dual {
            re: self.re * o.re,
            du: core::array::from_fn(|i| self.du[i].mul_add(o.re, self.re * o.du[i])),
        }
    }
}

impl<T: Coef, const N: usize> Neg for Dual<T, N> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Dual {
            re: -self.re,
            du: self.du.map(|d| -d),
        }
    }
}

// The quotient rule needs `*` and `-` inside `Div`.
#[allow(clippy::suspicious_arithmetic_impl)]
impl<T: Real, const N: usize> Div for Dual<T, N> {
    type Output = Self;
    #[inline(always)]
    fn div(self, o: Self) -> Self {
        let inv = o.re.recip();
        let q = self.re * inv;
        Dual {
            re: q,
            // (a/b)' = (a' − q b') / b
            du: core::array::from_fn(|i| (self.du[i] - q * o.du[i]) * inv),
        }
    }
}

impl<T: Coef, const N: usize> Coef for Dual<T, N> {
    #[inline(always)]
    fn zero() -> Self {
        Self::constant(T::zero())
    }
    #[inline(always)]
    fn one() -> Self {
        Self::constant(T::one())
    }
    #[inline(always)]
    fn from_i64(i: i64) -> Self {
        Self::constant(T::from_i64(i))
    }
    #[inline(always)]
    fn from_f64(f: f64) -> Self {
        Self::constant(T::from_f64(f))
    }
    #[inline(always)]
    fn from_ratio(num: i64, den: i64) -> Self {
        Self::constant(T::from_ratio(num, den))
    }
    /// The values' check: derivatives say nothing about whether a versor is unit.
    #[inline(always)]
    fn check_unit(deviations: &[Self]) {
        for d in deviations {
            T::check_unit(&[d.re]);
        }
    }
    #[inline(always)]
    fn note_renormalize() {
        T::note_renormalize();
    }
    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        Dual {
            re: self.re.mul_add(a.re, b.re),
            du: core::array::from_fn(|i| {
                self.du[i].mul_add(a.re, self.re.mul_add(a.du[i], b.du[i]))
            }),
        }
    }
    #[inline(always)]
    fn vectorize<R>(f: impl FnOnce() -> R) -> R {
        T::vectorize(f)
    }
}

impl<T: Real, const N: usize> Real for Dual<T, N> {
    const SCALAR: bool = T::SCALAR;

    #[inline(always)]
    fn sqrt(self) -> Self {
        let r = self.re.sqrt();
        self.chain(r, (r + r).recip())
    }
    #[inline(always)]
    fn recip(self) -> Self {
        let r = self.re.recip();
        self.chain(r, -(r * r))
    }
    #[inline(always)]
    fn abs(self) -> Self {
        let sign = T::select_lt(self.re, T::zero(), -T::one(), T::one());
        self.chain(self.re.abs(), sign)
    }
    #[inline(always)]
    fn sin(self) -> Self {
        let (s, c) = self.re.sin_cos();
        self.chain(s, c)
    }
    #[inline(always)]
    fn cos(self) -> Self {
        let (s, c) = self.re.sin_cos();
        self.chain(c, -s)
    }
    #[inline(always)]
    fn sin_cos(self) -> (Self, Self) {
        let (s, c) = self.re.sin_cos();
        (self.chain(s, c), self.chain(c, -s))
    }
    #[inline(always)]
    fn sinh(self) -> Self {
        self.chain(self.re.sinh(), self.re.cosh())
    }
    #[inline(always)]
    fn cosh(self) -> Self {
        self.chain(self.re.cosh(), self.re.sinh())
    }
    /// `atan2(y, x)`, with derivative `(x dy − y dx) / (x² + y²)`.
    #[inline(always)]
    fn atan2(self, x: Self) -> Self {
        let inv = (x.re * x.re + self.re * self.re).recip();
        Dual {
            re: self.re.atan2(x.re),
            du: core::array::from_fn(|i| (x.re * self.du[i] - self.re * x.du[i]) * inv),
        }
    }
    #[inline(always)]
    fn ln(self) -> Self {
        self.chain(self.re.ln(), self.re.recip())
    }
    #[inline(always)]
    fn exp(self) -> Self {
        let e = self.re.exp();
        self.chain(e, e)
    }
    /// Chooses on the values; the chosen operand's derivatives come along.
    #[inline(always)]
    fn select_lt(a: Self, b: Self, x: Self, y: Self) -> Self {
        Dual {
            re: T::select_lt(a.re, b.re, x.re, y.re),
            du: core::array::from_fn(|i| T::select_lt(a.re, b.re, x.du[i], y.du[i])),
        }
    }
    #[inline(always)]
    fn all_lt(a: Self, b: Self) -> bool {
        T::all_lt(a.re, b.re)
    }
    #[inline(always)]
    fn epsilon() -> Self {
        Self::constant(T::epsilon())
    }
}

/// `(f(x), f'(x))`.
///
/// ```
/// use gax::dual::derivative;
/// use gax::Real;
/// let (v, d) = derivative(|t| (t * t).sin(), 0.5f64);
/// assert!((v - 0.25f64.sin()).abs() < 1e-15 && (d - 0.25f64.cos()).abs() < 1e-15);
/// ```
#[inline]
pub fn derivative<T: Real>(f: impl FnOnce(Dual<T, 1>) -> Dual<T, 1>, x: T) -> (T, T) {
    let y = f(Dual::variable(x, 0));
    (y.re, y.du[0])
}

/// `(f(x), ∇f(x))` for a scalar function of `N` variables, in one evaluation.
#[inline]
pub fn gradient<T: Real, const N: usize>(
    f: impl FnOnce([Dual<T, N>; N]) -> Dual<T, N>,
    x: [T; N],
) -> (T, [T; N]) {
    let y = f(core::array::from_fn(|i| Dual::variable(x[i], i)));
    (y.re, y.du)
}

/// `(f(x), J)` for a function of `N` variables with `M` outputs: `J[m][n] = ∂fₘ/∂xₙ`, in one
/// evaluation.
#[inline]
#[allow(clippy::type_complexity)]
pub fn jacobian<T: Real, const N: usize, const M: usize>(
    f: impl FnOnce([Dual<T, N>; N]) -> [Dual<T, N>; M],
    x: [T; N],
) -> ([T; M], [[T; N]; M]) {
    let y = f(core::array::from_fn(|i| Dual::variable(x[i], i)));
    (y.map(|v| v.re), y.map(|v| v.du))
}
