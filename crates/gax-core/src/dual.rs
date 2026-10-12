//! Forward-mode automatic differentiation: dual numbers as a coefficient type.
//!
//! [`Jet<T, D>`] is a value with its first derivatives along a set of directions `D`,
//! `re + Σᵢ duᵢ εᵢ` with `εᵢ εⱼ = 0`. It implements [`Coef`] and [`Real`], so every generated
//! product, sandwich, map, solver, `exp` and `log` runs on it unchanged and carries derivatives
//! through: the derivative of a whole computation, exact to rounding, in one pass.
//!
//! The directions are either a kind or a count:
//!
//! * `Jet<T, A>` for a kind `A` holds one derivative per coefficient of `A`, stored as `A`'s own
//!   coefficient array. Derivatives with respect to a value of `A` come out as extensors:
//!   * [`jacobian`]: of `f: A -> B` (a value or a map), the map `B <- (A,)`, with `A`'s slot
//!     after `B`'s own (numga's `derivative`);
//!   * [`gradient`]: of a scalar function of `A`, the linear form `Scalar <- (A,)`;
//!   * [`hessian`]: its second derivative, the form `Scalar <- (A, A)`, so a Newton step is
//!     `-hessian.solve(gradient)`.
//! * [`Dual<T, N>`] is `Jet<T, Params<N>>`, `N` loose parameters:
//!   * [`derivative`]: `f'(x)` of a function of one variable;
//!   * [`gradient_array`]: `∇f` of a scalar function of `N` variables;
//!   * [`jacobian_array`]: the `M × N` Jacobian of a function with `M` outputs.
//!
//! `T` can be any [`Real`], SIMD lanes included (`Dual<f64x4, N>` differentiates four problems
//! at once). Branch-free code differentiates the branch it selects: [`Real::select_lt`] chooses
//! on the values and passes the chosen derivatives through, so piecewise definitions are
//! differentiated piece by piece (one-sided at a switch point).
//!
//! ```
//! use gax::dual::gradient;
//! use gax::pga3d::{Line, Point};
//! use gax::Real;
//!
//! // How the squared distance from a moved point to a target changes with the twist that
//! // moves it: the gradient is a linear form on twists.
//! fn misfit<T: Real>(twist: Line<(), T>) -> T {
//!     let p = Point::xyz(1.0, 0.0, 0.0).map_coefs(T::from_f64);
//!     let target = Point::xyz(1.0, 1.0, 0.0).map_coefs(T::from_f64);
//!     let r = twist.exp() >> p;
//!     let d: [T; 3] = core::array::from_fn(|i| r.to_euclidean()[i] - target.to_euclidean()[i]);
//!     d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
//! }
//! let (value, slope) = gradient(misfit, Line::<(), f64>::zero());
//! assert!((value - 1.0).abs() < 1e-12);
//! // A small twist `t` changes the misfit by `slope.of(t)` to first order.
//! let t = Line::new(0.0, 0.0, 1e-6, 0.0, 0.0, 0.0);
//! assert!((misfit(t) - value - slope.of(t).s()).abs() < 1e-11);
//! ```

use crate::coef::{Coef, Elem, Real};
use crate::kind::{Extensor, Kind, Retype};
use crate::slots::{Cat, Slots};
use core::marker::PhantomData;
use core::ops::{Add, Div, Mul, Neg, Sub};

/// A set of directions to differentiate along, with the array that holds one derivative per
/// direction: every [`Kind`] (its coefficient array) and [`Params<N>`] (`[X; N]`).
pub trait Directions: Copy + core::fmt::Debug + PartialEq + 'static {
    /// The number of directions.
    const N: usize;
    /// One `X` per direction.
    type Arr<X: Elem>: Elem + AsRef<[X]> + AsMut<[X]>;
    /// Build an array from its index.
    fn from_fn<X: Elem>(f: impl FnMut(usize) -> X) -> Self::Arr<X>;
    /// Elementwise map.
    fn map<X: Elem, Y: Elem>(a: &Self::Arr<X>, f: impl FnMut(&X) -> Y) -> Self::Arr<Y>;
    /// Elementwise combination.
    fn zip<X: Elem, Y: Elem, Z: Elem>(
        a: &Self::Arr<X>,
        b: &Self::Arr<Y>,
        f: impl FnMut(&X, &Y) -> Z,
    ) -> Self::Arr<Z>;
}

impl<A: Kind> Directions for A {
    const N: usize = A::N;
    type Arr<X: Elem> = A::Arr<X>;
    #[inline(always)]
    fn from_fn<X: Elem>(f: impl FnMut(usize) -> X) -> Self::Arr<X> {
        A::arr_from_fn(f)
    }
    #[inline(always)]
    fn map<X: Elem, Y: Elem>(a: &Self::Arr<X>, f: impl FnMut(&X) -> Y) -> Self::Arr<Y> {
        A::arr_map(a, f)
    }
    #[inline(always)]
    fn zip<X: Elem, Y: Elem, Z: Elem>(
        a: &Self::Arr<X>,
        b: &Self::Arr<Y>,
        f: impl FnMut(&X, &Y) -> Z,
    ) -> Self::Arr<Z> {
        A::arr_zip(a, b, f)
    }
}

/// `N` loose parameters as directions: the derivatives are a `[X; N]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Params<const N: usize>(PhantomData<[(); N]>);

impl<const N: usize> Directions for Params<N> {
    const N: usize = N;
    type Arr<X: Elem> = [X; N];
    #[inline(always)]
    fn from_fn<X: Elem>(f: impl FnMut(usize) -> X) -> [X; N] {
        core::array::from_fn(f)
    }
    #[inline(always)]
    fn map<X: Elem, Y: Elem>(a: &[X; N], mut f: impl FnMut(&X) -> Y) -> [Y; N] {
        a.map(|x| f(&x))
    }
    #[inline(always)]
    fn zip<X: Elem, Y: Elem, Z: Elem>(
        a: &[X; N],
        b: &[Y; N],
        mut f: impl FnMut(&X, &Y) -> Z,
    ) -> [Z; N] {
        core::array::from_fn(|i| f(&a[i], &b[i]))
    }
}

/// A value `re` with its derivatives `du` along the directions `D`: `re + Σᵢ du[i] εᵢ`,
/// `εᵢ εⱼ = 0`. For a kind `D`, `du` is laid out as a value of `D`: `du[i]` is the derivative
/// with respect to `D`'s coefficient `i`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Jet<T: Elem, D: Directions> {
    /// The value.
    pub re: T,
    /// The derivatives along each direction.
    pub du: D::Arr<T>,
}

/// A value with its derivatives along `N` loose parameters (forward-mode dual number).
pub type Dual<T, const N: usize = 1> = Jet<T, Params<N>>;

impl<T: Coef, D: Directions> Jet<T, D> {
    /// A value with the given derivatives.
    #[inline(always)]
    pub const fn new(re: T, du: D::Arr<T>) -> Self {
        Jet { re, du }
    }

    /// A constant: all derivatives zero.
    #[inline(always)]
    pub fn constant(re: T) -> Self {
        Jet {
            re,
            du: D::from_fn(|_| T::zero()),
        }
    }

    /// The `i`-th variable at `re`: derivative 1 along direction `i`, 0 along the others.
    #[inline(always)]
    pub fn variable(re: T, i: usize) -> Self {
        Jet {
            re,
            du: D::from_fn(|k| if k == i { T::one() } else { T::zero() }),
        }
    }
}

impl<T: Real, D: Directions> Jet<T, D> {
    /// `self` with the derivatives scaled by `f'` at the value: the chain rule for a function
    /// with value `f` and derivative `df` at `re`. A derivative that is exactly zero stays zero
    /// even where `df` is infinite: `√x` at `x = 0` inside `cos √x`, as the closed forms
    /// compute a rotation angle, has an infinite derivative but a zero tangent there, and the
    /// composition is smooth (plain forward mode would give `0 · ∞ = NaN`).
    #[inline(always)]
    fn chain(self, f: T, df: T) -> Self {
        Jet {
            re: f,
            du: D::map(&self.du, |&d| {
                T::select_lt(T::zero(), d.abs(), d * df, T::zero())
            }),
        }
    }
}

impl<T: Coef, D: Directions> From<T> for Jet<T, D> {
    /// A constant.
    #[inline(always)]
    fn from(re: T) -> Self {
        Self::constant(re)
    }
}

impl<T: Coef, D: Directions> Add for Jet<T, D> {
    type Output = Self;
    #[inline(always)]
    fn add(self, o: Self) -> Self {
        Jet {
            re: self.re + o.re,
            du: D::zip(&self.du, &o.du, |&a, &b| a + b),
        }
    }
}

impl<T: Coef, D: Directions> Sub for Jet<T, D> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, o: Self) -> Self {
        Jet {
            re: self.re - o.re,
            du: D::zip(&self.du, &o.du, |&a, &b| a - b),
        }
    }
}

impl<T: Coef, D: Directions> Mul for Jet<T, D> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, o: Self) -> Self {
        Jet {
            re: self.re * o.re,
            du: D::zip(&self.du, &o.du, |&a, &b| a.mul_add(o.re, self.re * b)),
        }
    }
}

impl<T: Coef, D: Directions> Neg for Jet<T, D> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Jet {
            re: -self.re,
            du: D::map(&self.du, |&d| -d),
        }
    }
}

// The quotient rule needs `*` and `-` inside `Div`.
#[allow(clippy::suspicious_arithmetic_impl)]
impl<T: Real, D: Directions> Div for Jet<T, D> {
    type Output = Self;
    #[inline(always)]
    fn div(self, o: Self) -> Self {
        let inv = o.re.recip();
        let q = self.re * inv;
        Jet {
            re: q,
            // (a/b)' = (a' − q b') / b
            du: D::zip(&self.du, &o.du, |&a, &b| (a - q * b) * inv),
        }
    }
}

impl<T: Coef, D: Directions> Coef for Jet<T, D> {
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
        Jet {
            re: self.re.mul_add(a.re, b.re),
            // du·a + (re·da + db), as two whole-array zips (indexing slices would not unroll).
            du: D::zip(
                &self.du,
                &D::zip(&a.du, &b.du, |&x, &y| self.re.mul_add(x, y)),
                |&s, &t| s.mul_add(a.re, t),
            ),
        }
    }
    #[inline(always)]
    fn vectorize<R>(f: impl FnOnce() -> R) -> R {
        T::vectorize(f)
    }
}

impl<T: Real, D: Directions> Real for Jet<T, D> {
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
        Jet {
            re: self.re.atan2(x.re),
            du: D::zip(&self.du, &x.du, |&dy, &dx| (x.re * dy - self.re * dx) * inv),
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
        Jet {
            re: T::select_lt(a.re, b.re, x.re, y.re),
            du: D::zip(&x.du, &y.du, |&p, &q| T::select_lt(a.re, b.re, p, q)),
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

/// `(f(x), ∇f(x))` for a scalar function of `N` loose variables, in one evaluation. For a
/// function of a gax value, [`gradient`] gives the gradient as a typed linear form.
#[inline]
pub fn gradient_array<T: Real, const N: usize>(
    f: impl FnOnce([Dual<T, N>; N]) -> Dual<T, N>,
    x: [T; N],
) -> (T, [T; N]) {
    let y = f(core::array::from_fn(|i| Dual::variable(x[i], i)));
    (y.re, y.du)
}

/// `(f(x), J)` for a function of `N` loose variables with `M` outputs: `J[m][n] = ∂fₘ/∂xₙ`,
/// in one evaluation. For a function of a gax value, [`jacobian`] gives a typed map.
#[inline]
#[allow(clippy::type_complexity)]
pub fn jacobian_array<T: Real, const N: usize, const M: usize>(
    f: impl FnOnce([Dual<T, N>; N]) -> [Dual<T, N>; M],
    x: [T; N],
) -> ([T; M], [[T; N]; M]) {
    let y = f(core::array::from_fn(|i| Dual::variable(x[i], i)));
    (y.map(|v| v.re), y.map(|v| v.du))
}

/// The coefficients of a value of kind `A`, each its own variable: `x` with derivative 1 along
/// its own coefficient. What [`jacobian`] and [`gradient`] pass to the function.
#[inline]
pub fn variables<X>(x: X) -> Retype<X, (), Jet<X::Coef, X::Kind>>
where
    X: Extensor<Slots = (), Coef: Real>,
{
    let c = x.coeffs().as_ref();
    Extensor::from_coeffs(<X::Kind as Kind>::arr_from_fn(|i| Jet::variable(c[i], i)))
}

/// The value of a function of a gax value, at the variables of kind `A`: the derivatives taken
/// with respect to a value of `A`.
type At<X, C> = Retype<X, (), C>;
/// First-order jets with respect to `X`'s kind.
type J1<X> = Jet<<X as Extensor>::Coef, <X as Extensor>::Kind>;
/// The scalar kind of `X`'s algebra.
type ScalarKind<X> = <<X as Extensor>::Kind as Kind>::Scalar;
/// The scalar kind of `X`'s algebra with slots `S`.
type ScalarOf<X, S> =
    <<<X as Extensor>::Kind as Kind>::Scalar as Kind>::Mv<S, <X as Extensor>::Coef>;

/// `(f(x), Df(x))` for a function from values of kind `A` to values or maps `Y`: the
/// derivative is a `Y` with one more slot, of kind `A`, after its own, so that binding a step
/// `dx` gives the first-order change, `f(x + dx) ≈ f(x) + Df(x).at::<last>().of(dx)` (for a
/// value `Y` simply `Df(x).of(dx)`). numga's `derivative`. In one evaluation of `f`, carrying
/// `A::N` derivatives per coefficient.
///
/// `f` sees its argument with [`Jet`] coefficients, so it is written generic over the
/// coefficient type, or as a closure over the jet type; constants enter with
/// `map_coefs(T::from_f64)`.
///
/// ```
/// use gax::ApproxEq;
/// use gax::dual::jacobian;
/// use gax::pga3d::{Line, Motor, Point};
/// use gax::Real;
///
/// // The motor of a twist, and its derivative: a map from twists to motors.
/// let b = Line::<(), f64>::new(0.3, -0.2, 0.5, 0.1, 0.4, -0.3);
/// let (m, dm) = jacobian(|b| b.exp().into_inner(), b);
/// let _: Motor<(Line,), f64> = dm;
/// assert!(m.approx_eq(&b.exp().into_inner(), 1e-15));
/// let db = Line::new(1e-7, 0.0, 0.0, 0.0, 2e-7, 0.0);
/// assert!((b + db).exp().into_inner().approx_eq(&(m + dm.of(db)), 1e-13));
/// ```
#[inline]
#[allow(clippy::type_complexity)]
pub fn jacobian<X, Y>(
    f: impl FnOnce(At<X, J1<X>>) -> Y,
    x: X,
) -> (
    Retype<Y, Y::Slots, X::Coef>,
    Retype<Y, Cat<Y::Slots, (X::Kind,)>, X::Coef>,
)
where
    X: Extensor<Slots = (), Coef: Real>,
    Y: Extensor<Coef = J1<X>>,
{
    let y = f(variables(x));
    let n = <X::Kind as Kind>::N;
    let value = <Y::Kind as Kind>::arr_map(y.coeffs(), |col| {
        <Y::Slots as Slots>::map(col, &mut |j: &J1<X>| j.re)
    });
    let slope = <Y::Kind as Kind>::arr_map(y.coeffs(), |col| {
        <Cat<Y::Slots, (X::Kind,)> as Slots>::from_flat(
            &mut |flat| <Y::Slots as Slots>::get_flat(col, flat / n).du.as_ref()[flat % n],
            0,
        )
    });
    (Extensor::from_coeffs(value), Extensor::from_coeffs(slope))
}

/// A scalar result: a coefficient, or a value of an algebra's scalar kind. What [`gradient`]
/// and [`hessian`] accept from the function.
pub trait IntoScalar<C> {
    /// The scalar as a coefficient.
    fn into_scalar(self) -> C;
}

impl<T: Elem, D: Directions> IntoScalar<Jet<T, D>> for Jet<T, D> {
    #[inline(always)]
    fn into_scalar(self) -> Self {
        self
    }
}

impl<M> IntoScalar<M::Coef> for M
where
    M: Extensor<Slots = ()>,
    M::Kind: Kind<Scalar = M::Kind>,
{
    #[inline(always)]
    fn into_scalar(self) -> M::Coef {
        self.coeffs().as_ref()[0]
    }
}

/// `(f(x), ∇f(x))` for a scalar function of a value of kind `A`: the gradient is the linear
/// form `Scalar <- (A,)` that sends a step to the first-order change of `f` (numga's
/// `derivative` of a scalar function), not an element of `A`. One evaluation of `f`.
///
/// ```
/// use gax::dual::gradient;
/// use gax::pga3d::Plane;
/// let x = Plane::<(), f64>::new(1.0, 2.0, 3.0, 4.0);
/// let (e, g) = gradient(|x: Plane<(), _>| x.norm_squared(), x);
/// assert_eq!(e, 14.0);
/// assert_eq!(g.of(Plane::new(0.0, 1.0, 0.0, 0.0)).s(), 4.0); // 2 x · step
/// ```
#[inline]
pub fn gradient<X, Y>(f: impl FnOnce(At<X, J1<X>>) -> Y, x: X) -> (X::Coef, ScalarOf<X, (X::Kind,)>)
where
    X: Extensor<Slots = (), Coef: Real>,
    Y: IntoScalar<J1<X>>,
{
    let y = f(variables(x)).into_scalar();
    let form = <<X::Kind as Kind>::Scalar as Kind>::arr_from_fn(|_| y.du);
    (y.re, Extensor::from_coeffs(form))
}

/// Second-order jets with respect to `X`'s kind: a jet of jets.
type J2<X> = Jet<J1<X>, <X as Extensor>::Kind>;

/// `(f(x), ∇f(x), ∇²f(x))` for a scalar function of a value of kind `A`: the gradient as the
/// form `Scalar <- (A,)` and the Hessian as the symmetric bilinear form `Scalar <- (A, A)`. A
/// Newton step is `-hessian.solve(gradient)`, a value of `A`. One evaluation of `f` on jets of
/// jets, `(1 + A::N)²` numbers per coefficient: written generic over the coefficient type, the
/// same `f` serves [`gradient`] and `hessian`.
///
/// ```
/// use gax::dual::hessian;
/// use gax::pga3d::Plane;
/// use gax::Real;
///
/// // A strictly convex energy of a plane's coefficients, least at (ln 1, ln 2, ln 3, ln 4).
/// fn energy<T: Real>(x: Plane<(), T>) -> T {
///     x.c.iter().zip(1..).fold(T::zero(), |e, (&c, t)| e + c.exp() - T::from_i64(t) * c)
/// }
/// let mut x = Plane::<(), f64>::zero();
/// for _ in 0..12 {
///     let (_, g, h) = hessian(energy, x); // Scalar <- Plane, Scalar <- (Plane, Plane)
///     x = x - h.solve(g); // a Newton step
/// }
/// assert!((x.c[3] - 4f64.ln()).abs() < 1e-14);
/// ```
#[inline]
#[allow(clippy::type_complexity)]
pub fn hessian<X, Y>(
    f: impl FnOnce(At<X, J2<X>>) -> Y,
    x: X,
) -> (
    X::Coef,
    ScalarOf<X, (X::Kind,)>,
    ScalarOf<X, (X::Kind, X::Kind)>,
)
where
    X: Extensor<Slots = (), Coef: Real>,
    Y: IntoScalar<J2<X>>,
{
    let c = x.coeffs().as_ref();
    // The outer and inner directions are the same coefficients.
    let vars = <X::Kind as Kind>::arr_from_fn(|i| {
        Jet::<J1<X>, X::Kind>::variable(Jet::variable(c[i], i), i)
    });
    let y = f(Extensor::from_coeffs(vars)).into_scalar();
    let grad = y.re.du;
    let hess = <X::Kind as Kind>::arr_map(&y.du, |row: &J1<X>| row.du);
    (
        y.re.re,
        Extensor::from_coeffs(<ScalarKind<X> as Kind>::arr_from_fn(|_| grad)),
        Extensor::from_coeffs(<ScalarKind<X> as Kind>::arr_from_fn(|_| hess)),
    )
}
