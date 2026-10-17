//! Geometric calculus: the derivative with respect to a multivector, through the reciprocal
//! frame of its kind (ADR-047).
//!
//! The derivative of `F` with respect to a value `X` of a kind with blades `e_J` is
//! `∂_X F = Σ_J e^J ∂_J F`, where `∂_J` is the derivative along the coefficient of `e_J` and
//! `e^J` its reciprocal blade, `e^J ∗ e_K = δ^J_K` (`∗` the scalar product). With respect to a
//! vector it is the vector derivative `∇F`, whose grade parts are the divergence `∇·F` and the
//! curl `∇∧F`: in spacetime algebra Maxwell's equations are `∇F = J`.
//!
//! [`jacobian`](crate::dual::jacobian) gives the derivative as a map, which needs no metric.
//! Taking its blades back to elements does: the reciprocal frame inverts the Gram matrix of the
//! kind's blades. A kind holding a null direction, such as any PGA kind with `e0` in a blade,
//! has a singular Gram matrix and no frame, so these functions do not exist for it (a compile
//! error that says so); its derivative stays the form that `gradient` returns.
//!
//! ```
//! use gax::calculus::{curl, divergence, nabla};
//! use gax::vga3d::{Bivector, Vector};
//! use gax::Coef;
//!
//! // A rotation field, x ↦ x ⌋ e12: divergence free, with curl 2 e12.
//! fn field<T: Coef>(x: Vector<(), T>) -> Vector<(), T> {
//!     x.lc(Bivector::new(T::zero(), T::zero(), T::one()))
//! }
//! let x = Vector::<(), f64>::new(0.3, -1.0, 2.0);
//! let (_, d) = nabla(field, x);
//! let (_, div) = divergence(field, x);
//! let (_, rot) = curl(field, x);
//! assert_eq!(div.s(), 0.0);
//! assert_eq!(rot, Bivector::new(0.0, 0.0, 2.0));
//! assert_eq!(d.grade::<2>(), rot);
//! ```

use crate::coef::Coef;
use crate::dual::{Jet, variables};
use crate::kind::{Extensor, Kind, Retype};
use crate::ops::{Gp, Lc, Wedge};
use core::ops::Add;

/// A kind whose blades have a reciprocal frame: the Gram matrix of its blades under the scalar
/// product is invertible. Generated for every such kind, from the exact inverse.
///
/// ```
/// use gax::calculus::reciprocal;
/// use gax::sta::Vector;
/// // In spacetime, x² = −1: the reciprocal of e1 is −e1, and e^1 ∗ e1 = 1.
/// let e1 = Vector::<(), f64>::new(0.0, 1.0, 0.0, 0.0);
/// let e1_up: Vector<(), f64> = reciprocal(1);
/// assert_eq!(e1_up, -e1);
/// assert_eq!(e1_up.scalar_product(e1).s(), 1.0);
/// ```
#[diagnostic::on_unimplemented(
    message = "`{Self}` has no reciprocal frame",
    label = "no reciprocal frame",
    note = "the Gram matrix of the kind's blades under the scalar product is singular: the kind holds a null direction (such as PGA's `e0`), so a derivative cannot be taken back to an element of it",
    note = "`gax::dual::gradient` and `jacobian` give the derivative as the form or map it is, which needs no metric"
)]
pub trait Reciprocal: Kind {
    /// The reciprocal blades: `FRAME[i]` lists `e^i` as `(coefficient index, numerator,
    /// denominator)` in this kind's layout.
    const FRAME: &'static [&'static [(usize, i64, i64)]];
}

/// The reciprocal blade `e^i` of a kind's coefficient `i`: `e^i ∗ e_k = δ_ik`.
#[inline]
pub fn reciprocal<X>(i: usize) -> X
where
    X: Extensor<Slots = ()>,
    X::Kind: Reciprocal,
{
    let mut c = <X::Kind as Kind>::arr_from_fn(|_| X::Coef::zero());
    for &(k, num, den) in <X::Kind as Reciprocal>::FRAME[i] {
        c.as_mut()[k] = X::Coef::from_ratio(num, den);
    }
    X::from_coeffs(c)
}

/// First-order jets with respect to `X`'s kind.
type J1<X> = Jet<<X as Extensor>::Coef, <X as Extensor>::Kind>;
/// Second-order jets with respect to `X`'s kind.
type J2<X> = Jet<J1<X>, <X as Extensor>::Kind>;
/// `X`'s kind with coefficients `C`.
type At<X, C> = Retype<X, (), C>;
/// A value of `Y`'s kind with the coefficients of `X`.
type ValueOf<Y, X> = Retype<Y, (), <X as Extensor>::Coef>;

/// `f(x)` and the derivatives `∂_k f(x)` along each coefficient of `x`, as values.
#[allow(clippy::type_complexity)]
fn columns<X, Y>(
    f: impl FnOnce(At<X, J1<X>>) -> Y,
    x: X,
) -> (ValueOf<Y, X>, impl Iterator<Item = ValueOf<Y, X>>)
where
    X: Extensor<Slots = ()>,
    Y: Extensor<Slots = (), Coef = J1<X>>,
{
    let y = f(variables(x));
    let value = Extensor::from_coeffs(<Y::Kind as Kind>::arr_map(y.coeffs(), |j: &J1<X>| j.re));
    let column = move |k: usize| -> ValueOf<Y, X> {
        Extensor::from_coeffs(<Y::Kind as Kind>::arr_map(y.coeffs(), |j: &J1<X>| {
            j.du.as_ref()[k]
        }))
    };
    (value, (0..<X::Kind as Kind>::N).map(column))
}

/// `(f(x), Σ_J product(e^J, ∂_J f(x)))`: the derivative combined with the reciprocal blades by
/// any product. [`nabla`], [`divergence`] and [`curl`] are it with `*`, `⌋` and `∧`; others give
/// the derivative projected on a subspace, such as `(dX · ∇) f = Σ (dX ⌊ e^J) ∂_J f` along a
/// simplex's measure `dX`, the operator of the fundamental theorem
/// (`gax::simplex`).
///
/// ```
/// use gax::calculus::contract;
/// use gax::vga3d::{Bivector, Scalar, Vector};
/// // The derivative of x² along the plane e12 only: its x and y parts, 2x and 2y (rotated
/// // into the plane, B ⌊ e^k ∂_k f).
/// let b = Bivector::<(), f64>::new(0.0, 0.0, 1.0);
/// let x = Vector::<(), f64>::new(1.0, 2.0, 3.0);
/// let (_, d) = contract(|x: Vector<(), _>| x.scalar_product(x), x, |up, column| b.rc(up) * column);
/// assert_eq!(d, Vector::new(4.0, -2.0, 0.0));
/// ```
#[inline]
#[allow(clippy::type_complexity)]
pub fn contract<X, Y, O>(
    f: impl FnOnce(At<X, J1<X>>) -> Y,
    x: X,
    product: impl Fn(X, ValueOf<Y, X>) -> O,
) -> (ValueOf<Y, X>, O)
where
    X: Extensor<Slots = ()>,
    X::Kind: Reciprocal,
    Y: Extensor<Slots = (), Coef = J1<X>>,
    O: Add<Output = O>,
{
    let (value, columns) = columns(f, x);
    let sum = columns
        .enumerate()
        .map(|(k, column)| product(reciprocal::<X>(k), column))
        .reduce(|a, b| a + b)
        .expect("a kind has at least one blade");
    (value, sum)
}

/// `(f(x), ∂_X f(x))`: the derivative with respect to the value `x`, `Σ_J e^J ∂_J f` with the
/// reciprocal blades on the left. With respect to a vector, the vector derivative `∇f`. Its kind
/// is that of the geometric product of `x` and `f(x)`.
///
/// `f` sees its argument with [`Jet`] coefficients, as for [`jacobian`](crate::dual::jacobian),
/// so it is written generic over the coefficient type, or as a closure over the jet type.
///
/// ```
/// use gax::calculus::nabla;
/// use gax::vga3d::{Scalar, Vector};
/// // ∇(x²) = 2x.
/// let x = Vector::<(), f64>::new(1.0, -2.0, 0.5);
/// let (r2, grad) = nabla(|x: Vector<(), _>| x * x, x);
/// assert_eq!(r2.cast::<Scalar>().s(), 5.25);
/// assert_eq!(grad.cast::<Vector>(), x * 2.0);
/// ```
#[inline]
#[allow(clippy::type_complexity)]
pub fn nabla<X, Y, O>(f: impl FnOnce(At<X, J1<X>>) -> Y, x: X) -> (ValueOf<Y, X>, O)
where
    X: Extensor<Slots = ()> + Gp<ValueOf<Y, X>, Output = O>,
    X::Kind: Reciprocal,
    Y: Extensor<Slots = (), Coef = J1<X>>,
    O: Add<Output = O>,
{
    contract(f, x, Gp::gp)
}

/// `(f(x), ∂_X ⌋ f(x))`: the inner part of the derivative, `Σ_J e^J ⌋ ∂_J f`. With respect to a
/// vector, the divergence `∇·f`: the grade-lowering part of [`nabla`].
///
/// ```
/// use gax::calculus::divergence;
/// use gax::vga3d::Vector;
/// // The identity field in three dimensions has divergence 3.
/// let (_, div) = divergence(|x: Vector<(), _>| x, Vector::<(), f64>::new(0.1, 0.2, 0.3));
/// assert_eq!(div.s(), 3.0);
/// ```
#[inline]
#[allow(clippy::type_complexity)]
pub fn divergence<X, Y, O>(f: impl FnOnce(At<X, J1<X>>) -> Y, x: X) -> (ValueOf<Y, X>, O)
where
    X: Extensor<Slots = ()> + Lc<ValueOf<Y, X>, Output = O>,
    X::Kind: Reciprocal,
    Y: Extensor<Slots = (), Coef = J1<X>>,
    O: Add<Output = O>,
{
    contract(f, x, Lc::lc)
}

/// `(f(x), ∂_X ∧ f(x))`: the outer part of the derivative, `Σ_J e^J ∧ ∂_J f`. With respect to a
/// vector, the curl `∇∧f`: the grade-raising part of [`nabla`].
///
/// ```
/// use gax::calculus::curl;
/// use gax::vga3d::Vector;
/// // A gradient field has no curl: ∇∧∇(x²) = ∇∧(2x) = 0.
/// let (_, rot) = curl(|x: Vector<(), _>| x + x, Vector::<(), f64>::new(0.1, 0.2, 0.3));
/// assert!(rot.c.iter().all(|&c| c == 0.0));
/// ```
#[inline]
#[allow(clippy::type_complexity)]
pub fn curl<X, Y, O>(f: impl FnOnce(At<X, J1<X>>) -> Y, x: X) -> (ValueOf<Y, X>, O)
where
    X: Extensor<Slots = ()> + Wedge<ValueOf<Y, X>, Output = O>,
    X::Kind: Reciprocal,
    Y: Extensor<Slots = (), Coef = J1<X>>,
    O: Add<Output = O>,
{
    contract(f, x, Wedge::wedge)
}

/// `(f(x), ∂_X ∗ ∂_X f(x))`: the second derivative contracted by the scalar product,
/// `Σ_{I,J} (e^I ∗ e^J) ∂_I ∂_J f`. With respect to a vector, the Laplacian `∇²f = ∇·∇f`, of the
/// kind of `f`. One evaluation of `f` on jets of jets, as [`hessian`](crate::dual::hessian).
///
/// ```
/// use gax::calculus::laplacian;
/// use gax::sta::{Scalar, Vector};
/// use gax::Real;
/// // The wave operator of spacetime: a plane wave f(t − x) satisfies □f = 0 (t² = 1, x² = −1).
/// let wave = |p: Vector<(), _>| {
///     let phase = p.c[0] - p.c[1];
///     Scalar::new(phase * phase * phase)
/// };
/// let (_, box_f) = laplacian(wave, Vector::<(), f64>::new(0.4, 0.1, 0.0, 0.0));
/// assert_eq!(box_f.s(), 0.0);
/// ```
#[inline]
#[allow(clippy::type_complexity)]
pub fn laplacian<X, Y>(f: impl FnOnce(At<X, J2<X>>) -> Y, x: X) -> (ValueOf<Y, X>, ValueOf<Y, X>)
where
    X: Extensor<Slots = ()>,
    X::Kind: Reciprocal,
    Y: Extensor<Slots = (), Coef = J2<X>>,
{
    let c = x.coeffs().as_ref();
    // The outer and inner directions are the same coefficients.
    let vars = <X::Kind as Kind>::arr_from_fn(|i| {
        Jet::<J1<X>, X::Kind>::variable(Jet::variable(c[i], i), i)
    });
    let y = f(Extensor::from_coeffs(vars));
    let value = Extensor::from_coeffs(<Y::Kind as Kind>::arr_map(y.coeffs(), |j: &J2<X>| j.re.re));
    // e^I ∗ e^J is the coefficient of e_J in e^I: the frame is the inverse Gram matrix.
    let lap = <Y::Kind as Kind>::arr_map(y.coeffs(), |j: &J2<X>| {
        let mut sum = X::Coef::zero();
        for (i, row) in <X::Kind as Reciprocal>::FRAME.iter().enumerate() {
            for &(k, num, den) in *row {
                sum = sum + X::Coef::from_ratio(num, den) * j.du.as_ref()[i].du.as_ref()[k];
            }
        }
        sum
    });
    (value, Extensor::from_coeffs(lap))
}
