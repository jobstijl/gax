//! Derivatives with respect to a whole field (ADR-046): the gradient as a field of forms, the
//! Hessian as a sparse field map of form cells, every site coupled to every other.

extern crate std;

use super::{Field, SparseMap};
use crate::coef::{Coef, Real};
use crate::dual::{IntoScalar, Jet, variables};
use crate::kind::{Extensor, Kind, Retype};
use std::vec::Vec;

/// First-order jets with respect to `X`'s kind.
type J1<X> = Jet<<X as Extensor>::Coef, <X as Extensor>::Kind>;
/// Second-order jets: jets of jets.
type J2<X> = Jet<J1<X>, <X as Extensor>::Kind>;
/// The scalar kind of `X`'s algebra with slots `S` and `X`'s coefficients.
type ScalarOf<X, S> =
    <<<X as Extensor>::Kind as Kind>::Scalar as Kind>::Mv<S, <X as Extensor>::Coef>;

/// The value of `f` and its gradient with respect to the field `x`: a field of linear forms
/// `Scalar <- A`, one per site, numga's `derivative` of a scalar function of `A[n]`. Binding a
/// step at each site and summing over the sites gives the first-order change.
///
/// `f` sees the field with [`Jet`] coefficients, so it is written generic over the coefficient
/// type, as for [`dual::gradient`](crate::dual::gradient). It is evaluated once per site, each
/// time with derivatives along that site's coefficients only, so the cost is `n` evaluations of
/// `f` carrying `A::N` derivatives.
///
/// ```
/// use gax::field::{gradient, Field};
/// use gax::pga3d::Plane;
/// use gax::Real;
///
/// // The squared lengths of the steps between neighbouring sites.
/// fn bending<T: Real>(x: &Field<Plane<(), T>>) -> T {
///     x.sites().windows(2).fold(T::zero(), |e, w| e + (w[1] - w[0]).norm_squared())
/// }
/// let x = Field::from_fn(3, |i| Plane::<(), f64>::new(i as f64 * i as f64, 0.0, 0.0, 0.0));
/// let (e, g) = gradient(bending, &x);
/// assert_eq!(e, 1.0 + 9.0);
/// // At the middle site: 2 (x1 − x0) − 2 (x2 − x1) along e1.
/// assert_eq!(g.site(1).c[0][0], 2.0 * 1.0 - 2.0 * 3.0);
/// ```
#[allow(clippy::type_complexity)]
pub fn gradient<X, Y>(
    mut f: impl FnMut(&Field<Retype<X, (), J1<X>>>) -> Y,
    x: &Field<X>,
) -> (X::Coef, Field<ScalarOf<X, (X::Kind,)>>)
where
    X: Extensor<Slots = (), Coef: Real>,
    Y: IntoScalar<J1<X>>,
{
    let constants: Field<Retype<X, (), J1<X>>> = x.map(|v| v.map_coefs(Jet::constant));
    if x.is_empty() {
        return (f(&constants).into_scalar().re, Field::new(Vec::new()));
    }
    let mut value = X::Coef::zero();
    let mut seeded = constants.clone();
    let grad = Field::from_fn(x.len(), |i| {
        seeded.sites_mut()[i] = variables(x.site(i));
        let y = f(&seeded).into_scalar();
        seeded.sites_mut()[i] = constants.site(i);
        value = y.re;
        Extensor::from_coeffs(<<X::Kind as Kind>::Scalar as Kind>::arr_from_fn(|_| y.du))
    });
    (value, grad)
}

/// The value of `f`, its gradient and its Hessian with respect to the field `x`: the gradient
/// as in [`gradient`], the Hessian as a sparse field map of form cells
/// `Scalar <- (A, A)`, one per pair of sites whose block is not identically zero at `x`. A
/// Newton step solves the one against the other: with the cells as maps,
/// `hessian.map_cells(Scalar::as_map).solve(&g)` for the gradient's forms read as values.
///
/// This is the general path: `n (n + 1) / 2` evaluations of `f` on jets of jets, each pair of
/// sites seeded once, so it suits small fields. An energy that is a sum of local terms (the
/// springs of a string, the elements of a mesh) has a sparse Hessian that the chain rule
/// assembles in one pass: with `D` the sparse map from the field to the terms' arguments
/// (differences of neighbours, a triangle's corners), the gradient is `~D * g` and the Hessian
/// `~D * diag(H) * D`, from [`dual::hessian`](crate::dual::hessian) on each term (see the guide,
/// section "Fields").
///
/// ```
/// use gax::field::{hessian, Field};
/// use gax::pga3d::{Plane, Scalar};
/// use gax::Real;
///
/// fn bending<T: Real>(x: &Field<Plane<(), T>>) -> T {
///     x.sites().windows(2).fold(T::zero(), |e, w| e + (w[1] - w[0]).norm_squared())
/// }
/// let x = Field::from_fn(3, |i| Plane::<(), f64>::new(i as f64, 0.0, 0.0, 0.0));
/// let (_, _, h) = hessian(bending, &x);
/// // Neighbours couple, the ends do not: a path's Laplacian, twice.
/// assert_eq!(h.couplings_len(), 7);
/// let middle = h.couplings().find(|&(r, c, _)| r == 1 && c == 1).unwrap().2;
/// assert_eq!(middle.c[0][0][0], 4.0);
/// ```
#[allow(clippy::type_complexity)]
pub fn hessian<X, Y>(
    mut f: impl FnMut(&Field<Retype<X, (), J2<X>>>) -> Y,
    x: &Field<X>,
) -> (
    X::Coef,
    Field<ScalarOf<X, (X::Kind,)>>,
    SparseMap<ScalarOf<X, (X::Kind, X::Kind)>>,
)
where
    X: Extensor<Slots = (), Coef: Real>,
    Y: IntoScalar<J2<X>>,
{
    type S<X> = <<X as Extensor>::Kind as Kind>::Scalar;
    let n = x.len();
    // Site `k` seeded with outer derivatives if `k == i` and inner ones if `k == j`.
    let seeded = |i: usize, j: usize| -> Field<Retype<X, (), J2<X>>> {
        Field::from_fn(n, |k| {
            let c = x.site(k);
            let c = c.coeffs().as_ref();
            Extensor::from_coeffs(<X::Kind as Kind>::arr_from_fn(|q| {
                let inner: J1<X> = if k == j {
                    Jet::variable(c[q], q)
                } else {
                    Jet::constant(c[q])
                };
                if k == i {
                    Jet::variable(inner, q)
                } else {
                    Jet::constant(inner)
                }
            }))
        })
    };
    let zero = X::Coef::zero();
    let mut value = zero;
    let mut grad = Vec::with_capacity(n);
    let mut triples = Vec::new();
    for i in 0..n {
        for j in i..n {
            let y = f(&seeded(i, j)).into_scalar();
            if i == j {
                value = y.re.re;
                let g = y.re.du;
                grad.push(Extensor::from_coeffs(<S<X> as Kind>::arr_from_fn(|_| g)));
            }
            // h[a][b] = ∂²f / ∂x_i,a ∂x_j,b.
            let h = <X::Kind as Kind>::arr_map(&y.du, |row: &J1<X>| row.du);
            let nonzero = h
                .as_ref()
                .iter()
                .any(|row| row.as_ref().iter().any(|&v| v != zero));
            if nonzero {
                triples.push((
                    i,
                    j,
                    Extensor::from_coeffs(<S<X> as Kind>::arr_from_fn(|_| h)),
                ));
                if i != j {
                    let t = <X::Kind as Kind>::arr_from_fn(|b| {
                        <X::Kind as Kind>::arr_from_fn(|a| h.as_ref()[a].as_ref()[b])
                    });
                    triples.push((
                        j,
                        i,
                        Extensor::from_coeffs(<S<X> as Kind>::arr_from_fn(|_| t)),
                    ));
                }
            }
        }
    }
    if n == 0 {
        value = f(&Field::new(Vec::new())).into_scalar().re.re;
    }
    (
        value,
        Field::new(grad),
        SparseMap::from_triples(triples, (n, n)),
    )
}
