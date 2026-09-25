//! Methods of maps and forms: inverse, solve, determinant, trace, SVD, eigenproblems.
//!
//! These are blanket traits over every generated kind, so they work for any algebra. The
//! generated types expose them as inherent methods (`map.inverse()`, `form.eigh_with(m)`),
//! which forward here. Results are typed: the inverse of a `B <- A` map is an `A <- B` map,
//! singular vectors and eigenvectors are values of the slot kinds.

use crate::coef::{Coef, Real};
use crate::fill::SplitLast;
use crate::kind::{Extensor, Kind};
use crate::linalg::{self, SquareArr};
use crate::slots::Slots;

/// A map `B <- A` between kinds with the same number of coefficients.
///
/// ```
/// use gax::pga3d::{Motor, Point};
/// let m = Motor::<(), f64>::rotation_about(0.0, 0.0, 1.0, 0.4);
/// let t: Point<(Point,), f64> = m >> Point::slot();
/// let p = Point::xyz(1.0, 2.0, 3.0);
/// let back = t.solve(t.of(p));
/// assert!((back.e032() - 1.0).abs() < 1e-12);
/// assert!((t.det() - 1.0).abs() < 1e-12);
/// let _undo: Point<(Point,), f64> = t.inverse();
/// ```
pub trait SquareMap: Extensor<Coef: Real> {
    /// The input kind `A`.
    type Input: Kind;
    /// The inverse map `A <- B`.
    type Inverse: Extensor<Kind = Self::Input, Coef = Self::Coef>;

    /// The inverse map, by LU factorization with partial pivoting.
    fn inverse(self) -> Self::Inverse;
    /// The determinant of the coefficient matrix.
    ///
    /// For a map between different kinds this depends on the blade layouts (orientation and
    /// order); for an endomorphism it is basis independent.
    fn det(self) -> Self::Coef;
    /// Solve `self(x) = rhs` for `x`. A right-hand side with slots of its own keeps them: the
    /// solution is then a map with those slots.
    fn solve<X>(self, rhs: X) -> <Self::Input as Kind>::Mv<X::Slots, Self::Coef>
    where
        X: Extensor<Kind = Self::Kind, Coef = Self::Coef>;
    /// Singular value decomposition: `(u, sigma, v)` with `self.of(v[i]) = sigma[i] * u[i]`,
    /// singular values descending.
    #[allow(clippy::type_complexity)]
    fn svd(
        self,
    ) -> (
        <Self::Kind as Kind>::Arr<<Self::Kind as Kind>::Mv<(), Self::Coef>>,
        <Self::Kind as Kind>::Arr<Self::Coef>,
        <Self::Input as Kind>::Arr<<Self::Input as Kind>::Mv<(), Self::Coef>>,
    );
}

/// The number of Jacobi sweeps used for eigen and singular value problems of size `n`.
pub fn sweeps(n: usize) -> usize {
    if n <= 4 {
        8
    } else if n <= 8 {
        10
    } else {
        14
    }
}

fn matrix_from<T: Real, M: SquareArr<T>>(f: impl Fn(usize, usize) -> T) -> M {
    let mut m = M::zero();
    for i in 0..M::N {
        for j in 0..M::N {
            m[i][j] = f(i, j);
        }
    }
    m
}

fn value_from<K: Kind, T: Real>(f: impl Fn(usize) -> T) -> K::Mv<(), T> {
    <K::Mv<(), T> as Extensor>::from_coeffs(K::arr_from_fn(f))
}

impl<M, A> SquareMap for M
where
    M: Extensor<Slots = (A,), Coef: Real>,
    A: Kind,
    <M::Kind as Kind>::Arr<A::Arr<M::Coef>>: SquareArr<M::Coef>,
    A::Arr<<M::Kind as Kind>::Arr<M::Coef>>: SquareArr<M::Coef>,
{
    type Input = A;
    type Inverse = A::Mv<(M::Kind,), M::Coef>;

    fn inverse(self) -> Self::Inverse {
        let inv = linalg::inverse(self.coeffs());
        <Self::Inverse as Extensor>::from_coeffs(matrix_from(|i, j| inv[i][j]))
    }

    fn det(self) -> M::Coef {
        linalg::det(self.coeffs())
    }

    fn solve<X>(self, rhs: X) -> A::Mv<X::Slots, M::Coef>
    where
        X: Extensor<Kind = M::Kind, Coef = M::Coef>,
    {
        let f = linalg::lu(self.coeffs());
        let n = <<M::Kind as Kind>::Arr<A::Arr<M::Coef>> as SquareArr<M::Coef>>::N;
        let size = <X::Slots as Slots>::SIZE;
        let cols = rhs.coeffs().as_ref();
        // One solve per entry of the right-hand side's slot arrays.
        let mut solutions: A::Arr<<X::Slots as Slots>::Arr<M::Coef>> =
            A::arr_from_fn(|_| <X::Slots as Slots>::from_flat(&mut |_| M::Coef::zero(), 0));
        for flat in 0..size {
            let mut b =
                <<M::Kind as Kind>::Arr<A::Arr<M::Coef>> as SquareArr<M::Coef>>::zero_vector();
            for i in 0..n {
                b[i] = <X::Slots as Slots>::get_flat(&cols[i], flat);
            }
            let x = linalg::lu_solve(&f, &b);
            for (i, s) in solutions.as_mut().iter_mut().enumerate() {
                let prev = *s;
                *s = <X::Slots as Slots>::from_flat(
                    &mut |k| {
                        if k == flat {
                            x[i]
                        } else {
                            <X::Slots as Slots>::get_flat(&prev, k)
                        }
                    },
                    0,
                );
            }
        }
        <A::Mv<X::Slots, M::Coef> as Extensor>::from_coeffs(solutions)
    }

    fn svd(
        self,
    ) -> (
        <M::Kind as Kind>::Arr<<M::Kind as Kind>::Mv<(), M::Coef>>,
        <M::Kind as Kind>::Arr<M::Coef>,
        A::Arr<A::Mv<(), M::Coef>>,
    ) {
        let n = <<M::Kind as Kind>::Arr<A::Arr<M::Coef>> as SquareArr<M::Coef>>::N;
        let (u, s, v) = linalg::svd(self.coeffs(), sweeps(n));
        (
            <M::Kind as Kind>::arr_from_fn(|k| value_from::<M::Kind, M::Coef>(|i| u[k][i])),
            <M::Kind as Kind>::arr_from_fn(|k| s[k]),
            A::arr_from_fn(|k| value_from::<A, M::Coef>(|i| v[k][i])),
        )
    }
}

/// A map from a kind to itself.
///
/// ```
/// use gax::pga3d::Point;
/// assert_eq!(Point::<(), f64>::slot().trace(), 4.0); // the identity on a 4-dimensional kind
/// ```
pub trait Endomorphism: SquareMap {
    /// The trace: the sum of the diagonal, a basis-independent invariant.
    fn trace(self) -> Self::Coef;
}

impl<M> Endomorphism for M
where
    M: SquareMap<Input = <M as Extensor>::Kind>,
{
    fn trace(self) -> M::Coef {
        let c = self.coeffs();
        let n = <M::Kind as Kind>::N;
        let mut t = <M::Slots as Slots>::get_flat(&c.as_ref()[0], 0);
        for i in 1..n {
            t = t + <M::Slots as Slots>::get_flat(&c.as_ref()[i], i);
        }
        t
    }
}

/// A bilinear form `Scalar <- (A, A)`.
///
/// ```
/// use gax::pga2d::{Line, Point, Scalar};
/// // Two positive definite forms on twists (points in PGA2D), from dyads of lines.
/// let mut k: Scalar<(Point, Point), f64> = Scalar::zero();
/// let mut m: Scalar<(Point, Point), f64> = Scalar::zero();
/// for (a, b, c) in [(1.0, 0.0, 0.5), (0.0, 1.0, -0.3), (0.7, 0.7, 0.2)] {
///     let l: Scalar<(Point,), f64> = Line::new(a, b, c) & Point::slot();
///     k += l * l;
///     m += (l * l).gp(2.0);
/// }
/// let (values, modes) = k.eigh_with(m); // modes are values of the slot kind
/// assert!(values.iter().all(|v| (v - 0.5).abs() < 1e-10));
/// let _: Point<(), f64> = modes[0];
/// ```
pub trait Form: Extensor<Coef: Real> {
    /// The slot kind.
    type Slot: Kind;

    /// Generalized symmetric eigenproblem against a positive definite `metric` form:
    /// `self(x, ·) = λ metric(x, ·)`. Returns the eigenvalues in ascending order and the
    /// eigenvectors as values of the slot kind, normalized to `metric(x, x) = 1`.
    ///
    /// This is how vibration modes come out as motions: `stiffness.eigh_with(inertia)`
    /// returns twists. If `metric` is not positive definite the results are NaN.
    #[allow(clippy::type_complexity)]
    fn eigh_with(
        self,
        metric: Self,
    ) -> (
        <Self::Slot as Kind>::Arr<Self::Coef>,
        <Self::Slot as Kind>::Arr<<Self::Slot as Kind>::Mv<(), Self::Coef>>,
    );

    /// Symmetric eigenproblem in the coefficient basis (identity metric). Meaningful when the
    /// slot's metric is the identity in its layout, as for Euclidean vectors and rotors.
    #[allow(clippy::type_complexity)]
    fn eigh(
        self,
    ) -> (
        <Self::Slot as Kind>::Arr<Self::Coef>,
        <Self::Slot as Kind>::Arr<<Self::Slot as Kind>::Mv<(), Self::Coef>>,
    );
}

impl<M, A> Form for M
where
    M: Extensor<Slots = (A, A), Coef: Real>,
    A: Kind,
    A::Arr<A::Arr<M::Coef>>: SquareArr<M::Coef>,
{
    type Slot = A;

    fn eigh_with(self, metric: M) -> (A::Arr<M::Coef>, A::Arr<A::Mv<(), M::Coef>>) {
        let a = &self.coeffs().as_ref()[0];
        let b = &metric.coeffs().as_ref()[0];
        let n = <A::Arr<A::Arr<M::Coef>> as SquareArr<M::Coef>>::N;
        let (vals, xs) = linalg::eigh_generalized(&symmetrize(a), &symmetrize(b), sweeps(n));
        (
            A::arr_from_fn(|k| vals[k]),
            A::arr_from_fn(|k| value_from::<A, M::Coef>(|i| xs[k][i])),
        )
    }

    fn eigh(self) -> (A::Arr<M::Coef>, A::Arr<A::Mv<(), M::Coef>>) {
        let a = &self.coeffs().as_ref()[0];
        let n = <A::Arr<A::Arr<M::Coef>> as SquareArr<M::Coef>>::N;
        let (mut vals, mut vecs) = linalg::eigh(&symmetrize(a), sweeps(n));
        linalg::sort_pairs(&mut vals, &mut vecs);
        (
            A::arr_from_fn(|k| vals[k]),
            A::arr_from_fn(|k| value_from::<A, M::Coef>(|i| vecs[k][i])),
        )
    }
}

/// A pairing `Scalar <- (A, B)` between two kinds with the same number of coefficients: a
/// bilinear form, or the regressive product of a plane and a point.
///
/// ```
/// use gax::pga3d::{Motor, Plane, Point, Scalar};
/// let m = Motor::<(), f64>::translation(1.0, 0.0, 0.0);
/// let t: Point<(Point,), f64> = m >> Point::slot();
/// let pairing: Scalar<(Plane, Point), f64> = Plane::slot() & Point::slot();
/// let on_planes: Plane<(Plane,), f64> = pairing.solve(Plane::slot() & t);
/// let (l, p) = (Plane::new(1.0, 0.0, 0.0, -2.0), Point::xyz(3.0, 1.0, 0.0));
/// assert!(((on_planes.of(l) & p).s() - (l & t.of(p)).s()).abs() < 1e-12);
/// ```
pub trait Pairing: Extensor<Coef: Real> {
    /// The first slot's kind.
    type First: Kind;
    /// The second slot's kind.
    type Second: Kind;

    /// Solve `self(x, ·) == rhs(l, ·)` for `x`, for every value of `rhs`'s leading slots `l`.
    ///
    /// With `rhs` a linear form (one slot) the solution is a value; with leading slots it is
    /// a map on them. The induced map of a point map `t` on planes, which exists even when `t`
    /// is singular, is `(Plane::slot() & Point::slot()).solve(Plane::slot() & t)`.
    fn solve<R>(
        self,
        rhs: R,
    ) -> <Self::First as Kind>::Mv<<R::Slots as SplitLast>::Init, Self::Coef>
    where
        R: Extensor<Kind = Self::Kind, Coef = Self::Coef>,
        R::Slots: SplitLast<Last = Self::Second>;
}

impl<M, A, B> Pairing for M
where
    M: Extensor<Slots = (A, B), Coef: Real>,
    A: Kind,
    B: Kind,
    A::Arr<B::Arr<M::Coef>>: SquareArr<M::Coef>,
    B::Arr<A::Arr<M::Coef>>: SquareArr<M::Coef>,
{
    type First = A;
    type Second = B;

    fn solve<R>(self, rhs: R) -> A::Mv<<R::Slots as SplitLast>::Init, M::Coef>
    where
        R: Extensor<Kind = M::Kind, Coef = M::Coef>,
        R::Slots: SplitLast<Last = B>,
    {
        type Init<R> = <<R as Extensor>::Slots as SplitLast>::Init;
        // form(x, y) = Σ_ij F[i][j] x_i y_j, so form(x, ·) = r(·) is Fᵀ x = r.
        let f = &self.coeffs().as_ref()[0];
        let ft: B::Arr<A::Arr<M::Coef>> = matrix_from(|i, j| f[j][i]);
        let lu = linalg::lu(&ft);
        let r = &rhs.coeffs().as_ref()[0];
        let nb = B::N;
        let solve_at = |l: usize| {
            let mut b = <B::Arr<A::Arr<M::Coef>> as SquareArr<M::Coef>>::zero_vector();
            for j in 0..nb {
                b[j] = <R::Slots as Slots>::get_flat(r, l * nb + j);
            }
            linalg::lu_solve(&lu, &b)
        };
        let coeffs = A::arr_from_fn(|i| <Init<R> as Slots>::from_flat(&mut |l| solve_at(l)[i], 0));
        <A::Mv<Init<R>, M::Coef> as Extensor>::from_coeffs(coeffs)
    }
}

/// `(F + Fᵀ) / 2`: a form's symmetric part, which is all an eigenproblem sees.
fn symmetrize<T: Real, S: SquareArr<T>>(f: &S) -> S {
    let half = T::from_f64(0.5);
    matrix_from(|i, j| (f[i][j] + f[j][i]) * half)
}
