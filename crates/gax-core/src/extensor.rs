//! Methods of maps and forms: inverse, solve, determinant, trace, SVD, eigenproblems.
//!
//! These are blanket traits over every generated kind, so they work for any algebra. The
//! generated types expose them as inherent methods (`map.inverse()`, `form.eigh_with(m)`),
//! which forward here. Results are typed: the inverse of a `B <- A` map is an `A <- B` map,
//! singular vectors and eigenvectors are values of the slot kinds.

use crate::coef::{Coef, Real};
use crate::kind::{Extensor, Kind};
use crate::linalg::{self, SquareArr};
use crate::slots::Slots;

/// A map `B <- A` between kinds with the same number of coefficients.
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

    /// Solve `self(x, ·) = linear(·)` for `x`, where `linear` is a linear form on the slot.
    fn solve<L>(self, linear: L) -> <Self::Slot as Kind>::Mv<(), Self::Coef>
    where
        L: Extensor<Kind = Self::Kind, Slots = (Self::Slot,), Coef = Self::Coef>;
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

    fn solve<L>(self, linear: L) -> A::Mv<(), M::Coef>
    where
        L: Extensor<Kind = M::Kind, Slots = (A,), Coef = M::Coef>,
    {
        // form(x, y) = Σ_ij F[i][j] x_i y_j, so form(x, ·) = linear(·) is Fᵀ x = l.
        let f = &self.coeffs().as_ref()[0];
        let ft: A::Arr<A::Arr<M::Coef>> = linalg::transpose(f);
        let l = &linear.coeffs().as_ref()[0];
        let mut b = <A::Arr<A::Arr<M::Coef>> as SquareArr<M::Coef>>::zero_vector();
        for i in 0..A::N {
            b[i] = <(A,) as Slots>::get_flat(l, i);
        }
        let x = linalg::lu_solve(&linalg::lu(&ft), &b);
        value_from::<A, M::Coef>(|i| x[i])
    }
}

/// `(F + Fᵀ) / 2`: a form's symmetric part, which is all an eigenproblem sees.
fn symmetrize<T: Real, S: SquareArr<T>>(f: &S) -> S {
    let half = T::from_f64(0.5);
    matrix_from(|i, j| (f[i][j] + f[j][i]) * half)
}
