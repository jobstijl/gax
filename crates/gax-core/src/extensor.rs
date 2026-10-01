//! Methods of maps and forms: inverse, solve, determinant, trace, SVD, eigenproblems, least
//! squares and the pseudo-inverse.
//!
//! These are blanket traits over every generated kind, so they work for any algebra. The
//! generated types expose them as inherent methods (`map.inverse()`, `form.eigh_with(m)`),
//! which forward here. Results are typed: the inverse of a `B <- A` map is an `A <- B` map,
//! singular vectors and eigenvectors are values of the slot kinds.

use crate::cast::SubKind;
use crate::coef::{Coef, Real};
use crate::fill::SplitLast;
use crate::kind::{Extensor, Kind};
use crate::linalg::{self, Column, SquareArr};
use crate::slots::{Slots, SplitFirst};

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

/// Contract an extensor's output with its first slot, which must be the output's own kind
/// (numga's `trace(slot)`): a blade-matching contraction with no metric. The result is a
/// scalar with the remaining slots. Use `m.trace_at::<I>()` to trace slot `I`.
pub trait TraceFirst: Extensor {
    /// The result: a scalar extensor with the remaining slots.
    type Output;
    /// `Σ_k m[k][k][rest]`.
    fn trace_first(self) -> Self::Output;
}

impl<M> TraceFirst for M
where
    M: Extensor,
    M::Slots: crate::slots::SplitFirst<Head = M::Kind>,
{
    type Output = <<M::Kind as Kind>::Scalar as Kind>::Mv<
        <M::Slots as crate::slots::SplitFirst>::Tail,
        M::Coef,
    >;
    fn trace_first(self) -> Self::Output {
        use crate::slots::{SlotArr, SplitFirst};
        type Tail<M> = <<M as Extensor>::Slots as SplitFirst>::Tail;
        let c = self.coeffs().as_ref();
        let term = |k: usize| {
            SlotArr::<Tail<M>, M::Coef>(<M::Slots as SplitFirst>::split(&c[k]).as_ref()[k])
        };
        let mut acc = term(0);
        for k in 1..c.len() {
            acc = acc + term(k);
        }
        let v = acc.0;
        <Self::Output as Extensor>::from_coeffs(<<M::Kind as Kind>::Scalar as Kind>::arr_from_fn(
            |_| v,
        ))
    }
}

/// One column of a least-squares problem in an extensor's layout: an output coefficient array
/// over the slots `S` (the image of one input blade).
pub struct Col<K: Kind, S: Slots, T: Coef>(pub K::Arr<S::Arr<T>>);

#[allow(clippy::expl_impl_clone_on_copy)]
impl<K: Kind, S: Slots, T: Coef> Clone for Col<K, S, T> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

impl<K: Kind, S: Slots, T: Coef> Copy for Col<K, S, T> {}

impl<K: Kind, S: Slots, T: Coef> PartialEq for Col<K, S, T> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<K: Kind, S: Slots, T: Coef> core::fmt::Debug for Col<K, S, T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.0.fmt(f)
    }
}

impl<K: Kind, S: Slots, T: Real> Column<T> for Col<K, S, T> {
    #[inline(always)]
    fn dot(&self, other: &Self) -> T {
        let (a, b) = (self.0.as_ref(), other.0.as_ref());
        let mut s = T::zero();
        for k in 0..K::N {
            for f in 0..S::SIZE {
                s = S::get_flat(&a[k], f).mul_add(S::get_flat(&b[k], f), s);
            }
        }
        s
    }

    #[inline(always)]
    fn rotate(a: &mut Self, b: &mut Self, c: T, s: T) {
        let na = K::arr_zip(&a.0, &b.0, |x, y| S::zip(x, y, &mut |p, q| c * *p - s * *q));
        let nb = K::arr_zip(&a.0, &b.0, |x, y| S::zip(x, y, &mut |p, q| s * *p + c * *q));
        a.0 = na;
        b.0 = nb;
    }
}

/// `H × H` coefficient matrices.
type Sq<H, T> = <H as Kind>::Arr<<H as Kind>::Arr<T>>;

/// The least-squares factorization of a map's first slot `H` (columns: the images of `H`'s
/// blades, each a `K` over the remaining slots `R`): the rotated columns, `V`, and the weights
/// `1/σᵢ²` above the cutoff.
#[allow(clippy::type_complexity)]
#[inline]
fn factor<K, H, R, S, T>(
    m: &K::Arr<S::Arr<T>>,
    rcond: Option<T>,
) -> (
    H::Arr<Col<K, R, T>>,
    Sq<H, T>,
    <Sq<H, T> as SquareArr<T>>::Vector,
)
where
    K: Kind,
    H: Kind,
    R: Slots,
    S: SplitFirst<Head = H, Tail = R>,
    T: Real,
    Sq<H, T>: SquareArr<T>,
{
    let rows = m.as_ref();
    let cols: H::Arr<Col<K, R, T>> =
        H::arr_from_fn(|h| Col(K::arr_from_fn(|k| S::split(&rows[k]).as_ref()[h])));
    let n = H::N;
    let (w, v): (H::Arr<Col<K, R, T>>, Sq<H, T>) = linalg::orthogonalize(&cols, sweeps(n));
    // NumPy's default cutoff for `lstsq`: machine epsilon times the larger dimension.
    let size = (K::N * R::SIZE).max(n);
    let cut = rcond.unwrap_or_else(|| T::epsilon() * T::from_i64(size as i64));
    let weights = linalg::pinv_weights::<T, Col<K, R, T>, _, Sq<H, T>>(&w, cut);
    (w, v, weights)
}

/// The Moore–Penrose pseudo-inverse of a map `K <- (H,)` of any shape, by one-sided Jacobi:
/// the map `H <- (K,)` that sends `b` to the minimum-norm least-squares solution of
/// `self.of(x) ≈ b`. Norms are those of the coefficients (as in `numpy.linalg` and numga), not the
/// algebra's metric.
///
/// ```
/// use gax::pga3d::{Line, Point};
/// use gax::ApproxEq;
/// // Points to the lines joining them to `q`: rank 3 of 4 (`q` itself maps to zero).
/// let q = Point::<(), f64>::xyz(0.0, 0.0, 1.0);
/// let through_q: Line<(Point,), f64> = q & Point::slot();
/// let back: Point<(Line,), f64> = through_q.pinv();
/// let l = through_q.of(Point::xyz(1.0, 2.0, 3.0));
/// // The least-norm point whose line through `q` is `l`.
/// let x = back.of(l);
/// assert!(through_q.of(x).approx_eq(&l, 1e-12));
/// ```
pub trait PseudoInverse: Extensor<Coef: Real> {
    /// The pseudo-inverse map, from the output kind back to the slot's.
    type Output: Extensor;
    /// With the default cutoff of `numpy.linalg.lstsq`: singular values at most `ε · max(rows, columns)` times the
    /// largest are treated as zero.
    fn pinv(self) -> Self::Output;
    /// With the cutoff `rcond` relative to the largest singular value.
    fn pinv_with(self, rcond: Self::Coef) -> Self::Output;
}

impl<M, H> PseudoInverse for M
where
    M: Extensor<Slots = (H,), Coef: Real>,
    H: Kind,
    Sq<H, M::Coef>: SquareArr<M::Coef>,
{
    type Output = H::Mv<(M::Kind,), M::Coef>;

    fn pinv(self) -> Self::Output {
        pinv_of(&self, None)
    }

    fn pinv_with(self, rcond: M::Coef) -> Self::Output {
        pinv_of(&self, Some(rcond))
    }
}

/// `P[h][k] = Σᵢ v[i][h] w_i[k] / σᵢ²`.
fn pinv_of<M, H>(m: &M, rcond: Option<M::Coef>) -> H::Mv<(M::Kind,), M::Coef>
where
    M: Extensor<Slots = (H,), Coef: Real>,
    H: Kind,
    Sq<H, M::Coef>: SquareArr<M::Coef>,
{
    M::Coef::vectorize(
        #[inline(always)]
        || {
            let (w, v, weights) = factor::<M::Kind, H, (), (H,), M::Coef>(m.coeffs(), rcond);
            let cols = w.as_ref();
            let p = H::arr_from_fn(|h| {
                <M::Kind as Kind>::arr_from_fn(|k| {
                    let mut s = M::Coef::zero();
                    for (i, col) in cols.iter().enumerate() {
                        s = (v[i][h] * weights[i]).mul_add(col.0.as_ref()[k], s);
                    }
                    s
                })
            });
            <H::Mv<(M::Kind,), M::Coef> as Extensor>::from_coeffs(p)
        },
    )
}

/// Least squares: the `x` minimizing `‖self.of(x) − rhs‖` with the least norm (coefficient
/// norms, as in `numpy.linalg` and numga).
///
/// * For a map `K <- (H,)` of any shape, `rhs` is a `K` (or a smaller kind, embedded) with any
///   slots, which the solution keeps: `m.lstsq(m.of(y)) == y` for a map `y` of full rank too.
/// * For a map with more slots, `K <- (H, R…)`, `rhs` has exactly the remaining slots `R…`, and
///   `x` is the `H` with `self.of(x) ≈ rhs` as maps on `R…`: the least-squares version of a
///   pairing's `solve` (`m.at::<I>()` solves for another slot).
///
/// ```
/// use gax::pga3d::{Line, Plane, Point, Scalar};
/// use gax::ApproxEq;
/// // Planes meeting the floor `z = 0` in a given line: `x ^ floor ≈ l`, 6 equations for 4
/// // unknowns, with the floor itself in the kernel. The least-norm answer is the plane
/// // through `l` orthogonal to the floor.
/// let floor = Plane::<(), f64>::new(0.0, 0.0, 1.0, 0.0);
/// let meet: Line<(Plane,), f64> = Plane::slot() ^ floor;
/// let l = Plane::new(1.0, 0.0, 0.0, -2.0) ^ floor; // the line x = 2 on the floor
/// let x: Plane<(), f64> = meet.lstsq(l);
/// assert!((x ^ floor).approx_eq(&l, 1e-12));
/// assert!(x.approx_eq(&Plane::new(1.0, 0.0, 0.0, -2.0), 1e-12));
///
/// // Two slots: the plane `x` with `x & p == r(p)` for every point `p`.
/// let pairing: Scalar<(Plane, Point), f64> = Plane::slot() & Point::slot();
/// let target = Plane::new(0.0, 1.0, 0.0, 3.0);
/// let found: Plane<(), f64> = pairing.lstsq(target & Point::slot());
/// assert!(found.approx_eq(&target, 1e-12));
/// ```
pub trait LeastSquares<X>: Extensor<Coef: Real> {
    /// The solution: a value of the first slot's kind, with `rhs`'s slots for a one-slot map.
    type Solution: Extensor;
    /// With the default cutoff (see [`PseudoInverse::pinv`]).
    fn lstsq(self, rhs: X) -> Self::Solution;
    /// With the cutoff `rcond` relative to the largest singular value.
    fn lstsq_with(self, rhs: X, rcond: Self::Coef) -> Self::Solution;
}

/// The dispatch behind [`LeastSquares`] on the slot list `(H, R…)` of a map with output kind
/// `K`: one slot takes a right-hand side with any slots, more slots one with exactly `R…`.
pub trait LstsqSlots<K: Kind, T: Real, X>: Slots {
    /// See [`LeastSquares::Solution`].
    type Solution: Extensor;
    /// See [`LeastSquares::lstsq_with`]; `None` for the default cutoff.
    fn lstsq(m: &K::Arr<Self::Arr<T>>, rhs: &X, rcond: Option<T>) -> Self::Solution;
}

impl<M, X> LeastSquares<X> for M
where
    M: Extensor<Coef: Real>,
    M::Slots: LstsqSlots<M::Kind, M::Coef, X>,
{
    type Solution = <M::Slots as LstsqSlots<M::Kind, M::Coef, X>>::Solution;

    fn lstsq(self, rhs: X) -> Self::Solution {
        <M::Slots as LstsqSlots<M::Kind, M::Coef, X>>::lstsq(self.coeffs(), &rhs, None)
    }

    fn lstsq_with(self, rhs: X, rcond: M::Coef) -> Self::Solution {
        <M::Slots as LstsqSlots<M::Kind, M::Coef, X>>::lstsq(self.coeffs(), &rhs, Some(rcond))
    }
}

impl<K, H, T, X> LstsqSlots<K, T, X> for (H,)
where
    K: Kind,
    H: Kind,
    T: Real,
    X: Extensor<Coef = T>,
    X::Kind: SubKind<K>,
    Sq<H, T>: SquareArr<T>,
{
    type Solution = H::Mv<X::Slots, T>;

    fn lstsq(m: &K::Arr<<(H,) as Slots>::Arr<T>>, rhs: &X, rcond: Option<T>) -> Self::Solution {
        T::vectorize(
            #[inline(always)]
            || {
                let p: H::Mv<(K,), T> = pinv_of::<K::Mv<(H,), T>, H>(
                    &<K::Mv<(H,), T> as Extensor>::from_coeffs(*m),
                    rcond,
                );
                let b = crate::cast::cast::<X, K>(rhs);
                let (p, b) = (p.coeffs().as_ref(), b.coeffs().as_ref());
                let x = H::arr_from_fn(|h| {
                    <X::Slots as Slots>::from_flat(
                        &mut |f| {
                            let mut s = T::zero();
                            for (k, bk) in b.iter().enumerate() {
                                s = p[h].as_ref()[k]
                                    .mul_add(<X::Slots as Slots>::get_flat(bk, f), s);
                            }
                            s
                        },
                        0,
                    )
                });
                <H::Mv<X::Slots, T> as Extensor>::from_coeffs(x)
            },
        )
    }
}

macro_rules! lstsq_slots {
    ($($R:ident),+) => {
        impl<K, H, $($R,)+ T, X> LstsqSlots<K, T, X> for (H, $($R,)+)
        where
            K: Kind,
            H: Kind,
            $($R: Kind,)+
            T: Real,
            X: Extensor<Coef = T, Slots = ($($R,)+)>,
            X::Kind: SubKind<K>,
            Sq<H, T>: SquareArr<T>,
        {
            type Solution = H::Mv<(), T>;

            fn lstsq(m: &K::Arr<<Self as Slots>::Arr<T>>, rhs: &X, rcond: Option<T>) -> Self::Solution {
                T::vectorize(
                    #[inline(always)]
                    || {
                        let (w, v, weights) = factor::<K, H, ($($R,)+), Self, T>(m, rcond);
                        let b = crate::cast::cast::<X, K>(rhs);
                        let x = linalg::pinv_apply(&w, &v, &weights, &Col::<K, ($($R,)+), T>(*b.coeffs()));
                        <H::Mv<(), T> as Extensor>::from_coeffs(H::arr_from_fn(|h| x[h]))
                    },
                )
            }
        }
    };
}

lstsq_slots!(A1);
lstsq_slots!(A1, A2);
lstsq_slots!(A1, A2, A3);
lstsq_slots!(A1, A2, A3, A4);
lstsq_slots!(A1, A2, A3, A4, A5);
lstsq_slots!(A1, A2, A3, A4, A5, A6);
lstsq_slots!(A1, A2, A3, A4, A5, A6, A7);
lstsq_slots!(A1, A2, A3, A4, A5, A6, A7, A8);
lstsq_slots!(A1, A2, A3, A4, A5, A6, A7, A8, A9);
lstsq_slots!(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10);
lstsq_slots!(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11);
