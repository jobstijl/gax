//! Batch drivers: chunking with padded remainders, array-of-structs transposes, and the
//! batch forms of maps and sandwiches.

use super::{Batch, Kernel, LaneElem, Mv, Soa, run, soa_map, soa_map2};
use crate::coef::{Coef, Real};
use crate::kind::{Extensor, Kind, Retype};

/// The most lanes a lane type may have (the remainder buffers are this long).
pub const MAX_LANES: usize = super::BLOCK;

/// The zero value of an extensor type.
#[inline(always)]
fn zero<M: Extensor<Slots = ()>>() -> M {
    M::from_coeffs(<M::Kind as Kind>::arr_from_fn(|_| M::Coef::zero()))
}

/// Transpose the first `L::LANES` values of `xs` into lanes.
///
/// # Panics
/// If `xs` is shorter than `L::LANES`.
#[inline(always)]
pub fn gather<M, L>(xs: &[M]) -> Retype<M, (), L>
where
    M: Extensor<Slots = ()>,
    L: Batch<Elem = M::Coef>,
    M::Coef: LaneElem,
{
    let xs = &xs[..L::LANES];
    Extensor::from_coeffs(<M::Kind as Kind>::arr_from_fn(
        #[inline(always)]
        |i| {
            L::from_fn(
                #[inline(always)]
                |l| xs[l].coeffs().as_ref()[i],
            )
        },
    ))
}

/// Transpose lanes into the first `L::LANES` values of `out`.
///
/// # Panics
/// If `out` is shorter than `L::LANES`.
#[inline(always)]
pub fn scatter<M, L>(v: &Retype<M, (), L>, out: &mut [M])
where
    M: Extensor<Slots = ()>,
    L: Batch<Elem = M::Coef>,
    M::Coef: LaneElem,
{
    // One vector store per coefficient into a buffer, then scalar moves.
    let cols = <M::Kind as Kind>::arr_map(
        v.coeffs(),
        #[inline(always)]
        |x| {
            let mut col = [M::Coef::zero(); MAX_LANES];
            x.store(&mut col);
            col
        },
    );
    let cols = cols.as_ref();
    for (l, o) in out[..L::LANES].iter_mut().enumerate() {
        *o = M::from_coeffs(<M::Kind as Kind>::arr_from_fn(
            #[inline(always)]
            |i| cols[i][l],
        ));
    }
}

/// Every lane set to the value `v`.
#[inline(always)]
pub fn splat<K: Kind, L: Batch>(v: Mv<K, L::Elem>) -> Mv<K, L> {
    Extensor::from_coeffs(K::arr_map(
        v.coeffs(),
        #[inline(always)]
        |x| L::splat(*x),
    ))
}

/// Lanes `f(0), ..., f(m - 1)`, with the last value repeated in the lanes past `m` (so
/// padding lanes compute on valid inputs). The values go through a buffer and one vector
/// load, which is faster than inserting them one by one.
///
/// # Panics
/// If `m == 0`.
#[inline(always)]
pub fn column<L: Batch>(m: usize, f: impl Fn(usize) -> L::Elem) -> L {
    assert!(m > 0 && L::LANES <= MAX_LANES);
    let mut col = [L::Elem::zero(); MAX_LANES];
    for (l, c) in col.iter_mut().enumerate().take(L::LANES) {
        *c = f(l.min(m - 1));
    }
    L::load(&col)
}

/// The lanes as an array (the first `L::LANES` entries), for scattering results.
#[inline(always)]
pub fn to_array<L: Batch>(v: L) -> [L::Elem; MAX_LANES] {
    let mut col = [L::Elem::zero(); MAX_LANES];
    v.store(&mut col);
    col
}

/// Call `f` on consecutive batches of exactly `L::LANES` elements of `xs` and `out`. The
/// last, partial batch goes through buffers padded with `pad_x` (results past the end are
/// dropped), so `f` is inlined once and always sees full batches.
///
/// # Panics
/// If `xs` and `out` differ in length, or `L::LANES > MAX_LANES`.
#[inline(always)]
pub fn chunks<L: Batch, X: Copy, Y: Copy>(
    xs: &[X],
    out: &mut [Y],
    pad_x: X,
    pad_y: Y,
    mut f: impl FnMut(&[X], &mut [Y]),
) {
    assert!(L::LANES <= MAX_LANES && xs.len() == out.len());
    let n = xs.len();
    let (mut tx, mut ty) = ([pad_x; MAX_LANES], [pad_y; MAX_LANES]);
    let mut i = 0;
    while i < n {
        let r = n - i;
        if r >= L::LANES {
            f(&xs[i..i + L::LANES], &mut out[i..i + L::LANES]);
        } else {
            tx[..r].copy_from_slice(&xs[i..]);
            f(&tx[..L::LANES], &mut ty[..L::LANES]);
            out[i..].copy_from_slice(&ty[..r]);
        }
        i += L::LANES;
    }
}

/// [`chunks`] with two inputs.
///
/// # Panics
/// If the slices differ in length, or `L::LANES > MAX_LANES`.
#[inline(always)]
pub fn chunks2<L: Batch, A: Copy, B: Copy, Y: Copy>(
    a: &[A],
    b: &[B],
    out: &mut [Y],
    pads: (A, B, Y),
    mut f: impl FnMut(&[A], &[B], &mut [Y]),
) {
    assert!(L::LANES <= MAX_LANES && a.len() == b.len() && b.len() == out.len());
    let n = a.len();
    let (mut ta, mut tb, mut ty) = (
        [pads.0; MAX_LANES],
        [pads.1; MAX_LANES],
        [pads.2; MAX_LANES],
    );
    let mut i = 0;
    while i < n {
        let r = n - i;
        if r >= L::LANES {
            let j = i + L::LANES;
            f(&a[i..j], &b[i..j], &mut out[i..j]);
        } else {
            ta[..r].copy_from_slice(&a[i..]);
            tb[..r].copy_from_slice(&b[i..]);
            f(&ta[..L::LANES], &tb[..L::LANES], &mut ty[..L::LANES]);
            out[i..].copy_from_slice(&ty[..r]);
        }
        i += L::LANES;
    }
}

/// A map from kind `X` to kind `Y` that is generic over the coefficient type, for [`map`].
///
/// ```
/// use gax::batch::{self, Map};
/// use gax::pga3d::{Line, Motor};
/// use gax::Real;
/// // The exponential of many twists, 8 at a time with vectorized sin and cos.
/// struct Exp;
/// impl Map for Exp {
///     type X = Line;
///     type Y = Motor;
///     #[inline(always)]
///     fn call<T: Real>(&self, b: Line<(), T>) -> Motor<(), T> {
///         b.exp().into_inner()
///     }
/// }
/// let twists: Vec<Line> = (0..20).map(|i| Line::new(0.1, 0.2, 0.3, 0.01 * i as f32, 0.5, 0.2)).collect();
/// let mut motors = vec![Motor::zero(); 20];
/// batch::map(&Exp, &twists, &mut motors);
/// let d = motors[3] - twists[3].exp().into_inner();
/// assert!(d.c.iter().all(|x| x.abs() < 1e-5));
/// ```
pub trait Map {
    /// The input kind.
    type X: Kind;
    /// The output kind.
    type Y: Kind;
    /// The map, for any coefficient type. Mark it `#[inline(always)]`: it is compiled into
    /// the dispatcher's per-level functions only when inlined.
    fn call<T: Real>(&self, x: Mv<Self::X, T>) -> Mv<Self::Y, T>;
}

/// `out[i] = f(xs[i])`, on SIMD lanes of the current [`level`](super::level).
///
/// # Panics
/// If `xs` and `out` differ in length.
#[inline]
pub fn map<F: Map, E: LaneElem>(f: &F, xs: &[Mv<F::X, E>], out: &mut [Mv<F::Y, E>]) {
    struct Run<'a, F: Map, E: LaneElem>(&'a F, &'a [Mv<F::X, E>], &'a mut [Mv<F::Y, E>]);
    impl<F: Map, E: LaneElem> Kernel<E> for Run<'_, F, E> {
        type Output = ();
        #[inline(always)]
        fn run<L: Batch<Elem = E>>(self) {
            let Run(f, xs, out) = self;
            chunks::<L, _, _>(
                xs,
                out,
                zero(),
                zero(),
                #[inline(always)]
                |x, o| {
                    let y: Mv<F::Y, L> = f.call(gather::<Mv<F::X, E>, L>(x));
                    scatter::<Mv<F::Y, E>, L>(&y, o);
                },
            );
        }
    }
    assert_eq!(xs.len(), out.len(), "batch::map: lengths differ");
    run(Run(f, xs, out));
}

/// `out[i] = f(xs[i])` on struct-of-arrays storage (no transposes); `out` is resized to
/// `xs.len()`.
#[inline]
pub fn map_soa<F: Map, E: LaneElem>(f: &F, xs: &Soa<F::X, E>, out: &mut Soa<F::Y, E>) {
    struct Run<'a, F: Map, E: LaneElem>(&'a F, &'a Soa<F::X, E>, &'a mut Soa<F::Y, E>);
    impl<F: Map, E: LaneElem> Kernel<E> for Run<'_, F, E> {
        type Output = ();
        #[inline(always)]
        fn run<L: Batch<Elem = E>>(self) {
            let Run(f, xs, out) = self;
            soa_map::<F::X, F::Y, L>(
                xs,
                out,
                #[inline(always)]
                |x| f.call(x),
            );
        }
    }
    out.resize(xs.len());
    run(Run(f, xs, out));
}

/// Marker for plain versors in [`SandwichKernel`].
#[derive(Clone, Copy, Debug)]
pub enum Plain {}
/// Marker for `Unit` versors in [`SandwichKernel`]: the kernels use the unit condition.
#[derive(Clone, Copy, Debug)]
pub enum Certified {}

/// The sandwich of versor kind `Self` on kind `X` at every coefficient type, for plain
/// (`U = Plain`) or `Unit` (`U = Certified`) versors. Implemented by the generated algebras;
/// [`BatchTransform`] builds the batch kernels on it.
pub trait SandwichKernel<X: Kind, U>: Kind {
    /// The result kind.
    type Y: Kind;
    /// The versor type: `Self<(), T>` or `Unit<Self<(), T>>`.
    type Versor<T: Coef>: Copy;
    /// The prepared action.
    type Prepared<T: Coef>: Copy;
    /// Wrap versor coefficients (certifying them for `Unit` versors).
    fn wrap<T: Coef>(v: Mv<Self, T>) -> Self::Versor<T>;
    /// The versor's coefficients.
    fn unwrap<T: Coef>(v: Self::Versor<T>) -> Mv<Self, T>;
    /// Prepare the action.
    fn prepare<T: Coef>(v: Self::Versor<T>) -> Self::Prepared<T>;
    /// Map the prepared entries to another coefficient type.
    fn map_prepared<T: Coef, W: Coef>(
        p: Self::Prepared<T>,
        f: impl FnMut(T) -> W,
    ) -> Self::Prepared<W>;
    /// Apply a prepared action.
    fn apply_prepared<T: Coef>(p: Self::Prepared<T>, x: Mv<X, T>) -> Mv<Self::Y, T>;
    /// The fused sandwich.
    fn apply<T: Coef>(v: Self::Versor<T>, x: Mv<X, T>) -> Mv<Self::Y, T>;
}

/// The versor kind and certification of a versor type: `M` or `Unit<M>`.
pub trait VersorType {
    /// The versor's kind.
    type Kind: Kind;
    /// [`Plain`] or [`Certified`].
    type Cert;
}
impl<M: Extensor<Slots = ()>> VersorType for M {
    type Kind = M::Kind;
    type Cert = Plain;
}
impl<M: Extensor<Slots = ()>> VersorType for crate::unit::Unit<M> {
    type Kind = M::Kind;
    type Cert = Certified;
}

/// The result kind of versor type `V` acting on kind `X`.
pub type OutOf<V, X> = <<V as VersorType>::Kind as SandwichKernel<X, <V as VersorType>::Cert>>::Y;

/// The result type of versor type `V` acting on a value of type `M`.
pub type OutputOf<V, M> = Mv<OutOf<V, <M as Extensor>::Kind>, <M as Extensor>::Coef>;

/// Batch sandwiches: `v x ~v` for many `x`, or for many pairs, on SIMD lanes.
///
/// Implemented for every versor value type and `Unit` versor; the methods apply wherever the
/// generated algebra has the sandwich, with `f32` or `f64` coefficients.
pub trait BatchTransform: VersorType + Copy {
    /// `out[i] = self >> xs[i]`: the action is prepared once and applied to 8 (`f32`) or
    /// 4 (`f64`) values at a time.
    ///
    /// # Panics
    /// If `xs` and `out` differ in length.
    #[inline]
    fn transform_slice<M>(self, xs: &[M], out: &mut [OutputOf<Self, M>])
    where
        M: Extensor<Slots = ()>,
        M::Coef: LaneElem,
        Self::Kind: SandwichKernel<M::Kind, Self::Cert, Versor<M::Coef> = Self>,
    {
        struct Run<'a, K: SandwichKernel<M::Kind, U>, U, M: Extensor, E: LaneElem>(
            K::Prepared<E>,
            &'a [M],
            &'a mut [Mv<K::Y, E>],
        );
        impl<K, U, M, E> Kernel<E> for Run<'_, K, U, M, E>
        where
            E: LaneElem,
            K: SandwichKernel<M::Kind, U>,
            M: Extensor<Slots = (), Coef = E>,
        {
            type Output = ();
            #[inline(always)]
            fn run<L: Batch<Elem = E>>(self) {
                let Run(p, xs, out) = self;
                let p = K::map_prepared(p, L::splat);
                chunks::<L, _, _>(
                    xs,
                    out,
                    zero(),
                    zero(),
                    #[inline(always)]
                    |x, o| {
                        let x: Mv<M::Kind, L> = gather::<M, L>(x);
                        scatter::<Mv<K::Y, E>, L>(&K::apply_prepared(p, x), o);
                    },
                );
            }
        }
        assert_eq!(xs.len(), out.len(), "transform_slice: lengths differ");
        run(Run::<Self::Kind, Self::Cert, M, M::Coef>(
            Self::Kind::prepare(self),
            xs,
            out,
        ));
    }

    /// `out[i] = vs[i] >> xs[i]` with the fused sandwich kernel.
    ///
    /// # Panics
    /// If the slices differ in length.
    #[inline]
    fn transform_each<M>(vs: &[Self], xs: &[M], out: &mut [OutputOf<Self, M>])
    where
        M: Extensor<Slots = ()>,
        M::Coef: LaneElem,
        Self::Kind: SandwichKernel<M::Kind, Self::Cert, Versor<M::Coef> = Self>,
    {
        struct Run<'a, K: SandwichKernel<M::Kind, U>, U, M: Extensor, E: LaneElem>(
            &'a [K::Versor<E>],
            &'a [M],
            &'a mut [Mv<K::Y, E>],
        );
        impl<K, U, M, E> Kernel<E> for Run<'_, K, U, M, E>
        where
            E: LaneElem,
            K: SandwichKernel<M::Kind, U>,
            M: Extensor<Slots = (), Coef = E>,
        {
            type Output = ();
            #[inline(always)]
            fn run<L: Batch<Elem = E>>(self) {
                let Run(vs, xs, out) = self;
                let pads = (K::wrap(zero()), zero(), zero());
                chunks2::<L, _, _, _>(
                    vs,
                    xs,
                    out,
                    pads,
                    #[inline(always)]
                    |v, x, o| {
                        let mut vv = [zero::<Mv<K, E>>(); MAX_LANES];
                        for (d, x) in vv.iter_mut().zip(v) {
                            *d = K::unwrap(*x);
                        }
                        let v = K::wrap(gather::<Mv<K, E>, L>(&vv));
                        let x: Mv<M::Kind, L> = gather::<M, L>(x);
                        scatter::<Mv<K::Y, E>, L>(&K::apply(v, x), o);
                    },
                );
            }
        }
        assert!(
            vs.len() == xs.len() && xs.len() == out.len(),
            "transform_each: lengths differ"
        );
        run(Run::<Self::Kind, Self::Cert, M, M::Coef>(vs, xs, out));
    }

    /// [`transform_slice`](Self::transform_slice) on struct-of-arrays storage; `out` is
    /// resized to `xs.len()`.
    #[inline]
    fn transform_soa<X: Kind, E: LaneElem>(self, xs: &Soa<X, E>, out: &mut Soa<OutOf<Self, X>, E>)
    where
        Self::Kind: SandwichKernel<X, Self::Cert, Versor<E> = Self>,
    {
        struct Run<'a, K: SandwichKernel<X, U>, X: Kind, U, E: LaneElem>(
            K::Prepared<E>,
            &'a Soa<X, E>,
            &'a mut Soa<K::Y, E>,
        );
        impl<K: SandwichKernel<X, U>, X: Kind, U, E: LaneElem> Kernel<E> for Run<'_, K, X, U, E> {
            type Output = ();
            #[inline(always)]
            fn run<L: Batch<Elem = E>>(self) {
                let Run(p, xs, out) = self;
                let p = K::map_prepared(p, L::splat);
                soa_map::<X, K::Y, L>(
                    xs,
                    out,
                    #[inline(always)]
                    |x| K::apply_prepared(p, x),
                );
            }
        }
        out.resize(xs.len());
        run(Run::<Self::Kind, X, Self::Cert, E>(
            Self::Kind::prepare(self),
            xs,
            out,
        ));
    }

    /// [`transform_each`](Self::transform_each) on struct-of-arrays storage: `vs` holds the
    /// versors' coefficients (certified as unit versors when `Self` is a `Unit`); `out` is
    /// resized to `xs.len()`.
    ///
    /// # Panics
    /// If `vs` and `xs` differ in length.
    #[inline]
    fn transform_each_soa<X: Kind, E: LaneElem>(
        vs: &Soa<Self::Kind, E>,
        xs: &Soa<X, E>,
        out: &mut Soa<OutOf<Self, X>, E>,
    ) where
        Self::Kind: SandwichKernel<X, Self::Cert, Versor<E> = Self>,
    {
        struct Run<'a, K: SandwichKernel<X, U>, X: Kind, U, E: LaneElem>(
            &'a Soa<K, E>,
            &'a Soa<X, E>,
            &'a mut Soa<K::Y, E>,
        );
        impl<K: SandwichKernel<X, U>, X: Kind, U, E: LaneElem> Kernel<E> for Run<'_, K, X, U, E> {
            type Output = ();
            #[inline(always)]
            fn run<L: Batch<Elem = E>>(self) {
                let Run(vs, xs, out) = self;
                soa_map2::<K, X, K::Y, L>(
                    vs,
                    xs,
                    out,
                    #[inline(always)]
                    |v, x| K::apply(K::wrap(v), x),
                );
            }
        }
        assert_eq!(vs.len(), xs.len(), "transform_each_soa: lengths differ");
        out.resize(xs.len());
        run(Run::<Self::Kind, X, Self::Cert, E>(vs, xs, out));
    }
}

impl<V: VersorType + Copy> BatchTransform for V {}
