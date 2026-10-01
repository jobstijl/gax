//! Slot lists: the open arguments of an extensor, as a tuple of kinds.
//!
//! `()` is the empty list (a value), `(Point,)` one open point (a linear map), and
//! `(Twist, Twist)` two open twists (a bilinear form). Coefficients are stored as nested
//! arrays, one level per slot: `(A, B)::Arr<X> = A::Arr<B::Arr<X>>`.
//!
//! The concatenation of two lists is the associated type [`HasCat::Cat`], and [`Slots`]
//! requires the law `Cat<S, ()> = S` as a supertrait equality. The compiler checks the law for
//! every tuple length, and generic code gets it for free: with only `S: Slots`, an expression
//! combining an `S`-slotted value with a plain value on either side has slots `S`.

use crate::coef::{Coef, Elem};
use crate::kind::{Extensor, Kind, Retype};
use core::ops::{Add, Mul, Neg, Sub};

/// The most open slots an extensor can have.
pub const MAX_SLOTS: usize = 12;

/// Concatenation of slot lists. Implemented for every [`Slots`] type.
pub trait HasCat {
    /// `Self` followed by `R`.
    type Cat<R: Slots>: Slots;
}

/// The slot list `A` followed by `B`.
pub type Cat<A, B> = <A as HasCat>::Cat<B>;

/// A list of open slots. Implemented for `()` and tuples of up to 12 kinds.
///
/// A function generic over `S: Slots` works on values (`S = ()`) and on maps and forms alike:
///
/// ```
/// use gax::pga3d::{Plane, Point};
/// use gax::Slots;
///
/// fn shadow<S: Slots>(light: Point<(), f64>, ground: Plane<(), f64>, p: Point<S, f64>) -> Point<S, f64> {
///     (light & p) ^ ground // `Cat<(), S> = S` and `Cat<S, ()> = S` for any S
/// }
/// # let (l, g) = (Point::xyz(0.0, 0.0, 9.0), Plane::from_normal([0.0, 0.0, 1.0], 0.0));
/// # let _ = shadow(l, g, Point::slot());
/// ```
pub trait Slots: HasCat<Cat<()> = Self> + Copy + 'static {
    /// Number of slots.
    const LEN: usize;
    /// Number of coefficients per output coefficient: the product of the slot dimensions.
    const SIZE: usize;
    /// The list with `H` in front.
    type Prepend<H: Kind>: Slots;
    /// Nested coefficient array over all slots, first slot outermost.
    type Arr<X: Elem>: Elem;

    /// Identity witness: `Prepend<H>::Arr<X>` is `H::Arr<Self::Arr<X>>`.
    fn prepend_arr<H: Kind, X: Elem>(
        a: H::Arr<Self::Arr<X>>,
    ) -> <Self::Prepend<H> as Slots>::Arr<X>;
    /// Build an array from its flat (row-major) index.
    fn from_flat<X: Elem>(f: &mut impl FnMut(usize) -> X, base: usize) -> Self::Arr<X>;
    /// Read the element at a flat index.
    fn get_flat<X: Elem>(a: &Self::Arr<X>, flat: usize) -> X;
    /// Elementwise map.
    fn map<X: Elem, Y: Elem>(a: &Self::Arr<X>, f: &mut impl FnMut(&X) -> Y) -> Self::Arr<Y>;
    /// Elementwise combination of two arrays.
    fn zip<X: Elem, Y: Elem, Z: Elem>(
        a: &Self::Arr<X>,
        b: &Self::Arr<Y>,
        f: &mut impl FnMut(&X, &Y) -> Z,
    ) -> Self::Arr<Z>;
    /// Outer product: every element of `a` combined with every element of `b`.
    fn outer<R: Slots, A: Elem, B: Elem, C: Elem>(
        a: &Self::Arr<A>,
        b: &R::Arr<B>,
        f: &mut impl FnMut(&A, &B) -> C,
    ) -> <Self::Cat<R> as Slots>::Arr<C>;
    /// `Some` for the empty list: the array is a single element. Folds at compile time.
    fn as_value<X: Elem>(a: &Self::Arr<X>) -> Option<X>;
    /// `Some` for the empty list. Folds at compile time.
    fn from_value<X: Elem>(x: X) -> Option<Self::Arr<X>>;

    /// Associativity witness: an array over `Cat<Cat<Self, B>, C>` as an array over
    /// `Cat<Self, Cat<B, C>>`. The two lists are the same slots in the same order, so this is
    /// the identity on the coefficients; it exists because the compiler cannot prove the two
    /// types equal for generic lists. See [`reassoc`].
    #[inline(always)]
    fn reassoc<B: Slots, C: Slots, X: Elem>(
        a: &<Cat<Cat<Self, B>, C> as Slots>::Arr<X>,
    ) -> <Cat<Self, Cat<B, C>> as Slots>::Arr<X> {
        <Cat<Self, Cat<B, C>> as Slots>::from_flat(
            &mut |i| <Cat<Cat<Self, B>, C> as Slots>::get_flat(a, i),
            0,
        )
    }
}

/// Rebracket the slots of `m` from `Cat<Cat<A, B>, C>` to `Cat<A, Cat<B, C>>`: the identity on
/// the coefficients, for generic code whose signature brackets differently from the
/// expression that computes it.
///
/// ```
/// use gax::pga3d::Point;
/// use gax::{Cat, Slots};
///
/// // `(a * b) * c` has slots `Cat<Cat<A, B>, C>`; the signature says `Cat<A, Cat<B, C>>`.
/// fn triple<A: Slots, B: Slots, C: Slots>(
///     a: Point<A, f64>,
///     b: Point<B, f64>,
///     c: Point<C, f64>,
/// ) -> Point<Cat<A, Cat<B, C>>, f64> {
///     gax::slots::reassoc::<A, B, C, _>((a * b) * c)
/// }
/// let (p, q, r) = (Point::xyz(1.0, 0.0, 0.0), Point::xyz(0.0, 1.0, 0.0), Point::xyz(0.0, 0.0, 1.0));
/// assert_eq!(triple(p, q, r), (p * q) * r);
/// ```
#[inline(always)]
pub fn reassoc<A: Slots, B: Slots, C: Slots, M>(m: M) -> Retype<M, Cat<A, Cat<B, C>>, M::Coef>
where
    M: Extensor<Slots = Cat<Cat<A, B>, C>>,
{
    Extensor::from_coeffs(<M::Kind as Kind>::arr_map(m.coeffs(), |c| {
        A::reassoc::<B, C, M::Coef>(c)
    }))
}

impl HasCat for () {
    type Cat<R: Slots> = R;
}

impl Slots for () {
    const LEN: usize = 0;
    const SIZE: usize = 1;
    type Prepend<H: Kind> = (H,);
    type Arr<X: Elem> = X;

    #[inline(always)]
    fn prepend_arr<H: Kind, X: Elem>(a: H::Arr<X>) -> H::Arr<X> {
        a
    }
    #[inline(always)]
    fn from_flat<X: Elem>(f: &mut impl FnMut(usize) -> X, base: usize) -> X {
        f(base)
    }
    #[inline(always)]
    fn get_flat<X: Elem>(a: &X, _flat: usize) -> X {
        *a
    }
    #[inline(always)]
    fn map<X: Elem, Y: Elem>(a: &X, f: &mut impl FnMut(&X) -> Y) -> Y {
        f(a)
    }
    #[inline(always)]
    fn zip<X: Elem, Y: Elem, Z: Elem>(a: &X, b: &Y, f: &mut impl FnMut(&X, &Y) -> Z) -> Z {
        f(a, b)
    }
    #[inline(always)]
    fn outer<R: Slots, A: Elem, B: Elem, C: Elem>(
        a: &A,
        b: &R::Arr<B>,
        f: &mut impl FnMut(&A, &B) -> C,
    ) -> R::Arr<C> {
        R::map(b, &mut |b| f(a, b))
    }
    #[inline(always)]
    fn as_value<X: Elem>(a: &X) -> Option<X> {
        Some(*a)
    }
    #[inline(always)]
    fn from_value<X: Elem>(x: X) -> Option<X> {
        Some(x)
    }
}

/// Sentinel for a concatenation longer than 12 slots. Using it is a compile-time error.
#[derive(Clone, Copy, Debug)]
pub struct TooManySlots;

impl HasCat for TooManySlots {
    type Cat<R: Slots> = TooManySlots;
}

impl Slots for TooManySlots {
    const LEN: usize = usize::MAX;
    const SIZE: usize = 0;
    type Prepend<H: Kind> = TooManySlots;
    type Arr<X: Elem> = ();

    fn prepend_arr<H: Kind, X: Elem>(_: H::Arr<()>) {
        const { panic!("gax: an extensor can have at most 12 open slots") }
    }
    fn from_flat<X: Elem>(_: &mut impl FnMut(usize) -> X, _: usize) {}
    fn get_flat<X: Elem>((): &(), _: usize) -> X {
        unreachable!()
    }
    fn map<X: Elem, Y: Elem>((): &(), _: &mut impl FnMut(&X) -> Y) {}
    fn zip<X: Elem, Y: Elem, Z: Elem>((): &(), (): &(), _: &mut impl FnMut(&X, &Y) -> Z) {}
    fn outer<R: Slots, A: Elem, B: Elem, C: Elem>(
        (): &(),
        _: &R::Arr<B>,
        _: &mut impl FnMut(&A, &B) -> C,
    ) {
    }
    fn as_value<X: Elem>((): &()) -> Option<X> {
        None
    }
    fn from_value<X: Elem>(_: X) -> Option<()> {
        None
    }
}

/// Non-empty slot lists: access to the first slot.
pub trait SplitFirst: Slots {
    /// The first slot's kind.
    type Head: Kind;
    /// The remaining slots.
    type Tail: Slots;
    /// Identity witness: `Self::Arr<X>` is `Head::Arr<Tail::Arr<X>>`.
    fn split<X: Elem>(a: &Self::Arr<X>)
    -> <Self::Head as Kind>::Arr<<Self::Tail as Slots>::Arr<X>>;
    /// Inverse of [`SplitFirst::split`].
    fn join<X: Elem>(a: <Self::Head as Kind>::Arr<<Self::Tail as Slots>::Arr<X>>) -> Self::Arr<X>;
}

macro_rules! tuple_slots {
    ($len:expr; $H:ident $(, $T:ident)*; $($overflow:ident)?) => {
        impl<$H: Kind $(, $T: Kind)*> HasCat for ($H, $($T,)*) {
            type Cat<R: Slots> = <<($($T,)*) as HasCat>::Cat<R> as Slots>::Prepend<$H>;
        }

        impl<$H: Kind $(, $T: Kind)*> Slots for ($H, $($T,)*) {
            const LEN: usize = $len;
            const SIZE: usize = $H::N * <($($T,)*) as Slots>::SIZE;
            type Prepend<H0: Kind> = tuple_slots!(@prepend H0, $H $(, $T)*; $($overflow)?);
            type Arr<X: Elem> = $H::Arr<<($($T,)*) as Slots>::Arr<X>>;

            #[inline(always)]
            fn prepend_arr<H0: Kind, X: Elem>(
                a: H0::Arr<Self::Arr<X>>,
            ) -> <Self::Prepend<H0> as Slots>::Arr<X> {
                tuple_slots!(@witness a; $($overflow)?)
            }
            #[inline(always)]
            fn from_flat<X: Elem>(f: &mut impl FnMut(usize) -> X, base: usize) -> Self::Arr<X> {
                $H::arr_from_fn(|i| <($($T,)*) as Slots>::from_flat(f, (base * $H::N + i)))
            }
            #[inline(always)]
            fn get_flat<X: Elem>(a: &Self::Arr<X>, flat: usize) -> X {
                let inner = <($($T,)*) as Slots>::SIZE;
                <($($T,)*) as Slots>::get_flat(&a.as_ref()[flat / inner], flat % inner)
            }
            #[inline(always)]
            fn map<X: Elem, Y: Elem>(a: &Self::Arr<X>, f: &mut impl FnMut(&X) -> Y) -> Self::Arr<Y> {
                $H::arr_map(a, |t| <($($T,)*) as Slots>::map(t, f))
            }
            #[inline(always)]
            fn zip<X: Elem, Y: Elem, Z: Elem>(
                a: &Self::Arr<X>,
                b: &Self::Arr<Y>,
                f: &mut impl FnMut(&X, &Y) -> Z,
            ) -> Self::Arr<Z> {
                $H::arr_zip(a, b, |x, y| <($($T,)*) as Slots>::zip(x, y, f))
            }
            #[inline(always)]
            fn outer<R: Slots, A: Elem, B: Elem, C: Elem>(
                a: &Self::Arr<A>,
                b: &R::Arr<B>,
                f: &mut impl FnMut(&A, &B) -> C,
            ) -> <Self::Cat<R> as Slots>::Arr<C> {
                let inner = $H::arr_map(a, |t| <($($T,)*) as Slots>::outer::<R, A, B, C>(t, b, f));
                <<($($T,)*) as HasCat>::Cat<R> as Slots>::prepend_arr::<$H, C>(inner)
            }
            #[inline(always)]
            fn as_value<X: Elem>(_: &Self::Arr<X>) -> Option<X> {
                None
            }
            #[inline(always)]
            fn from_value<X: Elem>(_: X) -> Option<Self::Arr<X>> {
                None
            }
        }

        impl<$H: Kind $(, $T: Kind)*> SplitFirst for ($H, $($T,)*) {
            type Head = $H;
            type Tail = ($($T,)*);
            #[inline(always)]
            fn split<X: Elem>(a: &Self::Arr<X>) -> $H::Arr<<($($T,)*) as Slots>::Arr<X>> {
                *a
            }
            #[inline(always)]
            fn join<X: Elem>(a: $H::Arr<<($($T,)*) as Slots>::Arr<X>>) -> Self::Arr<X> {
                a
            }
        }
    };
    (@prepend $H0:ident, $($T:ident),*; ) => { ($H0, $($T,)*) };
    (@prepend $H0:ident, $($T:ident),*; $overflow:ident) => { TooManySlots };
    (@witness $a:ident; ) => { $a };
    (@witness $a:ident; $overflow:ident) => {{
        let _ = $a;
        const { panic!("gax: an extensor can have at most 12 open slots") }
    }};
}

tuple_slots!(1; A0; );
tuple_slots!(2; A0, A1; );
tuple_slots!(3; A0, A1, A2; );
tuple_slots!(4; A0, A1, A2, A3; );
tuple_slots!(5; A0, A1, A2, A3, A4; );
tuple_slots!(6; A0, A1, A2, A3, A4, A5; );
tuple_slots!(7; A0, A1, A2, A3, A4, A5, A6; );
tuple_slots!(8; A0, A1, A2, A3, A4, A5, A6, A7; );
tuple_slots!(9; A0, A1, A2, A3, A4, A5, A6, A7, A8; );
tuple_slots!(10; A0, A1, A2, A3, A4, A5, A6, A7, A8, A9; );
tuple_slots!(11; A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10; );
tuple_slots!(12; A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11; overflow);

/// One output coefficient's array over the slots `S`, with arithmetic.
///
/// (`Clone` is written by hand because `derive` would require `S: Clone` and `T: Clone`.)
///
/// Generated kernels are written in terms of this wrapper: `+`, `-` and unary `-` act
/// elementwise, and `*` between two wrappers is the outer product, whose slots are the
/// concatenation. For values (`S = ()`) every operation is a single scalar operation.
#[repr(transparent)]
pub struct SlotArr<S: Slots, T: Coef>(pub S::Arr<T>);

#[allow(clippy::expl_impl_clone_on_copy)]
impl<S: Slots, T: Coef> Clone for SlotArr<S, T> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}
impl<S: Slots, T: Coef> Copy for SlotArr<S, T> {}
impl<S: Slots, T: Coef> PartialEq for SlotArr<S, T> {
    fn eq(&self, o: &Self) -> bool {
        self.0 == o.0
    }
}
impl<S: Slots, T: Coef> core::fmt::Debug for SlotArr<S, T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.0.fmt(f)
    }
}

impl<S: Slots, T: Coef> SlotArr<S, T> {
    /// Multiply every element by a scalar.
    #[inline(always)]
    #[must_use]
    pub fn scale(self, s: T) -> Self {
        SlotArr(S::map(&self.0, &mut |x| *x * s))
    }
}

impl<S: Slots, T: Coef> Add for SlotArr<S, T> {
    type Output = Self;
    #[inline(always)]
    fn add(self, o: Self) -> Self {
        SlotArr(S::zip(&self.0, &o.0, &mut |a, b| *a + *b))
    }
}

impl<S: Slots, T: Coef> Sub for SlotArr<S, T> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, o: Self) -> Self {
        SlotArr(S::zip(&self.0, &o.0, &mut |a, b| *a - *b))
    }
}

impl<S: Slots, T: Coef> Neg for SlotArr<S, T> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        SlotArr(S::map(&self.0, &mut |a| -*a))
    }
}

impl<S: Slots, R: Slots, T: Coef> Mul<SlotArr<R, T>> for SlotArr<S, T> {
    type Output = SlotArr<Cat<S, R>, T>;
    #[inline(always)]
    fn mul(self, o: SlotArr<R, T>) -> SlotArr<Cat<S, R>, T> {
        SlotArr(S::outer::<R, T, T, T>(&self.0, &o.0, &mut |a, b| *a * *b))
    }
}

/// The coefficients of a value (`S = ()`) as plain numbers; `None` for maps and forms.
///
/// Kernels use this to take the direct path for values. The check folds at compile time.
#[inline(always)]
pub fn values<S: Slots, T: Coef, const N: usize>(c: &[S::Arr<T>; N]) -> Option<[T; N]> {
    S::as_value(&c[0])?;
    // Loops rather than `array::from_fn`/`map`, which batch kernels need inlined.
    let mut out = [T::zero(); N];
    for (o, x) in out.iter_mut().zip(c) {
        *o = S::as_value(x).expect("value slots");
    }
    Some(out)
}

/// The inverse of [`values`]. Only called on the value path, where it cannot fail.
#[inline(always)]
pub fn from_values<S: Slots, T: Coef, const N: usize>(v: [T; N]) -> [S::Arr<T>; N] {
    let open = "from_values called with open slots";
    let mut out = [S::from_value(T::zero()).expect(open); N];
    for (o, x) in out.iter_mut().zip(v) {
        *o = S::from_value(x).expect(open);
    }
    out
}
