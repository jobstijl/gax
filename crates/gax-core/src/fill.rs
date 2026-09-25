//! Filling every slot of one kind with the same value: equality groups by kind.
//!
//! `m.fill(x)` binds each open slot whose kind is `x`'s kind to `x`, and leaves the other
//! slots open, in order. For a sandwich built with an open versor, `Point<(Motor, Motor)>`,
//! filling the motor gives the transformed point, and for a quadratic form it evaluates the
//! form on `x`. Kind equality is decided at the type level: each generated algebra
//! implements [`KindEq`] for every pair of its kinds.

use crate::coef::Coef;
use crate::kind::Kind;
use crate::slots::{SlotArr, Slots};

/// Type-level booleans.
pub trait Bit {}
/// Type-level true.
pub struct True;
/// Type-level false.
pub struct False;
impl Bit for True {}
impl Bit for False {}

/// Type-level equality of kinds, implemented by the generated algebras for every pair.
pub trait KindEq<K: Kind>: Kind {
    /// [`True`] if the kinds are the same, else [`False`].
    type Out: Bit;
}

/// Slot lists in which every slot of kind `K` can be filled.
///
/// (Deliberately not a subtrait of [`Slots`]: a `(A1,): FillList<K>` bound in an impl would
/// then shadow the tuple's own `Slots` impl and stop its associated types from normalizing.)
pub trait FillList<K: Kind> {
    /// The remaining slots.
    type Out: Slots;
    /// Contract every slot of kind `K` with `x`.
    fn fill<T: Coef>(a: &<Self as Slots>::Arr<T>, x: &K::Arr<T>) -> <Self::Out as Slots>::Arr<T>
    where
        Self: Slots;
}

/// One step of [`FillList`]: fill (for [`True`]) or keep (for [`False`]) the head slot `H`.
pub trait FillStep<K: Kind, H: Kind, Tail: Slots + FillList<K>>: Bit {
    /// The remaining slots after this step.
    type Out: Slots;
    /// Apply the step to an array `H::Arr<Tail::Arr<T>>`.
    fn step<T: Coef>(a: &H::Arr<Tail::Arr<T>>, x: &K::Arr<T>) -> <Self::Out as Slots>::Arr<T>;
}

impl<K: Kind, H: Kind, Tail: Slots + FillList<K>> FillStep<K, H, Tail> for True {
    type Out = Tail::Out;
    #[inline(always)]
    fn step<T: Coef>(a: &H::Arr<Tail::Arr<T>>, x: &K::Arr<T>) -> <Tail::Out as Slots>::Arr<T> {
        // H and K are the same kind here, so both arrays have its length.
        let (a, xs) = (a.as_ref(), x.as_ref());
        let term = |i: usize| SlotArr::<Tail::Out, T>(Tail::fill(&a[i], x)).scale(xs[i]);
        let mut acc = term(0);
        for i in 1..a.len() {
            acc = acc + term(i);
        }
        acc.0
    }
}

impl<K: Kind, H: Kind, Tail: Slots + FillList<K>> FillStep<K, H, Tail> for False {
    type Out = <Tail::Out as Slots>::Prepend<H>;
    #[inline(always)]
    fn step<T: Coef>(a: &H::Arr<Tail::Arr<T>>, x: &K::Arr<T>) -> <Self::Out as Slots>::Arr<T> {
        <Tail::Out as Slots>::prepend_arr::<H, T>(H::arr_map(a, |t| Tail::fill(t, x)))
    }
}

impl<K: Kind> FillList<K> for () {
    type Out = ();
    #[inline(always)]
    fn fill<T: Coef>(a: &T, _x: &K::Arr<T>) -> T {
        *a
    }
}

macro_rules! fill_list {
    ($H:ident $(, $T:ident)*) => {
        impl<K: Kind, $H: KindEq<K> $(, $T: Kind)*> FillList<K> for ($H, $($T,)*)
        where
            ($($T,)*): FillList<K>,
            <$H as KindEq<K>>::Out: FillStep<K, $H, ($($T,)*)>,
        {
            type Out = <<$H as KindEq<K>>::Out as FillStep<K, $H, ($($T,)*)>>::Out;
            #[inline(always)]
            fn fill<T: Coef>(a: &<Self as Slots>::Arr<T>, x: &K::Arr<T>) -> <Self::Out as Slots>::Arr<T> {
                <<$H as KindEq<K>>::Out as FillStep<K, $H, ($($T,)*)>>::step(a, x)
            }
        }
    };
}

fill_list!(A0);
fill_list!(A0, A1);
fill_list!(A0, A1, A2);
fill_list!(A0, A1, A2, A3);
fill_list!(A0, A1, A2, A3, A4);
fill_list!(A0, A1, A2, A3, A4, A5);
fill_list!(A0, A1, A2, A3, A4, A5, A6);
fill_list!(A0, A1, A2, A3, A4, A5, A6, A7);

/// Non-empty slot lists: access to the last slot.
pub trait SplitLast: Slots {
    /// All slots but the last.
    type Init: Slots;
    /// The last slot's kind.
    type Last: Kind;
}

macro_rules! split_last {
    ([$($I:ident),*] $L:ident) => {
        impl<$($I: Kind,)* $L: Kind> SplitLast for ($($I,)* $L,) {
            type Init = ($($I,)*);
            type Last = $L;
        }
    };
}

split_last!([] A0);
split_last!([A0] A1);
split_last!([A0, A1] A2);
split_last!([A0, A1, A2] A3);
split_last!([A0, A1, A2, A3] A4);
split_last!([A0, A1, A2, A3, A4] A5);
split_last!([A0, A1, A2, A3, A4, A5] A6);
split_last!([A0, A1, A2, A3, A4, A5, A6] A7);
