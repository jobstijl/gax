// Design B on plain tuples (up to 4 here; macro-generated like all_tuples!), Point<(Twist, Twist)> syntax.
use core::ops::Mul;
pub trait SlotType: Copy + 'static { type Arr<X: Copy>: Copy; fn map<X: Copy, Y: Copy>(a: &Self::Arr<X>, f: impl FnMut(&X) -> Y) -> Self::Arr<Y>; }
macro_rules! slot_type { ($n:ident, $d:literal) => {
    #[derive(Clone, Copy)] pub struct $n;
    impl SlotType for $n { type Arr<X: Copy> = [X; $d];
        fn map<X: Copy, Y: Copy>(a: &[X; $d], mut f: impl FnMut(&X) -> Y) -> [Y; $d] { core::array::from_fn(|i| f(&a[i])) } }
}}
slot_type!(PointT, 4); slot_type!(TwistT, 6);

/// Sentinel for "list too long" so that the longest tuple can still implement Prepend.
#[derive(Clone, Copy)] pub struct Overflow;

pub trait HasCat { type Cat<R: Slots>: Slots; }
pub type Cat<A, B> = <A as HasCat>::Cat<B>;
pub trait Slots: HasCat<Cat<()> = Self> + Copy + 'static {
    type Prepend<H: SlotType>: Slots;
    type Arr<X: Copy>: Copy;
    fn map<X: Copy, Y: Copy>(a: &Self::Arr<X>, f: &mut impl FnMut(&X) -> Y) -> Self::Arr<Y>;
    /// witness: Prepend<H>::Arr<X> == H::Arr<Self::Arr<X>>
    fn prepend_arr<H: SlotType, X: Copy>(a: H::Arr<Self::Arr<X>>) -> <Self::Prepend<H> as Slots>::Arr<X>;
    fn outer<R: Slots, A: Copy, B: Copy, C: Copy>(a: &Self::Arr<A>, b: &R::Arr<B>, f: &mut impl FnMut(&A, &B) -> C) -> <Self::Cat<R> as Slots>::Arr<C>;
}
impl HasCat for Overflow { type Cat<R: Slots> = Overflow; }
impl Slots for Overflow {
    type Prepend<H: SlotType> = Overflow; type Arr<X: Copy> = ();
    fn map<X: Copy, Y: Copy>(_: &(), _: &mut impl FnMut(&X) -> Y) {}
    fn prepend_arr<H: SlotType, X: Copy>(_: H::Arr<()>) { panic!("slot list too long") }
    fn outer<R: Slots, A: Copy, B: Copy, C: Copy>(_: &(), _: &R::Arr<B>, _: &mut impl FnMut(&A, &B) -> C) {}
}
impl HasCat for () { type Cat<R: Slots> = R; }
impl Slots for () {
    type Prepend<H: SlotType> = (H,); type Arr<X: Copy> = X;
    fn map<X: Copy, Y: Copy>(a: &X, f: &mut impl FnMut(&X) -> Y) -> Y { f(a) }
    fn prepend_arr<H: SlotType, X: Copy>(a: H::Arr<X>) -> H::Arr<X> { a }
    fn outer<R: Slots, A: Copy, B: Copy, C: Copy>(a: &A, b: &R::Arr<B>, f: &mut impl FnMut(&A, &B) -> C) -> R::Arr<C> { R::map(b, &mut |b| f(a, b)) }
}
macro_rules! tuple_slots {
    ($H:ident $(, $T:ident)* ; $($P:ty)?) => {
        impl<$H: SlotType $(, $T: SlotType)*> HasCat for ($H, $($T,)*) {
            type Cat<R: Slots> = <<($($T,)*) as HasCat>::Cat<R> as Slots>::Prepend<$H>;
        }
        impl<$H: SlotType $(, $T: SlotType)*> Slots for ($H, $($T,)*) {
            type Prepend<H0: SlotType> = tuple_slots!(@pre H0, $H $(, $T)* ; $($P)?);
            type Arr<X: Copy> = $H::Arr<<($($T,)*) as Slots>::Arr<X>>;
            fn map<X: Copy, Y: Copy>(a: &Self::Arr<X>, f: &mut impl FnMut(&X) -> Y) -> Self::Arr<Y> { $H::map(a, |t| <($($T,)*)>::map(t, f)) }
            fn prepend_arr<H0: SlotType, X: Copy>(a: H0::Arr<Self::Arr<X>>) -> <Self::Prepend<H0> as Slots>::Arr<X> { tuple_slots!(@pa a ; $($P)?) }
            fn outer<R: Slots, A: Copy, B: Copy, C: Copy>(a: &Self::Arr<A>, b: &R::Arr<B>, f: &mut impl FnMut(&A, &B) -> C) -> <Self::Cat<R> as Slots>::Arr<C> {
                let inner = $H::map(a, |ta| <($($T,)*)>::outer::<R, A, B, C>(ta, b, f));
                <<($($T,)*) as HasCat>::Cat<R> as Slots>::prepend_arr::<$H, C>(inner)
            }
        }
    };
    (@pre $H0:ident, $($T:ident),* ; ) => { ($H0, $($T,)*) };
    (@pre $H0:ident, $($T:ident),* ; $P:ty) => { Overflow };
    (@pa $a:ident ; ) => { $a };
    (@pa $a:ident ; $P:ty) => { { let _ = $a; panic!("slot list too long") } };
}
tuple_slots!(T0;);
tuple_slots!(T0, T1;);
tuple_slots!(T0, T1, T2;);
tuple_slots!(T0, T1, T2, T3; Overflow);

pub struct Point<S: Slots> { pub c: S::Arr<[f32; 4]> }
impl<S: Slots> Clone for Point<S> { fn clone(&self) -> Self { *self } }
impl<S: Slots> Copy for Point<S> {}
pub struct Motor<S: Slots> { pub c: S::Arr<[f32; 8]> }
impl<S1: Slots, S2: Slots> Mul<Point<S2>> for Point<S1> {
    type Output = Motor<Cat<S1, S2>>;
    fn mul(self, r: Point<S2>) -> Self::Output { Motor { c: S1::outer::<S2, _, _, _>(&self.c, &r.c, &mut |a, b| [a[0]*b[0]; 8]) } }
}
pub fn generic<S: Slots>(p: Point<S>, v: Point<()>) -> Point<S> { let _m: Motor<S> = p * v; let _n: Motor<S> = v * p; p }
fn main() {
    let v: Point<()> = Point { c: [2.0; 4] };
    let m: Point<(PointT,)> = Point { c: [[1.0; 4]; 4] };
    let bil: Motor<(PointT, PointT)> = m * m; // tuple-typed result, concrete normalization
    let arr: [[[f32; 8]; 4]; 4] = bil.c;
    let t: Motor<(PointT, PointT, PointT)> = bil_point(m, m, m);
    println!("{:?} {:?} {}", arr[0][0][0], (m * v).c[0][0], t.c[3][3][3][7]);
}
fn bil_point(a: Point<(PointT,)>, b: Point<(PointT,)>, c: Point<(PointT,)>) -> Motor<(PointT, PointT, PointT)> {
    let ab: Motor<(PointT, PointT)> = a * b; let _ = ab; let _ = c;
    Motor { c: [[[[1.0; 8]; 4]; 4]; 4] }
}
