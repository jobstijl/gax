// Design B + nested-array coefficient storage through GATs, with a generic outer product.
use core::marker::PhantomData;
use core::ops::{BitAnd, BitXor, Mul};

// ---- slot element types (the "kind" of an open slot) --------------------------------
pub trait SlotType: 'static {
    const DIM: usize;
    type Arr<X: Copy>: Copy;
    fn from_fn<X: Copy>(f: impl FnMut(usize) -> X) -> Self::Arr<X>;
    fn map<X: Copy, Y: Copy>(a: &Self::Arr<X>, f: impl FnMut(&X) -> Y) -> Self::Arr<Y>;
}
macro_rules! slot_type { ($n:ident, $d:literal) => {
    pub struct $n;
    impl SlotType for $n {
        const DIM: usize = $d;
        type Arr<X: Copy> = [X; $d];
        fn from_fn<X: Copy>(f: impl FnMut(usize) -> X) -> [X; $d] { core::array::from_fn(f) }
        fn map<X: Copy, Y: Copy>(a: &[X; $d], mut f: impl FnMut(&X) -> Y) -> [Y; $d] { core::array::from_fn(|i| f(&a[i])) }
    }
}}
slot_type!(PointT, 4);
slot_type!(PlaneT, 4);
slot_type!(TwistT, 6);

// ---- slot lists -------------------------------------------------------------------------
#[derive(Clone, Copy)] pub struct Nil;
pub struct Cons<H, T>(PhantomData<(H, T)>);
impl<H, T> Clone for Cons<H, T> { fn clone(&self) -> Self { *self } }
impl<H, T> Copy for Cons<H, T> {}

pub trait HasCat { type Cat<R: Slots>: Slots; }
pub trait Slots: HasCat<Cat<Nil> = Self> + Copy + 'static {
    const LEN: usize;
    const SIZE: usize; // product of dims
    type Arr<X: Copy>: Copy;
    fn from_fn<X: Copy>(f: &mut impl FnMut(usize) -> X, base: usize) -> Self::Arr<X>;
    fn map<X: Copy, Y: Copy>(a: &Self::Arr<X>, f: &mut impl FnMut(&X) -> Y) -> Self::Arr<Y>;
    fn outer<R: Slots, A: Copy, B: Copy, C: Copy>(
        a: &Self::Arr<A>, b: &R::Arr<B>, f: &mut impl FnMut(&A, &B) -> C,
    ) -> <Self::Cat<R> as Slots>::Arr<C>;
}
pub type Cat<A, B> = <A as HasCat>::Cat<B>;

impl HasCat for Nil { type Cat<R: Slots> = R; }
impl Slots for Nil {
    const LEN: usize = 0;
    const SIZE: usize = 1;
    type Arr<X: Copy> = X;
    fn from_fn<X: Copy>(f: &mut impl FnMut(usize) -> X, base: usize) -> X { f(base) }
    fn map<X: Copy, Y: Copy>(a: &X, f: &mut impl FnMut(&X) -> Y) -> Y { f(a) }
    fn outer<R: Slots, A: Copy, B: Copy, C: Copy>(a: &A, b: &R::Arr<B>, f: &mut impl FnMut(&A, &B) -> C) -> R::Arr<C> {
        R::map(b, &mut |b| f(a, b))
    }
}
impl<H: SlotType, T: Slots> HasCat for Cons<H, T> { type Cat<R: Slots> = Cons<H, T::Cat<R>>; }
impl<H: SlotType, T: Slots> Slots for Cons<H, T> {
    const LEN: usize = 1 + T::LEN;
    const SIZE: usize = H::DIM * T::SIZE; // fine as an associated const value
    type Arr<X: Copy> = H::Arr<T::Arr<X>>;
    fn from_fn<X: Copy>(f: &mut impl FnMut(usize) -> X, base: usize) -> Self::Arr<X> {
        H::from_fn(|i| T::from_fn(f, (base * H::DIM + i)))
    }
    fn map<X: Copy, Y: Copy>(a: &Self::Arr<X>, f: &mut impl FnMut(&X) -> Y) -> Self::Arr<Y> {
        H::map(a, |t| T::map(t, f))
    }
    fn outer<R: Slots, A: Copy, B: Copy, C: Copy>(
        a: &Self::Arr<A>, b: &R::Arr<B>, f: &mut impl FnMut(&A, &B) -> C,
    ) -> <Self::Cat<R> as Slots>::Arr<C> {
        H::map(a, |ta| T::outer::<R, A, B, C>(ta, b, f))
    }
}

// ---- multivector types ---------------------------------------------------------------
macro_rules! mv { ($($n:ident $k:literal),*) => { $(
    pub struct $n<S: Slots> { pub c: S::Arr<[f32; $k]> }
    impl<S: Slots> Clone for $n<S> { fn clone(&self) -> Self { *self } }
    impl<S: Slots> Copy for $n<S> {}
)* } }
mv!(Plane 4, Line 6, Point 4, Motor 8);

// toy "products": just enough arithmetic to observe the outer structure
impl<S1: Slots, S2: Slots> BitAnd<Point<S2>> for Plane<S1> {
    type Output = Line<Cat<S1, S2>>;
    fn bitand(self, r: Point<S2>) -> Self::Output {
        Line { c: S1::outer::<S2, _, _, _>(&self.c, &r.c, &mut |a, b| [a[0]*b[0], a[1]*b[1], a[2]*b[2], a[3]*b[3], 0.0, 0.0]) }
    }
}
impl<S1: Slots, S2: Slots> BitXor<Plane<S2>> for Line<S1> {
    type Output = Point<Cat<S1, S2>>;
    fn bitxor(self, r: Plane<S2>) -> Self::Output {
        Point { c: S1::outer::<S2, _, _, _>(&self.c, &r.c, &mut |a, b| [a[0]+b[0], a[1]+b[1], a[2]+b[2], a[3]+b[3]]) }
    }
}
impl<S1: Slots, S2: Slots> Mul<Point<S2>> for Point<S1> {
    type Output = Motor<Cat<S1, S2>>;
    fn mul(self, r: Point<S2>) -> Self::Output {
        Motor { c: S1::outer::<S2, _, _, _>(&self.c, &r.c, &mut |a, b| [a[0]*b[0], a[1]*b[1], a[2]*b[2], a[3]*b[3], 0.,0.,0.,0.]) }
    }
}
pub const LIGHT: Plane<Nil> = Plane { c: [1.0, 1.0, 1.0, 1.0] };
pub const GROUND: Plane<Nil> = Plane { c: [0.0, 0.0, 0.0, 1.0] };

// ---- user code: only `S: Slots` -----------------------------------------------------
pub fn shadow<S: Slots>(p: Point<S>) -> Point<S> { (LIGHT & p) ^ GROUND }
pub fn prod<S1: Slots, S2: Slots>(a: Point<S1>, b: Point<S2>) -> Motor<Cat<S1, S2>> { a * b }

fn main() {
    let v: Point<Nil> = Point { c: [1., 2., 3., 4.] };
    let s = shadow(v);
    println!("value: {:?}", s.c);
    // a linear map Point -> Point : c has type [[f32;4];4]
    let m: Point<Cons<PointT, Nil>> = Point { c: <Cons<PointT, Nil>>::from_fn(&mut |i| [i as f32; 4], 0) };
    let sm: Point<Cons<PointT, Nil>> = shadow(m);
    let arr: [[f32; 4]; 4] = sm.c; // concrete normalization of nested GAT storage
    println!("map:   {:?}", arr);
    let mm = prod(m, m);
    let arr2: [[[f32; 8]; 4]; 4] = mm.c;
    println!("bilinear[3][2] = {:?}", arr2[3][2]);
    println!("SIZE = {}", <Cons<TwistT, Cons<PointT, Nil>> as Slots>::SIZE);
}
