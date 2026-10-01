//! Slot permutations: `at::<I>()` moves slot `I` to the front.

use crate::coef::Elem;
use crate::kind::Kind;
use crate::slots::{MAX_SLOTS, Slots};

/// Slot lists whose slot `I` can be moved to the front.
///
/// ```
/// use gax::pga3d::{Line, Point};
/// let join: Line<(Point, Point), f64> = Point::slot() & Point::slot();
/// let (a, b) = (Point::xyz(0.0, 0.0, 0.0), Point::xyz(1.0, 0.0, 0.0));
/// assert_eq!(join.at::<1>().of(b).of(a), a & b); // fill the second slot first
/// ```
pub trait MoveToFront<const I: usize>: Slots {
    /// The list with slot `I` first and the others in their original order.
    type Moved: Slots;
    /// The dimensions of the slots, in the original order.
    fn dims() -> [usize; MAX_SLOTS];
    /// Permute a coefficient array accordingly.
    fn move_arr<X: Elem>(a: &Self::Arr<X>) -> <Self::Moved as Slots>::Arr<X>;
}

/// Read `a` in the order of the moved list: the flat index of the moved array maps to the
/// flat index of the original.
#[inline(always)]
fn permute<S: Slots, D: Slots, X: Elem>(a: &S::Arr<X>, dims: &[usize], i: usize) -> D::Arr<X> {
    let n = dims.len();
    D::from_flat(
        &mut |flat| {
            // Multi-index in the moved order: (j_i, j_0, ..., j_{i-1}, j_{i+1}, ...).
            let mut moved_dims = [0usize; MAX_SLOTS];
            moved_dims[0] = dims[i];
            let mut k = 1;
            for (d, &dim) in dims.iter().enumerate() {
                if d != i {
                    moved_dims[k] = dim;
                    k += 1;
                }
            }
            let mut rem = flat;
            let mut idx = [0usize; MAX_SLOTS];
            for p in (0..n).rev() {
                idx[p] = rem % moved_dims[p];
                rem /= moved_dims[p];
            }
            // Back to the original order and its flat index.
            let mut orig = [0usize; MAX_SLOTS];
            orig[i] = idx[0];
            let mut k = 1;
            for (d, o) in orig.iter_mut().enumerate().take(n) {
                if d != i {
                    *o = idx[k];
                    k += 1;
                }
            }
            let mut f = 0;
            for d in 0..n {
                f = f * dims[d] + orig[d];
            }
            S::get_flat(a, f)
        },
        0,
    )
}

macro_rules! move_impl {
    ($i:literal; [$($A:ident),+]; $moved:ty) => {
        impl<$($A: Kind),+> MoveToFront<$i> for ($($A,)+) {
            type Moved = $moved;
            #[inline(always)]
            fn dims() -> [usize; MAX_SLOTS] {
                let v = [$($A::N),+];
                let mut d = [0; MAX_SLOTS];
                d[..v.len()].copy_from_slice(&v);
                d
            }
            #[inline(always)]
            fn move_arr<X: Elem>(a: &Self::Arr<X>) -> <Self::Moved as Slots>::Arr<X> {
                let d = <Self as MoveToFront<$i>>::dims();
                permute::<Self, Self::Moved, X>(a, &d[..<Self as Slots>::LEN], $i)
            }
        }
    };
}

move_impl!(0; [A0]; (A0,));
move_impl!(0; [A0, A1]; (A0, A1));
move_impl!(1; [A0, A1]; (A1, A0));
move_impl!(0; [A0, A1, A2]; (A0, A1, A2));
move_impl!(1; [A0, A1, A2]; (A1, A0, A2));
move_impl!(2; [A0, A1, A2]; (A2, A0, A1));
move_impl!(0; [A0, A1, A2, A3]; (A0, A1, A2, A3));
move_impl!(1; [A0, A1, A2, A3]; (A1, A0, A2, A3));
move_impl!(2; [A0, A1, A2, A3]; (A2, A0, A1, A3));
move_impl!(3; [A0, A1, A2, A3]; (A3, A0, A1, A2));
move_impl!(0; [A0, A1, A2, A3, A4]; (A0, A1, A2, A3, A4));
move_impl!(1; [A0, A1, A2, A3, A4]; (A1, A0, A2, A3, A4));
move_impl!(2; [A0, A1, A2, A3, A4]; (A2, A0, A1, A3, A4));
move_impl!(3; [A0, A1, A2, A3, A4]; (A3, A0, A1, A2, A4));
move_impl!(4; [A0, A1, A2, A3, A4]; (A4, A0, A1, A2, A3));
move_impl!(0; [A0, A1, A2, A3, A4, A5]; (A0, A1, A2, A3, A4, A5));
move_impl!(1; [A0, A1, A2, A3, A4, A5]; (A1, A0, A2, A3, A4, A5));
move_impl!(2; [A0, A1, A2, A3, A4, A5]; (A2, A0, A1, A3, A4, A5));
move_impl!(3; [A0, A1, A2, A3, A4, A5]; (A3, A0, A1, A2, A4, A5));
move_impl!(4; [A0, A1, A2, A3, A4, A5]; (A4, A0, A1, A2, A3, A5));
move_impl!(5; [A0, A1, A2, A3, A4, A5]; (A5, A0, A1, A2, A3, A4));
move_impl!(0; [A0, A1, A2, A3, A4, A5, A6]; (A0, A1, A2, A3, A4, A5, A6));
move_impl!(1; [A0, A1, A2, A3, A4, A5, A6]; (A1, A0, A2, A3, A4, A5, A6));
move_impl!(2; [A0, A1, A2, A3, A4, A5, A6]; (A2, A0, A1, A3, A4, A5, A6));
move_impl!(3; [A0, A1, A2, A3, A4, A5, A6]; (A3, A0, A1, A2, A4, A5, A6));
move_impl!(4; [A0, A1, A2, A3, A4, A5, A6]; (A4, A0, A1, A2, A3, A5, A6));
move_impl!(5; [A0, A1, A2, A3, A4, A5, A6]; (A5, A0, A1, A2, A3, A4, A6));
move_impl!(6; [A0, A1, A2, A3, A4, A5, A6]; (A6, A0, A1, A2, A3, A4, A5));
move_impl!(0; [A0, A1, A2, A3, A4, A5, A6, A7]; (A0, A1, A2, A3, A4, A5, A6, A7));
move_impl!(1; [A0, A1, A2, A3, A4, A5, A6, A7]; (A1, A0, A2, A3, A4, A5, A6, A7));
move_impl!(2; [A0, A1, A2, A3, A4, A5, A6, A7]; (A2, A0, A1, A3, A4, A5, A6, A7));
move_impl!(3; [A0, A1, A2, A3, A4, A5, A6, A7]; (A3, A0, A1, A2, A4, A5, A6, A7));
move_impl!(4; [A0, A1, A2, A3, A4, A5, A6, A7]; (A4, A0, A1, A2, A3, A5, A6, A7));
move_impl!(5; [A0, A1, A2, A3, A4, A5, A6, A7]; (A5, A0, A1, A2, A3, A4, A6, A7));
move_impl!(6; [A0, A1, A2, A3, A4, A5, A6, A7]; (A6, A0, A1, A2, A3, A4, A5, A7));
move_impl!(7; [A0, A1, A2, A3, A4, A5, A6, A7]; (A7, A0, A1, A2, A3, A4, A5, A6));
move_impl!(0; [A0, A1, A2, A3, A4, A5, A6, A7, A8]; (A0, A1, A2, A3, A4, A5, A6, A7, A8));
move_impl!(1; [A0, A1, A2, A3, A4, A5, A6, A7, A8]; (A1, A0, A2, A3, A4, A5, A6, A7, A8));
move_impl!(2; [A0, A1, A2, A3, A4, A5, A6, A7, A8]; (A2, A0, A1, A3, A4, A5, A6, A7, A8));
move_impl!(3; [A0, A1, A2, A3, A4, A5, A6, A7, A8]; (A3, A0, A1, A2, A4, A5, A6, A7, A8));
move_impl!(4; [A0, A1, A2, A3, A4, A5, A6, A7, A8]; (A4, A0, A1, A2, A3, A5, A6, A7, A8));
move_impl!(5; [A0, A1, A2, A3, A4, A5, A6, A7, A8]; (A5, A0, A1, A2, A3, A4, A6, A7, A8));
move_impl!(6; [A0, A1, A2, A3, A4, A5, A6, A7, A8]; (A6, A0, A1, A2, A3, A4, A5, A7, A8));
move_impl!(7; [A0, A1, A2, A3, A4, A5, A6, A7, A8]; (A7, A0, A1, A2, A3, A4, A5, A6, A8));
move_impl!(8; [A0, A1, A2, A3, A4, A5, A6, A7, A8]; (A8, A0, A1, A2, A3, A4, A5, A6, A7));
move_impl!(0; [A0, A1, A2, A3, A4, A5, A6, A7, A8, A9]; (A0, A1, A2, A3, A4, A5, A6, A7, A8, A9));
move_impl!(1; [A0, A1, A2, A3, A4, A5, A6, A7, A8, A9]; (A1, A0, A2, A3, A4, A5, A6, A7, A8, A9));
move_impl!(2; [A0, A1, A2, A3, A4, A5, A6, A7, A8, A9]; (A2, A0, A1, A3, A4, A5, A6, A7, A8, A9));
move_impl!(3; [A0, A1, A2, A3, A4, A5, A6, A7, A8, A9]; (A3, A0, A1, A2, A4, A5, A6, A7, A8, A9));
move_impl!(4; [A0, A1, A2, A3, A4, A5, A6, A7, A8, A9]; (A4, A0, A1, A2, A3, A5, A6, A7, A8, A9));
move_impl!(5; [A0, A1, A2, A3, A4, A5, A6, A7, A8, A9]; (A5, A0, A1, A2, A3, A4, A6, A7, A8, A9));
move_impl!(6; [A0, A1, A2, A3, A4, A5, A6, A7, A8, A9]; (A6, A0, A1, A2, A3, A4, A5, A7, A8, A9));
move_impl!(7; [A0, A1, A2, A3, A4, A5, A6, A7, A8, A9]; (A7, A0, A1, A2, A3, A4, A5, A6, A8, A9));
move_impl!(8; [A0, A1, A2, A3, A4, A5, A6, A7, A8, A9]; (A8, A0, A1, A2, A3, A4, A5, A6, A7, A9));
move_impl!(9; [A0, A1, A2, A3, A4, A5, A6, A7, A8, A9]; (A9, A0, A1, A2, A3, A4, A5, A6, A7, A8));
move_impl!(0; [A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10]; (A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10));
move_impl!(1; [A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10]; (A1, A0, A2, A3, A4, A5, A6, A7, A8, A9, A10));
move_impl!(2; [A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10]; (A2, A0, A1, A3, A4, A5, A6, A7, A8, A9, A10));
move_impl!(3; [A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10]; (A3, A0, A1, A2, A4, A5, A6, A7, A8, A9, A10));
move_impl!(4; [A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10]; (A4, A0, A1, A2, A3, A5, A6, A7, A8, A9, A10));
move_impl!(5; [A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10]; (A5, A0, A1, A2, A3, A4, A6, A7, A8, A9, A10));
move_impl!(6; [A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10]; (A6, A0, A1, A2, A3, A4, A5, A7, A8, A9, A10));
move_impl!(7; [A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10]; (A7, A0, A1, A2, A3, A4, A5, A6, A8, A9, A10));
move_impl!(8; [A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10]; (A8, A0, A1, A2, A3, A4, A5, A6, A7, A9, A10));
move_impl!(9; [A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10]; (A9, A0, A1, A2, A3, A4, A5, A6, A7, A8, A10));
move_impl!(10; [A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10]; (A10, A0, A1, A2, A3, A4, A5, A6, A7, A8, A9));
move_impl!(0; [A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11]; (A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11));
move_impl!(1; [A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11]; (A1, A0, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11));
move_impl!(2; [A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11]; (A2, A0, A1, A3, A4, A5, A6, A7, A8, A9, A10, A11));
move_impl!(3; [A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11]; (A3, A0, A1, A2, A4, A5, A6, A7, A8, A9, A10, A11));
move_impl!(4; [A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11]; (A4, A0, A1, A2, A3, A5, A6, A7, A8, A9, A10, A11));
move_impl!(5; [A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11]; (A5, A0, A1, A2, A3, A4, A6, A7, A8, A9, A10, A11));
move_impl!(6; [A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11]; (A6, A0, A1, A2, A3, A4, A5, A7, A8, A9, A10, A11));
move_impl!(7; [A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11]; (A7, A0, A1, A2, A3, A4, A5, A6, A8, A9, A10, A11));
move_impl!(8; [A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11]; (A8, A0, A1, A2, A3, A4, A5, A6, A7, A9, A10, A11));
move_impl!(9; [A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11]; (A9, A0, A1, A2, A3, A4, A5, A6, A7, A8, A10, A11));
move_impl!(10; [A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11]; (A10, A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A11));
move_impl!(11; [A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11]; (A11, A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10));
