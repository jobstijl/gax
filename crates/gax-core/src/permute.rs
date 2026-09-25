//! Slot permutations: `at::<I>()` moves slot `I` to the front.

use crate::coef::Elem;
use crate::kind::Kind;
use crate::slots::Slots;

/// Slot lists whose slot `I` can be moved to the front.
pub trait MoveToFront<const I: usize>: Slots {
    /// The list with slot `I` first and the others in their original order.
    type Moved: Slots;
    /// The dimensions of the slots, in the original order.
    fn dims() -> [usize; 8];
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
            let mut moved_dims = [0usize; 8];
            moved_dims[0] = dims[i];
            let mut k = 1;
            for (d, &dim) in dims.iter().enumerate() {
                if d != i {
                    moved_dims[k] = dim;
                    k += 1;
                }
            }
            let mut rem = flat;
            let mut idx = [0usize; 8];
            for p in (0..n).rev() {
                idx[p] = rem % moved_dims[p];
                rem /= moved_dims[p];
            }
            // Back to the original order and its flat index.
            let mut orig = [0usize; 8];
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
            fn dims() -> [usize; 8] {
                let v = [$($A::N),+];
                let mut d = [0; 8];
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
