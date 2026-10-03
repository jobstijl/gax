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
    ($i:expr; [$($A:ident),+]; $moved:ty) => {
        impl<$($A: Kind),+> MoveToFront<{ $i }> for ($($A,)+) {
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
                let d = <Self as MoveToFront<{ $i }>>::dims();
                permute::<Self, Self::Moved, X>(a, &d[..<Self as Slots>::LEN], $i)
            }
        }
    };
}

// Slot `i` to the front, for `i` from `$i` on: `$c` is slot `i`, `$b` those before, `$a` after.
macro_rules! moves {
    ($i:expr; [$($b:ident),*]; $c:ident; [$($a:ident),*]) => {
        move_impl!($i; [$($b,)* $c $(, $a)*]; ($c, $($b,)* $($a,)*));
        moves!(@next $i + 1; [$($b,)* $c]; [$($a),*]);
    };
    (@next $i:expr; [$($b:ident),*]; []) => {};
    (@next $i:expr; [$($b:ident),*]; [$c:ident $(, $a:ident)*]) => {
        moves!($i; [$($b),*]; $c; [$($a),*]);
    };
}

// Every move of every arity from `$have`'s up to `MAX_SLOTS`.
macro_rules! all_moves {
    ([$f:ident $(, $h:ident)*]; []) => {
        moves!(0; []; $f; [$($h),*]);
    };
    ([$f:ident $(, $h:ident)*]; [$next:ident $(, $rest:ident)*]) => {
        moves!(0; []; $f; [$($h),*]);
        all_moves!([$f $(, $h)*, $next]; [$($rest),*]);
    };
}

all_moves!([A0]; [A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11]);

// Every arity has every move: the first, a middle and the last slot of the smallest and
// largest tuples.
const _: () = {
    const fn moves<const I: usize, T: MoveToFront<I>>() {}
    #[allow(dead_code)]
    fn every_arity<A: Kind>() {
        moves::<0, (A,)>();
        moves::<0, (A, A)>();
        moves::<1, (A, A)>();
        moves::<0, (A, A, A, A, A, A, A, A, A, A, A, A)>();
        moves::<6, (A, A, A, A, A, A, A, A, A, A, A, A)>();
        moves::<11, (A, A, A, A, A, A, A, A, A, A, A, A)>();
    }
};
