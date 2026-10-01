//! Moving values and maps between kinds: projection, embedding and grade parts.
//!
//! An algebra's kinds are a closed family, so the generator emits, for every pair of kinds
//! that share a blade, the table of shared blades ([`Cast`]), marks the pairs where one kind's
//! blades all lie in the other ([`SubKind`]), and names the kind that holds each grade part of
//! a kind ([`GradePart`]). Everything here is a loop over those constant tables, which the
//! compiler unrolls: no blade is searched for at run time.
//!
//! * `x.cast::<K>()` keeps the blades `x` shares with `K` (orientations converted) and sets
//!   `K`'s other blades to zero: a projection, an embedding, or both.
//! * `x.grade::<G>()` keeps the grade-`G` part, as the declared kind that holds it.
//! * `m.of(x)` accepts any `x` whose kind is a [`SubKind`] of the slot's kind, embedding it.
//!
//! All three act on maps and forms as on values: on the output, keeping the slots.
//!
//! ```
//! use gax::pga3d::{Line, Motor, Rotor, Scalar};
//! let m = Motor::<(), f64>::rotation_about(0.0, 0.0, 1.0, 0.5).into_inner();
//! let r: Rotor<(), f64> = m.cast::<Rotor>();    // the rotation part
//! let b: Line<(), f64> = m.grade::<2>();       // the bivector part
//! let s: Scalar<(), f64> = m.grade::<0>();
//! assert_eq!(r.cast::<Motor>().cast::<Rotor>(), r);
//! assert_eq!(s.cast::<Motor>() + b.cast::<Motor>() + m.grade::<4>().cast::<Motor>(), m);
//! ```

use crate::coef::Coef;
use crate::kind::{Extensor, Kind};
use crate::slots::{SlotArr, Slots};

/// The blades `Self` shares with `K`, as `(i, j, flip)`: blade `i` of `Self` is blade `j` of
/// `K`, with the opposite orientation when `flip`. Generated for every pair of kinds of an
/// algebra that share a blade, a kind with itself included.
#[diagnostic::on_unimplemented(
    message = "`{Self}` and `{K}` have no blade in common",
    label = "no blade of `{Self}` is a blade of `{K}`",
    note = "a cast keeps the blades two kinds share; between these two it would always be zero"
)]
pub trait Cast<K: Kind>: Kind {
    /// The shared blades: `(index in Self, index in K, orientations differ)`.
    const SHARED: &'static [(usize, usize, bool)];
}

/// `Self`'s blades all lie in `K`: a value of `Self` embeds in `K` without loss, and fills a
/// slot of kind `K`. Generated for every such pair, a kind with itself included.
#[diagnostic::on_unimplemented(
    message = "a `{Self}` does not fit a slot of kind `{K}`",
    label = "some blade of `{Self}` is not a blade of `{K}`",
    note = "a slot takes values whose blades are among its kind's; project first with `.cast::<{K}>()` if dropping blades is intended"
)]
pub trait SubKind<K: Kind>: Cast<K> {}

/// The grade-`G` part of `Self`, as the declared kind [`GradePart::Out`] that holds it: the
/// kind of exactly that grade where one is declared, else the smallest that contains it.
/// Generated for every grade a kind has.
#[diagnostic::on_unimplemented(
    message = "`{Self}` has no blade of grade {G}",
    note = "`grade::<G>()` exists for the grades of the kind's blades"
)]
pub trait GradePart<const G: usize>: Kind {
    /// The kind of the part.
    type Out: Kind;
    /// The grade-`G` blades of `Self`, as in [`Cast::SHARED`].
    const SHARED: &'static [(usize, usize, bool)];
}

/// Move the blades of `m` listed in `shared` into a `K`, every other coefficient zero.
#[inline(always)]
fn move_blades<M: Extensor, K: Kind>(
    m: &M,
    shared: &[(usize, usize, bool)],
) -> K::Mv<M::Slots, M::Coef> {
    let zero = <M::Slots as Slots>::from_flat(&mut |_| M::Coef::zero(), 0);
    let mut out = K::arr_from_fn(|_| zero);
    let (c, o) = (m.coeffs().as_ref(), out.as_mut());
    for &(i, j, flip) in shared {
        o[j] = if flip {
            (-SlotArr::<M::Slots, M::Coef>(c[i])).0
        } else {
            c[i]
        };
    }
    <K::Mv<M::Slots, M::Coef> as Extensor>::from_coeffs(out)
}

/// `m` as a `K`: the blades they share kept (orientations converted), `K`'s others zero.
#[inline(always)]
pub fn cast<M: Extensor, K: Kind>(m: &M) -> K::Mv<M::Slots, M::Coef>
where
    M::Kind: Cast<K>,
{
    move_blades::<M, K>(m, <M::Kind as Cast<K>>::SHARED)
}

/// The grade-`G` part of `m`, as the declared kind that holds it.
#[inline(always)]
pub fn grade<M: Extensor, const G: usize>(
    m: &M,
) -> <<M::Kind as GradePart<G>>::Out as Kind>::Mv<M::Slots, M::Coef>
where
    M::Kind: GradePart<G>,
{
    move_blades::<M, <M::Kind as GradePart<G>>::Out>(m, <M::Kind as GradePart<G>>::SHARED)
}
