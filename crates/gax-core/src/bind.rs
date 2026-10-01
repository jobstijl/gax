//! Binding and composition: filling slots with values or with other maps.

use crate::cast::{Cast, SubKind};
use crate::kind::{Extensor, Kind, Retype};
use crate::slots::{Cat, SlotArr, SplitFirst};

/// Fill the first slot of an extensor.
///
/// With `m` a map `C <- (B, Rest...)` and `x` of kind `B` with slots `Sx`, `m.of(x)` has
/// slots `Sx` followed by `Rest`: a value `x` fills the slot, and a map `x` is composed into
/// it, its own slots taking the slot's place.
///
/// `x` may also be of a smaller kind whose blades all lie in `B` (a [`SubKind`]): it is
/// embedded, as `x.cast::<B>()` would. Composing a map of the smaller kind narrows the slot.
///
/// ```
/// use gax::pga3d::{Flector, Motor, Point, Rotor};
/// let act: Flector<(Motor,), f64> = Motor::slot() * Point::xyz(1.0, 2.0, 3.0);
/// let r = Rotor::<(), f64>::from_coeffs([0.9, 0.1, 0.2, 0.3]);
/// assert_eq!(act.of(r), act.of(r.cast::<Motor>())); // a rotor fills a motor slot
/// let on_rotors: Flector<(Rotor,), f64> = act.of(Rotor::slot()); // the slot narrowed
/// assert_eq!(on_rotors.of(r), act.of(r));
/// ```
///
/// ```
/// use gax::pga3d::{Line, Point};
/// let q: Point<(), f64> = Point::xyz(0.0, 0.0, 1.0);
/// let lines_through_q: Line<(Point,), f64> = q & Point::slot();   // a map
/// let p = Point::xyz(1.0, 2.0, 3.0);
/// assert_eq!(lines_through_q.of(p), q & p);                       // fill the slot
/// let moved: Line<(Point,), f64> = lines_through_q.of(Point::slot().gp(2.0)); // compose
/// assert_eq!(moved.of(p), q & p.gp(2.0));
/// ```
#[diagnostic::on_unimplemented(
    message = "cannot fill the first slot of `{Self}` with `{X}`",
    note = "the argument's kind must be the slot's kind or a kind whose blades all lie in it (move another slot to the front with `.at::<I>()`)"
)]
/// ```
/// use gax::Of;
/// use gax::ApproxEq;
/// use gax::pga3d::{Motor, Point};
/// let m = Motor::<(), f64>::rotation_about(0.0, 0.0, 1.0, 0.4);
/// // The motor's action as a map with an open `Point` slot, filled later.
/// let map = m >> Point::slot();
/// let p = Point::xyz(1.0, 2.0, 3.0);
/// assert!(map.of(p).approx_eq(&(m >> p), 1e-12));
/// ```
pub trait Of<X> {
    /// The result type.
    type Output;
    /// Fill the first slot with `x`.
    fn of(self, x: X) -> Self::Output;
}

/// Result slots of binding `X` into the first slot of `M`.
pub type OfSlots<M, X> = Cat<<X as Extensor>::Slots, <<M as Extensor>::Slots as SplitFirst>::Tail>;

impl<M, X> Of<X> for M
where
    M: Extensor,
    M::Slots: SplitFirst,
    X: Extensor<Coef = M::Coef>,
    X::Kind: SubKind<<M::Slots as SplitFirst>::Head>,
{
    type Output = Retype<M, OfSlots<M, X>, M::Coef>;

    #[inline(always)]
    fn of(self, x: X) -> Self::Output {
        // `Σᵢ xs[i] ⊗ col[i]` per output coefficient, `xs` in the slot's layout.
        macro_rules! contract {
            ($xc:expr) => {{
                let xc = $xc;
                M::Kind::arr_map(self.coeffs(), |col| {
                    let col = <M::Slots as SplitFirst>::split(col);
                    let (c, xs) = (col.as_ref(), xc.as_ref());
                    let term = |i: usize| {
                        SlotArr::<X::Slots, M::Coef>(xs[i])
                            * SlotArr::<<M::Slots as SplitFirst>::Tail, M::Coef>(c[i])
                    };
                    let mut acc = term(0);
                    for i in 1..c.len() {
                        acc = acc + term(i);
                    }
                    acc.0
                })
            }};
        }
        let own = const {
            is_identity(
                <X::Kind as Cast<<M::Slots as SplitFirst>::Head>>::SHARED,
                <<M::Slots as SplitFirst>::Head as Kind>::N,
            )
        };
        let out = if own {
            contract!(x.coeffs())
        } else {
            // A sub-kind: placed in the slot's layout first (zeros on the slot's other blades).
            let embedded = crate::cast::cast::<X, <M::Slots as SplitFirst>::Head>(&x);
            contract!(embedded.coeffs())
        };
        <Self::Output as Extensor>::from_coeffs(out)
    }
}

/// Whether a [`Cast::SHARED`] table maps every blade of a kind with `n` blades to itself,
/// unflipped: a kind's own table (a sub-kind's can start the same way, with fewer blades).
const fn is_identity(shared: &[(usize, usize, bool)], n: usize) -> bool {
    if shared.len() != n {
        return false;
    }
    let mut k = 0;
    while k < shared.len() {
        let (i, j, flip) = shared[k];
        if i != k || j != k || flip {
            return false;
        }
        k += 1;
    }
    true
}
