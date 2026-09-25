//! Binding and composition: filling slots with values or with other maps.

use crate::kind::{Extensor, Kind, Retype};
use crate::slots::{Cat, SlotArr, SplitFirst};

/// Fill the first slot of an extensor.
///
/// With `m` a map `C <- (B, Rest...)` and `x` of kind `B` with slots `Sx`, `m.of(x)` has
/// slots `Sx` followed by `Rest`: a value `x` fills the slot, and a map `x` is composed into
/// it, its own slots taking the slot's place.
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
    note = "the argument's kind must be the slot's kind (move another slot to the front with `.at::<I>()`)"
)]
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
    X: Extensor<Coef = M::Coef, Kind = <M::Slots as SplitFirst>::Head>,
{
    type Output = Retype<M, OfSlots<M, X>, M::Coef>;

    #[inline(always)]
    fn of(self, x: X) -> Self::Output {
        let xc = x.coeffs();
        let out = M::Kind::arr_map(self.coeffs(), |col| {
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
        });
        <Self::Output as Extensor>::from_coeffs(out)
    }
}
