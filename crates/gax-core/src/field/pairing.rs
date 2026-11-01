//! The pairing in which a square field map is symmetric: its unknowns' kind with itself
//! (coefficient by coefficient), or with its complement (a correlation).

use crate::cast::Complement;
use crate::fill::{False, KindEq, True};
use crate::kind::{Extensor, Kind};

/// A map cell `B <- A` whose output pairs with its input to numbers, so that a field map of
/// such cells has a quadratic form, its energy `Σ (A x)ᵢ · xᵢ`, and can be symmetric and
/// definite: [`SparseMap::solve`](super::SparseMap::solve) (conjugate gradients) and
/// [`SparseMap::eigh`](super::SparseMap::eigh) need one. Two pairings, decided by the kinds:
///
/// * a map of a kind to itself, `K <- K`, pairs by the coefficients, `Σₖ yₖ xₖ` (a stiffness, a
///   mass, a Laplacian);
/// * a correlation, a map of a kind to its complement `C(K) <- K`, pairs by the regressive
///   product, `⟨y ∨ x⟩₀`, the output on the left ([`Complement`]): a conductance from potentials
///   to amounts (numga's `Amount- <- Scalar+`), a mobility from forces to velocities (PGA2D's
///   `Direction <- OriginLine`), the energy forms of DEC between primal and dual cochains.
///
/// Each blade of the input pairs with one blade of the output, to `±1`. Implemented for every
/// map cell between kinds of one algebra that pair in one of the two ways; a map between other
/// kinds of equal size (a kind to the metric dual of another orientation) solves only directly,
/// by [`SparseMap::solve_direct`](super::SparseMap::solve_direct). Code generic over the cells
/// names it as a bound, `C: CellPairing` (the generated tables decide it for concrete kinds).
#[diagnostic::on_unimplemented(
    message = "`{Self}` has no pairing of its output with its input",
    note = "conjugate gradients and `eigh` need a map cell of a kind to itself (`K <- K`) or to its complement (a correlation, `C(K) <- K`); `solve_direct` solves any square cells"
)]
pub trait CellPairing: Extensor {
    /// For input blade `i`: the output blade it pairs with, and whether the pairing is `−1`.
    fn pair(i: usize) -> (usize, bool);
}

/// The two pairings, chosen by whether the kinds are the same (an implementation detail of
/// [`CellPairing`]).
#[doc(hidden)]
pub trait PairStep<In: Kind, Out: Kind> {
    /// As [`CellPairing::pair`].
    fn pair(i: usize) -> (usize, bool);
}

impl<K: Kind> PairStep<K, K> for True {
    #[inline]
    fn pair(i: usize) -> (usize, bool) {
        (i, false)
    }
}

impl<In: Complement<Out = Out>, Out: Kind> PairStep<In, Out> for False {
    #[inline]
    fn pair(i: usize) -> (usize, bool) {
        In::PAIRS[i]
    }
}

impl<C, A> CellPairing for C
where
    C: Extensor<Slots = (A,)>,
    A: Kind,
    C::Kind: KindEq<A>,
    <C::Kind as KindEq<A>>::Out: PairStep<A, C::Kind>,
{
    #[inline]
    fn pair(i: usize) -> (usize, bool) {
        <<C::Kind as KindEq<A>>::Out as PairStep<A, C::Kind>>::pair(i)
    }
}
