//! Transposes of maps through a pairing: the adjugate (through the regressive product, no
//! metric) and the adjoint (through the scalar product, the metric). ADR-014, amended.
//!
//! A map `t: B <- A` has no transpose of its own: a transpose identifies each space with its
//! dual, and which identification is meant is a choice of pairing. These two are the ones
//! numga names, and both are a constant table per kind, so the transpose is a signed
//! rearrangement of the coefficients (the adjoint a sparse rational one), exact on every
//! coefficient type and defined for singular maps: it is not an inverse.
//!
//! * [`Adjugate`]: `t.adjugate()` maps the complement of `B` to the complement of `A`
//!   ([`Complement`]), with `(t.adjugate().of(c) & a)₀ == (c & t.of(a))₀`. Metric free, so it
//!   exists in PGA. For an outermorphism it is the classical adjugate, the determinant times the
//!   inverse of the outermorphism on the complementary grade.
//! * [`Adjoint`]: `t.adjoint()` maps `B` back to `A`, with
//!   `t.adjoint().of(b).scalar_product(a) == b.scalar_product(t.of(a))`. It needs the scalar
//!   product to be non-degenerate on both kinds ([`Reciprocal`]), so not on a PGA kind holding
//!   `e0`. A versor's sandwich has its inverse sandwich as its adjoint.
//!
//! ```
//! use gax::ApproxEq;
//! use gax::pga3d::{Plane, Point};
//! // A central projection onto the floor: singular, so no inverse, but an adjugate, the map
//! // on planes that incidence carries back.
//! let eye = Point::<(), f64>::xyz(0.0, 0.0, 5.0);
//! let floor = Plane::<(), f64>::new(0.0, 0.0, 1.0, 0.0);
//! let projection: Point<(Point,), f64> = (eye & Point::slot()) ^ floor;
//! let back: Plane<(Plane,), f64> = projection.adjugate();
//! let (l, p) = (Plane::new(1.0, 2.0, 0.5, -1.0), Point::xyz(0.3, -0.7, 2.0));
//! assert!((back.of(l) & p).approx_eq(&(l & projection.of(p)), 1e-12));
//! ```

use crate::calculus::Reciprocal;
use crate::cast::Complement;
use crate::coef::Coef;
use crate::kind::{Extensor, Kind};

/// The adjugate of a map `B <- A`: the map `C(A) <- C(B)` between the complement kinds that
/// incidence carries over, `(t.adjugate().of(c) & a)₀ == (c & t.of(a))₀` (the scalar part of
/// the regressive product, the complement on the left). numga's `adjugate`.
///
/// It needs no metric and does not depend on the scale of the pseudoscalar, so it exists in
/// degenerate algebras and for singular maps. It reverses composition,
/// `(t ∘ s).adjugate() == s.adjugate() ∘ t.adjugate()`, and fixes the identity. Applied twice
/// it gives `t` back up to the sign of swapping the pairing, `(−1)^(k(n−k))` per blade of
/// grade `k` in `n` dimensions: in even dimensions the grade involution on both sides (in
/// PGA3D the adjugate of the adjugate of a `Point <- Line` is its negative), in odd ones
/// nothing. A map into the complement of its own input (a correlation, `Point <- Plane`) has an
/// adjugate of its own type.
///
/// Implemented for every map with one slot whose input and output kinds have declared
/// complements ([`Complement`]); the result's kinds have the opposite orientations.
///
/// ```
/// use gax::ApproxEq;
/// use gax::pga3d::{Line, Point, intrinsic};
/// // Points to the lines joining them to `q`; its adjugate sends lines to planes, so that
/// // pairing the plane with any point is pairing the line with that point's join to `q`.
/// let q = Point::<(), f64>::xyz(1.0, 2.0, 3.0);
/// let through_q: intrinsic::Line<(Point,), f64> = q & Point::slot();
/// let adj = through_q.adjugate();
/// let l: Line<(), f64> = Line::new(0.1, 0.2, -0.3, 0.4, 0.5, -0.6);
/// let p = Point::xyz(-1.0, 0.5, 2.0);
/// assert!((adj.of(l) & p).approx_eq(&(l & through_q.of(p)), 1e-12));
/// ```
#[diagnostic::on_unimplemented(
    message = "`{Self}` has no adjugate",
    note = "the adjugate is defined for maps with one slot whose input and output kinds both have a declared complement kind (`gax::Complement`)"
)]
pub trait Adjugate: Extensor {
    /// The adjugate map, `C(A) <- C(B)`.
    type Output: Extensor<Coef = Self::Coef>;
    /// The adjugate (see the trait).
    fn adjugate(self) -> Self::Output;
}

impl<M, A> Adjugate for M
where
    M: Extensor<Slots = (A,)>,
    A: Complement,
    M::Kind: Complement,
{
    type Output = <A::Out as Kind>::Mv<(<M::Kind as Complement>::Out,), M::Coef>;

    #[inline]
    fn adjugate(self) -> Self::Output {
        // adj = P_A tᵀ P_Bᵀ with the pairings P signed permutations: the coefficient of input
        // blade l in output m moves to output C(l), input C(m), signed by both pairings.
        let c = self.coeffs().as_ref();
        let (pa, pb) = (A::PAIRS, <M::Kind as Complement>::PAIRS);
        let mut out = <A::Out as Kind>::arr_from_fn(|_| {
            <<M::Kind as Complement>::Out as Kind>::arr_from_fn(|_| M::Coef::zero())
        });
        for (l, &(k, na)) in pa.iter().enumerate() {
            for (m, &(j, nb)) in pb.iter().enumerate() {
                let x = c[m].as_ref()[l];
                out.as_mut()[k].as_mut()[j] = if na == nb { x } else { -x };
            }
        }
        Extensor::from_coeffs(out)
    }
}

/// The adjoint of a map `B <- A` through the scalar product: the map `A <- B` with
/// `t.adjoint().of(b).scalar_product(a) == b.scalar_product(t.of(a))`, the adjoint of
/// Hestenes and Sobczyk. numga's `adjoint`.
///
/// Each kind is paired with itself by the scalar product, the scalar part of the geometric
/// product, so the adjoint exists where that pairing is invertible on both kinds
/// ([`Reciprocal`]): not on a PGA kind holding `e0`, which is a compile error (use the
/// metric-free [`Adjugate`] there). It reverses composition, fixes the identity and is an
/// involution; a unit versor's sandwich has its inverse sandwich as its adjoint, and a map
/// into its own input is symmetric when it equals its adjoint.
///
/// ```
/// use gax::ApproxEq;
/// use gax::vga3d::{Rotor, Vector};
/// let r = Rotor::<(), f64>::rotation(Vector::new(0.0, 0.0, 1.0), 0.8);
/// let turn: Vector<(Vector,), f64> = r >> Vector::slot();
/// let back: Vector<(Vector,), f64> = r.into_inner().reverse() >> Vector::slot();
/// assert!(turn.adjoint().approx_eq(&back, 1e-12));
/// ```
#[diagnostic::on_unimplemented(
    message = "`{Self}` has no adjoint",
    note = "the adjoint is defined for maps with one slot on whose input and output kinds the scalar product is non-degenerate (`gax::Reciprocal`); in PGA a kind holding `e0` has none, and `adjugate` is the metric-free transpose"
)]
pub trait Adjoint: Extensor {
    /// The adjoint map, `A <- B`.
    type Output: Extensor<Coef = Self::Coef>;
    /// The adjoint (see the trait).
    fn adjoint(self) -> Self::Output;
}

impl<M, A> Adjoint for M
where
    M: Extensor<Slots = (A,)>,
    A: Reciprocal,
    M::Kind: Reciprocal,
{
    type Output = A::Mv<(M::Kind,), M::Coef>;

    #[inline]
    fn adjoint(self) -> Self::Output {
        // adj = G_A⁻¹ tᵀ G_B, with G the Gram matrices of the scalar product (symmetric).
        let c = self.coeffs().as_ref();
        let scaled = |x: M::Coef, num: i64, den: i64| match (num, den) {
            (1, 1) => x,
            (-1, 1) => -x,
            (_, 1) => x * M::Coef::from_i64(num),
            _ => x * M::Coef::from_ratio(num, den),
        };
        Extensor::from_coeffs(A::arr_from_fn(|k| {
            <M::Kind as Kind>::arr_from_fn(|j| {
                let mut acc: Option<M::Coef> = None;
                for &(l, num, den) in A::FRAME[k] {
                    for &(m, g) in <M::Kind as Reciprocal>::GRAM[j] {
                        let term = scaled(c[m].as_ref()[l], num * g, den);
                        acc = Some(acc.map_or(term, |a| a + term));
                    }
                }
                acc.unwrap_or_else(M::Coef::zero)
            })
        }))
    }
}
