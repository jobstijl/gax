//! Inclusion of a subalgebra's values by generator names (numga's `from_subalgebra`, ADR-057).
//!
//! An algebra contains another when every generator of the smaller one is among its own, by
//! name and with the same inner products: `pga2d`'s `e0, e1, e2` are `pga3d`'s, and `vga3d`'s
//! `e1, e2, e3` are STA's space. A value of the smaller algebra then has a place in the larger,
//! and which place follows from its orientation (ADR-051):
//!
//! * an **extrinsic** value keeps its blades: a line `a e1 + b e2 + c e0` of the plane is the
//!   plane of space with the same equation, a rotor the rotor about the same axis;
//! * an **intrinsic** value keeps its complement, `J(ι(J⁻¹ x))` with `J` each algebra's
//!   `dual`: a point of the plane is the point of space at `z = 0`, with its weight, and the
//!   join of two points of the plane the line of space through them.
//!
//! So joins and pairings carry over, unlike under the homomorphisms (`From`, which send a point
//! of the plane to the vertical line through it). The generated algebras implement
//! [`FromSubalgebra`] for every pair of standard algebras where one holds the other's
//! generators, and their kinds forward it as inherent methods, `pga3d::Point::from_subalgebra(p)`
//! and `p.include()`. The generator proves, exactly and for every pair of kinds, that the
//! inclusion `ι` keeps
//!
//! * every product but the join of two extrinsic values (`*`, `^`, `|`, `⌋`, `⌊`, the scalar
//!   product, the commutator and anticommutator), so the meet;
//! * the join of two intrinsic values, `ι(a & b) == ι(a) & ι(b)`;
//! * an extrinsic value's `*`, `^`, `&` and `⌋` with an intrinsic one on its right, so the
//!   pairing `plane & point`; with the intrinsic value on the left the sign may differ;
//! * every versor's `>>` on values of either orientation, and `dual` of an extrinsic value and
//!   `undual` of an intrinsic one.
//!
//! An inclusion is a signed reordering of coefficients: exact on every coefficient type. It
//! carries a map's output and keeps its slots, so `pga3d::Point::from_subalgebra(pga2d::Point::slot())`
//! is the inclusion itself as a map.
//!
//! ```
//! use gax::{pga2d, pga3d};
//! use gax::ApproxEq;
//!
//! let (p, q) = (pga2d::Point::xy(1.0, 2.0), pga2d::Point::xy(-0.5, 0.25));
//! // A point of the plane is the point of space at z = 0.
//! let p3: pga3d::Point = p.include();
//! assert!(p3.approx_eq(&pga3d::Point::xyz(1.0, 2.0, 0.0), 1e-15));
//! // The join of two points carries over: the line of space through both.
//! let line = pga3d::intrinsic::Line::from_subalgebra(p & q);
//! assert!(line.approx_eq(&(p3 & q.include::<pga3d::Point>()), 1e-15));
//! // A line of the plane is the plane of space with its equation, and the pairing carries over.
//! let l = pga2d::Line::new(0.3, -1.0, 0.5);
//! let plane: pga3d::Plane = l.include();
//! assert!(((l & p).s() - (plane & p3).s()).abs() < 1e-15);
//! ```

/// A conversion from a value of a subalgebra `A`, by generator names: extrinsic values keep
/// their blades, intrinsic ones their complements (see the [module](self)).
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not where `{A}` goes in this algebra",
    label = "no inclusion into `{Self}`",
    note = "an inclusion needs every generator of the value's algebra in the target's by name, with the same metric; an extrinsic value lands in the kind holding its blades, an intrinsic one in the kind holding its complements, in its own orientation (a `pga2d::Point` in `pga3d::Point`, a `pga2d::Line` in `pga3d::Plane`)",
    note = "`x.include()` with the result's type annotated, or `K::from_subalgebra(x)`; the table is in the guide, \"Between algebras\""
)]
pub trait FromSubalgebra<A>: Sized {
    /// The value `a` of a subalgebra in this algebra.
    fn from_subalgebra(a: A) -> Self;
}
