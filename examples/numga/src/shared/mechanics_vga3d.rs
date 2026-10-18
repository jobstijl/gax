//! What several mechanics examples in VGA3D share: turns in a plane, and two readings of a map
//! on directions, its adjoint through the metric and its curl.
//!
//! numga writes the curl as `(Vector ^ map(Vector)).contract(1, 2)`; gax has no contraction of
//! two input slots, so here it is the sum `Σ e_i ^ map(e_i)` over the orthonormal basis, which
//! is the same in VGA3D. gax has no adjoint as a method either: it is a pairing solve against
//! the metric `Vector | Vector`.

// Each binary that includes this module uses a different part of it.
#![allow(dead_code)]

use core::ops::BitXor;
use gax::vga3d::{Bivector, Rotor, Vector};
use gax::{Of, Orientation, Slots, Unit};

/// A direction of space.
pub type V = Vector<(), f64>;
/// A plane through the origin.
pub type B = Bivector<(), f64>;
/// A rotation: a unit rotor.
pub type Turn = Unit<Rotor<(), f64>>;
/// A map on directions, `Vector <- Vector`.
pub type Map = Vector<(Vector,), f64>;

/// The orthonormal basis.
pub const AXES: [V; 3] = [
    V::new(1.0, 0.0, 0.0),
    V::new(0.0, 1.0, 0.0),
    V::new(0.0, 0.0, 1.0),
];

/// The turn by `angle` in the plane of a unit bivector, from its first vector towards its
/// second.
pub fn turn(plane: B, angle: f64) -> Turn {
    (plane * (-angle / 2.0)).exp()
}

/// The adjoint through the metric: the map `A` with `A(a) | b == a | map(b)`, a pairing solve.
/// A map on directions of either orientation: on area normals (duals of patches, intrinsic) as
/// well as on edges.
pub fn adjoint<O: Orientation>(map: Vector<(Vector<(), f32, O>,), f64, O>) -> Vector<(Vector<(), f32, O>,), f64, O> {
    let open = Vector::<(), f64, O>::oriented_slot;
    let metric = open() | open();
    metric.solve(open() | map)
}

/// The curl of a map whose first slot takes a direction, `Σ e_i ^ map(e_i)`: the bivector whose
/// turning map is the map less its adjoint, zero for a symmetric map. The map's other slots
/// stay open.
pub fn curl<M, S: Slots>(map: M) -> Bivector<S, f64>
where
    M: Copy + Of<V, Output = Vector<S, f64>>,
    V: BitXor<Vector<S, f64>, Output = Bivector<S, f64>>,
{
    AXES.iter().map(|e| *e ^ map.of(*e)).sum()
}
