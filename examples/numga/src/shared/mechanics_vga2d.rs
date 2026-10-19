//! What the plane-flow examples share in VGA2D: the axes and the derivative of a map on
//! directions.
//!
//! numga writes the derivative as `(Vector * map(Vector)).contract(1, 2)`; gax has no
//! contraction of two input slots, so here it is the sum `Σ e_i map(e_i)` over the orthonormal
//! basis, which is the same in VGA2D.

use core::iter::Sum;
use core::ops::Mul;
use gax::Of;
use gax::vga2d::{Rotor, Vector};

/// A direction of the plane.
pub type V = Vector;
/// A map on directions, `Vector <- Vector`: a velocity gradient.
pub type Map = Vector<(Vector,)>;
/// An even element: a scalar plus a bivector, a complex number.
pub type Even = Rotor;

/// The orthonormal axes x and y.
pub const AXES: [V; 2] = [Vector::new(1.0, 0.0), Vector::new(0.0, 1.0)];

/// The derivative of a map whose slot takes a direction, `Σ e_i map(e_i)`. Of a velocity
/// gradient it is an even element, the divergence plus the vorticity; of a potential's gradient
/// map, a form, it is the gradient vector.
pub fn derivative<M, X, D>(map: M) -> D
where
    M: Copy + Of<V, Output = X>,
    V: Mul<X, Output = D>,
    D: Sum,
{
    AXES.iter().map(|e| *e * map.of(*e)).sum()
}
