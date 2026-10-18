//! numga's `coating/core.py`, the time domain: Maxwell's equations on a line of nodes along
//! `z`, stepped by a leapfrog that keeps the field's energy.

use crate::coating::{B, Constitutive, Excitation, t, z};
use gax::Orientation;
use gax::field::{Field, SparseMap};
use gax::sta::{Bivector, Scalar, intrinsic, oriented};

/// `t · (z ∧ X)`: a plane swept along the line, as the observer `t` sees it, a field's or an
/// excitation's. With a difference along `z` it is the curl in both Maxwell equations.
pub fn swept<O: Orientation>(x: oriented::Bivector<(), f64, O>) -> oriented::Bivector<(), f64, O> {
    t() | (z() ^ x)
}

/// numga's regressive product of complementary grades (with numga's metric dual): minus
/// gax's metric-free one in spacetime (see `electromagnetism_constitutive`).
pub fn pair(a: B, b: Excitation) -> Scalar<(), f64> {
    -(a & b)
}

/// An inverse response: an excitation in, its field out.
type InverseResponse = Bivector<(intrinsic::Bivector,), f64>;

/// A periodic line of cells along `z`. Electric planes sit on the nodes, magnetic planes
/// halfway between each node and the next.
pub struct Line {
    response: Field<Constitutive>,
    inverse: Field<InverseResponse>,
    between: Constitutive,
    difference: SparseMap<Scalar<(), f64>>,
    spacing: f64,
}

impl Line {
    /// The material at each node, and the one material between nodes, which only meets
    /// magnetic planes. The difference reads, at each node, the next node's value less its
    /// own: from the electric planes on either side of a magnetic plane.
    pub fn from_chi(nodes: Field<Constitutive>, between: Constitutive, spacing: f64) -> Line {
        let count = nodes.len();
        let s = Scalar::new;
        let cells = (0..count).flat_map(|_| [s(1.0), s(-1.0)]).collect();
        let rows = (0..count).flat_map(|i| [i, i]).collect();
        let cols = (0..count).flat_map(|i| [(i + 1) % count, i]).collect();
        Line {
            inverse: nodes.map(|chi| chi.inverse()),
            response: nodes,
            between,
            difference: SparseMap::new(cells, rows, cols, (count, count)),
            spacing,
        }
    }

    /// The field energy in each cell; magnetic planes pair with their excitation to minus
    /// their energy.
    pub fn energy(&self, electric: &Field<B>, magnetic: &Field<B>) -> Field<Scalar<(), f64>> {
        Field::from_fn(electric.len(), |i| {
            let (e, m) = (electric.site(i), magnetic.site(i));
            let electric_part = pair(e, self.response.site(i).of(e));
            (electric_part - pair(m, self.between.of(m))) * (self.spacing / 2.0)
        })
    }

    /// Both exterior Maxwell equations, `d ∧ F = 0` and `d ∧ G = 0`, as a leapfrog, one
    /// state per step, without end.
    ///
    /// The time part of `d ∧ X` is `t ∧ (dX / dt)`; contracting with the observer `t` undoes
    /// that wedge, so each update is minus [`swept`] of the difference along `z`. Magnetic
    /// planes move with the electric ones on either side; the excitation's spatial planes
    /// move with the excitation of the magnetic planes on either side, and the inverse
    /// response turns them back into electric planes.
    pub fn evolve(
        &self,
        electric: Field<B>,
        magnetic: Field<B>,
        interval: f64,
    ) -> impl Iterator<Item = (Field<B>, Field<B>)> + '_ {
        let rate = interval / self.spacing;
        // The reverse runs each difference back, from the magnetic planes to the electric
        // planes between them: the adjoint pair under which the leapfrog keeps its energy.
        let back = self.difference.reverse();
        let step = move |(electric, magnetic): &(Field<B>, Field<B>)| {
            let magnetic = magnetic - (&self.difference * electric).map(swept).scaled(rate);
            let excitation = magnetic.map(|m| self.between.of(m));
            let displacement = (&back * &excitation).map(swept).scaled(rate);
            let electric = electric + self.inverse.zip_with(&displacement, |i, d| i.of(d));
            Some((electric, magnetic))
        };
        std::iter::successors(Some((electric, magnetic)), step).skip(1)
    }
}
