//! numga's origami `core.py`: crease construction, polygon splitting and rigid folds of a
//! sheet of paper, in PGA3D.

use gax::Unit;
use gax::pga3d::{Line, Motor, Plane, Point, Pseudoscalar};
use std::ops::Range;

pub type P = Point<(), f64>;
pub type Pl = Plane<(), f64>;
pub type L = Line<(), f64>;
pub type M = Unit<Motor<(), f64>>;

/// The perpendicular bisector plane of two equally weighted finite points: the inner product of
/// their join with their sum, the plane through their midpoint orthogonal to the line.
pub fn point_bisector(first: P, second: P) -> Pl {
    ((first & second) | (first + second))
        .normalized()
        .into_inner()
}

/// The external angle bisector of two intersecting oriented lines in a sheet: the plane through
/// their crossing orthogonal to the sum of the unit lines. Folding along it lays one line on the
/// other, reversed.
pub fn line_bisector(first: L, second: L, sheet: Pl) -> Pl {
    let (first, second) = (
        first.normalized().into_inner(),
        second.normalized().into_inner(),
    );
    let crossing = (sheet | first) ^ second;
    ((first + second) | crossing).normalized().into_inner()
}

/// Which side of a unit plane a point lies on, beyond a tolerance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Side {
    Negative,
    On,
    Positive,
}

/// A corner, and the crossing of its outgoing edge with a cutting plane, each with its side.
type Slots = [Option<(P, Side)>; 2];

/// The part of a face a piece keeps: the corners alone, or every slot off one side.
enum Piece {
    Whole,
    Off(Side),
}

impl Piece {
    /// The points a piece keeps of a corner's slots.
    fn kept(&self, slots: &Slots) -> Vec<P> {
        let keeps = |slot: usize, s: Side| match self {
            Piece::Whole => slot == 0,
            Piece::Off(off) => s != *off,
        };
        (0..2)
            .filter_map(|k| slots[k].filter(|(_, s)| keeps(k, *s)).map(|(p, _)| p))
            .collect()
    }
}

/// The side of a unit plane a point lies on.
fn side(plane: Pl, point: P, tolerance: f64) -> Side {
    let distance = (plane & point).s();
    if distance < -tolerance {
        Side::Negative
    } else if distance > tolerance {
        Side::Positive
    } else {
        Side::On
    }
}

/// The paper: polygon corners in one list, the faces' corner counts delimiting them, each
/// face's offset (an infinitesimal separation between coincident layers, a direction measured
/// in layers), and the sheet's plane.
#[derive(Clone, Debug)]
pub struct Paper {
    pub points: Vec<P>,
    pub counts: Vec<usize>,
    pub offsets: Vec<P>,
    pub sheet: Pl,
}

impl Paper {
    /// The corner indices of each face.
    pub fn faces(&self) -> Vec<Range<usize>> {
        let mut start = 0;
        self.counts
            .iter()
            .map(|n| {
                start += n;
                start - n..start
            })
            .collect()
    }

    /// The index of each corner's successor round its face.
    pub fn following(&self) -> Vec<usize> {
        self.faces()
            .into_iter()
            .flat_map(|f| {
                f.clone()
                    .map(move |i| if i + 1 == f.end { f.start } else { i + 1 })
            })
            .collect()
    }

    /// The face of each corner.
    pub fn face_of_corners(&self) -> Vec<usize> {
        (0..self.counts.len())
            .flat_map(|f| std::iter::repeat_n(f, self.counts[f]))
            .collect()
    }

    /// The paper with the selected faces cut by a unit plane, each into its moving piece (on the
    /// plane's negative side) and then its stationary piece; other faces kept whole, in face
    /// order. Splitting duplicates the crease's points, so the two pieces can move apart while
    /// their common edge stays put. Returns the cut paper, which of its faces turn, and the
    /// face of this paper each came from. The tolerance is a distance to the plane that keeps
    /// points already on a crease there despite the round-off of earlier half turns.
    pub fn split(
        &self,
        plane: Pl,
        selected: &[usize],
        tolerance: f64,
    ) -> (Paper, Vec<bool>, Vec<usize>) {
        let slots = self.slots(plane, tolerance);
        let mut cut = Paper {
            points: vec![],
            counts: vec![],
            offsets: vec![],
            sheet: self.sheet,
        };
        let (mut turning, mut parents) = (vec![], vec![]);
        for (index, face) in self.faces().into_iter().enumerate() {
            let pieces = if selected.contains(&index) {
                vec![
                    (true, Piece::Off(Side::Positive)),
                    (false, Piece::Off(Side::Negative)),
                ]
            } else {
                vec![(false, Piece::Whole)]
            };
            for (turns, piece) in pieces {
                let kept: Vec<P> = face.clone().flat_map(|i| piece.kept(&slots[i])).collect();
                cut.counts.push(kept.len());
                cut.points.extend(kept);
                cut.offsets.push(self.offsets[index]);
                turning.push(turns);
                parents.push(index);
            }
        }
        (cut, turning, parents)
    }

    /// Each corner with its side of the plane, and, where its outgoing edge crosses the plane,
    /// the crossing (only crossing edges are met, so parallel ones never divide).
    fn slots(&self, plane: Pl, tolerance: f64) -> Vec<Slots> {
        let next = self.following();
        let sides: Vec<Side> = self
            .points
            .iter()
            .map(|p| side(plane, *p, tolerance))
            .collect();
        (0..self.points.len())
            .map(|i| {
                let (here, there) = (sides[i], sides[next[i]]);
                let crosses = here != there && here != Side::On && there != Side::On;
                let crossing = crosses.then(|| {
                    let cut = ((self.points[i] & self.points[next[i]]) ^ plane).unitized();
                    (cut, side(plane, cut, tolerance))
                });
                [Some((self.points[i], here)), crossing]
            })
            .collect()
    }

    /// The turning faces turned as one batch about their common crease by `angle`; the others
    /// stay put (they turn by zero, the identity motor).
    pub fn folded(&self, hinge: L, turning: &[bool], angle: f64) -> Paper {
        let motors = turns(hinge, turning, angle);
        let corner_faces = self.face_of_corners();
        let points = self
            .points
            .iter()
            .zip(corner_faces)
            .map(|(p, f)| motors[f] >> *p)
            .collect();
        Paper {
            points,
            counts: self.counts.clone(),
            offsets: self.folded_offsets(&motors, turning, angle),
            sheet: self.sheet,
        }
    }

    /// The layer separations carried rigidly through a fold, about a hinge raised so that a
    /// closed stack lands one layer beyond the stationary one (beneath it for a negative fold);
    /// the same rotation carries every separation continuously, reversing their order only as
    /// the paper turns over. The offsets count layers: drawing sets their thickness, and the
    /// creases stay exactly on the mathematical sheet.
    fn folded_offsets(&self, motors: &[M], turning: &[bool], angle: f64) -> Vec<P> {
        let sign = angle.signum();
        let heights = self.offsets.iter().map(|o| (self.sheet & *o).s() * sign);
        let highest = |turns: bool| {
            heights
                .clone()
                .zip(turning)
                .filter(|(_, t)| **t == turns)
                .map(|(h, _)| h)
                .fold(f64::NEG_INFINITY, f64::max)
        };
        let height = (highest(true) + highest(false) + 1.0) / 2.0;
        // The sheet times the pseudoscalar is its normal direction.
        let pivot: P = (self.sheet * Pseudoscalar::new(1.0)).gp(height * sign);
        self.offsets
            .iter()
            .zip(motors)
            .map(|(o, m)| (*m >> (*o - pivot)) + pivot)
            .collect()
    }
}

/// The motor of each face for a fold by `angle` about `hinge`: the turn for the turning faces,
/// the identity for the others.
pub fn turns(hinge: L, turning: &[bool], angle: f64) -> Vec<M> {
    let hinge = hinge.normalized().into_inner();
    let turn = hinge.gp(angle / 2.0).exp();
    let still = hinge.gp(0.0).exp();
    turning
        .iter()
        .map(|t| if *t { turn } else { still })
        .collect()
}
