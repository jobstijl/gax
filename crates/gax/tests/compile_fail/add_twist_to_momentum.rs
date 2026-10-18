// A twist (a turn about a line, extrinsic) and a momentum (a join of a point with a velocity,
// intrinsic) are both lines, but of opposite orientations: they cannot be added (ADR-051).
use gax::pga3d::{Line, Point};

fn main() {
    let twist = Line::new(0.0, 0.0, 0.0, 1.0, 0.0, 0.0);
    let p = Point::xyz(0.0, 1.0, 0.0);
    let momentum = p & p.commutator(twist);
    let _ = twist + momentum;
}
