// A map on twists (extrinsic lines) does not take a join of points (an intrinsic line): the
// value must be of the slot's orientation (ADR-051).
use gax::pga3d::{Line, Point};

fn main() {
    let map: Line<(Line,)> = Line::slot();
    let joined = Point::xyz(0.0, 0.0, 0.0) & Point::xyz(0.0, 0.0, 1.0);
    let _ = map.of(joined);
}
