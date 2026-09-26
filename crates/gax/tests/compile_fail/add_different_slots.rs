// Maps with different slot lists are different types: they cannot be added.
use gax::pga3d::{Line, Plane, Point};

fn main() {
    let f: Point<(Point,)> = Point::slot();
    let g: Point<(Line,)> = Line::slot() ^ Plane::new(0.0, 0.0, 1.0, 0.0);
    let _ = f + g;
}
