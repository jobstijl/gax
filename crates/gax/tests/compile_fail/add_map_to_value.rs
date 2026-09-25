// A map and a value are different types: they cannot be added.
use gax::pga3d::Point;

fn main() {
    let p = Point::new(1.0, 2.0, 3.0, 1.0);
    let _ = Point::slot() + p;
}
