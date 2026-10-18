// A map on points cannot be applied to a plane.
use gax::pga3d::{Plane, Point};

fn main() {
    let map = Point::new(0.0, 0.0, 1.0, 1.0) & Point::slot();
    let plane = Plane::new(0.0, 0.0, 1.0, 0.0);
    let _ = map.of(plane);
}
