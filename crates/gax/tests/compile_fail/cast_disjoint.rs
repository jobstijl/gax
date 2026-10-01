// A point and a plane share no blade: a cast between them would always be zero.
use gax::pga3d::{Plane, Point};

fn main() {
    let p = Point::xyz(1.0, 2.0, 3.0);
    let _ = p.cast::<Plane>();
}
