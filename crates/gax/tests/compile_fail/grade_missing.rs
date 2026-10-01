// A point has blades of grade 3 only.
use gax::pga3d::Point;

fn main() {
    let p = Point::xyz(1.0, 2.0, 3.0);
    let _ = p.grade::<1>();
}
