// The outer product of two points (grade 3 + 3 > 4) is identically zero: no such product.
use gax::pga3d::Point;

fn main() {
    let p = Point::new(1.0, 2.0, 3.0, 1.0);
    let _ = p ^ p;
}
