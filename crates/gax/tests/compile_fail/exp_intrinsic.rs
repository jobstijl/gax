// The exponential takes a generator, which is extrinsic: a join of two points is intrinsic and
// has none (ADR-051).
use gax::pga3d::Point;

fn main() {
    let axis = Point::xyz(0.0, 0.0, 0.0) & Point::xyz(0.0, 0.0, 1.0);
    let _ = axis.exp();
}
