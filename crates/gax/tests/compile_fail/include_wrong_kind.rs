// A point of the plane is intrinsic: it lands in space's points, not in its lines (that is the
// homomorphism, `From`).
use gax::{pga2d, pga3d};

fn main() {
    let p = pga2d::Point::xy(1.0, 2.0);
    let _: pga3d::intrinsic::Line = p.include();
}
