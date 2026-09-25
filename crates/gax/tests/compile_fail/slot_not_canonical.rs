// Slots are written with the bare kind name: `Point`, not `Point<(), f64>`.
use gax::pga3d::Point;

fn main() {
    let _m: Point<(Point<(), f64>,), f64> = todo!();
}
