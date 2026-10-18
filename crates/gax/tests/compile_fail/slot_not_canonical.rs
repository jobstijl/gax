// Slots are written with the bare kind name: `Point`, not `Point<(), f32>`.
use gax::pga3d::Point;

fn main() {
    let _m: Point<(Point<(), f32>,), f64> = todo!();
}
