// A structured map binds where its dense form would: an inertia (`Line <- Line`) does not fit a
// slot of kind `Point`.
use gax::pga3d::{Point, PrincipalInertia};

fn main() {
    let inertia = PrincipalInertia::new(2.0, [0.5, 0.75, 1.0]);
    let on_points: Point<(Point,), f64> = Point::slot();
    let _ = on_points.of(inertia);
}
