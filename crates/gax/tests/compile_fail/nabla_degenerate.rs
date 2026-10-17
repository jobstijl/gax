// PGA's planes hold the null direction e0: they have no reciprocal frame, so no derivative
// taken back to a plane (gradient gives it as a form).
use gax::calculus::nabla;
use gax::pga3d::Plane;

fn main() {
    let x = Plane::<(), f64>::new(1.0, 0.0, 0.0, 0.0);
    let _ = nabla(|x| x, x);
}
