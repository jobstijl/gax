// The scalar product is degenerate on PGA3D's planes (`e0 e0 = 0`): no adjoint, the
// adjugate is the transpose there (ADR-053).
use gax::pga3d::Plane;

fn main() {
    let m = Plane::<(Plane,), f64>::zero();
    let _ = m.adjoint();
}
