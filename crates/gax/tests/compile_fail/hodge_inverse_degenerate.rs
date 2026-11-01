// The Hodge dual of a degenerate algebra vanishes on the null blades, so it has no inverse.
use gax::pga3d::Plane;

fn main() {
    let p = Plane::new(1.0, 0.0, 0.0, 2.0);
    let _ = p.hodge().hodge_inverse();
}
