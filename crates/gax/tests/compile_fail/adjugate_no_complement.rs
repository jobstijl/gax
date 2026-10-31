// A PGA3D rotor's blades `1, e23, e31, e12` have complements `e0123, e01, e02, e03`, which no
// declared kind holds exactly: no adjugate (ADR-053).
use gax::pga3d::Rotor;

fn main() {
    let m = Rotor::<(Rotor,), f64>::zero();
    let _ = m.adjugate();
}
