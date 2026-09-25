// A map from lines (6 coefficients) to planes (4) has no inverse.
use gax::pga3d::{Line, Plane};

fn f(m: Plane<(Line,), f64>) {
    let _ = m.inverse();
}

fn main() {}
