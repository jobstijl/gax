// Solving needs a map between kinds of the same size; `lstsq` solves any shape.
use gax::pga3d::{Line, Plane};

fn f(m: Plane<(Line,), f64>, b: Plane<(), f64>) {
    let _ = m.solve(b);
}

fn main() {}
