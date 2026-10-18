// The products are generated for dense kinds only (ADR-044): a structured map takes part in a
// product through its dense form, `inertia.to_dense()`.
use gax::pga3d::{Line, PrincipalInertia};

fn main() {
    let inertia = PrincipalInertia::new(2.0, [0.5, 0.75, 1.0]);
    let _ = Line::<(), f64>::slot() & inertia;
}
