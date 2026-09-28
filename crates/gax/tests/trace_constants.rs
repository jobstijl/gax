//! Work on constants happens at trace time: a solver on a fixed matrix (its pivoting compares
//! values, `all_lt`) and arithmetic on named float constants fold, so the kernel keeps only
//! the work on its inputs.

#![cfg(all(feature = "trace", feature = "pga3d"))]

use gax::Coef;
use gax::pga3d::Point;
use gax::trace::{Sym, Tracer};

/// A colour matrix (not dyadic: named float constants), as a map on points that keeps the
/// weight.
fn fixed_map() -> Point<(Point,), Sym> {
    let w = Sym::from_f64;
    let z = Sym::zero();
    Point::from_coeffs([
        [w(0.8), w(0.1), w(0.1), z],
        [w(0.05), w(0.9), w(0.05), z],
        [w(0.1), w(0.2), w(0.7), z],
        [z, z, z, Sym::one()],
    ])
}

#[test]
fn a_fixed_matrix_and_its_inverse_fold_to_constants() {
    let mut t = Tracer::new();
    t.kernel("outset", |p: Point<(), Sym>| fixed_map().inverse().of(p));
    let r = &t.reports()[0];
    // Only the application: 3 rows of 3 products, no reciprocals left over from the solve.
    assert_eq!((r.cost.divs, r.cost.calls), (0, 0), "{:?}", r.cost);
    assert!(r.cost.muls <= 9, "{:?}", r.cost);
}
