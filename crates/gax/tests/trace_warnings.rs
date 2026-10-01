//! The tracer warns when a kernel renormalizes a `Unit` argument: tracing assumes it is exactly
//! unit, so the renormalization simplifies to nothing (VERIFY.md, friction 2).

#![cfg(all(feature = "trace", feature = "pga3d"))]

use gax::Unit;
use gax::pga3d::{Motor, Point};
use gax::trace::{Sym, Tracer};

#[test]
fn renormalizing_a_unit_argument_warns() {
    let mut t = Tracer::new();
    t.kernel("drifts", |m: Unit<Motor<(), Sym>>, p: Point<(), Sym>| {
        m.renormalize_fast() >> p
    });
    // A plain motor, renormalized: real work, and no warning.
    t.kernel("repairs", |m: Motor<(), Sym>| {
        Unit::new_unchecked(m).renormalize_fast().into_inner()
    });
    let r = t.reports();
    assert_eq!(r[0].warnings.len(), 1, "{:?}", r[0].warnings);
    assert_eq!(r[1].warnings, Vec::<String>::new());
    assert!(t.source().contains("**Warning:**"));
}
