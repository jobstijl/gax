//! `Motor::between` traces, and two tracer bugs it found stay fixed:
//! * the reciprocal of an exactly zero part (the rotation of a pure translation, a branch the
//!   select discards) recorded the relation `r 0 = 1`, which made every expression equal to
//!   zero, so the traced kernel returned the zero motor and still verified;
//! * the Gröbner basis of the relations could grow without bound inside one reduction (the
//!   S-pair cap does not limit it), and tracing the planes' kernel never finished.
//!
//! `examples/traced` checks the kernels numerically and validates their WGSL.

#![cfg(all(feature = "trace", feature = "pga3d"))]

use gax::pga3d::{Line, Motor, Plane, Point};
use gax::trace::{Sym, Tracer};

#[test]
fn between_points_is_a_translation_not_zero() {
    let mut t = Tracer::new();
    t.kernel("between_points", |a: Point<(), Sym>, b: Point<(), Sym>| {
        Motor::between(a, b).into_inner()
    });
    let c = t.reports()[0].cost;
    // Unitize both points and halve the difference: no calls, but real arithmetic.
    assert!(c.adds > 0 && c.muls > 0 && c.calls == 0, "{c:?}");
}

#[test]
fn between_planes_and_lines_trace() {
    let mut t = Tracer::new();
    t.kernel("between_planes", |a: Plane<(), Sym>, b: Plane<(), Sym>| {
        Motor::between(a, b).into_inner()
    });
    t.kernel("between_lines", |a: Line<(), Sym>, b: Line<(), Sym>| {
        Motor::between(a, b).into_inner()
    });
    for r in t.reports() {
        assert!(r.cost.muls > 0, "{}: {:?}", r.name, r.cost);
    }
}
