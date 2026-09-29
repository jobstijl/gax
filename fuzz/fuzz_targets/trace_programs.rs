//! The tracer against the generic code on arbitrary programs and inputs: a traced kernel must
//! compute what the code it was traced from computes (`crates/gax/tests/trace_programs.rs` runs
//! the same check on a fixed sample).
#![no_main]

#[path = "../../crates/gax/tests/support/random_program.rs"]
mod random_program;

use gax::pga3d::{Line, Motor, Point};
use gax::trace::{Sym, Tracer};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // 22 inputs (one byte each, in [-2, 2]), then up to 6 operations.
    if data.len() < 25 {
        return;
    }
    let (x, ops) = data.split_at(22);
    let ops = &ops[..ops.len().min(18)];
    let x: [f64; 22] = core::array::from_fn(|k| f64::from(x[k] as i8) / 64.0);
    let mut t = Tracer::new();
    let o = ops.to_vec();
    t.kernel(
        "p",
        move |m: Motor<(), Sym>, p: Point<(), Sym>, q: Point<(), Sym>, l: Line<(), Sym>| {
            random_program::run(&o, m, p, q, l)
        },
    );
    let got = t.eval("p", &[&x[..8], &x[8..12], &x[12..16], &x[16..]]);
    assert!(
        random_program::agrees(ops, &x, &got, 1e-7),
        "traced {got:?} for {ops:?} at {x:?}"
    );
});
