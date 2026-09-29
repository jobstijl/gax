//! Tracer options that change only how a kernel is written: `fma(false)` emits plain products
//! and sums instead of `mul_add`, and the kernel computes the same program.

#![cfg(all(feature = "trace", feature = "pga3d"))]

use gax::Unit;
use gax::pga3d::{Motor, Point};
use gax::trace::{Sym, Tracer};

#[test]
fn fma_off_writes_plain_products_and_sums() {
    let mut t = Tracer::new();
    t.kernel("fused", |m: Unit<Motor<(), Sym>>, p: Point<(), Sym>| m >> p);
    t.fma(false);
    t.kernel("plain", |m: Unit<Motor<(), Sym>>, p: Point<(), Sym>| m >> p);
    let src = t.source();
    let (fused, plain) = src.split_at(src.find("pub fn plain").expect("the second kernel"));
    assert!(fused.contains("mul_add"));
    assert!(!plain.contains("mul_add"));
    // The same verified program either way.
    let x = [0.3, -0.2, 0.5, 0.7, 0.1, -0.4, 0.25, 0.05];
    let p = [1.0, 2.0, 3.0, 1.0];
    assert_eq!(t.eval("fused", &[&x, &p]), t.eval("plain", &[&x, &p]));
}
