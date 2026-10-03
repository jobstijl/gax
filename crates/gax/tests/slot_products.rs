//! Products of maps large enough to run once per pair of slot entries
//! (`gax::slots::by_entries`, above `SLOT_UNROLL_MAX`) agree with the products of the values
//! they are applied to, with the slots in order: `(a * X).of(x) = a * x`, `(X * b).of(x) = x * b`,
//! and `(X * Y).of(x).of(y) = x * y`.

#![cfg(feature = "csta")]

#[path = "support/rng.rs"]
mod rng;
use rng::Rng;

use gax::csta::{Bivector, Even, Vector};

fn close(a: &[f64], b: &[f64]) -> bool {
    let scale = a.iter().chain(b).fold(1.0f64, |m, x| m.max(x.abs()));
    a.iter().zip(b).all(|(x, y)| (x - y).abs() <= 1e-12 * scale)
}

#[test]
fn large_slot_products_match_the_value_products() {
    let mut rng = Rng(0x0005_1075);
    for _ in 0..20 {
        let p = Even::<(), f64>::from_coeffs(core::array::from_fn(|_| rng.next_f64()));
        let b = Bivector::<(), f64>::from_coeffs(core::array::from_fn(|_| rng.next_f64()));
        let c = Bivector::<(), f64>::from_coeffs(core::array::from_fn(|_| rng.next_f64()));
        let v = Vector::<(), f64>::from_coeffs(core::array::from_fn(|_| rng.next_f64()));
        // One slot on either side, and a sandwich through two products.
        let left = p * Bivector::<(), f64>::slot();
        assert!(close(&left.of(b).c, &(p * b).c));
        let right = Bivector::<(), f64>::slot() * p;
        assert!(close(&right.of(b).c, &(b * p).c));
        let conj = p * Bivector::<(), f64>::slot() * p.reverse();
        assert!(close(&conj.of(b).c, &(p * b * p.reverse()).c));
        let refl = v * Bivector::<(), f64>::slot() * v;
        assert!(close(&refl.of(b).c, &(v * b * v).c));
        // Two slots, first from the left operand.
        let both = Bivector::<(), f64>::slot() * Bivector::<(), f64>::slot();
        assert!(close(&both.of(b).of(c).c, &(b * c).c));
        assert!(!close(&both.of(b).of(c).c, &(c * b).c));
        // Products of maps with maps keep both slot lists.
        let pair = (p * Bivector::<(), f64>::slot()) * (Bivector::<(), f64>::slot() * p);
        assert!(close(&pair.of(b).of(c).c, &((p * b) * (c * p)).c));
    }
}
