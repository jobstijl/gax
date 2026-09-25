//! Test harness: a deliberately naive dense multivector oracle and random inputs.
//!
//! The oracle stores all `2^n` blade coefficients in `f64` and multiplies every pair of
//! blades through the exact blade tables of `gax-gen` (which are themselves checked against
//! an independent rational oracle in `gax-gen/tests/oracle.rs`).
#![allow(dead_code)]

use gax::{Extensor, Kind, Slots};
use gax_gen::algebra::Algebra;
use gax_gen::spec::AlgebraSpec;
use gax_gen::table::{BinOp, UnOp, blade_binop, blade_unop};

/// A small deterministic generator (xorshift), so failures reproduce.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    pub fn next_f64(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }
}

pub struct Oracle {
    pub alg: Algebra,
}

pub type Dense = Vec<f64>;

impl Oracle {
    pub fn from_spec(src: &str) -> Oracle {
        Oracle {
            alg: AlgebraSpec::parse(src).expect("spec").algebra,
        }
    }

    /// Dense coefficients of a value of any generated kind.
    pub fn dense<M: Extensor<Slots = (), Coef = f64>>(&self, m: &M) -> Dense {
        let mut d = vec![0.0; self.alg.blade_count()];
        for (b, c) in <M::Kind as Kind>::BLADES.iter().zip(m.coeffs().as_ref()) {
            let (mask, sign) = self.alg.parse_blade(b).unwrap();
            d[mask as usize] += sign as f64 * c;
        }
        d
    }

    pub fn binop(&self, op: BinOp, a: &Dense, b: &Dense) -> Dense {
        let mut out = vec![0.0; a.len()];
        for (i, &x) in a.iter().enumerate() {
            for (j, &y) in b.iter().enumerate() {
                if x == 0.0 || y == 0.0 {
                    continue;
                }
                for (m, c) in blade_binop(&self.alg, op, i as u32, j as u32) {
                    out[m as usize] += c as f64 * x * y;
                }
            }
        }
        out
    }

    pub fn unop(&self, op: UnOp, a: &Dense) -> Dense {
        let mut out = vec![0.0; a.len()];
        for (i, &x) in a.iter().enumerate() {
            let (m, s) = blade_unop(&self.alg, op, i as u32);
            out[m as usize] += s as f64 * x;
        }
        out
    }
}

/// A random value of a generated kind.
pub fn random<M: Extensor<Slots = (), Coef = f64>>(rng: &mut Rng) -> M {
    M::from_coeffs(<M::Kind as Kind>::arr_from_fn(|_| rng.next_f64()))
}

/// A random map or form of a generated kind with arbitrary slots.
pub fn random_map<M: Extensor<Coef = f64>>(rng: &mut Rng) -> M {
    M::from_coeffs(<M::Kind as Kind>::arr_from_fn(|_| {
        <M::Slots as Slots>::from_flat(&mut |_| rng.next_f64(), 0)
    }))
}

/// Flattened coefficients of any extensor.
pub fn flat<M: Extensor<Coef = f64>>(m: &M) -> Vec<f64> {
    let mut v = Vec::new();
    for col in m.coeffs().as_ref() {
        for k in 0..<M::Slots as Slots>::SIZE {
            v.push(<M::Slots as Slots>::get_flat(col, k));
        }
    }
    v
}

#[track_caller]
pub fn assert_close(a: &[f64], b: &[f64], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: length");
    let scale = a.iter().chain(b).fold(1.0f64, |m, x| m.max(x.abs()));
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        // Inputs are in [-1, 1]; products of a few such numbers accumulate rounding of order
        // (number of terms) * eps * scale. 1e-12 relative is far above that and far below any
        // real error (which would be O(1)).
        assert!(
            (x - y).abs() <= 1e-12 * scale,
            "{what}: coefficient {i}: {x} vs {y}\n{a:?}\n{b:?}"
        );
    }
}
