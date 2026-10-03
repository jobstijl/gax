//! `Strict` coefficients give the same bits on scalars and on every SIMD level, without the
//! `deterministic` feature: the per-value form of determinism (docs/numerics.md).

#![cfg(all(feature = "batch", feature = "pga3d", not(feature = "deterministic")))]

#[path = "support/rng.rs"]
mod rng;
use rng::Rng;

use gax::batch::{self, Map};
use gax::pga3d::{Line, Point};
use gax::strict::Strict;
use gax::{Extensor, Real};

/// exp, the motor's action on a point, log, and `exp` of a coefficient.
struct Pipeline;
impl Map for Pipeline {
    type X = Line;
    type Y = Line;
    #[inline(always)]
    fn call<T: Real>(&self, b: Line<(), T>) -> Line<(), T> {
        let m = b.exp();
        let p = m
            >> Point::new(
                T::from_f64(0.3),
                T::from_f64(-1.0),
                T::from_f64(2.0),
                T::one(),
            );
        let k = (b.norm() * T::from_f64(0.5)).exp();
        (gax::Log::log(m) + (p & Point::new(T::zero(), T::zero(), T::zero(), T::one()))).gp(k)
    }
}

fn bits32(m: &Line<(), Strict<f32>>) -> Vec<u32> {
    m.coeffs().iter().map(|x| x.0.to_bits()).collect()
}
fn bits64(m: &Line<(), Strict<f64>>) -> Vec<u64> {
    m.coeffs().iter().map(|x| x.0.to_bits()).collect()
}

#[test]
fn every_level_gives_the_scalar_bits() {
    let mut rng = Rng(0x57_1c7);
    let n = 37;
    let lines: Vec<Line<(), f64>> = (0..n)
        .map(|_| Line::from_coeffs(core::array::from_fn(|_| rng.next_f64())))
        .collect();
    let s32: Vec<Line<(), Strict<f32>>> = lines
        .iter()
        .map(|b| Strict::wrap(Line::<(), f32>::from_coeffs(b.c.map(|x| x as f32))))
        .collect();
    let s64: Vec<Line<(), Strict<f64>>> = lines.iter().map(|&b| Strict::wrap(b)).collect();
    let want32: Vec<_> = s32.iter().map(|&b| Pipeline.call(b)).collect();
    let want64: Vec<_> = s64.iter().map(|&b| Pipeline.call(b)).collect();
    for level in batch::levels() {
        batch::with_level(level, || {
            let name = batch::level_name(level);
            let mut out = vec![Line::zero(); n];
            batch::map(&Pipeline, &s32, &mut out);
            for (g, w) in out.iter().zip(&want32) {
                assert_eq!(bits32(g), bits32(w), "{name}: Strict<f32>");
            }
            let mut out = vec![Line::zero(); n];
            batch::map(&Pipeline, &s64, &mut out);
            for (g, w) in out.iter().zip(&want64) {
                assert_eq!(bits64(g), bits64(w), "{name}: Strict<f64>");
            }
        });
    }
}

/// `Strict` computes what plain `f32` does, to rounding: the same algorithms, without fused
/// multiply-adds and with portable elementary functions.
#[test]
fn close_to_plain_floats() {
    let mut rng = Rng(3);
    for _ in 0..100 {
        let b = Line::<(), f32>::from_coeffs(core::array::from_fn(|_| rng.next_f64() as f32));
        let plain = Pipeline.call(b);
        let strict: Line<(), f32> = Strict::unwrap(Pipeline.call(Strict::wrap(b)));
        for (x, y) in plain.c.iter().zip(strict.c) {
            assert!((x - y).abs() <= 1e-5 * (1.0 + x.abs()), "{x} vs {y}");
        }
    }
}
