//! With the `deterministic` feature, the same kernel gives the same bits on scalars and on every
//! SIMD level: no fused multiply-add anywhere, and the elementary functions from pure Rust
//! (docs/numerics.md, ADR-027).

#![cfg(all(feature = "deterministic", feature = "batch", feature = "pga3d"))]

use gax::batch::{self, BatchTransform, Map};
use gax::pga3d::{Line, Motor, Point};
use gax::{Extensor, Real, Unit};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

fn bits<M: Extensor<Slots = (), Coef = f32>>(m: &M) -> Vec<u32> {
    m.coeffs().as_ref().iter().map(|x| x.to_bits()).collect()
}

/// exp, then the motor's action on a point, then log: sin, cos, atan2, sqrt and many products;
/// and `exp` of a coefficient (the lanes once built it from `sinh` and `cosh`, the scalars not).
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

#[test]
fn every_level_gives_the_scalar_bits() {
    let mut rng = Rng(42);
    let n = 37;
    let lines: Vec<Line> = (0..n)
        .map(|_| {
            Line::new(
                rng.next(),
                rng.next(),
                rng.next(),
                rng.next(),
                rng.next(),
                rng.next(),
            )
        })
        .collect();
    let motors: Vec<Unit<Motor>> = lines.iter().map(|b| b.exp()).collect();
    let points: Vec<Point> = (0..n)
        .map(|_| Point::xyz(rng.next(), rng.next(), rng.next()))
        .collect();

    let want_map: Vec<Line> = lines.iter().map(|&b| Pipeline.call(b)).collect();
    let want_each: Vec<Point> = motors.iter().zip(&points).map(|(&m, &p)| m >> p).collect();
    let prepared = motors[0].prepare::<Point>();
    let want_slice: Vec<Point> = points.iter().map(|&p| prepared >> p).collect();

    for level in batch::levels() {
        batch::with_level(level, || {
            let name = batch::level_name(level);
            let mut out = vec![Line::zero(); n];
            batch::map(&Pipeline, &lines, &mut out);
            for (g, w) in out.iter().zip(&want_map) {
                assert_eq!(bits(g), bits(w), "{name}: exp, sandwich, log");
            }
            let mut out = vec![Point::zero(); n];
            Unit::transform_each(&motors, &points, &mut out);
            for (g, w) in out.iter().zip(&want_each) {
                assert_eq!(bits(g), bits(w), "{name}: transform_each");
            }
            motors[0].transform_slice(&points, &mut out);
            for (g, w) in out.iter().zip(&want_slice) {
                assert_eq!(bits(g), bits(w), "{name}: transform_slice");
            }
        });
    }
}

/// The feature and `Strict` define the same deterministic result: plain `f32` with the feature
/// gives the bits of `Strict<f32>`.
#[test]
fn the_feature_computes_as_strict() {
    use gax::strict::Strict;
    let mut rng = Rng(7);
    for _ in 0..50 {
        let b = Line::new(
            rng.next(),
            rng.next(),
            rng.next(),
            rng.next(),
            rng.next(),
            rng.next(),
        );
        let plain = Pipeline.call(b);
        let strict: Line<(), f32> = Strict::unwrap(Pipeline.call(Strict::wrap(b)));
        assert_eq!(bits(&plain), bits(&strict));
    }
}
