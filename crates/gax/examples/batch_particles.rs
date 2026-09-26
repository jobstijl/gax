//! A particle swarm moved by one motor per frame, three ways: a scalar loop, the batch
//! kernel on an array of structs, and the batch kernel on struct-of-arrays storage.
//!
//! The batch kernels run on the best SIMD level the CPU has, detected at run time. Run with
//! `cargo run --release --example batch_particles --features batch`.

use gax::batch::{self, BatchTransform, Soa};
use gax::pga3d::{Motor, Point};
use std::time::Instant;

fn main() {
    let n = 100_000;
    let frames = 100;
    let points: Vec<Point> = (0..n)
        .map(|i| {
            let a = i as f32 * 0.001;
            Point::xyz(a.sin() * 10.0, a.cos() * 10.0, (i % 100) as f32 * 0.1)
        })
        .collect();
    // A little screw motion per frame.
    let step = Motor::translation(0.0, 0.0, 0.01) * Motor::rotation_about(0.0, 0.0, 1.0, 0.02);
    println!(
        "{n} points, {frames} frames, SIMD level: {}",
        batch::level_name(batch::level())
    );

    let mut scalar = points.clone();
    let t = Instant::now();
    for _ in 0..frames {
        for p in &mut scalar {
            *p = step >> *p;
        }
    }
    let t_scalar = t.elapsed();

    let (mut aos, mut next) = (points.clone(), points.clone());
    let t = Instant::now();
    for _ in 0..frames {
        step.transform_slice(&aos, &mut next);
        std::mem::swap(&mut aos, &mut next);
    }
    let t_aos = t.elapsed();

    let mut soa: Soa<Point> = points.iter().copied().collect();
    let mut soa_next = Soa::new();
    let t = Instant::now();
    for _ in 0..frames {
        step.transform_soa(&soa, &mut soa_next);
        std::mem::swap(&mut soa, &mut soa_next);
    }
    let t_soa = t.elapsed();

    let per = |d: std::time::Duration| d.as_secs_f64() * 1e9 / (n * frames) as f64;
    println!("scalar loop:           {:6.2} ns/point", per(t_scalar));
    println!(
        "batch, array of structs: {:4.2} ns/point ({:.1}x)",
        per(t_aos),
        t_scalar.as_secs_f64() / t_aos.as_secs_f64()
    );
    println!(
        "batch, struct of arrays: {:4.2} ns/point ({:.1}x)",
        per(t_soa),
        t_scalar.as_secs_f64() / t_soa.as_secs_f64()
    );

    // The three agree (up to rounding: the batch kernels use a prepared map and FMA).
    let worst = (0..n)
        .map(|i| {
            let (a, b, c) = (
                scalar[i].to_euclidean(),
                aos[i].to_euclidean(),
                soa.get(i).to_euclidean(),
            );
            (0..3)
                .map(|k| (a[k] - b[k]).abs().max((a[k] - c[k]).abs()))
                .fold(0.0f32, f32::max)
        })
        .fold(0.0f32, f32::max);
    println!("largest difference after {frames} frames: {worst:.2e}");
    assert!(worst < 1e-3);
}
