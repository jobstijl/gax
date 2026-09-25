//! Rigid transforms of points: gax against glam.

use criterion::{Criterion, criterion_group, criterion_main};
use gax::pga3d::Point;
use gax_bench::*;
use std::hint::black_box;

const N: usize = 1024;

fn points() -> Vec<[f32; 3]> {
    (0..N)
        .map(|i| {
            let x = i as f32;
            [x.sin(), (x * 0.7).cos(), x * 0.001]
        })
        .collect()
}

fn single(c: &mut Criterion) {
    let m = motor(0.7, [1.0, 2.0, 3.0], [0.5, -1.0, 2.0]);
    let p = Point::new(1.0, 2.0, 3.0, 1.0);
    let q = glam::Quat::from_axis_angle(glam::Vec3::new(1.0, 2.0, 3.0).normalize(), 0.7);
    let a = glam::Affine3A::from_rotation_translation(q, glam::Vec3::new(0.5, -1.0, 2.0));
    let v = glam::Vec3A::new(1.0, 2.0, 3.0);
    let mat = m >> Point::slot();
    let mut g = c.benchmark_group("single");
    g.bench_function("gax motor >> point", |b| {
        b.iter(|| black_box(m) >> black_box(p))
    });
    g.bench_function("gax matrix.of(point)", |b| {
        b.iter(|| black_box(mat).of(black_box(p)))
    });
    g.bench_function("glam affine.transform_point3a", |b| {
        b.iter(|| black_box(a).transform_point3a(black_box(v)))
    });
    g.bench_function("glam quat * vec3a", |b| {
        b.iter(|| black_box(q) * black_box(v))
    });
    g.finish();
}

fn batch(c: &mut Criterion) {
    let m = motor(0.7, [1.0, 2.0, 3.0], [0.5, -1.0, 2.0]);
    let pts: Vec<Point> = points()
        .iter()
        .map(|p| Point::new(p[0], p[1], p[2], 1.0))
        .collect();
    let q = glam::Quat::from_axis_angle(glam::Vec3::new(1.0, 2.0, 3.0).normalize(), 0.7);
    let a = glam::Affine3A::from_rotation_translation(q, glam::Vec3::new(0.5, -1.0, 2.0));
    let vs: Vec<glam::Vec3A> = points()
        .iter()
        .map(|p| glam::Vec3A::from_array(*p))
        .collect();
    let mut out = vec![Point::zero(); N];
    let mut vout = vec![glam::Vec3A::ZERO; N];
    let mut g = c.benchmark_group("batch1024");
    g.bench_function("gax motor >> point", |b| {
        b.iter(|| {
            let m = black_box(m);
            for (o, p) in out.iter_mut().zip(&pts) {
                *o = m >> *p;
            }
            black_box(&out);
        })
    });
    g.bench_function("gax matrix (built per batch)", |b| {
        b.iter(|| {
            let t = black_box(m) >> Point::slot();
            for (o, p) in out.iter_mut().zip(&pts) {
                *o = t.of(*p);
            }
            black_box(&out);
        })
    });
    // Struct-of-arrays: 128 lanes of 8 points each.
    use gax::simd::wide::f32x8;
    let soa: Vec<gax::pga3d::Point<(), f32x8>> = pts
        .chunks(8)
        .map(|c| {
            gax::pga3d::Point::from_coeffs(core::array::from_fn(|k| {
                f32x8::new(core::array::from_fn(|l| c[l].c[k]))
            }))
        })
        .collect();
    let mut soa_out = soa.clone();
    let m8 = gax::Unit::new_unchecked(gax::pga3d::Motor::<(), f32x8>::from_coeffs(
        m.into_inner().c.map(f32x8::splat),
    ));
    g.bench_function("gax SoA f32x8 motor >> point", |b| {
        b.iter(|| {
            let m8 = black_box(m8);
            for (o, p) in soa_out.iter_mut().zip(&soa) {
                *o = m8 >> *p;
            }
            black_box(&soa_out);
        })
    });
    g.bench_function("gax SoA f32x8 matrix", |b| {
        b.iter(|| {
            let t = black_box(m8) >> gax::pga3d::Point::slot();
            for (o, p) in soa_out.iter_mut().zip(&soa) {
                *o = t.of(*p);
            }
            black_box(&soa_out);
        })
    });
    g.bench_function("glam affine", |b| {
        b.iter(|| {
            let a = black_box(a);
            for (o, v) in vout.iter_mut().zip(&vs) {
                *o = a.transform_point3a(*v);
            }
            black_box(&vout);
        })
    });
    g.finish();
}

criterion_group!(benches, single, batch);
criterion_main!(benches);
