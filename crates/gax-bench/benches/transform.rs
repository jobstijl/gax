//! Common PGA3D operations: gax against glam.
//!
//! Run with `RUSTFLAGS="-C target-cpu=native" cargo bench -p gax-bench --bench transform`.
//! The methodology follows mathbench-rs: inputs are `black_box`ed and single operations are
//! measured one at a time, while batches run over 1024 points.
#![allow(missing_docs)] // `criterion_group!` generates an undocumented function

use criterion::{Criterion, criterion_group, criterion_main};
use gax::Unit;
use gax::pga3d::{Line, Motor, Point};
use gax::simd::wide::f32x8;
use gax_bench::motor;
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

fn glam_pose() -> (glam::Quat, glam::Affine3A) {
    let q = glam::Quat::from_axis_angle(glam::Vec3::new(1.0, 2.0, 3.0).normalize(), 0.7);
    (
        q,
        glam::Affine3A::from_rotation_translation(q, glam::Vec3::new(0.5, -1.0, 2.0)),
    )
}

fn single(c: &mut Criterion) {
    let m = motor(0.7, [1.0, 2.0, 3.0], [0.5, -1.0, 2.0]);
    let p = Point::new(1.0, 2.0, 3.0, 1.0);
    let (q, a) = glam_pose();
    let v = glam::Vec3A::new(1.0, 2.0, 3.0);
    let mat = m >> Point::slot();
    let prep = m.prepare::<Point>();
    let mut g = c.benchmark_group("transform one point");
    g.bench_function("gax Unit<Motor> >> Point", |b| {
        b.iter(|| black_box(m) >> black_box(p));
    });
    g.bench_function("gax prepared >> Point", |b| {
        b.iter(|| black_box(prep) >> black_box(p));
    });
    // The maps read from memory (as from an array, or a reference) rather than passed by
    // value: by value, the 13 entries of the prepared map are stored right before the kernel
    // reads them in column groups that straddle those stores, which stalls store forwarding.
    g.bench_function("gax prepared >> Point, map from memory", |b| {
        let r = &prep;
        b.iter(|| *black_box(r) >> black_box(p));
    });
    g.bench_function("gax Point<(Point,)>::of, map from memory", |b| {
        let r = &mat;
        b.iter(|| black_box(r).of(black_box(p)));
    });
    g.bench_function("gax Point<(Point,)>::of", |b| {
        b.iter(|| black_box(mat).of(black_box(p)));
    });
    g.bench_function("glam Affine3A::transform_point3a", |b| {
        b.iter(|| black_box(a).transform_point3a(black_box(v)));
    });
    g.bench_function("glam Quat * Vec3A (rotation only)", |b| {
        b.iter(|| black_box(q) * black_box(v));
    });
    g.finish();
}

fn batch(c: &mut Criterion) {
    let m = motor(0.7, [1.0, 2.0, 3.0], [0.5, -1.0, 2.0]);
    let pts: Vec<Point> = points()
        .iter()
        .map(|p| Point::new(p[0], p[1], p[2], 1.0))
        .collect();
    let (_, a) = glam_pose();
    let vs: Vec<glam::Vec3A> = points()
        .iter()
        .map(|p| glam::Vec3A::from_array(*p))
        .collect();
    let mut out = vec![Point::zero(); N];
    let mut vout = vec![glam::Vec3A::ZERO; N];
    let soa: Vec<Point<(), f32x8>> = pts
        .chunks(8)
        .map(|c| {
            Point::from_coeffs(core::array::from_fn(|k| {
                f32x8::new(core::array::from_fn(|l| c[l].c[k]))
            }))
        })
        .collect();
    let mut soa_out = soa.clone();
    let m8 = Unit::new_unchecked(Motor::<(), f32x8>::from_coeffs(
        m.into_inner().c.map(f32x8::splat),
    ));

    let mut g = c.benchmark_group("transform 1024 points");
    g.bench_function("gax direct m >> p", |b| {
        b.iter(|| {
            let m = black_box(m);
            for (o, p) in out.iter_mut().zip(&pts) {
                *o = m >> *p;
            }
            black_box(&out);
        });
    });
    g.bench_function("gax prepared", |b| {
        b.iter(|| {
            let t = black_box(m).prepare::<Point>();
            for (o, p) in out.iter_mut().zip(&pts) {
                *o = t >> *p;
            }
            black_box(&out);
        });
    });
    g.bench_function("gax dense map", |b| {
        b.iter(|| {
            let t = black_box(m) >> Point::slot();
            for (o, p) in out.iter_mut().zip(&pts) {
                *o = t.of(*p);
            }
            black_box(&out);
        });
    });
    g.bench_function("gax SoA f32x8 direct", |b| {
        b.iter(|| {
            let m8 = black_box(m8);
            for (o, p) in soa_out.iter_mut().zip(&soa) {
                *o = m8 >> *p;
            }
            black_box(&soa_out);
        });
    });
    g.bench_function("gax SoA f32x8 prepared", |b| {
        b.iter(|| {
            let t = black_box(m8).prepare::<Point>();
            for (o, p) in soa_out.iter_mut().zip(&soa) {
                *o = t >> *p;
            }
            black_box(&soa_out);
        });
    });
    g.bench_function("glam Affine3A", |b| {
        b.iter(|| {
            let a = black_box(a);
            for (o, v) in vout.iter_mut().zip(&vs) {
                *o = a.transform_point3a(*v);
            }
            black_box(&vout);
        });
    });
    g.finish();
}

fn motors(c: &mut Criterion) {
    let m1 = motor(0.7, [1.0, 2.0, 3.0], [0.5, -1.0, 2.0]);
    let m2 = motor(-0.3, [0.0, 1.0, 1.0], [1.0, 0.0, 0.5]);
    let (q1, a1) = glam_pose();
    let q2 = glam::Quat::from_axis_angle(glam::Vec3::new(0.0, 1.0, 1.0).normalize(), -0.3);
    let a2 = glam::Affine3A::from_rotation_translation(q2, glam::Vec3::new(1.0, 0.0, 0.5));
    let raw = m1.into_inner().gp(1.3);
    let qraw = q1 * 1.3;
    let biv = Line::new(0.2, 0.4, 0.6, 0.1, -0.2, 0.3);
    let axis = glam::Vec3::new(1.0, 2.0, 3.0).normalize();

    let mut g = c.benchmark_group("motors");
    g.bench_function("gax compose Unit<Motor> * Unit<Motor>", |b| {
        b.iter(|| black_box(m1) * black_box(m2));
    });
    g.bench_function("glam Quat * Quat", |b| {
        b.iter(|| black_box(q1) * black_box(q2));
    });
    g.bench_function("glam Affine3A * Affine3A", |b| {
        b.iter(|| black_box(a1) * black_box(a2));
    });
    g.bench_function("gax Motor::normalized", |b| {
        b.iter(|| black_box(raw).normalized());
    });
    g.bench_function("glam Quat::normalize", |b| {
        b.iter(|| black_box(qraw).normalize());
    });
    g.bench_function("gax Unit<Motor>::inverse", |b| {
        b.iter(|| black_box(m1).inverse());
    });
    g.bench_function("glam Affine3A::inverse", |b| {
        b.iter(|| black_box(a1).inverse());
    });
    g.bench_function("gax Line::exp", |b| b.iter(|| black_box(biv).exp()));
    g.bench_function("glam Quat::from_axis_angle", |b| {
        b.iter(|| glam::Quat::from_axis_angle(black_box(axis), black_box(0.7)));
    });
    // The fair comparison for an exponential: the rotation vector's length must be computed.
    let scaled = glam::Vec3::new(0.2, 0.4, 0.6);
    g.bench_function("glam Quat::from_scaled_axis", |b| {
        b.iter(|| glam::Quat::from_scaled_axis(black_box(scaled)));
    });
    g.bench_function("gax Unit<Motor>::log", |b| {
        b.iter(|| -> Line { black_box(m1).log() });
    });
    // 256 different arguments per iteration (rotations of 0.05 to 2.6 rad with translations):
    // with one constant argument, the elementary functions' branches are always predicted.
    let bivs: Vec<Line> = (0..256)
        .map(|i| {
            let t = i as f32 / 256.0;
            Line::new(0.1 + t, 0.4 * t, 0.6 - t, 0.1, -0.2 + t, 0.3)
        })
        .collect();
    let axes: Vec<glam::Vec3> = bivs
        .iter()
        .map(|l| glam::Vec3::new(l.c[0], l.c[1], l.c[2]))
        .collect();
    let motors: Vec<Unit<Motor>> = bivs.iter().map(|l| l.exp()).collect();
    g.bench_function("gax Line::exp, 256 arguments", |b| {
        b.iter(|| {
            for l in black_box(&bivs) {
                black_box(l.exp());
            }
        });
    });
    g.bench_function("glam Quat::from_scaled_axis, 256 arguments", |b| {
        b.iter(|| {
            for v in black_box(&axes) {
                black_box(glam::Quat::from_scaled_axis(*v));
            }
        });
    });
    g.bench_function("gax Unit<Motor>::log, 256 arguments", |b| {
        b.iter(|| {
            for m in black_box(&motors) {
                let l: Line = m.log();
                black_box(l);
            }
        });
    });
    g.bench_function("gax build map m >> Point::slot()", |b| {
        b.iter(|| black_box(m1) >> Point::slot());
    });
    g.bench_function("gax build map m.prepare().to_map()", |b| {
        b.iter(|| -> Point<(Point,)> { black_box(m1).prepare::<Point>().to_map() });
    });
    g.bench_function("glam Affine3A::from_rotation_translation", |b| {
        b.iter(|| {
            glam::Affine3A::from_rotation_translation(
                black_box(q1),
                black_box(glam::Vec3::new(0.5, -1.0, 2.0)),
            )
        });
    });
    g.finish();
}

criterion_group!(benches, single, batch, motors);
criterion_main!(benches);
