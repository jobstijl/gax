//! Runtime-dispatched batch kernels (`gax::batch`, feature `batch`) against scalar code,
//! `wide::f32x8` lanes and glam.
//!
//! Run both ways: `cargo bench -p gax-bench --bench batch` (the default x86-64 target, where
//! only the dispatcher reaches AVX2) and with `RUSTFLAGS="-C target-cpu=native"`.

use criterion::{
    BenchmarkGroup, Criterion, criterion_group, criterion_main, measurement::WallTime,
};
use gax::Unit;
use gax::batch::{self, BatchTransform, Map, Soa};
use gax::pga3d::{Line, Motor, Point};
use gax::simd::wide::f32x8;
use gax::{Extensor, Real};
use gax_bench::motor;
use std::hint::black_box;

const N: usize = 1024;

fn points(n: usize) -> Vec<Point> {
    (0..n)
        .map(|i| {
            let x = i as f32;
            Point::xyz(x.sin(), (x * 0.7).cos(), x * 0.001)
        })
        .collect()
}

fn motors(n: usize) -> Vec<Unit<Motor>> {
    (0..n)
        .map(|i| {
            let x = i as f32 * 0.01;
            motor(0.3 + x, [1.0, x, 0.5], [x, -1.0, 0.2])
        })
        .collect()
}

fn twists(n: usize) -> Vec<Line> {
    (0..n)
        .map(|i| {
            let x = i as f32 * 0.003;
            Line::new(0.1 + x, -0.2, 0.3, 0.5 - x, 0.2, 0.1 + x)
        })
        .collect()
}

/// Pack an array of structs into `f32x8` struct-of-arrays chunks (zero padded).
fn pack<M: Extensor<Slots = (), Coef = f32>>(xs: &[M]) -> Vec<gax::Retype<M, (), f32x8>> {
    xs.chunks(8)
        .map(|c| {
            gax::Retype::<M, (), f32x8>::from_coeffs(<M::Kind as gax::Kind>::arr_from_fn(|k| {
                f32x8::new(core::array::from_fn(|l| {
                    c.get(l).map_or(0.0, |x| x.coeffs().as_ref()[k])
                }))
            }))
        })
        .collect()
}

/// Benchmark `f` once per dispatch level (and once with the automatic choice).
fn per_level(g: &mut BenchmarkGroup<'_, WallTime>, name: &str, mut f: impl FnMut()) {
    g.bench_function(format!("{name} (auto)"), |b| b.iter(&mut f));
    for level in batch::levels() {
        g.bench_function(format!("{name} ({})", batch::level_name(level)), |b| {
            batch::with_level(level, || b.iter(&mut f));
        });
    }
}

fn uniform(c: &mut Criterion) {
    let m = motor(0.7, [1.0, 2.0, 3.0], [0.5, -1.0, 2.0]);
    let pts = points(N);
    let mut out = vec![Point::zero(); N];
    let soa: Soa<Point> = pts.iter().copied().collect();
    let mut soa_out: Soa<Point> = Soa::new();
    let packed = pack(&pts);
    let mut packed_out = packed.clone();
    let m8 = Unit::new_unchecked(Motor::<(), f32x8>::from_coeffs(
        m.into_inner().c.map(f32x8::splat),
    ));
    let a = glam::Affine3A::from_rotation_translation(
        glam::Quat::from_axis_angle(glam::Vec3::new(1.0, 2.0, 3.0).normalize(), 0.7),
        glam::Vec3::new(0.5, -1.0, 2.0),
    );
    let vs: Vec<glam::Vec3A> = pts
        .iter()
        .map(|p| glam::Vec3A::from_array(p.to_euclidean()))
        .collect();
    let mut vout = vec![glam::Vec3A::ZERO; N];

    let mut g = c.benchmark_group("batch: one motor, 1024 points");
    g.bench_function("scalar prepared loop", |b| {
        b.iter(|| {
            let t = black_box(m).prepare::<Point>();
            for (o, p) in out.iter_mut().zip(&pts) {
                *o = t >> *p;
            }
            black_box(&out);
        })
    });
    g.bench_function("wide f32x8 SoA prepared", |b| {
        b.iter(|| {
            let t = black_box(m8).prepare::<Point>();
            for (o, p) in packed_out.iter_mut().zip(&packed) {
                *o = t >> *p;
            }
            black_box(&packed_out);
        })
    });
    g.bench_function("glam Affine3A loop", |b| {
        b.iter(|| {
            let a = black_box(a);
            for (o, v) in vout.iter_mut().zip(&vs) {
                *o = a.transform_point3a(*v);
            }
            black_box(&vout);
        })
    });
    per_level(&mut g, "batch transform_slice", || {
        black_box(m).transform_slice(&pts, &mut out);
        black_box(&out);
    });
    per_level(&mut g, "batch transform_soa", || {
        black_box(m).transform_soa(&soa, &mut soa_out);
        black_box(&soa_out);
    });
    g.finish();
}

fn pairwise(c: &mut Criterion) {
    let ms = motors(N);
    let pts = points(N);
    let mut out = vec![Point::zero(); N];
    let (pm, pp) = (
        pack(&ms.iter().map(|m| m.into_inner()).collect::<Vec<_>>()),
        pack(&pts),
    );
    let mut packed_out = pp.clone();
    let gs: Vec<(glam::Affine3A, glam::Vec3A)> = ms
        .iter()
        .zip(&pts)
        .map(|(m, p)| {
            let r = m.into_inner().c;
            let q = glam::Quat::from_xyzw(r[1], r[2], r[3], r[0]).normalize();
            (
                glam::Affine3A::from_rotation_translation(q, glam::Vec3::new(r[4], r[5], r[6])),
                glam::Vec3A::from_array(p.to_euclidean()),
            )
        })
        .collect();
    let mut vout = vec![glam::Vec3A::ZERO; N];

    let mut g = c.benchmark_group("batch: 1024 motor-point pairs");
    g.bench_function("scalar loop", |b| {
        b.iter(|| {
            for ((o, m), p) in out.iter_mut().zip(&ms).zip(&pts) {
                *o = *m >> *p;
            }
            black_box(&out);
        })
    });
    g.bench_function("wide f32x8 SoA", |b| {
        b.iter(|| {
            for ((o, m), p) in packed_out.iter_mut().zip(&pm).zip(&pp) {
                *o = Unit::new_unchecked(*m) >> *p;
            }
            black_box(&packed_out);
        })
    });
    g.bench_function("glam Affine3A loop", |b| {
        b.iter(|| {
            for (o, (a, v)) in vout.iter_mut().zip(&gs) {
                *o = a.transform_point3a(*v);
            }
            black_box(&vout);
        })
    });
    let (vsoa, xsoa): (Soa<Motor>, Soa<Point>) = (
        ms.iter().map(|m| m.into_inner()).collect(),
        pts.iter().copied().collect(),
    );
    let mut soa_out: Soa<Point> = Soa::new();
    per_level(&mut g, "batch transform_each_soa", || {
        Unit::<Motor>::transform_each_soa(black_box(&vsoa), &xsoa, &mut soa_out);
        black_box(&soa_out);
    });
    per_level(&mut g, "batch transform_each", || {
        Unit::transform_each(black_box(&ms), &pts, &mut out);
        black_box(&out);
    });
    g.finish();
}

struct Exp;
impl Map for Exp {
    type X = Line;
    type Y = Motor;
    #[inline(always)]
    fn call<T: Real>(&self, b: Line<(), T>) -> Motor<(), T> {
        b.exp().into_inner()
    }
}

fn exp(c: &mut Criterion) {
    let bs = twists(N);
    let mut out = vec![Motor::zero(); N];
    let packed = pack(&bs);
    let mut packed_out: Vec<Motor<(), f32x8>> = vec![Motor::zero(); packed.len()];
    let axes: Vec<glam::Vec3> = bs
        .iter()
        .map(|b| glam::Vec3::new(b.c[3], b.c[4], b.c[5]))
        .collect();
    let mut qout = vec![glam::Quat::IDENTITY; N];

    let mut g = c.benchmark_group("batch: exp of 1024 twists");
    g.bench_function("scalar loop", |b| {
        b.iter(|| {
            for (o, x) in out.iter_mut().zip(&bs) {
                *o = black_box(*x).exp().into_inner();
            }
            black_box(&out);
        })
    });
    g.bench_function("wide f32x8 SoA", |b| {
        b.iter(|| {
            for (o, x) in packed_out.iter_mut().zip(&packed) {
                *o = black_box(*x).exp().into_inner();
            }
            black_box(&packed_out);
        })
    });
    g.bench_function("glam Quat::from_scaled_axis loop (rotation only)", |b| {
        b.iter(|| {
            for (o, a) in qout.iter_mut().zip(&axes) {
                *o = glam::Quat::from_scaled_axis(black_box(*a));
            }
            black_box(&qout);
        })
    });
    per_level(&mut g, "batch map exp", || {
        batch::map(&Exp, black_box(&bs), &mut out);
        black_box(&out);
    });
    g.finish();
}

fn rigid_body(c: &mut Criterion) {
    use gax_bench::fused::{rigid_step_fused, rigid_step_fused_batch};
    use gax_bench::kernels::rigid_step;
    let n = N;
    let ms: Vec<Motor> = motors(n).iter().map(|m| m.into_inner()).collect();
    let bs = twists(n);
    let fs: Vec<Line> = twists(n).iter().map(|b| b.gp(0.1)).collect();
    let (dt, mass, moments) = (0.01f32, 2.0f32, [1.0f32, 2.0, 3.0]);
    let mut out = vec![(Motor::zero(), Line::zero()); n];
    let (pm, pb, pf) = (pack(&ms), pack(&bs), pack(&fs));
    let mut packed_out = vec![(Motor::<(), f32x8>::zero(), Line::<(), f32x8>::zero()); pm.len()];
    let (dt8, mass8, mom8) = (
        f32x8::splat(dt),
        f32x8::splat(mass),
        moments.map(f32x8::splat),
    );

    let mut g = c.benchmark_group("batch: rigid body step (traced), 1024 bodies");
    g.bench_function("scalar generic loop", |b| {
        b.iter(|| {
            for (o, ((m, bb), f)) in out.iter_mut().zip(ms.iter().zip(&bs).zip(&fs)) {
                *o = rigid_step(*m, *bb, *f, black_box(dt), mass, moments);
            }
            black_box(&out);
        })
    });
    g.bench_function("scalar fused loop", |b| {
        b.iter(|| {
            for (o, ((m, bb), f)) in out.iter_mut().zip(ms.iter().zip(&bs).zip(&fs)) {
                *o = rigid_step_fused(*m, *bb, *f, black_box(dt), mass, moments);
            }
            black_box(&out);
        })
    });
    g.bench_function("wide f32x8 SoA fused", |b| {
        b.iter(|| {
            for (o, ((m, bb), f)) in packed_out.iter_mut().zip(pm.iter().zip(&pb).zip(&pf)) {
                *o = rigid_step_fused(*m, *bb, *f, black_box(dt8), mass8, mom8);
            }
            black_box(&packed_out);
        })
    });
    per_level(&mut g, "batch rigid_step_fused_batch", || {
        rigid_step_fused_batch(
            &ms,
            &bs,
            &fs,
            &[black_box(dt)],
            &[mass],
            &[moments],
            &mut out,
        );
        black_box(&out);
    });
    g.finish();
}

criterion_group!(benches, uniform, pairwise, exp, rigid_body);
criterion_main!(benches);
