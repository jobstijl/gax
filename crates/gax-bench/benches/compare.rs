//! gax against nalgebra, ultraviolet and the `geometric_algebra` crate, beyond the glam suite
//! in `transform.rs`: transforms, composition chains, solvers, a rigid-body step (generic and
//! build-time fused), and CGA/CSTA products.
//!
//! Run with `RUSTFLAGS="-C target-cpu=native" cargo bench -p gax-bench --bench compare`.

use criterion::{Criterion, criterion_group, criterion_main};
use gax::Unit;
use gax::pga3d::{Line, Motor, Plane, Point, Scalar};
use gax::simd::wide::f32x8;
use gax_bench::fused::{rigid_step_fixed_fused, rigid_step_fixed_plain, rigid_step_fused};
use gax_bench::kernels::{rigid_step, rigid_step_fixed};
use gax_bench::motor;
use std::hint::black_box;

fn transforms(c: &mut Criterion) {
    let m = motor(0.7, [1.0, 2.0, 3.0], [0.5, -1.0, 2.0]);
    let p = Point::new(1.0, 2.0, 3.0, 1.0);
    let axis = nalgebra::Unit::new_normalize(nalgebra::Vector3::new(1.0f32, 2.0, 3.0));
    let iso = nalgebra::Isometry3::from_parts(
        nalgebra::Translation3::new(0.5, -1.0, 2.0),
        nalgebra::UnitQuaternion::from_axis_angle(&axis, 0.7),
    );
    let np = nalgebra::Point3::new(1.0f32, 2.0, 3.0);
    let ga_m = geometric_algebra::ppga3d::Motor::new(0.94, 0.09, 0.18, 0.27, 0.1, -0.2, 0.3, 0.01);
    let ga_p = geometric_algebra::ppga3d::Point::new(1.0, 2.0, 3.0, 1.0);
    // ultraviolet: eight rotations of eight vectors at once (rotation only).
    let uv_r = ultraviolet::Rotor3x8::from_angle_plane(
        ultraviolet::f32x8::splat(0.7),
        ultraviolet::Bivec3x8::new(
            ultraviolet::f32x8::splat(0.27),
            ultraviolet::f32x8::splat(-0.53),
            ultraviolet::f32x8::splat(0.8),
        ),
    );
    let uv_v = ultraviolet::Vec3x8::splat(ultraviolet::Vec3::new(1.0, 2.0, 3.0));
    let m8 = Unit::new_unchecked(Motor::<(), f32x8>::from_coeffs(
        m.into_inner().c.map(f32x8::splat),
    ));
    let p8 = Point::<(), f32x8>::from_coeffs(p.c.map(f32x8::splat));
    let prep8 = m8.prepare::<Point>();

    let mut g = c.benchmark_group("compare: transform");
    g.bench_function("gax Unit<Motor> >> Point", |b| {
        b.iter(|| black_box(m) >> black_box(p))
    });
    g.bench_function("geometric_algebra Motor::transformation(Point)", |b| {
        use geometric_algebra::Transformation;
        b.iter(|| black_box(ga_m).transformation(black_box(ga_p)))
    });
    g.bench_function("nalgebra Isometry3 * Point3", |b| {
        b.iter(|| black_box(iso) * black_box(np))
    });
    g.bench_function("gax x8: prepared motor >> Point<f32x8>", |b| {
        b.iter(|| black_box(prep8) >> black_box(p8))
    });
    g.bench_function("gax x8: Unit<Motor<f32x8>> >> Point<f32x8>", |b| {
        b.iter(|| black_box(m8) >> black_box(p8))
    });
    g.bench_function("ultraviolet x8: Rotor3x8 * Vec3x8 (rotation only)", |b| {
        b.iter(|| black_box(uv_r) * black_box(uv_v))
    });
    g.finish();
}

fn chains(c: &mut Criterion) {
    let ms: Vec<Unit<Motor>> = (0..5)
        .map(|i| {
            motor(
                0.3 * i as f32 + 0.1,
                [1.0, i as f32, 2.0],
                [0.1 * i as f32, 1.0, -0.5],
            )
        })
        .collect();
    let affs: Vec<glam::Affine3A> = (0..5)
        .map(|i| {
            let q = glam::Quat::from_axis_angle(
                glam::Vec3::new(1.0, i as f32, 2.0).normalize(),
                0.3 * i as f32 + 0.1,
            );
            glam::Affine3A::from_rotation_translation(q, glam::Vec3::new(0.1 * i as f32, 1.0, -0.5))
        })
        .collect();
    let isos: Vec<nalgebra::Isometry3<f32>> = (0..5)
        .map(|i| {
            let axis = nalgebra::Unit::new_normalize(nalgebra::Vector3::new(1.0, i as f32, 2.0));
            nalgebra::Isometry3::from_parts(
                nalgebra::Translation3::new(0.1 * i as f32, 1.0, -0.5),
                nalgebra::UnitQuaternion::from_axis_angle(&axis, 0.3 * i as f32 + 0.1),
            )
        })
        .collect();
    let mut g = c.benchmark_group("compare: compose a chain of 5 rigid motions");
    g.bench_function("gax Unit<Motor> product", |b| {
        b.iter(|| {
            let v = black_box(&ms);
            v[0] * v[1] * v[2] * v[3] * v[4]
        })
    });
    g.bench_function("glam Affine3A product", |b| {
        b.iter(|| {
            let v = black_box(&affs);
            v[0] * v[1] * v[2] * v[3] * v[4]
        })
    });
    g.bench_function("nalgebra Isometry3 product", |b| {
        b.iter(|| {
            let v = black_box(&isos);
            v[0] * v[1] * v[2] * v[3] * v[4]
        })
    });
    g.finish();
}

fn solvers(c: &mut Criterion) {
    // A 6x6 map on twists (inertia-like, well conditioned), and the same in nalgebra.
    let mut stiffness: Scalar<(Line, Line), f64> = Scalar::zero();
    let mut inertia: Scalar<(Line, Line), f64> = Scalar::zero();
    for k in 0..8 {
        let f = k as f64;
        let a = Line::new(f.sin(), f.cos(), 0.3 * f, 1.0, -0.5 * f, 0.2);
        let bb = Line::new(
            1.0 + f,
            0.1 * f * f,
            f.cos(),
            (2.0 * f).sin(),
            0.7 - f,
            -0.2 * f,
        );
        let (pa, pb): (Scalar<(Line,), f64>, Scalar<(Line,), f64>) =
            (a & Line::slot(), bb & Line::slot());
        stiffness += pa * pa;
        inertia += pb * pb;
    }
    let map: Line<(Line,), f64> = Line::from_coeffs(core::array::from_fn(|i| {
        core::array::from_fn(|j| inertia.c[0][i][j] + if i == j { 1.0 } else { 0.0 })
    }));
    let nm = nalgebra::Matrix6::<f64>::from_fn(|i, j| map.c[i][j]);
    let rhs = Line::<(), f64>::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
    let nrhs = nalgebra::Vector6::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
    let ka = nalgebra::Matrix6::<f64>::from_fn(|i, j| stiffness.c[0][i][j]);
    let ma = nalgebra::Matrix6::<f64>::from_fn(|i, j| inertia.c[0][i][j]);
    let proj: Plane<(Point,), f64> = Plane::from_coeffs([
        [1.0, 0.2, 0.0, 0.1],
        [0.3, 1.0, 0.5, 0.0],
        [0.0, 0.1, 2.0, 0.4],
        [0.2, 0.0, 0.3, 1.0],
    ]);
    let nproj = nalgebra::Matrix4::<f64>::from_fn(|i, j| proj.c[i][j]);

    // The metric form must be positive definite (the modes are finite).
    let (vals, _) = stiffness.eigh_with(inertia);
    assert!(
        vals.iter().all(|v| v.is_finite()),
        "inertia form is not positive definite"
    );
    let mut g = c.benchmark_group("compare: solvers (f64)");
    g.bench_function("gax 6x6 map inverse", |b| {
        b.iter(|| black_box(map).inverse())
    });
    g.bench_function("nalgebra Matrix6::try_inverse", |b| {
        b.iter(|| black_box(nm).try_inverse())
    });
    g.bench_function("gax 6x6 map solve", |b| {
        b.iter(|| black_box(map).solve(black_box(rhs)))
    });
    g.bench_function("nalgebra Matrix6 LU solve", |b| {
        b.iter(|| black_box(nm).lu().solve(&black_box(nrhs)))
    });
    g.bench_function("gax 6x6 generalized eigh (modes)", |b| {
        b.iter(|| black_box(stiffness).eigh_with(black_box(inertia)))
    });
    g.bench_function("nalgebra 6x6 Cholesky + SymmetricEigen", |b| {
        b.iter(|| {
            let l = black_box(ma).cholesky().unwrap();
            let li = l.l().try_inverse().unwrap();
            let c = li * black_box(ka) * li.transpose();
            let e = c.symmetric_eigen();
            (e.eigenvalues, li.transpose() * e.eigenvectors)
        })
    });
    g.bench_function("gax 4x4 map SVD", |b| b.iter(|| black_box(proj).svd()));
    g.bench_function("nalgebra Matrix4 SVD", |b| {
        b.iter(|| black_box(nproj).svd(true, true))
    });
    g.finish();
}

fn rigid_body(c: &mut Criterion) {
    let m = motor(0.7, [1.0, 2.0, 3.0], [0.5, -1.0, 2.0]).into_inner();
    let b = Line::new(0.1, 0.2, 0.3, 0.05, -0.1, 0.2);
    let f = Line::new(0.0, 0.0, 0.0, 0.0, 0.0, -9.81);
    let (dt, mass, moments) = (1.0f32 / 60.0, 2.0f32, [0.5f32, 0.75, 1.0]);
    // The fused kernels compute the same step.
    let (m1, b1) = rigid_step(m, b, f, dt, mass, moments);
    let (m2, b2) = rigid_step_fused(m, b, f, dt, mass, moments);
    assert!(m1.c.iter().zip(m2.c).all(|(x, y)| (x - y).abs() < 1e-5));
    assert!(b1.c.iter().zip(b2.c).all(|(x, y)| (x - y).abs() < 1e-5));

    let mut g = c.benchmark_group("compare: rigid body Euler step (PGA3D)");
    g.bench_function("gax generic", |bch| {
        bch.iter(|| {
            rigid_step(
                black_box(m),
                black_box(b),
                black_box(f),
                black_box(dt),
                black_box(mass),
                black_box(moments),
            )
        })
    });
    g.bench_function("gax fused at build time", |bch| {
        bch.iter(|| {
            rigid_step_fused(
                black_box(m),
                black_box(b),
                black_box(f),
                black_box(dt),
                black_box(mass),
                black_box(moments),
            )
        })
    });
    g.bench_function("gax generic, constants inlined", |bch| {
        bch.iter(|| rigid_step_fixed(black_box(m), black_box(b), black_box(f)))
    });
    g.bench_function("gax fused, constants traced", |bch| {
        bch.iter(|| rigid_step_fixed_fused(black_box(m), black_box(b), black_box(f)))
    });
    g.bench_function("gax fused without mul_add, constants traced", |bch| {
        bch.iter(|| rigid_step_fixed_plain(black_box(m), black_box(b), black_box(f)))
    });
    // Eight bodies at once (SoA lanes): here the arithmetic count is what runs.
    let m8 = Motor::<(), f32x8>::from_coeffs(m.c.map(f32x8::splat));
    let b8 = Line::<(), f32x8>::from_coeffs(b.c.map(f32x8::splat));
    let f8 = Line::<(), f32x8>::from_coeffs(f.c.map(f32x8::splat));
    g.bench_function("x8 lanes: gax generic, constants inlined", |bch| {
        bch.iter(|| rigid_step_fixed(black_box(m8), black_box(b8), black_box(f8)))
    });
    g.bench_function("x8 lanes: gax fused, constants traced", |bch| {
        bch.iter(|| rigid_step_fixed_fused(black_box(m8), black_box(b8), black_box(f8)))
    });
    g.bench_function(
        "x8 lanes: gax fused without mul_add, constants traced",
        |bch| bch.iter(|| rigid_step_fixed_plain(black_box(m8), black_box(b8), black_box(f8))),
    );
    g.finish();
}

gax::algebra! {
    algebra csta "Conformal spacetime algebra R(4,2).";
    basis e0 = 1, e1 = -1, e2 = -1, e3 = -1, ep = 1, em = -1;
    kind Scalar = [1];
    versor Vector = [e0, e1, e2, e3, ep, em];
    kind Bivector = [e01, e02, e03, e12, e31, e23, e0p, e1p, e2p, e3p, e0m, e1m, e2m, e3m, epm];
}

fn conformal(c: &mut Criterion) {
    use gax::cga3d::{Even, Vector};
    let up = |x: f32, y: f32, z: f32| {
        Vector::<(), f32>::new(x, y, z, 1.0, 0.5 * (x * x + y * y + z * z))
    };
    let v = up(1.0, 2.0, 3.0);
    let versor: Even<(), f32> = up(0.5, 0.1, 0.2) * up(-0.3, 0.4, 1.0);
    let t = gax::cga3d::Twist::<(), f32>::new(0.1, 0.2, 0.3, 0.5, -0.2, 0.1);
    let cm = t.exp();
    let a = csta::Vector::<(), f32>::new(1.0, 0.2, 0.3, 0.4, 0.5, 0.6);
    let bb = csta::Vector::<(), f32>::new(0.3, -0.2, 0.1, 0.8, 0.2, 0.4);
    let mut g = c.benchmark_group("compare: conformal (f32)");
    g.bench_function("gax CGA3D Unit<Motor> >> point", |b| {
        b.iter(|| black_box(cm) >> black_box(v))
    });
    g.bench_function("gax CGA3D Even >> point (general versor)", |b| {
        b.iter(|| black_box(versor) >> black_box(v))
    });
    g.bench_function("gax CGA3D Twist::exp", |b| b.iter(|| black_box(t).exp()));
    g.bench_function("gax CSTA vector * vector", |b| {
        b.iter(|| black_box(a) * black_box(bb))
    });
    g.finish();

    // The 6D conformal group's logarithm (f64): the closed form through the invariant
    // decomposition against inverse scaling and squaring (docs/log6d.md).
    let mut g = c.benchmark_group("compare: CSTA log (f64)");
    let biv = gax::csta::Bivector::<(), f64>::from_coeffs(core::array::from_fn(|i| {
        0.3 * ((i as f64 * 0.7).sin())
    }));
    let r = biv.exp();
    g.bench_function("gax Unit<Even>::log (closed form)", |b| {
        b.iter(|| -> gax::csta::Bivector<(), f64> { black_box(r).log() })
    });
    g.bench_function(
        "gax Even::log_by_scaling (inverse scaling and squaring)",
        |b| b.iter(|| black_box(r).into_inner().log_by_scaling()),
    );
    g.finish();
}

criterion_group!(benches, transforms, chains, solvers, rigid_body, conformal);
criterion_main!(benches);
