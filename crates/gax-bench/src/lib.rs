//! Kernels under test, as non-inlined functions so their assembly can be inspected with
//! `cargo rustc -p gax-bench --release -- --emit asm` and compared across libraries.

use gax::Unit;

pub mod kernels;

/// Fused kernels traced from [`kernels`] at build time.
pub mod fused {
    #![allow(missing_docs)]
    include!(concat!(env!("OUT_DIR"), "/fused.rs"));
}
use gax::pga3d::{Motor, Point, Rotor};

/// gax: unit motor applied to a point (direct fused sandwich).
#[inline(never)]
pub fn gax_motor_point(m: Unit<Motor>, p: Point) -> Point {
    m >> p
}

/// gax: unit rotor applied to a point.
#[inline(never)]
pub fn gax_rotor_point(r: Unit<Rotor>, p: Point) -> Point {
    r >> p
}

/// gax: build the point map of a unit motor.
#[inline(never)]
pub fn gax_motor_matrix(m: Unit<Motor>) -> Point<(Point,)> {
    m >> Point::slot()
}

/// gax: apply a point map to a point.
#[inline(never)]
pub fn gax_apply_matrix(t: Point<(Point,)>, p: Point) -> Point {
    t.of(p)
}

/// gax: apply a prepared (sparse) motor action to a point.
#[inline(never)]
pub fn gax_apply_prepared(t: <Unit<Motor> as gax::Prepare<Point>>::Output, p: Point) -> Point {
    t >> p
}

/// gax: motor composition (geometric product, tier 1).
#[inline(never)]
pub fn gax_motor_motor(a: Motor, b: Motor) -> Motor {
    a * b
}

/// glam: quaternion rotation of a vector.
#[inline(never)]
pub fn glam_quat_vec(q: glam::Quat, v: glam::Vec3A) -> glam::Vec3A {
    q * v
}

/// glam: affine transform of a point.
#[inline(never)]
pub fn glam_affine_point(a: glam::Affine3A, v: glam::Vec3A) -> glam::Vec3A {
    a.transform_point3a(v)
}

/// glam: 4x4 matrix times a vector.
#[inline(never)]
pub fn glam_mat4_vec4(m: glam::Mat4, v: glam::Vec4) -> glam::Vec4 {
    m * v
}

/// glam: quaternion product.
#[inline(never)]
pub fn glam_quat_quat(a: glam::Quat, b: glam::Quat) -> glam::Quat {
    a * b
}

/// A motor for a rotation by `angle` about the axis `(ax, ay, az)` (through the origin),
/// followed by a translation `(tx, ty, tz)`.
pub fn motor(angle: f32, axis: [f32; 3], t: [f32; 3]) -> Unit<Motor> {
    Motor::translation(t[0], t[1], t[2]) * Motor::rotation_about(axis[0], axis[1], axis[2], angle)
}

/// Probe: the rigid-body step, generic code with constants inlined.
#[inline(never)]
pub fn probe_rigid_generic(
    m: Motor,
    b: gax::pga3d::Line,
    f: gax::pga3d::Line,
) -> (Motor, gax::pga3d::Line) {
    kernels::rigid_step_fixed(m, b, f)
}

/// Probe: the same, fused at build time.
#[inline(never)]
pub fn probe_rigid_fused(
    m: Motor,
    b: gax::pga3d::Line,
    f: gax::pga3d::Line,
) -> (Motor, gax::pga3d::Line) {
    fused::rigid_step_fixed_fused(m, b, f)
}

/// Batch transform of struct-of-arrays data (for inspecting the dispatched kernels).
#[inline(never)]
pub fn probe_batch_soa(
    m: Unit<Motor>,
    xs: &gax::batch::Soa<Point>,
    out: &mut gax::batch::Soa<Point>,
) {
    use gax::batch::BatchTransform;
    m.transform_soa(xs, out);
}

/// One lane multiply-add at the AVX2 level, portable level.
#[inline(never)]
pub fn probe_lanes_fma(
    a: gax::batch::Lanes<f32, 8>,
    b: gax::batch::Lanes<f32, 8>,
    c: gax::batch::Lanes<f32, 8>,
) -> gax::batch::Lanes<f32, 8> {
    a * b + c
}

/// Batch transform of array-of-structs data (for inspecting the transposes).
#[inline(never)]
pub fn probe_batch_aos(m: Unit<Motor>, xs: &[Point], out: &mut [Point]) {
    use gax::batch::BatchTransform;
    m.transform_slice(xs, out);
}

/// Batch pairwise sandwiches (for inspecting the dispatched kernel).
#[inline(never)]
pub fn probe_batch_each(ms: &[Unit<Motor>], xs: &[Point], out: &mut [Point]) {
    use gax::batch::BatchTransform;
    Unit::transform_each(ms, xs, out);
}

/// Batch exp of twists (for inspecting the dispatched kernel).
#[inline(never)]
pub fn probe_batch_exp(bs: &[gax::pga3d::Line], out: &mut [Motor]) {
    struct Exp;
    impl gax::batch::Map for Exp {
        type X = gax::pga3d::Line;
        type Y = Motor;
        #[inline(always)]
        fn call<T: gax::Real>(&self, b: gax::pga3d::Line<(), T>) -> Motor<(), T> {
            b.exp().into_inner()
        }
    }
    gax::batch::map(&Exp, bs, out);
}
