//! gax's array conversions (`from_quaternion`, `from_rotation_translation`, `to_matrix`,
//! `from_matrix`) against glam and nalgebra themselves: the same rotations, translations and
//! matrices, on random inputs.

use gax::pga2d;
use gax::pga3d::{Motor, Point, Rotor};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }
    /// A unit quaternion `[x, y, z, w]` and a translation.
    fn pose(&mut self) -> ([f64; 4], [f64; 3]) {
        let q: [f64; 4] = core::array::from_fn(|_| self.next());
        let n = q.iter().map(|x| x * x).sum::<f64>().sqrt();
        (
            q.map(|x| x / n),
            core::array::from_fn(|_| 3.0 * self.next()),
        )
    }
}

fn close(a: &[f64], b: &[f64], tol: f64) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < tol)
}

#[test]
fn rotations_and_translations_match_glam() {
    let mut rng = Rng(0x61a3);
    for _ in 0..200 {
        let (q, t) = rng.pose();
        let m = Motor::<(), f64>::from_rotation_translation(q, t);
        let gq = glam::DQuat::from_array(q);
        let affine = glam::DAffine3::from_rotation_translation(gq, glam::DVec3::from_array(t));
        let p: [f64; 3] = core::array::from_fn(|_| rng.next());
        let want = affine
            .transform_point3(glam::DVec3::from_array(p))
            .to_array();
        let got = (m >> Point::xyz(p[0], p[1], p[2])).to_euclidean();
        assert!(close(&got, &want, 1e-12), "{got:?} vs {want:?}");
        // The matrix, column by column, and back.
        let mat = glam::DMat4::from_rotation_translation(gq, glam::DVec3::from_array(t));
        let cols = m.to_matrix();
        assert!(close(cols.as_flattened(), &mat.to_cols_array(), 1e-12));
        let back = Motor::from_matrix(mat.to_cols_array_2d());
        assert!(close(
            back.to_matrix().as_flattened(),
            &mat.to_cols_array(),
            1e-12
        ));
        // The quaternion and the translation, up to the quaternion's sign.
        let (q2, t2) = m.to_rotation_translation();
        let sign = if q2[3] * q[3] + q2[0] * q[0] + q2[1] * q[1] + q2[2] * q[2] < 0.0 {
            -1.0
        } else {
            1.0
        };
        assert!(close(&q2.map(|x| x * sign), &q, 1e-12) && close(&t2, &t, 1e-12));
        let r = Rotor::<(), f64>::from_quaternion(q);
        let v = glam::DVec3::from_array(p);
        let want = (gq * v).to_array();
        let got = (Motor::from_rotor(r) >> Point::xyz(p[0], p[1], p[2])).to_euclidean();
        assert!(close(&got, &want, 1e-12));
    }
}

#[test]
fn isometries_match_nalgebra() {
    let mut rng = Rng(0x9a1b);
    for _ in 0..200 {
        let (q, t) = rng.pose();
        let m = Motor::<(), f64>::from_rotation_translation(q, t);
        let uq = nalgebra::UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
            q[3], q[0], q[1], q[2],
        ));
        let iso =
            nalgebra::Isometry3::from_parts(nalgebra::Translation3::new(t[0], t[1], t[2]), uq);
        let p: [f64; 3] = core::array::from_fn(|_| rng.next());
        let want = iso * nalgebra::Point3::new(p[0], p[1], p[2]);
        let got = (m >> Point::xyz(p[0], p[1], p[2])).to_euclidean();
        assert!(close(&got, want.coords.as_slice(), 1e-12));
        // nalgebra stores matrices column-major too.
        let h = iso.to_homogeneous();
        assert!(close(m.to_matrix().as_flattened(), h.as_slice(), 1e-12));
        let from = Motor::from_matrix(core::array::from_fn(|j| {
            core::array::from_fn(|i| h[(i, j)])
        }));
        assert!(close(from.to_matrix().as_flattened(), h.as_slice(), 1e-12));
    }
}

#[test]
fn planar_motions_match_glam() {
    let mut rng = Rng(0x2d2d);
    for _ in 0..200 {
        let angle = 3.0 * rng.next();
        let t = glam::DVec2::new(2.0 * rng.next(), 2.0 * rng.next());
        let mat = glam::DMat3::from_scale_angle_translation(glam::DVec2::ONE, angle, t);
        let m = pga2d::Motor::<(), f64>::from_matrix(mat.to_cols_array_2d());
        assert!(close(
            m.to_matrix().as_flattened(),
            &mat.to_cols_array(),
            1e-12
        ));
        let p = glam::DVec2::new(rng.next(), rng.next());
        let want = mat.transform_point2(p).to_array();
        let got = (m >> pga2d::Point::xy(p.x, p.y)).to_euclidean();
        assert!(close(&got, &want, 1e-12), "{got:?} vs {want:?}");
    }
}
