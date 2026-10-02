//! Conversions to the plain arrays other math libraries take: quaternions `[x, y, z, w]`,
//! translations `[x, y, z]` and column-major homogeneous matrices (glam's
//! `from_cols_array_2d`, nalgebra's `from_column_slice`, WGSL's and GLSL's `mat4x4`). No
//! dependency on those libraries: the arrays are the boundary (docs/guide.md, "Other math
//! libraries").

#[cfg(feature = "pga3d")]
mod pga3d_interop {
    use crate::pga3d::{Motor, Point, Rotor};
    use crate::{Real, Unit};

    impl<T: Real> Rotor<(), T> {
        /// The rotor of the unit quaternion `[x, y, z, w]` (glam's `Quat::to_array`, nalgebra's
        /// `coords`): the same rotation, right-handed, `w + x i + y j + z k` as
        /// `w − x e23 − y e31 − z e12`. Normalized, so a quaternion that drifted from unit
        /// length is fine.
        ///
        /// ```
        /// use gax::pga3d::{Motor, Point, Rotor};
        /// let h = std::f64::consts::FRAC_PI_4; // half of a quarter turn about z
        /// let r = Rotor::from_quaternion([0.0, 0.0, h.sin(), h.cos()]);
        /// let m = Motor::from_rotor(r);
        /// let [x, y, _] = (m >> Point::xyz(1.0, 0.0, 0.0)).to_euclidean();
        /// assert!(x.abs() < 1e-12 && (y - 1.0).abs() < 1e-12);
        /// ```
        #[inline]
        pub fn from_quaternion(q: [T; 4]) -> Unit<Self> {
            let [x, y, z, w] = q;
            Rotor::new(w, -x, -y, -z).normalized()
        }

        /// The unit quaternion `[x, y, z, w]` of this rotor's rotation (normalized). `r` and
        /// `-r` are the same rotation, as are `q` and `-q`.
        ///
        /// ```
        /// use gax::pga3d::Rotor;
        /// let q = [0.1, -0.2, 0.3, 0.927_361_849_549_570_3];
        /// let back = Rotor::<(), f64>::from_quaternion(q).to_quaternion();
        /// assert!(back.iter().zip(q).all(|(a, b)| (a - b).abs() < 1e-12));
        /// ```
        #[inline]
        pub fn to_quaternion(self) -> [T; 4] {
            let [w, a, b, c] = self.c;
            let s = (w * w + a * a + b * b + c * c).sqrt().recip();
            [-a * s, -b * s, -c * s, w * s]
        }
    }

    impl<T: Real> Motor<(), T> {
        /// A rotor as a motor (a rotation about a line through the origin).
        #[inline]
        pub fn from_rotor(r: Unit<Rotor<(), T>>) -> Unit<Self> {
            Unit::new_unchecked(r.into_inner().cast::<Motor>())
        }

        /// The rigid motion that rotates by the unit quaternion `q = [x, y, z, w]` and then
        /// translates by `t`: `x ↦ q x q⁻¹ + t`, as glam's `Affine3A::from_rotation_translation`
        /// and nalgebra's `Isometry3::from_parts` (the quaternion is normalized).
        ///
        /// ```
        /// use gax::pga3d::{Motor, Point};
        /// let h = std::f64::consts::FRAC_PI_4;
        /// let m = Motor::from_rotation_translation([0.0, 0.0, h.sin(), h.cos()], [1.0, 2.0, 3.0]);
        /// let [x, y, z] = (m >> Point::xyz(1.0, 0.0, 0.0)).to_euclidean();
        /// assert!((x - 1.0).abs() < 1e-12 && (y - 3.0).abs() < 1e-12 && (z - 3.0).abs() < 1e-12);
        /// ```
        #[inline]
        pub fn from_rotation_translation(q: [T; 4], t: [T; 3]) -> Unit<Self> {
            Self::translation(t[0], t[1], t[2]) * Self::from_rotor(Rotor::from_quaternion(q))
        }

        /// The unit quaternion `[x, y, z, w]` and the translation `[x, y, z]` of this motion
        /// (rotation first, then translation; see [`Motor::from_rotation_translation`]). Any
        /// motor of non-zero norm: the scale is divided out.
        ///
        /// ```
        /// use gax::pga3d::Motor;
        /// let m = Motor::<(), f64>::from_rotation_translation([0.6, 0.0, 0.0, 0.8], [1.0, -2.0, 0.5]);
        /// let (q, t) = m.to_rotation_translation();
        /// assert!(q.iter().zip([0.6, 0.0, 0.0, 0.8]).all(|(a, b)| (a - b).abs() < 1e-12));
        /// assert!(t.iter().zip([1.0, -2.0, 0.5]).all(|(a, b)| (a - b).abs() < 1e-12));
        /// ```
        #[inline]
        pub fn to_rotation_translation(self) -> ([T; 4], [T; 3]) {
            let [w, a, b, c, ..] = self.c;
            let q = Rotor::new(w, a, b, c).to_quaternion();
            let o = T::zero();
            (q, (self >> Point::xyz(o, o, o)).to_euclidean())
        }

        /// The column-major homogeneous 4×4 matrix of this motion on points: `cols[j]` is the
        /// image of the `j`-th basis vector `x`, `y`, `z`, `w` (the last column is the
        /// translation), as glam's `Mat4::from_cols_array_2d` and WGSL's `mat4x4` take it. Any
        /// motor of non-zero norm: the scale is divided out.
        ///
        /// ```
        /// use gax::pga3d::Motor;
        /// let cols = Motor::<(), f64>::translation(1.0, 2.0, 3.0).to_matrix();
        /// assert_eq!(cols[3], [1.0, 2.0, 3.0, 1.0]);
        /// assert_eq!(cols[0], [1.0, 0.0, 0.0, 0.0]);
        /// ```
        #[inline]
        pub fn to_matrix(self) -> [[T; 4]; 4] {
            // The map's rows are its outputs (x, y, z, w); its weight entry is the squared norm.
            let rows = (self >> Point::slot()).c;
            let s = rows[3][3].recip();
            core::array::from_fn(|j| core::array::from_fn(|i| rows[i][j] * s))
        }

        /// The motion of a rigid column-major homogeneous 4×4 matrix (a rotation and a
        /// translation, as [`Motor::to_matrix`] gives; scale and shear are not motions). The
        /// rotation comes from its quaternion by Shepperd's method, which is well conditioned
        /// for every rotation.
        ///
        /// ```
        /// use gax::pga3d::Motor;
        /// let m = Motor::<(), f64>::from_rotation_translation([0.0, 0.6, 0.0, 0.8], [1.0, 2.0, 3.0]);
        /// let back = Motor::from_matrix(m.to_matrix());
        /// let (a, b) = (m.to_matrix(), back.to_matrix());
        /// assert!(a.iter().flatten().zip(b.iter().flatten()).all(|(x, y)| (x - y).abs() < 1e-12));
        /// ```
        pub fn from_matrix(cols: [[T; 4]; 4]) -> Unit<Self> {
            let frame = core::array::from_fn(|k| [cols[k][0], cols[k][1], cols[k][2]]);
            let r = crate::moments::rotor_from_frame(frame);
            Self::translation(cols[3][0], cols[3][1], cols[3][2]) * r
        }
    }
}

#[cfg(feature = "pga2d")]
mod pga2d_interop {
    use crate::pga2d::{Motor, Point};
    use crate::{Real, Unit};

    impl<T: Real> Motor<(), T> {
        /// The column-major homogeneous 3×3 matrix of this motion on points: `cols[j]` is the
        /// image of the `j`-th basis vector `x`, `y`, `w` (the last column is the translation),
        /// as glam's `Mat3::from_cols_array_2d` takes it. Any motor of non-zero norm.
        ///
        /// ```
        /// use gax::pga2d::{Motor, Point};
        /// let m = Motor::<(), f64>::rotation(Point::xy(0.0, 0.0), std::f64::consts::FRAC_PI_2);
        /// let cols = m.to_matrix();
        /// assert!((cols[0][1] - 1.0).abs() < 1e-12 && cols[0][0].abs() < 1e-12);
        /// ```
        #[inline]
        pub fn to_matrix(self) -> [[T; 3]; 3] {
            let rows = (self >> Point::slot()).c;
            let s = rows[2][2].recip();
            core::array::from_fn(|j| core::array::from_fn(|i| rows[i][j] * s))
        }

        /// The motion of a rigid column-major homogeneous 3×3 matrix (a rotation and a
        /// translation, as [`Motor::to_matrix`] gives).
        ///
        /// ```
        /// use gax::pga2d::{Motor, Point};
        /// let m = Motor::<(), f64>::rotation(Point::xy(1.0, 2.0), 0.7);
        /// let back = Motor::from_matrix(m.to_matrix());
        /// let (a, b) = (m.to_matrix(), back.to_matrix());
        /// assert!(a.iter().flatten().zip(b.iter().flatten()).all(|(x, y)| (x - y).abs() < 1e-12));
        /// ```
        pub fn from_matrix(cols: [[T; 3]; 3]) -> Unit<Self> {
            let angle = cols[0][1].atan2(cols[0][0]);
            let o = T::zero();
            Self::translation(cols[2][0], cols[2][1]) * Self::rotation(Point::xy(o, o), angle)
        }
    }
}
