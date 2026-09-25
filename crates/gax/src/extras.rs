//! Hand-written conveniences for the standard algebras: constructors with geometric meaning.
//!
//! These are inherent functions on the generated types. For example, `Point::xyz(x, y, z)` and
//! `Motor::translation(dx, dy, dz)` are thin wrappers that fix the sign conventions in one place.

#[cfg(feature = "pga3d")]
mod pga3d_extras {
    use crate::pga3d::{Line, Motor, Plane, Point};
    use crate::{Real, Unit};

    impl<T: Real> Point<(), T> {
        /// The Euclidean point `(x, y, z)`, with weight 1: `x e032 + y e013 + z e021 + e123`.
        ///
        /// ```
        /// use gax::pga3d::Point;
        /// let p = Point::xyz(1.0, 2.0, 3.0);
        /// assert_eq!(p.e123(), 1.0);
        /// ```
        #[inline]
        pub fn xyz(x: T, y: T, z: T) -> Self {
            Point::new(x, y, z, T::one())
        }

        /// The direction `(x, y, z)`: a point at infinity, with weight 0.
        #[inline]
        pub fn direction(x: T, y: T, z: T) -> Self {
            Point::new(x, y, z, T::zero())
        }

        /// Euclidean coordinates `(x/w, y/w, z/w)`. Not finite for directions.
        #[inline]
        pub fn to_euclidean(self) -> [T; 3] {
            let r = self.e123().recip();
            [self.e032() * r, self.e013() * r, self.e021() * r]
        }
    }

    impl<T: Real> Plane<(), T> {
        /// The plane `n · x = d` with normal `n` (not necessarily unit length).
        #[inline]
        pub fn from_normal(n: [T; 3], d: T) -> Self {
            Plane::new(n[0], n[1], n[2], -d)
        }
    }

    impl<T: Real> Motor<(), T> {
        /// The translation by `(dx, dy, dz)`.
        ///
        /// ```
        /// use gax::pga3d::{Motor, Point};
        /// let t = Motor::translation(1.0, 2.0, 3.0);
        /// let p = t >> Point::xyz(0.0, 0.0, 0.0);
        /// assert_eq!(p.to_euclidean(), [1.0, 2.0, 3.0]);
        /// ```
        #[inline]
        pub fn translation(dx: T, dy: T, dz: T) -> Unit<Self> {
            let h = T::from_f64(-0.5);
            Unit::new_unchecked(Motor::new(
                T::one(),
                T::zero(),
                T::zero(),
                T::zero(),
                dx * h,
                dy * h,
                dz * h,
                T::zero(),
            ))
        }

        /// The rotation by `angle` (right-handed) about the `axis` line, which may be any line,
        /// not only one through the origin. For the axis through points `p` and `q`, pass
        /// `p & q`; the rotation is then counterclockwise when seen from `q` towards `p`.
        ///
        /// ```
        /// use gax::pga3d::{Motor, Point};
        /// let z_axis = Point::xyz(0.0, 0.0, 0.0) & Point::xyz(0.0, 0.0, 1.0);
        /// let r = Motor::rotation(z_axis, std::f64::consts::FRAC_PI_2);
        /// let p = (r >> Point::xyz(1.0, 0.0, 0.0)).to_euclidean();
        /// assert!((p[0] - 0.0).abs() < 1e-12 && (p[1] - 1.0).abs() < 1e-12);
        /// ```
        #[inline]
        pub fn rotation(axis: Line<(), T>, angle: T) -> Unit<Self> {
            let l = axis.normalized().into_inner();
            l.gp(angle * T::from_f64(-0.5)).exp()
        }

        /// The rotation by `angle` about the axis through the origin with direction `(x, y, z)`.
        #[inline]
        pub fn rotation_about(x: T, y: T, z: T, angle: T) -> Unit<Self> {
            let axis = Point::xyz(T::zero(), T::zero(), T::zero()) & Point::direction(x, y, z);
            Self::rotation(axis, angle)
        }
    }
}

#[cfg(feature = "pga2d")]
mod pga2d_extras {
    use crate::pga2d::{Motor, Point};
    use crate::{Real, Unit};

    impl<T: Real> Point<(), T> {
        /// The Euclidean point `(x, y)`, with weight 1: `x e20 + y e01 + e12`.
        #[inline]
        pub fn xy(x: T, y: T) -> Self {
            Point::new(x, y, T::one())
        }

        /// The direction `(x, y)`: a point at infinity.
        #[inline]
        pub fn direction(x: T, y: T) -> Self {
            Point::new(x, y, T::zero())
        }

        /// Euclidean coordinates `(x/w, y/w)`.
        #[inline]
        pub fn to_euclidean(self) -> [T; 2] {
            let r = self.e12().recip();
            [self.e20() * r, self.e01() * r]
        }
    }

    impl<T: Real> Motor<(), T> {
        /// The translation by `(dx, dy)`.
        #[inline]
        pub fn translation(dx: T, dy: T) -> Unit<Self> {
            let h = T::from_f64(0.5);
            Unit::new_unchecked(Motor::new(T::one(), T::zero(), dy * h, -dx * h))
        }

        /// The rotation by `angle` (counterclockwise) about the point `center`.
        #[inline]
        pub fn rotation(center: Point<(), T>, angle: T) -> Unit<Self> {
            let c = center.normalized().into_inner();
            c.gp(angle * T::from_f64(-0.5)).exp()
        }
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "pga3d")]
    #[test]
    fn pga3d_conventions() {
        use crate::pga3d::{Motor, Plane, Point};
        let close = |a: [f64; 3], b: [f64; 3]| a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-12);
        let t = Motor::translation(1.0, -2.0, 0.5);
        assert!(close(
            (t >> Point::xyz(1.0, 1.0, 1.0)).to_euclidean(),
            [2.0, -1.0, 1.5]
        ));
        let r = Motor::rotation_about(0.0, 0.0, 1.0, core::f64::consts::FRAC_PI_2);
        assert!(close(
            (r >> Point::xyz(1.0, 0.0, 0.0)).to_euclidean(),
            [0.0, 1.0, 0.0]
        ));
        // A rotation about an axis not through the origin: around the vertical line through
        // (1, 0, 0), the origin turns a quarter to (1, -1, 0).
        let axis = Point::xyz(1.0, 0.0, 0.0) & Point::xyz(1.0, 0.0, 1.0);
        let r2 = Motor::rotation(axis, core::f64::consts::FRAC_PI_2);
        assert!(close(
            (r2 >> Point::xyz(0.0, 0.0, 0.0)).to_euclidean(),
            [1.0, -1.0, 0.0]
        ));
        // Signed distance of a point above a plane.
        let floor = Plane::<(), f64>::from_normal([0.0, 0.0, 1.0], 1.0);
        assert!(((floor & Point::xyz(3.0, 4.0, 3.0)).s() - 2.0).abs() < 1e-12);
    }

    #[cfg(feature = "pga2d")]
    #[test]
    fn pga2d_conventions() {
        use crate::pga2d::{Motor, Point};
        let close = |a: [f64; 2], b: [f64; 2]| a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-12);
        let t = Motor::translation(1.0, -2.0);
        assert!(close(
            (t >> Point::xy(1.0, 1.0)).to_euclidean(),
            [2.0, -1.0]
        ));
        let r = Motor::rotation(Point::xy(0.0, 0.0), core::f64::consts::FRAC_PI_2);
        assert!(close((r >> Point::xy(1.0, 0.0)).to_euclidean(), [0.0, 1.0]));
        let r2 = Motor::rotation(Point::xy(1.0, 0.0), core::f64::consts::PI);
        assert!(close(
            (r2 >> Point::xy(0.0, 0.0)).to_euclidean(),
            [2.0, 0.0]
        ));
    }
}
