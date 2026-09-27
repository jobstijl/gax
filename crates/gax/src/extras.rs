//! Hand-written conveniences for the standard algebras: constructors with geometric meaning.
//!
//! These are inherent functions on the generated types. For example, `Point::xyz(x, y, z)` and
//! `Motor::translation(dx, dy, dz)` are thin wrappers that fix the sign conventions in one place.

#[cfg(feature = "pga3d")]
pub use pga3d_extras::PrincipalInertia;

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

        /// The same point with weight exactly `+1` (unitized): divided by its signed weight.
        /// A meet or a reflection can return a point with any weight, even a negative one, and
        /// `normalized()` keeps the sign; differences of unitized points are directions. Not
        /// for directions (weight 0).
        ///
        /// ```
        /// use gax::pga3d::Point;
        /// let p = Point::<(), f64>::new(2.0, 4.0, 6.0, -2.0);
        /// assert_eq!(p.unitized().c, [-1.0, -2.0, -3.0, 1.0]);
        /// ```
        #[inline]
        pub fn unitized(self) -> Self {
            self.gp(self.e123().recip())
        }

        /// The ideal norm `sqrt(x² + y² + z²)`: the length of a direction (a velocity, a
        /// displacement). `norm()` is the weight, which is 0 for directions.
        ///
        /// ```
        /// use gax::pga3d::Point;
        /// assert_eq!(Point::direction(2.0, 3.0, 6.0).ideal_norm(), 7.0);
        /// ```
        #[inline]
        pub fn ideal_norm(self) -> T {
            let (x, y, z) = (self.e032(), self.e013(), self.e021());
            (x * x + y * y + z * z).sqrt()
        }
    }

    impl<T: Real> Plane<(), T> {
        /// The plane `n · x = d` with normal `n` (not necessarily unit length).
        #[inline]
        pub fn from_normal(n: [T; 3], d: T) -> Self {
            Plane::new(n[0], n[1], n[2], -d)
        }

        /// The plane through the origin perpendicular to the direction `d`.
        #[inline]
        pub fn orthogonal_to(d: Point<(), T>) -> Self {
            Plane::new(d.e032(), d.e013(), d.e021(), T::zero())
        }

        /// Reflect a point (or a direction) in this plane: the sandwich with the normalized
        /// plane, which in PGA3D needs no sign correction for points or directions.
        ///
        /// ```
        /// use gax::pga3d::{Plane, Point};
        /// let wall = Plane::from_normal([1.0, 0.0, 0.0], 0.0);
        /// let v = wall.reflect(Point::direction(1.0, 0.5, 0.0));
        /// assert_eq!([v.e032(), v.e013(), v.e021()], [-1.0, 0.5, 0.0]);
        /// ```
        #[inline]
        pub fn reflect(self, x: Point<(), T>) -> Point<(), T> {
            self.normalized() >> x
        }
    }

    /// The inertia of a rigid body in its principal frame (centre of mass at the origin, principal
    /// axes along x, y, z): a map from twists to forques (momenta) stored as four numbers
    /// instead of the 36 of a dense `Line<(Line,)>`.
    ///
    /// It is the compact representation of `Σ m X ∨ (X × B)` over the body's mass points: the
    /// translation part of a twist maps to linear momentum through the mass, and the rotation
    /// part to angular momentum through the principal moments. Anything a dense map offers is
    /// available through [`PrincipalInertia::to_map`], for example moving the body with
    /// `m >> inertia.to_map().of(m << Line::slot())`.
    ///
    /// ```
    /// use gax::pga3d::{Line, PrincipalInertia};
    /// let inertia = PrincipalInertia::new(2.0, [0.5, 0.75, 1.0]);
    /// let twist = Line::new(0.1, 0.2, 0.3, 1.0, 0.0, 0.0);
    /// let momentum = inertia.of(twist);
    /// assert_eq!(inertia.inverse_of(momentum), twist);
    /// ```
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct PrincipalInertia<T> {
        /// The total mass.
        pub mass: T,
        /// The principal moments about x, y and z.
        pub moments: [T; 3],
    }

    impl<T: Real> PrincipalInertia<T> {
        /// A body of the given mass and principal moments.
        #[inline]
        pub fn new(mass: T, moments: [T; 3]) -> Self {
            PrincipalInertia { mass, moments }
        }

        /// The momentum of a twist: six products.
        #[inline]
        pub fn of(self, b: Line<(), T>) -> Line<(), T> {
            let (m, i) = (self.mass, self.moments);
            Line::new(
                m * b.e01(),
                m * b.e02(),
                m * b.e03(),
                i[0] * b.e23(),
                i[1] * b.e31(),
                i[2] * b.e12(),
            )
        }

        /// The twist of a momentum, the inverse map: four reciprocals and six products.
        #[inline]
        pub fn inverse_of(self, f: Line<(), T>) -> Line<(), T> {
            let r = self.mass.recip();
            let i = self.moments.map(Real::recip);
            Line::new(
                i[0] * f.e01(),
                i[1] * f.e02(),
                i[2] * f.e03(),
                r * f.e23(),
                r * f.e31(),
                r * f.e12(),
            )
        }

        /// The dense map `Line <- Line`.
        #[inline]
        pub fn to_map(self) -> Line<(Line,), T> {
            let z = T::zero();
            let (m, i) = (self.mass, self.moments);
            Line::from_coeffs([
                [z, z, z, m, z, z],
                [z, z, z, z, m, z],
                [z, z, z, z, z, m],
                [i[0], z, z, z, z, z],
                [z, i[1], z, z, z, z],
                [z, z, i[2], z, z, z],
            ])
        }

        /// The kinetic energy form `B & I[B]` on twists (twice the kinetic energy).
        #[inline]
        pub fn energy_form(self) -> crate::pga3d::Scalar<(Line, Line), T> {
            Line::slot() & self.to_map()
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

        /// The shortest rotation (about an axis through the origin) that turns the direction
        /// `from` into the direction `to`. With `A` and `B` the planes through the origin
        /// orthogonal to them, the axis is their meet `A ^ B` and the angle is
        /// `atan2(|A ^ B|, A | B)`: the meet's weight and the inner product are the sine and
        /// cosine, scaled alike, so no normalization is needed and the angle is well conditioned
        /// everywhere, a half turn included (a formula like `normalize(1 + B A)` loses the small
        /// scalar part to cancellation there). Opposite directions get a half turn about an
        /// axis perpendicular to both.
        ///
        /// ```
        /// use gax::pga3d::{Motor, Point};
        /// let z = Point::<(), f64>::direction(0.0, 0.0, 1.0);
        /// let r = Motor::rotation_between(z, Point::direction(0.0, 3.0, 0.0));
        /// let d = r >> Point::direction(0.0, 0.0, 1.0);
        /// assert!((d.e013() - 1.0).abs() < 1e-12 && d.e021().abs() < 1e-12);
        /// ```
        pub fn rotation_between(from: Point<(), T>, to: Point<(), T>) -> Unit<Self> {
            let (o, zero) = (T::one(), T::zero());
            let a = Plane::orthogonal_to(from);
            let b = Plane::orthogonal_to(to);
            let meet = a ^ b;
            let angle = meet.norm().atan2((a | b).s());
            // Nearly parallel or opposite, the meet vanishes: turn about an axis orthogonal to
            // `A` instead (the meet of `A` with a plane orthogonal to it, `e` less its part
            // along `A`). The angle is right either way; branch-free, so that it traces.
            let an = a.normalized().into_inner();
            let far = T::select_lt(an.e1().abs(), T::from_f64(0.9), o, zero);
            let e = Plane::new(far, o - far, zero, zero);
            let other = an ^ (e - an.gp((e | an).s()));
            let small = meet.norm() * (a.norm() * b.norm()).recip();
            let axis = Line::from_coeffs(core::array::from_fn(|i| {
                T::select_lt(small, T::from_f64(1e-6), other.c[i], meet.c[i])
            }));
            Self::rotation(axis, angle)
        }

        /// The motor of a camera (or any frame) at `eye` whose forward axis `+z` points at
        /// `target` and whose `+y` axis is as close to `up` as it can be (no roll): the
        /// rotation between the axes, then a roll about the forward axis, then the translation.
        ///
        /// ```
        /// use gax::pga3d::{Motor, Point};
        /// let m = Motor::look_at(
        ///     Point::<(), f64>::xyz(1.0, 2.0, 3.0),
        ///     Point::xyz(1.0, 2.0, 13.0),
        ///     Point::direction(0.0, 1.0, 0.0),
        /// );
        /// let f = m >> Point::direction(0.0, 0.0, 1.0);
        /// let u = m >> Point::direction(0.0, 1.0, 0.0);
        /// assert!((f.e021() - 1.0).abs() < 1e-12 && (u.e013() - 1.0).abs() < 1e-12);
        /// assert_eq!((m >> Point::xyz(0.0, 0.0, 0.0)).to_euclidean(), [1.0, 2.0, 3.0]);
        /// ```
        pub fn look_at(eye: Point<(), T>, target: Point<(), T>, up: Point<(), T>) -> Unit<Self> {
            let (o, zero) = (T::one(), T::zero());
            let [ex, ey, ez] = eye.to_euclidean();
            let [tx, ty, tz] = target.to_euclidean();
            let forward = Point::direction(tx - ex, ty - ey, tz - ez);
            let turn = Self::rotation_between(Point::direction(zero, zero, o), forward);
            // Where `+y` went, and where it should be: `up` less its part along `forward`.
            let f = Plane::orthogonal_to(forward).normalized().into_inner();
            let u = Plane::orthogonal_to(up);
            let want = u - f.gp((u | f).s());
            let have = Plane::orthogonal_to(turn >> Point::direction(zero, o, zero));
            let roll = Self::rotation_between(
                Point::direction(have.e1(), have.e2(), have.e3()),
                Point::direction(want.e1(), want.e2(), want.e3()),
            );
            // `up` along `forward`: no roll is defined; keep the turn alone.
            let id = Self::translation(zero, zero, zero);
            let n = want.norm();
            let roll = Unit::new_unchecked(Self::from_coeffs(core::array::from_fn(|i| {
                T::select_lt(n, T::from_f64(1e-9), id.c[i], roll.c[i])
            })));
            Self::translation(ex, ey, ez) * roll * turn
        }
    }

    impl<T: Real> Line<(), T> {
        /// The twist (a bivector: a line) of a translation at velocity `(vx, vy, vz)`:
        /// `(B * t).exp()` is `Motor::translation(t vx, t vy, t vz)`. Twists add.
        #[inline]
        pub fn translation_twist(vx: T, vy: T, vz: T) -> Self {
            let h = T::from_f64(-0.5);
            let z = T::zero();
            Line::new(z, z, z, vx * h, vy * h, vz * h)
        }

        /// The twist of a rotation at `omega` radians per unit time about the line `axis`
        /// (right-handed, as [`Motor::rotation`]): `(B * t).exp()` is
        /// `Motor::rotation(axis, t omega)`.
        #[inline]
        pub fn rotation_twist(axis: Line<(), T>, omega: T) -> Self {
            axis.normalized().into_inner().gp(omega * T::from_f64(-0.5))
        }
    }
}

#[cfg(feature = "pga2d")]
mod pga2d_extras {
    use crate::pga2d::{Line, Motor, Point};
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

        /// The same point with weight exactly `+1` (unitized): divided by its signed weight.
        /// A meet or a reflection can return a point with any weight, even a negative one (the
        /// sandwich of a line with a point comes out negated in PGA2D), and `normalized()`
        /// keeps the sign; differences of unitized points are directions. Not for directions
        /// (weight 0).
        ///
        /// ```
        /// use gax::pga2d::Point;
        /// let p = Point::<(), f64>::new(2.0, 4.0, -2.0);
        /// assert_eq!(p.unitized().c, [-1.0, -2.0, 1.0]);
        /// ```
        #[inline]
        pub fn unitized(self) -> Self {
            self.gp(self.e12().recip())
        }

        /// The ideal norm `sqrt(x² + y²)`: the length of a direction (a velocity, a
        /// displacement). `norm()` is the weight, which is 0 for directions.
        ///
        /// ```
        /// use gax::pga2d::Point;
        /// assert_eq!(Point::direction(3.0, 4.0).ideal_norm(), 5.0);
        /// ```
        #[inline]
        pub fn ideal_norm(self) -> T {
            let (x, y) = (self.e20(), self.e01());
            (x * x + y * y).sqrt()
        }

        /// The twist (a bivector: in PGA2D, a point) of a translation at velocity `(vx, vy)`:
        /// `(B * t).exp()` is `Motor::translation(t vx, t vy)`. Twists add, so a moving,
        /// turning body's twist is the sum of a translation and a rotation twist.
        ///
        /// ```
        /// use gax::pga2d::{Motor, Point};
        /// let b = Point::<(), f64>::translation_twist(3.0, -1.0);
        /// let p = (b.gp(2.0).exp() >> Point::xy(0.0, 0.0)).to_euclidean();
        /// assert!((p[0] - 6.0).abs() < 1e-12 && (p[1] + 2.0).abs() < 1e-12);
        /// ```
        #[inline]
        pub fn translation_twist(vx: T, vy: T) -> Self {
            let h = T::from_f64(0.5);
            Point::new(vy * h, -vx * h, T::zero())
        }

        /// The twist of a rotation at `omega` radians per unit time, counterclockwise, about
        /// `center`: `(B * t).exp()` is `Motor::rotation(center, t omega)`.
        #[inline]
        pub fn rotation_twist(center: Point<(), T>, omega: T) -> Self {
            center
                .normalized()
                .into_inner()
                .gp(omega * T::from_f64(-0.5))
        }
    }

    impl<T: Real> Line<(), T> {
        /// Reflect a point (or a direction) in this line. In PGA2D the sandwich of a line, an
        /// odd versor, with a point, an even element, comes out negated: harmless for a point
        /// (`-p` is the same point), but for a direction it is the opposite velocity. This
        /// applies the sign.
        ///
        /// ```
        /// use gax::pga2d::Point;
        /// let wall = Point::xy(0.0, 0.0) & Point::xy(0.0, 1.0);
        /// let v = wall.reflect(Point::direction(1.0, 0.5));
        /// assert_eq!([v.e20(), v.e01(), v.e12()], [-1.0, 0.5, 0.0]);
        /// ```
        #[inline]
        pub fn reflect(self, x: Point<(), T>) -> Point<(), T> {
            -(self.normalized() >> x)
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

        /// The shortest rotation about the origin that turns the direction `from` into the
        /// direction `to`. With `A` and `B` the lines through the origin orthogonal to them,
        /// the angle is `atan2` of the weight of their meet (the origin) and their inner
        /// product: the sine and cosine, scaled alike, so no normalization is needed and the
        /// angle is well conditioned everywhere, a half turn included. Its angle, the signed
        /// angle from `from` to `to` in `[-π, π]`, is [`Motor::angle`].
        ///
        /// ```
        /// use gax::pga2d::{Motor, Point};
        /// let r = Motor::rotation_between(Point::<(), f64>::direction(1.0, 0.0), Point::direction(0.0, 2.0));
        /// assert!((r.angle() - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
        /// ```
        pub fn rotation_between(from: Point<(), T>, to: Point<(), T>) -> Unit<Self> {
            let zero = T::zero();
            let a = Line::new(from.e20(), from.e01(), zero);
            let b = Line::new(to.e20(), to.e01(), zero);
            let sine = (a ^ b).e12();
            let angle = sine.atan2((a | b).s());
            Self::rotation(Point::xy(zero, zero), angle)
        }
    }

    impl<T: Real> Motor<(), T> {
        /// The angle of a rotation (a unit motor), counterclockwise, in `[-π, π]`: from its
        /// logarithm, whose point part is `-angle/2` times the (unit) centre.
        ///
        /// ```
        /// use gax::pga2d::{Motor, Point};
        /// let r = Motor::rotation(Point::<(), f64>::xy(3.0, 1.0), -0.4);
        /// assert!((r.angle() + 0.4).abs() < 1e-12);
        /// ```
        pub fn angle(self) -> T {
            let log: Point<(), T> = Unit::new_unchecked(self).log();
            log.e12() * T::from_f64(-2.0)
        }
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "pga3d")]
    #[test]
    #[allow(clippy::float_cmp)] // the same operations in the same order: exact equality
    fn principal_inertia_is_the_compact_point_mass_inertia() {
        use crate::pga3d::{Line, Point, PrincipalInertia};
        // Six unit masses on the axes: mass 6, moments (2·2² + 2·3², 2·1² + 2·3², 2·1² + 2·2²).
        let pts = [
            (1.0, 0.0, 0.0),
            (-1.0, 0.0, 0.0),
            (0.0, 2.0, 0.0),
            (0.0, -2.0, 0.0),
            (0.0, 0.0, 3.0),
            (0.0, 0.0, -3.0),
        ];
        let mut dense: Line<(Line,), f64> = Line::zero();
        for (x, y, z) in pts {
            let p = Point::xyz(x, y, z);
            dense += p & p.commutator(Line::slot()); // Σ m X ∨ (X × B), in extensor form
        }
        let compact = PrincipalInertia::new(6.0, [26.0, 20.0, 10.0]);
        assert_eq!(compact.to_map(), dense);
        let b = Line::new(0.3, -0.2, 0.1, 1.0, 2.0, -0.5);
        assert_eq!(compact.of(b), dense.of(b));
        let back = compact.inverse_of(compact.of(b));
        assert!(back.c.iter().zip(b.c).all(|(x, y)| (x - y).abs() < 1e-15));
        assert_eq!(
            dense
                .inverse()
                .of(compact.of(b))
                .c
                .map(|x: f64| (x * 1e12).round()),
            b.c.map(|x| (x * 1e12).round())
        );
    }

    /// Near a half turn the rotation between directions stays precise in `f32`: right to a
    /// few ulps however close to opposite the directions are (`normalize(1 + B A)` was off by
    /// up to `1e-4` there, the error of the normalization over the distance from a half turn).
    #[cfg(all(feature = "pga2d", feature = "pga3d"))]
    #[test]
    fn rotations_between_nearly_opposite_directions_are_precise() {
        for k in 1..=12 {
            let delta = 10f64.powi(-k / 2) * 0.3;
            let theta = core::f64::consts::PI - delta;
            // 2D: the angle, from the logarithm.
            let a = crate::pga2d::Point::<(), f32>::direction(1.0, 0.0);
            let b = crate::pga2d::Point::direction(theta.cos() as f32, theta.sin() as f32);
            let r = crate::pga2d::Motor::rotation_between(a, b);
            // Against the exact angle of the rounded input.
            let exact = f64::from(b.e01()).atan2(f64::from(b.e20()));
            let err = (f64::from(r.angle()) - exact).abs();
            assert!(err < 4e-7, "2D, {delta:e} from a half turn: off by {err:e}");
            // 3D: the image of `from` is `to`.
            let a = crate::pga3d::Point::<(), f32>::direction(1.0, 0.0, 0.0);
            let (c, s) = (theta.cos() as f32, theta.sin() as f32);
            let b = crate::pga3d::Point::direction(c, s * 0.6, s * 0.8);
            let d = crate::pga3d::Motor::rotation_between(a, b) >> a;
            let err =
                (d.e032() - c).abs() + (d.e013() - s * 0.6).abs() + (d.e021() - s * 0.8).abs();
            // A few ulps per coordinate: the rounding of the sandwich itself.
            assert!(
                err < 1.5e-6,
                "3D, {delta:e} from a half turn: off by {err:e}"
            );
        }
    }

    #[cfg(feature = "pga2d")]
    #[test]
    fn plane_rotations_between_directions() {
        use crate::pga2d::{Motor, Point};
        type Case = ((f64, f64), (f64, f64), f64);
        let cases: [Case; 4] = [
            ((1.0, 0.0), (0.0, 3.0), core::f64::consts::FRAC_PI_2),
            ((0.0, 1.0), (1.0, 0.0), -core::f64::consts::FRAC_PI_2),
            ((1.0, 1.0), (-1.0, 1.0), core::f64::consts::FRAC_PI_2),
            ((2.0, -1.0), (2.0, -1.0), 0.0),
        ];
        for (a, b, want) in cases {
            let r = Motor::rotation_between(Point::direction(a.0, a.1), Point::direction(b.0, b.1));
            assert!(
                (r.angle() - want).abs() < 1e-9,
                "{a:?} {b:?}: {}",
                r.angle()
            );
            let d = r >> Point::direction(a.0, a.1);
            let n = (a.0 * a.0 + a.1 * a.1).sqrt() / (b.0 * b.0 + b.1 * b.1).sqrt();
            assert!((d.e20() - b.0 * n).abs() < 1e-9 && (d.e01() - b.1 * n).abs() < 1e-9);
        }
        // Opposite: a half turn.
        let r = Motor::rotation_between(Point::direction(1.0f64, 0.0), Point::direction(-1.0, 0.0));
        assert!((r.angle().abs() - core::f64::consts::PI).abs() < 1e-9);
    }

    #[cfg(feature = "pga3d")]
    #[test]
    fn rotations_between_directions_and_look_at() {
        use crate::pga3d::{Motor, Point};
        let dirs: [[f64; 3]; 7] = [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.3, -0.8, 0.5],
            [-0.3, 0.8, -0.5],
            [0.0, 1.0, 0.0],
        ];
        let unit = |d: [f64; 3]| {
            let n = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            [d[0] / n, d[1] / n, d[2] / n]
        };
        for a in dirs {
            for b in dirs {
                let r = Motor::rotation_between(
                    Point::direction(a[0], a[1], a[2]),
                    Point::direction(b[0], b[1], b[2]),
                );
                let got = r >> Point::direction(a[0], a[1], a[2]);
                let n: f64 = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
                let (a, b) = (unit(a), unit(b));
                let g = [got.e032() / n, got.e013() / n, got.e021() / n];
                assert!(
                    g.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-9),
                    "{a:?} -> {b:?}: {g:?}"
                );
                // A rotation: lengths kept, no translation.
                let o = (r >> Point::xyz(0.0, 0.0, 0.0)).to_euclidean();
                assert!(o.iter().all(|x| x.abs() < 1e-12));
            }
        }
        // Look-at: forward to the target, right axis level, up upwards; and with `up` along
        // the view, still a proper frame.
        let up = Point::direction(0.0, 1.0, 0.0);
        for t in [
            [3.0, 1.0, 5.0],
            [-2.0, -4.0, 1.0],
            [0.0, 0.0, -7.0],
            [0.0, 5.0, 0.0],
        ] {
            let m = Motor::look_at(Point::xyz(0.0, 0.0, 0.0), Point::xyz(t[0], t[1], t[2]), up);
            let f = m >> Point::direction(0.0, 0.0, 1.0);
            let x = m >> Point::direction(1.0, 0.0, 0.0);
            let y = m >> Point::direction(0.0, 1.0, 0.0);
            let t = unit(t);
            assert!(
                ((f.e032() - t[0]).abs() + (f.e013() - t[1]).abs() + (f.e021() - t[2]).abs())
                    < 1e-9
            );
            if t[1].abs() < 0.99 {
                assert!(x.e013().abs() < 1e-9, "rolled: {t:?}");
                assert!(y.e013() > 0.0);
            }
            assert!((f.ideal_norm() - 1.0).abs() < 1e-9 && (x.ideal_norm() - 1.0).abs() < 1e-9);
        }
    }

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
        // Twists: exp(t B) is the motor of the motion for time t.
        let c = Point::xy(0.5, -1.5);
        for t in [0.3, 1.0, 2.5] {
            let tr = Point::translation_twist(3.0, -1.0).gp(t).exp().into_inner();
            let want = Motor::translation(3.0 * t, -t).into_inner();
            assert!(
                tr.c.iter()
                    .zip(want.c)
                    .all(|(a, b): (&f64, f64)| (a - b).abs() < 1e-12)
            );
            let ro = Point::rotation_twist(c, 0.7).gp(t).exp().into_inner();
            let want = Motor::rotation(c, 0.7 * t).into_inner();
            assert!(
                ro.c.iter()
                    .zip(want.c)
                    .all(|(a, b): (&f64, f64)| (a - b).abs() < 1e-12)
            );
        }
    }

    #[cfg(feature = "pga3d")]
    #[test]
    fn pga3d_twists() {
        use crate::pga3d::{Line, Motor, Point};
        let axis = Point::xyz(1.0, 0.0, 0.0) & Point::xyz(1.0, 0.5, 2.0);
        for t in [0.3, 1.0, 2.5] {
            let tr = Line::translation_twist(3.0, -1.0, 0.5)
                .gp(t)
                .exp()
                .into_inner();
            let want = Motor::translation(3.0 * t, -t, 0.5 * t).into_inner();
            assert!(
                tr.c.iter()
                    .zip(want.c)
                    .all(|(a, b): (&f64, f64)| (a - b).abs() < 1e-12)
            );
            let ro = Line::rotation_twist(axis, 0.7).gp(t).exp().into_inner();
            let want = Motor::rotation(axis, 0.7 * t).into_inner();
            assert!(
                ro.c.iter()
                    .zip(want.c)
                    .all(|(a, b): (&f64, f64)| (a - b).abs() < 1e-12)
            );
        }
    }
}
