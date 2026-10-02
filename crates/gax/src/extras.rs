//! Hand-written conveniences for the standard algebras: constructors with geometric meaning.
//!
//! These are inherent functions on the generated types. For example, `Point::xyz(x, y, z)` and
//! `Motor::translation(dx, dy, dz)` are thin wrappers that fix the sign conventions in one place.

#[cfg(feature = "pga3d")]
pub use pga3d_extras::PrincipalInertia;

#[cfg(any(feature = "pga2d", feature = "pga3d"))]
/// The elements a motor carries onto one another, for `Motor::between`: planes, lines and
/// points in PGA3D, lines and points in PGA2D.
///
/// ```
/// use gax::pga3d::{Motor, Plane};
/// let a = Plane::<(), f64>::from_normal([1.0, 0.0, 0.0], 0.0);
/// let b = Plane::from_normal([0.0, 1.0, 0.0], 2.0);
/// let m = Motor::between(a, b);
/// let moved = (m >> a).normalized().into_inner();
/// let want = b.normalized().into_inner();
/// assert!(moved.c.iter().zip(want.c).all(|(x, y)| (x - y).abs() < 1e-12));
/// ```
pub trait Between<M>: Sized {
    /// The motor that carries `a` onto `b`.
    fn between(a: Self, b: Self) -> crate::Unit<M>;
}

#[cfg(any(feature = "pga2d", feature = "pga3d"))]
/// `exp(log(b / a) / 2)`, the square root of the ratio: the motion that carries `a` onto `b`.
/// The ratio is normalized first, so `a` and `b` need not be.
macro_rules! between_by_ratio {
    ($alg:ident: $($k:ident),* => $biv:ident) => {$(
        impl<T: crate::Real> Between<crate::$alg::Motor<(), T>> for crate::$alg::$k<(), T> {
            #[inline]
            fn between(a: Self, b: Self) -> crate::Unit<crate::$alg::Motor<(), T>> {
                let half: crate::$alg::$biv<(), T> = (b / a).normalized().log();
                half.gp(T::from_f64(0.5)).exp()
            }
        }
    )*};
}
#[cfg(feature = "pga3d")]
between_by_ratio!(pga3d: Plane, Line => Line);
#[cfg(feature = "pga2d")]
between_by_ratio!(pga2d: Line => Point);

#[cfg(any(feature = "pga2d", feature = "pga3d"))]
/// Points: the translation between them. No motor carries a point onto its negative (motors
/// keep the sign of the weight), so the points are unitized first.
macro_rules! between_points {
    ($alg:ident => $biv:ident) => {
        impl<T: crate::Real> Between<crate::$alg::Motor<(), T>> for crate::$alg::Point<(), T> {
            #[inline]
            fn between(a: Self, b: Self) -> crate::Unit<crate::$alg::Motor<(), T>> {
                let half: crate::$alg::$biv<(), T> =
                    (b.unitized() / a.unitized()).normalized().log();
                half.gp(T::from_f64(0.5)).exp()
            }
        }
    };
}
#[cfg(feature = "pga3d")]
between_points!(pga3d => Line);
#[cfg(feature = "pga2d")]
between_points!(pga2d => Point);

/// Written once for PGA2D and PGA3D through the value traits: their kinds differ (a rotation is
/// about a point in 2D and about a line in 3D), the expressions do not.
#[cfg(any(feature = "pga2d", feature = "pga3d"))]
mod shared {
    use crate::{Exp, Extensor, Gp, GradePart, Log, Normalize, Real, Reverse, Unit};

    /// `−(rate/2) x̂`: the twist of a rotation at `rate` about the flat `x` (a line in PGA3D, a
    /// point in PGA2D), normalized first.
    #[inline]
    pub fn rotation_twist<X: Normalize + Gp<T, Output = X>, T: Real>(x: X, rate: T) -> X {
        x.normalized().into_inner().gp(rate * T::from_f64(-0.5))
    }

    /// The rigid motion from `a` to `b` at `t`, `a exp(t log(~a b))`, the shorter way: the
    /// relative motor taken with its scalar part non-negative (`m` and `−m` are one motion).
    #[inline]
    pub fn interpolate<M, B, T>(a: Unit<M>, b: Unit<M>, t: T) -> Unit<M>
    where
        T: Real,
        M: Extensor<Slots = (), Coef = T> + Reverse<Output = M> + Gp<M, Output = M>,
        M: Gp<T, Output = M>,
        M::Kind: GradePart<0>,
        Unit<M>: Log<B>,
        B: Gp<T, Output = B> + Exp<Output = Unit<M>>,
    {
        let rel = (a.reverse() * b).into_inner();
        let s = crate::cast::grade::<M, 0>(&rel).coeffs().as_ref()[0];
        let sign = T::select_lt(s, T::zero(), -T::one(), T::one());
        let rel: Unit<M> = Unit::new_unchecked(rel.gp(sign));
        a * rel.log().gp(t).exp()
    }
}

#[cfg(feature = "pga3d")]
mod pga3d_extras {
    use crate::pga3d::{Line, Motor, Plane, Point, Translator};
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
        ///
        /// ```
        /// use gax::pga3d::Point;
        /// let up = Point::<(), f64>::direction(0.0, 0.0, 1.0);
        /// assert_eq!(up.e123(), 0.0);
        /// // The difference of two points is a direction.
        /// assert_eq!(Point::xyz(1.0, 2.0, 3.0) - Point::xyz(1.0, 2.0, 2.0), up);
        /// ```
        #[inline]
        pub fn direction(x: T, y: T, z: T) -> Self {
            Point::new(x, y, z, T::zero())
        }

        /// Euclidean coordinates `(x/w, y/w, z/w)`. Not finite for directions.
        ///
        /// ```
        /// use gax::pga3d::Point;
        /// assert_eq!(Point::<(), f64>::new(2.0, 4.0, 6.0, 2.0).to_euclidean(), [1.0, 2.0, 3.0]);
        /// ```
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
        ///
        /// ```
        /// use gax::pga3d::{Plane, Point};
        /// let floor = Plane::<(), f64>::from_normal([0.0, 0.0, 1.0], 2.0); // z = 2
        /// // The join with a point is its signed distance, positive along the normal.
        /// assert_eq!((floor & Point::xyz(1.0, 1.0, 5.0)).s(), 3.0);
        /// ```
        #[inline]
        pub fn from_normal(n: [T; 3], d: T) -> Self {
            Plane::new(n[0], n[1], n[2], -d)
        }

        /// The plane through the origin perpendicular to the direction `d`.
        ///
        /// ```
        /// use gax::pga3d::{Plane, Point};
        /// let p = Plane::<(), f64>::orthogonal_to(Point::direction(0.0, 0.0, 2.0));
        /// // Through the origin, normal (0, 0, 2): the distance of a point, scaled by 2.
        /// assert_eq!((p & Point::xyz(3.0, 4.0, 1.0)).s(), 2.0);
        /// ```
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
        ///
        /// ```
        /// use gax::pga3d::PrincipalInertia;
        /// let body = PrincipalInertia::new(2.0, [0.5, 0.75, 1.0]);
        /// assert_eq!((body.mass, body.moments[2]), (2.0, 1.0));
        /// ```
        #[inline]
        pub fn new(mass: T, moments: [T; 3]) -> Self {
            PrincipalInertia { mass, moments }
        }

        /// The momentum of a twist: six products.
        ///
        /// ```
        /// use gax::pga3d::{Line, PrincipalInertia};
        /// let body = PrincipalInertia::new(2.0, [0.5, 0.75, 1.0]);
        /// let b = Line::new(0.1, 0.2, 0.3, 1.0, 0.0, 0.0);
        /// let f = body.of(b);
        /// // The velocity (the ideal part) goes through the mass to the Euclidean part of the
        /// // momentum, the angular velocity through the moments to its ideal part: a dual line.
        /// assert_eq!((f.e23(), f.e01()), (2.0 * b.e01(), 0.5 * b.e23()));
        /// ```
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
        ///
        /// ```
        /// use gax::pga3d::{Line, PrincipalInertia};
        /// let body = PrincipalInertia::new(4.0, [0.5, 0.25, 2.0]);
        /// let b = Line::new(1.0, -0.5, 0.25, 2.0, 0.5, -1.0);
        /// assert_eq!(body.inverse_of(body.of(b)), b);
        /// ```
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
        ///
        /// ```
        /// use gax::pga3d::{Line, PrincipalInertia};
        /// let body = PrincipalInertia::new(2.0, [0.5, 0.75, 1.0]);
        /// let b = Line::new(0.1, 0.2, 0.3, 1.0, -1.0, 0.5);
        /// assert_eq!(body.to_map().of(b), body.of(b));
        /// ```
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
        ///
        /// ```
        /// use gax::pga3d::{Line, PrincipalInertia};
        /// let body = PrincipalInertia::new(2.0, [0.5, 0.75, 1.0]);
        /// let b = Line::new(0.1, 0.2, 0.3, 1.0, -1.0, 0.5);
        /// let twice_energy: f64 = body.energy_form().of(b).of(b).s();
        /// assert!((twice_energy - (b & body.of(b)).s()).abs() < 1e-12);
        /// ```
        #[inline]
        pub fn energy_form(self) -> crate::pga3d::Scalar<(Line, Line), T> {
            Line::slot() & self.to_map()
        }
    }

    impl<T: Real> Motor<(), T> {
        /// The motor that carries `a` onto `b`: planes, lines or points. It is the square root
        /// of their ratio, `sqrt(b / a)`: `b / a` is the motion from `a` to `b` twice (the
        /// product of two reflections, for planes), and its square root goes halfway. The
        /// result is exact for oriented planes and lines, `m >> a == b` (`a` and `b` need not
        /// be normalized); points are taken with positive weight.
        ///
        /// Two planes meet in the axis of a rotation (parallel planes, a translation); two
        /// lines give a screw. Computed as `exp(log(b / a) / 2)`, which stays precise near a
        /// half turn, where `normalize(1 + b / a)` (the cheaper `(b / a).sqrt()` for unit `a`
        /// and `b`) loses the angle to cancellation. Not defined for `b = -a`: a half turn about
        /// any of infinitely many axes.
        ///
        /// ```
        /// use gax::pga3d::{Motor, Plane, Point};
        /// let (p, q) = (Point::<(), f64>::xyz(1.0, 2.0, 3.0), Point::xyz(0.0, 5.0, 3.0));
        /// let (r, s) = (Point::xyz(1.0, 2.0, 4.0), Point::xyz(0.0, 6.0, 3.0));
        /// // The motor that lays the line p r onto the line q s.
        /// let m = Motor::between(p & r, q & s);
        /// let l = (m >> (p & r)).normalized().into_inner();
        /// let want = (q & s).normalized().into_inner();
        /// assert!(l.c.iter().zip(want.c).all(|(x, y)| (x - y).abs() < 1e-12));
        /// assert!(((m >> p).to_euclidean()[0] - 0.0).abs() < 1e-12); // p lands on q s
        /// ```
        #[inline]
        pub fn between<X: crate::extras::Between<Self>>(a: X, b: X) -> Unit<Self> {
            X::between(a, b)
        }

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
            // A translator `1 − (d/2) e0`, as a motor.
            let t = Translator::new(T::one(), dx * h, dy * h, dz * h);
            Unit::new_unchecked(t.cast::<Motor>())
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
            crate::extras::shared::rotation_twist(axis, angle).exp()
        }

        /// The rotation by `angle` about the axis through the origin with direction `(x, y, z)`.
        ///
        /// ```
        /// use gax::pga3d::{Motor, Point};
        /// let r = Motor::<(), f64>::rotation_about(0.0, 0.0, 1.0, std::f64::consts::FRAC_PI_2);
        /// let [x, y, z] = (r >> Point::xyz(1.0, 0.0, 0.0)).to_euclidean();
        /// assert!(x.abs() < 1e-12 && (y - 1.0).abs() < 1e-12 && z.abs() < 1e-12);
        /// ```
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
            let axis = crate::select_lt(small, T::from_f64(1e-6), other, meet);
            Self::rotation(axis, angle)
        }

        /// The screw motion from `a` to `b`, at `t` (`0` is `a`, `1` is `b`): `a exp(t log(~a b))`,
        /// the shorter way. `m` and `-m` are the same motion, and constructions such as
        /// [`Motor::look_at`] may return either as their input moves; `log` of the one with a
        /// negative scalar part is the long way round, nearly a full turn. So the relative motor
        /// is taken with its scalar part non-negative. Branch-free, so that it traces.
        ///
        /// ```
        /// use gax::pga3d::{Motor, Point};
        /// let a = Motor::<(), f64>::rotation_about(0.0, 0.0, 1.0, 0.1);
        /// let b = Motor::rotation_about(0.0, 0.0, 1.0, 0.3);
        /// // The same motion as `b`, the other sign: still halfway is 0.2 rad, not a long turn.
        /// let b = gax::Unit::new_unchecked(b.into_inner() * -1.0);
        /// let d = Motor::interpolate(a, b, 0.5) >> Point::direction(1.0, 0.0, 0.0);
        /// assert!((d.e013() - 0.2f64.sin()).abs() < 1e-12);
        /// ```
        pub fn interpolate(a: Unit<Self>, b: Unit<Self>, t: T) -> Unit<Self> {
            crate::extras::shared::interpolate::<Self, Line<(), T>, T>(a, b, t)
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
            // The frame: z forward, x = up × z (normalized), y = z × x, then its rotor
            // (Shepperd's method, well conditioned for every rotation). Built from the frame
            // rather than as a turn and a roll: a shortest rotation between nearly opposite
            // directions takes its axis from rounding, and looking nearly along `up` (from above,
            // `+z` up) once turned an `f32` camera round.
            let unit = |v: [T; 3]| {
                let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
                v.map(|c| c / n)
            };
            let cross = |a: [T; 3], b: [T; 3]| {
                [
                    a[1] * b[2] - a[2] * b[1],
                    a[2] * b[0] - a[0] * b[2],
                    a[0] * b[1] - a[1] * b[0],
                ]
            };
            let z = unit([tx - ex, ty - ey, tz - ez]);
            let u = [up.e032(), up.e013(), up.e021()];
            let x = cross(u, z);
            // `up` along `forward`: no roll is defined; any x orthogonal to z (from the axis z is
            // least along).
            let other = T::select_lt(z[0].abs(), T::from_f64(0.5), o, zero);
            let fallback = cross([other, o - other, zero], z);
            let norm2 = |v: [T; 3]| v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
            let tiny = T::epsilon() * T::epsilon() * norm2(u);
            let x = unit(core::array::from_fn(|i| {
                T::select_lt(norm2(x), tiny, fallback[i], x[i])
            }));
            let y = cross(z, x);
            Self::translation(ex, ey, ez) * crate::moments::rotor_from_frame([x, y, z])
        }
    }

    impl<T: Real> Line<(), T> {
        /// The twist (a bivector: a line) of a translation at velocity `(vx, vy, vz)`:
        /// `(B * t).exp()` is `Motor::translation(t vx, t vy, t vz)`. Twists add.
        ///
        /// ```
        /// use gax::pga3d::{Line, Motor, Point};
        /// let twist = Line::<(), f64>::translation_twist(1.0, 2.0, 0.0);
        /// // Two seconds at that velocity.
        /// let p = (twist.gp(2.0).exp() >> Point::xyz(0.0, 0.0, 0.0)).to_euclidean();
        /// assert!((p[0] - 2.0).abs() < 1e-12 && (p[1] - 4.0).abs() < 1e-12);
        /// ```
        #[inline]
        pub fn translation_twist(vx: T, vy: T, vz: T) -> Self {
            let h = T::from_f64(-0.5);
            let z = T::zero();
            Line::new(z, z, z, vx * h, vy * h, vz * h)
        }

        /// The twist of a rotation at `omega` radians per unit time about the line `axis`
        /// (right-handed, as [`Motor::rotation`]): `(B * t).exp()` is
        /// `Motor::rotation(axis, t omega)`.
        ///
        /// ```
        /// use gax::pga3d::{Line, Motor, Point};
        /// let z_axis = Point::<(), f64>::xyz(0.0, 0.0, 0.0) & Point::xyz(0.0, 0.0, 1.0);
        /// let spin = Line::rotation_twist(z_axis, 0.5); // half a radian per second
        /// let after = spin.gp(3.0).exp(); // after three seconds
        /// let direct = Motor::rotation(z_axis, 1.5);
        /// assert!(after.c.iter().zip(direct.c).all(|(a, b)| (a - b).abs() < 1e-12));
        /// ```
        #[inline]
        pub fn rotation_twist(axis: Line<(), T>, omega: T) -> Self {
            crate::extras::shared::rotation_twist(axis, omega)
        }
    }
}

#[cfg(feature = "pga2d")]
mod pga2d_extras {
    use crate::pga2d::{Line, Motor, Point, Translator};
    use crate::{Real, Unit};

    impl<T: Real> Point<(), T> {
        /// The Euclidean point `(x, y)`, with weight 1: `x e20 + y e01 + e12`.
        ///
        /// ```
        /// use gax::pga2d::Point;
        /// let p = Point::<(), f64>::xy(3.0, 4.0);
        /// assert_eq!([p.e20(), p.e01(), p.e12()], [3.0, 4.0, 1.0]);
        /// ```
        #[inline]
        pub fn xy(x: T, y: T) -> Self {
            Point::new(x, y, T::one())
        }

        /// The direction `(x, y)`: a point at infinity.
        ///
        /// ```
        /// use gax::pga2d::Point;
        /// // The difference of two points is a direction, a point at infinity.
        /// assert_eq!(Point::<(), f64>::xy(3.0, 4.0) - Point::xy(1.0, 1.0), Point::direction(2.0, 3.0));
        /// ```
        #[inline]
        pub fn direction(x: T, y: T) -> Self {
            Point::new(x, y, T::zero())
        }

        /// Euclidean coordinates `(x/w, y/w)`.
        ///
        /// ```
        /// use gax::pga2d::Point;
        /// assert_eq!(Point::<(), f64>::new(6.0, -2.0, 2.0).to_euclidean(), [3.0, -1.0]);
        /// ```
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
        ///
        /// ```
        /// use gax::pga2d::{Motor, Point};
        /// let c = Point::<(), f64>::xy(1.0, 1.0);
        /// let after = Point::rotation_twist(c, 0.25).gp(2.0).exp(); // two seconds at 1/4 rad/s
        /// let direct = Motor::rotation(c, 0.5);
        /// assert!(after.c.iter().zip(direct.c).all(|(a, b)| (a - b).abs() < 1e-12));
        /// ```
        #[inline]
        pub fn rotation_twist(center: Point<(), T>, omega: T) -> Self {
            crate::extras::shared::rotation_twist(center, omega)
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
        /// The motor that carries `a` onto `b`, lines or points: the square root of their
        /// ratio, `sqrt(b / a)`, as in PGA3D (see `gax::pga3d::Motor::between`). Two lines
        /// give the rotation about their meet (parallel lines, a translation); two points, the
        /// translation between them.
        ///
        /// ```
        /// use gax::pga2d::{Motor, Point};
        /// let a = Point::<(), f64>::xy(0.0, 0.0) & Point::xy(1.0, 0.0);
        /// let b = Point::xy(2.0, 0.0) & Point::xy(2.0, 1.0);
        /// let m = Motor::between(a, b); // a quarter turn about (2, 0)
        /// assert!((m.angle() - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
        /// ```
        #[inline]
        pub fn between<X: crate::extras::Between<Self>>(a: X, b: X) -> Unit<Self> {
            X::between(a, b)
        }

        /// The translation by `(dx, dy)`.
        ///
        /// ```
        /// use gax::pga2d::{Motor, Point};
        /// let t = Motor::<(), f64>::translation(3.0, -1.0);
        /// assert_eq!((t >> Point::xy(1.0, 1.0)).to_euclidean(), [4.0, 0.0]);
        /// ```
        #[inline]
        pub fn translation(dx: T, dy: T) -> Unit<Self> {
            let h = T::from_f64(0.5);
            // A translator, as a motor.
            let t = Translator::new(T::one(), dy * h, -dx * h);
            Unit::new_unchecked(t.cast::<Motor>())
        }

        /// The rotation by `angle` (counterclockwise) about the point `center`.
        ///
        /// ```
        /// use gax::pga2d::{Motor, Point};
        /// // A quarter turn counterclockwise about (1, 0) takes (2, 0) to (1, 1).
        /// let r = Motor::<(), f64>::rotation(Point::xy(1.0, 0.0), std::f64::consts::FRAC_PI_2);
        /// let [x, y] = (r >> Point::xy(2.0, 0.0)).to_euclidean();
        /// assert!((x - 1.0).abs() < 1e-12 && (y - 1.0).abs() < 1e-12);
        /// ```
        #[inline]
        pub fn rotation(center: Point<(), T>, angle: T) -> Unit<Self> {
            crate::extras::shared::rotation_twist(center, angle).exp()
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

        /// The rigid motion from `a` to `b`, at `t`: `a exp(t log(~a b))`, the shorter way (the
        /// relative motor taken with its scalar part non-negative; see the 3D
        /// `Motor::interpolate`).
        ///
        /// ```
        /// use gax::pga2d::{Motor, Point};
        /// let (a, b) = (Motor::<(), f64>::translation(0.0, 0.0), Motor::translation(4.0, 2.0));
        /// let halfway = Motor::interpolate(a, b, 0.5);
        /// let [x, y] = (halfway >> Point::xy(0.0, 0.0)).to_euclidean();
        /// assert!((x - 2.0).abs() < 1e-12 && (y - 1.0).abs() < 1e-12);
        /// ```
        pub fn interpolate(a: Unit<Self>, b: Unit<Self>, t: T) -> Unit<Self> {
            crate::extras::shared::interpolate::<Self, Point<(), T>, T>(a, b, t)
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

#[cfg(any(feature = "cga2d", feature = "cga3d"))]
/// Conformal points: the up map from Euclidean space (quadratic, so a function rather than a
/// homomorphism) and back, spheres, and the PGA point of a round point.
macro_rules! cga_points {
    ($alg:ident, $pga:ident, $pga_feature:literal, [$($x:ident),+], $n:literal, $up_doc:literal) => {
        impl<T: crate::Real> crate::$alg::Vector<(), T> {
            #[doc = $up_doc]
            #[inline]
            pub fn up($($x: T),+) -> Self {
                let half = T::from_f64(0.5);
                let sq = T::zero() $(+ $x * $x)+;
                Self::new($($x,)+ T::one(), half * sq)
            }

            /// The Euclidean coordinates of a round point (of any weight): its vector part
            /// divided by its `eo` coefficient. Not finite for a point at infinity or a plane.
            #[inline]
            pub fn down(self) -> [T; $n] {
                let r = self.c[$n].recip();
                core::array::from_fn(|i| self.c[i] * r)
            }

            /// The sphere (or circle) with this centre and radius: `up(c) - ½r² ei`, in the
            /// dual representation (a point `X` lies on it when `X · s = 0`).
            #[inline]
            pub fn sphere(centre: [T; $n], radius: T) -> Self {
                let mut s = Self::from_coeffs(core::array::from_fn(|i| {
                    if i < $n { centre[i] } else if i == $n { T::one() } else { T::zero() }
                }));
                let sq = centre.iter().fold(T::zero(), |a, &x| a + x * x);
                s.c[$n + 1] = T::from_f64(0.5) * (sq - radius * radius);
                s
            }
        }

        #[cfg(feature = $pga_feature)]
        impl<T: crate::Real> crate::$alg::Vector<(), T> {
            /// The round point of a PGA point (of weight `w`): `w up(p / w)`.
            #[inline]
            pub fn from_point(p: crate::$pga::Point<(), T>) -> Self {
                let [$($x),+] = p.to_euclidean();
                Self::up($($x),+).gp(p.c[$n])
            }

            /// The PGA point of a round point: its vector part with its `eo` coefficient as the
            /// weight. Linear (it drops `ei`), so it also maps round points' maps.
            #[inline]
            pub fn to_point(self) -> crate::$pga::Point<(), T> {
                crate::$pga::Point::from_coeffs(core::array::from_fn(|i| self.c[i]))
            }
        }
    };
}
#[cfg(feature = "cga3d")]
cga_points!(
    cga3d,
    pga3d,
    "pga3d",
    [x, y, z],
    3,
    "The round point `eo + x + ½|x|² ei` of the Euclidean point `(x, y, z)`: a null vector, and `X · Y = -½|x - y|²` for two of them."
);
#[cfg(feature = "cga2d")]
cga_points!(
    cga2d,
    pga2d,
    "pga2d",
    [x, y],
    2,
    "The round point `eo + x + ½|x|² ei` of the Euclidean point `(x, y)`: a null vector, and `X · Y = -½|x - y|²` for two of them."
);

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
    /// Sweeping a camera round through looking backwards: `look_at` flips the sign of its
    /// motor on the way (the same motion), and a spring following it with `interpolate` still
    /// moves by the small true step every frame, never the long way round.
    #[cfg(feature = "pga3d")]
    #[test]
    fn interpolation_takes_the_shorter_way_through_a_sign_flip() {
        use crate::pga3d::{Motor, Point};
        let up = Point::direction(0.0, 1.0, 0.0);
        let eye = Point::xyz(0.0, 0.0, 0.0);
        let at = |k: f64| {
            let a = k * 0.01;
            Motor::look_at(eye, Point::xyz(a.sin(), 0.05, a.cos()), up)
        };
        let mut flips = 0;
        let mut cam = at(0.0);
        for k in 1..700 {
            let (prev, target) = (at(f64::from(k - 1)), at(f64::from(k)));
            flips += usize::from((prev.reverse() * target).s() < 0.0);
            let next = Motor::interpolate(cam, target, 0.5);
            // The camera's forward axis moves by at most the target's step (0.01 rad).
            let (f0, f1) = (
                cam >> Point::direction(0.0, 0.0, 1.0),
                next >> Point::direction(0.0, 0.0, 1.0),
            );
            let dot = f0.e032() * f1.e032() + f0.e013() * f1.e013() + f0.e021() * f1.e021();
            assert!(dot > 0.9999, "step {k}: the camera swung ({dot})");
            cam = next;
        }
        assert!(flips > 0, "the sweep never crossed a sign flip");
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

    /// Round points are null, come back down, and measure distance; the homomorphism from PGA
    /// keeps incidence: a point on a PGA plane lies on its CGA image, with `X · π` the plane's
    /// signed distance.
    #[cfg(all(feature = "cga3d", feature = "pga3d"))]
    #[test]
    #[allow(clippy::float_cmp)] // small integers, exact in f64
    fn cga_round_points_and_pga_planes() {
        use crate::{cga3d, pga3d};
        let x = cga3d::Vector::<(), f64>::up(1.0, 2.0, 3.0);
        assert!((x | x).s().abs() < 1e-12);
        assert_eq!(x.down(), [1.0, 2.0, 3.0]);
        let y = cga3d::Vector::up(4.0, 6.0, 3.0);
        assert!(((x | y).s() + 12.5).abs() < 1e-12); // -½ |(3, 4, 0)|²
        let s = cga3d::Vector::sphere([1.0, 2.0, 3.0], 5.0);
        assert!((y | s).s().abs() < 1e-12); // (4, 6, 3) is 5 from the centre
        let p = pga3d::Point::xyz(1.0, 2.0, 3.0);
        assert_eq!(cga3d::Vector::from_point(p).down(), [1.0, 2.0, 3.0]);
        assert_eq!(cga3d::Vector::from_point(p).to_point(), p);
        let plane = pga3d::Plane::from_normal([0.0, 0.0, 1.0], 1.0); // z = 1
        let dual: cga3d::Vector<(), f64> = plane.into();
        assert!(((x | dual).s() - (plane & p).s()).abs() < 1e-12);
        assert!(((x | dual).s() - 2.0).abs() < 1e-12);
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
