//! Hand-written conveniences for the standard algebras: constructors with geometric meaning,
//! and the geometry built on them.
//!
//! These are inherent functions on the generated types. Constructors such as
//! `Point::xyz(x, y, z)` and `Motor::translation(d)` fix the sign conventions in one place;
//! coordinates enter only through the plain ones (`xyz`, `direction`, `new`, CGA's `up`), and
//! the rest take gax values. On top of them come motors between flats ([`Between`]),
//! `look_at`, `interpolate`, `angle`, reflections, principal frames of inertia, and conformal
//! points and spheres.

#[cfg(feature = "pga3d")]
pub use pga3d_extras::{PrincipalInertia, PrincipalMobility};

#[cfg(any(
    feature = "pga2d",
    feature = "pga3d",
    feature = "vga2d",
    feature = "vga3d"
))]
/// The elements a motor (or a rotor) carries onto one another, for `Motor::between` and
/// `Rotor::between`: planes, lines and points in PGA3D, lines and points in PGA2D, vectors and
/// bivectors in VGA3D, vectors in VGA2D.
///
/// ```
/// use gax::ApproxEq;
/// use gax::pga3d::{Motor, Plane, Point};
/// let a = Plane::<(), f64>::with_normal(Point::direction(1.0, 0.0, 0.0), 0.0);
/// let b = Plane::with_normal(Point::direction(0.0, 1.0, 0.0), 2.0);
/// let m = Motor::between(a, b);
/// let moved = (m >> a).normalized();
/// assert!(moved.approx_eq(&b.normalized(), 1e-12));
/// ```
pub trait Between<M>: Sized {
    /// The motor that carries `a` onto `b`.
    fn between(a: Self, b: Self) -> crate::Unit<M>;
}

#[cfg(any(
    feature = "pga2d",
    feature = "pga3d",
    feature = "vga2d",
    feature = "vga3d"
))]
/// `exp(log(b / a) / 2)`, the square root of the ratio: the motion that carries `a` onto `b`.
/// The ratio is normalized first, so `a` and `b` need not be. A kind listed with a method
/// passes both through it first: points are unitized, as no motor carries a point onto its
/// negative (motors keep the sign of the weight).
macro_rules! between {
    ($alg:ident: $($k:ident $(.$prep:ident())?),* => $versor:ident, $biv:ident) => {$(
        // Of either orientation (ADR-051): the ratio of two elements of the same one is
        // extrinsic, a motion.
        impl<T: crate::Real, O: crate::Orientation> Between<crate::$alg::$versor<(), T>> for crate::$alg::oriented::$k<(), T, O> {
            #[inline]
            fn between(a: Self, b: Self) -> crate::Unit<crate::$alg::$versor<(), T>> {
                // `O · O` is extrinsic for either orientation (ADR-052): the ratio is a motion.
                let ratio = b$(.$prep())? / a$(.$prep())?;
                let half: crate::$alg::$biv<(), T> = ratio.unit().log();
                half.gp(T::from_f64(0.5)).exp()
            }
        }
    )*};
}
#[cfg(feature = "pga3d")]
between!(pga3d: Plane, Line, Point.unitized() => Motor, Line);
#[cfg(feature = "pga2d")]
between!(pga2d: Line, Point.unitized() => Motor, Twist);
#[cfg(feature = "vga3d")]
between!(vga3d: Vector, Bivector => Rotor, Bivector);
#[cfg(feature = "vga2d")]
between!(vga2d: Vector => Rotor, Bivector);

/// Written once for PGA2D, PGA3D and VGA3D through the value traits: their kinds differ (a
/// rotation is about a point in 2D, about a line in 3D, in a plane in VGA), the expressions do
/// not.
#[cfg(any(
    feature = "pga2d",
    feature = "pga3d",
    feature = "vga2d",
    feature = "vga3d"
))]
mod shared {
    use crate::{Exp, Extensor, Gp, GradePart, Log, Real, Reverse, Unit};

    /// `−(rate/2) x̂`: the twist of a rotation at `rate` about the flat `x` (a line in PGA3D, a
    /// point in PGA2D, a plane through the origin in VGA3D), normalized first.
    #[cfg(any(feature = "pga2d", feature = "pga3d", feature = "vga3d"))]
    #[inline]
    pub fn rotation_twist<X: crate::Normalize + Gp<T, Output = X>, T: Real>(x: X, rate: T) -> X {
        x.normalized().gp(rate * T::from_f64(-0.5))
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
        M: Log<Output = B>,
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
    use crate::map::{Dense, LinearMap, SquareLinearMap};
    use crate::pga3d::{Forque, Line, Motor, Plane, Point, Translator, oriented};
    use crate::{Of, Real, Slots, Unit};

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
    }

    // Methods of points of either orientation (ADR-051): positions are intrinsic, meets of
    // three planes extrinsic.
    impl<T: Real, O: crate::Orientation> oriented::Point<(), T, O> {
        /// The angle between this direction and `other` (points at infinity), in `[0, π]`: with
        /// `A` and `B` the planes through the origin orthogonal to them, `atan2` of the norm of
        /// their meet (a line through the origin, its norm the sine) and their inner product (the
        /// cosine), scaled alike, so the directions need no normalizing and the angle is well
        /// conditioned at every angle. (Directions have no signed angle in space: no orientation
        /// of a plane is singled out.)
        ///
        /// ```
        /// use gax::pga3d::Point;
        /// let (a, b) = (Point::<(), f64>::direction(1.0, 0.0, 0.0), Point::direction(0.0, 0.0, -2.0));
        /// assert!((a.angle_between(b) - std::f64::consts::FRAC_PI_2).abs() < 1e-15);
        /// ```
        pub fn angle_between(self, other: Self) -> T {
            let zero = T::zero();
            let a = Plane::new(self.e032(), self.e013(), self.e021(), zero);
            let b = Plane::new(other.e032(), other.e013(), other.e021(), zero);
            (a ^ b).norm().atan2((a | b).s())
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

        /// The position of this point: the intrinsic point of weight `+1` at the same place
        /// (ADR-051). A meet of planes, or of a joined line with a plane, is a point of either
        /// orientation, its sign a handedness; as a place to be it is this.
        ///
        /// ```
        /// use gax::pga3d::{Plane, Point};
        /// let (x, y, z) = (Plane::new(1.0, 0.0, 0.0, -1.0), Plane::new(0.0, 1.0, 0.0, -2.0), Plane::new(0.0, 0.0, 1.0, -3.0));
        /// let corner = x ^ y ^ z; // extrinsic
        /// assert_eq!(corner.position(), Point::<(), f64>::xyz(1.0, 2.0, 3.0));
        /// ```
        #[inline]
        pub fn position(self) -> Point<(), T> {
            self.unitized().with_orientation()
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
            self.ideal_norm_squared().sqrt()
        }

        /// The squared ideal norm `x² + y² + z²`: a squared length, without the square root.
        ///
        /// ```
        /// use gax::pga3d::Point;
        /// assert_eq!(Point::direction(2.0, 3.0, 6.0).ideal_norm_squared(), 49.0);
        /// ```
        #[inline]
        pub fn ideal_norm_squared(self) -> T {
            let (x, y, z) = (self.e032(), self.e013(), self.e021());
            x * x + y * y + z * z
        }

        /// The distance between this point and `q`, both finite (of any non-zero weight, either
        /// orientation): the ideal norm of the difference of their positions.
        ///
        /// ```
        /// use gax::pga3d::Point;
        /// let (p, q) = (Point::<(), f64>::xyz(1.0, 2.0, 3.0), Point::new(6.0, 6.0, 10.0, 2.0));
        /// assert_eq!(p.distance(q), 3.0); // q is (3, 3, 5)
        /// ```
        #[inline]
        pub fn distance<Q: crate::Orientation>(self, q: oriented::Point<(), T, Q>) -> T {
            (q.position() - self.position()).ideal_norm()
        }
    }

    impl<T: Real> Plane<(), T> {
        /// The plane `n · x = d` with normal `n`, a direction of either orientation (not
        /// necessarily of unit length; its weight is ignored). The inverse of
        /// [`normal`](oriented::Plane::normal).
        ///
        /// ```
        /// use gax::pga3d::{Plane, Point};
        /// let floor = Plane::<(), f64>::with_normal(Point::direction(0.0, 0.0, 1.0), 2.0); // z = 2
        /// // The join with a point is its signed distance, positive along the normal.
        /// assert_eq!((floor & Point::xyz(1.0, 1.0, 5.0)).s(), 3.0);
        /// assert_eq!(floor.normal(), Point::direction(0.0, 0.0, 1.0));
        /// ```
        #[inline]
        pub fn with_normal<O: crate::Orientation>(n: oriented::Point<(), T, O>, d: T) -> Self {
            Plane::new(n.e032(), n.e013(), n.e021(), -d)
        }

        /// The plane through the origin perpendicular to the direction `d` (of either
        /// orientation), with normal `d`.
        ///
        /// ```
        /// use gax::pga3d::{Plane, Point};
        /// let p = Plane::<(), f64>::orthogonal_to(Point::direction(0.0, 0.0, 2.0));
        /// // Through the origin, normal (0, 0, 2): the distance of a point, scaled by 2.
        /// assert_eq!((p & Point::xyz(3.0, 4.0, 1.0)).s(), 2.0);
        /// ```
        #[inline]
        pub fn orthogonal_to<O: crate::Orientation>(d: oriented::Point<(), T, O>) -> Self {
            Self::with_normal(d, T::zero())
        }
    }

    impl<T: Real, M: crate::Orientation> oriented::Plane<(), T, M> {
        /// The normal `(a, b, c)` of the plane `a x + b y + c z + d = 0`, as a direction: it
        /// points to the side where `plane & x` is positive, and its length is the plane's
        /// norm. For a join of three points `p & q & r` it is the normal of their winding,
        /// `(q − p) × (r − p)` for points of weight one, so that the points turn
        /// counterclockwise seen from where it points.
        ///
        /// ```
        /// use gax::pga3d::{Plane, Point};
        /// let tilted = Plane::<(), f64>::new(1.0, 2.0, 2.0, -3.0);
        /// assert_eq!(tilted.normal(), Point::direction(1.0, 2.0, 2.0));
        /// let (p, q, r) = (Point::xyz(1.0, 0.0, 0.0), Point::xyz(2.0, 0.0, 0.0), Point::xyz(1.0, 1.0, 0.0));
        /// assert_eq!((p & q & r).normal(), Point::direction(0.0, 0.0, 1.0));
        /// ```
        #[inline]
        pub fn normal(self) -> Point<(), T> {
            Point::direction(self.e1(), self.e2(), self.e3())
        }

        /// Reflect a point (or a direction) in this plane: `>>` with the normalized plane, whose
        /// sign the point's orientation picks (ADR-051). The plane's own orientation, a meet's
        /// or a join's, does not matter: it enters twice.
        ///
        /// ```
        /// use gax::pga3d::{Plane, Point};
        /// let wall = Plane::with_normal(Point::direction(1.0, 0.0, 0.0), 0.0);
        /// let v = wall.reflect(Point::direction(1.0, 0.5, 0.0));
        /// assert_eq!([v.e032(), v.e013(), v.e021()], [-1.0, 0.5, 0.0]);
        /// // The same wall as the join of three of its points.
        /// let joined = Point::xyz(0.0, 0.0, 0.0) & Point::xyz(0.0, 1.0, 0.0) & Point::xyz(0.0, 0.0, 1.0);
        /// assert_eq!(joined.reflect(Point::direction(1.0, 0.5, 0.0)), v);
        /// ```
        #[inline]
        pub fn reflect<O: crate::Orientation>(
            self,
            x: oriented::Point<(), T, O>,
        ) -> oriented::Point<(), T, O> {
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
    /// available through [`PrincipalInertia::to_dense`], for example moving the body with
    /// `m >> inertia.to_dense().of(m << Line::slot())`.
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
        ///
        /// A map with open slots is composed: `body.of(m)` is `body ∘ m`, scaled column by
        /// column, without building the dense map (ADR-044). A twist is extrinsic (a turn about
        /// a line), a momentum intrinsic (a line oriented along itself, ADR-051).
        #[inline]
        pub fn of<S: Slots>(self, b: Line<S, T>) -> Forque<S, T> {
            let (m, i) = (self.mass, self.moments);
            Forque::from_coeffs(swap_scaled::<S, T>(&b.c, [m, m, m, i[0], i[1], i[2]]))
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
        pub fn inverse_of<S: Slots>(self, f: Forque<S, T>) -> Line<S, T> {
            self.inverse().of(f)
        }

        /// The inverse map, from momenta to twists, in the same four numbers: the mass and
        /// moments inverted (ADR-044).
        ///
        /// ```
        /// use gax::pga3d::{Line, PrincipalInertia};
        /// let body = PrincipalInertia::new(4.0, [0.5, 0.25, 2.0]);
        /// let b = Line::new(1.0, -0.5, 0.25, 2.0, 0.5, -1.0);
        /// assert_eq!(body.inverse().of(body.of(b)), b);
        /// ```
        #[inline]
        pub fn inverse(self) -> PrincipalMobility<T> {
            PrincipalMobility {
                inverse_mass: self.mass.recip(),
                inverse_moments: self.moments.map(Real::recip),
            }
        }

        /// The dense map `Forque <- Line`, from twists to momenta.
        ///
        /// ```
        /// use gax::pga3d::{Line, PrincipalInertia};
        /// let body = PrincipalInertia::new(2.0, [0.5, 0.75, 1.0]);
        /// let b = Line::new(0.1, 0.2, 0.3, 1.0, -1.0, 0.5);
        /// assert_eq!(body.to_dense().of(b), body.of(b));
        /// ```
        #[inline]
        pub fn to_dense(self) -> Forque<(Line,), T> {
            self.of(Line::slot())
        }

        /// The kinetic energy form `B & I[B]` on twists (twice the kinetic energy): a twist joined
        /// with a momentum, an extrinsic scalar, unchanged by mirrors.
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
            Line::slot() & self.to_dense()
        }
    }

    /// The coefficients of a line with its two halves swapped (the Euclidean part to the ideal
    /// one and back) and row `r` scaled by `k[r]`: the action of the principal maps.
    #[inline(always)]
    fn swap_scaled<S: Slots, T: Real>(c: &[S::Arr<T>; 6], k: [T; 6]) -> [S::Arr<T>; 6] {
        core::array::from_fn(|r| S::map(&c[(r + 3) % 6], &mut |x: &T| k[r] * *x))
    }

    /// The inverse of a [`PrincipalInertia`]: a map from momenta (forques) to twists, stored as
    /// the inverse mass and the inverse principal moments (ADR-044).
    ///
    /// ```
    /// use gax::pga3d::{Line, PrincipalInertia};
    /// let mobility = PrincipalInertia::new(2.0, [0.5, 0.75, 0.25]).inverse();
    /// let f = Line::new(2.0, 0.0, 0.0, 0.0, 0.0, 1.0).with_orientation::<gax::Intrinsic>();
    /// let b = mobility.of(f);
    /// // The angular part of the momentum through the mass, the linear part through the moments.
    /// assert_eq!((b.e01(), b.e12()), (1.0, 4.0));
    /// ```
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct PrincipalMobility<T> {
        /// The reciprocal of the mass.
        pub inverse_mass: T,
        /// The reciprocals of the principal moments about x, y and z.
        pub inverse_moments: [T; 3],
    }

    impl<T: Real> PrincipalMobility<T> {
        /// The twist of a momentum: six products. A map is composed, as for
        /// [`PrincipalInertia::of`].
        #[inline]
        pub fn of<S: Slots>(self, f: Forque<S, T>) -> Line<S, T> {
            let (r, i) = (self.inverse_mass, self.inverse_moments);
            Line::from_coeffs(swap_scaled::<S, T>(&f.c, [i[0], i[1], i[2], r, r, r]))
        }

        /// The inertia this is the inverse of.
        #[inline]
        pub fn inverse(self) -> PrincipalInertia<T> {
            PrincipalInertia {
                mass: self.inverse_mass.recip(),
                moments: self.inverse_moments.map(Real::recip),
            }
        }

        /// The dense map `Line <- Forque`, from momenta to twists.
        #[inline]
        pub fn to_dense(self) -> Line<(Forque,), T> {
            self.of(Forque::slot())
        }
    }

    // The structured maps as maps (ADR-044): their dense forms for everything else, their own
    // paths for applying, solving and inverting, and for composing as the outer map.

    impl<T: Real> LinearMap for PrincipalInertia<T> {
        type Kind = Forque;
        type Slots = (Line,);
        type Coef = T;
        #[inline]
        fn to_dense(self) -> Dense<Self> {
            Self::to_dense(self)
        }
    }

    impl<T: Real> LinearMap for PrincipalMobility<T> {
        type Kind = Line;
        type Slots = (Forque,);
        type Coef = T;
        #[inline]
        fn to_dense(self) -> Dense<Self> {
            Self::to_dense(self)
        }
    }

    impl<T: Real> SquareLinearMap for PrincipalInertia<T> {
        type Input = Line;
        type Inverse = PrincipalMobility<T>;
        #[inline]
        fn of(self, x: Line<(), T>) -> Forque<(), T> {
            self.of(x)
        }
        #[inline]
        fn solve(self, b: Forque<(), T>) -> Line<(), T> {
            self.inverse_of(b)
        }
        #[inline]
        fn inverse(self) -> PrincipalMobility<T> {
            PrincipalInertia::inverse(self)
        }
    }

    impl<T: Real> SquareLinearMap for PrincipalMobility<T> {
        type Input = Forque;
        type Inverse = PrincipalInertia<T>;
        #[inline]
        fn of(self, x: Forque<(), T>) -> Line<(), T> {
            self.of(x)
        }
        #[inline]
        fn solve(self, b: Line<(), T>) -> Forque<(), T> {
            self.inverse().of(b)
        }
        #[inline]
        fn inverse(self) -> PrincipalInertia<T> {
            PrincipalMobility::inverse(self)
        }
    }

    impl<S: Slots, T: Real> Of<Line<S, T>> for PrincipalInertia<T> {
        type Output = Forque<S, T>;
        #[inline]
        fn of(self, b: Line<S, T>) -> Forque<S, T> {
            PrincipalInertia::of(self, b)
        }
    }

    impl<S: Slots, T: Real> Of<Forque<S, T>> for PrincipalMobility<T> {
        type Output = Line<S, T>;
        #[inline]
        fn of(self, f: Forque<S, T>) -> Line<S, T> {
            PrincipalMobility::of(self, f)
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
        /// use gax::ApproxEq;
        /// let (p, q) = (Point::<(), f64>::xyz(1.0, 2.0, 3.0), Point::xyz(0.0, 5.0, 3.0));
        /// let (r, s) = (Point::xyz(1.0, 2.0, 4.0), Point::xyz(0.0, 6.0, 3.0));
        /// // The motor that lays the line p r onto the line q s.
        /// let m = Motor::between(p & r, q & s);
        /// let l = (m >> (p & r)).normalized();
        /// assert!(l.approx_eq(&(q & s).normalized(), 1e-12));
        /// assert!((m >> p).to_euclidean()[0].abs() < 1e-12); // p lands on q s
        /// ```
        #[inline]
        pub fn between<X: crate::extras::Between<Self>>(a: X, b: X) -> Unit<Self> {
            X::between(a, b)
        }

        /// The translation by the direction `d` (a point at infinity, of either orientation:
        /// `Point::direction(x, y, z)`, or the difference `q - p` of two points of weight one,
        /// the translation from `p` to `q`). Only its ideal part is read.
        ///
        /// ```
        /// use gax::pga3d::{Motor, Point};
        /// let t = Motor::translation(Point::direction(1.0, 2.0, 3.0));
        /// let p = t >> Point::xyz(0.0, 0.0, 0.0);
        /// assert_eq!(p.to_euclidean(), [1.0, 2.0, 3.0]);
        /// let (a, b) = (Point::xyz(1.0, 1.0, 1.0), Point::xyz(2.0, 3.0, 4.0));
        /// assert_eq!(Motor::translation(b - a) >> a, b);
        /// ```
        #[inline]
        pub fn translation<O: crate::Orientation>(d: oriented::Point<(), T, O>) -> Unit<Self> {
            let h = T::from_f64(-0.5);
            // A translator `1 − (d/2) e0`, as a motor.
            let t = Translator::new(T::one(), d.e032() * h, d.e013() * h, d.e021() * h);
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
        /// assert!(p[0].abs() < 1e-12 && (p[1] - 1.0).abs() < 1e-12);
        /// ```
        ///
        /// The axis may have either orientation (ADR-051): a turn about it (a meet of planes,
        /// extrinsic) or an arrow along it (a join of points, intrinsic), which turns by the
        /// right-hand rule; the coefficients are the same.
        #[inline]
        pub fn rotation<O: crate::Orientation>(
            axis: oriented::Line<(), T, O>,
            angle: T,
        ) -> Unit<Self> {
            crate::extras::shared::rotation_twist(
                axis.with_orientation::<crate::Extrinsic>(),
                angle,
            )
            .exp()
        }

        /// The rotation by `angle` about the axis through the origin along the direction `d` (a
        /// point at infinity of either orientation, of any length; its weight is ignored),
        /// right-handed about `d`: [`Motor::rotation`] about the join of the origin and `d`.
        ///
        /// ```
        /// use gax::pga3d::{Motor, Point};
        /// let up = Point::<(), f64>::direction(0.0, 0.0, 1.0);
        /// let r = Motor::rotation_about(up, std::f64::consts::FRAC_PI_2);
        /// let [x, y, z] = (r >> Point::xyz(1.0, 0.0, 0.0)).to_euclidean();
        /// assert!(x.abs() < 1e-12 && (y - 1.0).abs() < 1e-12 && z.abs() < 1e-12);
        /// ```
        #[inline]
        pub fn rotation_about<O: crate::Orientation>(
            d: oriented::Point<(), T, O>,
            angle: T,
        ) -> Unit<Self> {
            let o = T::zero();
            let axis = Point::xyz(o, o, o) & Point::direction(d.e032(), d.e013(), d.e021());
            Self::rotation(axis, angle)
        }

        /// The shortest rotation (about an axis through the origin) that turns the direction
        /// `from` into the direction `to`. With `A` and `B` the planes through the origin
        /// orthogonal to them, the axis is their meet `A ^ B` and the angle is
        /// `atan2(|A ^ B|, A | B)`: the meet's weight and the inner product are the sine and
        /// cosine, scaled alike, so no normalization is needed and the angle is well conditioned
        /// everywhere, a half turn included (a formula like `normalize(1 + B A)` loses the small
        /// scalar part to cancellation there). Opposite directions get a half turn about an
        /// axis perpendicular to both. The directions may have either orientation.
        ///
        /// ```
        /// use gax::pga3d::{Motor, Point};
        /// let z = Point::<(), f64>::direction(0.0, 0.0, 1.0);
        /// let r = Motor::rotation_between(z, Point::direction(0.0, 3.0, 0.0));
        /// let d = r >> Point::direction(0.0, 0.0, 1.0);
        /// assert!((d.e013() - 1.0).abs() < 1e-12 && d.e021().abs() < 1e-12);
        /// ```
        pub fn rotation_between<A: crate::Orientation, B: crate::Orientation>(
            from: oriented::Point<(), T, A>,
            to: oriented::Point<(), T, B>,
        ) -> Unit<Self> {
            let (o, zero) = (T::one(), T::zero());
            let a = Plane::orthogonal_to(from);
            let b = Plane::orthogonal_to(to);
            let meet = a ^ b;
            let angle = meet.norm().atan2((a | b).s());
            // Nearly parallel or opposite, the meet vanishes: turn about an axis orthogonal to
            // `A` instead (the meet of `A` with a plane orthogonal to it, `e` less its part
            // along `A`). The angle is right either way; branch-free, so that it traces.
            let an = a.normalized();
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
        /// let z = Point::<(), f64>::direction(0.0, 0.0, 1.0);
        /// let (a, b) = (Motor::rotation_about(z, 0.1), Motor::rotation_about(z, 0.3));
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
        /// `up` is a direction of either orientation.
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
        pub fn look_at<U: crate::Orientation>(
            eye: Point<(), T>,
            target: Point<(), T>,
            up: oriented::Point<(), T, U>,
        ) -> Unit<Self> {
            let (o, zero) = (T::one(), T::zero());
            let [ex, ey, ez] = eye.to_euclidean();
            let [tx, ty, tz] = target.to_euclidean();
            // The frame: z forward, x = up × z (normalized), y = z × x, then its rotor
            // (Shepperd's method, well conditioned for every rotation). Built from the frame
            // rather than as a turn and a roll: a shortest rotation between nearly opposite
            // directions takes its axis from rounding, and looking nearly along `up` (from above,
            // `+z` up) once turned an `f32` camera round.
            let norm2 = |v: [T; 3]| v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
            let unit = |v: [T; 3]| {
                let n = norm2(v).sqrt();
                v.map(|c| c / n)
            };
            let cross = crate::moments::cross3;
            let z = unit([tx - ex, ty - ey, tz - ez]);
            let u = [up.e032(), up.e013(), up.e021()];
            let x = cross(u, z);
            // `up` along `forward`: no roll is defined; any x orthogonal to z (from the axis z is
            // least along).
            let other = T::select_lt(z[0].abs(), T::from_f64(0.5), o, zero);
            let fallback = cross([other, o - other, zero], z);
            let tiny = T::epsilon() * T::epsilon() * norm2(u);
            let x = unit(core::array::from_fn(|i| {
                T::select_lt(norm2(x), tiny, fallback[i], x[i])
            }));
            let y = cross(z, x);
            Self::translation(Point::direction(ex, ey, ez))
                * crate::moments::rotor_from_frame([x, y, z])
        }
    }

    // Twists: lines, extrinsic (a turn about a line; ADR-051), named `Twist`.
    impl<T: Real> Line<(), T> {
        /// The twist (a bivector: a line) of a translation at the velocity `v`, a direction of
        /// either orientation: `(B * t).exp()` is `Motor::translation(v * t)`. Twists add.
        ///
        /// ```
        /// use gax::pga3d::{Motor, Point, Twist};
        /// let twist = Twist::<(), f64>::translation(Point::direction(1.0, 2.0, 0.0));
        /// // Two seconds at that velocity.
        /// let p = (twist.gp(2.0).exp() >> Point::xyz(0.0, 0.0, 0.0)).to_euclidean();
        /// assert!((p[0] - 2.0).abs() < 1e-12 && (p[1] - 4.0).abs() < 1e-12);
        /// ```
        #[inline]
        pub fn translation<O: crate::Orientation>(v: oriented::Point<(), T, O>) -> Self {
            let h = T::from_f64(-0.5);
            let z = T::zero();
            Line::new(z, z, z, v.e032() * h, v.e013() * h, v.e021() * h)
        }

        /// The twist of a rotation at `omega` radians per unit time about the line `axis`
        /// (right-handed, as [`Motor::rotation`]): `(B * t).exp()` is
        /// `Motor::rotation(axis, t omega)`.
        ///
        /// ```
        /// use gax::pga3d::{Motor, Point, Twist};
        /// let z_axis = Point::<(), f64>::xyz(0.0, 0.0, 0.0) & Point::xyz(0.0, 0.0, 1.0);
        /// let spin = Twist::rotation(z_axis, 0.5); // half a radian per second
        /// let after = spin.gp(3.0).exp(); // after three seconds
        /// let direct = Motor::rotation(z_axis, 1.5);
        /// assert!(after.c.iter().zip(direct.c).all(|(a, b)| (a - b).abs() < 1e-12));
        /// ```
        ///
        /// The axis may be a turn about the line or an arrow along it (ADR-051), as for
        /// [`Motor::rotation`].
        #[inline]
        pub fn rotation<O: crate::Orientation>(axis: oriented::Line<(), T, O>, omega: T) -> Self {
            crate::extras::shared::rotation_twist(
                axis.with_orientation::<crate::Extrinsic>(),
                omega,
            )
        }
    }

    // Methods of lines of either orientation: joins (intrinsic) and meets (extrinsic) agree on
    // the lines they share (ADR-009).
    impl<T: Real, O: crate::Orientation> oriented::Line<(), T, O> {
        /// The direction along this line, a point at infinity: its Euclidean part
        /// `(e23, e31, e12)`, whose length is the line's norm. For the join `p & q` of two
        /// points of weight one it is `q - p`; for the meet `a ^ b` of two planes, the cross
        /// product of their normals, the way a right-handed turn about the line advances.
        ///
        /// ```
        /// use gax::pga3d::{Plane, Point};
        /// let (p, q) = (Point::<(), f64>::xyz(1.0, 2.0, 3.0), Point::xyz(2.0, 4.0, 3.0));
        /// assert_eq!((p & q).direction(), q - p);
        /// let (y0, z0) = (Plane::new(0.0, 1.0, 0.0, 0.0), Plane::new(0.0, 0.0, 1.0, 0.0));
        /// assert_eq!((y0 ^ z0).direction(), Point::direction(1.0, 0.0, 0.0));
        /// ```
        #[inline]
        pub fn direction(self) -> Point<(), T> {
            Point::direction(self.e23(), self.e31(), self.e12())
        }
    }
}

#[cfg(feature = "pga2d")]
mod pga2d_extras {
    use crate::pga2d::{Line, Motor, Point, Translator, extrinsic, oriented};
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

        /// The unit direction at `angle` counterclockwise from the x axis: its x coordinate is
        /// the angle's cosine, its y the sine (a phasor, for oscillations). The inverse of
        /// [`angle_to`](Self::angle_to) from the x axis.
        ///
        /// ```
        /// use gax::pga2d::Point;
        /// let d = Point::<(), f64>::direction_from_angle(std::f64::consts::FRAC_PI_2);
        /// assert!(d.e20().abs() < 1e-15 && (d.e01() - 1.0).abs() < 1e-15 && d.e12() == 0.0);
        /// ```
        #[inline]
        pub fn direction_from_angle(angle: T) -> Self {
            let (sine, cosine) = angle.sin_cos();
            Point::direction(cosine, sine)
        }
    }

    // Methods of points of either orientation (ADR-051).
    impl<T: Real, O: crate::Orientation> oriented::Point<(), T, O> {
        /// The signed angle from this direction to `other`, counterclockwise, in `[-π, π]`. With
        /// `A` and `B` the lines through the origin orthogonal to them, it is `atan2` of the
        /// weight of their meet (the origin) and their inner product: the sine and cosine, scaled
        /// alike, so the directions need no normalizing and the angle is well conditioned
        /// everywhere, a half turn included.
        ///
        /// ```
        /// use gax::pga2d::Point;
        /// let (a, b) = (Point::<(), f64>::direction(1.0, 0.0), Point::direction(0.0, -3.0));
        /// assert!((a.angle_to(b) + std::f64::consts::FRAC_PI_2).abs() < 1e-15);
        /// ```
        pub fn angle_to(self, other: Self) -> T {
            let (a, b) = (self.orthogonal_line(), other.orthogonal_line());
            (a ^ b).e12().atan2((a | b).s())
        }

        /// The angle between this direction and `other`, in `[0, π]`: the size of
        /// [`angle_to`](Self::angle_to), as in every dimension.
        ///
        /// ```
        /// use gax::pga2d::Point;
        /// let (a, b) = (Point::<(), f64>::direction(1.0, 0.0), Point::direction(0.0, -3.0));
        /// assert!((a.angle_between(b) - std::f64::consts::FRAC_PI_2).abs() < 1e-15);
        /// ```
        pub fn angle_between(self, other: Self) -> T {
            let (a, b) = (self.orthogonal_line(), other.orthogonal_line());
            (a ^ b).e12().abs().atan2((a | b).s())
        }

        /// The line through the origin orthogonal to this direction, with its weight.
        fn orthogonal_line(self) -> Line<(), T> {
            Line::new(self.e20(), self.e01(), T::zero())
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
        /// A meet can return a point with any weight, even a negative one, and `normalized()`
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

        /// The position of this point: the intrinsic point of weight `+1` at the same place
        /// (ADR-051). The meet of two joined lines is an extrinsic point, its sign the lines'
        /// handedness; as a place to be it is this.
        ///
        /// ```
        /// use gax::pga2d::Point;
        /// let across = Point::<(), f64>::xy(-1.0, 1.0) & Point::xy(1.0, 1.0);
        /// let up = Point::xy(0.5, -1.0) & Point::xy(0.5, 3.0);
        /// assert_eq!((across ^ up).position(), Point::xy(0.5, 1.0));
        /// ```
        #[inline]
        pub fn position(self) -> Point<(), T> {
            self.unitized().with_orientation()
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
            self.ideal_norm_squared().sqrt()
        }

        /// The squared ideal norm `x² + y²`: a squared length, without the square root.
        ///
        /// ```
        /// use gax::pga2d::Point;
        /// assert_eq!(Point::direction(3.0, 4.0).ideal_norm_squared(), 25.0);
        /// ```
        #[inline]
        pub fn ideal_norm_squared(self) -> T {
            let (x, y) = (self.e20(), self.e01());
            x * x + y * y
        }

        /// The distance between this point and `q`, both finite (of any non-zero weight, either
        /// orientation): the ideal norm of the difference of their positions.
        ///
        /// ```
        /// use gax::pga2d::Point;
        /// let (p, q) = (Point::<(), f64>::xy(1.0, 2.0), Point::new(8.0, 12.0, 2.0));
        /// assert_eq!(p.distance(q), 5.0); // q is (4, 6)
        /// ```
        #[inline]
        pub fn distance<Q: crate::Orientation>(self, q: oriented::Point<(), T, Q>) -> T {
            (q.position() - self.position()).ideal_norm()
        }
    }

    // Twists: in PGA2D bivectors, the blades of points, extrinsic (ADR-051): a turn about a
    // pivot, where a position is intrinsic. Named `Twist`.
    impl<T: Real> extrinsic::Point<(), T> {
        /// The twist (a bivector: in PGA2D, a point) of a translation at the velocity `v`, a
        /// direction of either orientation: `(B * t).exp()` is `Motor::translation(v * t)`.
        /// Twists add, so a moving, turning body's twist is the sum of a translation and a
        /// rotation twist.
        ///
        /// ```
        /// use gax::pga2d::{Point, Twist};
        /// let b = Twist::<(), f64>::translation(Point::direction(3.0, -1.0));
        /// let p = (b.gp(2.0).exp() >> Point::xy(0.0, 0.0)).to_euclidean();
        /// assert!((p[0] - 6.0).abs() < 1e-12 && (p[1] + 2.0).abs() < 1e-12);
        /// ```
        #[inline]
        pub fn translation<O: crate::Orientation>(v: oriented::Point<(), T, O>) -> Self {
            let h = T::from_f64(0.5);
            Point::new(v.e01() * h, -v.e20() * h, T::zero()).with_orientation()
        }

        /// The twist of a rotation at `omega` radians per unit time, counterclockwise, about
        /// `center`: `(B * t).exp()` is `Motor::rotation(center, t omega)`.
        ///
        /// ```
        /// use gax::pga2d::{Motor, Point, Twist};
        /// let c = Point::<(), f64>::xy(1.0, 1.0);
        /// let after = Twist::rotation(c, 0.25).gp(2.0).exp(); // two seconds at 1/4 rad/s
        /// let direct = Motor::rotation(c, 0.5);
        /// assert!(after.c.iter().zip(direct.c).all(|(a, b)| (a - b).abs() < 1e-12));
        /// ```
        ///
        /// The twist is the extrinsic point, a turn about `center` (ADR-051): a pivot, where the
        /// point itself is intrinsic, a position.
        #[inline]
        pub fn rotation<O: crate::Orientation>(
            center: oriented::Point<(), T, O>,
            omega: T,
        ) -> Self {
            crate::extras::shared::rotation_twist(
                center.with_orientation::<crate::Extrinsic>(),
                omega,
            )
        }
    }

    impl<T: Real> Line<(), T> {
        /// The line `n · x = d` with normal `n`, a direction of either orientation (not
        /// necessarily of unit length; its weight is ignored). The inverse of
        /// [`normal`](oriented::Line::normal).
        ///
        /// ```
        /// use gax::pga2d::{Line, Point};
        /// let floor = Line::<(), f64>::with_normal(Point::direction(0.0, 1.0), 2.0); // y = 2
        /// // The join with a point is its signed distance, positive along the normal.
        /// assert_eq!((floor & Point::xy(1.0, 5.0)).s(), 3.0);
        /// assert_eq!(floor.normal(), Point::direction(0.0, 1.0));
        /// ```
        #[inline]
        pub fn with_normal<O: crate::Orientation>(n: oriented::Point<(), T, O>, d: T) -> Self {
            Line::new(n.e20(), n.e01(), -d)
        }

        /// The line through the point `p` (finite, of any weight) along the direction `d`,
        /// both of either orientation: the join of `p` and `d` (of `p` and `p + d`), as an
        /// extrinsic line, so that its [`direction`](oriented::Line::direction) is `d` and its
        /// normal `d` turned a quarter counterclockwise.
        ///
        /// ```
        /// use gax::pga2d::{Line, Point};
        /// let l = Line::<(), f64>::through(Point::xy(1.0, 2.0), Point::direction(1.0, 0.0));
        /// assert_eq!(l, Line::with_normal(Point::direction(0.0, 1.0), 2.0)); // y = 2
        /// assert_eq!(l.direction(), Point::direction(1.0, 0.0));
        /// ```
        #[inline]
        pub fn through<P: crate::Orientation, D: crate::Orientation>(
            p: oriented::Point<(), T, P>,
            d: oriented::Point<(), T, D>,
        ) -> Self {
            (p.position() & Point::direction(d.e20(), d.e01())).with_orientation()
        }
    }

    // Methods of lines of either orientation: a join's coefficients and an extrinsic line's
    // agree on the line they describe.
    impl<T: Real, O: crate::Orientation> oriented::Line<(), T, O> {
        /// The normal `(a, b)` of the line `a x + b y + c = 0`, as a direction: it points to the
        /// side where `line & x` is positive, and its length is the line's norm. For a join
        /// `p & q` it points to the left of `q - p`.
        ///
        /// ```
        /// use gax::pga2d::{Line, Point};
        /// assert_eq!(Line::<(), f64>::new(3.0, 4.0, -1.0).normal(), Point::direction(3.0, 4.0));
        /// let along_x = Point::xy(0.0, 0.0) & Point::xy(2.0, 0.0);
        /// assert_eq!(along_x.normal(), Point::direction(0.0, 2.0));
        /// ```
        #[inline]
        pub fn normal(self) -> Point<(), T> {
            Point::direction(self.e1(), self.e2())
        }

        /// The direction along this line, a point at infinity: its normal turned a quarter
        /// clockwise, `(b, −a)`, whose length is the line's norm. For the join `p & q` of two
        /// points of weight one it is `q - p`.
        ///
        /// ```
        /// use gax::pga2d::Point;
        /// let (p, q) = (Point::<(), f64>::xy(1.0, 2.0), Point::xy(4.0, 6.0));
        /// assert_eq!((p & q).direction(), q - p);
        /// ```
        #[inline]
        pub fn direction(self) -> Point<(), T> {
            Point::direction(self.e2(), -self.e1())
        }
    }

    impl<T: Real, M: crate::Orientation> oriented::Line<(), T, M> {
        /// Reflect a point (or a direction) in this line: `>>` with the normalized line, whose sign
        /// the point's orientation picks (ADR-051), so a position keeps its weight and a direction
        /// is mirrored. The line's own orientation, a join's or a meet's, does not matter: it
        /// enters twice.
        ///
        /// ```
        /// use gax::pga2d::Point;
        /// let wall = Point::xy(0.0, 0.0) & Point::xy(0.0, 1.0);
        /// let v = wall.reflect(Point::direction(1.0, 0.5));
        /// assert_eq!([v.e20(), v.e01(), v.e12()], [-1.0, 0.5, 0.0]);
        /// ```
        #[inline]
        pub fn reflect<O: crate::Orientation>(
            self,
            x: oriented::Point<(), T, O>,
        ) -> oriented::Point<(), T, O> {
            self.normalized() >> x
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

        /// The translation by the direction `d` (a point at infinity, of either orientation:
        /// `Point::direction(x, y)`, or the difference `q - p` of two points of weight one, the
        /// translation from `p` to `q`). Only its ideal part is read.
        ///
        /// ```
        /// use gax::pga2d::{Motor, Point};
        /// let t = Motor::<(), f64>::translation(Point::direction(3.0, -1.0));
        /// assert_eq!((t >> Point::xy(1.0, 1.0)).to_euclidean(), [4.0, 0.0]);
        /// let (a, b) = (Point::xy(1.0, 1.0), Point::xy(2.0, 3.0));
        /// assert_eq!(Motor::translation(b - a) >> a, b);
        /// ```
        #[inline]
        pub fn translation<O: crate::Orientation>(d: oriented::Point<(), T, O>) -> Unit<Self> {
            let h = T::from_f64(0.5);
            // A translator, as a motor.
            let t = Translator::new(T::one(), d.e01() * h, -d.e20() * h);
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
        pub fn rotation<O: crate::Orientation>(
            center: oriented::Point<(), T, O>,
            angle: T,
        ) -> Unit<Self> {
            crate::extras::shared::rotation_twist(
                center.with_orientation::<crate::Extrinsic>(),
                angle,
            )
            .exp()
        }

        /// The shortest rotation about the origin that turns the direction `from` into the
        /// direction `to`: by their signed angle, [`Point::angle_to`] (no normalization needed,
        /// well conditioned everywhere, a half turn included), which is also its
        /// [`Motor::angle`]. The directions may have either orientation.
        ///
        /// ```
        /// use gax::pga2d::{Motor, Point};
        /// let r = Motor::rotation_between(Point::<(), f64>::direction(1.0, 0.0), Point::direction(0.0, 2.0));
        /// assert!((r.angle() - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
        /// ```
        pub fn rotation_between<A: crate::Orientation, B: crate::Orientation>(
            from: oriented::Point<(), T, A>,
            to: oriented::Point<(), T, B>,
        ) -> Unit<Self> {
            let zero = T::zero();
            Self::rotation(Point::xy(zero, zero), from.angle_to(to.with_orientation()))
        }

        /// The rigid motion from `a` to `b`, at `t`: `a exp(t log(~a b))`, the shorter way (the
        /// relative motor taken with its scalar part non-negative; see the 3D
        /// `Motor::interpolate`).
        ///
        /// ```
        /// use gax::pga2d::{Motor, Point};
        /// let (a, b) = (gax::Unit::<Motor<(), f64>>::identity(), Motor::translation(Point::direction(4.0, 2.0)));
        /// let halfway = Motor::interpolate(a, b, 0.5);
        /// let [x, y] = (halfway >> Point::xy(0.0, 0.0)).to_euclidean();
        /// assert!((x - 2.0).abs() < 1e-12 && (y - 1.0).abs() < 1e-12);
        /// ```
        pub fn interpolate(a: Unit<Self>, b: Unit<Self>, t: T) -> Unit<Self> {
            crate::extras::shared::interpolate::<Self, crate::pga2d::Twist<(), T>, T>(a, b, t)
        }
        /// The angle of a rotation (a unit motor), counterclockwise, in `[-π, π]`: from its
        /// logarithm, whose point part is `-angle/2` times the (unit) centre.
        ///
        /// ```
        /// use gax::pga2d::{Motor, Point};
        /// let r = Motor::rotation(Point::<(), f64>::xy(3.0, 1.0), -0.4);
        /// assert!((r.angle() + 0.4).abs() < 1e-12);
        /// ```
        pub fn angle(self) -> T {
            let log: crate::pga2d::Twist<(), T> = Unit::new_unchecked(self).log();
            log.e12() * T::from_f64(-2.0)
        }
    }
}

/// Angles of Euclidean vectors, from their outer and inner products: `angle_between`, unsigned,
/// in every dimension; `angle_to`, signed, in the plane, where an orientation is given; and the
/// unit vector `from_angle`, the inverse of `angle_to` from `e1`. Rotors: a rotation by an
/// angle, its angle, and the interpolation between two.
#[cfg(feature = "vga2d")]
mod vga2d_extras {
    use crate::vga2d::{Bivector, Rotor, Vector};
    use crate::{Real, Unit};

    impl<T: Real> Rotor<(), T> {
        /// The rotation by `angle`, counterclockwise (from `e1` towards `e2`):
        /// `exp(−e12 angle / 2)`, `cos(angle/2) − sin(angle/2) e12`.
        ///
        /// ```
        /// use gax::vga2d::{Rotor, Vector};
        /// let r = Rotor::<(), f64>::rotation(std::f64::consts::FRAC_PI_2);
        /// let v = r >> Vector::new(1.0, 0.0);
        /// assert!(v.e1().abs() < 1e-15 && (v.e2() - 1.0).abs() < 1e-15);
        /// ```
        #[inline]
        pub fn rotation(angle: T) -> Unit<Self> {
            Bivector::new(angle * T::from_f64(-0.5)).exp()
        }

        /// The angle of this rotation, counterclockwise, in `[-π, π]`: `atan2` of the sine and
        /// cosine of the full angle, `−2ab` and `a² − b²` for the rotor `a + b e12`, so `r` and
        /// `−r` (the same rotation) agree and any scale cancels.
        ///
        /// ```
        /// use gax::vga2d::Rotor;
        /// let r = Rotor::<(), f64>::rotation(-2.5);
        /// assert!((r.angle() + 2.5).abs() < 1e-15);
        /// assert_eq!(Rotor::<(), f64>::new(-2.0, 0.0).angle(), 0.0); // −2: no turn
        /// ```
        #[inline]
        pub fn angle(self) -> T {
            let (a, b) = (self.s(), self.e12());
            let sine = -(a * b + a * b);
            sine.atan2(a * a - b * b)
        }

        /// The rotation from `a` to `b` at `t` (`0` is `a`, `1` is `b`), the shorter way:
        /// `a exp(t log(~a b))` with the relative rotor's scalar part non-negative, as for
        /// motors (`gax::pga3d::Motor::interpolate`).
        ///
        /// ```
        /// use gax::vga2d::Rotor;
        /// let (a, b) = (Rotor::<(), f64>::rotation(0.25), Rotor::rotation(-0.75));
        /// assert!((Rotor::interpolate(a, b, 0.5).angle() + 0.25).abs() < 1e-15);
        /// ```
        pub fn interpolate(a: Unit<Self>, b: Unit<Self>, t: T) -> Unit<Self> {
            crate::extras::shared::interpolate::<Self, Bivector<(), T>, T>(a, b, t)
        }

        /// The rotor that carries the vector `a` onto the direction of `b`, `sqrt(b / a)`: the
        /// rotation by their signed angle ([`Between`](crate::vga2d::Between)).
        ///
        /// ```
        /// use gax::vga2d::{Rotor, Vector};
        /// let r = Rotor::between(Vector::<(), f64>::new(2.0, 0.0), Vector::new(-1.0, 1.0));
        /// assert!((r.angle() - 3.0 * std::f64::consts::FRAC_PI_4).abs() < 1e-15);
        /// ```
        #[inline]
        pub fn between<X: crate::extras::Between<Self>>(a: X, b: X) -> Unit<Self> {
            X::between(a, b)
        }
    }

    impl<T: Real> Vector<(), T> {
        /// The unit vector at `angle` counterclockwise from `e1`: its `e1` coordinate is the
        /// angle's cosine, its `e2` the sine. The inverse of [`angle_to`](Self::angle_to) from
        /// `e1`.
        ///
        /// ```
        /// use gax::vga2d::Vector;
        /// let v = Vector::<(), f64>::from_angle(std::f64::consts::PI);
        /// assert!((v.e1() + 1.0).abs() < 1e-15 && v.e2().abs() < 1e-15);
        /// ```
        #[inline]
        pub fn from_angle(angle: T) -> Self {
            let (sine, cosine) = angle.sin_cos();
            Vector::new(cosine, sine)
        }

        /// The signed angle from this vector to `other`, counterclockwise, in `[-π, π]`: `atan2`
        /// of their outer product (the sine, as a multiple of `e12`) and their inner product
        /// (the cosine), scaled alike, so no normalizing is needed and a half turn is exact.
        ///
        /// ```
        /// use gax::vga2d::Vector;
        /// let (a, b) = (Vector::<(), f64>::new(2.0, 0.0), Vector::new(-1.0, 1.0));
        /// assert!((a.angle_to(b) - 3.0 * std::f64::consts::FRAC_PI_4).abs() < 1e-15);
        /// ```
        pub fn angle_to(self, other: Self) -> T {
            (self ^ other).e12().atan2((self | other).s())
        }

        /// The angle between this vector and `other`, in `[0, π]`: the size of
        /// [`angle_to`](Self::angle_to), as in every dimension.
        ///
        /// ```
        /// use gax::vga2d::Vector;
        /// let (a, b) = (Vector::<(), f64>::new(2.0, 0.0), Vector::new(-1.0, -1.0));
        /// assert!((a.angle_between(b) - 3.0 * std::f64::consts::FRAC_PI_4).abs() < 1e-15);
        /// ```
        pub fn angle_between(self, other: Self) -> T {
            (self ^ other).e12().abs().atan2((self | other).s())
        }
    }
}

/// Angles of Euclidean vectors, from their outer and inner products (see the 2D module), and
/// rotors: about an axis, between two vectors or planes, and interpolated.
#[cfg(feature = "vga3d")]
mod vga3d_extras {
    use crate::vga3d::{Bivector, Rotor, Vector, oriented};
    use crate::{Real, Unit};

    impl<T: Real> Rotor<(), T> {
        /// The rotation by `angle` about the axis `axis` (a vector of either orientation, of
        /// any length), right-handed: `exp(−B angle / 2)` with `B` the unit bivector of the plane
        /// orthogonal to the axis (`e12` for `e3`), as `gax::pga3d::Motor::rotation_about`.
        ///
        /// ```
        /// use gax::vga3d::{Rotor, Vector};
        /// let r = Rotor::<(), f64>::rotation(Vector::new(0.0, 0.0, 2.0), std::f64::consts::FRAC_PI_2);
        /// let v = r >> Vector::new(1.0, 0.0, 0.0);
        /// assert!(v.e1().abs() < 1e-15 && (v.e2() - 1.0).abs() < 1e-15 && v.e3().abs() < 1e-15);
        /// ```
        #[inline]
        pub fn rotation<O: crate::Orientation>(
            axis: oriented::Vector<(), T, O>,
            angle: T,
        ) -> Unit<Self> {
            let plane = Bivector::new(axis.e1(), axis.e2(), axis.e3());
            crate::extras::shared::rotation_twist(plane, angle).exp()
        }

        /// The rotation from `a` to `b` at `t` (`0` is `a`, `1` is `b`), the shorter way:
        /// `a exp(t log(~a b))` with the relative rotor's scalar part non-negative, as for
        /// motors (`gax::pga3d::Motor::interpolate`).
        ///
        /// ```
        /// use gax::vga3d::{Rotor, Vector};
        /// let z = Vector::<(), f64>::new(0.0, 0.0, 1.0);
        /// let (a, b) = (Rotor::rotation(z, 0.1), Rotor::rotation(z, 0.3));
        /// let v = Rotor::interpolate(a, b, 0.5) >> Vector::new(1.0, 0.0, 0.0);
        /// assert!((v.e2() - 0.2f64.sin()).abs() < 1e-15);
        /// ```
        pub fn interpolate(a: Unit<Self>, b: Unit<Self>, t: T) -> Unit<Self> {
            crate::extras::shared::interpolate::<Self, Bivector<(), T>, T>(a, b, t)
        }

        /// The rotor that carries `a` onto `b`, vectors (onto its direction) or bivectors
        /// (planes through the origin): `sqrt(b / a)`, the shortest rotation, about the axis
        /// orthogonal to both vectors or along the line where the planes meet
        /// ([`Between`](crate::vga3d::Between)). Not defined for `b` opposite to `a`.
        ///
        /// ```
        /// use gax::vga3d::{Rotor, Vector};
        /// let (a, b) = (Vector::<(), f64>::new(1.0, 0.0, 0.0), Vector::new(0.0, 3.0, 4.0));
        /// let v = Rotor::between(a, b) >> a;
        /// assert!(v.e1().abs() < 1e-15 && (v.e2() - 0.6).abs() < 1e-15 && (v.e3() - 0.8).abs() < 1e-15);
        /// ```
        #[inline]
        pub fn between<X: crate::extras::Between<Self>>(a: X, b: X) -> Unit<Self> {
            X::between(a, b)
        }
    }

    impl<T: Real> Vector<(), T> {
        /// The angle between this vector and `other`, in `[0, π]`: `atan2` of the norm of their
        /// outer product (the sine, the area they span) and their inner product (the cosine),
        /// scaled alike, so no normalizing is needed and the angle is well conditioned at every
        /// angle (an arc cosine of the normalized inner product loses half the digits near 0
        /// and π).
        ///
        /// ```
        /// use gax::vga3d::Vector;
        /// let (a, b) = (Vector::<(), f64>::new(1.0, 0.0, 0.0), Vector::new(1.0, 1.0, 0.0));
        /// assert!((a.angle_between(b) - std::f64::consts::FRAC_PI_4).abs() < 1e-15);
        /// ```
        pub fn angle_between(self, other: Self) -> T {
            (self ^ other).norm().atan2((self | other).s())
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

            /// The sphere (or circle) about the round point `centre` (of any weight) with the
            /// given radius: `C - ½r² ei` for the centre `C` of weight one, in the dual
            /// representation (a point `X` lies on it when `X · s = 0`).
            #[inline]
            pub fn sphere(centre: Self, radius: T) -> Self {
                let mut s = centre.gp(centre.c[$n].recip());
                s.c[$n + 1] = s.c[$n + 1] - T::from_f64(0.5) * radius * radius;
                s
            }
        }

        #[cfg(feature = $pga_feature)]
        impl<T: crate::Real> crate::$alg::Vector<(), T> {
            /// The round point of a PGA point (of weight `w`, either orientation): `w up(p / w)`.
            #[inline]
            pub fn from_point<O: crate::Orientation>(p: crate::$pga::oriented::Point<(), T, O>) -> Self {
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
        // Twists to momenta: extrinsic in, intrinsic out (ADR-051).
        let mut dense: crate::pga3d::Forque<(Line,), f64> = crate::pga3d::Forque::zero();
        for (x, y, z) in pts {
            let p = Point::xyz(x, y, z);
            dense += p & p.commutator(Line::slot()); // Σ m X ∨ (X × B), in extensor form
        }
        let compact = PrincipalInertia::new(6.0, [26.0, 20.0, 10.0]);
        assert_eq!(compact.to_dense(), dense);
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
        let t = Motor::translation(Point::direction(1.0, -2.0, 0.5));
        assert!(close(
            (t >> Point::xyz(1.0, 1.0, 1.0)).to_euclidean(),
            [2.0, -1.0, 1.5]
        ));
        let r = Motor::rotation_about(
            Point::direction(0.0, 0.0, 1.0),
            core::f64::consts::FRAC_PI_2,
        );
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
        let floor = Plane::<(), f64>::with_normal(Point::direction(0.0, 0.0, 1.0), 1.0);
        assert!(((floor & Point::xyz(3.0, 4.0, 3.0)).s() - 2.0).abs() < 1e-12);
    }

    #[cfg(feature = "pga2d")]
    #[test]
    fn pga2d_conventions() {
        use crate::pga2d::{Motor, Point};
        let close = |a: [f64; 2], b: [f64; 2]| a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-12);
        let t = Motor::translation(Point::direction(1.0, -2.0));
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
            let tr = crate::pga2d::Twist::translation(crate::pga2d::Point::direction(3.0, -1.0))
                .gp(t)
                .exp()
                .into_inner();
            let want = Motor::translation(Point::direction(3.0 * t, -t)).into_inner();
            assert!(
                tr.c.iter()
                    .zip(want.c)
                    .all(|(a, b): (&f64, f64)| (a - b).abs() < 1e-12)
            );
            let ro = crate::pga2d::Twist::rotation(c, 0.7)
                .gp(t)
                .exp()
                .into_inner();
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
        let s = cga3d::Vector::sphere(cga3d::Vector::up(1.0, 2.0, 3.0), 5.0);
        assert!((y | s).s().abs() < 1e-12); // (4, 6, 3) is 5 from the centre
        let p = pga3d::Point::xyz(1.0, 2.0, 3.0);
        assert_eq!(cga3d::Vector::from_point(p).down(), [1.0, 2.0, 3.0]);
        assert_eq!(cga3d::Vector::from_point(p).to_point(), p);
        let plane = pga3d::Plane::with_normal(pga3d::Point::direction(0.0, 0.0, 1.0), 1.0); // z = 1
        let dual: cga3d::Vector<(), f64> = plane.into();
        assert!(((x | dual).s() - (plane & p).s()).abs() < 1e-12);
        assert!(((x | dual).s() - 2.0).abs() < 1e-12);
    }

    #[cfg(feature = "pga3d")]
    #[test]
    fn pga3d_twists() {
        use crate::pga3d::{Motor, Point, Twist};
        let axis = Point::xyz(1.0, 0.0, 0.0) & Point::xyz(1.0, 0.5, 2.0);
        for t in [0.3, 1.0, 2.5] {
            let tr = Twist::translation(Point::direction(3.0, -1.0, 0.5))
                .gp(t)
                .exp()
                .into_inner();
            let want = Motor::translation(Point::direction(3.0 * t, -t, 0.5 * t)).into_inner();
            assert!(
                tr.c.iter()
                    .zip(want.c)
                    .all(|(a, b): (&f64, f64)| (a - b).abs() < 1e-12)
            );
            let ro = Twist::rotation(axis, 0.7).gp(t).exp().into_inner();
            let want = Motor::rotation(axis, 0.7 * t).into_inner();
            assert!(
                ro.c.iter()
                    .zip(want.c)
                    .all(|(a, b): (&f64, f64)| (a - b).abs() < 1e-12)
            );
        }
    }
}
