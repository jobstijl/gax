//! Product and involution traits implemented by the generated algebras.
//!
//! ```
//! use gax::pga3d::{Line, Plane, Point};
//! let (p, q) = (Point::<(), f64>::xyz(0.0, 0.0, 0.0), Point::xyz(1.0, 1.0, 0.0));
//! let l: Line<(), f64> = p & q;           // join (regressive product)
//! let floor = Plane::from_normal([0.0, 0.0, 1.0], 0.0);
//! let _meet: Point<(), f64> = l ^ floor;  // meet (outer product)
//! assert!((floor & p).s().abs() < 1e-12); // incidence: the origin lies on the floor
//! ```
//!
//! Each binary product is a trait with one generated impl per pair of kinds whose product is
//! nonzero. The impls are generic over both slot lists and the coefficient type, and the output
//! slots are the concatenation. The operators `*`, `^`, `&` and `|` forward to [`Gp`],
//! [`Wedge`], [`Vee`] and [`Dot`].

macro_rules! binary_trait {
    ($(#[$doc:meta])* $Trait:ident, $method:ident, $msg:literal) => {
        $(#[$doc])*
        #[diagnostic::on_unimplemented(
            message = $msg,
            note = "the product is identically zero for these kinds (so it is not defined), or an operand is not a multivector of the same algebra and coefficient type (versor transport needs a declared versor kind as a value on the left)",
            note = "a structured map (`map::LinearMap`, such as `PrincipalInertia`) takes part in products as its dense form, `.to_dense()`"
        )]
        pub trait $Trait<Rhs> {
            /// The result type.
            type Output;
            $(#[$doc])*
            fn $method(self, rhs: Rhs) -> Self::Output;
        }
    };
}

binary_trait!(
    /// Geometric product.
    Gp, gp,
    "no geometric product of `{Self}` and `{Rhs}`"
);
binary_trait!(
    /// Outer (wedge) product, the meet in plane-based PGA. Metric free.
    Wedge, wedge,
    "no outer product (`^`) of `{Self}` and `{Rhs}`"
);
binary_trait!(
    /// Regressive (vee) product, the join in plane-based PGA. Metric free.
    Vee, vee,
    "no regressive product (`&`) of `{Self}` and `{Rhs}`"
);
binary_trait!(
    /// Symmetric inner product: the grade `|ga - gb|` part of the geometric product.
    Dot, dot,
    "no inner product (`|`) of `{Self}` and `{Rhs}`"
);
binary_trait!(
    /// Left contraction.
    Lc, lc,
    "no left contraction of `{Self}` and `{Rhs}`"
);
binary_trait!(
    /// Right contraction.
    Rc, rc,
    "no right contraction of `{Self}` and `{Rhs}`"
);
binary_trait!(
    /// Scalar product: the grade-0 part of the geometric product.
    ScalarProduct, scalar_product,
    "no scalar product of `{Self}` and `{Rhs}`"
);
binary_trait!(
    /// Commutator product `(ab - ba) / 2`.
    Commutator, commutator,
    "no commutator product of `{Self}` and `{Rhs}`"
);
binary_trait!(
    /// Anticommutator product `(ab + ba) / 2`.
    Anticommutator, anticommutator,
    "no anticommutator product of `{Self}` and `{Rhs}`"
);
binary_trait!(
    /// Versor transport `v x ~v` (the operator `v >> x`).
    Transform, transform,
    "no versor transport (`>>`) of `{Self}` and `{Rhs}`"
);
binary_trait!(
    /// Inverse transport `~v x v` (the operator `v << x`).
    TransformInv, transform_inv,
    "no inverse versor transport (`<<`) of `{Self}` and `{Rhs}`"
);

/// Division `a / b`: the geometric product with the inverse, `a b⁻¹` (right division, as in
/// ganja.js). Implemented by the generated algebras for every coefficient `b` and for every
/// value `b` of a kind with a closed-form inverse whose product with `a` exists; `a` may carry
/// slots, `b` may not (the inverse is not linear).
///
/// The quotient of two objects of the same kind is the motion between them, twice: for unit
/// planes `b / a` is the rotation (or translation) that takes `a` to `b` and on as far again,
/// and its square root takes `a` to `b` (`Motor::between`).
///
/// ```
/// use gax::pga3d::{Plane, Point};
/// let a = Plane::<(), f64>::from_normal([1.0, 0.0, 0.0], 0.0);
/// let b = Plane::from_normal([0.0, 1.0, 0.0], 0.0);
/// let r = (b / a).normalized(); // a quarter turn, twice: a half turn about z
/// let q = (r >> Point::xyz(1.0, 0.0, 0.0)).to_euclidean();
/// assert!((q[0] + 1.0).abs() < 1e-12 && q[1].abs() < 1e-12);
/// ```
#[diagnostic::on_unimplemented(
    message = "no division of `{Self}` by `{Rhs}`",
    note = "division is by a coefficient or by a value (no open slots) of a kind with an inverse, whose geometric product with the left side exists"
)]
pub trait DivBy<Rhs> {
    /// The result type.
    type Output;
    /// `self rhs⁻¹`.
    fn div_by(self, rhs: Rhs) -> Self::Output;
}

macro_rules! unary_trait {
    ($(#[$doc:meta])* $Trait:ident, $method:ident) => {
        $(#[$doc])*
        pub trait $Trait {
            /// The result type.
            type Output;
            $(#[$doc])*
            fn $method(self) -> Self::Output;
        }
    };
}

unary_trait!(
    /// Reverse `~a`: reverses the order of the vectors in every blade.
    Reverse, reverse
);
unary_trait!(
    /// Grade involution: negates the odd grades.
    Involute, involute
);
unary_trait!(
    /// Clifford conjugate: reverse composed with the grade involution.
    Conjugate, conjugate
);
unary_trait!(
    /// Right complement `J`, with `a ^ J(a) = I` for every basis blade. Metric free.
    Dual, dual
);
unary_trait!(
    /// Left complement, the inverse of [`Dual`].
    Undual, undual
);

/// The exponential of a bivector: a unit versor. Implemented by the generated algebras for the
/// kinds that have `exp`, so generic code can call it.
///
/// ```
/// use gax::{Exp, Gp, Normalize};
/// use gax::{pga2d, pga3d};
/// // The rotation by `angle` about a flat: a line in PGA3D, a point in PGA2D.
/// fn rotation<X>(about: X, angle: f64) -> <X as Exp>::Output
/// where
///     X: Normalize + Exp + Gp<f64, Output = X>,
/// {
///     about.normalized().into_inner().gp(-0.5 * angle).exp()
/// }
/// let z = pga3d::Point::<(), f64>::xyz(0.0, 0.0, 0.0) & pga3d::Point::xyz(0.0, 0.0, 1.0);
/// let r3 = rotation(z, 1.0);
/// let r2 = rotation(pga2d::Point::<(), f64>::xy(0.0, 0.0), 1.0);
/// assert!((r3.into_inner().s() - r2.into_inner().s()).abs() < 1e-15);
/// ```
pub trait Exp {
    /// The result: a unit versor.
    type Output;
    /// The exponential.
    fn exp(self) -> Self::Output;
}

/// The inverse under the geometric product, `x x⁻¹ = 1`: a closed form where a kind has one,
/// else the general inverse. Implemented by the generated algebras for the kinds that have
/// `inverse`. (Maps have their own `inverse`, `SquareMap::inverse`.)
pub trait Inverse {
    /// The kind of the inverse.
    type Output;
    /// The inverse.
    fn inverse(self) -> Self::Output;
}

/// The norm of a value: `norm_squared` is the scalar part of `x ~x`, `norm` the square root of
/// its absolute value. Implemented by the generated algebras for the kinds that have `norm`.
pub trait Norm: crate::kind::Extensor {
    /// The scalar part of `x ~x`.
    fn norm_squared(self) -> Self::Coef;
    /// `sqrt(|norm_squared|)`.
    fn norm(self) -> Self::Coef;
}

/// Normalization to a certified unit versor. Implemented by the generated algebras for the
/// kinds that have `normalized`.
pub trait Normalize: Sized {
    /// `self` scaled to unit norm, as a [`Unit`](crate::Unit).
    fn normalized(self) -> crate::unit::Unit<Self>;
}

/// The square root of a value. Implemented by the generated algebras for the kinds that have
/// `sqrt`.
pub trait Sqrt {
    /// The result.
    type Output;
    /// The square root.
    fn sqrt(self) -> Self::Output;
}

/// The logarithm of a unit versor, a bivector (implemented by the generated algebras for
/// `Unit<K>`; call it as `unit.log()`).
///
/// ```
/// use gax::pga3d::{Line, Motor};
/// let m = Motor::<(), f64>::rotation_about(0.0, 0.0, 1.0, 1.2);
/// let b: Line<(), f64> = m.log();
/// // Half the angle, about the z axis (the line through the origin along z).
/// assert!((b.norm() - 0.6).abs() < 1e-12);
/// ```
pub trait Log<Out> {
    /// The logarithm: `exp(self.log()) == self`.
    fn log(self) -> Out;
}

/// The outermorphism of a linear map on vectors (or antivectors): its extension to the kind
/// `B` of higher (or lower) grade, `M(a ∧ b) = T(a) ∧ T(b)` (for antivectors, with `∨`).
/// On the top grade it is the determinant. Implemented by the generated algebras for maps
/// `V <- V` on a kind holding every vector (or every antivector), to each kind `B` holding
/// every blade of one grade.
///
/// The extension to blades of `k` factors in `n` dimensions has `C(n, k)²` entries, each a
/// minor of `k!` terms, so it is generated only up to 16 384 terms (`gax_gen`'s
/// `OUTER_MAX_TERMS`): every extension up to 6 dimensions; in 7, those to 2, 3 and 7 factors
/// (bivectors, trivectors and the determinant, for a map on vectors); in 8 and 9, to 2. Larger
/// ones (the 4-vectors of an 8D algebra: 117 600 terms) take gigabytes to compile. For those,
/// build the extended map from its images with `B::<(B,), T>::from_images`: the image of a
/// basis blade `e_a ∧ e_b ∧ …` of `B` is `(t.of(e_a) ^ t.of(e_b) ^ …).cast::<B>()`.
///
/// ```
/// use gax::Outermorphism;
/// use gax::pga3d::{Line, Plane};
/// // A linear map on PGA3D's vectors (planes), extended to its bivectors (lines).
/// let t = Plane::<(Plane,), f64>::from_coeffs([
///     [2.0, 1.0, 0.0, 0.0],
///     [0.0, 3.0, 0.0, 1.0],
///     [1.0, 0.0, 4.0, 0.0],
///     [0.0, 0.0, 1.0, 1.0],
/// ]);
/// let t2 = Outermorphism::<Line>::outermorphism(t);
/// let (a, b) = (Plane::new(1.0, 2.0, 0.0, 1.0), Plane::new(0.0, 1.0, -1.0, 2.0));
/// // The extension maps a wedge to the wedge of the images.
/// assert_eq!(t2.of(a ^ b), t.of(a) ^ t.of(b));
/// ```
#[diagnostic::on_unimplemented(
    message = "no outermorphism from maps on `{Self}` to `{B}`",
    note = "outermorphisms extend maps on a kind holding every vector (or antivector) to a kind holding every blade of one grade, up to 16 384 terms of minors (C(n, k)² k! for k factors in n dimensions)",
    note = "for larger kinds, build the extended map from its images: `B::<(B,), T>::from_images`, the image of a blade `e_a ∧ e_b ∧ …` being `t.of(e_a) ^ t.of(e_b) ^ …`"
)]
pub trait Outermorphism<B> {
    /// The extended map `B <- B`.
    type Output;
    /// Extend the map to `B`.
    fn outermorphism(self) -> Self::Output;
}
