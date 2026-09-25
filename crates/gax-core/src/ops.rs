//! Product and involution traits implemented by the generated algebras.
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
            note = "the product is identically zero for these kinds (so it is not defined), or an operand is not a multivector of the same algebra and coefficient type (versor transport needs a declared versor kind as a value on the left)"
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

/// The logarithm of a unit versor, a bivector (implemented by the generated algebras for
/// `Unit<K>`; call it as `unit.log()`).
pub trait Log<Out> {
    /// The logarithm: `exp(self.log()) == self`.
    fn log(self) -> Out;
}
