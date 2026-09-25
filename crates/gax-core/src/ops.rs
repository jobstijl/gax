//! Product and involution traits implemented by the generated algebras.
//!
//! Each binary product is a trait with one generated impl per pair of kinds whose product is
//! nonzero. The impls are generic over both slot lists and the coefficient type, and the output
//! slots are the concatenation. The operators `*`, `^`, `&` and `|` forward to [`Gp`],
//! [`Wedge`], [`Vee`] and [`Dot`].

macro_rules! binary_trait {
    ($(#[$doc:meta])* $Trait:ident, $method:ident) => {
        $(#[$doc])*
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
    Gp, gp
);
binary_trait!(
    /// Outer (wedge) product, the meet in plane-based PGA. Metric free.
    Wedge, wedge
);
binary_trait!(
    /// Regressive (vee) product, the join in plane-based PGA. Metric free.
    Vee, vee
);
binary_trait!(
    /// Symmetric inner product: the grade `|ga - gb|` part of the geometric product.
    Dot, dot
);
binary_trait!(
    /// Left contraction.
    Lc, lc
);
binary_trait!(
    /// Right contraction.
    Rc, rc
);
binary_trait!(
    /// Scalar product: the grade-0 part of the geometric product.
    ScalarProduct, scalar_product
);
binary_trait!(
    /// Commutator product `(ab - ba) / 2`.
    Commutator, commutator
);
binary_trait!(
    /// Anticommutator product `(ab + ba) / 2`.
    Anticommutator, anticommutator
);
binary_trait!(
    /// Versor transport `v x ~v` (the operator `v >> x`).
    Transform, transform
);
binary_trait!(
    /// Inverse transport `~v x v` (the operator `v << x`).
    TransformInv, transform_inv
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
