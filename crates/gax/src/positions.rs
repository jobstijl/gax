//! The positions of the standard models of space, for integration over simplices
//! (`gax::simplex`, ADR-048): how each model forms the points of a simplex and its directed
//! measure.
//!
//! * VGA and STA: positions are vectors, and the measure of a `k`-simplex is the wedge of its
//!   edges from the first vertex over `k!`.
//! * PGA: positions are points of weight one, and the measure is the join of the vertices over
//!   `k!`: an edge's line (its norm the length), a triangle's plane (in 3D) and, for a simplex of
//!   the full dimension, a scalar, signed as the VGA measure's pseudoscalar coefficient. Joins of
//!   points are intrinsic (ADR-051): an edge's line points along it, a triangle's plane winds,
//!   and the volume's sign is a handedness, which a mirror reverses.
//! * CGA: positions are round points of weight one, `eo + x + ½|x|² e∞`, and the measure is
//!   `P₀ ∧ … ∧ P_k ∧ e∞` over `k!`, the flat through the vertices, signed as the VGA measure.
//!
//! The measure of a single vertex is the scalar one in every model.

use crate::Coef;
use crate::simplex::{Measure, Position};

/// `Σ wᵢ pᵢ`.
#[cfg(any(
    feature = "vga2d",
    feature = "vga3d",
    feature = "sta",
    feature = "pga2d",
    feature = "pga3d"
))]
#[inline]
fn combination<P, T>(points: &[P], weights: &[T]) -> P
where
    P: Copy + core::ops::Add<Output = P> + core::ops::Mul<T, Output = P>,
    T: Copy,
{
    points
        .iter()
        .zip(weights)
        .map(|(&p, &w)| p * w)
        .reduce(|a, b| a + b)
        .expect("a simplex has a vertex")
}

/// `1 / k!` as a coefficient.
#[inline]
fn inverse_factorial<T: Coef>(k: i64) -> T {
    T::from_ratio(1, (1..=k).product())
}

/// The vector models: positions are vectors, measures wedges of edges. `$k => $Blade` for each
/// simplex dimension from 1 to the algebra's.
macro_rules! vector_model {
    ($alg:ident, $feature:literal, $($n:literal: $Blade:ident = |$e:ident| $wedge:expr;)+) => {
        #[cfg(feature = $feature)]
        const _: () = {
            use crate::$alg::{Scalar, Vector};

            impl<T: Coef> Position for Vector<(), T> {
                type Coef = T;
                #[inline]
                fn affine(points: &[Self], weights: &[T]) -> Self {
                    combination(points, weights)
                }
            }

            impl<T: Coef> Measure<1> for Vector<(), T> {
                type Blade = Scalar<(), T>;
                #[inline]
                fn measure(_: &[Self; 1]) -> Scalar<(), T> {
                    Scalar::new(T::one())
                }
            }

            $(
                impl<T: Coef> Measure<$n> for Vector<(), T> {
                    type Blade = crate::$alg::$Blade<(), T>;
                    #[inline]
                    fn measure(v: &[Self; $n]) -> Self::Blade {
                        let $e: [Vector<(), T>; $n - 1] = core::array::from_fn(|i| v[i + 1] - v[0]);
                        ($wedge) * inverse_factorial::<T>($n - 1)
                    }
                }
            )+
        };
    };
}

vector_model!(vga2d, "vga2d",
    2: Vector = |e| e[0];
    3: Pseudoscalar = |e| e[0] ^ e[1];
);
vector_model!(vga3d, "vga3d",
    2: Vector = |e| e[0];
    3: Bivector = |e| e[0] ^ e[1];
    4: Pseudoscalar = |e| e[0] ^ e[1] ^ e[2];
);
vector_model!(sta, "sta",
    2: Vector = |e| e[0];
    3: Bivector = |e| e[0] ^ e[1];
    4: Trivector = |e| e[0] ^ e[1] ^ e[2];
    5: Pseudoscalar = |e| e[0] ^ e[1] ^ e[2] ^ e[3];
);

/// The PGA models: positions are points of weight one, measures joins. The full-dimensional
/// join is a scalar, `$sign` times the VGA measure's pseudoscalar coefficient.
macro_rules! point_model {
    ($alg:ident, $feature:literal, $($n:literal: $Blade:ident = |$v:ident| $join:expr;)+) => {
        #[cfg(feature = $feature)]
        const _: () = {
            use crate::$alg::{Point, Scalar};

            impl<T: Coef> Position for Point<(), T> {
                type Coef = T;
                #[inline]
                fn affine(points: &[Self], weights: &[T]) -> Self {
                    combination(points, weights)
                }
            }

            impl<T: Coef> Measure<1> for Point<(), T> {
                type Blade = Scalar<(), T>;
                #[inline]
                fn measure(_: &[Self; 1]) -> Scalar<(), T> {
                    Scalar::new(T::one())
                }
            }

            $(
                impl<T: Coef> Measure<$n> for Point<(), T> {
                    type Blade = crate::$alg::intrinsic::$Blade<(), T>;
                    #[inline]
                    fn measure($v: &[Self; $n]) -> Self::Blade {
                        ($join) * inverse_factorial::<T>($n - 1)
                    }
                }
            )+
        };
    };
}

point_model!(pga2d, "pga2d",
    2: Line = |v| v[0] & v[1];
    3: Scalar = |v| v[0] & v[1] & v[2];
);
point_model!(pga3d, "pga3d",
    2: Line = |v| v[0] & v[1];
    3: Plane = |v| v[0] & v[1] & v[2];
    4: Scalar = |v| v[0] & v[1] & v[2] & v[3];
);

/// The CGA models: positions are round points of weight one (`eo` coefficient 1, at index
/// `$o`), measures flats `P₀ ∧ … ∧ P_k ∧ e∞`.
macro_rules! conformal_model {
    ($alg:ident, $feature:literal, $dim:literal, $($n:literal: $Blade:ident = |$v:ident, $ei:ident| $flat:expr;)+) => {
        #[cfg(feature = $feature)]
        const _: () = {
            use crate::$alg::{Scalar, Vector};

            impl<T: Coef> Position for Vector<(), T> {
                type Coef = T;
                /// The round point of the affine combination of the Euclidean points.
                #[inline]
                fn affine(points: &[Self], weights: &[T]) -> Self {
                    let x: [T; $dim] = core::array::from_fn(|i| {
                        points.iter().zip(weights).fold(T::zero(), |a, (p, &w)| a + p.c[i] * w)
                    });
                    let sq = x.iter().fold(T::zero(), |a, &c| a + c * c);
                    Self::from_coeffs(core::array::from_fn(|i| match i {
                        i if i < $dim => x[i],
                        $dim => T::one(),
                        _ => sq * T::from_ratio(1, 2),
                    }))
                }
            }

            impl<T: Coef> Measure<1> for Vector<(), T> {
                type Blade = Scalar<(), T>;
                #[inline]
                fn measure(_: &[Self; 1]) -> Scalar<(), T> {
                    Scalar::new(T::one())
                }
            }

            $(
                impl<T: Coef> Measure<$n> for Vector<(), T> {
                    type Blade = crate::$alg::$Blade<(), T>;
                    #[inline]
                    fn measure($v: &[Self; $n]) -> Self::Blade {
                        let $ei = Vector::from_coeffs(core::array::from_fn(|i| {
                            if i == $dim + 1 { T::one() } else { T::zero() }
                        }));
                        ($flat) * inverse_factorial::<T>($n - 1)
                    }
                }
            )+
        };
    };
}

conformal_model!(cga2d, "cga2d", 2,
    2: Trivector = |v, ei| v[0] ^ v[1] ^ ei;
    3: Pseudoscalar = |v, ei| v[0] ^ v[1] ^ v[2] ^ ei;
);
conformal_model!(cga3d, "cga3d", 3,
    2: Trivector = |v, ei| v[0] ^ v[1] ^ ei;
    3: Quadvector = |v, ei| v[0] ^ v[1] ^ v[2] ^ ei;
    4: Pseudoscalar = |v, ei| -(v[0] ^ v[1] ^ v[2] ^ v[3] ^ ei);
);

/// Squared distances, for the metric of meshes (`gax::dec`): VGA's `|a − b|²`, PGA's squared norm
/// of the direction between two points of weight one, CGA's `−2 A ∗ B` between round points of
/// weight one.
macro_rules! distance {
    ($alg:ident, $feature:literal, $P:ident, |$a:ident, $b:ident| $d2:expr) => {
        #[cfg(feature = $feature)]
        impl<T: Coef> crate::dec::Distance for crate::$alg::$P<(), T> {
            #[inline]
            fn distance_squared($a: &Self, $b: &Self) -> T {
                $d2
            }
        }
    };
}

distance!(vga2d, "vga2d", Vector, |a, b| {
    let d = *a - *b;
    d.c[0] * d.c[0] + d.c[1] * d.c[1]
});
distance!(vga3d, "vga3d", Vector, |a, b| {
    let d = *a - *b;
    d.c[0] * d.c[0] + d.c[1] * d.c[1] + d.c[2] * d.c[2]
});
distance!(pga2d, "pga2d", Point, |a, b| {
    let d = *a - *b; // a direction: its weight is zero
    d.c[0] * d.c[0] + d.c[1] * d.c[1]
});
distance!(pga3d, "pga3d", Point, |a, b| {
    let d = *a - *b;
    d.c[0] * d.c[0] + d.c[1] * d.c[1] + d.c[2] * d.c[2]
});
distance!(cga2d, "cga2d", Vector, |a, b| {
    let s = a.scalar_product(*b).c[0];
    -(s + s)
});
distance!(cga3d, "cga3d", Vector, |a, b| {
    let s = a.scalar_product(*b).c[0];
    -(s + s)
});
