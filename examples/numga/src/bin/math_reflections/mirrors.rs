//! The mirrors of a triangle whose angles are a half turn divided by 2, 3 and `m`, in the
//! algebra its angle sum asks for.
//!
//! The first two mirror vectors are `x` and `x` turned in the `xy` plane, so that both point
//! out of the triangle and pair to `-cos(π/2)`. The third has a part in their plane, from its
//! pairings `-cos(π/m)` with them through the plane's reciprocal vectors (`second | plane⁻¹`
//! pairs to one with the first and to zero with the second, `plane⁻¹ | first` the other way
//! round), and a part along the plane's normal, the third basis vector. For the third to have
//! unit length, the normal must square to plus one where the part in the plane is shorter than
//! one, to zero where it has length one, and to minus one where it is longer: the sphere
//! `x+y+z+` (VGA3D), the flat plane `x+y+w0`, the hyperbolic plane `x+y+t-`. On the flat plane
//! the normal adds no length, and its part sets the size of the tiles.
//!
//! A point is a bivector, the dual of a direction. Its side of a mirror is the sign of its
//! regressive product with the mirror vector, which needs no metric, and its reflection is the
//! sandwich with the mirror vector (a point, a dual, is intrinsic, and `>>` reflects it so):
//! the same in the three geometries. The view is the unit disk mapped conformally,
//! stereographically onto the sphere, to scale onto the plane, and as the Poincaré disk onto
//! the hyperbolic plane: three cases of one formula in the square of the normal. One rotor, in the plane of `x` and the normal, turns the sphere, shifts the
//! plane and boosts the hyperbolic plane. The code is written once and stamped out per algebra
//! by a macro, as the kinds of the three algebras are different types.

use gax_numga_examples::{measure::roots, signal::phasor};

/// The angles at which the mirrors meet: the first and the second, and the first and the
/// third. The second and the third meet at a half turn over `m`.
pub const FIRST_SECOND: f64 = core::f64::consts::FRAC_PI_2;
pub const FIRST_THIRD: f64 = core::f64::consts::FRAC_PI_3;
/// On the flat plane, the third mirror's offset along the normal, which sets the tiles' size.
pub const FLAT_OFFSET: f64 = 0.5;

/// The cosine of an angle, the reach across of its phasor.
pub fn cos(angle: f64) -> f64 {
    phasor(angle).e20()
}

/// The sign of what the normal must square to for a triangle with angles a half turn over 2, 3
/// and `m`: the sign of the angle sum's excess over a half turn.
pub fn curvature(m: f64) -> i32 {
    let excess = FIRST_SECOND + FIRST_THIRD + core::f64::consts::PI / m - core::f64::consts::PI;
    if excess.abs() < 1e-12 {
        0
    } else {
        excess.signum() as i32
    }
}

/// The third mirror's part along the normal, given the square of its part in the plane: what
/// the unit length leaves, `√((1 - in_plane²) / curvature)` (half the width between the roots
/// of `h² - (1 - in_plane²) / curvature`), and on the flat plane the offset.
pub fn height(in_plane_squared: f64, curvature: i32) -> f64 {
    if curvature == 0 {
        return FLAT_OFFSET;
    }
    let square = (1.0 - in_plane_squared) / f64::from(curvature);
    roots(1.0, 0.0, -square).expect("a real height").1
}

gax::algebra! {
    algebra flat_ga "numga's `x+y+w0`: the flat plane's algebra, the normal to `x` and `y` null.";
    basis e1 = 1, e2 = 1, ew = 0;
    kind Scalar = [1];
    versor Vector = [e1, e2, ew];
    kind Bivector = [e12, e1w, e2w];
    versor Rotor = [1, e12, e1w, e2w];
}

gax::algebra! {
    algebra hyperbolic_ga "numga's `x+y+t-`: the hyperbolic plane's algebra, the normal squaring to minus one.";
    basis e1 = 1, e2 = 1, et = -1;
    kind Scalar = [1];
    versor Vector = [e1, e2, et];
    kind Bivector = [e12, e1t, e2t];
    versor Rotor = [1, e12, e1t, e2t];
}

/// The geometry of one algebra: its mirrors, its view of the disk, its motion along `x`, and
/// the side and reflection of a point.
macro_rules! geometry {
    ($name:ident, $algebra:path, $curvature:expr, $doc:literal) => {
        #[doc = $doc]
        pub mod $name {
            use super::{FIRST_SECOND, FIRST_THIRD, cos, height};
            use $algebra::{Bivector, Rotor, Vector};

            pub type V = Vector<(), f64>;
            /// A point: a dual of a direction, so intrinsic (ADR-051).
            pub type B = gax::Oriented<Bivector, gax::Intrinsic>;
            pub type R = Rotor<(), f64>;
            /// What the normal to the first two mirrors squares to.
            pub const CURVATURE: i32 = $curvature;

            pub fn x() -> V {
                Vector::new(1.0, 0.0, 0.0)
            }

            pub fn y() -> V {
                Vector::new(0.0, 1.0, 0.0)
            }

            /// The normal to the plane of `x` and `y`: the third basis vector.
            pub fn normal() -> V {
                Vector::new(0.0, 0.0, 1.0)
            }

            /// The mirror vectors of the triangle whose second and third mirrors meet at a half
            /// turn over `m`.
            pub fn mirrors(m: f64) -> [V; 3] {
                let first = x();
                let quarter = core::f64::consts::PI - FIRST_SECOND;
                let second = (x() ^ y()).gp(-quarter / 2.0).exp() >> first;
                let inverse = (first ^ second).inverse();
                let in_plane = (second | inverse).gp(-cos(FIRST_THIRD))
                    - (inverse | first).gp(cos(core::f64::consts::PI / m));
                let third = in_plane + normal().gp(height(in_plane.norm_squared(), CURVATURE));
                [first, second, third]
            }

            /// The point of the geometry over `(u, v)` of the unit disk: its direction,
            /// conformally mapped, dualized.
            pub fn view(u: f64, v: f64) -> B {
                let lift = 1.0 - f64::from(CURVATURE) * (u * u + v * v);
                (x().gp(2.0 * u) + y().gp(2.0 * v) + normal().gp(lift)).dual()
            }

            /// The motion along `x` by `distance`: the rotor in the plane of `x` and the normal,
            /// which turns the sphere, shifts the plane and boosts the hyperbolic plane.
            pub fn motion(distance: f64) -> R {
                (x() ^ normal()).gp(-distance / 2.0).exp().into_inner()
            }

            /// A point's side of a mirror: their regressive product.
            pub fn side(mirror: V, point: B) -> f64 {
                (mirror & point).s()
            }

            /// A point reflected in a mirror.
            pub fn reflect(mirror: V, point: B) -> B {
                mirror >> point
            }
        }
    };
}

geometry!(sphere, gax::vga3d, 1, "The sphere, `x+y+z+`: VGA3D.");
geometry!(flat, super::flat_ga, 0, "The flat plane, `x+y+w0`.");
geometry!(
    hyperbolic,
    super::hyperbolic_ga,
    -1,
    "The hyperbolic plane, `x+y+t-`."
);

#[cfg(test)]
mod tests {
    use super::*;

    /// The three triangles fall in the three signatures, each algebra's normal squares to its
    /// curvature, and the plane of `x` and the normal squares to minus that, so the same rotor
    /// turns, shifts or boosts; the mirror vectors pair to `-cos` of the angle between their
    /// mirrors (numga's checks).
    #[test]
    fn the_angle_sum_picks_the_signature_and_the_mirrors_meet_at_their_angles() {
        assert_eq!([5.0, 6.0, 7.0].map(curvature), [1, 0, -1]);
        macro_rules! check {
            ($g:ident, $m:expr) => {{
                use $g::*;
                assert_eq!(curvature($m), CURVATURE);
                assert!((normal().norm_squared() - f64::from(CURVATURE)).abs() < 1e-12);
                let plane = x() ^ normal();
                let square = (plane * plane).s();
                assert!((square + f64::from(CURVATURE)).abs() < 1e-12);
                let [first, second, third] = mirrors($m);
                let angle = core::f64::consts::PI / $m;
                for (one, other, between) in [
                    (first, second, FIRST_SECOND),
                    (first, third, FIRST_THIRD),
                    (second, third, angle),
                ] {
                    assert!(((one | other).s() + cos(between)).abs() < 1e-12);
                }
            }};
        }
        check!(sphere, 5.0);
        check!(flat, 6.0);
        check!(hyperbolic, 7.0);
    }
}
