//! The operations a renderer traces into its shaders, on bare points and generic over gax's
//! `Real` (so that they also run on symbolic coefficients): additive mixing, whitening, fading,
//! luminance and AgX tonemapping, with the colour maps they are built from. [`crate::Colour`]
//! uses the same functions on `f32`.

use gax::Real;
use gax::pga3d::{Plane, Point};

/// A colour as a bare homogeneous point.
pub type Raw<T = f32> = Point<(), T>;

/// A map of colour points: a gax map on points.
pub type Map<T = f32> = Point<(Point,), T>;

/// A linear map of colour coordinates, as a gax map on points that keeps the weight: `m` holds
/// the matrix's rows.
pub fn colour_map<T: Real>(m: [[f64; 3]; 3]) -> Point<(Point,), T> {
    affine_map(m, [0.0; 3])
}

/// An affine map of colour coordinates, `x ↦ m x + t`, as a gax map on points (for a point of
/// weight `w`, the translation is taken `w` times, as a homogeneous map does).
pub fn affine_map<T: Real>(m: [[f64; 3]; 3], t: [f64; 3]) -> Point<(Point,), T> {
    let w = |x: f64| T::from_f64(x);
    let (z, o) = (T::zero(), T::one());
    Point::from_coeffs([
        [w(m[0][0]), w(m[0][1]), w(m[0][2]), w(t[0])],
        [w(m[1][0]), w(m[1][1]), w(m[1][2]), w(t[1])],
        [w(m[2][0]), w(m[2][1]), w(m[2][2]), w(t[2])],
        [z, z, z, o],
    ])
}

/// A function of one coordinate applied to each colour coordinate of a point (the weight kept):
/// the transfer curves, which are functions of a single number by definition.
pub fn each<T: Real>(p: Raw<T>, f: impl Fn(T) -> T) -> Raw<T> {
    Raw::new(f(p.e032()), f(p.e013()), f(p.e021()), p.e123())
}

/// The affine (for lights, additive) mix `(1 - t) a + t b` of two points.
pub fn mix<T: Real>(a: Raw<T>, b: Raw<T>, t: T) -> Raw<T> {
    a * (T::one() - t) + b * t
}

/// `p` moved towards white by `t` at the same weight: mixed with the white of its own weight.
pub fn whiten<T: Real>(p: Raw<T>, t: T) -> Raw<T> {
    let w = p.e123();
    mix(p, Raw::new(w, w, w, w), t)
}

/// `p` at `k` times its weight: the same colour, more or less of it.
pub fn fade<T: Real>(p: Raw<T>, k: T) -> Raw<T> {
    p * k
}

/// The luminance of a linear RGB point, its weight included: its pairing with the plane of
/// Rec. 709's weights, `0.2126 R + 0.7152 G + 0.0722 B` (the `Y` of CIE XYZ). Colours of equal
/// luminance lie on planes parallel to it.
pub fn luma<T: Real>(p: Raw<T>) -> T {
    let w = |x: f64| T::from_f64(x);
    let plane = Plane::new(w(0.2126), w(0.7152), w(0.0722), T::zero());
    (plane & p).s()
}

/// AgX tonemapping (Troy Sobotka's, in Benjamin Wrensch's minimal fitted form) of linear RGB
/// whose coordinates are radiance, to linear display light: the inset map, a log encoding and a
/// contrast curve per coordinate, a look that moves away from the grey of the same luminance
/// by `saturation` (an affine combination of points), the outset map (the inset's inverse),
/// and the display's 2.2 gamma.
pub fn agx<T: Real>(p: Raw<T>, saturation: T) -> Raw<T> {
    let (min_ev, max_ev) = (T::from_f64(-12.47393), T::from_f64(4.026069));
    let inset = colour_map::<f64>([
        [0.842479062253094, 0.0784335999999992, 0.0792237451477643],
        [0.0423282422610123, 0.878468636469772, 0.0791661274605434],
        [0.0423756549057051, 0.0784336, 0.879142973793104],
    ]);
    // Both constants, computed in f64.
    let (inset, outset) = (
        inset.map_coefs(T::from_f64),
        inset.inverse().map_coefs(T::from_f64),
    );
    let log2 = T::from_f64(core::f64::consts::LOG2_E);
    let v = each(inset.of(p), |x| {
        let e = (x.max(T::from_f64(1e-10)).ln() * log2)
            .max(min_ev)
            .min(max_ev);
        let x = (e - min_ev) / (max_ev - min_ev);
        // The contrast curve.
        let c = |k: f64| T::from_f64(k);
        let (x2, x4) = (x * x, x * x * x * x);
        c(15.5) * x4 * x2 - c(40.14) * x4 * x + c(31.96) * x4 - c(6.868) * x2 * x
            + c(0.4298) * x2
            + c(0.1191) * x
            - c(0.00232)
    });
    // The look: away from the grey of the same luminance.
    let y = luma(v);
    let grey = Raw::new(y, y, y, v.e123());
    let v = outset.of(mix(grey, v, saturation));
    each(v, |x| {
        (T::from_f64(2.2) * x.max(T::from_f64(1e-10)).ln()).exp()
    })
}
