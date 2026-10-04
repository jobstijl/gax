//! Light as geometry.
//!
//! Grassmann's laws of colour mixing (1853) are projective geometry: a light is a homogeneous
//! point in linear RGB space, a PGA3D `Point` `(R, G, B, W)`, whose position `(R/W, G/W, B/W)`
//! is its colour and whose weight `W` is its intensity. Everything the game does with colour
//! is then an operation on points:
//!
//! * two lights add as points: the sum lies between them, weighted by their intensities, with
//!   their intensities summed, which is what additive glow does;
//! * fading scales the weight and keeps the point;
//! * whitening moves the point towards white along the line to it.
//!
//! Adding is physics, so it happens in linear RGB. Judging colours is perception, so hue,
//! saturation and gradients happen in OkLab (Björn Ottosson's perceptual space), where equal
//! steps look equal. A light's *tint* is the same kind of point there: its OkLab position
//! `(L, a, b)` with its intensity as the weight. The way in is a linear map (a gax point map
//! to cone responses), a cube root per coordinate, and another linear map; the way out
//! reverses them. In OkLab:
//!
//! * lightness is an axis, the line through black along `L`;
//! * a hue shift is a rotation about that axis, a motor, which keeps lightness and chroma;
//! * desaturation moves towards the tint's foot on the axis, `(t | axis) ^ axis`;
//! * a gradient is an affine combination of tints.
//!
//! The raw coordinates `(R, G, B)` are chromaticity times intensity: radiance. Shown on a
//! display, radiance goes through [`agx`], a tonemapper whose matrices are point maps too.
//!
//! The functions that a renderer may trace into shaders ([`mix`], [`whiten`], [`fade`], [`luma`],
//! [`agx`]) are generic over gax's `Real`; the rest work in `f32`.

use gax::Real;
use gax::pga3d::{Line, Motor, Plane, Point};
use std::sync::LazyLock;

/// A light: a homogeneous point in linear RGB space.
pub type Light<T = f32> = Point<(), T>;

/// The light of colour `(r, g, b)` at intensity `i`.
pub const fn light(r: f32, g: f32, b: f32, i: f32) -> Light {
    Point::new(r * i, g * i, b * i, i)
}

/// The light of a colour given in sRGB components (as colour pickers give them) at intensity
/// `i`: each component through the sRGB transfer curve to linear light.
pub fn srgb(r: f32, g: f32, b: f32, i: f32) -> Light {
    let linear = |c: f32| {
        if c <= 0.04045 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        }
    };
    light(linear(r), linear(g), linear(b), i)
}

/// The light of a colour given as an sRGB hex code `0xRRGGBB`, at intensity `i`.
pub fn hex(code: u32, i: f32) -> Light {
    let [r, g, b] = [16, 8, 0].map(|s| ((code >> s) & 0xff) as f32 / 255.0);
    srgb(r, g, b, i)
}

/// No light.
pub const DARK: Light = Point::new(0.0, 0.0, 0.0, 0.0);

/// The intensity: the weight.
pub fn intensity(l: Light) -> f32 {
    l.e123()
}

/// The additive mix `(1 - t) a + t b`.
pub fn mix<T: Real>(a: Light<T>, b: Light<T>, t: T) -> Light<T> {
    a * (T::one() - t) + b * t
}

/// `l` moved towards white by `t`, at the same intensity: mixed with white light of its own
/// weight.
pub fn whiten<T: Real>(l: Light<T>, t: T) -> Light<T> {
    let w = l.e123();
    mix(l, Light::new(w, w, w, w), t)
}

/// `l` at `k` times its intensity: the same point, a different weight.
pub fn fade<T: Real>(l: Light<T>, k: T) -> Light<T> {
    l * k
}

/// The luminance of a light: its pairing with the plane of Rec. 709's weights (`plane & light`,
/// `0.2126 R + 0.7152 G + 0.0722 B`); lights of equal luminance lie on planes parallel to it.
pub fn luma<T: Real>(l: Light<T>) -> T {
    let w = |x: f64| T::from_f64(x);
    let plane = Plane::new(w(0.2126), w(0.7152), w(0.0722), T::zero());
    (plane & l).s()
}

/// A linear map of colour, as a gax map on light points (the weight kept): `m` holds the
/// matrix's rows. Its inverse is the map's own `inverse()`.
pub fn colour_map<T: Real>(m: [[f64; 3]; 3]) -> Point<(Point,), T> {
    let w = |x: f64| T::from_f64(x);
    let (z, o) = (T::zero(), T::one());
    Point::from_coeffs([
        [w(m[0][0]), w(m[0][1]), w(m[0][2]), z],
        [w(m[1][0]), w(m[1][1]), w(m[1][2]), z],
        [w(m[2][0]), w(m[2][1]), w(m[2][2]), z],
        [z, z, z, o],
    ])
}

/// A function applied to each colour coordinate of a light (the weight kept).
fn each<T: Real>(l: Light<T>, f: impl Fn(T) -> T) -> Light<T> {
    Light::new(f(l.e032()), f(l.e013()), f(l.e021()), l.e123())
}

/// AgX tonemapping (Troy Sobotka's, in Benjamin Wrensch's minimal fitted form) of a light
/// whose coordinates are radiance, to display light: the inset map, a log encoding and a
/// contrast curve per coordinate, a look that moves away from the grey of the same luminance
/// by `saturation` (an affine combination of lights), the outset map, and the display's
/// 2.2 gamma.
pub fn agx<T: Real>(l: Light<T>, saturation: T) -> Light<T> {
    let (min_ev, max_ev) = (T::from_f64(-12.47393), T::from_f64(4.026069));
    let inset = colour_map::<f64>([
        [0.842479062253094, 0.0784335999999992, 0.0792237451477643],
        [0.0423282422610123, 0.878468636469772, 0.0791661274605434],
        [0.0423756549057051, 0.0784336, 0.879142973793104],
    ]);
    // The outset is the inverse of the inset: both constants, computed in f64.
    let (inset, outset) = (
        inset.map_coefs(T::from_f64),
        inset.inverse().map_coefs(T::from_f64),
    );
    let log2 = T::from_f64(core::f64::consts::LOG2_E);
    let v = each(inset.of(l), |x| {
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
    // The look: away from the grey light of the same luminance.
    let y = luma(v);
    let grey = Light::new(y, y, y, v.e123());
    let v = mix(grey, v, saturation);
    let v = outset.of(v);
    each(v, |x| {
        (T::from_f64(2.2) * x.max(T::from_f64(1e-10)).ln()).exp()
    })
}

/// A light in OkLab: the point `(L, a, b)` with the light's intensity as its weight.
pub type Tint = Point<(), f32>;

/// Linear sRGB to cone responses (LMS), and the cube-rooted responses to `(L, a, b)`:
/// Ottosson's `M1` and `M2`.
const TO_LMS: [[f64; 3]; 3] = [
    [0.412_221_470_8, 0.536_332_536_3, 0.051_445_992_9],
    [0.211_903_498_2, 0.680_699_545_1, 0.107_396_956_6],
    [0.088_302_461_9, 0.281_718_837_6, 0.629_978_700_5],
];
const TO_LAB: [[f64; 3]; 3] = [
    [0.210_454_255_3, 0.793_617_785_0, -0.004_072_046_8],
    [1.977_998_495_1, -2.428_592_205_0, 0.450_593_709_9],
    [0.025_904_037_1, 0.782_771_766_2, -0.808_675_766_0],
];

/// OkLab's two linear maps, and their inverses for the way out, built once.
struct Oklab {
    to_lms: Point<(Point,), f32>,
    to_lab: Point<(Point,), f32>,
    from_lab: Point<(Point,), f32>,
    from_lms: Point<(Point,), f32>,
}

static OKLAB: LazyLock<Oklab> = LazyLock::new(|| {
    let (to_lms, to_lab) = (colour_map(TO_LMS), colour_map(TO_LAB));
    Oklab {
        to_lms,
        to_lab,
        from_lab: to_lab.inverse(),
        from_lms: to_lms.inverse(),
    }
});

/// The tint of a light (no light: the tint of black, weightless).
pub fn tint(l: Light) -> Tint {
    let w = l.e123();
    if w <= 0.0 {
        return DARK;
    }
    let lms = OKLAB.to_lms.of(l).unitized();
    OKLAB.to_lab.of(each(lms, f32::cbrt)) * w
}

/// The light of a tint.
pub fn untint(t: Tint) -> Light {
    let w = t.e123();
    if w <= 0.0 {
        return DARK;
    }
    let lms = each(OKLAB.from_lab.of(t.unitized()), |x| x * x * x);
    // Outside the display's gamut a coordinate can go below zero: clipped, since negative
    // light would take light away where lines add up.
    each(OKLAB.from_lms.of(lms), |x| x.max(0.0)) * w
}

/// OkLab's lightness axis: black along `L`.
pub fn lightness_axis() -> Line<(), f32> {
    Point::xyz(0.0, 0.0, 0.0) & Point::direction(1.0, 0.0, 0.0)
}

/// `l` with its hue turned by `angle`: a rotation of its tint about the lightness axis, which
/// keeps lightness, chroma and intensity.
pub fn hue_shift(l: Light, angle: f32) -> Light {
    untint(Motor::rotation(lightness_axis(), angle) >> tint(l))
}

/// `l` moved towards the grey of the same lightness by `t`: its tint towards its foot on the
/// lightness axis.
pub fn desaturate(l: Light, t: f32) -> Light {
    let axis = lightness_axis();
    let c = tint(l);
    // The foot: the meet of the axis with the plane through the tint orthogonal to it.
    let foot = ((c | axis) ^ axis).unitized() * c.e123();
    untint(c + (foot - c) * t)
}

/// A perceptual gradient from `a` to `b` at `t`: the tints' positions and the intensities
/// both interpolated (an affine combination of the unit tints, weighted back).
pub fn blend(a: Light, b: Light, t: f32) -> Light {
    let (ta, tb) = (tint(a), tint(b));
    let (wa, wb) = (ta.e123(), tb.e123());
    let (ua, ub) = (ta.unitized(), tb.unitized());
    untint((ua + (ub - ua) * t) * (wa + (wb - wa) * t))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rgb(l: Light) -> [f32; 3] {
        l.to_euclidean()
    }

    fn close(a: [f32; 3], b: [f32; 3]) -> bool {
        a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-5)
    }

    /// Adding lights is additive mixing: the colour is the intensity-weighted mean, the
    /// intensity the sum.
    #[test]
    fn lights_add_like_light() {
        let red = light(1.0, 0.0, 0.0, 3.0);
        let blue = light(0.0, 0.0, 1.0, 1.0);
        let m = red + blue;
        assert!((intensity(m) - 4.0).abs() < 1e-6);
        assert!(close(rgb(m), [0.75, 0.0, 0.25]));
        // Radiance is the raw coordinates: the sum of the radiances.
        assert!((m.e032() - 3.0).abs() < 1e-6 && (m.e021() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn fading_keeps_the_colour_and_whitening_keeps_the_intensity() {
        let c = light(0.2, 0.6, 1.0, 2.0);
        assert!(close(rgb(fade(c, 0.25)), rgb(c)));
        assert!((intensity(fade(c, 0.25)) - 0.5).abs() < 1e-6);
        let w = whiten(c, 1.0);
        assert!(close(rgb(w), [1.0, 1.0, 1.0]) && (intensity(w) - 2.0).abs() < 1e-6);
    }

    fn near(a: [f32; 3], b: [f32; 3], eps: f32) -> bool {
        a.iter().zip(b).all(|(x, y)| (x - y).abs() < eps)
    }

    /// OkLab's reference values: white is `L = 1` on the axis, and a tint turns back into its
    /// light.
    #[test]
    fn tints_are_oklab() {
        let w = tint(light(1.0, 1.0, 1.0, 3.0));
        assert!(
            near(w.to_euclidean(), [1.0, 0.0, 0.0], 1e-4),
            "{:?}",
            w.to_euclidean()
        );
        assert!((w.e123() - 3.0).abs() < 1e-6);
        // sRGB red: L 0.628, a 0.225, b 0.126.
        let r = tint(light(1.0, 0.0, 0.0, 1.0)).to_euclidean();
        assert!(near(r, [0.628, 0.225, 0.126], 1e-3), "{r:?}");
        for c in [[0.2, 0.6, 1.0], [1.0, 0.36, 0.1], [0.05, 0.9, 0.4]] {
            let l = light(c[0], c[1], c[2], 1.7);
            let back = untint(tint(l));
            assert!(near(rgb(back), c, 1e-4) && (intensity(back) - 1.7).abs() < 1e-5);
        }
    }

    /// A hue shift turns the tint about the lightness axis: lightness, chroma and intensity
    /// stay, grey stays grey, and half a turn is the opposite hue.
    #[test]
    fn hue_is_an_angle_about_the_lightness_axis() {
        let c = light(0.2, 0.6, 1.0, 2.0);
        let t = tint(c).to_euclidean();
        let s = hue_shift(c, 0.8);
        let u = tint(s).to_euclidean();
        assert!((u[0] - t[0]).abs() < 1e-4);
        let chroma = |v: [f32; 3]| Point::direction(v[1], v[2], 0.0).ideal_norm();
        assert!((chroma(u) - chroma(t)).abs() < 1e-4);
        assert!((intensity(s) - 2.0).abs() < 1e-5);
        let grey = light(0.4, 0.4, 0.4, 1.0);
        assert!(near(rgb(hue_shift(grey, 1.0)), [0.4, 0.4, 0.4], 1e-4));
        let h = tint(hue_shift(c, core::f32::consts::PI)).to_euclidean();
        assert!((h[1] + t[1]).abs() < 1e-3 && (h[2] + t[2]).abs() < 1e-3);
    }

    /// Desaturated all the way, a colour is the grey of its own lightness.
    #[test]
    fn desaturation_moves_to_the_lightness_axis() {
        let c = light(1.0, 0.5, 0.0, 2.0);
        let g = desaturate(c, 1.0);
        let [r, gg, b] = rgb(g);
        assert!(
            (r - gg).abs() < 1e-4 && (gg - b).abs() < 1e-4,
            "{:?}",
            rgb(g)
        );
        assert!((tint(g).to_euclidean()[0] - tint(c).to_euclidean()[0]).abs() < 1e-4);
        assert!((intensity(g) - 2.0).abs() < 1e-5);
    }

    /// A gradient's middle is perceptually between its ends: its lightness is the mean.
    #[test]
    fn gradients_are_even_in_lightness() {
        let (a, b) = (light(0.1, 0.2, 1.0, 1.0), light(1.0, 0.9, 0.1, 3.0));
        let m = blend(a, b, 0.5);
        let l = |x: Light| tint(x).to_euclidean()[0];
        assert!((l(m) - 0.5 * (l(a) + l(b))).abs() < 1e-4);
        assert!((intensity(m) - 2.0).abs() < 1e-5);
    }
}
