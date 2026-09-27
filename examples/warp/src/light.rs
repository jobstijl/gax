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
//! The raw coordinates `(R, G, B)` are chromaticity times intensity: radiance. The GPU takes a
//! light as it is (a `PointGpu`) and emits its first three coefficients.

use gax::pga3d::{Line, Motor, Point};

/// A light: a homogeneous point in linear RGB space.
pub type Light = Point<(), f32>;

/// The light of colour `(r, g, b)` at intensity `i`.
pub const fn light(r: f32, g: f32, b: f32, i: f32) -> Light {
    Point::new(r * i, g * i, b * i, i)
}

/// No light.
pub const DARK: Light = Point::new(0.0, 0.0, 0.0, 0.0);

/// The intensity: the weight.
pub fn intensity(l: Light) -> f32 {
    l.e123()
}

/// `l` at `k` times its intensity.
pub fn fade(l: Light, k: f32) -> Light {
    crate::light_fade(l, k)
}

/// `l` moved towards white by `t`, at the same intensity.
pub fn whiten(l: Light, t: f32) -> Light {
    crate::light_whiten(l, t)
}

/// A light in OkLab: the point `(L, a, b)` with the light's intensity as its weight.
pub type Tint = Point<(), f32>;

/// A linear map of colour as a gax map on points (the weight kept): `m` holds the rows.
fn colour_map(m: [[f32; 3]; 3]) -> Point<(Point,), f32> {
    Point::from_coeffs([
        [m[0][0], m[0][1], m[0][2], 0.0],
        [m[1][0], m[1][1], m[1][2], 0.0],
        [m[2][0], m[2][1], m[2][2], 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
}

/// Linear sRGB to cone responses (LMS), and the cube-rooted responses to `(L, a, b)`.
const TO_LMS: [[f32; 3]; 3] = [
    [0.412_221_46, 0.536_332_5, 0.051_445_995],
    [0.211_903_5, 0.680_699_5, 0.107_396_96],
    [0.088_302_46, 0.281_718_85, 0.629_978_7],
];
const TO_LAB: [[f32; 3]; 3] = [
    [0.210_454_26, 0.793_617_8, -0.004_072_047],
    [1.977_998_5, -2.428_592_2, 0.450_593_7],
    [0.025_904_037, 0.782_771_77, -0.808_675_77],
];
const FROM_LAB: [[f32; 3]; 3] = [
    [1.0, 0.396_337_78, 0.215_803_76],
    [1.0, -0.105_561_346, -0.063_854_17],
    [1.0, -0.089_484_18, -1.291_485_5],
];
const FROM_LMS: [[f32; 3]; 3] = [
    [4.076_741_7, -3.307_711_6, 0.230_969_94],
    [-1.268_438, 2.609_757_4, -0.341_319_38],
    [-0.004_196_086_3, -0.703_418_6, 1.707_614_7],
];

/// `f` on each coordinate of a point of weight 1.
fn each(p: Point<(), f32>, f: impl Fn(f32) -> f32) -> Point<(), f32> {
    let [x, y, z] = p.to_euclidean();
    Point::xyz(f(x), f(y), f(z))
}

/// The tint of a light (no light: the tint of black, weightless).
pub fn tint(l: Light) -> Tint {
    let w = l.e123();
    if w <= 0.0 {
        return DARK;
    }
    let lms = colour_map(TO_LMS).of(l).unitized();
    colour_map(TO_LAB).of(each(lms, f32::cbrt)) * w
}

/// The light of a tint.
pub fn untint(t: Tint) -> Light {
    let w = t.e123();
    if w <= 0.0 {
        return DARK;
    }
    let lms = each(colour_map(FROM_LAB).of(t.unitized()), |x| x * x * x);
    // Outside the display's gamut a coordinate can go below zero: clipped, since negative
    // light would take light away where lines add up.
    each(colour_map(FROM_LMS).of(lms), |x| x.max(0.0)) * w
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
/// lightness axis, the meet of the axis with the plane through the tint orthogonal to it.
pub fn desaturate(l: Light, t: f32) -> Light {
    let axis = lightness_axis();
    let c = tint(l);
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

/// Gravitational redshift by `z` (`0` none): the spectrum slides towards red, blue through
/// green and yellow, which in OkLab is a turn of the hue about the lightness axis (towards
/// lower hue angles), and the light dims as `1 / (1 + z)²`.
pub fn redshift(l: Light, z: f32) -> Light {
    let k = 1.0 / (1.0 + z);
    fade(hue_shift(l, -1.4 * z / (1.0 + z)), k * k)
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

    /// Redshift slides blue towards green (the way to red) and dims it; a blueshift goes the
    /// other way and brightens.
    #[test]
    fn redshift_slides_blue_towards_red() {
        let blue = light(0.2, 0.3, 1.0, 2.0);
        let s = redshift(blue, 0.8);
        let [r, g, b] = rgb(s);
        assert!(g > 0.3 && b < 1.0 && r >= 0.0, "{:?}", rgb(s));
        assert!(intensity(s) < 1.0);
        assert!(intensity(redshift(blue, -0.3)) > 2.0);
    }
}
