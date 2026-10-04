//! The colour spaces: each a space of homogeneous points, reached from linear sRGB (the hub) by
//! maps. The linear steps are gax point maps (linear sRGB to CIE XYZ to cone responses, the
//! affine steps of CIELAB and Oklab); the nonlinear ones are the spaces' transfer curves, each a
//! function of one coordinate applied to every coordinate.
//!
//! Every space also has a grey axis, the line from its black to its white, and a half-plane
//! through it where hue is zero: the cylindrical geometry of [`crate::Colour`] (hue, chroma,
//! lightness) is about that axis.

use crate::ops::{Map, Raw, affine_map, colour_map, each};
use core::fmt::Debug;

/// A colour space.
pub trait Space: Copy + Clone + Default + Debug + PartialEq + 'static {
    /// Its name.
    const NAME: &'static str;
    /// The point of this space for a point of linear sRGB (both of weight 1).
    fn from_linear(p: Raw) -> Raw;
    /// The point of linear sRGB for a point of this space (both of weight 1).
    fn to_linear(p: Raw) -> Raw;
    /// Black: one end of the grey axis.
    fn black() -> Raw;
    /// White: the other end.
    fn white() -> Raw;
    /// A point of the half-plane through the grey axis where hue is zero.
    fn hue_zero() -> Raw;
}

/// Linear sRGB: the coordinates are proportional to light (radiance), so lights add as points.
/// The renderer's space.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LinearRgb;

/// sRGB as stored and shown: linear sRGB through the display's transfer curve (a power of about
/// 1/2.2), so that equal steps of the coordinates look roughly equal.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Srgb;

/// CIE 1931 XYZ (D65 white): `Y` is luminance, and every other space is defined from it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Xyz;

/// CIELAB (D65 white), with lightness from 0 to 1 (CIE's `L*` over 100, and `a*`, `b*` alike):
/// the classic perceptual space.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Lab;

/// Oklab (Björn Ottosson's): a perceptual space in which equal steps look equal and hue keeps
/// under changes of lightness and chroma. Lightness from 0 to 1.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Oklab;

/// Linear sRGB to XYZ, and back (IEC 61966-2-1).
const RGB_TO_XYZ: [[f64; 3]; 3] = [
    [0.412_456_4, 0.357_576_1, 0.180_437_5],
    [0.212_672_9, 0.715_152_2, 0.072_175_0],
    [0.019_333_9, 0.119_192_0, 0.950_304_1],
];
const XYZ_TO_RGB: [[f64; 3]; 3] = [
    [3.240_454_2, -1.537_138_5, -0.498_531_4],
    [-0.969_266_0, 1.876_010_8, 0.041_556_0],
    [0.055_643_4, -0.204_025_9, 1.057_225_2],
];

/// The D65 white point in XYZ.
const D65: [f64; 3] = [0.950_47, 1.0, 1.088_83];

/// Linear sRGB to cone responses (LMS), the cube-rooted responses to `(L, a, b)`, and their
/// inverses: Ottosson's `M1`, `M2` and theirs.
const RGB_TO_LMS: [[f64; 3]; 3] = [
    [0.412_221_470_8, 0.536_332_536_3, 0.051_445_992_9],
    [0.211_903_498_2, 0.680_699_545_1, 0.107_396_956_6],
    [0.088_302_461_9, 0.281_718_837_6, 0.629_978_700_5],
];
const LMS_TO_OKLAB: [[f64; 3]; 3] = [
    [0.210_454_255_3, 0.793_617_785_0, -0.004_072_046_8],
    [1.977_998_495_1, -2.428_592_205_0, 0.450_593_709_9],
    [0.025_904_037_1, 0.782_771_766_2, -0.808_675_766_0],
];
const OKLAB_TO_LMS: [[f64; 3]; 3] = [
    [1.0, 0.396_337_777_4, 0.215_803_757_3],
    [1.0, -0.105_561_345_8, -0.063_854_172_8],
    [1.0, -0.089_484_177_5, -1.291_485_548_0],
];
const LMS_TO_RGB: [[f64; 3]; 3] = [
    [4.076_741_662_1, -3.307_711_591_3, 0.230_969_929_2],
    [-1.268_438_004_6, 2.609_757_401_1, -0.341_319_396_5],
    [-0.004_196_086_3, -0.703_418_614_7, 1.707_614_701_0],
];

/// CIELAB's steps, scaled so that lightness runs from 0 to 1: XYZ over the white point, the
/// curve `f`, then `L = 1.16 f(y) - 0.16`, `a = 5 (f(x) - f(y))`, `b = 2 (f(y) - f(z))`.
const LAB_FROM_F: [[f64; 3]; 3] = [[0.0, 1.16, 0.0], [5.0, -5.0, 0.0], [0.0, 2.0, -2.0]];
const LAB_FROM_F_SHIFT: [f64; 3] = [-0.16, 0.0, 0.0];
/// The inverse: `f(y) = (L + 0.16) / 1.16`, `f(x) = f(y) + a / 5`, `f(z) = f(y) - b / 2`.
const F_FROM_LAB: [[f64; 3]; 3] = [
    [1.0 / 1.16, 0.2, 0.0],
    [1.0 / 1.16, 0.0, 0.0],
    [1.0 / 1.16, 0.0, -0.5],
];
const F_FROM_LAB_SHIFT: [f64; 3] = [0.16 / 1.16, 0.16 / 1.16, 0.16 / 1.16];

/// The diagonal map dividing by (or multiplying with) the white point.
fn white_scale(inverse: bool) -> Map {
    let s = |k: usize| if inverse { D65[k] } else { 1.0 / D65[k] };
    colour_map([[s(0), 0.0, 0.0], [0.0, s(1), 0.0], [0.0, 0.0, s(2)]])
}

/// sRGB's transfer curve, linear light to the stored value: a line near black, then a power.
fn encode(c: f32) -> f32 {
    if c <= 0.003_130_8 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

/// Its inverse, the stored value to linear light.
fn decode(c: f32) -> f32 {
    if c <= 0.040_45 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// CIELAB's curve: a cube root, and a line near black where the root would be too steep.
fn lab_f(t: f32) -> f32 {
    const D: f32 = 6.0 / 29.0;
    if t > D * D * D {
        t.cbrt()
    } else {
        t / (3.0 * D * D) + 4.0 / 29.0
    }
}

/// Its inverse.
fn lab_f_inverse(f: f32) -> f32 {
    const D: f32 = 6.0 / 29.0;
    if f > D {
        f * f * f
    } else {
        3.0 * D * D * (f - 4.0 / 29.0)
    }
}

impl Space for LinearRgb {
    const NAME: &'static str = "linear sRGB";
    fn from_linear(p: Raw) -> Raw {
        p
    }
    fn to_linear(p: Raw) -> Raw {
        p
    }
    fn black() -> Raw {
        Raw::xyz(0.0, 0.0, 0.0)
    }
    fn white() -> Raw {
        Raw::xyz(1.0, 1.0, 1.0)
    }
    fn hue_zero() -> Raw {
        Raw::xyz(1.0, 0.0, 0.0)
    }
}

impl Space for Srgb {
    const NAME: &'static str = "sRGB";
    fn from_linear(p: Raw) -> Raw {
        each(p, encode)
    }
    fn to_linear(p: Raw) -> Raw {
        each(p, decode)
    }
    fn black() -> Raw {
        Raw::xyz(0.0, 0.0, 0.0)
    }
    fn white() -> Raw {
        Raw::xyz(1.0, 1.0, 1.0)
    }
    fn hue_zero() -> Raw {
        Raw::xyz(1.0, 0.0, 0.0)
    }
}

impl Space for Xyz {
    const NAME: &'static str = "CIE XYZ";
    fn from_linear(p: Raw) -> Raw {
        colour_map(RGB_TO_XYZ).of(p)
    }
    fn to_linear(p: Raw) -> Raw {
        colour_map(XYZ_TO_RGB).of(p)
    }
    fn black() -> Raw {
        Raw::xyz(0.0, 0.0, 0.0)
    }
    fn white() -> Raw {
        Raw::xyz(D65[0] as f32, D65[1] as f32, D65[2] as f32)
    }
    fn hue_zero() -> Raw {
        Xyz::from_linear(Raw::xyz(1.0, 0.0, 0.0))
    }
}

impl Space for Lab {
    const NAME: &'static str = "CIELAB";
    fn from_linear(p: Raw) -> Raw {
        let relative = white_scale(false).of(Xyz::from_linear(p));
        affine_map(LAB_FROM_F, LAB_FROM_F_SHIFT).of(each(relative, lab_f))
    }
    fn to_linear(p: Raw) -> Raw {
        let f = affine_map(F_FROM_LAB, F_FROM_LAB_SHIFT).of(p);
        Xyz::to_linear(white_scale(true).of(each(f, lab_f_inverse)))
    }
    fn black() -> Raw {
        Raw::xyz(0.0, 0.0, 0.0)
    }
    fn white() -> Raw {
        Raw::xyz(1.0, 0.0, 0.0)
    }
    fn hue_zero() -> Raw {
        Raw::xyz(0.5, 1.0, 0.0)
    }
}

impl Space for Oklab {
    const NAME: &'static str = "Oklab";
    fn from_linear(p: Raw) -> Raw {
        let lms = colour_map(RGB_TO_LMS).of(p);
        colour_map(LMS_TO_OKLAB).of(each(lms, f32::cbrt))
    }
    fn to_linear(p: Raw) -> Raw {
        let lms = each(colour_map(OKLAB_TO_LMS).of(p), |x| x * x * x);
        colour_map(LMS_TO_RGB).of(lms)
    }
    fn black() -> Raw {
        Raw::xyz(0.0, 0.0, 0.0)
    }
    fn white() -> Raw {
        Raw::xyz(1.0, 0.0, 0.0)
    }
    fn hue_zero() -> Raw {
        Raw::xyz(0.5, 1.0, 0.0)
    }
}
