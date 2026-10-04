//! HSV, HSL and HWB: the piecewise models of the sRGB cube that colour pickers use, exactly as
//! defined, constructed in the cube.
//!
//! * The value is how far along its ray from black a colour is, against where the ray leaves
//!   the cube through a far face (a meet of the ray with the faces `r = 1`, `g = 1`, `b = 1`):
//!   its largest channel.
//! * The whiteness is how far above the near faces it is: the ray from the colour away from
//!   white along the grey direction meets the faces `r = 0`, `g = 0`, `b = 0`, and the colour
//!   is that meet plus a grey: its smallest channel.
//! * The hue is where the colour, without its grey and dilated about black to full value, lies
//!   on the hexagon of the cube's six fully saturated edges (red to yellow to green to cyan to
//!   blue to magenta and back): the edge whose line it is on (a join of norm zero), and how far
//!   along it. Hue is in radians, a sixth of a turn per edge, red at zero.
//!
//! The hexagon is why HSV's hue is not the angle about the grey axis of [`crate::Colour::hue`],
//! though both turn the same way and agree at the primaries and secondaries.

use crate::colour::Srgb;
use crate::ops::Raw;
use gax::pga3d::{Plane, Point};

const TAU: f32 = core::f32::consts::TAU;

fn black() -> Raw {
    Point::xyz(0.0, 0.0, 0.0)
}
fn white() -> Raw {
    Point::xyz(1.0, 1.0, 1.0)
}

/// The corners of the hexagon in hue order.
fn hexagon() -> [Raw; 6] {
    [
        Point::xyz(1.0, 0.0, 0.0),
        Point::xyz(1.0, 1.0, 0.0),
        Point::xyz(0.0, 1.0, 0.0),
        Point::xyz(0.0, 1.0, 1.0),
        Point::xyz(0.0, 0.0, 1.0),
        Point::xyz(1.0, 0.0, 1.0),
    ]
}

/// The value of a point of the cube (weight 1): its largest channel, the ratio of its distance
/// from black to the distance at which its ray from black leaves the cube.
pub(crate) fn value(p: Raw) -> f32 {
    if (black() & p).norm() < 1e-9 {
        return 0.0;
    }
    let far = [
        Plane::new(1.0, 0.0, 0.0, -1.0),
        Plane::new(0.0, 1.0, 0.0, -1.0),
        Plane::new(0.0, 0.0, 1.0, -1.0),
    ];
    let ray = black() & p;
    // The nearest far face along the ray: the smallest distance from black.
    let to_p = (black() & p).norm();
    let exit = far
        .iter()
        .map(|plane| ray ^ *plane)
        .filter(|m| m.e123().abs() > 1e-9)
        .map(|m| (black() & m.unitized()).norm())
        .fold(f32::MAX, f32::min);
    to_p / exit
}

/// The whiteness of a point of the cube: its smallest channel, how far it lies along the grey
/// direction above the nearest of the near faces.
pub(crate) fn whiteness(p: Raw) -> f32 {
    let grey = white() - black();
    let line = p & (p - grey);
    let near = [
        Plane::new(1.0, 0.0, 0.0, 0.0),
        Plane::new(0.0, 1.0, 0.0, 0.0),
        Plane::new(0.0, 0.0, 1.0, 0.0),
    ];
    let foot = near
        .iter()
        .map(|plane| (line ^ *plane).unitized())
        .min_by(|a, b| (p & *a).norm().total_cmp(&(p & *b).norm()))
        .expect("three faces");
    // Signed: below the faces (a channel under zero) is negative whiteness.
    let below = near.iter().any(|plane| (*plane & p).s() < 0.0);
    let w = (p & foot).norm() / (black() & white()).norm();
    if below { -w } else { w }
}

/// The hue of a point of the cube on the hexagon, in radians (0 for a grey).
fn hexagon_hue(p: Raw, value: f32, whiteness: f32) -> f32 {
    let chroma = value - whiteness;
    if chroma <= 1e-7 {
        return 0.0;
    }
    // Without its grey, dilated about black to full value: a point of the hexagon.
    let q = black() + (p - (white() - black()) * whiteness - black()) * chroma.recip();
    let corners = hexagon();
    let (edge, along) = (0..6)
        .map(|k| {
            let (a, b) = (corners[k], corners[(k + 1) % 6]);
            ((a & b & q).norm(), k, (a & q).norm())
        })
        .filter(|(off, _, along)| *off < 1e-4 && *along <= 1.0 + 1e-4)
        .map(|(_, k, along)| (k, along))
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .unwrap_or((0, 0.0));
    ((edge as f32 + along) / 6.0 * TAU).rem_euclid(TAU)
}

/// The point of the hexagon at `hue`: along its edge, a sixth of a turn per edge.
fn on_hexagon(hue: f32) -> Raw {
    let x = hue.rem_euclid(TAU) / TAU * 6.0;
    let k = (x.floor() as usize).min(5);
    let corners = hexagon();
    let (a, b) = (corners[k], corners[(k + 1) % 6]);
    a + (b - a) * (x - k as f32)
}

/// A colour in HSV: hue (radians), saturation, value, and alpha.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hsv {
    /// The hue, in radians, red at 0.
    pub hue: f32,
    /// Chroma over value.
    pub saturation: f32,
    /// The largest channel.
    pub value: f32,
    /// The alpha.
    pub alpha: f32,
}

/// A colour in HSL: hue (radians), saturation, lightness, and alpha.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hsl {
    /// The hue, in radians, red at 0.
    pub hue: f32,
    /// Chroma over the most chroma the lightness allows.
    pub saturation: f32,
    /// The mean of the largest and smallest channel.
    pub lightness: f32,
    /// The alpha.
    pub alpha: f32,
}

/// A colour in HWB: hue (radians), whiteness, blackness, and alpha.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hwb {
    /// The hue, in radians, red at 0.
    pub hue: f32,
    /// The smallest channel.
    pub whiteness: f32,
    /// One less the largest channel.
    pub blackness: f32,
    /// The alpha.
    pub alpha: f32,
}

/// The HWB of an sRGB colour: the shared construction of the three models.
fn hwb_of(c: Srgb) -> Hwb {
    let p = if c.is_transparent() {
        black()
    } else {
        c.unit()
    };
    let (v, w) = (value(p), whiteness(p));
    Hwb {
        hue: hexagon_hue(p, v, w),
        whiteness: w,
        blackness: 1.0 - v,
        alpha: c.alpha(),
    }
}

/// The sRGB colour of an HWB: the hexagon's point at the hue, dilated about black by the chroma
/// left, plus the grey of the whiteness. Whiteness and blackness together beyond 1 give a grey.
fn srgb_of(h: Hwb) -> Srgb {
    let (w, b) = (h.whiteness, h.blackness);
    let p = if w + b >= 1.0 {
        black() + (white() - black()) * (w / (w + b))
    } else {
        black() + (on_hexagon(h.hue) - black()) * (1.0 - w - b) + (white() - black()) * w
    };
    Srgb::from_point(p * h.alpha)
}

impl From<Srgb> for Hwb {
    fn from(c: Srgb) -> Hwb {
        hwb_of(c)
    }
}
impl From<Hwb> for Srgb {
    fn from(h: Hwb) -> Srgb {
        srgb_of(h)
    }
}

impl From<Srgb> for Hsv {
    fn from(c: Srgb) -> Hsv {
        let h = hwb_of(c);
        let v = 1.0 - h.blackness;
        let saturation = if v > 0.0 { (v - h.whiteness) / v } else { 0.0 };
        Hsv {
            hue: h.hue,
            saturation,
            value: v,
            alpha: h.alpha,
        }
    }
}
impl From<Hsv> for Srgb {
    fn from(h: Hsv) -> Srgb {
        srgb_of(Hwb {
            hue: h.hue,
            whiteness: h.value * (1.0 - h.saturation),
            blackness: 1.0 - h.value,
            alpha: h.alpha,
        })
    }
}

impl From<Srgb> for Hsl {
    fn from(c: Srgb) -> Hsl {
        let h = hwb_of(c);
        let (v, w) = (1.0 - h.blackness, h.whiteness);
        let lightness = (v + w) / 2.0;
        let room = 1.0 - (2.0 * lightness - 1.0).abs();
        let saturation = if room > 0.0 { (v - w) / room } else { 0.0 };
        Hsl {
            hue: h.hue,
            saturation,
            lightness,
            alpha: h.alpha,
        }
    }
}
impl From<Hsl> for Srgb {
    fn from(h: Hsl) -> Srgb {
        let chroma = (1.0 - (2.0 * h.lightness - 1.0).abs()) * h.saturation;
        let whiteness = h.lightness - chroma / 2.0;
        srgb_of(Hwb {
            hue: h.hue,
            whiteness,
            blackness: 1.0 - whiteness - chroma,
            alpha: h.alpha,
        })
    }
}
