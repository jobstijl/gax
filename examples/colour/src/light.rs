//! Light: colours of linear sRGB, whose coordinates are radiance. Lights add as points, a fade
//! is a change of weight, and a light is shown through a tonemapper ([`LinearRgb::agx`]).

use crate::colour::LinearRgb;
use crate::ops;
#[cfg(feature = "gpu")]
use gax::pga3d::{Point, PointGpu};

/// A light: a colour of linear sRGB, its weight its intensity.
pub type Light = LinearRgb;

/// The light of colour `(r, g, b)` (linear) at intensity `i`.
pub const fn light(r: f32, g: f32, b: f32, i: f32) -> Light {
    LinearRgb::new(r, g, b, i)
}

/// No light.
pub const DARK: Light = LinearRgb::transparent();

impl LinearRgb {
    /// The colour `(r, g, b)`, fully there.
    pub const fn rgb(r: f32, g: f32, b: f32) -> Self {
        LinearRgb::new(r, g, b, 1.0)
    }

    /// The light of a colour given in sRGB components (as colour pickers give them) at
    /// intensity `i`.
    pub fn from_srgb(r: f32, g: f32, b: f32, i: f32) -> Self {
        crate::Srgb::rgba(r, g, b, i).into()
    }

    /// The luminance of the light, its intensity included: the pairing of its point with the
    /// plane of luminance weights.
    pub fn luma(self) -> f32 {
        ops::luma(self.point())
    }

    /// The intensity: the weight.
    pub fn intensity(self) -> f32 {
        self.alpha()
    }

    /// Moved towards white by `t`, at the same intensity.
    pub fn whitened(self, t: f32) -> Self {
        LinearRgb::from_point(ops::whiten(self.point(), t))
    }

    /// The light `t` of the way to `other` by adding them in proportion: the physical mix,
    /// radiance and intensity both in proportion (where [`crate::Colour::mix`] interpolates the
    /// position and the alpha apart).
    pub fn mix_light(self, other: Self, t: f32) -> Self {
        LinearRgb::from_point(ops::mix(self.point(), other.point(), t))
    }

    /// The light `t` of the way to `other` as the eye sees it: mixed in Oklab, where equal steps
    /// look equal (lightness, and the intensities, in proportion).
    pub fn blend(self, other: Self, t: f32) -> Self {
        crate::Oklab::from(self).mix(other.into(), t).into()
    }

    /// This colour over `below`: the "over" of compositing, `self + below (1 - alpha)`, which in
    /// homogeneous points is a sum.
    pub fn over(self, below: Self) -> Self {
        self + below.faded(1.0 - self.alpha())
    }

    /// The light tonemapped for a display by AgX, as linear display light (`saturation` moves
    /// the look away from grey, 1 none).
    pub fn agx(self, saturation: f32) -> Self {
        LinearRgb::from_point(ops::agx(self.point(), saturation))
    }

    /// The opaque colour the light makes on black: its radiance as a colour of alpha 1 (what a
    /// pixel that gathered this light shows).
    pub fn on_black(self) -> Self {
        let p = self.point();
        LinearRgb::rgb(p.e032(), p.e013(), p.e021())
    }

    /// The radiance and the weight, `[r, g, b, w]` (premultiplied, for a GPU).
    pub fn to_premultiplied(self) -> [f32; 4] {
        let p = self.point();
        [p.e032(), p.e013(), p.e021(), p.e123()]
    }
}

macro_rules! arithmetic {
    ($($space:ty),*) => {$(
        impl core::ops::Add for crate::Colour<$space> {
            type Output = Self;
            /// The sum of the points: lights added, or a composite.
            fn add(self, o: Self) -> Self {
                Self::from_point(self.point() + o.point())
            }
        }
        impl core::ops::AddAssign for crate::Colour<$space> {
            fn add_assign(&mut self, o: Self) {
                *self = *self + o;
            }
        }
        impl core::ops::Sub for crate::Colour<$space> {
            type Output = Self;
            fn sub(self, o: Self) -> Self {
                Self::from_point(self.point() - o.point())
            }
        }
        impl core::ops::Mul<f32> for crate::Colour<$space> {
            type Output = Self;
            /// The colour `k` times as much: [`crate::Colour::faded`].
            fn mul(self, k: f32) -> Self {
                self.faded(k)
            }
        }
    )*};
}
// The spaces in which adding coordinates means something: physically (linear sRGB, XYZ) or
// perceptually (CIELAB, Oklab), as in bevy_color.
arithmetic!(
    crate::space::LinearRgb,
    crate::space::Xyz,
    crate::space::Lab,
    crate::space::Oklab
);

#[cfg(feature = "gpu")]
impl From<LinearRgb> for PointGpu {
    fn from(l: LinearRgb) -> PointGpu {
        l.point().into()
    }
}

#[cfg(feature = "gpu")]
impl From<PointGpu> for LinearRgb {
    fn from(p: PointGpu) -> LinearRgb {
        LinearRgb::from_point(Point::from(p))
    }
}
