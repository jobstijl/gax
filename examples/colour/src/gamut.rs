//! Luminance, contrast and the sRGB gamut, in linear sRGB where they are linear: luminance is the
//! pairing with a plane, and the gamut is the unit cube, bounded by six planes.

use crate::colour::{Colour, LinearRgb};
use crate::space::Space;
use gax::pga3d::{Plane, Point};

/// The six faces of the sRGB gamut in linear sRGB, each facing inwards: `r ≥ 0`, `r ≤ 1`, and so
/// on.
fn faces() -> [Plane<(), f32>; 6] {
    let face = |n: [f32; 3], d: f32| Plane::from_normal(n, d);
    [
        face([1.0, 0.0, 0.0], 0.0),
        face([-1.0, 0.0, 0.0], -1.0),
        face([0.0, 1.0, 0.0], 0.0),
        face([0.0, -1.0, 0.0], -1.0),
        face([0.0, 0.0, 1.0], 0.0),
        face([0.0, 0.0, -1.0], -1.0),
    ]
}

impl<S: Space> Colour<S> {
    /// The relative luminance (CIE `Y`, white 1): the pairing of the colour in XYZ with the
    /// plane `Y = 0`.
    pub fn luminance(self) -> f32 {
        (Plane::new(0.0, 1.0, 0.0, 0.0) & self.convert::<crate::space::Xyz>().unit()).s()
    }

    /// The same chromaticity at luminance `y`: in linear sRGB, the colour dilated about black.
    pub fn with_luminance(self, y: f32) -> Self {
        let linear = self.convert::<crate::space::LinearRgb>();
        let now = self.luminance();
        if now == 0.0 {
            return LinearRgb::grey_of(y).with_alpha(self.alpha()).convert();
        }
        let black = Point::xyz(0.0, 0.0, 0.0);
        let p = black + (linear.unit() - black) * (y / now);
        LinearRgb::from_point(p * self.alpha()).convert()
    }

    /// The WCAG contrast ratio of two colours, from 1 (none) to 21 (black on white):
    /// `(Y_light + 0.05) / (Y_dark + 0.05)`.
    pub fn contrast(self, other: Colour<impl Space>) -> f32 {
        let (a, b) = (self.luminance(), other.luminance());
        (a.max(b) + 0.05) / (a.min(b) + 0.05)
    }

    /// Whether the colour can be shown on an sRGB display: in linear sRGB it lies on the inner
    /// side of every face of the unit cube.
    pub fn in_gamut(self) -> bool {
        let p = self.convert::<crate::space::LinearRgb>().unit();
        faces().iter().all(|f| (*f & p).s() >= -1e-6)
    }

    /// The colour brought into the sRGB gamut keeping its hue and its grey: in linear sRGB, the
    /// segment from its grey to it meets the faces it crosses, and the meet nearest the grey is
    /// the colour, as saturated as the display allows. A grey outside (too bright, or negative)
    /// is clipped to white or black.
    pub fn to_gamut(self) -> Self {
        if self.in_gamut() {
            return self;
        }
        let linear = self.convert::<crate::space::LinearRgb>();
        let (black, white) = (Point::xyz(0.0, 0.0, 0.0), Point::xyz(1.0, 1.0, 1.0));
        let grey = linear.grey().unit();
        let lightness = linear.lightness().clamp(0.0, 1.0);
        let grey_in = black + (white - black) * lightness;
        if (grey & grey_in).norm() > 1e-6 {
            return LinearRgb::from_point(grey_in * self.alpha()).convert();
        }
        let p = linear.unit();
        let ray = grey & p;
        let nearest = faces()
            .iter()
            .filter(|f| (**f & p).s() < 0.0)
            .map(|f| (ray ^ *f).unitized())
            .min_by(|a, b| (grey & *a).norm().total_cmp(&(grey & *b).norm()))
            .unwrap_or(p);
        LinearRgb::from_point(nearest * self.alpha()).convert()
    }
}
