//! Chromaticity and colour temperature: the projective plane of colour. Every colour lies on a
//! ray from black in CIE XYZ; the ray meets the plane `X + Y + Z = 1` at the colour's
//! chromaticity, the point `(x, y)` of the chromaticity diagram, and the colour is that point
//! dilated about black to its luminance.

use crate::colour::{Colour, LinearRgb, Xyz};
use crate::space::{self, Space};
use gax::pga2d;
use gax::pga3d::{Plane, Point};

/// A point of the chromaticity diagram.
pub type Chromaticity = pga2d::Point<(), f32>;

/// The plane `X + Y + Z = 1` of XYZ.
fn unit_plane() -> Plane<(), f32> {
    Plane::new(1.0, 1.0, 1.0, -1.0)
}

impl<S: Space> Colour<S> {
    /// The chromaticity: where the colour's ray from black meets the plane `X + Y + Z = 1` in
    /// XYZ, as the point `(x, y)` of the chromaticity diagram. `None` for black, which has no
    /// ray.
    pub fn chromaticity(self) -> Option<Chromaticity> {
        let p = self.convert::<space::Xyz>().unit();
        let black = Point::xyz(0.0, 0.0, 0.0);
        let ray = black & p;
        if ray.norm() < 1e-9 {
            return None;
        }
        let [x, y, _] = (ray ^ unit_plane()).unitized().to_euclidean();
        Some(Chromaticity::xy(x, y))
    }

    /// The correlated colour temperature in kelvin, by McCamy's approximation: the line from the
    /// chromaticity to McCamy's epicentre `(0.3320, 0.1858)` has inverse slope `n` (read off the
    /// line's coefficients), and the temperature is a cubic in `n`. Good to a few kelvin from
    /// 2000 K to 12 500 K near the Planckian locus. `None` for black.
    pub fn temperature(self) -> Option<f32> {
        let line = Chromaticity::xy(0.3320, 0.1858) & self.chromaticity()?;
        let n = line.e2() / line.e1();
        Some(((449.0 * n + 3525.0) * n + 6823.3) * n + 5520.33)
    }
}

impl Xyz {
    /// The colour of chromaticity `xy` and luminance `luminance`: the point
    /// `(x, y, 1 - x - y)` of the unit plane, dilated about black until its `Y` is the luminance.
    pub fn from_chromaticity(xy: Chromaticity, luminance: f32) -> Xyz {
        let [x, y] = xy.to_euclidean();
        let c = Point::xyz(x, y, 1.0 - x - y);
        let black = Point::xyz(0.0, 0.0, 0.0);
        let scale = luminance / (Plane::new(0.0, 1.0, 0.0, 0.0) & c).s();
        Xyz::from_point(black + (c - black) * scale)
    }
}

impl LinearRgb {
    /// The colour of a black body at `kelvin` (1667 K to 25 000 K), at intensity `intensity`:
    /// its chromaticity on the Planckian locus (Kim et al.'s cubic splines, polynomials in
    /// `1 / T` and `x`), as a light, dilated about black until its brightest channel is 1 and
    /// brought into the gamut.
    pub fn from_temperature(kelvin: f32, intensity: f32) -> LinearRgb {
        // The splines in f64, with their published digits.
        let t = f64::from(kelvin.clamp(1667.0, 25_000.0));
        let (u, u2, u3) = (1e3 / t, 1e6 / (t * t), 1e9 / (t * t * t));
        let x = if t <= 4000.0 {
            -0.266_123_9 * u3 - 0.234_358_9 * u2 + 0.877_695_6 * u + 0.179_910
        } else {
            -3.025_846_9 * u3 + 2.107_037_9 * u2 + 0.222_634_7 * u + 0.240_390
        };
        let (a, b, c, d) = if t <= 2222.0 {
            (-1.106_381_4, -1.348_110_20, 2.185_558_32, -0.202_196_83)
        } else if t <= 4000.0 {
            (-0.954_947_6, -1.374_185_93, 2.091_370_15, -0.167_488_67)
        } else {
            (3.081_758_0, -5.873_386_70, 3.751_129_97, -0.370_014_83)
        };
        let y = ((a * x + b) * x + c) * x + d;
        let xy = Chromaticity::xy(x as f32, y as f32);
        let colour: LinearRgb = Xyz::from_chromaticity(xy, 1.0).into();
        colour.at_full_value().to_gamut().with_alpha(intensity)
    }

    /// The same colour dilated about black until its brightest channel is 1 (HSV's value 1).
    fn at_full_value(self) -> LinearRgb {
        let value = crate::models::value(self.unit());
        if value == 0.0 {
            return self;
        }
        let black = Point::xyz(0.0, 0.0, 0.0);
        LinearRgb::from_point((black + (self.unit() - black) * value.recip()) * self.alpha())
    }
}
