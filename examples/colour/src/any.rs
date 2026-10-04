//! A colour of any space, for storage that cannot be generic (as `bevy_color`'s `Color`).

use crate::colour::{Lab, LinearRgb, Oklab, Srgb, Xyz};

/// A colour of any of the spaces, converted on demand.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AnyColour {
    /// Linear sRGB.
    LinearRgb(LinearRgb),
    /// sRGB.
    Srgb(Srgb),
    /// CIE XYZ.
    Xyz(Xyz),
    /// CIELAB.
    Lab(Lab),
    /// Oklab.
    Oklab(Oklab),
}

impl AnyColour {
    /// The colour in linear sRGB, the hub of the conversions.
    pub fn linear(self) -> LinearRgb {
        match self {
            AnyColour::LinearRgb(c) => c,
            AnyColour::Srgb(c) => c.into(),
            AnyColour::Xyz(c) => c.into(),
            AnyColour::Lab(c) => c.into(),
            AnyColour::Oklab(c) => c.into(),
        }
    }

    /// The colour as the colour type `C`.
    pub fn to<C: From<LinearRgb>>(self) -> C {
        C::from(self.linear())
    }
}

macro_rules! any_from {
    ($($t:ident),*) => {$(
        impl From<$t> for AnyColour {
            fn from(c: $t) -> AnyColour {
                AnyColour::$t(c)
            }
        }
    )*};
}
any_from!(LinearRgb, Srgb, Xyz, Lab, Oklab);
