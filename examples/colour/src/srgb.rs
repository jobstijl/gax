//! sRGB as stored: hex codes and bytes.

use crate::colour::Srgb;

/// Why a hex code could not be read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HexError {
    /// Not 3, 4, 6 or 8 hex digits (after an optional `#`).
    Length(usize),
    /// A character that is not a hex digit.
    Digit(char),
}

impl core::fmt::Display for HexError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            HexError::Length(n) => write!(f, "a hex colour has 3, 4, 6 or 8 digits, not {n}"),
            HexError::Digit(c) => write!(f, "{c:?} is not a hex digit"),
        }
    }
}

impl std::error::Error for HexError {}

/// A byte as a stored value in `[0, 1]`, and back.
fn unit(b: u8) -> f32 {
    f32::from(b) / 255.0
}
fn byte(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}

impl Srgb {
    /// The colour `(r, g, b)` (stored values in `[0, 1]`), fully there.
    pub const fn rgb(r: f32, g: f32, b: f32) -> Self {
        Srgb::new(r, g, b, 1.0)
    }

    /// The colour `(r, g, b)` with alpha `a`.
    pub const fn rgba(r: f32, g: f32, b: f32, a: f32) -> Self {
        Srgb::new(r, g, b, a)
    }

    /// The colour of bytes `[r, g, b, a]`.
    pub fn from_u8(c: [u8; 4]) -> Self {
        Srgb::new(unit(c[0]), unit(c[1]), unit(c[2]), unit(c[3]))
    }

    /// The colour's bytes `[r, g, b, a]` (clamped to the gamut's range).
    pub fn to_u8(self) -> [u8; 4] {
        let p = if self.is_transparent() {
            gax::pga3d::Point::xyz(0.0, 0.0, 0.0)
        } else {
            self.unit()
        };
        [
            byte(p.e032()),
            byte(p.e013()),
            byte(p.e021()),
            byte(self.alpha()),
        ]
    }

    /// The colour of a hex code: `RGB`, `RGBA`, `RRGGBB` or `RRGGBBAA`, with or without `#`.
    pub fn hex(code: &str) -> Result<Self, HexError> {
        let digits = code.strip_prefix('#').unwrap_or(code);
        let values: Vec<u8> = digits
            .chars()
            .map(|c| c.to_digit(16).map(|d| d as u8).ok_or(HexError::Digit(c)))
            .collect::<Result<_, _>>()?;
        let pair = |i: usize| values[2 * i] * 16 + values[2 * i + 1];
        let single = |i: usize| values[i] * 17;
        let bytes = match values.len() {
            3 => [single(0), single(1), single(2), 255],
            4 => [single(0), single(1), single(2), single(3)],
            6 => [pair(0), pair(1), pair(2), 255],
            8 => [pair(0), pair(1), pair(2), pair(3)],
            n => return Err(HexError::Length(n)),
        };
        Ok(Srgb::from_u8(bytes))
    }

    /// The colour's hex code, `#rrggbb`, or `#rrggbbaa` when it is not fully there.
    pub fn to_hex(self) -> String {
        let [r, g, b, a] = self.to_u8();
        if a == 255 {
            format!("#{r:02x}{g:02x}{b:02x}")
        } else {
            format!("#{r:02x}{g:02x}{b:02x}{a:02x}")
        }
    }
}

/// The light of an sRGB hex code `0xRRGGBB` at intensity `i`.
pub fn hex(code: u32, i: f32) -> crate::Light {
    let [_, r, g, b] = code.to_be_bytes();
    Srgb::from_u8([r, g, b, 255])
        .convert::<crate::space::LinearRgb>()
        .with_alpha(i)
}
