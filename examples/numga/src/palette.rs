//! The examples' colours: a dark backdrop, and accents that stay apart for colour-blind eyes
//! (after Okabe and Ito), in linear light.

use crate::canvas::{Rgb, srgb};

/// The backdrop's top.
pub fn top() -> Rgb {
    srgb(0.085, 0.095, 0.13)
}

/// The backdrop's bottom.
pub fn bottom() -> Rgb {
    srgb(0.02, 0.022, 0.035)
}

/// Grid lines and axes.
pub fn grid() -> Rgb {
    srgb(0.22, 0.24, 0.30)
}

/// Text and light strokes.
pub fn ink() -> Rgb {
    srgb(0.88, 0.89, 0.92)
}

/// Orange.
pub fn orange() -> Rgb {
    srgb(0.90, 0.62, 0.0)
}

/// Sky blue.
pub fn sky() -> Rgb {
    srgb(0.34, 0.71, 0.91)
}

/// Bluish green.
pub fn green() -> Rgb {
    srgb(0.0, 0.62, 0.45)
}

/// Yellow.
pub fn yellow() -> Rgb {
    srgb(0.94, 0.89, 0.26)
}

/// Blue.
pub fn blue() -> Rgb {
    srgb(0.0, 0.45, 0.70)
}

/// Vermillion.
pub fn red() -> Rgb {
    srgb(0.84, 0.37, 0.0)
}

/// Reddish purple.
pub fn purple() -> Rgb {
    srgb(0.80, 0.47, 0.65)
}

/// The accents in order, for series.
pub fn series(i: usize) -> Rgb {
    [orange, sky, green, yellow, blue, red, purple][i % 7]()
}
