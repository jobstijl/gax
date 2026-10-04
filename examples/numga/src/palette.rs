//! The examples' colours: lights, as in `examples/warp` (the `gax-colour` crate). Accents glow
//! like neon on a dark blue backdrop; a light's weight is its intensity, so a fainter stroke is
//! the same light faded.

use gax_colour::{Light, light};

/// The backdrop's top.
pub fn top() -> Light {
    light(0.06, 0.08, 0.22, 0.12)
}

/// The backdrop's bottom.
pub fn bottom() -> Light {
    light(0.02, 0.03, 0.10, 0.06)
}

/// Grid lines, frames and axes: the lattice's blue.
pub fn grid() -> Light {
    light(0.3, 0.42, 1.0, 0.14)
}

/// Text and light strokes: the HUD's blue-white.
pub fn ink() -> Light {
    light(0.75, 0.85, 1.0, 1.4)
}

/// Orange.
pub fn orange() -> Light {
    light(1.0, 0.42, 0.08, 1.8)
}

/// Cyan.
pub fn sky() -> Light {
    light(0.15, 0.85, 1.0, 1.7)
}

/// Teal green.
pub fn green() -> Light {
    light(0.1, 1.0, 0.62, 1.6)
}

/// Yellow.
pub fn yellow() -> Light {
    light(1.0, 0.86, 0.1, 1.7)
}

/// Blue.
pub fn blue() -> Light {
    light(0.32, 0.48, 1.0, 1.9)
}

/// Red.
pub fn red() -> Light {
    light(1.0, 0.14, 0.18, 1.9)
}

/// Magenta.
pub fn purple() -> Light {
    light(1.0, 0.2, 0.75, 1.8)
}

/// The accents in order, for series.
pub fn series(i: usize) -> Light {
    [orange, sky, green, yellow, blue, red, purple][i % 7]()
}
