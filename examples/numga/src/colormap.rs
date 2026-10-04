//! Colormaps as functions of `t ∈ [0, 1]`: lights through stops given in sRGB (the stops of
//! matplotlib's maps of the same names, sampled), blended in OkLab between them, so that equal
//! steps of `t` look like equal steps.

use gax_colour::{Light, Oklab, Srgb};

/// The intensity of a colormap's lights.
pub const INTENSITY: f32 = 1.0;

/// A colormap through the sRGB stops `s`, spaced evenly over `t ∈ [0, 1]`: the two stops around
/// `t` mixed in Oklab, brought into the display's gamut (a perceptual mix of saturated stops
/// can leave it), as a light.
pub fn stops(t: f32, s: &[Srgb]) -> Light {
    let t = if t.is_finite() {
        t.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let x = t * (s.len() - 1) as f32;
    let i = (x.floor() as usize).min(s.len() - 2);
    let mixed = Oklab::from(s[i]).mix(s[i + 1].into(), x - i as f32);
    Light::from(mixed).to_gamut().with_alpha(INTENSITY)
}

/// Dark purple through teal to yellow.
pub fn viridis(t: f32) -> Light {
    stops(
        t,
        &[
            Srgb::rgb(0.267, 0.005, 0.329),
            Srgb::rgb(0.283, 0.141, 0.458),
            Srgb::rgb(0.254, 0.265, 0.530),
            Srgb::rgb(0.207, 0.372, 0.553),
            Srgb::rgb(0.164, 0.471, 0.558),
            Srgb::rgb(0.128, 0.567, 0.551),
            Srgb::rgb(0.135, 0.659, 0.518),
            Srgb::rgb(0.267, 0.749, 0.441),
            Srgb::rgb(0.478, 0.821, 0.317),
            Srgb::rgb(0.741, 0.873, 0.150),
            Srgb::rgb(0.993, 0.906, 0.144),
        ],
    )
}

/// Black through red and orange to pale yellow.
pub fn inferno(t: f32) -> Light {
    stops(
        t,
        &[
            Srgb::rgb(0.001, 0.000, 0.014),
            Srgb::rgb(0.087, 0.045, 0.225),
            Srgb::rgb(0.258, 0.039, 0.406),
            Srgb::rgb(0.416, 0.090, 0.433),
            Srgb::rgb(0.578, 0.148, 0.404),
            Srgb::rgb(0.735, 0.216, 0.330),
            Srgb::rgb(0.865, 0.317, 0.226),
            Srgb::rgb(0.954, 0.469, 0.098),
            Srgb::rgb(0.987, 0.645, 0.040),
            Srgb::rgb(0.964, 0.843, 0.273),
            Srgb::rgb(0.988, 0.998, 0.645),
        ],
    )
}

/// Blue through white to red (diverging, centred at `t = 0.5`).
pub fn coolwarm(t: f32) -> Light {
    stops(
        t,
        &[
            Srgb::rgb(0.230, 0.299, 0.754),
            Srgb::rgb(0.552, 0.690, 0.996),
            Srgb::rgb(0.866, 0.866, 0.866),
            Srgb::rgb(0.958, 0.604, 0.482),
            Srgb::rgb(0.706, 0.016, 0.150),
        ],
    )
}

/// Red through white to blue, reversed: blue for low, red for high (matplotlib's RdBu_r).
pub fn rdbu(t: f32) -> Light {
    stops(
        t,
        &[
            Srgb::rgb(0.020, 0.188, 0.380),
            Srgb::rgb(0.263, 0.576, 0.765),
            Srgb::rgb(0.969, 0.969, 0.969),
            Srgb::rgb(0.839, 0.376, 0.302),
            Srgb::rgb(0.404, 0.000, 0.122),
        ],
    )
}

/// A rainbow without its harshest steps.
pub fn turbo(t: f32) -> Light {
    stops(
        t,
        &[
            Srgb::rgb(0.190, 0.072, 0.232),
            Srgb::rgb(0.274, 0.397, 0.878),
            Srgb::rgb(0.157, 0.733, 0.927),
            Srgb::rgb(0.196, 0.948, 0.594),
            Srgb::rgb(0.643, 0.990, 0.236),
            Srgb::rgb(0.950, 0.785, 0.214),
            Srgb::rgb(0.979, 0.459, 0.110),
            Srgb::rgb(0.796, 0.161, 0.016),
            Srgb::rgb(0.480, 0.016, 0.011),
        ],
    )
}

/// White to dark blue.
pub fn blues(t: f32) -> Light {
    stops(
        t,
        &[
            Srgb::rgb(0.969, 0.984, 1.0),
            Srgb::rgb(0.776, 0.859, 0.937),
            Srgb::rgb(0.420, 0.682, 0.839),
            Srgb::rgb(0.129, 0.443, 0.710),
            Srgb::rgb(0.031, 0.188, 0.420),
        ],
    )
}

/// The hue circle at full saturation (`t` wraps).
pub fn hsv(t: f32) -> Light {
    let t = t.rem_euclid(1.0);
    stops(
        t,
        &[
            Srgb::rgb(1.0, 0.0, 0.0),
            Srgb::rgb(1.0, 1.0, 0.0),
            Srgb::rgb(0.0, 1.0, 0.0),
            Srgb::rgb(0.0, 1.0, 1.0),
            Srgb::rgb(0.0, 0.0, 1.0),
            Srgb::rgb(1.0, 0.0, 1.0),
            Srgb::rgb(1.0, 0.0, 0.0),
        ],
    )
}
