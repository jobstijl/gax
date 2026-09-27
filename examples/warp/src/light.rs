//! Light as geometry.
//!
//! Grassmann's laws of colour mixing (1853) are projective geometry: a light is a homogeneous
//! point in linear RGB space, a PGA3D `Point` `(R, G, B, W)`, whose position `(R/W, G/W, B/W)`
//! is its colour and whose weight `W` is its intensity. Everything the game does with colour
//! is then an operation on points:
//!
//! * two lights add as points: the sum lies between them, weighted by their intensities, with
//!   their intensities summed, which is what additive glow does;
//! * fading scales the weight and keeps the point;
//! * whitening moves the point towards white along the line to it;
//! * a hue shift is a rotation about the grey axis (the line from black to white), a motor;
//! * desaturation moves towards the point's projection onto the grey axis, `(l | axis) ^ axis`.
//!
//! The raw coordinates `(R, G, B)` are chromaticity times intensity: radiance. The GPU takes a
//! light as it is (a `PointGpu`) and emits its first three coefficients.

use gax::pga3d::{Line, Motor, Point};

/// A light: a homogeneous point in linear RGB space.
pub type Light = Point<(), f32>;

/// The light of colour `(r, g, b)` at intensity `i`.
pub const fn light(r: f32, g: f32, b: f32, i: f32) -> Light {
    Point::new(r * i, g * i, b * i, i)
}

/// No light.
pub const DARK: Light = Point::new(0.0, 0.0, 0.0, 0.0);

/// The intensity: the weight.
pub fn intensity(l: Light) -> f32 {
    l.e123()
}

/// `l` at `k` times its intensity.
pub fn fade(l: Light, k: f32) -> Light {
    crate::light_fade(l, k)
}

/// `l` moved towards white by `t`, at the same intensity.
pub fn whiten(l: Light, t: f32) -> Light {
    crate::light_whiten(l, t)
}

/// The additive mix `(1 - t) a + t b`.
pub fn mix(a: Light, b: Light, t: f32) -> Light {
    crate::light_mix(a, b, t)
}

/// The grey axis: black to white.
pub fn grey_axis() -> Line<(), f32> {
    Point::xyz(0.0, 0.0, 0.0) & Point::xyz(1.0, 1.0, 1.0)
}

/// `l` with its hue turned by `angle`: a rotation about the grey axis, which keeps intensity
/// and saturation.
pub fn hue_shift(l: Light, angle: f32) -> Light {
    Motor::rotation(grey_axis(), angle) >> l
}

/// `l` moved towards the grey of the same brightness by `t`: towards its foot on the grey
/// axis, the meet of the axis with the plane through `l` orthogonal to it.
pub fn desaturate(l: Light, t: f32) -> Light {
    let axis = grey_axis();
    let foot = (l | axis) ^ axis;
    // The same weight as `l`, so that the mix keeps the intensity.
    let foot = foot.gp(l.e123() / foot.e123());
    mix(l, foot, t)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rgb(l: Light) -> [f32; 3] {
        l.to_euclidean()
    }

    fn close(a: [f32; 3], b: [f32; 3]) -> bool {
        a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-5)
    }

    /// Adding lights is additive mixing: the colour is the intensity-weighted mean, the
    /// intensity the sum.
    #[test]
    fn lights_add_like_light() {
        let red = light(1.0, 0.0, 0.0, 3.0);
        let blue = light(0.0, 0.0, 1.0, 1.0);
        let m = red + blue;
        assert!((intensity(m) - 4.0).abs() < 1e-6);
        assert!(close(rgb(m), [0.75, 0.0, 0.25]));
        // Radiance is the raw coordinates: the sum of the radiances.
        assert!((m.e032() - 3.0).abs() < 1e-6 && (m.e021() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn fading_keeps_the_colour_and_whitening_keeps_the_intensity() {
        let c = light(0.2, 0.6, 1.0, 2.0);
        assert!(close(rgb(fade(c, 0.25)), rgb(c)));
        assert!((intensity(fade(c, 0.25)) - 0.5).abs() < 1e-6);
        let w = whiten(c, 1.0);
        assert!(close(rgb(w), [1.0, 1.0, 1.0]) && (intensity(w) - 2.0).abs() < 1e-6);
    }

    /// A hue shift is a rotation about the grey axis: a third of a turn takes red to green,
    /// and grey stays grey.
    #[test]
    fn hue_is_an_angle_about_the_grey_axis() {
        let red = light(1.0, 0.0, 0.0, 2.0);
        let g = hue_shift(red, core::f32::consts::TAU / 3.0);
        assert!(close(rgb(g), [0.0, 1.0, 0.0]), "{:?}", rgb(g));
        assert!((intensity(g) - 2.0).abs() < 1e-5);
        let grey = light(0.4, 0.4, 0.4, 1.0);
        assert!(close(rgb(hue_shift(grey, 1.0)), [0.4, 0.4, 0.4]));
    }

    #[test]
    fn desaturation_moves_to_the_grey_axis() {
        let c = light(1.0, 0.5, 0.0, 2.0);
        let g = desaturate(c, 1.0);
        assert!(close(rgb(g), [0.5, 0.5, 0.5]), "{:?}", rgb(g));
        assert!((intensity(g) - 2.0).abs() < 1e-5);
    }
}
