//! Light as geometry: the lights of `gax-light` (shared with the other examples), with the
//! game's own redshift, and fading and whitening through the traced kernels.

pub use gax_light::{DARK, Light, blend, desaturate, hue_shift, intensity, light};

/// `l` at `k` times its intensity.
pub fn fade(l: Light, k: f32) -> Light {
    crate::light_fade(l, k)
}

/// `l` moved towards white by `t`, at the same intensity.
pub fn whiten(l: Light, t: f32) -> Light {
    crate::light_whiten(l, t)
}

/// Gravitational redshift by `z` (`0` none): the spectrum slides towards red, blue through
/// green and yellow, which in OkLab is a turn of the hue about the lightness axis (towards
/// lower hue angles), and the light dims as `1 / (1 + z)²`.
pub fn redshift(l: Light, z: f32) -> Light {
    let k = 1.0 / (1.0 + z);
    fade(hue_shift(l, -1.4 * z / (1.0 + z)), k * k)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rgb(l: Light) -> [f32; 3] {
        l.to_euclidean()
    }

    /// Redshift slides blue towards green (the way to red) and dims it; a blueshift goes the
    /// other way and brightens.
    #[test]
    fn redshift_slides_blue_towards_red() {
        let blue = light(0.2, 0.3, 1.0, 2.0);
        let s = redshift(blue, 0.8);
        let [r, g, b] = rgb(s);
        assert!(g > 0.3 && b < 1.0 && r >= 0.0, "{:?}", rgb(s));
        assert!(intensity(s) < 1.0);
        assert!(intensity(redshift(blue, -0.3)) > 2.0);
    }
}
