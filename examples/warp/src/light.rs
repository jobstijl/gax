//! Light as geometry: the colours of `gax-colour` (shared with the other examples), lights being
//! colours of linear sRGB whose weight is their intensity, and the game's own redshift.

pub use gax_colour::{DARK, Light, light};

/// Gravitational redshift by `z` (`0` none): the spectrum slides towards red, blue through
/// green and yellow, which in Oklab is a turn of the hue about the lightness axis (towards
/// lower hue angles), brought back into the display's gamut (light cannot go negative), and
/// the light dims as `1 / (1 + z)²`.
pub fn redshift(l: Light, z: f32) -> Light {
    let k = 1.0 / (1.0 + z);
    l.perceptually(|c| c.rotate_hue(-1.4 * z / (1.0 + z)))
        .to_gamut()
        .faded(k * k)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Redshift slides blue towards green (the way to red) and dims it; a blueshift goes the
    /// other way and brightens.
    #[test]
    fn redshift_slides_blue_towards_red() {
        let blue = light(0.2, 0.3, 1.0, 2.0);
        let s = redshift(blue, 0.8);
        let [r, g, b] = s.unit().to_euclidean();
        assert!(g > 0.3 && b < 1.0 && r >= 0.0, "{s:?}");
        assert!(s.intensity() < 1.0);
        assert!(redshift(blue, -0.3).intensity() > 2.0);
    }
}
