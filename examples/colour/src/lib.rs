//! Colour as geometry, shared by the examples (`examples/warp`, `examples/numga`), after
//! `bevy_color`: the same spaces, operations and palettes, all on gax's PGA3D points.
//!
//! **A colour is a homogeneous point.** In each colour space a colour is a point whose position
//! is the colour and whose weight is how much of it there is: its alpha. Premultiplied RGBA is
//! exactly a homogeneous point, so alpha (paint's coverage) and intensity (a light's strength)
//! are the same weight, and both compositing and the mixing of lights are sums of points:
//! `top + bottom (1 - alpha(top))` is "over"; two lights add as they are. Grassmann's laws of
//! colour mixing (1853) are this projective geometry.
//!
//! **Spaces are maps.** [`LinearRgb`] (radiance, the renderer's), [`Srgb`] (as stored and
//! shown), [`Xyz`] (CIE 1931), [`Lab`] (CIELAB) and [`Oklab`] are reached from linear sRGB by
//! gax point maps, with each space's transfer curve (a function of one coordinate) between
//! them. [`Colour::to`] and `From` convert, keeping the alpha.
//!
//! **Hue, chroma and lightness are geometry about the grey axis** (the line from a space's black
//! to its white), so the cylindrical models (HSL and HSV over sRGB, LCh over CIELAB, OkLCh over
//! Oklab) are not more types but operations on any colour, in its own space:
//!
//! * lightness is the distance along the axis, and a change of lightness a translation along it;
//! * chroma is the distance from the axis, the norm of a join, and a change of chroma a dilation
//!   about the colour's grey (its foot on the axis);
//! * hue is the angle about the axis from the half-plane of hue zero, read off the logarithm of
//!   the rotation between the two half-planes, and a change of hue a rotation motor about the
//!   axis, which keeps lightness and chroma;
//! * mixing around the axis ([`Colour::mix_hue`]) is all three at once.
//!
//! **HSV, HSL and HWB** ([`Hsv`], [`Hsl`], [`Hwb`]) are the pickers' piecewise models of the
//! sRGB cube, exact and constructed in it: value and whiteness are meets of rays with the cube's
//! faces, and hue is the position on the hexagon of its saturated edges ([`models`]).
//!
//! **Chromaticity is a projective plane**: the colour's ray from black in XYZ meets the plane
//! `X + Y + Z = 1` at its chromaticity, and colour temperature lives there ([`chromaticity`]:
//! black bodies on the Planckian locus, the correlated temperature through McCamy's epicentre).
//!
//! **Luminance and gamut are planes** in linear sRGB: the relative luminance is the pairing with
//! the plane of luminance weights, and the sRGB gamut is the unit cube, inside six planes; a
//! colour is brought into it where the segment from its grey meets the faces
//! ([`Colour::to_gamut`]).
//!
//! The transfer curves (the display's power, CIELAB's and Oklab's cube roots, AgX's contrast
//! curve) are the only arithmetic on single numbers: each is a function of one coordinate by
//! definition, applied to every coordinate of a point ([`ops::each`]).
//!
//! The operations a renderer traces into shaders (mixing, fading, luminance, AgX) are also in
//! [`ops`], generic over gax's `Real`.
//!
//! ```
//! use gax_colour::{Gradient, LinearRgb, Oklab, Srgb, palettes::css};
//!
//! // A colour picked in sRGB, its hue turned a third of the way round in Oklab (a rotation
//! // about the grey axis, which keeps lightness and chroma), and stored again.
//! let orange = Srgb::hex("#ff8800").unwrap();
//! let turned: Srgb = orange.perceptually(|c| c.rotate_hue(core::f32::consts::TAU / 3.0));
//! let (a, b) = (Oklab::from(orange), Oklab::from(turned));
//! assert!((a.lightness() - b.lightness()).abs() < 1e-4 && (a.chroma() - b.chroma()).abs() < 1e-4);
//!
//! // Half covering white over black is the sum of two points: a mid grey in linear light.
//! let veil = LinearRgb::rgb(1.0, 1.0, 1.0).with_alpha(0.5);
//! let grey = veil.over(LinearRgb::rgb(0.0, 0.0, 0.0));
//! assert!((grey.luminance() - 0.5).abs() < 1e-5);
//!
//! // A perceptual gradient through named colours, kept within the display's gamut.
//! let sunset = Gradient::new([css::MIDNIGHT_BLUE, css::CRIMSON, css::GOLD].map(Oklab::from));
//! assert!(sunset.at(0.25).to_gamut().in_gamut());
//! ```

pub mod any;
pub mod chromaticity;
pub mod colour;
pub mod gamut;
pub mod gradient;
pub mod light;
pub mod models;
pub mod ops;
pub mod palettes;
pub mod space;
pub mod srgb;

pub use any::AnyColour;
pub use chromaticity::Chromaticity;
pub use colour::{Colour, Lab, LinearRgb, Oklab, Srgb, Xyz};
pub use gradient::{ColourRange, Gradient};
pub use light::{DARK, Light, light};
pub use models::{Hsl, Hsv, Hwb};
pub use space::Space;
pub use srgb::{HexError, hex};

/// Conversions between every two spaces, through linear sRGB.
macro_rules! conversions {
    ($($a:ident => $($b:ident),*);* $(;)?) => {$($(
        impl From<Colour<space::$a>> for Colour<space::$b> {
            fn from(c: Colour<space::$a>) -> Self {
                c.convert()
            }
        }
    )*)*};
}
conversions! {
    LinearRgb => Srgb, Xyz, Lab, Oklab;
    Srgb => LinearRgb, Xyz, Lab, Oklab;
    Xyz => LinearRgb, Srgb, Lab, Oklab;
    Lab => LinearRgb, Srgb, Xyz, Oklab;
    Oklab => LinearRgb, Srgb, Xyz, Lab;
}

#[cfg(test)]
mod tests {
    use super::*;
    use gax::pga3d::Point;

    /// Two colours of one space at the same position and alpha, to `eps`.
    fn near<S: Space>(a: Colour<S>, b: Colour<S>, eps: f32) -> bool {
        a.distance(b) < eps && (a.alpha() - b.alpha()).abs() < eps
    }

    /// A colour's coordinates (for comparing with published values).
    fn coords<S: Space>(c: Colour<S>) -> [f32; 3] {
        c.unit().to_euclidean()
    }

    fn close(a: [f32; 3], b: [f32; 3], eps: f32) -> bool {
        a.iter().zip(b).all(|(x, y)| (x - y).abs() < eps)
    }

    #[test]
    fn every_space_converts_there_and_back() {
        let samples = [
            Srgb::rgb(0.9, 0.2, 0.1),
            Srgb::rgba(0.1, 0.6, 0.8, 0.5),
            Srgb::rgb(0.5, 0.5, 0.5),
            Srgb::rgb(0.02, 0.9, 0.3),
        ];
        for c in samples {
            assert!(near(c.to::<Xyz>().to(), c, 1e-5), "XYZ {c:?}");
            assert!(near(c.to::<Lab>().to(), c, 1e-4), "Lab {c:?}");
            assert!(near(c.to::<Oklab>().to(), c, 1e-4), "Oklab {c:?}");
            assert!(near(LinearRgb::from(c).into(), c, 1e-5), "linear {c:?}");
            assert!((Oklab::from(c).alpha() - c.alpha()).abs() < 1e-6);
        }
    }

    /// Published values: sRGB red in Oklab (Ottosson) and in CIELAB, white at lightness 1, the
    /// D65 white in XYZ.
    #[test]
    fn spaces_match_their_definitions() {
        let red = Srgb::rgb(1.0, 0.0, 0.0);
        assert!(close(
            coords(red.to::<Oklab>()),
            [0.628, 0.225, 0.126],
            1e-3
        ));
        assert!(close(
            coords(red.to::<Lab>()),
            [0.5324, 0.8009, 0.6720],
            2e-3
        ));
        let white = Srgb::rgb(1.0, 1.0, 1.0);
        assert!(close(coords(white.to::<Oklab>()), [1.0, 0.0, 0.0], 1e-4));
        assert!(close(coords(white.to::<Lab>()), [1.0, 0.0, 0.0], 1e-4));
        assert!(close(coords(white.to::<Xyz>()), [0.9505, 1.0, 1.089], 1e-3));
        assert!(close(
            coords(Srgb::rgb(0.5, 0.5, 0.5).to::<LinearRgb>()),
            [0.214; 3],
            1e-3
        ));
    }

    #[test]
    fn lights_add_and_composite_as_points() {
        let red = light(1.0, 0.0, 0.0, 3.0);
        let blue = light(0.0, 0.0, 1.0, 1.0);
        let m = red + blue;
        assert!((m.intensity() - 4.0).abs() < 1e-6);
        assert!(close(coords(m), [0.75, 0.0, 0.25], 1e-6));
        // Half-covering white over black: the over operator is a sum of points.
        let white = LinearRgb::rgb(1.0, 1.0, 1.0).with_alpha(0.5);
        let black = LinearRgb::rgb(0.0, 0.0, 0.0);
        let o = white.over(black);
        assert!((o.alpha() - 1.0).abs() < 1e-6 && close(coords(o), [0.5; 3], 1e-6));
        // Fading keeps the colour; whitening keeps the intensity.
        let c = light(0.2, 0.6, 1.0, 2.0);
        assert!(close(coords(c.faded(0.25)), coords(c), 1e-6));
        let w = c.whitened(1.0);
        assert!(close(coords(w), [1.0; 3], 1e-6) && (w.intensity() - 2.0).abs() < 1e-6);
    }

    /// A turn of hue is a rotation about the grey axis: lightness and chroma stay, the hue
    /// moves by the turn, half a turn is the opposite hue, and a grey stays put.
    #[test]
    fn hue_turns_about_the_grey_axis() {
        let c = Srgb::rgb(0.2, 0.6, 1.0).to::<Oklab>();
        for turn in [0.3, 1.7, -2.0, 3.0] {
            let t = c.rotate_hue(turn);
            assert!((t.lightness() - c.lightness()).abs() < 1e-5);
            assert!((t.chroma() - c.chroma()).abs() < 1e-5);
            let expected = (c.hue() + turn).rem_euclid(core::f32::consts::TAU);
            let diff = (t.hue() - expected).abs();
            assert!(
                diff < 1e-4 || (diff - core::f32::consts::TAU).abs() < 1e-4,
                "{turn}"
            );
        }
        let opposite = coords(c.complement());
        let here = coords(c);
        assert!((opposite[1] + here[1]).abs() < 1e-5 && (opposite[2] + here[2]).abs() < 1e-5);
        let grey = Oklab::grey_of(0.6);
        assert!(near(grey.rotate_hue(1.0), grey, 1e-6));
        // Hue zero is +a in Oklab, red in sRGB.
        assert!(Oklab::new(0.5, 0.1, 0.0, 1.0).hue().abs() < 1e-5);
        assert!(Srgb::rgb(1.0, 0.0, 0.0).hue().abs() < 1e-5);
        let quarter = Oklab::new(0.5, 0.0, 0.1, 1.0).hue();
        assert!(
            (quarter - core::f32::consts::FRAC_PI_2).abs() < 1e-4,
            "{quarter}"
        );
    }

    #[test]
    fn lightness_and_chroma_move_along_and_across_the_axis() {
        let c = Srgb::rgb(0.8, 0.3, 0.2).to::<Oklab>();
        let l = c.with_lightness(0.4);
        assert!((l.lightness() - 0.4).abs() < 1e-6 && (l.chroma() - c.chroma()).abs() < 1e-6);
        assert!((l.hue() - c.hue()).abs() < 1e-5);
        let s = c.with_chroma(0.05);
        assert!((s.chroma() - 0.05).abs() < 1e-6 && (s.lightness() - c.lightness()).abs() < 1e-6);
        let g = c.desaturated(1.0);
        assert!(g.chroma() < 1e-6 && near(g, c.grey(), 1e-6));
        assert!((c.lighter(0.1).lightness() - c.lightness() - 0.1).abs() < 1e-6);
        // In sRGB the axis is the diagonal from black to white: grey has no chroma.
        assert!(Srgb::rgb(0.3, 0.3, 0.3).chroma() < 1e-6);
        assert!((Srgb::rgb(0.3, 0.3, 0.3).lightness() - 0.3).abs() < 1e-6);
    }

    #[test]
    fn mixing_around_the_axis_keeps_to_the_cylinder() {
        let (a, b) = (
            Oklab::new(0.6, 0.1, 0.0, 1.0),
            Oklab::new(0.6, 0.0, 0.1, 1.0),
        );
        let m = a.mix_hue(b, 0.5);
        assert!((m.chroma() - 0.1).abs() < 1e-5, "{}", m.chroma());
        assert!((m.hue() - core::f32::consts::FRAC_PI_4).abs() < 1e-4);
        // Straight mixing cuts the corner.
        assert!(a.mix(b, 0.5).chroma() < 0.08);
        // The shorter way round: from just below a full turn to just above zero.
        let (p, q) = (a.with_hue(6.0), a.with_hue(0.3));
        let h = p.mix_hue(q, 0.5).hue();
        let target =
            (6.0 + 0.5 * (0.3 + core::f32::consts::TAU - 6.0)).rem_euclid(core::f32::consts::TAU);
        assert!((h - target).abs() < 1e-3, "{h} {target}");
    }

    #[test]
    fn luminance_contrast_and_gamut() {
        let (white, black) = (Srgb::rgb(1.0, 1.0, 1.0), Srgb::rgb(0.0, 0.0, 0.0));
        assert!((white.luminance() - 1.0).abs() < 1e-5 && black.luminance().abs() < 1e-6);
        assert!((white.contrast(black) - 21.0).abs() < 1e-3);
        let c = Srgb::rgb(0.2, 0.5, 0.9);
        assert!((c.with_luminance(0.3).luminance() - 0.3).abs() < 1e-5);
        assert!(c.in_gamut());
        // A colour too saturated for sRGB comes back to the faces, keeping its grey and hue.
        let vivid = Oklab::new(0.7, 0.0, -0.4, 1.0);
        assert!(!vivid.in_gamut());
        let shown = vivid.to_gamut();
        assert!(shown.in_gamut());
        let (p, g) = (vivid.to::<LinearRgb>(), shown.to::<LinearRgb>());
        assert!((p.grey().unit() & g.grey().unit()).norm() < 1e-4);
        // On the segment from the grey to the colour.
        let line = p.grey().unit() & p.unit();
        assert!((line & g.unit()).norm() < 1e-4);
    }

    #[test]
    fn hex_codes_and_bytes() {
        let c = Srgb::hex("#ff8000").unwrap();
        assert_eq!(c.to_u8(), [255, 128, 0, 255]);
        assert_eq!(c.to_hex(), "#ff8000");
        assert_eq!(Srgb::hex("f80").unwrap().to_u8(), [255, 136, 0, 255]);
        assert_eq!(Srgb::hex("#ff800080").unwrap().to_hex(), "#ff800080");
        assert_eq!(Srgb::hex("#ff80"), Ok(Srgb::from_u8([255, 255, 136, 0])));
        assert_eq!(Srgb::hex("#12345"), Err(HexError::Length(5)));
        assert_eq!(Srgb::hex("#12x456"), Err(HexError::Digit('x')));
        assert!(near(
            hex(0xff8000, 2.0).to::<Srgb>().with_alpha(1.0),
            c,
            1e-5
        ));
    }

    #[test]
    fn gradients_pass_their_stops() {
        let stops = [palettes::css::RED, palettes::css::LIME, palettes::css::BLUE];
        let g = Gradient::new(stops.map(Oklab::from));
        assert!(near(g.at(0.0), stops[0].into(), 1e-6));
        assert!(near(g.at(0.5), stops[1].into(), 1e-6));
        assert!(near(g.at(1.0), stops[2].into(), 1e-6));
        // Around the grey axis, the middle of red to blue keeps its chroma.
        let around = Gradient::new([stops[0], stops[2]].map(Oklab::from)).around_grey();
        let mid = around.at(0.5).chroma();
        let ends = (Oklab::from(stops[0]).chroma() + Oklab::from(stops[2]).chroma()) / 2.0;
        assert!((mid - ends).abs() < 1e-4);
        let range = Oklab::from(stops[0])..Oklab::from(stops[2]);
        assert!(near(range.at(1.0), stops[2].into(), 1e-6));
    }

    #[test]
    fn the_tonemapper_shows_black_and_white() {
        let shown = LinearRgb::from_point(Point::xyz(0.0, 0.0, 0.0)).agx(1.0);
        assert!(coords(shown).iter().all(|x| *x < 0.01), "{shown:?}");
        let bright = LinearRgb::rgb(1.0, 1.0, 1.0).faded(16.0).agx(1.0);
        // The tonemapper works on radiance, the raw coordinates.
        assert!(bright.point().e032() > 0.9, "{bright:?}");
    }

    /// HSV, HSL and HWB match their textbook definitions (on the channels) and convert back.
    #[test]
    #[allow(clippy::float_cmp)]
    fn the_pickers_models_are_the_textbook_ones() {
        let tau = core::f32::consts::TAU;
        for (r, g, b) in [
            (1.0, 0.0, 0.0),
            (0.9, 0.6, 0.2),
            (0.2, 0.7, 0.4),
            (0.1, 0.3, 0.8),
            (0.7, 0.2, 0.6),
            (0.5, 0.5, 0.5),
            (0.95, 0.9, 0.3),
        ] {
            let c = Srgb::rgb(r, g, b);
            let (max, min) = (r.max(g).max(b), r.min(g).min(b));
            let chroma = max - min;
            // The textbook hue, in sixths of a turn.
            let sixth = if chroma == 0.0 {
                0.0
            } else if max == r {
                ((g - b) / chroma).rem_euclid(6.0)
            } else if max == g {
                (b - r) / chroma + 2.0
            } else {
                (r - g) / chroma + 4.0
            };
            let hsv = Hsv::from(c);
            assert!((hsv.value - max).abs() < 1e-5, "{c:?} {hsv:?}");
            assert!((hsv.hue - sixth / 6.0 * tau).abs() < 1e-4, "{c:?} {hsv:?}");
            let s = if max > 0.0 { chroma / max } else { 0.0 };
            assert!((hsv.saturation - s).abs() < 1e-5, "{c:?} {hsv:?}");
            let hwb = Hwb::from(c);
            assert!(
                (hwb.whiteness - min).abs() < 1e-5 && (hwb.blackness - (1.0 - max)).abs() < 1e-5
            );
            let hsl = Hsl::from(c);
            assert!((hsl.lightness - (max + min) / 2.0).abs() < 1e-5);
            for back in [Srgb::from(hsv), Srgb::from(hsl), Srgb::from(hwb)] {
                assert!(near(back, c, 1e-4), "{c:?} -> {back:?}");
            }
        }
    }

    /// Chromaticity is the ray's meet with the unit plane: D65 white sits at its published
    /// point, a colour and its dilations share it, and it gives the colour back with its
    /// luminance. Black bodies run from orange to blue, and McCamy reads their temperature.
    #[test]
    fn chromaticity_and_temperature() {
        let white = Srgb::rgb(1.0, 1.0, 1.0);
        let xy = white.chromaticity().unwrap().to_euclidean();
        assert!(
            close([xy[0], xy[1], 0.0], [0.3127, 0.3290, 0.0], 1e-3),
            "{xy:?}"
        );
        let c = LinearRgb::rgb(0.6, 0.3, 0.1);
        let dimmer = LinearRgb::rgb(0.3, 0.15, 0.05);
        let (a, b) = (c.chromaticity().unwrap(), dimmer.chromaticity().unwrap());
        assert!((a & b).norm() < 1e-6);
        let back = Xyz::from_chromaticity(a, c.luminance());
        assert!(near(back.to::<LinearRgb>(), c, 1e-5));
        let warm = LinearRgb::from_temperature(2700.0, 1.0);
        let cool = LinearRgb::from_temperature(9000.0, 1.0);
        let [wr, _, wb] = coords(warm);
        let [cr, _, cb] = coords(cool);
        assert!(wr > wb && cb > cr, "{warm:?} {cool:?}");
        for kelvin in [3000.0, 5000.0, 6500.0, 8000.0] {
            let read = LinearRgb::from_temperature(kelvin, 1.0)
                .temperature()
                .unwrap();
            assert!(
                (read - kelvin).abs() < kelvin * 0.02,
                "{kelvin} read as {read}"
            );
        }
        assert!(Srgb::rgb(0.0, 0.0, 0.0).chromaticity().is_none());
    }

    #[test]
    fn any_colour_converts_on_demand() {
        let c = Srgb::rgb(0.2, 0.5, 0.9);
        let any = AnyColour::from(Oklab::from(c));
        assert!(near(any.to::<Srgb>(), c, 1e-4));
    }
}
