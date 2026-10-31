//! The look of numga's ray-traced quadric scenes (`quadrics_csg_raytracer`,
//! `quadrics_glass_cyclide`), on the CPU: the place on the picture a pixel's ray goes through,
//! the checker laid over a solid by two planes' unit steps, the light of a lit surface, and the
//! sky a reflection sees, all as lights on the shared navy look. The geometry of the rays stays
//! with each example, in its own algebra.

use gax_colour::Light;
use gax_numga_examples::{Point2, Rect, palette};

/// numga's `screen`: a place on the picture as a direction across the view, right and up, in
/// half heights of the picture from its centre.
pub fn screen(q: Point2, picture: Rect) -> [f64; 2] {
    let offset = q.unitized() - picture.lo.unitized();
    let (x, y) = (offset.e20(), offset.e01());
    let (w, h) = (f64::from(picture.width()), f64::from(picture.height()));
    [(2.0 * f64::from(x) - w) / h, 1.0 - 2.0 * f64::from(y) / h]
}

/// Which square of a checker a point lies on: the parity of the whole unit steps of its
/// pairings with the checker's two planes.
pub fn parity(across: f64, along: f64) -> f64 {
    ((across.floor() + along.floor()) % 2.0).abs()
}

/// How a solid is lit: the light where the lamp does not reach, and the checker's darkening.
#[derive(Clone, Copy)]
pub struct Look {
    pub ambient: f64,
    pub checker: f64,
}

impl Look {
    /// A surface's light: its colour darkened on every other square of its checker, and lit by
    /// the lamp's share `lit` above the ambient light.
    pub fn surface(self, colour: Light, parity: f64, lit: f64) -> Light {
        let k = (1.0 - self.checker * parity) * (self.ambient + (1.0 - self.ambient) * lit);
        colour.faded(k as f32)
    }
}

/// The sky a reflection sees along a direction, by how far it looks up (its pairing with the
/// floor's plane): the backdrop's navy, lighter towards the zenith.
pub fn sky(up: f64) -> Light {
    palette::BOTTOM.mix_light(palette::TOP, up.clamp(0.0, 1.0) as f32)
}

/// A glint: the light of a highlight of brightness `k`.
pub fn glint(k: f64) -> Light {
    palette::INK.whitened(0.6).faded(k as f32)
}
