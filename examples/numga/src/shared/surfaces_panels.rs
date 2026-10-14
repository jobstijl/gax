//! What numga's `surfaces/projective_loop`, `surfaces/motor_loop`, `surfaces/koebe` and
//! `surfaces/weighted_dual` share in laying out their panels: a row of them under the caption,
//! and a label under each.

use gax_numga_examples::{Align, Canvas, Point2, Rect, font, palette};

/// The share of the canvas's height the caption takes at its top.
const CAPTION: f32 = 0.12;

/// `N` panels in a row under the caption, each inset by the margin on every side.
pub fn row<const N: usize>(c: &Canvas, margin: f32) -> [Rect; N] {
    let screen = c.rect();
    let below = Rect::new(
        0.0,
        screen.height() * CAPTION,
        screen.width(),
        screen.height(),
    );
    core::array::from_fn(|i| below.column(i, N).inset(margin, margin, margin, margin))
}

/// A label under the panel's middle, shrunk to fit its width.
pub fn label_under(c: &mut Canvas, rect: Rect, label: &str) {
    let fitting = 0.9 * rect.width() / font::width(label, 1.0);
    let size = (rect.height() / 30.0).clamp(7.0, 12.0).min(fitting);
    let at = rect.bottom_middle() - Point2::direction(0.0, size * 0.6);
    c.text(label, at, size, palette::ink(), Align::Center);
}
