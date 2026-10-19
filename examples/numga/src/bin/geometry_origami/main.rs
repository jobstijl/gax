//! numga's `geometry/origami`: a square of paper folded into a samurai helmet, the eleven-stage
//! Kabuto sequence of the origami example of ganja.js by Steven De Keninck, in PGA3D.
//!
//! The creases are constructed as planes. Folding one point onto another creases along their
//! perpendicular bisector, the inner product of their join with their sum; folding one line
//! onto another creases along the plane through their crossing orthogonal to the sum of the
//! unit lines. The paper is a list of convex faces; a crease plane splits the selected faces by
//! the signs of its regressive products with their corners (the signed distances), meeting each
//! crossing edge with the plane, into a moving and a stationary piece that share the crease's
//! points. The fold is the motor `exp(hinge θ/2)` about the crease's meet with the sheet,
//! applied to the moving pieces as a batch; the others turn by zero. Two folds reopen to leave
//! a crease for the horns. Each face also carries an infinitesimal offset off the sheet,
//! counted in layers and carried rigidly through the folds, which tells the renderer which of
//! two coincident layers lies on top (`picture.rs`).
//!
//! The animation folds the helmet stage by stage, seen from numga's view.

mod kabuto;
mod paper;
mod picture;
#[cfg(test)]
mod tests;

use gax_numga_examples::{Align, Anim, Canvas, Point2, Rect, backdrop, caption, palette, run};
use kabuto::{Stage, stages};
use paper::{P, Paper};
use picture::{Picture, View};
use std::sync::OnceLock;

/// numga's view: its elevation and azimuth, in degrees.
const ELEVATION: f64 = 35.0;
const AZIMUTH: f64 = 110.0;
/// Seconds per fold, and the hold on the finished helmet.
const PER_FOLD: f32 = 1.7;
const HOLD: f32 = 2.6;
/// The stages' names.
const NAMES: [&str; 11] = [
    "CORNER TO CORNER",
    "CORNER TO CORNER AGAIN",
    "BOTH FLAPS DOWN TO THE TIP",
    "THE TIPS UP TO THE TOP",
    "CREASE THE RIGHT HORN, REOPEN",
    "CREASE THE LEFT HORN, REOPEN",
    "OPEN THE LEFT HORN",
    "OPEN THE RIGHT HORN",
    "FOLD THE FRONT FLAP UP",
    "TURN THE BRIM UP",
    "THE LAST FLAP BEHIND",
];

/// The folds, found once, and one view that frames the paper through all of them.
fn recipe() -> &'static (Vec<Stage>, View) {
    static RECIPE: OnceLock<(Vec<Stage>, View)> = OnceLock::new();
    RECIPE.get_or_init(|| {
        let stages = stages();
        let points: Vec<P> = stages
            .iter()
            .flat_map(|s| [s.at(s.peak), s.at(s.end)])
            .flat_map(|paper: Paper| paper.points)
            .collect();
        let view = View::new(ELEVATION, AZIMUTH, &points);
        (stages, view)
    })
}

/// The fold under way at time `t`, and the paper then.
fn paper_at(t: f32) -> (usize, Paper) {
    let (stages, _) = recipe();
    let fold = ((t / PER_FOLD) as usize).min(stages.len() - 1);
    let fraction = f64::from((t / PER_FOLD - fold as f32).clamp(0.0, 1.0));
    let stage = &stages[fold];
    (fold, stage.at(stage.angle(fraction)))
}

/// The paper in a square of `rect`, pixel by pixel.
fn draw_paper(c: &mut Canvas, rect: Rect, paper: &Paper) {
    let (_, view) = recipe();
    let size = rect.width().min(rect.height());
    let corner = rect.centre() - Point2::direction(0.5 * size, 0.5 * size);
    let picture = Picture::of(paper, view, f64::from(size));
    c.clip(rect);
    c.shade(2, |q| {
        let offset = q - corner;
        picture.at(f64::from(offset.e20()), f64::from(offset.e01()))
    });
    c.unclip();
}

/// The recipe down the side, the fold under way lit.
fn draw_recipe(c: &mut Canvas, at: Point2, current: usize) {
    for (k, name) in NAMES.iter().enumerate() {
        let colour = match k.cmp(&current) {
            core::cmp::Ordering::Less => palette::INK.faded(0.5),
            core::cmp::Ordering::Equal => palette::ORANGE,
            core::cmp::Ordering::Greater => palette::INK.faded(0.25),
        };
        let line = format!("{:>2}  {name}", k + 1);
        c.text(
            &line,
            at + Point2::direction(0.0, 24.0 * k as f32),
            11.0,
            colour,
            Align::Left,
        );
    }
}

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let screen = c.rect();
    let (fold, paper) = paper_at(t);
    draw_paper(c, screen.part(0.0, 0.1, 0.62, 1.0), &paper);
    draw_recipe(c, screen.part(0.6, 0.22, 1.0, 1.0).lo, fold);
    caption(
        c,
        "ORIGAMI: FOLDING A KABUTO HELMET",
        "CREASES ARE BISECTOR PLANES; EACH FOLD IS ONE MOTOR ABOUT THE CREASE (PGA3D)",
    );
}

fn main() {
    let folds = recipe().0.len() as f32;
    run(
        Anim::new("origami", folds * PER_FOLD + HOLD).size(960, 540),
        draw,
    );
}
