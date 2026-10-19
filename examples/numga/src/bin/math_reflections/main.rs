//! numga's `math/reflections` (four notebooks): what mirrors make when they reflect each other.
//!
//! A mirror through the origin is a unit vector normal to it, and reflecting in it is `>>`
//! with that vector, the sandwich with the orientation's sign (ADR-051); the product of two
//! mirror vectors is a rotor turning by twice the angle between the mirrors. Mirrors at a half turn divided by whole numbers fold
//! all of space into the region between them: a point on the wrong side of a mirror (pairing
//! negatively with it) is reflected, round after round, and the number of reflections it took
//! tells which copy of the region it started in.
//!
//! * **Kaleidoscope** (`kaleidoscope.rs`): two mirrors of the plane at `π/n`, their rotor's
//!   `n`-th power a scalar; beads tumbling between them seen in every reflection.
//! * **Icosahedron** (`icosahedron.rs`): three mirrors through the centre of a sphere at `π/2`,
//!   `π/3` and `π/5`, cutting it into 120 triangles; one mirror outlines the icosahedron.
//! * **The 600-cell** (`six_hundred_cell.rs`): the products of the icosahedron's rotors close
//!   into a group of 120 unit spinors, the vertices of the 600-cell on the three-sphere.
//! * **Tilings** (`tilings.rs`): triangles at `π/2`, `π/3` and `π/m` tile the sphere, the
//!   plane or the hyperbolic plane as their angles sum to more than a half turn, to one, or to
//!   less: the angle sum picks the signature of the algebra (`mirrors.rs`).
//!
//! The animation shows the four scenes in turn.

mod icosahedron;
mod kaleidoscope;
mod mirrors;
mod six_hundred_cell;
mod tilings;

use gax_numga_examples::{
    Anim, Axes, Canvas, Light, ORIGIN2, Rect, backdrop, caption, light, run_with,
};
use six_hundred_cell::Polytope;

/// The seconds each scene shows.
const ACT: f32 = 7.0;
/// A scene: its title and its caption.
type Scene = (&'static str, &'static str);
/// The scenes, in turn (drawn by [`draw`]).
const ACTS: [Scene; 4] = [
    (
        "REFLECTIONS: A KALEIDOSCOPE",
        "TWO MIRRORS AT A HALF TURN OVER N, THEIR ROTOR'S N-TH POWER A SCALAR (VGA2D)",
    ),
    (
        "REFLECTIONS: THREE MIRRORS ON THE SPHERE",
        "AT HALF TURNS OVER 2, 3 AND 5 THEY CUT IT INTO 120 TRIANGLES (VGA3D)",
    ),
    (
        "REFLECTIONS: THE 600-CELL",
        "THE PRODUCTS OF THE MIRRORS' ROTORS: 120 UNIT SPINORS ON THE THREE-SPHERE (VGA3D)",
    ),
    (
        "REFLECTIONS: SPHERE, PLANE, HYPERBOLIC PLANE",
        "THE ANGLE SUM PICKS THE SIGNATURE: X+Y+Z+, X+Y+W0, X+Y+T-",
    ),
];

/// Reflect a point in each mirror it lies on the wrong side of (where `side` is negative),
/// round after round: the folded point, and how many reflections it took.
pub fn fold<M: Copy, P: Copy>(
    point: P,
    mirrors: &[M],
    rounds: usize,
    side: impl Fn(M, P) -> f64,
    reflect: impl Fn(M, P) -> P,
) -> (P, u32) {
    let (mut point, mut flips) = (point, 0);
    for _ in 0..rounds {
        for mirror in mirrors {
            if side(*mirror, point) < 0.0 {
                point = reflect(*mirror, point);
                flips += 1;
            }
        }
    }
    (point, flips)
}

/// The pixel centres of numga's square view of the unit disk, `resolution` on a side, rows
/// from the top: the coordinates `(u, v)`.
pub fn disk(resolution: usize) -> impl Iterator<Item = (f64, f64)> {
    let centre = move |k: usize| (k as f64 + 0.5) / resolution as f64 * 2.0 - 1.0;
    (0..resolution * resolution).map(move |i| (centre(i % resolution), -centre(i / resolution)))
}

/// Axes on the unit disk centred in `rect`, with a little room around it whatever the
/// rectangle's shape.
pub fn disk_axes(rect: Rect) -> Axes {
    const ROOM: f32 = 1.03;
    Axes::equal(
        rect,
        ORIGIN2,
        ROOM * (rect.height() / rect.width()).max(1.0),
    )
}

/// The colour of a tile by the parity of its reflections, an even number of them cream and an
/// odd number blue, darkened along the mirrors by the nearness to the closest one (one on it,
/// zero away from it).
pub fn tile(flips: u32, nearness: f64) -> Light {
    let base = if flips.is_multiple_of(2) {
        Light::from_srgb(0.93, 0.83, 0.62, 1.1)
    } else {
        Light::from_srgb(0.35, 0.55, 0.72, 1.4)
    };
    base.mix_light(
        light(0.01, 0.01, 0.03, 1.0),
        nearness.clamp(0.0, 1.0) as f32,
    )
}

/// The scene showing at `t`, at its own time, from 0 to [`ACT`].
fn draw(c: &mut Canvas, polytope: &Polytope, t: f32) {
    backdrop(c);
    let act = ((t / ACT) as usize).min(ACTS.len() - 1);
    let at = t - act as f32 * ACT;
    match act {
        0 => kaleidoscope::draw(c, at),
        1 => icosahedron::draw(c, at),
        2 => six_hundred_cell::draw(c, polytope, at),
        _ => tilings::draw(c, at),
    }
    let (title, sub) = ACTS[act];
    caption(c, title, sub);
}

fn main() {
    run_with(
        Anim::new("reflections", ACT * ACTS.len() as f32).size(960, 540),
        Polytope::new,
        draw,
    );
}

#[cfg(test)]
mod tests {
    #[test]
    fn every_scene_draws() {
        let polytope = super::Polytope::new();
        for act in 0..super::ACTS.len() {
            let t = (act as f32 + 0.4) * super::ACT;
            gax_numga_examples::app::assert_draws_with(super::draw, &polytope, t);
        }
    }
}
