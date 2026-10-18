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

use gax_numga_examples::{Anim, Axes, Canvas, Light, Point2, Rect, backdrop, caption, light, run};

/// The seconds each scene shows.
const ACT: f32 = 7.0;
/// A scene: its title, its caption, and its drawing at its own time, from 0 to [`ACT`].
type Scene = (&'static str, &'static str, fn(&mut Canvas, f32));
/// The scenes, in turn.
const ACTS: [Scene; 4] = [
    (
        "REFLECTIONS: A KALEIDOSCOPE",
        "TWO MIRRORS AT A HALF TURN OVER N, THEIR ROTOR'S N-TH POWER A SCALAR (VGA2D)",
        kaleidoscope::draw,
    ),
    (
        "REFLECTIONS: THREE MIRRORS ON THE SPHERE",
        "AT HALF TURNS OVER 2, 3 AND 5 THEY CUT IT INTO 120 TRIANGLES (VGA3D)",
        icosahedron::draw,
    ),
    (
        "REFLECTIONS: THE 600-CELL",
        "THE PRODUCTS OF THE MIRRORS' ROTORS: 120 UNIT SPINORS ON THE THREE-SPHERE (VGA3D)",
        six_hundred_cell::draw,
    ),
    (
        "REFLECTIONS: SPHERE, PLANE, HYPERBOLIC PLANE",
        "THE ANGLE SUM PICKS THE SIGNATURE: X+Y+Z+, X+Y+W0, X+Y+T-",
        tilings::draw,
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
        Point2::xy(0.0, 0.0),
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

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let act = ((t / ACT) as usize).min(ACTS.len() - 1);
    let (title, sub, scene) = ACTS[act];
    scene(c, t - act as f32 * ACT);
    caption(c, title, sub);
}

fn main() {
    run(
        Anim::new("reflections", ACT * ACTS.len() as f32).size(960, 540),
        draw,
    );
}

#[cfg(test)]
mod tests {
    #[test]
    fn every_scene_draws() {
        for act in 0..super::ACTS.len() {
            gax_numga_examples::app::assert_draws(super::draw, (act as f32 + 0.4) * super::ACT);
        }
    }
}
