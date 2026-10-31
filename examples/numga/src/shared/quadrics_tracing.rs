//! The look of numga's ray-traced quadric scenes (`quadrics_csg_raytracer`,
//! `quadrics_glass_cyclide`), on the CPU: the place on the picture a pixel's ray goes through,
//! the checker laid over a solid by two planes' unit steps, the light of a lit surface, and the
//! sky a reflection sees, all as lights on the shared navy look; and, for the solids bounded by
//! PGA3D quadrics (`quadrics_csg_raytracer`, `mechanics_rattleback`), numga's `quadric_regions`:
//! where a ray crosses a quadric, and the first point where it meets an intersection of them.

// Each example that includes this module uses a part of it.
#![allow(dead_code)]

use gax::pga3d::{Plane, Point};
use gax_colour::Light;
use gax_numga_examples::{Point2, Rect, palette};

/// A solid quadric of PGA3D: from points to their polar planes, its form `quadric(p) & p`
/// negative inside.
pub type Quadric = Plane<(Point,)>;

/// Distances beyond the scene, the step a ray takes off a surface before it looks again, and how
/// far a point may lie outside a quadric and still count as inside, for points on two surfaces.
pub const FAR: f64 = 1e6;
pub const CLEARANCE: f64 = 1e-3;
pub const TOLERANCE: f64 = 1e-4;

/// A quadric along a ray, from `start` along `direction`: its pairing with the ray's point, a
/// quadratic in the distance `t` along it, `a t² + 2 b t + c`.
pub fn along_ray(quadric: Quadric, start: Point, direction: Point) -> [f64; 3] {
    let (ahead, here) = (quadric.of(direction), quadric.of(start));
    [
        (ahead & direction).s(),
        ((ahead & start).s() + (here & direction).s()) / 2.0,
        (here & start).s(),
    ]
}

/// Where a ray crosses a quadric's surface, the roots of the quadratic along it, where it has
/// them: both without cancellation; a half-space, with no square term, has only the second, the
/// first at `FAR`.
pub fn crossings([a, b, c]: [f64; 3]) -> Option<[f64; 2]> {
    let discriminant = b * b - a * c;
    if discriminant < 0.0 {
        return None;
    }
    #[allow(clippy::disallowed_methods)] // the quadratic's roots
    let q = -(b + b.signum() * discriminant.sqrt());
    Some([if a == 0.0 { FAR } else { q / a }, c / q])
}

/// The first point between two distances where a ray meets the surface of an intersection of
/// quadrics: on one of them, inside all the others. Its distance, the tangent plane there and
/// the quadric's index.
pub fn region_hit(
    quadrics: &[Quadric],
    start: Point,
    direction: Point,
    nearest: f64,
    farthest: f64,
) -> Option<(f64, Plane, usize)> {
    let mut hit = None;
    let mut bound = farthest;
    for (own, quadric) in quadrics.iter().enumerate() {
        let Some(roots) = crossings(along_ray(*quadric, start, direction)) else {
            continue;
        };
        for distance in roots {
            let point = start + direction * distance;
            let inside = quadrics
                .iter()
                .enumerate()
                .all(|(k, q)| k == own || (q.of(point) & point).s() <= TOLERANCE);
            if distance > nearest && distance < bound && inside {
                bound = distance;
                hit = Some((distance, quadric.of(point), own));
            }
        }
    }
    hit
}

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
