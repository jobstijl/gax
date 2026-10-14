//! numga's `tilings` notebook: triangles at half turns over 2, 3 and 5, 6 or 7 tile the
//! sphere, the flat plane and the hyperbolic plane ([`super::mirrors`] builds their mirrors in
//! the algebra the angle sum picks). Each pixel of the disk is a point of the geometry; folded
//! into the triangle by up to twenty rounds of reflections, the parity of its reflections
//! colours its tile, and a side is drawn where it pairs with a mirror close to zero, relative
//! to its pairings with all three. The three views move along `x` together by one rotor each.

use super::mirrors::curvature;
use super::{fold, tile};
use gax_numga_examples::signal::haversine;
use gax_numga_examples::{Align, Canvas, Light, Point2, palette};

/// The geometries: where the second and third mirrors meet (a half turn over `m`), and the
/// name of what they tile.
const GEOMETRIES: [(f64, &str); 3] = [
    (5.0, "SPHERE, PI/5"),
    (6.0, "PLANE, PI/6"),
    (7.0, "HYPERBOLIC PLANE, PI/7"),
];
/// Enough rounds of reflections to fold every pixel into the triangle, except a few at the rim
/// of the hyperbolic disk, where the tiles are smaller than a pixel.
const ROUNDS: usize = 20;
/// How far the views move: an angle, a distance, a rapidity.
const TRAVEL: f64 = 1.2;
/// The width of a tile's side, relative to the sum of a point's pairings with the mirrors.
const SIDE: f64 = 0.02;

/// A view of the disk folded into a geometry's triangle: for a point `(u, v)` of the disk, the
/// folded point's pairings with the three mirrors, and how many reflections it took.
pub type Folding = Box<dyn Fn(f64, f64) -> ([f64; 3], u32) + Sync>;

/// The view of the triangle with angles a half turn over 2, 3 and `m`, moved along `x` by
/// `travel`, in the algebra its angle sum picks (numga's `SIGNATURES`).
pub fn folding(m: f64, travel: f64) -> Folding {
    macro_rules! in_algebra {
        ($geometry:ident) => {{
            use super::mirrors::$geometry::{mirrors, motion, reflect, side, view};
            let (mirrors, motion) = (mirrors(m), motion(travel));
            Box::new(move |u: f64, v: f64| {
                let (point, flips) = fold(motion >> view(u, v), &mirrors, ROUNDS, side, reflect);
                (mirrors.map(|mirror| side(mirror, point)), flips)
            })
        }};
    }
    match curvature(m) {
        1 => in_algebra!(sphere),
        0 => in_algebra!(flat),
        _ => in_algebra!(hyperbolic),
    }
}

/// A folded point's colour: its tile's, darkened where it lies near a mirror relative to its
/// pairings with all three; nothing where the fold did not settle.
fn colour(sides: [f64; 3], flips: u32) -> Option<Light> {
    let magnitudes = sides.map(f64::abs);
    let total: f64 = magnitudes.iter().sum();
    let nearest = magnitudes.into_iter().fold(f64::INFINITY, f64::min);
    let nearness = 1.0 - nearest / total / SIDE;
    nearness.is_finite().then(|| tile(flips, nearness))
}

/// The three disks of tiles at time `t` of the act, moving along `x` and back.
pub fn draw(c: &mut Canvas, t: f32) {
    let phase = core::f64::consts::TAU * f64::from(t / super::ACT);
    let travel = TRAVEL * haversine(phase);
    let screen = c.rect();
    for (k, (m, name)) in GEOMETRIES.into_iter().enumerate() {
        let column = screen.part(k as f32 / 3.0, 0.16, (k + 1) as f32 / 3.0, 0.92);
        let ax = super::disk_axes(column.inset(8.0, 8.0, 8.0, 30.0));
        let folding = folding(m, travel);
        ax.image(c, 1, |p| {
            let [u, v] = p.to_euclidean().map(f64::from);
            if u * u + v * v >= 1.0 {
                return None;
            }
            let (sides, flips) = folding(u, v);
            colour(sides, flips)
        });
        ax.text(
            c,
            Point2::xy(0.0, -1.15),
            name,
            11.0,
            palette::ink(),
            Align::Center,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// numga's check, at both ends of the motion: every folded point inside the disk (within
    /// 0.98 of its rim) lies on the right side of all three mirrors.
    #[test]
    fn every_point_of_the_disk_folds_into_the_triangle() {
        for travel in [0.0, TRAVEL] {
            for (m, _) in GEOMETRIES {
                let folding = folding(m, travel);
                for (u, v) in super::super::disk(240) {
                    if u * u + v * v >= 0.98 * 0.98 {
                        continue;
                    }
                    let (sides, _) = folding(u, v);
                    assert!(
                        sides.iter().all(|s| s.is_finite() && *s > -1e-9),
                        "{m}: {sides:?}"
                    );
                }
            }
        }
    }
}
