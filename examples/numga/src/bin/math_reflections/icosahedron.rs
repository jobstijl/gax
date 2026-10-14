//! numga's `icosahedron` notebook: three mirrors on the sphere, VGA3D.
//!
//! Three mirrors through the centre of a sphere, at angles of a half turn over 2, 3 and 5 to
//! each other, cut out a triangle on it ([`super::mirrors`]). Each pair meets in an axis, and
//! the product of their vectors is a rotor about it whose powers return to a scalar after 2, 3
//! or 5 steps. Reflected in one another over and over, the mirrors cut the whole sphere into
//! 120 copies of the triangle, coloured by the parity of the reflections that fold each point
//! into it. The six copies around each threefold axis make a face of an icosahedron with its
//! corners on the fivefold axes; the second mirror, from a fivefold axis to a twofold one,
//! draws its edges, bold.

use super::mirrors::sphere;
use super::{fold, tile};
use gax::vga3d::{Bivector, Rotor, Vector};
use gax_numga_examples::{Align, Canvas, Point2, measure::lift, palette};

pub type V = Vector<(), f64>;
pub type R = Rotor<(), f64>;

/// The second and third mirrors meet at a half turn over five.
const FIVEFOLD: f64 = 5.0;
/// Five rounds of reflections in the three mirrors fold any point of the sphere into the
/// triangle.
const ROUNDS: usize = 5;
/// The view's tilt towards the viewer, about `x`.
const TILT: f64 = 0.4;
/// The widths of the thin sides of the tiles and of the bold ones, as inner products.
const THIN: f64 = 0.012;
const BOLD: f64 = 0.03;

/// The icosahedral mirrors.
pub fn mirrors() -> [V; 3] {
    sphere::mirrors(FIVEFOLD)
}

/// The turns about the axes where two mirrors meet, each with how many of them make a scalar.
pub fn turns([first, second, third]: [V; 3]) -> [(R, usize); 3] {
    [(second * first, 2), (third * first, 3), (third * second, 5)]
}

/// A point of the sphere folded into the triangle, and how many reflections it took.
pub fn folded(mirrors: &[V; 3], point: V) -> (V, u32) {
    fold(point, mirrors, ROUNDS, |m, p| (m | p).s(), |m, p| -(m >> p))
}

/// The view at `angle` of the sphere's half turn about the axis where the first two mirrors
/// meet, a symmetry of the tiling, tilted towards the viewer.
pub fn view(angle: f64) -> R {
    let [first, second, _] = mirrors();
    let spin = (first ^ second).gp(-angle / 2.0).exp();
    let tilt = Bivector::new(1.0, 0.0, 0.0).gp(-TILT / 2.0).exp();
    (spin * tilt).into_inner()
}

/// The sphere's colour at a point of its visible half: the tile's, darkened along the mirrors
/// and along the outlining one, lit from the viewer's upper left.
fn shade(mirrors: &[V; 3], turned: V, screen: V) -> gax_numga_examples::Light {
    let (folded, flips) = folded(mirrors, turned);
    let nearest = mirrors
        .iter()
        .map(|m| (*m | folded).s().abs())
        .fold(f64::INFINITY, f64::min);
    let thin = (1.0 - nearest / THIN).clamp(0.0, 1.0);
    let bold = (1.0 - (mirrors[1] | folded).s().abs() / BOLD).clamp(0.0, 1.0);
    let sun = Vector::new(-0.4, 0.5, 0.77);
    let lit = 0.35 + 0.65 * (screen | sun).s().clamp(0.0, 1.0);
    tile(flips, (0.6 * thin).max(bold)).faded(lit as f32)
}

/// The turning sphere at time `t` of the act.
pub fn draw(c: &mut Canvas, t: f32) {
    let angle = core::f64::consts::PI * f64::from(t / super::ACT);
    let (mirrors, view) = (mirrors(), view(angle));
    let rect = c.rect().part(0.2, 0.14, 0.8, 0.96);
    let ax = super::disk_axes(rect);
    ax.image(c, 1, |p| {
        let [u, v] = p.to_euclidean().map(f64::from);
        let screen = lift(u, v)?;
        Some(shade(&mirrors, view >> screen, screen))
    });
    let notes = [
        "120 TRIANGLES",
        "MIRRORS AT PI/2, PI/3, PI/5",
        "BOLD: ICOSAHEDRON EDGES",
    ];
    for (k, note) in notes.into_iter().enumerate() {
        let at = Point2::xy(24.0, 120.0 + 22.0 * k as f32);
        c.text(note, at, 11.0, palette::ink().faded(0.7), Align::Left);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// numga's checks: the mirror vectors have unit length (their pairings are checked with
    /// the construction); each turn's power of its order is a scalar of size one; every folded
    /// point of the visible half lies on the right side of all three mirrors.
    #[test]
    fn the_turns_close_and_the_sphere_folds_into_the_triangle() {
        let mirrors = mirrors();
        for m in mirrors {
            assert!((m.norm_squared() - 1.0).abs() < 1e-12);
        }
        for (turn, order) in turns(mirrors) {
            let power = (1..order).fold(turn, |p, _| p * turn);
            assert!((power.s().abs() - 1.0).abs() < 1e-12);
            assert!(power.grade::<2>().norm() < 1e-12);
        }
        let view = view(0.0);
        for (u, v) in super::super::disk(240) {
            let Some(screen) = lift(u, v) else { continue };
            let (point, _) = folded(&mirrors, view >> screen);
            for m in mirrors {
                assert!((m | point).s() > -1e-12);
            }
        }
    }
}
