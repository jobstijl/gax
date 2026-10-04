//! numga's `quadrics/conformal_elliptical`: the thirteen mirror planes of the octahedral group
//! drawn on the unit sphere in Cl(3). A plane through the origin is a vector, its normal; a
//! point of the sphere lies on the plane's great circle where its inner product with the normal
//! vanishes, and the sign of that product tells the two hemispheres apart. The product of the
//! thirteen signs (softened by `tanh` to antialias) colours the cells of the arrangement like a
//! chessboard. The animation turns the sphere by the two generators of the cube's rotations, a
//! quarter turn about an axis and then a third turn about a body diagonal; each ends on the
//! picture it started from, because each carries every mirror plane onto a mirror plane.

use gax::vga3d::Vector;

use gax_numga_examples::disc::Disc;
use gax_numga_examples::{Anim, Canvas, Point2, backdrop, caption, palette, run};

mod elliptical {
    use gax::vga3d::{Bivector, Rotor, Vector};
    use gax_numga_examples::signal::phasor;

    pub type V = Vector<(), f64>;
    pub type R = gax::Unit<Rotor<(), f64>>;

    /// All 13 planes of the octahedral symmetry group: normals along the axes and the face and
    /// body diagonals. numga takes the first 13 of the 27 sign triples of `meshgrid([1, 0, -1])`,
    /// the ones before `(0, 0, 0)`: no two of them are opposite.
    pub fn octahedral_planes() -> Vec<V> {
        let vals = [1.0, 0.0, -1.0];
        (0..13)
            .map(|k| {
                // `meshgrid`'s default `xy` indexing: x varies along the second axis.
                let (a, b, c) = (k / 9, (k / 3) % 3, k % 3);
                Vector::new(vals[b], vals[a], vals[c])
                    .normalized()
                    .into_inner()
            })
            .collect()
    }

    /// The rotor `exp(B angle)` for a bivector `B` normalized first: a turn by twice `angle`.
    pub fn turn(b: Bivector<(), f64>, angle: f64) -> R {
        (b.normalized().into_inner() * angle).exp()
    }

    /// A quarter turn about z, `exp(xy π/4)`.
    pub fn quarter() -> R {
        turn(Bivector::new(0.0, 0.0, 1.0), core::f64::consts::FRAC_PI_4)
    }

    /// A third turn about the body diagonal, `exp((xy + yz + zx)/√3 π/3)`.
    pub fn third() -> R {
        turn(Bivector::new(1.0, 1.0, 1.0), core::f64::consts::FRAC_PI_3)
    }

    /// The planes at phase `t` (radians), seen tilted: over the first half of the loop a
    /// quarter turn about z, over the second a third turn about the body diagonal, each eased in
    /// and out. Both end where they started, since each carries the arrangement onto itself.
    pub fn planes_at(t: f64) -> Vec<V> {
        let s = (t / core::f64::consts::TAU).rem_euclid(1.0);
        let (generator, along) = if s < 0.5 {
            (quarter(), s * 2.0)
        } else {
            (third(), s * 2.0 - 1.0)
        };
        let eased = 0.5 - 0.5 * phasor(core::f64::consts::PI * along).e20();
        // The generator's logarithm scaled: a fraction of its turn.
        let partial: R = (generator.log() * eased).exp();
        let tilt = turn(Bivector::new(0.5, -0.3, 0.15), 0.45);
        octahedral_planes()
            .into_iter()
            .map(|p| tilt >> (partial >> p))
            .collect()
    }

    /// The softened product of the signs of `p | plane`, in `[-1, 1]`: `sharpness` sets the
    /// width of a circle.
    pub fn sides(p: V, planes: &[V], sharpness: f64) -> f64 {
        planes
            .iter()
            .map(|n| ((p | *n).s() * sharpness).tanh())
            .product()
    }
}

use elliptical::*;

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let screen = c.rect();
    let phase = f64::from(t) / 16.0 * core::f64::consts::TAU;
    let planes = planes_at(phase);
    // The sphere's disk, a little below the middle of the screen.
    let centre = screen.centre() + Point2::direction(0.0, screen.height() * 0.04);
    let radius = screen.height() * 0.42;
    let sharpness = 2.0 * f64::from(radius);
    // The cells cover the disk, so they shine far less than a stroke.
    let (cell_a, cell_b) = (palette::sky().faded(0.3), palette::blue().faded(0.15));
    // Lit from the upper left, so the disk reads as a sphere: the inner product of the normal
    // (on the unit sphere, the point itself) with the direction to the light.
    let light = Vector::new(-0.35, 0.45, 0.82);
    // The front hemisphere under each pixel, seen along -z: x right, y up, z towards the viewer.
    let disc = Disc::new(centre, radius);
    c.shade(2, |q: Point2| {
        let p = disc.point(q)?;
        let s = sides(p, &planes, sharpness) as f32;
        let colour = cell_b.mix_light(cell_a, (s + 1.0) / 2.0);
        let lit = (0.55 + 0.45 * (p | light).s()).clamp(0.2, 1.0);
        Some(colour.faded(lit as f32))
    });
    c.ring(centre, radius, 1.5, palette::ink().faded(0.6));
    caption(
        c,
        "THE OCTAHEDRAL MIRROR PLANES ON THE UNIT SPHERE",
        "13 PLANES AS VECTORS N; P IS ON A GREAT CIRCLE WHERE P | N = 0 (CL(3))",
    );
}

fn main() {
    run(Anim::new("conformal elliptical", 16.0).size(960, 540), draw);
}

#[cfg(test)]
mod tests {
    use super::elliptical::*;

    /// The planes are unit normals, no two of them equal up to sign.
    #[test]
    fn thirteen_distinct_unit_planes() {
        let planes = octahedral_planes();
        assert_eq!(planes.len(), 13);
        for (i, a) in planes.iter().enumerate() {
            assert!(((*a | *a).s() - 1.0).abs() < 1e-12);
            for b in &planes[i + 1..] {
                assert!((*a | *b).s().abs() < 1.0 - 1e-6);
            }
        }
    }

    /// The scenario's check: a quarter turn about an axis and a third turn about a body
    /// diagonal generate the rotations of the cube, and each carries every mirror plane onto a
    /// mirror plane.
    #[test]
    fn the_cube_s_rotations_permute_the_planes() {
        let planes = octahedral_planes();
        for rotor in [quarter(), third()] {
            for p in &planes {
                let turned = rotor >> *p;
                let best = planes
                    .iter()
                    .map(|q| (turned | *q).s().abs())
                    .fold(0.0, f64::max);
                assert!((best - 1.0).abs() < 1e-12, "{best}");
            }
        }
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws(super::draw, 0.5);
    }
}
