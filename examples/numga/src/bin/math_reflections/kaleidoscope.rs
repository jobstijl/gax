//! numga's `kaleidoscope` notebook: two mirrors of the plane, VGA2D.
//!
//! The first mirror lies along `x`, so its vector is `y`; the second's vector is turned from
//! the first by a half turn less the angle `π/n` between the mirrors, so that both point out of
//! the wedge between them and pair to `-cos(π/n)`. Their product is the rotor turning by twice
//! the angle, and its powers up to the `n`-th are the kaleidoscope's `n` rotations, the last a
//! scalar. Reflecting each point in the two mirrors in turn, as many rounds as there are
//! panels, brings it into the wedge; points that fold onto the same point see the same bead.

use super::fold;
use gax::vga2d::Vector;
use gax_numga_examples::rng::{Draw, rng};
use gax_numga_examples::signal::phasor;
use gax_numga_examples::{Canvas, Light, Point2, light, palette};

pub type V = Vector<(), f64>;

/// numga's kaleidoscopes: mirrors meeting at a half turn over three to six.
pub const PANELS: [usize; 4] = [3, 4, 5, 6];
/// The rounds of reflections that fold every point into the wedge: the most panels.
const ROUNDS: usize = 6;

/// The two mirror vectors of the kaleidoscope with `n` panels.
pub fn mirrors(n: usize) -> [V; 2] {
    let first = Vector::new(0.0, 1.0);
    let x = Vector::new(1.0, 0.0);
    let pi = core::f64::consts::PI;
    let turn = (x ^ first).gp(-(pi - pi / n as f64) / 2.0).exp();
    [first, turn >> first]
}

/// A point folded into the wedge of the mirrors.
pub fn folded(mirrors: &[V; 2], point: V) -> V {
    let side = |m: V, p: V| (m | p).s();
    fold(point, mirrors, ROUNDS, side, |m, p| m >> p).0
}

/// A bead tumbling round the kaleidoscope: where it starts (a radius and an angle), how many
/// turns it makes per loop and which way, its size and its colour.
struct Bead {
    radius: f64,
    angle: f64,
    speed: f64,
    size: f64,
    colour: Light,
}

/// numga's 40 beads scattered over the disk, from its short palette, drawn from a seeded
/// stream (numpy's cannot be reproduced).
fn beads() -> Vec<Bead> {
    const SEED: u64 = 7;
    let palette = [
        [0.91, 0.30, 0.24],
        [0.98, 0.77, 0.19],
        [0.18, 0.62, 0.48],
        [0.56, 0.27, 0.68],
        [0.20, 0.47, 0.85],
        [0.95, 0.55, 0.65],
        [0.99, 0.97, 0.90],
    ];
    let mut r = rng(SEED);
    (0..40)
        .map(|k| {
            let [red, green, blue] = palette[k % palette.len()];
            Bead {
                // The larger of two uniform draws spreads the beads evenly over the disk: its
                // density grows with the radius, as the area does (numga's square root of one).
                radius: r.uniform().max(r.uniform()),
                angle: r.range(0.0, core::f64::consts::TAU),
                speed: [-2.0, -1.0, 1.0, 2.0][(r.uniform() * 4.0) as usize],
                size: r.range(0.04, 0.13),
                colour: Light::from_srgb(red, green, blue, 1.5),
            }
        })
        .collect()
}

/// What shows at a folded point at a phase of the tumble: the beads painted over each other in
/// order, each covering by how far the point lies inside it, over the dark of the tube.
fn paint(beads: &[Bead], point: V, phase: f64) -> Light {
    const SOFT: f64 = 0.02;
    beads
        .iter()
        .fold(light(0.01, 0.015, 0.04, 1.0), |under, bead| {
            let at = phasor(bead.angle + bead.speed * phase);
            let centre = Vector::new(at.e20(), at.e01()).gp(bead.radius);
            let cover = ((bead.size - (point - centre).norm()) / SOFT).clamp(0.0, 1.0);
            under.mix_light(bead.colour, cover as f32)
        })
}

/// The kaleidoscopes side by side, the beads at time `t` of the act.
pub fn draw(c: &mut Canvas, t: f32) {
    let phase = core::f64::consts::TAU * f64::from(t / super::ACT);
    let beads = beads();
    let screen = c.rect();
    for (k, n) in PANELS.into_iter().enumerate() {
        let column = screen.part(0.25 * k as f32, 0.2, 0.25 * (k + 1) as f32, 0.9);
        let ax = super::disk_axes(column.inset(8.0, 8.0, 8.0, 30.0));
        let mirrors = mirrors(n);
        ax.image(c, 1, |p| {
            let [u, v] = p.to_euclidean().map(f64::from);
            let point = Vector::new(u, v);
            (point.norm_squared() < 1.0).then(|| paint(&beads, folded(&mirrors, point), phase))
        });
        let label = format!("{n} PANELS");
        let at = Point2::xy(0.0, -1.18);
        ax.text(
            c,
            at,
            &label,
            11.0,
            palette::ink(),
            gax_numga_examples::Align::Center,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The rotor of the kaleidoscope: the product of its mirrors, turning by twice their angle.
    fn turn([first, second]: [V; 2]) -> gax::vga2d::Rotor<(), f64> {
        second * first
    }

    /// numga's checks: the mirror vectors pair to `-cos(π/n)`; the `n`-th power of the turn is
    /// a scalar of size one; every folded point of the view lies on the right side of both
    /// mirrors.
    #[test]
    fn the_mirrors_turn_by_twice_their_angle_and_fold_into_the_wedge() {
        for n in PANELS {
            let pair = mirrors(n);
            let between = core::f64::consts::PI / n as f64;
            assert!(((pair[0] | pair[1]).s() + phasor(between).e20()).abs() < 1e-12);
            let rotor = turn(pair);
            let power = (1..n).fold(rotor, |p, _| p * rotor);
            assert!((power.s().abs() - 1.0).abs() < 1e-12 && power.e12().abs() < 1e-12);
            for (u, v) in super::super::disk(240) {
                let point = folded(&pair, Vector::new(u, v));
                for m in pair {
                    assert!((m | point).s() > -1e-12);
                }
            }
        }
    }
}
