//! numga's `electromagnetism/electrostriction`: an elastic lattice of induced dipoles, at
//! mechanical and electrical equilibrium, in VGA2D.
//!
//! Particles and their dipoles lie in a plane, and interact by the three-dimensional dipole field
//! restricted to it (four pi times the permittivity one). Central springs join neighbours, and
//! weak tethers to the reference sites support the lattice. Each dipole feels the applied field
//! and the fields of all the others, so the dipoles at a given shape come from one solve of a
//! field map over the particles, the response: one over the polarizability at each particle, less
//! the dipole field `(3 r (r · p) - r² p) / |r|⁵` between every pair, a `Vector <- Vector` map per
//! pair. At their equilibrium the electrical energy, the cost of inducing the dipoles and the work
//! exchanged with the source of the fixed applied field included, is minus half their pairing
//! with the field, and the lattice's shape minimizes the elastic energy plus that.
//!
//! numga relaxes the shape by Newton's method, differentiating the energy twice with the
//! solve inside it (jax). gax's field derivatives (`gax::field::hessian`) do the same, and the
//! tests use them, but they evaluate the energy, solve included, once for every pair of sites:
//! about 60 ms a Newton step, too slow to animate. The animation's Newton steps take the same
//! derivatives by the chain rule instead, in about 0.6 ms. With the dipoles held fixed the
//! energy is a sum of terms each of the difference of two sites (a bond, a pair of dipoles),
//! whose derivatives `gax::jet::hessian` gives and the differences carry to the sites, as
//! `docs/fields.md` describes. Letting the dipoles readjust
//! keeps the first derivative (they are at their minimum) and softens the second by
//! `Jᵀ R⁻¹ J`, `J` the change of the response's image of the dipoles with the shape (numga's
//! test of the readjustment): one factorization of the response, and its solve for each
//! coordinate of each site. The tests check the two Hessians agree.
//!
//! The animation turns the applied field once while it rises and falls: the dipoles and the
//! deformed lattice over its reference, and the lattice's strain along and across the field as
//! the field grows, against which the frame's strain is marked.

use gax::field::Field;
use gax::vga2d::Pseudoscalar;
use gax_numga_examples::signal::linspace;
use gax_numga_examples::{
    Anim, Axes, Canvas, Light, Marker, ORIGIN2, Point2, Rect, backdrop, caption, palette, reach2,
    run,
};
use std::sync::OnceLock;

mod lattice;
use gax_numga_examples::shared::electromagnetism_dipoles as pairs;
#[cfg(test)]
mod tests;

use lattice::*;

/// One loading cycle: numga's 72 frames of 70 ms.
const SECONDS: f32 = 5.04;
/// The dipole arrows' length per unit dipole (numga's 4, longer to show at the weaker fields),
/// and the applied field's arrow per unit field.
const DIPOLE_SCALE: f64 = 6.0;
const FIELD_SCALE: f64 = 0.65 / FIELD;
/// The margin around the reference lattice, and the room below it for the field's arrow.
const MARGIN: f32 = 0.55;
const FIELD_GAP: f32 = 0.8;
/// The strains the loading panel spans, in percent.
const STRAIN_PERCENT: [f32; 2] = [-9.0, 2.0];

/// The lattice and its loading curve, computed once.
struct Scene {
    lattice: Lattice,
    /// The field's strengths along `x`, and the strains along and across it.
    strengths: Vec<f64>,
    along: Vec<f64>,
    across: Vec<f64>,
}

fn scene() -> &'static Scene {
    static SCENE: OnceLock<Scene> = OnceLock::new();
    SCENE.get_or_init(|| {
        let lattice = lattice();
        let strengths = linspace(0.0, FIELD, LOAD_SAMPLES);
        let shapes: Vec<Field<V>> = strengths
            .iter()
            .map(|s| lattice.equilibrium(X * *s, NEWTON_STEPS))
            .collect();
        let strains = |direction: V| {
            shapes
                .iter()
                .map(|x| strain(x, &lattice.rest, direction))
                .collect()
        };
        Scene {
            along: strains(X),
            across: strains(Y),
            strengths,
            lattice,
        }
    })
}

/// The bonds between the positions.
fn bonds(lattice: &Lattice, positions: &Field<V>) -> Vec<[Point2; 2]> {
    lattice
        .bonds
        .iter()
        .map(|&[a, b]| [reach2(positions.site(a)), reach2(positions.site(b))])
        .collect()
}

/// The lattice at its equilibrium in the applied field: the reference dashed, the bonds and
/// particles, each dipole an arrow centred on its particle, and the field's arrow below.
fn lattice_panel(c: &mut Canvas, rect: Rect, lattice: &Lattice, positions: &Field<V>, applied: V) {
    let reach = lattice
        .rest
        .iter()
        .map(|p| p.norm() as f32)
        .fold(0.0, f32::max)
        + MARGIN;
    let centre = Point2::xy(0.0, -FIELD_GAP / 2.0);
    let ax = Axes::equal(rect, centre, reach + FIELD_GAP / 2.0);
    for segment in bonds(lattice, &lattice.rest) {
        ax.dashed(c, &segment, 1.0, 4.0, palette::GRID.faded(3.0));
    }
    ax.stroke(c, &bonds(lattice, positions), 2.0, palette::SKY);
    let particles: Vec<Point2> = positions.iter().map(reach2).collect();
    ax.scatter(c, &particles, Marker::Dot, 8.0, palette::SKY);
    let dipoles = lattice.dipoles(positions, applied);
    for (site, dipole) in positions.iter().zip(dipoles.iter()) {
        let arrow = dipole * DIPOLE_SCALE;
        centred_arrow(c, &ax, site, arrow, [2.0, 7.0], palette::ORANGE);
    }
    let below = V::new(0.0, -f64::from(reach + FIELD_GAP / 2.0));
    let arrow = applied * FIELD_SCALE;
    centred_arrow(c, &ax, below, arrow, [3.0, 10.0], palette::YELLOW);
}

/// An arrow centred on `centre`, its stroke's width and its head's size `[width, head]`.
fn centred_arrow(
    c: &mut Canvas,
    ax: &Axes,
    centre: V,
    arrow: V,
    [width, head]: [f32; 2],
    light: Light,
) {
    let half = arrow * 0.5;
    ax.arrow(
        c,
        reach2(centre - half),
        reach2(centre + half),
        width,
        head,
        light,
    );
}

/// The strain along and across the field over its strength, with the frame's strains marked.
fn loading_panel(c: &mut Canvas, rect: Rect, positions: &Field<V>, applied: V) {
    let s = scene();
    let ax = Axes::new(rect, [0.0, FIELD as f32], STRAIN_PERCENT);
    ax.frame(c, "CHANGE IN RMS EXTENT (%)", "APPLIED FIELD STRENGTH", "");
    ax.line(
        c,
        ORIGIN2,
        Point2::xy(FIELD as f32, 0.0),
        1.0,
        palette::GRID.faded(2.0),
    );
    let curve = |strains: &[f64]| -> Vec<Point2> {
        s.strengths
            .iter()
            .zip(strains)
            .map(|(f, e)| Point2::xy(*f as f32, (100.0 * e) as f32))
            .collect()
    };
    ax.polyline(c, &curve(&s.along), 2.2, palette::SKY);
    ax.polyline(c, &curve(&s.across), 2.2, palette::PURPLE);
    // The frame's strains along its field and a quarter turn across, at its strength.
    let strength = applied.norm();
    let along = if strength > 0.0 {
        applied * (1.0 / strength)
    } else {
        X
    };
    let across = along * Pseudoscalar::new(1.0);
    let marks = [(along, palette::SKY), (across, palette::PURPLE)];
    for (direction, colour) in marks {
        let at = Point2::xy(
            strength as f32,
            (100.0 * strain(positions, &s.lattice.rest, direction)) as f32,
        );
        ax.scatter(c, &[at], Marker::Ring, 12.0, colour);
    }
    ax.legend(
        c,
        &[
            ("ALONG THE FIELD", palette::SKY),
            ("ACROSS THE FIELD", palette::PURPLE),
        ],
    );
}

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let share = f64::from(t / SECONDS).rem_euclid(1.0);
    let applied = turning(share);
    let s = scene();
    let positions = s.lattice.equilibrium(applied, NEWTON_STEPS);
    let screen = c.rect();
    let (w, h) = (screen.width(), screen.height());
    let top = (h / 30.0).clamp(10.0, 22.0) * 3.6;
    lattice_panel(
        c,
        Rect::new(0.0, top, w * 0.5, h).inset(10.0, 4.0, 10.0, 10.0),
        &s.lattice,
        &positions,
        applied,
    );
    loading_panel(
        c,
        Rect::new(w * 0.5, top, w, h).inset(46.0, 30.0, 20.0, 44.0),
        &positions,
        applied,
    );
    caption(
        c,
        "ELECTROSTRICTION: AN ELASTIC LATTICE OF INDUCED DIPOLES",
        "DIPOLES FROM ONE SOLVE OF THE RESPONSE; SHAPE BY NEWTON ON THE EQUILIBRATED ENERGY (VGA2D)",
    );
}

fn main() {
    run(Anim::new("electrostriction", SECONDS).size(960, 540), draw);
}
