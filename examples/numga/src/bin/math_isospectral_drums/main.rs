//! numga's `math/isospectral_drums`: you cannot hear the shape of a drum. Gordon, Webb and
//! Wolpert's two drums of different shapes have the same frequencies, and a transplantation
//! carries each standing wave of one onto a standing wave of the other, as sparse maps between
//! fields on their lattices.
//!
//! Each drum is cut from seven equal right triangles. Restriction maps read a field's triangles;
//! a signed mixing makes each destination triangle from three source pieces, which the adjoint of
//! reading that triangle puts in place; and the destination's assembly welds the triangles into
//! one field. The transplant carries one drum's wave operator onto the other's, so standing waves
//! keep their frequencies: two eigensolves agree to rounding.
//!
//! Top, a mix of three standing waves on the source drum, each at its own frequency, and its
//! transplant on the target, moving in step; the nodal lines are drawn, and the tiles outlined.
//! Below, three signed pieces of the source wave that make one destination triangle, and their
//! sum.

mod drums;
mod scenes;
#[cfg(test)]
mod tests;

#[path = "../../shared/simplicial.rs"]
mod simplicial;
#[path = "../../shared/surfaces_plane.rs"]
mod surfaces_plane;

use drums::*;
use gax::field::Field;
use gax_numga_examples::signal::phasor;
use gax_numga_examples::{
    Anim, Axes, Canvas, Light, Point2, Rect, backdrop, caption, label_under, palette, run_with,
};
use scenes::*;
use simplicial::faces;
use surfaces_plane::Flat;

/// One loop: numga's 72 frames of 40 ms, one period of the slowest wave mixed in.
const SECONDS: f32 = 2.88;

struct Shown {
    pair: Pair,
    flats: [Flat; 2],
    /// Each drum's tiles' corners.
    tiles: [Vec<[Point2; 3]>; 2],
    /// The frequencies of the mixed waves.
    frequencies: [f64; 3],
    piece_flat: Flat,
    /// The three signed pieces of the destination triangle and their sum, on the reference tile.
    pieces: Vec<Vec<f64>>,
}

/// A standing wave's frequency, the root of its eigenvalue.
#[allow(clippy::disallowed_methods)] // a frequency from its square: a scalar law, not geometry
fn frequency(eigenvalue: f64) -> f64 {
    eigenvalue.sqrt()
}

/// A drum's mesh as drawn.
fn flat(drum: &Drum) -> Flat {
    let points: Vec<[f64; 2]> = drum.vertices.iter().map(coordinates).collect();
    Flat::new(&points, &faces(&drum.mesh))
}

/// The corners of a drum's tiles.
fn tile_corners(tiles: &[[usize; 3]]) -> Vec<[Point2; 3]> {
    tiles
        .iter()
        .map(|t| t.map(|i| Point2::xy((i % LATTICE) as f32, (i / LATTICE) as f32)))
        .collect()
}

fn shown() -> Shown {
    let pair = Pair::new(SUBDIVISIONS, MODES);
    let flats = [flat(&pair.source), flat(&pair.target)];
    let frequencies = MIXED_MODES.map(|k| frequency(pair.eigenvalues[k]));
    // The last mixed wave's pieces on its tiles, three signed ones making the destination tile.
    let samples = tile_samples(SUBDIVISIONS);
    let last = MIXED_MODES.len() - 1;
    let wave = pair.modes[MIXED_MODES[last]].map(|x| x * MIX_AMPLITUDES[last]);
    let on_tiles = pair.source.restriction.of(&wave);
    let tile = |t: usize| -> Vec<f64> {
        (0..samples)
            .map(|s| on_tiles.site(t * samples + s).s())
            .collect()
    };
    let mut pieces: Vec<Vec<f64>> = CONTRIBUTORS[TARGET_TILE]
        .iter()
        .zip(SIGNS[TARGET_TILE])
        .map(|(&t, sign)| tile(t).iter().map(|x| x * sign).collect())
        .collect();
    let sum = (0..samples)
        .map(|s| pieces.iter().map(|p| p[s]).sum())
        .collect();
    pieces.push(sum);
    let (local, mesh) = triangle(SUBDIVISIONS);
    let points: Vec<[f64; 2]> = local.iter().map(coordinates).collect();
    Shown {
        flats,
        tiles: [tile_corners(&SOURCE), tile_corners(&TARGET)],
        frequencies,
        piece_flat: Flat::new(&points, &faces(&mesh)),
        pieces,
        pair,
    }
}

/// A displacement's light: orange above, sky below.
fn displacement_light(v: f64) -> Light {
    let t = v.clamp(-1.0, 1.0) as f32;
    let accent = if t > 0.0 {
        palette::ORANGE
    } else {
        palette::SKY
    };
    palette::GRID.faded(0.5) + accent.faded(0.06 + 0.45 * t.abs())
}

/// A field shaded on a mesh, with its nodal lines.
fn membrane(c: &mut Canvas, ax: &Axes, flat: &Flat, values: &[f64]) {
    flat.shade_field(c, ax, values, |v| displacement_light(v / EXPOSURE));
    ax.stroke(c, &flat.level(values, 0.0), 1.0, palette::INK.faded(0.55));
    ax.stroke(c, &flat.rim(), 1.6, palette::INK.faded(0.8));
}

/// The two drums at time `t` of the period.
fn draw_drums(c: &mut Canvas, rect: Rect, s: &Shown, phase: f64) {
    let period = core::f64::consts::TAU / s.frequencies[0];
    let time = period * phase;
    let mut source = Field::<Scalar>::zeros(s.pair.modes[0].len());
    for (k, &mode) in MIXED_MODES.iter().enumerate() {
        let weight = MIX_AMPLITUDES[k] * phasor(s.frequencies[k] * time).e20();
        source = &source + &s.pair.modes[mode].map(|x| x * weight);
    }
    let fields = [
        s.pair.source.lift.of(&source),
        s.pair.target.lift.of(&s.pair.transplant.of(&source)),
    ];
    let labels = ["THE SOURCE DRUM", "THE TARGET DRUM: THE SAME FREQUENCIES"];
    for d in 0..2 {
        let panel = rect.column(d, 2).inset(10.0, 4.0, 10.0, 4.0);
        let ax = Axes::fitting(
            panel.part(0.0, 0.0, 1.0, 0.9),
            s.flats[d].points.iter().copied(),
            1.06,
        );
        let values: Vec<f64> = fields[d].iter().map(|x| x.s()).collect();
        membrane(c, &ax, &s.flats[d], &values);
        for t in &s.tiles[d] {
            ax.polyline(c, &[t[0], t[1], t[2], t[0]], 0.8, palette::GRID.faded(3.0));
        }
        label_under(c, panel, labels[d], 14.0, 0.8);
    }
}

/// The destination triangle's three signed source pieces and their sum.
fn draw_pieces(c: &mut Canvas, rect: Rect, s: &Shown) {
    let labels = ["PIECE", "MINUS PIECE", "PIECE", "SUM"];
    let source = CONTRIBUTORS[TARGET_TILE];
    for (k, values) in s.pieces.iter().enumerate() {
        let panel = rect.column(k, 4).inset(14.0, 2.0, 14.0, 2.0);
        let ax = Axes::fitting(
            panel.part(0.0, 0.0, 1.0, 0.82),
            s.piece_flat.points.iter().copied(),
            1.08,
        );
        membrane(c, &ax, &s.piece_flat, values);
        let label = if k < 3 {
            format!("{} OF TILE {}", labels[k], source[k])
        } else {
            format!("= TARGET TILE {TARGET_TILE}")
        };
        label_under(c, panel, &label, 9.0, 0.8);
    }
}

fn draw(c: &mut Canvas, s: &Shown, t: f32) {
    backdrop(c);
    let screen = c.rect();
    let (w, h) = (screen.width(), screen.height());
    let phase = f64::from((t / SECONDS).rem_euclid(1.0));
    draw_drums(c, Rect::new(0.0, h * 0.13, w, h * 0.7), s, phase);
    draw_pieces(c, Rect::new(w * 0.1, h * 0.71, w * 0.9, h * 0.98), s);
    caption(
        c,
        "ISOSPECTRAL DRUMS: TWO SHAPES, ONE SPECTRUM",
        "SEVEN TRIANGLES EACH; THREE SIGNED SOURCE PIECES MAKE EACH TARGET TRIANGLE, AND THE TRANSPLANT CARRIES EVERY WAVE ACROSS (VGA2D)",
    );
}

fn main() {
    run_with(
        Anim::new("isospectral drums", SECONDS).size(960, 540),
        shown,
        draw,
    );
}
