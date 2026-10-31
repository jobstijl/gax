//! numga's `math/eight_conics`: Penrose's eight-conic theorem, in the projective plane, PGA2D.
//!
//! A conic is a map from points to lines, `Line <- Point`: each point to its polar line, the
//! conic the points that lie on their own polar, `Q(p) & p = 0`. Adding the dyad `l (l & ·)` of
//! a line `l` changes the map everywhere but on that line, so the new conic meets the old one
//! where `l` meets it, with the same tangents there: double contact is an addition of maps.
//!
//! Put conics on the corners of a cube, each in double contact with its three neighbours along
//! the cube's edges: from a circle at one corner, three weighted contact lines give its three
//! neighbours, and a coupling per face gives the three corners across the faces, each sharing
//! one contact line with each of two neighbours (the four contact lines of a face meet in one
//! point). The eighth corner then follows in three ways, one more contact dyad from each of its
//! three neighbours, and the three are one map: Penrose's theorem. A mirror moves the whole
//! configuration with its inputs.
//!
//! The animation sweeps the three couplings round a loop, a third of a turn apart: left the
//! eighth conic (orange) with its three neighbours, each with their common contact line and their
//! two points of contact, and the other four conics faint; right the cube of corners and edges.

use gax::pga2d::{Line, Motor, Point};
use gax_numga_examples::{
    Align, Anim, Axes, Canvas, Light, Point2, Rect, backdrop, caption, contour, label_under,
    palette, points, run_with,
};

/// Penrose's configuration: conics as maps, their contacts.
mod conics {
    use super::{Line, Motor, Point};
    use gax_numga_examples::measure::roots;
    use gax_numga_examples::signal::phasor;

    /// A conic: each point to its polar line.
    pub type Conic = Line<(Point,)>;

    /// The three faces opposite the first neighbours use the other two contact lines.
    pub const FORWARD: [usize; 3] = [1, 2, 0];
    pub const BACKWARD: [usize; 3] = [2, 0, 1];
    /// The cube's edges between corners labelled in binary: each changes one of three choices.
    pub const EDGES: [[usize; 2]; 12] = [
        [0, 1],
        [0, 2],
        [0, 4],
        [1, 3],
        [2, 3],
        [1, 5],
        [4, 5],
        [2, 6],
        [4, 6],
        [3, 7],
        [5, 7],
        [6, 7],
    ];
    /// The edges joining the eighth conic to its three neighbours.
    pub const CLOSING: [usize; 3] = [9, 10, 11];
    /// numga's sweep: the couplings' means and amplitude.
    const MEAN_COUPLINGS: [f64; 3] = [0.30, -0.24, 0.27];
    const COUPLING_AMPLITUDE: f64 = 0.25;
    /// The contact lines' distances from the centre, and their weight: strong contact lines make
    /// the eight conics distinct ellipses; stronger ones turn some into hyperbolas.
    const OFFSETS: [f64; 3] = [0.20, 0.24, 0.16];
    const LINE_WEIGHT: f64 = 1.3;
    /// numga's frames over the loop.
    #[cfg_attr(not(test), allow(dead_code))] // numga's checks run on every frame
    pub const FRAMES: usize = 216;

    /// The dyad of a line: the map adding `l (l & p)`.
    pub fn dyad(l: Line) -> Conic {
        l * (l & Point::slot())
    }

    /// The cube completed from a conic, three weighted contact lines and three face
    /// couplings.
    pub struct Configuration {
        /// The eight conics, by binary corner label.
        pub conics: [Conic; 8],
        /// The twelve contact lines, by edge.
        pub edge_lines: [Line; 12],
        /// The eighth conic's three independent completions.
        #[cfg_attr(not(test), allow(dead_code))] // numga's check that they agree
        pub completions: [Conic; 3],
    }

    impl Configuration {
        pub fn new(base: Conic, contact: [Line; 3], couplings: [f64; 3]) -> Configuration {
            let c = couplings;
            let double = contact.map(dyad);
            let neighbours = double.map(|d| base + d);
            let left = FORWARD.map(|k| contact[k]);
            let right = BACKWARD.map(|k| contact[k]);
            let face_scales = c.map(|x| 1.0 - x * x);

            // Each face conic shares one contact line with each of two neighbours.
            let faces: [Conic; 3] = core::array::from_fn(|k| {
                let cross =
                    left[k] * (right[k] & Point::slot()) + right[k] * (left[k] & Point::slot());
                base.gp(face_scales[k]) + double[FORWARD[k]] + double[BACKWARD[k]] + cross.gp(c[k])
            });

            // The three closing contact lines determine the same eighth conic independently.
            let closing_scale =
                1.0 - c.iter().map(|x| x * x).sum::<f64>() - 2.0 * c[0] * c[1] * c[2];
            let closing: [Line; 3] = core::array::from_fn(|k| {
                let (f, b) = (c[FORWARD[k]], c[BACKWARD[k]]);
                contact[k].gp(face_scales[k]) + left[k].gp(b + c[k] * f) + right[k].gp(f + c[k] * b)
            });
            let completions: [Conic; 3] = core::array::from_fn(|k| {
                (faces[k].gp(closing_scale) + dyad(closing[k])).gp(1.0 / face_scales[k])
            });

            // Binary corner labels make each cube edge change one of the three choices.
            let conics = [
                base,
                neighbours[0],
                neighbours[1],
                faces[2],
                neighbours[2],
                faces[1],
                faces[0],
                completions[0],
            ];
            let left_lines: [Line; 3] = core::array::from_fn(|k| left[k].gp(c[k]) + right[k]);
            let right_lines: [Line; 3] = core::array::from_fn(|k| left[k] + right[k].gp(c[k]));
            let edge_lines = [
                contact[0],
                contact[1],
                contact[2],
                left_lines[2],
                right_lines[2],
                right_lines[1],
                left_lines[1],
                left_lines[0],
                right_lines[0],
                closing[2],
                closing[1],
                closing[0],
            ];
            Configuration {
                conics,
                edge_lines,
                completions,
            }
        }

        /// The two crossings of each edge's first conic with the edge's contact line.
        pub fn contacts(&self) -> [[Point; 2]; 12] {
            core::array::from_fn(|e| intersections(self.conics[EDGES[e][0]], self.edge_lines[e]))
        }
    }

    /// The unit circle about the origin: the inverse of its dual, the map from lines to points
    /// `Σ s (l & s) - o (l & o)` over the axes' directions `s` and the origin `o`.
    pub fn circle() -> Conic {
        let axes = [Point::direction(1.0, 0.0), Point::direction(0.0, 1.0)];
        let origin = Point::origin();
        let dual: Point<(Line,)> = axes.iter().fold(Point::<(Line,)>::zero(), |sum, s| {
            sum + *s * (Line::slot() & *s)
        }) - origin * (Line::slot() & origin);
        dual.inverse()
    }

    /// The three contact lines of the base circle's neighbours: normals a third of a turn apart,
    /// at numga's offsets, weighted.
    pub fn contact_lines() -> [Line; 3] {
        core::array::from_fn(|k| {
            let turn = Motor::rotation(Point::origin(), core::f64::consts::TAU * k as f64 / 3.0);
            let normal = turn >> Point::direction(1.0, 0.0);
            Line::with_normal(normal, OFFSETS[k]).gp(LINE_WEIGHT)
        })
    }

    /// The couplings at a share of the loop: their means plus the amplitude along the cosine
    /// of the phase, a third of a turn apart.
    pub fn couplings(share: f64) -> [f64; 3] {
        core::array::from_fn(|k| {
            let phase = core::f64::consts::TAU * (share + k as f64 / 3.0);
            MEAN_COUPLINGS[k] + COUPLING_AMPLITUDE * phasor(phase).e20()
        })
    }

    /// The configuration at a share of the loop.
    pub fn at(share: f64) -> Configuration {
        Configuration::new(circle(), contact_lines(), couplings(share))
    }

    /// numga's frames: the configuration at each of the loop's 216 frames.
    #[cfg_attr(not(test), allow(dead_code))] // numga's checks run on every frame
    pub fn frames() -> Vec<Configuration> {
        (0..FRAMES).map(|f| at(f as f64 / FRAMES as f64)).collect()
    }

    /// The two real crossings of a conic with a line: from the foot of the perpendicular from
    /// the origin, along the line, the roots of the conic's quadratic in the distance.
    pub fn intersections(conic: Conic, line: Line) -> [Point; 2] {
        let line = line.normalized();
        let origin = Point::origin();
        // Reflection gives the foot of the perpendicular from the origin.
        let anchor = (origin + (line >> origin)).gp(0.5);
        let direction = line.direction();
        let quadratic = (conic.of(direction) & direction).s();
        let linear = (conic.of(anchor) & direction).s();
        let constant = (conic.of(anchor) & anchor).s();
        let (middle, half) = roots(quadratic, linear, constant).expect("two real crossings");
        [middle - half, middle + half].map(|t| anchor + direction.gp(t))
    }
}

use conics::{CLOSING, Conic, EDGES};

/// The loop's seconds: numga's 216 frames of 40 ms.
const SECONDS: f32 = 8.64;
/// The grid the conics are contoured on, and its half width.
const SAMPLES: usize = 240;
const HALF_WIDTH: f32 = 2.5;

/// The eighth conic's colour, and its three neighbours'.
const EIGHTH: Light = palette::ORANGE;
const NEIGHBOURS: [Light; 3] = [palette::BLUE, palette::GREEN, palette::PURPLE];

/// The zero level of a conic, `Q(p) & p`, on the grid: its segments.
fn curve(conic: Conic) -> Vec<[Point2; 2]> {
    let conic = conic.map_coefs(|x| x as f32);
    let value = |p: Point2| (conic.of(p) & p).s();
    let range = [-HALF_WIDTH, HALF_WIDTH];
    contour::of_fn(value, range, range, SAMPLES, 0.0)
}

/// The view: a square around every conic of the first frame, with an eighth more.
fn framing() -> (Point2, f32) {
    let first = conics::at(0.0);
    let ends = first.conics.iter().flat_map(|c| curve(*c)).flatten();
    let [[x0, y0], [x1, y1]] = points::bounds(ends);
    let centre = Point2::xy(0.5 * (x0 + x1), 0.5 * (y0 + y1));
    (centre, 0.5 * (x1 - x0).max(y1 - y0) * 1.12)
}

/// The conics at a share of the loop: the other four faint, the three neighbours with their
/// contact lines dashed and their points of contact, the eighth bold.
fn conics_panel(c: &mut Canvas, rect: Rect, view: &(Point2, f32), share: f64) {
    let ax = Axes::equal(rect, view.0, view.1);
    let config = conics::at(share);
    let neighbours = CLOSING.map(|e| EDGES[e][0]);
    let contacts = config.contacts();
    for corner in (0..7).filter(|k| !neighbours.contains(k)) {
        ax.stroke(
            c,
            &curve(config.conics[corner]),
            1.2,
            palette::INK.faded(0.3),
        );
    }
    for (side, edge) in CLOSING.iter().enumerate() {
        let line = config.edge_lines[*edge].normalized();
        let foot = line >> Point::origin();
        let far = line.direction().gp(10.0);
        ax.clip(c);
        ax.dashed(
            c,
            &[foot - far, foot + far],
            1.0,
            5.0,
            NEIGHBOURS[side].faded(0.6),
        );
        c.unclip();
        ax.stroke(
            c,
            &curve(config.conics[neighbours[side]]),
            1.8,
            NEIGHBOURS[side],
        );
    }
    ax.stroke(c, &curve(config.conics[7]), 3.0, EIGHTH);
    for (side, edge) in CLOSING.iter().enumerate() {
        for p in contacts[*edge] {
            let px = ax.px(p);
            c.disk(px, 6.5, palette::BOTTOM.faded(4.0));
            c.disk(px, 4.5, NEIGHBOURS[side].whitened(0.3));
        }
    }
    label_under(
        c,
        rect,
        "THE EIGHTH CONIC IN DOUBLE CONTACT WITH ITS THREE NEIGHBOURS",
        30.0,
        0.9,
    );
}

/// The cube of corners: the eighth and its three neighbours in their colours, joined by their
/// edges, and the couplings now.
fn cube_panel(c: &mut Canvas, rect: Rect, share: f64) {
    let ax = Axes::equal(rect, Point2::xy(0.72, 0.62), 1.2);
    let place = |k: usize| {
        let bit = |b: usize| ((k >> b) & 1) as f32;
        Point2::xy(bit(0) + 0.45 * bit(2), bit(1) + 0.35 * bit(2))
    };
    let neighbours = CLOSING.map(|e| EDGES[e][0]);
    let colour = |k: usize| match neighbours.iter().position(|n| *n == k) {
        Some(side) => NEIGHBOURS[side],
        None if k == 7 => EIGHTH,
        None => palette::INK.faded(0.4),
    };
    for (e, [a, b]) in EDGES.iter().enumerate() {
        let edge = match CLOSING.iter().position(|x| *x == e) {
            Some(side) => NEIGHBOURS[side],
            None => palette::INK.faded(0.3),
        };
        ax.line(c, place(*a), place(*b), 1.4, edge);
    }
    for k in 0..8 {
        let px = ax.px(place(k));
        c.disk(px, 10.0, palette::BOTTOM.faded(4.0));
        c.ring(px, 9.0, 1.4, colour(k));
        let label = format!("{k}");
        c.text(
            &label,
            px + Point2::direction(0.0, 4.5),
            11.0,
            colour(k),
            Align::Center,
        );
    }
    let coupling = conics::couplings(share);
    for (k, value) in coupling.iter().enumerate() {
        let line = format!("FACE COUPLING {}: {value:+.2}", k + 1);
        let at = rect.bottom_left() + Point2::direction(0.0, 18.0 * k as f32 - 40.0);
        c.text(&line, at, 11.0, palette::INK, Align::Left);
    }
}

fn draw(c: &mut Canvas, view: &(Point2, f32), t: f32) {
    backdrop(c);
    let screen = c.rect();
    let share = f64::from(t / SECONDS);
    conics_panel(c, screen.part(0.02, 0.13, 0.62, 0.95), view, share);
    cube_panel(c, screen.part(0.66, 0.2, 0.98, 0.95), share);
    caption(
        c,
        "EIGHT CONICS: PENROSE'S THEOREM AS ADDITIONS OF MAPS (PGA2D)",
        "DOUBLE CONTACT ADDS A LINE'S DYAD; THE EIGHTH CONIC, FROM ANY OF ITS THREE NEIGHBOURS, IS ONE MAP",
    );
}

fn main() {
    run_with(
        Anim::new("eight_conics", SECONDS).size(960, 540),
        framing,
        draw,
    );
}

#[cfg(test)]
mod tests {
    use super::conics::{self, Configuration, EDGES};
    use super::{Line, Point};
    use gax::ApproxEq;

    const ROUND_OFF: f64 = 1e-12;

    /// The sum of the squared coefficients (numga's norm of the kernel, squared).
    fn coefficients_squared<const N: usize>(c: [f64; N]) -> f64 {
        c.iter().map(|x| x * x).sum()
    }

    /// Every coefficient within round-off of zero (numga's absolute tolerance).
    fn vanishes<const N: usize>(c: [f64; N]) -> bool {
        c.iter().all(|x| x.abs() < ROUND_OFF)
    }

    /// numga's first test, at every one of its 216 frames: the contacts are finite and
    /// distinct, each lies on both conics of its edge and on its contact line, where the two
    /// conics have the same tangent; and the four contact lines of each face meet in one point
    /// (its first two meet at a finite point the other two pass through).
    #[test]
    fn deforming_cube_preserves_double_contact_and_face_concurrence() {
        const FACES: [[usize; 4]; 6] = [
            [0, 1, 3, 4],
            [0, 2, 5, 6],
            [1, 2, 7, 8],
            [3, 5, 9, 10],
            [4, 7, 9, 11],
            [6, 8, 10, 11],
        ];
        for config in conics::frames() {
            let contacts = config.contacts();
            for (e, [a, b]) in EDGES.iter().enumerate() {
                let pair = contacts[e];
                assert!(pair.iter().all(|p| p.c.iter().all(|x| x.is_finite())));
                assert!(coefficients_squared((pair[0] & pair[1]).c) > 0.01);
                for p in pair {
                    let (first, second) = (config.conics[*a].of(p), config.conics[*b].of(p));
                    assert!(coefficients_squared(first.c) > 0.01);
                    assert!((first & p).s().abs() < ROUND_OFF);
                    assert!((second & p).s().abs() < ROUND_OFF);
                    assert!((config.edge_lines[e] & p).s().abs() < ROUND_OFF);
                    assert!(vanishes(first.wedge(second).c));
                }
            }
            for face in FACES {
                let lines = face.map(|e| config.edge_lines[e]);
                let centre = lines[0] ^ lines[1];
                assert!((Line::infinity() & centre).s().abs() > 0.01);
                for l in &lines[2..] {
                    assert!(vanishes((*l ^ centre).c));
                }
            }
        }
    }

    /// numga's second test: for three sets of couplings the three completions of the eighth
    /// conic agree, and the configuration of mirrored inputs is the mirrored configuration.
    #[test]
    fn three_completions_agree_and_reflect_with_their_inputs() {
        let first = conics::at(0.0);
        let base = first.conics[0];
        let initial: [Line; 3] = core::array::from_fn(|k| first.edge_lines[k]);
        let wall = Line::new(1.0, 0.7, -0.3).normalized();
        let mirrored_base = wall >> base.of(wall << Point::slot());
        let mirrored_initial = initial.map(|l| wall >> l);
        for couplings in [[-0.3, 0.2, 0.15], [0.4, -0.2, 0.1], [0.2, 0.45, -0.1]] {
            let config = Configuration::new(base, initial, couplings);
            let mirror = Configuration::new(mirrored_base, mirrored_initial, couplings);
            for k in 1..3 {
                assert!(config.completions[k].max_abs_diff(&config.completions[0]) < ROUND_OFF);
                assert!(mirror.completions[k].max_abs_diff(&mirror.completions[0]) < ROUND_OFF);
            }
            for (m, q) in mirror.conics.iter().zip(config.conics) {
                let moved = wall >> q.of(wall << Point::slot());
                assert!(m.max_abs_diff(&moved) < ROUND_OFF);
            }
            for (m, l) in mirror.edge_lines.iter().zip(config.edge_lines) {
                assert!(m.approx_eq(&(wall >> l), ROUND_OFF));
            }
        }
    }

    /// The scenario's checks: every contact lies on both conics of its edge, at every frame.
    #[test]
    fn every_contact_lies_on_both_conics() {
        for config in conics::frames() {
            for (e, pair) in config.contacts().iter().enumerate() {
                for p in pair {
                    for corner in EDGES[e] {
                        assert!((config.conics[corner].of(*p) & *p).s().abs() < 1e-9);
                    }
                }
            }
        }
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws_with(super::draw, &super::framing(), 1.0);
    }
}
