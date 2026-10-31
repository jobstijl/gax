//! numga's `quantum/process_tomography`: learning a qubit's noisy gate from prepared states and
//! measured probabilities, with the states of a qubit as the points of the Bloch ball in
//! projective space (PGA3D). Every outcome probability is an affine function of the state, and
//! an affine function on space is a plane: a state is a point of weight one (the plane at
//! infinity reads its weight, `e0 & state`), an effect is a plane, and an outcome's probability
//! is the pairing `effect & state`. A channel is a map from points to points, `Point <- Point`:
//! a coherent turn is a rotor about the centre, and noise mixes a state with its images,
//! linearly in the point.
//!
//! Four preparations in general position determine such a map. Each outcome's dual is the point
//! where the other three outcomes' planes meet, each preparation's dual the plane through the
//! other three preparations: the reciprocal frames are meets and joins, and no scalar product
//! enters. The recovered outputs times the readings of their preparations' dual planes are
//! dyads, and their sum is the whole channel.
//!
//! The animation applies each learned channel (a rotation; with phase noise; with relaxation
//! toward the north pole) over and over to the sphere of pure states: the rotation keeps the
//! sphere, dephasing flattens it toward the axis, relaxation shrinks it onto the north pole.
//! Below, the probabilities the learned maps predict after as many uses, for states the
//! reconstruction never saw, against the exact ones, and the measured probability tables.

use gax_numga_examples::scene3::panel3;
use gax_numga_examples::signal::wave;
use gax_numga_examples::{
    Align, Anim, Axes, Camera, Canvas, Lens, Light, Marker, ORIGIN2, ORIGIN3, Point2, Rect,
    backdrop, caption, colormap, palette, run_with,
};

mod tomography {
    use gax::pga3d::{Plane, Point, Rotor, Spin, intrinsic};

    /// An effect is a plane, which reads states; a channel is a map on states, the points of
    /// the Bloch ball.
    pub type Effect = Plane;
    pub type Channel = Point<(Point,)>;
    /// The join of two states, the line through them.
    pub type Join = intrinsic::Line;

    /// The centre of the ball, the completely mixed state, and the north pole, towards which
    /// relaxation pulls.
    pub fn centre() -> Point {
        Point::origin()
    }
    pub fn north() -> Point {
        Point::xyz(0.0, 0.0, 1.0)
    }
    /// The plane at infinity, `e0`: it reads a point's weight.
    pub fn infinity() -> Effect {
        Plane::new(0.0, 0.0, 0.0, 1.0)
    }
    /// The planes of the field (`xy`) and of the tilt from north (`zx`), through the centre.
    pub fn xy() -> Spin {
        Spin::new(0.0, 0.0, 1.0)
    }
    pub fn zx() -> Spin {
        Spin::new(0.0, 1.0, 0.0)
    }

    /// The open state.
    fn open() -> Channel {
        Point::slot()
    }

    /// The state with Bloch vector `(x, y, z)`: the centre displaced by it.
    pub fn state(x: f64, y: f64, z: f64) -> Point {
        centre() + Point::direction(x, y, z)
    }

    /// Each of four preparations or outcomes, with the other three.
    const OTHERS: [[usize; 3]; 4] = [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]];

    /// A coherent turn in `plane` after independent phase noise and relaxation toward the north
    /// pole, each a mixture of a state with its images, composed.
    #[allow(clippy::disallowed_methods)] // the coherence kept is an amplitude, a square root of a probability
    pub fn noisy_gate(angle: f64, phase_flip: f64, loss: f64, plane: Spin) -> Channel {
        // The half turn about the field.
        let flipped = xy().cast::<Rotor>() >> open();
        // With probability `phase_flip`, the state is replaced by its half turn about the field.
        let dephasing = open() + (flipped - open()) * phase_flip;
        // A decayed state is at north: the plane at infinity reads the state's weight, and
        // north receives it.
        let decay = north() * (infinity() & open());
        // The part of a state across the field: the part a half turn about the field reverses.
        let transverse = (open() - flipped) * 0.5;
        // A fraction `loss` of the way to north; an unobserved decay keeps the amplitude
        // sqrt(1 - loss) of the coherence across the field, more than the 1 - loss a mixture of
        // decayed and untouched states would keep.
        let kept = (1.0 - loss).sqrt() - (1.0 - loss);
        let damping = open() + (decay - open()) * loss + transverse * kept;
        let rotation = (plane * (-angle / 2.0)).exp();
        // A map applied to another map composes them; the rotor then turns every output.
        rotation >> damping.of(dephasing)
    }

    /// Each outcome's probability for each preparation passed through the channel,
    /// `[preparation][outcome]`: the effect's plane read at the output's point.
    pub fn probabilities(device: Channel, prepared: &[Point], effects: &[Effect]) -> Vec<Vec<f64>> {
        prepared
            .iter()
            .map(|p| {
                let out = device.of(*p);
                effects.iter().map(|e| (*e & out).s()).collect()
            })
            .collect()
    }

    /// The channel that gives the measured probabilities on four preparations in general
    /// position, extended to every state by linearity.
    pub fn reconstruct(prepared: &[Point], effects: &[Effect], measured: &[Vec<f64>]) -> Channel {
        // The point where the other three effects' planes meet, scaled so that its own effect
        // reads one there.
        let measurement_dual = OTHERS.iter().zip(effects).map(|([a, b, c], e)| {
            let meet = effects[*a] ^ effects[*b] ^ effects[*c];
            meet / (*e & meet)
        });
        let measurement_dual: Vec<Point> = measurement_dual.collect();
        // The plane through the other three preparations, scaled to read one on its own.
        let preparation_dual = OTHERS.iter().zip(prepared).map(|([a, b, c], p)| {
            let join = prepared[*a] & prepared[*b] & prepared[*c];
            join / (join & *p)
        });
        measured
            .iter()
            .zip(preparation_dual)
            .map(|(row, dual)| {
                let output: Point = row
                    .iter()
                    .zip(&measurement_dual)
                    .map(|(p, m)| *m * *p)
                    .sum();
                output * (dual & open())
            })
            .sum()
    }

    /// One use of the channel, then two, three, and so on, as composed maps.
    pub fn powers(device: Channel, uses: usize) -> Vec<Channel> {
        std::iter::successors(Some(device), |last| Some(device.of(*last)))
            .take(uses.max(1))
            .collect()
    }

    /// Pure states on a closed latitude-longitude mesh of the Bloch sphere, `[latitude][longitude]`:
    /// north, tilted down and turned about the field.
    pub fn sphere(latitudes: usize, longitudes: usize) -> Vec<Vec<Point>> {
        (0..=latitudes)
            .map(|i| {
                let inclination = core::f64::consts::PI * i as f64 / latitudes as f64;
                let tilt = (zx() * (-inclination / 2.0)).exp();
                (0..=longitudes)
                    .map(|j| {
                        let azimuth = core::f64::consts::TAU * j as f64 / longitudes as f64;
                        let turn = (xy() * (-azimuth / 2.0)).exp();
                        turn >> (tilt >> north())
                    })
                    .collect()
            })
            .collect()
    }

    // --- the scenario ---

    pub const LABELS: [&str; 3] = ["ROTATION", "ROTATION + DEPHASING", "ROTATION + RELAXATION"];
    pub const TURNS: [f64; 3] = [0.075, 0.075, 0.075];
    pub const PHASE_FLIP: [f64; 3] = [0.0, 0.06, 0.0];
    pub const LOSS: [f64; 3] = [0.0, 0.0, 0.10];
    pub const USES: usize = 48;

    pub fn rotation_plane() -> Spin {
        Spin::new(1.0, 1.0, 1.0).normalized()
    }

    /// The unit directions to a regular tetrahedron's corners, `[x, y, z]`.
    pub fn tetrahedron() -> [[f64; 3]; 4] {
        [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ]
        .map(|d| gax::vga3d::Vector::from_coeffs(d).normalized().c)
    }

    /// The four calibration states, at the tetrahedron's corners.
    pub fn prepared() -> Vec<Point> {
        tetrahedron()
            .iter()
            .map(|[x, y, z]| state(*x, *y, *z))
            .collect()
    }

    /// The detector's planes through the centre facing the tetrahedron's corners, with the
    /// plane at infinity: four effects that add up to it, the weight.
    pub fn effects() -> Vec<Effect> {
        tetrahedron()
            .iter()
            .map(|[x, y, z]| (infinity() + Plane::new(*x, *y, *z, 0.0)) * 0.25)
            .collect()
    }

    /// The three gates as they are.
    pub fn devices() -> Vec<Channel> {
        (0..3)
            .map(|c| noisy_gate(TURNS[c], PHASE_FLIP[c], LOSS[c], rotation_plane()))
            .collect()
    }

    /// The ideal measurement probabilities of the prepared states, `[channel][prep][outcome]`, and
    /// the reconstructed maps.
    #[allow(clippy::type_complexity)]
    pub fn run_tomography() -> (Vec<Vec<Vec<f64>>>, Vec<Channel>) {
        let (prepared, effects) = (prepared(), effects());
        let measured: Vec<Vec<Vec<f64>>> = devices()
            .iter()
            .map(|d| probabilities(*d, &prepared, &effects))
            .collect();
        let learned = measured
            .iter()
            .map(|m| reconstruct(&prepared, &effects, m))
            .collect();
        (measured, learned)
    }

    /// Pure and mixed states absent from the reconstruction: along `±x, ±y, ±z`, at Bloch
    /// lengths 1, 0.7, 0.3, 1, 0.7 and 0.
    pub fn heldout() -> Vec<Point> {
        let axes = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let lengths = [1.0, 0.7, 0.3, -1.0, -0.7, 0.0];
        axes.iter()
            .cycle()
            .zip(lengths)
            .map(|([x, y, z], l)| state(x * l, y * l, z * l))
            .collect()
    }

    /// Exact and predicted outcome probabilities for the held-out states, after `uses` uses of
    /// each gate: `[channel]` of `([state][outcome], [state][outcome])`.
    #[allow(clippy::type_complexity)]
    pub fn validation(learned: &[Channel], uses: usize) -> Vec<(Vec<Vec<f64>>, Vec<Vec<f64>>)> {
        let (states, effects) = (heldout(), effects());
        devices()
            .iter()
            .zip(learned)
            .map(|(actual, l)| {
                let actual = powers(*actual, uses)[uses - 1];
                let predicted = powers(*l, uses)[uses - 1];
                (
                    probabilities(actual, &states, &effects),
                    probabilities(predicted, &states, &effects),
                )
            })
            .collect()
    }

    /// What the tetrahedron and a single diameter of probes join to: a volume (six times the
    /// tetrahedron's), which every state lies in, and the line of the field, which only the
    /// states on it lie in.
    pub fn completeness() -> (intrinsic::Scalar, Join) {
        let p = prepared();
        let south = Point::xyz(0.0, 0.0, -1.0);
        (p[0] & p[1] & p[2] & p[3], north() & south)
    }
}

use tomography::*;

const SECONDS: f32 = 9.6;
const LATITUDES: usize = 12;
const LONGITUDES: usize = 24;

/// The ends of the ball's three axes.
fn axes() -> [gax::pga3d::Point; 3] {
    [state(1.0, 0.0, 0.0), state(0.0, 1.0, 0.0), north()]
}

/// What the frames share, computed once: the experiment, the learned maps and their powers.
struct Data {
    prepared: Vec<gax::pga3d::Point>,
    measured: Vec<Vec<Vec<f64>>>,
    learned: Vec<Channel>,
    powers: Vec<Vec<Channel>>,
    surface: Vec<Vec<gax::pga3d::Point>>,
    /// The tetrahedron's join, and the two opposite probes'.
    completeness: (f64, Join),
}

fn data() -> Data {
    let (measured, learned) = run_tomography();
    let powers = learned.iter().map(|l| powers(*l, USES)).collect();
    let (volume, line) = completeness();
    Data {
        prepared: prepared(),
        completeness: (volume.s(), line),
        measured,
        learned,
        powers,
        surface: sphere(LATITUDES, LONGITUDES),
    }
}

/// A table of values as coloured cells (a heat map), `[row][column]`, with the values written in
/// and the rows and columns named. The cells are dimmed so that the values, light added on
/// top, stand out.
fn table(
    c: &mut Canvas,
    rect: Rect,
    cells: &[Vec<f64>],
    names: &[&str],
    colour: impl Fn(f64) -> Light,
) {
    let rows = cells.len() as f32;
    let cols = cells[0].len() as f32;
    let (cw, ch) = (rect.width() / cols, rect.height() / rows);
    let (right, down) = (Point2::direction(cw, 0.0), Point2::direction(0.0, ch));
    let size = (ch * 0.32).min(10.0);
    // Text sits on its baseline: half a text height below a centre.
    let baseline = Point2::direction(0.0, size * 0.5);
    for (i, row) in cells.iter().enumerate() {
        for (j, v) in row.iter().enumerate() {
            let lo = rect.lo + right.gp(j as f32) + down.gp(i as f32);
            let cell = Rect {
                lo,
                hi: lo + right + down,
            };
            let corners = [cell.lo, cell.top_right(), cell.hi, cell.bottom_left()];
            c.fill(&corners, (colour(*v)).faded(0.45), 1.0);
            // Probabilities, without the leading zero, to fit the cells.
            let text = format!("{v:.2}");
            let at = cell.centre() + baseline;
            let text = text.trim_start_matches('0');
            c.text(text, at, size, palette::INK, Align::Center);
        }
        let name = rect.lo + down.gp(i as f32 + 0.5) + baseline - Point2::direction(4.0, 0.0);
        c.text(names[i], name, size, palette::INK, Align::Right);
    }
    for (j, name) in names.iter().enumerate().take(cols as usize) {
        let at = rect.bottom_left() + right.gp(j as f32 + 0.5) + baseline.gp(2.8);
        c.text(name, at, size, palette::INK, Align::Center);
    }
}

/// The colours of the three learned channels.
fn colours() -> [Light; 3] {
    [palette::SKY, palette::ORANGE, palette::PURPLE]
}

/// Learned channel `k` applied `uses` times to the sphere of pure states and to the probes,
/// hollow and filled markers pairing each preparation with its image.
fn draw_channel(
    c: &mut Canvas,
    rect: Rect,
    d: &Data,
    k: usize,
    uses: usize,
    azimuth: f32,
    small: f32,
) {
    let down = Point2::direction(0.0, 1.0);
    let colours = colours();
    let probe_colours = [
        palette::SKY,
        palette::ORANGE,
        palette::GREEN,
        palette::PURPLE,
    ];
    let map = d.powers[k][uses - 1];
    let image: Vec<Vec<_>> = d
        .surface
        .iter()
        .map(|row| row.iter().map(|s| map.of(*s)).collect())
        .collect();
    let cam = Camera::orbit(
        rect,
        ORIGIN3,
        4.6,
        azimuth,
        24.0f32.to_radians(),
        Lens::Perspective(0.62),
    );
    panel3(c, rect, cam, |s| {
        s.sphere_wire(centre(), 1.0, 16, palette::GRID.faded(0.45));
        // Each axis from its end through the centre to the opposite end, the centre's mirror
        // image of it.
        for a in axes() {
            s.seg(centre() + (centre() - a), a, 1.0, palette::GRID.faded(0.8));
        }
        // The image: a translucent surface with its mesh lines.
        for i in 0..LATITUDES {
            for j in 0..LONGITUDES {
                let (p, q, r, u) = (
                    image[i][j],
                    image[i + 1][j],
                    image[i + 1][j + 1],
                    image[i][j + 1],
                );
                let lit = s.lit(p, q, r, colours[k]);
                s.quad(p, q, r, u, lit, 0.16);
            }
        }
        let mesh = colours[k].faded(0.85);
        for row in &image {
            s.polyline(row, 1.0, mesh);
        }
        for j in (0..LONGITUDES).step_by(2) {
            let meridian: Vec<_> = image.iter().map(|row| row[j]).collect();
            s.polyline(&meridian, 1.0, mesh);
        }
        // Hollow and filled markers pair each preparation with its image.
        for (p, pc) in d.prepared.iter().zip(probe_colours) {
            s.dot(*p, Marker::Ring, 8.0, pc);
            s.dot(map.of(*p), Marker::Dot, 6.0, pc);
        }
    });
    // The title a line below the band's top, clear of the caption.
    let title = rect.top_middle() + down.gp(small * 1.2);
    c.text(LABELS[k], title, small * 1.15, colours[k], Align::Center);
}

/// The number of uses over the band of channels, and under its two bottom corners what each
/// probe set joins to.
fn draw_completeness(c: &mut Canvas, band: Rect, d: &Data, uses: usize, small: f32) {
    let w = band.width();
    let up = Point2::direction(0.0, -1.0);
    let note = band.bottom_middle() + up.gp(small * 0.8);
    let text = format!("{uses} USES OF EACH LEARNED CHANNEL");
    c.text(&text, note, small * 1.1, palette::INK, Align::Center);
    // What each probe set joins to, under the band's two bottom corners, a line below the note:
    // the tetrahedron a volume, every state; two opposite probes only the line through them.
    let corners = [
        (
            band.bottom_left() + Point2::direction(0.02 * w, small * 1.3),
            Align::Left,
        ),
        (
            band.hi + Point2::direction(-0.02 * w, small * 1.3),
            Align::Right,
        ),
    ];
    let (volume, line) = d.completeness;
    let texts = [
        format!(
            "FOUR TETRAHEDRAL PROBES JOIN TO A VOLUME {:.2}",
            volume.abs()
        ),
        format!(
            "TWO OPPOSITE PROBES JOIN TO A LINE, {:.0} LONG",
            line.norm()
        ),
    ];
    for (text, (at, align)) in texts.iter().zip(corners) {
        c.text(text, at, small * 0.85, palette::GRID, align);
    }
}

/// Predictions after `uses` uses for unseen states, against the exact probabilities.
fn draw_predictions(c: &mut Canvas, rect: Rect, d: &Data, uses: usize) {
    let colours = colours();
    // The probabilities reach a half; the axes reach a little further on both sides, so that
    // their ticks are a fifth apart and their labels clear of one another.
    let ax = Axes::equal(rect, Point2::xy(0.27, 0.27), 0.38);
    ax.frame(c, "UNSEEN STATES", "EXACT", "");
    ax.line(c, ORIGIN2, Point2::xy(0.54, 0.54), 1.0, palette::GRID);
    for ((exact, predicted), colour) in validation(&d.learned, uses).iter().zip(colours) {
        // Each probability as a point of the plot: exact across, predicted up.
        let pts: Vec<gax::pga2d::Point> = exact
            .iter()
            .flatten()
            .zip(predicted.iter().flatten())
            .map(|(a, b)| gax::pga2d::Point::xy(*a, *b))
            .collect();
        ax.scatter(c, &pts, Marker::Dot, 4.5, colour.faded(0.85));
    }
}

/// The measured probability tables, one per channel in the last three quarters of the width
/// between heights `y0` and `y1`: preparation by outcome, coloured on a log scale.
fn draw_tables(c: &mut Canvas, d: &Data, quarter: f32, [y0, y1]: [f32; 2], small: f32) {
    let up = Point2::direction(0.0, -1.0);
    let colours = colours();
    let names = ["A", "B", "C", "D"];
    let lowest = d
        .measured
        .iter()
        .flatten()
        .flatten()
        .copied()
        .filter(|v| *v > 0.0)
        .fold(f64::INFINITY, f64::min);
    let shade = |v: f64| {
        let t = (v.max(lowest).ln() - lowest.ln()) / (0.5f64.ln() - lowest.ln());
        colormap::turbo(t as f32)
    };
    let size = (y1 - y0).min(quarter * 0.7);
    for (k, table_values) in d.measured.iter().enumerate() {
        let lo = Point2::xy(quarter * (k as f32 + 1.0) + quarter * 0.22, y0);
        let rect = Rect {
            lo,
            hi: lo + Point2::direction(size, size),
        };
        table(c, rect, table_values, &names, shade);
        let title = rect.top_middle() + up.gp(small * 0.6);
        c.text(LABELS[k], title, small, colours[k], Align::Center);
    }
}

fn draw(c: &mut Canvas, d: &Data, t: f32) {
    backdrop(c);
    let screen = c.rect();
    let (w, h) = (screen.width(), screen.height());
    let phase = (t / SECONDS).clamp(0.0, 0.9999);
    let uses = 1 + (phase * USES as f32) as usize;
    let small = (h / 50.0).min(11.0);
    // Each learned channel in thirds of a band across the screen.
    let band = Rect::new(0.0, (h * 0.13).floor(), w, (h * 0.66).floor());
    let azimuth = (-54.0f32).to_radians() + 0.5 * wave(core::f32::consts::TAU * phase);
    for k in 0..3 {
        draw_channel(c, band.column(k, 3), d, k, uses, azimuth, small);
    }
    draw_completeness(c, band, d, uses, small);
    // Below, a square plot in the first quarter and the tables in the others.
    let quarter = w / 4.0;
    let (y0, y1) = (h * 0.76, h * 0.94);
    let side = Point2::direction(y1 - y0, y1 - y0);
    let lo = Point2::xy(quarter * 0.3, y0);
    draw_predictions(c, Rect { lo, hi: lo + side }, d, uses);
    draw_tables(c, d, quarter, [y0, y1], small);
    caption(
        c,
        "PROCESS TOMOGRAPHY: A NOISY GATE FROM FOUR PROBES",
        "PGA3D: STATES ARE POINTS, EFFECTS PLANES; DUAL FRAMES BY MEETS AND JOINS",
    );
}

fn main() {
    run_with(
        Anim::new("process tomography", SECONDS).size(960, 540),
        data,
        draw,
    );
}

#[cfg(test)]
mod tests {
    use super::tomography::*;
    use gax::pga3d::{Plane, Point, Spin};
    use gax::vga3d::{Pseudoscalar, Vector};
    use gax_numga_examples::rng::{Draw, rng};
    use gax_numga_examples::testing::near;

    /// Bloch vectors in random directions with lengths spread evenly over `[0, 1]`.
    fn blochs(seed: u64, n: usize) -> Vec<Vector> {
        let mut r = rng(seed);
        (0..n)
            .map(|k| r.direction::<Vector>() * (k as f64 / (n - 1) as f64))
            .collect()
    }

    /// The state with Bloch vector `r`.
    fn at(r: Vector) -> Point {
        state(r.e1(), r.e2(), r.e3())
    }

    /// Dephasing and amplitude damping followed by Rodrigues' rotation about the unit `axis`,
    /// in plain vectors of space, written out with the cross product (the vector at right
    /// angles to the plane `axis ^ v`) rather than a rotor: independent of the channel's points
    /// and planes.
    #[allow(clippy::disallowed_methods)] // the reference it is checked against
    fn analytical_bloch(r: Vector, angle: f64, phase_flip: f64, loss: f64, axis: Vector) -> Vector {
        let transverse = (1.0 - 2.0 * phase_flip) * (1.0 - loss).sqrt();
        let v = Vector::new(
            transverse * r.e1(),
            transverse * r.e2(),
            (1.0 - loss) * r.e3() + loss,
        );
        let cross = -(Pseudoscalar::new(1.0) * (axis ^ v));
        let along = (axis | v).s();
        v * angle.cos() + cross * angle.sin() + axis * ((1.0 - angle.cos()) * along)
    }

    #[test]
    fn channels_match_the_independent_bloch_law_and_normalized_born_probabilities() {
        let angles = [0.0, 0.47, -0.9, 0.3];
        let phase_flip = [0.0, 0.22, 0.08, 0.5];
        let loss = [0.0, 0.45, 1.0, 0.3];
        let axis = Vector::new(2.0, -1.0, 3.0).normalized();
        let plane = Spin::new(2.0, -1.0, 3.0).normalized();
        let blochs = blochs(180, 29);
        let detector = tetrahedron();
        let effects: Vec<Effect> = detector
            .iter()
            .map(|[x, y, z]| (infinity() + Plane::new(*x, *y, *z, 0.0)) * 0.25)
            .collect();
        // The planes through the centre across each axis read a point's coordinates.
        let readers = [
            Plane::new(1.0, 0.0, 0.0, 0.0),
            Plane::new(0.0, 1.0, 0.0, 0.0),
            Plane::new(0.0, 0.0, 1.0, 0.0),
        ];
        for c in 0..4 {
            let device = noisy_gate(angles[c], phase_flip[c], loss[c], plane);
            let prepared: Vec<Point> = blochs.iter().map(|r| at(*r)).collect();
            let probs = probabilities(device, &prepared, &effects);
            for (r, (p, row)) in blochs.iter().zip(prepared.iter().zip(&probs)) {
                let out = device.of(*p);
                let want = analytical_bloch(*r, angles[c], phase_flip[c], loss[c], axis);
                assert!(near(out, at(want), 1e-8), "{out:?} vs {want:?}");
                assert!(((infinity() & out).s() - 1.0).abs() < 1e-8);
                let read = Vector::from_coeffs(readers.map(|e| (e & out).s()));
                assert!(read.norm() <= 1.0 + 1e-8);
                for (prob, [x, y, z]) in row.iter().zip(&detector) {
                    let expected = (1.0 + (want | Vector::new(*x, *y, *z)).s()) / 4.0;
                    assert!((prob - expected).abs() < 1e-8);
                    assert!(*prob >= -1e-8);
                }
                assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-8);
            }
        }
    }

    #[test]
    fn reconstruction_predicts_mixed_states_with_an_independent_measurement_frame() {
        let angles = [0.31, -0.76];
        let phase_flip = [0.12, 0.07];
        let loss = [0.26, 0.54];
        let plane = Spin::new(1.0, 0.0, 1.0).normalized();
        // Four mixed preparations in general position, while the four detector directions are
        // rotated independently: the reconstruction cannot rely on matching the two frames.
        let prepared = [
            state(0.9, 0.1, 0.0),
            state(-0.2, 0.8, 0.1),
            state(0.1, -0.3, -0.7),
            state(-0.5, -0.4, 0.6),
        ];
        let yz = Spin::new(1.0, 0.0, 0.0);
        let rotation = (xy() * -0.23).exp() * (yz * 0.17).exp();
        let effects: Vec<Effect> = super::tomography::effects()
            .iter()
            .map(|e| rotation >> *e)
            .collect();
        let unseen: Vec<Point> = blochs(972, 37).into_iter().map(at).collect();
        for c in 0..2 {
            let device = noisy_gate(angles[c], phase_flip[c], loss[c], plane);
            let measured = probabilities(device, &prepared, &effects);
            let learned = reconstruct(&prepared, &effects, &measured);
            for s in &unseen {
                assert!(near(learned.of(*s), device.of(*s), 1e-12));
            }
        }
    }

    #[test]
    fn learned_channels_predict_unseen_measurements_and_repeated_applications() {
        let (measured, learned) = run_tomography();
        for (exact, predicted) in validation(&learned, 1) {
            for (a, b) in exact.iter().flatten().zip(predicted.iter().flatten()) {
                assert!((a - b).abs() < 1e-12);
            }
        }
        for (l, m) in learned.iter().zip(&measured) {
            let again = probabilities(*l, &prepared(), &effects());
            for (a, b) in again.iter().flatten().zip(m.iter().flatten()) {
                assert!((a - b).abs() < 1e-12);
            }
        }
        let blochs = [
            Vector::new(0.2, -0.3, 0.4),
            Vector::new(-0.6, 0.1, 0.2),
            Vector::new(0.0, 0.0, -1.0),
        ];
        let axis = Vector::new(1.0, 1.0, 1.0).normalized();
        for (c, l) in learned.iter().enumerate() {
            let mut expected = blochs;
            for accumulated in powers(*l, 4) {
                expected =
                    expected.map(|r| analytical_bloch(r, TURNS[c], PHASE_FLIP[c], LOSS[c], axis));
                for (r, want) in blochs.iter().zip(&expected) {
                    assert!(near(accumulated.of(at(*r)), at(*want), 1e-8));
                }
            }
        }
    }

    /// Four probes join to six times the volume of their tetrahedron; two opposite ones only to
    /// the line through them, which the completely mixed state lies on and the tetrahedron's
    /// states do not.
    #[test]
    #[allow(clippy::disallowed_methods)] // the volume's closed form
    fn completeness_distinguishes_a_spanning_tetrahedron_from_two_axial_probes() {
        let (volume, line) = completeness();
        let want = 16.0 / (3.0 * 3f64.sqrt());
        assert!((volume.s().abs() - want).abs() < 1e-12, "{volume:?}");
        assert!((line & centre()).c.iter().all(|x| x.abs() < 1e-12));
        let off = prepared()
            .iter()
            .flat_map(|p| (line & *p).c)
            .fold(0.0, |m: f64, x| m.max(x.abs()));
        assert!(off > 0.1, "{off}");
    }

    /// The display surface consists of pure states, of weight one on the unit sphere, before
    /// any channel acts, and the composed channels animate: two frames differ.
    #[test]
    fn the_surface_is_pure_and_the_animation_draws() {
        for row in sphere(8, 16) {
            for s in row {
                assert!(((infinity() & s).s() - 1.0).abs() < 1e-8);
                assert!(((centre() & s).norm() - 1.0).abs() < 1e-8, "{s:?}");
            }
        }
        let d = super::data();
        let mut draw = |c: &mut gax_numga_examples::Canvas, t| super::draw(c, &d, t);
        let anim = gax_numga_examples::Anim::new("t", super::SECONDS).size(480, 270);
        let a = gax_numga_examples::app::frame(&anim, 0.1, &mut draw);
        let b = gax_numga_examples::app::frame(&anim, 5.0, &mut draw);
        assert!(a.mean().luma() > 0.0);
        assert!(a.mean() != b.mean());
    }
}
