//! numga's `mechanics/seismic_waves`: elastic waves through a planet with a liquid core and a
//! free surface, after a moon strikes it, and the travelling normal modes of a ball (VGA2D).
//!
//! Points and displacements are antivectors (in the plane the Hodge duals of vectors,
//! intrinsic); gradients and forces are vectors, read by their pairing with a point. Corner
//! hats give a constant strain on each triangle: each corner's gradient is the side facing it
//! over twice the triangle's patch. The material is a work form on pairs of displacement
//! gradients, swelling plus shearing, the exterior product taking the rotation out of the
//! gradient; bound to a triangle's test and trial corner gradients it is a form on the corner
//! displacements, and the pairing solve of the regressive product makes it a map from a
//! displacement to a force. Summed over the triangles it is the planet's stiffness; a lumped
//! mass map turns forces into motion, and velocity Verlet steps it. Zero shear stiffness gives
//! a liquid; its shear viscosity, the shear part of the same work form applied to the
//! velocity, damps the shear motion it cannot store. The corner gradients paired with and
//! wedged against the corner velocities read the divergence and the rotation of the motion,
//! which separate compressional from shear waves: P waves cross the core (the phases PKP),
//! S waves stop at it, and a Rayleigh wave runs round the surface.
//!
//! The normal modes of a uniform ball of cork come from gax's sparse generalized `eigh`, the
//! stiffness against the masses; two modes that differ by a turn of the ball share their
//! frequency, and together, a quarter cycle apart, make a wave that travels round it.
//!
//! The animation shows the planet after the blow, its rate of dilation beside its vorticity
//! (the motion exaggerated eighty times, beads riding the surface), the ball's travelling waves
//! in turn, its dilation beside its rotation, the planet's energy as it rings down, and the
//! ball's spectrum with the pair shown.

mod seismic;
#[cfg(test)]
mod tests;

use gax::field::Field;
use gax_numga_examples::points::circle;
use gax_numga_examples::shared::simplicial;
use gax_numga_examples::signal::frame;
use gax_numga_examples::{
    Anim, Axes, Canvas, Light, Marker, Point2, Rect, backdrop, caption, label_above, palette,
    run_with,
};
use seismic::*;

/// One loop: the ball's travelling waves at 40 ms a frame, the planet's ringing twice.
const SECONDS: f32 = 11.52;
/// The displacement drawn eighty times over; the readings that saturate the colours.
const DISPLACEMENT_SCALE: f64 = 80.0;
const DILATION_SCALE: f64 = 0.1;
const VORTICITY_SCALE: f64 = 0.3;
/// The beads on the surface, and the frames of their trails.
const MARKERS: usize = 12;
const TRAIL_FRAMES: usize = 8;
/// The drawn half-width of a disk's panel, a share of the radius.
const VIEW: f32 = 1.12;

/// Everything drawn, computed once.
struct Scenes {
    planet: Planet,
    ringing: Ringing,
    /// The planet's energy at each frame.
    energies: Vec<f64>,
    ball: Operators,
    /// The ball's squared frequencies past its rigid motions, and the first of each pair.
    spectrum: Vec<f64>,
    pairs: Vec<usize>,
    waves: Vec<[Field<Point>; 2]>,
    /// The rim of the drawing: the free surface and the interface around the core.
    rim: Vec<[usize; 2]>,
    /// The surface vertices the beads ride.
    markers: Vec<usize>,
}

fn scenes() -> Scenes {
    scenes_at(RADIAL_DIVISIONS)
}

/// The scenes on a disk of `divisions` radial intervals.
fn scenes_at(divisions: usize) -> Scenes {
    let planet = planet(divisions);
    let ringing = ringing(&planet);
    let ops = &planet.operators;
    let (_, blow) = moon();
    let start = blow_velocity(ops.vertices(), blow);
    // The energy is the blow's from the moment it lands: the resting frames have none.
    let mut energies = vec![0.0; APPROACH_FRAMES];
    let states = evolve(
        ops,
        Field::zeros(start.len()),
        start,
        DAMPING_TIME,
        DT,
        FRAMES,
        SUBSTEPS,
    );
    energies.extend(states.iter().map(|(d, v)| ops.energy(d, v)));
    let ball = ball(divisions);
    let (spectrum, pairs, waves) = modes(&ball);
    let rim = rim(&ops.mesh, &planet.layers);
    let markers = markers(&ops.mesh);
    Scenes {
        planet,
        ringing,
        energies,
        ball,
        spectrum,
        pairs,
        waves,
        rim,
        markers,
    }
}

/// The ball's spectrum past its rigid motions, the first mode of each pair, and the pairs'
/// travelling waves.
fn modes(ball: &Operators) -> (Vec<f64>, Vec<usize>, Vec<[Field<Point>; 2]>) {
    let (values, shapes) = ball.normal_modes(RIGID_MOTIONS + MODES);
    let pairs = paired(&values[RIGID_MOTIONS..]);
    let waves = pairs
        .iter()
        .map(|&i| travelling_pair(&shapes[RIGID_MOTIONS + i], &shapes[RIGID_MOTIONS + i + 1]))
        .collect();
    (values[RIGID_MOTIONS..].to_vec(), pairs, waves)
}

/// The edges of the free surface and those between a liquid and a solid face.
fn rim(mesh: &gax_numga_examples::mesh::Mesh<Point>, layers: &[usize]) -> Vec<[usize; 2]> {
    let topology = gax::mesh::Topology::new(mesh.vertices.len(), &mesh.faces).expect("a mesh");
    let interface = simplicial::sides(&topology)
        .into_iter()
        .filter(|s| layers[s.faces[0]] != layers[s.faces[1]])
        .map(|s| s.edge);
    mesh.boundary_edges()
        .into_iter()
        .chain(interface)
        .map(|e| mesh.edges[e])
        .collect()
}

/// The surface vertices nearest points turned evenly around it, from the top.
fn markers(mesh: &gax_numga_examples::mesh::Mesh<Point>) -> Vec<usize> {
    let surface: Vec<usize> = mesh
        .boundary_vertices()
        .iter()
        .enumerate()
        .filter_map(|(v, &on)| on.then_some(v))
        .collect();
    (0..MARKERS)
        .map(|k| {
            let angle =
                core::f64::consts::FRAC_PI_2 + core::f64::consts::TAU * k as f64 / MARKERS as f64;
            let target = turned(point(SURFACE_RADIUS, 0.0), angle);
            let distance = |v: usize| (mesh.vertices.site(v) - target).norm_squared();
            *surface
                .iter()
                .min_by(|a, b| distance(**a).total_cmp(&distance(**b)))
                .expect("a surface")
        })
        .collect()
}

/// A place of the plane, to draw.
fn at(p: Point) -> Point2 {
    let [x, y] = coordinates(p);
    Point2::xy(x as f32, y as f32)
}

/// The places of the vertices moved by an exaggerated displacement.
fn places(vertices: &Field<Point>, displacement: &Field<Point>) -> Vec<Point2> {
    vertices
        .iter()
        .zip(displacement.iter())
        .map(|(v, d)| at(v + d * DISPLACEMENT_SCALE))
        .collect()
}

/// A signed reading's light on its layer: dim where it vanishes, deepening to one hue or the
/// other.
fn reading_light(reading: f64, layer: usize, hues: [Light; 2]) -> Light {
    let r = reading.clamp(-1.0, 1.0) as f32;
    LAYERS[layer].mix_light(hues[usize::from(r < 0.0)], r.abs())
}

/// The layers' colours where nothing moves: the liquid core's, the solid shell's.
const LAYERS: [Light; 2] = [
    gax_numga_examples::light(0.32, 0.48, 1.0, 0.07),
    gax_numga_examples::light(0.75, 0.85, 1.0, 0.03),
];

/// The dilation's hues, swelling and compression; the rotation's, counter-clockwise and
/// clockwise.
const DILATION_HUES: [Light; 2] = [palette::SKY, palette::RED];
const ROTATION_HUES: [Light; 2] = [palette::PURPLE, palette::GREEN];

/// A deformed section: each face lit by its reading, over its layer's colour (which fills the
/// seams between faces), and its rim.
fn draw_section(
    c: &mut Canvas,
    ax: &Axes,
    s: &Scenes,
    layers: &[usize],
    at: &[Point2],
    readings: &Field<Scalar>,
    hues: [Light; 2],
) {
    // The layers underneath, which fill the seams between the faces' covers.
    let disk = |radius: f64| circle(Point2::xy(0.0, 0.0), radius as f32, 96);
    ax.fill(c, &disk(0.995 * SURFACE_RADIUS), LAYERS[1], 1.0);
    if layers.contains(&0) {
        ax.fill(c, &disk(CORE_RADIUS), LAYERS[0], 1.0);
    }
    let mesh = &s.planet.operators.mesh;
    for (f, face) in mesh.faces.iter().enumerate() {
        let triangle = face.map(|v| at[v]);
        let light = reading_light(readings.site(f).s(), layers[f], hues);
        ax.fill(c, &triangle, light, 1.0);
    }
    let rim: Vec<[Point2; 2]> = s.rim.iter().map(|&[a, b]| [at[a], at[b]]).collect();
    ax.stroke(c, &rim, 1.0, palette::INK.faded(0.45));
}

/// The planet after the blow: a reading of the motion, the moon, and the beads on the surface
/// with their trails.
fn draw_ringing(c: &mut Canvas, rect: Rect, s: &Scenes, k: usize, vorticity: bool) {
    let ax = Axes::equal(rect, Point2::xy(0.0, 0.0), VIEW);
    let vertices = s.planet.operators.vertices();
    let (displacement, dilation, rotation) = &s.ringing.frames[k];
    let moved = places(vertices, displacement);
    let (readings, hues) = if vorticity {
        (rotation.scaled(1.0 / VORTICITY_SCALE), ROTATION_HUES)
    } else {
        (dilation.scaled(1.0 / DILATION_SCALE), DILATION_HUES)
    };
    draw_section(c, &ax, s, &s.planet.layers, &moved, &readings, hues);
    for &m in &s.markers {
        let trail: Vec<Point2> = s.ringing.frames[k.saturating_sub(TRAIL_FRAMES - 1)..=k]
            .iter()
            .map(|(d, _, _)| at_marker(vertices, d, m))
            .collect();
        ax.polyline(c, &trail, 1.0, palette::INK.faded(0.6));
        ax.scatter(c, &[moved[m]], Marker::Dot, 4.0, palette::INK);
    }
    let moon = circle(at(s.ringing.moon[k]), MOON_RADIUS as f32, 32);
    ax.fill(c, &moon, palette::INK.faded(0.5), 1.0);
    ax.polyline(c, &moon, 1.0, palette::INK);
}

/// A bead's place in a frame.
fn at_marker(vertices: &Field<Point>, displacement: &Field<Point>, m: usize) -> Point2 {
    at(vertices.site(m) + displacement.site(m) * DISPLACEMENT_SCALE)
}

/// The ball moved by a travelling wave, its dilation or its rotation, both on the scale of the
/// larger of the two in the frame, so that a wave of either kind shows at full strength and in
/// its proportion.
fn draw_mode(c: &mut Canvas, rect: Rect, s: &Scenes, wave: usize, phase: f64, rotation: bool) {
    let ax = Axes::equal(rect, Point2::xy(0.0, 0.0), VIEW);
    let ball = &s.ball;
    let displacement = travelling(&s.waves[wave], phase);
    let dilations = ball.divergence.of(&displacement);
    let rotations = ball.vorticity(&displacement);
    let largest = dilations
        .iter()
        .chain(rotations.iter())
        .map(|r| r.s().abs())
        .fold(f64::MIN_POSITIVE, f64::max);
    let (readings, hues) = if rotation {
        (rotations.scaled(1.0 / largest), ROTATION_HUES)
    } else {
        (dilations.scaled(1.0 / largest), DILATION_HUES)
    };
    let layers = vec![1; ball.mesh.faces.len()];
    let at = places(ball.vertices(), &displacement);
    draw_section(c, &ax, s, &layers, &at, &readings, hues);
}

/// The planet's energy as it rings down, against the frame.
fn draw_energy(c: &mut Canvas, rect: Rect, s: &Scenes, k: usize) {
    let top = s.energies.iter().copied().fold(0.0, f64::max) as f32;
    let n = s.energies.len() as f32;
    let ax = Axes::new(rect, [0.0, n], [0.0, 1.15 * top]);
    let curve: Vec<Point2> = s
        .energies
        .iter()
        .enumerate()
        .map(|(i, e)| Point2::xy(i as f32, *e as f32))
        .collect();
    ax.polyline(c, &curve, 1.0, palette::ORANGE.faded(0.35));
    ax.polyline(c, &curve[..=k], 1.6, palette::ORANGE);
    ax.scatter(
        c,
        &[curve[k]],
        Marker::Dot,
        6.0,
        palette::ORANGE.whitened(0.3),
    );
    ax.frame(c, "ENERGY: KINETIC + ELASTIC", "FRAME", "");
}

/// The ball's angular frequencies, the pairs that travel marked and the one shown lit.
fn draw_spectrum(c: &mut Canvas, rect: Rect, s: &Scenes, wave: usize) {
    let n = s.spectrum.len();
    #[allow(clippy::disallowed_methods)] // an angular frequency from its square
    let frequencies: Vec<f32> = s
        .spectrum
        .iter()
        .map(|w2| w2.max(0.0).sqrt() as f32)
        .collect();
    let top = frequencies.iter().copied().fold(0.0, f32::max);
    let ax = Axes::new(rect, [-1.0, n as f32], [0.0, 1.1 * top]);
    let dots: Vec<Point2> = frequencies
        .iter()
        .enumerate()
        .map(|(i, w)| Point2::xy(i as f32, *w))
        .collect();
    ax.scatter(c, &dots, Marker::Dot, 3.5, palette::INK.faded(0.5));
    for (j, &i) in s.pairs.iter().enumerate() {
        let light = if j == wave {
            palette::YELLOW
        } else {
            palette::YELLOW.faded(0.4)
        };
        ax.line(c, dots[i], dots[i + 1], 2.0, light);
    }
    let shown = s.pairs[wave];
    ax.scatter(
        c,
        &dots[shown..shown + 2],
        Marker::Ring,
        8.0,
        palette::YELLOW,
    );
    ax.frame(c, "THE BALL'S FREQUENCIES: PAIRS TRAVEL", "MODE", "");
}

fn draw(c: &mut Canvas, s: &Scenes, t: f32) {
    backdrop(c);
    let phase = (t / SECONDS).rem_euclid(1.0);
    let k = frame((2.0 * phase).rem_euclid(1.0), s.ringing.frames.len());
    let cycles = s.waves.len() * CYCLE_FRAMES;
    let m = frame(phase, cycles);
    let (wave, step) = (m / CYCLE_FRAMES, m % CYCLE_FRAMES);
    let wave_phase = core::f64::consts::TAU * step as f64 / CYCLE_FRAMES as f64;
    let screen = c.rect();
    let (w, h) = (screen.width(), screen.height());
    let size = (h / 50.0).clamp(6.0, 11.0);
    let row = Rect::new(0.0, h * 0.21, w, h * 0.68);
    let panel = |i: usize| {
        row.part(i as f32 / 4.0, 0.0, (i + 1) as f32 / 4.0, 1.0)
            .inset(4.0, 4.0, 4.0, 4.0)
    };
    draw_ringing(c, panel(0), s, k, false);
    draw_ringing(c, panel(1), s, k, true);
    draw_mode(c, panel(2), s, wave, wave_phase, false);
    draw_mode(c, panel(3), s, wave, wave_phase, true);
    for (i, name) in ["DILATION RATE", "VORTICITY", "DILATION", "ROTATION"]
        .iter()
        .enumerate()
    {
        label_above(c, panel(i), name, size, 2.0);
    }
    let header = |i: usize| row.part(i as f32 / 2.0, 0.0, (i + 1) as f32 / 2.0, 1.0);
    label_above(
        c,
        header(0),
        "A MOON STRIKES A PLANET WITH A LIQUID CORE",
        size,
        size * 1.6,
    );
    label_above(
        c,
        header(1),
        "TRAVELLING MODES OF A BALL OF CORK",
        size,
        size * 1.6,
    );
    let bottom = Rect::new(0.0, h * 0.68, w, h);
    draw_energy(
        c,
        bottom
            .part(0.0, 0.0, 0.5, 1.0)
            .inset(48.0, 26.0, 24.0, 30.0),
        s,
        k,
    );
    draw_spectrum(
        c,
        bottom
            .part(0.5, 0.0, 1.0, 1.0)
            .inset(48.0, 26.0, 24.0, 30.0),
        s,
        wave,
    );
    caption(
        c,
        "SEISMIC WAVES: P, S AND SURFACE WAVES IN A PLANET",
        "STIFFNESS = WORK FORM ON CORNER-HAT GRADIENTS, SOLVED BY THE PAIRING; DIVERGENCE g & v, ROTATION g ^ v (VGA2D)",
    );
}

fn main() {
    run_with(
        Anim::new("seismic_waves", SECONDS).size(960, 540),
        scenes,
        draw,
    );
}
