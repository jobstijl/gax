//! numga's `surfaces/dirac_waves`: waves of a spinor field on a surface, carried by the surface's
//! Dirac operator, in the geometric algebra of space, VGA3D.
//!
//! The field is a quaternion on each vertex, an even multivector, and the Dirac operator of the
//! spin transformations takes it to an odd multivector on each face. The operator `D` is a field
//! map ([`gax::field::SparseMap`]) with vector cells, one for each face and each of its corners:
//! the edge that corner faces over minus twice the face's area. Its product with a vertex field,
//! `D * vertices`, multiplies each cell into the quaternion at its corner and sums over each
//! face's corners, taking even vertex fields to odd face fields. Its reverse `~D` turns every
//! product with it around: every cell reversed, and every coupling running the other way, from a
//! face to its corners.
//!
//! A wave alternates the two: the face field moved on by the Dirac operator of the vertex field,
//! then the vertex field moved back by the reverse of the face field, weighted by the areas, a
//! leapfrog that keeps the field's energy, the area-weighted squared size of both. Twice over the
//! step is the operator `Q = ~D M2 D` of the spin transformations, taking vertex fields to vertex
//! fields. With the quaternion left open, `Q * Rotor::slot()` is a map on quaternion fields that
//! the eigensolver takes, so the standing waves are its eigenfields against the vertex areas; on
//! the unit sphere their frequencies are the whole numbers, each held eight times its own.
//!
//! The animation shows a pulse of the `xy` plane spreading from the north pole and gathering
//! again at the south pole, and one standing wave of each of the first five frequencies over its
//! period. Each sphere glows red, green and blue by the vertex field's planes `yz`, `zx` and `xy`
//! together with the face field's directions `x`, `y` and `z` across them. Lengths are in
//! units of the sphere's radius and times in units of the radius over the waves' speed.

#[path = "../shared/surfaces_sphere.rs"]
mod surfaces_sphere;
#[path = "../shared/surfaces_spin.rs"]
#[allow(dead_code)] // the cube and its checkerboard are spin_transformations' and arap's
mod surfaces_spin;

use gax::field::Field;
use gax_numga_examples::mesh::Mesh;
use gax_numga_examples::palette;
use gax_numga_examples::run;
use gax_numga_examples::{
    Anim, Camera, Canvas, Lens, Light, ORIGIN3, Rect, backdrop, caption, label_under,
};
use std::sync::OnceLock;
use surfaces_spin::{Hit, render};

mod waves {
    use gax::field::{Field, SparseMap};
    use gax::vga3d::{Odd, Rotor};
    use gax_numga_examples::mesh::{Mesh, Scalar3, Vector3};
    use gax_numga_examples::signal::phasor;

    /// A quaternion: the even multivector at a vertex.
    pub type Quaternion = Rotor<(), f64>;
    /// The odd multivector on a face.
    pub type Spinor = Odd<(), f64>;

    /// The Dirac operator from vertices to faces `[F, V]`, with the face and vertex areas.
    pub struct Dirac {
        pub operator: SparseMap<Vector3>,
        pub face_areas: SparseMap<Scalar3>,
        pub vertex_areas: Field<Scalar3>,
    }

    /// Each face takes each corner by the edge it faces, over minus twice the face's area.
    pub fn dirac(mesh: &Mesh) -> Dirac {
        let face_areas = mesh.triangle_areas();
        let inverse = SparseMap::from_diagonal(&face_areas.map(|a| a.inverse()));
        let corners = mesh.at_corners(&mesh.triangle_edges());
        Dirac {
            operator: (&inverse * &corners).map_cells(|e| e * -0.5),
            face_areas: SparseMap::from_diagonal(&face_areas),
            vertex_areas: mesh.vertex_areas(),
        }
    }

    /// The vertex field at each step of a wave from `start`, and the face field half a step
    /// after it.
    pub fn spread(
        mesh: &Mesh,
        start: Field<Quaternion>,
        interval: f64,
        count: usize,
    ) -> Vec<(Field<Quaternion>, Field<Spinor>)> {
        let Dirac {
            operator,
            face_areas,
            vertex_areas,
        } = dirac(mesh);
        let back = &operator.reverse() * &face_areas;
        let inverse_areas = vertex_areas.map(|a| a.inverse());
        let mut vertices = start;
        let mut faces = Field::<Spinor>::zeros(mesh.faces.len());
        let mut states = Vec::with_capacity(count);
        for _ in 0..count {
            // The face field moved on by the Dirac operator of the vertex field.
            faces = &faces + &(&operator * &vertices).scaled(interval);
            states.push((vertices.clone(), faces.clone()));
            // The vertex field moved back by the reverse of the face field, by the areas.
            let pull = (&back * &faces).zip_with(&inverse_areas, |q, a| a * q);
            vertices = &vertices - &pull.scaled(interval);
        }
        states
    }

    /// The `count` standing waves of least frequency: their frequencies squared and their vertex
    /// fields, the eigenfields of `~D M2 D` against the vertex areas, computed in Lanczos blocks
    /// of `block` fields (at least the most copies of one frequency).
    pub fn standing(mesh: &Mesh, count: usize, block: usize) -> (Vec<f64>, Vec<Field<Quaternion>>) {
        let Dirac {
            operator,
            face_areas,
            vertex_areas,
        } = dirac(mesh);
        let wave = &(&(&operator.reverse() * &face_areas) * &operator) * Quaternion::slot();
        let metric = &SparseMap::from_diagonal(&vertex_areas) * Quaternion::slot();
        wave.eigh_with_block(&metric, count, block)
    }

    /// A standing wave at a phase of its period: the vertex field a cosine, and the face field,
    /// the Dirac operator of it over the frequency, a sine.
    pub fn oscillation(
        dirac: &Dirac,
        wave: &Field<Quaternion>,
        frequency: f64,
        phase: f64,
    ) -> (Field<Quaternion>, Field<Spinor>) {
        let turn = phasor(phase);
        let faces = (&dirac.operator * wave).scaled(turn.e01() / frequency);
        (wave.scaled(turn.e20()), faces)
    }
}

mod scenes {
    use super::waves::*;
    use gax::field::Field;
    use gax_numga_examples::mesh::{Mesh, Scalar3, Vector3};

    /// The sphere's subdivisions, the time step, and the steps between frames.
    pub const SUBDIVISIONS: usize = 3;
    pub const INTERVAL: f64 = 0.04;
    pub const EVERY: usize = 2;
    /// The pulse: its width, and the steps it runs, a little past the time to reach the far
    /// pole.
    pub const WIDTH: f64 = 0.25;
    pub const STEPS: usize = 84;
    /// The standing waves: the frequencies shown, each held eight times its own, after the four
    /// constant fields of frequency zero.
    pub const LEVELS: usize = 5;
    pub const CONSTANT: usize = 4;
    /// The most copies of one frequency the icosphere's symmetry holds together (twelve), with
    /// room: the eigensolver's block.
    pub const BLOCK: usize = 16;

    /// A smooth bump on the sphere about the given point, of width `WIDTH`.
    pub fn bump(sphere: &Mesh, centre: Vector3) -> Field<Scalar3> {
        let falloff = -0.5 / (WIDTH * WIDTH);
        sphere
            .vertices
            .map(|v| Scalar3::new(((v - centre).norm_squared() * falloff).exp()))
    }

    /// A quaternion field of one plane, `plane` times the bump at each vertex.
    fn planar(bump: &Field<Scalar3>, plane: Quaternion) -> Field<Quaternion> {
        bump.map(|b| b * plane)
    }

    /// The pulse: a bump of the `xy` plane at the north pole, spreading; the vertex and face
    /// fields at every step.
    pub fn pulse(sphere: &Mesh) -> Vec<(Field<Quaternion>, Field<Spinor>)> {
        let north = Vector3::new(0.0, 0.0, 1.0);
        let xy = Quaternion::new(0.0, 0.0, 0.0, 1.0);
        spread(sphere, planar(&bump(sphere, north), xy), INTERVAL, STEPS)
    }

    /// The number of waves of frequency below each level: the constant fields, then eight
    /// times each frequency.
    pub fn sizes() -> Vec<usize> {
        (0..=LEVELS)
            .map(|level| CONSTANT + 4 * level * (level + 1))
            .collect()
    }

    /// The frequencies squared of the standing waves, as the eigensolver found them, and those
    /// of the first few frequencies, one of each: of each frequency's waves, the part of a bump
    /// on the equator they hold, in all three planes, so every frequency is drawn the same way
    /// round.
    pub fn modes(sphere: &Mesh) -> (Vec<f64>, Vec<Field<Quaternion>>) {
        let sizes = sizes();
        let (squared, waves) = standing(sphere, sizes[LEVELS], BLOCK);
        let areas = sphere.vertex_areas();
        let all_planes = Quaternion::new(0.0, 1.0, 1.0, 1.0);
        let probe = planar(&bump(sphere, Vector3::new(1.0, 0.0, 0.0)), all_planes);
        // The probe's part along a wave: its area-weighted pairing with the wave.
        let along = |wave: &Field<Quaternion>| {
            let paired = wave.zip_with(&probe, |w, p| w.reverse().scalar_product(p));
            paired.zip_with(&areas, |s, a| s * a).sum().s()
        };
        let chosen = sizes
            .windows(2)
            .map(|range| {
                let share = &waves[range[0]..range[1]];
                let parts = share.iter().map(|wave| wave.scaled(along(wave)));
                parts
                    .reduce(|sum, part| &sum + &part)
                    .expect("a frequency's waves")
            })
            .collect();
        (squared, chosen)
    }
}

use scenes::*;
use waves::{Dirac, Quaternion, Spinor, dirac, oscillation};

/// One pass of the pulse: numga's 42 frames of 80 ms; the standing waves go round three times.
const SECONDS: f32 = 3.36;
const PERIODS: f32 = 3.0;
/// The percentile of each sphere's amplitudes over all its frames that shows at full brightness.
const EXPOSURE: f64 = 0.993;
/// The phases of a standing wave numga exposes it over.
const PHASES: usize = 20;
/// The share of its accent's light each axis shows at full exposure; and the lightness and
/// chroma (in Oklab) their mix is given, so that it glows in the hue of its strongest axes, as
/// vivid as the accents of the other examples, rather than as a pale grey of all three.
const GAIN: f32 = 0.5;
const LIGHTNESS: f32 = 0.65;
const CHROMA: f32 = 0.2;
/// The standing waves' sphere: one split coarser than numga's, to start in a second (the checks
/// use numga's).
const MODE_SUBDIVISIONS: usize = 2;
/// The view: numga's elevation and azimuth, and the camera's distance in sphere radii.
const ELEVATION: f32 = 0.14;
const AZIMUTH: f32 = -0.96;
const DISTANCE: f32 = 4.2;

/// The amplitudes a sphere shows, in units of its exposure: of the vertex field's planes `yz`,
/// `zx` and `xy` at each vertex, and of the face field's directions `x`, `y` and `z` on each
/// face.
struct Amplitudes {
    corners: Vec<[f32; 3]>,
    faces: Vec<[f32; 3]>,
}

impl Amplitudes {
    fn new(vertices: &Field<Quaternion>, faces: &Field<Spinor>) -> Self {
        let planes = |q: Quaternion| [q.e23(), q.e31(), q.e12()].map(|a| a.abs() as f32);
        let directions = |o: Spinor| [o.e1(), o.e2(), o.e3()].map(|a| a.abs() as f32);
        Amplitudes {
            corners: vertices.iter().map(planes).collect(),
            faces: faces.iter().map(directions).collect(),
        }
    }

    /// Every amplitude.
    fn all(&self) -> impl Iterator<Item = f32> + '_ {
        self.corners.iter().chain(&self.faces).flatten().copied()
    }

    /// The amplitudes over the exposure.
    fn exposed(mut self, exposure: f32) -> Self {
        for amplitude in self.corners.iter_mut().chain(&mut self.faces).flatten() {
            *amplitude /= exposure;
        }
        self
    }
}

/// The amplitude at the given share of all of them, in order: the exposure of a sphere over all
/// its frames.
fn percentile(amplitudes: impl Iterator<Item = f32>, share: f64) -> f32 {
    let mut sorted: Vec<f32> = amplitudes.collect();
    sorted.sort_by(f32::total_cmp);
    sorted[((sorted.len() - 1) as f64 * share).round() as usize]
}

/// The pulse's sphere and exposed frames; the standing waves' sphere, Dirac operator and waves,
/// with each wave's exposure.
struct Shown {
    pulse_sphere: Mesh,
    pulse: Vec<Amplitudes>,
    mode_sphere: Mesh,
    dirac: Dirac,
    modes: Vec<Field<Quaternion>>,
    mode_exposures: Vec<f32>,
}

/// A standing wave of frequency `level + 1` at a phase.
fn mode_at(dirac: &Dirac, wave: &Field<Quaternion>, level: usize, phase: f64) -> Amplitudes {
    let (vertices, faces) = oscillation(dirac, wave, (level + 1) as f64, phase);
    Amplitudes::new(&vertices, &faces)
}

fn shown() -> &'static Shown {
    static SHOWN: OnceLock<Shown> = OnceLock::new();
    SHOWN.get_or_init(|| {
        let pulse_sphere = surfaces_sphere::icosphere(SUBDIVISIONS);
        let frames = pulse(&pulse_sphere).into_iter().step_by(EVERY);
        let frames: Vec<Amplitudes> = frames.map(|(v, f)| Amplitudes::new(&v, &f)).collect();
        let exposure = percentile(frames.iter().flat_map(Amplitudes::all), EXPOSURE);
        let pulse = frames.into_iter().map(|a| a.exposed(exposure)).collect();
        let mode_sphere = surfaces_sphere::icosphere(MODE_SUBDIVISIONS);
        let dirac = dirac(&mode_sphere);
        let (_, modes) = scenes::modes(&mode_sphere);
        let phase = |k: usize| core::f64::consts::TAU * k as f64 / PHASES as f64;
        let mode_exposures = modes
            .iter()
            .enumerate()
            .map(|(level, wave)| {
                let frames: Vec<Amplitudes> = (0..PHASES)
                    .map(|k| mode_at(&dirac, wave, level, phase(k)))
                    .collect();
                percentile(frames.iter().flat_map(Amplitudes::all), EXPOSURE)
            })
            .collect();
        Shown {
            pulse_sphere,
            pulse,
            mode_sphere,
            dirac,
            modes,
            mode_exposures,
        }
    })
}

/// The light at a hit, on the surface's faint blue: each axis's accent (`x` red, `y` green, `z`
/// blue) by the vertex field's amplitude spread over the face between its corners, joined in
/// quadrature with the face's own, and squared, as the field's energy goes; the mix made vivid,
/// its hue and intensity kept.
fn colour(amplitudes: &Amplitudes, mesh: &Mesh, hit: &Hit) -> Light {
    let corners = mesh.faces[hit.face].map(|v| amplitudes.corners[v]);
    let face = amplitudes.faces[hit.face];
    let energy = |axis: usize| {
        let spread: f32 = (0..3)
            .map(|k| hit.weights[k] as f32 * corners[k][axis])
            .sum();
        gax::vga2d::Vector::<(), f32>::new(spread, face[axis])
            .norm_squared()
            .min(1.0)
    };
    let accents = [palette::red(), palette::green(), palette::blue()];
    let glow = |axis: usize| accents[axis].faded(GAIN * energy(axis));
    let mix = glow(0) + glow(1) + glow(2);
    let vivid = mix.perceptually(|c| c.with_lightness(LIGHTNESS).with_chroma(CHROMA));
    palette::grid() + vivid.to_gamut()
}

/// One sphere in its panel, with a label under it.
fn sphere_panel(c: &mut Canvas, rect: Rect, sphere: &Mesh, amplitudes: &Amplitudes, label: &str) {
    let cam = Camera::orbit(
        rect,
        ORIGIN3,
        DISTANCE,
        AZIMUTH,
        ELEVATION,
        Lens::Perspective(0.5),
    );
    render(c, &cam, rect, sphere, |hit| colour(amplitudes, sphere, hit));
    label_under(c, rect, label, 16.0, 0.8);
}

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let s = shown();
    let screen = c.rect();
    let (top, split) = (screen.height() * 0.12, screen.height() * 0.56);
    let rows = [
        Rect::new(0.0, top, screen.width(), split),
        Rect::new(0.0, split, screen.width(), screen.height()),
    ];
    let tile = |i: usize| rows[i / 3].column(i % 3, 3);
    let share = (t / SECONDS).rem_euclid(1.0);

    let frame = ((share * s.pulse.len() as f32) as usize).min(s.pulse.len() - 1);
    let time = (frame * EVERY) as f64 * INTERVAL;
    let label = format!("PULSE FROM THE NORTH POLE, T = {time:.2}");
    sphere_panel(c, tile(0), &s.pulse_sphere, &s.pulse[frame], &label);

    let phase = core::f64::consts::TAU * f64::from(share * PERIODS);
    for (level, wave) in s.modes.iter().enumerate() {
        let amplitudes = mode_at(&s.dirac, wave, level, phase).exposed(s.mode_exposures[level]);
        let label = format!("FREQUENCY {}, ONE OF {}", level + 1, 8 * (level + 1));
        sphere_panel(c, tile(level + 1), &s.mode_sphere, &amplitudes, &label);
    }
    caption(
        c,
        "DIRAC WAVES: SPINOR FIELDS ON THE SPHERE",
        "EVEN VERTEX AND ODD FACE FIELDS LEAPFROGGED BY THE DIRAC OPERATOR (VGA3D); STANDING WAVES FROM ITS SQUARE",
    );
}

fn main() {
    run(Anim::new("dirac waves", SECONDS).size(960, 540), draw);
}

#[cfg(test)]
mod tests {
    use super::scenes::*;
    use super::surfaces_sphere::icosphere;
    use super::waves::*;
    use gax::field::Field;
    use gax_numga_examples::mesh::Vector3;

    /// The field's energy across one step: the vertex field's area-weighted squared size, and
    /// the face fields' before and after it, paired.
    fn energy(
        dirac: &Dirac,
        vertices: &Field<Quaternion>,
        before: &Field<Spinor>,
        after: &Field<Spinor>,
    ) -> f64 {
        let at_vertices = vertices.zip_with(&dirac.vertex_areas, |q, a| a * (q * q.reverse()).s());
        let paired = before.zip_with(after, |b, a| b.reverse().scalar_product(a));
        let at_faces = &dirac.face_areas * &paired;
        (at_vertices.sum() + at_faces.sum()).s()
    }

    /// numga's `pulse` checks.
    #[test]
    fn the_pulse_keeps_its_energy_and_gathers_at_the_south_pole() {
        let sphere = icosphere(SUBDIVISIONS);
        let states = pulse(&sphere);
        let operator = dirac(&sphere);
        // The leapfrog keeps the energy: the vertex field's area-weighted norm, and the face
        // field's paired across each step.
        let energies: Vec<f64> = states
            .windows(2)
            .map(|pair| energy(&operator, &pair[1].0, &pair[0].1, &pair[1].1))
            .collect();
        assert!(
            energies
                .iter()
                .all(|e| (e - energies[0]).abs() < 1e-12 * energies[0].abs())
        );
        // Half a turn round the sphere on, the pulse has gathered at the south pole.
        let half_turn = (core::f64::consts::PI / INTERVAL).round() as usize;
        let gathered = &states[half_turn].0;
        let density: Vec<f64> = gathered
            .zip_with(&operator.vertex_areas, |q, a| a * (q * q.reverse()).s())
            .iter()
            .map(|d| d.s())
            .collect();
        let down = Vector3::new(0.0, 0.0, -1.0);
        let south = sphere.vertices.iter().map(|v| (v | down).s() > 0.9);
        let held: f64 = south
            .zip(&density)
            .filter(|(s, _)| *s)
            .map(|(_, d)| d)
            .sum();
        assert!(held > 0.5 * density.iter().sum::<f64>());
    }

    /// numga's `modes` checks: the frequencies are the whole numbers, each held eight times its
    /// own, after the constant fields, up to the mesh's resolution.
    #[test]
    fn the_standing_waves_have_whole_frequencies() {
        let sphere = icosphere(SUBDIVISIONS);
        let (squared, chosen) = modes(&sphere);
        assert_eq!(chosen.len(), LEVELS);
        assert!(
            squared[..CONSTANT].iter().all(|s| s.abs() < 1e-9),
            "{squared:?}"
        );
        let expected = (1..=LEVELS).flat_map(|n| core::iter::repeat_n(n as f64, 8 * n));
        for (measured, expected) in squared[CONSTANT..].iter().zip(expected) {
            #[allow(clippy::disallowed_methods)] // a frequency from its square, as numga
            let frequency = measured.sqrt();
            assert!(
                (frequency - expected).abs() < 0.03 * expected,
                "{frequency} {expected}"
            );
        }
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws(super::draw, 1.0);
    }
}
