//! The disk's and the sky's light: blackbody colours at their visible luminance, the disk's
//! clumpy texture carried along its orbits, and a catalogue of stars over a dusty band.

use std::sync::OnceLock;

use crate::disk::carried;
use crate::kerr::{V, xy};
use crate::scenes::{FLOW_TIME, Glint, camera_hole};
use gax::{vga2d, vga3d};
use gax_numga_examples::rng::{Draw, rng};
use gax_numga_examples::signal::{phasor, wave};
use gax_numga_examples::{Light, light, measure};

/// Wavelengths across the visible band, nm, and the second radiation constant, nm K.
const WAVELENGTHS: [f64; 2] = [380.0, 780.0];
const RADIATION_CONSTANT: f64 = 1.4388e7;
/// The step of the sum over the visible band, nm.
const WAVELENGTH_STEP: f64 = 5.0;
/// A blackbody at this temperature, K, shows at [`MID_GREY`].
const EXPOSURE_TEMPERATURE: f64 = 6500.0;
/// The luminance the exposure temperature shows at: AgX's mid-grey.
const MID_GREY: f32 = 0.18;
/// Blackbody colours look pale; their chroma is exaggerated to show the shift, as numga's.
const SATURATION: f32 = 2.5;
/// The disk's clumpy texture: how many random waves make it, their seed, and its contrast.
const CLUMP_MODES: usize = 24;
const CLUMP_SEED: u64 = 5;
const CLUMP_CONTRAST: f64 = 0.35;
/// The clump waves turn this many times around the disk, at least and at most.
const MIN_TURNS: usize = 3;
const MAX_TURNS: usize = 17;
/// The sky: how many stars, their seed, and the brightness of its light.
const STAR_COUNT: usize = 15_000;
const SKY_SEED: u64 = 8;
const SKY_BRIGHTNESS: f32 = 0.3;
/// The stars are looked up on a grid of this many cells across each face of a cube.
const SKY_GRID: usize = 48;

/// The eye's sensitivity at a wavelength in nm, the CIE 1931 ȳ, by the two-lobe fit of
/// Wyman, Sloan and Shirley (2013).
fn eye_sensitivity(wavelength: f64) -> f64 {
    let lobe = |centre: f64, below: f64, above: f64| {
        let width = if wavelength < centre { below } else { above };
        (-0.5 * ((wavelength - centre) / width).powi(2)).exp()
    };
    0.821 * lobe(568.8, 46.9, 40.5) + 0.286 * lobe(530.9, 16.3, 31.1)
}

/// The visible luminance of a blackbody at a temperature in K, in relative units: Planck's
/// law weighed by the eye's sensitivity, summed over the visible band. A temperature of zero
/// is dark.
fn visible_luminance(temperature: f64) -> f64 {
    let [shortest, longest] = WAVELENGTHS;
    let samples = ((longest - shortest) / WAVELENGTH_STEP) as usize;
    (0..=samples)
        .map(|k| {
            let wavelength = shortest + WAVELENGTH_STEP * k as f64;
            let planck = (RADIATION_CONSTANT / (wavelength * temperature)).exp_m1();
            eye_sensitivity(wavelength) / (wavelength.powi(5) * planck)
        })
        .sum()
}

/// A blackbody's colour, on the Planckian locus, at the given luminance. Its chroma is
/// exaggerated in linear light, away from the grey of the same luminance, and any channel
/// pushed below zero clipped (numga does the same in display values).
pub fn blackbody_colour(temperature: f64, luminance: f32) -> Light {
    let pale = Light::from_temperature(temperature as f32, 1.0);
    let grey = pale.luma();
    let [r, g, b, _] = pale.to_premultiplied();
    let vivid = |channel: f32| (grey + SATURATION * (channel - grey)).max(0.0);
    let colour = light(vivid(r), vivid(g), vivid(b), 1.0);
    colour * (luminance / colour.luma())
}

/// The light of a blackbody at a temperature in K as the camera shows it: its colour, at its
/// visible luminance, the exposure temperature at mid-grey.
pub fn blackbody(temperature: f64) -> Light {
    static REFERENCE: OnceLock<f64> = OnceLock::new();
    let reference = *REFERENCE.get_or_init(|| visible_luminance(EXPOSURE_TEMPERATURE));
    let luminance = (visible_luminance(temperature) / reference) as f32 * MID_GREY;
    blackbody_colour(temperature, luminance)
}

/// One clump wave: its whole turns around the disk, its wavenumber across it and its phase.
struct ClumpMode {
    turns: usize,
    wavenumber: f64,
    phase: f64,
}

/// The clump waves, drawn once.
fn clump_modes() -> &'static [ClumpMode] {
    static MODES: OnceLock<Vec<ClumpMode>> = OnceLock::new();
    MODES.get_or_init(|| {
        let mut draw = rng(CLUMP_SEED);
        (0..CLUMP_MODES)
            .map(|_| ClumpMode {
                turns: draw.range(MIN_TURNS as f64, (MAX_TURNS + 1) as f64) as usize,
                wavenumber: draw.range(0.8, 4.0),
                phase: draw.range(0.0, core::f64::consts::TAU),
            })
            .collect()
    })
}

/// Random waves around and across the disk at a point, at oblate radius `radius`, summed to
/// unit variance. Each is the height of a turn: the point's direction from the `x` axis, a
/// rotor, raised to the wave's whole turns (which keep it continuous around), times its phase
/// across.
#[allow(clippy::disallowed_methods)] // a statistic: unit variance from waves of variance ½
fn clump_waves(point: V, radius: f64) -> f64 {
    let east = vga2d::Vector::new(1.0, 0.0);
    let direction = vga2d::Vector::new(point.e1(), point.e2()).normalized();
    let around = east * direction;
    let powers: Vec<vga2d::Rotor> =
        std::iter::successors(Some(vga2d::Rotor::new(1.0, 0.0)), |p| Some(*p * around))
            .take(MAX_TURNS + 1)
            .collect();
    let sum: f64 = clump_modes()
        .iter()
        .map(|mode| {
            let across = vga2d::Pseudoscalar::new(mode.wavenumber * radius + mode.phase).exp();
            (powers[mode.turns] * across.into_inner()).e12()
        })
        .sum();
    sum / (CLUMP_MODES as f64 / 2.0).sqrt()
}

/// An illustrative texture on the disk's emission: fine rings, which the orbital flow leaves
/// in place, and clumps from `waves`.
pub fn emissivity(radius: f64, waves: f64) -> f64 {
    let rings = 0.90 + 0.045 * wave(11.0 * radius) + 0.025 * wave(19.0 * radius);
    rings * (CLUMP_CONTRAST * waves).exp()
}

/// The clump waves at a glint, `progress` of the way through the animation's loop: read where
/// its gas was now and one loop before, and turned from the first to the second over the loop,
/// which closes it without a jump and without the loss of contrast an average would bring.
pub fn flowing_waves(glint: &Glint, progress: f64) -> f64 {
    let hole = camera_hole();
    let at = |elapsed: f64| {
        let gas = carried(glint.event, glint.radius, xy(), hole, elapsed);
        clump_waves(gas, glint.radius)
    };
    let now = at(FLOW_TIME * progress);
    let before = at(FLOW_TIME * (progress - 1.0));
    let fade = phasor(core::f64::consts::FRAC_PI_2 * progress);
    fade.e20() * now + fade.e01() * before
}

/// A direction of space.
type Direction = vga3d::Vector;

/// A star: its direction and its light.
struct Star {
    direction: Direction,
    light: Light,
}

/// A seamless sky: a catalogue of unresolved stars, filed by the cell of a cube's faces their
/// direction points through, over a faint dusty band.
pub struct Sky {
    cells: Vec<Vec<Star>>,
}

/// The cube face a direction points through and its cell there, `(face, column, row)`: the
/// face of its largest component, the other two over that one.
fn cube_cell(d: Direction) -> (usize, usize, usize) {
    let c = [d.e1(), d.e2(), d.e3()];
    let axis = (0..3)
        .max_by(|&i, &j| c[i].abs().total_cmp(&c[j].abs()))
        .expect("three axes");
    let face = 2 * axis + usize::from(c[axis] < 0.0);
    let on = |k: usize| {
        let along = c[(axis + k) % 3] / c[axis].abs();
        (((along + 1.0) / 2.0 * SKY_GRID as f64) as usize).min(SKY_GRID - 1)
    };
    (face, on(1), on(2))
}

/// The index of a cell in the catalogue.
fn cell_index(face: usize, column: usize, row: usize) -> usize {
    (face * SKY_GRID + row) * SKY_GRID + column
}

impl Sky {
    /// The catalogue: stars uniform on the sphere, a few bright, from blue-white to orange.
    pub fn new() -> Sky {
        let mut draw = rng(SKY_SEED);
        let mut cells: Vec<Vec<Star>> = (0..6 * SKY_GRID * SKY_GRID).map(|_| Vec::new()).collect();
        let (cool, warm) = (light(0.57, 0.72, 1.0, 1.0), light(1.0, 0.79, 0.51, 1.0));
        for _ in 0..STAR_COUNT {
            let direction: Direction = draw.direction();
            let brightness = 0.35 + 2.5 * draw.uniform().powi(6);
            let colour = cool.mix_light(warm, draw.uniform() as f32);
            let (face, column, row) = cube_cell(direction);
            cells[cell_index(face, column, row)].push(Star {
                direction,
                light: colour * brightness as f32,
            });
        }
        Sky { cells }
    }

    /// The faint band of the galaxy along the great circle about a pole, mottled by a few
    /// waves and darkened by a lane of dust along its middle (numga's widths, waves and
    /// colours).
    fn band(direction: Direction) -> Light {
        let pole = -measure::lift(-0.2, 0.6).expect("inside the unit disc");
        let height = (direction | pole).s();
        let along = |k: [f64; 3]| (direction | vga3d::Vector::new(k[0], k[1], k[2])).s();
        let cloud = wave(along([13.0, 9.0, -7.0]))
            + 0.45 * wave(along([31.0, -23.0, 19.0]))
            + 0.18 * wave(along([79.0, 47.0, 53.0]));
        let band = (-(height / 0.16).powi(2)).exp() * (0.75 + 0.2 * cloud);
        let dust = (-((height + 0.025 * cloud) / 0.035).powi(2)).exp();
        light(0.006, 0.009, 0.019, 1.0)
            + light(0.065, 0.058, 0.09, 1.0) * (band * (1.0 - 0.7 * dust)) as f32
    }

    /// The sky's light in a direction: the band, and each star near it as a small Gaussian
    /// image `width` across (radians), found in the direction's cell and its neighbours.
    pub fn light(&self, direction: Direction, width: f64) -> Light {
        let (face, column, row) = cube_cell(direction);
        let near = |i: usize, di: usize| (i + di).checked_sub(1).filter(|&j| j < SKY_GRID);
        let mut sum = Sky::band(direction);
        for dr in 0..3 {
            for dc in 0..3 {
                let (Some(r), Some(c)) = (near(row, dr), near(column, dc)) else {
                    continue;
                };
                for star in &self.cells[cell_index(face, c, r)] {
                    let apart = (direction - star.direction).norm_squared();
                    sum += star.light * (-apart / (2.0 * width * width)).exp() as f32;
                }
            }
        }
        sum * SKY_BRIGHTNESS
    }
}
