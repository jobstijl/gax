//! numga's scenes: vortex dipoles on the Stanford bunny's 128 lowest modes, and rippled shear
//! layers on a sphere's 80, each run from its start by Runge–Kutta in mode space.

use super::eigen::*;
use crate::simplicial::{closed, icosphere};
use gax::field::Field;
use gax::mesh::Topology;
use gax::vga3d;

/// The bunny's bounding box, its modes, and the modes shown on their own (from a global dipole
/// to a fine standing wave).
pub const BUNNY_SIZE: f64 = 2.0;
pub const MODES: usize = 128;
pub const SHOWN_MODES: [usize; 4] = [0, 7, 31, 119];
/// Opposing vortex dipoles across the bunny's shoulder, chest, flank, haunch, neck and back.
pub const VORTEX_CENTRES: [[f64; 3]; 12] = [
    [0.35, -0.54, -0.03],
    [0.40, -0.69, -0.29],
    [-0.08, -0.52, 0.03],
    [-0.16, -0.58, -0.27],
    [-0.54, -0.48, 0.02],
    [-0.45, -0.58, -0.39],
    [0.34, -0.31, 0.26],
    [0.09, -0.40, 0.20],
    [0.22, 0.34, -0.16],
    [-0.14, 0.28, -0.08],
    [-0.48, 0.26, -0.14],
    [0.06, 0.47, -0.30],
];
pub const VORTEX_STRENGTHS: [f64; 12] = [
    75.0, -75.0, -75.0, 75.0, 70.0, -70.0, -65.0, 65.0, 72.0, -72.0, 68.0, -68.0,
];
pub const VORTEX_RADIUS: f64 = 0.12;
/// The spectral taper's scale.
pub const EIGENVALUE_CUTOFF: f64 = 140.0;
/// The Runge–Kutta step, the steps, the viscosity of the viscous run, and the share of the run
/// the animation plays (its first half, every fourth step).
pub const DT: f64 = 0.007;
pub const STEPS: usize = 800;
pub const VISCOSITY: f64 = 0.0009;
pub const FLOW_STEPS: usize = 400;
pub const PLAYBACK_STRIDE: usize = 4;

/// The sphere: its subdivisions, modes (the complete bands through degree eight), the bands, and
/// its run.
pub const SPHERE_LEVELS: usize = 4;
pub const SPHERE_MODES: usize = 80;
pub const SPHERE_LATITUDE: f64 = 0.44;
pub const SPHERE_WIDTH: f64 = 0.16;
pub const SPHERE_STRENGTH: f64 = 9.0;
pub const SPHERE_WAVE_AMPLITUDE: f64 = 0.065;
pub const SPHERE_DT: f64 = 0.025;
pub const SPHERE_STEPS: usize = 400;
pub const SPHERE_STRIDE: usize = 4;
pub const SPHERE_VISCOSITY: f64 = 0.001;
/// The Lanczos blocks: the bunny has no symmetry; the icosphere repeats a value up to five
/// times (its symmetry's largest irreducible representation), twice that covers it.
pub const BUNNY_BLOCK: usize = 4;
pub const SPHERE_BLOCK: usize = 10;

/// The Stanford bunny (as decimated in libigl's tutorial data, built into the binary), its
/// triangles wound coherently and outward, centred in a box `BUNNY_SIZE` across and stood up
/// (its `y` turned onto `z`).
pub fn bunny() -> (Topology<3>, Field<Point>) {
    const BUNNY: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/models/bunny.obj"));
    let model = gax_numga_examples::mesh::Mesh::from_obj(BUNNY, BUNNY_SIZE);
    let wound = closed(&model.faces, &model.vertices);
    let upright = vga3d::Bivector::new(-core::f64::consts::FRAC_PI_4, 0.0, 0.0).exp();
    let vertices = model.vertices.map(|v| (upright >> v).hodge());
    let mesh = Topology::new(vertices.len(), &wound).expect("the bunny");
    (mesh, vertices)
}

/// Localized vortex dipoles, projected on the modes, tapered towards the finer ones.
pub fn initial_vorticity(
    surface: &Surface,
    modes: &[Field<Scalar>],
    eigenvalues: &Field<Scalar>,
) -> Field<Scalar> {
    let centres = VORTEX_CENTRES.map(|[x, y, z]| point(x, y, z));
    let spots = surface.vertices.map(|p| {
        let sum = centres
            .iter()
            .zip(VORTEX_STRENGTHS)
            .map(|(&c, strength)| {
                let d2 = (p - c).norm_squared();
                strength * (-0.5 * d2 / (VORTEX_RADIUS * VORTEX_RADIUS)).exp()
            })
            .sum();
        Scalar::new(sum)
    });
    let taper = eigenvalues.map(|l| {
        let x = l.s() / EIGENVALUE_CUTOFF;
        Scalar::new((-0.5 * x * x).exp())
    });
    surface
        .project(modes, &spots)
        .zip_with(&taper, |a, t| a * t)
}

/// A surface's flow in mode space: the surface, its modes and their eigenvalues, the bracket,
/// and the start.
pub struct Flow {
    pub surface: Surface,
    pub eigenvalues: Field<Scalar>,
    pub modes: Vec<Field<Scalar>>,
    pub bracket: Bracket,
    pub start: Field<Scalar>,
}

impl Flow {
    /// The bunny's flow.
    pub fn bunny() -> Flow {
        let (mesh, vertices) = bunny();
        let surface = Surface::new(mesh, vertices);
        let (values, modes) = surface.modes(MODES, BUNNY_BLOCK);
        let eigenvalues: Field<Scalar> = values.into_iter().map(Scalar::new).collect();
        let bracket = Bracket::new(&surface, &modes);
        let start = initial_vorticity(&surface, &modes, &eigenvalues);
        Flow {
            surface,
            eigenvalues,
            modes,
            bracket,
            start,
        }
    }

    /// The sphere's flow: rippled shear layers.
    pub fn sphere() -> Flow {
        let (mesh, vectors) = icosphere(SPHERE_LEVELS);
        let surface = Surface::new(mesh, vectors.map(|v| v.hodge()));
        let (values, modes) = surface.modes(SPHERE_MODES, SPHERE_BLOCK);
        let eigenvalues: Field<Scalar> = values.into_iter().map(Scalar::new).collect();
        let bracket = Bracket::new(&surface, &modes);
        let layers = shear_layers(
            &surface.vertices,
            &surface.dual_areas,
            SPHERE_LATITUDE,
            SPHERE_WIDTH,
            SPHERE_STRENGTH,
            SPHERE_WAVE_AMPLITUDE,
        );
        let start = surface.project(&modes, &layers);
        Flow {
            surface,
            eigenvalues,
            modes,
            bracket,
            start,
        }
    }

    /// The modal vorticity over `steps` steps of `dt`.
    pub fn run(&self, viscosity: f64, dt: f64, steps: usize) -> Vec<Field<Scalar>> {
        advect(
            &self.start,
            &self.bracket,
            &self.eigenvalues,
            viscosity,
            dt,
            steps,
        )
    }

    /// A modal state's vorticity and streamfunction on the vertices.
    pub fn fields(&self, amplitudes: &Field<Scalar>) -> (Field<Scalar>, Field<Scalar>) {
        let stream = amplitudes.zip_with(&self.eigenvalues, |w, l| w * l.inverse());
        (
            superpose(&self.modes, amplitudes),
            superpose(&self.modes, &stream),
        )
    }
}
