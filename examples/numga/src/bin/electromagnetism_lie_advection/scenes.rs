//! numga's scenes: a scalene triangle whose corners move, carrying a linear 1-form, and a disk of
//! nearly equilateral triangles stirred by an elliptical vortex, with two lobes of flux in it.

use super::advection::*;
use crate::simplicial::hex_disk;
use crate::sta2d::{Point, point};
use gax::field::Field;
use gax::mesh::Topology;

/// How far each corner moves in the drawn step.
pub const RIBBON_STEP: f64 = 0.55;
pub const CORNER_FORMS: [[f64; 3]; 3] = [[0.45, -0.65, 0.0], [1.35, 0.25, 0.0], [-0.55, 1.15, 0.0]];
pub const CORNER_VELOCITIES: [[f64; 2]; 3] = [[0.48, 0.26], [0.62, 0.34], [0.38, 0.44]];

pub const DISK_DIVISIONS: usize = 16;
pub const DISK_RADIUS: f64 = 1.0;
pub const SWIRL_RATE: f64 = 5.2;
pub const ELLIPTICITY: f64 = 0.2;
pub const LOBE_CENTRES: [[f64; 2]; 2] = [[-0.36, -0.06], [0.36, 0.06]];
pub const LOBE_STRENGTHS: [f64; 2] = [1.0, -0.65];
pub const LOBE_RADIUS: f64 = 0.28;
/// The ideal and the resistive case.
pub const RESISTIVITIES: [f64; 2] = [0.0, 0.015];
pub const DT: f64 = 0.0027;
pub const STEPS: usize = 356;
pub const STEPS_PER_FRAME: usize = 4;

/// A scalene triangle's corners.
pub fn single_triangle() -> [Point; 3] {
    [point(0.0, 0.0), point(1.15, 0.1), point(0.34, 0.92)]
}

/// The triangle's 1-form at its corners.
pub fn corner_forms() -> [Plane; 3] {
    CORNER_FORMS.map(|[x, y, t]| Plane::new(x, y, t))
}

/// The triangle's corners' velocities.
pub fn corner_velocities() -> [Point; 3] {
    CORNER_VELOCITIES.map(|[x, y]| point(x, y))
}

/// A disk of nearly equilateral triangles with its first vertex at the centre (numga's `disk`,
/// the shared hexagonal disk), its vertices as points of spacetime at time zero.
pub fn disk(radius: f64, divisions: usize) -> (Topology<3>, Field<Point>) {
    let (mesh, vectors) = hex_disk(radius, divisions);
    (mesh, vectors.map(|v| point(v.e1(), v.e2())))
}

/// The disk's flow: its operators, the streamfunction, the swept patches, the start, and the two
/// cases' evolution maps.
pub struct Disk {
    pub operators: Operators,
    pub stream: Field<Scalar>,
    #[cfg_attr(not(test), allow(dead_code))] // the tests check the sheets
    pub face_velocity: Field<Point>,
    #[cfg_attr(not(test), allow(dead_code))]
    pub swept: Field<Patch>,
    pub advection: gax::field::SparseMap<ScalarMap>,
    pub start: Field<Scalar>,
}

impl Disk {
    pub fn new(divisions: usize) -> Self {
        let (mesh, vertices) = disk(DISK_RADIUS, divisions);
        let operators = Operators::new(&mesh, &vertices);
        let stream = vortex_streamfunction(&vertices, SWIRL_RATE, ELLIPTICITY);
        let (face_velocity, fixed, comoving) = edge_world_sheets(&operators, &stream);
        let swept = &fixed - &comoving;
        let advection = advection(&operators, &swept);
        let centres = LOBE_CENTRES.map(|[x, y]| point(x, y));
        let start = initial_cochain(&operators, &centres, &LOBE_STRENGTHS, LOBE_RADIUS);
        Disk {
            operators,
            stream,
            face_velocity,
            swept,
            advection,
            start,
        }
    }

    /// The evolution map of a case.
    pub fn rate(&self, resistivity: f64) -> gax::field::SparseMap<ScalarMap> {
        rate(&self.operators, &self.advection, resistivity)
    }
}
