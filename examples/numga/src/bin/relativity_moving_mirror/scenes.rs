//! numga's scenes: a scalene spacetime triangle and its lattice, the wake of a source in a strip
//! at rest, and the cavity whose right mirror advances.

use super::spacetime::*;
use gax::field::Field;
use gax::mesh::Topology;

pub const TRIANGLE_SUBDIVISIONS: usize = 7;
/// The wake's strip, its duration, the source event and its width.
pub const WAKE_COLUMNS: usize = 81;
pub const WAKE_ROWS: usize = 140;
pub const WAKE_DURATION: f64 = 0.78;
pub const SOURCE: [f64; 2] = [0.5, 0.12];
pub const SOURCE_WIDTH: f64 = 0.032;
/// The cavity's strip, its duration, the mirror's stroke and its turn, the pulse, and the rows
/// before and after the reflection.
pub const MIRROR_COLUMNS: usize = 85;
pub const MIRROR_ROWS: usize = 380;
pub const MIRROR_DURATION: f64 = 1.25;
pub const INITIAL_LENGTH: f64 = 1.0;
pub const STROKE: f64 = 0.36;
pub const INFLECTION_TIME: f64 = 0.54;
pub const TRANSITION_TIME: f64 = 0.42;
pub const PULSE_CENTRE: f64 = 0.28;
pub const PULSE_WIDTH: f64 = 0.085;
pub const WAVELENGTH: f64 = 0.17;
pub const BEFORE_ROW: usize = 72;
pub const AFTER_ROW: usize = 310;

/// The barycentric lattice of `n` steps across a triangle, its boundary included.
pub fn triangle_samples(corners: [Point; 3], n: usize) -> Vec<Point> {
    let mut out = Vec::new();
    for first in 0..=n {
        for second in 0..=n - first {
            let w = [n - first - second, first, second].map(|k| k as f64 / n as f64);
            out.push(corners[0] * w[0] + corners[1] * w[1] + corners[2] * w[2]);
        }
    }
    out
}

/// A scalene spacetime triangle and its lattice.
pub fn single_triangle(n: usize) -> ([Point; 3], Vec<Point>) {
    let corners = [event(0.0, 0.0), event(1.15, 0.16), event(0.30, 0.92)];
    (corners, triangle_samples(corners, n))
}

/// The right mirror's place at time `t`.
pub fn mirror_wall(t: f64) -> f64 {
    mirror_trajectory(t, INITIAL_LENGTH, STROKE, INFLECTION_TIME, TRANSITION_TIME)
}

/// The wake's strip and its wake.
pub struct Wake {
    pub mesh: Topology<3>,
    pub vertices: Field<Point>,
    pub wake: Field<Scalar>,
    pub source: Point,
}

impl Wake {
    pub fn new() -> Self {
        let (mesh, vertices, _) =
            spacetime_strip(WAKE_COLUMNS, WAKE_ROWS, &[1.0; WAKE_ROWS], WAKE_DURATION);
        let operators = Operators::new(&mesh, &vertices);
        let source = event(SOURCE[0], SOURCE[1]);
        let wake = causal_wake(&operators, WAKE_COLUMNS, WAKE_ROWS, source, SOURCE_WIDTH);
        Wake {
            mesh,
            vertices,
            wake,
            source,
        }
    }
}

/// The cavity: its strip, and the pulse in it.
pub struct Cavity {
    pub mesh: Topology<3>,
    pub vertices: Field<Point>,
    pub wave: Wave,
}

impl Cavity {
    pub fn new() -> Self {
        let walls: Vec<f64> = (0..MIRROR_ROWS)
            .map(|r| mirror_wall(r as f64 * MIRROR_DURATION / (MIRROR_ROWS - 1) as f64))
            .collect();
        let (mesh, vertices, dt) =
            spacetime_strip(MIRROR_COLUMNS, MIRROR_ROWS, &walls, MIRROR_DURATION);
        let operators = Operators::new(&mesh, &vertices);
        let wave = moving_mirror_wave(
            &operators,
            MIRROR_COLUMNS,
            MIRROR_ROWS,
            dt,
            PULSE_CENTRE,
            PULSE_WIDTH,
            WAVELENGTH,
        );
        Cavity {
            mesh,
            vertices,
            wave,
        }
    }
}
