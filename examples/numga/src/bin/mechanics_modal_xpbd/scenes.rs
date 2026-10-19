//! Girders constrained at their corners: a spliced cantilever, a hinged chain under gravity,
//! and a spliced beam fixed at both ends and compressed until it buckles.

use crate::girder::{P, S, Shape};
use crate::xpbd::{Bodies, Constraints, Couple, M, Mobility, R, step};
use core::f64::consts::PI;
use gax::field::Field;
use gax::pga2d::{Motor, Point};

/// The chain's girder: its cells, length and height, its bars' stiffness, its density, and the
/// vibration modes it keeps.
pub const CELLS: usize = 4;
pub const LENGTH: f64 = 1.0;
pub const HEIGHT: f64 = 0.2;
pub const STIFFNESS: f64 = 300.0;
pub const DENSITY: f64 = 1.0;
pub const MODES: usize = 8;
/// The chain's moving girders, their modes' damping ratio, and the acceleration of gravity,
/// downward.
pub const LINKS: usize = 4;
pub const DAMPING: f64 = 0.02;
pub const GRAVITY: f64 = 4.0;
/// A constraint's compliance, rigid to round-off; it keeps the two constraints of a rigid
/// splice solvable.
pub const SPLICE: f64 = 1e-14;
/// The swing's time step, one per frame, and its frames.
pub const INTERVAL: f64 = 0.03;
pub const FRAMES: usize = 200;
/// The beam: its moving girders, their cells of unit length, their height, the bars'
/// stiffness, the modes' damping ratio, the final end displacement, its frames and the steps
/// between them.
pub const BEAM_GIRDERS: usize = 8;
pub const BEAM_CELLS: usize = 32;
pub const BEAM_HEIGHT: f64 = 1.0;
pub const BEAM_STIFFNESS: f64 = 1e11;
pub const BEAM_DAMPING: f64 = 3.0;
pub const END_DISPLACEMENT: f64 = 0.15;
pub const BEAM_FRAMES: usize = 120;
pub const BEAM_SUBSTEPS: usize = 2;
pub const BEAM_INTERVAL: f64 = 0.02;

/// Gravity, an ideal point.
pub fn gravity(acceleration: f64) -> P {
    Point::direction(0.0, -acceleration)
}

/// The chain's girder.
pub fn chain_girder() -> Shape {
    crate::girder::girder(CELLS, LENGTH, HEIGHT, STIFFNESS, DENSITY, MODES)
}

/// The beam's girder.
pub fn beam_girder() -> Shape {
    let length = BEAM_CELLS as f64;
    crate::girder::girder(
        BEAM_CELLS,
        length,
        BEAM_HEIGHT,
        BEAM_STIFFNESS,
        DENSITY,
        MODES,
    )
}

/// Girders end to end along x, some fixed: girder `i` centred `i - 1/2` girder lengths along,
/// at rest, a fixed one with no compliance in its modes and no inverse inertia.
pub fn girders(shape: &Shape, fixed: &[bool], damping: f64) -> Bodies {
    let rest = shape.rest.sites();
    let length = (rest[rest.len() - 1] - rest[1]).e20();
    let count = fixed.len();
    let still = || -> Vec<Field<S>> { shape.modes.iter().map(|_| Field::zeros(count)).collect() };
    let moving = |b: usize| if fixed[b] { 0.0 } else { 1.0 };
    let inverse = shape.inertia.inverse();
    Bodies {
        motor: (0..count)
            .map(|b| Motor::translation(Point::direction((b as f64 - 0.5) * length, 0.0)))
            .collect(),
        rate: vec![R::zero(); count],
        amplitudes: still(),
        rates: still(),
        compliance: shape
            .compliance
            .iter()
            .map(|c| Field::from_fn(count, |b| S::new(c * moving(b))))
            .collect(),
        frequencies: shape.frequencies.clone(),
        damping,
        mass: shape.mass(),
        inertia: shape.inertia,
        inverse_inertia: (0..count)
            .map(|b| inverse * moving(b))
            .collect::<Vec<Mobility>>(),
    }
}

/// Constraints between the given bodies at the given vertices, one pair of each per constraint.
fn constrained(shape: &Shape, bodies: Vec<[usize; 2]>, corners: &[[usize; 2]]) -> Constraints {
    let anchors = corners
        .iter()
        .map(|c| c.map(|v| shape.rest.site(v)))
        .collect();
    let modes = shape
        .modes
        .iter()
        .map(|mode| corners.iter().map(|c| c.map(|v| mode.site(v))).collect())
        .collect();
    Constraints {
        bodies,
        anchors,
        modes,
        compliance: SPLICE,
    }
}

/// Two constraints at each joint of a row of girders: the right end's two corners of one at
/// the left end's of the next.
pub fn splices(shape: &Shape, count: usize) -> Constraints {
    let last = shape.rest.len() - 1;
    let joints = (0..count - 1).flat_map(|j| [[j, j + 1], [j, j + 1]]);
    let corners: Vec<[usize; 2]> = (0..count - 1)
        .flat_map(|_| [[last - 1, 0], [last, 1]])
        .collect();
    constrained(shape, joints.collect(), &corners)
}

/// One constraint at each joint of a row of girders: the right end's upper corner of one at the
/// left end's of the next.
pub fn hinges(shape: &Shape, count: usize) -> Constraints {
    let last = shape.rest.len() - 1;
    let corners = vec![[last, 1]; count - 1];
    constrained(
        shape,
        (0..count - 1).map(|j| [j, j + 1]).collect(),
        &corners,
    )
}

/// Fixed at the first of `count` girders only.
fn first_fixed(count: usize) -> Vec<bool> {
    (0..count).map(|b| b == 0).collect()
}

/// A fixed girder followed by `count` spliced moving girders.
#[cfg_attr(not(test), allow(dead_code))] // numga's scenario for its test
pub fn cantilever(shape: &Shape, count: usize, damping: f64) -> (Bodies, Constraints) {
    let fixed = first_fixed(count + 1);
    (girders(shape, &fixed, damping), splices(shape, count + 1))
}

/// A fixed girder followed by `count` hinged moving girders.
pub fn hinged_chain(shape: &Shape, count: usize, damping: f64) -> (Bodies, Constraints) {
    let fixed = first_fixed(count + 1);
    (girders(shape, &fixed, damping), hinges(shape, count + 1))
}

/// `count` spliced moving girders between two fixed ones.
pub fn clamped_beam(shape: &Shape, count: usize, damping: f64) -> (Bodies, Constraints) {
    let fixed: Vec<bool> = (0..count + 2).map(|b| b == 0 || b == count + 1).collect();
    (girders(shape, &fixed, damping), splices(shape, count + 2))
}

/// The bodies' points at every frame, under gravity, one step per frame.
pub fn swing(
    shape: &Shape,
    bodies: &Bodies,
    constraints: &Constraints,
    gravity: P,
    dt: f64,
    frames: usize,
    couple: Couple,
) -> Vec<Vec<Field<P>>> {
    let mut bodies = bodies.clone();
    let mut points = Vec::with_capacity(frames);
    for _ in 0..frames {
        points.push(bodies.points(shape));
        bodies = step(&bodies, constraints, dt, gravity, couple);
    }
    points
}

/// The end's displacement at each of `frames` frames, from none up to [`END_DISPLACEMENT`].
pub fn displacements(frames: usize) -> Vec<f64> {
    (0..frames)
        .map(|k| END_DISPLACEMENT * k as f64 / frames as f64)
        .collect()
}

/// The bodies with the last one displaced inward along x from its rest pose.
pub fn displaced(bodies: &Bodies, rest: &[M], displacement: f64) -> Bodies {
    let last = rest.len() - 1;
    let mut moved = bodies.clone();
    moved.motor[last] = rest[last] * Motor::translation(Point::direction(-displacement, 0.0));
    moved
}

/// One frame of the beam: its girders' points, and its midspan, the middle of its two middle
/// girders' centres.
pub struct Frame {
    pub points: Vec<Field<P>>,
    pub midspan: P,
}

/// The beam at every end displacement, after so many steps without gravity.
pub fn compress(
    shape: &Shape,
    bodies: &Bodies,
    constraints: &Constraints,
    displacements: &[f64],
    dt: f64,
    substeps: usize,
    couple: Couple,
) -> Vec<Frame> {
    let rest = bodies.motor.clone();
    let middle = rest.len() / 2;
    let mut bodies = bodies.clone();
    let mut frames = Vec::with_capacity(displacements.len());
    for displacement in displacements {
        bodies = displaced(&bodies, &rest, *displacement);
        for _ in 0..substeps {
            bodies = step(&bodies, constraints, dt, gravity(0.0), couple);
        }
        let centres = [middle - 1, middle].map(|b| bodies.motor[b] >> Point::origin());
        frames.push(Frame {
            points: bodies.points(shape),
            midspan: (centres[0] + centres[1]) / 2.0,
        });
    }
    frames
}

/// The critical end displacement of a beam of two chords, clamped at both ends.
pub fn critical(height: f64, span: f64) -> f64 {
    PI * PI * height * height / span
}
