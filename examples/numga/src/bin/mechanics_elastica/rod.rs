//! numga's `elastica/core.py`: rods as chains of cross-sections, in PGA3D.
//!
//! A cross-section's placement is a motor, and a load is a line: a force along it, or a couple
//! (a line at infinity). A line also reads as a motion: the motor `exp(l / 2)` turns about the
//! line and shifts along it, as numga reads it (gax's twist is that line's half).

use gax::Unit;
use gax::motions::{Motions, Pga3d};
use gax::pga3d::{Line, Motor, Plane, Point};

/// A point of space.
pub type P = Point<(), f64>;
/// A line: a load, or a motion.
pub type L = Line<(), f64>;
/// The placement of a cross-section.
pub type Frame = Unit<Motor<(), f64>>;
/// A map on lines, `Line <- Line`: a compliance (a load to the motion per unit length it
/// causes), a stiffness (a joint's deviation per unit length to its load) or an inertia (a
/// rate to its momentum).
pub type LineMap = Line<(Line,), f64>;

/// A plane.
pub type Pl = Plane<(), f64>;

/// The coordinate planes `x = 0`, `y = 0` and `z = 0`, and the plane at infinity: numga's
/// `mv.x`, `mv.y`, `mv.z` and `mv.w`, whose meets are its lines `yz`, `wx` and the rest.
pub const X: Pl = Plane::new(1.0, 0.0, 0.0, 0.0);
pub const Y: Pl = Plane::new(0.0, 1.0, 0.0, 0.0);
pub const Z: Pl = Plane::new(0.0, 0.0, 1.0, 0.0);
pub const W: Pl = Plane::new(0.0, 0.0, 0.0, 1.0);

/// The centre of the cross-section at the rod's start.
pub fn centre() -> P {
    Point::xyz(0.0, 0.0, 0.0)
}

/// The motion carrying a cross-section along the unbent rod per unit length: a shift along
/// `z`, the line at infinity of the planes square to the rod.
pub fn shift_along_rod() -> L {
    Z ^ W
}

/// The axes of a cross-section through its centre: across, up and along the rod. A turn about
/// an axis is the axis itself and the couple about it the axis's dual; a shift along an axis is
/// the dual and the force along it the axis.
pub fn section_axes() -> [L; 3] {
    let along = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    along.map(|[x, y, z]| centre() & Point::direction(x, y, z))
}

/// How a cross-section moves per unit length under a load: about each of its axes, a turn by
/// the load's couple about the axis times the compliance to turning, and along it a shift by
/// the load's force along the axis times the compliance to shifting (zero for a rod that only
/// bends and twists).
pub fn compliance(turning: [f64; 3], shifting: [f64; 3]) -> LineMap {
    let slot = Line::slot();
    let axes = section_axes().into_iter().zip(turning).zip(shifting);
    axes.map(|((axis, turn), shift)| {
        let couple = axis.dual();
        axis * (axis & slot) * turn + couple * (couple & slot) * shift
    })
    .sum()
}

/// The frame of each cross-section of a rod of the given length `[pieces + 1]`, from the start
/// frame along, the rod carrying its load across every cross-section. The load seen from a
/// cross-section turns it, and with the shift moves it along the rod; the motion over each
/// piece is the one seen from halfway along it.
pub fn rod(load: L, flex: LineMap, start: Frame, length: f64, pieces: usize) -> Vec<Frame> {
    let step = length / pieces as f64;
    let motion = |frame: Frame| shift_along_rod() + flex.of(frame << load);
    let next = |&frame: &Frame| {
        let halfway = frame * (motion(frame) * (step / 4.0)).exp();
        Some(frame * (motion(halfway) * (step / 2.0)).exp())
    };
    std::iter::successors(Some(start), next)
        .take(pieces + 1)
        .collect()
}

/// The load a joint carries per unit length of its deviation from rest: about each of the
/// cross-section's axes, a couple for the turn about it times the rigidity against turning,
/// and along it a force for the shift along it times the rigidity against shifting.
pub fn stiffness(turning: [f64; 3], shifting: [f64; 3]) -> LineMap {
    let slot = Line::slot();
    let axes = section_axes().into_iter().zip(turning).zip(shifting);
    axes.map(|((axis, turn), shift)| {
        let couple = axis.dual();
        couple * (couple & slot) * turn + axis * (axis & slot) * shift
    })
    .sum()
}

/// A segment's inertia, from its rate to its momentum: its mass spread over its two ends and a
/// ring of the given radius around its middle, each point's momentum its join with its own
/// motion under the rate (`Motions::point_inertia`).
pub fn segment_inertia(mass: f64, length: f64, radius: f64) -> LineMap {
    let (r, h) = (radius, length / 2.0);
    let spread = [
        [r, 0.0, 0.0],
        [-r, 0.0, 0.0],
        [0.0, r, 0.0],
        [0.0, -r, 0.0],
        [0.0, 0.0, h],
        [0.0, 0.0, -h],
    ];
    let share = mass / spread.len() as f64;
    spread
        .into_iter()
        .map(|[x, y, z]| Pga3d::point_inertia(Point::xyz(x, y, z)) * share)
        .sum()
}

/// A joint's deviation from its rest placement: the line whose motion carries the rest
/// placement onto the relative placement of the frames it joins.
pub fn deviation(rest: Frame, before: Frame, after: Frame) -> L {
    (rest.reverse() * before.reverse() * after).log() * 2.0
}

/// The load each joint between neighbouring frames puts on the frame after it, in the world:
/// its stiffness applied to its deviation from rest, through the derivative of the
/// exponential (to its second order, `1 - ad / 2 + ad² / 12` with `ad = [d, ·]`), so that the
/// load is the derivative of the joint's energy.
pub fn joint_loads(frames: &[Frame], rest: Frame, stiff: LineMap, step: f64) -> Vec<L> {
    let load = |pair: &[Frame]| {
        let d = deviation(rest, pair[0], pair[1]);
        let load = stiff.of(d * step.recip());
        let load = load - d.commutator(load) * 0.5 + d.commutator(d.commutator(load)) / 12.0;
        pair[1] >> load
    };
    frames.windows(2).map(load).collect()
}

/// What a segment is: its inertia and the inverse, its weight (a direction of space, as long
/// as the force) and its damping per unit time.
#[derive(Clone, Copy, Debug)]
pub struct Segment {
    pub inertia: LineMap,
    pub mobility: LineMap,
    pub weight: P,
    pub damping: f64,
}

/// One time step of the segments between the first and last frame, which are held: each rate
/// changes by the forque on its segment (the loads of its two joints, its weight, a line
/// straight down through its centre, and its damping, against its momentum), less the
/// gyroscopic turn of its momentum by the rate; each frame then moves by its rate.
pub fn moving(
    frames: &mut [Frame],
    rates: &mut [L],
    rest: Frame,
    stiff: LineMap,
    step: f64,
    segment: &Segment,
    time_step: f64,
) {
    let loads = joint_loads(frames, rest, stiff, step);
    let inner = frames.len() - 1;
    for ((frame, rate), load) in frames[1..inner].iter_mut().zip(rates).zip(loads.windows(2)) {
        let momentum = segment.inertia.of(*rate);
        let weight = centre() & (*frame << segment.weight);
        let forque = (*frame << (load[0] - load[1])) + weight - momentum * segment.damping;
        *rate += segment
            .mobility
            .of(forque - Pga3d::coadjoint(momentum, *rate))
            * time_step;
        *frame = *frame * (*rate * (-time_step / 2.0)).exp();
    }
}

/// Frames between neighbouring frames, each pair joined by its relative motion in `count`
/// equal parts `[(frames - 1) count + 1]`.
pub fn between(frames: &[Frame], count: usize) -> Vec<Frame> {
    let parts = frames.windows(2).flat_map(|pair| {
        let relative = (pair[0].reverse() * pair[1]).log();
        (0..count).map(move |k| pair[0] * (relative * (k as f64 / count as f64)).exp())
    });
    parts.chain(frames.last().copied()).collect()
}
