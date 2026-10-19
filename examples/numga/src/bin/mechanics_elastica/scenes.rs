//! numga's `elastica/scenarios.py`: a strip curled by a couple at its end, Euler's elastica
//! pinned at both ends under end thrust, Kirchhoff's helices coiled by an oblique couple, a
//! wire under a fixed couple with its tilt swept, and a telephone cord pulled out and let go.

use crate::rod::{self, F, Frame, L, P, Segment, Stiffness, X, Y, Z};
use gax::pga3d::{Motor, Point};
use gax_numga_examples::signal::{haversine, phasor, smoothstep};
use std::f64::consts::{FRAC_PI_2, TAU};

/// The rod's length, the pieces it is cut into, and its compliances about and along the
/// cross-section's axes, across, up and along: to turning, bending across and up and
/// twisting, and none to shifting, a rod that only bends and twists.
pub const LENGTH: f64 = 1.0;
pub const PIECES: usize = 200;
pub const TURNING: [f64; 3] = [1.0, 1.0, 1.4];
pub const SHIFTING: [f64; 3] = [0.0; 3];
/// The curled strip: its couple in whole turns of its end.
pub const CURLS: [f64; 6] = [0.0, 0.25, 0.5, 0.75, 1.0, 2.0];
/// Euler's elastica: the angles in degrees at which the rod leaves its first pin, for the
/// plate, and the range of the sweep.
pub const END_ANGLES: [f64; 8] = [20.0, 60.0, 90.0, 110.0, 130.7, 150.0, 165.0, 175.0];
pub const SWEEP: [f64; 2] = [2.0, 176.0];
/// Kirchhoff's helices: the tilts of the rod's start from the couple, upward, in degrees, the
/// whole turns the couple coils the rod through, and the width of the ribbon drawn along it.
pub const TILTS: [f64; 4] = [80.0, 60.0, 40.0, 20.0];
pub const COILS: f64 = 3.0;
pub const RIBBON: f64 = 0.015;
/// The uncoiling wire: the whole turns its couple coils it through, and the width of its
/// ribbon.
pub const UNCOIL_TURNS: f64 = 1.5;
pub const UNCOIL_RIBBON: f64 = 0.035;
/// The telephone cord: its length in metres, its coils, the segments per coil, and the angle
/// in degrees of its wire to the plane of a coil; its rigidities against turning about the
/// cross-section's axes, bending across and up and twisting, in newton square metres, and
/// against shifting along them, shearing across and stretching along, in newtons; its mass
/// per metre in kilograms, the radius its segments' mass is spread over as a share of a
/// segment's length, gravity in metres per second squared, and its damping per second.
pub const CORD_LENGTH: f64 = 1.0;
pub const CORD_COILS: usize = 5;
pub const PER_COIL: usize = 6;
pub const CORD_PITCH: f64 = 10.0;
pub const CORD_TURNING: [f64; 3] = [1e-3, 1e-3, 7e-4];
pub const CORD_SHIFTING: [f64; 3] = [20.0, 20.0, 20.0];
pub const CORD_MASS: f64 = 0.05;
pub const CORD_SPREAD: f64 = 2.0;
pub const GRAVITY: f64 = 9.81;
pub const CORD_DAMPING: f64 = 1.0;
/// The pull: the time the cord hangs first, the time the handset takes out and the distance,
/// the time it is held, and the time it takes back, in seconds and metres; the time step, the
/// steps per frame, the frames, the drawn frames between neighbouring segments, and the width
/// of the drawn ribbon.
pub const HANG: f64 = 0.3;
pub const OUT: f64 = 0.9;
pub const REACH: f64 = 0.6;
pub const HOLD: f64 = 0.3;
pub const BACK: f64 = 0.4;
pub const TIME_STEP: f64 = 1e-3;
pub const STEPS_PER_FRAME: usize = 40;
pub const CORD_FRAMES: usize = 70;
pub const DRAWN_BETWEEN: usize = 4;
pub const CORD_RIBBON: f64 = 0.025;
/// The cord's segments.
pub const CORD_PIECES: usize = CORD_COILS * PER_COIL;

/// The turn by an angle in degrees about the axis across the rod's start.
pub fn turned(degrees: f64) -> Frame {
    ((Y ^ Z) * (-degrees.to_radians() / 2.0)).exp()
}

/// The rod of [`TURNING`] under a load from a start frame.
fn bent(load: F, start: Frame) -> Vec<Frame> {
    let flex = rod::compliance(TURNING, SHIFTING);
    rod::rod(load, flex, start, LENGTH, PIECES)
}

/// The strip's frames under a couple about the axis across it that turns its end by `turns`
/// whole turns: the couple whose rigidity over it is the strip's length over the turns, the
/// dual of the turn about that axis.
pub fn curling(turns: f64) -> Vec<Frame> {
    let couple = (Y ^ Z).dual() * (TAU / (TURNING[0] * LENGTH) * turns);
    bent(couple, Frame::identity())
}

/// The frames of a rod leaving its first pin at an angle in degrees to the line of thrust
/// through both pins, under the thrust that brings it back to that line at its full length.
pub fn euler(degrees: f64) -> Vec<Frame> {
    let line = rod::centre() & Point::direction(0.0, 0.0, 1.0);
    bent(line * thrust(degrees.to_radians()), turned(degrees))
}

/// The frames of a rod started at a tilt in degrees from an upward couple that coils it through
/// `turns` whole turns: Kirchhoff's helices, and the uncoiling wire.
pub fn coiled(tilt: f64, turns: f64) -> Vec<Frame> {
    bent(
        (X ^ Y).dual() * (TAU * turns / (TURNING[0] * LENGTH)),
        turned(tilt),
    )
}

/// The tilt of the uncoiling wire from its couple at a phase of its sweep (a turn for the
/// whole): from square to the couple, along it at half a turn, and back.
pub fn uncoiling_tilt(phase: f64) -> f64 {
    90.0 * (1.0 - haversine(phase))
}

/// The complete elliptic integrals of the first and second kind at the modulus `sin(half)`,
/// by the arithmetic-geometric mean of one and `cos(half)`: the first is a quarter turn over the
/// mean, the second the first times one less the deficit, the running sum of the squared half
/// differences, each weighted by twice the last.
#[allow(clippy::disallowed_methods)] // the geometric mean, a scalar step of the mean
fn elliptic(half: f64) -> (f64, f64) {
    let modulus = phasor(half);
    let (mut mean, mut geometric) = (1.0, modulus.e20());
    let (mut weight, mut deficit) = (0.5, modulus.e01().powi(2) / 2.0);
    for _ in 0..8 {
        let difference = (mean - geometric) / 2.0;
        (mean, geometric) = ((mean + geometric) / 2.0, (mean * geometric).sqrt());
        deficit += 2.0 * weight * difference.powi(2);
        weight *= 2.0;
    }
    let first = FRAC_PI_2 / mean;
    (first, first * (1.0 - deficit))
}

/// The end thrust under which a pinned rod leaving its pin at an angle (radians) reaches the
/// line of thrust again at its full length: twice the complete elliptic integral of the first
/// kind of the sine of half the angle, over the length, squared, over the compliance to
/// bending.
pub fn thrust(angle: f64) -> f64 {
    let (first, _) = elliptic(angle / 2.0);
    (2.0 * first / LENGTH).powi(2) / TURNING[0]
}

/// The distance between the pins of Euler's elastica at an angle (radians): the length times
/// twice the ratio of the complete elliptic integrals of the second and first kind, less one.
pub fn chord(angle: f64) -> f64 {
    let (first, second) = elliptic(angle / 2.0);
    LENGTH * (2.0 * second / first - 1.0)
}

/// The edges of a ribbon of the given width across a cross-section.
pub fn ribbon(frame: Frame, width: f64) -> [P; 2] {
    [-width / 2.0, width / 2.0].map(|x| frame >> Point::xyz(x, 0.0, 0.0))
}

/// The cord at rest: its motion per unit length, a shift along it with a bend and a twist that
/// coil it, and the rest frames of its segments with one held frame beyond each end
/// `[pieces + 2]`.
pub fn coil(pieces: usize) -> (L, Vec<Frame>) {
    let pitch = phasor(CORD_PITCH.to_radians());
    let turning = TAU * CORD_COILS as f64 / CORD_LENGTH;
    let rest = rod::shift_along_rod()
        + (Y ^ Z) * (turning * pitch.e20())
        + (X ^ Y) * (turning * pitch.e01());
    let step = CORD_LENGTH / pieces as f64;
    let frames = (-1..=pieces as i64)
        .map(|k| (rest * (k as f64 * step / 2.0)).exp())
        .collect();
    (rest, frames)
}

/// How far the handset is out at a time: eased out after the hang, held, and eased back.
pub fn reach(time: f64) -> f64 {
    let out = smoothstep((time - HANG) / OUT);
    let back = smoothstep((time - HANG - OUT - HOLD) / BACK);
    REACH * out * (1.0 - back)
}

/// The cord's joint at rest, its stiffness, the length of a segment, and a segment.
pub struct Cord {
    pub joint: Frame,
    pub stiff: Stiffness,
    pub step: f64,
    pub segment: Segment,
}

/// The telephone cord of [`CORD_PIECES`] segments.
pub fn cord() -> Cord {
    let step = CORD_LENGTH / CORD_PIECES as f64;
    let (rest, _) = coil(CORD_PIECES);
    let mass = CORD_MASS * step;
    let inertia = rod::segment_inertia(mass, step, CORD_SPREAD * step);
    Cord {
        joint: (rest * (step / 2.0)).exp(),
        stiff: rod::stiffness(CORD_TURNING, CORD_SHIFTING),
        step,
        segment: Segment {
            inertia,
            mobility: inertia.inverse(),
            weight: Point::direction(0.0, -GRAVITY * mass, 0.0),
            damping: CORD_DAMPING,
        },
    }
}

/// The cord's frames at every frame of the scene `[frames][pieces + 2]`: hanging from its base
/// under its weight, its handset pulled out along the coil's axis, held, and let back, the
/// cord following by its dynamics. The coil's axis is the direction of its turn; the handset
/// moves along it, keeping its turn.
pub fn pulled() -> Vec<Vec<Frame>> {
    let cord = cord();
    let (_, mut frames) = coil(CORD_PIECES);
    let mut rates = vec![L::zero(); CORD_PIECES];
    let handset = frames[CORD_PIECES + 1];
    let axis = phasor(CORD_PITCH.to_radians());
    let mut scene = Vec::with_capacity(CORD_FRAMES);
    for index in 0..CORD_FRAMES {
        scene.push(frames.clone());
        for tick in 0..STEPS_PER_FRAME {
            let out = reach((index * STEPS_PER_FRAME + tick) as f64 * TIME_STEP);
            frames[CORD_PIECES + 1] =
                Motor::translation(Point::direction(out * axis.e20(), 0.0, out * axis.e01()))
                    * handset;
            rod::moving(
                &mut frames,
                &mut rates,
                cord.joint,
                cord.stiff,
                cord.step,
                &cord.segment,
                TIME_STEP,
            );
        }
    }
    scene
}
