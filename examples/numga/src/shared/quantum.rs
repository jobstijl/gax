//! What several quantum examples share: values spread evenly over a range, the unit circle of a
//! dial, and a state in time as the cosine and the sine parts of a real Hamiltonian's turning.

// Each binary that includes this module uses a different part of it.
#![allow(dead_code)]

use core::iter::Sum;
use core::ops::Mul;
use gax_numga_examples::signal::phasor;
use gax_numga_examples::{ORIGIN2, Point2};

/// `count` values evenly from `from` to `to`, both ends included.
pub fn linspace(from: f64, to: f64, count: usize) -> Vec<f64> {
    (0..count)
        .map(|k| from + (to - from) * k as f64 / (count - 1) as f64)
        .collect()
}

/// The unit circle about the origin as `segments + 1` points, the first repeated at the end.
pub fn unit_circle(segments: usize) -> Vec<Point2> {
    (0..=segments)
        .map(|k| ORIGIN2 + phasor(core::f32::consts::TAU * k as f32 / segments as f32))
        .collect()
}

/// A state in time under a real Hamiltonian: the cosine and the sine of the Hamiltonian times
/// time applied to the start, two real states. Together they are the complex state, its real
/// and imaginary parts.
#[derive(Clone, Copy, Debug)]
pub struct InTime<S> {
    pub cosine: S,
    pub sine: S,
}

impl<S: Copy + Mul<f64, Output = S> + Sum> InTime<S> {
    /// The start after `time`, from its parts along the Hamiltonian's states (`shares`) and
    /// their energies: each part turns by its own energy.
    pub fn turned(energies: &[f64], shares: &[S], time: f64) -> InTime<S> {
        let turns: Vec<_> = energies.iter().map(|e| phasor(e * time)).collect();
        let part = |read: fn(&gax::pga2d::Point<(), f64>) -> f64| {
            shares.iter().zip(&turns).map(|(s, p)| *s * read(p)).sum()
        };
        InTime {
            cosine: part(|p| p.e20()),
            sine: part(|p| p.e01()),
        }
    }
}
