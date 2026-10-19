//! What the Hubbard examples share: a state in time as the cosine and the sine parts of a real
//! Hamiltonian's turning.

use crate::signal::phasor;
use core::iter::Sum;
use core::ops::Mul;

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
        let part = |read: fn(&gax::pga2d::Point) -> f64| {
            shares.iter().zip(&turns).map(|(s, p)| *s * read(p)).sum()
        };
        InTime {
            cosine: part(|p| p.e20()),
            sine: part(|p| p.e01()),
        }
    }
}
