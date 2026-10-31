//! numga's scene: a solid torus of 16 slices of 8 around the tube, its Beltrami modes, and a
//! mix of five relaxing towards the Taylor state.

use super::beltrami::*;
use gax::field::{Field, SparseMap};

pub const MAJOR_RADIUS: f64 = 1.0;
pub const MINOR_RADIUS: f64 = 0.42;
pub const N_MAJOR: usize = 16;
pub const N_MINOR: usize = 8;
/// The modes drawn (numga's negative indices: from the last), the modes mixed and their
/// weights, the frames and the relaxation's span.
pub const SHOWN: [isize; 2] = [-1, -4];
pub const SELECTED: [isize; 5] = [-1, -2, -4, -8, 3];
pub const INITIAL_WEIGHTS: [f64; 5] = [0.45, 0.70, 0.85, 0.85, 0.45];
pub const FRAMES: usize = 46;
pub const RELAXATION_TIME: f64 = 0.025;

/// numga's index into a list, negative from the end.
pub fn at(index: isize, len: usize) -> usize {
    if index < 0 {
        (len as isize + index) as usize
    } else {
        index as usize
    }
}

/// The torus cavity: its operators, the flux-preserving subspace, the inverse eigenvalues, the
/// modes on the edges, and the gradient directions' count.
pub struct Cavity {
    pub operators: Operators,
    #[cfg_attr(not(test), allow(dead_code))] // the tests check the subspace and the count
    pub free: SparseMap<ScalarMap>,
    pub mu: Vec<f64>,
    pub modes: Vec<Field<Scalar>>,
    #[cfg_attr(not(test), allow(dead_code))]
    pub null: usize,
}

impl Cavity {
    pub fn new() -> Self {
        let (mesh, vertices) = solid_torus(MAJOR_RADIUS, MINOR_RADIUS, N_MAJOR, N_MINOR);
        let free = flux_preserving(&mesh, N_MAJOR, N_MINOR);
        let operators = Operators::new(&mesh, &vertices);
        let (mu, free_modes, null) = coexact_helicity_eigh(
            &dense(&free, &operators.stiffness),
            &dense(&free, &operators.helicity),
        );
        let modes = free_modes.iter().map(|m| free.of(m)).collect();
        Cavity {
            operators,
            free,
            mu,
            modes,
            null,
        }
    }

    /// The selected modes, their eigenvalues `1 / μ`, and the amplitudes of each frame.
    pub fn relaxation(&self) -> (Vec<Field<Scalar>>, Vec<f64>, Vec<Vec<f64>>) {
        let picked: Vec<usize> = SELECTED.iter().map(|&k| at(k, self.mu.len())).collect();
        let modes = picked.iter().map(|&k| self.modes[k].clone()).collect();
        let eigenvalues: Vec<f64> = picked.iter().map(|&k| 1.0 / self.mu[k]).collect();
        let times: Vec<f64> = (0..FRAMES)
            .map(|k| RELAXATION_TIME * k as f64 / (FRAMES - 1) as f64)
            .collect();
        let states = selective_decay(&INITIAL_WEIGHTS, &eigenvalues, &times);
        (modes, eigenvalues, states)
    }
}
