//! numga's scene: Gordon, Webb and Wolpert's pair of drums, each seven right triangles of a 4 × 4
//! lattice, the transplantation's recipe (each destination triangle three signed source
//! triangles), and the standing waves mixed into the vibration.

use super::drums::*;
use gax::field::{Field, SparseMap};

pub const LATTICE: usize = 4;
pub const SOURCE: [[usize; 3]; 7] = [
    [0, 1, 4],
    [5, 1, 4],
    [5, 9, 4],
    [5, 9, 6],
    [10, 9, 6],
    [10, 9, 14],
    [10, 11, 6],
];
pub const TARGET: [[usize; 3]; 7] = [
    [5, 1, 4],
    [5, 9, 4],
    [8, 9, 4],
    [5, 9, 6],
    [10, 9, 6],
    [10, 11, 6],
    [10, 11, 14],
];
pub const CONTRIBUTORS: [[usize; 3]; 7] = [
    [1, 0, 6],
    [0, 2, 4],
    [1, 2, 3],
    [3, 0, 5],
    [4, 1, 5],
    [5, 2, 6],
    [4, 3, 6],
];
pub const SIGNS: [[f64; 3]; 7] = [
    [1.0, -1.0, 1.0],
    [1.0, 1.0, 1.0],
    [1.0, -1.0, 1.0],
    [1.0, -1.0, 1.0],
    [1.0, -1.0, -1.0],
    [1.0, -1.0, 1.0],
    [1.0, -1.0, -1.0],
];
/// The samples along each tile's legs, the standing waves solved for, the three mixed into the
/// vibration and their amplitudes, the displacement at the ends of the colour scale, and the
/// destination triangle whose pieces are drawn.
pub const SUBDIVISIONS: usize = 24;
pub const MODES: usize = 9;
pub const MIXED_MODES: [usize; 3] = [0, 2, 5];
pub const MIX_AMPLITUDES: [f64; 3] = [0.30, 0.22, 0.26];
pub const EXPOSURE: f64 = 0.45;
pub const TARGET_TILE: usize = 0;

/// The samples of a tile with `n` steps along its legs.
pub fn tile_samples(n: usize) -> usize {
    (n + 1) * (n + 2) / 2
}

/// The two drums, their wave operators, the transplantation, the source drum's standing waves
/// (scaled by `n`, so that their sizes do not shrink with the lattice) and their squared
/// frequencies.
pub struct Pair {
    pub source: Drum,
    pub target: Drum,
    #[cfg_attr(not(test), allow(dead_code))] // the scenario's check reads them
    pub source_laplacian: SparseMap<ScalarMap>,
    #[cfg_attr(not(test), allow(dead_code))]
    pub target_laplacian: SparseMap<ScalarMap>,
    /// The signed mixing of tiles (numga's), which the transplant composes.
    #[allow(dead_code)]
    pub mixing: SparseMap<ScalarMap>,
    pub transplant: SparseMap<ScalarMap>,
    pub eigenvalues: Vec<f64>,
    pub modes: Vec<Field<Scalar>>,
}

impl Pair {
    pub fn new(n: usize, modes: usize) -> Self {
        let source = Drum::new(&SOURCE, LATTICE, n);
        let target = Drum::new(&TARGET, LATTICE, n);
        let (source_laplacian, target_laplacian) = (source.laplacian(n), target.laplacian(n));
        let mixing = tile_mixing(&CONTRIBUTORS, &SIGNS, tile_samples(n));
        let transplant = target
            .assembly
            .compose(&mixing.compose(&source.restriction));
        // Inserting the zero rim keeps the interior field's inner product: the metric is the
        // identity.
        let measure = source.lift.adjoint().compose(&source.lift);
        let (eigenvalues, fields) = source_laplacian.eigh(&measure, modes);
        let modes = fields.iter().map(|m| m.map(|x| x * n as f64)).collect();
        Pair {
            source,
            target,
            source_laplacian,
            target_laplacian,
            mixing,
            transplant,
            eigenvalues,
            modes,
        }
    }
}
