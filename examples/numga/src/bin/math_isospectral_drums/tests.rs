//! numga's tests: the transplantation is invertible and intertwines the wave operators and the
//! seams at every lattice size; independent eigensolves of the two drums give the same
//! frequencies, and the transplanted modes are the target's; and the scenario's check.

use super::drums::*;
use super::scenes::*;
use gax::field::{Field, SparseMap};
use std::collections::HashMap;

const ROUND_OFF: f64 = 1e-10;

/// The largest entry of `a − b`, their couplings summed site by site.
fn largest_difference(a: &SparseMap<ScalarMap>, b: &SparseMap<ScalarMap>) -> f64 {
    let mut sums = HashMap::new();
    for (r, c, cell) in a.couplings() {
        *sums.entry((r, c)).or_insert(0.0) += cell.of(Scalar::new(1.0)).s();
    }
    for (r, c, cell) in b.couplings() {
        *sums.entry((r, c)).or_insert(0.0) -= cell.of(Scalar::new(1.0)).s();
    }
    sums.values().map(|x: &f64| x.abs()).fold(0.0, f64::max)
}

#[test]
fn transplantation_is_invertible_and_intertwines_every_field() {
    for n in [4, 5, 6, 7, 8] {
        let source = Drum::new(&SOURCE, LATTICE, n);
        let target = Drum::new(&TARGET, LATTICE, n);
        let mixing = tile_mixing(&CONTRIBUTORS, &SIGNS, tile_samples(n));
        let transplant = target
            .assembly
            .compose(&mixing.compose(&source.restriction));
        let (source_laplacian, target_laplacian) = (source.laplacian(n), target.laplacian(n));

        // Every nodal impulse comes back, so no field lies in the transplant's kernel: one
        // factorization, solved for each.
        let sites = source.interior.len();
        let lu = transplant.factor().expect("the transplant is invertible");
        for i in 0..sites {
            let impulse = Field::from_fn(sites, |k| Scalar::new(f64::from(u8::from(k == i))));
            let restored = lu.solve(&transplant.of(&impulse));
            let worst = restored
                .iter()
                .zip(impulse.iter())
                .map(|(a, b)| (a - b).s().abs())
                .fold(0.0, f64::max);
            assert!(worst < ROUND_OFF, "n = {n}, impulse {i}: {worst}");
        }
        let commutator = largest_difference(
            &target_laplacian.compose(&transplant),
            &transplant.compose(&source_laplacian),
        );
        assert!(commutator < ROUND_OFF, "n = {n}: {commutator}");
        let seams = largest_difference(
            &target.restriction.compose(&transplant),
            &mixing.compose(&source.restriction),
        );
        assert!(seams < ROUND_OFF, "n = {n}: {seams}");
    }
}

#[test]
fn independent_eigensolves_and_transplanted_modes_have_the_same_frequencies() {
    let (n, modes) = (12, 10);
    let source = Drum::new(&SOURCE, LATTICE, n);
    let target = Drum::new(&TARGET, LATTICE, n);
    let transplant = target
        .assembly
        .compose(&tile_mixing(&CONTRIBUTORS, &SIGNS, tile_samples(n)).compose(&source.restriction));
    let (source_laplacian, target_laplacian) = (source.laplacian(n), target.laplacian(n));
    let identity = identity(source.interior.len());
    let (source_values, source_modes) = source_laplacian.eigh(&identity, modes);
    let (target_values, _) = target_laplacian.eigh(&identity, modes);
    for (a, b) in source_values.iter().zip(&target_values) {
        assert!((a - b).abs() < ROUND_OFF, "{a} {b}");
    }
    for (mode, value) in source_modes.iter().zip(&source_values) {
        let moved = transplant.of(mode);
        let residual = &target_laplacian.of(&moved) - &moved.map(|x| x * *value);
        assert!(residual.iter().all(|r| r.s().abs() < ROUND_OFF));
    }
}

#[test]
fn the_scenarios_transplant_carries_the_waves_across() {
    let pair = Pair::new(SUBDIVISIONS, MODES);
    for mode in &pair.modes {
        let a = pair.target_laplacian.of(&pair.transplant.of(mode));
        let b = pair.transplant.of(&pair.source_laplacian.of(mode));
        assert!(
            a.iter()
                .zip(b.iter())
                .all(|(x, y)| (x - y).s().abs() < 1e-6)
        );
    }
}

#[test]
fn frames_draw() {
    let shown = super::shown();
    for t in [0.0, 1.4] {
        gax_numga_examples::app::assert_draws_with(super::draw, &shown, t);
    }
}
