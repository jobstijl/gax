//! numga's tests: Whitney commutation, the flux-preserving subspace and the chiral Beltrami
//! spectrum of a solid torus; the selective decay at fixed helicity towards the Woltjer–Taylor
//! state; and the scenario's checks.

use super::beltrami::*;
use super::scenes::*;
use gax::field::{Field, SparseMap};
use gax_numga_examples::signal::wave;

fn small(values: impl IntoIterator<Item = f64>, tolerance: f64) -> bool {
    values.into_iter().all(|v| v.abs() < tolerance)
}

fn cavity() -> &'static Cavity {
    static CAVITY: std::sync::OnceLock<Cavity> = std::sync::OnceLock::new();
    CAVITY.get_or_init(Cavity::new)
}

/// An energy map applied, as numbers.
fn amounts(map: &SparseMap<EnergyMap>, x: &Field<Scalar>) -> Vec<f64> {
    map.of(x)
        .iter()
        .map(|a| (a & Scalar::new(1.0)).s())
        .collect()
}

/// The pairing of an energy map's image with a field: `Σ (map x) & x`.
fn pairing(map: &SparseMap<EnergyMap>, x: &Field<Scalar>) -> f64 {
    amounts(map, x)
        .iter()
        .zip(x.iter())
        .map(|(a, x)| a * x.s())
        .sum()
}

/// A map's largest asymmetry, `map − map.adjugate()`, cell by cell.
fn asymmetry(map: &SparseMap<EnergyMap>) -> f64 {
    let n = map.shape().0;
    let mut a = vec![vec![0.0; n]; n];
    for (r, c, cell) in map.couplings() {
        a[r][c] += (cell.of(Scalar::new(1.0)) & Scalar::new(1.0)).s();
    }
    let mut b = vec![vec![0.0; n]; n];
    for (r, c, cell) in map.adjugate().couplings() {
        b[r][c] += (cell.of(Scalar::new(1.0)) & Scalar::new(1.0)).s();
    }
    (0..n)
        .flat_map(|i| (0..n).map(move |j| (i, j)))
        .map(|(i, j)| (a[i][j] - b[i][j]).abs())
        .fold(0.0, f64::max)
}

#[test]
fn whitney_commutation_flux_preserving_subspace_and_chiral_beltrami_spectrum() {
    let c = cavity();
    let o = &c.operators;
    assert!(asymmetry(&o.stiffness) < 1e-12);
    assert!(asymmetry(&o.helicity) < 1e-14);

    // Whitney reconstruction commutes with the exterior derivative.
    let (d0, d1) = (o.mesh.d::<Scalar>(0), o.mesh.d::<Scalar>(1));
    let potential = o.vertices.map(|p| {
        let [x, y, z] = coordinates(p);
        Scalar::new(0.7 * x - 1.1 * y + 0.5 * z)
    });
    let a = o.whitney1.of(&(&d0 * &potential));
    let b = o.gradient.of(&potential);
    assert!(small(
        a.iter().zip(b.iter()).flat_map(|(x, y)| (x - y).c),
        1e-12
    ));
    let free_test = c.free.of(&Field::from_fn(c.free.shape().1, |k| {
        Scalar::new(wave(k as f64))
    }));
    let a = o.curl.of(&free_test);
    let b = o.whitney2.of(&(&d1 * &free_test));
    assert!(small(
        a.iter().zip(b.iter()).flat_map(|(x, y)| (x - y).c),
        1e-11
    ));
    // The normal flux vanishes on every wall triangle.
    let flux = &d1 * &free_test;
    assert!(small(
        o.mesh.exterior(2).iter().map(|&f| flux.site(f).s()),
        1e-14
    ));

    // An interior vertex's gradient lies in the kernel of the energy and of the free helicity.
    let interior_phi = Field::from_fn(o.vertices.len(), |v| {
        Scalar::new(f64::from(u8::from(v < N_MAJOR)))
    });
    let gauge = &d0 * &interior_phi;
    assert!(small(amounts(&o.stiffness, &gauge), 1e-12));
    let free_helicity = c.free.adjugate().of(&o.helicity.of(&gauge));
    assert!(small(
        free_helicity.iter().map(|a| (a & Scalar::new(1.0)).s()),
        1e-13
    ));

    // Chiral symmetry and the eigenmodes' residuals.
    assert_eq!(c.null, N_MAJOR);
    let n = c.mu.len();
    assert!(small((0..n).map(|k| c.mu[k] + c.mu[n - 1 - k]), 1e-13));
    for index in [0, 1, -2, -1] {
        let k = at(index, n);
        let mode = &c.modes[k];
        assert!((pairing(&o.stiffness, mode) / 2.0 - 0.5).abs() < 1e-12);
        assert!((pairing(&o.helicity, mode) - c.mu[k]).abs() < 1e-12);
        let lambda = 1.0 / c.mu[k];
        let residual = &o.stiffness.of(mode) - &o.helicity.of(mode).map(|a| a * lambda);
        let free_residual = c.free.adjugate().of(&residual);
        assert!(small(
            free_residual.iter().map(|a| (a & Scalar::new(1.0)).s()),
            1e-11
        ));
    }
}

#[test]
fn selective_decay_helicity_conservation_and_woltjer_taylor_relaxation() {
    let c = cavity();
    let o = &c.operators;
    let (modes, eigenvalues, states) = c.relaxation();
    let energies: Vec<f64> = states
        .iter()
        .map(|a| a.iter().map(|x| x * x).sum::<f64>() / 2.0)
        .collect();
    let helicities: Vec<f64> = states
        .iter()
        .map(|a| a.iter().zip(&eigenvalues).map(|(x, l)| x * x / l).sum())
        .collect();
    // The fields reconstructed from the amplitudes are the curl of their superposition.
    let fields = reconstruct(&states, &modes, &o.curl);
    for (amplitudes, field) in states.iter().zip(&fields) {
        let cochain = amplitudes
            .iter()
            .zip(&modes)
            .fold(Field::<Scalar>::zeros(o.mesh.len(1)), |sum, (a, m)| {
                &sum + &m.map(|x| x * *a)
            });
        let direct = o.curl.of(&cochain);
        assert!(small(
            direct.iter().zip(field.iter()).flat_map(|(a, b)| (a - b).c),
            1e-12
        ));
    }
    assert!(small(helicities.iter().map(|h| h - helicities[0]), 1e-12));
    assert!(energies.windows(2).all(|w| w[1] < w[0]));
    let taylor = helicities[0] * eigenvalues[0] / 2.0;
    let last = energies[FRAMES - 1];
    assert!(energies[0] > 1.8 * taylor);
    assert!(last > taylor);
    assert!((last - taylor).abs() / taylor < 0.01);

    // The final field's energy and helicity, read on the mesh, are the modal ones.
    let cochain = superpose(&states[FRAMES - 1], &modes);
    assert!((pairing(&o.stiffness, &cochain) / 2.0 - last).abs() < 1e-12);
    assert!((pairing(&o.helicity, &cochain) - helicities[0]).abs() < 1e-12);
}

#[test]
fn the_scenario_relaxes_within_two_percent_of_the_taylor_bound() {
    // The scenario's checks: the shown modes' energies and helicities, and the bound.
    let c = cavity();
    let o = &c.operators;
    for index in SHOWN {
        let k = at(index, c.mu.len());
        assert!((pairing(&o.stiffness, &c.modes[k]) / 2.0 - 0.5).abs() < 1e-12);
        assert!((pairing(&o.helicity, &c.modes[k]) - c.mu[k]).abs() < 1e-12);
    }
    let (_, eigenvalues, states) = c.relaxation();
    let energy = |a: &[f64]| a.iter().map(|x| x * x).sum::<f64>() / 2.0;
    let helicity = states[0]
        .iter()
        .zip(&eigenvalues)
        .map(|(x, l)| x * x / l)
        .sum::<f64>();
    let taylor = helicity * eigenvalues[0] / 2.0;
    let last = energy(&states[FRAMES - 1]);
    assert!(last > taylor && (last - taylor) / taylor < 0.02);
}

#[test]
fn frames_draw() {
    let shown = super::shown();
    for t in [0.0, 1.2] {
        gax_numga_examples::app::assert_draws_with(super::draw, &shown, t);
    }
}
