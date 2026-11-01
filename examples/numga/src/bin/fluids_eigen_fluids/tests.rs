//! numga's tests: the Whitney gradients and the spherical harmonic spectrum on a sphere; on the
//! bunny the modes' orthonormality, tangent flows, the Jacobian pairing, the structure
//! constants' total antisymmetry, and the inviscid run's invariants against the viscous run's
//! selective decay; and the scenario's checks.

use super::eigen::*;
use super::scenes::*;
use crate::simplicial::icosphere;
use gax::field::Field;

/// The largest entry of the modes' Gram matrix in the mass, less the identity: the amounts the
/// mass gives one mode, joined with another and summed.
fn gram_error(surface: &Surface, modes: &[Field<Scalar>]) -> f64 {
    let mass = surface.mass();
    let mut worst: f64 = 0.0;
    for (i, a) in modes.iter().enumerate() {
        let image = mass.of(a);
        for (j, b) in modes.iter().enumerate() {
            let expected = if i == j { 1.0 } else { 0.0 };
            let pairing = image.zip_with(b, |m, x| m & x).sum().s();
            worst = worst.max((pairing - expected).abs());
        }
    }
    worst
}

#[test]
fn whitney_gradients_and_spherical_harmonic_spectrum() {
    let modes_count = 8;
    let (mesh, vectors) = icosphere(2);
    let surface = Surface::new(mesh, vectors.map(|v| v.hodge()));
    let (values, modes) = surface.modes(modes_count, SPHERE_BLOCK);

    let up = Plane::new(0.0, 0.0, 1.0);
    let height = surface.vertices.map(|p| up & p);
    let velocity = surface.velocity(&height);
    for (f, v) in velocity.iter().enumerate() {
        let tangential = up ^ surface.normals.site(f);
        assert!((v - tangential).c.iter().all(|c| c.abs() < 1e-13));
        assert!((v & surface.patches.site(f)).s().abs() < 1e-14);
    }
    assert!(values[..3].iter().all(|l| (l / 2.0 - 1.0).abs() < 2e-2));
    assert!(values[3..].iter().all(|l| (l / 6.0 - 1.0).abs() < 4e-2));
    assert!(gram_error(&surface, &modes) < 1e-12);
}

/// The bunny's flow, built once for the tests that share it.
fn bunny_flow() -> &'static Flow {
    static FLOW: std::sync::OnceLock<Flow> = std::sync::OnceLock::new();
    FLOW.get_or_init(Flow::bunny)
}

#[test]
fn bunny_modes_and_their_flows_and_brackets() {
    let flow = bunny_flow();
    let surface = &flow.surface;
    assert_eq!(surface.mesh.len(0), 3485);
    assert!(gram_error(surface, &flow.modes) < 1e-12);
    for &k in &SHOWN_MODES {
        let velocity = surface.velocity(&flow.modes[k]);
        for (v, p) in velocity.iter().zip(surface.patches.iter()) {
            assert!((v & p).s().abs() < 1e-14);
        }
    }
    // On every triangle the Jacobian wedge pairing equals the area-weighted directional
    // derivative.
    let gradients: Vec<Field<Plane>> = flow.modes[..6]
        .iter()
        .map(|m| surface.gradient.of(m))
        .collect();
    for f in 0..surface.mesh.len(2) {
        let (patch, normal, area) = (
            surface.patches.site(f),
            surface.normals.site(f),
            surface.areas.site(f),
        );
        for a in &gradients {
            for b in &gradients {
                let (a, b) = (a.site(f), b.site(f));
                let wedge = ((a ^ b) & patch).s();
                let directional = ((a & (b ^ normal)) * area).s();
                assert!((wedge - directional).abs() < 1e-14);
            }
        }
    }
    // The structure constants are totally antisymmetric in their three modes, to rounding.
    let b = &flow.bracket;
    let mut worst: f64 = 0.0;
    for i in 0..b.n {
        for j in 0..b.n {
            for k in 0..b.n {
                worst = worst
                    .max((b.at(i, j, k) + b.at(i, k, j)).abs())
                    .max((b.at(i, j, k) + b.at(j, i, k)).abs());
            }
        }
    }
    assert!(worst < 1e-13, "{worst}");
}

#[test]
fn inviscid_runs_conserve_and_viscosity_damps_enstrophy_first() {
    let flow = bunny_flow();
    let start = &flow.start;
    let rate = vorticity_rate(start, &flow.bracket, &flow.eigenvalues, 0.0);
    let stream = start.zip_with(&flow.eigenvalues, |w, l| w * l.inverse());
    assert!(stream.dot(&rate).abs() < 1e-11);
    assert!(start.dot(&rate).abs() < 1e-11);

    let inviscid = flow.run(0.0, DT, STEPS);
    let viscous = flow.run(VISCOSITY, DT, STEPS);
    let trace = |run: &[Field<Scalar>]| -> Vec<(f64, f64)> {
        run.iter()
            .map(|w| invariants(w, &flow.eigenvalues))
            .collect()
    };
    let (inviscid, viscous) = (trace(&inviscid), trace(&viscous));
    let (e0, z0) = inviscid[0];
    for &(e, z) in &inviscid {
        assert!((e / e0 - 1.0).abs() < 5e-5 && (z / z0 - 1.0).abs() < 5e-5);
    }
    for pair in viscous.windows(2) {
        assert!(pair[1].0 < pair[0].0 && pair[1].1 < pair[0].1);
    }
    let (last, first) = (viscous[STEPS], viscous[0]);
    assert!(last.1 / first.1 < last.0 / first.0);

    // The reconstructed flows of the sampled frames stay tangent.
    for w in [&flow.start] {
        let (_, stream) = flow.fields(w);
        let velocity = flow.surface.velocity(&stream);
        for (v, p) in velocity.iter().zip(flow.surface.patches.iter()) {
            assert!((v & p).s().abs() < 1e-13);
        }
    }
}

#[test]
fn the_sphere_flow_keeps_its_invariants() {
    let flow = Flow::sphere();
    // The complete bands through degree eight: 3, 5, … 17 copies near l (l + 1), the finest
    // within 3 % on this mesh (a check of the port's own).
    let values: Vec<f64> = flow.eigenvalues.iter().map(|l| l.s()).collect();
    let mut first = 0;
    for l in 1..=8 {
        let copies = 2 * l + 1;
        let expected = (l * (l + 1)) as f64;
        assert!(
            values[first..first + copies]
                .iter()
                .all(|v| (v / expected - 1.0).abs() < 0.03),
            "band {l}: {:?}",
            &values[first..first + copies]
        );
        first += copies;
    }
    let run = flow.run(0.0, SPHERE_DT, 40);
    let (e0, z0) = invariants(&run[0], &flow.eigenvalues);
    let (e, z) = invariants(&run[40], &flow.eigenvalues);
    assert!((e / e0 - 1.0).abs() < 5e-5 && (z / z0 - 1.0).abs() < 5e-5);
}

#[test]
fn frames_draw() {
    let shown = super::shown();
    for t in [0.0, 2.1] {
        gax_numga_examples::app::assert_draws_with(super::draw, &shown, t);
    }
}
