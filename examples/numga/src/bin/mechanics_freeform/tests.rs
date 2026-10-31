//! numga's tests of FreeForm elasticity: affine reproduction and skinning modes, the stress as
//! the energy's derivative (a flat collapse included) and its covariance, the reduced forces as
//! the potential's derivative, and the whitened minimizer; and the scenario's checks.

use super::freeform::*;
use super::numerics::{minimize_state, work};
use gax::field::Field;
use gax_numga_examples::rng::{Draw, rng};
use std::sync::OnceLock;

/// numga's cube: eight centres at its corners, twenty-seven queries inside, equal volumes, and
/// a tetrahedron's corners around it.
fn cube_samples() -> (Vec<Antivector>, Vec<Antivector>, Vec<f64>, [Antivector; 4]) {
    let grid = |values: &[f64]| -> Vec<Antivector> {
        let mut out = Vec::new();
        for &x in values {
            for &y in values {
                for &z in values {
                    out.push(point(x, y, z));
                }
            }
        }
        out
    };
    let centers = grid(&[-0.7, 0.7]);
    let queries = grid(&[-0.45, 0.0, 0.45]);
    let volumes = vec![1.0 / queries.len() as f64; queries.len()];
    let corners = [
        point(1.0, 1.0, 1.0),
        point(1.0, -1.0, -1.0),
        point(-1.0, 1.0, -1.0),
        point(-1.0, -1.0, 1.0),
    ];
    (queries, centers, volumes, corners)
}

fn x_of(p: Antivector) -> f64 {
    coordinates(p)[0]
}
fn y_of(p: Antivector) -> f64 {
    coordinates(p)[1]
}

#[test]
fn reproducing_kernels_and_material_weighted_skinning_modes() {
    let (queries, centers, volumes, _) = cube_samples();
    let radius = 1.1;
    let (evaluation, gradient) = kernels(&queries, &centers, radius);
    let slope = Vector::new(0.3, -0.7, 0.2);
    let affine = |p: Antivector| Scalar::new(1.2 + (slope & p).s());
    let nodal: Field<Scalar> = centers.iter().map(|c| affine(*c)).collect();
    let values = &evaluation * &nodal;
    let slopes = &gradient * &nodal;
    for (q, (v, g)) in queries.iter().zip(values.iter().zip(slopes.iter())) {
        assert!((v.s() - affine(*q).s()).abs() <= 1e-10);
        assert!((g - slope).c.iter().all(|c| c.abs() <= 1e-10), "{g:?}");
    }

    // A non-affine field tests the changing kernel correction, beyond the reproduced patch.
    let nodal: Field<Scalar> = centers
        .iter()
        .map(|c| Scalar::new(x_of(*c) * y_of(*c)))
        .collect();
    let heading = point(0.3, -0.7, 0.2);
    let step = 1e-5;
    let shifted = |sign: f64| -> Vec<Antivector> {
        queries
            .iter()
            .map(|q| *q + heading * (sign * step))
            .collect()
    };
    let (ahead, _) = kernels(&shifted(1.0), &centers, radius);
    let (behind, _) = kernels(&shifted(-1.0), &centers, radius);
    let finite = (&(&ahead * &nodal) - &(&behind * &nodal)).scaled(1.0 / (2.0 * step));
    for (f, g) in finite.iter().zip((&gradient * &nodal).iter()) {
        let exact = (g & heading).s();
        assert!(
            (f.s() - exact).abs() <= 1e-7 + 1e-7 * exact.abs(),
            "{f:?} {exact}"
        );
    }

    let n = queries.len();
    let lambdas: Vec<f64> = (0..n).map(|i| 1.0 + i as f64 / (n - 1) as f64).collect();
    let mus: Vec<f64> = (0..n).map(|i| 0.5 + i as f64 / (n - 1) as f64).collect();
    let (eigenvalues, modes) = modes(&evaluation, &gradient, &volumes, &lambdas, &mus, 4);
    let samples: Vec<Field<Scalar>> = modes.iter().map(|m| &evaluation * m).collect();
    let changes: Vec<Field<Vector>> = modes.iter().map(|m| &gradient * m).collect();
    for i in 0..4 {
        for j in 0..4 {
            let mass: f64 = (0..n)
                .map(|s| samples[i].site(s).s() * samples[j].site(s).s() * volumes[s])
                .sum();
            let stiffness: f64 = (0..n)
                .map(|s| {
                    let weight = volumes[s] * (lambdas[s] + 4.0 * mus[s]);
                    (changes[i].site(s) | changes[j].site(s)).s() * weight
                })
                .sum();
            let delta = if i == j { 1.0 } else { 0.0 };
            assert!(
                (mass - delta).abs() <= 1e-10 + 1e-10 * delta,
                "{i} {j} {mass}"
            );
            let expected = delta * eigenvalues[i];
            assert!(
                (stiffness - expected).abs() <= 1e-9 + 1e-9 * expected.abs(),
                "{i} {j} {stiffness} {expected}"
            );
        }
    }
    assert!(
        changes[0]
            .iter()
            .all(|g| g.c.iter().all(|c| c.abs() <= 1e-10))
    );
}

/// `a` times the map reading the coordinate along `axis`.
fn along(a: Antivector, axis: usize) -> Deformation {
    a * (planes()[axis] & Antivector::slot())
}

#[test]
fn hyperelastic_stress_is_energy_derivative_even_at_flat_collapse() {
    let [x, y, z] = directions();
    let identity = Antivector::slot();
    let deformed = identity + along(x, 1) * 0.3 - along(y, 2) * 0.1 + along(z, 2) * 0.2;
    let flat = identity - along(z, 2);
    let (lame_lambda, lame_mu) = (3.2, 1.7);
    let change = along(z, 0) * 0.2 - along(y, 1) * 0.4 + along(z, 2) * 0.3;
    let step = 1e-5;
    let rotor = (Vector::new(1.0, 0.0, 0.0).wedge(Vector::new(0.0, 1.0, 0.0)) * 0.3
        + Vector::new(0.0, 1.0, 0.0).wedge(Vector::new(0.0, 0.0, 1.0)) * 0.2)
        .exp();
    for local in [deformed, flat] {
        let (energy, stress) = constitutive(local, lame_lambda, lame_mu);
        let (ahead, _) = constitutive(local + change * step, lame_lambda, lame_mu);
        let (behind, _) = constitutive(local - change * step, lame_lambda, lame_mu);
        let finite = (ahead - behind) / (2.0 * step);
        // The work of the stress: the force on each plane across the axes, paired with the
        // change along its axis.
        let work: f64 = planes()
            .iter()
            .zip(directions())
            .map(|(p, d)| (stress.of(*p) & change.of(d)).s())
            .sum();
        assert!(
            (finite - work).abs() <= 1e-7 + 1e-7 * work.abs(),
            "{finite} {work}"
        );

        // A rigid placement changes the stress's output frame without changing elastic energy.
        let (placed_energy, placed_stress) = constitutive(rotor >> local, lame_lambda, lame_mu);
        assert!((placed_energy - energy).abs() <= 1e-10 + 1e-10 * energy.abs());
        let turned = rotor >> stress;
        assert!(
            placed_stress
                .c
                .iter()
                .flatten()
                .zip(turned.c.iter().flatten())
                .all(|(a, b)| (a - b).abs() <= 1e-10 + 1e-10 * b.abs())
        );
    }
}

#[test]
fn the_covector_pairs_as_the_metric() {
    let d = point(0.3, -1.2, 0.7);
    assert!(((covector(d) & d).s() - dot(d, d)).abs() < 1e-15);
    assert!((dot(d, d) - (0.09 + 1.44 + 0.49)).abs() < 1e-15);
    assert_eq!(unit_volume().s(), 1.0);
}

#[test]
fn reduced_forces_match_energy_and_metric_whitening_preserves_the_solution() {
    let (queries, centers, volumes, corners) = cube_samples();
    let radius = 1.1;
    let constant: Field<Scalar> = centers.iter().map(|_| Scalar::new(1.0)).collect();
    let curved: Field<Scalar> = centers
        .iter()
        .map(|c| Scalar::new(x_of(*c) * y_of(*c)))
        .collect();
    let (sampling, gradients) =
        deformation_basis(&queries, &centers, radius, &[constant, curved], &corners);
    let mut r = rng(4);
    let controls = 8;
    let mut normal = |scale: f64| point(scale * r.normal(), scale * r.normal(), scale * r.normal());
    let state: Field<Antivector> = (0..controls).map(|_| normal(0.04)).collect();
    let change: Field<Antivector> = (0..controls).map(|_| normal(0.1)).collect();
    let n = queries.len();
    let clamp: Vec<f64> = (0..n)
        .map(|i| 0.1 + 0.4 * i as f64 / (n - 1) as f64)
        .collect();
    let load: Vec<Vector> = (0..n)
        .map(|_| {
            let p = normal(0.03);
            covector(p)
        })
        .collect();
    let body = Body::new(sampling.clone(), &gradients, volumes.clone(), 3.2, 1.7);
    let potential = |value: &Field<Antivector>| body.potential(value, &clamp, &load);
    let (_, force) = potential(&state);
    let step = 1e-5;
    let (ahead, _) = potential(&(&state + &change.scaled(step)));
    let (behind, _) = potential(&(&state - &change.scaled(step)));
    let finite = (ahead - behind) / (2.0 * step);
    let exact = work(&force, &change);
    assert!(
        (finite - exact).abs() <= 1e-7 + 1e-7 * exact.abs(),
        "{finite} {exact}"
    );

    // The optimizer must preserve the physical field solution through its coordinate change.
    let metric = mass(&sampling, &volumes);
    let objective = |value: &Field<Antivector>| {
        let error = value - &state;
        let force = metric.of(&error);
        (work(&force, &error) / 2.0, force)
    };
    let result = minimize_state(&state.scaled(0.0), objective, &metric, 20, 1e-10);
    for (a, b) in result.iter().zip(state.iter()) {
        assert!(
            (a - b)
                .c
                .iter()
                .zip(b.c)
                .all(|(d, s)| d.abs() <= 1e-9 + 1e-9 * s.abs()),
            "{a:?} {b:?}"
        );
    }
}

/// The scenes, computed once for the tests that need them.
fn scenes() -> &'static super::Scenes {
    static SCENES: OnceLock<super::Scenes> = OnceLock::new();
    SCENES.get_or_init(super::scenes)
}

/// numga's scenario checks: the ear follows the pull and springs back after release, while the
/// base moves a tenth as far.
#[test]
fn the_ear_springs_back_and_the_base_holds() {
    let s = scenes();
    let peak = s.ear.iter().copied().fold(f64::MIN, f64::max);
    let last = s.ear[s.ear.len() - 1];
    assert!(peak > 0.3 && last < 0.05 * peak, "{peak} {last}");
    let rest = &s.pulled.states[0];
    #[allow(clippy::disallowed_methods)] // a distance from its square
    let farthest = s
        .pulled
        .states
        .iter()
        .flat_map(|v| {
            v.iter()
                .zip(rest.iter())
                .map(|(a, b)| dot(a - b, a - b).sqrt())
        })
        .fold(0.0, f64::max);
    let base = s.base.iter().copied().fold(0.0, f64::max);
    assert!(base < 0.1 * farthest, "{base} {farthest}");
}

#[test]
fn a_frame_draws() {
    for t in [0.3, 4.5] {
        gax_numga_examples::app::assert_draws_with(super::draw, scenes(), t);
    }
}
