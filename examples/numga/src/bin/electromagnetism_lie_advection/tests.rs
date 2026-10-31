//! numga's tests: Cartan's formula and the edge world sheets on a moving triangle; the swept
//! patches, a uniform field's stationarity and gauge invariance on a disk; ideal conservation of
//! flux and energy against resistive dissipation; and the scenario's checks at its size.

use super::advection::*;
use super::scenes::*;
use crate::sta2d::{self, Point, xy};
use gax::field::Field;
use gax_numga_examples::rng::{Draw, rng};

fn small(values: impl IntoIterator<Item = f64>, tolerance: f64) -> bool {
    values.into_iter().all(|v| v.abs() < tolerance)
}

/// The material rate of each side's line integral, by central differences of the moved triangle,
/// and the rate of its face flux.
fn material_rates(
    corners: [Point; 3],
    forms: [Plane; 3],
    velocities: [Point; 3],
) -> ([f64; 3], f64) {
    let gradients = hat_gradients(sides(corners));
    let motions = motions(velocities);
    let moved = |k: f64| [0, 1, 2].map(|i| corners[i] + motions[i] * k);
    let integrals = |c: [Point; 3]| {
        let s = sides(c);
        let middles = side_middles(c);
        let form_at = |p: Point| {
            (0..3)
                .map(|i| forms[i] * (Scalar::new(1.0) + (gradients[i] & (p - corners[i]))))
                .fold(Plane::zero(), |a, b| a + b)
        };
        let readings = [0, 1, 2].map(|k| (form_at(middles[k]) & s[k]).s());
        (readings, (s[0] & s[1]) * 0.5)
    };
    let ((plus, plus_patch), (minus, minus_patch)) =
        (integrals(moved(0.5)), integrals(moved(-0.5)));
    let face = cartan_decomposition(corners, forms, velocities).face_bivector;
    (
        [0, 1, 2].map(|k| plus[k] - minus[k]),
        (face & (plus_patch - minus_patch)).s(),
    )
}

#[test]
fn cartan_magic_formula_and_edge_world_sheets_on_moving_triangle() {
    let (corners, forms, velocities) = (single_triangle(), corner_forms(), corner_velocities());
    let cartan = cartan_decomposition(corners, forms, velocities);
    // Associativity of the regressive pairing: dA & (u & e) is ((dA & u) & e).
    let s = sides(corners);
    let middles = side_middles(motions(velocities));
    for k in 0..3 {
        let interior = (cartan.face_bivector & middles[k]) & s[k];
        assert!((cartan.swept_fluxes[k] - interior).s().abs() < 1e-13);
    }
    // Around the closed boundary the endpoint differences cancel.
    let total: f64 = cartan.endpoint_differences.iter().map(|d| d.s()).sum();
    assert!(total.abs() < 1e-15);
    // Cartan's sum is the exact material rate of the sides' line integrals.
    let (rates, face_rate) = material_rates(corners, forms, velocities);
    for (k, rate) in rates.iter().enumerate() {
        let sum = (cartan.endpoint_differences[k] + cartan.swept_fluxes[k]).s();
        assert!((sum - rate).abs() < 1e-13);
    }
    // The boundary's total rate is the face flux's rate.
    let swept: f64 = cartan.swept_fluxes.iter().map(|d| d.s()).sum();
    assert!((swept - face_rate).abs() < 1e-13);
}

#[test]
fn world_sheets_uniform_field_preservation_and_gauge_invariance() {
    let disk = Disk::new(6);
    let o = &disk.operators;
    let stream_differences = o.d0.of(&disk.stream);
    let gradients = o.gradient.of(&disk.stream);
    assert!(small(
        gradients
            .iter()
            .zip(disk.face_velocity.iter())
            .map(|(g, v)| (g & v).s()),
        1e-12
    ));
    let swept_area = disk.swept.map(|p| xy() & p);
    assert!(small(
        swept_area
            .iter()
            .zip(stream_differences.iter())
            .map(|(a, d)| (a + d).s()),
        1e-13
    ));
    assert!(small(o.d1.of(&swept_area).iter().map(|x| x.s()), 1e-13));

    // A uniform magnetic 2-form gives an exact gradient rate, −B₀ d₀ψ: its face flux stays.
    let strength = 1.35;
    let uniform = o
        .edge_midpoints
        .zip_with(&o.edges, |m, e| ((xy() & m) & e) * (strength / 2.0));
    assert!(small(
        o.curl
            .of(&uniform)
            .iter()
            .flat_map(|c| (c - xy() * strength).c),
        1e-12
    ));
    let uniform_rate = disk.advection.of(&uniform);
    assert!(small(
        uniform_rate
            .iter()
            .zip(stream_differences.iter())
            .map(|(r, d)| (r + d * strength).s()),
        1e-12
    ));
    assert!(small(o.d1.of(&uniform_rate).iter().map(|x| x.s()), 1e-12));

    // Adding an exact gradient d₀φ leaves the curl, the curl stiffness and Laplacian, and the
    // evolution as they were.
    let cochain = disk.start.clone();
    let gauge = o.vertices.map(|p| {
        let [x, y] = sta2d::coordinates(p);
        Scalar::new(0.8 * x * x - 0.55 * y)
    });
    let gauged = &cochain + &o.d0.of(&gauge);
    let apart = |a: Field<Scalar>, b: Field<Scalar>| {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).s())
            .fold(0.0, |m: f64, d| m.max(d.abs()))
    };
    assert!(apart(o.d1.of(&gauged), o.d1.of(&cochain)) < 1e-14);
    assert!(small(
        o.curl
            .of(&gauged)
            .iter()
            .zip(o.curl.of(&cochain).iter())
            .flat_map(|(a, b)| (a - b).c),
        1e-12
    ));
    let stiff = |a: &Field<Scalar>| o.curl_stiffness.of(a).map(|x| x.undual());
    assert!(apart(stiff(&gauged), stiff(&cochain)) < 1e-11);
    assert!(apart(o.curl_laplacian.of(&gauged), o.curl_laplacian.of(&cochain)) < 1e-11);
    for resistivity in RESISTIVITIES {
        let rate = disk.rate(resistivity);
        assert!(apart(rate.of(&gauged), rate.of(&cochain)) < 1e-12);
    }
}

/// The energies and total fluxes of a run.
fn traces(o: &Operators, states: &[Field<Scalar>]) -> (Vec<f64>, Vec<f64>) {
    states
        .iter()
        .map(|a| (o.energy(a), o.total_flux(a)))
        .unzip()
}

#[test]
fn ideal_flux_and_energy_conservation_versus_resistive_dissipation() {
    let disk = Disk::new(7);
    let o = &disk.operators;
    // For any readings, the ideal rate is orthogonal to the currents and the resistive one
    // dissipates. numpy's draw cannot be reproduced; any draw shows it.
    let mut random = rng(7);
    let readings = Field::from_fn(o.mesh.len(1), |_| Scalar::new(random.normal()));
    let currents = o.curl_stiffness.of(&readings);
    let pair = |rate: &Field<Scalar>| -> f64 {
        rate.iter()
            .zip(currents.iter())
            .map(|(r, c)| (r & c).s())
            .sum()
    };
    let energy = pair(&readings);
    assert!((pair(&disk.advection.of(&readings)) / energy).abs() < 1e-12);
    assert!(pair(&o.curl_laplacian.of(&readings).map(|x| -x)) < 0.0);

    for (case, resistivity) in RESISTIVITIES.into_iter().enumerate() {
        let states = advect(&disk.start, &disk.rate(resistivity), DT, 60);
        let (energies, fluxes) = traces(o, &states);
        let tolerance = [1e-14, 1e-13][case];
        assert!(small(fluxes.iter().map(|f| f - fluxes[0]), tolerance));
        if case == 0 {
            assert!(small(energies.iter().map(|e| e / energies[0] - 1.0), 1e-5));
        } else {
            assert!(energies.windows(2).all(|w| w[1] < w[0]));
        }
    }
}

#[test]
fn the_scenario_conserves_and_dissipates_at_its_size() {
    // The scenario's checks.
    let disk = Disk::new(DISK_DIVISIONS);
    let o = &disk.operators;
    let gradients = o.gradient.of(&disk.stream);
    assert!(small(
        gradients
            .iter()
            .zip(disk.face_velocity.iter())
            .map(|(g, v)| (g & v).s()),
        1e-12
    ));
    let stream_differences = o.d0.of(&disk.stream);
    assert!(small(
        disk.swept
            .iter()
            .zip(stream_differences.iter())
            .map(|(p, d)| ((xy() & p) + d).s()),
        1e-12
    ));
    let currents = o.curl_stiffness.of(&disk.start);
    let production: f64 = disk
        .advection
        .of(&disk.start)
        .iter()
        .zip(currents.iter())
        .map(|(r, c)| (r & c).s())
        .sum();
    assert!(production.abs() < 1e-12);
    for (case, resistivity) in RESISTIVITIES.into_iter().enumerate() {
        let states = advect(&disk.start, &disk.rate(resistivity), DT, STEPS);
        let (energies, fluxes) = traces(o, &states);
        let tolerance = [1e-13, 1e-14][case];
        assert!(small(fluxes.iter().map(|f| f - fluxes[0]), tolerance));
        if case == 0 {
            assert!(small(energies.iter().map(|e| e / energies[0] - 1.0), 1e-4));
        } else {
            assert!(energies.windows(2).all(|w| w[1] < w[0]));
        }
    }
}

#[test]
fn frames_draw() {
    let shown = super::shown();
    for t in [0.0, 2.2] {
        gax_numga_examples::app::assert_draws_with(super::draw, &shown, t);
    }
}
