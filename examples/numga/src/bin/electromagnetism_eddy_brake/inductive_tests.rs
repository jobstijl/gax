//! numga's tests of the inductive current patterns: their energy exchange with the rotation,
//! their closure, the periodic drive, and their free decay at second order in time.

use super::brake::{XY, fibre_conductivity};
use super::inductive::*;
use super::scenes::{PERMEABILITY, cold, released};
use super::tests::{CONDUCTANCE, INERTIA, RADIUS, kinetic, magnet};
use gax::ApproxEq;
use gax::field::Field;
use gax_numga_examples::mesh::{Bivector3, Mesh, Scalar3};

/// The patterns kept, and the drive's harmonics.
const COUNT: usize = 8;
const HARMONICS: usize = 16;

/// A small disc and its patterns for a material sharing the conductance as `shares`.
fn patterns(mesh: &Mesh, shares: [f64; 2]) -> CurrentModes {
    let principal = shares.map(|s| s * CONDUCTANCE);
    let conductivity = fibre_conductivity(mesh, principal, 0.0, RADIUS, CONDUCTANCE);
    current_modes(mesh, &conductivity, PERMEABILITY, COUNT)
}

/// The motor's torque `[steps][substeps]`, `torque` for the first `on` steps.
fn pulse(torque: f64, on: usize, steps: usize, substeps: usize) -> Vec<Vec<Bivector3>> {
    let at = |s: usize| XY * if s < on { torque } else { 0.0 };
    (0..steps).map(|s| vec![at(s); substeps]).collect()
}

/// The squares of a field's coefficients, summed.
fn squares(field: &Field<Scalar3>) -> f64 {
    field.dot(field)
}

/// Whether no face's heat drops.
fn heat_never_drops(states: &[InductiveMotion]) -> bool {
    states.windows(2).all(|w| {
        w[0].heat
            .iter()
            .zip(w[1].heat.iter())
            .all(|(a, b)| b.s() - a.s() >= -1e-14)
    })
}

#[test]
fn inductive_currents_exchange_energy_with_rotation_and_remain_closed() {
    let (strength, step, steps, substeps) = (1.0, 1e-4, 30, 2);
    let preparation_time = 0.01;
    let motor_steps = 10;
    let mesh = Mesh::disk(RADIUS, 3, 16);
    let coupling = magnetic_coupling(&mesh, PERMEABILITY);
    // A spinning disc; one at rest with a prepared current opposing the drive; one at rest
    // spun by the motor.
    let cases = [
        ([1.0, 1.0], 120.0, 0.0, 0.0),
        ([1.5, 0.5], 0.0, 1.0, 0.0),
        ([1.5, 0.5], 0.0, 0.0, 20.0),
    ];
    let (mut gap, mut projected_size) = (0.0, 0.0);
    for (case, (shares, speed, stored, torque)) in cases.into_iter().enumerate() {
        let modes = patterns(&mesh, shares);
        let start = released(speed);
        let unturned = start.orientation;
        let amplitudes = modes
            .drive(&mesh, &motional(&mesh, &magnet(strength), unturned))
            .scaled(-preparation_time * stored);
        let forcing = periodic_drive(&mesh, &modes, &magnet(strength), unturned, HARMONICS);
        let disc = InductiveDisc {
            mesh: &mesh,
            modes: &modes,
            forcing: &forcing,
            plane: XY,
            inertia: INERTIA,
        };
        let states = disc.braking(
            start,
            cold(&mesh),
            amplitudes,
            &pulse(torque, motor_steps, steps, substeps),
            step,
        );

        let energy = |s: &InductiveMotion| {
            s.heat.sum().s() + s.magnetic_energy + kinetic(INERTIA, s.motion.spin)
        };
        let first = energy(&states[0]);
        for s in &states {
            assert!((energy(s) - first - s.ledger.motor_work).abs() < 1e-9);
            assert!((s.heat.sum().s() - s.ledger.thermal_energy).abs() < 1e-11);
            // The stored magnetic energy agrees with the integrated interaction of the currents.
            let integrated = s.current.dot(&(&coupling * &s.current)) / 2.0;
            assert!((integrated - s.magnetic_energy).abs() < 1e-11);
            assert_closed(&mesh, s);
        }
        assert!(heat_never_drops(&states));
        let motor_work: Vec<f64> = states.iter().map(|s| s.ledger.motor_work).collect();
        match case {
            0 => assert!(states[1].magnetic_energy > 0.0),
            // The prepared opposing current accelerates the stationary rotor.
            1 => assert!(-(XY | states[1].motion.spin).s() > 0.0),
            // The motor supplies the disc's energy, then it coasts and brakes.
            _ => {
                assert!(motor_work[steps] > 0.0);
                assert!(
                    motor_work[motor_steps..]
                        .iter()
                        .all(|w| (w - motor_work[motor_steps]).abs() < 1e-12)
                );
                let spin = |k: usize| states[k].motion.spin.norm_squared();
                assert!(spin(steps) < spin(motor_steps));
            }
        }
        // Between its angular samples, the periodic drive agrees with projecting the magnet.
        for angle in [-0.37, 0.31, 2.11] {
            let turn = (XY * (-angle / 2.0)).exp() * unturned;
            let projected = modes.drive(&mesh, &motional(&mesh, &magnet(strength), turn));
            gap += squares(&(&forcing.at(angle) - &projected));
            projected_size += squares(&projected);
        }
    }
    assert!(gap < 1e-6 * projected_size);
}

/// Each face's outward flux is the oriented difference of its streamfunction along that edge:
/// shared edges cancel, and the constant stream on the rim leaks nothing.
fn assert_closed(mesh: &Mesh, state: &InductiveMotion) {
    let differences = &mesh.d0() * &state.streamfunction;
    let planes = mesh.face_planes();
    for (k, edges) in mesh.triangle_edges().iter().enumerate() {
        for f in 0..mesh.faces.len() {
            let outward = planes.site(f) | edges.site(f);
            let normal_flux = (state.current.site(f) | outward).s();
            let expected =
                differences.site(mesh.face_edges[f][k]).s() * mesh.face_edge_orientation[f][k];
            assert!((normal_flux - expected).abs() < 1e-9);
        }
    }
    let rim = mesh.boundary_edges();
    assert!(rim.iter().all(|&e| differences.site(e).s().abs() < 1e-12));
}

#[test]
fn unforced_current_memory_decays_to_heat_at_second_order_in_time() {
    let (speed, step, steps) = (20.0, 0.0002, 10);
    let mesh = Mesh::disk(RADIUS, 3, 16);
    let modes = patterns(&mesh, [1.0, 1.0]);
    let amplitudes = Field::from_fn(COUNT, |_| Scalar3::new(0.02));
    let start = released(speed);
    let forcing = periodic_drive(&mesh, &modes, &magnet(0.0), start.orientation, HARMONICS);
    let disc = InductiveDisc {
        mesh: &mesh,
        modes: &modes,
        forcing: &forcing,
        plane: XY,
        inertia: INERTIA,
    };
    // The zero-field current has its own exponential solution, every pattern included.
    let elapsed = step * steps as f64;
    let decayed = amplitudes.zip_with(&modes.decay, |a, d| a * (-d.s() * elapsed).exp());
    let expected = modes.currents(&decayed);
    let runs: Vec<Vec<InductiveMotion>> = [1, 2, 4]
        .iter()
        .map(|&substeps| {
            disc.braking(
                start,
                cold(&mesh),
                amplitudes.clone(),
                &vec![vec![XY * 0.0; substeps]; steps],
                step,
            )
        })
        .collect();
    let errors: Vec<f64> = runs
        .iter()
        .map(|states| {
            let gap = &states.last().expect("states").current - &expected;
            gap.dot(&gap)
        })
        .collect();
    let states = runs.last().expect("runs");
    let energy = |s: &InductiveMotion| s.heat.sum().s() + s.magnetic_energy;
    assert!(
        states
            .iter()
            .all(|s| (energy(s) - energy(&states[0])).abs() < 1e-11)
    );
    assert!(
        states
            .iter()
            .all(|s| s.motion.spin.max_abs_diff(&start.spin) < 1e-12)
    );
    assert!(heat_never_drops(states));
    assert!(
        states
            .windows(2)
            .all(|w| w[1].magnetic_energy < w[0].magnetic_energy)
    );
    // The errors squared, each under 0.3² of the coarser one's.
    assert!(
        errors[1] < 0.09 * errors[0] && errors[2] < 0.09 * errors[1],
        "{errors:?}"
    );
}
