//! numga's tests of the eddy brake: charge conservation, magnetic braking, the field's
//! covariance and the time steps' order; and the disc and magnet the inductive tests share.
//! numga batches the materials, speeds and fields of a test; here each is a loop.

use super::brake::*;
use super::inductive::*;
use super::sample::{Located, locate};
use super::scenes::cold;
use gax::field::{Field, SparseMap};
use gax::vga3d::{Scalar, Vector};
use gax::{ApproxEq, Extensor};
use gax_numga_examples::mesh::{Bivector3, Mesh, Scalar3, Vector3};
use std::collections::BTreeMap;

/// numga's test disc and magnet, in SI units: a thinner sheet than the scenes'.
pub(super) const RADIUS: f64 = 0.1;
const STRENGTH: f64 = 0.2;
const WIDTH: f64 = 0.02;
const OFFSET: f64 = 0.055;
pub(super) const CONDUCTANCE: f64 = 5.8e4;
pub(super) const INERTIA: f64 = 0.005;
const X: Vector3 = Vector3::new(1.0, 0.0, 0.0);
const Y: Vector3 = Vector3::new(0.0, 1.0, 0.0);
/// The isotropic, radial and circumferential materials, as shares of the conductance along the
/// radius and around it.
const MATERIALS: [[f64; 2]; 3] = [[1.0, 1.0], [1.9, 0.1], [0.1, 1.9]];

/// The test's magnet: along `x`, in the disc's plane.
pub(super) fn magnet(strength: f64) -> Magnet {
    Magnet {
        centre: X * OFFSET,
        plane: XY,
        strength,
        width: WIDTH,
    }
}

/// The open vector.
fn open() -> Conductivity {
    Vector::slot()
}

/// The same conductance on every face.
fn uniform(mesh: &Mesh, conductance: Conductivity) -> Field<Conductivity> {
    Field::from_fn(mesh.faces.len(), |_| conductance)
}

/// Fibres across the whole disc, the conductance shared out as `shares`.
fn fibres(mesh: &Mesh, shares: [f64; 2]) -> Field<Conductivity> {
    fibre_conductivity(
        mesh,
        shares.map(|s| s * CONDUCTANCE),
        0.0,
        RADIUS,
        CONDUCTANCE,
    )
}

/// The turn by `angle` in a plane, numga's `exp(plane * -angle / 2)`.
fn turn(plane: Bivector3, angle: f64) -> Turn {
    (plane * (-angle / 2.0)).exp()
}

/// The largest coefficient of a field.
fn largest<M: Extensor<Coef = f64>>(field: &Field<M>, zero: M) -> f64 {
    field
        .iter()
        .map(|v| v.max_abs_diff(&zero))
        .fold(0.0, f64::max)
}

/// The largest difference of two fields, coefficient by coefficient.
fn difference<M: Extensor<Coef = f64>>(a: &Field<M>, b: &Field<M>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| x.max_abs_diff(&y))
        .fold(0.0, f64::max)
}

/// The length of numbers taken together as one vector.
#[allow(clippy::disallowed_methods)] // a test's reference: numpy's norm
fn length(values: impl Iterator<Item = f64>) -> f64 {
    values.map(|x| x * x).sum::<f64>().sqrt()
}

/// The length of a field's coefficients, all together.
fn size(field: &Field<Vector3>) -> f64 {
    length(field.iter().flat_map(|v| v.c))
}

/// The charge each vertex gains from the dual-edge fluxes: zero where charge is conserved.
fn charge_balance(mesh: &Mesh, flux: &Field<Scalar3>) -> Field<Scalar3> {
    &mesh.d0().reverse() * flux
}

/// The speed about the disc's plane.
fn speed(spin: Bivector3) -> f64 {
    -(XY | spin).s()
}

/// The disc's turned brake under the test magnet.
fn brake<'a>(mesh: &'a Mesh, conductivity: &'a Field<Conductivity>, magnet: Magnet) -> Brake<'a> {
    Brake {
        conductor: Conductor::new(mesh, conductivity),
        magnet,
        inertia: INERTIA,
    }
}

/// The kinetic energy of the turning disc, `I |ω|² / 2`.
pub(super) fn kinetic(inertia: f64, spin: Bivector3) -> f64 {
    inertia / 2.0 * spin.norm_squared()
}

/// Rotational energy plus heat of each state.
fn energies(states: &[Motion]) -> Vec<f64> {
    states
        .iter()
        .map(|s| s.heat.sum().s() + kinetic(INERTIA, s.spin))
        .collect()
}

#[test]
fn a_uniform_field_is_cancelled_by_the_charge_distribution() {
    let (strength, speed) = (0.2, 1.0);
    let mesh = Mesh::disk(RADIUS, 6, 36);
    let conductivity = uniform(&mesh, (open() + X * (X | open())) * CONDUCTANCE);
    let magnetic = Field::from_fn(mesh.edges.len(), |_| XY * strength);
    let response = solve(&mesh, &magnetic, XY * speed, &conductivity);
    // The midpoint edge integral is exact for rigid rotation in a uniform field.
    let potential = mesh
        .vertices
        .map(|v| Scalar3::new(strength * speed * v.norm_squared() / 2.0));
    let first = potential.site(0);
    let expected = potential.map(|p| p - first);
    assert!(difference(&response.potential, &expected) < 1e-12);
    assert!(largest(&response.voltage, Scalar3::new(0.0)) < 1e-12);
    assert!(largest(&response.flux, Scalar3::new(0.0)) < 1e-9);
    assert!(largest(&response.current, Vector3::zero()) < 1e-8);
    assert!(response.torque.max_abs_diff(&Bivector3::zero()) < 1e-12);
}

#[test]
fn localized_field_conserves_charge_and_turns_mechanical_power_into_heat() {
    let mesh = Mesh::disk(RADIUS, 8, 48);
    let magnetic = magnet(STRENGTH).field(&mesh.edge_midpoints());
    let spin = XY;
    let heats = MATERIALS.map(|shares| {
        let response = solve(&mesh, &magnetic, spin, &fibres(&mesh, shares));
        let heat = response.heating.sum().s();
        let balance = charge_balance(&mesh, &response.flux);
        assert!(largest(&balance, Scalar3::new(0.0)) < 1e-9);
        let mechanical_loss = (spin | response.torque).s();
        assert!((mechanical_loss - heat).abs() < 1e-12);
        // The torque opposes the spin (bivectors square to minus one).
        assert!(heat > 0.0 && mechanical_loss > 0.0);
        heat
    });
    // With the same mean conductance, resistance across the fibres reduces both current loops.
    assert!(heats[1] < heats[0] / 2.0 && heats[2] < heats[0] / 2.0);
}

#[test]
fn speed_field_and_conductance_have_their_physical_scalings() {
    let mesh = Mesh::disk(RADIUS, 6, 36);
    let magnetic = magnet(STRENGTH).field(&mesh.edge_midpoints());
    let conductivity = uniform(&mesh, open() * CONDUCTANCE);
    let doubled = uniform(&mesh, open() * (2.0 * CONDUCTANCE));
    let base = solve(&mesh, &magnetic, XY, &conductivity);
    for (speed, strength) in [(1.0, 1.0), (2.0, 1.0), (1.0, 2.0), (-1.0, 1.0)] {
        let scaled = magnetic.map(|b| b * strength);
        let response = solve(&mesh, &scaled, XY * speed, &conductivity);
        let current = speed * strength;
        let torque = speed * strength * strength;
        let heat = current * current;
        assert!(difference(&response.current, &base.current.scaled(current)) < 1e-8);
        assert!(response.torque.max_abs_diff(&(base.torque * torque)) < 1e-12);
        assert!(difference(&response.heating, &base.heating.scaled(heat)) < 1e-12);
    }
    let twice = solve(&mesh, &magnetic, XY, &doubled);
    assert!(difference(&twice.potential, &base.potential) < 1e-12);
    assert!(difference(&twice.current, &base.current.scaled(2.0)) < 1e-8);
    assert!(twice.torque.max_abs_diff(&(base.torque * 2.0)) < 1e-12);
    assert!(difference(&twice.heating, &base.heating.scaled(2.0)) < 1e-12);
}

#[test]
fn turning_the_disc_and_magnet_turns_the_solution() {
    let mesh = Mesh::disk(RADIUS, 6, 36);
    // numga's `exp(xz * -0.35)`: `xz = -e31`.
    let rotation = Bivector3::new(0.0, 0.35, 0.0).exp();
    let turned_mesh = mesh.with_vertices(mesh.vertices.map(|v| rotation >> v));
    let turned_magnet = Magnet {
        centre: rotation >> (X * OFFSET),
        plane: rotation >> XY,
        ..magnet(STRENGTH)
    };
    let conductivity = uniform(&mesh, (open() + X * (X | open())) * CONDUCTANCE);
    let turned_conductivity = conductivity.map(|c| rotation >> c.of(rotation << open()));
    let response = solve(
        &mesh,
        &magnet(STRENGTH).field(&mesh.edge_midpoints()),
        XY,
        &conductivity,
    );
    let turned_field = turned_magnet.field(&turned_mesh.edge_midpoints());
    let turned = solve(
        &turned_mesh,
        &turned_field,
        rotation >> XY,
        &turned_conductivity,
    );
    assert!(difference(&turned.potential, &response.potential) < 1e-12);
    assert!(difference(&turned.flux, &response.flux) < 1e-9);
    assert!(difference(&turned.current, &response.current.map(|j| rotation >> j)) < 1e-8);
    assert!(turned.torque.max_abs_diff(&(rotation >> response.torque)) < 1e-12);
    assert!(difference(&turned.heating, &response.heating) < 1e-12);
}

#[test]
fn braking_power_converges_under_mesh_refinement() {
    let [coarse, middle, fine] = [1, 2, 4].map(|refinement| {
        let mesh = Mesh::disk(RADIUS, 4 * refinement, 24 * refinement);
        let magnetic = magnet(STRENGTH).field(&mesh.edge_midpoints());
        MATERIALS.map(|shares| {
            solve(&mesh, &magnetic, XY, &fibres(&mesh, shares))
                .heating
                .sum()
                .s()
        })
    });
    // Midpoint edge integration and the polygonal rim both approach the smooth-disc problem.
    for ((a, b), c) in coarse.iter().zip(middle).zip(fine) {
        assert!((c - b).abs() < 0.4 * (b - a).abs());
    }
}

#[test]
fn annular_fibres_lose_mesh_orientation_bias_under_uniform_refinement() {
    let (inner, outer) = (0.2 * RADIUS, 0.85 * RADIUS);
    let materials = [[1.9, 0.1], [0.1, 1.9]];
    let angles = [
        0.0,
        core::f64::consts::PI / 12.0,
        core::f64::consts::PI / 6.0,
    ];
    let samples = grid_inside(0.9 * RADIUS, 61);
    let errors: Vec<[f64; 2]> = [6, 12, 24]
        .iter()
        .map(|&divisions| {
            let mesh = Mesh::concentric_disk(&[inner, outer, RADIUS], divisions);
            // The full steady response of the closed currents, without a modal cutoff, so the
            // check measures the spatial discretization.
            let embedder = embedder(&mesh, &interior(&mesh));
            let current = closed_currents(&mesh, &embedder);
            let areas = mesh.triangle_areas();
            let fields = angles.map(|angle| {
                let turned = turn(XY, angle);
                let world = mesh.with_vertices(mesh.vertices.map(|v| turned >> v));
                let located: Vec<_> = samples
                    .iter()
                    .map(|p| locate(&world, *p).expect("inside the disc"))
                    .collect();
                let electric = motional(&mesh, &magnet(1.0), turned);
                let drive = current
                    .transpose()
                    .apply(&electric.zip_with(&areas, |e, a| e * a.s()));
                materials.map(|shares| {
                    let conductivity = fibre_conductivity(
                        &mesh,
                        shares.map(|s| s * CONDUCTANCE),
                        inner,
                        outer,
                        CONDUCTANCE,
                    );
                    let weighted = resistivity(&conductivity).zip_with(&areas, |r, a| r * a.s());
                    let resistance = current
                        .transpose()
                        .compose(&SparseMap::from_diagonal(&weighted).compose(&current));
                    let stream = &embedder * &resistance.solve_direct(&drive);
                    located
                        .iter()
                        .map(|at| interpolate(&world, &stream, *at))
                        .collect::<Vec<f64>>()
                })
            });
            // The world fields on one grid, including amplitude errors, against the unturned.
            core::array::from_fn(|m| {
                let reference = &fields[0][m];
                let gap =
                    |f: &[Vec<f64>; 2]| length(f[m].iter().zip(reference).map(|(a, b)| a - b));
                fields
                    .iter()
                    .map(|f| gap(f) / length(reference.iter().copied()))
                    .fold(0.0, f64::max)
            })
        })
        .collect();
    for m in 0..2 {
        assert!(
            errors[1][m] < 0.65 * errors[0][m] && errors[2][m] < 0.65 * errors[1][m],
            "{errors:?}"
        );
        assert!(errors[2][m] < 0.025, "{errors:?}");
    }
}

/// A vertex field's value at a located point, its corners' values weighted: linear across the
/// face, as numga's matplotlib interpolator.
fn interpolate(mesh: &Mesh, values: &Field<Scalar3>, at: Located) -> f64 {
    let corners = mesh.faces[at.face];
    (0..3)
        .map(|k| at.weights[k] * values.site(corners[k]).s())
        .sum()
}

/// numga's grid of `n` by `n` points across `±reach`, those inside the circle of that radius.
fn grid_inside(reach: f64, n: usize) -> Vec<Vector3> {
    let along = |i: usize| -reach + 2.0 * reach * i as f64 / (n - 1) as f64;
    (0..n)
        .flat_map(|i| (0..n).map(move |j| Vector3::new(along(j), along(i), 0.0)))
        .filter(|p| p.norm_squared() < reach * reach)
        .collect()
}

/// The couplings of a field map by the sites they couple, repeated couplings added.
fn merged(map: &SparseMap<Scalar<(Scalar,), f64>>) -> BTreeMap<(usize, usize), f64> {
    let mut cells = BTreeMap::new();
    for (row, column, cell) in map.couplings() {
        *cells.entry((row, column)).or_insert(0.0) += cell.c[0][0];
    }
    cells
}

#[test]
fn isotropic_material_recovers_the_cotangent_stiffness() {
    let conductance = 2.5;
    let mesh = Mesh::disk(RADIUS, 6, 36);
    let gradient = mesh
        .reconstruction()
        .compose(&(&mesh.d0() * Scalar::slot()));
    let material = SparseMap::from_diagonal(
        &mesh
            .triangle_areas()
            .map(|a| open() * (a.s() * conductance)),
    );
    let balance = merged(&gradient.transpose().compose(&material.compose(&gradient)));
    let ratio = SparseMap::from_diagonal(&mesh.edge_ratio().map(|r| r * conductance));
    let cotangent = merged(&(&(&(&mesh.d0().reverse() * &ratio) * &mesh.d0()) * Scalar::slot()));
    // The face gradient is constant, so integrating its material pairing at the centroid is exact.
    let keys: std::collections::BTreeSet<_> = balance.keys().chain(cotangent.keys()).collect();
    for k in keys {
        let (a, b) = (
            balance.get(k).unwrap_or(&0.0),
            cotangent.get(k).unwrap_or(&0.0),
        );
        assert!((a - b).abs() < 1e-10);
    }
}

#[test]
fn free_rotation_converts_kinetic_energy_into_heat_with_closed_currents() {
    let mesh = Mesh::disk(RADIUS, 6, 36);
    let conductivity = uniform(&mesh, (open() * 0.1 + X * (X | open()) * 1.8) * CONDUCTANCE);
    let brake = brake(&mesh, &conductivity, magnet(STRENGTH));
    let reconstruction = mesh.reconstruction();
    let areas = mesh.triangle_areas();
    for initial in [2.0, 1.0] {
        let states = brake.braking(turn(XY, 0.0), XY * initial, cold(&mesh), 0.1, 30);
        let energy = energies(&states);
        assert!(energy.iter().all(|e| (e - energy[0]).abs() < 1e-12));
        for state in &states {
            let sheet = state.current.zip_with(&areas, |j, a| j * a.s());
            let flux = reconstruction.transpose().apply(&sheet);
            assert!(largest(&charge_balance(&mesh, &flux), Scalar3::new(0.0)) < 1e-9);
        }
        let speeds: Vec<f64> = states.iter().map(|s| speed(s.spin)).collect();
        assert!(speeds.iter().all(|&s| s > 0.0));
        assert!(speeds.windows(2).all(|w| w[1] < w[0]));
        assert!(states.last().expect("states").heat.sum().s() > 0.0);
    }
}

#[test]
fn free_rotation_converges_at_second_order_in_time() {
    let mesh = Mesh::disk(RADIUS, 4, 24);
    let conductivity = uniform(&mesh, (open() * 0.5 + X * (X | open())) * CONDUCTANCE);
    let brake = brake(&mesh, &conductivity, magnet(STRENGTH));
    let (duration, steps) = (2.0, 10);
    let finals: Vec<Motion> = [1, 2, 4]
        .iter()
        .map(|&refinement| {
            let count = steps * refinement;
            let states = brake.braking(
                turn(XY, 0.0),
                XY * 2.0,
                cold(&mesh),
                duration / count as f64,
                count,
            );
            states.last().expect("states").clone()
        })
        .collect();
    // Halving the step quarters the endpoint error in both speed and orientation.
    let spin_gap = |a: &Motion, b: &Motion| (a.spin - b.spin).norm();
    assert!(spin_gap(&finals[2], &finals[1]) < 0.35 * spin_gap(&finals[1], &finals[0]));
    assert!(turn_gap(&finals[2], &finals[1]) < 0.35 * turn_gap(&finals[1], &finals[0]));
}

/// The distance between two states' turns, as rotors.
fn turn_gap(a: &Motion, b: &Motion) -> f64 {
    (a.orientation.into_inner() - b.orientation.into_inner()).norm()
}

#[test]
fn material_heat_and_motion_converge_at_second_order_in_time() {
    let (speed, elapsed, steps) = (2.0, 1.6, 8);
    let mesh = Mesh::disk(RADIUS, 4, 24);
    let radial = mesh.face_centers().map(|c| c.normalized());
    let conductivity = radial.map(|r| (open() * 0.3 + r * (r | open()) * 1.4) * CONDUCTANCE);
    let brake = brake(&mesh, &conductivity, magnet(STRENGTH));
    let finals = [1, 2, 4].map(|refinement| {
        let count = steps * refinement;
        let step = elapsed / count as f64;
        let states = brake.braking(turn(XY, 0.0), XY * speed, cold(&mesh), step, count);
        states.last().expect("states").clone()
    });
    // Refining time resolves both the magnetic heating swept over material faces and the turn.
    let heat_gap = |a: &Motion, b: &Motion| {
        length(a.heat.iter().zip(b.heat.iter()).map(|(x, y)| x.s() - y.s()))
    };
    assert!(heat_gap(&finals[2], &finals[1]) < 0.4 * heat_gap(&finals[1], &finals[0]));
    assert!(turn_gap(&finals[2], &finals[1]) < 0.4 * turn_gap(&finals[1], &finals[0]));
}

#[test]
fn axisymmetric_materials_brake_in_batches() {
    let mesh = Mesh::disk(RADIUS, 6, 36);
    let finals = MATERIALS.map(|shares| {
        let conductivity = fibres(&mesh, shares);
        let states = brake(&mesh, &conductivity, magnet(STRENGTH)).braking(
            turn(XY, 0.0),
            XY * 2.0,
            cold(&mesh),
            0.1,
            40,
        );
        let energy = energies(&states);
        assert!(energy.iter().all(|e| (e - energy[0]).abs() < 1e-12));
        assert!(
            states
                .windows(2)
                .all(|w| speed(w[1].spin) < speed(w[0].spin))
        );
        speed(states.last().expect("states").spin)
    });
    // With the same mean conductance, the isotropic disc brakes hardest and circumferential
    // fibres least.
    assert!(finals[0] < finals[1] && finals[1] < finals[2]);
}

#[test]
fn moving_material_faces_gain_heat_and_preserve_total_energy() {
    let mesh = Mesh::disk(RADIUS, 4, 24);
    for (shares, (initial, angle)) in [[1.9, 0.1], [0.1, 1.9]]
        .into_iter()
        .zip([(2.0, 0.0), (1.0, 0.3)])
    {
        let conductivity = fibres(&mesh, shares);
        let states = brake(&mesh, &conductivity, magnet(STRENGTH)).braking(
            turn(XY, angle),
            XY * initial,
            cold(&mesh),
            0.1,
            20,
        );
        let energy = energies(&states);
        assert!(energy.iter().all(|e| (e - energy[0]).abs() < 1e-12));
        for state in &states {
            let turned = mesh.vertices.map(|v| state.orientation >> v);
            assert!(difference(&state.vertices, &turned) < 1e-12);
        }
        for pair in states.windows(2) {
            assert!(
                pair[0]
                    .heat
                    .iter()
                    .zip(pair[1].heat.iter())
                    .all(|(a, b)| b.s() - a.s() >= -1e-14)
            );
        }
        let (first, last) = (&states[0], states.last().expect("states"));
        assert!(last.heat.sum().s() > 0.0);
        assert!(size(&(&last.vertices - &first.vertices)) > RADIUS);
    }
}

#[test]
fn coasting_transports_nonuniform_heat_with_the_material() {
    let (step, steps) = (0.1, 12);
    let mesh = Mesh::disk(RADIUS, 4, 24);
    let conductivity = uniform(&mesh, (X * (X | open()) + Y * (Y | open())) * CONDUCTANCE);
    // A warmer side identifies material faces even when no current adds heat.
    let heat = mesh
        .triangle_areas()
        .zip_with(&mesh.face_centers(), |a, c| {
            a * (1.0 + (c | X).s() / RADIUS)
        });
    for (initial, angle) in [(2.0, 0.0), (-1.0, 0.3)] {
        let spin = XY * initial;
        let states = brake(&mesh, &conductivity, magnet(0.0)).braking(
            turn(XY, angle),
            spin,
            heat.clone(),
            step,
            steps,
        );
        let elapsed = step * steps as f64;
        let coasted = (spin * (-elapsed / 2.0)).exp();
        assert!(states.iter().all(|s| difference(&s.heat, &heat) < 1e-12));
        assert!(states.iter().all(|s| (s.spin - spin).norm() < 1e-12));
        let (first, last) = (&states[0], states.last().expect("states"));
        assert!(difference(&last.vertices, &first.vertices.map(|v| coasted >> v)) < 1e-12);
        // The heat-weighted position turns in the world while each face keeps its own heat.
        let centre = |state: &Motion| {
            let centres = mesh.with_vertices(state.vertices.clone()).face_centers();
            centres.zip_with(&heat, |c, h| c * h.s()).sum() * (1.0 / heat.sum().s())
        };
        let (start, end) = (centre(first), centre(last));
        assert!(end.max_abs_diff(&(coasted >> start)) < 1e-12);
        assert!((end - start).norm() > RADIUS / 4.0);
    }
}

#[test]
fn material_frame_heating_matches_a_solve_on_the_rotated_world_mesh() {
    let (speed, angle, step) = (2.0, 0.4, 0.1);
    let mesh = Mesh::disk(RADIUS, 4, 24);
    // numga's `exp(xz * 0.2)`: `xz = -e31`.
    let tilt = Bivector3::new(0.0, -0.2, 0.0).exp();
    let plane = tilt >> XY;
    let tilted = Magnet {
        centre: tilt >> (X * OFFSET),
        plane,
        ..magnet(STRENGTH)
    };
    let orientation = turn(plane, angle) * tilt;
    let spin = plane * speed;
    let radial = mesh.face_centers().map(|c| c.normalized());
    let conductivity = radial.map(|r| (open() * 0.1 + r * (r | open()) * 1.8) * CONDUCTANCE);
    let states =
        brake(&mesh, &conductivity, tilted).braking(orientation, spin, cold(&mesh), step, 1);

    // Solved anew in world coordinates, the magnet fixed as the mesh turns.
    let midpoint = (spin * (-step / 4.0)).exp() * orientation;
    let world = mesh.with_vertices(mesh.vertices.map(|v| midpoint >> v));
    let material = conductivity.map(|c| midpoint >> c.of(midpoint << open()));
    let response = solve(
        &world,
        &tilted.field(&world.edge_midpoints()),
        plane,
        &material,
    );
    let drag = (plane | response.torque).s();
    let midpoint_spin = spin * (1.0 / (1.0 + drag * step / (2.0 * INERTIA)));
    let gained = response.heating.scaled(step * midpoint_spin.norm_squared());
    let last = states.last().expect("states");
    assert!(difference(&last.heat, &gained) < 1e-12);
    assert!(last.spin.max_abs_diff(&(midpoint_spin * 2.0 - spin)) < 1e-12);
}

/// The scenario checks of numga's `main`: the steady currents turn the mechanical power into
/// heat; the braking discs keep their rotational energy plus heat, and the inductive discs that
/// with their magnetic energy, less the motor's work.
#[test]
fn the_scenarios_turn_power_into_heat() {
    use super::scenes;
    let (mesh, conductivity) = scenes::materials();
    let magnetic = scenes::magnet(scenes::FIELD_STRENGTH).field(&mesh.edge_midpoints());
    let spin = XY * scenes::ANGULAR_SPEED;
    for c in &conductivity {
        let response = solve(&mesh, &magnetic, spin, c);
        let heat = response.heating.sum().s();
        assert!(((spin | response.torque).s() - heat).abs() <= 1e-12 * heat);
        let energy = energies(&scenes::materials_braking(&mesh, c));
        assert!(energy.iter().all(|e| (e - energy[0]).abs() < 1e-13));
        let inductive = scenes::inductive_material(&mesh, c);
        let energy: Vec<f64> = inductive
            .iter()
            .map(|s| {
                let held = s.heat.sum().s() + s.magnetic_energy - s.ledger.motor_work;
                held + kinetic(scenes::INERTIA, s.motion.spin)
            })
            .collect();
        assert!(energy.iter().all(|e| (e - energy[0]).abs() < 1e-8));
    }
}

#[test]
fn a_frame_draws() {
    gax_numga_examples::app::assert_draws(super::draw, 1.0);
}
