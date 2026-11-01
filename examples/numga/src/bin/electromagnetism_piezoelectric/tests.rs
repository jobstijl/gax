//! numga's tests: the material's covariance under rigid motions, its stress-free response
//! against the measured PZT-5A data, the patch tests, reciprocity of actuation and sensing, the
//! floating electrode, reversed poling, the assembled loads as the variation of the enthalpy, and
//! convergence to the slender bimorph's response.

use super::piezo::*;
use super::scenes::*;
use crate::sta2d::{self, time};
use gax::field::Field;

const DEVICE_WIDTH: f64 = 5.0;
const VOLTAGE: f64 = 1e-4;
const FORCE: f64 = 1e-4;
/// The vacuum's permittivity, F/m.
const EPSILON_0: f64 = 8.854_187_812_8e-12;

/// numga's test device: a strip 10 long and 1 thick, 40 by 6 cells.
fn device() -> &'static Device {
    static DEVICE: std::sync::OnceLock<Device> = std::sync::OnceLock::new();
    DEVICE.get_or_init(|| Device::new(10.0, 1.0, 40, 6, DEVICE_WIDTH))
}

fn small(values: impl IntoIterator<Item = f64>, tolerance: f64) -> bool {
    values.into_iter().all(|v| v.abs() < tolerance)
}

/// A rigid motion of space, as numga's tests make them: a rotation `exp(0.27 xy)` or the
/// reflection in `x`, applied to planes, points and fields.
#[derive(Clone, Copy)]
enum Motion {
    Rotation,
    Reflection,
}

impl Motion {
    fn plane(self, p: Plane) -> Plane {
        let v = sta2d::Vector::new(p.e1(), p.e2(), 0.0);
        match self {
            Motion::Rotation => {
                (sta2d::Bivector::new(0.27, 0.0, 0.0).exp() >> v).cast::<sta2d::Spatial>()
            }
            Motion::Reflection => (sta2d::x_plane() >> v).cast::<sta2d::Spatial>(),
        }
    }

    fn point(self, p: Point) -> Point {
        let b = sta2d::intrinsic::Bivector::from(p);
        match self {
            Motion::Rotation => {
                (sta2d::Bivector::new(0.27, 0.0, 0.0).exp() >> b).cast::<sta2d::Temporal>()
            }
            Motion::Reflection => (sta2d::x_plane() >> b).cast::<sta2d::Temporal>(),
        }
    }

    fn electric(self, e: Electric) -> Electric {
        let b = sta2d::Bivector::from(e);
        match self {
            Motion::Rotation => {
                (sta2d::Bivector::new(0.27, 0.0, 0.0).exp() >> b).cast::<sta2d::Temporal>()
            }
            Motion::Reflection => (sta2d::x_plane() >> b).cast::<sta2d::Temporal>(),
        }
    }
}

#[test]
fn material_follows_its_directions_under_rigid_transformations() {
    let material = pzt5a([x(), y()]);
    for motion in [Motion::Rotation, Motion::Reflection] {
        let moved = pzt5a([x(), y()].map(|d| motion.plane(d)));
        let displacement = axis() * 0.12 - pole() * 0.3;
        let test_displacement = axis() * -0.2 + pole() * 0.1;
        let gradient = x() * 0.7 + y() * 0.2;
        let test_gradient = x() * -0.6 + y() * 0.8;
        let electric = Electric::new(0.3, 0.7);
        let work = material
            .elastic
            .of(displacement)
            .of(gradient)
            .of(test_displacement)
            .of(test_gradient);
        let moved_work = moved
            .elastic
            .of(motion.point(displacement))
            .of(motion.plane(gradient))
            .of(motion.point(test_displacement))
            .of(motion.plane(test_gradient));
        assert!((moved_work - work).s().abs() < 1e-13);
        let polarized = moved
            .polarization
            .of(motion.point(displacement))
            .of(motion.plane(gradient));
        let expected = motion.point(material.polarization.of(displacement).of(gradient));
        assert!(small((polarized - expected).c, 1e-13));
        let induced = moved.permittivity.of(motion.electric(electric));
        let expected = motion.point(material.permittivity.of(electric));
        assert!(small((induced - expected).c, 1e-13));
    }
}

#[test]
fn stress_free_response_matches_measured_pzt5a_data() {
    let material = pzt5a([x(), y()]);
    // (field, free extensions, free shear, free relative permittivity, direction): along the
    // poling and across it, Table A3's free strains and permittivity.
    let cases = [
        (
            Electric::new(0.0, 1.0),
            [-171e-12, 374e-12],
            0.0,
            1700.0,
            pole(),
        ),
        (Electric::new(1.0, 0.0), [0.0, 0.0], 584e-12, 1730.0, axis()),
    ];
    for (electric, extension, shear, free_permittivity, direction) in cases {
        let changes = [
            (axis(), x(), extension[0] * FIELD_UNIT),
            (pole(), y(), extension[1] * FIELD_UNIT),
            (axis(), y(), shear * FIELD_UNIT / 2.0),
            (pole(), x(), shear * FIELD_UNIT / 2.0),
        ];
        // The constitutive maps hold fixed-strain data; allowing these strains recovers the
        // measurements, to the rounding of the published constants.
        for (test_displacement, test_gradient) in [(axis(), x()), (pole(), y()), (axis(), y())] {
            let mechanical: f64 = changes
                .iter()
                .map(|&(d, g, amplitude)| {
                    material
                        .elastic
                        .of(d)
                        .of(g)
                        .of(test_displacement)
                        .of(test_gradient)
                        .s()
                        * amplitude
                })
                .sum();
            let polarized = material
                .polarization
                .of(test_displacement)
                .of(test_gradient);
            let electrical = (electric & (polarized & time())).s();
            assert!((mechanical - electrical).abs() < 1e-5);
        }
        let induced = changes
            .iter()
            .map(|&(d, g, amplitude)| material.polarization.of(d).of(g) * amplitude)
            .fold(material.permittivity.of(electric), |a, b| a + b);
        let measured = direction * (free_permittivity * EPSILON_0 / PERMITTIVITY_UNIT);
        assert!(small((induced - measured).c, 1e-5));
    }
}

#[test]
fn affine_fields_and_rigid_motion_are_reproduced() {
    let d = device();
    let (mesh, vertices) = (&d.strip.mesh, &d.strip.vertices);
    let translation = vertices.map(|_| axis() * 0.3 + pole() * 0.7);
    let turning = vertices.map(|v| {
        sta2d::intrinsic::Bivector::from(v)
            .commutator(sta2d::Bivector::new(0.2, 0.0, 0.0))
            .cast::<sta2d::Temporal>()
    });
    for displacement in [&translation, &turning] {
        let strains = strain(mesh, displacement, &d.gradients);
        assert!(small(
            strains.iter().flat_map(|s| s.c.into_iter().flatten()),
            1e-12
        ));
        assert!(small(
            d.blocks
                .mechanical
                .of(displacement)
                .iter()
                .flat_map(|f| f.c),
            1e-10
        ));
        assert!(small(
            d.blocks
                .coupling
                .adjugate()
                .of(displacement)
                .iter()
                .map(|q| q.c[0]),
            1e-10
        ));
    }
    let displacement = vertices.map(|v| {
        let [h, w] = coordinates(v);
        axis() * (0.2 * h + 0.07 * w) + pole() * (0.07 * h - 0.3 * w)
    });
    for s in strain(mesh, &displacement, &d.gradients) {
        assert!(((axis() & s.of(axis())).s() - 0.2).abs() < 1e-12);
        assert!(((pole() & s.of(pole())).s() + 0.3).abs() < 1e-12);
        assert!(((axis() & s.of(pole())).s() - 0.07).abs() < 1e-12);
    }
    let potential = vertices.map(|v| {
        let [h, w] = coordinates(v);
        Scalar::new(2.0 * h - 3.0 * w + 0.7)
    });
    let expected = sweep(x() * 2.0 - y() * 3.0) * -1.0;
    for e in electric_field(mesh, &potential, &d.gradients) {
        assert!(small((e - expected).c, 1e-11));
    }
    let constant = d
        .strip
        .electrode_potential(0.0)
        .map(|p| p + Scalar::new(1.0));
    assert!(small(
        d.blocks.coupling.of(&constant).iter().flat_map(|f| f.c),
        1e-11
    ));
    assert!(small(
        d.blocks.dielectric.of(&constant).iter().map(|q| q.c[0]),
        1e-11
    ));
}

/// The total of the charges on some vertices.
fn charge_on(charges: &Field<Amount>, vertices: &[usize]) -> f64 {
    vertices
        .iter()
        .map(|&v| (Scalar::new(1.0) & charges.site(v)).s())
        .sum()
}

#[test]
fn actuation_and_short_circuit_sensing_are_reciprocal() {
    let d = device();
    let (actuated, _, _) = d.solve(VOLTAGE, 0.0, false);
    let (_, _, sensed) = d.solve(0.0, FORCE, false);
    let motion_per_voltage = d.strip.tip_motion(&actuated) / VOLTAGE;
    let charge_per_force = charge_on(&sensed, &d.strip.top) / FORCE;
    assert!(motion_per_voltage.abs() > 1.0);
    assert!((motion_per_voltage - charge_per_force).abs() < 1e-9);
}

#[test]
fn floating_electrode_is_equipotential_and_conserves_charge() {
    let d = device();
    let s = &d.strip;
    let (displacement, potential, charges) = d.solve(0.0, FORCE, true);
    let voltage = s.top.iter().map(|&v| potential.site(v).s()).sum::<f64>() / s.top.len() as f64;
    assert!(voltage.abs() > 1e-6);
    assert!(small(
        s.top.iter().map(|&v| potential.site(v).s() - voltage),
        1e-12
    ));
    assert!(small(
        s.bottom.iter().map(|&v| potential.site(v).s()),
        1e-12
    ));
    assert!(charge_on(&charges, &s.top).abs() < 1e-11);
    let interior: Vec<usize> = (0..s.mesh.len(0))
        .filter(|v| !s.top.contains(v) && !s.bottom.contains(v))
        .collect();
    assert!(small(interior.iter().map(|&v| charges.site(v).c[0]), 1e-11));
    assert!(charge_on(&charges, &(0..s.mesh.len(0)).collect::<Vec<_>>()).abs() < 1e-11);

    // The clamp carries the applied force; every other node is in equilibrium.
    let load = s.tip_load(FORCE);
    let reaction =
        &(&d.blocks.mechanical.of(&displacement) + &d.blocks.coupling.of(&potential)) - &load;
    let free: Vec<usize> = (0..s.mesh.len(0))
        .filter(|v| !s.clamped.contains(v))
        .collect();
    assert!(small(free.iter().flat_map(|&v| reaction.site(v).c), 1e-11));
    assert!(small(
        s.clamped.iter().flat_map(|&v| displacement.site(v).c),
        1e-12
    ));
    assert!(small((reaction.sum() + load.sum()).c, 1e-11));
}

#[test]
fn reversing_poling_reverses_the_coupled_responses() {
    let d = device();
    let reversed: Vec<f64> = d.strip.poling.iter().map(|p| -p).collect();
    let blocks = assemble(
        &d.strip.mesh,
        &d.areas,
        &d.gradients,
        &d.material,
        &reversed,
        DEVICE_WIDTH,
    );
    let flipped = Device {
        strip: Strip {
            poling: reversed,
            ..strip(10.0, 1.0, 40, 6)
        },
        areas: d.areas.clone(),
        gradients: d.gradients.clone(),
        material: d.material,
        blocks,
    };
    let (u, p, q) = d.solve(VOLTAGE, 0.0, false);
    let (ru, rp, rq) = flipped.solve(VOLTAGE, 0.0, false);
    assert!(small(
        u.iter().zip(ru.iter()).flat_map(|(a, b)| (a + b).c),
        1e-11
    ));
    assert!(small(
        p.iter().zip(rp.iter()).map(|(a, b)| (a - b).s()),
        1e-11
    ));
    assert!(small(
        q.iter().zip(rq.iter()).map(|(a, b)| (a - b).c[0]),
        1e-11
    ));
    let (u, p, _) = d.solve(0.0, FORCE, true);
    let (ru, rp, _) = flipped.solve(0.0, FORCE, true);
    assert!(small(
        u.iter().zip(ru.iter()).flat_map(|(a, b)| (a - b).c),
        1e-11
    ));
    assert!(small(
        p.iter().zip(rp.iter()).map(|(a, b)| (a + b).s()),
        1e-11
    ));
}

/// The electric enthalpy of a strip in the given state: per triangle, the stored elastic energy
/// less the field's work on the piezoelectric and the induced polarization, weighted by area and
/// width. numga's `trace(1, 2)` pairs a direction slot with a plane slot through `∨`: gax's
/// `trace_inputs`, each trace taking a whole deformation map into its displacement–gradient
/// pair.
fn enthalpy(
    d: &Device,
    poling: &[f64],
    displacement: &Field<Point>,
    potential: &Field<Scalar>,
) -> f64 {
    let strains = strain(&d.strip.mesh, displacement, &d.gradients);
    let fields = electric_field(&d.strip.mesh, potential, &d.gradients);
    let m = &d.material;
    (0..d.strip.mesh.len(2))
        .map(|f| {
            let deformation = strains[f].hodge();
            let stored = m
                .elastic
                .of(deformation)
                .trace_inputs()
                .of(deformation)
                .trace_inputs()
                .s();
            let polarized = m.polarization.of(deformation).trace_inputs() * poling[f];
            let field = fields[f];
            let induced = m.permittivity.of(field);
            let density = stored / 2.0
                - (field & (polarized & time())).s()
                - (field & (induced & time())).s() / 2.0;
            DEVICE_WIDTH * d.areas.site(f).s() * density
        })
        .sum()
}

#[test]
fn assembled_loads_are_the_variation_of_material_enthalpy() {
    let d = Device::new(10.0, 1.0, 16, 4, DEVICE_WIDTH);
    let vertices = &d.strip.vertices;
    let at = |f: &dyn Fn(f64, f64) -> f64| {
        vertices.map(|v| {
            let [h, w] = coordinates(v);
            Scalar::new(f(h, w))
        })
    };
    let displacement = vertices.map(|v| {
        let [h, w] = coordinates(v);
        axis() * (0.001 * h * h + 0.002 * h * w) + pole() * (0.003 * h * w - 0.004 * w * w)
    });
    let potential = at(&|h, w| 0.002 * h * w + 0.003 * h * h - 0.004 * w);
    let movement = vertices.map(|v| {
        let [h, w] = coordinates(v);
        axis() * (0.3 * h * w) + pole() * (0.2 * h * h + 0.4 * w * w)
    });
    let voltage = at(&|h, w| 0.3 * h * h - 0.7 * h * w + 0.2 * w * w);
    let step = 1e-4;
    let shifted = |k: f64| {
        let u = &displacement + &movement.map(|m| m * k);
        let p = &potential + &voltage.map(|v| v * k);
        enthalpy(&d, &d.strip.poling, &u, &p)
    };
    let difference = (shifted(step) - shifted(-step)) / (2.0 * step);
    let b = &d.blocks;
    let forces = &b.mechanical.of(&displacement) + &b.coupling.of(&potential);
    let charges = &b.coupling.adjugate().of(&displacement) + &b.dielectric.of(&potential);
    let work: f64 = forces
        .iter()
        .zip(movement.iter())
        .map(|(f, m)| (f & m).s())
        .sum::<f64>()
        + charges
            .iter()
            .zip(voltage.iter())
            .map(|(q, v)| (q & v).s())
            .sum::<f64>();
    assert!(work.abs() > 1e-4, "{work}");
    assert!((difference - work).abs() < 1e-10, "{difference} {work}");
}

#[test]
fn refinement_approaches_the_slender_bimorph_response() {
    let (length, thickness) = (10.0, 1.0);
    // With no transverse stress, these axial coefficients; constant electric displacement
    // through the thickness includes the field the bending itself induces.
    let (compliance, free_coupling) = (16.4e-12, -171e-12);
    let free_permittivity = 1700.0 * EPSILON_0;
    let stiffness = 1.0 / compliance / STRESS_UNIT;
    let piezoelectric = free_coupling / compliance * FIELD_UNIT / STRESS_UNIT;
    let permittivity =
        (free_permittivity - free_coupling * free_coupling / compliance) / PERMITTIVITY_UNIT;
    let beam = 1.5 * piezoelectric * permittivity
        / (stiffness * permittivity + piezoelectric * piezoelectric / 4.0)
        * (length / thickness)
        * (length / thickness);
    let errors: Vec<f64> = [(20, 4), (40, 8), (80, 16)]
        .into_iter()
        .map(|(n_length, n_thickness)| {
            let d = Device::new(length, thickness, n_length, n_thickness, DEVICE_WIDTH);
            let (displacement, _, _) = d.solve(VOLTAGE, 0.0, false);
            (d.strip.tip_motion(&displacement) / VOLTAGE / beam - 1.0).abs()
        })
        .collect();
    assert!(
        errors[1] < 0.7 * errors[0] && errors[2] < 0.7 * errors[1],
        "{errors:?}"
    );
    assert!(errors[2] < 0.08, "{errors:?}");
}

#[test]
fn the_scenario_bends_and_senses() {
    // The scenario's device at its size: the actuator bends, and the sensor's floating
    // electrode reads a voltage.
    let d = Device::new(LENGTH, THICKNESS, N_LENGTH, N_THICKNESS, WIDTH);
    let (actuated, _, _) = d.solve(APPLIED_VOLTAGE, 0.0, false);
    let (_, sensed, _) = d.solve(0.0, APPLIED_FORCE, true);
    assert!(d.strip.tip_motion(&actuated).abs() > 1e-3);
    assert!(sensed.site(d.strip.top[0]).s().abs() > 0.0);
}

#[test]
fn frames_draw() {
    let shown = super::shown();
    for t in [0.0, 1.1] {
        gax_numga_examples::app::assert_draws_with(super::draw, &shown, t);
    }
}
