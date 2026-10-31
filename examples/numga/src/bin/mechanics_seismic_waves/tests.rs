//! numga's tests of the seismic waves: affine material response, rigid motions, the liquid's
//! missing shear stiffness and its viscosity, and free elastic time evolution; and the frames.

use super::seismic::*;
use gax::field::Field;
use gax_numga_examples::testing::{close, near};

/// The unit displacements along the axes (numga's `mv.x.dual()`, `mv.y.dual()`).
fn ex() -> Point {
    point(1.0, 0.0)
}
fn ey() -> Point {
    point(0.0, 1.0)
}

/// numga's fixture: a coarse disk whose inner faces are denser and stiffer.
fn disk_operators() -> (Operators, Vec<f64>, Vec<f64>) {
    let mesh = disk(&[0.6, 1.0], 6);
    let inner: Vec<bool> = centres(&mesh)
        .iter()
        .map(|c| c.norm_squared() < 0.6 * 0.6)
        .collect();
    let pick =
        |a: f64, b: f64| -> Vec<f64> { inner.iter().map(|&i| if i { a } else { b }).collect() };
    let (density, lambda, mu) = (pick(2.0, 1.0), pick(3.0, 2.0), pick(1.4, 1.0));
    let material: Vec<Elastic> = lambda
        .iter()
        .zip(&mu)
        .map(|(l, m)| constitutive(*l, *m))
        .collect();
    let viscosity = vec![viscous(0.0); material.len()];
    (
        Operators::new(mesh, &material, &viscosity, &density),
        lambda,
        mu,
    )
}

/// A uniform disk of one material.
fn uniform(lambda: f64, mu: f64, viscosity: f64, density: f64) -> Operators {
    let mesh = disk(&[1.0], 5);
    let faces = mesh.faces.len();
    Operators::new(
        mesh,
        &vec![constitutive(lambda, mu); faces],
        &vec![viscous(viscosity); faces],
        &vec![density; faces],
    )
}

fn all_zero<V: gax::Extensor<Slots = (), Coef = f64>>(field: &Field<V>) -> bool {
    field
        .iter()
        .all(|v| v.coeffs().as_ref().iter().all(|c| c.abs() <= 1e-12))
}

/// Each vertex's coordinates.
fn xy(ops: &Operators) -> Vec<[f64; 2]> {
    ops.vertices().iter().map(coordinates).collect()
}

/// A displacement field from each vertex's coordinates.
fn by_coordinates(ops: &Operators, f: impl Fn(f64, f64) -> Point) -> Field<Point> {
    xy(ops).into_iter().map(|[x, y]| f(x, y)).collect()
}

#[test]
fn rigid_motions_do_no_elastic_work() {
    let (ops, _, _) = disk_operators();
    let translation = ops.vertices().map(|_| ex() * 0.3 - ey() * 0.2);
    let rotation = ops.vertices().map(|v| v.commutator(Bivector::new(0.2)));
    for displacement in [translation, rotation] {
        assert!(all_zero(&ops.stiffness.of(&displacement)));
    }
}

#[test]
fn affine_strain_recovers_heterogeneous_energy() {
    let (ops, lambda, mu) = disk_operators();
    let (longitudinal, transverse, shear) = (0.12, -0.04, 0.035);
    let displacement = by_coordinates(&ops, |x, y| {
        ex() * (longitudinal * x + shear * y) + ey() * (shear * x + transverse * y)
    });
    let density = |f: usize| {
        let dilation = longitudinal + transverse;
        lambda[f] * dilation * dilation / 2.0
            + mu[f] * (longitudinal * longitudinal + transverse * transverse + 2.0 * shear * shear)
    };
    let expected: f64 = (0..lambda.len())
        .map(|f| density(f) * ops.areas.site(f).s())
        .sum();
    let rest = Field::<Point>::zeros(displacement.len());
    let actual = ops.energy(&displacement, &rest);
    assert!(
        (actual - expected).abs() <= 1e-12 + 1e-11 * expected.abs(),
        "{actual} {expected}"
    );

    let (dilation, vorticity) = ops.reconstruct(&displacement);
    assert!(
        dilation
            .iter()
            .all(|d| close(d.s(), longitudinal + transverse, 1e-12))
    );
    assert!(vorticity.iter().all(|w| close(w.s(), 0.0, 1e-12)));
    assert!(near(
        ops.stiffness.of(&displacement).sum(),
        Force::zero(),
        1e-12
    ));

    let spinning = by_coordinates(&ops, |x, y| ey() * x - ex() * y);
    let (dilation, vorticity) = ops.reconstruct(&spinning);
    assert!(dilation.iter().all(|d| close(d.s(), 0.0, 1e-12)));
    assert!(vorticity.iter().all(|w| close(w.s(), 2.0, 1e-12)));
}

#[test]
fn free_dynamics_preserves_momentum_with_bounded_energy() {
    let (ops, _, _) = disk_operators();
    let centre = point(-0.4, 0.2);
    let velocity = ops
        .vertices()
        .map(|v| ex() * (0.02 * (-(v - centre).norm_squared() / 0.08).exp()));
    let total: f64 = ops.masses.iter().map(|m| m.s()).sum();
    let drift = ops.momentum(&velocity) * (1.0 / total);
    let velocity = velocity.map(|v| v - drift);
    let displacement = Field::<Point>::zeros(velocity.len());
    let initial_energy = ops.energy(&displacement, &velocity);
    let initial_momentum = ops.momentum(&velocity);
    let states = evolve(&ops, displacement, velocity, 0.0, 0.002, 20, 20);
    let mut worst: f64 = 0.0;
    for (d, v) in &states {
        worst = worst.max((ops.energy(d, v) / initial_energy - 1.0).abs());
        assert!(near(ops.momentum(v), initial_momentum, 1e-12));
    }
    assert!(worst < 0.002, "{worst}");
    let last = &states[states.len() - 1].0;
    assert!(last.iter().map(|d| d.norm()).fold(0.0, f64::max) > 1e-4);
}

#[test]
fn concentric_disk_has_oriented_faces_in_plane_and_space() {
    // In the plane: points are intrinsic vectors, a face's patch an intrinsic scalar.
    let mesh = disk(&[0.6, 1.0], 5);
    let [a, b, c] = mesh.corners();
    let unit = Bivector::new(1.0);
    for ((a, b), c) in a.iter().zip(b.iter()).zip(c.iter()) {
        assert!((unit & ((b - a) & (c - a))).s() > 0.0);
    }
    assert_eq!(coordinates(mesh.vertices.site(0)), [0.0, 0.0]);
    // In space: points are intrinsic bivectors, a face's patch an intrinsic vector.
    let space = gax_numga_examples::mesh::Mesh::concentric_disk(&[0.6, 1.0], 5);
    let [a, b, c] = space.corners().map(|f| f.map(|v| v.hodge()));
    let unit = gax::vga3d::Vector::new(1.0, 0.0, 0.0) ^ gax::vga3d::Vector::new(0.0, 1.0, 0.0);
    for ((a, b), c) in a.iter().zip(b.iter()).zip(c.iter()) {
        assert!((unit & ((b - a) & (c - a))).s() > 0.0);
    }
    assert_eq!(space.vertices.site(0).c, [0.0; 3]);
}

#[test]
fn inviscid_fluid_has_no_shear_restoring_force() {
    let ops = uniform(3.0, 0.0, 0.0, 2.0);
    let shearing = by_coordinates(&ops, |x, y| (ex() * y + ey() * x) * 0.1);
    let rest = Field::<Point>::zeros(shearing.len());
    assert!(all_zero(&ops.stiffness.of(&shearing)));
    assert!(close(ops.energy(&shearing, &rest), 0.0, 1e-12));
    let compression = ops.vertices().scaled(-0.1);
    assert!(ops.energy(&compression, &rest) > 0.0);
}

#[test]
fn viscous_liquid_drags_shear_but_not_rigid_motion() {
    let ops = uniform(3.0, 0.0, 0.1, 2.0);
    let translation = ops.vertices().map(|_| ex() * 0.3 - ey() * 0.2);
    let spin = ops.vertices().map(|v| v.commutator(Bivector::new(0.2)));
    for rigid in [translation, spin] {
        assert!(all_zero(&ops.damping.of(&rigid)));
    }
    // A shearing flow is dragged, and the drag takes energy at a positive rate.
    let shearing = by_coordinates(&ops, |x, y| (ex() * y + ey() * x) * 0.1);
    let rate: f64 = ops
        .damping
        .of(&shearing)
        .iter()
        .zip(shearing.iter())
        .map(|(f, v)| (f & v).s())
        .sum();
    assert!(rate > 0.0, "{rate}");
}

/// The ball's lowest modes, on a coarse mesh: its three rigid motions, then modes orthonormal
/// in the masses, some of them in pairs of one frequency (not a numga test; its scenario
/// computes them).
#[test]
fn the_ball_rings_in_pairs_after_its_rigid_motions() {
    let ball = ball(10);
    let (values, shapes) = ball.normal_modes(RIGID_MOTIONS + 12);
    let scale = values[values.len() - 1];
    assert!(
        values[..RIGID_MOTIONS]
            .iter()
            .all(|v| v.abs() < 1e-8 * scale),
        "{values:?}"
    );
    assert!(values[RIGID_MOTIONS] > 1e-3 * scale);
    let mass = |a: &Field<Point>, b: &Field<Point>| -> f64 {
        ball.masses
            .iter()
            .zip(a.iter().zip(b.iter()))
            .map(|(m, (x, y))| m.s() * (x | y).s())
            .sum()
    };
    for (i, a) in shapes.iter().enumerate() {
        for (j, b) in shapes.iter().enumerate() {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(close(mass(a, b), expected, 1e-8), "{i} {j}");
        }
    }
    assert!(!paired(&values[RIGID_MOTIONS..]).is_empty());
}

#[test]
fn a_frame_draws() {
    let scenes = super::scenes_at(12);
    for t in [0.2, 1.0, 6.0] {
        gax_numga_examples::app::assert_draws_with(super::draw, &scenes, t);
    }
}
