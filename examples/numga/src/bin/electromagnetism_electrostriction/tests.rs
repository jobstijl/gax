//! numga's tests of the electrostriction example: variational polarization, the coupled
//! mechanical derivatives, and stable electrostriction; and the chain rule's derivatives
//! against those through the solve.

use super::lattice::*;
use super::pairs::is_self_adjoint;
use gax::ApproxEq;
use gax::Real;
use gax::field::{Field, SparseMap, gradient, hessian};
use gax::jet::Dual;
use gax::vga2d::{Pseudoscalar, Scalar};
use gax_numga_examples::rng::{Draw, rng};
use std::sync::OnceLock;

const ROUND_OFF: f64 = 1e-12;
/// Newton steps converge quadratically; seven reach round-off at the strongest field.
const CONVERGED_STEPS: usize = 7;

/// The lattice relaxed in no field, in the strongest field and in its reverse: numga's module
/// fixture, shared by the tests.
struct Loading {
    lattice: Lattice,
    fields: [V; 3],
    positions: [Field<V>; 3],
}

fn loading() -> &'static Loading {
    static LOADING: OnceLock<Loading> = OnceLock::new();
    LOADING.get_or_init(|| {
        let lattice = lattice();
        let fields = [0.0, FIELD, -FIELD].map(|s| X * s);
        let positions = fields.map(|f| lattice.equilibrium(f, CONVERGED_STEPS));
        Loading {
            lattice,
            fields,
            positions,
        }
    })
}

/// The largest coefficient of a field of vectors.
fn largest(field: &Field<V>) -> f64 {
    field
        .iter()
        .flat_map(|v| v.c)
        .fold(0.0, |m, x| m.max(x.abs()))
}

/// numga's `assert_allclose`: `|actual - desired| <= atol + rtol |desired|`.
fn close(actual: f64, desired: f64, atol: f64, rtol: f64) -> bool {
    (actual - desired).abs() <= atol + rtol * desired.abs()
}

/// A field of linear forms bound to a displacement field, summed over the sites.
fn bound(gradient: &Field<Form>, step: &Field<V>) -> f64 {
    gradient
        .iter()
        .zip(step.iter())
        .map(|(g, d)| g.of(d).s())
        .sum()
}

/// The curvature as a map on displacement fields.
fn stiffness(curvature: &SparseMap<Form2>) -> FieldMap<f64> {
    curvature.map_cells(Scalar::as_map)
}

/// The identity on fields over `n` sites.
fn identity(n: usize) -> FieldMap<f64> {
    SparseMap::from_diagonal(&Field::from_fn(n, |_| V::slot()))
}

/// The lowest eigenvalue of a symmetric field map.
fn lowest(map: &FieldMap<f64>) -> f64 {
    map.eigh(&identity(map.shape().0), 1).0[0]
}

/// The derivatives through the dipoles' solve: gax's field derivatives of the energy.
fn through_the_solve(lattice: &Lattice, positions: &Field<V>, applied: V) -> Derivatives {
    let (_, gradient, curvature) = hessian(|x| lattice.energy(x, applied), positions);
    Derivatives {
        gradient,
        curvature,
    }
}

/// A field of `f64` values as one of coefficients `T`.
fn lifted<T: Real>(field: &Field<V>) -> Field<Vt<T>> {
    field.map(lift::<T>)
}

/// A random displacement field, `scale` large.
fn displacement(seed: u64, n: usize, scale: f64) -> Field<V> {
    let mut draws = rng(seed);
    Field::from_fn(n, |_| V::new(draws.normal(), draws.normal()) * scale)
}

#[test]
fn the_chain_rule_gives_the_derivatives_through_the_solve() {
    let lattice = lattice();
    // A shape off its equilibrium, in a field off the lattice's axes.
    let positions = &lattice.rest + &displacement(5, lattice.particles(), 0.05);
    let applied = V::from_angle(0.4) * FIELD;
    let chained = lattice.derivatives(&positions, applied);
    let solved = through_the_solve(&lattice, &positions, applied);
    let difference = |a: &Field<Form>, b: &Field<Form>| {
        a.iter()
            .zip(b.iter())
            .map(|(f, g)| f.max_abs_diff(&g))
            .fold(0.0, f64::max)
    };
    assert!(difference(&chained.gradient, &solved.gradient) < 1e-11);
    for seed in 0..3 {
        let step = displacement(seed, lattice.particles(), 1.0);
        let a = stiffness(&chained.curvature).of(&step);
        let b = stiffness(&solved.curvature).of(&step);
        assert!(largest(&(&a - &b)) < 1e-10);
    }
}

#[test]
fn polarization_minimizes_the_full_electrical_energy() {
    let Loading {
        lattice,
        fields,
        positions,
    } = loading();
    for (field, position) in fields.iter().zip(positions) {
        let response = lattice.response(position);
        let dipoles = lattice.dipoles(position, *field);
        let applied = Field::from_fn(lattice.particles(), |_| *field);
        assert!(largest(&(&response.of(&dipoles) - &applied)) < ROUND_OFF);
        assert!(is_self_adjoint(&response, ROUND_OFF));
        // The electrical energy of any dipoles at this shape, least at the solved ones.
        let electrical = |p: &Field<Vt<_>>| held_energy(lattice, &lifted(position), p, *field);
        let (_, stationary) = gradient(electrical, &dipoles);
        assert!(
            stationary
                .iter()
                .all(|g| g.max_abs_diff(&Form::zero()) < ROUND_OFF)
        );
        let reduced = lattice.energy(position, *field) - lattice.elastic_energy(position);
        let held = held_energy(lattice, position, &dipoles, *field);
        assert!((held - reduced).abs() < ROUND_OFF);
        assert!(lowest(&response) > 20.0);
    }
}

/// The electrical energy of dipoles held at `dipoles` at a shape: their cost and mutual
/// interaction, half their pairing with the response's image of them, less their pairing with
/// the applied field.
fn held_energy<T: Real>(
    lattice: &Lattice,
    shape: &Field<Vt<T>>,
    dipoles: &Field<Vt<T>>,
    field: V,
) -> T {
    let image = lattice.response(shape).of(dipoles);
    let field = lift::<T>(field);
    let half = T::from_f64(0.5);
    dipoles
        .iter()
        .zip(image.iter())
        .fold(T::zero(), |e, (p, r)| {
            e + (p | r).s() * half - (field | p).s()
        })
}

/// numga's test of the typed derivatives, at the strongest field: a displacement and another,
/// both small, and the central differences' increment.
struct Probe<'a> {
    lattice: &'a Lattice,
    position: &'a Field<V>,
    field: V,
    step: Field<V>,
    other: Field<V>,
    increment: f64,
}

impl Probe<'_> {
    fn new() -> Probe<'static> {
        let Loading {
            lattice,
            fields,
            positions,
        } = loading();
        let n = lattice.particles();
        Probe {
            lattice,
            position: &positions[1],
            field: fields[1],
            step: displacement(2, n, 0.01),
            other: displacement(3, n, 0.01),
            increment: 1e-3,
        }
    }

    /// The shape moved by the increment along the displacement, forward and back.
    fn shifted(&self) -> [Field<V>; 2] {
        [1.0, -1.0].map(|sign| self.position + &self.step.scaled(sign * self.increment))
    }

    /// The central difference of a field along the displacement.
    fn change(&self, f: impl Fn(&Field<V>) -> Field<V>) -> Field<V> {
        let [upper, lower] = self.shifted();
        (&f(&upper) - &f(&lower)).scaled(0.5 / self.increment)
    }
}

/// numga's `assert_allclose` of two fields of vectors, with its central-difference bounds.
fn all_close(actual: &Field<V>, desired: &Field<V>) -> bool {
    let pairs = actual.iter().zip(desired.iter());
    pairs
        .flat_map(|(a, d)| a.c.into_iter().zip(d.c))
        .all(|(a, d)| close(a, d, 3e-9, 1e-6))
}

#[test]
fn typed_derivatives_predict_changes_of_energy_and_response() {
    let probe = Probe::new();
    let (lattice, field, step) = (probe.lattice, probe.field, &probe.step);
    let Derivatives {
        gradient,
        curvature,
    } = lattice.derivatives(probe.position, field);
    // These bounds test central-difference accuracy rather than round-off.
    let [upper, lower] = probe.shifted();
    let energy = |x: &Field<V>| lattice.energy(x, field);
    let energy_change = (energy(&upper) - energy(&lower)) / (2.0 * probe.increment);
    assert!(close(bound(&gradient, step), energy_change, 3e-9, 1e-6));
    let k = stiffness(&curvature);
    let symmetric = step.dot(&k.of(&probe.other)) - probe.other.dot(&k.of(step));
    assert!(symmetric.abs() < ROUND_OFF);

    // The response's change with the shape, bound to the dipoles and a displacement: along the
    // displacement as the derivatives' one direction.
    let dipoles = lattice.dipoles(probe.position, field);
    let moving: Field<Vt<Dual<f64, 1>>> = Field::from_fn(lattice.particles(), |i| {
        let (p, d) = (probe.position.site(i), step.site(i));
        Vt::new(Dual::new(p.c[0], [d.c[0]]), Dual::new(p.c[1], [d.c[1]]))
    });
    let image = lattice
        .response(&moving)
        .of(&lifted::<Dual<f64, 1>>(&dipoles));
    let predicted = image.map(|v| v.map_coefs(|d| d.du[0]));
    let measured = probe.change(|x| lattice.response(x).of(&dipoles));
    assert!(all_close(&predicted, &measured));
}

/// numga's check that the curvature bound to the displacement is the gradient's change.
#[test]
fn the_curvature_predicts_the_change_of_the_gradient() {
    let probe = Probe::new();
    let (lattice, field) = (probe.lattice, probe.field);
    let curvature = lattice.derivatives(probe.position, field).curvature;
    let predicted = stiffness(&curvature).of(&probe.step);
    let measured = probe.change(|x| lattice.derivatives(x, field).gradient.map(through_metric));
    assert!(all_close(&predicted, &measured));
}

#[test]
fn dipole_readjustment_changes_stiffness_but_not_the_stationary_first_derivative() {
    let Loading {
        lattice,
        fields,
        positions,
    } = loading();
    let (position, field) = (&positions[1], fields[1]);
    let dipoles = lattice.dipoles(position, field);
    // The energy with the dipoles held at their equilibrium for this shape.
    let (_, relaxed) = gradient(|x| lattice.energy(x, field), position);
    let (_, fixed) = gradient(
        |x| lattice.elastic_energy(x) + held_energy(lattice, x, &lifted(&dipoles), field),
        position,
    );
    for (a, b) in relaxed.iter().zip(fixed.iter()) {
        assert!(a.max_abs_diff(&b) < ROUND_OFF);
    }
    let (_, _, full) = hessian(|x| lattice.energy(x, field), position);
    let (_, _, frozen) = hessian(
        |x| lattice.elastic_energy(x) + held_energy(lattice, x, &lifted(&dipoles), field),
        position,
    );
    let compression = position.map(|p| -X * (X | p).s());
    let pair =
        |curvature: &SparseMap<Form2>| compression.dot(&stiffness(curvature).of(&compression));
    // Allowing an internal degree of freedom to relax softens the mechanical response.
    assert!(pair(&frozen) - pair(&full) > 0.1);
}

#[test]
fn mechanical_equilibria_are_stable_and_even_in_the_field() {
    let Loading {
        lattice,
        fields,
        positions,
    } = loading();
    for (field, position) in fields.iter().zip(positions) {
        let solved = through_the_solve(lattice, position, *field);
        assert!(
            solved
                .gradient
                .iter()
                .all(|g| g.max_abs_diff(&Form::zero()) < ROUND_OFF)
        );
        assert!(lowest(&stiffness(&solved.curvature)) > 0.05);
    }
    assert!(largest(&(&positions[0] - &lattice.rest)) < ROUND_OFF);
    assert!(largest(&(&positions[1] - &positions[2])) < ROUND_OFF);
    let dipoles = [1, 2].map(|k| lattice.dipoles(&positions[k], fields[k]));
    assert!(largest(&(&dipoles[0] + &dipoles[1])) < ROUND_OFF);
    assert!(strain(&positions[1], &lattice.rest, X) < -0.05);
}

#[test]
fn rotating_the_lattice_and_field_carries_the_whole_response() {
    let Loading {
        lattice,
        fields,
        positions,
    } = loading();
    let turn = Pseudoscalar::new(-0.37).exp();
    let turned = Lattice {
        rest: lattice.rest.map(|p| turn >> p),
        ..lattice.clone()
    };
    for (field, position) in fields.iter().zip(positions) {
        let moved = position.map(|p| turn >> p);
        let change = turned.energy(&moved, turn >> *field) - lattice.energy(position, *field);
        assert!(change.abs() < ROUND_OFF);
        let dipoles = lattice.dipoles(position, *field).map(|p| turn >> p);
        let rotated = turned.dipoles(&moved, turn >> *field);
        assert!(largest(&(&rotated - &dipoles)) < ROUND_OFF);
    }
}

/// numga's scenario checks: the relaxation of numga's four Newton steps has converged at the
/// strongest field, the unloaded lattice rests at its reference sites, and the lattice contracts
/// along the field; and numga's numbers at the strongest field.
#[test]
fn the_loading_matches_numga() {
    let Loading {
        lattice,
        fields,
        positions,
    } = loading();
    let quick = lattice.equilibrium(fields[1], NEWTON_STEPS);
    let (_, residual) = gradient(|x| lattice.energy(x, fields[1]), &quick);
    assert!(
        residual
            .iter()
            .all(|g| g.max_abs_diff(&Form::zero()) < 1e-3)
    );
    assert!(largest(&(&lattice.equilibrium(X * 0.0, NEWTON_STEPS) - &lattice.rest)) < ROUND_OFF);
    assert!(strain(&quick, &lattice.rest, X) < 0.0);
    // numga: strains -0.0752077 along and 0.0017998 across, energy -3.4760410.
    assert!((strain(&positions[1], &lattice.rest, X) + 0.075_207_671_507).abs() < 1e-9);
    assert!((strain(&positions[1], &lattice.rest, Y) - 0.001_799_822_048).abs() < 1e-9);
    assert!((lattice.energy(&positions[1], fields[1]) + 3.476_040_95).abs() < 1e-7);
}

#[test]
fn a_frame_draws() {
    gax_numga_examples::app::assert_draws(super::draw, 1.0);
}
