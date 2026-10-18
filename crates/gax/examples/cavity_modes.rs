//! The modes of a square cavity: Maxwell's equations in spacetime algebra (`gax::calculus`), and
//! their frequencies from discrete exterior calculus (`gax::dec`).
//!
//! A transverse-electric mode of a square cavity of side 1 has the magnetic field
//! `B_z = cos(k_x x) cos(k_y y) cos(ω t)` and the electric field in the plane, with
//! `k = (mπ, nπ)` and `ω = |k|`. In spacetime algebra the field is one bivector, `F = E + I B`, and
//! Maxwell's four equations are `∇F = 0` in vacuum: checked here with `gax::calculus::nabla` at a
//! few events.
//!
//! The frequencies squared are the eigenvalues of the Laplacian on `B_z` with the cavity's
//! (Neumann) boundary, `ω² = π²(m² + n²)`. The discrete Laplacian on a triangulated square gives
//! them to second order, with the diagonal star and with the Galerkin one: the error falls about
//! fourfold each time the mesh is refined.
//!
//! Run with `cargo run --example cavity_modes`.

use gax::Real;
use gax::calculus::nabla;
use gax::dec::Geometry;
use gax::field::Field;
use gax::mesh::Mesh;
use std::f64::consts::PI;

type S = gax::vga2d::Scalar<(), f64>;

/// The field of the `(m, n)` mode at an event `t e₀ + x e₁ + y e₂ + z e₃`: `E + I B`, with
/// `σᵢ = eᵢ e₀` the relative vectors.
fn mode<T: Real>(m: f64, n: f64, event: gax::sta::Vector<(), T>) -> gax::sta::Bivector<(), T> {
    use gax::sta::{Bivector, Pseudoscalar};
    let (kx, ky) = (T::from_f64(m * PI), T::from_f64(n * PI));
    let w = T::from_f64(PI * (m * m + n * n).sqrt());
    let (t, x, y) = (event.c[0], event.c[1], event.c[2]);
    let (cx, sx, cy, sy) = (
        (kx * x).cos(),
        (kx * x).sin(),
        (ky * y).cos(),
        (ky * y).sin(),
    );
    let (ct, st) = ((w * t).cos(), (w * t).sin());
    let b_z = cx * cy * ct;
    let e_x = -(ky / w) * cx * sy * st;
    let e_y = (kx / w) * sx * cy * st;
    let zero = T::zero();
    // Bivector blades: [e10, e20, e30, e23, e31, e12]; σ₁ = e10, σ₂ = e20, σ₃ = e30.
    let e = Bivector::new(e_x, e_y, zero, zero, zero, zero);
    let b = Bivector::new(zero, zero, b_z, zero, zero, zero);
    e + (Pseudoscalar::new(T::one()) * b)
}

/// The unit square cut into `n × n` squares, each into two right triangles.
fn square(n: usize) -> Mesh<gax::vga2d::Vector<(), f64>, 3> {
    let id = |i: usize, j: usize| i * (n + 1) + j;
    let h = 1.0 / n as f64;
    let vertices: Field<_> = (0..=n)
        .flat_map(|i| (0..=n).map(move |j| gax::vga2d::Vector::new(i as f64 * h, j as f64 * h)))
        .collect();
    let faces: Vec<[usize; 3]> = (0..n)
        .flat_map(|i| {
            (0..n).flat_map(move |j| {
                [
                    [id(i, j), id(i + 1, j), id(i + 1, j + 1)],
                    [id(i, j), id(i + 1, j + 1), id(i, j + 1)],
                ]
            })
        })
        .collect();
    Mesh::new(vertices, &faces).expect("a valid complex")
}

fn main() {
    // Maxwell's equations hold for the analytic modes.
    for (m, n) in [(1.0, 0.0), (1.0, 1.0), (2.0, 1.0)] {
        for event in [[0.3, 0.2, 0.7], [1.1, 0.5, 0.25], [0.05, 0.9, 0.4]] {
            let x = gax::sta::Vector::<(), f64>::new(event[0], event[1], event[2], 0.0);
            let (_, maxwell) = nabla(|x| mode(m, n, x), x);
            let largest = maxwell.c.iter().fold(0.0f64, |a, c| a.max(c.abs()));
            assert!(
                largest < 1e-12,
                "∇F = 0 for the ({m}, {n}) mode: {largest:e}"
            );
        }
    }
    println!("∇F = 0: the modes satisfy Maxwell's equations");

    // Their frequencies from the discrete Laplacian.
    let exact: Vec<f64> = [0.0, 1.0, 1.0, 2.0, 4.0, 4.0, 5.0, 5.0]
        .iter()
        .map(|k| k * PI * PI)
        .collect();
    let mut errors = Vec::new();
    for n in [8, 16, 32] {
        let mesh = square(n);
        let g = Geometry::new(&mesh);
        let (form, star) = (
            &g.laplacian_form::<S>(0) * S::slot(),
            &g.star::<S>(0) * S::slot(),
        );
        let diagonal = form.eigh(&star, exact.len()).0;
        let (stiffness, mass) = (
            &g.stiffness::<S>(0) * S::slot(),
            &g.mass::<S>(0) * S::slot(),
        );
        let galerkin = stiffness.eigh(&mass, exact.len()).0;
        let error = |v: &[f64]| {
            v.iter()
                .zip(&exact)
                .map(|(a, b)| (a - b).abs() / b.max(1.0))
                .fold(0.0, f64::max)
        };
        let (ed, eg) = (error(&diagonal), error(&galerkin));
        println!(
            "{n:2} × {n:2}: ω²/π² = {:?}, error {ed:.2e} (diagonal), {eg:.2e} (Galerkin)",
            diagonal
                .iter()
                .map(|v| (v / (PI * PI) * 1000.0).round() / 1000.0)
                .collect::<Vec<_>>()
        );
        errors.push((ed, eg));
    }
    for w in errors.windows(2) {
        assert!(
            w[1].0 < w[0].0 / 3.0 && w[1].1 < w[0].1 / 3.0,
            "second order: {errors:?}"
        );
    }
    println!("second-order convergence, both stars");
}
