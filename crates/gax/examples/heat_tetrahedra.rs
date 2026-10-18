//! Heat flowing through a cube of tetrahedra (VGA3D, `gax::dec`).
//!
//! The cube is cut into small cubes and each into Freudenthal's six tetrahedra. Heat starts at
//! one corner and spreads by the heat equation `∂u/∂t = Δu`, discretized by linear finite
//! elements: the Galerkin pencil of the stiffness `K = d₀ᵀ M₁ d₀` and the mass `M₀`, stepped by
//! implicit Euler, `(M₀ + dt K) u' = M₀ u`. No heat crosses the boundary, so the total heat
//! `1ᵀ M₀ u` stays what it was, to rounding, while the temperatures even out.
//!
//! Run with `cargo run --example heat_tetrahedra`.

use gax::dec::Geometry;
use gax::field::{Field, SparseMap};
use gax::mesh::Mesh;
use gax::vga3d::{Scalar, Vector};

type S = Scalar<(), f64>;

/// A unit cube of `n³` small cubes, six tetrahedra each.
fn cube(n: usize) -> Mesh<Vector<(), f64>, 4> {
    let id = |i: usize, j: usize, k: usize| (i * (n + 1) + j) * (n + 1) + k;
    let h = 1.0 / n as f64;
    let vertices: Field<_> = (0..=n)
        .flat_map(|i| (0..=n).flat_map(move |j| (0..=n).map(move |k| (i, j, k))))
        .map(|(i, j, k)| Vector::new(i as f64 * h, j as f64 * h, k as f64 * h))
        .collect();
    let mut tets = Vec::new();
    for (i, j, k) in (0..n).flat_map(|i| (0..n).flat_map(move |j| (0..n).map(move |k| (i, j, k)))) {
        for axes in [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ] {
            let mut corner = [i, j, k];
            let mut path = [id(i, j, k); 4];
            for (step, &a) in axes.iter().enumerate() {
                corner[a] += 1;
                path[step + 1] = id(corner[0], corner[1], corner[2]);
            }
            tets.push(path);
        }
    }
    Mesh::new(vertices, &tets).expect("a valid complex")
}

fn main() {
    let mesh = cube(6);
    let c = mesh.complex();
    println!(
        "{} vertices, {} edges, {} triangles, {} tetrahedra",
        c.len(0),
        c.len(1),
        c.len(2),
        c.len(3)
    );
    assert_eq!(c.betti_numbers(), [1, 0, 0, 0]);

    let g = Geometry::new(&mesh);
    let (stiffness, mass) = (g.stiffness::<S>(0), g.mass::<S>(0));
    let dt = 0.002;
    let system: SparseMap<S> = &mass + &stiffness.map_cells(|c| c * dt);
    let system = &system * S::slot();

    let mut u: Field<S> = Field::from_fn(c.len(0), |v| S::new(if v == 0 { 1.0 } else { 0.0 }));
    let ones: Field<S> = Field::from_fn(c.len(0), |_| S::new(1.0));
    let total = |u: &Field<S>| ones.dot(&(&mass * u));
    let start = total(&u);
    let spread = |u: &Field<S>| {
        let (lo, hi) = u.sites().iter().fold((f64::MAX, f64::MIN), |(lo, hi), x| {
            (lo.min(x.s()), hi.max(x.s()))
        });
        hi - lo
    };
    for step in 1..=200 {
        u = system.solve(&(&mass * &u));
        if step % 50 == 0 {
            println!(
                "t = {:.1}: total heat {:.12}, spread {:.3e}",
                f64::from(step) * dt,
                total(&u),
                spread(&u)
            );
        }
    }
    assert!(
        (total(&u) - start).abs() < 1e-10 * start,
        "the heat is conserved"
    );
    assert!(spread(&u) < 0.05, "the temperatures even out");
    println!(
        "conserved to {:.1e}, the corner's heat spread through the cube",
        (total(&u) - start).abs() / start
    );
}
