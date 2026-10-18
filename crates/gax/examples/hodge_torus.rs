//! The Hodge decomposition of a field on a torus (VGA3D, `gax::dec`).
//!
//! A cochain on the edges, a discrete 1-form, splits into three parts, orthogonal in the inner
//! product of the Hodge star: `ω = dα + δβ + h`, a gradient, a curl, and a harmonic part. The
//! harmonic 1-forms are the null space of the Laplace–de Rham operator `Δ₁`; a torus has two, one
//! for each of its loops. The decomposition here:
//!
//! * the gradient part from the Poisson problem `d₀ᵀ ⋆₁ d₀ α = d₀ᵀ ⋆₁ ω`, whose null space, the
//!   constants, is pinned at a vertex;
//! * the harmonic part from the two eigenvectors of `(⋆₁Δ₁, ⋆₁)` at zero;
//! * the curl part as what is left, which is then checked to be co-closed, `δ₁ c = 0`.
//!
//! Run with `cargo run --example hodge_torus`.

use gax::dec::Geometry;
use gax::field::{Field, SparseMap};
use gax::mesh::Mesh;
use gax::vga3d::{Scalar, Vector};
use std::f64::consts::TAU;

type S = Scalar<(), f64>;

/// A torus of revolution as a lattice of `n × m` vertices skewed half a step per ring, so that
/// its triangles are near equilateral and the mesh well-centred.
fn torus(n: usize, m: usize, big: f64, small: f64) -> Mesh<Vector<(), f64>, 3> {
    // After m rings the skew has moved m/2 steps: the seam joins i to i + m/2.
    let id = |i: usize, j: usize| (j % m) * n + (i + (j / m) * (m / 2)) % n;
    let vertices: Field<_> = (0..m)
        .flat_map(|j| (0..n).map(move |i| (i, j)))
        .map(|(i, j)| {
            let (u, v) = (
                TAU * (i as f64 + 0.5 * j as f64) / n as f64,
                TAU * j as f64 / m as f64,
            );
            let r = big + small * v.cos();
            Vector::new(r * u.cos(), r * u.sin(), small * v.sin())
        })
        .collect();
    let faces: Vec<[usize; 3]> = (0..m)
        .flat_map(|j| {
            (0..n).flat_map(move |i| {
                [
                    [id(i, j), id(i + 1, j), id(i, j + 1)],
                    [id(i + 1, j), id(i + 1, j + 1), id(i, j + 1)],
                ]
            })
        })
        .collect();
    Mesh::new(vertices, &faces).expect("a valid complex")
}

fn main() {
    let mesh = torus(35, 10, 4.0, 1.0);
    let c = mesh.topology();
    let g = Geometry::new(&mesh);
    assert!(g.is_well_centered());
    println!(
        "a torus: {} vertices, {} edges, {} faces, Betti numbers {:?}",
        c.len(0),
        c.len(1),
        c.len(2),
        c.betti_numbers()
    );

    let (d0, star1) = (c.d::<S>(0), g.star::<S>(1));
    let inner = |a: &Field<S>, b: &Field<S>| a.dot(&(&star1 * b));

    // A field on the edges with all three parts: pseudo-random values.
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let omega: Field<S> = Field::from_fn(c.len(1), |_| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        S::new((state >> 11) as f64 / (1u64 << 53) as f64 - 0.5)
    });

    // The gradient part: d₀ᵀ ⋆₁ d₀ α = d₀ᵀ ⋆₁ ω. The right side sums to zero, so adding one at
    // vertex 0 to the matrix pins α₀ = 0 and changes nothing else.
    let d0t = d0.transpose_with(|x| x);
    let poisson: SparseMap<S> = &g.laplacian_form::<S>(0)
        + &SparseMap::new(vec![S::new(1.0)], vec![0], vec![0], (c.len(0), c.len(0)));
    let alpha = (&poisson * S::slot()).solve_direct(&(&d0t * &(&star1 * &omega)));
    let exact: Field<S> = &d0 * &alpha;

    // The harmonic part: the null space of Δ₁, projected on in the star's inner product.
    let (values, modes) = (&g.laplacian_form::<S>(1) * S::slot()).eigh(&(&star1 * S::slot()), 4);
    println!(
        "the smallest eigenvalues of Δ₁: {:.2e} {:.2e} {:.3} {:.3}",
        values[0], values[1], values[2], values[3]
    );
    assert!(
        values[0].abs() < 1e-9 && values[1].abs() < 1e-9 && values[2] > 1e-3,
        "two harmonic 1-forms"
    );
    let (h1, h2) = (&modes[0], &modes[1]);
    // Project on their span, through its 2 × 2 Gram matrix.
    let (g11, g12, g22) = (inner(h1, h1), inner(h1, h2), inner(h2, h2));
    let (b1, b2) = (inner(h1, &omega), inner(h2, &omega));
    let det = g11 * g22 - g12 * g12;
    let (x1, x2) = ((g22 * b1 - g12 * b2) / det, (g11 * b2 - g12 * b1) / det);
    let harmonic = &h1.scaled(x1) + &h2.scaled(x2);

    // The curl part: what is left.
    let coexact = &(&omega - &exact) - &harmonic;

    let norm = |a: &Field<S>| inner(a, a).sqrt();
    println!(
        "|ω| = {:.4}: gradient {:.4}, curl {:.4}, harmonic {:.4}",
        norm(&omega),
        norm(&exact),
        norm(&coexact),
        norm(&harmonic)
    );
    let scale = norm(&omega) * norm(&omega);
    for (name, value) in [
        ("gradient · curl", inner(&exact, &coexact)),
        ("gradient · harmonic", inner(&exact, &harmonic)),
        ("curl · harmonic", inner(&coexact, &harmonic)),
    ] {
        println!("{name}: {value:.1e}");
        assert!(value.abs() < 1e-9 * scale, "{name} orthogonal");
    }
    // The curl part is co-closed and the harmonic part closed and co-closed.
    let codifferential = g.codifferential::<S>(1);
    let largest = |f: &Field<S>| f.sites().iter().fold(0.0f64, |m, x| m.max(x.s().abs()));
    assert!(largest(&(&codifferential * &coexact)) < 1e-9);
    assert!(
        largest(&(&c.d::<S>(1) * &harmonic)) < 1e-9
            && largest(&(&codifferential * &harmonic)) < 1e-9
    );
    println!(
        "the three parts are orthogonal; the curl part is co-closed, the harmonic part closed and co-closed"
    );
}
