//! numga's `mechanics/simplex.py`: the inertia of simplices, shared by the inertia and bending
//! examples.
//!
//! A uniform simplex has the second moments of a few lumped, mass-weighted barycentric points,
//! exactly: [`weights`] gives them, [`samples`] places them in any algebra whose points add and
//! scale (PGA3D's tetrahedra, a beam section's PGA2D triangles). In PGA3D the inertia of point
//! masses is the map `p & [p, ·]` summed over the points, from rates to momenta, which a grid or
//! Monte Carlo sampling of the simplex approaches.
//!
//! numga weights each sample point by the square root of its mass, so that a product of two of
//! them carries the mass once; here the points stay unit and their masses are kept beside them.

// Each binary that includes this module uses a different part of it.
#![allow(dead_code)]

use core::iter::Sum;
use core::ops::{Div, Mul};
use gax::motions::{Motions, Pga3d};
use gax::pga3d::{Line, Point};
use gax_numga_examples::rng::{Draw, Rng};

/// A point of space.
pub type P = Point<(), f64>;
/// Rate to momentum.
pub type Inertia = Line<(Line,), f64>;

/// The barycentric weights mapping a uniform simplex to its equivalent lumped masses:
/// `(1/n + (I - 1/n) sqrt(1 / (1 + n))) / n`, one row per lumped point.
#[allow(clippy::disallowed_methods)] // numbers: numga's weights of a simplex's lumped masses
pub fn weights(n: usize) -> Vec<Vec<f64>> {
    let f = (1.0 / (1.0 + n as f64)).sqrt();
    (0..n)
        .map(|i| {
            (0..n)
                .map(|j| {
                    let d = f64::from(u8::from(i == j)) - 1.0 / n as f64;
                    (1.0 / n as f64 + d * f) / n as f64
                })
                .collect()
        })
        .collect()
}

/// Mass points from barycentric samples: each row of weights gives the point `Σ w c`, whose
/// weight `Σ w` is its mass; the points unitized, and their masses.
pub fn samples<Q>(weights: &[Vec<f64>], corners: &[Q]) -> (Vec<Q>, Vec<f64>)
where
    Q: Copy + Sum + Mul<f64, Output = Q> + Div<f64, Output = Q>,
{
    weights
        .iter()
        .map(|row| {
            let mass: f64 = row.iter().sum();
            let p: Q = row.iter().zip(corners).map(|(w, c)| *c * *w).sum();
            (p / mass, mass)
        })
        .unzip()
}

/// The inertia map of barycentric samples of a simplex: each mass point's `p & [p, ·]`
/// (`Motions::point_inertia`) times its mass.
pub fn inertia(weights: &[Vec<f64>], corners: &[P]) -> Inertia {
    let (points, masses) = samples(weights, corners);
    points
        .iter()
        .zip(&masses)
        .map(|(p, m)| Pga3d::point_inertia(*p) * *m)
        .sum()
}

/// The inertia of the uniform simplex of unit mass, by its lumped masses (exact).
pub fn lumped(corners: &[P]) -> Inertia {
    inertia(&weights(corners.len()), corners)
}

/// The barycentric grid of a tetrahedron: the midpoints of `n` cells along each of three
/// barycentric coordinates, those inside, completed by the fourth; unit mass in all.
pub fn grid(n: usize) -> Vec<Vec<f64>> {
    let s: Vec<f64> = (0..n).map(|i| (i as f64 + 0.5) / n as f64).collect();
    let mut rows = Vec::new();
    for a in &s {
        for b in &s {
            for c in &s {
                if a + b + c < 1.0 {
                    rows.push(vec![*a, *b, *c, 1.0 - a - b - c]);
                }
            }
        }
    }
    let count = rows.len() as f64;
    rows.iter()
        .map(|r| r.iter().map(|w| w / count).collect())
        .collect()
}

/// Uniform barycentric weights: exponential variates normalized (a flat Dirichlet).
pub fn random_weights(count: usize, corners: usize, rng: &mut Rng) -> Vec<Vec<f64>> {
    (0..count)
        .map(|_| {
            let e: Vec<f64> = (0..corners).map(|_| -(1.0 - rng.uniform()).ln()).collect();
            let sum: f64 = e.iter().sum();
            e.iter().map(|v| v / sum / count as f64).collect()
        })
        .collect()
}

/// numga's demo tetrahedron.
pub fn tetrahedron() -> Vec<P> {
    vec![
        Point::xyz(1.0, 0.0, 0.0),
        Point::xyz(-0.5, 0.866, 0.0),
        Point::xyz(-0.5, -0.866, 0.0),
        Point::xyz(0.0, 0.0, 1.414),
    ]
}

/// The relative difference of two inertias, by their coefficients.
#[allow(clippy::disallowed_methods)] // a relative error, the root of summed squares of coefficients
pub fn relative(a: Inertia, b: Inertia) -> f64 {
    let d = (a - b).c.iter().flatten().map(|v| v * v).sum::<f64>();
    let n = b.c.iter().flatten().map(|v| v * v).sum::<f64>();
    (d / n).sqrt()
}
