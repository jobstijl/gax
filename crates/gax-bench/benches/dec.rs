//! Meshes and discrete exterior calculus: building a complex and its metric, assembling the
//! operators, and solving with them (`docs/performance.md`, "Meshes and DEC").
#![allow(missing_docs)] // `criterion_group!` generates an undocumented function

use criterion::{Criterion, criterion_group, criterion_main};
use gax::dec::Geometry;
use gax::field::{Field, SparseMap};
use gax::mesh::{Mesh, SimplicialComplex};
use gax::vga3d::{Scalar, Vector};
use std::collections::BTreeMap;
use std::hint::black_box;

type S = Scalar<(), f64>;

/// The icosahedron subdivided `level` times, on the unit sphere.
fn icosphere(level: usize) -> (Vec<Vector<(), f64>>, Vec<[usize; 3]>) {
    let t = f64::midpoint(1.0, 5f64.sqrt());
    let mut points: Vec<[f64; 3]> = vec![
        [-1.0, t, 0.0],
        [1.0, t, 0.0],
        [-1.0, -t, 0.0],
        [1.0, -t, 0.0],
        [0.0, -1.0, t],
        [0.0, 1.0, t],
        [0.0, -1.0, -t],
        [0.0, 1.0, -t],
        [t, 0.0, -1.0],
        [t, 0.0, 1.0],
        [-t, 0.0, -1.0],
        [-t, 0.0, 1.0],
    ];
    let mut faces: Vec<[usize; 3]> = vec![
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ];
    for _ in 0..level {
        let mut midpoints = BTreeMap::new();
        let mut mid = |a: usize, b: usize, points: &mut Vec<[f64; 3]>| -> usize {
            *midpoints.entry((a.min(b), a.max(b))).or_insert_with(|| {
                points.push(core::array::from_fn(|i| {
                    f64::midpoint(points[a][i], points[b][i])
                }));
                points.len() - 1
            })
        };
        faces = faces
            .iter()
            .flat_map(|&[a, b, c]| {
                let (ab, bc, ca) = (
                    mid(a, b, &mut points),
                    mid(b, c, &mut points),
                    mid(c, a, &mut points),
                );
                [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
            })
            .collect();
    }
    let vertices = points
        .iter()
        .map(|p| {
            let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            Vector::new(p[0] / r, p[1] / r, p[2] / r)
        })
        .collect();
    (vertices, faces)
}

/// A unit cube of `n³` small cubes, six tetrahedra each.
fn cube(n: usize) -> (Vec<Vector<(), f64>>, Vec<[usize; 4]>) {
    let id = |i: usize, j: usize, k: usize| (i * (n + 1) + j) * (n + 1) + k;
    let h = 1.0 / n as f64;
    let vertices = (0..=n)
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
    (vertices, tets)
}

fn bench(c: &mut Criterion) {
    let (sv, sf) = icosphere(4);
    let (cv, ct) = cube(8);
    let sphere = Mesh::new(sv.iter().copied().collect::<Field<_>>(), &sf).unwrap();
    let solid = Mesh::new(cv.iter().copied().collect::<Field<_>>(), &ct).unwrap();
    let (gs, gc) = (Geometry::new(&sphere), Geometry::new(&solid));

    let mut g = c.benchmark_group("dec");
    g.sample_size(20);
    g.bench_function("complex: icosphere 4 (5120 triangles)", |b| {
        b.iter(|| SimplicialComplex::new(sv.len(), black_box(&sf)));
    });
    g.bench_function("complex: cube 8 (3072 tetrahedra)", |b| {
        b.iter(|| SimplicialComplex::new(cv.len(), black_box(&ct)));
    });
    g.bench_function("geometry: icosphere 4", |b| {
        b.iter(|| Geometry::new(black_box(&sphere)));
    });
    g.bench_function("geometry: cube 8", |b| {
        b.iter(|| Geometry::new(black_box(&solid)));
    });
    g.bench_function("mass M₁: cube 8", |b| {
        b.iter(|| black_box(&gc).mass::<S>(1));
    });
    g.bench_function("cotangent Laplacian: icosphere 4", |b| {
        b.iter(|| black_box(&gs).laplacian_form::<S>(0));
    });
    let system: SparseMap<S> = &gc.mass::<S>(0) + &gc.stiffness::<S>(0).map_cells(|x| x * 0.002);
    let system = &system * S::slot();
    let u: Field<S> = Field::from_fn(cv.len(), |v| S::new(if v == 0 { 1.0 } else { 0.0 }));
    let rhs = &gc.mass::<S>(0) * &u;
    g.bench_function("heat step (CG): cube 8", |b| {
        b.iter(|| system.solve(black_box(&rhs)));
    });
    let (form, star) = (
        &gs.laplacian_form::<S>(0) * S::slot(),
        &gs.star::<S>(0) * S::slot(),
    );
    g.sample_size(10);
    g.bench_function("16 Laplacian modes: icosphere 4", |b| {
        b.iter(|| form.eigh_with(black_box(&star), 16));
    });
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
