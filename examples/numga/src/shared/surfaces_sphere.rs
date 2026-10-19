//! What numga's `surfaces/spin_transformations` shares with `surfaces/dirac_waves`: the unit
//! sphere's surface mesh.

use crate::measure::GOLDEN;
use crate::mesh::{Mesh, Vector3};

/// A unit sphere: an icosahedron (its corners `(0, ±1, ±φ)` cycled) with each face split in four
/// at the midpoints of its edges, the given number of times, its vertices pushed out onto the
/// sphere at the end.
pub fn icosphere(levels: usize) -> Mesh {
    let corners = [
        [-1.0, GOLDEN, 0.0],
        [1.0, GOLDEN, 0.0],
        [-1.0, -GOLDEN, 0.0],
        [1.0, -GOLDEN, 0.0],
        [0.0, -1.0, GOLDEN],
        [0.0, 1.0, GOLDEN],
        [0.0, -1.0, -GOLDEN],
        [0.0, 1.0, -GOLDEN],
        [GOLDEN, 0.0, -1.0],
        [GOLDEN, 0.0, 1.0],
        [-GOLDEN, 0.0, -1.0],
        [-GOLDEN, 0.0, 1.0],
    ];
    let faces = vec![
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
    let vertices = corners
        .iter()
        .map(|&[x, y, z]| Vector3::new(x, y, z))
        .collect();
    let mut mesh = Mesh::new(vertices, faces);
    for _ in 0..levels {
        let vertices = mesh
            .vertices
            .iter()
            .chain(mesh.edge_midpoints().iter())
            .collect();
        mesh = Mesh::new(vertices, mesh.split_faces());
    }
    mesh.with_vertices(mesh.vertices.map(|v| v.normalized()))
}
