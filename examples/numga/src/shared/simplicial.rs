//! What numga's simplicial mesh (`examples/util/mesh.py`) adds to `gax::mesh` and stays in the
//! examples: the constructors some examples share (numga's torus and icosphere, a hexagonal
//! disk), the midpoint subdivision, the edges' sides and corners across, the first face at each
//! vertex, and a breadth-first tree of the edges. numga's batches of incidences (`incidence`,
//! `oriented`, `boundary`, `corners`), the boundary and interior lists, selection maps, coherent
//! and outward winding, and the generators of the first homology with their circulation are
//! gax's (`Topology::incidence`, `oriented`, `facets`, `exterior`, `interior`, `coherent`,
//! `cycles`, `circulation`, `mesh::corners`, `SparseMap::selection`, `Mesh::outward`; ADR-058).
//! Shared by the surfaces ports of Whitney forms, Dirac–Kähler fields, Regge calculus, frame
//! fields, developability and connections; each uses part of it.

use gax::Extensor;
use gax::field::Field;
use gax::mesh::Topology;
use gax::vga3d::{Bivector, Vector};
use std::collections::VecDeque;

/// The sign of the permutation that sorts `v`: `+1` for an even order of distinct indices.
pub fn parity(v: &[usize]) -> i8 {
    let inversions = (0..v.len())
        .flat_map(|i| (i + 1..v.len()).map(move |j| (i, j)))
        .filter(|&(i, j)| v[i] > v[j])
        .count();
    if inversions.is_multiple_of(2) { 1 } else { -1 }
}

/// The triangles of a surface's topology, as given.
pub fn faces(t: &Topology<3>) -> Vec<[usize; 3]> {
    t.simplices(2).map(|f| [f[0], f[1], f[2]]).collect()
}

/// The edges, each its two vertices, the lower first.
pub fn edges<const N: usize>(t: &Topology<N>) -> Vec<[usize; 2]> {
    t.simplices(1).map(|e| [e[0], e[1]]).collect()
}

/// Every face split in four at the vertices of its edges, numbered after the `vertices`
/// vertices, edge by edge: a middle face through them, then a face at each corner, the corner
/// and the middles of the edges after it and before it (numga's `split_faces`).
pub fn split_faces(t: &Topology<3>) -> Vec<[usize; 3]> {
    let n = t.len(0);
    let middles: Vec<[usize; 3]> = (0..t.len(2))
        .map(|f| core::array::from_fn(|k| n + t.faces(2, f)[k].0))
        .collect();
    let corners = (0..t.len(2)).flat_map(|f| {
        let face = t.simplex(2, f);
        let m = middles[f];
        [0, 1, 2].map(|k| [face[k], m[(k + 2) % 3], m[(k + 1) % 3]])
    });
    middles.iter().copied().chain(corners).collect()
}

/// The surface with every face split in four at the midpoints of its edges, `levels` times over
/// (numga's `midpoint_split`): its vertices the coarse ones followed, level by level, by the
/// midpoints, so that the finer mesh lies exactly on the coarser one.
pub fn midpoint_split<V>(
    t: &Topology<3>,
    vertices: &Field<V>,
    levels: usize,
) -> (Topology<3>, Field<V>)
where
    V: Extensor<Slots = (), Coef = f64>
        + core::ops::Add<Output = V>
        + core::ops::Mul<f64, Output = V>,
{
    let (mut t, mut vertices) = (t.clone(), vertices.clone());
    for _ in 0..levels {
        let midpoints = edges(&t)
            .into_iter()
            .map(|[a, b]| (vertices.site(a) + vertices.site(b)) * 0.5);
        vertices = vertices.iter().chain(midpoints).collect();
        t = Topology::new(vertices.len(), &split_faces(&t)).expect("the split faces");
    }
    (t, vertices)
}

/// The first face at each vertex: the least index of those with the vertex as a corner.
pub fn vertex_faces(t: &Topology<3>) -> Vec<usize> {
    let mut first = vec![usize::MAX; t.len(0)];
    for f in (0..t.len(2)).rev() {
        for &v in t.simplex(2, f) {
            first[v] = f;
        }
    }
    first
}

/// An interior edge, the faces beside it and the corners across it: the face on its left runs
/// along it (from its lower vertex to its higher), the face on its right against it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sides {
    /// The edge.
    pub edge: usize,
    /// The faces on its left and on its right.
    pub faces: [usize; 2],
    /// The corner the edge faces in each of them.
    pub across: [usize; 2],
}

/// The interior edges, in order, with the faces beside them and the corners across (numga's
/// `edge_faces` and `edge_opposites`).
pub fn sides(t: &Topology<3>) -> Vec<Sides> {
    let mut beside: Vec<[Option<(usize, usize)>; 2]> = vec![[None, None]; t.len(1)];
    for f in 0..t.len(2) {
        let face = t.simplex(2, f);
        for (k, &(e, sign)) in t.faces(2, f).iter().enumerate() {
            beside[e][usize::from(sign < 0)] = Some((f, face[k]));
        }
    }
    beside
        .into_iter()
        .enumerate()
        .filter_map(|(edge, [left, right])| {
            let ((l, a), (r, b)) = (left?, right?);
            Some(Sides {
                edge,
                faces: [l, r],
                across: [a, b],
            })
        })
        .collect()
}

/// A breadth-first tree of the edges from the first vertex, its neighbours in increasing order
/// (numga's `spanning_tree`): every vertex reached after the first, in the order reached, and
/// the edge joining each to its parent.
pub fn spanning_tree<const N: usize>(t: &Topology<N>) -> (Vec<usize>, Vec<usize>) {
    let mut neighbours: Vec<Vec<(usize, usize)>> = vec![Vec::new(); t.len(0)];
    for (e, [a, b]) in edges(t).into_iter().enumerate() {
        neighbours[a].push((b, e));
        neighbours[b].push((a, e));
    }
    for list in &mut neighbours {
        list.sort_unstable();
    }
    let mut reached = vec![false; t.len(0)];
    let (mut children, mut tree) = (Vec::new(), Vec::new());
    if t.is_empty() {
        return (children, tree);
    }
    reached[0] = true;
    let mut queue = VecDeque::from([0]);
    while let Some(v) = queue.pop_front() {
        for &(w, e) in &neighbours[v] {
            if !reached[w] {
                reached[w] = true;
                children.push(w);
                tree.push(e);
                queue.push_back(w);
            }
        }
    }
    (children, tree)
}

/// A closed torus about the `z` axis (numga's `Mesh.torus`): a circle of radius `minor` in the
/// `xz` plane, `major` from the axis, turned about it; vertex `u · minor_count + v` at the `u`-th
/// turn about the axis and the `v`-th about the tube, and two triangles to each cell, all the
/// `[00, 10, 11]` ones first, then the `[00, 11, 01]` ones, counter-clockwise from outside.
pub fn torus(
    major: f64,
    minor: f64,
    major_count: usize,
    minor_count: usize,
) -> (Topology<3>, Field<Vector>) {
    let turn = |plane: Bivector, share: f64| (plane * (-core::f64::consts::PI * share)).exp();
    let (xz, xy) = (Bivector::new(0.0, -1.0, 0.0), Bivector::new(0.0, 0.0, 1.0));
    let x = Vector::new(1.0, 0.0, 0.0);
    let tube: Vec<Vector> = (0..minor_count)
        .map(|v| x * major + (turn(xz, v as f64 / minor_count as f64) >> (x * minor)))
        .collect();
    let vertices = (0..major_count)
        .flat_map(|u| {
            let around = turn(xy, u as f64 / major_count as f64);
            tube.iter().map(move |&p| around >> p)
        })
        .collect();
    let at = |u: usize, v: usize| (u % major_count) * minor_count + v % minor_count;
    let cells = || (0..major_count).flat_map(|u| (0..minor_count).map(move |v| (u, v)));
    let first = cells().map(|(u, v)| [at(u, v), at(u + 1, v), at(u + 1, v + 1)]);
    let second = cells().map(|(u, v)| [at(u, v), at(u + 1, v + 1), at(u, v + 1)]);
    let faces: Vec<[usize; 3]> = first.chain(second).collect();
    let t = Topology::new(major_count * minor_count, &faces).expect("the torus");
    (t, vertices)
}

/// The unit sphere as numga builds it (`Mesh.icosphere`): an icosahedron with each face split in
/// four, `levels` times, every vertex pushed out onto the sphere after each split (the port's
/// `surfaces_sphere::icosphere` pushes them out once, at the end).
pub fn icosphere(levels: usize) -> (Topology<3>, Field<Vector>) {
    let g = crate::measure::GOLDEN;
    let corners = [
        [-1.0, g, 0.0],
        [1.0, g, 0.0],
        [-1.0, -g, 0.0],
        [1.0, -g, 0.0],
        [0.0, -1.0, g],
        [0.0, 1.0, g],
        [0.0, -1.0, -g],
        [0.0, 1.0, -g],
        [g, 0.0, -1.0],
        [g, 0.0, 1.0],
        [-g, 0.0, -1.0],
        [-g, 0.0, 1.0],
    ];
    let faces = [
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
    let mut vertices: Field<Vector> = corners
        .iter()
        .map(|&[x, y, z]| Vector::new(x, y, z).normalized())
        .collect();
    let mut t = Topology::new(vertices.len(), &faces).expect("the icosahedron");
    for _ in 0..levels {
        let midpoints = edges(&t)
            .into_iter()
            .map(|[a, b]| ((vertices.site(a) + vertices.site(b)) * 0.5).normalized());
        let finer: Field<Vector> = vertices.iter().chain(midpoints).collect();
        t = Topology::new(finer.len(), &split_faces(&t)).expect("the split icosahedron");
        vertices = finer;
    }
    (t, vertices)
}

/// A disk of nearly equilateral triangles with its first vertex at the centre (numga's
/// scenarios' `disk`): the hexagonal lattice's rings out to `divisions`, each ring mapped onto
/// the circle of its radius, and their Delaunay triangulation, counter-clockwise. Its vertices
/// as coordinates of the plane, in numga's order: the centre, then the lattice column by column.
pub fn hex_disk(radius: f64, divisions: usize) -> (Topology<3>, Field<gax::vga2d::Vector>) {
    use gax::vga2d;
    let spacing = radius / divisions as f64;
    let n = divisions as i64;
    let axes = [
        vga2d::Vector::new(1.0, 0.0),
        vga2d::Vector::from_angle(core::f64::consts::FRAC_PI_3),
    ];
    let mut vertices = vec![vga2d::Vector::new(0.0, 0.0)];
    for column in -n..=n {
        for row in -n..=n {
            let layer = column.abs().max(row.abs()).max((column + row).abs());
            if layer > 0 && layer <= n {
                let along = axes[0] * column as f64 + axes[1] * row as f64;
                vertices.push(along.normalized() * (spacing * layer as f64));
            }
        }
    }
    let vertices: Field<vga2d::Vector> = vertices.into_iter().collect();
    let placed = vertices.map(|v| Vector::new(v.e1(), v.e2(), 0.0));
    let faces = crate::mesh::Mesh::triangulate(placed).faces;
    let t = Topology::new(vertices.len(), &faces).expect("the disk's triangles");
    (t, vertices)
}

#[cfg(test)]
mod simplicial_tests {
    use super::*;
    use gax::vga3d::Scalar;

    #[test]
    fn the_torus_has_two_loops_and_the_sphere_none() {
        let (torus, _) = torus(2.0, 0.75, 8, 5);
        assert_eq!(torus.betti_numbers(), [1, 2, 1]);
        assert!(torus.is_oriented());
        let loops = torus.cycles();
        assert_eq!(loops.len(), 2);
        // Each loop is closed: a gradient's circulation around it is nothing.
        let potential = Field::from_fn(torus.len(0), |v| Scalar::new((v * v % 7) as f64 - 2.5));
        let steps = &torus.d::<Scalar>(0) * &potential;
        let around = &torus.circulation::<Scalar>(&loops) * &steps;
        assert!(around.iter().all(|c| c.s().abs() < 1e-12));
        let (sphere, vertices) = icosphere(2);
        assert!(sphere.cycles().is_empty());
        assert!(vertices.iter().all(|v| (v.norm() - 1.0).abs() < 1e-15));
        assert_eq!(sphere.euler_characteristic(), 2);
    }

    #[test]
    fn outward_winding_undoes_turned_faces() {
        let (sphere, vertices) = icosphere(1);
        let faces = faces(&sphere);
        let mixed: Vec<[usize; 3]> = faces
            .iter()
            .enumerate()
            .map(|(f, &[a, b, c])| if f % 3 == 1 { [a, b, c] } else { [b, a, c] })
            .collect();
        let mesh = gax::mesh::Mesh::new(vertices, &mixed).unwrap();
        let wound = mesh.outward().unwrap();
        assert!(wound.topology().is_oriented());
        assert_eq!(super::faces(wound.topology())[0], faces[0]);
        assert_eq!(sphere.exterior(0), Vec::<usize>::new());
        assert_eq!(sphere.interior(1).len(), sphere.len(1));
    }
}
