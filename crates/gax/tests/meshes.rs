//! Simplicial complexes and meshes (ADR-049): construction errors, the faces and boundary maps of
//! complexes from triangles to 4-simplices, and the Betti numbers of known spaces.

#![cfg(feature = "pga3d")]

mod support {
    pub mod complex_checks;
}

use gax::mesh::{MeshError, SimplicialComplex};
use std::collections::BTreeMap;
use support::complex_checks::check;

/// How a grid's sides are glued.
#[derive(Clone, Copy)]
enum Glue {
    /// Not at all: a disk.
    Disk,
    /// `j` wraps: a cylinder.
    Cylinder,
    /// Both wrap: a torus.
    Torus,
    /// `j` wraps with `i` reversed: a Möbius strip.
    Mobius,
    /// `i` wraps, and `j` wraps with `i` reversed: a Klein bottle.
    Klein,
}

/// An `n × m` grid of squares, each cut into two triangles, glued.
fn grid(n: usize, m: usize, glue: Glue) -> (usize, Vec<[usize; 3]>) {
    let mut ids = BTreeMap::new();
    let mut id = |i: usize, j: usize| -> usize {
        let wrap_i = matches!(glue, Glue::Torus | Glue::Klein);
        let (mut i, mut j) = (i, j);
        if j == m && !matches!(glue, Glue::Disk) {
            j = 0;
            if matches!(glue, Glue::Mobius | Glue::Klein) {
                i = if wrap_i { (n - i) % n } else { n - i };
            }
        }
        if wrap_i {
            i %= n;
        }
        let next = ids.len();
        *ids.entry((i, j)).or_insert(next)
    };
    let mut faces = Vec::new();
    for i in 0..n {
        for j in 0..m {
            let (a, b, c, d) = (id(i, j), id(i + 1, j), id(i + 1, j + 1), id(i, j + 1));
            faces.push([a, b, c]);
            faces.push([a, c, d]);
        }
    }
    (ids.len(), faces)
}

#[test]
fn errors_name_the_simplex() {
    assert_eq!(
        SimplicialComplex::new(3, &[[0, 1, 3]]),
        Err(MeshError::VertexOutOfRange {
            simplex: 0,
            vertex: 3,
            vertices: 3
        })
    );
    assert_eq!(
        SimplicialComplex::new(4, &[[0, 1, 2], [2, 3, 2]]),
        Err(MeshError::RepeatedVertex {
            simplex: 1,
            vertex: 2
        })
    );
    // The same triangle twice, in another order.
    assert_eq!(
        SimplicialComplex::new(4, &[[0, 1, 2], [1, 2, 3], [2, 0, 1]]),
        Err(MeshError::DuplicateSimplex {
            first: 0,
            second: 2
        })
    );
    let message = SimplicialComplex::new(3, &[[0, 1, 3]])
        .unwrap_err()
        .to_string();
    assert_eq!(message, "simplex 0 names vertex 3, outside the 3 vertices");
}

#[test]
fn faces_are_sorted_and_oriented() {
    // One triangle given clockwise in vertex numbers.
    let c = SimplicialComplex::new(3, &[[0, 2, 1]]).unwrap();
    assert_eq!(c.simplices(1).collect::<Vec<_>>(), [[0, 1], [0, 2], [1, 2]]);
    // The faces opposite vertices 0, 2 and 1 in turn: the edges (2, 1), (0, 1) with the sign
    // (−1)¹, and (0, 2). As stored, sorted: −[1, 2], −[0, 1], +[0, 2], the cycle 0 → 2 → 1 → 0.
    assert_eq!(c.faces(2, 0), [(2, -1), (0, -1), (1, 1)]);
    assert_eq!(c.faces(1, 2), [(2, 1), (1, -1)]);
    // d(0) lists an edge's ends in its order.
    let d0 = c.d::<gax::pga3d::Scalar<(), f64>>(0);
    let rows: Vec<_> = d0.couplings().map(|(r, col, x)| (r, col, x.s())).collect();
    assert_eq!(
        rows,
        [
            (0, 0, -1.0),
            (0, 1, 1.0),
            (1, 0, -1.0),
            (1, 2, 1.0),
            (2, 1, -1.0),
            (2, 2, 1.0)
        ]
    );
}

#[test]
fn betti_numbers_of_surfaces() {
    let cases = [
        (Glue::Disk, [1, 0, 0], true, 1),
        (Glue::Cylinder, [1, 1, 0], true, 0),
        (Glue::Torus, [1, 2, 1], true, 0),
        (Glue::Mobius, [1, 1, 0], false, 0),
        (Glue::Klein, [1, 1, 0], false, 0),
    ];
    for (glue, betti, oriented, euler) in cases {
        let (n, faces) = grid(4, 5, glue);
        let c = SimplicialComplex::new(n, &faces).unwrap();
        assert_eq!(c.betti_numbers(), betti);
        assert_eq!(c.is_oriented(), oriented);
        assert_eq!(c.euler_characteristic(), euler);
        assert!(c.is_pseudomanifold());
        check(n, &faces);
    }
    // The torus is closed; the disk's boundary is its rim.
    let (n, faces) = grid(4, 5, Glue::Torus);
    assert_eq!(
        SimplicialComplex::new(n, &faces).unwrap().boundary(),
        Vec::<usize>::new()
    );
    let (n, faces) = grid(4, 5, Glue::Disk);
    assert_eq!(
        SimplicialComplex::new(n, &faces).unwrap().boundary().len(),
        2 * (4 + 5)
    );
}

#[test]
fn the_octahedron_is_a_sphere() {
    // ±x, ±y, ±z as 0..6; the eight faces counter-clockwise from outside.
    let faces = [
        [0, 2, 4],
        [2, 1, 4],
        [1, 3, 4],
        [3, 0, 4],
        [2, 0, 5],
        [1, 2, 5],
        [3, 1, 5],
        [0, 3, 5],
    ];
    let c = SimplicialComplex::new(6, &faces).unwrap();
    assert_eq!(c.betti_numbers(), [1, 0, 1]);
    assert_eq!(c.euler_characteristic(), 2);
    assert!(c.is_oriented());
    // One face turned over breaks the orientation, not the topology.
    let mut flipped = faces;
    flipped[3] = [0, 3, 4];
    let c = SimplicialComplex::new(6, &flipped).unwrap();
    assert!(!c.is_oriented() && c.betti_numbers() == [1, 0, 1]);
    check(6, &faces);
}

#[test]
fn solids_and_spheres_in_higher_dimensions() {
    // A cube of six tetrahedra, along the paths from corner 0 to corner 7 (bits x, y, z).
    let tets: Vec<[usize; 4]> = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ]
    .iter()
    .map(|axes| {
        let mut v = 0;
        let mut path = [0; 4];
        for (k, &a) in axes.iter().enumerate() {
            v |= 1 << a;
            path[k + 1] = v;
        }
        path
    })
    .collect();
    let c = SimplicialComplex::new(8, &tets).unwrap();
    assert_eq!(c.betti_numbers(), [1, 0, 0, 0]);
    assert_eq!(c.boundary().len(), 12); // two triangles on each face of the cube
    assert_eq!(c.len(1), 19); // 12 edges, 6 face diagonals, the long diagonal
    check(8, &tets);
    // The boundary of a 4-simplex, five tetrahedra: a 3-sphere.
    let sphere: Vec<[usize; 4]> = (0..5)
        .map(|skip| {
            let v: Vec<usize> = (0..5).filter(|&v| v != skip).collect();
            [v[0], v[1], v[2], v[3]]
        })
        .collect();
    let c = SimplicialComplex::new(5, &sphere).unwrap();
    assert_eq!(c.betti_numbers(), [1, 0, 0, 1]);
    assert_eq!(c.boundary(), Vec::<usize>::new());
    check(5, &sphere);
    // And two disjoint triangles: two pieces.
    assert_eq!(
        SimplicialComplex::new(6, &[[0, 1, 2], [3, 4, 5]])
            .unwrap()
            .betti_numbers(),
        [2, 0, 0]
    );
}

#[test]
fn three_triangles_on_an_edge_are_not_a_pseudomanifold() {
    let c = SimplicialComplex::new(5, &[[0, 1, 2], [0, 1, 3], [0, 1, 4]]).unwrap();
    assert!(!c.is_pseudomanifold() && !c.is_oriented());
    assert_eq!(c.coface_counts(1)[0], 3);
}

/// Random lists of simplices in every dimension: errors, or complexes with their invariants
/// (the fuzz target `mesh_complex` runs the same checks on arbitrary input).
#[test]
fn random_complexes_keep_their_invariants() {
    let mut s = 0x2545_f491_4f6c_dd1du64;
    let mut next = |m: usize| {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s % m as u64) as usize
    };
    for _ in 0..200 {
        let vertices = 3 + next(10);
        let count = 1 + next(12);
        let t2: Vec<[usize; 2]> = (0..count)
            .map(|_| [next(vertices + 1), next(vertices)])
            .collect();
        let t3: Vec<[usize; 3]> = (0..count)
            .map(|_| core::array::from_fn(|_| next(vertices)))
            .collect();
        let t4: Vec<[usize; 4]> = (0..count)
            .map(|_| core::array::from_fn(|_| next(vertices)))
            .collect();
        let t5: Vec<[usize; 5]> = (0..count)
            .map(|_| core::array::from_fn(|_| next(vertices)))
            .collect();
        check(vertices, &t2);
        check(vertices, &t3);
        check(vertices, &t4);
        check(vertices, &t5);
    }
}
