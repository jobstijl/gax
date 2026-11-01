//! Simplicial complexes and meshes (ADR-049): construction errors, the faces and boundary maps of
//! complexes from triangles to 4-simplices, and the Betti numbers of known spaces.

#![cfg(feature = "pga3d")]

mod support {
    pub mod complex_checks;
}

use gax::mesh::{MeshError, Topology};
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
        Topology::new(3, &[[0, 1, 3]]),
        Err(MeshError::VertexOutOfRange {
            simplex: 0,
            vertex: 3,
            vertices: 3
        })
    );
    assert_eq!(
        Topology::new(4, &[[0, 1, 2], [2, 3, 2]]),
        Err(MeshError::RepeatedVertex {
            simplex: 1,
            vertex: 2
        })
    );
    // The same triangle twice, in another order.
    assert_eq!(
        Topology::new(4, &[[0, 1, 2], [1, 2, 3], [2, 0, 1]]),
        Err(MeshError::DuplicateSimplex {
            first: 0,
            second: 2
        })
    );
    let message = Topology::new(3, &[[0, 1, 3]]).unwrap_err().to_string();
    assert_eq!(message, "simplex 0 names vertex 3, outside the 3 vertices");
}

#[test]
fn faces_are_sorted_and_oriented() {
    // One triangle given clockwise in vertex numbers.
    let c = Topology::new(3, &[[0, 2, 1]]).unwrap();
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
        let c = Topology::new(n, &faces).unwrap();
        assert_eq!(c.betti_numbers(), betti);
        assert_eq!(c.is_oriented(), oriented);
        assert_eq!(c.euler_characteristic(), euler);
        assert!(c.is_pseudomanifold());
        check(n, &faces);
    }
    // The torus is closed; the disk's boundary is its rim.
    let (n, faces) = grid(4, 5, Glue::Torus);
    assert_eq!(
        Topology::new(n, &faces).unwrap().boundary(),
        Vec::<usize>::new()
    );
    let (n, faces) = grid(4, 5, Glue::Disk);
    assert_eq!(
        Topology::new(n, &faces).unwrap().boundary().len(),
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
    let c = Topology::new(6, &faces).unwrap();
    assert_eq!(c.betti_numbers(), [1, 0, 1]);
    assert_eq!(c.euler_characteristic(), 2);
    assert!(c.is_oriented());
    // One face turned over breaks the orientation, not the topology.
    let mut flipped = faces;
    flipped[3] = [0, 3, 4];
    let c = Topology::new(6, &flipped).unwrap();
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
    let c = Topology::new(8, &tets).unwrap();
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
    let c = Topology::new(5, &sphere).unwrap();
    assert_eq!(c.betti_numbers(), [1, 0, 0, 1]);
    assert_eq!(c.boundary(), Vec::<usize>::new());
    check(5, &sphere);
    // And two disjoint triangles: two pieces.
    assert_eq!(
        Topology::new(6, &[[0, 1, 2], [3, 4, 5]])
            .unwrap()
            .betti_numbers(),
        [2, 0, 0]
    );
}

#[test]
fn three_triangles_on_an_edge_are_not_a_pseudomanifold() {
    let c = Topology::new(5, &[[0, 1, 2], [0, 1, 3], [0, 1, 4]]).unwrap();
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

/// numga's batches: a tetrahedron given in an odd order reads its edges by its corners, signed
/// where they run against the stored edges; a triangle's corners and the edges facing them.
#[test]
fn incidence_batches_follow_numgas_order_and_signs() {
    use gax::mesh::corners;
    use gax::pga3d::Scalar;
    assert_eq!(corners(3, 1)[3], [1, 2]);
    assert_eq!(corners(2, 1), [vec![1, 2], vec![0, 2], vec![0, 1]]);
    assert_eq!(corners(1, 0), [vec![0], vec![1]]);
    let t = Topology::new(4, &[[1, 0, 2, 3]]).unwrap();
    // Its first local edge runs from vertex 1 to vertex 0: against the stored edge (0 1).
    let o = t.oriented(3, 1);
    assert_eq!(o.site(0, 0), t.index_of(1, &[0, 1]).unwrap());
    assert_eq!(o.sign(0, 0), -1);
    assert_eq!((o.width(), o.outputs(), o.inputs()), (6, 1, 6));
    // Its faces in the oriented batch: the face facing corner 0 is (0 2 3) as stored, run by
    // the tetrahedron's local order (0 2 3) of its global (1 0 2 3) minus corner 0, i.e. 0 2 3.
    let faces = t.oriented(3, 2);
    assert_eq!(t.simplex(2, faces.site(0, 0)), &[0, 2, 3]);
    assert_eq!(faces.sign(0, 0), 1);
    assert_eq!(t.incidence(3, 0).sites_of(0), &[1, 0, 2, 3]);
    let tri = Topology::new(3, &[[2, 0, 1]]).unwrap();
    let facing = tri.facets(2);
    // The edge facing corner 0 (vertex 2) is (0 1), which the winding 2 → 0 → 1 runs along.
    assert_eq!(tri.simplex(1, facing.site(0, 0)), &[0, 1]);
    assert_eq!(facing.sign(0, 0), 1);
    let s = |x: f64| Scalar::<(), f64>::new(x);
    let field = gax::field::Field::from_fn(3, |e| s(e as f64 + 1.0));
    let read = facing.read_all(&field);
    assert_eq!(read.len(), 3);
    assert_eq!(read[0].site(0).s(), field.site(facing.site(0, 0)).s());
    // An edge's facets: its tail −1, its head +1.
    let edges = tri.facets(1);
    assert_eq!((edges.sign(0, 0), edges.sign(0, 1)), (-1, 1));
    assert_eq!(
        (&edges.map::<Scalar<(), f64>>(1) * &gax::field::Field::from_fn(3, |v| s(v as f64)))
            .site(2)
            .s(),
        2.0
    );
}

/// The boundary of a fan and of a cube of tetrahedra, read in every dimension.
#[test]
fn exterior_and_interior_split_the_simplices() {
    let fan = Topology::new(5, &[[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1]]).unwrap();
    assert_eq!(fan.exterior(1), fan.boundary());
    assert_eq!(fan.interior(0), [0]);
    // A 3 × 3 × 3 cube of Kuhn's tetrahedra: of its 4³ vertices, 2³ are inside.
    let id = |x: usize, y: usize, z: usize| x + 4 * y + 16 * z;
    let mut tets = Vec::new();
    for x in 0..3 {
        for y in 0..3 {
            for z in 0..3 {
                for axes in [
                    [0, 1, 2],
                    [0, 2, 1],
                    [1, 0, 2],
                    [1, 2, 0],
                    [2, 0, 1],
                    [2, 1, 0],
                ] {
                    let mut p = [x, y, z];
                    let mut path = [id(x, y, z); 4];
                    for (k, &a) in axes.iter().enumerate() {
                        p[a] += 1;
                        path[k + 1] = id(p[0], p[1], p[2]);
                    }
                    tets.push(path);
                }
            }
        }
    }
    let cube = Topology::new(64, &tets).unwrap();
    assert_eq!(cube.interior(0).len(), 8);
    assert_eq!(cube.exterior(0).len(), 56);
    assert_eq!(cube.exterior(2).len(), 6 * 9 * 2);
}

/// Coherent winding undoes turned simplices, and a Möbius strip has none.
#[test]
fn coherent_winding_and_the_cycles_of_known_surfaces() {
    let (n, faces) = grid(6, 5, Glue::Torus);
    let mixed: Vec<[usize; 3]> = faces
        .iter()
        .enumerate()
        .map(|(f, &[a, b, c])| if f % 3 == 1 { [b, a, c] } else { [a, b, c] })
        .collect();
    let t = Topology::new(n, &mixed).unwrap();
    assert!(!t.is_oriented());
    let wound = t.coherent().unwrap();
    assert!(wound.is_oriented());
    assert_eq!(wound.simplex(2, 0), t.simplex(2, 0));
    assert_eq!(wound.turned().turned(), wound);
    assert!(wound.turned().is_oriented());
    let (n, faces) = grid(6, 5, Glue::Mobius);
    assert!(Topology::new(n, &faces).unwrap().coherent().is_none());
    // b₁ loops: two on a torus, one on a cylinder, a Möbius strip and a Klein bottle (over the
    // rationals), none on a disk. `check` holds them closed and independent.
    for (glue, b1) in [
        (Glue::Torus, 2),
        (Glue::Cylinder, 1),
        (Glue::Mobius, 1),
        (Glue::Klein, 1),
        (Glue::Disk, 0),
    ] {
        let (n, faces) = grid(7, 6, glue);
        let t = Topology::new(n, &faces).unwrap();
        assert_eq!(t.cycles().len(), b1);
        check(n, &faces);
    }
    // A solid torus of Kuhn's tetrahedra (a ring of four cubes): one loop, around the ring.
    let id = |x: usize, y: usize, z: usize| (x % 4) + 4 * y + 8 * z;
    let mut tets = Vec::new();
    for x in 0..4 {
        for axes in [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ] {
            let mut p = [x, 0, 0];
            let mut path = [id(x, 0, 0); 4];
            for (k, &a) in axes.iter().enumerate() {
                p[a] += 1;
                path[k + 1] = id(p[0], p[1], p[2]);
            }
            tets.push(path);
        }
    }
    let ring = Topology::new(16, &tets).unwrap();
    assert_eq!(ring.betti_numbers(), [1, 1, 0, 0]);
    let loops = ring.cycles();
    assert_eq!(loops.len(), 1);
    check(16, &tets);
    // A theta graph: two independent loops.
    let theta = [[0, 1], [1, 2], [2, 0], [0, 3], [3, 2]];
    assert_eq!(Topology::new(4, &theta).unwrap().cycles().len(), 2);
    check(4, &theta);
}

/// A closed surface wound outward whatever sense its faces come in, piece by piece.
#[cfg(feature = "vga3d")]
#[test]
fn closed_surfaces_wind_outward() {
    use gax::field::Field;
    use gax::mesh::Mesh;
    use gax::vga3d::Vector;
    // Two octahedra, the second inside out and far away, some faces turned.
    let p = [
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ];
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
    let vertices: Field<_> = (0..12)
        .map(|i| {
            let q = p[i % 6];
            let shift = if i < 6 { 0.0 } else { 5.0 };
            Vector::<(), f64>::new(q[0] + shift, q[1], q[2])
        })
        .collect();
    let mut tops: Vec<[usize; 3]> = faces.to_vec();
    tops.extend(faces.iter().map(|&[a, b, c]| [b + 6, a + 6, c + 6]));
    tops[3].swap(0, 1);
    tops[13].swap(1, 2);
    let mesh = Mesh::new(vertices, &tops).unwrap();
    let out = mesh.outward().unwrap();
    assert!(out.topology().is_oriented());
    let expected = Topology::new(12, &{
        let mut t: Vec<[usize; 3]> = faces.to_vec();
        t.extend(faces.iter().map(|&[a, b, c]| [a + 6, b + 6, c + 6]));
        t
    })
    .unwrap();
    for f in 0..16 {
        let (a, b) = (out.topology().simplex(2, f), expected.simplex(2, f));
        // The same cyclic order.
        let at = a.iter().position(|&v| v == b[0]).unwrap();
        assert_eq!(
            [a[at], a[(at + 1) % 3], a[(at + 2) % 3]],
            [b[0], b[1], b[2]],
            "face {f}"
        );
    }
    // A disk has a boundary: no outside.
    let disk = Mesh::new(mesh.vertices.clone(), &[[0, 2, 4]]).unwrap();
    assert!(disk.outward().is_none());
}
