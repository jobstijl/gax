//! The math: the two drums welded from triangles of a square lattice, their fields' restriction
//! to the tiles and assembly back, the five-point wave operator on each, and the transplantation
//! that carries one drum's standing waves onto the other's. No drawing.
//!
//! Every map is a sparse field map of numbers (VGA2D scalars): reading a tile's samples, its
//! adjoint putting a tile back in place, a signed mixing of three tiles into one, and the
//! assembly that welds the tiles of a drum into one field by averaging its seams.

use gax::field::{Field, SparseMap};
use gax::mesh::Topology;
use gax::vga2d;

use crate::simplicial::interior;

/// A number.
pub type Scalar = vga2d::Scalar;
/// A map of numbers.
pub type ScalarMap = vga2d::Scalar<(Scalar,)>;
/// A place on a drum, an antivector of the plane (numga's `Point-`).
pub type Point = vga2d::intrinsic::Vector;

/// The point at `(x, y)`.
pub fn point(x: f64, y: f64) -> Point {
    vga2d::Vector::new(x, y).hodge()
}

/// A point's place.
pub fn coordinates(p: Point) -> [f64; 2] {
    [
        (vga2d::Vector::new(1.0, 0.0) & p).s(),
        (vga2d::Vector::new(0.0, 1.0) & p).s(),
    ]
}

/// The map with these cells, a number each.
fn numbers(
    triples: impl IntoIterator<Item = (usize, usize, f64)>,
    shape: (usize, usize),
) -> SparseMap<ScalarMap> {
    let triples: Vec<_> = triples
        .into_iter()
        .map(|(r, c, x)| (r, c, Scalar::slot() * x))
        .collect();
    SparseMap::from_triples(triples, shape)
}

/// The identity on `n` sites.
pub fn identity(n: usize) -> SparseMap<ScalarMap> {
    numbers((0..n).map(|i| (i, i, 1.0)), (n, n))
}

/// The reference tile's samples, `n` steps along each leg: their integer columns and rows (the
/// row outer), its triangles, and each triangle's right-angled corner, the one its square's
/// diagonal faces (numga's `_samples`).
pub fn samples(n: usize) -> (Vec<[usize; 2]>, Vec<[usize; 3]>, Vec<usize>) {
    let mut local = vec![vec![usize::MAX; n + 1]; n + 1];
    let mut places = Vec::new();
    for (row, nodes) in local.iter_mut().enumerate() {
        for (column, node) in nodes.iter_mut().enumerate().take(n + 1 - row) {
            *node = places.len();
            places.push([column, row]);
        }
    }
    let lower = places.iter().filter(|&&[c, r]| r + c < n);
    let upper = places.iter().filter(|&&[c, r]| r + c + 1 < n);
    let mut faces: Vec<[usize; 3]> = lower
        .map(|&[c, r]| [local[r][c], local[r][c + 1], local[r + 1][c]])
        .collect();
    let lowers = faces.len();
    faces.extend(upper.map(|&[c, r]| [local[r][c + 1], local[r + 1][c + 1], local[r + 1][c]]));
    let right = (0..faces.len()).map(|f| usize::from(f >= lowers)).collect();
    (places, faces, right)
}

/// The reference triangle's samples as points, and its triangles.
pub fn triangle(n: usize) -> (Field<Point>, Topology<3>) {
    let (places, faces, _) = samples(n);
    let vertices = places
        .iter()
        .map(|&[c, r]| point(c as f64 / n as f64, r as f64 / n as f64))
        .collect();
    (
        vertices,
        Topology::new(places.len(), &faces).expect("the tile"),
    )
}

/// A drum welded from triangles of a square lattice, each tile three indices into the
/// lattice's points row by row.
pub struct Drum {
    pub vertices: Field<Point>,
    pub mesh: Topology<3>,
    /// The interior vertices, where the field lives (it is zero on the fixed rim).
    pub interior: Vec<usize>,
    /// The lattice neighbours between interior sites: steps along a row or a column.
    pub neighbors: SparseMap<ScalarMap>,
    /// The interior field read at every tile's samples, `[tile samples <- interior]`.
    pub restriction: SparseMap<ScalarMap>,
    /// The tiles welded back, each interior vertex the mean of its samples.
    pub assembly: SparseMap<ScalarMap>,
    /// The interior field on every vertex, zero on the rim, `[vertices <- interior]`.
    pub lift: SparseMap<ScalarMap>,
}

impl Drum {
    pub fn new(tiles: &[[usize; 3]], lattice: usize, n: usize) -> Self {
        let corners: Vec<[[i64; 2]; 3]> = tiles
            .iter()
            .map(|t| t.map(|i| [(i % lattice) as i64, (i / lattice) as i64]))
            .collect();
        let (places, local_faces, local_right) = samples(n);
        // Integer coordinates identify coincident samples exactly, the tiles' seams included.
        let locate = |c: &[[i64; 2]; 3], [column, row]: [usize; 2]| -> [i64; 2] {
            let (column, row, n) = (column as i64, row as i64, n as i64);
            [0, 1].map(|k| c[0][k] * n + (c[1][k] - c[0][k]) * column + (c[2][k] - c[0][k]) * row)
        };
        let mut nodes: Vec<[i64; 2]> = corners
            .iter()
            .flat_map(|c| places.iter().map(move |&p| locate(c, p)))
            .collect();
        nodes.sort_unstable();
        nodes.dedup();
        let tile_nodes: Vec<Vec<usize>> = corners
            .iter()
            .map(|c| {
                places
                    .iter()
                    .map(|&p| nodes.binary_search(&locate(c, p)).expect("a node"))
                    .collect()
            })
            .collect();
        let vertices: Field<Point> = nodes
            .iter()
            .map(|&[x, y]| point(x as f64 / n as f64, y as f64 / n as f64))
            .collect();
        // A tile turned the other way has its triangles' last two corners swapped, and with them
        // the right-angled corner's place.
        let mut faces = Vec::new();
        let mut right = Vec::new();
        for (c, tile) in corners.iter().zip(&tile_nodes) {
            let frame = [0, 1].map(|k| [c[k + 1][0] - c[0][0], c[k + 1][1] - c[0][1]]);
            let winding = frame[0][0] * frame[1][1] - frame[0][1] * frame[1][0];
            for (f, &r) in local_faces.iter().zip(&local_right) {
                let face = f.map(|v| tile[v]);
                if winding > 0 {
                    faces.push(face);
                    right.push(r);
                } else {
                    faces.push([face[0], face[2], face[1]]);
                    right.push([0, 2, 1][r]);
                }
            }
        }
        let mesh = Topology::new(vertices.len(), &faces).expect("the drum");
        let inside = interior(&mesh, 0);
        let site = |v: usize| inside.binary_search(&v).ok();
        let lift = numbers(
            inside.iter().enumerate().map(|(k, &v)| (v, k, 1.0)),
            (vertices.len(), inside.len()),
        );
        let samples_all: Vec<usize> = tile_nodes.iter().flatten().copied().collect();
        let restriction = numbers(
            samples_all
                .iter()
                .enumerate()
                .filter_map(|(s, &v)| site(v).map(|k| (s, k, 1.0))),
            (samples_all.len(), inside.len()),
        );
        let mut multiplicity = vec![0.0; inside.len()];
        for &v in &samples_all {
            if let Some(k) = site(v) {
                multiplicity[k] += 1.0;
            }
        }
        let assembly = numbers(
            samples_all
                .iter()
                .enumerate()
                .filter_map(|(s, &v)| site(v).map(|k| (k, s, 1.0 / multiplicity[k]))),
            (inside.len(), samples_all.len()),
        );
        // Every edge but the squares' diagonals, each triangle's side facing its right angle,
        // coupling interior sites both ways.
        let mut diagonal = vec![false; mesh.len(1)];
        for (f, &r) in right.iter().enumerate() {
            diagonal[mesh.faces(2, f)[r].0] = true;
        }
        let neighbors = numbers(
            (0..mesh.len(1))
                .filter(|&e| !diagonal[e])
                .filter_map(|e| {
                    let ends = mesh.simplex(1, e);
                    Some((site(ends[0])?, site(ends[1])?))
                })
                .flat_map(|(a, b)| [(a, b, 1.0), (b, a, 1.0)]),
            (inside.len(), inside.len()),
        );
        Drum {
            vertices,
            mesh,
            interior: inside,
            neighbors,
            restriction,
            assembly,
            lift,
        }
    }

    /// The five-point wave operator, `(4 − neighbours) / h²` with `h = 1 / n`.
    pub fn laplacian(&self, n: usize) -> SparseMap<ScalarMap> {
        let h2 = (n * n) as f64;
        (&identity(self.interior.len()).map_cells(|c| c * 4.0)
            + &self.neighbors.map_cells(|c| c * -1.0))
            .map_cells(|c| c * h2)
    }
}

/// Signed sums of tiles' fields in their common local coordinates: destination tile `i` takes
/// `Σ signs[i][k] · tile contributors[i][k]`, sample by sample, each tile read by a selection and
/// put in place by its adjoint.
pub fn tile_mixing(
    contributors: &[[usize; 3]],
    signs: &[[f64; 3]],
    samples: usize,
) -> SparseMap<ScalarMap> {
    let tiles = contributors.len();
    let read = |t: usize| {
        numbers(
            (0..samples).map(|s| (s, t * samples + s, 1.0)),
            (samples, tiles * samples),
        )
    };
    let mut mixing: Option<SparseMap<ScalarMap>> = None;
    for (i, (sources, sign)) in contributors.iter().zip(signs).enumerate() {
        for (&j, &w) in sources.iter().zip(sign) {
            let term = read(i).adjoint().compose(&read(j).map_cells(|c| c * w));
            mixing = Some(match mixing {
                Some(m) => &m + &term,
                None => term,
            });
        }
    }
    mixing.expect("tiles")
}
