//! Oriented triangle meshes, the fields on them, and their sparse incidence operators: numga's
//! `examples/mesh.py`.
//!
//! A [`Mesh`] is its vertices, a [`Field`] of any type, and its faces, counter-clockwise from
//! outside. From the faces it derives its edges, the edge facing each corner of each face and
//! that edge's orientation in the face. Quantities on the mesh are fields over its vertices,
//! edges or faces; a quantity at each corner of each face is three face fields, one per corner.
//!
//! The topology does not depend on what the vertices are. Its operators are field maps
//! ([`SparseMap`]) with scalar cells of the vertices' own algebra ([`ScalarOf`]): the
//! differences along the edges [`Mesh::d0`], the circulation around the faces [`Mesh::d1`], the
//! means over the edges and faces, and one step of Loop's subdivision [`Mesh::loop_step`]. They
//! act alike on VGA vectors and PGA points. The geometric quantities (edge vectors, face planes
//! and areas, the cotangent weights, normals, the Whitney reconstruction) read the vertices as
//! vectors of space, `Mesh<Vector3>`, the default.
//!
//! The disks are flat meshes in the xy plane: concentric rings ([`Mesh::disk`]), rings of nearly
//! equilateral cells ([`Mesh::concentric_disk`]), and a hexagonal lattice mapped onto a circle
//! ([`Mesh::triangular_disk`]); the last two are Delaunay triangulations of their vertices
//! ([`Mesh::triangulate`], written here with gax predicates, where numga calls scipy).

mod delaunay;
mod disks;

use core::ops::{Add, Mul, Sub};
use gax::dec::Geometry;
use gax::field::{Field, SparseMap};
use gax::mesh::Complex;
use gax::vga3d::{Bivector, Scalar, Vector};
use gax::{Extensor, Kind};

/// A vector of space: the default vertex.
pub type Vector3 = Vector<(), f64>;
/// A normal: the dual of a face's bivector, so oriented by the face's winding, intrinsic
/// (ADR-051), and reflected as such.
pub type Normal3 = Vector<(), f64, gax::Intrinsic>;
/// A scalar of space's algebra.
pub type Scalar3 = Scalar<(), f64>;
/// A bivector of space: a plane through the origin, with its orientation and area.
pub type Bivector3 = Bivector<(), f64>;
/// The scalar of the algebra of `V`: the cells of the topology's operators on fields of `V`.
pub type ScalarOf<V> = <<<V as Extensor>::Kind as Kind>::Scalar as Kind>::Mv<(), f64>;

/// The scalar `x` of the algebra of `V`.
pub fn scalar<V: Extensor>(x: f64) -> ScalarOf<V> {
    ScalarOf::<V>::from_coeffs(<<V::Kind as Kind>::Scalar as Kind>::arr_from_fn(|_| x))
}

/// The unit vector at `angle` counter-clockwise from `e1` in the xy plane, the disks' plane:
/// gax's `vga2d::Vector::from_angle`, set in space.
pub fn direction(angle: f64) -> Vector3 {
    let [x, y] = gax::vga2d::Vector::from_angle(angle).c;
    Vector3::new(x, y, 0.0)
}

/// An oriented triangle mesh: vertices of type `V` and faces counter-clockwise from outside,
/// with the edges and their orientations they make. A mesh does not change; its vertices move
/// by [`with_vertices`](Self::with_vertices).
#[derive(Clone, Debug)]
pub struct Mesh<V = Vector3> {
    /// The vertices.
    pub vertices: Field<V>,
    /// Each face's vertices, counter-clockwise.
    pub faces: Vec<[usize; 3]>,
    /// Each edge's two vertices, the lower first; the edges in order.
    pub edges: Vec<[usize; 2]>,
    /// The edge facing each corner of each face: from the corner after it to the one after that.
    pub face_edges: Vec<[usize; 3]>,
    /// Whether the face runs along that edge (`1`), from its lower vertex to its higher, or
    /// against it (`-1`).
    pub face_edge_orientation: Vec<[f64; 3]>,
    /// The topology, from which the edges and the incidence maps come.
    complex: Complex<3>,
}

impl<V: Extensor<Slots = (), Coef = f64>> Mesh<V> {
    /// The mesh of the vertices and faces.
    ///
    /// # Panics
    /// If a face names a vertex that is not there.
    #[must_use]
    pub fn new(vertices: Field<V>, faces: Vec<[usize; 3]>) -> Self {
        let complex = Complex::new(vertices.len(), &faces).unwrap_or_else(|e| panic!("{e}"));
        let edges = complex.simplices(1).map(|e| [e[0], e[1]]).collect();
        let face_edges = (0..faces.len())
            .map(|f| core::array::from_fn(|k| complex.faces(2, f)[k].0))
            .collect();
        let face_edge_orientation = (0..faces.len())
            .map(|f| core::array::from_fn(|k| f64::from(complex.faces(2, f)[k].1)))
            .collect();
        Mesh {
            vertices,
            faces,
            edges,
            face_edges,
            face_edge_orientation,
            complex,
        }
    }

    /// The same faces on other vertices, of any type (numga's `copy`).
    ///
    /// # Panics
    /// If there is not one vertex for each.
    #[must_use]
    pub fn with_vertices<W: Extensor<Slots = (), Coef = f64>>(
        &self,
        vertices: Field<W>,
    ) -> Mesh<W> {
        assert_eq!(vertices.len(), self.vertices.len(), "one vertex for each");
        Mesh {
            vertices,
            faces: self.faces.clone(),
            edges: self.edges.clone(),
            face_edges: self.face_edges.clone(),
            face_edge_orientation: self.face_edge_orientation.clone(),
            complex: self.complex.clone(),
        }
    }

    /// `[E, V]`: the difference along each edge, its higher vertex less its lower.
    #[must_use]
    pub fn d0(&self) -> SparseMap<ScalarOf<V>> {
        self.complex.d(0)
    }

    /// `[F, E]`: the oriented circulation around each face.
    #[must_use]
    pub fn d1(&self) -> SparseMap<ScalarOf<V>> {
        self.complex.d(1)
    }

    /// `[E, V]`: the mean of each edge's two vertices.
    #[must_use]
    pub fn edge_means(&self) -> SparseMap<ScalarOf<V>> {
        self.complex.means(1)
    }

    /// `[F, V]`: the mean of each face's three corners.
    #[must_use]
    pub fn face_means(&self) -> SparseMap<ScalarOf<V>> {
        self.complex.means(2)
    }

    /// The edges of only one face, in order.
    #[must_use]
    pub fn boundary_edges(&self) -> Vec<usize> {
        let mut faces_at = vec![0; self.edges.len()];
        for &e in self.face_edges.iter().flatten() {
            faces_at[e] += 1;
        }
        (0..self.edges.len())
            .filter(|&e| faces_at[e] == 1)
            .collect()
    }

    /// Whether each vertex is on the boundary: an end of an edge of only one face.
    #[must_use]
    pub fn boundary_vertices(&self) -> Vec<bool> {
        let mut on_boundary = vec![false; self.vertices.len()];
        for e in self.boundary_edges() {
            for v in self.edges[e] {
                on_boundary[v] = true;
            }
        }
        on_boundary
    }

    /// The number of edges at each vertex.
    #[must_use]
    pub fn valences(&self) -> Vec<usize> {
        let mut valence = vec![0; self.vertices.len()];
        for &v in self.edges.iter().flatten() {
            valence[v] += 1;
        }
        valence
    }

    /// `[V + E, V]`: one step of Loop's subdivision, from the vertices to those of the finer
    /// mesh, its vertices followed by one on each edge, with the `pinned` vertices staying where
    /// they are.
    ///
    /// An interior vertex of valence `n` keeps `1 - n β` of itself and takes `β` of each
    /// neighbour, with `β = (5/8 - (3/8 + cos(2π / n) / 4)²) / n`. An interior edge's vertex takes
    /// 3/8 of each end and 1/8 of each vertex across the edge: each face beside the edge gives
    /// 1/8 of the corner the edge faces and 3/16 of each of the other two. The boundary follows
    /// itself alone: a boundary vertex keeps 3/4 of itself and takes 1/8 of each neighbour along
    /// the boundary, and a boundary edge's vertex is the midpoint of its ends.
    ///
    /// # Panics
    /// If `pinned` does not name every vertex.
    #[must_use]
    pub fn loop_step(&self, pinned: &[bool]) -> SparseMap<ScalarOf<V>> {
        let n = self.vertices.len();
        assert_eq!(pinned.len(), n, "pinned: one flag for each vertex");
        let mut rim = vec![false; self.edges.len()];
        for e in self.boundary_edges() {
            rim[e] = true;
        }
        let on_rim = self.boundary_vertices();
        let valences = self.valences();
        let beta: Vec<f64> = valences.iter().map(|&n| loop_weight(n)).collect();
        let mut couplings = Vec::new();
        for v in 0..n {
            let own = match (pinned[v], on_rim[v]) {
                (true, _) => 1.0,
                (false, true) => 3.0 / 4.0,
                (false, false) => 1.0 - beta[v] * valences[v] as f64,
            };
            couplings.push((v, v, own));
        }
        // Each end's share in the other end, by the end's own rule.
        for (e, &[a, b]) in self.edges.iter().enumerate() {
            for (end, other) in [(a, b), (b, a)] {
                let share = match (pinned[end], on_rim[end]) {
                    (true, _) => 0.0,
                    (false, true) if rim[e] => 1.0 / 8.0,
                    (false, true) => 0.0,
                    (false, false) => beta[end],
                };
                couplings.push((end, other, share));
            }
        }
        // Each face's share in the vertex of the edge facing each corner, from each corner.
        for (f, corners) in self.faces.iter().enumerate() {
            for (j, &e) in self.face_edges[f].iter().enumerate() {
                for (k, &corner) in corners.iter().enumerate() {
                    let share = match (rim[e], j == k) {
                        (false, true) => 1.0 / 8.0,
                        (false, false) => 3.0 / 16.0,
                        (true, true) => 0.0,
                        (true, false) => 1.0 / 2.0,
                    };
                    couplings.push((n + e, corner, share));
                }
            }
        }
        couplings.retain(|&(_, _, share)| share != 0.0);
        SparseMap::new(
            couplings.iter().map(|c| scalar::<V>(c.2)).collect(),
            couplings.iter().map(|c| c.0).collect(),
            couplings.iter().map(|c| c.1).collect(),
            (n + self.edges.len(), n),
        )
    }

    /// The faces of the finer mesh of a Loop step: each face split in four at the vertices of
    /// its edges, a middle face through them and a face at each corner.
    #[must_use]
    pub fn split_faces(&self) -> Vec<[usize; 3]> {
        let n = self.vertices.len();
        let middles: Vec<[usize; 3]> = self.face_edges.iter().map(|e| e.map(|e| n + e)).collect();
        let corners = self.faces.iter().zip(&middles).flat_map(|(f, m)| {
            // The corner, then the middles of the edges after it and before it.
            [0, 1, 2].map(|k| [f[k], m[(k + 2) % 3], m[(k + 1) % 3]])
        });
        middles.iter().copied().chain(corners).collect()
    }

    /// The pinned vertices of the finer mesh of a Loop step: the same vertices; those of the
    /// edges are free.
    #[must_use]
    pub fn split_pins(&self, pinned: &[bool]) -> Vec<bool> {
        let free = core::iter::repeat_n(false, self.edges.len());
        pinned.iter().copied().chain(free).collect()
    }

    /// `[3]` face fields: each face's vertex at each of its corners.
    #[must_use]
    pub fn corners(&self) -> [Field<V>; 3] {
        [0, 1, 2].map(|k| {
            self.faces
                .iter()
                .map(|f| self.vertices.site(f[k]))
                .collect()
        })
    }

    /// `[F, E]`: couplings from each face to the edge facing each of its corners, through the
    /// cells of three face fields, one per corner.
    #[must_use]
    pub fn by_corner<C: Extensor>(&self, cells: &[Field<C>; 3]) -> SparseMap<C> {
        SparseMap::from_columns(&self.face_edges, self.edges.len(), |f, k| cells[k].site(f))
    }

    /// `[F, V]`: couplings from each face to the vertex at each of its corners, through the
    /// cells of three face fields, one per corner.
    #[must_use]
    pub fn at_corners<C: Extensor>(&self, cells: &[Field<C>; 3]) -> SparseMap<C> {
        SparseMap::from_columns(&self.faces, self.vertices.len(), |f, k| cells[k].site(f))
    }
}

impl<V> Mesh<V>
where
    V: Extensor<Slots = (), Coef = f64> + Add<Output = V> + Sub<Output = V>,
    ScalarOf<V>: Mul<V, Output = V>,
{
    /// The finer mesh of one Loop step, with the `pinned` vertices staying where they are: the
    /// step applied to the vertices, on the split faces.
    #[must_use]
    pub fn subdivide_loop(&self, pinned: &[bool]) -> Self {
        Mesh::new(&self.loop_step(pinned) * &self.vertices, self.split_faces())
    }

    /// `[E]`: each edge's head less its tail.
    #[must_use]
    pub fn edge_vectors(&self) -> Field<V> {
        &self.d0() * &self.vertices
    }

    /// `[E]`: the midpoint of each edge.
    #[must_use]
    pub fn edge_midpoints(&self) -> Field<V> {
        &self.edge_means() * &self.vertices
    }

    /// `[F]`: the centroid of each face.
    #[must_use]
    pub fn face_centers(&self) -> Field<V> {
        &self.face_means() * &self.vertices
    }

    /// `[3]` face fields: the edge facing each corner, counter-clockwise, from the corner after
    /// it to the corner after that.
    #[must_use]
    pub fn triangle_edges(&self) -> [Field<V>; 3] {
        let corners = self.corners();
        [0, 1, 2].map(|k| &corners[(k + 2) % 3] - &corners[(k + 1) % 3])
    }
}

/// Loop's `β` for a vertex of valence `n`, `(5/8 - (3/8 + cos(2π / n) / 4)²) / n`, the cosine the
/// first coordinate of the direction a turn of `2π / n` takes `e1` to.
fn loop_weight(n: usize) -> f64 {
    let n = n as f64;
    let cosine = direction(core::f64::consts::TAU / n).e1();
    (5.0 / 8.0 - (3.0 / 8.0 + cosine / 4.0).powi(2)) / n
}

/// The edges after and before each corner's facing edge: the two sides of the corner, each
/// leaving it.
fn sides_at(edges: &[Field<Vector3>; 3], k: usize) -> (&Field<Vector3>, &Field<Vector3>) {
    (&edges[(k + 1) % 3], &edges[(k + 2) % 3])
}

/// Each face's bivector, `e0 ^ e1` of two of its edges: its plane, twice its area large.
fn face_bivectors(edges: &[Field<Vector3>; 3]) -> Field<Bivector3> {
    edges[0].zip_with(&edges[1], |a, b| a ^ b)
}

impl Mesh<Vector3> {
    /// `[F]`: each face's unit plane, counter-clockwise seen from outside.
    #[must_use]
    pub fn face_planes(&self) -> Field<Bivector3> {
        face_bivectors(&self.triangle_edges()).map(|b| b.normalized().into_inner())
    }

    /// `[F]`: each face's area.
    #[must_use]
    pub fn triangle_areas(&self) -> Field<Scalar3> {
        face_bivectors(&self.triangle_edges()).map(|b| Scalar3::new(b.norm() / 2.0))
    }

    /// The mesh's metric (`gax::dec`): volumes, circumcentric duals, Hodge stars.
    #[must_use]
    pub fn geometry(&self) -> Geometry<f64, 3> {
        Geometry::from_parts(&self.complex, &self.vertices)
    }

    /// `[E]`: dual over primal edge length, half the cotangents of the angles facing each edge:
    /// the diagonal Hodge star on edges.
    #[must_use]
    pub fn edge_ratio(&self) -> Field<Scalar3> {
        self.geometry()
            .star::<Scalar3>(1)
            .couplings()
            .map(|(_, _, c)| c)
            .collect()
    }

    /// `[V, V]`: the cotangent Laplacian `~d0 H d0`, `H` the edge ratios on the diagonal. For a
    /// field `x` linear on each face, `x · L x` is its squared gradient integrated over the faces.
    #[must_use]
    pub fn laplacian(&self) -> SparseMap<Scalar3> {
        self.geometry().laplacian_form(0)
    }

    /// `[V]`: the unit normal at each vertex, the mean of its faces' normals weighted by their
    /// areas.
    #[must_use]
    pub fn vertex_normals(&self) -> Field<Normal3> {
        let face_normals = face_bivectors(&self.triangle_edges()).map(|b| b.dual());
        (&self.face_means().reverse() * &face_normals).map(|n| n.normalized().into_inner())
    }

    /// `[V]`: a third of the area of each triangle at the vertex.
    #[must_use]
    pub fn vertex_areas(&self) -> Field<Scalar3> {
        &self.face_means().reverse() * &self.triangle_areas()
    }

    /// `[3]` face fields: the cosine of each corner's angle, which a conformal map keeps.
    #[must_use]
    pub fn corner_cosines(&self) -> [Field<Scalar3>; 3] {
        let edges = self.triangle_edges();
        [0, 1, 2].map(|k| {
            let (after, before) = sides_at(&edges, k);
            after.zip_with(before, |a, b| -(a | b) * (1.0 / (a.norm() * b.norm())))
        })
    }

    /// `[3]` face fields: the gradient of each corner's hat function on the face, which points
    /// across the edge the corner faces: that edge turned a quarter turn in the face's plane,
    /// over twice the face's area.
    #[must_use]
    pub fn hat_gradients(&self) -> [Field<Vector3>; 3] {
        let (planes, areas) = (self.face_planes(), self.triangle_areas());
        self.triangle_edges().map(|edges| {
            let across = edges.zip_with(&planes, |e, p| e | p);
            across.zip_with(&areas, |g, a| g * (0.5 / a.s()))
        })
    }

    /// `[F, E]`, cells `Vector <- Scalar`: the Whitney edge integrals evaluated at the face
    /// centroids, a vector on each face from a number on each edge.
    #[must_use]
    pub fn reconstruction(&self) -> SparseMap<Vector<(Scalar,), f64>> {
        let gradients = self.hat_gradients();
        // All three hat functions are a third at the centroid; the difference of two gradients
        // reconstructs an oriented edge integral, with the mesh edge's sign.
        let cells = [0, 1, 2].map(|k| {
            let difference = &gradients[(k + 2) % 3] - &gradients[(k + 1) % 3];
            Field::from_fn(self.faces.len(), |f| {
                difference.site(f) * (self.face_edge_orientation[f][k] / 3.0) * Scalar::slot()
            })
        });
        self.by_corner(&cells)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type Plane = Bivector3;

    fn disks(radius: f64) -> [Mesh; 3] {
        [
            Mesh::disk(radius, 8, 48),
            Mesh::triangular_disk(radius, 10),
            Mesh::concentric_disk(&[0.2 * radius, 0.85 * radius, radius], 10),
        ]
    }

    /// The squared size of a field: the sum of its squared coefficients.
    fn squared_size<M: Extensor<Coef = f64>>(field: &Field<M>) -> f64 {
        field.dot(field)
    }

    /// The topology and the incidence maps as this module built them before `gax::mesh`
    /// (numga's `mesh.py`), kept to check that the move changed no number: the edges, each
    /// face's edges and their orientations, and the couplings of `d0`, `d1` and the means, bit
    /// for bit.
    #[test]
    fn gax_mesh_keeps_every_number() {
        type Couplings = Vec<(usize, usize, u64)>;
        fn couplings(m: &SparseMap<Scalar3>) -> Couplings {
            m.couplings()
                .map(|(r, c, x)| (r, c, x.s().to_bits()))
                .collect()
        }
        /// The ends of the edge facing each corner of the face, counter-clockwise.
        fn facing([a, b, c]: [usize; 3]) -> [[usize; 2]; 3] {
            [[b, c], [c, a], [a, b]]
        }
        type Reference = (
            Vec<[usize; 2]>,
            Vec<[usize; 3]>,
            Vec<[f64; 3]>,
            [Couplings; 4],
        );
        fn reference(mesh: &Mesh) -> Reference {
            let sorted = |[a, b]: [usize; 2]| [a.min(b), a.max(b)];
            let mut edges: Vec<[usize; 2]> = mesh
                .faces
                .iter()
                .flat_map(|&f| facing(f).map(sorted))
                .collect();
            edges.sort_unstable();
            edges.dedup();
            let index = |ends: [usize; 2]| edges.binary_search(&sorted(ends)).unwrap();
            let face_edges: Vec<[usize; 3]> =
                mesh.faces.iter().map(|&f| facing(f).map(index)).collect();
            let orientation: Vec<[f64; 3]> = mesh
                .faces
                .iter()
                .map(|&f| facing(f).map(|[a, b]| if a < b { 1.0 } else { -1.0 }))
                .collect();
            let n = mesh.vertices.len();
            let d0 =
                SparseMap::from_columns(&edges, n, |_, end| scalar::<Vector3>([-1.0, 1.0][end]));
            let d1 = SparseMap::from_columns(&face_edges, edges.len(), |f, k| {
                scalar::<Vector3>(orientation[f][k])
            });
            let em = SparseMap::from_columns(&edges, n, |_, _| scalar::<Vector3>(0.5));
            let fm = SparseMap::from_columns(&mesh.faces, n, |_, _| scalar::<Vector3>(1.0 / 3.0));
            let maps = [
                couplings(&d0),
                couplings(&d1),
                couplings(&em),
                couplings(&fm),
            ];
            (edges, face_edges, orientation, maps)
        }
        let coarse = Mesh::disk(1.0, 3, 9);
        let finer = coarse.subdivide_loop(&vec![false; coarse.vertices.len()]);
        for mesh in disks(1.0).into_iter().chain([finer]) {
            let (edges, face_edges, orientation, maps) = reference(&mesh);
            assert_eq!(mesh.edges, edges);
            assert_eq!(mesh.face_edges, face_edges);
            assert_eq!(mesh.face_edge_orientation, orientation);
            let now = [
                couplings(&mesh.d0()),
                couplings(&mesh.d1()),
                couplings(&mesh.edge_means()),
                couplings(&mesh.face_means()),
            ];
            assert_eq!(now, maps);
        }
    }

    /// The library's metric of meshes (`gax::dec`) gives the numbers this module built by hand:
    /// the half cotangents are the diagonal Hodge star on edges, the vertex areas the row sums of
    /// the Galerkin mass on vertices, and the reconstruction the Whitney map at the centroids.
    #[test]
    fn gax_dec_gives_the_metric() {
        use gax::dec::Geometry;
        for mesh in disks(1.0) {
            let lib = gax::mesh::Mesh::new(mesh.vertices.clone(), &mesh.faces).unwrap();
            let g = Geometry::new(&lib);
            let close = |a: f64, b: f64| (a - b).abs() < 1e-12 * (1.0 + a.abs());
            let star: Vec<f64> = g
                .star::<Scalar3>(1)
                .couplings()
                .map(|(_, _, c)| c.s())
                .collect();
            // The edge ratios as this module built them by hand: half the cotangent at each
            // corner, its sides' inner product over their wedge, summed onto the edge it faces.
            let edges = mesh.triangle_edges();
            let half_cotangents = [0, 1, 2].map(|k| {
                let (after, before) = sides_at(&edges, k);
                after.zip_with(before, |a, b| -(a | b) * (0.5 / (a ^ b).norm()))
            });
            let ones = Field::from_fn(mesh.faces.len(), |_| Scalar3::new(1.0));
            let by_hand = &mesh.by_corner(&half_cotangents).reverse() * &ones;
            for (h, x) in by_hand.sites().iter().zip(&star) {
                assert!(close(h.s(), *x), "{} {x}", h.s());
            }
            let mut lumped = vec![0.0; mesh.vertices.len()];
            for (r, _, c) in g.mass::<Scalar3>(0).couplings() {
                lumped[r] += c.s();
            }
            for (a, x) in mesh.vertex_areas().sites().iter().zip(&lumped) {
                assert!(close(a.s(), *x), "{} {x}", a.s());
            }
            let cochain: Field<Scalar3> = (0..mesh.edges.len())
                .map(|e| Scalar3::new(((e * 37) % 11) as f64 * 0.2 - 1.0))
                .collect();
            let reconstructed = mesh.reconstruction().apply(&cochain);
            for f in 0..mesh.faces.len() {
                let w = g.whitney_1(&lib, &cochain, f, &[1.0 / 3.0; 3]);
                let r = reconstructed.site(f);
                assert!(
                    (w - r).c.iter().all(|d| d.abs() < 1e-12),
                    "face {f}: {w:?} {r:?}"
                );
            }
        }
    }

    #[test]
    fn affine_field_reconstruction_and_energy_follow_a_turned_mesh() {
        // numga's `mv.xz * 0.31`: `e13 = -e31`.
        let turn = Plane::new(0.0, -0.31, 0.0).exp();
        for mesh in disks(1.7) {
            let mesh = mesh.with_vertices(mesh.vertices.map(|v| turn >> v));
            let gradient = turn >> Vector3::new(1.0, 2.0, 0.0);
            let potential = mesh.vertices.map(|v| v | gradient);
            let differences = &mesh.d0() * &potential;

            // Integrating a gradient around a triangle vanishes, and the edge integrals
            // reconstruct the same constant vector on every face, after turning the whole mesh.
            assert!(
                (&mesh.d1() * &mesh.d0())
                    .couplings()
                    .all(|(_, _, c)| c.s() == 0.0)
            );
            assert!(squared_size(&(&mesh.d1() * &differences)) < 1e-24);
            let reconstructed = mesh.reconstruction().apply(&differences);
            assert!(reconstructed.iter().all(|g| (g - gradient).norm() < 1e-12));

            // The cotangent Hodge measures the energy of integrating the squared gradient.
            let edge_energy = mesh
                .edge_ratio()
                .zip_with(&differences, |w, d| w * d * d)
                .sum();
            let face_energy = mesh.triangle_areas().sum() * gradient.norm_squared();
            assert!((edge_energy - face_energy).s().abs() < 1e-11);
        }
    }

    #[test]
    fn boundary_integral_matches_oriented_face_area() {
        let turn = Plane::new(-0.24, 0.0, 0.0).exp();
        for mesh in disks(2.0) {
            // numga's `turn >> mesh.vertices + mv.x`: the shift first, by precedence.
            let shift = Vector3::new(1.0, 0.0, 0.0);
            let mesh = mesh.with_vertices(mesh.vertices.map(|v| turn >> (v + shift)));

            // Interior edges cancel in the sum of face boundaries. The surviving edges enclose
            // the same oriented area as the faces.
            let ones = Field::from_fn(mesh.faces.len(), |_| Scalar3::new(1.0));
            let boundary = &mesh.d1().reverse() * &ones;
            let swept: Field<Plane> = mesh
                .edges
                .iter()
                .map(|&[tail, head]| mesh.vertices.site(tail) ^ mesh.vertices.site(head))
                .collect();
            let boundary_area = boundary.zip_with(&swept, |b, a| b * a).sum() * 0.5;
            let face_area = mesh
                .triangle_areas()
                .zip_with(&mesh.face_planes(), |a, p| a * p)
                .sum();
            assert!((boundary_area - face_area).norm() < 1e-11);
            let nonzero: Vec<usize> = (0..boundary.len())
                .filter(|&e| boundary.site(e).s() != 0.0)
                .collect();
            assert_eq!(nonzero, mesh.boundary_edges());
        }
    }

    /// Every face in the xy plane, counter-clockwise from above.
    pub(super) fn assert_flat(mesh: &Mesh, tolerance: f64) {
        let xy = Plane::new(0.0, 0.0, 1.0);
        assert!(
            mesh.face_planes()
                .iter()
                .all(|p| (p - xy).norm() < tolerance)
        );
    }

    #[test]
    fn the_loop_step_keeps_constants_and_the_mesh_flat() {
        // Every row of the step sums to one, so a constant stays and a flat disk stays flat.
        let mesh = Mesh::disk(1.0, 3, 12);
        let pinned = vec![false; mesh.vertices.len()];
        let ones = Field::from_fn(mesh.vertices.len(), |_| Scalar3::new(1.0));
        assert!(
            (&mesh.loop_step(&pinned) * &ones)
                .iter()
                .all(|s| (s.s() - 1.0).abs() < 1e-14)
        );
        let finer = mesh.subdivide_loop(&pinned);
        assert_eq!(finer.faces.len(), 4 * mesh.faces.len());
        assert_flat(&finer, 1e-12);
    }
}
