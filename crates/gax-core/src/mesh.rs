//! Simplicial complexes and meshes (ADR-049): the topology of a mesh of any dimension, its
//! boundary maps as field maps, and its vertices.
//!
//! A [`Topology`] is built from its top simplices, `N` vertex indices each: triangles
//! for a surface (`N = 3`), tetrahedra for a solid (`N = 4`). It derives every face of every
//! dimension: the `k`-simplices for `k` from 0 (the vertices) to `N − 1` (the top simplices,
//! kept in the order and orientation given). Below the top, a simplex is stored with its
//! vertices in increasing order, and the simplices of each dimension in lexicographic order.
//!
//! Quantities on a mesh are [`Field`]s over its simplices of one dimension, and its boundary maps
//! `d(k)`, from the `k`-simplices to the `(k+1)`-simplices, are [`SparseMap`]s with cells `±1`:
//! the discrete exterior derivative, with `d(k+1) ∘ d(k) = 0`. A [`Mesh`] is a topology with a
//! field of vertices of any kind, a VGA vector or a PGA point; `dec::Geometry` adds its metric.
//!
//! ```
//! use gax::field::Field;
//! use gax::mesh::Mesh;
//! use gax::vga3d::Vector;
//!
//! // A square of two triangles, counter-clockwise.
//! let v = |x: f64, y: f64| Vector::<(), f64>::new(x, y, 0.0);
//! let vertices: Field<_> = [v(0.0, 0.0), v(1.0, 0.0), v(1.0, 1.0), v(0.0, 1.0)].into_iter().collect();
//! let square = Mesh::new(vertices, &[[0, 1, 2], [0, 2, 3]]).unwrap();
//! let c = square.topology();
//! assert_eq!((c.len(0), c.len(1), c.len(2)), (4, 5, 2));
//! assert_eq!(c.simplex(1, 1), &[0, 2]); // the diagonal
//! assert_eq!(c.euler_characteristic(), 1); // a disk
//! assert_eq!(c.betti_numbers(), [1, 0, 0]);
//! assert_eq!(c.boundary(), [0, 2, 3, 4]); // all edges but the diagonal
//! // The circulation around each face of the differences along the edges is zero.
//! let curl_grad = &square.d(1) * &(&square.d(0) * &square.vertices);
//! assert!(curl_grad.sites().iter().all(|x| x.c == [0.0; 3]));
//! ```

extern crate std;

use crate::coef::{Coef, Real};
use crate::field::{Field, SparseMap};
use crate::fp::Fp;
use crate::kind::{Extensor, Kind};
use crate::simplex::Measure;
use core::fmt;
use core::ops::Neg;
use std::collections::{BTreeMap, VecDeque};
use std::vec::Vec;

/// Why a list of top simplices is not a complex.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MeshError {
    /// A simplex names a vertex that is not there.
    VertexOutOfRange {
        /// The simplex.
        simplex: usize,
        /// The vertex it names.
        vertex: usize,
        /// The number of vertices.
        vertices: usize,
    },
    /// A simplex names a vertex twice.
    RepeatedVertex {
        /// The simplex.
        simplex: usize,
        /// The vertex it repeats.
        vertex: usize,
    },
    /// Two top simplices have the same vertices.
    DuplicateSimplex {
        /// The first.
        first: usize,
        /// The second.
        second: usize,
    },
}

impl fmt::Display for MeshError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            MeshError::VertexOutOfRange {
                simplex,
                vertex,
                vertices,
            } => write!(
                f,
                "simplex {simplex} names vertex {vertex}, outside the {vertices} vertices"
            ),
            MeshError::RepeatedVertex { simplex, vertex } => {
                write!(f, "simplex {simplex} names vertex {vertex} twice")
            }
            MeshError::DuplicateSimplex { first, second } => {
                write!(f, "simplices {first} and {second} have the same vertices")
            }
        }
    }
}

impl std::error::Error for MeshError {}

/// The topology of a simplicial complex: the top simplices of `N` vertices each and all their
/// faces, the simplices of every dimension with their incidences and boundary maps. It holds the
/// combinatorics only: a [`Mesh`] adds the vertices' positions, `dec::Geometry` the metric.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Topology<const N: usize> {
    /// `simplices[k]`: the `k`-simplices, `k + 1` vertices each, one after another.
    simplices: Vec<Vec<usize>>,
    /// `faces[k]` for `k ≥ 1`: for each `k`-simplex, the `(k−1)`-face opposite each of its
    /// vertices in turn, with the orientation the simplex induces on it (`±1`).
    faces: Vec<Vec<(usize, i8)>>,
    /// The top simplices' vertices sorted, in lexicographic order, with each one's index: to find
    /// a top simplex by its vertices.
    tops: Vec<(Vec<usize>, usize)>,
}

/// The sign of the permutation that sorts `v`.
fn parity(v: &[usize]) -> i8 {
    let inversions = (0..v.len())
        .flat_map(|i| (i + 1..v.len()).map(move |j| (i, j)))
        .filter(|&(i, j)| v[i] > v[j])
        .count();
    if inversions.is_multiple_of(2) { 1 } else { -1 }
}

/// `s` without its `i`-th vertex.
fn without(s: &[usize], i: usize) -> Vec<usize> {
    s.iter()
        .enumerate()
        .filter(|&(j, _)| j != i)
        .map(|(_, &v)| v)
        .collect()
}

impl<const N: usize> Topology<N> {
    /// The topology of `vertices` vertices (some may be in no simplex) and the top simplices
    /// `top`, each oriented by the order of its vertices.
    ///
    /// # Errors
    /// A [`MeshError`] if a simplex names a vertex outside `vertices` or twice, or two top
    /// simplices have the same vertices.
    pub fn new(vertices: usize, top: &[[usize; N]]) -> Result<Self, MeshError> {
        const { assert!(N >= 1, "a simplex has at least one vertex") };
        for (s, simplex) in top.iter().enumerate() {
            for (i, &v) in simplex.iter().enumerate() {
                if v >= vertices {
                    return Err(MeshError::VertexOutOfRange {
                        simplex: s,
                        vertex: v,
                        vertices,
                    });
                }
                if simplex[..i].contains(&v) {
                    return Err(MeshError::RepeatedVertex {
                        simplex: s,
                        vertex: v,
                    });
                }
            }
        }
        let mut sorted: Vec<([usize; N], usize)> = top
            .iter()
            .enumerate()
            .map(|(s, simplex)| {
                let mut key = *simplex;
                key.sort_unstable();
                (key, s)
            })
            .collect();
        sorted.sort_unstable();
        if let Some(w) = sorted.windows(2).find(|w| w[0].0 == w[1].0) {
            return Err(MeshError::DuplicateSimplex {
                first: w[0].1,
                second: w[1].1,
            });
        }

        let mut simplices: Vec<Vec<usize>> = (0..N).map(|_| Vec::new()).collect();
        simplices[0] = (0..vertices).collect();
        simplices[N - 1] = top.iter().flatten().copied().collect();
        // Each dimension's faces, from the one above: sorted, in lexicographic order.
        for k in (1..N.saturating_sub(1)).rev() {
            let mut faces: Vec<Vec<usize>> = simplices[k + 1]
                .chunks_exact(k + 2)
                .flat_map(|s| {
                    (0..k + 2).map(move |i| {
                        let mut f = without(s, i);
                        f.sort_unstable();
                        f
                    })
                })
                .collect();
            faces.sort_unstable();
            faces.dedup();
            simplices[k] = faces.into_iter().flatten().collect();
        }

        // Each simplex's faces, opposite each vertex in turn, with the induced orientation.
        let mut faces: Vec<Vec<(usize, i8)>> = (0..N).map(|_| Vec::new()).collect();
        for k in 1..N {
            let below: Vec<&[usize]> = simplices[k - 1].chunks_exact(k).collect();
            let mut out = Vec::with_capacity(simplices[k].len());
            for s in simplices[k].chunks_exact(k + 1) {
                for i in 0..=k {
                    let mut face = without(s, i);
                    let sign = if i.is_multiple_of(2) { 1 } else { -1 } * parity(&face);
                    face.sort_unstable();
                    let index = below
                        .binary_search(&face.as_slice())
                        .expect("every face of a simplex is in the topology");
                    out.push((index, sign));
                }
            }
            faces[k] = out;
        }
        let tops = sorted
            .into_iter()
            .map(|(key, s)| (key.to_vec(), s))
            .collect();
        Ok(Topology {
            simplices,
            faces,
            tops,
        })
    }

    /// The number of `k`-simplices.
    #[must_use]
    pub fn len(&self, k: usize) -> usize {
        self.simplices[k].len() / (k + 1)
    }

    /// Whether the topology has no vertex.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.simplices[0].is_empty()
    }

    /// The vertices of the `i`-th `k`-simplex.
    #[must_use]
    pub fn simplex(&self, k: usize, i: usize) -> &[usize] {
        &self.simplices[k][i * (k + 1)..(i + 1) * (k + 1)]
    }

    /// The `k`-simplices, in order.
    pub fn simplices(&self, k: usize) -> impl ExactSizeIterator<Item = &[usize]> {
        self.simplices[k].chunks_exact(k + 1)
    }

    /// The index of the `k`-simplex with these vertices, sorted (below the top), or in the order
    /// given (a top simplex, found by its sorted vertices too).
    #[must_use]
    pub fn index_of(&self, k: usize, vertices: &[usize]) -> Option<usize> {
        if k + 1 == N {
            let mut key = vertices.to_vec();
            key.sort_unstable();
            return self
                .tops
                .binary_search_by(|(t, _)| t.as_slice().cmp(&key))
                .ok()
                .map(|i| self.tops[i].1);
        }
        let n = self.len(k);
        let (mut lo, mut hi) = (0, n);
        while lo < hi {
            let mid = usize::midpoint(lo, hi);
            match self.simplex(k, mid).cmp(vertices) {
                core::cmp::Ordering::Less => lo = mid + 1,
                core::cmp::Ordering::Greater => hi = mid,
                core::cmp::Ordering::Equal => return Some(mid),
            }
        }
        None
    }

    /// The faces of the `i`-th `k`-simplex (`k ≥ 1`): the `(k−1)`-simplex opposite each of its
    /// vertices in turn, with the orientation it induces there (`±1`), so that its boundary is
    /// `Σ sign · face`.
    #[must_use]
    pub fn faces(&self, k: usize, i: usize) -> &[(usize, i8)] {
        &self.faces[k][i * (k + 1)..(i + 1) * (k + 1)]
    }

    /// The boundary map `d(k)` from fields on the `k`-simplices to fields on the
    /// `(k+1)`-simplices, cells `±1` of any scalar kind (a kind of one coefficient: another is a
    /// compile error): the discrete exterior derivative. Row
    /// `σ` holds `σ`'s faces with their orientations. An edge lists its ends in its order (the
    /// first `−1`, the second `+1`), a simplex of two or more dimensions the face opposite each
    /// vertex in turn.
    ///
    /// # Panics
    /// If `k + 1` is not a dimension of the topology.
    #[must_use]
    pub fn d<C: Extensor<Slots = ()>>(&self, k: usize) -> SparseMap<C> {
        const {
            assert!(<C::Kind as Kind>::N == 1, "the cells of d are scalars");
        }
        assert!(
            k + 1 < N,
            "d({k}) maps to {}-simplices, above the top of a topology of {N}-vertex simplices",
            k + 1
        );
        let cell = |sign: i8| {
            C::from_coeffs(<C::Kind as Kind>::arr_from_fn(|_| {
                C::Coef::from_i64(sign.into())
            }))
        };
        let triples = (0..self.len(k + 1)).flat_map(|r| {
            let faces = self.faces(k + 1, r);
            let n = faces.len();
            // An edge's faces (its vertices) in reverse, every other face in order.
            (0..n).map(move |i| (r, faces[if k == 0 { n - 1 - i } else { i }]))
        });
        let triples = triples.map(|(r, (face, sign))| (r, face, cell(sign)));
        SparseMap::from_triples(triples, (self.len(k + 1), self.len(k)))
    }

    /// The map from fields on the vertices to fields on the `k`-simplices that takes the mean
    /// over each simplex's vertices, in the simplex's order (cells `1/(k+1)`).
    #[must_use]
    pub fn means<C: Extensor<Slots = ()>>(&self, k: usize) -> SparseMap<C> {
        const {
            assert!(
                <C::Kind as Kind>::N == 1,
                "the cells of the means are scalars"
            );
        }
        let w = C::from_coeffs(<C::Kind as Kind>::arr_from_fn(|_| {
            C::Coef::from_ratio(1, k as i64 + 1)
        }));
        // The simplices' vertices, k + 1 to a row.
        let vertices = self.simplices[k].iter().enumerate();
        let triples = vertices.map(|(i, &v)| (i / (k + 1), v, w));
        SparseMap::from_triples(triples, (self.len(k), self.len(0)))
    }

    /// The number of `(k+1)`-simplices each `k`-simplex is a face of. For `k = 0`, the valences.
    #[must_use]
    pub fn coface_counts(&self, k: usize) -> Vec<usize> {
        let mut counts = std::vec![0; self.len(k)];
        if k + 1 < N {
            for &(face, _) in &self.faces[k + 1] {
                counts[face] += 1;
            }
        }
        counts
    }

    /// The boundary: the `(N−2)`-simplices that are a face of exactly one top simplex, in order.
    #[must_use]
    pub fn boundary(&self) -> Vec<usize> {
        if N < 2 {
            return Vec::new();
        }
        let counts = self.coface_counts(N - 2);
        (0..counts.len()).filter(|&f| counts[f] == 1).collect()
    }

    /// Whether every `(N−2)`-face is a face of at most two top simplices: a pseudomanifold,
    /// possibly with boundary. (Two surfaces meeting at one vertex pass: it checks the faces,
    /// not the stars of vertices.)
    #[must_use]
    pub fn is_pseudomanifold(&self) -> bool {
        N < 2 || self.coface_counts(N - 2).iter().all(|&c| c <= 2)
    }

    /// Whether the top simplices are consistently oriented: a pseudomanifold whose neighbours
    /// induce opposite orientations on each face they share.
    #[must_use]
    pub fn is_oriented(&self) -> bool {
        if N < 2 {
            return true;
        }
        let mut induced = std::vec![0i8; self.len(N - 2)];
        let mut count = std::vec![0u8; self.len(N - 2)];
        for &(face, sign) in &self.faces[N - 1] {
            count[face] += 1;
            match count[face] {
                1 => induced[face] = sign,
                2 if induced[face] == -sign => {}
                _ => return false,
            }
        }
        true
    }

    /// `Σ (−1)ᵏ` times the number of `k`-simplices.
    #[must_use]
    pub fn euler_characteristic(&self) -> i64 {
        (0..N)
            .map(|k| if k.is_multiple_of(2) { 1 } else { -1 } * self.len(k) as i64)
            .sum()
    }

    /// The Betti numbers `b₀ … b_{N−1}`: the dimensions of the homology (over the rationals),
    /// from exact ranks of the boundary maps over `ℤ/p`, `p = 2⁶¹ − 1`. The two agree unless the
    /// homology has torsion of order `p`. `b₀` counts the connected pieces; a closed surface of
    /// genus `g` has `(1, 2g, 1)`.
    #[must_use]
    pub fn betti_numbers(&self) -> Vec<usize> {
        let ranks: Vec<usize> = (0..N)
            .map(|k| if k + 1 < N { self.rank(k) } else { 0 })
            .collect();
        (0..N)
            .map(|k| self.len(k) - ranks[k] - if k > 0 { ranks[k - 1] } else { 0 })
            .collect()
    }

    /// The rank of `d(k)` over `ℤ/p`, by sparse elimination of its rows.
    fn rank(&self, k: usize) -> usize {
        let mut pivots: BTreeMap<usize, BTreeMap<usize, Fp>> = BTreeMap::new();
        for r in 0..self.len(k + 1) {
            let mut row: BTreeMap<usize, Fp> = BTreeMap::new();
            for &(face, sign) in self.faces(k + 1, r) {
                let entry = row.entry(face).or_insert(Fp::zero());
                *entry = *entry + Fp::from_i64(sign.into());
            }
            row.retain(|_, x| *x != Fp::zero());
            while let Some((&lead, &x)) = row.iter().next() {
                let Some(pivot) = pivots.get(&lead) else {
                    pivots.insert(lead, row);
                    break;
                };
                let factor = x / pivot[&lead];
                for (&c, &y) in pivot {
                    let entry = row.entry(c).or_insert(Fp::zero());
                    *entry = *entry - factor * y;
                }
                row.retain(|_, x| *x != Fp::zero());
            }
        }
        pivots.len()
    }
}

/// The scalar kind of `V`'s algebra, with `V`'s coefficients.
pub type ScalarOf<V> =
    <<<V as Extensor>::Kind as Kind>::Scalar as Kind>::Mv<(), <V as Extensor>::Coef>;

/// A topology with a field of vertices, of any kind.
#[derive(Clone, Debug, PartialEq)]
pub struct Mesh<V, const N: usize> {
    /// The vertices.
    pub vertices: Field<V>,
    topology: Topology<N>,
}

impl<V: Extensor<Slots = ()>, const N: usize> Mesh<V, N> {
    /// The mesh of the vertices and the top simplices.
    ///
    /// # Errors
    /// A [`MeshError`] if the top simplices do not make a complex on these vertices.
    pub fn new(vertices: Field<V>, top: &[[usize; N]]) -> Result<Self, MeshError> {
        let topology = Topology::new(vertices.len(), top)?;
        Ok(Mesh { vertices, topology })
    }

    /// The topology: the simplices and their incidences, without the positions.
    #[must_use]
    pub fn topology(&self) -> &Topology<N> {
        &self.topology
    }

    /// The same topology on other vertices, of any kind.
    ///
    /// # Panics
    /// If there is not one vertex for each.
    #[must_use]
    pub fn with_vertices<W: Extensor<Slots = ()>>(&self, vertices: Field<W>) -> Mesh<W, N> {
        assert_eq!(vertices.len(), self.vertices.len(), "one vertex for each");
        Mesh {
            vertices,
            topology: self.topology.clone(),
        }
    }

    /// The boundary map `d(k)` with scalar cells of the vertices' algebra
    /// ([`Topology::d`]).
    #[must_use]
    pub fn d(&self, k: usize) -> SparseMap<ScalarOf<V>> {
        self.topology.d(k)
    }

    /// The means over the `k`-simplices' vertices ([`Topology::means`]).
    #[must_use]
    pub fn means(&self, k: usize) -> SparseMap<ScalarOf<V>> {
        self.topology.means(k)
    }
}

/// The local corners of each `m`-simplex of an `n`-simplex (`m < n`), in numga's order: for
/// `m = 0` the corners themselves; for `m = n − 1` the facet facing each corner in turn (from
/// a triangle up; an edge's facets are its two ends); in between the `(m+1)`-subsets of the
/// corners in lexicographic order (a tetrahedron's edges `(0 1), (0 2), (0 3), (1 2), (1 3),
/// (2 3)`). Each list is increasing. [`Topology::incidence`] and [`Topology::oriented`] read the
/// simplices of a mesh in this order.
///
/// ```
/// use gax::mesh::corners;
/// assert_eq!(corners(2, 1), [vec![1, 2], vec![0, 2], vec![0, 1]]); // facing each corner
/// assert_eq!(corners(3, 1)[3], [1, 2]);
/// ```
///
/// # Panics
/// If `m` is not below `n`.
#[must_use]
pub fn corners(n: usize, m: usize) -> Vec<Vec<usize>> {
    assert!(m < n, "a {n}-simplex has no {m}-simplices below it");
    if m == 0 {
        (0..=n).map(|i| std::vec![i]).collect()
    } else if m == n - 1 {
        (0..=n)
            .map(|i| (0..=n).filter(|&j| j != i).collect())
            .collect()
    } else {
        local_subsets(n + 1, m + 1)
    }
}

/// The `k`-subsets of `0..n`, each increasing, in lexicographic order.
fn local_subsets(n: usize, k: usize) -> Vec<Vec<usize>> {
    if k == 0 {
        return std::vec![Vec::new()];
    }
    (0..n)
        .flat_map(|first| {
            local_subsets(n - first - 1, k - 1)
                .into_iter()
                .map(move |rest| {
                    core::iter::once(first)
                        .chain(rest.into_iter().map(|r| r + first + 1))
                        .collect()
                })
        })
        .collect()
}

/// For each `n`-simplex of a topology, its `m`-simplices in the order of [`corners`]`(n, m)`,
/// each with a sign: numga's batch of incidence maps (`mesh.incidence[n, m]`,
/// `mesh.oriented[n, m]`, `mesh.boundary[n, n − 1]`). Entry `j` of the batch reads the `j`-th
/// `m`-simplex of every `n`-simplex: as a field map from the `m`-simplices to the
/// `n`-simplices ([`map`](Self::map)), applied to a field ([`read`](Self::read)), or weighted
/// per simplex and summed over the entries into one map ([`assemble`](Self::assemble)), which
/// is how per-corner formulas (a triangle's gradients, its Whitney forms) become field maps.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Incidence {
    /// `sites[r * width + j]`: the `j`-th `m`-simplex of `n`-simplex `r`.
    sites: Vec<usize>,
    /// Its sign there, `±1`.
    signs: Vec<i8>,
    width: usize,
    inputs: usize,
}

impl Incidence {
    /// The number of `m`-simplices of each `n`-simplex.
    #[must_use]
    pub fn width(&self) -> usize {
        self.width
    }

    /// The number of `n`-simplices: the output sites of the batch's maps.
    #[must_use]
    pub fn outputs(&self) -> usize {
        self.sites.len() / self.width
    }

    /// The number of `m`-simplices: the input sites of the batch's maps.
    #[must_use]
    pub fn inputs(&self) -> usize {
        self.inputs
    }

    /// The index of the `j`-th `m`-simplex of `n`-simplex `r`.
    #[must_use]
    pub fn site(&self, r: usize, j: usize) -> usize {
        self.sites[r * self.width + j]
    }

    /// The sign of the `j`-th `m`-simplex of `n`-simplex `r`.
    #[must_use]
    pub fn sign(&self, r: usize, j: usize) -> i8 {
        self.signs[r * self.width + j]
    }

    /// The `m`-simplices of `n`-simplex `r`, in order.
    #[must_use]
    pub fn sites_of(&self, r: usize) -> &[usize] {
        &self.sites[r * self.width..(r + 1) * self.width]
    }

    /// Entry `j` as a field map from the `m`-simplices to the `n`-simplices, cells `±1` of a
    /// scalar kind: row `r` reads the `j`-th `m`-simplex of `r`, times its sign.
    #[must_use]
    pub fn map<C: Extensor<Slots = ()>>(&self, j: usize) -> SparseMap<C> {
        const {
            assert!(
                <C::Kind as Kind>::N == 1,
                "the cells of an incidence map are scalars"
            );
        }
        let cell = |sign: i8| {
            C::from_coeffs(<C::Kind as Kind>::arr_from_fn(|_| {
                C::Coef::from_i64(sign.into())
            }))
        };
        let triples = (0..self.outputs()).map(|r| (r, self.site(r, j), cell(self.sign(r, j))));
        SparseMap::from_triples(triples, (self.outputs(), self.inputs))
    }

    /// The batch weighted per simplex and summed: the field map whose row `r` takes the `j`-th
    /// `m`-simplex of `r` through `cell(r, j)` times its sign, for every `j` (numga's
    /// `(spdiag(cells) * batch).sum(axis=0)`). With a constant cell it is the sum of the batch:
    /// `d(n − 1)` for [`Topology::facets`].
    #[must_use]
    pub fn assemble<C>(&self, mut cell: impl FnMut(usize, usize) -> C) -> SparseMap<C>
    where
        C: Copy + Neg<Output = C>,
    {
        let mut triples = Vec::with_capacity(self.sites.len());
        for r in 0..self.outputs() {
            for j in 0..self.width {
                let c = cell(r, j);
                let c = if self.sign(r, j) < 0 { -c } else { c };
                triples.push((r, self.site(r, j), c));
            }
        }
        SparseMap::from_triples(triples, (self.outputs(), self.inputs))
    }

    /// Entry `j` applied to a field on the `m`-simplices, unsigned: the value at each
    /// `n`-simplex's `j`-th `m`-simplex.
    ///
    /// # Panics
    /// If the field is not one value per `m`-simplex.
    #[must_use]
    #[track_caller]
    pub fn pick<V: Extensor>(&self, j: usize, field: &Field<V>) -> Field<V> {
        assert_eq!(field.len(), self.inputs, "one value per input simplex");
        (0..self.outputs())
            .map(|r| field.site(self.site(r, j)))
            .collect()
    }

    /// Entry `j` applied to a field on the `m`-simplices, signed (numga's `batch[j] * field`).
    ///
    /// # Panics
    /// If the field is not one value per `m`-simplex.
    #[must_use]
    #[track_caller]
    pub fn read<V: Extensor + Neg<Output = V>>(&self, j: usize, field: &Field<V>) -> Field<V> {
        assert_eq!(field.len(), self.inputs, "one value per input simplex");
        (0..self.outputs())
            .map(|r| {
                let v = field.site(self.site(r, j));
                if self.sign(r, j) < 0 { -v } else { v }
            })
            .collect()
    }

    /// Every entry applied to a field, signed: `[read(j) for j]`.
    #[must_use]
    #[track_caller]
    pub fn read_all<V: Extensor + Neg<Output = V>>(&self, field: &Field<V>) -> Vec<Field<V>> {
        (0..self.width).map(|j| self.read(j, field)).collect()
    }
}

/// The batches of incidences, the boundary and interior, orientation, and the first homology.
impl<const N: usize> Topology<N> {
    /// The indices of the `m`-simplices of every `n`-simplex, in the order of [`corners`].
    fn local(&self, n: usize, m: usize) -> Vec<usize> {
        assert!(
            m < n && n < N,
            "the {m}-simplices of {n}-simplices, in a topology of dimension {}",
            N - 1
        );
        let order = corners(n, m);
        let mut out = Vec::with_capacity(self.len(n) * order.len());
        for r in 0..self.len(n) {
            let simplex = self.simplex(n, r);
            if m == 0 {
                out.extend_from_slice(simplex);
            } else if m == n - 1 {
                out.extend(self.faces(n, r).iter().map(|&(face, _)| face));
            } else {
                out.extend(order.iter().map(|c| {
                    let mut vertices: Vec<usize> = c.iter().map(|&i| simplex[i]).collect();
                    vertices.sort_unstable();
                    self.index_of(m, &vertices).expect("a face of the topology")
                }));
            }
        }
        out
    }

    /// The `m`-simplices of each `n`-simplex (`m < n`), unsigned, in the order of
    /// [`corners`]`(n, m)` (numga's `mesh.incidence[n, m]`): a triangle's corners in its own
    /// order, the edge facing each of them, a tetrahedron's six edges by their corners.
    ///
    /// ```
    /// use gax::mesh::Topology;
    /// let t = Topology::new(4, &[[0, 1, 2], [0, 2, 3]]).unwrap();
    /// let edges = t.incidence(2, 1);
    /// // The second triangle's edge facing its first corner, 0, is (2 3).
    /// assert_eq!(t.simplex(1, edges.site(1, 0)), &[2, 3]);
    /// assert_eq!(t.incidence(2, 0).sites_of(1), &[0, 2, 3]);
    /// ```
    ///
    /// # Panics
    /// If `m` is not below `n` or `n` is above the dimension.
    #[must_use]
    pub fn incidence(&self, n: usize, m: usize) -> Incidence {
        let sites = self.local(n, m);
        Incidence {
            signs: std::vec![1; sites.len()],
            width: corners(n, m).len(),
            inputs: self.len(m),
            sites,
        }
    }

    /// The `m`-simplices of each `n`-simplex as [`incidence`](Self::incidence) lists them,
    /// signed `+1` where the stored `m`-simplex (below the top, its vertices increasing) runs as
    /// the `n`-simplex's own order runs it, `−1` where it runs the other way (numga's
    /// `mesh.oriented[n, m]`). A formula written in a simplex's own order (a Whitney form from a
    /// local tail to a local head) reaches the stored simplices through these signs.
    ///
    /// # Panics
    /// If `m` is not below `n` or `n` is above the dimension.
    #[must_use]
    pub fn oriented(&self, n: usize, m: usize) -> Incidence {
        let sites = self.local(n, m);
        let order = corners(n, m);
        let mut signs = Vec::with_capacity(sites.len());
        for r in 0..self.len(n) {
            let simplex = self.simplex(n, r);
            signs.extend(order.iter().map(|c| {
                let global: Vec<usize> = c.iter().map(|&i| simplex[i]).collect();
                parity(&global)
            }));
        }
        Incidence {
            signs,
            width: order.len(),
            inputs: self.len(m),
            sites,
        }
    }

    /// The facets of each `n`-simplex (`n ≥ 1`), signed by the orientation the simplex induces on
    /// them (numga's `mesh.boundary[n, n − 1]`): an edge's tail `−1` and head `+1`, a
    /// triangle's edge facing each corner `+1` where its winding runs along it, a tetrahedron's
    /// faces `+1` where they face outward. Summed it is `d(n − 1)`:
    /// `facets(n).assemble(|_, _| one) == d(n − 1)`.
    ///
    /// # Panics
    /// If `n` is zero or above the dimension.
    #[must_use]
    pub fn facets(&self, n: usize) -> Incidence {
        assert!(
            n >= 1 && n < N,
            "the facets of {n}-simplices, in a topology of dimension {}",
            N - 1
        );
        let (sites, signs) = if n == 1 {
            let sites = self.simplices[1].clone();
            let signs = (0..self.len(1)).flat_map(|_| [-1, 1]).collect();
            (sites, signs)
        } else {
            self.faces[n].iter().copied().unzip()
        };
        Incidence {
            sites,
            signs,
            width: n + 1,
            inputs: self.len(n - 1),
        }
    }

    /// The `k`-simplices on the boundary (`k ≤ N − 2`): the faces of the `(N−2)`-simplices that
    /// bound a single top simplex, in increasing order (numga's `exterior`). For `k = N − 2`,
    /// [`boundary`](Self::boundary).
    ///
    /// ```
    /// use gax::mesh::Topology;
    /// // A fan of four triangles around vertex 0: every vertex but the centre is on the rim.
    /// let fan = Topology::new(5, &[[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1]]).unwrap();
    /// assert_eq!(fan.exterior(0), [1, 2, 3, 4]);
    /// assert_eq!(fan.interior(0), [0]);
    /// assert_eq!(fan.interior(1).len(), 4); // the spokes
    /// ```
    ///
    /// # Panics
    /// If `k + 2 > N`.
    #[must_use]
    pub fn exterior(&self, k: usize) -> Vec<usize> {
        assert!(
            k + 2 <= N,
            "the boundary of a topology of dimension {} has no {k}-simplices",
            N - 1
        );
        let mut on_boundary = self.boundary();
        for j in (k + 1..=N - 2).rev() {
            let mut below: Vec<usize> = if j == 1 {
                on_boundary
                    .iter()
                    .flat_map(|&s| self.simplex(1, s).iter().copied())
                    .collect()
            } else {
                on_boundary
                    .iter()
                    .flat_map(|&s| self.faces(j, s).iter().map(|&(f, _)| f))
                    .collect()
            };
            below.sort_unstable();
            below.dedup();
            on_boundary = below;
        }
        on_boundary
    }

    /// The `k`-simplices off the boundary, in increasing order: all but
    /// [`exterior`](Self::exterior)'s.
    ///
    /// # Panics
    /// If `k + 2 > N`.
    #[must_use]
    pub fn interior(&self, k: usize) -> Vec<usize> {
        let outside = self.exterior(k);
        (0..self.len(k))
            .filter(|i| outside.binary_search(i).is_err())
            .collect()
    }

    /// The top simplices, each given as its vertices.
    fn tops(&self) -> Vec<[usize; N]> {
        self.simplices(N - 1)
            .map(|s| core::array::from_fn(|i| s[i]))
            .collect()
    }

    /// Each top simplex run the other way (its first two vertices swapped), in the same order:
    /// every orientation reversed, the numbering of every simplex kept.
    #[must_use]
    pub fn turned(&self) -> Self {
        let mut tops = self.tops();
        if N >= 2 {
            for s in &mut tops {
                s.swap(0, 1);
            }
        }
        Topology::new(self.len(0), &tops).expect("the same complex")
    }

    /// For each top simplex, whether to turn it so that neighbours agree, and its connected
    /// piece (across the `(N−2)`-faces): breadth first from each piece's first simplex, which
    /// keeps its sense.
    fn windings(&self) -> (Vec<bool>, Vec<usize>) {
        let tops = self.len(N - 1);
        if N < 2 {
            return (std::vec![false; tops], (0..tops).collect());
        }
        let top = N - 1;
        let mut owners: Vec<Vec<(usize, i8)>> = std::vec![Vec::new(); self.len(top - 1)];
        for s in 0..tops {
            for &(facet, sign) in self.faces(top, s) {
                owners[facet].push((s, sign));
            }
        }
        let mut flips: Vec<Option<bool>> = std::vec![None; tops];
        let mut piece = std::vec![0; tops];
        let mut pieces = 0;
        for root in 0..tops {
            if flips[root].is_some() {
                continue;
            }
            flips[root] = Some(false);
            piece[root] = pieces;
            let mut queue = VecDeque::from([root]);
            while let Some(s) = queue.pop_front() {
                let flipped = flips[s] == Some(true);
                for &(facet, sign) in self.faces(top, s) {
                    for &(other, other_sign) in &owners[facet] {
                        if other != s && flips[other].is_none() {
                            // Neighbours that run their shared facet the same way disagree.
                            flips[other] = Some(flipped ^ (sign == other_sign));
                            piece[other] = pieces;
                            queue.push_back(other);
                        }
                    }
                }
            }
            pieces += 1;
        }
        (flips.into_iter().map(|f| f == Some(true)).collect(), piece)
    }

    /// The same complex with some top simplices turned so that neighbours induce opposite
    /// orientations on every face they share ([`is_oriented`](Self::is_oriented)); each
    /// connected piece keeps the sense of its first top simplex, and every simplex keeps its
    /// number (numga's `coherent`). `None` if the complex is not orientable (a Möbius strip) or
    /// not a pseudomanifold.
    ///
    /// ```
    /// use gax::mesh::Topology;
    /// // A square of two triangles, the second wound the wrong way.
    /// let t = Topology::new(4, &[[0, 1, 2], [0, 3, 2]]).unwrap();
    /// assert!(!t.is_oriented());
    /// let c = t.coherent().unwrap();
    /// assert!(c.is_oriented());
    /// assert_eq!(c.simplex(2, 0), &[0, 1, 2]); // the first keeps its sense
    /// ```
    #[must_use]
    pub fn coherent(&self) -> Option<Self> {
        let (flips, _) = self.windings();
        let mut tops = self.tops();
        for (s, &flip) in tops.iter_mut().zip(&flips) {
            if flip {
                s.swap(0, 1);
            }
        }
        let wound = Topology::new(self.len(0), &tops).expect("the same complex");
        wound.is_oriented().then_some(wound)
    }

    /// Closed loops of vertices whose classes are a basis of the first homology `H₁` (over the
    /// rationals): `b₁` of them ([`betti_numbers`](Self::betti_numbers)), `2g` on a closed
    /// surface of genus `g`, one around each hole of a disk with holes, none on a sphere.
    ///
    /// Each loop is closed by one edge off a breadth-first spanning tree of the edges (from the
    /// lowest vertex of each piece, neighbours in increasing order) and runs through the tree:
    /// from that edge's second vertex up to where the two paths to the root meet, and down to
    /// its first vertex, the step from the last vertex back to the first being the edge itself
    /// (numga's `generators`). The edges are chosen off the triangles' boundaries: among the
    /// edges off the tree, those independent of the triangles' boundaries modulo the tree, by
    /// exact elimination over `ℤ/p` (on a surface it reduces to Eppstein's tree and cotree, in
    /// near-linear time). Within the classes the loops are not shortest.
    ///
    /// ```
    /// use gax::mesh::Topology;
    /// // A torus: a 4 × 4 grid of squares with opposite sides glued.
    /// let id = |i: usize, j: usize| (i % 4) * 4 + j % 4;
    /// let faces: Vec<[usize; 3]> = (0..4)
    ///     .flat_map(|i| (0..4).flat_map(move |j| {
    ///         [[id(i, j), id(i + 1, j), id(i + 1, j + 1)], [id(i, j), id(i + 1, j + 1), id(i, j + 1)]]
    ///     }))
    ///     .collect();
    /// let torus = Topology::new(16, &faces).unwrap();
    /// let loops = torus.cycles();
    /// assert_eq!(loops.len(), 2);
    /// // A gradient circulates by nothing around each loop.
    /// type S = gax::vga3d::Scalar<(), f64>;
    /// let f = gax::field::Field::from_fn(16, |v| S::new((v * v) as f64));
    /// let df = &torus.d::<S>(0) * &f;
    /// let around: gax::field::Field<S> = &torus.circulation::<S>(&loops) * &df;
    /// assert!(around.iter().all(|c| c.s() == 0.0));
    /// ```
    #[must_use]
    pub fn cycles(&self) -> Vec<Vec<usize>> {
        if N < 2 {
            return Vec::new();
        }
        let (nv, ne) = (self.len(0), self.len(1));
        let ends = |e: usize| {
            let s = self.simplex(1, e);
            (s[0], s[1])
        };
        // A breadth-first spanning forest of the edges.
        let mut neighbours: Vec<Vec<(usize, usize)>> = std::vec![Vec::new(); nv];
        for e in 0..ne {
            let (a, b) = ends(e);
            neighbours[a].push((b, e));
            neighbours[b].push((a, e));
        }
        for list in &mut neighbours {
            list.sort_unstable();
        }
        let mut parent: Vec<Option<usize>> = std::vec![None; nv];
        let mut reached = std::vec![false; nv];
        let mut in_tree = std::vec![false; ne];
        for root in 0..nv {
            if reached[root] {
                continue;
            }
            reached[root] = true;
            let mut queue = VecDeque::from([root]);
            while let Some(v) = queue.pop_front() {
                for &(w, e) in &neighbours[v] {
                    if !reached[w] {
                        reached[w] = true;
                        parent[w] = Some(v);
                        in_tree[e] = true;
                        queue.push_back(w);
                    }
                }
            }
        }
        // The triangles' boundaries modulo the tree, as rows on the edges off it, in reverse
        // breadth-first order across those edges: on a surface each triangle then meets the edge
        // it was reached by for the first time, which becomes its pivot without elimination.
        let mut pivot_of: Vec<Option<usize>> = std::vec![None; ne];
        if N >= 3 {
            let triangles = self.len(2);
            let mut beside: Vec<Vec<usize>> = std::vec![Vec::new(); ne];
            for t in 0..triangles {
                for &(e, _) in self.faces(2, t) {
                    if !in_tree[e] {
                        beside[e].push(t);
                    }
                }
            }
            let mut order = Vec::with_capacity(triangles);
            let mut seen_triangle = std::vec![false; triangles];
            for root in 0..triangles {
                if seen_triangle[root] {
                    continue;
                }
                seen_triangle[root] = true;
                let mut queue = VecDeque::from([root]);
                while let Some(t) = queue.pop_front() {
                    order.push(t);
                    for &(e, _) in self.faces(2, t) {
                        for &u in &beside[e] {
                            if !seen_triangle[u] {
                                seen_triangle[u] = true;
                                queue.push_back(u);
                            }
                        }
                    }
                }
            }
            eliminate_cycles(self, &in_tree, order.into_iter().rev(), &mut pivot_of);
        }
        let to_root = |mut v: usize| {
            let mut path = std::vec![v];
            while let Some(p) = parent[v] {
                path.push(p);
                v = p;
            }
            path
        };
        (0..ne)
            .filter(|&e| !in_tree[e] && pivot_of[e].is_none())
            .map(|e| {
                let (tail, head) = ends(e);
                let (mut up, mut down) = (to_root(head), to_root(tail));
                // Drop the common part above the lowest common ancestor, keeping it once.
                while up.len() > 1 && down.len() > 1 && up[up.len() - 2] == down[down.len() - 2] {
                    up.pop();
                    down.pop();
                }
                down.pop();
                up.extend(down.into_iter().rev());
                up
            })
            .collect()
    }

    /// The circulation of an edge cochain around closed loops of vertices: row `l` sums the
    /// values along loop `l` in its own order, the step from its last vertex back to its first
    /// included, each edge `+1` where the loop runs along it and `−1` against (cells of a scalar
    /// kind; numga's `circulation`, without the faces, which are `d(1)`). On a basis of `H₁`
    /// ([`cycles`](Self::cycles)) a closed cochain's circulations are its periods.
    ///
    /// # Panics
    /// If two consecutive vertices of a loop are not joined by an edge.
    #[must_use]
    pub fn circulation<C: Extensor<Slots = ()>>(&self, loops: &[Vec<usize>]) -> SparseMap<C> {
        const {
            assert!(
                <C::Kind as Kind>::N == 1,
                "the cells of a circulation are scalars"
            );
        }
        let cell =
            |x: i64| C::from_coeffs(<C::Kind as Kind>::arr_from_fn(|_| C::Coef::from_i64(x)));
        let mut triples = Vec::new();
        for (l, path) in loops.iter().enumerate() {
            for i in 0..path.len() {
                let (a, b) = (path[i], path[(i + 1) % path.len()]);
                let e = self
                    .index_of(1, &[a.min(b), a.max(b)])
                    .unwrap_or_else(|| panic!("loop {l} steps from {a} to {b} along no edge"));
                let along = self.simplex(1, e)[0] == a;
                triples.push((l, e, cell(if along { 1 } else { -1 })));
            }
        }
        SparseMap::from_triples(triples, (loops.len(), self.len(1)))
    }
}

/// Exact elimination of the triangles' boundaries modulo the spanning tree, rows in `order`,
/// marking each pivot edge with its row (ADR-058). A row takes as its pivot an edge no earlier
/// row met, with no elimination (a key above every earlier one), or else is reduced by the
/// rows whose pivots it holds, highest key first, and takes a remaining edge (a key below every
/// earlier one). Each kept row holds, besides its pivot, only the pivots of rows of lower keys,
/// so the reduction ends and the pivots are independent.
fn eliminate_cycles<const N: usize>(
    t: &Topology<N>,
    in_tree: &[bool],
    order: impl Iterator<Item = usize>,
    pivot_of: &mut [Option<usize>],
) {
    let mut rows: Vec<(BTreeMap<usize, Fp>, i64)> = Vec::new();
    let mut seen = std::vec![false; in_tree.len()];
    let (mut high, mut low) = (0i64, -1i64);
    for tri in order {
        let mut row: BTreeMap<usize, Fp> = BTreeMap::new();
        for &(e, sign) in t.faces(2, tri) {
            if !in_tree[e] {
                let entry = row.entry(e).or_insert(Fp::zero());
                *entry = *entry + Fp::from_i64(sign.into());
            }
        }
        row.retain(|_, x| *x != Fp::zero());
        if let Some(&fresh) = row.keys().find(|&&e| !seen[e]) {
            for &e in row.keys() {
                seen[e] = true;
            }
            pivot_of[fresh] = Some(rows.len());
            rows.push((row, high));
            high += 1;
            continue;
        }
        loop {
            let next = row
                .keys()
                .filter_map(|&e| pivot_of[e].map(|r| (rows[r].1, e, r)))
                .max();
            let Some((_, e, r)) = next else { break };
            let factor = row[&e] / rows[r].0[&e];
            for (&c, &y) in &rows[r].0 {
                let entry = row.entry(c).or_insert(Fp::zero());
                *entry = *entry - factor * y;
            }
            row.retain(|_, x| *x != Fp::zero());
        }
        if let Some(&e) = row.keys().next() {
            for &c in row.keys() {
                seen[c] = true;
            }
            pivot_of[e] = Some(rows.len());
            rows.push((row, low));
            low -= 1;
        }
    }
}

impl<V: Extensor<Slots = ()>, const N: usize> Mesh<V, N> {
    /// The same mesh with its top simplices wound coherently ([`Topology::coherent`]); `None`
    /// if it is not orientable.
    #[must_use]
    pub fn coherent(&self) -> Option<Self> {
        Some(Mesh {
            vertices: self.vertices.clone(),
            topology: self.topology.coherent()?,
        })
    }

    /// The same mesh with every orientation reversed ([`Topology::turned`]).
    #[must_use]
    pub fn turned(&self) -> Self {
        Mesh {
            vertices: self.vertices.clone(),
            topology: self.topology.turned(),
        }
    }
}

/// `Mesh::outward` for closed hypersurfaces of `$n`-vertex simplices, whose cones from a point
/// are `$m`-vertex simplices.
macro_rules! outward {
    ($($n:literal => $m:literal),*) => {$(
        impl<V> Mesh<V, $n>
        where
            V: Extensor<Slots = (), Coef: Real> + Measure<$m>,
            <V as Measure<$m>>::Blade: Extensor<Slots = (), Coef = <V as Extensor>::Coef>,
        {
            /// A closed hypersurface (a closed surface in space, a closed curve in the plane)
            /// wound coherently and outward, whatever sense its simplices come in (numga's
            /// `Mesh.closed`): each connected piece made [`coherent`](Self::coherent), then
            /// turned if the volume it encloses is negative, the sum of the measures of the
            /// cones its simplices span from one of its vertices (the model's [`Measure`], whose
            /// one coefficient is signed as VGA's pseudoscalar). `None` if it has a boundary or
            /// is not orientable.
            #[must_use]
            pub fn outward(&self) -> Option<Self> {
                const {
                    assert!(
                        <<<V as Measure<$m>>::Blade as Extensor>::Kind as Kind>::N == 1,
                        "the measure of a cone of the full dimension has one coefficient"
                    );
                }
                if !self.topology.boundary().is_empty() {
                    return None;
                }
                let wound = self.topology.coherent()?;
                let (_, piece) = wound.windings();
                let pieces = piece.iter().max().map_or(0, |&p| p + 1);
                let v = self.vertices.sites();
                let zero = <V as Extensor>::Coef::zero();
                let mut volume = std::vec![zero; pieces];
                let mut apex: Vec<Option<V>> = std::vec![None; pieces];
                for (s, simplex) in wound.simplices($n - 1).enumerate() {
                    let at = *apex[piece[s]].get_or_insert(v[simplex[0]]);
                    let cone: [V; $m] =
                        core::array::from_fn(|i| if i == 0 { at } else { v[simplex[i - 1]] });
                    let m = <V as Measure<$m>>::measure(&cone);
                    volume[piece[s]] = volume[piece[s]] + m.coeffs().as_ref()[0];
                }
                let mut tops = wound.tops();
                for (s, top) in tops.iter_mut().enumerate() {
                    if <V as Extensor>::Coef::all_lt(volume[piece[s]], zero) {
                        top.swap(0, 1);
                    }
                }
                let topology = Topology::new(v.len(), &tops).expect("the same complex");
                Some(Mesh {
                    vertices: self.vertices.clone(),
                    topology,
                })
            }
        }
    )*};
}
outward!(2 => 3, 3 => 4, 4 => 5);
