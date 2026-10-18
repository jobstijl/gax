//! Simplicial complexes and meshes (ADR-049): the topology of a mesh of any dimension, its
//! boundary maps as field maps, and its vertices.
//!
//! A [`SimplicialComplex`] is built from its top simplices, `N` vertex indices each: triangles
//! for a surface (`N = 3`), tetrahedra for a solid (`N = 4`). It derives every face of every
//! dimension: the `k`-simplices for `k` from 0 (the vertices) to `N − 1` (the top simplices,
//! kept in the order and orientation given). Below the top, a simplex is stored with its
//! vertices in increasing order, and the simplices of each dimension in lexicographic order.
//!
//! Quantities on a mesh are [`Field`]s over its simplices of one dimension, and its boundary maps
//! `d(k)`, from the `k`-simplices to the `(k+1)`-simplices, are [`SparseMap`]s with cells `±1`:
//! the discrete exterior derivative, with `d(k+1) ∘ d(k) = 0`. A [`Mesh`] is a complex with a
//! field of vertices of any kind, a VGA vector or a PGA point.
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
//! let c = square.complex();
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

use crate::coef::Coef;
use crate::field::{Field, SparseMap};
use crate::fp::Fp;
use crate::kind::{Extensor, Kind};
use core::fmt;
use std::collections::BTreeMap;
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

/// A simplicial complex: the top simplices of `N` vertices each, and all their faces.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SimplicialComplex<const N: usize> {
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

impl<const N: usize> SimplicialComplex<N> {
    /// The complex of `vertices` vertices (some may be in no simplex) and the top simplices
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
                        .expect("every face of a simplex is in the complex");
                    out.push((index, sign));
                }
            }
            faces[k] = out;
        }
        let tops = sorted
            .into_iter()
            .map(|(key, s)| (key.to_vec(), s))
            .collect();
        Ok(SimplicialComplex {
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

    /// Whether the complex has no vertex.
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
    /// If `k + 1` is not a dimension of the complex.
    #[must_use]
    pub fn d<C: Extensor<Slots = ()>>(&self, k: usize) -> SparseMap<C> {
        const {
            assert!(<C::Kind as Kind>::N == 1, "the cells of d are scalars");
        }
        assert!(
            k + 1 < N,
            "d({k}) maps to {}-simplices, above the top of a complex of {N}-vertex simplices",
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

/// A complex with a field of vertices, of any kind.
#[derive(Clone, Debug, PartialEq)]
pub struct Mesh<V, const N: usize> {
    /// The vertices.
    pub vertices: Field<V>,
    complex: SimplicialComplex<N>,
}

impl<V: Extensor<Slots = ()>, const N: usize> Mesh<V, N> {
    /// The mesh of the vertices and the top simplices.
    ///
    /// # Errors
    /// A [`MeshError`] if the top simplices do not make a complex on these vertices.
    pub fn new(vertices: Field<V>, top: &[[usize; N]]) -> Result<Self, MeshError> {
        let complex = SimplicialComplex::new(vertices.len(), top)?;
        Ok(Mesh { vertices, complex })
    }

    /// The complex.
    #[must_use]
    pub fn complex(&self) -> &SimplicialComplex<N> {
        &self.complex
    }

    /// The same complex on other vertices, of any kind.
    ///
    /// # Panics
    /// If there is not one vertex for each.
    #[must_use]
    pub fn with_vertices<W: Extensor<Slots = ()>>(&self, vertices: Field<W>) -> Mesh<W, N> {
        assert_eq!(vertices.len(), self.vertices.len(), "one vertex for each");
        Mesh {
            vertices,
            complex: self.complex.clone(),
        }
    }

    /// The boundary map `d(k)` with scalar cells of the vertices' algebra
    /// ([`SimplicialComplex::d`]).
    #[must_use]
    pub fn d(&self, k: usize) -> SparseMap<ScalarOf<V>> {
        self.complex.d(k)
    }

    /// The means over the `k`-simplices' vertices ([`SimplicialComplex::means`]).
    #[must_use]
    pub fn means(&self, k: usize) -> SparseMap<ScalarOf<V>> {
        self.complex.means(k)
    }
}
