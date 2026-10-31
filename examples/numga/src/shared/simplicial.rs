//! What numga's simplicial mesh (`examples/util/mesh.py`) adds to `gax::mesh`: the incidences of
//! the simplices of every dimension in numga's local order and signs, the index lists and
//! selection maps that read a mesh's boundary and interior, the subdivision and the constructors
//! some examples share (numga's torus and icosphere), coherent winding, and the graph and
//! homology of the edges (a breadth-first tree, the generators of the first homology, the
//! circulation around faces and loops). Shared by the surfaces ports of Whitney forms,
//! Dirac–Kähler fields, Regge calculus, frame fields, developability and connections; each
//! uses part of it.
//!
//! A mesh is a [`Topology`] of `N`-vertex top simplices with a [`Field`] of vertices of any
//! kind: VGA2D or VGA3D vectors. The topology already numbers every simplex as numga does (the
//! top simplices as given, every lower one by its sorted vertices, each dimension in
//! lexicographic order) and gives each face its orientation in the simplex it bounds; numga's
//! batches of incidence maps are [`Batch`]es here: for each `n`-simplex its `m`-simplices, in
//! numga's local order, each with a sign.
//!
//! * [`incidence`] (`mesh.incidence[n, m]`): unsigned. Entry `i` of the corners (`m = 0`) is
//!   corner `i` in the simplex's order; from triangles up, entry `i` of the facets
//!   (`m = n − 1`) is the facet facing corner `i`; in between (a tetrahedron's edges) the
//!   `m`-simplices come in the order of their corners, `(0 1), (0 2), (0 3), (1 2), (1 3), (2 3)`.
//!   An edge's corners are its tail and head, the lower vertex first.
//! * [`oriented`] (`mesh.oriented[n, m]`): signed `+1` where the stored `m`-simplex (its
//!   vertices increasing) runs as the `n`-simplex's own order runs it.
//! * [`boundary`] (`mesh.boundary[n, n − 1]`): the facets signed by their orientation in the
//!   simplex, an edge's tail `−1` and head `+1`, a face's edges `+1` where its winding runs
//!   along them. Summed over the batch it is `gax::mesh::Topology::d(n − 1)`.
//! * [`corners`] (`mesh.corners[n, m]`): the local corners of each of those `m`-simplices.
//!
//! The geometry stays in the examples, as numga keeps it out of its mesh module: what is here
//! knows only the simplices' vertex lists, except for the constructors, which place vertices,
//! and [`closed`], which reads the enclosed volume to wind a surface outward.

use gax::field::{Field, SparseMap};
use gax::mesh::Topology;
use gax::vga3d::{Bivector, Vector};
use gax::{Extensor, Kind};
use std::collections::VecDeque;

/// The value `x` of a scalar kind: the cells of incidence and selection maps.
pub fn scalar<C: Extensor<Slots = (), Coef = f64>>(x: f64) -> C {
    C::from_coeffs(<C::Kind as Kind>::arr_from_fn(|_| x))
}

/// The sign of the permutation that sorts `v`: `+1` for an even order of distinct indices.
pub fn parity(v: &[usize]) -> i8 {
    let inversions = (0..v.len())
        .flat_map(|i| (i + 1..v.len()).map(move |j| (i, j)))
        .filter(|&(i, j)| v[i] > v[j])
        .count();
    if inversions.is_multiple_of(2) { 1 } else { -1 }
}

/// `corners[n, m]`: the local corners of each `m`-simplex of an `n`-simplex (`m < n`), in
/// numga's order: the corners themselves; the facet facing each corner (from triangles up, and
/// an edge's two ends); the `m`-subsets in lexicographic order between.
///
/// # Panics
/// If `m` is not below `n`.
pub fn corners(n: usize, m: usize) -> Vec<Vec<usize>> {
    assert!(m < n, "a {n}-simplex has no {m}-simplices below it");
    if m == 0 {
        (0..=n).map(|i| vec![i]).collect()
    } else if m == n - 1 {
        (0..=n)
            .map(|i| (0..=n).filter(|&j| j != i).collect())
            .collect()
    } else {
        subsets(n + 1, m + 1)
    }
}

/// The `k`-subsets of `0..n`, each increasing, in lexicographic order.
fn subsets(n: usize, k: usize) -> Vec<Vec<usize>> {
    if k == 0 {
        return vec![vec![]];
    }
    (0..n)
        .flat_map(|first| {
            subsets(n, k - 1)
                .into_iter()
                .filter(move |rest| rest.first().is_none_or(|&r| r > first))
                .map(move |rest| [vec![first], rest].concat())
        })
        .collect()
}

/// For each `n`-simplex of a mesh, its `m`-simplices in numga's local order, each with a sign:
/// numga's batch of `k` maps from the `m`-simplices to the `n`-simplices, entry `j` of the
/// batch reading the `j`-th of each.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Batch {
    /// `sites[r][j]`: the index of the `j`-th `m`-simplex of `n`-simplex `r`.
    pub sites: Vec<Vec<usize>>,
    /// Its sign there.
    pub signs: Vec<Vec<i8>>,
    /// The number of `m`-simplices: the maps' input sites.
    pub inputs: usize,
}

impl Batch {
    /// The number of `m`-simplices of each `n`-simplex: the batch's size.
    pub fn width(&self) -> usize {
        self.sites.first().map_or(0, Vec::len)
    }

    /// The number of `n`-simplices: the maps' output sites.
    pub fn outputs(&self) -> usize {
        self.sites.len()
    }

    /// Entry `j` of the batch, signed: the map reading the `j`-th `m`-simplex of each
    /// `n`-simplex, times its sign.
    pub fn map<C: Extensor<Slots = (), Coef = f64>>(&self, j: usize) -> SparseMap<C> {
        let triples = (0..self.outputs()).map(|r| {
            (
                r,
                self.sites[r][j],
                scalar::<C>(f64::from(self.signs[r][j])),
            )
        });
        SparseMap::from_triples(triples, (self.outputs(), self.inputs))
    }

    /// The batch weighted and summed: `Σⱼ diag(cell(·, j)) · sign · map_j`, the map whose row `r`
    /// takes the `j`-th `m`-simplex through `cell(r, j)` times its sign (numga's
    /// `(spdiag(cells) * batch).sum(axis=0)`). With `cell` a constant scalar it is the batch's
    /// sum, `d` for the boundary.
    pub fn assemble<C>(&self, mut cell: impl FnMut(usize, usize) -> C) -> SparseMap<C>
    where
        C: Copy + core::ops::Mul<f64, Output = C>,
    {
        let triples = (0..self.outputs()).flat_map(|r| (0..self.width()).map(move |j| (r, j)));
        let triples: Vec<_> = triples
            .map(|(r, j)| {
                let sign = f64::from(self.signs[r][j]);
                (r, self.sites[r][j], cell(r, j) * sign)
            })
            .collect();
        SparseMap::from_triples(triples, (self.outputs(), self.inputs))
    }

    /// Entry `j` applied to a field on the `m`-simplices, unsigned: the value of each
    /// `n`-simplex's `j`-th `m`-simplex.
    pub fn pick<V: Extensor>(&self, j: usize, field: &Field<V>) -> Field<V> {
        self.sites.iter().map(|s| field.site(s[j])).collect()
    }

    /// Entry `j` applied to a field on the `m`-simplices, signed: numga's `batch[j] * field`.
    pub fn read<V>(&self, j: usize, field: &Field<V>) -> Field<V>
    where
        V: Extensor + core::ops::Mul<f64, Output = V>,
    {
        (self.sites.iter().zip(&self.signs))
            .map(|(s, sign)| field.site(s[j]) * f64::from(sign[j]))
            .collect()
    }

    /// Every entry applied to a field, signed: `[batch[j] * field for j]`.
    pub fn read_all<V>(&self, field: &Field<V>) -> Vec<Field<V>>
    where
        V: Extensor + core::ops::Mul<f64, Output = V>,
    {
        (0..self.width()).map(|j| self.read(j, field)).collect()
    }
}

/// The index of each `m`-simplex of each `n`-simplex, in the order of [`corners`].
fn local<const N: usize>(t: &Topology<N>, n: usize, m: usize) -> Vec<Vec<usize>> {
    (0..t.len(n))
        .map(|r| {
            let simplex = t.simplex(n, r);
            if m == 0 {
                simplex.to_vec()
            } else if m == n - 1 && n >= 2 {
                t.faces(n, r).iter().map(|&(face, _)| face).collect()
            } else {
                corners(n, m)
                    .iter()
                    .map(|c| {
                        let mut vertices: Vec<usize> = c.iter().map(|&i| simplex[i]).collect();
                        vertices.sort_unstable();
                        t.index_of(m, &vertices).expect("a face of the topology")
                    })
                    .collect()
            }
        })
        .collect()
}

/// `mesh.incidence[n, m]`: the `m`-simplices of each `n`-simplex, unsigned.
pub fn incidence<const N: usize>(t: &Topology<N>, n: usize, m: usize) -> Batch {
    let sites = local(t, n, m);
    let signs = sites.iter().map(|s| vec![1; s.len()]).collect();
    Batch {
        sites,
        signs,
        inputs: t.len(m),
    }
}

/// `mesh.oriented[n, m]`: the `m`-simplices of each `n`-simplex, `+1` where the stored one (its
/// vertices increasing) runs the way the `n`-simplex's own order runs it, `−1` where it runs
/// the other way. A local formula written in a simplex's own order (a Whitney form from a
/// local tail to a local head) reaches the stored simplices through these signs.
pub fn oriented<const N: usize>(t: &Topology<N>, n: usize, m: usize) -> Batch {
    let sites = local(t, n, m);
    let local_corners = corners(n, m);
    let signs = (0..t.len(n))
        .map(|r| {
            let simplex = t.simplex(n, r);
            let global = |c: &Vec<usize>| -> Vec<usize> { c.iter().map(|&i| simplex[i]).collect() };
            local_corners.iter().map(|c| parity(&global(c))).collect()
        })
        .collect();
    Batch {
        sites,
        signs,
        inputs: t.len(m),
    }
}

/// `mesh.boundary[n, n − 1]`: the facets of each `n`-simplex, signed by their orientation in it:
/// an edge's tail `−1` and head `+1`, a face's edges `+1` where its winding runs along them, a
/// tetrahedron's faces `+1` where they face outward. Summed, it is `Topology::d(n − 1)`.
///
/// # Panics
/// If `n` is zero.
pub fn boundary<const N: usize>(t: &Topology<N>, n: usize) -> Batch {
    assert!(n >= 1, "a vertex has no boundary");
    if n == 1 {
        let sites = (0..t.len(1)).map(|e| t.simplex(1, e).to_vec()).collect();
        let signs = (0..t.len(1)).map(|_| vec![-1, 1]).collect();
        return Batch {
            sites,
            signs,
            inputs: t.len(0),
        };
    }
    let (sites, signs) = (0..t.len(n))
        .map(|r| t.faces(n, r).iter().copied().unzip())
        .unzip();
    Batch {
        sites,
        signs,
        inputs: t.len(n - 1),
    }
}

/// The `k`-simplices on the boundary: the faces of the `(N − 2)`-simplices that bound a single
/// top simplex, in increasing order (numga's `exterior`).
pub fn exterior<const N: usize>(t: &Topology<N>, k: usize) -> Vec<usize> {
    let top = N - 2;
    let mut on_boundary = t.boundary();
    for j in (k + 1..=top).rev() {
        let mut below: Vec<usize> = on_boundary
            .iter()
            .flat_map(|&s| t.faces(j, s).iter().map(|&(f, _)| f))
            .collect();
        below.sort_unstable();
        below.dedup();
        on_boundary = below;
    }
    on_boundary
}

/// The `k`-simplices off the boundary, in increasing order: all but [`exterior`]'s.
pub fn interior<const N: usize>(t: &Topology<N>, k: usize) -> Vec<usize> {
    let outside = exterior(t, k);
    (0..t.len(k))
        .filter(|i| outside.binary_search(i).is_err())
        .collect()
}

/// The selection map reading a field over `len` sites at the given ones, `[indices] <- [len]`
/// (numga's `mesh.reading`); its transpose adds values at them back onto the sites.
pub fn reading<C: Extensor<Slots = (), Coef = f64>>(indices: &[usize], len: usize) -> SparseMap<C> {
    let triples = indices
        .iter()
        .enumerate()
        .map(|(r, &i)| (r, i, scalar::<C>(1.0)));
    SparseMap::from_triples(triples, (indices.len(), len))
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

/// The top simplices with their first two vertices swapped: run the other way.
pub fn turned<const N: usize>(tops: &[[usize; N]]) -> Vec<[usize; N]> {
    tops.iter()
        .map(|s| {
            let mut s = *s;
            s.swap(0, 1);
            s
        })
        .collect()
}

/// The top simplices of a manifold, some turned, so that the two sharing a facet run it in
/// opposite senses, as the simplices of an oriented manifold do; each connected piece keeps the
/// sense of its first simplex (numga's `coherent`). Walked breadth first across the facets.
///
/// # Panics
/// If the simplices do not make a complex on `vertices` vertices.
pub fn coherent<const N: usize>(vertices: usize, tops: &[[usize; N]]) -> Vec<[usize; N]> {
    let t = Topology::new(vertices, tops).expect("a complex");
    let top = N - 1;
    // The owners of each facet, with the sense each runs it in.
    let mut owners: Vec<Vec<(usize, i8)>> = vec![Vec::new(); t.len(top - 1)];
    for s in 0..t.len(top) {
        for &(facet, sign) in t.faces(top, s) {
            owners[facet].push((s, sign));
        }
    }
    let mut flips: Vec<Option<bool>> = vec![None; tops.len()];
    for root in 0..tops.len() {
        if flips[root].is_some() {
            continue;
        }
        flips[root] = Some(false);
        let mut queue = VecDeque::from([root]);
        while let Some(s) = queue.pop_front() {
            let flipped = flips[s].expect("reached");
            for &(facet, sign) in t.faces(top, s) {
                for &(other, other_sign) in &owners[facet] {
                    if other != s && flips[other].is_none() {
                        // Neighbours that run their shared facet the same way disagree.
                        flips[other] = Some(flipped ^ (sign == other_sign));
                        queue.push_back(other);
                    }
                }
            }
        }
    }
    let swapped = turned(tops);
    (0..tops.len())
        .map(|s| {
            if flips[s] == Some(true) {
                swapped[s]
            } else {
                tops[s]
            }
        })
        .collect()
}

/// A closed surface's faces wound coherently and counter-clockwise seen from outside, whatever
/// sense they come in (numga's `Mesh.closed`): coherent, then all turned if the volume they
/// enclose, the sum of the cones each face spans from the origin, `a ∧ b ∧ c`, is negative.
pub fn closed(faces: &[[usize; 3]], vertices: &Field<Vector>) -> Vec<[usize; 3]> {
    let wound = coherent(vertices.len(), faces);
    let volume: f64 = wound
        .iter()
        .map(|&[a, b, c]| (vertices.site(a) ^ vertices.site(b) ^ vertices.site(c)).e123())
        .sum();
    if volume < 0.0 { turned(&wound) } else { wound }
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

/// Closed loops of vertices that generate the first homology of a closed surface: `2g` of them
/// on a surface of genus `g`, none on a sphere. By tree and cotree (Eppstein 2003): a
/// breadth-first tree of the edges, a breadth-first tree of the faces across the edges not in
/// it, and each edge in neither closing a loop through the first tree. Each loop runs from the
/// edge's head back through the tree to its tail, the edge closing it.
pub fn generators(t: &Topology<3>) -> Vec<Vec<usize>> {
    let (children, tree) = spanning_tree(t);
    let mut parent = vec![None; t.len(0)];
    let mut in_tree = vec![false; t.len(1)];
    let all = edges(t);
    for (&child, &e) in children.iter().zip(&tree) {
        in_tree[e] = true;
        let [a, b] = all[e];
        parent[child] = Some(if a == child { b } else { a });
    }
    // The faces beside each edge, and a tree of the faces across the edges off the first tree.
    let mut beside: Vec<Vec<usize>> = vec![Vec::new(); t.len(1)];
    for f in 0..t.len(2) {
        for &(e, _) in t.faces(2, f) {
            beside[e].push(f);
        }
    }
    let mut in_cotree = vec![false; t.len(1)];
    let mut reached = vec![false; t.len(2)];
    for root in 0..t.len(2) {
        if reached[root] {
            continue;
        }
        reached[root] = true;
        let mut queue = VecDeque::from([root]);
        while let Some(f) = queue.pop_front() {
            for &(e, _) in t.faces(2, f) {
                if in_tree[e] {
                    continue;
                }
                for &g in &beside[e] {
                    if !reached[g] {
                        reached[g] = true;
                        in_cotree[e] = true;
                        queue.push_back(g);
                    }
                }
            }
        }
    }
    let to_root = |mut v: usize| {
        let mut path = vec![v];
        while let Some(p) = parent[v] {
            path.push(p);
            v = p;
        }
        path
    };
    (0..t.len(1))
        .filter(|&e| !in_tree[e] && !in_cotree[e])
        .map(|e| {
            let [tail, head] = all[e];
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

/// `[F + L, E]`: the circulation of edge values counter-clockwise around each face (`d(1)`),
/// then around each closed loop of vertices, in its own order, the step from its last vertex
/// back to its first included (numga's `circulation`).
///
/// # Panics
/// If two consecutive vertices of a loop are not joined by an edge.
pub fn circulation<C, const N: usize>(t: &Topology<N>, loops: &[Vec<usize>]) -> SparseMap<C>
where
    C: Extensor<Slots = (), Coef = f64>,
{
    let faces = t.len(2);
    let around = (0..faces).flat_map(|f| {
        t.faces(2, f)
            .iter()
            .map(move |&(e, sign)| (f, e, scalar::<C>(f64::from(sign))))
    });
    let along = loops.iter().enumerate().flat_map(|(l, path)| {
        (0..path.len()).map(move |i| {
            let (a, b) = (path[i], path[(i + 1) % path.len()]);
            let e = t
                .index_of(1, &[a.min(b), a.max(b)])
                .expect("an edge of the loop");
            (faces + l, e, scalar::<C>(if a < b { 1.0 } else { -1.0 }))
        })
    });
    SparseMap::from_triples(
        around.chain(along).collect::<Vec<_>>(),
        (faces + loops.len(), t.len(1)),
    )
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

    /// A tetrahedron given in an odd order, and two triangles.
    fn tet() -> Topology<4> {
        Topology::new(4, &[[1, 0, 2, 3]]).unwrap()
    }

    #[test]
    fn the_batches_follow_numgas_order_and_signs() {
        // A tetrahedron's edges by their corners (0 1), (0 2), (0 3), (1 2), (1 3), (2 3).
        assert_eq!(corners(3, 1)[3], [1, 2]);
        assert_eq!(corners(2, 1), [vec![1, 2], vec![0, 2], vec![0, 1]]);
        let t = tet();
        // Its first local edge runs from vertex 1 to vertex 0: against the stored edge (0 1).
        let o = oriented(&t, 3, 1);
        assert_eq!(o.sites[0][0], t.index_of(1, &[0, 1]).unwrap());
        assert_eq!(o.signs[0][0], -1);
        // The boundary summed is d.
        for n in 1..4 {
            let b = boundary(&t, n);
            let ones = Field::from_fn(b.inputs, |i| Scalar::new(i as f64 * 0.7 - 1.0));
            let summed = b.assemble(|_, _| Scalar::new(1.0));
            let d = t.d::<Scalar>(n - 1);
            assert_eq!(&summed * &ones, &d * &ones);
        }
    }

    #[test]
    fn the_torus_has_two_loops_and_the_sphere_none() {
        let (torus, _) = torus(2.0, 0.75, 8, 5);
        assert_eq!(torus.betti_numbers(), [1, 2, 1]);
        assert!(torus.is_oriented());
        let loops = generators(&torus);
        assert_eq!(loops.len(), 2);
        // Each loop is closed: a gradient's circulation around it, as around every face, is
        // nothing.
        let circulation = circulation::<Scalar, 3>(&torus, &loops);
        assert_eq!(circulation.shape(), (torus.len(2) + 2, torus.len(1)));
        let potential = Field::from_fn(torus.len(0), |v| Scalar::new((v * v % 7) as f64 - 2.5));
        let steps = &torus.d::<Scalar>(0) * &potential;
        assert!((&circulation * &steps).iter().all(|c| c.s().abs() < 1e-12));
        let (sphere, vertices) = icosphere(2);
        assert!(generators(&sphere).is_empty());
        assert!(vertices.iter().all(|v| (v.norm() - 1.0).abs() < 1e-15));
        assert_eq!(sphere.euler_characteristic(), 2);
    }

    #[test]
    fn coherent_winding_undoes_turned_faces() {
        let (sphere, vertices) = icosphere(1);
        let faces = faces(&sphere);
        let mixed: Vec<[usize; 3]> = faces
            .iter()
            .enumerate()
            .map(|(f, &[a, b, c])| if f % 3 == 1 { [b, a, c] } else { [a, b, c] })
            .collect();
        let back = closed(&turned(&mixed), &vertices);
        let wound = Topology::new(vertices.len(), &back).unwrap();
        assert!(wound.is_oriented());
        assert_eq!(back[0].map(|v| v), faces[0]);
        assert_eq!(exterior(&sphere, 0), Vec::<usize>::new());
        assert_eq!(interior(&sphere, 1).len(), sphere.len(1));
    }
}
