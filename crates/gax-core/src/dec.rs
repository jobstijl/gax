//! Discrete exterior calculus (ADR-050): the metric of a mesh, its Hodge stars, and the maps
//! between fields and cochains.
//!
//! The metric enters through squared distances alone. A model of space gives the squared distance
//! between two positions ([`Distance`]), and everything else follows intrinsically, simplex by
//! simplex, from the Gram matrix of its edges, `Gᵢⱼ = ½(d₀ᵢ² + d₀ⱼ² − dᵢⱼ²)`: volumes,
//! circumcentres, the inner products of the barycentric gradients. So the same code serves VGA's
//! vectors, PGA's points and CGA's round points.
//!
//! * [`Geometry`] holds a mesh's primal and circumcentric dual volumes and gives the two Hodge
//!   stars: the diagonal one, `⋆ₖ = |σ*| / |σ|` (Hirani 2003), and the Galerkin one, the mass
//!   matrix of the Whitney forms.
//! * [`de_rham`] integrates a `k`-vector field over each `k`-simplex: a cochain. The Whitney
//!   map [`Geometry::whitney_1`] takes a cochain on the edges back to a vector field.
//!
//! ```
//! use gax::dec::Geometry;
//! use gax::field::Field;
//! use gax::mesh::Mesh;
//! use gax::vga3d::{Scalar, Vector};
//!
//! // Two right triangles making a unit square.
//! let v = |x: f64, y: f64| Vector::<(), f64>::new(x, y, 0.0);
//! let vertices: Field<_> = [v(0.0, 0.0), v(1.0, 0.0), v(1.0, 1.0), v(0.0, 1.0)].into_iter().collect();
//! let square = Mesh::new(vertices, &[[0, 1, 2], [0, 2, 3]]).unwrap();
//! let g = Geometry::new(&square);
//! assert_eq!(g.volumes(2), [0.5, 0.5]);
//! // The circumcentres lie on the diagonal (index 1), so its dual edge has length zero.
//! assert!(g.dual_volumes(1)[1].abs() < 1e-15);
//! // The cotangent Laplacian d₀ᵀ ⋆₁ d₀ (on triangles the Galerkin star gives the same): it
//! // annihilates the constants.
//! let d0 = square.d(0);
//! let laplacian = &d0.transpose_with(|c| c) * &(&g.star::<Scalar<(), f64>>(1) * &d0);
//! let ones: Field<_> = (0..4).map(|_| Scalar::<(), f64>::new(1.0)).collect();
//! let image: Field<Scalar<(), f64>> = &laplacian * &ones;
//! assert!(image.sites().iter().all(|x| x.s().abs() < 1e-15));
//! ```

extern crate std;

use crate::coef::Real;
use crate::field::{Field, SparseMap};
use crate::kind::{Extensor, Kind};
use crate::mesh::{Complex, Mesh};
use crate::ops::{Reverse, ScalarProduct};
use crate::simplex::{Measure, Position, Rule, Simplex};
use std::vec::Vec;

/// A model of Euclidean space: the squared distance between two positions.
pub trait Distance: Position {
    /// `|a − b|²`.
    fn distance_squared(a: &Self, b: &Self) -> Self::Coef;
}

/// Solve `A x = b` for a small dense `A` (row-major `n × n`), by Gaussian elimination with
/// partial pivoting. `None` if `A` is singular to working precision.
fn solve<T: Real>(mut a: Vec<T>, mut b: Vec<T>) -> Option<Vec<T>> {
    let n = b.len();
    for c in 0..n {
        let p = (c..n).max_by(|&i, &j| {
            if T::all_lt(a[i * n + c].abs(), a[j * n + c].abs()) {
                core::cmp::Ordering::Less
            } else {
                core::cmp::Ordering::Greater
            }
        })?;
        if a[p * n + c] == T::zero() {
            return None;
        }
        if p != c {
            for k in 0..n {
                a.swap(p * n + k, c * n + k);
            }
            b.swap(p, c);
        }
        let inv = a[c * n + c].recip();
        for r in c + 1..n {
            let f = a[r * n + c] * inv;
            for k in c..n {
                a[r * n + k] = a[r * n + k] - f * a[c * n + k];
            }
            b[r] = b[r] - f * b[c];
        }
    }
    let mut x = std::vec![T::zero(); n];
    for r in (0..n).rev() {
        let mut s = b[r];
        for k in r + 1..n {
            s = s - a[r * n + k] * x[k];
        }
        x[r] = s * a[r * n + r].recip();
    }
    Some(x)
}

/// The determinant of a small dense matrix, by elimination with partial pivoting.
fn determinant<T: Real>(mut a: Vec<T>, n: usize) -> T {
    let mut det = T::one();
    for c in 0..n {
        let Some(p) = (c..n).max_by(|&i, &j| {
            if T::all_lt(a[i * n + c].abs(), a[j * n + c].abs()) {
                core::cmp::Ordering::Less
            } else {
                core::cmp::Ordering::Greater
            }
        }) else {
            return T::one();
        };
        if a[p * n + c] == T::zero() {
            return T::zero();
        }
        if p != c {
            for k in 0..n {
                a.swap(p * n + k, c * n + k);
            }
            det = -det;
        }
        det = det * a[c * n + c];
        let inv = a[c * n + c].recip();
        for r in c + 1..n {
            let f = a[r * n + c] * inv;
            for k in c..n {
                a[r * n + k] = a[r * n + k] - f * a[c * n + k];
            }
        }
    }
    det
}

/// The Gram matrix `Gᵢⱼ = ½(d₀ᵢ + d₀ⱼ − dᵢⱼ)` of a simplex's edges from its first vertex, from the
/// squared distances `d` (row-major over its `k + 1` vertices): `k × k`.
fn gram<T: Real>(d: &[T], k: usize) -> Vec<T> {
    let half = T::from_ratio(1, 2);
    let n = k + 1;
    let mut g = std::vec![T::zero(); k * k];
    for i in 1..=k {
        for j in 1..=k {
            g[(i - 1) * k + j - 1] = half * (d[i] + d[j] - d[i * n + j]);
        }
    }
    g
}

/// The unsigned `k`-volume of a simplex from its squared distances: `√det G / k!`.
fn volume<T: Real>(d: &[T], k: usize) -> T {
    if k == 0 {
        return T::one();
    }
    let det = determinant(gram(d, k), k);
    let fact: i64 = (1..=k as i64).product();
    det.abs().sqrt() * T::from_ratio(1, fact)
}

/// The circumcentre of a simplex, in its barycentric coordinates, from its squared distances:
/// `x` solves `G x = ½ diag G` and the coordinates are `(1 − Σx, x)`.
fn circumcentre<T: Real>(d: &[T], k: usize) -> Vec<T> {
    if k == 0 {
        return std::vec![T::one()];
    }
    let g = gram(d, k);
    let half = T::from_ratio(1, 2);
    let rhs = (0..k).map(|i| half * g[i * k + i]).collect();
    let x = solve(g, rhs).expect("a simplex of positive volume");
    let mut out = std::vec![T::one() - x.iter().fold(T::zero(), |a, &b| a + b)];
    out.extend(x);
    out
}

/// The squared distances between points given in barycentric coordinates of one simplex,
/// `|Σ wᵢ vᵢ|² = −½ Σ wᵢ wⱼ dᵢⱼ` for `w` the difference of two of them (`Σ w = 0`).
fn barycentric_distances<T: Real>(points: &[Vec<T>], d: &[T]) -> Vec<T> {
    let n = points[0].len();
    let m = points.len();
    let half = T::from_ratio(1, 2);
    let mut out = std::vec![T::zero(); m * m];
    for p in 0..m {
        for q in 0..m {
            let w: Vec<T> = (0..n).map(|i| points[p][i] - points[q][i]).collect();
            let mut s = T::zero();
            for i in 0..n {
                for j in 0..n {
                    s = s + w[i] * w[j] * d[i * n + j];
                }
            }
            out[p * m + q] = -(half * s);
        }
    }
    out
}

/// The metric of a mesh: its primal and circumcentric dual volumes, and its Hodge stars.
#[derive(Clone, Debug)]
pub struct Geometry<T, const N: usize> {
    complex: Complex<N>,
    /// The squared distances between the vertices of each top simplex, row-major `N × N`.
    distances: Vec<Vec<T>>,
    volumes: Vec<Vec<T>>,
    dual_volumes: Vec<Vec<T>>,
}

impl<T: Real, const N: usize> Geometry<T, N> {
    /// The geometry of a mesh, from the squared distances of its model.
    #[must_use]
    pub fn new<V: Distance<Coef = T> + Extensor<Slots = ()>>(mesh: &Mesh<V, N>) -> Self {
        let complex = mesh.complex().clone();
        let v = mesh.vertices.sites();
        let distances_of = |s: &[usize]| -> Vec<T> {
            s.iter()
                .flat_map(|&a| s.iter().map(move |&b| V::distance_squared(&v[a], &v[b])))
                .collect()
        };
        let volumes: Vec<Vec<T>> = (0..N)
            .map(|k| {
                complex
                    .simplices(k)
                    .map(|s| volume(&distances_of(s), k))
                    .collect()
            })
            .collect();
        let distances: Vec<Vec<T>> = complex.simplices(N - 1).map(distances_of).collect();
        let mut dual_volumes: Vec<Vec<T>> = (0..N)
            .map(|k| std::vec![T::zero(); complex.len(k)])
            .collect();
        for (t, top) in complex.simplices(N - 1).enumerate() {
            let d = &distances[t];
            // Every flag from the top simplex down to each face: (local vertices, its
            // circumcentre in the top simplex's barycentric coordinates).
            let local: Vec<usize> = (0..N).collect();
            let centre = circumcentre(d, N - 1);
            let mut flag: Vec<Vec<T>> = std::vec![centre.clone()];
            descend(
                &complex,
                top,
                &local,
                &centre,
                T::one(),
                d,
                &mut flag,
                &mut dual_volumes,
            );
        }
        Geometry {
            complex,
            distances,
            volumes,
            dual_volumes,
        }
    }

    /// The unsigned volumes of the `k`-simplices (lengths, areas, …; one for a vertex).
    #[must_use]
    pub fn volumes(&self, k: usize) -> &[T] {
        &self.volumes[k]
    }

    /// The signed volumes of the circumcentric duals of the `k`-simplices: the cell of the
    /// points nearer to the simplex's circumcentre than to the others', its pieces in each top
    /// simplex signed by which side of each face its circumcentres fall (Hirani 2003, as `PyDEC`
    /// computes them). Positive on a well-centred mesh; for the top simplices, one.
    #[must_use]
    pub fn dual_volumes(&self, k: usize) -> &[T] {
        &self.dual_volumes[k]
    }

    /// The diagonal (circumcentric) Hodge star on `k`-cochains: `|σ*| / |σ|`, cells of a scalar
    /// kind.
    #[must_use]
    pub fn star<C: Extensor<Slots = (), Coef = T>>(&self, k: usize) -> SparseMap<C> {
        const {
            assert!(
                <C::Kind as Kind>::N == 1,
                "the cells of a Hodge star are scalars"
            );
        }
        let n = self.complex.len(k);
        let cell = |x: T| C::from_coeffs(<C::Kind as Kind>::arr_from_fn(|_| x));
        SparseMap::new(
            (0..n)
                .map(|i| cell(self.dual_volumes[k][i] * self.volumes[k][i].recip()))
                .collect(),
            (0..n).collect(),
            (0..n).collect(),
            (n, n),
        )
    }

    /// Whether every simplex contains its circumcentre strictly inside (every barycentric
    /// coordinate of every face's circumcentre positive, in each top simplex): the condition for
    /// every diagonal star to be positive.
    #[must_use]
    pub fn is_well_centered(&self) -> bool {
        self.dual_volumes
            .iter()
            .all(|vols| vols.iter().all(|&v| T::all_lt(T::zero(), v)))
    }

    /// The Galerkin Hodge star on `k`-cochains: the mass matrix `M_k = ∫ Wᵢ · Wⱼ dV` of the
    /// Whitney forms, symmetric and positive definite on any mesh of positive volumes. Its
    /// entries are exact, from `∫ λₐ λ_b dV = (1 + δₐ_b) V / ((n+1)(n+2))` and the inner products
    /// of the barycentric gradients.
    #[must_use]
    pub fn mass<C: Extensor<Slots = (), Coef = T>>(&self, k: usize) -> SparseMap<C> {
        const {
            assert!(
                <C::Kind as Kind>::N == 1,
                "the cells of a mass matrix are scalars"
            );
        }
        let n = N - 1;
        let mut triples = Vec::new();
        for (t, top) in self.complex.simplices(N - 1).enumerate() {
            let h = gradients(&self.distances[t], n);
            let vol = self.volumes[N - 1][t];
            let pair = |a: usize, b: usize| -> T {
                vol * T::from_ratio(if a == b { 2 } else { 1 }, ((n + 1) * (n + 2)) as i64)
            };
            // The k-faces of the top simplex, as local vertices in the face's own order (sorted
            // by global number below the top; the top simplex as given), and their indices.
            let faces: Vec<(Vec<usize>, usize)> = if k == n {
                std::vec![((0..N).collect(), t)]
            } else {
                subsets(N, k + 1)
                    .into_iter()
                    .map(|local| {
                        let mut global: Vec<usize> = local.iter().map(|&l| top[l]).collect();
                        // Sort the face's vertices by their global numbers, keeping them local.
                        let mut order: Vec<usize> = (0..local.len()).collect();
                        order.sort_by_key(|&i| global[i]);
                        let local: Vec<usize> = order.iter().map(|&i| local[i]).collect();
                        global.sort_unstable();
                        let index = self
                            .complex
                            .index_of(k, &global)
                            .expect("a face of the complex");
                        (local, index)
                    })
                    .collect()
            };
            let fact: i64 = (1..=k as i64).product();
            let scale = T::from_i64(fact * fact);
            for (s, si) in &faces {
                for (r, ri) in &faces {
                    let mut entry = T::zero();
                    for i in 0..=k {
                        for j in 0..=k {
                            let sign = if (i + j) % 2 == 0 {
                                T::one()
                            } else {
                                -T::one()
                            };
                            let rows: Vec<usize> =
                                (0..=k).filter(|&x| x != i).map(|x| s[x]).collect();
                            let cols: Vec<usize> =
                                (0..=k).filter(|&x| x != j).map(|x| r[x]).collect();
                            let minor: Vec<T> = rows
                                .iter()
                                .flat_map(|&a| {
                                    cols.iter().map(|&b| h[a * (n + 1) + b]).collect::<Vec<_>>()
                                })
                                .collect();
                            entry = entry + sign * pair(s[i], r[j]) * determinant(minor, k);
                        }
                    }
                    triples.push((*si, *ri, entry * scale));
                }
            }
        }
        let cell = |x: T| C::from_coeffs(<C::Kind as Kind>::arr_from_fn(|_| x));
        let m = self.complex.len(k);
        // Repeated couplings add: each top simplex's share of an entry.
        SparseMap::new(
            triples.iter().map(|&(_, _, x)| cell(x)).collect(),
            triples.iter().map(|&(r, _, _)| r).collect(),
            triples.iter().map(|&(_, c, _)| c).collect(),
            (m, m),
        )
    }
}

impl<T: Real, const N: usize> Geometry<T, N> {
    /// The Whitney map on 1-cochains: the vector field `Σ c_e (λₐ ∇λ_b − λ_b ∇λₐ)` of the cochain
    /// `c` on the edges, on top simplex `top` at its barycentric point `at`. The gradients are the
    /// reciprocal frame of the simplex's edges in the model, `∇λᵢ = Σⱼ (G⁻¹)ᵢⱼ (vⱼ − v₀)`: vectors
    /// in VGA, directions (differences of points) in PGA. Its tangential part is continuous across
    /// faces, and its integral along each edge is the cochain's value there.
    ///
    /// # Panics
    /// If `at` does not have `N` coordinates or `cochain` one value per edge.
    #[must_use]
    pub fn whitney_1<V, S>(&self, mesh: &Mesh<V, N>, cochain: &Field<S>, top: usize, at: &[T]) -> V
    where
        V: Extensor<Slots = (), Coef = T>
            + core::ops::Add<Output = V>
            + core::ops::Sub<Output = V>
            + crate::ops::Gp<T, Output = V>,
        S: Extensor<Slots = (), Coef = T>,
    {
        assert_eq!(
            at.len(),
            N,
            "a point of a top simplex has {N} barycentric coordinates"
        );
        assert_eq!(
            cochain.len(),
            self.complex.len(1),
            "a 1-cochain has a value on each edge"
        );
        let n = N - 1;
        let s = self.complex.simplex(n, top);
        let v = mesh.vertices.sites();
        // The inverse Gram matrix, and the gradients as model vectors.
        let g = gram(&self.distances[top], n);
        let edges: Vec<V> = (1..N).map(|j| v[s[j]] - v[s[0]]).collect();
        let mut grads: Vec<V> = Vec::with_capacity(N);
        for i in 0..n {
            let e = (0..n)
                .map(|r| if r == i { T::one() } else { T::zero() })
                .collect();
            let row = solve(g.clone(), e).expect("a simplex of positive volume");
            let grad = (0..n)
                .map(|j| edges[j].gp(row[j]))
                .reduce(|a, b| a + b)
                .expect("an edge");
            grads.push(grad);
        }
        let first = grads
            .iter()
            .copied()
            .reduce(|a, b| a + b)
            .expect("an edge")
            .gp(-T::one());
        grads.insert(0, first);
        let mut out: Option<V> = None;
        for a in 0..N {
            for b in a + 1..N {
                let mut key = [s[a], s[b]];
                key.sort_unstable();
                let e = self
                    .complex
                    .index_of(1, &key)
                    .expect("an edge of the complex");
                // The edge runs from its lower vertex to its higher.
                let (lo, hi) = if s[a] < s[b] { (a, b) } else { (b, a) };
                let c = cochain.sites()[e].coeffs().as_ref()[0];
                let term = (grads[hi].gp(at[lo]) - grads[lo].gp(at[hi])).gp(c);
                out = Some(match out {
                    Some(o) => o + term,
                    None => term,
                });
            }
        }
        out.expect("a simplex has an edge")
    }
}

/// The inner products `⟨dλₐ, dλ_b⟩` of the barycentric gradients of a simplex of dimension `n`,
/// `(n+1) × (n+1)`: the inverse Gram matrix for `a, b ≥ 1`, and `dλ₀ = −Σ dλᵢ`.
fn gradients<T: Real>(d: &[T], n: usize) -> Vec<T> {
    let g = gram(d, n);
    let mut inv = std::vec![T::zero(); n * n];
    for c in 0..n {
        let e = (0..n)
            .map(|r| if r == c { T::one() } else { T::zero() })
            .collect();
        let col = solve(g.clone(), e).expect("a simplex of positive volume");
        for r in 0..n {
            inv[r * n + c] = col[r];
        }
    }
    let m = n + 1;
    let mut h = std::vec![T::zero(); m * m];
    for a in 0..m {
        for b in 0..m {
            let entry = |i: usize, j: usize| inv[(i - 1) * n + j - 1];
            h[a * m + b] = match (a, b) {
                (0, 0) => (1..m)
                    .flat_map(|i| (1..m).map(move |j| (i, j)))
                    .fold(T::zero(), |s, (i, j)| s + entry(i, j)),
                (0, b) => -(1..m).fold(T::zero(), |s, i| s + entry(i, b)),
                (a, 0) => -(1..m).fold(T::zero(), |s, j| s + entry(a, j)),
                (a, b) => entry(a, b),
            };
        }
    }
    h
}

/// The `size`-element subsets of `0..n`, in lexicographic order.
fn subsets(n: usize, size: usize) -> Vec<Vec<usize>> {
    if size == 0 {
        return std::vec![Vec::new()];
    }
    (0..n)
        .flat_map(|first| {
            subsets(n - first - 1, size - 1)
                .into_iter()
                .map(move |rest| {
                    core::iter::once(first)
                        .chain(rest.into_iter().map(|r| r + first + 1))
                        .collect()
                })
        })
        .collect()
}

/// The circumcentric dual volume pieces of a face of a top simplex and of its faces below it,
/// along one flag (`PyDEC`'s `__compute_dual_volume`, intrinsically). `face` holds local vertices
/// of the top simplex, `flag` the circumcentres from the top down to this face, `sign` the
/// product of the side signs so far.
#[allow(clippy::too_many_arguments)]
fn descend<T: Real, const N: usize>(
    complex: &Complex<N>,
    top: &[usize],
    face: &[usize],
    centre: &[T],
    sign: T,
    d: &[T],
    flag: &mut Vec<Vec<T>>,
    duals: &mut [Vec<T>],
) {
    let k = face.len() - 1;
    let mut global: Vec<usize> = face.iter().map(|&l| top[l]).collect();
    global.sort_unstable();
    let index = complex.index_of(k, &global).expect("a face of the complex");
    let m = flag.len() - 1;
    let piece = volume(&barycentric_distances(flag, d), m);
    duals[k][index] = duals[k][index] + sign * piece;
    if k == 0 {
        return;
    }
    for (o, &opposite) in face.iter().enumerate() {
        let below: Vec<usize> = face.iter().copied().filter(|&l| l != opposite).collect();
        // Its circumcentre, in its own barycentric coordinates, then in the top simplex's.
        let sub: Vec<T> = below
            .iter()
            .flat_map(|&a| below.iter().map(move |&b| d[a * N + b]))
            .collect();
        let own = circumcentre(&sub, k - 1);
        let mut c = std::vec![T::zero(); N];
        for (&l, &x) in below.iter().zip(&own) {
            c[l] = x;
        }
        // The side: the sign of this face's centre's coordinate at the opposite vertex, as seen
        // from the parent's own barycentric coordinates.
        let side = if T::all_lt(centre_in(face, centre, o), T::zero()) {
            -T::one()
        } else {
            T::one()
        };
        flag.push(c.clone());
        descend(complex, top, &below, &c, sign * side, d, flag, duals);
        flag.pop();
    }
}

/// The barycentric coordinate, in `face`, of a point of `face`'s span given in the top simplex's
/// coordinates, at `face`'s `o`-th vertex.
fn centre_in<T: Real>(face: &[usize], centre: &[T], o: usize) -> T {
    centre[face[o]]
}

/// The de Rham map: the `k`-cochain of a `k`-vector field `F`, its integral over each
/// `k`-simplex paired with the simplex's measure, `∫_σ F ∗ ~dX`, by the rule (`M = k + 1`
/// vertices).
pub fn de_rham<V, F, S, const N: usize, const M: usize>(
    mesh: &Mesh<V, N>,
    rule: Rule,
    f: impl Fn(V) -> F,
) -> Field<S>
where
    V: Measure<M> + Extensor<Slots = ()>,
    V::Blade: Reverse<Output = V::Blade>,
    F: core::ops::Add<Output = F>
        + crate::ops::Gp<<V as Position>::Coef, Output = F>
        + ScalarProduct<V::Blade, Output = S>,
    S: Extensor,
{
    let v = mesh.vertices.sites();
    mesh.complex()
        .simplices(M - 1)
        .map(|s| {
            let s = Simplex::<V, M>(core::array::from_fn(|i| v[s[i]]));
            s.mean(rule, &f).scalar_product(s.measure().reverse())
        })
        .collect()
}
