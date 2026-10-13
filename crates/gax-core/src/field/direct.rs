//! A sparse direct solver for square field maps: LU with partial pivoting after a
//! bandwidth-reducing ordering of the sites.
//!
//! The method is the left-looking LU of Gilbert and Peierls, as in Davis's `CSparse` (`cs_lu`):
//! each column of `L` and `U` comes from a sparse triangular solve whose nonzero pattern is
//! found by a depth-first search in the graph of `L`, so the work is proportional to the
//! arithmetic. The columns are taken in a reverse Cuthill–McKee order of the sites, which keeps
//! the factors of meshes and chains banded; the rows are chosen by threshold partial pivoting,
//! preferring the diagonal while it is within a factor of 10 of the column's largest candidate.

extern crate std;

use super::{Field, SparseMap};
use crate::coef::Real;
use crate::kind::{Extensor, Kind};
use core::fmt;
use core::marker::PhantomData;
use std::vec;
use std::vec::Vec;

/// A diagonal entry within this factor of the column's largest candidate pivot is taken as the
/// pivot: it keeps the order (and so the sparsity) for little loss of stability.
const DIAGONAL_PREFERENCE: f64 = 0.1;

/// The factorization of a field map found no pivot: the map is singular. `site` and `blade` name
/// the unknown (an input site and a coefficient of its kind) whose column had no nonzero left.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Singular {
    /// The input site of the column.
    pub site: usize,
    /// The coefficient of the site's kind.
    pub blade: usize,
}

impl fmt::Display for Singular {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "the field map is singular: no pivot for site {}, blade {}",
            self.site, self.blade
        )
    }
}

impl core::error::Error for Singular {}

/// A sparse matrix in compressed columns.
#[derive(Clone, Debug)]
struct Csc<T> {
    /// Column `j`'s entries are `p[j]..p[j + 1]`.
    p: Vec<usize>,
    /// Their rows.
    i: Vec<usize>,
    /// Their values.
    x: Vec<T>,
}

/// The sparse LU factorization of a square field map of map cells `K <- K`: `P A Q = L U` on
/// the block matrix of its coefficients. Factor once with [`SparseMap::factor`], then
/// [`solve`](Self::solve) for as many right-hand sides as needed (the steps of an implicit time
/// integration, the columns of a block).
///
/// ```
/// use gax::field::{Field, SparseMap};
/// use gax::pga3d::{Plane, Scalar};
///
/// // A saddle-point system, which conjugate gradients cannot solve: two coupled sites and a
/// // constraint that their sum vanishes, [[2, 1, 1], [1, 3, 1], [1, 1, 0]].
/// let s = Scalar::<(), f64>::new;
/// let w = [[2.0, 1.0, 1.0], [1.0, 3.0, 1.0], [1.0, 1.0, 0.0]];
/// let (mut cells, mut rows, mut cols) = (vec![], vec![], vec![]);
/// for i in 0..3 {
///     for j in 0..3 {
///         if w[i][j] != 0.0 {
///             cells.push(s(w[i][j]) * Plane::slot());
///             rows.push(i);
///             cols.push(j);
///         }
///     }
/// }
/// let map = SparseMap::new(cells, rows, cols, (3, 3));
/// let lu = map.factor().expect("nonsingular");
/// let b = Field::from_fn(3, |i| Plane::new(1.0, i as f64, 0.0, 0.0));
/// let x = lu.solve(&b);
/// let r = &map.apply(&x) - &b;
/// assert!(r.dot(&r) < 1e-28);
/// ```
#[derive(Clone, Debug)]
pub struct SparseLu<K: Kind, T> {
    sites: usize,
    /// The columns' order: factor column `k` is unknown `q[k]`.
    q: Vec<usize>,
    /// The rows' order: unknown row `i` is factor row `pinv[i]`.
    pinv: Vec<usize>,
    /// Unit lower triangular, its diagonal stored first in each column, rows in factor order.
    l: Csc<T>,
    /// Upper triangular, its diagonal stored last in each column.
    u: Csc<T>,
    marker: PhantomData<K>,
}

impl<K: Kind, T: Real> SparseLu<K, T> {
    /// The number of sites.
    #[must_use]
    pub fn sites(&self) -> usize {
        self.sites
    }

    /// The number of stored entries of `L` and `U` together (the fill): a measure of the
    /// factorization's memory and of the cost of a solve.
    #[must_use]
    pub fn fill(&self) -> usize {
        self.l.i.len() + self.u.i.len()
    }

    /// Solve `A x = b` with the factors.
    ///
    /// # Panics
    /// If `b` is not over the map's sites.
    #[must_use]
    #[track_caller]
    pub fn solve(&self, b: &Field<K::Mv<(), T>>) -> Field<K::Mv<(), T>> {
        assert!(
            b.len() == self.sites,
            "a factored field map over {} sites solved against a field over {} sites",
            self.sites,
            b.len()
        );
        let flat: Vec<T> = b
            .sites()
            .iter()
            .flat_map(|v| v.coeffs().as_ref().iter().copied())
            .collect();
        let x = self.solve_flat(&flat);
        Field::from_fn(self.sites, |s| {
            Extensor::from_coeffs(K::arr_from_fn(|i| x[s * K::N + i]))
        })
    }
}

impl<K: Kind, T: Real> SparseLu<K, T> {
    /// `A x = b` on flattened coefficients (`site * K::N + blade`).
    pub(super) fn solve_flat(&self, b: &[T]) -> Vec<T> {
        let n = self.q.len();
        let mut y = vec![T::zero(); n];
        for (i, &v) in b.iter().enumerate() {
            y[self.pinv[i]] = v;
        }
        // L y' = P b, L unit lower triangular.
        for j in 0..n {
            let yj = y[j];
            if yj != T::zero() {
                for p in self.l.p[j] + 1..self.l.p[j + 1] {
                    y[self.l.i[p]] = (-self.l.x[p]).mul_add(yj, y[self.l.i[p]]);
                }
            }
        }
        // U x' = y'.
        for j in (0..n).rev() {
            let last = self.u.p[j + 1] - 1;
            y[j] = y[j] / self.u.x[last];
            let yj = y[j];
            if yj != T::zero() {
                for p in self.u.p[j]..last {
                    y[self.u.i[p]] = (-self.u.x[p]).mul_add(yj, y[self.u.i[p]]);
                }
            }
        }
        // x = Q x'.
        let mut x = vec![T::zero(); n];
        for (k, &col) in self.q.iter().enumerate() {
            x[col] = y[k];
        }
        x
    }
}

impl<C, K, T> SparseMap<C>
where
    C: Extensor<Kind = K, Slots = (K,), Coef = T>,
    K: Kind,
    T: Real,
{
    /// The sparse LU factorization of this square map's block matrix, for
    /// [`SparseLu::solve`]: any nonsingular map, symmetric or not, definite or not, such as a
    /// saddle-point system of constraints. The sites are ordered by minimum degree, so the
    /// factors of meshes stay sparse. For scalar coefficients.
    ///
    /// # Errors
    /// [`Singular`] if a column has no nonzero pivot left: the map is singular.
    ///
    /// # Panics
    /// If the map is not square.
    #[track_caller]
    pub fn factor(&self) -> Result<SparseLu<K, T>, Singular> {
        assert!(
            self.shape.0 == self.shape.1,
            "factoring needs a square field map, not {} <- {} sites",
            self.shape.0,
            self.shape.1
        );
        let sites = self.shape.0;
        let a = self.assemble();
        let order = minimum_degree(self.site_graph());
        let q: Vec<usize> = order
            .iter()
            .flat_map(|&s| (0..K::N).map(move |b| s * K::N + b))
            .collect();
        let (l, u, pinv) = lu(&a, &q, K::N)?;
        Ok(SparseLu {
            sites,
            q,
            pinv,
            l,
            u,
            marker: PhantomData,
        })
    }

    /// Solve `self.apply(x) == b` exactly, by [`factor`](Self::factor) and
    /// [`SparseLu::solve`]: any nonsingular square map. Factor once with `factor` to solve for
    /// several right-hand sides.
    ///
    /// # Panics
    /// If the map is not square, `b` is not over its sites, or the map is singular.
    #[must_use]
    #[track_caller]
    pub fn solve_direct(&self, b: &Field<K::Mv<(), T>>) -> Field<K::Mv<(), T>> {
        match self.factor() {
            Ok(lu) => lu.solve(b),
            Err(e) => panic!("SparseMap::solve_direct: {e}"),
        }
    }

    /// The sites' graph: each site's neighbours (by a coupling either way), sorted.
    fn site_graph(&self) -> Vec<Vec<usize>> {
        let n = self.shape.0;
        let mut adj: Vec<Vec<usize>> = (0..n).map(|_| Vec::new()).collect();
        for (r, c, _) in self.couplings() {
            if r != c {
                adj[r].push(c);
                adj[c].push(r);
            }
        }
        for a in &mut adj {
            a.sort_unstable();
            a.dedup();
        }
        adj
    }

    /// The block matrix's coefficients in compressed columns, duplicates summed and exact zeros
    /// dropped.
    fn assemble(&self) -> Csc<T> {
        let n = K::N;
        let m = self.shape.0 * n;
        // Entries by column: (row, value).
        let mut by_col: Vec<Vec<(usize, T)>> = (0..m).map(|_| Vec::new()).collect();
        for (r, c, cell) in self.couplings() {
            let cell = cell.coeffs().as_ref();
            for (o, row) in cell.iter().enumerate() {
                for (i, &v) in row.as_ref().iter().enumerate() {
                    if v != T::zero() {
                        by_col[c * n + i].push((r * n + o, v));
                    }
                }
            }
        }
        let mut p = Vec::with_capacity(m + 1);
        let (mut rows, mut vals) = (Vec::new(), Vec::new());
        for mut col in by_col {
            p.push(rows.len());
            col.sort_by_key(|e| e.0);
            for (r, v) in col {
                if rows.len() > *p.last().expect("pushed") && *rows.last().expect("nonempty") == r {
                    let last = vals.len() - 1;
                    vals[last] = vals[last] + v;
                } else {
                    rows.push(r);
                    vals.push(v);
                }
            }
        }
        p.push(rows.len());
        Csc {
            p,
            i: rows,
            x: vals,
        }
    }
}

/// The sites in minimum-degree order: one at a time, the site with the fewest neighbours left
/// in the elimination graph, its neighbours then joined into a clique (the fill its
/// elimination makes). Ties go to the lower site, so the order is deterministic.
fn minimum_degree(adj: Vec<Vec<usize>>) -> Vec<usize> {
    use core::cmp::Reverse;
    use std::collections::BinaryHeap;
    let n = adj.len();
    let mut adj = adj;
    let mut eliminated = vec![false; n];
    let mut degree: Vec<usize> = adj.iter().map(Vec::len).collect();
    let mut heap: BinaryHeap<Reverse<(usize, usize)>> =
        (0..n).map(|v| Reverse((degree[v], v))).collect();
    let mut order = Vec::with_capacity(n);
    while let Some(Reverse((d, v))) = heap.pop() {
        if eliminated[v] || d != degree[v] {
            continue; // a stale entry
        }
        eliminated[v] = true;
        order.push(v);
        let clique = core::mem::take(&mut adj[v]);
        for &u in &clique {
            // u's neighbours, less v, joined with the rest of the clique.
            let merged = union_without(&adj[u], &clique, u, v);
            degree[u] = merged.len();
            adj[u] = merged;
            heap.push(Reverse((degree[u], u)));
        }
    }
    order
}

/// The sorted union of `a` and `b`, without `skip` and `gone`.
fn union_without(a: &[usize], b: &[usize], skip: usize, gone: usize) -> Vec<usize> {
    let mut out = Vec::with_capacity(a.len() + b.len());
    let (mut i, mut j) = (0, 0);
    while i < a.len() || j < b.len() {
        let next = match (a.get(i), b.get(j)) {
            (Some(&x), Some(&y)) if x == y => {
                i += 1;
                j += 1;
                x
            }
            (Some(&x), Some(&y)) if x < y => {
                i += 1;
                x
            }
            (Some(&x), None) => {
                i += 1;
                x
            }
            (Some(_) | None, Some(&y)) => {
                j += 1;
                y
            }
            (None, None) => break,
        };
        if next != skip && next != gone {
            out.push(next);
        }
    }
    out
}

/// Left-looking LU with threshold partial pivoting, the columns of `a` taken in the order `q`:
/// `(L, U, pinv)`. `blades` coefficients make a site, for the error.
#[allow(clippy::type_complexity)]
fn lu<T: Real>(
    a: &Csc<T>,
    q: &[usize],
    blades: usize,
) -> Result<(Csc<T>, Csc<T>, Vec<usize>), Singular> {
    let n = q.len();
    let none = usize::MAX;
    let nnz = a.i.len();
    let mut l = Csc {
        p: Vec::with_capacity(n + 1),
        i: Vec::with_capacity(2 * nnz + n),
        x: Vec::with_capacity(2 * nnz + n),
    };
    let mut u = Csc {
        p: Vec::with_capacity(n + 1),
        i: Vec::with_capacity(2 * nnz + n),
        x: Vec::with_capacity(2 * nnz + n),
    };
    let mut pinv = vec![none; n];
    let mut x = vec![T::zero(); n];
    let mut pattern = Vec::with_capacity(n);
    let mut search = Reach::new(n);
    let tol = T::from_f64(DIAGONAL_PREFERENCE);
    for (k, &col) in q.iter().enumerate() {
        l.p.push(l.i.len());
        u.p.push(u.i.len());
        // x = L \ A(:, col) on the rows reachable from the column's pattern.
        search.reach(&l, a, col, &pinv, &mut pattern);
        for p in a.p[col]..a.p[col + 1] {
            x[a.i[p]] = a.x[p];
        }
        // Topological order: a row's column of L is applied before the rows it reaches.
        for &j in pattern.iter().rev() {
            let jj = pinv[j];
            if jj == none {
                continue;
            }
            let xj = x[j];
            if xj == T::zero() {
                continue;
            }
            for p in l.p[jj] + 1..l.p[jj + 1] {
                x[l.i[p]] = (-l.x[p]).mul_add(xj, x[l.i[p]]);
            }
        }
        // The rows already pivotal go to U; of the others, the pivot.
        let (mut pivot_row, mut largest) = (none, T::zero());
        for &i in &pattern {
            if pinv[i] == none {
                let v = x[i].abs();
                if T::all_lt(largest, v) {
                    largest = v;
                    pivot_row = i;
                }
            } else if x[i] != T::zero() {
                u.i.push(pinv[i]);
                u.x.push(x[i]);
            }
        }
        if pivot_row == none {
            return Err(Singular {
                site: col / blades,
                blade: col % blades,
            });
        }
        // The diagonal, if it is close enough to the largest.
        if pinv[col] == none && !T::all_lt(x[col].abs(), tol * largest) {
            pivot_row = col;
        }
        let pivot = x[pivot_row];
        u.i.push(k);
        u.x.push(pivot);
        pinv[pivot_row] = k;
        l.i.push(pivot_row);
        l.x.push(T::one());
        let inv = pivot.recip();
        for &i in &pattern {
            if pinv[i] == none && x[i] != T::zero() {
                l.i.push(i);
                l.x.push(x[i] * inv);
            }
            x[i] = T::zero();
        }
    }
    l.p.push(l.i.len());
    u.p.push(u.i.len());
    // L's rows in factor order.
    for i in &mut l.i {
        *i = pinv[*i];
    }
    Ok((l, u, pinv))
}

/// Depth-first search for the nonzero pattern of a sparse triangular solve.
struct Reach {
    marked: Vec<bool>,
    stack: Vec<(usize, usize)>,
}

impl Reach {
    fn new(n: usize) -> Self {
        Reach {
            marked: vec![false; n],
            stack: Vec::new(),
        }
    }

    /// The rows reachable from column `col` of `a` through the columns of `l` already
    /// computed, in reverse topological order (finish order), into `pattern`.
    fn reach<T>(
        &mut self,
        l: &Csc<T>,
        a: &Csc<T>,
        col: usize,
        pinv: &[usize],
        pattern: &mut Vec<usize>,
    ) {
        pattern.clear();
        for p in a.p[col]..a.p[col + 1] {
            let start = a.i[p];
            if self.marked[start] {
                continue;
            }
            self.marked[start] = true;
            self.stack.push((start, first(l, pinv, start)));
            while let Some(&mut (j, ref mut next)) = self.stack.last_mut() {
                let end = last(l, pinv, j);
                let mut descended = false;
                while *next < end {
                    let i = l.i[*next];
                    *next += 1;
                    if !self.marked[i] {
                        self.marked[i] = true;
                        let f = first(l, pinv, i);
                        self.stack.push((i, f));
                        descended = true;
                        break;
                    }
                }
                if !descended {
                    self.stack.pop();
                    pattern.push(j);
                }
            }
        }
        for &j in pattern.iter() {
            self.marked[j] = false;
        }
    }
}

/// The first entry of row `j`'s column of `L` past its diagonal (none if `j` is not pivotal).
fn first<T>(l: &Csc<T>, pinv: &[usize], j: usize) -> usize {
    match pinv.get(j) {
        Some(&jj) if jj != usize::MAX => l.p[jj] + 1,
        _ => 0,
    }
}

/// The end of row `j`'s column of `L` (none if `j` is not pivotal).
fn last<T>(l: &Csc<T>, pinv: &[usize], j: usize) -> usize {
    match pinv.get(j) {
        Some(&jj) if jj != usize::MAX => l.p[jj + 1],
        _ => 0,
    }
}
