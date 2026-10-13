//! Fields and sparse field maps (ADR-046, a spike): numga's bead string (`docs/fields.md`),
//! checked against numga's output (`docs/research/experiments/numga_fields.py`).

#![cfg(feature = "vga3d")]
// Index loops read like numga's flattened kernels.
#![allow(clippy::needless_range_loop)]

use gax::Real;
use gax::dual::hessian;
use gax::field::{Field, SparseMap};
use gax::vga3d::{Scalar, Vector};

type V = Vector<(), f64>;
type S = Scalar<(), f64>;

/// The fixture's numbers on the line that starts with `name` (after `|` if it has one).
fn fixture(name: &str) -> Vec<f64> {
    let text = include_str!("fixtures/numga_fields.txt");
    let line = text
        .lines()
        .find(|l| l.split_whitespace().next() == Some(name))
        .unwrap_or_else(|| panic!("no {name} in the fixture"));
    let tail = match line.split_once('|') {
        Some((_, t)) => t,
        None => line.split_once(' ').unwrap().1,
    };
    tail.split_whitespace()
        .map(|w| w.parse().unwrap())
        .collect()
}

fn vectors(c: &[f64]) -> Field<V> {
    Field::from_fn(c.len() / 3, |i| {
        Vector::new(c[3 * i], c[3 * i + 1], c[3 * i + 2])
    })
}

fn assert_field(f: &Field<V>, want: &[f64], tol: f64) {
    assert_eq!(f.len() * 3, want.len());
    for (i, v) in f.sites().iter().enumerate() {
        for k in 0..3 {
            assert!(
                (v.c[k] - want[3 * i + k]).abs() <= tol,
                "site {i}, {k}: {v:?} vs {want:?}"
            );
        }
    }
}

/// The dense blocks `[row][col][a][b]` of a field map with map cells (numga's kernel layout).
fn blocks(m: &SparseMap<Vector<(Vector,), f64>>) -> Vec<f64> {
    let (r, c) = m.shape();
    let mut out = vec![0.0; r * c * 9];
    for (i, j, cell) in m.couplings() {
        for a in 0..3 {
            for b in 0..3 {
                out[((i * c + j) * 3 + a) * 3 + b] += cell.c[a][b];
            }
        }
    }
    out
}

/// The stiffness of the string: each bead tied to its neighbours and to its rest position,
/// `weights[i][j]` times the identity on vectors, as map cells `Vector <- Vector`.
fn stiffness() -> SparseMap<Vector<(Vector,), f64>> {
    let w = fixture("weights");
    let (mut cells, mut rows, mut cols) = (vec![], vec![], vec![]);
    for i in 0..4 {
        for j in 0..4 {
            if w[4 * i + j] != 0.0 {
                cells.push(Vector::slot().gp(w[4 * i + j]));
                rows.push(i);
                cols.push(j);
            }
        }
    }
    SparseMap::new(cells, rows, cols, (4, 4))
}

/// The three springs, each the difference of its two beads: scalar cells.
fn ends() -> SparseMap<S> {
    let s = Scalar::new;
    SparseMap::new(
        vec![s(-1.0), s(1.0), s(-1.0), s(1.0), s(-1.0), s(1.0)],
        vec![0, 0, 1, 1, 2, 2],
        vec![0, 1, 1, 2, 2, 3],
        (3, 4),
    )
}

#[test]
fn forces_displacements_composition() {
    let k = stiffness();
    let loads = vectors(&fixture("loads"));
    assert_field(&k.apply(&loads), &fixture("forces"), 0.0);
    assert_field(&k.solve(&loads), &fixture("displacements"), 1e-15);
    let composed = blocks(&k.compose(&k));
    let want = fixture("composed");
    assert_eq!(composed, want);
}

#[test]
fn sparse_laplacian() {
    let positions = Field::from_fn(4, |i| Vector::new(i as f64, 0.0, 0.0));
    let ends = ends();
    assert_field(&(&ends * &positions), &fixture("stretches"), 0.0);
    let anchors = SparseMap::from_diagonal(&Field::from_fn(4, |_| Scalar::new(1.0)));
    let laplacian = &(&ends.reverse() * &ends) + &anchors;
    let sparse_stiffness = &laplacian * Vector::<(), f64>::slot();
    let loads = vectors(&fixture("loads"));
    assert_field(
        &sparse_stiffness.solve(&loads),
        &fixture("sparse_displacements"),
        1e-15,
    );
    // The same couplings as the dense weights.
    assert_eq!(blocks(&sparse_stiffness), blocks(&stiffness()));
}

/// Half the squared stretch of a spring.
fn spring<T: Real>(d: Vector<(), T>) -> T {
    d.scalar_product(d).s() * T::from_f64(0.5)
}

/// A bead's anchor spring, less the work of its load.
fn anchor<T: Real>(x: Vector<(), T>, pull: V) -> T {
    x.scalar_product(x).s() * T::from_f64(0.5) - pull.map_coefs(T::from_f64).scalar_product(x).s()
}

/// The derivatives with respect to the whole string, by the chain rule through the springs'
/// differences: the gradient `~ends g + g_anchor` (a field of forms `Scalar <- Vector`), the
/// curvature `~ends H ends + H_anchor` (form cells `Scalar <- (Vector, Vector)`), and the
/// Newton step that solves the one against the other.
#[test]
fn typed_newton_step() {
    let ends = ends();
    let pulls = vectors(&fixture("loads"));
    let beads = Field::<V>::zeros(4);
    let stretches = &ends * &beads;
    let springs: Vec<_> = stretches
        .sites()
        .iter()
        .map(|d| hessian(spring, *d))
        .collect();
    let anchors: Vec<_> = beads
        .sites()
        .iter()
        .zip(pulls.sites())
        .map(|(x, p)| hessian(|x| anchor(x, *p), *x))
        .collect();
    let slope_springs = Field::new(springs.iter().map(|d| d.1).collect());
    let slope_anchors = Field::new(anchors.iter().map(|d| d.1).collect());
    let curve_springs =
        SparseMap::from_diagonal(&Field::new(springs.iter().map(|d| d.2).collect()));
    let curve_anchors =
        SparseMap::from_diagonal(&Field::new(anchors.iter().map(|d| d.2).collect()));
    let gradient = &(&ends.reverse() * &slope_springs) + &slope_anchors;
    let curvature = &(&(&ends.reverse() * &curve_springs) * &ends) + &curve_anchors;
    // Against numga's derivative of the energy of `vector[4]`.
    let g = fixture("gradient");
    for (i, form) in gradient.sites().iter().enumerate() {
        for k in 0..3 {
            assert_eq!(form.c[0][k], g[3 * i + k], "gradient {i} {k}");
        }
    }
    assert_eq!(
        blocks(&curvature.map_cells(Scalar::as_map)),
        fixture("curvature")
    );
    // The Newton step: the curvature's cells as maps, the gradient's forms as vectors.
    let rhs = gradient.map(|f| Vector::from_coeffs(f.c[0]));
    let step = curvature.map_cells(Scalar::as_map).solve(&rhs).scaled(-1.0);
    assert_field(&step, &fixture("newton_step"), 1e-15);
}

/// The string's energy, `½ Σ w_ij x_i·x_j − Σ pull_i·x_i` over the beads.
fn string_energy<T: Real>(x: &Field<Vector<(), T>>, w: &[f64], pulls: &Field<V>) -> T {
    let n = x.len();
    let mut e = T::zero();
    for i in 0..n {
        for j in 0..n {
            e = e + T::from_f64(0.5 * w[n * i + j]) * x.site(i).scalar_product(x.site(j)).s();
        }
        e = e - pulls
            .site(i)
            .map_coefs(T::from_f64)
            .scalar_product(x.site(i))
            .s();
    }
    e
}

/// The general derivatives with respect to the whole field, against numga's: the gradient and
/// the curvature of the string's energy, every site seeded against every other.
#[test]
fn field_derivatives_against_numga() {
    let w = fixture("weights");
    let pulls = vectors(&fixture("loads"));
    let rest = Field::<V>::zeros(4);
    let (_, g) = gax::field::gradient(|x| string_energy(x, &w, &pulls), &rest);
    let (_, g2, h) = gax::field::hessian(|x| string_energy(x, &w, &pulls), &rest);
    let want = fixture("gradient");
    for (i, form) in g.sites().iter().enumerate() {
        for k in 0..3 {
            assert_eq!(form.c[0][k], want[3 * i + k], "gradient {i} {k}");
        }
    }
    assert_eq!(g, g2);
    assert_eq!(blocks(&h.map_cells(Scalar::as_map)), fixture("curvature"));
    // The coupled Newton step, the whole string at once.
    let step = -&h
        .map_cells(Scalar::as_map)
        .solve(&g.map(|f| Vector::from_coeffs(f.c[0])));
    assert_field(&step, &fixture("newton_step"), 1e-15);
}

/// A map with rows of different lengths, some empty, over a number of sites that is not a
/// multiple of the lanes.
fn ragged(n: usize) -> SparseMap<Vector<(Vector,), f64>> {
    let (mut cells, mut rows, mut cols) = (vec![], vec![], vec![]);
    for i in 0..n {
        if i % 5 == 4 {
            continue; // an empty row
        }
        for j in (0..n).filter(|j| (i * 7 + j * 3) % 11 < 2 + i % 3) {
            let k = (3 * i + j) as f64;
            cells.push(Vector::<(Vector,), f64>::from_coeffs(core::array::from_fn(
                |o| core::array::from_fn(|q| ((k + (o * 3 + q) as f64) * 0.37).sin()),
            )));
            rows.push(i);
            cols.push(j);
        }
    }
    SparseMap::new(cells, rows, cols, (n, n))
}

/// The packed form agrees with the scalar one (to rounding: it fuses multiply-adds).
#[cfg(feature = "batch")]
#[test]
fn packed_apply_agrees() {
    use gax::ApproxEq;
    for n in [1, 15, 16, 17, 37, 100] {
        let map = ragged(n);
        let x = Field::from_fn(n, |i| Vector::new((i as f64).sin(), 1.0, i as f64 * 0.1));
        let (scalar, packed) = (map.apply(&x), map.packed().apply(&x));
        for (a, b) in scalar.sites().iter().zip(packed.sites()) {
            assert!(a.approx_eq(b, 1e-13), "{n}: {a:?} vs {b:?}");
        }
    }
}

/// Padding never reads a value: a NaN at one site reaches only the rows that couple to it.
#[cfg(feature = "batch")]
#[test]
fn packed_padding_is_masked() {
    let map = ragged(37);
    let mut x = Field::from_fn(37, |i| Vector::new(1.0, i as f64, 0.0));
    x.sites_mut()[0] = Vector::new(f64::NAN, 0.0, 0.0);
    let y = map.packed().apply(&x);
    for (r, v) in y.sites().iter().enumerate() {
        let touches = map.couplings().any(|(row, col, _)| row == r && col == 0);
        assert_eq!(v.c.iter().any(|c| c.is_nan()), touches, "row {r}");
    }
}

/// Conjugate gradients and LU on the assembled matrix agree; a non-symmetric
/// map solves densely; an indefinite one is refused.
#[test]
fn solvers() {
    use gax::ApproxEq;
    let k = stiffness();
    let loads = vectors(&fixture("loads"));
    let (cg, dense) = (k.cg(&loads, 1e-15, 100), k.solve_dense(&loads));
    assert!(cg.converged && cg.iterations <= 12, "{cg:?}");
    for (a, b) in cg.x.sites().iter().zip(dense.sites()) {
        assert!(a.approx_eq(b, 1e-14));
    }
    // A non-symmetric map (diagonally dominant, so nonsingular).
    let map =
        &ragged(23) + &SparseMap::from_diagonal(&Field::from_fn(23, |_| Vector::slot().gp(6.0)));
    let b = Field::from_fn(23, |i| Vector::new(1.0, -(i as f64), 0.5));
    let x = map.solve_dense(&b);
    for (a, c) in map.apply(&x).sites().iter().zip(b.sites()) {
        assert!(a.approx_eq(c, 1e-12), "{a:?} vs {c:?}");
    }
}

#[test]
#[should_panic(expected = "conjugate gradients did not converge")]
fn solve_refuses_an_indefinite_map() {
    let indefinite = SparseMap::from_diagonal(&Field::from_fn(3, |i| {
        Vector::<(), f64>::slot().gp(if i == 1 { -1.0 } else { 1.0 })
    }));
    let _ = indefinite.solve(&Field::from_fn(3, |_| Vector::new(1.0, 1.0, 1.0)));
}

/// The transpose is the adjoint under the fields' coefficient inner product, and composition is
/// the block matrix product.
#[test]
fn transpose_and_compose() {
    let a = ragged(19);
    let b = ragged(19).transpose();
    let x = Field::from_fn(19, |i| Vector::new(1.0, (i as f64).cos(), -2.0));
    let y = Field::from_fn(19, |i| Vector::new((i as f64).sin(), 0.5, 1.0));
    let (left, right) = (y.dot(&a.apply(&x)), a.transpose().apply(&y).dot(&x));
    assert!((left - right).abs() < 1e-12 * left.abs().max(1.0));
    let (ab, seq) = (a.compose(&b).apply(&x), a.apply(&b.apply(&x)));
    for (p, q) in ab.sites().iter().zip(seq.sites()) {
        assert!(
            (*p - *q).c.iter().all(|d| d.abs() < 1e-12),
            "{p:?} vs {q:?}"
        );
    }
}

/// Reductions and site-by-site operations.
#[test]
fn field_operations() {
    let f: Field<V> = (0..4).map(|i| Vector::new(f64::from(i), 1.0, 0.0)).collect();
    assert_eq!(f.mean(), Vector::new(1.5, 1.0, 0.0));
    assert_eq!(f.sum(), Vector::new(6.0, 4.0, 0.0));
    assert_eq!((&f + &(-&f)).sum(), Vector::zero());
    assert_eq!(f.scaled(2.0).site(3), Vector::new(6.0, 2.0, 0.0));
    // `~ends` runs the springs back: the transposed incidence, its cells reversed.
    let ends = ends();
    assert_eq!(ends.reverse().shape(), (4, 3));
    assert_eq!(ends.reverse().couplings_len(), 6);
}

/// Matching site counts are checked at run time, with both counts in the message.
#[test]
#[should_panic(expected = "a field map with 4 input sites applied to a field over 3 sites")]
fn site_counts_must_match() {
    let _ = stiffness().apply(&Field::<V>::zeros(3));
}
