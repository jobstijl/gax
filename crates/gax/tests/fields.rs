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

/// The batch form agrees with the scalar one (to rounding: it fuses multiply-adds), for a
/// string long enough to fill lanes and leave a remainder.
#[cfg(feature = "batch")]
#[test]
fn batch_apply_agrees() {
    use gax::ApproxEq;
    let n: usize = 37;
    let (mut cells, mut rows, mut cols) = (vec![], vec![], vec![]);
    for i in 0..n {
        for j in i.saturating_sub(1)..(i + 2).min(n) {
            let k = (3 * i + j) as f64;
            cells.push(Vector::<(Vector,), f64>::from_coeffs(core::array::from_fn(
                |o| core::array::from_fn(|q| ((k + (o * 3 + q) as f64) * 0.37).sin()),
            )));
            rows.push(i);
            cols.push(j);
        }
    }
    let map = SparseMap::new(cells, rows, cols, (n, n));
    let x = Field::from_fn(n, |i| Vector::new((i as f64).sin(), 1.0, i as f64 * 0.1));
    let (scalar, batch, packed) = (map.apply(&x), map.apply_batch(&x), map.packed().apply(&x));
    for ((a, b), c) in scalar.sites().iter().zip(batch.sites()).zip(packed.sites()) {
        assert!(a.approx_eq(b, 1e-14), "{a:?} vs {b:?}");
        assert!(a.approx_eq(c, 1e-14), "{a:?} vs {c:?}");
    }
}

/// Matching site counts are checked at run time, with both counts in the message.
#[test]
#[should_panic(expected = "a field map with 4 input sites applied to a field over 3 sites")]
fn site_counts_must_match() {
    let _ = stiffness().apply(&Field::<V>::zeros(3));
}
