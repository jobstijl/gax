//! Fields and sparse field maps (ADR-046, a spike): numga's bead string (`docs/fields.md`),
//! checked against numga's output (`docs/research/experiments/numga_fields.py`).

#![cfg(feature = "vga3d")]
// Index loops read like numga's flattened kernels.
#![allow(clippy::needless_range_loop)]

use gax::Real;
use gax::dual::hessian;
use gax::field::{Field, SparseMap};
use gax::vga3d::{Bivector, Scalar, Vector};

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

/// The residual `‖A x − b‖²` of a solve.
fn residual(map: &SparseMap<Vector<(Vector,), f64>>, x: &Field<V>, b: &Field<V>) -> f64 {
    let r = &map.apply(x) - b;
    r.dot(&r)
}

/// Conjugate gradients and the sparse LU agree on the string's stiffness, to rounding.
#[test]
fn solvers_agree() {
    use gax::ApproxEq;
    let k = stiffness();
    let loads = vectors(&fixture("loads"));
    let (cg, direct) = (k.cg(&loads, 1e-15, 100), k.solve_direct(&loads));
    assert!(cg.converged && cg.iterations <= 12, "{cg:?}");
    for (a, b) in cg.x.sites().iter().zip(direct.sites()) {
        assert!(a.approx_eq(b, 1e-14), "{a:?} vs {b:?}");
    }
    assert_field(&direct, &fixture("displacements"), 1e-15);
}

/// The sparse LU solves non-symmetric maps, and maps whose diagonal is zero (pivoting).
#[test]
fn direct_solves_what_cg_cannot() {
    // Non-symmetric and diagonally dominant.
    let map =
        &ragged(23) + &SparseMap::from_diagonal(&Field::from_fn(23, |_| Vector::slot().gp(6.0)));
    let b = Field::from_fn(23, |i| Vector::new(1.0, -(i as f64), 0.5));
    let x = map.solve_direct(&b);
    assert!(residual(&map, &x, &b) < 1e-24);
    // A cyclic shift of the sites: every diagonal block is zero.
    let n = 9;
    let shift = SparseMap::new(
        (0..n).map(|i| Vector::slot().gp(1.0 + i as f64)).collect(),
        (0..n).collect(),
        (0..n).map(|i| (i + 1) % n).collect(),
        (n, n),
    );
    let b = Field::from_fn(n, |i| Vector::new(i as f64, 1.0, -1.0));
    let x = shift.solve_direct(&b);
    assert!(residual(&shift, &x, &b) < 1e-24);
}

/// A saddle-point system: the string's stiffness with a constraint that ties the first and the
/// last bead together, through a Lagrange multiplier site. Symmetric and indefinite, so
/// conjugate gradients fails where the sparse LU does not.
#[test]
fn saddle_point() {
    let k = stiffness();
    let id = |w: f64| Vector::<(), f64>::slot().gp(w);
    let mut triples: Vec<(usize, usize, Vector<(Vector,), f64>)> = k.couplings().collect();
    // Site 4 is the multiplier: x0 − x3 = 0.
    triples.extend([
        (4, 0, id(1.0)),
        (4, 3, id(-1.0)),
        (0, 4, id(1.0)),
        (3, 4, id(-1.0)),
    ]);
    let (rows, rest): (Vec<_>, Vec<_>) = triples.iter().map(|t| (t.0, (t.1, t.2))).unzip();
    let (cols, cells): (Vec<_>, Vec<_>) = rest.into_iter().unzip();
    let kkt = SparseMap::new(cells, rows, cols, (5, 5));
    let mut loads = vectors(&fixture("loads")).into_sites();
    loads[0] = Vector::new(0.0, 1.0, 0.0);
    loads.push(Vector::zero());
    let b = Field::new(loads);
    let x = kkt.solve_direct(&b);
    assert!(residual(&kkt, &x, &b) < 1e-26);
    assert!((x.site(0) - x.site(3)).c.iter().all(|d| d.abs() < 1e-15));
    assert!(!kkt.cg(&b, 1e-12, 100).converged);
}

/// The factors solve again for other right-hand sides; a singular map is reported, with the
/// site whose column found no pivot.
#[test]
fn factor_once_singular_reported() {
    let k = stiffness();
    let lu = k.factor().expect("nonsingular");
    for j in 0..4 {
        let b = Field::from_fn(4, |i| Vector::new(f64::from(u8::from(i == j)), 0.5, -1.0));
        assert!(residual(&k, &lu.solve(&b), &b) < 1e-28);
    }
    // ragged() leaves every fifth row empty: no pivot can be found for all its columns.
    let singular = ragged(10);
    let err = singular.factor().expect_err("singular");
    assert!(err.site < 10 && err.blade < 3, "{err}");
    assert!(err.to_string().contains("singular"));
}

/// A mesh: a 30 × 30 grid of sites, each tied to its four neighbours and to its rest position.
/// The sparse LU and conjugate gradients agree, and the ordering keeps the fill near the band
/// of a grid (`n√n`), far below a dense factor (`n²`).
#[test]
fn grid_mesh() {
    let side = 30;
    let n = side * side;
    let id = |w: f64| Vector::<(), f64>::slot().gp(w);
    let (mut cells, mut rows, mut cols) = (vec![], vec![], vec![]);
    for r in 0..side {
        for c in 0..side {
            let s = r * side + c;
            let mut neighbours = vec![];
            if r > 0 {
                neighbours.push(s - side);
            }
            if r + 1 < side {
                neighbours.push(s + side);
            }
            if c > 0 {
                neighbours.push(s - 1);
            }
            if c + 1 < side {
                neighbours.push(s + 1);
            }
            cells.push(id(1.0 + neighbours.len() as f64));
            rows.push(s);
            cols.push(s);
            for t in neighbours {
                cells.push(id(-1.0));
                rows.push(s);
                cols.push(t);
            }
        }
    }
    let grid = SparseMap::new(cells, rows, cols, (n, n));
    let b = Field::from_fn(n, |i| Vector::new((i as f64).sin(), 1.0, 0.0));
    let lu = grid.factor().expect("nonsingular");
    let (direct, cg) = (lu.solve(&b), grid.solve(&b));
    assert!(residual(&grid, &direct, &b) < 1e-20);
    let d = &direct - &cg;
    assert!(d.dot(&d) < 1e-20 * direct.dot(&direct));
    let unknowns = 3 * n;
    assert!(lu.fill() < 3 * unknowns * side * 2, "fill {}", lu.fill());
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
    let f: Field<V> = (0..4)
        .map(|i| Vector::new(f64::from(i), 1.0, 0.0))
        .collect();
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

/// A free chain of `n` sites: the Laplacian of its `n − 1` springs, map cells `Vector <- Vector`.
fn free_chain(n: usize) -> SparseMap<Vector<(Vector,), f64>> {
    let s = Scalar::<(), f64>::new;
    let ends = SparseMap::new(
        (0..n - 1).flat_map(|_| [s(-1.0), s(1.0)]).collect(),
        (0..n - 1).flat_map(|e| [e, e]).collect(),
        (0..n - 1).flat_map(|e| [e, e + 1]).collect(),
        (n - 1, n),
    );
    &(&ends.reverse() * &ends) * Vector::slot()
}

/// A free chain's spectrum is `2 − 2 cos(kπ/n)`, each value three times on vectors: the block
/// Lanczos finds every copy, in order, orthonormal in the metric.
#[test]
fn eigh_free_chain() {
    let n = 40;
    let k = free_chain(n);
    let mass = SparseMap::from_diagonal(&Field::from_fn(n, |_| Vector::<(), f64>::slot()));
    let (values, modes) = k.eigh(&mass, 9);
    for (i, v) in values.iter().enumerate() {
        let wave = (i / 3) as f64;
        let want = 2.0 - 2.0 * (wave * std::f64::consts::PI / n as f64).cos();
        assert!(
            (v - want).abs() < 1e-7_f64.max(1e-12 * want),
            "{i}: {v} vs {want}"
        );
    }
    for (i, a) in modes.iter().enumerate() {
        for (j, b) in modes.iter().enumerate() {
            let want = if i == j { 1.0 } else { 0.0 };
            assert!((a.dot(&mass.apply(b)) - want).abs() < 1e-9, "{i} {j}");
        }
    }
}

/// On a long chain the gap above the rigid shift is small (`(π/n)²` of the scale), so the
/// eigenvectors must be accurate well beyond the residual: each mode lies in its exact eigenspace,
/// the cosine wave `cos((j + ½) kπ / n)` on each coefficient. (Without the final step of inverse
/// iteration, the first wave was 1.5·10⁻¹¹ outside.)
#[test]
fn eigh_eigenvectors_across_a_small_gap() {
    let n = 400;
    let mass = SparseMap::from_diagonal(&Field::from_fn(n, |_| Vector::<(), f64>::slot()));
    let (values, modes) = free_chain(n).eigh(&mass, 6);
    for (i, (value, mode)) in values.iter().zip(&modes).enumerate() {
        let wave = (i / 3) as f64;
        let exact = |j: usize| (((j as f64 + 0.5) * wave * std::f64::consts::PI) / n as f64).cos();
        let norm2: f64 = (0..n).map(|j| exact(j) * exact(j)).sum();
        // The mode less its projection on the exact eigenspace (each coefficient separately).
        let outside: f64 = (0..3)
            .map(|k| {
                let along: f64 = (0..n).map(|j| mode.site(j).c[k] * exact(j)).sum::<f64>() / norm2;
                (0..n)
                    .map(|j| (mode.site(j).c[k] - along * exact(j)).powi(2))
                    .sum::<f64>()
            })
            .sum();
        let want = 2.0 - 2.0 * (wave * std::f64::consts::PI / n as f64).cos();
        assert!(
            outside.sqrt() < 1e-11,
            "mode {i}: {:e} outside its eigenspace",
            outside.sqrt()
        );
        assert!((value - want).abs() < 1e-14, "{i}: {value} vs {want}");
    }
}

/// The stopping test does not depend on the pencil's scale: a chain of stiffness `10¹¹` has the
/// spectrum of the unit chain, scaled, to rounding (a stiff girder's modes once came out
/// `5·10⁻⁴` off).
#[test]
fn eigh_is_scale_invariant() {
    let n = 40;
    let stiffness = 1e11;
    let unit = free_chain(n);
    let stiff = unit.map_cells(|c| c.gp(stiffness));
    let mass = SparseMap::from_diagonal(&Field::from_fn(n, |_| Vector::<(), f64>::slot()));
    let (values, _) = unit.eigh(&mass, 12);
    let (stiff_values, _) = stiff.eigh(&mass, 12);
    for (v, s) in values.iter().zip(&stiff_values) {
        assert!(
            (s / stiffness - v).abs() < 1e-12,
            "{v} vs {}",
            s / stiffness
        );
    }
}

/// A generalized problem with varying masses, and a square grid whose modes along its two axes
/// coincide (six copies on vectors): residuals `A x − λ M x` at rounding, the metric orthonormal.
#[test]
fn eigh_generalized_and_degenerate() {
    let side = 12;
    let n = side * side;
    let id = |w: f64| Vector::<(), f64>::slot().gp(w);
    let (mut cells, mut rows, mut cols) = (vec![], vec![], vec![]);
    for r in 0..side {
        for c in 0..side {
            let s = r * side + c;
            let mut neighbours = vec![];
            if r > 0 {
                neighbours.push(s - side);
            }
            if r + 1 < side {
                neighbours.push(s + side);
            }
            if c > 0 {
                neighbours.push(s - 1);
            }
            if c + 1 < side {
                neighbours.push(s + 1);
            }
            cells.push(id(neighbours.len() as f64));
            rows.push(s);
            cols.push(s);
            for t in neighbours {
                cells.push(id(-1.0));
                rows.push(s);
                cols.push(t);
            }
        }
    }
    let grid = SparseMap::new(cells, rows, cols, (n, n));
    for (name, mass) in [
        (
            "unit",
            SparseMap::from_diagonal(&Field::from_fn(n, |_| id(1.0))),
        ),
        (
            "varying",
            SparseMap::from_diagonal(&Field::from_fn(n, |i| id(1.0 + (i % 5) as f64 * 0.25))),
        ),
    ] {
        let (values, modes) = grid.eigh(&mass, 12);
        for (v, x) in values.iter().zip(&modes) {
            let r = &grid.apply(x) - &mass.apply(x).scaled(*v);
            assert!(
                r.dot(&r).sqrt() < 1e-7,
                "{name}: residual {} at {v}",
                r.dot(&r).sqrt()
            );
            assert!((x.dot(&mass.apply(x)) - 1.0).abs() < 1e-9);
        }
        // The rigid shifts (three), then the first wave along each axis (six together).
        assert!(
            values[..3].iter().all(|v| v.abs() < 1e-7),
            "{name}: {values:?}"
        );
        if name == "unit" {
            let wave = 2.0 - 2.0 * (std::f64::consts::PI / side as f64).cos();
            assert!(
                values[3..9].iter().all(|v| (v - wave).abs() < 1e-10),
                "{values:?}"
            );
        }
    }
}

/// Rectangular maps from bivectors to vectors, `out × inp` sites, with every cell its own, and
/// a strong identity on the leading diagonal so that the map has full rank (columns or rows).
fn rectangular(out: usize, inp: usize) -> SparseMap<Vector<(Bivector,), f64>> {
    let (mut cells, mut rows, mut cols) = (vec![], vec![], vec![]);
    for r in 0..out.min(inp) {
        cells.push(Vector::<(Bivector,), f64>::from_coeffs(
            core::array::from_fn(|o| core::array::from_fn(|i| if o == i { 4.0 } else { 0.0 })),
        ));
        rows.push(r);
        cols.push(r);
    }
    for r in 0..out {
        for c in (0..inp).filter(|c| (r * 5 + c * 3) % 7 < 3) {
            let k = (r * inp + c) as f64;
            cells.push(Vector::<(Bivector,), f64>::from_coeffs(
                core::array::from_fn(|o| {
                    core::array::from_fn(|i| ((k + (o * 3 + i) as f64) * 0.71).sin())
                }),
            ));
            rows.push(r);
            cols.push(c);
        }
    }
    SparseMap::new(cells, rows, cols, (out, inp))
}

/// Least squares: a consistent over-determined system is solved exactly, an inconsistent one
/// as the normal equations solve it, an under-determined one with the least norm, and a
/// singular one with its gauge at zero.
#[test]
fn lstsq_cases() {
    // Over-determined and consistent.
    let tall = rectangular(30, 12);
    let truth = Field::from_fn(12, |i| Bivector::new(1.0, -(i as f64) * 0.1, 0.5));
    let b = tall.apply(&truth);
    let x = tall.lstsq(&b);
    let d = &x - &truth;
    assert!(d.dot(&d) < 1e-20, "{}", d.dot(&d));
    // Inconsistent: against the normal equations, Aᵀ A x = Aᵀ b.
    let noisy = Field::from_fn(30, |i| Vector::new((i as f64).cos(), 1.0, -0.5));
    let x = tall.lstsq(&noisy);
    let normal = tall.transpose().compose(&tall);
    let want = normal.solve_direct(&tall.transpose().apply(&noisy));
    let d = &x - &want;
    assert!(d.dot(&d) < 1e-18 * want.dot(&want), "{}", d.dot(&d));
    // Under-determined: the least-norm solution is Aᵀ (A Aᵀ)⁻¹ b.
    let wide = rectangular(8, 20);
    let b = Field::from_fn(8, |i| Vector::new(1.0, i as f64, 0.0));
    let x = wide.lstsq(&b);
    let y = wide.compose(&wide.transpose()).solve_direct(&b);
    let want = wide.transpose().apply(&y);
    let d = &x - &want;
    assert!(d.dot(&d) < 1e-18 * want.dot(&want), "{}", d.dot(&d));
    // Singular: a free chain's Laplacian, its rigid shifts left at zero.
    let k = free_chain(15);
    let b = Field::from_fn(15, |i| {
        Vector::new(
            i as f64 - 7.0,
            0.0,
            1.0 - 2.0 * f64::from(u8::from(i % 2 == 0)) + if i == 14 { 1.0 } else { 0.0 },
        )
    });
    let b = &b - &Field::from_fn(15, |_| b.mean()); // in the range: zero mean
    let x = k.lstsq(&b);
    assert!(x.mean().c.iter().all(|c| c.abs() < 1e-10), "{:?}", x.mean());
    let r = &k.apply(&x) - &b;
    assert!(r.dot(&r) < 1e-18);
}
