//! Fields over a fixed number of sites (ADR-045): numga's bead string as a field kind,
//! `Sites<Vector, 4>`, checked against the run-time fields and sparse maps of `tests/fields.rs`.

#![cfg(feature = "vga3d")]

use gax::SingularValues;
use gax::field::{Field, SparseMap};
use gax::sites::{SiteField, Sites};
use gax::vga3d::{Bivector, Scalar, Vector};

type V = Vector<(), f64>;
type String4 = Sites<Vector, 4>;
type Beads = SiteField<Vector, 4, (), f64>;
type BeadMap = SiteField<Vector, 4, (String4,), f64>;

/// The bead string's weights (`tests/fixtures/numga_fields.txt`): each bead tied to its
/// neighbours and to its rest position.
const WEIGHTS: [[f64; 4]; 4] = [
    [2.0, -1.0, 0.0, 0.0],
    [-1.0, 3.0, -1.0, 0.0],
    [0.0, -1.0, 3.0, -1.0],
    [0.0, 0.0, -1.0, 2.0],
];

/// The stiffness as a run-time sparse map, map cells `Vector <- Vector`.
fn sparse_stiffness() -> SparseMap<Vector<(Vector,), f64>> {
    let mut couplings = (vec![], vec![], vec![]);
    for (i, row) in WEIGHTS.iter().enumerate() {
        for (j, &w) in row.iter().enumerate().filter(|(_, w)| **w != 0.0) {
            couplings.0.push(Vector::slot().gp(w));
            couplings.1.push(i);
            couplings.2.push(j);
        }
    }
    SparseMap::new(couplings.0, couplings.1, couplings.2, (4, 4))
}

/// The same stiffness as one field map `Vector[4] <- Vector[4]`.
fn stiffness() -> BeadMap {
    SiteField::from_blocks(|i, j| Vector::slot().gp(WEIGHTS[i][j]))
}

fn pulls() -> Beads {
    SiteField::from_sites([
        Vector::new(0.0, 0.0, -1.0),
        Vector::new(0.5, 0.0, -1.0),
        Vector::new(0.0, 0.25, -1.0),
        Vector::new(-1.0, 0.0, 2.0),
    ])
}

fn assert_close(a: &[f64], b: &[f64], tol: f64) {
    assert_eq!(a.len(), b.len());
    for (k, (x, y)) in a.iter().zip(b).enumerate() {
        assert!((x - y).abs() <= tol, "coefficient {k}: {x} vs {y}");
    }
}

/// A field map's entries, row by row.
fn entries(m: &BeadMap) -> Vec<f64> {
    m.c.as_ref()
        .iter()
        .flat_map(AsRef::as_ref)
        .copied()
        .collect()
}

fn coefficients(f: &Field<V>) -> Vec<f64> {
    f.iter().flat_map(|v| v.c).collect()
}

#[test]
fn sparse_and_fixed_agree() {
    let k = stiffness();
    assert_eq!(SiteField::from_sparse(&sparse_stiffness()), k);
    assert_eq!(k.block(1, 2), Vector::slot().gp(-1.0));
    assert_eq!(k.block(0, 3), Vector::slot().gp(0.0));

    let b = pulls();
    let x = k.solve(b);
    assert_close(k.of(x).c.as_ref(), b.c.as_ref(), 1e-14);
    let sparse = sparse_stiffness().solve_direct(&b.to_field());
    assert_close(x.c.as_ref(), &coefficients(&sparse), 1e-14);
    assert_eq!(Beads::from_field(&x.to_field()), x);
}

#[test]
fn inverse_det_and_composition() {
    let k = stiffness();
    let identity = BeadMap::from_blocks(|i, j| Vector::slot().gp(if i == j { 1.0 } else { 0.0 }));
    assert_close(&entries(&k.inverse().of(k)), &entries(&identity), 1e-14);

    // Composition sums over the shared sites, as the sparse product does.
    let sparse = sparse_stiffness();
    let squared: SparseMap<Vector<(Vector,), f64>> = sparse.compose(&sparse);
    assert_eq!(k.of(k), SiteField::from_sparse(&squared));

    // The vector blocks are `w` times the identity: det = det(w)³ (det(w) = 21 for these weights).
    assert!((k.det() - 21f64.powi(3)).abs() < 1e-9);
}

#[test]
fn eigh_matches_the_sparse_solver() {
    let (values, modes) = stiffness().eigh();
    let metric = SparseMap::from_diagonal(&Field::from_fn(4, |_| Vector::slot()));
    let (sparse, _) = sparse_stiffness().eigh(&metric, 12);
    assert_close(values.as_ref(), &sparse, 1e-12);
    // Each mode is an eigenvector of the whole coupled system.
    let k = stiffness();
    for (i, mode) in modes.as_ref().iter().enumerate() {
        let residual = k.of(*mode) - mode.scaled(values[i]);
        assert!(
            residual.c.as_ref().iter().all(|r| r.abs() < 1e-12),
            "mode {i}"
        );
    }
    // Symmetric positive definite: the singular values are the eigenvalues, descending.
    let singular = SingularValues::svd(k).values;
    let descending: Vec<f64> = sparse.iter().rev().copied().collect();
    assert_close(singular.as_ref(), &descending, 1e-12);
}

#[test]
fn products_act_site_by_site() {
    let beads = pulls();
    let turn = Bivector::<(), f64>::new(0.0, 0.0, 0.3).exp();
    let turned = turn >> beads;
    let lengths = beads | beads;
    for i in 0..4 {
        assert_eq!(turned.site(i), turn >> beads.site(i));
        assert_eq!(lengths.site(i), beads.site(i) | beads.site(i));
        assert_eq!((beads * Scalar::new(2.0)).site(i), beads.site(i).gp(2.0));
        assert_eq!(
            (Vector::new(1.0, 0.0, 0.0) ^ beads).site(i),
            Vector::new(1.0, 0.0, 0.0) ^ beads.site(i)
        );
    }
    assert_eq!(turned.map_sites(|v| v | v).sum(), lengths.sum());
}

#[test]
#[should_panic(expected = "a field over 3 sites read as a field over 4")]
fn site_count_checked_at_run_time_boundaries() {
    let _ = Beads::from_field(&Field::<V>::zeros(3));
}
