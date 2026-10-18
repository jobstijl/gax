//! The invariants of a simplicial complex, shared by `tests/meshes.rs` and the `mesh_complex`
//! fuzz target: built from any list of top simplices, `SimplicialComplex::new` returns an error
//! or a complex whose boundary maps compose to zero, whose faces are its own, and whose Betti
//! numbers and Euler characteristic agree (Euler–Poincaré).

use gax::mesh::{MeshError, SimplicialComplex};
use gax::pga3d::Scalar;

type S = Scalar<(), f64>;

/// Check every invariant of the complex of `vertices` vertices and `top` simplices, or that its
/// construction failed for a reason that holds.
pub fn check<const N: usize>(vertices: usize, top: &[[usize; N]]) {
    let complex = match SimplicialComplex::new(vertices, top) {
        Ok(c) => c,
        Err(MeshError::VertexOutOfRange {
            simplex, vertex, ..
        }) => {
            assert!(top[simplex].contains(&vertex) && vertex >= vertices);
            return;
        }
        Err(MeshError::RepeatedVertex { simplex, vertex }) => {
            assert!(top[simplex].iter().filter(|&&v| v == vertex).count() > 1);
            return;
        }
        Err(MeshError::DuplicateSimplex { first, second }) => {
            let (mut a, mut b) = (top[first], top[second]);
            a.sort_unstable();
            b.sort_unstable();
            assert!(first != second && a == b);
            return;
        }
    };
    assert_eq!(complex.len(0), vertices);
    assert_eq!(complex.len(N - 1), top.len());
    for k in 1..N {
        for i in 0..complex.len(k) {
            let s = complex.simplex(k, i);
            for (j, &(face, _)) in complex.faces(k, i).iter().enumerate() {
                // The face opposite vertex j holds the other vertices.
                let f = complex.simplex(k - 1, face);
                assert!(s.iter().enumerate().all(|(m, v)| m == j || f.contains(v)));
            }
        }
    }
    for k in 0..N.saturating_sub(2) {
        let dd = &complex.d::<S>(k + 1) * &complex.d::<S>(k);
        assert!(
            dd.couplings().all(|(_, _, c)| c.s() == 0.0),
            "d({}) ∘ d({k}) ≠ 0",
            k + 1
        );
    }
    let betti = complex.betti_numbers();
    let alternating: i64 = betti
        .iter()
        .enumerate()
        .map(|(k, &b)| if k % 2 == 0 { b as i64 } else { -(b as i64) })
        .sum();
    assert_eq!(alternating, complex.euler_characteristic());
}
