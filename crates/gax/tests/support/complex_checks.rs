//! The invariants of a simplicial complex, shared by `tests/meshes.rs` and the `mesh_complex`
//! fuzz target: built from any list of top simplices, `Topology::new` returns an error
//! or a topology whose boundary maps compose to zero, whose faces are its own, and whose Betti
//! numbers and Euler characteristic agree (Euler–Poincaré); whose facets sum to its boundary
//! maps; whose boundary and interior split its simplices; whose coherent winding is oriented; and
//! whose cycles are a basis of the first homology.

use gax::Coef;
use gax::fp::Fp;
use gax::mesh::{MeshError, Topology};
use gax::pga3d::Scalar;
use std::collections::BTreeMap;

type S = Scalar<(), f64>;

/// Check every invariant of the topology of `vertices` vertices and `top` simplices, or that its
/// construction failed for a reason that holds.
pub fn check<const N: usize>(vertices: usize, top: &[[usize; N]]) {
    let topology = match Topology::new(vertices, top) {
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
    assert_eq!(topology.len(0), vertices);
    assert_eq!(topology.len(N - 1), top.len());
    for k in 1..N {
        for i in 0..topology.len(k) {
            let s = topology.simplex(k, i);
            for (j, &(face, _)) in topology.faces(k, i).iter().enumerate() {
                // The face opposite vertex j holds the other vertices.
                let f = topology.simplex(k - 1, face);
                assert!(s.iter().enumerate().all(|(m, v)| m == j || f.contains(v)));
            }
        }
    }
    for k in 0..N.saturating_sub(2) {
        let dd = &topology.d::<S>(k + 1) * &topology.d::<S>(k);
        assert!(
            dd.couplings().all(|(_, _, c)| c.s() == 0.0),
            "d({}) ∘ d({k}) ≠ 0",
            k + 1
        );
    }
    let betti = topology.betti_numbers();
    let alternating: i64 = betti
        .iter()
        .enumerate()
        .map(|(k, &b)| if k % 2 == 0 { b as i64 } else { -(b as i64) })
        .sum();
    assert_eq!(alternating, topology.euler_characteristic());

    for n in 1..N {
        let summed = topology.facets(n).assemble(|_, _| S::new(1.0));
        assert_eq!(
            sums(&summed),
            sums(&topology.d::<S>(n - 1)),
            "facets({n}) is d({})",
            n - 1
        );
    }
    if N >= 2 {
        for k in 0..=N - 2 {
            let (outside, inside) = (topology.exterior(k), topology.interior(k));
            assert_eq!(outside.len() + inside.len(), topology.len(k));
        }
    }
    match topology.coherent() {
        Some(wound) => {
            assert!(wound.is_oriented());
            for k in 0..N {
                let sorted = |t: &Topology<N>, i: usize| {
                    let mut s = t.simplex(k, i).to_vec();
                    s.sort_unstable();
                    s
                };
                assert!((0..topology.len(k)).all(|i| sorted(&wound, i) == sorted(&topology, i)));
            }
            if topology.is_oriented() {
                assert_eq!(wound, topology);
            }
        }
        None => assert!(!topology.is_oriented()),
    }
    if N >= 2 {
        let cycles = topology.cycles();
        assert_eq!(cycles.len(), betti[1], "a cycle per class of H₁");
        let circulation = topology.circulation::<S>(&cycles);
        // Each cycle is closed: its boundary, the circulation's image under d(0)ᵀ, is nothing.
        let closed = &circulation * &topology.d::<S>(0);
        assert!(
            sums(&closed).values().all(|&x| x == 0.0),
            "a cycle is closed"
        );
        // And independent of the others modulo the boundaries of the triangles.
        let mut rows: Vec<BTreeMap<usize, Fp>> = Vec::new();
        if N >= 3 {
            rows.extend(chains(&topology.d::<S>(1)));
        }
        let boundaries = rank(rows.clone());
        rows.extend(chains(&circulation));
        assert_eq!(
            rank(rows),
            boundaries + cycles.len(),
            "the cycles are independent"
        );
    }
}

/// A map's couplings, repeated ones added.
fn sums(m: &gax::field::SparseMap<S>) -> BTreeMap<(usize, usize), f64> {
    let mut out = BTreeMap::new();
    for (r, c, x) in m.couplings() {
        *out.entry((r, c)).or_insert(0.0) += x.s();
    }
    out.retain(|_, x| *x != 0.0);
    out
}

/// A map's rows as chains over `ℤ/p`.
fn chains(m: &gax::field::SparseMap<S>) -> Vec<BTreeMap<usize, Fp>> {
    let mut rows = vec![BTreeMap::new(); m.shape().0];
    for (r, c, x) in m.couplings() {
        let entry = rows[r].entry(c).or_insert(Fp::zero());
        *entry = *entry + Fp::from_i64(x.s() as i64);
    }
    rows
}

/// The rank of chains over `ℤ/p`, by elimination.
fn rank(rows: Vec<BTreeMap<usize, Fp>>) -> usize {
    let mut pivots: BTreeMap<usize, BTreeMap<usize, Fp>> = BTreeMap::new();
    for mut row in rows {
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
