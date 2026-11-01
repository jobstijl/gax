//! Field maps: their arithmetic, solves of correlations and of other square cells, block
//! systems over two fields, the dense whole spectrum, and the sparse `eigh`'s shifts, refinement
//! and warm starts (ADR-046, ADR-056).

#![cfg(all(feature = "vga3d", feature = "pga2d"))]

use gax::field::{BlockMap, Field, SparseMap};
use gax::{pga2d, vga3d};

type S = vga3d::Scalar<(), f64>;
type V = vga3d::Vector<(), f64>;

/// A deterministic pseudo-random number in `[-1, 1)`.
fn noise(k: usize) -> f64 {
    let x = (k as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    let x = (x ^ (x >> 29)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    (x >> 11) as f64 / (1u64 << 52) as f64 - 1.0
}

/// The edges of a path of `n` sites, as differences `head − tail`.
fn path_ends(n: usize) -> SparseMap<S> {
    let s = S::new;
    SparseMap::new(
        (0..n - 1).flat_map(|_| [s(-1.0), s(1.0)]).collect(),
        (0..n - 1).flat_map(|e| [e, e]).collect(),
        (0..n - 1).flat_map(|e| [e, e + 1]).collect(),
        (n - 1, n),
    )
}

/// The Laplacian of an `n × n` grid of sites, its edges weighted by `w(edge)`, as numbers.
fn grid_laplacian(n: usize, w: impl Fn(usize) -> f64) -> SparseMap<S> {
    let mut triples = Vec::new();
    let mut edge = 0;
    let site = |i: usize, j: usize| i * n + j;
    for i in 0..n {
        for j in 0..n {
            for (a, b) in [(i + 1, j), (i, j + 1)] {
                if a < n && b < n {
                    let (p, q, k) = (site(i, j), site(a, b), w(edge));
                    edge += 1;
                    triples.extend([
                        (p, p, S::new(k)),
                        (q, q, S::new(k)),
                        (p, q, S::new(-k)),
                        (q, p, S::new(-k)),
                    ]);
                }
            }
        }
    }
    SparseMap::from_triples(triples, (n * n, n * n))
}

#[test]
fn field_maps_negate_scale_and_subtract() {
    let m = &path_ends(5) * V::slot();
    let x = Field::from_fn(5, |i| V::new(noise(i), noise(i + 10), noise(i + 20)));
    let y = m.of(&x);
    assert_eq!((-&m).of(&x), -&y);
    assert_eq!((-m.clone()).of(&x), -&y);
    let near = |a: Field<V>, b: Field<V>| (&a - &b).norm() < 1e-15 * b.norm();
    assert!(near((&m * 2.5).of(&x), y.scaled(2.5)));
    assert!(near((2.5 * &m).of(&x), y.scaled(2.5)));
    assert!(near((m.clone() * 2.5).of(&x), y.scaled(2.5)));
    assert!(near(m.scaled(-0.5).of(&x), y.scaled(-0.5)));
    let twice = &m + &m;
    assert!((&(&twice - &m).of(&x) - &y).norm() < 1e-15);
    assert_eq!((&m - &m).of(&x).norm(), 0.0);
    assert_eq!(
        SparseMap::<vga3d::Vector<(vga3d::Vector,), f64>>::zeros((4, 5))
            .of(&x)
            .norm(),
        0.0
    );
}

type Amount = pga2d::intrinsic::Pseudoscalar<(), f64>;
type Conductance = pga2d::intrinsic::Pseudoscalar<(pga2d::Scalar,), f64>;

/// A grid's conductance, potentials to amounts (numga's `Amount- <- Scalar+`), with every site
/// tied a little to ground: a correlation, definite in the pairing of an amount with a
/// potential.
fn conductance(n: usize, sign: f64) -> SparseMap<Conductance> {
    let unit = Amount::new(sign) * pga2d::Scalar::slot();
    let laplacian = grid_laplacian(n, |e| 1.0 + 0.5 * noise(e));
    let ground = SparseMap::from_diagonal(&Field::from_fn(n * n, |_| S::new(0.01)));
    (&laplacian + &ground).map_cells(|c| unit * c.s())
}

/// The stand-in the ports wrote: cells and amounts through the undual, to numbers.
fn through_the_undual(
    map: &SparseMap<Conductance>,
    b: &Field<Amount>,
) -> Field<pga2d::Scalar<(), f64>> {
    map.map_cells(Conductance::undual)
        .solve_direct(&b.map(Amount::undual))
}

#[test]
fn a_conductance_solves_as_a_correlation_of_either_sign() {
    let n = 12;
    let amounts = Field::from_fn(n * n, |i| Amount::new(noise(i)));
    for sign in [1.0, -1.0] {
        let g = conductance(n, sign);
        let direct = g.solve_direct(&amounts);
        let cg = g.solve(&amounts);
        let reference = through_the_undual(&g, &amounts);
        let scale = reference.norm();
        assert!((&direct - &reference).norm() < 1e-13 * scale);
        assert!((&cg - &reference).norm() < 1e-11 * scale, "sign {sign}");
        assert!((&g.of(&cg) - &amounts).norm() < 1e-12 * amounts.norm());
        let report = g.cg(&amounts, 1e-10, 1000);
        assert!(report.converged && report.residual < 1e-10);
    }
}

/// Forces (lines through the origin) to velocities (directions): PGA2D's `Direction <-
/// OriginLine`, a correlation. Every pair of beads coupled, a mobility that falls off with
/// distance, as the Stokes swimmer's.
#[test]
fn a_mobility_solves_as_a_correlation() {
    let n = 9;
    let along = pga2d::OriginLine::<(), f64>::slot()
        .dual()
        .cast::<pga2d::Direction>();
    let mobility = SparseMap::from_triples(
        (0..n * n).map(|k| {
            let (i, j) = (k / n, k % n);
            let d = (i as f64 - j as f64).abs();
            (i, j, along * (1.0 / (1.0 + d * d)))
        }),
        (n, n),
    );
    let velocities = Field::from_fn(n, |i| pga2d::Direction::new(noise(i), noise(i + 50)));
    let forces = mobility.solve(&velocities);
    let direct = mobility.solve_direct(&velocities);
    assert!((&mobility.of(&forces) - &velocities).norm() < 1e-13);
    assert!((&forces - &direct).norm() < 1e-12 * direct.norm());
}

/// Cells between kinds that do not pair (an extrinsic vector from an extrinsic bivector, the
/// metric dual's kinds, not the complement's) solve directly.
#[test]
fn other_square_cells_solve_directly() {
    let n = 30;
    let dual = vga3d::Bivector::<(), f64>::slot()
        .hodge()
        .with_orientation::<gax::Extrinsic>();
    let ends = path_ends(n);
    let coupling =
        &(&ends.reverse() * &ends) + &SparseMap::from_diagonal(&Field::from_fn(n, |_| S::new(3.0)));
    let map = coupling.map_cells(|c| dual * c.s());
    let b = Field::from_fn(n, |i| V::new(noise(i), noise(i + 7), noise(i + 13)));
    let x: Field<vga3d::Bivector<(), f64>> = map.solve_direct(&b);
    assert!((&map.of(&x) - &b).norm() < 1e-13);
    let lu = map.factor().expect("regular");
    assert_eq!(lu.solve(&b), x);
}

/// A saddle point over two fields, displacements of vectors and a constraint's forces on
/// numbers, against the same system flattened into one map of numbers.
#[test]
fn a_block_saddle_point_matches_its_flattened_map() {
    let (n, m) = (20, 4);
    let ends = path_ends(n);
    let stiffness = &(&(&ends.reverse() * &ends)
        + &SparseMap::from_diagonal(&Field::from_fn(n, |i| S::new(1.0 + noise(i).abs()))))
        * V::slot();
    // Each constraint reads the x and z of a few sites.
    let read = |k: usize, i: usize| {
        (noise(7 * k + i) > 0.2).then(|| V::new(noise(k + i), 0.0, noise(k * i + 3)))
    };
    let mut triples = Vec::new();
    for k in 0..m {
        for i in 0..n {
            if let Some(v) = read(k, i) {
                triples.push((k, i, v | V::slot()));
            }
        }
    }
    let constraint = SparseMap::from_triples(triples, (m, n));
    let spread = constraint.transpose();
    let system = BlockMap::saddle(stiffness.clone(), spread.clone(), constraint.clone());
    let f = Field::from_fn(n, |i| {
        V::new(noise(i + 100), noise(i + 200), noise(i + 300))
    });
    let g = Field::from_fn(m, |k| S::new(noise(k + 400)));
    let (x, y) = system.solve_direct(&f, &g);
    let (fx, gx) = system.of(&x, &y);
    assert!((&fx - &f).norm() < 1e-12 && (&gx - &g).norm() < 1e-12);
    // The flattened map: every coefficient a site of numbers.
    let mut flat = Vec::new();
    for (r, c, cell) in stiffness.couplings() {
        for o in 0..3 {
            for i in 0..3 {
                flat.push((3 * r + o, 3 * c + i, S::new(cell.c[o][i]) * S::slot()));
            }
        }
    }
    for (k, i, cell) in constraint.couplings() {
        for o in 0..3 {
            let v = cell.c[0][o];
            flat.push((3 * n + k, 3 * i + o, S::new(v) * S::slot()));
            flat.push((3 * i + o, 3 * n + k, S::new(v) * S::slot()));
        }
    }
    let flat = SparseMap::from_triples(flat, (3 * n + m, 3 * n + m));
    let rhs: Field<S> = f
        .iter()
        .flat_map(|v| v.c.map(S::new))
        .chain(g.iter())
        .collect();
    let solved = flat.solve_direct(&rhs);
    for i in 0..n {
        for o in 0..3 {
            assert!((solved.site(3 * i + o).s() - x.site(i).c[o]).abs() < 1e-12);
        }
    }
    for k in 0..m {
        assert!((solved.site(3 * n + k).s() - y.site(k).s()).abs() < 1e-12);
    }
    let lu = system.factor().expect("regular");
    assert_eq!(lu.sites(), (n, m));
    assert_eq!(lu.solve(&f, &g), (x, y));
    // Without the constraint's equations the multipliers are free: singular.
    let free = BlockMap::saddle(
        stiffness,
        spread,
        SparseMap::<vga3d::Scalar<(vga3d::Vector,), f64>>::zeros((m, n)),
    );
    assert!(free.try_solve_direct(&f, &g).is_none());
}

/// A free string's whole spectrum against a mass that varies along it: every pair a pair, the
/// fields orthonormal in the mass, the lowest ones the sparse `eigh`'s.
#[test]
fn eigh_all_gives_every_pair() {
    let n = 40;
    let ends = path_ends(n);
    let laplacian = &(&ends.reverse() * &ends) * V::slot();
    let mass = SparseMap::from_diagonal(&Field::from_fn(n, |i| V::slot() * (1.0 + 0.5 * noise(i))));
    let (values, modes) = laplacian.eigh_all(&mass);
    assert_eq!(values.len(), 3 * n);
    assert!(values.windows(2).all(|w| w[0] <= w[1]));
    for (v, x) in values.iter().zip(&modes) {
        let r = &laplacian.of(x) - &mass.of(x).scaled(*v);
        assert!(r.norm() < 1e-12, "{v}: {}", r.norm());
        assert!((x.dot(&mass.of(x)) - 1.0).abs() < 1e-12);
    }
    assert!(modes[3].dot(&mass.of(&modes[7])).abs() < 1e-12);
    let (sparse, _) = laplacian.eigh(&mass, 9);
    for (a, b) in sparse.iter().zip(&values) {
        assert!((a - b).abs() < 1e-11, "{a} {b}");
    }
    // Against the Laplacian, blind to the rigid shift: the 3 (n − 1) pairs on its range, the
    // inverse values.
    let (inverse, inverse_modes) = mass.eigh_all(&laplacian);
    for (mu, x) in inverse.iter().zip(&inverse_modes) {
        let r = &mass.of(x) - &laplacian.of(x).scaled(*mu);
        assert!(r.norm() < 1e-11 * mu, "{mu}: {}", r.norm());
        assert!((x.dot(&laplacian.of(x)) - 1.0).abs() < 1e-11);
    }
    assert_eq!(inverse.len(), 3 * (n - 1));
    let finite: Vec<f64> = values[3..].iter().map(|v| 1.0 / v).collect();
    let mut finite = finite;
    finite.sort_by(f64::total_cmp);
    for (a, b) in inverse.iter().zip(&finite) {
        assert!((a - b).abs() < 1e-9 * b.abs(), "{a} {b}");
    }
}

/// The eigenpairs of a pencil of correlations are those of the numbers the stand-ins read
/// through the undual.
#[test]
fn eigh_of_correlations_is_eigh_through_the_undual() {
    let n = 10;
    let a = conductance(n, -1.0);
    let unit = Amount::new(-1.0) * pga2d::Scalar::slot();
    let m = SparseMap::from_diagonal(&Field::from_fn(n * n, |i| unit * (1.0 + 0.3 * noise(i))));
    let (values, modes) = a.eigh(&m, 6);
    let (reference, _) = a
        .map_cells(Conductance::undual)
        .eigh(&m.map_cells(Conductance::undual), 6);
    for (x, y) in values.iter().zip(&reference) {
        assert!((x - y).abs() < 1e-12 * y.abs(), "{x} {y}");
    }
    for (v, x) in values.iter().zip(&modes) {
        assert!((&a.of(x) - &m.of(x).scaled(*v)).norm() < 1e-11);
    }
    let (all, _) = a.eigh_all(&m);
    for (x, y) in values.iter().zip(&all) {
        assert!((x - y).abs() < 1e-11 * y.abs(), "{x} {y}");
    }
}

/// A free grid of vectors with a small lumped mass (sites of area `h²`): its rigid shifts, a
/// three-fold null space, come out exactly. The start vectors' Ritz values are averages over
/// the spectrum, far above zero; a shift there made a connection Laplacian's two parallel
/// fields take 18 s instead of 0.1 s, and the null vectors chosen on the inverted operator
/// came back with Rayleigh quotients of `10⁻¹²` before their refinement by inverse iteration.
#[test]
fn a_null_space_under_a_small_mass_comes_out_exactly() {
    let n = 30;
    let h2 = 1.0 / (n * n) as f64;
    let laplacian = &grid_laplacian(n, |e| 1.0 + 0.2 * noise(e)) * V::slot();
    let mass = SparseMap::from_diagonal(&Field::from_fn(n * n, |i| {
        V::slot() * (h2 * (1.0 + 0.1 * noise(i)))
    }));
    for count in [2, 3, 5] {
        let (values, modes) = laplacian.eigh(&mass, count);
        for v in &values[..count.min(3)] {
            assert!(v.abs() < 1e-10, "count {count}: {values:?}");
        }
        // A null mode is constant across the sites.
        for x in &modes[..count.min(3)] {
            let first = x.site(0);
            assert!(x.iter().all(|v| (v - first).norm() < 1e-10 * first.norm()));
        }
    }
}

/// A warm start gives the pairs a fresh search gives, from a good start (the last step of a
/// slowly changing pencil) or a poor one.
#[test]
fn eigh_from_any_start_gives_the_same_pairs() {
    let n = 16;
    let pencil = |t: f64| {
        let k = &grid_laplacian(n, |e| 1.0 + t * noise(e)) * V::slot();
        let anchors = SparseMap::from_diagonal(&Field::from_fn(n * n, |i| {
            V::slot() * if i % n == 0 { 1.0 } else { 0.0 }
        }));
        &k + &anchors
    };
    let mass = SparseMap::from_diagonal(&Field::from_fn(n * n, |_| V::slot()));
    let (_, before) = pencil(0.30).eigh(&mass, 4);
    let next = pencil(0.31);
    let (fresh, _) = next.eigh(&mass, 4);
    let (warm, modes) = next.eigh_from(&mass, 4, &before);
    let poor: Vec<Field<V>> = (0..5)
        .map(|j| Field::from_fn(n * n, |i| V::new(noise(i + j), noise(3 * i + j), 1.0)))
        .collect();
    let (cold, _) = next.eigh_from(&mass, 4, &poor);
    for ((a, b), c) in fresh.iter().zip(&warm).zip(&cold) {
        assert!(
            (a - b).abs() < 1e-12 * a && (a - c).abs() < 1e-12 * a,
            "{a} {b} {c}"
        );
    }
    for (v, x) in warm.iter().zip(&modes) {
        assert!((&next.of(x) - &mass.of(x).scaled(*v)).norm() < 1e-10);
        assert!((x.dot(&mass.of(x)) - 1.0).abs() < 1e-12);
    }
}
