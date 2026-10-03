//! `Motions`: one generic check, run for every algebra that implements it.

#![cfg(all(
    feature = "pga2d",
    feature = "pga3d",
    feature = "vga2d",
    feature = "vga3d"
))]

#[path = "support/rng.rs"]
mod rng;
use rng::{Draw, rng};

use gax::motions::{Linear, Motions, Pga2d, Pga3d, Vga2d, Vga3d};
use gax::{Extensor, Of};

fn close<X: Extensor<Slots = (), Coef = f64>>(a: X, b: X, tol: f64) -> bool {
    a.coeffs()
        .as_ref()
        .iter()
        .zip(b.coeffs().as_ref())
        .all(|(x, y)| (x - y).abs() < tol)
}

fn laws<G: Motions<f64>>(seed: u64, euclidean: bool) {
    let mut rng = rng(seed);
    for _ in 0..50 {
        // Points: coordinates round trip.
        let c = G::coords_from_fn(|_| 3.0 * rng.next_f64());
        let back = G::coords(G::point(c));
        assert!(
            c.as_ref()
                .iter()
                .zip(back.as_ref())
                .all(|(a, b)| (a - b).abs() < 1e-12)
        );

        // exp and log, for a twist small enough to be the principal one.
        let b: G::Twist = rng.value(0.4);
        let name = std::any::type_name::<G>();
        assert!(
            close(G::log(G::exp(b)), b, 1e-12),
            "{name}: log(exp({b:?})) = {:?}",
            G::log(G::exp(b))
        );
        let m = G::exp(b);
        assert!(close(G::log(G::reverse(m) * m), Linear::zero(), 1e-12));

        // A motion moves forques and twists alike: their pairing is invariant.
        let (f, t): (G::Forque, G::Twist) = (rng.value(1.0), rng.value(1.0));
        assert!((G::pair(m >> f, m >> t) - G::pair(f, t)).abs() < 1e-12);
        // ... and `<<` undoes `>>`.
        assert!(close(m << (m >> t), t, 1e-12));

        // The commutator is antisymmetric, and `ad` is it as a map.
        let a: G::Twist = rng.value(1.0);
        assert!(close(G::commutator(a, t), -G::commutator(t, a), 1e-12));
        assert!(close(G::ad(a).of(t), G::commutator(a, t), 1e-12));
        assert!(close(G::identity_map().of(t), t, 1e-15));

        // An inertia of enough points is invertible, and its kinetic energy positive.
        let mut inertia = G::Inertia::zero();
        for k in 0..=G::DIM {
            let c =
                G::coords_from_fn(|i| if i + 1 == k { 2.0 } else { 0.0 } + 0.3 * rng.next_f64());
            inertia += G::point_inertia(G::point(c));
        }
        let mobility = G::mobility(inertia);
        assert!(close(mobility.of(inertia.of(t)), t, 1e-9));
        assert!(!euclidean || G::pair(inertia.of(t), t) > 0.0);
    }
    assert!(close(G::log(G::identity()), Linear::zero(), 1e-15));
}

#[test]
fn every_algebra() {
    laws::<Pga2d>(0x0011, true);
    laws::<Pga3d>(0x0022, true);
    laws::<Vga2d>(0x0033, true);
    laws::<Vga3d>(0x0044, true);
}

/// `pair(I(b), b)` over `⟨v, v⟩` for unit masses at random points, `v` the point's velocity
/// under `exp(t b)` (central differences of its coordinates) and `sig` the metric's diagonal
/// on the coordinates: twice the kinetic energy, in one convention for every algebra.
fn energy_ratios<G: Motions<f64>>(seed: u64, sig: &[f64]) -> Vec<f64> {
    let mut rng = rng(seed);
    (0..20)
        .map(|_| {
            let p = G::point(G::coords_from_fn(|_| 2.0 * rng.next_f64()));
            let b: G::Twist = rng.value(0.5);
            let h = 1e-5;
            let at = |t: f64| G::coords(G::exp(b * t) >> p);
            let (ahead, behind) = (at(h), at(-h));
            let vv: f64 = ahead
                .as_ref()
                .iter()
                .zip(behind.as_ref())
                .zip(sig)
                .map(|((x, y), s)| s * ((x - y) / (2.0 * h)).powi(2))
                .sum();
            G::pair(G::point_inertia(p).of(b), b) / vv
        })
        .collect()
}

#[cfg(all(
    feature = "sta",
    feature = "stap",
    feature = "cga2d",
    feature = "cga3d",
    feature = "csta"
))]
#[test]
fn the_spacetime_and_conformal_algebras() {
    use gax::motions::{Cga2d, Cga3d, Csta, Sta, Stap};
    laws::<Cga2d>(0x0055, true);
    laws::<Cga3d>(0x0066, true);
    laws::<Sta>(0x0077, false);
    laws::<Stap>(0x0088, false);
    laws::<Csta>(0x0099, false);
    // Every inertia pairs a twist to the same multiple of `⟨v, v⟩` as PGA3D's.
    let reference = energy_ratios::<Pga3d>(1, &[1.0; 3])[0];
    let space = [1.0; 3];
    let time_first = [1.0, -1.0, -1.0, -1.0];
    let time_last = [1.0, 1.0, 1.0, -1.0];
    for (name, ratios) in [
        ("pga2d", energy_ratios::<Pga2d>(2, &space)),
        ("pga3d", energy_ratios::<Pga3d>(3, &space)),
        ("vga2d", energy_ratios::<Vga2d>(4, &space)),
        ("vga3d", energy_ratios::<Vga3d>(5, &space)),
        ("cga2d", energy_ratios::<Cga2d>(6, &space)),
        ("cga3d", energy_ratios::<Cga3d>(7, &space)),
        ("sta", energy_ratios::<Sta>(8, &time_first)),
        ("stap", energy_ratios::<Stap>(9, &time_last)),
        ("csta", energy_ratios::<Csta>(10, &time_last)),
    ] {
        for r in ratios {
            assert!(
                (r - reference).abs() < 1e-5 * reference.abs(),
                "{name}: {r} vs {reference}"
            );
        }
    }
}
