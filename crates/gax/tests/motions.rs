//! `Motions`: one generic check, run for every algebra that implements it.

#![cfg(all(
    feature = "pga2d",
    feature = "pga3d",
    feature = "vga2d",
    feature = "vga3d"
))]

use gax::motions::{Linear, Motions, Pga2d, Pga3d, Vga2d, Vga3d};
use gax::{Extensor, Kind, Of};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }
}

fn value<X: Extensor<Slots = (), Coef = f64>>(rng: &mut Rng, scale: f64) -> X {
    X::from_coeffs(<X::Kind as Kind>::arr_from_fn(|_| scale * rng.next()))
}

fn close<X: Extensor<Slots = (), Coef = f64>>(a: X, b: X, tol: f64) -> bool {
    a.coeffs()
        .as_ref()
        .iter()
        .zip(b.coeffs().as_ref())
        .all(|(x, y)| (x - y).abs() < tol)
}

fn laws<G: Motions<f64>>(seed: u64) {
    let mut rng = Rng(seed);
    for _ in 0..50 {
        // Points: coordinates round trip.
        let c = G::coords_from_fn(|_| 3.0 * rng.next());
        let back = G::coords(G::point(c));
        assert!(
            c.as_ref()
                .iter()
                .zip(back.as_ref())
                .all(|(a, b)| (a - b).abs() < 1e-12)
        );

        // exp and log, for a twist small enough to be the principal one.
        let b: G::Twist = value(&mut rng, 0.4);
        assert!(close(G::log(G::exp(b)), b, 1e-12));
        let m = G::exp(b);
        assert!(close(
            G::log(G::reverse(m) * m),
            value(&mut Rng(1), 0.0),
            1e-12
        ));

        // A motion moves forques and twists alike: their pairing is invariant.
        let (f, t): (G::Forque, G::Twist) = (value(&mut rng, 1.0), value(&mut rng, 1.0));
        assert!((G::pair(m >> f, m >> t) - G::pair(f, t)).abs() < 1e-12);
        // ... and `<<` undoes `>>`.
        assert!(close(m << (m >> t), t, 1e-12));

        // The commutator is antisymmetric, and `ad` is it as a map.
        let a: G::Twist = value(&mut rng, 1.0);
        assert!(close(G::commutator(a, t), -G::commutator(t, a), 1e-12));
        assert!(close(G::ad(a).of(t), G::commutator(a, t), 1e-12));
        assert!(close(G::identity_map().of(t), t, 1e-15));

        // An inertia of enough points is invertible, and its kinetic energy positive.
        let mut inertia = G::Inertia::zero();
        for k in 0..=G::DIM {
            let c = G::coords_from_fn(|i| if i + 1 == k { 2.0 } else { 0.0 } + 0.3 * rng.next());
            inertia += G::point_inertia(G::point(c));
        }
        let mobility = G::mobility(inertia);
        assert!(close(mobility.of(inertia.of(t)), t, 1e-9));
        assert!(G::pair(inertia.of(t), t) > 0.0);
    }
    assert!(close(G::log(G::identity()), value(&mut Rng(1), 0.0), 1e-15));
}

#[test]
fn every_algebra() {
    laws::<Pga2d>(0x0011);
    laws::<Pga3d>(0x0022);
    laws::<Vga2d>(0x0033);
    laws::<Vga3d>(0x0044);
}
