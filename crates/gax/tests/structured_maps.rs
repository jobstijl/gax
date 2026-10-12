//! Structured maps as maps (ADR-044, a spike): `PrincipalInertia`, a `Line <- Line` stored as
//! four numbers, composes, binds into other maps and forms, and solves without `to_map()`, with
//! the results of its dense form.

#![cfg(feature = "pga3d")]

#[path = "support/rng.rs"]
mod rng;
use rng::{Draw, rng};

use gax::ApproxEq;
use gax::map::{LinearMap, SquareLinearMap};
use gax::pga3d::{Line, Point, PrincipalInertia, PrincipalMobility, Scalar};

fn random_body(rng: &mut impl Draw) -> PrincipalInertia<f64> {
    let mut positive = || 0.5 + rng.next_f64().abs();
    PrincipalInertia::new(positive(), [positive(), positive(), positive()])
}

fn random_line(rng: &mut impl Draw) -> Line<(), f64> {
    Line::from_coeffs(core::array::from_fn(|_| rng.next_f64()))
}

fn random_map(rng: &mut impl Draw) -> Line<(Line,), f64> {
    Line::from_coeffs(core::array::from_fn(|_| {
        core::array::from_fn(|_| rng.next_f64())
    }))
}

/// A dense map composed with the structured one, either way round, and the structured one bound
/// into a form: the same coefficients as with its dense form. Binding through the dense form
/// rounds as the dense form does; the structured map's own composition scales whole columns,
/// where the dense contraction adds exact zeros, so both are exact.
#[test]
fn composes_and_binds() {
    let mut rng = rng(0x5a1e);
    for _ in 0..100 {
        let body = random_body(&mut rng);
        let dense = body.to_map();
        let m = random_map(&mut rng);
        assert_eq!(m.of(body), m.of(dense)); // m ∘ body
        assert_eq!(body.of(m), dense.of(m)); // body ∘ m
        // Into a form's first slot, and the kinetic energy form built from it.
        let pairing: Scalar<(Line, Line), f64> = Line::slot() & Line::slot();
        assert_eq!(pairing.of(body), pairing.of(dense));
        assert_eq!(Line::slot() & body.of(Line::slot()), body.energy_form());
        // A point map composed into the twist slot of a map on points.
        let p = Point::xyz(rng.next_f64(), rng.next_f64(), rng.next_f64());
        let velocity: Point<(Line,), f64> = Line::slot().commutator(p);
        assert_eq!(velocity.of(body), velocity.of(dense));
    }
}

/// Solving and inverting by the four numbers, against LU on the dense form.
#[test]
fn solves_and_inverts() {
    let mut rng = rng(0x501e);
    for _ in 0..100 {
        let body = random_body(&mut rng);
        let dense = body.to_map();
        let f = random_line(&mut rng);
        assert!(SquareLinearMap::solve(body, f).approx_eq(&dense.solve(f), 1e-13));
        let mobility: PrincipalMobility<f64> = SquareLinearMap::inverse(body);
        assert!(mobility.to_dense().approx_eq(&dense.inverse(), 1e-13));
        assert!(
            SquareLinearMap::inverse(mobility)
                .to_map()
                .approx_eq(&dense, 1e-13)
        );
    }
}

/// One function over any representation of an inertia: the kinetic energy of the twist that a
/// momentum gives the body, `½ f & I⁻¹(f)`.
fn energy_of_momentum<I>(inertia: I, momentum: Line<(), f64>) -> f64
where
    I: SquareLinearMap<Kind = Line, Input = Line, Coef = f64>,
{
    0.5 * (momentum & inertia.solve(momentum)).s()
}

#[test]
fn generic_code_takes_either() {
    let mut rng = rng(0x6e1e);
    for _ in 0..100 {
        let body = random_body(&mut rng);
        let f = random_line(&mut rng);
        let (compact, dense) = (
            energy_of_momentum(body, f),
            energy_of_momentum(body.to_map(), f),
        );
        assert!((compact - dense).abs() < 1e-12 * compact.abs().max(1.0));
    }
}

/// A dense map's `apply` is its `of`, bit for bit.
#[test]
fn dense_apply_is_of() {
    let mut rng = rng(0xa991);
    for _ in 0..100 {
        let m = random_map(&mut rng);
        let b = random_line(&mut rng);
        assert_eq!(SquareLinearMap::apply(m, b), m.of(b));
    }
}
