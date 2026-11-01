//! STA2D, the spacetime of the plane (ADR-055, amended 2026-11-01): the subalgebra of STA that
//! `e1, e2, e4` generate. Its boosts, fields and Hodge duals are STA's through the homomorphism,
//! and numga's reading of it (points as antivectors) works as the ports had it.

#![cfg(all(feature = "sta", feature = "sta2d"))]

use gax::{sta, sta2d};

fn close(a: &[f64], b: &[f64]) -> bool {
    a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-15)
}

#[test]
fn boosts_are_stas() {
    let d = sta2d::Vector::new(0.6, 0.8, 0.0);
    let ours = sta::Even::from(sta2d::Even::boost(d, 0.4).into_inner());
    let theirs = sta::Even::boost(sta::Vector::new(0.6, 0.8, 0.0, 0.0), 0.4);
    assert!(close(&ours.c, &theirs.c));
    // A boost moves an event as STA's does, through the homomorphism (φ(m >> x) = φ(m) >> φ(x)).
    let b = sta2d::Even::boost(d, 0.4);
    let x = sta2d::Vector::new(0.3, -0.2, 1.5);
    let moved = sta::Vector::from(b >> x);
    let want = sta::Even::boost(sta::Vector::new(0.6, 0.8, 0.0, 0.0), 0.4) >> sta::Vector::from(x);
    assert!(close(&moved.c, &want.c));
}

#[test]
fn fields_and_duals_are_stas() {
    // A field's blades keep their names: `e12` magnetic, `e14, e24` electric.
    let f = sta2d::Bivector::new(0.5, -1.0, 2.0);
    let g = sta::Bivector::from(f);
    assert_eq!(g.c, [0.0, 0.0, 0.5, -1.0, 2.0, 0.0]);
    // The field's invariant `F²` (scalar part) is the same.
    assert_eq!((f * f).s(), (g * g).s());
    // numga's metric (its `dot`) on the plane's fields: magnetic positive, electric negative.
    assert_eq!(f.metric(f).s(), 0.25 - 1.0 - 4.0);
}

#[test]
fn numgas_points_are_antivectors() {
    // The point at (x, y), numga's `x X + y Y` with `X, Y` the duals of `x, y`: a bivector
    // through time, which the planes `x` and `y` read back.
    let p: sta2d::intrinsic::Bivector = sta2d::Vector::new(2.0, 3.0, 0.0).hodge();
    let (x, y) = (
        sta2d::Vector::new(1.0, 0.0, 0.0),
        sta2d::Vector::new(0.0, 1.0, 0.0),
    );
    assert_eq!([(x & p).s(), (y & p).s()], [2.0, 3.0]);
    // As a part: the bivectors through time.
    let t: sta2d::intrinsic::Temporal = p.cast::<sta2d::Temporal>();
    assert_eq!(sta2d::intrinsic::Bivector::from(t).c, p.c);
    // The observer's direction in time, minus the dual of `t`, pairs to one with `t`.
    let time: sta2d::intrinsic::Bivector = -sta2d::Vector::new(0.0, 0.0, 1.0).hodge();
    assert_eq!((sta2d::Vector::new(0.0, 0.0, 1.0) & time).s(), 1.0);
    // numga's `dual_inverse` undoes it.
    assert_eq!(p.hodge_inverse().c, [2.0, 3.0, 0.0]);
}
