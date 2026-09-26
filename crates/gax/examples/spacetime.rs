//! Special relativity in two algebras beyond the defaults.
//!
//! STAP, the projective spacetime algebra `R(3,1,1)`: events are points (quadvectors) and a
//! Poincaré motion (boosts, rotations, translations in space and time) is a motor, the
//! exponential of a bivector. Boosts keep the interval `t² - x²`, and rapidities add.
//!
//! CSTA, the conformal spacetime algebra `R(4,2)`: an event `x` is the null vector
//! `x + ½x² ei + eo`, and the inner product of two events is minus half their interval, so
//! light-like separation is orthogonality.
//!
//! Run with `cargo run --example spacetime --features stap,csta`.

fn stap() {
    use gax::stap::{Bivector, Event};
    // Events: x e0324 + y e0134 + z e0214 + t e0123 + w e1234 (w = 1 for a finite event).
    let event = |t: f64, x: f64| Event::<(), f64>::new(x, 0.0, 0.0, t, 1.0);
    let coords = |e: Event<(), f64>| (e.e0123() / e.e1234(), e.e0324() / e.e1234());

    // A boost along x: the exponential of the generator e41 (time with space) scaled by half
    // the rapidity. Bivector order: e01, e02, e03, e04, e23, e31, e12, e41, e42, e43.
    let boost = |rapidity: f64| {
        Bivector::<(), f64>::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5 * rapidity, 0.0, 0.0).exp()
    };

    // The clock at rest at the origin, one unit of its time later, seen from a frame moving
    // at rapidity 0.8: time dilates by cosh 0.8 and the event moves by sinh 0.8.
    let (t, x) = coords(boost(0.8) >> event(1.0, 0.0));
    println!(
        "STAP: boosted event (t, x) = ({t:.4}, {x:.4}), interval {:.4}",
        t * t - x * x
    );
    assert!((t - 0.8f64.cosh()).abs() < 1e-12 && (x.abs() - 0.8f64.sinh()).abs() < 1e-12);
    assert!((t * t - x * x - 1.0).abs() < 1e-12);

    // Rapidities add: velocities combine as (u + v) / (1 + u v).
    let (u, v) = (0.6f64, 0.7f64);
    let composed = boost(u.atanh()) * boost(v.atanh());
    let (t, x) = coords(composed >> event(1.0, 0.0));
    let w = x.abs() / t;
    println!(
        "STAP: 0.6c + 0.7c = {w:.6}c (formula {:.6}c)",
        (u + v) / (1.0 + u * v)
    );
    assert!((w - (u + v) / (1.0 + u * v)).abs() < 1e-12);

    // A translation in time (generator e04) shifts events by a constant.
    let later = Bivector::<(), f64>::new(0.0, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0).exp();
    let (t, x) = coords(later >> event(0.0, 2.0));
    println!("STAP: time translation of (0, 2): ({t:.4}, {x:.4})");
    assert!((t.abs() - 3.0).abs() < 1e-12 && (x - 2.0).abs() < 1e-12);

    // A flash of light (t = x) stays on the light cone under any boost.
    let (t, x) = coords(boost(-1.3) >> event(2.0, 2.0));
    println!("STAP: boosted light-like event ({t:.4}, {x:.4})");
    assert!((t.abs() - x.abs()).abs() < 1e-12);
}

fn csta() {
    use gax::csta::Vector;
    // Minkowski square with e1, e2, e3 spacelike (+1) and e4 timelike (-1).
    let up = |t: f64, x: f64, y: f64| {
        let sq = x * x + y * y - t * t;
        Vector::<(), f64>::new(x, y, 0.0, t, 1.0, 0.5 * sq)
    };
    let interval = |a: Vector<(), f64>, b: Vector<(), f64>| -2.0 * a.dot(b).s();
    let origin = up(0.0, 0.0, 0.0);
    for (name, e) in [
        ("light-like", up(3.0, 3.0, 0.0)),
        ("time-like", up(3.0, 1.0, 0.0)),
        ("space-like", up(1.0, 3.0, 0.0)),
    ] {
        let s = interval(origin, e);
        println!("CSTA: {name} event: -2 (o . x) = {s:+.3}");
        assert!((s - (e.e1() * e.e1() + e.e2() * e.e2() - e.e4() * e.e4())).abs() < 1e-12);
    }
    // Events are null vectors: x . x = 0.
    let e = up(1.5, -0.5, 2.0);
    assert!(e.dot(e).s().abs() < 1e-12);
}

fn main() {
    stap();
    csta();
}
