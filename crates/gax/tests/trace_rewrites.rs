//! Experiment: do the rewrites the laws license (docs/laws.md §4) change what the tracer
//! produces? Each pair of kernels computes the same thing, written two ways; the tracer's cost at
//! every expansion limit is printed, together with the cost it keeps (the cheapest).
//!
//! Run with `cargo test -p gax --features trace --test trace_rewrites -- --ignored --nocapture`.
//! The results are recorded in docs/performance.md ("Law-based rewrites in the tracer").

#![cfg(all(feature = "trace", feature = "pga3d"))]

use gax::Unit;
use gax::pga3d::{Line, Motor, Point};
use gax::trace::{Sym, Tracer};

fn report(t: &Tracer) {
    for r in t.reports() {
        let limits: Vec<String> = r
            .limits
            .iter()
            .map(|(l, c)| {
                let l = l.map_or_else(|| "∞".to_string(), |l| l.to_string());
                let c = c.map_or_else(|| "—".to_string(), |c| format!("{}m {}a", c.muls, c.adds));
                format!("{l}: {c}")
            })
            .collect();
        println!(
            "{:28} kept {:>4} mul {:>4} add | generic {:>4} mul {:>4} add | {}",
            r.name,
            r.cost.muls,
            r.cost.adds,
            r.naive.muls,
            r.naive.adds,
            limits.join(", ")
        );
    }
}

#[test]
#[ignore = "an experiment: prints costs"]
fn rewrites() {
    type M = Unit<Motor<(), Sym>>;
    type P = Point<(), Sym>;
    let mut t = Tracer::new();

    // Folding a versor chain: a >> (b >> (c >> x)) against (a * b * c) >> x.
    t.kernel("chain_nested", |a: M, b: M, c: M, p: P| {
        a >> (b >> (c >> p))
    });
    t.kernel("chain_folded", |a: M, b: M, c: M, p: P| (a * b * c) >> p);
    // The same with plain motors (no unit condition to exploit).
    t.kernel(
        "chain_nested_plain",
        |a: Motor<(), Sym>, b: Motor<(), Sym>, p: P| a >> (b >> p),
    );
    t.kernel(
        "chain_folded_plain",
        |a: Motor<(), Sym>, b: Motor<(), Sym>, p: P| (a * b) >> p,
    );
    // Folding with one versor fixed at run time and the matrix path: build the chain's matrix
    // once, then apply it.
    t.kernel("chain_matrix", |a: M, b: M, p: P| {
        ((a * b) >> Point::slot()).of(p)
    });
    // Reassociating composition of maps: f(g(h(x))) against (f ∘ g ∘ h)(x), with the maps
    // built from motors (kernel arguments are values).
    let map = |m: M| m >> Point::slot();
    t.kernel("compose_applied", move |a: M, b: M, c: M, p: P| {
        map(a).of(map(b).of(map(c).of(p)))
    });
    t.kernel("compose_first", move |a: M, b: M, c: M, p: P| {
        map(a).of(map(b)).of(map(c)).of(p)
    });
    // Pulling a scalar through a product (linearity): (m * b) * s against m * (b * s).
    t.kernel(
        "scalar_outside",
        |m: Motor<(), Sym>, b: Line<(), Sym>, s: Sym| (m * b).gp(s),
    );
    t.kernel(
        "scalar_inside",
        |m: Motor<(), Sym>, b: Line<(), Sym>, s: Sym| m * b.gp(s),
    );
    report(&t);
}
