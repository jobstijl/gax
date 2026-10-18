//! numga's `mechanics/hanging_chain` (a notebook): a chain of beads on stiff springs, hanging
//! between two pins, found at rest by Newton's method on the whole chain at once (VGA2D).
//!
//! The beads' positions are one field, `Field<Vector>`, and the energy one scalar function of
//! it: each spring's stretch energy, `k / (2 l) (|link| - l)²` for stiffness `k` and rest length
//! `l`, less the work of the beads' weight. The links are the field the incidence maps make of
//! the beads and the pins: link `k` is bead `k` less bead `k - 1`, the first from the left pin
//! and the last to the right one, a `SparseMap` of scalar cells `-1` and `1`.
//!
//! `gax::field::hessian` differentiates that energy with respect to the whole field: the
//! gradient, a field of forms `Scalar <- Vector`, and the curvature, a field map of forms
//! `Scalar <- (Vector, Vector)` with blocks only between a bead, itself and its neighbours (the
//! zero blocks are not stored, where numga keeps all 62 × 62 coefficients dense). Read through
//! the metric `a | b`, the forms become vectors and maps, and one sparse direct solve of the
//! curvature against the gradient moves every bead at once: the Newton step. Minus the gradient
//! read as vectors is the force each bead feels unbalanced, which at rest vanishes.
//!
//! Newton's method needs every spring stretched (a compressed spring's curvature is negative
//! across it), so the chain starts on a circular arc through the pins, each bead the arc's
//! lowest point turned about its centre by a rotor. Once the largest unbalanced force is below
//! a bead's weight, every step squares it. At rest the chain hangs just below the catenary of
//! its rest length, by the stretch its weight gives the springs.
//!
//! The animation draws the iterates one after another over the dashed catenary, and the
//! largest unbalanced force at each, in bead weights, on a logarithmic scale.

use gax_numga_examples::{
    Align, Anim, Axes, Canvas, Marker, Point2, Rect, backdrop, caption, palette, reach2, run,
};
use std::sync::OnceLock;

mod hanging_chain {
    use gax::Real;
    use gax::field::{Field, SparseMap};
    use gax::vga2d::{Pseudoscalar, Scalar, Vector};
    use gax_numga_examples::measure;
    use gax_numga_examples::signal::phasor;

    /// A position in the plane of the chain, from the point midway between the pins.
    pub type V = Vector<(), f64>;
    /// The beads' positions.
    pub type Chain = Field<V>;

    pub const BEADS: usize = 31;
    pub const LINKS: usize = BEADS + 1;
    /// The distance between the pins, and the chain's length with its springs at rest.
    pub const SPAN: f64 = 2.0;
    pub const LENGTH: f64 = 2.5;
    /// Each spring's force per unit of strain (its stretch over its rest length).
    pub const STIFFNESS: f64 = 1e3;
    pub const BEAD_MASS: f64 = 0.1;
    pub const GRAVITY: f64 = 9.81;
    pub const STEPS: usize = 8;
    /// The start: the beads evenly along a circular arc through the pins, each pin this angle
    /// from the arc's lowest point.
    pub const START_TURN: f64 = 1.3;

    pub const X: V = V::new(1.0, 0.0);
    pub const Y: V = V::new(0.0, 1.0);

    /// The pins.
    pub fn pins() -> [V; 2] {
        [X * (-SPAN / 2.0), X * (SPAN / 2.0)]
    }

    /// A bead's weight.
    pub fn weight() -> V {
        Y * (-BEAD_MASS * GRAVITY)
    }

    /// A spring's rest length.
    pub fn rest() -> f64 {
        LENGTH / LINKS as f64
    }

    /// The incidence of the links on the beads: link `k` has bead `k` as its head (`+1`) and
    /// bead `k - 1` as its tail (`-1`).
    pub fn bead_ends() -> SparseMap<Scalar<(), f64>> {
        let heads = (0..BEADS).map(|k| (k, k, 1.0));
        let tails = (0..BEADS).map(|k| (k + 1, k, -1.0));
        let (rows, (cols, cells)): (Vec<usize>, (Vec<usize>, Vec<_>)) = heads
            .chain(tails)
            .map(|(link, bead, sign)| (link, (bead, Scalar::new(sign))))
            .unzip();
        SparseMap::new(cells, rows, cols, (LINKS, BEADS))
    }

    /// The pins' share of the links: the left pin is the first link's tail, the right pin the
    /// last link's head.
    pub fn anchors() -> Field<V> {
        let [left, right] = pins();
        let pin_ends = SparseMap::new(
            vec![Scalar::<(), f64>::new(-1.0), Scalar::new(1.0)],
            vec![0, LINKS - 1],
            vec![0, 1],
            (LINKS, 2),
        );
        &pin_ends * &Field::new(vec![left, right])
    }

    /// The springs' stretch energy less the work of the beads' weight, generic over the
    /// coefficients so that `gax::field` can differentiate it.
    pub fn energy<T: Real>(beads: &Field<Vector<(), T>>) -> T {
        let c = T::from_f64;
        let lift = |v: V| v.map_coefs(c);
        let ends = bead_ends().map_cells(|e| e.map_coefs(c));
        let links = &(&ends * beads) + &anchors().map(lift);
        let stretch = links.map(|l| Scalar::new(l.norm() - c(rest())));
        let springs = stretch.map(|s| s * s).sum().s() * c(STIFFNESS / (2.0 * rest()));
        let work = beads.map(|b| lift(weight()).scalar_product(b)).sum().s();
        springs - work
    }

    /// The metric of the plane, `a | b`: it reads a form as the vector that pairs as it does,
    /// and a bilinear form as the map.
    fn metric() -> Scalar<(Vector, Vector), f64> {
        Vector::slot() | Vector::slot()
    }

    /// The start: the beads evenly along the arc through the pins whose ends are
    /// [`START_TURN`] from its lowest point, by rotors about the arc's centre.
    pub fn start() -> Chain {
        let pin = phasor(START_TURN);
        let radius = SPAN / 2.0 / pin.e01();
        let centre = Y * (radius * pin.e20());
        let lowest = Y * -radius;
        Field::from_fn(BEADS, |i| {
            let t = (i + 1) as f64 / LINKS as f64;
            let angle = START_TURN * (2.0 * t - 1.0);
            let turn = Pseudoscalar::<(), f64>::new(-angle / 2.0).exp();
            centre + (turn >> lowest)
        })
    }

    /// The force each bead feels unbalanced: minus the energy's gradient, read as vectors.
    pub fn unbalanced(beads: &Chain) -> Chain {
        let (_, gradient) = gax::field::gradient(energy, beads);
        gradient.map(|g| -metric().solve_pairing(g))
    }

    /// One Newton step of the whole chain: the curvature solved against the gradient, both
    /// read through the metric, in one sparse direct solve.
    pub fn newton(beads: &Chain) -> Chain {
        let (_, gradient, curvature) = gax::field::hessian(energy, beads);
        let curvature = curvature.map_cells(|h| metric().solve_pairing(h));
        let step = curvature.solve_direct(&gradient.map(|g| metric().solve_pairing(g)));
        beads - &step
    }

    /// The start and the chain after each Newton step.
    pub fn iterates() -> Vec<Chain> {
        let mut chains = vec![start()];
        for _ in 0..STEPS {
            let next = newton(chains.last().expect("the start"));
            chains.push(next);
        }
        chains
    }

    /// The largest unbalanced force on a bead, in bead weights.
    pub fn largest_residual(beads: &Chain) -> f64 {
        let weight = BEAD_MASS * GRAVITY;
        unbalanced(beads)
            .iter()
            .map(|f| f.norm() / weight)
            .fold(0.0, f64::max)
    }

    /// The catenary's radius of curvature at its bottom, `a` with `2 a sinh(span / 2a)` the
    /// chain's length, by bisection (the length falls as `a` grows) over numga's bracket, halved
    /// to rounding.
    pub fn catenary_scale() -> f64 {
        let length = |a: f64| 2.0 * a * (SPAN / (2.0 * a)).sinh();
        let steps = f64::MANTISSA_DIGITS as usize + 10;
        measure::bisect(SPAN / 20.0, 100.0 * SPAN, steps, |a| length(a) > LENGTH)
    }

    /// The catenary of the rest length at `across`: `a cosh(x / a)`, lowered to put the pins
    /// at height zero.
    pub fn catenary_height(scale: f64, across: f64) -> f64 {
        scale * ((across / scale).cosh() - (SPAN / (2.0 * scale)).cosh())
    }
}

use hanging_chain::*;

/// The catenary's samples.
const CURVE_SAMPLES: usize = 200;
/// Seconds per Newton step shown, and the loop: every step and a pause on the chain at rest.
const STEP_SECONDS: f32 = 1.0;
/// The residual chart's floor, in bead weights.
const FLOOR: f32 = 1e-14;
const SECONDS: f32 = STEP_SECONDS * (STEPS as f32 + 3.0);

/// The chain's iterates, their largest residuals, and the catenary.
struct Solve {
    chains: Vec<Chain>,
    residuals: Vec<f64>,
    catenary: Vec<V>,
}

fn solve() -> &'static Solve {
    static SOLVE: OnceLock<Solve> = OnceLock::new();
    SOLVE.get_or_init(|| {
        let chains = iterates();
        let residuals = chains.iter().map(largest_residual).collect();
        let scale = catenary_scale();
        let catenary = (0..=CURVE_SAMPLES)
            .map(|i| {
                let across = SPAN * (i as f64 / CURVE_SAMPLES as f64 - 0.5);
                X * across + Y * catenary_height(scale, across)
            })
            .collect();
        Solve {
            chains,
            residuals,
            catenary,
        }
    })
}

/// A chain as the polyline from pin to pin through its beads.
fn polyline(beads: &Chain) -> Vec<Point2> {
    let [left, right] = pins();
    let through = beads.iter().map(reach2);
    [reach2(left)]
        .into_iter()
        .chain(through)
        .chain([reach2(right)])
        .collect()
}

/// The chain's shape: the iterates so far, earlier ones fainter, the current one with its
/// beads, over the catenary.
fn draw_shapes(c: &mut Canvas, rect: Rect, shown: usize) {
    let s = solve();
    let ax = Axes::equal(rect, Point2::xy(0.0, -0.4), 0.95);
    let catenary: Vec<Point2> = s.catenary.iter().map(|v| reach2(*v)).collect();
    ax.dashed(c, &catenary, 2.2, 7.0, palette::orange().faded(0.8));
    for (k, chain) in s.chains[..shown].iter().enumerate() {
        let age = (shown - 1 - k) as f32;
        ax.polyline(
            c,
            &polyline(chain),
            1.2,
            palette::sky().faded(0.55 / (1.0 + age)),
        );
    }
    let now = polyline(&s.chains[shown - 1]);
    ax.polyline(c, &now, 1.8, palette::sky());
    ax.scatter(c, &now[1..now.len() - 1], Marker::Dot, 5.0, palette::ink());
    ax.scatter(
        c,
        &[now[0], now[now.len() - 1]],
        Marker::Square,
        10.0,
        palette::yellow(),
    );
    let label = format!("NEWTON STEP {}", shown - 1);
    ax.text(
        c,
        ax.at(0.03, 0.06),
        &label,
        13.0,
        palette::ink(),
        Align::Left,
    );
    ax.legend(
        c,
        &[("CHAIN", palette::sky()), ("CATENARY", palette::orange())],
    );
}

/// The largest unbalanced force at each iterate shown, in bead weights, over the level of one
/// weight, below which each step squares it.
fn draw_residuals(c: &mut Canvas, rect: Rect, shown: usize) {
    let steps = [-0.4, STEPS as f32 + 0.4];
    let ax = Axes::new(rect, steps, [FLOOR, 1e2]).log_y();
    ax.frame(c, "FORCE / BEAD WEIGHT", "NEWTON STEP", "");
    // Rounding leaves residuals below the axes' floor; they are drawn on it.
    let points: Vec<Point2> = solve().residuals[..shown]
        .iter()
        .enumerate()
        .map(|(k, r)| Point2::xy(k as f32, (*r as f32).max(FLOOR)))
        .collect();
    let one = steps.map(|k| Point2::xy(k, 1.0));
    ax.line(c, one[0], one[1], 1.0, palette::grid());
    ax.polyline(c, &points, 1.8, palette::sky());
    ax.scatter(c, &points, Marker::Dot, 7.0, palette::sky());
}

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let screen = c.rect();
    let (w, h) = (screen.width(), screen.height());
    let shown = ((t.rem_euclid(SECONDS) / STEP_SECONDS) as usize + 1).min(STEPS + 1);
    draw_shapes(
        c,
        Rect::new(0.0, 0.1 * h, 0.58 * w, h).inset(16.0, 10.0, 8.0, 16.0),
        shown,
    );
    let chart = Rect::new(0.58 * w, 0.0, w, h).inset(0.09 * w, 0.24 * h, 0.05 * w, 0.13 * h);
    draw_residuals(c, chart, shown);
    caption(
        c,
        "A HANGING CHAIN BY NEWTON'S METHOD",
        "THE ENERGY OF THE FIELD OF BEADS, ITS GRADIENT AND CURVATURE, ONE SPARSE SOLVE (VGA2D)",
    );
}

fn main() {
    run(Anim::new("hanging_chain", SECONDS).size(960, 540), draw);
}

#[cfg(test)]
mod tests {
    use super::hanging_chain::*;
    use super::solve;

    /// The notebook's checks: at rest every bead's unbalanced force vanishes to round-off
    /// against its weight; once below a weight, each Newton step squares it, in units of the
    /// weight; and the beads lie on the catenary of the rest length to within the stretch the
    /// chain's weight gives springs this stiff.
    #[test]
    fn newton_converges_quadratically_onto_the_catenary() {
        let s = solve();
        let largest = &s.residuals;
        assert!(largest[STEPS] < 1e-9, "{largest:?}");
        for k in 0..STEPS {
            if largest[k] < 1.0 {
                assert!(
                    largest[k + 1] < largest[k] * largest[k] + 1e-9,
                    "{largest:?}"
                );
            }
        }
        let scale = catenary_scale();
        let sag = LENGTH * BEADS as f64 * BEAD_MASS * GRAVITY / STIFFNESS;
        for bead in s.chains[STEPS].iter() {
            let below = (bead | Y).s() - catenary_height(scale, (bead | X).s());
            assert!(below.abs() < sag, "{bead:?} {below}");
        }
    }

    /// The residuals of the iterates match numga's (the notebook run at numga e2765b5) until
    /// rounding takes over.
    #[test]
    fn the_iterates_match_numga() {
        let numga = [
            6.270089,
            2.122154,
            1.399434,
            0.4286976,
            2.143497e-2,
            4.881062e-5,
        ];
        for (ours, theirs) in solve().residuals.iter().zip(numga) {
            assert!((ours - theirs).abs() < 1e-6 * theirs, "{ours} vs {theirs}");
        }
    }

    /// The start stretches every spring, as Newton's method needs.
    #[test]
    fn the_start_stretches_every_spring() {
        let chain = start();
        let [left, right] = pins();
        let beads: Vec<V> = chain.iter().collect();
        let ends = [left].into_iter().chain(beads).chain([right]);
        let links: Vec<V> = ends.clone().zip(ends.skip(1)).map(|(a, b)| b - a).collect();
        assert!(links.iter().all(|l| l.norm() > rest()));
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws(super::draw, 0.5);
    }
}
