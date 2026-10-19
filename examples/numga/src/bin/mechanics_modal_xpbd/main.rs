//! numga's `mechanics/modal_xpbd`: flexible bodies joined by point constraints, in PGA2D.
//!
//! Each body is a girder of bars. Its rigid motion is a motor and its deformation a combination
//! of its lowest vibration modes, the eigenfields of its sparse stiffness against its masses
//! (`SparseMap::eigh`), each of unit mass. Bodies are connected by point constraints between
//! anchor points. Time stepping is modal extended position-based dynamics: rigid bodies under
//! compliant constraints, each mode one more constraint.
//!
//! For an anchor, `anchor.commutator(Twist)` maps a twist of its body to the anchor's
//! displacement, `Direction <- Twist`; over all constraints and bodies these maps form a sparse
//! field map. Its adjugate maps forces at the constraints to forques on the bodies: a force is
//! held as the direction it pulls in, its work along a displacement is its polar (numga's force,
//! a line through the origin) joined with the displacement, and the forque a force exerts is the
//! line whose pairing with every twist is that work, a pairing solve. Composed through the
//! inverse inertias, with the modes' couplings and the constraints' compliance added, they give
//! one sparse system for all constraints, solved once per step by sparse LU. Stiffness enters
//! as compliance, so arbitrarily stiff modes and constraints stay well conditioned, down to the
//! rigid limit of zero compliance, and modes too fast for the step are damped, not unstable.
//! The rigid motion between projections is the shared Lie-group RK4 under gravity.
//!
//! Inertia is evaluated in the rest configuration and the coupling of rotation and vibration is
//! neglected, which assumes rotation slow compared with the retained vibrations. The couplings
//! can also be found by differentiating the gaps (`derived`), and step the chain alike.
//!
//! In the notation of extended position-based dynamics the system reads as
//! `S = J_b M_b⁻¹ J_bᵀ + J_q W J_qᵀ + α̃_p`, with `W = α̃_m / (1 + α̃_m)`.
//!
//! Top left, a chain of four girders hinged at their corners swings down under gravity from a
//! fixed one, flexing as it goes; below, a beam of eight spliced girders between two fixed ones
//! is compressed from its end, and past Euler's critical load it buckles (the deflection of its
//! midspan, top right, against the end's displacement).

#[cfg(test)]
mod derived;
mod girder;
use gax_numga_examples::shared::mechanics_lie as lie;
mod scenes;
mod xpbd;

use gax::field::Field;
use gax_numga_examples::signal::frame;
use gax_numga_examples::{
    Align, Anim, Axes, Canvas, Light, Marker, Point2, Pos2, Rect, backdrop, caption, label_above,
    palette, run,
};
use girder::{P, Shape};
use scenes::*;
use std::sync::OnceLock;

/// numga's swing: 200 frames of 30 ms.
const SECONDS: f32 = 6.0;
/// The margin around the drawn motion, as a share of its extent.
const MARGIN: f32 = 0.06;

/// The chain's swing and the beam's compression, computed once.
struct Scene {
    chain: Shape,
    swing: Vec<Vec<Field<P>>>,
    beam: Shape,
    displacements: Vec<f64>,
    compression: Vec<Frame>,
}

fn scene() -> &'static Scene {
    static S: OnceLock<Scene> = OnceLock::new();
    S.get_or_init(|| {
        let chain = chain_girder();
        let (bodies, hinges) = hinged_chain(&chain, LINKS, DAMPING);
        let swing = swing(
            &chain,
            &bodies,
            &hinges,
            gravity(GRAVITY),
            INTERVAL,
            FRAMES,
            xpbd::coupling,
        );
        let beam = beam_girder();
        let (bodies, splices) = clamped_beam(&beam, BEAM_GIRDERS, BEAM_DAMPING);
        let displacements = displacements(BEAM_FRAMES);
        let compression = compress(
            &beam,
            &bodies,
            &splices,
            &displacements,
            BEAM_INTERVAL,
            BEAM_SUBSTEPS,
            xpbd::coupling,
        );
        Scene {
            chain,
            swing,
            beam,
            displacements,
            compression,
        }
    })
}

/// A body's light: the moving girders by turns, the fixed ones grey.
fn body_light(body: usize, fixed: bool) -> Light {
    if fixed {
        palette::INK.faded(0.35)
    } else {
        palette::series(body)
    }
}

/// The bars of every body.
fn draw_bars(
    c: &mut Canvas,
    ax: &Axes,
    shape: &Shape,
    bodies: &[Field<P>],
    fixed: impl Fn(usize) -> bool,
    width: f32,
) {
    for (b, points) in bodies.iter().enumerate() {
        let bars: Vec<[Point2; 2]> = shape
            .edges
            .iter()
            .map(|e| e.map(|v| points.site(v).point2()))
            .collect();
        ax.stroke(c, &bars, width, body_light(b, fixed(b)));
    }
}

/// The equal axes in `rect` around every point the frames reach, with a margin.
fn around(rect: Rect, frames: &[Vec<Field<P>>]) -> Axes {
    let points = frames
        .iter()
        .flatten()
        .flat_map(|f| f.iter().map(Pos2::point2));
    Axes::fitting(rect, points, 1.0 + 2.0 * MARGIN)
}

/// The hinged chain at a frame, with its pivot.
fn draw_chain(c: &mut Canvas, rect: Rect, s: &Scene, frame: usize) {
    let ax = around(rect, &s.swing);
    let bodies = &s.swing[frame];
    // The pivot: where the fixed girder holds the first moving one.
    let last = s.chain.rest.len() - 1;
    let pivot = bodies[0].site(last).point2();
    draw_bars(c, &ax, &s.chain, bodies, |b| b == 0, 1.6);
    ax.scatter(c, &[pivot], Marker::Dot, 8.0, palette::INK);
    label_above(c, rect, "A HINGED CHAIN SWINGING", 12.0, 6.0);
}

/// The compressed beam at a frame, its heights stretched to show the buckle: its moving girders
/// coloured, the fixed ones grey.
fn draw_beam(c: &mut Canvas, rect: Rect, s: &Scene, frame: usize) {
    let all = s
        .compression
        .iter()
        .flat_map(|f| f.points.iter().flat_map(Field::iter));
    let (mut left, mut right, mut reach) = (f64::MAX, f64::MIN, 0.0f64);
    for p in all {
        let [x, y] = p.to_euclidean();
        (left, right, reach) = (left.min(x), right.max(x), reach.max(y.abs()));
    }
    let reach = (reach * (1.0 + 2.0 * f64::from(MARGIN))) as f32;
    let ax = Axes::new(rect, [left as f32, right as f32], [-reach, reach]);
    let girders = s.compression[frame].points.len();
    let fixed = |b: usize| b == 0 || b == girders - 1;
    draw_bars(c, &ax, &s.beam, &s.compression[frame].points, fixed, 0.8);
    let stretch = (rect.height() / (2.0 * reach)) / (rect.width() / (right - left) as f32);
    let title =
        format!("A CLAMPED BEAM COMPRESSED FROM ITS END (HEIGHTS STRETCHED {stretch:.0} TIMES)");
    label_above(c, rect, &title, 12.0, 6.0);
}

/// The midspan's deflection against the end's displacement so far, and the critical
/// displacement.
fn draw_buckling(c: &mut Canvas, rect: Rect, s: &Scene, frame: usize) {
    let deflection: Vec<f64> = s
        .compression
        .iter()
        .map(|f| f.midspan.e01().abs())
        .collect();
    let top = deflection.iter().copied().fold(0.0, f64::max) as f32 * 1.1;
    let ax = Axes::new(rect, [0.0, END_DISPLACEMENT as f32], [0.0, top]);
    ax.frame(c, "MIDSPAN DEFLECTION", "END DISPLACEMENT", "");
    let critical = critical(BEAM_HEIGHT, (BEAM_GIRDERS * BEAM_CELLS) as f64) as f32;
    ax.dashed(
        c,
        &[Point2::xy(critical, 0.0), Point2::xy(critical, top)],
        1.2,
        4.0,
        palette::RED,
    );
    ax.text(
        c,
        Point2::xy(critical, top * 0.92),
        " CRITICAL",
        10.0,
        palette::RED,
        Align::Left,
    );
    let curve: Vec<Point2> = (0..=frame)
        .map(|k| Point2::xy(s.displacements[k] as f32, deflection[k] as f32))
        .collect();
    ax.polyline(c, &curve, 1.8, palette::SKY);
    ax.scatter(c, &curve[frame..], Marker::Dot, 8.0, palette::SKY);
}

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let s = scene();
    let fraction = (t / SECONDS).rem_euclid(1.0);
    let at = |frames: usize| frame(fraction, frames);
    let screen = c.rect();
    let top = screen.part(0.0, 0.15, 1.0, 0.7);
    draw_chain(
        c,
        top.part(0.0, 0.0, 0.58, 1.0).inset(20.0, 18.0, 10.0, 6.0),
        s,
        at(FRAMES),
    );
    let chart = top.part(0.58, 0.0, 1.0, 1.0).inset(56.0, 26.0, 24.0, 40.0);
    draw_buckling(c, chart, s, at(BEAM_FRAMES));
    let bottom = screen
        .part(0.0, 0.74, 1.0, 1.0)
        .inset(20.0, 22.0, 20.0, 8.0);
    draw_beam(c, bottom, s, at(BEAM_FRAMES));
    caption(
        c,
        "MODAL XPBD: FLEXIBLE GIRDERS UNDER POINT CONSTRAINTS",
        "MOTORS AND VIBRATION MODES, ALL CONSTRAINTS IN ONE SPARSE SOLVE PER STEP (PGA2D)",
    );
}

fn main() {
    run(Anim::new("modal_xpbd", SECONDS).size(960, 540), draw);
}

#[cfg(test)]
mod tests {
    use super::girder::{D, P, Shape, girder, stiffness, truss, work};
    use super::scenes::*;
    use super::xpbd::{Bodies, Constraints, Couple, coupling, step};
    use super::{derived, xpbd};
    use gax::Extensor;
    use gax::field::{Field, SparseMap};
    use gax::pga2d::{Direction, Motor, Point};

    /// The largest coordinate difference between the points of two sets of bodies.
    fn farthest(a: &[Field<P>], b: &[Field<P>]) -> f64 {
        let pairs = a.iter().zip(b).flat_map(|(x, y)| x.iter().zip(y.iter()));
        pairs
            .flat_map(|(p, q)| (p - q).c)
            .fold(0.0, |m: f64, d| m.max(d.abs()))
    }

    /// Steps of the bodies.
    fn run(
        bodies: &Bodies,
        constraints: &Constraints,
        dt: f64,
        steps: usize,
        g: P,
        couple: Couple,
    ) -> Bodies {
        (0..steps).fold(bodies.clone(), |b, _| step(&b, constraints, dt, g, couple))
    }

    #[test]
    fn work_is_the_dot_product_of_force_and_displacement() {
        let (force, along) = (Direction::new(2.0, -1.0), Direction::new(0.5, 3.0));
        assert!((work().of(force).of(along).s() - (1.0 - 3.0)).abs() < 1e-15);
    }

    /// The full truss's sag at its last vertex under gravity, clamped at its first two
    /// vertices: its stiffness on the free vertices, solved against their weights.
    fn full_truss_sag(cells: usize, bar_stiffness: f64, shape: &Shape, acceleration: f64) -> f64 {
        let full = stiffness(&truss(cells, 1.0, 0.2), bar_stiffness);
        // The couplings among the free vertices, numbered from the third.
        let kept: Vec<_> = full
            .couplings()
            .filter_map(|(r, c, cell)| Some((r.checked_sub(2)?, c.checked_sub(2)?, cell)))
            .collect();
        let n = shape.rest.len() - 2;
        let reduced = SparseMap::new(
            kept.iter().map(|k| k.2).collect(),
            kept.iter().map(|k| k.0).collect(),
            kept.iter().map(|k| k.1).collect(),
            (n, n),
        );
        let loads = Field::from_fn(n, |v| {
            D::new(0.0, shape.masses.site(v + 2).s() * acceleration)
        });
        reduced.solve_direct(&loads).site(n - 1).e01()
    }

    #[test]
    fn cantilever_static_sag_matches_full_truss() {
        let (cells, bar_stiffness) = (3, 100.0);
        let shape = girder(
            cells,
            1.0,
            0.2,
            bar_stiffness,
            1.0,
            2 * (2 * (cells + 1)) - 4,
        );
        let (bodies, constraints) = cantilever(&shape, 1, 2.0);
        let initial = bodies.points(&shape);
        // Implicit compliance keeps large steps stable; the over-damped girder settles within
        // them.
        let (dt, steps, acceleration) = (0.2, 60, -0.1);
        let settled = run(
            &bodies,
            &constraints,
            dt,
            steps,
            gravity(-acceleration),
            coupling,
        );
        let points = settled.points(&shape);
        let tip = points[1].site(shape.rest.len() - 1) - initial[1].site(shape.rest.len() - 1);
        let reference = full_truss_sag(cells, bar_stiffness, &shape, acceleration);
        assert!(
            (tip.e01() / reference - 1.0).abs() < 0.02,
            "{tip:?} {reference}"
        );
        let rates: f64 = settled
            .rate
            .iter()
            .map(|r| r.c.iter().map(|x| x * x).sum::<f64>())
            .sum();
        assert!(rates < 1e-10, "{rates}");
        assert!(farthest(&points[..1], &initial[..1]) <= 1e-11);
    }

    /// The chain's energy: the bodies' kinetic energy, rigid and modal, the modes' elastic
    /// energy, and the potential of the centres of mass in gravity (an ideal point).
    fn energy(bodies: &Bodies, gravity: P) -> f64 {
        let modes = 0..bodies.frequencies.len();
        let rigid: f64 = bodies
            .rate
            .iter()
            .map(|r| (*r & bodies.inertia.of(*r)).s() / 2.0)
            .sum();
        let modal: f64 = modes
            .clone()
            .map(|m| bodies.rates[m].dot(&bodies.rates[m]) / 2.0)
            .sum();
        let elastic: f64 = modes
            .map(|m| {
                bodies.amplitudes[m].dot(&bodies.amplitudes[m]) * bodies.frequencies[m].powi(2)
                    / 2.0
            })
            .sum();
        let pull = gravity.cast::<Direction>();
        let potential: f64 = bodies
            .motor
            .iter()
            .map(|m| {
                -work()
                    .of(pull)
                    .of(((*m >> Point::origin()) - Point::origin()).cast::<Direction>())
                    .s()
                    * bodies.mass
            })
            .sum();
        rigid + modal + elastic + potential
    }

    #[test]
    fn hinged_chain_sustains_large_rotations_with_small_flex() {
        let shape = girder(4, 1.0, 0.2, 300.0, 1.0, 8);
        let (mut bodies, hinges) = hinged_chain(&shape, 4, 0.02);
        let initial = bodies.points(&shape);
        let g = gravity(4.0);
        let (mut peak_gap, mut peak_flex) = (0.0f64, 0.0f64);
        for _ in 0..200 {
            bodies = step(&bodies, &hinges, 0.008, g, coupling);
            let gaps = hinges.gaps(&bodies);
            peak_gap = gaps
                .iter()
                .flat_map(|d| d.c)
                .fold(peak_gap, |m, x| m.max(x.abs()));
            for b in 0..bodies.motor.len() {
                let flexed = shape.deformed(|m| bodies.amplitudes[m].site(b).s());
                let offsets = flexed
                    .iter()
                    .zip(shape.rest.iter())
                    .map(|(p, q)| (p - q).ideal_norm());
                peak_flex = offsets.fold(peak_flex, f64::max);
            }
        }
        let points = bodies.points(&shape);
        assert!(points.iter().flat_map(|f| f.iter()).all(|p| p.is_finite()));
        assert!(energy(&bodies, g) < 1e-8, "{}", energy(&bodies, g));
        assert!(peak_gap < 0.001, "{peak_gap}");
        assert!(0.01 < peak_flex && peak_flex < 0.05, "{peak_flex}");
        assert!(farthest(&points, &initial) > 2.0);
        assert!(farthest(&points[..1], &initial[..1]) <= 1e-11);
    }

    #[test]
    fn a_clamped_beam_buckles_past_its_euler_load() {
        let beam = beam_girder();
        let (mut bodies, constraints) = clamped_beam(&beam, BEAM_GIRDERS, BEAM_DAMPING);
        // The two middle girders lifted by a hair: an imperfection to buckle from, so that the
        // buckle grows within a few steps rather than from round-off.
        let middle = BEAM_GIRDERS / 2;
        for motor in &mut bodies.motor[middle..middle + 2] {
            *motor = *motor * Motor::translation(Point::direction(0.0, 1e-6));
        }
        let (frames, substeps, dt) = (24, 4, 0.02);
        let displacements = displacements(frames);
        let compressed = compress(
            &beam,
            &bodies,
            &constraints,
            &displacements,
            dt,
            substeps,
            coupling,
        );
        let critical = critical(BEAM_HEIGHT, (BEAM_GIRDERS * BEAM_CELLS) as f64);
        for (d, frame) in displacements.iter().zip(&compressed) {
            let deflection = frame.midspan.e01().abs();
            if *d < 0.8 * critical {
                assert!(deflection < 1e-6, "{d} {deflection}");
            }
            if *d > 2.0 * critical {
                assert!(deflection > 0.5, "{d} {deflection}");
            }
        }
    }

    /// Couplings found by differentiating the gaps step the chain as the hand-built sparse
    /// couplings do.
    #[test]
    fn the_derived_step_follows_the_sparse_step() {
        let shape = girder(4, 1.0, 0.2, 300.0, 1.0, 8);
        let (bodies, hinges) = hinged_chain(&shape, 4, 0.02);
        let g = gravity(4.0);
        let by_hand = run(&bodies, &hinges, 0.002, 100, g, xpbd::coupling);
        let derived = run(&bodies, &hinges, 0.002, 100, g, derived::coupling);
        let difference = farthest(&by_hand.points(&shape), &derived.points(&shape));
        assert!(difference <= 1e-10, "{difference}");
    }

    /// Regression against numga (e2765b5): the girders' frequencies, and where the chain's last
    /// vertex has swung to after numga's test's 200 steps.
    #[test]
    fn frequencies_and_the_swing_match_numga() {
        let chain = [
            74.350_533_401_654,
            158.533_638_130_99,
            173.070_238_403_602,
            387.093_087_496_484,
        ];
        let beam = [
            4_629.420_903_794_6,
            12_572.189_224_082_5,
            24_128.174_385_385_1,
            94_815.484_224_559_8,
        ];
        let found = [chain_girder().frequencies, beam_girder().frequencies];
        for (frequencies, expected) in found.iter().zip([chain, beam]) {
            let (first, last) = (&frequencies[..3], frequencies[MODES - 1]);
            for (ours, theirs) in first.iter().chain([&last]).zip(expected) {
                assert!((ours / theirs - 1.0).abs() < 1e-9, "{ours} {theirs}");
            }
        }
        let shape = chain_girder();
        let (bodies, hinges) = hinged_chain(&shape, LINKS, DAMPING);
        let swung = run(&bodies, &hinges, 0.008, 200, gravity(GRAVITY), coupling);
        let end = swung.points(&shape)[LINKS]
            .site(shape.rest.len() - 1)
            .to_euclidean();
        let expected = [-0.615_507_530_497, -3.864_089_406_524];
        assert!(
            end.iter().zip(expected).all(|(a, b)| (a - b).abs() < 1e-8),
            "{end:?}"
        );
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws(super::draw, 3.0);
    }
}
