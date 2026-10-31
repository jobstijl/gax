//! numga's `math/e_series`: the exceptional root systems E6, E7 and E8, nested, from the
//! first six, seven or eight of E8's mirrors.
//!
//! In R(8,0) (shared with `math_e8`), the first six, seven or eight of E8's simple roots
//! generate E6, E7 or E8. Each system's reflections, composed as maps on vectors, are its
//! Coxeter motion, which repeats after 12, 18 or 30 steps; its orbits through one seed each
//! (each simple root reflected in the mirrors before it) hold all 72, 126 or 240 roots, and every
//! generating mirror permutes them. Averaged with its adjoint, a pairing solve against the
//! metric, the motion is symmetric, and one eigensolve gives its planes of turning: E6 turns
//! three planes 1, 4 and 5 times in twelve steps, E7 three planes 1, 5 and 7 times in eighteen
//! and reverses one more direction, E8 four planes 1, 7, 11 and 13 times in thirty. Eigenvectors
//! outside the roots' span have cosine one and sort last.
//!
//! The roots' span is a projector, the sum of each root's dyad: its trace is the rank, it keeps
//! every root, the smaller systems' projectors sit inside the larger ones', and the planar turns
//! with the reversed direction rebuild the whole motion. A mirror moves everything covariantly:
//! the roots of the reflected mirrors are the reflected roots.
//!
//! The animation: left all 240 roots in E8's Coxeter plane, coloured by the first system that
//! holds them, the systems lit in turn; right E8's diagram, coloured by the stage at which each
//! mirror enters.

#[path = "../shared/math_e8.rs"]
mod e8;

use gax_numga_examples::{
    Align, Anim, Axes, Canvas, Light, ORIGIN2, Point2, Rect, backdrop, caption, font, palette,
    run_with,
};

/// The three systems: E6, E7 and E8.
mod series {
    use super::e8::{self, Vector, VectorMap};

    /// The systems' names.
    pub const NAMES: [&str; 3] = ["E6", "E7", "E8"];
    /// Their ranks, the number of mirrors that generate each.
    pub const RANKS: [usize; 3] = [6, 7, 8];
    /// Their Coxeter numbers, the steps after which each motion repeats.
    pub const PERIODS: [usize; 3] = [12, 18, 30];

    /// The Coxeter motion, the generating reflections composed, and the roots it carries,
    /// `[step][orbit]`, one seed from each orbit through.
    pub fn root_system(simple: &[Vector], n_steps: usize) -> (VectorMap, Vec<Vec<Vector>>) {
        // Composing from the right matches the order of the prefixes that pick the seeds.
        let action = simple
            .iter()
            .map(|r| e8::reflection(*r))
            .reduce(|left, right| left.of(right))
            .expect("mirrors");
        let seeds = e8::reflection_seeds(simple);
        (action, e8::orbit(action, &seeds, n_steps))
    }

    /// A Coxeter motion's orthogonal planes of turning, from slowest to fastest: their frames
    /// and their angles.
    pub fn rotation_planes(action: VectorMap, rank: usize) -> (Vec<Vector>, Vec<Vector>, Vec<f64>) {
        let (cosines, eigenvectors) = e8::symmetric(action).eigh();
        // The eigenvectors outside the root span have cosine one and sort last. Paired cosines
        // describe turns; an unpaired minus one describes a reflection.
        let picks: Vec<usize> = (1..rank).rev().step_by(2).collect();
        let plane_x: Vec<Vector> = picks.iter().map(|k| eigenvectors[*k]).collect();
        let plane_y: Vec<Vector> = picks
            .iter()
            .zip(&plane_x)
            .map(|(k, x)| (action.of(*x) - x.gp(cosines[*k])).normalized())
            .collect();
        let angles = plane_x
            .iter()
            .zip(&plane_y)
            .map(|(x, y)| e8::turn_angle(action, *x, *y))
            .collect();
        (plane_x, plane_y, angles)
    }

    /// One system of the series: its motion, roots, planes and angles.
    pub struct System {
        #[cfg_attr(not(test), allow(dead_code))] // numga's checks of the motion
        pub action: VectorMap,
        pub roots: Vec<Vec<Vector>>,
        pub plane_x: Vec<Vector>,
        pub plane_y: Vec<Vector>,
        pub angles: Vec<f64>,
    }

    impl System {
        /// Every root, orbit by orbit within each step.
        pub fn all(&self) -> impl Iterator<Item = Vector> + '_ {
            self.roots.iter().flatten().copied()
        }
    }

    /// Each system's Coxeter motion and roots, with its invariant plane frames and turns.
    pub fn family() -> [System; 3] {
        let simple = e8::simple_roots();
        core::array::from_fn(|k| {
            let (action, roots) = root_system(&simple[..RANKS[k]], PERIODS[k]);
            let (plane_x, plane_y, angles) = rotation_planes(action, RANKS[k]);
            System {
                action,
                roots,
                plane_x,
                plane_y,
                angles,
            }
        })
    }

    /// The projector onto the roots' span: the sum of their dyads, over twice the steps (each
    /// orbit's dyads sum to twice the projector's share).
    #[cfg_attr(not(test), allow(dead_code))] // numga's checks of the span
    pub fn span(roots: &[Vec<Vector>]) -> VectorMap {
        let dyads = roots.iter().flatten().map(|r| *r * (*r | Vector::slot()));
        dyads
            .fold(VectorMap::zero(), |sum, d| sum + d)
            .gp(1.0 / (2 * roots.len()) as f64)
    }
}

use e8::Vector;
use e8::look::{coordinates, orbit_colour};
use series::{NAMES, PERIODS, RANKS};

/// The seconds each system stays lit.
const STAGE: f32 = 3.0;

/// The three systems' colours: numga's orbit colours 6, 1 and 4.
fn family_colour(stage: usize) -> Light {
    orbit_colour([5, 0, 3][stage])
}

/// What the frames draw: E8's roots in its Coxeter plane, the stage that first holds each, the
/// nearest pairs, and the counts.
struct Scene {
    points: Vec<Point2>,
    stages: Vec<usize>,
    edges: Vec<(usize, usize)>,
    reach: f64,
    turns: [Vec<f64>; 3],
    simple: [Vector; 8],
}

impl Scene {
    fn new() -> Scene {
        let family = series::family();
        let e8 = &family[2];
        let (x, y) = (e8.plane_x[0], e8.plane_y[0]);
        let roots: Vec<Vector> = e8.all().collect();
        // Membership is measured in the full root space, before any projection.
        let stages = roots
            .iter()
            .map(|r| {
                let held = |s: &series::System| s.all().any(|q| (*r - q).norm() < 1e-8);
                family.iter().position(held).expect("E8 holds every root")
            })
            .collect();
        let projection = |r: &Vector| x.gp((x | *r).s()) + y.gp((y | *r).s());
        let reach = roots
            .iter()
            .map(|r| projection(r).norm())
            .fold(0.0, f64::max);
        let turns = core::array::from_fn(|k| {
            let period = PERIODS[k] as f64 / core::f64::consts::TAU;
            family[k]
                .angles
                .iter()
                .map(|a| (a * period).round())
                .collect()
        });
        Scene {
            points: roots.iter().map(|r| coordinates(*r, x, y)).collect(),
            stages,
            edges: e8::nearest_pairs(&roots),
            reach,
            turns,
            simple: e8::simple_roots(),
        }
    }
}

/// E6 inside E7 inside E8, all in E8's Coxeter plane: the roots coloured by the first system
/// holding them, those of the systems up to `lit` bright; the edges a faint haze.
fn roots_panel(c: &mut Canvas, scene: &Scene, rect: Rect, lit: usize) {
    let ax = Axes::equal(rect, ORIGIN2, (scene.reach * 1.045) as f32);
    for &(i, j) in &scene.edges {
        let stage = scene.stages[i].max(scene.stages[j]);
        let haze = if stage <= lit { 0.08 } else { 0.025 };
        ax.line(
            c,
            scene.points[i],
            scene.points[j],
            0.5,
            family_colour(stage).faded(haze),
        );
    }
    for (p, stage) in scene.points.iter().zip(&scene.stages) {
        let px = ax.px(*p);
        let bright = if *stage <= lit { 1.0 } else { 0.22 };
        c.disk(px, 3.0, palette::BOTTOM.faded(4.0));
        c.disk(px, 2.2, family_colour(*stage).faded(bright));
    }
    let mut counts = [0; 3];
    for s in &scene.stages {
        counts[*s] += 1;
    }
    for k in 0..3 {
        let label = if k == 0 {
            format!("{}: {}", NAMES[k], counts[k])
        } else {
            format!("{}: +{}", NAMES[k], counts[k])
        };
        let across = rect.width() * (0.23 + 0.27 * k as f32);
        let at = rect.bottom_left() + Point2::direction(across, 0.0);
        let shade = if k <= lit { 1.0 } else { 0.4 };
        c.text(
            &label,
            at,
            14.0,
            family_colour(k).faded(shade),
            Align::Center,
        );
    }
}

/// E8's diagram, each node coloured by the stage at which its mirror enters, those up to `lit`
/// bright, joined where two mirrors pair negatively; a bracket under E6's chain, and each
/// system's turns.
fn diagram_panel(c: &mut Canvas, scene: &Scene, rect: Rect, lit: usize) {
    const PLACES: [[f32; 2]; 8] = [
        [0.0, 0.0],
        [2.0, 1.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [3.0, 0.0],
        [4.0, 0.0],
        [5.0, 0.0],
        [6.0, 0.0],
    ];
    let ax = Axes::new(rect, [-0.5, 6.5], [-1.0, 1.4]);
    let place = |k: usize| Point2::xy(PLACES[k][0], PLACES[k][1]);
    let stage = |k: usize| RANKS.iter().position(|r| k < *r).expect("eight nodes");
    let shade = |s: usize| if s <= lit { 1.0 } else { 0.35 };
    for i in 0..8 {
        for j in i + 1..8 {
            if (scene.simple[i] | scene.simple[j]).s() < 0.0 {
                let s = stage(i).max(stage(j));
                ax.line(
                    c,
                    place(i),
                    place(j),
                    1.6,
                    family_colour(s).faded(0.7 * shade(s)),
                );
            }
        }
    }
    for k in 0..8 {
        let px = ax.px(place(k));
        let s = stage(k);
        c.disk(px, 12.0, palette::BOTTOM.faded(4.0));
        c.disk(px, 10.0, family_colour(s).faded(0.5 * shade(s)));
        c.ring(px, 10.0, 1.2, family_colour(s).faded(shade(s)));
        let number = format!("{}", k + 1);
        c.text(
            &number,
            px + Point2::direction(0.0, 5.0),
            13.0,
            palette::INK,
            Align::Center,
        );
    }
    let bracket = [
        Point2::xy(0.0, -0.3),
        Point2::xy(0.0, -0.4),
        Point2::xy(4.0, -0.4),
        Point2::xy(4.0, -0.3),
    ];
    ax.polyline(c, &bracket, 1.0, family_colour(0).faded(shade(0)));
    for (k, x) in [2.0, 5.0, 6.0].into_iter().enumerate() {
        let at = ax.px(Point2::xy(x, -0.75));
        c.text(
            NAMES[k],
            at,
            15.0,
            family_colour(k).faded(shade(k)),
            Align::Center,
        );
    }
    let mut lines: Vec<(String, Light)> = (0..3)
        .map(|k| {
            let turns: Vec<String> = scene.turns[k].iter().map(|t| format!("{t}")).collect();
            let line = format!(
                "{}: {} STEPS; TURNS {}",
                NAMES[k],
                PERIODS[k],
                turns.join(", ")
            );
            (line, family_colour(k).faded(shade(k)))
        })
        .collect();
    let reversal = "E7 ALSO REVERSES ONE DIRECTION".to_string();
    lines.push((reversal, family_colour(1).faded(0.6 * shade(1))));
    // One size for all the lines, the largest at which the longest fits the panel.
    let longest = lines
        .iter()
        .map(|(l, _)| font::width(l, 1.0))
        .fold(0.0, f32::max);
    let size = (rect.width() / longest).min(12.0);
    for (k, (line, colour)) in lines.iter().enumerate() {
        let at = rect.bottom_left() + Point2::direction(0.0, 40.0 + 2.1 * size * k as f32);
        c.text(line, at, size, *colour, Align::Left);
    }
}

fn draw(c: &mut Canvas, scene: &Scene, t: f32) {
    backdrop(c);
    let screen = c.rect();
    let lit = ((t / STAGE) as usize).min(2);
    roots_panel(c, scene, screen.part(0.02, 0.13, 0.52, 0.95), lit);
    diagram_panel(c, scene, screen.part(0.56, 0.2, 0.97, 0.5), lit);
    caption(
        c,
        "E6, E7, E8: NESTED ROOT SYSTEMS FROM NESTED MIRRORS (R(8,0))",
        "THE FIRST 6, 7 OR 8 OF E8'S MIRRORS; ALL ROOTS IN E8'S COXETER PLANE, COLOURED BY THE FIRST SYSTEM HOLDING THEM",
    );
}

fn main() {
    run_with(
        Anim::new("e_series", 3.0 * STAGE).size(960, 540),
        Scene::new,
        draw,
    );
}

#[cfg(test)]
mod tests {
    use super::e8::{self, Vector, VectorMap};
    use super::series::{self, PERIODS, RANKS};
    use gax::ApproxEq;

    const TOLERANCE: f64 = 1e-10;

    /// The distance from `x` to the nearest of `roots`.
    fn nearest(x: Vector, roots: &[Vector]) -> f64 {
        roots
            .iter()
            .map(|r| (x - *r).norm())
            .fold(f64::INFINITY, f64::min)
    }

    /// The cosine and sine of an angle, a phasor's coordinates (gax's direction at an angle).
    fn cos_sin(angle: f64) -> (f64, f64) {
        let p = gax::pga2d::Point::<(), f64>::direction_from_angle(angle);
        (p.e20(), p.e01())
    }

    /// numga's parametrized test, for E6, E7 and E8: the roots are distinct, of squared length
    /// two, and every generating mirror permutes them; the motion closes each orbit, and moves
    /// every orbit away from its seed. The span is a projector of trace the rank holding every
    /// root; the plane frames are orthonormal, turn at the system's rates, and rebuild the
    /// motion with the remaining reflection, which E7 has and E6 and E8 have not.
    #[test]
    fn root_closure_and_rotation_planes() {
        let family = series::family();
        let cases: [(usize, &[f64]); 3] = [
            (72, &[1.0, 4.0, 5.0]),
            (126, &[1.0, 5.0, 7.0]),
            (240, &[1.0, 7.0, 11.0, 13.0]),
        ];
        for (k, (n_roots, rates)) in cases.into_iter().enumerate() {
            let s = &family[k];
            let (rank, n_steps) = (RANKS[k], PERIODS[k]);
            let simple = &e8::simple_roots()[..rank];
            let roots: Vec<Vector> = s.all().collect();
            let span = series::span(&s.roots);
            let plane_span = s
                .plane_x
                .iter()
                .zip(&s.plane_y)
                .map(|(x, y)| *x * (*x | Vector::slot()) + *y * (*y | Vector::slot()))
                .fold(VectorMap::zero(), |sum, p| sum + p);
            let remaining = span - plane_span;
            let frame: Vec<Vector> = s
                .plane_x
                .iter()
                .zip(&s.plane_y)
                .flat_map(|(x, y)| [*x, *y])
                .collect();

            // Independent planar turns reconstruct the motion; E7 also reverses one direction.
            let turning = s
                .plane_x
                .iter()
                .zip(&s.plane_y)
                .zip(&s.angles)
                .map(|((x, y), a)| (*x ^ *y).gp(-a / 2.0).exp())
                .reduce(|a, b| a * b)
                .expect("planes");
            let reconstructed = (turning >> Vector::slot()).of(Vector::slot() - remaining.gp(2.0));

            assert_eq!(roots.len(), n_roots);
            for (i, a) in roots.iter().enumerate() {
                for b in &roots[i + 1..] {
                    assert!((*a - *b).norm() > 1.0);
                }
                assert!((e8::length_squared(*a) - 2.0).abs() < TOLERANCE);
            }

            // Each generating reflection permutes the complete set, including negative roots.
            for mirror in simple {
                for r in &roots {
                    assert!(nearest(mirror.unit() >> *r, &roots) < TOLERANCE);
                }
            }
            for (last, first) in s.roots[n_steps - 1].iter().zip(&s.roots[0]) {
                assert!(s.action.of(*last).approx_eq(first, TOLERANCE));
            }
            for step in &s.roots[1..] {
                let moved: f64 = step
                    .iter()
                    .zip(&s.roots[0])
                    .map(|(a, b)| e8::length_squared(*a - *b))
                    .sum();
                assert!(moved > 1.0);
            }

            assert!(span.of(span).approx_eq(&span, TOLERANCE));
            for r in &roots {
                assert!(span.of(*r).approx_eq(r, TOLERANCE));
            }
            assert!((span.trace() - rank as f64).abs() < TOLERANCE);
            for (i, a) in frame.iter().enumerate() {
                for (j, b) in frame.iter().enumerate() {
                    let expected = if i == j { 1.0 } else { 0.0 };
                    assert!(((*a | *b).s() - expected).abs() < TOLERANCE);
                }
            }
            for (angle, rate) in s.angles.iter().zip(rates) {
                let turns = angle * n_steps as f64 / core::f64::consts::TAU;
                assert!((turns - rate).abs() < TOLERANCE, "{turns} {rate}");
            }
            assert_eq!(s.angles.len(), rates.len());
            for ((x, y), angle) in s.plane_x.iter().zip(&s.plane_y).zip(&s.angles) {
                let (cos, sin) = cos_sin(*angle);
                let x_moved = s.action.of(*x) - x.gp(cos) - y.gp(sin);
                let y_moved = s.action.of(*y) + x.gp(sin) - y.gp(cos);
                assert!(x_moved.approx_eq(&Vector::zero(), TOLERANCE));
                assert!(y_moved.approx_eq(&Vector::zero(), TOLERANCE));
            }
            assert!((remaining.trace() - (rank % 2) as f64).abs() < TOLERANCE);
            assert!(remaining.of(remaining).approx_eq(&remaining, TOLERANCE));
            assert!(
                s.action
                    .of(remaining)
                    .approx_eq(&remaining.gp(-1.0), TOLERANCE)
            );
            assert!(reconstructed.approx_eq(&s.action, TOLERANCE));
        }
    }

    /// numga's second test: each system's roots lie in the next, and the spans nest; a mirror
    /// moves the root systems of the mirrored simple roots covariantly: their roots are the
    /// mirrored roots, their motions and spans the conjugated ones.
    #[test]
    fn nested_root_systems_and_mirror_covariance() {
        let family = series::family();
        let a = |k: usize| e8::direction(k);
        let mirror = (a(1) + a(2).gp(2.0) - a(3).gp(3.0) + a(8)).unit();
        let spans: Vec<VectorMap> = family.iter().map(|s| series::span(&s.roots)).collect();

        for k in 0..2 {
            let outer: Vec<Vector> = family[k + 1].all().collect();
            for r in family[k].all() {
                assert!(nearest(r, &outer) < TOLERANCE);
            }
            assert!(spans[k + 1].of(spans[k]).approx_eq(&spans[k], TOLERANCE));
            assert!(spans[k].of(spans[k + 1]).approx_eq(&spans[k], TOLERANCE));
        }

        let mirrored: VectorMap = mirror >> Vector::slot();
        let unmirrored: VectorMap = mirror << Vector::slot();
        for (k, s) in family.iter().enumerate() {
            let simple: Vec<Vector> = e8::simple_roots()[..RANKS[k]]
                .iter()
                .map(|r| mirror >> *r)
                .collect();
            let (action, roots) = series::root_system(&simple, PERIODS[k]);
            let span = series::span(&roots);
            for (a, b) in roots.iter().flatten().zip(s.all()) {
                assert!(a.approx_eq(&(mirror >> b), TOLERANCE));
            }
            let conjugated = mirrored.of(s.action.of(unmirrored));
            assert!(action.approx_eq(&conjugated, TOLERANCE));
            assert!(span.approx_eq(&mirrored.of(spans[k].of(unmirrored)), TOLERANCE));
        }
    }

    /// The scenario's checks: the orbits hold 72, 126 and 240 roots of squared length two.
    #[test]
    fn the_systems_hold_72_126_and_240_roots() {
        for (s, n) in series::family().iter().zip([72, 126, 240]) {
            assert_eq!(s.all().count(), n);
            for r in s.all() {
                assert!((e8::length_squared(r) - 2.0).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn every_stage_draws() {
        let scene = super::Scene::new();
        for t in [1.0, 4.0, 7.0] {
            gax_numga_examples::app::assert_draws_with(super::draw, &scene, t);
        }
    }
}
