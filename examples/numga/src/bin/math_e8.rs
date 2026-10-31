//! numga's `math/e8`: the 240 roots of E8, eight orbits of thirty under the Coxeter element,
//! and the four planes it turns in.
//!
//! In the geometric algebra of eight Euclidean dimensions, R(8,0), the eight simple roots are
//! mirrors, vectors of squared length two, and reflecting in one is `>>` with it. The product of
//! the eight unit mirrors is a rotor, and its sandwich on vectors is the Coxeter element, all
//! eight reflections as one map, which repeats after thirty steps. One root from each of eight
//! orbits (each simple root reflected in the mirrors before it in the product) seeds orbits of
//! thirty that hold all 240 roots, and every mirror permutes them, signs included.
//!
//! The Coxeter element is orthogonal, so its average with its adjoint (a pairing solve against
//! the metric) is symmetric, and one eigensolve splits space into four perpendicular planes, the
//! eigenvalues in pairs, each the cosine of one plane's turn: in thirty steps the planes turn
//! 1, 7, 11 and 13 times. The four turns commute, the planes wedge to zero with themselves,
//! and their exponentials multiply back to the Coxeter rotor; one exponential of the four
//! planes at once moves the roots continuously between the root system's symmetries. Seen in
//! the slowest plane, the Coxeter plane, the roots are E8's famous picture.
//!
//! The animation: left the roots in the Coxeter plane, coloured by orbit and joined to their
//! nearest neighbours; right the four invariant planes, each with one root of every orbit
//! riding its circle at the plane's rate.

// Not in the library: an 8D algebra that would add about 2 minutes to every library build.
#[path = "../shared/math_e8.rs"]
mod e8;

use e8::look::{coordinates, orbit_colour};
use e8::{COXETER_ORDER, RootSystem, Vector};
use gax_numga_examples::{
    Anim, Axes, Canvas, ORIGIN2, Point2, Rect, backdrop, caption, label_under, palette, points,
    run_with,
};

/// The loop's seconds: one Coxeter period of thirty steps.
const SECONDS: f32 = 15.0;
/// The share of the period each rider's trail covers: one Coxeter step.
const TRAIL: f64 = 1.0;
/// The samples along each trail.
const TRAIL_SAMPLES: usize = 24;

/// What the frames draw: the root system, its roots in one list, and their nearest pairs.
struct Scene {
    system: RootSystem,
    roots: Vec<Vector>,
    edges: Vec<(usize, usize)>,
}

impl Scene {
    fn new() -> Scene {
        let system = RootSystem::new();
        let roots: Vec<Vector> = system.all().collect();
        let edges = e8::nearest_pairs(&roots);
        Scene {
            system,
            roots,
            edges,
        }
    }
}

/// The roots in the Coxeter plane, the slowest plane: coloured by orbit, nearest neighbours
/// joined by faint edges of their orbits' colours mixed.
fn coxeter_panel(c: &mut Canvas, scene: &Scene, rect: Rect) {
    let s = &scene.system;
    let (x, y) = (s.plane_x[0], s.plane_y[0]);
    let at: Vec<Point2> = scene.roots.iter().map(|r| coordinates(*r, x, y)).collect();
    let reach = scene
        .roots
        .iter()
        .map(|r| (*r | s.planes[0]).norm())
        .fold(0.0, f64::max);
    let ax = Axes::equal(rect, ORIGIN2, (reach * 1.05) as f32);
    let colour = |i: usize| orbit_colour(i % 8);
    for &(i, j) in &scene.edges {
        let edge = colour(i).mix_light(colour(j), 0.5).faded(0.09);
        ax.line(c, at[i], at[j], 0.5, edge);
    }
    for (i, p) in at.iter().enumerate() {
        let px = ax.px(*p);
        c.disk(px, 3.2, palette::BOTTOM.faded(4.0));
        c.disk(px, 2.2, colour(i));
    }
    label_under(
        c,
        rect,
        "THE 240 ROOTS IN THE COXETER PLANE, COLOURED BY ORBIT",
        24.0,
        0.9,
    );
}

/// The four invariant planes at Coxeter step `step` (a real number): in each, every orbit's
/// circle and its roots faint, and one root of every orbit riding its circle with a trail one
/// step long. The riders are the seeds turned by the exponential of the four planes at once.
fn plane_panels(c: &mut Canvas, scene: &Scene, rect: Rect, step: f64) {
    let s = &scene.system;
    let generator = s.generator();
    let turned = |at: f64| -> Vec<Vector> {
        let rotor = generator.gp(at).exp();
        s.seeds.iter().map(|x| rotor >> *x).collect()
    };
    let trail: Vec<Vec<Vector>> = (0..=TRAIL_SAMPLES)
        .map(|k| turned(step - TRAIL * (1.0 - k as f64 / TRAIL_SAMPLES as f64)))
        .collect();
    let reach = (0..4)
        .flat_map(|k| scene.roots.iter().map(move |r| (*r | s.planes[k]).norm()))
        .fold(0.0, f64::max);
    for k in 0..4 {
        let cell = rect.part(
            0.5 * (k % 2) as f32,
            0.5 * (k / 2) as f32,
            0.5 * (k % 2 + 1) as f32,
            0.5 * (k / 2 + 1) as f32,
        );
        let room = cell.inset(6.0, 6.0, 6.0, 18.0);
        let ax = Axes::equal(room, ORIGIN2, (reach * 1.08) as f32);
        let (x, y) = (s.plane_x[k], s.plane_y[k]);
        for (o, seed) in s.seeds.iter().enumerate() {
            let radius = (*seed | s.planes[k]).norm() as f32;
            let ring = points::circle(ORIGIN2, radius, 96);
            ax.polyline(c, &ring, 0.6, orbit_colour(o).faded(0.26));
        }
        let faint: Vec<Point2> = scene.roots.iter().map(|r| coordinates(*r, x, y)).collect();
        for (i, p) in faint.iter().enumerate() {
            c.disk(ax.px(*p), 1.0, orbit_colour(i % 8).faded(0.35));
        }
        for o in 0..8 {
            let path: Vec<Point2> = trail.iter().map(|t| coordinates(t[o], x, y)).collect();
            for (n, pair) in path.windows(2).enumerate() {
                let share = (n + 1) as f32 / TRAIL_SAMPLES as f32;
                let fade = 0.12 + 0.73 * share;
                ax.line(c, pair[0], pair[1], 1.2, orbit_colour(o).faded(fade));
            }
            let rider = ax.px(path[TRAIL_SAMPLES]);
            c.disk(rider, 3.4, palette::INK.faded(0.7));
            c.disk(rider, 2.4, orbit_colour(o));
        }
        let rate = s.rates[k].round();
        let plural = if rate == 1.0 { "" } else { "S" };
        let turns = format!("PLANE {}: {rate} TURN{plural} IN 30 STEPS", k + 1);
        label_under(c, cell, &turns, 22.0, 0.9);
    }
}

fn draw(c: &mut Canvas, scene: &Scene, t: f32) {
    backdrop(c);
    let screen = c.rect();
    coxeter_panel(c, scene, screen.part(0.02, 0.13, 0.5, 0.97));
    let step = COXETER_ORDER as f64 * f64::from(t / SECONDS);
    plane_panels(c, scene, screen.part(0.52, 0.12, 0.98, 0.98), step);
    caption(
        c,
        "E8: EIGHT MIRRORS, ONE ROTOR, FOUR PLANES (R(8,0))",
        "THE COXETER ELEMENT, THE PRODUCT OF THE MIRRORS, TURNS FOUR PLANES 1, 7, 11 AND 13 TIMES IN 30 STEPS",
    );
}

fn main() {
    run_with(Anim::new("e8", SECONDS).size(960, 540), Scene::new, draw);
}

#[cfg(test)]
mod tests {
    use super::e8::{self, Bivector, COXETER_ORDER, RootSystem, Vector, VectorMap};
    use gax::ApproxEq;
    use std::collections::HashSet;

    const TOLERANCE: f64 = 1e-10;

    /// The roots that are distinct to nine decimals.
    fn distinct(roots: impl Iterator<Item = Vector>) -> usize {
        let key = |v: Vector| v.c.map(|x| (x * 1e9).round() as i64);
        roots.map(key).collect::<HashSet<_>>().len()
    }

    /// The distance from `x` to the nearest of `roots`.
    fn nearest(x: Vector, roots: &[Vector]) -> f64 {
        roots
            .iter()
            .map(|r| (x - *r).norm())
            .fold(f64::INFINITY, f64::min)
    }

    /// numga's test (and its scenario's checks): the orbits hold 240 distinct roots of squared
    /// length two, the Coxeter rotor closes each orbit, and every generating mirror permutes the
    /// roots, signs included. The plane frames are orthonormal, the planes turn 1, 7, 11 and 13
    /// times, commute, and wedge to zero with themselves; their exponentials multiply back to
    /// the Coxeter element; and the exponential of the four planes at once, in six substeps per
    /// step, passes through every root of every orbit and closes after thirty steps.
    #[test]
    fn roots_and_four_plane_rotation() {
        let s = RootSystem::new();
        let roots: Vec<Vector> = s.all().collect();
        assert_eq!(roots.len(), 240);
        assert_eq!(distinct(roots.iter().copied()), 240);
        for r in &roots {
            assert!((e8::length_squared(*r) - 2.0).abs() < TOLERANCE);
        }
        for (last, seed) in s.roots[COXETER_ORDER - 1].iter().zip(&s.seeds) {
            assert!((s.coxeter >> *last).approx_eq(seed, TOLERANCE));
        }

        // Every generating mirror permutes all 240 roots, including their signs.
        for mirror in e8::simple_roots() {
            for r in &roots {
                let reflected = mirror.unit() >> *r;
                assert!(nearest(reflected, &roots) < TOLERANCE);
            }
        }

        let frame: Vec<Vector> = (0..4).flat_map(|k| [s.plane_x[k], s.plane_y[k]]).collect();
        for (i, a) in frame.iter().enumerate() {
            for (j, b) in frame.iter().enumerate() {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(((*a | *b).s() - expected).abs() < TOLERANCE, "{i} {j}");
            }
        }
        for (rate, expected) in s.rates.iter().zip([1.0, 7.0, 11.0, 13.0]) {
            assert!((rate - expected).abs() < TOLERANCE, "{rate}");
        }
        for a in &s.planes {
            assert!(a.wedge(*a).approx_eq(&e8::Even::zero(), TOLERANCE));
            for b in &s.planes {
                assert!(a.commutator(*b).approx_eq(&Bivector::zero(), TOLERANCE));
            }
        }

        let step = -core::f64::consts::PI / COXETER_ORDER as f64;
        let factored = (0..4)
            .map(|k| s.planes[k].gp(s.rates[k] * step).exp())
            .reduce(|a, b| a * b)
            .expect("four planes");
        let coxeter: VectorMap = s.coxeter >> Vector::slot();
        assert!((factored >> Vector::slot()).approx_eq(&coxeter, TOLERANCE));

        // A bounded exponential advances the animation; its samples recover every orbit.
        let substeps = 6;
        let advance = s.generator().gp(1.0 / substeps as f64).exp() >> Vector::slot();
        let interpolated = e8::orbit(advance, &s.seeds, COXETER_ORDER * substeps + 1);
        for (step, roots) in s.roots.iter().enumerate() {
            for (a, b) in interpolated[step * substeps].iter().zip(roots) {
                assert!(a.approx_eq(b, TOLERANCE));
            }
        }
        for (a, b) in interpolated[COXETER_ORDER * substeps].iter().zip(&s.seeds) {
            assert!(a.approx_eq(b, TOLERANCE));
        }
    }

    /// Each root has 56 nearest neighbours, at squared distance two: the edges drawn.
    #[test]
    fn the_nearest_neighbours_are_the_root_polytope_s_edges() {
        let roots: Vec<Vector> = RootSystem::new().all().collect();
        let edges = e8::nearest_pairs(&roots);
        assert_eq!(edges.len(), 240 * 56 / 2);
        let (i, j) = edges[0];
        assert!((e8::length_squared(roots[i] - roots[j]) - 2.0).abs() < TOLERANCE);
    }

    #[test]
    fn a_frame_draws() {
        let scene = super::Scene::new();
        gax_numga_examples::app::assert_draws_with(super::draw, &scene, 1.0);
    }
}
