//! Kernels written once and traced at build time (`build.rs`) into fused Rust functions (with
//! SIMD batch forms) and WGSL functions: the GPU runs them in its shaders, and the tests run the
//! same programs on the CPU as the GPU's oracle (`render/cpu_grid.rs`). The simulation calls
//! the generic `segment_distance` (on `f32`) for its collisions, and the game `wave` for its
//! oscillations.
//!
//! Points are PGA2D points: a finite point is `(x, y, 1)`, a direction (a velocity, a force)
//! a point at infinity `(x, y, 0)`. The difference of two finite points is the direction
//! between them, and the norm of their join is their distance.

use gax::Real;
use gax::pga2d::Point;

/// One step of a lattice node of warped space.
///
/// * `p`, `v`: the node's position (a finite point) and velocity (a direction);
/// * `rest`: where it rests;
/// * `n`: its four neighbours' positions (a missing neighbour is `p` itself, which exerts no
///   force);
/// * `force`: the external acceleration (explosions, wells), a direction;
/// * `k`: `[spring stiffness, anchor stiffness, damping, dt]`.
///
/// The springs have rest length zero: each pulls by `spring (q - p)`, linear in the positions,
/// so the lattice pinches smoothly under a well or a blast instead of folding (a spring with a
/// rest length pushes back when compressed, and nodes pass through their neighbours). The
/// regular lattice is still at rest: its neighbours' pulls cancel.
///
/// Returns the new position and velocity.
pub fn grid_node<T: Real>(
    p: Point<(), T>,
    v: Point<(), T>,
    rest: Point<(), T>,
    n: [Point<(), T>; 4],
    force: Point<(), T>,
    k: [T; 4],
) -> (Point<(), T>, Point<(), T>) {
    let [spring, anchor, damping, dt] = k;
    let mut acc = force + (rest - p) * anchor;
    for q in n {
        acc += (q - p) * spring;
    }
    let v = (v + acc * dt) * (T::one() - damping * dt);
    (p + v * dt, v)
}

/// The rest position of the lattice's node in column `i` and row `j`: the lower-left node
/// `origin` moved by the direction `(i, j) spacing`. `k` is `[i, j, spacing]`.
pub fn lattice_rest<T: Real>(origin: Point<(), T>, k: [T; 3]) -> Point<(), T> {
    let [i, j, spacing] = k;
    origin + Point::direction(i * spacing, j * spacing)
}

/// The acceleration a source at `s` gives a body at `p`: `pull (s - p) / (|s - p|² + r²)`,
/// towards the source for a positive pull (a well), away for a negative one (a blast).
/// `k` is `[pull, r²]`.
pub fn source_force<T: Real>(p: Point<(), T>, s: Point<(), T>, k: [T; 2]) -> Point<(), T> {
    let [pull, r2] = k;
    let d2 = (p & s).norm_squared();
    (s - p) * (pull / (d2 + r2))
}

/// `acc` plus the acceleration of a source (`source_force`): forces add, as directions.
pub fn add_source_force<T: Real>(
    acc: Point<(), T>,
    p: Point<(), T>,
    s: Point<(), T>,
    k: [T; 2],
) -> Point<(), T> {
    acc + source_force(p, s, k)
}

/// One step of a particle: velocity from the external acceleration, drag, then motion.
/// `k` is `[drag, dt]`.
pub fn particle_step<T: Real>(
    p: Point<(), T>,
    v: Point<(), T>,
    force: Point<(), T>,
    k: [T; 2],
) -> (Point<(), T>, Point<(), T>) {
    let [drag, dt] = k;
    let v = (v + force * dt) * (T::one() - drag * dt);
    (p + v * dt, v)
}

/// A corner of the quad drawn around the segment `a → b` (the line renderer's vertex shader):
/// `k` is `[along, side, extent]`, with `along` and `side` in `{-1, 1}`. The quad reaches
/// `extent` past both ends and to both sides. The directions are the line `a & b`'s: along it
/// `b - a`, and its normal, that turned a quarter counterclockwise; a degenerate segment points
/// along `x`.
pub fn segment_corner<T: Real>(a: Point<(), T>, b: Point<(), T>, k: [T; 3]) -> Point<(), T> {
    let [along, side, extent] = k;
    let (zero, one) = (T::zero(), T::one());
    let l = a & b;
    let len = l.norm();
    let degenerate = |fallback, d: Point<(), T>| {
        gax::select_lt(
            len,
            T::from_f64(1e-6),
            fallback,
            d * (one / (len + T::from_f64(1e-30))),
        )
    };
    let u = degenerate(Point::direction(one, zero), l.direction());
    let n = degenerate(Point::direction(zero, one), l.normal());
    let reach = T::select_lt(along, zero, -extent, len + extent);
    a + u * reach + n * (side * extent)
}

/// The distance from `p` to the segment `a → b`, all with joins: inside the strip between
/// the lines through `a` and `b` perpendicular to the segment (`l | a`, `l | b`), it is the
/// distance to the line `l = a & b`; outside, to the nearer end.
pub fn segment_distance<T: Real>(a: Point<(), T>, b: Point<(), T>, p: Point<(), T>) -> T {
    let l = a & b;
    let len = l.norm();
    // Signed distances (up to the lines' norms) from the perpendiculars at the ends.
    let sa = ((l | a) & p).s();
    let sb = ((l | b) & p).s();
    let to_line = (l & p).s().abs() / (len + T::from_f64(1e-30));
    let to_ends = (a & p).norm().min((b & p).norm());
    T::select_lt(sa * sb, T::zero(), to_line, to_ends)
}

/// How hot a lattice edge `a → b` looks, `0..1`: stretched from its rest length, or moving.
/// `k` is `[rest length, stretch gain, motion gain]`.
pub fn edge_heat<T: Real>(
    a: Point<(), T>,
    b: Point<(), T>,
    va: Point<(), T>,
    vb: Point<(), T>,
    k: [T; 3],
) -> T {
    let [rest, stretch_gain, motion_gain] = k;
    let one = T::one();
    let stretch = ((a & b).norm() / rest - one).abs() * stretch_gain;
    let motion = (va.ideal_norm() + vb.ideal_norm()) * motion_gain;
    stretch.max(motion).min(one)
}

/// Lights: homogeneous points of linear sRGB, shared with the other examples (`gax-colour`);
/// the renderer traces these into its shaders.
#[cfg_attr(not(test), allow(unused_imports))] // Traced by build.rs; the tests check them.
pub use gax_colour::ops::{
    agx, fade as light_fade, luma, mix as light_mix, whiten as light_whiten,
};

/// The tail of a streak: where a point moving at `v` was `dt` ago.
pub fn streak_tail<T: Real>(p: Point<(), T>, v: Point<(), T>, dt: T) -> Point<(), T> {
    p - v * dt
}

/// A shock ripple: the point `p` pushed away from `centre` along its radial direction by a
/// ring at `k[0]` from the centre, of strength `k[1]`.
pub fn ripple<T: Real>(p: Point<(), T>, centre: Point<(), T>, k: [T; 2]) -> Point<(), T> {
    let [radius, strength] = k;
    let d = p - centre;
    let r = d.ideal_norm();
    let off = (r - radius) * T::from_f64(18.0);
    let ring = (-off * off).exp();
    p - d * (ring * strength / r.max(T::from_f64(1e-4)))
}

/// `p` scaled by `k` about `centre` (a homothety): the screen's edge effects.
pub fn scale_about<T: Real>(p: Point<(), T>, centre: Point<(), T>, k: T) -> Point<(), T> {
    centre + (p - centre) * k
}

/// The distance between two points: the norm of their join.
pub fn distance<T: Real>(a: Point<(), T>, b: Point<(), T>) -> T {
    (a & b).norm()
}

/// The height of the unit phasor at `phase` from `(1, 0)`: a sine wave in time.
pub fn wave<T: Real>(phase: T) -> T {
    Point::direction_from_angle(phase).e01()
}

#[cfg(test)]
mod tests {
    use crate::sim::rng::Rng;
    use gax::pga2d::Point;

    /// Two numbers equal to within rounding.
    fn close(x: f32, y: f32) -> bool {
        (x - y).abs() <= 1e-4 * (1.0 + y.abs())
    }

    /// Two coefficient arrays equal to within rounding.
    fn same(a: &[f32], b: &[f32]) -> bool {
        a.iter().zip(b).all(|(&x, &y)| close(x, y))
    }

    /// The traced kernels equal the code they were traced from.
    #[test]
    fn traced_kernels_equal_their_source() {
        let mut rng = Rng::new(2);
        for _ in 0..500 {
            let mut pt = || Point::xy(rng.range(-5.0, 5.0), rng.range(-5.0, 5.0));
            let (p, rest, s) = (pt(), pt(), pt());
            let n = [pt(), pt(), pt(), pt()];
            let v = Point::direction(0.3, -0.7);
            let f = Point::direction(1.5, 0.2);
            let k = [95.0, 5.0, 4.5, crate::sim::DT];
            let (a, b) = (
                crate::grid_node(p, v, rest, n, f, k),
                crate::kernels::grid_node(p, v, rest, n, f, k),
            );
            assert!(same(&a.0.c, &b.0.c) && same(&a.1.c, &b.1.c), "{a:?} {b:?}");
            let (a, b) = (
                crate::particle_step(p, v, f, [2.0, crate::sim::DT]),
                crate::kernels::particle_step(p, v, f, [2.0, crate::sim::DT]),
            );
            assert!(same(&a.0.c, &b.0.c) && same(&a.1.c, &b.1.c));
            let pairs = [
                (
                    crate::source_force(p, s, [-40.0, 2.0]),
                    crate::kernels::source_force(p, s, [-40.0, 2.0]),
                ),
                (
                    crate::add_source_force(f, p, s, [60.0, 2.0]),
                    crate::kernels::add_source_force(f, p, s, [60.0, 2.0]),
                ),
                (
                    crate::lattice_rest(rest, [7.0, 3.0, 0.5]),
                    crate::kernels::lattice_rest(rest, [7.0, 3.0, 0.5]),
                ),
                // The line renderer's geometry, a streak, and post-processing: the ripple and
                // edge scaling.
                (
                    crate::segment_corner(p, rest, [1.0, -1.0, 0.3]),
                    crate::kernels::segment_corner(p, rest, [1.0, -1.0, 0.3]),
                ),
                (
                    crate::streak_tail(p, v, 0.03),
                    crate::kernels::streak_tail(p, v, 0.03),
                ),
                (
                    crate::ripple(p, s, [0.3, 0.05]),
                    crate::kernels::ripple(p, s, [0.3, 0.05]),
                ),
                (
                    crate::scale_about(p, s, 0.9),
                    crate::kernels::scale_about(p, s, 0.9),
                ),
            ];
            for (a, b) in pairs {
                assert!(same(&a.c, &b.c), "{a:?} {b:?}");
            }
            assert!(close(
                crate::segment_distance(p, rest, s),
                crate::kernels::segment_distance(p, rest, s)
            ));
            let k = [0.5, 1.2, 0.012];
            assert!(close(
                crate::edge_heat(p, rest, v, f, k),
                crate::kernels::edge_heat(p, rest, v, f, k)
            ));
            // The traced kernels work on the lights' bare points.
            let l1 = crate::light::light(rng.unit(), rng.unit(), rng.unit(), 2.0).point();
            let l2 = crate::light::light(rng.unit(), rng.unit(), rng.unit(), 0.5).point();
            let t = rng.unit();
            let pairs = [
                (
                    crate::light_mix(l1, l2, t),
                    crate::kernels::light_mix(l1, l2, t),
                ),
                (
                    crate::light_whiten(l1, t),
                    crate::kernels::light_whiten(l1, t),
                ),
                (crate::light_fade(l1, t), crate::kernels::light_fade(l1, t)),
            ];
            for (a, b) in pairs {
                assert!(same(&a.c, &b.c));
            }
            // Luminance, the tonemapper, distance, the phasor.
            assert!(close(crate::luma(l1), crate::kernels::luma(l1)));
            let (a, b) = (crate::agx(l1, 1.35), crate::kernels::agx(l1, 1.35));
            assert!(
                a.c.iter().zip(&b.c).all(|(x, y)| (x - y).abs() < 1e-3),
                "{a:?} {b:?}"
            );
            assert!(close(crate::distance(p, s), crate::kernels::distance(p, s)));
            assert!(close(crate::wave(t * 7.0), crate::kernels::wave(t * 7.0)));
        }
    }

    /// The traced segment distance is the Euclidean distance to the segment.
    #[test]
    fn segment_distance_by_joins_is_the_distance() {
        let (a, b) = (Point::xy(0.0f32, 0.0), Point::xy(4.0, 0.0));
        let d = |x: f32, y: f32| crate::segment_distance(a, b, Point::xy(x, y));
        assert!((d(2.0, 1.5) - 1.5).abs() < 1e-5);
        assert!((d(-3.0, 4.0) - 5.0).abs() < 1e-5);
        assert!((d(7.0, -4.0) - 5.0).abs() < 1e-5);
        // A degenerate segment is a point.
        assert!((crate::segment_distance(a, a, Point::xy(3.0, 4.0)) - 5.0).abs() < 1e-5);
    }

    /// A lattice node with its neighbours where they rest feels no spring: the pulls of
    /// springs of rest length zero cancel on a regular lattice.
    #[test]
    fn a_regular_lattice_is_at_rest() {
        let p = Point::xy(1.0f32, 2.0);
        let n = [
            Point::xy(0.5, 2.0),
            Point::xy(1.5, 2.0),
            Point::xy(1.0, 1.5),
            Point::xy(1.0, 2.5),
        ];
        let zero = Point::zero();
        let (q, v) = super::grid_node(p, zero, p, n, zero, [95.0, 5.0, 4.5, crate::sim::DT]);
        assert_eq!((q, v), (p, zero));
    }
}
