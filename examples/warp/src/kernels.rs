//! Kernels written once and traced at build time (`build.rs`) into fused Rust functions (with
//! SIMD batch forms) and WGSL functions: the GPU runs them in compute shaders, and the CPU runs
//! the same programs as the fallback and as the test oracle.
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
/// * `k`: `[spring stiffness, rest length, anchor stiffness, damping, dt]`.
///
/// Returns the new position and velocity.
pub fn grid_node<T: Real>(
    p: Point<(), T>,
    v: Point<(), T>,
    rest: Point<(), T>,
    n: [Point<(), T>; 4],
    force: Point<(), T>,
    k: [T; 5],
) -> (Point<(), T>, Point<(), T>) {
    let [spring, len, anchor, damping, dt] = k;
    let eps = T::from_f64(1e-4);
    let mut acc = force + (rest - p).gp(anchor);
    for q in n {
        // Hooke along each spring: the join's norm is the distance.
        let dist = (p & q).norm();
        acc += (q - p).gp(spring * (dist - len) / (dist + eps));
    }
    let v = (v + acc.gp(dt)).gp(T::one() - damping * dt);
    (p + v.gp(dt), v)
}

/// The acceleration a source at `s` gives a body at `p`: `strength (s - p) / (|s - p|² + r²)`,
/// towards the source for a positive strength (a well), away for a negative one (a blast).
/// `k` is `[strength, r²]`.
pub fn source_force<T: Real>(p: Point<(), T>, s: Point<(), T>, k: [T; 2]) -> Point<(), T> {
    let [strength, r2] = k;
    let d2 = (p & s).norm_squared();
    (s - p).gp(strength / (d2 + r2))
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
    let v = (v + force.gp(dt)).gp(T::one() - drag * dt);
    (p + v.gp(dt), v)
}

/// `if a < b { x } else { y }`, coefficient-wise on points (branch-free, so it traces).
fn select_point<T: Real>(a: T, b: T, x: Point<(), T>, y: Point<(), T>) -> Point<(), T> {
    Point::from_coeffs(core::array::from_fn(|i| T::select_lt(a, b, x.c[i], y.c[i])))
}

/// A corner of the quad drawn around the segment `a → b` (the line renderer's vertex shader):
/// `k` is `[along, side, extent]`, with `along` and `side` in `{-1, 1}`. The quad reaches
/// `extent` past both ends and to both sides. The side direction is the segment's direction
/// turned a quarter about the origin; a degenerate segment points along `x`.
pub fn segment_corner<T: Real>(a: Point<(), T>, b: Point<(), T>, k: [T; 3]) -> Point<(), T> {
    let [along, side, extent] = k;
    let (zero, one) = (T::zero(), T::one());
    let d = b - a;
    let len = d.ideal_norm();
    let u = select_point(
        len,
        T::from_f64(1e-6),
        Point::direction(one, zero),
        d.gp(one / (len + T::from_f64(1e-30))),
    );
    let quarter = gax::pga2d::Motor::rotation(
        Point::xy(zero, zero),
        T::from_f64(core::f64::consts::FRAC_PI_2),
    );
    let n = quarter >> u;
    let reach = T::select_lt(along, zero, -extent, len + extent);
    a + u.gp(reach) + n.gp(side * extent)
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
    let to_ends = -(-(a & p).norm()).max(-(b & p).norm());
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
    -(-stretch.max(motion)).max(-one)
}

/// Lights: homogeneous points in linear RGB space (`light.rs`). Grassmann's laws of colour
/// mixing are affine combinations of these points.
pub type Light<T> = gax::pga3d::Point<(), T>;

/// The additive mix `(1 - t) a + t b`.
pub fn light_mix<T: Real>(a: Light<T>, b: Light<T>, t: T) -> Light<T> {
    a.gp(T::one() - t) + b.gp(t)
}

/// `l` whitened by `t` at the same intensity: mixed with white light of its own weight.
pub fn light_whiten<T: Real>(l: Light<T>, t: T) -> Light<T> {
    let w = l.e123();
    light_mix(l, Light::new(w, w, w, w), t)
}

/// `l` at `k` times the intensity: the same point, a different weight.
pub fn light_fade<T: Real>(l: Light<T>, k: T) -> Light<T> {
    l.gp(k)
}

/// The tail of a streak: where a point moving at `v` was `dt` ago.
pub fn streak_tail<T: Real>(p: Point<(), T>, v: Point<(), T>, dt: T) -> Point<(), T> {
    p - v.gp(dt)
}
