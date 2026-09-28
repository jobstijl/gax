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
    let mut acc = force + (rest - p) * anchor;
    for q in n {
        // Hooke along each spring: the join's norm is the distance.
        let dist = (p & q).norm();
        acc += (q - p) * (spring * (dist - len) / (dist + eps));
    }
    let v = (v + acc * dt) * (T::one() - damping * dt);
    (p + v * dt, v)
}

/// The acceleration a source at `s` gives a body at `p`: `strength (s - p) / (|s - p|² + r²)`,
/// towards the source for a positive strength (a well), away for a negative one (a blast).
/// `k` is `[strength, r²]`.
pub fn source_force<T: Real>(p: Point<(), T>, s: Point<(), T>, k: [T; 2]) -> Point<(), T> {
    let [strength, r2] = k;
    let d2 = (p & s).norm_squared();
    (s - p) * (strength / (d2 + r2))
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
/// `extent` past both ends and to both sides. The side direction is the segment's direction
/// turned a quarter about the origin; a degenerate segment points along `x`.
pub fn segment_corner<T: Real>(a: Point<(), T>, b: Point<(), T>, k: [T; 3]) -> Point<(), T> {
    let [along, side, extent] = k;
    let (zero, one) = (T::zero(), T::one());
    let d = b - a;
    let len = d.ideal_norm();
    let u = gax::select_lt(
        len,
        T::from_f64(1e-6),
        Point::direction(one, zero),
        d * (one / (len + T::from_f64(1e-30))),
    );
    let quarter = gax::pga2d::Motor::rotation(
        Point::xy(zero, zero),
        T::from_f64(core::f64::consts::FRAC_PI_2),
    );
    let n = quarter >> u;
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
    a * (T::one() - t) + b * t
}

/// `l` whitened by `t` at the same intensity: mixed with white light of its own weight.
pub fn light_whiten<T: Real>(l: Light<T>, t: T) -> Light<T> {
    let w = l.e123();
    light_mix(l, Light::new(w, w, w, w), t)
}

/// `l` at `k` times the intensity: the same point, a different weight.
pub fn light_fade<T: Real>(l: Light<T>, k: T) -> Light<T> {
    l * k
}

/// The tail of a streak: where a point moving at `v` was `dt` ago.
pub fn streak_tail<T: Real>(p: Point<(), T>, v: Point<(), T>, dt: T) -> Point<(), T> {
    p - v * dt
}

/// The luminance of a light: its pairing with the plane of Rec. 709's weights (`plane & light`,
/// `0.2126 R + 0.7152 G + 0.0722 B`); lights of equal luminance lie on planes parallel to it.
pub fn luma<T: Real>(l: Light<T>) -> T {
    let w = |x: f64| T::from_f64(x);
    let plane = gax::pga3d::Plane::new(w(0.2126), w(0.7152), w(0.0722), T::zero());
    (plane & l).s()
}

/// A linear map of colour, as a gax map on light points (the weight kept): `m` holds the
/// matrix's rows. Its inverse is the map's own `inverse()`.
pub fn colour_map<T: Real>(m: [[f64; 3]; 3]) -> gax::pga3d::Point<(gax::pga3d::Point,), T> {
    let w = |x: f64| T::from_f64(x);
    let (z, o) = (T::zero(), T::one());
    gax::pga3d::Point::from_coeffs([
        [w(m[0][0]), w(m[0][1]), w(m[0][2]), z],
        [w(m[1][0]), w(m[1][1]), w(m[1][2]), z],
        [w(m[2][0]), w(m[2][1]), w(m[2][2]), z],
        [z, z, z, o],
    ])
}

/// A function applied to each colour coordinate of a light (the weight kept).
fn each<T: Real>(l: Light<T>, f: impl Fn(T) -> T) -> Light<T> {
    Light::new(f(l.e032()), f(l.e013()), f(l.e021()), l.e123())
}

/// AgX tonemapping (Troy Sobotka's, in Benjamin Wrensch's minimal fitted form) of a light
/// whose coordinates are radiance, to display light: the inset map, a log encoding and a
/// contrast curve per coordinate, a look that moves away from the grey of the same luminance
/// by `saturation` (an affine combination of lights), the outset map, and the display's
/// 2.2 gamma.
pub fn agx<T: Real>(l: Light<T>, saturation: T) -> Light<T> {
    let (min_ev, max_ev) = (T::from_f64(-12.47393), T::from_f64(4.026069));
    let inset = colour_map::<T>([
        [0.842479062253094, 0.0784335999999992, 0.0792237451477643],
        [0.0423282422610123, 0.878468636469772, 0.0791661274605434],
        [0.0423756549057051, 0.0784336, 0.879142973793104],
    ]);
    // The outset is the inverse of the inset (folded to constants when traced).
    let outset = inset.inverse();
    let log2 = T::from_f64(core::f64::consts::LOG2_E);
    let v = each(inset.of(l), |x| {
        let e = (x.max(T::from_f64(1e-10)).ln() * log2)
            .max(min_ev)
            .min(max_ev);
        let x = (e - min_ev) / (max_ev - min_ev);
        // The contrast curve.
        let c = |k: f64| T::from_f64(k);
        let (x2, x4) = (x * x, x * x * x * x);
        c(15.5) * x4 * x2 - c(40.14) * x4 * x + c(31.96) * x4 - c(6.868) * x2 * x
            + c(0.4298) * x2
            + c(0.1191) * x
            - c(0.00232)
    });
    // The look: away from the grey light of the same luminance.
    let y = luma(v);
    let grey = Light::new(y, y, y, v.e123());
    let v = light_mix(grey, v, saturation);
    let v = outset.of(v);
    each(v, |x| {
        (T::from_f64(2.2) * x.max(T::from_f64(1e-10)).ln()).exp()
    })
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

/// A phasor's height: `(1, 0)` turned by `phase` (a sine in time).
pub fn wave<T: Real>(phase: T) -> T {
    let o = Point::xy(T::zero(), T::zero());
    (gax::pga2d::Motor::rotation(o, phase) >> Point::direction(T::one(), T::zero())).e01()
}
