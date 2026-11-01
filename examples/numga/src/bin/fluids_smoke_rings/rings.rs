//! The math: the smoke ring flow's velocity, the vector area, the Biot–Savart velocity of
//! polygonal filaments, Runge–Kutta, Hoffmann's Bäcklund transformation and its periodic offsets
//! (the lap's Lorentz rotor), the turns along a curve; and the scenes' curves.

use core::f64::consts::{PI, TAU};
use gax::Unit;
use gax::vga3d::{Bivector, Multivector, Paravector, Pseudoscalar, Rotor, Vector, intrinsic};
use gax_numga_examples::signal::wave;

/// Points, and their differences, the edges and offsets: antivectors, intrinsic.
pub type Point = intrinsic::Bivector;
/// A polygon's vector area, the join of its points: an intrinsic vector, a patch.
pub type Patch = intrinsic::Vector;
/// The filament's circulation, and the Bäcklund transformation's parameter: intrinsic scalars,
/// changing sign under a mirror.
pub type Circulation = intrinsic::Scalar;

/// An intrinsic scalar of the given value: the value times the unit volume's metric dual.
pub fn intrinsic(value: f64) -> Circulation {
    Pseudoscalar::new(value).hodge()
}

/// The point a vector reaches: numga's metric `dual`.
pub fn point(v: Vector) -> Point {
    v.hodge()
}

/// The unit points along `x`, `y` and `z`.
pub fn x_point() -> Point {
    point(Vector::new(1.0, 0.0, 0.0))
}
pub fn y_point() -> Point {
    point(Vector::new(0.0, 1.0, 0.0))
}
pub fn z_point() -> Point {
    point(Vector::new(0.0, 0.0, 1.0))
}

/// The axis `z`, as a plane: its pairing with a point is the point's height.
pub fn axis() -> Vector {
    Vector::new(0.0, 0.0, 1.0)
}

/// The height of a point along the axis.
pub fn height(p: Point) -> f64 {
    (p & axis()).s()
}

/// The distance of a point from the axis.
pub fn radius(p: Point) -> f64 {
    (p - z_point() * height(p)).norm()
}

/// The cosine-like pairing of two points: numga's `dot`, `⟨a ~b⟩₀`.
pub fn dot(a: Point, b: Point) -> f64 {
    a.scalar_product(b.reverse()).s()
}

/// The edges of a closed polygon, edge `k` from point `k` to point `k + 1`.
pub fn edges(points: &[Point]) -> Vec<Point> {
    let n = points.len();
    (0..n).map(|k| points[(k + 1) % n] - points[k]).collect()
}

/// The sides of an open polyline.
pub fn sides(points: &[Point]) -> Vec<Point> {
    points.windows(2).map(|w| w[1] - w[0]).collect()
}

/// The smoke ring flow's velocity at each vertex of a closed filament of the given
/// circulation: the commutator of the two edges, the bivector of the turn, over their lengths
/// plus their pairing, the tangent of half the turning angle; twice that, times the
/// circulation, over the mean edge length. Square to both edges, so no edge changes length.
pub fn velocity(points: &[Point], circulation: Circulation) -> Vec<Point> {
    let outgoing = edges(points);
    let n = points.len();
    (0..n)
        .map(|k| {
            let (out, inc) = (outgoing[k], outgoing[(k + n - 1) % n]);
            let lengths = out.norm() * inc.norm();
            let step = (inc.commutator(out) * circulation) * (2.0 / (lengths + dot(out, inc)));
            step * (2.0 / (out.norm() + inc.norm()))
        })
        .collect()
}

/// The vector area of a closed polygon, twice over: each point joined with the next, summed.
pub fn impulse(points: &[Point]) -> Patch {
    let n = points.len();
    (0..n).map(|k| points[k] & points[(k + 1) % n]).sum()
}

/// The velocity at every vertex of closed polygonal vortex filaments, of the given circulations,
/// induced by every segment of every filament (Biot–Savart, the vortex filament of Rosenhead and
/// Moore): the commutator of the vertex's separation from the segment's middle with the segment,
/// over the softened cube of the separation.
pub fn induced(rings: &[Vec<Point>], circulations: &[f64], core_radius: f64) -> Vec<Vec<Point>> {
    let segments: Vec<(Point, Point, Circulation)> = rings
        .iter()
        .zip(circulations)
        .flat_map(|(ring, g)| {
            let strength = intrinsic(g / (2.0 * TAU));
            ring.iter()
                .zip(edges(ring))
                .map(move |(p, e)| (*p + e * 0.5, e, strength))
        })
        .collect();
    rings
        .iter()
        .map(|ring| {
            ring.iter()
                .map(|x| {
                    let pulls = segments.iter().map(|(middle, segment, strength)| {
                        let separation = *x - *middle;
                        let softened = separation.norm_squared() + core_radius * core_radius;
                        #[allow(clippy::disallowed_methods)] // Biot–Savart's softened cube
                        let cube = softened * softened.sqrt();
                        (separation.commutator(*segment) * *strength) * (1.0 / cube)
                    });
                    pulls.sum()
                })
                .collect()
        })
        .collect()
}

/// Polygons moved by a rate for a time.
fn moved(points: &[Vec<Point>], rate: &[Vec<Point>], time: f64) -> Vec<Vec<Point>> {
    let rings = points.iter().zip(rate);
    rings
        .map(|(p, r)| p.iter().zip(r).map(|(p, r)| *p + *r * time).collect())
        .collect()
}

/// One classical Runge–Kutta step of polygons with the given velocity.
pub fn runge_kutta(
    points: &[Vec<Point>],
    rate: &impl Fn(&[Vec<Point>]) -> Vec<Vec<Point>>,
    step: f64,
) -> Vec<Vec<Point>> {
    let first = rate(points);
    let second = rate(&moved(points, &first, step / 2.0));
    let third = rate(&moved(points, &second, step / 2.0));
    let fourth = rate(&moved(points, &third, step));
    let mut result = points.to_vec();
    for (r, ring) in result.iter_mut().enumerate() {
        for (k, p) in ring.iter_mut().enumerate() {
            let sum = first[r][k] + (second[r][k] + third[r][k]) * 2.0 + fourth[r][k];
            *p += sum * (step / 6.0);
        }
    }
    result
}

/// The polygons at every `stride`-th of `steps` Runge–Kutta steps, the start first.
pub fn flow(
    points: Vec<Vec<Point>>,
    rate: impl Fn(&[Vec<Point>]) -> Vec<Vec<Point>>,
    step: f64,
    steps: usize,
    stride: usize,
) -> Vec<Vec<Vec<Point>>> {
    let mut now = points;
    let mut kept = vec![now.clone()];
    for k in 1..=steps {
        now = runge_kutta(&now, &rate, step);
        if k % stride == 0 {
            kept.push(now.clone());
        }
    }
    kept
}

/// The next offset of a Bäcklund transformation across an edge: the current one turned about
/// the diagonal `edge - offset` of the quad between them, by the quaternion the diagonal and the
/// parameter make together. The quaternion enters twice, so its orientation does not matter.
pub fn fold(edge: Point, offset: Point, twist: Circulation) -> Point {
    let diagonal: Bivector = (edge - offset).with_orientation();
    let quaternion = diagonal.cast::<Rotor>() + Rotor::new(twist.s(), 0.0, 0.0, 0.0);
    quaternion.unit() >> offset
}

/// One Bäcklund transformation of a polyline with the given parameter, from the offset at its
/// first vertex: every later offset is the one before, folded across the side between them.
pub fn backlund(points: &[Point], start: Point, twist: Circulation) -> Vec<Point> {
    let mut offset = start;
    let mut result = vec![points[0] + offset];
    for (side, p) in sides(points).iter().zip(&points[1..]) {
        offset = fold(*side, offset, twist);
        result.push(*p + offset);
    }
    result
}

/// The fold across a unit edge as the Lorentz rotor of its offsets' paravectors: the boost
/// along the direction the edge's point stands for, `rapidity` (`S.dual_inverse()`), and the
/// turn about it by `angle`, `exp(-(rapidity S* + angle S) / 2)`, the exponential of a
/// paravector-plus-bivector multivector (the two generators commute, each the other times the
/// pseudoscalar, so it is the boost times the turn).
pub fn lorentz_rotor(edge: Point, rapidity: f64, angle: f64) -> Multivector {
    let boost = edge.hodge_inverse().cast::<Multivector>() * rapidity;
    // The turn's generator: the angle (intrinsic, a parameter's) times the edge (a point).
    let turn: Bivector = (edge * angle).with_orientation();
    ((boost + turn.cast::<Multivector>()) * -0.5).exp()
}

/// The fold's rapidity and turning angle for a parameter: `rapidity + i angle` is the logarithm
/// of `1 - 2 i / twist`.
pub fn fold_rapidity_and_angle(twist: f64) -> (f64, f64) {
    (
        (1.0 + 4.0 / (twist * twist)).ln() / 2.0,
        -(2.0 / twist).atan(),
    )
}

/// An offset carried by a Lorentz rotor: its null paravector `1 + v*` sandwiched, the vector
/// part over the scalar part, back to a point.
pub fn carried(rotor: Multivector, offset: Point) -> Point {
    let paravector = Paravector::new(1.0, 0.0, 0.0, 0.0) + offset.hodge().cast::<Paravector>();
    let image = rotor * paravector.cast::<Multivector>() * rotor.reverse();
    let scale = image.cast::<gax::vga3d::Scalar>().s();
    (image.cast::<Vector>() * (1.0 / scale))
        .hodge()
        .with_orientation()
}

/// The unit offset at the first vertex of a closed polygon of unit edges that comes round to
/// itself after a lap: minus the last edge, carried once around by the lap's Lorentz rotor,
/// which lands it on the lap's attracting fixed point.
pub fn periodic(points: &[Point], twist: f64) -> Point {
    let sides = edges(points);
    let (rapidity, angle) = fold_rapidity_and_angle(twist);
    // The first fold acts first: the lap is the product of the folds' rotors, the last on the
    // left.
    let lap = sides
        .iter()
        .rev()
        .map(|s| lorentz_rotor(*s, rapidity, angle))
        .reduce(|a, b| a * b)
        .expect("edges");
    carried(lap, -sides[sides.len() - 1])
}

/// The polygon closed: its first vertex again at the end.
pub fn closed(points: &[Point]) -> Vec<Point> {
    let mut c = points.to_vec();
    c.push(points[0]);
    c
}

/// One step of the discrete-time smoke ring flow of a closed polygon of unit edges: the
/// Bäcklund transformation with periodic offsets, relabelled by one vertex, a time of about half
/// the parameter along the flow.
pub fn advance(points: &[Point], twist: f64) -> Vec<Point> {
    let start = periodic(points, twist);
    backlund(&closed(points), start, intrinsic(twist))[1..].to_vec()
}

/// The turn at every inner vertex of a polyline of unit edges: the rotor from the incoming side
/// to the outgoing one.
pub fn turns(points: &[Point]) -> Vec<Unit<Rotor>> {
    let s = sides(points);
    s.windows(2)
        .map(|w| {
            let turn: Rotor = (w[1] * w[0].reverse()).with_orientation();
            turn.unit()
        })
        .collect()
}

/// The angle of each turn: the size of its logarithm.
pub fn turn_angles(points: &[Point]) -> Vec<f64> {
    turns(points).iter().map(|t| t.log().norm()).collect()
}

/// A regular polygon of `count` unit edges around the z axis: the unit point on x turned, over
/// the chord of a `count`-th of a turn.
pub fn ring(count: usize) -> Vec<Point> {
    let chord = 2.0 * wave(PI / count as f64);
    let xy = Bivector::new(0.0, 0.0, 1.0);
    (0..count)
        .map(|k| ((xy * (-TAU * k as f64 / count as f64 / 2.0)).exp() >> x_point()) * (1.0 / chord))
        .collect()
}

/// Regular polygons of `count` vertices and the given radius around the z axis, one at each
/// height.
pub fn coaxial_rings(radius: f64, count: usize, heights: &[f64]) -> Vec<Vec<Point>> {
    let chord = 2.0 * wave(PI / count as f64);
    heights
        .iter()
        .map(|h| {
            ring(count)
                .iter()
                .map(|p| *p * (radius * chord) + z_point() * *h)
                .collect()
        })
        .collect()
}

/// A closed polygon of `count` unit edges around the z axis, each edge tilted out of the plane by
/// `tilt` times the sine of `mode` times its angle around the axis: a ring with a standing wave.
pub fn wavy_ring(count: usize, mode: usize, tilt: f64) -> Vec<Point> {
    let (xy, yz) = (Bivector::new(0.0, 0.0, 1.0), Bivector::new(1.0, 0.0, 0.0));
    let mut points = vec![Point::zero()];
    for k in 0..count - 1 {
        let angle = TAU * k as f64 / count as f64;
        let lifted = (yz * (-tilt * wave(mode as f64 * angle) / 2.0)).exp() >> y_point();
        let step = (xy * (-angle / 2.0)).exp() >> lifted;
        points.push(points[k] + step);
    }
    points
}

/// A closed polygon of `count` points on a torus knot, winding `around_axis` times around the z
/// axis and `around_tube` times around the tube of the given radii.
pub fn torus_knot(
    around_axis: usize,
    around_tube: usize,
    major: f64,
    minor: f64,
    count: usize,
) -> Vec<Point> {
    let (xy, zx) = (Bivector::new(0.0, 0.0, 1.0), Bivector::new(0.0, 1.0, 0.0));
    (0..count)
        .map(|k| {
            let turn = TAU * k as f64 / count as f64;
            let tube = (zx * (-(around_tube as f64) * turn / 2.0)).exp() >> (x_point() * minor);
            (xy * (-(around_axis as f64) * turn / 2.0)).exp() >> (x_point() * major + tube)
        })
        .collect()
}

// --- scenes

pub const RING: usize = 60;
pub const WAVY: usize = 90;
pub const WAVY_MODE: usize = 3;
pub const WAVY_TILT: f64 = 0.4;
pub const KNOT_WINDINGS: (usize, usize) = (2, 3);
pub const KNOT_RADII: (f64, f64) = (6.0, 2.5);
pub const KNOT: usize = 240;
pub const SURFACE_STEP: f64 = 0.4;
pub const SURFACE_STEPS: usize = 750;
pub const SURFACE_STRIDE: usize = 10;
pub const KNOT_STEP: f64 = 1.0 / 30.0;
pub const KNOT_STEPS: usize = 3600;
pub const KNOT_STRIDE: usize = 40;
pub const DISCRETE: usize = 40;
pub const PARAMETER: f64 = 0.5;
pub const DISCRETE_STEPS: usize = 100;
pub const FROG_RADIUS: f64 = 5.0;
pub const FROG: usize = 48;
pub const FROG_HEIGHTS: [f64; 2] = [0.0, 2.5];
pub const FROG_CIRCULATIONS: [f64; 2] = [10.0, 10.0];
pub const FROG_CORE: f64 = 0.4;
pub const FROG_STEP: f64 = 0.4 / 3.0;
pub const FROG_STEPS: usize = 150;
pub const SOLITON: usize = 75;
pub const SOLITON_OFFSET_LENGTH: f64 = 4.0;
pub const SOLITON_PARAMETER: f64 = 2.0;
pub const SOLITON_LEAN: f64 = 1e-5;
pub const CARRY_PARAMETER: f64 = 1.0;
pub const SOLITON_STEPS: usize = 90;

/// The smoke ring flow's own circulation, one in its rescaled time.
pub fn circulation() -> Circulation {
    intrinsic(1.0)
}

/// The smoke ring flow's velocity at every vertex of each closed curve.
pub fn flying(curves: &[Vec<Point>]) -> Vec<Vec<Point>> {
    curves.iter().map(|c| velocity(c, circulation())).collect()
}

/// The flow's speed at every vertex of an open line: inside it as on a closed curve; at the two
/// ends, which the closing edge would bend, their neighbours'.
pub fn line_speeds(curve: &[Point]) -> Vec<f64> {
    let inner: Vec<f64> = velocity(curve, circulation())
        .iter()
        .map(|v| v.norm())
        .collect();
    let n = inner.len();
    let mut speeds = inner.clone();
    speeds[0] = inner[1];
    speeds[n - 1] = inner[n - 2];
    speeds
}

/// The velocity the leapfrogging rings induce at their vertices.
pub fn pushed(rings: &[Vec<Point>]) -> Vec<Vec<Point>> {
    induced(rings, &FROG_CIRCULATIONS, FROG_CORE)
}

/// A curve through the smoke ring flow, every `stride` steps.
pub fn sweep(points: Vec<Point>, step: f64, steps: usize, stride: usize) -> Vec<Vec<Point>> {
    let kept = flow(vec![points], flying, step, steps, stride);
    kept.into_iter().map(|mut c| c.remove(0)).collect()
}

/// The Hashimoto surface of the wavy ring: its curves through the flow.
pub fn wavy_surface() -> Vec<Vec<Point>> {
    sweep(
        wavy_ring(WAVY, WAVY_MODE, WAVY_TILT),
        SURFACE_STEP,
        SURFACE_STEPS,
        SURFACE_STRIDE,
    )
}

/// The trefoil of vortex through the flow.
pub fn knot() -> Vec<Vec<Point>> {
    let (a, b) = KNOT_WINDINGS;
    let (major, minor) = KNOT_RADII;
    sweep(
        torus_knot(a, b, major, minor, KNOT),
        KNOT_STEP,
        KNOT_STEPS,
        KNOT_STRIDE,
    )
}

/// The plain and the wavy ring through the discrete-time flow, one Bäcklund transformation per
/// step: `[steps + 1][rings]`, the wavy ring's sweeping a discrete Hashimoto surface.
pub fn discrete() -> Vec<[Vec<Point>; 2]> {
    let mut now = [ring(DISCRETE), wavy_ring(DISCRETE, WAVY_MODE, WAVY_TILT)];
    let mut kept = vec![now.clone()];
    for _ in 0..DISCRETE_STEPS {
        now = now.map(|c| advance(&c, PARAMETER));
        kept.push(now.clone());
    }
    kept
}

/// A loop made on a straight vortex line by one Bäcklund transformation, from an offset leaning
/// just off the line's forward direction, and carried along it by the discrete-time flow.
pub fn soliton() -> Vec<Vec<Point>> {
    let line: Vec<Point> = (0..SOLITON)
        .map(|k| x_point() * (k as f64 - SOLITON as f64 / 2.0))
        .collect();
    let zx = Bivector::new(0.0, 1.0, 0.0);
    let start = ((zx * SOLITON_LEAN).exp() >> x_point()) * SOLITON_OFFSET_LENGTH;
    let mut curve = backlund(&line, start, intrinsic(SOLITON_PARAMETER));
    let mut kept = vec![curve.clone()];
    for _ in 0..SOLITON_STEPS {
        // Offsets of unit length from minus the first side: on the straight end each vertex
        // moves onto the one before, and relabelled the end stays where it is.
        let moved = backlund(&curve, curve[0] - curve[1], intrinsic(CARRY_PARAMETER));
        let mut next = moved[1..].to_vec();
        // The line runs on straight past its last vertex.
        let n = next.len();
        next.push(next[n - 1] * 2.0 - next[n - 2]);
        curve = next;
        kept.push(curve.clone());
    }
    kept
}

/// Two coaxial vortex rings moving each other by the Biot–Savart law, through two cycles of
/// leapfrogging.
pub fn leapfrog() -> Vec<Vec<Vec<Point>>> {
    let rings = coaxial_rings(FROG_RADIUS, FROG, &FROG_HEIGHTS);
    flow(rings, pushed, FROG_STEP, FROG_STEPS, 1)
}

/// The mean height and radius of each ring at each time, and the times just before one ring
/// passes through the other.
pub fn passes(frogs: &[Vec<Vec<Point>>]) -> (Vec<[f64; 2]>, Vec<[f64; 2]>, Vec<usize>) {
    let mean = |ring: &[Point], f: fn(Point) -> f64| {
        ring.iter().map(|p| f(*p)).sum::<f64>() / ring.len() as f64
    };
    let heights: Vec<[f64; 2]> = frogs
        .iter()
        .map(|r| [mean(&r[0], height), mean(&r[1], height)])
        .collect();
    let radii = frogs
        .iter()
        .map(|r| [mean(&r[0], radius), mean(&r[1], radius)])
        .collect();
    let crossings = (0..heights.len() - 1)
        .filter(|t| {
            let sign = |h: [f64; 2]| (h[1] - h[0]).signum();
            sign(heights[*t]) != sign(heights[t + 1])
        })
        .collect();
    (heights, radii, crossings)
}
