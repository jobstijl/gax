//! numga's `geometry/cyclides`: ray tracing Dupin cyclides on the 3-sphere, in its conformal
//! model. A point of S³ is a unit vector `p` of R⁴, carried as the null vector `p + e` of the
//! algebra R(4,1) with `x, y, z, w` squaring to 1 and `e` to -1; spheres are vectors, points
//! their duals, and a quadric is a linear map from a point to its polar sphere.
//!
//! The eye sits at `w` and looks along `-x`. Turned towards a direction `d` it moves along a
//! great circle, `origin + linear(d) sin t + quadratic(d, d) (1 - cos t)`, and in the half angle
//! `u = tan(t/2)` that circle is a parabola whose bend is the antipode of the eye. Substituting
//! the parabola into a quadric's form gives a quartic in `u`, one form per power: built once per
//! frame as maps with open direction slots, filled per pixel, and solved for the nearest hit
//! along the whole circle.
//!
//! The surfaces start as a spherical cylinder (a tube about a great circle) and a spherical
//! cone. A conformal dilation bends the cylinder's core into a small circle: a torus when the
//! dilation keeps the spin about the core, a Dupin cyclide when it does not; a dilation of the
//! cone pulls its two vertices together into a spindle cyclide. The flat tracer's scenes are
//! built in the stereographic chart about the eye. The animation cycles through the scenes:
//! tori of growing tube, a torus rolling around a circle, a lopsided cyclide, spindles seen
//! from several sides, and the flat scenes, their parts turning about their vortex circles.

gax::algebra! {
    algebra s3 "The conformal model of the 3-sphere, R(4,1): x, y, z and w square to 1, e (numga's `e`) to -1.";
    basis ex = 1, ey = 1, ez = 1, ew = 1, ee = -1;
    kind Scalar = [1];
    kind Direction = [ex, ey, ez];
    versor Vector = [ex, ey, ez, ew, ee];
    kind Bivector = [exy, exz, exw, exe, eyz, eyw, eye, ezw, eze, ewe];
    kind Quadvector = [eyzwe, exzwe, exywe, exyze, exyzw];
    kind Pseudoscalar = [exyzwe];
    versor Even = [1, exy, exz, exw, exe, eyz, eyw, eye, ezw, eze, ewe, eyzwe, exzwe, exywe, exyze, exyzw];
}

use gax_light::{Light, fade, light, mix};
use gax_numga_examples::points::box_map;
use gax_numga_examples::{Align, Anim, Canvas, Point2, backdrop, palette, run};

// numga's scenes in full: the tests check every one, the animation shows a selection.
#[cfg_attr(not(test), allow(dead_code))]
mod cyclides {
    use super::s3::{Bivector, Direction, Even, Pseudoscalar, Quadvector, Scalar, Vector};

    /// A sphere (or a great sphere, or a point of S³ as a unit vector plus `e`): a vector.
    pub type Sphere = Vector<(), f64>;
    /// A point: the Hodge dual of a null vector (see [`point`]).
    pub type Point = Quadvector<(), f64>;
    /// A direction at the eye, in `x, y, z`.
    pub type Dir = Direction<(), f64>;
    pub type Biv = Bivector<(), f64>;
    /// A rotation of R⁴ or a conformal dilation: a unit even versor.
    pub type Motor = gax::Unit<Even<(), f64>>;
    /// A quadric: a point to its polar sphere.
    pub type Quadric = Vector<(Quadvector,), f64>;

    pub fn x() -> Sphere {
        Vector::new(1.0, 0.0, 0.0, 0.0, 0.0)
    }
    pub fn y() -> Sphere {
        Vector::new(0.0, 1.0, 0.0, 0.0, 0.0)
    }
    pub fn z() -> Sphere {
        Vector::new(0.0, 0.0, 1.0, 0.0, 0.0)
    }
    pub fn w() -> Sphere {
        Vector::new(0.0, 0.0, 0.0, 1.0, 0.0)
    }
    pub fn e() -> Sphere {
        Vector::new(0.0, 0.0, 0.0, 0.0, 1.0)
    }

    /// The scalar 1 as an even versor.
    fn one() -> Even<(), f64> {
        Scalar::new(1.0).cast::<Even>()
    }

    /// The identity motor.
    pub fn identity() -> Motor {
        Biv::zero().exp()
    }

    /// The open point slot.
    fn slot() -> Quadvector<(Quadvector,), f64> {
        Quadvector::slot()
    }

    /// The point of a null vector: its Hodge dual, the product with the pseudoscalar. numga's
    /// `dual` is the right Hodge dual, which uses the metric; gax's `dual` is the metric-free
    /// complement, which would flip the sign of the `e` part here (`e² = -1`), so the pairing
    /// `sphere ∨ point` would no longer be the inner product of the sphere with the null vector.
    pub fn point(v: Sphere) -> Point {
        v * Pseudoscalar::new(1.0)
    }

    /// The eye, at `w` on S³.
    pub fn origin() -> Point {
        point(w() + e())
    }

    /// The point opposite the eye.
    pub fn antipode() -> Point {
        point(e() - w())
    }

    /// The flat chart about the eye, the stereographic projection from its antipode: the eye's
    /// null vector is the chart's origin and the antipode's its point at infinity.
    /// Constructions of the flat conformal model written with these two draw the same surfaces
    /// on S³.
    pub fn chart_origin() -> Sphere {
        (w() + e()).gp(0.5)
    }

    pub fn chart_infinity() -> Sphere {
        e() - w()
    }

    /// The great circle of the eye turned towards a unit direction, as maps of the direction.
    pub struct Rays {
        /// Half the rotation's generator, `(d ∧ w) / 2`.
        pub rotation: Bivector<(Direction,), f64>,
        /// The circle's first order term, `2 [rotation, origin]`.
        pub linear: Quadvector<(Direction,), f64>,
        /// The second order term, `2 [rotation, linear]`.
        pub quadratic: Quadvector<(Direction, Direction), f64>,
        /// The parabola's bend in `u = tan(t/2)`: `origin (d | d) + 2 quadratic`, the antipode.
        pub bend: Quadvector<(Direction, Direction), f64>,
    }

    pub fn rays() -> Rays {
        let d = Direction::<(), f64>::slot();
        let metric = d.dot(d);
        let rotation = (d ^ w()).gp(0.5);
        let linear = rotation.commutator(origin()).gp(2.0);
        let quadratic = rotation.commutator(linear).gp(2.0);
        // Times `1 + u²`: origin + linear(d) 2u + bend(d, d) u².
        let bend = origin() * metric + quadratic.gp(2.0);
        Rays {
            rotation,
            linear,
            quadratic,
            bend,
        }
    }

    // --- shapes ---------------------------------------------------------------------------

    /// The dyad `s (s ∨ P)`: a rank-one quadric.
    fn dyad(s: Sphere) -> Quadric {
        s * (s & slot())
    }

    /// The points at angle `tube` from the great circle in the xy plane:
    /// `z² + w² = sin(tube)²`.
    pub fn cylinder(tube: f64) -> Quadric {
        dyad(z()) + dyad(w()) - dyad(e()).gp(tube.sin().powi(2))
    }

    /// The cone with vertex `z`, its axis towards `x` and half-opening `opening`; its geodesics
    /// from `z` meet again at `-z`, its second vertex.
    pub fn cone(opening: f64) -> Quadric {
        dyad(x()) + (dyad(z()) - dyad(e())).gp(opening.cos().powi(2))
    }

    /// The conformal map of S³ that pushes points towards `aim` (a point as a unit vector),
    /// keeping it and its antipode fixed. Seen from `aim` it is a uniform scaling.
    pub fn dilation(aim: Sphere, strength: f64) -> Motor {
        (aim ^ e()).gp(strength / 2.0).exp()
    }

    /// A quadric moved by a motor: pull the point back, apply, push the sphere out. A
    /// sandwich by an even element of R(4,1) is typed with every grade of its parity (an even
    /// element need not be a versor, even with `x ~x = 1`), so each side is cast back to its
    /// kind; for the products of exponentials used here the other grades vanish.
    pub fn moved(m: Motor, q: Quadric) -> Quadric {
        (m >> q.of((m << slot()).cast::<Quadvector>())).cast::<Vector>()
    }

    /// A sphere moved by a motor (cast back to a vector, as in [`moved`]).
    pub fn moved_sphere(m: Motor, s: Sphere) -> Sphere {
        (m >> s).cast::<Vector>()
    }

    /// The unit versor `exp(b angle)`.
    pub fn turn(b: Biv, angle: f64) -> Motor {
        b.gp(angle).exp()
    }

    /// Turn the point `toward` onto the point `ahead` rad in front of the eye, after a tilt
    /// about it.
    pub fn placement(toward: Sphere, ahead: f64, tilt: f64) -> Motor {
        let target = moved_sphere(turn(w() ^ x(), ahead / 2.0), w());
        (one() + target * toward).normalized() * turn(y() ^ w(), tilt / 2.0)
    }

    /// A rotation of R⁴ by the given angles in the planes xy, xz, xw, yz, yw and zw, in turn.
    pub fn orientation(angles: [f64; 6]) -> Motor {
        let planes = [
            x() ^ y(),
            x() ^ z(),
            x() ^ w(),
            y() ^ z(),
            y() ^ w(),
            z() ^ w(),
        ];
        planes
            .iter()
            .zip(angles)
            .fold(identity(), |r, (p, a)| r * turn(*p, a))
    }

    /// Turn the shape, carry its centre to `ahead` rad in front of the eye, then turn the eye
    /// onto it by a yaw and a pitch.
    pub fn view(centre: Sphere, angles: [f64; 6], ahead: f64, yaw: f64, pitch: f64) -> Motor {
        let turned = orientation(angles);
        let target = moved_sphere(turn(w() ^ x(), ahead / 2.0), w());
        let carry = (one() + target * moved_sphere(turned, centre)).normalized();
        let look = turn(x() ^ y(), yaw / 2.0) * turn(x() ^ z(), pitch / 2.0);
        look * carry * turned
    }

    // --- tracing --------------------------------------------------------------------------

    /// The real roots of `a[0] rⁿ + ... + a[n]` (degree up to four, coefficients descending),
    /// largest first, by isolation: the real roots of the derivative split the line into
    /// monotone pieces, and each piece with a change of sign holds one root, found by
    /// bisection. With `all` false it stops at the largest. numga takes every root of the
    /// companion matrix instead (NumPy's `eigvals`) and keeps the nearly real ones.
    pub fn real_roots(a: &[f64], all: bool) -> Vec<f64> {
        let scale = a.iter().fold(0.0f64, |m, c| m.max(c.abs()));
        let Some(first) = a.iter().position(|c| c.abs() > 1e-14 * scale) else {
            return vec![];
        };
        let a = &a[first..];
        let n = a.len() - 1;
        match n {
            0 => vec![],
            1 => vec![-a[1] / a[0]],
            2 => {
                let (p, q, r) = (a[0], a[1], a[2]);
                let disc = q * q - 4.0 * p * r;
                if disc < 0.0 {
                    return vec![];
                }
                // The stable form: no cancellation in either root.
                let s = -0.5 * (q + q.signum() * disc.sqrt());
                let (r1, r2) = if s == 0.0 { (0.0, 0.0) } else { (s / p, r / s) };
                vec![r1.max(r2), r1.min(r2)]
            }
            _ => {
                let eval = |r: f64| a.iter().fold(0.0, |acc, c| acc * r + c);
                // Every root lies within Cauchy's bound.
                let bound = 1.0 + a[1..].iter().fold(0.0f64, |m, c| m.max((c / a[0]).abs()));
                let derivative: Vec<f64> = a[..n]
                    .iter()
                    .enumerate()
                    .map(|(i, c)| c * (n - i) as f64)
                    .collect();
                let mut stops = vec![bound];
                stops.extend(real_roots(&derivative, true));
                stops.push(-bound);
                let mut roots = vec![];
                for pair in stops.windows(2) {
                    let (hi, lo) = (pair[0], pair[1]);
                    let (fh, fl) = (eval(hi), eval(lo));
                    if fh * fl <= 0.0 {
                        roots.push(bisect(eval, lo, hi, fl));
                        if !all {
                            break;
                        }
                    }
                }
                roots
            }
        }
    }

    /// The root of `f` in `[lo, hi]`, where `f` changes sign, by bisection to the last bit.
    fn bisect(f: impl Fn(f64) -> f64, mut lo: f64, mut hi: f64, mut flo: f64) -> f64 {
        for _ in 0..80 {
            let mid = 0.5 * (lo + hi);
            if mid <= lo || mid >= hi {
                break;
            }
            let fm = f(mid);
            if (fm <= 0.0) == (flo <= 0.0) {
                lo = mid;
                flo = fm;
            } else {
                hi = mid;
            }
        }
        0.5 * (lo + hi)
    }

    /// The nearest hit's angle along the great circle, from the ascending coefficients of the
    /// quartic in `u = tan(t/2)`. The reversed polynomial is solved, for `r = 1/u`: a root maps
    /// to `t = 2 atan2(1, r)` in `(0, 2π)`, decreasing in `r`, so hits past the antipode are
    /// ordered along the whole circle and the nearest hit is the largest root.
    pub fn nearest_angle(c: [f64; 5]) -> Option<f64> {
        real_roots(&c, false)
            .first()
            .map(|r| 2.0 * 1.0f64.atan2(*r))
    }

    /// One surface with the forms of its quartic: `form(X, Y) = X ∨ surface(Y)` along
    /// `X = origin + linear(d) 2u + bend(d, d) u²`, one form per power of `u`, each a map of the
    /// pixel direction.
    pub struct Traced {
        pub surface: Quadric,
        constant: f64,
        linear: Scalar<(Direction,), f64>,
        quadratic: Scalar<(Direction, Direction), f64>,
        cubic: Scalar<(Direction, Direction, Direction), f64>,
        quartic: Scalar<(Direction, Direction, Direction, Direction), f64>,
    }

    /// A hit: the angle along the great circle and the headlight facing, the cosine between
    /// the ray and the surface normal.
    #[derive(Clone, Copy, Debug)]
    pub struct Hit {
        pub angle: f64,
        pub facing: f64,
    }

    impl Traced {
        pub fn new(surface: Quadric, rays: &Rays) -> Traced {
            let o = origin();
            let (linear, bend) = (rays.linear, rays.bend);
            Traced {
                surface,
                constant: (o & surface.of(o)).s(),
                linear: (o & surface.of(linear)).gp(4.0),
                quadratic: (linear & surface.of(linear)).gp(4.0) + (o & surface.of(bend)).gp(2.0),
                cubic: (linear & surface.of(bend)).gp(4.0),
                quartic: bend & surface.of(bend),
            }
        }

        /// The ascending coefficients of the quartic in `u` for a unit direction.
        pub fn coefficients(&self, d: Dir) -> [f64; 5] {
            [
                self.constant,
                self.linear.of(d).s(),
                self.quadratic.fill(d).s(),
                self.cubic.fill(d).s(),
                self.quartic.fill(d).s(),
            ]
        }

        /// The nearest hit along the ray of a unit direction, if any.
        pub fn hit(&self, rays: &Rays, d: Dir) -> Option<Hit> {
            let t = nearest_angle(self.coefficients(d))?;
            let (l, q) = (rays.linear.of(d), rays.quadratic.fill(d));
            let at = origin() + l.gp(t.sin()) + q.gp(1.0 - t.cos());
            let velocity = l.gp(t.cos()) + q.gp(t.sin());
            // At the hit the polar sphere is the tangent sphere; its length is that of the
            // surface's gradient.
            let polar = self.surface.of(at);
            let facing = -(polar & velocity).s() / polar.dot(polar).s().abs().sqrt();
            Some(Hit { angle: t, facing })
        }
    }

    /// The unit direction of a pinhole pixel looking along `-x`: `u` across from -1 to 1, `v`
    /// up, scaled by the aspect.
    pub fn direction(u: f64, v: f64, fov: f64) -> Dir {
        let k = (fov / 2.0).tan();
        Direction::new(-1.0, u * k, v * k).normalized().into_inner()
    }

    /// numga's sensor: the pixel centres of a `rows` x `cols` image, row-major.
    pub fn sensor(rows: usize, cols: usize, fov: f64) -> Vec<Dir> {
        let mut out = Vec::with_capacity(rows * cols);
        for i in 0..rows {
            for j in 0..cols {
                let u = 2.0 * (j as f64 + 0.5) / cols as f64 - 1.0;
                let v = (1.0 - 2.0 * (i as f64 + 0.5) / rows as f64) * rows as f64 / cols as f64;
                out.push(direction(u, v, fov));
            }
        }
        out
    }

    /// numga's image shape and field of view.
    pub const SHAPE: (usize, usize) = (240, 320);
    pub const FOV: f64 = core::f64::consts::FRAC_PI_2;

    // --- scenes -----------------------------------------------------------------------------

    /// A torus: the cylinder of tube `tube`, dilated towards z (a quarter turn from its whole
    /// core) so the core shrinks evenly into a circle. Seen from z the dilation is a uniform
    /// scaling, so it only sizes the torus; the tube sets its shape.
    pub fn torus_on_s3(tube: f64, strength: f64) -> Quadric {
        let place = placement(z(), 1.1, 0.8) * dilation(z(), strength);
        moved(place, cylinder(tube))
    }

    /// numga's three tori: tubes of 0.15, 0.5 and 0.9 rad.
    pub fn tori() -> [Quadric; 3] {
        [(0.15, 1.55), (0.5, 2.0), (0.9, 2.4)].map(|(t, s)| torus_on_s3(t, s))
    }

    /// The middle torus carried around a circle, at phase `t`: its own core circle tilted by
    /// 0.4 rad in the yw plane. Around the core itself the flow would only spin the tube in
    /// place; tilted, it rolls and deforms the torus.
    pub fn vortex(t: f64) -> Quadric {
        let place = placement(z(), 1.1, 0.8) * dilation(z(), 2.0);
        let torus = moved(place, cylinder(0.5));
        // The meet of two great spheres carried by a unit versor: a unit circle,
        // `circle * circle == -1`.
        let circle = ((place * turn(y() ^ w(), 0.2)) >> (z() ^ w())).cast::<Bivector>();
        moved(turn(circle, t / 2.0), torus)
    }

    /// A lopsided Dupin cyclide: a dilation aimed off z, leaning by `lean` towards the core point
    /// x, compresses the tube on the side of the aim and swells it on the other.
    pub fn dupin(lean: f64, strength: f64, tube: f64, ahead: f64, tilt: f64) -> Quadric {
        let aim = moved_sphere(turn(z() ^ x(), -lean / 2.0), z());
        let bent = moved(dilation(aim, strength), cylinder(tube));
        moved(placement(aim, ahead, tilt), bent)
    }

    /// numga's three Dupin cyclides.
    pub fn dupins() -> [Quadric; 3] {
        [
            (0.7, 1.5, 0.25, 1.2, -0.7),
            (0.8, 1.2, 0.3, 1.3, 0.6),
            (0.5, 1.8, 0.2, 1.1, 1.0),
        ]
        .map(|(l, s, t, a, ti)| dupin(l, s, t, a, ti))
    }

    /// The cone of half-opening 0.35 dilated strongly towards its axis point, leaning by
    /// `lean` towards y: its two vertices close into a spindle cyclide, and the lean bends the
    /// inner sheet into a banana. Also the midpoint of its dilated vertices, where the eye aims.
    pub fn spindle(lean: f64) -> (Quadric, Sphere) {
        let dil = dilation(moved_sphere(turn(x() ^ y(), -lean / 2.0), x()), 3.0);
        let bent = moved(dil, cone(0.35));
        // The vertices, z and -z carried by the dilation, are null vectors; scaled to one unit
        // of e each, their sum less its e part, normalized, is the midpoint as a unit vector.
        let unit_e = |v: Sphere| v.gp(-1.0 / (v | e()).s());
        let a = unit_e(moved_sphere(dil, z() + e()));
        let b = unit_e(moved_sphere(dil, e() - z()));
        (bent, (a + b - e().gp(2.0)).normalized().into_inner())
    }

    /// One of numga's spindle shots: which spindle (lean 0 or 0.7), its orientation in the
    /// six planes, the distance, and the eye's yaw and pitch.
    pub type SpindleShot = (usize, [f64; 6], f64, f64, f64);

    pub const SHOTS: [SpindleShot; 5] = [
        (
            0,
            [-0.186, -0.692, 2.658, 0.181, -0.282, -0.225],
            1.10,
            -0.021,
            0.061,
        ),
        (
            0,
            [1.124, 0.598, 0.155, 0.889, -0.164, -0.741],
            1.05,
            -0.051,
            -0.062,
        ),
        (
            0,
            [0.464, 0.073, 0.536, -2.263, 0.817, -0.768],
            1.05,
            -0.001,
            0.060,
        ),
        (
            1,
            [-0.257, 0.18, 0.46, -0.999, -1.384, -0.004],
            0.85,
            0.137,
            -0.377,
        ),
        (
            1,
            [-0.511, -0.64, -0.64, 1.096, -1.168, -0.477],
            0.85,
            -0.297,
            -0.358,
        ),
    ];

    /// A spindle seen in a shot.
    pub fn spindle_shot(lean: f64, angles: [f64; 6], ahead: f64, yaw: f64, pitch: f64) -> Quadric {
        let (bent, midpoint) = spindle(lean);
        moved(view(midpoint, angles, ahead, yaw, pitch), bent)
    }

    /// numga's five spindle shots.
    pub fn spindles() -> Vec<Quadric> {
        SHOTS
            .iter()
            .map(|&(i, angles, ahead, yaw, pitch)| {
                spindle_shot([0.0, 0.7][i], angles, ahead, yaw, pitch)
            })
            .collect()
    }

    // --- the flat tracer's scenes ---------------------------------------------------------

    /// A part of a flat scene: a surface, and the circle it turns about (zero for none).
    pub type Part = (Quadric, Biv);

    /// A quadric moved by a unit vector, an inversion or a reflection.
    fn inverted(v: gax::Unit<Vector<(), f64>>, q: Quadric) -> Quadric {
        v >> q.of(v << slot())
    }

    /// The translation of the chart by `t`: `exp(-(t ∧ ∞) / 2)`.
    fn shift(t: Sphere) -> Motor {
        (t ^ chart_infinity()).gp(-0.5).exp()
    }

    /// The chart's sphere about its origin of squared radius `r2`.
    fn chart_sphere(r2: f64) -> Sphere {
        chart_origin() - chart_infinity().gp(r2 / 2.0)
    }

    /// The chart's torus of core radius `radius` and tube `tube` about z.
    pub fn torus(radius: f64, tube: f64) -> Quadric {
        let sphere = chart_sphere(radius * radius + tube * tube);
        dyad(sphere) + (dyad(z()) - dyad(chart_infinity()).gp(tube * tube)).gp(radius * radius)
    }

    fn still(q: Quadric) -> Vec<Part> {
        vec![(q, Biv::zero())]
    }

    pub fn cyclide() -> Vec<Part> {
        // An off-centre sphere inversion gives the torus unequal tube widths.
        let inversion = moved_sphere(shift(x().gp(3.2)), chart_sphere(9.0)).normalized();
        let pose = turn(x() ^ y(), 0.12) * turn(y() ^ z(), -0.22);
        still(moved(pose, inverted(inversion, torus(1.4, 0.5))))
    }

    pub fn peanut() -> Vec<Part> {
        let (a, b) = (1.25f64, 1.31f64);
        let sphere = chart_sphere(a * a);
        let surface = dyad(sphere) + (dyad(y()) + dyad(z())).gp(a * a)
            - dyad(chart_infinity()).gp(b.powi(4) / 4.0);
        still(moved(turn(x() ^ y(), 0.10), surface))
    }

    fn ring(squash: f64) -> Vec<Part> {
        let surface = torus(1.4, 0.5) + dyad(y()).gp(squash * 1.4 * 1.4);
        still(moved(turn(y() ^ z(), -0.18), surface))
    }

    pub fn elliptic_ring() -> Vec<Part> {
        ring(0.10)
    }

    pub fn split_ring() -> Vec<Part> {
        ring(0.32)
    }

    /// The hyperboloid `z² - x² - y² + 1`, closed at a singular point by an inversion.
    fn hyperboloid_of_one_sheet() -> Quadric {
        dyad(z()) - dyad(x()) - dyad(y()) + dyad(chart_infinity())
    }

    pub fn pinched_surface() -> Quadric {
        inverted(
            (chart_origin() - chart_infinity()).normalized(),
            hyperboloid_of_one_sheet(),
        )
    }

    pub fn pinched() -> Vec<Part> {
        still(moved(turn(y() ^ z(), -0.18), pinched_surface()))
    }

    /// The paper's diagonal sphere-model quadrics, as sums of dyads.
    fn sphere_family(weights: [f64; 4]) -> Quadric {
        let spheres = [x(), y(), z(), chart_sphere(1.0)];
        spheres
            .iter()
            .zip(weights)
            .fold(Quadric::zero(), |q, (s, k)| q + dyad(*s).gp(k))
    }

    pub fn six_families() -> Vec<Part> {
        let pose = ((x() ^ y()).gp(0.25) + (y() ^ z()).gp(-0.15)).exp();
        still(moved(pose, sphere_family([-2.0, -1.0, 1.0, 2.0])))
    }

    /// The inverted hyperboloid, with its sign reversed since this inversion's centre lies
    /// outside the solid, and the inversion.
    fn inverted_hyperboloid() -> (Quadric, gax::Unit<Vector<(), f64>>) {
        let surface = hyperboloid_of_one_sheet() - dyad(y()).gp(1.0 / (0.65 * 0.65) - 1.0);
        let inversion =
            moved_sphere(shift(x().gp(1.7)), chart_origin() - chart_infinity()).normalized();
        (-inverted(inversion, surface), inversion)
    }

    pub fn hyperboloid() -> Vec<Part> {
        still(inverted_hyperboloid().0)
    }

    pub fn two_lobed() -> Vec<Part> {
        let pose = ((x() ^ y()).gp(0.12) + (y() ^ z()).gp(-0.12)).exp();
        still(moved(pose, sphere_family([-3.0, -0.25, 1.0, 1.0])))
    }

    /// The core circle of radius `radius` about z, in the plane z = 0, as a unit bivector.
    fn core_circle(radius: f64) -> Biv {
        (z() ^ chart_sphere(radius * radius))
            .normalized()
            .into_inner()
    }

    pub fn linked_vortex() -> Vec<Part> {
        let (surface, inversion) = inverted_hyperboloid();
        // Invert a circle about the uninverted hyperboloid's waist together with the body:
        // the vortex circle threads its opening transversely.
        let sphere = chart_sphere(2.5 * 2.5);
        let circle = (inversion >> (z() ^ sphere)).normalized().into_inner();
        // The inverted circle still lies in z = 0. Normalize its sphere to recover its radius,
        // then give it a thin tube of constant thickness.
        let sphere = inversion >> sphere;
        let sphere = sphere.gp(-1.0 / (sphere | chart_infinity()).s());
        let radius_squared = sphere.dot(sphere).s();
        let tube = 0.02;
        let sphere = sphere - chart_infinity().gp(tube * tube / 2.0);
        let ring =
            dyad(sphere) + (dyad(z()) - dyad(chart_infinity()).gp(tube * tube)).gp(radius_squared);
        vec![(surface, circle), (ring, Biv::zero())]
    }

    pub fn linked_tori() -> Vec<Part> {
        // Matching core radii, with a small shift along the tilt axis; matching the shift to
        // `radius sin(tilt)` keeps the initial gap fairly even.
        let (radius, offset) = (1.4f64, 0.45f64);
        let place = shift(x().gp(offset)) * turn(y() ^ z(), (offset / radius).asin() / 2.0);
        vec![
            (moved(place, torus(radius, 0.16)), core_circle(radius)),
            (torus(radius, 0.02), Biv::zero()),
        ]
    }

    pub fn pinched_vortex() -> Vec<Part> {
        let place = shift(x().gp(1.8)) * turn(x() ^ z(), core::f64::consts::FRAC_PI_4);
        vec![
            (moved(place, pinched_surface()), core_circle(1.4)),
            (torus(1.4, 0.02), Biv::zero()),
        ]
    }

    /// A flat scene: its name, its parts, the camera's position and target in the chart, and
    /// the field of view in degrees.
    pub struct Flat {
        pub name: &'static str,
        pub build: fn() -> Vec<Part>,
        pub position: [f64; 3],
        pub target: [f64; 3],
        pub degrees: f64,
    }

    pub const FLAT_SCENES: [Flat; 11] = [
        Flat {
            name: "cyclide",
            build: cyclide,
            position: [0.2, -8.5, 5.8],
            target: [-0.8, 0.0, 0.0],
            degrees: 43.0,
        },
        Flat {
            name: "peanut",
            build: peanut,
            position: [0.2, -7.0, 4.5],
            target: [0.0, 0.0, 0.0],
            degrees: 40.0,
        },
        Flat {
            name: "elliptic ring",
            build: elliptic_ring,
            position: [0.2, -7.0, 4.5],
            target: [0.0, 0.0, 0.0],
            degrees: 40.0,
        },
        Flat {
            name: "split ring",
            build: split_ring,
            position: [0.2, -7.0, 4.5],
            target: [0.0, 0.0, 0.0],
            degrees: 40.0,
        },
        Flat {
            name: "pinched",
            build: pinched,
            position: [0.2, -7.0, 4.5],
            target: [0.0, 0.0, 0.0],
            degrees: 40.0,
        },
        Flat {
            name: "six families",
            build: six_families,
            position: [0.6, -9.0, 6.5],
            target: [0.0, 0.0, 0.0],
            degrees: 40.0,
        },
        Flat {
            name: "hyperboloid",
            build: hyperboloid,
            position: [10.0, -2.0, 1.8],
            target: [0.6, 0.0, 0.0],
            degrees: 40.0,
        },
        Flat {
            name: "two lobed",
            build: two_lobed,
            position: [0.5, -12.0, 8.0],
            target: [0.0, 0.0, 0.0],
            degrees: 40.0,
        },
        Flat {
            name: "linked vortex",
            build: linked_vortex,
            position: [2.8, -9.0, 4.8],
            target: [1.0, 0.0, 0.0],
            degrees: 44.0,
        },
        Flat {
            name: "linked tori",
            build: linked_tori,
            position: [0.5, -7.0, 4.8],
            target: [0.0, 0.0, 0.0],
            degrees: 40.0,
        },
        Flat {
            name: "pinched vortex",
            build: pinched_vortex,
            position: [0.5, -10.0, 7.0],
            target: [0.5, 0.0, 0.0],
            degrees: 44.0,
        },
    ];

    pub fn chart_point(p: [f64; 3]) -> Sphere {
        Vector::new(p[0], p[1], p[2], 0.0, 0.0)
    }

    /// The motor that brings the flat tracer's camera to the eye. Its pose carries a camera at
    /// the origin looking along -x with z up to `position` looking at `target`, and the eye of
    /// S³ sits at the chart's origin looking along -x. Great circles through the eye are the
    /// chart's straight lines through its origin, so the view is the flat tracer's.
    pub fn flat_camera(position: [f64; 3], target: [f64; 3]) -> Motor {
        let (position, target) = (chart_point(position), chart_point(target));
        let forward = (target - position).normalized().into_inner();
        // `(forward ∧ z) xyz⁻¹`, the cross product with z (numga computes it with the
        // trivector; this algebra declares no trivector kind).
        let right = Vector::new(forward.c[1], -forward.c[0], 0.0, 0.0, 0.0)
            .normalized()
            .into_inner();
        let aim = (one() - forward * x()).normalized();
        let roll = (one() + right * moved_sphere(aim, y())).normalized();
        (shift(position) * roll * aim).inverse()
    }

    /// A flat scene at phase `t` on S³: every part turned about its own vortex circle, all
    /// moved to the eye. Also the field of view in radians.
    pub fn flat_scene(scene: &Flat, t: f64) -> (Vec<Quadric>, f64) {
        let camera = flat_camera(scene.position, scene.target);
        flat_scene_from(scene, camera, t)
    }

    /// The same from another camera motor.
    pub fn flat_scene_from(scene: &Flat, camera: Motor, t: f64) -> (Vec<Quadric>, f64) {
        let surfaces = (scene.build)()
            .into_iter()
            .map(|(surface, circle)| moved(camera * turn(circle, t / 2.0), surface))
            .collect();
        (surfaces, scene.degrees.to_radians())
    }
}

use cyclides::*;

/// numga's headlight colours, in linear light: teal for a single surface, one colour per body
/// in the flat scenes.
const TEAL: Light = light(0.055, 0.42, 0.39, 1.0);
const BODIES: [Light; 4] = [
    TEAL,
    light(0.65, 0.38, 0.08, 1.0),
    light(0.35, 0.13, 0.40, 1.0),
    light(0.12, 0.35, 0.60, 1.0),
];

/// Seconds per shot.
const SHOT: f64 = 3.0;

/// One shot of the animation: its title, and its surfaces and field of view at `s` in `[0, 1)`.
struct Shot {
    title: &'static str,
    note: &'static str,
    /// Whether both sides are lit (the unsigned cosine), as numga does for the spindles and
    /// the flat scenes.
    both_sides: bool,
    scene: Scene,
}

/// What a shot shows.
enum Scene {
    /// Surfaces and a field of view at `s` in `[0, 1)`.
    Built(fn(f64) -> (Vec<Quadric>, f64)),
    /// One of the flat tracer's scenes, by index.
    Flat(usize),
}

/// Eased there and back over `s` in `[0, 1)`: 0, up to 1 and back to 0.
fn there_and_back(s: f64) -> f64 {
    0.5 - 0.5 * (core::f64::consts::TAU * s).cos()
}

fn lerp_shot(a: &SpindleShot, b: &SpindleShot, k: f64) -> ([f64; 6], f64, f64, f64) {
    let l = |x: f64, y: f64| x + (y - x) * k;
    (
        core::array::from_fn(|i| l(a.1[i], b.1[i])),
        l(a.2, b.2),
        l(a.3, b.3),
        l(a.4, b.4),
    )
}

fn shots() -> Vec<Shot> {
    let mut list = vec![
        Shot {
            title: "TORI",
            note: "A TUBE ABOUT A GREAT CIRCLE, DILATED: TUBE 0.15 TO 0.9 RAD",
            both_sides: false,
            scene: Scene::Built(|s| {
                let k = there_and_back(s);
                (vec![torus_on_s3(0.15 + 0.75 * k, 1.55 + 0.85 * k)], FOV)
            }),
        },
        Shot {
            title: "VORTEX",
            note: "THE TORUS CARRIED AROUND ITS TILTED CORE CIRCLE",
            both_sides: false,
            scene: Scene::Built(|s| (vec![vortex(core::f64::consts::TAU * s)], FOV)),
        },
        Shot {
            title: "DUPIN CYCLIDE",
            note: "THE DILATION AIMED OFF THE CORE: THE TUBE GOES LOPSIDED",
            both_sides: false,
            scene: Scene::Built(|s| {
                (
                    vec![dupin(0.8 * there_and_back(s), 1.5, 0.25, 1.2, -0.7)],
                    FOV,
                )
            }),
        },
        Shot {
            title: "SPINDLE CYCLIDE",
            note: "A CONE DILATED UNTIL ITS VERTICES CLOSE, FROM THREE SIDES",
            both_sides: true,
            scene: Scene::Built(|s| {
                // Around numga's three straight shots, and back to the first.
                let k = 3.0 * s;
                let i = (k as usize).min(2);
                let (angles, ahead, yaw, pitch) = lerp_shot(
                    &SHOTS[i],
                    &SHOTS[(i + 1) % 3],
                    0.5 - 0.5 * (core::f64::consts::PI * (k - i as f64)).cos(),
                );
                (vec![spindle_shot(0.0, angles, ahead, yaw, pitch)], FOV)
            }),
        },
        Shot {
            title: "BANANA SPINDLE",
            note: "THE DILATION LEANING TOWARDS Y BENDS THE INNER SHEET",
            both_sides: true,
            scene: Scene::Built(|s| {
                let (angles, ahead, yaw, pitch) =
                    lerp_shot(&SHOTS[3], &SHOTS[4], there_and_back(s));
                (vec![spindle_shot(0.7, angles, ahead, yaw, pitch)], FOV)
            }),
        },
    ];
    for (i, scene) in FLAT_SCENES.iter().enumerate() {
        let animated = (scene.build)()
            .iter()
            .any(|(_, c)| c.c.iter().any(|v| *v != 0.0));
        list.push(Shot {
            title: scene.name,
            note: if animated {
                "FLAT SCENE ON S3: PARTS TURN ABOUT THEIR VORTEX CIRCLES"
            } else {
                "FLAT SCENE ON S3, BUILT IN THE CHART ABOUT THE EYE"
            },
            both_sides: true,
            scene: Scene::Flat(i),
        });
    }
    list
}

/// A flat scene at `s`: animated scenes turn their parts once about their circles; still ones
/// swing the camera about the chart's z axis and back.
fn flat_at(index: usize, s: f64) -> (Vec<Quadric>, f64) {
    let scene = &FLAT_SCENES[index];
    let animated = (scene.build)()
        .iter()
        .any(|(_, c)| c.c.iter().any(|v| *v != 0.0));
    if animated {
        flat_scene(scene, core::f64::consts::TAU * s)
    } else {
        // The camera's position turned about the vertical through the target: the offset
        // from the target turned in the chart's xy plane.
        let swing = 0.35 * (core::f64::consts::TAU * s).sin();
        let (p, t) = (chart_point(scene.position), chart_point(scene.target));
        let turned = t + moved_sphere(turn(x() ^ y(), -swing / 2.0), p - t);
        let position = [0, 1, 2].map(|i| turned.c[i]);
        flat_scene_from(scene, flat_camera(position, scene.target), 0.0)
    }
}

/// The surface colour lit from the eye, numga's headlight: `facing` is the cosine between the
/// ray and the normal. The diffuse light and a white glint where the surface faces the eye
/// add up.
fn headlight(facing: f64, colour: Light) -> Light {
    let f = facing.clamp(0.0, 1.0) as f32;
    fade(colour, 0.2 + 0.8 * f) + light(1.0, 1.0, 1.0, 0.12 * f.powi(16))
}

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let list = shots();
    let total = SHOT * list.len() as f64;
    let time = f64::from(t).rem_euclid(total);
    let index = ((time / SHOT) as usize).min(list.len() - 1);
    let local = time - SHOT * index as f64;
    let shot = &list[index];
    let (surfaces, fov) = match shot.scene {
        Scene::Flat(i) => flat_at(i, local / SHOT),
        Scene::Built(f) => f(local / SHOT),
    };
    let rays = rays();
    let traced: Vec<Traced> = surfaces
        .into_iter()
        .map(|q| Traced::new(q, &rays))
        .collect();
    let screen = c.rect();
    let single = matches!(shot.scene, Scene::Built(_));
    // numga's pinhole sensor: the canvas onto `u` across from -1 to 1, `v` up, scaled by the
    // aspect.
    let aspect = screen.height() / screen.width();
    let sensor = box_map(
        [screen.lo, screen.hi],
        [Point2::xy(-1.0, aspect), Point2::xy(1.0, -aspect)],
    );
    c.shade(1, |q: Point2| {
        let [u, v] = sensor.of(q).to_euclidean();
        let d = direction(f64::from(u), f64::from(v), fov);
        let (k, hit) = traced
            .iter()
            .enumerate()
            .filter_map(|(k, s)| s.hit(&rays, d).map(|hit| (k, hit)))
            .min_by(|a, b| a.1.angle.total_cmp(&b.1.angle))?;
        let facing = if shot.both_sides {
            hit.facing.abs()
        } else {
            hit.facing
        };
        let colour = if single {
            TEAL
        } else {
            BODIES[k % BODIES.len()]
        };
        Some(headlight(facing, colour))
    });
    // A short fade between shots: the backdrop's bottom drawn over everything.
    let edge = local.min(SHOT - local);
    let veil = (1.0 - edge / 0.25).clamp(0.0, 1.0) as f32;
    if veil > 0.0 {
        let corners = [
            screen.lo,
            screen.top_right(),
            screen.hi,
            screen.bottom_left(),
        ];
        c.fill(&corners, palette::bottom(), veil);
    }
    let title = format!(
        "DUPIN CYCLIDES ON THE 3-SPHERE: {}",
        shot.title.to_uppercase()
    );
    gax_numga_examples::caption(c, &title, shot.note);
    // The shot's number in the bottom right corner.
    let number = format!("{}/{}", index + 1, list.len());
    let corner = screen.hi + Point2::direction(-8.0, -8.0);
    let dim = mix(palette::ink(), palette::bottom(), 0.4);
    c.text(&number, corner, 10.0, dim, Align::Right);
}

fn main() {
    let seconds = SHOT as f32 * shots().len() as f32;
    run(Anim::new("cyclides", seconds).size(480, 360).scale(2), draw);
}

#[cfg(test)]
mod tests {
    use super::cyclides::*;
    use super::s3::{Direction, Quadvector};

    fn dir(x: f64, y: f64, z: f64) -> Dir {
        Direction::new(x, y, z)
    }

    fn probes() -> [Sphere; 5] {
        [x(), y(), z(), w(), e()]
    }

    /// Whether any of numga's sensor pixels sees the surface.
    fn any_hit(surface: Quadric, fov: f64) -> bool {
        let rays = rays();
        let traced = Traced::new(surface, &rays);
        sensor(SHAPE.0, SHAPE.1, fov)
            .into_iter()
            .any(|d| traced.hit(&rays, d).is_some())
    }

    #[test]
    fn the_largest_root_of_a_quartic() {
        // (r - 3)(r + 1)(r - 0.5)(r + 2) = r⁴ - 0.5 r³ - 7 r² - 2.5 r + 3
        let roots = real_roots(&[1.0, -0.5, -7.0, -2.5, 3.0], true);
        for (got, want) in roots.iter().zip([3.0, 0.5, -1.0, -2.0]) {
            assert!((got - want).abs() < 1e-12, "{roots:?}");
        }
        // (r² + 1)(r² + 4) has no real root; (r² + 1)(r - 2)(r + 5) has two.
        assert!(real_roots(&[1.0, 0.0, 5.0, 0.0, 4.0], true).is_empty());
        let two = real_roots(&[1.0, 3.0, -9.0, 3.0, -10.0], true);
        assert!(two.len() == 2 && (two[0] - 2.0).abs() < 1e-12 && (two[1] + 5.0).abs() < 1e-12);
    }

    /// The eye turned exactly by t, and the circle through the ray maps, lie on the sphere of
    /// radius t about the eye.
    #[test]
    fn the_ray_is_a_great_circle() {
        let rays = rays();
        let d = dir(0.0, 0.6, 0.8);
        for angle in [0.7f64, 2.5] {
            let exact = (rays.rotation.of(d).gp(angle).exp() >> origin()).cast::<Quadvector>();
            let circle = origin()
                + rays.linear.of(d).gp(angle.sin())
                + rays.quadratic.fill(d).gp(1.0 - angle.cos());
            let reach = w() + e().gp(angle.cos());
            assert!((reach & exact).s().abs() < 1e-7);
            assert!((reach & circle).s().abs() < 1e-12);
        }
    }

    /// In the half angle u the circle is a parabola, and its bend is the antipode of the eye.
    #[test]
    fn the_bend_of_the_ray_parabola_is_the_antipode() {
        let rays = rays();
        let d = dir(0.0, 0.6, 0.8);
        let bend = rays.bend.fill(d);
        for p in probes() {
            assert!((p & (bend - antipode())).s().abs() < 1e-12);
        }
        for angle in [0.7f64, 2.5] {
            let u = (angle / 2.0).tan();
            let parabola = origin() + rays.linear.of(d).gp(2.0 * u) + bend.gp(u * u);
            let circle = origin()
                + rays.linear.of(d).gp(angle.sin())
                + rays.quadratic.fill(d).gp(1.0 - angle.cos());
            for p in probes() {
                assert!((p & (parabola.gp(1.0 / (1.0 + u * u)) - circle)).s().abs() < 1e-12);
            }
        }
    }

    /// Built from great spheres and e only, the cylinder's quartic in u has equal outer and
    /// opposite odd coefficients, so its hits come in antipodal pairs; an off-axis dilation
    /// breaks that symmetry.
    #[test]
    fn the_cylinder_polynomial_is_palindromic_and_a_dupin_cyclide_breaks_it() {
        let rays = rays();
        let aim: Sphere = z().gp(0.7f64.cos()) + x().gp(0.7f64.sin());
        let lopsided = moved(dilation(aim, 1.5), cylinder(0.25));
        // numga's `PIXELS[::20000]`.
        let sample: Vec<Dir> = sensor(SHAPE.0, SHAPE.1, FOV)
            .into_iter()
            .step_by(20000)
            .collect();
        let gaps: Vec<f64> = [cylinder(0.3), lopsided]
            .into_iter()
            .map(|surface| {
                let traced = Traced::new(surface, &rays);
                let (mut outer, mut odd) = (0.0f64, 0.0f64);
                for d in &sample {
                    let c = traced.coefficients(*d);
                    outer = outer.max((c[4] - c[0]).abs());
                    odd = odd.max((c[3] + c[1]).abs());
                }
                outer + odd
            })
            .collect();
        assert!(gaps[0] < 1e-12 && gaps[1] > 1e-3, "{gaps:?}");
    }

    #[test]
    fn the_scenes_are_hit() {
        for surface in tori().into_iter().chain(dupins()).chain(spindles()) {
            assert!(any_hit(surface, FOV));
        }
    }

    #[test]
    fn the_vortex_stays_in_view() {
        for k in 0..3 {
            assert!(any_hit(
                vortex(core::f64::consts::TAU * k as f64 / 3.0),
                FOV
            ));
        }
    }

    #[test]
    fn the_flat_scenes_are_hit() {
        for scene in &FLAT_SCENES {
            for t in [0.0, core::f64::consts::PI] {
                let (surfaces, fov) = flat_scene(scene, t);
                for surface in surfaces {
                    assert!(any_hit(surface, fov), "{}", scene.name);
                }
            }
        }
    }

    /// Facing from the eye: a torus seen from outside faces the eye where it is hit, so most
    /// headlight cosines are positive.
    #[test]
    fn a_torus_faces_the_eye() {
        let rays = rays();
        let traced = Traced::new(tori()[1], &rays);
        let hits: Vec<f64> = sensor(60, 80, FOV)
            .into_iter()
            .filter_map(|d| traced.hit(&rays, d))
            .map(|h| h.facing)
            .collect();
        let positive = hits.iter().filter(|f| **f > 0.0).count();
        assert!(
            positive * 10 > hits.len() * 9,
            "{positive} of {}",
            hits.len()
        );
    }

    #[test]
    fn a_frame_draws() {
        let mut draw = super::draw;
        let c = gax_numga_examples::app::frame(
            &gax_numga_examples::Anim::new("t", 1.0).size(160, 120),
            1.5,
            &mut draw,
        );
        assert!(gax_light::luma(c.mean()) > 0.0);
    }
}
