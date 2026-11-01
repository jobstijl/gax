//! numga's `fluids/kirchhoff`: Kirchhoff's equations, a solid ellipsoid moving through still,
//! ideal fluid (PGA3D).
//!
//! A body moving through water has to push the water aside, and the water it sets moving
//! carries kinetic energy of its own. In an ideal fluid, one without viscosity that cannot be
//! compressed, that energy depends only on the body's motion, quadratically, so the fluid acts
//! as extra inertia carried along with the body: its added mass. Unlike the body's own mass,
//! the added mass depends on the direction of motion: an elongated body moving end-on pushes
//! little water aside, moving broadside a lot. Its momentum then no longer points along its
//! motion, and a body launched nearly end-on turns broadside, the Munk moment that makes
//! airships and submarines unstable without fins.
//!
//! The body's shape is a map on directions, `Shape = Direction <- Direction`, with its squared
//! semi-axes along its own axes; the ellipsoids confocal with it are `shape + widening ·
//! identity`, every squared semi-axis grown by the same amount. The flow around an ellipsoid
//! is built from that confocal family: averaged over it, weighted by the body's volume over
//! each confocal ellipsoid's, the inverse of the family gives the fluid's response to shifts, a
//! map on directions, and its outermorphism (`outermorphism::<Moment>()`, the extension by `∨`
//! to the lines at infinity) the response to turns, a map on couples. Added to the body's own
//! inertia, the result is one map from rates to momenta, `Forque <- Twist`, whose form
//! `rate & inertia(rate)` is twice the kinetic energy of body and fluid together. With no force
//! on either, the impulse, the momentum of body and fluid, stays fixed in space, and the body
//! moves at each moment at the rate whose momentum is that impulse as the body sees it.
//!
//! The average over the confocal family is a Gauss–Legendre rule (numpy's `leggauss`,
//! `gax::quadrature::gauss_legendre`).
//!
//! The animation shows the fluid's kinetic energy over the solid's for spheroids from a disc to
//! a rod (left, Lamb's coefficients), and an elongated body given a blow (right, seen from
//! above): without the fluid it keeps its heading; with it, launched nearly broadside it stays
//! so, and launched nearly end-on it turns past broadside.

use gax_numga_examples::{
    Align, Anim, Axes, Canvas, Light, Marker, Point2, Rect, backdrop, caption, palette, run_with,
};

mod kirchhoff {
    use gax::Unit;
    use gax::pga3d::{Direction, Forque, Line, Moment, Motor, Plane, Point, Twist};

    /// A stretch, carrying the unit sphere onto the body, and a shape, the squared semi-axes
    /// along the body's axes: maps on directions.
    pub type Shape = Direction<(Direction,)>;
    /// A map on couples, the momenta of turns.
    pub type Couples = Moment<(Moment,)>;
    /// An inertia: the momentum per rate.
    pub type Inertia = Forque<(Twist,)>;
    /// A placement of the body.
    pub type M = Unit<Motor>;

    /// The body's centre.
    pub fn centre() -> Point {
        Point::origin()
    }

    /// The planes through the centre square to its axes.
    pub fn axes() -> [Plane; 3] {
        [
            Plane::new(1.0, 0.0, 0.0, 0.0),
            Plane::new(0.0, 1.0, 0.0, 0.0),
            Plane::new(0.0, 0.0, 1.0, 0.0),
        ]
    }

    /// The directions along the axes: the planes' complements.
    pub fn directions() -> [Direction; 3] {
        axes().map(|a| a.dual().cast::<Direction>())
    }

    /// The velocity of the centre under a rate.
    pub fn centre_velocity() -> Direction<(Twist,)> {
        centre().commutator(Twist::slot()).cast::<Direction>()
    }

    /// The couple of a turn: half the sum, over the axes, of each direction joined with its own
    /// velocity (two directions join in a line at infinity, a moment). A shift moves no
    /// direction, so it has no couple.
    pub fn turn_couple() -> Moment<(Twist,)> {
        let joined: Moment<(Twist,)> = directions()
            .iter()
            .map(|d| *d & d.commutator(Twist::slot()).cast::<Direction>())
            .sum();
        joined * 0.5
    }

    /// An ellipsoid in its own frame: the unit sphere stretched by each semi-axis along its
    /// axis. The shape is the stretch applied twice.
    pub fn stretch(semi_axes: [f64; 3]) -> Shape {
        directions()
            .iter()
            .zip(axes())
            .zip(semi_axes)
            .map(|((d, a), s)| (*d * s) * (a & Direction::slot()))
            .sum()
    }

    /// The shape of a stretch: applied twice.
    pub fn shape_of(stretch: Shape) -> Shape {
        stretch.of(stretch)
    }

    /// The inertia of a solid ellipsoid of the shape and the given density, together with the
    /// ideal fluid of the given density it moves through.
    pub fn inertia(shape: Shape, density: f64, fluid_density: f64, n_widenings: usize) -> Inertia {
        let (widenings, weights) = quadrature(n_widenings);
        let id = Direction::slot();
        // The mean squared semi-axis, the trace over the axes, sets the scale of the widenings,
        // and the body's volume is a square root of the shape's determinant.
        let scale = shape.trace() / 3.0;
        #[allow(clippy::disallowed_methods)] // an ellipsoid's volume: 4π/3 times √det
        let volume = 4.0 / 3.0 * core::f64::consts::PI * shape.det().sqrt();
        // The confocal ellipsoids, every squared semi-axis grown by the same widening, each
        // weighed by the body's volume over its own. Averaged over the family, the inverse shape
        // gives the response to a shift along each axis, and its outermorphism the response to a
        // turn in each plane. The first has trace two.
        let mut shifting = Shape::zero();
        let mut turning = Couples::zero();
        for (widening, weight) in widenings.iter().zip(&weights) {
            let confocal = shape + id * (scale * widening);
            #[allow(clippy::disallowed_methods)] // the volumes' ratio, a ratio of √det
            let weight = (shape.det() / confocal.det()).sqrt() * scale * weight;
            let inverse = confocal.inverse();
            shifting += inverse * weight;
            turning += inverse.outermorphism::<Moment>() * weight;
        }
        // On each couple, the outermorphism multiplies the squared semi-axes in its plane and
        // `spread` adds them: the mixed term of the outermorphism of the shape plus the
        // identity, the shape acting on one direction of the couple at a time. Their
        // discriminant is the square of their difference, which vanishes for a turn about an
        // axis of symmetry: such a turn moves no fluid.
        let areas = shape.outermorphism::<Moment>();
        let spread = (id + shape).outermorphism::<Moment>() - Moment::slot() - areas;
        let discriminant = spread.of(spread) - areas * 4.0;
        // The fluid's mass under a shift and its inertia under a turn, per unit of the displaced
        // fluid's mass.
        let shift_mass = (id * 2.0 - shifting).solve(shifting);
        let turn_mass =
            (Moment::slot() * 2.0 - spread.of(turning)).solve(discriminant.of(turning)) * 0.2;
        // The solid's own: its mass whichever way it moves, and a fifth of it times the spread
        // under a turn.
        let masses = id * (volume * density) + shift_mass * (volume * fluid_density);
        let turns = spread * (volume * density / 5.0) + turn_mass * (volume * fluid_density);
        // A shift's momentum is a line through the centre, a turn's a couple.
        (centre() & masses.of(centre_velocity())) + turns.of(turn_couple()).cast::<Line>()
    }

    /// The body's placement from the start and after each step, with no force on body or
    /// fluid: the impulse stays fixed in space, and the body moves at the rate whose momentum is
    /// the impulse as the body sees it, taken at the midpoint of each step.
    pub fn glide(start: M, impulse: Forque, inertia: Inertia, dt: f64, steps: usize) -> Vec<M> {
        let mobility = inertia.inverse();
        core::iter::successors(Some(start), |m| {
            let halfway = *m * (mobility.of(*m << impulse) * (-dt / 4.0)).exp();
            Some(*m * (mobility.of(halfway << impulse) * (-dt / 2.0)).exp())
        })
        .take(steps + 1)
        .collect()
    }

    /// Widenings from zero to infinity in units of the mean squared semi-axis, and their
    /// weights: Gauss–Legendre nodes in `t` on the unit interval, at widenings `1 / t² − 1`, so
    /// that the integrands fall off as powers of `t`.
    pub fn quadrature(n: usize) -> (Vec<f64>, Vec<f64>) {
        let (nodes, weights) = gax::quadrature::gauss_legendre::<f64>(n);
        let t: Vec<f64> = nodes.iter().map(|x| (x + 1.0) / 2.0).collect();
        (
            t.iter().map(|t| 1.0 / (t * t) - 1.0).collect(),
            weights
                .iter()
                .zip(&t)
                .map(|(w, t)| w / (t * t * t))
                .collect(),
        )
    }

    // --- scenes

    /// Spheroids from a disc to a rod: the semi-axis along the axis of symmetry over the other
    /// two, 41 of them from 0.1 to 10, evenly in the logarithm.
    pub const N_ASPECTS: usize = 41;
    pub fn aspects() -> Vec<f64> {
        (0..N_ASPECTS)
            .map(|k| 10f64.powf(-1.0 + 2.0 * k as f64 / (N_ASPECTS - 1) as f64))
            .collect()
    }
    /// The samples of the confocal family.
    pub const N_WIDENINGS: usize = 48;
    /// The gliding body: its semi-axes, its density and the fluid's, the blow it is given along
    /// the first axis of space, and its long axis's angle from the blow at the start, in each
    /// case: without the fluid, nearly end-on and nearly broadside.
    pub const SEMI_AXES: [f64; 3] = [3.0, 1.0, 1.0];
    pub const DENSITY: f64 = 1.0;
    pub const FLUID_DENSITIES: [f64; 3] = [0.0, 1.0, 1.0];
    pub const BLOW: f64 = 12.0;
    pub const HEADINGS: [f64; 3] = [0.1, 0.1, core::f64::consts::FRAC_PI_2 - 0.1];
    /// The time step, the steps, the steps per frame and the samples around the outline.
    pub const DT: f64 = 0.16;
    pub const N_STEPS: usize = 250;
    pub const N_STEPS_PER_FRAME: usize = 2;
    pub const N_OUTLINE: usize = 48;

    /// The rates the coefficients read: shifting along the axis of symmetry, shifting across
    /// it, and turning it (`e01`, `e02`, `e31`).
    pub fn rates() -> [Twist; 3] {
        let [x, y, z] = axes();
        let w = Plane::new(0.0, 0.0, 0.0, 1.0);
        [w ^ x, w ^ y, z ^ x]
    }

    /// The fluid's kinetic energy over the solid's for a shape, at equal densities, for each
    /// of the rates.
    pub fn ratios(shape: Shape) -> [f64; 3] {
        let solid = inertia(shape, 1.0, 0.0, N_WIDENINGS);
        let fluid = inertia(shape, 0.0, 1.0, N_WIDENINGS);
        rates().map(|r| (r & fluid.of(r)).s() / (r & solid.of(r)).s())
    }

    /// The ratios for spheroids of each aspect.
    pub fn coefficients() -> Vec<[f64; 3]> {
        aspects()
            .iter()
            .map(|a| ratios(shape_of(stretch([*a, 1.0, 1.0]))))
            .collect()
    }

    /// The long axis, in the body's frame.
    pub fn long_axis() -> Direction {
        directions()[0]
    }

    /// Each case's inertia, the impulse of the blow along the first axis of space through the
    /// body's centre, and the body's placements at the start, its long axis turned from the
    /// blow about the vertical.
    pub fn launch() -> ([Inertia; 3], Forque, [M; 3]) {
        let shape = shape_of(stretch(SEMI_AXES));
        let inertias = FLUID_DENSITIES.map(|f| inertia(shape, DENSITY, f, N_WIDENINGS));
        let impulse = (centre() & long_axis()) * BLOW;
        let up = Point::direction(0.0, 0.0, 1.0);
        let starts = HEADINGS.map(|h| Motor::rotation_about(up, h));
        (inertias, impulse, starts)
    }

    /// Each case's placements over the time span: `[case][step]`.
    pub fn gliding() -> [Vec<M>; 3] {
        let (inertias, impulse, starts) = launch();
        core::array::from_fn(|k| glide(starts[k], impulse, inertias[k], DT, N_STEPS))
    }

    /// The body's outline in the plane of its first two axes, in its own frame: the long axis
    /// turned about the vertical round a circle, stretched onto the body.
    pub fn outline() -> Vec<Point> {
        let stretch = stretch(SEMI_AXES);
        let up = Point::direction(0.0, 0.0, 1.0);
        (0..=N_OUTLINE)
            .map(|k| {
                let angle = core::f64::consts::TAU * k as f64 / N_OUTLINE as f64;
                let circle = Motor::rotation_about(up, angle) >> long_axis();
                centre() + stretch.of(circle)
            })
            .collect()
    }

    /// The tip of the long axis, in the body's frame.
    pub fn nose() -> Point {
        centre() + stretch(SEMI_AXES).of(long_axis())
    }

    /// The heading of a direction about the vertical, from the first axis of space.
    #[cfg(test)]
    pub fn heading(d: Direction) -> f64 {
        let [x, y, _] = axes();
        let flat = gax::pga2d::Point::direction((x & d).s(), (y & d).s());
        gax::pga2d::Point::direction(1.0, 0.0).angle_to(flat)
    }
}

use gax::pga3d::{Plane, Point};
use kirchhoff::*;

/// The loop: numga's 126 frames of 40 ms, a frame every two steps.
const SECONDS: f32 = 5.04;
/// The free space between the lanes, and around the bodies at the ends of the view.
const GAP: f64 = 1.0;

/// What the animation draws: the coefficients by aspect, and every case's outline, centre and
/// nose at every step, in space.
struct Scenes {
    aspects: Vec<f64>,
    coefficients: Vec<[f64; 3]>,
    impulse: gax::pga3d::Forque,
    outlines: [Vec<Vec<Point>>; 3],
    centres: [Vec<Point>; 3],
    noses: [Vec<Point>; 3],
}

fn scenes() -> Scenes {
    let motors = gliding();
    let (_, impulse, _) = launch();
    let (shape, nose) = (outline(), nose());
    let place = |m: &M, p: &Point| *m >> *p;
    Scenes {
        aspects: aspects(),
        coefficients: coefficients(),
        impulse,
        outlines: motors.each_ref().map(|ms| {
            ms.iter()
                .map(|m| shape.iter().map(|p| place(m, p)).collect())
                .collect()
        }),
        centres: motors
            .each_ref()
            .map(|ms| ms.iter().map(|m| place(m, &centre())).collect()),
        noses: motors
            .each_ref()
            .map(|ms| ms.iter().map(|m| place(m, &nose)).collect()),
    }
}

/// The lights of the three coefficients: shifting along the axis of symmetry, across it, and
/// turning it.
const COEFFICIENT: [Light; 3] = [palette::SKY, palette::ORANGE, palette::GREEN];
/// Each case's light: without the fluid, nearly end-on, nearly broadside.
const CASE: [Light; 3] = [palette::INK, palette::SKY, palette::ORANGE];

/// A point of space seen from above, at unit weight: its pairings with the planes across `x`
/// and `y`.
fn above(p: Point) -> Point2 {
    let p = p.unitized();
    let [x, y, _] = axes();
    Point2::xy((x & p).s() as f32, (y & p).s() as f32)
}

/// The fluid's kinetic energy over the solid's for spheroids of each aspect, against the
/// aspect: discs on the left, rods on the right, the sphere at the grey line.
fn draw_coefficients(c: &mut Canvas, rect: Rect, s: &Scenes) {
    let top = s
        .coefficients
        .iter()
        .flatten()
        .fold(0.0, |a: f64, v| a.max(*v));
    let ax = Axes::new(rect, [0.1, 10.0], [0.0, (top * 1.05) as f32]).log_x();
    ax.line(
        c,
        Point2::xy(1.0, 0.0),
        Point2::xy(1.0, top as f32),
        1.0,
        palette::GRID.faded(2.0),
    );
    for (k, light) in COEFFICIENT.iter().enumerate() {
        let curve: Vec<Point2> = s
            .aspects
            .iter()
            .zip(&s.coefficients)
            .map(|(a, r)| Point2::xy(*a as f32, r[k] as f32))
            .collect();
        ax.polyline(c, &curve, 1.8, *light);
    }
    ax.frame(
        c,
        "ENERGY, FLUID OVER SOLID",
        "ASPECT: AXIS OF SYMMETRY",
        "",
    );
    ax.legend(
        c,
        &[
            ("SHIFT ALONG", COEFFICIENT[0]),
            ("SHIFT ACROSS", COEFFICIENT[1]),
            ("TURN", COEFFICIENT[2]),
        ],
    );
    let at = ax.at(0.03, 0.04);
    ax.text(c, at, "DISCS", 8.0, palette::INK.faded(0.6), Align::Left);
    let at = ax.at(0.97, 0.04);
    ax.text(c, at, "RODS", 8.0, palette::INK.faded(0.6), Align::Right);
}

/// Every case's body seen from above, each in its own lane, the first furthest, with its path
/// so far, its nose and the line of its impulse.
fn draw_gliding(c: &mut Canvas, rect: Rect, s: &Scenes, step: usize) {
    let [x, y, _] = axes();
    // Each lane as wide as the bodies ever reach across it, and the view as long as they reach
    // along it.
    let all = s.outlines.iter().flatten().flatten();
    let width = all
        .clone()
        .map(|p| (y & p.unitized()).s().abs())
        .fold(0.0, f64::max);
    let lane = 2.0 * width + GAP;
    let along = all.map(|p| (x & p.unitized()).s());
    let (lo, hi) = along.fold((f64::MAX, f64::MIN), |(a, b), v| (a.min(v), b.max(v)));
    let extent = [lo - GAP / 2.0, hi + GAP / 2.0];
    let height = lane * 3.0;
    let mid = Point2::xy(
        ((extent[0] + extent[1]) / 2.0) as f32,
        ((height - lane) / 2.0) as f32,
    );
    let half =
        (height / 2.0).max((extent[1] - extent[0]) / 2.0 * f64::from(rect.height() / rect.width()));
    let ax = Axes::equal(rect, mid, half as f32);
    // The ends of the impulse's line within the extent: where it meets the planes square to x
    // there.
    let ends = extent.map(|e| (s.impulse ^ Plane::new(1.0, 0.0, 0.0, -e)).unitized());
    let names = ["NO FLUID", "NEARLY END-ON", "NEARLY BROADSIDE"];
    for (case, (light, name)) in CASE.iter().zip(names).enumerate() {
        // Each case moved across into its own lane, the first furthest.
        let across = lane * (2 - case) as f64;
        let lane_shift = Point::from(directions()[1] * across);
        let shifted = |p: Point| above(p + lane_shift);
        let line: Vec<Point2> = ends.iter().map(|p| shifted(p.with_orientation())).collect();
        ax.dashed(c, &line, 0.8, 6.0, palette::INK.faded(0.35));
        let path: Vec<Point2> = s.centres[case][..=step]
            .iter()
            .map(|p| shifted(*p))
            .collect();
        ax.polyline(c, &path, 1.4, light.faded(0.7));
        let body: Vec<Point2> = s.outlines[case][step].iter().map(|p| shifted(*p)).collect();
        ax.fill(c, &body, *light, 0.35);
        let px: Vec<Point2> = body.iter().map(|p| ax.px(*p)).collect();
        let sides: Vec<[Point2; 2]> = px.windows(2).map(|w| [w[0], w[1]]).collect();
        c.outline(&sides, 1.2, palette::BOTTOM, 0.9);
        ax.polyline(c, &body, 1.4, *light);
        let nose = shifted(s.noses[case][step]);
        ax.scatter(c, &[nose], Marker::Dot, 6.0, light.whitened(0.3));
        let at = Point2::xy(extent[0] as f32 + 0.2, (across + lane / 2.0 - 0.35) as f32);
        ax.text(c, at, name, 9.0, light.faded(0.8), Align::Left);
    }
}

fn draw(c: &mut Canvas, s: &Scenes, t: f32) {
    backdrop(c);
    let frames = N_STEPS / N_STEPS_PER_FRAME + 1;
    let frame = gax_numga_examples::signal::frame((t / SECONDS).rem_euclid(1.0), frames);
    let step = frame * N_STEPS_PER_FRAME;
    let screen = c.rect();
    let (w, h) = (screen.width(), screen.height());
    let body = Rect::new(0.0, h * 0.15, w, h);
    draw_coefficients(
        c,
        body.part(0.0, 0.0, 0.38, 1.0).inset(40.0, 34.0, 14.0, 52.0),
        s,
    );
    draw_gliding(
        c,
        body.part(0.38, 0.0, 1.0, 1.0).inset(16.0, 8.0, 12.0, 8.0),
        s,
        step,
    );
    let time = step as f64 * DT;
    caption(
        c,
        "KIRCHHOFF'S EQUATIONS: AN ELLIPSOID IN IDEAL FLUID",
        &format!(
            "ADDED MASS FROM THE CONFOCAL FAMILY, ITS TURNS BY THE OUTERMORPHISM ON MOMENTS (PGA3D). T = {time:.1}"
        ),
    );
}

fn main() {
    run_with(Anim::new("kirchhoff", SECONDS).size(960, 540), scenes, draw);
}

#[cfg(test)]
mod tests {
    use super::kirchhoff::*;
    use gax::pga3d::{Direction, Line, Motor, Point};
    use gax_numga_examples::rng::{Draw, rng};

    /// The Gauss–Legendre rule integrates every polynomial up to degree `2n − 1` exactly.
    #[test]
    fn gauss_legendre_is_exact_on_polynomials() {
        let (nodes, weights) = gax::quadrature::gauss_legendre::<f64>(N_WIDENINGS);
        assert!(nodes.windows(2).all(|w| w[0] < w[1]));
        for degree in [0, 1, 2, 10, 51, 94, 95] {
            let sum: f64 = nodes
                .iter()
                .zip(&weights)
                .map(|(x, w)| w * x.powi(degree))
                .sum();
            let exact = if degree % 2 == 1 {
                0.0
            } else {
                2.0 / (degree as f64 + 1.0)
            };
            assert!((sum - exact).abs() < 1e-14, "{degree}: {sum} {exact}");
        }
    }

    /// The directions are the planes' complements, and a stretch stretches along them.
    #[test]
    fn a_stretch_stretches_along_the_axes() {
        assert_eq!(
            directions()[0],
            Point::direction(1.0, 0.0, 0.0).cast::<Direction>()
        );
        let s = stretch([3.0, 2.0, 0.5]);
        for (d, a) in directions().iter().zip([3.0, 2.0, 0.5]) {
            assert_eq!(s.of(*d), *d * a);
        }
    }

    /// For a prolate spheroid of aspect three the fluid's kinetic energy over the solid's is
    /// 0.122 shifting along its axis, 0.803 across it, and 0.465 turning it, in Lamb's table, to
    /// its three digits. The table is laid out by eccentricity, and lists this spheroid's aspect
    /// rounded to 2.99.
    #[test]
    fn spheroid_matches_lambs_table() {
        let ratios = ratios(shape_of(stretch([3.0, 1.0, 1.0])));
        for (r, lamb) in ratios.iter().zip([0.122, 0.803, 0.465]) {
            assert!((r - lamb).abs() < 1e-3, "{ratios:?}");
        }
    }

    /// The kinetic energy is a symmetric form on rates, for an ellipsoid with three different
    /// axes, and an ellipsoid turned by a motor has the turned inertia: the confocal family
    /// needs no principal frame.
    #[test]
    fn inertia_is_symmetric_and_moves_with_the_body() {
        let s = stretch([2.0, 1.3, 0.6]);
        let inertia = inertia(shape_of(s), 1.5, 1.0, N_WIDENINGS);
        let mut r = rng(0);
        let mut draw = || Line::<(), f64>::from_coeffs(core::array::from_fn(|_| r.normal()));
        let (first, second) = (draw(), draw());
        let (a, b) = (
            (first & inertia.of(second)).s(),
            (second & inertia.of(first)).s(),
        );
        assert!((a - b).abs() <= 1e-12 * a.abs(), "{a} {b}");
        let [x, y, z] = axes();
        let motor = ((x ^ y) * -0.4).exp() * ((y ^ z) * 0.7).exp();
        let turned = motor >> s.of(motor << Direction::slot());
        let moved = super::kirchhoff::inertia(shape_of(turned), 1.5, 1.0, N_WIDENINGS);
        let c = ((motor >> second) & moved.of(motor >> first)).s();
        let d = (second & inertia.of(first)).s();
        assert!((c - d).abs() <= 1e-10 * d.abs(), "{c} {d}");
    }

    /// A spheroid turned about its axis of symmetry stays where it was, so the fluid stays at
    /// rest.
    #[test]
    fn turning_about_the_axis_of_symmetry_moves_no_fluid() {
        let fluid = inertia(shape_of(stretch([0.3, 1.0, 1.0])), 0.0, 1.0, N_WIDENINGS);
        let [_, y, z] = axes();
        let spin = y ^ z;
        assert!((spin & fluid.of(spin)).s().abs() < 1e-12);
    }

    /// A sphere's added mass is half the mass of the fluid it displaces, whichever way it
    /// moves, and a turn moves no fluid.
    #[test]
    fn a_sphere_carries_half_its_displaced_fluid() {
        let sphere = inertia(shape_of(stretch([1.0, 1.0, 1.0])), 0.0, 1.0, N_WIDENINGS);
        let half = 2.0 * core::f64::consts::PI / 3.0;
        for (r, expected) in rates().iter().zip([half, half, 0.0]) {
            let e = (*r & sphere.of(*r)).s();
            assert!((e - expected).abs() < 1e-12, "{e} {expected}");
        }
    }

    /// The kinetic energy of body and fluid is kept to within the step's error. Without the
    /// fluid the body keeps its heading; given the blow nearly broadside it stays so; nearly
    /// end-on it turns past broadside.
    #[test]
    fn the_glides_keep_their_energy_and_turn_broadside() {
        let (inertias, impulse, _) = launch();
        let motors = gliding();
        for (case, ms) in motors.iter().enumerate() {
            let mobility = inertias[case].inverse();
            let energy = |m: &M| {
                let seen = *m << impulse;
                (mobility.of(seen) & seen).s()
            };
            let first = energy(&ms[0]);
            for m in ms {
                assert!((energy(m) / first - 1.0).abs() < 1e-2);
            }
        }
        let headings: Vec<Vec<f64>> = motors
            .iter()
            .map(|ms| ms.iter().map(|m| heading(*m >> long_axis())).collect())
            .collect();
        assert!(headings[0].iter().all(|h| (h - HEADINGS[0]).abs() < 1e-12));
        let broadside = core::f64::consts::FRAC_PI_2;
        let swing = headings[2]
            .iter()
            .map(|h| (h - broadside).abs())
            .fold(0.0, f64::max);
        assert!(swing < 2.0 * (broadside - HEADINGS[2]), "{swing}");
        assert!(headings[1].iter().any(|h| *h > broadside));
    }

    /// The placements stay rigid motions.
    #[test]
    fn the_placements_stay_unit() {
        for m in gliding().iter().flatten() {
            let n: Motor = m.into_inner() * m.into_inner().reverse();
            assert!((n.c[0] - 1.0).abs() < 1e-12 && n.c[1..].iter().all(|c| c.abs() < 1e-12));
        }
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws_with(super::draw, &super::scenes(), 1.0);
    }
}
