//! numga's `mechanics/jeffery`: Jeffery orbits, rigid ellipsoids tumbling in a shear flow, in
//! VGA3D.
//!
//! A slow, sticky flow (honey sheared between two plates) moves each point at a velocity that
//! changes linearly across the flow: a map on offsets, the gradient `Vector <- Vector`. A drop
//! of the fluid would both turn and stretch with it; a rigid particle can only turn, and the
//! fluid presses on it until it turns at the one rate that leaves no torque on it, found by
//! Jeffery in 1922.
//!
//! The particle is an ellipsoid, a stretch of the unit sphere along its own axes (a sum of
//! dyads `a_i e_i (e_i | ·)`), and its shape is the stretch applied twice. Jeffery's spin, a
//! bivector, makes the shape's product with the particle's turning map, from both sides, equal
//! to the gradient's product with the shape less the shape's with the gradient's adjoint:
//!
//! `shape(turning) + turning(shape) == gradient(shape) - shape(gradient*)`
//!
//! with `turning = Vector | spin`, the velocity each offset has under the spin. The adjoint
//! `gradient*` is the map with `gradient*(a) | b == a | gradient(b)`, a pairing solve against
//! the metric `Vector | Vector`. Both sides are antisymmetric maps, so their curls carry them
//! whole. numga writes the curl as `(Vector ^ map(Vector)).contract(1, 2)`; gax has no
//! contraction of two input slots, so here it is the sum `Σ e_i ^ map(e_i)` over the
//! orthonormal basis, which is the same in VGA3D. The left side's curl is a map from the
//! spin's bivector to a bivector (`Bivector <- Bivector`), and the spin is one solve. A sphere
//! spins at the flow's own rate, half the curl of the gradient. A rod lingers along the flow,
//! then flips end over end, and its axis comes back to where it started after a fixed time, on
//! a closed path on the sphere of directions: a Jeffery orbit. An ellipsoid with three
//! different axes never quite repeats.
//!
//! The animation shows, for a sphere, a rod and an ellipsoid with three axes, the paths of the
//! long axis from five starts on the sphere of directions (above), and the body from the middle
//! start tumbling in the flow (below). The flow runs to the right, its gradient points up and
//! the vorticity axis towards the viewer.

use gax_numga_examples::scene3::panel3;
use gax_numga_examples::{
    Align, Anim, Camera, Canvas, Light, Marker, ORIGIN3, Point2, Point3, Rect, backdrop, caption,
    colormap, palette, run,
};
use std::sync::OnceLock;

#[path = "../shared/mechanics_vga3d.rs"]
mod space;

mod jeffery {
    pub use super::space::{AXES, B, Map, Turn, V, adjoint, curl, turn};
    use gax::vga3d::{Bivector, Vector};

    /// An ellipsoid in its own frame: the unit sphere stretched by each semi-axis along its
    /// axis.
    pub fn stretch(semi_axes: [f64; 3]) -> Map {
        AXES.iter()
            .zip(semi_axes)
            .map(|(e, a)| (*e * a) * (*e | Vector::slot()))
            .sum()
    }

    /// The spin of a rigid ellipsoid of the given shape in a flow of the given gradient, free
    /// of torque: Jeffery's.
    pub fn spin(shape: Map, gradient: Map) -> B {
        // Vector <- (Vector, Bivector): the velocity of an offset under a spin.
        let turning = Vector::slot() | Bivector::slot();
        let left = shape.of(turning) + turning.of(shape);
        let right = gradient.of(shape) - shape.of(adjoint(gradient));
        // Both sides are antisymmetric, so their curls carry them whole: a map on spins, and a
        // bivector.
        curl(left).solve(curl(right))
    }

    /// The shape of a body, its stretch applied twice, turned by its orientation.
    pub fn placed(stretch: Map, turn: Turn) -> Map {
        turn >> stretch.of(stretch).of(turn << Vector::slot())
    }

    /// The body turned for a time `dt` by its spin, taken at the midpoint of the step.
    pub fn step(turn: Turn, stretch: Map, gradient: Map, dt: f64) -> Turn {
        let halfway = (spin(placed(stretch, turn), gradient) * (-dt / 4.0)).exp() * turn;
        (spin(placed(stretch, halfway), gradient) * (-dt / 2.0)).exp() * turn
    }

    /// The body's orientations, from the start and after each of the steps.
    pub fn tumble(start: Turn, stretch: Map, gradient: Map, dt: f64, steps: usize) -> Vec<Turn> {
        core::iter::successors(Some(start), |t| Some(step(*t, stretch, gradient, dt)))
            .take(steps + 1)
            .collect()
    }

    /// The bodies: their names and semi-axes. A sphere, a rod, and an ellipsoid with three
    /// different axes.
    pub const NAMES: [&str; 3] = ["SPHERE", "ROD", "THREE AXES"];
    pub const SEMI_AXES: [[f64; 3]; 3] = [[1.0, 1.0, 1.0], [ASPECT, 1.0, 1.0], [4.0, 2.0, 1.0]];
    /// The rod's long semi-axis over its short ones.
    pub const ASPECT: f64 = 4.0;
    /// The flow's velocity along the first axis, per offset along the second.
    pub const SHEAR_RATE: f64 = 1.0;
    /// The angles of each body's long axis from the vorticity axis at the start, in the plane
    /// of the flow and the vorticity.
    pub const TILTS: [f64; 5] = [0.15, 0.4, 0.7, 1.0, 1.3];
    /// The time span in periods of the rod's orbit, and the steps per period.
    pub const PERIODS: usize = 2;
    pub const STEPS_PER_PERIOD: usize = 120;

    /// Simple shear: the velocity along the flow grows with the offset along the gradient.
    pub fn flow() -> Map {
        let [x, y, _] = AXES;
        (x * SHEAR_RATE) * (y | Vector::slot())
    }

    /// The time a rod of the aspect takes to come back to where it started, Jeffery's period.
    pub fn period() -> f64 {
        core::f64::consts::TAU * (ASPECT + 1.0 / ASPECT) / SHEAR_RATE
    }

    /// The step of the integration.
    pub fn dt() -> f64 {
        period() / STEPS_PER_PERIOD as f64
    }

    /// Each body from each start over the time span: `[body][start][step]`.
    pub fn tumbling() -> Vec<Vec<Vec<Turn>>> {
        let [x, _, z] = AXES;
        // The long axis turned from the flow towards the vorticity axis, to its tilt from it.
        let starts = TILTS.map(|tilt| turn(x ^ z, core::f64::consts::FRAC_PI_2 - tilt));
        SEMI_AXES
            .iter()
            .map(|semi_axes| {
                let shape = stretch(*semi_axes);
                let steps = PERIODS * STEPS_PER_PERIOD;
                starts
                    .iter()
                    .map(|s| tumble(*s, shape, flow(), dt(), steps))
                    .collect()
            })
            .collect()
    }
}

use jeffery::*;

fn bodies() -> &'static Vec<Vec<Vec<Turn>>> {
    static T: OnceLock<Vec<Vec<Vec<Turn>>>> = OnceLock::new();
    T.get_or_init(tumbling)
}

/// The length of the loop in seconds: the two periods of the rod's orbit.
const LOOP: f32 = 14.0;
/// The start the lower row shows, and the steps a tip's trail lasts.
const SHOWN: usize = 2;
const TRAIL: usize = 75;
/// The heights at which the flow is drawn, below and above the body.
const HEIGHTS: [f64; 6] = [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0];
/// The view, as numga's: turned from the vorticity axis towards the flow by 28 degrees, and
/// raised towards the gradient by 18.
const AZIMUTH: f32 = 28f32.to_radians();
const ELEVATION: f32 = 18f32.to_radians();
/// The reach of each panel's view from the centre to its top: the sphere of directions with
/// its axes, and numga's half width of the tumbling view with a margin.
const ORBIT_REACH: f32 = 1.4;
const BODY_REACH: f32 = 4.3;
/// The length of the flow and vorticity arrows on the sphere of directions.
const ARROW: f64 = 1.35;
/// The longitudes and latitudes of a body's surface, multiples of four and two so that no
/// quad straddles an octant.
const LONGITUDES: usize = 32;
const LATITUDES: usize = 16;

/// A direction of the flow, drawn: along the vorticity axis, the flow and the gradient, so
/// that the gradient is the drawing's up (`z`) and the frame stays right-handed.
fn direction(v: V) -> Point3 {
    Point3::direction(v.e3() as f32, v.e1() as f32, v.e2() as f32)
}

/// The point an offset from the body's centre reaches, drawn.
fn position(v: V) -> Point3 {
    ORIGIN3 + direction(v)
}

/// The colour of each start, by its tilt.
fn start_colour(start: usize) -> Light {
    colormap::viridis(0.3 + 0.7 * start as f32 / (TILTS.len() - 1) as f32)
}

/// The point of the unit sphere at longitude `u` and latitude `v` (both from 0 to 1, the
/// latitude from the south pole): the first axis raised towards the third, then turned towards
/// the second.
fn on_sphere(u: f64, v: f64) -> V {
    let [x, y, z] = AXES;
    let pi = core::f64::consts::PI;
    turn(x ^ y, 2.0 * pi * u) >> (turn(x ^ z, pi * (v - 0.5)) >> x)
}

/// One panel's parallel camera, `reach` units from the centre to the panel's top.
fn camera(rect: Rect, reach: f32) -> Camera {
    Camera::parallel(rect, 0.5 * rect.height() / reach, AZIMUTH, ELEVATION)
}

/// The step shown at time `t` of the loop.
fn step_at(t: f32) -> usize {
    let steps = PERIODS * STEPS_PER_PERIOD;
    ((t.rem_euclid(LOOP) / LOOP * steps as f32) as usize).min(steps)
}

/// The sphere of directions: the path of the long axis from each start, both ends drawn, a dot
/// where it is now, and the flow and vorticity axes.
fn draw_orbits(c: &mut Canvas, rect: Rect, body: usize, now: usize) {
    let [x, _, z] = AXES;
    let cam = panel3(c, rect, camera(rect, ORBIT_REACH), |s| {
        s.sphere_wire(ORIGIN3, 1.0, 24, palette::grid().faded(0.6));
        for axis in [x, z] {
            s.arrow(
                ORIGIN3,
                direction(axis * ARROW),
                1.4,
                7.0,
                palette::ink().faded(0.4),
            );
        }
        for (start, turns) in bodies()[body].iter().enumerate() {
            let colour = start_colour(start);
            for end in [1.0, -1.0] {
                let path: Vec<Point3> = turns.iter().map(|t| position(*t >> (x * end))).collect();
                s.polyline(&path, 1.1, colour.faded(0.5));
                s.dot(path[now], Marker::Dot, 9.0, colour);
            }
        }
    });
    // Each label just past its arrow's tip, and below it.
    let label = |c: &mut Canvas, axis: V, text: &str| {
        if let Some(tip) = cam.px(position(axis * (ARROW + 0.1))) {
            let at = tip + Point2::direction(0.0, 12.0);
            c.text(text, at, 9.0, palette::ink().faded(0.6), Align::Center);
        }
    };
    label(c, x, "FLOW");
    label(c, z, "VORTICITY");
}

/// One body at one step: its surface checkered by its octants and lit, the stubs of its long
/// axis, the recent trail of the axis's tips, and the shear flow's velocity around it.
fn draw_body(c: &mut Canvas, rect: Rect, body: usize, now: usize) {
    let [x, y, _] = AXES;
    let turns = &bodies()[body][SHOWN];
    let turn = turns[now];
    let shape = stretch(SEMI_AXES[body]);
    let long = SEMI_AXES[body][0];
    panel3(c, rect, camera(rect, BODY_REACH), |s| {
        for height in HEIGHTS {
            let velocity = flow().of(y * height);
            let base = y * height - velocity * 0.5;
            s.arrow(
                position(base),
                direction(velocity),
                1.4,
                7.0,
                palette::grid().faded(2.5),
            );
        }
        // The colour of a patch: light or dark by the octant of the body's own direction there.
        let checker = |u: f32, v: f32| {
            let d = on_sphere(f64::from(u), f64::from(v));
            if d.e1() * d.e2() * d.e3() > 0.0 {
                palette::sky().faded(0.55)
            } else {
                palette::blue().faded(0.35)
            }
        };
        let surface =
            |u: f32, v: f32| position(turn >> shape.of(on_sphere(f64::from(u), f64::from(v))));
        s.surface(surface, LONGITUDES, LATITUDES, checker, 0.96, None);
        let tips: Vec<V> = turns[now.saturating_sub(TRAIL)..=now]
            .iter()
            .map(|t| *t >> (x * long))
            .collect();
        for end in [1.0, -1.0] {
            let tip = tips[tips.len() - 1] * end;
            s.seg(position(tip), position(tip * 1.15), 2.2, palette::orange());
            let trail: Vec<Point3> = tips.iter().map(|t| position(*t * end)).collect();
            s.polyline(&trail, 1.6, palette::orange().faded(0.8));
        }
    });
}

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let screen = c.rect();
    let (w, h) = (screen.width(), screen.height());
    let now = step_at(t);
    let top = h * 0.13;
    let middle = h * 0.56;
    for (body, name) in NAMES.iter().enumerate() {
        let column = Rect::new(0.0, top, w, h).column(body, 3);
        let orbits = Rect::new(column.lo.e20(), top, column.hi.e20(), middle);
        let tumbling = Rect::new(column.lo.e20(), middle, column.hi.e20(), h);
        draw_orbits(c, orbits.inset(4.0, 16.0, 4.0, 0.0), body, now);
        draw_body(c, tumbling.inset(4.0, 0.0, 4.0, 4.0), body, now);
        let at = Point2::xy(column.centre().e20(), top + 8.0);
        c.text(name, at, 12.0, palette::ink(), Align::Center);
    }
    let periods = now as f64 / STEPS_PER_PERIOD as f64;
    caption(
        c,
        "JEFFERY ORBITS: ELLIPSOIDS TUMBLING IN A SHEAR FLOW",
        &format!(
            "SPIN: ONE SOLVE ON CURLS (VGA3D). ABOVE: LONG AXES ON THE SPHERE OF DIRECTIONS. {periods:.2} ROD PERIODS"
        ),
    );
}

fn main() {
    run(Anim::new("jeffery", LOOP).size(960, 540), draw);
}

#[cfg(test)]
mod tests {
    use super::jeffery::*;
    use gax::vga3d::{Rotor, Vector};

    /// A gradient from its entries `x_i | gradient(x_j)`: a sum of dyads.
    fn gradient_of(entries: [[f64; 3]; 3]) -> Map {
        (0..3)
            .flat_map(|i| (0..3).map(move |j| (i, j)))
            .map(|(i, j)| (AXES[i] * entries[i][j]) * (AXES[j] | Vector::slot()))
            .sum()
    }

    /// For an ellipsoid along the axes, the spin in each plane is the fluid's, plus the strain
    /// rate in that plane times the difference over the sum of the squared semi-axes.
    #[test]
    fn spin_is_jefferys_formula_in_the_principal_frame() {
        let semi_axes = [3.0, 1.5, 0.5];
        let entries = [[0.3, -1.2, 0.4], [0.7, 0.1, -0.5], [-0.2, 0.9, -0.4]];
        let gradient = gradient_of(entries);
        let identity = Rotor::<(), f64>::new(1.0, 0.0, 0.0, 0.0).normalized();
        let spin = spin(placed(stretch(semi_axes), identity), gradient);
        let fluid = curl(gradient) * 0.5;
        for (first, second) in [(0, 1), (1, 2), (0, 2)] {
            let plane = AXES[first] ^ AXES[second];
            let relative = (spin - fluid).scalar_product(plane.reverse()).s();
            let strain = (entries[first][second] + entries[second][first]) / 2.0;
            let (a, b) = (semi_axes[first].powi(2), semi_axes[second].powi(2));
            let expected = (a - b) / (a + b) * strain;
            assert!((relative - expected).abs() < 1e-12, "{relative} {expected}");
        }
    }

    /// The axis of a very long needle turns as a line of fluid does: its velocity is the
    /// gradient on it, less its own stretching.
    #[test]
    fn needle_axis_moves_with_the_fluid() {
        let [x, y, z] = AXES;
        let gradient = x * (y | Vector::slot()) + (z * 0.3) * (x | Vector::slot());
        let turned = turn(x ^ z, 0.8) * turn(x ^ y, -0.6);
        let spin = spin(placed(stretch([1e4, 1.0, 1.0]), turned), gradient);
        let axis = turned >> x;
        let along = gradient.of(axis);
        let error = (axis | spin) - (along - axis * (axis | along).s());
        assert!(error.norm() < 1e-6, "{error:?}");
    }

    /// The rotors stay unit over the whole span: the bodies stay rigid.
    #[test]
    fn the_bodies_stay_rigid() {
        for turn in super::bodies().iter().flatten().flatten() {
            assert!((turn.into_inner().norm_squared() - 1.0).abs() < 1e-11);
        }
    }

    /// A sphere spins at the fluid's own rate, half the curl of the gradient.
    #[test]
    fn a_sphere_spins_with_the_fluid() {
        for turns in &super::bodies()[0] {
            let sphere = placed(stretch(SEMI_AXES[0]), turns[turns.len() - 1]);
            let error = spin(sphere, flow()) - curl(flow()) * 0.5;
            assert!(error.norm() < 1e-14, "{error:?}");
        }
    }

    /// The rod's axis is back where it started after one period, on a closed orbit along which
    /// Jeffery's orbit constant holds; the tolerances test the step. It spends most of its time
    /// near the flow's direction, where its spin nearly vanishes.
    #[test]
    fn the_rod_follows_a_closed_jeffery_orbit() {
        let [x, y, z] = AXES;
        let constant = |a: V| {
            let (flowing, across, spinning) = ((a | x).s(), (a | y).s(), (a | z).s());
            (flowing.powi(2) + ASPECT.powi(2) * across.powi(2))
                / (ASPECT.powi(2) * spinning.powi(2))
        };
        let (mut near, mut all) = (0, 0);
        for turns in &super::bodies()[1] {
            let axes: Vec<V> = turns.iter().map(|t| *t >> x).collect();
            let back = (axes[STEPS_PER_PERIOD] - axes[0]).norm();
            assert!(back < 2e-2, "{back}");
            for a in &axes {
                let ratio = constant(*a) / constant(axes[0]);
                assert!((ratio - 1.0).abs() < 2e-2, "{ratio}");
            }
            // The sine of the start's tilt: how far along the flow the axis starts.
            let sine = (axes[0] | x).s();
            near += axes
                .iter()
                .filter(|a| (**a | x).s().abs() > 0.8 * sine)
                .count();
            all += axes.len();
        }
        assert!(near * 2 > all, "{near} of {all}");
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws(super::draw, 3.0);
    }
}
