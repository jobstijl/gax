//! numga's `mechanics/manipulability`: the velocity and force ellipsoids at the gripper of the
//! scenegraph's robot arm, as quadrics in PGA3D built from the joint axes alone.
//!
//! Each joint turns about an axis, a line; carried into the world by its pivot, the axis is the
//! joint's twist per unit rate. Its commutator with the tip is the tip's velocity per unit rate,
//! a direction. The tip velocities reachable with unit joint rates fill an ellipsoid: as a dual
//! quadric, a map from planes to points, the sum of those directions' dyads about the tip. An
//! axis joined with the tip is the plane through both; a tip force within it loads the joint
//! not at all, and across it the joint's torque is the force read by that plane. The forces the
//! joints resist with unit torques fill an ellipsoid, a point quadric: the sum of those planes'
//! dyads. Each is the other's polar in the unit sphere about the tip, the principle of virtual
//! work. The animation swings the arm around a loop in joint space and ray-casts both: the
//! boxes are the unit cube pulled back through each body's map, the quadric's form bound to a
//! ray is a quadratic in its parameter.

#[path = "../shared/geometry_scenegraph.rs"]
mod scenegraph;

use gax_numga_examples::canvas::{mix, scale, srgb};
use gax_numga_examples::{Align, Anim, Canvas, Rgb, backdrop, caption, palette, run};

mod manipulability {
    use super::scenegraph::{self as arm, L, P, Pl, PointMap};
    use gax::pga3d::{Plane, Point};

    /// A point quadric takes each point to its polar plane.
    pub type Quadric = Plane<(Point,), f64>;
    /// A dual quadric takes each plane to its pole.
    pub type DualQuadric = Point<(Plane,), f64>;

    /// The plane at infinity.
    pub fn w() -> Pl {
        Plane::new(0.0, 0.0, 0.0, 1.0)
    }

    /// The coordinate planes through the origin, normal to `x`, `y` and `z`.
    pub fn coordinate_planes() -> [Pl; 3] {
        [
            Plane::new(1.0, 0.0, 0.0, 0.0),
            Plane::new(0.0, 1.0, 0.0, 0.0),
            Plane::new(0.0, 0.0, 1.0, 0.0),
        ]
    }

    /// The arm's body maps, its gripper tip, and each joint's axis in the world: the base yaws
    /// about `z`, the shoulder, elbow and wrist pitch about `y`, each axis the rotation's
    /// generator carried into the world by its pivot.
    pub fn arm_axes(angles: [f64; 4]) -> ([PointMap; 5], P, [L; 4]) {
        let (bodies, pivots) = arm::robot_arm(angles);
        // The top face of the gripper, on the wrist.
        let tip = pivots[3] >> arm::point([0.0, 0.0, 0.3]);
        let axes = [
            pivots[0] >> arm::xy(),
            pivots[1] >> arm::zx(),
            pivots[2] >> arm::zx(),
            pivots[3] >> arm::zx(),
        ];
        (bodies, tip, axes)
    }

    /// The tip velocities reachable with unit joint rates: the dyads of the velocities about the
    /// tip, `Σ v (Plane & v) - tip (Plane & tip)`. Any frame, the same one for tip and axes.
    pub fn velocity_ellipsoid(tip: P, axes: &[L]) -> DualQuadric {
        let plane = Plane::slot();
        let mut q = (tip * (plane & tip)).gp(-1.0);
        for a in axes {
            // The tip velocity per unit rate of this joint.
            let v = a.commutator(tip);
            q += v * (plane & v);
        }
        q
    }

    /// The tip forces resisted with unit joint torques: the dyads of the planes through each
    /// joint's axis and the tip, a point quadric, `Σ T (T & Point) - w (w & Point)`.
    pub fn force_ellipsoid(tip: P, axes: &[L]) -> Quadric {
        let x = Point::slot();
        let mut q = (w() * (w() & x)).gp(-1.0);
        for a in axes {
            // This joint's axis joined with the tip.
            let torque_free = *a & tip;
            q += torque_free * (torque_free & x);
        }
        q
    }

    /// The unit sphere about a point: the dyads of the coordinate planes through it (for the
    /// duality check).
    #[cfg(test)]
    pub fn unit_sphere(centre: P) -> Quadric {
        let x = Point::slot();
        let mut q = (w() * (w() & x)).gp(-1.0);
        for plane in coordinate_planes() {
            let through = plane - w().gp((plane & centre).s());
            q += through * (through & x);
        }
        q
    }

    /// The arm in one pose: body maps, tip, and both ellipsoids as point quadrics. The display
    /// scales grow the velocity ellipsoid and shrink the force ellipsoid, whose radii are
    /// reciprocal to it.
    pub struct Pose {
        pub bodies: [PointMap; 5],
        pub tip: P,
        pub velocity: Quadric,
        pub force: Quadric,
    }

    pub fn ellipsoids(angles: [f64; 4], velocity_scale: f64, force_scale: f64) -> Pose {
        let (bodies, tip, axes) = arm_axes(angles);
        // The velocity ellipsoid as a point quadric: the dual quadric's inverse.
        let velocity = velocity_ellipsoid(tip, &axes.map(|a| a.gp(velocity_scale))).inverse();
        let force = force_ellipsoid(tip, &axes.map(|a| a.gp(force_scale)));
        Pose {
            bodies,
            tip,
            velocity,
            force,
        }
    }

    /// Joint angles: base yaw, shoulder, elbow and wrist pitch, of numga's still figure (whose
    /// checks are the tests; the animation is the sweep).
    #[cfg(test)]
    pub const REACHING: [f64; 4] = [0.35, 0.5, 1.2, 0.4];
    #[cfg(test)]
    pub const NEARLY_STRAIGHT: [f64; 4] = [0.35, 0.2, 0.15, 0.1];

    /// The arm around a closed loop through joint space at `phase` (radians). Each joint swings
    /// about a centre, (centre, amplitude, frequency, phase); the elbow stays bent by at least
    /// half a radian, so the arm never straightens and neither ellipsoid degenerates.
    pub fn sweep(phase: f64) -> [f64; 4] {
        let swing = [
            (0.35, 0.9, 1.0, 0.0),
            (0.35, 0.35, 1.0, 1.5),
            (1.25, 0.75, 1.0, 0.3),
            (0.4, 0.8, 2.0, 0.0),
        ];
        swing.map(|(centre, amplitude, frequency, offset)| {
            centre + amplitude * (frequency * phase + offset).sin()
        })
    }
}

/// Implicit renders: every pixel's ray meets the arm's boxes, the floor and an ellipsoid;
/// the nearest box wins, and the ellipsoid lies over it, translucent, where the ray enters it
/// first.
mod render {
    use super::manipulability::{Pose, Quadric};
    use super::scenegraph::{P, Pl};
    use gax::pga3d::{Plane, Point};

    /// An orthographic view: the direction towards the viewer, the screen's right and up, and
    /// the lamp.
    #[derive(Clone, Copy)]
    pub struct View {
        pub centre: [f64; 3],
        pub extent: f64,
        pub back: [f64; 3],
        pub right: [f64; 3],
        pub up: [f64; 3],
        pub lamp: [f64; 3],
    }

    /// numga's fixed view of the workspace: elevation and azimuth in degrees, centre, half width.
    pub fn view(elevation: f64, azimuth: f64, centre: [f64; 3], extent: f64) -> View {
        let (tilt, turn) = (elevation.to_radians(), azimuth.to_radians());
        let back = [tilt.cos() * turn.cos(), tilt.cos() * turn.sin(), tilt.sin()];
        let right = [-turn.sin(), turn.cos(), 0.0];
        let up = [
            back[1] * right[2] - back[2] * right[1],
            back[2] * right[0] - back[0] * right[2],
            back[0] * right[1] - back[1] * right[0],
        ];
        let lamp: [f64; 3] = core::array::from_fn(|i| back[i] - 0.4 * right[i] + 0.7 * up[i]);
        let n = lamp.iter().map(|v| v * v).sum::<f64>().sqrt();
        View {
            centre,
            extent,
            back,
            right,
            up,
            lamp: lamp.map(|v| v / n),
        }
    }

    impl View {
        /// The ray at screen offsets `(u, v)` from the centre: its origin far out towards the
        /// viewer, and its heading.
        pub fn ray(&self, u: f64, v: f64) -> (P, P) {
            let o: [f64; 3] = core::array::from_fn(|i| {
                self.centre[i]
                    + u * self.right[i]
                    + v * self.up[i]
                    + 20.0 * self.extent * self.back[i]
            });
            (
                Point::xyz(o[0], o[1], o[2]),
                Point::direction(-self.back[0], -self.back[1], -self.back[2]),
            )
        }

        /// The screen offsets of a world point.
        pub fn screen(&self, p: [f64; 3]) -> [f64; 2] {
            let d: [f64; 3] = core::array::from_fn(|i| p[i] - self.centre[i]);
            let dot = |a: [f64; 3]| a[0] * d[0] + a[1] * d[1] + a[2] * d[2];
            [dot(self.right), dot(self.up)]
        }
    }

    /// A box, prepared for rays: its inverse body map, and the world planes of its slabs'
    /// faces for their normals (the slabs pushed through the induced map on planes).
    #[derive(Clone, Copy)]
    pub struct BoxShape {
        to_box: gax::pga3d::Point<(Point,), f64>,
        faces: [Pl; 3],
    }

    /// The unit cube lies within half a unit of each of these planes.
    fn slabs() -> [Pl; 3] {
        super::manipulability::coordinate_planes()
    }

    pub fn prepare(body: gax::pga3d::Point<(Point,), f64>) -> BoxShape {
        let to_box = body.inverse();
        // A world plane `F` with `F & x == slab & to_box(x)`: the slab's plane in the world.
        let on_planes = (Plane::slot() & Point::slot()).solve(Plane::slot() & to_box);
        BoxShape {
            to_box,
            faces: slabs().map(|s| on_planes.of(s)),
        }
    }

    impl BoxShape {
        /// The ray parameter of the first hit and the world normal of the face hit.
        pub fn hit(&self, origin: P, heading: P) -> Option<(f64, [f64; 3])> {
            let (o, h) = (self.to_box.of(origin), self.to_box.of(heading));
            let weight = o.e123();
            let (mut enter, mut leave, mut face) = (f64::NEG_INFINITY, f64::INFINITY, 0);
            for (i, slab) in slabs().iter().enumerate() {
                let start = (*slab & o).s() / weight;
                let rate = (*slab & h).s() / weight;
                let (low, high) = ((-0.5 - start) / rate, (0.5 - start) / rate);
                let (near, far) = (low.min(high), low.max(high));
                if near > enter {
                    (enter, face) = (near, i);
                }
                leave = leave.min(far);
            }
            (enter < leave && enter > 0.0).then(|| {
                let f = self.faces[face];
                (enter, [f.e1(), f.e2(), f.e3()])
            })
        }
    }

    /// The ray parameter where the ray enters the solid quadric, and the normal there, from its
    /// polar plane. Bound to the ray in both slots the form is a quadratic in the parameter.
    pub fn quadric_hit(q: Quadric, origin: P, heading: P) -> Option<(f64, [f64; 3])> {
        let qh = q.of(heading);
        let (a, b) = ((qh & heading).s(), (qh & origin).s());
        let c = (q.of(origin) & origin).s();
        let disc = b * b - a * c;
        if disc < 0.0 {
            return None;
        }
        let root = disc.sqrt();
        // The entering root is the smaller one, whichever sign the form has.
        let t = ((-b - root) / a).min((-b + root) / a);
        let polar = q.of(origin + heading.gp(t));
        Some((t, [polar.e1(), polar.e2(), polar.e3()]))
    }

    fn unit(v: [f64; 3]) -> [f64; 3] {
        let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt() + 1e-12;
        v.map(|x| x / n)
    }

    fn lit(n: [f64; 3], lamp: [f64; 3], ambient: f64) -> f64 {
        let n = unit(n);
        ambient + (1.0 - ambient) * (n[0] * lamp[0] + n[1] * lamp[1] + n[2] * lamp[2]).abs()
    }

    /// The colour along one ray: boxes opaque and shaded by how squarely their face meets the
    /// lamp, a floor tiled about the base, and the quadric over them where the ray enters it
    /// first. `None` where the ray meets nothing.
    pub fn shade(
        boxes: &[BoxShape],
        colours: &[[f32; 3]],
        quadric: Quadric,
        colour: [f32; 3],
        view: &View,
        u: f64,
        v: f64,
    ) -> Option<[f32; 3]> {
        let (origin, heading) = view.ray(u, v);
        let mut nearest: Option<(f64, [f32; 3])> = None;
        for (shape, col) in boxes.iter().zip(colours) {
            if let Some((t, n)) = shape.hit(origin, heading)
                && nearest.is_none_or(|(best, _)| t < best)
            {
                let k = lit(n, view.lamp, 0.35) as f32;
                nearest = Some((t, col.map(|x| x * k)));
            }
        }
        // The floor `z = 0`, tiled within 2.4 of the base.
        let floor = Plane::new(0.0, 0.0, 1.0, 0.0);
        let t = -(floor & origin).s() / (floor & heading).s();
        if t > 0.0 && nearest.is_none_or(|(best, _)| t < best) {
            let [x, y, _] = (origin + heading.gp(t)).to_euclidean();
            if x.abs() < 2.4 && y.abs() < 2.4 {
                let odd = ((x / 0.4).floor() + (y / 0.4).floor()).rem_euclid(2.0) > 0.5;
                let g = if odd { 0.03 } else { 0.045 };
                nearest = Some((t, [g, g * 1.05, g * 1.2]));
            }
        }
        let over = quadric_hit(quadric, origin, heading)
            .filter(|(t, _)| nearest.is_none_or(|(best, _)| *t < best));
        const ALPHA: f32 = 0.55;
        match (nearest, over) {
            (None, None) => None,
            (Some((_, c)), None) => Some(c),
            (behind, Some((_, n))) => {
                let k = lit(n, view.lamp, 0.45) as f32;
                let base = behind.map_or(super::PANEL, |(_, c)| c);
                Some(core::array::from_fn(|i| {
                    (1.0 - ALPHA) * base[i] + ALPHA * colour[i] * k
                }))
            }
        }
    }

    /// The pose's ellipsoid of either kind.
    pub fn quadric(pose: &Pose, velocity: bool) -> Quadric {
        if velocity { pose.velocity } else { pose.force }
    }
}

use manipulability::*;

const SECONDS: f32 = 8.0;
/// The panels' backdrop.
const PANEL: Rgb = [0.012, 0.014, 0.022];

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let phase = f64::from(t / SECONDS) * core::f64::consts::TAU;
    let pose = ellipsoids(sweep(phase), 0.25, 2.5);
    let view = render::view(18.0, -60.0, [-0.9, 0.25, 1.3], 2.2);
    let boxes: Vec<render::BoxShape> = pose.bodies.iter().map(|b| render::prepare(*b)).collect();
    let colours = [
        srgb(0.45, 0.47, 0.52),
        srgb(0.62, 0.64, 0.70),
        srgb(0.78, 0.60, 0.30),
        srgb(0.30, 0.55, 0.62),
        srgb(0.55, 0.40, 0.60),
    ];
    let (w, h) = (c.width as f32, c.height as f32);
    let top = h * 0.15;
    let side = (h - top - h * 0.06).min((w - w * 0.09) / 2.0);
    let gap = w - 2.0 * side - w * 0.06;
    let s = (h / 30.0).clamp(7.0, 22.0) * 0.62;
    for (k, (velocity, title, sub, colour)) in [
        (
            true,
            "VELOCITY ELLIPSOID",
            "TIP SPEEDS AT UNIT JOINT RATES",
            srgb(0.25, 0.45, 0.85),
        ),
        (
            false,
            "FORCE ELLIPSOID",
            "TIP FORCES AT UNIT JOINT TORQUES",
            srgb(0.92, 0.55, 0.20),
        ),
    ]
    .into_iter()
    .enumerate()
    {
        let x0 = w * 0.03 + k as f32 * (side + gap);
        let rect = [x0, top, x0 + side, top + side];
        c.fill(
            &[
                [rect[0], rect[1]],
                [rect[2], rect[1]],
                [rect[2], rect[3]],
                [rect[0], rect[3]],
            ],
            PANEL,
            1.0,
        );
        let q = render::quadric(&pose, velocity);
        let extent = view.extent;
        let to_screen = move |x: f32, y: f32| {
            (
                (f64::from((x - rect[0]) / side) * 2.0 - 1.0) * extent,
                (1.0 - f64::from((y - rect[1]) / side) * 2.0) * extent,
            )
        };
        c.clip(rect);
        let boxes = &boxes;
        c.shade(2, |x, y| {
            let (u, v) = to_screen(x, y);
            render::shade(boxes, &colours, q, colour, &view, u, v)
        });
        // The tip.
        let [u, v] = view.screen(pose.tip.to_euclidean());
        let px = [
            rect[0] + ((u / extent + 1.0) * 0.5) as f32 * side,
            rect[1] + ((1.0 - v / extent) * 0.5) as f32 * side,
        ];
        c.disk(px, 2.5, palette::ink(), 1.0);
        c.unclip();
        c.polyline(
            &[
                [rect[0], rect[1]],
                [rect[2], rect[1]],
                [rect[2], rect[3]],
                [rect[0], rect[3]],
            ],
            1.0,
            palette::grid(),
            1.0,
            true,
        );
        c.text(
            title,
            rect[0] + s * 0.6,
            rect[1] + s * 1.5,
            s,
            mix(colour, palette::ink(), 0.4),
            Align::Left,
        );
        c.text(
            sub,
            rect[0] + s * 0.6,
            rect[1] + s * 2.8,
            s * 0.75,
            scale(palette::ink(), 0.6),
            Align::Left,
        );
    }
    caption(
        c,
        "MANIPULABILITY: VELOCITY AND FORCE ELLIPSOIDS AT THE GRIPPER",
        "QUADRICS FROM THE JOINT AXES, EACH THE OTHER'S POLAR IN THE UNIT SPHERE (PGA3D)",
    );
}

fn main() {
    run(
        Anim::new("manipulability", SECONDS).size(640, 360).scale(2),
        draw,
    );
}

#[cfg(test)]
mod tests {
    use super::manipulability::*;
    use super::scenegraph::point;
    use gax::pga3d::Point;

    /// numga's checks inside `ellipsoids`, for one pose.
    fn checks(angles: [f64; 4]) {
        let (_, tip, axes) = arm_axes(angles);
        let unit_velocity = velocity_ellipsoid(tip, &axes);
        let unit_force = force_ellipsoid(tip, &axes);
        // Duality: the force ellipsoid is the velocity ellipsoid's polar in the unit sphere
        // about the tip.
        let sphere = unit_sphere(tip);
        let polar = sphere.of(unit_velocity.of(sphere));
        for (a, b) in unit_force.c.iter().flatten().zip(polar.c.iter().flatten()) {
            assert!(
                (a - b).abs() <= 1e-8 + 1e-8 * b.abs(),
                "{unit_force:?} vs {polar:?}"
            );
        }
        // Both against the joint torques a unit tip force produces: each joint's axis paired
        // with the force's line through the tip.
        for f in [[0.3, -0.2, 0.5], [1.0, 0.4, -0.7]] {
            let n = (f[0] * f[0] + f[1] * f[1] + f[2] * f[2]) as f64;
            let along = Point::direction(f[0], f[1], f[2]).gp(1.0 / n.sqrt());
            let torques = axes.map(|a| (a & (tip & along)).s());
            let squared: f64 = torques.iter().map(|t| t * t).sum();
            // A force lies on the force ellipsoid exactly when its joint torques have unit
            // squared sum.
            let held = tip + along;
            let on = (held & unit_force.of(held)).s();
            assert!(
                (on - (1.0 - squared)).abs() < 1e-8,
                "{on} vs {}",
                1.0 - squared
            );
            // The velocity ellipsoid reaches, along a direction, as far as a unit force that
            // way loads the joints: the plane normal to it at that distance is tangent.
            let reach = squared.sqrt();
            let normal = gax::pga3d::Plane::orthogonal_to(along);
            let tangent = normal - w().gp((normal & (tip + along.gp(reach))).s());
            let touch = (tangent & unit_velocity.of(tangent)).s();
            assert!(touch.abs() < 1e-8, "{touch}");
        }
    }

    /// Both poses of numga's figure pass the checks.
    #[test]
    fn figure_poses_pass_their_checks() {
        checks(REACHING);
        checks(NEARLY_STRAIGHT);
        let pose = ellipsoids(REACHING, 0.25, 2.5);
        let _ = (pose.velocity, pose.force);
    }

    /// The sweep's poses pass them too, around the whole loop.
    #[test]
    fn the_sweep_passes_its_checks() {
        for k in 0..12 {
            checks(sweep(core::f64::consts::TAU * k as f64 / 12.0));
        }
        let (a, b) = (sweep(0.0), sweep(core::f64::consts::TAU));
        assert!(a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-12));
    }

    /// The tip and the tip velocities per unit joint rate agree with numga's (whose points
    /// carry the opposite sign, ADR-009, so the velocities are compared up to it). numga's
    /// own coordinates of one point differ between its bodies by about 1e-11.
    #[test]
    fn tip_and_velocities_agree_with_numga() {
        let (_, tip, axes) = arm_axes(REACHING);
        let want = [-1.715236583157, 0.626110228226, 1.37280074864];
        let got = tip.to_euclidean();
        assert!(
            got.iter().zip(want).all(|(a, b)| (a - b).abs() < 1e-9),
            "{got:?}"
        );
        // The same tip from a chain of 4 x 4 rotation and translation matrices in numpy: numga's
        // exponentials are what is off by 1e-11.
        let exact = [-1.715236583149619, 0.626110228225186, 1.372800748592965];
        assert!(got.iter().zip(exact).all(|(a, b)| (a - b).abs() < 1e-14));
        let velocities = [
            [0.626110228227, 1.715236583158, 0.0],
            [-0.725947935741, 0.264991682308, -1.825938266781],
            [0.263304598555, -0.096113681292, -1.25062762046],
            [0.142271596446, -0.051933186708, -0.258962810002],
        ];
        let sign = -axes[0].commutator(tip).e032() / velocities[0][0];
        assert!((sign.abs() - 1.0).abs() < 1e-9);
        for (a, want) in axes.iter().zip(velocities) {
            let v = a.commutator(tip);
            assert!(v.e123().abs() < 1e-12);
            for (got, want) in [v.e032(), v.e013(), v.e021()].into_iter().zip(want) {
                assert!((got + sign * want).abs() < 1e-9, "{v:?}");
            }
        }
        let _ = point([0.0; 3]);
    }

    /// Rays enter the ellipsoids where the form changes sign, and boxes are hit where the
    /// pulled-back ray crosses the cube's faces.
    #[test]
    fn rays_meet_the_ellipsoid_and_the_boxes() {
        let pose = ellipsoids(REACHING, 0.25, 2.5);
        let tip = pose.tip.to_euclidean();
        let heading = Point::direction(0.0, 1.0, 0.0);
        let origin = point([tip[0], tip[1] - 30.0, tip[2]]);
        for q in [pose.velocity, pose.force] {
            let (t, _) = super::render::quadric_hit(q, origin, heading).expect("a hit");
            let p = origin + heading.gp(t);
            assert!((p & q.of(p)).s().abs() < 1e-9);
            assert!(t > 0.0 && t < 30.0);
        }
        // The pedestal from straight above: its top face at z = 0.25.
        let shape = super::render::prepare(pose.bodies[0]);
        let (t, n) = shape
            .hit(point([0.1, 0.1, 5.0]), Point::direction(0.0, 0.0, -1.0))
            .expect("the pedestal");
        assert!((t - 4.75).abs() < 1e-12);
        assert!(n[0].abs() < 1e-12 && n[1].abs() < 1e-12 && n[2].abs() > 0.0);
    }

    #[test]
    fn a_frame_draws() {
        let mut draw = super::draw;
        let c = gax_numga_examples::app::frame(
            &gax_numga_examples::Anim::new("t", 1.0).size(160, 90),
            0.5,
            &mut draw,
        );
        assert!(c.mean()[0] > 0.0);
    }
}
