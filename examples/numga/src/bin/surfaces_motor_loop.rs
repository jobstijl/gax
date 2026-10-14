//! numga's `surfaces/motor_loop`: Loop's subdivision of motors, in PGA3D.
//!
//! A mesh can carry a motor at every vertex, a rigid placement of a frame, and the Loop step
//! averages them like anything else at the sites: the same field map of shares
//! ([`Mesh::loop_step`]), applied to the motors as they are. An average of motors is not a
//! motor, so each step is followed by normalizing every averaged motor back to a motor, and the
//! next step starts from there: average in the algebra, return to the group. Between two frames
//! turned about an axis, the average is the frame turned halfway, on the arc between them; the
//! average of the two positions cuts across the chord instead.
//!
//! A ribbon down the `z` axis carries a frame at each vertex of its control net, its corners
//! pinned. Wound up by a twist that grows towards its far end, the surface refined from the
//! points the motors carry wrings thin towards the axis, while the surface refined from the
//! motors keeps its width (first pair of panels). Held flat with every frame turning back and
//! forth about its own origin, the points do not move at all, and only the turns between the
//! frames bend the motor surface (second pair). Each panel shows the control net and the frames'
//! axes, `x` red, `y` green and `z` blue.

#[path = "../shared/surfaces_loop.rs"]
mod surfaces_loop;
#[path = "../shared/surfaces_panels.rs"]
mod surfaces_panels;

use gax::field::Field;
use gax_numga_examples::scene3::panel3;
use gax_numga_examples::{Anim, Canvas, Rect, Scene3, backdrop, caption, palette, run};
use surfaces_loop::{Point64, framing, net, separation, surface};
use surfaces_panels::{label_under, row};

mod motors {
    use super::surfaces_loop::{Point64, refine};
    use gax::field::{Field, SparseMap};
    use gax::pga3d::{Motor, Scalar};
    use gax_numga_examples::mesh::{Mesh, Vector3};

    /// A motor with `f64` coefficients.
    pub type Motor64 = Motor<(), f64>;

    /// So many Loop steps on a mesh of motors, the `pinned` vertices staying where they are:
    /// the steps composed into one map, and the finer mesh, every averaged motor normalized back
    /// to a motor after each step.
    pub fn subdivision(
        mesh: &Mesh<Motor64>,
        pinned: &[bool],
        depth: usize,
    ) -> (SparseMap<Scalar<(), f64>>, Mesh<Motor64>) {
        refine(mesh, pinned, depth, |m| m.normalized().into_inner())
    }

    /// The turn by an angle about an axis through the origin along a direction.
    pub fn turn(axis: Vector3, angle: f64) -> Motor64 {
        Motor::rotation_about(axis.e1(), axis.e2(), axis.e3(), angle).into_inner()
    }

    /// The frame on a ribbon down the `z` axis: slid along it and out across it along `x`, then
    /// turned about the axis by an angle.
    pub fn ribbon(along: f64, across: f64, angle: f64) -> Motor64 {
        let slide = |x, z| Motor::translation(x, 0.0, z).into_inner();
        turn(Vector3::new(0.0, 0.0, 1.0), angle) * slide(0.0, along) * slide(across, 0.0)
    }

    /// The origin of every frame, where its motor takes the origin.
    pub fn origins(motors: &Field<Motor64>) -> Field<Point64> {
        let origin = Point64::xyz(0.0, 0.0, 0.0);
        motors.map(|m| m >> origin)
    }
}

mod scenes {
    use super::motors::*;
    use super::surfaces_loop::Point64;
    use gax::field::Field;
    use gax_numga_examples::mesh::{Mesh, Vector3};
    use gax_numga_examples::rng::{Draw, rng};
    use gax_numga_examples::signal::{phasor, wave};

    /// The ribbon's control mesh: its vertices along and across, its length and width; its four
    /// corners are pinned.
    pub const ALONG_COUNT: usize = 5;
    pub const ACROSS_COUNT: usize = 3;
    pub const LENGTH: f64 = 4.0;
    pub const WIDTH: f64 = 1.2;
    /// The turn of the far end at the top of the wind-up, the near end held, the twist building
    /// towards the far end as the square of the distance.
    pub const WOUND: f64 = 3.0 * core::f64::consts::PI;
    /// How far each frame turns either way about its own axis as the ribbon lies flat, and the
    /// seed of the axes and of the phases of the turns.
    pub const TURNING: f64 = core::f64::consts::PI / 3.0;
    pub const TURNING_SEED: u64 = 2;
    /// Loop steps taken, and the length of the drawn frame axes.
    pub const DEPTH: usize = 3;
    pub const AXIS: f64 = 0.3;

    /// How far along and across the ribbon each vertex of its control net lies, row by row.
    pub fn grid() -> Vec<(f64, f64)> {
        let spread =
            |i: usize, count: usize, size: f64| size * (i as f64 / (count - 1) as f64 - 0.5);
        (0..ALONG_COUNT)
            .flat_map(|i| {
                (0..ACROSS_COUNT).map(move |j| {
                    (
                        spread(i, ALONG_COUNT, LENGTH),
                        spread(j, ACROSS_COUNT, WIDTH),
                    )
                })
            })
            .collect()
    }

    /// The control net's faces, row by row, two triangles per cell, all turning the same way.
    pub fn grid_faces() -> Vec<[usize; 3]> {
        let at = |i: usize, j: usize| i * ACROSS_COUNT + j;
        let cells = (0..ALONG_COUNT - 1).flat_map(|i| (0..ACROSS_COUNT - 1).map(move |j| (i, j)));
        let corner = cells
            .clone()
            .map(|(i, j)| [at(i, j), at(i, j + 1), at(i + 1, j + 1)]);
        let across = cells.map(|(i, j)| [at(i, j), at(i + 1, j + 1), at(i + 1, j)]);
        corner.chain(across).collect()
    }

    /// The four corners of the control net, pinned.
    pub fn pinned() -> Vec<bool> {
        let last = ALONG_COUNT * ACROSS_COUNT - 1;
        let corners = [0, ACROSS_COUNT - 1, last + 1 - ACROSS_COUNT, last];
        (0..=last).map(|v| corners.contains(&v)).collect()
    }

    /// The control frames at a phase of the wind-up and release: the far end turned by
    /// `WOUND (1 - cos phase) / 2`, each frame by that times the square of its share of the way
    /// along.
    pub fn winding(phase: f64) -> Field<Motor64> {
        let wound = WOUND * (1.0 - phasor(phase).e20()) / 2.0;
        let held = |along: f64| ((along + LENGTH / 2.0) / LENGTH).powi(2);
        grid()
            .into_iter()
            .map(|(along, across)| ribbon(along, across, wound * held(along)))
            .collect()
    }

    /// Each frame's axis and phase for turning back and forth.
    pub struct Turning {
        pub axes: Vec<Vector3>,
        pub phases: Vec<f64>,
    }

    impl Turning {
        pub fn new() -> Turning {
            let mut draws = rng(TURNING_SEED);
            let count = ALONG_COUNT * ACROSS_COUNT;
            let axes = (0..count).map(|_| draws.direction()).collect();
            let phases = (0..count)
                .map(|_| draws.range(0.0, core::f64::consts::TAU))
                .collect();
            Turning { axes, phases }
        }

        /// The ribbon held flat at a phase, every frame turned about its own axis through its
        /// origin by `TURNING sin(phase + its own phase)`.
        pub fn at(&self, phase: f64) -> Field<Motor64> {
            let turned = grid().into_iter().enumerate().map(|(v, (along, across))| {
                let angle = TURNING * wave(phase + self.phases[v]);
                ribbon(along, across, 0.0) * turn(self.axes[v], angle)
            });
            turned.collect()
        }
    }

    /// The control frames, and the surface refined two ways: as the points the motors carry,
    /// through the steps composed into one map, and as the motors themselves, normalized after
    /// every step.
    pub struct Refined {
        pub control: Mesh<Motor64>,
        pub points: Field<Point64>,
        pub motors: Mesh<Motor64>,
    }

    pub fn refined(motors: Field<Motor64>) -> Refined {
        let control = Mesh::new(motors, grid_faces());
        let (subdivide, refined) = subdivision(&control, &pinned(), DEPTH);
        Refined {
            points: &subdivide * &origins(&control.vertices),
            motors: refined,
            control,
        }
    }

    /// The ends of each frame's x, y and z axes, `AXIS` long, from its origin.
    pub fn axis_ends(motors: &Field<Motor64>) -> [Field<Point64>; 3] {
        let ends = [[AXIS, 0.0, 0.0], [0.0, AXIS, 0.0], [0.0, 0.0, AXIS]];
        ends.map(|[x, y, z]| motors.map(|m| m >> Point64::xyz(x, y, z)))
    }
}

use motors::origins;
use scenes::*;

/// One wind-up and release, and one turn back and forth: numga's 48 frames of 60 ms.
const SECONDS: f32 = 2.88;
/// The view: numga's elevation and azimuth.
const ELEVATION: f32 = 0.21;
const AZIMUTH: f32 = -1.05;
/// The width of the frames' axes.
const AXIS_WIDTH: f32 = 2.0;

/// One refined surface over the control net, with the control frames.
fn panel(c: &mut Canvas, rect: Rect, refined: &Refined, points: &Field<Point64>, label: &str) {
    let control = &refined.control;
    let origins = origins(&control.vertices);
    let cam = framing(rect, origins.sites(), AZIMUTH, ELEVATION);
    panel3(c, rect, cam, |s: &mut Scene3| {
        surface(s, points, &refined.motors.faces, |_| {
            palette::sky().faded(0.5)
        });
        net(s, &origins, &control.edges);
        let lights = [palette::red(), palette::green(), palette::blue()];
        for (ends, light) in axis_ends(&control.vertices).iter().zip(lights) {
            for (o, e) in origins.iter().zip(ends.iter()) {
                s.seg(o, e, AXIS_WIDTH, light);
            }
        }
    });
    label_under(c, rect, label);
}

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let phase = core::f64::consts::TAU * f64::from(t / SECONDS);
    let panels: [Rect; 4] = row(c, 4.0);
    let wound = refined(winding(phase));
    let turning = refined(Turning::new().at(phase));
    let scenes = [(&wound, "WOUND"), (&turning, "TURNING")];
    for ((scene, name), pair) in scenes.into_iter().zip(panels.chunks(2)) {
        let carried = origins(&scene.motors.vertices);
        let apart = separation(&carried, &scene.points);
        let label = format!("{name}: POINTS");
        panel(c, pair[0], scene, &scene.points, &label);
        let label = format!("{name}: MOTORS, {apart:.2} OFF");
        panel(c, pair[1], scene, &carried, &label);
    }
    caption(
        c,
        "MOTOR LOOP: SUBDIVIDING FRAMES",
        "LOOP STEPS AS SPARSE MAPS ON PGA3D MOTORS, EACH AVERAGE NORMALIZED BACK TO A MOTOR",
    );
}

fn main() {
    run(Anim::new("motor loop", SECONDS).size(960, 540), draw);
}

#[cfg(test)]
mod tests {
    use super::motors::*;
    use super::scenes::*;
    use super::surfaces_loop::{Point64, separation};
    use gax::field::Field;
    use gax::pga3d::Point;
    use gax_numga_examples::mesh::Vector3;

    /// numga's frames of one wind-up, or one turn.
    const FRAMES: usize = 48;

    fn phase(frame: usize) -> f64 {
        core::f64::consts::TAU * frame as f64 / FRAMES as f64
    }

    /// The mean distance of points from the `z` axis: the norm of the plane joining the axis to
    /// each point.
    fn reach(points: &Field<Point64>) -> f64 {
        let axis = Point::xyz(0.0, 0.0, 0.0) & Point::direction(0.0, 0.0, 1.0);
        let distances = points.iter().map(|p| (axis & p.unitized()).norm());
        distances.sum::<f64>() / points.len() as f64
    }

    #[test]
    fn motors_keep_the_twisted_ribbon_wide_where_points_wring_it_thin() {
        let [rest, peak] = [0, FRAMES / 2].map(|frame| {
            let surfaces = refined(winding(phase(frame)));
            [
                reach(&surfaces.points),
                reach(&origins(&surfaces.motors.vertices)),
            ]
        });
        // Untwisted the two agree; wound up, the points wring the ribbon towards its axis and
        // the motors do not.
        assert!((rest[0] - rest[1]).abs() < 1e-12);
        assert!(
            peak[0] < 0.7 * rest[0] && peak[1] >= rest[1],
            "{rest:?} {peak:?}"
        );
        // No random draws enter the wind-up, so the reaches are numga's, to its printed digits.
        let numga = [[0.31764706, 0.31764706], [0.2189553, 0.34936678]];
        for (ours, theirs) in [rest, peak].iter().flatten().zip(numga.iter().flatten()) {
            assert!((ours - theirs).abs() < 1e-8, "{ours} {theirs}");
        }
        gax_numga_examples::app::assert_draws(super::draw, 0.0);
    }

    #[test]
    fn only_the_turns_between_frames_bend_the_motor_surface() {
        let turning = Turning::new();
        let frames: Vec<Refined> = (0..FRAMES).map(|f| refined(turning.at(phase(f)))).collect();
        // Turning the frames about their origins leaves the points alone and bends the motors.
        let first = &frames[0].points;
        let bent = |r: &Refined| separation(&origins(&r.motors.vertices), &r.points);
        assert!(frames.iter().all(|r| separation(&r.points, first) < 1e-12));
        assert!(frames.iter().map(bent).fold(0.0, f64::max) > 0.05);
        // A turn common to every frame changes nothing.
        let flat: Field<Motor64> = grid()
            .into_iter()
            .map(|(along, across)| ribbon(along, across, 0.0))
            .collect();
        let common = flat.map(|m| m * turn(Vector3::new(1.0, 0.0, 0.0), 0.7));
        let [plain, turned] =
            [flat, common].map(|motors| origins(&refined(motors).motors.vertices));
        assert!(separation(&plain, &turned) < 1e-12);
    }
}
