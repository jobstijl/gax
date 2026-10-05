//! numga's `geometry/scenegraph`: a robot arm photographed by a two-lens camera, as one PGA3D
//! point map per body. Each body is the unit box shaped by an anisotropic scaling and placed by
//! its chain of joint motors (`Point <- Point` maps composed with motors); the camera pulls the
//! world into its frame, joins each point with the pupil into a ray, refracts the ray through
//! two thin lenses (maps on lines), meets the sensor and goes through the viewport to pixels.
//! All of it collapses into one map per body, from the canonical box straight to pixels. The
//! animation swings the arm through a loop of joint angles: the scene with the camera's optics
//! and three traced rays on the left, the photograph on the sensor on the right.

#[path = "../shared/geometry_scenegraph.rs"]
mod scenegraph;

use gax::pga3d::{Motor, Plane, Point};
use gax_numga_examples::points::box_map;
use gax_numga_examples::{
    Align, Anim, Camera, Canvas, Lens, Light, Marker, ORIGIN2, Point2, Rect, Scene3, backdrop,
    caption, from_above, palette, run, signal::wave,
};
use scenegraph::{BOX_FACES, M, P, axis};

mod scene {
    use super::scenegraph::*;
    use gax::pga3d::{Plane, Point};
    use gax_numga_examples::signal::phasor;

    /// The compound camera: pose, front (objective) and rear (relay) lens, the rear lens plane,
    /// pupil, sensor plane, and the world-to-pixel map.
    pub struct Rig {
        pub pose: M,
        pub front: LineMap,
        pub rear: LineMap,
        pub rear_plane: Pl,
        pub pupil: P,
        pub sensor: Pl,
        pub world_to_pixel: PointMap,
    }

    /// The sensor in pixels, and in metres.
    pub const PIXELS: [f64; 2] = [640.0, 480.0];
    pub const CHIP: [f64; 2] = [1.6, 1.2];

    pub fn camera_rig() -> Rig {
        let pose = look_at(Point::xyz(0.0, -3.6, 1.3), Point::xyz(0.0, 0.0, 1.18));
        // The objective at the origin and the relay lens 0.3 behind it.
        let (front, rear, rear_plane) = lens_train(1.0, 0.8, -0.3);
        // The pupil point on the entrance aperture, off the optical centre so that rays bend.
        let pupil = Point::xyz(0.04, 0.0, 0.0);
        // The sensor plane `z = -1.25`.
        let sensor = Plane::from_normal([0.0, 0.0, 1.0], -1.25);
        let camera = lens_camera(pose, front, rear, pupil, sensor);
        // The 1.6 x 1.2 chip onto 640 x 480 pixels.
        let world_to_pixel = viewport(PIXELS[0], PIXELS[1], CHIP[0], CHIP[1]).of(camera);
        Rig {
            pose,
            front,
            rear,
            rear_plane,
            pupil,
            sensor,
            world_to_pixel,
        }
    }

    /// One photograph: the bodies' vertices in the world and on the sensor in pixels, and rays
    /// traced from gripper corners.
    pub struct Photo {
        pub world: [[P; 8]; 5],
        pub pixels: [[P; 8]; 5],
        pub rays: Vec<[P; 4]>,
    }

    /// The arm posed by `angles`, photographed; rays from the gripper `corners`.
    pub fn photograph(rig: &Rig, angles: [f64; 4], corners: &[usize]) -> Photo {
        let unit_box = canonical_unit_box();
        let (bodies, _) = robot_arm(angles);
        // The whole pipeline, kinematics to pixels, as one map per body.
        let local_to_pixel = bodies.map(|b| rig.world_to_pixel.of(b));
        let pixels = local_to_pixel.map(|m| unit_box.map(|v| project(m, v)));
        let world = bodies.map(|b| unit_box.map(|v| b.of(v)));
        let rays = corners
            .iter()
            .map(|&k| {
                trace_ray(
                    world[4][k],
                    rig.pose,
                    rig.front,
                    rig.rear,
                    rig.rear_plane,
                    rig.pupil,
                    rig.sensor,
                )
            })
            .collect();
        Photo {
            world,
            pixels,
            rays,
        }
    }

    /// numga's still: the arm posed once, three gripper corners traced (the animation is its
    /// sweep; the still is checked in the tests).
    #[cfg(test)]
    pub fn scenegraph(rig: &Rig) -> Photo {
        photograph(rig, [0.35, -0.45, 0.85, -0.40], &[7, 6, 2])
    }

    /// The looping joint trajectory at phase `t` (radians): the joints swing with the phasor at
    /// `t` (its height and its reach across) and the wrist with the one at `2 t`.
    pub fn sweep(t: f64) -> [f64; 4] {
        let (once, twice) = (phasor(t), phasor(2.0 * t));
        [
            0.45 * once.e01(),
            -0.40 + 0.25 * once.e20(),
            0.85 + 0.35 * once.e01(),
            -0.45 + 0.25 * twice.e20(),
        ]
    }
}

use scene::*;

const SECONDS: f32 = 6.0;

fn body_colour(i: usize) -> Light {
    [
        Light::from_srgb(0.29, 0.333, 0.408, 1.4),
        Light::from_srgb(0.169, 0.424, 0.69, 1.4),
        Light::from_srgb(0.192, 0.51, 0.808, 1.4),
        Light::from_srgb(0.259, 0.6, 0.882, 1.4),
        Light::from_srgb(0.929, 0.537, 0.212, 1.4),
    ][i]
}

fn ray_colour(i: usize) -> Light {
    [
        Light::from_srgb(0.898, 0.243, 0.243, 1.7),
        Light::from_srgb(0.22, 0.631, 0.412, 1.7),
        Light::from_srgb(0.839, 0.62, 0.18, 1.7),
    ][i % 3]
}

/// A face's edges: one faint stroke around it.
fn edge_light() -> Light {
    palette::grid().faded(0.8)
}

/// A circle of `radius` at height `z` in the camera frame, in the world: a point of it turned
/// about the camera's axis.
fn rim(pose: M, radius: f64, z: f64) -> Vec<P> {
    let start = Point::xyz(radius, 0.0, z);
    (0..=48)
        .map(|k| {
            let turn = Motor::rotation(
                axis(0.0, 0.0, 1.0),
                core::f64::consts::TAU * k as f64 / 48.0,
            );
            (pose * turn) >> start
        })
        .collect()
}

/// The scene in 3D: floor, arm, the camera's lens rims and sensor, and the traced rays.
fn scene_3d(c: &mut Canvas, t: f32, rig: &Rig, photo: &Photo) {
    let sway = 0.12 * wave(core::f32::consts::TAU * t / SECONDS);
    let cam = Camera::orbit(
        c.rect(),
        Point::xyz(0.1, -2.4, 1.25),
        15.5,
        (-55f32).to_radians() + sway,
        18f32.to_radians(),
        Lens::Perspective(0.42),
    );
    let mut s = Scene3::new(cam);
    let floor = |x: f64, y: f64| Point::xyz(x, y, 0.0);
    for k in 0..7 {
        let g = -1.2 + 0.4 * f64::from(k);
        let line = (palette::grid().mix_light(palette::ink(), 0.25)).faded(0.8);
        s.seg(floor(g, -1.2), floor(g, 1.2), 1.0, line);
        s.seg(floor(-1.2, g), floor(1.2, g), 1.0, line);
    }
    for (i, v) in photo.world.iter().enumerate() {
        for f in BOX_FACES {
            let [a, b, cc, d] = f.map(|k| v[k]);
            let col = s.lit(a, b, cc, body_colour(i));
            s.quad(a, b, cc, d, col, 1.0);
            s.polyline(&[a, b, cc, d, a], 0.8, edge_light());
        }
    }
    // The front lens rim (radius 0.3 at z = 0) and the rear one (0.25 at z = -0.3).
    let (front, rear) = (
        Light::from_srgb(0.192, 0.592, 0.584, 1.6),
        Light::from_srgb(0.502, 0.353, 0.835, 1.6),
    );
    s.polyline(&rim(rig.pose, 0.3, 0.0), 2.0, front);
    s.polyline(&rim(rig.pose, 0.25, -0.3), 2.0, rear);
    // The 1.6 x 1.2 sensor at z = -1.25.
    let corner = |x: f64, y: f64| rig.pose >> Point::xyz(x, y, -1.25);
    let chip = [
        corner(-0.8, -0.6),
        corner(0.8, -0.6),
        corner(0.8, 0.6),
        corner(-0.8, 0.6),
    ];
    let orange = Light::from_srgb(0.867, 0.42, 0.125, 1.6);
    let [a, b, cc, d] = chip;
    s.quad(a, b, cc, d, orange, 0.55);
    s.polyline(&[a, b, cc, d, a], 1.5, orange);
    // The rays: scene point to pupil, pupil to the rear lens, rear lens to the sensor.
    for (i, p) in photo.rays.iter().enumerate() {
        for (leg, width) in [1.4, 1.8, 1.8].into_iter().enumerate() {
            s.seg(p[leg], p[leg + 1], width, (ray_colour(i)).faded(0.9));
        }
        s.dot(p[3], Marker::Dot, 6.0, ray_colour(i));
    }
    s.draw(c);
}

/// The photograph: the sensor's pixels in `rect`, bodies painted from the base to the gripper,
/// each face shaded by how squarely its plane in the world faces a light from the camera's
/// upper left: the inner product of the planes, over the face's norm.
fn photograph_panel(c: &mut Canvas, rect: Rect, photo: &Photo) {
    // The sensor's pixels onto the panel: its corners onto the panel's (both with rows growing
    // downward), the sensor point seen along its `z`.
    let sensor = Point2::xy(PIXELS[0] as f32, PIXELS[1] as f32);
    let to_panel = box_map([ORIGIN2, sensor], [rect.lo, rect.hi]);
    let px = |p: P| to_panel.of(from_above(p)).unitized();
    let frame = [rect.lo, rect.top_right(), rect.hi, rect.bottom_left()];
    c.fill(&frame, Light::from_srgb(0.11, 0.12, 0.16, 1.0), 1.0);
    c.clip(rect);
    let facing = Plane::orthogonal_to(Point::direction(-0.4, -0.8, 0.45))
        .normalized()
        .into_inner();
    for (i, (pixels, w)) in photo.pixels.iter().zip(&photo.world).enumerate() {
        for f in BOX_FACES {
            let poly: Vec<Point2> = f.iter().map(|&j| px(pixels[j])).collect();
            let face = w[f[0]] & w[f[1]] & w[f[3]];
            let lit = 0.55 + 0.45 * ((face | facing).s() / face.norm().max(1e-9)).abs() as f32;
            c.fill(&poly, (body_colour(i)).faded(lit), 0.88);
            c.polyline(&poly, 1.0, edge_light(), true);
        }
        for p in pixels {
            c.disk(px(*p), 1.6, palette::ink().faded(0.9));
        }
    }
    c.unclip();
    c.polyline(&frame, 1.2, palette::grid(), true);
}

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let screen = c.rect();
    let (w, h) = (screen.width(), screen.height());
    let rig = camera_rig();
    let phase = f64::from(t / SECONDS) * core::f64::consts::TAU;
    let photo = photograph(&rig, sweep(phase), &[7, 6, 2]);
    // The scene on the left, drawn on its own canvas of the full height (so that the backdrop
    // continues) and set in.
    let left = (w * 0.58).floor();
    let mut sub = c.sub(Rect::new(0.0, 0.0, left, h));
    backdrop(&mut sub);
    scene_3d(&mut sub, t, &rig, &photo);
    c.blit(&sub, screen.lo);
    // The photograph on the right.
    let x0 = left + w * 0.01;
    let x1 = w - w * 0.03;
    let ph = (x1 - x0) * 0.75;
    let y0 = (h - ph) * 0.5 + h * 0.03;
    let rect = Rect::new(x0, y0, x1, y0 + ph);
    photograph_panel(c, rect, &photo);
    let s = (h / 45.0).clamp(7.0, 12.0);
    let down = Point2::direction(0.0, 1.0);
    c.text(
        "ON THE SENSOR, 640 X 480 PIXELS",
        rect.lo - down.gp(s * 0.8),
        s,
        palette::ink(),
        Align::Left,
    );
    for (k, line) in [
        "BOX TO PIXEL: VIEWPORT . SENSOR . LENSES",
        "  . PUPIL . POSE . JOINTS . SCALE",
    ]
    .into_iter()
    .enumerate()
    {
        let at = rect.bottom_left() + down.gp(s * (1.8 + 1.2 * k as f32));
        c.text(line, at, s * 0.8, palette::grid(), Align::Left);
    }
    caption(
        c,
        "SCENEGRAPH: A ROBOT ARM THROUGH A TWO-LENS CAMERA",
        "ONE POINT MAP PER BODY, FROM THE UNIT BOX TO PIXELS (PGA3D)",
    );
}

fn main() {
    run(Anim::new("scenegraph", SECONDS).size(960, 540), draw);
}

#[cfg(test)]
mod tests {
    use super::scene::*;
    use super::scenegraph::*;
    use gax::ApproxEq;
    use gax::pga3d::{Motor, Plane, Point};
    use gax_numga_examples::measure::turn;

    /// Whether `p` is the Euclidean point `want` to within `tol` in each coordinate.
    fn at(p: P, want: [f64; 3], tol: f64) -> bool {
        let [x, y, z] = want;
        p.unitized().max_abs_diff(&Point::xyz(x, y, z)) <= tol
    }

    /// The anisotropic scaling scales each coordinate and keeps the weight.
    #[test]
    fn anisotropic_scale_extensor() {
        let scaled = anisotropic_scale(2.0, 0.5, 3.0).of(Point::xyz(3.0, 4.0, 5.0));
        assert!(scaled.approx_eq(&Point::xyz(6.0, 2.0, 15.0), 1e-12));
    }

    /// Forward kinematics gives five rigidly placed bodies, whose vertices keep unit weight,
    /// and four joint pivots; numga's vertices of the gripper, for the still's pose.
    #[test]
    fn robot_arm_kinematics() {
        let (bodies, pivots) = robot_arm([0.2, -0.3, 0.5, -0.2]);
        assert_eq!((bodies.len(), pivots.len()), (5, 4));
        for b in bodies {
            for v in canonical_unit_box() {
                assert!((b.of(v).e123() - 1.0).abs() < 1e-10);
            }
        }
        // The pedestal rests on the floor.
        let (bodies, _) = robot_arm([0.35, -0.45, 0.85, -0.40]);
        assert!(at(
            bodies[0].of(Point::xyz(0.0, 0.0, -0.5)),
            [0.0; 3],
            1e-14
        ));
        // From numga, run on the same pose.
        let gripper = canonical_unit_box().map(|v| bodies[4].of(v));
        assert!(at(
            gripper[0],
            [-0.031010286433, -0.063198174656, 2.601597516832],
            1e-11
        ));
        assert!(at(
            gripper[6],
            [0.280019766211, -0.027697380946, 2.90159751683],
            1e-11
        ));
    }

    /// The two-lens camera sends world points onto the sensor plane `z = -1.25`.
    #[test]
    fn compound_optics_lands_on_the_sensor() {
        let pose = look_at(Point::xyz(0.0, -4.0, 1.5), Point::xyz(0.0, 0.0, 1.0));
        let (front, rear, _) = lens_train(1.0, 0.8, -0.3);
        let pupil = Point::xyz(0.04, 0.0, 0.0);
        let sensor = Plane::from_normal([0.0, 0.0, 1.0], -1.25);
        let camera = lens_camera(pose, front, rear, pupil, sensor);
        let hit = camera.of(Point::xyz(0.1, 0.2, 1.0)).unitized();
        assert!((sensor & hit).s().abs() < 1e-10);
        // The camera looks along its +z, towards the target, with its x axis level.
        let ahead = pose >> Point::direction(0.0, 0.0, 1.0);
        assert!(ahead.e013() > 0.99 && ahead.e021() < 0.0);
        let across = pose >> Point::direction(1.0, 0.0, 0.0);
        assert!(across.approx_eq(&Point::direction(1.0, 0.0, 0.0), 1e-12));
        // As numga builds it: a pitch about x, after the translation. The pitch is the turn
        // from straight ahead (4 along y) to the target (down 0.5).
        let level = gax::vga2d::Vector::new(4.0, 0.0);
        let pitch = turn(level, gax::vga2d::Vector::new(4.0, 1.0 - 1.5));
        let numga = Motor::translation(0.0, -4.0, 1.5)
            * Motor::rotation(axis(1.0, 0.0, 0.0), pitch - core::f64::consts::FRAC_PI_2);
        let p = Point::xyz(0.3, -0.2, 0.7);
        assert!((pose >> p).approx_eq(&(numga >> p), 1e-12));
    }

    /// The scenario's check: the collapsed map per body agrees with kinematics, camera and
    /// viewport applied in turn; and the photograph and rays agree with numga's.
    #[test]
    fn collapsed_maps_agree_with_the_pipeline_and_with_numga() {
        let rig = camera_rig();
        let photo = scenegraph(&rig);
        for (world, pixels) in photo.world.iter().zip(&photo.pixels) {
            for (v, p) in world.iter().zip(pixels) {
                let sequential = project(rig.world_to_pixel, *v);
                assert!(sequential.max_abs_diff(p) < 1e-8, "{sequential:?} vs {p:?}");
            }
        }
        // Each body's first vertex on the sensor, and the gripper's, from numga. The two
        // libraries round the exponentials and the lens shears differently, and the optics
        // magnify that to about 1e-8 pixels.
        let first = [
            [357.5869391748, 373.782620074394],
            [339.699114567071, 335.887373098189],
            [321.125057128149, 293.467576054822],
            [266.264056485336, 191.21030591223],
            [309.970901155386, 94.735848590647],
        ];
        for (pixels, [u, v]) in photo.pixels.iter().zip(first) {
            // On the sensor plane `z = -1.25` of the camera frame.
            assert!(at(pixels[0], [u, v, -1.25], 1e-6), "{:?}", pixels[0]);
        }
        assert!(at(
            photo.pixels[4][6],
            [278.30194022331, 65.190187677343, -1.25],
            1e-6
        ));
        // The first ray's hits on the rear lens plane and on the sensor.
        let ray = photo.rays[0];
        assert!(at(ray[1], [0.04, -3.6, 1.3], 1e-12));
        assert!(at(
            ray[2],
            [0.029910193552, -3.904599591948, 1.167010853104],
            1e-10
        ));
        assert!(at(
            ray[3],
            [-0.037559215044, -4.863505199123, 0.91567157478],
            1e-10
        ));
    }

    /// The sweep loops.
    #[test]
    fn the_sweep_loops() {
        let (a, b) = (sweep(0.0), sweep(core::f64::consts::TAU));
        assert!(a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-12));
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws(super::draw, 0.5);
    }
}
