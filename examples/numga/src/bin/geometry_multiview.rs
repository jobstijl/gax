//! numga's `geometry/multiview`: N-camera reconstruction and bundle adjustment with perspective
//! cone quadrics. A camera is a projective map on points, `Camera = Point <- Point`; a quadric is
//! its polarity map, `Plane <- Point`, with value `quadric(p) & p` at a point. A pixel's
//! precision on the sensor is a disc; feeding the camera into it and carrying its polar planes
//! back through the camera (the induced map on planes, solved from the incidence pairing) lifts
//! it into a sight cone in the scene whose cross-section widens with depth.
//!
//! Bundle adjustment works on the cones directly: sum each point's cones over its cameras and
//! take the fused cone's vertex (a gauge dyad on the weight makes it the pole of the plane at
//! infinity), then take a Gauss-Newton step on each camera's pose twist with the cone value as
//! the cost. A variant folds the points' response into the cameras' curvature by a Schur
//! complement, which also gives each camera's marginal information.
//!
//! The core is written once, as a macro instantiated for PGA2D (the drawings) and PGA3D (the
//! tests). The animation plays numga's four convergence scenarios in the plane: one, two and
//! three moving cameras, and the Schur complement with each camera's pose covariance. The sight
//! cones (a pixel wide) and the fused splats are rasterized per pixel from the quadrics.

use gax::Extensor;
use gax_numga_examples::canvas::mix;
use gax_numga_examples::{
    Align, Anim, Axes, Canvas, Marker, Rgb, backdrop, caption, palette, plot, run,
};
use std::sync::OnceLock;

/// The multiview core over one algebra: `gax::$ga` with `$Plane` its vectors (numga's `Plane`,
/// a line in the plane), and `gax::$vga::Vector` standing for numga's `Direction` (the ideal
/// points), embedded into the points by a map.
macro_rules! multiview_core {
    ($ga:ident, $Plane:ident, $vga:ident) => {
        use gax::$ga::$Plane as Plane;
        use gax::$ga::{Motor, Point, Scalar, Twist};
        use gax::{Kind, Unit};

        pub type P = Point<(), f64>;
        pub type Pl = Plane<(), f64>;
        pub type Tw = Twist<(), f64>;
        pub type M = Unit<Motor<(), f64>>;
        /// A camera: a projective map on points.
        pub type Camera = Point<(Point,), f64>;
        /// A quadric as its polarity map: a point to its polar plane.
        pub type Quadric = Plane<(Point,), f64>;
        /// The curvature of a cost over pose twists.
        pub type Information = Scalar<(Twist, Twist), f64>;
        /// A displacement of a point, an ideal point. gax has no kind for the ideal points of
        /// PGA, so a Euclidean vector of the matching VGA stands in, embedded by `embed()`.
        pub type Direction = gax::$vga::Vector;
        /// The dimension of the space.
        pub const DIM: usize = <Point as Kind>::N - 1;

        /// The finite point at Euclidean coordinates (gax's points store them first, then the
        /// weight).
        pub fn point(coords: &[f64]) -> P {
            P::from_coeffs(<Point as Kind>::arr_from_fn(|i| {
                if i < DIM { coords[i] } else { 1.0 }
            }))
        }

        /// The basis plane `i` (`x`, `y`, `z`, then `w`, the plane at infinity).
        pub fn plane(i: usize) -> Pl {
            Pl::from_coeffs(<Plane as Kind>::arr_from_fn(
                |k| if k == i { 1.0 } else { 0.0 },
            ))
        }

        /// The plane at infinity, `w = e0`: paired with a point it reads the weight.
        pub fn w() -> Pl {
            plane(DIM)
        }

        /// The pinhole camera at the origin with its sensor in the plane one unit ahead along
        /// the last axis (`y = 1` in the plane, `z = 1` in space).
        pub fn pinhole() -> Camera {
            (point(&[0.0; 3][..DIM]) & Point::slot()) ^ (plane(DIM - 1) - w())
        }

        /// The directions embedded as ideal points, a map `Point <- Direction`.
        pub fn embed() -> Point<(Direction,), f64> {
            Point::from_images(<Direction as Kind>::arr_from_fn(|i| {
                P::from_coeffs(<Point as Kind>::arr_from_fn(
                    |k| if k == i { 1.0 } else { 0.0 },
                ))
            }))
        }

        /// A point scaled so that the plane at infinity reads it as one, as numga normalizes
        /// projections.
        pub fn reading(p: P) -> P {
            p.gp(1.0 / (w() & p).s())
        }

        /// Per-pixel transverse precision dyads on the sensor line of a planar camera: the line
        /// through the pixel across the sensor, joined with its own readout.
        pub fn sensor_disk(pixel: P) -> Quadric {
            let normal = plane(0) - w().gp((plane(0) & pixel).s());
            normal * (normal & Point::slot())
        }

        /// A sensor quadric given at the principal point, moved to a pixel by the translation
        /// between them (numga's square root of their ratio).
        #[cfg_attr(not(test), allow(dead_code))]
        pub fn sensor_disk_at(pixel: P, principal: P, sensor: Quadric) -> Quadric {
            let trans = Motor::between(principal.unitized(), pixel.unitized());
            trans >> sensor.of(trans << Point::slot())
        }

        /// The map on planes induced by a map on points, through incidence: `on_planes(t)(l) & p
        /// == l & t(p)` for every plane and point, whether or not `t` is invertible.
        pub fn on_planes(collineation: Camera) -> Plane<(Plane,), f64> {
            (Plane::slot() & Point::slot()).solve(Plane::slot() & collineation)
        }

        /// A sensor disc pulled back through the camera into a perspective cone.
        pub fn make_cone(camera: Camera, disc: Quadric) -> Quadric {
            on_planes(camera).of(disc.of(camera))
        }

        /// The cones moved from each camera's frame to the world, `[point][camera]`.
        pub fn world_cones(motors: &[M], cones: &[Vec<Quadric>]) -> Vec<Vec<Quadric>> {
            cones
                .iter()
                .map(|row| {
                    row.iter()
                        .zip(motors)
                        .map(|(q, m)| *m >> q.of(*m << Point::slot()))
                        .collect()
                })
                .collect()
        }

        /// Scene points as the vertices of the fused cones: quadratic constraints add, and the
        /// fused cone's polar of its vertex vanishes; adding the gauge dyad on the weight makes
        /// the vertex the pole of the plane at infinity.
        pub fn triangulate(motors: &[M], cones: &[Vec<Quadric>]) -> (Vec<P>, Vec<Quadric>) {
            let gauge = w() * (w() & Point::slot());
            world_cones(motors, cones)
                .into_iter()
                .map(|row| {
                    let fused = row.into_iter().fold(Plane::zero(), |a, b| a + b);
                    let p: P = (fused + gauge).solve(w());
                    (p.unitized(), fused)
                })
                .unzip()
        }

        /// The motion of a camera-frame point per unit right step of its camera's pose: minus
        /// the commutator with the twist.
        pub fn motion(local: P) -> Point<(Twist,), f64> {
            -Twist::<(), f64>::slot().commutator(local)
        }

        /// One Gauss-Newton step per camera on the cone value, its twist solved by least
        /// squares with a cutoff, scaled by `free` (0 holds the anchored cameras).
        fn step_motors(
            motors: &[M],
            cones: &[Vec<Quadric>],
            points: &[P],
            information: &[Information],
            damping: f64,
            free: &[f64],
        ) -> Vec<M> {
            motors
                .iter()
                .enumerate()
                .map(|(c, m)| {
                    let mut gradient: Scalar<(Twist,), f64> = Scalar::zero();
                    for (p, row) in points.iter().zip(cones) {
                        let local = *m << *p;
                        gradient += row[c].of(local) & motion(local);
                    }
                    let step: Tw = information[c].lstsq_with(-gradient, 1e-4);
                    *m * step.gp(0.5 * damping * free[c]).exp()
                })
                .collect()
        }

        /// The curvature of the cone value over each camera's twist: the moved point's polar
        /// joined with its motion.
        pub fn camera_curvature(
            motors: &[M],
            cones: &[Vec<Quadric>],
            points: &[P],
        ) -> Vec<Information> {
            motors
                .iter()
                .enumerate()
                .map(|(c, m)| {
                    let mut h: Information = Scalar::zero();
                    for (p, row) in points.iter().zip(cones) {
                        let moving = motion(*m << *p);
                        h += row[c].of(moving) & moving;
                    }
                    h
                })
                .collect()
        }

        /// Camera poses and points by alternating triangulation with damped Gauss-Newton steps
        /// on the poses; `free` is 1 for each camera that moves, 0 for the anchors.
        pub fn bundle_adjust(
            initial: &[M],
            cones: &[Vec<Quadric>],
            iterations: usize,
            damping: f64,
            free: &[f64],
        ) -> (Vec<M>, Vec<P>, Vec<Quadric>) {
            let mut motors = initial.to_vec();
            for _ in 0..iterations {
                let (points, _) = triangulate(&motors, cones);
                let h = camera_curvature(&motors, cones, &points);
                motors = step_motors(&motors, cones, &points, &h, damping, free);
            }
            let (points, fused) = triangulate(&motors, cones);
            (motors, points, fused)
        }

        /// The cones in pixel units: each divided by the square of the depth of its point in
        /// its camera, the weight of the projected point, iterated since the points depend on
        /// the weights.
        pub fn reweight_cones(
            camera: Camera,
            motors: &[M],
            cones: &[Vec<Quadric>],
        ) -> Vec<Vec<Quadric>> {
            let mut weighted = cones.to_vec();
            for _ in 0..3 {
                let (points, _) = triangulate(motors, &weighted);
                weighted = cones
                    .iter()
                    .zip(&points)
                    .map(|(row, p)| {
                        row.iter()
                            .zip(motors)
                            .map(|(q, m)| {
                                let z = (w() & camera.of(*m << *p)).s().abs().max(0.1);
                                q.gp(1.0 / (z * z))
                            })
                            .collect()
                    })
                    .collect();
            }
            weighted
        }

        /// Joint Gauss-Newton on the poses with depth reweighting and a Schur complement: each
        /// point's response to a camera step, from its own curvature against the cross term,
        /// is folded into the camera's curvature. Returns the poses, points, fused quadrics
        /// and each camera's marginal information (zero on the anchors).
        pub fn bundle_adjust_schur(
            camera: Camera,
            initial: &[M],
            cones: &[Vec<Quadric>],
            iterations: usize,
            damping: f64,
            free: &[f64],
        ) -> (Vec<M>, Vec<P>, Vec<Quadric>, Vec<Information>) {
            let mut motors = initial.to_vec();
            let mut information = vec![Scalar::zero(); motors.len()];
            for _ in 0..iterations {
                let scaled = reweight_cones(camera, &motors, cones);
                let (points, _) = triangulate(&motors, &scaled);
                // A world displacement of a point, carried into each camera.
                let moved: Vec<Point<(Direction,), f64>> =
                    motors.iter().map(|m| *m << embed()).collect();
                let h_cam = camera_curvature(&motors, &scaled, &points);
                information = h_cam;
                for (p, row) in points.iter().zip(&scaled) {
                    let mut h_pt: Scalar<(Direction, Direction), f64> = Scalar::zero();
                    for (q, mv) in row.iter().zip(&moved) {
                        h_pt += q.of(*mv) & *mv;
                    }
                    for (c, m) in motors.iter().enumerate() {
                        let moving = motion(*m << *p);
                        let polar = row[c].of(moving);
                        let cross = polar & moved[c];
                        // The point's displacement per camera step, then its compliance.
                        let response = h_pt.solve(cross);
                        information[c] -= polar & (*m << embed().of(response));
                    }
                }
                motors = step_motors(&motors, &scaled, &points, &information, damping, free);
            }
            let (points, fused) = triangulate(&motors, &reweight_cones(camera, &motors, cones));
            let information = information
                .into_iter()
                .zip(free)
                .map(|(i, f)| i.gp(*f))
                .collect();
            (motors, points, fused, information)
        }

        /// The objective: every point's cone value, summed over its cameras.
        pub fn cone_cost(motors: &[M], points: &[P], cones: &[Vec<Quadric>]) -> f64 {
            let mut sum = 0.0;
            for (p, row) in points.iter().zip(cones) {
                for (q, m) in row.iter().zip(motors) {
                    let local = *m << *p;
                    sum += (q.of(local) & local).s();
                }
            }
            sum
        }

        /// The RMS distance between unit points: their difference is a direction.
        pub fn rmse(points: &[P], truth: &[P]) -> f64 {
            let sum: f64 = points
                .iter()
                .zip(truth)
                .map(|(p, t)| (p.unitized() - t.unitized()).ideal_norm().powi(2))
                .sum();
            (sum / points.len() as f64).sqrt()
        }
    };
}

/// The plane: numga's scenarios and the drawings.
mod plane {
    multiview_core!(pga2d, Line, vga2d);
    use gax::pga2d::Line;

    /// Landmarks in front of the cameras, at depths from 0.85 to 2.55.
    pub const LANDMARKS: [[f64; 2]; 6] = [
        [0.15, 0.85],
        [-0.43, 1.15],
        [0.50, 1.50],
        [0.03, 1.85],
        [0.65, 2.20],
        [-0.60, 2.55],
    ];
    pub const BASELINE: f64 = 0.75;
    pub fn gaze() -> f64 {
        18f64.to_radians()
    }

    /// Camera poses at offsets along the x axis, each panned by its gaze: numga's
    /// `exp(xw offset / 2) exp(xy gaze / 2)`, with `xw = -e01` and `xy = e12`.
    pub fn rig(offsets: &[f64], gazes: &[f64]) -> Vec<M> {
        offsets
            .iter()
            .zip(gazes)
            .map(|(o, g)| {
                Point::new(0.0, -o / 2.0, 0.0).exp() * Point::new(0.0, 0.0, g / 2.0).exp()
            })
            .collect()
    }

    /// The landmarks, and the sight cones of their pixels in each camera, `[point][camera]`.
    pub fn observe(true_motors: &[M]) -> (Vec<P>, Vec<Vec<Quadric>>) {
        let truth: Vec<P> = LANDMARKS.iter().map(|c| point(c)).collect();
        let camera = pinhole();
        let cones = truth
            .iter()
            .map(|p| {
                true_motors
                    .iter()
                    .map(|m| {
                        let pixel = reading(camera.of(*m << *p));
                        make_cone(camera, sensor_disk(pixel))
                    })
                    .collect()
            })
            .collect();
        (truth, cones)
    }

    /// Left and right cameras panned inwards, and a third at the origin looking ahead.
    pub fn three_camera_truth() -> Vec<M> {
        rig(&[-BASELINE, BASELINE, 0.0], &[gaze(), -gaze(), 0.0])
    }

    /// Two convergent cameras, the second panned 5% too far, aligned by eight steps; the
    /// scenario's check is that the cone cost vanishes.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn bundle_adjustment() -> (Vec<M>, Vec<P>, f64) {
        let truth = rig(&[-BASELINE, BASELINE], &[gaze(), -gaze()]);
        let (_, cones) = observe(&truth);
        let initial = rig(&[-BASELINE, BASELINE], &[gaze(), -gaze() * 1.05]);
        let (motors, points, _) = bundle_adjust(&initial, &cones, 8, 0.8, &[0.0, 1.0]);
        let cost = cone_cost(&motors, &points, &cones);
        (motors, points, cost)
    }

    /// A convergence scenario: its name, the rig after each step from the initial poses, and
    /// for the Schur scenario each camera's marginal information at each state.
    pub struct Scenario {
        pub name: &'static str,
        pub free: Vec<f64>,
        pub states: Vec<Vec<M>>,
        pub information: Option<Vec<Vec<Information>>>,
    }

    fn convergence(
        name: &'static str,
        initial: Vec<M>,
        damping: f64,
        free: Vec<f64>,
        iterations: usize,
    ) -> Scenario {
        let (_, cones) = observe(&three_camera_truth());
        let mut states = vec![initial];
        for _ in 0..iterations {
            let last = states.last().expect("a state");
            let (next, _, _) = bundle_adjust(last, &cones, 1, damping, &free);
            states.push(next);
        }
        Scenario {
            name,
            free,
            states,
            information: None,
        }
    }

    /// One moving camera, the right one with a 25% pan mismatch; the others anchored.
    pub fn one_camera(iterations: usize) -> Scenario {
        let initial = rig(&[-BASELINE, BASELINE, 0.0], &[gaze(), -gaze() * 1.25, 0.0]);
        convergence(
            "ONE MOVING CAMERA",
            initial,
            0.38,
            vec![0.0, 1.0, 0.0],
            iterations,
        )
    }

    /// Two moving cameras, both perturbed; the left one anchored.
    pub fn two_cameras(iterations: usize) -> Scenario {
        let initial = rig(
            &[-BASELINE, BASELINE, -0.1],
            &[gaze(), -gaze() * 1.25, 0.05],
        );
        convergence(
            "TWO MOVING CAMERAS",
            initial,
            0.35,
            vec![0.0, 1.0, 1.0],
            iterations,
        )
    }

    /// All three cameras perturbed and free, without anchors.
    pub fn three_cameras(iterations: usize) -> Scenario {
        let initial = rig(
            &[-BASELINE * 0.95, BASELINE, -0.1],
            &[gaze() * 1.05, -gaze() * 1.25, 0.05],
        );
        convergence(
            "THREE FREE CAMERAS",
            initial,
            0.32,
            vec![1.0; 3],
            iterations,
        )
    }

    /// One moving camera as in `one_camera`, by the joint step with the Schur complement; each
    /// state carries the marginal information from the step taken there.
    pub fn schur(iterations: usize) -> Scenario {
        let (_, cones) = observe(&three_camera_truth());
        let free = vec![0.0, 1.0, 0.0];
        let mut motors = rig(&[-BASELINE, BASELINE, 0.0], &[gaze(), -gaze() * 1.25, 0.0]);
        let (mut states, mut information) = (Vec::new(), Vec::new());
        for _ in 0..=iterations {
            let (next, _, _, info) = bundle_adjust_schur(pinhole(), &motors, &cones, 1, 0.7, &free);
            states.push(motors);
            information.push(info);
            motors = next;
        }
        Scenario {
            name: "SCHUR COMPLEMENT, WITH POSE COVARIANCE",
            free,
            states,
            information: Some(information),
        }
    }

    /// A pose's covariance from its information form: the form as a map from twists to
    /// forques (solved from the incidence pairing), pseudo-inverted with numga's cutoff.
    pub fn pose_covariance(information: Information) -> Point<(Line,), f64> {
        let map: Line<(Point,), f64> = (Line::slot() & Point::slot()).solve(information);
        map.pinv_with(1e-4)
    }

    /// The position of a camera's centre.
    pub fn centre(motor: M) -> [f32; 2] {
        let [x, y] = (motor >> point(&[0.0, 0.0])).to_euclidean();
        [x as f32, y as f32]
    }

    /// A camera's viewing direction, the image of its local `+y`.
    pub fn axis(motor: M) -> [f32; 2] {
        let d = motor >> Point::direction(0.0, 1.0);
        let n = d.e20().hypot(d.e01());
        [(d.e20() / n) as f32, (d.e01() / n) as f32]
    }
}

/// Space: the same core, exercised by the tests (which use part of it).
#[cfg(test)]
#[allow(dead_code)]
mod space {
    multiview_core!(pga3d, Plane, vga3d);
}

/// Each convergence scenario's states.
const ITERATIONS: usize = 10;
/// Seconds per Gauss-Newton step, and the pause after each scenario.
const PER_STEP: f32 = 0.33;
const HOLD: f32 = 1.2;
/// The viewport, and the angular half-width of a pixel (the cone radius per unit depth).
const X_RANGE: [f32; 2] = [-1.25, 1.25];
const Y_RANGE: [f32; 2] = [-0.3, 2.95];
const PIXEL_ANGLE: f64 = 0.035;

struct Data {
    scenarios: Vec<plane::Scenario>,
    truth: Vec<plane::P>,
    cones: Vec<Vec<plane::Quadric>>,
}

fn data() -> &'static Data {
    static DATA: OnceLock<Data> = OnceLock::new();
    DATA.get_or_init(|| {
        let (truth, cones) = plane::observe(&plane::three_camera_truth());
        Data {
            scenarios: vec![
                plane::one_camera(ITERATIONS),
                plane::two_cameras(ITERATIONS),
                plane::three_cameras(ITERATIONS),
                plane::schur(ITERATIONS),
            ],
            truth,
            cones,
        }
    })
}

fn camera_colour(c: usize) -> Rgb {
    [palette::sky(), palette::purple(), palette::green()][c % 3]
}

/// Smooth coverage of the level set `sqrt(value) <= level`, a pixel wide at the edge.
fn coverage(value: f64, level: f64, pixel: f64) -> f64 {
    let d = (level - value.max(0.0).sqrt()) / (0.75 * pixel);
    if d > 12.0 {
        1.0
    } else if d < -12.0 {
        0.0
    } else {
        1.0 / (1.0 + (-d).exp())
    }
}

/// The scene at some poses: the sight cones and fused splats rasterized from their quadrics,
/// then the cameras.
fn scene(c: &mut Canvas, ax: &Axes, motors: &[plane::M], cones: &[Vec<plane::Quadric>]) {
    use gax::pga2d::{Line, Point};
    let world = plane::world_cones(motors, cones);
    let (points, fused) = plane::triangulate(motors, cones);
    // Each camera's focal line, oriented so that a point ahead reads a positive depth.
    let focal: Vec<Line<(), f64>> = motors
        .iter()
        .map(|m| {
            let l = *m >> Line::new(0.0, 1.0, 0.0);
            let ahead = *m >> Point::xy(0.0, 1.0);
            if (l & ahead).s() > 0.0 { l } else { -l }
        })
        .collect();
    let depth = |p: plane::P, c: usize| (focal[c] & p).s();
    let value = |q: &plane::Quadric, p: plane::P| (q.of(p) & p).s();
    let floors: Vec<f64> = fused
        .iter()
        .zip(&points)
        .map(|(f, p)| value(f, *p).max(0.0))
        .collect();
    let radii: Vec<f64> = points
        .iter()
        .zip(&floors)
        .map(|(p, floor)| {
            let d = (0..motors.len()).map(|c| depth(*p, c)).sum::<f64>() / motors.len() as f64;
            ((PIXEL_ANGLE * d.max(0.2)).powi(2) + floor).sqrt()
        })
        .collect();
    let pixel = f64::from((ax.x[1] - ax.x[0]) / (ax.rect[2] - ax.rect[0]));
    let height = c.height as f32;
    let splat = palette::orange();
    let me = *ax;
    ax.image(c, 1, |x, y| {
        let p = Point::xy(f64::from(x), f64::from(y));
        let row = me.px([x, y])[1];
        let mut colour = mix(palette::top(), palette::bottom(), row / height);
        let mut ahead = false;
        for (k, cam) in (0..motors.len()).map(|k| (k, camera_colour(k))) {
            let d = depth(p, k);
            if d <= 0.02 {
                continue;
            }
            ahead = true;
            let mut keep = 1.0;
            for cone in &world {
                keep *= 1.0 - 0.3 * coverage(value(&cone[k], p), PIXEL_ANGLE * d.max(0.05), pixel);
            }
            colour = mix(cam, colour, keep as f32);
        }
        if ahead {
            let mut keep = 1.0;
            for ((f, floor), r) in fused.iter().zip(&floors).zip(&radii) {
                keep *= 1.0 - 0.95 * coverage(value(f, p) - floor, *r, pixel);
            }
            colour = mix(splat, colour, keep as f32);
        }
        Some(colour)
    });
    // The cameras: field-of-view wedge, sensor line and optical axis.
    for (k, m) in motors.iter().enumerate() {
        let (o, a) = (plane::centre(*m), plane::axis(*m));
        let n = [-a[1], a[0]];
        let (scale, spread) = (0.28f32, 38f32.to_radians().tan() * 0.28);
        let at = |s: f32, t: f32| [o[0] + a[0] * s + n[0] * t, o[1] + a[1] * s + n[1] * t];
        let (l, r) = (at(scale, -spread), at(scale, spread));
        let colour = camera_colour(k);
        ax.fill(c, &[o, l, r], colour, 0.25);
        ax.line(c, o, l, 1.1, colour, 0.7);
        ax.line(c, o, r, 1.1, colour, 0.7);
        ax.line(c, l, r, 2.0, colour, 1.0);
        ax.dashed(c, &[o, at(scale * 1.15, 0.0)], 1.2, 3.0, colour, 1.0);
    }
    let pts: Vec<[f32; 2]> = points
        .iter()
        .map(|p| {
            let [x, y] = p.to_euclidean();
            [x as f32, y as f32]
        })
        .collect();
    ax.scatter(c, &pts, Marker::Dot, 3.0, palette::ink(), 1.0);
}

/// Each moving camera's position ellipse (scaled for display) and turning fan, from its
/// marginal information.
fn covariances(c: &mut Canvas, ax: &Axes, motors: &[plane::M], information: &[plane::Information]) {
    use gax::pga2d::{Line, Point};
    for (k, (m, info)) in motors.iter().zip(information).enumerate() {
        let cov = plane::pose_covariance(*info);
        let size = cov
            .coeffs()
            .iter()
            .flatten()
            .fold(0.0f64, |a, v| a.max(v.abs()));
        if size == 0.0 {
            continue;
        }
        // The centre moves by the commutator of a local twist with the origin; reading that
        // displacement with a line is reading the twist with another line.
        let origin = Point::xy(0.0, 0.0);
        let shift = Point::slot().commutator(origin);
        let readout = (Line::slot() & Point::slot()).solve(Line::slot() & shift);
        let spread = readout & cov.of(readout);
        let (values, modes) = spread.eigh();
        let axes: Vec<([f64; 2], f64)> = modes
            .iter()
            .zip(values)
            .filter(|(l, _)| l.e1().hypot(l.e2()) > 0.5)
            .map(|(l, v)| {
                let n = l.e1().hypot(l.e2());
                (
                    [l.e1() / n, l.e2() / n],
                    (v.max(1e-8).sqrt() * 0.08).min(1.0),
                )
            })
            .collect();
        let ring: Vec<[f32; 2]> = (0..=64)
            .map(|i| {
                let a = core::f64::consts::TAU * f64::from(i) / 64.0;
                let mut q = [0.0, 0.0];
                for (j, (n, r)) in axes.iter().enumerate() {
                    let s = if j == 0 { a.cos() } else { a.sin() } * r;
                    q = [q[0] + n[0] * s, q[1] + n[1] * s];
                }
                let [x, y] = (*m >> Point::xy(q[0], q[1])).to_euclidean();
                [x as f32, y as f32]
            })
            .collect();
        let colour = camera_colour(k);
        ax.fill(c, &ring, colour, 0.3);
        ax.dashed(c, &ring, 1.6, 4.0, colour, 1.0);
        // The turning's standard deviation: the line at infinity reads a twist's rotation.
        let w = Line::new(0.0, 0.0, 1.0);
        let turn = (w & cov.of(w)).s().max(0.0).sqrt().to_degrees() * 0.35;
        let (o, a) = (plane::centre(*m), plane::axis(*m));
        let heading = a[1].atan2(a[0]);
        let half = (turn as f32).min(180.0).to_radians();
        let arc: Vec<[f32; 2]> = (0..=24)
            .map(|i| {
                let t = heading - half + 2.0 * half * i as f32 / 24.0;
                [o[0] + 0.38 * t.cos(), o[1] + 0.38 * t.sin()]
            })
            .collect();
        ax.dashed(c, &[arc[0], o, arc[24]], 1.1, 2.0, colour, 1.0);
        ax.dashed(c, &arc, 1.3, 2.0, colour, 1.0);
    }
}

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let (w, h) = (c.width as f32, c.height as f32);
    let data = data();
    let per = PER_STEP * ITERATIONS as f32 + HOLD;
    let which = ((t / per) as usize).min(data.scenarios.len() - 1);
    let s = &data.scenarios[which];
    let local = (t - which as f32 * per).max(0.0);
    let progress = (local / PER_STEP).min(ITERATIONS as f32);
    let k = (progress as usize).min(ITERATIONS - 1);
    let f = f64::from(progress - k as f32);
    let f = f * f * (3.0 - 2.0 * f);
    let motors: Vec<plane::M> = s.states[k]
        .iter()
        .zip(&s.states[k + 1])
        .map(|(a, b)| gax::pga2d::Motor::interpolate(*a, *b, f))
        .collect();
    let top = 64.0;
    let rect = plot::inset([0.0, top, w * 0.5, h], 8.0, 4.0, 8.0, 8.0);
    let span = (Y_RANGE[1] - Y_RANGE[0]) * 0.5;
    let ax = Axes::equal(rect, [0.0, 0.5 * (Y_RANGE[0] + Y_RANGE[1])], span);
    let _ = X_RANGE;
    scene(c, &ax, &motors, &data.cones);
    if let Some(info) = &s.information {
        let blended: Vec<plane::Information> = info[k]
            .iter()
            .zip(&info[k + 1])
            .map(|(a, b)| *a + (*b - *a).gp(f))
            .collect();
        covariances(c, &ax, &motors, &blended);
    }
    // Right: the point error along each scenario's steps, log scale.
    let chart = plot::inset([w * 0.5, top + 30.0, w, h * 0.8], 60.0, 20.0, 20.0, 40.0);
    let errors = Axes::new(chart, [0.0, ITERATIONS as f32], [1e-5, 1.0]).log_y();
    errors.frame(c, "POINT RMSE ALONG THE STEPS", "GAUSS-NEWTON STEP", "RMSE");
    for (i, sc) in data.scenarios.iter().enumerate() {
        let curve: Vec<[f32; 2]> = sc
            .states
            .iter()
            .enumerate()
            .map(|(j, m)| {
                let (points, _) = plane::triangulate(m, &data.cones);
                [j as f32, plane::rmse(&points, &data.truth).max(1e-5) as f32]
            })
            .collect();
        let shown = if i == which { 2.4 } else { 1.0 };
        let alpha = if i == which { 1.0 } else { 0.45 };
        errors.polyline(c, &curve, shown, palette::series(i), alpha);
        if i == which {
            let (now, _) = plane::triangulate(&motors, &data.cones);
            let cost = plane::cone_cost(&motors, &now, &data.cones);
            c.text(
                &format!("CONE COST {cost:.2e}"),
                chart[2],
                chart[1] - 6.0,
                11.0,
                palette::grid(),
                Align::Right,
            );
            let e = plane::rmse(&now, &data.truth).max(1e-5) as f32;
            errors.scatter(
                c,
                &[[progress, e]],
                Marker::Dot,
                9.0,
                palette::series(i),
                1.0,
            );
        }
    }
    let names: Vec<(&str, Rgb)> = data
        .scenarios
        .iter()
        .enumerate()
        .map(|(i, sc)| (sc.name, palette::series(i)))
        .collect();
    errors.legend(c, &names);
    let anchors: Vec<String> = s
        .free
        .iter()
        .enumerate()
        .map(|(i, f)| format!("CAM {i}: {}", if *f == 0.0 { "ANCHORED" } else { "FREE" }))
        .collect();
    c.text(
        &anchors.join("   "),
        w * 0.5 + 60.0,
        h * 0.8 + 30.0,
        12.0,
        palette::ink(),
        Align::Left,
    );
    c.text(
        "SIGHT CONES A PIXEL WIDE; ORANGE: FUSED SPLATS",
        w * 0.5 + 60.0,
        h * 0.8 + 52.0,
        11.0,
        palette::grid(),
        Align::Left,
    );
    caption(
        c,
        "MULTIVIEW: BUNDLE ADJUSTMENT ON CONE QUADRICS",
        &format!("{} (PGA2D), STEP {}", s.name, progress.round() as usize),
    );
}

fn main() {
    let seconds = (PER_STEP * ITERATIONS as f32 + HOLD) * 4.0;
    run(Anim::new("multiview", seconds).size(960, 540), draw);
}

#[cfg(test)]
mod tests {
    use super::{plane, space};
    use gax::Extensor;

    /// A small xorshift generator for numga's uniform landmarks (its stream cannot be
    /// reproduced).
    struct Rng(u64);
    impl Rng {
        fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            lo + (hi - lo) * ((self.0 >> 11) as f64 / (1u64 << 53) as f64)
        }
    }

    /// Cone pullback and fusion reconstruct the landmarks given the true poses, and the fused
    /// precision is positive on every displacement: the splat is a bounded ellipse.
    #[test]
    fn triangulate_cones() {
        let mut rng = Rng(123);
        let xy: Vec<[f64; 2]> = (0..6)
            .map(|_| [rng.uniform(-0.6, 0.6), rng.uniform(1.0, 2.5)])
            .collect();
        let truth: Vec<plane::P> = xy.iter().map(|c| plane::point(c)).collect();
        let motors = plane::rig(&[-0.75, 0.75], &[plane::gaze(), -plane::gaze()]);
        let camera = plane::pinhole();
        // A sensor quadric at the principal point, moved to each pixel, pulled back.
        let x = plane::plane(0);
        let sensor = x * (x & gax::pga2d::Point::slot());
        let principal = plane::point(&[0.0, 1.0]);
        let cones: Vec<Vec<plane::Quadric>> = truth
            .iter()
            .map(|p| {
                motors
                    .iter()
                    .map(|m| {
                        let pixel = plane::reading(camera.of(*m << *p));
                        plane::make_cone(camera, plane::sensor_disk_at(pixel, principal, sensor))
                    })
                    .collect()
            })
            .collect();
        let (points, fused) = plane::triangulate(&motors, &cones);
        for (p, c) in points.iter().zip(&xy) {
            let [a, b] = p.to_euclidean();
            assert!(
                (a - c[0]).abs() < 1e-5 && (b - c[1]).abs() < 1e-5,
                "{a} {b} {c:?}"
            );
        }
        for f in &fused {
            for i in 0..12 {
                let a = core::f64::consts::PI * f64::from(i) / 12.0;
                let d = gax::pga2d::Point::direction(a.cos(), a.sin());
                assert!((f.of(d) & d).s() > 0.0);
            }
        }
    }

    /// Alternating Newton steps on the cone value recover the scene from a perturbed camera.
    #[test]
    fn bundle_adjust_converges() {
        let (truth, cones) = plane::observe(&plane::three_camera_truth());
        let g = plane::gaze();
        let initial = plane::rig(&[-0.75, 0.75, 0.0], &[g, -g * 1.05, 0.0]);
        let (start, _) = plane::triangulate(&initial, &cones);
        let (_, points, _) = plane::bundle_adjust(&initial, &cones, 12, 0.9, &[0.0, 1.0, 0.0]);
        let (e0, e1) = (plane::rmse(&start, &truth), plane::rmse(&points, &truth));
        assert!(e1 < e0 && e1 < 0.02, "{e0} {e1}");
    }

    /// The joint step with the Schur complement converges and returns the pose information:
    /// symmetric, zero on the anchored cameras, positive definite on the moving one.
    #[test]
    fn schur_bundle_adjust_converges_and_yields_information() {
        let (truth, cones) = plane::observe(&plane::three_camera_truth());
        let g = plane::gaze();
        let initial = plane::rig(&[-0.75, 0.75, 0.0], &[g, -g * 1.05, 0.0]);
        let (start, _) = plane::triangulate(&initial, &cones);
        let (_, points, _, information) = plane::bundle_adjust_schur(
            plane::pinhole(),
            &initial,
            &cones,
            10,
            0.7,
            &[0.0, 1.0, 0.0],
        );
        let (e0, e1) = (plane::rmse(&start, &truth), plane::rmse(&points, &truth));
        assert!(e1 < e0 && e1 < 0.02, "{e0} {e1}");
        assert_eq!(information.len(), 3);
        for (k, info) in information.iter().enumerate() {
            let gram = info.coeffs()[0];
            // Symmetry compares entry (i, j) with (j, i): indices read best.
            #[allow(clippy::needless_range_loop)]
            for i in 0..3 {
                for j in 0..3 {
                    assert!((gram[i][j] - gram[j][i]).abs() < 1e-9);
                    if k != 1 {
                        assert_eq!(gram[i][j], 0.0);
                    }
                }
            }
        }
        let (values, _) = information[1].eigh();
        assert!(values.iter().all(|v| *v > 0.0), "{values:?}");
    }

    /// The figure's scenario: two cameras aligned, the cone cost vanishes.
    #[test]
    fn the_two_camera_rig_aligns() {
        let (_, _, cost) = plane::bundle_adjustment();
        assert!(cost.abs() < 1e-12, "{cost}");
    }

    /// Each convergence scenario lowers the cone cost at its fused points, step by step. The
    /// alternating solver converges linearly and slowly (about 7% of the pan error per step at
    /// numga's damping 0.38: the points follow the moving camera), the Schur complement step,
    /// which moves the points with the cameras, to a hundredth of a millimetre in ten steps.
    #[test]
    fn the_scenarios_converge() {
        let (truth, cones) = plane::observe(&plane::three_camera_truth());
        for s in [
            plane::one_camera(10),
            plane::two_cameras(10),
            plane::three_cameras(10),
            plane::schur(10),
        ] {
            let cost =
                |m: &[plane::M]| plane::cone_cost(m, &plane::triangulate(m, &cones).0, &cones);
            let costs: Vec<f64> = s.states.iter().map(|m| cost(m)).collect();
            assert!(
                costs.windows(2).all(|w| w[1] < w[0]),
                "{} {costs:?}",
                s.name
            );
            assert!(costs[10] < 0.5 * costs[0], "{} {costs:?}", s.name);
        }
        let s = plane::schur(10);
        let error = plane::rmse(&plane::triangulate(&s.states[10], &cones).0, &truth);
        assert!(error < 1e-4, "{error}");
    }

    // --- the same core in space ------------------------------------------------------------

    /// numga's PGA3D twists by blade name, as gax `Line` coefficients
    /// `[e23, e31, e12, e01, e02, e03]` (`xw = -e01`, `yw = -e02`, `zw = -e03`).
    fn twist(yz: f64, zx: f64, xy: f64, xw: f64, yw: f64, zw: f64) -> gax::pga3d::Line<(), f64> {
        gax::pga3d::Line::new(yz, zx, xy, -xw, -yw, -zw)
    }

    const XYZ: [[f64; 3]; 8] = [
        [0.15, -0.20, 1.20],
        [-0.43, 0.15, 1.45],
        [0.50, -0.10, 1.70],
        [0.03, 0.25, 1.95],
        [0.65, -0.15, 2.30],
        [-0.60, 0.10, 2.65],
        [0.20, 0.30, 2.10],
        [-0.25, -0.25, 1.60],
    ];

    /// Three convergent cameras and eight landmarks in space with their sight cones: left
    /// panned right, right panned left, and a central one raised and looking slightly down.
    fn rig_3d() -> (Vec<space::M>, Vec<Vec<space::Quadric>>) {
        let theta = 18f64.to_radians();
        let truth = vec![
            twist(0.0, 0.0, 0.0, -0.375, 0.0, 0.0).exp()
                * twist(0.0, -theta / 2.0, 0.0, 0.0, 0.0, 0.0).exp(),
            twist(0.0, 0.0, 0.0, 0.375, 0.0, 0.0).exp()
                * twist(0.0, theta / 2.0, 0.0, 0.0, 0.0, 0.0).exp(),
            twist(0.0, 0.0, 0.0, 0.0, 0.175, 0.0).exp()
                * twist(-6f64.to_radians(), 0.0, 0.0, 0.0, 0.0, 0.0).exp(),
        ];
        let camera = space::pinhole();
        // 2D transverse uncertainty on the sensor plane z = 1 around the principal point.
        let (x, y) = (space::plane(0), space::plane(1));
        let p = gax::pga3d::Point::slot();
        let sensor = x * (x & p) + y * (y & p);
        let principal = space::point(&[0.0, 0.0, 1.0]);
        let cones = XYZ
            .iter()
            .map(|c| {
                let q = space::point(c);
                truth
                    .iter()
                    .map(|m| {
                        let pixel = space::reading(camera.of(*m << q));
                        space::make_cone(camera, space::sensor_disk_at(pixel, principal, sensor))
                    })
                    .collect()
            })
            .collect();
        (truth, cones)
    }

    /// In space the moving camera returns from pose errors of tens of degrees and most of a
    /// metre.
    #[test]
    fn bundle_adjust_3d_recovers_a_badly_perturbed_camera() {
        let (truth, cones) = rig_3d();
        for (rotation, translation, iterations) in [
            ([30.0, -20.0, 25.0], [-0.40, 0.30, 0.45], 30),
            ([50.0, 35.0, -40.0], [0.60, -0.50, 0.80], 40),
        ] {
            let [rx, ry, rz]: [f64; 3] = rotation.map(|d: f64| d.to_radians());
            let [tx, ty, tz] = translation;
            let perturbation = twist(0.0, 0.0, 0.0, tx, ty, tz).gp(0.5).exp()
                * twist(rx, ry, rz, 0.0, 0.0, 0.0).gp(0.5).exp();
            let motors = vec![truth[0], perturbation * truth[1], truth[2]];
            let (start, _) = space::triangulate(&motors, &cones);
            let initial_cost = space::cone_cost(&motors, &start, &cones);
            let (est, points, _) =
                space::bundle_adjust(&motors, &cones, iterations, 0.7, &[0.0, 1.0, 0.0]);
            let centre = (est[1] >> space::point(&[0.0, 0.0, 0.0])).to_euclidean();
            for (a, b) in centre.iter().zip([0.75, 0.0, 0.0]) {
                assert!((a - b).abs() < 0.02, "{centre:?}");
            }
            for (p, c) in points.iter().zip(&XYZ) {
                for (a, b) in p.to_euclidean().iter().zip(c) {
                    assert!((a - b).abs() < 0.02);
                }
            }
            // The residual rotation's scalar part is the cosine of half its angle.
            let residual = (est[1] * truth[1].reverse()).into_inner().s();
            let angle = 2.0 * residual.abs().min(1.0).acos();
            assert!(angle.to_degrees() < 0.5, "{}", angle.to_degrees());
            assert!(space::cone_cost(&est, &points, &cones) < initial_cost * 1e-3);
        }
    }

    /// In space, the second camera panned 5% too far aligns with the other two anchored.
    #[test]
    fn bundle_adjust_3d_converges_from_a_small_perturbation() {
        let (truth, cones) = rig_3d();
        let theta = 18f64.to_radians();
        let motors = vec![
            truth[0],
            twist(0.0, 0.0, 0.0, 0.375, 0.0, 0.0).exp()
                * twist(0.0, theta * 1.05 / 2.0, 0.0, 0.0, 0.0, 0.0).exp(),
            truth[2],
        ];
        let landmarks: Vec<space::P> = XYZ.iter().map(|c| space::point(c)).collect();
        let (start, _) = space::triangulate(&motors, &cones);
        let (_, points, _) = space::bundle_adjust(&motors, &cones, 10, 0.9, &[0.0, 1.0, 0.0]);
        let (e0, e1) = (
            space::rmse(&start, &landmarks),
            space::rmse(&points, &landmarks),
        );
        assert!(e1 < e0 && e1 < 0.02, "{e0} {e1}");
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws(super::draw, 1.0);
    }
}
