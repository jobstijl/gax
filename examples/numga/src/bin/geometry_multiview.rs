//! numga's `estimation/multiview`: N-camera reconstruction and camera alignment with perspective
//! cone quadrics. A camera is a projective map on points, `Camera = Point <- Point`; a quadric is
//! its polarity map, `Plane <- Point`, with value `quadric(p) & p` at a point. A pixel's
//! precision on the sensor is a ball; feeding the camera into it and carrying its polar planes
//! back through the camera's adjugate (the induced map on planes, which a central projection
//! has though it has no inverse) lifts it into a sight cone in the scene whose cross-section
//! widens with depth.
//!
//! Alignment works on the cones directly: sum each point's cones over its cameras into a splat
//! and take its centre (a gauge dyad on the weight makes it the pole of the plane at infinity),
//! then take a Gauss-Newton step on each camera's pose twist with the cone value as the cost.
//! The joint variant weighs the cones into pixel units and folds the points' response into the
//! cameras' curvature by a Schur complement, which couples every pair of cameras through the
//! points they share: the information is a form over the field of the cameras' twists, and one
//! solve moves them all. A third aligns each pixel's sight line to the other cameras' belief
//! without placing a point, and lands where the joint steps do.
//!
//! The core is written once, as a macro instantiated for PGA2D (the drawings) and PGA3D (the
//! tests). The animation plays numga's four alignments in the plane: one and two moving cameras
//! a step at a time, three free cameras at an even pace along their screws, and the joint steps
//! with each moving camera's pose covariance. The sight cones (a pixel wide) and the fused
//! splats are rasterized per pixel from the quadrics.

use gax::ApproxEq;
use gax::motions::Linear;

use gax_numga_examples::signal::smoothstep;
use gax_numga_examples::{
    Align, Anim, Axes, Canvas, Light, Marker, Point2, Rect, backdrop, caption, palette, run_with,
};

/// The multiview core over one algebra: `gax::$ga` with `$Plane` its vectors (numga's `Plane`,
/// a line in the plane), and its `Direction`, the ideal points.
macro_rules! multiview_core {
    ($ga:ident, $Plane:ident) => {
        use gax::sites::{SiteField, Sites};
        use gax::$ga::$Plane as Plane;
        use gax::$ga::{Direction, Motor, Point, Scalar, Twist};
        use gax::{Kind, Unit};

        pub type P = Point;
        pub type Pl = Plane;
        pub type Tw = Twist;
        pub type M = Unit<Motor>;
        /// A camera: a projective map on points.
        pub type Camera = Point<(Point,)>;
        /// A quadric as its polarity map: a point to its polar plane.
        pub type Quadric = Plane<(Point,)>;
        /// The curvature of a cost over pose twists.
        pub type Information = Scalar<(Twist, Twist)>;
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

        /// The directions widened into points (of weight zero), a map `Point <- Direction`.
        pub fn embed() -> Point<(Direction,)> {
            Direction::slot().cast::<Point>()
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

        /// A sensor disc pulled back through the camera into a perspective cone: the disc of
        /// the camera's images, carried back to planes by the camera's adjugate, the map on
        /// planes induced through incidence (`adjugate(t)(l) & p == l & t(p)` for every plane
        /// and point; a central projection has no inverse, but an adjugate).
        pub fn make_cone(camera: Camera, disc: Quadric) -> Quadric {
            camera.adjugate().of(disc.of(camera))
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
                    let fused: Quadric = row.into_iter().sum();
                    let p: P = (fused + gauge).solve(w());
                    (p.unitized(), fused)
                })
                .unzip()
        }

        /// The motion of a camera-frame point per unit right step of its camera's pose: minus
        /// the commutator with the twist.
        pub fn motion(local: P) -> Point<(Twist,)> {
            -Twist::slot().commutator(local)
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
                    let gradient: Scalar<(Twist,)> = points
                        .iter()
                        .zip(cones)
                        .map(|(p, row)| {
                            let local = *m << *p;
                            row[c].of(local) & motion(local)
                        })
                        .sum();
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
                    points
                        .iter()
                        .zip(cones)
                        .map(|(p, row)| {
                            let moving = motion(*m << *p);
                            row[c].of(moving) & moving
                        })
                        .sum()
                })
                .collect()
        }

        /// Camera poses and points by alternating triangulation with damped Gauss-Newton steps
        /// on the poses; `free` is 1 for each camera that moves, 0 for the anchors.
        pub fn align(
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

        /// Uncertainty that does not grow with depth, in units of depth: calibration, the
        /// extent of a point and whatever else the cones leave out. Pixel noise grows with
        /// depth, and the two variances add.
        pub const FLOOR: f64 = 0.1;

        /// The cones in pixel units: each divided by the square of the depth of its point in
        /// its camera, the weight of the projected point, plus the floor's square, iterated
        /// since the points depend on the weights (numga's three times).
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
                                let depth = (w() & camera.of(*m << *p)).s();
                                q.gp(1.0 / (depth * depth + FLOOR * FLOOR))
                            })
                            .collect()
                    })
                    .collect();
            }
            weighted
        }

        /// The information on all `N` poses together, a form over the field of the cameras'
        /// twists.
        pub type Joint<const N: usize> = Scalar<(Sites<Twist, N>, Sites<Twist, N>)>;

        /// Camera `i`'s twist placed in the field of the cameras' twists, `Twist[N] <- Twist`.
        pub fn place<const N: usize>(i: usize) -> SiteField<Twist, N, (Twist,), f64> {
            SiteField::from_sites(core::array::from_fn(|k| {
                Twist::slot().gp(if k == i { 1.0 } else { 0.0 })
            }))
        }

        /// Camera `i`'s twist read from the field of the cameras' twists, `Twist <- Twist[N]`.
        pub fn pick<const N: usize>(i: usize) -> Twist<(Sites<Twist, N>,)> {
            SiteField::<Twist, N, (Sites<Twist, N>,), f64>::from_blocks(|a, b| {
                Twist::slot().gp(if a == b { 1.0 } else { 0.0 })
            })
            .site(i)
        }

        /// Joint Gauss-Newton on the poses with depth reweighting and a Schur complement. A
        /// step of one camera moves the points it sees, and a displaced point changes the cost
        /// seen by every camera that observes it: solving each point's curvature against the
        /// cross curvature gives its response to a step of each camera, and folded into the
        /// cameras' own curvature as the Schur complement it couples every pair of cameras
        /// through the points they share. The information is a form over the field of the
        /// `N` cameras' twists, one least-squares solve moves all of them at once, and the
        /// anchored cameras, zeroed in the form, fix where the scene lies. Returns the poses,
        /// points, fused quadrics and the information of the last step.
        pub fn align_jointly<const N: usize>(
            camera: Camera,
            initial: &[M],
            cones: &[Vec<Quadric>],
            iterations: usize,
            damping: f64,
            free: &[f64],
        ) -> (Vec<M>, Vec<P>, Vec<Quadric>, Joint<N>) {
            assert_eq!(initial.len(), N);
            let mut motors = initial.to_vec();
            let mut information = Joint::<N>::zero();
            for _ in 0..iterations {
                let scaled = reweight_cones(camera, &motors, cones);
                let (points, _) = triangulate(&motors, &scaled);
                // A world displacement of a point, carried into each camera.
                let moved: Vec<Point<(Direction,)>> =
                    motors.iter().map(|m| *m << embed()).collect();
                let own = camera_curvature(&motors, &scaled, &points);
                let mut blocks: Vec<Vec<Information>> = (0..N)
                    .map(|i| {
                        (0..N)
                            .map(|j| if i == j { own[i] } else { Scalar::zero() })
                            .collect()
                    })
                    .collect();
                let mut slope: Scalar<(Sites<Twist, N>,)> = Scalar::zero();
                for (p, row) in points.iter().zip(&scaled) {
                    let point_curvature: Scalar<(Direction, Direction)> =
                        row.iter().zip(&moved).map(|(q, mv)| q.of(*mv) & *mv).sum();
                    let locals: Vec<P> = motors.iter().map(|m| *m << *p).collect();
                    let polars: Vec<Plane<(Twist,)>> = (row.iter().zip(&locals))
                        .map(|(q, local)| q.of(motion(*local)))
                        .collect();
                    // The point's displacement per step of each camera.
                    let responses: Vec<Direction<(Twist,)>> = (polars.iter().zip(&moved))
                        .map(|(polar, mv)| point_curvature.solve(*polar & *mv))
                        .collect();
                    for i in 0..N {
                        for (j, response) in responses.iter().enumerate() {
                            blocks[i][j] -= polars[i] & moved[i].of(*response);
                        }
                        let own_slope = row[i].of(locals[i]) & motion(locals[i]);
                        slope += own_slope.of(pick::<N>(i)).gp(free[i]);
                    }
                }
                information = Joint::<N>::zero();
                for (i, row) in blocks.iter().enumerate() {
                    for (j, block) in row.iter().enumerate() {
                        information += block
                            .of_both(pick::<N>(i), pick::<N>(j))
                            .gp(free[i] * free[j]);
                    }
                }
                let step: SiteField<Twist, N, (), f64> = information.lstsq_with(-slope, 1e-4);
                motors = (motors.iter().enumerate())
                    .map(|(c, m)| *m * step.site(c).gp(0.5 * damping).exp())
                    .collect();
            }
            let (points, fused) = triangulate(&motors, &reweight_cones(camera, &motors, cones));
            (motors, points, fused, information)
        }

        /// The block of the joint information between cameras `i` and `j`; camera `i`'s own,
        /// `block(joint, i, i)`, is its information with the other cameras held in place.
        pub fn block<const N: usize>(joint: Joint<N>, i: usize, j: usize) -> Information {
            joint.of_both(place::<N>(i), place::<N>(j))
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
        #[cfg_attr(not(test), allow(dead_code))]
        #[allow(clippy::disallowed_methods)] // the root of a mean square, a statistic
        pub fn rmse(points: &[P], truth: &[P]) -> f64 {
            let sum: f64 = points
                .iter()
                .zip(truth)
                .map(|(p, t)| (p.unitized() - t.unitized()).ideal_norm_squared())
                .sum();
            (sum / points.len() as f64).sqrt()
        }
    };
}

/// The plane: numga's scenarios and the drawings.
mod plane {
    multiview_core!(pga2d, Line);
    use gax::pga2d::Forque;

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

    /// Camera poses at offsets along the x axis, each looking along its `+y` panned clockwise
    /// (towards `+x`) by its gaze: numga's `exp(xw offset / 2) exp(xy gaze / 2)`.
    pub fn rig(offsets: &[f64], gazes: &[f64]) -> Vec<M> {
        offsets
            .iter()
            .zip(gazes)
            .map(|(o, g)| {
                Motor::translation(Point::direction(*o, 0.0)) * Motor::rotation(Point::origin(), -g)
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
    pub fn aligned_pair() -> (Vec<M>, Vec<P>, f64) {
        let truth = rig(&[-BASELINE, BASELINE], &[gaze(), -gaze()]);
        let (_, cones) = observe(&truth);
        let initial = rig(&[-BASELINE, BASELINE], &[gaze(), -gaze() * 1.05]);
        let (motors, points, _) = align(&initial, &cones, 8, 0.8, &[0.0, 1.0]);
        let cost = cone_cost(&motors, &points, &cones);
        (motors, points, cost)
    }

    /// Camera alignment without placing a point: each pixel's sight line is brought through
    /// the belief of the other cameras about its point (numga's `align_rays_to_splats`). A
    /// cone is minus twice the log-likelihood of its pixel as a function of where the point
    /// lies, flat along the sight line, so a splat, the cones summed, is minus twice the log of
    /// the belief about its point. A pixel's cost is the least value the splat of the other
    /// cameras takes along its sight line: a sharp belief penalizes a line that misses it, a
    /// broad one barely does. Along the line through the pinhole with heading `h` that least
    /// value is the meet of the two points' polar planes paired with the line, over the
    /// stiffness along the line, `polar(h) & h`; leaving each pixel's own cone out (it
    /// vanishes along its own line) changes neither the cost nor its slope and lets the steps
    /// converge as fast as `align_jointly`. Where along each line the belief is least, its
    /// depth, weights the cones into pixel units for the next step. The pinhole and the pixels
    /// are in the cameras' frames, `pixels[point][camera]`.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn align_rays_to_splats(
        initial: &[M],
        cones: &[Vec<Quadric>],
        pinhole: P,
        pixels: &[Vec<P>],
        iterations: usize,
        damping: f64,
        free: &[f64],
    ) -> (Vec<M>, Vec<Quadric>) {
        // Each pixel's heading and sight line, and how the pinhole, the headings and the sight
        // lines move per unit step of the pose; all fixed in the cameras' frames.
        let headings: Vec<Vec<P>> = (pixels.iter())
            .map(|row| row.iter().map(|p| *p - pinhole).collect())
            .collect();
        let pinhole_motion = Twist::slot().commutator(pinhole);
        let mut motors = initial.to_vec();
        let mut weighted = cones.to_vec();
        let splats = |motors: &[M], weighted: &[Vec<Quadric>]| -> Vec<Quadric> {
            world_cones(motors, weighted)
                .into_iter()
                .map(|row| row.into_iter().sum())
                .collect()
        };
        for _ in 0..iterations {
            let beliefs = splats(&motors, &weighted);
            let mut slopes = vec![Scalar::<(Twist,)>::zero(); motors.len()];
            let mut curvatures = vec![Information::zero(); motors.len()];
            let mut depths = vec![vec![0.0; motors.len()]; pixels.len()];
            for (p, belief) in beliefs.iter().enumerate() {
                for (c, m) in motors.iter().enumerate() {
                    // The belief as camera `c` sees it, less its own cone.
                    let other = (*m << belief.of(*m >> Point::slot())) - weighted[p][c];
                    let heading = headings[p][c];
                    let ray = pinhole & heading;
                    let ray_motion = Twist::slot().commutator(ray);
                    let heading_motion = Twist::slot().commutator(heading);
                    // The polar of the sight line, the meet of its points' polar planes, and
                    // how it moves.
                    let (polar_pinhole, polar_heading) = (other.of(pinhole), other.of(heading));
                    let polar = polar_pinhole ^ polar_heading;
                    let polar_motion = (other.of(pinhole_motion) ^ polar_heading)
                        + (polar_pinhole ^ other.of(heading_motion));
                    let stiffness = (polar_heading & heading).s();
                    // The polar joined with the line's motion is the slope, the polar's motion
                    // joined with the line's motion the curvature.
                    slopes[c] += (polar & ray_motion).gp(1.0 / stiffness);
                    curvatures[c] += (polar_motion & ray_motion).gp(1.0 / stiffness);
                    depths[p][c] = -(polar_pinhole & heading).s() / stiffness;
                }
            }
            motors = (motors.iter().enumerate())
                .map(|(c, m)| {
                    let step: Twist = curvatures[c].lstsq_with(-slopes[c], 1e-4);
                    *m * step.gp(0.5 * damping * free[c]).exp()
                })
                .collect();
            // Where along each sight line the belief is least puts the next fusion in pixel
            // units; its sign drops out of its square.
            weighted = (cones.iter().zip(&depths))
                .map(|(row, ds)| {
                    (row.iter().zip(ds))
                        .map(|(q, d)| q.gp(1.0 / (d * d + FLOOR * FLOOR)))
                        .collect()
                })
                .collect();
        }
        let beliefs = splats(&motors, &weighted);
        (motors, beliefs)
    }

    /// A convergence scenario: its name, the rig after each step from the initial poses, and
    /// for the joint steps each camera's own information at each state.
    pub struct Scenario {
        pub name: &'static str,
        pub free: Vec<f64>,
        /// Whether it plays at an even pace along its states (else a step at a time).
        pub even: bool,
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
            let (next, _, _) = align(last, &cones, 1, damping, &free);
            states.push(next);
        }
        Scenario {
            name,
            free,
            even: false,
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

    /// All three cameras misplaced and free, without anchors: numga's README picture, each
    /// offset scaled by `[0.7, 1.2, 1]` and shifted by `[0, 0, -0.45]`, each turn scaled by
    /// `[1.6, 2.2, 1]` and turned further by `[0, 0, 0.35]`, a tenth of each step taken.
    pub fn three_cameras(iterations: usize) -> Scenario {
        let initial = rig(
            &[-BASELINE * 0.7, BASELINE * 1.2, -0.45],
            &[gaze() * 1.6, -gaze() * 2.2, 0.35],
        );
        Scenario {
            even: true,
            ..convergence("THREE FREE CAMERAS", initial, 0.1, vec![1.0; 3], iterations)
        }
    }

    /// The depth at which a camera's turn counts as much as its move, about where the
    /// landmarks are.
    pub const LEVER: f64 = 1.5;

    /// The distance travelled along an alignment, at each state from the start: per step the
    /// pinholes' moves plus their cameras' turns counted at the lever's depth (twice the lever
    /// times the norm of the screw, its log, which is half the angle), summed over the
    /// cameras. Resampled at even shares of it, the alignment plays at an even pace.
    pub fn pace(states: &[Vec<M>]) -> Vec<f64> {
        let mut travelled = vec![0.0];
        for pair in states.windows(2) {
            let step: f64 = (pair[0].iter().zip(&pair[1]))
                .map(|(a, b)| {
                    let screw = (*b * a.reverse()).log();
                    let (from, to) = (*a >> Point::origin(), *b >> Point::origin());
                    (to - from).ideal_norm() + 2.0 * LEVER * screw.norm()
                })
                .sum();
            travelled.push(travelled.last().expect("a start") + step);
        }
        travelled
    }

    /// One moving camera as in `one_camera`, by the joint steps; each state carries each
    /// camera's own block of the information from the step taken there.
    pub fn jointly(iterations: usize) -> Scenario {
        let (_, cones) = observe(&three_camera_truth());
        let free = vec![0.0, 1.0, 0.0];
        let mut motors = rig(&[-BASELINE, BASELINE, 0.0], &[gaze(), -gaze() * 1.25, 0.0]);
        let (mut states, mut information) = (Vec::new(), Vec::new());
        for _ in 0..=iterations {
            let (next, _, _, joint) = align_jointly::<3>(pinhole(), &motors, &cones, 1, 0.7, &free);
            states.push(motors);
            information.push((0..3).map(|i| block(joint, i, i)).collect());
            motors = next;
        }
        Scenario {
            name: "JOINT STEPS, WITH POSE COVARIANCE",
            free,
            even: false,
            states,
            information: Some(information),
        }
    }

    /// A pose's covariance from its information form: the form as a map from twists to
    /// forques (solved from the incidence pairing), pseudo-inverted with numga's cutoff.
    pub fn pose_covariance(information: Information) -> Twist<(Forque,)> {
        let map: Forque<(Twist,)> = (Forque::slot() & Twist::slot()).solve(information);
        map.pinv_with(1e-4)
    }
}

/// Space: the same core, exercised by the tests (which use part of it).
#[cfg(test)]
#[allow(dead_code)]
mod space {
    multiview_core!(pga3d, Plane);
}

/// Each convergence scenario's steps, and the three free cameras' (numga's 10 and 30).
const ITERATIONS: usize = 10;
const FREE_ITERATIONS: usize = 30;
/// Seconds per Gauss-Newton step, and the pause after each scenario stepped through (numga's
/// frames of a third of a second).
const PER_STEP: f32 = 0.33;
const HOLD: f32 = 1.2;
/// The seconds of the three free cameras' alignment, played at an even pace, and of the
/// aligned rig held (numga's 108 and 18 frames of 40 ms).
const EVEN: f32 = 4.32;
const EVEN_HOLD: f32 = 0.72;
/// The viewport's height, and the angular half-width of a pixel (the cone radius per unit
/// depth).
const Y_RANGE: [f32; 2] = [-0.3, 2.95];
const PIXEL_ANGLE: f64 = 0.035;

struct Data {
    scenarios: Vec<plane::Scenario>,
    cones: Vec<Vec<plane::Quadric>>,
}

/// The four scenarios, and the sight cones they are measured against.
fn data() -> Data {
    let (_, cones) = plane::observe(&plane::three_camera_truth());
    Data {
        scenarios: vec![
            plane::one_camera(ITERATIONS),
            plane::two_cameras(ITERATIONS),
            plane::three_cameras(FREE_ITERATIONS),
            plane::jointly(ITERATIONS),
        ],
        cones,
    }
}

fn camera_colour(c: usize) -> Light {
    [palette::SKY, palette::PURPLE, palette::GREEN][c % 3]
}

/// Smooth coverage of the level set `sqrt(value) <= level`, a pixel wide at the edge.
#[allow(clippy::disallowed_methods)] // a quadric's value is a squared width; the edge, its root
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
    use gax::pga2d::{Line, Motor, Point};
    let world = plane::world_cones(motors, cones);
    let (points, fused) = plane::triangulate(motors, cones);
    // Each camera's focal line, oriented so that a point ahead reads a positive depth.
    let focal: Vec<Line> = motors
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
    // Each splat's width: a pixel at its depth and the fused residual, in quadrature.
    #[allow(clippy::disallowed_methods)] // the root of a sum of squared widths
    let radii: Vec<f64> = points
        .iter()
        .zip(&floors)
        .map(|(p, floor)| {
            let d = (0..motors.len()).map(|c| depth(*p, c)).sum::<f64>() / motors.len() as f64;
            ((PIXEL_ANGLE * d.max(0.2)).powi(2) + floor).sqrt()
        })
        .collect();
    // A pixel's width in the scene, and the backdrop's light at a pixel, by its depth down
    // the canvas.
    let pixel = f64::from(ax.scale().recip());
    let height = c.rect().height();
    let backdrop_at =
        |q: Point2| palette::TOP.mix_light(palette::BOTTOM, q.unitized().e01() / height);
    let splat = palette::ORANGE;
    let me = *ax;
    ax.image(c, 1, |at| {
        let p = at.map_coefs(f64::from);
        let mut colour = backdrop_at(me.px(at));
        let mut ahead = false;
        // The cones glow faintly in their camera's light, so that where they cross the splats
        // still stand out.
        for (k, cam) in (0..motors.len()).map(|k| (k, (camera_colour(k)).faded(0.35))) {
            let d = depth(p, k);
            if d <= 0.02 {
                continue;
            }
            ahead = true;
            let keep: f64 = world
                .iter()
                .map(|cone| {
                    1.0 - 0.3 * coverage(value(&cone[k], p), PIXEL_ANGLE * d.max(0.05), pixel)
                })
                .product();
            colour = cam.mix_light(colour, keep as f32);
        }
        if ahead {
            let keep: f64 = fused
                .iter()
                .zip(&floors)
                .zip(&radii)
                .map(|((f, floor), r)| 1.0 - 0.95 * coverage(value(f, p) - floor, *r, pixel))
                .product();
            colour = splat.mix_light(colour, keep as f32);
        }
        Some(colour)
    });
    // The cameras, drawn in their own frames (looking along +y): a field-of-view wedge 38
    // degrees either side of the axis to a depth of 0.28, the sensor line across it, and the
    // optical axis.
    let origin = Point::origin();
    let depth = 0.28;
    for (k, m) in motors.iter().enumerate() {
        let edge = |angle: f64| {
            let d = Motor::rotation(origin, angle) >> Point::direction(0.0, 1.0);
            *m >> (origin + d.gp(depth / d.e01()))
        };
        let (o, l, r) = (
            *m >> origin,
            edge(-38f64.to_radians()),
            edge(38f64.to_radians()),
        );
        let colour = camera_colour(k);
        ax.fill(c, &[o, l, r], colour, 0.25);
        ax.polyline(c, &[l, o, r], 1.1, colour.faded(0.7));
        ax.line(c, l, r, 2.0, colour);
        let ahead = *m >> Point::xy(0.0, depth * 1.15);
        ax.dashed(c, &[o, ahead], 1.2, 3.0, colour);
    }
    ax.scatter(c, &points, Marker::Dot, 3.0, palette::INK);
}

/// Each moving camera's position ellipse (scaled for display) and turning fan, from its
/// marginal information.
fn covariances(c: &mut Canvas, ax: &Axes, motors: &[plane::M], information: &[plane::Information]) {
    use gax::pga2d::{Forque, Line, Motor, Point, Scalar, Twist};
    let origin = Point::origin();
    // The centre moves by the commutator of a local twist with the origin; reading that
    // displacement with a line is reading the twist with a forque.
    let shift = Twist::slot().commutator(origin);
    let readout = (Forque::slot() & Twist::slot()).solve(Line::slot() & shift);
    // The line at infinity, the join of two directions, reads a twist's rotation.
    let w = Point::direction(1.0, 0.0) & Point::direction(0.0, 1.0);
    for (k, (m, info)) in motors.iter().zip(information).enumerate() {
        let cov = plane::pose_covariance(*info);
        if cov.max_abs_diff(&Linear::zero()) == 0.0 {
            continue;
        }
        // The ellipse's axes: the modes of the position's spread that read a position (lines
        // with a direction), their normals with their standard deviations.
        let (values, modes) = (readout & cov.of(readout)).eigh(Scalar::identity());
        #[allow(clippy::disallowed_methods)] // standard deviations from variances
        let axes: Vec<(Point, f64)> = modes
            .iter()
            .zip(values)
            .filter(|(l, _)| l.norm() > 0.5)
            .map(|(l, v)| {
                let n = l.normalized();
                (
                    Point::direction(n.e1(), n.e2()),
                    (v.max(1e-8).sqrt() * 0.02).min(1.0),
                )
            })
            .collect();
        // Round the ellipse: the unit direction at each angle, its coordinates along the axes.
        let ring: Vec<Point> = (0..=64)
            .map(|i| {
                let angle = core::f64::consts::TAU * f64::from(i) / 64.0;
                let u = Point::direction_from_angle(angle);
                let offset = axes
                    .iter()
                    .zip([u.e20(), u.e01()])
                    .fold(origin, |q, ((n, r), s)| q + n.gp(r * s));
                *m >> offset
            })
            .collect();
        let colour = camera_colour(k);
        ax.fill(c, &ring, colour, 0.3);
        ax.dashed(c, &ring, 1.6, 4.0, colour);
        // The turning's standard deviation, a fan either side of the axis (scaled for display).
        #[allow(clippy::disallowed_methods)] // a standard deviation from a variance
        let turn = (w & cov.of(w)).s().max(0.0).sqrt() * 0.09;
        let half = turn.min(core::f64::consts::PI);
        let arc: Vec<Point> = (0..=24)
            .map(|i| {
                let angle = half * (2.0 * f64::from(i) / 24.0 - 1.0);
                *m >> (Motor::rotation(origin, angle) >> Point::xy(0.0, 0.38))
            })
            .collect();
        ax.dashed(c, &[arc[0], *m >> origin, arc[24]], 1.1, 2.0, colour);
        ax.dashed(c, &arc, 1.3, 2.0, colour);
    }
}

/// The scene at the poses now, on the left: the cones and splats, then each moving camera's
/// spread, its information blended between the steps as its pose is.
fn draw_poses(
    c: &mut Canvas,
    rect: Rect,
    data: &Data,
    s: &plane::Scenario,
    at: (usize, f64),
    motors: &[plane::M],
) {
    let (k, f) = at;
    let span = (Y_RANGE[1] - Y_RANGE[0]) * 0.5;
    let ax = Axes::equal(rect, Point2::xy(0.0, 0.5 * (Y_RANGE[0] + Y_RANGE[1])), span);
    scene(c, &ax, motors, &data.cones);
    if let Some(info) = &s.information {
        let blended: Vec<plane::Information> = info[k]
            .iter()
            .zip(&info[k + 1])
            .map(|(a, b)| *a + (*b - *a).gp(f))
            .collect();
        covariances(c, &ax, motors, &blended);
    }
}

/// The cone cost along each scenario's steps, log scale: the scenario shown bold, with a dot at
/// its progress. The cost, not the points' distance from the truth, measures the alignment: with
/// every camera free the rig may settle anywhere, and with one anchored at any scale.
fn draw_errors(
    c: &mut Canvas,
    chart: Rect,
    data: &Data,
    which: usize,
    motors: &[plane::M],
    progress: f32,
) {
    const FLOOR: f64 = 1e-13;
    let cost = |m: &[plane::M]| {
        let (points, _) = plane::triangulate(m, &data.cones);
        plane::cone_cost(m, &points, &data.cones).max(FLOOR) as f32
    };
    let errors = Axes::new(chart, [0.0, FREE_ITERATIONS as f32], [FLOOR as f32, 10.0]).log_y();
    errors.frame(c, "CONE COST ALONG THE STEPS", "GAUSS-NEWTON STEP", "COST");
    for (i, sc) in data.scenarios.iter().enumerate() {
        // The chart's points: (step, cost).
        let curve: Vec<Point2> = (sc.states.iter().enumerate())
            .map(|(j, m)| Point2::xy(j as f32, cost(m)))
            .collect();
        let shown = if i == which { 2.4 } else { 1.0 };
        let strength = if i == which { 1.0 } else { 0.45 };
        errors.polyline(c, &curve, shown, (palette::series(i)).faded(strength));
        if i == which {
            let here = Point2::xy(progress, cost(motors));
            errors.scatter(c, &[here], Marker::Dot, 9.0, palette::series(i));
        }
    }
    // The legend names the scenarios by their first two words.
    let short: Vec<String> = data
        .scenarios
        .iter()
        .map(|sc| sc.name.split(' ').take(2).collect::<Vec<_>>().join(" "))
        .collect();
    let names: Vec<(&str, Light)> = short
        .iter()
        .enumerate()
        .map(|(i, name)| (name.trim_end_matches(','), palette::series(i)))
        .collect();
    errors.legend(c, &names);
}

/// Notes under the chart's frame (and its axis labels): which cameras move, the cost now, and
/// what the scene shows.
fn draw_notes(c: &mut Canvas, chart: Rect, data: &Data, s: &plane::Scenario, motors: &[plane::M]) {
    let down = Point2::direction(0.0, 1.0);
    let pick = |free: bool| -> String {
        let cams: Vec<String> = (s.free.iter().enumerate())
            .filter(|(_, f)| (**f != 0.0) == free)
            .map(|(i, _)| i.to_string())
            .collect();
        if cams.is_empty() {
            "NONE".to_string()
        } else {
            cams.join(", ")
        }
    };
    let (now, _) = plane::triangulate(motors, &data.cones);
    let cost = plane::cone_cost(motors, &now, &data.cones);
    for (k, (text, size, light)) in [
        (
            format!("ANCHORED: {}   FREE: {}", pick(false), pick(true)),
            12.0,
            palette::INK,
        ),
        (format!("CONE COST {cost:.2e}"), 11.0, palette::GRID),
        ("SIGHT CONES A PIXEL WIDE;".to_string(), 11.0, palette::GRID),
        ("ORANGE: THE FUSED SPLATS".to_string(), 11.0, palette::GRID),
    ]
    .iter()
    .enumerate()
    {
        let at = chart.bottom_left() + down.gp(64.0 + 17.0 * k as f32);
        c.text(text, at, *size, *light, Align::Left);
    }
}

/// The seconds a scenario plays.
fn duration(s: &plane::Scenario) -> f32 {
    if s.even {
        EVEN + EVEN_HOLD
    } else {
        PER_STEP * (s.states.len() - 1) as f32 + HOLD
    }
}

/// Where a scenario is, `local` seconds in: the step `k` and the share `f` of the next one
/// taken. Stepped scenarios ease into each step; the even one resamples its states at equal
/// shares of the distance travelled along the screws between them.
fn position(s: &plane::Scenario, local: f32) -> (usize, f64) {
    let steps = s.states.len() - 1;
    if s.even {
        let pace = plane::pace(&s.states);
        let share = f64::from((local / EVEN).min(1.0)) * pace[steps];
        let k = (pace.partition_point(|p| *p <= share).max(1) - 1).min(steps - 1);
        let f = ((share - pace[k]) / (pace[k + 1] - pace[k])).clamp(0.0, 1.0);
        (k, f)
    } else {
        let progress = (local / PER_STEP).min(steps as f32);
        let k = (progress as usize).min(steps - 1);
        (k, smoothstep(f64::from(progress - k as f32)))
    }
}

fn draw(c: &mut Canvas, data: &Data, t: f32) {
    backdrop(c);
    let screen = c.rect();
    let (w, h) = (screen.width(), screen.height());
    let (mut which, mut local) = (0, t);
    while which + 1 < data.scenarios.len() && local >= duration(&data.scenarios[which]) {
        local -= duration(&data.scenarios[which]);
        which += 1;
    }
    let s = &data.scenarios[which];
    let (k, f) = position(s, local.max(0.0));
    let progress = k as f32 + f as f32;
    let motors: Vec<plane::M> = s.states[k]
        .iter()
        .zip(&s.states[k + 1])
        .map(|(a, b)| gax::pga2d::Motor::interpolate(*a, *b, f))
        .collect();
    let top = 64.0;
    let rect = Rect::new(0.0, top, w * 0.5, h).inset(8.0, 4.0, 8.0, 8.0);
    draw_poses(c, rect, data, s, (k, f), &motors);
    let chart = Rect::new(w * 0.5, top + 30.0, w, h * 0.8).inset(60.0, 20.0, 20.0, 40.0);
    draw_errors(c, chart, data, which, &motors, progress);
    draw_notes(c, chart, data, s, &motors);
    caption(
        c,
        "MULTIVIEW: CAMERA ALIGNMENT ON CONE QUADRICS",
        &format!("{} (PGA2D), STEP {}", s.name, progress.round() as usize),
    );
}

fn main() {
    let seconds = data().scenarios.iter().map(duration).sum();
    run_with(Anim::new("multiview", seconds).size(960, 540), data, draw);
}

#[cfg(test)]
mod tests {
    use super::{plane, space};
    use gax::pga3d::{Motor, Point};
    use gax::{ApproxEq, Extensor};
    use gax_numga_examples::rng::{Draw, rng};

    /// Cone pullback and fusion reconstruct the landmarks given the true poses, and the fused
    /// precision is positive on every displacement: the splat is a bounded ellipse.
    #[test]
    fn triangulate_cones() {
        let mut rng = rng(123);
        let truth: Vec<plane::P> = (0..6)
            .map(|_| gax::pga2d::Point::xy(rng.range(-0.6, 0.6), rng.range(1.0, 2.5)))
            .collect();
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
        for (p, t) in points.iter().zip(&truth) {
            assert!(p.max_abs_diff(t) < 1e-5, "{p:?} {t:?}");
        }
        for f in &fused {
            for i in 0..12 {
                let a = core::f64::consts::PI * f64::from(i) / 12.0;
                let d = gax::pga2d::Point::direction_from_angle(a);
                assert!((f.of(d) & d).s() > 0.0);
            }
        }
    }

    /// Alternating Newton steps on the cone value recover the scene from a perturbed camera.
    #[test]
    fn alternating_alignment_converges() {
        let (truth, cones) = plane::observe(&plane::three_camera_truth());
        let g = plane::gaze();
        let initial = plane::rig(&[-0.75, 0.75, 0.0], &[g, -g * 1.05, 0.0]);
        let (start, _) = plane::triangulate(&initial, &cones);
        let (_, points, _) = plane::align(&initial, &cones, 12, 0.9, &[0.0, 1.0, 0.0]);
        let (e0, e1) = (plane::rmse(&start, &truth), plane::rmse(&points, &truth));
        assert!(e1 < e0 && e1 < 0.02, "{e0} {e1}");
    }

    /// With two cameras free, a point's response to a step of one changes the cost seen by the
    /// other: the joint step that accounts for it settles within a few steps. With one camera
    /// anchored the scale of the scene is free, so the cone cost, not the distance to the true
    /// poses, measures convergence.
    #[test]
    fn the_joint_step_moves_two_free_cameras_together() {
        let (_, cones) = plane::observe(&plane::three_camera_truth());
        let g = plane::gaze();
        let mut motors = plane::rig(&[-0.75, 0.75 * 1.1, 0.15], &[g, -g * 1.25, 0.12]);
        let cost = |m: &[plane::M]| plane::cone_cost(m, &plane::triangulate(m, &cones).0, &cones);
        let initial = cost(&motors);
        let mut previous = motors.clone();
        for _ in 0..5 {
            previous = motors.clone();
            motors = plane::align_jointly::<3>(
                plane::pinhole(),
                &motors,
                &cones,
                1,
                1.0,
                &[0.0, 1.0, 1.0],
            )
            .0;
        }
        assert!(
            cost(&motors) < 1e-5 * initial,
            "{} {initial}",
            cost(&motors)
        );
        for (a, b) in motors.iter().zip(&previous) {
            assert!(a.max_abs_diff(b) < 1e-5);
        }
    }

    /// Aligning sight lines to the other cameras' beliefs, weighted into pixel units by depths
    /// read off those beliefs, recovers the perturbed camera as the joint step does, without
    /// placing a point.
    #[test]
    fn ray_alignment_matches_the_joint_step() {
        let truth_motors = plane::three_camera_truth();
        let (truth, cones) = plane::observe(&truth_motors);
        let g = plane::gaze();
        let initial = plane::rig(&[-0.75, 0.75, 0.0], &[g, -g * 1.05, 0.0]);
        let free = [0.0, 1.0, 0.0];
        let camera = plane::pinhole();
        let pixels: Vec<Vec<plane::P>> = (truth.iter())
            .map(|p| {
                (truth_motors.iter())
                    .map(|m| plane::reading(camera.of(*m << *p)))
                    .collect()
            })
            .collect();
        let (by_rays, _) = plane::align_rays_to_splats(
            &initial,
            &cones,
            plane::point(&[0.0, 0.0]),
            &pixels,
            4,
            1.0,
            &free,
        );
        let (jointly, _, _, _) =
            plane::align_jointly::<3>(camera, &initial, &cones, 10, 1.0, &free);
        for ((a, b), t) in by_rays.iter().zip(&jointly).zip(&truth_motors) {
            assert!(a.max_abs_diff(t) < 1e-10, "{a:?} {t:?}");
            assert!(a.max_abs_diff(b) < 1e-10, "{a:?} {b:?}");
        }
    }

    /// The joint step with the Schur complement converges and returns the pose information:
    /// symmetric, zero on the anchored cameras, positive definite on the moving one.
    #[test]
    fn joint_alignment_converges_and_yields_information() {
        let (truth, cones) = plane::observe(&plane::three_camera_truth());
        let g = plane::gaze();
        let initial = plane::rig(&[-0.75, 0.75, 0.0], &[g, -g * 1.05, 0.0]);
        let (start, _) = plane::triangulate(&initial, &cones);
        let (_, points, _, information) = plane::align_jointly::<3>(
            plane::pinhole(),
            &initial,
            &cones,
            10,
            0.7,
            &[0.0, 1.0, 0.0],
        );
        let (e0, e1) = (plane::rmse(&start, &truth), plane::rmse(&points, &truth));
        assert!(e1 < e0 && e1 < 0.02, "{e0} {e1}");
        // The information over the field of the three cameras' twists, block by block.
        for i in 0..3 {
            for j in 0..3 {
                let (a, b) = (
                    plane::block(information, i, j),
                    plane::block(information, j, i),
                );
                let (a, b) = (a.coeffs()[0], b.coeffs()[0]);
                // Symmetry compares entry (k, l) with (l, k): indices read best.
                #[allow(clippy::needless_range_loop)]
                for k in 0..3 {
                    for l in 0..3 {
                        assert!((a[k][l] - b[l][k]).abs() < 1e-9);
                        if i != 1 || j != 1 {
                            assert_eq!(a[k][l], 0.0);
                        }
                    }
                }
            }
        }
        let (values, _) = plane::block(information, 1, 1).eigh(plane::Information::identity());
        assert!(values.iter().all(|v| *v > 0.0), "{values:?}");
    }

    /// The figure's scenario: two cameras aligned, the cone cost vanishes.
    #[test]
    fn the_two_camera_rig_aligns() {
        let (_, _, cost) = plane::aligned_pair();
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
            plane::three_cameras(30),
            plane::jointly(10),
        ] {
            let cost =
                |m: &[plane::M]| plane::cone_cost(m, &plane::triangulate(m, &cones).0, &cones);
            let costs: Vec<f64> = s.states.iter().map(|m| cost(m)).collect();
            assert!(
                costs.windows(2).all(|w| w[1] < w[0]),
                "{} {costs:?}",
                s.name
            );
            let last = costs.len() - 1;
            assert!(costs[last] < 0.5 * costs[0], "{} {costs:?}", s.name);
        }
        let s = plane::jointly(10);
        let error = plane::rmse(&plane::triangulate(&s.states[10], &cones).0, &truth);
        assert!(error < 1e-4, "{error}");
    }

    /// The even pace counts a camera's move by its length and its turn by its angle times the
    /// lever, summed over the cameras and the steps.
    #[test]
    fn the_pace_counts_moves_and_turns() {
        let states = vec![
            plane::rig(&[0.0, 1.0], &[0.0, 0.0]),
            plane::rig(&[0.3, 1.0], &[0.0, 0.0]),
            plane::rig(&[0.3, 1.0], &[0.0, 0.2]),
        ];
        let pace = plane::pace(&states);
        assert!((pace[1] - 0.3).abs() < 1e-12, "{pace:?}");
        // The turn about the pinhole moves it nowhere.
        assert!(
            (pace[2] - 0.3 - plane::LEVER * 0.2).abs() < 1e-12,
            "{pace:?}"
        );
    }

    // --- the same core in space ------------------------------------------------------------

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

    /// The landmarks.
    fn landmarks() -> Vec<space::P> {
        XYZ.iter().map(|c| space::point(c)).collect()
    }

    /// The right camera of the rig in space, panned left by `pan` about its vertical.
    fn right_camera(pan: f64) -> space::M {
        Motor::translation(Point::direction(0.75, 0.0, 0.0))
            * Motor::rotation_about(Point::direction(0.0, 1.0, 0.0), -pan)
    }

    /// Three convergent cameras and eight landmarks in space with their sight cones: left
    /// panned right, right panned left (18 degrees each), and a central one raised and looking
    /// 12 degrees down.
    fn rig_3d() -> (Vec<space::M>, Vec<Vec<space::Quadric>>) {
        let theta = 18f64.to_radians();
        let truth = vec![
            Motor::translation(Point::direction(-0.75, 0.0, 0.0))
                * Motor::rotation_about(Point::direction(0.0, 1.0, 0.0), theta),
            right_camera(theta),
            Motor::translation(Point::direction(0.0, 0.35, 0.0))
                * Motor::rotation_about(Point::direction(1.0, 0.0, 0.0), 12f64.to_radians()),
        ];
        let camera = space::pinhole();
        // 2D transverse uncertainty on the sensor plane z = 1 around the principal point.
        let (x, y) = (space::plane(0), space::plane(1));
        let p = gax::pga3d::Point::slot();
        let sensor = x * (x & p) + y * (y & p);
        let principal = space::point(&[0.0, 0.0, 1.0]);
        let cones = landmarks()
            .into_iter()
            .map(|q| {
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
    fn alignment_3d_recovers_a_badly_perturbed_camera() {
        let (truth, cones) = rig_3d();
        // Rotation vectors in degrees (a turn by the length about the direction), translations
        // and iterations.
        for (rotation, translation, iterations) in [
            ([-30.0, 20.0, -25.0], [-0.40, 0.30, 0.45], 30),
            ([-50.0, -35.0, 40.0], [0.60, -0.50, 0.80], 40),
        ] {
            let spin: Point = Point::direction(rotation[0], rotation[1], rotation[2]);
            let [tx, ty, tz] = translation;
            let perturbation = Motor::translation(Point::direction(tx, ty, tz))
                * Motor::rotation_about(
                    Point::direction(rotation[0], rotation[1], rotation[2]),
                    spin.ideal_norm().to_radians(),
                );
            let motors = vec![truth[0], perturbation * truth[1], truth[2]];
            let (start, _) = space::triangulate(&motors, &cones);
            let initial_cost = space::cone_cost(&motors, &start, &cones);
            let (est, points, _) = space::align(&motors, &cones, iterations, 0.7, &[0.0, 1.0, 0.0]);
            let centre = est[1] >> space::point(&[0.0, 0.0, 0.0]);
            let true_centre = space::point(&[0.75, 0.0, 0.0]);
            assert!(centre.max_abs_diff(&true_centre) < 0.02, "{centre:?}");
            for (p, l) in points.iter().zip(&landmarks()) {
                assert!(p.max_abs_diff(l) < 0.02);
            }
            // The residual rotation, taken the shorter way (a motor and its negative move
            // alike): its logarithm's Euclidean part is half its angle.
            let mut residual = (est[1] * truth[1].reverse()).into_inner();
            if residual.s() < 0.0 {
                residual = -residual;
            }
            let log = gax::Unit::new_unchecked(residual).log();
            let angle = 2.0 * log.norm();
            assert!(angle.to_degrees() < 0.5, "{}", angle.to_degrees());
            assert!(space::cone_cost(&est, &points, &cones) < initial_cost * 1e-3);
        }
    }

    /// In space, the second camera panned 5% too far aligns with the other two anchored.
    #[test]
    fn alignment_3d_converges_from_a_small_perturbation() {
        let (truth, cones) = rig_3d();
        let motors = vec![truth[0], right_camera(18f64.to_radians() * 1.05), truth[2]];
        let landmarks = landmarks();
        let (start, _) = space::triangulate(&motors, &cones);
        let (_, points, _) = space::align(&motors, &cones, 10, 0.9, &[0.0, 1.0, 0.0]);
        let (e0, e1) = (
            space::rmse(&start, &landmarks),
            space::rmse(&points, &landmarks),
        );
        assert!(e1 < e0 && e1 < 0.02, "{e0} {e1}");
    }

    /// The rig is numga's: its twists by blade name (`xw = -e01`, `zx = e31`, ...), exponentiated.
    #[test]
    fn the_rigs_are_numgas() {
        let theta = 18f64.to_radians();
        let twist = |c: [f64; 6]| gax::pga3d::Line::new(c[0], c[1], c[2], c[3], c[4], c[5]).exp();
        let numga = [
            twist([0.0, 0.0, 0.0, 0.375, 0.0, 0.0])
                * twist([0.0, -theta / 2.0, 0.0, 0.0, 0.0, 0.0]),
            twist([0.0, 0.0, 0.0, -0.375, 0.0, 0.0])
                * twist([0.0, theta / 2.0, 0.0, 0.0, 0.0, 0.0]),
            twist([0.0, 0.0, 0.0, 0.0, -0.175, 0.0])
                * twist([-6f64.to_radians(), 0.0, 0.0, 0.0, 0.0, 0.0]),
        ];
        for (m, n) in rig_3d().0.iter().zip(&numga) {
            assert!(
                m.into_inner().approx_eq(&n.into_inner(), 1e-15),
                "{m:?} {n:?}"
            );
        }
        let numga = plane::Tw::from_coeffs([0.0, -0.375, 0.0]).exp()
            * plane::Tw::from_coeffs([0.0, 0.0, theta / 2.0]).exp();
        let m = plane::rig(&[0.75], &[theta])[0];
        assert!(m.into_inner().approx_eq(&numga.into_inner(), 1e-15));
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws_with(super::draw, &super::data(), 1.0);
    }
}
