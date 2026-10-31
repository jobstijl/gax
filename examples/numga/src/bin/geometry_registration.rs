//! numga's `geometry/registration`: a rigid motion from corresponding points, in flat space and
//! on the three-sphere, by the same lines. The one-sided fit sets up `target M - M source = 0`,
//! linear in the open motor `M`, sums the squared coefficients of the residual into a form on
//! motors (the bulk norm plus the weight norm, the bulk norm of the dual, which also sees the
//! ideal blades of flat space), and takes the least mode against the motor's own metric (the
//! scalar part of `M ~M`): in flat space it has no unit for a shift, so the eigensolver finds
//! for each turn the shift that fits best, centring without an origin; on the three-sphere,
//! numga's `w+x+y+z+`, every motion is a turn and the metric measures all of it. The classical
//! centered fit, which numga has since dropped, stays for comparison: align the centered clouds
//! by the rotor that maximizes the sandwich alignment, then carry the rotated centroid onto the
//! target's. The animation flies the source cloud along each estimated motion onto the target,
//! with each point tied to its correspondent; the three-sphere's points are drawn by their
//! `x`, `y`, `z` readings, as numga draws them.
//!
//! The motor metric is singular on the translation part of flat space; numga's general
//! eigensolver sends those modes to infinity, and so does gax's `eigh_semidefinite`, which lists
//! the finite modes first.

use gax::pga3d::{Motor, Point};

use gax_numga_examples::canvas::HEIGHT;
use gax_numga_examples::scene3::panel3;
use gax_numga_examples::signal::{haversine, wave};
use gax_numga_examples::{
    Align, Anim, Camera, Canvas, Lens, Marker, Point2, Rect, backdrop, caption, palette, run,
};

/// The three-sphere as `R(4,0,0)`, numga's `w+x+y+z+`, in PGA3D's layout: points are the
/// antivectors (intrinsic), motors the rotors, all six of whose planes turn.
mod sga3d {
    gax::algebra! {
        algebra sga3d "The three-sphere, R(4,0,0), numga's `w+x+y+z+`, in the PGA3D layout.";
        basis e0 = 1, e1 = 1, e2 = 1, e3 = 1;
        kind Scalar = [1];
        versor Plane = [e1, e2, e3, e0];
        kind Line = [e23, e31, e12, e01, e02, e03];
        versor Point = [e032, e013, e021, e123] intrinsic;
        kind Pseudoscalar = [e0123];
        versor Motor = [1, e23, e31, e12, e01, e02, e03, e0123];
        versor Flector = [e1, e2, e3, e0, e032, e013, e021, e123];
    }
    pub use sga3d::*;
}

/// The registration core over one algebra with PGA3D's layout: flat space (`gax::pga3d`) or
/// the three-sphere (`sga3d`).
macro_rules! registration_core {
    ($($ga:ident)::+) => {
        use gax::Unit;
        use gax_numga_examples::rng::{Draw, Rng};
        use $($ga)::+::{Line, Motor, Plane, Point, Scalar, intrinsic};

        pub type P = Point;
        pub type M = Unit<Motor>;
        pub type MotorForm = Scalar<(Motor, Motor)>;

        /// The source cloud's points, its spreads along `x`, `y` and `z`; the true motion, a
        /// shift after a turn; and the noise that perturbs each target point.
        pub const N_POINTS: usize = 60;
        pub const CLOUD_SPREAD: [f64; 3] = [2.0, 1.0, 0.5];
        pub const SHIFT: [f64; 3] = [1.5, -0.5, 2.0];
        pub const TURN: [f64; 3] = [0.8, -0.6, 1.4];
        pub const NOISE: f64 = 0.02;

        /// The point at `(x, y, z)` with unit `w`, normalized (on the three-sphere, onto it).
        pub fn point(x: f64, y: f64, z: f64) -> P {
            P::from_coeffs([x, y, z, 1.0]).normalized()
        }

        /// A point's readings by the planes `x`, `y`, `z`, normalized: its coordinates in flat
        /// space, the first three of its unit 4-vector on the three-sphere.
        #[cfg_attr(not(test), allow(dead_code))] // flat points draw as they are
        pub fn coordinates(p: P) -> [f64; 3] {
            let p = p.normalized();
            [0, 1, 2].map(|i| {
                let mut c = [0.0; 4];
                c[i] = 1.0;
                (Plane::<()>::from_coeffs(c) & p).s()
            })
        }

        /// The motor's own metric: the scalar part of `M ~M`, the squares of its turn.
        pub fn motor_metric() -> MotorForm {
            let m = Motor::slot();
            m.reverse().scalar_product(m)
        }

        /// The residual of the one-sided equations for one correspondence, linear in the open
        /// motor: intrinsic, as the points are.
        pub fn residual(source: P, target: P) -> intrinsic::Flector<(Motor,)> {
            let (s, t) = (source.normalized(), target.normalized());
            t * Motor::slot() - Motor::slot() * s
        }

        /// The summed squared coefficients of the residuals: the bulk norm (the scalar part of
        /// `r ~r`, blind to ideal blades in flat space) plus the weight norm (the bulk norm of
        /// the dual).
        pub fn misfit(source: &[P], target: &[P]) -> MotorForm {
            source
                .iter()
                .zip(target)
                .fold(Scalar::zero(), |acc, (s, t)| {
                    let r = residual(*s, *t);
                    let (d, rd) = (r.dual(), r.dual().reverse());
                    acc + r.reverse().scalar_product(r) + rd.scalar_product(d)
                })
        }

        /// Fit a motor by the one-sided residual: the least finite mode of the misfit against
        /// the motor metric, normalized.
        pub fn fit_motor(source: &[P], target: &[P]) -> M {
            let (_, modes) = misfit(source, target).eigh_semidefinite(motor_metric());
            modes[0].unit()
        }

        /// A cloud spread anisotropically, by `CLOUD_SPREAD` along each coordinate.
        pub fn cloud(n: usize, rng: &mut Rng) -> Vec<P> {
            let [a, b, c] = CLOUD_SPREAD;
            (0..n)
                .map(|_| point(a * rng.normal(), b * rng.normal(), c * rng.normal()))
                .collect()
        }

        /// Each point perturbed by an independent Gaussian motor: every coefficient of its
        /// generator of deviation `sigma / 2`.
        pub fn jitter(points: &[P], sigma: f64, rng: &mut Rng) -> Vec<P> {
            points
                .iter()
                .map(|p| {
                    let noise = Line::from_coeffs(core::array::from_fn(|_| 0.5 * sigma * rng.normal()));
                    noise.exp() >> *p
                })
                .collect()
        }

        /// A shift after a turn, each the exponential of half its generator: the shift's along
        /// numga's `xw, yw, zw` (gax's `-e01, -e02, -e03`), the turn's in `xy, yz, zx` (`e12,
        /// e23, e31`).
        pub fn motion(shift: [f64; 3], turn: [f64; 3]) -> M {
            let [sx, sy, sz] = shift.map(|v| -0.5 * v);
            let [xy, yz, zx] = turn.map(|v| 0.5 * v);
            Line::from_coeffs([0.0, 0.0, 0.0, sx, sy, sz]).exp()
                * Line::from_coeffs([yz, zx, xy, 0.0, 0.0, 0.0]).exp()
        }

        /// The true motion.
        pub fn truth() -> M {
            motion(SHIFT, TURN)
        }

        /// The seeded, noisy correspondences: the source cloud and its image under the true
        /// motion, each point perturbed.
        pub fn correspondences() -> (Vec<P>, Vec<P>) {
            let mut rng = gax_numga_examples::rng::rng(0x5eed_0001);
            let source = cloud(N_POINTS, &mut rng);
            let moved: Vec<P> = source.iter().map(|p| truth() >> *p).collect();
            let target = jitter(&moved, NOISE, &mut rng);
            (source, target)
        }

        /// The squared distances summed between two clouds: two unit points differ by a
        /// difference whose dual's norm is their distance (in flat space; the chord on the
        /// three-sphere).
        pub fn cartesian(a: &[P], b: &[P]) -> f64 {
            a.iter()
                .zip(b)
                .map(|(p, q)| (p.normalized() - q.normalized()).dual().norm_squared())
                .sum()
        }
    };
}

/// Flat space, and numga's centered fit there.
mod registration {
    registration_core!(gax::pga3d);
    use gax::pga3d::Rotor;

    /// The rotor that maximizes the alignment of corresponding vectors (planes through the
    /// origin): `Σ target · (R source ~R)`, a symmetric form on rotors.
    pub fn fit_rotor(source: &[Plane], target: &[Plane]) -> Unit<Rotor> {
        let alignment: Scalar<(Rotor, Rotor)> = source
            .iter()
            .zip(target)
            .map(|(s, t)| t.scalar_product((Rotor::slot() * *s) * Rotor::slot().reverse()))
            .sum();
        let (_, rotors) = alignment.eigh(Scalar::identity());
        rotors[3].unit()
    }

    /// The mean of unit points (a point of unit weight).
    pub fn mean(points: &[Point]) -> Point {
        let sum: Point = points.iter().map(|p| p.unitized()).sum();
        sum.gp(1.0 / points.len() as f64)
    }

    /// A centered point as the plane through the origin normal to its offset: its vector.
    fn vector(p: Point, centre: Point) -> Plane {
        let d = p.unitized() - centre;
        Plane::orthogonal_to(d)
    }

    /// The centered fit: the rotor aligning the centered clouds, then the translation carrying
    /// the rotated source centroid onto the target's.
    pub fn fit_motor_alignment(source: &[Point], target: &[Point]) -> M {
        let (sm, tm) = (mean(source), mean(target));
        let sv: Vec<_> = source.iter().map(|p| vector(*p, sm)).collect();
        let tv: Vec<_> = target.iter().map(|p| vector(*p, tm)).collect();
        let rotation = fit_rotor(&sv, &tv);
        Motor::between(rotation >> sm, tm) * rotation.widen::<Motor>()
    }
}

/// The three-sphere.
mod spherical {
    registration_core!(super::sga3d);
}

use registration::*;

/// One fit in the rectangle `rect`, its points as flat ones: the source, the target, and the
/// source moved part of the way along the estimated motion, seen from azimuth `az`, with the
/// title above and the remaining distance (`rms`, in the fit's own space) below.
#[allow(clippy::too_many_arguments)]
fn panel(
    c: &mut Canvas,
    rect: Rect,
    (source, target, moving): (&[Point], &[Point], &[Point]),
    rms: f64,
    distance: f32,
    az: f32,
    title: &str,
) {
    let centre = (mean(source) + mean(target)).gp(0.5);
    let cam = Camera::orbit(rect, centre, distance, az, 0.5, Lens::Perspective(0.6));
    panel3(c, rect, cam, |scene| {
        for ((p, q), m) in source.iter().zip(target).zip(moving) {
            scene.dot(*p, Marker::Dot, 4.0, palette::GRID);
            scene.dot(*q, Marker::Cross, 7.0, palette::SKY);
            scene.seg(*m, *q, 0.8, palette::RED.faded(0.45));
            scene.dot(*m, Marker::Dot, 5.0, palette::RED);
        }
    });
    // Scaled with the panel: 13 in a panel as tall as the canvas.
    let size = 13.0 * rect.height() / HEIGHT;
    // The title a sixth of the way down the middle, the distance near the bottom.
    let down = |f: f32| rect.top_middle() + Point2::direction(0.0, f * rect.height());
    c.text(title, down(0.17), size, palette::INK, Align::Center);
    let note = format!("RMS DISTANCE {rms:.4}");
    c.text(&note, down(0.94), size, palette::INK, Align::Center);
}

/// The root of the mean of the squared distances.
#[allow(clippy::disallowed_methods)] // a statistic: the root of the mean squared distance
fn rms(squares: f64, n: usize) -> f64 {
    (squares / n as f64).sqrt()
}

const SECONDS: f32 = 6.0;

fn draw(c: &mut Canvas, t: f32) {
    let phase = f64::from(t / SECONDS) * core::f64::consts::TAU;
    // Fly over, hold, and fly back.
    let s = (1.4 * haversine(phase)).min(1.0);
    let az = 0.6 + 0.35 * wave(phase as f32);
    backdrop(c);
    let screen = c.rect();
    // Flat space: the centered fit and the one-sided one.
    let (source, target) = correspondences();
    let identity = Motor::translation(Point::direction(0.0, 0.0, 0.0));
    let fits = [
        (fit_motor_alignment(&source, &target), "CENTERED ALIGNMENT"),
        (fit_motor(&source, &target), "ONE-SIDED RESIDUAL"),
    ];
    for (i, (estimate, title)) in fits.into_iter().enumerate() {
        let along = Motor::interpolate(identity, estimate, s);
        let moving: Vec<Point> = source.iter().map(|p| along >> *p).collect();
        let error = rms(cartesian(&moving, &target), source.len());
        let clouds = (&source[..], &target[..], &moving[..]);
        panel(c, screen.column(i, 3), clouds, error, 24.0, az, title);
    }
    // The three-sphere, its points drawn by their x, y, z readings.
    let (source, target) = spherical::correspondences();
    let along = spherical::fit_motor(&source, &target).log().gp(s).exp();
    let moving: Vec<spherical::P> = source.iter().map(|p| along >> *p).collect();
    let error = rms(spherical::cartesian(&moving, &target), source.len());
    let flat = |points: &[spherical::P]| -> Vec<Point> {
        (points.iter())
            .map(|p| {
                let [x, y, z] = spherical::coordinates(*p);
                Point::xyz(x, y, z)
            })
            .collect()
    };
    let (source, target, moving) = (flat(&source), flat(&target), flat(&moving));
    let clouds = (&source[..], &target[..], &moving[..]);
    let title = "ONE-SIDED, THREE-SPHERE";
    panel(c, screen.column(2, 3), clouds, error, 8.0, az, title);
    caption(
        c,
        "REGISTRATION: A MOTOR FROM CORRESPONDING POINTS",
        "SOURCE (GREY), TARGET (BLUE), SOURCE MOVED BY THE FIT (RED)",
    );
}

fn main() {
    run(Anim::new("registration", SECONDS).size(960, 480), draw);
}

/// numga's tests, run in flat space and on the three-sphere, as numga parametrizes them.
#[cfg(test)]
macro_rules! registration_tests {
    ($tests:ident, $core:ident) => {
        mod $tests {
            use crate::$core::*;
            use gax::ApproxEq;
            use gax_numga_examples::rng::rng;

            fn moved(m: M, points: &[P]) -> Vec<P> {
                points.iter().map(|p| m >> *p).collect()
            }

            /// The largest coordinate difference between corresponding points.
            fn gap(a: &[P], b: &[P]) -> f64 {
                (a.iter().zip(b))
                    .flat_map(|(p, q)| {
                        let (p, q) = (coordinates(*p), coordinates(*q));
                        [0, 1, 2].map(|i| (p[i] - q[i]).abs())
                    })
                    .fold(0.0, f64::max)
            }

            /// The residual is linear in the motor, vanishes at the truth, and its misfit is a
            /// symmetric form on motors (8 x 8).
            #[test]
            fn correspondence_residual_is_linear_in_the_motor() {
                let source = cloud(20, &mut rng(1));
                let target = moved(truth(), &source);
                for (s, t) in source.iter().zip(&target) {
                    let r = residual(*s, *t).of(truth().into_inner());
                    assert!(r.max_abs_diff(&r.gp(0.0)) < 1e-8, "{r:?}");
                }
                let m = misfit(&source, &target);
                assert!(m.max_abs_diff(&m.swap()) < 1e-12);
            }

            #[test]
            fn exact_correspondences_recover_the_motor() {
                let source = cloud(30, &mut rng(2));
                let target = moved(truth(), &source);
                let estimate = fit_motor(&source, &target);
                assert!(gap(&moved(estimate, &source), &target) < 1e-10);
            }

            /// The mean pose error under numga's noise, a Gaussian motor of deviation 0.05 on
            /// each point, stays within numga's 0.03 (the stream is the port's).
            #[test]
            fn noisy_correspondences_recover_the_pose_within_noise() {
                let mut draws = rng(3);
                let source = cloud(200, &mut draws);
                let target = jitter(&moved(truth(), &source), 0.05, &mut draws);
                let estimate = fit_motor(&source, &target);
                let (fitted, true_) = (moved(estimate, &source), moved(truth(), &source));
                let error: f64 = (fitted.iter().zip(&true_))
                    .map(|(p, q)| (p.normalized() - q.normalized()).dual().norm())
                    .sum::<f64>()
                    / 200.0;
                assert!(error < 0.03, "{error}");
            }

            #[test]
            fn estimate_is_a_trusted_unit_motor() {
                let source = cloud(30, &mut rng(5));
                let estimate = fit_motor(&source, &moved(truth(), &source)).into_inner();
                let one = estimate * estimate.reverse();
                let identity = motion([0.0; 3], [0.0; 3]).into_inner();
                assert!(one.max_abs_diff(&identity) < 1e-12);
            }

            /// numga's shift alone, `exp(0.75 xw - 0.25 yw + zw)`.
            #[test]
            fn translation_is_recovered_by_the_same_fit() {
                let source = cloud(100, &mut rng(6));
                let target = moved(motion([1.5, -0.5, 2.0], [0.0; 3]), &source);
                let estimate = fit_motor(&source, &target);
                assert!(gap(&moved(estimate, &source), &target) < 1e-10);
            }

            /// numga's scenario check: the fitted motor brings the cloud within the noise of the
            /// target, an RMS distance below 0.15.
            #[test]
            fn the_scenario_lands_within_the_noise() {
                let (source, target) = correspondences();
                let aligned = moved(fit_motor(&source, &target), &source);
                let rms = crate::rms(cartesian(&aligned, &target), N_POINTS);
                assert!(rms < 0.15, "{rms}");
            }
        }
    };
}

#[cfg(test)]
registration_tests!(flat_tests, registration);
#[cfg(test)]
registration_tests!(spherical_tests, spherical);

#[cfg(test)]
mod tests {
    use super::registration::*;
    use gax::ApproxEq;
    use gax::pga3d::Point;

    /// The coordinates read by the planes are the point's.
    #[test]
    fn coordinates_read_the_point() {
        let c = coordinates(Point::xyz(0.3, -1.2, 2.5).gp(2.0));
        assert!(
            (c[0] - 0.3).abs() + (c[1] + 1.2).abs() + (c[2] - 2.5).abs() < 1e-15,
            "{c:?}"
        );
    }

    /// The true motion is numga's: its shift and turn as numga's twists.
    #[test]
    fn the_true_motion_is_numgas() {
        let shift = gax::pga3d::Line::new(0.0, 0.0, 0.0, -0.75, 0.25, -1.0).exp();
        let turn = gax::pga3d::Line::new(-0.3, 0.7, 0.4, 0.0, 0.0, 0.0).exp();
        assert!(
            truth()
                .into_inner()
                .approx_eq(&(shift * turn).into_inner(), 1e-15)
        );
    }

    /// Centering then aligning minimizes the summed squared distances, so the one-sided fit
    /// cannot beat it on that measure (numga has dropped the centered fit; the port keeps the
    /// comparison, an exact optimality, not a matter of rounding).
    #[test]
    fn sandwich_alignment_is_the_cartesian_optimum() {
        let (source, target) = correspondences();
        let a = fit_motor_alignment(&source, &target);
        let b = fit_motor(&source, &target);
        let aligned: Vec<Point> = source.iter().map(|p| a >> *p).collect();
        let one_sided: Vec<Point> = source.iter().map(|p| b >> *p).collect();
        assert!(cartesian(&aligned, &target) <= cartesian(&one_sided, &target) * (1.0 + 1e-12));
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws(super::draw, 0.5);
    }
}
