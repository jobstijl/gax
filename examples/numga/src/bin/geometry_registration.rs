//! numga's `geometry/registration`: a rigid motion from corresponding points, two ways, in
//! PGA3D. The centered fit is the classical one: align the centered clouds by the rotor that
//! maximizes the sandwich alignment (a symmetric eigenproblem on rotors), then carry the rotated
//! centroid onto the target's by the square root of a ratio of points. The one-sided fit sets up
//! `target M - M source = 0`, linear in the open motor `M`, sums the squared coefficients of
//! the residual into a form on motors, and takes the least mode against the motor's own metric
//! (the scalar part of `M ~M`), which measures the rotor part alone; normalizing makes it a
//! motor. Both reach the least-squares pose. The animation flies the source cloud along each
//! estimated motion onto the target, with each point tied to its correspondent.
//!
//! The motor metric is singular on the translation part; numga's general eigensolver sends
//! those modes to infinity, and so does gax's `eigh_semidefinite`, which lists the finite
//! modes first.

use gax::Unit;
use gax::pga3d::{Line, Motor, Plane, Point, Rotor, Scalar};
use gax_light::fade;
use gax_numga_examples::rng::{Draw, Rng, rng};
use gax_numga_examples::scene3::panel3;
use gax_numga_examples::{
    Align, Anim, Camera, Canvas, Lens, Marker, Point2, Rect, backdrop, caption, palette, run,
};

mod registration {
    use super::*;

    pub type P = Point<(), f64>;
    pub type M = Unit<Motor<(), f64>>;
    pub type MotorForm = Scalar<(Motor, Motor), f64>;

    /// The motor's own metric: the scalar part of `M ~M`, the squares of the rotor part.
    pub fn motor_metric() -> MotorForm {
        let m = Motor::slot();
        m.reverse().scalar_product(m)
    }

    /// The residual of the one-sided equations for one correspondence, linear in the open motor.
    pub fn residual(source: P, target: P) -> gax::pga3d::Flector<(Motor,), f64> {
        let (s, t) = (source.unitized(), target.unitized());
        t * Motor::slot() - Motor::slot() * s
    }

    /// The summed squared coefficients of the residuals: the bulk norm (the scalar part of
    /// `r ~r`, blind to ideal blades) plus the weight norm (the bulk norm of the dual).
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

    /// Fit a motor by the one-sided residual: the least finite mode of the misfit against the
    /// motor metric, normalized.
    pub fn fit_motor(source: &[P], target: &[P]) -> M {
        let (_, modes) = misfit(source, target).eigh_semidefinite(motor_metric());
        modes[0].normalized()
    }

    /// The rotor that maximizes the alignment of corresponding vectors (planes through the
    /// origin): `Σ target · (R source ~R)`, a symmetric form on rotors.
    pub fn fit_rotor(source: &[Plane<(), f64>], target: &[Plane<(), f64>]) -> Unit<Rotor<(), f64>> {
        let alignment = source
            .iter()
            .zip(target)
            .fold(Scalar::<(Rotor, Rotor), f64>::zero(), |acc, (s, t)| {
                acc + t.scalar_product((Rotor::slot() * *s) * Rotor::slot().reverse())
            });
        let (_, rotors) = alignment.eigh();
        rotors[3].normalized()
    }

    /// The mean of unit points (a point of unit weight).
    pub fn mean(points: &[P]) -> P {
        let sum = points
            .iter()
            .fold(Point::zero(), |acc, p| acc + p.unitized());
        sum.gp(1.0 / points.len() as f64)
    }

    /// A centered point as the plane through the origin normal to its offset: its vector.
    fn vector(p: P, centre: P) -> Plane<(), f64> {
        let d = p.unitized() - centre;
        Plane::orthogonal_to(d)
    }

    /// The centered fit: the rotor aligning the centered clouds, then the translation carrying
    /// the rotated source centroid onto the target's.
    pub fn fit_motor_alignment(source: &[P], target: &[P]) -> M {
        let (sm, tm) = (mean(source), mean(target));
        let sv: Vec<_> = source.iter().map(|p| vector(*p, sm)).collect();
        let tv: Vec<_> = target.iter().map(|p| vector(*p, tm)).collect();
        let rotation = fit_rotor(&sv, &tv);
        Motor::between(rotation >> sm, tm) * rotation.widen::<Motor<(), f64>>()
    }

    /// A cloud stretched along x.
    pub fn cloud(n: usize, rng: &mut Rng) -> Vec<P> {
        (0..n)
            .map(|_| Point::xyz(2.0 * rng.normal(), rng.normal(), 0.5 * rng.normal()))
            .collect()
    }

    /// Move each point by an independent Gaussian translation.
    pub fn jitter(points: &[P], sigma: f64, rng: &mut Rng) -> Vec<P> {
        points
            .iter()
            .map(|p| {
                let t = Motor::translation(
                    sigma * rng.normal(),
                    sigma * rng.normal(),
                    sigma * rng.normal(),
                );
                t >> *p
            })
            .collect()
    }

    /// The true motion: a translation after a rotation, both from bivectors.
    pub fn truth() -> M {
        Line::new(0.0, 0.0, 0.0, -0.75, 0.25, -1.0).exp()
            * Line::new(-0.3, 0.7, 0.4, 0.0, 0.0, 0.0).exp()
    }

    /// The same seeded, noisy correspondences for both fits.
    pub fn correspondences() -> (Vec<P>, Vec<P>) {
        let mut rng = rng(0x5eed_0001);
        let source = cloud(60, &mut rng);
        let moved: Vec<P> = source.iter().map(|p| truth() >> *p).collect();
        let target = jitter(&moved, 0.02, &mut rng);
        (source, target)
    }

    /// The squared distances summed between two clouds: two unit points differ by a
    /// direction, whose ideal norm is their distance.
    pub fn cartesian(a: &[P], b: &[P]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(p, q)| (p.unitized() - q.unitized()).ideal_norm_squared())
            .sum()
    }
}

use registration::*;

/// One fit in the rectangle `rect`: the source moved part of the way `s` along the estimated
/// motion toward the target, seen from azimuth `az`, with the title above and the remaining
/// distance below.
#[allow(clippy::too_many_arguments)]
fn panel(
    c: &mut Canvas,
    rect: Rect,
    source: &[P],
    target: &[P],
    estimate: M,
    s: f64,
    az: f32,
    title: &str,
) {
    let centre = (mean(source) + mean(target)).gp(0.5);
    let cam = Camera::orbit(
        rect.width() as usize,
        rect.height() as usize,
        centre,
        15.0,
        az,
        0.5,
        Lens::Perspective(0.6),
    );
    let identity = Motor::<(), f64>::translation(0.0, 0.0, 0.0);
    let along = Motor::interpolate(identity, estimate, s);
    let moving: Vec<P> = source.iter().map(|p| along >> *p).collect();
    panel3(c, rect, cam, |scene| {
        for ((p, q), m) in source.iter().zip(target).zip(&moving) {
            scene.dot(*p, Marker::Dot, 4.0, palette::grid());
            scene.dot(*q, Marker::Cross, 7.0, palette::sky());
            scene.seg(*m, *q, 0.8, fade(palette::red(), 0.45));
            scene.dot(*m, Marker::Dot, 5.0, palette::red());
        }
    });
    let size = (rect.height() / 32.0).clamp(8.0, 13.0);
    // The title at a sixth of the way down the middle, the distance near the bottom.
    let down = |f: f32| rect.top_middle() + Point2::direction(0.0, f * rect.height());
    c.text(title, down(0.17), size, palette::ink(), Align::Center);
    let rms = (cartesian(&moving, target) / source.len() as f64).sqrt();
    let note = format!("RMS DISTANCE {rms:.4}");
    c.text(&note, down(0.94), size, palette::ink(), Align::Center);
}

const SECONDS: f32 = 6.0;

fn draw(c: &mut Canvas, t: f32) {
    let phase = f64::from(t / SECONDS) * core::f64::consts::TAU;
    // Fly over, hold, and fly back.
    let s = (1.4 * (0.5 - 0.5 * phase.cos())).min(1.0);
    let az = 0.6 + 0.35 * (phase as f32).sin();
    let (source, target) = correspondences();
    let fits = [
        (
            fit_motor_alignment(&source, &target),
            "CENTERED SANDWICH ALIGNMENT",
        ),
        (fit_motor(&source, &target), "ONE-SIDED MOTOR RESIDUAL"),
    ];
    backdrop(c);
    let screen = c.rect();
    for (i, (estimate, title)) in fits.into_iter().enumerate() {
        let rect = screen.column(i, 2);
        panel(c, rect, &source, &target, estimate, s, az, title);
    }
    caption(
        c,
        "REGISTRATION: A MOTOR FROM CORRESPONDING POINTS",
        "SOURCE (GREY), TARGET (BLUE), SOURCE MOVED BY THE FIT (RED)",
    );
}

fn main() {
    run(Anim::new("registration", SECONDS).size(960, 480), draw);
}

#[cfg(test)]
mod tests {
    use super::registration::*;
    use gax::ApproxEq;
    use gax::pga3d::Motor;
    use gax_numga_examples::rng::rng;

    fn close(a: &[P], b: &[P], tol: f64) -> bool {
        a.iter().zip(b).all(|(p, q)| p.approx_eq(q, tol))
    }

    /// The residual is linear in the motor, vanishes at the truth, and its misfit is a
    /// symmetric form on motors (8 x 8).
    #[test]
    fn correspondence_residual_is_linear_in_the_motor() {
        let source = cloud(20, &mut rng(1));
        let target: Vec<P> = source.iter().map(|p| truth() >> *p).collect();
        for (s, t) in source.iter().zip(&target) {
            let r = residual(*s, *t).of(truth().into_inner());
            assert!(r.max_abs_diff(&gax::pga3d::Flector::zero()) < 1e-8, "{r:?}");
        }
        let m = misfit(&source, &target);
        assert!(m.approx_eq(&m.swap(), 1e-12));
    }

    #[test]
    fn exact_correspondences_recover_the_motor() {
        let source = cloud(30, &mut rng(2));
        let target: Vec<P> = source.iter().map(|p| truth() >> *p).collect();
        let estimate = fit_motor(&source, &target);
        let moved: Vec<P> = source.iter().map(|p| estimate >> *p).collect();
        assert!(close(&moved, &target, 1e-10));
    }

    /// The mean pose error under noise stays well inside the noise (the seed differs from
    /// numga's; the bound is numga's).
    #[test]
    fn noisy_correspondences_recover_the_pose_within_noise() {
        let mut draws = rng(3);
        let source = cloud(200, &mut draws);
        let moved: Vec<P> = source.iter().map(|p| truth() >> *p).collect();
        let target = jitter(&moved, 0.05, &mut draws);
        let estimate = fit_motor(&source, &target);
        let error: f64 = source
            .iter()
            .map(|p| ((estimate >> *p) & (truth() >> *p)).norm())
            .sum::<f64>()
            / 200.0;
        assert!(error < 0.01, "{error}");
    }

    #[test]
    fn estimate_is_a_trusted_unit_motor() {
        let source = cloud(30, &mut rng(5));
        let target: Vec<P> = source.iter().map(|p| truth() >> *p).collect();
        let estimate = fit_motor(&source, &target).into_inner();
        let one = estimate * estimate.reverse();
        let identity = Motor::<(), f64>::translation(0.0, 0.0, 0.0).into_inner();
        assert!(one.approx_eq(&identity, 1e-12));
    }

    #[test]
    fn translation_is_recovered_by_the_same_fit() {
        let source = cloud(100, &mut rng(6));
        let shift = gax::pga3d::Line::new(0.0, 0.0, 0.0, -0.75, 0.25, -1.0).exp();
        let target: Vec<P> = source.iter().map(|p| shift >> *p).collect();
        let estimate = fit_motor(&source, &target);
        let moved: Vec<P> = source.iter().map(|p| estimate >> *p).collect();
        assert!(close(&moved, &target, 1e-10));
    }

    /// Centering then aligning minimizes the summed squared distances, so the one-sided fit
    /// cannot beat it on that measure.
    #[test]
    fn sandwich_alignment_is_the_cartesian_optimum() {
        let (source, target) = correspondences();
        let a = fit_motor_alignment(&source, &target);
        let b = fit_motor(&source, &target);
        let aligned: Vec<P> = source.iter().map(|p| a >> *p).collect();
        let one_sided: Vec<P> = source.iter().map(|p| b >> *p).collect();
        assert!(cartesian(&aligned, &target) <= cartesian(&one_sided, &target) * (1.0 + 1e-12));
        // Both land within the noise of the target.
        assert!(cartesian(&one_sided, &target) / 60.0 < 4.0 * 3.0 * 0.02 * 0.02);
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws(super::draw, 0.5);
    }
}
