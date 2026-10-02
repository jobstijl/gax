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
//! The motor metric is singular on the translation part, and numga solves the pencil with a
//! general (non-symmetric) eigensolver that sends those modes to infinity. gax has the symmetric
//! definite solver only, so the pencil is turned round: the finite modes of `(A, B)`, `B`
//! semidefinite and `A + σB` definite, are the modes of `B.eigh_with(A + σB)` with eigenvalue
//! `μ = 1 / (λ + σ)`, and the infinite ones have `μ = 0`. The least `λ` is the greatest `μ`.

use gax::Unit;
use gax::pga3d::{Line, Motor, Plane, Point, Rotor, Scalar};
use gax_numga_examples::{
    Align, Anim, Camera, Canvas, Lens, Marker, Scene3, backdrop, caption, palette, run,
};

mod registration {
    use super::*;

    pub type P = Point<(), f64>;
    pub type M = Unit<Motor<(), f64>>;
    pub type MotorForm = Scalar<(Motor, Motor), f64>;

    /// A small xorshift generator with Gaussian draws (numga's NumPy streams cannot be
    /// reproduced, so the seeds differ from numga's).
    pub struct Rng(pub u64);
    impl Rng {
        pub fn uniform(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        }
        /// A standard normal draw (Box and Muller).
        pub fn normal(&mut self) -> f64 {
            let (u, v) = (self.uniform(), self.uniform());
            (-2.0 * u.ln()).sqrt() * (core::f64::consts::TAU * v).cos()
        }
    }

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

    /// The mean of a form's diagonal: its scale.
    pub fn scale(form: MotorForm) -> f64 {
        (0..8).map(|i| form.c[0][i][i]).sum::<f64>() / 8.0
    }

    /// Fit a motor by the one-sided residual: the least finite mode of the misfit against the
    /// motor metric, found as the greatest mode of the turned-round pencil, then normalized.
    pub fn fit_motor(source: &[P], target: &[P]) -> M {
        let a = misfit(source, target);
        let b = motor_metric();
        let sigma = scale(a) / scale(b);
        let (_, modes) = b.eigh_with(a + b.gp(sigma));
        modes[7].normalized()
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
        Plane::new(d.e032(), d.e013(), d.e021(), 0.0)
    }

    /// The centered fit: the rotor aligning the centered clouds, then the translation carrying
    /// the rotated source centroid onto the target's (the square root of their ratio).
    pub fn fit_motor_alignment(source: &[P], target: &[P]) -> M {
        let (sm, tm) = (mean(source), mean(target));
        let sv: Vec<_> = source.iter().map(|p| vector(*p, sm)).collect();
        let tv: Vec<_> = target.iter().map(|p| vector(*p, tm)).collect();
        let rotation = fit_rotor(&sv, &tv);
        let translation = (tm / (rotation >> sm)).sqrt();
        translation.widen::<Motor<(), f64>>() * rotation.widen::<Motor<(), f64>>()
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
        let mut rng = Rng(0x5eed_0001);
        let source = cloud(60, &mut rng);
        let moved: Vec<P> = source.iter().map(|p| truth() >> *p).collect();
        let target = jitter(&moved, 0.02, &mut rng);
        (source, target)
    }

    /// The squared distances summed between two clouds.
    pub fn cartesian(a: &[P], b: &[P]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(p, q)| {
                let (p, q) = (p.to_euclidean(), q.to_euclidean());
                (0..3).map(|i| (p[i] - q[i]).powi(2)).sum::<f64>()
            })
            .sum()
    }
}

use registration::*;

fn xyz(p: P) -> [f32; 3] {
    let [x, y, z] = p.to_euclidean();
    [x as f32, y as f32, z as f32]
}

fn panel(c: &mut Canvas, source: &[P], target: &[P], estimate: M, s: f64, az: f32, title: &str) {
    backdrop(c);
    let (sm, tm) = (xyz(mean(source)), xyz(mean(target)));
    let centre = [0, 1, 2].map(|i| 0.5 * (sm[i] + tm[i]));
    let cam = Camera::orbit(
        c.width,
        c.height,
        centre,
        15.0,
        az,
        0.5,
        Lens::Perspective(0.6),
    );
    let mut scene = Scene3::new(cam);
    // The source moved part of the way along the estimated motion.
    let identity = Motor::<(), f64>::translation(0.0, 0.0, 0.0);
    let along = Motor::interpolate(identity, estimate, s);
    let moving: Vec<P> = source.iter().map(|p| along >> *p).collect();
    for ((p, q), m) in source.iter().zip(target).zip(&moving) {
        scene.dot(xyz(*p), Marker::Dot, 4.0, palette::grid());
        scene.dot(xyz(*q), Marker::Cross, 7.0, palette::sky());
        scene.seg(xyz(*m), xyz(*q), 0.8, palette::red(), 0.45);
        scene.dot(xyz(*m), Marker::Dot, 5.0, palette::red());
    }
    scene.draw(c);
    let size = (c.height as f32 / 32.0).clamp(8.0, 13.0);
    let mid = c.width as f32 * 0.5;
    c.text(
        title,
        mid,
        c.height as f32 * 0.17,
        size,
        palette::ink(),
        Align::Center,
    );
    let rms = (cartesian(&moving, target) / source.len() as f64).sqrt();
    c.text(
        &format!("RMS DISTANCE {rms:.4}"),
        mid,
        c.height as f32 * 0.94,
        size,
        palette::ink(),
        Align::Center,
    );
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
    let w = c.width / 2;
    for (i, (estimate, title)) in fits.into_iter().enumerate() {
        let mut sub = Canvas::new(w, c.height);
        panel(&mut sub, &source, &target, estimate, s, az, title);
        c.blit(&sub, i * w, 0);
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
    use gax::pga3d::Motor;

    fn close(a: &[P], b: &[P], tol: f64) -> bool {
        a.iter().zip(b).all(|(p, q)| {
            let (p, q) = (p.to_euclidean(), q.to_euclidean());
            (0..3).all(|i| (p[i] - q[i]).abs() <= tol)
        })
    }

    /// The residual is linear in the motor, vanishes at the truth, and its misfit is a
    /// symmetric form on motors (8 x 8).
    #[test]
    fn correspondence_residual_is_linear_in_the_motor() {
        let source = cloud(20, &mut Rng(1));
        let target: Vec<P> = source.iter().map(|p| truth() >> *p).collect();
        for (s, t) in source.iter().zip(&target) {
            let r = residual(*s, *t).of(truth().into_inner());
            assert!(r.c.iter().all(|v| v.abs() < 1e-8), "{r:?}");
        }
        let m = misfit(&source, &target);
        for i in 0..8 {
            for j in 0..8 {
                assert!((m.c[0][i][j] - m.c[0][j][i]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn exact_correspondences_recover_the_motor() {
        let source = cloud(30, &mut Rng(2));
        let target: Vec<P> = source.iter().map(|p| truth() >> *p).collect();
        let estimate = fit_motor(&source, &target);
        let moved: Vec<P> = source.iter().map(|p| estimate >> *p).collect();
        assert!(close(&moved, &target, 1e-10));
    }

    /// The mean pose error under noise stays well inside the noise (the seed differs from
    /// numga's; the bound is numga's).
    #[test]
    fn noisy_correspondences_recover_the_pose_within_noise() {
        let mut rng = Rng(3);
        let source = cloud(200, &mut rng);
        let moved: Vec<P> = source.iter().map(|p| truth() >> *p).collect();
        let target = jitter(&moved, 0.05, &mut rng);
        let estimate = fit_motor(&source, &target);
        let error: f64 = source
            .iter()
            .map(|p| cartesian(&[estimate >> *p], &[truth() >> *p]).sqrt())
            .sum::<f64>()
            / 200.0;
        assert!(error < 0.01, "{error}");
    }

    #[test]
    fn estimate_is_a_trusted_unit_motor() {
        let source = cloud(30, &mut Rng(5));
        let target: Vec<P> = source.iter().map(|p| truth() >> *p).collect();
        let estimate = fit_motor(&source, &target).into_inner();
        let one = estimate * estimate.reverse();
        let identity = Motor::<(), f64>::translation(0.0, 0.0, 0.0).into_inner();
        for (a, b) in one.c.iter().zip(identity.c) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn translation_is_recovered_by_the_same_fit() {
        let source = cloud(100, &mut Rng(6));
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
        let mut draw = super::draw;
        let c = gax_numga_examples::app::frame(
            &gax_numga_examples::Anim::new("t", 1.0).size(320, 180),
            0.5,
            &mut draw,
        );
        assert!(c.mean()[0] > 0.0);
    }
}
