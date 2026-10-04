//! numga's `geometry/camera_fit`: a camera pose recovered by gradient descent, in PGA3D. The
//! camera is a join then a meet with the point left open, `(O & X) ^ screen`, a central
//! projection onto the plane `z = 1`; the rig moves it by conjugation, `rig >> camera(rig << X)`,
//! and the rig is the exponential of a bivector. The image misfit is differentiated with respect
//! to that bivector through `exp`, the sandwiches, the bind and the unitization. numga uses
//! JAX's reverse mode; here the coefficients are gax's forward-mode dual numbers (`Dual<f64, 6>`,
//! one derivative per bivector coefficient), so the same generic code returns the misfit and
//! its gradient in one pass. Nothing is linearized by hand. The animation replays the descent:
//! the estimated camera (red) flies onto the true one (blue), its images onto the observed ones,
//! and the misfit falls.

use gax::dual::{Dual, gradient};
use gax::pga3d::{Line, Plane, Point};
use gax::{Real, Unit};
use gax_light::{fade, mix};
use gax_numga_examples::rng::{Draw, rng};
use gax_numga_examples::{
    Align, Anim, Axes, Camera, Canvas, Lens, Light, Marker, Point2, Scene3, backdrop, caption,
    from_above, palette, run,
};
use std::sync::OnceLock;

mod camera_fit {
    use super::*;

    pub type P = Point<(), f64>;
    pub type B = Line<(), f64>;
    pub type D = Dual<f64, 6>;

    /// The world: thirty points about the origin.
    pub fn world() -> Vec<P> {
        let mut rng = rng(0xca3e_0001);
        (0..30)
            .map(|_| Point::xyz(rng.normal(), rng.normal(), rng.normal()))
            .collect()
    }

    /// The images of the world points seen from the rig, as points on its screen `z = 1`, in the
    /// rig's frame. Generic over the coefficients, so that dual numbers carry the gradient.
    pub fn image<T: Real>(generator: Line<(), T>, world: &[P]) -> Vec<Point<(), T>> {
        let c = T::from_f64;
        let origin = Point::xyz(c(0.0), c(0.0), c(0.0));
        let screen = Plane::new(c(0.0), c(0.0), c(1.0), c(-1.0));
        // The camera: the join with the centre, met with the screen; a map on points.
        let camera = (origin & Point::slot()) ^ screen;
        let rig = generator.exp();
        // The camera moved by the rig: the map conjugated.
        let moved = rig >> camera.of(rig << Point::slot());
        world
            .iter()
            .map(|w| (rig << moved.of(w.map_coefs(T::from_f64))).unitized())
            .collect()
    }

    /// The mean squared distance between two sets of images: two images of unit weight differ
    /// by a direction, the displacement on the screen.
    pub fn misfit<T: Real>(images: &[Point<(), T>], observed: &[P]) -> T {
        let c = T::from_f64;
        let mut sum = c(0.0);
        for (a, o) in images.iter().zip(observed) {
            sum = sum + (*a - o.map_coefs(c)).ideal_norm_squared();
        }
        sum / c(images.len() as f64)
    }

    /// The true rig's generator, and the start of the descent.
    pub fn truth() -> B {
        Line::new(0.1, -0.2, 0.15, -0.3, 0.1, -2.5)
    }

    pub fn start() -> B {
        truth() + Line::new(0.3, 0.3, 0.3, -0.3, -0.3, -0.3)
    }

    /// The misfit and its gradient with respect to the generator's six coefficients.
    pub fn misfit_and_gradient(generator: B, world: &[P], observed: &[P]) -> (f64, B) {
        let (value, grad) = gradient(
            |g: [D; 6]| misfit(&image(Line::from_coeffs(g), world), observed),
            generator.c,
        );
        (value, Line::from_coeffs(grad))
    }

    /// Plain gradient descent with step 0.25: the generator after each step, with its misfit.
    pub fn descend(steps: usize) -> Vec<(B, f64)> {
        let world = world();
        let observed = image(truth(), &world);
        let mut g = start();
        let mut path = Vec::with_capacity(steps + 1);
        for _ in 0..=steps {
            let (value, grad) = misfit_and_gradient(g, &world, &observed);
            path.push((g, value));
            g -= grad.gp(0.25);
        }
        path
    }

    /// The pose error between two generators: the log of the relative motor, whose Euclidean
    /// part turns and whose ideal part shifts (the dual swaps the two).
    pub fn pose_error(a: B, b: B) -> (f64, f64) {
        let relative: B = (a.exp().inverse() * b.exp()).log();
        (relative.norm(), relative.dual().norm())
    }

    pub const STEPS: usize = 600;
}

use camera_fit::*;

/// The descent, computed once.
fn path() -> &'static [(B, f64)] {
    static PATH: OnceLock<Vec<(B, f64)>> = OnceLock::new();
    PATH.get_or_init(|| descend(STEPS))
}

/// A camera's frustum: its centre, and the corners of its screen.
fn frustum(s: &mut Scene3, generator: B, colour: Light, width: f32) {
    let rig: Unit<_> = generator.exp();
    let at = |x: f64, y: f64, z: f64| rig >> Point::xyz(x, y, z);
    let centre = at(0.0, 0.0, 0.0);
    let corners = [
        at(-0.8, -0.6, 1.0),
        at(0.8, -0.6, 1.0),
        at(0.8, 0.6, 1.0),
        at(-0.8, 0.6, 1.0),
    ];
    let colour = fade(colour, 0.9);
    for corner in corners {
        s.seg(centre, corner, width, colour);
    }
    let [a, b, c, d] = corners;
    s.polyline(&[a, b, c, d, a], width, colour);
    s.dot(centre, Marker::Dot, 6.0, colour);
}

const SECONDS: f32 = 10.0;

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let screen = c.rect();
    let (w, h) = (screen.width(), screen.height());
    let path = path();
    // Replay the descent on a logarithmic clock, so that the fast start is seen, then hold.
    let s = ((t / SECONDS) / 0.85).min(1.0);
    let k = (((STEPS + 1) as f32).powf(s) - 1.0).round() as usize;
    let (g, value) = path[k.min(STEPS)];
    let world = world();
    let observed = image(truth(), &world);
    let images = image(g, &world);
    // The scene on the left, its view turning slowly.
    let wl = (w * 0.55) as usize;
    let mut left = Canvas::new(wl, c.height);
    backdrop(&mut left);
    // Centred between the world's origin and the true camera.
    let origin = Point::xyz(0.0, 0.0, 0.0);
    let middle = (origin + (truth().exp() >> origin)).gp(0.5);
    let cam = Camera::orbit(
        wl,
        c.height,
        middle,
        17.0,
        -1.2 + 0.5 * (t / SECONDS * core::f32::consts::TAU).sin(),
        0.35,
        Lens::Perspective(0.65),
    );
    let mut scene = Scene3::new(cam);
    for p in &world {
        scene.dot(
            *p,
            Marker::Dot,
            4.0,
            mix(palette::grid(), palette::ink(), 0.5),
        );
    }
    frustum(&mut scene, truth(), palette::sky(), 2.0);
    frustum(&mut scene, g, palette::red(), 1.5);
    // The rays of the estimate, from its centre through each world point.
    let centre = g.exp() >> origin;
    for p in &world {
        scene.seg(centre, *p, 0.6, fade(palette::red(), 0.25));
    }
    scene.draw(&mut left);
    c.blit(&left, 0, 0);
    // The screen: the observed images and the current ones, tied together.
    let right = screen
        .part(0.55, 0.0, 1.0, 0.6)
        .inset(w * 0.05, h * 0.14, w * 0.02, h * 0.06);
    let ax = Axes::equal(right, Point2::xy(0.0, 0.0), 0.75);
    ax.frame(c, "THE SCREEN Z = 1", "", "");
    // The screen `z = 1` seen from above: its points with their `z` dropped.
    for (a, o) in images.iter().zip(&observed) {
        ax.line(
            c,
            from_above(*a),
            from_above(*o),
            0.8,
            fade(palette::red(), 0.5),
        );
    }
    let obs: Vec<_> = observed.iter().copied().map(from_above).collect();
    let cur: Vec<_> = images.iter().copied().map(from_above).collect();
    ax.scatter(c, &obs, Marker::Cross, 8.0, palette::sky());
    ax.scatter(c, &cur, Marker::Dot, 5.0, palette::red());
    // The misfit over the steps, on a log scale.
    let lower = screen
        .part(0.55, 0.6, 1.0, 1.0)
        .inset(w * 0.05, h * 0.06, w * 0.02, h * 0.075);
    let least = path.iter().fold(f64::MAX, |m, p| m.min(p.1)).max(1e-30) as f32;
    let most = path.iter().fold(0.0f64, |m, p| m.max(p.1)) as f32;
    let mx = Axes::new(lower, [1.0, (STEPS + 1) as f32], [least * 0.5, most * 2.0])
        .log_x()
        .log_y();
    mx.frame(c, "", "STEP", "MEAN SQUARED IMAGE MISFIT");
    // The chart's points: (step, misfit).
    let curve: Vec<Point2> = path
        .iter()
        .enumerate()
        .map(|(i, p)| Point2::xy((i + 1) as f32, p.1 as f32))
        .collect();
    mx.polyline(c, &curve, 1.0, palette::grid());
    mx.polyline(c, &curve[..=k.min(STEPS)], 2.0, palette::yellow());
    let now = Point2::xy((k + 1) as f32, value as f32);
    mx.scatter(c, &[now], Marker::Dot, 7.0, palette::yellow());
    let (turn, shift) = pose_error(truth(), g);
    c.text(
        &format!("STEP {k}   POSE ERROR: TURN {turn:.4}  SHIFT {shift:.4}"),
        screen.bottom_left() + Point2::direction(w * 0.02, -14.0),
        12.0,
        palette::ink(),
        Align::Left,
    );
    caption(
        c,
        "CAMERA FIT: DESCENT THROUGH EXP, SANDWICH AND BIND",
        "THE GRADIENT BY GAX'S DUAL NUMBERS (PGA3D)",
    );
}

fn main() {
    run(Anim::new("camera fit", SECONDS).size(960, 540), draw);
}

#[cfg(test)]
mod tests {
    use super::camera_fit::*;

    /// numga's checks: descent on the image misfit alone recovers the pose.
    #[test]
    fn descent_recovers_the_pose() {
        let path = descend(STEPS);
        let (g, last) = path[STEPS];
        let (turn, shift) = pose_error(truth(), g);
        assert!(turn < 0.02 && shift < 0.02, "{turn} {shift}");
        assert!(last < 1e-3 * path[0].1, "{} -> {last}", path[0].1);
    }

    /// The dual-number gradient agrees with central differences.
    #[test]
    fn the_gradient_matches_finite_differences() {
        let world = world();
        let observed = image(truth(), &world);
        let g = start();
        let (_, grad) = misfit_and_gradient(g, &world, &observed);
        let h = 1e-6;
        for i in 0..6 {
            let (mut a, mut b) = (g, g);
            a.c[i] += h;
            b.c[i] -= h;
            let fd = (misfit(&image(a, &world), &observed) - misfit(&image(b, &world), &observed))
                / (2.0 * h);
            assert!(
                (fd - grad.c[i]).abs() < 1e-6 * (1.0 + fd.abs()),
                "{i}: {fd} {}",
                grad.c[i]
            );
        }
    }

    /// The images lie on the screen, and the truth fits exactly.
    #[test]
    fn images_lie_on_the_screen() {
        let world = world();
        let screen = gax::pga3d::Plane::from_normal([0.0, 0.0, 1.0], 1.0);
        for p in image(truth(), &world) {
            assert!((screen & p).s().abs() < 1e-12);
        }
        assert!(misfit(&image(truth(), &world), &image(truth(), &world)) < 1e-30);
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws(super::draw, 0.5);
    }
}
