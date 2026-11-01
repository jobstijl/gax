//! numga's `mechanics/freeform`: FreeForm elasticity, a mesh-free elastic Stanford bunny pulled by
//! one ear, held at its base, then released (VGA3D).
//!
//! Points, displacements and the controls of the motion are directions, antivectors (in space
//! intrinsic bivectors); the gradient of a scalar field and the force on a point are covectors,
//! vectors, read by pairing with a direction. A few weight fields over the solid, the lowest
//! skinning modes of reproducing kernels about farthest-point centres, blend whole affine
//! motions, each set by the displacements of four control corners; the local deformation is
//! the identity plus every control displacement paired with its feature's gradient, a map on
//! directions. Its Neo-Hookean stress maps covectors to covectors: the metric part, the
//! deformation's adjoint carried to covectors by its adjugate, and the cofactor, its
//! outermorphism onto the planes two directions span, over the reference volume. The adjugate
//! of the sampling map collects every local force onto the controls. Each implicit time step
//! minimizes the potential plus the inertia of departing from free flight, by limited-memory
//! BFGS whose first guess of the inverse curvature is the control-space mass's inverse.
//!
//! The animation shows the bunny as a short pull on its ear (the arrow) stretches it and it
//! springs back, held at its base, beside the ear's travel along the pull and the base's.

mod freeform;
mod geometry;
mod numerics;
#[cfg(test)]
mod tests;

use freeform::*;
use gax::field::Field;
use gax_numga_examples::shared::simplicial;
use gax_numga_examples::signal::{frame, haversine};
use gax_numga_examples::{
    Anim, Axes, Camera, Canvas, Lens, Marker, Point2, Point3, Rect, backdrop, caption, label_above,
    palette, run_with, scene3,
};
use geometry::Surface;
use numerics::{minimize_state, work};

/// The bunny's height, its volume samples and kernel centres, the kernels' radius, and the
/// weight modes.
const HEIGHT: f64 = 2.0;
const QUADRATURE: usize = 1600;
const CENTERS: usize = 96;
const RADIUS: f64 = 0.3;
const MODES: usize = 9;
/// The material: its Lamé parameters and density.
const LAME_LAMBDA: f64 = 20.0;
const LAME_MU: f64 = 10.0;
const DENSITY: f64 = 1.0;
/// The compliant support at the base: its depth and stiffness.
const CLAMP_DEPTH: f64 = 0.12;
const CLAMP_STIFFNESS: f64 = 500.0;
/// The pull near one ear's tip: where, how wide, the total force, and how long it lasts.
fn pull_center() -> Antivector {
    point(-0.25, 0.34, 1.92)
}
const PULL_WIDTH: f64 = 0.23;
fn pull_force() -> Vector {
    Vector::new(1.0, 0.7, 0.0) * 0.6
}
const PULL_DURATION: f64 = 0.4;
/// The time step, the steps and the steps per frame drawn, and the minimizer's iterations and
/// tolerance per step.
const TIME_STEP: f64 = 0.02;
const STEPS: usize = 200;
const STEPS_PER_FRAME: usize = 2;
const ITERATIONS: usize = 30;
const TOLERANCE: f64 = 1e-7;

/// The pulled bunny: its surface, and its vertices and the pull's strength at every step.
pub struct Pulled {
    pub surface: Surface,
    pub states: Vec<Field<Antivector>>,
    pub loading: Vec<f64>,
}

/// The pull's strength at each step: rising and falling as the square of a sine over its
/// duration, then nothing.
fn loading() -> Vec<f64> {
    (0..STEPS)
        .map(|k| {
            let share = (k as f64 * TIME_STEP / PULL_DURATION).min(1.0);
            haversine(core::f64::consts::TAU * share)
        })
        .collect()
}

/// The surface's vertices over the steps, from rest.
fn bunny() -> Pulled {
    let (surface, samples, volumes, centers) = geometry::load(HEIGHT, QUADRATURE, CENTERS);
    let (evaluation, gradient) = kernels(&samples, &centers, RADIUS);
    let lambdas = vec![LAME_LAMBDA; samples.len()];
    let mus = vec![LAME_MU; samples.len()];
    let (_, modes) = modes(&evaluation, &gradient, &volumes, &lambdas, &mus, MODES);
    // Four control points around the bunny, the corners of a tetrahedron.
    let corners = [
        point(-1.0, -1.0, 0.0),
        point(1.0, -1.0, 2.0),
        point(-1.0, 1.0, 2.0),
        point(1.0, 1.0, 0.0),
    ];
    let (sampling, gradients) = deformation_basis(&samples, &centers, RADIUS, &modes, &corners);
    let (moving, _) = deformation_basis(&surface.vertices, &centers, RADIUS, &modes, &corners);
    let masses: Vec<f64> = volumes.iter().map(|v| v * DENSITY).collect();
    let mass = mass(&sampling, &masses);
    let body = Body::new(sampling, &gradients, volumes.clone(), LAME_LAMBDA, LAME_MU);

    // A compliant support holds the base; the pull is localized near one ear tip, normalized
    // over the solid, so that the profile distributes the stated total force.
    let up = Vector::new(0.0, 0.0, 1.0);
    let clamp: Vec<f64> = samples
        .iter()
        .map(|s| {
            let height = (up & *s).s();
            (-height * height / (CLAMP_DEPTH * CLAMP_DEPTH)).exp() * CLAMP_STIFFNESS
        })
        .collect();
    let profile: Vec<f64> = samples
        .iter()
        .map(|s| (-dot(*s - pull_center(), *s - pull_center()) / (PULL_WIDTH * PULL_WIDTH)).exp())
        .collect();
    let total: f64 = profile.iter().zip(&volumes).map(|(p, v)| p * v).sum();
    let load: Vec<Vector> = profile.iter().map(|p| pull_force() * (p / total)).collect();

    let rest = geometry::field(&surface.vertices);
    let loading = loading();
    let inertia = &mass * (1.0 / (TIME_STEP * TIME_STEP));
    let controls = 4 * MODES;
    let mut state = Field::<Antivector>::zeros(controls);
    let mut velocity = Field::<Antivector>::zeros(controls);
    let mut states = vec![&rest + &(&moving * &state)];
    for &strength in &loading {
        let predicted = &state + &velocity.scaled(TIME_STEP);
        let pulled: Vec<Vector> = load.iter().map(|l| *l * strength).collect();
        let objective = |trial: &Field<Antivector>| {
            let (energy, gradient) = body.potential(trial, &clamp, &pulled);
            // Inertia resists departing from the free-flight prediction.
            let change = trial - &predicted;
            let inertial = inertia.of(&change);
            (
                energy + work(&inertial, &change) / 2.0,
                &gradient + &inertial,
            )
        };
        let following = minimize_state(&state, objective, &inertia, ITERATIONS, TOLERANCE);
        velocity = (&following - &state).scaled(1.0 / TIME_STEP);
        state = following;
        states.push(&rest + &(&moving * &state));
    }
    Pulled {
        surface,
        states,
        loading,
    }
}

/// A point to draw.
fn at(p: Antivector) -> Point3 {
    let [x, y, z] = coordinates(p).map(|c| c as f32);
    Point3::xyz(x, y, z)
}

/// The view: numga's elevation and azimuth.
const ELEVATION: f32 = 24.0 * core::f32::consts::PI / 180.0;
const AZIMUTH: f32 = 54.0 * core::f32::consts::PI / 180.0;
/// The loop: numga's 101 frames of 40 ms, and a breath at the end.
const PLAYING: f32 = 4.04;
const SECONDS: f32 = 5.0;

/// How far the ear has gone along the pull, and the farthest any base vertex has moved, at
/// each step.
fn travels(p: &Pulled) -> (Vec<f64>, Vec<f64>) {
    let rest = &p.states[0];
    let up = Vector::new(0.0, 0.0, 1.0);
    let base: Vec<usize> = (0..rest.len())
        .filter(|&v| (up & rest.site(v)).s() < 0.05)
        .collect();
    let along = pull_force();
    p.states
        .iter()
        .map(|s| {
            let pulled = rest
                .iter()
                .zip(s.iter())
                .map(|(r, v)| (along & (v - r)).s())
                .fold(f64::MIN, f64::max);
            #[allow(clippy::disallowed_methods)] // a distance from its square, for a plot
            let moved = base
                .iter()
                .map(|&v| dot(s.site(v) - rest.site(v), s.site(v) - rest.site(v)).sqrt())
                .fold(0.0, f64::max);
            (pulled, moved)
        })
        .unzip()
}

/// Everything drawn, computed once.
struct Scenes {
    pulled: Pulled,
    ear: Vec<f64>,
    base: Vec<f64>,
}

fn scenes() -> Scenes {
    let pulled = bunny();
    let (ear, base) = travels(&pulled);
    Scenes { pulled, ear, base }
}

/// The bunny at a step, lit by its faces, with the pull's arrow while it lasts.
fn draw_bunny(c: &mut Canvas, rect: Rect, p: &Pulled, step: usize) {
    let rest = &p.states[0];
    let centre = rest
        .iter()
        .fold(Point3::xyz(0.0, 0.0, 0.0) * 0.0, |s, v| s + at(v));
    let centre = centre.unitized();
    // numga's framing: the first shape with room around it to flex into.
    let half = 0.75 * HEIGHT as f32 * (1.0 + 0.12) * (rect.height() / rect.width()).max(0.8);
    let cam = Camera::orbit(
        rect,
        centre,
        10.0,
        AZIMUTH - core::f32::consts::FRAC_PI_2,
        ELEVATION,
        Lens::Parallel(half),
    );
    let vertices = &p.states[step];
    scene3::panel3(c, rect, cam, |s| {
        let skin = palette::SKY.faded(0.35);
        for f in &p.surface.faces {
            let [a, b, d] = f.map(|v| at(vertices.site(v)));
            let light = s.lit(a, b, d, skin);
            s.tri(a, b, d, light, 1.0);
        }
        let strength = p.loading.get(step).copied().unwrap_or(0.0);
        if strength > 0.02 {
            let tip = at(pull_center() + pull_force().hodge() * (0.8 * strength));
            let from = at(pull_center());
            s.arrow(from, tip - from, 2.0, 8.0, palette::ORANGE);
        }
    });
}

/// The ear's travel along the pull and the base's, against time, with the pull's strength.
fn draw_travel(c: &mut Canvas, rect: Rect, s: &Scenes, step: usize) {
    let top = s.ear.iter().copied().fold(0.0, f64::max) as f32;
    let duration = STEPS as f32 * TIME_STEP as f32;
    let ax = Axes::new(rect, [0.0, duration], [-0.1 * top, 1.55 * top]);
    let time = |k: usize| k as f32 * TIME_STEP as f32;
    let strength: Vec<Point2> = s
        .pulled
        .loading
        .iter()
        .enumerate()
        .map(|(k, l)| Point2::xy(time(k), *l as f32 * top))
        .collect();
    ax.dashed(c, &strength, 1.0, 4.0, palette::ORANGE.faded(0.6));
    for (values, light) in [(&s.ear, palette::SKY), (&s.base, palette::GREEN)] {
        let curve: Vec<Point2> = values
            .iter()
            .enumerate()
            .map(|(k, v)| Point2::xy(time(k), *v as f32))
            .collect();
        ax.polyline(c, &curve, 1.0, light.faded(0.35));
        ax.polyline(c, &curve[..=step], 1.6, light);
        ax.scatter(c, &[curve[step]], Marker::Dot, 6.0, light.whitened(0.3));
    }
    ax.legend(
        c,
        &[
            ("EAR, ALONG THE PULL", palette::SKY),
            ("BASE, FARTHEST", palette::GREEN),
            ("PULL", palette::ORANGE),
        ],
    );
    ax.frame(c, "HOW FAR THEY MOVE", "TIME", "");
}

fn draw(c: &mut Canvas, s: &Scenes, t: f32) {
    backdrop(c);
    let time = t.rem_euclid(SECONDS);
    let frames = s.pulled.states.len().div_ceil(STEPS_PER_FRAME);
    let step = frame((time / PLAYING).min(1.0), frames) * STEPS_PER_FRAME;
    let screen = c.rect();
    let (w, h) = (screen.width(), screen.height());
    let size = (h / 50.0).clamp(6.0, 11.0);
    let left = Rect::new(0.0, h * 0.14, w * 0.58, h).inset(8.0, 16.0, 4.0, 8.0);
    draw_bunny(c, left, &s.pulled, step);
    label_above(c, left, "PULLED BY AN EAR, HELD AT ITS BASE", size, 2.0);
    let right = Rect::new(w * 0.58, h * 0.2, w, h * 0.85).inset(40.0, 20.0, 24.0, 30.0);
    draw_travel(c, right, s, step);
    caption(
        c,
        "FREEFORM ELASTICITY: A MESH-FREE BUNNY",
        "F = 1 + CONTROLS x (GRADIENTS & D); STRESS = MU F*ADJ + BULK (J - T) COFACTOR; SAMPLING'S ADJUGATE COLLECTS FORCES (VGA3D)",
    );
}

fn main() {
    run_with(Anim::new("freeform", SECONDS).size(960, 540), scenes, draw);
}
