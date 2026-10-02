//! numga's `quadrics/elliptic_physics`: rigid quadric bodies on a sphere, one engine for S² in
//! `Cl(3)` (gax's `vga3d`) and S³ in `Cl(4)` (declared with `gax::algebra!` in
//! `shared/quadrics_s3.rs`). Bodies are ellipsoids of the sphere placed by rotors; their inertia
//! comes from mass points, their momenta are stepped by a Lie midpoint rule, and contacts come
//! from blends of their quadratic forms (see `engine.rs`).
//!
//! The animation replays two simulations side by side: on the left an S² scene seen on the front
//! hemisphere, with its energy and momentum (and, tumbling, its body rates); on the right an S³
//! scene ray-traced from its eye by projection, as in the S³ ray tracer. `--s2 crowded|tumbling|
//! hyperbolic` and `--s3 crowd|gap|needle|tunnel` choose them (crowded and crowd by default).

mod engine;
#[path = "../../shared/quadrics_s3.rs"]
mod s3;
mod scenes;

use gax_numga_examples::{
    Align, Anim, Axes, Canvas, Rgb, backdrop, canvas, caption, palette, plot, run,
};
use scenes::{Scene3, s2};

/// A small xorshift generator: numga's streams (numpy's PCG64) cannot be reproduced, so the S³
/// populations differ from numga's, and their checks are robust to the stream.
mod rng {
    pub struct Rng(u64);

    impl Rng {
        pub fn new(seed: u64) -> Rng {
            Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
        }

        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }

        /// Uniform in `[0, 1)`.
        pub fn uniform(&mut self) -> f64 {
            (self.next() >> 11) as f64 / (1u64 << 53) as f64
        }

        /// A standard normal sample (Box–Muller).
        pub fn normal(&mut self) -> f64 {
            let u = 1.0 - self.uniform();
            let v = self.uniform();
            (-2.0 * u.ln()).sqrt() * (core::f64::consts::TAU * v).cos()
        }

        /// A uniform direction in `n` dimensions (empty for `n = 0`).
        pub fn unit_vector(&mut self, n: usize) -> Vec<f64> {
            let v: Vec<f64> = (0..n).map(|_| self.normal()).collect();
            let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            v.iter().map(|x| x / norm).collect()
        }

        /// A random permutation, in place.
        pub fn shuffle<T>(&mut self, items: &mut [T]) {
            for i in (1..items.len()).rev() {
                let j = (self.uniform() * (i + 1) as f64) as usize;
                items.swap(i, j.min(i));
            }
        }
    }
}

/// The time step of the S² scenes and of the S³ scenes.
const DT2: f64 = 0.015;
const SECONDS: f32 = 12.0;
const FRAMES: usize = 240;

/// The two simulations shown.
struct Show {
    s2_name: String,
    s2: s2::Trajectory,
    s3_name: String,
    s3: Scene3,
}

fn show(s2_name: &str, s3_name: &str, frames: usize) -> Show {
    let s2 = match s2_name {
        "tumbling" => scenes::tumbling(frames),
        "hyperbolic" => scenes::hyperbolic(frames),
        _ => scenes::crowded(frames),
    };
    let s3 = match s3_name {
        "gap" => scenes::gap(frames),
        "needle" => scenes::needle(frames),
        "tunnel" => scenes::tunnel(frames).scene,
        _ => scenes::crowd(frames),
    };
    if let Some(f) = scenes::eye_inside(&s3) {
        eprintln!("note: a body covers the eye from frame {f} of {s3_name}");
    }
    Show {
        s2_name: s2_name.to_uppercase(),
        s2,
        s3_name: s3_name.to_uppercase(),
        s3,
    }
}

fn display(c: [f64; 3]) -> Rgb {
    canvas::srgb(c[0] as f32, c[1] as f32, c[2] as f32)
}

/// The front hemisphere of S² seen along z in a disc: a pixel is a point of S², and it belongs
/// to the last body whose primal form is negative there. The spherical quadric is a cone through
/// the centre, so the front hemisphere shows both halves of a body.
fn hemisphere(
    c: &mut Canvas,
    centre: [f32; 2],
    radius: f32,
    surfaces: &[s2::Quadric],
    colors: &[Rgb],
) {
    let (disk, rim) = (
        canvas::srgb(0.067, 0.094, 0.153),
        canvas::srgb(0.2, 0.255, 0.333),
    );
    c.clip([
        centre[0] - radius - 1.0,
        centre[1] - radius - 1.0,
        centre[0] + radius + 1.0,
        centre[1] + radius + 1.0,
    ]);
    c.shade(2, |x, y| {
        let (u, v) = ((x - centre[0]) / radius, (centre[1] - y) / radius);
        let r2 = u * u + v * v;
        if r2 > 1.0 {
            return None;
        }
        let p = gax::vga3d::Bivector::new(
            f64::from(u),
            f64::from(v),
            f64::from((1.0 - r2).max(0.0).sqrt()),
        );
        let body = surfaces.iter().rposition(|s| (p & s.of(p)).s() < 0.0);
        Some(match body {
            Some(k) => colors[k],
            None if r2 > 0.985 => rim,
            None => disk,
        })
    });
    c.unclip();
}

/// The S² scene's invariants over time, normalized by their start, and its body rates when it
/// tumbles; a cursor at frame `f`.
fn plot_invariants(show: &Show, c: &mut Canvas, rect: [f32; 4], f: usize) {
    let frames = show.s2.energy.len();
    let time = |k: usize| (k as f64 * DT2) as f32;
    let tumbling = show.s2_name == "TUMBLING";
    let range = if tumbling {
        [-16.0, 16.0]
    } else {
        [0.95, 1.05]
    };
    let ax = Axes::new(rect, [0.0, time(frames - 1).max(1e-3)], range);
    let (energy_drift, momentum_drift) = scenes::conserved_s2(&show.s2);
    let title = format!(
        "{}: DRIFT {energy_drift:.0E}, {momentum_drift:.0E}",
        if tumbling {
            "RATES, ENERGY, MOMENTUM"
        } else {
            "ENERGY, MOMENTUM"
        }
    );
    ax.frame(c, &title, "", "");
    if tumbling {
        let colours = [palette::sky(), palette::red(), palette::yellow()];
        for (i, colour) in colours.into_iter().enumerate() {
            let rate: Vec<[f32; 2]> = (0..frames)
                .map(|k| [time(k), show.s2.rates[k][0].c[i] as f32])
                .collect();
            ax.polyline(c, &rate, 1.5, colour, 1.0);
        }
    }
    let (e0, m0) = (show.s2.energy[0], show.s2.momentum[0].norm());
    let energy: Vec<[f32; 2]> = (0..frames)
        .map(|k| [time(k), (show.s2.energy[k] / e0) as f32])
        .collect();
    let momentum: Vec<[f32; 2]> = (0..frames)
        .map(|k| [time(k), (show.s2.momentum[k].norm() / m0) as f32])
        .collect();
    ax.polyline(c, &energy, 1.8, palette::green(), 1.0);
    ax.polyline(c, &momentum, 1.4, palette::purple(), 1.0);
    ax.line(
        c,
        [time(f), range[0]],
        [time(f), range[1]],
        1.0,
        palette::ink(),
        0.6,
    );
}

fn draw(show: &Show, c: &mut Canvas, t: f32) {
    backdrop(c);
    let (w, h) = (c.width as f32, c.height as f32);
    let frames = show.s2.energy.len();
    let f = (((t / SECONDS).fract() * frames as f32) as usize).min(frames - 1);
    caption(
        c,
        "QUADRIC RIGID BODIES ON S2 AND S3",
        "ONE ENGINE: BLENDS OF FORMS FOR CONTACTS, LIE MIDPOINT STEPS",
    );
    let size = (h / 40.0).clamp(7.0, 12.0);
    let top = h * 0.13;

    // S²: the hemisphere, and the invariants under it.
    let colors: Vec<Rgb> = show.s2.colors.iter().map(|c| display(*c)).collect();
    let radius = (w * 0.2).min((h - top - 110.0) / 2.0).max(4.0);
    let centre = [w * 0.22, top + 10.0 + radius];
    hemisphere(c, centre, radius, &show.s2.surfaces[f], &colors);
    c.text(
        &format!("S2: {}", show.s2_name),
        centre[0],
        centre[1] + radius + size * 1.4,
        size,
        palette::ink(),
        Align::Center,
    );
    let rect = plot::inset(
        [0.0, centre[1] + radius + size * 1.6, w * 0.44, h],
        34.0,
        16.0,
        8.0,
        20.0,
    );
    if rect[3] - rect[1] > 10.0 {
        plot_invariants(show, c, rect, f);
    }

    // S³: traced from the eye, a 4:3 picture.
    let width = w - 8.0 - w * 0.46;
    let panel = [
        w * 0.46,
        top,
        w - 8.0,
        (top + width * 0.75).min(h - 2.0 * size),
    ];
    let trajectory = &show.s3.trajectory;
    let view = s3::View {
        eye: show.s3.eye,
        surfaces: trajectory.surfaces[f.min(trajectory.surfaces.len() - 1)].clone(),
        colors: trajectory.colors.clone(),
        light: show.s3.light,
        fov: 120f64.to_radians(),
    };
    let tracer = view.tracer();
    let background = canvas::srgb(0.02, 0.02, 0.02);
    c.clip(panel);
    c.shade(2, |x, y| {
        let px = s3::chart(
            view.fov,
            f64::from(x - panel[0]),
            f64::from(y - panel[1]),
            f64::from(panel[2] - panel[0]),
            f64::from(panel[3] - panel[1]),
        );
        Some(tracer.color(px).map_or(background, display))
    });
    c.unclip();
    c.text(
        &format!(
            "S3: {}, {} ELLIPSOIDS, {} IMPULSES IN ALL",
            show.s3_name,
            view.surfaces.len(),
            trajectory.impulses
        ),
        panel[0] + 4.0,
        panel[3] + size * 1.5,
        size,
        palette::ink(),
        Align::Left,
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let arg = |name: &str, default: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .cloned()
            .unwrap_or_else(|| default.to_string())
    };
    let show = show(&arg("--s2", "crowded"), &arg("--s3", "crowd"), FRAMES);
    run(
        Anim::new("elliptic physics", SECONDS)
            .size(800, 450)
            .scale(2),
        move |c, t| draw(&show, c, t),
    );
}

#[cfg(test)]
mod tests {
    use super::rng::Rng;
    use super::scenes::{self, s2, s3};
    use gax::vga3d::{Bivector, Vector};

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    /// Great circles tangent to an ellipse satisfy `π ∨ Q(π) = 0`.
    #[test]
    fn ellipse_tangency() {
        let (tx, ty) = (30f64.to_radians().tan(), 15f64.to_radians().tan());
        let q = s2::ellipsoid([tx, ty]);
        for phi in [0.0f64, 30.0, 75.0, 120.0, 200.0, 310.0] {
            let phi = phi.to_radians();
            let tangent = Vector::new(phi.cos() / tx, phi.sin() / ty, 1.0);
            assert!((tangent & q.of(tangent)).s().abs() < 1e-12);
        }
    }

    /// The margin is positive when apart, near zero when touching, negative when overlapping.
    #[test]
    fn overlap_margin_sign() {
        let deg = f64::to_radians;
        let c1 = s2::ellipsoid([deg(20.0).tan(); 2]).inverse();
        let c2 = s2::ellipsoid([deg(25.0).tan(); 2]).inverse();
        // Centres apart, touching at 45° (the sum of the radii), and overlapping. numga's `xz`
        // turn is a `zx` turn the other way.
        let margins: Vec<f64> = [55.0, 45.0, 35.0]
            .iter()
            .map(|a| {
                let turn = Bivector::new(0.0, deg(*a) / 2.0, 0.0).exp();
                s2::overlap(c1, turn >> c2.of(turn << Bivector::slot()), 12).0
            })
            .collect();
        assert!(margins[0] > 0.005);
        assert!(margins[1].abs() < 2e-3);
        assert!(margins[2] < -0.005);
    }

    /// A 35° by 10° ellipse of unit mass has three distinct moments, `Iz < Ix < Iy`, with numga's
    /// values.
    #[test]
    fn ellipse_inertia_has_an_intermediate_axis() {
        let (points, masses) =
            scenes::ellipse_mesh([35f64.to_radians(), 10f64.to_radians()], 1.0, 1536, 10);
        let inertia = s2::pointcloud_inertia(&points, &masses);
        let moment = |axis: Bivector<(), f64>| (axis & inertia.of(axis)).s().abs();
        let (ix, iy, iz) = (
            moment(Bivector::new(1.0, 0.0, 0.0)),
            moment(Bivector::new(0.0, 1.0, 0.0)),
            moment(Bivector::new(0.0, 0.0, 1.0)),
        );
        assert!(iz < ix && ix < iy);
        for (got, want) in [iz, ix, iy].iter().zip([0.094, 0.913, 0.992]) {
            assert!(close(*got, want, 0.03), "{got} {want}");
        }
    }

    /// The Lie midpoint step keeps the body-frame momentum's norm exactly and the energy
    /// closely.
    #[test]
    fn free_flight_conserves_energy_and_momentum() {
        let (points, masses) =
            scenes::ellipse_mesh([35f64.to_radians(), 10f64.to_radians()], 1.0, 384, 10);
        let inertia = s2::pointcloud_inertia(&points, &masses);
        let i_inv = inertia.inverse();
        let (mut motor, mut momentum) = (s2::identity(), inertia.of(Bivector::new(2.5, 0.0, 0.05)));
        let energy = (i_inv.of(momentum) & momentum).s();
        let size = momentum.norm();
        for _ in 0..200 {
            (motor, momentum) = s2::step_motor(motor, momentum, i_inv, 0.01);
        }
        // numga's `assert_allclose(..., atol=1e-14)` keeps its default relative tolerance of
        // 1e-7 (its own drift here is 1.4e-14).
        assert!(close(momentum.norm(), size, 1e-14 + 1e-7 * size));
        assert!(close(
            (i_inv.of(momentum) & momentum).s(),
            energy,
            0.01 * energy
        ));
    }

    /// Two overlapping, approaching ellipses: one elastic impulse keeps the total energy and the
    /// world momentum.
    #[test]
    fn collision_conserves_energy_and_momentum() {
        let specs: Vec<scenes::Ellipse> = [(0.0, -2.5), (28.0, 2.0)]
            .iter()
            .map(|(angle, spin)| scenes::Ellipse {
                half_angles: [25.0, 15.0],
                mass: 1.0,
                placement: Bivector::new(-f64::to_radians(*angle) / 2.0, 0.0, 0.0).exp(),
                rate: [*spin, 0.0, 0.0],
                color: 0x38bdf8,
            })
            .collect();
        let mut bodies = scenes::ellipses(&specs, 96);
        for b in &mut bodies {
            b.motor = scenes::camera().inverse() * b.motor;
        }
        let before = bodies.clone();
        let applied = s2::collide(&mut bodies, &[(0, 1)], 0.001);
        assert_eq!(applied, 1);
        let (e0, e1) = (s2::kinetic_energy(&before), s2::kinetic_energy(&bodies));
        assert!(close(e1, e0, 1e-12 * e0));
        let (m0, m1) = (s2::total_momentum(&before), s2::total_momentum(&bodies));
        assert!((m1 - m0).norm() / m0.norm() < 1e-7);
    }

    /// The sampler's points fill the inside of any quadric, an ellipsoid or a torus, and carry
    /// its mass.
    #[test]
    fn filled_points_lie_inside_with_the_given_mass() {
        let mut rng = Rng::new(0);
        for q in [
            s3::ellipsoid([0.3, 0.2, 0.1]),
            s3::quadric([-1.0, -1.0, 2.0, 3.0]),
        ] {
            let (points, masses) = s3::filled(q, 2.0, 200, &mut rng);
            let c = q.inverse();
            for p in &points {
                assert!((*p & c.of(*p)).s() < 0.0);
            }
            assert!(close(masses.iter().sum::<f64>(), 2.0, 1e-12));
        }
    }

    /// The S² scenes conserve their invariants (numga's checks), apply numga's numbers of
    /// impulses, and the oval tumbles: its rate about the intermediate axis changes sign at
    /// least twice (the Dzhanibekov effect).
    #[test]
    fn s2_scenes() {
        for (name, trajectory, drift, impulses) in [
            ("crowded", scenes::crowded(41), 1e-3, 5),
            ("hyperbolic", scenes::hyperbolic(41), 5e-4, 15),
            ("tumbling", scenes::tumbling(100), 5e-4, 0),
        ] {
            let (energy, momentum) = scenes::conserved_s2(&trajectory);
            assert!(energy < drift, "{name}: energy drift {energy}");
            assert!(momentum < 1e-11, "{name}: momentum drift {momentum}");
            assert_eq!(trajectory.impulses, impulses, "{name}");
            if name == "tumbling" {
                let spin: Vec<f64> = trajectory.rates.iter().map(|r| r[0].c[0]).collect();
                let flips = spin
                    .windows(2)
                    .filter(|w| w[0].signum() != w[1].signum())
                    .count();
                assert!(flips >= 2);
            }
        }
    }

    /// Each S³ scene collides and keeps its energy (elastic impulses with a symplectic step);
    /// in the tunnel, nobody ends in the wall, the lit wall point is on the wall, and the light
    /// is halfway between it and the eye.
    #[test]
    fn s3_scenes() {
        let tunnel = scenes::tunnel(12);
        for (name, scene) in [
            ("crowd", scenes::crowd(12)),
            ("gap", scenes::gap(12)),
            ("needle", scenes::needle(12)),
            ("tunnel", tunnel.scene),
        ] {
            let t = &scene.trajectory;
            assert!(t.impulses > 0, "{name}: no collisions");
            let drift = scenes::drift(&t.energy);
            assert!(drift < 2e-2, "{name}: energy drift {drift}");
            if name == "tunnel" {
                let last = t.surfaces.last().expect("frames");
                for body in &last[1..] {
                    assert!(s3::overlap(last[0], *body, 12).0 > -1e-2);
                }
                let wall = t.surfaces[0][0];
                assert!((tunnel.wall_hit & wall.of(tunnel.wall_hit)).s().abs() < 1e-12);
                let light = scene.light;
                assert!(close(
                    (tunnel.camera | light).s(),
                    (light | tunnel.wall_hit).s(),
                    1e-12
                ));
            }
        }
    }

    /// No body of the crowd shown by default passes over the eye in the animation's frames.
    #[test]
    fn the_crowd_keeps_off_the_eye() {
        assert_eq!(scenes::eye_inside(&scenes::crowd(super::FRAMES)), None);
    }

    #[test]
    fn a_frame_draws() {
        let show = super::show("crowded", "crowd", 3);
        let c = gax_numga_examples::app::frame(
            &gax_numga_examples::Anim::new("t", 1.0).size(320, 180),
            0.5,
            &mut |c, t| super::draw(&show, c, t),
        );
        assert!(c.mean()[0] > 0.0);
    }
}
