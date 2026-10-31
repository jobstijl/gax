//! numga's `mechanics/spinning_top`: a spinning top of three ellipsoids on flat ground, in
//! PGA3D, ray traced.
//!
//! Every shape is a quadric, a map from points to planes (`Plane<(Point,)>`): a point's polar
//! plane. The top's parts are solid ellipsoids in its own frame; the ground is the solid below
//! a paraboloid of a given curvature, flat here (numga's ground can be a bowl or a dome). A
//! part's lowest point against the ground is the pole of its tangent plane facing the ground;
//! how far that point lies below the ground is its depth, zero for a part in the air. Contact is
//! position based: after a free step the parts are corrected one after another, so that each
//! sees the corrections before it (Gauss–Seidel): pressed out along the ground's normal by its
//! depth, then held against sliding if the full correction fits within the friction cone, and
//! left to slip if not, each a single constraint along one line. The rate is read back from the
//! motor's step, then slowed by each part's dynamic friction in turn.
//!
//! The contact point is found in one pass: the ground's normal is taken at the part's centre,
//! and the part's lowest point against that normal is the contact. That is accurate while the
//! part is more sharply curved than the ground, a small body on a large one, as here.
//!
//! The animation ray traces the base top (each pixel's ray meets the nearest quadric, its
//! entry the root of the quadric's form along the ray where it falls through zero) as it spins,
//! precesses and wanders, and plots the lean and spin of numga's four variations: the base, a
//! sharper tip, a heavier disc and a more slippery ground.

#[path = "../shared/mechanics_contact.rs"]
mod contact;
#[path = "../shared/quadrics_tracing.rs"]
mod tracing;

use gax::pga3d::{Direction, Line, Motor, Plane, Point};
use gax_colour::light;
use gax_numga_examples::{
    Anim, Axes, Camera, Canvas, Lens, Light, Point2, Rect, backdrop, caption, palette, par_map,
    run_with,
};

use gax_numga_examples::shared::mechanics_lie as lie;

mod top {
    use super::*;
    pub use crate::contact::*;
    pub use crate::lie::rigid::{G, Inertia, InertiaInv, M, P, R};

    /// The coordinate planes `x = 0`, `y = 0`, `z = 0`.
    pub fn axes() -> [Plane; 3] {
        [
            Plane::new(1.0, 0.0, 0.0, 0.0),
            Plane::new(0.0, 1.0, 0.0, 0.0),
            Plane::new(0.0, 0.0, 1.0, 0.0),
        ]
    }

    /// The directions of the x, y and z axes.
    pub fn directions() -> [P; 3] {
        [
            Point::direction(1.0, 0.0, 0.0),
            Point::direction(0.0, 1.0, 0.0),
            Point::direction(0.0, 0.0, 1.0),
        ]
    }

    // --- shapes and mass ----------------------------------------------------------------------

    /// A solid ellipsoid about a centre with the given semi-axes: the dyads of the planes
    /// through the centre, less the dyad of the plane at infinity. Its form `Q(p) & p` is
    /// negative inside.
    pub fn ellipsoid(centre: P, semi: [f64; 3]) -> Quadric {
        ellipsoid_in(centre, axes(), semi)
    }

    /// Six mass points per solid ellipsoid with its mass, centroid and second moments, at
    /// `sqrt(3 / 5)` of each semi-axis on either side.
    #[allow(clippy::disallowed_methods)] // a number: the points' moment r²/3 matches the solid's 1/5
    pub fn sigma_points(centres: &[P], semi: &[[f64; 3]], mass: &[f64]) -> (Vec<P>, Vec<f64>) {
        let reach = (3.0f64 / 5.0).sqrt();
        let (mut points, mut masses) = (Vec::new(), Vec::new());
        for ((c, s), m) in centres.iter().zip(semi).zip(mass) {
            for sign in [1.0, -1.0] {
                for (axis, half) in directions().into_iter().zip(s) {
                    points.push(*c + axis * (sign * half * reach));
                    masses.push(m / 6.0);
                }
            }
        }
        (points, masses)
    }
}

mod scenarios {
    use super::top::*;
    use super::*;
    use crate::lie::Lie;

    /// A top and its ground: the spin it starts with, its tip's radius, its disc's height above
    /// the tip and mass, the ground's curvature, the dynamic friction, and the air's drag rate.
    #[derive(Clone, Copy, Debug)]
    pub struct Setup {
        pub spin_rate: f64,
        pub tip_radius: f64,
        pub disc_height: f64,
        pub disc_mass: f64,
        pub ground_curvature: f64,
        pub dynamic_friction: f64,
        pub drag: f64,
    }

    /// The base top, spun on flat ground.
    pub const BASE: Setup = Setup {
        spin_rate: 18.0,
        tip_radius: 0.05,
        disc_height: 0.10,
        disc_mass: 1.0,
        ground_curvature: 0.0,
        dynamic_friction: 0.3,
        drag: 0.13,
    };

    /// numga's variations of the base top.
    pub const VARIATIONS: [(&str, Setup); 4] = [
        ("BASE", BASE),
        (
            "SHARP TIP",
            Setup {
                tip_radius: 0.02,
                ..BASE
            },
        ),
        (
            "HEAVY DISC",
            Setup {
                disc_mass: 2.0,
                ..BASE
            },
        ),
        (
            "SLIPPERY",
            Setup {
                dynamic_friction: 0.1,
                ..BASE
            },
        ),
    ];
    pub const GRAVITY: f64 = 9.81;
    pub const STATIC_FRICTION: f64 = 0.4;
    /// The top's disc and stem: their semi-axes, how far the stem's centre sits above the
    /// disc's, its mass; the tip's height and mass.
    pub const DISC_SEMI: [f64; 3] = [0.25, 0.25, 0.04];
    pub const STEM_SEMI: [f64; 3] = [0.02, 0.02, 0.15];
    pub const STEM_ABOVE_DISC: f64 = 0.16;
    pub const STEM_MASS: f64 = 0.05;
    pub const TIP_HEIGHT: f64 = 0.14;
    pub const TIP_MASS: f64 = 0.15;
    /// The tilt it is spun at, and how far the tip starts clear of the ground.
    pub const TILT: f64 = 0.15;
    pub const CLEARANCE: f64 = 0.002;
    /// The run: its length in seconds, the time step, and the steps per frame.
    pub const RUNTIME: f64 = 5.0;
    pub const STEP: f64 = 4e-3;
    pub const N_STEPS_PER_FRAME: usize = 12;
    /// numga's view: elevation and azimuth in degrees, the centre and the half width.
    pub const VIEW: (f64, f64) = (30.0, -60.0);
    pub const CENTRE: P = Point::new(0.0, 0.0, 0.15, 1.0);
    pub const EXTENT: f64 = 0.55;

    /// The top's parts, inertia, mass and the height of its centre of mass above the tip.
    pub struct Top {
        pub parts: Vec<Quadric>,
        pub inertia: Inertia,
        pub inertia_inv: InertiaInv,
        pub mass: f64,
        pub height: f64,
    }

    /// A disc, a stem and a tip, each a solid ellipsoid, in the frame of their centre of mass.
    pub fn top(tip_radius: f64, disc_height: f64, disc_mass: f64) -> Top {
        let centres = [disc_height, disc_height + STEM_ABOVE_DISC, TIP_HEIGHT]
            .map(|z| Point::xyz(0.0, 0.0, z));
        let semi = [DISC_SEMI, STEM_SEMI, [tip_radius, tip_radius, TIP_HEIGHT]];
        let mass = [disc_mass, STEM_MASS, TIP_MASS];
        let total: f64 = mass.iter().sum();
        // The centre of mass, and the parts moved to put it at the origin.
        let com = centres.iter().zip(mass).map(|(c, m)| *c * m).sum::<P>() / total;
        let centred = Motor::between(com, Point::origin());
        let centres = centres.map(|c| centred >> c);
        let parts = centres
            .iter()
            .zip(semi)
            .map(|(c, s)| ellipsoid(*c, s))
            .collect();
        let (points, masses) = sigma_points(&centres, &semi, &mass);
        let (inertia, inertia_inv) = G::inertia_of(&points, &masses);
        Top {
            parts,
            inertia,
            inertia_inv,
            mass: total,
            height: com.to_euclidean()[2],
        }
    }

    /// One frame: the pose, the lean of the top's axis from vertical, and the spin about the
    /// vertical (numga's `-(xy | rate)`).
    #[derive(Clone, Copy, Debug)]
    pub struct Sample {
        pub motor: M,
        pub lean: f64,
        pub spin: f64,
    }

    /// The angle of the top's axis from vertical.
    pub fn lean(motor: M) -> f64 {
        let up = Point::direction(0.0, 0.0, 1.0);
        let angle = up.angle_between(motor >> up);
        angle.min(core::f64::consts::PI - angle)
    }

    /// The pose and the rate after every frame of a few steps, simulated from a slight tilt.
    pub fn spin(
        setup: Setup,
        seconds: f64,
        dt: f64,
        per_frame: usize,
    ) -> (Top, Quadric, Vec<Sample>) {
        let body = top(setup.tip_radius, setup.disc_height, setup.disc_mass);
        let ground = bowl(setup.ground_curvature);
        let origin = Point::origin();
        // Tilt, then lift the centre of mass so that the tip, `height` below it, clears the
        // ground.
        let tilt = Motor::rotation_about(Point::direction(0.0, 1.0, 0.0), TILT);
        let [_, _, tip] = (tilt >> Point::xyz(0.0, 0.0, -body.height)).to_euclidean();
        let mut motor = Motor::translation(Point::direction(0.0, 0.0, CLEARANCE - tip)) * tilt;
        let mut rate = Line::new(0.0, 0.0, setup.spin_rate, 0.0, 0.0, 0.0);
        // The weight, a force line down through the centre of mass, and the air's drag,
        // opposing the momentum; in the body frame.
        let weight = Point::direction(0.0, 0.0, -GRAVITY * body.mass);
        let forces = |m: M, r: R| (m << ((m >> origin) & weight)) - body.inertia.of(r) * setup.drag;
        let frames = (seconds / dt).round() as usize / per_frame;
        let mut samples = Vec::with_capacity(frames);
        for _ in 0..frames {
            for _ in 0..per_frame {
                let before = motor;
                // Predict with a free step, then correct against the ground. The free step's
                // rate is not needed: the contacts read it back from the motors.
                let (m, _) =
                    G::explicit_rk4(motor, rate, body.inertia, body.inertia_inv, dt, &forces);
                (motor, rate) = project_contacts(
                    before,
                    m.renormalize_fast(),
                    body.inertia_inv,
                    &body.parts,
                    ground,
                    [STATIC_FRICTION, setup.dynamic_friction],
                    dt,
                );
            }
            samples.push(Sample {
                motor,
                lean: lean(motor),
                spin: rate.e12(),
            });
        }
        (body, ground, samples)
    }

    /// Where a ray from `origin` along the unit direction `heading` enters the solid quadric:
    /// its distance and the point. Bound to the ray in both slots, the form is a quadratic in
    /// the distance; the ray enters where it falls through zero, its slope there twice the
    /// negative square root of the discriminant: the first root where the half slope at the
    /// start `b` is positive, the second where it is negative (numga's `entry`; a half-space,
    /// the flat ground, has only the second).
    pub fn hit(surface: Quadric, origin: P, heading: P) -> Option<(f64, P)> {
        let along = crate::tracing::along_ray(surface, origin, heading);
        let roots = crate::tracing::crossings(along)?;
        let distance = if along[1] >= 0.0 { roots[0] } else { roots[1] };
        (distance.is_finite() && distance > 0.0 && distance < crate::tracing::FAR)
            .then(|| (distance, origin + heading * distance))
    }

    /// The normal of the quadric at a point on it: the direction of its polar plane.
    pub fn normal_at(surface: Quadric, p: P) -> Direction {
        surface.of(p).dual().cast::<Direction>()
    }
}

use scenarios::*;
use top::*;

/// The time a frame covers.
const FRAME: f64 = STEP * N_STEPS_PER_FRAME as f64;

/// The four variations, simulated once: the base's parts and ground, and every top's samples.
struct Runs {
    parts: Vec<Quadric>,
    ground: Quadric,
    samples: Vec<Vec<Sample>>,
}

fn runs() -> Runs {
    let all = par_map(&VARIATIONS, |(_, setup)| {
        spin(*setup, RUNTIME, STEP, N_STEPS_PER_FRAME)
    });
    let (base, ground, _) = &all[0];
    Runs {
        parts: base.parts.clone(),
        ground: *ground,
        samples: all.iter().map(|(_, _, s)| s.clone()).collect(),
    }
}

/// The colours of the disc, the stem and the tip (sRGB), and their sector contrasts.
const PART_COLOURS: [[f32; 3]; 3] = [[0.85, 0.30, 0.25], [0.35, 0.35, 0.40], [0.80, 0.70, 0.35]];
const SECTOR_CONTRAST: [f32; 3] = [0.25, 0.0, 0.0];
/// The intensity of the lit surfaces, facing the lamp squarely.
const SURFACE: f32 = 1.0;

/// The ray-traced top placed by the motor, in the square `view` of the canvas.
fn render(c: &mut Canvas, motor: M, parts: &[Quadric], ground: Quadric, view: Rect) {
    let (elevation, azimuth) = (VIEW.0.to_radians(), VIEW.1.to_radians());
    let cam = Camera::orbit(
        view,
        CENTRE,
        (20.0 * EXTENT) as f32,
        azimuth as f32,
        elevation as f32,
        Lens::Parallel(EXTENT as f32),
    );
    // The lamp, above the viewer's left shoulder: back toward the viewer (the camera looks
    // along its +z), and up and to the right on screen (its +y and -x).
    let to64 = |p: Point<(), f32>| p.map_coefs(f64::from);
    let lamp = to64(cam.pose >> Point::direction(0.3, 0.8, -1.0)).cast::<Direction>();
    let facing = |n: Direction| (dot(n, lamp) / (length(n) * length(lamp))).abs() as f32;
    // The parts in the world, and four planes through the top's axis an eighth of a turn
    // apart: crossing any of them flips the sign of the product of a point's distances to
    // them, so that sign alternates from one eighth of a turn about the axis to the next.
    let placed: Vec<Quadric> = parts.iter().map(|q| moved(motor, *q)).collect();
    let spokes = [
        Plane::new(1.0, 0.0, 0.0, 0.0),
        Plane::new(1.0, -1.0, 0.0, 0.0),
        Plane::new(0.0, 1.0, 0.0, 0.0),
        Plane::new(1.0, 1.0, 0.0, 0.0),
    ];
    let colours = PART_COLOURS.map(|[r, g, b]| Light::from_srgb(r, g, b, SURFACE));
    // The bowl in the lattice's blue, as the floors elsewhere.
    let ground_tone = light(0.3, 0.42, 1.0, 0.12);
    c.clip(view);
    c.shade(2, |q| {
        let (o, d) = cam.ray(q);
        let (origin, heading) = (to64(o), to64(d));
        let mut best: Option<(f64, Light)> = None;
        if let Some((distance, p)) = hit(ground, origin, heading) {
            let [hx, hy, _] = p.to_euclidean();
            let checker = ((hx / 0.15).floor() + (hy / 0.15).floor()).rem_euclid(2.0) as f32;
            let shade = 0.45 + 0.55 * facing(normal_at(ground, p));
            best = Some((distance, ground_tone.faded((0.62 + 0.18 * checker) * shade)));
        }
        for (i, q) in placed.iter().enumerate() {
            if let Some((distance, p)) = hit(*q, origin, heading)
                && best.is_none_or(|(b, _)| distance < b)
            {
                // Alternating sectors about the top's axis, in its own frame, show its spin.
                let local = motor << p;
                let sign: f64 = spokes.iter().map(|s| (*s & local).s()).product();
                let sector = f32::from(u8::from(sign <= 0.0));
                let tint = colours[i].faded(1.0 - SECTOR_CONTRAST[i] * (1.0 - sector));
                let shade = 0.35 + 0.65 * facing(normal_at(*q, p));
                best = Some((distance, tint.faded(shade)));
            }
        }
        best.map(|(_, l)| l)
    });
    c.unclip();
}

fn draw(c: &mut Canvas, runs: &Runs, t: f32) {
    backdrop(c);
    let screen = c.rect();
    let (w, h) = (screen.width(), screen.height());
    let u = h / 270.0;
    let k = ((f64::from(t) / FRAME) as usize).min(runs.samples[0].len() - 1);
    // The view: a square at the bottom left.
    let side = h * 0.86;
    let corner = screen.bottom_left() + Point2::direction(4.0 * u, -2.0 * u);
    let view = Rect {
        lo: corner - Point2::direction(0.0, side),
        hi: corner + Point2::direction(side, 0.0),
    };
    let motor = runs.samples[0][k].motor;
    render(c, motor, &runs.parts, runs.ground, view);
    caption(
        c,
        "SPINNING TOP: QUADRICS ON THE GROUND",
        "CONTACTS CORRECTED PART BY PART, STICKING OR SLIPPING; RAY TRACED (PGA3D)",
    );
    // The lean and the spin of the four variations.
    let time = ((k + 1) as f64 * FRAME) as f32;
    let x0 = side + 10.0 * u;
    let panels = [
        ("LEAN (DEGREES)", [0.0, 25.0]),
        ("SPIN (RAD/S)", [0.0, 20.0]),
    ];
    for (p, (title, range)) in panels.iter().enumerate() {
        let top = h * 0.12 + p as f32 * h * 0.44;
        let row = Rect::new(x0, top, w, top + h * 0.44);
        let rect = row.inset(24.0 * u, 14.0 * u, 6.0 * u, 16.0 * u);
        let ax = Axes::new(rect, [0.0, RUNTIME as f32], *range);
        ax.frame(c, title, if p == 1 { "TIME (S)" } else { "" }, "");
        for (v, samples) in runs.samples.iter().enumerate() {
            // The chart's points: (time, value).
            let pts: Vec<gax::pga2d::Point> = samples[..=k]
                .iter()
                .enumerate()
                .map(|(j, s)| {
                    let value = if p == 0 { s.lean.to_degrees() } else { s.spin };
                    gax::pga2d::Point::xy((j + 1) as f64 * FRAME, value)
                })
                .collect();
            let width = if v == 0 { 1.8 } else { 1.1 };
            ax.polyline(c, &pts, width, palette::series(v));
        }
        let (low, high) = (Point2::xy(time, range[0]), Point2::xy(time, range[1]));
        ax.line(c, low, high, 0.8, palette::GRID);
        // The legend where the spin has decayed.
        if p == 1 {
            let entries: Vec<(&str, Light)> = VARIATIONS
                .iter()
                .enumerate()
                .map(|(v, (name, _))| (*name, palette::series(v)))
                .collect();
            ax.legend(c, &entries);
        }
    }
}

fn main() {
    run_with(
        Anim::new("spinning top", RUNTIME as f32)
            .size(480, 270)
            .scale(2),
        runs,
        draw,
    );
}

#[cfg(test)]
mod tests {
    use super::scenarios::*;
    use super::top::*;
    use gax::pga3d::{Direction, Point};
    use gax_numga_examples::testing::close;

    /// The lowest point of a sphere against the vertical is its bottom, and of an ellipsoid
    /// against a slanted normal the point where its tangent plane has that normal; the flat
    /// ground's normal points up, and a point below it is inside the ground.
    #[test]
    fn contact_geometry() {
        let sphere = ellipsoid(Point::xyz(0.1, 0.0, 1.0), [0.5; 3]);
        let up = Point::direction(0.0, 0.0, 1.0).cast::<Direction>();
        let low = lowest_point(sphere, up);
        let [x, y, z] = low.to_euclidean();
        assert!(close(x, 0.1, 1e-12) && close(y, 0.0, 1e-12) && close(z, 0.5, 1e-12));
        let e = ellipsoid(Point::xyz(0.0, 0.2, 0.3), [0.5, 0.3, 0.2]);
        let slanted = Point::direction(0.3, -0.2, 1.0).cast::<Direction>();
        let p = lowest_point(e, slanted);
        assert!((e.of(p) & p).s().abs() < 1e-12, "on the surface");
        let tangent = e.of(p).dual().cast::<Direction>();
        let across = (Point::from(tangent) & Point::from(slanted)).norm();
        assert!(
            across < 1e-12 && dot(tangent, slanted) < 0.0,
            "facing against the normal"
        );
        let ground = bowl(BASE.ground_curvature);
        let n = ground_normal(ground, Point::xyz(0.3, -0.2, 0.1));
        assert!(close(n.e021(), 1.0, 1e-12) && close(n.e032(), 0.0, 1e-12));
        let p = Point::xyz(0.3, 0.0, -0.01);
        assert!((ground.of(p) & p).s() < 0.0);
    }

    /// numga's test: a short spin stays up, leaning less than twice its start, and renders.
    #[test]
    fn a_short_spin_stays_up_and_renders() {
        let (body, ground, samples) = spin(BASE, 0.1, STEP, 20);
        let most = samples.iter().map(|s| s.lean).fold(0.0, f64::max);
        assert!(most < 2.0 * TILT, "{most}");
        let mut c = gax_numga_examples::Canvas::new(32, 32);
        let view = gax_numga_examples::Rect::new(0.0, 0.0, 32.0, 32.0);
        super::render(
            &mut c,
            samples[samples.len() - 1].motor,
            &body.parts,
            ground,
            view,
        );
        assert!(c.mean().luma() > 0.0);
    }

    /// numga's scenario checks: every top stays up over the run, leaning less than it would
    /// lying down, and keeps spinning the same way while the air's drag and the ground's
    /// friction slow it; its tip neither sinks into the ground nor flies off. Each top leans and
    /// spins as numga's does (35797de's numbers, which the port reproduces to rounding).
    #[test]
    fn every_top_stays_up_and_keeps_spinning() {
        let runs = super::runs();
        for (samples, (name, _)) in runs.samples.iter().zip(VARIATIONS) {
            let most = samples.iter().map(|s| s.lean).fold(0.0, f64::max);
            assert!(most < core::f64::consts::FRAC_PI_4, "{name}: {most}");
            assert!(samples.iter().all(|s| s.spin > 0.0), "{name}");
            assert!(samples[samples.len() - 1].spin < samples[0].spin, "{name}");
        }
        let numga = [
            (0.356_061_041_069_150_1, 0.715_730_655_944_576_1),
            (0.352_187_062_169_433_7, 0.559_378_015_783_595_8),
            (0.355_885_673_373_888_27, 4.598_048_553_651_883),
            (0.355_840_223_163_805_15, 7.408_428_919_411_593),
        ];
        for (samples, (lean, spin)) in runs.samples.iter().zip(numga) {
            let most = samples.iter().map(|s| s.lean).fold(0.0, f64::max);
            let last = samples[samples.len() - 1].spin;
            assert!(
                (most - lean).abs() < 1e-8 && (last - spin).abs() < 1e-6,
                "{most} {last}"
            );
        }
        let base = &runs.samples[0];
        for s in base {
            let placed = moved(s.motor, runs.parts[2]);
            let low = lowest_point(placed, ground_normal(runs.ground, centre(placed)));
            let g = runs.ground.of(low);
            let height = (g & low).s() / (2.0 * g.norm());
            assert!(height > -2e-3 && height < 2e-2, "{height}");
        }
    }

    #[test]
    fn a_frame_draws() {
        let runs = super::runs();
        let c = gax_numga_examples::app::frame(
            &gax_numga_examples::Anim::new("t", 1.0).size(160, 90),
            0.5,
            &mut |c, t| super::draw(c, &runs, t),
        );
        assert!(c.mean().luma() > 0.0);
    }
}
