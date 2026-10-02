//! numga's `mechanics/spinning_top`: a spinning top of three ellipsoids on a concave ground, in
//! PGA3D, ray traced.
//!
//! Every shape is a quadric, a map from points to planes (`Plane<(Point,)>`): a point's polar
//! plane. The top's parts are solid ellipsoids in its own frame; the ground is the solid below a
//! shallow paraboloid, a concave quadric. A part's lowest point against the ground is the pole
//! of its tangent plane facing the ground; how far that point lies below the ground is its
//! depth, zero for a part in the air. Contact is position based: after a free step each part is
//! pressed out along the ground's normal by its depth, then held against sliding up to the
//! friction cone and against turning about the normal up to the drilling limit, each a single
//! constraint along one line, summed over the parts. The rate is read back from the motor's
//! step.
//!
//! The contact point is found in one pass: the ground's normal is taken at the part's centre,
//! and the part's lowest point against that normal is the contact. That is accurate while the
//! part is more sharply curved than the ground, a small body on a large one, as here.
//!
//! The animation ray traces the base top (each pixel's ray meets the nearest quadric, its
//! entry the root of the quadric's form along the ray) as it spins, precesses and wanders in
//! its bowl, and plots the tilt and spin of numga's four variations: the base, a sharper tip,
//! a heavier disc and a slippery bowl.

use gax::pga3d::{Line, Motor, Plane, Point};
use gax_numga_examples::{
    Anim, Axes, Camera, Canvas, Lens, backdrop, canvas, caption, palette, plot, run,
};

#[path = "../shared/mechanics_lie.rs"]
mod lie;

/// The Lie steppers for PGA3D motors: rates and forques are both lines.
mod rigid {
    use gax::pga3d::{Line as Rate, Line as Forque, Motor, Point};
    crate::lie::integrators!();
}

mod top {
    use super::*;
    pub use crate::rigid::{InertiaInv, M, P, R};

    /// A quadric: each point to its polar plane.
    pub type Quadric = Plane<(Point,), f64>;
    /// A plane.
    pub type Pl = Plane<(), f64>;

    /// The plane at infinity, `e0`: its pairing with a point is the point's weight.
    pub fn w() -> Pl {
        Plane::new(0.0, 0.0, 0.0, 1.0)
    }

    /// The coordinate planes `x = 0`, `y = 0`, `z = 0`.
    pub fn axes() -> [Pl; 3] {
        [
            Plane::new(1.0, 0.0, 0.0, 0.0),
            Plane::new(0.0, 1.0, 0.0, 0.0),
            Plane::new(0.0, 0.0, 1.0, 0.0),
        ]
    }

    /// The direction part of a point: numga's cast to its `Direction` type, the degenerate
    /// antivectors (gax has no kind for that set of blades, so this drops the weight by hand).
    pub fn direction(p: P) -> P {
        Point::direction(p.e032(), p.e013(), p.e021())
    }

    /// The Euclidean inner product of two directions: of the planes through the origin they are
    /// normal to.
    pub fn dot(a: P, b: P) -> f64 {
        (a.dual() | b.dual()).s()
    }

    /// The Euclidean length of a direction.
    pub fn length(d: P) -> f64 {
        d.dual().norm()
    }

    /// The value with its magnitude held to the limit.
    pub fn clamp(value: f64, limit: f64) -> f64 {
        let limit = limit.max(0.0);
        value.clamp(-limit, limit)
    }

    // --- shapes and mass ----------------------------------------------------------------------

    /// A solid ellipsoid about a centre with the given semi-axes: the dyads of the planes
    /// through the centre, less the dyad of the plane at infinity. Its form `Q(p) & p` is
    /// negative inside.
    pub fn ellipsoid(centre: P, semi: [f64; 3]) -> Quadric {
        let slot = Point::slot();
        let mut q = -(w() * (w() & slot));
        for (a, s) in axes().into_iter().zip(semi) {
            let through = a - w() * (a & centre).s();
            q += through * (through & slot) * (1.0 / (s * s));
        }
        q
    }

    /// The ground: solid below the paraboloid whose height is the curvature times the squared
    /// distance from the z axis; negative inside. A bowl for positive curvature.
    pub fn bowl(curvature: f64) -> Quadric {
        let slot = Point::slot();
        let [x, y, z] = axes();
        (z * (w() & slot) + w() * (z & slot)) * 0.5 - (x * (x & slot) + y * (y & slot)) * curvature
    }

    /// Six mass points per solid ellipsoid with its mass, centroid and second moments, at
    /// `sqrt(3 / 5)` of each semi-axis on either side.
    pub fn sigma_points(
        centres: &[[f64; 3]],
        semi: &[[f64; 3]],
        mass: &[f64],
    ) -> (Vec<P>, Vec<f64>) {
        let (mut points, mut masses) = (Vec::new(), Vec::new());
        for ((c, s), m) in centres.iter().zip(semi).zip(mass) {
            for sign in [1.0, -1.0] {
                for axis in 0..3 {
                    let mut x = *c;
                    x[axis] += sign * s[axis] * (3.0f64 / 5.0).sqrt();
                    points.push(Point::xyz(x[0], x[1], x[2]));
                    masses.push(m / 6.0);
                }
            }
        }
        (points, masses)
    }

    // --- contact -------------------------------------------------------------------------------

    /// The ground's unit normal at a point, toward the air, where its form grows. A point's
    /// polar plane under the ground quadric is the gradient of the ground's form there; read as
    /// a direction and divided by its length it is the normal, on the ground or off it.
    pub fn ground_normal(ground: Quadric, p: P) -> P {
        let g = ground.of(p);
        direction(g.dual()) * (1.0 / g.norm())
    }

    /// The quadric's point furthest against the normal: the pole of its tangent plane on that
    /// side. The plane through the quadric's centre normal to the direction has its pole at
    /// infinity, the direction conjugate to it, along which the centre reaches the planes
    /// tangent to the quadric parallel to it; scaled to reach the surface, it leads from the
    /// centre to the lowest point.
    pub fn lowest_point(surface: Quadric, normal: P) -> P {
        // Planes to their poles.
        let dual = surface.inverse();
        let pole = dual.of(w());
        let centre = pole * (1.0 / (w() & pole).s());
        // The plane through the centre, normal to the direction, and its pole, a direction.
        let plane = normal.dual() - w() * (normal.dual() & centre).s();
        let conjugate = dual.of(plane);
        centre + conjugate * (1.0 / (-(plane & conjugate).s() * (w() & pole).s()).sqrt())
    }

    /// The direction less its part along the ground's normal: its part in the ground's plane.
    pub fn along_ground(d: P, normal: P) -> P {
        d - normal * dot(d, normal)
    }

    /// The body's twist per unit forque along a line, and its compliance along that line.
    pub fn compliance(motor: M, inertia_inv: InertiaInv, line: Line<(), f64>) -> (R, f64) {
        let body = motor << line;
        let step = inertia_inv.of(body);
        (step, (step & body).s())
    }

    /// Carry the body by the summed twists of the forques.
    pub fn move_by(motor: M, twist: R) -> M {
        (motor * (twist * -0.5).exp()).normalized()
    }

    /// The two principal curvatures at a point on the surface, in order; convex surfaces curve
    /// positively (numga's `geometry/surface_curvature`). The Hessian form `Q(p) & p'` with
    /// both slots composed with the projector onto the tangent plane is the second fundamental
    /// form up to the length of the gradient; against the metric its eigenvalues are the
    /// principal curvatures and zeros. numga takes the form on the directions only; gax has no
    /// kind for them, so the form lives on all points, the projector also drops the weight, and
    /// the metric adds the weight's square to stay positive definite: two more zero
    /// eigenvalues, the normal's and the weight's.
    pub fn principal(surface: Quadric, p: P) -> [f64; 2] {
        let tangent = surface.of(p);
        let normal = direction(tangent.dual());
        let slot = Point::slot();
        let origin = Point::xyz(0.0, 0.0, 0.0);
        // A point's direction part, then its projection along the normal onto the tangent plane.
        let strip = slot - origin * (w() & slot);
        let project = strip - normal * (tangent & strip) * (1.0 / (tangent & normal).s());
        let hessian = surface & slot;
        let second = hessian.of(project).at::<1>().of(project) * (1.0 / length(normal));
        let metric = (slot.dual() | slot.dual()) + (w() & slot) * (w() & slot);
        let (values, _) = second.eigh_with(metric);
        // The two values that are not the zeros.
        let mut v = values;
        v.sort_by(|a, b| a.abs().total_cmp(&b.abs()));
        let [lo, hi] = [v[2].min(v[3]), v[2].max(v[3])];
        [lo, hi]
    }

    /// Contact and friction, as numga's `project_contacts`: correct the predicted pose, pressing
    /// each part out by its depth, then holding it against sliding and against turning about
    /// the normal, within the friction cone and the drilling limit; return the corrected motor
    /// and the rate it implies, less the contacts' sliding velocity up to dynamic friction.
    #[allow(clippy::too_many_arguments)]
    pub fn project_contacts(
        before: M,
        mut motor: M,
        inertia_inv: InertiaInv,
        parts: &[Quadric],
        ground: Quadric,
        friction: [f64; 2],
        indentation: f64,
        dt: f64,
    ) -> (M, R) {
        let [stat, dynamic] = friction;
        // The parts in the world, and their centres: the poles of the plane at infinity.
        let placed: Vec<Quadric> = parts
            .iter()
            .map(|q| motor >> q.of(motor << Point::slot()))
            .collect();
        // Each part's lowest point against the ground's normal at its centre is its contact.
        let contact: Vec<P> = placed
            .iter()
            .map(|q| {
                let c = q.solve(w());
                lowest_point(*q, ground_normal(ground, c * (1.0 / (w() & c).s())))
            })
            .collect();
        let normal: Vec<P> = contact.iter().map(|c| ground_normal(ground, *c)).collect();
        // The ground's form at the contact over its gradient's length: how far the point lies
        // below the ground, to first order in the depth; a part in the air has none.
        let depth: Vec<f64> = contact
            .iter()
            .map(|c| {
                let g = ground.of(*c);
                (-(g & *c).s() / (2.0 * g.norm())).max(0.0)
            })
            .collect();
        let n = parts.len();

        // Press out along the normal line through each contact, by its depth.
        let mut pressed = vec![0.0; n];
        let mut twist = Line::zero();
        for i in 0..n {
            let (step, give) = compliance(motor, inertia_inv, contact[i] & normal[i]);
            pressed[i] = depth[i] / give;
            twist += step * pressed[i];
        }
        motor = move_by(motor, twist);

        // Hold against sliding: back toward where the contact's material point was, in the
        // ground's plane.
        let mut twist = Line::zero();
        for i in 0..n {
            let slid = along_ground(contact[i] - (before >> (motor << contact[i])), normal[i]);
            let back = length(slid);
            let line = contact[i] & (slid * (-1.0 / (back + 1e-12)));
            let (step, give) = compliance(motor, inertia_inv, line);
            twist += step * clamp(back / give, pressed[i] * stat);
        }
        motor = move_by(motor, twist);

        // Hold against turning about the normal: undo the step's turn, with at most static
        // friction times the pressing forque times the contact patch's radius, the square root
        // of the radius of curvature times the indentation. The couple is the torque about the
        // normal: the line at infinity of the planes normal to it.
        let turn: R = (motor * before.reverse()).log();
        let mut twist = Line::zero();
        for i in 0..n {
            let couple = normal[i].dual() ^ w();
            let turned = (couple & (turn * -2.0)).s();
            let [k1, k2] = principal(placed[i], contact[i]);
            let patch = ((2.0 / (k1 + k2)).abs() * indentation).sqrt();
            let (step, give) = compliance(motor, inertia_inv, couple);
            twist += step * clamp(-turned / give, pressed[i] * stat * patch);
        }
        motor = move_by(motor, twist);

        // Dynamic friction: the rate the corrected step implies, less the contacts' sliding
        // velocity, with at most the dynamic coefficient times the normal impulse.
        let rate = crate::rigid::rate_between(before, motor, dt);
        let mut twist = Line::zero();
        for i in 0..n {
            let sliding = along_ground(contact[i].commutator(motor >> rate), normal[i]);
            let speed = length(sliding);
            let line = contact[i] & (sliding * (-1.0 / (speed + 1e-12)));
            let (step, give) = compliance(motor, inertia_inv, line);
            twist += step * clamp(speed / give, pressed[i] * dynamic / dt);
        }
        (motor, rate + twist)
    }
}

mod scenarios {
    use super::top::*;
    use super::*;

    /// A top and its ground: the spin it starts with, its tip's radius, its disc's height above
    /// the tip and mass, the bowl's curvature, the dynamic friction, the air's drag rate, and
    /// the indentation of the tip that sets the drilling friction.
    #[derive(Clone, Copy, Debug)]
    pub struct Setup {
        pub spin_rate: f64,
        pub tip_radius: f64,
        pub disc_height: f64,
        pub disc_mass: f64,
        pub bowl_curvature: f64,
        pub dynamic_friction: f64,
        pub drag: f64,
        pub indentation: f64,
    }

    pub const BASE: Setup = Setup {
        spin_rate: 18.0,
        tip_radius: 0.05,
        disc_height: 0.10,
        disc_mass: 1.0,
        bowl_curvature: 0.15,
        dynamic_friction: 0.3,
        drag: 0.13,
        indentation: 0.0,
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
    pub const STATIC_FRICTION: f64 = 0.4;
    /// numga's view: elevation and azimuth in degrees, the centre and the half width.
    pub const VIEW: (f64, f64) = (30.0, -60.0);
    pub const CENTRE: [f64; 3] = [0.0, 0.0, 0.15];
    pub const EXTENT: f64 = 0.55;

    /// The top's parts, inertia, mass and the height of its centre of mass above the tip.
    pub struct Top {
        pub parts: Vec<Quadric>,
        pub inertia: crate::rigid::Inertia,
        pub inertia_inv: InertiaInv,
        pub mass: f64,
        pub height: f64,
    }

    /// A disc, a stem and a tip, each a solid ellipsoid, in the frame of their centre of mass.
    pub fn top(tip_radius: f64, tip_height: f64, disc_height: f64, disc_mass: f64) -> Top {
        let centres = [
            [0.0, 0.0, disc_height],
            [0.0, 0.0, disc_height + 0.16],
            [0.0, 0.0, tip_height],
        ];
        let semi = [
            [0.25, 0.25, 0.04],
            [0.02, 0.02, 0.15],
            [tip_radius, tip_radius, tip_height],
        ];
        let mass = [disc_mass, 0.05, 0.15];
        let total: f64 = mass.iter().sum();
        let com = (0..3).fold(0.0, |a, i| a + centres[i][2] * mass[i]) / total;
        let centres = centres.map(|c| [c[0], c[1], c[2] - com]);
        let parts = (0..3)
            .map(|i| {
                let [x, y, z] = centres[i];
                ellipsoid(Point::xyz(x, y, z), semi[i])
            })
            .collect();
        let (points, masses) = sigma_points(&centres, &semi, &mass);
        let (inertia, inertia_inv) = crate::rigid::inertia_of(&points, &masses);
        Top {
            parts,
            inertia,
            inertia_inv,
            mass: total,
            height: com,
        }
    }

    /// One recorded moment: the pose, the axis's angle from vertical, and the spin.
    #[derive(Clone, Copy, Debug)]
    pub struct Sample {
        pub motor: M,
        pub tilt: f64,
        pub spin: f64,
    }

    /// Simulate from a slight tilt, recording every few steps.
    pub fn spin(setup: Setup, seconds: f64, dt: f64, every: usize) -> (Top, Quadric, Vec<Sample>) {
        let body = top(setup.tip_radius, 0.14, setup.disc_height, setup.disc_mass);
        let ground = bowl(setup.bowl_curvature);
        let origin = Point::xyz(0.0, 0.0, 0.0);
        let up = Point::direction(0.0, 0.0, 1.0);
        let tilt: f64 = 0.15;
        // Lift the centre of mass so that the tip clears, then tilt.
        let mut motor = Motor::translation(0.0, 0.0, body.height * tilt.cos() + 0.002)
            * Motor::rotation_about(0.0, 1.0, 0.0, tilt);
        let mut rate = Line::new(0.0, 0.0, setup.spin_rate, 0.0, 0.0, 0.0);
        // The weight, a force line down through the centre of mass, and the air's drag,
        // opposing the momentum; in the body frame.
        let weight = Point::direction(0.0, 0.0, -9.81 * body.mass);
        let forces = |m: M, r: R| (m << ((m >> origin) & weight)) - body.inertia.of(r) * setup.drag;
        let mut samples = Vec::new();
        for i in 0..(seconds / dt).round() as usize {
            let before = motor;
            // Predict with a free step, then correct against the ground.
            // The free step's rate is not needed: the contacts read it back from the motors.
            let (m, _) = crate::rigid::explicit_rk4(
                motor,
                rate,
                body.inertia,
                body.inertia_inv,
                dt,
                &forces,
            );
            (motor, rate) = project_contacts(
                before,
                m.normalized(),
                body.inertia_inv,
                &body.parts,
                ground,
                [STATIC_FRICTION, setup.dynamic_friction],
                setup.indentation,
                dt,
            );
            if i % every == 0 {
                let lean = dot(motor >> up, up).abs().min(1.0).acos();
                samples.push(Sample {
                    motor,
                    tilt: lean,
                    spin: rate.e12().abs(),
                });
            }
        }
        (body, ground, samples)
    }

    /// Where a ray from `origin` along the unit direction `heading` enters the solid quadric:
    /// its distance and the point. Bound to the ray in both slots, the form is a quadratic in
    /// the distance; at the entering root it falls through zero, so the root is the one with
    /// `a distance + b = -sqrt(discriminant)`.
    pub fn hit(surface: Quadric, origin: P, heading: P) -> Option<(f64, P)> {
        let along = surface.of(heading);
        let a = (along & heading).s();
        let b = (along & origin).s();
        let c = (surface.of(origin) & origin).s();
        let discriminant = b * b - a * c;
        if discriminant < 0.0 {
            return None;
        }
        let distance = (-b - discriminant.sqrt()) / a;
        (distance.is_finite() && distance > 0.0).then(|| (distance, origin + heading * distance))
    }

    /// The unit normal of the quadric at a point on it: the normal of its polar plane.
    pub fn normal_at(surface: Quadric, p: P) -> [f64; 3] {
        let n = direction(surface.of(p).dual());
        let l = length(n);
        [n.e032() / l, n.e013() / l, n.e021() / l]
    }
}

use scenarios::*;
use top::*;

const SECONDS: f64 = 5.0;
const DT: f64 = 1e-3;
const EVERY: usize = 10;

/// The four variations, simulated once: the base's parts and ground, and every top's samples.
struct Runs {
    parts: Vec<Quadric>,
    ground: Quadric,
    samples: Vec<Vec<Sample>>,
}

fn runs() -> &'static Runs {
    static RUNS: std::sync::OnceLock<Runs> = std::sync::OnceLock::new();
    RUNS.get_or_init(|| {
        let all: Vec<_> = std::thread::scope(|s| {
            let handles: Vec<_> = VARIATIONS
                .iter()
                .map(|(_, setup)| s.spawn(move || spin(*setup, SECONDS, DT, EVERY)))
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("a run"))
                .collect()
        });
        let (base, ground, _) = &all[0];
        Runs {
            parts: base.parts.clone(),
            ground: *ground,
            samples: all.iter().map(|(_, _, s)| s.clone()).collect(),
        }
    })
}

/// The colours of the disc, the stem and the tip (sRGB), and their sector contrasts.
const PART_COLOURS: [[f32; 3]; 3] = [[0.85, 0.30, 0.25], [0.35, 0.35, 0.40], [0.80, 0.70, 0.35]];
const SECTOR_CONTRAST: [f32; 3] = [0.25, 0.0, 0.0];

/// The ray-traced top placed by the motor, in the square `[x0, y0, side]` of the canvas.
fn render(c: &mut Canvas, motor: M, parts: &[Quadric], ground: Quadric, at: [f32; 3]) {
    let [x0, y0, side] = at;
    let (elevation, azimuth) = (VIEW.0.to_radians(), VIEW.1.to_radians());
    let cam = Camera::orbit(
        side as usize,
        side as usize,
        CENTRE.map(|v| v as f32),
        (20.0 * EXTENT) as f32,
        azimuth as f32,
        elevation as f32,
        Lens::Parallel(EXTENT as f32),
    );
    // The lamp, above the viewer's left shoulder.
    let back = [
        elevation.cos() * azimuth.cos(),
        elevation.cos() * azimuth.sin(),
        elevation.sin(),
    ];
    let right = [-azimuth.sin(), azimuth.cos(), 0.0];
    let up = [
        back[1] * right[2] - back[2] * right[1],
        back[2] * right[0] - back[0] * right[2],
        back[0] * right[1] - back[1] * right[0],
    ];
    let lamp = [0, 1, 2].map(|i| back[i] - 0.3 * right[i] + 0.8 * up[i]);
    let ll = (lamp[0] * lamp[0] + lamp[1] * lamp[1] + lamp[2] * lamp[2]).sqrt();
    let lamp = lamp.map(|v| v / ll);
    let facing = |n: [f64; 3]| (n[0] * lamp[0] + n[1] * lamp[1] + n[2] * lamp[2]).abs() as f32;
    // The parts in the world.
    let placed: Vec<Quadric> = parts
        .iter()
        .map(|q| motor >> q.of(motor << Point::slot()))
        .collect();
    let to64 =
        |p: Point<(), f32>| Point::new(p.c[0] as f64, p.c[1] as f64, p.c[2] as f64, p.c[3] as f64);
    let colours = PART_COLOURS.map(|[r, g, b]| canvas::srgb(r, g, b));
    let ground_tone = canvas::srgb(0.95, 0.93, 0.88);
    c.clip([x0, y0, x0 + side, y0 + side]);
    c.shade(2, |x, y| {
        let (o, d) = cam.ray([x - x0, y - y0]);
        let (origin, heading) = (to64(o), to64(d));
        let mut best: Option<(f64, [f32; 3])> = None;
        if let Some((distance, p)) = hit(ground, origin, heading) {
            let [hx, hy, _] = p.to_euclidean();
            let checker = ((hx / 0.15).floor() + (hy / 0.15).floor()).rem_euclid(2.0) as f32;
            let shade = 0.45 + 0.55 * facing(normal_at(ground, p));
            best = Some((
                distance,
                canvas::scale(ground_tone, (0.62 + 0.18 * checker) * shade),
            ));
        }
        for (i, q) in placed.iter().enumerate() {
            if let Some((distance, p)) = hit(*q, origin, heading)
                && best.is_none_or(|(b, _)| distance < b)
            {
                // Alternating sectors about the top's axis, in its own frame, show its spin.
                let [lx, ly, _] = (motor << p).to_euclidean();
                let sector = (ly.atan2(lx) / core::f64::consts::FRAC_PI_4)
                    .floor()
                    .rem_euclid(2.0) as f32;
                let tint = canvas::scale(colours[i], 1.0 - SECTOR_CONTRAST[i] * (1.0 - sector));
                let shade = 0.35 + 0.65 * facing(normal_at(*q, p));
                best = Some((distance, canvas::scale(tint, shade)));
            }
        }
        best.map(|(_, rgb)| rgb)
    });
    c.unclip();
}

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let runs = runs();
    let (w, h) = (c.width as f32, c.height as f32);
    let u = h / 270.0;
    let k = ((f64::from(t) / (DT * EVERY as f64)) as usize).min(runs.samples[0].len() - 1);
    let side = h * 0.86;
    render(
        c,
        runs.samples[0][k].motor,
        &runs.parts,
        runs.ground,
        [4.0 * u, h - side - 2.0 * u, side],
    );
    caption(
        c,
        "SPINNING TOP: QUADRICS IN A BOWL",
        "CONTACT, FRICTION AND RAY TRACING IN PGA3D",
    );
    // The tilt and the spin of the four variations.
    let time = (k as f64 * DT * EVERY as f64) as f32;
    let x0 = side + 10.0 * u;
    let panels = [
        ("TILT (DEGREES)", [0.0, 25.0]),
        ("SPIN (RAD/S)", [0.0, 20.0]),
    ];
    for (p, (title, range)) in panels.iter().enumerate() {
        let top = h * 0.12 + p as f32 * h * 0.44;
        let rect = plot::inset(
            [x0, top, w, top + h * 0.44],
            24.0 * u,
            14.0 * u,
            6.0 * u,
            16.0 * u,
        );
        let ax = Axes::new(rect, [0.0, SECONDS as f32], *range);
        ax.frame(c, title, if p == 1 { "TIME (S)" } else { "" }, "");
        for (v, samples) in runs.samples.iter().enumerate() {
            let pts: Vec<[f32; 2]> = samples[..=k]
                .iter()
                .enumerate()
                .map(|(j, s)| {
                    let value = if p == 0 { s.tilt.to_degrees() } else { s.spin };
                    [(j as f64 * DT * EVERY as f64) as f32, value as f32]
                })
                .collect();
            ax.polyline(
                c,
                &pts,
                if v == 0 { 1.8 } else { 1.1 },
                palette::series(v),
                1.0,
            );
        }
        ax.line(
            c,
            [time, range[0]],
            [time, range[1]],
            0.8,
            palette::grid(),
            1.0,
        );
        // The legend where the spin has decayed.
        if p == 1 {
            let entries: Vec<(&str, gax_numga_examples::Rgb)> = VARIATIONS
                .iter()
                .enumerate()
                .map(|(v, (name, _))| (*name, palette::series(v)))
                .collect();
            ax.legend(c, &entries);
        }
    }
}

fn main() {
    run(
        Anim::new("spinning top", SECONDS as f32)
            .size(480, 270)
            .scale(2),
        draw,
    );
}

#[cfg(test)]
mod tests {
    use super::scenarios::*;
    use super::top::*;
    use gax::pga3d::Point;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    /// A sphere curves by the inverse of its radius both ways; an ellipsoid at the end of its
    /// semi-axis `c` by `c / a²` and `c / b²`.
    #[test]
    fn principal_curvatures_of_spheres_and_ellipsoids() {
        let sphere = ellipsoid(Point::xyz(0.2, -0.1, 0.3), [0.5; 3]);
        let [k1, k2] = principal(sphere, Point::xyz(0.2, 0.2, 0.3 + 0.4));
        assert!(close(k1, 2.0, 1e-9) && close(k2, 2.0, 1e-9), "{k1} {k2}");
        let (a, b, c) = (0.5, 0.3, 0.2);
        let e = ellipsoid(Point::xyz(0.0, 0.0, 0.0), [a, b, c]);
        let [k1, k2] = principal(e, Point::xyz(0.0, 0.0, c));
        assert!(
            close(k1, c / (a * a), 1e-9) && close(k2, c / (b * b), 1e-9),
            "{k1} {k2}"
        );
        // The bowl is concave from the air: it curves negatively, by twice its curvature at
        // the bottom.
        let [k1, k2] = principal(bowl(0.15), Point::xyz(0.0, 0.0, 0.0));
        assert!(close(k1, -0.3, 1e-9) && close(k2, -0.3, 1e-9), "{k1} {k2}");
    }

    /// The lowest point of a sphere against the vertical is its bottom; the bowl's normal at its
    /// bottom points up, and a point below it is inside the ground.
    #[test]
    fn contact_geometry() {
        let sphere = ellipsoid(Point::xyz(0.1, 0.0, 1.0), [0.5; 3]);
        let low = lowest_point(sphere, Point::direction(0.0, 0.0, 1.0));
        let [x, y, z] = low.to_euclidean();
        assert!(close(x, 0.1, 1e-12) && close(y, 0.0, 1e-12) && close(z, 0.5, 1e-12));
        let ground = bowl(0.15);
        let n = ground_normal(ground, Point::xyz(0.0, 0.0, 0.0));
        assert!(close(n.e021(), 1.0, 1e-12));
        let p = Point::xyz(0.3, 0.0, 0.0);
        assert!((ground.of(p) & p).s() < 0.0);
        // Off centre the normal leans toward the axis, by the slope 2 k x.
        let n = ground_normal(ground, p);
        assert!(close(-n.e032() / n.e021(), 2.0 * 0.15 * 0.3, 1e-12));
    }

    /// numga's render test: one sample, one small frame.
    #[test]
    fn frame_renders() {
        let (body, ground, samples) = spin(BASE, 0.01, 1e-3, 1);
        let mut c = gax_numga_examples::Canvas::new(32, 32);
        super::render(
            &mut c,
            samples[0].motor,
            &body.parts,
            ground,
            [0.0, 0.0, 32.0],
        );
        assert!(c.mean()[0] > 0.0);
    }

    /// The top stays on the ground, spinning: its tip neither sinks in nor flies off, it stays
    /// upright, and drag slows its spin.
    #[test]
    fn the_top_spins_on_the_ground() {
        let (body, ground, samples) = spin(BASE, 1.0, 1e-3, 10);
        let last = samples[samples.len() - 1];
        assert!(last.tilt < 0.5, "{}", last.tilt);
        assert!(last.spin > 10.0 && last.spin < 18.0, "{}", last.spin);
        for s in &samples {
            let placed = s.motor >> body.parts[2].of(s.motor << Point::slot());
            let c = placed.solve(w());
            let low = lowest_point(placed, ground_normal(ground, c * (1.0 / (w() & c).s())));
            let g = ground.of(low);
            let height = (g & low).s() / (2.0 * g.norm());
            assert!(height > -2e-3 && height < 2e-2, "{height}");
        }
    }

    #[test]
    fn a_frame_draws() {
        let mut draw = super::draw;
        let c = gax_numga_examples::app::frame(
            &gax_numga_examples::Anim::new("t", 1.0).size(160, 90),
            0.5,
            &mut draw,
        );
        assert!(c.mean()[0] > 0.0);
    }
}
