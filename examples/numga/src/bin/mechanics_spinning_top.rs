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

use gax::pga3d::{Direction, Line, Motor, Plane, Point};
use gax_colour::light;
use gax_numga_examples::measure::roots;
use gax_numga_examples::{
    Anim, Axes, Camera, Canvas, Lens, Light, Point2, Rect, backdrop, caption, palette, run,
};

#[path = "../shared/mechanics_lie.rs"]
mod lie;

mod top {
    use super::*;
    pub use crate::lie::rigid::{G, Inertia, InertiaInv, M, P, R};

    /// A quadric: each point to its polar plane.
    pub type Quadric = Plane<(Point,), f64>;
    /// A plane.
    pub type Pl = Plane<(), f64>;
    /// A direction: an ideal point, a point's weightless part (numga's `Direction` type).
    pub type D = Direction<(), f64>;

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

    /// The directions of the x, y and z axes.
    pub fn directions() -> [P; 3] {
        [
            Point::direction(1.0, 0.0, 0.0),
            Point::direction(0.0, 1.0, 0.0),
            Point::direction(0.0, 0.0, 1.0),
        ]
    }

    /// The Euclidean inner product of two directions: of the planes through the origin they are
    /// normal to.
    pub fn dot(a: D, b: D) -> f64 {
        (a.dual() | b.dual()).s()
    }

    /// The Euclidean length of a direction: the norm of the planes normal to it.
    pub fn length(d: D) -> f64 {
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
            q += through * (through & slot) / (s * s);
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

    /// A quadric moved by a motor: its input and its output.
    pub fn moved(motor: M, quadric: Quadric) -> Quadric {
        motor >> quadric.of(motor << Point::slot())
    }

    /// A quadric's centre: the pole of the plane at infinity, unitized.
    pub fn centre(quadric: Quadric) -> P {
        quadric.solve(w()).unitized()
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

    // --- contact -------------------------------------------------------------------------------

    /// The ground's unit normal at a point, toward the air, where its form grows. A point's
    /// polar plane under the ground quadric is the gradient of the ground's form there; its
    /// direction divided by its length is the normal, on the ground or off it.
    pub fn ground_normal(ground: Quadric, p: P) -> D {
        let g = ground.of(p);
        g.dual().cast::<Direction>() / g.norm()
    }

    /// The quadric's point furthest against the normal: the pole of its tangent plane on that
    /// side. The plane through the quadric's centre normal to the direction has its pole at
    /// infinity, the direction conjugate to it, along which the centre reaches the planes
    /// tangent to the quadric parallel to it; scaled to reach the surface, it leads from the
    /// centre to the lowest point.
    pub fn lowest_point(surface: Quadric, normal: D) -> P {
        // Planes to their poles.
        let dual = surface.inverse();
        let pole = dual.of(w());
        let centre = pole / (w() & pole).s();
        // The plane through the centre, normal to the direction, and its pole, a direction.
        let plane = normal.dual() - w() * (normal.dual() & centre).s();
        let conjugate = dual.of(plane);
        // Along the conjugate the form is `t² Q(conjugate) + Q(centre)` (no cross term, by
        // conjugacy), with `Q(conjugate) = plane & conjugate` and `Q(centre) = 1 / (w & pole)`:
        // its roots are a pair centred on the centre, and the reach is their radius (none, not
        // a number, if the quadric is not closed).
        let a = (plane & conjugate).s();
        let reach = roots(a, 0.0, (w() & pole).s().recip()).map_or(f64::NAN, |(_, r)| r);
        centre + conjugate * reach
    }

    /// The direction less its part along the ground's normal: its part in the ground's plane.
    pub fn along_ground(d: D, normal: D) -> D {
        d - normal * dot(d, normal)
    }

    /// The body's twist per unit forque along a line, and its compliance along that line.
    pub fn compliance(motor: M, inertia_inv: InertiaInv, line: Line<(), f64>) -> (R, f64) {
        let body = motor << line;
        let step = inertia_inv.of(body);
        (step, (step & body).s())
    }

    /// Carry the body by the summed twists of the forques (renormalized against drift).
    pub fn move_by(motor: M, twist: R) -> M {
        motor.mul_renormalized((twist * -0.5).exp())
    }

    /// The two principal curvatures at a point on the surface, in order; convex surfaces curve
    /// positively (numga's `geometry/surface_curvature`). The Hessian form `Q(p) & p'` pulled
    /// back on both sides through the projector of directions onto the tangent plane is the
    /// second fundamental form up to the length of the gradient; against the Euclidean metric
    /// on directions its eigenvalues are the principal curvatures and a zero, the normal's.
    pub fn principal(surface: Quadric, p: P) -> [f64; 2] {
        let tangent = surface.of(p);
        let normal = tangent.dual().cast::<Direction>();
        // Directions, projected along the normal onto the tangent plane.
        let d = Direction::<(), f64>::slot();
        let project = d.cast::<Point>() - normal * ((tangent & d) / (tangent & normal).s());
        let hessian = surface & Point::slot();
        let second = hessian.of_both(project, project) / length(normal);
        let (values, _) = second.eigh_with(d.dual() | d.dual());
        // The two values that are not the zero.
        let mut v = values;
        v.sort_by(|a, b| a.abs().total_cmp(&b.abs()));
        [v[1].min(v[2]), v[1].max(v[2])]
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
        use crate::lie::Lie;
        let [stat, dynamic] = friction;
        // The parts in the world. Each part's lowest point against the ground's normal at its
        // centre is its contact.
        let placed: Vec<Quadric> = parts.iter().map(|q| moved(motor, *q)).collect();
        let contact: Vec<P> = placed
            .iter()
            .map(|q| lowest_point(*q, ground_normal(ground, centre(*q))))
            .collect();
        let normal: Vec<D> = contact.iter().map(|c| ground_normal(ground, *c)).collect();
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
            let shift = contact[i] - (before >> (motor << contact[i]));
            let slid = along_ground(shift.cast::<Direction>(), normal[i]);
            let back = length(slid);
            let line = contact[i] & (slid / -(back + 1e-12));
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
            #[allow(clippy::disallowed_methods)]
            // Hertz's law of contact, the patch radius sqrt(R δ)
            let patch = ((2.0 / (k1 + k2)).abs() * indentation).sqrt();
            let (step, give) = compliance(motor, inertia_inv, couple);
            twist += step * clamp(-turned / give, pressed[i] * stat * patch);
        }
        motor = move_by(motor, twist);

        // Dynamic friction: the rate the corrected step implies, less the contacts' sliding
        // velocity, with at most the dynamic coefficient times the normal impulse.
        let rate = G::rate_between(before, motor, dt);
        let mut twist = Line::zero();
        for i in 0..n {
            let velocity = contact[i].commutator(motor >> rate).cast::<Direction>();
            let sliding = along_ground(velocity, normal[i]);
            let speed = length(sliding);
            let line = contact[i] & (sliding / -(speed + 1e-12));
            let (step, give) = compliance(motor, inertia_inv, line);
            twist += step * clamp(speed / give, pressed[i] * dynamic / dt);
        }
        (motor, rate + twist)
    }
}

mod scenarios {
    use super::top::*;
    use super::*;
    use crate::lie::Lie;

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
    pub fn top(tip_radius: f64, tip_height: f64, disc_height: f64, disc_mass: f64) -> Top {
        let centres =
            [disc_height, disc_height + 0.16, tip_height].map(|z| Point::xyz(0.0, 0.0, z));
        let semi = [
            [0.25, 0.25, 0.04],
            [0.02, 0.02, 0.15],
            [tip_radius, tip_radius, tip_height],
        ];
        let mass = [disc_mass, 0.05, 0.15];
        let total: f64 = mass.iter().sum();
        // The centre of mass, and the parts moved to put it at the origin.
        let com = centres.iter().zip(mass).map(|(c, m)| *c * m).sum::<P>() / total;
        let centred = Motor::between(com, Point::xyz(0.0, 0.0, 0.0));
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
        // Tilt, then lift the centre of mass so that the tip, `height` below it, clears the
        // ground by 2 mm.
        let tilt = Motor::rotation_about(0.0, 1.0, 0.0, 0.15);
        let [_, _, tip] = (tilt >> Point::xyz(0.0, 0.0, -body.height)).to_euclidean();
        let mut motor = Motor::translation(0.0, 0.0, 0.002 - tip) * tilt;
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
            let (m, _) = G::explicit_rk4(motor, rate, body.inertia, body.inertia_inv, dt, &forces);
            (motor, rate) = project_contacts(
                before,
                m.renormalize_fast(),
                body.inertia_inv,
                &body.parts,
                ground,
                [STATIC_FRICTION, setup.dynamic_friction],
                setup.indentation,
                dt,
            );
            if i % every == 0 {
                // The axis's angle from vertical: twice the size of the turn between them.
                let lean = Motor::rotation_between(up, motor >> up).log().norm() * 2.0;
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
    /// the distance, whose roots are a pair of points on the ray ([`roots`]); at the
    /// entering root the form falls through zero, the nearer root where it opens upwards
    /// (`a > 0`) and the further one where it opens downwards.
    pub fn hit(surface: Quadric, origin: P, heading: P) -> Option<(f64, P)> {
        let along = surface.of(heading);
        let a = (along & heading).s();
        let b = (along & origin).s();
        let c = (surface.of(origin) & origin).s();
        let (middle, radius) = roots(a, b, c)?;
        let distance = middle - radius * a.signum();
        (distance.is_finite() && distance > 0.0).then(|| (distance, origin + heading * distance))
    }

    /// The normal of the quadric at a point on it: the direction of its polar plane.
    pub fn normal_at(surface: Quadric, p: P) -> D {
        surface.of(p).dual().cast::<Direction>()
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
    let facing = |n: D| (dot(n, lamp) / (length(n) * length(lamp))).abs() as f32;
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

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let runs = runs();
    let screen = c.rect();
    let (w, h) = (screen.width(), screen.height());
    let u = h / 270.0;
    let k = ((f64::from(t) / (DT * EVERY as f64)) as usize).min(runs.samples[0].len() - 1);
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
        let row = Rect::new(x0, top, w, top + h * 0.44);
        let rect = row.inset(24.0 * u, 14.0 * u, 6.0 * u, 16.0 * u);
        let ax = Axes::new(rect, [0.0, SECONDS as f32], *range);
        ax.frame(c, title, if p == 1 { "TIME (S)" } else { "" }, "");
        for (v, samples) in runs.samples.iter().enumerate() {
            // The chart's points: (time, value).
            let pts: Vec<gax::pga2d::Point<(), f64>> = samples[..=k]
                .iter()
                .enumerate()
                .map(|(j, s)| {
                    let value = if p == 0 { s.tilt.to_degrees() } else { s.spin };
                    gax::pga2d::Point::xy(j as f64 * DT * EVERY as f64, value)
                })
                .collect();
            let width = if v == 0 { 1.8 } else { 1.1 };
            ax.polyline(c, &pts, width, palette::series(v));
        }
        let (low, high) = (Point2::xy(time, range[0]), Point2::xy(time, range[1]));
        ax.line(c, low, high, 0.8, palette::grid());
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
    use gax::pga3d::{Direction, Point};

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
        let up = Point::direction(0.0, 0.0, 1.0).cast::<Direction>();
        let low = lowest_point(sphere, up);
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
            gax_numga_examples::Rect::new(0.0, 0.0, 32.0, 32.0),
        );
        assert!(c.mean().luma() > 0.0);
    }

    /// The top stays on the ground, spinning: its tip neither sinks in nor flies off, it stays
    /// upright, and drag slows its spin.
    #[test]
    fn the_top_spins_on_the_ground() {
        let (body, ground, samples) = spin(BASE, 1.0, 1e-3, 10);
        // It starts tilted by 0.15 rad.
        assert!(close(samples[0].tilt, 0.15, 1e-3), "{}", samples[0].tilt);
        let last = samples[samples.len() - 1];
        assert!(last.tilt < 0.5, "{}", last.tilt);
        assert!(last.spin > 10.0 && last.spin < 18.0, "{}", last.spin);
        for s in &samples {
            let placed = moved(s.motor, body.parts[2]);
            let low = lowest_point(placed, ground_normal(ground, centre(placed)));
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
        assert!(c.mean().luma() > 0.0);
    }
}
