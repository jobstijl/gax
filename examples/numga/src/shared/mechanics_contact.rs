//! numga's `mechanics/spinning_top/core.py` contacts, shared by the tops that use them
//! (`mechanics_spinning_top`, `mechanics_tippe_top`): solid quadrics against a ground quadric,
//! a part's lowest point against the ground's normal, and the position-based correction of
//! numga's `project_contacts`, part after part (Gauss–Seidel): pressed out by its depth, held
//! against sliding when the full correction fits the static friction cone, then the rate the
//! corrected step implies slowed by dynamic friction, part by part.

use crate::shared::mechanics_lie::rigid::{F, G, InertiaInv, M, P, R};
use gax::pga3d::{Direction, Plane, Point};

/// A quadric: each point to its polar plane, its form `Q(p) & p` negative inside.
pub type Quadric = Plane<(Point,)>;

/// A solid ellipsoid about a centre, its axes square to the planes of a frame, with the given
/// semi-axes: the dyads of the frame's planes through the centre, less the dyad of the plane at
/// infinity.
pub fn ellipsoid_in(centre: P, frame: [Plane; 3], semi: [f64; 3]) -> Quadric {
    let slot = Point::slot();
    let mut q = -(Plane::infinity() * (Plane::infinity() & slot));
    for (a, s) in frame.into_iter().zip(semi) {
        let through = a - Plane::infinity() * (a & centre).s();
        q += through * (through & slot) / (s * s);
    }
    q
}

/// The Euclidean inner product of two directions: of the planes through the origin they are
/// normal to.
pub fn dot(a: Direction, b: Direction) -> f64 {
    (a.dual() | b.dual()).s()
}

/// The Euclidean length of a direction: the norm of the planes normal to it.
pub fn length(d: Direction) -> f64 {
    d.dual().norm()
}

/// The value with its magnitude held to the limit (numga's, branch free: the smaller of the
/// magnitude and the limit, over the magnitude).
pub fn clamp(value: f64, limit: f64) -> f64 {
    let magnitude = value.abs();
    let least = (magnitude + limit - (magnitude - limit).abs()) * 0.5;
    value * least / (magnitude + 1e-12)
}

/// The ground: solid below the paraboloid whose height is the curvature times the squared
/// distance from the z axis; negative inside. A bowl for positive curvature.
pub fn bowl(curvature: f64) -> Quadric {
    let slot = Point::slot();
    let [x, y, z] = [
        Plane::new(1.0, 0.0, 0.0, 0.0),
        Plane::new(0.0, 1.0, 0.0, 0.0),
        Plane::new(0.0, 0.0, 1.0, 0.0),
    ];
    (z * (Plane::infinity() & slot) + Plane::infinity() * (z & slot)) * 0.5
        - (x * (x & slot) + y * (y & slot)) * curvature
}

/// A quadric moved by a motor: its input and its output.
pub fn moved(motor: M, quadric: Quadric) -> Quadric {
    motor >> quadric.of(motor << Point::slot())
}

/// A quadric's centre: the pole of the plane at infinity, unitized.
pub fn centre(quadric: Quadric) -> P {
    quadric.solve(Plane::infinity()).unitized()
}

// --- contact -------------------------------------------------------------------------------

/// The ground's unit normal at a point, toward the air, where its form grows. A point's
/// polar plane under the ground quadric is the gradient of the ground's form there; its
/// direction divided by its length is the normal, on the ground or off it.
pub fn ground_normal(ground: Quadric, p: P) -> Direction {
    let g = ground.of(p);
    g.dual().cast::<Direction>() / g.norm()
}

/// The quadric's point furthest against the normal: the pole of its tangent plane on that
/// side. The plane through the quadric's centre normal to the direction has its pole at
/// infinity, the direction conjugate to it, along which the centre reaches the planes
/// tangent to the quadric parallel to it; scaled to reach the surface, it leads from the
/// centre to the lowest point.
pub fn lowest_point(surface: Quadric, normal: Direction) -> P {
    // Planes to their poles.
    let dual = surface.inverse();
    let pole = dual.of(Plane::infinity());
    let centre = pole / (Plane::infinity() & pole).s();
    // The plane through the centre, normal to the direction, and its pole, a direction.
    let plane = normal.dual() - Plane::infinity() * (normal.dual() & centre).s();
    let conjugate = dual.of(plane);
    // Along the conjugate the form is `t² Q(conjugate) + Q(centre)` (no cross term, by
    // conjugacy), with `Q(conjugate) = plane & conjugate` and `Q(centre) = 1 / (w & pole)`:
    // the surface lies at the `t` where it vanishes.
    #[allow(clippy::disallowed_methods)] // the reach along the conjugate, from its square
    let reach = (-(plane & conjugate).s() * (Plane::infinity() & pole).s()).sqrt();
    centre + conjugate / reach
}

/// The direction less its part along the ground's normal: its part in the ground's plane.
pub fn along_ground(d: Direction, normal: Direction) -> Direction {
    d - normal * dot(d, normal)
}

/// The body's twist per unit forque along a line, and its compliance along that line.
pub fn compliance(motor: M, inertia_inv: InertiaInv, line: F) -> (R, f64) {
    let body = motor << line;
    let step = inertia_inv.of(body);
    (step, (step & body).s())
}

/// Carry the body by the summed twists of the forques (renormalized against drift).
pub fn move_by(motor: M, twist: R) -> M {
    motor.mul_renormalized((twist * -0.5).exp())
}

/// A part placed by the motor touching the ground: its contact, the ground's normal there,
/// and how far it lies below the ground (to first order in the depth; none in the air).
pub fn touching(motor: M, part: Quadric, ground: Quadric) -> (P, Direction, f64) {
    let placed = moved(motor, part);
    let contact = lowest_point(placed, ground_normal(ground, centre(placed)));
    let g = ground.of(contact);
    let depth = (-(g & contact).s() / (2.0 * g.norm())).max(0.0);
    (contact, ground_normal(ground, contact), depth)
}

/// Contact and friction, as numga's `project_contacts`: the predicted pose corrected one
/// part after another, so that each sees the corrections before it: pressed out by its
/// depth, then held against sliding if its full correction fits the static friction limit,
/// else left to slip; then the rate the corrected step implies, slowed by each part's
/// dynamic friction in turn, up to the dynamic coefficient times its pressing impulse.
#[allow(clippy::too_many_arguments)]
pub fn project_contacts(
    before: M,
    mut motor: M,
    inertia_inv: InertiaInv,
    parts: &[Quadric],
    ground: Quadric,
    friction: [f64; 2],
    dt: f64,
) -> (M, R) {
    use crate::shared::mechanics_lie::Lie;
    let [stat, dynamic] = friction;
    let mut pressures = Vec::with_capacity(parts.len());
    for part in parts {
        let (contact, normal, depth) = touching(motor, *part, ground);
        // Press out along the normal line through the contact, by its depth.
        let (step, give) = compliance(motor, inertia_inv, contact & normal);
        let press = depth / give;
        motor = move_by(motor, step * press);
        // Sticking is possible only when its full correction fits the static friction
        // limit: back toward where the contact's material point was, in the ground's plane.
        let shift = contact - (before >> (motor << contact));
        let slip = along_ground(shift.cast::<Direction>(), normal);
        let slip_length = length(slip);
        let line = contact & (slip / -(slip_length + 1e-12));
        let (step, give) = compliance(motor, inertia_inv, line);
        let stick = slip_length / (give + 1e-12);
        if stick <= press * stat {
            motor = move_by(motor, step * stick);
        }
        pressures.push(press);
    }
    let mut rate = G::rate_between(before, motor, dt);
    for (part, press) in parts.iter().zip(pressures) {
        let (contact, normal, _) = touching(motor, *part, ground);
        // The contact's remaining slip, held by at most the dynamic coefficient times its
        // pressing impulse.
        let velocity = contact.commutator(motor >> rate).cast::<Direction>();
        let sliding = along_ground(velocity, normal);
        let speed = length(sliding);
        let line = contact & (sliding / -(speed + 1e-12));
        let (step, give) = compliance(motor, inertia_inv, line);
        rate += step * clamp(speed / (give + 1e-12), press * dynamic / dt);
    }
    (motor, rate)
}
