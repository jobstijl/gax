//! numga's `kerr/core.py`, the geometry and the rays, in gauge theory gravity: the Kerr–Schild
//! field of a rotating hole, the position gauge field that carries a local frame's directions
//! to the Kerr geometry's, the rays' Hamiltonian motion through it, the rotation gauge field
//! that carries a camera's screen along them, and the camera's rays.
//!
//! Events and the directions between them are antivectors, intrinsic trivectors (ADR-051): the
//! lab's time is `T = −t*`, one unit along each axis `X = x*`, `Y = y*`, `Z = z*` (the Hodge
//! dual, numga's `dual`). A momentum is a vector, extrinsic, which pairs with a direction to a
//! number, `p ∨ d`; so do the slices `t x y z` that read coordinates, `x ∨ X = 1`.

use core::ops::{Add, Mul};
use gax::sta::{Bivector, Even, Scalar, Vector, intrinsic};

/// An event, or a direction between events: numga's antivector.
pub type D = intrinsic::Trivector;
/// A momentum, a gradient or a slice: numga's vector, which reads directions.
pub type V = Vector;
pub type B = Bivector;
/// The hole's spin per unit mass: intrinsic, the patch of the plane it turns in.
pub type Spin = intrinsic::Bivector;
/// A linear map on directions, numga's `Antivector <- Antivector`: the gauge field, the null
/// direction's derivative.
pub type DirectionMap = intrinsic::Trivector<(D,)>;
/// The camera's two screen axes, a map from screen directions to directions (numga's
/// `Antivector <- Screen`). gax's STA has no kind for numga's `Screen`, the directions `X` and
/// `Y`, so the map takes every direction and reads only `x ∨ d` and `y ∨ d`.
pub type ScreenFrame = DirectionMap;
/// A unit rotor of spacetime: the camera's orientation, its boost.
pub type Rotor = gax::Unit<Even>;

/// The slices `t x y z`, which read an event's coordinates.
pub fn t() -> V {
    Vector::new(0.0, 0.0, 0.0, 1.0)
}
pub fn x() -> V {
    Vector::new(1.0, 0.0, 0.0, 0.0)
}
pub fn y() -> V {
    Vector::new(0.0, 1.0, 0.0, 0.0)
}
pub fn z() -> V {
    Vector::new(0.0, 0.0, 1.0, 0.0)
}

/// One unit along the lab's time, the direction across its slice of space `t`: `−t*`.
pub fn time() -> D {
    -t().hodge()
}
/// One unit along `x`, `y` and `z`: their duals.
pub fn along_x() -> D {
    x().hodge()
}
pub fn along_y() -> D {
    y().hodge()
}
pub fn along_z() -> D {
    z().hodge()
}

/// The event, or direction, with coordinates `t, x, y, z`.
#[cfg_attr(not(test), allow(dead_code))] // numga's core, which the tests check
pub fn event(t: f64, x: f64, y: f64, z: f64) -> D {
    time() * t + along_x() * x + along_y() * y + along_z() * z
}

/// A slice's reading of a direction, `slice ∨ d`.
pub fn read(slice: V, d: D) -> f64 {
    (slice & d).s()
}

/// The equatorial plane `x ∧ y`: the plane the hole and the disk's gas turn in.
pub fn xy() -> B {
    x() ^ y()
}

/// numga's `squared`: a direction times itself, a scalar. A spatial direction squares to its
/// squared length, the lab's time to minus one.
pub fn squared(d: D) -> f64 {
    d.dot(d).s()
}

/// The part of a direction within the lab's slice of space, `d − T (t ∨ d)`.
pub fn space(d: D) -> D {
    d - time() * read(t(), d)
}

/// The same as a map.
fn space_map() -> DirectionMap {
    D::slot() - time() * (t() & D::slot())
}

/// A rotating black hole: its mass, and its spin per unit mass, of magnitude J/M.
#[derive(Clone, Copy, Debug)]
pub struct Hole {
    pub spin: Spin,
    pub mass: f64,
}

impl Hole {
    /// A hole spinning by `spin` in the plane `x ∧ y`, from `x` towards `y`.
    pub fn about_z(spin: f64, mass: f64) -> Hole {
        Hole {
            spin: xy().hodge() * spin,
            mass,
        }
    }

    /// The plane the hole turns in, whose dual is its spin (numga's `dual_inverse`: the Hodge
    /// dual squares to minus one on STA's bivectors).
    pub fn turning(self) -> B {
        -self.spin.hodge()
    }

    /// The spin squared, `a²`.
    pub fn spin_squared(self) -> f64 {
        self.spin.dot(self.spin).s()
    }

    /// The outer horizon's oblate radius, `M + √(M² − a²)`.
    #[allow(clippy::disallowed_methods)] // the Kerr horizon: a physical law
    pub fn horizon(self) -> f64 {
        self.mass + (self.mass * self.mass - self.spin_squared()).sqrt()
    }

    /// The spin counted along the unit plane `orbit`: positive when the orbit turns with
    /// the hole.
    pub fn along(self, orbit: B) -> f64 {
        (orbit & self.spin).s()
    }
}

/// The Kerr–Schild geometry at an event: the oblate radius, the profile `H` that weighs the
/// null dyad, the null direction `l`, the profile's gradient (a vector, which a small step
/// reads as the profile's change), the null direction's derivative (a map from a step to the
/// change it makes) and the curl of the null covector.
#[derive(Clone, Copy, Debug)]
pub struct Field {
    pub radius: f64,
    pub profile: f64,
    pub null: D,
    pub profile_gradient: V,
    pub null_gradient: DirectionMap,
    pub null_curl: B,
}

impl Field {
    /// The null direction's plane, the light cone's tangent plane along it, which contains the
    /// direction: minus its dual.
    pub fn null_plane(&self) -> V {
        -self.null.hodge()
    }
}

/// The Kerr profile, null direction and their derivatives at an event.
#[allow(clippy::disallowed_methods)] // the oblate radius, a quadratic's root: the Kerr–Schild radius
pub fn field(position: D, hole: Hole) -> Field {
    let spatial = space(position);
    let turning = hole.turning();
    // The volume the position spans with the turning plane: the plane times the position's
    // height off it, held by its complement along time. Wedged with the plane it is that
    // height along the spin, times the spin squared; the plane enters twice, so its sense of
    // turning drops out.
    let volume = spatial.anticommutator(turning);
    let height = volume ^ turning;
    let volume_squared = volume.dot(volume).s();
    let spin_squared = hole.spin_squared();
    let offset = squared(spatial) - spin_squared;
    let discriminant = (offset * offset - 4.0 * volume_squared).sqrt();
    let radius_squared = (offset + discriminant) / 2.0;
    let radius = radius_squared.sqrt();
    let scale = (radius_squared + spin_squared).recip();
    // Paired with a small step, how far the step changes the oblate radius.
    let radius_gradient =
        (spatial * radius_squared - height).hodge() * (radius * discriminant).recip();
    // The hole turns as `exp(−½ angle turning)` does, and drags the frame with it: the outgoing
    // direction twists against the turning, by the position's commutator with the plane.
    let direction =
        (spatial * radius - spatial.commutator(turning) - height * radius.recip()) * scale;
    let lever = spatial - direction * (2.0 * radius) + height * radius_squared.recip();
    let space = space_map();
    let direction_gradient: DirectionMap = (lever * (radius_gradient & D::slot()) + space * radius
        - space.commutator(turning)
        - (space.anticommutator(turning) ^ turning) * radius.recip())
        * scale;
    let denominator = radius_squared * radius_squared - volume_squared;
    let profile = hole.mass * radius.powi(3) / denominator;
    let profile_gradient = (radius_gradient * (3.0 / radius)
        - (radius_gradient * (4.0 * radius.powi(3)) - height.hodge() * 2.0) * denominator.recip())
        * profile;
    // The curl of the null covector, `−Σ eᵏ ∧ (D(Eₖ))*` over the directions `Eₖ` and the slices
    // `eᵏ` that read them, piece by piece: the rank-one piece gives its two vectors wedged, the
    // twist by the plane twice the plane, the rest none.
    let null_curl = ((radius_gradient ^ lever.hodge()) - turning * 2.0) * scale;
    Field {
        radius,
        profile,
        null: time() - direction,
        profile_gradient,
        null_gradient: -direction_gradient,
        null_curl,
    }
}

/// The null dyad, weighted by the profile: the null direction, read by its null plane,
/// `d ↦ H l (l♭ ∨ d)`. It squares to zero.
pub fn dyad(geometry: &Field) -> DirectionMap {
    geometry.null * (geometry.null_plane() & D::slot()) * geometry.profile
}

/// The position gauge field: the identity plus the weighted null dyad, from a direction of
/// the local frame to one of the Kerr geometry.
pub fn gauge(geometry: &Field) -> DirectionMap {
    D::slot() + dyad(geometry)
}

/// The gauge field's inverse: the dyad squares to zero, so the identity less the dyad.
pub fn gauge_inverse(geometry: &Field) -> DirectionMap {
    D::slot() - dyad(geometry)
}

/// A momentum's direction in the local frame: raised flat, `−p*`, and carried through the
/// gauge field. Minus half its flat square is the ray's Hamiltonian.
pub fn local(geometry: &Field, momentum: V) -> D {
    gauge(geometry).of(-momentum.hodge())
}

/// The squared proper length of a timelike direction: minus the flat square of the direction
/// carried back into the local frame.
pub fn proper_squared(geometry: &Field, direction: D) -> f64 {
    -squared(gauge_inverse(geometry).of(direction))
}

/// The ray Hamiltonian, positive for a timelike momentum in the physical `+−−−` convention.
#[cfg_attr(not(test), allow(dead_code))] // numga's core, which the tests check
pub fn hamiltonian(geometry: &Field, momentum: V) -> f64 {
    -squared(local(geometry, momentum)) / 2.0
}

/// The gradient of the momentum's pairing with the null direction, `p ∨ l`: the momentum
/// pulled back through the null direction's derivative, its adjugate.
pub fn overlap_gradient(geometry: &Field, momentum: V) -> V {
    geometry.null_gradient.adjugate().of(momentum)
}

/// What a ray carries: its event and its momentum. It is also the form of its rate (the
/// tangent and the momentum's rate), so Runge–Kutta adds and scales it.
#[derive(Clone, Copy, Debug)]
pub struct Ray {
    pub event: D,
    pub momentum: V,
}

/// A ray with its camera screen, transported along.
#[derive(Clone, Copy, Debug)]
pub struct PolarizedRay {
    pub ray: Ray,
    pub screen: ScreenFrame,
}

impl Add for Ray {
    type Output = Ray;
    fn add(self, other: Ray) -> Ray {
        Ray {
            event: self.event + other.event,
            momentum: self.momentum + other.momentum,
        }
    }
}

impl Mul<f64> for Ray {
    type Output = Ray;
    fn mul(self, k: f64) -> Ray {
        Ray {
            event: self.event * k,
            momentum: self.momentum * k,
        }
    }
}

impl Add for PolarizedRay {
    type Output = PolarizedRay;
    fn add(self, other: PolarizedRay) -> PolarizedRay {
        PolarizedRay {
            ray: self.ray + other.ray,
            screen: self.screen + other.screen,
        }
    }
}

impl Mul<f64> for PolarizedRay {
    type Output = PolarizedRay;
    fn mul(self, k: f64) -> PolarizedRay {
        PolarizedRay {
            ray: self.ray * k,
            screen: self.screen * k,
        }
    }
}

/// A state the integrator steps and the disk reads: a ray, with whatever it carries along.
pub trait RayState: Copy + Add<Output = Self> + Mul<f64, Output = Self> {
    fn ray(&self) -> Ray;
}

impl RayState for Ray {
    fn ray(&self) -> Ray {
        *self
    }
}

impl RayState for PolarizedRay {
    fn ray(&self) -> Ray {
        self.ray
    }
}

/// The two Hamiltonian derivatives at a field, as a ray's rate: the tangent, a direction, and
/// minus the position gradient, a vector.
pub fn motion(geometry: &Field, momentum: V) -> Ray {
    let overlap = (momentum & geometry.null).s();
    // Minus half the flat square of the momentum's local direction is the Hamiltonian; its
    // momentum derivative is that direction carried through the gauge field once more.
    let tangent = gauge(geometry).of(local(geometry, momentum));
    // The gradient of `H (p ∨ l)²`, by the product rule.
    let gradient = geometry.profile_gradient * (overlap * overlap)
        + overlap_gradient(geometry, momentum) * (2.0 * geometry.profile * overlap);
    Ray {
        event: tangent,
        momentum: -gradient,
    }
}

/// The ray's rate in its affine parameter.
pub fn rates(ray: Ray, hole: Hole) -> Ray {
    motion(&field(ray.event, hole), ray.momentum)
}

/// How fast coordinate time runs back along a past-directed tangent, `−t ∨ v`.
pub fn clock(tangent: D) -> f64 {
    -read(t(), tangent)
}

/// Past-directed motion per unit of decreasing coordinate time.
#[cfg_attr(not(test), allow(dead_code))] // numga's core, which the tests check
pub fn camera_rates(ray: Ray, hole: Hole) -> Ray {
    let rate = rates(ray, hole);
    rate * clock(rate.event).recip()
}

/// The rotation gauge field: the plane the local frame turns or boosts in, per unit step
/// along a direction.
pub fn rotation(geometry: &Field, direction: D) -> B {
    // The null plane's reading of the direction.
    let null = geometry.null_plane();
    let along = (null & direction).s();
    // The gradient of `H along`.
    let turning = geometry.profile_gradient * along
        + overlap_gradient(geometry, -direction.hodge()) * geometry.profile;
    -(turning ^ null) - geometry.null_curl * (geometry.profile * along)
}

/// Hamiltonian ray motion and the parallel transport of its camera screen, per unit of
/// decreasing coordinate time.
pub fn polarized_rates(state: PolarizedRay, hole: Hole) -> PolarizedRay {
    let geometry = field(state.ray.event, hole);
    let rate = PolarizedRay {
        ray: motion(&geometry, state.ray.momentum),
        screen: -rotation(&geometry, local(&geometry, state.ray.momentum)).commutator(state.screen),
    };
    rate * clock(rate.ray.event).recip()
}

/// One fourth-order step of a state along its rates.
pub fn runge_kutta<S: Copy + Add<Output = S> + Mul<f64, Output = S>>(
    rates: impl Fn(S) -> S,
    state: S,
    step: f64,
) -> S {
    let first = rates(state);
    let second = rates(state + first * (step / 2.0));
    let third = rates(state + second * (step / 2.0));
    let fourth = rates(state + third * step);
    state + (first + (second + third) * 2.0 + fourth) * (step / 6.0)
}

/// A future null momentum with the given unit spatial direction in the Kerr–Schild frame.
pub fn initial_momentum(position: D, direction: D, hole: Hole) -> V {
    let geometry = field(position, hole);
    let local_null = time() + direction;
    // Carried back out of the local frame and lowered flat.
    let momentum = -gauge_inverse(&geometry).of(local_null).hodge();
    momentum * (momentum & time()).s().recip()
}

/// A ray's states at fourth-order steps of its affine parameter, the first its start. The
/// step shrinks only inside the horizon, slowing captured rays towards a surface at
/// `stop_fraction` of it: this changes the sampling, not the path, and keeps every stage
/// away from the singularity.
pub fn trace(
    start: Ray,
    hole: Hole,
    step_size: f64,
    steps: usize,
    stop_fraction: f64,
    dynamics: fn(Ray, Hole) -> Ray,
) -> impl Iterator<Item = Ray> {
    let horizon = hole.horizon();
    std::iter::successors(Some(start), move |ray| {
        let radius = field(ray.event, hole).radius;
        let clock = ((radius / horizon - stop_fraction) / (1.0 - stop_fraction)).clamp(0.0, 1.0);
        Some(runge_kutta(|r| dynamics(r, hole), *ray, step_size * clock))
    })
    .take(steps + 1)
}

/// The camera's own axes in its rest frame, before it is turned: it looks along `−Z`, with its
/// right `+Y` and its up `−X`.
pub fn camera_forward() -> D {
    -along_z()
}
pub fn camera_right() -> D {
    along_y()
}
pub fn camera_up() -> D {
    -along_x()
}

/// A coordinate-static camera's rest frame in the Kerr–Schild frame, outside the static
/// limit: the boost from the lab's time to the static observer carried into the local frame.
#[allow(clippy::disallowed_methods)] // a proper length: the square root of its square
pub fn camera_boost(geometry: &Field) -> Rotor {
    let observer =
        gauge_inverse(geometry).of(time()) * proper_squared(geometry, time()).sqrt().recip();
    (Scalar::new(1.0).cast::<Even>() - observer * time()).unit()
}

/// A direction scaled to unit length: a spatial direction squares to its length squared.
#[allow(clippy::disallowed_methods)] // a length: the square root of its square
pub fn unit_length(d: D) -> D {
    d * squared(d).sqrt().recip()
}

/// Pixel-centre rays of a coordinate-static camera at `eye`, row by row, looking along its
/// orientation's `−Z`, with right `+Y` and up `−X`; `half_view` is the tangent of half the
/// vertical field of view.
pub fn camera_rays(
    eye: D,
    orientation: Rotor,
    [width, height]: [usize; 2],
    half_view: f64,
    hole: Hole,
) -> Vec<Ray> {
    let centre = |i: usize, n: usize| (i as f64 + 0.5) / n as f64 * 2.0 - 1.0;
    let aspect = width as f64 / height as f64;
    let geometry = field(eye, hole);
    let boost = camera_boost(&geometry);
    let inverse = gauge_inverse(&geometry);
    (0..width * height)
        .map(|k| {
            let across = centre(k % width, width) * half_view * aspect;
            let up = -centre(k / width, height) * half_view;
            let look =
                orientation >> (camera_forward() + camera_right() * across + camera_up() * up);
            let direction = unit_length(look);
            // The future photon arrives opposite the direction the camera looks; its rest
            // frame is boosted into the Kerr–Schild frame before its heading is lowered into a
            // coordinate momentum.
            let heading = boost >> (time() - direction);
            let momentum = -inverse.of(heading).hodge();
            Ray {
                event: eye,
                momentum: momentum * -(momentum & time()).s().recip(),
            }
        })
        .collect()
}

/// The screen frame with the given directions for the screen's `x` and `y`.
pub fn screen_frame(right: D, up: D) -> ScreenFrame {
    right * (x() & D::slot()) + up * (y() & D::slot())
}

/// Orthonormal transverse camera axes, with horizontal `x` and vertical `y` screen inputs.
#[allow(clippy::disallowed_methods)] // a length: the square root of its square
pub fn camera_screen(ray: Ray, orientation: Rotor, hole: Hole) -> ScreenFrame {
    let geometry = field(ray.event, hole);
    let boost = camera_boost(&geometry);
    let heading = boost << local(&geometry, ray.momentum);
    let direction = space(heading) * -read(t(), heading).recip();
    // The turn carrying the camera's forward onto the direction it looks to see the ray: the
    // reflection in the forward, then in the direction halfway between the two.
    let forward = orientation >> camera_forward();
    let halfway = direction + forward;
    let turn = (halfway * forward) * squared(halfway).sqrt().recip();
    // The camera's own right and up, turned with it.
    let frame = (boost.into_inner() * turn * orientation.into_inner()).unit();
    screen_frame(frame >> camera_right(), frame >> camera_up())
}

/// One integration step of a ray: its state at the start and at the end, and its oblate
/// radius at the end.
#[derive(Clone, Copy, Debug)]
pub struct Step<S> {
    pub start: S,
    pub end: S,
    pub radius: f64,
}

impl<S: RayState> Step<S> {
    /// Whether the step crosses the central plane `normal`, the slice of height above it.
    pub fn crosses(&self, normal: V) -> bool {
        read(normal, self.start.ray().event) * read(normal, self.end.ray().event) <= 0.0
    }
}

/// Fourth-order steps of a camera ray, each a fraction of its oblate radius, at most `steps`
/// of them; the consumer stops when the ray is resolved. The state is a ray, with its screen
/// when `dynamics` transports one.
pub fn camera_trace<S: RayState>(
    start: S,
    hole: Hole,
    step_fraction: f64,
    steps: usize,
    dynamics: fn(S, Hole) -> S,
) -> impl Iterator<Item = Step<S>> {
    let mut state = start;
    let mut radius = field(start.ray().event, hole).radius;
    (0..steps).map(move |_| {
        let end = runge_kutta(|s| dynamics(s, hole), state, step_fraction * radius);
        let step = Step {
            start: state,
            end,
            radius: field(end.ray().event, hole).radius,
        };
        (state, radius) = (end, step.radius);
        step
    })
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::disk::{Disk, disk_crossing, resolve};
    use gax::ApproxEq;

    pub const ROUND_OFF: f64 = 1e-11;

    /// The directions along time and the axes, and the slices that read them.
    pub fn directions() -> [D; 4] {
        [time(), along_x(), along_y(), along_z()]
    }
    pub fn slices() -> [V; 4] {
        [t(), x(), y(), z()]
    }

    /// A map on directions as a matrix, `[out][in]`: each image of an axis read by each slice.
    pub fn matrix(map: DirectionMap) -> [[f64; 4]; 4] {
        slices().map(|s| directions().map(|d| read(s, map.of(d))))
    }

    /// The largest coefficient of a value or map.
    pub fn largest<E: ApproxEq + Mul<f64, Output = E> + Copy>(e: E) -> f64 {
        e.max_abs_diff(&(e * 0.0))
    }

    /// A hole whose spin is tilted off the equator's axis: `(0.7 xy + 0.2 yz)*`.
    pub fn tilted_hole() -> Hole {
        Hole {
            spin: (xy() * 0.7 + (y() ^ z()) * 0.2).hodge(),
            mass: 1.0,
        }
    }

    /// The momentum along the direction a turn in `plane` moves each event.
    fn angular(ray: &Ray, plane: B) -> f64 {
        (ray.momentum & plane.commutator(ray.event)).s()
    }

    #[test]
    fn slices_read_the_directions_as_coordinates() {
        for (i, s) in slices().into_iter().enumerate() {
            for (j, d) in directions().into_iter().enumerate() {
                assert_eq!(read(s, d), if i == j { 1.0 } else { 0.0 });
            }
        }
        // The turning plane's dual is the spin.
        let hole = tilted_hole();
        assert!(hole.turning().hodge().max_abs_diff(&hole.spin) < ROUND_OFF);
    }

    #[test]
    fn null_congruence_and_exact_inverse_gauge() {
        let positions = [
            event(0.0, 4.0, 2.0, 1.0),
            event(3.0, -3.0, 1.0, 2.0),
            event(-1.0, 2.0, -4.0, -1.0),
        ];
        let hole = tilted_hole();
        for position in positions {
            let geometry = field(position, hole);
            let momentum = initial_momentum(position, along_x(), hole);
            // The null direction follows straight lines in the background geometry; its dyad
            // squares to zero, so the inverse changes only the sign.
            assert!(squared(geometry.null).abs() < ROUND_OFF);
            assert!(largest(geometry.null_gradient.of(geometry.null)) < ROUND_OFF);
            // The closed-form curl is the derivative of the null covector: each slice wedged
            // with the dual of the null direction's change along the direction it reads, less.
            let curl: B = slices()
                .into_iter()
                .zip(directions())
                .map(|(s, d)| -(s ^ geometry.null_gradient.of(d).hodge()))
                .sum();
            assert!(geometry.null_curl.max_abs_diff(&curl) < ROUND_OFF);
            let round_trip = gauge(&geometry).of(gauge_inverse(&geometry));
            assert!(round_trip.max_abs_diff(&D::slot()) < ROUND_OFF);
            // A light ray's momentum, carried into the local frame, is flat null.
            assert!(squared(local(&geometry, momentum)).abs() < ROUND_OFF);
            assert!(((momentum & time()).s() - 1.0).abs() < ROUND_OFF);

            // With zero spin the profile is the mass divided by ordinary spatial distance.
            let schwarzschild = field(position, Hole::about_z(0.0, hole.mass));
            let radius = space(position).norm();
            assert!((schwarzschild.radius - radius).abs() < ROUND_OFF);
            assert!((schwarzschild.profile - hole.mass / radius).abs() < ROUND_OFF);
        }
    }

    #[test]
    fn directional_derivatives_drive_hamiltonian_rates() {
        let position = event(1.0, 4.0, 2.0, 1.0);
        let momentum = Vector::new(1.0, 0.8, -0.3, 0.2);
        let hole = tilted_hole();
        let increment = 1e-4;
        let geometry = field(position, hole);
        let rate = rates(
            Ray {
                event: position,
                momentum,
            },
            hole,
        );
        let difference = |ahead: f64, behind: f64| (ahead - behind) / (2.0 * increment);
        let hamiltonian =
            |g: &Field, p: V| -(p & p.hodge()).s() / 2.0 + g.profile * (p & g.null).s().powi(2);
        // These tolerances measure centred-difference accuracy, not round-off alone.
        for (direction, slice) in directions().into_iter().zip(slices()) {
            let ahead = field(position + direction * increment, hole);
            let behind = field(position - direction * increment, hole);
            let profile_change = difference(ahead.profile, behind.profile);
            let along = (geometry.profile_gradient & direction).s();
            assert!((profile_change - along).abs() < 2e-9);
            let null_change = (ahead.null - behind.null) * (1.0 / (2.0 * increment));
            assert!(null_change.max_abs_diff(&geometry.null_gradient.of(direction)) < 2e-9);
            let force_change = difference(
                hamiltonian(&ahead, momentum),
                hamiltonian(&behind, momentum),
            );
            assert!((force_change + (rate.momentum & direction).s()).abs() < 2e-9);
            let velocity_change = difference(
                hamiltonian(&geometry, momentum + slice * increment),
                hamiltonian(&geometry, momentum - slice * increment),
            );
            assert!((velocity_change - read(slice, rate.event)).abs() < 2e-9);
        }
    }

    #[test]
    fn rays_conserve_their_invariants_and_spin_reversal_mirrors_the_paths() {
        let mass = 1.0;
        // The mirror across the plane y = 0, on events and on covectors; it reverses the spin.
        let mirror = |d: D| d - along_y() * (2.0 * read(y(), d));
        let covector_mirror = |p: V| p - y() * (2.0 * (p & along_y()).s());
        let (step_size, steps, stop_fraction) = (0.04, 250, 0.8);
        let trace_from = |start: D, spin: f64| {
            let hole = Hole::about_z(spin, mass);
            let ray = Ray {
                event: start,
                momentum: initial_momentum(start, along_x(), hole),
            };
            trace(ray, hole, step_size, steps, stop_fraction, rates).collect::<Vec<_>>()
        };
        for offset in [-7.0, 2.0, 7.0] {
            let start = along_x() * -8.0 + along_y() * offset;
            let rays = trace_from(start, 0.7);
            let mirrored = trace_from(mirror(start), -0.7);
            for (spin, path) in [(0.7, &rays), (-0.7, &mirrored)] {
                let hole = Hole::about_z(spin, mass);
                let (energy, angular_momentum) = (
                    (path[0].momentum & time()).s(),
                    angular(&path[0], hole.turning()),
                );
                // Null propagation and axial angular momentum constrain the integrated
                // dynamics; their tolerances measure the accuracy of the finite RK4 steps.
                for ray in path {
                    let h = hamiltonian(&field(ray.event, hole), ray.momentum);
                    assert!(h.abs() < 2e-6, "{h}");
                    assert!((angular(ray, hole.turning()) - angular_momentum).abs() < 1e-7);
                    assert!(((ray.momentum & time()).s() - energy).abs() < ROUND_OFF);
                }
            }
            for (ray, image) in rays.iter().zip(&mirrored) {
                assert!(mirror(ray.event).max_abs_diff(&image.event) < ROUND_OFF);
                assert!(covector_mirror(ray.momentum).max_abs_diff(&image.momentum) < ROUND_OFF);
            }
            // The central ray crosses the horizon and stops safely inside it; the outer rays
            // escape.
            let hole = Hole::about_z(0.7, mass);
            let last = field(rays[steps].event, hole).radius;
            if offset == 2.0 {
                let horizon = hole.horizon();
                assert!(last < horizon && last > stop_fraction * horizon, "{last}");
            } else {
                assert!(last > 5.0, "{last}");
            }
        }
    }

    #[test]
    #[allow(clippy::disallowed_methods)] // the analytic reference it is checked against
    fn camera_and_disk_geometry_rotate_with_the_black_hole() {
        let hole = Hole::about_z(0.7, 1.0);
        let orientation = ((x() ^ z()) * 0.31).exp();
        let eye = (orientation >> along_z()) * 25.0;
        let rotation = ((y() ^ z()) * 0.23).exp();
        let (size, half_view) = ([5, 3], 0.4);
        let rays = camera_rays(eye, orientation, size, half_view, hole);
        let rotated_hole = Hole {
            spin: rotation >> hole.spin,
            ..hole
        };
        let rotated = camera_rays(
            rotation >> eye,
            rotation * orientation,
            size,
            half_view,
            rotated_hole,
        );
        for (ray, turned) in rays.iter().zip(&rotated) {
            // Camera rays point into the past and remain null in the curved metric.
            let tangent = rates(*ray, hole).event;
            assert!((ray.momentum & tangent).s().abs() < ROUND_OFF);
            assert!(read(t(), tangent) < 0.0);
            assert!(turned.event.max_abs_diff(&(rotation >> ray.event)) < ROUND_OFF);
            assert!(turned.momentum.max_abs_diff(&(rotation >> ray.momentum)) < ROUND_OFF);
        }
        // The central pixel looks at the hole, in the static camera's rest frame.
        let centre = rays[size[0] * size[1] / 2];
        let geometry = field(centre.event, hole);
        let heading = space(camera_boost(&geometry) << local(&geometry, centre.momentum));
        let toward_hole = unit_length(-eye);
        assert!(unit_length(heading).max_abs_diff(&toward_hole) < ROUND_OFF);

        // Intersect a tilted disk between events whose heights have opposite signs.
        let at = |e: D| Ray {
            event: rotation >> e,
            momentum: V::zero(),
        };
        let (crossing, radius) = disk_crossing(
            at(event(1.0, 3.0, 4.0, 2.0)),
            at(event(6.0, 8.0, -1.0, -3.0)),
            rotation >> z(),
            rotated_hole,
        );
        let expected = rotation >> event(3.0, 5.0, 2.0, 0.0);
        assert!(crossing.event.max_abs_diff(&expected) < ROUND_OFF);
        assert!((radius - (29.0f64 - 0.7 * 0.7).sqrt()).abs() < ROUND_OFF);
    }

    #[test]
    #[allow(clippy::disallowed_methods)] // the photon sphere's impact parameter, a reference
    fn backward_camera_rays_resolve_the_schwarzschild_capture_boundary() {
        let hole = Hole::about_z(0.0, 1.0);
        let eye = along_x() * 25.0;
        // Inside the photon sphere, where no ray turns back out.
        let (capture_radius, escape_radius) = (2.1, 40.0);
        // No disk: every ray keeps its light until it escapes or is captured.
        let clear = Disk::clear(z(), escape_radius);
        for slope in [0.14, 0.18, 0.20, 0.28] {
            let direction = unit_length(-along_x() + along_y() * slope);
            let start = Ray {
                event: eye,
                momentum: -initial_momentum(eye, -direction, hole),
            };
            let energy = (start.momentum & time()).s();
            let angular_momentum = angular(&start, xy());
            let expected_capture =
                (angular_momentum / energy).abs() < 3.0 * 3f64.sqrt() * hole.mass;
            let steps = camera_trace(start, hole, 0.1, 300, camera_rates);
            let (image, end) = resolve(steps, &clear, escape_radius, capture_radius, hole);
            let final_radius = field(end.event, hole).radius;
            let escaped = final_radius > escape_radius;
            // Every ray ends, escaped with all its light or captured.
            assert!(escaped || final_radius < capture_radius);
            assert_eq!(image.sky.is_some(), escaped);
            assert_eq!(!escaped, expected_capture, "slope {slope}");
            if escaped {
                // The finite-step check constrains the escaped rays after their full
                // encounter.
                assert!((end.momentum & rates(end, hole).event).s().abs() < 2e-5);
                assert!((angular(&end, xy()) - angular_momentum).abs() < 2e-5);
                assert!(((end.momentum & time()).s() - energy).abs() < ROUND_OFF);
            }
        }
    }

    /// The inverse of a 4 × 4 matrix, by Gauss–Jordan elimination with partial pivoting.
    pub fn inverse(m: [[f64; 4]; 4]) -> [[f64; 4]; 4] {
        let mut a = m;
        let mut inv = [[0.0; 4]; 4];
        for (i, row) in inv.iter_mut().enumerate() {
            row[i] = 1.0;
        }
        for col in 0..4 {
            let pivot = (col..4)
                .max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))
                .unwrap();
            a.swap(col, pivot);
            inv.swap(col, pivot);
            let scale = a[col][col].recip();
            for k in 0..4 {
                a[col][k] *= scale;
                inv[col][k] *= scale;
            }
            for row in 0..4 {
                if row != col {
                    let factor = a[row][col];
                    for k in 0..4 {
                        a[row][k] -= factor * a[col][k];
                        inv[row][k] -= factor * inv[col][k];
                    }
                }
            }
        }
        inv
    }

    /// The product of two 4 × 4 matrices.
    pub fn product(a: [[f64; 4]; 4], b: [[f64; 4]; 4]) -> [[f64; 4]; 4] {
        core::array::from_fn(|i| core::array::from_fn(|j| (0..4).map(|k| a[i][k] * b[k][j]).sum()))
    }

    /// numga recovers the transport independently, from finite changes of the metric and the
    /// frame, in coordinates: the frame takes local directions to coordinate ones, and the
    /// Christoffel symbols follow from the metric's changes along each axis.
    #[test]
    fn connection_matches_directional_metric_variation() {
        let position = event(1.0, 4.0, 2.0, 1.0);
        let hole = tilted_hole();
        let velocity = event(1.0, 0.4, -0.2, 0.3);
        let increment = 1e-4;
        let geometry = field(position, hole);
        let at = |k: usize, sign: f64| field(position + directions()[k] * (sign * increment), hole);
        // The coordinate metric: minus the flat products of the axes carried back into the
        // local frame.
        let metric = |g: &Field| {
            let carried = directions().map(|d| gauge_inverse(g).of(d));
            carried.map(|a| carried.map(|b| -a.dot(b).s()))
        };
        let frame = matrix(D::slot() + dyad(&geometry));
        // `[axis][a][b]`, centred differences along each axis.
        let frame_gradient: [[[f64; 4]; 4]; 4] = core::array::from_fn(|k| {
            let (ahead, behind) = (matrix(dyad(&at(k, 1.0))), matrix(dyad(&at(k, -1.0))));
            core::array::from_fn(|a| {
                core::array::from_fn(|b| (ahead[a][b] - behind[a][b]) / (2.0 * increment))
            })
        });
        let metric_gradient: [[[f64; 4]; 4]; 4] = core::array::from_fn(|k| {
            let (ahead, behind) = (metric(&at(k, 1.0)), metric(&at(k, -1.0)));
            core::array::from_fn(|a| {
                core::array::from_fn(|b| (ahead[a][b] - behind[a][b]) / (2.0 * increment))
            })
        });
        let inverse_metric = inverse(metric(&geometry));
        // Γᵃ_bc = ½ gᵃᵈ (∂_b g_dc + ∂_c g_db − ∂_d g_bc).
        let christoffel = |a: usize, b: usize, c: usize| {
            (0..4)
                .map(|d| {
                    inverse_metric[a][d]
                        * (metric_gradient[b][d][c] + metric_gradient[c][d][b]
                            - metric_gradient[d][b][c])
                })
                .sum::<f64>()
                / 2.0
        };
        let components = slices().map(|s| read(s, velocity));
        let change: [[f64; 4]; 4] = core::array::from_fn(|a| {
            core::array::from_fn(|j| {
                let along: f64 = (0..4)
                    .map(|b| components[b] * frame_gradient[b][a][j])
                    .sum();
                let turned: f64 = (0..4)
                    .flat_map(|b| (0..4).map(move |c| (b, c)))
                    .map(|(b, c)| christoffel(a, b, c) * components[b] * frame[c][j])
                    .sum();
                -(along + turned)
            })
        });
        let expected = product(inverse(frame), change);
        let local_velocity = gauge_inverse(&geometry).of(velocity);
        let transported = -rotation(&geometry, local_velocity).commutator(D::slot());
        let flat = rotation(&field(position, Hole { mass: 0.0, ..hole }), local_velocity);
        // The finite-difference tolerance checks the connection, including its transport sign.
        let got = matrix(transported);
        let error = (0..16)
            .map(|k| (got[k / 4][k % 4] - expected[k / 4][k % 4]).abs())
            .fold(0.0, f64::max);
        assert!(error < 2e-8, "{error}");
        assert!(largest(flat) < ROUND_OFF);
    }

    #[test]
    fn parallel_transport_preserves_the_transverse_screen_along_kerr_rays() {
        let hole = tilted_hole();
        let orientation = ((x() ^ z()) * 0.31).exp();
        let eye = (orientation >> along_z()) * 18.0;
        let (inner_radius, outer_radius) = (2.02, 40.0);
        let clear = Disk::clear(z(), outer_radius);
        let gram = screen_frame(along_x(), along_y()).dot(screen_frame(along_x(), along_y()));
        let (mut transverse_error, mut gram_error) = (0.0f64, 0.0f64);
        for ray in camera_rays(eye, orientation, [5, 3], 0.4, hole) {
            let start = PolarizedRay {
                ray,
                screen: camera_screen(ray, orientation, hole),
            };
            let steps = camera_trace(start, hole, 0.08, 250, polarized_rates);
            let (_, end) = resolve(steps, &clear, outer_radius, inner_radius, hole);
            let heading = local(&field(end.ray.event, hole), end.ray.momentum);
            let heading = heading * -read(t(), heading).recip();
            transverse_error = transverse_error.max(largest(end.screen.dot(heading)));
            gram_error = gram_error.max(end.screen.dot(end.screen).max_abs_diff(&gram));
        }
        // Finite-step transport keeps both screen axes unit, perpendicular, and transverse, to
        // the fourth-order error of the steps, about 2e-5 here, at the end of every ray.
        assert!(
            transverse_error < 5e-5 && gram_error < 5e-5,
            "{transverse_error} {gram_error}"
        );
    }

    /// The geometry, a ray's rates and the rotation gauge field against numga's numbers (numga
    /// 35797de): the tangent read by the slices `t x y z`, the force paired with `T X Y Z`.
    #[test]
    fn geometry_and_rates_agree_with_numga() {
        let hole = tilted_hole();
        let position = event(1.0, 4.0, 2.0, 1.0);
        let geometry = field(position, hole);
        assert!((geometry.radius - 4.536444972620433).abs() < ROUND_OFF);
        assert!((geometry.profile - 0.21927199062402092).abs() < ROUND_OFF);
        let momentum = Vector::new(1.0, 0.8, -0.3, 0.2);
        let rate = rates(
            Ray {
                event: position,
                momentum,
            },
            hole,
        );
        let tangent = [
            -0.200082442486748,
            -0.6282980022965416,
            -0.67732015878881,
            0.38278431161649895,
        ];
        let force = [
            0.0,
            0.03949794662162885,
            -0.02609643968871523,
            0.053692297315368495,
        ];
        for k in 0..4 {
            assert!((read(slices()[k], rate.event) - tangent[k]).abs() < ROUND_OFF);
            assert!(((rate.momentum & directions()[k]).s() - force[k]).abs() < ROUND_OFF);
        }
        assert!((hamiltonian(&geometry, momentum) + 0.6625029556549445).abs() < ROUND_OFF);
        // Bivectors in numga's blades and order, `yz zx xy xt yt zt`.
        let turning = rotation(&geometry, local(&geometry, momentum));
        let expected = Bivector::new(
            0.005760149485281401,
            -0.025940037570786407,
            -0.018324537679069764,
            -0.04078691965981568,
            0.004837869392765291,
            -0.03489413491134928,
        );
        assert!(turning.max_abs_diff(&expected) < ROUND_OFF);
        let curl = Bivector::new(
            -0.029697406377509807,
            -0.0098015967664604,
            -0.006614113884098463,
            0.0,
            0.0,
            0.0,
        );
        assert!(geometry.null_curl.max_abs_diff(&curl) < ROUND_OFF);
    }
}
