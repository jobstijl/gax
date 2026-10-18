//! numga's `kerr/core.py`, the geometry and the rays: the Kerr–Schild field of a rotating hole,
//! the rays' Hamiltonian motion through it, the parallel transport of a camera's screen along
//! them, and the camera's rays.

use core::ops::{Add, Mul};
use gax::sta::{Bivector, Even, Scalar, Trivector, Vector};

pub type V = Vector<(), f64>;
pub type B = Bivector<(), f64>;
/// A linear map on vectors, numga's `Vector <- Vector`: a metric, a frame, a derivative.
pub type Map = Vector<(Vector,), f64>;
/// The camera's two screen axes, a map from the screen to spacetime (numga's
/// `Vector <- Screen`). gax's STA has no kind for the screen plane `x y`, so the map takes
/// vectors and reads only their `x` and `y` parts: its `t` and `z` columns stay zero.
pub type ScreenFrame = Map;
/// A unit rotor of spacetime: the camera's orientation.
pub type Rotor = gax::Unit<Even<(), f64>>;

/// The event with time `t` and position `x, y, z`.
pub fn event(t: f64, x: f64, y: f64, z: f64) -> V {
    Vector::new(t, x, y, z)
}

pub fn t() -> V {
    event(1.0, 0.0, 0.0, 0.0)
}
pub fn x() -> V {
    event(0.0, 1.0, 0.0, 0.0)
}
pub fn y() -> V {
    event(0.0, 0.0, 1.0, 0.0)
}
pub fn z() -> V {
    event(0.0, 0.0, 0.0, 1.0)
}

/// The equatorial plane `x ∧ y`: the spin's and the disk's orbits'.
pub fn xy() -> B {
    x() ^ y()
}

/// The spatial pseudoscalar `x ∧ y ∧ z`: a spatial bivector's product with it is the vector
/// along its axis, and back.
pub fn xyz() -> Trivector<(), f64> {
    x() ^ y() ^ z()
}

/// The part of a vector across the time axis, `t ⌋ (t ∧ v)`.
pub fn spatial(v: V) -> V {
    t() | (t() ^ v)
}

/// The same as a map: the identity on space, zero on time.
fn spatial_identity() -> Map {
    t() | (t() ^ Vector::slot())
}

/// The identity on the screen plane `x y`, zero across it (numga's `Screen` as a map): each
/// axis times the pairing with its reciprocal, `-x` for `x`, since spatial axes square to
/// minus one.
pub fn screen_identity() -> Map {
    -(x() * (x() | Vector::slot())) - y() * (y() | Vector::slot())
}

/// The metric's pairing of two vectors, both left open.
fn minkowski() -> Scalar<(Vector, Vector), f64> {
    Vector::slot() | Vector::slot()
}

/// The adjoint of a map under the metric, `M*` with `w · M(v) = M*(w) · v` for all `v` and
/// `w`: a pairing solve.
pub fn adjoint(map: Map) -> Map {
    minkowski().solve(Vector::slot() | map)
}

/// A rotating black hole: its mass, and its spin, a spatial bivector of magnitude J/M.
#[derive(Clone, Copy, Debug)]
pub struct Hole {
    pub spin: B,
    pub mass: f64,
}

impl Hole {
    /// A hole spinning by `spin` about `+z`, turning from `x` to `y`.
    pub fn about_z(spin: f64, mass: f64) -> Hole {
        Hole {
            spin: xy() * spin,
            mass,
        }
    }

    /// The outer horizon's oblate radius, `M + √(M² − a²)`.
    #[allow(clippy::disallowed_methods)] // the Kerr horizon: a physical law
    pub fn horizon(self) -> f64 {
        self.mass + (self.mass * self.mass - self.spin.norm_squared()).sqrt()
    }

    /// The spin counted along the unit plane `orbit`: positive when the orbit turns with
    /// the hole.
    pub fn along(self, orbit: B) -> f64 {
        -self.spin.scalar_product(orbit).s()
    }
}

/// The Kerr–Schild geometry at an event: the oblate radius, the profile `H` that weighs
/// the null dyad, the null direction `l`, the profile's gradient, the null direction's
/// derivative (a map from a step to the change it makes, which is spatial) and its curl.
#[derive(Clone, Copy, Debug)]
pub struct Field {
    pub radius: f64,
    pub profile: f64,
    pub null: V,
    pub profile_gradient: V,
    pub null_gradient: Map,
    pub null_curl: B,
}

/// The oblate radius `r` from the squared distance from the centre, the squared spin and
/// the position's projection on the spin axis: the positive root of
/// `r⁴ − (|x|² − a²) r² − (a · x)² = 0`. Returns it with the square root of the
/// quadratic's discriminant.
#[allow(clippy::disallowed_methods)] // a quadratic's root: the Kerr–Schild radius
fn oblate_radius(distance_squared: f64, spin_squared: f64, projection: f64) -> (f64, f64) {
    let offset = distance_squared - spin_squared;
    let discriminant = (offset * offset + 4.0 * projection * projection).sqrt();
    (((offset + discriminant) / 2.0).sqrt(), discriminant)
}

/// The Kerr profile, null direction and their derivatives at an event.
pub fn field(position: V, hole: Hole) -> Field {
    let Hole { spin, mass } = hole;
    let spatial = spatial(position);
    let normal = spin | xyz();
    let projection = (normal | spatial).s();
    let spin_squared = spin.norm_squared();
    // A vector's norm squared is its square, minus its length squared for a spatial one.
    let (radius, discriminant) = oblate_radius(-spatial.norm_squared(), spin_squared, projection);
    let radius_squared = radius * radius;
    let scale = (radius_squared + spin_squared).recip();
    // Its inner product with a small step is how far the step changes the oblate radius.
    let radius_gradient =
        (spatial * radius_squared - normal * projection) * (-1.0 / (radius * discriminant));
    let direction = (spatial * radius + (spatial | spin) - normal * (projection / radius)) * scale;
    let lever = spatial - direction * (2.0 * radius) + normal * (projection / radius_squared);
    let step = Vector::slot();
    let direction_gradient: Map = (lever * (radius_gradient | step)
        + spatial_identity() * radius
        + (spatial_identity() | spin)
        - normal * (normal | step) * radius.recip())
        * scale;
    let denominator = radius_squared * radius_squared + projection * projection;
    let profile = mass * radius.powi(3) / denominator;
    let profile_gradient = (radius_gradient * (3.0 / radius)
        - (radius_gradient * (4.0 * radius.powi(3)) + normal * (2.0 * projection))
            * denominator.recip())
        * profile;
    // The derivative's curl, `Σ eⁱ ∧ D(eᵢ)`, piece by piece: the rank-one piece gives its
    // two vectors wedged, the turn by the spin twice the spin, the rest none.
    let null_curl = ((radius_gradient ^ lever) + spin * 2.0) * -scale;
    Field {
        radius,
        profile,
        null: t() - direction,
        profile_gradient,
        null_gradient: -direction_gradient,
        null_curl,
    }
}

/// The null dyad weighed by the profile, `v ↦ H l (l · v)`; it squares to zero.
pub fn dyad(geometry: &Field) -> Map {
    geometry.null * (geometry.null | Vector::slot()) * geometry.profile
}

/// The identity minus twice the dyad: a vector to its covector.
#[cfg_attr(not(test), allow(dead_code))] // numga's core, which the tests check
pub fn metric(geometry: &Field) -> Map {
    Vector::slot() - dyad(geometry) * 2.0
}

/// The identity plus twice the dyad: covector momentum to the ray's tangent.
pub fn inverse_metric(geometry: &Field) -> Map {
    Vector::slot() + dyad(geometry) * 2.0
}

/// The identity plus the dyad: covector momentum to its direction in the local frame.
pub fn frame(geometry: &Field) -> Map {
    Vector::slot() + dyad(geometry)
}

/// The identity minus the dyad: a vector to its image in the local frame, the inverse of
/// [`frame`]. The metric is its square, so a vector's length under the metric is the
/// Minkowski length of its image.
pub fn vector_frame(geometry: &Field) -> Map {
    Vector::slot() - dyad(geometry)
}

/// `v` scaled to unit length under the metric.
pub fn proper_unit(geometry: &Field, v: V) -> V {
    v * vector_frame(geometry).of(v).norm().recip()
}

/// The gradient of the null direction's overlap with a fixed vector, `l · v`: the vector
/// pulled back through the null direction's derivative, whose changes are spatial.
fn overlap_gradient(geometry: &Field, v: V) -> V {
    adjoint(geometry.null_gradient).of(spatial(v))
}

/// What a ray carries: its event and its momentum. It is also the form of its rate (the
/// tangent and the momentum's rate), so Runge–Kutta adds and scales it.
#[derive(Clone, Copy, Debug)]
pub struct Ray {
    pub event: V,
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

/// The two Hamiltonian derivatives at a field, as a ray's rate: the tangent and minus the
/// position gradient. Half the momentum paired with its image under the inverse metric is
/// the Hamiltonian; its momentum derivative is that image.
pub fn motion(geometry: &Field, momentum: V) -> Ray {
    let overlap = (geometry.null | momentum).s();
    let tangent = momentum + geometry.null * (2.0 * geometry.profile * overlap);
    // The gradient of `H (l · p)²`, by the product rule.
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

/// How fast coordinate time runs back along a past-directed tangent.
fn clock(tangent: V) -> f64 {
    -(t() | tangent).s()
}

/// Past-directed motion per unit of decreasing coordinate time.
#[cfg_attr(not(test), allow(dead_code))] // numga's core, which the tests check
pub fn camera_rates(ray: Ray, hole: Hole) -> Ray {
    let rate = rates(ray, hole);
    rate * clock(rate.event).recip()
}

/// The plane the local Kerr–Schild frame turns or boosts in, per unit step along a
/// direction.
pub fn connection(geometry: &Field, direction: V) -> B {
    let along = (geometry.null | direction).s();
    // The gradient of `H (l · d)`.
    let turning = geometry.profile_gradient * along
        + overlap_gradient(geometry, direction) * geometry.profile;
    (turning ^ geometry.null) + geometry.null_curl * (geometry.profile * along)
}

/// Ray motion and the parallel transport of its camera screen, per unit of decreasing
/// coordinate time.
pub fn polarized_rates(state: PolarizedRay, hole: Hole) -> PolarizedRay {
    let geometry = field(state.ray.event, hole);
    let local_momentum = frame(&geometry).of(state.ray.momentum);
    let rate = PolarizedRay {
        ray: motion(&geometry, state.ray.momentum),
        screen: -connection(&geometry, local_momentum).commutator(state.screen),
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

/// A future null covector with the given unit spatial direction in the Kerr–Schild frame.
pub fn initial_momentum(position: V, direction: V, hole: Hole) -> V {
    let geometry = field(position, hole);
    let local_null = t() + direction;
    // The dyad squares to zero, so the inverse frame just changes its sign.
    let momentum = local_null - dyad(&geometry).of(local_null);
    momentum * (momentum | t()).s().recip()
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
) -> impl Iterator<Item = Ray> {
    let horizon = hole.horizon();
    std::iter::successors(Some(start), move |ray| {
        let radius = field(ray.event, hole).radius;
        let clock = ((radius / horizon - stop_fraction) / (1.0 - stop_fraction)).clamp(0.0, 1.0);
        Some(runge_kutta(|r| rates(r, hole), *ray, step_size * clock))
    })
    .take(steps + 1)
}

/// Pixel-centre rays from `eye`, row by row, looking along the orientation's `-z`, with
/// right `+y` and up `-x`; `half_view` is the tangent of half the vertical field of view.
pub fn camera_rays(
    eye: V,
    orientation: Rotor,
    [width, height]: [usize; 2],
    half_view: f64,
    hole: Hole,
) -> Vec<Ray> {
    let centre = |i: usize, n: usize| (i as f64 + 0.5) / n as f64 * 2.0 - 1.0;
    let aspect = width as f64 / height as f64;
    (0..width * height)
        .map(|k| {
            let across = centre(k % width, width) * half_view * aspect;
            let up = -centre(k / width, height) * half_view;
            let look = orientation >> (-z() + y() * across - x() * up);
            let direction = look.normalized();
            Ray {
                event: eye,
                momentum: -initial_momentum(eye, -direction, hole),
            }
        })
        .collect()
}

/// Orthonormal axes across a camera ray, with horizontal `x` and vertical `y` screen
/// inputs.
pub fn camera_screen(ray: Ray, orientation: Rotor, hole: Hole) -> ScreenFrame {
    let local_momentum = frame(&field(ray.event, hole)).of(ray.momentum);
    let direction = spatial(local_momentum) * -(t() | local_momentum).s().recip();
    let right = orientation >> y();
    let right = (right + direction * (direction | right).s()).normalized();
    let up = (direction ^ right) | xyz();
    -(right * (x() | Vector::slot())) - up * (y() | Vector::slot())
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
    /// Whether the step crosses the central plane normal to `normal`.
    pub fn crosses(&self, normal: V) -> bool {
        (normal | self.start.ray().event).s() * (normal | self.end.ray().event).s() <= 0.0
    }
}

/// Fourth-order steps of a camera ray, each a fraction of its oblate radius, at most
/// `steps` of them; the consumer stops when the ray is resolved. The state is a ray, with its
/// screen when `dynamics` transports one.
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
    use crate::disk::disk_crossing;
    use gax::ApproxEq;

    pub const ROUND_OFF: f64 = 1e-11;

    /// The basis vectors `t, x, y, z`.
    fn basis() -> [V; 4] {
        [t(), x(), y(), z()]
    }

    /// The reciprocal of a basis vector, `eⁱ` with `eⁱ · eᵢ = 1`: over its square.
    fn reciprocal(e: V) -> V {
        e * e.norm_squared().recip()
    }

    /// A component of `v`: its pairing with the reciprocal of a basis vector.
    fn component(v: V, e: V) -> f64 {
        (reciprocal(e) | v).s()
    }

    /// The largest coefficient of a value or map.
    fn largest<E: ApproxEq + Mul<f64, Output = E> + Copy>(e: E) -> f64 {
        e.max_abs_diff(&(e * 0.0))
    }

    /// A hole whose spin is tilted off `z`.
    fn tilted_hole() -> Hole {
        Hole {
            spin: xy() * 0.7 + (y() ^ z()) * 0.2,
            mass: 1.0,
        }
    }

    #[test]
    fn null_congruence_and_exact_inverse_metric() {
        let positions = [
            event(0.0, 4.0, 2.0, 1.0),
            event(3.0, -3.0, 1.0, 2.0),
            event(-1.0, 2.0, -4.0, -1.0),
        ];
        let hole = tilted_hole();
        for position in positions {
            let geometry = field(position, hole);
            let momentum = initial_momentum(position, x(), hole);
            // The null direction follows straight lines in the background geometry; its dyad
            // squares to zero, so the inverse changes only the sign.
            assert!((geometry.null | geometry.null).s().abs() < ROUND_OFF);
            assert!(largest(geometry.null_gradient.of(geometry.null)) < ROUND_OFF);
            // The closed-form curl is the null direction's derivative wedged with the
            // reciprocal basis.
            let curl: B = basis()
                .map(|e| reciprocal(e) ^ geometry.null_gradient.of(e))
                .into_iter()
                .sum();
            assert!(geometry.null_curl.max_abs_diff(&curl) < ROUND_OFF);
            let composed = metric(&geometry).of(inverse_metric(&geometry));
            assert!(composed.max_abs_diff(&Vector::slot()) < ROUND_OFF);
            let tangent = inverse_metric(&geometry).of(momentum);
            assert!((momentum | tangent).s().abs() < ROUND_OFF);
            assert!(((momentum | t()).s() - 1.0).abs() < ROUND_OFF);

            // With zero spin the profile is the mass divided by ordinary spatial distance.
            let schwarzschild = field(position, Hole::about_z(0.0, hole.mass));
            let radius = spatial(position).norm();
            assert!((schwarzschild.radius - radius).abs() < ROUND_OFF);
            assert!((schwarzschild.profile - hole.mass / radius).abs() < ROUND_OFF);
        }
    }

    #[test]
    fn directional_derivatives_drive_hamiltonian_rates() {
        let position = event(1.0, 4.0, 2.0, 1.0);
        let momentum = event(1.0, 0.8, -0.3, 0.2);
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
            |g: &Field, p: V| (p | p).s() / 2.0 + g.profile * (g.null | p).s().powi(2);
        // These tolerances measure centred-difference accuracy, not round-off alone.
        for direction in basis() {
            let ahead = field(position + direction * increment, hole);
            let behind = field(position - direction * increment, hole);
            let profile_change = difference(ahead.profile, behind.profile);
            assert!((profile_change - (geometry.profile_gradient | direction).s()).abs() < 2e-9);
            let null_change = (ahead.null - behind.null) * (1.0 / (2.0 * increment));
            assert!(null_change.max_abs_diff(&geometry.null_gradient.of(direction)) < 2e-9);
            let force_change = difference(
                hamiltonian(&ahead, momentum),
                hamiltonian(&behind, momentum),
            );
            assert!((force_change + (rate.momentum | direction).s()).abs() < 2e-9);
            let velocity_change = difference(
                hamiltonian(&geometry, momentum + direction * increment),
                hamiltonian(&geometry, momentum - direction * increment),
            );
            assert!((velocity_change - (rate.event | direction).s()).abs() < 2e-9);
        }
    }

    #[test]
    fn rays_conserve_their_invariants_and_spin_reversal_mirrors_the_paths() {
        let mass = 1.0;
        let mirror = |v: V| v + y() * (2.0 * (y() | v).s());
        let (step_size, steps, stop_fraction) = (0.04, 250, 0.8);
        let trace_from = |start: V, spin: f64| {
            let hole = Hole::about_z(spin, mass);
            let ray = Ray {
                event: start,
                momentum: initial_momentum(start, x(), hole),
            };
            trace(ray, hole, step_size, steps, stop_fraction).collect::<Vec<_>>()
        };
        for offset in [-7.0, 2.0, 7.0] {
            let start = x() * -8.0 + y() * offset;
            let rays = trace_from(start, 0.7);
            let mirrored = trace_from(mirror(start), -0.7);
            for (spin, path) in [(0.7, &rays), (-0.7, &mirrored)] {
                let hole = Hole::about_z(spin, mass);
                let angular = |r: &Ray| ((r.event ^ r.momentum) | hole.spin).s();
                let (energy, angular_momentum) = ((t() | path[0].momentum).s(), angular(&path[0]));
                // Null propagation and axial angular momentum constrain the integrated
                // dynamics; their tolerances measure the accuracy of the finite RK4 steps.
                for ray in path {
                    let g = field(ray.event, hole);
                    let h = (ray.momentum | ray.momentum).s() / 2.0
                        + g.profile * (g.null | ray.momentum).s().powi(2);
                    assert!(h.abs() < 2e-6, "{h}");
                    assert!((angular(ray) - angular_momentum).abs() < 1e-7);
                    assert!(((t() | ray.momentum).s() - energy).abs() < ROUND_OFF);
                }
            }
            for (ray, image) in rays.iter().zip(&mirrored) {
                assert!(mirror(ray.event).max_abs_diff(&image.event) < ROUND_OFF);
                assert!(mirror(ray.momentum).max_abs_diff(&image.momentum) < ROUND_OFF);
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
        let eye = (orientation >> z()) * 25.0;
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
            assert!((ray.momentum | tangent).s().abs() < ROUND_OFF);
            assert!((t() | tangent).s() < 0.0);
            assert!(turned.event.max_abs_diff(&(rotation >> ray.event)) < ROUND_OFF);
            assert!(turned.momentum.max_abs_diff(&(rotation >> ray.momentum)) < ROUND_OFF);
        }
        let centre = rays[size[0] * size[1] / 2];
        let local_null = frame(&field(centre.event, hole)).of(centre.momentum);
        let looking = spatial(local_null).normalized();
        let toward_hole = (-eye).normalized();
        assert!(looking.max_abs_diff(&toward_hole) < ROUND_OFF);

        // Intersect a tilted disk between events whose heights have opposite signs.
        let at = |e: V| Ray {
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
        let eye = x() * 25.0;
        // Inside the photon sphere, where no ray turns back out.
        let (capture_radius, escape_radius) = (2.1, 40.0);
        for slope in [0.14, 0.18, 0.20, 0.28] {
            let direction = (-x() + y() * slope).normalized();
            let start = Ray {
                event: eye,
                momentum: -initial_momentum(eye, -direction, hole),
            };
            let energy = (t() | start.momentum).s();
            let angular = |r: &Ray| ((r.event ^ r.momentum) | xy()).s();
            let angular_momentum = angular(&start);
            let expected_capture =
                (angular_momentum / energy).abs() < 3.0 * 3f64.sqrt() * hole.mass;
            let mut captured = None;
            for step in camera_trace(start, hole, 0.1, 300, camera_rates) {
                if step.radius < capture_radius {
                    captured = Some(true);
                    break;
                }
                if step.radius > escape_radius {
                    // The finite-step check constrains escaped rays after their full encounter.
                    let end = step.end;
                    assert!((end.momentum | rates(end, hole).event).s().abs() < 2e-5);
                    assert!((angular(&end) - angular_momentum).abs() < 2e-5);
                    assert!(((t() | end.momentum).s() - energy).abs() < ROUND_OFF);
                    captured = Some(false);
                    break;
                }
            }
            assert_eq!(captured, Some(expected_capture), "slope {slope}");
        }
    }

    /// numga recovers the transport independently, from finite changes of the metric and the
    /// frame, with Christoffel symbols in components; here the same with forms. The metric's
    /// form is `g(u, w) = u · G(w)`, and the Christoffel map `Γ(v, ·)` solves
    /// `g(Γ(v, y), z) = ½ [∂_v g(y, z) + ∂_y g(v, z) − ∂_z g(v, y)]` by a pairing solve, with
    /// `y` and `z` left open (Koszul's formula).
    #[test]
    fn connection_matches_directional_metric_variation() {
        let position = event(1.0, 4.0, 2.0, 1.0);
        let hole = tilted_hole();
        let velocity = event(1.0, 0.4, -0.2, 0.3);
        let increment = 1e-4;
        let geometry = field(position, hole);
        let dyad_at = |p: V| dyad(&field(p, hole));
        // The dyad's partial derivatives, by centred differences along each basis vector,
        // and its derivative along a vector from them.
        let partials = basis().map(|e| {
            (dyad_at(position + e * increment) - dyad_at(position - e * increment))
                * (0.5 / increment)
        });
        let along = |v: V| -> Map {
            basis()
                .iter()
                .zip(&partials)
                .map(|(e, d)| *d * component(v, *e))
                .sum()
        };
        // The metric's change along a basis vector is minus twice the dyad's.
        let metric_partials = partials.map(|d| (Vector::slot() | d) * -2.0);
        let open_component = |e: V| reciprocal(e) | Vector::slot();
        let metric_along_v = (Vector::slot() | along(velocity)) * -2.0;
        let by_y: Scalar<(Vector, Vector), f64> = basis()
            .iter()
            .zip(&metric_partials)
            .map(|(e, m)| open_component(*e) * m.of(velocity))
            .sum();
        let by_z: Scalar<(Vector, Vector), f64> = basis()
            .iter()
            .zip(&metric_partials)
            .map(|(e, m)| m.of(velocity) * open_component(*e))
            .sum();
        let koszul = (metric_along_v + by_y - by_z) * 0.5;
        let christoffel: Map = (Vector::slot() | metric(&geometry)).solve(koszul);
        let local_frame = frame(&geometry);
        let expected = -local_frame.solve(along(velocity) + christoffel.of(local_frame));

        let local_velocity = vector_frame(&geometry).of(velocity);
        let transported = -connection(&geometry, local_velocity).commutator(Vector::slot());
        let flat = connection(&field(position, Hole { mass: 0.0, ..hole }), local_velocity);
        // The finite-difference tolerance checks the connection, including its transport sign.
        let error = transported.max_abs_diff(&expected);
        assert!(error < 2e-8, "{error}");
        assert!(largest(flat) < ROUND_OFF);
    }

    #[test]
    fn parallel_transport_preserves_the_transverse_screen_along_kerr_rays() {
        let hole = tilted_hole();
        let orientation = ((x() ^ z()) * 0.31).exp();
        let eye = (orientation >> z()) * 18.0;
        let (inner_radius, outer_radius) = (2.02, 40.0);
        let gram = screen_identity() | screen_identity();
        let (mut transverse_error, mut gram_error) = (0.0f64, 0.0f64);
        for ray in camera_rays(eye, orientation, [5, 3], 0.4, hole) {
            let start = PolarizedRay {
                ray,
                screen: camera_screen(ray, orientation, hole),
            };
            for step in camera_trace(start, hole, 0.1, 250, polarized_rates) {
                let end = step.end;
                let local = frame(&field(end.ray.event, hole)).of(end.ray.momentum);
                let local = local * -(t() | local).s().recip();
                transverse_error = transverse_error.max(largest(end.screen | local));
                gram_error = gram_error.max((end.screen | end.screen).max_abs_diff(&gram));
                if !(inner_radius < step.radius && step.radius < outer_radius) {
                    break;
                }
            }
        }
        // Finite-step transport keeps both screen axes unit, perpendicular, and transverse.
        assert!(
            transverse_error < 1e-5 && gram_error < 1e-5,
            "{transverse_error} {gram_error}"
        );
    }

    /// The geometry and a ray's rates against numga's numbers (numga 93c2ae3).
    #[test]
    fn geometry_and_rates_agree_with_numga() {
        let hole = tilted_hole();
        let position = event(1.0, 4.0, 2.0, 1.0);
        let geometry = field(position, hole);
        assert!((geometry.radius - 4.536444972620433).abs() < ROUND_OFF);
        assert!((geometry.profile - 0.219271990624021).abs() < ROUND_OFF);
        let rate = rates(
            Ray {
                event: position,
                momentum: event(1.0, 0.8, -0.3, 0.2),
            },
            hole,
        );
        let tangent = event(
            1.742298624661946,
            0.110357435421541,
            -0.527615780484615,
            0.04640495525232,
        );
        let force = event(
            0.0,
            -0.093278223592489,
            -0.143087233116015,
            -0.031330371641152,
        );
        assert!(rate.event.max_abs_diff(&tangent) < ROUND_OFF);
        assert!(rate.momentum.max_abs_diff(&force) < ROUND_OFF);
    }
}
