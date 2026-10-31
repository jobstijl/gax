//! numga's `kerr/metric.py`: the same light from the metric and its Levi-Civita connection,
//! which numga's gauge formulation replaced and its comparison notebook sets beside it.
//!
//! The metric maps directions to covectors, `Vector <- Antivector`; its inverse takes a
//! momentum to the ray's tangent. The metric's directional derivatives give the Christoffel
//! connection, which carries the coordinate screen (the gauge field's image of the local one)
//! along the ray. An orthonormal source frame, the gauge field, fixes the camera's angles and
//! the disk's prescribed polarization only. The field, the disk and the integrator are
//! [`crate::kerr`]'s and [`crate::disk`]'s.
// numga's metric description in full: the checks use it, the animation does not.
#![cfg_attr(not(test), allow(dead_code))]

use crate::disk::Coherency;
use crate::kerr::{
    B, D, Field, Hole, PolarizedRay, Ray, Rotor, ScreenFrame, V, camera_forward, camera_right,
    camera_up, clock, field, gauge, gauge_inverse, overlap_gradient, read, screen_frame, space,
    squared, t, time, unit_length,
};
use gax::sta::{Even, Scalar, Vector, intrinsic};

/// The metric: from a direction to its covector, numga's `Vector <- Antivector`.
pub type Metric = Vector<(D,)>;
/// The inverse metric: from a momentum to its coordinate tangent, `Antivector <- Vector`.
pub type InverseMetric = intrinsic::Trivector<(V,)>;

/// The physical metric: a direction's covector, with positive timelike square.
pub fn metric(geometry: &Field) -> Metric {
    let plane = geometry.null_plane();
    -D::slot().hodge() - plane * (plane & D::slot()) * (2.0 * geometry.profile)
}

/// The inverse metric, taking a covector momentum to its coordinate tangent.
pub fn inverse_metric(geometry: &Field) -> InverseMetric {
    let null = geometry.null;
    -V::slot().hodge() + null * (V::slot() & null) * (2.0 * geometry.profile)
}

/// An orthonormal frame specifying the camera and the prescribed emission, in coordinates:
/// the same local source directions as the Kerr–Schild frame. Gravitational evolution uses
/// the metric and the Christoffel connection; this frame only defines local observations.
pub fn source_frame(geometry: &Field) -> crate::kerr::DirectionMap {
    gauge(geometry)
}

/// Coordinate directions resolved in the source frame.
pub fn source_components(geometry: &Field) -> crate::kerr::DirectionMap {
    gauge_inverse(geometry)
}

/// The metric's reading of a direction on itself, positive for a timelike direction.
pub fn proper_squared(geometry: &Field, direction: D) -> f64 {
    (metric(geometry).of(direction) & direction).s()
}

/// Half the momentum's reading of its image under the inverse metric.
pub fn hamiltonian(geometry: &Field, momentum: V) -> f64 {
    (momentum & inverse_metric(geometry).of(momentum)).s() / 2.0
}

/// The two Hamiltonian derivatives at a field: differentiating the quadratic Hamiltonian
/// with respect to momentum applies the inverse metric.
pub fn motion(geometry: &Field, momentum: V) -> Ray {
    let overlap = (momentum & geometry.null).s();
    let gradient = geometry.profile_gradient * (overlap * overlap)
        + overlap_gradient(geometry, momentum) * (2.0 * geometry.profile * overlap);
    Ray {
        event: inverse_metric(geometry).of(momentum),
        momentum: -gradient,
    }
}

/// The ray's rate in its affine parameter.
pub fn rates(ray: Ray, hole: Hole) -> Ray {
    motion(&field(ray.event, hole), ray.momentum)
}

/// Past-directed motion per unit of decreasing coordinate time.
pub fn camera_rates(ray: Ray, hole: Hole) -> Ray {
    let rate = rates(ray, hole);
    rate * clock(rate.event).recip()
}

/// The metric's directional derivative along a coordinate step, applied to a fixed
/// direction.
pub fn metric_change(geometry: &Field, along: D, direction: D) -> V {
    let plane = geometry.null_plane();
    let plane_change = -geometry.null_gradient.of(along).hodge();
    let (reading, change_reading) = ((plane & direction).s(), (plane_change & direction).s());
    (plane * ((geometry.profile_gradient & along).s() * reading)
        + (plane_change * reading + plane * change_reading) * geometry.profile)
        * -2.0
}

/// The torsion-free metric connection in the fixed Cartesian coordinate basis.
pub fn christoffel(geometry: &Field, along: D, direction: D) -> D {
    let plane = geometry.null_plane();
    let (along_reading, direction_reading) = ((plane & along).s(), (plane & direction).s());
    let along_gradient = overlap_gradient(geometry, -along.hodge());
    let direction_gradient = overlap_gradient(geometry, -direction.hodge());
    // Differentiate the scalar metric reading with both coordinate directions held fixed.
    let reading_gradient = (geometry.profile_gradient * (along_reading * direction_reading)
        + (along_gradient * direction_reading + direction_gradient * along_reading)
            * geometry.profile)
        * -2.0;
    // Two directional derivatives and the scalar reading's gradient give the lowered
    // connection; the inverse metric returns its direction.
    inverse_metric(geometry).of(metric_change(geometry, along, direction)
        + metric_change(geometry, direction, along)
        - reading_gradient)
        * 0.5
}

/// The screen frame's two axes, its images of `X` and `Y`.
pub fn screen_axes(screen: ScreenFrame) -> [D; 2] {
    [
        screen.of(crate::kerr::along_x()),
        screen.of(crate::kerr::along_y()),
    ]
}

/// Hamiltonian motion and coordinate-screen transport, per unit of decreasing coordinate
/// time. (The connection is bilinear; on the screen it acts on each axis.)
pub fn polarized_rates(state: PolarizedRay, hole: Hole) -> PolarizedRay {
    let geometry = field(state.ray.event, hole);
    let ray = motion(&geometry, state.ray.momentum);
    let [right, up] =
        screen_axes(state.screen).map(|axis| -christoffel(&geometry, ray.event, axis));
    let rate = PolarizedRay {
        ray,
        screen: screen_frame(right, up),
    };
    rate * clock(rate.ray.event).recip()
}

/// Unit-intensity disk emission along the transported coordinate screen: the prescribed
/// magnetic pattern is specified in the source frame, so the screen is resolved there first
/// (the momentum's heading there is [`crate::kerr::local`]'s, which the disk's coherency
/// reads).
pub fn disk_coherency(
    event: D,
    momentum: V,
    screen: ScreenFrame,
    normal: V,
    hole: Hole,
    polarization_degree: f64,
) -> Coherency {
    let local_screen = source_components(&field(event, hole)).of(screen);
    crate::disk::disk_coherency(
        event,
        momentum,
        local_screen,
        normal,
        hole,
        polarization_degree,
    )
}

/// The unit velocity of orbiting gas, normalized with the physical metric.
#[allow(clippy::disallowed_methods)] // a proper length: the square root of its square
pub fn orbiting_gas(event: D, orbit: B, hole: Hole) -> D {
    let geometry = field(event, hole);
    let flow = crate::disk::gas_flow(event, &geometry, orbit, hole);
    flow * proper_squared(&geometry, flow).sqrt().recip()
}

/// The frequency a camera at rest at `camera` sees over the one orbiting gas emits, with
/// the physical metric's observers.
#[allow(clippy::disallowed_methods)] // a proper length: the square root of its square
pub fn frequency_ratio(event: D, momentum: V, camera: D, orbit: B, hole: Hole) -> f64 {
    let gas = orbiting_gas(event, orbit, hole);
    let at_rest = time() * proper_squared(&field(camera, hole), time()).sqrt().recip();
    (momentum & at_rest).s() / (momentum & gas).s()
}

/// A future null covector with the given unit spatial direction in the source frame,
/// lowered with the physical metric.
pub fn initial_momentum(position: D, direction: D, hole: Hole) -> V {
    let geometry = field(position, hole);
    let momentum = metric(&geometry).of(source_frame(&geometry).of(time() + direction));
    momentum * (momentum & time()).s().recip()
}

/// A coordinate-static camera's rest frame in the source frame, outside the static limit.
#[allow(clippy::disallowed_methods)] // a proper length: the square root of its square
pub fn camera_boost(geometry: &Field) -> Rotor {
    let observer =
        source_components(geometry).of(time()) * proper_squared(geometry, time()).sqrt().recip();
    (Scalar::new(1.0).cast::<Even>() - observer * time()).unit()
}

/// Pixel-centre rays of a coordinate-static camera, as [`crate::kerr::camera_rays`], their
/// headings lowered with the metric.
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
    let lower = metric(&geometry).of(source_frame(&geometry));
    (0..width * height)
        .map(|k| {
            let across = centre(k % width, width) * half_view * aspect;
            let up = -centre(k / width, height) * half_view;
            let look =
                orientation >> (camera_forward() + camera_right() * across + camera_up() * up);
            let heading = boost >> (time() - unit_length(look));
            let momentum = lower.of(heading);
            Ray {
                event: eye,
                momentum: momentum * -(momentum & time()).s().recip(),
            }
        })
        .collect()
}

/// The camera's screen in coordinates: the source frame's image of the local one.
#[allow(clippy::disallowed_methods)] // a length: the square root of its square
pub fn camera_screen(ray: Ray, orientation: Rotor, hole: Hole) -> ScreenFrame {
    let geometry = field(ray.event, hole);
    let boost = camera_boost(&geometry);
    let heading =
        boost << source_components(&geometry).of(inverse_metric(&geometry).of(ray.momentum));
    let direction = space(heading) * -read(t(), heading).recip();
    let forward = orientation >> camera_forward();
    let halfway = direction + forward;
    let turn = (halfway * forward) * squared(halfway).sqrt().recip();
    let frame = (boost.into_inner() * turn * orientation.into_inner()).unit();
    let to = source_frame(&geometry);
    screen_frame(to.of(frame >> camera_right()), to.of(frame >> camera_up()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::disk::{Disk, disk_temperature, innermost_stable_orbit, resolve};
    use crate::kerr::tests::{ROUND_OFF, largest, slices, tilted_hole};
    use crate::kerr::{along_z, camera_trace, event, rotation, trace, x, xy, z};
    use gax::ApproxEq;
    use gax_numga_examples::signal::tangent;

    #[test]
    fn metric_and_hamiltonian_agree_with_the_frame_on_general_momenta() {
        let positions = [
            event(0.0, 4.0, 2.0, 1.0),
            event(3.0, -3.0, 1.0, 2.0),
            event(-1.0, 2.0, -4.0, -1.0),
        ];
        let momenta = [
            [1.0, 0.8, -0.3, 0.2],
            [2.0, 0.1, 0.5, -0.7],
            [-1.0, 0.4, 0.2, 0.9],
        ];
        let hole = tilted_hole();
        for (position, m) in positions.into_iter().zip(momenta) {
            // numga's `SLICES * [t, x, y, z]`.
            let momentum: V = slices().into_iter().zip(m).map(|(s, c)| s * c).sum();
            let geometry = field(position, hole);
            let tangent = inverse_metric(&geometry).of(momentum);
            assert!(metric(&geometry).of(tangent).max_abs_diff(&momentum) < ROUND_OFF);
            let gauge_hamiltonian = crate::kerr::hamiltonian(&geometry, momentum);
            assert!((hamiltonian(&geometry, momentum) - gauge_hamiltonian).abs() < ROUND_OFF);
            let gauge_tangent = crate::kerr::motion(&geometry, momentum).event;
            assert!(tangent.max_abs_diff(&gauge_tangent) < ROUND_OFF);
            // Both lower a null direction of the source frame to the same momentum, and step
            // a camera ray alike.
            let along = crate::kerr::along_x();
            let null = initial_momentum(position, along, hole);
            let gauge_null = crate::kerr::initial_momentum(position, along, hole);
            assert!(null.max_abs_diff(&gauge_null) < ROUND_OFF);
            let ray = Ray {
                event: position,
                momentum: -null,
            };
            let rate = camera_rates(ray, hole);
            let gauge_rate = crate::kerr::camera_rates(ray, hole);
            assert!(rate.event.max_abs_diff(&gauge_rate.event) < ROUND_OFF);
            assert!(rate.momentum.max_abs_diff(&gauge_rate.momentum) < ROUND_OFF);
        }
        // Non-null momenta distinguish the Hamiltonian's sign, which a null check cannot.
        let flat = field(crate::kerr::along_x() * 4.0, Hole { mass: 0.0, ..hole });
        assert!((crate::kerr::hamiltonian(&flat, t()) - 0.5).abs() < ROUND_OFF);
        assert!((crate::kerr::hamiltonian(&flat, x()) + 0.5).abs() < ROUND_OFF);
    }

    #[test]
    fn christoffel_is_symmetric_and_matches_the_changing_local_frame() {
        let position = event(1.0, 4.0, 2.0, 1.0);
        let hole = tilted_hole();
        let (along, direction) = (event(1.0, 0.4, -0.2, 0.3), event(0.2, -0.3, 0.8, -0.1));
        let geometry = field(position, hole);
        let coordinate_direction = gauge(&geometry).of(direction);
        let increment = 1e-4;
        let carried =
            |sign: f64| gauge(&field(position + along * (sign * increment), hole)).of(direction);
        let frame_change = (carried(1.0) - carried(-1.0)) * (0.5 / increment);
        let local_along = gauge_inverse(&geometry).of(along);
        let local_rate = -rotation(&geometry, local_along).commutator(direction);
        let coordinate_rate = frame_change + gauge(&geometry).of(local_rate);
        let connection = christoffel(&geometry, along, coordinate_direction);
        assert!(largest(connection + coordinate_rate) < 2e-9);
        let swapped = christoffel(&geometry, direction, along);
        assert!(christoffel(&geometry, along, direction).max_abs_diff(&swapped) < ROUND_OFF);
        // The covariant derivative of the metric vanishes.
        let variation = (metric_change(&geometry, along, direction) & direction).s();
        let connection = christoffel(&geometry, along, direction);
        let reading = (metric(&geometry).of(connection) & direction).s();
        assert!((variation - 2.0 * reading).abs() < ROUND_OFF);
        // numga's connection here (numga 35797de), read by the slices `t x y z`.
        let expected = [
            0.03830050160131386,
            -0.02543150405452107,
            -0.007685956513472914,
            -0.013243980584361397,
        ];
        for (s, e) in slices().into_iter().zip(expected) {
            assert!((read(s, connection) - e).abs() < ROUND_OFF);
        }
    }

    /// numga's comparison camera: the coordinate-static observer at 28 M, 1.2 rad from the
    /// spin axis.
    fn camera() -> (Hole, Rotor, D) {
        let hole = Hole::about_z(0.8, 1.0);
        let orientation = ((z() ^ x()) * -0.6).exp();
        (hole, orientation, (orientation >> along_z()) * 28.0)
    }

    #[test]
    #[allow(clippy::disallowed_methods)] // a proper length: the square root of its square
    fn both_cameras_use_the_coordinate_static_observer_for_screen_and_frequency() {
        let (hole, orientation, eye) = camera();
        let rays = crate::kerr::camera_rays(eye, orientation, [5, 3], 0.35, hole);
        let metric_rays = camera_rays(eye, orientation, [5, 3], 0.35, hole);
        for (ray, metric_ray) in rays.iter().zip(&metric_rays) {
            let local_screen = crate::kerr::camera_screen(*ray, orientation, hole);
            let coordinate_screen = camera_screen(*metric_ray, orientation, hole);
            let geometry = field(ray.event, hole);
            let observer = time() * proper_squared(&geometry, time()).sqrt().recip();
            let tangent = inverse_metric(&geometry).of(ray.momentum);
            assert!(metric_ray.momentum.max_abs_diff(&ray.momentum) < ROUND_OFF);
            let carried = gauge(&geometry).of(local_screen);
            assert!(coordinate_screen.max_abs_diff(&carried) < ROUND_OFF);
            for axis in screen_axes(coordinate_screen) {
                assert!((metric(&geometry).of(observer) & axis).s().abs() < ROUND_OFF);
                assert!((ray.momentum & axis).s().abs() < ROUND_OFF);
            }
            assert!((ray.momentum & tangent).s().abs() < ROUND_OFF);
            // The camera boost maps its rest-time direction to precisely the static observer.
            let boosted = gauge(&geometry).of(crate::kerr::camera_boost(&geometry) >> time());
            assert!(boosted.max_abs_diff(&observer) < ROUND_OFF);
        }
    }

    #[test]
    fn disk_images_and_polarization_agree_after_coordinate_transport() {
        let (hole, orientation, eye) = camera();
        let disk = Disk {
            normal: z(),
            orbit: xy(),
            inner_radius: innermost_stable_orbit(xy(), hole),
            outer_radius: 12.0,
            depth: 4.0,
        };
        let (capture, escape) = (1.68, 60.0);
        let images = crate::kerr::camera_rays(eye, orientation, [9, 7], 0.35, hole)
            .into_iter()
            .map(|ray| {
                let screen = crate::kerr::camera_screen(ray, orientation, hole);
                let local = PolarizedRay { ray, screen };
                let coordinate = PolarizedRay {
                    ray,
                    screen: gauge(&field(ray.event, hole)).of(screen),
                };
                let first = camera_trace(local, hole, 0.08, 720, crate::kerr::polarized_rates);
                let second = camera_trace(coordinate, hole, 0.08, 720, polarized_rates);
                (
                    resolve(first, &disk, escape, capture, hole).0,
                    resolve(second, &disk, escape, capture, hole).0,
                )
            })
            .collect::<Vec<_>>();
        let (first, second): (Vec<_>, Vec<_>) = images.into_iter().unzip();
        let (mut coherency_error, mut crossings) = (0.0f64, 0);
        for (a, b) in first.iter().zip(&second) {
            // The same pixels meet the disk and the sky, with the same weights.
            assert_eq!(a.disk.len(), b.disk.len());
            assert_eq!(a.sky.is_some(), b.sky.is_some());
            for (p, q) in a.disk.iter().zip(&b.disk) {
                assert!((p.weight - q.weight).abs() < 1e-9);
                let (r, s) = (p.state.ray, q.state.ray);
                let gauge_coherency = crate::disk::disk_coherency(
                    r.event,
                    r.momentum,
                    p.state.screen,
                    z(),
                    hole,
                    0.8,
                );
                let metric_coherency =
                    disk_coherency(s.event, s.momentum, q.state.screen, z(), hole, 0.8);
                coherency_error =
                    coherency_error.max(gauge_coherency.max_abs_diff(&metric_coherency));
                let gauge_shift =
                    crate::disk::frequency_ratio(r.event, r.momentum, eye, xy(), hole);
                let metric_shift = frequency_ratio(s.event, s.momentum, eye, xy(), hole);
                assert!((gauge_shift - metric_shift).abs() < 1e-9);
                crossings += 1;
            }
        }
        assert!(crossings > 20);
        // Linear interpolation to a crossing and RK4 in different variables leave a
        // finite-step error.
        assert!(coherency_error < 2e-3, "{coherency_error}");
    }

    /// The checks of numga's comparison notebook: the two Hamiltonians and ray equations agree
    /// on the same incoming momenta, the equatorial rays at spin 0.8 to 1e-8 over 400 steps,
    /// and the camera (12 x 8 pixels, as numga's test of the notebook) sees the same pixels
    /// and the same temperatures.
    #[test]
    fn comparison_notebook_checks() {
        let hole = Hole::about_z(0.8, 1.0);
        let starts: Vec<Ray> = (0..31)
            .map(|i| {
                let offset = -8.0 + 16.0 * i as f64 / 30.0;
                let event = crate::kerr::along_x() * -12.0 + crate::kerr::along_y() * offset;
                Ray {
                    event,
                    momentum: crate::kerr::initial_momentum(event, crate::kerr::along_x(), hole),
                }
            })
            .collect();
        let mut path_error = 0.0f64;
        for start in &starts {
            let geometry = field(start.event, hole);
            let gap = hamiltonian(&geometry, start.momentum)
                - crate::kerr::hamiltonian(&geometry, start.momentum);
            assert!(gap.abs() < 1e-11);
            let gauge_path = trace(*start, hole, 0.1, 400, 0.8, crate::kerr::rates);
            let metric_path = trace(*start, hole, 0.1, 400, 0.8, rates);
            for (a, b) in gauge_path.zip(metric_path) {
                path_error = path_error.max(a.event.max_abs_diff(&b.event));
            }
        }
        assert!(path_error < 1e-8, "{path_error}");

        // numga's camera at 75° from the spin axis, 19° up and down.
        let orientation = ((z() ^ x()) * -(75f64.to_radians() / 2.0)).exp();
        let eye = (orientation >> along_z()) * 28.0;
        let disk = Disk {
            normal: z(),
            orbit: xy(),
            inner_radius: innermost_stable_orbit(xy(), hole),
            outer_radius: 12.0,
            depth: 4.0,
        };
        let capture = 1.05 * hole.horizon();
        let half_view = tangent(19f64.to_radians());
        let mut temperature_error = 0.0f64;
        for ray in crate::kerr::camera_rays(eye, orientation, [12, 8], half_view, hole) {
            let screen = crate::kerr::camera_screen(ray, orientation, hole);
            let local = PolarizedRay { ray, screen };
            let coordinate = PolarizedRay {
                ray,
                screen: gauge(&field(ray.event, hole)).of(screen),
            };
            let first = camera_trace(local, hole, 0.1, 720, crate::kerr::polarized_rates);
            let second = camera_trace(coordinate, hole, 0.1, 720, polarized_rates);
            let (a, _) = resolve(first, &disk, 100.0, capture, hole);
            let (b, _) = resolve(second, &disk, 100.0, capture, hole);
            assert_eq!(a.disk.len(), b.disk.len());
            assert_eq!(a.sky.is_some(), b.sky.is_some());
            for (p, q) in a.disk.iter().zip(&b.disk) {
                let seen = |c: &crate::disk::Crossing<PolarizedRay>, shift: f64| {
                    shift * disk_temperature(c.radius, xy(), hole, 58_500.0)
                };
                let r = p.state.ray;
                let s = q.state.ray;
                let gauge_temperature = seen(
                    p,
                    crate::disk::frequency_ratio(r.event, r.momentum, eye, xy(), hole),
                );
                let metric_temperature =
                    seen(q, frequency_ratio(s.event, s.momentum, eye, xy(), hole));
                temperature_error =
                    temperature_error.max((gauge_temperature - metric_temperature).abs());
            }
        }
        // Different screen variables carry finite-step error, not the temperatures.
        assert!(temperature_error < 1e-4, "{temperature_error}");
    }
}
