//! numga's `kerr/core.py`, the disk: a thin disk of gas orbiting in the hole's equatorial
//! plane, its temperature and opacity, the frequency shift of its light, the polarization of
//! its emission, and how a camera ray shares its light between the disk, the sky and the hole.

use crate::kerr::{
    B, Hole, Map, PolarizedRay, RayState, Step, V, adjoint, field, frame, inverse_metric,
    proper_unit, screen_identity, spatial, t,
};
use gax::sta::Vector;

/// The light's coherency, a symmetric map on the screen (numga's `Screen <- Screen`), zero
/// off the screen plane.
pub type Coherency = Map;

/// Unit-intensity disk emission resolved along the transported camera axes. Emission is
/// polarized perpendicular to the magnetic direction, around the disk's `normal`, projected
/// across the ray; the polarization fades as the ray looks along the field.
pub fn disk_coherency(
    state: PolarizedRay,
    normal: V,
    hole: Hole,
    polarization_degree: f64,
) -> Coherency {
    let ray = state.ray;
    let local_momentum = frame(&field(ray.event, hole)).of(ray.momentum);
    let direction = spatial(local_momentum) * (t() | local_momentum).s().recip();
    // The magnetic direction circles the normal, square to the plane through the normal and
    // the emitter. Emission is polarized perpendicular to it across the ray: in that plane,
    // square to the ray.
    let meridian = normal ^ spatial(ray.event);
    let electric = (meridian | direction) * meridian.norm().recip();
    let polarization = adjoint(state.screen).of(electric);
    let polarized_fraction = -polarization_degree * polarization.norm_squared();
    screen_identity() * ((1.0 - polarized_fraction) / 2.0)
        - polarization * (polarization | Vector::slot()) * polarization_degree
}

/// The power a unit analyzer passes: its quadratic readout of the coherency.
pub fn transmitted(coherency: Coherency, analyzer: V) -> f64 {
    -(analyzer | coherency.of(analyzer)).s()
}

/// The frequency a camera at rest at `camera` sees, over the one emitted by gas on a
/// circular orbit in the unit plane `orbit`, for a ray whose momentum is given at the gas.
/// The momentum paired with time is conserved along the ray, so the camera's frequency
/// reads here.
pub fn frequency_ratio(event: V, momentum: V, camera: V, orbit: B, hole: Hole) -> f64 {
    let gas = orbiting_gas(event, orbit, hole);
    let at_rest = proper_unit(&field(camera, hole), t());
    (momentum | at_rest).s() / (momentum | gas).s()
}

/// The unit velocity of gas on a circular equatorial orbit in the unit plane `orbit`.
pub fn orbiting_gas(event: V, orbit: B, hole: Hole) -> V {
    let geometry = field(event, hole);
    let angular_speed = orbital_speed(geometry.radius, orbit, hole);
    let flow = t() + orbit.commutator(spatial(event)) * angular_speed;
    proper_unit(&geometry, flow)
}

/// Kepler's angular speed, per coordinate time, of a circular equatorial orbit in the unit
/// plane `orbit` at an oblate radius.
#[allow(clippy::disallowed_methods)] // Kepler's law in the Kerr geometry
pub fn orbital_speed(radius: f64, orbit: B, hole: Hole) -> f64 {
    let root = hole.mass.sqrt();
    root / (radius * radius.sqrt() + hole.along(orbit) * root)
}

/// Where the gas the camera sees at a disk event was, followed back along its orbit to
/// time zero once a coordinate time `elapsed` has passed. The event's own time counts too,
/// earlier for light that travelled further, and inner rings lap outer ones.
pub fn carried(event: V, radius: f64, orbit: B, hole: Hole, elapsed: f64) -> V {
    let emitted = elapsed + (event | t()).s();
    let angle = orbital_speed(radius, orbit, hole) * emitted;
    (orbit * (-angle / 2.0)).exp() >> event
}

/// The radius of the innermost stable circular equatorial orbit in the unit plane `orbit`:
/// 6M without spin, M at the extreme (Bardeen, Press and Teukolsky, 1972).
#[allow(clippy::disallowed_methods)] // Bardeen, Press and Teukolsky's radius: a law
pub fn innermost_stable_orbit(orbit: B, hole: Hole) -> f64 {
    let ratio = hole.along(orbit) / hole.mass;
    let first = 1.0 + (1.0 - ratio * ratio).cbrt() * ((1.0 + ratio).cbrt() + (1.0 - ratio).cbrt());
    let second = (3.0 * ratio * ratio + first * first).sqrt();
    hole.mass * (3.0 + second - ((3.0 - first) * (3.0 + first + 2.0 * second)).sqrt())
}

/// A relativistic thin disk's temperature at an oblate radius, from the flux of Page and
/// Thorne (1974), for gas orbiting in the unit plane `orbit`: zero at the innermost stable
/// orbit, falling as the radius to the power -3/4 far out. `scale` is
/// `(accretion rate c⁶ / (σ G² M²))^(1/4)`, K.
#[allow(clippy::disallowed_methods)] // Page and Thorne's flux: a physical law
pub fn disk_temperature(radius: f64, orbit: B, hole: Hole, scale: f64) -> f64 {
    let ratio = hole.along(orbit) / hole.mass;
    let x = (radius / hole.mass).sqrt();
    let edge = (innermost_stable_orbit(orbit, hole) / hole.mass).sqrt();
    // The roots of x³ − 3x + 2 ratio, by trisecting the angle, and the weight of each in
    // the torque; the product of a root's differences from the other two is the cubic's
    // derivative there, 3 root² − 3.
    let pi = core::f64::consts::PI;
    let torque = [-pi, pi, 3.0 * pi]
        .into_iter()
        .map(|shift| 2.0 * ((ratio.acos() + shift) / 3.0).cos())
        .fold(x - edge - 1.5 * ratio * (x / edge).ln(), |torque, root| {
            let weight = (root - ratio).powi(2) / (root * (root * root - 1.0));
            torque - weight * ((x - root) / (edge - root)).ln()
        });
    // The torque the disk carries, zero at the inner edge; round-off kept non-negative.
    let flux =
        3.0 / (8.0 * pi) * torque.max(0.0) / (x.powi(4) * (x.powi(3) - 3.0 * x + 2.0 * ratio));
    scale * flux.powf(0.25)
}

/// The disk's optical depth straight through its thickness at an oblate radius: `depth`
/// at the inner edge, where the gas is densest, thinning to none at the outer edge.
pub fn disk_depth(radius: f64, inner_radius: f64, outer_radius: f64, depth: f64) -> f64 {
    let fraction = (radius - inner_radius) / (outer_radius - inner_radius);
    depth * (1.0 - fraction) * (1.0 - fraction)
}

/// The share of a ray's light the disk absorbs where the ray crosses it. Its optical depth
/// along the ray is the depth through its thickness over the cosine of the ray's slant as
/// the orbiting gas sees it: the ray's energy in the gas's frame over its momentum along
/// the normal. Gas moving away from the light's source sees the ray more obliquely.
pub fn disk_opacity(event: V, momentum: V, depth: f64, normal: V, orbit: B, hole: Hole) -> f64 {
    let gas = orbiting_gas(event, orbit, hole);
    let slant = ((momentum | gas).s() / (momentum | normal).s()).abs();
    1.0 - (-depth * slant).exp()
}

/// How far along a segment from `start` to `end` it crosses the plane normal to `normal`.
pub fn crossing_fraction(start: V, end: V, normal: V) -> f64 {
    -(start | normal).s() / ((end - start) | normal).s()
}

/// A ray's state where its step from `start` to `end` crosses the central plane normal to
/// `normal`, and its oblate radius there.
pub fn disk_crossing<S: RayState>(start: S, end: S, normal: V, hole: Hole) -> (S, f64) {
    let fraction = crossing_fraction(start.ray().event, end.ray().event, normal);
    let state = start * (1.0 - fraction) + end * fraction;
    (state, field(state.ray().event, hole).radius)
}

/// A thin disk in the central plane normal to `normal`, its gas orbiting in the plane
/// `orbit`, between two oblate radii; `depth` is its optical depth straight through its
/// thickness at the inner edge.
#[derive(Clone, Copy, Debug)]
pub struct Disk {
    pub normal: V,
    pub orbit: B,
    pub inner_radius: f64,
    pub outer_radius: f64,
    pub depth: f64,
}

impl Disk {
    /// Whether the disk reaches an oblate radius.
    fn spans(&self, radius: f64) -> bool {
        (self.inner_radius..=self.outer_radius).contains(&radius)
    }

    /// The share of a ray's light the disk absorbs where the ray crosses it with
    /// `momentum` at `event`, at oblate radius `radius`.
    pub fn opacity(&self, event: V, momentum: V, radius: f64, hole: Hole) -> f64 {
        let depth = disk_depth(radius, self.inner_radius, self.outer_radius, self.depth);
        disk_opacity(event, momentum, depth, self.normal, self.orbit, hole)
    }
}

/// A crossing of the disk: the ray's state there, the oblate radius, and the share of the
/// pixel's light the crossing gives.
#[derive(Clone, Copy, Debug)]
pub struct Crossing<S> {
    pub state: S,
    pub radius: f64,
    pub weight: f64,
}

/// An escaped ray: its direction, and the share of its pixel's light left for the sky.
#[derive(Clone, Copy, Debug)]
pub struct Escape {
    pub direction: V,
    pub weight: f64,
}

/// What a camera ray met: its crossings of the disk, in order, and its escape, if it
/// escaped. A pixel seen through the disk's clear edge has more than one crossing.
#[derive(Clone, Debug)]
pub struct Resolution<S> {
    pub disk: Vec<Crossing<S>>,
    pub sky: Option<Escape>,
}

/// What a camera ray met, from its steps. Each crossing of the disk takes the disk's
/// opacity times the share of the ray not yet absorbed. The ray ends when absorbed,
/// captured inside `capture_radius`, or escaped beyond `escape_radius`, its direction then
/// its tangent, the inverse metric of its momentum; one that runs out of steps ends there.
pub fn resolve<S: RayState>(
    steps: impl Iterator<Item = Step<S>>,
    disk: &Disk,
    escape_radius: f64,
    capture_radius: f64,
    hole: Hole,
) -> Resolution<S> {
    let mut resolution = Resolution {
        disk: Vec::new(),
        sky: None,
    };
    let mut remaining = 1.0;
    for step in steps {
        if step.crosses(disk.normal) {
            let (state, radius) = disk_crossing(step.start, step.end, disk.normal, hole);
            if disk.spans(radius) {
                let at = state.ray();
                let weight = remaining * disk.opacity(at.event, at.momentum, radius, hole);
                remaining -= weight;
                resolution.disk.push(Crossing {
                    state,
                    radius,
                    weight,
                });
            }
        }
        if remaining <= 0.0 || step.radius < capture_radius {
            break;
        }
        if step.radius > escape_radius {
            let end = step.end.ray();
            let direction = inverse_metric(&field(end.event, hole)).of(end.momentum);
            resolution.sky = Some(Escape {
                direction,
                weight: remaining,
            });
            break;
        }
    }
    resolution
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::kerr::tests::ROUND_OFF;
    use crate::kerr::{
        Ray, camera_rates, camera_rays, camera_screen, camera_trace, initial_momentum, x, xy, y, z,
    };
    use gax_numga_examples::signal::wave;

    /// The share of each pixel's light the disk and the sky take.
    pub fn shares<S>(resolutions: &[Resolution<S>]) -> Vec<(f64, f64)> {
        resolutions
            .iter()
            .map(|r| {
                let disk = r.disk.iter().map(|c| c.weight).sum();
                (disk, r.sky.map_or(0.0, |e| e.weight))
            })
            .collect()
    }

    /// numga's small camera: 9 x 7 pixels over a dense disk.
    fn small_camera() -> Vec<Resolution<Ray>> {
        let hole = Hole::about_z(0.7, 1.0);
        let orientation = ((z() ^ x()) * 0.6).exp();
        let eye = (orientation >> z()) * 25.0;
        let disk = Disk {
            normal: z(),
            orbit: xy(),
            inner_radius: 3.4,
            outer_radius: 12.0,
            depth: 40.0,
        };
        let capture_radius = 1.05 * hole.horizon();
        camera_rays(eye, orientation, [9, 7], 0.35, hole)
            .into_iter()
            .map(|ray| {
                let steps = camera_trace(ray, hole, 0.2, 300, camera_rates);
                resolve(steps, &disk, 100.0, capture_radius, hole)
            })
            .collect()
    }

    #[test]
    fn disk_coherency_through_a_rotating_polarizer() {
        let position = x() * 6.0;
        let hole = Hole::about_z(0.7, 0.0);
        let orientation = (xy() * 0.0).exp();
        let ray = Ray {
            event: position,
            momentum: -initial_momentum(position, z(), hole),
        };
        let state = PolarizedRay {
            ray,
            screen: camera_screen(ray, orientation, hole),
        };
        let analyzer = |a: f64| (xy() * (a / 2.0)).exp() >> x();
        for degree in [0.0, 0.6, 1.0] {
            let coherency = disk_coherency(state, z(), hole, degree);
            // An azimuthal magnetic field and a normal ray give maximal linear polarization.
            assert!((coherency.trace() - 1.0).abs() < ROUND_OFF);
            for k in 0..17 {
                let angle = core::f64::consts::PI * k as f64 / 16.0;
                let passed = transmitted(coherency, analyzer(angle));
                let expected = (1.0 - degree) / 2.0 + degree * wave(angle).powi(2);
                assert!((passed - expected).abs() < ROUND_OFF);
                let opposite = transmitted(coherency, analyzer(angle + core::f64::consts::PI));
                assert!((opposite - passed).abs() < ROUND_OFF);
            }
        }
    }

    #[test]
    fn every_camera_ray_shares_its_light_once_between_disk_sky_and_hole() {
        let image = small_camera();
        // Crossings lie in the disk plane between its radii, each giving part of its pixel's
        // light.
        for crossing in image.iter().flat_map(|r| &r.disk) {
            assert!((z() | crossing.state.event).s().abs() < ROUND_OFF);
            assert!(3.4 < crossing.radius && crossing.radius < 12.0);
            assert!(crossing.weight > 0.0);
        }
        // A ray escapes at most once (its escape is one value); escaped rays share all their
        // light, captured ones lose the rest.
        let shares = shares(&image);
        for ((disk, sky), r) in shares.iter().zip(&image) {
            assert!(disk + sky <= 1.0 + ROUND_OFF);
            if r.sky.is_some() {
                assert!((disk + sky - 1.0).abs() < ROUND_OFF);
            }
        }
        // Some rays meet the disk where it clears, and its body absorbs others all but whole.
        assert!(shares.iter().any(|(disk, _)| 0.0 < *disk && *disk < 0.5));
        assert!(shares.iter().any(|(disk, _)| *disk > 0.999));
        // The central ray, aimed at the hole, passes inside the disk's inner edge and is
        // captured.
        assert_eq!(shares[shares.len() / 2], (0.0, 0.0));
    }

    #[test]
    #[allow(clippy::disallowed_methods)] // a reference direction at 45°
    fn disk_opacity_face_on_and_where_its_gas_recedes() {
        let (hole, depth) = (Hole::about_z(0.8, 1.0), 0.7);
        let (far, near) = (1e6, 6.0);
        // Far out the gas barely moves and the field is flat: a ray along the normal meets the
        // bare depth.
        let face_on = initial_momentum(x() * far, -z(), hole);
        let straight = disk_opacity(x() * far, face_on, depth, z(), xy(), hole);
        // This tolerance tests the far field's residual orbital speed and curvature, about 1e-6.
        assert!((straight - (1.0 - (-depth).exp())).abs() < 1e-4);
        // Two rays at one slant through opposite sides of the orbit: the gas at +x moves along
        // +y, towards where the light goes, and the gas at -x away from it.
        let slant = (y() - z()).normalized();
        let opacity = |side: V| {
            let momentum = initial_momentum(side, slant, hole);
            disk_opacity(side, momentum, depth, z(), xy(), hole)
        };
        assert!(opacity(-x() * near) > opacity(x() * near));
    }

    #[test]
    #[allow(clippy::disallowed_methods)] // the analytic reference it is checked against
    fn frequency_ratio_of_orbiting_gas_seen_by_a_camera_at_rest() {
        let (radius, distance) = (8.0, 1e6);
        let position = x() * radius;
        let still = Hole::about_z(0.0, 1.0);
        let radial = initial_momentum(position, x(), still);
        let schwarzschild = frequency_ratio(position, radial, x() * distance, xy(), still);
        let empty = Hole::about_z(0.0, 0.0);
        let flat = frequency_ratio(position, radial, x() * distance, xy(), empty);
        // A radial photon from a circular Schwarzschild orbit: time dilation of the orbit and
        // of the well.
        let expected = (1.0 - 3.0 / radius).sqrt() / (1.0 - 2.0 / distance).sqrt();
        assert!((schwarzschild - expected).abs() < ROUND_OFF);
        assert!((flat - 1.0).abs() < ROUND_OFF);
        // Gas coming towards the camera is blueshifted, gas going away redshifted.
        let spinning = Hole::about_z(0.8, 1.0);
        let seen = |heading: V| {
            let momentum = initial_momentum(position, heading, spinning);
            frequency_ratio(position, momentum, heading * distance, xy(), spinning)
        };
        assert!(seen(y()) > 1.0 && 1.0 > seen(-y()));
    }

    #[test]
    #[allow(clippy::disallowed_methods)] // Page and Thorne's Schwarzschild flux, the reference
    fn relativistic_thin_disk_temperature() {
        let nearly_still = Hole::about_z(1e-9, 1.0);
        let (root3, root6) = (3f64.sqrt(), 6f64.sqrt());
        for radius in [6.5f64, 8.0, 12.0, 40.0] {
            let root = radius.sqrt();
            let logarithm =
                ((root + root3) * (root6 - root3) / ((root - root3) * (root6 + root3))).ln();
            let schwarzschild =
                (3.0 / (8.0 * core::f64::consts::PI * radius.powi(3)) / (1.0 - 3.0 / radius)
                    * (1.0 - (6.0 / radius).sqrt() + (3.0 / (4.0 * radius)).sqrt() * logarithm))
                    .powf(0.25);
            // Without spin, the Kerr flux is Page and Thorne's Schwarzschild one.
            let kerr = disk_temperature(radius, xy(), nearly_still, 1.0);
            assert!((kerr - schwarzschild).abs() <= 1e-6 * schwarzschild);
        }
        // Zero at the innermost stable orbit; a faster spin reaches deeper and runs hotter.
        let fast = Hole::about_z(0.8, 1.0);
        assert!(disk_temperature(innermost_stable_orbit(xy(), fast), xy(), fast, 1.0) < 1e-3);
        let peak = |spin: f64| {
            let hole = Hole::about_z(spin, 1.0);
            let inner = innermost_stable_orbit(xy(), hole);
            // Geometrically spaced radii from the inner edge to 30 M.
            (0..2000)
                .map(|k| inner * (30.0 / inner).powf(k as f64 / 1999.0))
                .map(|r| disk_temperature(r, xy(), hole, 1.0))
                .fold(0.0, f64::max)
        };
        assert!(peak(0.5) < peak(0.8) && peak(0.8) < peak(0.95));
    }

    #[test]
    fn innermost_stable_orbit_at_rest_spinning_and_extreme() {
        // 6M without spin, about 2.91M at spin 0.8M, and the hole's own mass at the extreme.
        let radius = |spin: f64| innermost_stable_orbit(xy(), Hole::about_z(spin, 1.0));
        assert!((radius(0.8) - 2.9066).abs() < 1e-4);
        assert!((radius(0.0) - 6.0).abs() < ROUND_OFF);
        assert!((radius(1.0) - 1.0).abs() < ROUND_OFF);
    }

    /// The disk's numbers against numga's (numga 93c2ae3): the innermost stable orbit, a
    /// temperature and a frequency ratio at spin 0.8, and the small camera's disk shares at
    /// two pixels where the disk clears.
    #[test]
    fn disk_agrees_with_numga() {
        let spinning = Hole::about_z(0.8, 1.0);
        let inner = innermost_stable_orbit(xy(), spinning);
        assert!((inner - 2.906643854464197).abs() < ROUND_OFF);
        let temperature = disk_temperature(6.0, xy(), spinning, 58_500.0);
        assert!((temperature - 6046.864224565353).abs() < 1e-8);
        let toward = initial_momentum(x() * 8.0, y(), spinning);
        let ratio = frequency_ratio(x() * 8.0, toward, y() * 1e6, xy(), spinning);
        assert!((ratio - 1.30501309975741).abs() < ROUND_OFF);
        let shares = shares(&small_camera());
        assert!((shares[3].0 - 0.464996544063331).abs() < 1e-9);
        assert!((shares[45].0 - 0.171896945778985).abs() < 1e-9);
    }
}
