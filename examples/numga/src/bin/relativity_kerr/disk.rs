//! numga's `kerr/core.py`, the disk: a thin disk of gas orbiting in the hole's equatorial
//! plane, its temperature and opacity, the frequency shift of its light, the polarization of
//! its emission, and how a camera ray shares its light between the disk, the sky and the hole.

use crate::kerr::{
    B, D, Field, Hole, RayState, ScreenFrame, Step, V, field, local, proper_squared, rates, read,
    space, squared, t, time, x, y,
};
use gax::sta::{Vector, intrinsic};

/// The light's polarization covariance, from the slices that read the screen to screen
/// directions (numga's `Screen <- ScreenSlice`): a map from vectors that reads only their `x`
/// and `y` parts, to directions with only `X` and `Y` parts.
pub type Coherency = intrinsic::Trivector<(V,)>;

/// The camera's slices `x` and `y`, which read screen directions (numga's `ScreenSlice`): the
/// projection of a vector onto them.
pub fn screen_slice() -> Vector<(V,)> {
    x() * (x() | V::slot()) + y() * (y() | V::slot())
}

/// Unit-intensity disk emission resolved along the transported camera axes; `normal` is the
/// disk's plane, the slice that measures height above it.
#[allow(clippy::disallowed_methods)] // a degree of polarization: the square of its share
pub fn disk_coherency(
    event: D,
    momentum: V,
    screen: ScreenFrame,
    normal: V,
    hole: Hole,
    polarization_degree: f64,
) -> Coherency {
    let heading = local(&field(event, hole), momentum);
    let direction = space(heading) * read(t(), heading).recip();
    // The magnetic direction circles the normal, square to the plane through the normal and
    // the emitter. Emission is polarized perpendicular to it across the ray: in that plane,
    // square to the ray.
    let meridian = space(event).hodge() ^ normal;
    let polarization_axis = meridian.commutator(direction) * meridian.norm().recip();
    let polarization = screen.adjoint().of(polarization_axis);
    // Polarization vanishes smoothly when looking along the magnetic direction.
    let polarized_fraction = polarization_degree * squared(polarization);
    // Unpolarized light is round, where the metric enters; the polarized part weights the
    // polarization by each slice's reading of it.
    screen_slice().hodge() * ((1.0 - polarized_fraction) / 2.0)
        + polarization * (screen_slice() & polarization) * polarization_degree
}

/// The power a unit analyzer passes, an analyzer being the slice that reads the screen
/// across its axis's normal: its reading of the coherency's image of itself.
pub fn transmitted(coherency: Coherency, analyzer: V) -> f64 {
    (analyzer & coherency.of(analyzer)).s()
}

/// The frequency a camera at rest at `camera` sees, over the one emitted by gas on a
/// circular orbit in the unit plane `orbit`, for a ray whose momentum is given at the gas.
/// The momentum paired with time is conserved along the ray, so the camera's frequency
/// reads here.
#[allow(clippy::disallowed_methods)] // a proper length: the square root of its square
pub fn frequency_ratio(event: D, momentum: V, camera: D, orbit: B, hole: Hole) -> f64 {
    let gas = orbiting_gas(event, orbit, hole);
    let at_rest = time() * proper_squared(&field(camera, hole), time()).sqrt().recip();
    (momentum & at_rest).s() / (momentum & gas).s()
}

/// The direction gas on a circular equatorial orbit in the unit plane `orbit` moves in per
/// unit of coordinate time, at an event.
pub fn gas_flow(event: D, geometry: &Field, orbit: B, hole: Hole) -> D {
    let angular_speed = orbital_speed(geometry.radius, orbit, hole);
    // A point turned by `exp(−½ angle orbit)` moves by its commutator with the orbit's plane.
    time() + space(event).commutator(orbit) * angular_speed
}

/// The unit velocity of gas on a circular equatorial orbit in the unit plane `orbit`.
#[allow(clippy::disallowed_methods)] // a proper length: the square root of its square
pub fn orbiting_gas(event: D, orbit: B, hole: Hole) -> D {
    let geometry = field(event, hole);
    let flow = gas_flow(event, &geometry, orbit, hole);
    flow * proper_squared(&geometry, flow).sqrt().recip()
}

/// Kepler's angular speed, per coordinate time, of a circular equatorial orbit in the unit
/// plane `orbit` at an oblate radius, the spin counted along the orbit.
#[allow(clippy::disallowed_methods)] // Kepler's law in the Kerr geometry
pub fn orbital_speed(radius: f64, orbit: B, hole: Hole) -> f64 {
    let root = hole.mass.sqrt();
    root / (radius * radius.sqrt() + hole.along(orbit) * root)
}

/// The gas the camera sees at a disk event once a coordinate time `elapsed` has passed,
/// followed back along its orbit to time zero. The event's own time counts too, earlier for
/// light that travelled further, and inner rings lap outer ones; the clock starts when light
/// leaves the hole's centre for the camera at its `distance`, so that moving the camera does
/// not turn the disk.
pub fn carried(event: D, radius: f64, orbit: B, hole: Hole, distance: f64, elapsed: f64) -> D {
    let emitted = elapsed + read(t(), event) + distance;
    let angle = orbital_speed(radius, orbit, hole) * emitted;
    (orbit * (angle / 2.0)).exp() >> event
}

/// The radius of the innermost stable circular equatorial orbit in the unit plane `orbit`,
/// the spin counted along the orbit, negative for gas orbiting against the hole: 6M without
/// spin, and at the extreme M with the hole and 9M against it (Bardeen, Press and Teukolsky,
/// 1972).
#[allow(clippy::disallowed_methods)] // Bardeen, Press and Teukolsky's radius: a law
pub fn innermost_stable_orbit(orbit: B, hole: Hole) -> f64 {
    let ratio = hole.along(orbit) / hole.mass;
    let first = 1.0 + (1.0 - ratio * ratio).cbrt() * ((1.0 + ratio).cbrt() + (1.0 - ratio).cbrt());
    let second = (3.0 * ratio * ratio + first * first).sqrt();
    // Gas orbiting with the hole reaches inward of the radius without spin, gas orbiting
    // against it outward.
    let reach = ((3.0 - first) * (3.0 + first + 2.0 * second)).sqrt();
    hole.mass * (3.0 + second - if ratio < 0.0 { -reach } else { reach })
}

/// A relativistic thin disk's temperature at an oblate radius, from the flux of Page and
/// Thorne (1974), for gas orbiting in the unit plane `orbit`, the spin counted along the
/// orbit: zero at the innermost stable orbit, falling as the radius to the power -3/4 far
/// out. `scale` is `(accretion rate c⁶ / (σ G² M²))^(1/4)`, K.
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
    scale * flux.sqrt().sqrt()
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
/// the normal. Gas moving away from the light's source sees the ray more obliquely, and
/// absorbs more.
pub fn disk_opacity(event: D, momentum: V, depth: f64, normal: V, orbit: B, hole: Hole) -> f64 {
    let gas = orbiting_gas(event, orbit, hole);
    let slant = ((momentum & gas).s() / (momentum & normal.hodge()).s()).abs();
    1.0 - (-depth * slant).exp()
}

/// How far along a segment from `start` to `end` it crosses the plane `normal`, the slice of
/// height above it.
pub fn crossing_fraction(start: D, end: D, normal: V) -> f64 {
    -read(normal, start) / read(normal, end - start)
}

/// A ray's state where its step from `start` to `end` crosses the central plane `normal`,
/// and its oblate radius there.
pub fn disk_crossing<S: RayState>(start: S, end: S, normal: V, hole: Hole) -> (S, f64) {
    let fraction = crossing_fraction(start.ray().event, end.ray().event, normal);
    let state = start * (1.0 - fraction) + end * fraction;
    (state, field(state.ray().event, hole).radius)
}

/// The crossings of the disk a camera ray keeps: its direct, secondary and tertiary images,
/// and one more.
pub const CROSSINGS: usize = 4;

/// A thin disk in the central plane `normal`, its gas orbiting in the plane `orbit`, between
/// two oblate radii; `depth` is its optical depth straight through its thickness at the inner
/// edge.
#[derive(Clone, Copy, Debug)]
pub struct Disk {
    pub normal: V,
    pub orbit: B,
    pub inner_radius: f64,
    pub outer_radius: f64,
    pub depth: f64,
}

impl Disk {
    /// No disk: one in the plane `normal` that takes none of the light, out to `outer_radius`.
    #[cfg(test)]
    pub fn clear(normal: V, outer_radius: f64) -> Disk {
        Disk {
            normal,
            orbit: crate::kerr::xy(),
            inner_radius: 0.0,
            outer_radius,
            depth: 0.0,
        }
    }

    /// Whether the disk reaches an oblate radius.
    fn spans(&self, radius: f64) -> bool {
        (self.inner_radius..=self.outer_radius).contains(&radius)
    }

    /// The share of a ray's light the disk absorbs where the ray crosses it with
    /// `momentum` at `event`, at oblate radius `radius`.
    pub fn opacity(&self, event: D, momentum: V, radius: f64, hole: Hole) -> f64 {
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
    pub direction: D,
    pub weight: f64,
}

/// What a camera ray met: its crossings of the disk, in order, and its escape, if it
/// escaped. A pixel seen through the disk's clear edge has more than one crossing.
#[derive(Clone, Debug)]
pub struct Resolution<S> {
    pub disk: Vec<Crossing<S>>,
    pub sky: Option<Escape>,
}

/// What a camera ray met, from its steps, and the state it stopped in. Each crossing of the
/// disk takes the disk's opacity times the share of the ray not yet absorbed, and is kept
/// as the next of its [`CROSSINGS`]. The ray ends when absorbed, captured inside
/// `capture_radius`, or escaped beyond `escape_radius`, its direction then its tangent; one
/// that runs out of steps ends there.
pub fn resolve<S: RayState>(
    steps: impl Iterator<Item = Step<S>>,
    disk: &Disk,
    escape_radius: f64,
    capture_radius: f64,
    hole: Hole,
) -> (Resolution<S>, S) {
    let mut resolution = Resolution {
        disk: Vec::new(),
        sky: None,
    };
    let (mut remaining, mut filled, mut last) = (1.0, 0, None);
    for step in steps {
        last = Some(step.end);
        if step.crosses(disk.normal) {
            let (state, radius) = disk_crossing(step.start, step.end, disk.normal, hole);
            if disk.spans(radius) {
                let at = state.ray();
                let weight = remaining * disk.opacity(at.event, at.momentum, radius, hole);
                remaining -= weight;
                if filled < CROSSINGS && weight > 0.0 {
                    resolution.disk.push(Crossing {
                        state,
                        radius,
                        weight,
                    });
                }
                filled += 1;
            }
        }
        if remaining > 0.0 && step.radius > escape_radius {
            resolution.sky = Some(Escape {
                direction: rates(step.end.ray(), hole).event,
                weight: remaining,
            });
        }
        if remaining <= 0.0 || step.radius < capture_radius || step.radius > escape_radius {
            break;
        }
    }
    (resolution, last.expect("a ray takes a step"))
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::kerr::tests::ROUND_OFF;
    use crate::kerr::{
        Ray, along_x, along_y, along_z, camera_rates, camera_rays, camera_screen, camera_trace,
        initial_momentum, xy, z,
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
        let eye = (orientation >> along_z()) * 25.0;
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
                resolve(steps, &disk, 100.0, capture_radius, hole).0
            })
            .collect()
    }

    #[test]
    fn disk_coherency_through_a_rotating_polarizer() {
        let position = along_x() * 6.0;
        let hole = Hole::about_z(0.7, 0.0);
        let orientation = (xy() * 0.0).exp();
        let ray = Ray {
            event: position,
            momentum: -initial_momentum(position, along_z(), hole),
        };
        let screen = camera_screen(ray, orientation, hole);
        let analyzer = |a: f64| (xy() * (a / 2.0)).exp() >> x();
        for degree in [0.0, 0.6, 1.0] {
            let coherency = disk_coherency(ray.event, ray.momentum, screen, z(), hole, degree);
            // An azimuthal magnetic field and a normal ray give maximal linear polarization.
            // The intensity: the coherency's form traced against the metric on the camera's
            // slices.
            let intensity = transmitted(coherency, x()) + transmitted(coherency, y());
            assert!((intensity - 1.0).abs() < ROUND_OFF);
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
            assert!(read(z(), crossing.state.event).abs() < ROUND_OFF);
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
        let face_on = initial_momentum(along_x() * far, -along_z(), hole);
        let straight = disk_opacity(along_x() * far, face_on, depth, z(), xy(), hole);
        // This tolerance tests the far field's residual orbital speed and curvature, about 1e-6.
        assert!((straight - (1.0 - (-depth).exp())).abs() < 1e-4);
        // Two rays at one slant through opposite sides of the orbit: the gas at +x moves along
        // +y, towards where the light goes, and the gas at -x away from it.
        let slant = (along_y() - along_z()) * 0.5f64.sqrt();
        let opacity = |side: D| {
            let momentum = initial_momentum(side, slant, hole);
            disk_opacity(side, momentum, depth, z(), xy(), hole)
        };
        assert!(opacity(-along_x() * near) > opacity(along_x() * near));
    }

    #[test]
    #[allow(clippy::disallowed_methods)] // the analytic reference it is checked against
    fn frequency_ratio_of_orbiting_gas_seen_by_a_camera_at_rest() {
        let (radius, distance) = (8.0, 1e6);
        let position = along_x() * radius;
        let still = Hole::about_z(0.0, 1.0);
        let radial = initial_momentum(position, along_x(), still);
        let camera = along_x() * distance;
        let schwarzschild = frequency_ratio(position, radial, camera, xy(), still);
        let flat = frequency_ratio(position, radial, camera, xy(), Hole::about_z(0.0, 0.0));
        // A radial photon from a circular Schwarzschild orbit: time dilation of the orbit and
        // of the well.
        let expected = (1.0 - 3.0 / radius).sqrt() / (1.0 - 2.0 / distance).sqrt();
        assert!((schwarzschild - expected).abs() < ROUND_OFF);
        assert!((flat - 1.0).abs() < ROUND_OFF);
        // Gas coming towards the camera is blueshifted, gas going away redshifted.
        let spinning = Hole::about_z(0.8, 1.0);
        let seen = |heading: D| {
            let momentum = initial_momentum(position, heading, spinning);
            frequency_ratio(position, momentum, heading * distance, xy(), spinning)
        };
        assert!(seen(along_y()) > 1.0 && 1.0 > seen(-along_y()));
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
        assert!((radius(0.0) - 6.0).abs() < 1e-4);
        assert!((radius(0.8) - 2.9066).abs() < 1e-4);
        assert!((radius(1.0) - 1.0).abs() < 1e-4);
        assert!((radius(0.0) - 6.0).abs() < ROUND_OFF);
        assert!((radius(1.0) - 1.0).abs() < ROUND_OFF);
    }

    /// The disk's numbers against numga's (numga 35797de): the innermost stable orbit with
    /// and against a spin of 0.8, a temperature and a frequency ratio, and the small camera's
    /// disk and sky shares at every pixel.
    #[test]
    fn disk_agrees_with_numga() {
        let spinning = Hole::about_z(0.8, 1.0);
        let inner = innermost_stable_orbit(xy(), spinning);
        assert!((inner - 2.9066438544641966).abs() < ROUND_OFF);
        let against = innermost_stable_orbit(xy(), Hole::about_z(-0.8, 1.0));
        assert!((against - 8.43175783080649).abs() < ROUND_OFF);
        let temperature = disk_temperature(6.0, xy(), spinning, 58_500.0);
        assert!((temperature - 6046.864224565353).abs() < 1e-8);
        let toward = initial_momentum(along_x() * 8.0, along_y(), spinning);
        let ratio = frequency_ratio(along_x() * 8.0, toward, along_y() * 1e6, xy(), spinning);
        assert!((ratio - 1.30501309975741).abs() < ROUND_OFF);
        let shares = shares(&small_camera());
        let (disk, sky): (Vec<f64>, Vec<f64>) = shares.into_iter().unzip();
        for (got, expected) in [(&disk, &NUMGA_DISK_SHARES), (&sky, &NUMGA_SKY_SHARES)] {
            for (g, e) in got.iter().zip(expected.iter()) {
                assert!((g - e).abs() < 1e-9, "{g} {e}");
            }
        }
    }

    /// The small camera's shares in numga 35797de, pixel by pixel.
    #[rustfmt::skip]
    const NUMGA_DISK_SHARES: [f64; 63] = [0.0, 0.0, 0.0, 0.006609799843241393, 0.024161712331960628, 0.006129044211533263, 0.0, 0.0, 0.0, 0.6251527576508324, 0.9749865784748378, 0.9994242609974684, 1.0, 0.9998712083627999, 0.999999999987583, 0.9932398412585313, 0.9149703754354435, 0.45361456347406537, 0.999999095704686, 0.999999999999987, 1.0, 1.0, 1.0, 1.0, 0.999999999999984, 0.9999999628828334, 0.9992517363424235, 0.9999999999998573, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.999999999999989, 0.9999961397488587, 0.9999768620582494, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.999999986765525, 0.8753022121435485, 0.0, 0.9999047089658132, 1.0, 1.0, 0.9999999999996396, 0.9999999999999745, 0.9999993670355303, 0.22754647223639735, 0.0, 0.0, 0.0, 0.5271702360163106, 0.9997828890915788, 0.99992610384546, 0.9550603707478001, 0.0, 0.0, 0.0];
    #[rustfmt::skip]
    const NUMGA_SKY_SHARES: [f64; 63] = [1.0, 1.0, 1.0, 0.9933902001567586, 0.9758382876680394, 0.9938709557884667, 1.0, 1.0, 1.0, 0.37484724234916755, 0.02501342152516217, 0.0005757390025316468, 0.0, 0.00012879163720014297, 1.2417007404896184e-11, 0.006760158741468736, 0.08502962456455654, 0.5463854365259346, 9.042953139637433e-07, 1.2989609388114332e-14, 0.0, 0.0, 0.0, 0.0, 1.5987211554602254e-14, 3.7117166562516957e-08, 0.00074826365757652, 1.4266365866433262e-13, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.099120794378905e-14, 3.860251141296622e-06, 2.313794175057282e-05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.323447496037744e-08, 0.12469778785645147, 1.0, 9.529103418681739e-05, 0.0, 0.0, 3.603783937933258e-13, 2.55351295663786e-14, 6.329644697267156e-07, 0.7724535277636027, 1.0, 1.0, 1.0, 0.47282976398368937, 0.00021711090842124747, 7.389615454000964e-05, 0.04493962925219985, 1.0, 1.0, 1.0];

    /// The small camera's first ray and screen, and numga's polarized camera (the comparison
    /// notebook's, 9 x 7 pixels, steps of 0.08 of the radius) against numga's numbers (numga
    /// 35797de): every crossing's pixel, weight, power through an analyzer and frequency ratio.
    #[test]
    fn polarized_camera_agrees_with_numga() {
        use crate::kerr::tests::{directions, slices};
        use crate::kerr::{PolarizedRay, polarized_rates};
        let hole = Hole::about_z(0.7, 1.0);
        let orientation = ((z() ^ x()) * 0.6).exp();
        let eye = (orientation >> along_z()) * 25.0;
        let first = camera_rays(eye, orientation, [9, 7], 0.35, hole)[0];
        let momentum = [
            -1.0,
            -0.8861711106906873,
            0.3763221713761683,
            0.644714341351133,
        ];
        let screen = camera_screen(first, orientation, hole);
        let right = [
            -0.01390431629974506,
            0.31536403129761326,
            0.932425838231708,
            -0.17697715664282426,
        ];
        let up = [
            0.011233551006663175,
            -0.5988964188423295,
            0.050681059420401575,
            -0.7993001328550364,
        ];
        for k in 0..4 {
            assert!(((first.momentum & directions()[k]).s() - momentum[k]).abs() < ROUND_OFF);
            assert!((read(slices()[k], screen.of(along_x())) - right[k]).abs() < ROUND_OFF);
            assert!((read(slices()[k], screen.of(along_y())) - up[k]).abs() < ROUND_OFF);
        }

        let hole = Hole::about_z(0.8, 1.0);
        let orientation = ((z() ^ x()) * -0.6).exp();
        let eye = (orientation >> along_z()) * 28.0;
        let disk = Disk {
            normal: z(),
            orbit: xy(),
            inner_radius: innermost_stable_orbit(xy(), hole),
            outer_radius: 12.0,
            depth: 4.0,
        };
        let analyzer = (xy() * (-0.3 / 2.0)).exp() >> x();
        let crossings: Vec<_> = camera_rays(eye, orientation, [9, 7], 0.35, hole)
            .into_iter()
            .enumerate()
            .flat_map(|(pixel, ray)| {
                let start = PolarizedRay {
                    ray,
                    screen: camera_screen(ray, orientation, hole),
                };
                let steps = camera_trace(start, hole, 0.08, 720, polarized_rates);
                let (image, _) = resolve(steps, &disk, 60.0, 1.68, hole);
                image.disk.into_iter().map(move |c| (pixel, c))
            })
            .collect();
        assert_eq!(crossings.len(), NUMGA_PIXELS.len());
        for (k, (pixel, crossing)) in crossings.iter().enumerate() {
            let PolarizedRay { ray, screen } = crossing.state;
            let coherency = disk_coherency(ray.event, ray.momentum, screen, z(), hole, 0.8);
            let shift = frequency_ratio(ray.event, ray.momentum, eye, xy(), hole);
            assert_eq!(*pixel, NUMGA_PIXELS[k]);
            assert!((crossing.weight - NUMGA_WEIGHTS[k]).abs() < 1e-9);
            assert!((transmitted(coherency, analyzer) - NUMGA_TRANSMITTED[k]).abs() < 1e-9);
            assert!((shift - NUMGA_SHIFTS[k]).abs() < 1e-9);
        }
    }

    /// numga 35797de's polarized camera: the pixel, weight, power through the analyzer at
    /// 0.3 rad and frequency ratio of each disk crossing, in order.
    #[rustfmt::skip]
    const NUMGA_PIXELS: [usize; 41] = [4, 5, 11, 12, 13, 14, 15, 16, 19, 20, 21, 24, 24, 25, 26, 27, 28, 29, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 42, 42, 43, 44, 46, 47, 48, 49, 49, 50, 50, 51, 52];
    #[rustfmt::skip]
    const NUMGA_WEIGHTS: [f64; 41] = [0.08440542445894317, 0.030221233766829947, 0.3437632444158588, 0.8665643779222696, 0.9685369614062003, 0.9795219012752723, 0.8593109117627874, 0.02084598559451889, 0.5357532974982042, 0.9361922856305335, 0.5600915558518986, 0.9998691733370937, 0.0001307837641689851, 0.9358436229737478, 0.07313479508834542, 0.3870369974891529, 0.8794216142784372, 0.9907923238671051, 0.9999996878093333, 0.9950156190538891, 0.7008717243556963, 0.1882327798294655, 0.5901744588345293, 0.8647714479584754, 0.9684469001987142, 0.9909824557704975, 0.9906264169293011, 0.9630929319409943, 0.03690491856257314, 2.0513921886166853e-06, 0.8011815507631865, 0.329468594603076, 0.010958185511236174, 0.09533292649330094, 0.19464954877705698, 0.24708887534811041, 0.7122056589159441, 0.22232643335643665, 0.7322777996284965, 0.12429579475188612, 0.01622383981524711];
    #[rustfmt::skip]
    const NUMGA_TRANSMITTED: [f64; 41] = [0.18659116269174608, 0.2647263754057526, 0.20626934388769974, 0.16471477916002636, 0.26404737058463496, 0.603898988482792, 0.5411374567370449, 0.4546295806743681, 0.36684220318567257, 0.4025510607027303, 0.5194760035253239, 0.5978388209026066, 0.626060390824152, 0.5813008630156625, 0.5400959717785705, 0.5361827464446843, 0.5581273952524504, 0.5594896687983502, 0.2734426474813757, 0.46253251120564576, 0.5306987729224171, 0.537730925198926, 0.478681743806162, 0.38196762290430875, 0.2659226613938501, 0.1759113373015529, 0.15486800120429153, 0.207100698715073, 0.6666726005021637, 0.7286990133777, 0.2976274510054264, 0.38856852903025974, 0.39421931902808677, 0.30857661243357576, 0.22850951712949513, 0.17218239193472482, 0.1375597352058839, 0.1526312089985403, 0.15961710145351887, 0.17227169723671512, 0.22285649107774783];
    #[rustfmt::skip]
    const NUMGA_SHIFTS: [f64; 41] = [0.8885557020424335, 0.8397666778442441, 1.0449688919408104, 0.9211145359263325, 0.6504725989325366, 0.5592486949566617, 0.6577435362984279, 0.7544736210198665, 1.1781548444390149, 1.1736319376378355, 0.952258556438728, 0.4088213624680935, 0.44989127384980515, 0.6160338878815941, 0.7115309161756826, 1.2805277407334028, 1.316571470019731, 1.3376161066829024, 0.4128115548655099, 0.5648462026349756, 0.6477075294798545, 1.238620156063645, 1.1904108314101107, 1.0864865368689647, 0.9283985582895731, 0.7736011217872489, 0.6841490353006004, 0.6578201600215248, 0.3761605687278652, 0.665704564170704, 0.664319140252992, 0.6835017623428986, 1.1064708405212846, 1.0344142652731358, 0.9549076125009454, 0.8803034477371814, 0.7245401417472633, 0.8205396248285465, 0.6568277167830787, 0.7794995335494068, 0.7557572689552151];
}
