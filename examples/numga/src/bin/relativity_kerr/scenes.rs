//! numga's `kerr/scenarios.py`: equatorial rays as the spin sweeps, and a camera looking at a
//! thin disk around a spinning hole.

use crate::disk::{
    Coherency, Crossing, Disk, Resolution, disk_coherency, disk_temperature, frequency_ratio,
    innermost_stable_orbit, resolve,
};
use crate::kerr::{
    Hole, PolarizedRay, Ray, Rotor, V, camera_rays, camera_screen, camera_trace, initial_momentum,
    polarized_rates, trace, x, xy, y, z,
};
use crate::par_map;
use gax_numga_examples::signal::{phasor, tangent};

pub const MASS: f64 = 1.0;
pub const MAX_SPIN: f64 = 0.8 * MASS;
/// The sweep's rays start this far left of the hole, and the view reaches as far.
pub const EXTENT: f64 = 12.0 * MASS;
pub const RAYS: usize = 31;
/// The rays start up to this far to either side of the hole.
const SPREAD: f64 = 8.0 * MASS;
const STEP_SIZE: f64 = 0.1 * MASS;
const STEPS: usize = 400;
/// Captured rays slow down towards this share of the horizon.
const STOP_FRACTION: f64 = 0.8;

pub const CAMERA_SPIN: f64 = 0.8 * MASS;
const CAMERA_DISTANCE: f64 = 28.0 * MASS;
/// The camera's tilt from the spin axis, degrees.
const INCLINATION: f64 = 75.0;
/// Half the camera's vertical field of view, degrees.
const HALF_VIEW: f64 = 19.0;
/// Each camera step, as a share of the ray's oblate radius.
const STEP_FRACTION: f64 = 0.2;
const CAMERA_STEPS: usize = 720;
const DISK_OUTER_RADIUS: f64 = 12.0 * MASS;
/// The disk's optical depth straight through its thickness, at its inner edge.
const DISK_DEPTH: f64 = 4.0;
/// `(accretion rate c⁶ / (σ G² M²))^(1/4)`, K.
const DISK_TEMPERATURE_SCALE: f64 = 58_500.0;
const ESCAPE_RADIUS: f64 = 100.0 * MASS;
const POLARIZATION_DEGREE: f64 = 0.8;
/// The coordinate time the disk turns through in one loop of the animation.
pub const FLOW_TIME: f64 = 50.0 * MASS;

/// The hole with spin `spin` about `+z`.
pub fn hole(spin: f64) -> Hole {
    Hole::about_z(spin, MASS)
}

/// The hole the camera looks at.
pub fn camera_hole() -> Hole {
    hole(CAMERA_SPIN)
}

/// The spin of frame `frame` of `frames` of the sweep: from `-MAX_SPIN` up and back.
pub fn sweep_spin(frame: usize, frames: usize) -> f64 {
    let angle = core::f64::consts::TAU * frame as f64 / frames as f64;
    -MAX_SPIN * phasor(angle).e20()
}

/// The paths of the equatorial rays at one spin: starting at `x = -EXTENT`, heading along
/// `+x`, spread across.
pub fn sweep(spin: f64) -> Vec<Vec<V>> {
    let hole = hole(spin);
    (0..RAYS)
        .map(|i| {
            let offset = -SPREAD + 2.0 * SPREAD * i as f64 / (RAYS - 1) as f64;
            let event = x() * -EXTENT + y() * offset;
            let start = Ray {
                event,
                momentum: initial_momentum(event, x(), hole),
            };
            trace(start, hole, STEP_SIZE, STEPS, STOP_FRACTION)
                .map(|ray| ray.event)
                .collect()
        })
        .collect()
}

/// The camera's orientation: tilted from the spin axis in the plane `z ∧ x`.
fn orientation() -> Rotor {
    ((z() ^ x()) * (INCLINATION.to_radians() / 2.0)).exp()
}

/// Where the camera is: on its own `z` axis.
pub fn eye() -> V {
    (orientation() >> z()) * CAMERA_DISTANCE
}

/// The tangent of half the camera's vertical field of view.
pub fn half_view() -> f64 {
    tangent(HALF_VIEW.to_radians())
}

/// The disk in the equatorial plane, its gas orbiting with the hole: the thin disk ends at
/// the innermost stable orbit.
pub fn disk(hole: Hole) -> Disk {
    Disk {
        normal: z(),
        orbit: xy(),
        inner_radius: innermost_stable_orbit(xy(), hole),
        outer_radius: DISK_OUTER_RADIUS,
        depth: DISK_DEPTH,
    }
}

/// Captured inside every photon orbit, where no ray turns back out.
fn capture_radius(hole: Hole) -> f64 {
    1.05 * hole.horizon()
}

/// What each camera pixel's ray met, row by row, its screen transported along.
pub fn camera_resolutions(size: [usize; 2]) -> Vec<Resolution<PolarizedRay>> {
    let hole = camera_hole();
    let disk = disk(hole);
    let rays = camera_rays(eye(), orientation(), size, half_view(), hole);
    par_map(&rays, |ray| {
        let start = PolarizedRay {
            ray: *ray,
            screen: camera_screen(*ray, orientation(), hole),
        };
        let steps = camera_trace(start, hole, STEP_FRACTION, CAMERA_STEPS, polarized_rates);
        resolve(steps, &disk, ESCAPE_RADIUS, capture_radius(hole), hole)
    })
}

/// A disk crossing as the camera sees it: where (its event and oblate radius), the share
/// of its pixel's light it gives, its observed temperature in K, and its coherency.
#[derive(Clone, Copy, Debug)]
pub struct Glint {
    pub event: V,
    pub radius: f64,
    pub weight: f64,
    pub temperature: f64,
    pub coherency: Coherency,
}

/// The crossing as the camera sees it. The gas orbits with the hole in its equatorial
/// plane, and a blackbody shifted in frequency is a blackbody at the shifted temperature.
pub fn glint(crossing: &Crossing<PolarizedRay>) -> Glint {
    let hole = camera_hole();
    let ray = crossing.state.ray;
    let shift = frequency_ratio(ray.event, ray.momentum, eye(), xy(), hole);
    let temperature = disk_temperature(crossing.radius, xy(), hole, DISK_TEMPERATURE_SCALE);
    Glint {
        event: ray.event,
        radius: crossing.radius,
        weight: crossing.weight,
        temperature: shift * temperature,
        coherency: disk_coherency(crossing.state, z(), hole, POLARIZATION_DEGREE),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::disk::tests::shares;
    use crate::kerr::rates;

    /// numga's checks of its camera, at 48 x 32 pixels rather than 480 x 320 (a hundredth of
    /// the rays keeps the test quick): each pixel's light is shared at most once between the
    /// disk and the sky, all of it when its ray escapes; where the rays cross the disk they
    /// stay null, to the integrator's accuracy.
    #[test]
    fn scenes_pass_their_checks() {
        let image = camera_resolutions([48, 32]);
        for ((disk, sky), r) in shares(&image).iter().zip(&image) {
            assert!(disk + sky <= 1.0 + 1e-13);
            if r.sky.is_some() {
                assert!((disk + sky - 1.0).abs() < 1e-13);
            }
        }
        let hole = camera_hole();
        for crossing in image.iter().flat_map(|r| &r.disk) {
            let ray = crossing.state.ray;
            // numga's tolerance: the camera's steps are a fifth of the oblate radius.
            assert!((ray.momentum | rates(ray, hole).event).s().abs() < 7e-2);
        }
    }
}
