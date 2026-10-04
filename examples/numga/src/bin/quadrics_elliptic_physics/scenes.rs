//! Scenes of quadric physics on S² and on S³, one engine expanded for each sphere.
//!
//! On S² the bodies are ellipses seen on the front hemisphere: a crowd of needles, discs and
//! ovals; a single oval tumbling about its intermediate axis; and a giant oval with an agile
//! swarm. On S³ each scene is chosen for what the sphere does to the view. The crowd: ellipsoids
//! all over the 3-sphere, seen from a fixed eye. The gap: one huge ellipsoid leaves, between
//! itself and its antipodal image, a belt around the great sphere orthogonal to its centre; small
//! ellipsoids bounce in it, and the eye sits in the belt looking along it. The needle: an
//! ellipsoid 80° long almost meets itself through the antipode of its centre, and the eye looks
//! at the gap between its tips. The tunnel: the Clifford torus of the points at a fixed angle
//! from the great circle `x = y = 0` is a wall that splits the sphere into two linked solid
//! tori; ellipsoids bounce inside one, and the eye on its core circle looks down the tube.

use gax_numga_examples::rng::{Draw, Rng, rng};
use rand::seq::SliceRandom;

/// The engine on S², in `Cl(3)`: points are bivectors, planes vectors, momenta vectors. (One
/// engine for both spheres: not every item is used on each.)
#[allow(dead_code)]
pub mod s2 {
    pub use gax::vga3d::{
        Bivector as Point, Bivector as Rate, Rotor, Scalar, Vector as Momentum, Vector as Plane,
    };
    /// The dimension of the ambient space.
    pub const DIM: usize = 3;
    crate::engine::engine!();
}

/// The engine on S³, in `Cl(4)`: points are trivectors, planes vectors, momenta bivectors.
#[allow(dead_code)]
pub mod s3 {
    pub use crate::s3::cl4::Scalar;
    pub use crate::s3::{
        Bivector as Momentum, Bivector as Rate, Rotor, Trivector as Point, Vector as Plane,
    };
    /// The dimension of the ambient space.
    pub const DIM: usize = 4;
    crate::engine::engine!();
}

/// A colour from its hex code, as display values.
pub fn hex(c: u32) -> [f64; 3] {
    [16, 8, 0].map(|s| f64::from((c >> s) & 0xff) / 255.0)
}

/// matplotlib's `hsv` colour map, as display values.
pub fn hue(t: f64) -> [f64; 3] {
    let h = t.rem_euclid(1.0) * 6.0;
    let f = h - h.floor();
    match h as usize {
        0 => [1.0, f, 0.0],
        1 => [1.0 - f, 1.0, 0.0],
        2 => [0.0, 1.0, f],
        3 => [0.0, 1.0 - f, 1.0],
        4 => [f, 0.0, 1.0],
        _ => [1.0, 0.0, 1.0 - f],
    }
}

/// The spread of a series relative to its first value, `ptp(x) / |x[0]|`.
pub fn drift(x: &[f64]) -> f64 {
    let (lo, hi) = x
        .iter()
        .fold((f64::MAX, f64::MIN), |(lo, hi), v| (lo.min(*v), hi.max(*v)));
    (hi - lo) / x[0].abs()
}

// --- S² ------------------------------------------------------------------------------------

/// The time between frames of the S² scenes, each in six steps.
pub const DT2: f64 = 0.015;

/// The camera rotor, folded into the initial orientation of every body.
pub fn camera() -> s2::M {
    s2::Rate::new(0.2714904022294391, 0.6097774271693677, 1.0148400517968847).exp()
}

/// A polar grid of mass points over an ellipse with half-angles `(a, b)` (radians), in its
/// gnomonic chart, carrying `mass` in all.
pub fn ellipse_mesh(
    half_angles: [f64; 2],
    mass: f64,
    n_phi: usize,
    n_r: usize,
) -> (Vec<s2::P>, Vec<f64>) {
    let (tx, ty) = (half_angles[0].tan(), half_angles[1].tan());
    let (mut points, mut masses) = (Vec::new(), Vec::new());
    for i in 0..=n_phi {
        let phi = core::f64::consts::TAU * i as f64 / n_phi as f64;
        // Trapezoid weights: half at both ends of the angle.
        let w_phi = if i == 0 || i == n_phi { 0.5 } else { 1.0 };
        for j in 1..=n_r {
            let r = j as f64 / n_r as f64;
            let w_r = if j == n_r { 0.5 } else { 1.0 };
            let (x, y) = (tx * r * phi.cos(), ty * r * phi.sin());
            // The sphere's area element in the gnomonic chart, so the point masses approximate
            // a uniform mass over the ellipse. The omitted node at r = 0 has zero area element.
            let area = tx * ty * r / (1.0 + x * x + y * y).powf(1.5);
            masses.push(area * w_phi * w_r);
            points.push(s2::Point::new(x, y, 1.0).normalized().into_inner());
        }
    }
    let total: f64 = masses.iter().sum();
    (points, masses.iter().map(|m| m * mass / total).collect())
}

/// The rotor carrying the pole to polar angle `theta` at azimuth `phi`: about the plane `zx`
/// turned about z by the azimuth.
pub fn toward(theta: f64, phi: f64) -> s2::M {
    let axis = s2::Rate::new(0.0, 0.0, -phi / 2.0).exp() >> s2::Rate::new(0.0, 1.0, 0.0);
    axis.gp(theta / 2.0).exp()
}

/// One ellipse: half-angles in degrees, mass, placement under the camera, body-frame rate, and
/// colour.
pub struct Ellipse {
    pub half_angles: [f64; 2],
    pub mass: f64,
    pub placement: s2::M,
    pub rate: s2::B,
    pub color: u32,
}

/// Bodies from ellipse specifications, with `n_phi` angular nodes in their meshes.
pub fn ellipses(specs: &[Ellipse], n_phi: usize) -> Vec<s2::Body> {
    specs
        .iter()
        .map(|e| {
            let half = e.half_angles.map(f64::to_radians);
            let (points, masses) = ellipse_mesh(half, e.mass, n_phi, 10);
            s2::body(
                hex(e.color),
                s2::ellipsoid(half.map(f64::tan)),
                camera() * e.placement,
                e.rate,
                &points,
                &masses,
            )
        })
        .collect()
}

/// The relative spread of the energy and of the total momentum's norm over a trajectory.
pub fn conserved_s2(t: &s2::Trajectory) -> (f64, f64) {
    let norms: Vec<f64> = t.momentum.iter().map(|m| m.norm()).collect();
    (drift(&t.energy), drift(&norms))
}

/// Seven extreme shapes, needles to discs, scattered over S² and colliding.
pub fn crowded(frames: usize) -> s2::Trajectory {
    let half = [
        [30.0, 6.0],
        [17.0, 17.0],
        [28.0, 6.5],
        [24.0, 5.5],
        [15.0, 4.0],
        [26.0, 9.0],
        [9.0, 9.0],
    ];
    let mass = [1.0, 1.2, 0.9, 0.8, 0.5, 1.1, 0.4];
    let theta = [0.35, 1.05, 1.15, 1.10, 1.25, 0.90, 1.55];
    let phi = [0.2, 0.7, 2.1, 3.6, 4.9, -0.67, 2.90];
    let rate = [
        [0.8, 2.6, 0.5],
        [1.8, -1.2, 0.6],
        [-1.7, 1.5, -0.6],
        [2.0, 1.0, -0.5],
        [-2.2, -1.8, 0.7],
        [1.6, 1.4, 0.7],
        [-2.4, 0.8, -0.4],
    ];
    let colors = [
        0x38bdf8, 0xf43f5e, 0xfbbf24, 0x34d399, 0xa855f7, 0xfb923c, 0xec4899,
    ];
    let specs: Vec<Ellipse> = (0..7)
        .map(|k| Ellipse {
            half_angles: half[k],
            mass: mass[k],
            placement: toward(theta[k], phi[k]),
            rate: s2::Rate::from_coeffs(rate[k]),
            color: colors[k],
        })
        .collect();
    s2::simulate(ellipses(&specs, 384), frames, DT2, 6)
}

/// One oval 40° by 8° spinning near its intermediate axis: it flips over and back,
/// periodically.
pub fn tumbling(frames: usize) -> s2::Trajectory {
    // numga's `xz` turn is a `zx` turn the other way.
    let tilt = s2::Rate::new(0.0, -15f64.to_radians() / 2.0, 0.0).exp()
        * s2::Rate::new(10f64.to_radians() / 2.0, 0.0, 0.0).exp();
    let spec = Ellipse {
        half_angles: [40.0, 8.0],
        mass: 1.0,
        placement: tilt,
        rate: s2::Rate::new(14.0, -0.1, 0.05),
        color: 0x38bdf8,
    };
    s2::simulate(ellipses(&[spec], 384), frames, DT2, 6)
}

/// A giant oval, 75° by 48°, dominating the sphere, and a swarm of small bodies in the channel
/// around it.
pub fn hyperbolic(frames: usize) -> s2::Trajectory {
    let giant = Ellipse {
        half_angles: [75.0, 48.0],
        mass: 6.0,
        placement: toward(0.0, 0.0),
        rate: s2::Rate::new(0.3, 0.2, 0.4),
        color: 0x38bdf8,
    };
    let half = [[18.0, 4.5], [10.0, 10.0], [15.0, 4.0], [16.0, 5.0]];
    let mass = [0.6, 0.7, 0.5, 0.5];
    let phi = [0.35, 1.10, 1.85, 2.65];
    let rate = [
        [2.5, 1.5, -0.8],
        [-2.2, 1.2, 0.6],
        [1.8, -2.0, 0.7],
        [-1.6, -1.8, -0.5],
    ];
    let colors = [0xf43f5e, 0xfbbf24, 0x34d399, 0xa855f7];
    let swarm: Vec<Ellipse> = (0..4)
        .map(|k| Ellipse {
            half_angles: half[k],
            mass: mass[k],
            placement: toward(1.57, phi[k]),
            rate: s2::Rate::from_coeffs(rate[k]),
            color: colors[k],
        })
        .collect();
    let mut bodies = ellipses(&[giant], 1536);
    bodies.extend(ellipses(&swarm, 384));
    s2::simulate(bodies, frames, DT2, 6)
}

// --- S³ ------------------------------------------------------------------------------------

use crate::s3::{Bivector, Rate, Tangent, TangentPlane, Trivector, along, hit, motion, turning};

/// The rotor carrying the origin to a point, spun by a rotation bivector: the square root of
/// the ratio of the two unit points.
pub fn placed(place: s3::P, spin: s3::B) -> s3::M {
    let carry = (crate::s3::unit(place) / s3::origin()).sqrt();
    carry * spin.gp(0.5).exp()
}

/// One large body at the origin, unturned.
pub fn resting(
    color: [f64; 3],
    q: s3::DualQuadric,
    rate: s3::B,
    mass: f64,
    rng: &mut Rng,
) -> s3::Body {
    let (points, masses) = s3::filled(q, mass, 400, rng);
    s3::body(color, q, s3::identity(), rate, &points, &masses)
}

/// Random ellipsoids: half-widths log-uniform per axis between the sizes, so needles, discs and
/// blobs, placed uniformly on the 3-sphere. Tennis-racket rates: a spin of 5 about the
/// intermediate axis (the middle half-width, whose bivector is the plane it is normal to), plus a
/// nudge and a drift across the sphere.
pub fn population(rng: &mut Rng, candidates: usize, sizes: (f64, f64)) -> Vec<s3::Body> {
    let (lo, hi) = (sizes.0.log10(), sizes.1.log10());
    let mut hues: Vec<f64> = (0..candidates)
        .map(|k| k as f64 / candidates as f64)
        .collect();
    hues.shuffle(rng);
    (0..candidates)
        .map(|k| {
            let half: [f64; 3] = core::array::from_fn(|_| 10f64.powf(rng.range(lo, hi)));
            // On `yz`, `zx`, `xy` (turns) and `xw`, `yw`, `zw` (drifts).
            let scale = [0.3, 0.3, 0.3, 0.6, 0.6, 0.6];
            let mut rate: [f64; 6] = core::array::from_fn(|i| rng.normal() * scale[i]);
            let mut order = [0, 1, 2];
            order.sort_by(|a, b| half[*a].total_cmp(&half[*b]));
            rate[order[1]] = if rng.uniform() < 0.5 { -5.0 } else { 5.0 };
            let place = Trivector::from_coeffs(rng.direction::<4>());
            let spin = turning(TangentPlane::new(rng.normal(), rng.normal(), rng.normal()));
            let q = s3::ellipsoid(half);
            let (points, masses) = s3::filled(q, half.iter().product::<f64>() * 200.0, 400, rng);
            s3::body(
                hue(hues[k]),
                q,
                placed(place, spin),
                Bivector::from_coeffs(rate),
                &points,
                &masses,
            )
        })
        .collect()
}

/// The first `given` bodies plus the first `count` of the rest that overlap none kept before
/// them: one overlap test over the pairs within reach, then a greedy pass.
pub fn admitted(bodies: Vec<s3::Body>, given: usize, count: usize) -> Vec<s3::Body> {
    let n = bodies.len();
    let motors: Vec<s3::M> = bodies.iter().map(|b| b.motor).collect();
    let mut overlapping = vec![false; n * n];
    for (i, j) in s3::candidates_near(&bodies) {
        let hit = s3::margin(&bodies, &motors, i, j).0 < 0.0;
        overlapping[i * n + j] = hit;
        overlapping[j * n + i] = hit;
    }
    let mut keep: Vec<usize> = (0..given).collect();
    for candidate in given..n {
        if keep.len() == given + count {
            break;
        }
        if !keep.iter().any(|k| overlapping[candidate * n + k]) {
            keep.push(candidate);
        }
    }
    assert!(
        keep.len() == given + count,
        "only {} admissible non-overlapping candidates",
        keep.len() - given
    );
    keep.iter().map(|k| bodies[*k]).collect()
}

/// The first frame at which the eye is inside a body, if any.
pub fn eye_inside(scene: &Scene3) -> Option<usize> {
    let eye = scene.eye >> s3::origin();
    scene
        .trajectory
        .surfaces
        .iter()
        .position(|frame| frame.iter().any(|s| (eye & s.of(eye)).s() < 0.0))
}

/// An S³ scene: the trajectory, the eye and the light.
pub struct Scene3 {
    pub trajectory: s3::Trajectory,
    pub eye: s3::M,
    pub light: s3::P,
}

/// 28 ellipsoids all over the 3-sphere, seen from the origin. The light is a point 0.8 rad from
/// the eye, above and behind it, just outside the 120° frustum. Whatever drifts behind the eye
/// reappears ahead near the antipode.
pub fn crowd(frames: usize) -> Scene3 {
    // numga seeds 3; with this generator, seed 9 keeps every body off the eye for the
    // animation's 240 frames (a body over the eye fills the view with its dark inside).
    let mut rng = rng(9);
    let bodies = admitted(population(&mut rng, 120, (0.05, 0.5)), 0, 28);
    let light = motion(along(Tangent::new(-0.34, 0.0, 0.94)).gp(0.8)) >> s3::origin();
    finish(bodies, frames, s3::identity(), light)
}

fn finish(bodies: Vec<s3::Body>, frames: usize, eye: s3::M, light: s3::P) -> Scene3 {
    Scene3 {
        trajectory: s3::simulate(bodies, frames, 0.02, 8),
        eye,
        light,
    }
}

/// A huge ellipsoid reaching 60°, 75° and 70° along x, y and z, so the belt around the great
/// sphere `w = 0` between it and its antipodal image is 30°, 15° and 20° thick on either side;
/// sixty small ellipsoids in the belt. The eye a quarter turn along x, in the thick part, looking
/// along y where it thins; the light behind the eye.
pub fn gap(frames: usize) -> Scene3 {
    // numga's seed 5 also keeps every body off the eye for the animation's 240 frames
    // with this generator (a body over the eye fills the view with its dark inside).
    let mut rng = rng(5);
    let deg = f64::to_radians;
    let shape = s3::ellipsoid([deg(60.0).tan(), deg(75.0).tan(), deg(70.0).tan()]);
    let huge = resting([0.75, 0.7, 0.6], shape, Rate::zero(), 500.0, &mut rng);
    let mut all = vec![huge];
    all.extend(population(&mut rng, 2000, (0.03, 0.15)));
    let bodies = admitted(all, 1, 60);
    let quarter = core::f64::consts::FRAC_PI_2;
    let eye = motion(along(Tangent::new(quarter, 0.0, 0.0)))
        * motion(turning(TangentPlane::new(0.0, 0.0, -quarter)));
    let light = (eye * motion(along(Tangent::new(-0.3, 0.25, 0.2)))) >> s3::origin();
    finish(bodies, frames, eye, light)
}

/// A heavy needle 80° long along x, spinning about its own axis, almost meets itself through
/// the antipode of its centre; the eye at the ideal point of x, lifted 35° along z and looking
/// back down at the gap between the tips.
pub fn needle(frames: usize) -> Scene3 {
    // numga seeds 7; with this generator, seed 5 keeps every body off the eye for the
    // animation's 240 frames (a body over the eye fills the view with its dark inside).
    let mut rng = rng(5);
    let deg = f64::to_radians;
    let shape = s3::ellipsoid([deg(80.0).tan(), deg(4.0).tan(), deg(3.0).tan()]);
    let spin = turning(TangentPlane::new(2.0, 0.0, 0.0));
    let long = resting([0.9, 0.85, 0.3], shape, spin, 50.0, &mut rng);
    let mut all = vec![long];
    all.extend(population(&mut rng, 1000, (0.05, 0.3)));
    let bodies = admitted(all, 1, 40);
    let quarter = core::f64::consts::FRAC_PI_2;
    let eye = motion(along(Tangent::new(quarter, 0.0, 0.0)))
        * motion(along(Tangent::new(0.0, 0.0, deg(35.0))))
        * motion(turning(TangentPlane::new(0.0, -quarter, 0.0)));
    let light = (eye * motion(along(Tangent::new(-0.3, 0.2, 0.3)))) >> s3::origin();
    finish(bodies, frames, eye, light)
}

/// The tunnel scene, with the point where the wall is lit and the camera's position.
pub struct Tunnel {
    pub scene: Scene3,
    // Read by the tests, numga's checks of the light's placement.
    #[cfg_attr(not(test), allow(dead_code))]
    pub wall_hit: s3::P,
    #[cfg_attr(not(test), allow(dead_code))]
    pub camera: s3::P,
}

/// A torus around the great circle `x = y = 0`: the dual quadric with -1 across the tube in x
/// and `-1/1.5²` in y (an elliptical cross-section), and `1/tan²(radius)` along it, the tube's
/// angular radius 20° at w and 40° a quarter turn along the core (z). Its inside, in the sense of
/// its form, is the complementary solid torus; the crowd lives in the tube. The eye in the wide
/// section at `+z` looks toward the narrow waist at `-w`; the light halfway to the wall.
pub fn tunnel(frames: usize) -> Tunnel {
    // numga seeds 11; with this generator, seed 3 keeps every body off the eye for the
    // animation's 240 frames (a body over the eye fills the view with its dark inside).
    let mut rng = rng(3);
    let deg = f64::to_radians;
    let tube = s3::quadric([
        -1.0,
        -1.0 / (1.5 * 1.5),
        1.0 / deg(40.0).tan().powi(2),
        1.0 / deg(20.0).tan().powi(2),
    ]);
    let torus = resting([0.55, 0.65, 0.75], tube, Rate::zero(), 500.0, &mut rng);
    let mut all = vec![torus];
    all.extend(population(&mut rng, 2000, (0.03, 0.1)));
    let bodies = admitted(all, 1, 50);
    let quarter = core::f64::consts::FRAC_PI_2;
    let eye = motion(along(Tangent::new(0.0, 0.0, quarter)))
        * motion(turning(TangentPlane::new(0.0, quarter, 0.0)));
    let camera = eye >> s3::origin();
    // The pixel up and to the side, a quarter turn off the line of sight, and the wall there.
    let side_up = crate::s3::ScreenPoint::new(0.0, 1.0, 1.0).gp(0.5f64.sqrt());
    let (conic, polar) = crate::s3::project(eye, bodies[0].world());
    let wall_hit = hit(eye, crate::s3::reproject(conic, polar, side_up), side_up);
    let light = (camera + wall_hit).normalized().into_inner();
    Tunnel {
        scene: finish(bodies, frames, eye, light),
        wall_hit,
        camera,
    }
}
