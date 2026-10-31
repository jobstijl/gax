//! numga's `quadrics/csg_raytracer`: solids bounded by quadrics in PGA3D, and the motors that
//! place them and the camera, ray traced (numga's WGSL tracer, here on the CPU).
//!
//! A solid quadric is a map from points to planes, `Plane <- Point`, whose pairing with a point,
//! `quadric(point) & point`, is zero on its surface and negative inside; at a point on the
//! surface the plane it gives is the tangent plane. A solid is the intersection of a few such
//! quadrics: an ellipsoid; a ruled hyperboloid between two half-spaces; a bowl, an ellipsoid
//! less a smaller one, cut by a plane. A half-space is a quadric too, the product of its plane's
//! pairing with the plane at infinity's, so the planes that cut and the floor are traced as
//! every other quadric is. Negating a quadric swaps inside and outside, and subtracts its solid.
//! A motor places a quadric by moving each point back, measuring it, and moving its plane
//! forward: `motor >> quadric(motor << Point)`.
//!
//! Along a ray from `start` along `direction` the pairing is a quadratic in the distance, whose
//! roots are where the ray crosses the quadric's surface; a crossing is on the solid's surface
//! where the point is inside all its other quadrics. The first such point is lit by the lamp
//! through a shadow ray, laid with its solid's checker and glint, and mixed with what its
//! reflection sees, by the solid's reflectivity: the reflection is the sandwich of the ray's
//! direction by the tangent plane.
//!
//! The animation orbits the camera about the scene: a ruled hyperboloid, a bowl cut open and a
//! mirror ball on a checkered floor.

#[path = "../shared/quadrics_tracing.rs"]
mod tracing;

use gax_colour::Light;
use gax_numga_examples::{Anim, Canvas, Rect, backdrop, caption, run_with};

mod csg {
    use gax::Unit;
    use gax::pga3d::{Motor, Plane, Point};

    /// A solid quadric: from points to their polar planes.
    pub type Quadric = Plane<(Point,)>;
    /// A placement.
    pub type M = Unit<Motor>;

    /// The planes through the origin square to each axis.
    pub fn axes() -> [Plane; 3] {
        [
            Plane::new(1.0, 0.0, 0.0, 0.0),
            Plane::new(0.0, 1.0, 0.0, 0.0),
            Plane::new(0.0, 0.0, 1.0, 0.0),
        ]
    }

    /// The plane at infinity, whose pairing with a point is its weight.
    pub fn w() -> Plane {
        Plane::new(0.0, 0.0, 0.0, 1.0)
    }

    /// A quadric with the axes' squared pairings over the signed squared semi-axes, less the
    /// plane at infinity's.
    fn diagonal(semi_axes: [f64; 3], signs: [f64; 3]) -> Quadric {
        let terms: Quadric = (axes().iter().zip(semi_axes).zip(signs))
            .map(|((a, s), sign)| *a * (*a & Point::slot()) * (sign / (s * s)))
            .sum();
        terms - w() * (w() & Point::slot())
    }

    /// A solid ellipsoid about the origin.
    pub fn ellipsoid(semi_axes: [f64; 3]) -> Quadric {
        diagonal(semi_axes, [1.0, 1.0, 1.0])
    }

    /// A solid ruled hyperboloid of one sheet about the origin, its waist around z: as an
    /// ellipsoid, with the pairing along z subtracted.
    pub fn hyperboloid(semi_axes: [f64; 3]) -> Quadric {
        diagonal(semi_axes, [1.0, 1.0, -1.0])
    }

    /// The points on a plane's negative side, as the quadric that pairs them with the plane and
    /// with the plane at infinity, symmetrized.
    pub fn half_space(plane: Plane) -> Quadric {
        (plane * (w() & Point::slot()) + w() * (plane & Point::slot())) * 0.5
    }

    /// A quadric placed by a motor: every point moved back, measured, and its plane moved
    /// forward.
    pub fn placed(motor: M, quadric: Quadric) -> Quadric {
        motor >> quadric.of(motor << Point::slot())
    }

    /// A camera's motor, from one at the origin looking along its own `−z` with `y` up: backed
    /// off by `distance`, tilted to look from `elevation` above the horizon, turned to
    /// `azimuth` about the vertical, and carried to the target.
    pub fn orbit(target: Point, distance: f64, elevation: f64, azimuth: f64) -> M {
        let back_off = Motor::translation(Point::direction(0.0, 0.0, distance));
        let pitch = Motor::rotation_about(
            Point::direction(1.0, 0.0, 0.0),
            core::f64::consts::FRAC_PI_2 - elevation,
        );
        let turn = Motor::rotation_about(Point::direction(0.0, 0.0, 1.0), azimuth);
        let carry = Motor::translation(target - Point::origin());
        carry * turn * pitch * back_off
    }

    /// Points around the waist of a circular hyperboloid, and the directions of the two
    /// straight lines through each that lie in its surface: the point and its directions turned
    /// about z together.
    #[cfg(test)]
    pub fn ruling_lines(semi_axes: [f64; 3], n: usize) -> Vec<[Point; 3]> {
        let [radius, _, height] = semi_axes;
        (0..n)
            .map(|k| {
                let angle = core::f64::consts::TAU * k as f64 / n as f64;
                let turn = Motor::rotation_about(Point::direction(0.0, 0.0, 1.0), angle);
                [
                    turn >> Point::xyz(radius, 0.0, 0.0),
                    turn >> Point::direction(0.0, radius, height),
                    turn >> Point::direction(0.0, radius, -height),
                ]
            })
            .collect()
    }

    // --- the trace

    /// Distances beyond the scene, the step a ray takes off a surface before it looks again,
    /// and how far a point may lie outside a quadric and still count as inside, for points on
    /// two surfaces.
    pub const FAR: f64 = 1e6;
    pub const CLEARANCE: f64 = 1e-3;
    pub const TOLERANCE: f64 = 1e-4;

    /// A quadric along a ray, from `start` along `direction`: its pairing with the ray's point,
    /// a quadratic in the distance `t` along it, `a t² + 2 b t + c`.
    pub fn along_ray(quadric: Quadric, start: Point, direction: Point) -> [f64; 3] {
        let (ahead, here) = (quadric.of(direction), quadric.of(start));
        [
            (ahead & direction).s(),
            ((ahead & start).s() + (here & direction).s()) / 2.0,
            (here & start).s(),
        ]
    }

    /// Where a ray crosses a quadric's surface, the roots of the quadratic along it, where it
    /// has them: both without cancellation; a half-space, with no square term, has only the
    /// second, the first at `FAR`.
    pub fn crossings([a, b, c]: [f64; 3]) -> Option<[f64; 2]> {
        let discriminant = b * b - a * c;
        if discriminant < 0.0 {
            return None;
        }
        #[allow(clippy::disallowed_methods)] // the quadratic's roots
        let q = -(b + b.signum() * discriminant.sqrt());
        Some([if a == 0.0 { FAR } else { q / a }, c / q])
    }

    /// A solid: the intersection of its quadrics, each negative inside it; its colour (sRGB),
    /// the share of light it reflects as a mirror, and two planes whose unit steps lay a
    /// checker over it.
    pub struct Solid {
        pub quadrics: Vec<Quadric>,
        pub colour: [f32; 3],
        pub reflectivity: f64,
        pub across: Plane,
        pub along: Plane,
    }

    /// The first point between two distances where a ray meets a solid's surface: on one of its
    /// quadrics, inside all the others. Its distance and tangent plane.
    pub fn region_hit(
        quadrics: &[Quadric],
        start: Point,
        direction: Point,
        nearest: f64,
        farthest: f64,
    ) -> Option<(f64, Plane)> {
        let mut hit = None;
        let mut bound = farthest;
        for (own, quadric) in quadrics.iter().enumerate() {
            let Some(roots) = crossings(along_ray(*quadric, start, direction)) else {
                continue;
            };
            for distance in roots {
                let point = start + direction * distance;
                let inside = quadrics
                    .iter()
                    .enumerate()
                    .all(|(k, q)| k == own || (q.of(point) & point).s() <= TOLERANCE);
                if distance > nearest && distance < bound && inside {
                    bound = distance;
                    hit = Some((distance, quadric.of(point)));
                }
            }
        }
        hit
    }

    /// The surface a ray meets first: how far along it, its tangent plane there, and its solid.
    pub fn first_hit(
        solids: &[Solid],
        start: Point,
        direction: Point,
    ) -> Option<(f64, Plane, usize)> {
        let mut hit = None;
        let mut bound = FAR;
        for (index, solid) in solids.iter().enumerate() {
            if let Some((distance, tangent)) =
                region_hit(&solid.quadrics, start, direction, CLEARANCE, bound)
            {
                bound = distance;
                hit = Some((distance, tangent, index));
            }
        }
        hit
    }

    // --- the scene

    /// The hyperboloid's waist radius and its half-height, where two planes cut it; its place
    /// and tilt.
    pub const WAIST_AXES: [f64; 3] = [0.55, 0.55, 0.85];
    pub const HYPERBOLOID_PLACE: [f64; 3] = [-1.2, 0.4, 0.85];
    pub const HYPERBOLOID_TILT: f64 = 12.0;
    /// The bowl: an ellipsoid less a smaller one, cut by a plane tilted off its top.
    pub const BOWL_AXES: [f64; 3] = [0.75, 0.75, 0.75];
    pub const WALL_FRACTION: f64 = 0.8;
    pub const BOWL_PLACE: [f64; 3] = [1.1, -0.2, 0.75];
    pub const BOWL_TILTS: [f64; 2] = [35.0, 15.0];
    pub const MIRROR_RADIUS: f64 = 0.55;
    pub const MIRROR_PLACE: [f64; 3] = [0.0, 0.1, 0.55];
    /// Each solid's colour and reflectivity: hyperboloid, bowl, mirror, floor.
    pub const COLOURS: [[f32; 3]; 4] = [
        [0.84, 0.48, 0.24],
        [0.15, 0.45, 0.55],
        [0.95, 0.95, 0.96],
        [0.88, 0.87, 0.85],
    ];
    pub const REFLECTIVITIES: [f64; 4] = [0.0, 0.0, 0.88, 0.0];
    /// Unit steps of each solid's checker: stripes crossing the hyperboloid's rulings, tiles on
    /// the floor.
    pub const STRIPES: f64 = 8.0;
    pub const TILES: f64 = 1.4;
    /// The camera orbiting a point above the floor, and the tangent of half its view.
    pub const TARGET: [f64; 3] = [0.0, 0.2, 0.7];
    pub const DISTANCE: f64 = 4.6;
    pub const ELEVATION: f64 = 28.0;
    pub const HALF_VIEW: f64 = 0.42;
    pub const N_FRAMES: usize = 96;

    /// The lamp's direction (numga's, not unit).
    pub fn lamp() -> Point {
        Point::direction(0.5, -0.5, 0.7)
    }

    pub fn target() -> Point {
        let [x, y, z] = TARGET;
        Point::xyz(x, y, z)
    }

    /// The translation to a place, from the origin.
    fn to([x, y, z]: [f64; 3]) -> M {
        Motor::translation(Point::direction(x, y, z))
    }

    /// The turn about a plane's axis by an angle in degrees, numga's `exp(plane · (−angle / 2))`
    /// for the bivector of two of the axes' planes.
    fn turn(plane: gax::pga3d::Line, degrees: f64) -> M {
        (plane * (-degrees.to_radians() / 2.0)).exp()
    }

    /// Every solid: hyperboloid, bowl, mirror and floor, each placed by one motor.
    pub fn solids() -> Vec<Solid> {
        let [x, y, z] = axes();
        let lift = to(HYPERBOLOID_PLACE) * turn(x ^ z, HYPERBOLOID_TILT);
        let bowl = to(BOWL_PLACE) * turn(y ^ z, BOWL_TILTS[0]) * turn(x ^ z, BOWL_TILTS[1]);
        let mirror = to(MIRROR_PLACE);
        let height = WAIST_AXES[2];
        // The hyperboloid between the planes a half-height above and below its waist.
        let hyperboloid_parts = [
            hyperboloid(WAIST_AXES),
            half_space(z - w() * height),
            half_space(-z - w() * height),
        ];
        // The bowl: inside the outer ellipsoid, outside the inner one, below the cut.
        let cut = (y * 0.15 + z * 0.98).normalized() - w() * 0.1;
        let inner = BOWL_AXES.map(|a| a * WALL_FRACTION);
        let bowl_parts = [ellipsoid(BOWL_AXES), -ellipsoid(inner), half_space(cut)];
        let none = Plane::new(0.0, 0.0, 0.0, 0.0);
        let solid = |quadrics: Vec<Quadric>, k: usize, across: Plane, along: Plane| Solid {
            quadrics,
            colour: COLOURS[k],
            reflectivity: REFLECTIVITIES[k],
            across,
            along,
        };
        vec![
            solid(
                hyperboloid_parts.iter().map(|q| placed(lift, *q)).collect(),
                0,
                lift >> ((x + z) * STRIPES),
                lift >> ((y - z) * STRIPES),
            ),
            solid(
                bowl_parts.iter().map(|q| placed(bowl, *q)).collect(),
                1,
                none,
                none,
            ),
            solid(
                vec![placed(mirror, ellipsoid([MIRROR_RADIUS; 3]))],
                2,
                none,
                none,
            ),
            // The floor: everything below the plane z = 0.
            solid(vec![half_space(z)], 3, x * TILES, y * TILES),
        ]
    }

    /// The camera at a frame of the orbit, starting in front of the scene.
    pub fn camera(frame: usize) -> M {
        let azimuth = core::f64::consts::TAU * frame as f64 / N_FRAMES as f64;
        orbit(target(), DISTANCE, ELEVATION.to_radians(), azimuth)
    }
}

use csg::*;
use gax::pga3d::{Plane, Point};

/// The loop: numga's 96 frames of 40 ms, once round.
const SECONDS: f32 = 3.84;
/// The look: light where the lamp does not reach, a shadow's light, the checker's darkening,
/// and the glint's sharpness and brightness; the intensity of each solid's light, the floor's
/// low so that it lies dim under the navy look.
const LOOK: tracing::Look = tracing::Look {
    ambient: 0.28,
    checker: 0.15,
};
const SHADOW: f64 = 0.3;
const SHININESS: i32 = 40;
const GLOSS: f64 = 0.3;
const INTENSITY: [f32; 4] = [0.9, 0.9, 0.9, 0.14];

/// What a ray sees: the solid it meets, lit by the lamp, shadowed, checkered and glinting, its
/// point, unit tangent plane and reflectivity; or nothing (the sky).
fn sight(solids: &[Solid], start: Point, direction: Point) -> Option<(Light, Point, Plane, f64)> {
    let (distance, tangent, index) = first_hit(solids, start, direction)?;
    let solid = &solids[index];
    let point = start + direction * distance;
    let normal = tangent.normalized();
    let lamp = lamp();
    // The lamp's light falls by the cosine of its direction off the normal, a plane paired
    // with a direction; a shadow ray towards it dims it where it meets a solid.
    let shadowed = first_hit(solids, point + lamp * CLEARANCE, lamp).is_some();
    let lit = (normal & lamp).s().max(0.0) * if shadowed { SHADOW } else { 1.0 };
    let parity = tracing::parity((solid.across & point).s(), (solid.along & point).s());
    // The view ray reflected in the tangent plane, against the lamp: `−d.dual()` is the plane
    // square to `d`.
    let reflected = normal >> direction;
    let glint = (-(reflected.dual() & lamp).s()).max(0.0).powi(SHININESS);
    let [r, g, b] = solid.colour;
    let base = LOOK.surface(Light::from_srgb(r, g, b, INTENSITY[index]), parity, lit);
    Some((
        base + tracing::glint(GLOSS * glint * lit * f64::from(INTENSITY[index])),
        point,
        normal,
        solid.reflectivity,
    ))
}

/// What the ray through a place on the picture sees from a camera: along the camera's own `−z`,
/// turned by its motor; a mirror adds what its reflection sees.
fn sample(solids: &[Solid], camera: M, place: [f64; 2]) -> Option<Light> {
    let [x, y] = place.map(|v| v * HALF_VIEW);
    let toward = camera >> Point::direction(x, y, -1.0);
    let direction = toward * (1.0 / gax::IdealNorm::ideal_norm(toward));
    let eye = camera >> Point::origin();
    let (seen, point, normal, reflectivity) = sight(solids, eye, direction)?;
    if reflectivity == 0.0 {
        return Some(seen);
    }
    let reflected = normal >> direction;
    let up = (axes()[2] & reflected).s();
    let mirrored =
        sight(solids, point + reflected * CLEARANCE, reflected).map_or(tracing::sky(up), |s| s.0);
    Some(seen.mix_light(mirrored, reflectivity as f32))
}

struct Scenes {
    solids: Vec<Solid>,
}

fn scenes() -> Scenes {
    Scenes { solids: solids() }
}

/// The scene from a camera over a picture.
fn trace(c: &mut Canvas, picture: Rect, solids: &[Solid], camera: M) {
    c.clip(picture);
    c.shade(1, |q| sample(solids, camera, tracing::screen(q, picture)));
    c.unclip();
}

fn draw(c: &mut Canvas, s: &Scenes, t: f32) {
    backdrop(c);
    let frame = gax_numga_examples::signal::frame((t / SECONDS).rem_euclid(1.0), N_FRAMES);
    trace(c, c.rect(), &s.solids, camera(frame));
    caption(
        c,
        "CSG RAY TRACER: SOLIDS BOUNDED BY QUADRICS",
        "A QUADRIC Plane <- Point; ITS PAIRING ALONG A RAY A QUADRATIC; PLACED BY motor >> q(motor << Point) (PGA3D)",
    );
}

fn main() {
    run_with(
        Anim::new("csg_raytracer", SECONDS).size(960, 540),
        scenes,
        draw,
    );
}

#[cfg(test)]
mod tests {
    use super::csg::*;
    use gax::pga3d::{Motor, Plane, Point};
    use gax_numga_examples::{Canvas, Rect};

    const ROUND_OFF: f64 = 1e-12;

    /// Both families of straight lines through the hyperboloid's waist lie in its surface.
    #[test]
    fn hyperboloid_rulings_lie_in_its_surface() {
        let surface = hyperboloid(WAIST_AXES);
        for n in [8, 12] {
            for [waist, rising, falling] in ruling_lines(WAIST_AXES, n) {
                for direction in [rising, falling] {
                    for offset in [-2.0, -1.0, 0.0, 1.0, 2.0] {
                        let p = waist + direction * offset;
                        assert!((surface.of(p) & p).s().abs() < ROUND_OFF);
                    }
                }
            }
        }
    }

    /// A placed quadric pairs the moved points as the quadric did the points; the half-space is
    /// negative below its plane, by the height times the weight.
    #[test]
    fn placed_quadrics_and_half_spaces_follow_their_motor() {
        let [x, _, z] = axes();
        let shift = Motor::translation(Point::direction(1.0, -2.0, 0.5));
        // numga's translator `exp((w ^ d.dual()) / 2)`, the same motor.
        let numga = ((w() ^ Point::direction(1.0, -2.0, 0.5).dual()) * 0.5).exp();
        assert!(
            (shift.into_inner() - numga.into_inner())
                .c
                .iter()
                .all(|c| c.abs() < ROUND_OFF)
        );
        let motor = shift * ((x ^ z) * -0.3).exp();
        let ball = ellipsoid([0.5, 0.7, 0.9]);
        let floor = half_space(z);
        let points = [
            Point::xyz(0.5, 0.0, 0.0),
            Point::xyz(0.0, 0.7, 0.0),
            Point::xyz(0.0, 0.0, -2.0),
            Point::xyz(0.1, 0.2, 0.3),
        ];
        for q in [ball, floor] {
            let moved = placed(motor, q);
            for p in points {
                let m = motor >> p;
                assert!(((moved.of(m) & m).s() - (q.of(p) & p).s()).abs() < ROUND_OFF);
            }
        }
        let heights: Vec<f64> = points.iter().map(|p| (floor.of(*p) & *p).s()).collect();
        for (h, e) in heights.iter().zip([0.0, 0.0, -2.0, 0.3]) {
            assert!((h - e).abs() < ROUND_OFF, "{heights:?}");
        }
    }

    /// Every camera looks at the target, from where numga's do.
    #[test]
    fn orbit_cameras_look_at_the_target() {
        for frame in [0, 16, 24, 50, 72] {
            let camera = camera(frame);
            let eye = camera >> Point::origin();
            let look = camera >> Point::direction(0.0, 0.0, -1.0);
            let error = (eye + look * DISTANCE).unitized() - target();
            assert!(error.c.iter().all(|c| c.abs() < ROUND_OFF), "{error:?}");
        }
        // numga's first two eyes: in front of the scene, then a quarter turn round.
        let eyes = [
            [0.0, -3.861_558_927_151, 2.859_569_188_815],
            [4.061_558_927_151, 0.2, 2.859_569_188_815],
        ];
        for (frame, expected) in [0, N_FRAMES / 4].iter().zip(eyes) {
            let eye = (camera(*frame) >> Point::origin()).to_euclidean();
            assert!(
                eye.iter().zip(expected).all(|(a, b)| (a - b).abs() < 1e-11),
                "{eye:?}"
            );
        }
    }

    /// A mirror reflects a direction as a mirror does: the floor sends a ray going down back up.
    #[test]
    fn a_plane_reflects_a_direction() {
        let floor = Plane::new(0.0, 0.0, 1.0, 0.0);
        assert_eq!(
            floor >> Point::direction(1.0, 0.0, -1.0),
            Point::direction(1.0, 0.0, 1.0)
        );
    }

    /// The bowl's teal shows right of the centre, and the floor's grey along the bottom row.
    #[test]
    fn trace_shows_the_solids() {
        let solids = solids();
        let (w, h) = (144, 90);
        let mut c = Canvas::with_scale(w, h, 1.0);
        super::trace(
            &mut c,
            Rect::new(0.0, 0.0, w as f32, h as f32),
            &solids,
            camera(0),
        );
        let rgb = |x: usize, y: usize| {
            let [r, g, b, _] = c.get(x, y).to_premultiplied();
            [r, g, b]
        };
        let teal = (30..60)
            .flat_map(|y| (90..130).map(move |x| (x, y)))
            .any(|(x, y)| {
                let [r, _, b] = rgb(x, y);
                b > r + 0.1 * b.max(1e-3) && b > 0.02
            });
        assert!(teal);
        for x in 0..w {
            let [r, g, b] = rgb(x, h - 1);
            let top = r.max(g).max(b);
            assert!(
                top > 0.0 && (r.min(g).min(b) / top) > 0.85,
                "{x}: {r} {g} {b}"
            );
        }
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws_with(super::draw, &super::scenes(), 1.0);
    }
}
