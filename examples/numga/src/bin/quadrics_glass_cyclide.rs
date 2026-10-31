//! numga's `quadrics/glass_cyclide`: a glass Dupin cyclide with a bite taken out of it, a ball
//! behind it, on a checkered floor, ray traced through the glass (numga's CGA3D `x+y+z+w+e-`;
//! numga's WGSL tracer, here on the CPU).
//!
//! A point of space is a null antivector, placed by translators carrying the origin, and a
//! quadric maps a point to its polar sphere, `Sphere <- Point`; its pairing with a point,
//! `point & quadric(point)`, is zero on the surface and negative inside, and at a point of the
//! surface its sphere is the tangent sphere. In this model the quadrics take in the surfaces
//! Euclidean geometry calls quartic: a torus is one, and a sphere inversion, a versor, carries
//! it to a Dupin cyclide, whose tube swells on one side and narrows on the other. A ball and a
//! half-space are quadrics too, the product of a sphere's or a plane's pairing with that of the
//! plane at infinity. A solid is the intersection of a few quadrics, and negating one subtracts
//! it.
//!
//! How far a point lies outside a solid is the largest of its quadrics' forms; a ray's points
//! are its start carried by translators along its direction, and the ray meets the solid where
//! that form changes sign, found by stepping across the solid's bounding sphere and halving the
//! step that changed. At glass the path refracts by Snell's law and adds what the reflected ray
//! glimpses by Fresnel's share (Schlick's), and inside glass the light fades by the glass's tint
//! per unit length: a filter, a diagonal map on light points (gax-colour's `colour_map`).
//!
//! The algebra and its conformal basics are `shared/quadrics_diagonal_cga.rs`; the look shared
//! with the CSG tracer `shared/quadrics_tracing.rs`. The animation orbits the camera about the
//! cyclide.

#[path = "../shared/quadrics_diagonal_cga.rs"]
mod cga;
#[path = "../shared/quadrics_tracing.rs"]
mod tracing;

use gax_colour::{Light, ops};
use gax_numga_examples::{Anim, Canvas, Rect, backdrop, caption, run_with};

mod cyclide {
    use super::cga::*;

    /// The solid torus about the z axis: a core circle of `radius` in the xy plane, and a `tube`
    /// about it.
    pub fn torus(radius: f64, tube: f64) -> Quadric {
        let core = sphere(origin(), radius * radius + tube * tube);
        let slot = Quadvector::slot();
        core * (core & slot)
            + (z() * (z() & slot) - horizon() * (horizon() & slot) * (tube * tube))
                * (radius * radius)
    }

    /// The turn in the plane of two vectors, numga's `exp(a ^ b · (−angle / 2))`.
    pub fn turn(a: Vector, b: Vector, angle: f64) -> Motor {
        ((a ^ b) * (-angle / 2.0)).exp()
    }

    /// The direction along an axis, `x.dual() · s`.
    pub fn along(axis: Vector, s: f64) -> Direction {
        direction_of(axis) * s
    }

    #[cfg(test)]
    /// Points on a torus's surface: a point on the tube's circle turned about the core, then
    /// about z, `[across][around]`.
    pub fn torus_points(radius: f64, tube: f64, around: usize, across: usize) -> Vec<Vec<Point>> {
        let tau = core::f64::consts::TAU;
        (0..across)
            .map(|j| {
                let on_tube = turn(x(), z(), tau * j as f64 / across as f64)
                    >> (translator(along(x(), tube)) >> origin());
                (0..around)
                    .map(|i| {
                        let about = turn(x(), y(), tau * i as f64 / around as f64);
                        about >> (translator(along(x(), radius)) >> on_tube)
                    })
                    .collect()
            })
            .collect()
    }

    /// The plane through the origin parallel to a surface's tangent plane at a point, from its
    /// tangent sphere there: that sphere flattened at the point is the tangent plane, and its
    /// part within `x y z` the plane through the origin parallel to it.
    pub fn flattened(tangent: Sphere, point: Point) -> Vector {
        let plane = tangent + dual_inverse(point) * tangent.dot(horizon()).s();
        plane.cast::<OriginPlane>().into()
    }

    /// The outward direction square to a surface at a point: the dual of the plane through the
    /// origin parallel to its tangent plane.
    #[cfg(test)]
    pub fn outward(tangent: Sphere, point: Point) -> Direction {
        direction_of(flattened(tangent, point))
    }

    // --- the trace

    pub const FAR: f64 = 1e6;
    /// The step a ray takes off a surface before it looks again; the steps across a bounding
    /// sphere, the halvings of the one that crosses, and the bounces of a path.
    pub const CLEARANCE: f64 = 1e-3;
    pub const N_STEPS: usize = 40;
    pub const N_HALVINGS: usize = 18;
    pub const N_BOUNCES: usize = 8;

    /// How a point moves per unit distance as the translator along a unit direction carries it:
    /// its commutator with the translator's generator, the dual of the line joining the point at
    /// infinity with the direction.
    pub fn velocity(start: Point, direction: Direction) -> Point {
        start.commutator((infinity() & Quadvector::from(direction)).hodge())
    }

    /// The point a distance along a ray: its start carried by the translator along its unit
    /// direction, written out for a start of unit weight: the start, plus the distance times its
    /// velocity, plus half the distance squared times the point at infinity.
    pub fn ahead(start: Point, direction: Direction, distance: f64) -> Point {
        start + velocity(start, direction) * distance + infinity() * (distance * distance / 2.0)
    }

    /// A solid: the intersection of its quadrics, each negative inside it, within a bounding
    /// sphere; its colour, the tint glass takes on per unit length; the share of light it lets
    /// through and its refractive index; two planes whose unit steps lay a checker over it.
    pub struct Solid {
        pub quadrics: Vec<Quadric>,
        pub centre: Point,
        pub radius: f64,
        pub colour: [f32; 3],
        pub transparency: f64,
        pub refraction: f64,
        pub across: Vector,
        pub along: Vector,
    }

    /// How far a point lies outside a solid: the largest of its quadrics' forms, negative inside
    /// them all.
    pub fn outside(solid: &Solid, point: Point) -> f64 {
        solid
            .quadrics
            .iter()
            .map(|q| (point & q.of(point)).s())
            .fold(-FAR, f64::max)
    }

    /// The first distance along a ray at which it crosses a solid's surface.
    pub fn crossing(solid: &Solid, start: Point, direction: Direction) -> Option<f64> {
        // The stretch of the ray inside the bounding sphere: how far along the ray the sphere's
        // centre lies, and the square of half the chord.
        let bounding = sphere(solid.centre, solid.radius * solid.radius);
        let along = (bounding & velocity(start, direction)).s();
        let half_chord_squared = along * along + 2.0 * (bounding & start).s();
        if half_chord_squared <= 0.0 {
            return None;
        }
        #[allow(clippy::disallowed_methods)] // half the chord, from its square
        let half_chord = half_chord_squared.sqrt();
        let (first, last) = ((along - half_chord).max(CLEARANCE), along + half_chord);
        if last <= first {
            return None;
        }
        // The ray's points, `ahead` with the velocity and the point at infinity taken once.
        let (velocity, infinity) = (velocity(start, direction), infinity());
        let point = |d: f64| start + velocity * d + infinity * (d * d / 2.0);
        let out = |d: f64| outside(solid, point(d)) > 0.0;
        let started_outside = out(first);
        let mut before = first;
        for step in 1..=N_STEPS {
            let after = first + (last - first) * step as f64 / N_STEPS as f64;
            if out(after) != started_outside {
                let (mut low, mut high) = (before, after);
                for _ in 0..N_HALVINGS {
                    let middle = (low + high) / 2.0;
                    if out(middle) == started_outside {
                        low = middle;
                    } else {
                        high = middle;
                    }
                }
                return Some(high);
            }
            before = after;
        }
        None
    }

    /// The nearest crossing of any solid along a ray: its distance and solid.
    pub fn nearest(solids: &[Solid], start: Point, direction: Direction) -> Option<(f64, usize)> {
        solids
            .iter()
            .enumerate()
            .filter_map(|(k, s)| crossing(s, start, direction).map(|d| (d, k)))
            .min_by(|a, b| a.0.total_cmp(&b.0))
    }

    /// A solid's outward normal at a point on it, as the unit plane through the origin parallel
    /// to the tangent plane: from the tangent sphere of the quadric the point lies least inside.
    pub fn normal(solid: &Solid, point: Point) -> Vector {
        let form = |q: &Quadric| (point & q.of(point)).s();
        let surface = solid
            .quadrics
            .iter()
            .max_by(|a, b| form(a).total_cmp(&form(b)))
            .expect("a solid has a quadric");
        flattened(surface.of(point), point).normalized()
    }

    // --- the scene

    /// The torus the cyclide comes from, and the sphere it is inverted in: off the torus's
    /// centre, so the tube swells on the far side.
    pub const TORUS_RADIUS: f64 = 2.5;
    pub const TORUS_TUBE: f64 = 0.9;
    pub fn inversion() -> Vector {
        (translator(along(x(), 8.0)) >> sphere(origin(), 6.25 * 6.25)).normalized()
    }
    /// Where the inverted torus lies, and how it is stood up above the floor: the centre of its
    /// bounding sphere carried by the lift, after a tilt about x.
    pub const CYCLIDE_CENTRE: f64 = 2.04;
    pub const CYCLIDE_RADIUS: f64 = 2.6;
    pub const CYCLIDE_TILT: f64 = 55.0;
    pub const LIFT: f64 = 2.25;
    /// A ball taken out of the cyclide's thick side, and a ball resting on the floor behind it.
    pub const BITE_CENTRE: [f64; 3] = [0.6, 1.2, 0.9];
    pub const BITE_RADIUS: f64 = 0.9;
    pub const BALL_CENTRE: [f64; 3] = [0.6, 3.4, 1.1];
    pub const BALL_RADIUS: f64 = 1.1;
    pub const FLOOR_RADIUS: f64 = 60.0;
    /// Each solid's colour or glass tint, transparency and refractive index: cyclide, ball,
    /// floor.
    pub const COLOURS: [[f32; 3]; 3] = [[0.9, 0.96, 0.97], [0.85, 0.22, 0.16], [0.88, 0.87, 0.85]];
    pub const TRANSPARENCIES: [f64; 3] = [0.96, 0.0, 0.0];
    pub const REFRACTIONS: [f64; 3] = [1.5, 1.0, 1.0];
    pub const TILES: f64 = 0.8;
    /// The lamp: `x` raised towards `z` by its elevation, then turned towards `y` by its
    /// azimuth, in degrees.
    pub const LAMP_ELEVATION: f64 = 49.0;
    pub const LAMP_AZIMUTH: f64 = -50.0;
    /// The camera orbits the cyclide: its target's height, distance and elevation, and the
    /// tangent of half its view.
    pub const TARGET_HEIGHT: f64 = 1.9;
    pub const DISTANCE: f64 = 11.0;
    pub const ELEVATION: f64 = 20.0;
    pub const HALF_VIEW: f64 = 0.31;
    pub const N_FRAMES: usize = 96;

    /// The point at an offset from the origin.
    pub fn at([a, b, c]: [f64; 3]) -> Point {
        translator(along(x(), a) + along(y(), b) + along(z(), c)) >> origin()
    }

    /// The lamp's direction (turned once, then kept).
    pub fn lamp() -> Direction {
        static LAMP: std::sync::OnceLock<Direction> = std::sync::OnceLock::new();
        *LAMP.get_or_init(|| {
            let turn = turn(x(), y(), LAMP_AZIMUTH.to_radians())
                * turn(x(), z(), LAMP_ELEVATION.to_radians());
            (turn >> Quadvector::from(direction_of(x()))).cast::<Direction>()
        })
    }

    /// The cyclide's motor: its bounding centre to the origin, tilted about x, lifted above
    /// the floor.
    pub fn placement() -> Motor {
        translator(along(z(), LIFT))
            * turn(y(), z(), CYCLIDE_TILT.to_radians())
            * translator(along(x(), -CYCLIDE_CENTRE))
    }

    /// The cyclide: the solid torus inverted in the sphere, a versor.
    pub fn cyclide() -> Quadric {
        let inversion = inversion();
        inversion >> torus(TORUS_RADIUS, TORUS_TUBE).of(inversion << Quadvector::slot())
    }

    /// Every solid: the glass cyclide less the bite, the ball, the floor.
    pub fn solids() -> Vec<Solid> {
        let place = placement();
        let none = Vector::new(0.0, 0.0, 0.0, 0.0, 0.0);
        let solid = |quadrics, k: usize, centre, radius, across, along| Solid {
            quadrics,
            centre,
            radius,
            colour: COLOURS[k],
            transparency: TRANSPARENCIES[k],
            refraction: REFRACTIONS[k],
            across,
            along,
        };
        let bite = place >> sphere(at(BITE_CENTRE), BITE_RADIUS * BITE_RADIUS);
        vec![
            solid(
                vec![placed(place, cyclide()), -ball(bite)],
                0,
                translator(along(z(), LIFT)) >> origin(),
                CYCLIDE_RADIUS,
                none,
                none,
            ),
            solid(
                vec![ball(sphere(at(BALL_CENTRE), BALL_RADIUS * BALL_RADIUS))],
                1,
                at(BALL_CENTRE),
                BALL_RADIUS * 1.01,
                none,
                none,
            ),
            // The floor: everything below the plane z = 0.
            solid(
                vec![ball(-z())],
                2,
                origin(),
                FLOOR_RADIUS,
                x() * TILES,
                y() * TILES,
            ),
        ]
    }

    /// The camera at a frame of its orbit: a turn from one looking along `−z` with `y` up, and
    /// an eye.
    pub fn camera(frame: usize) -> (Motor, Point) {
        let azimuth = core::f64::consts::TAU * frame as f64 / N_FRAMES as f64;
        let pitch = turn(
            y(),
            z(),
            core::f64::consts::FRAC_PI_2 - ELEVATION.to_radians(),
        );
        let turn = turn(x(), y(), azimuth) * pitch;
        let back = (turn >> Quadvector::from(direction_of(z()))).cast::<Direction>() * DISTANCE;
        let target = translator(along(z(), TARGET_HEIGHT)) >> origin();
        (turn, translator(back) >> target)
    }
}

use cga::{Direction, Point, Quadvector, Vector, dual_inverse, z};
use cyclide::*;

/// The loop: numga's 96 frames of 40 ms, once round.
const SECONDS: f32 = 3.84;
/// The look: light where the lamp does not reach, the light glass lets into its shadow, the
/// checker's darkening; the sun seen in a reflection, its brightness, and how tightly it
/// gathers about the lamp's direction; the intensity of each solid's light.
const LOOK: tracing::Look = tracing::Look {
    ambient: 0.3,
    checker: 0.18,
};
const GLASS_SHADOW: f64 = 0.75;
const SUN: f64 = 6.0;
const SUN_SIZE: i32 = 2000;
const INTENSITY: [f32; 3] = [0.9, 0.9, 0.14];

/// A light point, the radiance a path gathers.
type Radiance = ops::Raw;
/// A filter on light points: the share of each colour a path passes on.
type Filter = ops::Map;

/// The filter passing each colour by its share.
fn filter([r, g, b]: [f64; 3]) -> Filter {
    ops::colour_map([[r, 0.0, 0.0], [0.0, g, 0.0], [0.0, 0.0, b]])
}

/// The sky along a direction: the navy by how far it looks up, and the sun where it looks into
/// the lamp.
fn sky(direction: Direction) -> Light {
    let up = (z() & Quadvector::from(direction)).s();
    let sunward = (dual_inverse(Quadvector::from(lamp())) & Quadvector::from(direction)).s();
    tracing::sky(up) + tracing::glint(SUN * sunward.max(0.0).powi(SUN_SIZE))
}

/// The light towards the lamp from a point: dimmed by each solid in the way, opaque ones
/// fully.
fn sunlight(solids: &[Solid], point: Point) -> f64 {
    solids
        .iter()
        .filter(|s| crossing(s, point, lamp()).is_some())
        .map(|s| s.transparency * GLASS_SHADOW)
        .product()
}

/// An opaque surface's light at a point: lit by the lamp, shadowed, and checkered.
fn surface(solids: &[Solid], k: usize, point: Point, facing: Vector) -> Light {
    let solid = &solids[k];
    let off = ahead(point, cga::direction_of(facing), CLEARANCE);
    let lit = (facing & Quadvector::from(lamp())).s().max(0.0) * sunlight(solids, off);
    let parity = tracing::parity((solid.across & point).s(), (solid.along & point).s());
    let [r, g, b] = solid.colour;
    LOOK.surface(Light::from_srgb(r, g, b, INTENSITY[k]), parity, lit)
}

/// What a ray glimpses without passing through glass: the surface it meets, or the sky.
fn glimpse(solids: &[Solid], start: Point, direction: Direction) -> Light {
    let Some((distance, k)) = nearest(solids, start, direction) else {
        return sky(direction);
    };
    let point = ahead(start, direction, distance);
    let facing = normal(&solids[k], point);
    surface(solids, k, point, facing).mix_light(sky(direction), solids[k].transparency as f32)
}

/// The light along the ray through a place on the picture, from the camera's own `−z` turned by
/// its rotor: a path through the glass, refracting and gathering what its reflections glimpse.
fn sample(solids: &[Solid], (turn, eye): (cga::Motor, Point), place: [f64; 2]) -> Option<Light> {
    let [a, b] = place.map(|v| v * HALF_VIEW);
    let toward = turn
        >> Quadvector::from(
            cga::direction_of(-z())
                + cga::direction_of(cga::x()) * a
                + cga::direction_of(cga::y()) * b,
        );
    // A direction's length squared is its pairing with the plane square to it.
    #[allow(clippy::disallowed_methods)] // a direction's length, from its square
    let length = (dual_inverse(toward) & toward).s().sqrt();
    let mut direction = (toward * (1.0 / length)).cast::<Direction>();
    let mut position = eye;
    let mut colour = Radiance::new(0.0, 0.0, 0.0, 0.0);
    let mut throughput = filter([1.0; 3]);
    for bounce in 0..N_BOUNCES {
        let Some((distance, k)) = nearest(solids, position, direction) else {
            // The primary ray's sky is the backdrop.
            if bounce == 0 {
                return None;
            }
            colour += throughput.of(sky(direction).point());
            break;
        };
        let solid = &solids[k];
        position = ahead(position, direction, distance);
        let outward = normal(solid, position);
        let entering = (outward & Quadvector::from(direction)).s() < 0.0;
        // Light that crossed glass to get here faded by its tint, per unit length.
        if !entering && solid.transparency > 0.0 {
            throughput = throughput.of(filter(solid.colour.map(|c| f64::from(c).powf(distance))));
        }
        let facing = if entering { outward } else { -outward };
        let opaque =
            surface(solids, k, position, facing).point() * (1.0 - solid.transparency) as f32;
        colour += throughput.of(opaque);
        // At glass the ray refracts, by Snell's law, and reflects in the plane along the
        // surface; past the critical angle all of it reflects.
        let cosine = -(facing & Quadvector::from(direction)).s();
        let ratio = if entering {
            1.0 / solid.refraction
        } else {
            solid.refraction
        };
        let sine_squared = ratio * ratio * (1.0 - cosine * cosine);
        let reflected = (facing >> Quadvector::from(direction)).cast::<Direction>();
        #[allow(clippy::disallowed_methods)] // Snell's law
        let across = (1.0 - sine_squared).max(0.0).sqrt();
        let refracted = direction * ratio + cga::direction_of(facing) * (ratio * cosine - across);
        // Schlick's approximation to Fresnel's share of reflected light.
        let normal_reflectance = ((solid.refraction - 1.0) / (solid.refraction + 1.0)).powi(2);
        let reflectance = normal_reflectance + (1.0 - normal_reflectance) * (1.0 - cosine).powi(5);
        let total = sine_squared > 1.0;
        let share = if total { 0.0 } else { reflectance };
        let glimpsed = glimpse(solids, ahead(position, reflected, CLEARANCE), reflected);
        colour += throughput.of(glimpsed.point() * (solid.transparency * share) as f32);
        let passed = solid.transparency * if total { 1.0 } else { 1.0 - reflectance };
        throughput = throughput * passed as f32;
        direction = if total { reflected } else { refracted };
        position = ahead(position, direction, CLEARANCE);
        if passed < 0.01 || solid.transparency == 0.0 {
            break;
        }
    }
    Some(Light::from_point(colour))
}

struct Scenes {
    solids: Vec<Solid>,
}

fn scenes() -> Scenes {
    Scenes { solids: solids() }
}

/// The scene from a camera over a picture.
fn trace(c: &mut Canvas, picture: Rect, solids: &[Solid], camera: (cga::Motor, Point)) {
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
        "GLASS CYCLIDE: A TORUS INVERTED IN A SPHERE",
        "QUADRICS Sphere <- Point OF CGA3D x+y+z+w+e-; RAYS MARCHED BY TRANSLATORS, REFRACTED (SNELL, SCHLICK)",
    );
}

fn main() {
    run_with(
        Anim::new("glass_cyclide", SECONDS).size(480, 270).scale(2),
        scenes,
        draw,
    );
}

#[cfg(test)]
mod tests {
    use super::cga::*;
    use super::cyclide::*;
    use gax_numga_examples::{Canvas, Rect};

    const ROUND_OFF: f64 = 1e-10;

    /// The conformal basics read as numga's: the origin is the dual of `(w + e) / 2`, numga's
    /// `−(xyzw + xyze) / 2`, and the plane at infinity reads its weight as numga's does, `−1`; a
    /// point carried along a ray is the translator's.
    #[test]
    fn the_conformal_basics() {
        let o = origin();
        assert!(
            (dual_inverse(o) - (w() + e()) * 0.5)
                .c
                .iter()
                .all(|c| c.abs() < ROUND_OFF)
        );
        assert_eq!(o.c, [0.0, 0.0, 0.0, -0.5, -0.5]);
        assert!(((horizon() & o).s() + 1.0).abs() < ROUND_OFF);
        assert!(
            (o & dual_inverse(o)).s().abs() < ROUND_OFF,
            "the origin is null"
        );
        let d = along(x(), 0.6) + along(y(), -0.48) + along(z(), 0.64);
        let start = at([1.0, -2.0, 0.5]);
        let moved = translator(d * 2.5) >> start;
        let marched = ahead(start, d, 2.5);
        assert!(
            (moved - marched).c.iter().all(|c| c.abs() < ROUND_OFF),
            "{moved:?} {marched:?}"
        );
        // The plane across z faces up, and reflects a ray going down back up.
        let up = direction_of(z());
        assert!(((z() & Quadvector::from(up)).s() - 1.0).abs() < ROUND_OFF);
        let down = along(x(), 1.0) - along(z(), 1.0);
        let bounced = (z() >> Quadvector::from(down)).cast::<Direction>();
        let expected = along(x(), 1.0) + along(z(), 1.0);
        assert!(
            (bounced - expected).c.iter().all(|c| c.abs() < ROUND_OFF),
            "{bounced:?}"
        );
    }

    /// The torus's points, inverted in the sphere, lie on the cyclide. The form scales with the
    /// square of a point's weight, its pairing with the plane at infinity.
    #[test]
    fn inverted_torus_lies_on_the_cyclide() {
        let inversion = inversion();
        let surface = cyclide();
        for (around, across) in [(16, 8), (24, 12)] {
            for p in torus_points(TORUS_RADIUS, TORUS_TUBE, around, across)
                .iter()
                .flatten()
            {
                let q = inversion >> *p;
                let form = (q & surface.of(q)).s() / (horizon() & q).s().powi(2);
                assert!(form.abs() < ROUND_OFF, "{form}");
            }
        }
        // The core circle of the tube is inside the torus, and stays inside the cyclide.
        let core = translator(along(x(), TORUS_RADIUS)) >> origin();
        assert!((core & torus(TORUS_RADIUS, TORUS_TUBE).of(core)).s() < 0.0);
        let inside = inversion >> core;
        assert!((inside & surface.of(inside)).s() < 0.0);
    }

    /// The tangent sphere gives the outward direction: square to the sphere, along the radius,
    /// pointing out of it; inside the ball the form is negative, outside positive, and a negated
    /// quadric swaps them.
    #[test]
    fn tangent_sphere_gives_the_outward_direction() {
        let centre = translator(along(x(), 1.0) - along(z(), 0.5)) >> origin();
        let solid = ball(sphere(centre, 0.64));
        let radial = (along(x(), 0.6) + along(y(), 0.48) + along(z(), 0.64)) * 0.8;
        let point = translator(radial) >> centre;
        let direction = outward(solid.of(point), point);
        let across = Quadvector::from(direction) & Quadvector::from(radial);
        assert!(across.c.iter().all(|c| c.abs() < ROUND_OFF), "{across:?}");
        let beyond = translator(direction * 0.01) >> point;
        assert!((beyond & solid.of(beyond)).s() > 0.0);
        let outside = translator(along(x(), 2.0)) >> centre;
        assert!((centre & solid.of(centre)).s() < 0.0 && 0.0 < (outside & solid.of(outside)).s());
        assert!((centre & (-solid).of(centre)).s() > 0.0);
    }

    /// The red ball shows through the ring's hole, in the middle of the first frame.
    #[test]
    fn trace_shows_the_ball_through_the_ring() {
        let solids = solids();
        let (w, h) = (144, 90);
        let mut c = Canvas::with_scale(w, h, 1.0);
        super::trace(
            &mut c,
            Rect::new(0.0, 0.0, w as f32, h as f32),
            &solids,
            camera(0),
        );
        let mut redness: Vec<f32> = (32..44)
            .flat_map(|y| (74..88).map(move |x| (x, y)))
            .map(|(x, y)| {
                let [r, _, b, _] = c.get(x, y).to_premultiplied();
                r - b
            })
            .collect();
        redness.sort_by(f32::total_cmp);
        let median = redness[redness.len() / 2];
        assert!(median > 0.05, "{median}");
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws_with(super::draw, &super::scenes(), 1.0);
    }
}
