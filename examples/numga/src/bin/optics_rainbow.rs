//! numga's `optics/rainbow`: sunlight refracted into a raindrop, reflected inside it once or
//! twice, and refracted out again (VGA3D).
//!
//! Sunlight entering a spherical drop is bent at its surface, reflected from the back and bent
//! again on the way out, and comes back towards the sun turned by an angle that depends on how
//! far from the centre it entered. That angle has an extreme: light entering near one distance
//! leaves at nearly one angle and piles up into the bow, about 42 degrees from the point
//! opposite the sun. Light reflected twice has its own extreme near 51 degrees, the fainter
//! secondary bow, and none leaves between the two: Alexander's dark band. Water bends blue light
//! more than red, so each colour has its bow at its own angle.
//!
//! Directions are antivectors, intrinsic bivectors, and so are the points of the drop's surface,
//! as directions from its centre. Half a turn about the surface's normal, `normal >> direction`
//! (the normal as a rotor, a unit bivector), reverses the part of a direction along the surface:
//! the part along it is half the difference, and Snell's law keeps that part times the
//! refractive index, which holds the law of sines and the plane of incidence at once; unit
//! length fixes the rest. Inside a sphere every chord meets the surface at the same angle at both
//! ends, so each point where the light meets the surface, and each chord, is the last one turned
//! about the centre by one rotor: the path inside the drop is the powers of one turn. Fresnel's
//! coefficients give the light that each pass keeps.
//!
//! The animation shows numga's two: light swept across the drop beside the angle each colour
//! leaves at against where it entered, and the double bow on the sky opposite the sun, the light
//! of 4000 rays per colour gathered into rings of a twentieth of a degree and spread over the
//! sun's disc.

use gax_numga_examples::{
    Align, Anim, Axes, Canvas, Light, Marker, Point2, Rect, backdrop, caption, palette, run_with,
    signal::frame,
};

mod rainbow {
    use gax::Unit;
    use gax::vga3d::{Rotor, Vector, intrinsic};

    /// A direction, or a point of the drop's surface as the direction from its centre: an
    /// antivector, intrinsic; unit ones square to minus one.
    pub type Direction = intrinsic::Bivector;
    /// A turn about the drop's centre.
    pub type Turn = Unit<Rotor>;

    /// The direction along a vector: numga's metric dual.
    pub fn direction(v: Vector) -> Direction {
        v.hodge()
    }

    /// The length of the other leg of a right triangle with a unit hypotenuse and a leg of
    /// squared length `squared`: the cosine from the squared sine.
    #[allow(clippy::disallowed_methods)] // the unit length that fixes a part along a normal
    fn other_leg(squared: f64) -> f64 {
        (1.0 - squared).sqrt()
    }

    /// Half a turn about a unit direction: the direction as a rotor.
    fn half_turn(axis: Direction) -> Turn {
        axis.with_orientation::<gax::Extrinsic>()
            .cast::<Rotor>()
            .unit()
    }

    /// The direction of light crossing a surface with unit `normal`, pointing into the medium it
    /// enters, which is `ratio` times as refractive as the one it leaves.
    pub fn refract(light: Direction, normal: Direction, ratio: f64) -> Direction {
        // Half a turn about the normal reverses the part along the surface; Snell's law keeps
        // that part times the refractive index.
        let surface_part = (light - (half_turn(normal) >> light)) * (0.5 / ratio);
        surface_part + normal * other_leg(surface_part.norm_squared())
    }

    /// Where light along `sunlight`, passing the centre of the unit drop at `offset` square to
    /// it, meets the drop: the point on its sunlit side.
    pub fn entry(sunlight: Direction, offset: Direction) -> Direction {
        offset - sunlight * other_leg(offset.norm_squared())
    }

    /// The turn about the drop's centre carrying a point where light meets the surface from
    /// inside, and the chord arriving there, to the next point and the next chord.
    pub fn bounce(vertex: Direction, chord: Direction) -> Turn {
        // The next point is this one reflected across the plane through the centre square to
        // the chord.
        let next = -(half_turn(chord) >> vertex);
        // The rotor taking one unit direction to the other, in their plane.
        (-(next * vertex)).sqrt()
    }

    /// The fraction of light the surface reflects, polarized across and along the plane of
    /// incidence (Fresnel's coefficients), the same wherever the light meets it again.
    pub fn reflectances(offset: Direction, index: f64) -> (f64, f64) {
        let sine = offset.norm_squared();
        let incidence = other_leg(sine);
        let refraction = other_leg(sine / (index * index));
        let across = (incidence - index * refraction) / (incidence + index * refraction);
        let along = (refraction - index * incidence) / (refraction + index * incidence);
        (across * across, along * along)
    }

    /// Light leaving the drop after some reflections inside: where it leaves, the direction it
    /// leaves in, and the fraction of unpolarized sunlight leaving there.
    #[derive(Clone, Copy, Debug)]
    pub struct Exit {
        pub vertex: Direction,
        pub exit: Direction,
        pub passed: f64,
    }

    /// For each number of reflections inside the drop, from none up to `reflections`, the light
    /// leaving.
    pub fn paths(
        sunlight: Direction,
        offset: Direction,
        index: f64,
        reflections: usize,
    ) -> Vec<Exit> {
        let mut vertex = entry(sunlight, offset);
        let mut chord = refract(sunlight, -vertex, index);
        // Every chord of a sphere meets it at one angle at both ends: each point and chord
        // inside is the last one turned by the same rotor.
        let turn = bounce(vertex, chord);
        let (across, along) = reflectances(offset, index);
        // The light transmitted into the drop and out again, in each polarization.
        let (mut passed_across, mut passed_along) = ((1.0 - across).powi(2), (1.0 - along).powi(2));
        (0..=reflections)
            .map(|_| {
                vertex = turn >> vertex;
                let exit = Exit {
                    vertex,
                    exit: refract(chord, vertex, 1.0 / index),
                    passed: (passed_across + passed_along) / 2.0,
                };
                chord = turn >> chord;
                (passed_across, passed_along) = (passed_across * across, passed_along * along);
                exit
            })
            .collect()
    }

    /// The cosine of the angle between two unit directions: numga's `dot`, `⟨a ~b⟩₀`.
    pub fn cosine(a: Direction, b: Direction) -> f64 {
        a.scalar_product(b.reverse()).s()
    }

    /// The angle in degrees from the point opposite the sun at which light leaving along `exit`
    /// is seen: looking back against it, from the sunlight.
    pub fn sight_angle(exit: Direction) -> f64 {
        (-exit)
            .hodge()
            .angle_between(forward().hodge())
            .to_degrees()
    }

    // --- scenarios

    /// The direction sunlight travels, and the direction up, square to it.
    pub fn forward() -> Direction {
        direction(Vector::new(1.0, 0.0, 0.0))
    }
    pub fn up() -> Direction {
        direction(Vector::new(0.0, 1.0, 0.0))
    }

    /// The sunlight's wavelengths in nm, and the refractive index of water for each: Cauchy's
    /// fit, a constant and a term falling with the wavelength squared (nm²).
    pub const COLOURS: usize = 31;
    pub fn wavelength(colour: usize) -> f64 {
        400.0 + 300.0 * colour as f64 / (COLOURS - 1) as f64
    }
    pub fn index(colour: usize) -> f64 {
        1.3238 + 3140.0 / wavelength(colour).powi(2)
    }
    /// The index the swept light is drawn for, that of yellow light.
    pub const DRAWN_INDEX: f64 = 1.333;
    /// Rays across the drop's radius for the sky, frames of the sweep, rays of the bundle.
    pub const RAYS: usize = 4000;
    pub const FRAMES: usize = 144;
    pub const BUNDLE: usize = 48;
    /// Reflections inside: once for the primary bow, twice for the secondary.
    pub const REFLECTIONS: usize = 2;
    /// The sun's height and the width of its disc, in degrees.
    pub const SUN_ELEVATION: f64 = 12.0;
    pub const SUN_WIDTH: f64 = 0.53;

    /// Offsets at the middle of `count` equal steps across the drop's radius.
    pub fn offsets(count: usize) -> Vec<Direction> {
        (0..count)
            .map(|k| up() * ((k as f64 + 0.5) / count as f64))
            .collect()
    }

    /// The sky: per number of reflections (from one), per colour, per ray, the cosine of the
    /// angle from the point opposite the sun at which the light is seen, the angle in degrees,
    /// and the sunlight it carries, from the ring of offsets of its size.
    pub struct Sky {
        #[cfg_attr(not(test), allow(dead_code))] // numga's check of Descartes's angles
        pub cosines: Vec<Vec<Vec<f64>>>,
        pub angles: Vec<Vec<Vec<f64>>>,
        pub flux: Vec<Vec<Vec<f64>>>,
    }

    pub fn sky(offsets: &[Direction]) -> Sky {
        let mut cosines = vec![vec![Vec::new(); COLOURS]; REFLECTIONS];
        let mut angles = cosines.clone();
        let mut flux = vec![vec![Vec::new(); COLOURS]; REFLECTIONS];
        for colour in 0..COLOURS {
            for offset in offsets {
                let bows = paths(forward(), *offset, index(colour), REFLECTIONS);
                for (order, bow) in bows[1..].iter().enumerate() {
                    // Light leaving along `exit` is seen looking back against it; the point
                    // opposite the sun lies along the sunlight.
                    cosines[order][colour].push(cosine(-bow.exit, forward()));
                    angles[order][colour].push(sight_angle(bow.exit));
                    flux[order][colour].push(bow.passed * offset.norm());
                }
            }
        }
        Sky {
            cosines,
            angles,
            flux,
        }
    }

    /// Light at an offset as drawn: where it enters, and the light leaving after each number of
    /// reflections.
    #[derive(Clone, Debug)]
    pub struct Swept {
        pub entry: Direction,
        pub exits: Vec<Exit>,
    }

    pub fn sweep(offsets: &[Direction]) -> Vec<Swept> {
        let swept = offsets.iter().map(|o| Swept {
            entry: entry(forward(), *o),
            exits: paths(forward(), *o, DRAWN_INDEX, REFLECTIONS),
        });
        swept.collect()
    }
}

/// The colours of light: CIE 1931's colour matching functions as gax_colour's XYZ.
mod spectrum {
    use gax_colour::{Light, LinearRgb, Oklab, Xyz};

    /// The CIE 1931 colour matching functions at a wavelength in nm, by the multi-lobe fit of
    /// Wyman, Sloan and Shirley (2013): `x̄, ȳ, z̄`.
    pub fn matching(wavelength: f64) -> [f64; 3] {
        let lobe = |centre: f64, below: f64, above: f64| {
            let width = if wavelength < centre { below } else { above };
            (-0.5 * ((wavelength - centre) / width).powi(2)).exp()
        };
        [
            1.056 * lobe(599.8, 37.9, 31.0) + 0.362 * lobe(442.0, 16.0, 26.7)
                - 0.065 * lobe(501.1, 20.4, 26.2),
            0.821 * lobe(568.8, 46.9, 40.5) + 0.286 * lobe(530.9, 16.3, 31.1),
            1.217 * lobe(437.0, 11.8, 36.0) + 0.681 * lobe(459.0, 26.0, 13.8),
        ]
    }

    /// Light of tristimulus values `xyz` (luminance at most one), within the display's gamut
    /// at its luminance.
    pub fn of_xyz([x, y, z]: [f64; 3]) -> Light {
        let xyz = Xyz::new(x as f32, y as f32, z as f32, 1.0);
        xyz.convert::<gax_colour::space::LinearRgb>().to_gamut()
    }

    /// The hue of a wavelength at one perceived lightness: its colour in Oklab at lightness
    /// `lightness`, brought into the gamut.
    pub fn hue(wavelength: f64, lightness: f32) -> Light {
        let [x, y, z] = matching(wavelength);
        let xyz = Xyz::new(x as f32, y as f32, z as f32, 1.0);
        let even: Oklab = xyz
            .convert::<gax_colour::space::Oklab>()
            .with_lightness(lightness);
        let light: LinearRgb = even.convert();
        light.to_gamut()
    }
}

use rainbow::*;

/// The loop: numga's sweep, 144 frames of 40 ms.
const SECONDS: f32 = 5.76;
/// How far light is drawn outside the drop, in drop radii.
const RAY_LENGTH: f64 = 2.4;
/// The sky's rings, in degrees, the half width of its view, and the power its light is raised
/// to for display, brightening faint light more than bright.
const BIN: f64 = 0.05;
const BINS: usize = 1800;
const VIEW_HALF_WIDTH: f32 = 62.0;
const TONE: f32 = 0.65;
/// The intensity of the brightest ring.
const SKY_GAIN: f32 = 0.45;
/// Every so many rays of a colour's angle curve drawn.
const CURVE_STEP: usize = 10;

/// The data, computed once: the sweep and the bundle, each colour's angle curves, and the
/// light of the sky's rings.
struct Scenes {
    swept: Vec<Swept>,
    bundle: Vec<Swept>,
    /// Per order, per colour: (offset, angle in degrees) along the curve.
    curves: Vec<Vec<Vec<Point2>>>,
    /// The bows' extreme angles, in degrees.
    bows: [f64; 2],
    rings: Vec<Light>,
}

fn scenes() -> Scenes {
    let rays = offsets(RAYS);
    let sky = sky(&rays);
    let curves = sky
        .angles
        .iter()
        .map(|per_colour| {
            per_colour
                .iter()
                .map(|angles| {
                    let along = angles.iter().enumerate().step_by(CURVE_STEP);
                    let points = along.map(|(k, a)| {
                        let height = (k as f64 + 0.5) / RAYS as f64;
                        Point2::xy(height as f32, *a as f32)
                    });
                    points.collect()
                })
                .collect()
        })
        .collect();
    let middle = COLOURS / 2;
    let extreme = |order: usize, pick: fn(f64, f64) -> f64, start: f64| {
        let angles = sky.angles[order][middle].iter().copied();
        angles.fold(start, pick)
    };
    Scenes {
        swept: sweep(&offsets(FRAMES)),
        bundle: sweep(&offsets(BUNDLE)),
        curves,
        bows: [extreme(0, f64::max, 0.0), extreme(1, f64::min, 180.0)],
        rings: rings(&sky),
    }
}

/// The light per solid angle in each ring of the sky about the point opposite the sun, as a
/// light: each colour's rays gathered into rings `BIN` degrees wide, spread over the sun's disc,
/// divided by the ring's size, summed in XYZ over the colours, and toned.
fn rings(sky: &Sky) -> Vec<Light> {
    // The sun's disc across, its profile a half circle.
    let half = (SUN_WIDTH / 2.0 / BIN) as i64;
    let disc: Vec<f64> = (-half..=half)
        .map(|k| {
            let across = 2.0 * k as f64 * BIN / SUN_WIDTH;
            #[allow(clippy::disallowed_methods)] // the sun's disc, a half circle's profile
            let height = (1.0 - across * across).max(0.0).sqrt();
            height
        })
        .collect();
    let total: f64 = disc.iter().sum();
    let mut xyz = vec![[0.0; 3]; BINS];
    for colour in 0..COLOURS {
        let mut ring = vec![0.0; BINS];
        for (angles, flux) in sky.angles.iter().zip(&sky.flux) {
            for (a, f) in angles[colour].iter().zip(&flux[colour]) {
                let bin = (a / BIN) as usize;
                if bin < BINS {
                    ring[bin] += f;
                }
            }
        }
        let matching = spectrum::matching(wavelength(colour));
        for (bin, sum) in xyz.iter_mut().enumerate() {
            let spread: f64 = disc
                .iter()
                .enumerate()
                .filter_map(|(k, d)| {
                    let from = (bin as i64 + k as i64 - half).clamp(-1, BINS as i64);
                    usize::try_from(from)
                        .ok()
                        .and_then(|i| ring.get(i))
                        .map(|r| r * d / total)
                })
                .sum();
            let angle = (bin as f64 + 0.5) * BIN;
            let size = gax_numga_examples::signal::wave(angle.to_radians());
            for (s, m) in sum.iter_mut().zip(matching) {
                *s += spread / size * m;
            }
        }
    }
    let brightest = xyz.iter().map(|c| c[1]).fold(0.0, f64::max);
    xyz.iter()
        .map(|c| {
            let luminance = (c[1] / brightest) as f32;
            let colour = spectrum::of_xyz(c.map(|v| v / brightest));
            // Toned: shown at the luminance raised to the tone's power.
            let gain = luminance.max(1e-6).powf(TONE - 1.0);
            colour.faded(SKY_GAIN * gain)
        })
        .collect()
}

/// The light of the sunlight, of light reflected once and twice, and of the drop.
fn sunlight() -> Light {
    palette::YELLOW
}
/// numga's ground, a dark green.
fn ground() -> Light {
    Light::from_srgb(0.08, 0.33, 0.18, 0.3)
}
fn primary() -> Light {
    palette::ORANGE
}
fn secondary() -> Light {
    palette::SKY
}

/// A point of the drop's cross-section along the sunlight and up, drawn.
fn flat(d: Direction) -> Point2 {
    use gax::vga3d::Vector;
    let along = |v: Vector| (v & d).s() as f32;
    Point2::xy(
        along(Vector::new(1.0, 0.0, 0.0)),
        along(Vector::new(0.0, 1.0, 0.0)),
    )
}

/// One ray's path, reflected `order` times inside: from the sun to the drop, the points where it
/// meets the surface, and on past the drop.
fn ray(swept: &Swept, order: usize) -> Vec<Point2> {
    let sunward = swept.entry - forward() * RAY_LENGTH;
    let mut path = vec![flat(sunward), flat(swept.entry)];
    path.extend(swept.exits[..=order].iter().map(|e| flat(e.vertex)));
    let last = swept.exits[order];
    path.push(flat(last.vertex + last.exit * RAY_LENGTH));
    path
}

/// The drop's cross-section: the bundle's light leaving after one and two reflections, and the
/// swept light's paths.
fn draw_drop(c: &mut Canvas, rect: Rect, s: &Scenes, now: usize) {
    let ax = Axes::equal(rect, Point2::xy(0.0, 0.0), (RAY_LENGTH * 0.9) as f32);
    let circle = gax_numga_examples::points::circle(Point2::xy(0.0, 0.0), 1.0, 96);
    ax.fill(c, &circle, palette::BLUE.faded(0.06), 1.0);
    ax.polyline(c, &circle, 1.0, palette::BLUE.faded(0.8));
    for (order, light) in [(1, primary()), (2, secondary())] {
        for swept in &s.bundle {
            let path = ray(swept, order);
            ax.polyline(c, &path[path.len() - 2..], 0.5, light.faded(0.35));
        }
    }
    let swept = &s.swept[now];
    for (order, light) in [(2, secondary()), (1, primary())] {
        ax.polyline(c, &ray(swept, order)[1..], 1.3, light);
    }
    ax.polyline(c, &ray(swept, 1)[..2], 1.3, sunlight());
    c.text(
        "SUNLIGHT",
        ax.px(Point2::xy(-2.0, 1.95)),
        10.0,
        sunlight(),
        Align::Left,
    );
}

/// The angle each colour is seen at against the offset of the light, after one and two
/// reflections, with the swept light's marks.
fn draw_angles(c: &mut Canvas, rect: Rect, s: &Scenes, now: usize) {
    let ax = Axes::new(rect, [0.0, 1.0], [0.0, 90.0]);
    for order_curves in &s.curves {
        for (colour, curve) in order_curves.iter().enumerate() {
            let hue = spectrum::hue(wavelength(colour), 0.75);
            ax.polyline(c, curve, 0.6, hue.faded(0.22));
        }
    }
    for bow in s.bows {
        ax.line(
            c,
            Point2::xy(0.0, bow as f32),
            Point2::xy(1.0, bow as f32),
            0.8,
            palette::GRID,
        );
        let label = format!("{bow:.0}");
        let at = Point2::xy(0.02, bow as f32 + 2.0);
        ax.text(c, at, &label, 9.0, palette::INK, Align::Left);
    }
    let swept = &s.swept[now];
    let height = (now as f32 + 0.5) / FRAMES as f32;
    for (order, light) in [(1, primary()), (2, secondary())] {
        let at = Point2::xy(height, sight_angle(swept.exits[order].exit) as f32);
        ax.scatter(c, &[at], Marker::Dot, 8.0, light.whitened(0.3));
    }
    ax.frame(c, "", "OFFSET OF THE LIGHT", "ANGLE FROM OPPOSITE THE SUN");
}

/// The sky opposite the sun: each pixel the light of the ring at its angle from the point
/// opposite the sun (as on a chart about that point, where the angle is the distance), above
/// the ground, which hides the point by the sun's height.
fn draw_sky(c: &mut Canvas, rect: Rect, s: &Scenes) {
    let elevation = SUN_ELEVATION as f32;
    let (bottom, top) = (elevation - 6.0, elevation + VIEW_HALF_WIDTH * 0.8);
    let half = 0.5 * (top - bottom);
    let ax = Axes::equal(rect, Point2::xy(0.0, bottom + half), half);
    let opposite = Point2::xy(0.0, 0.0);
    let night = palette::BOTTOM.faded(1.5);
    ax.image(c, 1, |q| {
        let [_, height] = q.to_euclidean();
        if height < elevation {
            return Some(ground());
        }
        let angle = f64::from(q.distance(opposite));
        let ring = (angle / BIN) as usize;
        Some(s.rings.get(ring).map_or(night, |l| night + *l))
    });
    let corners = [rect.lo, rect.top_right(), rect.hi, rect.bottom_left()];
    c.polyline(&corners, 1.0, palette::GRID, true);
}

fn draw(c: &mut Canvas, s: &Scenes, t: f32) {
    backdrop(c);
    let now = frame((t / SECONDS).rem_euclid(1.0), FRAMES);
    let screen = c.rect();
    let (w, h) = (screen.width(), screen.height());
    let top = Rect::new(0.0, h * 0.13, w, h * 0.55);
    draw_drop(c, top.column(0, 2).inset(8.0, 4.0, 8.0, 4.0), s, now);
    draw_angles(c, top.column(1, 2).inset(48.0, 18.0, 24.0, 30.0), s, now);
    let sky = Rect::new(w * 0.08, h * 0.6, w * 0.92, h * 0.97);
    draw_sky(c, sky, s);
    let at = sky.top_middle() - Point2::direction(0.0, 5.0);
    let title = "THE DOUBLE BOW OPPOSITE THE SUN, AND ALEXANDER'S DARK BAND BETWEEN";
    c.text(title, at, 10.0, palette::INK, Align::Center);
    caption(
        c,
        "THE RAINBOW: A RAINDROP'S PATHS AS POWERS OF ONE TURN",
        "SNELL KEEPS THE PART ALONG THE SURFACE, D - (N >> D) / 2, TIMES THE INDEX (VGA3D)",
    );
}

fn main() {
    run_with(Anim::new("rainbow", SECONDS).size(960, 540), scenes, draw);
}

#[cfg(test)]
mod tests {
    use super::rainbow::*;

    /// Each bow lies where the angle of the light leaving is stationary, Descartes's angle:
    /// after one reflection the largest from the point opposite the sun, after two the
    /// smallest, for every colour.
    #[test]
    fn bows_lie_at_descartes_angles() {
        let sky = sky(&offsets(RAYS));
        for colour in 0..COLOURS {
            let n = index(colour);
            for (order, cosines) in sky.cosines.iter().enumerate() {
                let k = (order + 1) as f64;
                #[allow(clippy::disallowed_methods)] // the closed form, Descartes's
                let deflection = {
                    let incidence = ((n * n - 1.0) / (k * (k + 2.0))).sqrt().acos();
                    let refraction = (incidence.sin() / n).asin();
                    2.0 * (incidence - refraction) + k * (core::f64::consts::PI - 2.0 * refraction)
                };
                #[allow(clippy::disallowed_methods)]
                let expected = -deflection.cos();
                let each = cosines[colour].iter().copied();
                let extreme = if order == 0 {
                    each.fold(1.0, f64::min)
                } else {
                    each.fold(-1.0, f64::max)
                };
                assert!((extreme - expected).abs() <= 1e-6, "{colour} {order}");
            }
        }
        // Each fraction of sunlight is less after more reflections.
        for colour in 0..COLOURS {
            let (once, twice) = (&sky.flux[0][colour], &sky.flux[1][colour]);
            assert!(once.iter().zip(twice).all(|(a, b)| b < a));
        }
    }

    /// Each chord inside, turned by the bounce, is the last one reflected off the surface where
    /// they meet, and the light meets the surface at one angle at both ends of every chord.
    #[test]
    fn light_inside_reflects_off_the_surface() {
        for k in 0..7 {
            let offset = up() * (0.1 + 0.85 * k as f64 / 6.0);
            let vertex = entry(forward(), offset);
            let chord = refract(forward(), -vertex, 1.333);
            let turn = bounce(vertex, chord);
            let next = turn >> vertex;
            let half_turn = next
                .with_orientation::<gax::Extrinsic>()
                .cast::<gax::vga3d::Rotor>();
            let reflected = (turn >> chord) + (half_turn.unit() >> chord);
            assert!(reflected.norm() <= 1e-12);
            assert!((cosine(chord, vertex) + cosine(chord, next)).abs() <= 1e-12);
        }
    }

    /// The light leaving is a unit direction.
    #[test]
    fn the_light_leaving_is_a_unit_direction() {
        for swept in sweep(&offsets(FRAMES)) {
            for e in &swept.exits {
                assert!((e.exit.norm_squared() - 1.0).abs() <= 1e-6);
            }
        }
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws_with(super::draw, &super::scenes(), 0.5);
    }
}
