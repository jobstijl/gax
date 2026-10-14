//! numga's `mechanics/foucault`: Foucault's pendulum in VGA3D. The plane a pendulum swings in
//! turns against its floor by the part of the earth's turn that lies in the floor.
//!
//! Space is the earth's: `x` and `y` span the equator, `x` through the prime meridian, and the
//! axis is `z`. The earth turns in the plane `x ^ y`. A pendulum hung at a site swings along a
//! line of its floor, the plane through the site across the local vertical, and nothing on the
//! pendulum can turn that line within the floor: the floor turns underneath it. The part of the
//! earth's turn that lies in the floor's own plane, the scalar product of `x ^ y` with the
//! floor's bivector over the floor's square, is the sine of the latitude in size (its sign
//! says the floor, north towards east, turns the other way round): at the pole all of the
//! earth's turn lies in the floor, at the equator none. So the swing turns against its floor at
//! that fraction of the earth's rate, clockwise from above in the north: once a day at the pole,
//! never at the equator, and in Paris three quarters of a turn a day.
//!
//! The same turn is a holonomy. Carry the swing's direction around the latitude circle, step
//! by step, each step the least turn `(1 + b a)` (normalized) that takes one site's vertical
//! `a` to the next `b`, and it comes back turned by the area of the polar cap the circle
//! encloses on the unit sphere, a full turn less the pendulum's. How far a direction on the
//! floor has turned is its signed angle to north in the floor's plane (gax's `angle_to`, the
//! `arctan2` numga takes).
//!
//! The animation runs through a day: on the left the earth turning the floor in Paris under the
//! pendulum's swing, with the latitude circle and its polar cap; on the right the bob's track on
//! the floor at four latitudes, coloured from the start of the day to now.

use gax::vga2d;

use gax_numga_examples::plot::Axes;
use gax_numga_examples::points;
use gax_numga_examples::scene3::panel3;
use gax_numga_examples::{
    Align, Anim, Camera, Canvas, Light, Point2, Point3, Rect, backdrop, caption, colormap, palette,
    reach2, reach3, run,
};
use std::sync::OnceLock;

#[path = "../shared/mechanics_vga3d.rs"]
mod space;

mod foucault {
    pub use super::space::{AXES, B, Turn, V, turn};
    use gax::vga3d::Rotor;

    /// The equator's `x` and `y`, and the axis `z`.
    pub const X: V = AXES[0];
    pub const Y: V = AXES[1];
    pub const Z: V = AXES[2];

    /// The plane the earth turns in.
    pub fn equator() -> B {
        X ^ Y
    }

    /// The direction from the earth's centre at a latitude and longitude (radians), the prime
    /// meridian raised towards the axis, then turned east.
    pub fn on_globe(latitude: f64, longitude: f64) -> V {
        turn(equator(), longitude) >> (turn(X ^ Z, latitude) >> X)
    }

    /// The site on the prime meridian at a latitude: its vertical, the direction north along
    /// its floor, and the direction east.
    pub struct Compass {
        pub vertical: V,
        pub north: V,
        pub east: V,
    }

    /// The compass of the site at a latitude (radians).
    pub fn compass(latitude: f64) -> Compass {
        let quarter = core::f64::consts::FRAC_PI_2;
        Compass {
            vertical: on_globe(latitude, 0.0),
            north: on_globe(latitude + quarter, 0.0),
            east: Y,
        }
    }

    /// How much of the earth's turn lies in a floor's plane: the earth's plane projected onto
    /// it, as a fraction of the floor's own turn.
    pub fn felt(floor: B) -> f64 {
        equator().scalar_product(floor).s() / floor.scalar_product(floor).s()
    }

    /// The direction a pendulum at a latitude swings along, on its floor, after the earth has
    /// turned by an angle: started north, and turned back against the floor by the part of the
    /// earth's turn the floor feels.
    pub fn swing(latitude: f64, earth_turn: f64) -> V {
        let Compass { north, east, .. } = compass(latitude);
        // The floor's plane, north towards east.
        let floor = north ^ east;
        turn(floor, -felt(floor) * earth_turn) >> north
    }

    /// The turn a direction on the floor makes when carried once around a latitude circle in
    /// so many steps, each the least turn taking one site's vertical to the next.
    pub fn carried(latitude: f64, steps: usize) -> Turn {
        let tau = core::f64::consts::TAU;
        let sites: Vec<V> = (0..=steps)
            .map(|k| on_globe(latitude, tau * k as f64 / steps as f64))
            .collect();
        let one = Rotor::new(1.0, 0.0, 0.0, 0.0);
        sites
            .windows(2)
            .map(|ab| (one + ab[1] * ab[0]).normalized())
            .fold(one.normalized(), |total, step| step * total)
    }

    /// The latitudes in degrees: the equator, thirty degrees north, the Panthéon in Paris
    /// where Foucault hung his pendulum, and the north pole.
    pub const LATITUDES: [f64; 4] = [0.0, 30.0, 48.85, 90.0];
    pub const PARIS: usize = 2;
    /// The swings drawn in a day, far fewer than the thousands a real pendulum makes, so that
    /// each stays apart; the samples of a day; and the steps a direction is carried around a
    /// latitude circle in.
    pub const SWINGS: f64 = 40.0;
    pub const SAMPLES: usize = 4000;
    pub const STEPS: usize = 720;

    /// The latitudes in radians.
    pub fn latitudes() -> [f64; 4] {
        LATITUDES.map(f64::to_radians)
    }

    /// Where each pendulum's bob is on its floor through the day `[latitude][sample]`, swinging
    /// to and fro along its turning line.
    pub fn rosettes() -> Vec<Vec<V>> {
        let tau = core::f64::consts::TAU;
        latitudes()
            .iter()
            .map(|latitude| {
                (0..SAMPLES)
                    .map(|k| {
                        let earth_turn = tau * k as f64 / (SAMPLES - 1) as f64;
                        let reach = gax_numga_examples::signal::wave(SWINGS * earth_turn);
                        swing(*latitude, earth_turn) * reach
                    })
                    .collect()
            })
            .collect()
    }
}

use foucault::*;

fn tracks() -> &'static Vec<Vec<V>> {
    static T: OnceLock<Vec<Vec<V>>> = OnceLock::new();
    T.get_or_init(rosettes)
}

/// The length of the loop in seconds: one day.
const LOOP: f32 = 16.0;
/// The globe's reach from its centre to the top of its panel, in earth radii.
const GLOBE_REACH: f32 = 1.4;
/// The half width of the floor drawn on the globe.
const FLOOR: f64 = 0.3;
/// The view of the globe, as numga's: turned from the prime meridian against the earth's turn
/// by 35 degrees, and raised by 22.
const AZIMUTH: f32 = (-35f32).to_radians();
const ELEVATION: f32 = 22f32.to_radians();
/// The meridians and parallels of the globe's surface, the meridians drawn as lines, and the
/// samples of a circle.
const MERIDIANS: usize = 36;
const PARALLELS: usize = 18;
const LINES: usize = 12;
const CIRCLE: usize = 96;
/// The radius at which the cap and the lines are drawn, just above the globe's surface so that
/// it does not cover them.
const ABOVE: f64 = 1.003;
/// The samples of a track drawn as one stroke of one colour.
const CHUNK: usize = 50;
/// The titles of the four floors.
const TITLES: [&str; 4] = ["EQUATOR", "30 N", "PARIS 48.85 N", "NORTH POLE"];

/// A curve just above the globe: the directions `at(angle)` for angles from 0 to a full turn.
fn round(at: impl Fn(f64) -> V) -> Vec<Point3> {
    let tau = core::f64::consts::TAU;
    (0..=CIRCLE)
        .map(|k| reach3(at(tau * k as f64 / CIRCLE as f64) * ABOVE))
        .collect()
}

/// East and north of a direction on a floor, as a vector of the floor's plane.
fn on_floor(v: V, site: &Compass) -> vga2d::Vector<(), f64> {
    vga2d::Vector::new((v | site.east).s(), (v | site.north).s())
}

/// How far a direction on its floor has turned from north, clockwise seen from above: the
/// signed turn from the direction to north.
fn turned(v: V, site: &Compass) -> f64 {
    on_floor(v, site).angle_to(vga2d::Vector::new(0.0, 1.0))
}

/// The colour of a track at a fraction of the day.
fn track_colour(day: f64) -> Light {
    colormap::viridis(0.25 + 0.75 * day as f32)
}

/// The globe turned by the earth's turn so far: its surface and meridians, the equator, the
/// site's latitude circle and the polar cap it bounds, and the site's floor with the line the
/// pendulum swings along and a tick to the north.
fn draw_globe(c: &mut Canvas, rect: Rect, earth_turn: f64) {
    let pi = core::f64::consts::PI;
    let tau = core::f64::consts::TAU;
    let paris = latitudes()[PARIS];
    let site = compass(paris);
    let earth = turn(equator(), earth_turn);
    let cam = Camera::parallel(rect, 0.5 * rect.height() / GLOBE_REACH, AZIMUTH, ELEVATION);
    panel3(c, rect, cam, |s| {
        // The surface, `u` along the longitude and `v` from the south pole up.
        let globe = |u: f32, v: f32| {
            let (u, v) = (f64::from(u), f64::from(v));
            reach3(on_globe(pi * (v - 0.5), tau * u))
        };
        let blue = |_: f32, _: f32| palette::blue().faded(0.15);
        s.surface(globe, MERIDIANS, PARALLELS, blue, 0.3, None);
        // The cap north of the site.
        let cap = |u: f32, v: f32| {
            let (u, v) = (f64::from(u), f64::from(v));
            reach3(on_globe(paris + (pi / 2.0 - paris) * v, tau * u) * ABOVE)
        };
        let sky = |_: f32, _: f32| palette::sky().faded(0.35);
        s.surface(cap, MERIDIANS, 6, sky, 0.45, None);
        // The meridians turn with the earth; the equator and the latitude circle stay.
        for k in 0..LINES {
            let longitude = tau * k as f64 / LINES as f64;
            let meridian = round(|a| earth >> on_globe(a / 2.0 - pi / 2.0, longitude));
            s.polyline(&meridian, 0.8, palette::grid().faded(0.8));
        }
        let parallel = |latitude: f64| round(|a| on_globe(latitude, a));
        s.polyline(&parallel(0.0), 1.0, palette::ink().faded(0.35));
        s.polyline(&parallel(paris), 1.6, palette::sky());
        // The floor, its swing and its north, carried by the earth.
        let centre = earth >> site.vertical;
        let (across, along) = ((earth >> site.east) * FLOOR, (earth >> site.north) * FLOOR);
        let [p, q, r, t] = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]
            .map(|(u, v)| reach3(centre + across * u + along * v));
        let lit = s.lit(p, q, r, palette::yellow().faded(0.2));
        s.quad(p, q, r, t, lit, 0.9);
        s.polyline(&[p, q, r, t, p], 1.0, palette::yellow().faded(0.6));
        // The swing just above the floor, so that the floor does not cover it.
        let line = (earth >> swing(paris, earth_turn)) * FLOOR;
        let above = centre * ABOVE.powi(3);
        s.seg(
            reach3(above - line),
            reach3(above + line),
            2.6,
            palette::orange(),
        );
        s.seg(
            reach3(centre + along),
            reach3(centre + along * 1.4),
            1.2,
            palette::ink(),
        );
    });
    let north = cam.px(reach3(Z * 1.12)).unwrap_or(rect.top_middle());
    c.text("N", north, 11.0, palette::ink().faded(0.7), Align::Center);
    // The holonomy: north carried once round the latitude circle, turned as the swing is in a
    // day.
    let holonomy = turned(carried(paris, STEPS) >> site.north, &site).to_degrees();
    let note = format!(
        "HOLONOMY AT {} N: {:.0} DEG",
        LATITUDES[PARIS],
        holonomy.rem_euclid(360.0)
    );
    let at = rect.bottom_middle() + Point2::direction(0.0, -14.0);
    c.text(&note, at, 10.0, palette::sky(), Align::Center);
}

/// The bob's track on one floor from the start of the day to `reached`, coloured by the time of
/// day, the line it swings along now, and a tick to the north.
fn draw_floor(c: &mut Canvas, rect: Rect, latitude: usize, reached: usize, earth_turn: f64) {
    let site = compass(latitudes()[latitude]);
    let ax = Axes::equal(rect, Point2::xy(0.0, 0.08), 1.3);
    let ring = points::circle(gax::pga2d::Point::xy(0.0, 0.0), 1.0, CIRCLE);
    ax.polyline(c, &ring, 1.0, palette::grid());
    let track: Vec<Point2> = tracks()[latitude][..=reached]
        .iter()
        .map(|v| reach2(on_floor(*v, &site)))
        .collect();
    for (k, chunk) in track.chunks(CHUNK).enumerate() {
        // Each chunk joined to the next one's first sample.
        let end = ((k + 1) * CHUNK).min(track.len() - 1);
        let stroke = [chunk, &track[end..=end]].concat();
        let day = (k * CHUNK) as f64 / (SAMPLES - 1) as f64;
        ax.polyline(c, &stroke, 1.0, track_colour(day));
    }
    let now = on_floor(swing(latitudes()[latitude], earth_turn), &site);
    ax.line(c, reach2(-now), reach2(now), 2.4, palette::orange());
    let north = |r: f32| ax.px(Point2::xy(0.0, r));
    c.line(north(1.05), north(1.15), 1.2, palette::ink());
    c.text(
        "N",
        north(1.25),
        10.0,
        palette::ink().faded(0.7),
        Align::Center,
    );
    // The turn so far, clockwise from north, from 0 to 360 degrees (`+ 0.0` makes a rounded
    // `-0` read `0`).
    let degrees = turned(swing(latitudes()[latitude], earth_turn), &site).to_degrees();
    let degrees = degrees.round().rem_euclid(360.0) + 0.0;
    let below = |r: f32| ax.px(Point2::xy(0.0, -r));
    c.text(
        TITLES[latitude],
        below(1.18),
        11.0,
        palette::ink(),
        Align::Center,
    );
    let note = format!("TURNED {degrees:.0} DEG CLOCKWISE");
    c.text(
        &note,
        below(1.38),
        9.0,
        palette::ink().faded(0.6),
        Align::Center,
    );
}

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let screen = c.rect();
    let (w, h) = (screen.width(), screen.height());
    let day = f64::from(t.rem_euclid(LOOP) / LOOP);
    let earth_turn = core::f64::consts::TAU * day;
    let reached = (day * (SAMPLES - 1) as f64).round() as usize;
    let top = h * 0.13;
    draw_globe(c, Rect::new(0.0, top, w * 0.42, h), earth_turn);
    let floors = Rect::new(w * 0.42, top, w, h);
    for latitude in 0..LATITUDES.len() {
        // Two rows of two.
        let (row, half) = ((latitude / 2) as f32, (h - top) / 2.0);
        let row = Rect::new(
            floors.lo.e20(),
            top + half * row,
            w,
            top + half * (row + 1.0),
        );
        draw_floor(
            c,
            row.column(latitude % 2, 2).inset(6.0, 6.0, 6.0, 22.0),
            latitude,
            reached,
            earth_turn,
        );
    }
    let hours = 24.0 * day;
    caption(
        c,
        "FOUCAULT'S PENDULUM: THE FLOOR TURNS UNDER THE SWING",
        &format!(
            "THE SWING TURNS BY THE EARTH'S TURN IN THE FLOOR, SIN(LATITUDE) A DAY (VGA3D). {hours:04.1} H"
        ),
    );
}

fn main() {
    run(Anim::new("foucault", LOOP).size(960, 540), draw);
}

#[cfg(test)]
mod tests {
    use super::foucault::*;
    use super::turned;
    use gax_numga_examples::signal::wave;

    /// After a day each swing has turned clockwise from north by the sine of the latitude times
    /// a full turn, and a direction carried once around the latitude circle comes back turned
    /// the same way; early in the day every swing off the equator has turned clockwise. These
    /// are also numga's scenario checks (its `main`).
    #[test]
    fn a_day_turns_each_swing_by_the_sine_of_its_latitude_clockwise_and_the_holonomy_agrees() {
        let tau = core::f64::consts::TAU;
        for (k, latitude) in latitudes().into_iter().enumerate() {
            let site = compass(latitude);
            let expected = tau * wave(latitude);
            let end = turned(swing(latitude, tau), &site);
            assert!(wave(end - expected).abs() < 1e-10, "{end} {expected}");
            let early = turned(swing(latitude, 1e-3), &site);
            assert!(k == 0 || early > 0.0, "{early}");
            let carried = turned(carried(latitude, STEPS) >> site.north, &site);
            assert!(
                wave(carried - expected).abs() < 1e-4,
                "{carried} {expected}"
            );
        }
    }

    #[test]
    fn the_rosettes_and_a_frame_of_the_day_draw() {
        gax_numga_examples::app::assert_draws(super::draw, 5.0);
    }
}
