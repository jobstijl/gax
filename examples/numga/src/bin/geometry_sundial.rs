//! numga's `geometry/sundial`: the hour lines and the seasons' shadow paths of a horizontal
//! sundial, in VGA3D.
//!
//! Space at the dial is east, north and up. The sky turns once a day about the celestial pole,
//! north raised towards up by the latitude. Every position of the sun is two rotor turns of the
//! celestial equator's highest point (the sun at noon on the equinoxes): towards the pole by
//! the declination, the season's angle from the equator, then about the pole towards the west
//! by the hour angle, a twenty-fourth of a turn per hour from noon.
//!
//! A dial reads the sun by the shadow of a gnomon whose edge, the style, points at the pole.
//! The plane through the pole and the sun at a given hour holds the sun at that hour on every
//! day of the year, since the declination turns the sun within it: each hour's shadow falls
//! along one line through the foot of the style, at the angle `atan(sin φ tan H)` of
//! gnomonics from the noon line. The tip of the gnomon, the nodus, casts a point whose path
//! over a day is the season's. The day's rays from the nodus to the sun keep their angle to the
//! pole, so they sweep a cone, the quadratic form `pole (pole | v) - sin² δ v` on rays, and the
//! dial cuts it in a conic. The form only sees the declination squared, so a summer day and the
//! winter day as far from the equinox are the two branches of one hyperbola; at the equinoxes
//! the cone flattens into the equator's plane, and the shadow runs straight.
//!
//! The animation runs through a year: on the left the day's rays past the nodus down to the
//! dial, sweeping their cone about the pole; on the right the dial's face, with the hour lines,
//! the seasons' paths, and the day's path.

use gax_numga_examples::{
    Align, Anim, Axes, Camera, Canvas, Lens, Light, Marker, ORIGIN3, Point2, Rect, backdrop,
    caption, colormap, palette, reach3, run, scene3::panel3,
};

mod sundial {
    use gax::vga3d::{Bivector, Rotor, Vector};
    use gax_numga_examples::signal::wave;

    pub type V = Vector<(), f64>;
    pub type B = Bivector<(), f64>;
    pub type R = Rotor<(), f64>;

    pub fn east() -> V {
        Vector::new(1.0, 0.0, 0.0)
    }

    pub fn north() -> V {
        Vector::new(0.0, 1.0, 0.0)
    }

    pub fn up() -> V {
        Vector::new(0.0, 0.0, 1.0)
    }

    /// The turn by an angle in a plane through the origin, from its first vector towards its
    /// second.
    pub fn turn(plane: B, angle: f64) -> R {
        plane.gp(-angle / 2.0).exp().into_inner()
    }

    /// The celestial pole, north raised towards up by the latitude, and the celestial
    /// equator's highest point, a quarter turn further: the sun at noon on the equinoxes.
    pub fn sky(latitude: f64) -> (V, V) {
        let meridian = north() ^ up();
        let quarter = core::f64::consts::FRAC_PI_2;
        (
            turn(meridian, latitude) >> north(),
            turn(meridian, latitude + quarter) >> north(),
        )
    }

    /// The direction to the sun at a declination and hour angle: the equator's highest point
    /// turned towards the pole by the declination, then about the pole towards the west by the
    /// hour angle.
    pub fn sun(pole: V, equator: V, declination: f64, hour: f64) -> V {
        let noon = turn(equator ^ pole, declination) >> equator;
        turn(equator ^ -east(), hour) >> noon
    }

    /// Where the shadow of the nodus falls on the dial, the horizontal plane through the
    /// origin: from the nodus, away from the sun, as far as the dial.
    pub fn shadow(nodus: V, sun: V) -> V {
        nodus - sun.gp((nodus | up()).s() / (sun | up()).s())
    }

    /// numga's dial: its latitude, the height of the nodus above the dial, and the tilt of the
    /// earth's axis, the sun's declination at the solstices.
    pub const LATITUDE: f64 = 52.0 * core::f64::consts::PI / 180.0;
    pub const HEIGHT: f64 = 1.0;
    pub const TILT: f64 = 23.44 * core::f64::consts::PI / 180.0;
    /// The seasons drawn lie a whole number of months from an equinox, up to a solstice.
    pub const MONTHS: i32 = 3;
    /// The hours drawn, from six in the morning to six at night.
    pub const HOURS: core::ops::RangeInclusive<i32> = 6..=18;

    /// The nodus, straight above the origin.
    pub fn nodus() -> V {
        up().gp(HEIGHT)
    }

    /// The sun's hour angle at an hour of local solar time: a fifteenth of a quarter turn per
    /// hour from noon, westwards in the afternoon.
    pub fn hour_angle(hour: f64) -> f64 {
        (15.0 * (hour - 12.0)).to_radians()
    }

    /// The seasons' declinations, a whole number of months from an equinox.
    pub fn seasons() -> Vec<f64> {
        let quarter = core::f64::consts::FRAC_PI_2;
        (-MONTHS..=MONTHS)
            .map(|k| TILT * wave(quarter * f64::from(k) / f64::from(MONTHS)))
            .collect()
    }

    /// The sun through a day at a declination, from three in the morning to nine at night.
    pub fn day(declination: f64) -> Vec<V> {
        let (pole, equator) = sky(LATITUDE);
        (0..=360)
            .map(|k| hour_angle(3.0 + 0.05 * f64::from(k)))
            .map(|h| sun(pole, equator, declination, h))
            .collect()
    }

    /// The sun at an hour, from the winter solstice to the summer one.
    pub fn hour_line(hour: i32) -> Vec<V> {
        let (pole, equator) = sky(LATITUDE);
        (0..25)
            .map(|k| -TILT + 2.0 * TILT * f64::from(k) / 24.0)
            .map(|d| sun(pole, equator, d, hour_angle(f64::from(hour))))
            .collect()
    }

    /// The declination a fraction of the way through the year, from the spring equinox.
    pub fn declination(fraction: f64) -> f64 {
        TILT * wave(core::f64::consts::TAU * fraction)
    }
}

use sundial::*;

const SECONDS: f32 = 12.0;
/// The part of the dial drawn, east and north of the point below the nodus.
const DIAL_EAST: [f64; 2] = [-4.5, 4.5];
const DIAL_NORTH: [f64; 2] = [-2.0, 4.5];
/// How far inside the edge of the dial the hour numerals sit.
const INSET: f64 = 0.3;
/// How far each ray is drawn beyond the nodus towards the sun.
const SUNWARD: f64 = 2.0;

fn day_colour() -> Light {
    palette::orange()
}

/// A colour per season by how far it lies from the equinox: a summer and a winter season as
/// far from it alike.
fn season_colour(declination: f64) -> Light {
    colormap::inferno(0.4 + 0.55 * (declination.abs() / TILT) as f32)
}

/// East and north of the nodus's shadow for a sun, or nothing where the sun is below the
/// horizon or the shadow falls off the dial.
fn shadow_on_dial(sun: V) -> Option<Point2> {
    let tip = shadow(nodus(), sun);
    let [x, y] = [tip.e1(), tip.e2()];
    let on_dial = x > DIAL_EAST[0] && x < DIAL_EAST[1] && y > DIAL_NORTH[0] && y < DIAL_NORTH[1];
    ((sun | up()).s() > 0.0 && on_dial).then(|| Point2::xy(x as f32, y as f32))
}

/// The shadows of a run of suns, broken where they leave the dial.
fn shadow_paths(suns: &[V]) -> Vec<Vec<Point2>> {
    let mut paths = vec![Vec::new()];
    for sun in suns {
        match shadow_on_dial(*sun) {
            Some(p) => paths.last_mut().expect("one path").push(p),
            None if !paths.last().expect("one path").is_empty() => paths.push(Vec::new()),
            None => {}
        }
    }
    paths.retain(|p| p.len() > 1);
    paths
}

/// The dial's face in `rect`: each season's shadow path, each hour line numbered where it
/// leaves the dial, the gnomon's edge along the noon line, and the day's path.
fn face_panel(c: &mut Canvas, rect: Rect, declination: f64) {
    // The whole dial, with room for the numerals at its sides.
    let half_width = 0.5 * (DIAL_EAST[1] - DIAL_EAST[0]) as f32 + 0.2;
    let half_height = (0.5 * (DIAL_NORTH[1] - DIAL_NORTH[0]) as f32 + 0.1)
        .max(half_width * rect.height() / rect.width());
    let ax = Axes::equal(rect, Point2::xy(0.0, 1.25), half_height);
    let corners = [[0, 0], [1, 0], [1, 1], [0, 1]]
        .map(|[i, j]| Point2::xy(DIAL_EAST[i] as f32, DIAL_NORTH[j] as f32));
    ax.fill(c, &corners, palette::grid(), 0.25);
    for hour in HOURS {
        for path in shadow_paths(&hour_line(hour)) {
            ax.polyline(c, &path, 1.0, palette::ink().faded(0.4));
            let (first, last) = (path[0], path[path.len() - 1]);
            let outward = (first - last).gp(1.0 / (first & last).norm());
            let label = first - outward.gp(INSET as f32);
            let text = format!("{hour}");
            let lifted = label + Point2::direction(0.0, -0.1);
            ax.text(
                c,
                lifted,
                &text,
                9.0,
                palette::ink().faded(0.7),
                Align::Center,
            );
        }
    }
    for season in seasons() {
        for path in shadow_paths(&day(season)) {
            ax.polyline(c, &path, 1.6, season_colour(season).faded(0.6));
        }
    }
    // The gnomon's edge seen from above, from the foot of the style to below the nodus.
    let (pole, _) = sky(LATITUDE);
    let foot = shadow(nodus(), pole);
    let base = [Point2::xy(0.0, foot.e2() as f32), Point2::xy(0.0, 0.0)];
    ax.polyline(c, &base, 3.0, palette::ink());
    for path in shadow_paths(&day(declination)) {
        ax.polyline(c, &path, 2.6, day_colour());
    }
}

/// The dial in space in `rect`, with its gnomon, and the day's rays of sunlight past the nodus
/// down to the dial, sweeping the cone about the pole, and their cut with the dial.
fn sky_panel(c: &mut Canvas, rect: Rect, declination: f64) {
    let cam = Camera::orbit(
        rect,
        gax::pga3d::Point::xyz(0.0, 1.0, 0.8),
        27.0,
        (-125f32).to_radians(),
        16f32.to_radians(),
        Lens::Perspective(0.38),
    );
    panel3(c, rect, cam, |s| {
        let corner = |i: usize, j: usize| gax::pga3d::Point::xyz(DIAL_EAST[i], DIAL_NORTH[j], 0.0);
        let dial = [corner(0, 0), corner(1, 0), corner(1, 1), corner(0, 1)];
        s.quad(dial[0], dial[1], dial[2], dial[3], palette::grid(), 0.45);
        s.polyline(
            &[dial[0], dial[1], dial[2], dial[3], dial[0]],
            1.0,
            palette::grid(),
        );
        // The gnomon: the triangle of the style's foot, the nodus and the point below it.
        let (pole, equator) = sky(LATITUDE);
        let (tip, foot) = (reach3(nodus()), reach3(shadow(nodus(), pole)));
        s.tri(foot, tip, ORIGIN3, palette::ink().faded(0.5), 0.8);
        s.polyline(&[foot, tip, ORIGIN3], 1.5, palette::ink());
        // A ray every quarter of an hour, from four in the morning to nine at night.
        let hours = (0..68).map(|k| hour_angle(4.0 + 0.25 * f64::from(k)));
        for sun in hours.map(|h| sun(pole, equator, declination, h)) {
            if shadow_on_dial(sun).is_some() {
                let sunward = reach3(nodus() + sun.gp(SUNWARD));
                let end = reach3(shadow(nodus(), sun));
                s.seg(sunward, end, 0.8, palette::yellow().faded(0.55));
            }
        }
        for path in shadow_paths(&day(declination)) {
            let on_dial: Vec<_> = path
                .iter()
                .map(|p| {
                    let [x, y] = p.to_euclidean().map(f64::from);
                    gax::pga3d::Point::xyz(x, y, 0.0)
                })
                .collect();
            s.polyline(&on_dial, 2.4, day_colour());
        }
        s.dot(tip, Marker::Dot, 6.0, palette::yellow());
    });
}

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let screen = c.rect();
    let fraction = t / SECONDS;
    let declination = declination(f64::from(fraction));
    sky_panel(c, screen.part(0.0, 0.08, 0.5, 1.0), declination);
    face_panel(c, screen.part(0.5, 0.14, 0.98, 0.96), declination);
    let label = format!(
        "DAY {} AFTER THE EQUINOX: DECLINATION {:+.1} DEG",
        (fraction * 365.0) as u32,
        declination.to_degrees()
    );
    // Under the dial's face, centred on it.
    let at = screen.part(0.5, 0.0, 0.98, 0.98).bottom_middle();
    c.text(&label, at, 10.0, day_colour(), Align::Center);
    caption(
        c,
        "SUNDIAL: HOUR LINES AND THE SEASONS' CONICS",
        "THE SUN IS TWO ROTOR TURNS; A DAY'S RAYS SWEEP A CONE ABOUT THE POLE (VGA3D, 52 DEG N)",
    );
}

fn main() {
    run(Anim::new("sundial", SECONDS).size(960, 540), draw);
}

#[cfg(test)]
mod tests {
    use super::sundial::*;
    use gax::vga2d;
    use gax::vga3d::Vector;
    use gax_numga_examples::measure::angle;
    use gax_numga_examples::signal::{phasor, wave};

    /// A quadratic form on rays.
    type Cone = Vector<(Vector,), f64>;

    /// The cone of rays at a declination's angle from the celestial equator: zero on a ray
    /// exactly when its angle to the pole is a quarter turn less the declination, or more.
    fn cone(pole: V, declination: f64) -> Cone {
        let v = Vector::slot();
        let sine = wave(declination);
        pole * (pole | v) - v.gp(sine * sine)
    }

    /// The plane through the pole and the sun at an hour angle, on every day of the year.
    fn hour_plane(pole: V, equator: V, hour: f64) -> B {
        pole ^ sun(pole, equator, 0.0, hour)
    }

    /// At noon the sun stands a quarter turn less the latitude, plus the declination, above
    /// the horizon: its elevation is a quarter turn less its angle from up.
    #[test]
    fn the_noon_sun_stands_a_quarter_turn_less_the_latitude_plus_the_declination_high() {
        let (pole, equator) = sky(LATITUDE);
        let quarter = core::f64::consts::FRAC_PI_2;
        for k in 0..5 {
            let declination = -TILT + 2.0 * TILT * f64::from(k) / 4.0;
            let noon = sun(pole, equator, declination, 0.0);
            let elevation = quarter - angle(noon, up());
            let expected = quarter - LATITUDE + declination;
            assert!(
                (elevation - expected).abs() < 1e-12,
                "{elevation} {expected}"
            );
        }
    }

    /// Every shadow of the nodus through each season's day lies on that season's cone of rays,
    /// and each hour's shadows at every season lie on one line through the foot of the style,
    /// at the angle of gnomonics: the sine of the angle between the line and the expected
    /// bearing, `(sin φ sin H, cos H)` east and north, is their unit directions' wedge.
    #[test]
    fn shadows_lie_on_their_cones_and_hour_lines_on_the_gnomonic_angle() {
        let (pole, _) = sky(LATITUDE);
        for season in seasons() {
            let shape = cone(pole, season);
            for sun in day(season) {
                let ray = (nodus() - shadow(nodus(), sun)).normalized().into_inner();
                assert!((ray | shape.of(ray)).s().abs() < 1e-10);
            }
        }
        let foot = shadow(nodus(), pole);
        for hour in HOURS {
            let h = hour_angle(f64::from(hour));
            let expected = vga2d::Vector::new(wave(LATITUDE) * wave(h), phasor(h).e20());
            for sun in hour_line(hour) {
                let offset = shadow(nodus(), sun) - foot;
                let bearing = vga2d::Vector::new(offset.e1(), offset.e2());
                let sine = bearing.normalized().into_inner() ^ expected.normalized().into_inner();
                assert!(sine.e12().abs() < 1e-10, "{hour}: {}", sine.e12());
            }
        }
    }

    /// The plane through the pole and the sun at an hour holds the sun at that hour whatever
    /// the season: why each hour casts its shadow along one line.
    #[test]
    fn an_hour_plane_holds_the_sun_at_that_hour_all_year() {
        let (pole, equator) = sky(LATITUDE);
        for hour in HOURS {
            let plane = hour_plane(pole, equator, hour_angle(f64::from(hour)));
            for sun in hour_line(hour) {
                assert!((plane ^ sun).e123().abs() < 1e-12);
            }
        }
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws(super::draw, 2.0);
    }
}
