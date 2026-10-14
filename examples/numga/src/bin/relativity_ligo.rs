//! numga's `relativity/ligo`: gravitational-wave antenna patterns, how strongly an L-shaped
//! interferometer such as LIGO responds to a wave from each direction on the sky, in the
//! geometric algebra of space (VGA3D).
//!
//! A passing gravitational wave stretches space along one direction across its path and
//! squeezes it along the other: a strain, a map on offsets with the offset left open,
//! `Vector<(Vector,)>`. For a wave from a given direction two strains span them all: plus,
//! `e (e · x) - n (n · x)` with `e` and `n` the axes across the wave, and cross,
//! `e (n · x) + n (e · x)`, the same turned by an eighth of a turn. An arm along the unit vector
//! `a` lengthens by the fraction `a · S(a) / 2`, and the interferometer reads the difference of
//! its two arms. The responses to a direction's plus and cross strains are the detector's antenna
//! patterns. Turning the axes across the wave by an angle turns the pair of responses by twice
//! that angle (the strain is a spin-two field), so the pair's length does not depend on it: the
//! detector's sensitivity to that direction, one straight overhead and zero for waves in the
//! plane of the arms along the lines halfway between them.
//!
//! Places on Earth and directions on the sky share one frame: the rotor that turns the first axis
//! to the point at a latitude and longitude turns the second to east and the third to north
//! there. At a detector's site it gives the level plane its arms lie in; for a direction on the
//! sky, the axes across a wave from there.
//!
//! The animation: a ring of free masses across a wave from straight above Hanford, stretched by
//! the plus strain as its polarization turns half around (left, the strain far beyond any real
//! one so that it shows), beside sky maps (longitude across, latitude up; numga draws them in
//! Mollweide's projection): Hanford's response to that strain, its sensitivity, the two
//! detectors' combined sensitivity, and how alike their readings of one wave are (nearly
//! opposite over most of the sky, as they were built).

use std::sync::OnceLock;

use gax_numga_examples::{
    Align, Anim, Axes, Canvas, Light, Marker, ORIGIN2, Point2, Rect, backdrop, caption, colormap,
    palette, run, signal::phasor,
};

mod ligo {
    use gax::Unit;
    use gax::vga3d::{Rotor, Vector};

    pub type V = Vector<(), f64>;
    /// A frame: the rotor turning the axes to a place's zenith (or a wave's direction), east and
    /// north.
    pub type Frame = Unit<Rotor<(), f64>>;
    /// A strain: the fractional stretch of each offset.
    pub type Strain = Vector<(Vector,), f64>;
    /// A detector's responses to the plus and cross strains of one wave, as a vector of the plane
    /// of polarizations: a turn of the axes across the wave turns it by twice the angle.
    pub type Responses = gax::vga2d::Vector<(), f64>;

    /// Fixed to the Earth: to longitude zero on the equator.
    pub fn x() -> V {
        Vector::new(1.0, 0.0, 0.0)
    }

    /// To ninety degrees east on the equator.
    pub fn y() -> V {
        Vector::new(0.0, 1.0, 0.0)
    }

    /// To the north pole.
    pub fn z() -> V {
        Vector::new(0.0, 0.0, 1.0)
    }

    /// The frame at a latitude and longitude, in radians: a turn from the equator towards the
    /// pole, then about the pole.
    pub fn place(latitude: f64, longitude: f64) -> Frame {
        ((x() ^ y()) * (-longitude / 2.0)).exp() * ((x() ^ z()) * (-latitude / 2.0)).exp()
    }

    /// The frame turned about its first axis, the wave's direction, by a polarization angle.
    pub fn polarized(frame: Frame, angle: f64) -> Frame {
        frame * ((y() ^ z()) * (-angle / 2.0)).exp()
    }

    /// The plus and cross strains of a wave from the direction of the frame's first axis, with
    /// its second and third axes across the wave.
    pub fn strains(frame: Frame) -> [Strain; 2] {
        let (east, north) = (frame >> y(), frame >> z());
        // The dyad `a (b · x)`: offsets along `b` to `a`.
        let dyad = |a: V, b: V| a * (b | Vector::slot());
        [
            dyad(east, east) - dyad(north, north),
            dyad(east, north) + dyad(north, east),
        ]
    }

    /// A detector's arms at its site, from their azimuths in radians clockwise from north: north
    /// turned towards east in the site's level plane.
    pub fn arms(site: Frame, azimuths: [f64; 2]) -> [V; 2] {
        let (east, north) = (site >> y(), site >> z());
        azimuths.map(|azimuth| ((north ^ east) * (-azimuth / 2.0)).exp() >> north)
    }

    /// The interferometer's reading: the difference of its arms' stretches, each lengthening by
    /// the fraction `a · S(a) / 2`.
    pub fn response(arms: [V; 2], strain: Strain) -> f64 {
        let stretch = |arm: V| (arm | strain.of(arm)).s() / 2.0;
        stretch(arms[0]) - stretch(arms[1])
    }

    /// The detector's responses to the plus and cross strains of a wave from the frame's first
    /// axis.
    pub fn responses(arms: [V; 2], frame: Frame) -> Responses {
        let [plus, cross] = strains(frame);
        Responses::new(response(arms, plus), response(arms, cross))
    }

    /// The sensitivity: the length of the pair of responses, the same for every turn of the axes
    /// across the wave.
    pub fn sensitivity(responses: Responses) -> f64 {
        responses.norm()
    }

    // --- the scenes -------------------------------------------------------------------

    /// A detector: its site in degrees and its arms' azimuths in degrees clockwise from north.
    pub struct Detector {
        pub name: &'static str,
        pub latitude: f64,
        pub longitude: f64,
        pub azimuths: [f64; 2],
    }

    impl Detector {
        /// The frame at its site.
        pub fn site(&self) -> Frame {
            place(self.latitude.to_radians(), self.longitude.to_radians())
        }

        /// Its arms.
        pub fn arms(&self) -> [V; 2] {
            arms(self.site(), self.azimuths.map(f64::to_radians))
        }
    }

    /// The two LIGO detectors, at Hanford, Washington, and Livingston, Louisiana.
    pub const DETECTORS: [Detector; 2] = [
        Detector {
            name: "HANFORD",
            latitude: 46.4553,
            longitude: -119.4078,
            azimuths: [324.0, 234.0],
        },
        Detector {
            name: "LIVINGSTON",
            latitude: 30.5631,
            longitude: -90.7744,
            azimuths: [252.0, 162.0],
        },
    ];

    /// Latitudes sampled from pole to pole; twice as many longitudes once around.
    pub const SKY_SAMPLES: usize = 61;
    pub const SKY_COLUMNS: usize = 2 * SKY_SAMPLES - 1;

    /// The latitude of a row of the sky, in degrees.
    pub fn latitude(row: usize) -> f64 {
        -90.0 + 180.0 * row as f64 / (SKY_SAMPLES - 1) as f64
    }

    /// The longitude of a column of the sky, in degrees.
    pub fn longitude(column: usize) -> f64 {
        -180.0 + 360.0 * column as f64 / (SKY_COLUMNS - 1) as f64
    }

    /// The frame of each direction on the sky, row by row from the south pole: its first axis
    /// to the direction, the others across.
    pub fn sky() -> Vec<Frame> {
        (0..SKY_SAMPLES * SKY_COLUMNS)
            .map(|k| {
                let (row, column) = (k / SKY_COLUMNS, k % SKY_COLUMNS);
                place(latitude(row).to_radians(), longitude(column).to_radians())
            })
            .collect()
    }

    /// The free masses on the ring.
    pub const MASSES: usize = 24;
    /// The strain drawn on the ring, far beyond any real one so that it shows.
    pub const STRAIN_SHOWN: f64 = 0.3;

    /// A ring of free masses across a wave from straight above the site, stretched by the plus
    /// strain at the polarization angle: straight overhead, the axes across the wave are the
    /// site's east and north.
    pub fn ring(site: Frame, angle: f64) -> Vec<V> {
        let (east, north) = (site >> y(), site >> z());
        let [overhead, _] = strains(polarized(site, angle));
        (0..MASSES)
            .map(|k| {
                let turn = core::f64::consts::TAU * k as f64 / MASSES as f64;
                let rest = ((east ^ north) * (-turn / 2.0)).exp() >> east;
                rest + overhead.of(rest) * STRAIN_SHOWN
            })
            .collect()
    }
}

use ligo::*;

/// One loop: the polarization turns half around.
const SECONDS: f32 = 6.0;
/// How far the arms reach on the ring.
const ARM_REACH: f64 = 1.25;

/// What the frames share: the sky's frames, and the maps that do not change: Hanford's
/// sensitivity, the network's, and the cosine between the detectors' response pairs.
struct Sky {
    frames: Vec<Frame>,
    hanford: Vec<f64>,
    network: Vec<f64>,
    alike: Vec<f64>,
}

fn sky_maps() -> &'static Sky {
    static SKY: OnceLock<Sky> = OnceLock::new();
    SKY.get_or_init(|| {
        let frames = sky();
        let arms = DETECTORS.map(|d| d.arms());
        let pairs: Vec<[Responses; 2]> = frames
            .iter()
            .map(|f| arms.map(|a| responses(a, *f)))
            .collect();
        Sky {
            hanford: pairs.iter().map(|[h, _]| sensitivity(*h)).collect(),
            network: pairs
                .iter()
                .map(|[h, l]| Responses::new(sensitivity(*h), sensitivity(*l)).norm())
                .collect(),
            alike: pairs
                .iter()
                .map(|[h, l]| (h.normalized().into_inner() | l.normalized().into_inner()).s())
                .collect(),
            frames,
        }
    })
}

/// A sky map's value at a longitude and latitude in degrees, interpolated between its samples.
fn sample(values: &[f64], longitude_deg: f32, latitude_deg: f32) -> f64 {
    let column = (f64::from(longitude_deg) + 180.0) / 360.0 * (SKY_COLUMNS - 1) as f64;
    let row = (f64::from(latitude_deg) + 90.0) / 180.0 * (SKY_SAMPLES - 1) as f64;
    let (c0, r0) = (
        (column.floor() as usize).min(SKY_COLUMNS - 2),
        (row.floor() as usize).min(SKY_SAMPLES - 2),
    );
    let (u, v) = (column - c0 as f64, row - r0 as f64);
    let at = |r: usize, c: usize| values[r * SKY_COLUMNS + c];
    let bottom = at(r0, c0) * (1.0 - u) + at(r0, c0 + 1) * u;
    let top = at(r0 + 1, c0) * (1.0 - u) + at(r0 + 1, c0 + 1) * u;
    bottom * (1.0 - v) + top * v
}

/// A map over the sky in `rect`: each value through `colour`, a faint graticule, and the sites.
fn sky_panel(
    c: &mut Canvas,
    rect: Rect,
    title: &str,
    values: &[f64],
    colour: impl Fn(f64) -> Light + Sync,
) {
    let ax = Axes::new(rect, [-180.0, 180.0], [-90.0, 90.0]);
    ax.image(c, 1, |p| {
        let [longitude, latitude] = p.to_euclidean();
        Some(colour(sample(values, longitude, latitude)))
    });
    let faint = palette::ink().faded(0.12);
    let meridians = (-2..=2).map(|k| {
        let lon = 60.0 * k as f32;
        [Point2::xy(lon, -90.0), Point2::xy(lon, 90.0)]
    });
    let parallels = (-1..=1).map(|k| {
        let lat = 45.0 * k as f32;
        [Point2::xy(-180.0, lat), Point2::xy(180.0, lat)]
    });
    let graticule: Vec<[Point2; 2]> = meridians.chain(parallels).collect();
    ax.stroke(c, &graticule, 0.8, faint);
    for d in &DETECTORS {
        let at = Point2::xy(d.longitude as f32, d.latitude as f32);
        ax.scatter(c, &[at], Marker::Star, 9.0, palette::sky());
    }
    let r = ax.rect;
    c.polyline(
        &[r.lo, r.top_right(), r.hi, r.bottom_left()],
        1.0,
        palette::grid(),
        true,
    );
    let size = (rect.height() / 12.0).clamp(5.0, 11.0);
    let above = r.lo + Point2::direction(0.0, -size * 0.6);
    c.text(title, above, size, palette::ink(), Align::Left);
}

/// The signed colour map: blue for minus one, red for plus one.
fn signed(value: f64) -> Light {
    colormap::rdbu((0.5 + 0.5 * value) as f32)
}

/// The ring of free masses across a wave from overhead, at rest and stretched, with the
/// detector's arms, seen from above: east to the right, north up.
fn draw_ring(c: &mut Canvas, rect: Rect, angle: f64) {
    let detector = &DETECTORS[0];
    let site = detector.site();
    // East and north of a vector at the site: its coordinates in the site's frame.
    let level = |v: V| {
        let local = site << v;
        Point2::xy(local.e2() as f32, local.e3() as f32)
    };
    // A square view, so that the stretched ring and the arms fit across as well as up.
    let side = rect.width().min(rect.height());
    let square = Rect::around(rect.centre(), Point2::direction(side / 2.0, side / 2.0));
    let ax = Axes::equal(square, ORIGIN2, 1.5);
    let rest: Vec<Point2> = (0..=96)
        .map(|k| ORIGIN2 + phasor(core::f32::consts::TAU * k as f32 / 96.0))
        .collect();
    ax.polyline(c, &rest, 1.0, palette::grid().faded(1.5));
    for arm in detector.arms() {
        ax.line(c, ORIGIN2, level(arm * ARM_REACH), 4.0, palette::orange());
    }
    let mut masses: Vec<Point2> = ring(site, angle).into_iter().map(level).collect();
    ax.scatter(c, &masses, Marker::Dot, 6.0, palette::sky());
    masses.push(masses[0]);
    ax.polyline(c, &masses, 1.0, palette::sky().faded(0.5));
    let size = (rect.height() / 26.0).clamp(6.0, 13.0);
    let below = rect.bottom_middle() + Point2::direction(0.0, -size * 0.5);
    let degrees = angle.to_degrees();
    let readout = format!("POLARIZATION {degrees:5.1} DEG");
    c.text(&readout, below, size, palette::ink(), Align::Center);
    let above = rect.top_middle() + Point2::direction(0.0, size * 1.2);
    let title = format!("FREE MASSES ABOVE {}", detector.name);
    c.text(&title, above, size, palette::ink(), Align::Center);
}

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let s = sky_maps();
    let screen = c.rect();
    let (w, h) = (screen.width(), screen.height());
    let angle = f64::from(t / SECONDS) * core::f64::consts::PI;
    let top = h * 0.16;
    draw_ring(c, Rect::new(w * 0.02, top, w * 0.34, h - h * 0.04), angle);

    // Hanford's response to the plus strain at the polarization angle, as it turns.
    let arms = DETECTORS[0].arms();
    let plus_responses: Vec<f64> = s
        .frames
        .iter()
        .map(|f| response(arms, strains(polarized(*f, angle))[0]))
        .collect();
    let maps = Rect::new(w * 0.37, top, w * 0.99, h - h * 0.03);
    let cell = |i: usize, j: usize| {
        maps.part(
            0.5 * i as f32,
            0.5 * j as f32,
            0.5 * (i + 1) as f32,
            0.5 * (j + 1) as f32,
        )
        .inset(6.0, 22.0, 6.0, 8.0)
    };
    let sensitive = |v: f64| colormap::inferno(v as f32);
    sky_panel(
        c,
        cell(0, 0),
        "HANFORD, PLUS STRAIN",
        &plus_responses,
        signed,
    );
    sky_panel(c, cell(1, 0), "HANFORD SENSITIVITY", &s.hanford, sensitive);
    // The network's sensitivity is at most the length of two unit sensitivities, √2.
    let most = Responses::new(1.0, 1.0).norm();
    let network = |v: f64| colormap::inferno((v / most) as f32);
    sky_panel(c, cell(0, 1), "NETWORK SENSITIVITY", &s.network, network);
    sky_panel(
        c,
        cell(1, 1),
        "ALIKE: -1 OPPOSITE, +1 SAME",
        &s.alike,
        signed,
    );
    caption(
        c,
        "LIGO: GRAVITATIONAL-WAVE ANTENNA PATTERNS",
        "VGA3D: A STRAIN IS A MAP ON OFFSETS; THE RESPONSE PAIR TURNS AT TWICE THE POLARIZATION",
    );
}

fn main() {
    run(Anim::new("ligo", SECONDS).size(960, 540), draw);
}

#[cfg(test)]
mod tests {
    use super::ligo::*;
    use super::sky_maps;
    use gax_numga_examples::signal::phasor;

    /// For arms along two horizontal axes at the pole, a wave from polar angle `θ` and azimuth
    /// `φ` from the first arm is seen with the squared sensitivity
    /// `((1 + cos²θ) / 2)² cos²2φ + cos²θ sin²2φ`.
    #[test]
    #[allow(clippy::disallowed_methods)] // the closed form it is checked against
    fn sensitivity_is_the_closed_form_for_a_detector_at_the_pole() {
        let pole = place(core::f64::consts::FRAC_PI_2, 0.0);
        let arms = arms(pole, [0.0, 90f64.to_radians()]);
        for i in 0..9 {
            let latitude = (-80.0 + 20.0 * i as f64).to_radians();
            for j in 0..13 {
                let longitude = (-170.0 + 340.0 * j as f64 / 12.0).to_radians();
                let seen = sensitivity(responses(arms, place(latitude, longitude)));
                let (polar, azimuth) = (core::f64::consts::FRAC_PI_2 - latitude, longitude);
                let (c, c2, s2) = (polar.cos(), (2.0 * azimuth).cos(), (2.0 * azimuth).sin());
                let expected = ((1.0 + c * c) / 2.0).powi(2) * c2 * c2 + c * c * s2 * s2;
                assert!((seen * seen - expected).abs() < 1e-12, "{seen} {expected}");
            }
        }
    }

    /// The smallest turn that carries the unit vector `from` to the unit vector `to`.
    fn turning(from: V, to: V) -> Frame {
        (gax::vga3d::Rotor::from(gax::vga3d::Scalar::new(1.0)) + to * from).normalized()
    }

    /// numga's scenario checks: straight overhead each detector the sensitivity is one, and
    /// nowhere is it more; a wave along the line halfway between the arms, in their plane, is
    /// not seen at all; turning the axes across the wave by an angle mixes plus and cross by
    /// twice that angle; over most of the sky the two detectors read a wave with opposite
    /// signs.
    #[test]
    fn scenes_pass_their_checks() {
        let sky = sky_maps();
        for d in &DETECTORS {
            let arms = d.arms();
            assert!((sensitivity(responses(arms, d.site())) - 1.0).abs() < 1e-12);
            let halfway = (arms[0] + arms[1]).normalized().into_inner();
            let blind = turning(x(), halfway);
            assert!(sensitivity(responses(arms, blind)) < 1e-12);
        }
        assert!(sky.hanford.iter().all(|s| *s <= 1.0 + 1e-12));
        let arms = DETECTORS[0].arms();
        for k in 0..12 {
            let angle = core::f64::consts::PI * k as f64 / 12.0;
            let mix = phasor(2.0 * angle);
            for frame in sky.frames.iter().step_by(37) {
                let [plus, cross] = responses(arms, *frame).c;
                let turned = response(arms, strains(polarized(*frame, angle))[0]);
                assert!((turned - (plus * mix.e20() + cross * mix.e01())).abs() < 1e-12);
            }
        }
        let opposite = sky.alike.iter().filter(|a| **a < -0.5).count();
        assert!(opposite as f64 > 0.9 * sky.alike.len() as f64);
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws(super::draw, 0.5);
    }
}
