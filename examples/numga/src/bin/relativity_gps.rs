//! numga's `relativity/gps`: satellite navigation, a reception event and its uncertainty
//! recovered from transmission events, in projective spacetime algebra (STAP, `x+y+z+t-w0`).
//!
//! Events are intrinsic antivectors and readouts extrinsic vectors. A step between events is a
//! `Direction` (an ideal event), and the readouts that see steps are the hyperplanes through the
//! origin event, `OriginHyperplane` (`e1..e4`), its complement: what a clock or a ruler reads.
//! Each signal's light cone is a correlation from events to hyperplanes, the line from the
//! transmission to an event contracted with the transmission, `emitted | (emitted & Event)`: the
//! event pairs with its plane to the line's interval, zero exactly on the light cone. Two cones
//! of one Lorentz form differ by a pair of planes, so four cones meet in the line of three
//! radical planes and a quadratic on it: the reception event, exactly. With more satellites and
//! noisy timestamps, Gauss–Newton fits the reception to all the cones: each signal's readout of
//! a small step of the reception, twice its cone's tangent plane there, gives a positive dyad
//! `readout (readout & Step)`, weighed by how far its timestamp's error moves the interval; their
//! sum is the information map `OriginHyperplane <- Direction`, and its inverse the covariance
//! `Direction <- OriginHyperplane` of the position and the receiver's clock together.
//!
//! Lengths are in Earth radii and times in the light-travel time of an Earth radius, so light's
//! speed is one. The timestamps' error is set far above a real receiver's (1500 km of range) so
//! that the uncertainty shows at the scale of the orbits.
//!
//! The animation shows six satellites on polar orbits as they move from a narrow patch of the
//! receiver's sky to a wider spread (and back), their signals to the receiver at the pole, and
//! the fitted position's one-standard-deviation ellipsoid with its principal diameters; beside
//! it, the ellipsoid's semi-axes against the satellites' phase.

use gax_numga_examples::scene3::{Scene3, panel3};
use gax_numga_examples::{
    Anim, Axes, Camera, Canvas, Lens, Light, Marker, Point2, Point3, Rect, backdrop, caption,
    on_sphere, palette, run_with,
};

mod gps {
    use gax::stap::{Bivector, Direction, Event, OriginHyperplane, Scalar, Vector};

    /// A light cone: from events to the hyperplanes they pair with to the interval.
    pub type Cone = Vector<(Event,)>;
    /// The information of the timestamps: from steps of the reception to readouts.
    pub type Information = OriginHyperplane<(Direction,)>;
    /// Its inverse, the covariance: from readouts to steps.
    pub type Covariance = Direction<(OriginHyperplane,)>;

    /// The hyperplane `a x + b y + c z + d t + δ = 0`.
    fn hyperplane(x: f64, y: f64, z: f64, t: f64, w: f64) -> Vector {
        Vector::new(w, x, y, z, t)
    }
    /// The planes across `x`, `y` and `z`, the clock `t` (its reading of an event of unit
    /// weight is the event's time), and the plane at infinity `w`.
    pub fn x() -> Vector {
        hyperplane(1.0, 0.0, 0.0, 0.0, 0.0)
    }
    pub fn y() -> Vector {
        hyperplane(0.0, 1.0, 0.0, 0.0, 0.0)
    }
    pub fn z() -> Vector {
        hyperplane(0.0, 0.0, 1.0, 0.0, 0.0)
    }
    pub fn clock() -> Vector {
        hyperplane(0.0, 0.0, 0.0, 1.0, 0.0)
    }
    pub fn w() -> Vector {
        hyperplane(0.0, 0.0, 0.0, 0.0, 1.0)
    }

    /// The origin event, the complement of the plane at infinity; the steps along `x`, `y` and
    /// `z`, the complements of the planes across them; and the step along time, the complement
    /// of the clock (numga writes `-t.dual()`, its metric dual, `t` squaring to `−1`).
    pub fn origin() -> Event {
        w().dual()
    }
    pub fn across() -> Event {
        x().dual()
    }
    pub fn sideways() -> Event {
        y().dual()
    }
    pub fn upwards() -> Event {
        z().dual()
    }
    pub fn time() -> Event {
        clock().dual()
    }

    /// The satellite on a circular orbit in a plane through the centre at coordinate time `t`,
    /// in the local inertial frame: `across` turned in the plane by its phase, at the radius.
    pub fn orbit_position(plane: Bivector, phase: f64, radius: f64, speed: f64, t: f64) -> Event {
        let rotation = (plane * (-(phase + speed * t) / 2.0)).exp();
        origin() + (rotation >> across()) * radius + time() * t
    }

    /// The distance between events of unit weight in the frame whose time the clock reads:
    /// minus the squared norm of the line joining them, their interval, with the square of the
    /// time between them added back.
    pub fn spatial_distance(first: Event, second: Event) -> f64 {
        let between = (clock() & (first - second)).s();
        #[allow(clippy::disallowed_methods)] // a length from the interval: the spacetime split
        let distance = (between * between - (first & second).norm_squared()).sqrt();
        distance
    }

    /// The retarded transmission events on the orbits whose signals reach the receiver
    /// together: the transmission time found by fixed-point steps on the light-travel time.
    pub fn emitted_events(
        receiver: Event,
        planes: &[Bivector],
        phases: &[f64],
        radius: f64,
        speed: f64,
        steps: usize,
    ) -> Vec<Event> {
        let reception_time = (clock() & receiver).s();
        planes
            .iter()
            .zip(phases)
            .map(|(plane, phase)| {
                let at = |t| orbit_position(*plane, *phase, radius, speed, t);
                let mut emission_time = reception_time;
                for _ in 0..steps {
                    emission_time = reception_time - spatial_distance(receiver, at(emission_time));
                }
                at(emission_time)
            })
            .collect()
    }

    /// The light cone whose vertex is a transmission: the line from the transmission to an
    /// event, contracted with the transmission, a plane the event pairs with to the line's
    /// interval, zero exactly when the line is a light ray.
    pub fn light_cone(emitted: Event) -> Cone {
        emitted.dot(emitted & Event::slot())
    }

    #[cfg(test)]
    /// An extrinsic event (a meet) at unit weight: divided by the plane at infinity's reading,
    /// which also gives it the intrinsic orientation of an event.
    fn at_unit_weight(e: gax::stap::extrinsic::Quadvector) -> Event {
        (e * (1.0 / (w() & e).s())).with_orientation::<gax::Intrinsic>()
    }

    #[cfg(test)]
    /// The two intersections of four light cones, in a finite coordinate-time chart.
    pub fn intersect_cones(cones: &[Cone; 4]) -> [Event; 2] {
        // Two cones of one Lorentz form differ by a pair of planes, the plane at infinity and
        // their radical plane: the difference sends an event to the radical plane times its
        // weight plus the plane at infinity times the radical plane's reading of it. Read at the
        // origin, the second term is half the difference's own reading of the origin.
        let radicals: [Vector; 3] = core::array::from_fn(|k| {
            let at_origin = (cones[k + 1] - cones[0]).of(origin());
            at_origin - w() * ((origin() & at_origin).s() / 2.0)
        });
        let axis = radicals[0] ^ radicals[1] ^ radicals[2];
        // Two time slices locate points on their common line.
        let base = at_unit_weight(axis ^ clock());
        let tip = at_unit_weight(axis ^ (clock() - w()));
        let direction = tip - base;
        // The remaining cone cuts the line in two points.
        let quadratic = (direction & cones[0].of(direction)).s();
        let linear = (base & cones[0].of(direction)).s();
        let constant = (base & cones[0].of(base)).s();
        #[allow(clippy::disallowed_methods)] // the quadratic's roots
        let root = (linear * linear - quadratic * constant).sqrt();
        [-linear - root, -linear + root].map(|p| base + direction * (p / quadratic))
    }

    #[cfg(test)]
    /// Of the cones' two intersections, the one every signal reaches after it left.
    pub fn reception(candidates: [Event; 2], emitted: &[Event]) -> Event {
        let after = |c: Event| emitted.iter().all(|e| (clock() & (c - *e)).s() > 0.0);
        if after(candidates[0]) {
            candidates[0]
        } else {
            candidates[1]
        }
    }

    /// Each signal's interval from its transmission to a reception event,
    /// `receiver & cone(receiver)`, which vanishes on the light cone; its readout of a small step
    /// of the reception, twice the cone's tangent plane there (the tangent plane's part across
    /// the plane at infinity reads no step); and how far a timestamp's error moves the interval,
    /// twice the light-travel time.
    pub fn intervals(
        cones: &[Cone],
        emitted: &[Event],
        receiver: Event,
    ) -> Vec<(f64, OriginHyperplane, f64)> {
        cones
            .iter()
            .zip(emitted)
            .map(|(cone, e)| {
                let tangent = cone.of(receiver);
                let readout = tangent.cast::<OriginHyperplane>() * 2.0;
                let lever = 2.0 * (clock() & (receiver - *e)).s();
                ((receiver & tangent).s(), readout, lever)
            })
            .collect()
    }

    /// The information of the readouts: each contributes a positive dyad for the steps it can
    /// see, weighed by how far its timestamp's uncertainty moves its interval.
    fn information(readings: &[(f64, OriginHyperplane, f64)], sigma: f64) -> Information {
        readings
            .iter()
            .map(|(_, r, lever)| {
                let dyad = (*r * (*r & Direction::slot())).cast::<OriginHyperplane>();
                dyad * (1.0 / (lever * sigma).powi(2))
            })
            .sum()
    }

    /// The reception event fitted to noisy timestamps, each to its light cone, by Gauss–Newton
    /// steps; and at the fit, the information map and its inverse, the covariance.
    pub fn locate(
        emitted: &[Event],
        initial: Event,
        sigma: f64,
        steps: usize,
    ) -> (Event, Information, Covariance) {
        let cones: Vec<Cone> = emitted.iter().map(|e| light_cone(*e)).collect();
        let mut receiver = initial;
        for _ in 0..steps {
            let readings = intervals(&cones, emitted, receiver);
            let mismatch: OriginHyperplane = readings
                .iter()
                .map(|(interval, r, lever)| *r * (interval / (lever * sigma).powi(2)))
                .sum();
            let step = information(&readings, sigma).solve(mismatch);
            receiver -= Event::from(step);
        }
        let information = information(&intervals(&cones, emitted, receiver), sigma);
        (receiver, information, information.inverse())
    }

    /// The semi-axes of the position's one-standard-deviation ellipsoid: over the principal
    /// planes of the covariance between position readouts (the readouts of space, PGA3D's planes
    /// through the origin in spacetime), each plane's covariance over its standard deviation.
    pub fn position_spread(covariance: Covariance) -> [Direction; 3] {
        let spatial = Vector::from(gax::pga3d::OriginPlane::slot()).cast::<OriginHyperplane>();
        let form = spatial & covariance.of(spatial);
        let (variances, principal) = form.eigh(Scalar::identity());
        core::array::from_fn(|k| {
            #[allow(clippy::disallowed_methods)] // a standard deviation from its variance
            let deviation = variances[k].sqrt();
            covariance.of(spatial.of(principal[k])) * (1.0 / deviation)
        })
    }

    // --- scenes

    /// Earth's radius, the orbits' altitude, light's speed and Earth's gravitational
    /// parameter, in metres and seconds.
    pub const EARTH_RADIUS_METRES: f64 = 6_371_000.0;
    pub const ORBIT_ALTITUDE_METRES: f64 = 20_200_000.0;
    pub const LIGHT_SPEED: f64 = 299_792_458.0;
    pub const GRAVITATIONAL_PARAMETER: f64 = 3.986004418e14;
    /// The satellites, the error of a range in metres and of the receiver's clock in seconds,
    /// and each satellite's phase along its orbit, in degrees, offset from the scene's.
    pub const N_SATELLITES: usize = 6;
    pub const RANGE_ERROR_METRES: f64 = 1_500_000.0;
    pub const CLOCK_ERROR_SECONDS: f64 = 30e-6;
    pub const PHASE_OFFSETS: [f64; N_SATELLITES] = [-12.0, -5.0, 8.0, 15.0, -9.0, 4.0];
    /// The steps finding the retarded transmissions and the steps of the fit, and the start's
    /// error off the receiver, along y and z, in Earth radii.
    pub const N_EMISSION_STEPS: usize = 3;
    pub const N_POSITION_STEPS: usize = 5;
    pub const INITIAL_Y: f64 = 0.02;
    pub const INITIAL_Z: f64 = 0.01;
    /// The scene's phase, and the phases the satellites move between, in degrees; the frames.
    #[cfg(test)]
    pub const PHASE: f64 = 20.0;
    pub const FIRST_PHASE: f64 = 5.0;
    pub const LAST_PHASE: f64 = 50.0;
    pub const N_FRAMES: usize = 98;
    /// numga's draws of its timestamps' noise, `default_rng(7).normal(size=6)`, in units of the
    /// range error: numpy's stream cannot be reproduced, and with these the scene is numga's.
    pub const NOISE: [f64; N_SATELLITES] = [
        0.001_230_153_357_482_57,
        0.298_745_537_508_469_9,
        -0.274_137_855_362_217_6,
        -0.890_591_838_757_274_2,
        -0.454_670_785_171_722_55,
        -0.991_646_554_996_462_4,
    ];

    /// Six circular orbits sharing the polar axis `x`, their distinct phases giving different
    /// elevations, which separates radial position error from clock error.
    pub fn planes() -> Vec<Bivector> {
        (0..N_SATELLITES)
            .map(|k| {
                let turn = (y() ^ z()) * (-(k as f64) * core::f64::consts::TAU / 6.0 / 2.0);
                turn.exp() >> (x() ^ y())
            })
            .collect()
    }
    /// The orbits' radius, in Earth radii.
    pub fn orbit_radius() -> f64 {
        (EARTH_RADIUS_METRES + ORBIT_ALTITUDE_METRES) / EARTH_RADIUS_METRES
    }
    /// The orbits' angular speed, per light-travel time of an Earth radius: Kepler's.
    pub fn angular_speed() -> f64 {
        let r = EARTH_RADIUS_METRES + ORBIT_ALTITUDE_METRES;
        #[allow(clippy::disallowed_methods)] // Kepler's third law
        let speed = (GRAVITATIONAL_PARAMETER / (r * r * r)).sqrt();
        speed * EARTH_RADIUS_METRES / LIGHT_SPEED
    }
    /// The receiver on Earth's surface at the pole, at time zero.
    pub fn receiver() -> Event {
        origin() + across()
    }
    /// A timestamp's error, and the receiver clock's reading off, in Earth radii.
    pub fn sigma() -> f64 {
        RANGE_ERROR_METRES / EARTH_RADIUS_METRES
    }
    pub fn clock_reading() -> f64 {
        CLOCK_ERROR_SECONDS * LIGHT_SPEED / EARTH_RADIUS_METRES
    }
    /// The satellites' phases, in radians, at the scene's phase in degrees.
    pub fn phases(phase: f64) -> Vec<f64> {
        PHASE_OFFSETS
            .iter()
            .map(|o| (o + phase).to_radians())
            .collect()
    }
    /// The transmissions reaching the receiver at a phase, without noise.
    pub fn transmissions(receiver: Event, phase: f64) -> Vec<Event> {
        let (radius, speed) = (orbit_radius(), angular_speed());
        emitted_events(
            receiver,
            &planes(),
            &phases(phase),
            radius,
            speed,
            N_EMISSION_STEPS,
        )
    }
    /// The start of the fit: the receiver's clock and an approximate location.
    pub fn initial(receiver: Event, clock_error: f64) -> Event {
        receiver + sideways() * INITIAL_Y + upwards() * INITIAL_Z + time() * clock_error
    }

    /// The transmissions at a phase (in degrees), and the reception event recovered from their
    /// six noisy timestamps, with its covariance.
    pub fn positioning(phase: f64) -> (Vec<Event>, Event, Covariance) {
        let emitted = transmissions(receiver(), phase);
        let observed: Vec<Event> = emitted
            .iter()
            .zip(NOISE)
            .map(|(e, n)| *e + time() * (n * sigma()))
            .collect();
        let start = initial(receiver(), clock_reading());
        let (estimate, _, covariance) = locate(&observed, start, sigma(), N_POSITION_STEPS);
        (emitted, estimate, covariance)
    }
}

use gax::stap::{Direction, Event};
use gps::*;

/// The loop: numga's 98 frames of 40 ms, forward then back.
const SECONDS: f32 = 2.0 * 98.0 * 0.04;
/// The view: turned 32 degrees about the polar axis and raised 22 above the equator, at the
/// orbits' scale.
const AZIMUTH: f32 = 32f32.to_radians();
const ELEVATION: f32 = 22f32.to_radians();

/// One frame: the transmissions, the estimate and its spread.
struct Frame {
    phase: f64,
    emitted: Vec<Event>,
    estimate: Event,
    spread: [Direction; 3],
}

struct Scenes {
    frames: Vec<Frame>,
}

fn scenes() -> Scenes {
    let frames = (0..N_FRAMES)
        .map(|k| {
            let phase = FIRST_PHASE + (LAST_PHASE - FIRST_PHASE) * k as f64 / (N_FRAMES - 1) as f64;
            let (emitted, estimate, covariance) = positioning(phase);
            Frame {
                phase,
                emitted,
                estimate,
                spread: position_spread(covariance),
            }
        })
        .collect();
    Scenes { frames }
}

/// Where an event is in space, seen by the frame whose clock is `t`: its place at unit weight,
/// drawn with the polar axis `x` up (the drawing's `z`), `y` and `z` across.
fn place(e: Event) -> Point3 {
    let weight = (w() & e).s();
    let along = |plane: gax::stap::Vector| ((plane & e).s() / weight) as f32;
    Point3::xyz(along(y()), along(z()), along(x()))
}

/// A step, drawn the same way.
fn step(d: Direction) -> Point3 {
    let e = Event::from(d);
    let along = |plane: gax::stap::Vector| (plane & e).s() as f32;
    Point3::direction(along(y()), along(z()), along(x()))
}

/// A box about `centre`, its half sizes along the drawing's axes, lit.
fn cuboid(s: &mut Scene3, centre: Point3, half: [f32; 3], light: Light) {
    let corner = |k: usize| {
        let sign = |bit: usize| if k >> bit & 1 == 1 { 1.0 } else { -1.0 };
        centre + Point3::direction(sign(0) * half[0], sign(1) * half[1], sign(2) * half[2])
    };
    for [a, b, c, d] in [
        [0, 1, 3, 2],
        [4, 5, 7, 6],
        [0, 1, 5, 4],
        [2, 3, 7, 6],
        [0, 2, 6, 4],
        [1, 3, 7, 5],
    ] {
        let lit = s.lit(corner(a), corner(b), corner(c), light);
        s.quad(corner(a), corner(b), corner(c), corner(d), lit, 1.0);
    }
}

/// Earth, the transmitting satellites, their signals to the receiver, the fitted reception and
/// its ellipsoid with its principal diameters.
fn draw_scene(c: &mut Canvas, rect: Rect, f: &Frame) {
    let radius = orbit_radius() as f32;
    let target = Point3::xyz(0.0, 0.0, radius * 0.3);
    let cam = Camera::orbit(
        rect,
        target,
        40.0,
        AZIMUTH,
        ELEVATION,
        Lens::Parallel(radius * 0.8),
    );
    let receiver = place(receiver());
    panel3(c, rect, cam, |s| {
        // Earth: lit, with its latitudes every 30 degrees and its meridians.
        let earth = |u: f32, v: f32| {
            let d = on_sphere(
                core::f32::consts::TAU * u,
                core::f32::consts::PI * (v - 0.5),
            );
            gax_numga_examples::reach3(d)
        };
        s.surface(earth, 36, 18, |_, _| palette::BLUE.faded(0.35), 1.0, None);
        s.sphere_wire(
            gax_numga_examples::ORIGIN3,
            1.003,
            12,
            palette::GRID.faded(2.5),
        );
        for e in &f.emitted {
            let sat = place(*e);
            s.seg(sat, receiver, 1.0, palette::SKY.faded(0.45));
            cuboid(s, sat, [0.065; 3], palette::ORANGE);
            for side in [-0.19, 0.19] {
                let panel = sat + Point3::direction(0.0, 0.0, side);
                cuboid(s, panel, [0.065, 0.012, 0.105], palette::BLUE);
            }
        }
        s.dot(receiver, Marker::Dot, 6.0, palette::ORANGE.whitened(0.3));
        let centre = place(f.estimate);
        let axes = f.spread.map(step);
        let ellipsoid = |u: f32, v: f32| {
            let d = on_sphere(
                core::f32::consts::TAU * u,
                core::f32::consts::PI * (v - 0.5),
            );
            let [a, b, c] = d.c;
            centre + axes[0].gp(a) + axes[1].gp(b) + axes[2].gp(c)
        };
        s.surface(ellipsoid, 32, 16, |_, _| palette::SKY, 0.3, None);
        for a in axes {
            s.seg(centre - a, centre + a, 1.4, palette::SKY.whitened(0.2));
        }
        s.dot(centre, Marker::Dot, 5.0, palette::SKY.whitened(0.4));
    });
}

/// The ellipsoid's semi-axes against the satellites' phase, the frame shown marked.
fn draw_spread(c: &mut Canvas, rect: Rect, s: &Scenes, now: usize) {
    let lengths: Vec<[f32; 3]> = s
        .frames
        .iter()
        .map(|f| f.spread.map(|d| step(d).ideal_norm()))
        .collect();
    let top = lengths.iter().flatten().fold(0.0, |a: f32, v| a.max(*v));
    let ax = Axes::new(
        rect,
        [FIRST_PHASE as f32, LAST_PHASE as f32],
        [0.0, top * 1.1],
    );
    let lights = [palette::GREEN, palette::YELLOW, palette::SKY];
    for (k, light) in lights.iter().enumerate() {
        let curve: Vec<Point2> = s
            .frames
            .iter()
            .zip(&lengths)
            .map(|(f, l)| Point2::xy(f.phase as f32, l[k]))
            .collect();
        ax.polyline(c, &curve, 1.6, *light);
        ax.scatter(c, &[curve[now]], Marker::Dot, 6.0, light.whitened(0.3));
    }
    ax.frame(c, "SPREAD, EARTH RADII", "SATELLITE PHASE (DEGREES)", "");
}

fn draw(c: &mut Canvas, s: &Scenes, t: f32) {
    backdrop(c);
    // Forward, then back.
    let n = s.frames.len();
    let k = gax_numga_examples::signal::frame((t / SECONDS).rem_euclid(1.0), 2 * n);
    let now = if k < n { k } else { 2 * n - 1 - k };
    let f = &s.frames[now];
    let screen = c.rect();
    let (w, h) = (screen.width(), screen.height());
    let body = Rect::new(0.0, h * 0.13, w, h);
    draw_scene(
        c,
        body.part(0.0, 0.0, 0.66, 1.0).inset(8.0, 4.0, 8.0, 8.0),
        f,
    );
    draw_spread(
        c,
        body.part(0.66, 0.0, 1.0, 1.0).inset(44.0, 60.0, 24.0, 90.0),
        s,
        now,
    );
    // The fit's error: its distance from the reception in kilometres, its time off in
    // milliseconds.
    let off = spatial_distance(f.estimate, receiver()) * EARTH_RADIUS_METRES / 1e3;
    let late = (clock() & (f.estimate - receiver())).s() * EARTH_RADIUS_METRES / LIGHT_SPEED * 1e3;
    caption(
        c,
        "GPS: A RECEPTION EVENT FROM SIX LIGHT CONES",
        &format!(
            "INFORMATION OriginHyperplane <- Direction, ITS INVERSE THE COVARIANCE (STAP). FIT OFF BY {off:.0} KM, {late:+.1} MS"
        ),
    );
}

fn main() {
    run_with(Anim::new("gps", SECONDS).size(960, 540), scenes, draw);
}

#[cfg(test)]
mod tests {
    use super::gps::*;
    use gax::stap::{Direction, Event, OriginHyperplane};
    use gax_numga_examples::rng::{Draw, rng};

    /// A real receiver's range error, in Earth radii: small enough for the linearised
    /// covariance to hold.
    const SIGMA: f64 = 3.0 / EARTH_RADIUS_METRES;

    fn close(a: Event, b: Event, tol: f64) -> bool {
        a.c.iter().zip(b.c).all(|(x, y)| (x - y).abs() <= tol)
    }

    /// The readouts read what they say: the clock an event's time, the planes across `x`, `y`
    /// and `z` its place, the plane at infinity its weight.
    #[test]
    fn the_readouts_read_coordinates() {
        let e = origin() + across() * 1.5 + sideways() * -2.0 + upwards() * 0.25 + time() * 3.0;
        let read = |v: gax::stap::Vector| (v & e).s();
        assert_eq!(
            [read(x()), read(y()), read(z()), read(clock()), read(w())],
            [1.5, -2.0, 0.25, 3.0, 1.0]
        );
        let other = origin() + across() * 4.5 + sideways() * 2.0 + upwards() * 0.25 - time() * 1.0;
        let d = spatial_distance(e, other);
        assert!((d - 5.0).abs() < 1e-12, "{d}");
    }

    /// Light cones built on the transmissions reach the receiver; their joined signal lines are
    /// null; four of them meet in two events, one the receiver, after every transmission.
    #[test]
    fn light_cones_recover_the_future_reception_event() {
        for phase in [5.0, 20.0, 50.0] {
            let receiver = receiver();
            let emitted = transmissions(receiver, phase);
            let cones: Vec<Cone> = emitted.iter().map(|e| light_cone(*e)).collect();
            for (cone, e) in cones.iter().zip(&emitted) {
                assert!(cone.of(*e).c.iter().all(|c| c.abs() < 1e-12));
                assert!((receiver & cone.of(receiver)).s().abs() < 1e-12);
                assert!((*e & receiver).norm_squared().abs() < 1e-12);
            }
            let four: [Cone; 4] = core::array::from_fn(|k| cones[k]);
            let candidates = intersect_cones(&four);
            for c in candidates {
                for cone in &four {
                    assert!((c & cone.of(c)).s().abs() < 1e-11);
                }
            }
            let located = reception(candidates, &emitted[..4]);
            assert!(close(located, receiver, 1e-11), "{located:?}");
            assert!(emitted.iter().all(|e| (clock() & (located - *e)).s() > 0.0));
        }
    }

    /// A Poincaré motion carries the cones and their intersection along: the cones of moved
    /// transmissions are the moved cones, and reception after transmission is the same in every
    /// frame a forward-in-time change reaches.
    #[test]
    fn light_cones_and_their_intersections_follow_a_common_frame_change() {
        let receiver = receiver();
        let emitted = transmissions(receiver, 20.0);
        let cones: Vec<Cone> = emitted.iter().map(|e| light_cone(*e)).collect();
        let transform = ((x() ^ w()) * 0.3 - (clock() ^ w()) * 0.15).exp()
            * ((x() ^ clock()) * 0.2 + (y() ^ z()) * 0.1).exp();
        let moved: Vec<Event> = emitted.iter().map(|e| transform >> *e).collect();
        let moved_receiver = transform >> receiver;
        for (cone, e) in cones.iter().zip(&moved) {
            let transported = transform >> cone.of(transform << Event::slot());
            let rebuilt = light_cone(*e);
            assert!(
                (transported - rebuilt)
                    .c
                    .iter()
                    .flatten()
                    .all(|c| c.abs() < 1e-12)
            );
        }
        let rebuilt: [Cone; 4] = core::array::from_fn(|k| light_cone(moved[k]));
        let nearest = reception(intersect_cones(&rebuilt), &moved[..4]);
        assert!(close(nearest, moved_receiver, 1e-11), "{nearest:?}");
        let moved_clock = transform >> clock();
        for (m, e) in moved.iter().zip(&emitted) {
            let there = (moved_clock & (nearest - *m)).s();
            let here = (clock() & (receiver - *e)).s();
            assert!((there - here).abs() < 1e-11 && there > 0.0);
        }
    }

    /// Without noise the fit recovers the reception and the receiver clock's error; a larger
    /// timestamp error changes the uncertainty of the answer, not the answer; the information
    /// is positive and the covariance symmetric (its own adjugate).
    #[test]
    fn noiseless_fit_recovers_position_and_receiver_clock_error() {
        for (phase, reception_time) in [(5.0, -3.0), (20.0, 0.0), (50.0, 4.0)] {
            let receiver = receiver() + time() * reception_time;
            let clock_error = clock_reading();
            let reading = (clock() & receiver).s() + clock_error;
            let emitted = transmissions(receiver, phase);
            let start = initial(receiver, clock_error);
            let (estimate, information, covariance) = locate(&emitted, start, SIGMA, 5);
            assert!(close(estimate, receiver, 1e-12), "{estimate:?}");
            let recovered = reading - (clock() & estimate).s();
            assert!((recovered - clock_error).abs() < 1e-12);
            assert!(
                emitted
                    .iter()
                    .all(|e| (clock() & (estimate - *e)).s() > 0.0)
            );
            let (same, _, larger) = locate(&emitted, start, SIGMA * 2.0, 5);
            assert!(close(same, estimate, 1e-12));
            for (a, b) in larger.c.iter().flatten().zip(covariance.c.iter().flatten()) {
                assert!((a - 4.0 * b).abs() <= 1e-9 * (4.0 * b).abs() + 1e-24);
            }
            for d in [across(), sideways(), upwards(), time()] {
                let d = d.cast::<Direction>();
                assert!((d & information.of(d)).s() > 0.0);
            }
            let asymmetry = covariance - covariance.adjugate();
            assert!(
                asymmetry.c.iter().flatten().all(|c| c.abs() < 1e-23),
                "{asymmetry:?}"
            );
        }
    }

    /// The readouts predict the change of the intervals under small steps of the reception.
    #[test]
    fn cone_readouts_predict_changes_of_the_intervals_under_event_displacements() {
        let receiver = receiver();
        let emitted = transmissions(receiver, 20.0);
        let cones: Vec<Cone> = emitted.iter().map(|e| light_cone(*e)).collect();
        let trial = receiver + upwards() * 0.12 + time() * 0.03;
        let readouts = intervals(&cones, &emitted, trial);
        let epsilon = 1e-4;
        for d in [across(), sideways(), upwards(), time()] {
            let plus = intervals(&cones, &emitted, trial + d * epsilon);
            let minus = intervals(&cones, &emitted, trial - d * epsilon);
            for k in 0..emitted.len() {
                let measured = (plus[k].0 - minus[k].0) / (2.0 * epsilon);
                let predicted = (readouts[k].1 & d.cast::<Direction>()).s();
                assert!(
                    (measured - predicted).abs() < 1e-8,
                    "{measured} {predicted}"
                );
            }
        }
    }

    /// Fitted to many noisy draws of the timestamps, the estimates scatter as the covariance
    /// predicts, for readouts of the position, the clock, and mixtures of both. numpy's stream
    /// cannot be reproduced; with `rand`'s seed 2026 the ratios fall within numga's bounds.
    #[test]
    fn timestamp_noise_matches_the_predicted_event_covariance() {
        let receiver = receiver();
        let emitted = transmissions(receiver, 20.0);
        let start = initial(receiver, clock_reading());
        let (_, _, covariance) = locate(&emitted, start, SIGMA, 5);
        let mut r = rng(2026);
        let trials = 512;
        let probes = [
            x(),
            y(),
            z(),
            clock(),
            x() + clock(),
            y() + z(),
            x() + y() - clock(),
        ]
        .map(|p| p.cast::<OriginHyperplane>());
        let errors: Vec<Vec<f64>> = (0..trials)
            .map(|_| {
                let observed: Vec<Event> = emitted
                    .iter()
                    .map(|e| *e + time() * (r.normal() * SIGMA))
                    .collect();
                let (estimate, _, _) = locate(&observed, start, SIGMA, 5);
                let error = (estimate - receiver).cast::<Direction>();
                probes.iter().map(|p| (*p & error).s()).collect()
            })
            .collect();
        for (k, p) in probes.iter().enumerate() {
            let mean = errors.iter().map(|e| e[k]).sum::<f64>() / trials as f64;
            let variance =
                errors.iter().map(|e| (e[k] - mean).powi(2)).sum::<f64>() / (trials - 1) as f64;
            let predicted = (*p & covariance.of(*p)).s();
            assert!(
                (variance / predicted - 1.0).abs() < 0.3,
                "{k}: {}",
                variance / predicted
            );
            #[allow(clippy::disallowed_methods)] // a standard deviation
            let deviation = predicted.sqrt();
            assert!((mean / deviation).abs() < 0.2, "{k}: {}", mean / deviation);
        }
    }

    /// The scene's fit at its phase is numga's, given numga's noise: the estimate and the first
    /// transmission, to the digits numga prints.
    #[test]
    fn the_scene_matches_numga() {
        let (emitted, estimate, covariance) = positioning(PHASE);
        let read = |e: Event| [x(), y(), z(), clock(), w()].map(|v| (v & e).s());
        let expected = [
            1.681_482_953_736_636,
            0.011_409_711_650_419,
            0.214_327_839_096_337,
            -0.667_695_400_236_398,
            1.0,
        ];
        for (a, b) in read(estimate).iter().zip(expected) {
            assert!((a - b).abs() < 1e-9, "{:?}", read(estimate));
        }
        let first = [
            4.130_034_424_737_846,
            0.580_396_953_540_662,
            0.0,
            -3.183_390_664_640_967,
            1.0,
        ];
        for (a, b) in read(emitted[0]).iter().zip(first) {
            assert!((a - b).abs() < 1e-9, "{:?}", read(emitted[0]));
        }
        // The spread's semi-axes, by their lengths, as numga's.
        let mut lengths: Vec<f64> = position_spread(covariance)
            .iter()
            .map(|d| {
                let e = Event::from(*d);
                [x(), y(), z()]
                    .iter()
                    .map(|v| (*v & e).s().powi(2))
                    .sum::<f64>()
            })
            .collect();
        lengths.sort_by(f64::total_cmp);
        let numga = [
            0.047_006_608_101_275,
            0.080_380_221_436_445,
            1.312_158_324_906_984,
        ];
        for (a, b) in lengths.iter().zip(numga) {
            assert!((a - b).abs() < 1e-8, "{lengths:?}");
        }
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws_with(super::draw, &super::scenes(), 1.0);
    }
}
