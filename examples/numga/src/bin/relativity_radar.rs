//! numga's `relativity/radar`: radar ranging, a moving reflector's trajectory reconstructed from
//! one clock's echo times, in the projective spacetime plane `x+t-w0` (declared here with
//! `gax::algebra!`, PGA2D with a Lorentzian metric).
//!
//! The signs of `x` and `t` give the Minkowski metric, and the degenerate `w` the homogeneous
//! coordinate, so boosts and translations act on events and on line equations by one motor
//! sandwich. Events are intrinsic antivectors (bivectors), worldlines and simultaneous slices
//! extrinsic line equations (vectors), and a slice pairs with an event by the join: `t & e` is
//! the event's clock reading. Joining an event to a light direction and meeting a worldline is
//! a linear map on events (`Event <- Event`); its weight is a constant times the input's, fixed
//! once by a reference event of unit weight, so the map stays linear. Composing the outward leg
//! with the return leg maps emissions to receptions.
//!
//! The clock's own readings reconstruct the reflections: half the sum of emission and
//! reception is the radar time, half the difference the range. Two echoes give the stretch of
//! the echo times, and from it the reflector's velocity and its range at the clock's zero. A
//! common boost, translation or mirror of the whole experiment changes no reading.
//!
//! The animation shows numga's two: light pulses going out to a receding reflector and back, the
//! clock reading each emission and reception; and the reconstruction of the middle echo (radar
//! time and range, every recovered reflection on the reflector's worldline) while the whole
//! experiment is boosted about its middle and translated.

use gax_numga_examples::{
    Align, Anim, Axes, Canvas, Light, Marker, Point2, Rect, backdrop, caption, palette, run_with,
};

gax::algebra! {
    algebra radar "numga's projective spacetime plane `x+t-w0`: space `x` squares to one, time `t` to minus one, `w` to zero.";
    basis ex = 1, et = -1, ew = 0;
    kind Scalar = [1];
    versor Vector = [ex, et, ew] "A line equation: a worldline or a simultaneous slice; as a versor, the reflection in it.";
    kind Bivector = [etw, ewx, ext] "A boost (`xt`) or a translation (`wx`, `tw`) generator; intrinsic, an event.";
    kind Pseudoscalar = [extw];
    versor Even = [1, etw, ewx, ext] "A motion of spacetime: boosts and translations.";
    versor Odd = [ex, et, ew, extw];
    alias Event = Bivector intrinsic;
}

mod ranging {
    pub use super::radar::{Bivector, Even, Event, Vector};
    use gax::Unit;

    /// A map on events.
    pub type EventMap = Event<(Event,)>;

    /// The slices `x = 0`, `t = 0` and the weight: an event's position is `x() & e`, its
    /// clock reading `t() & e`, its weight `w() & e`.
    pub fn x() -> Vector {
        Vector::new(1.0, 0.0, 0.0)
    }
    pub fn t() -> Vector {
        Vector::new(0.0, 1.0, 0.0)
    }
    pub fn w() -> Vector {
        Vector::new(0.0, 0.0, 1.0)
    }

    /// The generators: of boosts (`xt`, squaring to one) and of translations along space
    /// (`xw`) and along time (`tw`), squaring to zero.
    pub fn xt() -> Bivector {
        Bivector::new(0.0, 0.0, 1.0)
    }
    pub fn xw() -> Bivector {
        Bivector::new(0.0, -1.0, 0.0)
    }
    pub fn tw() -> Bivector {
        Bivector::new(1.0, 0.0, 0.0)
    }

    /// The origin, of weight `w & origin = 1`: the complement of the weight.
    pub fn origin() -> Event {
        w().dual()
    }
    /// The ideal events along space and along time, of weight zero, each read as one by its
    /// slice: `x & across = t & time = 1`. (numga writes `-t.dual()`: its complement carries
    /// the metric's sign on `t`, gax's `dual()` is metric-free.)
    pub fn across() -> Event {
        x().dual()
    }
    pub fn time() -> Event {
        t().dual()
    }
    /// The light's directions; joined to the origin they are lines of zero square.
    pub fn outgoing() -> Event {
        time() + across()
    }
    pub fn incoming() -> Event {
        time() - across()
    }

    /// An event's weight, its pairing with the slice of weights.
    pub fn weight(e: Event) -> f64 {
        (w() & e).s()
    }

    /// The reflection and echo maps for straight timelike worldlines and fixed light
    /// directions: join an event to the outgoing direction and meet the reflector's line, then
    /// the same back to the receiver.
    pub fn echo_maps(
        receiver: Vector,
        reflector: Vector,
        outward: Event,
        inward: Event,
        reference: Event,
    ) -> (EventMap, EventMap) {
        let reflection = (Event::slot() & outward) ^ reflector;
        let returning = (Event::slot() & inward) ^ receiver;
        // Fixed directions and straight worldlines make the output's weight a constant times
        // the input's: a unit-weight reference fixes that one scale for the whole map.
        let reflection = reflection * (1.0 / weight(reflection.of(reference)));
        let returning = returning * (1.0 / weight(returning.of(reference)));
        (reflection, returning.of(reflection))
    }

    /// The reflections located by an inertial observer's clock: the radar time (half the sum
    /// of the readings), the event simultaneous with it on the clock's worldline, the range
    /// (half the difference), and the located event, the range along the outgoing spatial
    /// direction.
    #[derive(Clone, Debug)]
    pub struct Located {
        pub located: Vec<Event>,
        pub simultaneous: Vec<Event>,
        pub times: Vec<f64>,
        pub ranges: Vec<f64>,
    }

    pub fn reconstruct(
        emitted: &[Event],
        received: &[Event],
        clock: Vector,
        reference: Event,
        tangent: Event,
        spatial: Event,
    ) -> Located {
        let pairs = emitted.iter().zip(received);
        let times: Vec<f64> = pairs
            .clone()
            .map(|(e, r)| (clock & (*r + *e)).s() / 2.0)
            .collect();
        let ranges: Vec<f64> = pairs.map(|(e, r)| (clock & (*r - *e)).s() / 2.0).collect();
        let simultaneous: Vec<Event> = times.iter().map(|t| reference + tangent * *t).collect();
        let located = simultaneous
            .iter()
            .zip(&ranges)
            .map(|(s, r)| *s + spatial * *r);
        Located {
            located: located.collect(),
            simultaneous,
            times,
            ranges,
        }
    }

    /// The echo stretch, the relative velocity and the range at the clock's zero, from the
    /// first and last echoes: an inertial reflector makes the reception time an affine
    /// function of the emission time.
    #[cfg_attr(not(test), allow(dead_code))] // numga's check of the inference
    pub fn infer_motion(emitted: &[Event], received: &[Event], clock: Vector) -> (f64, f64, f64) {
        let (first, last) = (0, emitted.len() - 1);
        let stretch = (clock & (received[last] - received[first])).s()
            / (clock & (emitted[last] - emitted[first])).s();
        let velocity = (stretch - 1.0) / (stretch + 1.0);
        let first_time = (clock & (received[first] + emitted[first])).s() / 2.0;
        let first_range = (clock & (received[first] - emitted[first])).s() / 2.0;
        (stretch, velocity, first_range - velocity * first_time)
    }

    /// The motion placing a reflector at `distance` moving at `velocity`: the boost by the
    /// rapidity (`xt` squares to one), then the translation along space.
    pub fn placement(distance: f64, velocity: f64) -> Unit<Even> {
        (xw() * (distance / 2.0)).exp() * (xt() * (velocity.atanh() / 2.0)).exp()
    }

    // --- scenarios

    /// The reflector's distance at the clock's zero and its velocity; the pulses, the time
    /// between them, and how long the worldlines run past the last echo.
    pub const DISTANCE: f64 = 2.0;
    pub const VELOCITY: f64 = 0.25;
    pub const PULSES: usize = 3;
    pub const PULSE_SPACING: f64 = 0.75;
    pub const END_MARGIN: f64 = 0.5;
    /// The moving frames: the last frame's rapidity and its shifts in space and time.
    pub const FINAL_RAPIDITY: f64 = -0.65;
    pub const SPATIAL_SHIFT: f64 = 0.3;
    pub const TIME_SHIFT: f64 = 0.2;

    /// The experiment: the two worldline segments (clock and reflector), and the emitted,
    /// reflected and received events.
    #[derive(Clone, Debug)]
    pub struct Exchange {
        pub worldlines: [[Event; 2]; 2],
        pub emitted: Vec<Event>,
        pub reflected: Vec<Event>,
        pub received: Vec<Event>,
    }

    impl Exchange {
        /// The experiment moved by a versor: every event, the clock's readings unchanged.
        pub fn moved(&self, motor: Unit<Even>) -> Exchange {
            let all = |e: &[Event]| -> Vec<Event> { e.iter().map(|e| motor >> *e).collect() };
            Exchange {
                worldlines: self.worldlines.map(|w| w.map(|e| motor >> e)),
                emitted: all(&self.emitted),
                reflected: all(&self.reflected),
                received: all(&self.received),
            }
        }
    }

    pub fn exchange() -> Exchange {
        let placement = placement(DISTANCE, VELOCITY);
        let target = placement >> x();
        let emitted: Vec<Event> = (0..PULSES)
            .map(|k| origin() + time() * (k as f64 * PULSE_SPACING))
            .collect();
        let (reflection, echo) = echo_maps(x(), target, outgoing(), incoming(), origin());
        let reflected = emitted.iter().map(|e| reflection.of(*e)).collect();
        let received: Vec<Event> = emitted.iter().map(|e| echo.of(*e)).collect();
        let finish = (t() & received[PULSES - 1]).s() + END_MARGIN;
        // Where the reflector's worldline meets the slice at the finish.
        let end = target ^ (t() - w() * finish);
        let end = end.with_orientation() * (1.0 / weight(end.with_orientation()));
        Exchange {
            worldlines: [
                [origin(), origin() + time() * finish],
                [placement >> origin(), end],
            ],
            emitted,
            reflected,
            received,
        }
    }

    /// The motion of the experiment at `fraction` of the way through the frames: a boost about
    /// the experiment's middle event, then a translation.
    pub fn frame_motion(experiment: &Exchange, fraction: f64) -> Unit<Even> {
        let middle = (t() & (experiment.emitted[0] + experiment.received[PULSES - 1])).s() / 2.0;
        let centre = (-tw() * (middle / 2.0)).exp();
        let boost = (xt() * (FINAL_RAPIDITY * fraction / 2.0)).exp();
        // The translation generator squares to zero: its exponential is one plus it.
        let translation = ((xw() * SPATIAL_SHIFT - tw() * TIME_SHIFT) * (fraction / 2.0)).exp();
        // Conjugating the boost by the centre anchors it at the middle event.
        translation * centre * boost * centre.reverse()
    }
}

use ranging::*;

/// The loop: numga's frames, 99 of 40 ms, the exchange twice while the frames go out and back.
const FRAMES: usize = 99;
const SECONDS: f32 = 7.92;
/// The diagrams' windows (x, then t), numga's.
const EXCHANGE_WINDOW: [[f32; 2]; 2] = [[-1.55, 4.8], [-0.65, 8.75]];
const FRAME_WINDOW: [[f32; 2]; 2] = [[-4.0, 6.0], [-2.65, 10.15]];

/// The experiment, computed once.
struct Scenes {
    exchange: Exchange,
}

fn scenes() -> Scenes {
    Scenes {
        exchange: exchange(),
    }
}

/// An event drawn: its position across and its clock reading up, read at the drawing boundary
/// (of a unit-weight event).
fn at(e: Event) -> Point2 {
    Point2::xy((x() & e).s() as f32, (t() & e).s() as f32)
}

/// The lights: the clock's worldline, the reflector's, the light going out and coming back,
/// and the reconstruction.
fn clock_light() -> Light {
    palette::INK
}
fn reflector_light() -> Light {
    palette::INK.faded(0.25)
}
fn out_light() -> Light {
    palette::ORANGE
}
fn back_light() -> Light {
    palette::SKY
}
fn found_light() -> Light {
    palette::GREEN
}

/// Equal axes in `rect` showing a window.
fn axes(rect: Rect, [[x0, x1], [t0, t1]]: [[f32; 2]; 2]) -> Axes {
    let half = 0.5 * (t1 - t0).max((x1 - x0) * rect.height() / rect.width());
    Axes::equal(rect, Point2::xy(0.5 * (x0 + x1), 0.5 * (t0 + t1)), half)
}

/// The space and time axes through the origin, faint, labelled.
fn draw_axes(c: &mut Canvas, ax: &Axes, [[x0, x1], [t0, t1]]: [[f32; 2]; 2]) {
    let margin = 0.2;
    let faint = palette::GRID;
    ax.line(
        c,
        Point2::xy(x0 + margin, 0.0),
        Point2::xy(x1 - margin, 0.0),
        1.0,
        faint,
    );
    ax.line(
        c,
        Point2::xy(0.0, t0 + margin),
        Point2::xy(0.0, t1 - margin),
        1.0,
        faint,
    );
    let ink = palette::INK.faded(0.6);
    ax.text(
        c,
        Point2::xy(x1 - margin, -0.45),
        "X",
        10.0,
        ink,
        Align::Center,
    );
    ax.text(
        c,
        Point2::xy(0.15, t1 - margin),
        "T",
        10.0,
        ink,
        Align::Left,
    );
}

/// A clock reading written left of its event.
fn reading(c: &mut Canvas, ax: &Axes, e: Event, text: &str, light: Light) {
    let place = ax.px(at(e)) + Point2::direction(-8.0, 3.0);
    c.text(text, place, 9.0, light, Align::Right);
}

/// Light pulses advancing through the diagram at clock time `now`; their completed paths stay.
fn draw_exchange(c: &mut Canvas, rect: Rect, s: &Exchange, now: f64) {
    let ax = axes(rect, EXCHANGE_WINDOW);
    draw_axes(c, &ax, EXCHANGE_WINDOW);
    let [clock, reflector] = s.worldlines;
    ax.line(c, at(clock[0]), at(clock[1]), 2.0, clock_light());
    ax.line(
        c,
        at(reflector[0]),
        at(reflector[1]),
        2.0,
        reflector_light(),
    );
    // The present: the events at clock time `now` pair to zero with this slice, the earlier
    // ones negatively.
    let present = t() - w() * now;
    let past = |e: Event| (present & e).s() <= 0.0;
    let legs = (s
        .emitted
        .iter()
        .zip(&s.reflected)
        .map(|(a, b)| (*a, *b, out_light())))
    .chain(
        s.reflected
            .iter()
            .zip(&s.received)
            .map(|(a, b)| (*a, *b, back_light())),
    );
    for (start, end, light) in legs {
        ax.line(c, at(start), at(end), 1.0, light.faded(0.16));
        if !past(start) {
            continue;
        }
        // In flight, the pulse is where its line meets the present; arrived, at its end.
        let front = if past(end) {
            end
        } else {
            let crossing: Event = ((start & end) ^ present).with_orientation();
            crossing * (1.0 / weight(crossing))
        };
        ax.line(c, at(start), at(front), 2.0, light);
        if !past(end) {
            ax.scatter(c, &[at(front)], Marker::Dot, 8.0, light.whitened(0.3));
        }
    }
    let reflected = s.reflected.iter().filter(|e| past(**e));
    let reflected: Vec<Point2> = reflected.map(|e| at(*e)).collect();
    ax.scatter(c, &reflected, Marker::Dot, 6.0, out_light());
    for e in s.emitted.iter().chain(&s.received).filter(|e| past(**e)) {
        ax.scatter(c, &[at(*e)], Marker::Dot, 6.0, clock_light());
        let text = format!("{:.2}", (t() & *e).s());
        reading(c, &ax, *e, &text, palette::INK);
    }
}

/// The radar assignment for the middle pulse in a frame moved by `motor`: every recovered
/// reflection marked, the middle one's readings, radar time and range written beside it.
fn draw_reconstruction(c: &mut Canvas, rect: Rect, s: &Exchange, motor: gax::Unit<Even>) {
    let ax = axes(rect, FRAME_WINDOW);
    draw_axes(c, &ax, FRAME_WINDOW);
    let s = s.moved(motor);
    let clock = motor >> t();
    let found = reconstruct(
        &s.emitted,
        &s.received,
        clock,
        motor >> origin(),
        motor >> time(),
        motor >> across(),
    );
    let middle = PULSES / 2;
    let triangle = [s.emitted[middle], s.reflected[middle], s.received[middle]].map(at);
    ax.fill(c, &triangle, back_light().faded(0.3), 0.15);
    let [clock_line, reflector] = s.worldlines;
    ax.line(c, at(clock_line[0]), at(clock_line[1]), 2.0, clock_light());
    ax.line(
        c,
        at(reflector[0]),
        at(reflector[1]),
        2.0,
        reflector_light(),
    );
    for k in 0..PULSES {
        let width = if k == middle { 2.0 } else { 1.0 };
        let fade = if k == middle { 1.0 } else { 0.25 };
        ax.line(
            c,
            at(s.emitted[k]),
            at(s.reflected[k]),
            width,
            out_light().faded(fade),
        );
        ax.line(
            c,
            at(s.reflected[k]),
            at(s.received[k]),
            width,
            back_light().faded(fade),
        );
    }
    let (simultaneous, located) = (found.simultaneous[middle], found.located[middle]);
    ax.dashed(c, &[at(simultaneous), at(located)], 1.7, 6.0, found_light());
    let recovered: Vec<Point2> = found.located.iter().map(|e| at(*e)).collect();
    ax.scatter(c, &recovered, Marker::Ring, 9.0, found_light());
    ax.scatter(
        c,
        &[at(s.emitted[middle]), at(s.received[middle])],
        Marker::Dot,
        6.0,
        clock_light(),
    );
    ax.scatter(c, &[at(simultaneous)], Marker::Dot, 6.0, found_light());
    let readings = [s.emitted[middle], s.received[middle]].map(|e| (clock & e).s());
    reading(
        c,
        &ax,
        s.emitted[middle],
        &format!("T- = {:.2}", readings[0]),
        palette::INK,
    );
    reading(
        c,
        &ax,
        s.received[middle],
        &format!("T+ = {:.2}", readings[1]),
        palette::INK,
    );
    let time = found.times[middle];
    reading(
        c,
        &ax,
        simultaneous,
        &format!("T = {time:.2}"),
        found_light(),
    );
    // The range written halfway along the slice of simultaneity to the reflection.
    let halfway = ax.px(at((simultaneous + located) * 0.5)) - Point2::direction(0.0, 8.0);
    let range = format!("R = {:.2}", found.ranges[middle]);
    c.text(&range, halfway, 9.0, found_light(), Align::Center);
}

fn draw(c: &mut Canvas, s: &Scenes, t: f32) {
    backdrop(c);
    let screen = c.rect();
    let (w, h) = (screen.width(), screen.height());
    let loop_phase = (t / SECONDS).rem_euclid(1.0);
    // The exchange twice a loop, from just before the first emission to after the last echo.
    let first = (ranging::t() & s.exchange.emitted[0]).s();
    let finish = (ranging::t() & s.exchange.received[PULSES - 1]).s();
    let exchange_phase = f64::from((2.0 * loop_phase).rem_euclid(1.0));
    let now = first - 0.3 + (finish + 0.7 - (first - 0.3)) * exchange_phase;
    // The frames out and back.
    let out_and_back = 1.0 - (2.0 * loop_phase - 1.0).abs();
    let step = (out_and_back * (FRAMES - 1) as f32).round() as usize;
    let fraction = step as f64 / (FRAMES - 1) as f64;
    let body = Rect::new(0.0, h * 0.14, w, h);
    let left = body.column(0, 2).inset(16.0, 22.0, 16.0, 10.0);
    let right = body.column(1, 2).inset(16.0, 22.0, 16.0, 10.0);
    draw_exchange(c, left, &s.exchange, now);
    draw_reconstruction(c, right, &s.exchange, frame_motion(&s.exchange, fraction));
    let label = |c: &mut Canvas, rect: Rect, text: &str| {
        let at = rect.top_middle() - Point2::direction(0.0, 6.0);
        c.text(text, at, 10.0, palette::INK, Align::Center);
    };
    label(c, left, "ECHOES FROM A RECEDING REFLECTOR");
    label(c, right, "LOCATED FROM THE CLOCK, IN A MOVING FRAME");
    caption(
        c,
        "RADAR: A TRAJECTORY FROM ONE CLOCK'S ECHO TIMES",
        "JOIN AN EVENT TO THE LIGHT, MEET THE WORLDLINE: A MAP ON EVENTS (X+T-W0)",
    );
}

fn main() {
    run_with(Anim::new("radar", SECONDS).size(960, 540), scenes, draw);
}

#[cfg(test)]
mod tests {
    use super::ranging::*;

    const ATOL: f64 = 1e-9;

    /// Every coefficient of an event within `ATOL` of zero.
    fn vanishes(e: Event) -> bool {
        e.c.iter().all(|x| x.abs() <= ATOL)
    }

    #[test]
    fn echo_maps_and_reconstruction() {
        for velocity in [-0.6, 0.0, 0.25, 0.85] {
            let distance = 2.3;
            let times = [0.2, 0.6, 1.1];
            let placement = placement(distance, velocity);
            let target = placement >> x();
            let emitted: Vec<Event> = times.iter().map(|t| origin() + time() * *t).collect();
            let (reflection, echo) = echo_maps(x(), target, outgoing(), incoming(), origin());
            let reflected: Vec<Event> = emitted.iter().map(|e| reflection.of(*e)).collect();
            let received: Vec<Event> = emitted.iter().map(|e| echo.of(*e)).collect();

            // The round trips in closed form, independent of the construction.
            let stretch = (1.0 + velocity) / (1.0 - velocity);
            for k in 0..times.len() {
                let expected = 2.0 * distance / (1.0 - velocity) + stretch * times[k];
                assert!(((t() & received[k]).s() - expected).abs() <= ATOL);
                let (where_, when) = ((x() & reflected[k]).s(), (t() & reflected[k]).s());
                assert!((where_ - (distance + velocity * when)).abs() <= ATOL);
                assert!((target & reflected[k]).s().abs() <= ATOL);
                assert!((x() & received[k]).s().abs() <= ATOL);
                for ray in [emitted[k] & reflected[k], reflected[k] & received[k]] {
                    assert!(ray.gp(ray).s().abs() <= ATOL, "light travels on null lines");
                }
                assert!((t() & (reflected[k] - emitted[k])).s() > 0.0);
                assert!((t() & (received[k] - reflected[k])).s() > 0.0);
            }

            let found = reconstruct(&emitted, &received, t(), origin(), time(), across());
            for (k, hit) in reflected.iter().enumerate() {
                assert!(vanishes(found.located[k] - *hit));
                assert!((t() & (found.located[k] - found.simultaneous[k])).s().abs() <= ATOL);
                assert!((found.times[k] - (t() & *hit).s()).abs() <= ATOL);
                assert!((found.ranges[k] - (x() & *hit).s()).abs() <= ATOL);
            }
            let (inferred_stretch, inferred_velocity, initial) =
                infer_motion(&emitted, &received, t());
            assert!((inferred_stretch - stretch).abs() <= ATOL);
            assert!((inferred_velocity - velocity).abs() <= ATOL);
            assert!((initial - distance).abs() <= ATOL);
        }
    }

    /// A common boost, translation and (or not) mirror of the whole experiment keeps every
    /// event's relations and every clock reading.
    #[test]
    fn common_frame_change_preserves_events_and_clock_readings() {
        let s = exchange();
        let motion = (xw() * 0.4 - tw() * 0.3).exp() * (xt() * -0.4).exp();
        let target_placement = xw().exp() * (xt() * (0.25f64.atanh() / 2.0)).exp();
        let check = |moved: &dyn Fn(Event) -> Event, slice: &dyn Fn(Vector) -> Vector| {
            let target = slice(target_placement >> x());
            let (clock, reference) = (slice(t()), moved(origin()));
            let sent: Vec<Event> = s.emitted.iter().map(|e| moved(*e)).collect();
            let (reflection, echo) = echo_maps(
                slice(x()),
                target,
                moved(outgoing()),
                moved(incoming()),
                reference,
            );
            let hit: Vec<Event> = sent.iter().map(|e| reflection.of(*e)).collect();
            let returned: Vec<Event> = sent.iter().map(|e| echo.of(*e)).collect();
            for k in 0..PULSES {
                assert!(vanishes(hit[k] - moved(s.reflected[k])));
                assert!(vanishes(returned[k] - moved(s.received[k])));
                assert!(((clock & returned[k]).s() - (t() & s.received[k]).s()).abs() <= ATOL);
            }
            let found = reconstruct(
                &sent,
                &returned,
                clock,
                reference,
                moved(time()),
                moved(across()),
            );
            for (k, at) in hit.iter().enumerate() {
                assert!(vanishes(found.located[k] - *at));
                assert!((found.ranges[k] - (x() & s.reflected[k]).s()).abs() <= ATOL);
            }
            let (stretch, velocity, distance) = infer_motion(&sent, &returned, clock);
            assert!((stretch - 5.0 / 3.0).abs() <= ATOL);
            assert!((velocity - 0.25).abs() <= ATOL);
            assert!((distance - 2.0).abs() <= ATOL);
        };
        check(&|e| motion >> e, &|v| motion >> v);
        // Mirrored: the motion after the reflection in the slice `x = 0`.
        let mirrored = motion * x().unit();
        check(&|e| mirrored >> e, &|v| mirrored >> v);
    }

    /// numga's notebook check: the scenario's reflections are located, at the velocity sent.
    #[test]
    fn the_scenario_locates_its_reflections() {
        let s = exchange();
        let found = reconstruct(&s.emitted, &s.received, t(), origin(), time(), across());
        for k in 0..PULSES {
            assert!(vanishes(found.located[k] - s.reflected[k]));
        }
        let (_, velocity, _) = infer_motion(&s.emitted, &s.received, t());
        assert!((velocity - 0.25).abs() <= ATOL);
        // And in every moved frame.
        for fraction in [0.0, 0.5, 1.0] {
            let motor = frame_motion(&s, fraction);
            let moved = s.moved(motor);
            let clock = motor >> t();
            let found = reconstruct(
                &moved.emitted,
                &moved.received,
                clock,
                motor >> origin(),
                motor >> time(),
                motor >> across(),
            );
            for k in 0..PULSES {
                assert!(vanishes(found.located[k] - moved.reflected[k]));
            }
        }
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws_with(super::draw, &super::scenes(), 0.5);
    }
}
