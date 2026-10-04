//! numga's `relativity/impulse`: strain-preserving impulses on a relativistic rod, the ladder
//! paradox and Bell's spaceships, in the spacetime plane `R(1,1)` (declared here with
//! `gax::algebra!`). Coordinates are time and position along `t` and `x`, in units where the
//! speed of light and the rod's proper length are one.
//!
//! An impulse changes a worldline's slope, not its position. Start in the frame where the rod
//! reverses everywhere at once at the same speed: before and after are mirror images and the
//! length is unchanged. A boost, the sandwich of an open vector slot `R >> Vector::slot()`,
//! changes the observer of the whole drawing: the line joining the kinks tilts into the
//! hyperbolic bisector of the motions, and the observer's length changes continuously as the
//! ends reach their kinks at different times. Repeating one fixed boost and one fixed
//! translation builds a train of such steps.
//!
//! The animation grows the worldlines up the spacetime diagrams with a moving equal-time slice
//! (`NOW`) that reads the rod's current length, through three scenes: the impulse, the ladder
//! in the barn (it fits in motion, but cannot stop inside relaxed), and Bell's spaceships
//! (synchronized clocks stretch the rope; the bisector train does not).

use std::sync::OnceLock;

use gax::pga2d::Point;
use gax_numga_examples::canvas::srgb;
use gax_numga_examples::font;
use gax_numga_examples::{
    Align, Anim, Axes, Canvas, Marker, Point2, Pos2, Rgb, backdrop, caption, palette, run,
};

gax::algebra! {
    algebra spacetime "The spacetime plane R(1,1): time `et` and one direction of space `ex`.";
    basis et = 1, ex = -1;
    kind Scalar = [1];
    versor Vector = [et, ex];
    kind Bivector = [etx];
    versor Even = [1, etx];
}

mod impulse {
    use super::spacetime::{Bivector, Vector};

    pub type V = Vector<(), f64>;
    /// A map on events and directions: an observer change.
    pub type VectorMap = Vector<(Vector,), f64>;

    pub fn t() -> V {
        Vector::new(1.0, 0.0)
    }
    pub fn x() -> V {
        Vector::new(0.0, 1.0)
    }
    /// The event at time `time` and position `position`.
    pub fn event(time: f64, position: f64) -> V {
        Vector::new(time, position)
    }
    /// The unit bivector `t ∧ x`; it squares to +1.
    pub fn tx() -> Bivector<(), f64> {
        Bivector::new(1.0)
    }
    /// An event's time, `v · t`.
    pub fn time_of(v: V) -> f64 {
        (v | t()).s()
    }
    /// An event's position, `-v · x` (`x` squares to -1).
    pub fn position_of(v: V) -> f64 {
        -(v | x()).s()
    }

    /// The Lorentz boost adding `rapidity` to every velocity, as an open sandwich on events and
    /// directions.
    pub fn boost(rapidity: f64) -> VectorMap {
        (tx() * (-rapidity / 2.0)).exp() >> Vector::slot()
    }

    /// Join unit future timelike directions with a unit-proper-length step: the summed
    /// directions give the midpoint observer's time axis, its perpendicular (the product with
    /// `t ∧ x`) the kink line, scaled so that both rest frames measure unit proper spacing.
    /// The left kink is at the origin.
    pub fn strain_preserving_kinks(before: V, after: V) -> [V; 2] {
        let separation = ((before + after) * tx()) * (1.0 / (1.0 + (before | after).s()));
        [Vector::zero(), separation]
    }

    /// A rigid-to-rigid step anchored at the left end's kink: the two kink events, and the unit
    /// directions before and after.
    pub fn velocity_step(before_rapidity: f64, after_rapidity: f64) -> ([V; 2], [V; 2]) {
        let directions = [
            boost(before_rapidity).of(t()),
            boost(after_rapidity).of(t()),
        ];
        (
            strain_preserving_kinks(directions[0], directions[1]),
            directions,
        )
    }

    /// One fixed boost and one fixed translation, composed again and again: the kink events
    /// per step (left, right), and the directions before, between and after the steps. `dt` is
    /// the left end's proper time between kinks.
    pub fn small_impulses(final_rapidity: f64, count: usize, dt: f64) -> (Vec<[V; 2]>, Vec<V>) {
        let step = boost(final_rapidity / count as f64);
        let before = t();
        let mut after = step.of(before);
        let mut events = strain_preserving_kinks(before, after);
        let offset = after * dt;
        let mut history = vec![events];
        let mut directions = vec![before, after];
        for _ in 1..count {
            // Both the boost and the translation stay fixed throughout the train.
            events = events.map(|e| step.of(e) + offset);
            after = step.of(after);
            history.push(events);
            directions.push(after);
        }
        (history, directions)
    }

    /// The event at observer time `time` on the straight worldline through `at` along
    /// `direction`.
    pub fn coasting(at: V, direction: V, time: f64) -> V {
        at + direction * ((time - time_of(at)) / time_of(direction))
    }

    /// Each end's event at observer time `time` on its piecewise straight worldline: start on
    /// the worldline before the first kink, then add each velocity jump once that end has
    /// crossed its kink.
    pub fn worldline_events(time: f64, kinks: &[[V; 2]], directions: &[V]) -> [V; 2] {
        let rates: Vec<V> = directions
            .iter()
            .map(|d| *d * (1.0 / time_of(*d)))
            .collect();
        core::array::from_fn(|end| {
            let elapsed = |s: usize| time - time_of(kinks[s][end]);
            let mut p = kinks[0][end] + rates[0] * elapsed(0);
            for s in 0..kinks.len() {
                p += (rates[s + 1] - rates[s]) * elapsed(s).max(0.0);
            }
            p
        })
    }

    /// Each end's worldline as a polyline over the observer times `span`, through its kinks.
    pub fn worldlines(kinks: &[[V; 2]], directions: &[V], span: [f64; 2]) -> Vec<[V; 2]> {
        let mut out = vec![worldline_events(span[0], kinks, directions)];
        out.extend_from_slice(kinks);
        out.push(worldline_events(span[1], kinks, directions));
        out
    }

    /// An underdamped mode that starts at length 0.6 with zero rate and settles to length 1.
    pub fn ringing_length(time: f64, damping_ratio: f64, natural_frequency: f64) -> f64 {
        let damping = damping_ratio * natural_frequency;
        let frequency = natural_frequency * (1.0 - damping_ratio * damping_ratio).sqrt();
        1.0 - 0.4
            * (-damping * time).exp()
            * ((frequency * time).cos() + damping / frequency * (frequency * time).sin())
    }

    // --- the scenes -------------------------------------------------------------------

    /// A symmetric reversal, the same drawing boosted, and a train of ten small impulses.
    // Some fields serve only the tests (numga's scenario checks).
    #[cfg_attr(not(test), allow(dead_code))]
    pub struct Impulse {
        /// Per view (symmetric, boosted): the worldlines, the kinks, the slices before and
        /// after, and the directions before and after.
        pub worldlines: [Vec<[V; 2]>; 2],
        pub kinks: [[V; 2]; 2],
        pub slices: [[[V; 2]; 2]; 2],
        pub directions: [[V; 2]; 2],
        /// The symmetric observer's time axis.
        pub bisector: V,
        pub train: Vec<[V; 2]>,
        pub steps: Vec<[V; 2]>,
        pub train_slice: [V; 2],
    }

    pub fn impulse() -> Impulse {
        let speed: f64 = 0.5;
        let half_rapidity = speed.atanh();
        // Start in the symmetric frame: two identical velocity reversals at time zero.
        let (kinks, directions) = velocity_step(-half_rapidity, half_rapidity);
        let bisector = (directions[0] + directions[1]).normalized().into_inner();
        // Leave the passenger open to obtain the observer change as a map; apply each to both
        // events and tangents, preserving incidence.
        let views = [Vector::slot(), boost(half_rapidity)];
        let view_kinks = views.map(|m| kinks.map(|k| m.of(k)));
        let view_directions = views.map(|m| directions.map(|d| m.of(d)));
        let view_worldlines = core::array::from_fn(|v| {
            worldlines(&[view_kinks[v]], &view_directions[v], [-0.60, 1.45])
        });
        let slices = core::array::from_fn(|v| {
            [-0.30, 1.05].map(|time| worldline_events(time, &[view_kinks[v]], &view_directions[v]))
        });
        // Repeat a fixed boost and a fixed translation to build a train of impulses.
        let (steps, train_directions) = small_impulses(2.0 * half_rapidity, 10, 0.1);
        // The front's last kink ends the train.
        let finish = time_of(steps[steps.len() - 1][1]);
        let train = worldlines(&steps, &train_directions, [-0.30, finish + 0.40]);
        let train_slice = worldline_events(finish + 0.20, &steps, &train_directions);
        Impulse {
            worldlines: view_worldlines,
            kinks: view_kinks,
            slices,
            directions: view_directions,
            bisector,
            train,
            steps,
            train_slice,
        }
    }

    /// The ladder paradox, in four panels: the barn frame, the incoming ladder frame, the
    /// strain-preserving stop, and the stop inside with its ringing.
    // Some fields serve only the tests (numga's scenario checks).
    #[cfg_attr(not(test), allow(dead_code))]
    pub struct Ladder {
        pub barn_doors: Vec<[V; 2]>,
        pub closure: [V; 2],
        pub barn_ladder: Vec<[V; 2]>,
        pub incoming: [V; 2],
        pub moving_doors: Vec<[V; 2]>,
        pub moving_closure: [V; 2],
        pub moving_ladder: Vec<[V; 2]>,
        pub moving_events: [V; 2],
        pub moving_direction: V,
        pub ladder_cut: [V; 2],
        pub barn_cut: [V; 2],
        pub doors: Vec<[V; 2]>,
        pub stop_ladder: Vec<[V; 2]>,
        pub stop_kinks: [V; 2],
        pub stop_directions: [V; 2],
        pub stopped: [V; 2],
        pub contact: V,
        pub ring_incoming: Vec<[V; 2]>,
        pub ring: Vec<[V; 2]>,
        pub relaxed: Vec<[V; 2]>,
        pub contacts: [V; 2],
        pub relaxed_cut: [V; 2],
    }

    /// `[a, b]` as events at both ends of each of the given times.
    fn over(times: [f64; 2], f: impl Fn(f64) -> [V; 2]) -> Vec<[V; 2]> {
        times.iter().map(|t| f(*t)).collect()
    }

    /// A ladder of relaxed proper length 1 enters a barn of length 0.8 at 0.8 of the speed of
    /// light: its barn-frame length is 0.6, so both doors can close with it inside. In its own
    /// frame the closures are not simultaneous. Stopping it by the bisector timing restores its
    /// full length beyond the exit; stopping every point at once keeps it at 0.6, compressed,
    /// and the compression rings about the relaxed length.
    pub fn ladder() -> Ladder {
        let (beta, barn_length): (f64, f64) = (0.8, 0.8);
        let rapidity = beta.atanh();
        let moving_length = 1.0 / rapidity.cosh();
        let rear_at_closure = (barn_length - moving_length) / 2.0;
        // The door-closure events, and the ladder's ends at closure.
        let closure = [event(0.0, 0.0), event(0.0, barn_length)];
        let incoming = [
            event(0.0, rear_at_closure),
            event(0.0, rear_at_closure + moving_length),
        ];
        let incoming_direction = boost(rapidity).of(t());
        let at_rest = t();
        let coast = |events: [V; 2], direction: V| {
            move |time: f64| events.map(|e| coasting(e, direction, time))
        };

        // 1. The ordinary paradox: simultaneous door closure, without a stop.
        let span = [-0.6, 0.65];
        let barn_doors = over(span, coast(closure, at_rest));
        let barn_ladder = over(span, coast(incoming, incoming_direction));

        // 2. One map transforms all events and tangents to the incoming rest frame.
        let to_ladder = boost(-rapidity);
        let moving_closure = closure.map(|e| to_ladder.of(e));
        let moving_events = incoming.map(|e| to_ladder.of(e));
        let moving_direction = to_ladder.of(incoming_direction);
        let moving_doors_direction = to_ladder.of(at_rest);
        let span = [-1.62, 0.30];
        let moving_doors = over(span, coast(moving_closure, moving_doors_direction));
        let moving_ladder = over(span, coast(moving_events, moving_direction));
        let ladder_cut = coast(moving_events, moving_direction)(-1.40);
        let barn_cut = coast(moving_closure, moving_doors_direction)(-0.72);

        // 3. The unchanged-proper-length stop is the same velocity-step building block.
        let limits = [-0.48, 2.12];
        let (kinks, stop_directions) = velocity_step(rapidity, 0.0);
        let stop_kinks = kinks.map(|k| k + event(0.0, rear_at_closure));
        let contact_time = (barn_length - rear_at_closure - moving_length) / beta;
        let stopped = worldline_events(1.43, &[stop_kinks], &stop_directions);
        let doors = over(limits, coast(closure, at_rest));
        let stop_ladder = worldlines(&[stop_kinks], &stop_directions, limits);
        let contact = worldline_events(contact_time, &[stop_kinks], &stop_directions)[1];

        // 4. Stopping inside leaves compression, whose relaxation rings about the relaxed
        //    length, centred in the barn. First contact with both doors is found on the
        //    expanding part of the mode.
        let (damping_ratio, natural_frequency) = (0.25, 5.0);
        let centre = barn_length / 2.0;
        let ring_times: Vec<f64> = (0..1000).map(|i| 2.12 * i as f64 / 999.0).collect();
        let lengths: Vec<f64> = ring_times
            .iter()
            .map(|t| ringing_length(*t, damping_ratio, natural_frequency))
            .collect();
        let ring = ring_times
            .iter()
            .zip(&lengths)
            .map(|(t, l)| [event(*t, centre - 0.5 * l), event(*t, centre + 0.5 * l)])
            .collect();
        let relaxed_ends = [event(0.0, centre - 0.5), event(0.0, centre + 0.5)];
        let first_peak = core::f64::consts::PI
            / (natural_frequency * (1.0 - damping_ratio * damping_ratio).sqrt());
        let k = (1..ring_times.len())
            .find(|&k| ring_times[k] <= first_peak && lengths[k] >= barn_length)
            .expect("the ring reaches the doors");
        let fraction = (barn_length - lengths[k - 1]) / (lengths[k] - lengths[k - 1]);
        let ringing_contact = ring_times[k - 1] + fraction * (ring_times[k] - ring_times[k - 1]);
        Ladder {
            barn_doors,
            closure,
            barn_ladder,
            incoming,
            moving_doors,
            moving_closure,
            moving_ladder,
            moving_events,
            moving_direction,
            ladder_cut,
            barn_cut,
            doors,
            stop_ladder,
            stop_kinks,
            stop_directions,
            stopped,
            contact,
            ring_incoming: over([-0.48, 0.0], coast(incoming, incoming_direction)),
            ring,
            relaxed: over([0.0, 2.12], coast(relaxed_ends, at_rest)),
            contacts: coast(closure, at_rest)(ringing_contact),
            relaxed_cut: coast(relaxed_ends, at_rest)(1.91),
        }
    }

    /// Bell's spaceships: the strain-preserving train against identical programs on clocks
    /// synchronized in the lab, per schedule: worldlines, kinks, the slices at the start and in
    /// the end, and the last kinks seen in the final rest frame (from the rear's).
    // Some fields serve only the tests (numga's scenario checks).
    #[cfg_attr(not(test), allow(dead_code))]
    pub struct Spaceships {
        pub tracks: [Vec<[V; 2]>; 2],
        pub schedules: [Vec<[V; 2]>; 2],
        pub slices: [[[V; 2]; 2]; 2],
        pub final_events: [[V; 2]; 2],
        pub directions: Vec<V>,
        pub final_frame: VectorMap,
    }

    pub fn spaceships() -> Spaceships {
        let (beta, count, dt): (f64, usize, f64) = (0.8, 10, 0.1);
        let rapidity = beta.atanh();
        let (preserved, directions) = small_impulses(rapidity, count, dt);
        // Bell's front follows an exact spatial translation of the same rear path.
        let bell: Vec<[V; 2]> = preserved.iter().map(|s| [s[0], s[0] + x()]).collect();
        let schedules = [preserved, bell];
        // Boost the final coasting worldlines, then measure at equal final-frame time, from the
        // rear's final kink. The front's final worldline is at rest there, so its position is
        // the gap.
        let final_frame = boost(-rapidity);
        let final_events = core::array::from_fn(|s| {
            let last = schedules[s][count - 1];
            last.map(|e| final_frame.of(e - last[0]))
        });
        // The front's last kink ends the train.
        let end_time = time_of(schedules[0][count - 1][1]) + 0.53;
        let tracks =
            core::array::from_fn(|s| worldlines(&schedules[s], &directions, [-0.30, end_time]));
        let slices = core::array::from_fn(|s| {
            [-0.18, end_time - 0.22].map(|time| worldline_events(time, &schedules[s], &directions))
        });
        Spaceships {
            tracks,
            schedules,
            slices,
            final_events,
            directions,
            final_frame,
        }
    }
}

use impulse::*;

/// Each scene's share of the loop, in seconds.
const SCENE: f32 = 7.0;

fn rear() -> Rgb {
    palette::sky()
}
fn front() -> Rgb {
    palette::orange()
}
fn kink() -> Rgb {
    palette::green()
}
fn door() -> Rgb {
    srgb(0.55, 0.58, 0.64)
}
fn elastic() -> Rgb {
    palette::purple()
}
fn contact() -> Rgb {
    srgb(0.92, 0.36, 0.36)
}

/// An event where the diagrams draw it: position across, time up.
impl Pos2 for V {
    fn point2(self) -> Point2 {
        Point::xy(position_of(self), time_of(self)).point2()
    }
}

/// One end's worldline from a list of `[left, right]` events.
fn end_line(events: &[[V; 2]], end: usize) -> Vec<V> {
    events.iter().map(|e| e[end]).collect()
}

/// Axes of equal scales showing `xr` by `yr`, as large as fits in `rect`, centred.
fn fit(rect: [f32; 4], xr: [f32; 2], yr: [f32; 2]) -> Axes {
    let (w, h) = (rect[2] - rect[0], rect[3] - rect[1]);
    let k = (w / (xr[1] - xr[0])).min(h / (yr[1] - yr[0]));
    let (pw, ph) = (k * (xr[1] - xr[0]), k * (yr[1] - yr[0]));
    let (x0, y0) = (rect[0] + (w - pw) * 0.5, rect[1] + (h - ph) * 0.5);
    Axes::new([x0, y0, x0 + pw, y0 + ph], xr, yr)
}

/// A panel title above the axes, shrunk to fit their width.
fn title(c: &mut Canvas, ax: &Axes, text: &str, colour: Rgb) {
    let [x0, y0, x1, _] = ax.rect;
    let size = (12.0f32).min((x1 - x0 + 30.0) / font::width(text, 1.0));
    c.text(
        text,
        (x0 + x1) * 0.5,
        y0 - size * 1.1,
        size,
        colour,
        Align::Center,
    );
}

/// The part of a worldline (events in time order) up to the observer time `tau`.
fn upto(events: &[V], tau: f64) -> Vec<V> {
    let mut out = vec![];
    for w in events.windows(2) {
        let (a, b) = (w[0], w[1]);
        if out.is_empty() {
            if time_of(a) > tau {
                return out;
            }
            out.push(a);
        }
        if time_of(b) <= tau {
            out.push(b);
        } else {
            out.push(coasting(a, b - a, tau));
            return out;
        }
    }
    out
}

/// Where a worldline is at the observer time `tau`, if it is there then.
fn at(events: &[V], tau: f64) -> Option<V> {
    events.windows(2).find_map(|w| {
        let (a, b) = (w[0], w[1]);
        (time_of(a) <= tau && tau <= time_of(b) && time_of(b) > time_of(a))
            .then(|| coasting(a, b - a, tau))
    })
}

/// A spacetime diagram with its worldlines growing up to the time `tau`.
struct Diagram<'a> {
    ax: Axes,
    /// The present, an observer time.
    tau: f64,
    size: f32,
    c: &'a mut Canvas,
}

impl Diagram<'_> {
    fn new<'a>(
        c: &'a mut Canvas,
        rect: [f32; 4],
        xr: [f32; 2],
        yr: [f32; 2],
        reveal: f32,
    ) -> Diagram<'a> {
        let ax = fit(rect, xr, yr);
        let size = ((ax.rect[2] - ax.rect[0]) / 26.0).clamp(7.0, 11.0);
        // The present rises from the bottom of the diagram to its top.
        let tau = f64::from(yr[0] + (yr[1] - yr[0]) * reveal);
        // Light cones through the origin, faint: the null directions `t ± x`, far out.
        let big = 10.0;
        for null in [t() + x(), t() - x()] {
            ax.line(c, null * -big, null * big, 0.8, palette::grid(), 0.35);
        }
        Diagram { ax, tau, size, c }
    }

    /// A worldline: faint in full, bright up to the present.
    fn worldline(&mut self, events: &[V], colour: Rgb, width: f32) {
        self.ax.polyline(self.c, events, width * 0.6, colour, 0.18);
        self.ax
            .polyline(self.c, &upto(events, self.tau), width, colour, 1.0);
    }

    /// Both ends' worldlines.
    fn rod(&mut self, events: &[[V; 2]], width: f32) {
        self.worldline(&end_line(events, 0), rear(), width);
        self.worldline(&end_line(events, 1), front(), width);
    }

    /// A dashed line through events, shown once the present has passed its first.
    fn dashed(&mut self, events: &[V], colour: Rgb, width: f32) {
        let first = events
            .iter()
            .map(|e| time_of(*e))
            .fold(f64::INFINITY, f64::min);
        let alpha = if first <= self.tau { 0.95 } else { 0.25 };
        self.ax.dashed(self.c, events, width, 4.0, colour, alpha);
    }

    /// Events, bright once the present has passed them.
    fn events(&mut self, events: &[V], marker: Marker, size: f32, colour: Rgb) {
        for e in events {
            let alpha = if time_of(*e) <= self.tau { 1.0 } else { 0.25 };
            self.ax.scatter(self.c, &[*e], marker, size, colour, alpha);
        }
    }

    /// A length bar across two equal-time events, labelled above (`dy > 0`) or below, once the
    /// present has reached it.
    fn bar(&mut self, [a, b]: [V; 2], label: &str, dy: f32, colour: Rgb) {
        if time_of(a) > self.tau {
            return;
        }
        self.ax.line(self.c, a, b, 5.0, colour, 0.35);
        let [px, py] = self.ax.px((a + b) * 0.5);
        let y = if dy > 0.0 {
            py - self.size * 0.8
        } else {
            py + self.size * 1.7
        };
        self.c.text(label, px, y, self.size, colour, Align::Center);
    }

    /// The present: the equal-time slice between two worldlines, with its length.
    fn now(&mut self, left: &[V], right: &[V]) {
        if let (Some(a), Some(b)) = (at(left, self.tau), at(right, self.tau)) {
            self.ax.line(self.c, a, b, 1.6, palette::yellow(), 0.9);
            let [px, py] = self.ax.px(b);
            let text = format!("{:.2}", position_of(b) - position_of(a));
            self.c.text(
                &text,
                px + self.size * 0.6,
                py + self.size * 0.4,
                self.size,
                palette::yellow(),
                Align::Left,
            );
        }
    }

    fn title(&mut self, text: &str) {
        title(self.c, &self.ax, text, palette::ink());
    }

    fn text(&mut self, at: impl Pos2, text: &str, colour: Rgb) {
        self.ax
            .text(self.c, at, text, self.size, colour, Align::Center);
    }

    fn frame(&mut self) {
        self.ax.frame(self.c, "", "X / L0", "");
        let [x0, y0, ..] = self.ax.rect;
        let at = [x0 + 4.0, y0 + self.size * 1.4];
        self.c
            .text("CT", at[0], at[1], self.size, palette::ink(), Align::Left);
    }
}

/// The scenes' geometry, computed once.
fn scenes() -> &'static (Impulse, Ladder, Spaceships) {
    static SCENES: OnceLock<(Impulse, Ladder, Spaceships)> = OnceLock::new();
    SCENES.get_or_init(|| (impulse(), ladder(), spaceships()))
}

/// Smoothly from 0 to 1 over the first 85% of a scene, then held.
fn reveal(u: f32) -> f32 {
    let s = (u / 0.85).clamp(0.0, 1.0);
    s * s * (3.0 - 2.0 * s)
}

fn draw_impulse(c: &mut Canvas, r: f32) {
    let (s, _, _) = scenes();
    let (w, h) = (c.width as f32, c.height as f32);
    let rect = |i: usize| {
        [
            w * i as f32 / 3.0 + 40.0,
            h * 0.2,
            w * (i + 1) as f32 / 3.0 - 8.0,
            h - 60.0,
        ]
    };
    let titles = [
        "SYMMETRIC FRAME: -0.5C -> +0.5C",
        "BOOSTED FRAME: 0 -> 0.8C",
    ];
    for (v, heading) in titles.into_iter().enumerate() {
        let mut d = Diagram::new(c, rect(v), [-0.16, 1.93], [-0.70, 1.53], r);
        d.rod(&s.worldlines[v], 2.4);
        d.dashed(&s.kinks[v], kink(), 2.0);
        d.events(&s.kinks[v], Marker::Dot, 7.0, kink());
        for (cut, prefix) in s.slices[v].iter().zip(["BEFORE", "AFTER"]) {
            let length = position_of(cut[1]) - position_of(cut[0]);
            let dy = if time_of(cut[0]) < 0.0 { -1.0 } else { 1.0 };
            d.bar(
                *cut,
                &format!("{prefix}: {length:.3} L0"),
                dy,
                palette::ink(),
            );
        }
        let label = if v == 0 {
            "SIMULTANEOUS REVERSAL"
        } else {
            "SAME LINE, BOOSTED"
        };
        d.text(s.kinks[v][1] * 0.5 + event(0.16, 0.35), label, kink());
        d.now(
            &end_line(&s.worldlines[v], 0),
            &end_line(&s.worldlines[v], 1),
        );
        d.frame();
        d.title(heading);
    }
    let end = time_of(s.train[s.train.len() - 1][0]);
    let mut d = Diagram::new(c, rect(2), [-0.16, 2.7], [-0.38, end as f32 + 0.30], r);
    d.rod(&s.train, 2.4);
    for step in &s.steps {
        d.dashed(step, kink(), 1.3);
        d.events(step, Marker::Dot, 5.0, kink());
    }
    let length = position_of(s.train_slice[1]) - position_of(s.train_slice[0]);
    d.bar(
        s.train_slice,
        &format!("AFTER: {length:.3} L0"),
        1.0,
        palette::ink(),
    );
    d.text(event(end - 0.05, 0.62), "ONE FIXED MAP, REPEATED", kink());
    d.now(&end_line(&s.train, 0), &end_line(&s.train, 1));
    d.frame();
    d.title("10 IMPULSES: 0 -> 0.8C");
    caption(
        c,
        "STRAIN-PRESERVING IMPULSES: START WITH A SYMMETRIC REVERSAL",
        "R(1,1) BY GAX::ALGEBRA!: BOOSTS ARE OPEN SANDWICHES; THE KINKS SIT ON THE BISECTOR",
    );
}

/// The barn between its door worldlines, and the door closures.
fn barn(d: &mut Diagram, doors: &[[V; 2]], closure: &[V; 2]) {
    let (l, r) = (end_line(doors, 0), end_line(doors, 1));
    let mut poly = l.clone();
    poly.extend(r.iter().rev());
    d.ax.fill(d.c, &poly, door(), 0.08);
    d.ax.dashed(d.c, &l, 1.2, 4.0, door(), 0.8);
    d.ax.dashed(d.c, &r, 1.2, 4.0, door(), 0.8);
    d.events(closure, Marker::Square, 7.0, door());
}

fn draw_ladder(c: &mut Canvas, r: f32) {
    let (_, s, _) = scenes();
    let (w, h) = (c.width as f32, c.height as f32);
    let rect = |i: usize| {
        [
            w * i as f32 / 4.0 + 34.0,
            h * 0.2,
            w * (i + 1) as f32 / 4.0 - 6.0,
            h - 60.0,
        ]
    };

    // 1. In the barn frame it fits at closure.
    let mut d = Diagram::new(c, rect(0), [-0.46, 1.32], [-0.6, 0.65], r);
    barn(&mut d, &s.barn_doors, &s.closure);
    d.rod(&s.barn_ladder, 2.2);
    d.bar(s.incoming, "LADDER 0.60", 1.0, palette::ink());
    d.text(event(-0.45, 0.4), "DOORS CLOSE TOGETHER", door());
    d.now(&end_line(&s.barn_ladder, 0), &end_line(&s.barn_ladder, 1));
    d.frame();
    d.title("1. BARN FRAME: IT FITS");

    // 2. The same events in the incoming ladder frame.
    let mut d = Diagram::new(c, rect(1), [-0.31, 2.32], [-1.62, 0.30], r);
    barn(&mut d, &s.moving_doors, &s.moving_closure);
    d.rod(&s.moving_ladder, 2.2);
    d.bar(s.ladder_cut, "LADDER 1.00", 1.0, palette::ink());
    d.bar(s.barn_cut, "BARN 0.48", 1.0, door());
    d.text(event(-1.25, 1.5), "EXIT CLOSES FIRST", door());
    d.now(
        &end_line(&s.moving_ladder, 0),
        &end_line(&s.moving_ladder, 1),
    );
    d.frame();
    d.title("2. LADDER FRAME: NOT TOGETHER");

    // 3. Staggered braking on the bisector: no proper-length mismatch, and it no longer fits.
    let mut d = Diagram::new(c, rect(2), [-0.44, 1.44], [-0.48, 2.12], r);
    barn(&mut d, &s.doors, &s.closure);
    d.rod(&s.stop_ladder, 2.2);
    d.dashed(&s.stop_kinks, kink(), 2.0);
    d.events(&s.stop_kinks, Marker::Dot, 7.0, kink());
    d.bar(s.incoming, "0.60 AT CLOSURE", -1.0, palette::ink());
    d.bar(s.stopped, "AT REST 1.00", 1.0, palette::ink());
    d.events(&[s.contact], Marker::Star, 11.0, contact());
    d.text(event(1.85, 0.5), "TOO LONG FOR THE BARN", palette::ink());
    d.now(&end_line(&s.stop_ladder, 0), &end_line(&s.stop_ladder, 1));
    d.frame();
    d.title("3. STRAIN-PRESERVING STOP");

    // 4. Stopping inside leaves compression, which rings about the relaxed length.
    let mut d = Diagram::new(c, rect(3), [-0.44, 1.44], [-0.48, 2.12], r);
    barn(&mut d, &s.doors, &s.closure);
    d.rod(&s.ring_incoming, 2.2);
    d.rod(&s.ring, 2.2);
    d.ax.dashed(d.c, &end_line(&s.relaxed, 0), 1.0, 2.0, rear(), 0.5);
    d.ax.dashed(d.c, &end_line(&s.relaxed, 1), 1.0, 2.0, front(), 0.5);
    d.dashed(&s.incoming, elastic(), 2.2);
    d.events(&s.incoming, Marker::Dot, 7.0, elastic());
    d.bar(s.incoming, "COMPRESSED 0.60", -1.0, elastic());
    d.events(&s.contacts, Marker::Star, 11.0, contact());
    d.bar(s.relaxed_cut, "RELAXED 1.00", 1.0, palette::ink());
    d.text(event(1.05, 0.4), "ELASTIC RINGING", elastic());
    let mut left = end_line(&s.ring_incoming, 0);
    left.extend(end_line(&s.ring, 0));
    let mut right = end_line(&s.ring_incoming, 1);
    right.extend(end_line(&s.ring, 1));
    d.now(&left, &right);
    d.frame();
    d.title("4. STOP INSIDE: RINGING");
    caption(
        c,
        "THE LADDER PARADOX: FITTING IN MOTION, STOPPING IN A SHORTER BARN",
        "LADDER L0 = 1, BARN 0.8 L0, INCOMING 0.8C: ONE OPEN SANDWICH CHANGES THE FRAME",
    );
}

fn draw_spaceships(c: &mut Canvas, r: f32) {
    let (_, _, s) = scenes();
    let (w, h) = (c.width as f32, c.height as f32);
    let end = time_of(s.tracks[0][s.tracks[0].len() - 1][0]);
    let titles = ["STRAIN-PRESERVING TRAIN", "BELL: IDENTICAL CLOCK PROGRAMS"];
    let notes = ["FRONT IMPULSES ON THE BISECTORS", "MATCHING CLOCK READINGS"];
    let colours = [kink(), elastic()];
    for k in 0..2 {
        let rect = [
            w * 0.36 * k as f32 + 40.0,
            h * 0.2,
            w * 0.36 * (k + 1) as f32 - 8.0,
            h - 60.0,
        ];
        let mut d = Diagram::new(c, rect, [-0.20, 3.05], [-0.40, end as f32 + 0.18], r);
        d.rod(&s.tracks[k], 2.4);
        for step in &s.schedules[k] {
            d.dashed(step, colours[k], 1.3);
            d.events(step, Marker::Dot, 5.0, colours[k]);
        }
        for (cut, label) in s.slices[k].iter().zip(["INITIALLY", "FINAL LAB GAP"]) {
            let gap = position_of(cut[1]) - position_of(cut[0]);
            let dy = if time_of(cut[0]) < 0.0 { -1.0 } else { 1.0 };
            d.bar(*cut, &format!("{label}: {gap:.2} L0"), dy, palette::ink());
        }
        d.text(event(end - 0.1, 1.4), notes[k], colours[k]);
        d.now(&end_line(&s.tracks[k], 0), &end_line(&s.tracks[k], 1));
        d.frame();
        d.title(titles[k]);
    }

    // The rope each schedule demands, in the final shared rest frame, fading in as the
    // diagrams complete.
    let alpha = ((r - 0.6) / 0.4).clamp(0.0, 1.0);
    let ax = fit(
        [w * 0.73, h * 0.2, w - 10.0, h - 60.0],
        [-0.22, 2.05],
        [-0.35, 3.03],
    );
    let size = 10.0;
    title(c, &ax, "IN THE FINAL REST FRAME", palette::ink());
    let gaps = s.final_events.map(|e| position_of(e[1]));
    let rows = [
        (
            gaps[0],
            2.05,
            kink(),
            "STRAIN-PRESERVING",
            "1.00 L0, NO EXTENSION".to_string(),
        ),
        (
            gaps[1],
            0.82,
            elastic(),
            "SYNCHRONIZED CLOCKS",
            format!(
                "{:.2} L0, {:.1}% EXTENSION",
                gaps[1],
                100.0 * (gaps[1] - 1.0)
            ),
        ),
    ];
    // Each rope a row: its rear ship at zero, its front ship at the gap, the relaxed length
    // dashed across both.
    let relaxed = [Point::xy(1.0, 0.65), Point::xy(1.0, 2.25)];
    ax.dashed(c, &relaxed, 1.0, 3.0, door(), alpha);
    for (gap, y, colour, head, label) in rows {
        let (rear_ship, front_ship) = (Point::xy(0.0, y), Point::xy(gap, y));
        let (above, below) = (Point::direction(0.0, 0.36), Point::direction(0.0, -0.3));
        ax.text(c, rear_ship + above, head, size, colour, Align::Left);
        ax.line(c, rear_ship, front_ship, 3.0, colour, alpha);
        for (at, ship) in [(rear_ship, rear()), (front_ship, front())] {
            let [px, py] = ax.px(at);
            c.fill(
                &[[px + 9.0, py], [px - 6.0, py - 7.0], [px - 6.0, py + 7.0]],
                ship,
                alpha,
            );
        }
        // The label under the rope's middle.
        let under = (rear_ship + front_ship).gp(0.5) + below;
        ax.text(c, under, &label, size, colour, Align::Center);
    }
    ax.text(
        c,
        Point::xy(0.0, 2.85),
        "SAME FINAL SPEED: 0.8C",
        size,
        door(),
        Align::Left,
    );
    ax.text(
        c,
        Point::xy(0.0, 0.1),
        "THE ROPE MUST STRETCH",
        size,
        elastic(),
        Align::Left,
    );
    caption(
        c,
        "BELL'S SPACESHIPS: THE TIMING ACROSS THE ROPE MATTERS",
        "10 MATCHING IMPULSES, 0 -> 0.8C, THE SAME REAR HISTORY IN BOTH",
    );
}

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let t = t.rem_euclid(3.0 * SCENE);
    let (scene, u) = ((t / SCENE) as usize, (t % SCENE) / SCENE);
    let r = reveal(u);
    match scene {
        0 => draw_impulse(c, r),
        1 => draw_ladder(c, r),
        _ => draw_spaceships(c, r),
    }
    // The legend.
    let (w, h) = (c.width as f32, c.height as f32);
    let size = (h / 50.0).clamp(7.0, 11.0);
    let entries = [
        ("REAR END", rear()),
        ("FRONT END", front()),
        ("IMPULSES", kink()),
        ("NOW", palette::yellow()),
    ];
    for (i, (label, colour)) in entries.iter().enumerate() {
        let x = w * (0.3 + 0.13 * i as f32);
        c.line(
            [x, h - size * 1.4],
            [x + size * 2.0, h - size * 1.4],
            2.5,
            *colour,
            1.0,
        );
        c.text(
            label,
            x + size * 2.6,
            h - size,
            size,
            palette::ink(),
            Align::Left,
        );
    }
}

fn main() {
    run(
        Anim::new("relativistic impulse", 3.0 * SCENE).size(960, 540),
        draw,
    );
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)]
mod tests {
    use super::impulse::*;
    use gax::ApproxEq;

    const ATOL: f64 = 1e-9;

    fn velocity(d: V) -> f64 {
        position_of(d) / time_of(d)
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= ATOL
    }

    #[test]
    fn velocity_step_preserves_proper_spacing_and_requested_directions() {
        let (before, after) = (0.6f64, -0.75f64);
        let (events, directions) = velocity_step(before, after);
        assert!(events[0].max_abs_diff(&V::zero()) <= ATOL);
        for (d, r) in directions.iter().zip([before, after]) {
            assert!(close(d.norm_squared(), 1.0));
            assert!(close(velocity(*d), r.tanh()));
        }
        let separation = events[1] - events[0];
        assert!(separation.norm_squared() < 0.0);
        for d in directions {
            assert!(close((d ^ separation).etx(), 1.0));
        }
        assert!(close(
            ((directions[0] + directions[1]) | separation).s(),
            0.0
        ));
    }

    #[test]
    fn impulse_train_forms_continuous_worldlines_with_fixed_proper_spacing() {
        for (final_velocity, dt) in [(0.8f64, 0.1), (-0.8, 0.2)] {
            let (events, directions) = small_impulses(final_velocity.atanh(), 10, dt);
            assert_eq!((events.len(), directions.len()), (10, 11));
            assert!(close(velocity(directions[0]), 0.0));
            assert!(close(velocity(directions[10]), final_velocity));
            for (s, e) in events.iter().enumerate() {
                let separation = e[1] - e[0];
                for d in [directions[s], directions[s + 1]] {
                    assert!(close((d ^ separation).etx(), 1.0));
                }
            }
            for s in 0..9 {
                for end in 0..2 {
                    let advance = events[s + 1][end] - events[s][end];
                    assert!(time_of(advance) > 0.0);
                    assert!(close((advance ^ directions[s + 1]).etx(), 0.0));
                }
                assert!(close((events[s + 1][0] - events[s][0]).norm(), dt));
            }
            // Compare the ends at one common observer time after their final kinks.
            let last = time_of(events[9][0]).max(time_of(events[9][1])) + 0.5;
            let ends = worldline_events(last, &events, &directions);
            let length = position_of(ends[1]) - position_of(ends[0]);
            assert!(close(
                length,
                (1.0 - final_velocity * final_velocity).sqrt()
            ));
        }
    }

    #[test]
    fn one_impulse_matches_a_single_velocity_step() {
        for rapidity in [-1.1, 1.1] {
            let (train, directions) = small_impulses(rapidity, 1, 0.2);
            let (events, step_directions) = velocity_step(0.0, rapidity);
            for end in 0..2 {
                assert!(train[0][end].max_abs_diff(&events[end]) <= ATOL);
            }
            for (a, b) in directions.iter().zip(step_directions) {
                assert!(close(velocity(*a), velocity(b)));
            }
        }
    }

    /// numga's checks inside `impulse()`.
    #[test]
    fn impulse_scenario_checks() {
        let s = impulse();
        // The kinks are simultaneous for the symmetric observer, whose time axis bisects the
        // motions.
        assert!(close(time_of(s.bisector), 1.0));
        for k in s.kinks[0] {
            assert!(close((k | s.bisector).s(), 0.0));
        }
        // Boosted, the same drawing is a step from rest to the relativistic sum of the speeds,
        // and the kinks are not simultaneous.
        let boosted = s.directions[1];
        assert!(close(velocity(boosted[0]), 0.0));
        assert!(close(velocity(boosted[1]), 2.0 * 0.5 / (1.0 + 0.25)));
        assert!(close(time_of(s.kinks[1][0]), 0.0) && close(time_of(s.kinks[1][1]), 0.5));
    }

    /// numga's checks inside `ladder()`.
    #[test]
    fn ladder_scenario_checks() {
        let s = ladder();
        let rapidity = 0.8f64.atanh();
        // In its own frame the incoming ladder is at rest with its full proper length, and the
        // exit closes first.
        assert!(close(position_of(s.moving_direction), 0.0));
        let ends = s.moving_events;
        assert!(close(position_of(ends[1] - ends[0]), 1.0));
        assert!(close(time_of(s.moving_closure[0]), 0.0));
        assert!(close(time_of(s.moving_closure[1]), -rapidity.sinh() * 0.8));
        // The strain-preserving stop joins kinks perpendicular to the rapidity bisector; the
        // front stops beyond the exit, and at rest the ladder has its relaxed length.
        let [d0, d1] = s.stop_directions;
        assert!(close(
            ((d0 + d1) | (s.stop_kinks[1] - s.stop_kinks[0])).s(),
            0.0
        ));
        assert!(close(position_of(s.stop_kinks[0]), 0.1));
        assert!(close(position_of(s.stop_kinks[1]), 1.1));
        assert!(close(position_of(s.stopped[1] - s.stopped[0]), 1.0));
    }

    /// numga's checks inside `spaceships()`.
    #[test]
    fn spaceships_scenario_checks() {
        let s = spaceships();
        let (dt, rapidity) = (0.1, 0.8f64.atanh());
        // Proper time between kinks: the rear keeps one evenly spaced program in both
        // schedules, and Bell's ships run it on clocks started together at lab time zero.
        for (k, schedule) in s.schedules.iter().enumerate() {
            for w in schedule.windows(2) {
                for end in 0..2 {
                    if k == 1 || end == 0 {
                        assert!(close((w[1][end] - w[0][end]).norm(), dt));
                    }
                }
            }
        }
        let bell = &s.schedules[1];
        assert!(close(time_of(bell[0][0]), 0.0) && close(time_of(bell[0][1]), 0.0));
        // Both ships end at rest in the final frame: the strain-preserving rope keeps its
        // length, Bell's is stretched by gamma. In the lab it is the other way round.
        let last = s.final_frame.of(s.directions[s.directions.len() - 1]);
        assert!(close(position_of(last), 0.0));
        assert!(close(position_of(s.final_events[0][1]), 1.0));
        assert!(close(position_of(s.final_events[1][1]), rapidity.cosh()));
        let gap = |cut: [V; 2]| position_of(cut[1] - cut[0]);
        assert!(close(gap(s.slices[0][1]), 1.0 / rapidity.cosh()));
        assert!(close(gap(s.slices[1][1]), 1.0));
        for i in 0..300 {
            let time = -0.3 + 3.3 * i as f64 / 299.0;
            assert!(close(gap(worldline_events(time, bell, &s.directions)), 1.0));
        }
    }

    /// A frame of each scene.
    #[test]
    fn a_frame_draws() {
        for t in [0.5, super::SCENE + 3.0, 2.0 * super::SCENE + 6.5] {
            gax_numga_examples::app::assert_draws(super::draw, t);
        }
    }
}
