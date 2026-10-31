//! numga's `thermodynamics/generic`: a mass on a spring with a damper that warms a thermal store,
//! reversible motion and irreversible friction in one state, as a GENERIC system, in a
//! metric-free algebra (`q0p0s0`, declared here with `gax::algebra!`).
//!
//! A state holds the mass's position and momentum and the store's entropy, an intrinsic bivector
//! (numga's antivector) spanned by `q.dual()`, `p.dual()` and `s.dual()`, each read back by the
//! regressive product, `q & state`. The energy's gradient, the spring's force, the mass's
//! velocity and the store's temperature, is an extrinsic vector that pairs with a change of state
//! to the change of energy. Two maps from gradients to changes of state drive the motion. The
//! Poisson map, skew, sends the energy's gradient to the swing of the oscillator and the
//! entropy's gradient, `s`, to nothing. The friction mobility, a positive dyad along the
//! direction trading momentum for the store's heat, sends the entropy's gradient to the
//! irreversible motion and the energy's gradient to nothing. So the energy holds, the entropy
//! never falls, and the oscillation settles with all its energy in the store. Steps are the
//! classical Runge-Kutta method on the sum of the two motions.
//!
//! The animation: the oscillator with its damper coloured as it warms; the phase portrait with
//! the level lines of the store's entropy at the start's total energy (ellipses of equal
//! mechanical energy, each marked with its entropy) and the two rates as arrows, the reversible
//! along them and the dissipative across; the entropy against time, rising in steps; and the
//! entropy against the mechanical energy left, along the fixed total energy.

gax::algebra! {
    algebra oscillator "Position, momentum and the store's entropy, `q0p0s0`: every generator squares to zero.";
    basis eq = 0, ep = 0, es = 0;
    kind Scalar = [1];
    kind Vector = [eq, ep, es];
    kind Bivector = [eps, esq, eqp];
    kind Pseudoscalar = [eqps];
}

use charts::{BOLD, Diagram, Family, MEDIUM};
use gax_numga_examples::points::Point3;
use gax_numga_examples::scene3::panel3;
use gax_numga_examples::shared::thermodynamics_charts as charts;
use gax_numga_examples::{Anim, Canvas, Point2, Rect, caption, run_with};

mod generic {
    use super::oscillator::{Scalar, Vector, intrinsic};
    use gax::Real;
    use gax_numga_examples::shared::mechanics_lie::rk4;

    /// A state: position, momentum and entropy, an intrinsic bivector (numga's `Antivector-`).
    pub type State<T = f64> = intrinsic::Bivector<(), T>;
    /// A gradient, a reading of changes of state (numga's `Vector+`).
    pub type Gradient = Vector<(), f64>;
    /// `State <- Gradient`.
    pub type Map = intrinsic::Bivector<(Vector,), f64>;
    /// `Scalar <- State`.
    pub type Reading = Scalar<(intrinsic::Bivector,), f64>;

    /// The gradients of the position, the momentum and the entropy: `q`, `p`, `s`.
    pub fn q() -> Gradient {
        Vector::new(1.0, 0.0, 0.0)
    }
    pub fn p() -> Gradient {
        Vector::new(0.0, 1.0, 0.0)
    }
    pub fn s() -> Gradient {
        Vector::new(0.0, 0.0, 1.0)
    }
    /// The state at a position, momentum and entropy.
    pub fn state(position: f64, momentum: f64, entropy: f64) -> State {
        q().dual() * position + p().dual() * momentum + s().dual() * entropy
    }
    /// The readings of a state.
    pub fn position() -> Reading {
        q() & State::slot()
    }
    pub fn momentum() -> Reading {
        p() & State::slot()
    }
    pub fn entropy() -> Reading {
        s() & State::slot()
    }

    /// The canonical Poisson map: the change of position follows the momentum's slope, the
    /// change of momentum minus the position's.
    pub fn poisson_map() -> Map {
        q().dual() * (Vector::slot() & p().dual()) - p().dual() * (Vector::slot() & q().dual())
    }
    /// The entropy's gradient: the store's entropy, the one reading of a state it is.
    pub fn entropy_gradient() -> Gradient {
        s()
    }

    /// A gradient's reading of a state with coefficients of any type (jets included).
    fn read<T: Real>(g: Gradient, state: State<T>) -> T {
        (g.map_coefs(T::from_f64) & state).s()
    }

    pub fn temperature<T: Real>(state: State<T>, capacity: f64, reference: f64) -> T {
        T::from_f64(reference) * (read(s(), state) / T::from_f64(capacity)).exp()
    }
    pub fn kinetic_energy<T: Real>(state: State<T>, mass: f64) -> T {
        let momentum = read(p(), state);
        momentum * momentum / T::from_f64(2.0 * mass)
    }
    pub fn potential_energy<T: Real>(state: State<T>, stiffness: f64) -> T {
        let position = read(q(), state);
        T::from_f64(stiffness) * position * position / T::from_f64(2.0)
    }
    pub fn internal_energy<T: Real>(state: State<T>, capacity: f64, reference: f64) -> T {
        T::from_f64(capacity) * temperature(state, capacity, reference)
    }

    /// The oscillator's physical constants.
    #[derive(Clone, Copy, Debug)]
    pub struct Oscillator {
        pub mass: f64,
        pub stiffness: f64,
        pub friction: f64,
        pub capacity: f64,
        pub reference: f64,
    }

    impl Oscillator {
        pub fn total_energy<T: Real>(&self, state: State<T>) -> T {
            kinetic_energy(state, self.mass)
                + potential_energy(state, self.stiffness)
                + internal_energy(state, self.capacity, self.reference)
        }

        /// The energy's slopes along position, momentum and entropy: the spring's force, the
        /// velocity, the temperature.
        pub fn energy_gradient(&self, state: State) -> Gradient {
            q() * (self.stiffness * position().of(state).s())
                + p() * (momentum().of(state).s() / self.mass)
                + s() * temperature(state, self.capacity, self.reference)
        }

        /// A positive dyad along the direction in which the momentum lost is the store's heat
        /// gained.
        pub fn friction_mobility(&self, state: State) -> Map {
            let store = temperature(state, self.capacity, self.reference);
            let velocity = momentum().of(state).s() / self.mass;
            let direction = s().dual() * (velocity / store) - p().dual();
            (direction * (Vector::slot() & direction)).gp(self.friction * store)
        }

        /// The reversible motion, the Poisson map of the energy's gradient, and the irreversible,
        /// the friction mobility of the entropy's.
        pub fn rates(&self, state: State) -> (State, State) {
            let reversible = poisson_map().of(self.energy_gradient(state));
            let irreversible = self.friction_mobility(state).of(entropy_gradient());
            (reversible, irreversible)
        }

        /// A step of the combined motion by the classical fourth-order Runge-Kutta method.
        pub fn advance(&self, state: State, dt: f64) -> State {
            let rate = |x: State| {
                let (reversible, irreversible) = self.rates(x);
                reversible + irreversible
            };
            rk4(rate, state, dt)
        }
    }

    // --- the scenario: a mass released on a spring --------------------------------------

    pub const OSCILLATOR: Oscillator = Oscillator {
        mass: 1.0,
        stiffness: 4.0,
        friction: 0.35,
        capacity: 2.0,
        reference: 1.0,
    };
    pub const DISPLACEMENT: f64 = 2.0;

    /// The mass displaced and at rest, the store at the reference temperature.
    pub fn start() -> State {
        state(DISPLACEMENT, 0.0, 0.0)
    }

    /// The kinetic, potential and internal energies.
    pub fn energies_of(state: State) -> [f64; 3] {
        let o = OSCILLATOR;
        [
            kinetic_energy(state, o.mass),
            potential_energy(state, o.stiffness),
            internal_energy(state, o.capacity, o.reference),
        ]
    }

    /// The store's entropy when the total energy is the start's: at fixed energy, the entropy as
    /// a function of the position and momentum, all the energy they leave over held by the
    /// store (NaN where they leave none).
    pub fn surface_entropy_of(state: State) -> f64 {
        let o = OSCILLATOR;
        let [kinetic, potential, _] = energies_of(state);
        let internal = o.total_energy(start()) - kinetic - potential;
        o.capacity * (internal / (o.capacity * o.reference)).ln()
    }

    /// The state at each frame, so many steps of the combined motion apart.
    pub fn release(frames: usize, steps_per_frame: usize, dt: f64) -> Vec<State> {
        let mut state = start();
        (0..frames)
            .map(|_| {
                let now = state;
                for _ in 0..steps_per_frame {
                    state = OSCILLATOR.advance(state, dt);
                }
                now
            })
            .collect()
    }
}

use generic::*;

/// numga's frames: 150, two steps of 0.04 apart, shown 40 ms each.
const FRAMES: usize = 150;
const STEPS_PER_FRAME: usize = 2;
const DT: f64 = 0.04;
const FRAME_SECONDS: f32 = 0.04;
const HOLD: f32 = 1.2;

/// The charts' ranges: position and momentum, time and entropy, mechanical energy.
const POSITION_RANGE: [f64; 2] = [-2.6, 2.6];
const MOMENTUM_RANGE: [f64; 2] = [-4.6, 4.6];
const TIME_RANGE: [f64; 2] = [0.0, 12.5];
const ENTROPY_RANGE: [f64; 2] = [0.0, 3.6];
const MECHANICAL_RANGE: [f64; 2] = [0.0, 8.6];
/// The entropies of the level lines.
const LEVELS: [f64; 7] = [0.0, 1.0, 1.8, 2.3, 2.7, 3.0, 3.15];
/// How far ahead the rates' arrows look, in seconds, and how long they are at most, in numga
/// pixels; the decades of the damper's remaining warming its colour spans.
const LOOK_AHEAD: f64 = 0.25;
const ARROW_MOST: f32 = 44.0;
const DECADES: f64 = 2.0;

/// The rates' colours: reversible blue, dissipative red.
fn reversible_light() -> gax_colour::Light {
    gax_colour::hex(0x7088ff, 1.7)
}
fn dissipative_light() -> gax_colour::Light {
    gax_colour::hex(0xff5040, 1.7)
}

/// What is computed once: the release, the level lines with their labels' places, and the
/// entropy's curve against the mechanical energy.
struct Scene {
    states: Vec<State>,
    level_lines: Vec<Vec<Point2>>,
    level_labels: Vec<(Point2, f64)>,
    curve: Vec<Point2>,
}

/// A state on the phase portrait: its position and momentum, read by pairing.
fn phase_point(state: State) -> Point2 {
    charts::at(position().of(state).s(), momentum().of(state).s())
}

fn scene() -> Scene {
    let states = release(FRAMES, STEPS_PER_FRAME, DT);
    let sample = |range: [f64; 2], n: usize| -> Vec<f64> {
        (0..n)
            .map(|k| range[0] + (range[1] - range[0]) * k as f64 / (n - 1) as f64)
            .collect()
    };
    let (positions, momenta) = (sample(POSITION_RANGE, 121), sample(MOMENTUM_RANGE, 121));
    let values: Vec<f64> = momenta
        .iter()
        .flat_map(|&m| {
            positions
                .iter()
                .map(move |&x| surface_entropy_of(state(x, m, 0.0)))
        })
        .map(|v| if v.is_finite() { v } else { -10.0 })
        .collect();
    let mut level_lines = Vec::new();
    let mut level_labels = Vec::new();
    for level in LEVELS {
        for line in charts::level_lines(&positions, &momenta, &values, level) {
            // Each level's entropy at its upper right: its point furthest along the chart's
            // diagonal, the reading of the plane x / 5.2 + y / 9.2.
            let diagonal = |p: &Point2| {
                let [x, y] = p.to_euclidean();
                x / 5.2 + y / 9.2
            };
            let corner = line
                .iter()
                .copied()
                .max_by(|a, b| diagonal(a).total_cmp(&diagonal(b)));
            if let Some(corner) = corner
                && f64::from(corner.to_euclidean()[1]) < MOMENTUM_RANGE[1] - 0.3
            {
                level_labels.push((corner, level));
            }
            level_lines.push(line);
        }
    }
    // The entropy at fixed energy against the mechanical energy, along the positions at rest.
    let curve = sample([0.0, 2.0], 200)
        .into_iter()
        .map(|x| {
            let rest = state(x, 0.0, 0.0);
            let [kinetic, potential, _] = energies_of(rest);
            charts::at(kinetic + potential, surface_entropy_of(rest))
        })
        .collect();
    Scene {
        states,
        level_lines,
        level_labels,
        curve,
    }
}

/// The mass on its floor, held to the wall by a spring and a damper, the damper coloured by how
/// warm it is.
fn apparatus(c: &mut Canvas, rect: Rect, position: f64, warmth: f64) {
    let x = position as f32;
    let corners: Vec<Point3> = [-2.95, 2.5]
        .into_iter()
        .flat_map(|x| [-0.65, 0.65].map(|y| (x, y)))
        .flat_map(|(x, y)| [0.1, 1.4].map(|z| Point3::xyz(x, y, z)))
        .collect();
    let square = Rect::around(
        rect.centre(),
        Point2::direction(rect.width() * 0.5, rect.width() * 0.4),
    );
    let cam = charts::apparatus_camera(square, &corners, -12.0, 18.0);
    panel3(c, square, cam, |s| {
        let grey = charts::GREY;
        charts::cuboid(s, Point3::xyz(0.0, 0.0, 0.18), [2.9, 0.62, 0.05], grey, 1.0);
        charts::cuboid(
            s,
            Point3::xyz(-2.75, 0.0, 0.78),
            [0.08, 0.58, 0.55],
            grey,
            1.0,
        );
        let body = Point3::xyz(x, 0.0, 0.65);
        charts::cuboid(s, body, [0.3, 0.4, 0.38], charts::BLUE.faded(0.6), 1.0);
        let damper = charts::thermal(warmth);
        charts::cuboid(
            s,
            Point3::xyz(-2.0, 0.0, 0.4),
            [0.5, 0.17, 0.15],
            damper,
            1.0,
        );
        // The rod from the damper to the mass's hook.
        let (from, to) = (-1.5f32, x - 0.3);
        let middle = Point3::xyz(0.5 * (from + to), 0.0, 0.4);
        let rod = [0.5 * (to - from).abs(), 0.045, 0.045];
        charts::cuboid(s, middle, rod, grey, 1.0);
        // The spring's coil: points along its axis from the wall to the hook, each with the
        // coil's radius turned about that axis.
        let (fixed, hook) = (
            Point3::xyz(-2.67, 0.0, 0.98),
            Point3::xyz(x - 0.3, 0.0, 0.98),
        );
        let axis = Point3::direction(1.0, 0.0, 0.0);
        let coil: Vec<Point3> = (0..160)
            .map(|k| {
                let f = k as f32 / 159.0;
                let turn =
                    gax::pga3d::Motor::rotation_about(axis, 12.0 * core::f32::consts::PI * f);
                fixed + (hook - fixed).gp(f) + (turn >> Point3::direction(0.0, 0.0, 0.13))
            })
            .collect();
        s.polyline(&coil, 1.4, charts::INK);
    });
}

/// The phase portrait: the spiral so far over the entropy's level lines, with the rates' arrows
/// at its end, from the state toward where each rate alone would carry it a look ahead on.
fn phase(c: &mut Canvas, rect: Rect, sc: &Scene, frame: usize) {
    let d = Diagram::new(rect, POSITION_RANGE, MOMENTUM_RANGE, [0.0, 0.0], ["Q", "P"]);
    d.family(c, &sc.level_lines, Family::Entropy);
    let entropy = Family::Entropy.light();
    for &(corner, level) in &sc.level_labels {
        d.label(
            c,
            corner,
            &format!("{level}"),
            entropy,
            charts::TICK,
            charts::LB,
            [2.0, 2.0],
        );
    }
    let top_left = |down: f64| charts::at(POSITION_RANGE[0], MOMENTUM_RANGE[1] - down);
    let text = charts::TEXT;
    d.label(
        c,
        top_left(0.0),
        "REVERSIBLE",
        reversible_light(),
        text,
        charts::LT,
        [4.0, 0.0],
    );
    d.label(
        c,
        top_left(0.8),
        "DISSIPATIVE",
        dissipative_light(),
        text,
        charts::LT,
        [4.0, 0.0],
    );
    d.float(
        c,
        &charts::along(&sc.level_lines, 6),
        "S = CONST",
        entropy,
        charts::LT,
    );
    let path: Vec<Point2> = sc.states[..=frame]
        .iter()
        .map(|s| phase_point(*s))
        .collect();
    d.polyline(c, &path, BOLD, charts::ORANGE);
    let state = sc.states[frame];
    let now = phase_point(state);
    let (reversible, irreversible) = OSCILLATOR.rates(state);
    for (rate, light) in [
        (reversible, reversible_light()),
        (irreversible, dissipative_light()),
    ] {
        let toward = phase_point(state + rate * LOOK_AHEAD);
        // The arrow's length on the canvas, in numga pixels.
        let length = d.px(toward).distance(d.px(now)) / d.unit;
        if length > 3.0 {
            d.arrow(c, now, toward, length.min(ARROW_MOST), MEDIUM, light);
        }
    }
    d.dots(c, &[now], charts::ORANGE);
    d.axes(c);
    d.note(c, "PHASE PLANE: S LEVELS AT FIXED E");
}

/// The entropy against time, rising in steps.
fn rising(c: &mut Canvas, rect: Rect, sc: &Scene, frame: usize) {
    let d = Diagram::new(rect, TIME_RANGE, ENTROPY_RANGE, [0.0, 0.0], ["TIME", "S"]);
    let frame_time = STEPS_PER_FRAME as f64 * DT;
    let path: Vec<Point2> = (sc.states[..=frame].iter().enumerate())
        .map(|(k, s)| charts::at(k as f64 * frame_time, entropy().of(*s).s()))
        .collect();
    d.polyline(c, &path, BOLD, charts::ORANGE);
    d.dots(c, &path[frame..], charts::ORANGE);
    d.axes(c);
    d.note(c, "THE ENTROPY NEVER FALLS");
}

/// The entropy against the mechanical energy left, along the fixed total energy: the state
/// slides to the curve's top, where the store holds everything.
fn hill(c: &mut Canvas, rect: Rect, sc: &Scene, frame: usize) {
    let d = Diagram::new(
        rect,
        MECHANICAL_RANGE,
        ENTROPY_RANGE,
        [0.0, 0.0],
        ["E_MECH", "S"],
    );
    let entropy_light = Family::Entropy.light();
    d.polyline(c, &sc.curve, MEDIUM, entropy_light);
    let path: Vec<Point2> = sc.states[..=frame]
        .iter()
        .map(|s| {
            let [kinetic, potential, _] = energies_of(*s);
            charts::at(kinetic + potential, entropy().of(*s).s())
        })
        .collect();
    d.polyline(c, &path, BOLD, charts::ORANGE);
    d.dots(c, &path[frame..], charts::ORANGE);
    // The curve's name at the chart's right edge, level with the curve's end: the line through
    // that end along the mechanical energy, met with the edge.
    let end = charts::level_with(sc.curve[sc.curve.len() - 1], MECHANICAL_RANGE[1]);
    d.label(
        c,
        end,
        "E = CONST",
        entropy_light,
        charts::TEXT,
        charts::RB,
        [0.0, 4.0],
    );
    d.axes(c);
    d.note(c, "SLIDING UP THE FIXED ENERGY");
}

fn draw(c: &mut Canvas, sc: &Scene, t: f32) {
    gax_numga_examples::backdrop(c);
    let frame = ((t / FRAME_SECONDS) as usize).min(FRAMES - 1);
    let state = sc.states[frame];
    let panels = charts::row(c, &[0.85, 1.0, 1.0, 1.0]);
    // The share of its warming the damper still has to do, half as it is and half on its
    // logarithm.
    let start = energies_of(sc.states[0]);
    let total: f64 = start.iter().sum();
    let internal = energies_of(state)[2];
    let remaining = (total - internal).max(1e-300) / (total - start[2]);
    let warmth = 1.0 - 0.5 * remaining - 0.5 * (remaining.log10() / DECADES + 1.0).clamp(0.0, 1.0);
    apparatus(c, panels[0], position().of(state).s(), warmth);
    phase(c, panels[1], sc, frame);
    rising(c, panels[2], sc, frame);
    hill(c, panels[3], sc, frame);
    caption(
        c,
        "GENERIC: A DAMPED OSCILLATOR WARMS ITS DAMPER",
        "A SKEW POISSON MAP MOVES ALONG THE ENTROPY'S LEVELS, A POSITIVE FRICTION MAP UP THEM (Q0P0S0)",
    );
}

fn main() {
    run_with(
        Anim::new("generic", FRAME_SECONDS * FRAMES as f32 + HOLD).size(960, 540),
        scene,
        draw,
    );
}

#[cfg(test)]
mod tests {
    use super::generic::*;
    use gax::ApproxEq;

    const ROUND_OFF: f64 = 1e-12;

    /// The Poisson map ignores the entropy and is skew; the friction mobility ignores the energy
    /// and is symmetric and positive; together they conserve the energy and produce entropy at
    /// `friction v² / T`.
    #[test]
    fn degeneracies_symmetries_and_production() {
        let o = OSCILLATOR;
        let (positions, momenta, entropies) = ([1.5, -2.0, 0.5], [2.0, -1.0, 0.0], [0.0, 1.2, 2.5]);
        let alpha = q() * 2.0 + p() * -1.5 + s() * 0.7;
        let beta = q() * -0.5 + p() * 3.0 + s() * 1.2;
        let poisson = poisson_map();
        assert!(
            poisson
                .of(entropy_gradient())
                .abs_diff_eq(&State::zero(), ROUND_OFF)
        );
        let skew = (poisson.of(beta) & alpha).s() + (poisson.of(alpha) & beta).s();
        assert!(skew.abs() < ROUND_OFF);
        for k in 0..3 {
            let x = state(positions[k], momenta[k], entropies[k]);
            let energy = o.energy_gradient(x);
            let mobility = o.friction_mobility(x);
            assert!(mobility.of(energy).abs_diff_eq(&State::zero(), ROUND_OFF));
            let symmetric = (mobility.of(beta) & alpha).s() - (mobility.of(alpha) & beta).s();
            assert!(symmetric.abs() < ROUND_OFF);
            assert!((mobility.of(alpha) & alpha).s() >= -ROUND_OFF);
            let (reversible, irreversible) = o.rates(x);
            let rate = reversible + irreversible;
            assert!((energy & rate).s().abs() < ROUND_OFF);
            let temperature = o.reference * (entropies[k] / o.capacity).exp();
            let production = o.friction * (momenta[k] / o.mass).powi(2) / temperature;
            assert!(((entropy_gradient() & rate).s() - production).abs() < ROUND_OFF);
        }
    }

    /// The energy's gradient read against changes of state is the total energy's derivative.
    #[test]
    fn energy_gradient_is_the_energy_derivative() {
        let x = state(0.7, -1.3, 0.4);
        let reading = OSCILLATOR.energy_gradient(x) & State::slot();
        let (_, derivative) = gax::jet::gradient(|y| OSCILLATOR.total_energy(y), x);
        assert!(
            reading.abs_diff_eq(&derivative, ROUND_OFF),
            "{reading:?} {derivative:?}"
        );
    }

    /// Released from rest, the oscillator keeps its energy of 10 to 1e-4 over 150 frames, its
    /// store's entropy never falls, and nearly all the energy ends in the store.
    #[test]
    fn release_conserves_energy_and_heats_the_damper() {
        let states = release(150, 2, 0.04);
        for x in &states {
            assert!((OSCILLATOR.total_energy(*x) - 10.0).abs() < 1e-4);
        }
        for pair in states.windows(2) {
            assert!(entropy().of(pair[1]).s() - entropy().of(pair[0]).s() >= -1e-10);
        }
        assert!(energies_of(states[states.len() - 1])[2] > 9.8);
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws_with(super::draw, &super::scene(), 3.0);
    }
}
