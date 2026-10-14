//! numga's `mechanics/kepler`: close Kepler passages as the smooth motion of a scaled spinor
//! (VGA3D), the Kustaanheimo–Stiefel regularization in Hestenes's geometric form.
//!
//! The position is carried as a spinor `U`, an even element `Rotor` that need not be unit: its
//! sandwich `U x ~U` turns and scales the fixed reference direction `x` onto the position, and
//! `U ~U` is the radius, the distance from the attracting centre at the origin. With both spinors
//! left open, `(Rotor::slot() * x) * Rotor::slot().reverse()` cast to a `Vector` is the placement,
//! a map `Vector <- (Rotor, Rotor)`; the velocity is that sandwich differentiated in both slots,
//! over the radius. A spinor's extra freedom is a turn about `x` on the right, a gauge that moves
//! nothing.
//!
//! Let physical time advance by the radius, `dt = U ~U ds`. Then a bound inverse-square orbit of
//! energy `E` becomes the oscillator `U'' = (E / 2) U` in the regularized clock `s`: no
//! inverse-cube factor and no singularity at the centre. Its exact solution is a cosine and a
//! sine of the phase `ω s` with `ω² = -E / 2`, and integrating the radius over it gives the
//! physical clock in closed form; the clock is monotone, so bisection inverts it. One orbit is
//! half a spinor cycle: opposite spinors place `x` at the same position.
//!
//! Three orbits of one semimajor axis and period, eccentricities 0.6, 0.9 and 0.97, start at
//! apoapsis. Kick-drift-kick Verlet with equal steps in physical time and the same Verlet on the
//! spinor oscillator (its clock by Simpson's rule) both keep the angular momentum and converge
//! at second order, but the spinor clock spends its steps near periapsis and resolves the close
//! passage far better.
//!
//! The animation runs uniformly in the regularized clock over two revolutions of the most
//! eccentric orbit: top left the orbit and its spinor's scalar and `xy` parts (the body goes
//! round the orbit twice while its spinor closes once, the opposite spinor as a ring), top right
//! the errors of both schemes against the step count. Below, the three passages are revealed at
//! the body's physical time, each with its periapsis enlarged.

use gax::vga3d::{Rotor, Vector};

use gax_numga_examples::{
    Align, Anim, Axes, Canvas, Marker, Point2, Rect, backdrop, caption, colormap, from_above,
    palette, reach3, run,
};
use std::sync::OnceLock;

mod kepler {
    use super::*;
    use core::f64::consts::{PI, TAU};
    use gax_numga_examples::signal::phasor;

    pub type V = Vector<(), f64>;
    /// A position spinor: an even element of VGA3D, scaled so that `U ~U` is the radius.
    pub type Spinor = Rotor<(), f64>;
    /// The placement `U x ~U` with both spinors open: a vector from two spinors.
    pub type Placement = Vector<(Rotor, Rotor), f64>;

    /// The reference direction a spinor carries onto the position.
    pub const REFERENCE: V = Vector::new(1.0, 0.0, 0.0);

    /// The placement, the sandwich of the reference direction with both spinors open (its
    /// trivector part, which only the gauge rate reaches, cast away).
    pub fn placement() -> Placement {
        ((Rotor::slot() * REFERENCE) * Rotor::slot().reverse()).cast::<Vector>()
    }

    /// Physical time, position and velocity.
    #[derive(Clone, Copy, Debug)]
    pub struct State {
        pub time: f64,
        pub position: V,
        pub velocity: V,
    }

    impl State {
        /// The orbital energy per unit mass about a centre of strength `gravity`.
        pub fn energy(&self, gravity: f64) -> f64 {
            self.velocity.norm_squared() / 2.0 - gravity / self.position.norm()
        }

        /// The state of a position spinor and its rate in the regularized clock.
        pub fn from_spinor(spinor: Spinor, rate: Spinor, time: f64) -> State {
            let radius = spinor.norm_squared();
            // The sandwich differentiated in both slots; their trivector parts cancel.
            let open = placement();
            let velocity = (open.of(rate).of(spinor) + open.of(spinor).of(rate)) / radius;
            State {
                time,
                position: spinor >> REFERENCE,
                velocity,
            }
        }
    }

    /// A bound Kepler orbit lifted into a position spinor, its rate and the energy.
    #[derive(Clone, Copy, Debug)]
    pub struct BoundOrbit {
        pub spinor: Spinor,
        pub rate: Spinor,
        pub energy: f64,
    }

    impl BoundOrbit {
        /// Lift a physical state through a spinor that places the reference direction at its
        /// position. This rate gives the state's velocity without turning about the radius.
        pub fn lift(start: &State, spinor: Spinor, gravity: f64) -> BoundOrbit {
            let rate = (start.velocity * spinor) * REFERENCE / 2.0;
            BoundOrbit {
                spinor,
                rate,
                energy: start.energy(gravity),
            }
        }

        /// The oscillator's angular frequency in the regularized clock.
        #[allow(clippy::disallowed_methods)] // the oscillator's law ω² = -E / 2
        pub fn frequency(&self) -> f64 {
            (-self.energy / 2.0).sqrt()
        }

        /// The cosine and sine of the oscillator's phase at `parameter`.
        fn phase(&self, parameter: f64) -> (f64, f64) {
            let p = phasor(self.frequency() * parameter);
            (p.e20(), p.e01())
        }

        /// The exact position spinor at `parameter` in the regularized clock.
        pub fn spinor_at(&self, parameter: f64) -> Spinor {
            let (cosine, sine) = self.phase(parameter);
            self.spinor * cosine + self.rate * (sine / self.frequency())
        }

        /// The physical time at `parameter`: the radius integrated over the exact spinor. The
        /// clock slows smoothly near the centre.
        pub fn time_at(&self, parameter: f64) -> f64 {
            let w = self.frequency();
            let (cosine, sine) = self.phase(parameter);
            let initial_radius = self.spinor.norm_squared();
            let rate_radius = self.rate.norm_squared() / (w * w);
            let overlap = self.spinor.scalar_product(self.rate.reverse()).s() / w;
            // The sine of twice the phase is twice its sine and cosine.
            (initial_radius + rate_radius) * parameter / 2.0
                + (initial_radius - rate_radius) * sine * cosine / (2.0 * w)
                + overlap * sine * sine / w
        }

        /// The exact state at `parameter`, with its physical clock.
        pub fn sample(&self, parameter: f64) -> State {
            let w = self.frequency();
            let (cosine, sine) = self.phase(parameter);
            let rate = self.rate * cosine - self.spinor * (w * sine);
            State::from_spinor(self.spinor_at(parameter), rate, self.time_at(parameter))
        }

        /// The exact state at a physical time: the monotone clock inverted by bisection over
        /// half a spinor cycle either side of the mean-radius guess.
        pub fn at_time(&self, time: f64, iterations: usize) -> State {
            let w = self.frequency();
            let mean_radius =
                (self.spinor.norm_squared() + self.rate.norm_squared() / (w * w)) / 2.0;
            let guess = time / mean_radius;
            let (mut lower, mut upper) = (guess - PI / w, guess + PI / w);
            for _ in 0..iterations {
                let middle = (lower + upper) / 2.0;
                if self.time_at(middle) < time {
                    lower = middle;
                } else {
                    upper = middle;
                }
            }
            self.sample((lower + upper) / 2.0)
        }

        /// The RMS position error of a trajectory against the exact orbit at its own times,
        /// weighted by elapsed physical time (trapezoidal), so clustered samples do not dominate.
        #[allow(clippy::disallowed_methods)] // the root of a mean square, a statistic
        pub fn position_error(&self, trajectory: &[State], iterations: usize) -> f64 {
            let squared: Vec<f64> = trajectory
                .iter()
                .map(|s| (s.position - self.at_time(s.time, iterations).position).norm_squared())
                .collect();
            let integral: f64 = trajectory
                .windows(2)
                .zip(squared.windows(2))
                .map(|(s, e)| (e[0] + e[1]) * (s[1].time - s[0].time) / 2.0)
                .sum();
            let span = trajectory[trajectory.len() - 1].time - trajectory[0].time;
            (integral / span).sqrt()
        }
    }

    /// The inverse-square pull of a centre of strength `gravity` at the origin.
    pub fn pull(position: V, gravity: f64) -> V {
        position * (-gravity / position.norm().powi(3))
    }

    /// Kick-drift-kick integration with equal steps in physical time.
    pub fn physical_verlet(initial: State, gravity: f64, step: f64, steps: usize) -> Vec<State> {
        let mut state = initial;
        let mut acceleration = pull(state.position, gravity);
        let mut states = vec![initial];
        for _ in 0..steps {
            let half_velocity = state.velocity + acceleration * (step / 2.0);
            let position = state.position + half_velocity * step;
            acceleration = pull(position, gravity);
            state = State {
                time: state.time + step,
                position,
                velocity: half_velocity + acceleration * (step / 2.0),
            };
            states.push(state);
        }
        states
    }

    /// Kick-drift-kick integration of the spinor oscillator from a start time, with the
    /// physical clock (the radius integrated by Simpson's rule over each drift).
    pub fn regularized_verlet(
        orbit: &BoundOrbit,
        time: f64,
        step: f64,
        steps: usize,
    ) -> Vec<State> {
        let kick = orbit.energy * step / 4.0;
        let (mut spinor, mut rate, mut time) = (orbit.spinor, orbit.rate, time);
        let mut states = vec![State::from_spinor(spinor, rate, time)];
        for _ in 0..steps {
            let half_rate = rate + spinor * kick;
            let following = spinor + half_rate * step;
            // The midpoint's radius is a quarter of `|U + U'|²`, weighted four in Simpson's rule.
            time += step / 6.0
                * (spinor.norm_squared()
                    + (spinor + following).norm_squared()
                    + following.norm_squared());
            rate = half_rate + following * kick;
            spinor = following;
            states.push(State::from_spinor(spinor, rate, time));
        }
        states
    }

    // --- scenarios

    pub const GRAVITY: f64 = 1.0;
    pub const SEMIMAJOR_AXIS: f64 = 1.0;
    pub const ECCENTRICITIES: [f64; 3] = [0.6, 0.9, 0.97];
    pub const CASES: usize = ECCENTRICITIES.len();
    /// The case the lift shows: the most eccentric.
    pub const FEATURED_CASE: usize = 2;
    /// Steps per orbit in the passage.
    pub const STEPS: usize = 128;
    /// Samples of the exact passage.
    pub const REFERENCE_SAMPLES: usize = 1024;
    /// Samples of the lift, once round the orbit.
    pub const LIFT_SAMPLES: usize = 64;
    /// Steps per orbit in the convergence study.
    pub const COUNTS: [usize; 4] = [64, 128, 256, 512];
    /// Bisection steps inverting the clock.
    pub const CLOCK_ITERATIONS: usize = 44;

    /// Errors per step count (rows) and case (columns).
    pub type Table = [[f64; CASES]; COUNTS.len()];

    /// The period of every case, by Kepler's third law `2π √(a³ / GM)`.
    #[allow(clippy::disallowed_methods)] // Kepler's third law
    pub fn period() -> f64 {
        TAU * (SEMIMAJOR_AXIS.powi(3) / GRAVITY).sqrt()
    }

    /// `n + 1` equal steps from zero to `end`.
    pub fn steps_to(end: f64, n: usize) -> impl Iterator<Item = f64> {
        (0..=n).map(move |k| end * k as f64 / n as f64)
    }

    /// The orbit of an eccentricity started at apoapsis at time zero: the state and its lift
    /// through the unit spinor scaled by the root of the radius.
    #[allow(clippy::disallowed_methods)] // vis-viva's speed and the spinor's scale √radius
    pub fn initial_orbit(eccentricity: f64) -> (State, BoundOrbit) {
        let apoapsis_radius = SEMIMAJOR_AXIS * (1.0 + eccentricity);
        let speed = (GRAVITY / SEMIMAJOR_AXIS * (1.0 - eccentricity) / (1.0 + eccentricity)).sqrt();
        let spinor = Rotor::new(apoapsis_radius.sqrt(), 0.0, 0.0, 0.0);
        let start = State {
            time: 0.0,
            position: spinor >> REFERENCE,
            velocity: Vector::new(0.0, speed, 0.0),
        };
        (start, BoundOrbit::lift(&start, spinor, GRAVITY))
    }

    /// Every case's start and orbit.
    pub fn initial_orbits() -> [(State, BoundOrbit); CASES] {
        ECCENTRICITIES.map(initial_orbit)
    }

    /// The featured orbit sampled along the regularized clock: its positions and spinors.
    #[derive(Clone, Debug)]
    pub struct Lift {
        pub positions: Vec<V>,
        pub spinors: Vec<Spinor>,
    }

    /// The featured orbit at `samples + 1` equal steps of the regularized clock over
    /// `revolutions` orbits (half a spinor cycle each).
    pub fn lift(samples: usize, revolutions: usize) -> Lift {
        let (_, orbit) = initial_orbit(ECCENTRICITIES[FEATURED_CASE]);
        let end = revolutions as f64 * PI / orbit.frequency();
        let spinors: Vec<Spinor> = steps_to(end, samples).map(|s| orbit.spinor_at(s)).collect();
        let positions = spinors.iter().map(|u| *u >> REFERENCE).collect();
        Lift { positions, spinors }
    }

    /// One orbit by `steps` equal steps of physical time.
    pub fn physical_orbit(start: State, steps: usize) -> Vec<State> {
        physical_verlet(start, GRAVITY, period() / steps as f64, steps)
    }

    /// One orbit by `steps` equal steps of the regularized clock (half a spinor cycle).
    pub fn regularized_orbit(start: State, orbit: &BoundOrbit, steps: usize) -> Vec<State> {
        let step = PI / orbit.frequency() / steps as f64;
        regularized_verlet(orbit, start.time, step, steps)
    }

    /// One case's passage: the exact orbit and both schemes at `STEPS` steps.
    #[derive(Clone, Debug)]
    pub struct Passage {
        pub exact: Vec<State>,
        pub physical: Vec<State>,
        pub regularized: Vec<State>,
    }

    /// Every case's passage.
    pub fn passage() -> [Passage; CASES] {
        initial_orbits().map(|(start, orbit)| {
            let end = PI / orbit.frequency();
            Passage {
                exact: steps_to(end, REFERENCE_SAMPLES)
                    .map(|s| orbit.sample(s))
                    .collect(),
                physical: physical_orbit(start, STEPS),
                regularized: regularized_orbit(start, &orbit, STEPS),
            }
        })
    }

    /// The errors of both schemes over one orbit at each step count, in semimajor axes:
    /// physical, then regularized.
    pub fn convergence() -> (Table, Table) {
        let orbits = initial_orbits();
        let table = |run: &dyn Fn(State, &BoundOrbit, usize) -> Vec<State>| {
            COUNTS.map(|count| {
                orbits.map(|(start, orbit)| {
                    let error = orbit.position_error(&run(start, &orbit, count), CLOCK_ITERATIONS);
                    error / SEMIMAJOR_AXIS
                })
            })
        };
        (
            table(&|start, _, count| physical_orbit(start, count)),
            table(&regularized_orbit),
        )
    }
}

use kepler::*;

/// The loop: numga's animation, 120 frames of 60 ms over two revolutions.
const SECONDS: f32 = 7.2;
/// Samples of the drawn orbit and spinor loop over the two revolutions.
const CURVE_SAMPLES: usize = 240;

/// What the frames show, computed once.
struct Scene {
    orbit: BoundOrbit,
    /// The featured orbit over two revolutions, and its spinor's closed loop.
    curve: Lift,
    /// numga's lift: equal steps of the regularized clock over one revolution.
    dots: Lift,
    passages: [Passage; CASES],
    errors: (Table, Table),
}

fn scene() -> &'static Scene {
    static S: OnceLock<Scene> = OnceLock::new();
    S.get_or_init(|| Scene {
        orbit: initial_orbit(ECCENTRICITIES[FEATURED_CASE]).1,
        curve: lift(CURVE_SAMPLES, 2),
        dots: lift(LIFT_SAMPLES, 1),
        passages: passage(),
        errors: convergence(),
    })
}

/// A position in the orbital plane, as drawn.
fn plane(v: V) -> Point2 {
    from_above(reach3(v))
}

/// A spinor's scalar and `xy` parts, the plane its planar orbit's lift stays in.
fn spinor_plane(u: Spinor) -> Point2 {
    Point2::xy(u.s() as f32, u.e12() as f32)
}

/// The attracting centre.
fn centre(ax: &Axes, c: &mut Canvas) {
    ax.scatter(
        c,
        &[Point2::xy(0.0, 0.0)],
        Marker::Dot,
        9.0,
        palette::yellow(),
    );
}

/// A path through equal steps of the regularized clock, its samples coloured along it, and the
/// body on it.
fn lifted_path(c: &mut Canvas, ax: &Axes, path: &[Point2], dots: &[Point2], body: Point2) {
    ax.polyline(c, path, 1.4, palette::sky().faded(0.7));
    for (k, d) in dots.iter().enumerate() {
        let colour = colormap::viridis(k as f32 / (dots.len() - 1) as f32);
        ax.scatter(c, &[*d], Marker::Dot, 3.0, colour);
    }
    centre(ax, c);
    ax.scatter(c, &[body], Marker::Dot, 11.0, palette::purple());
}

/// The featured orbit and the body on it.
fn draw_orbit(c: &mut Canvas, rect: Rect, s: &Scene, spinor: Spinor) {
    let path: Vec<Point2> = s.curve.positions.iter().map(|v| plane(*v)).collect();
    let ax = Axes::fitting(rect, path.iter().copied(), 1.15);
    let dots: Vec<Point2> = s.dots.positions.iter().map(|v| plane(*v)).collect();
    let title = format!("ORBIT, E = {}", ECCENTRICITIES[FEATURED_CASE]);
    ax.frame(c, &title, "X", "Y");
    lifted_path(c, &ax, &path, &dots, plane(spinor >> REFERENCE));
}

/// The featured spinor's half-turn per orbit: its loop, the spinor and the opposite one, which
/// places the body at the same position.
fn draw_spinor(c: &mut Canvas, rect: Rect, s: &Scene, spinor: Spinor) {
    let path: Vec<Point2> = s.curve.spinors.iter().map(|u| spinor_plane(*u)).collect();
    let ax = Axes::fitting(rect, path.iter().copied(), 1.15);
    let dots: Vec<Point2> = s.dots.spinors.iter().map(|u| spinor_plane(*u)).collect();
    ax.frame(c, "ITS SPINOR", "SCALAR", "XY");
    lifted_path(c, &ax, &path, &dots, spinor_plane(spinor));
    let opposite = spinor_plane(spinor * -1.0);
    ax.scatter(
        c,
        &[opposite],
        Marker::Ring,
        11.0,
        palette::purple().faded(0.6),
    );
}

/// numga's convergence chart in one: both schemes' errors against the step count, a marker
/// shape per eccentricity.
fn draw_errors(c: &mut Canvas, rect: Rect, (physical, regularized): &(Table, Table)) {
    let (x, y) = ([48.0, 680.0], [1e-5, 1e3]);
    let ax = Axes::new(rect, x, y).log_x().log_y();
    // The frame's ticks would be decades; the step counts label the columns instead, so the
    // frame is drawn by axes with no decade across.
    Axes::new(rect, [2.0, 3.0], y).log_x().log_y().frame(
        c,
        "RMS POSITION ERROR / A",
        "STEPS PER ORBIT",
        "",
    );
    // The frame's tick label size.
    let size = (rect.height() / 26.0).clamp(3.0, 14.0) * 0.8;
    for count in COUNTS {
        let foot = ax.px(Point2::xy(count as f32, y[0])) + Point2::direction(0.0, size * 1.6);
        c.text(
            &count.to_string(),
            foot,
            size,
            palette::ink(),
            Align::Center,
        );
    }
    let markers = [Marker::Dot, Marker::Square, Marker::Triangle];
    for (table, colour) in [(physical, palette::orange()), (regularized, palette::sky())] {
        for (case, marker) in markers.iter().enumerate() {
            let points: Vec<Point2> = COUNTS
                .iter()
                .zip(table)
                .map(|(n, row)| Point2::xy(*n as f32, row[case] as f32))
                .collect();
            ax.polyline(c, &points, 1.4, colour);
            ax.scatter(c, &points, *marker, 7.0, colour);
        }
    }
    // The eccentricities beside the physical clock's last errors; the spinor clock's coincide.
    let last = COUNTS.len() - 1;
    for (case, e) in ECCENTRICITIES.iter().enumerate() {
        let beside = Point2::xy(COUNTS[last] as f32 * 1.12, physical[last][case] as f32);
        ax.text(
            c,
            beside,
            &format!("E {e}"),
            size,
            palette::orange(),
            Align::Left,
        );
    }
    ax.legend(
        c,
        &[
            ("PHYSICAL CLOCK", palette::orange()),
            ("SPINOR CLOCK", palette::sky()),
        ],
    );
}

/// The states of a trajectory up to a physical time, as drawn.
fn until(states: &[State], time: f64) -> Vec<Point2> {
    states
        .iter()
        .take_while(|s| s.time <= time)
        .map(|s| plane(s.position))
        .collect()
}

/// A passage up to a physical time on axes: the exact orbit whole and faint, both schemes so
/// far, with their bodies.
fn passage_on(c: &mut Canvas, ax: &Axes, p: &Passage, time: f64) {
    let exact: Vec<Point2> = p.exact.iter().map(|s| plane(s.position)).collect();
    ax.polyline(c, &exact, 2.5, palette::grid());
    for (states, colour) in [
        (&p.physical, palette::orange()),
        (&p.regularized, palette::sky()),
    ] {
        let shown = until(states, time);
        ax.polyline(c, &shown, 1.6, colour);
        if let Some(body) = shown.last() {
            ax.scatter(c, &[*body], Marker::Dot, 7.0, colour);
        }
    }
    centre(ax, c);
}

/// One case's passage: the whole orbit, and its periapsis enlarged.
fn draw_passage(c: &mut Canvas, rect: Rect, p: &Passage, eccentricity: f64, time: f64) {
    let exact: Vec<Point2> = p.exact.iter().map(|s| plane(s.position)).collect();
    let whole = Axes::fitting(rect.part(0.0, 0.0, 0.6, 1.0), exact.iter().copied(), 1.1);
    whole.frame(c, &format!("E = {eccentricity}"), "", "");
    passage_on(c, &whole, p, time);
    // numga's detail: a square about periapsis, reaching an eighth of the orbit's span (the
    // major axis) each way.
    let periapsis = SEMIMAJOR_AXIS * (1.0 - eccentricity);
    let reach = 0.12 * 2.0 * SEMIMAJOR_AXIS;
    let detail_rect = rect.part(0.66, 0.0, 1.0, 1.0);
    let detail_rect = Rect::around(
        detail_rect.centre(),
        Point2::direction(detail_rect.width() / 2.0, detail_rect.width() / 2.0),
    );
    let detail = Axes::equal(
        detail_rect,
        Point2::xy(-periapsis as f32, 0.0),
        reach as f32,
    );
    detail.frame(c, "PERIAPSIS", "", "");
    passage_on(c, &detail, p, time);
}

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let s = scene();
    // Two revolutions, uniformly in the regularized clock: one is half a spinor cycle.
    let fraction = f64::from((t / SECONDS).rem_euclid(1.0));
    let parameter = fraction * core::f64::consts::TAU / s.orbit.frequency();
    let spinor = s.orbit.spinor_at(parameter);
    let time = s.orbit.time_at(parameter).rem_euclid(period());
    let screen = c.rect();
    let top = screen.part(0.0, 0.13, 1.0, 0.53);
    draw_orbit(
        c,
        top.part(0.0, 0.0, 0.34, 1.0).inset(36.0, 22.0, 10.0, 34.0),
        s,
        spinor,
    );
    draw_spinor(
        c,
        top.part(0.34, 0.0, 0.6, 1.0).inset(36.0, 22.0, 10.0, 34.0),
        s,
        spinor,
    );
    draw_errors(
        c,
        top.part(0.6, 0.0, 1.0, 1.0).inset(48.0, 22.0, 44.0, 34.0),
        &s.errors,
    );
    let bottom = screen.part(0.0, 0.57, 1.0, 1.0);
    for (case, (p, e)) in s.passages.iter().zip(ECCENTRICITIES).enumerate() {
        let rect = bottom.column(case, CASES).inset(30.0, 22.0, 6.0, 20.0);
        draw_passage(c, rect, p, e, time);
    }
    let note = format!("ONE ORBIT AT {STEPS} STEPS, PHYSICAL TIME {time:.2}");
    let at = bottom.lo + Point2::direction(16.0, 0.0);
    c.text(&note, at, 10.0, palette::ink().faded(0.6), Align::Left);
    caption(
        c,
        "KEPLER: THROUGH PERIAPSIS WITH A POSITION SPINOR",
        "POSITION U X ~U, PHYSICAL TIME AT THE RADIUS U ~U: A SMOOTH OSCILLATOR (VGA3D)",
    );
}

fn main() {
    run(Anim::new("kepler", SECONDS).size(960, 540), draw);
}

#[cfg(test)]
mod tests {
    use super::kepler::*;
    use core::f64::consts::{PI, TAU};
    use gax::vga3d::{Bivector, Vector};

    type B = Bivector<(), f64>;

    /// The angular momentum per unit mass, a bivector.
    fn momentum(s: &State) -> B {
        s.position ^ s.velocity
    }

    /// Every coefficient of `a` within `atol` of `b`'s, as numpy's `assert_allclose`.
    fn near(a: &[f64], b: &[f64], atol: f64) {
        for (x, y) in a.iter().zip(b) {
            assert!((x - y).abs() <= atol, "{a:?} vs {b:?}");
        }
    }

    fn turn(plane: B) -> Spinor {
        plane.exp().into_inner()
    }

    #[test]
    #[allow(clippy::disallowed_methods)] // the test's spinor scale √1.8
    fn lift_reconstructs_velocity_and_exact_motion_in_tilted_planes() {
        for angle in [0.2, 0.5, -0.3] {
            let rotation =
                turn(Bivector::new(angle, 0.0, 0.0)) * turn(Bivector::new(0.0, 0.0, angle / 2.0));
            let spinor = rotation * 1.8f64.sqrt();
            let velocity = rotation >> Vector::new(0.13, 0.3, 0.0);
            let start = State {
                time: 0.0,
                position: spinor >> REFERENCE,
                velocity,
            };
            let orbit = BoundOrbit::lift(&start, spinor, 1.0);
            let initial = State::from_spinor(orbit.spinor, orbit.rate, start.time);
            let trajectory: Vec<State> = steps_to(PI / orbit.frequency(), 80)
                .map(|s| orbit.sample(s))
                .collect();
            let semimajor = -1.0 / (2.0 * orbit.energy);
            let period = TAU * semimajor.powf(1.5);
            let last = trajectory[80];

            near(&initial.velocity.c, &velocity.c, 1e-12);
            // The lift does not turn about the radius: the placement is symmetric in it.
            let open = placement();
            let (a, b) = (
                open.of(orbit.rate).of(orbit.spinor),
                open.of(orbit.spinor).of(orbit.rate),
            );
            near(&a.c, &b.c, 1e-12);
            near(&last.position.c, &initial.position.c, 1e-11);
            near(&last.velocity.c, &initial.velocity.c, 1e-11);
            near(&[last.time], &[period], 1e-10);
            for state in &trajectory {
                near(&[state.energy(1.0)], &[orbit.energy], 1e-10);
                near(&momentum(state).c, &momentum(&initial).c, 1e-11);
            }
        }
    }

    #[test]
    fn spinor_gauge_changes_neither_orbit_nor_clock() {
        for ((_, orbit), angle) in initial_orbits().iter().zip([0.3, -0.5, 0.7]) {
            // A turn about the reference direction on the right.
            let gauge = turn(Bivector::new(angle, 0.0, 0.0));
            let shifted = BoundOrbit {
                spinor: orbit.spinor * gauge,
                rate: orbit.rate * gauge,
                energy: orbit.energy,
            };
            for phase in steps_to(TAU, 90) {
                let (first, second) = (orbit.sample(phase), shifted.sample(phase));
                near(&first.position.c, &second.position.c, 1e-11);
                near(&first.velocity.c, &second.velocity.c, 1e-10);
                near(&[first.time], &[second.time], 1e-10);
            }
        }
    }

    #[test]
    fn physical_clock_inversion_and_inverse_square_acceleration() {
        let delta = 1e-4;
        for (_, orbit) in initial_orbits() {
            for k in 0..7 {
                let time = 0.2 + 0.8 * k as f64 / 6.0;
                let state = orbit.at_time(time, CLOCK_ITERATIONS);
                let before = orbit.at_time(time - delta, CLOCK_ITERATIONS);
                let after = orbit.at_time(time + delta, CLOCK_ITERATIONS);
                let acceleration = (after.velocity - before.velocity) / (2.0 * delta);
                let force = pull(state.position, GRAVITY);

                near(&[state.time], &[time], 1e-10);
                for (a, f) in acceleration.c.iter().zip(force.c) {
                    assert!((a - f).abs() <= 1e-7 + 1e-6 * f.abs(), "{a} vs {f}");
                }
            }
        }
    }

    /// Both central-force updates keep the angular momentum; the resolved orbits converge at
    /// second order in either clock, and the spinor clock resolves the closest passage far
    /// better.
    #[test]
    fn both_verlet_schemes_converge_and_regularization_resolves_periapsis() {
        let (physical_errors, regularized_errors) = convergence();
        for ((start, _), p) in initial_orbits().iter().zip(passage()) {
            for state in p.physical.iter().chain(&p.regularized) {
                near(&momentum(state).c, &momentum(start).c, 1e-10);
            }
        }
        let ratio = |t: &Table, k: usize, case: usize| t[k + 1][case] / t[k][case];
        for k in 0..COUNTS.len() - 1 {
            near(&[ratio(&physical_errors, k, 0)], &[0.25], 0.05);
            for case in 0..CASES {
                near(&[ratio(&regularized_errors, k, case)], &[0.25], 0.05);
            }
        }
        for (physical, regularized) in physical_errors.iter().zip(&regularized_errors) {
            assert!(regularized[CASES - 1] < physical[CASES - 1] / 100.0);
        }
    }

    /// Regression against numga's convergence errors (e2765b5): the corners of both tables.
    #[test]
    fn convergence_errors_match_numga() {
        let (physical, regularized) = convergence();
        let expected = [
            (physical[0][0], 5.068_938_060_41e-2),
            (physical[3][2], 3.421_014_213_41),
            (regularized[0][0], 1.243_099_290_32e-3),
            (regularized[3][2], 2.458_410_827_81e-5),
        ];
        for (ours, theirs) in expected {
            assert!((ours / theirs - 1.0).abs() < 1e-8, "{ours} vs {theirs}");
        }
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws(super::draw, 0.5);
    }
}
