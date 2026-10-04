//! numga's `quantum/magnetic_resonance`: a spin in a magnetic field, driven by a radio-frequency
//! field and relaxing, in the Pauli algebra of space (VGA3D). The state of a spin (or of an
//! ensemble) is `(1 + r) / 2` for its Bloch vector `r`: a scalar plus a vector, its own reverse,
//! VGA3D's `Paravector`. The pseudoscalar `I` squares to -1 and commutes with everything.
//!
//! The state changes by a linear map, the generator. In the frame that turns with the drive the
//! Hamiltonian is half the detuning along z plus half the drive along x, and the state turns by
//! `-2 I` times its commutator with it. Relaxation enters through processes `L`, each adding
//! `L rho ~L - {~L L, rho} / 2`: `x (1 - z) / 2` turns spins against the field over onto it at the
//! rate `1 / T1`, and `z` scrambles the phase. Where the generator vanishes the state is steady,
//! a linear solve with the scalar part pinned. A short step of evolution is the exponential of
//! the generator to fourth order, a sum of powers of a map; composed with itself it spans twice
//! the time, and composed with the pulses (sandwiches, maps too) a whole pulse sequence is one map.
//!
//! The animation is a spin echo: an ensemble of spins with detunings scattered by an uneven field
//! is tipped into the transverse plane, fans out (seen from above the field, coloured by
//! detuning), is turned over by a half-turn pulse and refocuses, while the signal (the mean
//! transverse spin) collapses and returns at twice the delay, reduced only by the true dephasing.
//! Below: spins nutating into their steady states, the absorption lines under weak and strong
//! drive, and the echo against the free decay as composed maps.

use gax::vga3d::{Bivector, Vector};
use gax_numga_examples::scene3::panel3;
use gax_numga_examples::{
    Align, Anim, Axes, Camera, Canvas, Lens, Marker, ORIGIN3, Point2, backdrop, caption, colormap,
    palette, run,
};
use std::sync::OnceLock;

mod resonance {
    use gax::Unit;
    use gax::vga3d::{Multivector, Paravector, Pseudoscalar, Rotor, Scalar, Vector};
    use gax_numga_examples::rng::{Draw, rng};

    /// A spin's state, `(1 + r) / 2`: a scalar plus a vector.
    pub type St = Paravector<(), f64>;
    /// A rate of change of states, and an evolution over a stretch of time: maps on states.
    pub type Rates = Paravector<(Paravector,), f64>;
    pub type Evolution = Paravector<(Paravector,), f64>;
    /// A relaxation process, a vector plus a bivector (VGA3D declares no kind of just these
    /// grades, so it is a multivector).
    pub type Pr = Multivector<(), f64>;

    /// The relaxation times, in microseconds: typical of an electron spin in a solid.
    pub const T1: f64 = 10.0;
    pub const T2: f64 = 4.0;

    pub fn x() -> Vector<(), f64> {
        Vector::new(1.0, 0.0, 0.0)
    }
    pub fn z() -> Vector<(), f64> {
        Vector::new(0.0, 0.0, 1.0)
    }
    /// The pseudoscalar, the imaginary unit.
    pub fn i() -> Pseudoscalar<(), f64> {
        Pseudoscalar::new(1.0)
    }
    pub fn one() -> St {
        Paravector::new(1.0, 0.0, 0.0, 0.0)
    }

    /// The open state.
    fn open() -> Rates {
        Paravector::slot()
    }

    /// The state with Bloch vector `r`.
    pub fn state(r: Vector<(), f64>) -> St {
        (one() + r.cast::<Paravector>()) * 0.5
    }

    /// All spins along the field.
    pub fn equilibrium() -> St {
        state(z())
    }

    /// The Bloch vector of a state: twice its vector part.
    pub fn bloch(rho: St) -> Vector<(), f64> {
        rho.cast::<Vector>() * 2.0
    }

    /// `x` times the state of a spin against the field: turns that part of the state over onto the
    /// field.
    pub fn raise() -> Pr {
        x() * state(-z())
    }

    /// The change of a state `rho` due to a relaxation `process`:
    /// `process rho ~process - {back, rho} / 2`, with `back = ~process process`. The sandwich of a
    /// state by a vector plus a bivector is a state again (it keeps the state its own reverse), so
    /// its other grades are dropped.
    pub fn relaxation(process: Pr) -> Rates {
        let sandwich = (process * open() * process.reverse()).cast::<Paravector>();
        let back = (process.reverse() * process).cast::<Paravector>();
        sandwich - back.anticommutator(open()).cast::<Paravector>()
    }

    /// The rate of change of a state in the frame that turns with the drive: turning about the axis
    /// `drive x + detuning z` at its length, relaxing toward +z at `1 / t1`, and dephasing along z
    /// at `1 / t2 - 1 / (2 t1)`, so that the transverse part decays at `1 / t2`.
    pub fn generator(detuning: f64, drive: f64, t1: f64, t2: f64) -> Rates {
        let hamiltonian = (z() * detuning + x() * drive) * 0.5;
        let turning = (i() * hamiltonian.commutator(open())).cast::<Paravector>() * -2.0;
        let dephasing = z().cast::<Multivector>();
        let relaxing = relaxation(raise()) * (1.0 / t1)
            + relaxation(dephasing) * (0.5 * (1.0 / t2 - 1.0 / (2.0 * t1)));
        turning + relaxing
    }

    /// The state that does not change. The generator keeps the scalar part, so it is singular;
    /// adding the scalar part as a dyad on one pins it: the sum sends a state to its scalar part
    /// plus its change, and solving that against one half gives the steady state.
    pub fn steady(rates: Rates) -> St {
        let scalar_part = Scalar::new(1.0).scalar_product(open()).cast::<Paravector>();
        (rates + scalar_part).solve(one() * 0.5)
    }

    /// What becomes of each state over `dt`: the exponential of the generator times dt to fourth
    /// order, its powers by composition.
    pub fn evolution(rates: Rates, dt: f64) -> Evolution {
        let small = rates * dt;
        let mut term = open();
        let mut total = term;
        for order in 1..=4 {
            term = small.of(term) / f64::from(order);
            total += term;
        }
        total
    }

    /// The states before each of `steps` steps, and the state after the last.
    pub fn evolve(each_step: Evolution, mut rho: St, steps: usize) -> (Vec<St>, St) {
        let mut before = Vec::with_capacity(steps);
        for _ in 0..steps {
            before.push(rho);
            rho = each_step.of(rho);
        }
        (before, rho)
    }

    /// A short strong pulse along x as a rotor: it turns the Bloch vector about x by the angle, in
    /// the sense of the drive, by its sandwich.
    pub fn pulse(angle: f64) -> Unit<Rotor<(), f64>> {
        (i() * x() * (-angle / 2.0)).exp()
    }

    /// The pulse as a map on states. Its sandwich keeps a state a state; gax types a rotor's
    /// sandwich of a scalar plus a vector as a multivector, so the empty grades are dropped.
    pub fn pulsed(angle: f64) -> Evolution {
        (pulse(angle) >> open()).cast::<Paravector>()
    }

    /// The states through a spin echo: tipped onto -y, left for `before` steps, turned half a turn
    /// about x, and left for `after` steps.
    pub fn echo(each_step: Evolution, rho: St, before: usize, after: usize) -> Vec<St> {
        let half_pi = core::f64::consts::FRAC_PI_2;
        let (mut states, rho) = evolve(each_step, pulsed(half_pi).of(rho), before);
        let (rest, _) = evolve(each_step, pulsed(2.0 * half_pi).of(rho), after);
        states.extend(rest);
        states
    }

    /// The evolution over its own span, then over twice and four times that span, and so on: each
    /// composed with itself for the next.
    pub fn doublings(mut span: Evolution, count: usize) -> Vec<Evolution> {
        let mut out = Vec::with_capacity(count);
        for _ in 0..count {
            out.push(span);
            span = span.of(span);
        }
        out
    }

    /// Normal deviates of the given spread (numga's NumPy streams cannot be reproduced; the
    /// checks on them hold for any scattered detunings).
    pub fn normals(seed: u64, n: usize, spread: f64) -> Vec<f64> {
        let mut r = rng(seed);
        (0..n).map(|_| spread * r.normal()).collect()
    }

    /// Spins switched on to a steady drive at the given detunings, from equilibrium: their states
    /// every few steps (`[time][detuning]`), the last state of each, and the steady states.
    pub fn nutation(
        detunings: &[f64],
        drive: f64,
        seconds: f64,
        dt: f64,
        every: usize,
    ) -> (Vec<Vec<St>>, Vec<St>, Vec<St>) {
        let steps = (seconds / dt) as usize + 1;
        let runs: Vec<(Vec<St>, Vec<St>)> = detunings
            .iter()
            .map(|d| {
                let rates = generator(*d, drive, T1, T2);
                let (states, _) = evolve(evolution(rates, dt), equilibrium(), steps);
                (states, vec![steady(rates)])
            })
            .collect();
        let sampled = (0..steps)
            .step_by(every)
            .map(|k| runs.iter().map(|(s, _)| s[k]).collect())
            .collect();
        let last = runs.iter().map(|(s, _)| s[steps - 1]).collect();
        let settled = runs.iter().map(|(_, s)| s[0]).collect();
        (sampled, last, settled)
    }

    /// The steady states over a grid of detunings and drive strengths, `[detuning][drive]`.
    pub fn lines(detunings: &[f64], drives: &[f64]) -> Vec<Vec<St>> {
        detunings
            .iter()
            .map(|d| {
                drives
                    .iter()
                    .map(|w| steady(generator(*d, *w, T1, T2)))
                    .collect()
            })
            .collect()
    }

    /// A spin echo: an ensemble with detunings scattered by an uneven field, tipped, left to fan
    /// out for the delay, turned over and left to refocus. The detunings, and the ensemble every
    /// few steps (`[time][spin]`).
    pub fn spin_echo(
        spins: usize,
        spread: f64,
        delay: f64,
        seconds: f64,
        dt: f64,
        every: usize,
        seed: u64,
    ) -> (Vec<f64>, Vec<Vec<St>>) {
        let detunings = normals(seed, spins, spread);
        let waits = (delay / dt).round() as usize;
        let total = (seconds / dt) as usize;
        let runs: Vec<Vec<St>> = detunings
            .iter()
            .map(|d| {
                let step = evolution(generator(*d, 0.0, T1, T2), dt);
                echo(step, equilibrium(), waits, total - waits)
            })
            .collect();
        let ensemble = (0..total)
            .step_by(every)
            .map(|k| runs.iter().map(|r| r[k]).collect())
            .collect();
        (detunings, ensemble)
    }

    /// The echo and the free decay of an ensemble as maps from state to state: one step on the
    /// open state, doubled for delays from dt to `2^(count - 1) dt`, composed with the pulses and
    /// averaged over the spins. The times after the first pulse, and the states the two sequences
    /// leave from equilibrium.
    pub fn echo_decay(
        spins: usize,
        spread: f64,
        dt: f64,
        count: usize,
        seed: u64,
    ) -> (Vec<f64>, Vec<St>, Vec<St>) {
        let detunings = normals(seed, spins, spread);
        let half_pi = core::f64::consts::FRAC_PI_2;
        let (tip, turn) = (pulsed(half_pi), pulsed(2.0 * half_pi));
        let mut echo = vec![Evolution::zero(); count];
        let mut decay = vec![Evolution::zero(); count];
        for d in &detunings {
            let waiting = doublings(evolution(generator(*d, 0.0, T1, T2), dt), count);
            for (k, wait) in waiting.iter().enumerate() {
                echo[k] += wait.of(turn.of(wait.of(tip)));
                decay[k] += wait.of(wait.of(tip));
            }
        }
        let mean = 1.0 / spins as f64;
        let times = (0..count).map(|k| 2.0 * dt * 2f64.powi(k as i32)).collect();
        let echoed = echo.iter().map(|m| (*m * mean).of(equilibrium())).collect();
        let faded = decay
            .iter()
            .map(|m| (*m * mean).of(equilibrium()))
            .collect();
        (times, echoed, faded)
    }

    /// The length of a state's transverse Bloch vector, what a pick-up coil sees of it: the
    /// area its Bloch vector spans with the field's direction.
    pub fn transverse(rho: St) -> f64 {
        (bloch(rho) ^ z()).norm()
    }

    /// The state of the whole ensemble: the mean of its spins' states.
    pub fn mean(ensemble: &[St]) -> St {
        ensemble.iter().fold(St::zero(), |acc, rho| acc + *rho) / ensemble.len() as f64
    }

    /// The ensemble's mean transverse Bloch vector, its length.
    pub fn signal(ensemble: &[St]) -> f64 {
        transverse(mean(ensemble))
    }
}

use resonance::*;

const SECONDS: f32 = 10.0;
const DELAY: f64 = 2.5;
const ECHO_DT: f64 = 0.01;
const ECHO_EVERY: usize = 4;
const DRIVES: [f64; 3] = [0.05, 0.3, 1.0];

/// What the frames share, computed once.
struct Data {
    detunings: Vec<f64>,
    ensemble: Vec<Vec<St>>,
    signal: Vec<f64>,
    nutation: Vec<Vec<St>>,
    settled: Vec<St>,
    line_detunings: Vec<f64>,
    lines: Vec<Vec<St>>,
    decay_times: Vec<f64>,
    echoed: Vec<St>,
    faded: Vec<St>,
}

fn data() -> &'static Data {
    static D: OnceLock<Data> = OnceLock::new();
    D.get_or_init(|| {
        let (detunings, ensemble) = spin_echo(300, 2.0, DELAY, 8.0, ECHO_DT, ECHO_EVERY, 0);
        let signal = ensemble.iter().map(|e| signal(e)).collect();
        let (nutation, _, settled) = nutation(&[0.0, 0.5, 1.0], 1.0, 40.0, 0.02, 5);
        let line_detunings: Vec<f64> = (0..241).map(|k| -3.0 + 6.0 * k as f64 / 240.0).collect();
        let lines = lines(&line_detunings, &DRIVES);
        let (decay_times, echoed, faded) = echo_decay(300, 2.0, 0.01, 11, 0);
        Data {
            detunings,
            ensemble,
            signal,
            nutation,
            settled,
            line_detunings,
            lines,
            decay_times,
            echoed,
            faded,
        }
    })
}

/// A Bloch vector seen from above the field: its part in the `x y` plane.
fn from_above(r: Vector<(), f64>) -> gax::vga2d::Vector<(), f64> {
    gax::vga2d::Vector::new(r.e1(), r.e2())
}

/// A point of a plot, from its two values.
fn on_plot(across: f64, up: f64) -> gax::pga2d::Point<(), f64> {
    gax::pga2d::Point::xy(across, up)
}

/// The unit directions of space.
fn axes() -> [Vector<(), f64>; 3] {
    [x(), Vector::new(0.0, 1.0, 0.0), z()]
}

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let d = data();
    let (w, h) = (c.width, c.height);
    let (wf, hf) = (w as f32, h as f32);
    let phase = (t / SECONDS).clamp(0.0, 1.0);
    let small = (hf / 50.0).clamp(7.0, 11.0);

    // The ensemble seen from above the field, coloured by detuning.
    let records = d.ensemble.len();
    let index = ((phase * records as f32) as usize).min(records - 1);
    let dt = (ECHO_DT * ECHO_EVERY as f64) as f32;
    let now = index as f32 * dt;
    let top_rect = [wf * 0.02, hf * 0.15, wf * 0.34, hf * 0.58];
    let ax = Axes::equal(top_rect, Point2::xy(0.0, 0.0), 1.12);
    // The unit circle: x turned about the field.
    let circle: Vec<_> = (0..=96)
        .map(|k| {
            let a = core::f64::consts::TAU * k as f64 / 96.0;
            from_above((Bivector::new(0.0, 0.0, 1.0) * (-a / 2.0)).exp() >> x())
        })
        .collect();
    ax.polyline(c, &circle, 1.0, palette::grid(), 1.0);
    for a in [x(), Vector::new(0.0, 1.0, 0.0)] {
        let reach = from_above(a * 1.1);
        ax.line(c, -reach, reach, 1.0, palette::grid(), 0.6);
    }
    let mut order: Vec<usize> = (0..d.detunings.len()).collect();
    order.sort_by(|a, b| d.detunings[*a].total_cmp(&d.detunings[*b]));
    for k in order {
        let r = from_above(bloch(d.ensemble[index][k]));
        let tone = colormap::coolwarm(0.5 + d.detunings[k] as f32 / 8.0);
        ax.scatter(c, &[r], Marker::Dot, 3.5, tone, 0.9);
    }
    let signal = from_above(bloch(mean(&d.ensemble[index])));
    ax.arrow(c, Point2::xy(0.0, 0.0), signal, 2.5, 9.0, palette::yellow());
    let stage = if now < 0.3 {
        "TIPPED ONTO -Y"
    } else if now < DELAY as f32 - 0.05 {
        "FANNING OUT"
    } else if now < DELAY as f32 + 0.3 {
        "HALF-TURN PULSE"
    } else if (now - 2.0 * DELAY as f32).abs() < 0.3 {
        "ECHO"
    } else if now < 2.0 * DELAY as f32 {
        "REFOCUSING"
    } else {
        "FANNING OUT AGAIN"
    };
    c.text(
        "SPINS FROM ABOVE",
        (top_rect[0] + top_rect[2]) * 0.5,
        top_rect[1] - 6.0,
        small * 0.9,
        palette::ink(),
        Align::Center,
    );
    c.text(
        &format!("T = {now:.2} US: {stage}"),
        (top_rect[0] + top_rect[2]) * 0.5,
        top_rect[3] + small * 1.2,
        small * 0.9,
        palette::ink(),
        Align::Center,
    );

    // The signal so far.
    let total = records as f32 * dt;
    let ax = Axes::new(
        [wf * 0.42, hf * 0.17, wf * 0.97, hf * 0.52],
        [0.0, total],
        [0.0, 1.05],
    );
    ax.frame(c, "MEAN TRANSVERSE SPIN", "TIME (US)", "");
    for (at, name) in [(DELAY as f32, "PI PULSE"), (2.0 * DELAY as f32, "ECHO")] {
        ax.line(
            c,
            Point2::xy(at, 0.0),
            Point2::xy(at, 1.05),
            1.0,
            palette::grid(),
            1.0,
        );
        ax.text(
            c,
            Point2::xy(at + 0.06, 0.95),
            name,
            small,
            palette::grid(),
            Align::Left,
        );
    }
    let decay: Vec<Point2> = (0..=100)
        .map(|k| {
            let s = total * k as f32 / 100.0;
            Point2::xy(s, (-s / T2 as f32).exp())
        })
        .collect();
    ax.dashed(c, &decay, 1.0, 5.0, palette::grid(), 1.0);
    let trace: Vec<Point2> = d.signal[..=index]
        .iter()
        .enumerate()
        .map(|(k, s)| Point2::xy(k as f32 * dt, *s as f32))
        .collect();
    ax.polyline(c, &trace, 1.8, palette::sky(), 1.0);
    ax.scatter(
        c,
        &[Point2::xy(now, d.signal[index] as f32)],
        Marker::Dot,
        6.0,
        palette::sky(),
        1.0,
    );

    // Spins switched on to a steady drive, nutating into their steady states.
    let row = hf * 0.64;
    let ball = [0.0, row, wf * 0.3, hf];
    let cam = Camera::orbit(
        (ball[2] - ball[0]) as usize,
        (ball[3] - ball[1]) as usize,
        ORIGIN3,
        4.4,
        (-50.0f32).to_radians() + 0.3 * phase,
        18.0f32.to_radians(),
        Lens::Perspective(0.6),
    );
    let shown = ((phase * d.nutation.len() as f32) as usize).clamp(1, d.nutation.len());
    let colours = [palette::red(), palette::purple(), palette::sky()];
    panel3(c, ball, cam, |s| {
        s.sphere_wire(ORIGIN3, 1.0, 16, palette::grid(), 0.45);
        for a in axes() {
            s.seg(-a, a, 1.0, palette::grid(), 1.0);
        }
        for (j, colour) in colours.iter().enumerate() {
            let path: Vec<_> = d.nutation[..shown]
                .iter()
                .map(|row| bloch(row[j]))
                .collect();
            s.polyline(&path, 1.2, *colour, 0.9);
            s.dot(bloch(d.settled[j]), Marker::Ring, 7.0, *colour);
            s.dot(*path.last().expect("a state"), Marker::Dot, 6.0, *colour);
        }
    });
    c.text(
        "NUTATION TO THE STEADY STATE",
        wf * 0.15,
        row + 4.0,
        small * 0.9,
        palette::ink(),
        Align::Center,
    );

    // Absorption against detuning, per unit drive: a stronger drive broadens and flattens it.
    let ax = Axes::new(
        [wf * 0.36, hf * 0.7, wf * 0.63, hf * 0.9],
        [-3.0, 3.0],
        [0.0, 4.2],
    );
    ax.frame(c, "ABSORPTION PER UNIT DRIVE", "DETUNING (RAD/US)", "");
    let mut legend = Vec::new();
    for (j, drive) in DRIVES.iter().enumerate() {
        let pts: Vec<_> = d
            .line_detunings
            .iter()
            .zip(&d.lines)
            .map(|(det, row)| on_plot(*det, -bloch(row[j]).e2() / drive))
            .collect();
        ax.polyline(c, &pts, 1.5, colours[j], 1.0);
        legend.push((format!("DRIVE {drive}"), colours[j]));
    }
    // A cursor sweeping the detuning.
    let sweep = -3.0 + 6.0 * (0.5 - 0.5 * (core::f32::consts::TAU * phase).cos());
    ax.line(
        c,
        Point2::xy(sweep, 0.0),
        Point2::xy(sweep, 4.2),
        1.0,
        palette::grid(),
        1.0,
    );
    let entries: Vec<(&str, _)> = legend.iter().map(|(s, c)| (s.as_str(), *c)).collect();
    ax.legend(c, &entries);

    // The echo against the free decay, from composed maps, on a logarithmic time axis.
    let ax = Axes::new(
        [wf * 0.69, hf * 0.7, wf * 0.97, hf * 0.9],
        [0.015, 30.0],
        [0.0, 1.05],
    )
    .log_x();
    ax.frame(c, "ECHO AND FREE DECAY", "TIME AFTER TIP (US)", "");
    let fine: Vec<Point2> = (0..=60)
        .map(|k| {
            let s = 0.02 * (1000.0f32).powf(k as f32 / 60.0);
            Point2::xy(s, (-s / T2 as f32).exp())
        })
        .collect();
    ax.dashed(c, &fine, 1.0, 5.0, palette::grid(), 1.0);
    for (states, colour) in [(&d.echoed, palette::sky()), (&d.faded, palette::red())] {
        let pts: Vec<_> = d
            .decay_times
            .iter()
            .zip(states)
            .map(|(s, rho)| on_plot(*s, transverse(*rho)))
            .collect();
        ax.polyline(c, &pts, 1.5, colour, 1.0);
        ax.scatter(c, &pts, Marker::Dot, 4.5, colour, 1.0);
    }
    if now > 0.02 {
        ax.line(
            c,
            Point2::xy(now, 0.0),
            Point2::xy(now, 1.05),
            1.0,
            palette::yellow(),
            0.7,
        );
    }
    ax.legend(
        c,
        &[("ECHO", palette::sky()), ("FREE DECAY", palette::red())],
    );

    caption(
        c,
        "MAGNETIC RESONANCE: A SPIN ECHO",
        "BLOCH AND LINDBLAD AS MAPS ON STATES (1 + R)/2, T1 10 US, T2 4 US",
    );
}

fn main() {
    run(
        Anim::new("magnetic resonance", SECONDS).size(960, 540),
        draw,
    );
}

#[cfg(test)]
mod tests {
    use super::resonance::*;
    use gax::ApproxEq;
    use gax::vga3d::{Multivector, Paravector, Vector};

    fn close(a: St, b: St, tol: f64) -> bool {
        a.max_abs_diff(&b) <= tol
    }

    /// The generator is the Bloch equations: the Bloch vector changes by `w x r` less the
    /// relaxation, with `w = (drive, 0, detuning)`, the transverse part decaying at `1 / T2` and
    /// the part along the field relaxing to one at `1 / T1`.
    #[test]
    fn the_generator_is_the_bloch_equations() {
        let (detuning, drive, t1, t2) = (0.7, 1.3, 5.0, 2.0);
        let rates = generator(detuning, drive, t1, t2);
        let r = Vector::new(0.4, -0.2, 0.6);
        let change = bloch(rates.of(state(r)));
        let w = Vector::new(drive, 0.0, detuning);
        // The cross product `w × r`, the vector at right angles to the plane `w ^ r`.
        let cross = -(i() * (w ^ r));
        let relaxing = Vector::new(r.e1() / t2, r.e2() / t2, (r.e3() - 1.0) / t1);
        let want = cross - relaxing;
        assert!(change.max_abs_diff(&want) < 1e-12, "{change:?} vs {want:?}");
    }

    /// The parts the generator drops by casting are zero: the sandwich of a state by a process,
    /// `~L L` and the commutator term all stay states.
    #[test]
    fn relaxation_keeps_states_their_own_reverse() {
        let states = |m: Multivector<(), f64>| m.cast::<Paravector>().cast::<Multivector>();
        for p in [raise(), z().cast::<Multivector>()] {
            let s = Paravector::new(0.5, 0.1, -0.3, 0.2);
            let full = p * s * p.reverse();
            assert!(full.max_abs_diff(&states(full)) < 1e-15);
            let back = p.reverse() * p;
            assert!(back.max_abs_diff(&states(back)) < 1e-15);
        }
        // The raising process is a vector plus a bivector.
        let r = raise();
        assert!(r.grade::<0>().s() == 0.0 && r.grade::<3>().e123() == 0.0);
    }

    /// numga's `nutation` check: after several T1 the driven spins have reached the steady
    /// states the solve predicts.
    #[test]
    fn driven_spins_settle_into_the_steady_state() {
        for (detunings, seconds, dt, every) in [
            (vec![0.0, 0.5], 60.0, 0.05, 20),
            (vec![0.0, 0.5, 1.0], 40.0, 0.02, 5),
        ] {
            let (sampled, last, settled) = nutation(&detunings, 1.0, seconds, dt, every);
            assert_eq!(sampled[0].len(), detunings.len());
            for (a, b) in last.iter().zip(&settled) {
                assert!(close(*a, *b, 1e-2), "{a:?} vs {b:?}");
            }
        }
    }

    /// numga's `lines` checks: the steady states do not change, and match the closed form: along
    /// the field `(1 + (D T2)²) / d` and absorbing `W T2 / d`, `d = 1 + (D T2)² + W² T1 T2`.
    #[test]
    fn steady_states_match_the_closed_form() {
        let cases: [(Vec<f64>, Vec<f64>); 2] = [
            (
                (0..9).map(|k| -2.0 + 0.5 * k as f64).collect(),
                vec![0.1, 1.0],
            ),
            (
                (0..241).map(|k| -3.0 + 6.0 * k as f64 / 240.0).collect(),
                vec![0.05, 0.3, 1.0],
            ),
        ];
        for (detunings, drives) in cases {
            let settled = lines(&detunings, &drives);
            for (dd, row) in detunings.iter().zip(&settled) {
                for (ww, rho) in drives.iter().zip(row) {
                    let change = generator(*dd, *ww, T1, T2).of(*rho);
                    assert!(change.c.iter().all(|v| v.abs() < 1e-12));
                    let d = 1.0 + (dd * T2).powi(2) + ww * ww * T1 * T2;
                    let r = bloch(*rho);
                    assert!((r.e3() - (1.0 + (dd * T2).powi(2)) / d).abs() < 1e-12);
                    assert!((-r.e2() - ww * T2 / d).abs() < 1e-12);
                }
            }
        }
    }

    /// numga's `echo` check: at twice the delay the spins have refocused, the mean transverse
    /// Bloch vector reduced only by the true dephasing, `exp(-2 delay / T2)`.
    #[test]
    fn the_spins_refocus_at_twice_the_delay() {
        for (spins, seconds, every, seed) in [(40, 6.0, 20, 1), (300, 8.0, 4, 0)] {
            let (delay, dt) = (2.5, 0.01);
            let (_, ensemble) = spin_echo(spins, 2.0, delay, seconds, dt, every, seed);
            let at = (2.0 * delay / (dt * every as f64)).round() as usize;
            let got = signal(&ensemble[at]);
            let want = (-2.0 * delay / T2).exp();
            assert!((got - want).abs() <= 0.02 * want, "{got} vs {want}");
        }
    }

    /// numga's `echo_decay` check: the echo, built from composed maps, has lost only the true
    /// dephasing, `exp(-t / T2)`, whatever the spread of the field.
    #[test]
    fn the_echo_as_composed_maps_loses_only_the_true_dephasing() {
        for (spins, count, seed) in [(40, 8, 1), (300, 11, 0)] {
            let (times, echoed, faded) = echo_decay(spins, 2.0, 0.01, count, seed);
            for (t, rho) in times.iter().zip(&echoed) {
                let want = (-t / T2).exp();
                assert!((transverse(*rho) - want).abs() <= 1e-5 * want, "{t}");
            }
            assert_eq!(faded.len(), count);
        }
    }

    #[test]
    fn a_frame_draws() {
        let mut draw = super::draw;
        let anim = gax_numga_examples::Anim::new("t", super::SECONDS).size(480, 270);
        let a = gax_numga_examples::app::frame(&anim, 0.5, &mut draw);
        let b = gax_numga_examples::app::frame(&anim, 6.0, &mut draw);
        assert!(a.mean()[0] > 0.0);
        assert!(a.mean() != b.mean());
    }
}
