//! numga's `quantum/stabilizer`: two qubits kept in four, so that any single qubit going wrong is
//! noticed, the smallest stabilizer code that catches bit flips and phase flips alike, in the
//! spinors of Cl(4,4). The real-amplitude states of four qubits have sixteen real components,
//! and the real linear operators on them are the algebra itself: every pattern of bit and phase
//! flips is, up to sign, one of its 256 blades.
//!
//! Four commuting elements that square to one, `a`, `be`, `cf` and `dg`, give four projectors
//! whose product is an idempotent, a minimal left ideal: its left multiples are a space of
//! sixteen dimensions, closed under left multiplication, the spinors. Sixteen blades times the
//! idempotent are a basis of it (the `State` kind), an operator acts on a state by its product
//! on the left, read back in that basis, and the scalar part of `cdeh ~psi efgh phi` (times 16)
//! is a positive pairing in which the basis is orthonormal.
//!
//! Each qubit's bit flip and phase flip are blades that square to one and anticommute; those
//! of different qubits commute. Two checks, flipping all four qubits and phase-flipping all
//! four, commute with each other and with the flips and phase flips of two logical qubits (pairs
//! of qubits); the states both checks leave alone are the code. Any single error changes the
//! outcome of at least one check, so it is noticed and the run discarded: the runs kept go
//! wrong only when two errors strike, at the square of the error rate, while a bare qubit goes
//! wrong at the rate itself.
//!
//! The animation turns the two logical qubits stored in the code (their flips' and phase flips'
//! expectations on the dials) while each of the sixteen single errors strikes in turn: the
//! state's sixteen spinor components change and the checks that notice it light up. Beside it,
//! the runs kept and the infidelities under noise.

use gax_numga_examples::points::circle;
use gax_numga_examples::signal::{phasor, wave};
use gax_numga_examples::{
    Align, Anim, Axes, Canvas, Light, Marker, ORIGIN2, Point2, Rect, backdrop, caption, palette,
    run,
};
use std::sync::OnceLock;

gax::algebra! {
    algebra cl44 "Cl(4,4): four directions that square to one (ea eb ec ed) and four that square to minus one (ee ef eg eh).";
    basis ea = 1, eb = 1, ec = 1, ed = 1, ee = -1, ef = -1, eg = -1, eh = -1;
    kind Scalar = [1];
    kind Vector = [ea, eb, ec, ed, ee, ef, eg, eh];
    kind State = [1, eb, ec, ed, eh, ebc, ebd, ecd, ebh, ech, edh, ebcd, ebch, ebdh, ecdh, ebcdh];
}

mod stabilizer {
    use super::cl44::{Multivector, Scalar, State, Vector};
    use gax_numga_examples::signal::linspace;
    use gax_numga_examples::signal::phasor;

    /// An operator, any element of the algebra, and a state: a spinor of the ideal.
    pub type Full = Multivector;
    pub type St = State;

    /// The qubits, the kinds of error on each (none, bit flip, phase flip, both), and the two
    /// logical qubits.
    pub const QUBITS: usize = 4;
    pub const KINDS: usize = 4;
    pub const LOGICAL: usize = 2;

    pub fn one() -> Full {
        Scalar::new(1.0).cast::<Multivector>()
    }

    /// The basis vector named by a letter, `a` to `h`.
    fn vector(name: char) -> Vector {
        let mut c = [0.0; 8];
        c["abcdefgh".find(name).expect("a basis vector a to h")] = 1.0;
        Vector::from_coeffs(c)
    }

    /// The blade of the basis vectors named, in order: `blade("bce")` is `b c e`.
    pub fn blade(names: &str) -> Full {
        names.chars().fold(one(), |product, n| product * vector(n))
    }

    /// The exponential of an element that squares to minus one: a turn, `cos + x sin`. (In
    /// closed form, since the element is any multivector of the 8D algebra, not a bivector.)
    fn turn(x: Full, angle: f64) -> Full {
        let p = phasor(angle);
        one() * p.e20() + x * p.e01()
    }

    /// The code: the spinor space and its pairing, each qubit's flips, the checks and their
    /// projectors.
    pub struct Code {
        ideal: Full,
        pairing_left: Full,
        pairing_middle: Full,
        pub flips: [Full; QUBITS],
        pub phase_flips: [Full; QUBITS],
        pub checks: [Full; 2],
        pub projector: Full,
    }

    impl Code {
        pub fn new() -> Code {
            let projector = |x: &str| (one() + blade(x)) * 0.5;
            let ideal = projector("a") * projector("be") * projector("cf") * projector("dg");
            // Square-one blades that anticommute within each qubit's pair and commute across
            // qubits; the phase flips leave the ground state alone.
            let flips = [blade("b"), -blade("bce"), -blade("bcdef"), -blade("ah")];
            let phase_flips = [blade("be"), blade("cf"), blade("dg"), -blade("abcdefg")];
            let all = |ops: &[Full; QUBITS]| ops.iter().fold(one(), |p, x| p * *x);
            let checks = [all(&flips), all(&phase_flips)];
            let passes = |check: Full| (one() + check) * 0.5;
            Code {
                ideal,
                pairing_left: blade("cdeh"),
                pairing_middle: blade("efgh"),
                flips,
                phase_flips,
                projector: passes(checks[0]) * passes(checks[1]),
                checks,
            }
        }

        /// A state as an element of the ideal: its blades times the idempotent.
        fn embed(&self, state: St) -> Full {
            state * self.ideal
        }

        /// An operator acting on a state: its product with the state on the left, read back in
        /// the state's basis (the factor 16 undoes the basis's coefficient in the idempotent).
        pub fn act(&self, op: Full, state: St) -> St {
            (op * self.embed(state) * 16.0).cast::<State>()
        }

        /// The positive pairing of two states, in which the State basis is orthonormal.
        pub fn pairing(&self, a: St, b: St) -> f64 {
            let between = self.embed(a).reverse() * self.pairing_middle * self.embed(b);
            16.0 * self.pairing_left.scalar_product(between).s()
        }

        /// A state at unit length in the pairing.
        #[allow(clippy::disallowed_methods)] // a length in the spinors' pairing, a form gax's norm does not know
        fn normalized(&self, state: St) -> St {
            state / self.pairing(state, state).sqrt()
        }

        /// The expectation of an operator on a state.
        pub fn expectation(&self, op: Full, state: St) -> f64 {
            self.pairing(state, self.act(op, state)) / self.pairing(state, state)
        }

        /// What can happen to a qubit: nothing, a bit flip, a phase flip, or both.
        pub fn errors(&self, qubit: usize) -> [Full; KINDS] {
            let (f, p) = (self.flips[qubit], self.phase_flips[qubit]);
            [one(), f, p, f * p]
        }

        /// The flips of the two logical qubits: pairs of qubits, commuting with both checks.
        pub fn logical_flips(&self) -> [Full; LOGICAL] {
            let f = self.flips;
            [f[0] * f[1], f[0] * f[2]]
        }
        pub fn logical_phase_flips(&self) -> [Full; LOGICAL] {
            let p = self.phase_flips;
            [p[0] * p[2], p[0] * p[1]]
        }

        /// Two qubits turned from 0 towards 1 by the exponentials of their flips times their
        /// phase flips: these square to minus one and turn the states in their pairing.
        pub fn turned(
            &self,
            start: St,
            flips: [Full; 2],
            phase_flips: [Full; 2],
            angles: [f64; 2],
        ) -> St {
            let [a, b] = [0, 1].map(|k| turn(flips[k] * phase_flips[k], angles[k] / 2.0));
            self.act(a * b, start)
        }

        /// Two logical qubits at the given angles, stored in the code.
        pub fn encoded(&self, angles: [f64; 2]) -> St {
            let start = self.normalized(self.act(self.projector, ground()));
            self.turned(
                start,
                self.logical_flips(),
                self.logical_phase_flips(),
                angles,
            )
        }

        /// Every pattern of errors on the first `qubits` qubits: the operator, and the kind of
        /// error on each qubit.
        fn patterns(&self, qubits: usize) -> Vec<(Full, Vec<usize>)> {
            let mut out = vec![(one(), Vec::new())];
            for qubit in 0..qubits {
                let errors = self.errors(qubit);
                out = out
                    .iter()
                    .flat_map(|(op, kinds)| {
                        errors.iter().enumerate().map(move |(kind, e)| {
                            let mut more = kinds.clone();
                            more.push(kind);
                            (*op * *e, more)
                        })
                    })
                    .collect();
            }
            out
        }

        /// The acceptance and the unconditional fidelity of a stored state under noise, at each
        /// rate: the chance both checks pass is the part of the corrupted state left in the code,
        /// and the fidelity its squared overlap with the stored state. Rejected states have no
        /// overlap with the code, so the fidelity over the acceptance is that of the runs kept.
        pub fn detected(&self, stored: St, rates: &[f64]) -> (Vec<f64>, Vec<f64>) {
            let outcomes: Vec<(f64, f64, Vec<usize>)> = self
                .patterns(QUBITS)
                .into_iter()
                .map(|(op, kinds)| {
                    let corrupted = self.act(op, stored);
                    let passed = self.pairing(corrupted, self.act(self.projector, corrupted));
                    (passed, self.pairing(stored, corrupted).powi(2), kinds)
                })
                .collect();
            rates
                .iter()
                .map(|r| {
                    outcomes
                        .iter()
                        .fold((0.0, 0.0), |(kept, fidelity), (p, f, kinds)| {
                            let w = weight(kinds, *r);
                            (kept + w * p, fidelity + w * f)
                        })
                })
                .unzip()
        }

        /// The fidelity of two bare qubits with their intended state under the same noise.
        pub fn unprotected(&self, angles: [f64; 2], rates: &[f64]) -> Vec<f64> {
            let flips = [self.flips[0], self.flips[1]];
            let phase_flips = [self.phase_flips[0], self.phase_flips[1]];
            let stored = self.turned(ground(), flips, phase_flips, angles);
            let overlaps: Vec<(f64, Vec<usize>)> = self
                .patterns(LOGICAL)
                .into_iter()
                .map(|(op, kinds)| (self.pairing(stored, self.act(op, stored)).powi(2), kinds))
                .collect();
            rates
                .iter()
                .map(|r| {
                    overlaps
                        .iter()
                        .map(|(o, kinds)| weight(kinds, *r) * o)
                        .sum()
                })
                .collect()
        }

        /// Both checks after each kind of error on each qubit, `[qubit][kind][check]`.
        pub fn single_errors(&self, state: St) -> [[[f64; 2]; KINDS]; QUBITS] {
            core::array::from_fn(|q| {
                self.errors(q)
                    .map(|e| self.checks.map(|c| self.expectation(c, self.act(e, state))))
            })
        }
    }

    /// All qubits 0.
    pub fn ground() -> St {
        Scalar::new(1.0).cast::<State>()
    }

    /// The chance of a pattern of errors at an error rate: each qubit is left alone, or hit by
    /// one of the three errors with equal chance.
    fn weight(kinds: &[usize], rate: f64) -> f64 {
        kinds
            .iter()
            .map(|k| if *k == 0 { 1.0 - rate } else { rate / 3.0 })
            .product()
    }

    /// The angles of the two stored logical qubits, numga's.
    pub const ANGLES: [f64; 2] = [0.7, 1.9];

    /// The rates of numga's noise figure: 51 from none to a half.
    pub fn rates() -> Vec<f64> {
        linspace(0.0, 0.5, 51)
    }

    /// Two logical qubits at `angles`, stored in the code and bare, under noise at each rate:
    /// the runs kept, the infidelity of those kept, and the bare qubits' infidelity.
    pub fn noisy(code: &Code, angles: [f64; 2], rates: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let (kept, fidelity) = code.detected(code.encoded(angles), rates);
        let encoded = kept
            .iter()
            .zip(&fidelity)
            .map(|(k, f)| 1.0 - f / k)
            .collect();
        let bare = code
            .unprotected(angles, rates)
            .iter()
            .map(|f| 1.0 - f)
            .collect();
        (kept, encoded, bare)
    }
}

use stabilizer::*;

const SECONDS: f32 = 16.0;
/// The error kinds' short and long names, and the State basis blades' names.
const KIND_NAMES: [&str; KINDS] = ["NONE", "BIT", "PHASE", "BOTH"];
const STRIKES: [&str; KINDS] = ["NO ERROR", "A BIT FLIP", "A PHASE FLIP", "BOTH FLIPS"];
const BLADES: [&str; 16] = [
    "1", "B", "C", "D", "H", "BC", "BD", "CD", "BH", "CH", "DH", "BCD", "BCH", "BDH", "CDH", "BCDH",
];

/// What the frames share, computed once.
struct Data {
    code: Code,
    rates: Vec<f64>,
    kept: Vec<f64>,
    encoded: Vec<f64>,
    bare: Vec<f64>,
}

fn data() -> &'static Data {
    static D: OnceLock<Data> = OnceLock::new();
    D.get_or_init(|| {
        let code = Code::new();
        let rates = rates();
        let (kept, encoded, bare) = noisy(&code, ANGLES, &rates);
        Data {
            code,
            rates,
            kept,
            encoded,
            bare,
        }
    })
}

/// The colour of a check's outcome: passed, or noticed.
fn outcome(value: f64) -> Light {
    if value > 0.0 {
        palette::GREEN
    } else {
        palette::RED
    }
}

/// The two logical qubits on their dials: each one's phase flip read across and its flip up.
fn draw_logical(c: &mut Canvas, area: Rect, d: &Data, state: St) {
    let below = Point2::direction(0.0, 1.0);
    let flips = d.code.logical_flips();
    let phase_flips = d.code.logical_phase_flips();
    for k in 0..LOGICAL {
        let cell = area.column(k, LOGICAL);
        let ax = Axes::equal(cell, ORIGIN2, 1.3);
        ax.polyline(c, &circle(ORIGIN2, 1.0, 64), 1.0, palette::GRID);
        let reading = Point2::xy(
            d.code.expectation(phase_flips[k], state) as f32,
            d.code.expectation(flips[k], state) as f32,
        );
        ax.arrow(c, ORIGIN2, reading, 2.2, 8.0, palette::SKY);
        let name = cell.bottom_middle() + below.gp(8.0);
        let text = format!("LOGICAL {}", k + 1);
        c.text(&text, name, 10.0, palette::SKY, Align::Center);
    }
    let note = area.bottom_middle() + below.gp(24.0);
    let text = "ACROSS: PHASE FLIP, UP: FLIP";
    c.text(text, note, 9.0, palette::INK, Align::Center);
}

/// The sixteen spinor components of a state as bars over their blades' names (staggered in two
/// rows, to fit).
fn draw_components(c: &mut Canvas, ax: &Axes, state: St) {
    let (lo, hi) = (Point2::xy(-0.5, 0.0), Point2::xy(15.5, 0.0));
    ax.line(c, lo, hi, 1.0, palette::GRID);
    let below = Point2::direction(0.0, 1.0);
    for (k, (value, name)) in state.c.iter().zip(BLADES).enumerate() {
        let (x, v) = (k as f32, *value as f32);
        let bar = [
            Point2::xy(x - 0.35, 0.0),
            Point2::xy(x + 0.35, 0.0),
            Point2::xy(x + 0.35, v),
            Point2::xy(x - 0.35, v),
        ];
        ax.fill(c, &bar, palette::PURPLE, 0.8);
        let row = (k % 2) as f32;
        let label = ax.px(Point2::xy(x, ax.y[0])) + below.gp(10.0 + 11.0 * row);
        c.text(name, label, 8.0, palette::INK, Align::Center);
    }
}

/// Both checks after every single error, a pair of lights per qubit and kind, with the error
/// striking now ringed.
fn draw_checks(
    c: &mut Canvas,
    ax: &Axes,
    readings: &[[[f64; 2]; KINDS]; QUBITS],
    now: (usize, usize),
) {
    let up = Point2::direction(0.0, -1.0);
    let title = ax.rect.top_middle() + up.gp(6.0);
    let text = "CHECKS AFTER EACH SINGLE ERROR";
    c.text(text, title, 10.0, palette::INK, Align::Center);
    for (q, row) in readings.iter().enumerate() {
        let y = (QUBITS - 1 - q) as f32;
        let name = format!("QUBIT {}", q + 1);
        ax.text(
            c,
            Point2::xy(-0.55, y - 0.1),
            &name,
            9.0,
            palette::INK,
            Align::Right,
        );
        for (k, pair) in row.iter().enumerate() {
            let x = k as f32;
            for (j, value) in pair.iter().enumerate() {
                let at = Point2::xy(x - 0.13 + 0.26 * j as f32, y);
                ax.scatter(c, &[at], Marker::Square, 8.0, outcome(*value));
            }
            if (q, k) == now {
                let at = [Point2::xy(x, y)];
                ax.scatter(c, &at, Marker::Ring, 26.0, palette::YELLOW);
            }
        }
    }
    for (k, name) in KIND_NAMES.iter().enumerate() {
        let at = Point2::xy(k as f32, -0.75);
        ax.text(c, at, name, 9.0, palette::INK, Align::Center);
    }
}

/// What the lights of the check table mean.
fn draw_key(c: &mut Canvas, area: Rect) {
    let lines = [
        ("LEFT: THE CHECK FLIPPING ALL", palette::INK),
        ("RIGHT: PHASE-FLIPPING ALL", palette::INK),
        ("GREEN: PASSED", palette::GREEN),
        ("RED: NOTICED", palette::RED),
    ];
    for (k, (text, colour)) in lines.iter().enumerate() {
        let at = area.lo + Point2::direction(0.0, 18.0 * k as f32);
        c.text(text, at, 9.0, *colour, Align::Left);
    }
}

/// The runs kept and the two infidelities against the error rate, with a cursor.
fn draw_noise(c: &mut Canvas, ax: &Axes, d: &Data, rate: f64) {
    ax.frame(c, "UNDER NOISE", "CHANCE OF AN ERROR ON EACH QUBIT", "");
    let curve = |values: &[f64]| -> Vec<Point2> {
        d.rates
            .iter()
            .zip(values)
            .map(|(r, v)| Point2::xy(*r as f32, *v as f32))
            .collect()
    };
    ax.dashed(c, &curve(&d.kept), 1.4, 5.0, palette::INK.faded(0.6));
    ax.polyline(c, &curve(&d.bare), 1.8, palette::RED);
    ax.polyline(c, &curve(&d.encoded), 1.8, palette::SKY);
    let r = rate as f32;
    ax.line(
        c,
        Point2::xy(r, 0.0),
        Point2::xy(r, 1.0),
        1.0,
        palette::GRID,
    );
    ax.legend(
        c,
        &[
            ("RUNS KEPT", palette::INK),
            ("BARE INFIDELITY", palette::RED),
            ("KEPT RUNS' INFIDELITY", palette::SKY),
        ],
    );
}

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let d = data();
    let screen = c.rect();
    let progress = (t / SECONDS).fract();
    let tau = core::f64::consts::TAU;
    let swing = f64::from(wave(core::f32::consts::TAU * progress));
    let angles = [
        ANGLES[0] + tau * f64::from(progress),
        ANGLES[1] + 0.8 * swing,
    ];
    let state = d.code.encoded(angles);
    let strike = ((progress * (QUBITS * KINDS) as f32) as usize).min(QUBITS * KINDS - 1);
    let (qubit, kind) = (strike / KINDS, strike % KINDS);
    let corrupted = d.code.act(d.code.errors(qubit)[kind], state);

    draw_logical(c, screen.part(0.03, 0.14, 0.33, 0.4), d, state);
    let readings = d.code.single_errors(state);
    let checks = Axes::new(screen.part(0.46, 0.2, 0.7, 0.44), [-0.5, 3.5], [-0.5, 3.5]);
    draw_checks(c, &checks, &readings, (qubit, kind));
    draw_key(c, screen.part(0.73, 0.24, 0.98, 0.44));
    let components = Axes::new(
        screen.part(0.04, 0.6, 0.48, 0.86),
        [-0.6, 15.6],
        [-0.75, 0.75],
    );
    draw_components(c, &components, corrupted);
    let label = components.rect.top_middle() + Point2::direction(0.0, -10.0);
    let text = format!("THE SPINOR AFTER {} ON QUBIT {}", STRIKES[kind], qubit + 1);
    c.text(&text, label, 10.0, palette::YELLOW, Align::Center);
    // The cursor on the noise plot eases over the rates, from none to a half and back.
    let rate = 0.5 * (0.5 - 0.5 * f64::from(phasor(core::f32::consts::TAU * progress).e20()));
    let noise = Axes::new(screen.part(0.58, 0.58, 0.97, 0.88), [0.0, 0.5], [0.0, 1.0]);
    draw_noise(c, &noise, d, rate);
    caption(
        c,
        "A STABILIZER CODE: TWO QUBITS KEPT IN FOUR",
        "CL(4,4): ERRORS ARE BLADES, STATES THE SPINORS OF A MINIMAL LEFT IDEAL",
    );
}

fn main() {
    run(Anim::new("stabilizer", SECONDS).size(960, 540), draw);
}

#[cfg(test)]
#[allow(clippy::disallowed_methods)] // the logical qubits' sines and cosines, the references checked against
mod tests {
    use super::stabilizer::*;

    fn small(x: Full) -> bool {
        x.c.iter().all(|v| v.abs() < 1e-12)
    }

    /// numga's `test_checks_commute_with_each_other_and_with_the_logical_qubits`: the checks
    /// square to one and commute with each other and with the logical qubits' flips and phase
    /// flips; each logical qubit's flip and phase flip anticommute and commute with the other
    /// qubit's; the four outcomes' projectors add up to one.
    #[test]
    fn the_checks_commute_with_each_other_and_with_the_logical_qubits() {
        let code = Code::new();
        let [a, b] = code.checks;
        assert!(small(a * a - one()) && small(b * b - one()));
        assert!(small(a.commutator(b)));
        let (flips, phase_flips) = (code.logical_flips(), code.logical_phase_flips());
        for check in code.checks {
            for op in flips.iter().chain(&phase_flips) {
                assert!(small(check.commutator(*op)));
            }
        }
        for k in 0..LOGICAL {
            assert!(small(flips[k].anticommutator(phase_flips[k])));
            assert!(small(flips[k].commutator(phase_flips[1 - k])));
        }
        let half = |x: Full, sign: f64| (one() + x * sign) * 0.5;
        let projectors: Full = [(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)]
            .iter()
            .map(|(s, t)| half(a, *s) * half(b, *t))
            .sum();
        assert!(small(projectors - one()));
    }

    /// numga's `test_stored_state_passes_both_checks_and_every_single_error_is_caught`: the
    /// stored state passes both checks, its logical qubits read the sines and cosines of their
    /// angles, and every error but none fails at least one check.
    #[test]
    fn the_stored_state_passes_and_every_single_error_is_caught() {
        let code = Code::new();
        let state = code.encoded(ANGLES);
        for check in code.checks {
            assert!((code.expectation(check, state) - 1.0).abs() < 1e-12);
        }
        let (flips, phase_flips) = (code.logical_flips(), code.logical_phase_flips());
        for k in 0..LOGICAL {
            assert!((code.expectation(flips[k], state) - ANGLES[k].sin()).abs() < 1e-12);
            assert!((code.expectation(phase_flips[k], state) - ANGLES[k].cos()).abs() < 1e-12);
        }
        for row in code.single_errors(state) {
            assert!(row[0].iter().all(|v| (v - 1.0).abs() < 1e-12));
            assert!(
                row[1..].iter().all(|pair| pair.iter().any(|v| *v < 0.0)),
                "{row:?}"
            );
        }
    }

    /// numga's `test_kept_runs_go_wrong_at_second_order_and_the_figure_draws`: doubling a small
    /// rate quadruples the infidelity of the runs kept and doubles the bare one.
    #[test]
    fn kept_runs_go_wrong_at_second_order() {
        let code = Code::new();
        let rates = rates();
        let (kept, encoded, bare) = noisy(&code, ANGLES, &rates);
        assert!(
            (encoded[2] / encoded[1] / 4.0 - 1.0).abs() < 0.05,
            "{}",
            encoded[2] / encoded[1]
        );
        assert!(
            (bare[2] / bare[1] / 2.0 - 1.0).abs() < 0.05,
            "{}",
            bare[2] / bare[1]
        );
        assert!(kept.iter().all(|k| *k <= 1.0 + 1e-12));
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws(super::draw, 0.5);
    }
}
