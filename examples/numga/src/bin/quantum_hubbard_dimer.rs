//! numga's `quantum/hubbard_dimer`: two electrons hopping between two sites, with an energy cost
//! for sharing a site, in R(4,0). The four basis vectors are the orbitals: spin up and down on
//! the left site (`a`, `b`) and on the right (`c`, `d`). A state of two electrons is a
//! bivector, the wedge of the two occupied orbitals: it vanishes when both are the same orbital
//! (exclusion) and changes sign when they trade places (exchange). A one-electron map acts on a
//! pair by acting on each electron in turn, `lift(h)(u ^ v) = h(u) ^ v + u ^ h(v)`: in a
//! Euclidean basis, `Σ h(eᵢ) ^ (eᵢ ⌋ pair)`, each orbital taken out of the pair in turn and its
//! image put back. The repulsion is a sum of dyads on the two doubly occupied configurations.
//! (numga writes the lift without the metric, a join with an open complement and a trace; the
//! contraction here is the same in a Euclidean algebra.)
//!
//! Hopping takes the singlet only to the symmetric double occupancy and back. On those two
//! states the Hamiltonian less half the repulsion, applied twice, is the square of a rate: the
//! length of its two parts, half the repulsion and twice the hopping. So the ground state is the
//! singlet projected by the rate less the shifted Hamiltonian, the triplets stay at zero energy,
//! and the singlet-triplet gap, the exchange, is the rate less half the repulsion; for a strong
//! repulsion it is `4 t² / U`. Time acts through the cosine and the sine of the Hamiltonian
//! times time, so a state in time is a pair of real states, its cosine and its sine part: each
//! configuration's amplitude is a point of a plane, its probability the squared distance.
//!
//! The animation runs one exchange cycle from spin up on the left and spin down on the right,
//! without repulsion, at a moderate one and at a strong one: each configuration's amplitude is
//! an arrow on its dial, the triplet's share (grey) then the singlet block's (purple). Without
//! repulsion both electrons often share a site; with a strong one the spins swap with hardly any
//! charge moving, as localized spins coupled by exchange. Beside them: the six levels against
//! the repulsion, the swap against the spin-only prediction, the swap frozen by a field that
//! differs between the sites, and the charge an asymmetry between the sites moves over.

use gax_numga_examples::points::circle;
use gax_numga_examples::signal::{linspace, wave};
use gax_numga_examples::{
    Align, Anim, Axes, Canvas, Light, Marker, ORIGIN2, Point2, Rect, backdrop, caption, palette,
    run,
};
use std::sync::OnceLock;

#[path = "../shared/quantum.rs"]
mod quantum;

gax::algebra! {
    algebra dimer "Two electrons on two sites, R(4,0): spin up and down on the left (ea, eb) and on the right (ec, ed).";
    basis ea = 1, eb = 1, ec = 1, ed = 1;
    kind Scalar = [1];
    kind Orbital = [ea, eb, ec, ed];
    kind Pair = [eab, eac, ead, ebc, ebd, ecd];
}

mod hubbard {
    use super::dimer::{Orbital, Pair};
    pub use super::quantum::InTime;
    use core::f64::consts::FRAC_1_SQRT_2;
    use gax_numga_examples::signal::linspace;
    use gax_numga_examples::signal::phasor;

    /// An orbital, and a state of two electrons.
    pub type O = Orbital<(), f64>;
    pub type P = Pair<(), f64>;
    /// A one-electron map, and a map on pairs.
    pub type Hopping = Orbital<(Orbital,), f64>;
    pub type Hamiltonian = Pair<(Pair,), f64>;

    /// The orbitals: up and down on the left, up and down on the right.
    pub fn orbitals() -> [O; 4] {
        [
            Orbital::new(1.0, 0.0, 0.0, 0.0),
            Orbital::new(0.0, 1.0, 0.0, 0.0),
            Orbital::new(0.0, 0.0, 1.0, 0.0),
            Orbital::new(0.0, 0.0, 0.0, 1.0),
        ]
    }

    /// Both electrons in the orbitals `i` and `j`.
    pub fn pair(i: usize, j: usize) -> P {
        let o = orbitals();
        o[i] ^ o[j]
    }

    /// The six two-electron configurations: `ab`, `ac`, `ad`, `bc`, `bd`, `cd`.
    pub fn configurations() -> [P; 6] {
        [
            pair(0, 1),
            pair(0, 2),
            pair(0, 3),
            pair(1, 2),
            pair(1, 3),
            pair(2, 3),
        ]
    }
    /// The configuration both electrons start in: up on the left, down on the right.
    pub fn up_down() -> P {
        pair(0, 3)
    }
    /// The configuration of the swapped spins: down on the left, up on the right.
    pub const SWAPPED: usize = 3;
    pub fn singlet() -> P {
        (pair(0, 3) - pair(1, 2)) * FRAC_1_SQRT_2
    }
    /// The three triplets: both up, the triplet of zero spin, both down.
    pub fn triplets() -> [P; 3] {
        [
            pair(0, 2),
            (pair(0, 3) + pair(1, 2)) * FRAC_1_SQRT_2,
            pair(1, 3),
        ]
    }

    /// The dyad that measures a state and returns it.
    fn dyad(p: P) -> Hamiltonian {
        p * p.reverse().scalar_product(Pair::slot())
    }

    /// How many sites hold both electrons.
    pub fn double_occupancy() -> Hamiltonian {
        dyad(pair(0, 1)) + dyad(pair(2, 3))
    }

    /// The correlation of the two spins, in ħ²: -3/4 on the singlet, 1/4 on the triplets, and
    /// zero where a site is empty and the other doubly occupied.
    pub fn spin_correlation() -> Hamiltonian {
        (Pair::slot() - double_occupancy() - dyad(singlet()) * 4.0) * 0.25
    }

    /// An electron moved to the other site with its spin kept, at `-strength`.
    pub fn hopping(strength: f64) -> Hopping {
        let o = orbitals();
        let across = |from: O, to: O| to * (from | Orbital::slot());
        (across(o[0], o[2]) + across(o[2], o[0]) + across(o[1], o[3]) + across(o[3], o[1]))
            * -strength
    }

    /// A one-electron map acting on either electron of a pair in turn: each orbital taken out of
    /// the pair by the contraction and its image wedged back in.
    pub fn lift(single: Hopping) -> Hamiltonian {
        orbitals()
            .into_iter()
            .map(|e| single.of(e) ^ e.lc(Pair::slot()))
            .sum()
    }

    /// Either electron hops, and sharing a site costs the repulsion.
    pub fn hamiltonian(strength: f64, repulsion: f64) -> Hamiltonian {
        lift(hopping(strength)) + double_occupancy() * repulsion
    }

    /// Each orbital's number, half signed, lifted to pairs.
    fn half_signed(signs: [f64; 4]) -> Hamiltonian {
        let numbers = orbitals()
            .into_iter()
            .zip(signs)
            .map(|(e, s)| e * (e | Orbital::slot()) * (0.5 * s))
            .sum();
        lift(numbers)
    }

    /// A unit field along z that differs between the sites, half on each: up raised and down
    /// lowered on the left, the other way round on the right.
    pub fn field_difference() -> Hamiltonian {
        half_signed([1.0, -1.0, -1.0, 1.0])
    }

    /// A unit asymmetry between the sites, half on each: the left lowered, the right raised.
    pub fn site_difference() -> Hamiltonian {
        half_signed([-1.0, -1.0, 1.0, 1.0])
    }

    /// The rate of the singlet's block: on the singlet and the symmetric double occupancy the
    /// Hamiltonian less half the repulsion is half the repulsion along one and twice the hopping
    /// across, so it squares to their squared length.
    pub fn block_rate(strength: f64, repulsion: f64) -> f64 {
        gax::vga2d::Vector::new(repulsion / 2.0, 2.0 * strength).norm()
    }

    /// The singlet-triplet gap: the rate less half the repulsion.
    pub fn exchange_gap(strength: f64, repulsion: f64) -> f64 {
        block_rate(strength, repulsion) - repulsion / 2.0
    }

    /// The Hamiltonian less half the repulsion, on the singlet.
    fn shifted_singlet(strength: f64, repulsion: f64) -> P {
        hamiltonian(strength, repulsion).of(singlet()) - singlet() * (repulsion / 2.0)
    }

    /// The lowest state: the singlet projected onto the lower state of its block by the rate
    /// less the shifted Hamiltonian, divided by its norm. (A sum of two pairs is not a blade:
    /// `normalized` would scale it to a unit versor, `x ~x = 1` with the 4-vector part too, which
    /// is not a state's normalization.)
    pub fn ground(strength: f64, repulsion: f64) -> P {
        let lower =
            singlet() * block_rate(strength, repulsion) - shifted_singlet(strength, repulsion);
        lower / lower.norm()
    }

    impl InTime<P> {
        /// The chance of each configuration: its squared amplitude in both parts.
        pub fn probabilities(&self) -> [f64; 6] {
            let (c, s) = (probabilities(self.cosine), probabilities(self.sine));
            core::array::from_fn(|k| c[k] + s[k])
        }

        /// The expectation of an observable, from both parts.
        pub fn expectation(&self, observable: Hamiltonian) -> f64 {
            expectation(self.cosine, observable) + expectation(self.sine, observable)
        }

        /// The part along a state, in both parts.
        pub fn along(&self, p: P) -> InTime<P> {
            let project = |x: P| p * p.reverse().scalar_product(x).s();
            InTime {
                cosine: project(self.cosine),
                sine: project(self.sine),
            }
        }
    }

    /// Spin up on the left and down on the right, half singlet and half triplet of zero spin,
    /// after `time`, in closed form: the triplet, at zero energy, stays in the cosine part. On
    /// the singlet's block the Hamiltonian is half the repulsion plus the shifted part, whose
    /// cosine and sine act on the singlet as those of the rate; the half repulsion turns the two
    /// into each other by its own angle.
    pub fn exchange(strength: f64, repulsion: f64, time: f64) -> InTime<P> {
        let rate = block_rate(strength, repulsion);
        let block = phasor(rate * time);
        let along = singlet() * block.e20();
        let across = shifted_singlet(strength, repulsion) * (block.e01() / rate);
        let half = phasor(0.5 * repulsion * time);
        let (c, s) = (half.e20(), half.e01());
        InTime {
            cosine: (triplets()[1] + along * c - across * s) * FRAC_1_SQRT_2,
            sine: (along * s + across * c) * FRAC_1_SQRT_2,
        }
    }

    /// The chance of each configuration in a real state: its squared amplitude.
    pub fn probabilities(state: P) -> [f64; 6] {
        configurations().map(|c| c.reverse().scalar_product(state).s().powi(2))
    }

    /// The expectation of an observable in a real state.
    pub fn expectation(state: P, observable: Hamiltonian) -> f64 {
        state.reverse().scalar_product(observable.of(state)).s()
    }

    /// The hopping, the unit of energy, and the three repulsions: none, moderate, strong.
    pub const HOPPING: f64 = 1.0;
    pub const REPULSIONS: [f64; 3] = [0.0, 4.0, 20.0];

    /// The six levels at a repulsion, ascending, and the gap from the singlet's block.
    pub fn spectrum(repulsion: f64) -> ([f64; 6], f64) {
        let (energies, _) = hamiltonian(HOPPING, repulsion).eigh();
        (energies, exchange_gap(HOPPING, repulsion))
    }

    /// The ground state's configuration probabilities, double occupancy and spin correlation.
    pub fn ground_state(repulsion: f64) -> ([f64; 6], f64, f64) {
        let g = ground(HOPPING, repulsion);
        (
            probabilities(g),
            expectation(g, double_occupancy()),
            expectation(g, spin_correlation()),
        )
    }

    /// Up on the left and down on the right over the exchange phases, on the timescale the gap
    /// sets.
    pub fn exchange_cycle(repulsion: f64, phases: &[f64]) -> Vec<InTime<P>> {
        let gap = exchange_gap(HOPPING, repulsion);
        phases
            .iter()
            .map(|p| exchange(HOPPING, repulsion, p / gap))
            .collect()
    }

    /// The same at the strongest repulsion, with a field differing between the sites of `ratio`
    /// times the exchange, from the Hamiltonian's states: each keeps its share of the start and
    /// turns by its own energy.
    pub fn field_exchange(ratio: f64, phases: &[f64]) -> Vec<InTime<P>> {
        let strong = REPULSIONS[2];
        let gap = exchange_gap(HOPPING, strong);
        let h = hamiltonian(HOPPING, strong) + field_difference() * (ratio * gap);
        let (energies, modes) = h.eigh();
        let shares = modes.map(|m| m * m.reverse().scalar_product(up_down()).s());
        phases
            .iter()
            .map(|p| InTime::turned(&energies, &shares, p / gap))
            .collect()
    }

    /// How much charge the ground state moves to the lower site under an asymmetry: its
    /// occupation of the left site less the right.
    pub fn charge_transfer(repulsion: f64, asymmetry: f64) -> f64 {
        let h = hamiltonian(HOPPING, repulsion) + site_difference() * asymmetry;
        let (_, modes) = h.eigh();
        -2.0 * expectation(modes[0], site_difference())
    }

    /// One cycle of the exchange phase.
    pub fn phases(count: usize) -> Vec<f64> {
        linspace(0.0, core::f64::consts::TAU, count)
    }
}

use hubbard::*;

const SECONDS: f32 = 10.0;
/// The samples of the exchange cycle, of the repulsion sweep and of the asymmetry sweep.
const CYCLE: usize = 241;
const SWEEP: usize = 121;
const ASYMMETRIES: usize = 201;
/// The field differences, in units of the exchange at the strongest repulsion.
const FIELD_RATIOS: [f64; 3] = [0.0, 1.0, 5.0];
/// The configurations an electron pair started up-down can reach, and their names.
const DIALS: [usize; 4] = [0, 2, 3, 5];
const DIAL_NAMES: [&str; 4] = ["L UP L DN", "L UP R DN", "L DN R UP", "R UP R DN"];

/// What the frames share, computed once.
struct Data {
    phases: Vec<f64>,
    cycles: Vec<Vec<InTime<P>>>,
    /// The ground state's spin correlation at each repulsion.
    correlations: Vec<f64>,
    sweep: Vec<f64>,
    levels: Vec<[f64; 6]>,
    fields: Vec<Vec<InTime<P>>>,
    asymmetries: Vec<f64>,
    transfer: Vec<Vec<f64>>,
}

fn data() -> &'static Data {
    static D: OnceLock<Data> = OnceLock::new();
    D.get_or_init(|| {
        let phases = phases(CYCLE);
        let strong = REPULSIONS[2];
        let sweep = linspace(0.0, strong, SWEEP);
        let asymmetries = linspace(0.0, 2.0 * strong, ASYMMETRIES);
        Data {
            cycles: REPULSIONS
                .iter()
                .map(|u| exchange_cycle(*u, &phases))
                .collect(),
            correlations: REPULSIONS.iter().map(|u| ground_state(*u).2).collect(),
            levels: sweep.iter().map(|u| spectrum(*u).0).collect(),
            fields: FIELD_RATIOS
                .iter()
                .map(|r| field_exchange(*r, &phases))
                .collect(),
            transfer: REPULSIONS
                .iter()
                .map(|u| {
                    asymmetries
                        .iter()
                        .map(|a| charge_transfer(*u, *a))
                        .collect()
                })
                .collect(),
            phases,
            sweep,
            asymmetries,
        }
    })
}

/// The cases' colours.
fn colours() -> [Light; 3] {
    [palette::purple(), palette::sky(), palette::orange()]
}

/// A configuration's amplitude in a state in time as a point of its dial: the cosine part
/// across, the sine part up.
fn amplitude(state: &InTime<P>, configuration: usize) -> Point2 {
    let c = configurations()[configuration];
    let read = |p: P| c.reverse().scalar_product(p).s() as f32;
    Point2::xy(read(state.cosine), read(state.sine))
}

/// One dial: the unit circle, the path of the amplitude over the cycle, and the amplitude now as
/// the triplet's share followed by the singlet block's.
fn draw_dial(c: &mut Canvas, ax: &Axes, cycle: &[InTime<P>], now: usize, configuration: usize) {
    ax.polyline(c, &circle(ORIGIN2, 1.0, 64), 1.0, palette::grid());
    let path: Vec<Point2> = cycle.iter().map(|s| amplitude(s, configuration)).collect();
    ax.polyline(c, &path, 1.0, palette::purple().faded(0.3));
    let state = cycle[now];
    let triplet = amplitude(&state.along(triplets()[1]), configuration);
    let total = amplitude(&state, configuration);
    ax.line(c, ORIGIN2, triplet, 2.0, palette::ink().faded(0.5));
    ax.line(c, triplet, total, 2.0, palette::purple());
    ax.scatter(c, &[total], Marker::Dot, 5.0, palette::ink());
}

/// The dials: one row per repulsion, one dial per configuration that can be reached.
fn draw_dials(c: &mut Canvas, area: Rect, d: &Data, now: usize) {
    let (rows, columns) = (REPULSIONS.len(), DIALS.len());
    let below = Point2::direction(0.0, 1.0);
    let cases = REPULSIONS.iter().zip(&d.cycles).zip(&d.correlations);
    for (row, ((repulsion, cycle), correlation)) in cases.enumerate() {
        let band = area.part(
            0.0,
            row as f32 / rows as f32,
            1.0,
            (row + 1) as f32 / rows as f32,
        );
        let label = band.left_middle() - Point2::direction(4.0, 0.0);
        let text = format!("U/T = {repulsion}");
        c.text(&text, label, 10.0, colours()[row], Align::Right);
        // Under it, how often both electrons share a site now, and the ground state's spin
        // correlation (-3/4 for localized spins in a singlet).
        let shared = cycle[now].expectation(double_occupancy());
        let notes = [
            format!("SHARED {shared:.2}"),
            format!("GROUND S.S {correlation:.2}"),
        ];
        for (k, note) in notes.iter().enumerate() {
            let at = label + below.gp(16.0 + 11.0 * k as f32);
            c.text(note, at, 8.0, palette::ink(), Align::Right);
        }
        for (column, configuration) in DIALS.iter().enumerate() {
            let cell = band.column(column, columns);
            // The unit circle with a margin, fitted to the cell's narrower side.
            let ax = Axes::equal(
                cell,
                ORIGIN2,
                1.15 * (cell.height() / cell.width()).max(1.0),
            );
            draw_dial(c, &ax, cycle, now, *configuration);
            if row == rows - 1 {
                let name = cell.bottom_middle() + below.gp(10.0);
                c.text(DIAL_NAMES[column], name, 8.0, palette::ink(), Align::Center);
            }
        }
    }
}

/// The six levels against the repulsion: the singlet ground state, the three triplets and the
/// two excited singlets.
fn draw_levels(c: &mut Canvas, ax: &Axes, d: &Data) {
    ax.frame(c, "LEVELS", "REPULSION / HOPPING", "ENERGY");
    let groups = [
        (0..1, palette::purple()),
        (1..4, palette::sky()),
        (4..6, palette::red()),
    ];
    for (levels, colour) in groups {
        for level in levels {
            let curve: Vec<Point2> = d
                .sweep
                .iter()
                .zip(&d.levels)
                .map(|(u, e)| Point2::xy(*u as f32, e[level] as f32))
                .collect();
            ax.polyline(c, &curve, 1.4, colour);
        }
    }
}

/// Values over the exchange cycle as points: the phase in turns of pi across.
fn on_cycle(phases: &[f64], values: impl Iterator<Item = f64>) -> Vec<Point2> {
    phases
        .iter()
        .zip(values)
        .map(|(p, v)| Point2::xy((p / core::f64::consts::PI) as f32, v as f32))
        .collect()
}

/// The swap's chance over the cycle for each case, against the spin-only prediction
/// `sin²(phase / 2)`, with markers at `now`.
fn draw_swaps(
    c: &mut Canvas,
    ax: &Axes,
    d: &Data,
    cycles: &[Vec<InTime<P>>],
    title: &str,
    now: usize,
) {
    ax.frame(c, title, "EXCHANGE PHASE / PI", "");
    let spin_only = d.phases.iter().map(|p| wave(p / 2.0).powi(2));
    ax.dashed(
        c,
        &on_cycle(&d.phases, spin_only),
        1.0,
        4.0,
        palette::ink().faded(0.5),
    );
    for (cycle, colour) in cycles.iter().zip(colours()) {
        let swap = on_cycle(&d.phases, cycle.iter().map(|s| s.probabilities()[SWAPPED]));
        ax.polyline(c, &swap, 1.5, colour);
        ax.scatter(c, &[swap[now]], Marker::Dot, 6.0, colour);
    }
}

/// The charge the ground state moves to the lower site against the asymmetry, per repulsion.
fn draw_transfer(c: &mut Canvas, ax: &Axes, d: &Data) {
    ax.frame(c, "CHARGE TRANSFER", "SITE ASYMMETRY / HOPPING", "");
    for (curve, colour) in d.transfer.iter().zip(colours()) {
        let pts: Vec<Point2> = d
            .asymmetries
            .iter()
            .zip(curve)
            .map(|(a, q)| Point2::xy(*a as f32, *q as f32))
            .collect();
        ax.polyline(c, &pts, 1.5, colour);
    }
}

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let d = data();
    let screen = c.rect();
    let now = (((t / SECONDS).fract() * CYCLE as f32) as usize).min(CYCLE - 1);
    draw_dials(c, screen.part(0.15, 0.14, 0.53, 0.9), d, now);
    let strong = REPULSIONS[2] as f32;
    let levels = Axes::new(
        screen.part(0.6, 0.16, 0.78, 0.46),
        [0.0, strong],
        [-4.5, 26.0],
    );
    draw_levels(c, &levels, d);
    let transfer = Axes::new(
        screen.part(0.84, 0.16, 0.98, 0.46),
        [0.0, 2.0 * strong],
        [-0.05, 2.05],
    );
    draw_transfer(c, &transfer, d);
    let cycle = [0.0, 2.0];
    let swaps = Axes::new(screen.part(0.6, 0.6, 0.78, 0.88), cycle, [-0.02, 1.05]);
    draw_swaps(c, &swaps, d, &d.cycles, "SPINS SWAPPED", now);
    let frozen = Axes::new(screen.part(0.84, 0.6, 0.98, 0.88), cycle, [-0.02, 1.05]);
    draw_swaps(c, &frozen, d, &d.fields, "FIELD 0, 1, 5 J", now);
    caption(
        c,
        "THE HUBBARD DIMER: FROM HOPPING TO EXCHANGE",
        "R(4,0): ORBITALS ARE VECTORS, PAIRS BIVECTORS; ARROWS ARE AMPLITUDES (COSINE, SINE)",
    );
}

fn main() {
    run(Anim::new("hubbard dimer", SECONDS).size(960, 540), draw);
}

#[cfg(test)]
#[allow(clippy::disallowed_methods)] // the two-site solution's closed forms, the references checked against
mod tests {
    use super::hubbard::*;
    use gax_numga_examples::signal::linspace;

    const ROUND_OFF: f64 = 1e-10;
    /// The strong case's bounds, numga's: they test the localized-spin approximation at a
    /// finite repulsion, not rounding.
    const STRONG_CHARGE_LIMIT: f64 = 0.025;
    const STRONG_SWAP_ERROR: f64 = 0.025;
    const EXCHANGE_RELATIVE_ERROR: f64 = 0.011;

    /// Both electrons on one site, the left and the right alike.
    fn symmetric_double() -> P {
        (pair(0, 1) + pair(2, 3)) * core::f64::consts::FRAC_1_SQRT_2
    }

    fn small(p: P) -> bool {
        p.c.iter().all(|x| x.abs() < ROUND_OFF)
    }

    /// numga's `test_hopping_product_rule_exclusion_and_interacting_spin_sectors`: the lift is
    /// the product rule on every pair of orbitals; the triplets do not move, with or without
    /// repulsion; hopping takes the singlet to the symmetric double occupancy and back.
    #[test]
    fn the_lift_is_the_product_rule_and_keeps_the_spin_sectors() {
        let single = hopping(HOPPING);
        let lifted = hamiltonian(HOPPING, 0.0);
        for first in orbitals() {
            for second in orbitals() {
                let rule = (single.of(first) ^ second) + (first ^ single.of(second));
                assert!(small(lifted.of(first ^ second) - rule));
            }
        }
        for u in REPULSIONS {
            let h = hamiltonian(HOPPING, u);
            assert!(triplets().iter().all(|t| small(h.of(*t))));
            assert!(small(
                h.of(singlet()) + symmetric_double() * (2.0 * HOPPING)
            ));
            let back =
                h.of(symmetric_double()) - symmetric_double() * u + singlet() * (2.0 * HOPPING);
            assert!(small(back));
        }
    }

    /// The repulsions of numga's sweep.
    fn sweep() -> Vec<f64> {
        linspace(0.0, REPULSIONS[2], 241)
    }

    /// numga's `test_spectrum_and_ground_state_match_the_two_site_solution`.
    #[test]
    fn the_spectrum_and_the_ground_state_are_the_two_site_solution() {
        let t = HOPPING;
        for u in sweep() {
            let separation = (u * u + 16.0 * t * t).sqrt();
            let gap = 8.0 * t * t / (separation + u);
            let doubles = (1.0 - u / separation) / 2.0;
            let (energies, from_block) = spectrum(u);
            let want = [-gap, 0.0, 0.0, 0.0, u, u + gap];
            for (e, w) in energies.iter().zip(want) {
                assert!((e - w).abs() < ROUND_OFF, "{u}: {energies:?}");
            }
            assert!((from_block - gap).abs() < ROUND_OFF);
            let (probabilities, d, correlation) = ground_state(u);
            let single = (1.0 - doubles) / 2.0;
            let want = [doubles / 2.0, 0.0, single, single, 0.0, doubles / 2.0];
            for (p, w) in probabilities.iter().zip(want) {
                assert!((p - w).abs() < ROUND_OFF, "{u}: {probabilities:?}");
            }
            assert!((d - doubles).abs() < ROUND_OFF);
            assert!((correlation + 3.0 * (1.0 - doubles) / 4.0).abs() < ROUND_OFF);
        }
    }

    /// numga's scene check: on the singlet's block (the singlet and the symmetric double
    /// occupancy) the Hamiltonian less half the repulsion, applied twice, is the rate squared.
    #[test]
    fn the_singlet_block_squares_to_its_rate() {
        for u in sweep() {
            let h = hamiltonian(HOPPING, u);
            let shifted = |p: P| h.of(p) - p * (u / 2.0);
            let rate = block_rate(HOPPING, u);
            for p in [singlet(), symmetric_double()] {
                assert!(small(shifted(shifted(p)) - p * (rate * rate)), "{u}");
            }
        }
    }

    /// numga's cycle: 1201 phases over one turn.
    fn cycle() -> Vec<f64> {
        phases(1201)
    }

    /// numga's `test_exchange_matches_free_hopping_and_interacting_charge_dynamics`: the state
    /// stays normalized; without repulsion the chances are free hopping's; with it, the double
    /// occupancy and the swap follow the singlet's closed form, its phase against the triplet's.
    #[test]
    fn the_exchange_matches_the_closed_forms() {
        let t = HOPPING;
        let phases = cycle();
        for u in REPULSIONS {
            let states = exchange_cycle(u, &phases);
            let separation = (u * u + 16.0 * t * t).sqrt();
            let gap = 8.0 * t * t / (separation + u);
            for (phase, s) in phases.iter().zip(&states) {
                let norm = s.cosine.norm_squared() + s.sine.norm_squared();
                assert!((norm - 1.0).abs() < ROUND_OFF);
                let p = s.probabilities();
                if u == 0.0 {
                    let (half_c, half_s) = ((phase / 2.0).cos(), (phase / 2.0).sin());
                    let free = [
                        phase.sin().powi(2) / 4.0,
                        0.0,
                        half_c.powi(4),
                        half_s.powi(4),
                        0.0,
                        phase.sin().powi(2) / 4.0,
                    ];
                    assert!(
                        p.iter().zip(free).all(|(a, b)| (a - b).abs() < ROUND_OFF),
                        "{p:?}"
                    );
                }
                let time = phase / gap;
                let charge = separation * time / 2.0;
                let doubles = 8.0 * t * t / (separation * separation) * charge.sin().powi(2);
                assert!((s.expectation(double_occupancy()) - doubles).abs() < ROUND_OFF);
                // The singlet's amplitude, `exp(-i U t / 2) (cos(charge) + i U / separation
                // sin(charge))`, as its real and imaginary parts.
                let (re0, im0) = ((u * time / 2.0).cos(), -(u * time / 2.0).sin());
                let (re1, im1) = (charge.cos(), u / separation * charge.sin());
                let (re, im) = (re0 * re1 - im0 * im1, re0 * im1 + im0 * re1);
                let swap = ((1.0 - re).powi(2) + im * im) / 4.0;
                assert!((p[SWAPPED] - swap).abs() < ROUND_OFF, "{u} {phase}");
            }
        }
    }

    /// numga's `test_large_repulsion_approaches_localized_spin_exchange`: at the strongest
    /// repulsion the charge hardly moves, the swap follows the spin-only `sin²(phase / 2)`, and
    /// the gap is near `4 t² / U`.
    #[test]
    fn a_strong_repulsion_approaches_localized_spins() {
        let phases = cycle();
        let strong = REPULSIONS[2];
        let states = exchange_cycle(strong, &phases);
        let doubles = states
            .iter()
            .map(|s| s.expectation(double_occupancy()))
            .fold(0.0, f64::max);
        assert!(doubles < STRONG_CHARGE_LIMIT, "{doubles}");
        let error = phases
            .iter()
            .zip(&states)
            .map(|(p, s)| (s.probabilities()[SWAPPED] - (p / 2.0).sin().powi(2)).abs())
            .fold(0.0, f64::max);
        assert!(error < STRONG_SWAP_ERROR, "{error}");
        let perturbative = 4.0 * HOPPING * HOPPING / strong;
        let (_, gap) = spectrum(strong);
        assert!((perturbative / gap - 1.0).abs() < EXCHANGE_RELATIVE_ERROR);
    }

    /// numga's `test_the_dials_draw` draws the dials of three steps; here a frame of them draws
    /// at three times.
    #[test]
    fn the_dials_draw() {
        for t in [0.5, 3.0, 7.0] {
            gax_numga_examples::app::assert_draws(super::draw, t);
        }
    }

    /// numga's `test_a_field_difference_keeps_up_down_and_freezes_the_swap`: up-down is a state
    /// of the field difference; without it the swap completes, at five times the exchange it is
    /// frozen.
    #[test]
    fn a_field_difference_freezes_the_swap() {
        assert!(small(field_difference().of(up_down()) - up_down()));
        let phases = cycle();
        let swaps: Vec<f64> = [0.0, 5.0]
            .iter()
            .map(|ratio| {
                let states = field_exchange(*ratio, &phases);
                for s in &states {
                    let total: f64 = s.probabilities().iter().sum();
                    assert!((total - 1.0).abs() < ROUND_OFF);
                }
                states
                    .iter()
                    .map(|s| s.probabilities()[SWAPPED])
                    .fold(0.0, f64::max)
            })
            .collect();
        assert!(swaps[0] > 0.99 && swaps[1] < 0.05, "{swaps:?}");
    }

    /// numga's `test_strong_repulsion_holds_one_electron_per_site_until_the_asymmetry_outweighs_it`.
    #[test]
    fn a_strong_repulsion_holds_one_electron_per_site() {
        let strong = REPULSIONS[2];
        let transfer = |u: f64, a: f64| charge_transfer(u, a * strong);
        for u in REPULSIONS {
            assert!(transfer(u, 0.0).abs() < ROUND_OFF);
        }
        assert!(transfer(0.0, 0.5) > 1.9);
        assert!(transfer(strong, 0.5) < 0.1);
        assert!(transfer(strong, 2.0) > 1.9);
    }
}
