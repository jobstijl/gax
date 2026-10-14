//! numga's `relativity/spinors`: Dirac, Weyl and Majorana states in one real spinor space, the
//! even multivectors of the spacetime algebra (STA). A spinor `ψ` carries the observer's frame
//! to its own: the image of the time axis, `ψ t ψ̃`, is its current, that of the `z` axis its
//! spin, and the current's time part the density the observer measures.
//!
//! The spinor operations are maps on the same space, each right multiplication by a bivector
//! with the spinor left open (`Even::slot() * b`): the phase by `y x`, the chirality by `z t`,
//! charge conjugation by `y t`. A Lorentz rotor acts on the left, so every right action commutes
//! with it. The chirality squares to one, and its projectors `(1 ± Γ) / 2` split a spinor into
//! its right- and left-handed (Weyl) parts, whose currents are null; their sum is the Dirac
//! current, their difference the spin. Charge conjugation also squares to one and anticommutes
//! with the phase: it reverses the phase instead of commuting with it, and its fixed spinors,
//! `(1 + C) / 2` of any spinor, are the Majorana states.
//!
//! The animation: a spinor turned through two full turns (left), its spin arrow going round
//! twice while the two outputs of an interferometer against the unturned spinor trade their
//! light once, the spinor changing sign after one turn; the Weyl currents of a boosted spinor on
//! the light cone (middle), the boost sweeping; and the densities of a Majorana state in the two
//! eigenspaces of charge conjugation as a phase turns it (right).

use gax_numga_examples::{
    Align, Anim, Axes, Canvas, Light, Marker, ORIGIN2, Point2, Rect, backdrop, caption, palette,
    run,
    signal::{phasor, wave},
};

mod spinors {
    use gax::sta::{Bivector, Even, Vector};
    use gax_numga_examples::signal::phasor;

    pub type V = Vector<(), f64>;
    pub type B = Bivector<(), f64>;
    /// A Dirac spinor is an even multivector.
    pub type Spinor = Even<(), f64>;
    /// A linear map on spinors.
    pub type SpinorMap = Even<(Even,), f64>;

    /// The observer's time axis.
    pub fn t() -> V {
        Vector::new(1.0, 0.0, 0.0, 0.0)
    }

    pub fn x() -> V {
        Vector::new(0.0, 1.0, 0.0, 0.0)
    }

    pub fn y() -> V {
        Vector::new(0.0, 0.0, 1.0, 0.0)
    }

    /// The spin axis.
    pub fn z() -> V {
        Vector::new(0.0, 0.0, 0.0, 1.0)
    }

    /// The unit spinor, the identity frame.
    pub fn one() -> Spinor {
        Spinor::from(gax::sta::Scalar::new(1.0))
    }

    /// Right multiplication of the open spinor by a bivector.
    fn times(b: B) -> SpinorMap {
        Even::slot() * b
    }

    /// The phase: right multiplication by `y x`, which squares to minus one.
    pub fn phase() -> SpinorMap {
        times(y() ^ x())
    }

    /// The chirality: right multiplication by `z t`, which squares to one.
    pub fn chirality() -> SpinorMap {
        times(z() ^ t())
    }

    /// Charge conjugation: right multiplication by `y t`, which squares to one.
    pub fn charge_conjugation() -> SpinorMap {
        times(y() ^ t())
    }

    /// The projector onto right-handed spinors, `(1 + Γ) / 2`.
    pub fn right_handed() -> SpinorMap {
        (Even::slot() + chirality()) * 0.5
    }

    /// The projector onto left-handed spinors, `(1 - Γ) / 2`.
    pub fn left_handed() -> SpinorMap {
        (Even::slot() - chirality()) * 0.5
    }

    /// The projector onto Majorana spinors, those charge conjugation fixes, `(1 + C) / 2`.
    pub fn majorana() -> SpinorMap {
        (Even::slot() + charge_conjugation()) * 0.5
    }

    /// The future current: the observer's time axis carried by the spinor.
    pub fn current(psi: Spinor) -> V {
        psi >> t()
    }

    /// The spin current: the spin axis carried by the spinor.
    pub fn spin(psi: Spinor) -> V {
        psi >> z()
    }

    /// The probability density the observer measures: the current's part along the time axis.
    pub fn density(psi: Spinor) -> f64 {
        (current(psi) | t()).s()
    }

    /// The intensity at one output of an interferometer recombining two equal paths, one with
    /// the spinor and one with a reference: the density of their mean.
    pub fn interference(psi: Spinor, reference: Spinor) -> f64 {
        density((psi + reference) * 0.5)
    }

    // --- the scenes -------------------------------------------------------------------

    /// The rapidity of the boost whose Weyl currents are shown, at its largest.
    pub const RAPIDITY: f64 = 0.7;

    /// A spinor turned by `angle` in the `xz` plane: its spin, and the intensities at the sum
    /// and difference outputs of an interferometer against the unturned spinor. The spin comes
    /// back after one turn; the outputs only after two.
    pub fn rotation(angle: f64) -> (V, [f64; 2]) {
        let psi = ((x() ^ z()) * (angle / 2.0)).exp().into_inner();
        let ports = [interference(psi, one()), interference(psi, -one())];
        (spin(psi), ports)
    }

    /// The currents of the left- and right-handed parts of a spinor boosted along `z` by the
    /// rapidity, and the Dirac current of the whole: `[left, right, Dirac]`.
    pub fn chiral_currents(rapidity: f64) -> [V; 3] {
        let psi = ((t() ^ z()) * (-rapidity / 2.0)).exp().into_inner();
        [
            current(left_handed().of(psi)),
            current(right_handed().of(psi)),
            current(psi),
        ]
    }

    /// The densities of a Majorana state turned by a phase `angle`, in the two eigenspaces of
    /// charge conjugation (eigenvalues `+1` and `-1`), as shares of its own density: a phase
    /// turn trades density between them.
    pub fn majorana_shares(angle: f64) -> [f64; 2] {
        let psi = majorana().of(one());
        // The phase map squares to minus one: its exponential is `cos + sin` times it.
        let turn = phasor(angle);
        let phased = psi * turn.e20() + phase().of(psi) * turn.e01();
        let conjugated = charge_conjugation().of(phased);
        let total = density(psi);
        [
            density((phased + conjugated) * 0.5) / total,
            density((phased - conjugated) * 0.5) / total,
        ]
    }
}

use spinors::*;

/// One loop: the spinor turns twice, `4π`.
const SECONDS: f32 = 8.0;
/// Samples of the Majorana curves over a full phase turn.
const PHASE_SAMPLES: usize = 129;

/// A spacetime vector as a drawing point: its `across` component across, `up` component up.
fn in_plane(v: V, across: fn(&V) -> f64, up: fn(&V) -> f64) -> Point2 {
    Point2::xy(across(&v) as f32, up(&v) as f32)
}

/// The colours of the left-handed, right-handed and Dirac currents.
fn hands() -> [Light; 3] {
    [palette::orange(), palette::sky(), palette::ink()]
}

/// A closed circle about `centre`, as a polygon.
fn circle(centre: Point2, radius: f32) -> Vec<Point2> {
    (0..=96)
        .map(|k| centre + phasor(core::f32::consts::TAU * k as f32 / 96.0).gp(radius))
        .collect()
}

/// A panel's title, centred above it.
fn title(c: &mut Canvas, panel: Rect, text: &str, size: f32) {
    let at = panel.top_middle() - Point2::direction(0.0, size * 0.8);
    c.text(text, at, size, palette::ink(), Align::Center);
}

/// The spin in the `xz` plane as an arrow on the unit circle, and the interferometer's two
/// outputs as disks lit by their intensity.
fn draw_rotation(c: &mut Canvas, rect: Rect, angle: f64, size: f32) {
    let (spin, ports) = rotation(angle);
    let disc = rect.part(0.0, 0.0, 1.0, 0.66);
    let ax = Axes::equal(disc, ORIGIN2, 1.3);
    let reach = 1.2;
    let axes = [
        [Point2::xy(-reach, 0.0), Point2::xy(reach, 0.0)],
        [Point2::xy(0.0, -reach), Point2::xy(0.0, reach)],
    ];
    ax.stroke(c, &axes, 1.0, palette::grid());
    ax.text(
        c,
        Point2::xy(reach, 0.06),
        "X",
        size * 0.8,
        palette::grid().faded(4.0),
        Align::Right,
    );
    ax.text(
        c,
        Point2::xy(0.06, reach - 0.1),
        "Z",
        size * 0.8,
        palette::grid().faded(4.0),
        Align::Left,
    );
    ax.polyline(c, &circle(ORIGIN2, 1.0), 1.2, palette::grid().faded(2.0));
    let tip = ORIGIN2 + in_plane(spin, V::e1, V::e3);
    ax.arrow(c, ORIGIN2, tip, 2.4, 10.0, palette::yellow());
    let turns = angle / core::f64::consts::PI;
    title(c, rect, &format!("SPIN, TURNED {turns:.2} PI"), size);

    let row = rect.part(0.0, 0.7, 1.0, 1.0);
    for (k, (intensity, label)) in ports
        .iter()
        .zip(["SUM PORT", "DIFFERENCE PORT"])
        .enumerate()
    {
        let cell = row.column(k, 2);
        let centre = cell.centre() - Point2::direction(0.0, size * 0.6);
        let radius = cell.height() * 0.24;
        c.ring(centre, radius + 3.0, 1.0, palette::grid().faded(2.0));
        // The port's light covers the dark disk by its intensity.
        let disk = circle(centre, radius);
        c.fill(&disk, palette::sky(), intensity.clamp(0.0, 1.0) as f32);
        let below = centre + Point2::direction(0.0, radius + size * 1.5);
        c.text(label, below, size * 0.75, palette::ink(), Align::Center);
    }
}

/// The light cone in the `z t` plane, with the Weyl currents on it and the Dirac current inside.
fn draw_chirality(c: &mut Canvas, rect: Rect, rapidity: f64, size: f32) {
    let extent = 1.45;
    let ax = Axes::new(
        rect.inset(30.0, 0.0, 6.0, 30.0),
        [-extent, extent],
        [0.0, extent],
    );
    let cone = [
        Point2::xy(-extent, extent),
        ORIGIN2,
        Point2::xy(extent, extent),
    ];
    ax.fill(c, &cone, palette::grid(), 0.25);
    ax.polyline(c, &cone, 1.2, palette::grid().faded(2.0));
    let names = ["LEFT", "RIGHT", "DIRAC"];
    for ((current, colour), name) in chiral_currents(rapidity).iter().zip(hands()).zip(names) {
        let tip = in_plane(*current, V::e3, V::e0);
        ax.arrow(c, ORIGIN2, tip, 2.2, 10.0, colour);
        let label = ax.px(tip) + Point2::direction(size * 0.6, -size * 0.4);
        c.text(name, label, size * 0.8, colour, Align::Left);
    }
    ax.frame(c, "", "Z CURRENT", "");
    let readout = format!("RAPIDITY {rapidity:+.2}");
    let inside = ax.rect.lo + Point2::direction(size * 0.6, size * 1.6);
    c.text(&readout, inside, size * 0.8, palette::ink(), Align::Left);
    title(c, rect, "WEYL CURRENTS (Z, T)", size);
}

/// The Majorana state's densities in the two eigenspaces of charge conjugation over a phase
/// turn, with markers at the current phase.
fn draw_majorana(c: &mut Canvas, rect: Rect, angle: f64, size: f32) {
    // Room above the curves for the legend.
    let ax = Axes::new(rect.inset(34.0, 0.0, 10.0, 30.0), [0.0, 2.0], [0.0, 1.3]);
    let at =
        |phase: f64, share: f64| Point2::xy((phase / core::f64::consts::PI) as f32, share as f32);
    let phases: Vec<f64> = (0..PHASE_SAMPLES)
        .map(|k| core::f64::consts::TAU * k as f64 / (PHASE_SAMPLES - 1) as f64)
        .collect();
    let colours = [palette::sky(), palette::orange()];
    let shares: Vec<[f64; 2]> = phases.iter().map(|p| majorana_shares(*p)).collect();
    let now = majorana_shares(angle);
    for (k, colour) in colours.iter().enumerate() {
        let curve: Vec<Point2> = phases
            .iter()
            .zip(&shares)
            .map(|(p, s)| at(*p, s[k]))
            .collect();
        ax.polyline(c, &curve, 1.8, *colour);
        ax.scatter(c, &[at(angle, now[k])], Marker::Dot, 7.0, *colour);
    }
    ax.frame(c, "", "PHASE / PI", "");
    ax.legend(
        c,
        &[
            ("CONJUGATION +1", colours[0]),
            ("CONJUGATION -1", colours[1]),
        ],
    );
    title(c, rect, "MAJORANA DENSITIES", size);
}

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let screen = c.rect();
    let (w, h) = (screen.width(), screen.height());
    let size = (h / 42.0).clamp(6.0, 12.0);
    let loop_phase = f64::from(t / SECONDS) * core::f64::consts::TAU;
    let row = Rect::new(w * 0.01, h * 0.24, w * 0.99, h * 0.95);
    let panel = |i: usize| row.column(i, 3).inset(w * 0.012, 0.0, w * 0.012, 0.0);
    draw_rotation(c, panel(0), 2.0 * loop_phase, size);
    draw_chirality(c, panel(1), RAPIDITY * wave(loop_phase), size);
    draw_majorana(c, panel(2), loop_phase, size);
    caption(
        c,
        "SPINORS: DIRAC, WEYL AND MAJORANA STATES",
        "STA EVEN MULTIVECTORS: A SPINOR NEEDS TWO TURNS, CHIRAL HALVES ARE NULL",
    );
}

fn main() {
    run(Anim::new("spinors", SECONDS).size(960, 540), draw);
}

#[cfg(test)]
mod tests {
    use super::spinors::*;
    use gax::ApproxEq;
    use gax::sta::Even;
    use gax_numga_examples::rng::{Draw, Rng, rng};

    const TOLERANCE: f64 = 1e-11;
    const SAMPLES: usize = 24;

    /// Spinors with normal coefficients of standard deviation 0.3.
    fn spinors(rng: &mut Rng) -> Vec<Spinor> {
        (0..SAMPLES)
            .map(|_| Even::from_coeffs(core::array::from_fn(|_| rng.normal() * 0.3)))
            .collect()
    }

    /// The rotor `exp(b angle)`.
    fn turn(b: B, angle: f64) -> Spinor {
        (b * angle).exp().into_inner()
    }

    #[test]
    fn spinor_observables_transform_with_lorentz_rotors_and_ignore_phase() {
        let mut rng = rng(931);
        for psi in spinors(&mut rng) {
            let rotor = turn(t() ^ x(), rng.range(-0.4, 0.4))
                * turn(y() ^ z(), rng.range(-0.6, 0.6))
                * turn(x() ^ y(), rng.range(-0.6, 0.6));
            let phase = turn(y() ^ x(), rng.range(-1.0, 1.0));
            let (moved, phased) = (rotor * psi, psi * phase);
            // The spinor transforms on one side; its vector observables follow the spacetime
            // rotation and boost. A right phase leaves them unchanged.
            for observable in [current, spin] {
                let carried = rotor >> observable(psi);
                assert!(observable(moved).max_abs_diff(&carried) < TOLERANCE);
                assert!(observable(phased).max_abs_diff(&observable(psi)) < TOLERANCE);
            }
        }
    }

    #[test]
    fn weyl_projections_split_the_current_into_two_covariant_null_currents() {
        let mut rng = rng(932);
        let rotor = turn(t() ^ x(), 0.27) * turn(y() ^ z(), -0.38);
        let (right, left) = (right_handed(), left_handed());
        for psi in spinors(&mut rng) {
            let (r, l) = (right.of(psi), left.of(psi));
            // Complementary projections commute with every left Lorentz action.
            assert!((r + l).max_abs_diff(&psi) < TOLERANCE);
            assert!(right.of(r).max_abs_diff(&r) < TOLERANCE);
            assert!(left.of(l).max_abs_diff(&l) < TOLERANCE);
            assert!(right.of(l).max_abs_diff(&Even::zero()) < TOLERANCE);
            assert!(left.of(r).max_abs_diff(&Even::zero()) < TOLERANCE);
            assert!(right.of(rotor * psi).max_abs_diff(&(rotor * r)) < TOLERANCE);
            assert!(left.of(rotor * psi).max_abs_diff(&(rotor * l)) < TOLERANCE);
            // Each current is future null, while their sum is the Dirac current and their
            // difference is the spin current.
            let (right_current, left_current) = (current(r), current(l));
            assert!(right_current.norm_squared().abs() < TOLERANCE);
            assert!(left_current.norm_squared().abs() < TOLERANCE);
            let sum = right_current + left_current;
            assert!(sum.max_abs_diff(&current(psi)) < TOLERANCE);
            let difference = right_current - left_current;
            assert!(difference.max_abs_diff(&spin(psi)) < TOLERANCE);
            assert!((right_current | t()).s() > 0.0 && (left_current | t()).s() > 0.0);
            assert!(current(psi).norm_squared() > 0.0);
        }
    }

    #[test]
    fn charge_conjugation_exchanges_chirality_and_reverses_the_phase() {
        let mut rng = rng(933);
        let conjugate = charge_conjugation();
        let phase = turn(y() ^ x(), 0.43);
        let rotor = turn(t() ^ y(), -0.32) * turn(x() ^ z(), 0.24);
        let (mut generic, mut phased_majorana) = (0.0f64, 0.0f64);
        for psi in spinors(&mut rng) {
            let fixed = majorana().of(psi);
            // Charge conjugation is an involution, commutes with Lorentz action, and reverses
            // the chosen phase rather than being linear over that phase.
            assert!(conjugate.of(conjugate.of(psi)).max_abs_diff(&psi) < TOLERANCE);
            let moved = conjugate.of(rotor * psi);
            assert!(moved.max_abs_diff(&(rotor * conjugate.of(psi))) < TOLERANCE);
            let anticommuted = conjugate.of(super::spinors::phase().of(psi))
                + super::spinors::phase().of(conjugate.of(psi));
            assert!(anticommuted.max_abs_diff(&Even::zero()) < TOLERANCE);
            let reversed = conjugate.of(psi) * phase.reverse();
            assert!(conjugate.of(psi * phase).max_abs_diff(&reversed) < TOLERANCE);
            let swapped = left_handed().of(conjugate.of(psi));
            assert!(conjugate.of(right_handed().of(psi)).max_abs_diff(&swapped) < TOLERANCE);
            // These commuting Majorana states are fixed by conjugation and carry a null
            // current. Real coefficients alone do not impose this constraint; a generic phase
            // also takes a fixed state out of the fixed subspace.
            assert!(conjugate.of(fixed).max_abs_diff(&fixed) < TOLERANCE);
            assert!(current(fixed).norm_squared().abs() < TOLERANCE);
            assert!(spin(fixed).max_abs_diff(&V::zero()) < TOLERANCE);
            generic = generic.max(conjugate.of(psi).max_abs_diff(&psi));
            let turned = fixed * phase;
            phased_majorana = phased_majorana.max(conjugate.of(turned).max_abs_diff(&turned));
        }
        assert!(generic > 0.1 && phased_majorana > 0.1);
    }

    /// The scenes, against numga's numbers: two turns bring the spinor back, one turn negates
    /// it (the sum port dark, the spin as it was), a quarter turn takes the spin from `z` to
    /// `-x`; the Weyl currents of the boost are null and add to the Dirac current
    /// `cosh(0.7) t + sinh(0.7) z`; the Majorana state's two shares make up its density, all of
    /// it in `+1` at no phase and in `-1` at a quarter turn.
    #[test]
    fn scenes_turn_twice_and_split_their_densities() {
        let pi = core::f64::consts::PI;
        let (start, ports) = rotation(0.0);
        assert!((ports[0] - 1.0).abs() < TOLERANCE && ports[1].abs() < TOLERANCE);
        let (quarter, _) = rotation(pi / 2.0);
        assert!(quarter.max_abs_diff(&-x()) < TOLERANCE);
        let (one_turn, ports) = rotation(2.0 * pi);
        assert!(one_turn.max_abs_diff(&start) < TOLERANCE);
        assert!(ports[0].abs() < TOLERANCE && (ports[1] - 1.0).abs() < TOLERANCE);
        let [left, right, dirac] = chiral_currents(RAPIDITY);
        assert!(left.norm_squared().abs() < TOLERANCE && right.norm_squared().abs() < TOLERANCE);
        assert!((left + right).max_abs_diff(&dirac) < TOLERANCE);
        let numga = t() * 1.25516901 + z() * 0.7585837;
        assert!(dirac.max_abs_diff(&numga) < 1e-8);
        for k in 0..16 {
            let [fixed, negated] = majorana_shares(pi * k as f64 / 8.0);
            assert!((fixed + negated - 1.0).abs() < TOLERANCE);
        }
        let [fixed, _] = majorana_shares(0.0);
        let [_, negated] = majorana_shares(pi / 2.0);
        assert!((fixed - 1.0).abs() < TOLERANCE && (negated - 1.0).abs() < TOLERANCE);
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws(super::draw, 0.5);
    }
}
