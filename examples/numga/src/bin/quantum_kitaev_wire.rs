//! numga's `quantum/kitaev_wire`: Majorana modes at the ends of a gated Kitaev wire, and moving
//! one by moving a gate. Each site of the wire holds two real Majorana amplitudes, the two
//! coefficients of a VGA2D vector, and the wire's state is a field of them. Hopping and pairing
//! couple neighbouring sites through map cells `Vector <- Vector` (a cell reads one component
//! at the next site and writes the other here), and the chemical potential turns each site's
//! vector in its plane, by the commutator with `xy`. The sum is a skew field map, the
//! generator: it is the action, on vectors, of one bivector of the Clifford algebra of all the
//! Majoranas, so time turns the field by a rotor of that big algebra.
//!
//! The excitation energies are the singular values of the generator, from the sparse
//! eigenproblem of its transpose times itself. Where the chemical potential is below twice the
//! hopping the wire is topological, and a gate that lowers it between two boundaries leaves a
//! pair of zero modes, one at each boundary; mixing the pair by the site index's eigenvectors
//! (a symmetric map on the plane of the two modes) separates them. The Cayley step, the
//! sparse LU of `1 - G dt / 2`, keeps the field's norm exactly.
//!
//! The animation moves the left gate boundary inward, slowly and fast: slowly, the mode rides
//! along with the boundary (it stays the instantaneous boundary mode); fast, it is left behind
//! in the bulk. Beside it, the spectrum as the gate's potential is lowered past the threshold
//! (two energies drop to zero), and the splitting of the boundary modes as the segment
//! shortens, exponentially small while they do not overlap.

use gax_numga_examples::signal::linspace;
use gax_numga_examples::{
    Align, Anim, Axes, Canvas, Light, Marker, Point2, backdrop, caption, palette, run,
};
use std::sync::OnceLock;

mod kitaev {
    use gax::field::{Field, SparseMap};
    use gax::vga2d::{Pseudoscalar, Scalar, Vector};
    use gax_numga_examples::par_map;
    use gax_numga_examples::signal::linspace;

    /// Two Majorana amplitudes, a map on them, and a field of them over the wire's sites.
    pub type Majorana = Vector<(), f64>;
    pub type Map = Vector<(Vector,), f64>;
    pub type Modes = Field<Majorana>;
    /// The couplings between sites, and the chemical potential at each.
    pub type Generator = SparseMap<Map>;
    pub type Potential = Field<Scalar<(), f64>>;

    fn x() -> Majorana {
        Vector::new(1.0, 0.0)
    }
    fn y() -> Majorana {
        Vector::new(0.0, 1.0)
    }
    fn xy() -> Pseudoscalar<(), f64> {
        Pseudoscalar::new(1.0)
    }
    fn open() -> Map {
        Vector::slot()
    }

    /// A chain of sites with its bonds: the couplings between neighbours.
    pub struct Wire {
        bonds: Generator,
    }

    impl Wire {
        /// Each bond's cell reads a Majorana component on the next site and writes the other on
        /// this one; reversing a coupling reverses its sign, so the bonds are skew.
        pub fn chain(sites: usize, hopping: f64, pairing: f64) -> Wire {
            let cell = y() * (x() | open()) * (hopping + pairing)
                + x() * (y() | open()) * (pairing - hopping);
            let rows = (0..sites - 1).collect();
            let cols = (1..sites).collect();
            let forward = SparseMap::new(vec![cell; sites - 1], rows, cols, (sites, sites));
            let back = forward.transpose().map_cells(|c| -c);
            Wire {
                bonds: &forward + &back,
            }
        }

        fn sites(&self) -> usize {
            self.bonds.shape().0
        }

        /// The identity on the wire's sites.
        fn identity(&self) -> Generator {
            SparseMap::from_diagonal(&Field::from_fn(self.sites(), |_| open()))
        }

        /// The bonds plus each site's turning by its chemical potential.
        pub fn generator(&self, potential: &Potential) -> Generator {
            let local = potential.map(|mu| xy().commutator(open()) * -mu.s());
            &self.bonds + &SparseMap::from_diagonal(&local)
        }

        /// The `count` lowest excitation energies and their real, orthonormal quadratures: the
        /// eigenvectors of the generator's transpose times itself, each energy the norm of the
        /// generator's image of its mode (no square root of an eigenvalue that rounding made
        /// slightly negative).
        pub fn modes(&self, generator: &Generator, count: usize) -> (Vec<f64>, Vec<Modes>) {
            let normal = generator.transpose().compose(generator);
            let (_, modes) = normal.eigh(&self.identity(), count);
            let energies = modes.iter().map(|m| generator.apply(m).norm()).collect();
            (energies, modes)
        }

        /// The lowest two modes mixed into the two most localized: the plane of the two as
        /// VGA2D's vectors, the mean site index between them a symmetric map on it, and its
        /// eigenvectors the mixings.
        pub fn localized(&self, modes: &[Modes]) -> [Modes; 2] {
            let index = |a: &Modes, b: &Modes| -> f64 {
                a.iter()
                    .zip(b.iter())
                    .enumerate()
                    .map(|(site, (u, v))| site as f64 * (u | v).s())
                    .sum()
            };
            let (a, b) = (&modes[0], &modes[1]);
            let position = Map::from_images([
                Vector::new(index(a, a), index(b, a)),
                Vector::new(index(a, b), index(b, b)),
            ]);
            let (_, mixings) = position.eigh();
            mixings.map(|m| &a.scaled(m.e1()) + &b.scaled(m.e2()))
        }

        /// The Cayley step over `dt`, with the generator at the step's midpoint: the sparse LU of
        /// `1 - G dt / 2` solved against `(1 + G dt / 2) state`.
        pub fn step(&self, state: &Modes, potential: &Potential, dt: f64) -> Modes {
            let half = self.generator(potential).map_cells(|c| c * (dt / 2.0));
            let behind = &self.identity() + &half.map_cells(|c| -c);
            behind.solve_direct(&(state + &half.apply(state)))
        }
    }

    /// A smooth well of chemical potential between two gate boundaries, `inside` within and
    /// `outside` beyond, each boundary a tanh of `width` sites.
    pub fn gate(
        sites: usize,
        left: f64,
        right: f64,
        inside: f64,
        outside: f64,
        width: f64,
    ) -> Potential {
        Field::from_fn(sites, |s| {
            let s = s as f64;
            let window = (((s - left) / width).tanh() - ((s - right) / width).tanh()) / 2.0;
            Scalar::new(outside + (inside - outside) * window)
        })
    }

    /// How much of a state lies in a pair of modes.
    pub fn boundary_share(modes: &[Modes], state: &Modes) -> f64 {
        modes.iter().map(|m| state.dot(m).powi(2)).sum()
    }

    /// A gate's ramp with no speed and no acceleration at either end.
    pub fn smooth_progress(progress: f64) -> f64 {
        progress.powi(3) * (10.0 + progress * (-15.0 + 6.0 * progress))
    }

    /// A state taken through the gate profiles, one Cayley step each over `duration`: the state,
    /// then every `stride` steps.
    pub fn transport(
        wire: &Wire,
        state: &Modes,
        potentials: &[Potential],
        duration: f64,
        stride: usize,
    ) -> Vec<Modes> {
        let dt = duration / potentials.len() as f64;
        let mut current = state.clone();
        let mut out = vec![current.clone()];
        for block in potentials.chunks(stride) {
            for potential in block {
                current = wire.step(&current, potential, dt);
            }
            out.push(current.clone());
        }
        out
    }

    /// The wire and its gate, numga's: sites, energies in the hopping, the gate's potentials
    /// inside and outside, its boundaries' width, places and the left one's destination.
    pub const SITES: usize = 96;
    pub const HOPPING: f64 = 1.0;
    pub const PAIRING: f64 = 0.8;
    pub const INSIDE: f64 = 1.4;
    pub const OUTSIDE: f64 = 3.5;
    pub const WIDTH: f64 = 1.2;
    pub const LEFT: f64 = 18.0;
    pub const RIGHT: f64 = 82.0;
    pub const DESTINATION: f64 = 48.0;
    /// The slow and the fast transport's durations, its steps and the steps per record.
    pub const DURATIONS: [f64; 2] = [180.0, 12.0];
    pub const STEPS: usize = 720;
    pub const STRIDE: usize = 6;

    fn wire() -> Wire {
        Wire::chain(SITES, HOPPING, PAIRING)
    }

    /// The standard gate with the given potential inside and left boundary.
    fn gated(inside: f64, left: f64) -> Potential {
        gate(SITES, left, RIGHT, inside, OUTSIDE, WIDTH)
    }

    /// The spectrum as the potential inside the gate is lowered from outside's to `INSIDE`: the
    /// potentials, and the lowest `count` energies at each.
    pub fn formation(samples: usize, count: usize) -> (Vec<f64>, Vec<Vec<f64>>) {
        let wire = wire();
        let potentials = linspace(OUTSIDE, INSIDE, samples);
        let energies = par_map(&potentials, |mu| {
            wire.modes(&wire.generator(&gated(*mu, LEFT)), count).0
        });
        (potentials, energies)
    }

    /// The gate profile when the left boundary has gone `progress` of its way.
    pub fn gate_at(progress: f64) -> Potential {
        gated(
            INSIDE,
            LEFT + (DESTINATION - LEFT) * smooth_progress(progress),
        )
    }

    /// The left boundary's mode carried along as the boundary moves, slowly and fast.
    pub struct Moving {
        /// The fraction of the gate's way at each record, and the profile there.
        pub progress: Vec<f64>,
        pub potentials: Vec<Potential>,
        /// The carried state at each record, slow then fast.
        pub history: [Vec<Modes>; 2],
        /// The instantaneous left boundary mode at each record, and the carried states' shares
        /// of the instantaneous pair of boundary modes.
        pub target: Vec<Modes>,
        pub share: [Vec<f64>; 2],
    }

    pub fn moving() -> Moving {
        let wire = wire();
        let progress: Vec<f64> = (0..=STEPS / STRIDE)
            .map(|k| (k * STRIDE) as f64 / STEPS as f64)
            .collect();
        let potentials: Vec<Potential> = progress.iter().map(|p| gate_at(*p)).collect();
        let pairs = par_map(&potentials, |p| wire.modes(&wire.generator(p), 2).1);
        let target: Vec<Modes> = pairs
            .iter()
            .map(|pair| {
                let [left, _] = wire.localized(pair);
                left
            })
            .collect();
        let midpoints: Vec<Potential> = (0..STEPS)
            .map(|k| gate_at((k as f64 + 0.5) / STEPS as f64))
            .collect();
        let history = par_map(&DURATIONS, |d| {
            transport(&wire, &target[0], &midpoints, *d, STRIDE)
        });
        let history: [Vec<Modes>; 2] = history.try_into().expect("two durations");
        let share = [0, 1].map(|case| {
            pairs
                .iter()
                .zip(&history[case])
                .map(|(pair, state)| boundary_share(pair, state))
                .collect()
        });
        Moving {
            progress,
            potentials,
            history,
            target,
            share,
        }
    }

    /// The lowest excitation energy as the segment shortens to each separation, its right
    /// boundary fixed: the mean of the boundary pair's two quadratures.
    pub fn overlap(separations: &[f64]) -> Vec<f64> {
        let wire = wire();
        par_map(separations, |s| {
            let (energies, _) = wire.modes(&wire.generator(&gated(INSIDE, RIGHT - s)), 2);
            (energies[0] + energies[1]) / 2.0
        })
    }
}

use kitaev::*;

const SECONDS: f32 = 12.0;
/// The spectrum's samples and energies, and the separations of the overlap scan.
const SPECTRAL_SAMPLES: usize = 100;
const ENERGIES: usize = 8;
const SEPARATIONS: (f64, f64, usize) = (4.0, 32.0, 100);

/// What the frames share, computed once.
struct Data {
    potentials: Vec<f64>,
    spectrum: Vec<Vec<f64>>,
    moving: Moving,
    /// The largest weight of a mode on a site, over the whole transport.
    peak: f64,
    separations: Vec<f64>,
    splitting: Vec<f64>,
}

fn data() -> &'static Data {
    static D: OnceLock<Data> = OnceLock::new();
    D.get_or_init(|| {
        let (potentials, spectrum) = formation(SPECTRAL_SAMPLES, ENERGIES);
        let (from, to, count) = SEPARATIONS;
        let separations = linspace(from, to, count);
        let splitting = overlap(&separations);
        let moving = moving();
        let peak = moving
            .history
            .iter()
            .flatten()
            .chain(&moving.target)
            .flat_map(|f| f.iter().map(|v| v.norm_squared()))
            .fold(0.0, f64::max);
        Data {
            potentials,
            spectrum,
            moving,
            peak,
            separations,
            splitting,
        }
    })
}

/// The slow and the fast transport's colours and names.
fn cases() -> [(Light, &'static str); 2] {
    [(palette::green(), "SLOW"), (palette::orange(), "FAST")]
}

/// How deep a chemical potential lies in the topological regime: zero at the threshold, twice
/// the hopping, one at the gate's lowest potential.
fn depth(mu: f64) -> f32 {
    ((2.0 * HOPPING - mu) / (2.0 * HOPPING - INSIDE)).clamp(0.0, 1.0) as f32
}

/// The topological stretch of the wire, shaded by its depth: a column per site.
fn draw_gate(c: &mut Canvas, ax: &Axes, potential: &Potential) {
    let [lo, hi] = ax.y;
    for (s, mu) in potential.iter().enumerate() {
        let shade = depth(mu.s());
        if shade > 0.0 {
            let (a, b) = (s as f32 - 0.5, s as f32 + 0.5);
            let column = [
                Point2::xy(a, lo),
                Point2::xy(b, lo),
                Point2::xy(b, hi),
                Point2::xy(a, hi),
            ];
            ax.fill(c, &column, palette::purple().faded(0.06), 0.5 * shade);
        }
    }
}

/// A field's weight at each site, as a curve over the sites.
fn weights(field: &Modes) -> Vec<Point2> {
    field
        .iter()
        .enumerate()
        .map(|(s, v)| Point2::xy(s as f32, v.norm_squared() as f32))
        .collect()
}

/// One transport's carried mode at record `now`, over the gate, with the instantaneous
/// boundary mode dashed.
fn draw_transport(c: &mut Canvas, ax: &Axes, m: &Moving, case: usize, now: usize) {
    let (colour, name) = cases()[case];
    draw_gate(c, ax, &m.potentials[now]);
    ax.frame(c, "", "SITE", "MODE WEIGHT");
    ax.dashed(
        c,
        &weights(&m.target[now]),
        1.0,
        4.0,
        palette::ink().faded(0.6),
    );
    ax.polyline(c, &weights(&m.history[case][now]), 2.0, colour);
    let label = Point2::xy(SITES as f32 * 0.01, ax.y[1] * 0.86);
    let text = format!("{name}: {} TIME UNITS", DURATIONS[case]);
    ax.text(c, label, &text, 10.0, colour, Align::Left);
}

/// The carried states' weight outside the boundary pair, up to record `now`.
fn draw_leakage(c: &mut Canvas, ax: &Axes, m: &Moving, now: usize) {
    ax.frame(c, "", "FRACTION OF THE GATE'S WAY", "BULK WEIGHT");
    for (case, (colour, _)) in cases().into_iter().enumerate() {
        let curve: Vec<Point2> = m.progress[..=now]
            .iter()
            .zip(&m.share[case])
            .map(|(p, s)| Point2::xy(*p as f32, (1.0 - s) as f32))
            .collect();
        ax.polyline(c, &curve, 1.8, colour);
        ax.scatter(
            c,
            &[*curve.last().expect("a record")],
            Marker::Dot,
            6.0,
            colour,
        );
    }
}

/// The excitation energies as the potential inside the gate is lowered past twice the hopping.
fn draw_spectrum(c: &mut Canvas, ax: &Axes, d: &Data) {
    ax.frame(c, "THE GATE'S SPECTRUM", "POTENTIAL INSIDE", "ENERGY");
    let threshold = 2.0 * HOPPING as f32;
    let across = [
        Point2::xy(threshold, ax.y[0]),
        Point2::xy(threshold, ax.y[1]),
    ];
    ax.dashed(c, &across, 1.0, 3.0, palette::grid());
    // Each energy is held by two quadratures: one curve per pair.
    for level in (0..ENERGIES).step_by(2) {
        let curve: Vec<Point2> = d
            .potentials
            .iter()
            .zip(&d.spectrum)
            .map(|(mu, e)| Point2::xy(*mu as f32, e[level] as f32))
            .collect();
        let colour = if level == 0 {
            palette::sky()
        } else {
            palette::ink().faded(0.6)
        };
        ax.polyline(c, &curve, 1.4, colour);
    }
}

/// The boundary modes' splitting as the segment shortens, on a logarithmic axis.
fn draw_splitting(c: &mut Canvas, ax: &Axes, d: &Data) {
    ax.frame(
        c,
        "THE ENDS' SPLITTING",
        "SEGMENT LENGTH (SITES)",
        "LOWEST ENERGY",
    );
    let curve: Vec<Point2> = d
        .separations
        .iter()
        .zip(&d.splitting)
        .map(|(s, e)| Point2::xy(*s as f32, *e as f32))
        .collect();
    ax.polyline(c, &curve, 1.8, palette::sky());
}

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let d = data();
    let screen = c.rect();
    let records = d.moving.progress.len();
    let now = (((t / SECONDS).fract() * records as f32) as usize).min(records - 1);
    let top = d.peak as f32 * 1.15;
    let sites = [0.0, (SITES - 1) as f32];
    for case in 0..2 {
        let rect = screen.part(
            0.06,
            0.14 + 0.28 * case as f32,
            0.6,
            0.34 + 0.28 * case as f32,
        );
        let ax = Axes::new(rect, sites, [0.0, top]);
        draw_transport(c, &ax, &d.moving, case, now);
    }
    let leakage = Axes::new(
        screen.part(0.06, 0.71, 0.6, 0.91),
        [0.0, 1.0],
        [-0.02, 1.02],
    );
    draw_leakage(c, &leakage, &d.moving, now);
    let spectrum = Axes::new(
        screen.part(0.7, 0.15, 0.97, 0.45),
        [OUTSIDE as f32, INSIDE as f32],
        [0.0, 1.7],
    );
    draw_spectrum(c, &spectrum, d);
    let (from, to, _) = SEPARATIONS;
    let splitting = Axes::new(
        screen.part(0.7, 0.6, 0.97, 0.88),
        [from as f32, to as f32],
        [1e-8, 1.0],
    )
    .log_y();
    draw_splitting(c, &splitting, d);
    let fraction = d.moving.progress[now];
    caption(
        c,
        "THE KITAEV WIRE: MOVING A MAJORANA MODE",
        &format!(
            "VGA2D: TWO MAJORANAS PER SITE, A SKEW FIELD MAP  GATE {:.0}% OF ITS WAY",
            100.0 * fraction
        ),
    );
}

fn main() {
    run(Anim::new("kitaev wire", SECONDS).size(960, 540), draw);
}

#[cfg(test)]
mod tests {
    use super::kitaev::*;
    use gax::field::Field;
    use gax::vga2d::{Scalar, Vector};
    use gax_numga_examples::signal::linspace;

    gax::algebra! {
        algebra majoranas "The Clifford algebra of the four Majoranas of a wire of two sites, R(4,0): site 0's ea eb, site 1's ec ed.";
        basis ea = 1, eb = 1, ec = 1, ed = 1;
        kind Scalar = [1];
        kind Vector = [ea, eb, ec, ed];
        kind Bivector = [eab, eac, ead, ebc, ebd, ecd];
        versor Even = [1, eab, eac, ead, ebc, ebd, ecd, eabcd];
    }

    const ROUND_OFF: f64 = 1e-10;

    /// A field over two sites as one vector of the big algebra.
    fn joined(f: &Modes) -> majoranas::Vector<(), f64> {
        let (a, b) = (f.site(0), f.site(1));
        majoranas::Vector::new(a.e1(), a.e2(), b.e1(), b.e2())
    }

    /// A field over two sites from one vector of the big algebra.
    fn split(v: majoranas::Vector<(), f64>) -> Modes {
        let c = v.c;
        Field::new(vec![Vector::new(c[0], c[1]), Vector::new(c[2], c[3])])
    }

    fn potential(values: &[f64]) -> Potential {
        values.iter().map(|mu| Scalar::new(*mu)).collect()
    }

    /// The bivector of the big algebra whose action is the generator of a wire of two sites.
    fn bivector(mu: [f64; 2], hopping: f64, pairing: f64) -> majoranas::Bivector<(), f64> {
        // In the order eab, eac, ead, ebc, ebd, ecd.
        majoranas::Bivector::new(
            -mu[0],
            0.0,
            pairing - hopping,
            hopping + pairing,
            0.0,
            -mu[1],
        )
    }

    fn largest(f: &Modes) -> f64 {
        f.iter().flat_map(|v| v.c).fold(0.0, |m, x| m.max(x.abs()))
    }

    /// numga's `test_sparse_generator_is_the_clifford_bivector_action`: on each Majorana the
    /// generator is the commutator with one bivector of R(4,0), and it is skew.
    #[test]
    fn the_generator_is_a_bivector_s_action() {
        let (hopping, pairing, mu) = (1.0, 0.7, [0.6, 1.1]);
        let wire = Wire::chain(2, hopping, pairing);
        let g = wire.generator(&potential(&mu));
        let skew = &g + &g.transpose();
        let b = bivector(mu, hopping, pairing);
        for k in 0..4 {
            let mut c = [0.0; 4];
            c[k] = 1.0;
            let basis = majoranas::Vector::from_coeffs(c);
            let want = b.commutator(basis);
            let got = joined(&g.apply(&split(basis)));
            assert!(
                (got - want).c.iter().all(|d| d.abs() < ROUND_OFF),
                "{k}: {got:?} vs {want:?}"
            );
            assert!(largest(&skew.apply(&split(basis))) < ROUND_OFF);
        }
    }

    /// numga's `test_decoupled_majoranas_and_normal_chain_spectrum`: at the sweet spot (pairing
    /// equal to hopping, no potential) the end Majoranas are free and the bulk sits at twice the
    /// hopping; without pairing the excitations are the tight-binding band's distances from the
    /// potential, each twice.
    #[test]
    #[allow(clippy::disallowed_methods)] // the band's closed form, the reference it is checked against
    fn the_sweet_spot_and_the_normal_chain_have_their_spectra() {
        let (sites, hopping, count) = (12, 1.0, 8);
        let sweet = Wire::chain(sites, hopping, hopping);
        let g = sweet.generator(&potential(&vec![0.0; sites]));
        let end = |site: usize, v: Vector<(), f64>| {
            Field::from_fn(sites, |s| if s == site { v } else { Vector::zero() })
        };
        assert!(largest(&g.apply(&end(0, Vector::new(1.0, 0.0)))) < ROUND_OFF);
        assert!(largest(&g.apply(&end(sites - 1, Vector::new(0.0, 1.0)))) < ROUND_OFF);
        let (energies, _) = sweet.modes(&g, count);
        for (k, e) in energies.iter().enumerate() {
            let want = if k < 2 { 0.0 } else { 2.0 * hopping };
            assert!((e - want).abs() < ROUND_OFF, "{k}: {e}");
        }

        let mu = 0.6;
        let normal = Wire::chain(sites, hopping, 0.0);
        let (energies, _) = normal.modes(&normal.generator(&potential(&vec![mu; sites])), count);
        let mut band: Vec<f64> = (0..sites)
            .map(|i| {
                (-mu - 2.0
                    * hopping
                    * (core::f64::consts::PI * (i + 1) as f64 / (sites + 1) as f64).cos())
                .abs()
            })
            .collect();
        band.sort_by(f64::total_cmp);
        for (k, e) in energies.iter().enumerate() {
            assert!(
                (e - band[k / 2]).abs() < ROUND_OFF,
                "{k}: {e} vs {}",
                band[k / 2]
            );
        }
    }

    /// numga's `test_cayley_evolution_converges_to_the_rotor`: the Cayley steps keep the norm and
    /// converge at second order to the sandwich by the exponential of the bivector.
    #[test]
    fn the_cayley_steps_converge_to_the_rotor() {
        let wire = Wire::chain(2, 1.0, 0.7);
        let mu = [0.6, 1.1];
        let start = majoranas::Vector::<(), f64>::new(1.0, 2.0, 3.0, 4.0).normalized();
        let duration = 0.8;
        let exact = (bivector(mu, 1.0, 0.7) * (duration / 2.0)).exp() >> start;
        let error = |steps: usize| {
            let profiles = vec![potential(&mu); steps];
            let run = transport(&wire, &split(start), &profiles, duration, steps);
            let last = run.last().expect("the end");
            assert!((last.dot(last) - 1.0).abs() < ROUND_OFF);
            (joined(last) - exact).norm()
        };
        let (coarse, fine) = (error(40), error(80));
        assert!(fine < 1e-4, "{fine}");
        assert!((3.9..4.1).contains(&(coarse / fine)), "{}", coarse / fine);
    }

    /// numga's `test_localization_does_not_depend_on_the_eigensolver_basis`: any turn of the two
    /// modes localizes to the same pair, one at each boundary.
    #[test]
    fn localization_does_not_depend_on_the_basis() {
        let wire = Wire::chain(SITES, HOPPING, PAIRING);
        let p = gate(SITES, LEFT, RIGHT, INSIDE, OUTSIDE, WIDTH);
        let (_, modes) = wire.modes(&wire.generator(&p), 2);
        let localized = wire.localized(&modes);
        let (a, b) = (&modes[0], &modes[1]);
        let mixed = [
            &a.scaled(0.6) + &b.scaled(-0.8),
            &a.scaled(0.8) + &b.scaled(0.6),
        ];
        let recovered = wire.localized(&mixed);
        for (l, r) in localized.iter().zip(&recovered) {
            for (u, v) in l.iter().zip(r.iter()) {
                assert!((u.norm_squared() - v.norm_squared()).abs() < ROUND_OFF);
            }
        }
        let centre = |f: &Modes| -> f64 {
            f.iter()
                .enumerate()
                .map(|(s, v)| s as f64 * v.norm_squared())
                .sum()
        };
        assert!(centre(&localized[1]) - centre(&localized[0]) > 50.0);
    }

    /// numga's `test_slow_transport_follows_the_boundary_and_fast_transport_leaks`.
    #[test]
    fn slow_transport_follows_and_fast_transport_leaks() {
        let m = moving();
        for state in m.history.iter().flatten() {
            assert!((state.dot(state) - 1.0).abs() < ROUND_OFF);
        }
        let (slow, fast) = (m.share[0].last().unwrap(), m.share[1].last().unwrap());
        assert!(*slow > 0.999 && *fast < 0.01, "{slow} {fast}");
        let last = m.history[0].last().expect("records");
        let overlap = last.dot(m.target.last().expect("records")).powi(2);
        assert!(overlap > 0.999, "{overlap}");
    }

    /// numga's `test_overlap_lifts_the_lowest_excitation`: the shorter the segment, the more its
    /// two boundary modes overlap and split.
    #[test]
    fn overlap_lifts_the_lowest_excitation() {
        let (from, to, count) = super::SEPARATIONS;
        let energies = overlap(&linspace(from, to, count));
        assert!(energies[0] > 0.1, "{}", energies[0]);
        assert!(
            *energies.last().unwrap() < 2e-7,
            "{}",
            energies.last().unwrap()
        );
        assert!(energies.windows(2).all(|w| w[1] < w[0]));
    }

    /// numga's `test_scenes_draw`: the frames draw, and two records apart they differ.
    #[test]
    fn the_frames_draw_and_move() {
        let mut draw = super::draw;
        let anim = gax_numga_examples::Anim::new("t", 1.0).size(320, 180);
        let a = gax_numga_examples::app::frame(&anim, 0.5, &mut draw);
        let b = gax_numga_examples::app::frame(&anim, 6.0, &mut draw);
        assert!(a.mean().luma() > 0.0);
        assert!(a.mean() != b.mean());
    }
}
