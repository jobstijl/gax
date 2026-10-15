//! numga's `quantum/hubbard_ring`: four electrons hopping around a ring of four sites, with an
//! energy cost for sharing a site, in R(8,0). The eight basis vectors are the orbitals, spin up
//! and spin down on each site, and a state of four electrons is a 4-vector, the wedge of the
//! four occupied orbitals: it vanishes when two share an orbital and changes sign when two trade
//! places, also when an electron hops across the ring's seam past the other three.
//!
//! A one-electron map acts on each electron in turn, `Σ h(eᵢ) ^ (eᵢ ⌋ state)` (each orbital
//! taken out of the state and its image put back), the same one line for any number of
//! electrons as for the two of the Hubbard dimer. (numga writes it without the metric, as a join
//! with an open complement and a trace; in a Euclidean algebra the contraction is the same.)
//! Hopping lifts that way, and so does each orbital's projector, the number of electrons in it;
//! composing a site's up and down numbers counts whether it holds both. Energies are in units
//! of the hopping, and time acts through the cosine and the sine of the Hamiltonian, from its
//! 70 states.
//!
//! These maps keep the spin, so each is a pair: a map on the spin-up orbitals and one on the
//! spin-down ones, two kinds of four vectors. (numga's one-electron maps are on all eight. When
//! this port was written, a kind of all eight vectors made gax derive its outermorphism to the
//! 4-vectors, 4900 minors, past the build's memory; that extension is now over gax-gen's size
//! limit and not generated.)
//!
//! At a strong repulsion one electron sits on each site, and the sixteen lowest levels are
//! those of the Heisenberg ring of four spins, coupled by the exchange `4 t² / U`: one, two and
//! three exchanges above the ground state. The animation starts the spins up, up, up, down and
//! runs one exchange cycle: the flipped spin walks around the ring and reaches the opposite site
//! halfway, the more cleanly the stronger the repulsion. Below, the lowest levels against the
//! repulsion, approaching the Heisenberg ring's.

use gax_numga_examples::points::circle;
use gax_numga_examples::signal::linspace;
use gax_numga_examples::{
    Align, Anim, Axes, Canvas, Light, Marker, ORIGIN2, Point2, Rect, backdrop, caption, palette,
    run,
};
use std::sync::OnceLock;

#[path = "../shared/quantum.rs"]
mod quantum;

gax::algebra! {
    algebra ring "Four electrons on a ring of four sites, R(8,0): spin up and down on each site (ea eb, ec ed, ee ef, eg eh).";
    basis ea = 1, eb = 1, ec = 1, ed = 1, ee = 1, ef = 1, eg = 1, eh = 1;
    kind Scalar = [1];
    kind Up = [ea, ec, ee, eg];
    kind Down = [eb, ed, ef, eh];
    kind State = [eabcd, eabce, eabcf, eabcg, eabch, eabde, eabdf, eabdg, eabdh, eabef, eabeg,
        eabeh, eabfg, eabfh, eabgh, eacde, eacdf, eacdg, eacdh, eacef, eaceg, eaceh, eacfg, eacfh,
        eacgh, eadef, eadeg, eadeh, eadfg, eadfh, eadgh, eaefg, eaefh, eaegh, eafgh, ebcde, ebcdf,
        ebcdg, ebcdh, ebcef, ebceg, ebceh, ebcfg, ebcfh, ebcgh, ebdef, ebdeg, ebdeh, ebdfg, ebdfh,
        ebdgh, ebefg, ebefh, ebegh, ebfgh, ecdef, ecdeg, ecdeh, ecdfg, ecdfh, ecdgh, ecefg, ecefh,
        ecegh, ecfgh, edefg, edefh, edegh, edfgh, eefgh];
}

mod hubbard {
    use super::quantum::InTime;
    use super::ring::{Down, Multivector, State, Up};

    /// An orbital (or a sum of them) as a multivector of the whole algebra, and a state of four
    /// electrons.
    pub type Mv = Multivector<(), f64>;
    pub type S = State<(), f64>;
    /// A map on states.
    pub type Hamiltonian = State<(State,), f64>;

    /// The sites around the ring, and the two spins.
    pub const SITES: usize = 4;
    pub const UP: usize = 0;
    pub const DOWN: usize = 1;

    /// A site's place among the orbitals of one spin, its coefficients: a one on the site, going
    /// round the ring past the last.
    fn on_site(site: usize) -> [f64; SITES] {
        core::array::from_fn(|k| f64::from(k == site % SITES))
    }

    /// The spin-up and the spin-down orbital on a site.
    fn up(site: usize) -> Up<(), f64> {
        Up::from_coeffs(on_site(site))
    }
    fn down(site: usize) -> Down<(), f64> {
        Down::from_coeffs(on_site(site))
    }

    /// The orbital of a spin on a site.
    pub fn orbital(site: usize, spin: usize) -> Mv {
        match spin {
            UP => up(site).cast::<Multivector>(),
            _ => down(site).cast::<Multivector>(),
        }
    }

    /// The state with four orbitals occupied: their wedge.
    pub fn occupied(orbitals: [Mv; 4]) -> S {
        let [a, b, c, d] = orbitals;
        (a ^ b ^ c ^ d).cast::<State>()
    }

    /// The seventy states with four of the eight orbitals occupied, the basis of `State`.
    fn basis() -> [S; 70] {
        core::array::from_fn(|k| State::from_coeffs(core::array::from_fn(|j| f64::from(j == k))))
    }

    /// A one-electron map that keeps the spin: a map on the spin-up orbitals and one on the
    /// spin-down.
    pub struct OneElectron {
        up: Up<(Up,), f64>,
        down: Down<(Down,), f64>,
    }

    impl OneElectron {
        /// The map applied to orbitals: each spin's part by its own map.
        pub fn of(&self, orbitals: Mv) -> Mv {
            let up = self.up.of(orbitals.cast::<Up>()).cast::<Multivector>();
            up + self.down.of(orbitals.cast::<Down>()).cast::<Multivector>()
        }

        /// The map acting on each of the four electrons in turn: each orbital taken out of a
        /// state by the contraction and its image wedged back in, with its exchange sign. Built
        /// from its images of the basis states, so that every product is between plain
        /// multivectors (with the map's open slot carried through, the products would be
        /// multivectors of seventy columns, 140 kB each).
        pub fn lift(&self) -> Hamiltonian {
            let orbitals: Vec<Mv> = (0..SITES)
                .flat_map(|site| [UP, DOWN].map(|spin| orbital(site, spin)))
                .collect();
            let image = |state: S| {
                let moved: Mv = orbitals.iter().map(|e| self.of(*e) ^ e.lc(state)).sum();
                moved.cast::<State>()
            };
            Hamiltonian::from_images(basis().map(image))
        }
    }

    /// An electron moved to a neighbouring site with its spin kept, at `-strength`: around the
    /// ring each site's orbital goes to its two neighbours', alike for both spins.
    pub fn single_hop(strength: f64) -> OneElectron {
        let around = SITES - 1;
        OneElectron {
            up: Up::<(Up,), f64>::from_images(core::array::from_fn(|site| {
                (up(site + 1) + up(site + around)) * -strength
            })),
            down: Down::<(Down,), f64>::from_images(core::array::from_fn(|site| {
                (down(site + 1) + down(site + around)) * -strength
            })),
        }
    }

    /// The number of electrons in one orbital: its projector, lifted.
    fn number(site: usize, spin: usize) -> Hamiltonian {
        let (none_up, none_down) = (Up::<(Up,), f64>::zero(), Down::<(Down,), f64>::zero());
        let (u, d) = (up(site), down(site));
        let single = match spin {
            UP => OneElectron {
                up: u * (u | Up::slot()),
                down: none_down,
            },
            _ => OneElectron {
                up: none_up,
                down: d * (d | Down::slot()),
            },
        };
        single.lift()
    }

    /// The ring's maps on states, built once: hopping, each orbital's number (`[site][spin]`),
    /// and how many sites hold both spins (on each site the up number composed with the down
    /// number).
    pub struct Ring {
        pub hopping: Hamiltonian,
        numbers: Vec<[Hamiltonian; 2]>,
        pub double_occupancy: Hamiltonian,
    }

    impl Ring {
        pub fn new() -> Ring {
            let numbers: Vec<[Hamiltonian; 2]> = (0..SITES)
                .map(|site| [UP, DOWN].map(|spin| number(site, spin)))
                .collect();
            Ring {
                hopping: single_hop(HOPPING).lift(),
                double_occupancy: numbers.iter().map(|[up, down]| up.of(*down)).sum(),
                numbers,
            }
        }

        /// Electrons hop around the ring, and sharing a site costs the repulsion.
        pub fn hamiltonian(&self, repulsion: f64) -> Hamiltonian {
            self.hopping + self.double_occupancy * repulsion
        }

        /// How many electrons each orbital holds, `[site][spin]`, from both parts of a state in
        /// time.
        pub fn occupation(&self, state: &InTime<S>) -> Occupation {
            let held =
                |n: &Hamiltonian| expectation(state.cosine, *n) + expectation(state.sine, *n);
            core::array::from_fn(|site| self.numbers[site].each_ref().map(held))
        }

        /// The lowest levels above the ground state, in units of the exchange.
        pub fn levels(&self, repulsion: f64) -> Vec<f64> {
            let (energies, _) = self.hamiltonian(repulsion).eigh();
            energies[..SPIN_STATES]
                .iter()
                .map(|e| (e - energies[0]) / exchange(repulsion))
                .collect()
        }

        /// The flipped spin over the exchange phases: how many electrons each orbital holds.
        /// The start's share in each of the Hamiltonian's states turns by its own energy.
        pub fn walk(&self, repulsion: f64, phases: &[f64]) -> Vec<Occupation> {
            let (energies, modes) = self.hamiltonian(repulsion).eigh();
            let shares = modes.map(|m| m * m.reverse().scalar_product(start()).s());
            phases
                .iter()
                .map(|p| {
                    let time = p / exchange(repulsion);
                    self.occupation(&InTime::turned(&energies, &shares, time))
                })
                .collect()
        }
    }

    /// How many electrons each orbital holds, `[site][spin]`.
    pub type Occupation = [[f64; 2]; SITES];

    /// The expectation of an observable in a real state.
    fn expectation(state: S, observable: Hamiltonian) -> f64 {
        state.reverse().scalar_product(observable.of(state)).s()
    }

    /// The hopping, the unit of energy, and the spin states one electron per site allows.
    pub const HOPPING: f64 = 1.0;
    pub const SPIN_STATES: usize = 16;

    /// The exchange between neighbouring spins at a strong repulsion, `4 t² / U`.
    pub fn exchange(repulsion: f64) -> f64 {
        4.0 * HOPPING * HOPPING / repulsion
    }

    /// Spin up on the first three sites, spin down on the last.
    pub fn start() -> S {
        occupied([
            orbital(0, UP),
            orbital(1, UP),
            orbital(2, UP),
            orbital(3, DOWN),
        ])
    }
}

use hubbard::*;

const SECONDS: f32 = 10.0;
/// The repulsions of the walk, the phases of its cycle, and the sweep of the levels.
const REPULSIONS: [f64; 3] = [4.0, 8.0, 20.0];
const FRAMES: usize = 151;
const SWEEP: (f64, f64, usize) = (1.0, 40.0, 157);
/// The Heisenberg ring's levels above its ground state, in exchanges.
const HEISENBERG: [f64; 3] = [1.0, 2.0, 3.0];

/// What the frames share, computed once.
struct Data {
    walks: Vec<Vec<Occupation>>,
    sweep: Vec<f64>,
    levels: Vec<Vec<f64>>,
}

fn data() -> &'static Data {
    static D: OnceLock<Data> = OnceLock::new();
    D.get_or_init(|| {
        let ring = Ring::new();
        let phases = linspace(0.0, core::f64::consts::TAU, FRAMES);
        let (from, to, count) = SWEEP;
        let sweep = linspace(from, to, count);
        Data {
            walks: REPULSIONS.iter().map(|u| ring.walk(*u, &phases)).collect(),
            levels: sweep.iter().map(|u| ring.levels(*u)).collect(),
            sweep,
        }
    })
}

/// The sites on a square, in the order the hops go round: bottom left, bottom right, top right,
/// top left.
fn place(site: usize) -> Point2 {
    let corners = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)];
    let (x, y) = corners[site];
    Point2::xy(0.7 * x, 0.7 * y)
}

/// One ring at a moment: each site's spin-up count as an arrow up, its spin-down count as an
/// arrow down.
fn draw_ring(c: &mut Canvas, ax: &Axes, counts: &Occupation) {
    ax.polyline(c, &circle(ORIGIN2, 1.0, 96), 1.0, palette::grid());
    let up = Point2::direction(0.0, 1.0);
    for (site, [n_up, n_down]) in counts.iter().enumerate() {
        let at = place(site);
        ax.scatter(c, &[at], Marker::Dot, 6.0, palette::ink());
        let spins: [(f64, f32, Light); 2] = [
            (*n_up, 1.0, palette::red()),
            (*n_down, -1.0, palette::sky()),
        ];
        for (count, sense, colour) in spins {
            let length = 0.5 * count as f32;
            if length > 0.02 {
                ax.arrow(c, at, at + up.gp(sense * length), 3.0, 6.0, colour);
            }
        }
    }
}

/// The lowest levels above the ground state against the repulsion, in exchanges, with the
/// Heisenberg ring's levels dashed.
fn draw_levels(c: &mut Canvas, ax: &Axes, d: &Data) {
    ax.frame(
        c,
        "",
        "REPULSION / HOPPING",
        "ABOVE THE GROUND STATE, IN EXCHANGES 4T^2/U",
    );
    let end = d.sweep.last().copied().unwrap_or_default() as f32;
    for level in HEISENBERG {
        let across = [Point2::xy(0.0, level as f32), Point2::xy(end, level as f32)];
        ax.dashed(c, &across, 1.0, 5.0, palette::ink().faded(0.5));
    }
    for k in 0..SPIN_STATES {
        let curve: Vec<Point2> = d
            .sweep
            .iter()
            .zip(&d.levels)
            .map(|(u, l)| Point2::xy(*u as f32, l[k] as f32))
            .collect();
        ax.polyline(c, &curve, 1.2, palette::purple());
    }
}

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let d = data();
    let screen = c.rect();
    let progress = (t / SECONDS).fract();
    let now = ((progress * FRAMES as f32) as usize).min(FRAMES - 1);
    let band = Rect::new(screen.width() * 0.08, 70.0, screen.width() * 0.92, 320.0);
    for (k, (repulsion, walk)) in REPULSIONS.iter().zip(&d.walks).enumerate() {
        let cell = band.column(k, REPULSIONS.len());
        let ax = Axes::equal(cell, ORIGIN2, 1.25);
        draw_ring(c, &ax, &walk[now]);
        let label = cell.bottom_middle() + Point2::direction(0.0, 4.0);
        let text = format!("U/T = {repulsion}");
        c.text(&text, label, 12.0, palette::ink(), Align::Center);
    }
    let note = band.top_middle() + Point2::direction(0.0, 18.0);
    let phase = 2.0 * progress;
    let text = format!("EXCHANGE PHASE {phase:.2} PI: SPIN UP RED, SPIN DOWN BLUE");
    c.text(&text, note, 10.0, palette::ink(), Align::Center);
    let (from, to, _) = SWEEP;
    let levels = Axes::new(
        screen.part(0.08, 0.68, 0.96, 0.92),
        [from as f32, to as f32],
        [-0.1, 3.6],
    );
    draw_levels(c, &levels, d);
    caption(
        c,
        "THE HUBBARD RING: A FLIPPED SPIN WALKS AROUND",
        "R(8,0): ORBITALS ARE VECTORS, FOUR ELECTRONS A 4-VECTOR, 70 STATES",
    );
}

fn main() {
    run(Anim::new("hubbard ring", SECONDS).size(960, 540), draw);
}

#[cfg(test)]
mod tests {
    use super::hubbard::*;
    use gax_numga_examples::signal::linspace;

    fn small(s: S) -> bool {
        s.c.iter().all(|x| x.abs() < 1e-12)
    }

    /// numga's `test_the_lift_acts_on_each_of_four_electrons_in_turn`: lifted hopping is the
    /// product rule on four electrons, and a state with two sites full and two empty has two
    /// doubly occupied sites.
    #[test]
    fn the_lift_acts_on_each_of_four_electrons_in_turn() {
        let single = single_hop(HOPPING);
        let [a, b, c, d] = [
            orbital(0, UP),
            orbital(1, DOWN),
            orbital(2, UP),
            orbital(3, DOWN),
        ];
        let rule = occupied([single.of(a), b, c, d])
            + occupied([a, single.of(b), c, d])
            + occupied([a, b, single.of(c), d])
            + occupied([a, b, c, single.of(d)]);
        let ring = Ring::new();
        assert!(small(ring.hopping.of(occupied([a, b, c, d])) - rule));
        let full = occupied([
            orbital(0, UP),
            orbital(0, DOWN),
            orbital(1, UP),
            orbital(1, DOWN),
        ]);
        assert!(small(ring.double_occupancy.of(full) - full * 2.0));
    }

    /// numga's `test_spins_approach_the_heisenberg_ring_and_a_flipped_spin_crosses_it`: at a
    /// repulsion of 40 the levels are one, two and three exchanges up (three, seven and five of
    /// them), within numga's 0.05 (the Heisenberg ring is the limit of an infinite repulsion);
    /// the flipped spin reaches the opposite site as `sin⁴(phase / 2)`, nearly.
    #[test]
    #[allow(clippy::disallowed_methods)] // the Heisenberg ring's closed form, the reference checked against
    fn spins_approach_the_heisenberg_ring_and_a_flipped_spin_crosses_it() {
        let ring = Ring::new();
        let l = ring.levels(40.0);
        for (range, want) in [(1..4, 1.0), (4..11, 2.0), (11..16, 3.0)] {
            assert!(l[range].iter().all(|e| (e - want).abs() < 0.05), "{l:?}");
        }
        let phases = linspace(0.0, core::f64::consts::TAU, 9);
        let counts = ring.walk(40.0, &phases);
        for (p, n) in phases.iter().zip(&counts) {
            let total: f64 = n.iter().flatten().sum();
            assert!((total - 4.0).abs() < 1e-11);
            let opposite = n[1][DOWN];
            assert!(
                (opposite - (p / 2.0).sin().powi(4)).abs() < 0.03,
                "{p}: {opposite}"
            );
        }
    }

    /// numga's scene checks on the drawn walks: four electrons all along, and at a repulsion of
    /// 20 the flipped spin is on the opposite site halfway through the cycle.
    #[test]
    fn the_drawn_walks_keep_four_electrons_and_cross_halfway() {
        let d = super::data();
        for counts in d.walks.iter().flatten() {
            let total: f64 = counts.iter().flatten().sum();
            assert!((total - 4.0).abs() < 1e-11);
        }
        let strongest = d.walks.last().expect("walks");
        let halfway = strongest[super::FRAMES / 2][1][DOWN];
        assert!(halfway > 0.95, "{halfway}");
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws(super::draw, 0.5);
    }
}
