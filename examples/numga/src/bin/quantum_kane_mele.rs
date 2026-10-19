//! numga's `quantum/kane_mele`: the edge states of a graphene flake with spin-orbit coupling, in
//! the geometric algebra of space (VGA3D). An electron hops between neighbouring carbon atoms,
//! and its state on each atom is a spinor, an even multivector; the flake's state is a field of
//! spinors. The Hamiltonian is a sparse field map whose cells multiply from the left: minus one
//! for each hop to a neighbour, the spin-orbit strength times `xy` (signed by the sense in which
//! the path turns through the atom between) for each hop to a second neighbour, and a mass
//! signed by sublattice on each atom.
//!
//! Every cell is a scalar or a multiple of `xy`: an even multivector of the plane, VGA2D's
//! rotor. Multiplying from the left by these keeps the spin along z, so the spin-up spinors (the
//! even multivectors of the `xy` plane, VGA2D's rotors themselves) and the spin-down ones (the
//! planes through z) are sectors of their own; the spin-up levels are the sparse eigenproblem
//! with VGA2D rotor cells, embedded in VGA3D by the homomorphism. In the notation of Kane and
//! Mele, with `i` for `xy`, `H = -Σ c†ᵢ cⱼ + i λ Σ νᵢⱼ c†ᵢ s_z cⱼ + m Σ ξᵢ c†ᵢ cᵢ`.
//!
//! Where the mass is below `3 √3` times the spin-orbit coupling, the bulk gap holds states on
//! the flake's rim, and the two spins run around it in opposite senses; a larger mass opens a
//! plain gap. Time turns a state of energy `E` by `exp(-xy E t)` on its right, and an
//! electron's spin density on each atom is `psi z ~psi`.
//!
//! The animation shows an electron started at the middle of the edge facing x with its spin
//! along x (half up, half down): its spin-up half (red) runs clockwise around the rim, its
//! spin-down half (blue) counterclockwise. Beside it, the spin-up levels nearest zero against
//! the mass, shaded by their share on the rim, and the state nearest zero below and above the
//! transition.

use gax::vga3d::Vector;
use gax_numga_examples::{
    Align, Anim, Axes, Canvas, Marker, ORIGIN2, Point2, Rect, backdrop, caption, colormap,
    from_above, palette, reach3, run_with,
};

mod kane_mele {
    use gax::field::{Field, SparseMap};
    use gax::vga2d::Rotor as Up;
    use gax::vga3d::{Bivector, Rotor, Vector};
    use gax::{Extensor, Kind};
    use gax_numga_examples::par_map;
    use gax_numga_examples::signal::linspace;

    /// A coupling between atoms: a scalar plus a multiple of `xy`, the even multivectors of the
    /// plane.
    pub type Coupling = gax::vga2d::Rotor;
    /// A spinor, and the flake's state: a spinor on every atom.
    pub type Spinor = Rotor;
    pub type Spinors = Field<Spinor>;

    pub fn x() -> Vector {
        Vector::new(1.0, 0.0, 0.0)
    }
    pub fn z() -> Vector {
        Vector::new(0.0, 0.0, 1.0)
    }
    pub fn xy() -> Bivector {
        Bivector::new(0.0, 0.0, 1.0)
    }

    /// A hexagonal graphene flake with armchair edges: the atoms' positions, their sublattices
    /// (plus and minus one), the pairs of neighbours and of second neighbours (each both ways
    /// round) with the sense each second neighbour's path turns, which atoms are on the rim, and
    /// the atom in the middle of the edge facing x.
    pub struct Flake {
        pub positions: Field<Vector>,
        pub sublattice: Vec<f64>,
        pub first: Vec<[usize; 2]>,
        pub second: Vec<[usize; 2]>,
        pub senses: Vec<f64>,
        pub rim: Vec<bool>,
        pub start: usize,
    }

    /// An atom of the honeycomb: its cell `(i, j)` in the lattice spanned by two directions 60
    /// degrees apart, and its sublattice, a third of the way along the cell's long diagonal for
    /// the second.
    #[derive(Clone, Copy)]
    struct Atom {
        i: i64,
        j: i64,
        s: usize,
    }

    impl Atom {
        /// The atom's coordinates along the two lattice directions.
        fn uv(self) -> (f64, f64) {
            let third = self.s as f64 / 3.0;
            (self.i as f64 + third, self.j as f64 + third)
        }
        /// Its largest projection on the three lattice directions, in half lattice constants.
        fn reach(self) -> f64 {
            let (u, v) = self.uv();
            (2.0 * u + v)
                .abs()
                .max((u + 2.0 * v).abs())
                .max((v - u).abs())
        }
        /// Plus one on the first sublattice, minus one on the second.
        fn sign(self) -> i64 {
            1 - 2 * self.s as i64
        }
        /// Its three neighbours' places: on the other sublattice, in its own cell and in the
        /// cells before it along each direction (after it, for the second sublattice).
        fn neighbours(self) -> [Atom; 3] {
            let k = self.sign();
            [(0, 0), (-1, 0), (0, -1)].map(|(di, dj)| Atom {
                i: self.i + k * di,
                j: self.j + k * dj,
                s: 1 - self.s,
            })
        }
        /// Its six second neighbours' places, one cell away on its own sublattice, and the
        /// sense the path through the atom between turns: one way for three, the other way for
        /// the rest, opposite on the two sublattices.
        fn second_neighbours(self) -> [(Atom, f64); 6] {
            const STEPS: [(i64, i64, f64); 6] = [
                (1, 0, -1.0),
                (-1, 1, -1.0),
                (0, -1, -1.0),
                (-1, 0, 1.0),
                (1, -1, 1.0),
                (0, 1, 1.0),
            ];
            let k = self.sign() as f64;
            STEPS.map(|(di, dj, turn)| {
                let at = Atom {
                    i: self.i + di,
                    j: self.j + dj,
                    s: self.s,
                };
                (at, k * turn)
            })
        }
    }

    /// The atoms' indices by place, `None` where there is none.
    struct Lookup {
        size: i64,
        table: Vec<Option<usize>>,
    }

    impl Lookup {
        fn new(atoms: &[Atom], size: i64) -> Lookup {
            let side = (2 * size + 3) as usize;
            let mut lookup = Lookup {
                size,
                table: vec![None; side * side * 2],
            };
            for (k, a) in atoms.iter().enumerate() {
                let slot = lookup.slot(*a).expect("inside the table");
                lookup.table[slot] = Some(k);
            }
            lookup
        }
        fn slot(&self, a: Atom) -> Option<usize> {
            let side = 2 * self.size + 3;
            let (i, j) = (a.i + self.size + 1, a.j + self.size + 1);
            ((0..side).contains(&i) && (0..side).contains(&j))
                .then(|| ((i * side + j) * 2) as usize + a.s)
        }
        fn get(&self, a: Atom) -> Option<usize> {
            self.slot(a).and_then(|k| self.table[k])
        }
    }

    /// The atoms of the honeycomb whose projections on the three lattice directions, in half
    /// lattice constants, are at most `size`, less those with fewer than two neighbours; the rim
    /// is their outer share `rim` by that measure.
    pub fn flake(size: i64, rim: f64) -> Flake {
        let cells = || -size..=size;
        let candidates: Vec<Atom> = cells()
            .flat_map(|i| cells().flat_map(move |j| [0, 1].map(|s| Atom { i, j, s })))
            .filter(|a| a.reach() <= size as f64 + 1e-9)
            .collect();
        let lookup = Lookup::new(&candidates, size);
        let degree = |a: &Atom| {
            a.neighbours()
                .iter()
                .filter(|n| lookup.get(**n).is_some())
                .count()
        };
        let atoms: Vec<Atom> = candidates
            .iter()
            .filter(|a| degree(a) >= 2)
            .copied()
            .collect();
        let lookup = Lookup::new(&atoms, size);

        let mut first = Vec::new();
        let mut second = Vec::new();
        let mut senses = Vec::new();
        for (k, a) in atoms.iter().enumerate() {
            first.extend(
                a.neighbours()
                    .iter()
                    .filter_map(|n| lookup.get(*n))
                    .map(|m| [k, m]),
            );
            for (n, sense) in a.second_neighbours() {
                if let Some(m) = lookup.get(n) {
                    second.push([k, m]);
                    senses.push(sense);
                }
            }
        }
        // The second lattice direction: x turned by 60 degrees.
        let b = (xy() * (-core::f64::consts::PI / 6.0)).exp() >> x();
        let positions = atoms
            .iter()
            .map(|a| {
                let (u, v) = a.uv();
                x() * u + b * v
            })
            .collect();
        let facing = |a: &Atom| {
            let (u, v) = a.uv();
            2.0 * u + v
        };
        let front = atoms.iter().map(facing).fold(f64::MIN, f64::max);
        let start = (0..atoms.len())
            .filter(|k| facing(&atoms[*k]) == front)
            .min_by(|p, q| atoms[*p].uv().1.abs().total_cmp(&atoms[*q].uv().1.abs()))
            .expect("an edge facing x");
        Flake {
            positions,
            sublattice: atoms.iter().map(|a| a.sign() as f64).collect(),
            first,
            second,
            senses,
            rim: atoms
                .iter()
                .map(|a| a.reach() > (1.0 - rim) * size as f64)
                .collect(),
            start,
        }
    }

    /// Couplings between the atoms of the given pairs.
    fn couplings(pairs: &[[usize; 2]], cells: Vec<Coupling>, atoms: usize) -> SparseMap<Coupling> {
        let rows = pairs.iter().map(|p| p[0]).collect();
        let cols = pairs.iter().map(|p| p[1]).collect();
        SparseMap::new(cells, rows, cols, (atoms, atoms))
    }

    /// The flake's Hamiltonian: minus one for each hop between neighbours, the spin-orbit
    /// coupling times `xy` (signed by the sense the path turns) for each hop to a second
    /// neighbour, and the mass signed by sublattice on each atom.
    pub fn hamiltonian(flake: &Flake, spin_orbit: f64, mass: f64) -> SparseMap<Coupling> {
        let atoms = flake.sublattice.len();
        let hop = couplings(
            &flake.first,
            vec![Coupling::new(-1.0, 0.0); flake.first.len()],
            atoms,
        );
        let turns = flake
            .senses
            .iter()
            .map(|s| Coupling::new(0.0, s * spin_orbit))
            .collect();
        let turn = couplings(&flake.second, turns, atoms);
        let stagger = flake
            .sublattice
            .iter()
            .map(|s| Coupling::new(s * mass, 0.0))
            .collect();
        &(&hop + &turn) + &SparseMap::from_diagonal(&Field::new(stagger))
    }

    /// Energies and the fields of their states, values of the kind `K`.
    type Eigenpairs<K> = (Vec<f64>, Vec<Field<<K as Kind>::Mv<(), f64>>>);

    /// The `count` eigenpairs nearest zero of field map cells `C` on one sector, against the
    /// identity on that sector (`unit`, its open slot): orthonormal states.
    fn eigenstates<C, K>(energy: &SparseMap<C>, unit: C, count: usize) -> Eigenpairs<K>
    where
        C: Extensor<Kind = K, Slots = (K,), Coef = f64> + core::ops::Add<Output = C>,
        K: Kind,
    {
        let identity = SparseMap::from_diagonal(&Field::from_fn(energy.shape().0, |_| unit));
        energy.eigh(&identity, count)
    }

    /// The `count` spin-up levels nearest zero (each held twice, by a state and the state times
    /// `xy`), solved among the spin-up spinors and embedded as spinors of space.
    pub fn up_levels(energy: &SparseMap<Coupling>, count: usize) -> (Vec<f64>, Vec<Spinors>) {
        let (values, states) = eigenstates(&(energy * Up::slot()), Up::slot(), count);
        let embedded = states.iter().map(|f| f.map(Rotor::from)).collect();
        (values, embedded)
    }

    /// The `count` levels nearest zero among all spinors, both spins.
    pub fn levels(energy: &SparseMap<Coupling>, count: usize) -> (Vec<f64>, Vec<Spinors>) {
        let cells = energy.map_cells(|c| Rotor::from(c) * Rotor::slot());
        eigenstates(&cells, Rotor::slot(), count)
    }

    /// A state's density on each atom.
    pub fn density(state: &Spinors) -> Vec<f64> {
        state.iter().map(|s| s.norm_squared()).collect()
    }

    /// How much of a density lies on the rim.
    pub fn rim_share(flake: &Flake, density: &[f64]) -> f64 {
        let on_rim: f64 = density
            .iter()
            .zip(&flake.rim)
            .filter(|(_, r)| **r)
            .map(|(d, _)| d)
            .sum();
        on_rim / density.iter().sum::<f64>()
    }

    /// An electron started on atom `start` with `spinor`, keeping only its part in the given
    /// states, after `time`: each state turns on its right at the rate of its energy, and the
    /// spin density on each atom is `psi z ~psi`, as long as the electron's density there.
    pub fn spread(
        states: &[Spinors],
        energies: &[f64],
        start: usize,
        spinor: Spinor,
        time: f64,
    ) -> Field<Vector> {
        let turned: Vec<Spinor> = states
            .iter()
            .zip(energies)
            .map(|(s, e)| {
                let weight = s.site(start).reverse().scalar_product(spinor).s();
                (xy() * (-e * time)).exp().into_inner() * weight
            })
            .collect();
        let atoms = states[0].len();
        Field::from_fn(atoms, |a| {
            let psi: Spinor = states
                .iter()
                .zip(&turned)
                .map(|(s, t)| s.site(a) * *t)
                .sum();
            psi >> z()
        })
    }

    /// The flake, numga's: its size (edge to centre, in half lattice constants) and the outer
    /// share counted as its rim; the spin-orbit coupling and the masses swept past it.
    pub const SIZE: i64 = 12;
    pub const RIM: f64 = 0.3;
    pub const SPIN_ORBIT: f64 = 0.1;
    pub const MASS_SWEEP: (f64, f64) = (0.0, 0.8);
    /// The levels followed through the sweep, and the states inside the gap without a mass.
    pub const LEVELS: usize = 32;
    pub const IN_GAP: usize = 32;
    /// The bulk gap's half width per unit of spin-orbit coupling, `3 √3`.
    pub const GAP_PER_COUPLING: f64 = 5.196_152_422_706_632;
    /// The time for the edge states to run once around the flake.
    pub const LAP: f64 = 50.0;

    /// The electron's spinor: its spin along x, half up and half down, `(1 - zx) / √2`.
    pub fn spinor_along_x() -> Spinor {
        Rotor::new(1.0, 0.0, -1.0, 0.0).normalized()
    }

    /// `count` masses evenly over the sweep.
    pub fn masses(count: usize) -> Vec<f64> {
        let (from, to) = MASS_SWEEP;
        linspace(from, to, count)
    }

    /// The spin-up levels nearest zero at one mass: their energies (ascending), each one's share
    /// on the rim, and the density of the state nearest zero.
    pub struct Spectrum {
        pub energies: Vec<f64>,
        pub shares: Vec<f64>,
        pub nearest: Vec<f64>,
    }

    /// The spin-up spectrum at each mass, one sparse eigenproblem each. The state nearest zero
    /// is one of the `LEVELS` (numga solves for it again, with two levels). Where two levels are
    /// equally near, at `E` and `-E` without a mass, their states have the same density: the
    /// sublattice sign times the reverse takes one to the other.
    pub fn sweep(flake: &Flake, masses: &[f64]) -> Vec<Spectrum> {
        par_map(masses, |m| {
            let (energies, states) = up_levels(&hamiltonian(flake, SPIN_ORBIT, *m), LEVELS);
            let densities: Vec<Vec<f64>> = states.iter().map(density).collect();
            let nearest = (0..energies.len())
                .min_by(|a, b| energies[*a].abs().total_cmp(&energies[*b].abs()))
                .expect("levels");
            Spectrum {
                shares: densities.iter().map(|d| rim_share(flake, d)).collect(),
                nearest: densities[nearest].clone(),
                energies,
            }
        })
    }

    /// The states inside the gap without a mass, both spins: their energies and states.
    pub fn helical(flake: &Flake) -> (Vec<f64>, Vec<Spinors>) {
        levels(&hamiltonian(flake, SPIN_ORBIT, 0.0), IN_GAP)
    }
}

use kane_mele::*;

const SECONDS: f32 = 12.0;
/// The masses of the drawn sweep, half numga's 41: each is a sparse eigenproblem of 32 levels.
const DRAWN_MASSES: usize = 21;

/// What the frames share, computed once.
struct Data {
    flake: Flake,
    masses: Vec<f64>,
    spectra: Vec<Spectrum>,
    in_gap: (Vec<f64>, Vec<Spinors>),
}

fn data() -> Data {
    let flake = kane_mele::flake(SIZE, RIM);
    let masses = masses(DRAWN_MASSES);
    let spectra = sweep(&flake, &masses);
    let in_gap = helical(&flake);
    Data {
        flake,
        masses,
        spectra,
        in_gap,
    }
}

/// An atom's place in the drawing's plane.
fn place(p: Vector) -> Point2 {
    from_above(reach3(p))
}

/// Axes that fit the flake in a rectangle.
fn flake_axes(rect: Rect, f: &Flake) -> Axes {
    let reach = f.positions.iter().map(|p| p.norm()).fold(0.0, f64::max) as f32;
    Axes::equal(rect, ORIGIN2, reach * 1.08)
}

/// The flake's bonds, faint.
fn draw_bonds(c: &mut Canvas, ax: &Axes, f: &Flake) {
    let segs: Vec<[Point2; 2]> = f
        .first
        .iter()
        .filter(|[a, b]| a < b)
        .map(|[a, b]| [place(f.positions.site(*a)), place(f.positions.site(*b))])
        .collect();
    ax.stroke(c, &segs, 0.8, palette::GRID.faded(0.6));
}

/// The spin density of the electron at `time`: a dot on each atom sized by the density, red
/// where the spin is up and blue where it is down.
fn draw_spin(c: &mut Canvas, ax: &Axes, d: &Data, time: f64) {
    let (energies, states) = &d.in_gap;
    let spins = spread(states, energies, d.flake.start, spinor_along_x(), time);
    let largest = spins.iter().map(|s| s.norm()).fold(0.0, f64::max);
    for (p, s) in d.flake.positions.iter().zip(spins.iter()) {
        let share = (s.norm() / largest) as f32;
        if share > 0.02 {
            let tone = colormap::coolwarm(0.5 + 0.5 * (s.e3() / largest) as f32);
            ax.scatter(c, &[place(p)], Marker::Dot, 3.0 + 13.0 * share, tone);
        }
    }
}

/// A density on the flake, as dots sized and shaded by it.
fn draw_density(c: &mut Canvas, ax: &Axes, f: &Flake, density: &[f64]) {
    let largest = density.iter().copied().fold(0.0, f64::max);
    for (p, rho) in f.positions.iter().zip(density) {
        let share = (rho / largest) as f32;
        if share > 0.02 {
            ax.scatter(
                c,
                &[place(p)],
                Marker::Dot,
                2.0 + 7.0 * share,
                colormap::inferno(0.25 + 0.75 * share),
            );
        }
    }
}

/// The spin-up levels nearest zero against the mass, each shaded by its share on the rim, and
/// the mass at which the bulk gap closes.
fn draw_levels(c: &mut Canvas, ax: &Axes, d: &Data, mass: f64) {
    ax.frame(c, "", "MASS", "");
    let title = ax.rect.top_middle() - Point2::direction(0.0, 26.0);
    let text = "SPIN-UP ENERGIES NEAREST ZERO";
    c.text(text, title, 11.0, palette::INK, Align::Center);
    let transition = (GAP_PER_COUPLING * SPIN_ORBIT) as f32;
    let across = [
        Point2::xy(transition, ax.y[0]),
        Point2::xy(transition, ax.y[1]),
    ];
    ax.dashed(c, &across, 1.0, 4.0, palette::INK.faded(0.5));
    for (m, spectrum) in d.masses.iter().zip(&d.spectra) {
        for (e, share) in spectrum.energies.iter().zip(&spectrum.shares) {
            let tone = colormap::inferno(0.15 + 0.85 * *share as f32);
            ax.scatter(
                c,
                &[Point2::xy(*m as f32, *e as f32)],
                Marker::Dot,
                3.5,
                tone,
            );
        }
    }
    let cursor = [
        Point2::xy(mass as f32, ax.y[0]),
        Point2::xy(mass as f32, ax.y[1]),
    ];
    ax.polyline(c, &cursor, 1.0, palette::ORANGE.faded(0.6));
    let note = Point2::xy(transition + 0.02, ax.y[1] * 0.9);
    ax.text(c, note, "GAP CLOSES", 9.0, palette::INK, Align::Left);
}

fn draw(c: &mut Canvas, d: &Data, t: f32) {
    backdrop(c);
    let screen = c.rect();
    let progress = f64::from((t / SECONDS).fract());
    let time = LAP * progress;
    let below = Point2::direction(0.0, 1.0);

    let (from, to) = MASS_SWEEP;
    let sweep_axes = Axes::new(
        screen.part(0.08, 0.3, 0.32, 0.82),
        [from as f32, to as f32],
        [-0.62, 0.62],
    );
    draw_levels(c, &sweep_axes, d, from + (to - from) * progress);

    let main = screen.part(0.36, 0.17, 0.72, 0.93);
    let ax = flake_axes(main, &d.flake);
    draw_bonds(c, &ax, &d.flake);
    draw_spin(c, &ax, d, time);
    let title = main.top_middle() - below.gp(4.0);
    let text = format!("AN ELECTRON FROM THE EDGE  T = {time:.1}");
    c.text(&text, title, 11.0, palette::INK, Align::Center);
    let note = main.bottom_middle() + below.gp(12.0);
    let text = "SPIN UP (RED) CLOCKWISE, SPIN DOWN (BLUE) COUNTERCLOCKWISE";
    c.text(text, note, 9.0, palette::INK, Align::Center);

    // The state nearest zero at the sweep's ends, below the transition and above.
    let ends = [0, d.masses.len() - 1];
    for (k, end) in ends.into_iter().enumerate() {
        let (mass, density) = (d.masses[end], &d.spectra[end].nearest);
        let rect = screen.part(0.76, 0.17 + 0.42 * k as f32, 0.98, 0.53 + 0.42 * k as f32);
        let ax = flake_axes(rect, &d.flake);
        draw_bonds(c, &ax, &d.flake);
        draw_density(c, &ax, &d.flake, density);
        let label = rect.top_middle() - below.gp(4.0);
        let text = format!("NEAREST ZERO, MASS {mass}");
        c.text(&text, label, 10.0, palette::INK, Align::Center);
    }
    caption(
        c,
        "THE KANE-MELE FLAKE: HELICAL EDGE STATES",
        "VGA3D: A SPINOR ON EACH ATOM, THE HAMILTONIAN A SPARSE FIELD MAP",
    );
}

fn main() {
    run_with(Anim::new("kane mele", SECONDS).size(960, 540), data, draw);
}

#[cfg(test)]
mod tests {
    use super::kane_mele::*;
    use gax::field::{Field, SparseMap};
    use gax::vga2d::Scalar;
    use gax::vga3d::Vector;

    /// The masses of numga's sweep.
    const MASSES: usize = 41;

    /// numga's `test_spin_up_levels_are_those_of_the_complex_hamiltonian`: with the spin-orbit
    /// hop `i` times the turn's sense, the complex Hamiltonian of the spin-up amplitudes has the
    /// levels of the spin-up spinors. numga solves the complex matrix with NumPy; here it is
    /// written out from the flake's pairs as the real symmetric matrix of its real and imaginary
    /// parts, on twice the atoms, which holds each level twice.
    #[test]
    fn spin_up_levels_are_those_of_the_complex_hamiltonian() {
        let f = flake(8, RIM);
        let (spin_orbit, mass, count) = (0.1, 0.3, 12);
        let n = f.sublattice.len();
        let s = Scalar::new;
        let mut triples = Vec::new();
        for part in [0, n] {
            for (a, m) in f.sublattice.iter().enumerate() {
                triples.push((part + a, part + a, s(mass * m)));
            }
            for [a, b] in &f.first {
                triples.push((part + a, part + b, s(-1.0)));
            }
        }
        // `i λ ν` couples the imaginary parts into the real ones and back.
        for ([a, b], sense) in f.second.iter().zip(&f.senses) {
            triples.push((*a, n + b, s(-spin_orbit * sense)));
            triples.push((n + a, *b, s(spin_orbit * sense)));
        }
        let (rows, (cols, cells)): (Vec<_>, (Vec<_>, Vec<_>)) =
            triples.into_iter().map(|(r, c, x)| (r, (c, x))).unzip();
        let complex = &SparseMap::new(cells, rows, cols, (2 * n, 2 * n)) * Scalar::slot();
        let identity = SparseMap::from_diagonal(&Field::from_fn(2 * n, |_| Scalar::slot()));
        let (expected, _) = complex.eigh(&identity, 2 * count);
        let (energies, _) = up_levels(&hamiltonian(&f, spin_orbit, mass), 2 * count);
        for (got, want) in energies.iter().zip(expected) {
            assert!((got - want).abs() < 1e-10, "{got} vs {want}");
        }
    }

    /// numga's checks of the sweep: the Hamiltonian is its own reverse; every spin-up level holds
    /// its energy twice; without a mass the levels inside the bulk gap lie on the rim, and with
    /// the mass well past the transition the gap is empty.
    #[test]
    fn the_sweep_has_rim_states_in_the_gap_until_the_mass_closes_it() {
        let f = flake(SIZE, RIM);
        let energy = hamiltonian(&f, SPIN_ORBIT, 0.0);
        let reversed = energy.reverse();
        let couplings = |m: &SparseMap<Coupling>| {
            let mut c: Vec<_> = m.couplings().map(|(r, k, x)| (r, k, x.c)).collect();
            c.sort_by_key(|(row, column, _)| (*row, *column));
            c
        };
        assert_eq!(couplings(&reversed), couplings(&energy));
        let masses = masses(MASSES);
        let spectra = sweep(&f, &masses);
        for s in &spectra {
            let e = &s.energies;
            assert!(e.chunks(2).all(|p| (p[0] - p[1]).abs() < 1e-8), "{e:?}");
        }
        let half = |m: f64| (GAP_PER_COUPLING * SPIN_ORBIT - m).abs();
        let (first, last) = (&spectra[0], spectra.last().expect("masses"));
        let inside: Vec<usize> = (0..LEVELS)
            .filter(|k| first.energies[*k].abs() < 0.9 * half(masses[0]))
            .collect();
        assert!(inside.len() >= 8);
        assert!(inside.iter().all(|k| first.shares[*k] > 0.9));
        let lowest = last.energies.iter().fold(f64::MAX, |m, e| m.min(e.abs()));
        assert!(lowest > 0.5 * half(masses[MASSES - 1]), "{lowest}");
    }

    /// numga's check of the nearest states: without a mass the state nearest zero lies on the
    /// rim; with a large one it spreads inside.
    #[test]
    fn the_nearest_state_leaves_the_rim_past_the_transition() {
        let f = flake(SIZE, RIM);
        let spectra = sweep(&f, &[0.0, 0.8]);
        let shares: Vec<f64> = spectra.iter().map(|s| rim_share(&f, &s.nearest)).collect();
        assert!(shares[0] > 0.9 && shares[1] < 0.6, "{shares:?}");
    }

    /// numga's checks of the helical scene: the states are inside the bulk gap, on the rim, and
    /// four to an energy; the electron keeps its density, stays on the rim, and a quarter lap on
    /// its spin-up half has turned clockwise around the flake and its spin-down half
    /// counterclockwise.
    #[test]
    fn the_two_spins_run_around_the_rim_in_opposite_senses() {
        let f = flake(SIZE, RIM);
        let (energies, states) = helical(&f);
        let gap = GAP_PER_COUPLING * SPIN_ORBIT;
        assert!(energies.iter().all(|e| e.abs() < gap));
        let fourfold = energies.chunks(4).all(|q| (q[0] - q[3]).abs() < 1e-8);
        assert!(fourfold, "{energies:?}");
        assert!(states.iter().all(|s| rim_share(&f, &density(s)) > 0.9));

        const FRAMES: usize = 50;
        let at = |frame: usize| {
            spread(
                &states,
                &energies,
                f.start,
                spinor_along_x(),
                LAP * frame as f64 / FRAMES as f64,
            )
        };
        let total = |spins: &Field<Vector>| spins.iter().map(|s| s.norm()).sum::<f64>();
        let first = total(&at(0));
        for frame in 0..=FRAMES {
            let spins = at(frame);
            let lengths: Vec<f64> = spins.iter().map(|s| s.norm()).collect();
            assert!((total(&spins) - first).abs() < 1e-9 * first);
            assert!(rim_share(&f, &lengths) > 0.9);
        }
        let spins = at(FRAMES / 4);
        let start = f.positions.site(f.start).normalized();
        let sine = |sign: f64| {
            let half: Vec<f64> = spins
                .iter()
                .map(|s| (s.norm() + sign * s.e3()) / 2.0)
                .collect();
            let centre: Vector = f
                .positions
                .iter()
                .zip(&half)
                .map(|(p, h)| p * *h)
                .sum::<Vector>()
                / half.iter().sum::<f64>();
            (xy() | (centre.normalized() ^ start)).s()
        };
        let (up, down) = (sine(1.0), sine(-1.0));
        assert!(up < -0.5 && down > 0.5, "{up} {down}");
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws_with(super::draw, &super::data(), 0.5);
    }
}
