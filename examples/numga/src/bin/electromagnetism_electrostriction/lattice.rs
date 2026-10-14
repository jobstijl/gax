//! The lattice of polarizable particles: its response, its energies, and their derivatives,
//! through the dipoles' solve and by the chain rule.

use super::pairs::between_others;
use gax::Real;
use gax::dual::hessian;
use gax::field::{Field, SparseLu, SparseMap};
use gax::vga2d::{Pseudoscalar, Scalar, Vector};
use gax_numga_examples::mesh::from_columns;

pub type V = Vector<(), f64>;
/// A vector with coefficients `T`: a position, or one carrying derivatives.
pub type Vt<T> = Vector<(), T>;
/// A map on one site's vectors, `Vector <- Vector`, with coefficients `T`.
pub type Cell<T> = Vector<(Vector,), T>;
/// A field map over the particles with map cells.
pub type FieldMap<T> = SparseMap<Cell<T>>;
/// A linear form on one site's displacement: the energy's first derivative there.
pub type Form = Scalar<(Vector,), f64>;
/// A form on two sites' displacements: the energy's second derivative between them.
pub type Form2 = Scalar<(Vector, Vector), f64>;

pub const X: V = V::new(1.0, 0.0);
pub const Y: V = V::new(0.0, 1.0);

/// The unit vector at `angle` counter-clockwise from `x`: `x` turned by `exp(-angle/2 e12)`.
pub fn direction(angle: f64) -> V {
    Pseudoscalar::new(-angle / 2.0).exp() >> X
}

/// A value of `f64` as one of coefficients `T`.
pub fn lift<T: Real>(v: V) -> Vt<T> {
    v.map_coefs(T::from_f64)
}

/// The field a dipole makes at separation `r` from it, as a map from the dipole.
pub fn dipole_field<T: Real>(r: Vt<T>) -> Cell<T> {
    let open = Vector::slot();
    let squared = r.norm_squared();
    let fifth = squared * squared * r.norm();
    let three = T::from_f64(3.0);
    (r * (r | open) * three - open * squared) * (T::one() / fifth)
}

/// The energy of two dipoles held fixed at separation `r`: minus the one's pairing with the
/// other's field.
pub fn pair_energy<T: Real>(r: Vt<T>, p: V, q: V) -> T {
    -(lift::<T>(p) | dipole_field(r).of(lift(q))).s()
}

/// A bond's stretching energy at `link`, for its rest length and stiffness.
pub fn bond_energy<T: Real>(link: Vt<T>, rest: f64, stiffness: f64) -> T {
    let extension = link.norm() - T::from_f64(rest);
    T::from_f64(stiffness / 2.0) * extension * extension
}

/// An elastic lattice of polarizable particles.
#[derive(Clone, Debug)]
pub struct Lattice {
    pub rest: Field<V>,
    /// Each bond's two particles.
    pub bonds: Vec<[usize; 2]>,
    pub polarizability: f64,
    pub stiffness: f64,
    pub tether: f64,
}

/// The derivatives of the energy at a shape: a form per site, and a form per pair of sites.
pub struct Derivatives {
    pub gradient: Field<Form>,
    pub curvature: SparseMap<Form2>,
}

impl Lattice {
    pub fn particles(&self) -> usize {
        self.rest.len()
    }

    /// Each bond's second particle less its first, from the particles to the bonds.
    pub fn bond_ends<T: Real>(&self) -> SparseMap<Scalar<(), T>> {
        let sign = |end: usize| Scalar::new(T::from_f64([-1.0, 1.0][end]));
        from_columns(&self.bonds, self.particles(), |_, end| sign(end))
    }

    /// Each bond, directed from its first particle to its second.
    pub fn links<T: Real>(&self, positions: &Field<Vt<T>>) -> Field<Vt<T>> {
        &self.bond_ends() * positions
    }

    /// Each bond's rest length.
    pub fn lengths(&self) -> Vec<f64> {
        self.links(&self.rest).iter().map(|l| l.norm()).collect()
    }

    /// The dipoles' own restoring field, less the field supplied by the other dipoles: the
    /// map from the dipoles to the applied field that holds them.
    pub fn response<T: Real>(&self, positions: &Field<Vt<T>>) -> FieldMap<T> {
        let n = positions.len();
        let restoring = Vt::<T>::slot() * T::from_f64(1.0 / self.polarizability);
        let others = between_others(n, |i, j| {
            -dipole_field(positions.site(i) - positions.site(j))
        });
        &SparseMap::from_diagonal(&Field::from_fn(n, |_| restoring)) + &others
    }

    /// The dipoles at a shape in the same applied field at every particle.
    pub fn dipoles<T: Real>(&self, positions: &Field<Vt<T>>, applied: Vt<T>) -> Field<Vt<T>> {
        let fields = Field::from_fn(positions.len(), |_| applied);
        self.response(positions).solve_direct(&fields)
    }

    /// The stretching energy of the bonds and the displacement energy of the tethers.
    #[cfg_attr(not(test), allow(dead_code))] // the tests differentiate it through the solve
    pub fn elastic_energy<T: Real>(&self, positions: &Field<Vt<T>>) -> T {
        let links = self.links(positions);
        let bonds = self
            .lengths()
            .iter()
            .zip(links.iter())
            .fold(T::zero(), |e, (&rest, link)| {
                e + bond_energy(link, rest, self.stiffness)
            });
        let tethers = positions
            .iter()
            .zip(self.rest.iter())
            .fold(T::zero(), |e, (p, r)| e + (p - lift(r)).norm_squared());
        bonds + T::from_f64(self.tether / 2.0) * tethers
    }

    /// The electrical energy at fixed applied field once the dipoles have equilibrated:
    /// building them costs energy, and at their minimum the whole electric energy, that cost
    /// and their mutual interaction included, is minus half their pairing with the field.
    #[cfg_attr(not(test), allow(dead_code))] // the tests differentiate it through the solve
    pub fn electrical_energy<T: Real>(&self, positions: &Field<Vt<T>>, applied: V) -> T {
        let applied = lift::<T>(applied);
        let dipoles = self.dipoles(positions, applied);
        let pairing = dipoles.iter().fold(T::zero(), |e, p| e + (applied | p).s());
        -pairing * T::from_f64(0.5)
    }

    /// The mechanical energy plus the polarization energy after the dipoles have equilibrated.
    #[cfg_attr(not(test), allow(dead_code))] // the tests differentiate it through the solve
    pub fn energy<T: Real>(&self, positions: &Field<Vt<T>>, applied: V) -> T {
        self.elastic_energy(positions) + self.electrical_energy(positions, applied)
    }

    /// The energy's derivatives at a shape, by the chain rule over its terms: those of the
    /// elastic energy and of the dipoles' pairs with the dipoles held fixed, less the
    /// softening as the dipoles readjust.
    pub fn derivatives(&self, positions: &Field<V>, applied: V) -> Derivatives {
        let n = self.particles();
        let lu = self
            .response(positions)
            .factor()
            .expect("a stable response");
        let dipoles = lu.solve(&Field::from_fn(n, |_| applied));
        let mut terms = Assembly::new(n);
        self.elastic_terms(positions, &mut terms);
        held_dipole_terms(positions, &dipoles, &mut terms);
        self.readjustment_terms(positions, &dipoles, &lu, &mut terms);
        terms.finish()
    }

    /// The derivatives of the bonds' stretching and the tethers' displacement energies.
    fn elastic_terms(&self, positions: &Field<V>, terms: &mut Assembly) {
        let links = self.links(positions);
        for (([a, b], link), rest) in self.bonds.iter().zip(links.iter()).zip(self.lengths()) {
            let (_, g, h) = hessian(|l| bond_energy(l, rest, self.stiffness), link);
            terms.difference(*b, *a, g, h);
        }
        let tether = Form2::from_coeffs([[[self.tether, 0.0], [0.0, self.tether]]]);
        for i in 0..self.particles() {
            let offset = (positions.site(i) - self.rest.site(i)) * self.tether;
            terms.site(i, Form::from_coeffs([offset.c]), tether);
        }
    }

    /// The softening as the dipoles readjust, `-Jᵀ R⁻¹ J`: `J` the change of the response's
    /// image of the dipoles as each coordinate of each site moves, solved by the response's
    /// factorization `lu`.
    fn readjustment_terms(
        &self,
        positions: &Field<V>,
        dipoles: &Field<V>,
        lu: &SparseLu<Vector, f64>,
        terms: &mut Assembly,
    ) {
        let changes: Vec<[Field<V>; 2]> = (0..self.particles())
            .map(|k| self.response_change(positions, dipoles, k))
            .collect();
        let solved: Vec<[Field<V>; 2]> = changes
            .iter()
            .map(|columns| columns.each_ref().map(|c| lu.solve(c)))
            .collect();
        for (i, change) in changes.iter().enumerate() {
            for (k, solution) in solved.iter().enumerate() {
                let entry = |a: usize, c: usize| -change[a].dot(&solution[c]);
                let rows = [0, 1].map(|a| [0, 1].map(|c| entry(a, c)));
                terms.couple(i, k, Form2::from_coeffs([rows]));
            }
        }
    }

    /// How the response's image of the dipoles changes as site `k` moves along `x` and `y`:
    /// two fields over the particles, from one evaluation with the site's coordinates as
    /// the derivatives' directions.
    fn response_change(&self, positions: &Field<V>, dipoles: &Field<V>, k: usize) -> [Field<V>; 2] {
        type D = gax::dual::Dual<f64, 2>;
        let seeded: Field<Vt<D>> = Field::from_fn(positions.len(), |i| {
            let v = positions.site(i);
            let coefficient = |c: usize| {
                if i == k {
                    D::variable(v.c[c], c)
                } else {
                    D::constant(v.c[c])
                }
            };
            Vt::<D>::new(coefficient(0), coefficient(1))
        });
        let image: Field<Vt<D>> = self.response(&seeded).apply(&dipoles.map(lift::<D>));
        [0, 1].map(|c| image.map(|v| v.map_coefs(|d| d.du[c])))
    }

    /// One Newton step: the displacement that solves the curvature against the gradient.
    pub fn newton_step(&self, positions: &Field<V>, applied: V) -> Field<V> {
        let Derivatives {
            gradient,
            curvature,
        } = self.derivatives(positions, applied);
        let stiffness = curvature.map_cells(Scalar::as_map);
        stiffness.solve_direct(&gradient.map(through_metric))
    }

    /// Newton relaxation from the reference lattice.
    pub fn equilibrium(&self, applied: V, steps: usize) -> Field<V> {
        (0..steps).fold(self.rest.clone(), |x, _| {
            &x - &self.newton_step(&x, applied)
        })
    }
}

/// The derivatives of the dipoles' pair energies with the dipoles held fixed: each pair's
/// energy depends on the difference of the two sites alone.
fn held_dipole_terms(positions: &Field<V>, dipoles: &Field<V>, terms: &mut Assembly) {
    let n = positions.len();
    for i in 0..n {
        for j in i + 1..n {
            let (p, q) = (dipoles.site(i), dipoles.site(j));
            let r = positions.site(i) - positions.site(j);
            let (_, g, h) = hessian(|r| pair_energy(r, p, q), r);
            terms.difference(i, j, g, h);
        }
    }
}

/// The vector a form on the plane's vectors measures with: the same coefficients, through the
/// Euclidean metric.
pub fn through_metric(form: Form) -> V {
    V::from_coeffs(form.c[0])
}

/// The derivatives of a sum of terms, added up: each term's gradient at its site, and its
/// second derivative between two sites.
struct Assembly {
    gradient: Vec<Form>,
    couplings: Vec<(usize, usize, Form2)>,
}

impl Assembly {
    fn new(n: usize) -> Assembly {
        Assembly {
            gradient: vec![Form::zero(); n],
            couplings: Vec::new(),
        }
    }

    /// A term of one site.
    fn site(&mut self, i: usize, gradient: Form, curvature: Form2) {
        self.gradient[i] += gradient;
        self.couple(i, i, curvature);
    }

    /// A term of the difference `x_i - x_j` of two sites: its derivatives carried to both.
    fn difference(&mut self, i: usize, j: usize, gradient: Form, curvature: Form2) {
        self.gradient[i] += gradient;
        self.gradient[j] -= gradient;
        self.couple(i, i, curvature);
        self.couple(j, j, curvature);
        self.couple(i, j, -curvature);
        self.couple(j, i, -curvature);
    }

    /// A second derivative between two sites.
    fn couple(&mut self, i: usize, j: usize, curvature: Form2) {
        self.couplings.push((i, j, curvature));
    }

    fn finish(self) -> Derivatives {
        let n = self.gradient.len();
        let (rows, cols): (Vec<usize>, Vec<usize>) =
            self.couplings.iter().map(|c| (c.0, c.1)).unzip();
        let cells = self.couplings.into_iter().map(|c| c.2).collect();
        Derivatives {
            gradient: Field::new(self.gradient),
            curvature: SparseMap::new(cells, rows, cols, (n, n)),
        }
    }
}

/// The root mean square distance along a direction, measured from the field's centre.
pub fn extent(positions: &Field<V>, direction: V) -> f64 {
    let centre = positions.mean();
    let squares = positions.iter().map(|p| {
        let along = ((p - centre) | direction).s();
        along * along
    });
    #[allow(clippy::disallowed_methods)] // a root mean square: a statistic
    (squares.sum::<f64>() / positions.len() as f64).sqrt()
}

/// The relative change of the root mean square extent along a direction, from the rest
/// sites.
pub fn strain(positions: &Field<V>, rest: &Field<V>, direction: V) -> f64 {
    extent(positions, direction) / extent(rest, direction) - 1.0
}

/// The axial indices of a hexagonal patch of a triangular lattice, `rings` around a centre,
/// and its nearest-neighbour bonds, each lower index first.
pub fn hexagonal_indices(rings: i64) -> (Vec<[i64; 2]>, Vec<[usize; 2]>) {
    let layer = |a: i64, b: i64| a.abs().max(b.abs()).max((a + b).abs());
    let indices: Vec<[i64; 2]> = (-rings..=rings)
        .flat_map(|a| (-rings..=rings).map(move |b| [a, b]))
        .filter(|&[a, b]| layer(a, b) <= rings)
        .collect();
    let neighbours = |p: [i64; 2], q: [i64; 2]| layer(p[0] - q[0], p[1] - q[1]) == 1;
    let bonds = (0..indices.len())
        .flat_map(|i| (i + 1..indices.len()).map(move |j| [i, j]))
        .filter(|&[i, j]| neighbours(indices[i], indices[j]))
        .collect();
    (indices, bonds)
}

// --- scenes ---------------------------------------------------------------------------

pub const RINGS: i64 = 2;
pub const SPACING: f64 = 1.0;
pub const POLARIZABILITY: f64 = 0.035 * SPACING * SPACING * SPACING;
pub const STIFFNESS: f64 = 1.0;
pub const TETHER: f64 = 0.15 * STIFFNESS;
pub const FIELD: f64 = 3.05;
pub const NEWTON_STEPS: usize = 4;
pub const LOAD_SAMPLES: usize = 25;

/// The lattice: its two directions meet at sixty degrees.
pub fn lattice() -> Lattice {
    let (indices, bonds) = hexagonal_indices(RINGS);
    let across = direction(core::f64::consts::PI / 3.0);
    let rest = indices
        .iter()
        .map(|&[a, b]| (X * a as f64 + across * b as f64) * SPACING)
        .collect();
    Lattice {
        rest,
        bonds,
        polarizability: POLARIZABILITY,
        stiffness: STIFFNESS,
        tether: TETHER,
    }
}

/// The applied field a share of the way round its loading cycle: it turns once while a
/// smooth rise and fall closes the loop at the unloaded lattice. Every share is its own
/// static equilibrium; the share is a control setting rather than time.
pub fn turning(share: f64) -> V {
    let turned = core::f64::consts::TAU * share;
    let strength = FIELD * (0.5 - 0.5 * gax_numga_examples::signal::phasor(turned).e20());
    direction(turned) * strength
}
