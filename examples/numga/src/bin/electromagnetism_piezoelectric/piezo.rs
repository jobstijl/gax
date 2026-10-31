//! The math: a piezoelectric material's three maps from its directions, the strain and the
//! triangles of a strip's mesh, the enthalpy's three blocks pulled back onto the nodes, and the
//! coupled equilibrium of displacements and potentials. No drawing.
//!
//! In numga's spacetime of the plane (`x+y+t-`, `shared/sta2d.rs`), the planes of space
//! (`Spatial`, `x y`) read positions and displacements, which are their intrinsic complements
//! (`Temporal`, `xt yt`, intrinsic: numga's `Point-`); an electric field is a plane through time
//! (`Temporal`, extrinsic), a charge an intrinsic amount. The elastic map pairs two changes of
//! shape, each a displacement with the gradient it varies along; the piezoelectric map sends one
//! to a polarization; the dielectric map sends a field to the polarization it induces.

use gax::field::{Field, SparseMap};
use gax::mesh::Topology;

use crate::simplicial::{boundary, incidence};
use crate::sta2d::{self, time};

/// A number.
pub type Scalar = sta2d::Scalar;
/// A plane of space, a gradient or a force (numga's `Plane+`, `x y`).
pub type Plane = sta2d::Spatial;
/// A position or a displacement, a plane's complement (numga's `Point-`, `xt yt`).
pub type Point = sta2d::intrinsic::Temporal;
/// An electric field, a plane through time (numga's `Field+`, `xt yt`).
pub type Electric = sta2d::Temporal;
/// A charge (numga's `Amount-`).
pub type Amount = sta2d::intrinsic::Pseudoscalar;
/// A strain: directions to stretch readings (numga's `Strain`).
pub type Strain = sta2d::Spatial<(Point,)>;
/// The elastic form, pairing two changes of shape (numga's `Elastic`).
pub type Elastic = sta2d::Scalar<(Point, Plane, Point, Plane)>;
/// The piezoelectric map, a change of shape to a polarization (numga's `Polarization`).
pub type Polarization = sta2d::intrinsic::Temporal<(Point, Plane)>;
/// The dielectric map, a field to the polarization it induces (numga's `Permittivity`).
pub type Permittivity = sta2d::intrinsic::Temporal<(Electric,)>;
/// The blocks' cells: displacement to force, potential to force, potential to charge.
pub type MechanicalMap = sta2d::Spatial<(Point,)>;
pub type CouplingMap = sta2d::Spatial<(Scalar,)>;
pub type DielectricMap = sta2d::intrinsic::Pseudoscalar<(Scalar,)>;

/// The planes `x` and `y` of space.
pub fn x() -> Plane {
    Plane::new(1.0, 0.0)
}

pub fn y() -> Plane {
    Plane::new(0.0, 1.0)
}

/// A plane's complement, the direction it reads (numga's `dual`, the metric one).
pub fn dual(p: Plane) -> Point {
    sta2d::Vector::new(p.ex(), p.ey(), 0.0)
        .hodge()
        .cast::<sta2d::Temporal>()
}

/// A direction's plane (numga's `dual_inverse`): the dual twice is minus one on vectors here.
pub fn undual(p: Point) -> Plane {
    (-sta2d::intrinsic::Bivector::from(p).hodge()).cast::<sta2d::Spatial>()
}

/// The strip's axis and its poling direction, the duals of `x` and `y`.
pub fn axis() -> Point {
    dual(x())
}

pub fn pole() -> Point {
    dual(y())
}

/// A point's place, read by `x` and `y`.
pub fn coordinates(p: Point) -> [f64; 2] {
    [(x() & p).s(), (y() & p).s()]
}

/// The point at `(x, y)`.
pub fn point(x: f64, y: f64) -> Point {
    axis() * x + pole() * y
}

/// The material's constants, transverse and parallel to its poling (numga's arrays).
pub struct Constants {
    pub stretch_stiffness: [f64; 2],
    pub cross_stiffness: f64,
    pub shear_stiffness: f64,
    pub stretch_polarization: [f64; 2],
    pub shear_polarization: f64,
    pub permittivities: [f64; 2],
}

/// A material's three maps.
#[derive(Clone, Copy)]
pub struct Material {
    pub elastic: Elastic,
    pub polarization: Polarization,
    pub permittivity: Permittivity,
}

/// The constitutive maps from orthonormal material planes, transverse then parallel to poling.
/// Each plane reads extension along its own direction: a displacement read by the plane, times
/// the gradient read on the plane's direction; the mixed reading is shear. Extension polarizes
/// along the poling direction, shear across it; the field is read on each material direction
/// swept through time.
pub fn constitutive(directions: [Plane; 2], c: &Constants) -> Material {
    let [transverse, poled] = directions;
    let reading = |d: Plane, along: Plane| (d & Point::slot()) * (Plane::slot() & dual(along));
    let stretches = directions.map(|d| reading(d, d));
    let shear = reading(transverse, poled) + reading(poled, transverse);
    let elastic = stretches[0] * stretches[0] * c.stretch_stiffness[0]
        + stretches[1] * stretches[1] * c.stretch_stiffness[1]
        + (stretches[0] * stretches[1] + stretches[1] * stretches[0]) * c.cross_stiffness
        + shear * shear * c.shear_stiffness;
    let polarization = (dual(poled)
        * (stretches[0] * c.stretch_polarization[0] + stretches[1] * c.stretch_polarization[1])
        + dual(transverse) * shear * c.shear_polarization)
        .cast::<sta2d::Temporal>();
    let permittivity = (0..2)
        .map(|i| {
            let d = dual(directions[i]);
            (d * (Electric::slot() & (d & time())) * c.permittivities[i]).cast::<sta2d::Temporal>()
        })
        .fold(Permittivity::zero(), |a, b| a + b);
    Material {
        elastic,
        polarization,
        permittivity,
    }
}

/// Each triangle's area and its corners' shape-function gradients: a corner's shape function
/// rises from nothing on the facing side to one at the corner, so its gradient is the plane
/// through that side and time, over twice the signed area (the sides joined, read against time).
pub fn triangles(
    mesh: &Topology<3>,
    vertices: &Field<Point>,
) -> (Field<Scalar>, [Field<Plane>; 3]) {
    let edges = &mesh.d::<Scalar>(0) * vertices;
    let sides = boundary(mesh, 2).read_all(&edges);
    let double = sides[0].zip_with(&sides[1], |a, b| (a & b) & time());
    let gradients = [0, 1, 2]
        .map(|k| sides[k].zip_with(&double, |s, d| ((time() & s) * d.inverse()).cast::<Plane>()));
    (double.map(|d| Scalar::new(d.norm() / 2.0)), gradients)
}

/// The symmetric displacement gradient on each triangle, a map from directions to stretch
/// readings: the corners' displacements, each as a plane, times their gradients' readings,
/// summed, and made symmetric with its adjugate.
pub fn strain(
    mesh: &Topology<3>,
    displacement: &Field<Point>,
    gradients: &[Field<Plane>; 3],
) -> Vec<Strain> {
    let at = incidence(mesh, 2, 0);
    (0..mesh.len(2))
        .map(|f| {
            let gradient = (0..3)
                .map(|k| {
                    (undual(displacement.site(at.sites[f][k]))
                        * (gradients[k].site(f) & Point::slot()))
                    .cast::<sta2d::Spatial>()
                })
                .fold(Strain::zero(), |a, b| a + b);
            (gradient + gradient.adjugate()) * 0.5
        })
        .collect()
}

/// The electric field of a potential on each triangle: minus the potential's gradient swept
/// through time.
#[cfg(test)]
pub fn electric_field(
    mesh: &Topology<3>,
    potential: &Field<Scalar>,
    gradients: &[Field<Plane>; 3],
) -> Vec<Electric> {
    let at = incidence(mesh, 2, 0);
    (0..mesh.len(2))
        .map(|f| {
            (0..3)
                .map(|k| sweep(gradients[k].site(f)) * -potential.site(at.sites[f][k]).s())
                .fold(Electric::zero(), |a, b| a + b)
        })
        .collect()
}

/// A plane of space swept through time: the plane `g ^ t`.
pub fn sweep(g: Plane) -> Electric {
    (g ^ sta2d::t_plane()).cast::<sta2d::Temporal>()
}

/// The enthalpy's three blocks, as maps from the nodal fields to their work-duals: forces from
/// displacements, forces from potentials (whose adjugate, read backwards, is charge from
/// motion), charges from potentials.
pub struct Blocks {
    pub mechanical: SparseMap<MechanicalMap>,
    pub coupling: SparseMap<CouplingMap>,
    pub dielectric: SparseMap<DielectricMap>,
}

/// The blocks of a strip: each shape gradient bound into the continuum maps gives a triangle's
/// nodal work, polarization and field, the test corner first; the work form read as the force
/// paired with a test displacement; the polarization swept through time, the field's work-dual,
/// read backwards for the converse effect; and every triangle's blocks summed onto the pairs of
/// its corners, weighted by its area and the strip's width.
pub fn assemble(
    mesh: &Topology<3>,
    areas: &Field<Scalar>,
    gradients: &[Field<Plane>; 3],
    material: &Material,
    poling: &[f64],
    width: f64,
) -> Blocks {
    let pairing = Plane::slot() & Point::slot();
    let at = incidence(mesh, 2, 0);
    let (mut mechanical, mut coupling, mut dielectric) = (Vec::new(), Vec::new(), Vec::new());
    for (f, &sign) in poling.iter().enumerate() {
        let g = [0, 1, 2].map(|k| gradients[k].site(f));
        let weight = width * areas.site(f).s();
        let fields = g.map(|g| ((sweep(g) * -1.0) * Scalar::slot()).cast::<sta2d::Temporal>());
        let polarized = g.map(|g| material.polarization.at::<1>().of(g) * sign);
        for a in 0..3 {
            for b in 0..3 {
                let (va, vb) = (at.sites[f][a], at.sites[f][b]);
                let work = material.elastic.at::<1>().of(g[b]).at::<2>().of(g[a]);
                let force: MechanicalMap = pairing.solve(work);
                mechanical.push((va, vb, force * weight));
                let mixed = -((polarized[a] & time()).adjugate().of(fields[b]));
                coupling.push((va, vb, mixed.cast::<sta2d::Spatial>() * weight));
                let induced =
                    (material.permittivity.of(fields[b]) & time()).cast::<sta2d::Spatial>();
                let charge = -(fields[a].adjugate().of(induced));
                dielectric.push((va, vb, charge * weight));
            }
        }
    }
    let n = mesh.len(0);
    Blocks {
        mechanical: SparseMap::from_triples(mechanical, (n, n)),
        coupling: SparseMap::from_triples(coupling, (n, n)),
        dielectric: SparseMap::from_triples(dielectric, (n, n)),
    }
}

/// The free displacements put back at their vertices, and the voltages: each interior node
/// its own, the floating electrode's nodes all one.
pub struct Boundary {
    pub movement: SparseMap<sta2d::intrinsic::Temporal<(Point,)>>,
    pub voltage: SparseMap<sta2d::Scalar<(Scalar,)>>,
}

/// The coupled equilibrium: the displacements and potentials at which the forces balance the
/// load and the free charge vanishes, with the electrodes held at `prescribed` (zero elsewhere)
/// and the floating one's voltage free.
///
/// numga's sparse solve takes a map of one field type and this system couples two, so the
/// blocks become one scalar field map, each cell's work form read on the basis of its test and
/// trial fields (the adjugate of the coupling is its transpose there), and solved by sparse LU.
pub fn equilibrium(
    blocks: &Blocks,
    boundary: &Boundary,
    prescribed: &Field<Scalar>,
    force: &Field<Plane>,
) -> (Field<Point>, Field<Scalar>) {
    let (movement, voltage) = (&boundary.movement, &boundary.voltage);
    let elastic = movement
        .adjugate()
        .compose(&blocks.mechanical.compose(movement));
    let mixed = movement
        .adjugate()
        .compose(&blocks.coupling.compose(voltage));
    let capacitance = voltage
        .adjugate()
        .compose(&blocks.dielectric.compose(voltage));
    let load = movement
        .adjugate()
        .of(&(force - &blocks.coupling.of(prescribed)));
    let charge = -&voltage.adjugate().of(&blocks.dielectric.of(prescribed));

    let basis = [
        Point::from_coeffs([1.0, 0.0]),
        Point::from_coeffs([0.0, 1.0]),
    ];
    let (free, volts) = (elastic.shape().0, capacitance.shape().0);
    let number = |x: f64| Scalar::slot() * x;
    let mut triples = Vec::new();
    for (i, j, cell) in elastic.couplings() {
        for (r, &test) in basis.iter().enumerate() {
            for (c, &trial) in basis.iter().enumerate() {
                triples.push((2 * i + r, 2 * j + c, number((test & cell.of(trial)).s())));
            }
        }
    }
    for (i, v, cell) in mixed.couplings() {
        for (r, &test) in basis.iter().enumerate() {
            let w = (test & cell.of(Scalar::new(1.0))).s();
            triples.push((2 * i + r, 2 * free + v, number(w)));
            triples.push((2 * free + v, 2 * i + r, number(w)));
        }
    }
    for (v, w, cell) in capacitance.couplings() {
        let value = (Scalar::new(1.0) & cell.of(Scalar::new(1.0))).s();
        triples.push((2 * free + v, 2 * free + w, number(value)));
    }
    let size = 2 * free + volts;
    let system = SparseMap::from_triples(triples, (size, size));
    let rhs: Field<Scalar> = load
        .iter()
        .flat_map(|l| basis.map(|b| b & l))
        .chain(charge.iter().map(|q| Scalar::new(1.0) & q))
        .collect();
    let solved = system.solve_direct(&rhs);
    let displacement: Field<Point> = (0..free)
        .map(|i| basis[0] * solved.site(2 * i).s() + basis[1] * solved.site(2 * i + 1).s())
        .collect();
    let potential: Field<Scalar> = (0..volts).map(|v| solved.site(2 * free + v)).collect();
    (
        movement.of(&displacement),
        prescribed + &voltage.of(&potential),
    )
}

/// The free charge on each vertex: minus the converse coupling of the displacements and the
/// dielectric's of the potentials.
pub fn charges(
    blocks: &Blocks,
    displacement: &Field<Point>,
    potential: &Field<Scalar>,
) -> Field<Amount> {
    -&(&blocks.coupling.adjugate().of(displacement) + &blocks.dielectric.of(potential))
}

/// Each triangle's axial strain, its strain's reading of the axis on the axis.
pub fn axial_strain(
    mesh: &Topology<3>,
    displacement: &Field<Point>,
    gradients: &[Field<Plane>; 3],
) -> Vec<f64> {
    strain(mesh, displacement, gradients)
        .iter()
        .map(|s| (axis() & s.of(axis())).s())
        .collect()
}
