//! The math: a solid tetrahedral torus at rest in one instant of spacetime (`gax::sta`, numga's
//! `x+y+z+t-`), its Whitney edge and face forms, the curl, the magnetic energy and helicity of
//! edge potentials, the flux-preserving subspace, the Beltrami eigenmodes, and their selective
//! decay at fixed helicity. No drawing.
//!
//! Points are antivectors, intrinsic trivectors, directions of space (numga's `Point-`); three
//! edges joined are six times a tetrahedron's volume (`Volume-`, an intrinsic vector), and the
//! volume joined with the observer's time is the prism it sweeps in a unit of time, a number. A
//! hat is constant in time, so its gradient is the patch its opposite face sweeps through time
//! over three times the prism. The magnetic field is a plane of space and reads its flux through
//! a patch by the join.

use gax::field::{Field, SparseMap};
use gax::mesh::Topology;
use gax::sta;

use crate::simplicial::{
    boundary, corners as local_corners, exterior, incidence, interior, oriented,
};

/// A number (numga's `Scalar+`).
pub type Scalar = sta::Scalar;
/// A gradient (numga's `Plane+`).
pub type Plane = sta::Vector;
/// The magnetic field, a bivector of space (numga's `Field+`; the full bivector here, its time
/// parts zero).
pub type Magnetic = sta::Bivector;
/// A point or a direction of space (numga's `Point-`).
pub type Point = sta::intrinsic::Trivector;
/// A map of numbers.
pub type ScalarMap = sta::Scalar<(Scalar,)>;
/// An energy's cell, from edge amplitudes to amounts (numga's `Amount- <- Scalar+`).
pub type EnergyMap = sta::intrinsic::Pseudoscalar<(Scalar,)>;

/// The point at `(x, y, z)` at time zero: the metric dual of its vector.
pub fn point(x: f64, y: f64, z: f64) -> Point {
    Plane::new(x, y, z, 0.0).hodge()
}

/// The observer's direction in time, which pairs to one with `t` (numga's `-mv.t.dual()`).
pub fn time() -> Point {
    -Plane::new(0.0, 0.0, 0.0, 1.0).hodge()
}

/// A point's place, read by the planes of space.
pub fn coordinates(p: Point) -> [f64; 3] {
    [
        (Plane::new(1.0, 0.0, 0.0, 0.0) & p).s(),
        (Plane::new(0.0, 1.0, 0.0, 0.0) & p).s(),
        (Plane::new(0.0, 0.0, 1.0, 0.0) & p).s(),
    ]
}

/// The rotor turning by `angle` in the plane `plane`, numga's `exp(plane · (−angle / 2))`.
fn turn(plane: Magnetic, angle: f64) -> gax::Unit<sta::Even> {
    (plane * (-angle / 2.0)).exp()
}

/// The planes `xy` and `xz` of space (`xz` is `−zx`).
pub fn xy() -> Magnetic {
    Magnetic::new(0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
}

pub fn xz() -> Magnetic {
    Magnetic::new(0.0, -1.0, 0.0, 0.0, 0.0, 0.0)
}

/// The Whitney maps, the curl, and the energy and helicity of a tetrahedral mesh.
pub struct Operators {
    pub mesh: Topology<4>,
    pub vertices: Field<Point>,
    pub centres: Field<Point>,
    /// Each tetrahedron's signed volume: its first three edges joined, over six.
    pub volumes: Field<sta::intrinsic::Vector>,
    /// Vertex potentials to piecewise-constant gradients, with no metric.
    #[cfg_attr(not(test), allow(dead_code))] // numga's operators, which the tests check
    pub gradient: SparseMap<sta::Vector<(Scalar,)>>,
    /// Edge amplitudes to their Whitney 1-forms at the centroids.
    #[cfg_attr(not(test), allow(dead_code))] // numga's operators, which the tests check
    pub whitney1: SparseMap<sta::Vector<(Scalar,)>>,
    /// Edge amplitudes to the derivative of their Whitney forms: the magnetic field.
    pub curl: SparseMap<sta::Bivector<(Scalar,)>>,
    /// Face fluxes to their Whitney 2-forms at the centroids.
    #[cfg_attr(not(test), allow(dead_code))] // numga's operators, which the tests check
    pub whitney2: SparseMap<sta::Bivector<(Scalar,)>>,
    pub stiffness: SparseMap<EnergyMap>,
    pub helicity: SparseMap<EnergyMap>,
}

impl Operators {
    pub fn new(mesh: &Topology<4>, vertices: &Field<Point>) -> Self {
        let slot = Scalar::slot();
        let corners: Vec<Field<Point>> = (0..4)
            .map(|k| incidence(mesh, 3, 0).pick(k, vertices))
            .collect();
        let centres = Field::from_fn(mesh.len(3), |t| {
            (0..4)
                .map(|k| corners[k].site(t))
                .fold(Point::zero(), |a, b| a + b)
                * 0.25
        });
        let edges = &mesh.d::<Scalar>(0) * vertices;
        let own = oriented(mesh, 3, 1).read_all(&edges);
        let volumes = Field::from_fn(mesh.len(3), |t| {
            (own[0].site(t) & own[1].site(t) & own[2].site(t)) * (1.0 / 6.0)
        });
        let prisms = volumes.map(|v| v & time());
        // Each face's patch, half the join of two sides, read as the face facing each corner,
        // outward.
        let sides = boundary(mesh, 2).read_all(&edges);
        let patches = sides[0].zip_with(&sides[1], |a, b| (a & b) * 0.5);
        let face_patches = boundary(mesh, 3).read_all(&patches);
        let thrice = prisms.map(|p| (p * 3.0).inverse());
        let corner_gradients: Vec<Field<Plane>> = (0..4)
            .map(|k| face_patches[k].zip_with(&thrice, |f, p| (f & time()) * p))
            .collect();
        let gradient = incidence(mesh, 3, 0).assemble(|t, k| corner_gradients[k].site(t) * slot);
        // At each centroid the four hats are a quarter: an edge's 1-form is a quarter of the
        // difference of its ends' gradients.
        let ends = local_corners(3, 1);
        let own_edges = oriented(mesh, 3, 1);
        let whitney1 = own_edges.assemble(|t, e| {
            let (tail, head) = (ends[e][0], ends[e][1]);
            (corner_gradients[head].site(t) - corner_gradients[tail].site(t)) * 0.25 * slot
        });
        // The derivative of an edge's Whitney form is twice the wedge of its ends' gradients.
        let curl = own_edges.assemble(|t, e| {
            let (tail, head) = (ends[e][0], ends[e][1]);
            (corner_gradients[tail].site(t) ^ corner_gradients[head].site(t)) * 2.0 * slot
        });
        let whitney2 = boundary(mesh, 3).assemble(|t, k| {
            (((centres.site(t) - corners[k].site(t)) & time()) * thrice.site(t)) * slot
        });
        // The energy pulls the vacuum, each tetrahedron's field to the excitation it causes
        // weighted by its volume, back through the curl; the helicity pulls the metric-free volume
        // pairing back through the centroid 1-forms and the curl.
        let vacuum = SparseMap::from_diagonal(&prisms.map(|p| Magnetic::slot().hodge() * p.norm()));
        let stiffness = curl.adjugate().compose(&vacuum.compose(&curl));
        let volume_weights = SparseMap::from_diagonal(&volumes.map(|v| Plane::slot() ^ v));
        let raw = curl.adjugate().compose(&volume_weights.compose(&whitney1));
        let helicity = &(&raw + &raw.adjugate()) * 0.5;
        Operators {
            mesh: mesh.clone(),
            vertices: vertices.clone(),
            centres,
            volumes,
            gradient,
            whitney1,
            curl,
            whitney2,
            stiffness,
            helicity,
        }
    }
}

/// A reflection-symmetric solid torus about the `z` axis: a core circle, and rings about it, each
/// slice's wedge three tetrahedra between it and the next, the slices' diagonals alternating, every
/// tetrahedron turned to a positive volume (numga's `solid_torus`).
pub fn solid_torus(
    major: f64,
    minor: f64,
    n_major: usize,
    n_minor: usize,
) -> (Topology<4>, Field<Point>) {
    let unit = point(1.0, 0.0, 0.0);
    let around_axis = |u: usize| turn(xy(), core::f64::consts::TAU * u as f64 / n_major as f64);
    let around_tube = |v: usize| turn(xz(), core::f64::consts::TAU * v as f64 / n_minor as f64);
    let core_circle = (0..n_major).map(|u| around_axis(u) >> (unit * major));
    let rim = (0..n_major).flat_map(|u| {
        (0..n_minor)
            .map(move |v| around_axis(u) >> (unit * major + (around_tube(v) >> (unit * minor))))
    });
    let vertices: Field<Point> = core_circle.chain(rim).collect();
    let slice = |u: usize, local: usize| {
        if local == 0 {
            u
        } else {
            n_major + u * n_minor + local - 1
        }
    };
    let mut blocks: [Vec<[usize; 4]>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for u in 0..n_major {
        let next = (u + 1) % n_major;
        let (u0, u1) = if u % 2 == 0 { (u, next) } else { (next, u) };
        for v in 0..n_minor {
            let mut cross = [0, 1 + v, 1 + (v + 1) % n_minor];
            cross.sort_unstable();
            let [a, b, d] = cross;
            let (a0, b0, d0) = (slice(u0, a), slice(u0, b), slice(u0, d));
            let (a1, b1, d1) = (slice(u1, a), slice(u1, b), slice(u1, d));
            blocks[0].push([a0, b0, d0, d1]);
            blocks[1].push([a0, b0, d1, b1]);
            blocks[2].push([a0, b1, d1, a1]);
        }
    }
    let raw: Vec<[usize; 4]> = blocks.concat();
    // Each tetrahedron's three edges from its first corner, joined, span six times its signed
    // volume; where it is negative, the last two corners swap.
    let tets: Vec<[usize; 4]> = raw
        .iter()
        .map(|&t| {
            let p = t.map(|v| vertices.site(v));
            let volume = ((p[1] - p[0]) & (p[2] - p[0]) & (p[3] - p[0])) & time();
            if volume.s() < 0.0 {
                [t[0], t[1], t[3], t[2]]
            } else {
                t
            }
        })
        .collect();
    let mesh = Topology::new(vertices.len(), &tets).expect("the solid torus");
    (mesh, vertices)
}

/// The flux-preserving edge subspace, `[edges <- free]`: one value per interior edge, then the
/// amplitude of the closed poloidal form on the wall, which carries the net toroidal flux while
/// the normal flux through every wall triangle stays nothing. Each wall edge that steps around
/// the tube takes the amplitude weighted by its step, over the root of their count.
pub fn flux_preserving(mesh: &Topology<4>, n_major: usize, n_minor: usize) -> SparseMap<ScalarMap> {
    let wall = exterior(mesh, 1);
    let inside = interior(mesh, 1);
    let ring = |v: usize| (v as i64 - n_major as i64).rem_euclid(n_minor as i64);
    let half = (n_minor / 2) as i64;
    let steps: Vec<(usize, i64)> = wall
        .iter()
        .filter_map(|&e| {
            let ends = mesh.simplex(1, e);
            let step = (ring(ends[1]) - ring(ends[0]) + half).rem_euclid(n_minor as i64) - half;
            (step != 0).then_some((e, step))
        })
        .collect();
    let count = root(steps.len() as f64);
    let n = inside.len();
    let triples: Vec<_> = inside
        .iter()
        .enumerate()
        .map(|(k, &e)| (e, k, 1.0))
        .chain(steps.iter().map(|&(e, step)| (e, n, step as f64 / count)))
        .map(|(r, c, x)| (r, c, Scalar::slot() * x))
        .collect();
    SparseMap::from_triples(triples, (mesh.len(1), n + 1))
}

/// The square root of a number, for the numerics of scalar sums (an energy normalisation): not
/// geometry. (gax's `Scalar::sqrt` is
/// the versor root `normalize(1 + s)`, one for any positive number.)
#[allow(clippy::disallowed_methods)]
pub fn root(x: f64) -> f64 {
    x.sqrt()
}

/// The inverse Beltrami eigenvalues `μ = 1 / λ`, ascending, the coexact eigenmodes over the free
/// values, orthonormal in the energy, and the number of gradient (zero-energy) directions: the
/// energy and the helicity pulled back onto the free values, the helicity diagonalized against
/// the energy on the energy's range (gax's dense `eigh_all` against a semidefinite metric drops
/// the gradients, which the energy does not measure), and the achiral (zero-helicity) modes
/// dropped.
pub fn coexact_helicity_eigh(
    free: &SparseMap<ScalarMap>,
    stiffness: &SparseMap<EnergyMap>,
    helicity: &SparseMap<EnergyMap>,
) -> (Vec<f64>, Vec<Field<Scalar>>, usize) {
    let reduced = |map: &SparseMap<EnergyMap>| free.adjugate().compose(&map.compose(free));
    let energy = reduced(stiffness);
    let (mu, modes) = reduced(helicity).eigh_all(&energy);
    let null = energy.shape().0 - mu.len();
    let largest = mu.iter().map(|x| x.abs()).fold(0.0, f64::max);
    let (mu, modes) = mu
        .into_iter()
        .zip(modes)
        .filter(|(m, _)| m.abs() > 1e-8 * largest)
        .unzip();
    (mu, modes, null)
}

/// The modal amplitudes of a selective decay at fixed helicity: each damped by `exp(−λ² t)`, the
/// finer faster, and all scaled together to keep their initial helicity `Σ a² / λ`.
pub fn selective_decay(initial: &[f64], eigenvalues: &[f64], times: &[f64]) -> Vec<Vec<f64>> {
    let helicity = |a: &[f64]| {
        a.iter()
            .zip(eigenvalues)
            .map(|(a, l)| a * a / l)
            .sum::<f64>()
    };
    let start = helicity(initial);
    times
        .iter()
        .map(|&t| {
            let damped: Vec<f64> = initial
                .iter()
                .zip(eigenvalues)
                .map(|(a, l)| a * (-l * l * t).exp())
                .collect();
            let scale = root(start / helicity(&damped));
            damped.iter().map(|a| a * scale).collect()
        })
        .collect()
}

/// The edge field of modal amplitudes.
pub fn superpose(amplitudes: &[f64], modes: &[Field<Scalar>]) -> Field<Scalar> {
    let mut out = Field::<Scalar>::zeros(modes[0].len());
    for (a, m) in amplitudes.iter().zip(modes) {
        out = &out + &m.map(|x| x * *a);
    }
    out
}

/// The magnetic field of each modal state: the curl of its edge field.
pub fn reconstruct(
    states: &[Vec<f64>],
    modes: &[Field<Scalar>],
    curl: &SparseMap<sta::Bivector<(Scalar,)>>,
) -> Vec<Field<Magnetic>> {
    states
        .iter()
        .map(|a| curl.of(&superpose(a, modes)))
        .collect()
}
