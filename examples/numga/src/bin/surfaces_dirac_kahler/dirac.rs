//! The math: a triangle mesh's Whitney maps and Dirac–Kähler operators, the Hodge decomposition
//! of a velocity on a perforated disk, and the waves of the even field; and numga's two domains.
//! No drawing.
//!
//! Points are antivectors, intrinsic (numga's `Point-`), the Hodge duals of vectors; a corner's
//! hat gradient is the side facing it over twice the patch, a vector (numga's `Plane+`).

use crate::simplicial::{
    boundary, corners as local_corners, hex_disk, incidence, oriented, reading,
};
use gax::field::{Field, SparseMap};
use gax::mesh::Topology;
use gax::vga2d;

pub type Scalar<S = ()> = vga2d::Scalar<S>;
pub type Plane<S = ()> = vga2d::Vector<S>;
pub type Bivector<S = ()> = vga2d::Bivector<S>;
/// The even subalgebra: a scalar and a bivector (numga's `Even+`).
pub type Even<S = ()> = vga2d::Rotor<S>;
/// Every grade (numga's `Full+`).
pub type Full<S = ()> = vga2d::Multivector<S>;
pub type Point<S = ()> = vga2d::intrinsic::Vector<S>;
/// An area-weighted reading of the even field, its dual (numga's `Even-`).
pub type EvenAmount<S = ()> = vga2d::intrinsic::Rotor<S>;
/// An amount, a quantity per area (numga's `Antiscalar-`).
pub type Amount<S = ()> = vga2d::intrinsic::Pseudoscalar<S>;

/// The point at `(x, y)`.
pub fn point(x: f64, y: f64) -> Point {
    Plane::new(x, y).hodge()
}

/// A point's coordinates, read by joining it with the slices across `x` and `y`.
pub fn coordinates(p: Point) -> [f64; 2] {
    [
        (Plane::new(1.0, 0.0) & p).s(),
        (Plane::new(0.0, 1.0) & p).s(),
    ]
}

/// The Whitney maps and the mixed-grade Dirac–Kähler operators of an oriented triangle mesh.
#[cfg_attr(not(test), allow(dead_code))] // numga's operators, the tests read every one
pub struct Operators {
    pub mesh: Topology<3>,
    pub d1: SparseMap<Scalar>,
    pub areas: Field<Scalar>,
    pub corner_gradients: [Field<Plane>; 3],
    /// Vertex, edge and face cochains to their Whitney forms at the face centroids.
    pub whitney0: SparseMap<Scalar<(Scalar,)>>,
    pub gradient: SparseMap<Plane<(Scalar,)>>,
    pub whitney1: SparseMap<Plane<(Scalar,)>>,
    pub whitney2: SparseMap<Bivector<(Scalar,)>>,
    /// The corner gradients times a vertex multivector: divergence in grade 0, gradient plus the
    /// turned bivector's gradient in grade 1, curl in grade 2.
    pub dirac: SparseMap<Full<(Full,)>>,
    /// A velocity back to the gradient that drives it, over the area (numga's resistance; the
    /// medium drives the velocities below).
    pub isotropic_resistance: SparseMap<Plane<(Point,)>>,
    /// The corner gradients on even vertex fields: scalar potentials and stream bivectors
    /// together, to a 1-form on the faces; and its skew partner back.
    pub even_gradient: SparseMap<Plane<(Even,)>>,
    pub vertex_areas: Field<Scalar>,
    pub even_divergence: SparseMap<Even<(Plane,)>>,
    /// Velocities from a potential through the medium, from a stream function without a metric,
    /// and from both at once.
    pub potential_velocity: SparseMap<Point<(Scalar,)>>,
    pub stream_velocity: SparseMap<Point<(Scalar,)>>,
    pub even_velocity: SparseMap<Point<(Even,)>>,
    pub stream_laplacian: SparseMap<Amount<(Scalar,)>>,
    pub even_laplacian: SparseMap<EvenAmount<(Even,)>>,
}

impl Operators {
    pub fn new(mesh: &Topology<3>, vertices: &Field<Point>) -> Self {
        let d1 = mesh.d::<Scalar>(1);
        let edges = &mesh.d::<Scalar>(0) * vertices;
        let facing = boundary(mesh, 2);
        let sides: [Field<Point>; 3] = core::array::from_fn(|k| facing.read(k, &edges));
        let patches = sides[0].zip_with(&sides[1], |a, b| (a & b) * 0.5);
        let areas = patches.map(|p| Scalar::new(p.s().abs()));

        // Each opposite side over twice the signed patch: the three constant corner gradients.
        let twice = patches.map(|p| (p * 2.0).inverse());
        let corner_gradients: [Field<Plane>; 3] =
            core::array::from_fn(|k| sides[k].zip_with(&twice, |s, t| s * t));
        let at_corners = incidence(mesh, 2, 0);
        let whitney0 = &mesh.means::<Scalar>(2) * Scalar::slot();
        let gradient = at_corners.assemble(|f, k| corner_gradients[k].site(f) * Scalar::slot());

        // At each centroid the hats are a third: an edge's 1-form is a third of the difference
        // of its ends' gradients.
        let ends = local_corners(2, 1);
        let whitney1 = oriented(mesh, 2, 1).assemble(|f, k| {
            let (tail, head) = (ends[k][0], ends[k][1]);
            (corner_gradients[head].site(f) - corner_gradients[tail].site(f))
                * (1.0 / 3.0)
                * Scalar::slot()
        });
        let unit = Bivector::new(1.0);
        let whitney2 = SparseMap::from_diagonal(
            &patches.map(|p| unit * (unit & p).inverse() * Scalar::slot()),
        );

        // The corner gradients times a full multivector: every grade of the derivative at once.
        let dirac = at_corners.assemble(|f, k| corner_gradients[k].site(f) * Full::slot());

        let medium = Plane::slot().hodge();
        let isotropic_medium = SparseMap::from_diagonal(&areas.map(|a| medium * a.s()));
        let resistance = medium.inverse();
        let isotropic_resistance =
            SparseMap::from_diagonal(&areas.map(|a| resistance * a.s().recip()));

        // On the even subalgebra the scalar potentials and stream bivectors map together into a
        // pure 1-form on the faces.
        let even_gradient = at_corners.assemble(|f, k| corner_gradients[k].site(f) * Even::slot());
        let vertex_areas = &mesh.means::<Scalar>(2).reverse() * &areas;
        let mass = Even::slot().hodge().inverse();
        let inverse_vertex_mass =
            SparseMap::from_diagonal(&vertex_areas.map(|a| mass * a.s().recip()));
        // The 2D join flips the sign of odd grades and keeps that of even ones, so the adjugate
        // of the even gradient is already the skew partner that keeps the waves' energy.
        let even_divergence =
            inverse_vertex_mass.compose(&even_gradient.adjugate().compose(&isotropic_medium));

        // A velocity from a potential through the medium, and from a stream function without a
        // metric: its gradient met with the patch runs along its level curves.
        let potential_velocity = isotropic_medium.compose(&gradient);
        let along = SparseMap::from_diagonal(&patches.map(|p| Plane::slot() ^ p));
        let stream_velocity = along.compose(&gradient);
        let even_velocity = isotropic_medium.compose(&even_gradient);
        let stream_laplacian = stream_velocity
            .adjugate()
            .compose(&isotropic_resistance.compose(&stream_velocity));
        let even_laplacian = even_gradient.adjugate().compose(&even_velocity);
        Operators {
            mesh: mesh.clone(),
            d1,
            areas,
            corner_gradients,
            whitney0,
            gradient,
            whitney1,
            whitney2,
            dirac,
            isotropic_resistance,
            even_gradient,
            vertex_areas,
            even_divergence,
            potential_velocity,
            stream_velocity,
            even_velocity,
            stream_laplacian,
            even_laplacian,
        }
    }
}

/// The free sites of a field as a map from them onto all `n`, `[n <- free]`, with cells `K <- K`.
fn free_sites<K>(free: &[usize], n: usize, slot: K) -> SparseMap<K>
where
    K: gax::Extensor + core::ops::Mul<f64, Output = K>,
{
    reading::<Scalar>(free, n).transpose_with(|c| slot * c.s())
}

/// The harmonic stream function, one on the inner rim and zero on the outer, and its velocity,
/// which circulates around the hole.
pub fn harmonic_circulation(
    ops: &Operators,
    inner_ring: &[usize],
    interior: &[usize],
) -> (Field<Scalar>, Field<Point>) {
    let n = ops.mesh.len(0);
    let boundary_stream = Field::from_fn(n, |v| {
        Scalar::new(if inner_ring.contains(&v) { 1.0 } else { 0.0 })
    });
    let free = free_sites(interior, n, Scalar::slot());
    let rhs = -&free
        .adjugate()
        .of(&ops.stream_laplacian.of(&boundary_stream));
    let system = free
        .adjugate()
        .compose(&ops.stream_laplacian.compose(&free));
    let solved = system.solve_direct(&rhs);
    let stream = &boundary_stream + &free.of(&solved);
    let velocity = ops.stream_velocity.of(&stream);
    (stream, velocity)
}

/// The parts of a velocity on the faces: one mixed-grade Poisson solve on the even subalgebra
/// recovers the scalar potential and the stream bivector together, which split the velocity into
/// its potential, stream and harmonic parts, orthogonal.
#[cfg_attr(not(test), allow(dead_code))] // numga's core, checked by the tests
pub struct Decomposition {
    pub potential: Field<Scalar>,
    pub stream: Field<Scalar>,
    pub potential_flow: Field<Point>,
    pub stream_flow: Field<Point>,
    pub harmonic_flow: Field<Point>,
}

pub fn hodge_decomposition(
    ops: &Operators,
    flow: &Field<Point>,
    interior: &[usize],
) -> Decomposition {
    let n = ops.mesh.len(0);
    let free = free_sites(interior, n, Even::slot());
    // On every interior vertex star the adjugate of the even gradient reads the flow's
    // divergence in grade 0 and its circulation in grade 2.
    let readings = free.adjugate().of(&ops.even_gradient.adjugate().of(flow));
    let system = free.adjugate().compose(&ops.even_laplacian.compose(&free));
    let solved = system.solve_direct(&readings);
    let even = free.of(&solved);
    let potential = even.map(|e| e.grade::<0>());
    let stream = even.map(|e| -(e.grade::<2>() & Bivector::new(1.0).hodge()));
    let potential_flow = ops
        .even_velocity
        .of(&potential.map(|s| Even::new(s.s(), 0.0)));
    let stream_flow = ops.even_velocity.of(&even.map(|e| Even::new(0.0, e.e12())));
    let harmonic_flow = &(flow - &potential_flow) - &stream_flow;
    Decomposition {
        potential,
        stream,
        potential_flow,
        stream_flow,
        harmonic_flow,
    }
}

/// The waves: the even field on the vertices and the 1-form on the faces, stepped by fourth
/// order Runge–Kutta under the Dirac–Kähler operators, each state from the first.
pub fn propagate(
    ops: &Operators,
    even: Field<Even>,
    faces: Field<Plane>,
    dt: f64,
    steps: usize,
) -> Vec<(Field<Even>, Field<Plane>)> {
    let (div, grad) = (&ops.even_divergence, &ops.even_gradient);
    let rate = |(e, f): &(Field<Even>, Field<Plane>)| (div.of(f), grad.of(e));
    let shifted = |(e, f): &(Field<Even>, Field<Plane>),
                   (de, df): &(Field<Even>, Field<Plane>),
                   h: f64| { (e + &de.scaled(h), f + &df.scaled(h)) };
    let mut states = vec![(even, faces)];
    for _ in 0..steps {
        let y = states.last().expect("a state");
        let k1 = rate(y);
        let k2 = rate(&shifted(y, &k1, 0.5 * dt));
        let k3 = rate(&shifted(y, &k2, 0.5 * dt));
        let k4 = rate(&shifted(y, &k3, dt));
        let sum = |a: &Field<Even>, b: &Field<Even>, c: &Field<Even>, d: &Field<Even>| {
            &(&(a + &b.scaled(2.0)) + &c.scaled(2.0)) + d
        };
        let sum_faces = |a: &Field<Plane>, b: &Field<Plane>, c: &Field<Plane>, d: &Field<Plane>| {
            &(&(a + &b.scaled(2.0)) + &c.scaled(2.0)) + d
        };
        let next = (
            &y.0 + &sum(&k1.0, &k2.0, &k3.0, &k4.0).scaled(dt / 6.0),
            &y.1 + &sum_faces(&k1.1, &k2.1, &k3.1, &k4.1).scaled(dt / 6.0),
        );
        states.push(next);
    }
    states
}

/// The waves' energy: the area-weighted squared sizes of the even field and the 1-form, halved.
#[cfg_attr(not(test), allow(dead_code))] // numga's core, checked by the tests
pub fn energy(ops: &Operators, even: &Field<Even>, faces: &Field<Plane>) -> f64 {
    let at_vertices = ops.vertex_areas.zip_with(even, |a, e| a * e.norm_squared());
    let at_faces = ops.areas.zip_with(faces, |a, f| a * f.norm_squared());
    0.5 * (at_vertices.sum() + at_faces.sum()).s()
}

/// A disk of nearly equilateral triangles with its first vertex at the centre.
pub fn disk(radius: f64, divisions: usize) -> (Topology<3>, Field<Point>) {
    let (mesh, coordinates) = hex_disk(radius, divisions);
    (mesh, coordinates.map(|v| v.hodge()))
}

/// The perforated disk: staggered rings of vertices from the hole to the rim, their Delaunay
/// triangles less those spanning the hole (all three corners on the innermost ring); with the
/// inner ring's vertices and those off both rims.
pub struct Annulus {
    pub mesh: Topology<3>,
    pub vertices: Field<Point>,
    pub inner_ring: Vec<usize>,
    pub interior: Vec<usize>,
}

pub fn annulus(hole: f64, rim: f64, rings: usize) -> Annulus {
    let spacing = (rim - hole) / (rings - 1) as f64;
    // The side of an equilateral triangle whose height is the spacing.
    let edge = spacing / vga2d::Vector::from_angle(core::f64::consts::FRAC_PI_3).e2();
    let mut places = Vec::new();
    let mut ring_of = Vec::new();
    for ring in 0..rings {
        let radius = hole + spacing * ring as f64;
        let around = core::f64::consts::TAU * radius / edge;
        let population = (around.round_ties_even() as usize).max(18);
        for k in 0..population {
            let stagger = if ring % 2 == 1 { 0.5 } else { 0.0 };
            let angle = core::f64::consts::TAU * (k as f64 + stagger) / population as f64;
            places.push(vga2d::Vector::from_angle(angle) * radius);
            ring_of.push(ring);
        }
    }
    let placed: Field<gax::vga3d::Vector> = places
        .iter()
        .map(|v| gax::vga3d::Vector::new(v.e1(), v.e2(), 0.0))
        .collect();
    let faces: Vec<[usize; 3]> = gax_numga_examples::mesh::Mesh::triangulate(placed)
        .faces
        .into_iter()
        .filter(|f| !f.iter().all(|&v| ring_of[v] == 0))
        .collect();
    let mesh = Topology::new(places.len(), &faces).expect("the annulus");
    Annulus {
        mesh,
        vertices: places.iter().map(|v| v.hodge()).collect(),
        inner_ring: (0..places.len()).filter(|&v| ring_of[v] == 0).collect(),
        interior: (0..places.len())
            .filter(|&v| ring_of[v] > 0 && ring_of[v] < rings - 1)
            .collect(),
    }
}
