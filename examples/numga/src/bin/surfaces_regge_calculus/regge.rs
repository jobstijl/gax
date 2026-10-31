//! The math: a triangle's Regge metric from its edge lengths, the intrinsic geometry of a mesh
//! from them (areas, corner turns, angle defects), the conductivities they give, and steady
//! conduction; and numga's scenes. No drawing.
//!
//! Points are antivectors, intrinsic (numga's `Point-`), and gradients vectors (`Plane+`). A
//! metric is a correlation from displacements to planes (numga's `Plane+ <- Point-`), read on an
//! edge by the join: `metric(edge) & edge` is its squared length.

use crate::simplicial::{
    boundary, corners as local_corners, exterior, hex_disk, incidence, reading,
};
use gax::field::{Field, SparseMap};
use gax::mesh::Topology;
use gax::vga2d;
use gax_numga_examples::measure::roots;

pub type Scalar<S = ()> = vga2d::Scalar<S>;
pub type Plane<S = ()> = vga2d::Vector<S>;
pub type Bivector<S = ()> = vga2d::Bivector<S>;
pub type Rotor<S = ()> = vga2d::Rotor<S>;
pub type Point<S = ()> = vga2d::intrinsic::Vector<S>;
pub type Patch<S = ()> = vga2d::intrinsic::Scalar<S>;
/// An amount, a quantity per area (numga's `Amount-`).
pub type Amount<S = ()> = vga2d::intrinsic::Pseudoscalar<S>;
/// A metric: a displacement to the plane that reads it (numga's `Plane+ <- Point-`).
pub type Metric = Plane<(Point,)>;
/// A conductivity: a gradient to the current it drives (numga's `Point- <- Plane+`).
pub type Conductivity = Point<(Plane,)>;

pub const DISK_DIVISIONS: usize = 9;
pub const DISK_RADIUS: f64 = 1.0;
pub const ELECTRODE_THRESHOLD: f64 = 0.78;
/// The stretch of each side's squared length of the triangle.
pub const EDGE_STRETCHES: [f64; 3] = [1.45, 0.65, 1.25];
/// The three metrics of the disk: flat, radially stretched, stretched around.
pub const RADIAL_RATES: [f64; 3] = [0.0, 3.2, -0.45];
pub const HOOP_RATES: [f64; 3] = [0.0, -0.78, 3.2];

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

/// `√q`, half the distance between the roots of `t² − q` (`measure::roots`).
pub fn root(q: f64) -> f64 {
    roots(1.0, 0.0, -q).map_or(0.0, |(_, half)| half)
}

/// The sides of a triangle facing each corner, counter-clockwise.
pub fn sides(corners: [Point; 3]) -> [Point; 3] {
    [0, 1, 2].map(|k| corners[(k + 2) % 3] - corners[(k + 1) % 3])
}

/// The Regge basis of a triangle from its corner gradients: for the side facing each corner,
/// the symmetric product of the gradients of its two ends, negated and halved, which reads one
/// on that side and nothing on the other two. Along a side at one of its ends one gradient
/// vanishes; along the opposite side the two pair to `−1` and `+1`.
pub fn basis_of(gradients: [Plane; 3]) -> [Metric; 3] {
    let ends = local_corners(2, 1);
    core::array::from_fn(|k| {
        let (first, second) = (gradients[ends[k][0]], gradients[ends[k][1]]);
        -(first * (second & Point::slot()) + second * (first & Point::slot())) * 0.5
    })
}

/// A triangle's Regge basis, its corner gradients and its signed patch.
pub fn regge_metric_basis(corners: [Point; 3]) -> ([Metric; 3], [Plane; 3], Patch) {
    let sides = sides(corners);
    let patch = (sides[0] & sides[1]) * 0.5;
    let twice = (patch * 2.0).inverse();
    let gradients = sides.map(|s| s * twice);
    (basis_of(gradients), gradients, patch)
}

/// The intrinsic area of a triangle with these sides under a metric: the wedge of the metric's
/// readings of two sides carries its area scaling, paired with the reference patch.
pub fn intrinsic_area(metric: Metric, sides: [Point; 3], patch: Patch) -> f64 {
    let scaled = (metric.of(sides[0]) ^ metric.of(sides[1])) & patch;
    root(scaled.s() * 0.5)
}

/// The rotor at each corner: the metric inner product of the two sides there with twice the
/// intrinsic area in the bivector, normalized; its log is the corner's turn.
pub fn corner_rotors(metric: Metric, sides: [Point; 3], area: f64) -> [Rotor; 3] {
    let ends = local_corners(2, 1);
    core::array::from_fn(|k| {
        let dot = -(metric.of(sides[ends[k][0]]) & sides[ends[k][1]]).s();
        Rotor::new(dot, 2.0 * area).normalized()
    })
}

/// The corners' dot products under a metric (the metric inner products of the two sides at
/// each corner).
fn dots(metric: Metric, sides: [Point; 3]) -> [f64; 3] {
    let ends = local_corners(2, 1);
    core::array::from_fn(|k| -(metric.of(sides[ends[k][0]]) & sides[ends[k][1]]).s())
}

/// A triangle under prescribed squared side lengths: its metric, intrinsic area, unit corner
/// rotors and corner turns.
pub fn triangle_geometry(
    corners: [Point; 3],
    squared: [f64; 3],
) -> (Metric, f64, [Rotor; 3], [Bivector; 3]) {
    let (basis, _, patch) = regge_metric_basis(corners);
    let metric = basis[0] * squared[0] + basis[1] * squared[1] + basis[2] * squared[2];
    let sides = sides(corners);
    let area = intrinsic_area(metric, sides, patch);
    let rotors = corner_rotors(metric, sides, area);
    let turns = rotors.map(|r| r.unit().log());
    (metric, area, rotors, turns)
}

/// A mesh's intrinsic geometry under prescribed squared edge lengths: each face's metric and
/// intrinsic area, each vertex's angle defect and its curvature (the defect over the vertex's
/// area), and each face's conductivity.
#[cfg_attr(not(test), allow(dead_code))] // numga's core, checked by the tests
pub struct Intrinsic {
    pub metrics: Vec<Metric>,
    pub areas: Field<Scalar>,
    pub defects: Field<Bivector>,
    pub curvatures: Field<Bivector>,
    pub conductivities: Vec<Conductivity>,
    /// Half the cotangent at each corner of each face, under the metric.
    pub cotangents: Vec<[f64; 3]>,
}

pub fn mesh_geometry(
    mesh: &Topology<3>,
    vertices: &Field<Point>,
    squared: &Field<Scalar>,
) -> Intrinsic {
    let edges = &mesh.d::<Scalar>(0) * vertices;
    let facing = boundary(mesh, 2);
    let edge_of = incidence(mesh, 2, 1);
    let faces = mesh.len(2);
    let mut metrics = Vec::with_capacity(faces);
    let mut conductivities = Vec::with_capacity(faces);
    let mut cotangents = Vec::with_capacity(faces);
    let mut areas = Vec::with_capacity(faces);
    let mut turns: Vec<[Bivector; 3]> = Vec::with_capacity(faces);
    let all_sides = facing.read_all(&edges);
    for f in 0..faces {
        let sides: [Point; 3] = core::array::from_fn(|k| all_sides[k].site(f));
        let patch = (sides[0] & sides[1]) * 0.5;
        let twice = (patch * 2.0).inverse();
        let basis = basis_of(sides.map(|s| s * twice));
        let lengths: [f64; 3] = core::array::from_fn(|k| squared.site(edge_of.sites[f][k]).s());
        let metric = basis[0] * lengths[0] + basis[1] * lengths[1] + basis[2] * lengths[2];
        let area = intrinsic_area(metric, sides, patch);
        let rotors = corner_rotors(metric, sides, area);
        turns.push(rotors.map(|r| r.unit().log()));
        // Each side's dyad turns a gradient reading into flow along the side; the corner
        // cotangents weight the response of the intrinsic triangle.
        let cot = dots(metric, sides).map(|d| d / (2.0 * area));
        let dyad = |k: usize| sides[k] * (Plane::slot() & sides[k]) * (cot[k] / 2.0);
        conductivities.push(dyad(0) + dyad(1) + dyad(2));
        cotangents.push(cot.map(|c| c / 2.0));
        metrics.push(metric);
        areas.push(Scalar::new(area));
    }
    let areas: Field<Scalar> = areas.into_iter().collect();
    // Each vertex holds a third of the intrinsic area of the faces around it; its corner turns,
    // carried back from the faces and summed, fall short of the flat turn by the defect: a full
    // turn inside, half a turn on the rim.
    let vertex_areas = &mesh.means::<Scalar>(2).reverse() * &areas;
    let rim = exterior(mesh, 0);
    let mut defects: Vec<Bivector> = (0..mesh.len(0))
        .map(|v| {
            let flat = if rim.contains(&v) { 1.0 } else { 2.0 };
            Bivector::new(flat * core::f64::consts::PI)
        })
        .collect();
    let corner = incidence(mesh, 2, 0);
    for (f, t) in turns.iter().enumerate() {
        for (&v, &turn) in corner.sites[f].iter().zip(t) {
            defects[v] -= turn;
        }
    }
    let defects: Field<Bivector> = defects.into_iter().collect();
    let curvatures = defects.zip_with(&vertex_areas, |d, a| d * a.s().recip());
    Intrinsic {
        metrics,
        areas,
        defects,
        curvatures,
        conductivities,
        cotangents,
    }
}

/// The face gradient: vertex potentials to piecewise-constant gradients, each corner's the side
/// facing it over twice the patch.
pub fn gradient(mesh: &Topology<3>, vertices: &Field<Point>) -> SparseMap<Plane<(Scalar,)>> {
    let edges = &mesh.d::<Scalar>(0) * vertices;
    let facing = boundary(mesh, 2);
    let sides: [Field<Point>; 3] = core::array::from_fn(|k| facing.read(k, &edges));
    let twice = sides[0].zip_with(&sides[1], |a, b| ((a & b) * 0.5 * 2.0).inverse());
    incidence(mesh, 2, 0).assemble(|f, k| sides[k].site(f) * twice.site(f) * Scalar::slot())
}

/// The conductance of the vertices: the faces' conductivities pulled back through the face
/// gradient by its adjugate.
pub fn conductance(
    mesh: &Topology<3>,
    vertices: &Field<Point>,
    conductivities: &[Conductivity],
) -> SparseMap<Amount<(Scalar,)>> {
    let g = gradient(mesh, vertices);
    let material = SparseMap::from_diagonal(&conductivities.iter().copied().collect());
    g.adjugate().compose(&material.compose(&g))
}

/// Solve a conductance for the potentials that give the amounts: cells and amounts through the
/// undual, to scalars (negative definite, as the join pairs the current before the gradient).
fn solve_amounts(system: &SparseMap<Amount<(Scalar,)>>, amounts: &Field<Amount>) -> Field<Scalar> {
    system
        .map_cells(|c| c.undual())
        .solve_direct(&amounts.map(|a| a.undual()))
}

/// The map from the listed sites onto all `n`, `[n <- listed]`.
fn onto(listed: &[usize], n: usize) -> SparseMap<Scalar<(Scalar,)>> {
    reading::<Scalar>(listed, n).transpose_with(|c| c * Scalar::slot())
}

/// Steady conduction with the electrodes held at the prescribed potential: the potential, the
/// current density on each face (the integrated flux over the reference area), and the
/// streamfunction.
pub fn intrinsic_conduction(
    mesh: &Topology<3>,
    vertices: &Field<Point>,
    conductivities: &[Conductivity],
    electrodes: &[usize],
    prescribed: &Field<Scalar>,
) -> (Field<Scalar>, Field<Point>, Field<Scalar>) {
    let n = vertices.len();
    let g = gradient(mesh, vertices);
    let conductance = conductance(mesh, vertices, conductivities);
    let free_vertices: Vec<usize> = (0..n).filter(|v| !electrodes.contains(v)).collect();
    let free = onto(&free_vertices, n);
    let rhs = -&free.adjugate().of(&conductance.of(prescribed));
    let system = free.adjugate().compose(&conductance.compose(&free));
    let potential = prescribed + &free.of(&solve_amounts(&system, &rhs));
    let gradients = g.of(&potential);
    let flux: Field<Point> = (0..mesh.len(2))
        .map(|f| -conductivities[f].of(gradients.site(f)))
        .collect();
    let edges = &mesh.d::<Scalar>(0) * vertices;
    let facing = boundary(mesh, 2);
    let reference = facing
        .read(0, &edges)
        .zip_with(&facing.read(1, &edges), |a, b| {
            Scalar::new(((a & b) * 0.5).s().abs())
        });
    let current = flux.zip_with(&reference, |q, a| q * a.s().recip());

    // The integrated flux has no divergence on the reference mesh, so turned a quarter turn it
    // is the area-weighted reference gradient of the streamfunction.
    let quarter_turn = Bivector::new(-core::f64::consts::FRAC_PI_4).exp();
    let isotropic = Plane::slot().hodge();
    let laplacian = {
        let material = SparseMap::from_diagonal(&reference.map(|a| isotropic * a.s()));
        g.adjugate().compose(&material.compose(&g))
    };
    let unpinned: Vec<usize> = (1..n).collect();
    let unpinned = onto(&unpinned, n);
    let turned = flux.map(|q| quarter_turn >> q);
    let stream_rhs = unpinned.adjugate().of(&g.adjugate().of(&turned));
    let stream_system = unpinned.adjugate().compose(&laplacian.compose(&unpinned));
    let streamfunction = unpinned.of(&solve_amounts(&stream_system, &stream_rhs));
    (potential, current, streamfunction)
}

/// numga's scalene reference triangle.
pub fn single_triangle() -> [Point; 3] {
    [point(0.0, 0.0), point(1.2, 0.12), point(0.38, 0.96)]
}

/// A disk of nearly equilateral triangles with its first vertex at the centre.
pub fn disk(radius: f64, divisions: usize) -> (Topology<3>, Field<Point>) {
    let (mesh, coordinates) = hex_disk(radius, divisions);
    (mesh, coordinates.map(|v| v.hodge()))
}

/// The squared edge lengths of a case: each edge's squared length, plus the radial rate times
/// the square of its midpoint's inner product with it (radial motion times radius), plus the
/// hoop rate times the square of their join (the swept area: motion around times radius).
pub fn stretched(mesh: &Topology<3>, vertices: &Field<Point>, case: usize) -> Field<Scalar> {
    let edges = &mesh.d::<Scalar>(0) * vertices;
    let midpoints = &mesh.means::<Scalar>(1) * vertices;
    edges.zip_with(&midpoints, |e, m| {
        let along = (m | e).s();
        let around = (m & e).s();
        Scalar::new(
            e.norm_squared()
                + RADIAL_RATES[case] * along * along
                + HOOP_RATES[case] * around * around,
        )
    })
}

/// The electrodes, the rim vertices farther than the threshold from the `y` axis, and the
/// potential holding them at their `x`.
pub fn electrodes(mesh: &Topology<3>, vertices: &Field<Point>) -> (Vec<usize>, Field<Scalar>) {
    let x = |v: usize| coordinates(vertices.site(v))[0];
    let held: Vec<usize> = exterior(mesh, 0)
        .into_iter()
        .filter(|&v| x(v).abs() > ELECTRODE_THRESHOLD)
        .collect();
    let prescribed = Field::from_fn(vertices.len(), |v| {
        Scalar::new(if held.contains(&v) { x(v) } else { 0.0 })
    });
    (held, prescribed)
}
