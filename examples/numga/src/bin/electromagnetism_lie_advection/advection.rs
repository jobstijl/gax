//! The math: Cartan's formula on a moving triangle, the operators of a disk's mesh, the edge world
//! sheets of a prescribed flow, and the Runge–Kutta evolution of edge readings. No drawing.
//!
//! In numga's spacetime of the plane (`x+y+t-`, `shared/sta2d.rs`), a point is an intrinsic
//! bivector and its motion per unit time is `time + velocity`. The join of two points is a patch
//! (`Patch-`, an intrinsic vector), the join of a point and a patch a number. An edge sweeps a
//! world sheet, its motion joined with its side, and a 2-form reads the flux through it. The
//! magnetic field is the plane `xy` of space, and reads only a sheet's spatial part.

use gax::field::{Field, SparseMap};
use gax::mesh::Topology;

use crate::simplicial::{
    boundary, corners as local_corners, incidence, interior, oriented, reading,
};
use crate::sta2d::{self, Bivector, Point, Vector, time, xy};

/// A number (numga's `Scalar+`).
pub type Scalar = sta2d::Scalar;
/// A 1-form, a plane (numga's `Plane+`).
pub type Plane = Vector;
/// A patch, the join of two points (numga's `Patch-`).
pub type Patch = sta2d::intrinsic::Vector;
/// An amount, the excitation's pairing (numga's `Amount-`).
pub type Amount = sta2d::intrinsic::Pseudoscalar;
/// A map of numbers.
pub type ScalarMap = sta2d::Scalar<(Scalar,)>;
/// The curl's cell, from an edge reading to a magnetic field (numga's `Magnetic+ <- Scalar+`;
/// the full bivector here, its time parts zero).
pub type CurlMap = sta2d::Bivector<(Scalar,)>;

/// The sides of a triangle facing each corner, in its winding.
pub fn sides(corners: [Point; 3]) -> [Point; 3] {
    [0, 1, 2].map(|k| corners[(k + 2) % 3] - corners[(k + 1) % 3])
}

/// The corner hat gradients of a triangle: the plane through the side facing each corner and
/// time, over twice the prism the triangle sweeps in a unit of time. Each hat is constant in
/// time.
pub fn hat_gradients(sides: [Point; 3]) -> [Plane; 3] {
    let patch = (sides[0] & sides[1]) * 0.5;
    let twice = ((time() & patch) * 2.0).inverse();
    sides.map(|s| (time() & s) * twice)
}

/// Cartan's decomposition of a linear 1-form's rate on a moving triangle.
pub struct Cartan {
    /// The differences of the form's readings on the corners' motions along each side.
    pub endpoint_differences: [Scalar; 3],
    /// The 2-form's flux through each side's world sheet.
    pub swept_fluxes: [Scalar; 3],
    /// Each side's world sheet: its middle's motion joined with it (numga returns them).
    #[allow(dead_code)]
    pub world_sheets: [Patch; 3],
    /// The form's exterior derivative, constant on the face.
    #[cfg_attr(not(test), allow(dead_code))]
    pub face_bivector: Bivector,
}

/// The motion of each corner per unit time: along its world line, `time + velocity`.
pub fn motions(velocities: [Point; 3]) -> [Point; 3] {
    velocities.map(|v| time() + v)
}

/// The middle of each side's corners' values, the side facing each corner.
pub fn side_middles(values: [Point; 3]) -> [Point; 3] {
    [0, 1, 2].map(|k| (values[(k + 1) % 3] + values[(k + 2) % 3]) * 0.5)
}

/// The endpoint differences, the world-sheet fluxes, the world sheets and the 2-form of a
/// linear 1-form (its values at the corners) on a triangle whose corners move.
pub fn cartan_decomposition(
    corners: [Point; 3],
    forms: [Plane; 3],
    velocities: [Point; 3],
) -> Cartan {
    let sides = sides(corners);
    let gradients = hat_gradients(sides);
    // Wedging each corner hat's gradient with its corner's form sums to the constant 2-form.
    let face_bivector = (0..3)
        .map(|k| gradients[k] ^ forms[k])
        .fold(Bivector::zero(), |a, b| a + b);
    let motions = motions(velocities);
    let readings: [Scalar; 3] = [0, 1, 2].map(|k| forms[k] & motions[k]);
    let endpoint_differences = [0, 1, 2].map(|k| readings[(k + 2) % 3] - readings[(k + 1) % 3]);
    let world_sheets: [Patch; 3] = {
        let middles = side_middles(motions);
        [0, 1, 2].map(|k| middles[k] & sides[k])
    };
    let swept_fluxes = world_sheets.map(|s| face_bivector & s);
    Cartan {
        endpoint_differences,
        swept_fluxes,
        world_sheets,
        face_bivector,
    }
}

/// The topological derivatives, the Whitney maps and the curl-curl operators of a disk's mesh.
pub struct Operators {
    pub mesh: Topology<3>,
    pub vertices: Field<Point>,
    #[cfg_attr(not(test), allow(dead_code))]
    pub d0: SparseMap<ScalarMap>,
    #[cfg_attr(not(test), allow(dead_code))]
    pub d1: SparseMap<ScalarMap>,
    pub edges: Field<Point>,
    pub edge_midpoints: Field<Point>,
    /// Edge readings to the flux density on each face.
    pub flux_density: SparseMap<ScalarMap>,
    pub normals: Field<Patch>,
    /// Vertex streamfunctions to the constant gradient on each face, with no metric.
    pub gradient: SparseMap<sta2d::Vector<(Scalar,)>>,
    /// Edge readings to the constant 2-form of their Whitney forms on each face.
    #[cfg_attr(not(test), allow(dead_code))]
    pub curl: SparseMap<CurlMap>,
    /// Face values averaged onto each edge from the faces beside it.
    pub edge_velocity_map: SparseMap<sta2d::intrinsic::Bivector<(Point,)>>,
    pub edge_curl: SparseMap<CurlMap>,
    /// The curl pulled back through the vacuum: edge readings to edge currents.
    pub curl_stiffness: SparseMap<sta2d::intrinsic::Pseudoscalar<(Scalar,)>>,
    /// A unit resistance on each interior edge: the resistive rate of the readings.
    pub curl_laplacian: SparseMap<ScalarMap>,
}

/// The face value of a map averaged onto each edge from its faces, as a scalar map.
fn face_to_edge(mesh: &Topology<3>) -> SparseMap<Scalar> {
    let mut count = vec![0.0; mesh.len(1)];
    for f in 0..mesh.len(2) {
        for &(e, _) in mesh.faces(2, f) {
            count[e] += 1.0;
        }
    }
    let triples: Vec<_> = (0..mesh.len(2))
        .flat_map(|f| mesh.faces(2, f).iter().map(move |&(e, _)| (e, f)))
        .map(|(e, f)| (e, f, Scalar::new(1.0 / count[e])))
        .collect();
    SparseMap::from_triples(triples, (mesh.len(1), mesh.len(2)))
}

impl Operators {
    pub fn new(mesh: &Topology<3>, vertices: &Field<Point>) -> Self {
        let slot = Scalar::slot();
        let d0 = &mesh.d::<Scalar>(0) * slot;
        let d1 = &mesh.d::<Scalar>(1) * slot;
        let edges = &mesh.d::<Scalar>(0) * vertices;
        let edge_midpoints = &mesh.means::<Scalar>(1) * vertices;
        let sides = boundary(mesh, 2).read_all(&edges);
        let patches: Field<Patch> = sides[0].zip_with(&sides[1], |a, b| (a & b) * 0.5);
        // The prism each face sweeps in a unit of time; its size is the face's area.
        let prisms = patches.map(|p| time() & p);
        let areas = prisms.map(|p| Scalar::new(p.norm()));
        let flux_density =
            SparseMap::from_diagonal(&areas.map(|a| slot * (1.0 / a.s()))).compose(&d1);
        let normals = patches.zip_with(&areas, |p, a| p * (1.0 / a.s()));
        let twice = prisms.map(|p| (p * 2.0).inverse());
        let corner_gradients: [Field<Plane>; 3] = [0, 1, 2]
            .map(|k| Field::from_fn(mesh.len(2), |f| (time() & sides[k].site(f)) * twice.site(f)));
        let gradient = incidence(mesh, 2, 0).assemble(|f, k| corner_gradients[k].site(f) * slot);
        // The derivative of each edge's Whitney form is twice the wedge of its ends' gradients.
        let ends = local_corners(2, 1);
        let curl = oriented(mesh, 2, 1).assemble(|f, k| {
            let (t, h) = (ends[k][0], ends[k][1]);
            (corner_gradients[t].site(f) ^ corner_gradients[h].site(f)) * 2.0 * slot
        });
        // Averaging the faces beside each edge keeps uniform 2-forms exact and makes the ideal
        // advection skew in the magnetic energy.
        let to_edges = face_to_edge(mesh);
        let edge_velocity_map = &to_edges * Point::slot();
        let edge_curl = (&to_edges * Bivector::slot()).compose(&curl);
        // The vacuum takes each face's field to the excitation it causes, weighted by its area.
        let vacuum = SparseMap::from_diagonal(&areas.map(|a| Bivector::slot().hodge() * a.s()));
        let curl_stiffness = curl.adjugate().compose(&vacuum.compose(&curl));
        // A unit resistance on each interior edge turns its current into the rate of its reading;
        // the boundary's circulation stays fixed.
        let inside = interior(mesh, 1);
        let on_interior = (&reading::<Scalar>(&inside, mesh.len(1)) * slot).transpose();
        let unit_resistance =
            SparseMap::from_diagonal(&Field::from_fn(inside.len(), |_| Amount::slot().undual()));
        let curl_laplacian = on_interior
            .compose(&unit_resistance.compose(&on_interior.adjugate().compose(&curl_stiffness)));
        Operators {
            mesh: mesh.clone(),
            vertices: vertices.clone(),
            d0,
            d1,
            edges,
            edge_midpoints,
            flux_density,
            normals,
            gradient,
            curl,
            edge_velocity_map,
            edge_curl,
            curl_stiffness,
            curl_laplacian,
        }
    }

    /// The magnetic energy of edge readings: half their pairing with the currents they drive.
    pub fn energy(&self, cochain: &Field<Scalar>) -> f64 {
        let currents = self.curl_stiffness.of(cochain);
        cochain
            .iter()
            .zip(currents.iter())
            .map(|(a, c)| (a & c).s())
            .sum::<f64>()
            / 2.0
    }

    /// The total flux through the faces.
    #[cfg(test)]
    pub fn total_flux(&self, cochain: &Field<Scalar>) -> f64 {
        self.d1.of(cochain).iter().map(|x| x.s()).sum()
    }
}

/// The streamfunction of an elliptical shearing vortex that vanishes smoothly on the unit circle:
/// `swirl (1 − r²)² (1 + ellipticity (x² − y²))`, with `r²` the point's square.
pub fn vortex_streamfunction(
    vertices: &Field<Point>,
    swirl: f64,
    ellipticity: f64,
) -> Field<Scalar> {
    vertices.map(|p| {
        let [x, y] = sta2d::coordinates(p);
        let envelope = Scalar::new(1.0) - (p * p).grade::<0>();
        envelope * envelope * (swirl * (1.0 + ellipticity * (x * x - y * y)))
    })
}

/// The initial edge readings: Gaussian lobes of circulating potential about the centres, the
/// unit bivector's reading of the displacement from each, read along every edge.
pub fn initial_cochain(
    operators: &Operators,
    centres: &[Point],
    strengths: &[f64],
    radius: f64,
) -> Field<Scalar> {
    let potential = operators.edge_midpoints.map(|m| {
        centres
            .iter()
            .zip(strengths)
            .map(|(&c, &strength)| {
                let d = m - c;
                let envelope = (-(d * d).grade::<0>().s() / (2.0 * radius * radius)).exp();
                (xy() & d) * (envelope * strength)
            })
            .fold(Plane::zero(), |a, b| a + b)
    });
    potential.zip_with(&operators.edges, |p, e| p & e)
}

/// The face velocity of a streamfunction, and the world sheets each edge sweeps per unit time
/// held fixed and carried by the flow.
pub fn edge_world_sheets(
    operators: &Operators,
    stream: &Field<Scalar>,
) -> (Field<Point>, Field<Patch>, Field<Patch>) {
    let face_velocity = operators
        .gradient
        .of(stream)
        .zip_with(&operators.normals, |g, n| g ^ n);
    let edge_velocity = operators.edge_velocity_map.of(&face_velocity);
    let fixed = operators.edges.map(|e| time() & e);
    let comoving = edge_velocity.zip_with(&operators.edges, |v, e| (time() + v) & e);
    (face_velocity, fixed, comoving)
}

/// The ideal advection map: a fixed edge's reading changes by the magnetic flux through the
/// patch between its own world sheet and the comoving one, the face fields averaged onto it.
pub fn advection(operators: &Operators, swept: &Field<Patch>) -> SparseMap<ScalarMap> {
    let reading = SparseMap::from_diagonal(&swept.map(|p| Bivector::slot() & p));
    reading.compose(&operators.edge_curl)
}

/// The evolution of the readings: ideal advection less resistive diffusion.
pub fn rate(
    operators: &Operators,
    advection: &SparseMap<ScalarMap>,
    resistivity: f64,
) -> SparseMap<ScalarMap> {
    advection + &operators.curl_laplacian.map_cells(|c| c * -resistivity)
}

/// The edge readings after each of `steps` classical fourth-order Runge–Kutta steps, the start
/// first.
pub fn advect(
    start: &Field<Scalar>,
    rate: &SparseMap<ScalarMap>,
    dt: f64,
    steps: usize,
) -> Vec<Field<Scalar>> {
    let mut states = vec![start.clone()];
    let mut a = start.clone();
    for _ in 0..steps {
        let first = rate.of(&a);
        let second = rate.of(&(&a + &first.map(|x| x * (dt / 2.0))));
        let third = rate.of(&(&a + &second.map(|x| x * (dt / 2.0))));
        let fourth = rate.of(&(&a + &third.map(|x| x * dt)));
        let sum = &(&first + &second.map(|x| x * 2.0)) + &(&third.map(|x| x * 2.0) + &fourth);
        a = &a + &sum.map(|x| x * (dt / 6.0));
        states.push(a.clone());
    }
    states
}
