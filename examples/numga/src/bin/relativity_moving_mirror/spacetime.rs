//! The math: corner hats and edge forms on a triangle of the spacetime plane, the gradient and
//! the wave operator of a triangulated worldsheet, the causal wake of a source, and a pulse in a
//! cavity with a moving mirror. No drawing.
//!
//! numga's spacetime plane `x+t-` has no standard gax algebra; it is declared here. Its events
//! are antivectors, intrinsic (numga's `Point-`): the metric duals of the vectors, so that the
//! unit displacements along space and into the future pair to one with `x` and `t`. A triangle's
//! patch, the join of two sides, is a number (`Patch-`); a side over twice the patch is the
//! gradient of the hat of the corner it faces, a vector (`Plane+`), with no metric. The metric
//! enters once, through the dual that turns a gradient into the current it drives, and pulled back
//! through the gradient's adjugate it gives the wave operator.

use gax::field::{Field, SparseMap};
use gax::mesh::Topology;

gax::algebra! {
    algebra lorentz "numga's spacetime plane `x+t-`: space `x` squares to one, time `t` to minus one.";
    basis ex = 1, et = -1;
    kind Scalar = [1];
    versor Vector = [ex, et] "A gradient: a plane of constant phase, `a x + b t`; as a versor, the reflection in its normal.";
    kind Pseudoscalar = [ext];
}

/// A number (numga's `Scalar+`).
pub type Scalar = lorentz::Scalar;
/// A gradient, a 1-form (numga's `Plane+`).
pub type Plane = lorentz::Vector;
/// An event or a displacement (numga's `Point-`).
pub type Point = lorentz::intrinsic::Vector;
/// A triangle's patch (numga's `Patch-`).
pub type Patch = lorentz::intrinsic::Scalar;
/// An amount on a vertex (numga's `Amount-`).
pub type Amount = lorentz::intrinsic::Pseudoscalar;
/// The cell of the gradient (numga's `Plane+ <- Scalar+`).
pub type GradientMap = lorentz::Vector<(Scalar,)>;
/// The cell of the wave operator (numga's `Amount- <- Scalar+`).
pub type WaveMap = lorentz::intrinsic::Pseudoscalar<(Scalar,)>;
/// A scalar map, the cell of a selection.
pub type ScalarMap = lorentz::Scalar<(Scalar,)>;

/// The unit spatial displacement, `x`'s dual.
pub fn space() -> Point {
    Plane::new(1.0, 0.0).hodge()
}

/// The unit displacement into the future: minus `t`'s dual.
pub fn time() -> Point {
    -Plane::new(0.0, 1.0).hodge()
}

/// The event at `(x, t)`.
pub fn event(x: f64, t: f64) -> Point {
    space() * x + time() * t
}

/// The plane `x`, which reads an event's place.
pub fn x_plane() -> Plane {
    Plane::new(1.0, 0.0)
}

/// The plane `t`, which reads an event's time.
pub fn t_plane() -> Plane {
    Plane::new(0.0, 1.0)
}

/// An event's place and time.
pub fn coordinates(p: Point) -> [f64; 2] {
    [(x_plane() & p).s(), (t_plane() & p).s()]
}

/// The Lorentzian dual of a gradient: the direction it drives, numga's `Plane.dual()`.
pub fn dual() -> lorentz::intrinsic::Vector<(Plane,)> {
    Plane::slot().hodge()
}

/// The sides of a triangle facing each corner, counter-clockwise from the corner after it to the
/// one after that.
pub fn sides(corners: [Point; 3]) -> [Point; 3] {
    [0, 1, 2].map(|k| corners[(k + 2) % 3] - corners[(k + 1) % 3])
}

/// The hats at each point, the constant corner gradients, and the triangle's signed patch: each
/// side over twice the patch is the gradient of the hat of the corner it faces, which falls by
/// one from its corner to that side.
pub fn barycentric_hats(
    corners: [Point; 3],
    points: &[Point],
) -> (Vec<[Scalar; 3]>, [Plane; 3], Patch) {
    let sides = sides(corners);
    let patch = (sides[0] & sides[1]) * 0.5;
    let twice = (patch * 2.0).inverse();
    let gradients = sides.map(|s| s * twice);
    let hats = points
        .iter()
        .map(|&p| [0, 1, 2].map(|k| Scalar::new(1.0) + (gradients[k] & (p - corners[k]))))
        .collect();
    (hats, gradients, patch)
}

/// The three edge Whitney 1-forms at each point, each circulating around the corner its edge
/// faces: the displacement from that corner over twice the patch.
pub fn edge_whitney_forms(corners: [Point; 3], points: &[Point]) -> Vec<[Plane; 3]> {
    let twice = (patch(corners) * 2.0).inverse();
    points
        .iter()
        .map(|&p| corners.map(|c| (p - c) * twice))
        .collect()
}

/// A triangle's signed patch.
pub fn patch(corners: [Point; 3]) -> Patch {
    let s = sides(corners);
    (s[0] & s[1]) * 0.5
}

/// The operators of a spacetime mesh: the gradient, the wave operator, and the face areas.
pub struct Operators {
    pub mesh: Topology<3>,
    pub vertices: Field<Point>,
    /// Each face's area in the picture's measure, the size of its patch.
    pub areas: Field<Scalar>,
    /// Vertex potentials to the constant 1-form on each face, assembled by corner, no metric.
    pub gradient: SparseMap<GradientMap>,
    /// The Lorentzian dual pulled back through the gradient: a wave operator whose
    /// characteristics are the null directions.
    pub wave: SparseMap<WaveMap>,
}

impl Operators {
    pub fn new(mesh: &Topology<3>, vertices: &Field<Point>) -> Self {
        let edges = &mesh.d::<Scalar>(0) * vertices;
        let sides = mesh.facets(2).read_all(&edges);
        let patches = sides[0].zip_with(&sides[1], |a, b| (a & b) * 0.5);
        let areas = patches.map(|p| Scalar::new(p.norm()));
        let twice = patches.map(|p| (p * 2.0).inverse());
        let gradient = mesh
            .incidence(2, 0)
            .assemble(|f, k| (sides[k].site(f) * twice.site(f)) * Scalar::slot());
        let response = SparseMap::from_diagonal(&areas.map(|a| dual() * a.s()));
        let wave = gradient.adjugate().compose(&response.compose(&gradient));
        Operators {
            mesh: mesh.clone(),
            vertices: vertices.clone(),
            areas,
            gradient,
            wave,
        }
    }

    /// The vertex field of a face field, each vertex taking the area-weighted mean of the faces
    /// around it.
    pub fn to_vertices(&self, field: &Field<Scalar>) -> Field<Scalar> {
        let n = self.vertices.len();
        let (mut sum, mut weight) = (vec![0.0; n], vec![0.0; n]);
        for f in 0..self.mesh.len(2) {
            let a = self.areas.site(f).s();
            for &v in self.mesh.simplex(2, f) {
                sum[v] += a * field.site(f).s();
                weight[v] += a;
            }
        }
        sum.iter()
            .zip(&weight)
            .map(|(s, w)| Scalar::new(s / w))
            .collect()
    }
}

/// The vertices of a strip's grid, `width` to a row, in the given rows and columns, row by row.
pub fn grid_sites(
    width: usize,
    rows: core::ops::Range<usize>,
    columns: core::ops::Range<usize>,
) -> Vec<usize> {
    rows.flat_map(|r| columns.clone().map(move |c| r * width + c))
        .collect()
}

/// The map from values at the given vertices to a field on all `n`, zero elsewhere.
pub fn placing(sites: &[usize], n: usize) -> SparseMap<ScalarMap> {
    (&SparseMap::<Scalar>::selection(sites, n) * Scalar::slot()).transpose()
}

/// Solve the interior rows of the wave equation for the potential on the future rows: the
/// system's cells and amounts read through the undual, the numbers the direct solver takes.
fn march(
    operators: &Operators,
    columns: usize,
    rows: usize,
    amounts: &Field<Amount>,
) -> Field<Scalar> {
    let n = operators.vertices.len();
    let interior = placing(&grid_sites(columns, 1..rows - 1, 1..columns - 1), n);
    let future = placing(&grid_sites(columns, 2..rows, 1..columns - 1), n);
    let system = interior
        .adjugate()
        .compose(&operators.wave.compose(&future));
    future.of(&system.solve_direct(&interior.adjugate().of(amounts)))
}

/// The causal wake of a localized source at `source`: a time derivative of a Gaussian radiates
/// two sharp pulses along the forward null rays. The first two rows and the walls stay at rest,
/// and each interior equation fixes the next row: initial and boundary values choose the causal
/// solution, which the Lorentzian metric alone does not.
pub fn causal_wake(
    operators: &Operators,
    columns: usize,
    rows: usize,
    source: Point,
    width: f64,
) -> Field<Scalar> {
    let mean_area =
        operators.areas.iter().map(|a| a.s()).sum::<f64>() / operators.areas.len() as f64;
    let amounts = operators.vertices.map(|v| {
        let d = v - source;
        let delay = (t_plane() & d).s();
        let along = (x_plane() & d).s();
        let distance_squared = along * along + delay * delay;
        let profile = delay / width * (-distance_squared / (2.0 * width * width)).exp();
        Scalar::new(profile * mean_area).hodge()
    });
    march(operators, columns, rows, &amounts)
}

/// The pulse in the cavity: the potential, its gradient on the faces, the wave rate on the
/// vertices, and the energy in each slab between two rows of the strip.
pub struct Wave {
    #[cfg_attr(not(test), allow(dead_code))] // the tests read the potential, the drawing its rate
    pub potential: Field<Scalar>,
    /// The spacetime gradient on each face, which numga returns beside the rest.
    #[allow(dead_code)]
    pub field: Field<Plane>,
    pub rate: Field<Scalar>,
    pub slab_energies: Vec<f64>,
}

/// A right-moving pulse, `exp(−(φ/w)²) cos(2π φ/λ)` in the phase `φ = (x − t) − centre` read
/// by the null plane `x − t`, given on the first two rows and marched up the strip between the
/// walls. The observer's energy density reflects the dual across the slice `t`, which flips the
/// time part's sign and makes the Lagrangian density the positive Hamiltonian one.
pub fn moving_mirror_wave(
    operators: &Operators,
    columns: usize,
    rows: usize,
    dt: f64,
    centre: f64,
    width: f64,
    wavelength: f64,
) -> Wave {
    let n = operators.vertices.len();
    let null = x_plane() - t_plane();
    let pulse = operators.vertices.map(|v| {
        let phase = (null & v).s() - centre;
        let envelope = (-(phase / width) * (phase / width)).exp();
        let carrier =
            gax_numga_examples::signal::phasor(core::f64::consts::TAU * phase / wavelength).e20();
        Scalar::new(envelope * carrier)
    });
    let seed = grid_sites(columns, 0..2, 1..columns - 1);
    let initial = placing(&seed, n).of(&(&SparseMap::<Scalar>::selection(&seed, n) * &pulse));
    let rhs = -&operators.wave.of(&initial);
    let potential = &initial + &march(operators, columns, rows, &rhs);

    let field = operators.gradient.of(&potential);
    let face_rate = field.map(|g| g & time());
    let rate = operators.to_vertices(&face_rate);
    let reflected = t_plane() >> dual();
    let density = field.map(|g| -(g & reflected.of(g)) * 0.5);
    let mut slab_energies = vec![0.0; rows - 1];
    for f in 0..operators.mesh.len(2) {
        let lowest = operators
            .mesh
            .simplex(2, f)
            .iter()
            .copied()
            .min()
            .expect("corners");
        slab_energies[lowest / columns] += (operators.areas.site(f) * density.site(f)).s() / dt;
    }
    Wave {
        potential,
        field,
        rate,
        slab_energies,
    }
}

/// A strip of spacetime between a wall at rest at the origin and a wall at `walls[k]` at row
/// `k`, `columns` evenly across it, `rows` rows over `duration`: its mesh, events and time step.
/// Each cell of the grid splits into two triangles, all the `[00, 10, 11]` ones first.
pub fn spacetime_strip(
    columns: usize,
    rows: usize,
    walls: &[f64],
    duration: f64,
) -> (Topology<3>, Field<Point>, f64) {
    let dt = duration / (rows - 1) as f64;
    let vertices = (0..rows)
        .flat_map(|r| {
            (0..columns)
                .map(move |c| event(c as f64 / (columns - 1) as f64 * walls[r], r as f64 * dt))
        })
        .collect();
    let at = |r: usize, c: usize| r * columns + c;
    let cells = || (0..rows - 1).flat_map(|r| (0..columns - 1).map(move |c| (r, c)));
    let first = cells().map(|(r, c)| [at(r, c), at(r, c + 1), at(r + 1, c + 1)]);
    let second = cells().map(|(r, c)| [at(r, c), at(r + 1, c + 1), at(r + 1, c)]);
    let faces: Vec<[usize; 3]> = first.chain(second).collect();
    let mesh = Topology::new(columns * rows, &faces).expect("the strip");
    (mesh, vertices, dt)
}

/// The right mirror's place along a smooth timelike worldline: from `length`, in by `stroke`,
/// the turn at `inflection` taking about `transition`.
pub fn mirror_trajectory(
    t: f64,
    length: f64,
    stroke: f64,
    inflection: f64,
    transition: f64,
) -> f64 {
    length - stroke * (1.0 + ((t - inflection) / transition).tanh()) / 2.0
}
