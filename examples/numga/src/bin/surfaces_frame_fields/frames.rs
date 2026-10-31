//! The math: rotors blended across a triangle and the grades of their Darboux derivative, a
//! disk of rotors that wraps the sphere of directors, and frame fields over a curved surface,
//! harmonic in the even subalgebra or relaxed on unit rotors; and numga's scenes. No drawing.
//!
//! Points are antivectors, intrinsic bivectors whose coefficients are the coordinates (numga's
//! `Point-`); a corner's hat gradient is a vector (`Plane+`); a frame is a rotor.

use crate::simplicial::{boundary, corners as local_corners, incidence, interior, reading};
use gax::field::{Field, SparseMap};
use gax::mesh::Topology;
use gax::vga3d;

pub type Scalar<S = ()> = vga3d::Scalar<S>;
pub type Plane<S = ()> = vga3d::Vector<S>;
pub type Bivector<S = ()> = vga3d::Bivector<S>;
pub type Rotor<S = ()> = vga3d::Rotor<S>;
pub type Point<S = ()> = vga3d::intrinsic::Bivector<S>;
/// A step to the mixed-grade change of a rotor field (numga's `EvenMap`).
pub type EvenMap = Rotor<(Point,)>;
/// A step to a turning rate (numga's `RateMap`).
pub type RateMap = Bivector<(Point,)>;
/// A step to a change of the chord's size (numga's `StretchMap`).
pub type StretchMap = Scalar<(Point,)>;

/// The point at `(x, y, z)`.
pub fn point(x: f64, y: f64, z: f64) -> Point {
    Plane::new(x, y, z).hodge()
}

/// Straight up.
pub fn vertical() -> Point {
    point(0.0, 0.0, 1.0)
}

/// A point's coordinates, read by joining it with the slices across `x`, `y` and `z`.
pub fn coordinates(p: Point) -> [f64; 3] {
    [
        Plane::new(1.0, 0.0, 0.0),
        Plane::new(0.0, 1.0, 0.0),
        Plane::new(0.0, 0.0, 1.0),
    ]
    .map(|s| (s & p).s())
}

/// numga's `a.dot(b)` on points: `⟨a ~b⟩`, their Euclidean inner product.
pub fn dot(a: Point, b: Point) -> f64 {
    a.scalar_product(b.reverse()).s()
}

/// The sides of a triangle facing each corner, counter-clockwise.
pub fn sides(corners: [Point; 3]) -> [Point; 3] {
    [0, 1, 2].map(|k| corners[(k + 2) % 3] - corners[(k + 1) % 3])
}

/// The corner gradients of a triangle: each side contracted with the patch's inverse, halved.
pub fn gradients(sides: [Point; 3]) -> [Plane; 3] {
    let patch = (sides[0] & sides[1]) * 0.5;
    sides.map(|s| (s | patch.inverse()) * 0.5)
}

/// The blend of corner rotors at a point of a triangle and its Darboux derivative's grades.
#[cfg_attr(not(test), allow(dead_code))] // numga's core, checked by the tests
pub struct Darboux {
    /// The chord: the rotors blended linearly, inside the unit rotor sphere.
    pub chord: Rotor,
    /// The chord made unit again: a rigid turn.
    pub unit: Rotor,
    /// `2 ~chord · d(chord)`: a step to the mixed-grade change.
    pub darboux: EvenMap,
    /// Its bivector part from the edges' Whitney forms, and its scalar part from their
    /// symmetric partners.
    pub whitney_rates: RateMap,
    pub chord_stretch: StretchMap,
    /// The rates over the chord's squared size: the unit rotors' Maurer–Cartan form.
    pub angular_velocity: RateMap,
    /// The three edge Whitney 1-forms there.
    pub whitney_forms: [Plane; 3],
}

/// Linearly blended corner rotors on a triangle, at a point: their Darboux derivative splits
/// into the edges' Whitney 1-forms in grade two and symmetric Regge differentials in grade zero.
pub fn triangle_darboux(corners: [Point; 3], at: Point, rotors: [Rotor; 3]) -> Darboux {
    let gradients = gradients(sides(corners));
    let hats = [0, 1, 2].map(|k| 1.0 + (gradients[k] & (at - corners[k])).s());
    // Linear blending cuts inside the unit rotor sphere; normalization restores a rigid turn.
    let chord = rotors[0] * hats[0] + rotors[1] * hats[1] + rotors[2] * hats[2];
    let differential: EvenMap = (0..3)
        .map(|k| rotors[k] * (gradients[k] & Point::slot()))
        .reduce(|a, b| a + b)
        .expect("three corners");
    let darboux = chord.reverse() * differential * 2.0;
    // Reversing an edge flips its transition's turning plane and keeps its scalar part: the
    // antisymmetric Whitney forms and the symmetric products of the ends.
    let ends = local_corners(2, 1);
    let parts: [(RateMap, StretchMap, Plane); 3] = core::array::from_fn(|k| {
        let (t, h) = (ends[k][0], ends[k][1]);
        let transition = rotors[t].reverse() * rotors[h];
        let edge_bivector = transition.grade::<2>() * 2.0;
        let sagitta = 2.0 * (transition.s() - 1.0);
        let form = gradients[h] * hats[t] - gradients[t] * hats[h];
        let symmetric = gradients[h] * hats[t] + gradients[t] * hats[h];
        (
            edge_bivector * (form & Point::slot()),
            (symmetric & Point::slot()) * sagitta,
            form,
        )
    });
    let whitney_rates = parts[0].0 + parts[1].0 + parts[2].0;
    let chord_stretch = parts[0].1 + parts[1].1 + parts[2].1;
    let whitney_forms = parts.map(|p| p.2);
    let size = chord.norm_squared();
    Darboux {
        chord,
        unit: chord.normalized(),
        darboux,
        whitney_rates,
        chord_stretch,
        angular_velocity: whitney_rates * size.recip(),
        whitney_forms,
    }
}

/// A radial rotor texture on a disk, wrapping the sphere of directors, and what its frame
/// derivative reads on each face.
#[cfg_attr(not(test), allow(dead_code))] // numga's core, checked by the tests
pub struct Skyrmion {
    pub rotors: Field<Rotor>,
    pub directors: Field<Point>,
    pub scalar_parts: Vec<f64>,
    pub bivector_norms: Vec<f64>,
    pub energy_density: Vec<f64>,
    pub solid_angles: Vec<f64>,
}

/// The texture of a twist: the radial position and the vertical span the plane of turning, and
/// at a half turn across the unit disk the director runs from the north pole to the south.
pub fn skyrmion_winding(mesh: &Topology<3>, vertices: &Field<Point>, twist: f64) -> Skyrmion {
    let up = vertical();
    let rotors = vertices.map(|v| (v.commutator(up) * (-twist / 2.0)).exp().into_inner());
    let directors = rotors.map(|r| r >> up);
    let edges = &mesh.d::<Scalar>(0) * vertices;
    let facing = boundary(mesh, 2);
    let sides = facing.read_all(&edges);
    let at = incidence(mesh, 2, 0);
    let ends = local_corners(2, 1);
    let mut energy_density = Vec::new();
    let mut solid_angles = Vec::new();
    for f in 0..mesh.len(2) {
        let s: [Point; 3] = core::array::from_fn(|k| sides[k].site(f));
        let g = gradients(s);
        let r: [Rotor; 3] = core::array::from_fn(|k| rotors.site(at.sites[f][k]));
        let chord = (r[0] + r[1] + r[2]) * (1.0 / 3.0);
        let differential: EvenMap = (0..3)
            .map(|k| r[k] * (g[k] & Point::slot()))
            .reduce(|a, b| a + b)
            .expect("three corners");
        let rates: RateMap =
            (chord.reverse() * differential * 2.0).grade::<2>() * chord.norm_squared().recip();
        // The cotangent weights turn edge readings into the frame energy, whatever the
        // directions.
        let area = ((s[0] & s[1]) * 0.5).norm();
        let density: f64 = (0..3)
            .map(|k| {
                let cot = -dot(s[ends[k][0]], s[ends[k][1]]) / (2.0 * area);
                cot / (2.0 * area) * rates.of(s[k]).norm_squared()
            })
            .sum();
        energy_density.push(density);
        // The two turning rates' commutator read against the director's equatorial plane: the
        // swept area. The full frame connection is flat; its derivative cancels this term.
        let swept = rates.of(s[1]).commutator(rates.of(s[0])) * 0.5;
        solid_angles.push((swept & Bivector::new(0.0, 0.0, 1.0).hodge()).s());
    }
    Skyrmion {
        scalar_parts: rotors.iter().map(|r| r.s()).collect(),
        bivector_norms: rotors.iter().map(|r| r.grade::<2>().norm()).collect(),
        rotors,
        directors,
        energy_density,
        solid_angles,
    }
}

/// A frame field's operators on a surface: the difference along each edge on rotors, the
/// cotangent Laplacian on them, the edges' cotangent weights, the faces' areas.
#[cfg_attr(not(test), allow(dead_code))] // numga's core, checked by the tests
pub struct RotorLaplacian {
    pub d0: SparseMap<Rotor<(Rotor,)>>,
    pub laplacian: SparseMap<Rotor<(Rotor,)>>,
    pub weights: Field<Scalar>,
    pub areas: Field<Scalar>,
}

pub fn rotor_laplacian(mesh: &Topology<3>, vertices: &Field<Point>) -> RotorLaplacian {
    let edges = &mesh.d::<Scalar>(0) * vertices;
    let sides = boundary(mesh, 2).read_all(&edges);
    let areas = sides[0].zip_with(&sides[1], |a, b| Scalar::new(((a & b) * 0.5).norm()));
    let ends = local_corners(2, 1);
    let facing = incidence(mesh, 2, 1);
    let mut weights = vec![0.0; mesh.len(1)];
    for f in 0..mesh.len(2) {
        for (k, end) in ends.iter().enumerate() {
            let cot =
                -dot(sides[end[0]].site(f), sides[end[1]].site(f)) / (2.0 * areas.site(f).s());
            weights[facing.sites[f][k]] += cot / 2.0;
        }
    }
    let weights: Field<Scalar> = weights.into_iter().map(Scalar::new).collect();
    let d0 = &mesh.d::<Scalar>(0) * Rotor::slot();
    let weighted = SparseMap::from_diagonal(&weights.map(|w| Rotor::slot() * w.s()));
    let laplacian = d0.adjoint().compose(&weighted.compose(&d0));
    RotorLaplacian {
        d0,
        laplacian,
        weights,
        areas,
    }
}

/// The map from the listed sites onto all `n`, cells on rotors.
fn onto(listed: &[usize], n: usize) -> SparseMap<Rotor<(Rotor,)>> {
    reading::<Scalar>(listed, n).transpose_with(|c| Rotor::slot() * c.s())
}

/// The harmonic chord field in the even subalgebra with the boundary's rotors pinned (linear),
/// and its unit rotors.
pub fn harmonic_rotor_frames(
    mesh: &Topology<3>,
    vertices: &Field<Point>,
    rim: &[usize],
    prescribed: &Field<Rotor>,
) -> (Field<Rotor>, Field<Rotor>) {
    let n = vertices.len();
    let operators = rotor_laplacian(mesh, vertices);
    let free = onto(&interior(mesh, 0), n);
    let pinned = onto(rim, n);
    let held = pinned.of(&pinned.adjoint().of(prescribed));
    let rhs = -&free.adjoint().of(&operators.laplacian.of(&held));
    let system = free.adjoint().compose(&operators.laplacian.compose(&free));
    let chords = &held + &free.of(&system.solve(&rhs));
    let rotors = chords.map(|c| c.normalized());
    (chords, rotors)
}

/// Projected cotangent steps that smooth a frame field with its boundary held: each step moves
/// the free vertices down the Laplacian and makes every rotor unit again; the states with
/// their energies, the first included.
pub fn relax_rotor_frames(
    mesh: &Topology<3>,
    vertices: &Field<Point>,
    start: &Field<Rotor>,
    step: f64,
    steps: usize,
) -> Vec<(Field<Rotor>, f64)> {
    let operators = rotor_laplacian(mesh, vertices);
    let free = onto(&interior(mesh, 0), vertices.len());
    let mut current = start.clone();
    let mut states = Vec::with_capacity(steps + 1);
    for k in 0..=steps {
        let differences = operators.d0.of(&current);
        let energy = 0.5
            * operators
                .weights
                .zip_with(&differences, |w, d| w * d.norm_squared())
                .sum()
                .s();
        states.push((current.clone(), energy));
        if k < steps {
            let pull = free.of(&free.adjoint().of(&operators.laplacian.of(&current)));
            current = (&current - &pull.scaled(step)).map(|r| r.normalized());
        }
    }
    states
}

pub const TRIANGLE_SUBDIVISIONS: usize = 6;
pub const FINE_LEVELS: usize = 4;
pub const SKYRMION_DIVISIONS: usize = 12;
pub const SADDLE_DIVISIONS: usize = 6;
pub const DISK_RADIUS: f64 = 1.0;
pub const SKYRMION_TWIST: f64 = core::f64::consts::PI;
pub const SADDLE_HEIGHT: f64 = 0.28;
pub const TILT_RATES: [f64; 3] = [0.35, 0.85, -0.75];
pub const TWIST_RATES: [f64; 3] = [0.0, 1.15, -1.15];
pub const RELAXATION_STEP: f64 = 0.08;
pub const RELAXATION_STEPS: usize = 45;

/// The interior points of the barycentric lattice of `n` steps across a triangle.
pub fn triangle_samples(corners: [Point; 3], n: usize) -> Vec<Point> {
    let mut out = Vec::new();
    for first in 1..n {
        for second in 1..n - first {
            let w = [n - first - second, first, second].map(|k| k as f64 / n as f64);
            out.push(corners[0] * w[0] + corners[1] * w[1] + corners[2] * w[2]);
        }
    }
    out
}

/// A bivector from numga's planes: `yz`, `zx`, `xy`.
fn plane(yz: f64, zx: f64, xy: f64) -> Bivector {
    Bivector::new(yz, zx, xy)
}

/// numga's tilted triangle, its three corner rotors, and its samples.
pub fn single_triangle(n: usize) -> ([Point; 3], [Rotor; 3], Vec<Point>) {
    let corners = [
        point(0.0, 0.0, 0.05),
        point(1.15, 0.14, -0.12),
        point(0.36, 0.98, 0.24),
    ];
    let rotors = [
        plane(0.25 / 2.0, 0.0, -0.35 / 2.0),
        plane(-1.25 / 2.0, 0.55 / 2.0, 0.0),
        plane(0.0, -1.35 / 2.0, -0.65 / 2.0),
    ]
    .map(|b| b.exp().into_inner());
    (corners, rotors, triangle_samples(corners, n))
}

/// A disk of nearly equilateral triangles (the port's relaxed `triangular_disk`), its vertices as
/// points of the level plane.
pub fn triangular_disk(radius: f64, divisions: usize) -> (Topology<3>, Field<Point>) {
    let disk = gax_numga_examples::mesh::Mesh::triangular_disk(radius, divisions);
    let topology = Topology::new(disk.vertices.len(), &disk.faces).expect("the disk");
    (topology, disk.vertices.map(|v| v.hodge()))
}

/// A curved saddle over a disk, with its rim, three prescriptions of boundary frames (a tilt
/// out of the level plane and a twist about each radius), and a twisted start for the
/// relaxation.
pub struct Saddle {
    pub mesh: Topology<3>,
    pub vertices: Field<Point>,
    pub rim: Vec<usize>,
    pub prescribed: [Field<Rotor>; 3],
    pub start: Field<Rotor>,
}

pub fn saddle_surface(radius: f64, divisions: usize, height: f64) -> Saddle {
    let (mesh, flat) = triangular_disk(radius, divisions);
    let up = vertical();
    let vertices = flat.map(|p| {
        let [x, y, _] = coordinates(p);
        p + up * (height * (x * x - y * y))
    });
    let rim = crate::simplicial::exterior(&mesh, 0);
    let (yz, zx, xy) = (
        plane(1.0, 0.0, 0.0),
        plane(0.0, 1.0, 0.0),
        plane(0.0, 0.0, 1.0),
    );
    let prescribed = core::array::from_fn(|case| {
        flat.map(|p| {
            let [x, y, _] = coordinates(p);
            let tilt = p.commutator(up) * (-TILT_RATES[case] / 2.0);
            let twist = (yz * x + zx * y) * (TWIST_RATES[case] / 2.0);
            (tilt + twist).exp().into_inner()
        })
    });
    let start = flat.zip_with(&prescribed[1], |p, r| {
        let [x, y, _] = coordinates(p);
        let envelope = 1.0 - (x * x + y * y);
        let twist =
            p.commutator(up) * 1.65 + xy * (1.55 * x - 1.25 * y) + (yz - zx) * (2.25 * x * y);
        (twist * envelope).exp().into_inner() * r
    });
    Saddle {
        mesh,
        vertices,
        rim,
        prescribed,
        start,
    }
}
