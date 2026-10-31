//! E8's root system in the geometric algebra of eight dimensions, R(8,0): what `math_e8`,
//! `math_e_series` and `quantum_affine_toda` share (numga's `examples/math/e8/core.py`, the
//! root system of its scenarios, and the look of its render).
//!
//! The eight simple roots are mirrors, vectors of squared length two, and reflecting in one is
//! `>>` with it. Their product is a rotor whose sandwich on vectors is the Coxeter element, all
//! eight reflections as one map. One root from each of its eight orbits (each reflected in the
//! mirrors before it in the product) seeds the eight orbits of thirty that hold all 240 roots.
//! The motion averaged with its adjoint, a pairing solve against the metric, is symmetric: its
//! eigenvectors come in pairs, each pair the plane of one of the Coxeter element's four turns,
//! and the cosine of the turn its eigenvalue.

// Each of the three binaries uses part of this module.
#![allow(dead_code, unused_imports)]

gax::algebra! {
    algebra r80 "Euclidean 8D, R(8,0): numga's `a+b+c+d+e+f+g+h+`, E8's root space.";
    basis e1 = 1, e2 = 1, e3 = 1, e4 = 1, e5 = 1, e6 = 1, e7 = 1, e8 = 1;
    kind Scalar = [1];
    versor Vector = [e1, e2, e3, e4, e5, e6, e7, e8];
    kind Bivector = [e12, e13, e23, e14, e24, e34, e15, e25, e35, e45, e16, e26, e36, e46, e56, e17, e27, e37, e47, e57, e67, e18, e28, e38, e48, e58, e68, e78];
    versor Even = [1, e12, e13, e23, e14, e24, e34, e15, e25, e35, e45, e16, e26, e36, e46, e56, e17, e27, e37, e47, e57, e67, e18, e28, e38, e48, e58, e68, e78, e1234, e1235, e1245, e1345, e2345, e1236, e1246, e1346, e2346, e1256, e1356, e2356, e1456, e2456, e3456, e1237, e1247, e1347, e2347, e1257, e1357, e2357, e1457, e2457, e3457, e1267, e1367, e2367, e1467, e2467, e3467, e1567, e2567, e3567, e4567, e1238, e1248, e1348, e2348, e1258, e1358, e2358, e1458, e2458, e3458, e1268, e1368, e2368, e1468, e2468, e3468, e1568, e2568, e3568, e4568, e1278, e1378, e2378, e1478, e2478, e3478, e1578, e2578, e3578, e4578, e1678, e2678, e3678, e4678, e5678, e123456, e123457, e123467, e123567, e124567, e134567, e234567, e123458, e123468, e123568, e124568, e134568, e234568, e123478, e123578, e124578, e134578, e234578, e123678, e124678, e134678, e234678, e125678, e135678, e235678, e145678, e245678, e345678, e12345678];
    // quantum_affine_toda's field: antivectors, intrinsic, read by the roots with `&`.
    kind Antivector = [e2345678, e1345678, e1245678, e1235678, e1234678, e1234578, e1234568, e1234567] intrinsic;
}

pub use r80::{Antivector, Bivector, Even, Scalar, Vector};

/// A map on vectors, `Vector <- Vector`.
pub type VectorMap = Vector<(Vector,)>;
/// A unit rotor.
pub type Rotor = gax::Unit<Even>;

/// The number of steps of the Coxeter element before it repeats, E8's Coxeter number.
pub const COXETER_ORDER: usize = 30;

/// The vector with the given coefficients.
pub fn vector(c: [f64; 8]) -> Vector {
    Vector::from_coeffs(c)
}

/// The basis direction `e_k`, from one.
pub fn direction(k: usize) -> Vector {
    vector(core::array::from_fn(|i| if i + 1 == k { 1.0 } else { 0.0 }))
}

/// Eight mirrors of squared length two whose reflections generate E8's root system (numga's
/// order: the spinor-weight root first, then `a + b` and the chain `e_{k+1} - e_k`).
pub fn simple_roots() -> [Vector; 8] {
    let e = direction;
    // (a + h - b - c - d - e - f - g) / 2
    let first = (e(1) + e(8) - (2..8).map(e).sum::<Vector>()).gp(0.5);
    [
        first,
        e(1) + e(2),
        e(2) - e(1),
        e(3) - e(2),
        e(4) - e(3),
        e(5) - e(4),
        e(6) - e(5),
        e(7) - e(6),
    ]
}

/// The reflection in a mirror, as a map on vectors: the sandwich with the unit mirror.
pub fn reflection(mirror: Vector) -> VectorMap {
    mirror.unit() >> Vector::slot()
}

/// One root from each Coxeter orbit: each simple root reflected in the mirrors preceding it.
pub fn reflection_seeds(roots: &[Vector]) -> Vec<Vector> {
    let mut prefix: VectorMap = Vector::slot();
    roots
        .iter()
        .map(|root| {
            let seed = prefix.of(*root);
            prefix = prefix.of(reflection(*root));
            seed
        })
        .collect()
}

/// The first `n_steps` images of each seed under one motion, step by step: `[step][seed]`.
pub fn orbit(action: VectorMap, seeds: &[Vector], n_steps: usize) -> Vec<Vec<Vector>> {
    let mut now = seeds.to_vec();
    (0..n_steps)
        .map(|_| {
            let step = now.clone();
            now = now.iter().map(|x| action.of(*x)).collect();
            step
        })
        .collect()
}

/// The adjoint of a map on vectors under the metric, `M*(a) | b = a | M(b)`: a pairing solve.
pub fn adjoint(map: VectorMap) -> VectorMap {
    let metric = Vector::slot() | Vector::slot();
    metric.solve(Vector::slot() | map)
}

/// A map's symmetric part under the metric, the motion averaged with its adjoint.
pub fn symmetric(map: VectorMap) -> VectorMap {
    (map + adjoint(map)).gp(0.5)
}

/// The angle by which a motion turns the unit vector `x` in its plane with the unit vector
/// `y` (perpendicular to it): the direction of the image's two coordinates in that plane.
pub fn turn_angle(action: VectorMap, x: Vector, y: Vector) -> f64 {
    use gax::pga2d::Point;
    let image = action.of(x);
    let east = Point::<(), f64>::direction(1.0, 0.0);
    east.angle_between(Point::direction((image | x).s(), (image | y).s()))
}

/// The squared length of a vector, `x | x`.
pub fn length_squared(x: Vector) -> f64 {
    x.dot(x).s()
}

/// E8's root system as numga's scenarios build it.
pub struct RootSystem {
    /// The roots, `[step][orbit]`: thirty steps of the eight orbits.
    pub roots: Vec<Vec<Vector>>,
    /// Each invariant plane's first frame vector, from the slowest plane to the fastest.
    pub plane_x: [Vector; 4],
    /// Each plane's second frame vector, the direction the motion turns the first towards.
    pub plane_y: [Vector; 4],
    /// The oriented unit planes, `x ^ y`.
    pub planes: [Bivector; 4],
    /// How many turns each plane makes in the thirty steps: 1, 7, 11 and 13.
    pub rates: [f64; 4],
    /// The Coxeter element as a rotor, the product of the eight unit mirrors.
    pub coxeter: Rotor,
    /// The orbits' seeds, `roots[0]`.
    pub seeds: Vec<Vector>,
}

impl RootSystem {
    /// The eight root orbits, the four oriented plane frames, their winding rates and their
    /// rotor.
    pub fn new() -> RootSystem {
        let simple = simple_roots();
        let mirrors = simple.map(|r| r.unit());
        // Mirror by mirror, in pairs, so that each product stays even.
        let coxeter = (0..4)
            .map(|k| mirrors[2 * k] * mirrors[2 * k + 1])
            .reduce(|a, b| a * b)
            .expect("eight mirrors");
        let action = coxeter >> Vector::slot();
        // Prefix reflections pick one seed per orbit; their thirty-step paths exhaust the roots.
        let seeds = reflection_seeds(&simple);
        let roots = orbit(action, &seeds, COXETER_ORDER);

        // The motion and its adjoint average to the cosine of the turn in each invariant plane.
        // Equal eigenvalues come in pairs; descending cosines put the slowest plane first.
        let (cosines, eigenvectors) = symmetric(action).eigh();
        let pick = |k: usize| 7 - 2 * k;
        let plane_x: [Vector; 4] = core::array::from_fn(|k| eigenvectors[pick(k)]);
        let plane_y: [Vector; 4] = core::array::from_fn(|k| {
            (action.of(plane_x[k]) - plane_x[k].gp(cosines[pick(k)])).normalized()
        });
        let planes = core::array::from_fn(|k| plane_x[k] ^ plane_y[k]);
        let period = COXETER_ORDER as f64 / core::f64::consts::TAU;
        let rates = core::array::from_fn(|k| turn_angle(action, plane_x[k], plane_y[k]) * period);
        RootSystem {
            roots,
            plane_x,
            plane_y,
            planes,
            rates,
            coxeter,
            seeds,
        }
    }

    /// Every root, orbit by orbit within each step.
    pub fn all(&self) -> impl Iterator<Item = Vector> + '_ {
        self.roots.iter().flatten().copied()
    }

    /// The bivector whose exponential is one Coxeter step: each plane turned by its rate's
    /// share of a whole turn, `-Σ planes · rates · π / 30`.
    pub fn generator(&self) -> Bivector {
        let step = -core::f64::consts::PI / COXETER_ORDER as f64;
        (0..4).fold(Bivector::zero(), |sum, k| {
            sum + self.planes[k].gp(self.rates[k] * step)
        })
    }
}

impl Default for RootSystem {
    fn default() -> Self {
        RootSystem::new()
    }
}

/// The pairs of roots at the nearest distance, the edges of the root polytope, by index into
/// the flat list.
pub fn nearest_pairs(roots: &[Vector]) -> Vec<(usize, usize)> {
    let gap = |i: usize, j: usize| length_squared(roots[i] - roots[j]);
    let pairs = (0..roots.len()).flat_map(|i| (i + 1..roots.len()).map(move |j| (i, j)));
    let nearest = pairs
        .clone()
        .map(|(i, j)| gap(i, j))
        .filter(|d| *d > 1e-8)
        .fold(f64::INFINITY, f64::min);
    pairs
        .filter(|&(i, j)| (gap(i, j) - nearest).abs() <= 1e-8 + 1e-5 * nearest)
        .collect()
}

/// numga's render: the orbits' colours and a vector's place in a drawing plane.
pub mod look {
    use super::Vector;
    use gax_numga_examples::{Light, Point2};

    /// The eight orbits' colours, numga's, as lights.
    pub fn orbit_colour(orbit: usize) -> Light {
        const SRGB: [[f32; 3]; 8] = [
            [0.98, 0.72, 0.27],
            [0.96, 0.39, 0.32],
            [0.83, 0.40, 0.72],
            [0.60, 0.48, 0.92],
            [0.32, 0.63, 0.95],
            [0.24, 0.80, 0.82],
            [0.35, 0.82, 0.59],
            [0.76, 0.84, 0.40],
        ];
        let [r, g, b] = SRGB[orbit % 8];
        Light::from_srgb(r, g, b, 1.7)
    }

    /// A vector's coordinates in a drawing plane, its inner products with the plane's two
    /// orthonormal vectors, as a point of the drawing.
    pub fn coordinates(v: Vector, horizontal: Vector, vertical: Vector) -> Point2 {
        Point2::xy((v | horizontal).s() as f32, (v | vertical).s() as f32)
    }
}
