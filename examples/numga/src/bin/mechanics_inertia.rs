//! numga's `mechanics/simplex.py` and `mechanics/inertia.py`: the inertia of simplices, and the
//! principal frame of an inertia recovered with two eigenproblems.
//!
//! The inertia of point masses is the map `p & [p, ·]` summed over the points, from rates to
//! momenta. A uniform simplex has the inertia of a few lumped, mass-weighted barycentric points
//! (exactly), which a grid or Monte Carlo sampling of the simplex approaches. From an inertia
//! alone the second-moment map (`Point <- Plane`) comes back by least squares against the
//! construction `Point & [Plane.dual(), ·]`; its form against the metric on planes gives the
//! principal planes (a generalized symmetric eigenproblem), and the motor taking them onto the
//! coordinate planes is a common eigenvector of the maps `R ↦ reference R source`. The same
//! text runs in Euclidean PGA3D, in spherical R(4,0,0) and in the PGA of four dimensions.
//!
//! The animation shows a tetrahedron tumbling freely (the shared Lie integrators), its
//! principal frame and second-moment ellipsoid recovered every frame from its moving inertia
//! alone, and the convergence of grid and Monte Carlo sampling of a tetrahedron's inertia to
//! the lumped one, which equals gax's mesh moments exactly.

use gax::pga3d::{Line, Point};
use gax_numga_examples::{
    Anim, Axes, Camera, Canvas, Lens, Marker, Scene3, backdrop, caption, palette, plot, run,
};

#[path = "../shared/mechanics_lie.rs"]
mod lie;
#[allow(unused_imports)]
use lie::Lie as _;

/// R(4,0,0): numga's spherical model `x+y+z+w+`, laid out as PGA3D with `e0` squaring to +1.
mod sga3d {
    gax::algebra! {
        algebra sga3d "Spherical 3-space as R(4,0,0), with the PGA3D layout.";
        basis e0 = 1, e1 = 1, e2 = 1, e3 = 1;
        kind Scalar = [1];
        kind Plane = [e1, e2, e3, e0];
        kind Line = [e23, e31, e12, e01, e02, e03];
        kind Point = [e032, e013, e021, e123];
        kind Pseudoscalar = [e0123];
        versor Motor = [1, e23, e31, e12, e01, e02, e03, e0123];
    }
    pub use sga3d::*;
}

/// R(4,0,1): numga's `x+y+z+v+w0`, the PGA of four dimensions.
mod pga4d {
    gax::algebra! {
        algebra pga4d "Plane-based PGA of 4D Euclidean space, R(4,0,1).";
        basis e0 = 0, e1 = 1, e2 = 1, e3 = 1, e4 = 1;
        kind Scalar = [1];
        kind Plane = [e1, e2, e3, e4, e0];
        kind Line = [e12, e13, e14, e23, e24, e34, e01, e02, e03, e04];
        kind Forque = [e234, e134, e124, e123, e034, e024, e023, e014, e013, e012];
        kind Point = [e0234, e0134, e0124, e0123, e1234];
        kind Pseudoscalar = [e01234];
        versor Motor = [1, e12, e13, e14, e23, e24, e34, e01, e02, e03, e04,
            e1234, e0234, e0134, e0124, e0123];
    }
    pub use pga4d::*;
}

/// A small xorshift generator: numga's NumPy streams cannot be reproduced.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    /// Uniform in `(0, 1]`.
    pub fn unit(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 11) as f64 + 1.0) / (1u64 << 53) as f64
    }

    /// Standard normal, by Box and Muller.
    pub fn normal(&mut self) -> f64 {
        let (u, v) = (self.unit(), self.unit());
        (-2.0 * u.ln()).sqrt() * (core::f64::consts::TAU * v).cos()
    }
}

/// numga's `inertia.py` over the kinds `Plane`, `Point`, `Line` (the bivectors, rates),
/// `Forque` (the antibivectors) and `Motor` (the even versor) in scope, for an algebra whose
/// points have `$n` coefficients (and whose motors list the scalar first).
macro_rules! principal_frame {
    ($n:literal) => {
        /// A rate-to-momentum map.
        pub type Inertia = Forque<(Line,), f64>;
        /// A second-moment map: `b & S(a)` is `Σ m (a & p)(b & p)` over the mass points.
        pub type Moment = Point<(Plane,), f64>;
        pub type Pl = Plane<(), f64>;
        pub type Pt = Point<(), f64>;
        pub type Mo = gax::Unit<Motor<(), f64>>;
        const N: usize = $n;

        /// The `i`-th basis plane and point.
        pub fn plane_blade(i: usize) -> Pl {
            Plane::from_coeffs(core::array::from_fn(|k| f64::from(u8::from(k == i))))
        }
        pub fn point_blade(i: usize) -> Pt {
            Point::from_coeffs(core::array::from_fn(|k| f64::from(u8::from(k == i))))
        }

        /// The inertia of point masses, `Σ m p & [p, ·]`.
        pub fn inertia_of(points: &[Pt], masses: &[f64]) -> Inertia {
            let mut inertia = Inertia::zero();
            for (p, m) in points.iter().zip(masses) {
                inertia += (*p & p.commutator(Line::slot())) * *m;
            }
            inertia
        }

        /// The point cloud's second-moment map, recovered from its inertia alone.
        ///
        /// The construction `Point & [Plane.dual(), Line]` has slots (Point, Plane, Line) and
        /// gives a forque; the inertia matches its Line slot, leaving the second moment, a
        /// tensor on (Point, Plane), as the unknown. `lstsq_pair` unbinds both slots at once,
        /// as numga's `lstsq` does. The system is overdetermined yet consistent, and singular
        /// (the construction only sees the moment's symmetric part): the pseudo-inverse gives
        /// the least-norm solution, as numga's.
        pub fn second_moment(inertia: Inertia) -> Moment {
            let construction = Point::slot() & Plane::slot().dual().commutator(Line::slot());
            construction.lstsq_pair(inertia)
        }

        /// The principal planes of a second moment, with their eigenvalues: the generalized
        /// eigenvectors of the metric on planes `Plane | Plane` against the moment form
        /// `Plane & moment` (this order gives PGA's plane at infinity a zero eigenvalue rather
        /// than an infinite one), sorted by norm, descending, so that PGA's null plane is last.
        pub fn principal_planes(moment: Moment) -> [(f64, Pl); N] {
            let metric = Plane::slot() | Plane::slot();
            let form = Plane::slot() & moment;
            let (values, planes) = metric.eigh_with(form);
            let mut pairs: [(f64, Pl); N] = core::array::from_fn(|k| (values[k], planes[k]));
            // numga's `argsort()[::-1]`: a stable ascending sort, reversed.
            pairs.sort_by(|a, b| a.1.norm().total_cmp(&b.1.norm()));
            pairs.reverse();
            pairs
        }

        /// The basis planes in numga's reference order: by norm, descending (a stable sort,
        /// reversed), so that PGA's null plane comes last.
        pub fn reference() -> [Pl; N] {
            let mut planes: [Pl; N] = core::array::from_fn(plane_blade);
            planes.sort_by(|a, b| a.norm().total_cmp(&b.norm()));
            planes.reverse();
            planes
        }

        /// A motor taking the principal planes onto the reference planes, in order, up to
        /// their orientations.
        ///
        /// A matching motor `R` is an eigenvector of each map `R ↦ reference R source`, with
        /// the eigenvalue the source plane's norm times either sign; the maps commute, so their
        /// weighted sum (weights of descending powers of two keep the sign choices apart) finds
        /// the common eigenvectors in one eigenproblem. PGA's null planes contribute zero on
        /// the even versors. numga solves it with a general complex `eig`; gax's `eig` would too,
        /// but the null planes' repeated zero eigenvalues make its eigenvectors fragile, and here
        /// the spectrum is known in advance, one eigenvalue
        /// per sign choice `Σ w σ |source|`, so each eigenvector comes from a few steps of
        /// inverse iteration with gax's `solve`. As in numga, the one with the largest scalar
        /// part is the motor (the pure ideal candidates of PGA have none).
        pub fn diagonalizing_motor(source: &[(f64, Pl); N], reference: &[Pl; N]) -> Mo {
            let slot = Motor::<(), f64>::slot();
            let mut t = Motor::<(Motor,), f64>::zero();
            let mut weights = [0.0; N];
            for k in 0..N {
                weights[k] = f64::from(1u32 << (N - 1 - k));
                t += (reference[k] * slot * source[k].1).cast::<Motor>() * weights[k];
            }
            let norms: Vec<f64> = source.iter().map(|(_, p)| p.norm()).collect();
            let scale = norms.iter().zip(&weights).map(|(n, w)| n * w).sum::<f64>();
            let live = norms.iter().filter(|n| **n > 1e-9 * norms[0]).count();
            let size = |x: Motor<(), f64>| x.c.iter().map(|v| v * v).sum::<f64>().sqrt();
            let mut best: Option<Motor<(), f64>> = None;
            for mask in 0..1usize << live {
                let lam: f64 = (0..live)
                    .map(|k| if mask >> k & 1 == 1 { -1.0 } else { 1.0 } * weights[k] * norms[k])
                    .sum();
                let shifted = t - slot * (lam + 1e-9 * scale);
                let mut x = Motor::from_coeffs(core::array::from_fn(|k| 1.0 + 0.1 * k as f64));
                for _ in 0..4 {
                    x = shifted.solve(x);
                    x = x * (1.0 / size(x));
                }
                let residual = size(t.of(x) - x * lam);
                if residual < 1e-6 * scale
                    && best.is_none_or(|b: Motor<(), f64>| x.c[0].abs() > b.c[0].abs())
                {
                    best = Some(x);
                }
            }
            best.expect("a motor eigenvector").normalized()
        }

        /// The inertia moved by a motor: its input and its output.
        pub fn moved(m: Mo, inertia: Inertia) -> Inertia {
            m >> inertia.of(m << Line::slot())
        }

        /// The energy form of an inertia on the bivector blades, `a & I(b)`.
        pub fn energies(inertia: Inertia) -> Vec<Vec<f64>> {
            let blade = |i: usize| {
                Line::<(), f64>::from_coeffs(core::array::from_fn(|k| f64::from(u8::from(k == i))))
            };
            let nb = Line::<(), f64>::zero().c.len();
            (0..nb)
                .map(|a| {
                    (0..nb)
                        .map(|b| (blade(a) & inertia.of(blade(b))).s())
                        .collect()
                })
                .collect()
        }

        /// Random points normalized, placed with a random motor, with masses in `[0.5, 1.5]`.
        pub fn cloud(seed: u64) -> (Vec<Pt>, Vec<f64>) {
            let mut rng = crate::Rng::new(seed);
            let points: Vec<Pt> = (0..80)
                .map(|_| {
                    let p = Point::from_coeffs(core::array::from_fn(|_| rng.normal()));
                    p * (1.0 / p.norm())
                })
                .collect();
            let placement =
                Line::<(), f64>::from_coeffs(core::array::from_fn(|_| 0.3 * rng.normal())).exp();
            let masses = (0..80).map(|_| 0.5 + rng.unit()).collect();
            (points.into_iter().map(|p| placement >> p).collect(), masses)
        }

        /// numga's `main`: the two constructions, an arbitrary mass cloud and a moved diagonal
        /// tensor, each diagonalized from its inertia alone. Returns the two motors and the
        /// energy forms of the aligned cloud, the recovered tensor and the diagonal one.
        pub fn principal_frames(seed: u64) -> (Mo, Mo, [Vec<Vec<f64>>; 3]) {
            let (points, masses) = cloud(seed);
            let reference = reference();
            let mut rng = crate::Rng::new(seed + 1);
            let placement =
                Line::<(), f64>::from_coeffs(core::array::from_fn(|_| 0.2 * rng.normal())).exp();
            // 1. Construct and diagonalize inertia from arbitrary mass points.
            let cloud_inertia = inertia_of(&points, &masses);
            let cloud_motor =
                diagonalizing_motor(&principal_planes(second_moment(cloud_inertia)), &reference);
            let cloud_diagonal = moved(cloud_motor, cloud_inertia);
            // 2. Diagonal second moments induce a diagonal energy form on bivectors: move that
            // inertia, then find a diagonalizing motor from the moved tensor.
            let basis: Vec<Pt> = (0..N).map(point_blade).collect();
            let second: Vec<f64> = (0..N).map(|k| f64::from(1u32 << k)).collect();
            let diagonal = inertia_of(&basis, &second);
            let moved_diagonal = moved(placement, diagonal);
            let motor =
                diagonalizing_motor(&principal_planes(second_moment(moved_diagonal)), &reference);
            let recovered = moved(motor, moved_diagonal);
            (
                cloud_motor,
                motor,
                [
                    energies(cloud_diagonal),
                    energies(recovered),
                    energies(diagonal),
                ],
            )
        }
    };
}

/// Euclidean PGA3D, numga's `x+y+z+w0`.
pub mod e3 {
    use gax::pga3d::{Line, Line as Forque, Motor, Plane, Point};
    principal_frame!(4);
}

/// Spherical R(4,0,0), numga's `x+y+z+w+`.
pub mod s3 {
    use crate::sga3d::{Line, Line as Forque, Motor, Plane, Point};
    principal_frame!(4);
}

/// The PGA of four dimensions, numga's `x+y+z+v+w0`.
pub mod e4 {
    use crate::pga4d::{Forque, Line, Motor, Plane, Point};
    principal_frame!(5);
}

/// The Lie steppers for PGA3D motors.
mod rigid {
    /// The algebra.
    pub type G = gax::motions::Pga3d;
    pub type M = crate::lie::M<G>;
}

/// numga's `simplex.py`: inertia maps of simplices, in PGA3D.
mod simplex {
    use super::*;
    pub use crate::e3::Inertia;
    pub type P = Point<(), f64>;

    /// The barycentric weights mapping a uniform simplex to its equivalent lumped masses:
    /// `(1/n + (I - 1/n) sqrt(1 / (1 + n))) / n`, one row per lumped point.
    pub fn weights(n: usize) -> Vec<Vec<f64>> {
        let f = (1.0 / (1.0 + n as f64)).sqrt();
        (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| {
                        let d = f64::from(u8::from(i == j)) - 1.0 / n as f64;
                        (1.0 / n as f64 + d * f) / n as f64
                    })
                    .collect()
            })
            .collect()
    }

    /// Mass-weighted points from barycentric samples: each row of weights gives the point
    /// `Σ w c`, of weight (mass) `Σ w`, scaled by `1 / sqrt(Σ w)` so that `p & [p, ·]`, which
    /// is quadratic in `p`, carries the mass once.
    pub fn samples(weights: &[Vec<f64>], corners: &[P]) -> Vec<P> {
        weights
            .iter()
            .map(|row| {
                let mass: f64 = row.iter().sum();
                let p = row
                    .iter()
                    .zip(corners)
                    .fold(Point::zero(), |acc, (w, c)| acc + *c * *w);
                p * (1.0 / mass.sqrt())
            })
            .collect()
    }

    /// The inertia map of barycentric samples of a simplex.
    pub fn inertia(weights: &[Vec<f64>], corners: &[P]) -> Inertia {
        let points = samples(weights, corners);
        crate::e3::inertia_of(&points, &vec![1.0; points.len()])
    }

    /// The inertia of the uniform simplex of unit mass, by its lumped masses (exact).
    pub fn lumped(corners: &[P]) -> Inertia {
        inertia(&weights(corners.len()), corners)
    }

    /// The barycentric grid of a tetrahedron: the midpoints of `n` cells along each of three
    /// barycentric coordinates, those inside, completed by the fourth; unit mass in all.
    pub fn grid(n: usize) -> Vec<Vec<f64>> {
        let s: Vec<f64> = (0..n).map(|i| (i as f64 + 0.5) / n as f64).collect();
        let mut rows = Vec::new();
        for a in &s {
            for b in &s {
                for c in &s {
                    if a + b + c < 1.0 {
                        rows.push(vec![*a, *b, *c, 1.0 - a - b - c]);
                    }
                }
            }
        }
        let count = rows.len() as f64;
        rows.iter()
            .map(|r| r.iter().map(|w| w / count).collect())
            .collect()
    }

    /// Uniform barycentric weights: exponential variates normalized (a flat Dirichlet).
    pub fn random_weights(count: usize, corners: usize, rng: &mut crate::Rng) -> Vec<Vec<f64>> {
        (0..count)
            .map(|_| {
                let e: Vec<f64> = (0..corners).map(|_| -rng.unit().ln()).collect();
                let sum: f64 = e.iter().sum();
                e.iter().map(|v| v / sum / count as f64).collect()
            })
            .collect()
    }

    /// numga's demo tetrahedron.
    pub fn tetrahedron() -> Vec<P> {
        vec![
            Point::xyz(1.0, 0.0, 0.0),
            Point::xyz(-0.5, 0.866, 0.0),
            Point::xyz(-0.5, -0.866, 0.0),
            Point::xyz(0.0, 0.0, 1.414),
        ]
    }

    /// The relative difference of two inertias, by their coefficients.
    pub fn relative(a: Inertia, b: Inertia) -> f64 {
        let d = (a - b).c.iter().flatten().map(|v| v * v).sum::<f64>();
        let n = b.c.iter().flatten().map(|v| v * v).sum::<f64>();
        (d / n).sqrt()
    }
}

/// The scene: a lopsided tetrahedron, centred on its centre of mass, tumbling freely.
struct Scene {
    corners: Vec<simplex::P>,
    inertia: e3::Inertia,
    motors: Vec<rigid::M>,
    /// Grid and Monte Carlo errors against the lumped inertia: (samples, error).
    grid: Vec<[f64; 2]>,
    monte_carlo: Vec<[f64; 2]>,
    /// The mesh moments' difference from the lumped second moment.
    mesh: f64,
}

const SECONDS: f32 = 12.0;
const DT: f64 = 0.01;

fn scene() -> &'static Scene {
    static SCENE: std::sync::OnceLock<Scene> = std::sync::OnceLock::new();
    SCENE.get_or_init(|| {
        let raw = [
            [1.3, 0.0, -0.2],
            [-0.6, 0.9, 0.0],
            [-0.5, -0.8, 0.1],
            [0.1, 0.3, 1.5],
        ];
        let c = (0..3)
            .map(|k| raw.iter().map(|v| v[k]).sum::<f64>() / 4.0)
            .collect::<Vec<_>>();
        let corners: Vec<simplex::P> = raw
            .iter()
            .map(|v| Point::xyz(v[0] - c[0], v[1] - c[1], v[2] - c[2]))
            .collect();
        let inertia = simplex::lumped(&corners);
        let inertia_inv = inertia.inverse();
        // A spin about a skew axis through the centre of mass.
        let mut motor = Line::<(), f64>::zero().exp();
        let mut rate = Line::new(0.4, 2.2, 0.9, 0.0, 0.0, 0.0);
        let mut motors = vec![motor];
        for _ in 0..(f64::from(SECONDS) / DT).round() as usize {
            (motor, rate) =
                rigid::G::explicit_rkmk4(motor, rate, inertia, inertia_inv, DT, &rigid::G::free);
            motors.push(motor);
        }
        // numga's demo tetrahedron, sampled.
        let tetra = simplex::tetrahedron();
        let exact = simplex::lumped(&tetra);
        let grid = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 100]
            .iter()
            .map(|&n| {
                let w = simplex::grid(n);
                [
                    w.len() as f64,
                    simplex::relative(simplex::inertia(&w, &tetra), exact),
                ]
            })
            .collect();
        // One Monte Carlo stream, its running mean read at logarithmic checkpoints.
        let mut rng = Rng::new(0);
        let draws = simplex::random_weights(100_000, 4, &mut rng);
        let mut sum = e3::Inertia::zero();
        let mut monte_carlo = Vec::new();
        let mut next = 10;
        for (i, row) in draws.iter().enumerate() {
            // Unit mass per draw; the running mean divides by the count.
            let unit: Vec<f64> = row.iter().map(|w| w * draws.len() as f64).collect();
            sum += simplex::inertia(&[unit], &tetra);
            if i + 1 == next {
                monte_carlo.push([
                    next as f64,
                    simplex::relative(sum * (1.0 / next as f64), exact),
                ]);
                next = (next as f64 * 1.5).ceil() as usize;
            }
        }
        // gax's mesh moments of the same tetrahedron, per unit volume.
        let faces = [[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]];
        let mesh = gax::pga3d::Moments::of_mesh(&tetra, &faces);
        let moment = e3::second_moment(exact);
        let planes: Vec<e3::Pl> = (0..4).map(e3::plane_blade).collect();
        let mut worst: f64 = 0.0;
        for a in &planes {
            for b in &planes {
                let ours = (*b & moment.of(*a)).s();
                let theirs = mesh.form.of(*a).of(*b).s() / mesh.volume();
                worst = worst.max((ours - theirs).abs());
            }
        }
        Scene {
            corners,
            inertia,
            motors,
            grid,
            monte_carlo,
            mesh: worst,
        }
    })
}

fn xyz(p: Point<(), f64>) -> [f32; 3] {
    p.to_euclidean().map(|v| v as f32)
}

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let sc = scene();
    let (w, h) = (c.width as f32, c.height as f32);
    let u = h / 540.0;
    let k = ((f64::from(t.rem_euclid(SECONDS)) / DT) as usize).min(sc.motors.len() - 1);
    let m = sc.motors[k];
    caption(
        c,
        "INERTIA: PRINCIPAL FRAMES FROM INERTIA ALONE",
        "LEAST SQUARES, THEN TWO EIGENPROBLEMS (PGA3D)",
    );
    // The body's inertia in the world, and its principal frame recovered from it alone.
    let world = e3::moved(m, sc.inertia);
    let moment = e3::second_moment(world);
    let planes = e3::principal_planes(moment);
    let frame = e3::diagonalizing_motor(&planes, &e3::reference());
    let back = frame.reverse();
    // The total mass is the moment form on the plane at infinity.
    let at_infinity = e3::plane_blade(3);
    let mass = (at_infinity & moment.of(at_infinity)).s();
    // The reference order is z, y, x: the source plane k is normal to axis 2 - k. Its second
    // moment is 1 / its eigenvalue, and the equivalent solid ellipsoid has semi-axes
    // sqrt(5 c / m).
    let semi: [f64; 3] = core::array::from_fn(|axis| (5.0 / (planes[2 - axis].0 * mass)).sqrt());
    let left = w * 0.55;
    let cam = Camera::orbit(
        left as usize,
        c.height,
        [0.0, 0.0, 0.0],
        9.0,
        0.5 + 0.2 * t,
        0.35,
        Lens::Perspective(0.42),
    );
    let mut s = Scene3::new(cam);
    let corners: Vec<[f32; 3]> = sc.corners.iter().map(|p| xyz(m >> *p)).collect();
    for f in [[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]] {
        let (a, b, d) = (corners[f[0]], corners[f[1]], corners[f[2]]);
        let col = s.lit(a, b, d, palette::purple());
        s.tri(a, b, d, col, 0.55);
        for (x, y) in [(a, b), (b, d), (d, a)] {
            s.seg(x, y, 1.2 * u, palette::ink(), 0.7);
        }
    }
    // The principal axes and the second-moment ellipsoid, in the recovered frame.
    let colours = [palette::red(), palette::green(), palette::sky()];
    let in_frame = |v: [f64; 3]| xyz(back >> Point::xyz(v[0], v[1], v[2]));
    for axis in 0..3 {
        let mut tip = [0.0; 3];
        tip[axis] = semi[axis];
        let (a, b) = (in_frame(tip.map(|v| -v)), in_frame(tip));
        s.seg(a, b, 2.2 * u, colours[axis], 1.0);
        s.dot(b, Marker::Dot, 6.0 * u, colours[axis]);
        // The ellipse in the plane of the other two axes.
        let (i, j) = ((axis + 1) % 3, (axis + 2) % 3);
        let ring: Vec<[f32; 3]> = (0..=64)
            .map(|q| {
                let a = core::f64::consts::TAU * q as f64 / 64.0;
                let mut v = [0.0; 3];
                v[i] = semi[i] * a.cos();
                v[j] = semi[j] * a.sin();
                in_frame(v)
            })
            .collect();
        s.polyline(&ring, 1.2 * u, colours[axis], 0.6);
    }
    s.draw(c);
    // Sampling a tetrahedron's inertia: the error of a grid and of Monte Carlo against the
    // lumped inertia, revealed over the loop.
    let ax = Axes::new(
        plot::inset([left, 0.0, w, h], 60.0 * u, 110.0 * u, 24.0 * u, 64.0 * u),
        [1.0, 1e5],
        [1e-17, 1e5],
    )
    .log_x()
    .log_y();
    ax.frame(c, "SAMPLED SIMPLEX INERTIA", "SAMPLES", "RELATIVE ERROR");
    let shown = (t.rem_euclid(SECONDS) / SECONDS * 1.25).min(1.0);
    let reveal = |pts: &[[f64; 2]]| -> Vec<[f32; 2]> {
        let n = ((pts.len() as f32 * shown).ceil() as usize).clamp(1, pts.len());
        pts[..n]
            .iter()
            .map(|p| [p[0] as f32, p[1] as f32])
            .collect()
    };
    let (g, r) = (reveal(&sc.grid), reveal(&sc.monte_carlo));
    ax.polyline(c, &g, 1.8, palette::orange(), 1.0);
    ax.scatter(c, &g, Marker::Dot, 5.0 * u, palette::orange(), 1.0);
    ax.polyline(c, &r, 1.4, palette::sky(), 1.0);
    // The lumped inertia takes four points; gax's mesh moments agree with it to rounding.
    let floor = 2e-16f32;
    ax.scatter(
        c,
        &[[4.0, floor]],
        Marker::Star,
        13.0 * u,
        palette::green(),
        1.0,
    );
    ax.scatter(
        c,
        &[[4.0, (sc.mesh as f32).max(floor) * 8.0]],
        Marker::Square,
        9.0 * u,
        palette::yellow(),
        1.0,
    );
    ax.legend(
        c,
        &[
            ("GRID", palette::orange()),
            ("MONTE CARLO", palette::sky()),
            ("LUMPED: 4 POINTS", palette::green()),
            ("MESH MOMENTS", palette::yellow()),
        ],
    );
}

fn main() {
    run(Anim::new("inertia", SECONDS).size(960, 540), draw);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// numga's check in `main`: both aligned energy forms are diagonal, and the round trip
    /// recovers the diagonal it started from, up to a permutation of the axes.
    fn check(energies: &[Vec<Vec<f64>>; 3]) {
        for energy in &energies[..2] {
            let largest = energy
                .iter()
                .flatten()
                .fold(0.0, |m: f64, v| m.max(v.abs()));
            for (a, row) in energy.iter().enumerate() {
                for (b, v) in row.iter().enumerate() {
                    if a != b {
                        assert!(v.abs() < 1e-12 * largest, "{a} {b}: {v} of {largest}");
                    }
                }
            }
        }
        let diag = |e: &Vec<Vec<f64>>| {
            let mut d: Vec<f64> = (0..e.len()).map(|i| e[i][i]).collect();
            d.sort_by(f64::total_cmp);
            d
        };
        for (a, b) in diag(&energies[1]).iter().zip(diag(&energies[2])) {
            assert!((a - b).abs() <= 1e-8 * b.abs(), "{a} vs {b}");
        }
    }

    #[test]
    fn principal_frames_diagonalize_inertia_in_pga3d() {
        check(&e3::principal_frames(0).2);
    }

    #[test]
    fn principal_frames_diagonalize_inertia_in_spherical_space() {
        check(&s3::principal_frames(0).2);
    }

    #[test]
    fn principal_frames_diagonalize_inertia_in_pga4d() {
        check(&e4::principal_frames(0).2);
    }

    /// The second moment recovered from the inertia is the cloud's own: `b & S(a)` is
    /// `Σ m (a & p)(b & p)`.
    #[test]
    fn second_moment_is_the_clouds() {
        let (points, masses) = e3::cloud(3);
        let moment = e3::second_moment(e3::inertia_of(&points, &masses));
        let planes = [
            e3::plane_blade(0) + e3::plane_blade(3) * 0.5,
            e3::plane_blade(1) - e3::plane_blade(2) * 0.3,
            e3::plane_blade(2),
        ];
        for a in planes {
            for b in planes {
                let direct: f64 = points
                    .iter()
                    .zip(&masses)
                    .map(|(p, m)| m * (a & *p).s() * (b & *p).s())
                    .sum();
                let ours = (b & moment.of(a)).s();
                assert!(
                    (ours - direct).abs() < 1e-9 * direct.abs().max(1.0),
                    "{ours} vs {direct}"
                );
            }
        }
    }

    /// The lumped inertia of a tetrahedron equals gax's mesh moments (both exact), the grid and
    /// Monte Carlo approach it, and the inertia maps rates to momenta and back (numga's demo).
    #[test]
    fn simplex_inertia() {
        let sc = scene();
        assert!(sc.mesh < 1e-14, "{}", sc.mesh);
        let (n, grid) = (sc.grid[sc.grid.len() - 1][0], sc.grid[sc.grid.len() - 1][1]);
        assert!(n > 1e5 && grid < 2e-2, "{n} {grid}");
        let mc = sc.monte_carlo[sc.monte_carlo.len() - 1][1];
        assert!(mc < 2e-2, "{mc}");
        // The grid converges.
        assert!(sc.grid[4][1] > 2.0 * grid);
        let tetra = simplex::tetrahedron();
        let inertia = simplex::lumped(&tetra);
        let rate = Line::new(1.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        let momentum = inertia.of(rate);
        let back = inertia.solve(momentum);
        assert!((back - rate).c.iter().all(|v| v.abs() < 1e-12));
        assert!((rate & momentum).s() * 0.5 > 0.0);
    }

    /// The frame recovered from the moving body's inertia turns with the body.
    #[test]
    fn the_recovered_frame_is_fixed_in_the_body() {
        let sc = scene();
        let frame = |m: rigid::M| {
            let planes = e3::principal_planes(e3::second_moment(e3::moved(m, sc.inertia)));
            e3::diagonalizing_motor(&planes, &e3::reference())
        };
        let start = frame(sc.motors[0]);
        for &k in &[100, 500, 1000] {
            let m = sc.motors[k];
            // The body-frame axes of the recovered frame: the same lines as at the start, up
            // to their orientation.
            for axis in [
                Point::direction(1.0, 0.0, 0.0),
                Point::direction(0.0, 1.0, 0.0),
            ] {
                let now = m.reverse() >> (frame(m).reverse() >> axis);
                let then = start.reverse() >> axis;
                let cos =
                    now.e032() * then.e032() + now.e013() * then.e013() + now.e021() * then.e021();
                assert!((cos.abs() - 1.0).abs() < 1e-9, "{cos}");
            }
        }
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws(super::draw, 0.5);
    }
}
