//! numga's field lines (its render's `field_lines`): smooth, confined magnetic field lines inside
//! the torus, traced by Runge–Kutta through the field smoothed around the axis and across the
//! tube and tapered off the wall. In space (VGA3D), the drawing's boundary: the field at each
//! tetrahedron's centre, a direction, is read along its cylindrical frame (out from the axis,
//! around it, up); at a point, those readings averaged with smooth weights by how near the samples
//! lie around the axis and across the tube are rebuilt along the point's own frame.

use super::beltrami::{Magnetic, Point, coordinates, time};
use gax::field::Field;
use gax::vga3d::{Bivector, Vector};
use gax_numga_examples::signal::phasor;

/// The torus the lines are traced in, its tube tapering their motion off the wall.
const TUBE_MAJOR: f64 = 1.0;
const TUBE_MINOR: f64 = 0.33;
/// The shells of seeds: each its share of the tube's radius, its count of seeds, and the angles
/// around the tube and the axis of its first seed.
pub const SEED_SHARES: [f64; 3] = [0.24, 0.52, 0.78];
pub const SEEDS: [usize; 3] = [3, 4, 5];
const SEED_ANGLES_AROUND_TUBE: [f64; 3] = [0.20, 1.10, 2.15];
const SEED_ANGLES_AROUND_AXIS: [f64; 3] = [
    0.0,
    core::f64::consts::FRAC_PI_4,
    core::f64::consts::PI / 5.0,
];
/// The traced steps either way from a seed, their length, and every how many a point is kept.
const TRACE_STEPS: usize = 170;
const TRACE_STEP: f64 = 0.018;
const TRACE_STRIDE: usize = 2;
/// The smoothing's reach around the axis (a chord of the unit circle) and across the tube; the
/// field's floor in its normalisation, and the inward pull at the wall, as shares of its
/// typical strength.
const SMOOTHING_TURN: f64 = 0.58;
const SMOOTHING_REACH: f64 = 0.20;
const SPEED_FLOOR: f64 = 0.14;
const WALL_PULL: f64 = 0.08;

/// A point or direction of space, read from spacetime.
pub fn space(p: Point) -> Vector {
    let [x, y, z] = coordinates(p);
    Vector::new(x, y, z)
}

/// The cylindrical frame at a point: out from the axis, around it, and up.
fn frame(p: Vector) -> ([Vector; 3], f64, f64) {
    let height = p.e3();
    let level = Vector::new(p.e1(), p.e2(), 0.0);
    let distance = level.norm();
    let outward = level * (1.0 / distance);
    let around = outward.commutator(Bivector::new(0.0, 0.0, 1.0));
    (
        [outward, around, Vector::new(0.0, 0.0, 1.0)],
        distance,
        height,
    )
}

/// The samples of a field: each wedge's two outer tetrahedra averaged, to remove the alternation
/// of the slices' diagonals, then their frames and readings.
pub struct Samples {
    outwards: Vec<Vector>,
    distances: Vec<f64>,
    heights: Vec<f64>,
    readings: Vec<[f64; 3]>,
    typical: f64,
}

impl Samples {
    pub fn new(
        centres: &Field<Point>,
        volumes: &Field<gax::sta::intrinsic::Vector>,
        magnetic: &Field<Magnetic>,
    ) -> Self {
        // The field as a direction: the bivector times the unit volume.
        let directions: Vec<Vector> = magnetic
            .iter()
            .zip(volumes.iter())
            .map(|(b, v)| {
                let unit = v * (1.0 / (v & time()).norm());
                space((b * unit).grade::<3>())
            })
            .collect();
        let sites: Vec<Vector> = centres.iter().map(space).collect();
        let wedge = sites.len() / 3;
        let average = |x: &[Vector]| -> Vec<Vector> {
            (0..wedge)
                .map(|k| (x[k] + x[wedge + k]) * 0.5)
                .chain(x[2 * wedge..].iter().copied())
                .collect()
        };
        let (sites, fields) = (average(&sites), average(&directions));
        let mut samples = Samples {
            outwards: Vec::new(),
            distances: Vec::new(),
            heights: Vec::new(),
            readings: Vec::new(),
            typical: 0.0,
        };
        for (&s, &f) in sites.iter().zip(&fields) {
            let (axes, distance, height) = frame(s);
            samples.outwards.push(axes[0]);
            samples.distances.push(distance);
            samples.heights.push(height);
            samples.readings.push(axes.map(|a| (a | f).s()));
        }
        let mean = fields.iter().map(|f| f.norm_squared()).sum::<f64>() / fields.len() as f64;
        samples.typical = super::beltrami::root(mean);
        samples
    }

    /// The unit direction of the smoothed, tapered field at a point, along `sense`.
    fn velocity(&self, p: Vector, sense: f64) -> Vector {
        let (axes, distance, height) = frame(p);
        let (mut sum, mut weights) = ([0.0; 3], 0.0);
        for k in 0..self.readings.len() {
            let chord = 2.0 - 2.0 * (axes[0] | self.outwards[k]).s();
            let across =
                (distance - self.distances[k]).powi(2) + (height - self.heights[k]).powi(2);
            let w = (-chord / (SMOOTHING_TURN * SMOOTHING_TURN)
                - across / (SMOOTHING_REACH * SMOOTHING_REACH))
                .exp();
            weights += w;
            for (s, r) in sum.iter_mut().zip(self.readings[k]) {
                *s += w * r;
            }
        }
        let field = (0..3)
            .map(|i| axes[i] * (sense * sum[i] / weights))
            .fold(Vector::zero(), |a, b| a + b);
        // Taper the motion out from the tube's core towards the wall, so that the lines glide
        // along the flux surfaces.
        let offset = p - axes[0] * TUBE_MAJOR;
        let reach = offset.norm();
        let tube_out = offset * (1.0 / reach);
        let share = (reach / TUBE_MINOR).clamp(0.0, 1.0);
        let out = (field | tube_out).s();
        let cosine = phasor(core::f64::consts::FRAC_PI_2 * share).e20();
        let tapered = out * cosine * cosine - WALL_PULL * share.powi(3) * self.typical;
        let field = field + tube_out * (tapered - out);
        let floor = SPEED_FLOOR * self.typical;
        // The field over its length softened by the floor: the norm of the pair (|field|, floor).
        field * gax::vga2d::Vector::new(field.norm(), floor).norm().recip()
    }

    /// The field lines from every seed, both ways, each from its far end back through the seed,
    /// and the shell each starts on.
    pub fn lines(&self) -> Vec<(Vec<Vector>, usize)> {
        let x = Vector::new(1.0, 0.0, 0.0);
        let mut out = Vec::new();
        for (shell, &count) in SEEDS.iter().enumerate() {
            for k in 0..count {
                let spacing = core::f64::consts::TAU * k as f64 / count as f64;
                let tube = rotor(
                    Bivector::new(0.0, -1.0, 0.0),
                    SEED_ANGLES_AROUND_TUBE[shell] + spacing,
                );
                let axis = rotor(
                    Bivector::new(0.0, 0.0, 1.0),
                    SEED_ANGLES_AROUND_AXIS[shell] + spacing,
                );
                let seed =
                    axis >> (x * TUBE_MAJOR + (tube >> x) * (SEED_SHARES[shell] * TUBE_MINOR));
                let ways = [-1.0, 1.0].map(|sense| self.trace(seed, sense));
                let line: Vec<Vector> = ways[0]
                    .iter()
                    .skip(1)
                    .rev()
                    .chain(&ways[1])
                    .copied()
                    .collect();
                out.push((line, shell));
            }
        }
        out
    }

    /// The points of one way from a seed, every `TRACE_STRIDE` steps.
    fn trace(&self, seed: Vector, sense: f64) -> Vec<Vector> {
        let mut p = seed;
        let mut kept = vec![p];
        for step in 1..=TRACE_STEPS {
            let k1 = self.velocity(p, sense);
            let k2 = self.velocity(p + k1 * (0.5 * TRACE_STEP), sense);
            let k3 = self.velocity(p + k2 * (0.5 * TRACE_STEP), sense);
            let k4 = self.velocity(p + k3 * TRACE_STEP, sense);
            p += (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (TRACE_STEP / 6.0);
            if step % TRACE_STRIDE == 0 {
                kept.push(p);
            }
        }
        kept
    }
}

/// numga's rotor `exp(plane · (−angle / 2))`.
fn rotor(plane: Bivector, angle: f64) -> gax::Unit<gax::vga3d::Rotor> {
    (plane * (-angle / 2.0)).exp()
}
