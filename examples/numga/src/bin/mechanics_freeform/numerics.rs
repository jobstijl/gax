//! What numga takes from scipy, written here: the minimizer of the implicit time step (numga
//! calls scipy's L-BFGS-B in coordinates whitened by the control-space mass; here a limited-
//! memory BFGS whose first guess of the inverse curvature is the mass's inverse, the same
//! method in the original coordinates), and the low-discrepancy sequence of the volume samples
//! (numga's scrambled Sobol; here a Halton sequence with a seeded random shift).

use gax::field::{Field, SparseMap};
use gax_numga_examples::rng::{Draw, rng};

use super::freeform::{Antivector, Mass, Vector};

/// The pairing of a field of forces with a field of displacements: the work, summed.
pub fn work(force: &Field<Vector>, displacement: &Field<Antivector>) -> f64 {
    force
        .iter()
        .zip(displacement.iter())
        .map(|(f, d)| (f & d).s())
        .sum()
}

/// The pairs of the memory: a step and the change of the force along it, and their pairing.
struct Pair {
    step: Field<Antivector>,
    change: Field<Vector>,
    curvature: f64,
}

/// The pairs remembered, as scipy's default.
const MEMORY: usize = 10;
/// The steps of the backtracking line search, at most, and its sufficient decrease.
const HALVINGS: usize = 40;
const DECREASE: f64 = 1e-4;

/// The state that minimizes `objective` (energy and forces, the forces its gradient), from
/// `initial`, at most `iterations` iterations: limited-memory BFGS whose first guess of the
/// inverse curvature is the inverse of the positive-definite `metric`, from forces back to
/// displacements, with a backtracking line search. It stops when the energy falls by less than
/// `tolerance` relative to its size (scipy's `ftol`), or when the force measured by the
/// metric's inverse is below `tolerance` (scipy's `gtol` in the whitened coordinates).
pub fn minimize_state(
    initial: &Field<Antivector>,
    objective: impl Fn(&Field<Antivector>) -> (f64, Field<Vector>),
    metric: &SparseMap<Mass>,
    iterations: usize,
    tolerance: f64,
) -> Field<Antivector> {
    // The metric as a map of displacements to themselves, through the dual that takes a force
    // to the point it pairs with, factored once.
    let to_points =
        SparseMap::from_diagonal(&(0..initial.len()).map(|_| Vector::slot().hodge()).collect());
    let lu = to_points
        .compose(metric)
        .factor()
        .expect("a positive-definite metric");
    let inverse =
        |force: &Field<Vector>| -> Field<Antivector> { lu.solve(&force.map(|f| f.hodge())) };

    let mut state = initial.clone();
    let (mut energy, mut force) = objective(&state);
    let mut memory: Vec<Pair> = Vec::new();
    for _ in 0..iterations {
        let whitened = inverse(&force);
        if work(&force, &whitened).max(0.0) <= tolerance * tolerance {
            break;
        }
        // The two-loop recursion: the remembered pairs correct the metric's guess.
        let mut q = force.clone();
        let mut alphas = Vec::with_capacity(memory.len());
        for pair in memory.iter().rev() {
            let alpha = work(&q, &pair.step) / pair.curvature;
            q = &q - &pair.change.scaled(alpha);
            alphas.push(alpha);
        }
        let scale = memory.last().map_or(1.0, |pair| {
            pair.curvature / work(&pair.change, &inverse(&pair.change))
        });
        let mut r = inverse(&q).scaled(scale);
        for (pair, alpha) in memory.iter().zip(alphas.iter().rev()) {
            let beta = work(&pair.change, &r) / pair.curvature;
            r = &r + &pair.step.scaled(alpha - beta);
        }
        let direction = -r;
        let slope = work(&force, &direction);
        if slope >= 0.0 {
            // Not downhill: forget the pairs and start over from the metric's guess.
            memory.clear();
            continue;
        }
        let mut length = 1.0;
        let mut accepted = None;
        for _ in 0..HALVINGS {
            let trial = &state + &direction.scaled(length);
            let (e, f) = objective(&trial);
            if e.is_finite() && e <= energy + DECREASE * length * slope {
                accepted = Some((trial, e, f));
                break;
            }
            length /= 2.0;
        }
        let Some((trial, e, f)) = accepted else {
            break;
        };
        let step = &trial - &state;
        let change = &f - &force;
        let curvature = work(&change, &step);
        let fall = (energy - e) / energy.abs().max(e.abs()).max(1.0);
        (state, energy, force) = (trial, e, f);
        if curvature > 0.0 {
            if memory.len() == MEMORY {
                memory.remove(0);
            }
            memory.push(Pair {
                step,
                change,
                curvature,
            });
        }
        if fall <= tolerance {
            break;
        }
    }
    state
}

/// The radical inverse of `index` in `base`: its digits mirrored about the point.
fn radical_inverse(mut index: u64, base: u64) -> f64 {
    let mut value = 0.0;
    let mut unit = 1.0 / base as f64;
    while index > 0 {
        value += (index % base) as f64 * unit;
        index /= base;
        unit /= base as f64;
    }
    value
}

/// A low-discrepancy sequence filling the unit cube: Halton's in bases 2, 3 and 5, shifted by
/// a random offset (modulo one) drawn from `seed`.
pub struct Halton {
    next: u64,
    shift: [f64; 3],
}

impl Halton {
    pub fn new(seed: u64) -> Halton {
        let mut r = rng(seed);
        Halton {
            next: 1,
            shift: [r.uniform(), r.uniform(), r.uniform()],
        }
    }

    /// The next `count` points.
    pub fn take(&mut self, count: usize) -> Vec<[f64; 3]> {
        (0..count)
            .map(|_| {
                let i = self.next;
                self.next += 1;
                let bases = [2, 3, 5];
                core::array::from_fn(|k| (radical_inverse(i, bases[k]) + self.shift[k]).fract())
            })
            .collect()
    }
}
