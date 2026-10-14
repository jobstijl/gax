//! The steps outside the algebra, where numga calls scipy (`linprog`, and `minimize` with SLSQP):
//! convex programs over a field of numbers, solved here by a logarithmic barrier. Each Newton
//! step is one sparse solve of gax field maps: the conditions' map `G`, its reverse (its
//! transpose, for scalar cells), and the barrier's weights on the diagonal between them.

use gax::field::{Field, SparseMap};
use gax_numga_examples::mesh::Scalar3;

/// A field of numbers.
pub type Numbers = Field<Scalar3>;

/// A convex program over a field of numbers `x`: least `½ x·Q x + c·x` where `G x ≥ h`, the
/// total `e·x` kept where it starts if `e` is given.
pub struct Program {
    pub curvature: SparseMap<Scalar3>,
    pub cost: Numbers,
    pub conditions: SparseMap<Scalar3>,
    pub bounds: Numbers,
    pub kept: Option<Numbers>,
}

/// The barrier's weight at the start, and the share of it each round keeps.
const START: f64 = 0.1;
const SHRINK: f64 = 0.2;
/// The bound on the gap to the least value at which the rounds stop: the barrier's weight times
/// the number of conditions.
const GAP: f64 = 1e-10;
/// Newton steps at most per round, the decrement at which a round is done, and the shortest
/// step the line search tries.
const NEWTON_STEPS: usize = 50;
const DECREMENT: f64 = 1e-14;
const SHORTEST: f64 = 1e-12;

impl Program {
    /// How far each condition is met: `G x - h`, positive inside.
    fn slack(&self, x: &Numbers) -> Numbers {
        &(&self.conditions * x) - &self.bounds
    }

    /// The objective with the barrier of weight `mu`, `½ x·Q x + c·x - μ Σ ln(G x - h)`, or
    /// `None` outside the conditions.
    fn barrier(&self, x: &Numbers, mu: f64) -> Option<f64> {
        let slack = self.slack(x);
        if slack.iter().any(|s| s.s() <= 0.0) {
            return None;
        }
        let logs: f64 = slack.iter().map(|s| s.s().ln()).sum();
        Some(0.5 * x.dot(&(&self.curvature * x)) + self.cost.dot(x) - mu * logs)
    }

    /// The Newton step of the barrier of weight `mu` at `x`, keeping the total if asked, and its
    /// decrement, the squared size of the step in the barrier's curvature.
    fn newton(&self, x: &Numbers, mu: f64) -> (Numbers, f64) {
        let inverse = self.slack(x).map(|s| s.inverse());
        let push = (&self.conditions.reverse() * &inverse).scaled(mu);
        let gradient = &(&(&self.curvature * x) + &self.cost) - &push;
        let weights = SparseMap::from_diagonal(&inverse.map(|i| i * i * mu));
        let hessian =
            &self.curvature + &(&(&self.conditions.reverse() * &weights) * &self.conditions);
        let lu = (&hessian * Scalar3::slot())
            .factor()
            .expect("the barrier's curvature is positive definite");
        let mut step = -lu.solve(&gradient);
        if let Some(kept) = &self.kept {
            // The step less the part of the curvature's direction to the total that changes it.
            let toward = lu.solve(kept);
            step = &step - &toward.scaled(kept.dot(&step) / kept.dot(&toward));
        }
        let decrement = -gradient.dot(&step);
        (step, decrement)
    }

    /// The first point along the step, halving it from its full length, that stays inside the
    /// conditions and does not raise the barrier above `value`.
    fn line_search(&self, x: &Numbers, step: &Numbers, mu: f64, value: f64) -> Option<Numbers> {
        let mut length = 1.0;
        while length > SHORTEST {
            let next = x + &step.scaled(length);
            if self.barrier(&next, mu).is_some_and(|v| v <= value) {
                return Some(next);
            }
            length *= 0.5;
        }
        None
    }

    /// The least value from a start strictly inside the conditions: rounds of Newton steps on
    /// the barrier, its weight shrinking each round, each step halved until it lowers the
    /// barrier.
    ///
    /// # Panics
    /// If the start is not strictly inside the conditions.
    pub fn solve(&self, start: Numbers) -> Numbers {
        let mut x = start;
        let mut mu = START;
        let count = self.bounds.len() as f64;
        while mu * count > GAP {
            for _ in 0..NEWTON_STEPS {
                let value = self.barrier(&x, mu).expect("a point inside the conditions");
                let (step, decrement) = self.newton(&x, mu);
                if decrement < DECREMENT {
                    break;
                }
                match self.line_search(&x, &step, mu, value) {
                    Some(next) => x = next,
                    // No step lowers the barrier: this round is as close as it gets.
                    None => break,
                }
            }
            mu *= SHRINK;
        }
        x
    }
}

/// A coupling of a field map of numbers: its row, its column and its cell.
pub type Coupling = (usize, usize, Scalar3);

/// The field map of the couplings.
pub fn collect(couplings: Vec<Coupling>, shape: (usize, usize)) -> SparseMap<Scalar3> {
    let rows = couplings.iter().map(|c| c.0).collect();
    let columns = couplings.iter().map(|c| c.1).collect();
    let cells = couplings.into_iter().map(|c| c.2).collect();
    SparseMap::new(cells, rows, columns, shape)
}

/// The bounds on every weight.
pub const WEIGHT_BOUNDS: (f64, f64) = (0.5, 2.0);

/// The conditions `G w ≥ h` that keep `n` weights within their bounds, `w ≥ low` and
/// `-w ≥ -high`, from row `first` on.
fn within_bounds(n: usize, first: usize) -> Vec<Coupling> {
    let low = (0..n).map(|v| (first + v, v, Scalar3::new(1.0)));
    let high = (0..n).map(|v| (first + n + v, v, Scalar3::new(-1.0)));
    low.chain(high).collect()
}

/// The bounds `h` of [`within_bounds`].
fn bound_values(n: usize) -> impl Iterator<Item = f64> {
    let (low, high) = WEIGHT_BOUNDS;
    core::iter::repeat_n(low, n).chain(core::iter::repeat_n(-high, n))
}

/// The weights that make the smallest condition `A w ≥ 0` as large as it can be, with the
/// weights within their bounds and summing to their count, and that smallest condition:
/// negative when no weights meet them all. A linear program over the weights and the smallest
/// condition `t`, the last number: least `-t` where `A w - t ≥ 0`.
pub fn widest(conditions: &SparseMap<Scalar3>) -> (Numbers, f64) {
    let (m, n) = conditions.shape();
    let mut couplings: Vec<Coupling> = conditions.couplings().collect();
    couplings.extend((0..m).map(|r| (r, n, Scalar3::new(-1.0))));
    couplings.extend(within_bounds(n, m));
    let bounds = core::iter::repeat_n(0.0, m).chain(bound_values(n));
    let program = Program {
        curvature: SparseMap::new(Vec::new(), Vec::new(), Vec::new(), (n + 1, n + 1)),
        cost: Field::from_fn(n + 1, |v| Scalar3::new(if v == n { -1.0 } else { 0.0 })),
        conditions: collect(couplings, (m + 2 * n, n + 1)),
        bounds: bounds.map(Scalar3::new).collect(),
        kept: Some(Field::from_fn(n + 1, |v| {
            Scalar3::new(if v < n { 1.0 } else { 0.0 })
        })),
    };
    // Even weights, and a smallest condition below every one of theirs.
    let even = Field::from_fn(n, |_| Scalar3::new(1.0));
    let lowest = (conditions * &even)
        .iter()
        .map(|c| c.s())
        .fold(f64::INFINITY, f64::min);
    let start = even.iter().chain([Scalar3::new(lowest - 1.0)]).collect();
    let solved = program.solve(start);
    let weights = Field::from_fn(n, |v| solved.site(v));
    (weights, solved.site(n).s())
}

/// The weights nearest the start, by the summed squares of the changes, that leave every
/// condition `A w ≥ margin` and keep the weights within their bounds, from weights strictly
/// inside them: a quadratic program, least `|w|² - 2 start·w`.
pub fn nearest(
    conditions: &SparseMap<Scalar3>,
    margin: f64,
    start: &Numbers,
    inside: Numbers,
) -> Numbers {
    let (m, n) = conditions.shape();
    let mut couplings: Vec<Coupling> = conditions.couplings().collect();
    couplings.extend(within_bounds(n, m));
    let bounds = core::iter::repeat_n(margin, m).chain(bound_values(n));
    let twice = Field::from_fn(n, |_| Scalar3::new(2.0));
    let program = Program {
        curvature: SparseMap::from_diagonal(&twice),
        cost: start.scaled(-2.0),
        conditions: collect(couplings, (m + 2 * n, n)),
        bounds: bounds.map(Scalar3::new).collect(),
        kept: None,
    };
    program.solve(inside)
}

/// Each condition scaled to unit size: every row of the map divided by its length.
pub fn unit_rows(conditions: &SparseMap<Scalar3>) -> SparseMap<Scalar3> {
    let (rows, columns) = conditions.shape();
    let mut dense = vec![vec![0.0; columns]; rows];
    for (r, c, cell) in conditions.couplings() {
        dense[r][c] += cell.s();
    }
    #[allow(clippy::disallowed_methods)] // a row's length over the weights, not a geometric one
    let lengths: Field<Scalar3> = dense
        .iter()
        .map(|row| Scalar3::new(1.0 / row.iter().map(|x| x * x).sum::<f64>().sqrt()))
        .collect();
    &SparseMap::from_diagonal(&lengths) * conditions
}
