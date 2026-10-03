//! numga's `geometry/odometry`: the most likely poses of a robot's lap, and how uncertain each
//! one is (pose-graph SLAM). The robot reads, at every stop, how far it moved since the last one;
//! back at the start it recognises it and reads one more relative pose, closing the loop.
//!
//! A small error of a pose is a twist on its right, `pose * (twist / 2).exp()`, and its
//! uncertainty a covariance, `Covariance = Twist <- Forque`, whose inverse, the information, is a
//! quadric on twists. A reading measures the twist at its head minus the twist at its tail
//! carried there, `relative << twist`. The mismatches pull on the poses (the gradient), the pull
//! of what a correction measures is the curvature, applied but never assembled; conjugate
//! gradients solve it and Gauss-Newton moves the poses. Each pose's own uncertainty is the solve
//! of the pull of every unit error a reading or an anchor allows.
//!
//! The core is written once, as a macro instantiated for PGA2D and for PGA3D (numga instantiates
//! one module per algebra). The animation closes the lap a fifth of the way at a time: in the
//! plane on the left with each pose's 2σ ellipse, and a short lap in space on the right.

use gax_numga_examples::canvas::mix;
use gax_numga_examples::rng::{Draw, rng};
use gax_numga_examples::{
    Align, Anim, Axes, Camera, Canvas, Lens, Marker, Scene3, backdrop, caption, f32s, palette,
    plot, run,
};
use std::sync::OnceLock;

/// How far a step reading and the closing reading are trusted, how well the first pose is known,
/// and how vaguely the others are known before any reading, as translation and rotation standard
/// deviations.
const READING: (f64, f64) = (0.03, 0.01);
const CLOSING: (f64, f64) = (0.001, 0.0003);
const KNOWN: (f64, f64) = (0.01, 0.01);
const VAGUE: (f64, f64) = (100.0, 30.0);

/// The odometry core over one algebra: `gax::$ga`'s `Motor`, `Twist` and `Forque` (numga's
/// `Line`, the antibivector that reads a twist). Everything else is the same text for the plane
/// and for space.
macro_rules! odometry_core {
    ($ga:ident) => {
        use super::{CLOSING, Draw, KNOWN, READING, VAGUE, rng};
        use gax::$ga::{Forque, Motor, Twist};
        use gax::{ApproxEq, Kind, Unit};

        pub type M = Unit<Motor<(), f64>>;
        pub type Tw = Twist<(), f64>;
        pub type Fq = Forque<(), f64>;
        /// A twist's uncertainty: `Twist <- Forque`.
        pub type Covariance = Twist<(Forque,), f64>;
        /// Its inverse, a quadric on twists: `Forque <- Twist`.
        pub type Information = Forque<(Twist,), f64>;
        /// The number of a twist's coefficients: 3 in the plane, 6 in space.
        pub const SIZE: usize = <Twist as Kind>::N;

        /// The open twist and the open forque.
        pub fn twists() -> Twist<(Twist,), f64> {
            Twist::<(), f64>::slot()
        }
        pub fn forques() -> Forque<(Forque,), f64> {
            Forque::<(), f64>::slot()
        }

        /// The basis twist `i`.
        pub fn basis(i: usize) -> Tw {
            Tw::from_coeffs(<Twist as Kind>::arr_from_fn(
                |k| if k == i { 1.0 } else { 0.0 },
            ))
        }

        /// The identity motor.
        pub fn identity() -> M {
            Tw::zero().exp()
        }

        /// The pose after a small error on its right.
        pub fn nudged(pose: M, twist: Tw) -> M {
            pose * twist.gp(0.5).exp()
        }

        // --- problem ------------------------------------------------------------------------

        /// How far a relative motor is from what was read: a twist on the right of the reading.
        pub fn mismatch(reading: M, relative: M) -> Tw {
            let r: Tw = (reading.inverse() * relative).log();
            r.gp(2.0)
        }

        /// The covariance of a reading with isotropic translation noise and rotation noise
        /// about its head: a basis twist turns if its blade has no `e0`, and slides if it has.
        pub fn isotropic(translation_std: f64, rotation_std: f64) -> Covariance {
            (0..SIZE)
                .map(|i| {
                    let turning = !<Twist as Kind>::BLADES[i].contains('0');
                    let std = if turning {
                        rotation_std
                    } else {
                        translation_std
                    };
                    basis(i) * (basis(i) & forques()) * (std * std)
                })
                .fold(Twist::zero(), |a, b| a + b)
        }

        /// Twists whose outer products sum to an uncertainty: its images of lines orthonormal
        /// in its own form.
        pub fn modes(uncertainty: Covariance) -> Vec<Tw> {
            let spread = forques() & uncertainty;
            let (_, lines) = spread.eigh_with(spread);
            lines.as_ref().iter().map(|l| uncertainty.of(*l)).collect()
        }

        /// The readings and their covariances and the poses they link (a reading reads its
        /// head from its tail), and every pose's anchor and prior.
        #[derive(Clone)]
        pub struct Problem {
            pub readings: Vec<M>,
            pub noises: Vec<Covariance>,
            pub weights: Vec<Information>,
            pub tails: Vec<usize>,
            pub heads: Vec<usize>,
            pub anchors: Vec<M>,
            pub priors: Vec<Covariance>,
            pub anchor_weights: Vec<Information>,
        }

        impl Problem {
            /// The same problem with only the first `count` readings (the tests use it in
            /// space only).
            #[allow(dead_code)]
            pub fn first(&self, count: usize) -> Problem {
                Problem {
                    readings: self.readings[..count].to_vec(),
                    noises: self.noises[..count].to_vec(),
                    weights: self.weights[..count].to_vec(),
                    tails: self.tails[..count].to_vec(),
                    heads: self.heads[..count].to_vec(),
                    ..self.clone()
                }
            }

            /// Each reading's head, from its tail.
            pub fn relative(&self, poses: &[M]) -> Vec<M> {
                self.tails
                    .iter()
                    .zip(&self.heads)
                    .map(|(t, h)| poses[*t].inverse() * poses[*h])
                    .collect()
            }

            /// What mismatches at the readings and at the anchors pull on every pose: each
            /// weighted by the inverse of its covariance, a reading's at its head, and carried
            /// to its tail with the opposite sign; an anchor's at its pose.
            pub fn pull(&self, relative: &[M], at_readings: &[Tw], at_anchors: &[Tw]) -> Vec<Fq> {
                let mut pulled: Vec<Fq> = self
                    .anchor_weights
                    .iter()
                    .zip(at_anchors)
                    .map(|(w, a)| w.of(*a))
                    .collect();
                for k in 0..self.readings.len() {
                    let weighted = self.weights[k].of(at_readings[k]);
                    pulled[self.heads[k]] += weighted;
                    pulled[self.tails[k]] -= relative[k] >> weighted;
                }
                pulled
            }

            /// The gradient of half the squared mismatches of every reading and of every pose
            /// from its anchor, each measured by its covariance: their pull.
            pub fn gradient(&self, poses: &[M]) -> Vec<Fq> {
                let relative = self.relative(poses);
                let at_readings: Vec<Tw> = self
                    .readings
                    .iter()
                    .zip(&relative)
                    .map(|(r, q)| mismatch(*r, *q))
                    .collect();
                let at_anchors: Vec<Tw> = self
                    .anchors
                    .iter()
                    .zip(poses)
                    .map(|(a, p)| mismatch(*a, *p))
                    .collect();
                self.pull(&relative, &at_readings, &at_anchors)
            }

            /// The information applied to a correction of every pose: the pull of what the
            /// readings and the anchors measure of it.
            pub fn curvature(&self, relative: &[M], twists: &[Tw]) -> Vec<Fq> {
                let measured: Vec<Tw> = (0..self.readings.len())
                    .map(|k| twists[self.heads[k]] - (relative[k] << twists[self.tails[k]]))
                    .collect();
                self.pull(relative, &measured, twists)
            }

            /// How sharply the objective curves as each pose moves alone: its anchor's weight,
            /// every reading's at its head, and the weight carried back to its tail.
            pub fn curvature_alone(&self, relative: &[M]) -> Vec<Information> {
                let mut alone = self.anchor_weights.clone();
                for k in 0..self.readings.len() {
                    let w = self.weights[k];
                    alone[self.heads[k]] += w;
                    alone[self.tails[k]] += relative[k] >> w.of(relative[k] << twists());
                }
                alone
            }

            /// The correction the information sends to the given pull, by conjugate gradients
            /// preconditioned by each pose's curvature alone; exact after as many steps as
            /// there are unknowns. Stops early only if the residual vanishes exactly.
            pub fn conjugate_gradients(&self, relative: &[M], right: &[Fq]) -> Vec<Tw> {
                let alone = self.curvature_alone(relative);
                let pair = |a: &[Fq], b: &[Tw]| -> f64 {
                    a.iter().zip(b).map(|(f, t)| (*f & *t).s()).sum()
                };
                let solve = |r: &[Fq]| -> Vec<Tw> {
                    alone.iter().zip(r).map(|(a, f)| a.solve(*f)).collect()
                };
                let mut correction = vec![Tw::zero(); right.len()];
                let mut residual = right.to_vec();
                let mut direction = solve(&residual);
                let mut aligned = pair(&residual, &direction);
                for _ in 0..SIZE * right.len() {
                    let pushed = self.curvature(relative, &direction);
                    let curv = pair(&pushed, &direction);
                    if aligned == 0.0 || curv == 0.0 {
                        break;
                    }
                    let step = aligned / curv;
                    for (c, d) in correction.iter_mut().zip(&direction) {
                        *c += d.gp(step);
                    }
                    for (r, p) in residual.iter_mut().zip(&pushed) {
                        *r -= p.gp(step);
                    }
                    let preconditioned = solve(&residual);
                    let realigned = pair(&residual, &preconditioned);
                    let beta = realigned / aligned;
                    for (d, p) in direction.iter_mut().zip(&preconditioned) {
                        *d = *p + d.gp(beta);
                    }
                    aligned = realigned;
                }
                correction
            }

            /// One damped Gauss-Newton step: solve the information at the poses for the
            /// correction the gradient asks for, and move by the damping's share of it.
            pub fn gauss_newton_step(&self, poses: &[M], damping: f64) -> Vec<M> {
                let right: Vec<Fq> = self.gradient(poses).into_iter().map(|g| -g).collect();
                let correction = self.conjugate_gradients(&self.relative(poses), &right);
                poses
                    .iter()
                    .zip(&correction)
                    .map(|(p, c)| nudged(*p, c.gp(damping)))
                    .collect()
            }

            /// Each pose's uncertainty under dead reckoning, along the readings from each pose
            /// to the next: the first pose's prior and each reading's noise entering at the
            /// pose it reaches, summed in the world frame, and read at every pose.
            pub fn reckon(&self, poses: &[M]) -> Vec<Covariance> {
                let mut summed = Twist::zero();
                let mut out = Vec::new();
                for (i, p) in poses.iter().enumerate() {
                    let entering = if i == 0 {
                        self.priors[0]
                    } else {
                        self.noises[i - 1]
                    };
                    summed += *p >> entering.of(*p << forques());
                    out.push(*p << summed.of(*p >> forques()));
                }
                out
            }

            /// Each pose's own uncertainty: the correction the pull of every unit error a
            /// reading or an anchor allows asks for, the outer products summed at each pose.
            /// The unit errors are solved for independently, on all cores.
            pub fn marginals(&self, poses: &[M]) -> Vec<Covariance> {
                let relative = self.relative(poses);
                let (readings, count) = (self.readings.len(), poses.len());
                let sites: Vec<(usize, Tw)> = self
                    .noises
                    .iter()
                    .chain(&self.priors)
                    .enumerate()
                    .flat_map(|(s, c)| modes(*c).into_iter().map(move |m| (s, m)))
                    .collect();
                let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
                let chunk = sites.len().div_ceil(threads).max(1);
                let partial: Vec<Vec<Covariance>> = std::thread::scope(|scope| {
                    let handles: Vec<_> = sites
                        .chunks(chunk)
                        .map(|part| {
                            let relative = &relative;
                            scope.spawn(move || {
                                let mut sum = vec![Twist::zero(); count];
                                for (site, mode) in part {
                                    let mut at_readings = vec![Tw::zero(); readings];
                                    let mut at_anchors = vec![Tw::zero(); count];
                                    if *site < readings {
                                        at_readings[*site] = *mode;
                                    } else {
                                        at_anchors[*site - readings] = *mode;
                                    }
                                    let right = self.pull(relative, &at_readings, &at_anchors);
                                    let errors = self.conjugate_gradients(relative, &right);
                                    for (s, e) in sum.iter_mut().zip(&errors) {
                                        *s += *e * (*e & forques());
                                    }
                                }
                                sum
                            })
                        })
                        .collect();
                    handles
                        .into_iter()
                        .map(|h| h.join().expect("a solve"))
                        .collect()
                });
                let mut total = vec![Twist::zero(); count];
                for part in partial {
                    for (t, p) in total.iter_mut().zip(part) {
                        *t += p;
                    }
                }
                total
            }
        }

        /// A lap along the true steps: the true and dead-reckoned poses, and the problem. Every
        /// step is read, then the last pose from the first; the first pose is known, the others
        /// only vaguely, near where dead reckoning puts them.
        pub fn survey(steps: &[M], seed: u64) -> (Vec<M>, Vec<M>, Problem) {
            let n = steps.len();
            let mut rng = rng(seed);
            // Each step composes on the right, in the robot's own frame.
            let mut truth = vec![identity()];
            for s in steps {
                let last = *truth.last().expect("a pose");
                truth.push(last * *s);
            }
            let tails: Vec<usize> = (0..n).chain([0]).collect();
            let heads: Vec<usize> = (1..=n).chain([n]).collect();
            let mut noises = vec![isotropic(READING.0, READING.1); n];
            noises.push(isotropic(CLOSING.0, CLOSING.1));
            let readings: Vec<M> = (0..=n)
                .map(|k| {
                    let error = modes(noises[k])
                        .into_iter()
                        .fold(Tw::zero(), |a, m| a + m.gp(rng.normal()));
                    nudged(truth[tails[k]].inverse() * truth[heads[k]], error)
                })
                .collect();
            let mut dead = vec![truth[0]];
            for k in 0..n {
                dead.push(dead[k] * readings[k]);
            }
            let anchors: Vec<M> = [truth[0]]
                .into_iter()
                .chain(dead[1..].iter().copied())
                .collect();
            let mut priors = vec![isotropic(KNOWN.0, KNOWN.1)];
            priors.extend(vec![isotropic(VAGUE.0, VAGUE.1); n]);
            let problem = Problem {
                weights: noises.iter().map(|c| c.inverse()).collect(),
                anchor_weights: priors.iter().map(|c| c.inverse()).collect(),
                readings,
                noises,
                tails,
                heads,
                anchors,
                priors,
            };
            (truth, dead, problem)
        }

        /// A run of damped Gauss-Newton from dead reckoning with each pose's uncertainty along
        /// it: the poses of each step keep the rest of their error, so the rest's square of
        /// their uncertainty beyond what every reading allows there.
        pub struct Run {
            pub truth: Vec<M>,
            pub dead: Vec<M>,
            /// Dead reckoning's uncertainties (drawn for the plane).
            #[allow(dead_code)]
            pub reckoned: Vec<Covariance>,
            pub problem: Problem,
            /// The poses and their uncertainties: dead reckoning's, then after each step.
            pub iterates: Vec<(Vec<M>, Vec<Covariance>)>,
        }

        pub fn damped(steps: &[M], damping: f64, iterations: usize, seed: u64) -> Run {
            let (truth, dead, problem) = survey(steps, seed);
            let reckoned = problem.reckon(&dead);
            let mut iterates = vec![(dead.clone(), reckoned.clone())];
            for _ in 0..iterations {
                let (poses, uncertainty) = iterates.last().expect("a state");
                let poses = problem.gauss_newton_step(poses, damping);
                let posterior = problem.marginals(&poses);
                let keep = (1.0 - damping) * (1.0 - damping);
                let uncertainty = posterior
                    .iter()
                    .zip(uncertainty)
                    .map(|(p, u)| *p + (*u - *p).gp(keep))
                    .collect();
                iterates.push((poses, uncertainty));
            }
            Run {
                truth,
                dead,
                reckoned,
                problem,
                iterates,
            }
        }

        /// The largest coefficient of a list of forques.
        pub fn max_abs(v: &[Fq]) -> f64 {
            v.iter()
                .map(|f| f.max_abs_diff(&Fq::zero()))
                .fold(0.0, f64::max)
        }

        /// The most likely poses given every reading, by full Gauss-Newton steps, and the
        /// scenario's check: the gradient falls a millionfold from dead reckoning.
        #[cfg_attr(not(test), allow(dead_code))]
        pub fn closing(steps: &[M], iterations: usize, seed: u64) -> Run {
            let run = damped(steps, 1.0, iterations, seed);
            let at_dead = max_abs(&run.problem.gradient(&run.dead));
            let (poses, _) = run.iterates.last().expect("a state");
            let at_optimum = max_abs(&run.problem.gradient(poses));
            assert!(at_optimum <= 1e-6 * at_dead, "{at_optimum} vs {at_dead}");
            run
        }
    };
}

/// The lap in the plane, and its drawing.
mod plane {
    odometry_core!(pga2d);
    use gax::pga2d::{Line, Point};

    /// A point of the plane.
    pub type P = Point<(), f64>;

    /// The steps of a lap in the plane.
    pub const STEPS: usize = 36;

    /// The point each pose carries: the origin.
    pub fn origin() -> P {
        Point::xy(0.0, 0.0)
    }

    /// Where a pose carries the origin.
    pub fn position(pose: M) -> P {
        pose >> origin()
    }

    /// A lap of 36 steps of 0.7 m, turning counterclockwise faster and slower twice a lap, and
    /// slipping sideways a little, one way and back twice a lap, which leaves the lap closed.
    /// Each step is a screw: the exponential of its twist, the sum of a translation and a
    /// rotation twist (numga's lap, mirrored to turn counterclockwise).
    pub fn lap() -> Vec<M> {
        (0..STEPS)
            .map(|k| {
                let phase = core::f64::consts::TAU * k as f64 / STEPS as f64;
                let turn =
                    (1.0 + 0.4 * (2.0 * phase).sin()) * core::f64::consts::TAU / STEPS as f64;
                let slip = -0.1 * (2.0 * phase).cos();
                (Point::translation_twist(0.7, slip) + Point::rotation_twist(origin(), turn)).exp()
            })
            .collect()
    }

    /// How far out the ellipses lie, in standard deviations.
    pub const SIGMAS: f64 = 2.0;
    /// A quadric on points, as the polar map `Line <- Point`.
    pub type Quadric = Line<(Point,), f64>;

    /// The quadric `SIGMAS` standard deviations out within which a pose carries the origin. A
    /// twist moves the carried point by its commutator with it; reading that motion with a line
    /// is reading the twist with another line, solved from the incidence pairing. So the
    /// twist's uncertainty gives the point's; its second moment's inverse, paired twice with a
    /// point of unit weight, is one plus the squared number of standard deviations to it.
    pub fn ellipse(pose: M, uncertainty: Covariance) -> Quadric {
        let here = position(pose);
        let shift = twists().commutator(here).of(pose >> twists());
        let readout = (Line::slot() & Point::slot()).solve(Line::slot() & shift);
        let moment = here * (Line::slot() & here) + shift.of(uncertainty.of(readout));
        let w = Line::new(0.0, 0.0, 1.0);
        moment.inverse() - (w * (w & Point::slot())).gp(1.0 + SIGMAS * SIGMAS)
    }

    /// The zero level of a quadric as a ring of points: along rays from its centre `here` (unit
    /// directions, turned round) the quadric is a quadratic in the distance, whose positive root
    /// lies on the ellipse.
    pub fn ring(quadric: Quadric, here: P, n: usize) -> Vec<P> {
        let value = |a: P, b: P| (quadric.of(a) & b).s();
        let h = here.unitized();
        let c = value(h, h);
        (0..=n)
            .map(|i| {
                let angle = core::f64::consts::TAU * i as f64 / n as f64;
                let d = Motor::rotation(origin(), angle) >> Point::direction(1.0, 0.0);
                let (a, b) = (value(d, d), value(h, d) + value(d, h));
                let s = (-b + (b * b - 4.0 * a * c).max(0.0).sqrt()) / (2.0 * a);
                h + d.gp(s)
            })
            .collect()
    }
}

/// The short lap in space.
mod space {
    odometry_core!(pga3d);
    use gax::pga3d::{Line, Point};

    /// The steps of the lap in space.
    pub const STEPS: usize = 6;

    /// Where a pose carries the origin.
    pub fn position(pose: M) -> Point<(), f64> {
        pose >> Point::xyz(0.0, 0.0, 0.0)
    }

    /// A turn of six steps of 0.7 m, pitching and rolling once a lap, and slipping sideways and
    /// rising and falling twice a lap, which leaves the lap closed to 4 mm. Each step is the screw of a
    /// translation twist plus a rotation twist about the axis of its angular velocity (numga's
    /// lap, mirrored to turn counterclockwise).
    pub fn lap() -> Vec<M> {
        let origin = Point::xyz(0.0, 0.0, 0.0);
        (0..STEPS)
            .map(|k| {
                let phase = core::f64::consts::TAU * k as f64 / STEPS as f64;
                let turn = core::f64::consts::TAU / STEPS as f64;
                let spin = Point::direction(0.1 * phase.sin(), -0.1 * phase.cos(), turn);
                let velocity = Line::translation_twist(
                    0.7,
                    -0.1 * (2.0 * phase).cos(),
                    0.05 * (2.0 * phase).sin(),
                );
                (velocity + Line::rotation_twist(origin & spin, spin.ideal_norm())).exp()
            })
            .collect()
    }
}

/// Seconds per Gauss-Newton step in the animation, the steps, and the pause at the end.
const PER_STEP: f32 = 0.45;
const ITERATIONS: usize = 20;
const HOLD: f32 = 2.5;

/// Both laps, closed a fifth of the way at a time.
fn runs() -> &'static (plane::Run, space::Run) {
    static RUNS: OnceLock<(plane::Run, space::Run)> = OnceLock::new();
    RUNS.get_or_init(|| {
        (
            plane::damped(&plane::lap(), 0.2, ITERATIONS, 9),
            space::damped(&space::lap(), 0.2, ITERATIONS, 9),
        )
    })
}

/// The state between two iterates: poses interpolated along the motor between them, and
/// uncertainties linearly.
fn between<P: Copy, C: Copy + core::ops::Add<Output = C> + core::ops::Sub<Output = C>>(
    iterates: &[(Vec<P>, Vec<C>)],
    t: f32,
    interpolate: impl Fn(P, P, f64) -> P,
    scale: impl Fn(C, f64) -> C,
) -> (Vec<P>, Vec<C>, usize) {
    let s = (t / PER_STEP).max(0.0);
    let k = (s as usize).min(iterates.len() - 1);
    let f = if k + 1 < iterates.len() {
        f64::from(s - k as f32)
    } else {
        0.0
    };
    let (a, b) = (&iterates[k], &iterates[(k + 1).min(iterates.len() - 1)]);
    // Ease in and out within a step.
    let f = f * f * (3.0 - 2.0 * f);
    let poses =
        a.0.iter()
            .zip(&b.0)
            .map(|(p, q)| interpolate(*p, *q, f))
            .collect();
    let unc =
        a.1.iter()
            .zip(&b.1)
            .map(|(u, v)| *u + scale(*v - *u, f))
            .collect();
    (poses, unc, k)
}

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let (w, h) = (c.width as f32, c.height as f32);
    let (lap, space_lap) = runs();
    let (poses, uncertainty, k) =
        between(&lap.iterates, t, gax::pga2d::Motor::interpolate, |c, f| {
            c.gp(f)
        });
    // The plane, left: truth, dead reckoning with its ellipses, and the poses with theirs.
    let positions =
        |ps: &[plane::M]| -> Vec<plane::P> { ps.iter().map(|p| plane::position(*p)).collect() };
    let (truth, dead) = (positions(&lap.truth), positions(&lap.dead));
    let (lo, hi) = truth.iter().chain(&dead).map(|p| p.to_euclidean()).fold(
        ([f64::MAX; 2], [f64::MIN; 2]),
        |(l, u), p| {
            (
                [l[0].min(p[0]), l[1].min(p[1])],
                [u[0].max(p[0]), u[1].max(p[1])],
            )
        },
    );
    let left = plot::inset([0.0, 50.0, w * 0.6, h], 10.0, 10.0, 10.0, 10.0);
    let aspect = f64::from((left[3] - left[1]) / (left[2] - left[0]));
    let half = 0.5 * (hi[1] - lo[1]).max((hi[0] - lo[0]) * aspect);
    let ax = Axes::equal(
        left,
        f32s([0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1])]),
        half as f32 * 1.15,
    );
    ax.polyline(c, &truth, 4.0, palette::grid(), 1.0);
    ax.dashed(c, &dead, 1.5, 5.0, palette::orange(), 0.9);
    for ((p, u), at) in lap.dead.iter().zip(&lap.reckoned).zip(&dead) {
        let ring = plane::ring(plane::ellipse(*p, *u), *at, 64);
        ax.polyline(c, &ring, 1.0, palette::orange(), 0.35);
    }
    let likely = positions(&poses);
    for ((p, u), at) in poses.iter().zip(&uncertainty).zip(&likely) {
        let ring = plane::ring(plane::ellipse(*p, *u), *at, 64);
        ax.polyline(c, &ring, 1.2, palette::sky(), 0.85);
    }
    ax.polyline(c, &likely, 1.4, palette::sky(), 1.0);
    ax.scatter(c, &likely, Marker::Dot, 4.0, palette::sky(), 1.0);
    ax.scatter(c, &truth[..1], Marker::Dot, 9.0, palette::ink(), 1.0);
    ax.legend(
        c,
        &[
            ("TRUTH", palette::grid()),
            ("DEAD RECKONING", palette::orange()),
            ("MOST LIKELY", palette::sky()),
        ],
    );
    // Space, right: the short lap seen from an orbiting camera.
    let (sposes, _, _) = between(
        &space_lap.iterates,
        t,
        gax::pga3d::Motor::interpolate,
        |c, f| c.gp(f),
    );
    let (x0, y0) = ((w * 0.6) as usize, 60usize);
    let (pw, ph) = (c.width - x0, c.height - y0);
    let mut sub = Canvas::new(pw, ph);
    sub.backdrop(
        mix(palette::top(), palette::bottom(), y0 as f32 / h),
        palette::bottom(),
    );
    let positions = |ps: &[space::M]| -> Vec<gax::pga3d::Point<(), f64>> {
        ps.iter().map(|p| space::position(*p)).collect()
    };
    let truth = positions(&space_lap.truth);
    // The camera circles the centre of the true lap: the sum of its points, unitized.
    let centre = truth
        .iter()
        .fold(gax::pga3d::Point::zero(), |a, p| a + *p)
        .unitized();
    let cam = Camera::orbit(
        pw,
        ph,
        centre,
        2.6,
        -1.2 + 0.25 * t,
        0.55,
        Lens::Perspective(0.8),
    );
    let mut scene = Scene3::new(cam);
    scene.polyline(&truth, 3.5, palette::grid(), 1.0);
    scene.polyline(&positions(&space_lap.dead), 1.4, palette::orange(), 0.9);
    let spath = positions(&sposes);
    scene.polyline(&spath, 1.6, palette::sky(), 1.0);
    for (p, q) in sposes.iter().zip(&spath) {
        scene.dot(*q, Marker::Dot, 5.0, palette::sky());
        // Each pose's heading, a short arrow along its local x.
        let ahead = *p >> gax::pga3d::Point::xyz(0.25, 0.0, 0.0);
        scene.seg(*q, ahead, 1.4, palette::yellow(), 0.9);
    }
    scene.dot(truth[0], Marker::Dot, 8.0, palette::ink());
    scene.draw(&mut sub);
    sub.text(
        "A SHORT LAP IN SPACE (PGA3D)",
        8.0,
        16.0,
        11.0,
        palette::ink(),
        Align::Left,
    );
    c.blit(&sub, x0, y0);
    caption(
        c,
        "ODOMETRY: CLOSING THE LAP",
        "POSE-GRAPH GAUSS-NEWTON BY CONJUGATE GRADIENTS, 2 SIGMA ELLIPSES",
    );
    // The gradient at the iterate the animation is leaving.
    let gradient = plane::max_abs(&lap.problem.gradient(&lap.iterates[k].0));
    c.text(
        &format!("GRADIENT {gradient:.1e}"),
        w * 0.6 - 12.0,
        h - 32.0,
        12.0,
        palette::grid(),
        Align::Right,
    );
    c.text(
        &format!("STEP {} / {ITERATIONS}, DAMPED TO 1/5", k.min(ITERATIONS)),
        w * 0.6 - 12.0,
        h - 14.0,
        12.0,
        palette::ink(),
        Align::Right,
    );
}

fn main() {
    let seconds = PER_STEP * ITERATIONS as f32 + HOLD;
    run(Anim::new("odometry", seconds).size(960, 540), draw);
}

#[cfg(test)]
mod tests {
    use super::{plane, space};
    use gax::Extensor;

    /// Each pose's block of the inverse of the objective's curvature in the twists of every
    /// pose (42 unknowns for the lap in space), by central differences of its gradient.
    fn inverse_curvature(problem: &space::Problem, poses: &[space::M]) -> Vec<[[f64; 6]; 6]> {
        const N: usize = 42;
        let count = poses.len();
        assert_eq!(count * space::SIZE, N);
        let mut h = [[0.0f64; N]; N];
        for pose in 0..count {
            for coefficient in 0..space::SIZE {
                let nudge = space::basis(coefficient).gp(1e-6);
                let moved = |sign: f64| -> Vec<f64> {
                    let mut ps = poses.to_vec();
                    ps[pose] = space::nudged(ps[pose], nudge.gp(sign));
                    problem
                        .gradient(&ps)
                        .iter()
                        .flat_map(|g| g.coeffs().to_vec())
                        .collect()
                };
                let (up, down) = (moved(1.0), moved(-1.0));
                for (row, (u, d)) in up.iter().zip(&down).enumerate() {
                    h[row][pose * space::SIZE + coefficient] = (u - d) / 2e-6;
                }
            }
        }
        let inverse = gax::linalg::inverse(&h);
        (0..count)
            .map(|p| {
                core::array::from_fn(|i| core::array::from_fn(|j| inverse[p * 6 + i][p * 6 + j]))
            })
            .collect()
    }

    fn assert_blocks(got: &[space::Covariance], expected: &[[[f64; 6]; 6]], atol: f64) {
        for (g, e) in got.iter().zip(expected) {
            for (row, erow) in g.coeffs().iter().zip(e) {
                for (a, b) in row.iter().zip(erow) {
                    assert!((a - b).abs() <= atol, "{a} vs {b}");
                }
            }
        }
    }

    /// Each pose's uncertainty, from dead reckoning and from the information, is its block of
    /// the inverse curvature of the objective.
    #[test]
    fn uncertainties_are_the_inverse_curvature() {
        let steps = space::lap();
        let (_, dead, problem) = space::survey(&steps, 0);
        // The step readings alone: dead reckoning's uncertainty, up to the other poses' vague
        // priors.
        let along = problem.first(steps.len());
        let expected = inverse_curvature(&along, &dead);
        assert_blocks(&problem.reckon(&dead), &expected, 1e-5);
        // Every reading: at the most likely poses the marginals are the Gauss-Newton ones; the
        // full curvature differs by what the remaining mismatches bend.
        let run = space::damped(&steps, 1.0, 6, 0);
        let (poses, _) = run.iterates.last().expect("a state");
        let expected = inverse_curvature(&run.problem, poses);
        assert_blocks(&run.problem.marginals(poses), &expected, 1e-5);
    }

    /// The lap closes in the plane, every pose's ellipse lies two standard deviations out from
    /// the point it carries, and a damped run yields a state per step.
    #[test]
    fn the_lap_closes_in_the_plane() {
        let run = plane::closing(&plane::lap(), 5, 0);
        let (poses, uncertainty) = run.iterates.last().expect("a state");
        for (p, u) in poses.iter().zip(uncertainty) {
            let here = plane::position(*p);
            let q = plane::ellipse(*p, *u);
            let level = (q.of(here) & here).s();
            assert!(
                (level + plane::SIGMAS * plane::SIGMAS).abs() < 1e-6,
                "{level}"
            );
        }
        // The closed lap ends where it started, far closer than dead reckoning.
        let gap = |ps: &[plane::M]| {
            (plane::position(ps[0]) - plane::position(ps[ps.len() - 1])).ideal_norm()
        };
        assert!(
            gap(poses) < 0.01 && gap(&run.dead) > 0.05,
            "{} {}",
            gap(poses),
            gap(&run.dead)
        );
        let damped = plane::damped(&plane::lap(), 0.2, 2, 0);
        assert_eq!(damped.iterates.len(), 3);
    }

    /// The same core in space: a short lap that pitches and rolls reaches its most likely
    /// poses by the scenario's check. Its twists have six coefficients, the plane's three.
    #[test]
    fn the_same_core_in_space() {
        space::closing(&space::lap(), 5, 0);
        assert_eq!(space::SIZE, 6);
        assert_eq!(plane::SIZE, 3);
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws(super::draw, 1.0);
    }
}
