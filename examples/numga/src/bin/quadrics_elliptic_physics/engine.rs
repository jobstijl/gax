//! Rigid quadric bodies on a sphere: one engine for S² in `Cl(3)` and S³ in `Cl(4)`.
//!
//! A body is a dual quadric, an ellipsoid of the sphere, placed by a rotor of the ambient space.
//! Its inertia is the sum over its mass points of the regressive product of each point with its
//! own motion under an open bivector rate, and its momentum is kept in the body frame, stepped by
//! a Lie midpoint rule. Two bodies meet when no blend `A + B w` of their forms with a positive
//! weight is positive semidefinite; the least eigenvector of the best blend is the deepest point,
//! and the first body's polar plane there is the contact plane, whose inner product with its pole
//! is the contact forque. Elastic impulses along that forque reverse the closing rate.
//!
//! Nothing here knows its dimension. numga instantiates its module once per algebra; here the
//! macro [`engine!`] is expanded inside a module that names the algebra's kinds. (The roles of
//! `gax::motions::Motions` cover the motors, rates, momenta and points, but not the planes and
//! the quadrics, maps between planes and points, with their solves and eigenproblems; a trait
//! bound for each of those would be longer than the engine.) The kinds:
//!
//! * `Point`: a point of the sphere (numga's antivector), `Plane`: a plane through the centre
//!   (the vectors), `Rate`: the bivectors, `Momentum`: the antibivectors, `Rotor`: the motors;
//! * `DIM`: the dimension of the ambient space, `Scalar`: the scalar kind.

/// The engine, expanded in a module that brings `Point`, `Plane`, `Rate`, `Momentum`, `Rotor`,
/// `Scalar` and `DIM` into scope.
macro_rules! engine {
    () => {
        use gax_numga_examples::rng::{Draw, Rng};

        /// A point of the sphere.
        pub type P = Point<(), f64>;
        /// A rate: a bivector, in the body frame.
        pub type B = Rate<(), f64>;
        /// A momentum: an antibivector, in the body frame.
        pub type Mom = Momentum<(), f64>;
        /// A rotation of the ambient space.
        pub type M = gax::Unit<Rotor<(), f64>>;
        /// The primal quadric: a point to its polar plane, negative inside.
        pub type Quadric = Plane<(Point,), f64>;
        /// The dual quadric: a plane to its pole.
        pub type DualQuadric = Point<(Plane,), f64>;
        /// The inertia: a rate to its momentum.
        pub type Inertia = Momentum<(Rate,), f64>;
        /// The inverse inertia: a momentum to its rate.
        pub type InverseInertia = Rate<(Momentum,), f64>;

        /// The poles of the basis planes.
        pub fn basis() -> [P; DIM] {
            core::array::from_fn(|i| {
                Point::from_coeffs(core::array::from_fn(|j| if i == j { 1.0 } else { 0.0 }))
            })
        }

        /// The last basis point: the centre of every body in its own frame.
        pub fn origin() -> P {
            basis()[DIM - 1]
        }

        /// The identity rotor.
        pub fn identity() -> M {
            Rate::<(), f64>::zero().exp()
        }

        /// The angle between two unit points of the sphere, or between one and the other's
        /// antipode where that is smaller: the angle whose sine is the norm of their join (the
        /// great circle through them) and whose cosine is their inner product, up to sign.
        pub fn arc(a: P, b: P) -> f64 {
            (a & b).norm().atan2((a | b).s().abs())
        }

        /// One body: its colour (a light), its motor, its momentum in the body frame, its
        /// shape in its own frame, dual (a plane to its pole) and primal (a point to its polar
        /// plane, negative inside), its inverse inertia, and the angular radius of its bounding
        /// ball for the broad phase.
        #[derive(Clone, Copy)]
        pub struct Body {
            pub color: gax_light::Light,
            pub motor: M,
            pub momentum: Mom,
            pub q: DualQuadric,
            pub c: Quadric,
            pub i_inv: InverseInertia,
            pub reach: f64,
        }

        impl Body {
            /// The rate in the body frame.
            pub fn rate(&self) -> B {
                self.i_inv.of(self.momentum)
            }

            /// The primal form placed in the world.
            pub fn world(&self) -> Quadric {
                self.motor >> self.c.of(self.motor << Point::slot())
            }
        }

        /// The kinetic energy of a population.
        pub fn kinetic_energy(bodies: &[Body]) -> f64 {
            bodies
                .iter()
                .map(|b| (b.rate() & b.momentum).s())
                .sum::<f64>()
                * 0.5
        }

        /// The momenta of all bodies carried into the world frame and summed.
        pub fn total_momentum(bodies: &[Body]) -> Mom {
            bodies
                .iter()
                .fold(Mom::zero(), |acc, b| acc + (b.motor >> b.momentum))
        }

        /// The simulation, frame by frame: the bodies' colours, the world forms and body rates
        /// per body, the energy and the total momentum (in the world frame) per frame, and the
        /// impulses applied.
        pub struct Trajectory {
            pub colors: Vec<gax_light::Light>,
            pub surfaces: Vec<Vec<Quadric>>,
            pub rates: Vec<Vec<B>>,
            pub energy: Vec<f64>,
            pub momentum: Vec<Mom>,
            pub impulses: usize,
        }

        // --- shapes and bodies -------------------------------------------------------------

        /// The dual quadric diagonal on the basis points: each point paired with itself,
        /// weighted; negative weights mark the inside's core, positive ones its extents.
        pub fn quadric(diagonal: [f64; DIM]) -> DualQuadric {
            let plane = Plane::slot();
            let mut q = DualQuadric::zero();
            for (p, d) in basis().iter().zip(diagonal) {
                q += (*p & plane) * p.gp(d);
            }
            q
        }

        /// The ellipsoid about the origin with the given half-widths along the axes, each the
        /// tangent of an angular half-width.
        pub fn ellipsoid(half_widths: [f64; DIM - 1]) -> DualQuadric {
            let mut d = [-1.0; DIM];
            for (k, h) in half_widths.iter().enumerate() {
                d[k] = h * h;
            }
            quadric(d)
        }

        /// The momentum of a cloud under an open rate: each point's join with its own motion,
        /// mass-weighted.
        pub fn pointcloud_inertia(points: &[P], masses: &[f64]) -> Inertia {
            let rate = Rate::slot();
            points
                .iter()
                .zip(masses)
                .fold(Inertia::zero(), |acc, (p, m)| {
                    let motion = p.commutator(rate);
                    acc + (*p & motion).gp(*m)
                })
        }

        /// A body of any quadric shape, from mass points filling it: inertia from the points,
        /// momentum from the body-frame rate, and the reach of the points from the pole.
        pub fn body(
            color: gax_light::Light,
            q: DualQuadric,
            motor: M,
            rate: B,
            points: &[P],
            masses: &[f64],
        ) -> Body {
            let inertia = pointcloud_inertia(points, masses);
            let reach = points.iter().map(|p| arc(*p, origin())).fold(0.0, f64::max);
            Body {
                color,
                motor,
                momentum: inertia.of(rate),
                q,
                c: q.inverse(),
                i_inv: inertia.inverse(),
                reach,
            }
        }

        /// The symmetric form `P ∨ Q⁻¹(P)` of a dual quadric, on points.
        pub fn form(q: DualQuadric) -> Scalar<(Point, Point), f64> {
            Point::slot() & q.solve(Point::slot())
        }

        /// A point uniform on the unit sphere of the span of orthonormal `axes`: normal draws on
        /// them, normalized.
        fn uniform_on(axes: &[P], rng: &mut Rng) -> P {
            axes.iter()
                .fold(P::zero(), |p, a| p + a.gp(rng.normal()))
                .normalized()
                .into_inner()
        }

        /// Mass points filling the inside of any quadric, where its form is negative. In the
        /// form's eigenbasis the inside is where the negative block outweighs the positive one,
        /// so a point is a core point in the span of the negative block, an extent point in the
        /// span of the positive block, both uniform on their spheres, and the angle between
        /// them, up to the angle where the blocks balance; the angle is drawn uniformly and
        /// weighted by the sphere's measure `cos^(k-1) sin^(n-1-k)` for `k` core axes out of
        /// `n`, so the weighted points are uniform in the inside.
        pub fn filled(
            q: DualQuadric,
            mass: f64,
            count: usize,
            rng: &mut Rng,
        ) -> (Vec<P>, Vec<f64>) {
            let form = form(q);
            let (values, principal) = form.eigh();
            let k = values.iter().filter(|v| **v < 0.0).count();
            let mut points = Vec::with_capacity(count);
            let mut weights = Vec::with_capacity(count);
            for _ in 0..count {
                let core = uniform_on(&principal[..k], rng);
                let extent = uniform_on(&principal[k..], rng);
                let (inward, outward) = (-form.fill(core).s(), form.fill(extent).s());
                let balance = (inward / outward).sqrt().atan();
                let angle = balance * rng.uniform();
                weights.push(
                    angle.cos().powi(k as i32 - 1)
                        * angle.sin().powi((DIM - 1 - k) as i32)
                        * balance,
                );
                points.push(core.gp(angle.cos()) + extent.gp(angle.sin()));
            }
            let total: f64 = weights.iter().sum();
            (points, weights.iter().map(|w| w * mass / total).collect())
        }

        // --- math --------------------------------------------------------------------------

        /// Advance a body by `dt` with a Lie midpoint step; the momentum is kept in the body
        /// frame, so the free step only turns it.
        pub fn step_motor(motor: M, momentum: Mom, i_inv: InverseInertia, dt: f64) -> (M, Mom) {
            let half = motor.mul_renormalized(i_inv.of(momentum).gp(0.25 * dt).exp());
            let rate = i_inv.of((motor.inverse() * half) << momentum);
            let moved = motor.mul_renormalized(rate.gp(0.5 * dt).exp());
            (moved, (motor.inverse() * moved) << momentum)
        }

        /// The least eigenvalue of the blend `a + b tan φ` and its eigenvector.
        fn least(a: Quadric, b: Quadric, phi: f64) -> (f64, P) {
            let blend = a + b.gp(phi.tan());
            let (values, points) = (Point::slot() & blend.of(Point::slot())).eigh();
            (values[0], points[0])
        }

        /// Whether the insides of two quadrics (where their forms are negative) meet: they are
        /// apart if and only if some blend `a + b w` with a positive weight is positive
        /// semidefinite (the S-lemma), so the largest over the blends of the least eigenvalue is
        /// negative exactly when they overlap. The least eigenvalue is concave in the weight,
        /// hence unimodal in `φ = atan w` between 0 and π/2, and golden section finds its
        /// maximum; the least eigenvector there is the deepest point. Twelve iterations bracket
        /// `φ` to 0.005 rad.
        pub fn overlap(a: Quadric, b: Quadric, iterations: usize) -> (f64, P) {
            // Two probes c < d split the bracket in the golden ratio; whichever side holds the
            // larger value keeps the bracket, and the surviving probe already sits at the golden
            // point of the shrunk bracket, so each step evaluates one fresh probe.
            let golden = (5f64.sqrt() - 1.0) / 2.0;
            let (mut lo, mut hi) = (0.0, core::f64::consts::FRAC_PI_2);
            let (mut c, mut d) = (hi - golden * (hi - lo), lo + golden * (hi - lo));
            let (mut fc, mut fd) = (least(a, b, c).0, least(a, b, d).0);
            for _ in 0..iterations {
                let left = fc > fd;
                if left {
                    hi = d;
                } else {
                    lo = c;
                }
                (c, d) = (hi - golden * (hi - lo), lo + golden * (hi - lo));
                if left {
                    (fc, fd) = (least(a, b, c).0, fc);
                } else {
                    (fc, fd) = (fd, least(a, b, d).0);
                }
            }
            least(a, b, (lo + hi) / 2.0)
        }

        /// Broad phase: the pairs whose poles are closer than their reaches summed.
        pub fn candidates_near(bodies: &[Body]) -> Vec<(usize, usize)> {
            let poles: Vec<P> = bodies.iter().map(|b| b.motor >> origin()).collect();
            let mut pairs = Vec::new();
            for i in 0..bodies.len() {
                for j in i + 1..bodies.len() {
                    if arc(poles[i], poles[j]) < bodies[i].reach + bodies[j].reach {
                        pairs.push((i, j));
                    }
                }
            }
            pairs
        }

        /// The margin and deepest point of a pair, in the first body's frame, and the relative
        /// motor from the first body's frame to the second's.
        pub fn margin(bodies: &[Body], motors: &[M], i: usize, j: usize) -> (f64, P, M) {
            let relative = motors[i].inverse() * motors[j];
            let (m, deepest) = overlap(
                bodies[i].c,
                relative >> bodies[j].c.of(relative << Point::slot()),
                12,
            );
            (m, deepest, relative)
        }

        /// Contact test and elastic impulses for the candidate pairs, each in the frame of its
        /// first body. The margin is negative while the bodies meet. An impulse is applied only
        /// while the overlap deepens, judged by a virtual step of all bodies along their current
        /// rates, so no orientation of the contact forque is assumed; the impulse reflects the
        /// closing rate along the forque whatever its sign. The geometry is computed for all
        /// pairs first; the impulses go one pair at a time, each against the momenta the earlier
        /// ones left. Returns the number applied.
        pub fn collide(bodies: &mut [Body], pairs: &[(usize, usize)], dt: f64) -> usize {
            let now: Vec<M> = bodies.iter().map(|b| b.motor).collect();
            let ahead: Vec<M> = bodies
                .iter()
                .map(|b| b.motor.mul_renormalized(b.rate().gp(0.5 * dt).exp()))
                .collect();
            let mut contacts = Vec::new();
            for &(i, j) in pairs {
                let (m, deepest, relative) = margin(bodies, &now, i, j);
                if m < 0.0 && margin(bodies, &ahead, i, j).0 < m {
                    // The first body's polar plane at the deepest point is the contact plane,
                    // and its pole the contact point. The forque is the line through the contact
                    // point normal to the plane: their inner product.
                    let plane = bodies[i].c.of(deepest).normalized().into_inner();
                    let point = bodies[i].q.of(plane);
                    let one: Mom = plane | point;
                    contacts.push((i, j, one, relative << one));
                }
            }
            for &(a, b, one, other) in &contacts {
                let (r_one, r_other) = (bodies[a].i_inv.of(one), bodies[b].i_inv.of(other));
                let closing = (r_one & bodies[a].momentum).s() - (r_other & bodies[b].momentum).s();
                let compliance = (one & r_one).s() + (other & r_other).s();
                // Elastic: the closing rate reverses.
                let impulse = -2.0 * closing / compliance;
                bodies[a].momentum += one.gp(impulse);
                bodies[b].momentum -= other.gp(impulse);
            }
            contacts.len()
        }

        /// Step every body freely by `dt`, then resolve the contacts among the pairs within
        /// reach.
        pub fn advance(bodies: &mut [Body], dt: f64) -> usize {
            for b in bodies.iter_mut() {
                (b.motor, b.momentum) = step_motor(b.motor, b.momentum, b.i_inv, dt);
            }
            let pairs = candidates_near(bodies);
            collide(bodies, &pairs, dt)
        }

        /// The initial state and `frames - 1` more, each `dt` apart in `substeps`.
        pub fn simulate(
            mut bodies: Vec<Body>,
            frames: usize,
            dt: f64,
            substeps: usize,
        ) -> Trajectory {
            let mut t = Trajectory {
                colors: bodies.iter().map(|b| b.color).collect(),
                surfaces: Vec::new(),
                rates: Vec::new(),
                energy: Vec::new(),
                momentum: Vec::new(),
                impulses: 0,
            };
            for frame in 0..frames {
                if frame > 0 {
                    for _ in 0..substeps {
                        t.impulses += advance(&mut bodies, dt / substeps as f64);
                    }
                }
                t.surfaces.push(bodies.iter().map(Body::world).collect());
                t.rates.push(bodies.iter().map(Body::rate).collect());
                t.energy.push(kinetic_energy(&bodies));
                t.momentum.push(total_momentum(&bodies));
            }
            t
        }
    };
}

pub(crate) use engine;
