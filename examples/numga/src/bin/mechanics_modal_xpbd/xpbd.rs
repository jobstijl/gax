//! Modal extended position-based dynamics: flexible bodies under point constraints, every
//! constraint and mode coupled in one sparse system per step.

use crate::girder::{D, DMap, Inertia, P, S, Shape, origin, work};
use crate::lie::{self, Lie};
use gax::field::{Field, SparseMap};
use gax::motions::Pga2d;
use gax::pga2d::{Direction, Forque, Point, Scalar, Twist};

/// A pose.
pub type M = lie::M<Pga2d>;
/// A rate in the body frame, or a twist: an extrinsic point of the plane (a turn about it).
pub type R = lie::R<Pga2d>;
/// Forque to twist.
pub type Mobility = lie::InertiaInv<Pga2d>;
/// How a gap changes with a body's twist.
pub type RigidCell = Direction<(Twist,), f64>;
/// How a gap changes with a body's amplitude of one mode.
pub type ModalCell = Direction<(Scalar,), f64>;

/// Flexible bodies in motion: per body its pose and rate, and per mode a field over the bodies
/// of its amplitudes, their rates, and its compliance (zero for a fixed body).
#[derive(Clone, Debug)]
pub struct Bodies {
    pub motor: Vec<M>,
    pub rate: Vec<R>,
    pub amplitudes: Vec<Field<S>>,
    pub rates: Vec<Field<S>>,
    pub compliance: Vec<Field<S>>,
    pub frequencies: Vec<f64>,
    /// The modes' damping ratio.
    pub damping: f64,
    /// Each body's mass and inertia, and its inverse inertia, zero for a fixed body.
    pub mass: f64,
    pub inertia: Inertia,
    pub inverse_inertia: Vec<Mobility>,
}

impl Bodies {
    /// Each body's points in the world: its rest points moved by its modes, then by its motor.
    pub fn points(&self, shape: &Shape) -> Vec<Field<P>> {
        (0..self.motor.len())
            .map(|b| {
                let local = shape.deformed(|m| self.amplitudes[m].site(b).s());
                local.map(|p| self.motor[b] >> p)
            })
            .collect()
    }
}

/// Point constraints between anchor points of two bodies.
#[derive(Clone, Debug)]
pub struct Constraints {
    /// The body at each side of each constraint.
    pub bodies: Vec<[usize; 2]>,
    /// The anchor at each side, in its body's frame at rest.
    pub anchors: Vec<[P; 2]>,
    /// Per mode, its displacement of the anchor at each side of each constraint.
    pub modes: Vec<Vec<[D; 2]>>,
    /// Each constraint's compliance.
    pub compliance: f64,
}

/// A gap is the second anchor's position less the first's.
pub const SIDES: [f64; 2] = [-1.0, 1.0];

impl Constraints {
    /// The anchor at a side of a constraint in its body's frame, moved by the body's modes.
    pub fn local_anchor(&self, bodies: &Bodies, c: usize, side: usize) -> P {
        let body = self.bodies[c][side];
        let offset: D = (0..self.modes.len())
            .map(|m| self.modes[m][c][side] * bodies.amplitudes[m].site(body).s())
            .sum();
        self.anchors[c][side] + offset.cast::<Point>()
    }

    /// Each constraint's gap, in the world.
    pub fn gaps(&self, bodies: &Bodies) -> Field<D> {
        Field::from_fn(self.bodies.len(), |c| {
            let ends = [0, 1].map(|side| {
                let body = self.bodies[c][side];
                bodies.motor[body] >> self.local_anchor(bodies, c, side)
            });
            (ends[1] - ends[0]).cast::<Direction>()
        })
    }
}

/// The maps from the bodies' rigid motion and modes to the constraints' gaps, and the gaps.
#[derive(Clone, Debug)]
pub struct Coupling {
    /// The change in each gap for the bodies' twists, `[constraints, bodies]`.
    pub rigid: SparseMap<RigidCell>,
    /// Per mode, the change in each gap for the bodies' amplitudes.
    pub modal: Vec<SparseMap<ModalCell>>,
    pub gap: Field<D>,
}

/// A way to find the couplings: by hand ([`coupling`]) or by differentiating the gaps.
pub type Couple = fn(&Bodies, &Constraints) -> Coupling;

/// A field map from the bodies to the constraints, one cell per side of each constraint.
pub fn per_side<C: Copy>(
    constraints: &Constraints,
    bodies: usize,
    cell: impl Fn(usize, usize) -> C,
) -> SparseMap<C> {
    let sides = constraints
        .bodies
        .iter()
        .enumerate()
        .flat_map(|(c, b)| [(c, 0, b[0]), (c, 1, b[1])]);
    let (mut cells, mut rows, mut cols) = (Vec::new(), Vec::new(), Vec::new());
    for (c, side, body) in sides {
        cells.push(cell(c, side));
        rows.push(c);
        cols.push(body);
    }
    SparseMap::new(cells, rows, cols, (constraints.bodies.len(), bodies))
}

/// The couplings built by hand: each anchor's commutator with an open twist, moved into the
/// world, for the rigid motion; each mode's displacement of each anchor, moved into the world,
/// for the modes; both signed for the two sides.
pub fn coupling(bodies: &Bodies, constraints: &Constraints) -> Coupling {
    let count = bodies.motor.len();
    let motor = |c: usize, side: usize| bodies.motor[constraints.bodies[c][side]];
    let rigid = per_side(constraints, count, |c, side| {
        let anchor = constraints.local_anchor(bodies, c, side);
        let motion = motor(c, side) >> anchor.commutator(Twist::oriented_slot());
        motion.cast::<Direction>() * SIDES[side]
    });
    let modal = (0..constraints.modes.len())
        .map(|m| {
            per_side(constraints, count, |c, side| {
                let shape = (motor(c, side) >> constraints.modes[m][c][side].cast::<Point>())
                    .cast::<Direction>();
                (shape * SIDES[side] * Scalar::slot()).cast::<Direction>()
            })
        })
        .collect();
    Coupling {
        rigid,
        modal,
        gap: constraints.gaps(bodies),
    }
}

/// The forque a force exerts through a rigid coupling (the coupling's adjugate under the work
/// pairing): the line whose pairing with any twist is the work the force does along the
/// displacement that twist causes. A pairing solve.
fn forque_of(cell: RigidCell) -> Forque<(Direction,), f64> {
    let pairing = Forque::oriented_slot() & Twist::oriented_slot();
    pairing.solve(work().of_both(Direction::slot(), cell))
}

/// The load a force puts on a mode through a modal coupling: the work it does along the mode's
/// displacement.
fn load_of(cell: ModalCell) -> Scalar<(Direction,), f64> {
    work().of(cell.of(S::new(1.0)))
}

/// One mode of one body over a step: its spring's compliance over the step squared, softened
/// by the implicit dashpot; its amplitude with the change over the step weighted in by the
/// dashpot, the residual; and its response to a unit load, `1 / (1 + compliance)` for a mode of
/// unit mass.
#[derive(Clone, Copy, Debug)]
struct ModalTerm {
    compliance: f64,
    residual: f64,
    response: f64,
}

/// A mode's term over a step, from its spring's compliance, its frequency, its damping ratio,
/// and its amplitude now and before the step. The dashpot is twice the damping ratio times the
/// frequency times the spring's compliance, over the step.
fn modal_term(
    compliance: f64,
    frequency: f64,
    damping: f64,
    amplitude: [f64; 2],
    dt: f64,
) -> ModalTerm {
    let [now, previous] = amplitude;
    let dashpot = 2.0 * damping * frequency * compliance / dt;
    let softened = compliance / (dt * dt) / (1.0 + dashpot);
    ModalTerm {
        compliance: softened,
        residual: (now + dashpot * (now - previous)) / (1.0 + dashpot),
        response: 1.0 / (1.0 + softened),
    }
}

/// Each mode's compliance, residual and response over a step, per mode a field over the bodies.
pub struct ModalTerms {
    pub compliance: Vec<Field<S>>,
    pub residual: Vec<Field<S>>,
    pub response: Vec<Field<S>>,
}

/// The modal terms of every mode of every body, against the amplitudes before the step.
pub fn modal_terms(bodies: &Bodies, previous: &[Field<S>], dt: f64) -> ModalTerms {
    let term = |m: usize, b: usize| {
        let amplitude = [bodies.amplitudes[m].site(b).s(), previous[m].site(b).s()];
        let compliance = bodies.compliance[m].site(b).s();
        modal_term(
            compliance,
            bodies.frequencies[m],
            bodies.damping,
            amplitude,
            dt,
        )
    };
    let count = bodies.motor.len();
    let part = |of: fn(ModalTerm) -> f64| -> Vec<Field<S>> {
        (0..bodies.frequencies.len())
            .map(|m| Field::from_fn(count, |b| S::new(of(term(m, b)))))
            .collect()
    };
    ModalTerms {
        compliance: part(|t| t.compliance),
        residual: part(|t| t.residual),
        response: part(|t| t.response),
    }
}

/// The product of two scalar fields, site by site.
fn times(a: &Field<S>, b: &Field<S>) -> Field<S> {
    a.zip_with(b, |x, y| x * y)
}

/// The change in the gaps for forces at the constraints (held as the directions they pull in):
/// the forques the forces exert, through the inverse inertias to twists, and the twists to the
/// gaps; the loads on the modes, through each mode's mobility (its response times its
/// compliance) to amplitudes, and the amplitudes to the gaps; and the constraints' own
/// compliance over the step squared.
fn system(
    coupling: &Coupling,
    forques: &SparseMap<Forque<(Direction,), f64>>,
    loads: &[SparseMap<Scalar<(Direction,), f64>>],
    inverse: &SparseMap<Mobility>,
    mobility: &[Field<S>],
    compliance: f64,
) -> SparseMap<DMap> {
    let mut system: SparseMap<DMap> = coupling.rigid.compose(&inverse.compose(forques));
    for ((modal, load), mobility) in coupling.modal.iter().zip(loads).zip(mobility) {
        let mobility = SparseMap::from_diagonal(&mobility.map(|k| Scalar::slot() * k.s()));
        let through: SparseMap<DMap> = modal.compose(&mobility.compose(load));
        system = &system + &through;
    }
    let own = Field::from_fn(coupling.gap.len(), |_| Direction::slot() * compliance);
    &system + &SparseMap::from_diagonal(&own)
}

/// The bodies displaced to satisfy all constraints, the modes' springs reacting with them, given
/// the reactions the springs have already received this step: one sparse system for every
/// constraint, given how far the modes move on their own, solved once.
pub fn project(
    bodies: &Bodies,
    constraints: &Constraints,
    previous: &[Field<S>],
    received: &[Field<S>],
    dt: f64,
    couple: Couple,
) -> Bodies {
    let terms = modal_terms(bodies, previous, dt);
    let coupling = couple(bodies, constraints);
    let modes = 0..terms.compliance.len();
    // The load of each mode's spring, less the reactions it has already received this step, and
    // how far each mode moves under it alone.
    let spring_load: Vec<Field<S>> = modes
        .clone()
        .map(|m| -(&terms.residual[m] + &times(&terms.compliance[m], &received[m])))
        .collect();
    let mobility: Vec<Field<S>> = modes
        .clone()
        .map(|m| times(&terms.response[m], &terms.compliance[m]))
        .collect();
    let forques = coupling.rigid.transpose_with(forque_of);
    let loads: Vec<_> = coupling
        .modal
        .iter()
        .map(|m| m.transpose_with(load_of))
        .collect();
    let inverse = SparseMap::from_diagonal(&Field::new(bodies.inverse_inertia.clone()));
    let compliance = constraints.compliance / (dt * dt);
    let system = system(&coupling, &forques, &loads, &inverse, &mobility, compliance);
    // The reactions at the constraints that close every gap: position-level, a force times the
    // step squared, so through an inverse inertia they give a displacement.
    let mut target = -&coupling.gap;
    for (modal, (spring, response)) in coupling
        .modal
        .iter()
        .zip(spring_load.iter().zip(&terms.response))
    {
        target = &target - &modal.apply(&times(response, spring));
    }
    let reactions = system.solve_direct(&target);
    let displacement: Field<R> = inverse.apply(&forques.apply(&reactions));
    let mut moved = bodies.clone();
    for (motor, twist) in moved.motor.iter_mut().zip(displacement.iter()) {
        *motor = *motor * (twist * -0.5).exp();
    }
    for m in modes {
        // The mode's load from the reactions, and its spring's reaction to the whole.
        let mode_load = loads[m].apply(&reactions);
        let change = times(&terms.response[m], &(&spring_load[m] - &mode_load));
        moved.amplitudes[m] = &(&moved.amplitudes[m] + &mode_load) + &change;
    }
    moved
}

/// One time step: the rigid motion predicted from the rigid-body equations under gravity (an
/// ideal point) and the modes moved by their rates; each mode's spring relaxed on its own; the
/// bodies displaced so that every constraint holds, the springs relaxing with them; and the
/// rates from the change over the step, the motor's by the logarithm of its change.
pub fn step(
    bodies: &Bodies,
    constraints: &Constraints,
    dt: f64,
    gravity: P,
    couple: Couple,
) -> Bodies {
    // Gravity acts at each body's centre of mass, the origin of its frame.
    let weight = |motor: M, _: R| (origin() & (motor << gravity)) * bodies.mass;
    let mut moved = bodies.clone();
    for (b, motor) in moved.motor.iter_mut().enumerate() {
        let inverse = bodies.inverse_inertia[b];
        (*motor, _) =
            Pga2d::explicit_rk4(*motor, bodies.rate[b], bodies.inertia, inverse, dt, &weight);
    }
    for (amplitude, rate) in moved.amplitudes.iter_mut().zip(&bodies.rates) {
        *amplitude = &*amplitude + &rate.scaled(dt);
    }
    let terms = modal_terms(&moved, &bodies.amplitudes, dt);
    let received: Vec<Field<S>> = terms
        .residual
        .iter()
        .zip(&terms.response)
        .map(|(r, k)| -times(r, k))
        .collect();
    for (amplitude, relaxed) in moved.amplitudes.iter_mut().zip(&received) {
        *amplitude = &*amplitude + relaxed;
    }
    let mut moved = project(
        &moved,
        constraints,
        &bodies.amplitudes,
        &received,
        dt,
        couple,
    );
    for (b, rate) in moved.rate.iter_mut().enumerate() {
        *rate = Pga2d::rate_between(bodies.motor[b], moved.motor[b], dt);
    }
    moved.rates = moved
        .amplitudes
        .iter()
        .zip(&bodies.amplitudes)
        .map(|(now, before)| (now - before).scaled(1.0 / dt))
        .collect();
    moved
}
