//! numga's `core.py` and `scenarios.py` of the tippe top: the hollow body built of quadrics, its
//! mass read from them, its launch, and its steps: a free rigid step, then the spinning top's
//! contact corrections at the shell and the tip.

use gax::motions::Motions;
use gax::pga3d::{Direction, Motor, Plane, Point};

use crate::contact::{Quadric, bowl, ellipsoid_in, lowest_point, project_contacts};
use gax_numga_examples::shared::mechanics_lie::Lie;
use gax_numga_examples::shared::mechanics_lie::rigid::{G, Inertia, InertiaInv, M, R};

/// The planes across `x`, `y` and `z`, and the plane at infinity.
pub fn axes() -> [Plane; 3] {
    [
        Plane::new(1.0, 0.0, 0.0, 0.0),
        Plane::new(0.0, 1.0, 0.0, 0.0),
        Plane::new(0.0, 0.0, 1.0, 0.0),
    ]
}
pub fn w() -> Plane {
    Plane::infinity()
}

/// The origin, the shell's centre, and the direction up.
pub fn origin() -> Point {
    w().dual()
}
pub fn up() -> Direction {
    axes()[2].dual().cast::<Direction>()
}

/// The table: everything below the plane `z = 0`.
pub fn ground() -> Quadric {
    bowl(0.0)
}

/// The turn in the plane of two planes by an angle, numga's `exp(a ^ b · (−angle / 2))`.
pub fn turn(a: Plane, b: Plane, angle: f64) -> M {
    ((a ^ b) * (-angle / 2.0)).exp()
}

/// The top: the shell carved by the cavity, joined to the stem and the ball at its end, one
/// solid of uniform density; its mass, a point at the centre of mass weighted by the mass, and
/// its inertia about the shell's centre.
#[derive(Clone, Copy)]
pub struct Body {
    pub shell: Quadric,
    pub cavity: Quadric,
    pub stem: Quadric,
    pub tip: Quadric,
    pub mass: Point,
    pub inertia: Inertia,
    pub inertia_inv: InertiaInv,
}

impl Body {
    /// The two surfaces that meet the table: the shell and the tip.
    pub fn contacts(&self) -> [Quadric; 2] {
        [self.shell, self.tip]
    }
}

/// The shell carved by the cavity, joined to the stem and the tip: one solid of uniform
/// density, its pieces counted once where they overlap, its mass and inertia summed over the
/// samples inside it, each carrying its box's volume.
pub fn build(
    shell: Quadric,
    cavity: Quadric,
    stem: Quadric,
    tip: Quadric,
    samples: &[Point],
    volume: f64,
) -> Body {
    let inside = |q: Quadric, p: Point| (q.of(p) & p).s() <= 0.0;
    let material: Vec<Point> = samples
        .iter()
        .copied()
        .filter(|p| {
            (inside(shell, *p) && !inside(cavity, *p)) || inside(stem, *p) || inside(tip, *p)
        })
        .collect();
    let mass: Point = material.iter().map(|p| *p * volume).sum();
    // Each mass sample joins its velocity to its position, contributing to the body's momentum.
    let inertia: Inertia = material.iter().map(|p| G::point_inertia(*p) * volume).sum();
    Body {
        shell,
        cavity,
        stem,
        tip,
        mass,
        inertia,
        inertia_inv: G::mobility(inertia),
    }
}

/// Opposite spins about the centre of mass, with the same lean and no initial penetration: the
/// lean about `y`, then lifted so that the lower of the contacts' supports touches the table.
pub fn launch(body: &Body, spins: &[f64], lean: f64) -> Vec<(M, R)> {
    let [x, y, z] = axes();
    let tilt = turn(x, z, -lean);
    let normal = (tilt << Point::from(up())).cast::<Direction>();
    let height = body
        .contacts()
        .iter()
        .map(|q| (z & (tilt >> lowest_point(*q, normal))).s())
        .fold(f64::MAX, f64::min);
    let motor = Motor::translation(Point::from(up() * -height)) * tilt;
    let centre = body.mass.unitized();
    let spin_axis = (x - w() * (x & centre).s()) ^ (y - w() * (y & centre).s());
    spins.iter().map(|s| (motor, spin_axis * *s)).collect()
}

/// The top advanced by `n_steps` steps: free rigid motion under gravity, then the contact
/// corrections (numga's XPBD) at the shell and the tip.
#[allow(clippy::too_many_arguments)]
pub fn advance(
    body: &Body,
    motor: M,
    rate: R,
    gravity: f64,
    friction: [f64; 2],
    dt: f64,
    n_steps: usize,
) -> (M, R) {
    // The weight: the mass, joined with gravity's direction in the body.
    let load = |placement: M, _: R| body.mass & (placement << Point::from(up() * -gravity));
    let mut state = (motor, rate);
    for _ in 0..n_steps {
        let (before, velocity) = state;
        let (predicted, _) =
            G::explicit_rk4(before, velocity, body.inertia, body.inertia_inv, dt, &load);
        state = project_contacts(
            before,
            predicted.renormalize_fast(),
            body.inertia_inv,
            &body.contacts(),
            ground(),
            friction,
            dt,
        );
    }
    state
}

/// The kinetic and gravitational energy of the top.
#[cfg(test)]
pub fn energy(body: &Body, motor: M, rate: R, gravity: f64) -> f64 {
    (body.inertia.of(rate) & rate).s() / 2.0 + gravity * (axes()[2] & (motor >> body.mass)).s()
}

/// How far up the top's axis points: its pairing with the plane across `z`.
pub fn vertical(motor: M) -> f64 {
    (axes()[2] & (motor >> Point::from(up()))).s()
}

// --- scenes

/// The outer shell about the origin; the cavity carved from it, open at the top, turned and
/// tilted from the shell's axes; the stem and the ball at its end.
pub const SHELL_SEMIAXES: [f64; 3] = [1.05, 0.95, 1.01];
pub const CAVITY_CENTRE: [f64; 3] = [-0.05, -0.04, 0.47];
pub const CAVITY_SEMIAXES: [f64; 3] = [0.72, 0.65, 0.59];
pub const CAVITY_ANGLE: f64 = 0.16;
pub const CAVITY_TILT: f64 = 0.07;
pub const STEM_CENTRE: [f64; 3] = [0.0, 0.0, 0.21];
pub const STEM_SEMIAXES: [f64; 3] = [0.18, 0.18, 1.02];
pub const TIP_CENTRE: [f64; 3] = [0.0, 0.0, 1.23];
pub const TIP_RADIUS: f64 = 0.20;
/// The mass samples: their spacing, and the box they fill.
pub const SAMPLE_SPACING: f64 = 0.025;
pub const SAMPLE_LOWER: [f64; 3] = [-1.125, -1.125, -1.125];
pub const SAMPLE_UPPER: [f64; 3] = [1.125, 1.125, 1.525];
/// Equal and opposite spins, the lean they start at, gravity and friction.
pub const SPINS: [f64; 2] = [-13.0, 13.0];
pub const LEAN: f64 = 0.1;
pub const GRAVITY: f64 = 1.0;
pub const DYNAMIC_FRICTION: f64 = 0.35;
pub const STATIC_FRICTION: f64 = 1.2 * DYNAMIC_FRICTION;
pub const STEP: f64 = 0.004;
/// The states recorded: one every frame interval over the run.
pub const FRAME_INTERVAL: f64 = 0.16;
pub const RUN_TIME: f64 = 80.0;

/// The steps per frame, and the states recorded (the start and one per frame).
pub fn n_steps_per_frame() -> usize {
    (FRAME_INTERVAL / STEP).round() as usize
}
pub fn n_states() -> usize {
    (RUN_TIME / FRAME_INTERVAL).round() as usize + 1
}

/// The friction coefficients, static and dynamic.
pub const FRICTION: [f64; 2] = [STATIC_FRICTION, DYNAMIC_FRICTION];

/// Midpoints of equal-volume boxes covering the solid (numpy's `arange` along each axis).
pub fn quadrature_grid() -> Vec<Point> {
    let along = |k: usize| {
        let (low, high) = (SAMPLE_LOWER[k], SAMPLE_UPPER[k]);
        let n = ((high - low) / SAMPLE_SPACING).ceil() as usize;
        (0..n)
            .map(|i| low + i as f64 * SAMPLE_SPACING + SAMPLE_SPACING / 2.0)
            .collect::<Vec<f64>>()
    };
    let (xs, ys, zs) = (along(0), along(1), along(2));
    let mut grid = Vec::with_capacity(xs.len() * ys.len() * zs.len());
    for x in &xs {
        for y in &ys {
            for z in &zs {
                grid.push(Point::xyz(*x, *y, *z));
            }
        }
    }
    grid
}

/// The four quadrics and the uniform material they enclose.
pub fn body() -> Body {
    let [x, y, z] = axes();
    let point = |[a, b, c]: [f64; 3]| Point::xyz(a, b, c);
    let cavity_turn = turn(y, z, CAVITY_TILT) * turn(x, y, CAVITY_ANGLE);
    let shell = ellipsoid_in(origin(), axes(), SHELL_SEMIAXES);
    let cavity = ellipsoid_in(
        point(CAVITY_CENTRE),
        axes().map(|a| cavity_turn >> a),
        CAVITY_SEMIAXES,
    );
    let stem = ellipsoid_in(point(STEM_CENTRE), axes(), STEM_SEMIAXES);
    let tip = ellipsoid_in(point(TIP_CENTRE), axes(), [TIP_RADIUS; 3]);
    let volume = SAMPLE_SPACING * SAMPLE_SPACING * SAMPLE_SPACING;
    build(shell, cavity, stem, tip, &quadrature_grid(), volume)
}

/// Both launches over the run: their placements and rates at every recorded state.
pub fn evolve(body: &Body) -> [Vec<(M, R)>; 2] {
    let starts = launch(body, &SPINS, LEAN);
    let per = n_steps_per_frame();
    let run = |(motor, rate): (M, R)| {
        let mut states = vec![(motor, rate)];
        for _ in 1..n_states() {
            let (m, r) = states[states.len() - 1];
            states.push(advance(body, m, r, GRAVITY, FRICTION, STEP, per));
        }
        states
    };
    let runs = gax_numga_examples::par_map(&starts, |s| run(*s));
    [runs[0].clone(), runs[1].clone()]
}
