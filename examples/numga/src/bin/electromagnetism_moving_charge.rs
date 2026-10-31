//! numga's `electromagnetism/moving_charge`: the field of a charge moving at any speed, from the
//! derivatives of its potential, in spacetime algebra (STA, `x+y+z+t-`, as numga's).
//!
//! The charge is a Plummer-softened point charge: beyond its core a point charge, its density
//! smooth and finite at the centre. Events and the steps between them are antivectors (the
//! lab's time is `-t*`, the step across the slice of space `t`). At rest the potential points
//! along the lab's time slice and falls off as one over the distance, and the gradients of the
//! potential and of the field at an event are maps from steps, `Vector <- Antivector` and
//! `Bivector <- Antivector`, written in closed form. A moving charge is the charge at rest seen
//! through a boost: its gradients are the rest ones with the boost sandwiched around them, the
//! step pulled into the rest frame and the change pushed back out.
//!
//! The derivative of a gradient pairs each step with its vector by incidence alone, `Σₖ eₖ ∧
//! g(Eᵏ)` with `eₗ ∨ Eᵏ = δₗᵏ` (numga's `(Vector ^ g(Antivector)).trace(1, 2)`): of the
//! potential it is the field; of the field it vanishes; and of the excitation the vacuum makes
//! of the field (its Hodge dual) it is the current, an antivector, which the lab's slice reads as
//! the charge density, `t ∨ J`. The metric enters through the vacuum alone, and the distances of
//! the charge's own potential.
//!
//! A point charge circling at constant speed radiates. Its potential at an event is set by where
//! the charge was when it sent the light that reaches the event, the retarded time, and the
//! gradient of that potential is again a closed-form map. The animation speeds the soft charge up
//! to rapidity two and back (left: the lab's electric field strength with the charge's outline,
//! and the magnetic field across the plane, in numga's upright window), and circles point
//! charges at half and nine tenths of the speed of light (right: the magnetic field times the
//! distance, the waves they shed).

use gax::sta::Bivector;
use gax_numga_examples::signal::haversine;
use gax_numga_examples::{
    Align, Anim, Axes, Canvas, Light, Marker, ORIGIN2, Point2, Rect, backdrop, caption, colormap,
    palette, run_with,
};

mod moving_charge {
    // numga's API in full: the tests use parts the animation does not.
    #![cfg_attr(not(test), allow(dead_code))]

    use core::ops::{Add, BitXor};
    use gax::sta::{Bivector, Even, Scalar, Vector, intrinsic};

    pub type Rotor = gax::Unit<Even>;
    /// A step, an event or a current: an antivector (a trivector, intrinsic).
    pub type Antivector = intrinsic::Trivector;
    /// The change in the potential for a small step, `Vector <- Antivector`.
    pub type PotentialGradient = Vector<(Antivector,)>;
    /// The change in the field for a small step, `Bivector <- Antivector`.
    pub type FieldGradient = Bivector<(Antivector,)>;

    pub const T: Vector = Vector::new(0.0, 0.0, 0.0, 1.0);
    pub const X: Vector = Vector::new(1.0, 0.0, 0.0, 0.0);
    pub const Y: Vector = Vector::new(0.0, 1.0, 0.0, 0.0);
    pub const Z: Vector = Vector::new(0.0, 0.0, 1.0, 0.0);
    /// The planes `xt` (the boost along x) and `xy` (the turn from x towards y, as
    /// `exp(-θ/2 xy)`).
    pub const XT: Bivector = Bivector::new(0.0, 0.0, 0.0, 1.0, 0.0, 0.0);
    pub const XY: Bivector = Bivector::new(0.0, 0.0, 1.0, 0.0, 0.0, 0.0);
    /// The retarded time: steps that always close in, by at least the orbit's speed each, then
    /// Newton's.
    const SETTLE_STEPS: usize = 6;
    const NEWTON_STEPS: usize = 6;

    /// One unit along the lab's time: the step across the lab's slice of space `t`, `-t*` (the
    /// Hodge dual, numga's `dual`, carrying the minus of `t`'s square).
    pub fn time() -> Antivector {
        -T.hodge()
    }

    /// The step along x, `x*`, and along y.
    pub fn step_x() -> Antivector {
        X.hodge()
    }
    pub fn step_y() -> Antivector {
        Y.hodge()
    }

    /// The vacuum: from a field to the excitation it causes, the Hodge dual.
    pub fn vacuum(field: Bivector) -> intrinsic::Bivector {
        field.hodge()
    }

    /// A soft charge: its strength, its core's radius, the boost from its rest frame to the lab,
    /// and where it is at lab time zero.
    #[derive(Clone, Copy, Debug)]
    pub struct Charge {
        pub charge: f64,
        pub core_radius: f64,
        pub boost: Rotor,
        pub position: Antivector,
    }

    /// A point charge circling the origin in the `xy` plane at a fraction of the speed of light.
    #[derive(Clone, Copy, Debug)]
    pub struct Orbit {
        pub charge: f64,
        pub radius: f64,
        pub speed: f64,
    }

    /// The open step.
    fn open() -> intrinsic::Trivector<(Antivector,)> {
        Antivector::slot()
    }

    /// An event's place: the event less the lab's time as far as the lab's slice reads it.
    pub fn place(event: Antivector) -> Antivector {
        event - time() * (T & event).s()
    }

    /// The spatial separation of an event from the charge in its rest frame, the field per unit
    /// separation, and that weight's change per unit of `separation · step`.
    struct Profile {
        separation: Antivector,
        weight: f64,
        weight_change: f64,
    }

    fn profile(charge: &Charge, event: Antivector) -> Profile {
        let separation = place(charge.boost << (event - charge.position));
        // A spatial step squares to its length squared.
        let softened = charge.core_radius * charge.core_radius + separation.dot(separation).s();
        #[allow(clippy::disallowed_methods)] // Plummer's softened distance √(a² + r²): a law
        let distance = softened.sqrt();
        let weight = charge.charge / (4.0 * core::f64::consts::PI) / (softened * distance);
        Profile {
            separation,
            weight,
            weight_change: -3.0 * weight / softened,
        }
    }

    /// The change in the potential for a small step from the event: at rest it points along the
    /// lab's time slice and changes with the spatial distance.
    pub fn potential_gradient(charge: &Charge, event: Antivector) -> PotentialGradient {
        let p = profile(charge, event);
        let at_rest = (T * (p.separation | open())).gp(p.weight);
        charge.boost >> at_rest.of(charge.boost << open())
    }

    /// The change in the field for a small step from the event: the radial field changes with
    /// the distance and turns with the step. Each is the plane of a step and time, their join,
    /// which the dual turns into a field.
    pub fn field_gradient(charge: &Charge, event: Antivector) -> FieldGradient {
        let p = profile(charge, event);
        let radial =
            ((p.separation & time()).hodge() * (p.separation | open())).gp(p.weight_change);
        let turning = (open() & time()).hodge().gp(p.weight);
        charge.boost >> (radial + turning).of(charge.boost << open())
    }

    /// The steps dual to the basis vectors by incidence alone, `Eᵏ` with `eₗ ∨ Eᵏ = δₗᵏ`: each
    /// basis vector's complement, scaled to meet it at one.
    fn incidence_frame() -> [(Vector, Antivector); 4] {
        [X, Y, Z, T].map(|e| {
            let complement = e.dual();
            (e, complement * (1.0 / (e & complement).s()))
        })
    }

    /// The derivative of a gradient: each step's vector wedged with the gradient's change along
    /// it, `Σₖ eₖ ∧ g(Eᵏ)` over the steps dual to the basis by incidence, numga's
    /// `(Vector ^ g(Antivector)).trace(1, 2)`. gax's traces pair a map's output with one of its
    /// slots; this one pairs the open vector with the open step, so it is written out.
    pub fn derivative<G, R>(gradient: impl Fn(Antivector) -> G) -> R
    where
        Vector: BitXor<G, Output = R>,
        R: Add<Output = R>,
    {
        incidence_frame()
            .map(|(e, step)| e ^ gradient(step))
            .into_iter()
            .reduce(Add::add)
            .expect("four basis vectors")
    }

    /// The derivative of the potential: the field.
    pub fn field(gradient: PotentialGradient) -> Bivector {
        derivative(|step| gradient.of(step))
    }

    /// The derivative of the excitation the vacuum makes of the field: the current, an
    /// antivector, Maxwell's equation with sources.
    pub fn current(gradient: FieldGradient) -> Antivector {
        derivative(|step| vacuum(gradient.of(step)))
    }

    /// The circling charge at lab time `time`, its change per unit lab time, and that change's
    /// change: the offset turned in the orbit's plane, moving under that turn by its commutator
    /// with the plane.
    pub fn worldline(orbit: &Orbit, lab_time: f64) -> (Antivector, Antivector, Antivector) {
        let turning = orbit.speed / orbit.radius;
        let offset = (XY * (-0.5 * turning * lab_time)).exp() >> (step_x() * orbit.radius);
        let tangent = time() + offset.commutator(XY) * turning;
        let bend = (tangent - time()).commutator(XY) * turning;
        (time() * lab_time + offset, tangent, bend)
    }

    /// The lab time at which the charge sent the light that reaches the event: a few steps that
    /// move the time back by the distance light covers, which always close in, then Newton's.
    pub fn retarded(orbit: &Orbit, event: Antivector) -> f64 {
        let now = (T & event).s();
        // How far light from the charge at `time` falls short of the event, its rate of change,
        // and the distance it covers.
        let light = |lab_time: f64| {
            let (position, tangent, _) = worldline(orbit, lab_time);
            let across = place(event - position);
            let distance = across.norm();
            let rate = 1.0 - (across | tangent).s() / distance;
            (now - lab_time - distance, rate, distance)
        };
        let mut lab_time = now;
        for _ in 0..SETTLE_STEPS {
            lab_time = now - light(lab_time).2;
        }
        for _ in 0..NEWTON_STEPS {
            let (shortfall, rate, _) = light(lab_time);
            lab_time += shortfall / rate;
        }
        lab_time
    }

    /// The change in the circling charge's potential for a small step from the event: the
    /// Liénard–Wiechert potential's gradient, from the charge's velocity and acceleration at the
    /// retarded time and the light ray from it.
    pub fn orbit_potential_gradient(orbit: &Orbit, event: Antivector) -> PotentialGradient {
        let (position, tangent, bend) = worldline(orbit, retarded(orbit, event));
        #[allow(clippy::disallowed_methods)] // the time dilation 1/√(1 - v²): a law
        let dilation = 1.0 / (1.0 - orbit.speed * orbit.speed).sqrt();
        let (velocity, acceleration) = (tangent * dilation, bend * (dilation * dilation));
        let ray = event - position;
        // The charge's own slice of space, which reads its velocity as one: minus the velocity's
        // dual, as for the lab's `time`; and how the slice turns with the acceleration.
        let (rest_slice, slice_change) = (-velocity.hodge(), -acceleration.hodge());
        // The distance as the charge saw it when it sent the light: the ray's extent, as its own
        // slice reads it.
        let rest_distance = (rest_slice & ray).s();
        // A step moves the emission along the ray by what the ray's slice, the light cone's
        // tangent, reads of it; that changes the slice and the distance.
        let emission: Scalar<(Antivector,)> = (-ray.hodge() & open()) * (1.0 / rest_distance);
        let distance_change = (rest_slice & open()) + emission * ((slice_change & ray).s() - 1.0);
        let weight = orbit.charge / (4.0 * core::f64::consts::PI) / rest_distance;
        // As at rest, the potential is minus the charge's slice, here over the distance.
        (slice_change * emission - rest_slice * distance_change * (1.0 / rest_distance)) * -weight
    }

    /// Events at lab time zero on a grid in the plane `z = 0`, at the centres of its cells, row
    /// by row.
    pub fn grid(half_width: f64, half_height: f64, columns: usize, rows: usize) -> Vec<Antivector> {
        let centre =
            |i: usize, n: usize, half: f64| ((i as f64 + 0.5) / n as f64 * 2.0 - 1.0) * half;
        (0..rows)
            .flat_map(|r| {
                (0..columns).map(move |c| {
                    step_x() * centre(c, columns, half_width)
                        + step_y() * centre(r, rows, half_height)
                })
            })
            .collect()
    }

    // --- readings -------------------------------------------------------------------------

    /// The strength of the electric field the lab reads: the size of the field met with the
    /// lab's time.
    pub fn electric_strength(field: Bivector) -> f64 {
        (field & time()).norm()
    }

    /// The lab's magnetic field across the plane: the field read on the patch of the plane of x
    /// and y.
    pub fn magnetic_across(field: Bivector) -> f64 {
        (field & XY.hodge()).s()
    }

    /// The charge density: the current crossing the lab's time slice.
    pub fn density(current: Antivector) -> f64 {
        (T & current).s()
    }

    /// An event's distance from the origin in the lab's space.
    pub fn distance(event: Antivector) -> f64 {
        place(event).norm()
    }

    // --- scenes ---------------------------------------------------------------------------

    pub const CHARGE: f64 = 1.0;
    pub const CORE_RADIUS: f64 = 0.08;
    pub const TOP_RAPIDITY: f64 = 2.0;
    /// The lab at time zero, an upright window about the charge, which runs across it, so that
    /// its field gathers into an upright band.
    pub const HALF_WIDTH: f64 = 0.45;
    pub const HALF_HEIGHT: f64 = 0.9;
    /// The coarser window the checks sample.
    pub const CHECK_COLUMNS: usize = 30;
    pub const CHECK_ROWS: usize = 20;
    pub const ORBIT_CHARGE: f64 = 1.0;
    pub const ORBIT_RADIUS: f64 = 1.0;
    pub const ORBIT_SPEEDS: [(&str, f64); 2] = [("SLOW", 0.5), ("FAST", 0.9)];
    pub const ORBIT_HALF_WIDTH: f64 = 10.0;

    /// The charge moving along x at the rapidity, through the origin at lab time zero.
    pub fn charge(rapidity: f64) -> Charge {
        Charge {
            charge: CHARGE,
            core_radius: CORE_RADIUS,
            boost: (XT * (rapidity / 2.0)).exp(),
            position: Antivector::zero(),
        }
    }

    /// A circling charge at a fraction of the speed of light.
    pub fn orbit(speed: f64) -> Orbit {
        Orbit {
            charge: ORBIT_CHARGE,
            radius: ORBIT_RADIUS,
            speed,
        }
    }

    /// The period of a turn of the orbit.
    pub fn period(orbit: &Orbit) -> f64 {
        core::f64::consts::TAU * orbit.radius / orbit.speed
    }
}

use moving_charge::*;

/// One rise to the top speed and back, and one turn of each orbit.
const SECONDS: f32 = 6.0;
/// The electric field strengths the colours span, logarithmically, and the magnetic field at
/// which they saturate.
const ELECTRIC_RANGE: [f64; 2] = [0.05, 50.0];
const MAGNETIC_LIMIT: f64 = 10.0;
/// The density of the charge's outline, and the samples across its panel it is traced from.
const DENSITY_LEVEL: f32 = 5.0;
const OUTLINE_SAMPLES: usize = 100;
/// The waves' colour scale: from the magnetic field times the distance beyond `WAVE_ZONE` from
/// the orbit's centre, its percentile `WAVE_PERCENTILE`, the sharpest crests saturating; and a
/// power on the scaled field. numga takes 0.4, on a colour map white at zero, to keep both a fast
/// charge's narrow pulses and the field between them; here the light grows from the dark
/// backdrop, and a power above one keeps the weak field between the pulses dim.
const WAVE_ZONE: f64 = 4.0;
const WAVE_PERCENTILE: f64 = 0.99;
const WAVE_POWER: f32 = 2.0;
/// Samples across the grid on which a wave's colour scale is measured.
const WAVE_SAMPLES: usize = 120;

/// A point of the plane, as drawn: the event's readings on the slices x and y.
fn planar(event: Antivector) -> Point2 {
    Point2::xy((X & event).s() as f32, (Y & event).s() as f32)
}

/// The event at lab time `time` under a point of the plane.
fn event(p: Point2, time: f64) -> Antivector {
    let [x, y] = p.to_euclidean().map(f64::from);
    step_x() * x + step_y() * y + moving_charge::time() * time
}

/// A signed value in `[-1, 1]` as light: orange for positive, sky for negative, glowing with its
/// size.
fn signed_light(v: f32) -> Light {
    let base = if v > 0.0 {
        palette::ORANGE
    } else {
        palette::SKY
    };
    base.faded(v.abs().min(1.0))
}

/// An electric field strength as light: through inferno's embers, logarithmically from the
/// bottom of the range.
fn strength_light(strength: f64) -> Light {
    let [low, high] = ELECTRIC_RANGE;
    let t = (strength / low).ln() / (high / low).ln();
    colormap::inferno(t.clamp(0.0, 1.0) as f32)
}

/// A narrow panel's frame: its outline and its title, without ticks, which would crowd it (as
/// numga's window, a picture of the field, has none).
fn titled(c: &mut Canvas, rect: Rect, title: &str) {
    let size = (rect.height() / 26.0).clamp(3.0, 14.0);
    let corners = [rect.lo, rect.top_right(), rect.hi, rect.bottom_left()];
    c.polyline(&corners, 1.0, palette::GRID, true);
    let above = rect.top_middle() + Point2::direction(0.0, -size * 0.6);
    c.text(title, above, size * 1.1, palette::INK, Align::Center);
}

/// The panel showing the soft charge's upright window.
fn charge_axes(rect: Rect) -> Axes {
    Axes::equal(rect, ORIGIN2, HALF_HEIGHT as f32)
}

/// The moving charge's field at lab time zero under a point of the plane.
fn lab_field(moving: &Charge, p: Point2) -> Bivector {
    field(potential_gradient(moving, event(p, 0.0)))
}

/// The electric field strength of the charge moving at `rapidity`, with its outline where its
/// density is high.
fn electric_panel(c: &mut Canvas, rect: Rect, moving: &Charge, rapidity: f64) {
    let ax = charge_axes(rect);
    ax.image(c, 1, |p| {
        Some(strength_light(electric_strength(lab_field(moving, p))))
    });
    let outline = |p: Point2| density(current(field_gradient(moving, event(p, 0.0)))).abs() as f32;
    ax.contour(
        c,
        outline,
        OUTLINE_SAMPLES,
        DENSITY_LEVEL,
        1.4,
        palette::YELLOW,
    );
    titled(c, rect, &format!("E AT {:.2} C", rapidity.tanh()));
}

/// The magnetic field across the plane of the moving charge.
fn magnetic_panel(c: &mut Canvas, rect: Rect, moving: &Charge) {
    let ax = charge_axes(rect);
    ax.image(c, 1, |p| {
        let across = magnetic_across(lab_field(moving, p)) / MAGNETIC_LIMIT;
        Some(signed_light(across as f32))
    });
    titled(c, rect, "B ACROSS");
}

/// The magnetic field across the plane times the distance from the orbit's centre, so that the
/// waves keep their strength across the picture.
fn scaled_waves(orbit: &Orbit, event: Antivector) -> f64 {
    let field = field(orbit_potential_gradient(orbit, event));
    magnetic_across(field) * distance(event)
}

/// The colour scale of an orbit's waves over a whole turn: their percentile far from the orbit
/// at lab time zero, the sharpest crests of a fast charge's beam saturating.
fn wave_limit(orbit: &Orbit) -> f64 {
    let mut far: Vec<f64> = grid(
        ORBIT_HALF_WIDTH,
        ORBIT_HALF_WIDTH,
        WAVE_SAMPLES,
        WAVE_SAMPLES,
    )
    .into_iter()
    .filter(|e| distance(*e) > WAVE_ZONE)
    .map(|e| scaled_waves(orbit, e).abs())
    .collect();
    far.sort_by(f64::total_cmp);
    far[((far.len() - 1) as f64 * WAVE_PERCENTILE) as usize]
}

/// A circling charge as drawn: its name, its orbit, and its waves' colour scale.
struct Radiator {
    name: &'static str,
    orbit: Orbit,
    limit: f64,
}

/// The two circling charges, computed once.
fn radiators() -> [Radiator; 2] {
    ORBIT_SPEEDS.map(|(name, speed)| {
        let orbit = orbit(speed);
        Radiator {
            name,
            limit: wave_limit(&orbit),
            orbit,
        }
    })
}

/// A circling charge's waves at a share `turned` of its turn, with the charge marked.
fn wave_panel(c: &mut Canvas, rect: Rect, radiator: &Radiator, turned: f64) {
    let Radiator { name, orbit, limit } = radiator;
    let time = turned * period(orbit);
    let ax = Axes::equal(rect, ORIGIN2, ORBIT_HALF_WIDTH as f32);
    ax.image(c, 1, |p| {
        let scaled = (scaled_waves(orbit, event(p, time)) / limit) as f32;
        Some(signed_light(
            scaled.signum() * scaled.abs().powf(WAVE_POWER),
        ))
    });
    let (position, _, _) = worldline(orbit, time);
    ax.scatter(c, &[planar(position)], Marker::Dot, 7.0, palette::YELLOW);
    let title = format!("{name}: {:.1} C", orbit.speed);
    ax.frame(c, &title, "", "");
}

/// A rectangle of the given aspect (width over height), as large as fits in `room`, centred.
fn fitted(room: Rect, aspect: f32) -> Rect {
    let height = room.height().min(room.width() / aspect);
    let half = Point2::direction(height * aspect / 2.0, height / 2.0);
    Rect::around(room.centre(), half)
}

fn draw(c: &mut Canvas, radiators: &[Radiator; 2], t: f32) {
    backdrop(c);
    let share = f64::from(t / SECONDS).rem_euclid(1.0);
    let rapidity = TOP_RAPIDITY * haversine(core::f64::consts::TAU * share);
    let screen = c.rect();
    let (w, h) = (screen.width(), screen.height());
    let top = (h / 30.0).clamp(10.0, 22.0) * 3.6;
    // Left, the soft charge's upright window twice, its electric and its magnetic field side by
    // side; right, the two circling charges.
    let split = 0.46;
    let left = Rect::new(0.0, top, w * split, h);
    let moving = charge(rapidity);
    let aspect = (HALF_WIDTH / HALF_HEIGHT) as f32;
    let room = |i: usize| left.column(i, 2).inset(14.0, 24.0, 4.0, 12.0);
    electric_panel(c, fitted(room(0), aspect), &moving, rapidity);
    magnetic_panel(c, fitted(room(1), aspect), &moving);
    let right = Rect::new(w * split, top, w, h);
    for (i, radiator) in radiators.iter().enumerate() {
        let panel = right.column(i, 2).inset(30.0, 20.0, 8.0, 22.0);
        wave_panel(c, fitted(panel, 1.0), radiator, share);
    }
    caption(
        c,
        "MOVING CHARGES: FIELDS FROM THE DERIVATIVES OF POTENTIALS",
        "A SOFT CHARGE SPED UP AND SLOWED; POINT CHARGES CIRCLING AT 0.5 AND 0.9 C SHED WAVES (STA)",
    );
}

fn main() {
    run_with(
        Anim::new("moving_charge", SECONDS).size(960, 540),
        radiators,
        draw,
    );
}

#[cfg(test)]
mod tests {
    use super::moving_charge::*;
    use gax::sta::{Bivector, Vector, intrinsic};
    use gax_numga_examples::testing::near;

    #[test]
    fn maxwell_by_incidence_with_the_metric_in_the_vacuum() {
        let moving = charge(TOP_RAPIDITY);
        let (step, size) = (step_x() + step_y() * 0.4 - time() * 0.7, 1e-6);
        for event in grid(HALF_WIDTH, HALF_HEIGHT, 15, 10) {
            // The field's derivative vanishes.
            let gradients = field_gradient(&moving, event);
            let curl: gax::sta::Trivector = derivative(|e| gradients.of(e));
            assert!(near(curl, gax::sta::Trivector::zero(), 1e-12));
            // The field's gradient is the derivative of the potential's field along each step.
            let ahead = field(potential_gradient(&moving, event + step * size));
            let behind = field(potential_gradient(&moving, event - step * size));
            assert!(near(
                gradients.of(step),
                (ahead - behind) * (1.0 / (2.0 * size)),
                1e-5
            ));
            // The potential keeps the Lorenz gauge: the derivative of its dual vanishes.
            let potentials = potential_gradient(&moving, event);
            let gauge: intrinsic::Pseudoscalar = derivative(|e| potentials.of(e).hodge());
            assert!(near(gauge, intrinsic::Pseudoscalar::zero(), 1e-12));
            // The current is the Plummer density carried along the charge's time direction, and
            // positive.
            let separation = place(moving.boost << event);
            let softened = CORE_RADIUS * CORE_RADIUS + separation.dot(separation).s();
            #[allow(clippy::disallowed_methods)] // the Plummer density it is checked against
            let expected = 3.0 * CHARGE * CORE_RADIUS * CORE_RADIUS
                / (4.0 * core::f64::consts::PI)
                / (softened * softened * softened.sqrt());
            let current = current(gradients);
            assert!(near(current, (moving.boost >> time()) * expected, 1e-9));
            assert!(density(current) > 0.0);
        }
    }

    #[test]
    fn the_circling_charge_is_seen_along_light_rays() {
        let fast = orbit(ORBIT_SPEEDS[1].1);
        for spot in grid(ORBIT_HALF_WIDTH, ORBIT_HALF_WIDTH, 12, 12) {
            let event = spot + time() * 3.0;
            // The light from the charge at the retarded time reaches each event.
            let (position, _, _) = worldline(&fast, retarded(&fast, event));
            let ray = event - position;
            assert!(ray.dot(ray).s().abs() < 1e-10);
            // The radiating potential keeps the Lorenz gauge.
            let potentials = orbit_potential_gradient(&fast, event);
            let gauge: intrinsic::Pseudoscalar = derivative(|e| potentials.of(e).hodge());
            assert!(near(gauge, intrinsic::Pseudoscalar::zero(), 1e-12));
        }
        // Slow and far away, the circling charge has the field of a charge at rest, pointing
        // away from it.
        let slow = orbit(1e-3);
        let distance = 29.0;
        let far = step_x() * (ORBIT_RADIUS + distance);
        let outward = (field(orbit_potential_gradient(&slow, far)) & (step_x() & time())).s();
        let expected = ORBIT_CHARGE / (4.0 * core::f64::consts::PI * distance * distance);
        assert!(
            ((outward - expected) / expected).abs() < 1e-2,
            "{outward} {expected}"
        );
    }

    /// numga's scenario checks: at the top speed, on the coarser window, the field's derivative
    /// vanishes and the current is the softened density carried along the charge's own time.
    #[test]
    fn scenario_checks() {
        let moving = charge(TOP_RAPIDITY);
        for event in grid(HALF_WIDTH, HALF_HEIGHT, CHECK_COLUMNS, CHECK_ROWS) {
            let gradients = field_gradient(&moving, event);
            let curl: gax::sta::Trivector = derivative(|e| gradients.of(e));
            assert!(near(curl, gax::sta::Trivector::zero(), 1e-12));
            let separation = place(moving.boost << event);
            let softened = CORE_RADIUS * CORE_RADIUS + separation.dot(separation).s();
            #[allow(clippy::disallowed_methods)] // the Plummer density it is checked against
            let expected = 3.0 * CHARGE * CORE_RADIUS * CORE_RADIUS
                / (4.0 * core::f64::consts::PI)
                / (softened * softened * softened.sqrt());
            assert!(near(
                current(gradients),
                (moving.boost >> time()) * expected,
                1e-9
            ));
        }
    }

    /// numga 35797de's values at a few events: the soft charge's field at rapidity one, its
    /// current near the centre, and the fast orbit's worldline, retarded time and field.
    #[test]
    fn fields_match_numga() {
        let moving = charge(1.0);
        let at = step_x() * 0.3 + step_y() * 0.2;
        let expected = Bivector::new(0.0, 0.0, 0.140_515_357, 0.276_752_433, 0.184_501_622, 0.0);
        assert!(near(field(potential_gradient(&moving, at)), expected, 1e-8));
        // Its potential's gradient at a later event, along time and along x.
        let later = step_x() * 2.0 - step_y() * 1.5 + time() * 3.0;
        let gradient = potential_gradient(&moving, later);
        let along_time = Vector::new(-0.012_597_888_199, 0.0, 0.0, 0.016_541_471_728);
        let along_x = Vector::new(0.016_541_471_728, 0.0, 0.0, -0.021_719_536_053);
        assert!(near(gradient.of(time()), along_time, 1e-10));
        assert!(near(gradient.of(step_x()), along_x, 1e-10));
        let near_centre = step_x() * 0.02 + step_y() * 0.01;
        let current = current(field_gradient(&moving, near_centre));
        let expected = Antivector::new(-491.738_136_17, 374.504_890_77, 0.0, 0.0);
        assert!(near(current, expected, 1e-5));
        let fast = orbit(0.9);
        let (position, tangent, bend) = worldline(&fast, 0.7);
        assert!(near(
            position,
            Antivector::new(-0.7, 0.808_027_508_312, 0.589_144_757_942, 0.0),
            1e-10
        ));
        assert!(near(
            tangent,
            Antivector::new(-1.0, -0.530_230_282_148, 0.727_224_757_481, 0.0),
            1e-10
        ));
        assert!(near(
            bend,
            Antivector::new(0.0, -0.654_502_281_733, -0.477_207_253_933, 0.0),
            1e-10
        ));
        let event = step_x() * 2.0 - step_y() * 1.5 + time() * 3.0;
        assert!((retarded(&fast, event) - 0.644_491_497_036).abs() < 1e-8);
        let expected = Bivector::new(
            0.0,
            0.0,
            0.007_616_599_861,
            0.006_996_047_348,
            0.003_104_819_486,
            0.0,
        );
        assert!(near(
            field(orbit_potential_gradient(&fast, event)),
            expected,
            1e-8
        ));
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws_with(super::draw, &super::radiators(), 1.0);
    }
}
