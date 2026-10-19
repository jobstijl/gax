//! numga's `electromagnetism/moving_charge`: the field of a charge moving at any speed, from the
//! derivatives of its potential, in spacetime algebra (STA, `t² = 1`, `x² = y² = z² = -1`).
//!
//! The charge is a Plummer-softened point charge: beyond its core a point charge, its density
//! smooth and finite at the centre. At rest its potential points along its time direction and
//! falls off as one over the distance, and the gradients of the potential and of the field at an
//! event are maps on displacements, `Vector <- Vector` and `Bivector <- Vector`, written in closed
//! form. A moving charge is the charge at rest seen through a boost: its gradients are the rest
//! ones with the boost sandwiched around them, the displacement pulled into the rest frame and
//! the change pushed back out.
//!
//! Contracting a gradient against an open vector through the metric, `Σᵢ eⁱ g(eᵢ)` (numga's
//! `(Vector * g(Vector)).contract(1, 2)`, written out over the reciprocal frame), gives the
//! derivative: of the potential, the field plus the Lorenz-gauge scalar, which vanishes; of the
//! field, the current plus a trivector, which vanishes. Both halves of Maxwell's equations also
//! follow without a metric, from the incidence trace of the field's gradient against the
//! antivectors dual to the basis by the regressive product alone: on the field it is the
//! current, on its Hodge dual (numga's `dual`, gax's `hodge`) it vanishes.
//!
//! A point charge circling at constant speed radiates. Its potential at an event is set by where
//! the charge was when it sent the light that reaches the event, the retarded time, and the
//! gradient of that potential is again a closed-form map. The animation speeds the soft charge up
//! to rapidity two and back (left: the lab's electric field strength with the charge's outline,
//! and the magnetic field across the plane), and circles point charges at half and nine tenths
//! of the speed of light (right: the magnetic field times the distance, the waves they shed).

use gax::sta::{Bivector, Vector};
use gax_numga_examples::signal::haversine;
use gax_numga_examples::{
    Anim, Axes, Canvas, Light, Marker, ORIGIN2, Point2, Rect, backdrop, caption, colormap, palette,
    run,
};
use std::sync::OnceLock;

mod moving_charge {
    use core::ops::{Add, BitAnd};
    use gax::sta::{Bivector, Even, Odd, Scalar, Vector, intrinsic};

    pub type Rotor = gax::Unit<Even>;
    /// The change in the potential for a small displacement, `Vector <- Vector`.
    pub type PotentialGradient = Vector<(Vector,)>;
    /// The change in the field for a small displacement, `Bivector <- Vector`.
    pub type FieldGradient = Bivector<(Vector,)>;

    pub const T: Vector = Vector::new(1.0, 0.0, 0.0, 0.0);
    pub const X: Vector = Vector::new(0.0, 1.0, 0.0, 0.0);
    pub const Y: Vector = Vector::new(0.0, 0.0, 1.0, 0.0);
    pub const Z: Vector = Vector::new(0.0, 0.0, 0.0, 1.0);
    /// numga's planes `xt` and `xy`, gax's `e10` and `e12`.
    pub const XT: Bivector = Bivector::new(1.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    pub const XY: Bivector = Bivector::new(0.0, 0.0, 0.0, 0.0, 0.0, 1.0);
    /// The retarded time: steps that always close in, by at least the orbit's speed each, then
    /// Newton's.
    const SETTLE_STEPS: usize = 6;
    const NEWTON_STEPS: usize = 6;

    /// A soft charge: its strength, its core's radius, the boost from its rest frame to the lab,
    /// and where it is at lab time zero.
    #[derive(Clone, Copy, Debug)]
    pub struct Charge {
        pub charge: f64,
        pub core_radius: f64,
        pub boost: Rotor,
        pub position: Vector,
    }

    /// A point charge circling the origin in the `xy` plane at a fraction of the speed of light.
    #[derive(Clone, Copy, Debug)]
    pub struct Orbit {
        pub charge: f64,
        pub radius: f64,
        pub speed: f64,
    }

    /// The open vector: a displacement left open.
    fn open() -> PotentialGradient {
        Vector::slot()
    }

    /// A vector's part across the lab's time: its spatial part.
    pub fn spatial(v: Vector) -> Vector {
        v - T * (v | T).s()
    }

    /// The spatial separation of an event from the charge in its rest frame, the field per unit
    /// separation, and that weight's change per unit of `separation · displacement`.
    struct Profile {
        separation: Vector,
        weight: f64,
        weight_change: f64,
    }

    fn profile(charge: &Charge, event: Vector) -> Profile {
        let separation = spatial(charge.boost << (event - charge.position));
        let softened = charge.core_radius * charge.core_radius - separation.norm_squared();
        #[allow(clippy::disallowed_methods)] // Plummer's softened distance √(a² + r²): a law
        let distance = softened.sqrt();
        let weight = charge.charge / (4.0 * core::f64::consts::PI) / (softened * distance);
        Profile {
            separation,
            weight,
            weight_change: 3.0 * weight / softened,
        }
    }

    /// The change in the potential for a small displacement from the event: at rest it points
    /// along the rest time and changes with the spatial distance.
    pub fn potential_gradient(charge: &Charge, event: Vector) -> PotentialGradient {
        let p = profile(charge, event);
        let at_rest = (T * (p.separation | open())) * p.weight;
        charge.boost >> at_rest.of(charge.boost << open())
    }

    /// The change in the field for a small displacement from the event: the radial field changes
    /// with the distance and turns with the displacement.
    pub fn field_gradient(charge: &Charge, event: Vector) -> FieldGradient {
        let p = profile(charge, event);
        let radial = ((p.separation ^ T) * (p.separation | open())) * p.weight_change;
        let turning = (open() ^ T) * p.weight;
        charge.boost >> (radial + turning).of(charge.boost << open())
    }

    /// The basis vectors with their reciprocals: `eⁱ · eⱼ = δᵢⱼ`.
    fn reciprocal_frame() -> [(Vector, Vector); 4] {
        [T, X, Y, Z].map(|e| (e, e.inverse()))
    }

    /// A gradient traced against a frame dual to the basis, `Σᵢ eⁱ ∘ g(eᵢ)` for each basis
    /// vector `eᵢ` paired with its dual `eⁱ`, and the product `∘`.
    fn traced<D, G, R: Add<Output = R>>(
        frame: [(Vector, D); 4],
        gradient: impl Fn(Vector) -> G,
        product: impl Fn(D, G) -> R,
    ) -> R {
        frame
            .map(|(e, dual)| product(dual, gradient(e)))
            .into_iter()
            .reduce(Add::add)
            .expect("four basis vectors")
    }

    /// A gradient contracted with an open vector through the metric, `Σᵢ eⁱ ∘ g(eᵢ)` over the
    /// reciprocal frame: numga's `(Vector ∘ g(Vector)).contract(1, 2)`.
    fn contracted<G, R: Add<Output = R>>(
        gradient: impl Fn(Vector) -> G,
        product: impl Fn(Vector, G) -> R,
    ) -> R {
        traced(reciprocal_frame(), gradient, product)
    }

    /// The potential's derivative: the field plus the Lorenz-gauge scalar.
    #[cfg_attr(not(test), allow(dead_code))] // the tests check the gauge; the animation needs the field
    pub fn potential_derivative(gradient: PotentialGradient) -> Even {
        contracted(|e| gradient.of(e), |a, b| a * b)
    }

    /// The field: the bivector part of the potential's derivative.
    pub fn field(gradient: PotentialGradient) -> Bivector {
        contracted(|e| gradient.of(e), |a, b| a ^ b)
    }

    /// The field's derivative: the current plus a trivector that vanishes, Maxwell's equations
    /// through the metric.
    #[cfg_attr(not(test), allow(dead_code))] // the tests check it against the metric-free trace
    pub fn field_derivative(gradient: FieldGradient) -> Odd {
        contracted(|e| gradient.of(e), |a, b| a * b)
    }

    /// An antivector: the complement of a vector, of the other orientation (intrinsic).
    type Antivector = intrinsic::Trivector;

    /// The antivectors dual to the basis by incidence alone, `Eⁱ` with `eⱼ & Eⁱ = δᵢⱼ`: each
    /// basis vector's complement, scaled to meet it at one.
    fn incidence_frame() -> [(Vector, Antivector); 4] {
        [T, X, Y, Z].map(|e| {
            let complement = e.dual();
            (e, complement * (1.0 / (e & complement).s()))
        })
    }

    /// The incidence trace of a bivector-valued gradient: `Σᵢ g(eᵢ) & Eⁱ`, with no metric. Of
    /// a field it is a vector, of a dual field an intrinsic one.
    fn incidence_trace<G: BitAnd<Antivector, Output = R>, R: Add<Output = R>>(
        gradient: impl Fn(Vector) -> G,
    ) -> R {
        traced(incidence_frame(), gradient, |dual, g| g & dual)
    }

    /// Maxwell's equations without a metric: the incidence trace of the field's gradient, the
    /// current, and of its dual's, the magnetic source (of the dual's orientation), which
    /// vanishes.
    pub fn sources(gradient: FieldGradient) -> (Vector, intrinsic::Vector) {
        let current = incidence_trace(|e| gradient.of(e));
        let magnetic = incidence_trace(|e| gradient.of(e).hodge());
        (current, magnetic)
    }

    /// The circling charge at lab time `time`, its change per unit lab time, and that change's
    /// change.
    pub fn worldline(orbit: &Orbit, time: f64) -> (Vector, Vector, Vector) {
        let turning = orbit.speed / orbit.radius;
        let offset = (XY * (-0.5 * turning * time)).exp() >> (X * orbit.radius);
        let tangent = T + (offset | XY) * turning;
        let bend = ((tangent - T) | XY) * turning;
        (T * time + offset, tangent, bend)
    }

    /// The lab time at which the charge sent the light that reaches the event: a few steps that
    /// move the time back by the distance light covers, which always close in, then Newton's.
    pub fn retarded(orbit: &Orbit, event: Vector) -> f64 {
        let now = (event | T).s();
        // How far light from the charge at `time` falls short of the event, its rate of change,
        // and the distance it covers.
        let light = |time: f64| {
            let (position, tangent, _) = worldline(orbit, time);
            let across = spatial(event - position);
            let distance = across.norm();
            let rate = 1.0 + (across | tangent).s() / distance;
            (now - time - distance, rate, distance)
        };
        let mut time = now;
        for _ in 0..SETTLE_STEPS {
            time = now - light(time).2;
        }
        for _ in 0..NEWTON_STEPS {
            let (shortfall, rate, _) = light(time);
            time += shortfall / rate;
        }
        time
    }

    /// The change in the circling charge's potential for a small displacement from the event:
    /// the Liénard–Wiechert potential's gradient, from the charge's velocity and acceleration
    /// at the retarded time and the light ray from it.
    pub fn orbit_potential_gradient(orbit: &Orbit, event: Vector) -> PotentialGradient {
        let (position, tangent, bend) = worldline(orbit, retarded(orbit, event));
        // The tangent squares to `1 - v²`: one over its norm is the time dilation.
        let dilation = 1.0 / tangent.norm();
        let (velocity, acceleration) = (tangent * dilation, bend * (dilation * dilation));
        let ray = event - position;
        let reach = (ray | velocity).s();
        // A displacement moves the emission along the ray, which changes the velocity and reach.
        let emission: Scalar<(Vector,)> = (ray | open()) * (1.0 / reach);
        let reach_change = (velocity | open()) + emission * ((ray | acceleration).s() - 1.0);
        let weight = orbit.charge / (4.0 * core::f64::consts::PI) / reach;
        (acceleration * emission - velocity * reach_change * (1.0 / reach)) * weight
    }

    /// Events at lab time zero on a grid in the plane `z = 0`, at the centres of its cells, row
    /// by row.
    pub fn grid(half_width: f64, half_height: f64, columns: usize, rows: usize) -> Vec<Vector> {
        let centre =
            |i: usize, n: usize, half: f64| ((i as f64 + 0.5) / n as f64 * 2.0 - 1.0) * half;
        (0..rows)
            .flat_map(|r| {
                (0..columns).map(move |c| {
                    X * centre(c, columns, half_width) + Y * centre(r, rows, half_height)
                })
            })
            .collect()
    }

    // --- scenes ---------------------------------------------------------------------------

    pub const CHARGE: f64 = 1.0;
    pub const CORE_RADIUS: f64 = 0.08;
    pub const TOP_RAPIDITY: f64 = 2.0;
    pub const HALF_HEIGHT: f64 = 1.0;
    pub const ORBIT_CHARGE: f64 = 1.0;
    pub const ORBIT_RADIUS: f64 = 1.0;
    pub const ORBIT_SPEEDS: [(&str, f64); 2] = [("SLOW", 0.5), ("FAST", 0.9)];
    pub const ORBIT_HALF_WIDTH: f64 = 10.0;

    /// The charge moving along `x` at the rapidity, through the origin at lab time zero.
    pub fn charge(rapidity: f64) -> Charge {
        Charge {
            charge: CHARGE,
            core_radius: CORE_RADIUS,
            boost: (XT * (rapidity / 2.0)).exp(),
            position: Vector::zero(),
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

    /// The lab's electric field strength: the length of the field's part along the lab's time.
    pub fn electric_strength(field: Bivector) -> f64 {
        (field | T).norm()
    }

    /// The lab's magnetic field across the plane: its dual's part along the lab's time, along `z`.
    pub fn magnetic_across(field: Bivector) -> f64 {
        (field.hodge() | T).e3()
    }

    /// The charge density: the current's part along the lab's time.
    pub fn density(current: Vector) -> f64 {
        (current | T).s()
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

/// A point of the plane, as drawn.
fn planar(v: Vector) -> Point2 {
    Point2::xy(v.e1() as f32, v.e2() as f32)
}

/// The event at lab time `time` under a point of the plane.
fn event(p: Point2, time: f64) -> Vector {
    let [x, y] = p.to_euclidean().map(f64::from);
    T * time + X * x + Y * y
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

/// The panel showing the soft charge's plane at its aspect.
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
    let outline = |p: Point2| {
        let (current, _) = sources(field_gradient(moving, event(p, 0.0)));
        density(current).abs() as f32
    };
    ax.contour(
        c,
        outline,
        OUTLINE_SAMPLES,
        DENSITY_LEVEL,
        1.4,
        palette::YELLOW,
    );
    let title = format!("ELECTRIC FIELD AT {:.2} C", rapidity.tanh());
    ax.frame(c, &title, "", "");
}

/// The magnetic field across the plane of the moving charge.
fn magnetic_panel(c: &mut Canvas, rect: Rect, moving: &Charge) {
    let ax = charge_axes(rect);
    ax.image(c, 1, |p| {
        let across = magnetic_across(lab_field(moving, p)) / MAGNETIC_LIMIT;
        Some(signed_light(across as f32))
    });
    ax.frame(c, "MAGNETIC FIELD ACROSS THE PLANE", "", "");
}

/// The magnetic field across the plane times the distance from the orbit's centre, so that the
/// waves keep their strength across the picture.
fn scaled_waves(orbit: &Orbit, event: Vector) -> f64 {
    let field = field(orbit_potential_gradient(orbit, event));
    magnetic_across(field) * spatial(event).norm()
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
    .filter(|e| e.norm() > WAVE_ZONE)
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
fn radiators() -> &'static [Radiator; 2] {
    static RADIATORS: OnceLock<[Radiator; 2]> = OnceLock::new();
    RADIATORS.get_or_init(|| {
        ORBIT_SPEEDS.map(|(name, speed)| {
            let orbit = orbit(speed);
            Radiator {
                name,
                limit: wave_limit(&orbit),
                orbit,
            }
        })
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

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let share = f64::from(t / SECONDS).rem_euclid(1.0);
    let rapidity = TOP_RAPIDITY * haversine(core::f64::consts::TAU * share);
    let screen = c.rect();
    let (w, h) = (screen.width(), screen.height());
    let top = (h / 30.0).clamp(10.0, 22.0) * 3.6;
    let left = Rect::new(0.0, top, w * 0.4, h);
    let moving = charge(rapidity);
    let upper = left.part(0.0, 0.0, 1.0, 0.5).inset(30.0, 20.0, 8.0, 22.0);
    electric_panel(c, upper, &moving, rapidity);
    let lower = left.part(0.0, 0.5, 1.0, 1.0).inset(30.0, 20.0, 8.0, 22.0);
    magnetic_panel(c, lower, &moving);
    let right = Rect::new(w * 0.4, top, w, h);
    for (i, radiator) in radiators().iter().enumerate() {
        let panel = right.column(i, 2);
        let side = panel.width().min(panel.height()) - 44.0;
        let square = Rect::around(panel.centre(), Point2::direction(side / 2.0, side / 2.0));
        wave_panel(c, square, radiator, share);
    }
    caption(
        c,
        "MOVING CHARGES: FIELDS FROM CONTRACTED POTENTIAL GRADIENTS",
        "A SOFT CHARGE SPED UP AND SLOWED; POINT CHARGES CIRCLING AT 0.5 AND 0.9 C SHED WAVES (STA)",
    );
}

fn main() {
    run(Anim::new("moving_charge", SECONDS).size(960, 540), draw);
}

#[cfg(test)]
mod tests {
    use super::moving_charge::*;
    use gax::sta::{Bivector, Even, Odd, Vector, intrinsic};
    use gax_numga_examples::testing::near;

    /// numga's window on the soft charge reaches this far along `x`.
    const HALF_WIDTH: f64 = 1.5;

    #[test]
    fn maxwell_with_and_without_the_metric() {
        let moving = charge(TOP_RAPIDITY);
        for event in grid(HALF_WIDTH, HALF_HEIGHT, 15, 10) {
            // The potential's derivative is the field alone: the Lorenz-gauge scalar vanishes.
            let potentials = potential_gradient(&moving, event);
            let field = field(potentials).cast::<Even>();
            assert!(near(potential_derivative(potentials), field, 1e-12));
            // The metric-free trace agrees with the metric contraction, and gives no magnetic
            // charge.
            let gradients = field_gradient(&moving, event);
            let (current, magnetic) = sources(gradients);
            assert!(near(
                field_derivative(gradients),
                current.cast::<Odd>(),
                1e-12
            ));
            assert!(near(magnetic, intrinsic::Vector::zero(), 1e-12));
            // The current is the Plummer density carried along the charge's time direction.
            let separation = spatial(moving.boost << event);
            let softened = CORE_RADIUS * CORE_RADIUS - separation.norm_squared();
            #[allow(clippy::disallowed_methods)] // the Plummer density it is checked against
            let density = 3.0 * CHARGE * CORE_RADIUS * CORE_RADIUS
                / (4.0 * core::f64::consts::PI)
                / (softened * softened * softened.sqrt());
            assert!(near(current, (moving.boost >> T) * density, 1e-9));
        }
    }

    #[test]
    fn the_circling_charge_is_seen_along_light_rays() {
        let fast = orbit(ORBIT_SPEEDS[1].1);
        for spot in grid(ORBIT_HALF_WIDTH, ORBIT_HALF_WIDTH, 12, 12) {
            let event = spot + T * 3.0;
            // The light from the charge at the retarded time reaches each event.
            let (position, _, _) = worldline(&fast, retarded(&fast, event));
            let ray = event - position;
            assert!(ray.norm_squared().abs() < 1e-10);
            // The radiating potential keeps the Lorenz gauge.
            let potentials = orbit_potential_gradient(&fast, event);
            let field = field(potentials).cast::<Even>();
            assert!(near(potential_derivative(potentials), field, 1e-12));
        }
    }

    /// numga's values at a few events: the soft charge's field at rapidity one, its current
    /// near the centre, and the fast orbit's retarded time and field.
    #[test]
    fn fields_match_numga() {
        let moving = charge(1.0);
        let at = Vector::new(0.0, 0.3, 0.2, 0.0);
        let expected = Bivector::new(0.276_752_43, 0.184_501_62, 0.0, 0.0, 0.0, -0.140_515_36);
        assert!(near(field(potential_gradient(&moving, at)), expected, 1e-8));
        let (current, _) = sources(field_gradient(&moving, Vector::new(0.0, 0.02, 0.01, 0.0)));
        assert!(near(
            current,
            Vector::new(491.738_136, 374.504_891, 0.0, 0.0),
            1e-5
        ));
        let fast = orbit(0.9);
        let event = Vector::new(3.0, 2.0, -1.5, 0.0);
        assert!((retarded(&fast, event) - 1.293_433_67).abs() < 1e-8);
        let expected = Bivector::new(-0.002_597_46, -0.012_722_41, 0.0, 0.0, 0.0, 0.012_846_04);
        assert!(near(
            field(orbit_potential_gradient(&fast, event)),
            expected,
            1e-8
        ));
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws(super::draw, 1.0);
    }
}
