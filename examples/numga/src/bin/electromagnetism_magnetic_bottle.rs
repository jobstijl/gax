//! numga's `electromagnetism/magnetic_bottle`: charged particles trapped between two coils,
//! bouncing back and forth along the field, in numga's spacetime algebra `x+y+z+t-`.
//!
//! A charged particle in a magnetic field circles around the field's lines while it drifts
//! along them. The magnetic force does no work, so the particle's speed never changes; and where
//! the field changes slowly on the scale of the circles, the circling keeps its magnetic moment,
//! the momentum of the circling squared over the field. Moving into a stronger field, where the
//! field's lines crowd together near a coil, the particle must circle faster, which takes speed
//! from its drift, until the drift stops and reverses: the coil is a magnetic mirror, and two
//! make a bottle. Particles aimed too closely along the field never run out of drift before the
//! field stops growing, and escape through the ends: the loss cone, `sin² α < B_min / B_max`.
//!
//! Events and velocities are directions in spacetime, intrinsic trivectors (numga's
//! antivectors, ADR-051), and the magnetic field a plane of space, an extrinsic bivector, built
//! from the coils by Biot-Savart's law: each piece of wire contributes the plane it spans with
//! the line to the event, `piece.commutator(separation)`, over the cube of the distance, with no
//! right-hand rule to pick a perpendicular. The field turns the particle's velocity at the rate
//! `velocity.commutator(field)`, and the Lorentz force is the field paired with the velocity,
//! `velocity & field`, a pairing of the intrinsic with the extrinsic rather than a contraction
//! by the metric (its Hodge dual is the turn). A step in the particle's own time is the rotor of
//! that turn, so the speed is kept exactly however long the step; the magnetic moment is the
//! force squared over twice the field cubed.
//!
//! The animation: the particles' tracks trailing behind them, coloured by their angle to the
//! axis, through a sheet of the field's strength (its logarithm) in the plane through the axis,
//! between the coils' wire. The shallow ones run out through the coils; the steep ones bounce.

use gax_numga_examples::scene3::{Scene3, panel3};
use gax_numga_examples::{
    Align, Anim, Camera, Canvas, Lens, Light, ORIGIN3, Point2, Point3, backdrop, caption, colormap,
    palette, reach3, run_with,
};

#[path = "../shared/spacetime.rs"]
mod spacetime;

mod bottle {
    use super::spacetime::{Bivector, Direction, in_space, space, time, x, y, z};
    #[cfg(test)]
    use super::spacetime::{Vector, t};
    #[cfg(test)]
    use gax::ApproxEq;

    /// An event, a velocity or a displacement: an intrinsic trivector.
    pub type Event = Direction;
    /// A magnetic field: an extrinsic bivector.
    pub type Field = Bivector;

    /// Unit directions of space: across the bottle, sideways, and along its axis.
    pub fn across() -> Direction {
        space()[0]
    }
    pub fn axis() -> Direction {
        space()[2]
    }

    /// Circular coils of `radius` around the axis at each height, as straight pieces of wire:
    /// each piece's displacement along the current, and its midpoint.
    pub fn coils(radius: f64, heights: &[f64], pieces: usize) -> (Vec<Event>, Vec<Event>) {
        let xy = (x() ^ y()).cast::<Bivector>();
        let rim: Vec<Event> = (0..=pieces)
            .map(|k| core::f64::consts::TAU * k as f64 / pieces as f64)
            .map(|angle| xy.gp(-angle / 2.0).exp() >> across().gp(radius))
            .collect();
        let mut displacements = Vec::new();
        let mut midpoints = Vec::new();
        for h in heights {
            let wire: Vec<Event> = rim.iter().map(|r| *r + axis().gp(*h)).collect();
            for w in wire.windows(2) {
                displacements.push(w[1] - w[0]);
                midpoints.push((w[1] + w[0]).gp(0.5));
            }
        }
        (displacements, midpoints)
    }

    /// The wire of the coils.
    pub struct Wire {
        pub pieces: Vec<Event>,
        pub midpoints: Vec<Event>,
        pub current: f64,
    }

    impl Wire {
        /// The magnetic field of steady current in the wire at an event: by Biot-Savart's law,
        /// each piece contributes the plane it spans with the line to the event, over the cube
        /// of the distance.
        pub fn field(&self, event: Event) -> Field {
            // The event's place in the lab's space.
            let position = in_space(event);
            let sum = self.pieces.iter().zip(&self.midpoints).fold(
                Field::zero(),
                |sum, (piece, midpoint)| {
                    let separation = position - *midpoint;
                    let distance = separation.norm();
                    // The commutator of two directions is the bivector that turns one towards
                    // the other.
                    sum + piece
                        .commutator(separation)
                        .gp(1.0 / (distance * distance * distance))
                },
            );
            sum * self.current
        }

        /// One step in the particles' own time, of unit charge over mass: half a step along the
        /// velocity, the velocity turned by the field there, and the other half step.
        pub fn step(&self, event: Event, velocity: Event, proper_step: f64) -> (Event, Event) {
            let halfway = event + velocity.gp(proper_step / 2.0);
            // The velocity turns at the rate `velocity.commutator(field)`: the rotor of that
            // turn over the step.
            let turn = self.field(halfway).gp(proper_step / 2.0).exp();
            let velocity = turn >> velocity;
            (halfway + velocity.gp(proper_step / 2.0), velocity)
        }

        /// The events and velocities from a start, at every one of `steps` steps.
        pub fn evolve(
            &self,
            start: (Event, Event),
            proper_step: f64,
            steps: usize,
        ) -> Vec<(Event, Event)> {
            let mut states = vec![start];
            for _ in 1..steps {
                let (e, v) = states[states.len() - 1];
                states.push(self.step(e, v, proper_step));
            }
            states
        }
    }

    /// The magnetic moment of the circling, per unit charge over mass: the Lorentz force squared
    /// over twice the field cubed, the momentum of the circling squared over twice the field.
    #[cfg(test)]
    pub fn moment(velocity: Event, magnetic: Field) -> f64 {
        let force: Vector = velocity & magnetic;
        let strength = magnetic.norm();
        // The force of a magnetic field is square to the lab's time, and squares to its size
        // squared.
        (force * force).s() / (2.0 * strength * strength * strength)
    }

    /// The Lorentz force less the Hodge dual (numga's dual) of the field's turn of the velocity.
    #[cfg(test)]
    pub fn force_mismatch(velocity: Event, magnetic: Field) -> f64 {
        let turn = velocity.commutator(magnetic).hodge();
        turn.max_abs_diff(&(velocity & magnetic))
    }

    /// How far a velocity runs in the lab's time per unit of its own: its Lorentz factor.
    #[cfg(test)]
    pub fn lorentz_factor(velocity: Event) -> f64 {
        (t() & velocity).s()
    }

    /// How high along the axis an event is.
    #[cfg(test)]
    pub fn height(event: Event) -> f64 {
        (z() & event).s()
    }

    // --- the scene ----------------------------------------------------------------------

    /// The coils' radius, their heights along the axis, the straight pieces each is made of,
    /// and the current in them, which makes the field at the centre two units.
    pub const COIL_RADIUS: f64 = 1.0;
    pub const COIL_HEIGHTS: [f64; 2] = [-2.0, 2.0];
    pub const PIECES: usize = 48;
    pub const CURRENT: f64 = 1.78;
    /// The particles' speed as a fraction of light's, their angles to the axis in degrees and
    /// their distances from it at launch, the steepest furthest out.
    pub const SPEED: f64 = 0.3;
    pub const PARTICLES: usize = 10;
    /// The step in the particles' own time, and the steps.
    pub const PROPER_STEP: f64 = 0.1;
    pub const STEPS: usize = 1000;
    /// The points on the axis between the coils the loss cone is read from.
    #[cfg(test)]
    pub const AXIS_POINTS: usize = 401;

    pub fn pitch(k: usize) -> f64 {
        15.0 + 65.0 * k as f64 / (PARTICLES - 1) as f64
    }

    pub fn wire() -> Wire {
        let (pieces, midpoints) = coils(COIL_RADIUS, &COIL_HEIGHTS, PIECES);
        Wire {
            pieces,
            midpoints,
            current: CURRENT,
        }
    }

    /// The particles in the bottle's middle plane, each at its distance from the axis, moving
    /// at its angle to the axis, tilted from it sideways: unit velocities, time plus the speed
    /// along their direction, normalized.
    pub fn launch() -> Vec<(Event, Event)> {
        let zy = (z() ^ y()).cast::<Bivector>();
        (0..PARTICLES)
            .map(|k| {
                let tilt = pitch(k).to_radians();
                let direction = zy.gp(-tilt / 2.0).exp() >> axis();
                let offset = 0.6 * k as f64 / (PARTICLES - 1) as f64;
                (
                    across().gp(offset),
                    (time() + direction.gp(SPEED)).normalized(),
                )
            })
            .collect()
    }

    /// The particles' events and velocities at every step, `[particles][steps]`.
    pub fn flights(wire: &Wire) -> Vec<Vec<(Event, Event)>> {
        launch()
            .into_iter()
            .map(|start| wire.evolve(start, PROPER_STEP, STEPS))
            .collect()
    }

    /// The field's strength on the axis between the coils.
    #[cfg(test)]
    pub fn axis_strengths(wire: &Wire) -> Vec<f64> {
        let [lo, hi] = COIL_HEIGHTS;
        (0..AXIS_POINTS)
            .map(|k| lo + (hi - lo) * k as f64 / (AXIS_POINTS - 1) as f64)
            .map(|h| wire.field(axis().gp(h)).norm())
            .collect()
    }
}

use bottle::*;
use spacetime::{Bivector, Direction, Even};

/// The steps between the frames drawn, the steps of each trail, and numga's frame of 40 ms.
const STEPS_PER_FRAME: usize = 8;
const TRAIL: usize = 60;
const SECONDS: f32 = (STEPS / STEPS_PER_FRAME) as f32 * 0.04;
/// The sheet of the field's strength: its half widths along the axis and across it, and its
/// cells each way (numga's sheet has 120 by 240 samples; drawn as quads, fewer serve).
const HALF_LENGTH: f64 = 3.0;
const HALF_WIDTH: f64 = 1.2;
const CELLS: [usize; 2] = [24, 60];

/// What the frames draw: the tracks, `[particles][steps]` events, and the sheet's log
/// strength at its grid's points, `[across][along]`.
struct Scene {
    tracks: Vec<Vec<Event>>,
    sheet: Vec<Vec<f64>>,
}

fn scene() -> Scene {
    let wire = wire();
    let tracks = flights(&wire)
        .into_iter()
        .map(|f| f.into_iter().map(|(e, _)| e).collect())
        .collect();
    let sheet = (0..=CELLS[0])
        .map(|i| {
            (0..=CELLS[1])
                .map(|j| wire.field(sheet_point(i, j)).norm().ln())
                .collect()
        })
        .collect();
    Scene { tracks, sheet }
}

/// A point of the sheet's grid, in the plane through the axis.
fn sheet_point(i: usize, j: usize) -> Event {
    let a = -HALF_WIDTH + 2.0 * HALF_WIDTH * i as f64 / CELLS[0] as f64;
    let b = -HALF_LENGTH + 2.0 * HALF_LENGTH * j as f64 / CELLS[1] as f64;
    across().gp(a) + axis().gp(b)
}

/// The drawing's frame: a third turn cycles `z` to `x`, `x` to `y` and `y` to `z`, laying the
/// bottle's axis across the picture.
fn to_scene() -> Even {
    let cycle = Bivector::new(1.0, 1.0, 1.0, 0.0, 0.0, 0.0).normalized();
    cycle.gp(-core::f64::consts::PI / 3.0).exp().into_inner()
}

/// An event's place in the drawing.
fn place(turn: Even, e: Direction) -> Point3 {
    reach3(turn >> spacetime::in_space(e))
}

/// The colour of the particle launched at the `k`-th angle.
fn colour(k: usize) -> Light {
    colormap::inferno(0.45 + 0.5 * k as f32 / (PARTICLES - 1) as f32).faded(1.6)
}

fn draw(c: &mut Canvas, sc: &Scene, t: f32) {
    backdrop(c);
    let screen = c.rect();
    let turn = to_scene();
    let end = ((t / SECONDS * (STEPS / STEPS_PER_FRAME) as f32) as usize + 1) * STEPS_PER_FRAME;
    let end = end.min(STEPS);
    let rect = screen.part(0.0, 0.1, 1.0, 1.0);
    let cam = Camera::orbit(
        rect,
        ORIGIN3,
        9.5,
        (-118f32).to_radians(),
        22f32.to_radians(),
        Lens::Perspective(0.42),
    );
    let (lo, hi) = sc
        .sheet
        .iter()
        .flatten()
        .fold((f64::MAX, f64::MIN), |(lo, hi), v| (lo.min(*v), hi.max(*v)));
    panel3(c, rect, cam, |s: &mut Scene3| {
        for i in 0..CELLS[0] {
            for j in 0..CELLS[1] {
                let value = (sc.sheet[i][j] + sc.sheet[i + 1][j + 1]) / 2.0;
                let shade = colormap::blues((0.9 - 0.75 * (value - lo) / (hi - lo)) as f32);
                let corners = [(i, j), (i, j + 1), (i + 1, j + 1), (i + 1, j)]
                    .map(|(a, b)| place(turn, sheet_point(a, b)));
                s.quad(
                    corners[0],
                    corners[1],
                    corners[2],
                    corners[3],
                    shade.faded(0.6),
                    0.55,
                );
            }
        }
        let xy = Bivector::new(0.0, 0.0, 1.0, 0.0, 0.0, 0.0);
        for h in COIL_HEIGHTS {
            let ring: Vec<Point3> = (0..=96)
                .map(|k| core::f64::consts::TAU * k as f64 / 96.0)
                .map(|a| xy.gp(-a / 2.0).exp() >> across().gp(COIL_RADIUS))
                .map(|r| place(turn, r + axis().gp(h)))
                .collect();
            s.polyline(&ring, 1.8, palette::YELLOW);
        }
        for (k, track) in sc.tracks.iter().enumerate() {
            let trail: Vec<Point3> = track[end.saturating_sub(TRAIL)..end]
                .iter()
                .map(|e| place(turn, *e))
                .collect();
            s.polyline(&trail, 1.2, colour(k));
            s.dot(
                trail[trail.len() - 1],
                gax_numga_examples::Marker::Dot,
                6.0,
                colour(k),
            );
        }
    });
    let legend = format!(
        "LAUNCHED AT {:.0} TO {:.0} DEGREES TO THE AXIS, AT {:.1} c",
        pitch(0),
        pitch(PARTICLES - 1),
        SPEED
    );
    let at = screen.bottom_middle() - Point2::direction(0.0, 14.0);
    c.text(&legend, at, 12.0, palette::INK, Align::Center);
    caption(
        c,
        "THE MAGNETIC BOTTLE",
        "THE FIELD TURNS THE VELOCITY BY A ROTOR, KEEPING ITS SPEED; THE SHALLOW ONES ESCAPE (STA)",
    );
}

fn main() {
    run_with(
        Anim::new("magnetic bottle", SECONDS).size(960, 540),
        scene,
        draw,
    );
}

#[cfg(test)]
mod tests {
    use super::bottle::*;

    /// The Lorentz factor of the launch speed, the reference the velocities keep.
    #[allow(clippy::disallowed_methods)] // the Lorentz factor 1/√(1 - v²) they are checked against
    fn gamma() -> f64 {
        1.0 / (1.0 - SPEED * SPEED).sqrt()
    }

    /// numga's first test: after 49 steps the Lorentz force is the Hodge dual of the field's
    /// turn of the velocity, and the field has kept the speed.
    #[test]
    fn field_turns_velocity_by_the_lorentz_force_at_constant_speed() {
        let wire = wire();
        for start in launch() {
            let (event, velocity) = wire.evolve(start, PROPER_STEP, 50)[49];
            assert!(force_mismatch(velocity, wire.field(event)) < 1e-10);
            assert!((lorentz_factor(velocity) - gamma()).abs() < 1e-11);
        }
    }

    /// numga's second test and scenario checks: escape through the coils exactly inside the
    /// loss cone of the field on the axis, some escaping and some not; the trapped particles
    /// keep their magnetic moment, averaged over a turn, to a fifth through their bounces, the
    /// escaping ones lose it; every step keeps the speed and the force.
    #[test]
    fn loss_cone_and_magnetic_moment() {
        let wire = wire();
        let strengths = axis_strengths(&wire);
        let (weakest, strongest) = strengths
            .iter()
            .fold((f64::MAX, 0.0f64), |(lo, hi), s| (lo.min(*s), hi.max(*s)));
        #[allow(clippy::disallowed_methods)] // the loss cone's angle, sin² α = B_min / B_max
        let loss_cone = (weakest / strongest).sqrt().asin().to_degrees();
        let n_turn = (core::f64::consts::TAU / (weakest * PROPER_STEP)) as usize;
        let flights = flights(&wire);
        let mut escapes = Vec::new();
        for (k, flight) in flights.iter().enumerate() {
            let escaped = height(flight[STEPS - 1].0).abs() > COIL_HEIGHTS[1];
            assert_eq!(escaped, pitch(k) < loss_cone, "{k} {loss_cone}");
            escapes.push(escaped);
            let moments: Vec<f64> = flight
                .iter()
                .map(|(e, v)| {
                    let field = wire.field(*e);
                    assert!(force_mismatch(*v, field) < 1e-10);
                    assert!((lorentz_factor(*v) - gamma()).abs() < 1e-11);
                    moment(*v, field)
                })
                .collect();
            let means: Vec<f64> = moments
                .windows(n_turn)
                .map(|w| w.iter().sum::<f64>() / n_turn as f64)
                .collect();
            let drift = means
                .iter()
                .fold(0.0f64, |d, m| d.max((m / means[0] - 1.0).abs()));
            if escaped {
                assert!(drift > 1.0, "{k} {drift}");
            } else {
                assert!(drift < 0.2, "{k} {drift}");
            }
        }
        assert!(escapes.iter().any(|e| *e) && !escapes.iter().all(|e| *e));
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws_with(super::draw, &super::scene(), 2.0);
    }
}
