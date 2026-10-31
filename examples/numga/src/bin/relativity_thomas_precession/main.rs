//! numga's `relativity/thomas_precession`: two boosts in different directions do not make a
//! boost, they leave a turn behind, and a spin carried around any closed loop of velocities
//! comes back turned by the area the loop encloses in the space of velocities; with a magnetic
//! moment, the same frame-turning bivector gives the Bargmann-Michel-Telegdi equation that the
//! muon g-2 storage ring measures. In numga's spacetime algebra `x+y+z+t-`.
//!
//! A velocity is a direction in time, an intrinsic trivector (numga's antivector, ADR-051): the
//! lab's time direction under every boost, `boost >> time`, a hyperboloid whose geometry is the
//! hyperbolic plane. The pure boost from one velocity to another is the square root of their
//! product, `(head * tail.inverse()).sqrt()`, and carrying a spin along a path of velocities is
//! the product of those boosts. A spin, intrinsic like every angular momentum, is the patch of
//! the plane the gyroscope turns in, its Hodge dual (numga's `dual`), and an observer at rest
//! reads it after the pure boost back to rest.
//!
//! Composing two boosts and undoing the straight one leaves the Wigner turn, `tan(θ/2) =
//! tanh²(η/2)` for two equal perpendicular boosts of rapidity `η`, the area of their triangle of
//! velocities. On a circular orbit the spin turns back by the cap's area, `2π (γ - 1)`: Thomas
//! precession. A gyroscope bouncing round a square box turns only in its bounces, by the area
//! the square of momenta encloses on the hyperboloid each round trip.
//!
//! In a field bivector, with charge over mass one, the force is the field read along the
//! velocity, `velocity & field`, and with the particle's own slice of space, `-velocity.hodge()`,
//! the bivector `(1 + a) field - a (slice ^ force)` turns the particle's whole frame: its
//! velocity as the Lorentz force does, whatever the anomaly `a`, and its spin by the
//! Bargmann-Michel-Telegdi equation. At the magic Lorentz factor `√(1 + 1/a)` an electric field
//! drops out of the anomalous rate, as in the rings at Brookhaven and Fermilab.
//!
//! The animation shows numga's five figures: the rocket's triangle of velocities in the
//! Poincaré disk with its axis at each corner (the faint arrow is where the straight boost would
//! leave it); spins carried round three orbits; a gyroscope bouncing in a box; a muon in a
//! storage ring, its spin (orange) lapping its momentum (grey) with the anomaly exaggerated to
//! 0.1; and the anomalous rate against the electric field at a Lorentz factor of 2 and at the
//! magic one.
//!
//! `precession.rs` holds numga's mathematics and scenes, and their tests.

#[path = "../../shared/spacetime.rs"]
mod spacetime;

mod precession;

use gax_numga_examples::points::circle;
use gax_numga_examples::signal::frame;
use gax_numga_examples::{
    Align, Anim, Axes, Canvas, Light, Marker, ORIGIN2, Point2, Rect, backdrop, caption,
    label_under, palette, run_with,
};
use precession::*;
use spacetime::{Direction, space, time};

/// The loop: long enough for the box's three round trips at numga's 40 ms a frame.
const SECONDS: f32 = 12.0;
/// numga's frame.
const FRAME: f32 = 0.04;
/// Arrow lengths: in the Poincaré disk, around the orbits, and as a share of the box's or
/// ring's half width.
const DISK_ARROW: f64 = 0.16;
const ORBIT_ARROW: f64 = 0.14;
const SCENE_ARROW: f64 = 0.3;
/// The orbits' colours.
const ORBIT_COLOURS: [Light; 3] = [palette::GREEN, palette::BLUE, palette::PURPLE];

/// A figure drawn in a panel at a time.
type Figure = fn(&mut Canvas, Rect, &Scene, f32);

/// What the frames draw, computed once.
struct Scene {
    boosts: Vec<TwoBoosts>,
    orbits: Vec<Vec<Velocity>>,
    orbit_spins: Vec<Vec<Spin>>,
    box_positions: Vec<Velocity>,
    box_spins: Vec<Spin>,
    ring: Vec<RingState>,
    ring_radius: f64,
    sweep: [Vec<f64>; 2],
}

fn scene() -> Scene {
    let orbits: Vec<Vec<Velocity>> = ORBIT_GAMMAS
        .iter()
        .map(|g| orbit_velocities(*g, ORBIT_STEPS))
        .collect();
    let (positions, path) = box_path(BOX_HALF_WIDTH, BOX_MOMENTUM, FLIGHT_FRAMES, BOUNCE_STEPS);
    let box_positions: Vec<Velocity> = (0..LOOPS).flat_map(|_| positions.clone()).collect();
    let box_path: Vec<Velocity> = (0..LOOPS).flat_map(|_| path.clone()).collect();
    let magic = magic(MUON_ANOMALY);
    Scene {
        boosts: rapidities()
            .into_iter()
            .map(|r| two_boosts(r, EDGE_SAMPLES))
            .collect(),
        orbit_spins: orbits.iter().map(|o| orbit_spins(o)).collect(),
        orbits,
        box_spins: bouncing(&box_path),
        box_positions,
        ring: ring(
            magic,
            (RING_MAGNETIC, 0.0),
            MUON_ANOMALY,
            RING_TURNS,
            RING_TURN_STEPS,
            RING_STRIDE,
        ),
        ring_radius: ring_radius(magic, RING_MAGNETIC, 0.0),
        sweep: magic_sweep(),
    }
}

/// Planar coordinates as a point of the drawing.
fn point(p: [f64; 2]) -> Point2 {
    Point2::xy(p[0] as f32, p[1] as f32)
}

/// A direction's place in the plane `xy`.
fn plane(d: Direction) -> Point2 {
    point(planar(d))
}

/// A velocity as a point of the Poincaré disk: where the line from minus the lab's time through
/// it meets the lab's slice of space.
fn disk(v: Velocity) -> Point2 {
    plane((v + time()).gp(1.0 / (1.0 + lab_time(v))) - time())
}

/// The part of a direction in the lab's slice of space, at unit length.
fn unit(d: Direction) -> Point2 {
    plane(spacetime::in_space(d).normalized())
}

/// A spin's arrow in the plane, at unit length.
fn spin_arrow(s: Spin) -> Point2 {
    point(spin_planar(s.normalized()))
}

/// The frame of a figure shown over `frames` frames at numga's pace, looping, at time `t`.
fn at(t: f32, frames: usize) -> usize {
    frame((t / (FRAME * frames as f32)).fract(), frames)
}

/// An empty Poincaré disk about `centre`, its rim the speed of light.
fn disk_axes(c: &mut Canvas, rect: Rect, centre: Point2, half_width: f32) -> Axes {
    let ax = Axes::equal(rect, centre, half_width);
    ax.polyline(c, &circle(ORIGIN2, 1.0, 128), 1.0, palette::GRID.faded(3.0));
    ax
}

/// The rocket's two boosts, the swell played forward and back.
fn boosts_panel(c: &mut Canvas, rect: Rect, sc: &Scene, t: f32) {
    let n = sc.boosts.len();
    let k = at(t, 2 * n - 2);
    let now = &sc.boosts[if k < n { k } else { 2 * n - 2 - k }];
    let ax = disk_axes(c, rect, Point2::xy(0.5, 0.2), 0.62);
    let outline: Vec<Point2> = now.triangle.iter().map(|v| disk(*v)).collect();
    ax.fill(c, &outline, palette::BLUE.faded(0.3), 0.4);
    ax.polyline(c, &outline, 1.0, palette::INK);
    let tails = now.corners.map(disk);
    let straight = unit(space()[0]).gp(DISK_ARROW as f32);
    ax.arrow(
        c,
        tails[2],
        tails[2] + straight,
        1.5,
        6.0,
        palette::INK.faded(0.35),
    );
    for (tail, axis) in tails.iter().zip(now.axes) {
        let tip = *tail + unit(axis).gp(DISK_ARROW as f32);
        ax.arrow(c, *tail, tip, 1.8, 7.0, palette::ORANGE);
    }
    ax.scatter(c, &tails, Marker::Dot, 5.0, palette::INK);
}

/// Spins carried round the circular orbits.
fn orbits_panel(c: &mut Canvas, rect: Rect, sc: &Scene, t: f32) {
    let ax = disk_axes(c, rect, ORIGIN2, 0.66);
    let k = at(t, ORBIT_STEPS / ORBIT_STRIDE + 1) * ORBIT_STRIDE;
    for ((orbit, spins), colour) in sc.orbits.iter().zip(&sc.orbit_spins).zip(ORBIT_COLOURS) {
        let outline: Vec<Point2> = orbit.iter().map(|v| disk(*v)).collect();
        ax.polyline(c, &outline, 0.8, colour.faded(0.5));
        let start = disk(orbit[0]);
        let arrow = |s: Spin| spin_arrow(s).gp(ORBIT_ARROW as f32);
        ax.arrow(
            c,
            start,
            start + arrow(spins[0]),
            1.2,
            5.0,
            palette::INK.faded(0.3),
        );
        let now = disk(orbit[k]);
        ax.arrow(c, now, now + arrow(spins[k]), 1.6, 6.0, colour);
        ax.scatter(c, &[now], Marker::Dot, 5.0, palette::INK);
    }
}

/// The gyroscope bouncing round the box.
fn box_panel(c: &mut Canvas, rect: Rect, sc: &Scene, t: f32) {
    let h = BOX_HALF_WIDTH as f32;
    let ax = Axes::equal(rect, ORIGIN2, 1.15 * h);
    let [ex, ey, _] = space();
    let corners: Vec<Point2> = (0..5)
        .map(|k| {
            let quarter = exp(xy().gp(-(k as f64) * core::f64::consts::FRAC_PI_4));
            plane(quarter >> (ex + ey).gp(BOX_HALF_WIDTH))
        })
        .collect();
    ax.polyline(c, &corners, 1.3, palette::INK);
    let walls: Vec<Point2> = [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0), (1.0, 0.0)]
        .map(|(x, y)| Point2::xy(x * h, y * h))
        .to_vec();
    ax.polyline(c, &walls, 0.7, palette::GRID.faded(3.0));
    let k = at(t, sc.box_positions.len());
    let p = plane(sc.box_positions[k]);
    let arrow = spin_arrow(sc.box_spins[k]).gp((SCENE_ARROW * BOX_HALF_WIDTH) as f32);
    ax.arrow(c, p, p + arrow, 1.8, 7.0, palette::ORANGE);
    ax.scatter(c, &[p], Marker::Dot, 7.0, palette::INK);
}

/// The muon going round the storage ring.
fn ring_panel(c: &mut Canvas, rect: Rect, sc: &Scene, t: f32) {
    let r = sc.ring_radius as f32;
    let ax = Axes::equal(rect, ORIGIN2, 1.3 * r);
    ax.polyline(c, &circle(ORIGIN2, r, 128), 1.3, palette::GRID.faded(3.0));
    let now = &sc.ring[at(t, sc.ring.len())];
    let p = plane(now.event);
    let reach = (SCENE_ARROW * sc.ring_radius) as f32;
    ax.arrow(
        c,
        p,
        p + unit(now.velocity).gp(reach),
        1.4,
        6.0,
        palette::INK.faded(0.5),
    );
    ax.arrow(
        c,
        p,
        p + spin_arrow(now.spin).gp(reach),
        1.8,
        7.0,
        palette::ORANGE,
    );
    ax.scatter(c, &[p], Marker::Dot, 7.0, palette::INK);
}

/// The anomalous rate over the anomaly times the magnetic field, against the electric field
/// over the magnetic field.
fn magic_panel(c: &mut Canvas, rect: Rect, sc: &Scene) {
    let top = sc.sweep.iter().flatten().fold(1.0f64, |m, v| m.max(*v)) as f32 * 1.1;
    let ax = Axes::new(rect, [-0.33, 0.33], [0.0, top]);
    ax.frame(
        c,
        "",
        "ELECTRIC FIELD OVER MAGNETIC FIELD",
        "RATE / (ANOMALY x FIELD)",
    );
    ax.line(
        c,
        Point2::xy(-0.33, 1.0),
        Point2::xy(0.33, 1.0),
        0.8,
        palette::GRID.faded(3.0),
    );
    let labels = ["LORENTZ FACTOR 2", "MAGIC LORENTZ FACTOR"];
    for ((rates, colour), label) in sc
        .sweep
        .iter()
        .zip([palette::SKY, palette::ORANGE])
        .zip(labels)
    {
        let points: Vec<Point2> = rates
            .iter()
            .enumerate()
            .map(|(k, r)| Point2::xy(electric_field(k) as f32 / RING_MAGNETIC as f32, *r as f32))
            .collect();
        ax.polyline(c, &points, 1.4, colour);
        ax.scatter(c, &points, Marker::Dot, 6.0, colour);
        let end = points[points.len() - 1] + Point2::direction(0.0, 0.08 * top);
        ax.text(c, end, label, 10.0, colour, Align::Right);
    }
}

fn draw(c: &mut Canvas, sc: &Scene, t: f32) {
    backdrop(c);
    let screen = c.rect();
    // A cell of the grid, three across and two down under the caption, and its panel above
    // the label.
    let cell = |i: usize, j: usize, span: usize| {
        let (w, h) = (1.0 / 3.0, 0.43);
        let (x0, y0) = (i as f32 * w, 0.13 + j as f32 * h);
        screen.part(x0, y0, x0 + span as f32 * w, y0 + h)
    };
    let panel = |r: Rect| r.inset(10.0, 4.0, 10.0, 26.0);
    let panels: [(Figure, Rect, &str); 4] = [
        (boosts_panel, cell(0, 0, 1), "TWO BOOSTS LEAVE A TURN"),
        (orbits_panel, cell(1, 0, 1), "THOMAS PRECESSION ON ORBITS"),
        (box_panel, cell(2, 0, 1), "A GYROSCOPE IN A BOX"),
        (ring_panel, cell(0, 1, 1), "A MUON IN A STORAGE RING"),
    ];
    for (figure, rect, label) in panels {
        figure(c, panel(rect), sc, t);
        label_under(c, rect, label, 14.0, 0.9);
    }
    magic_panel(c, cell(1, 1, 2).inset(60.0, 26.0, 20.0, 34.0), sc);
    caption(
        c,
        "THOMAS PRECESSION AND THE MUON'S ANOMALY",
        "BOOSTS AS ROTORS OF X+Y+Z+T-: LOOPS OF VELOCITIES TURN A SPIN BY THEIR AREA",
    );
}

fn main() {
    run_with(
        Anim::new("thomas precession", SECONDS).size(960, 540),
        scene,
        draw,
    );
}

#[cfg(test)]
mod tests {
    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws_with(super::draw, &super::scene(), 1.0);
    }
}
