//! numga's `mechanics/wing`: potential flow past a wing, a velocity whose geometric derivative
//! vanishes (VGA2D).
//!
//! The velocity is a vector field. Its geometric derivative, `Σ e_i gradient(e_i)` over the
//! orthonormal basis (numga's `(Vector * gradient(Vector)).contract(1, 2)`, shared with the
//! vortices in `shared/mechanics_vga2d.rs`), has a scalar part,
//! the divergence, and a bivector part, the vorticity. Air past a wing, taken as ideal, has
//! neither: its velocity's derivative vanishes everywhere.
//!
//! Past a cylinder in a uniform stream the velocity is a formula: the stream, less the stream's
//! image in the cylinder `r² p⁻¹ stream p⁻¹`, less the swirl of a circulation around it, itself
//! a bivector. How the inverse of the position changes with a step, `-p⁻¹ step p⁻¹`, gives its
//! gradient map. A step changes the velocity potential by `velocity | step` and the stream
//! function, a bivector, by `velocity ^ step`: the two parts of `velocity step`. The
//! circulation is the one that lets the flow leave the wing's sharp trailing edge smoothly, the
//! Kutta condition, `4 π (stream ^ edge)`.
//!
//! The Joukowski map, `p + (critical.dual() >> p⁻¹)`, adds to each point its inverse reflected
//! in the line along the critical direction, the mirror whose normal is its dual. Its Jacobian
//! `1 - (critical p⁻¹) >> ·` turns and scales every step, keeping angles, except at the two
//! critical points, where it vanishes. The cylinder through
//! the critical point, the trailing edge, becomes the wing, its smooth edge folded there into a
//! cusp. The potential and the stream function keep their values at a point's image, so the
//! potential's gradient map composed with the inverse of the Jacobian is the wing's, and its
//! derivative the velocity there. To draw the plane of the wing, each point is taken back: with
//! `P = x p`, `Z = x z` and `C = x c` in the even subalgebra (the complex numbers), the map reads
//! `Z = P + C² P⁻¹`, and its preimages are the roots of `P² - Z P + C² = 0`; the one outside the
//! cylinder is the flow's.
//!
//! Left, the wing pitching up and back in a steady stream: the pressure around it (warm where
//! the flow is fast and the pressure low, cool at the stagnation points), the streamlines, and
//! the Kutta–Joukowski lift. Right, the flow past the cylinder deformed into the flow past the
//! wing as the critical points move apart, their images marked, with the cylinder's centre.

use gax_numga_examples::points::box_map;
use gax_numga_examples::signal::haversine;
use gax_numga_examples::{
    Align, Anim, Axes, Canvas, Light, Marker, ORIGIN2, Point2, Rect, backdrop, caption, contour,
    label_above, palette, reach2, run, vector2,
};

use gax_numga_examples::shared::mechanics_vga2d as plane;

mod wing {
    pub use super::plane::{AXES, Even, Map as VelocityGradient, V, derivative};
    use core::f64::consts::TAU;
    use gax::Complex;
    use gax::vga2d::{Pseudoscalar, Rotor, Scalar, Vector};

    /// A form on steps: how the velocity potential changes.
    pub type PotentialGradient = Scalar<(Vector,)>;

    /// A wing: the cylinder's centre before the map, its scale (the cylinder passes through
    /// `scale chord`, the trailing edge), and the unit chord from the leading to the trailing
    /// edge.
    #[derive(Clone, Copy, Debug)]
    pub struct Wing {
        pub centre: V,
        pub scale: f64,
        pub chord: V,
    }

    impl Wing {
        /// The trailing edge: the map's critical point that makes the wing's sharp edge.
        pub fn trailing_edge(&self) -> V {
            self.chord * self.scale
        }

        /// From the cylinder's centre to the trailing edge, on the cylinder.
        fn edge(&self) -> V {
            self.trailing_edge() - self.centre
        }

        /// The wing turned nose up by the angle of attack: clockwise, the leading edge (towards
        /// -x) rising.
        pub fn pitched(&self, attack: f64) -> Wing {
            let nose_up = Rotor::rotation(-attack);
            Wing {
                centre: nose_up >> self.centre,
                scale: self.scale,
                chord: nose_up >> self.chord,
            }
        }

        /// Whether a point before the map lies outside the cylinder (or on it).
        pub fn outside(&self, p: V) -> bool {
            (p - self.centre).norm_squared() >= self.edge().norm_squared() * (1.0 - 1e-12)
        }
    }

    /// The flow at a point around the wing.
    #[derive(Clone, Copy, Debug)]
    pub struct Flow {
        /// The point, in the wing's plane. The drawing samples the plane at its pixels; numga's
        /// checks integrate around rings of these points.
        #[cfg_attr(not(test), allow(dead_code))]
        pub point: V,
        pub velocity: V,
        /// The stream function, constant along streamlines.
        pub stream: Pseudoscalar,
    }

    /// The circulation that makes the trailing edge a stagnation point of the cylinder's flow.
    pub fn circulation(wing: &Wing, stream: V) -> Pseudoscalar {
        (stream ^ wing.edge()) * (2.0 * TAU)
    }

    /// The flow past the cylinder at a point: its velocity, the velocity's gradient map, and the
    /// stream function.
    pub fn cylinder(wing: &Wing, stream: V, point: V) -> (V, VelocityGradient, Pseudoscalar) {
        let offset = point - wing.centre;
        let radius_squared = wing.edge().norm_squared();
        let swirl = circulation(wing, stream);
        let inverse = offset.inverse();
        // The stream, less its image in the cylinder, and the swirl around it.
        let image = (inverse * stream) * inverse;
        let velocity = stream - image * radius_squared - (swirl * inverse) / TAU;
        // A step changes the inverse by minus the step between two inverses, and the velocity
        // with it.
        let change = -((inverse * Vector::slot()) * inverse);
        let image_change = (change * stream) * inverse + (inverse * stream) * change;
        let gradient = -(image_change * radius_squared) - (swirl * change) / TAU;
        // The stream function: the flow crossing from the centre's line along the stream, the
        // image's, and the swirl's, which grows with the logarithm of the distance.
        let swirling = swirl * (offset.norm_squared().ln() / (2.0 * TAU));
        let flux = (stream ^ offset) + (inverse ^ stream) * radius_squared - swirling;
        (velocity, gradient, flux)
    }

    /// Where the map with the given critical point sends a point: its inverse reflected in the
    /// mirror along the critical direction, whose normal is the critical direction's dual.
    pub fn joukowski(critical: V, point: V) -> V {
        point + (critical.dual() >> point.inverse())
    }

    /// The principal square root of an even element, as the complex number it is (`e12² = -1`).
    fn root(e: Even) -> Even {
        let z = Complex::new(e.s(), e.e12()).sqrt();
        Rotor::new(z.re, z.im)
    }

    /// The two points the map sends to `z`: with `P = x p`, `Z = x z` and `C = x c`, the roots
    /// of `P² - Z P + C² = 0`.
    pub fn preimages(critical: V, z: V) -> [V; 2] {
        let x = AXES[0];
        let (image, c) = (x * z, x * critical);
        let root = root(image * image - c * c * 4.0);
        [(image + root) / 2.0, (image - root) / 2.0].map(|p| x * p)
    }

    /// The flow at the image, under the Joukowski map with the given critical point, of a point
    /// around the cylinder; the wing's critical point is its trailing edge.
    pub fn flow(wing: &Wing, stream: V, point: V, critical: V) -> Flow {
        let (velocity, _, flux) = cylinder(wing, stream, point);
        // How the map moves a small step: less the step turned and scaled by `critical p⁻¹`.
        let jacobian = Vector::slot() - ((critical * point.inverse()) >> Vector::slot());
        // The potential keeps its value at a point's image: a step there is first undone by the
        // Jacobian.
        let potential_gradient: PotentialGradient =
            (velocity | Vector::slot()).of(jacobian.inverse());
        Flow {
            point: joukowski(critical, point),
            velocity: derivative(potential_gradient),
            stream: flux,
        }
    }

    /// The flow at a point of the wing's plane, taken back through the map to the cylinder's;
    /// `None` inside the wing.
    pub fn flow_at(wing: &Wing, stream: V, z: V, critical: V) -> Option<Flow> {
        let [first, second] = preimages(critical, z);
        let distance = |p: V| (p - wing.centre).norm_squared();
        let outer = if distance(first) >= distance(second) {
            first
        } else {
            second
        };
        wing.outside(outer)
            .then(|| flow(wing, stream, outer, critical))
    }

    /// The Kutta–Joukowski lift: the density times the circulation contracted with the stream.
    pub fn lift(wing: &Wing, stream: V, density: f64) -> V {
        (circulation(wing, stream) | stream) * density
    }

    /// Points on circles around the cylinder's centre, counterclockwise, from its surface out
    /// to `reach` times its radius, half an angle step off the trailing edge: `[rings][angles]`.
    pub fn rings(wing: &Wing, rings: usize, angles: usize, reach: f64) -> Vec<Vec<V>> {
        let growth = |k: usize| {
            let share = if rings > 1 {
                k as f64 / (rings - 1) as f64
            } else {
                0.0
            };
            (reach.ln() * share).exp()
        };
        (0..rings)
            .map(|k| {
                (0..angles)
                    .map(|a| {
                        let around = Rotor::rotation(TAU * (a as f64 + 0.5) / angles as f64);
                        wing.centre + (around >> wing.edge()) * growth(k)
                    })
                    .collect()
            })
            .collect()
    }

    // --- scenarios

    /// The cylinder's centre, behind and above the origin before the map: the wing's thickness
    /// and camber.
    pub const THICKNESS: f64 = 0.1;
    pub const CAMBER: f64 = 0.08;
    pub const SCALE: f64 = 1.0;
    pub const SPEED: f64 = 1.0;
    pub const DENSITY: f64 = 1.0;
    /// The angles of attack the wing pitches between, in degrees.
    pub const LOWEST_ATTACK: f64 = -4.0;
    pub const HIGHEST_ATTACK: f64 = 12.0;
    /// The angle of attack of the morph, in degrees.
    pub const MORPH_ATTACK: f64 = 8.0;

    /// The wing level, its chord along the stream.
    pub fn wing() -> Wing {
        Wing {
            centre: Vector::new(-THICKNESS, CAMBER),
            scale: SCALE,
            chord: AXES[0],
        }
    }

    /// The stream.
    pub fn stream() -> V {
        AXES[0] * SPEED
    }
}

use wing::*;

/// One pitch up and back, and one morph there and back.
const SECONDS: f32 = 6.0;
/// The window on the plane of the wing (numga's, taller to fill the frame).
const WINDOW: [f32; 4] = [-3.2, 3.2, -2.4, 2.4];
/// The pressure coefficient's range: suction down to -3, stagnation up to 1.
const SUCTION: f64 = 3.0;
const STAGNATION: f64 = 1.0;
/// The streamlines: 21 levels of the stream function, from -3 to 3.
const STREAM_LEVELS: usize = 21;
const STREAM_REACH: f32 = 3.0;
/// Samples of the stream function across the window's width, for its level lines.
const STREAM_SAMPLES: usize = 220;
/// The lift arrow's length per unit lift.
const LIFT_SCALE: f64 = 0.35;
/// Points around the wing's outline.
const OUTLINE: usize = 180;

/// The smooth sweep from 0 up to 1 and back over a loop: the haversine of `2π t / T`.
fn sweep(t: f32) -> f64 {
    let phase = core::f64::consts::TAU * f64::from(t / SECONDS);
    haversine(phase)
}

/// The light of a pressure coefficient: warm where the flow is fast (suction), cool where it
/// stagnates, from the backdrop's dark blue.
fn shade(pressure: f64) -> Light {
    let (level, hue) = if pressure < 0.0 {
        (-pressure / SUCTION, palette::ORANGE)
    } else {
        (pressure / STAGNATION, palette::SKY)
    };
    palette::BOTTOM.blend(hue.faded(0.7), level.min(1.0) as f32)
}

/// The pressure coefficient around the wing, `1 - |v|² / U²`.
fn draw_pressure(c: &mut Canvas, ax: &Axes, wing: &Wing, critical: V) {
    ax.image(c, 1, |q| {
        let flow = flow_at(wing, stream(), vector2(q), critical)?;
        Some(shade(1.0 - flow.velocity.norm_squared() / (SPEED * SPEED)))
    });
}

/// The streamlines: level lines of the stream function, sampled once over the window.
fn draw_streamlines(c: &mut Canvas, ax: &Axes, wing: &Wing, critical: V) {
    let aspect = (WINDOW[3] - WINDOW[2]) / (WINDOW[1] - WINDOW[0]);
    let (columns, rows) = (STREAM_SAMPLES, (STREAM_SAMPLES as f32 * aspect) as usize);
    let corners = [
        Point2::xy(WINDOW[0], WINDOW[2]),
        Point2::xy(WINDOW[1], WINDOW[3]),
    ];
    let last = Point2::xy((columns - 1) as f32, (rows - 1) as f32);
    let to_plane = box_map([ORIGIN2, last], corners);
    let values: Vec<f32> = (0..columns * rows)
        .map(|k| {
            let at = to_plane.of(Point2::xy((k % columns) as f32, (k / columns) as f32));
            flow_at(wing, stream(), vector2(at), critical)
                .map_or(f32::NAN, |f| f.stream.e12() as f32)
        })
        .collect();
    for level in 0..STREAM_LEVELS {
        let value = -STREAM_REACH + 2.0 * STREAM_REACH * level as f32 / (STREAM_LEVELS - 1) as f32;
        let segments: Vec<[Point2; 2]> = contour::segments(&values, columns, rows, value)
            .into_iter()
            .map(|[a, b]| [to_plane.of(a), to_plane.of(b)])
            .collect();
        ax.stroke(c, &segments, 0.8, palette::INK.faded(0.3));
    }
}

/// The wing's outline: the cylinder's surface carried by the map.
fn draw_outline(c: &mut Canvas, ax: &Axes, wing: &Wing, critical: V) {
    let surface = &rings(wing, 1, OUTLINE, 1.0)[0];
    let outline: Vec<Point2> = surface
        .iter()
        .map(|p| reach2(joukowski(critical, *p)))
        .collect();
    ax.fill(c, &outline, palette::BOTTOM, 1.0);
    let mut closed = outline.clone();
    closed.push(outline[0]);
    ax.polyline(c, &closed, 1.4, palette::INK.faded(0.8));
}

/// The flow past a wing with the given critical point: pressure, streamlines, the outline and
/// the lift from the origin.
fn draw_flow(c: &mut Canvas, rect: Rect, wing: &Wing, critical: V) -> Axes {
    let ax = Axes::new(rect, [WINDOW[0], WINDOW[1]], [WINDOW[2], WINDOW[3]]);
    draw_pressure(c, &ax, wing, critical);
    draw_streamlines(c, &ax, wing, critical);
    draw_outline(c, &ax, wing, critical);
    let arrow = lift(wing, stream(), DENSITY) * LIFT_SCALE;
    ax.arrow(c, ORIGIN2, reach2(arrow), 2.4, 10.0, palette::GREEN);
    let corners = [rect.lo, rect.top_right(), rect.hi, rect.bottom_left()];
    c.polyline(&corners, 1.0, palette::GRID, true);
    ax
}

/// A panel's title above it and a note under it.
fn label(c: &mut Canvas, rect: Rect, title: &str, note: &str) {
    label_above(c, rect, title, 13.0, 8.0);
    let below = rect.bottom_middle() + Point2::direction(0.0, 20.0);
    c.text(note, below, 11.0, palette::INK.faded(0.6), Align::Center);
}

/// The wing pitching between its lowest and highest angle of attack.
fn draw_pitching(c: &mut Canvas, rect: Rect, t: f32) {
    let attack = LOWEST_ATTACK + (HIGHEST_ATTACK - LOWEST_ATTACK) * sweep(t);
    let pitched = wing().pitched(attack.to_radians());
    draw_flow(c, rect, &pitched, pitched.trailing_edge());
    let lift = lift(&pitched, stream(), DENSITY).norm();
    let note = format!("ATTACK {attack:.1} DEGREES, LIFT {lift:.2}");
    label(c, rect, "A WING PITCHING IN A STREAM", &note);
}

/// The cylinder deforming into the wing as the critical points move apart, with their images
/// (where the map doubles angles) and the cylinder's centre before the map.
fn draw_morph(c: &mut Canvas, rect: Rect, t: f32) {
    let pitched = wing().pitched(MORPH_ATTACK.to_radians());
    let critical = pitched.trailing_edge() * sweep(t);
    let ax = draw_flow(c, rect, &pitched, critical);
    let images = [reach2(critical * 2.0), reach2(critical * -2.0)];
    ax.scatter(c, &images, Marker::Dot, 8.0, palette::YELLOW);
    ax.scatter(
        c,
        &[reach2(pitched.centre)],
        Marker::Ring,
        8.0,
        palette::INK,
    );
    let note = format!("CRITICAL POINTS {:.2} OF THE WAY OUT", sweep(t));
    label(c, rect, "FROM THE CYLINDER TO THE WING", &note);
}

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let body = c.rect().part(0.0, 0.2, 1.0, 0.92);
    let panel = |i: usize| {
        let column = body.column(i, 2).inset(12.0, 0.0, 12.0, 0.0);
        let height = column.width() * (WINDOW[3] - WINDOW[2]) / (WINDOW[1] - WINDOW[0]);
        Rect::around(
            column.centre(),
            Point2::direction(column.width() / 2.0, height / 2.0),
        )
    };
    draw_pitching(c, panel(0), t);
    draw_morph(c, panel(1), t);
    caption(
        c,
        "WING: POTENTIAL FLOW, A VELOCITY WITH NO DERIVATIVE",
        "PAST A CYLINDER BY FORMULA, CARRIED TO THE WING BY JOUKOWSKI'S MAP P + C P^-1 C (VGA2D)",
    );
}

fn main() {
    run(Anim::new("wing", SECONDS).size(960, 540), draw);
}

#[cfg(test)]
mod tests {
    use super::wing::*;
    use core::f64::consts::TAU;
    use gax::vga2d::Rotor;

    /// The rings and angles numga's checks sample, and how far out the rings reach.
    const RINGS: usize = 140;
    const ANGLES: usize = 360;
    const REACH: f64 = 6.0;

    /// The flow at every point of a ring.
    fn ring_flow(wing: &Wing, ring: &[V], critical: V) -> Vec<Flow> {
        ring.iter()
            .map(|p| flow(wing, stream(), *p, critical))
            .collect()
    }

    /// Each step around a closed ring of flow: a point and the next.
    fn steps(ring: &[Flow]) -> impl Iterator<Item = (Flow, Flow)> + '_ {
        ring.iter()
            .zip(ring.iter().cycle().skip(1))
            .map(|(a, b)| (*a, *b))
    }

    /// The pressure on a closed surface, half the density times the squared speed along the
    /// inward normal (each step turned a quarter turn back), summed by the trapezoidal rule.
    fn pressure_force(surface: &[Flow]) -> V {
        let back = Rotor::rotation(-TAU / 4.0);
        steps(surface)
            .map(|(here, next)| {
                let squared = here.velocity.norm_squared() + next.velocity.norm_squared();
                (back >> (next.point - here.point)) * (0.5 * DENSITY * squared / 2.0)
            })
            .sum()
    }

    /// Around a closed ring, the velocity times each step, summed by the trapezoidal rule: the
    /// circulation (scalar) and the flux through the ring (bivector).
    fn loop_integral(ring: &[Flow]) -> Even {
        steps(ring)
            .map(|(here, next)| ((here.velocity + next.velocity) / 2.0) * (next.point - here.point))
            .sum()
    }

    /// Every coefficient within `atol` of zero.
    fn vanishes(c: &[f64], atol: f64) -> bool {
        c.iter().all(|x| x.abs() <= atol)
    }

    /// The wing at its highest angle of attack.
    fn highest() -> Wing {
        wing().pitched(HIGHEST_ATTACK.to_radians())
    }

    /// Past the cylinder the velocity has neither divergence nor vorticity.
    fn no_derivative(wing: &Wing, rings: &[Vec<V>]) {
        for p in rings.iter().flatten() {
            let (_, gradient, _) = cylinder(wing, stream(), *p);
            assert!(vanishes(&derivative(gradient).c, 1e-12));
        }
    }

    /// The pressure around the surface is the Kutta–Joukowski lift, with no drag.
    fn pressure_is_the_lift(wing: &Wing) {
        let surface = ring_flow(wing, &rings(wing, 1, ANGLES, 1.0)[0], wing.trailing_edge());
        let force = pressure_force(&surface) - lift(wing, stream(), DENSITY);
        assert!(vanishes(&force.c, 1e-3), "{force:?}");
    }

    #[test]
    fn the_velocity_has_no_derivative_and_the_wing_lifts_by_its_circulation() {
        let wing = highest();
        let (plane, edge) = (rings(&wing, 20, ANGLES, REACH), wing.trailing_edge());
        no_derivative(&wing, &plane);
        // A step changes the stream function by the velocity's flux through it, at the wing as
        // at the cylinder.
        let step = V::new(0.3, -0.2) * 1e-6;
        for p in plane.iter().flatten() {
            let here = flow(&wing, stream(), *p, edge);
            let forward = flow(&wing, stream(), *p + step, edge);
            let backward = flow(&wing, stream(), *p - step, edge);
            let crossing = (forward.point - backward.point) / 2.0;
            let change = (forward.stream - backward.stream) / 2.0 - (here.velocity ^ crossing);
            assert!(vanishes(&change.c, 1e-12), "{change:?}");
        }
        // Far away the flow is the stream.
        for p in &rings(&wing, 2, 8, 1e4)[1] {
            let far = flow(&wing, stream(), *p, edge);
            assert!(vanishes(&(far.velocity - stream()).c, 1e-3));
        }
        pressure_is_the_lift(&wing);
    }

    /// numga's scenario checks, on its finer rings.
    #[test]
    fn around_every_ring_the_circulation_and_no_flux() {
        let wing = highest();
        let plane = rings(&wing, RINGS, ANGLES, REACH);
        no_derivative(&wing, &plane);
        // Around every ring about the wing the velocity times each step adds up to the
        // circulation, with no flux: the derivative between any two rings vanishes too.
        let loops: Vec<Even> = plane
            .iter()
            .map(|ring| loop_integral(&ring_flow(&wing, ring, wing.trailing_edge())))
            .collect();
        let circulation = Even::new(circulation(&wing, stream()).dual().s(), 0.0);
        for l in &loops {
            assert!(vanishes(&(*l - loops[0]).c, 1e-11), "{l:?}");
            assert!(vanishes(&(*l - circulation).c, 1e-3), "{l:?}");
        }
        pressure_is_the_lift(&wing);
    }

    /// The drawing takes each point of the wing's plane back through the map: the preimage
    /// outside the cylinder maps forward to the point again.
    #[test]
    fn points_of_the_plane_come_back_through_the_map() {
        let wing = highest();
        for p in rings(&wing, 6, 24, REACH).iter().flatten() {
            let z = joukowski(wing.trailing_edge(), *p);
            let back = flow_at(&wing, stream(), z, wing.trailing_edge()).expect("outside");
            assert!(vanishes(&(back.point - z).c, 1e-9), "{p:?}");
        }
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws(super::draw, 1.0);
    }
}
