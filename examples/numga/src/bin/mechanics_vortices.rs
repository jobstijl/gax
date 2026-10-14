//! numga's `mechanics/vortices`: vortices in a plane of fluid, the derivative of their flow,
//! and their motion (VGA2D).
//!
//! Each vortex turns the fluid around its centre in the plane `xy`, at a rate that falls off as
//! one over the squared distance beyond a small core: the vortex blob of Rosenhead and Moore,
//! finite at its own centre. The fluid at a separation `s` from a vortex moves along `s | xy`,
//! the separation turned a quarter turn in the plane. A pattern of vortices may repeat at a set
//! of offsets, its periodic copies; a pattern that does not repeat has the zero offset alone.
//! The velocity is a sum over the vortices and their copies, and so is the velocity gradient at
//! a point: a map on displacements, `Vector <- Vector`, the change in velocity over a small
//! step, in closed form.
//!
//! Contracting the gradient against an open vector gives the derivative of the flow, an even
//! element: numga's `(Vector * gradient(Vector)).contract(1, 2)`, here the sum
//! `Σ e_i gradient(e_i)` over the orthonormal basis (`shared/mechanics_vga2d.rs`, shared with
//! the wing). Its scalar part is the divergence, zero for this flow, and
//! its bivector part the vorticity, the rate at which the fluid turns. The trace of the gradient
//! applied twice needs no metric and tells swirl from strain: `-0.5 trace(gradient(gradient))`
//! is positive where the fluid turns faster than it stretches. Each vortex moves with the flow
//! at its centre, to which its own blob adds nothing, stepped by the classical Runge–Kutta rule.
//!
//! Four scenes: two pairs leapfrogging, a shear layer rolling up (64 vortices repeating along
//! x), a gas of vortices of both senses, and a walkabout, where the inner pair is too narrow to
//! leapfrog stably and the vortices swing far out once before leapfrogging on. Each shows the
//! vorticity (warm counterclockwise, cool clockwise), the first three with the line where swirl
//! gives way to strain, the walkabout with the vortices' paths.

use gax_numga_examples::{
    Align, Anim, Axes, Canvas, Light, Marker, Point2, Rect, backdrop, caption, palette, reach2,
    run, vector2,
};
use std::sync::OnceLock;

#[path = "../shared/mechanics_vga2d.rs"]
mod plane;

mod vortices {
    use super::plane::derivative;
    use super::plane::turn;
    pub use super::plane::{AXES, Even, Map as Gradient, PLANE, V};
    use core::f64::consts::{PI, TAU};
    use gax::vga2d::Vector;
    use gax_numga_examples::rng::{self, Draw};
    use gax_numga_examples::signal::wave;
    use rand::seq::SliceRandom;

    /// No copies: the zero offset alone.
    pub const ALONE: [V; 1] = [Vector::new(0.0, 0.0)];

    /// Vortex blobs, and the offsets their pattern repeats at.
    #[derive(Clone, Debug)]
    pub struct Vortices {
        pub centres: Vec<V>,
        pub circulations: Vec<f64>,
        pub core_radius: f64,
        pub copies: Vec<V>,
    }

    impl Vortices {
        /// From every vortex and copy to a point, with the vortex's circulation.
        fn separations(&self, point: V) -> impl Iterator<Item = (V, f64)> + '_ {
            self.centres
                .iter()
                .zip(&self.circulations)
                .flat_map(move |(c, g)| self.copies.iter().map(move |o| (point - (*c + *o), *g)))
        }

        /// The blob's squared distance, `|s|² + a²`.
        fn blob(&self, separation: V) -> f64 {
            separation.norm_squared() + self.core_radius * self.core_radius
        }

        /// The rate at which the fluid turns about a centre, and its change with the squared
        /// distance.
        fn profile(&self, separation: V, circulation: f64) -> (f64, f64) {
            let blob = self.blob(separation);
            let rate = circulation / TAU / blob;
            (rate, -rate / blob)
        }

        /// The velocity the vortices and their copies induce at a point: each moves the fluid
        /// along the separation turned a quarter turn in the plane.
        pub fn velocity(&self, point: V) -> V {
            self.separations(point)
                .map(|(s, g)| (s | PLANE) * self.profile(s, g).0)
                .sum()
        }

        /// The velocity gradient at a point. A displacement turns with the rate at the point, and
        /// changes the rate by its component along the separation, which acts on the turned
        /// separation.
        pub fn gradient(&self, point: V) -> Gradient {
            self.separations(point)
                .map(|(s, g)| {
                    let (rate, change) = self.profile(s, g);
                    let turning = (Vector::slot() | PLANE) * rate;
                    let spreading = ((s | PLANE) * (2.0 * change)) * (s | Vector::slot());
                    turning + spreading
                })
                .sum()
        }

        /// The derivative of the flow at a point: the divergence plus the vorticity.
        pub fn derivative(&self, point: V) -> Even {
            derivative(self.gradient(point))
        }

        /// The blobs' vorticity at a point, in closed form: `Γ a² / (π (|s|² + a²)²)` summed.
        #[cfg_attr(not(test), allow(dead_code))] // numga's check of the derivative
        pub fn vorticity(&self, point: V) -> f64 {
            let a2 = self.core_radius * self.core_radius;
            self.separations(point)
                .map(|(s, g)| g * a2 / PI / self.blob(s).powi(2))
                .sum()
        }

        /// The velocity of each vortex: the flow at its centre.
        pub fn drift(&self) -> Vec<V> {
            self.centres.iter().map(|c| self.velocity(*c)).collect()
        }

        /// The vortices moved at the given rates for a time `dt`.
        fn moved(&self, rates: &[V], dt: f64) -> Vortices {
            let centres = self.centres.iter().zip(rates).map(|(c, r)| *c + *r * dt);
            Vortices {
                centres: centres.collect(),
                ..self.clone()
            }
        }

        /// The vortices moved for a time `dt` by the classical fourth-order Runge–Kutta rule.
        pub fn step(&self, dt: f64) -> Vortices {
            let first = self.drift();
            let second = self.moved(&first, dt / 2.0).drift();
            let third = self.moved(&second, dt / 2.0).drift();
            let fourth = self.moved(&third, dt).drift();
            let rates: Vec<V> = (0..first.len())
                .map(|i| (first[i] + (second[i] + third[i]) * 2.0 + fourth[i]) / 6.0)
                .collect();
            self.moved(&rates, dt)
        }

        /// After `steps` steps of `dt`.
        pub fn run(&self, dt: f64, steps: usize) -> Vortices {
            (0..steps).fold(self.clone(), |v, _| v.step(dt))
        }

        /// The vortices at each of `frames` frames, `substeps` steps of `dt` apart.
        pub fn frames(&self, dt: f64, substeps: usize, frames: usize) -> Vec<Vortices> {
            let mut states = vec![self.clone()];
            for _ in 1..frames {
                states.push(states[states.len() - 1].run(dt, substeps));
            }
            states
        }

        /// The vortices' mean position.
        pub fn mean(&self) -> V {
            self.centres.iter().copied().sum::<V>() / self.centres.len() as f64
        }
    }

    /// How much faster the fluid turns than it stretches; positive inside a vortex core.
    pub fn swirl(gradient: Gradient) -> f64 {
        -0.5 * gradient.of(gradient).trace()
    }

    // --- scenarios

    /// Two pairs on one axis, each pair turning in opposite senses so that it travels along x;
    /// the inner pair is wider than 0.38 of the outer, so the two take turns passing, and keep
    /// doing so.
    pub const LEAPFROG_SPANS: [f64; 4] = [1.0, -1.0, 0.55, -0.55];
    pub const LEAPFROG_CIRCULATIONS: [f64; 4] = [1.0, -1.0, 1.0, -1.0];
    pub const LEAPFROG_CORE: f64 = 0.15;

    /// A shear layer: a row of equal vortices, repeating along x, displaced by a wave of two
    /// rolls per period and a weaker wave of one, which seeds the rolls' pairing.
    pub const LAYER_PERIOD: f64 = 1.0;
    pub const LAYER_VORTICES: usize = 64;
    pub const LAYER_COPIES: i32 = 4;
    pub const LAYER_JUMP: f64 = 1.0;
    pub const LAYER_CORE: f64 = 0.05;
    pub const LAYER_AMPLITUDE: f64 = 0.01;
    pub const LAYER_SEED: f64 = 0.5;

    /// A gas of vortices scattered over a disc, most turning one way: the excess keeps the gas
    /// together, and the few turning the other way pair up with neighbours and shoot through it.
    pub const GAS_TURNING: usize = 32;
    pub const GAS_COUNTER: usize = 8;
    pub const GAS_RADIUS: f64 = 1.5;
    pub const GAS_CORE: f64 = 0.08;
    pub const GAS_SEED: u64 = 3;

    /// A walkabout: the inner pair at 0.35 of the outer, where leapfrogging is unstable, and one
    /// inner vortex nudged along x to break the mirror symmetry. One lap, and back to
    /// leapfrogging, on a strip around the whole trip.
    pub const WALKABOUT_SPANS: [f64; 4] = [1.0, -1.0, 0.35, -0.35];
    pub const WALKABOUT_NUDGES: [f64; 4] = [0.0, 0.0, 0.015, 0.0];
    pub const WALKABOUT_DT: f64 = 0.2;
    pub const WALKABOUT_FRAMES: usize = 130;
    pub const WALKABOUT_SUBSTEPS: usize = 5;
    pub const WALKABOUT_ALONG: f64 = 9.2;
    pub const WALKABOUT_ACROSS: f64 = 1.4;
    pub const WALKABOUT_HALF_WIDTH: f64 = 9.8;
    pub const WALKABOUT_HALF_HEIGHT: f64 = 2.9;
    pub const WALKABOUT_VORTICITY_LIMIT: f64 = 8.0;

    /// A scene: its vortices, the window around them (half extents and numga's samples), the
    /// step, the steps per frame and the frames, and the vorticity drawn in full colour.
    #[derive(Clone, Debug)]
    pub struct Scene {
        pub name: &'static str,
        pub vortices: Vortices,
        pub half_width: f64,
        pub half_height: f64,
        // The animation samples the flow at its pixels; numga's samples serve its check.
        #[cfg_attr(not(test), allow(dead_code))]
        pub columns: usize,
        #[cfg_attr(not(test), allow(dead_code))]
        pub rows: usize,
        pub dt: f64,
        pub substeps: usize,
        pub frames: usize,
        pub vorticity_limit: f64,
    }

    impl Scene {
        /// The vortices at every frame.
        pub fn evolve(&self) -> Vec<Vortices> {
            self.vortices.frames(self.dt, self.substeps, self.frames)
        }
    }

    /// Vortices on the y axis at the given spans, with the leapfrog's circulations and core.
    fn pairs(centres: Vec<V>) -> Vortices {
        Vortices {
            centres,
            circulations: LEAPFROG_CIRCULATIONS.to_vec(),
            core_radius: LEAPFROG_CORE,
            copies: ALONE.to_vec(),
        }
    }

    pub fn leapfrog() -> Scene {
        let centres = LEAPFROG_SPANS.iter().map(|s| AXES[1] * *s).collect();
        Scene {
            name: "LEAPFROG",
            vortices: pairs(centres),
            half_width: 2.4,
            half_height: 1.6,
            columns: 300,
            rows: 200,
            dt: 0.1,
            substeps: 2,
            frames: 100,
            vorticity_limit: 8.0,
        }
    }

    pub fn shear_layer() -> Scene {
        let along = |k: usize| (k as f64 + 0.5) / LAYER_VORTICES as f64 * LAYER_PERIOD;
        let centres = (0..LAYER_VORTICES)
            .map(|k| {
                let phase = TAU * along(k) / LAYER_PERIOD;
                let across = LAYER_AMPLITUDE * (wave(2.0 * phase) + LAYER_SEED * wave(phase));
                Vector::new(along(k) - LAYER_PERIOD / 2.0, across)
            })
            .collect();
        // The layer's circulation per period is the jump in velocity across it times the period.
        let circulation = -LAYER_JUMP * LAYER_PERIOD / LAYER_VORTICES as f64;
        let copies = (-LAYER_COPIES..=LAYER_COPIES)
            .map(|k| AXES[0] * (f64::from(k) * LAYER_PERIOD))
            .collect();
        Scene {
            name: "SHEAR LAYER",
            vortices: Vortices {
                centres,
                circulations: vec![circulation; LAYER_VORTICES],
                core_radius: LAYER_CORE,
                copies,
            },
            half_width: LAYER_PERIOD,
            half_height: 0.4,
            columns: 120,
            rows: 48,
            dt: 0.025,
            substeps: 2,
            frames: 160,
            vorticity_limit: 10.0,
        }
    }

    pub fn gas() -> Scene {
        let mut rng = rng::rng(GAS_SEED);
        let count = GAS_TURNING + GAS_COUNTER;
        let centres = (0..count)
            .map(|_| disc_point(GAS_RADIUS, rng.uniform(), rng.uniform()))
            .collect();
        let mut circulations = vec![1.0; GAS_TURNING];
        circulations.extend([-1.0; GAS_COUNTER]);
        circulations.shuffle(&mut rng);
        Scene {
            name: "GAS",
            vortices: Vortices {
                centres,
                circulations,
                core_radius: GAS_CORE,
                copies: ALONE.to_vec(),
            },
            half_width: 3.6,
            half_height: 3.0,
            columns: 240,
            rows: 200,
            dt: 0.01,
            substeps: 10,
            frames: 150,
            vorticity_limit: 15.0,
        }
    }

    /// A point uniform over a disc for two uniform draws: a radius growing as the square root
    /// of the first (equal areas), turned through the second's share of a turn.
    #[allow(clippy::disallowed_methods)] // a radius uniform over the disc's area
    fn disc_point(radius: f64, radius_draw: f64, turn_draw: f64) -> V {
        turn(TAU * turn_draw) >> (AXES[0] * (radius * radius_draw.sqrt()))
    }

    /// The nudged pairs.
    pub fn walkabout() -> Vortices {
        let centres = WALKABOUT_SPANS
            .iter()
            .zip(WALKABOUT_NUDGES)
            .map(|(s, n)| Vector::new(n, *s))
            .collect();
        pairs(centres)
    }

    /// The walkabout's vortices at every frame.
    pub fn walk() -> Vec<Vortices> {
        walkabout().frames(WALKABOUT_DT, WALKABOUT_SUBSTEPS, WALKABOUT_FRAMES)
    }
}

use vortices::*;

/// The loop: the longest scene, numga's 160 frames of 60 ms.
const SECONDS: f32 = 9.6;
/// Samples of the swirl across a panel, along its longer side.
const SWIRL_SAMPLES: usize = 140;

/// The scenes and their evolutions, computed once.
struct Scenes {
    scenes: [(Scene, Vec<Vortices>); 3],
    walk: Vec<Vortices>,
}

fn scenes() -> &'static Scenes {
    static S: OnceLock<Scenes> = OnceLock::new();
    S.get_or_init(|| Scenes {
        scenes: [leapfrog(), shear_layer(), gas()].map(|s| {
            let states = s.evolve();
            (s, states)
        }),
        walk: walk(),
    })
}

/// The frame a fraction of the way through `frames`.
fn frame_at(fraction: f32, frames: usize) -> usize {
    ((fraction * frames as f32) as usize).min(frames - 1)
}

/// The half height of equal axes in `rect` that show a window of the given half extents whole.
fn half_height(rect: Rect, half_width: f64, half_height: f64) -> f32 {
    (half_height as f32).max(half_width as f32 * rect.height() / rect.width())
}

/// The light of a sense of turning: counterclockwise (positive) warm, clockwise cool.
fn sense(turning: f64) -> Light {
    if turning > 0.0 {
        palette::orange()
    } else {
        palette::sky()
    }
}

/// The light of a vorticity: by its sense, from the backdrop's dark blue to full at the limit.
fn shade(vorticity: f64, limit: f64) -> Light {
    let level = (vorticity.abs() / limit).min(1.0) as f32;
    palette::bottom().blend(sense(vorticity).faded(0.8), level)
}

/// The vorticity over the axes, from the derivative of the flow.
fn draw_vorticity(c: &mut Canvas, ax: &Axes, vortices: &Vortices, limit: f64) {
    ax.image(c, 1, |q| {
        let curl = vortices.derivative(vector2(q)).e12();
        Some(shade(curl, limit))
    });
}

/// The line where swirl gives way to strain.
fn swirl_line() -> Light {
    palette::ink().faded(0.35)
}

/// The vortices' centres, by their sense.
fn draw_centres(c: &mut Canvas, ax: &Axes, vortices: &Vortices, size: f32) {
    for (centre, circulation) in vortices.centres.iter().zip(&vortices.circulations) {
        let light = sense(*circulation).whitened(0.5);
        ax.scatter(c, &[reach2(*centre)], Marker::Ring, size, light);
    }
}

/// A scene at a frame: the vorticity in the window around the vortices, the line where swirl
/// gives way to strain, and the centres.
fn draw_scene(c: &mut Canvas, rect: Rect, scene: &Scene, vortices: &Vortices) {
    let half = half_height(rect, scene.half_width, scene.half_height);
    let ax = Axes::equal(rect, reach2(vortices.mean()), half);
    draw_vorticity(c, &ax, vortices, scene.vorticity_limit);
    let swirling = |p: Point2| swirl(vortices.gradient(vector2(p))) as f32;
    ax.contour(c, swirling, SWIRL_SAMPLES, 0.0, 1.0, swirl_line());
    // A repeating layer's vortices are too many to mark.
    if scene.vortices.copies.len() == 1 {
        draw_centres(c, &ax, vortices, 5.0);
    }
    let corners = [rect.lo, rect.top_right(), rect.hi, rect.bottom_left()];
    c.polyline(&corners, 1.0, palette::grid(), true);
    let title = rect.top_middle() - Point2::direction(0.0, 6.0);
    c.text(scene.name, title, 12.0, palette::ink(), Align::Center);
}

/// The walkabout at a frame: the vorticity on the fixed strip, and each vortex's path so far.
fn draw_walkabout(c: &mut Canvas, rect: Rect, states: &[Vortices], frame: usize) {
    let centre = Point2::xy(WALKABOUT_ALONG as f32, WALKABOUT_ACROSS as f32);
    let ax = Axes::equal(
        rect,
        centre,
        half_height(rect, WALKABOUT_HALF_WIDTH, WALKABOUT_HALF_HEIGHT),
    );
    let now = &states[frame];
    draw_vorticity(c, &ax, now, WALKABOUT_VORTICITY_LIMIT);
    for k in 0..now.centres.len() {
        let path: Vec<Point2> = states[..=frame]
            .iter()
            .map(|v| reach2(v.centres[k]))
            .collect();
        ax.polyline(c, &path, 1.0, palette::ink().faded(0.4));
    }
    draw_centres(c, &ax, now, 7.0);
    ax.frame(c, "WALKABOUT: ONE LAP OUT, THEN LEAPFROGGING ON", "", "");
    ax.legend(
        c,
        &[
            ("COUNTERCLOCKWISE", sense(1.0)),
            ("CLOCKWISE", sense(-1.0)),
            ("SWIRL = STRAIN", swirl_line()),
        ],
    );
}

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let s = scenes();
    let fraction = (t / SECONDS).rem_euclid(1.0);
    let screen = c.rect();
    let top = screen
        .part(0.0, 0.17, 1.0, 0.55)
        .inset(16.0, 8.0, 16.0, 4.0);
    let widths = [0.32, 0.36, 0.32];
    let mut left = 0.0;
    for ((scene, states), width) in s.scenes.iter().zip(widths) {
        let rect = top
            .part(left, 0.0, left + width, 1.0)
            .inset(6.0, 0.0, 6.0, 0.0);
        draw_scene(c, rect, scene, &states[frame_at(fraction, scene.frames)]);
        left += width;
    }
    let bottom = screen
        .part(0.0, 0.6, 1.0, 1.0)
        .inset(40.0, 26.0, 22.0, 26.0);
    draw_walkabout(c, bottom, &s.walk, frame_at(fraction, WALKABOUT_FRAMES));
    caption(
        c,
        "VORTICES: THE DERIVATIVE OF THEIR FLOW",
        "SUM E_I GRADIENT(E_I): NO DIVERGENCE, THE VORTICITY; -TR(G(G))/2 > 0 WHERE IT SWIRLS (VGA2D)",
    );
}

fn main() {
    run(Anim::new("vortices", SECONDS).size(960, 540), draw);
}

#[cfg(test)]
mod tests {
    use super::vortices::*;

    /// Offsets on a grid centred on the origin, at the centres of its cells, row by row.
    fn grid(half_width: f64, half_height: f64, columns: usize, rows: usize) -> Vec<V> {
        let at = |k: usize, n: usize, half: f64| (k as f64 + 0.5) / n as f64 * 2.0 * half - half;
        (0..rows)
            .flat_map(|r| {
                (0..columns)
                    .map(move |c| V::new(at(c, columns, half_width), at(r, rows, half_height)))
            })
            .collect()
    }

    /// numga's window on a scene at a frame: the grid centred on the vortices' mean.
    fn window(scene: &Scene, vortices: &Vortices) -> Vec<V> {
        let offsets = grid(
            scene.half_width,
            scene.half_height,
            scene.columns,
            scene.rows,
        );
        offsets.into_iter().map(|o| vortices.mean() + o).collect()
    }

    /// Every coefficient of an even element within `atol` of zero.
    fn vanishes(e: Even, atol: f64) -> bool {
        e.c.iter().all(|x| x.abs() <= atol)
    }

    /// The derivative has no divergence, and its bivector is the blobs' vorticity.
    fn derivative_is_the_vorticity(vortices: &Vortices, points: &[V], atol: f64) {
        for p in points {
            let curl = Even::new(0.0, vortices.vorticity(*p));
            let difference = vortices.derivative(*p) - curl;
            assert!(vanishes(difference, atol), "{difference:?}");
        }
    }

    #[test]
    fn the_derivative_of_the_flow_and_the_leapfrog() {
        let scene = leapfrog();
        let vortices = &scene.vortices;
        let points = grid(2.0, 1.5, 9, 7);
        // The gradient is the change in velocity over a small displacement.
        let step = V::new(0.3, -0.2) * 1e-5;
        for p in &points {
            let change = (vortices.velocity(*p + step) - vortices.velocity(*p - step)) / 2.0;
            let error = vortices.gradient(*p).of(step) - change;
            assert!(error.c.iter().all(|x| x.abs() <= 1e-12), "{error:?}");
        }
        derivative_is_the_vorticity(vortices, &points, 1e-12);
        // Without divergence, swirl is the gradient's determinant.
        for p in &points {
            let g = vortices.gradient(*p);
            assert!((swirl(g) - g.det()).abs() <= 1e-12, "{p:?}");
        }
        // The two pairs take turns in the lead.
        let mut moving = vortices.clone();
        let mut leads = Vec::new();
        for _ in 0..scene.frames {
            moving = moving.run(scene.dt, scene.substeps);
            leads.push((moving.centres[2] - moving.centres[0]).dot(AXES[0]).s());
        }
        let turns = leads
            .windows(2)
            .filter(|w| w[0].signum() != w[1].signum())
            .count();
        assert!(turns >= 6, "{turns}");
    }

    #[test]
    fn the_shear_layer_rolls_up() {
        let scene = shear_layer();
        let spread = |v: &Vortices| {
            let across = v.centres.iter().map(|c| c.c[1]);
            across.clone().fold(f64::MIN, f64::max) - across.fold(f64::MAX, f64::min)
        };
        let rolled = scene.vortices.run(scene.dt, scene.frames * scene.substeps);
        assert!(spread(&rolled) > 10.0 * spread(&scene.vortices));
    }

    #[test]
    fn the_gas_keeps_its_impulses() {
        let scene = gas();
        let impulses = |v: &Vortices| {
            let weighted = v.centres.iter().zip(&v.circulations);
            let linear: V = weighted.clone().map(|(c, g)| *c * *g).sum();
            let angular: f64 = weighted.map(|(c, g)| g * c.norm_squared()).sum();
            (linear, angular)
        };
        let (linear, angular) = impulses(&scene.vortices);
        let (linear_after, angular_after) = impulses(&scene.vortices.run(scene.dt, 300));
        // Runge–Kutta keeps the linear impulse to round-off; the angular one to the method's
        // accuracy.
        assert!((linear_after - linear).c.iter().all(|x| x.abs() <= 1e-12));
        assert!((angular_after - angular).abs() <= 1e-5);
    }

    /// The vortices swing far out of the leapfrog's band over one unbroken stretch, then
    /// leapfrog on inside it.
    #[test]
    fn the_walkabout_takes_one_lap_and_leapfrogs_on() {
        let mut walker = walkabout();
        let mut farthest = Vec::new();
        for _ in 0..WALKABOUT_FRAMES {
            walker = walker.run(WALKABOUT_DT, WALKABOUT_SUBSTEPS);
            farthest.push(
                walker
                    .centres
                    .iter()
                    .map(|c| c.c[1].abs())
                    .fold(0.0, f64::max),
            );
        }
        let wide: Vec<usize> = (0..farthest.len()).filter(|k| farthest[*k] > 2.0).collect();
        assert!(farthest.iter().copied().fold(0.0, f64::max) > 3.0);
        assert!(wide.windows(2).all(|w| w[1] == w[0] + 1));
        let last = &farthest[farthest.len() - 30..];
        assert!(last.iter().copied().fold(0.0, f64::max) < 1.5);
    }

    /// numga's scenario check: on each scene's last frame, over its window, the derivative of
    /// the flow is the blobs' vorticity.
    #[test]
    fn each_scene_ends_with_the_blobs_vorticity() {
        for scene in [leapfrog(), shear_layer(), gas()] {
            let last = scene.evolve().pop().expect("frames");
            derivative_is_the_vorticity(&last, &window(&scene, &last), 1e-9);
        }
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws(super::draw, 0.5);
    }
}
