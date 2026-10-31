//! numga's `fluids/vortices`: vortices in a plane of fluid, the derivative of their flow, and
//! their motion (VGA2D). Points, displacements and velocities are antivectors (intrinsic), the
//! duals of vectors; a vortex's circulation is a bivector, its sign the sense of its turn, so a
//! mirror turns every circulation the other way along with every centre.
//!
//! Each vortex turns the fluid around its centre in the plane `xy`, at a rate that falls off as
//! one over the squared distance beyond a small core: the vortex blob of Rosenhead and Moore,
//! finite at its own centre. The fluid at a separation `s` from a vortex moves along `s | xy`,
//! the separation turned a quarter turn in the plane: numga's `s.commutator(rate)`, the rate
//! the circulation sets there. A pattern of vortices may repeat at a set
//! of offsets, its periodic copies; a pattern that does not repeat has the zero offset alone.
//! The velocity is a sum over the vortices and their copies, and so is the velocity gradient at
//! a point: a map on displacements, `Point <- Point`, the change in velocity over a small step,
//! in closed form.
//!
//! Contracting the gradient against an open point gives the derivative of the flow, an even
//! element: numga's `(Point * gradient(Point)).contract(1, 2)`, here the sum
//! `Σ e_i gradient(e_i)` over an orthonormal basis of displacements. Its scalar part is the divergence, zero for this flow, and
//! its bivector part the vorticity, the rate at which the fluid turns. The trace of the gradient
//! applied twice needs no metric and tells swirl from strain: `-0.5 trace(gradient(gradient))`
//! is positive where the fluid turns faster than it stretches. Each vortex moves with the flow
//! at its centre, to which its own blob adds nothing, stepped by the classical Runge–Kutta rule.
//!
//! Four scenes: two pairs leapfrogging, a shear layer rolling up (64 vortices repeating along
//! x), a gas of vortices of both senses, and a walkabout, where the inner pair is too narrow to
//! leapfrog stably and the vortices swing far out once before leapfrogging on. Each shows the
//! vorticity (warm counterclockwise, cool clockwise), the first three with the line where swirl
//! gives way to strain, the walkabout with the vortices' paths. Beside it, numga's worldlines
//! through the plane and time: a ring of five equal vortices turning rigidly, a braid, and the
//! leapfrogging pairs.

use gax_numga_examples::{
    Align, Anim, Axes, Camera, Canvas, Lens, Light, Marker, Point2, Rect, backdrop, caption,
    palette, run_with, scene3::panel3, signal::frame,
};

mod vortices {
    use core::f64::consts::{PI, TAU};
    use gax::vga2d::{Pseudoscalar, Rotor, Vector, intrinsic};
    use gax_numga_examples::rng::{self, Draw};
    use gax_numga_examples::signal::wave;
    use rand::seq::SliceRandom;

    /// A point of the plane, and a displacement or velocity of one: an antivector, intrinsic,
    /// numga's `Point`. In the plane an antivector is the dual of a vector.
    pub type P = intrinsic::Vector;
    /// A vortex's circulation, the turn it gives the fluid: a bivector, its size the vortex's
    /// strength and its sign the sense of the turn; and the rate at which the fluid turns.
    pub type Circulation = Pseudoscalar;
    pub type Rate = Pseudoscalar;
    /// The velocity gradient, a map on displacements.
    pub type Gradient = intrinsic::Vector<(P,)>;
    /// The derivative of the flow: the divergence plus the vorticity.
    pub type Even = Rotor;

    /// The point at `(x, y)`: the dual of the vector, as numga's `(x mv.x + y mv.y).dual()`.
    pub fn at(x: f64, y: f64) -> P {
        Vector::new(x, y).dual()
    }

    /// A point's coordinates: its readings by the lines `x` and `y`, numga's `mv.x & point`.
    pub fn coordinates(p: P) -> [f64; 2] {
        [Vector::new(1.0, 0.0), Vector::new(0.0, 1.0)].map(|line| (line & p).s())
    }

    /// The turn of the fluid in the plane by `circulation`.
    pub fn turn(circulation: f64) -> Circulation {
        Pseudoscalar::new(circulation)
    }

    /// No copies: the zero offset alone.
    pub fn alone() -> Vec<P> {
        vec![at(0.0, 0.0)]
    }

    /// Vortex blobs, and the offsets their pattern repeats at.
    #[derive(Clone, Debug)]
    pub struct Vortices {
        pub centres: Vec<P>,
        pub circulations: Vec<Circulation>,
        pub core_radius: f64,
        pub copies: Vec<P>,
    }

    impl Vortices {
        /// From every vortex and copy to a point, with the vortex's circulation.
        fn separations(&self, point: P) -> impl Iterator<Item = (P, Circulation)> + '_ {
            self.centres
                .iter()
                .zip(&self.circulations)
                .flat_map(move |(c, g)| self.copies.iter().map(move |o| (point - (*c + *o), *g)))
        }

        /// The blob's squared distance, `s | s + a²`.
        fn blob(&self, separation: P) -> f64 {
            separation.dot(separation).s() + self.core_radius * self.core_radius
        }

        /// The rate at which the fluid turns about a centre, and its change with the squared
        /// distance.
        fn profile(&self, separation: P, circulation: Circulation) -> (Rate, Rate) {
            let blob = self.blob(separation);
            let rate = circulation.gp(1.0 / (TAU * blob));
            (rate, rate.gp(-1.0 / blob))
        }

        /// The velocity the vortices and their copies induce at a point: each moves the fluid
        /// along the separation's commutator with its rate, the separation turned a quarter
        /// turn; the sense comes from the circulations alone.
        pub fn velocity(&self, point: P) -> P {
            self.separations(point)
                .map(|(s, g)| s.commutator(self.profile(s, g).0))
                .sum()
        }

        /// The velocity gradient at a point. A displacement turns with the rate at the point, and
        /// changes the rate by its component along the separation, which acts on the turned
        /// separation.
        pub fn gradient(&self, point: P) -> Gradient {
            self.separations(point)
                .map(|(s, g)| {
                    let (rate, change) = self.profile(s, g);
                    let turning = P::slot().commutator(rate);
                    let spreading = s.commutator(change).gp(2.0) * s.dot(P::slot());
                    turning + spreading
                })
                .sum()
        }

        /// The derivative of the flow at a point: the divergence plus the vorticity.
        pub fn derivative(&self, point: P) -> Even {
            derivative(self.gradient(point))
        }

        /// The blobs' vorticity at a point, in closed form: `Γ a² / (π (|s|² + a²)²)` summed.
        #[cfg_attr(not(test), allow(dead_code))] // numga's check of the derivative
        pub fn vorticity(&self, point: P) -> Rate {
            let a2 = self.core_radius * self.core_radius;
            self.separations(point)
                .map(|(s, g)| g.gp(a2 / PI / self.blob(s).powi(2)))
                .sum()
        }

        /// The velocity of each vortex: the flow at its centre.
        pub fn drift(&self) -> Vec<P> {
            self.centres.iter().map(|c| self.velocity(*c)).collect()
        }

        /// The vortices moved at the given rates for a time `dt`.
        fn moved(&self, rates: &[P], dt: f64) -> Vortices {
            let centres = self.centres.iter().zip(rates).map(|(c, r)| *c + r.gp(dt));
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
            let rates: Vec<P> = (0..first.len())
                .map(|i| (first[i] + (second[i] + third[i]).gp(2.0) + fourth[i]).gp(1.0 / 6.0))
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
        pub fn mean(&self) -> P {
            self.centres
                .iter()
                .copied()
                .sum::<P>()
                .gp(1.0 / self.centres.len() as f64)
        }
    }

    /// The derivative of a gradient: numga's `(Point * gradient(Point)).contract(1, 2)`, here
    /// the sum `Σ e_i gradient(e_i)` over an orthonormal basis of displacements.
    pub fn derivative(gradient: Gradient) -> Even {
        [at(1.0, 0.0), at(0.0, 1.0)]
            .iter()
            .map(|e| *e * gradient.of(*e))
            .sum()
    }

    /// How much faster the fluid turns than it stretches; positive inside a vortex core.
    pub fn swirl(gradient: Gradient) -> f64 {
        -0.5 * gradient.of(gradient).trace()
    }

    /// The centres at every step: one curve through the plane and time per vortex.
    pub fn worldlines(vortices: &Vortices, dt: f64, steps: usize) -> Vec<Vec<P>> {
        let mut current = vortices.clone();
        let mut lines = vec![current.centres.clone()];
        for _ in 0..steps {
            current = current.step(dt);
            lines.push(current.centres.clone());
        }
        lines
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
    pub const WALKABOUT_DT: f64 = 1.0 / 6.0;
    pub const WALKABOUT_FRAMES: usize = 203;
    pub const WALKABOUT_SUBSTEPS: usize = 4;
    pub const WALKABOUT_ALONG: f64 = 10.0;
    pub const WALKABOUT_ACROSS: f64 = 1.4;
    pub const WALKABOUT_HALF_WIDTH: f64 = 10.6;
    pub const WALKABOUT_HALF_HEIGHT: f64 = 2.9;
    pub const WALKABOUT_VORTICITY_LIMIT: f64 = 8.0;

    /// Worldlines: five equal vortices evenly spaced on a ring around the origin, turning
    /// rigidly, and the leapfrogging pairs, traced for one and a half turns of the ring and three
    /// periods of the leapfrog: the step, and the steps of each.
    pub const RING: usize = 5;
    pub const RING_RADIUS: f64 = 1.0;
    pub const RING_CORE: f64 = 0.05;
    pub const WORLDLINE_DT: f64 = 0.2;
    pub const RING_STEPS: usize = 150;
    pub const LEAPFROG_STEPS: usize = 75;

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

    /// Vortices at the given centres, with the leapfrog's circulations and core.
    fn pairs(centres: Vec<P>) -> Vortices {
        Vortices {
            centres,
            circulations: LEAPFROG_CIRCULATIONS.map(turn).to_vec(),
            core_radius: LEAPFROG_CORE,
            copies: alone(),
        }
    }

    pub fn leapfrog() -> Scene {
        let centres = LEAPFROG_SPANS.iter().map(|s| at(0.0, *s)).collect();
        Scene {
            name: "LEAPFROG",
            vortices: pairs(centres),
            half_width: 2.4,
            half_height: 1.6,
            columns: 300,
            rows: 200,
            dt: 1.0 / 15.0,
            substeps: 2,
            frames: 150,
            vorticity_limit: 8.0,
        }
    }

    pub fn shear_layer() -> Scene {
        let along = |k: usize| (k as f64 + 0.5) / LAYER_VORTICES as f64 * LAYER_PERIOD;
        let centres = (0..LAYER_VORTICES)
            .map(|k| {
                let phase = TAU * along(k) / LAYER_PERIOD;
                let across = LAYER_AMPLITUDE * (wave(2.0 * phase) + LAYER_SEED * wave(phase));
                at(along(k) - LAYER_PERIOD / 2.0, across)
            })
            .collect();
        // The layer's circulation per period is the jump in velocity across it times the period.
        let circulation = -LAYER_JUMP * LAYER_PERIOD / LAYER_VORTICES as f64;
        let copies = (-LAYER_COPIES..=LAYER_COPIES)
            .map(|k| at(f64::from(k) * LAYER_PERIOD, 0.0))
            .collect();
        Scene {
            name: "SHEAR LAYER",
            vortices: Vortices {
                centres,
                circulations: vec![turn(circulation); LAYER_VORTICES],
                core_radius: LAYER_CORE,
                copies,
            },
            half_width: LAYER_PERIOD,
            half_height: 0.4,
            columns: 300,
            rows: 120,
            dt: 1.0 / 60.0,
            substeps: 2,
            frames: 240,
            vorticity_limit: 10.0,
        }
    }

    pub fn gas() -> Scene {
        let mut rng = rng::rng(GAS_SEED);
        let count = GAS_TURNING + GAS_COUNTER;
        let centres = (0..count)
            .map(|_| disc_point(GAS_RADIUS, rng.uniform(), rng.uniform()))
            .collect();
        let mut circulations = vec![turn(1.0); GAS_TURNING];
        circulations.extend([turn(-1.0); GAS_COUNTER]);
        circulations.shuffle(&mut rng);
        Scene {
            name: "GAS",
            vortices: Vortices {
                centres,
                circulations,
                core_radius: GAS_CORE,
                copies: alone(),
            },
            half_width: 3.6,
            half_height: 3.0,
            columns: 300,
            rows: 250,
            dt: 1.0 / 120.0,
            substeps: 8,
            frames: 225,
            vorticity_limit: 15.0,
        }
    }

    /// A point uniform over a disc for two uniform draws: a radius growing as the square root
    /// of the first (equal areas), turned through the second's share of a turn.
    #[allow(clippy::disallowed_methods)] // a radius uniform over the disc's area
    fn disc_point(radius: f64, radius_draw: f64, turn_draw: f64) -> P {
        Rotor::rotation(TAU * turn_draw) >> at(radius * radius_draw.sqrt(), 0.0)
    }

    /// The nudged pairs.
    pub fn walkabout() -> Vortices {
        let centres = WALKABOUT_SPANS
            .iter()
            .zip(WALKABOUT_NUDGES)
            .map(|(s, n)| at(n, *s))
            .collect();
        pairs(centres)
    }

    /// The walkabout's vortices at every frame.
    pub fn walk() -> Vec<Vortices> {
        walkabout().frames(WALKABOUT_DT, WALKABOUT_SUBSTEPS, WALKABOUT_FRAMES)
    }

    /// Five equal vortices evenly spaced on a ring: the first on x, each next turned a fifth of a
    /// turn.
    pub fn ring() -> Vortices {
        let first = at(RING_RADIUS, 0.0);
        Vortices {
            centres: (0..RING)
                .map(|k| Rotor::rotation(TAU * k as f64 / RING as f64) >> first)
                .collect(),
            circulations: vec![turn(1.0); RING],
            core_radius: RING_CORE,
            copies: alone(),
        }
    }

    /// The worldlines of the ring, turning rigidly about its centre, and of the leapfrogging
    /// pairs.
    pub fn braids() -> [Vec<Vec<P>>; 2] {
        [
            worldlines(&ring(), WORLDLINE_DT, RING_STEPS),
            worldlines(&leapfrog().vortices, WORLDLINE_DT, LEAPFROG_STEPS),
        ]
    }
}

use vortices::*;

/// The loop: the longest scene, numga's 240 frames of 40 ms.
const SECONDS: f32 = 9.6;
/// Samples of the swirl across a panel, along its longer side.
const SWIRL_SAMPLES: usize = 140;

/// The scenes and their evolutions, computed once.
struct Scenes {
    scenes: [(Scene, Vec<Vortices>); 3],
    walk: Vec<Vortices>,
    braids: [Vec<Vec<P>>; 2],
}

fn scenes() -> Scenes {
    Scenes {
        scenes: [leapfrog(), shear_layer(), gas()].map(|s| {
            let states = s.evolve();
            (s, states)
        }),
        walk: walk(),
        braids: braids(),
    }
}

/// The height of a braid's drawn time against its width, as numga draws it.
const BRAID_HEIGHT: f64 = 2.5;

/// Worldlines through the plane and time, time upward, traced up to a share of their steps: the
/// curves of `lines[step][vortex]`, coloured by each vortex's sense, centred on their start.
fn draw_braid(
    c: &mut Canvas,
    rect: Rect,
    title: &str,
    lines: &[Vec<P>],
    circulations: &[f64],
    share: f32,
    azimuth: f32,
) {
    let flat: Vec<[f64; 2]> = lines.iter().flatten().map(|p| coordinates(*p)).collect();
    let (lo, hi) = flat
        .iter()
        .fold(([f64::MAX; 2], [f64::MIN; 2]), |(lo, hi), [x, y]| {
            (
                [lo[0].min(*x), lo[1].min(*y)],
                [hi[0].max(*x), hi[1].max(*y)],
            )
        });
    let width = (hi[0] - lo[0]).max(hi[1] - lo[1]);
    let middle = [0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1])];
    let steps = lines.len() - 1;
    let height = BRAID_HEIGHT * width;
    let place3 = |k: usize, p: P| {
        let [x, y] = coordinates(p);
        let t = height * (k as f64 / steps as f64 - 0.5);
        gax::pga3d::Point::xyz(x - middle[0], y - middle[1], t)
    };
    let camera = Camera::orbit(
        rect,
        gax::pga3d::Point::xyz(0.0, 0.0, 0.0),
        20.0,
        azimuth,
        0.25,
        Lens::Parallel(0.62 * height as f32),
    );
    let upto = ((share * steps as f32) as usize).min(steps);
    panel3(c, rect, camera, |sc| {
        for (v, circulation) in circulations.iter().enumerate() {
            let curve: Vec<_> = (0..=upto).map(|k| place3(k, lines[k][v])).collect();
            sc.polyline(&curve, 1.6, sense(*circulation));
            sc.dot(
                curve[upto],
                Marker::Dot,
                5.0,
                sense(*circulation).whitened(0.4),
            );
        }
    });
    let head = rect.top_middle() + Point2::direction(0.0, 12.0);
    c.text(title, head, 10.0, palette::INK, Align::Center);
}

/// The point of the plane under a drawn point.
fn point(q: Point2) -> P {
    let [x, y] = q.to_euclidean();
    at(f64::from(x), f64::from(y))
}

/// Where a point of the plane is drawn.
fn place(p: P) -> Point2 {
    let [x, y] = coordinates(p);
    Point2::xy(x as f32, y as f32)
}

/// The frame a fraction of the way through `frames`.
fn frame_at(fraction: f32, frames: usize) -> usize {
    frame(fraction, frames)
}

/// The half height of equal axes in `rect` that show a window of the given half extents whole.
fn half_height(rect: Rect, half_width: f64, half_height: f64) -> f32 {
    (half_height as f32).max(half_width as f32 * rect.height() / rect.width())
}

/// The light of a sense of turning: counterclockwise (positive) warm, clockwise cool.
fn sense(turning: f64) -> Light {
    if turning > 0.0 {
        palette::ORANGE
    } else {
        palette::SKY
    }
}

/// The light of a vorticity: by its sense, from the backdrop's dark blue to full at the limit.
fn shade(vorticity: f64, limit: f64) -> Light {
    let level = (vorticity.abs() / limit).min(1.0) as f32;
    palette::BOTTOM.blend(sense(vorticity).faded(0.8), level)
}

/// The vorticity over the axes, from the derivative of the flow.
fn draw_vorticity(c: &mut Canvas, ax: &Axes, vortices: &Vortices, limit: f64) {
    ax.image(c, 1, |q| {
        let curl = vortices.derivative(point(q)).e12();
        Some(shade(curl, limit))
    });
}

/// The line where swirl gives way to strain.
fn swirl_line() -> Light {
    palette::INK.faded(0.35)
}

/// The vortices' centres, by their sense.
fn draw_centres(c: &mut Canvas, ax: &Axes, vortices: &Vortices, size: f32) {
    for (centre, circulation) in vortices.centres.iter().zip(&vortices.circulations) {
        let light = sense(circulation.e12()).whitened(0.5);
        ax.scatter(c, &[place(*centre)], Marker::Ring, size, light);
    }
}

/// A scene at a frame: the vorticity in the window around the vortices, the line where swirl
/// gives way to strain, and the centres.
fn draw_scene(c: &mut Canvas, rect: Rect, scene: &Scene, vortices: &Vortices) {
    let half = half_height(rect, scene.half_width, scene.half_height);
    let ax = Axes::equal(rect, place(vortices.mean()), half);
    draw_vorticity(c, &ax, vortices, scene.vorticity_limit);
    let swirling = |p: Point2| swirl(vortices.gradient(point(p))) as f32;
    ax.contour(c, swirling, SWIRL_SAMPLES, 0.0, 1.0, swirl_line());
    // A repeating layer's vortices are too many to mark.
    if scene.vortices.copies.len() == 1 {
        draw_centres(c, &ax, vortices, 5.0);
    }
    let corners = [rect.lo, rect.top_right(), rect.hi, rect.bottom_left()];
    c.polyline(&corners, 1.0, palette::GRID, true);
    let title = rect.top_middle() - Point2::direction(0.0, 6.0);
    c.text(scene.name, title, 12.0, palette::INK, Align::Center);
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
            .map(|v| place(v.centres[k]))
            .collect();
        ax.polyline(c, &path, 1.0, palette::INK.faded(0.4));
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

fn draw(c: &mut Canvas, s: &Scenes, t: f32) {
    backdrop(c);
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
        .part(0.0, 0.6, 0.7, 1.0)
        .inset(40.0, 26.0, 10.0, 26.0);
    draw_walkabout(c, bottom, &s.walk, frame_at(fraction, WALKABOUT_FRAMES));
    let azimuth = -1.2 + core::f32::consts::TAU * fraction;
    let worldlines = screen.part(0.7, 0.58, 1.0, 1.0);
    let ring = ring()
        .circulations
        .iter()
        .map(|g| g.e12())
        .collect::<Vec<_>>();
    draw_braid(
        c,
        worldlines.part(0.0, 0.0, 0.5, 1.0),
        "RING: A BRAID",
        &s.braids[0],
        &ring,
        fraction,
        azimuth,
    );
    draw_braid(
        c,
        worldlines.part(0.5, 0.0, 1.0, 1.0),
        "LEAPFROG",
        &s.braids[1],
        &LEAPFROG_CIRCULATIONS,
        fraction,
        azimuth,
    );
    caption(
        c,
        "VORTICES: THE DERIVATIVE OF THEIR FLOW",
        "SUM E_I GRADIENT(E_I): NO DIVERGENCE, THE VORTICITY; -TR(G(G))/2 > 0 WHERE IT SWIRLS (VGA2D)",
    );
}

fn main() {
    run_with(Anim::new("vortices", SECONDS).size(960, 540), scenes, draw);
}

#[cfg(test)]
mod tests {
    use super::vortices::*;
    use gax::vga2d::Vector;

    /// Offsets on a grid centred on the origin, at the centres of its cells, row by row.
    fn grid(half_width: f64, half_height: f64, columns: usize, rows: usize) -> Vec<P> {
        let at_cell =
            |k: usize, n: usize, half: f64| (k as f64 + 0.5) / n as f64 * 2.0 * half - half;
        (0..rows)
            .flat_map(|r| {
                (0..columns).map(move |c| {
                    at(
                        at_cell(c, columns, half_width),
                        at_cell(r, rows, half_height),
                    )
                })
            })
            .collect()
    }

    /// numga's window on a scene at a frame: the grid centred on the vortices' mean.
    fn window(scene: &Scene, vortices: &Vortices) -> Vec<P> {
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
    fn derivative_is_the_vorticity(vortices: &Vortices, points: &[P], atol: f64) {
        for p in points {
            let curl = Even::new(0.0, vortices.vorticity(*p).e12());
            let difference = vortices.derivative(*p) - curl;
            assert!(vanishes(difference, atol), "{difference:?}");
        }
    }

    /// The circulations as numbers, their senses signed.
    fn strengths(vortices: &Vortices) -> Vec<f64> {
        vortices.circulations.iter().map(|g| g.e12()).collect()
    }

    /// The linear and angular impulses: the circulation-weighted sums of the centres and of
    /// their squared distances from the origin.
    fn impulses(circulations: &[f64], centres: &[P]) -> (P, f64) {
        let linear: P = centres
            .iter()
            .zip(circulations)
            .map(|(c, g)| c.gp(*g))
            .sum();
        let angular: f64 = centres
            .iter()
            .zip(circulations)
            .map(|(c, g)| g * c.dot(*c).s())
            .sum();
        (linear, angular)
    }

    /// A point's coordinates are those it was made from.
    #[test]
    fn points_read_their_coordinates() {
        let [x, y] = coordinates(at(0.3, -1.2));
        assert!(
            (x - 0.3).abs() < 1e-15 && (y + 1.2).abs() < 1e-15,
            "{x} {y}"
        );
    }

    #[test]
    fn the_derivative_of_the_flow_and_the_leapfrog() {
        let scene = leapfrog();
        let vortices = &scene.vortices;
        let points = grid(2.0, 1.5, 9, 7);
        // The gradient is the change in velocity over a small displacement.
        let step = at(0.3, -0.2).gp(1e-5);
        for p in &points {
            let change = (vortices.velocity(*p + step) - vortices.velocity(*p - step)).gp(0.5);
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
            leads.push(coordinates(moving.centres[2] - moving.centres[0])[0]);
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
            let across = v.centres.iter().map(|c| coordinates(*c)[1]);
            across.clone().fold(f64::MIN, f64::max) - across.fold(f64::MAX, f64::min)
        };
        let rolled = scene.vortices.run(scene.dt, scene.frames * scene.substeps);
        assert!(spread(&rolled) > 10.0 * spread(&scene.vortices));
    }

    #[test]
    fn the_gas_keeps_its_impulses() {
        let scene = gas();
        let strengths = strengths(&scene.vortices);
        let (linear, angular) = impulses(&strengths, &scene.vortices.centres);
        let after = scene.vortices.run(scene.dt, 300);
        let (linear_after, angular_after) = impulses(&strengths, &after.centres);
        // Runge–Kutta keeps the linear impulse to round-off; the angular one to the method's
        // accuracy.
        assert!((linear_after - linear).c.iter().all(|x| x.abs() <= 1e-12));
        assert!((angular_after - angular).abs() <= 1e-5);
    }

    /// A mirror turns every circulation the other way, along with every centre: the mirror image
    /// of the gas moves as its mirror image.
    #[test]
    fn the_mirror_image_of_the_gas_moves_as_its_mirror_image() {
        let vortices = gas().vortices;
        let mirror = Vector::new(1.0, 0.4).normalized();
        let reflect = |ps: &[P]| -> Vec<P> { ps.iter().map(|p| mirror >> *p).collect() };
        let mirrored = Vortices {
            centres: reflect(&vortices.centres),
            circulations: vortices.circulations.iter().map(|g| mirror >> *g).collect(),
            copies: reflect(&vortices.copies),
            ..vortices.clone()
        };
        let (g, turned) = (vortices.circulations[0], mirror >> vortices.circulations[0]);
        assert!((turned.e12() + g.e12()).abs() < 1e-15, "{turned:?}");
        let (moved, moved_mirrored) = (vortices.run(0.01, 10), mirrored.run(0.01, 10));
        for (a, b) in reflect(&moved.centres).iter().zip(&moved_mirrored.centres) {
            assert!((*a - *b).c.iter().all(|x| x.abs() <= 1e-12), "{a:?} {b:?}");
        }
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
                    .map(|c| coordinates(*c)[1].abs())
                    .fold(0.0, f64::max),
            );
        }
        let wide: Vec<usize> = (0..farthest.len()).filter(|k| farthest[*k] > 2.0).collect();
        assert!(farthest.iter().copied().fold(0.0, f64::max) > 3.0);
        assert!(wide.windows(2).all(|w| w[1] == w[0] + 1));
        let last = &farthest[farthest.len() - 30..];
        assert!(last.iter().copied().fold(0.0, f64::max) < 1.5);
    }

    /// numga's worldline checks: Runge–Kutta keeps the linear impulse of the ring and of the
    /// leapfrogging pairs to round-off and the angular impulse to the method's accuracy; and
    /// the ring turns rigidly: each vortex's place times its first place is the same for every
    /// vortex.
    #[test]
    fn the_worldlines_keep_their_impulses_and_the_ring_turns_rigidly() {
        let [turning, passing] = braids();
        let ring = strengths(&ring());
        for (circulations, lines) in [
            (&ring[..], &turning),
            (&LEAPFROG_CIRCULATIONS[..], &passing),
        ] {
            let (linear, angular) = impulses(circulations, &lines[0]);
            for centres in lines {
                let (l, a) = impulses(circulations, centres);
                assert!((l - linear).c.iter().all(|x| x.abs() <= 1e-12));
                assert!((a - angular).abs() <= 1e-5, "{a} {angular}");
            }
        }
        for centres in &turning {
            let turned: Vec<Even> = centres
                .iter()
                .zip(&turning[0])
                .map(|(c, f)| *c * *f)
                .collect();
            for t in &turned {
                assert!(vanishes(*t - turned[0], 1e-12), "{t:?} {:?}", turned[0]);
            }
        }
        // One and a half turns of the ring: its first vortex has gone round once and halfway.
        assert_eq!(turning.len(), RING_STEPS + 1);
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
        gax_numga_examples::app::assert_draws_with(super::draw, &super::scenes(), 0.5);
    }
}
