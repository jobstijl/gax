//! numga's `fluids/stokes_swimmer`: swimming without inertia, Purcell's three-link swimmer
//! beside a reciprocal flapper (PGA2D).
//!
//! In slow, thick flow nothing coasts: at every instant the pushes a swimmer's body gives the
//! fluid balance, with no net force and no net turning, and the body moves at exactly the rate
//! that keeps them balanced. The swimmer is a chain of beads; a push on the fluid at a bead sets
//! all of it moving, less with distance, the force spread over a blob of the bead's size (a
//! regularized Stokeslet). A force is a line through the origin (`OriginLine`), read by its
//! pairing with a velocity, a direction; over every pair of beads the Stokeslets make one map
//! from the beads' forces to their velocities.
//!
//! The body's own motion is a rate, a twist, and the commutator of each bead with the rate is
//! that bead's velocity: `bead.commutator(Twist::slot())`, a map `Direction <- Twist`. Its
//! adjugate carries forces back to the forque on the body, the map across the regressive
//! product: a forque paired with a rate is the power the forces deliver to beads moving at that
//! rate. As the swimmer changes shape, one solve finds the rate whose forque cancels the shape's.
//! The rate is linear in the rate of the shape's change, so how far a stroke carries the swimmer
//! depends only on the path the shape takes: a stroke that goes back along its path undoes
//! itself (Purcell's scallop theorem), while Purcell's swimmer takes its two hinges around a
//! loop and steps ahead with every cycle.
//!
//! The animation shows both swimmers in their flows (the speed of the fluid as light, its
//! direction as short strokes), their trails and the beads' forces; how far each has come along
//! x; and their strokes in the plane of the two hinge angles.

use gax_numga_examples::signal::phasor;
use gax_numga_examples::{
    Align, Anim, Axes, Canvas, Light, Marker, ORIGIN2, Point2, Rect, backdrop, caption, palette,
    reach2, run_with,
};

mod swimmer {
    use gax::Unit;
    use gax::field::{Field, SparseMap};
    use gax::pga2d::{Direction, Forque, Line, Motor, OriginLine, Point, Twist};

    /// A force: a line through the origin, extrinsic, read by its pairing with a velocity.
    pub type Force = OriginLine;
    /// The flow per unit blob force: `Direction <- Force`.
    pub type Mobility = Direction<(Force,)>;
    /// A bead's velocity per unit body rate.
    pub type Rigid = Direction<(Twist,)>;
    /// The forque on the body per unit body rate.
    pub type Resistance = Forque<(Twist,)>;

    /// The force whose metric-free complement is a direction (a force turned with the body is
    /// carried as the direction it stands for).
    pub fn force_of(d: Direction) -> Force {
        d.undual().cast::<Force>()
    }

    /// The unit force along `x` and `y`, read through the directions they pair with.
    pub fn unit_force() -> Direction<(Force,)> {
        Force::slot().dual().cast::<Direction>()
    }

    /// The fluid velocity at a target per unit blob force at a source, for regularized
    /// Stokeslets of blob `radius`: along the force, and further along the separation by the
    /// force's pairing with it, both falling off with the softened distance.
    pub fn stokeslet(target: Point, source: Point, radius: f64) -> Mobility {
        // The difference of two points of one weight is a direction.
        let separation = (target - source).cast::<Direction>();
        // The squared distance, the squared norm of the joining line, softened by the radius.
        let softened = (target & source).norm_squared() + radius * radius;
        #[allow(clippy::disallowed_methods)] // the Stokeslet's softened distance cubed
        let cube = softened * softened.sqrt();
        let along = unit_force() * (softened + radius * radius);
        let across = separation * (Force::slot() & separation);
        (along + across) * (1.0 / cube)
    }

    /// The beads' map from their forces to their velocities, every pair of beads a cell: a
    /// correlation, symmetric and positive definite in the pairing of a force with a velocity,
    /// its power.
    pub fn mobility(sites: &[Point], radius: f64) -> SparseMap<Mobility> {
        let n = sites.len();
        let cells = (0..n * n).map(|k| {
            let (i, j) = (k / n, k % n);
            (i, j, stokeslet(sites[i], sites[j], radius))
        });
        SparseMap::from_triples(cells, (n, n))
    }

    /// The forces that move the beads at the given velocities: conjugate gradients on the
    /// correlation.
    fn forces_for(mobility: &SparseMap<Mobility>, velocities: Vec<Direction>) -> Vec<Force> {
        mobility.solve(&Field::new(velocities)).into_sites()
    }

    /// Each bead's velocity per unit body rate: its commutator with the rate.
    pub fn rigid(sites: &[Point]) -> Vec<Rigid> {
        sites
            .iter()
            .map(|s| s.commutator(Twist::slot()).cast::<Direction>())
            .collect()
    }

    /// The forque the forces put on the body: each bead's rigid map carried back across the
    /// regressive product, its adjugate, applied to its force, summed.
    pub fn net_forque(rigid: &[Rigid], forces: &[Force]) -> Forque {
        rigid
            .iter()
            .zip(forces)
            .map(|(r, f)| r.adjugate().of(*f))
            .sum()
    }

    /// The body's rate and the beads' forces of a force- and torque-free swimmer with the given
    /// beads and shape velocities, in the body's frame.
    pub fn swim(sites: &[Point], shape_velocity: &[Direction], radius: f64) -> (Twist, Vec<Force>) {
        let mobility = mobility(sites, radius);
        let rigid = rigid(sites);
        // The forque per unit rate, column by column of the twists.
        let basis = [
            Twist::new(1.0, 0.0, 0.0),
            Twist::new(0.0, 1.0, 0.0),
            Twist::new(0.0, 0.0, 1.0),
        ];
        let columns = basis.map(|t| {
            let velocities = rigid.iter().map(|r| r.of(t)).collect();
            net_forque(&rigid, &forces_for(&mobility, velocities))
        });
        let resistance = Resistance::from_images(columns);
        // Changing shape alone would push on the fluid; the body moves at the rate whose forque
        // cancels it.
        let shape_forque = net_forque(&rigid, &forces_for(&mobility, shape_velocity.to_vec()));
        let rate = -resistance.solve(shape_forque);
        let velocities = rigid
            .iter()
            .zip(shape_velocity)
            .map(|(r, v)| r.of(rate) + *v)
            .collect();
        (rate, forces_for(&mobility, velocities))
    }

    /// The fluid's velocity at a point from the beads' forces.
    pub fn flow_at(point: Point, beads: &[Point], forces: &[Force], radius: f64) -> Direction {
        beads
            .iter()
            .zip(forces)
            .map(|(b, f)| stokeslet(point, *b, radius).of(*f))
            .sum()
    }

    /// The pose after each step of the body's rate, each in the body's own frame (on the right).
    pub fn integrate_pose(rates: &[Twist], dt: f64) -> Vec<Unit<Motor>> {
        let mut poses = vec![Unit::<Motor>::identity()];
        for rate in rates {
            let last = poses[poses.len() - 1];
            poses.push(last * (*rate * (-dt / 2.0)).exp());
        }
        poses
    }

    /// The beads of a three-link Purcell swimmer at the hinge angles, and their velocities in
    /// the body's frame at the hinges' rates: the centre link along x through the origin, each
    /// arm turned about its hinge, a twist, the meet of the lines along and across the swimmer.
    pub fn three_link_shape(
        half_span: f64,
        centre: &[f64],
        arm: &[f64],
        angles: [f64; 2],
        rates: [f64; 2],
    ) -> (Vec<Point>, Vec<Direction>) {
        let along = Point::direction(1.0, 0.0);
        let across = Line::new(0.0, 1.0, 0.0);
        let left_hinge: Twist = Line::new(1.0, 0.0, half_span) ^ across;
        let right_hinge: Twist = Line::new(1.0, 0.0, -half_span) ^ across;
        let left_turn = (left_hinge * (angles[0] / 2.0)).exp();
        let right_turn = (right_hinge * (-angles[1] / 2.0)).exp();
        let origin = Point::origin();
        let left: Vec<Point> = arm
            .iter()
            .rev()
            .map(|a| left_turn >> (origin - along * (half_span + a)))
            .collect();
        let middle: Vec<Point> = centre.iter().map(|c| origin + along * *c).collect();
        let right: Vec<Point> = arm
            .iter()
            .map(|a| right_turn >> (origin + along * (half_span + a)))
            .collect();
        // Each arm turns about its hinge at the hinge's rate (the left one's angle counting the
        // other way round); the centre link does not move in the body's frame.
        let velocity =
            |b: &Point, rate: Twist| -> Direction { b.commutator(rate).cast::<Direction>() };
        let mut velocities: Vec<Direction> = left
            .iter()
            .map(|b| velocity(b, left_hinge * -rates[0]))
            .collect();
        velocities.extend(middle.iter().map(|_| Direction::new(0.0, 0.0)));
        velocities.extend(right.iter().map(|b| velocity(b, right_hinge * rates[1])));
        let sites = left.into_iter().chain(middle).chain(right).collect();
        (sites, velocities)
    }

    // --- scenes

    pub const HINGE_HALF_SPAN: f64 = 0.8;
    pub const CENTRE_OFFSETS: [f64; 3] = [-0.5, 0.0, 0.5];
    pub const ARM_OFFSETS: [f64; 3] = [0.35, 0.7, 1.05];
    pub const BEAD_RADIUS: f64 = 0.14;
    pub const AMPLITUDE: f64 = 1.05;
    pub const CYCLES: usize = 3;
    pub const STEPS_PER_CYCLE: usize = 60;

    /// A swimmer's stroke over the cycles: the hinge angles, the beads in the body's frame, the
    /// body's rate and the beads' forces at every step, and the poses.
    pub struct Stroke {
        pub angles: Vec<[f64; 2]>,
        pub sites: Vec<Vec<Point>>,
        pub rates: Vec<Twist>,
        pub forces: Vec<Vec<Force>>,
        pub poses: Vec<Unit<Motor>>,
    }

    impl Stroke {
        /// The swimmer's centre after each step.
        pub fn centres(&self) -> Vec<Point> {
            self.poses.iter().map(|m| *m >> Point::origin()).collect()
        }
        /// The beads in the world at a step.
        pub fn world_sites(&self, step: usize) -> Vec<Point> {
            self.sites[step]
                .iter()
                .map(|s| self.poses[step] >> *s)
                .collect()
        }
        /// The beads' forces in the world at a step, carried as the directions they stand for.
        pub fn world_forces(&self, step: usize) -> Vec<Force> {
            let m = self.poses[step];
            self.forces[step]
                .iter()
                .map(|f| force_of(m >> f.dual().cast::<Direction>()))
                .collect()
        }
    }

    /// Purcell's stroke, a circle in the hinge angles (`loop_ = true`), or the reciprocal one,
    /// an asymmetric line retraced: `angles(phase)` and their rates.
    pub fn stroke(loop_: bool, phasor: impl Fn(f64) -> (f64, f64)) -> Stroke {
        let steps = CYCLES * STEPS_PER_CYCLE;
        let dt = core::f64::consts::TAU / STEPS_PER_CYCLE as f64;
        let mut s = Stroke {
            angles: Vec::new(),
            sites: Vec::new(),
            rates: Vec::new(),
            forces: Vec::new(),
            poses: Vec::new(),
        };
        for k in 0..steps {
            let (c, si) = phasor(k as f64 * dt);
            let (angles, rates) = if loop_ {
                (
                    [AMPLITUDE * c, AMPLITUDE * si],
                    [-AMPLITUDE * si, AMPLITUDE * c],
                )
            } else {
                (
                    [AMPLITUDE * c, -0.5 * AMPLITUDE * c],
                    [-AMPLITUDE * si, 0.5 * AMPLITUDE * si],
                )
            };
            let (sites, velocities) = three_link_shape(
                HINGE_HALF_SPAN,
                &CENTRE_OFFSETS,
                &ARM_OFFSETS,
                angles,
                rates,
            );
            let (rate, forces) = swim(&sites, &velocities, BEAD_RADIUS);
            s.angles.push(angles);
            s.sites.push(sites);
            s.rates.push(rate);
            s.forces.push(forces);
        }
        s.poses = integrate_pose(&s.rates, dt);
        s
    }
}

use swimmer::*;

/// The loop: numga's 90 frames of 40 ms, a frame every two steps.
const SECONDS: f32 = 3.6;
const STEPS_PER_FRAME: usize = 2;
/// The flow's samples across and up each swimmer's window, and its half height.
const FLOW_COLUMNS: usize = 41;
const FLOW_ROWS: usize = 19;
const WATER_HALF_HEIGHT: f32 = 1.55;

/// The two strokes, computed once.
struct Scenes {
    strokes: [Stroke; 2],
}

fn cos_sin(angle: f64) -> (f64, f64) {
    let p = phasor(angle);
    (p.e20(), p.e01())
}

fn scenes() -> Scenes {
    Scenes {
        strokes: [stroke(true, cos_sin), stroke(false, cos_sin)],
    }
}

/// The swimmers' lights: Purcell's, the reciprocal one's.
fn swimmer_light(k: usize) -> Light {
    [palette::SKY, palette::ORANGE][k]
}

/// The body through its beads and hinges: the hinges a bead and a half from the centre link's
/// end beads.
fn body(beads: &[gax::pga2d::Point]) -> (Vec<Point2>, [Point2; 2]) {
    let p = |i: usize| reach2(beads[i] - gax::pga2d::Point::origin());
    let hinge = |end: usize| p(end) + (p(end) - p(4)) * 0.5;
    let hinges = [hinge(3), hinge(5)];
    let mut chain: Vec<Point2> = (0..3).map(p).collect();
    chain.push(hinges[0]);
    chain.extend((3..6).map(p));
    chain.push(hinges[1]);
    chain.extend((6..9).map(p));
    (chain, hinges)
}

/// A swimmer in its flow at a step: the fluid's speed as light, its direction as strokes, the
/// trail of its centre, the body and the beads' forces.
fn draw_swimmer(c: &mut Canvas, rect: Rect, s: &Stroke, step: usize, light: Light, name: &str) {
    let centres = s.centres();
    let xs: Vec<f32> = centres.iter().map(|p| p.to_euclidean()[0] as f32).collect();
    let (left, right) = (
        xs.iter().copied().fold(0.0, f32::min) - 2.3,
        xs.iter().copied().fold(0.0, f32::max) + 2.3,
    );
    let ax = Axes::new(rect, [left, right], [-WATER_HALF_HEIGHT, WATER_HALF_HEIGHT]);
    let (beads, forces) = (s.world_sites(step), s.world_forces(step));
    let flow = |q: Point2| {
        let [x, y] = q.to_euclidean();
        let at = gax::pga2d::Point::xy(f64::from(x), f64::from(y));
        flow_at(at, &beads, &forces, BEAD_RADIUS)
    };
    ax.image(c, 1, |q| {
        let speed = gax::IdealNorm::ideal_norm(flow(q)) as f32;
        Some(palette::BOTTOM.mix_light(palette::BLUE.faded(0.5), (speed / 1.2).min(1.0)))
    });
    let [x0, x1, y0, y1] = [left, right, -WATER_HALF_HEIGHT, WATER_HALF_HEIGHT];
    for i in 0..FLOW_COLUMNS {
        for j in 0..FLOW_ROWS {
            let at = Point2::xy(
                x0 + (x1 - x0) * (i as f32 + 0.5) / FLOW_COLUMNS as f32,
                y0 + (y1 - y0) * (j as f32 + 0.5) / FLOW_ROWS as f32,
            );
            let v = reach2(flow(at)) - ORIGIN2;
            let length = v.ideal_norm().max(1e-9);
            let step = v.gp(0.12 / length.max(0.3));
            ax.line(c, at - step, at + step, 0.8, palette::INK.faded(0.4));
        }
    }
    let trail: Vec<Point2> = centres[..=step]
        .iter()
        .map(|p| reach2(*p - gax::pga2d::Point::origin()))
        .collect();
    ax.polyline(c, &trail, 1.2, palette::RED.faded(0.8));
    let (chain, hinges) = body(&beads);
    ax.polyline(c, &chain, 2.6, light);
    let dots: Vec<Point2> = beads
        .iter()
        .map(|b| reach2(*b - gax::pga2d::Point::origin()))
        .collect();
    ax.scatter(c, &dots, Marker::Dot, 7.0, light.whitened(0.3));
    ax.scatter(c, &hinges, Marker::Ring, 7.0, palette::INK);
    // The forces at one scale for the whole stroke: the strongest half a unit long.
    let strongest = s
        .forces
        .iter()
        .flatten()
        .map(|f| f.norm())
        .fold(0.0, f64::max);
    for (b, f) in dots.iter().zip(&forces) {
        let push = reach2(f.dual().cast::<gax::pga2d::Direction>()) - ORIGIN2;
        let tip = *b + push.gp((0.5 / strongest) as f32);
        ax.arrow(c, *b, tip, 1.2, 5.0, palette::GREEN);
    }
    ax.frame(c, name, "", "");
}

/// How far each swimmer has come along x against time.
fn draw_paths(c: &mut Canvas, rect: Rect, s: &Scenes, step: usize) {
    let steps = s.strokes[0].poses.len() as f32;
    let paths: Vec<Vec<Point2>> = s
        .strokes
        .iter()
        .map(|st| {
            st.centres()
                .iter()
                .enumerate()
                .map(|(k, p)| Point2::xy(k as f32 / steps, p.to_euclidean()[0] as f32))
                .collect()
        })
        .collect();
    let ys = paths.iter().flatten().map(|p| p.to_euclidean()[1]);
    let (lo, hi) = (ys.clone().fold(0.0, f32::min), ys.fold(0.0, f32::max));
    let ax = Axes::new(rect, [0.0, 1.0], [lo - 0.1, hi + 0.1]);
    for (k, path) in paths.iter().enumerate() {
        ax.polyline(c, path, 1.4, swimmer_light(k));
        ax.scatter(
            c,
            &[path[step]],
            Marker::Dot,
            6.0,
            swimmer_light(k).whitened(0.3),
        );
    }
    ax.frame(c, "", "TIME", "X");
}

/// The two strokes in the plane of the hinge angles.
fn draw_hinges(c: &mut Canvas, rect: Rect, s: &Scenes, step: usize) {
    let half = (AMPLITUDE * 1.25) as f32;
    let ax = Axes::equal(rect, Point2::xy(0.0, 0.0), half);
    ax.line(
        c,
        Point2::xy(-half * 2.0, 0.0),
        Point2::xy(half * 2.0, 0.0),
        1.0,
        palette::GRID,
    );
    ax.line(
        c,
        Point2::xy(0.0, -half),
        Point2::xy(0.0, half),
        1.0,
        palette::GRID,
    );
    for (k, st) in s.strokes.iter().enumerate() {
        let path: Vec<Point2> = st
            .angles
            .iter()
            .map(|a| Point2::xy(a[0] as f32, a[1] as f32))
            .collect();
        ax.polyline(c, &path, 1.4, swimmer_light(k));
        let now = path[step.min(path.len() - 1)];
        ax.scatter(c, &[now], Marker::Dot, 7.0, swimmer_light(k).whitened(0.3));
    }
    ax.text(
        c,
        Point2::xy(half * 0.55, -0.22),
        "LEFT HINGE",
        8.0,
        palette::INK.faded(0.6),
        Align::Center,
    );
    ax.text(
        c,
        Point2::xy(0.06, half * 0.92),
        "RIGHT HINGE",
        8.0,
        palette::INK.faded(0.6),
        Align::Left,
    );
}

fn draw(c: &mut Canvas, s: &Scenes, t: f32) {
    backdrop(c);
    let steps = CYCLES * STEPS_PER_CYCLE;
    let frames = steps / STEPS_PER_FRAME;
    let frame = (((t / SECONDS).rem_euclid(1.0)) * frames as f32) as usize % frames;
    let step = frame * STEPS_PER_FRAME;
    let screen = c.rect();
    let (w, h) = (screen.width(), screen.height());
    let top = Rect::new(0.0, h * 0.13, w * 0.66, h);
    let names = [
        "PURCELL'S LOOP: A STEP EVERY CYCLE",
        "A RECIPROCAL STROKE: BACK WHERE IT STARTED",
    ];
    for (k, stroke) in s.strokes.iter().enumerate() {
        let rect = top
            .part(0.0, k as f32 * 0.5, 1.0, k as f32 * 0.5 + 0.5)
            .inset(30.0, 20.0, 8.0, 8.0);
        draw_swimmer(c, rect, stroke, step, swimmer_light(k), names[k]);
    }
    let side = Rect::new(w * 0.66, h * 0.13, w, h);
    draw_paths(
        c,
        side.part(0.0, 0.0, 1.0, 0.5).inset(30.0, 20.0, 14.0, 30.0),
        s,
        step,
    );
    draw_hinges(
        c,
        side.part(0.0, 0.5, 1.0, 1.0).inset(14.0, 10.0, 14.0, 8.0),
        s,
        step,
    );
    caption(
        c,
        "STOKES SWIMMERS: NOTHING COASTS",
        "BEAD VELOCITY = BEAD X TWIST; ITS ADJUGATE SUMS THE FORCES INTO THE FORQUE, SOLVED TO ZERO (PGA2D)",
    );
}

fn main() {
    run_with(
        Anim::new("stokes_swimmer", SECONDS).size(960, 540),
        scenes,
        draw,
    );
}

#[cfg(test)]
mod tests {
    use super::swimmer::*;
    use gax::pga2d::{Line, Point};

    /// At zero separation the regularized Stokeslet is `2 / radius` times the force's dual.
    #[test]
    fn a_blob_at_its_own_bead() {
        let own = stokeslet(Point::origin(), Point::origin(), BEAD_RADIUS);
        let expected = unit_force() * (2.0 / BEAD_RADIUS);
        assert!(
            (own - expected)
                .c
                .iter()
                .flatten()
                .all(|x| x.abs() <= 1e-12)
        );
    }

    /// Every step is free of force and torque; after each cycle the reciprocal swimmer is back
    /// at its starting pose, while Purcell's advances by the same distance each cycle.
    #[test]
    fn strokes_are_force_free_and_only_the_loop_advances() {
        let cos_sin = super::cos_sin;
        let [purcell, flapper] = [stroke(true, cos_sin), stroke(false, cos_sin)];
        for s in [&purcell, &flapper] {
            for (sites, forces) in s.sites.iter().zip(&s.forces) {
                let forque = net_forque(&rigid(sites), forces);
                assert!(forque.c.iter().all(|x| x.abs() <= 1e-12), "{forque:?}");
            }
        }
        // The tolerance tests the steps per cycle.
        for cycle in 1..=CYCLES {
            let pose = flapper.poses[cycle * STEPS_PER_CYCLE];
            let start = flapper.poses[0];
            assert!(
                pose.c
                    .iter()
                    .zip(start.c)
                    .all(|(a, b)| (a - b).abs() <= 1e-4)
            );
        }
        let x = Line::new(1.0, 0.0, 0.0);
        let centres = purcell.centres();
        let along: Vec<f64> = (0..=CYCLES)
            .map(|k| (x & (centres[k * STEPS_PER_CYCLE] - Point::origin())).s())
            .collect();
        let advances: Vec<f64> = along.windows(2).map(|w| w[1] - w[0]).collect();
        assert!(advances.iter().all(|a| a.abs() > 0.04), "{advances:?}");
        assert!(
            advances
                .iter()
                .all(|a| (a / advances[0] - 1.0).abs() <= 1e-2),
            "{advances:?}"
        );
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws_with(super::draw, &super::scenes(), 0.5);
    }
}
