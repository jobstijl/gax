//! numga's `math/spherical_synchronization`: directions on a sphere turning while attracting one
//! another, carried by one evolving Möbius transformation, in the algebra `x+y+z+t-` of the
//! sphere's conformal geometry (numga's spacetime algebra).
//!
//! Spherical synchronization is the Kuramoto model on a sphere: a population of directions,
//! each turning at the same natural rate and each attracted towards the population's mean,
//! gathers into one direction. However many directions there are, the population follows a
//! single Möbius transformation of the sphere (Lipton, Mirollo and Strogatz), six degrees of
//! freedom.
//!
//! A point of the sphere is a null ray, `direction + time`, an intrinsic trivector (numga's
//! antivector, ADR-051), and a circle a cut, a vector, through which a point passes when
//! `cut & point == 0`. A rotor of the algebra moves points and cuts alike and keeps the
//! sphere's circles and their angles: it is the Möbius transformation, and one rotor carries
//! the population and a grid of circles together. Its generator is the shared rotation less a
//! boost towards the population's mean, `mean.commutator(time)`, the turn from time towards the
//! mean, which on the sphere is a tangential attraction; the population feeds back into it as it
//! moves. Each step takes the generator at the step's midpoint.
//!
//! In the vector notation of the Kuramoto model each direction moves as `ẋᵢ = ω × xᵢ + K (x̄ −
//! (x̄ · xᵢ) xᵢ)`; the tests integrate those equations directly (numga: scipy's `solve_ivp`;
//! here the classical Runge-Kutta steps of `mechanics_lie::rk4`, at a fine step) and find the
//! common rotor converging to them at second order.
//!
//! The animation: the population, coloured per particle, gathering on the see-through sphere
//! while the carried circles (meridians and parallels at the start) keep their circles and
//! their angles; the far side shows faintly through the near one.

use gax_numga_examples::points::circle;
use gax_numga_examples::signal::frame;
use gax_numga_examples::{
    Anim, Camera, Canvas, Lens, Light, ORIGIN3, Point2, Rect, backdrop, caption, palette, reach3,
    run_with,
};

#[path = "../shared/spacetime.rs"]
mod spacetime;

mod synchronization {
    #[cfg(test)]
    use super::spacetime::Vector;
    use super::spacetime::{Bivector, Direction, Even, space, t, time, x, y, z};

    /// A circle on the sphere: a cut, a vector.
    #[cfg(test)]
    pub type Cut = Vector;
    /// A point of the sphere, a null ray `direction + time`: an intrinsic trivector.
    pub type Point = Direction;
    /// A Möbius transformation.
    pub type Rotor = Even;

    /// The unit directions on the sphere of null rays: each scaled to unit time, less its time.
    pub fn directions(p: Point) -> Point {
        p.gp(1.0 / (t() & p).s()) - time()
    }

    /// The mean of directions.
    pub fn mean(points: impl ExactSizeIterator<Item = Point>) -> Point {
        let n = points.len() as f64;
        points.fold(Point::zero(), |sum, p| sum + p).gp(1.0 / n)
    }

    /// The shared rotation, less a boost towards the population's mean direction.
    pub fn generator(rotor: Rotor, seeds: &[Point], rotation: Bivector, coupling: f64) -> Bivector {
        let mean = mean(seeds.iter().map(|s| directions(rotor >> *s)));
        // A boost towards the mean becomes tangential attraction on the sphere.
        rotation - mean.commutator(time()) * coupling
    }

    /// One step of the common transformation, the generator taken at the step's midpoint.
    pub fn step(
        rotor: Rotor,
        seeds: &[Point],
        rotation: Bivector,
        coupling: f64,
        dt: f64,
    ) -> Rotor {
        let flow = |at: Rotor, share: f64| {
            generator(at, seeds, rotation, coupling)
                .gp(share)
                .exp()
                .into_inner()
        };
        let halfway = flow(rotor, dt / 4.0) * rotor;
        flow(halfway, dt / 2.0) * rotor
    }

    /// The common transformation from `initial` and after each of `steps` steps.
    pub fn motion(
        initial: Rotor,
        seeds: &[Point],
        rotation: Bivector,
        coupling: f64,
        dt: f64,
        steps: usize,
    ) -> Vec<Rotor> {
        let mut rotors = vec![initial];
        for _ in 0..steps {
            let last = rotors[rotors.len() - 1];
            rotors.push(step(last, seeds, rotation, coupling, dt));
        }
        rotors
    }

    /// The unit direction at a longitude and a latitude: `x` raised towards `z`, then turned
    /// about `z`, as a point of the sphere.
    pub fn sphere_point(longitude: f64, latitude: f64) -> Point {
        let xy = (x() ^ y()).cast::<Bivector>();
        let zx = (z() ^ x()).cast::<Bivector>();
        let turn = xy.gp(-longitude / 2.0).exp() * zx.gp(latitude / 2.0).exp();
        (turn >> space()[0]) + time()
    }

    // --- the scene ----------------------------------------------------------------------

    /// The population, spread over the sphere by the golden angle; the time the motion runs
    /// for and its steps, a frame each; the strength of the attraction, the shared rotation's
    /// speed and the boost of the starting transformation.
    pub const PARTICLES: usize = 192;
    pub const SPAN: f64 = 5.0;
    pub const STEPS: usize = 150;
    pub const COUPLING: f64 = 1.2;
    pub const ROTATION_SPEED: f64 = 0.25;
    pub const INITIAL_BIAS: f64 = 0.24;
    /// The grid: meridians through the poles, and parallels at latitudes up to a third of a
    /// turn either way.
    pub const MERIDIANS: usize = 4;
    pub const PARALLELS: usize = 5;

    /// The population at the start, spread evenly in height and by the golden angle around.
    pub fn seeds() -> Vec<Point> {
        // The golden angle, π (3 - √5) = 2π (2 - φ).
        let golden_angle = core::f64::consts::TAU * (2.0 - gax_numga_examples::measure::GOLDEN);
        (0..PARTICLES)
            .map(|k| {
                let longitude = (golden_angle * k as f64) % core::f64::consts::TAU;
                let n = PARTICLES as f64;
                let height = -1.0 + 1.0 / n + (2.0 - 2.0 / n) * k as f64 / (n - 1.0);
                // The latitudes of points evenly spread in height: a sampling rule, numga's.
                #[allow(clippy::disallowed_methods)]
                let latitude = height.asin();
                sphere_point(longitude, latitude)
            })
            .collect()
    }

    /// The shared rotation, about `z`.
    pub fn rotation() -> Bivector {
        (x() ^ y()).cast::<Bivector>() * -ROTATION_SPEED
    }

    /// The starting transformation: a boost along `x`.
    pub fn initial() -> Rotor {
        let xt = (x() ^ t()).cast::<Bivector>();
        xt.gp(-INITIAL_BIAS).exp().into_inner()
    }

    /// The parallels' latitudes, evenly from minus a third of a turn to a third.
    pub fn parallel_latitude(k: usize) -> f64 {
        let third = core::f64::consts::TAU / 6.0;
        -third + 2.0 * third * k as f64 / (PARALLELS - 1) as f64
    }

    /// The grid's cuts: great circles through the poles, the cut `y` turned about `z`, and the
    /// parallels, `z + t sin(latitude)`, whose points lie at height minus that sine.
    #[cfg(test)]
    pub fn grid() -> Vec<Cut> {
        let xy = (x() ^ y()).cast::<Bivector>();
        let meridians = (0..MERIDIANS).map(|k| {
            let longitude = core::f64::consts::PI * k as f64 / MERIDIANS as f64;
            xy.gp(-longitude / 2.0).exp() >> y()
        });
        let parallels = (0..PARALLELS).map(|k| {
            let height = super::spacetime::coordinates(sphere_point(0.0, parallel_latitude(k)))[2];
            z() + t() * height
        });
        meridians.chain(parallels).collect()
    }

    /// Points along each of the grid's circles: the meridians' great circles, and the
    /// parallels at minus their latitude.
    pub fn grid_points(samples: usize) -> Vec<Vec<Point>> {
        let around = |k: usize| core::f64::consts::TAU * k as f64 / samples as f64;
        let meridians = (0..MERIDIANS).map(|m| {
            let longitude = core::f64::consts::PI * m as f64 / MERIDIANS as f64;
            (0..=samples)
                .map(|k| sphere_point(longitude, around(k)))
                .collect()
        });
        let parallels = (0..PARALLELS).map(|p| {
            (0..=samples)
                .map(|k| sphere_point(around(k), -parallel_latitude(p)))
                .collect()
        });
        meridians.chain(parallels).collect()
    }

    /// numga's scene: the transformation the population's own attraction drives, from the start
    /// and after each step.
    pub fn synchronization() -> Vec<Rotor> {
        let seeds = seeds();
        motion(
            initial(),
            &seeds,
            rotation(),
            COUPLING,
            SPAN / STEPS as f64,
            STEPS,
        )
    }
}

use spacetime::Direction;
use synchronization::*;

/// numga's animation: a frame of 40 ms per step.
const SECONDS: f32 = (STEPS + 1) as f32 * 0.04;
/// The samples around each of the grid's circles.
const SAMPLES: usize = 96;
/// The particles' colours, cycled.
const COLOURS: [Light; 5] = [
    palette::ORANGE,
    palette::YELLOW,
    palette::SKY,
    palette::BLUE,
    palette::PURPLE,
];

/// What the frames draw: the transformation at each step, the population at the start, and
/// points along the grid's circles at the start.
struct Scene {
    rotors: Vec<Rotor>,
    seeds: Vec<Point>,
    grid: Vec<Vec<Point>>,
}

fn scene() -> Scene {
    Scene {
        rotors: synchronization(),
        seeds: seeds(),
        grid: grid_points(SAMPLES),
    }
}

/// The view: looking at the sphere from a little above, facing the side the population gathers
/// on, showing a half width of 1.07 radii.
fn camera(rect: Rect) -> Camera {
    Camera::orbit(
        rect,
        ORIGIN3,
        6.0,
        (-135f32).to_radians(),
        20f32.to_radians(),
        Lens::Parallel(1.07),
    )
}

/// A point of the sphere as a point of space, and whether it lies on the near side.
fn place(cam: &Camera, p: Direction) -> (Option<Point2>, bool) {
    let at = reach3(directions(p));
    (cam.px(at), cam.depth(at) < cam.depth(ORIGIN3))
}

/// One side of the sphere: the carried circles and the population, `faded` on the far side.
fn side(c: &mut Canvas, cam: &Camera, sc: &Scene, rotor: Rotor, near: bool) {
    let strength = if near { 1.0 } else { 0.22 };
    for samples in &sc.grid {
        let points: Vec<_> = samples.iter().map(|p| place(cam, rotor >> *p)).collect();
        let segments: Vec<[Point2; 2]> = points
            .windows(2)
            .filter_map(|w| match (w[0], w[1]) {
                ((Some(a), sa), (Some(b), sb)) if sa == near && sb == near => Some([a, b]),
                _ => None,
            })
            .collect();
        c.stroke(&segments, 1.1, palette::GREEN.faded(0.8 * strength));
    }
    for (k, seed) in sc.seeds.iter().enumerate() {
        if let (Some(at), facing) = place(cam, rotor >> *seed)
            && facing == near
        {
            let colour = COLOURS[k % COLOURS.len()];
            c.disk(at, if near { 3.0 } else { 2.2 }, colour.faded(strength));
        }
    }
}

fn draw(c: &mut Canvas, sc: &Scene, t: f32) {
    backdrop(c);
    let screen = c.rect();
    let step = frame(t / SECONDS, sc.rotors.len());
    let rotor = sc.rotors[step];
    let side_length = screen.height() * 0.86;
    let rect = Rect::around(
        screen.centre() + Point2::direction(0.0, screen.height() * 0.06),
        Point2::direction(side_length / 2.0, side_length / 2.0),
    );
    let cam = camera(rect);
    side(c, &cam, sc, rotor, false);
    // The sphere, see-through, and its outline.
    if let Some(centre) = cam.px(ORIGIN3) {
        let radius = side_length / 2.0 / 1.07;
        let outline = circle(centre, radius, 128);
        c.fill(&outline, palette::BLUE.faded(0.15), 0.3);
        c.polyline(&outline, 1.0, palette::GRID.faded(4.0), false);
    }
    side(c, &cam, sc, rotor, true);
    let population: Vec<Point> = sc.seeds.iter().map(|s| directions(rotor >> *s)).collect();
    let mean = spacetime::coordinates(mean(population.into_iter()));
    let coherence: f64 = mean[..3].iter().map(|m| m * m).sum();
    let note = format!("STEP {step:3} OF {STEPS}, MEAN DIRECTION SQUARED {coherence:.3}");
    let at = screen.bottom_middle() - Point2::direction(0.0, 14.0);
    c.text(
        &note,
        at,
        12.0,
        palette::YELLOW,
        gax_numga_examples::Align::Center,
    );
    caption(
        c,
        "SPHERICAL SYNCHRONIZATION",
        "ONE ROTOR OF X+Y+Z+T- CARRIES THE POPULATION AND THE CIRCLES: A MOBIUS FLOW",
    );
}

fn main() {
    run_with(
        Anim::new("spherical synchronization", SECONDS).size(960, 540),
        scene,
        draw,
    );
}

#[cfg(test)]
mod tests {
    use super::spacetime::{
        self, Bivector, Direction, Vector, coordinates, space, t, time, x, y, z,
    };
    use super::synchronization::*;
    use core::ops::{Add, Mul};
    use gax::ApproxEq;
    use gax::vga3d;
    use gax_numga_examples::shared::mechanics_lie::rk4;

    /// numga's test population.
    fn population() -> Vec<Point> {
        let longitudes = [-0.6, 0.4, 1.2, 2.3, 4.2];
        let latitudes = [-0.3, 0.6, 0.4, -0.2, 0.1];
        longitudes
            .into_iter()
            .zip(latitudes)
            .map(|(lo, la)| sphere_point(lo, la))
            .collect()
    }

    /// numga's tests' rotation, `-0.3 xy + 0.2 yz`.
    fn test_rotation() -> Bivector {
        Bivector::new(0.2, 0.0, -0.3, 0.0, 0.0, 0.0)
    }

    /// A direction's place on the unit sphere, as a vector of space.
    fn on_sphere(p: Direction) -> vga3d::Vector {
        let [x, y, z, _] = coordinates(directions(p));
        vga3d::Vector::new(x, y, z)
    }

    /// The five particles' places, stepped together.
    #[derive(Clone, Copy)]
    struct Swarm([vga3d::Vector; 5]);
    impl Add for Swarm {
        type Output = Swarm;
        fn add(self, o: Swarm) -> Swarm {
            Swarm(core::array::from_fn(|i| self.0[i] + o.0[i]))
        }
    }
    impl Mul<f64> for Swarm {
        type Output = Swarm;
        fn mul(self, h: f64) -> Swarm {
            Swarm(self.0.map(|v| v * h))
        }
    }

    /// numga's first test: the common rotor against each particle's tangential attraction
    /// integrated on its own (numga: `solve_ivp` to 1e-10; here 16 000 classical Runge-Kutta
    /// steps), at second order.
    #[test]
    fn common_rotor_matches_particle_equations_with_second_order_convergence() {
        let seeds = population();
        let (coupling, duration) = (1.1, 1.6);
        let omega = vga3d::Vector::new(-0.2, 0.0, 0.3);
        let velocity = |s: Swarm| {
            let mean = s.0.iter().fold(vga3d::Vector::zero(), |m, v| m + *v) * 0.2;
            Swarm(s.0.map(|x| {
                let turning = (omega ^ x).dual().with_orientation();
                let attraction = mean - x * (mean | x).s();
                turning + attraction * coupling
            }))
        };
        let mut reference = Swarm(core::array::from_fn(|i| on_sphere(seeds[i])));
        let fine = 16_000;
        for _ in 0..fine {
            reference = rk4(velocity, reference, duration / fine as f64);
        }
        let errors: Vec<f64> = [0.04, 0.02]
            .into_iter()
            .map(|dt| {
                let steps = (duration / dt).round() as usize;
                let rotors = motion(
                    spacetime::Even::new(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                    &seeds,
                    test_rotation(),
                    coupling,
                    dt,
                    steps,
                );
                let last = rotors[steps];
                seeds
                    .iter()
                    .zip(reference.0)
                    .map(|(s, r)| on_sphere(last >> *s).max_abs_diff(&r))
                    .fold(0.0, f64::max)
            })
            .collect();
        assert!(errors[1] < 0.0005, "{errors:?}");
        assert!(errors[1] < errors[0] / 3.0, "{errors:?}");
    }

    /// numga's second test: the common map keeps null rays null, their cross-ratios, the
    /// circles through a direction and the angle between them there.
    #[test]
    fn common_map_preserves_null_rays_cross_ratios_and_circle_angles() {
        let seeds = population();
        let generator = Bivector::new(0.18, 0.0, 0.0, 0.37, -0.21, 0.0);
        let rotor = generator.exp().into_inner();
        let mapped: Vec<Point> = seeds
            .iter()
            .map(|s| directions(rotor >> *s) + time())
            .collect();
        for m in &mapped {
            assert!((*m * *m).s().abs() < 1e-11);
        }
        let cross_ratio = |p: &[Point]| {
            let dot = |a: Point, b: Point| (a | b).s();
            dot(p[0], p[1]) * dot(p[2], p[3]) / (dot(p[0], p[2]) * dot(p[1], p[3]))
        };
        let (before, after) = (cross_ratio(&seeds), cross_ratio(&mapped));
        assert!(
            (before - after).abs() < 1e-11 * (1.0 + before.abs()),
            "{before} {after}"
        );

        // Two circles intersect at this direction; their tangent directions give their angle.
        #[allow(clippy::disallowed_methods)] // numga's direction, √0.84 x + 0.4 z
        let direction = (x() * 0.84f64.sqrt() + z() * 0.4).hodge() + time();
        let circles = [y(), z() + y() * (1.0 / 3.0) - t() * 0.4];
        let mapped_direction = directions(rotor >> direction) + time();
        let mapped_circles = circles.map(|c| rotor >> c);
        for c in mapped_circles.iter().chain(&circles) {
            let point = if circles.contains(c) {
                direction
            } else {
                mapped_direction
            };
            assert!((*c & point).s().abs() < 1e-11);
        }
        // The Euclidean angle between two circles at a point on both, from their cut planes'
        // normals.
        let angle = |point: Point, cuts: [Vector; 2]| {
            let position = coordinates(directions(point));
            let normals = cuts.map(|c| {
                let temporal = (c & time()).s();
                let [ax, ay, az] = space().map(|d| (c & d).s());
                vga3d::Vector::new(ax, ay, az)
                    + vga3d::Vector::new(position[0], position[1], position[2]) * temporal
            });
            (normals[0].normalized() | normals[1].normalized()).s()
        };
        let (before, after) = (
            angle(direction, circles),
            angle(mapped_direction, mapped_circles),
        );
        assert!((before - after).abs() < 1e-11, "{before} {after}");
    }

    /// numga's third test: the alignment grows step by step to above 0.98, and mirroring the
    /// population and the rotation mirrors the population at every step.
    #[test]
    fn alignment_grows_and_mirroring_inputs_mirrors_the_population() {
        let seeds = population();
        let (coupling, dt, steps) = (1.1, 0.025, 160);
        let identity = spacetime::Even::new(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        let rotors = motion(identity, &seeds, test_rotation(), coupling, dt, steps);
        let coherence: Vec<f64> = rotors
            .iter()
            .map(|r| {
                let m = coordinates(mean(seeds.iter().map(|s| directions(*r >> *s))));
                m[..3].iter().map(|c| c * c).sum()
            })
            .collect();
        assert!(coherence.windows(2).all(|w| w[1] - w[0] > -1e-8));
        assert!(coherence[steps] > 0.98, "{}", coherence[steps]);

        let mirror = (x() + y() * (1.0 / 3.0) - z() * 0.2).normalized();
        let mirrored_seeds: Vec<Point> = seeds.iter().map(|s| mirror >> *s).collect();
        let reflected = motion(
            identity,
            &mirrored_seeds,
            mirror >> test_rotation(),
            coupling,
            dt,
            steps,
        );
        for (r, m) in rotors.iter().zip(&reflected) {
            for (s, ms) in seeds.iter().zip(&mirrored_seeds) {
                let expected = mirror >> directions(*r >> *s);
                assert!(directions(*m >> *ms).max_abs_diff(&expected) < 1e-10);
            }
        }
    }

    /// numga's scenario checks: every direction stays a unit direction on the sphere, the
    /// population gathers to a twentieth of its spread; the grid's points lie on their cuts,
    /// and stay on them.
    #[test]
    fn the_population_gathers_and_the_grid_stays_on_its_circles() {
        let seeds = seeds();
        let rotors = synchronization();
        let spread = |r: &Rotor| {
            let population: Vec<Point> = seeds.iter().map(|s| directions(*r >> *s)).collect();
            let m = mean(population.iter().copied());
            population
                .iter()
                .map(|p| {
                    let q = *p + time();
                    assert!((q * q).s().abs() < 1e-10);
                    let d = *p - m;
                    (d * d).s().abs()
                })
                .sum::<f64>()
                / population.len() as f64
        };
        assert!(spread(&rotors[STEPS]) < 0.05 * spread(&rotors[0]));
        let grid = grid();
        for r in [rotors[0], rotors[STEPS]] {
            for (cut, points) in grid.iter().zip(grid_points(12)) {
                for p in points {
                    assert!(((r >> *cut) & (r >> p)).s().abs() < 1e-10);
                }
            }
        }
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws_with(super::draw, &super::scene(), 2.0);
    }
}
