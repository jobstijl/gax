//! The game, without any windowing: bodies are PGA2D motors moved by twists, shapes are
//! polygons of points, and hit tests are signs of `line ∧ point`. A shot cuts a rock along its
//! path into the two pieces of its outline, with masses and inertias from their shapes
//! (`gax::pga2d::Moments`).

use gax::Unit;
use gax::batch::BatchTransform;
use gax::pga2d::{Line, Moments, Motor, Point};
use rand::Rng as _;

/// The width and height of the world (it wraps around at the edges).
pub const WORLD: [f32; 2] = [160.0, 100.0];

const SHIP_TURN: f32 = 4.0; // rad/s
const SHIP_THRUST: f32 = 40.0; // units/s²
const SHIP_DRAG: f32 = 0.35; // fraction of speed lost per second
const BULLET_SPEED: f32 = 70.0;
const BULLET_LIFE: f32 = 1.1;
const FIRE_DELAY: f32 = 0.18;
const INVULNERABLE: f32 = 2.5;
/// Rock mass per unit area.
const DENSITY: f32 = 1.0;
/// A bullet's mass: its momentum knocks a rock before it breaks.
const BULLET_MASS: f32 = 6.0;
/// The momentum with which a rock's two pieces fly apart, along the cut's normal.
const SPLIT_MOMENTUM: f32 = 400.0;
/// Pieces of a smaller area turn to dust.
const MIN_PIECE_AREA: f32 = 8.0;

/// The origin of the plane.
const ORIGIN: Point = Point::new(0.0, 0.0, 1.0);

/// The ship's nose, in its body frame: forwards.
const FORWARDS: Point = Point::new(0.0, 1.0, 0.0);

/// The direction of `length` at `angle` from the x axis: `(length, 0)` turned by a rotation.
fn polar(length: f32, angle: f32) -> Point {
    Motor::rotation(ORIGIN, angle) >> Point::direction(length, 0.0)
}

/// Keys held (or pressed) this frame.
#[derive(Clone, Copy, Debug, Default)]
pub struct Input {
    /// Turn counterclockwise.
    pub left: bool,
    /// Turn clockwise.
    pub right: bool,
    /// Accelerate forwards.
    pub thrust: bool,
    /// Shoot.
    pub fire: bool,
}

/// A rigid body: a pose (a unit motor taking body coordinates to world coordinates), a world
/// velocity and a spin about the body's own origin.
#[derive(Clone, Copy, Debug)]
pub struct Body {
    /// Body to world.
    pub pose: Unit<Motor>,
    /// World velocity: a direction.
    pub vel: Point,
    /// Angular velocity, counterclockwise, rad/s.
    pub spin: f32,
}

impl Body {
    /// A body at `p`, turned by `angle`, at rest.
    pub fn at(p: Point, angle: f32) -> Body {
        Body {
            pose: Motor::between(ORIGIN, p) * Motor::rotation(ORIGIN, angle),
            vel: Point::direction(0.0, 0.0),
            spin: 0.0,
        }
    }

    /// Advance by `dt`: translate in the world frame, spin in the body frame, wrap around.
    pub fn step(&mut self, dt: f32) {
        // The twists (bivectors: in PGA2D, points) of the velocities; exp(t B) moves for time t.
        let travel = self.translation_twist().gp(dt).exp();
        let turn = Point::rotation_twist(ORIGIN, self.spin).gp(dt).exp();
        self.pose = travel * self.pose * turn;
        // Across an edge: carried to the other side, a world's width or height away.
        let p = self.position();
        let wrapped = wrap(p);
        if !near(p, wrapped, 1.0) {
            self.pose = Motor::between(p, wrapped) * self.pose;
        }
        // Keep the motor unit despite rounding (one Newton step, no square root).
        self.pose = self.pose.renormalize_fast();
    }

    /// The body's origin in the world.
    pub fn position(&self) -> Point {
        self.pose >> ORIGIN
    }

    /// The twist of a translation at `vel`.
    fn translation_twist(&self) -> Point {
        Point::translation_twist(self.vel.e20(), self.vel.e01())
    }

    /// The world twist of its motion: a translation at `vel` plus a turn at `spin` about its
    /// origin. Twists add.
    pub fn twist(&self) -> Point {
        self.translation_twist() + Point::rotation_twist(self.position(), self.spin)
    }

    /// The world velocity of the body's point at `p` (a unit point): the rate of `exp(t B) p
    /// exp(-t B)`, `B p - p B`, twice gax's commutator.
    pub fn velocity_at(&self, p: Point) -> Point {
        self.twist().commutator(p).gp(2.0)
    }

    /// A body direction (such as [`FORWARDS`]) in the world.
    pub fn direction(&self, d: Point) -> Point {
        self.pose >> d
    }
}

/// A convex polygon in body coordinates, counterclockwise.
#[derive(Clone, Debug)]
pub struct Shape {
    /// The vertices.
    pub points: Vec<Point>,
    /// The largest distance of a vertex from the origin.
    pub radius: f32,
    /// Its mass (area times density).
    pub mass: f32,
    /// Its moment of inertia about its centre of mass.
    pub inertia: f32,
}

impl Shape {
    fn new(points: Vec<Point>) -> Shape {
        let radius = points
            .iter()
            .map(|p| (*p & ORIGIN).norm())
            .fold(0.0, f32::max);
        let m = Moments::<f32>::of_polygon(&points);
        Shape {
            points,
            radius,
            mass: m.area() * DENSITY,
            inertia: m.polar_moment(DENSITY),
        }
    }

    /// The shape of `points` moved so that its centre of mass is the body origin (a rigid body
    /// spins about it), and that centre in the old coordinates.
    fn centred(points: &[Point]) -> (Shape, Point) {
        let c = Moments::<f32>::of_polygon(points).centroid();
        let back = Motor::between(c, ORIGIN);
        (Shape::new(points.iter().map(|&p| back >> p).collect()), c)
    }

    /// Its area.
    pub fn area(&self) -> f32 {
        self.mass / DENSITY
    }

    /// The vertices placed by `pose` (all at once, on SIMD lanes).
    pub fn placed(&self, pose: Unit<Motor>, out: &mut Vec<Point>) {
        out.resize(self.points.len(), Point::zero());
        pose.transform_slice(&self.points, out);
    }
}

/// The convex polygon `poly` cut by `line` into the parts on its two sides: each edge whose ends
/// lie on different sides contributes the meet of its line with `line` to both.
pub fn cut(poly: &[Point], line: Line) -> [Vec<Point>; 2] {
    let side = |p: Point| (line ^ p).e012() * p.e12().signum();
    let mut parts = [Vec::new(), Vec::new()];
    for (&a, &b) in poly.iter().zip(poly.iter().cycle().skip(1)) {
        let (sa, sb) = (side(a), side(b));
        parts[usize::from(sa < 0.0)].push(a);
        if (sa < 0.0) != (sb < 0.0) {
            let x = ((a & b) ^ line).unitized();
            parts[0].push(x);
            parts[1].push(x);
        }
    }
    parts
}

/// Whether `p` lies inside the convex polygon `poly` (world vertices, either orientation):
/// on the same side of every edge line as the polygon's `inside` point. The side of a point
/// with respect to a line is the sign of their outer product.
pub fn contains(poly: &[Point], inside: Point, p: Point) -> bool {
    poly.iter()
        .zip(poly.iter().cycle().skip(1))
        .all(|(&a, &b)| {
            let edge: Line = a & b;
            let (s, t) = ((edge ^ p).e012(), (edge ^ inside).e012());
            s * t >= 0.0
        })
}

/// The random generator, seeded.
pub type Rng = rand::rngs::StdRng;

/// An asteroid: a body, a shape, and a size class (3 large, 2 medium, 1 small).
#[derive(Clone, Debug)]
pub struct Asteroid {
    /// Where it is and how it moves.
    pub body: Body,
    /// Its outline.
    pub shape: Shape,
    /// Size class.
    pub size: u8,
}

impl Asteroid {
    fn random(rng: &mut Rng, at: Point, size: u8) -> Asteroid {
        let r = 3.5 * f32::from(size);
        let n = 7 + usize::from(size) * 2;
        let points: Vec<Point> = (0..n)
            .map(|i| {
                let a = std::f32::consts::TAU * (i as f32 + rng.random_range(-0.3..0.3)) / n as f32;
                let d = r * rng.random_range(0.75..1.1);
                Motor::rotation(ORIGIN, a) >> Point::xy(d, 0.0)
            })
            .collect();
        let mut body = Body::at(at, rng.random_range(0.0..std::f32::consts::TAU));
        let speed = rng.random_range(4.0..14.0) * (4.0 - f32::from(size)) / 2.0;
        let heading = rng.random_range(0.0..std::f32::consts::TAU);
        body.vel = polar(speed, heading);
        body.spin = rng.random_range(-1.2..1.2);
        Asteroid {
            body,
            shape: Shape::centred(&points).0,
            size,
        }
    }

    /// The pieces of this rock when a bullet at `hit` with velocity `bullet` (a direction)
    /// strikes it. The
    /// bullet's momentum first knocks the rock (through its mass and, about its centre of
    /// mass, its inertia); then the rock is cut along the bullet's path, and each piece keeps
    /// the rock's rigid motion at its own centre of mass, the two pushed apart with equal and
    /// opposite momentum. Small rocks and slivers turn to dust.
    pub fn shatter(&self, hit: Point, bullet: Point) -> Vec<Asteroid> {
        if self.size <= 1 {
            return Vec::new();
        }
        let (pose, shape) = (self.body.pose, &self.shape);
        // The knock: Δv = J / m, and Δω the moment of the impulse about the centre of mass over
        // the inertia. The moment is the join of the impulse's line, `hit & J`, with the centre.
        let j = bullet.gp(BULLET_MASS);
        let mut knocked = self.body;
        knocked.vel += j.gp(shape.mass.recip());
        knocked.spin += ((hit & j) & (pose >> ORIGIN)).s() / shape.inertia;
        // The cut, in body coordinates: the line through the hit along the bullet.
        let path = (pose << hit) & (pose << bullet);
        // The cut in the world, unit: the side a point lies on is the sign of its join with it,
        // positive towards the line's normal.
        let line = hit & bullet;
        let line = line.gp(line.norm().max(1e-6).recip());
        let normal = Point::direction(line.e1(), line.e2());
        cut(&shape.points, path)
            .iter()
            .filter_map(|part| {
                let (piece, centre) = Shape::centred(part);
                if piece.area() < MIN_PIECE_AREA {
                    return None;
                }
                let mut body = knocked;
                body.pose = pose * Motor::between(ORIGIN, centre);
                // The knocked rock's velocity at the piece's centre, plus the push apart, away
                // from the cut on the piece's side of it.
                let at = body.pose >> ORIGIN;
                let push = (line & at).s().signum() * SPLIT_MOMENTUM / piece.mass;
                body.vel = knocked.velocity_at(at) + normal.gp(push);
                let size = (self.size - 1).min(if piece.area() >= 60.0 { 2 } else { 1 });
                Some(Asteroid {
                    body,
                    shape: piece,
                    size,
                })
            })
            .collect()
    }
}

/// A shot.
#[derive(Clone, Copy, Debug)]
pub struct Bullet {
    /// Position.
    pub pos: Point,
    /// Velocity: a direction.
    pub vel: Point,
    /// Seconds left.
    pub life: f32,
}

/// A spark of an explosion.
#[derive(Clone, Copy, Debug)]
pub struct Spark {
    /// Position.
    pub pos: Point,
    /// Velocity: a direction.
    pub vel: Point,
    /// Seconds left.
    pub life: f32,
}

/// The game state.
#[derive(Clone, Debug)]
pub struct Game {
    /// The player's ship.
    pub ship: Body,
    /// The ship's outline.
    pub ship_shape: Shape,
    /// Rocks.
    pub asteroids: Vec<Asteroid>,
    /// Shots in flight.
    pub bullets: Vec<Bullet>,
    /// Explosion sparks.
    pub sparks: Vec<Spark>,
    /// Points.
    pub score: u32,
    /// Ships left.
    pub lives: u32,
    /// The wave number.
    pub wave: u32,
    /// Whether the ship is thrusting (for drawing the flame).
    pub thrusting: bool,
    /// Seconds of invulnerability left after a respawn.
    pub shield: f32,
    /// Whether the game is over.
    pub over: bool,
    cooldown: f32,
    rng: Rng,
    scratch: [Vec<Point>; 2],
}

impl Game {
    /// A new game.
    pub fn new(seed: u64) -> Game {
        let mut g = Game {
            ship: Body::at(ORIGIN, 0.0),
            ship_shape: Shape::new(vec![
                Point::xy(0.0, 3.0),
                Point::xy(-1.8, -2.0),
                Point::xy(0.0, -1.2),
                Point::xy(1.8, -2.0),
            ]),
            asteroids: Vec::new(),
            bullets: Vec::new(),
            sparks: Vec::new(),
            score: 0,
            lives: 3,
            wave: 0,
            thrusting: false,
            shield: INVULNERABLE,
            over: false,
            cooldown: 0.0,
            rng: <Rng as rand::SeedableRng>::seed_from_u64(seed),
            scratch: [Vec::new(), Vec::new()],
        };
        g.next_wave();
        g
    }

    fn next_wave(&mut self) {
        self.wave += 1;
        for _ in 0..(2 + self.wave).min(9) {
            // Spawn away from the ship.
            let at = loop {
                let x = self.rng.random_range(-WORLD[0] / 2.0..WORLD[0] / 2.0);
                let y = self.rng.random_range(-WORLD[1] / 2.0..WORLD[1] / 2.0);
                let p = Point::xy(x, y);
                if !near(p, self.ship.position(), 30.0) {
                    break p;
                }
            };
            let a = Asteroid::random(&mut self.rng, at, 3);
            self.asteroids.push(a);
        }
    }

    fn explode(&mut self, at: Point, n: usize, speed: f32) {
        for _ in 0..n {
            let a = self.rng.random_range(0.0..std::f32::consts::TAU);
            let s = self.rng.random_range(0.2..1.0) * speed;
            self.sparks.push(Spark {
                pos: at,
                vel: polar(s, a),
                life: self.rng.random_range(0.3..0.9),
            });
        }
    }

    /// Advance the game by `dt` seconds.
    pub fn update(&mut self, dt: f32, input: Input) {
        if self.over {
            return;
        }
        // The ship: turn, thrust along its nose, drag.
        let turn = f32::from(i8::from(input.left) - i8::from(input.right));
        self.ship.spin = turn * SHIP_TURN;
        self.thrusting = input.thrust;
        if input.thrust {
            self.ship.vel += self.ship.direction(FORWARDS).gp(SHIP_THRUST * dt);
        }
        let keep = (1.0 - SHIP_DRAG * dt).max(0.0);
        self.ship.vel = self.ship.vel.gp(keep);
        self.ship.step(dt);
        self.shield = (self.shield - dt).max(0.0);

        // Shots leave from the nose, adding the ship's velocity.
        self.cooldown -= dt;
        if input.fire && self.cooldown <= 0.0 {
            self.cooldown = FIRE_DELAY;
            let nose = self.ship.pose >> Point::xy(0.0, 3.0);
            self.bullets.push(Bullet {
                pos: nose,
                vel: self.ship.vel + self.ship.direction(FORWARDS).gp(BULLET_SPEED),
                life: BULLET_LIFE,
            });
        }
        for b in &mut self.bullets {
            b.pos = wrap(b.pos + b.vel.gp(dt));
            b.life -= dt;
        }
        self.bullets.retain(|b| b.life > 0.0);
        for s in &mut self.sparks {
            s.pos += s.vel.gp(dt);
            s.life -= dt;
        }
        self.sparks.retain(|s| s.life > 0.0);
        for a in &mut self.asteroids {
            a.body.step(dt);
        }

        self.hits();
        if self.asteroids.is_empty() {
            self.next_wave();
        }
    }

    /// Bullets against asteroids, then the ship against asteroids.
    fn hits(&mut self) {
        let mut split: Vec<(usize, Bullet)> = Vec::new();
        let [placed, ship] = &mut self.scratch;
        for (i, a) in self.asteroids.iter().enumerate() {
            let c = a.body.position();
            a.shape.placed(a.body.pose, placed);
            let hit = self
                .bullets
                .iter()
                .position(|b| near(c, b.pos, a.shape.radius) && contains(placed, c, b.pos));
            if let Some(j) = hit {
                let b = self.bullets.swap_remove(j);
                split.push((i, b));
            }
        }
        // The ship: a vertex of one polygon inside the other.
        let mut crashed = false;
        if self.shield <= 0.0 {
            self.ship_shape.placed(self.ship.pose, ship);
            let sc = self.ship.position();
            for a in &self.asteroids {
                let c = a.body.position();
                if !near(c, sc, a.shape.radius + self.ship_shape.radius) {
                    continue;
                }
                a.shape.placed(a.body.pose, placed);
                if ship.iter().any(|&p| contains(placed, c, p))
                    || placed.iter().any(|&p| contains(ship, sc, p))
                {
                    crashed = true;
                    break;
                }
            }
        }
        for &(i, b) in split.iter().rev() {
            let a = self.asteroids.swap_remove(i);
            self.score += [0, 100, 50, 20][usize::from(a.size)];
            self.explode(b.pos, 6 + 4 * usize::from(a.size), 25.0);
            self.asteroids.extend(a.shatter(b.pos, b.vel));
        }
        if crashed {
            let at = self.ship.position();
            self.explode(at, 30, 35.0);
            if self.lives > 1 {
                self.lives -= 1;
                self.ship = Body::at(ORIGIN, 0.0);
                self.shield = INVULNERABLE;
            } else {
                self.lives = 0;
                self.over = true;
            }
        }
    }
}

/// Whether the unit points `a` and `b` are within `r`: the norm of their join is their
/// distance.
fn near(a: Point, b: Point, r: f32) -> bool {
    (a & b).norm() <= r
}

/// `p` wrapped into the world (a torus): its coordinates taken modulo the world's size.
fn wrap(p: Point) -> Point {
    let [x, y] = p.to_euclidean();
    let w = |x: f32, s: f32| (x + s / 2.0).rem_euclid(s) - s / 2.0;
    Point::xy(w(x, WORLD[0]), w(y, WORLD[1]))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Points (or directions) with the same coefficients, to rounding.
    fn close(a: Point, b: Point) -> bool {
        a.c.iter().zip(b.c).all(|(x, y)| (x - y).abs() < 1e-4)
    }

    #[test]
    fn twists_move_as_intended() {
        // Translation twist: after 2 s at (3, -1) the origin is at (6, -2).
        let mut b = Body::at(ORIGIN, 0.0);
        b.vel = Point::direction(3.0, -1.0);
        b.step(2.0);
        assert!(
            close(b.position(), Point::xy(6.0, -2.0)),
            "{:?}",
            b.position()
        );
        // Rotation twist: a quarter turn counterclockwise takes the nose (0, 1) to (-1, 0).
        let mut b = Body::at(Point::xy(5.0, 5.0), 0.0);
        b.spin = std::f32::consts::FRAC_PI_2;
        b.step(1.0);
        let nose = b.direction(FORWARDS);
        assert!(close(nose, Point::direction(-1.0, 0.0)), "{nose:?}");
        assert!(close(b.position(), Point::xy(5.0, 5.0)));
    }

    #[test]
    fn bodies_wrap_around_the_world() {
        let mut b = Body::at(Point::xy(WORLD[0] / 2.0 - 1.0, 0.0), 0.0);
        b.vel = Point::direction(2.0, 0.0);
        b.step(1.0);
        assert!(
            close(b.position(), Point::xy(-WORLD[0] / 2.0 + 1.0, 0.0)),
            "{:?}",
            b.position()
        );
    }

    #[test]
    fn point_in_polygon_by_line_signs() {
        let square = [
            Point::xy(-1.0, -1.0),
            Point::xy(1.0, -1.0),
            Point::xy(1.0, 1.0),
            Point::xy(-1.0, 1.0),
        ];
        let c = Point::xy(0.0, 0.0);
        assert!(contains(&square, c, Point::xy(0.5, -0.9)));
        assert!(!contains(&square, c, Point::xy(1.5, 0.0)));
        // Either orientation.
        let rev: Vec<Point> = square.iter().rev().copied().collect();
        assert!(contains(&rev, c, Point::xy(-0.9, 0.9)));
        assert!(!contains(&rev, c, Point::xy(0.0, -1.1)));
    }

    #[test]
    fn placed_shapes_follow_the_pose() {
        let shape = Shape::new(vec![Point::xy(1.0, 0.0), Point::xy(0.0, 1.0)]);
        let mut out = Vec::new();
        let pose = Motor::translation(2.0, 3.0)
            * Motor::rotation(Point::xy(0.0, 0.0), std::f32::consts::FRAC_PI_2);
        shape.placed(pose, &mut out);
        assert!(close(out[0], Point::xy(2.0, 4.0)));
        assert!(close(out[1], Point::xy(1.0, 3.0)));
    }

    #[test]
    fn a_shot_splits_an_asteroid() {
        let mut g = Game::new(1);
        g.asteroids.truncate(1);
        g.asteroids[0].body = Body::at(Point::xy(0.0, 20.0), 0.0);
        let before = g.score;
        // Fire straight up until the rock breaks.
        for _ in 0..120 {
            g.update(
                1.0 / 60.0,
                Input {
                    fire: true,
                    ..Input::default()
                },
            );
            if g.score > before {
                break;
            }
        }
        assert!(g.score > before, "no hit");
        assert!(g.asteroids.iter().all(|a| a.size == 2));
    }

    /// A shot cuts a rock into its two pieces: their areas add up to the rock's, the momentum
    /// of rock and bullet is conserved, and the pieces move apart.
    #[test]
    fn shattering_conserves_area_and_momentum() {
        let mut rng = <Rng as rand::SeedableRng>::seed_from_u64(3);
        for k in 0..50 {
            let mut rock = Asteroid::random(&mut rng, Point::xy(5.0, -3.0), 3);
            rock.body.spin = 0.7;
            let c = rock.body.position();
            // A bullet through a point near the centre, from below.
            let (dx, dy) = (rng.random_range(-2.0..2.0), rng.random_range(-2.0..2.0));
            let hit = c + Point::direction(dx, dy);
            let bullet = Point::direction(rng.random_range(-20.0..20.0), 70.0);
            let pieces = rock.shatter(hit, bullet);
            assert_eq!(pieces.len(), 2, "rock {k}");
            let area: f32 = pieces.iter().map(|p| p.shape.area()).sum();
            assert!((area - rock.shape.area()).abs() < 1e-3 * rock.shape.area());
            // Momenta are directions; they add.
            let before = rock.body.vel.gp(rock.shape.mass) + bullet.gp(BULLET_MASS);
            let after = pieces.iter().fold(Point::direction(0.0, 0.0), |sum, p| {
                sum + p.body.vel.gp(p.shape.mass)
            });
            assert!(
                (after - before).ideal_norm() < 1e-3 * before.ideal_norm().max(100.0),
                "{after:?} vs {before:?}"
            );
            // Apart: a moment later the centres are farther from each other.
            let (a, b) = (pieces[0].body.position(), pieces[1].body.position());
            let (va, vb) = (pieces[0].body.vel, pieces[1].body.vel);
            let later = (a + va.gp(0.01)) & (b + vb.gp(0.01));
            assert!(
                later.norm() > (a & b).norm(),
                "rock {k}: the pieces close in"
            );
        }
    }

    #[test]
    fn a_long_game_runs() {
        let mut g = Game::new(7);
        let mut t = 0.0f32;
        for i in 0..20_000 {
            let input = Input {
                left: (i / 50) % 3 == 0,
                right: (i / 70) % 4 == 0,
                thrust: (i / 30) % 2 == 0,
                fire: i % 7 == 0,
            };
            g.update(1.0 / 60.0, input);
            t += 1.0 / 60.0;
            if g.over {
                break;
            }
        }
        assert!(t > 1.0);
    }

    #[test]
    fn an_off_centre_knock_spins_the_rock_the_right_way() {
        // A shot up along x = 2, right of the centre: r x J = 2 J > 0, counterclockwise.
        let mut rng = <Rng as rand::SeedableRng>::seed_from_u64(3);
        let rock = Asteroid::random(&mut rng, ORIGIN, 3);
        let c = rock.body.position();
        let pieces = rock.shatter(c + Point::direction(2.0, 0.0), Point::direction(0.0, 50.0));
        assert!(!pieces.is_empty());
        for p in &pieces {
            let want = rock.body.spin + 2.0 * BULLET_MASS * 50.0 / rock.shape.inertia;
            assert!(
                (p.body.spin - want).abs() < 1e-3 * want.abs(),
                "{} vs {want}",
                p.body.spin
            );
        }
    }
}
