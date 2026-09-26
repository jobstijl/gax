//! The game, without any windowing: bodies are PGA2D motors moved by twists, shapes are
//! polygons of points, and hit tests are signs of `line ∧ point`.

use gax::Unit;
use gax::batch::BatchTransform;
use gax::pga2d::{Line, Motor, Point};

/// The width and height of the world (it wraps around at the edges).
pub const WORLD: [f32; 2] = [160.0, 100.0];

const SHIP_TURN: f32 = 4.0; // rad/s
const SHIP_THRUST: f32 = 40.0; // units/s²
const SHIP_DRAG: f32 = 0.35; // fraction of speed lost per second
const BULLET_SPEED: f32 = 70.0;
const BULLET_LIFE: f32 = 1.1;
const FIRE_DELAY: f32 = 0.18;
const INVULNERABLE: f32 = 2.5;

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
    /// World velocity.
    pub vel: [f32; 2],
    /// Angular velocity, counterclockwise, rad/s.
    pub spin: f32,
}

/// The twist (a bivector: an ideal point) of a translation at velocity `v`: `exp(t B)` is
/// the translation by `t v`.
pub fn translation_twist(v: [f32; 2]) -> Point {
    Point::new(0.5 * v[1], -0.5 * v[0], 0.0)
}

/// The twist of a rotation about the origin at `w` rad/s counterclockwise.
pub fn rotation_twist(w: f32) -> Point {
    Point::new(0.0, 0.0, -0.5 * w)
}

impl Body {
    /// A body at `(x, y)`, turned by `angle`, at rest.
    pub fn at(x: f32, y: f32, angle: f32) -> Body {
        Body {
            pose: Motor::translation(x, y) * Motor::rotation(Point::xy(0.0, 0.0), angle),
            vel: [0.0, 0.0],
            spin: 0.0,
        }
    }

    /// Advance by `dt`: translate in the world frame, spin in the body frame, wrap around.
    pub fn step(&mut self, dt: f32) {
        let travel = translation_twist(self.vel).gp(dt).exp();
        let turn = rotation_twist(self.spin).gp(dt).exp();
        self.pose = travel * self.pose * turn;
        let [x, y] = self.position();
        let wrap = |p: f32, size: f32| {
            if p > size / 2.0 {
                -size
            } else if p < -size / 2.0 {
                size
            } else {
                0.0
            }
        };
        let (dx, dy) = (wrap(x, WORLD[0]), wrap(y, WORLD[1]));
        if dx != 0.0 || dy != 0.0 {
            self.pose = Motor::translation(dx, dy) * self.pose;
        }
        // Keep the motor unit despite rounding.
        self.pose = self.pose.into_inner().normalized();
    }

    /// The body's origin in the world.
    pub fn position(&self) -> [f32; 2] {
        (self.pose >> Point::xy(0.0, 0.0)).to_euclidean()
    }

    /// A body direction (such as forwards, `(0, 1)`) in the world.
    pub fn direction(&self, d: [f32; 2]) -> [f32; 2] {
        let p = self.pose >> Point::direction(d[0], d[1]);
        [p.e20(), p.e01()]
    }
}

/// A convex polygon in body coordinates, counterclockwise.
#[derive(Clone, Debug)]
pub struct Shape {
    /// The vertices.
    pub points: Vec<Point>,
    /// The largest distance of a vertex from the origin.
    pub radius: f32,
}

impl Shape {
    fn new(points: Vec<Point>) -> Shape {
        let radius = points
            .iter()
            .map(|p| {
                let [x, y] = p.to_euclidean();
                x.hypot(y)
            })
            .fold(0.0, f32::max);
        Shape { points, radius }
    }

    /// The vertices placed by `pose` (all at once, on SIMD lanes).
    pub fn placed(&self, pose: Unit<Motor>, out: &mut Vec<Point>) {
        out.resize(self.points.len(), Point::zero());
        pose.transform_slice(&self.points, out);
    }
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

/// A small xorshift generator (the game needs no crate for randomness).
#[derive(Clone, Debug)]
pub struct Rng(u64);

impl Rng {
    /// A generator from a seed.
    pub fn new(seed: u64) -> Rng {
        Rng(seed.max(1))
    }
    /// A number in `[0, 1)`.
    pub fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }
    /// A number in `[a, b)`.
    pub fn range(&mut self, a: f32, b: f32) -> f32 {
        a + (b - a) * self.next()
    }
}

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
    fn random(rng: &mut Rng, x: f32, y: f32, size: u8) -> Asteroid {
        let r = 3.5 * f32::from(size);
        let n = 7 + usize::from(size) * 2;
        let points = (0..n)
            .map(|i| {
                let a = std::f32::consts::TAU * (i as f32 + rng.range(-0.3, 0.3)) / n as f32;
                let d = r * rng.range(0.75, 1.1);
                Point::xy(d * a.cos(), d * a.sin())
            })
            .collect();
        let mut body = Body::at(x, y, rng.range(0.0, std::f32::consts::TAU));
        let speed = rng.range(4.0, 14.0) * (4.0 - f32::from(size)) / 2.0;
        let heading = rng.range(0.0, std::f32::consts::TAU);
        body.vel = [speed * heading.cos(), speed * heading.sin()];
        body.spin = rng.range(-1.2, 1.2);
        Asteroid {
            body,
            shape: Shape::new(points),
            size,
        }
    }
}

/// A shot.
#[derive(Clone, Copy, Debug)]
pub struct Bullet {
    /// Position.
    pub pos: [f32; 2],
    /// Velocity.
    pub vel: [f32; 2],
    /// Seconds left.
    pub life: f32,
}

/// A spark of an explosion.
#[derive(Clone, Copy, Debug)]
pub struct Spark {
    /// Position.
    pub pos: [f32; 2],
    /// Velocity.
    pub vel: [f32; 2],
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
            ship: Body::at(0.0, 0.0, 0.0),
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
            rng: Rng::new(seed),
            scratch: [Vec::new(), Vec::new()],
        };
        g.next_wave();
        g
    }

    fn next_wave(&mut self) {
        self.wave += 1;
        for _ in 0..(2 + self.wave).min(9) {
            // Spawn away from the ship.
            let (x, y) = loop {
                let x = self.rng.range(-WORLD[0] / 2.0, WORLD[0] / 2.0);
                let y = self.rng.range(-WORLD[1] / 2.0, WORLD[1] / 2.0);
                let [sx, sy] = self.ship.position();
                if (x - sx).hypot(y - sy) > 30.0 {
                    break (x, y);
                }
            };
            let a = Asteroid::random(&mut self.rng, x, y, 3);
            self.asteroids.push(a);
        }
    }

    fn explode(&mut self, at: [f32; 2], n: usize, speed: f32) {
        for _ in 0..n {
            let a = self.rng.range(0.0, std::f32::consts::TAU);
            let s = self.rng.range(0.2, 1.0) * speed;
            self.sparks.push(Spark {
                pos: at,
                vel: [s * a.cos(), s * a.sin()],
                life: self.rng.range(0.3, 0.9),
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
            let f = self.ship.direction([0.0, 1.0]);
            self.ship.vel[0] += f[0] * SHIP_THRUST * dt;
            self.ship.vel[1] += f[1] * SHIP_THRUST * dt;
        }
        let keep = (1.0 - SHIP_DRAG * dt).max(0.0);
        self.ship.vel = self.ship.vel.map(|v| v * keep);
        self.ship.step(dt);
        self.shield = (self.shield - dt).max(0.0);

        // Shots leave from the nose, adding the ship's velocity.
        self.cooldown -= dt;
        if input.fire && self.cooldown <= 0.0 {
            self.cooldown = FIRE_DELAY;
            let nose = (self.ship.pose >> Point::xy(0.0, 3.0)).to_euclidean();
            let f = self.ship.direction([0.0, 1.0]);
            self.bullets.push(Bullet {
                pos: nose,
                vel: [
                    self.ship.vel[0] + f[0] * BULLET_SPEED,
                    self.ship.vel[1] + f[1] * BULLET_SPEED,
                ],
                life: BULLET_LIFE,
            });
        }
        for b in &mut self.bullets {
            b.pos = wrap([b.pos[0] + b.vel[0] * dt, b.pos[1] + b.vel[1] * dt]);
            b.life -= dt;
        }
        self.bullets.retain(|b| b.life > 0.0);
        for s in &mut self.sparks {
            s.pos = [s.pos[0] + s.vel[0] * dt, s.pos[1] + s.vel[1] * dt];
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
        let mut split: Vec<usize> = Vec::new();
        let [placed, ship] = &mut self.scratch;
        for (i, a) in self.asteroids.iter().enumerate() {
            let c = a.body.position();
            let center = Point::xy(c[0], c[1]);
            a.shape.placed(a.body.pose, placed);
            let hit = self.bullets.iter().position(|b| {
                near(c, b.pos, a.shape.radius)
                    && contains(placed, center, Point::xy(b.pos[0], b.pos[1]))
            });
            if let Some(j) = hit {
                self.bullets.swap_remove(j);
                split.push(i);
            }
        }
        // The ship: a vertex of one polygon inside the other.
        let mut crashed = false;
        if self.shield <= 0.0 {
            self.ship_shape.placed(self.ship.pose, ship);
            let sc = self.ship.position();
            let ship_center = Point::xy(sc[0], sc[1]);
            for a in &self.asteroids {
                let c = a.body.position();
                if !near(c, sc, a.shape.radius + self.ship_shape.radius) {
                    continue;
                }
                a.shape.placed(a.body.pose, placed);
                let center = Point::xy(c[0], c[1]);
                if ship.iter().any(|&p| contains(placed, center, p))
                    || placed.iter().any(|&p| contains(ship, ship_center, p))
                {
                    crashed = true;
                    break;
                }
            }
        }
        for &i in split.iter().rev() {
            let a = self.asteroids.swap_remove(i);
            let [x, y] = a.body.position();
            self.score += [0, 100, 50, 20][usize::from(a.size)];
            self.explode([x, y], 6 + 4 * usize::from(a.size), 25.0);
            if a.size > 1 {
                for _ in 0..2 {
                    let child = Asteroid::random(&mut self.rng, x, y, a.size - 1);
                    self.asteroids.push(child);
                }
            }
        }
        if crashed {
            let at = self.ship.position();
            self.explode(at, 30, 35.0);
            if self.lives > 1 {
                self.lives -= 1;
                self.ship = Body::at(0.0, 0.0, 0.0);
                self.shield = INVULNERABLE;
            } else {
                self.lives = 0;
                self.over = true;
            }
        }
    }
}

fn near(a: [f32; 2], b: [f32; 2], r: f32) -> bool {
    (a[0] - b[0]).hypot(a[1] - b[1]) <= r
}

fn wrap(p: [f32; 2]) -> [f32; 2] {
    let w = |x: f32, s: f32| (x + s / 2.0).rem_euclid(s) - s / 2.0;
    [w(p[0], WORLD[0]), w(p[1], WORLD[1])]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: [f32; 2], b: [f32; 2]) -> bool {
        (a[0] - b[0]).abs() < 1e-4 && (a[1] - b[1]).abs() < 1e-4
    }

    #[test]
    fn twists_move_as_intended() {
        // Translation twist: after 2 s at (3, -1) the origin is at (6, -2).
        let mut b = Body::at(0.0, 0.0, 0.0);
        b.vel = [3.0, -1.0];
        b.step(2.0);
        assert!(close(b.position(), [6.0, -2.0]), "{:?}", b.position());
        // Rotation twist: a quarter turn counterclockwise takes the nose (0, 1) to (-1, 0).
        let mut b = Body::at(5.0, 5.0, 0.0);
        b.spin = std::f32::consts::FRAC_PI_2;
        b.step(1.0);
        assert!(
            close(b.direction([0.0, 1.0]), [-1.0, 0.0]),
            "{:?}",
            b.direction([0.0, 1.0])
        );
        assert!(close(b.position(), [5.0, 5.0]));
    }

    #[test]
    fn bodies_wrap_around_the_world() {
        let mut b = Body::at(WORLD[0] / 2.0 - 1.0, 0.0, 0.0);
        b.vel = [2.0, 0.0];
        b.step(1.0);
        assert!(
            close(b.position(), [-WORLD[0] / 2.0 + 1.0, 0.0]),
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
        assert!(close(out[0].to_euclidean(), [2.0, 4.0]));
        assert!(close(out[1].to_euclidean(), [1.0, 3.0]));
    }

    #[test]
    fn a_shot_splits_an_asteroid() {
        let mut g = Game::new(1);
        g.asteroids.truncate(1);
        g.asteroids[0].body = Body::at(0.0, 20.0, 0.0);
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
}
