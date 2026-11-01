//! numga's `core.py` and `scenarios.py` of the rattleback: the rolling dynamics with the contact
//! reaction eliminated by an adjugate, the Runge–Kutta–Munthe-Kaas step, and the body built of
//! quadrics, its mass read from them.

use gax::Unit;
use gax::pga3d::{Direction, Forque, Line, Motor, OriginPlane, Plane, Point, Spin, Twist};

use gax_numga_examples::shared::mechanics_lie::{Lie, rigid::G};

/// The support map, from the table's plane through the shell's centre to the direction of the
/// touching point (the shell's squared semi-axes along its axes): `Direction <- OriginPlane`.
pub type Shape = Direction<(OriginPlane,)>;
/// The motions that keep the touching point at rest, per spin about the centre of mass.
pub type Rolling = Twist<(Spin,)>;
/// The momentum per rate.
pub type Inertia = Forque<(Twist,)>;
/// A quadric: each point to its polar plane, negative inside, zero on the surface, where the
/// plane is the tangent plane.
pub type Quadric = Plane<(Point,)>;
/// A placement.
pub type M = Unit<Motor>;

/// The plane at infinity, and the origin, its complement: the centre of mass.
pub fn infinity() -> Plane {
    Plane::new(0.0, 0.0, 0.0, 1.0)
}
pub fn origin() -> Point {
    infinity().dual()
}

/// The planes across `x`, `y` and `z`.
pub fn mirrors() -> [Plane; 3] {
    [
        Plane::new(1.0, 0.0, 0.0, 0.0),
        Plane::new(0.0, 1.0, 0.0, 0.0),
        Plane::new(0.0, 0.0, 1.0, 0.0),
    ]
}

/// The direction of a step, from its coordinates.
pub fn direction(x: f64, y: f64, z: f64) -> Direction {
    Direction::new(x, y, z)
}

// --- math

/// What rolling needs of a placement: the table's plane in the body, the direction from the
/// shell's centre to the lowest point, the shell centre's height, and the motions that keep the
/// lowest point instantaneously at rest.
pub struct Contact {
    pub normal: OriginPlane,
    pub support: Direction,
    pub height: f64,
    pub rolling: Rolling,
}

/// The contact of a placement: the table's plane read in the body, the support map's direction
/// for it over the height (the pairing `normal & shape(normal)` under its square root); a spin
/// about the centre of mass carried by the translation to the touching point, `centre +
/// support` from the centre of mass, turns the body about the contact (numga's
/// `exp((centre + support) origin⁻¹ / 2)`, the translation along it).
pub fn contact_geometry(motor: M, shape: Shape, centre: Direction, up: Direction) -> Contact {
    let normal = (motor << Point::from(up)).undual().cast::<OriginPlane>();
    #[allow(clippy::disallowed_methods)] // the support function: the height is √(n · S n)
    let height = (normal & shape.of(normal)).s().sqrt();
    let support = shape.of(normal) * (-1.0 / height);
    let to_contact = Motor::translation(Point::from(centre + support));
    Contact {
        normal,
        support,
        height,
        rolling: to_contact >> Spin::slot(),
    }
}

/// The rolling rate and the angular acceleration of the body, with the contact's reaction
/// eliminated: the adjugate of the rolling map pulls forques back onto spins, and a reaction at
/// the resting contact does no work under any rolling motion, so it pulls back to nothing.
#[allow(clippy::too_many_arguments)]
pub fn dynamics(
    motor: M,
    spin: Spin,
    shape: Shape,
    centre: Direction,
    inertia: Inertia,
    mass: f64,
    gravity: Direction,
    up: Direction,
) -> (Twist, Spin) {
    let Contact {
        normal,
        support,
        height,
        rolling,
    } = contact_geometry(motor, shape, centre, up);
    let rate = rolling.of(spin);
    // The touching point changes as the body rocks; differentiate its support geometry. The
    // shell's centre is fixed in the body, so only the support moves.
    let normal_rate = spin.commutator(normal);
    let contact_rate = shape.of(normal_rate) * (-1.0 / height)
        - support * ((normal & shape.of(normal_rate)).s() / (height * height));
    let rolling_change = (Point::from(contact_rate) * origin().inverse()).commutator(spin);
    let force = (origin() & (motor << gravity)) * mass;
    // The adjugate pairs forces with allowed rolling motions, so the resting contact does no
    // work: `Moment <- Forque`, the angular momenta the forques pull back to.
    let pullback = rolling.adjugate();
    let effective = pullback.of(inertia.of(rolling));
    let momentum = inertia.of(rate);
    let unbalanced = force - momentum.commutator(rate) - inertia.of(rolling_change.cast::<Line>());
    let acceleration = effective.solve(pullback.of(unbalanced));
    (rate, acceleration)
}

/// The kinetic and gravitational energy of a rolling body.
#[allow(clippy::too_many_arguments)]
pub fn energy(
    motor: M,
    spin: Spin,
    shape: Shape,
    centre: Direction,
    inertia: Inertia,
    mass: f64,
    gravity: Direction,
    up: Direction,
) -> f64 {
    let rate = contact_geometry(motor, shape, centre, up).rolling.of(spin);
    let potential = -mass * (Point::from(gravity).undual() & (motor >> origin())).s();
    (rate & inertia.of(rate)).s() / 2.0 + potential
}

/// A fourth-order Runge–Kutta–Munthe-Kaas step in the body's motion algebra: the rate pulled
/// through `dexpinv` of the increment at each stage (the shared Lie steppers' polynomial).
pub fn step(
    motor: M,
    spin: Spin,
    dt: f64,
    motion: &impl Fn(M, Spin) -> (Twist, Spin),
) -> (M, Spin) {
    let derivative = |increment: Twist, angular: Spin| {
        let (rate, acceleration) = motion(motor * increment.exp(), angular);
        (G::dexpinv(increment).of(rate * -0.5), acceleration)
    };
    let (m1, s1) = derivative(Twist::zero(), spin);
    let (m2, s2) = derivative(m1 * (dt / 2.0), spin + s1 * (dt / 2.0));
    let (m3, s3) = derivative(m2 * (dt / 2.0), spin + s2 * (dt / 2.0));
    let (m4, s4) = derivative(m3 * dt, spin + s3 * dt);
    let increment = (m1 + m2 * 2.0 + m3 * 2.0 + m4) * (dt / 6.0);
    let spin = spin + (s1 + s2 * 2.0 + s3 * 2.0 + s4) * (dt / 6.0);
    (motor * increment.exp(), spin)
}

// --- the body

/// The rattleback as built, in the frame of its centre of mass, by intersections of quadrics: a
/// wooden solid inside its underside and its cuts and outside its balls, and the steel balls,
/// each filling the pocket its complement cuts into the wood.
#[derive(Clone)]
pub struct Body {
    pub underside: Quadric,
    pub deck: Quadric,
    pub chamfers: [Quadric; 2],
    pub balls: [Quadric; 2],
}

impl Body {
    /// The wooden solid's quadrics, intersected: inside the underside and the cuts, outside the
    /// balls.
    pub fn wood(&self) -> Vec<Quadric> {
        let mut wood = vec![self.underside, self.deck];
        wood.extend(self.chamfers);
        wood.extend(self.balls.map(|b| -b));
        wood
    }

    /// Every part placed by a motor.
    pub fn placed(&self, motor: M) -> Body {
        let place = |q: Quadric| motor >> q.of(motor << Point::slot());
        Body {
            underside: place(self.underside),
            deck: place(self.deck),
            chamfers: self.chamfers.map(place),
            balls: self.balls.map(place),
        }
    }
}

/// The points on a plane's negative side, as the quadric pairing them with the plane and with
/// the plane at infinity, symmetrized.
pub fn half_space(cut: Plane) -> Quadric {
    (cut * (infinity() & Point::slot()) + infinity() * (cut & Point::slot())) * 0.5
}

/// A solid ball about a point of unit weight: the planes across the axes through the centre,
/// each one at a radius from it, less the weight squared.
pub fn ball(centre: Point, radius: f64) -> Quadric {
    let across: Quadric = mirrors()
        .iter()
        .map(|m| {
            let through = *m - infinity() * (*m & centre).s();
            through * (through & Point::slot())
        })
        .sum();
    across * (1.0 / (radius * radius)) - infinity() * (infinity() & Point::slot())
}

/// The shell's semi-axes, the body's mass and gravity.
pub const SEMI_AXES: [f64; 3] = [5.0, 17.0 / 16.0, 1.0];
pub const MASS: f64 = 4.0;
pub const GRAVITY: f64 = 9.8;
/// The deck's height above the shell's centre; the chamfers' tip about the long axis and along
/// it (degrees), and a point they pass through.
pub const DECK_HEIGHT: f64 = 0.5;
pub const CHAMFER_ROLL: f64 = 35.0;
pub const CHAMFER_PITCH: f64 = 30.0;
pub const CHAMFER_THROUGH: [f64; 3] = [3.2, 0.0, 0.2];
/// The steel balls' radius, their place along the shell, and the share of their diameter
/// pressed into the wood; the densities.
pub const BALL_RADIUS: f64 = 0.4;
pub const BALL_ALONG: f64 = 3.0;
pub const BALL_SINK: f64 = 0.6;
pub const WOOD_DENSITY: f64 = 0.7;
pub const STEEL_DENSITY: f64 = 7.85;
/// The spacing of the grid the mass is read from.
pub const SAMPLE_SPACING: f64 = 0.15;
/// The two launches: tipped from rest, and spinning the unfavourable way with a little
/// pitching. Tilts in degrees.
pub const TILTS: [f64; 2] = [15.0, 0.0];
pub const SPINS: [f64; 2] = [0.0, -1.5];
pub const PITCH_RATES: [f64; 2] = [0.0, 0.1];
/// The time step and the steps, each step a frame drawn.
pub const TIME_STEP: f64 = 0.04;
pub const N_STEPS: usize = 250;

/// The turn in the plane of two planes by an angle, numga's `exp(a ^ b · (−angle / 2))`.
pub fn turn(a: Plane, b: Plane, angle: f64) -> M {
    ((a ^ b) * (-angle / 2.0)).exp()
}

/// numpy's `arange(start, stop, step)`, each value moved by half a step: the grid's samples.
fn samples_between(start: f64, stop: f64) -> Vec<f64> {
    let n = ((stop - start) / SAMPLE_SPACING).ceil() as usize;
    (0..n)
        .map(|k| start + k as f64 * SAMPLE_SPACING + SAMPLE_SPACING / 2.0)
        .collect()
}

/// The grid points inside the body, its centre of mass, and its inertia about that: each point
/// weighted by the density of the part it lies inside, inside a part where it pairs negatively
/// with each of its quadrics.
pub fn distribution(wood: &[Quadric], steel: &[Quadric]) -> (Vec<Point>, Point, Inertia) {
    let [a, b, c] = SEMI_AXES;
    let reach = b + 2.0 * BALL_RADIUS;
    let (xs, ys, zs) = (
        samples_between(-a, a),
        samples_between(-reach, reach),
        samples_between(-c, DECK_HEIGHT),
    );
    let form = |q: &Quadric, p: Point| (q.of(p) & p).s();
    let mut weighted = Vec::new();
    for x in &xs {
        for y in &ys {
            for z in &zs {
                let p = Point::xyz(*x, *y, *z);
                let in_wood = wood.iter().all(|q| form(q, p) <= 0.0);
                let in_steel = steel.iter().any(|q| form(q, p) <= 0.0);
                let density = WOOD_DENSITY * f64::from(u8::from(in_wood))
                    + STEEL_DENSITY * f64::from(u8::from(in_steel));
                if density > 0.0 {
                    weighted.push((p, density));
                }
            }
        }
    }
    let total: f64 = weighted.iter().map(|(_, d)| d).sum();
    let centre: Point = weighted.iter().map(|(p, d)| *p * (d / total)).sum();
    // Each sample joined with its own motion under a rate is its momentum; the inertia is their
    // weighted sum.
    let inertia: Inertia = weighted
        .iter()
        .map(|(p, d)| {
            let relative = origin() + (*p - centre);
            (relative & relative.commutator(Twist::slot())) * (d / total * MASS)
        })
        .sum();
    (
        weighted.into_iter().map(|(p, _)| p).collect(),
        centre,
        inertia,
    )
}

/// The rattleback: its shell's support map, the shell's centre seen from the centre of mass,
/// the inertia about the centre of mass, and the body as built and the points its mass is read
/// from, in the frame of its centre of mass.
pub struct Rattleback {
    pub shape: Shape,
    pub centre: Direction,
    pub inertia: Inertia,
    pub body: Body,
    pub samples: Vec<Point>,
}

/// The body: a wooden boat with an ellipsoidal underside, a flat deck and two cuts across
/// opposite corners of the top, and a steel ball pressed into each side, the second of every
/// cut and ball half a turn about the vertical from the first.
pub fn rattleback() -> Rattleback {
    let [x, y, z] = mirrors();
    let w = infinity();
    let directions = mirrors().map(|m| m.dual().cast::<Direction>());
    let outline: Direction<(Direction,)> = directions
        .iter()
        .zip(mirrors())
        .zip(SEMI_AXES)
        .map(|((d, m), a)| (*d * a) * (m & Direction::slot()))
        .sum();
    let shape: Shape = directions
        .iter()
        .zip(SEMI_AXES)
        .map(|(d, a)| (*d * (a * a)) * (OriginPlane::slot() & *d))
        .sum();
    // The unit ball's planes across its axes carried onto the shell: each pairs with a
    // stretched direction as the ball's plane does with the direction, so the stretch's
    // adjugate carries it back.
    let carry_back = outline.adjugate();
    let underside: Quadric = mirrors()
        .iter()
        .map(|m| {
            let across = carry_back.solve(m.cast::<OriginPlane>()).cast::<Plane>();
            across * (across & Point::slot())
        })
        .sum::<Quadric>()
        - w * (w & Point::slot());
    let deck = half_space(z - w * DECK_HEIGHT);
    // Every cut and ball has its partner half a turn about the vertical.
    let sides = [0.0, core::f64::consts::PI].map(|a| turn(x, y, -a));
    let tipped =
        (turn(y, z, CHAMFER_ROLL.to_radians()) * turn(x, z, -CHAMFER_PITCH.to_radians())) >> z;
    let [cx, cy, cz] = CHAMFER_THROUGH;
    let corner = origin() + direction(cx, cy, cz);
    let chamfers = sides.map(|s| half_space(s >> (tipped - w * (tipped & corner).s())));
    // A ball on the shell's side, its diameter pressed in by the sink.
    #[allow(clippy::disallowed_methods)] // the ellipse's half width at the ball's place
    let side = SEMI_AXES[1] * (1.0 - (BALL_ALONG / SEMI_AXES[0]).powi(2)).sqrt();
    let seat = origin()
        + direction(
            BALL_ALONG,
            side + BALL_RADIUS * (1.0 - 2.0 * BALL_SINK),
            0.0,
        );
    let balls = sides.map(|s| ball(s >> seat, BALL_RADIUS));
    let built = Body {
        underside,
        deck,
        chamfers,
        balls,
    };
    let (samples, centre, inertia) = distribution(&built.wood(), &balls);
    // Everything moved so that the centre of mass is the origin.
    let shift = Motor::translation(origin() - centre);
    Rattleback {
        shape,
        centre: (origin() - centre).cast::<Direction>(),
        inertia,
        body: built.placed(shift),
        samples: samples.iter().map(|p| shift >> *p).collect(),
    }
}

/// The direction up, the complement of the plane across `z`, and gravity.
pub fn up() -> Direction {
    mirrors()[2].dual().cast::<Direction>()
}
pub fn gravity() -> Direction {
    up() * -GRAVITY
}

/// One launch over the steps: its placements, body-frame spins and energies.
pub struct Launch {
    pub motors: Vec<M>,
    pub spins: Vec<Spin>,
    pub energies: Vec<f64>,
}

impl Launch {
    /// The spin about the vertical in the world at each step.
    pub fn spin_about_vertical(&self) -> Vec<f64> {
        self.motors
            .iter()
            .zip(&self.spins)
            .map(|(m, s)| (*m >> s.cast::<Line>()).e12())
            .collect()
    }
}

/// Both launches from their start over the steps, each lifted by its centre of mass's height so
/// that the shell rests on the table: the shell's centre's height, less how far that centre
/// sits above the centre of mass.
pub fn launches(r: &Rattleback, dt: f64, n_steps: usize) -> [Launch; 2] {
    let [x, y, z] = mirrors();
    let motion =
        |m: M, s: Spin| dynamics(m, s, r.shape, r.centre, r.inertia, MASS, gravity(), up());
    core::array::from_fn(|k| {
        let spin = ((x ^ y) * SPINS[k] - (x ^ z) * PITCH_RATES[k]).cast::<Spin>();
        let tilt = turn(x, z, TILTS[k].to_radians());
        let normal = (tilt << Point::from(up())).undual().cast::<OriginPlane>();
        #[allow(clippy::disallowed_methods)] // the support function's height
        let height = (normal & r.shape.of(normal)).s().sqrt() - (normal & r.centre).s();
        let mut state = (Motor::translation(Point::from(up() * height)) * tilt, spin);
        let mut launch = Launch {
            motors: Vec::new(),
            spins: Vec::new(),
            energies: Vec::new(),
        };
        for _ in 0..=n_steps {
            let (m, s) = state;
            launch.motors.push(m);
            launch.spins.push(s);
            launch.energies.push(energy(
                m,
                s,
                r.shape,
                r.centre,
                r.inertia,
                MASS,
                gravity(),
                up(),
            ));
            state = step(m, s, dt, &motion);
        }
        launch
    })
}
