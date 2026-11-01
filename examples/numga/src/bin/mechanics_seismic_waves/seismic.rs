//! The math: the plane-strain material as a work form on displacement gradients, its pullback
//! through the corner hats to the vertices, the lumped masses, velocity Verlet, the dilation and
//! rotation of the motion, and the normal modes. No drawing.

use gax::field::{Field, SparseMap};
use gax::vga2d;
use gax_numga_examples::mesh::Mesh;

/// A scalar (numga's `Scalar+`).
pub type Scalar<S = ()> = vga2d::Scalar<S>;
/// A point or a displacement of the plane, the Hodge dual of its vector (numga's
/// `Antivector-`).
pub type Point<S = ()> = vga2d::intrinsic::Vector<S>;
/// A gradient, a face's normal, a force: a vector, read by its pairing with a point (numga's
/// `Vector+`).
pub type Plane<S = ()> = vga2d::Vector<S>;
/// A force at a vertex (numga's `Force = Vector+`).
pub type Force<S = ()> = vga2d::Vector<S>;
/// A rotation rate (numga's `Bivector+`).
pub type Bivector<S = ()> = vga2d::Bivector<S>;
/// A work form on pairs of displacement gradients: a face's normal, the gradient the
/// displacement varies along, the displacement, and the test displacement the traction does
/// work on.
pub type Elastic = Scalar<(Plane, Plane, Point, Point)>;
/// A traction: `Plane <- (normal, gradient, displacement)`.
pub type Traction = Plane<(Plane, Plane, Point)>;
/// A vertex's force from a vertex's displacement.
pub type Stiffness = Force<(Point,)>;
/// A map of displacements: an acceleration from a displacement.
pub type Motion = Point<(Point,)>;

/// The point at `(x, y)`: the Hodge dual of its vector (numga's `x * mv.x.dual() + ...`).
pub fn point(x: f64, y: f64) -> Point {
    Plane::new(x, y).hodge()
}

/// A point's coordinates, read by pairing it with the unit vectors along the axes.
pub fn coordinates(p: Point) -> [f64; 2] {
    [
        (Plane::new(1.0, 0.0) & p).s(),
        (Plane::new(0.0, 1.0) & p).s(),
    ]
}

/// The swelling traction: the face's normal times the divergence, the gradient paired with
/// the displacement.
pub fn dilation_traction() -> Traction {
    Plane::slot() * (Plane::slot() & Point::slot())
}

/// The shearing traction: twice the direct shear, less the exterior-product term, which
/// removes rotation from the gradient and leaves the two symmetric shear readings.
pub fn shear_traction() -> Traction {
    let along = || Point::slot().undual();
    let direct: Traction = (Plane::slot() | Plane::slot()) * along();
    direct * 2.0 - (Plane::slot() | (Plane::slot() ^ along()))
}

/// A plane-strain isotropic elastic work form on pairs of displacement gradients.
pub fn constitutive(lame_lambda: f64, lame_mu: f64) -> Elastic {
    let traction = dilation_traction() * lame_lambda + shear_traction() * lame_mu;
    traction & Point::slot()
}

/// A shear viscosity as a work form on pairs of velocity gradients: the elastic form's shear
/// part less two thirds of its dilation, the stress of a liquid with no bulk viscosity in a
/// plane-strain slice of three-dimensional space, so that it resists the rate of shear and not
/// the rate of swelling.
pub fn viscous(viscosity: f64) -> Elastic {
    let traction = (shear_traction() - dilation_traction() * (2.0 / 3.0)) * viscosity;
    traction & Point::slot()
}

/// The planet's assembled elastic and viscous response and inverse inertia, shared by every
/// time step.
pub struct Operators {
    pub mesh: Mesh<Point>,
    /// Each face's area.
    pub areas: Field<Scalar>,
    pub stiffness: SparseMap<Stiffness>,
    pub damping: SparseMap<Stiffness>,
    /// Each vertex's lumped mass.
    pub masses: Field<Scalar>,
    pub acceleration: SparseMap<Motion>,
    pub drag: SparseMap<Motion>,
    /// The rate of dilation and of rotation of each face, read directly from the corner
    /// velocities.
    pub divergence: SparseMap<Scalar<(Point,)>>,
    pub rotation: SparseMap<Bivector<(Point,)>>,
}

impl Operators {
    /// The operators of a mesh whose faces carry a material, a viscosity and a density.
    pub fn new(
        mesh: Mesh<Point>,
        material: &[Elastic],
        viscosity: &[Elastic],
        density: &[f64],
    ) -> Operators {
        let corners = mesh.corners();
        // The side facing each corner, counter-clockwise: from the corner after it to the one
        // after that; a face's patch is half the join of two of them.
        let sides: [Field<Point>; 3] =
            [0, 1, 2].map(|k| &corners[(k + 2) % 3] - &corners[(k + 1) % 3]);
        let patches = sides[0].zip_with(&sides[1], |a, b| (a & b) * 0.5);
        let areas = patches.map(|p| Scalar::new(p.norm()));
        // Each side over twice the patch: the gradient that falls by one from its corner to it.
        let corner_gradients = sides
            .each_ref()
            .map(|s| s.zip_with(&patches, |side, patch| side * (patch * 2.0).inverse()));
        let mut operators = Operators {
            stiffness: assemble(&mesh, &corner_gradients, &areas, material),
            damping: assemble(&mesh, &corner_gradients, &areas, viscosity),
            masses: Field::zeros(0),
            acceleration: SparseMap::new(vec![], vec![], vec![], (0, 0)),
            drag: SparseMap::new(vec![], vec![], vec![], (0, 0)),
            divergence: mesh.at_corners(
                &corner_gradients
                    .each_ref()
                    .map(|g| g.map(|g| g & Point::slot())),
            ),
            rotation: mesh.at_corners(
                &corner_gradients
                    .each_ref()
                    .map(|g| g.map(|g| g ^ Point::slot().undual())),
            ),
            mesh,
            areas,
        };
        // Each affine corner hat owns one third of its triangle's mass.
        let n = operators.mesh.vertices.len();
        let mut masses = vec![0.0; n];
        for (f, face) in operators.mesh.faces.iter().enumerate() {
            for &v in face {
                masses[v] += density[f] * operators.areas.site(f).s() / 3.0;
            }
        }
        operators.masses = masses.into_iter().map(Scalar::new).collect();
        let inverse_inertia =
            SparseMap::from_diagonal(&operators.masses.map(|m| to_point() * (1.0 / m.s())));
        operators.acceleration = -inverse_inertia.compose(&operators.stiffness);
        operators.drag = -inverse_inertia.compose(&operators.damping);
        operators
    }

    /// The vertices.
    pub fn vertices(&self) -> &Field<Point> {
        &self.mesh.vertices
    }

    /// The deformed planet's rate of dilation and signed vorticity, from its velocity.
    pub fn reconstruct(&self, velocity: &Field<Point>) -> (Field<Scalar>, Field<Scalar>) {
        (self.divergence.of(velocity), self.vorticity(velocity))
    }

    /// The signed rotation each face makes: its rotation rate read against the unit plane.
    pub fn vorticity(&self, velocity: &Field<Point>) -> Field<Scalar> {
        let unit = Bivector::new(1.0).hodge();
        self.rotation.of(velocity).map(|b| b & unit)
    }

    /// Kinetic plus elastic energy of the discrete field.
    pub fn energy(&self, displacement: &Field<Point>, velocity: &Field<Point>) -> f64 {
        let kinetic: f64 = self
            .masses
            .iter()
            .zip(velocity.iter())
            .map(|(m, v)| m.s() * (v.undual() & v).s())
            .sum();
        let elastic: f64 = self
            .stiffness
            .of(displacement)
            .iter()
            .zip(displacement.iter())
            .map(|(f, d)| (f & d).s())
            .sum();
        (kinetic + elastic) / 2.0
    }

    /// The momentum: the masses times the velocities, summed.
    #[cfg(test)]
    pub fn momentum(&self, velocity: &Field<Point>) -> Point {
        self.masses
            .iter()
            .zip(velocity.iter())
            .map(|(m, v)| v * m.s())
            .sum()
    }

    /// The `count` lowest normal modes: their angular frequencies squared and their shapes,
    /// orthonormal in the masses. The stiffness, a map from displacements to forces, becomes a
    /// map of displacements to themselves through the dual that takes a force to a point, and so
    /// meets the masses on the same footing.
    pub fn normal_modes(&self, count: usize) -> (Vec<f64>, Vec<Field<Point>>) {
        let n = self.masses.len();
        let to_points = SparseMap::from_diagonal(&Field::from_fn(n, |_| to_point()));
        let masses = SparseMap::from_diagonal(&self.masses.map(|m| Point::slot() * m.s()));
        to_points.compose(&self.stiffness).eigh(&masses, count)
    }
}

/// The dual that takes a force to the point along it.
pub fn to_point() -> Point<(Force,)> {
    Force::slot().hodge()
}

/// A work form on pairs of gradients, bound to each triangle's test and trial corner gradients:
/// a form on the corner motions, summed over the triangles. Each block, a map from a trial
/// corner's displacement to the test corner's force, is the pairing solve of the bound form.
fn assemble(
    mesh: &Mesh<Point>,
    gradients: &[Field<Plane>; 3],
    areas: &Field<Scalar>,
    material: &[Elastic],
) -> SparseMap<Stiffness> {
    let pairing = Force::slot() & Point::slot();
    let mut triples = Vec::with_capacity(9 * mesh.faces.len());
    for (f, face) in mesh.faces.iter().enumerate() {
        let area = areas.site(f).s();
        for (a, &test) in face.iter().enumerate() {
            let bound = material[f].of(gradients[a].site(f));
            for (b, &trial) in face.iter().enumerate() {
                let work = bound.of(gradients[b].site(f));
                let block: Stiffness = pairing.solve(work) * area;
                triples.push((test, trial, block));
            }
        }
    }
    let n = mesh.vertices.len();
    SparseMap::from_triples(triples, (n, n))
}

/// Velocity Verlet advances the free elastic field, giving displacement and velocity at the
/// start and after each frame of `substeps` steps.
///
/// The material is viscoelastic: it responds to its strain plus the damping time times its rate
/// of strain, so that a mode of angular frequency `ω` loses amplitude at the rate
/// `damping_time ω² / 2`; the liquid's viscosity adds a drag on its rate of shear. The rate of
/// strain is read from the half-step velocity, which keeps the step explicit.
pub fn evolve(
    operators: &Operators,
    displacement: Field<Point>,
    velocity: Field<Point>,
    damping_time: f64,
    dt: f64,
    frames: usize,
    substeps: usize,
) -> Vec<(Field<Point>, Field<Point>)> {
    let accelerate = |d: &Field<Point>, v: &Field<Point>| {
        &operators.acceleration.of(&(d + &v.scaled(damping_time))) + &operators.drag.of(v)
    };
    let (mut displacement, mut velocity) = (displacement, velocity);
    let mut acceleration = accelerate(&displacement, &velocity);
    let mut states = Vec::with_capacity(frames + 1);
    states.push((displacement.clone(), velocity.clone()));
    for _ in 0..frames {
        for _ in 0..substeps {
            velocity = &velocity + &acceleration.scaled(dt / 2.0);
            displacement = &displacement + &velocity.scaled(dt);
            acceleration = accelerate(&displacement, &velocity);
            velocity = &velocity + &acceleration.scaled(dt / 2.0);
        }
        states.push((displacement.clone(), velocity.clone()));
    }
    states
}

// --- scenes

pub const CORE_RADIUS: f64 = 0.55;
pub const SURFACE_RADIUS: f64 = 1.0;
pub const RADIAL_DIVISIONS: usize = 64;
/// The liquid core and the solid shell: the liquid stores compression energy and has no shear
/// stiffness; its shear viscosity, while the shell's damping is its viscoelasticity alone.
pub const LAYER_DENSITIES: [f64; 2] = [5.0, 1.0];
pub const LAYER_LAMBDAS: [f64; 2] = [5.0, 1.0];
pub const LAYER_MUS: [f64; 2] = [0.0, 1.0];
pub const LAYER_VISCOSITIES: [f64; 2] = [0.001, 0.0];
/// The place of the blow, on the diagonal towards the upper left, where the moon comes in from
/// the frame's corner: its angle from straight up, depth, width, support, speed.
pub const IMPACT_ANGLE: f64 = core::f64::consts::FRAC_PI_4;
pub const SOURCE_DEPTH: f64 = 0.05;
pub const SOURCE_WIDTH: f64 = 0.07;
pub const SOURCE_SUPPORT: f64 = 3.0;
pub const SOURCE_SPEED: f64 = 0.05;
pub const DAMPING_TIME: f64 = 3e-4;
/// The moon: its radius, its speed, the angle of its approach from the surface's normal, the
/// shares of its normal speed it rebounds with and of its tangential speed it keeps, and the
/// frames it takes to come in.
pub const MOON_RADIUS: f64 = 0.07;
pub const MOON_SPEED: f64 = 2.0;
pub const APPROACH_ANGLE: f64 = 40.0 * core::f64::consts::PI / 180.0;
pub const RESTITUTION: f64 = 0.5;
pub const SLIDING: f64 = 0.6;
pub const APPROACH_FRAMES: usize = 12;
pub const DT: f64 = 0.00448;
pub const FRAMES: usize = 100;
pub const SUBSTEPS: usize = 8;
/// The normal modes, of a uniform ball of cork, which does not spread sideways when squeezed,
/// so that its P waves run only the square root of two faster than its S waves: its Lamé
/// parameters and density, its rigid motions, the modes found, how close two frequencies are
/// when their modes differ by a turn of the ball, the travelling waves such pairs make that
/// are shown, the frames of each one's cycle, and the largest displacement they are shown at.
pub const BALL_LAMBDA: f64 = 0.0;
pub const BALL_MU: f64 = 1.0;
pub const BALL_DENSITY: f64 = 1.0;
pub const RIGID_MOTIONS: usize = 3;
pub const MODES: usize = 40;
pub const PAIR_TOLERANCE: f64 = 1e-3;
pub const TRAVELLING_WAVES: usize = 12;
pub const CYCLE_FRAMES: usize = 24;
pub const MODE_PEAK: f64 = 0.0006;

/// The concentric disk of `boundaries` as a mesh of points of the plane.
pub fn disk(boundaries: &[f64], divisions: usize) -> Mesh<Point> {
    let mesh = Mesh::concentric_disk(boundaries, divisions);
    let points = mesh.vertices.map(|v| point(v.e1(), v.e2()));
    mesh.with_vertices(points)
}

/// Each face's centre: the mean of its corners.
pub fn centres(mesh: &Mesh<Point>) -> Field<Point> {
    let [a, b, c] = mesh.corners();
    (&(&a + &b) + &c).scaled(1.0 / 3.0)
}

/// The turn by `angle`, counter-clockwise.
pub fn turned(p: Point, angle: f64) -> Point {
    (Bivector::new(1.0) * (-angle / 2.0)).exp() >> p
}

/// The planet: its mesh, each face's layer (zero for the liquid core, one for the solid
/// shell), and its operators.
pub struct Planet {
    pub layers: Vec<usize>,
    pub operators: Operators,
}

pub fn planet(divisions: usize) -> Planet {
    let mesh = disk(&[CORE_RADIUS, SURFACE_RADIUS], divisions);
    let layers: Vec<usize> = centres(&mesh)
        .iter()
        .map(|c| usize::from(c.norm() > CORE_RADIUS))
        .collect();
    let material: Vec<Elastic> = layers
        .iter()
        .map(|&l| constitutive(LAYER_LAMBDAS[l], LAYER_MUS[l]))
        .collect();
    let viscosity: Vec<Elastic> = layers
        .iter()
        .map(|&l| viscous(LAYER_VISCOSITIES[l]))
        .collect();
    let density: Vec<f64> = layers.iter().map(|&l| LAYER_DENSITIES[l]).collect();
    Planet {
        operators: Operators::new(mesh, &material, &viscosity, &density),
        layers,
    }
}

/// The blow: the moon comes in at an angle to the normal of the surface where it strikes, and
/// rebounds with part of its normal speed reversed and part of its tangential speed kept; the
/// planet's blow is the momentum the moon gives up. The moon's place at each frame, from the
/// approach, and the direction of the blow.
pub fn moon() -> (Vec<Point>, Point) {
    let normal = impact_normal();
    let incoming = (Bivector::new(1.0) * (APPROACH_ANGLE / 2.0)).exp() >> -normal;
    let incoming = incoming * MOON_SPEED;
    let normal_part = normal * (incoming | normal).s();
    let outgoing = (incoming - normal_part) * SLIDING - normal_part * RESTITUTION;
    let blow = (incoming - outgoing).normalized();
    let contact = normal * (SURFACE_RADIUS + MOON_RADIUS);
    let frame = SUBSTEPS as f64 * DT;
    let places = (0..=APPROACH_FRAMES + FRAMES)
        .map(|k| {
            let time = (k as f64 - APPROACH_FRAMES as f64) * frame;
            contact + if time < 0.0 { incoming } else { outgoing } * time
        })
        .collect();
    (places, blow)
}

/// Straight up, turned towards the upper left.
pub fn impact_normal() -> Point {
    turned(point(0.0, 1.0), IMPACT_ANGLE)
}

/// A short blow just below the free surface: the velocity of a tapered Gaussian about the
/// source along the blow.
pub fn blow_velocity(vertices: &Field<Point>, blow: Point) -> Field<Point> {
    let centre = impact_normal() * (SURFACE_RADIUS - SOURCE_DEPTH);
    vertices.map(|v| {
        let r2 = (v - centre).norm_squared() / (SOURCE_WIDTH * SOURCE_WIDTH);
        let support = SOURCE_SUPPORT * SOURCE_SUPPORT;
        let envelope = if r2 < support {
            let taper = 1.0 - r2 / support;
            (-r2 / 2.0).exp() * taper * taper
        } else {
            0.0
        };
        blow * (SOURCE_SPEED * envelope)
    })
}

/// The planet's frames: at rest while the moon comes in, then ringing after the blow; the
/// displacement, the rate of dilation and the vorticity at each.
pub struct Ringing {
    pub frames: Vec<(Field<Point>, Field<Scalar>, Field<Scalar>)>,
    pub moon: Vec<Point>,
}

pub fn ringing(planet: &Planet) -> Ringing {
    let ops = &planet.operators;
    let (moon, blow) = moon();
    let velocity = blow_velocity(ops.vertices(), blow);
    let rest = Field::<Point>::zeros(velocity.len());
    let resting = (0..APPROACH_FRAMES).map(|_| (rest.clone(), rest.clone()));
    let states = evolve(
        ops,
        rest.clone(),
        velocity,
        DAMPING_TIME,
        DT,
        FRAMES,
        SUBSTEPS,
    );
    let frames = resting
        .chain(states)
        .map(|(d, v)| {
            let (dilation, vorticity) = ops.reconstruct(&v);
            (d, dilation, vorticity)
        })
        .collect();
    Ringing { frames, moon }
}

/// The uniform ball of cork.
pub fn ball(divisions: usize) -> Operators {
    let mesh = disk(&[CORE_RADIUS, SURFACE_RADIUS], divisions);
    let faces = mesh.faces.len();
    Operators::new(
        mesh,
        &vec![constitutive(BALL_LAMBDA, BALL_MU); faces],
        &vec![viscous(0.0); faces],
        &vec![BALL_DENSITY; faces],
    )
}

/// The pairs among the modes past the rigid motions: the first of each two whose frequencies
/// agree, two modes that differ by a turn of the ball, the first `TRAVELLING_WAVES`.
pub fn paired(squared_frequencies: &[f64]) -> Vec<usize> {
    (0..squared_frequencies.len() - 1)
        .filter(|&i| {
            let (a, b) = (squared_frequencies[i], squared_frequencies[i + 1]);
            b - a < PAIR_TOLERANCE * b
        })
        .take(TRAVELLING_WAVES)
        .collect()
}

/// A pair's travelling wave, the first mode a quarter cycle ahead of the second: both scaled so
/// that the first's largest displacement is `MODE_PEAK`.
pub fn travelling_pair(first: &Field<Point>, second: &Field<Point>) -> [Field<Point>; 2] {
    let largest = first.iter().map(|p| p.norm()).fold(0.0, f64::max);
    let scale = MODE_PEAK / largest;
    [first.scaled(scale), second.scaled(scale)]
}

/// The displacement of a travelling wave at a phase of its cycle, in which it travels one
/// wavelength round the ball.
pub fn travelling(pair: &[Field<Point>; 2], phase: f64) -> Field<Point> {
    let p = Plane::from_angle(phase);
    &pair[0].scaled(p.e1()) + &pair[1].scaled(p.e2())
}
