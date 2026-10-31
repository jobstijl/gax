//! numga's `mechanics/symmetry`: which responses survive averaging over a crystal's point group
//! (Neumann's principle)? A response is a map, and moving a map is a sandwich of its output
//! composed with the inverse sandwich of its input; averaging over a group (the Reynolds
//! projection) leaves the closest invariant response. Heat conduction (a map on vectors, VGA3D)
//! loses components group by group, from six to one; the same 24 cube rotations that make a
//! lattice's conduction isotropic leave its rank-four elasticity cubic; and a rotary power, the
//! turn of light's polarization per distance along a direction, is an intrinsic scalar, a form
//! on two directions (antivectors) with a sense of its own: mirrors and the inversion, odd
//! versors, reverse it, so a group with the inversion allows none, the cube's rotations leave
//! one in every direction, and a rotoinversion leaves opposite senses along two axes. The
//! animation shows the three in turn, with a driving field sweeping round the heat-flow
//! ellipsoids, the rotary-power surfaces coloured by their sense, and the lattice responses
//! turning. numga has replaced the three-armed flywheel by the optical activity; the port keeps
//! the flywheel's inertia check among its tests.

use gax::Unit;
#[cfg(test)]
use gax::motions::{Motions, Pga3d};
use gax::{pga3d, vga3d};

use gax_numga_examples::scene3::panel3;
use gax_numga_examples::{
    Align, Anim, Camera, Canvas, Lens, Light, Marker, Point2, Rect, Scene3, backdrop, caption,
    on_sphere, palette, reach3, run_with,
};

mod symmetry {
    use super::*;
    use core::f64::consts::PI;
    pub use vga3d::{Bivector, Scalar, Vector};

    pub type R = Unit<vga3d::Rotor>;
    /// A response of vectors to vectors: a conductivity.
    pub type Response = Vector<(Vector,)>;
    /// A form with four vector slots: an elasticity.
    pub type Elasticity = Scalar<(Vector, Vector, Vector, Vector)>;
    pub type P3 = pga3d::Point;
    /// A rate of turning: an axis line, the meet of two planes.
    #[cfg(test)]
    pub type L3 = pga3d::Line;
    /// Inertia: rigid-body velocities (twists, extrinsic lines) to momenta (forques).
    #[cfg(test)]
    pub type Inertia = pga3d::Forque<(pga3d::Line,)>;

    pub fn x() -> Vector {
        Vector::new(1.0, 0.0, 0.0)
    }
    pub fn y() -> Vector {
        Vector::new(0.0, 1.0, 0.0)
    }
    pub fn z() -> Vector {
        Vector::new(0.0, 0.0, 1.0)
    }
    /// The planes `yz`, `zx` and `xy` as unit bivectors.
    pub fn yz() -> Bivector {
        Bivector::new(1.0, 0.0, 0.0)
    }
    pub fn zx() -> Bivector {
        Bivector::new(0.0, 1.0, 0.0)
    }
    pub fn xy() -> Bivector {
        Bivector::new(0.0, 0.0, 1.0)
    }

    /// The cyclic group of `order` equal turns in a unit plane, the identity first. A rotor turns
    /// by twice its angle: element `k`, `exp(plane π k / order)`, turns by `2 π k / order`.
    pub fn turns(plane: Bivector, order: usize) -> Vec<R> {
        (0..order)
            .map(|k| plane.gp(PI * k as f64 / order as f64).exp())
            .collect()
    }

    /// The 24 rotations of a cube: each of the six faces turned to the top, then four quarter
    /// turns about `z`.
    pub fn cube_rotations() -> Vec<R> {
        let tilt = [0.0, 1.0, 2.0, 3.0, 0.0, 0.0];
        let roll = [0.0, 0.0, 0.0, 0.0, 1.0, -1.0];
        let quarter = turns(xy(), 4);
        tilt.iter()
            .zip(roll)
            .flat_map(|(t, r)| {
                let face = (yz().gp(PI / 4.0 * t) + zx().gp(PI / 4.0 * r)).exp();
                quarter.iter().map(move |q| face * *q)
            })
            .collect()
    }

    /// A sphere of unit driving fields, 33 latitudes by 65 longitudes.
    pub fn directions() -> Vec<Vector> {
        (0..33)
            .flat_map(|i| {
                let lat = -PI / 2.0 + PI * i as f64 / 32.0;
                (0..65).map(move |j| on_sphere(2.0 * PI * j as f64 / 64.0, lat))
            })
            .collect()
    }

    /// Equal-mass samples of a thin rectangular flywheel arm on a massless hub: 24 along `x`
    /// from 0.25 to 2.2, 5 across from -0.14 to 0.14. (numga moves the origin by translations;
    /// the points are the same.)
    #[cfg(test)]
    pub fn arm_samples() -> Vec<P3> {
        (0..24)
            .flat_map(|i| {
                let a = 0.25 + (2.2 - 0.25) * i as f64 / 23.0;
                (0..5).map(move |j| pga3d::Point::xyz(a, -0.14 + 0.28 * j as f64 / 4.0, 0.0))
            })
            .collect()
    }

    /// The sites of a simple cubic lattice, and the central site's two neighbour shells.
    pub fn lattice_samples() -> (Vec<P3>, Vec<P3>, Vec<P3>) {
        let (mut sites, mut axial, mut diagonal) = (Vec::new(), Vec::new(), Vec::new());
        for i in -1i32..=1 {
            for j in -1i32..=1 {
                for k in -1i32..=1 {
                    let p = pga3d::Point::xyz(f64::from(i), f64::from(j), f64::from(k));
                    sites.push(p);
                    match i * i + j * j + k * k {
                        1 => axial.push(p),
                        2 => diagonal.push(p),
                        _ => {}
                    }
                }
            }
        }
        (sites, axial, diagonal)
    }

    /// A positive conductivity with principal `axes` and `gains`, followed by its average over
    /// each group. Each principal axis measures one component of the driving field and conducts
    /// along itself. Averaging pulls the input into each rotated frame, applies the conductivity
    /// and turns the output back; a rotation of the group only permutes the terms, so the mean
    /// is invariant: the closest invariant response in the Frobenius norm.
    pub fn conductivities(axes: &[Vector; 3], gains: [f64; 3], groups: &[Vec<R>]) -> Vec<Response> {
        let conductivity: Response = axes
            .iter()
            .zip(gains)
            .map(|(a, g)| (*a * (*a | Vector::slot())).gp(g))
            .sum();
        let average = |group: &Vec<R>| {
            let sum: Response = group
                .iter()
                .map(|g| *g >> conductivity.of(*g << Vector::slot()))
                .sum();
            sum.gp(1.0 / group.len() as f64)
        };
        core::iter::once(conductivity)
            .chain(groups.iter().map(average))
            .collect()
    }

    /// One arm's inertia (unit mass), and the flywheel's: the arm's turned by each rotation and
    /// added. Each sample's inertia is `Motions::point_inertia`: the commutator gives its velocity
    /// under the open rigid-motion slot, and the sample joined with its velocity is its momentum.
    #[cfg(test)]
    pub fn flywheel_inertia(arm: &[P3], rotations: &[Unit<pga3d::Motor>]) -> (Inertia, Inertia) {
        let arm_inertia = arm
            .iter()
            .map(|p| Pga3d::point_inertia(*p))
            .sum::<Inertia>()
            .gp(1.0 / arm.len() as f64);
        // Unlike conductivity's mean, this sum assembles masses: the wheel has mass three.
        let inertia: Inertia = rotations
            .iter()
            .map(|r| *r >> arm_inertia.of(*r << pga3d::Line::slot()))
            .sum();
        (arm_inertia, inertia)
    }

    /// Axial and face-diagonal bond families, each the orbit of a seed under the cube group,
    /// averaged and added: the conductivity and the elasticity. A unit bond `n` measures a
    /// gradient by `n | g`; an imposed strain `s d (d | v)` stretches it by `s (n | d)²`, so the
    /// spring energy has four factors `n | d`: a form with four vector slots.
    pub fn lattice_responses(seeds: &[Vector; 2], cube: &[R]) -> (Response, Elasticity) {
        let mut conductivity: Response = Vector::zero();
        let mut elasticity: Elasticity = Scalar::zero();
        let k = 1.0 / cube.len() as f64;
        for seed in seeds {
            for g in cube {
                let bond = *g >> *seed;
                let projection = bond | Vector::slot();
                conductivity += (bond * projection).gp(k);
                elasticity += (projection * projection * projection * projection).gp(k);
            }
        }
        (conductivity, elasticity)
    }

    /// A light path's direction: a direction carried by geometry, an antivector (intrinsic).
    pub type Direction = vga3d::intrinsic::Bivector;
    /// A turn per distance with a sense of its own: an intrinsic scalar.
    pub type RotaryPower = vga3d::intrinsic::Scalar;
    /// A rotary power along each direction: a form on two directions.
    pub type Gyration = vga3d::intrinsic::Scalar<(Direction, Direction)>;
    /// An improper rotation, a mirror or the inversion: an odd versor.
    pub type Improper = vga3d::Odd;

    /// The inversion, the pseudoscalar as an odd versor.
    pub fn inversion() -> Improper {
        vga3d::Odd::new(0.0, 0.0, 0.0, 1.0)
    }

    /// A unit direction: the antivector of a unit vector, its dual.
    pub fn direction(v: Vector) -> Direction {
        v.dual()
    }

    /// A candidate rotary power with principal `planes` and `powers`, followed by its average
    /// over each point group, its rotations and its improper rotations. Light crossing an
    /// optically active crystal along a unit direction `d` has its plane of polarization turned,
    /// in proportion to the distance, by `gyration(d, d)`; each principal plane reads the
    /// direction, `plane & d`. The turn has a sense, so the rotary power is intrinsic: an
    /// improper rotation reverses it, and a group with mirrors or the inversion allows less
    /// optical activity than its rotations alone, and none with the inversion.
    pub fn rotary_powers(
        planes: &[Vector; 3],
        powers: [f64; 3],
        groups: &[(Vec<R>, Vec<Improper>)],
    ) -> Vec<Gyration> {
        let gyration: Gyration = planes
            .iter()
            .zip(powers)
            .map(|(plane, power)| {
                let reading = *plane & Direction::slot();
                RotaryPower::new(power) * (reading * reading)
            })
            .sum();
        let average = |(rotations, improper): &(Vec<R>, Vec<Improper>)| {
            // Each element turns both directions into its frame and carries the rotary power
            // back out; an improper rotation reverses its sense.
            let turned = rotations.iter().map(|g| {
                let pull = *g << Direction::slot();
                *g >> gyration.of_both(pull, pull)
            });
            let reflected = improper.iter().map(|g| {
                let pull = *g << Direction::slot();
                *g >> gyration.of_both(pull, pull)
            });
            let sum: Gyration = turned.chain(reflected).sum();
            sum.gp(1.0 / (rotations.len() + improper.len()) as f64)
        };
        core::iter::once(gyration)
            .chain(groups.iter().map(average))
            .collect()
    }

    /// numga's principal frame of the measured responses, turned in two planes, and the
    /// candidate rotary powers across its planes.
    pub fn pose() -> R {
        (xy().gp(0.3) + yz().gp(0.16)).exp()
    }
    pub const GAINS: [f64; 3] = [4.5, 1.5, 0.6];
    pub const ROTARY_VALUES: [f64; 3] = [3.0, -1.0, 0.5];

    /// The optical activity scene: a tilted candidate rotary power averaged over five point
    /// groups: the quarter turns, with the inversion, through the inversion (the half turns
    /// and the quarter turns times the inversion, a rotoinversion), the cube's rotations, and
    /// with the inversion.
    pub fn optical_activity() -> Vec<Gyration> {
        let pose = pose();
        let planes = [pose >> x(), pose >> y(), pose >> z()];
        let quarter = turns(xy(), 4);
        let cube = cube_rotations();
        let inverted = |rotors: &[R]| -> Vec<Improper> {
            rotors
                .iter()
                .map(|r| inversion() * r.into_inner())
                .collect()
        };
        let halves: Vec<R> = quarter.iter().step_by(2).copied().collect();
        let odd_quarters: Vec<R> = quarter.iter().skip(1).step_by(2).copied().collect();
        let groups = [
            (quarter.clone(), vec![]),
            (quarter.clone(), inverted(&quarter)),
            (halves, inverted(&odd_quarters)),
            (cube.clone(), vec![]),
            (cube.clone(), inverted(&cube)),
        ];
        rotary_powers(&planes, ROTARY_VALUES, &groups)
    }

    /// The conductivity scene: the response labels and the responses (measured, then averaged
    /// over half turns about x and z, quarter turns about z, and the cube's rotations).
    pub fn heat_conduction() -> Vec<Response> {
        let pose = pose();
        let axes = [pose >> x(), pose >> y(), pose >> z()];
        let gains = GAINS;
        let half: Vec<R> = turns(yz(), 2)
            .iter()
            .flat_map(|a| turns(xy(), 2).into_iter().map(move |b| *a * b))
            .collect();
        let groups = [half, turns(xy(), 4), cube_rotations()];
        let responses = conductivities(&axes, gains, &groups);

        // Averaging keeps the trace, so the cube-invariant conductivity is the mean gain times
        // the identity.
        let cube = responses[3];
        let mean = GAINS.iter().sum::<f64>() / 3.0;
        for d in directions() {
            let e = cube.of(d) - d.gp(mean);
            assert!(e.norm() < 1e-7);
        }
        // Quarter turns about z conduct equally along x and y.
        let quarter = responses[2];
        assert!(((quarter.of(x()) | x()).s() - (quarter.of(y()) | y()).s()).abs() < 1e-6);
        responses
    }

    /// The flywheel scene: the arms' samples, and the inertia of one arm and of the wheel.
    #[cfg(test)]
    pub fn flywheel() -> (Vec<Vec<P3>>, Inertia, Inertia) {
        let arm = arm_samples();
        let rotations: Vec<_> = (0..3)
            .map(|k| {
                pga3d::Motor::rotation_about(
                    pga3d::Point::direction(0.0, 0.0, 1.0),
                    2.0 * PI * k as f64 / 3.0,
                )
            })
            .collect();
        let (arm_inertia, inertia) = flywheel_inertia(&arm, &rotations);
        let arms = rotations
            .iter()
            .map(|r| arm.iter().map(|p| *r >> *p).collect())
            .collect();

        // Threefold symmetry makes every transverse moment equal; about z it is twice as large.
        let moments: Vec<f64> = (0..181)
            .map(|k| moment(&inertia, probe(2.0 * PI * k as f64 / 180.0)))
            .collect();
        let (lo, hi) = moments
            .iter()
            .fold((f64::MAX, f64::MIN), |(a, b), m| (a.min(*m), b.max(*m)));
        assert!(hi - lo < 1e-8 * hi.abs());
        let mean = moments.iter().sum::<f64>() / moments.len() as f64;
        let axial = moment(&inertia, z_axis());
        assert!((axial - 2.0 * mean).abs() < 1e-8 * axial.abs());
        (arms, arm_inertia, inertia)
    }

    /// The axis through the hub at `angle` from `x` in the wheel's plane.
    #[cfg(test)]
    pub fn probe(angle: f64) -> L3 {
        // The x axis, where the planes y = 0 and z = 0 meet.
        let x_axis = pga3d::Plane::new(0.0, 1.0, 0.0, 0.0) ^ pga3d::Plane::new(0.0, 0.0, 1.0, 0.0);
        pga3d::Motor::rotation_about(pga3d::Point::direction(0.0, 0.0, 1.0), angle) >> x_axis
    }

    /// The wheel's axis, where the planes x = 0 and y = 0 meet.
    #[cfg(test)]
    pub fn z_axis() -> L3 {
        pga3d::Plane::new(1.0, 0.0, 0.0, 0.0) ^ pga3d::Plane::new(0.0, 1.0, 0.0, 0.0)
    }

    /// The moment about a unit axis: the regressive pairing of the rotation with the momentum
    /// it gives.
    #[cfg(test)]
    pub fn moment(inertia: &Inertia, axis: L3) -> f64 {
        (axis & inertia.of(axis)).s()
    }

    /// The lattice scene's responses.
    pub fn crystal_lattice() -> (Response, Elasticity) {
        let seeds = [x(), (x() + y()).normalized()];
        let (conductivity, elasticity) = lattice_responses(&seeds, &cube_rotations());
        // Conduction is isotropic, 2/3 in every direction; stiffness is 1/2 along the cube's
        // axes and 1/3 along its body diagonals.
        for d in directions() {
            assert!(((d | conductivity.of(d)).s() - 2.0 / 3.0).abs() < 1e-7 * 2.0 / 3.0);
        }
        let diagonal = (x() + y() + z()).normalized();
        assert!((elasticity.fill(x()).s() - 0.5).abs() < 1e-7 * 0.5);
        assert!((elasticity.fill(diagonal).s() - 1.0 / 3.0).abs() < 1e-7 / 3.0);
        (conductivity, elasticity)
    }
}

use symmetry::*;

/// Everything the frames show, computed once.
struct Scenes {
    conduction: Vec<Response>,
    conductivity: Response,
    elasticity: Elasticity,
    rotary: Vec<Gyration>,
}

fn scenes() -> Scenes {
    let (conductivity, elasticity) = crystal_lattice();
    Scenes {
        conduction: heat_conduction(),
        conductivity,
        elasticity,
        rotary: optical_activity(),
    }
}

/// Seconds per scene.
const SCENE: f32 = 7.0;

/// A camera orbiting the origin at `azimuth` and `elevation`, parallel, showing `half` units
/// above and below the centre of the panel `rect`.
fn orbit(rect: Rect, azimuth: f32, elevation: f32, half: f32) -> Camera {
    let origin = P3::origin();
    Camera::orbit(rect, origin, 20.0, azimuth, elevation, Lens::Parallel(half))
}

/// A surface `radius(d) d` (or the image `f(d)` of the unit sphere) in a scene: longitude by
/// latitude.
fn sphere_surface(sc: &mut Scene3, f: impl Fn(Vector) -> Vector, colour: Light, opacity: f32) {
    sc.surface(
        |u, v| {
            let d = on_sphere(
                core::f64::consts::TAU * f64::from(u),
                core::f64::consts::PI * (f64::from(v) - 0.5),
            );
            reach3(f(d))
        },
        28,
        14,
        |_, _| colour,
        opacity,
        Some((colour.faded(0.6), 0.6)),
    );
}

/// The middle of a rectangle's top edge, and of its bottom edge.
fn top_middle(r: Rect) -> Point2 {
    r.top_middle()
}
fn bottom_middle(r: Rect) -> Point2 {
    r.bottom_middle()
}

/// The angle between two vectors in degrees.
fn degrees(a: Vector, b: Vector) -> f64 {
    a.angle_between(b).to_degrees()
}

fn conduction_scene(c: &mut Canvas, sc_data: &Scenes, s: f32) {
    let screen = c.rect();
    let h = screen.height();
    // The band of the panels, under the caption.
    let band = screen.inset(0.0, h * 0.16, 0.0, h * 0.14);
    let down = Point2::direction(0.0, 1.0);
    let labels = [
        ("MEASURED", "6 COMPONENTS"),
        ("HALF TURNS", "3 COMPONENTS"),
        ("QUARTER TURNS", "2 COMPONENTS"),
        ("CUBE ROTATIONS", "1 COMPONENT"),
    ];
    // The driving field sweeps a cone about the vertical.
    let phase = f64::from(s) * core::f64::consts::TAU;
    let driving = on_sphere(phase, 0.45);
    for (i, (k, (title, sub))) in sc_data.conduction.iter().zip(labels).enumerate() {
        let rect = band.column(i, 4);
        let origin = P3::origin();
        let flux = k.of(driving);
        panel3(c, rect, orbit(rect, -2.1 + 0.6 * s, 0.42, 7.4), |sc| {
            sc.axes(origin, 1.2);
            sphere_surface(sc, |d| k.of(d), palette::SKY, 0.3);
            sc.arrow(origin, driving.gp(4.0), 2.0, 9.0, palette::INK);
            sc.arrow(origin, flux, 3.0, 11.0, palette::ORANGE);
        });
        let (head, foot) = (top_middle(rect), bottom_middle(rect));
        c.text(
            title,
            head + down.gp(14.0),
            13.0,
            palette::INK,
            Align::Center,
        );
        c.text(
            sub,
            head + down.gp(30.0),
            10.0,
            palette::GRID,
            Align::Center,
        );
        let deflection = format!("DEFLECTION {:.1} DEG", degrees(flux, driving));
        let at = foot + down.gp(22.0);
        c.text(&deflection, at, 11.0, palette::ORANGE, Align::Center);
    }
    caption(
        c,
        "SYMMETRY: HEAT FLOW ALLOWED BY CRYSTAL ROTATIONS",
        "SURFACES: K OF EVERY UNIT FIELD. WHITE: FIELD, ORANGE: HEAT FLOW (VGA3D)",
    );
}

/// The sign of a rotary power, by colour: orange where light turns one way, blue the other.
fn sense(power: f64) -> Light {
    if power >= 0.0 {
        palette::ORANGE
    } else {
        palette::SKY
    }
}

/// The rotary powers along every direction, as a surface at their size, coloured by their
/// sense: the candidate and its averages over five point groups.
fn optical_scene(c: &mut Canvas, sd: &Scenes, s: f32) {
    let screen = c.rect();
    let h = screen.height();
    let band = screen.inset(0.0, h * 0.13, 0.0, h * 0.02);
    let down = Point2::direction(0.0, 1.0);
    let labels = [
        ("MEASURED", "ANY ROTARY POWER"),
        ("QUARTER TURNS", "ALONG Z AND ACROSS IT"),
        ("WITH THE INVERSION", "NONE"),
        ("THROUGH THE INVERSION", "OPPOSITE ALONG X AND Y"),
        ("CUBE ROTATIONS", "ONE, IN EVERY DIRECTION"),
        ("WITH THE INVERSION", "NONE"),
    ];
    // The unit direction at a point of the parameter square.
    let at = |u: f32, v: f32| {
        on_sphere(
            core::f64::consts::TAU * f64::from(u),
            core::f64::consts::PI * (f64::from(v) - 0.5),
        )
    };
    for (i, (g, (title, sub))) in sd.rotary.iter().zip(labels).enumerate() {
        let power = |d: Vector| g.of(direction(d)).of(direction(d)).s();
        let row = band.part(0.0, 0.5 * (i / 3) as f32, 1.0, 0.5 * (i / 3 + 1) as f32);
        let rect = row.column(i % 3, 3);
        let origin = P3::origin();
        panel3(c, rect, orbit(rect, -2.1 + 0.6 * s, 0.42, 3.3), |sc| {
            sc.axes(origin, 1.2);
            sc.surface(
                |u, v| {
                    let d = at(u, v);
                    reach3(d.gp(power(d).abs()))
                },
                28,
                14,
                |u, v| sense(power(at(u, v))),
                0.45,
                None,
            );
        });
        let head = top_middle(rect);
        c.text(
            title,
            head + down.gp(12.0),
            11.0,
            palette::INK,
            Align::Center,
        );
        c.text(sub, head + down.gp(25.0), 9.0, palette::GRID, Align::Center);
    }
    caption(
        c,
        "SYMMETRY: OPTICAL ACTIVITY ALLOWED BY A POINT GROUP",
        "A ROTARY POWER IS INTRINSIC: MIRRORS AND THE INVERSION REVERSE IT (VGA3D)",
    );
}

fn lattice_scene(c: &mut Canvas, sd: &Scenes, s: f32) {
    let screen = c.rect();
    let h = screen.height();
    // The band of the panels, under the caption.
    let band = screen.inset(0.0, h * 0.16, 0.0, h * 0.12);
    let down = Point2::direction(0.0, 1.0);
    let azimuth = -1.0 + 1.2 * s;
    let (sites, axial, diagonal) = lattice_samples();
    let titles = [
        ("TWO SEED BONDS GENERATE", "6 AXIAL, 12 DIAGONAL BONDS"),
        ("CONDUCTIVITY: ISOTROPIC", "D.K(D) = 2/3 EVERYWHERE"),
        ("STIFFNESS: CUBIC", "C: 1/2 ON AXES, 1/3 DIAGONAL"),
    ];
    for (i, (title, sub)) in titles.iter().enumerate() {
        let rect = band.column(i, 3);
        let half = if i == 0 { 1.9 } else { 0.85 };
        let origin = P3::origin();
        panel3(c, rect, orbit(rect, azimuth, 0.4, half), |sc| match i {
            0 => {
                for q in &sites {
                    sc.dot(*q, Marker::Dot, 6.0, palette::GRID);
                }
                for (shell, colour) in [(&axial, palette::ORANGE), (&diagonal, palette::SKY)] {
                    for q in shell {
                        sc.seg(origin, *q, 2.0, colour);
                        sc.dot(*q, Marker::Dot, 8.0, colour);
                    }
                }
                sc.dot(origin, Marker::Dot, 11.0, palette::INK);
            }
            1 => {
                let k = sd.conductivity;
                sphere_surface(sc, |d| d.gp((d | k.of(d)).s()), palette::SKY, 0.85);
            }
            _ => {
                let e4 = sd.elasticity;
                sphere_surface(sc, |d| d.gp(e4.fill(d).s()), palette::ORANGE, 0.85);
            }
        });
        let (head, foot) = (top_middle(rect), bottom_middle(rect));
        c.text(
            title,
            head + down.gp(14.0),
            13.0,
            palette::INK,
            Align::Center,
        );
        c.text(
            sub,
            foot + down.gp(18.0),
            10.0,
            palette::GRID,
            Align::Center,
        );
    }
    caption(
        c,
        "SYMMETRY: ONE CUBE GROUP, TWO KINDS OF RESPONSE",
        "24 CUBE ROTATIONS: RANK-2 CONDUCTION ISOTROPIC, RANK-4 ELASTICITY CUBIC",
    );
}

fn draw(c: &mut Canvas, scenes: &Scenes, t: f32) {
    backdrop(c);
    let u = (t / SCENE).rem_euclid(3.0);
    let s = u.fract();
    match u as usize {
        0 => conduction_scene(c, scenes, s),
        1 => optical_scene(c, scenes, s),
        _ => lattice_scene(c, scenes, s),
    }
}

fn main() {
    run_with(
        Anim::new("symmetry", 3.0 * SCENE).size(960, 540),
        scenes,
        draw,
    );
}

#[cfg(test)]
mod tests {
    use super::symmetry::*;
    use gax::vga3d;
    use gax_numga_examples::on_sphere;

    /// Every pair of the 24 rotors differs, even up to sign: their scalar overlap is not plus or
    /// minus one.
    #[test]
    fn cube_rotations_are_24_distinct_rotations() {
        let cube = cube_rotations();
        assert_eq!(cube.len(), 24);
        for a in &cube {
            let same = cube
                .iter()
                .filter(|b| a.reverse().scalar_product(b.into_inner()).s().abs() > 1.0 - 1e-9)
                .count();
            assert_eq!(same, 1);
        }
    }

    /// The scenarios' checks (they assert inside) and every frame's scene draws.
    #[test]
    fn scenario_checks_and_frames() {
        let responses = heat_conduction();
        assert_eq!(responses.len(), 4);
        let _ = flywheel();
        let _ = crystal_lattice();
        let scenes = super::scenes();
        for t in [0.5, super::SCENE + 0.5, 2.0 * super::SCENE + 0.5] {
            let c = gax_numga_examples::app::frame(
                &gax_numga_examples::Anim::new("t", 1.0).size(320, 180),
                t,
                &mut |c, t| super::draw(c, &scenes, t),
            );
            assert!(c.mean().luma() > 0.0);
        }
    }

    /// numga's optical-activity checks: the inversion removes every rotary power; the cube's
    /// rotations leave the mean one in every direction, as in a sugar solution; the
    /// rotoinversion leaves opposite senses along x and y and none along z; and the same
    /// principal values as a conductivity, which pairs a field with a field, survive the
    /// inversion.
    #[test]
    fn the_inversion_removes_rotary_power_and_the_cube_makes_it_isotropic() {
        let powers = optical_activity();
        assert_eq!(powers.len(), 6);
        let along = |g: &Gyration, v: Vector| {
            let d = direction(v);
            g.of(d).of(d).s()
        };
        let mean = ROTARY_VALUES.iter().sum::<f64>() / 3.0;
        for v in directions() {
            for centrosymmetric in [&powers[2], &powers[5]] {
                assert!(along(centrosymmetric, v).abs() < 1e-12);
            }
            assert!((along(&powers[4], v) - mean).abs() < 1e-12);
        }
        let rotoinversion = &powers[3];
        assert!((along(rotoinversion, x()) + along(rotoinversion, y())).abs() < 1e-12);
        assert!(along(rotoinversion, z()).abs() < 1e-12);
        // The rotoinversion's crystal is its own mirror image, yet turns light along x.
        assert!(along(rotoinversion, x()).abs() > 0.1);
        // The candidate turns light both ways, by direction.
        let candidate = &powers[0];
        let pose = pose();
        assert!((along(candidate, pose >> x()) - ROTARY_VALUES[0]).abs() < 1e-12);
        assert!((along(candidate, pose >> y()) - ROTARY_VALUES[1]).abs() < 1e-12);
        let candidate = conductivities(&[x(), y(), z()].map(|a| pose >> a), ROTARY_VALUES, &[])[0];
        let inverted = inversion() >> candidate.of(inversion() << Vector::slot());
        for v in directions() {
            assert!((inverted.of(v) - candidate.of(v)).norm() < 1e-12);
        }
    }

    /// A mirror reverses a rotary power: the mirror image of a crystal turns light the other
    /// way.
    #[test]
    fn a_mirror_reverses_the_sense_of_rotary_power() {
        let candidate = optical_activity()[0];
        let mirror = vga3d::Odd::new(0.0, 0.0, 1.0, 0.0);
        let mirrored =
            mirror >> candidate.of_both(mirror << Direction::slot(), mirror << Direction::slot());
        for v in directions() {
            let d = direction(v);
            let (a, b) = (
                candidate.of(d).of(d).s(),
                mirrored.of(mirror >> d).of(mirror >> d).s(),
            );
            assert!((a + b).abs() < 1e-12, "{a} {b}");
        }
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws_with(super::draw, &super::scenes(), 0.5);
    }

    /// Averaging over a group keeps the trace (every term is a rotated copy) and makes the
    /// response commute with the group's rotations.
    #[test]
    fn averages_keep_the_trace_and_commute_with_the_group() {
        let responses = heat_conduction();
        let trace = responses[0].trace();
        for r in &responses {
            assert!((r.trace() - trace).abs() < 1e-12);
        }
        let quarter = responses[2];
        for g in turns(xy(), 4) {
            let d = on_sphere(0.3, 0.7);
            let a = g >> quarter.of(d);
            let b = quarter.of(g >> d);
            assert!((a - b).norm() < 1e-12);
        }
    }
}
