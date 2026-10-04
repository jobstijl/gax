//! numga's `mechanics/symmetry`: which responses survive averaging over an object's rotation
//! symmetries (Neumann's principle)? A response is a map, and rotating a map is a sandwich of its
//! output composed with the inverse sandwich of its input; averaging over a group (the Reynolds
//! projection) leaves the closest invariant response. Heat conduction (a map on vectors, VGA3D)
//! loses components group by group, from six to one; a three-armed flywheel's inertia (a map
//! from twists to momenta, PGA3D), assembled by adding rotated copies of one arm's, has equal
//! moments about every axis in its plane; and the same 24 cube rotations that make a lattice's
//! conduction isotropic leave its rank-four elasticity cubic. The animation shows the three in
//! turn, with a driving field sweeping round the heat-flow ellipsoids, a probe axis sweeping the
//! wheel's plane, and the lattice responses turning.

use gax::Unit;
use gax::motions::{Motions, Pga3d};
use gax::{pga2d, pga3d, vga3d};

use gax_numga_examples::scene3::panel3;
use gax_numga_examples::{
    Align, Anim, Axes, Camera, Canvas, Lens, Light, Marker, Point2, Rect, Scene3, backdrop,
    caption, from_above, palette, reach3, run,
};
use std::sync::OnceLock;

mod symmetry {
    use super::*;
    use core::f64::consts::PI;
    pub use vga3d::{Bivector, Scalar, Vector};

    pub type V = Vector<(), f64>;
    pub type R = Unit<vga3d::Rotor<(), f64>>;
    /// A response of vectors to vectors: a conductivity.
    pub type Response = Vector<(Vector,), f64>;
    /// A form with four vector slots: an elasticity.
    pub type Elasticity = Scalar<(Vector, Vector, Vector, Vector), f64>;
    pub type P3 = pga3d::Point<(), f64>;
    pub type L3 = pga3d::Line<(), f64>;
    /// Inertia: rigid-body velocities (lines) to momenta (lines).
    pub type Inertia = pga3d::Line<(pga3d::Line,), f64>;

    pub fn x() -> V {
        Vector::new(1.0, 0.0, 0.0)
    }
    pub fn y() -> V {
        Vector::new(0.0, 1.0, 0.0)
    }
    pub fn z() -> V {
        Vector::new(0.0, 0.0, 1.0)
    }
    /// The planes `yz`, `zx` and `xy` as unit bivectors.
    pub fn yz() -> Bivector<(), f64> {
        Bivector::new(1.0, 0.0, 0.0)
    }
    pub fn zx() -> Bivector<(), f64> {
        Bivector::new(0.0, 1.0, 0.0)
    }
    pub fn xy() -> Bivector<(), f64> {
        Bivector::new(0.0, 0.0, 1.0)
    }

    /// The cyclic group of `order` equal turns in a unit plane, the identity first. A rotor turns
    /// by twice its angle: element `k`, `exp(plane π k / order)`, turns by `2 π k / order`.
    pub fn turns(plane: Bivector<(), f64>, order: usize) -> Vec<R> {
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

    /// A unit direction: `x` turned up towards `z` by the latitude, then about `z` by the
    /// longitude.
    pub fn direction(longitude: f64, latitude: f64) -> V {
        // numga's `xz` is `-zx`.
        let r = xy().gp(-longitude / 2.0).exp() * zx().gp(latitude / 2.0).exp();
        r >> x()
    }

    /// A sphere of unit driving fields, 33 latitudes by 65 longitudes.
    pub fn directions() -> Vec<V> {
        let mut out = Vec::new();
        for i in 0..33 {
            for j in 0..65 {
                let lat = -PI / 2.0 + PI * i as f64 / 32.0;
                let lon = 2.0 * PI * j as f64 / 64.0;
                out.push(direction(lon, lat));
            }
        }
        out
    }

    /// Equal-mass samples of a thin rectangular flywheel arm on a massless hub: 24 along `x`
    /// from 0.25 to 2.2, 5 across from -0.14 to 0.14. (numga moves the origin by translations;
    /// the points are the same.)
    pub fn arm_samples() -> Vec<P3> {
        let mut out = Vec::new();
        for i in 0..24 {
            for j in 0..5 {
                let a = 0.25 + (2.2 - 0.25) * i as f64 / 23.0;
                let b = -0.14 + 0.28 * j as f64 / 4.0;
                out.push(pga3d::Point::xyz(a, b, 0.0));
            }
        }
        out
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
    pub fn conductivities(axes: &[V; 3], gains: [f64; 3], groups: &[Vec<R>]) -> Vec<Response> {
        let conductivity = axes
            .iter()
            .zip(gains)
            .fold(Vector::zero(), |s: Response, (a, g)| {
                s + (*a * (*a | Vector::slot())).gp(g)
            });
        let mut out = vec![conductivity];
        for group in groups {
            let sum = group.iter().fold(Vector::zero(), |s: Response, g| {
                s + (*g >> conductivity.of(*g << Vector::slot()))
            });
            out.push(sum.gp(1.0 / group.len() as f64));
        }
        out
    }

    /// One arm's inertia (unit mass), and the flywheel's: the arm's turned by each rotation and
    /// added. Each sample's inertia is `Motions::point_inertia`: the commutator gives its velocity
    /// under the open rigid-motion slot, and the sample joined with its velocity is its momentum.
    pub fn flywheel_inertia(
        arm: &[P3],
        rotations: &[Unit<pga3d::Motor<(), f64>>],
    ) -> (Inertia, Inertia) {
        let arm_inertia = arm
            .iter()
            .fold(pga3d::Line::zero(), |s: Inertia, p| {
                s + Pga3d::point_inertia(*p)
            })
            .gp(1.0 / arm.len() as f64);
        // Unlike conductivity's mean, this sum assembles masses: the wheel has mass three.
        let inertia = rotations.iter().fold(pga3d::Line::zero(), |s: Inertia, r| {
            s + (*r >> arm_inertia.of(*r << pga3d::Line::slot()))
        });
        (arm_inertia, inertia)
    }

    /// Axial and face-diagonal bond families, each the orbit of a seed under the cube group,
    /// averaged and added: the conductivity and the elasticity. A unit bond `n` measures a
    /// gradient by `n | g`; an imposed strain `s d (d | v)` stretches it by `s (n | d)²`, so the
    /// spring energy has four factors `n | d`: a form with four vector slots.
    pub fn lattice_responses(seeds: &[V; 2], cube: &[R]) -> (Response, Elasticity) {
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

    /// The conductivity scene: the response labels and the responses (measured, then averaged
    /// over half turns about x and z, quarter turns about z, and the cube's rotations).
    pub fn heat_conduction() -> Vec<Response> {
        let pose = (xy().gp(0.3) + yz().gp(0.16)).exp();
        let axes = [pose >> x(), pose >> y(), pose >> z()];
        let gains = [4.5, 1.5, 0.6];
        let half: Vec<R> = turns(yz(), 2)
            .iter()
            .flat_map(|a| turns(xy(), 2).into_iter().map(move |b| *a * b))
            .collect();
        let groups = [half, turns(xy(), 4), cube_rotations()];
        let responses = conductivities(&axes, gains, &groups);

        // Averaging keeps the trace, so the cube-invariant conductivity is the mean gain, 2.2,
        // times the identity.
        let cube = responses[3];
        for d in directions() {
            let e = cube.of(d) - d.gp(2.2);
            assert!(e.norm() < 1e-7);
        }
        // Quarter turns about z conduct equally along x and y.
        let quarter = responses[2];
        assert!(((quarter.of(x()) | x()).s() - (quarter.of(y()) | y()).s()).abs() < 1e-6);
        responses
    }

    /// The flywheel scene: the arms' samples, and the inertia of one arm and of the wheel.
    pub fn flywheel() -> (Vec<Vec<P3>>, Inertia, Inertia) {
        let arm = arm_samples();
        let rotations: Vec<_> = (0..3)
            .map(|k| pga3d::Motor::rotation_about(0.0, 0.0, 1.0, 2.0 * PI * k as f64 / 3.0))
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
    pub fn probe(angle: f64) -> L3 {
        let x_axis = pga3d::Point::xyz(0.0, 0.0, 0.0) & pga3d::Point::direction(1.0, 0.0, 0.0);
        pga3d::Motor::rotation_about(0.0, 0.0, 1.0, angle) >> x_axis
    }

    /// The wheel's axis.
    pub fn z_axis() -> L3 {
        pga3d::Point::xyz(0.0, 0.0, 0.0) & pga3d::Point::direction(0.0, 0.0, 1.0)
    }

    /// The moment about a unit axis: the regressive pairing of the rotation with the momentum
    /// it gives.
    pub fn moment(inertia: &Inertia, axis: L3) -> f64 {
        (axis & inertia.of(axis)).s()
    }

    /// The lattice scene's responses.
    pub fn crystal_lattice() -> (Response, Elasticity) {
        let seeds = [x(), (x() + y()).normalized().into_inner()];
        let (conductivity, elasticity) = lattice_responses(&seeds, &cube_rotations());
        // Conduction is isotropic, 2/3 in every direction; stiffness is 1/2 along the cube's
        // axes and 1/3 along its body diagonals.
        for d in directions() {
            assert!(((d | conductivity.of(d)).s() - 2.0 / 3.0).abs() < 1e-7 * 2.0 / 3.0);
        }
        let diagonal = (x() + y() + z()).normalized().into_inner();
        assert!((elasticity.fill(x()).s() - 0.5).abs() < 1e-7 * 0.5);
        assert!((elasticity.fill(diagonal).s() - 1.0 / 3.0).abs() < 1e-7 / 3.0);
        (conductivity, elasticity)
    }
}

use symmetry::*;

/// Everything the frames show, computed once.
struct Scenes {
    conduction: Vec<Response>,
    arms: Vec<Vec<P3>>,
    arm_inertia: Inertia,
    inertia: Inertia,
    conductivity: Response,
    elasticity: Elasticity,
}

fn scenes() -> &'static Scenes {
    static S: OnceLock<Scenes> = OnceLock::new();
    S.get_or_init(|| {
        let (arms, arm_inertia, inertia) = flywheel();
        let (conductivity, elasticity) = crystal_lattice();
        Scenes {
            conduction: heat_conduction(),
            arms,
            arm_inertia,
            inertia,
            conductivity,
            elasticity,
        }
    })
}

/// Seconds per scene.
const SCENE: f32 = 7.0;

/// A camera orbiting the origin at `azimuth` and `elevation`, parallel, showing `half` units
/// above and below the centre of the panel `rect`.
fn orbit(rect: Rect, azimuth: f32, elevation: f32, half: f32) -> Camera {
    let (w, h) = (rect.width() as usize, rect.height() as usize);
    let origin = P3::xyz(0.0, 0.0, 0.0);
    Camera::orbit(w, h, origin, 20.0, azimuth, elevation, Lens::Parallel(half))
}

/// A surface `radius(d) d` (or the image `f(d)` of the unit sphere) in a scene: longitude by
/// latitude.
fn sphere_surface(sc: &mut Scene3, f: impl Fn(V) -> V, colour: Light, opacity: f32) {
    sc.surface(
        |u, v| {
            let d = direction(
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

/// The angle between two vectors in degrees: the norm of the logarithm of the rotor from one to
/// the other, `b a` normalized.
fn degrees(a: V, b: V) -> f64 {
    gax_numga_examples::measure::angle(a, b).to_degrees()
}

fn conduction_scene(c: &mut Canvas, s: f32) {
    let sc_data = scenes();
    let screen = c.rect();
    let h = screen.height();
    // Lengths in pixels at 960 by 540, scaled with the canvas.
    let unit = c.unit();
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
    let driving = direction(phase, 0.45);
    for (i, (k, (title, sub))) in sc_data.conduction.iter().zip(labels).enumerate() {
        let rect = band.column(i, 4);
        let origin = P3::xyz(0.0, 0.0, 0.0);
        let flux = k.of(driving);
        panel3(c, rect, orbit(rect, -2.1 + 0.6 * s, 0.42, 7.4), |sc| {
            sc.axes(origin, 1.2);
            sphere_surface(sc, |d| k.of(d), palette::sky(), 0.3);
            sc.arrow(origin, driving.gp(4.0), 2.0, 9.0, palette::ink());
            sc.arrow(origin, flux, 3.0, 11.0, palette::orange());
        });
        let (head, foot) = (top_middle(rect), bottom_middle(rect));
        c.text(
            title,
            head + down.gp(14.0 * unit),
            13.0 * unit,
            palette::ink(),
            Align::Center,
        );
        c.text(
            sub,
            head + down.gp(30.0 * unit),
            10.0 * unit,
            palette::grid(),
            Align::Center,
        );
        let deflection = format!("DEFLECTION {:.1} DEG", degrees(flux, driving));
        let at = foot + down.gp(22.0 * unit);
        c.text(
            &deflection,
            at,
            11.0 * unit,
            palette::orange(),
            Align::Center,
        );
    }
    caption(
        c,
        "SYMMETRY: HEAT FLOW ALLOWED BY CRYSTAL ROTATIONS",
        "SURFACES: K OF EVERY UNIT FIELD. WHITE: FIELD, ORANGE: HEAT FLOW (VGA3D)",
    );
}

fn flywheel_scene(c: &mut Canvas, s: f32) {
    let sd = scenes();
    let screen = c.rect();
    let (w, h) = (screen.width(), screen.height());
    // Lengths in pixels at 960 by 540, scaled with the canvas.
    let unit = c.unit();
    let angle = f64::from(s) * core::f64::consts::TAU;
    // The wheel from above, the probe axis through the hub.
    let left = screen
        .column(0, 2)
        .inset(w * 0.04, h * 0.17, w * 0.02, h * 0.06);
    let hub = pga2d::Point::xy(0.0, 0.0);
    let ax = Axes::equal(left, hub, 2.4);
    let probe_dir = polar(1.0, angle) - hub;
    ax.axline(c, hub, probe_dir, 1.0, palette::grid());
    let east = pga2d::Point::direction(1.0, 0.0);
    ax.axline(c, hub, east, 0.8, palette::grid().faded(0.5));
    for (k, arm) in sd.arms.iter().enumerate() {
        let colour = if k == 0 {
            palette::orange()
        } else {
            palette::sky()
        };
        let pts: Vec<_> = arm.iter().map(|p| from_above(*p)).collect();
        ax.scatter(c, &pts, Marker::Dot, 4.0, colour.faded(0.9));
    }
    ax.scatter(c, &[hub], Marker::Dot, 9.0, palette::ink());
    let reach = probe_dir.gp(2.3);
    ax.line(c, hub - reach, hub + reach, 2.5, palette::yellow());
    let note = ax.at(0.0, 1.0) + Point2::direction(0.1, -0.3);
    let text = "THREE UNIT-MASS ARMS, 120 DEG APART";
    ax.text(c, note, text, 11.0 * unit, palette::ink(), Align::Left);
    // The moments about axes in the wheel's plane, as a polar plot traced up to the probe.
    let right = screen
        .column(1, 2)
        .inset(w * 0.04, h * 0.17, w * 0.04, h * 0.06);
    let n = 180;
    // The point at the moment about each axis, out along it.
    let curve = |inertia: &Inertia| -> Vec<pga2d::Point<(), f64>> {
        (0..=n)
            .map(|k| {
                let a = core::f64::consts::TAU * k as f64 / n as f64;
                polar(moment(inertia, probe(a)), a)
            })
            .collect()
    };
    let arm_curve = curve(&sd.arm_inertia);
    let wheel_curve = curve(&sd.inertia);
    // The largest moment: the farthest point from the hub.
    let rmax = arm_curve
        .iter()
        .chain(&wheel_curve)
        .map(|p| (*p & hub).norm())
        .fold(0.0, f64::max);
    let pax = Axes::equal(right, hub, rmax as f32 * 1.15);
    polar_grid(&pax, c, rmax);
    for (xy, colour) in [
        (&arm_curve, palette::orange()),
        (&wheel_curve, palette::sky()),
    ] {
        pax.polyline(c, xy, 1.0, colour.faded(0.3));
        let upto = ((s * n as f32) as usize).min(n);
        pax.polyline(c, &xy[..=upto], 2.5, colour);
        pax.scatter(c, &[xy[upto]], Marker::Dot, 8.0, colour);
    }
    pax.legend(
        c,
        &[
            ("ONE ARM", palette::orange()),
            ("WHOLE WHEEL", palette::sky()),
        ],
    );
    let axial = moment(&sd.inertia, z_axis());
    let transverse = moment(&sd.inertia, probe(angle));
    let note = pax.at(0.0, 0.0) + Point2::direction(0.0, rmax as f32 * 0.05);
    let text = format!("ABOUT Z: {axial:.3} = 2 X {transverse:.3}");
    pax.text(c, note, &text, 11.0 * unit, palette::ink(), Align::Left);
    caption(
        c,
        "SYMMETRY: THREE ARMS, AXIALLY SYMMETRIC INERTIA",
        "ONE ARM TURNED AND ADDED. RADIUS: MOMENT ABOUT EACH AXIS IN PLANE (PGA3D)",
    );
}

/// The point at radius `r` and angle `a` in the plane: `(r, 0)` turned about the origin.
fn polar(r: f64, a: f64) -> pga2d::Point<(), f64> {
    let origin = pga2d::Point::xy(0.0, 0.0);
    pga2d::Motor::rotation(origin, a) >> pga2d::Point::xy(r, 0.0)
}

/// Rings and spokes of a polar plot of radius up to `rmax`.
fn polar_grid(ax: &Axes, c: &mut Canvas, rmax: f64) {
    let turn = |j: usize, n: usize| core::f64::consts::TAU * j as f64 / n as f64;
    let origin = pga2d::Point::xy(0.0, 0.0);
    for k in 1..=4 {
        let r = rmax * k as f64 / 4.0;
        let ring: Vec<_> = (0..=96).map(|j| polar(r, turn(j, 96))).collect();
        ax.polyline(c, &ring, 0.8, palette::grid().faded(0.8));
    }
    for j in 0..12 {
        let spoke = polar(rmax, turn(j, 12));
        ax.line(c, origin, spoke, 0.8, palette::grid().faded(0.5));
    }
}

fn lattice_scene(c: &mut Canvas, s: f32) {
    let sd = scenes();
    let screen = c.rect();
    let h = screen.height();
    // Lengths in pixels at 960 by 540, scaled with the canvas.
    let unit = c.unit();
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
        let origin = P3::xyz(0.0, 0.0, 0.0);
        panel3(c, rect, orbit(rect, azimuth, 0.4, half), |sc| match i {
            0 => {
                for q in &sites {
                    sc.dot(*q, Marker::Dot, 6.0, palette::grid());
                }
                for (shell, colour) in [(&axial, palette::orange()), (&diagonal, palette::sky())] {
                    for q in shell {
                        sc.seg(origin, *q, 2.0, colour);
                        sc.dot(*q, Marker::Dot, 8.0, colour);
                    }
                }
                sc.dot(origin, Marker::Dot, 11.0, palette::ink());
            }
            1 => {
                let k = sd.conductivity;
                sphere_surface(sc, |d| d.gp((d | k.of(d)).s()), palette::sky(), 0.85);
            }
            _ => {
                let e4 = sd.elasticity;
                sphere_surface(sc, |d| d.gp(e4.fill(d).s()), palette::orange(), 0.85);
            }
        });
        let (head, foot) = (top_middle(rect), bottom_middle(rect));
        c.text(
            title,
            head + down.gp(14.0 * unit),
            13.0 * unit,
            palette::ink(),
            Align::Center,
        );
        c.text(
            sub,
            foot + down.gp(18.0 * unit),
            10.0 * unit,
            palette::grid(),
            Align::Center,
        );
    }
    caption(
        c,
        "SYMMETRY: ONE CUBE GROUP, TWO KINDS OF RESPONSE",
        "24 CUBE ROTATIONS: RANK-2 CONDUCTION ISOTROPIC, RANK-4 ELASTICITY CUBIC",
    );
}

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let u = (t / SCENE).rem_euclid(3.0);
    let s = u.fract();
    match u as usize {
        0 => conduction_scene(c, s),
        1 => flywheel_scene(c, s),
        _ => lattice_scene(c, s),
    }
}

fn main() {
    run(Anim::new("symmetry", 3.0 * SCENE).size(960, 540), draw);
}

#[cfg(test)]
mod tests {
    use super::symmetry::*;

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
        let mut draw = super::draw;
        for t in [0.5, super::SCENE + 0.5, 2.0 * super::SCENE + 0.5] {
            let c = gax_numga_examples::app::frame(
                &gax_numga_examples::Anim::new("t", 1.0).size(320, 180),
                t,
                &mut draw,
            );
            assert!(c.mean().luma() > 0.0);
        }
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws(super::draw, 0.5);
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
            let d = direction(0.3, 0.7);
            let a = g >> quarter.of(d);
            let b = quarter.of(g >> d);
            assert!((a - b).norm() < 1e-12);
        }
    }
}
