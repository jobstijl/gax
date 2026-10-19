//! numga's `orientation/mirror_energy`: three masses turning on the unit sphere beside their
//! mirror image, in VGA3D read as spherical geometry.
//!
//! On the sphere a position is a bivector, the plane through the origin normal to it: an
//! intrinsic one (ADR-051), numga's antivector. A rate of turning is a `Twist` (the extrinsic
//! bivector), and the velocity of a mass is its position's commutator with the rate. Joining a
//! position to its velocity gives the mass's momentum, a `Forque` (the intrinsic vector), and
//! leaving the rate open in that construction gives the inertia, the map `Forque<(Twist,)>` from
//! rates to momenta. Pairing the momentum with the rate, `forque & twist`, is twice the kinetic
//! energy, an extrinsic scalar.
//!
//! The mirror reflects the inertia on both slots, `mirror >> inertia.of(mirror << slot)` (the
//! guide's "Moving a map"), and the rate with it; or the masses, and the inertia is built
//! again. Either way the energy is numga's `1.07`, because gax's `>>` gives the forque and the
//! twist opposite signs under a reflection. The plain sandwich `m x̂ m⁻¹`, which treats every
//! element of a grade alike, gives the momentum the other sign and the energy `-1.07`: the
//! orientation is not in the grades.
//!
//! The animation shows the turn on the left and its mirror image on the right: each mass's
//! path, the triangle between them, the masses (sized by mass) and their velocities, and each
//! side's kinetic energy.

use gax::pga3d;
use gax_numga_examples::scene3::{Scene3, panel3};
use gax_numga_examples::{
    Align, Anim, Camera, Canvas, Lens, Light, Marker, ORIGIN3, Rect, backdrop, caption,
    label_under, on_sphere, palette, reach3, run_with,
};

mod energy {
    use gax::vga3d::{Bivector, Forque, Scalar, Twist, Vector, intrinsic};

    /// A position on the unit sphere: the plane through the origin normal to it.
    pub type Position = intrinsic::Bivector;
    /// The map from rates to momenta.
    pub type Inertia = Forque<(Twist,)>;

    /// The mirror: the plane `x = 0`, by its normal.
    pub fn mirror() -> Vector {
        Vector::new(1.0, 0.0, 0.0)
    }

    /// numga's masses: their positions, `yz zx xy`, and their masses.
    pub fn masses() -> ([Position; 3], [f64; 3]) {
        let positions = [
            Position::new(1.0, 0.0, 0.0),
            Position::new(0.0, 0.8, 0.6),
            Position::new(-0.6, 0.0, -0.8),
        ];
        (positions, [1.0, 1.5, 0.5])
    }

    /// The rate: a unit turn in the `xy` plane.
    pub fn rate() -> Twist {
        Bivector::new(0.0, 0.0, 1.0)
    }

    /// The inertia of point masses: each position joined to its velocity, the rate left open,
    /// weighted by its mass.
    pub fn inertia(positions: &[Position], masses: &[f64]) -> Inertia {
        positions
            .iter()
            .zip(masses)
            .fold(Inertia::zero(), |sum, (p, m)| {
                sum + (*p & p.commutator(Twist::slot())).gp(*m)
            })
    }

    /// Half the pairing of the momentum with the rate.
    pub fn kinetic(inertia: Inertia, rate: Twist) -> Scalar {
        (inertia.of(rate) & rate).gp(0.5)
    }

    /// The energy four ways: as it is; with the inertia and the rate reflected; with the
    /// inertia built again from the reflected masses; and with the momentum and the rate
    /// reflected by the plain sandwich of their grade involutions, which takes no orientation.
    pub fn energies(
        positions: &[Position],
        masses: &[f64],
        rate: Twist,
        mirror: Vector,
    ) -> [f64; 4] {
        let inertia = inertia(positions, masses);
        let reflected_rate = mirror >> rate;
        let reflected = mirror >> inertia.of(mirror << Twist::slot());
        let images: Vec<Position> = positions.iter().map(|p| mirror >> *p).collect();
        let rebuilt = self::inertia(&images, masses);
        let momentum = inertia.of(rate);
        let plain_momentum: Forque =
            (mirror * momentum.involute() * mirror.inverse()).cast::<Vector>();
        let plain_rate: Twist = (mirror * rate.involute() * mirror.inverse()).cast::<Bivector>();
        [
            kinetic(inertia, rate).s(),
            kinetic(reflected, reflected_rate).s(),
            kinetic(rebuilt, reflected_rate).s(),
            (plain_momentum & plain_rate).gp(0.5).s(),
        ]
    }

    /// The masses after turning by the rate for `time`, and their velocities.
    pub fn motion(positions: &[Position], rate: Twist, time: f64) -> Vec<(Position, Position)> {
        let turn = rate.gp(-time / 2.0).exp();
        positions
            .iter()
            .map(|p| {
                let at = turn >> *p;
                (at, at.commutator(rate))
            })
            .collect()
    }
}

use energy::*;

/// numga's animation: 36 frames of 90 milliseconds for one turn.
const SECONDS: f32 = 36.0 * 0.09;
/// The samples of each path.
const PATH: usize = 96;
/// The masses' colours.
const COLOURS: [Light; 3] = [palette::ORANGE, palette::SKY, palette::PURPLE];

/// What the frames draw: each side's paths, and the energies.
struct Scene {
    /// The paths of the masses, `[side][mass][sample]`, the mirror image second.
    paths: [Vec<Vec<Position>>; 2],
    energies: [f64; 4],
}

/// A side's view of a position: the mirror image reflected.
fn side(k: usize, p: Position) -> Position {
    if k == 0 { p } else { mirror() >> p }
}

fn scene() -> Scene {
    let (positions, weights) = masses();
    let path = |k: usize| {
        (0..3)
            .map(|i| {
                (0..=PATH)
                    .map(|s| core::f64::consts::TAU * s as f64 / PATH as f64)
                    .map(|time| side(k, motion(&positions, rate(), time)[i].0))
                    .collect()
            })
            .collect()
    };
    Scene {
        paths: [path(0), path(1)],
        energies: energies(&positions, &weights, rate(), mirror()),
    }
}

/// A position on the sphere as a point of space: the normal of its plane, its dual.
fn place(p: Position) -> pga3d::Point {
    reach3(p.dual()).map_coefs(f64::from)
}

/// The sphere's three great circles through the axes, faint.
fn great_circles(s: &mut Scene3) {
    let tau = core::f64::consts::TAU;
    let at = |k: usize| tau * k as f64 / 96.0;
    let rings: [Vec<pga3d::Point>; 3] = [
        (0..=96)
            .map(|k| reach3(on_sphere(at(k), 0.0)))
            .map(|p| p.map_coefs(f64::from))
            .collect(),
        (0..=96)
            .map(|k| reach3(on_sphere(0.0, at(k))))
            .map(|p| p.map_coefs(f64::from))
            .collect(),
        (0..=96)
            .map(|k| reach3(on_sphere(tau / 4.0, at(k))))
            .map(|p| p.map_coefs(f64::from))
            .collect(),
    ];
    for ring in &rings {
        s.polyline(ring, 0.8, palette::GRID.faded(2.0));
    }
}

/// One side in `rect`: the sphere, the masses' paths, and at `time` the triangle of the
/// masses, the masses and their velocities.
fn side_panel(c: &mut Canvas, rect: Rect, sc: &Scene, k: usize, time: f64) {
    let cam = Camera::orbit(
        rect,
        ORIGIN3,
        9.0,
        (-63f32).to_radians(),
        23f32.to_radians(),
        Lens::Perspective(0.32),
    );
    let (positions, weights) = masses();
    let now: Vec<(Position, Position)> = motion(&positions, rate(), time)
        .into_iter()
        .map(|(p, v)| (side(k, p), side(k, v)))
        .collect();
    panel3(c, rect, cam, |s| {
        great_circles(s);
        for (path, colour) in sc.paths[k].iter().zip(COLOURS) {
            let path: Vec<pga3d::Point> = path.iter().map(|p| place(*p)).collect();
            s.polyline(&path, 1.2, colour.faded(0.16));
        }
        let corners: Vec<pga3d::Point> = now
            .iter()
            .chain(&now[..1])
            .map(|(p, _)| place(*p))
            .collect();
        s.polyline(&corners, 1.2, palette::INK.faded(0.25));
        for (((p, v), m), colour) in now.iter().zip(weights).zip(COLOURS) {
            s.dot(place(*p), Marker::Dot, 8.0 + 6.0 * m as f32, colour);
            s.arrow(place(*p), v.dual().gp(0.45), 2.4, 9.0, colour);
        }
    });
}

fn draw(c: &mut Canvas, sc: &Scene, t: f32) {
    backdrop(c);
    let screen = c.rect();
    let time = core::f64::consts::TAU * f64::from(t / SECONDS);
    let [energy, reflected, rebuilt, plain] = sc.energies;
    let labels = [
        format!("KINETIC ENERGY {energy:.3}"),
        format!("REFLECTED {reflected:.3}, REBUILT {rebuilt:.3}"),
    ];
    for (k, label) in labels.iter().enumerate() {
        let rect = screen.part(0.5 * k as f32, 0.12, 0.5 * (k + 1) as f32, 0.88);
        side_panel(c, rect, sc, k, time);
        label_under(c, rect, label, 20.0, 0.9);
    }
    let note =
        format!("THE PLAIN SANDWICH OF THE GRADE INVOLUTIONS, WITH NO ORIENTATION: {plain:.3}");
    let at = screen.bottom_middle() - gax_numga_examples::Point2::direction(0.0, 14.0);
    c.text(&note, at, 11.0, palette::RED, Align::Center);
    caption(
        c,
        "ENERGY IN A MIRROR",
        "MOMENTUM IS A FORQUE, THE RATE A TWIST: REFLECTED BY >>, THE ENERGY STAYS (VGA3D ON THE SPHERE)",
    );
}

fn main() {
    run_with(
        Anim::new("mirror energy", SECONDS).size(960, 540),
        scene,
        draw,
    );
}

#[cfg(test)]
mod tests {
    use super::energy::*;
    use gax::vga3d::{Bivector, Forque, intrinsic};
    use gax_numga_examples::testing::{close, near};

    /// numga's inertia, momentum and energies: `1.07` as it is, reflected and rebuilt, and
    /// `-1.07` by the plain sandwich, which gives the momentum the wrong sign.
    #[test]
    fn the_reflected_energy_is_numgas() {
        let (positions, weights) = masses();
        let inertia = inertia(&positions, &weights);
        let columns = [
            Forque::new(1.82, 0.0, -0.24),
            Forque::new(0.0, 2.04, -0.72),
            Forque::new(-0.24, -0.72, 2.14),
        ];
        let rates = [
            Bivector::new(1.0, 0.0, 0.0),
            Bivector::new(0.0, 1.0, 0.0),
            rate(),
        ];
        for (rate, column) in rates.into_iter().zip(columns) {
            assert!(
                near(inertia.of(rate), column, 1e-12),
                "{:?}",
                inertia.of(rate)
            );
        }
        let m = mirror();
        let reflected = m >> inertia.of(m << Bivector::slot());
        assert!(near(m >> rate(), Bivector::new(0.0, 0.0, -1.0), 1e-12));
        let momentum = reflected.of(m >> rate());
        assert!(
            near(momentum, Forque::new(-0.24, 0.72, -2.14), 1e-12),
            "{momentum:?}"
        );
        let [energy, reflected, rebuilt, plain] = energies(&positions, &weights, rate(), m);
        for e in [energy, reflected, rebuilt] {
            assert!(close(e, 1.07, 1e-12), "{e}");
        }
        assert!(close(plain, -1.07, 1e-12), "{plain}");
    }

    /// numga's motion test: the energy is half the sum of the masses times their squared
    /// speeds, the masses stay on the unit sphere, and the reflected masses move by the
    /// reflected rate at every sampled instant; with numga's second frame.
    #[test]
    fn the_reflected_masses_move_by_the_reflected_rate() {
        let (positions, weights) = masses();
        let m = mirror();
        for k in 0..36 {
            let time = core::f64::consts::TAU * f64::from(k) / 36.0;
            let now = motion(&positions, rate(), time);
            let speeds: f64 = now
                .iter()
                .zip(weights)
                .map(|((_, v), w)| w * (*v | *v).s().abs())
                .sum();
            assert!(close(speeds / 2.0, 1.07, 1e-12), "{speeds}");
            for (p, v) in now {
                assert!(close((p | p).s().abs(), 1.0, 1e-12));
                let image = m >> p;
                assert!(near(image.commutator(m >> rate()), m >> v, 1e-12));
            }
        }
        let second = motion(&positions, rate(), core::f64::consts::TAU / 36.0);
        let expected = [
            ([0.984808, 0.173648, 0.0], [-0.173648, 0.984808, 0.0]),
            ([-0.138919, 0.787846, 0.6], [-0.787846, -0.138919, 0.0]),
            ([-0.590885, -0.104189, -0.8], [0.104189, -0.590885, 0.0]),
        ];
        for ((p, v), (q, w)) in second.into_iter().zip(expected) {
            assert!(near(p, intrinsic::Bivector::from_coeffs(q), 1e-6), "{p:?}");
            assert!(near(v, intrinsic::Bivector::from_coeffs(w), 1e-6), "{v:?}");
        }
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws_with(super::draw, &super::scene(), 1.0);
    }
}
