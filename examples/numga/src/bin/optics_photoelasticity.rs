//! numga's `optics/photoelasticity`: the stress around a traction-free hole in a stretched
//! plate, read through polarized light.
//!
//! The plate is thin, isotropic and infinite, with uniform stress far from its circular hole. Its
//! in-plane stress is a map from surface normals to tractions, a VGA2D vector map with the
//! normal left open, `Vector<(Vector,)>`; Kirsch's solution resolves the remote stress in the
//! local radial and tangential directions and writes the stress back as three dyads. Its
//! eigenvectors are the principal directions, its eigenvalues the principal stresses.
//!
//! Light's polarization is a point on the Poincaré sphere, a unit Stokes vector of VGA3D: linear
//! polarizations on its equator, at twice their angle in the plate, circular ones at its poles.
//! The state polarized along a direction `d` of the plate is `x` turned by the rotor `d x`,
//! which doubles the angle. The weak stress-optic law makes the stressed plate a rotor of that
//! sphere, about the state polarized along a principal direction, by the retardance that the
//! principal stress difference builds through the thickness: the exponential of the plane square
//! to that state times half the retardance. The plane holds the linear state an eighth of a turn
//! from the principal direction, and the circular state. A polarizer passes the share `(1 + a · s) / 2` of light in the
//! state `s` (Malus's law). Light travels normal to the plate; absorption, reflection and ray
//! bending are neglected.
//!
//! The animation: a circular polariscope (light entering circularly polarized, leaving through
//! the opposite circular analyser) as the load rises from nothing to full tension and back, the
//! fringes running in from the far field and crowding at the hole's sides (left); the principal
//! stress difference over the applied stress, and two crossed linear polariscopes at the same
//! load, whose dark bands (isoclinics) mark where a principal direction lies along the polarizer.

use std::sync::OnceLock;

use gax_numga_examples::points::circle;
use gax_numga_examples::{
    Align, Anim, Axes, Canvas, Light, ORIGIN2, Point2, Rect, backdrop, caption, colormap, palette,
    plot, run, signal::haversine,
};

mod photoelasticity {
    use gax::vga2d::{self, Pseudoscalar};
    use gax::vga3d;

    /// A vector lying in the plate: a position, a surface normal or a traction.
    pub type Planar = vga2d::Vector<(), f64>;
    /// The stress: a map from surface normals to tractions.
    pub type Stress = vga2d::Vector<(vga2d::Vector,), f64>;
    /// A unit Stokes vector on the Poincaré sphere.
    pub type Polarization = vga3d::Vector<(), f64>;
    /// Half a turn of the Poincaré sphere: a rotor's logarithm.
    pub type HalfTurn = vga3d::Bivector<(), f64>;

    /// The plate's `x` axis.
    pub fn x() -> Planar {
        Planar::new(1.0, 0.0)
    }

    /// The plate's `y` axis.
    pub fn y() -> Planar {
        Planar::new(0.0, 1.0)
    }

    /// The circular state, at the sphere's pole.
    pub fn circular() -> Polarization {
        Polarization::new(0.0, 0.0, 1.0)
    }

    /// The dyad `a (b · n)`, the normal `n` left open.
    pub fn dyad(a: Planar, b: Planar) -> Stress {
        a * (b | Planar::slot())
    }

    /// The exterior stress at a position for a circular hole of the given radius under any
    /// uniform symmetric remote stress. At the rim the radial traction vanishes; far away the
    /// stress is the remote one.
    pub fn kirsch(position: Planar, radius: f64, remote: Stress) -> Stress {
        let ratio = radius * radius / position.norm_squared();
        let radial = position.normalized().into_inner();
        let tangent = Pseudoscalar::new(1.0) | radial;
        // The remote traction map resolved in the local radial and tangential directions.
        let mean = remote.trace() / 2.0;
        let anisotropy = (radial | remote.of(radial)).s() - mean;
        let shear = (radial | remote.of(tangent)).s();
        let radial_stress =
            mean * (1.0 - ratio) + anisotropy * (1.0 - 4.0 * ratio + 3.0 * ratio * ratio);
        let tangential_stress = mean * (1.0 + ratio) - anisotropy * (1.0 + 3.0 * ratio * ratio);
        let shear_stress = shear * (1.0 + 2.0 * ratio - 3.0 * ratio * ratio);
        dyad(radial, radial) * radial_stress
            + dyad(tangent, tangent) * tangential_stress
            + (dyad(radial, tangent) + dyad(tangent, radial)) * shear_stress
    }

    /// Light polarized linearly along a unit direction in the plate, on the Poincaré sphere: `x`
    /// turned by the rotor `direction x`, by twice the direction's angle from `x`.
    pub fn polarization(direction: Planar) -> Polarization {
        ((direction * x()) >> x()).into()
    }

    /// The principal stress difference.
    pub fn principal_difference(stress: Stress) -> f64 {
        let (principal, _) = stress.eigh();
        principal[1] - principal[0]
    }

    /// Half the plate's turn of the Poincaré sphere, with the retardance per stress
    /// `stress_phase` (2π times the stress-optic coefficient times the thickness over the
    /// wavelength): about the state polarized along the principal direction of the larger
    /// stress, by the retardance of the principal stress difference. The mean stress delays
    /// both principal directions alike and turns nothing.
    pub fn half_turn(stress: Stress, stress_phase: f64) -> HalfTurn {
        let (principal, directions) = stress.eigh();
        let retardance = stress_phase * (principal[1] - principal[0]);
        // The plane of the turn, square to its axis: the linear state an eighth of a turn from
        // the principal direction, and the circular state.
        let eighth_turn = Pseudoscalar::new(-core::f64::consts::PI / 8.0).exp();
        (polarization(eighth_turn >> directions[1]) ^ circular()) * (retardance / 2.0)
    }

    /// The plate's turn of the Poincaré sphere.
    #[cfg_attr(not(test), allow(dead_code))] // numga's core, which the tests check
    pub fn retarder(stress: Stress, stress_phase: f64) -> gax::Unit<vga3d::Rotor<(), f64>> {
        half_turn(stress, stress_phase).exp()
    }

    /// The share of unit-intensity light in a state that a polarizer passing another lets
    /// through: Malus's law, one plus their inner product, halved.
    pub fn transmitted(state: Polarization, analyser: Polarization) -> f64 {
        (1.0 + (analyser | state).s()) / 2.0
    }

    /// The share of light entering in one state that leaves through the analyser, after a plate
    /// whose half turn of the sphere is `turn`. Scaling the load scales the turn: the principal
    /// directions, and so the axis, stay put.
    pub fn polariscope(turn: HalfTurn, incident: Polarization, analyser: Polarization) -> f64 {
        transmitted(turn.exp() >> incident, analyser)
    }

    // --- the scenes -------------------------------------------------------------------

    /// The hole's radius, mm.
    pub const RADIUS: f64 = 10.0;
    /// Half the width of the window into the infinite plate, mm.
    pub const EXTENT: f64 = 30.0;
    /// Pixels across the window, each way.
    pub const PIXELS: usize = 200;
    /// The remote tension along `x`, MPa.
    pub const TENSION: f64 = 30.0;
    /// The plate's thickness, mm.
    const THICKNESS: f64 = 1.0;
    /// The light's wavelength, mm.
    const WAVELENGTH: f64 = 550e-6;
    /// The stress-optic coefficient, per MPa: an illustrative sensitivity.
    const STRESS_OPTIC: f64 = 5.5e-5;
    /// The retardance per unit of principal stress difference.
    pub const STRESS_PHASE: f64 = core::f64::consts::TAU * STRESS_OPTIC * THICKNESS / WAVELENGTH;

    /// A tension along `x`: averaging the identity and the reflection in `y` (in the mirror with
    /// normal `y`, which keeps `x`) keeps axial traction and cancels transverse traction.
    pub fn uniaxial(tension: f64) -> Stress {
        (Planar::slot() + (y() >> Planar::slot())) * (tension / 2.0)
    }

    /// The remote stress.
    pub fn remote() -> Stress {
        uniaxial(TENSION)
    }

    /// The pixel centres of the window, row by row from the bottom.
    pub fn window() -> Vec<Planar> {
        let centre = |i: usize| (i as f64 + 0.5) * (2.0 * EXTENT / PIXELS as f64) - EXTENT;
        (0..PIXELS * PIXELS)
            .map(|k| Planar::new(centre(k % PIXELS), centre(k / PIXELS)))
            .collect()
    }

    /// The exterior stress at each pixel centre; the pixels inside the hole carry the formula's
    /// values and are covered when drawn.
    pub fn stress_field() -> Vec<Stress> {
        window()
            .into_iter()
            .map(|p| kirsch(p, RADIUS, remote()))
            .collect()
    }

    /// The linear polariscopes' polarizer angles in the plate: along `x`, and an eighth of a
    /// quarter turn from it.
    pub const POLARIZER_ANGLES: [f64; 2] = [0.0, core::f64::consts::FRAC_PI_8];

    /// The states light enters the polariscopes in: the two linear states, and the circular
    /// one. Each analyser passes the opposite state, so without stress every view is dark.
    pub fn incident_states() -> [Polarization; 3] {
        let linear = POLARIZER_ANGLES.map(|angle| polarization(Planar::from_angle(angle)));
        [linear[0], linear[1], circular()]
    }
}

use photoelasticity::*;

/// One loop: from no load to full tension and back.
const SECONDS: f32 = 6.0;

/// What the frames share: the plate's half turns at full load, and the principal stress
/// difference over the applied stress, at each pixel.
struct Plate {
    turns: Vec<HalfTurn>,
    difference: Vec<f64>,
    /// The largest difference over the plate, outside the hole.
    largest: f64,
}

fn plate() -> &'static Plate {
    static PLATE: OnceLock<Plate> = OnceLock::new();
    PLATE.get_or_init(|| {
        let stress = stress_field();
        let difference: Vec<f64> = stress
            .iter()
            .map(|s| principal_difference(*s) / TENSION)
            .collect();
        let outside = window().into_iter().map(|p| p.norm() >= RADIUS);
        let largest = difference
            .iter()
            .zip(outside)
            .filter(|(_, out)| *out)
            .fold(0.0f64, |m, (d, _)| m.max(*d));
        Plate {
            turns: stress.iter().map(|s| half_turn(*s, STRESS_PHASE)).collect(),
            difference,
            largest,
        }
    })
}

/// The load at time `t`: a smooth cycle from nothing to the full tension and back.
fn load(t: f32) -> f64 {
    let turn = f64::from(t / SECONDS) * core::f64::consts::TAU;
    haversine(turn)
}

/// A value of the window's pixel grid at a point of the plate (mm), interpolated between the
/// pixel centres.
fn sample(values: &[f64], p: Point2) -> f64 {
    let [x, y] = p.to_euclidean();
    let index = |v: f32| (f64::from(v) + EXTENT) / (2.0 * EXTENT) * PIXELS as f64 - 0.5;
    plot::bilinear(values, PIXELS, index(x), index(y))
}

/// An image of the plate's window in `rect`, `shade` of each value, with the hole drawn over it
/// and a title above.
fn plate_panel(
    c: &mut Canvas,
    rect: Rect,
    title: &str,
    values: &[f64],
    shade: impl Fn(f64) -> Light + Sync,
) {
    let ax = Axes::equal(rect, ORIGIN2, EXTENT as f32);
    ax.image(c, 1, |p| Some(shade(sample(values, p))));
    let hole = circle(ORIGIN2, RADIUS as f32, 96);
    ax.fill(c, &hole, palette::bottom(), 1.0);
    ax.polyline(c, &hole, 1.0, palette::grid().faded(2.0));
    let r = ax.rect;
    c.polyline(
        &[r.lo, r.top_right(), r.hi, r.bottom_left()],
        1.0,
        palette::grid(),
        true,
    );
    let size = (rect.height() / 14.0).clamp(5.0, 12.0);
    let above = r.lo + Point2::direction(0.0, -size * 0.6);
    c.text(title, above, size, palette::ink(), Align::Left);
}

/// Transmitted light as a light: white-blue on the dark plate, its intensity the share passed.
fn light_through(share: f64) -> Light {
    palette::ink().faded(share.clamp(0.0, 1.0) as f32 * 0.85)
}

/// The colour scale of the stress difference, from none to its largest value, as a bar at the
/// left of `rect` with its ends labelled.
fn colour_bar(c: &mut Canvas, rect: Rect, largest: f64, size: f32) {
    let bar = rect.part(0.08, 0.05, 0.2, 0.95);
    let ax = Axes::new(bar, [0.0, 1.0], [0.0, largest as f32]);
    ax.image(c, 1, |p| {
        let [_, v] = p.to_euclidean();
        Some(colormap::viridis(v / largest as f32))
    });
    let beside = |fy: f32| ax.px(ax.at(1.0, fy)) + Point2::direction(size * 0.6, size * 0.4);
    let ink = palette::ink();
    c.text(
        &format!("{largest:.1}, AT THE RIM"),
        beside(1.0),
        size,
        ink,
        Align::Left,
    );
    c.text(
        "1, FAR AWAY",
        beside(1.0 / largest as f32),
        size,
        ink,
        Align::Left,
    );
    c.text("0", beside(0.0), size, ink, Align::Left);
}

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let p = plate();
    let screen = c.rect();
    let (w, h) = (screen.width(), screen.height());
    let load = load(t);
    let views: Vec<Vec<f64>> = incident_states()
        .into_iter()
        .map(|state| {
            p.turns
                .iter()
                .map(|turn| polariscope(*turn * load, state, -state))
                .collect()
        })
        .collect();
    let top = h * 0.18;
    let side = (h - top - h * 0.06).min(w * 0.5);
    let left = w * 0.03;
    let big = Rect::new(left, top, left + side, top + side);
    plate_panel(
        c,
        big,
        "CIRCULAR POLARISCOPE: FRINGES",
        &views[2],
        light_through,
    );
    let readout = format!("LOAD {:3.0}% OF {TENSION:.0} MPA", 100.0 * load);
    let below = big.bottom_middle() + Point2::direction(0.0, h * 0.045);
    c.text(
        &readout,
        below,
        (h / 40.0).clamp(6.0, 13.0),
        palette::yellow(),
        Align::Center,
    );

    let right = Rect::new(left + side + w * 0.04, top, w * 0.98, top + side);
    let cell = |i: usize, j: usize| {
        right
            .part(
                0.5 * i as f32,
                0.5 * j as f32,
                0.5 * (i + 1) as f32,
                0.5 * (j + 1) as f32,
            )
            .inset(8.0, 24.0, 8.0, 4.0)
    };
    let difference = |v: f64| colormap::viridis((v / p.largest) as f32);
    plate_panel(
        c,
        cell(0, 0),
        "STRESS DIFFERENCE / TENSION",
        &p.difference,
        difference,
    );
    colour_bar(c, cell(1, 0), p.largest, (h / 44.0).clamp(5.0, 12.0));
    let names = ["LINEAR AT 0 DEG", "LINEAR AT 22.5 DEG"];
    for (k, name) in names.into_iter().enumerate() {
        plate_panel(c, cell(k, 1), name, &views[k], light_through);
    }
    caption(
        c,
        "PHOTOELASTICITY: STRESS AROUND A HOLE IN POLARIZED LIGHT",
        "KIRSCH STRESS AS A VGA2D MAP; THE PLATE TURNS THE POINCARE SPHERE (VGA3D ROTORS)",
    );
}

fn main() {
    run(Anim::new("photoelasticity", SECONDS).size(960, 540), draw);
}

#[cfg(test)]
mod tests {
    use super::photoelasticity::*;
    use gax::ApproxEq;
    use gax::vga2d::{Pseudoscalar, Vector};
    use gax_numga_examples::signal::wave;

    const ROUND_OFF: f64 = 1e-10;

    /// A uniform symmetric stress from its normal stresses along `x` and `y` and its shear.
    fn uniform_stress(along_x: f64, along_y: f64, shear: f64) -> Stress {
        dyad(x(), x()) * along_x
            + dyad(y(), y()) * along_y
            + (dyad(x(), y()) + dyad(y(), x())) * shear
    }

    #[test]
    fn free_rim_and_uniaxial_stress_concentration() {
        let (radius, tension, samples) = (1.4, 2.3, 73);
        let remote = uniform_stress(1.7, -0.4, 0.6);
        for k in 0..samples {
            let normal = Planar::from_angle(core::f64::consts::TAU * k as f64 / samples as f64);
            let traction = kirsch(normal * radius, radius, remote).of(normal);
            assert!(traction.max_abs_diff(&Planar::zero()) < ROUND_OFF);
        }
        // The hoop stress at the rim: minus the tension where the load leaves it, three times
        // it at the sides.
        for (end, expected) in [(x(), -tension), (y(), 3.0 * tension)] {
            let tangent = Pseudoscalar::new(1.0) | end;
            let rim = kirsch(end * radius, radius, uniaxial(tension));
            assert!(((tangent | rim.of(tangent)).s() - expected).abs() < ROUND_OFF);
        }
    }

    #[test]
    fn stress_rotates_with_load_and_positions_and_recovers_remote_field() {
        let (radius, far_distance) = (0.8, 1e7 * 0.8);
        let turn = Pseudoscalar::new(-0.37).exp();
        let remote = uniform_stress(2.1, 0.3, -0.5);
        let moved_remote = turn >> remote.of(turn << Vector::slot());
        for p in [
            Planar::new(1.2, 0.4),
            Planar::new(-1.6, 0.7),
            Planar::new(0.3, -2.1),
        ] {
            let stress = kirsch(p, radius, remote);
            let moved = kirsch(turn >> p, radius, moved_remote);
            let rotated = turn >> stress.of(turn << Vector::slot());
            assert!(moved.max_abs_diff(&rotated) < ROUND_OFF);
            // Symmetric: the pairing of two normals through it does not depend on their order.
            let form = Vector::slot() | stress.of(Vector::slot());
            assert!(form.max_abs_diff(&form.swap()) < ROUND_OFF);
            let far = kirsch(p.normalized().into_inner() * far_distance, radius, remote);
            assert!(far.max_abs_diff(&remote) < ROUND_OFF);
        }
    }

    #[test]
    fn stress_has_zero_divergence_away_from_the_hole() {
        let (radius, step) = (0.7, 1e-4 * 0.7);
        let remote = uniform_stress(1.8, -0.2, 0.4);
        for p in [
            Planar::new(1.1, 0.3),
            Planar::new(-1.3, 0.8),
            Planar::new(0.4, -1.7),
        ] {
            let divergence = [x(), y()].into_iter().fold(Planar::zero(), |sum, d| {
                let change =
                    kirsch(p + d * step, radius, remote) - kirsch(p - d * step, radius, remote);
                sum + change.of(d) * (1.0 / (2.0 * step))
            });
            // This tolerance tests the centred derivative's truncation error.
            assert!(divergence.max_abs_diff(&Planar::zero()) < 1e-7);
        }
    }

    #[test]
    fn retarder_turns_the_sphere_and_ignores_mean_stress() {
        let stress_phase = 83.0 * core::f64::consts::PI + 0.41;
        let base = uniform_stress(2.4, -0.8, 0.5);
        let isotropic = Vector::slot() * 3.7;
        let turns = [base, base + isotropic, isotropic].map(|s| retarder(s, stress_phase));
        let states = [
            Polarization::new(1.0, 0.0, 0.0),
            circular(),
            Polarization::new(0.6, -0.48, 0.64),
        ];
        for state in states {
            let exits = turns.map(|turn| turn >> state);
            // States stay on the sphere, the mean stress turns nothing, and alone leaves light
            // as it was.
            assert!(
                exits
                    .iter()
                    .all(|e| (e.norm_squared() - 1.0).abs() < ROUND_OFF)
            );
            assert!(exits[0].max_abs_diff(&exits[1]) < ROUND_OFF);
            assert!(exits[2].max_abs_diff(&state) < ROUND_OFF);
        }
    }

    #[test]
    fn linear_and_circular_dark_ports_match_the_retardance_law() {
        let orientations = 19;
        let retardances = [
            0.0,
            0.01,
            0.83,
            core::f64::consts::PI,
            core::f64::consts::TAU,
            7.1,
            80.0 * core::f64::consts::PI + 0.4,
        ];
        for k in 0..orientations {
            let angle = core::f64::consts::PI * k as f64 / orientations as f64;
            let turn = Pseudoscalar::new(-angle / 2.0).exp();
            let base = uniaxial(1.0);
            let stress = turn >> base.of(turn << Vector::slot());
            for retardance in retardances {
                let plate = retarder(stress * retardance, 1.0);
                let linear = transmitted(plate >> polarization(x()), polarization(y()));
                let circular_power = transmitted(plate >> circular(), -circular());
                let phase_intensity = wave(retardance / 2.0).powi(2);
                // The rotor exponential over forty turns of the sphere is good to about 4e-7.
                assert!((linear - wave(2.0 * angle).powi(2) * phase_intensity).abs() < 1e-4);
                assert!((circular_power - phase_intensity).abs() < 1e-4);
            }
        }
    }

    #[test]
    fn loading_cycle_returns_to_dark_and_matches_the_loaded_plate() {
        let (radius, stress_phase) = (0.8, 13.7);
        let loads = [0.0, 0.5, 1.0, 0.5, 0.0];
        let remote = uniform_stress(2.1, 0.3, -0.5);
        let incident = [polarization(x()), circular()];
        for p in [
            Planar::new(1.2, 0.4),
            Planar::new(-1.6, 0.7),
            Planar::new(0.3, -2.1),
        ] {
            let turn = half_turn(kirsch(p, radius, remote), stress_phase);
            for state in incident {
                let frames = loads.map(|l| polariscope(turn * l, state, -state));
                // Recompute the stress at every remote load, rather than scaling the reference.
                let expected = loads.map(|l| {
                    let plate = retarder(kirsch(p, radius, remote * l), stress_phase);
                    transmitted(plate >> state, -state)
                });
                for (k, f) in frames.iter().enumerate() {
                    assert!((f - expected[k]).abs() < ROUND_OFF);
                    assert!((f - frames[loads.len() - 1 - k]).abs() < ROUND_OFF);
                }
                assert!(frames[0].abs() < ROUND_OFF);
            }
        }
    }

    /// The scene against numga's numbers: the principal stress difference and the three
    /// polariscopes at the pixel centre `(21.15, 0.15)` mm.
    #[test]
    fn scene_matches_numga() {
        let p = Planar::new(170.5 * 0.3 - EXTENT, 100.5 * 0.3 - EXTENT);
        let stress = kirsch(p, RADIUS, remote());
        assert!((principal_difference(stress) - 14.387127181521555).abs() < 1e-9);
        let turn = half_turn(stress, STRESS_PHASE);
        let numga = [5.63624131e-04, 5.04987414e-01, 9.63384315e-01];
        for (state, expected) in incident_states().into_iter().zip(numga) {
            assert!((polariscope(turn, state, -state) - expected).abs() < 1e-8);
        }
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws(super::draw, 0.5);
    }
}
