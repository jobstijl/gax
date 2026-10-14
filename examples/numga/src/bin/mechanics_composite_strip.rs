//! numga's `mechanics/composite_strip`: the stretch and twist of balanced angle-ply strips under a
//! uniform pull, the same fibres and tension with the plies ordered to keep or to cancel the
//! twist (VGA3D).
//!
//! The strip's state holds its axial strain as a scalar and its thickness times twist per length
//! in the `yz` plane: numga's subspace kind `1 yz`, declared here as the part `State = [1, e23]`
//! of a VGA3D written with `gax::algebra!` (gax's own VGA3D has no such kind). The strain at a
//! height through the thickness is a map on directions with the state left open (`Vector <-
//! (Vector, State)`): the stretch along the strip, and the shear the twist brings at that height,
//! opposite above and below the middle. Each ply reads that strain along its fibres, across them
//! and as their shear (forms on the state), and its stresses are the same readings through its
//! moduli. Pairing readings with stresses and summing through the thickness (two Gauss samples a
//! ply integrate the quadratic energy exactly) gives the energy, a form on the state, in which
//! plies at opposite angles above and below the middle couple stretch to twist. The width relaxes
//! until its force vanishes, which takes a rank-one form off the energy, and the form solved
//! against the pull (a pairing solve) gives the state.
//!
//! The twist is read with a minus sign, `-yz · state`, because `yz` squares to -1. This is a small
//! displacement, uniform curvature model of balanced stacks: the interior response, without the
//! boundary layers near the grips.
//!
//! The picture: both strips held at one end under a pull that grows and eases, magnified five
//! times. The stack `-45 -45 45 45` twists as it stretches; `-45 45 45 -45`, the same plies
//! mirrored about the middle, only stretches. The lines on each strip are its top ply's fibres.

gax::algebra! {
    algebra vga3d "Euclidean space, R(3,0,0), with the composite strip's state `1 + yz`.";
    basis e1 = 1, e2 = 1, e3 = 1;
    kind Scalar = [1];
    versor Vector = [e1, e2, e3];
    kind Bivector = [e23, e31, e12];
    kind Pseudoscalar = [e123];
    part State = [1, e23];
    versor Rotor = [1, e23, e31, e12];
    versor Odd = [e1, e2, e3, e123];
}

use gax::{Of, Slots};
use gax_numga_examples::scene3::panel3;
use gax_numga_examples::signal::haversine;
use gax_numga_examples::{
    Align, Anim, Camera, Canvas, Lens, Light, Point2, Point3, Rect, Scene3, backdrop, caption,
    contour, palette, run,
};
use std::ops::{Add, BitOr, Mul};
use std::sync::OnceLock;

mod strip {
    use super::vga3d::{Bivector, Rotor, Scalar, State, Vector};
    use super::*;

    pub type V = Vector<(), f64>;
    /// Axial strain, and thickness times twist per length.
    pub type St = State<(), f64>;
    /// A linear form on the state.
    pub type Measurement = Scalar<(State,), f64>;
    /// A bilinear form on the state.
    pub type Energy = Scalar<(State, State), f64>;
    /// A strain at a height through the thickness, with the state left open.
    pub type Strain = Vector<(Vector, State), f64>;
    /// A rotation: a ply's fibre angle, or a cross-section's twist.
    pub type Turn = gax::Unit<Rotor<(), f64>>;

    pub const X: V = Vector::new(1.0, 0.0, 0.0);
    pub const Y: V = Vector::new(0.0, 1.0, 0.0);
    pub const Z: V = Vector::new(0.0, 0.0, 1.0);
    /// The planes of the strip and of its cross-sections.
    pub const XY: Bivector<(), f64> = Bivector::new(0.0, 0.0, 1.0);
    pub const YZ: Bivector<(), f64> = Bivector::new(1.0, 0.0, 0.0);
    /// Gauss's two-point nodes on `[-1, 1]`, `±1/√3`.
    const GAUSS_NODE: f64 = 0.577_350_269_189_625_8;

    /// The layups, bottom to top, in degrees.
    pub const ANGLES: [[f64; 4]; 2] = [[-45.0, -45.0, 45.0, 45.0], [-45.0, 45.0, 45.0, -45.0]];
    /// The plies' moduli in MPa, and their Poisson ratio.
    pub const MATERIAL: Material = Material {
        longitudinal: 130000.0,
        transverse: 10000.0,
        poisson: 0.3,
        shear: 5000.0,
    };
    /// The strip's thickness, length and width in mm, and the pull in N per mm of width.
    pub const THICKNESS: f64 = 1.0;
    pub const LENGTH: f64 = 100.0;
    pub const WIDTH: f64 = 20.0;
    pub const TENSION: f64 = 30.0;

    /// The axial strain of a state: its scalar part.
    pub fn extension() -> Measurement {
        Scalar::new(1.0).scalar_product(State::slot())
    }

    /// Thickness times twist per length: the `yz` part, read against `-yz` because `yz` squares
    /// to -1.
    pub fn twist() -> Measurement {
        (-YZ).scalar_product(State::slot())
    }

    /// The strain at `height` through the thickness (along `z`): the stretch along the strip,
    /// and the shear the twist brings there, opposite above and below the middle. Twist tilts the
    /// normal both ways in the plane; dividing by the thickness gives angle per length.
    pub fn strain(height: V, thickness: f64) -> Strain {
        let stretch = X * (X | Vector::slot()) * extension();
        let shear = (X * (Y | Vector::slot()) + Y * (X | Vector::slot())) * twist();
        stretch + shear * (-(height | Z).s() / thickness)
    }

    /// A strain's readings along a ply's fibres, across them and as their shear, with any slots
    /// of the strain left open.
    pub fn read<M, S: Slots>(strain: M, fibre: V) -> [Scalar<S, f64>; 3]
    where
        M: Copy + Of<V, Output = Vector<S, f64>>,
        V: BitOr<Vector<S, f64>, Output = Scalar<S, f64>>,
        Scalar<S, f64>: Mul<f64, Output = Scalar<S, f64>>,
    {
        let across = XY | fibre;
        [
            fibre | strain.of(fibre),
            across | strain.of(across),
            (fibre | strain.of(across)) * 2.0,
        ]
    }

    /// A ply's moduli along and across its fibres and in shear, and its Poisson ratio.
    #[derive(Clone, Copy, Debug)]
    pub struct Material {
        pub longitudinal: f64,
        pub transverse: f64,
        pub poisson: f64,
        pub shear: f64,
    }

    impl Material {
        /// A ply's stresses along and across its fibres and in shear, from its strain read the
        /// same ways, of any arity: the normal stresses coupled by Poisson contraction, shear on
        /// its own.
        pub fn stress<R>(self, [along, across, sliding]: [R; 3]) -> [R; 3]
        where
            R: Copy + Add<Output = R> + Mul<f64, Output = R>,
        {
            let Material {
                longitudinal,
                transverse,
                poisson,
                shear,
            } = self;
            let denominator = 1.0 - poisson * poisson * transverse / longitudinal;
            [
                (along * longitudinal + across * (poisson * transverse)) * (1.0 / denominator),
                (along * poisson + across) * (transverse / denominator),
                sliding * shear,
            ]
        }
    }

    /// A ply: its fibre direction, and the heights of its two Gauss samples with their weight.
    #[derive(Clone, Copy, Debug)]
    pub struct Ply {
        pub fibre: V,
        pub heights: [V; 2],
        pub weight: f64,
    }

    /// The plies of a stack of `thickness`, at `angles` (degrees, bottom to top) from the strip's
    /// axis towards `y`.
    pub fn layup(angles: &[f64], thickness: f64) -> Vec<Ply> {
        let ply = thickness / angles.len() as f64;
        angles
            .iter()
            .enumerate()
            .map(|(k, angle)| {
                let centre = (k as f64 + 0.5) * ply - thickness / 2.0;
                let turn: Turn = (XY * (-angle.to_radians() / 2.0)).exp();
                Ply {
                    fibre: turn >> X,
                    heights: [-GAUSS_NODE, GAUSS_NODE].map(|node| Z * (centre + node * ply / 2.0)),
                    weight: ply / 2.0,
                }
            })
            .collect()
    }

    /// The axial strain and twist, and the relaxed width strain, under `tension` (force per
    /// width).
    pub fn response(plies: &[Ply], material: Material, thickness: f64, tension: f64) -> (St, f64) {
        // A unit width strain, read by each ply as the state's strains are.
        let widening = Y * (Y | Vector::slot());
        let mut energy = Energy::zero();
        let mut coupling = Measurement::zero();
        let mut width_stiffness = 0.0;
        for ply in plies {
            let width_readings = read(widening, ply.fibre);
            let width_stresses = material.stress(width_readings);
            for height in ply.heights {
                let readings = read(strain(height, thickness), ply.fibre);
                let stresses = material.stress(readings);
                for k in 0..3 {
                    // Half `energy(state, state)` is the stored energy per area; with the width
                    // held, `coupling(state)` is the force across per length.
                    energy += readings[k] * stresses[k] * ply.weight;
                    coupling += stresses[k] * (width_readings[k].s() * ply.weight);
                    width_stiffness += (width_readings[k] * width_stresses[k]).s() * ply.weight;
                }
            }
        }
        // Free edges let the width contract until its force vanishes.
        let relaxed = energy - coupling * coupling * (1.0 / width_stiffness);
        let state: St = relaxed.solve(extension() * tension);
        (state, -coupling.of(state).s() / width_stiffness)
    }

    /// Both layups' response to the pull.
    pub fn strips() -> [(St, f64); 2] {
        ANGLES.map(|angles| response(&layup(&angles, THICKNESS), MATERIAL, THICKNESS, TENSION))
    }

    /// The turn of the cross-section `along` the strip, by the twist accumulated from the held
    /// end.
    pub fn section_turn(along: f64, state: St, thickness: f64) -> Turn {
        (YZ * (-along * twist().of(state).s() / thickness / 2.0)).exp()
    }

    /// A point of the strip deformed: the uniform stretch and width contraction, with the
    /// cross-section turned about the strip's axis. A point off the midsurface turns with its
    /// cross-section.
    pub fn deform(point: V, state: St, width_strain: f64, thickness: f64) -> V {
        let [along, across, height] = [X, Y, Z].map(|e| (point | e).s());
        let section = Y * (across * (1.0 + width_strain)) + Z * height;
        X * (along * (1.0 + extension().of(state).s()))
            + (section_turn(along, state, thickness) >> section)
    }
}

use strip::*;
use vga3d::Vector;

/// One loop of the pull: numga's 60 frames at 60 ms.
const SECONDS: f32 = 60.0 * 0.06;
/// The response is magnified for display.
const MAGNIFICATION: f64 = 5.0;
/// Mm between drawn fibre lines, across the fibres, and between the strips' centres.
const FIBRE_SPACING: f64 = 5.0;
const SEPARATION: f64 = 1.6 * WIDTH;
/// The drawn mesh: cells along the length and across the width.
const CELLS: [usize; 2] = [24, 6];
/// The view: the eye's azimuth from the strips' axis and its elevation, in degrees.
const AZIMUTH: f32 = -164.0;
const ELEVATION: f32 = 22.0;
/// The view's half height, in mm.
const VIEW_HALF_HEIGHT: f32 = 32.0;
/// The samples per side of the flat strip where the fibre lines are found.
const FIBRE_SAMPLES: usize = 61;
/// Mm of arrow per unit of pull.
const PULL_ARROW: f64 = 16.0;

fn solved() -> &'static [(St, f64); 2] {
    static S: OnceLock<[(St, f64); 2]> = OnceLock::new();
    S.get_or_init(strips)
}

/// The pull at time `t`, from nothing to full and back once a loop.
fn load(t: f32) -> f64 {
    let angle = std::f64::consts::TAU * f64::from(t.rem_euclid(SECONDS) / SECONDS);
    haversine(angle)
}

/// A point of the strip for drawing.
fn spot(v: V) -> Point3 {
    Point3::xyz(v.e1() as f32, v.e2() as f32, v.e3() as f32)
}

/// A displacement for drawing.
fn heading(v: V) -> Point3 {
    Point3::direction(v.e1() as f32, v.e2() as f32, v.e3() as f32)
}

/// One strip as drawn: its state (scaled by the pull and magnified), its width strain, and where
/// its centreline sits across.
#[derive(Clone, Copy)]
struct Shown {
    state: St,
    width_strain: f64,
    placement: V,
}

impl Shown {
    /// The deformed point at `along` and `across` of the flat strip (mm), at `height`.
    fn at(self, along: f64, across: f64, height: f64) -> V {
        let flat = X * along + Y * across + Z * height;
        deform(flat, self.state, self.width_strain, THICKNESS) + self.placement
    }

    /// The flat strip's point a fraction `u` along and `v` across.
    fn grid(self, u: f64, v: f64) -> V {
        self.at(u * LENGTH, (v - 0.5) * WIDTH, 0.0)
    }

    /// The top face's normal at `along`: the cross-section's turn of `z`.
    fn up(self, along: f64) -> V {
        section_turn(along, self.state, THICKNESS) >> Z
    }
}

fn face_light() -> Light {
    palette::blue().faded(0.32)
}

fn underside_light() -> Light {
    palette::grid()
}

fn fibre_light() -> Light {
    palette::orange()
}

/// The strip's surface, lit and two-sided: the top face blue where it is towards the eye, the
/// underside dim.
fn surface(s: &mut Scene3, strip: Shown, eye: V) {
    let [nu, nv] = CELLS;
    let p = |i: usize, j: usize| spot(strip.grid(i as f64 / nu as f64, j as f64 / nv as f64));
    for i in 0..nu {
        let along = (i as f64 + 0.5) / nu as f64 * LENGTH;
        let colour = if (strip.up(along) | eye).s() > 0.0 {
            face_light()
        } else {
            underside_light()
        };
        for j in 0..nv {
            let (a, b, c, d) = (p(i, j), p(i + 1, j), p(i + 1, j + 1), p(i, j + 1));
            let lit = s.lit(a, b, c, colour);
            s.quad(a, b, c, d, lit, 0.9);
        }
    }
}

/// The flat strip's outline at rest, the grip at the held end, and the pull at the free end.
fn rest_grip_and_pull(s: &mut Scene3, strip: Shown, rest: Shown, pull: f64) {
    let corners = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)];
    let outline: Vec<Point3> = corners
        .iter()
        .map(|&(u, v)| spot(rest.grid(u, v)))
        .collect();
    s.polyline(&outline, 1.0, palette::grid());
    let grip = [-0.6, 0.6].map(|k| spot(rest.at(0.0, k * WIDTH, 0.0)));
    s.seg(grip[0], grip[1], 5.0, palette::ink().faded(0.5));
    let end = spot(strip.at(LENGTH, 0.0, 0.0));
    s.arrow(
        end,
        heading(X * (PULL_ARROW * pull)),
        2.0,
        9.0,
        palette::yellow(),
    );
}

/// The top ply's fibres on a strip: the level lines of the distance across them, `FIBRE_SPACING`
/// apart, found on the flat strip and carried onto the deformed top face, drawn where that face
/// is towards the eye.
fn fibres(c: &mut Canvas, cam: &Camera, strip: Shown, top: V, eye: V) {
    let normal = XY | top;
    let across = |q: Point2| {
        let [along, sideways] = q.to_euclidean().map(f64::from);
        ((X * along + Y * sideways) | normal).s() as f32
    };
    let (x, y) = (
        [0.0, LENGTH as f32],
        [-WIDTH as f32 / 2.0, WIDTH as f32 / 2.0],
    );
    let corners = [
        Point2::xy(x[0], y[0]),
        Point2::xy(x[1], y[0]),
        Point2::xy(x[0], y[1]),
        Point2::xy(x[1], y[1]),
    ];
    let values = corners.map(across);
    let (low, high) = values
        .iter()
        .fold((f32::MAX, f32::MIN), |(l, h), v| (l.min(*v), h.max(*v)));
    let spacing = FIBRE_SPACING as f32;
    let levels = (low / spacing).ceil() as i32..=(high / spacing).floor() as i32;
    let on_top = |q: Point2| {
        let [along, sideways] = q.to_euclidean().map(f64::from);
        let facing = (strip.up(along) | eye).s() > 0.0;
        let p = cam.px(spot(strip.at(along, sideways, THICKNESS / 2.0)));
        p.filter(|_| facing)
    };
    let segments: Vec<[Point2; 2]> = levels
        .flat_map(|k| contour::of_fn(across, x, y, FIBRE_SAMPLES, k as f32 * spacing))
        .filter_map(|[a, b]| Some([on_top(a)?, on_top(b)?]))
        .collect();
    c.stroke(&segments, 1.4, fibre_light());
}

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let screen = c.rect();
    let rect = Rect::new(0.0, screen.height() * 0.1, screen.width(), screen.height());
    let target = Point3::xyz(LENGTH as f32 * 0.4, 0.0, 0.0);
    let (azimuth, elevation) = (AZIMUTH.to_radians(), ELEVATION.to_radians());
    let lens = Lens::Parallel(VIEW_HALF_HEIGHT);
    let cam = Camera::orbit(rect, target, 1000.0, azimuth, elevation, lens);
    let towards = cam.eye().unitized() - target;
    let eye = Vector::new(
        f64::from(towards.e032()),
        f64::from(towards.e013()),
        f64::from(towards.e021()),
    );
    let pull = load(t);
    let shown: [(Shown, Shown); 2] = core::array::from_fn(|k| {
        let (state, width_strain) = solved()[k];
        let placement = Y * (SEPARATION * (k as f64 - 0.5));
        let scale = MAGNIFICATION * pull;
        let strip = Shown {
            state: state * scale,
            width_strain: width_strain * scale,
            placement,
        };
        let rest = Shown {
            state: St::zero(),
            width_strain: 0.0,
            placement,
        };
        (strip, rest)
    });
    let cam = panel3(c, rect, cam, |s| {
        for (strip, rest) in shown {
            rest_grip_and_pull(s, strip, rest, pull);
            surface(s, strip, eye);
        }
    });
    for (k, (strip, _)) in shown.iter().enumerate() {
        let top = layup(&ANGLES[k], THICKNESS).last().expect("a ply").fibre;
        fibres(c, &cam, *strip, top, eye);
        label(c, &cam, *strip, k);
    }
    caption(
        c,
        "COMPOSITE STRIPS UNDER A PULL",
        "STATE 1 + YZ: STRETCH AND TWIST. ENERGY A FORM ON IT, SOLVED AGAINST THE PULL (VGA3D)",
    );
}

/// A strip's layup and its response at full pull, under its held end.
fn label(c: &mut Canvas, cam: &Camera, strip: Shown, k: usize) {
    let Some(grip) = cam.px(spot(strip.at(0.0, 0.0, 0.0))) else {
        return;
    };
    let (state, _) = solved()[k];
    let stack: Vec<String> = ANGLES[k].iter().map(|a| format!("{a:+.0}")).collect();
    let twist_rate = twist().of(state).s() / THICKNESS;
    let twisting = if twist_rate.abs() < 1e-12 {
        "NO TWIST".to_string()
    } else {
        format!("TWIST {:.2} DEG/MM", twist_rate.to_degrees())
    };
    let size = c.rect().height() / 44.0;
    let lines = [
        (stack.join(" "), palette::ink()),
        (
            format!("STRAIN {:.3}%", 100.0 * extension().of(state).s()),
            palette::ink().faded(0.6),
        ),
        (twisting, palette::ink().faded(0.6)),
    ];
    for (i, (text, light)) in lines.iter().enumerate() {
        let line = grip + Point2::direction(0.0, size * (2.5 + 1.5 * i as f32));
        c.text(text, line, size, *light, Align::Center);
    }
}

fn main() {
    run(Anim::new("composite_strip", SECONDS).size(960, 540), draw);
}

#[cfg(test)]
mod tests {
    use super::strip::*;

    /// numga's reference, the classical laminate (ABD) response to a pull along the strip, as
    /// plain arithmetic: each ply's stiffness turned to its fibre angle, integrated through the
    /// thickness into the membrane, coupling and bending blocks, and all six membrane strains and
    /// curvatures solved densely (gax's LU).
    #[allow(clippy::disallowed_methods)] // the textbook's ply transformation, in cosines and sines
    fn laminate_response(angles: &[f64], thickness: f64, m: Material, tension: f64) -> [f64; 6] {
        let denominator = 1.0 - m.poisson * m.poisson * m.transverse / m.longitudinal;
        let (along, across) = (m.longitudinal / denominator, m.transverse / denominator);
        let (coupling, shear) = (m.poisson * across, m.shear);
        let ply = thickness / angles.len() as f64;
        let mut system = [[0.0; 6]; 6];
        for (k, angle) in angles.iter().enumerate() {
            let (c, s) = (angle.to_radians().cos(), angle.to_radians().sin());
            let mixed = c * c * s * s;
            let quartic = c.powi(4) + s.powi(4);
            let q00 =
                along * c.powi(4) + across * s.powi(4) + 2.0 * (coupling + 2.0 * shear) * mixed;
            let q11 =
                along * s.powi(4) + across * c.powi(4) + 2.0 * (coupling + 2.0 * shear) * mixed;
            let q01 = (along + across - 4.0 * shear) * mixed + coupling * quartic;
            let q02 = (along - coupling - 2.0 * shear) * c.powi(3) * s
                - (across - coupling - 2.0 * shear) * c * s.powi(3);
            let q12 = (along - coupling - 2.0 * shear) * c * s.powi(3)
                - (across - coupling - 2.0 * shear) * c.powi(3) * s;
            let q22 = (along + across - 2.0 * coupling - 2.0 * shear) * mixed + shear * quartic;
            let q = [[q00, q01, q02], [q01, q11, q12], [q02, q12, q22]];
            let (low, high) = (
                k as f64 * ply - thickness / 2.0,
                (k + 1) as f64 * ply - thickness / 2.0,
            );
            let moment = |power: i32| (high.powi(power) - low.powi(power)) / f64::from(power);
            // Membrane, coupling and bending blocks: the stiffness times the thickness's first,
            // second and third moments.
            for (i, j) in (0..3).flat_map(|i| (0..3).map(move |j| (i, j))) {
                system[i][j] += q[i][j] * moment(1);
                system[i][j + 3] += q[i][j] * moment(2);
                system[i + 3][j] += q[i][j] * moment(2);
                system[i + 3][j + 3] += q[i][j] * moment(3);
            }
        }
        let pull = [tension, 0.0, 0.0, 0.0, 0.0, 0.0];
        gax::linalg::lu_solve(&gax::linalg::lu(&system), &pull)
    }

    /// numga's `test_tension_response_matches_full_laminate_and_reverses_with_stack`: the strip
    /// agrees with all six free laminate strains and curvatures, for both layups and for both
    /// reversed; reversing a stack reverses its twist, and only the antisymmetric one twists.
    #[test]
    fn tension_response_matches_full_laminate_and_reverses_with_stack() {
        let reversed = ANGLES.map(|mut angles| {
            angles.reverse();
            angles
        });
        let stacks = [ANGLES, reversed].concat();
        let mut extensions = Vec::new();
        let mut twists = Vec::new();
        for angles in &stacks {
            let plies = layup(angles, THICKNESS);
            let (state, width_strain) = response(&plies, MATERIAL, THICKNESS, TENSION);
            let reference = laminate_response(angles, THICKNESS, MATERIAL, TENSION);
            let extension = extension().of(state).s();
            let twist = twist().of(state).s() / THICKNESS;
            assert!(
                (extension - reference[0]).abs() < 1e-12,
                "{extension} {reference:?}"
            );
            assert!((width_strain - reference[1]).abs() < 1e-12);
            assert!(
                (twist + reference[5] / 2.0).abs() < 1e-12,
                "{twist} {reference:?}"
            );
            assert!(
                reference[2..5].iter().all(|r| r.abs() < 1e-12),
                "{reference:?}"
            );
            extensions.push(extension);
            twists.push(twist);
        }
        for k in 0..2 {
            assert!((extensions[k] - extensions[k + 2]).abs() < 1e-12);
            assert!((twists[k] + twists[k + 2]).abs() < 1e-12);
        }
        assert!(twists[1].abs() < 1e-12 && twists[3].abs() < 1e-12);
        assert!(twists[0] > 1e-3);
    }

    /// Numbers from numga's `response`: the twisting layup, and an unbalanced one.
    #[test]
    fn numbers_match_numga() {
        let cases = [
            (
                [-45.0, -45.0, 45.0, 45.0],
                [
                    1.9555190815574216e-3,
                    1.2237826071691863e-3,
                    -1.04448091844258e-3,
                ],
            ),
            (
                [30.0, -60.0, 10.0, 20.0],
                [
                    4.205600107994566e-4,
                    3.6600892163358844e-5,
                    -2.4455162737575e-4,
                ],
            ),
        ];
        for (angles, [axial, twist_rate, width]) in cases {
            let (state, width_strain) =
                response(&layup(&angles, THICKNESS), MATERIAL, THICKNESS, TENSION);
            assert!((extension().of(state).s() - axial).abs() < 1e-15);
            assert!((twist().of(state).s() - twist_rate).abs() < 1e-15);
            assert!((width_strain - width).abs() < 1e-15);
        }
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws(super::draw, 1.0);
    }
}
