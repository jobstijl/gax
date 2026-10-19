//! numga's `mechanics/stress`: stress in an elastic cube, the traction on each face, the
//! principal frame and Mohr's circle (VGA3D).
//!
//! A strain is a map on vectors (`Vector <- Vector`): how far each point of the material moves,
//! for where it sits. In an isotropic material the stress is a map of the same kind,
//! `lame * strain.trace() + 2 * shear_modulus * strain`, the first Lamé parameter times the open
//! vector scaled by the strain's trace, plus twice the shear modulus times the strain. Given a
//! face's normal it returns the traction, the force per area across the face, which splits into
//! a part along the normal and a shear along the face.
//!
//! Seen from a turned frame, `turn << stress(turn >> Vector)`, the cube's faces feel other
//! tractions. The stress's eigenvectors are the principal directions: in the frame the rotor
//! `1 + d x` (normalized) turns onto them, every face's normal goes to a multiple of itself and
//! no face is sheared. Turning the frame in the shear plane, the normal and shear traction on
//! one face trace a circle, Mohr's, whose radius is the largest shear any face feels: an eighth
//! of a turn from the principal frame.
//!
//! The animation turns the frame half a turn in the shear plane from the principal frame, as
//! numga's does: left the deformed cube seen from that frame, with the normal traction on each
//! face (cyan), its shear (orange) and the principal directions (green); right Mohr's circles,
//! with the diameter joining the tractions on the faces facing x and y, and those of numga's
//! three fixed frames: the lab, the principal frame and the frame of greatest shear.

use gax_numga_examples::points;
use gax_numga_examples::scene3::panel3;
use gax_numga_examples::{
    Align, Anim, Axes, Camera, Canvas, Lens, Light, Marker, ORIGIN3, Point2, Rect, Scene3,
    backdrop, caption, palette, reach3, run_with,
};

use gax_numga_examples::shared::mechanics_vga3d as space;

mod stress {
    use super::space::turn;
    pub use super::space::{AXES, Map, Turn, V};
    use core::f64::consts::FRAC_PI_4;
    use gax::vga3d::{Bivector, Rotor, Vector};

    /// The plane the shear acts in, from x towards y.
    pub const SHEAR_PLANE: Bivector = Bivector::new(0.0, 0.0, 1.0);

    /// The corners of the cube of unit side about the origin: corner `4 i + 2 j + k` sits at the
    /// `i`-th, `j`-th and `k`-th of `-0.5` and `0.5` along x, y and z.
    pub fn corners() -> [V; 8] {
        let side = |bit: usize| if bit == 0 { -0.5 } else { 0.5 };
        core::array::from_fn(|c| Vector::new(side(c >> 2 & 1), side(c >> 1 & 1), side(c & 1)))
    }

    /// The outward normals of the cube's faces: those facing x, y and z, then their opposites.
    pub fn normals() -> [V; 6] {
        core::array::from_fn(|f| if f < 3 { AXES[f] } else { -AXES[f - 3] })
    }

    /// The stress of an isotropic material under the strain: Lamé's formula.
    pub fn cauchy_stress(strain: Map, lame: f64, shear_modulus: f64) -> Map {
        Vector::slot() * (lame * strain.trace()) + strain * (2.0 * shear_modulus)
    }

    /// The traction on a face of the given normal, split into its part along the normal and
    /// the shear along the face.
    pub fn tractions(stress: Map, face: V) -> (V, V) {
        let total = stress.of(face);
        let along = face * face.dot(total).s();
        (along, total - along)
    }

    /// The principal stresses, from least to greatest, the principal directions, and the turn
    /// taking x onto the first of them: for a shear in the xy plane, the turn in that plane
    /// onto the principal frame.
    pub fn principal_frame(stress: Map) -> ([f64; 3], [V; 3], Turn) {
        let (values, directions) = stress.eigh();
        let one = Rotor::new(1.0, 0.0, 0.0, 0.0);
        (values, directions, (one + directions[0] * AXES[0]).unit())
    }

    /// The centre of Mohr's circle in the shear plane, and its radius, the largest shear: from
    /// the least and the greatest principal stress.
    pub fn mohr_circle(values: [f64; 3]) -> (f64, f64) {
        (
            (values[0] + values[2]) / 2.0,
            (values[2] - values[0]).abs() / 2.0,
        )
    }

    /// The cube seen from one frame: its deformed corners, the centres of its deformed faces,
    /// the normal and shear traction on each face, and the principal directions.
    #[derive(Clone, Copy, Debug)]
    pub struct View {
        pub corners: [V; 8],
        pub centres: [V; 6],
        pub normal: [V; 6],
        pub shear: [V; 6],
        pub principal: [V; 3],
    }

    /// A map as the frame sees it: its input turned into the frame, its output back.
    fn seen_from(frame: Turn, map: Map) -> Map {
        frame << map.of(frame >> Vector::slot())
    }

    /// The cube seen from a frame.
    pub fn view(material: &Material, frame: Turn) -> View {
        let strain = seen_from(frame, material.strain);
        let stress = seen_from(frame, material.stress);
        let faces = normals();
        let split = faces.map(|n| tractions(stress, n));
        View {
            corners: corners().map(|c| c + strain.of(c)),
            centres: faces.map(|n| (n + strain.of(n)) * 0.5),
            normal: split.map(|(along, _)| along),
            shear: split.map(|(_, across)| across),
            principal: material.directions.map(|d| frame << d),
        }
    }

    // --- scenarios

    /// The material: the first Lamé parameter and the shear modulus.
    pub const LAME: f64 = 2.0;
    pub const SHEAR_MODULUS: f64 = 1.0;
    /// The strain: stretches along x, y and z, and a shear in the xy plane.
    pub const STRETCHES: [f64; 3] = [0.05, -0.05, 0.08];
    pub const SHEAR: f64 = 0.30;

    /// The strain, the stress it causes, and the principal stresses, directions and frame.
    #[derive(Clone, Copy, Debug)]
    pub struct Material {
        pub strain: Map,
        pub stress: Map,
        pub values: [f64; 3],
        pub directions: [V; 3],
        pub principal: Turn,
    }

    /// The scene's material.
    pub fn material() -> Material {
        let [x, y, _] = AXES;
        let stretch: Map = AXES
            .iter()
            .zip(STRETCHES)
            .map(|(a, s)| (*a * s) * (*a | Vector::slot()))
            .sum();
        let shear = (x * (y | Vector::slot()) + y * (x | Vector::slot())) * SHEAR;
        let strain = stretch + shear;
        let stress = cauchy_stress(strain, LAME, SHEAR_MODULUS);
        let (values, directions, principal) = principal_frame(stress);
        Material {
            strain,
            stress,
            values,
            directions,
            principal,
        }
    }

    /// The frame turned by `angle` in the shear plane from the principal frame.
    pub fn turned(material: &Material, angle: f64) -> Turn {
        turn(SHEAR_PLANE, angle) * material.principal
    }

    /// numga's three frames: the lab, the principal frame, and the frame an eighth of a turn
    /// on, where the shear is greatest.
    pub fn frames(material: &Material) -> [Turn; 3] {
        let lab = Rotor::new(1.0, 0.0, 0.0, 0.0).unit();
        [lab, material.principal, turned(material, FRAC_PI_4)]
    }

    /// The names of the three frames.
    pub const FRAME_NAMES: [&str; 3] = ["LAB", "PRINCIPAL", "MOST SHEAR"];
}

use stress::*;

/// Half a turn of the frame: numga's 60 frames, at twice their 50 ms.
const SECONDS: f32 = 6.0;
/// The arrows' lengths per unit traction.
const NORMAL_SCALE: f64 = 0.4;
const SHEAR_SCALE: f64 = 0.5;
/// The cube's edges, by the corners' indices.
const EDGES: [[usize; 2]; 12] = [
    [0, 1],
    [1, 3],
    [3, 2],
    [2, 0],
    [4, 5],
    [5, 7],
    [7, 6],
    [6, 4],
    [0, 4],
    [1, 5],
    [2, 6],
    [3, 7],
];
/// The cube's faces, by the corners' indices.
const FACES: [[usize; 4]; 6] = [
    [4, 5, 7, 6],
    [2, 6, 7, 3],
    [1, 3, 7, 5],
    [0, 2, 3, 1],
    [0, 1, 5, 4],
    [0, 4, 6, 2],
];
/// Points around each of Mohr's circles.
const CIRCLE_SAMPLES: usize = 96;
/// The camera: numga's view, 25 degrees round from x and 22 up.
const AZIMUTH: f32 = 25.0;
const ELEVATION: f32 = 22.0;

/// The undeformed cube, faint, and the deformed one, shaded with its edges.
fn cube(s: &mut Scene3, view: &View) {
    let rest = corners();
    for [a, b] in EDGES {
        s.seg(
            reach3(rest[a]),
            reach3(rest[b]),
            1.0,
            palette::GRID.faded(2.0),
        );
    }
    let at = |i: usize| reach3(view.corners[i]);
    for [a, b, c, d] in FACES {
        let light = s.lit(at(a), at(b), at(c), palette::BLUE.faded(0.5));
        s.quad(at(a), at(b), at(c), at(d), light, 0.35);
    }
    for [a, b] in EDGES {
        s.seg(at(a), at(b), 1.6, palette::INK.faded(0.8));
    }
}

/// The tractions on the faces, from their centres, and the principal directions through the
/// cube.
fn tractions_on(s: &mut Scene3, view: &View) {
    for f in 0..6 {
        let centre = reach3(view.centres[f]);
        s.arrow(
            centre,
            view.normal[f] * NORMAL_SCALE,
            2.4,
            10.0,
            palette::SKY,
        );
        s.arrow(
            centre,
            view.shear[f] * SHEAR_SCALE,
            3.0,
            11.0,
            palette::ORANGE,
        );
    }
    for d in view.principal {
        s.seg(reach3(-d), reach3(d), 1.2, palette::GREEN.faded(0.5));
    }
}

/// The cube seen from a frame, in `rect`.
fn draw_cube(c: &mut Canvas, rect: Rect, view: &View) {
    let cam = Camera::orbit(
        rect,
        ORIGIN3,
        4.2,
        AZIMUTH.to_radians(),
        ELEVATION.to_radians(),
        Lens::Perspective(0.62),
    );
    panel3(c, rect, cam, |s| {
        cube(s, view);
        tractions_on(s, view);
    });
}

/// A circle on axes, centred on the normal-stress axis.
fn circle(ax: &Axes, c: &mut Canvas, centre: f64, radius: f64, width: f32, light: Light) {
    let ring = points::circle(gax::pga2d::Point::xy(centre, 0.0), radius, CIRCLE_SAMPLES);
    ax.polyline(c, &ring, width, light);
}

/// The diameter of a view on Mohr's circles: the face facing x, at its normal traction along x
/// and its shear along y, and the face facing y, with the opposite shear.
fn diameter(view: &View) -> [Point2; 2] {
    let shear = view.shear[0].c[1] as f32;
    [
        Point2::xy(view.normal[0].c[0] as f32, shear),
        Point2::xy(view.normal[1].c[1] as f32, -shear),
    ]
}

/// One view's diameter: the face facing x a dot, the face facing y a ring.
fn draw_diameter(ax: &Axes, c: &mut Canvas, view: &View, width: f32, size: f32, light: Light) {
    let [x_face, y_face] = diameter(view);
    ax.line(c, x_face, y_face, width, light);
    ax.scatter(c, &[x_face], Marker::Dot, size, light);
    ax.scatter(c, &[y_face], Marker::Ring, size, light);
}

/// Mohr's circles of the principal stresses, the turning view's diameter and those of the
/// three fixed frames.
fn draw_mohr(c: &mut Canvas, rect: Rect, material: &Material, turning: &View) {
    let v = material.values;
    let (centre, radius) = mohr_circle(v);
    // The circle near the left edge, the legend to its right.
    let half_height = 1.25 * radius;
    let half_width = half_height * f64::from(rect.width() / rect.height());
    let middle = centre - 1.15 * radius + half_width;
    let ax = Axes::equal(rect, Point2::xy(middle as f32, 0.0), half_height as f32);
    ax.frame(
        c,
        "MOHR'S CIRCLES",
        "NORMAL STRESS (ACROSS) AND SHEAR (UP)",
        "",
    );
    let [x0, x1] = ax.x;
    ax.line(
        c,
        Point2::xy(x0, 0.0),
        Point2::xy(x1, 0.0),
        1.0,
        palette::GRID,
    );
    for (low, high, width, light) in [
        (0, 2, 2.0, palette::INK),
        (0, 1, 1.0, palette::INK.faded(0.4)),
        (1, 2, 1.0, palette::INK.faded(0.4)),
    ] {
        let (centre, radius) = mohr_circle([v[low], 0.0, v[high]]);
        circle(&ax, c, centre, radius, width, light);
    }
    let size = rect.height() / 40.0;
    for (k, stress) in v.iter().enumerate() {
        let at = ax.px(Point2::xy(*stress as f32, 0.0)) + Point2::direction(0.0, size * 2.4);
        let name = format!("S{}", k + 1);
        c.text(&name, at, size, palette::GREEN, Align::Center);
    }
    let fixed = frames(material).map(|frame| view(material, frame));
    let lights = [palette::YELLOW, palette::GREEN, palette::RED];
    for (view, light) in fixed.iter().zip(lights) {
        draw_diameter(&ax, c, view, 1.0, 7.0, light.faded(0.6));
    }
    draw_diameter(&ax, c, turning, 2.2, 11.0, palette::PURPLE);
    let mut entries: Vec<(&str, Light)> = FRAME_NAMES.into_iter().zip(lights).collect();
    entries.push(("TURNING", palette::PURPLE));
    ax.legend(c, &entries);
}

fn draw(c: &mut Canvas, material: &Material, t: f32) {
    backdrop(c);
    let angle = core::f64::consts::PI * f64::from((t / SECONDS).rem_euclid(1.0));
    let turning = view(material, turned(material, angle));
    let screen = c.rect();
    let body = screen.part(0.0, 0.12, 1.0, 1.0);
    let left = body.part(0.0, 0.0, 0.42, 1.0);
    draw_cube(c, left, &turning);
    let right = body.part(0.42, 0.0, 1.0, 1.0).inset(64.0, 40.0, 16.0, 56.0);
    draw_mohr(c, right, material, &turning);
    let note = format!("FRAME {:.0} DEGREES FROM PRINCIPAL", angle.to_degrees());
    let at = left.bottom_middle() - Point2::direction(0.0, 16.0);
    c.text(&note, at, 12.0, palette::INK.faded(0.6), Align::Center);
    caption(
        c,
        "STRESS: TRACTIONS ON A CUBE, AND MOHR'S CIRCLE",
        "STRESS = LAME TR(STRAIN) + 2 MU STRAIN, A MAP ON VECTORS SEEN FROM A TURNING FRAME (VGA3D)",
    );
}

fn main() {
    run_with(Anim::new("stress", SECONDS).size(960, 540), material, draw);
}

#[cfg(test)]
mod tests {
    use super::stress::*;
    use core::f64::consts::PI;
    use gax::vga3d::Vector;

    #[test]
    fn isotropic_stress_is_lames_formula() {
        // Entry (i, j) of the stress, x_i | stress(x_j), is lame tr(strain) δ_ij + 2 μ strain_ij.
        let (lame, shear_modulus) = (1.8, 1.1);
        let matrix = [
            [0.02, 0.015, -0.005],
            [0.015, -0.01, 0.008],
            [-0.005, 0.008, 0.025],
        ];
        let mut strain = Map::zero();
        for (i, row) in matrix.iter().enumerate() {
            for (j, m) in row.iter().enumerate() {
                strain += (AXES[i] * *m) * (AXES[j] | Vector::slot());
            }
        }
        let stress = cauchy_stress(strain, lame, shear_modulus);
        let trace = matrix[0][0] + matrix[1][1] + matrix[2][2];
        for i in 0..3 {
            for j in 0..3 {
                let entry = AXES[i].dot(stress.of(AXES[j])).s();
                let delta = f64::from(u8::from(i == j));
                let expected = lame * trace * delta + 2.0 * shear_modulus * matrix[i][j];
                assert!(
                    (entry - expected).abs() <= 1e-12 * expected.abs(),
                    "{entry} {expected}"
                );
            }
        }
    }

    /// In the principal frame no face is sheared, and the normal tractions on the faces facing
    /// x, y and z are the principal stresses, the turned axes their directions.
    #[test]
    fn the_principal_frame_shears_no_face() {
        let material = material();
        let [_, principal, _] = frames(&material).map(|frame| view(&material, frame));
        for shear in principal.shear {
            assert!(shear.norm() <= 1e-14, "{shear:?}");
        }
        let mut normal: Vec<f64> = (0..3)
            .map(|f| principal.normal[f].dot(AXES[f]).s())
            .collect();
        normal.sort_by(f64::total_cmp);
        for (n, value) in normal.iter().zip(material.values) {
            assert!((n - value).abs() <= 1e-12 * value.abs(), "{n} {value}");
        }
        for axis in AXES {
            let turned = material.principal >> axis;
            assert!((material.stress.of(turned) ^ turned).norm() <= 1e-14);
        }
    }

    /// An eighth of a turn on, the face facing x feels the largest shear, the radius of Mohr's
    /// circle.
    #[test]
    fn an_eighth_of_a_turn_on_the_shear_is_greatest() {
        let material = material();
        let [.., greatest] = frames(&material).map(|frame| view(&material, frame));
        let (_, radius) = mohr_circle(material.values);
        let shear = greatest.shear[0].norm();
        assert!((shear - radius).abs() <= 1e-8 * radius, "{shear} {radius}");
    }

    /// The frames of numga's turning view, over half a turn.
    const FRAMES: usize = 60;

    /// The normal and shear traction on the face facing x go around Mohr's circle.
    #[test]
    fn a_turning_frame_goes_around_mohrs_circle() {
        let material = material();
        let (centre, radius) = mohr_circle(material.values);
        for k in 0..FRAMES {
            let angle = PI * k as f64 / FRAMES as f64;
            let v = view(&material, turned(&material, angle));
            let normal = v.normal[0].dot(AXES[0]).s();
            let shear = v.shear[0].dot(AXES[1]).s();
            let squared = (normal - centre).powi(2) + shear * shear;
            assert!((squared - radius * radius).abs() <= 1e-8 * radius * radius);
        }
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws_with(super::draw, &material(), 0.5);
    }
}
