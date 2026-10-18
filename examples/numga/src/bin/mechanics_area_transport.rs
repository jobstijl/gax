//! numga's `mechanics/area_transport`: edges, oriented areas and volume carried through a
//! deformation of space, down to its flat collapse (VGA3D).
//!
//! The deformation is a map on vectors, built with the vector left open: a squeeze along `z`
//! and a slide of the horizontal layers along `x`. Its outermorphism moves an oriented patch by
//! moving its two edges, `areas(a ^ b) = F(a) ^ F(b)`, and its determinant scales volumes. The
//! adjugate carries a direction back so that it measures against a patch what the direction
//! measures against the moved patch, `adj(c) & patch = c & areas(patch)`. That is a pairing
//! solve, gax's induced map, with no inverse in it, so it exists at zero volume too, where the
//! normal it returns pairs to nothing with every patch through the lost direction, the patches
//! the area map crushes. Its adjoint through the metric `|`, the cofactor, carries the faces'
//! area normals forward, their lengths the moved areas. gax has neither `adjugate` nor
//! `adjoint` as a method; both are the pairing solves above.
//!
//! A face's oriented patch is the wedge of its two edge directions, wound counter-clockwise
//! seen from outside, and its area normal is the patch's dual: an intrinsic vector, a
//! measurement against patches rather than an edge, so the adjugate and the cofactor are maps
//! on normals. gax's `dual` is the metric-free complement, which in VGA3D takes a bivector to
//! the same vector as numga's (Hodge) dual.
//!
//! The animation squeezes a unit cube flat and back while shearing it: its faces, their area
//! normals carried by the cofactor, and the faces' areas and the volume as bars. At the flat
//! sheet the volume is zero while the faces still have area. From the centre, in yellow, the
//! adjugate's image of the horizontal faces' normal, which at the flat sheet lies along the
//! direction the deformation loses.

use gax_numga_examples::{
    Align, Anim, Axes, Camera, Canvas, Lens, Light, ORIGIN3, Point2, Rect, Scene3, backdrop,
    caption, palette, reach3, run, signal,
};

#[path = "../shared/mechanics_vga3d.rs"]
mod space;

mod area_transport {
    use super::space::adjoint;
    use gax::vga3d::{Bivector, Vector, intrinsic};

    /// A direction, or a position from the cube's centre.
    pub type V = Vector<(), f64>;
    /// An oriented area.
    pub type B = Bivector<(), f64>;
    /// An area normal: the dual of a patch, intrinsic.
    pub type N = intrinsic::Vector<(), f64>;
    /// A map on directions, `Vector <- Vector`.
    pub type Linear = Vector<(Vector,), f64>;
    /// A map on area normals.
    pub type NormalMap = intrinsic::Vector<(intrinsic::Vector,), f64>;
    /// A map on oriented areas, `Bivector <- Bivector`.
    pub type AreaMap = Bivector<(Bivector,), f64>;

    pub const X: V = V::new(1.0, 0.0, 0.0);
    pub const Y: V = V::new(0.0, 1.0, 0.0);
    pub const Z: V = V::new(0.0, 0.0, 1.0);

    /// The cube's side.
    pub const SIDE: f64 = 1.0;
    /// The slide along `x` at the flat sheet, per unit of height.
    pub const MAX_SHEAR: f64 = 0.7;

    /// A squeeze along `z` to `thickness` while the horizontal layers slide along `x` by
    /// `shear` times their height: the identity plus two dyads.
    pub fn deformation(thickness: f64, shear: f64) -> Linear {
        let height = Z | Vector::slot();
        Vector::slot() + (Z * height) * (thickness - 1.0) + (X * height) * shear
    }

    /// The adjugate of an area map, from its defining pairing: `adj(c) & patch` equals
    /// `c & areas(patch)` for every normal `c` and patch.
    pub fn adjugate(areas: AreaMap) -> NormalMap {
        let pairing = N::slot() & Bivector::slot();
        pairing.solve_pairing(N::slot() & areas)
    }

    /// A deformation with its areas carried forward and its measurements carried back.
    #[derive(Clone, Copy, Debug)]
    pub struct Transport {
        pub deformation: Linear,
        /// The outermorphism on patches.
        pub areas: AreaMap,
        /// Pulls a volume measurement back through the area map.
        pub adjugate: NormalMap,
        /// Pushes area normals forward.
        pub cofactor: NormalMap,
        /// The determinant: the factor on volumes, negative when the deformation mirrors.
        pub volume: f64,
    }

    pub fn transport(deformation: Linear) -> Transport {
        let areas = deformation.outermorphism::<Bivector>();
        let adjugate = adjugate(areas);
        Transport {
            deformation,
            areas,
            adjugate,
            cofactor: adjoint(adjugate),
            volume: deformation.det(),
        }
    }

    /// A surface: its corners, its faces' centres, their oriented patches and their area
    /// normals (outward, as long as the face's area).
    #[derive(Clone, Debug)]
    pub struct Surface {
        pub vertices: Vec<V>,
        pub centres: Vec<V>,
        pub patches: Vec<B>,
        pub area_normals: Vec<N>,
    }

    /// A cube centred at the origin. Its corners in numga's order (corner `i` has the bits of
    /// `i` as its signs along `x`, `y`, `z`, `x` the highest), its faces facing `+x`, `+y`, `+z`,
    /// `-x`, `-y`, `-z`, each patch the wedge of its edges, as large as the face, and each area
    /// normal the patch's dual.
    pub fn cube(side: f64) -> Surface {
        let half = side / 2.0;
        let sign = |i: usize, bit: usize| if i >> bit & 1 == 1 { half } else { -half };
        let vertices = (0..8)
            .map(|i| V::new(sign(i, 2), sign(i, 1), sign(i, 0)))
            .collect();
        let outward = [X, Y, Z, -X, -Y, -Z];
        // The faces at +x, +y and +z; the opposite faces are wound the other way.
        let facing = [Y ^ Z, Z ^ X, X ^ Y];
        let patches: Vec<B> = (facing.iter().copied())
            .chain(facing.iter().map(|f| -*f))
            .map(|f| f * (side * side))
            .collect();
        Surface {
            vertices,
            centres: outward.iter().map(|n| *n * half).collect(),
            area_normals: patches.iter().map(|p| p.dual()).collect(),
            patches,
        }
    }

    /// The surface moved: its points by the deformation, its patches by the area map, its
    /// area normals by the cofactor.
    pub fn carried(surface: &Surface, t: &Transport) -> Surface {
        let on = |map: Linear, vs: &[V]| vs.iter().map(|v| map.of(*v)).collect();
        Surface {
            vertices: on(t.deformation, &surface.vertices),
            centres: on(t.deformation, &surface.centres),
            patches: surface.patches.iter().map(|p| t.areas.of(*p)).collect(),
            area_normals: (surface.area_normals.iter())
                .map(|n| t.cofactor.of(*n))
                .collect(),
        }
    }

    /// The cube through a collapse and recovery: at `cosine` 1 the solid, at -1 the flat sheet,
    /// its thickness `(1 + cosine) / 2` and its slide growing as it thins.
    pub fn collapse(cosine: f64) -> (Transport, Surface) {
        let thickness = (1.0 + cosine) / 2.0;
        let t = transport(deformation(thickness, MAX_SHEAR * (1.0 - thickness)));
        let surface = carried(&cube(SIDE), &t);
        (t, surface)
    }
}

use area_transport::*;

const SECONDS: f32 = 8.0;
/// numga's faces, as corners in order around each, and their colours by axis (`x`, `y`, `z`).
const FACES: [[usize; 4]; 6] = [
    [4, 5, 7, 6],
    [2, 3, 7, 6],
    [1, 3, 7, 5],
    [0, 2, 3, 1],
    [0, 1, 5, 4],
    [0, 4, 6, 2],
];
/// The drawn length of an area normal per unit of area.
const ARROW_SCALE: f64 = 0.36;
/// The drawn length of the adjugate's image of the vertical, per unit.
const ADJUGATE_SCALE: f64 = 0.45;
/// The bars' half width, in bar spacings.
const BAR: f32 = 0.29;
/// numga's view of the cube: its azimuth and elevation, in degrees.
const AZIMUTH: f32 = -57.0;
const ELEVATION: f32 = 23.0;
/// The bars' labels.
const MEASURES: [&str; 4] = ["YZ", "ZX", "XY", "VOLUME"];

/// The colour of the faces facing along axis `k` (numga's blue, teal and orange), and of the
/// volume.
fn colour(k: usize) -> Light {
    [
        palette::blue(),
        palette::green(),
        palette::orange(),
        palette::ink(),
    ][k]
}

/// The deformed cube: its faces lit and translucent, its edges, its area normals, and the
/// adjugate's image of the vertical.
fn draw_cube(c: &mut Canvas, rect: Rect, transport: &Transport, surface: &Surface) {
    let (azimuth, elevation) = (AZIMUTH.to_radians(), ELEVATION.to_radians());
    let cam = Camera::orbit(
        rect,
        ORIGIN3,
        4.4,
        azimuth,
        elevation,
        Lens::Perspective(0.42),
    );
    let mut s = Scene3::new(cam);
    let corner = |i: usize| reach3(surface.vertices[i]);
    for (f, face) in FACES.iter().enumerate() {
        let [a, b, d, e] = face.map(corner);
        let lit = s.lit(a, b, d, colour(f % 3).faded(0.35));
        s.quad(a, b, d, e, lit, 0.3);
        for (p, q) in [(a, b), (b, d), (d, e), (e, a)] {
            s.seg(p, q, 1.2, palette::ink().faded(0.5));
        }
    }
    for (f, (centre, normal)) in surface
        .centres
        .iter()
        .zip(&surface.area_normals)
        .enumerate()
    {
        s.arrow(
            reach3(*centre),
            *normal * ARROW_SCALE,
            2.0,
            8.0,
            colour(f % 3),
        );
    }
    let lost = transport.adjugate.of((X ^ Y).dual()) * ADJUGATE_SCALE;
    s.arrow(ORIGIN3, lost, 2.0, 8.0, palette::yellow());
    s.draw(c);
}

/// The three faces' areas and the volume as bars, with their values, over a baseline and the
/// unit cube's level.
fn draw_measures(c: &mut Canvas, rect: Rect, surface: &Surface, volume: f64) {
    let (left, right) = (-0.6, 3.6);
    let ax = Axes::new(rect, [left, right], [0.0, 1.35]);
    let size = (rect.height() / 26.0).clamp(3.0, 13.0);
    let title = rect.top_middle() + Point2::direction(0.0, -size);
    let heading = "AREAS AND VOLUME (CUBE: 1)";
    c.text(heading, title, size * 1.1, palette::ink(), Align::Center);
    let unit = [Point2::xy(left, 1.0), Point2::xy(right, 1.0)];
    ax.dashed(c, &unit, 1.0, 4.0, palette::ink().faded(0.4));
    let mut values: Vec<f64> = surface.area_normals[..3].iter().map(|n| n.norm()).collect();
    values.push(volume);
    for (k, (value, label)) in values.iter().zip(MEASURES).enumerate() {
        let (x, top) = (k as f32, *value as f32);
        let corners =
            [(-BAR, 0.0), (BAR, 0.0), (BAR, top), (-BAR, top)].map(|(dx, y)| Point2::xy(x + dx, y));
        ax.fill(c, &corners, colour(k).faded(0.25), 0.8);
        ax.polyline(c, &[corners[3], corners[2]], 2.5, colour(k));
        let above = ax.px(Point2::xy(x, top)) + Point2::direction(0.0, -0.6 * size);
        c.text(
            &format!("{value:.2}"),
            above,
            size,
            colour(k),
            Align::Center,
        );
        let below = ax.px(Point2::xy(x, 0.0)) + Point2::direction(0.0, 1.8 * size);
        c.text(label, below, size * 0.9, palette::ink(), Align::Center);
    }
    let baseline = [Point2::xy(left, 0.0), Point2::xy(right, 0.0)];
    ax.polyline(c, &baseline, 1.2, palette::grid());
}

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let screen = c.rect();
    let (w, h) = (screen.width(), screen.height());
    let phase = core::f64::consts::TAU * f64::from(t / SECONDS);
    let (transport, surface) = collapse(signal::phasor(phase).e20());
    draw_cube(
        c,
        Rect::new(0.0, 0.08 * h, 0.58 * w, h),
        &transport,
        &surface,
    );
    let bars = Rect::new(0.58 * w, 0.0, w, h).inset(0.07 * w, 0.2 * h, 0.04 * w, 0.14 * h);
    draw_measures(c, bars, &surface, transport.volume);
    caption(
        c,
        "AREA TRANSPORT: A CUBE SQUEEZED FLAT",
        "PATCHES BY THE OUTERMORPHISM, NORMALS BY THE COFACTOR, VOLUME BY THE DETERMINANT (VGA3D)",
    );
}

fn main() {
    run(Anim::new("area_transport", SECONDS).size(960, 540), draw);
}

#[cfg(test)]
mod tests {
    use super::area_transport::*;
    use gax::vga3d::{Bivector, Vector};

    /// Two values within `tolerance`, coefficient by coefficient.
    fn close<const N: usize>(a: [f64; N], b: [f64; N], tolerance: f64) -> bool {
        a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tolerance)
    }

    /// numga's scene checks, at its three cases (solid, compressed and sheared, flat sheet):
    /// the carried area normals are the carried patches' duals, and the deformation and its area
    /// map scale every volume by the determinant, at the flat sheet too.
    #[test]
    fn scene_checks() {
        for (thickness, shear) in [(1.0, 0.0), (0.35, 0.45), (0.0, MAX_SHEAR)] {
            let t = transport(deformation(thickness, shear));
            let surface = carried(&cube(SIDE), &t);
            for (n, p) in surface.area_normals.iter().zip(&surface.patches) {
                assert!(close(n.c, p.dual().c, 1e-12), "{n:?} {p:?}");
            }
            let volumes = t.deformation.of(Vector::slot()) ^ t.areas.of(Bivector::slot());
            let scaled = (Vector::slot() ^ Bivector::slot()) * t.volume;
            let error = (volumes - scaled).c;
            assert!(error.iter().flatten().flatten().all(|v| v.abs() < 1e-12));
        }
    }

    /// numga's `test_transported_faces_and_volume_agree_with_the_deformed_edges`: invertible,
    /// collapsed and mirroring deformations, turned.
    #[test]
    fn transported_faces_and_volume_agree_with_the_deformed_edges() {
        let turn = Bivector::<(), f64>::new(0.17, 0.0, 0.31).exp();
        for (thickness, shear) in [(1.0, 0.0), (0.3, 0.5), (0.0, 0.7), (-0.4, 0.2)] {
            let t = transport(turn >> deformation(thickness, shear));
            let reference = cube(1.0);
            let surface = carried(&reference, &t);
            let v = &surface.vertices;
            let [first, second, third] = [4, 2, 1].map(|i| v[i] - v[0]);
            let patches = [second ^ third, third ^ first, first ^ second];
            for (edges, moved) in patches.iter().zip(&surface.patches) {
                assert!(close(edges.c, moved.c, 1e-12), "{edges:?} {moved:?}");
            }
            let volume = (first ^ second ^ third).e123();
            assert!((volume - t.volume).abs() < 1e-12);
            for (patch, moved) in reference.patches.iter().zip(&surface.patches) {
                let normal = t.cofactor.of(patch.dual());
                assert!(close(normal.c, moved.dual().c, 1e-12));
            }
            assert!((t.volume - thickness).abs() < 1e-12);
        }
    }

    /// numga's `test_the_adjugate_identifies_the_lost_direction_at_collapse`.
    #[test]
    fn the_adjugate_identifies_the_lost_direction_at_collapse() {
        let flat = transport(deformation(0.0, 0.7));
        let lost = flat.adjugate.of((X ^ Y).dual());
        let expected = ((X ^ Y) - (Y ^ Z) * 0.7).dual();
        assert!(close(lost.c, expected.c, 1e-12), "{lost:?}");
        // A patch through the lost direction: the adjugate's image pairs to nothing with it,
        // and the area map crushes it.
        let crushed = (Z - X * 0.7) ^ Y;
        assert!((lost & crushed).s().abs() < 1e-12);
        assert!(close(flat.areas.of(crushed).c, [0.0; 3], 1e-12));
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws(super::draw, 0.5);
    }
}
