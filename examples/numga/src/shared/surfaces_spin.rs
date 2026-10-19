//! What numga's `surfaces/spin_transformations` shares with `surfaces/arap`: the cube's surface
//! mesh; and with `surfaces/dirac_waves` too: surfaces drawn as numga draws them, with a depth
//! buffer: at each sample of a panel the nearest face and the sample's barycentric weights in it,
//! so that colours and normals are interpolated across the faces and each side of a face can have
//! its own colour.

use crate::mesh::{Mesh, Vector3};
use crate::{Camera, Canvas, Light, Point2, Rect, palette, reach3};
use gax::field::Field;
use gax::{Extensor, vga3d};
use std::collections::BTreeSet;

/// The surface of the cube `[-1, 1]³`, each side a grid of `divisions` by `divisions` squares, two
/// triangles each, counter-clockwise from outside. The vertices are in lexicographic order of
/// their coordinates, as numga's `np.unique` leaves them.
pub fn cube(divisions: usize) -> Mesh {
    // Lattice coordinates: `k` stands for `-1 + 2k / divisions`, and `last` for the side `+1`.
    let last = divisions;
    let mut sides: Vec<Vec<[usize; 3]>> = Vec::new();
    for axis in 0..3 {
        for outward in [true, false] {
            // The point at grid place `(i, j)` of the side: on the side's axis at its sign, `i`
            // along the next axis (mirrored on the negative side), `j` along the one after.
            let side = (0..=last).flat_map(|i| {
                (0..=last).map(move |j| {
                    let mut point = [0; 3];
                    point[axis] = if outward { last } else { 0 };
                    point[(axis + 1) % 3] = if outward { i } else { last - i };
                    point[(axis + 2) % 3] = j;
                    point
                })
            });
            sides.push(side.collect());
        }
    }
    let lattice: Vec<[usize; 3]> = sides
        .iter()
        .flatten()
        .copied()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let index = |point: [usize; 3]| lattice.binary_search(&point).expect("a lattice point");
    let at = |k: usize| -1.0 + 2.0 * k as f64 / divisions as f64;
    let vertices = lattice
        .iter()
        .map(|&[x, y, z]| Vector3::new(at(x), at(y), at(z)))
        .collect();

    // Each square of a side: its corners `a b c d` counter-clockwise, split along `a c`.
    let squares: Vec<[usize; 4]> = (0..divisions)
        .flat_map(|i| {
            (0..divisions).map(move |j| {
                let a = i * (last + 1) + j;
                [a, a + last + 1, a + last + 2, a + 1]
            })
        })
        .collect();
    let mut faces = Vec::new();
    for side in &sides {
        let corner = |k: usize| index(side[k]);
        faces.extend(squares.iter().map(|&[a, b, c, _]| [a, b, c].map(corner)));
        faces.extend(squares.iter().map(|&[a, _, c, d]| [a, c, d].map(corner)));
    }
    Mesh::new(vertices, faces)
}

/// Samples per pixel along each axis, against jagged edges.
const SAMPLES: usize = 2;
/// The share of a surface's colour that facing away from the eye cannot darken, and the
/// strength of the glint where it faces the eye.
const AMBIENT: f32 = 0.3;
const GLINT: f32 = 0.15;

/// The light of the checkerboard's light and dark squares.
const LIGHT_SQUARE: f32 = 0.6;
const DARK_SQUARE: f32 = 0.25;

/// A checkerboard's square: orange on a surface's outside and blue on its inside, light or dark.
pub fn checker_light(dark: bool, outside: bool) -> Light {
    let side = if outside {
        palette::ORANGE
    } else {
        palette::SKY
    };
    side.faded(if dark { DARK_SQUARE } else { LIGHT_SQUARE })
}

/// The nearest face at a sample, the sample's barycentric weights in it, and whether the camera
/// sees the face's outside, where it turns counter-clockwise.
#[derive(Clone, Copy, Debug)]
pub struct Hit {
    pub face: usize,
    pub weights: [f64; 3],
    pub outside: bool,
}

impl Hit {
    /// A field on the mesh's vertices at the sample: its corners' values, weighted.
    pub fn at<V>(&self, mesh: &Mesh, field: &Field<V>) -> V
    where
        V: Extensor<Slots = (), Coef = f64>
            + core::ops::Add<Output = V>
            + core::ops::Mul<f64, Output = V>,
    {
        let [a, b, c] = mesh.faces[self.face].map(|v| field.site(v));
        let [wa, wb, wc] = self.weights;
        a * wa + b * wb + c * wc
    }
}

/// A depth buffer: the nearest hit at each sample of a panel, `SAMPLES` per pixel along each
/// axis.
struct Raster {
    corner: Point2,
    columns: usize,
    rows: usize,
    nearest: Vec<Option<(f32, Hit)>>,
}

impl Raster {
    /// The mesh seen by the camera in the panel.
    fn new(cam: &Camera, rect: Rect, mesh: &Mesh) -> Raster {
        let corner = Point2::xy(rect.lo.e20().floor(), rect.lo.e01().floor());
        let span = |length: f32| (length.ceil() as usize + 1) * SAMPLES;
        let (columns, rows) = (span(rect.width()), span(rect.height()));
        let mut raster = Raster {
            corner,
            columns,
            rows,
            nearest: vec![None; columns * rows],
        };
        let points: Vec<_> = mesh.vertices.iter().map(reach3).collect();
        let pixels: Vec<Option<Point2>> = points.iter().map(|&p| cam.px(p)).collect();
        let depths: Vec<f32> = points.iter().map(|&p| cam.depth(p)).collect();
        for (face, corners) in mesh.faces.iter().enumerate() {
            if let [Some(a), Some(b), Some(c)] = corners.map(|v| pixels[v]) {
                raster.cover(face, [a, b, c], corners.map(|v| depths[v]));
            }
        }
        raster
    }

    /// A canvas point's place in the buffer, in samples along each axis.
    fn place(&self, q: Point2) -> [f32; 2] {
        let offset = q - self.corner;
        [offset.e20(), offset.e01()].map(|t| t * SAMPLES as f32)
    }

    /// The canvas point at the centre of sample `(i, j)`.
    fn sample(&self, i: usize, j: usize) -> Point2 {
        let step = |k: usize| (k as f32 + 0.5) / SAMPLES as f32;
        self.corner + Point2::direction(step(i), step(j))
    }

    /// Mark the samples a face covers where it is the nearest so far. A sample's barycentric
    /// weights are the joins of the sample with the face's other two corners over the join of
    /// all three: signed areas, whose sign says which side of the face the camera sees (its
    /// outside turns clockwise on the canvas, whose `y` points down).
    fn cover(&mut self, face: usize, [a, b, c]: [Point2; 3], depths: [f32; 3]) {
        let area = (a & b & c).s();
        if area == 0.0 {
            return;
        }
        let range = |axis: usize, limit: usize| {
            let along = [a, b, c].map(|p| self.place(p)[axis]);
            let low = along.iter().copied().fold(f32::INFINITY, f32::min);
            let high = along.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            (low.floor().max(0.0) as usize)..(high.ceil().max(0.0) as usize).min(limit)
        };
        let (rows, columns) = (range(1, self.rows), range(0, self.columns));
        for j in rows {
            for i in columns.clone() {
                let q = self.sample(i, j);
                let weights = [(q & b & c).s(), (a & q & c).s(), (a & b & q).s()].map(|w| w / area);
                if weights.iter().any(|&w| w < 0.0) {
                    continue;
                }
                let depth: f32 = weights.iter().zip(depths).map(|(w, d)| w * d).sum();
                let slot = &mut self.nearest[j * self.columns + i];
                if slot.is_none_or(|(nearest, _)| depth < nearest) {
                    let hit = Hit {
                        face,
                        weights: weights.map(f64::from),
                        outside: area < 0.0,
                    };
                    *slot = Some((depth, hit));
                }
            }
        }
    }

    /// The hit at the sample nearest a canvas point.
    fn at(&self, q: Point2) -> Option<Hit> {
        let [x, y] = self.place(q);
        let inside =
            x >= 0.0 && y >= 0.0 && (x as usize) < self.columns && (y as usize) < self.rows;
        inside
            .then(|| self.nearest[y as usize * self.columns + x as usize])
            .flatten()
            .map(|(_, hit)| hit)
    }
}

/// The mesh in the panel, each sample in the colour `colour(hit)`, lit from the eye across the
/// smooth normals (the vertex normals, weighted): dimmed as the surface turns from the eye, and
/// a glint where it faces it.
pub fn render(
    c: &mut Canvas,
    cam: &Camera,
    rect: Rect,
    mesh: &Mesh,
    colour: impl Fn(&Hit) -> Light + Sync,
) {
    let raster = Raster::new(cam, rect, mesh);
    let normals = mesh.vertex_normals();
    let [x, y, z] = cam.eye().to_euclidean().map(f64::from);
    let eye = vga3d::Vector::new(x, y, z);
    c.clip(rect);
    c.shade(SAMPLES, |q| {
        let hit = raster.at(q)?;
        let side = if hit.outside { 1.0 } else { -1.0 };
        let normal = hit.at(mesh, &normals) * side;
        let to_eye = eye - hit.at(mesh, &mesh.vertices);
        let facing = ((normal | to_eye).s() / (normal.norm() * to_eye.norm())).max(0.0) as f32;
        let glint = crate::light(1.0, 1.0, 1.0, GLINT * facing.powi(16));
        Some(colour(&hit).faded(AMBIENT + (1.0 - AMBIENT) * facing) + glint)
    });
    c.unclip();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Lens, ORIGIN3, Point3};

    #[test]
    fn a_camera_on_the_outside_sees_the_outside() {
        // From far out on +x, the middle of the panel shows the outside of the cube's +x side.
        let rect = Rect::new(0.0, 0.0, 40.0, 40.0);
        let eye = Point3::xyz(10.0, 0.5, 1.0);
        let cam = Camera::looking(rect, eye, ORIGIN3, Lens::Perspective(0.6));
        let mesh = cube(2);
        let hit = Raster::new(&cam, rect, &mesh)
            .at(rect.centre())
            .expect("the cube");
        assert!(hit.outside);
        assert!((hit.at(&mesh, &mesh.vertices).e1() - 1.0).abs() < 1e-6);
    }
}
