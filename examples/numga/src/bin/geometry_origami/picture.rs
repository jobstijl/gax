//! numga's origami `render.py`: depth-tested paper layers with two-sided diffuse lighting.
//!
//! Folded paper lies in coincident layers, and its faces cross each other, so no order of whole
//! polygons draws it: each pixel takes the face nearest the viewer there. A face is a plane in
//! screen space (x across, y down, depth towards the viewer), and its depth at a pixel is the
//! meet of the plane with the pixel's line of sight. The faces' edges are inked where they lie
//! on a face at least as near as the nearest, so covered creases leave no mark. Each face shows
//! its front or its back, lit by a fixed light and an ambient fill.

use super::paper::{P, Paper};
use gax::Unit;
use gax::pga2d;
use gax::pga3d::{Motor, Plane, Point, intrinsic};
use gax_numga_examples::Light;
use std::ops::Range;

/// The paper's two sides, and its creases' ink.
fn front() -> Light {
    Light::from_srgb(0.90, 0.46, 0.30, 1.3)
}

fn back() -> Light {
    Light::from_srgb(0.98, 0.88, 0.65, 1.2)
}

fn ink() -> Light {
    Light::from_srgb(0.37, 0.24, 0.17, 0.5)
}

/// The visible thickness of one layer of paper.
const LAYER_SPACING: f64 = 1e-6;
/// The share of the light that reaches every face, lit or not.
const AMBIENT: f64 = 0.28;
/// Half the width of an inked edge, in pixels.
const PEN: f64 = 0.9;
/// How far behind the nearest face an edge may lie and still be inked, in pixels of depth.
const EDGE_DEPTH: f64 = 1e-7;
/// The share of the picture the paper spans.
const FILL: f64 = 0.88;
/// A face seen this nearly edge on (its projected area, in square pixels) covers nothing.
const EDGE_ON: f64 = 1e-10;

/// A view of the paper: numga's camera at an elevation and azimuth, framed on given points.
#[derive(Clone, Copy, Debug)]
pub struct View {
    /// From the world into the camera's frame: x right, y up, z towards the viewer.
    turn: Unit<Motor<(), f64>>,
    /// The middle of the framed points in the camera's frame, and the scale that fits them
    /// into a picture of unit size.
    centre: P,
    scale: f64,
}

impl View {
    /// The view from `elevation` degrees above the sheet at `azimuth` degrees about its normal,
    /// framing `points`. The turn from the world: about `z` to put the eye in the `xz` plane,
    /// about `y` to put it on `z`, and a quarter turn about `z` to stand the view up.
    pub fn new(elevation: f64, azimuth: f64, points: &[P]) -> View {
        let quarter = core::f64::consts::FRAC_PI_2;
        let turn = Motor::rotation_about(Point::direction(0.0, 0.0, 1.0), -quarter)
            * Motor::rotation_about(
                Point::direction(0.0, 1.0, 0.0),
                elevation.to_radians() - quarter,
            )
            * Motor::rotation_about(Point::direction(0.0, 0.0, 1.0), -azimuth.to_radians());
        let seen: Vec<[f64; 3]> = points.iter().map(|p| (turn >> *p).to_euclidean()).collect();
        let range = |k: usize| {
            let values = seen.iter().map(|v| v[k]);
            let low = values.clone().fold(f64::INFINITY, f64::min);
            (low, values.fold(f64::NEG_INFINITY, f64::max))
        };
        let [x, y, z] = [range(0), range(1), range(2)];
        let middle = |(low, high): (f64, f64)| 0.5 * (low + high);
        View {
            turn,
            centre: Point::xyz(middle(x), middle(y), middle(z)),
            scale: FILL / (x.1 - x.0).max(y.1 - y.0),
        }
    }

    /// A point in a picture `size` pixels square: across, down, and its depth towards the
    /// viewer, in pixels.
    pub fn screen(&self, p: P, size: f64) -> P {
        let offset = (self.turn >> p) - self.centre;
        let [x, y, z] = [offset.e032(), offset.e013(), offset.e021()];
        let k = self.scale * size;
        Point::xyz(0.5 * size + k * x, 0.5 * size - k * y, k * z)
    }

    /// The direction from the paper towards the viewer.
    pub fn eye(&self) -> P {
        self.turn.reverse() >> Point::direction(0.0, 0.0, 1.0)
    }
}

/// A face in screen space: its corners and edges across the picture, its plane, its colour,
/// and the box it covers.
struct Facet {
    corners: Vec<pga2d::Point<(), f64>>,
    edges: Vec<pga2d::intrinsic::Line<(), f64>>,
    ends: Vec<[P; 2]>,
    plane: Winding,
    colour: Light,
    bounds: [f64; 4],
}

/// A plane oriented by a winding within it (intrinsic), as a join of points is.
type Winding = intrinsic::Plane<(), f64>;

/// The plane of a polygon: the sum of the planes of its fan of triangles, each the join of its
/// corners, so the polygon's plane weighted by its area and oriented by its winding.
fn polygon_plane(corners: &[P]) -> Winding {
    (1..corners.len() - 1).fold(Winding::zero(), |sum, i| {
        sum + (corners[0] & corners[i] & corners[i + 1])
    })
}

/// A screen point seen from above: across and down.
fn across(p: P) -> pga2d::Point<(), f64> {
    let [x, y, _] = p.to_euclidean();
    pga2d::Point::xy(x, y)
}

impl Facet {
    fn new(corners: &[P], colour: Light) -> Facet {
        let flat: Vec<_> = corners.iter().map(|p| across(*p)).collect();
        let n = flat.len();
        let edges = (0..n).map(|i| flat[i] & flat[(i + 1) % n]).collect();
        let ends = (0..n).map(|i| [corners[i], corners[(i + 1) % n]]).collect();
        let xs = flat.iter().map(|p| p.to_euclidean());
        let bounds = xs.fold(
            [
                f64::INFINITY,
                f64::INFINITY,
                f64::NEG_INFINITY,
                f64::NEG_INFINITY,
            ],
            |b, [x, y]| [b[0].min(x), b[1].min(y), b[2].max(x), b[3].max(y)],
        );
        Facet {
            corners: flat,
            edges,
            ends,
            plane: polygon_plane(corners),
            colour,
            bounds,
        }
    }

    /// Whether the face is seen edge on, covering nothing.
    fn edge_on(&self) -> bool {
        self.plane.e3().abs() < EDGE_ON
    }

    /// Whether a screen point lies within `pad` pixels of the face's box.
    fn near(&self, x: f64, y: f64, pad: f64) -> bool {
        let [x0, y0, x1, y1] = self.bounds;
        x > x0 - pad && x < x1 + pad && y > y0 - pad && y < y1 + pad
    }

    /// Whether the face covers a point across the picture: it lies on one side of every edge.
    fn covers(&self, q: pga2d::Point<(), f64>) -> bool {
        let sides: Vec<f64> = self.edges.iter().map(|e| (*e & q).s()).collect();
        sides.iter().all(|s| *s >= 0.0) || sides.iter().all(|s| *s <= 0.0)
    }

    /// The face's depth on the line of sight through `(x, y)`: the line's meet with its plane.
    fn depth(&self, x: f64, y: f64) -> f64 {
        let sight = Point::xyz(x, y, 0.0) & Point::direction(0.0, 0.0, 1.0);
        (sight ^ self.plane).to_euclidean()[2]
    }

    /// The depth of the face's ink at a point within the pen of an edge, or `None` where no
    /// edge is that close: on the face's plane, or along the edge for a face seen edge on.
    fn ink_depth(&self, q: pga2d::Point<(), f64>, x: f64, y: f64) -> Option<f64> {
        let n = self.corners.len();
        let (i, along) = (0..n).find_map(|i| {
            let (a, b) = (self.corners[i], self.corners[(i + 1) % n]);
            segment_foot(a, b, q).map(|t| (i, t))
        })?;
        if !self.edge_on() {
            return Some(self.depth(x, y));
        }
        let [start, end] = self.ends[i].map(|p| p.to_euclidean()[2]);
        Some(start + along * (end - start))
    }
}

/// Where along the segment from `a` to `b` (unit points) the point nearest `q` lies, 0 at `a`
/// and 1 at `b`, if `q` lies within the pen of it. Between the perpendiculars through the ends
/// (`l | a` and `l | b`, whose joins with `q` differ in sign there) the nearest point is the
/// foot on the line, and the distance the join of the unit line with `q`; beyond them, the
/// nearer end, at the norm of its join with `q`.
fn segment_foot(
    a: pga2d::Point<(), f64>,
    b: pga2d::Point<(), f64>,
    q: pga2d::Point<(), f64>,
) -> Option<f64> {
    let line = a & b;
    if line.norm() > 0.0 {
        let l = line.normalized();
        let (before, after) = (((l | a) & q).s(), ((l | b) & q).s());
        if before * after <= 0.0 {
            return ((l & q).s().abs() <= PEN).then(|| before / (before - after));
        }
    }
    let (near_a, near_b) = ((q & a).norm(), (q & b).norm());
    if near_a <= near_b {
        (near_a <= PEN).then_some(0.0)
    } else {
        (near_b <= PEN).then_some(1.0)
    }
}

/// Faces in screen space, ready to be seen pixel by pixel.
pub struct Picture {
    facets: Vec<Facet>,
}

impl Picture {
    /// The faces with corners `screen` (across, down, depth), by index ranges, in colours.
    pub fn new(screen: &[P], faces: &[Range<usize>], colours: &[Light]) -> Picture {
        let facets = faces
            .iter()
            .zip(colours)
            .map(|(f, c)| Facet::new(&screen[f.clone()], *c))
            .collect();
        Picture { facets }
    }

    /// The paper seen in a view, `size` pixels square: each face's corners lifted off the sheet
    /// by its layer offset, and coloured by the side it shows and the light it gets.
    pub fn of(paper: &Paper, view: &View, size: f64) -> Picture {
        let faces = paper.faces();
        let lifted: Vec<P> = faces
            .iter()
            .zip(&paper.offsets)
            .flat_map(|(f, o)| {
                paper.points[f.clone()]
                    .iter()
                    .map(move |p| *p + o.gp(LAYER_SPACING))
            })
            .collect();
        let screen: Vec<P> = lifted.iter().map(|p| view.screen(*p, size)).collect();
        let colours: Vec<Light> = faces
            .iter()
            .map(|f| shade(polygon_plane(&lifted[f.clone()]), view.eye()))
            .collect();
        Picture::new(&screen, &faces, &colours)
    }

    /// What shows at a point of the picture: the colour of the nearest face covering it, or ink
    /// where an edge lies on a face at least as near; `None` where no paper is.
    pub fn at(&self, x: f64, y: f64) -> Option<Light> {
        let q = pga2d::Point::xy(x, y);
        let nearest = self
            .facets
            .iter()
            .filter(|f| f.near(x, y, 0.0) && !f.edge_on() && f.covers(q))
            .map(|f| (f.depth(x, y), f.colour))
            .fold(None, |best: Option<(f64, Light)>, (d, c)| match best {
                Some((nearer, _)) if nearer > d => best,
                _ => Some((d, c)),
            });
        let visible = nearest.map_or(f64::NEG_INFINITY, |(d, _)| d);
        let inked = self
            .facets
            .iter()
            .filter(|f| f.near(x, y, PEN))
            .filter_map(|f| f.ink_depth(q, x, y))
            .any(|d| d >= visible - EDGE_DEPTH);
        if inked {
            Some(ink())
        } else {
            nearest.map(|(_, c)| c)
        }
    }
}

/// A face's colour: its front where its plane faces the viewer (the plane's inner product with
/// the one facing the eye is positive), else its back, under a fixed diffuse light from the
/// upper left and an ambient fill. The light scales the paper's linear reflectance.
fn shade(face: Winding, eye: P) -> Light {
    let unit = face.normalized();
    let facing = (unit | Plane::orthogonal_to(eye)).s() > 0.0;
    let outward = if facing { unit } else { -unit };
    let light = Plane::orthogonal_to(Point::direction(-0.4, 0.5, 1.0)).normalized();
    let illumination = AMBIENT + (1.0 - AMBIENT) * (outward | light).s().max(0.0);
    let reflectance = if facing { front() } else { back() };
    reflectance.faded(illumination as f32)
}
