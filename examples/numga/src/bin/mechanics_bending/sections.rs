//! numga's `bending/mesh.py`: the four cross-sections, as triangles and boundaries of points of
//! the section's plane (PGA2D), across to the right and up.

use gax::pga2d::{Motor, Point};

/// A point of the section, or a direction in it.
pub type P2 = Point<(), f64>;

pub const NAMES: [&str; 4] = ["I-SECTION", "TURNED I-SECTION", "Z-SECTION", "SLOTTED TUBE"];
/// The I-section's turn for the second section, in degrees.
pub const TURN: f64 = 30.0;
/// The I- and Z-sections' height, flange width and wall thickness.
pub const HEIGHT: f64 = 1.0;
pub const FLANGE: f64 = 0.6;
pub const THICKNESS: f64 = 0.08;
/// The tube's outer radius and wall, the slot's opening angle in degrees, and the arc's
/// segments.
pub const TUBE_RADIUS: f64 = 0.5;
pub const TUBE_WALL: f64 = 0.08;
pub const SLOT_ANGLE: f64 = 60.0;
pub const ARC_SEGMENTS: usize = 48;

/// A cross-section: its triangles, counterclockwise, and its boundary.
#[derive(Clone, Debug)]
pub struct Section {
    pub triangles: Vec<[P2; 3]>,
    pub boundary: Vec<P2>,
}

/// Which side of the line from `a` to `b` the point `c` is on: positive to the left, the join
/// of the three points (twice the triangle's area for unit points).
pub fn orientation(a: P2, b: P2, c: P2) -> f64 {
    (a & b & c).s()
}

/// Triangle corners for a counterclockwise simple polygon, by ear clipping as numga's: the first
/// corner that turns left and whose triangle holds no other remaining corner, not even on its
/// edges, is cut off, until three remain.
pub fn triangulate(outline: &[P2]) -> Vec<[usize; 3]> {
    let mut remaining: Vec<usize> = (0..outline.len()).collect();
    let mut faces = Vec::new();
    while remaining.len() > 3 {
        let n = remaining.len();
        let ear_at = |i: usize| [(i + n - 1) % n, i, (i + 1) % n].map(|k| remaining[k]);
        let is_ear = |ear: [usize; 3]| {
            let [a, b, c] = ear.map(|k| outline[k]);
            let inside = |p: P2| {
                orientation(a, b, p) >= 0.0
                    && orientation(b, c, p) >= 0.0
                    && orientation(c, a, p) >= 0.0
            };
            let blocked = remaining
                .iter()
                .any(|k| !ear.contains(k) && inside(outline[*k]));
            orientation(a, b, c) > 0.0 && !blocked
        };
        let i = (0..n)
            .find(|&i| is_ear(ear_at(i)))
            .expect("a simple polygon has an ear");
        faces.push(ear_at(i));
        remaining.remove(i);
    }
    faces.push([remaining[0], remaining[1], remaining[2]]);
    faces
}

/// The points of an outline given by coordinates, across and up.
fn points(coordinates: &[[f64; 2]]) -> Vec<P2> {
    coordinates.iter().map(|[x, y]| Point::xy(*x, *y)).collect()
}

/// A polygonal section from its counterclockwise outline, turned about the origin by `degrees`.
fn polygon(outline: &[[f64; 2]], degrees: f64) -> Section {
    let turned = Motor::rotation(Point::xy(0.0, 0.0), degrees.to_radians());
    let boundary: Vec<P2> = points(outline).iter().map(|p| turned >> *p).collect();
    let triangles = triangulate(&boundary)
        .iter()
        .map(|face| face.map(|k| boundary[k]))
        .collect();
    Section {
        triangles,
        boundary,
    }
}

/// An I-section: two flanges and a web.
fn i_outline() -> Vec<[f64; 2]> {
    let (half_height, half_flange, half_web) = (HEIGHT / 2.0, FLANGE / 2.0, THICKNESS / 2.0);
    let shoulder = half_height - THICKNESS;
    vec![
        [-half_flange, -half_height],
        [half_flange, -half_height],
        [half_flange, -shoulder],
        [half_web, -shoulder],
        [half_web, shoulder],
        [half_flange, shoulder],
        [half_flange, half_height],
        [-half_flange, half_height],
        [-half_flange, shoulder],
        [-half_web, shoulder],
        [-half_web, -shoulder],
        [-half_flange, -shoulder],
    ]
}

/// A Z-section: the I-section's flanges moved to opposite sides of the web.
fn z_outline() -> Vec<[f64; 2]> {
    let (half_height, half_web) = (HEIGHT / 2.0, THICKNESS / 2.0);
    let shoulder = half_height - THICKNESS;
    let reach = FLANGE - half_web;
    vec![
        [-reach, -half_height],
        [half_web, -half_height],
        [half_web, shoulder],
        [reach, shoulder],
        [reach, half_height],
        [-half_web, half_height],
        [-half_web, -shoulder],
        [-reach, -shoulder],
    ]
}

/// An annular arc with its slot facing right: the outer and inner arcs by turns of a point on
/// the `x` axis, and the strip between them in triangles.
fn slotted_tube() -> Section {
    let opening = SLOT_ANGLE.to_radians();
    let sweep = core::f64::consts::TAU - opening;
    let arc = |radius: f64| -> Vec<P2> {
        (0..=ARC_SEGMENTS)
            .map(|k| {
                let angle = opening / 2.0 + sweep * k as f64 / ARC_SEGMENTS as f64;
                Motor::rotation(Point::xy(0.0, 0.0), angle) >> Point::xy(radius, 0.0)
            })
            .collect()
    };
    let (outer, inner) = (arc(TUBE_RADIUS), arc(TUBE_RADIUS - TUBE_WALL));
    let mut triangles = Vec::new();
    for k in 0..ARC_SEGMENTS {
        triangles.push([outer[k], outer[k + 1], inner[k + 1]]);
        triangles.push([outer[k], inner[k + 1], inner[k]]);
    }
    let boundary = outer.iter().chain(inner.iter().rev()).copied().collect();
    Section {
        triangles,
        boundary,
    }
}

/// The four sections, in the order of [`NAMES`].
pub fn sections() -> [Section; 4] {
    [
        polygon(&i_outline(), 0.0),
        polygon(&i_outline(), TURN),
        polygon(&z_outline(), 0.0),
        slotted_tube(),
    ]
}
