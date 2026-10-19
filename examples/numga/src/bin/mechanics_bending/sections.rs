//! numga's `bending/mesh.py`: the four thin-walled cross-sections, each the nodes of its
//! midline, the walls between pairs of nodes and the walls' thickness, as points of the
//! section's plane (PGA2D), across to the right and up.

use gax::pga2d::{Motor, Point};

/// A point of the section, or a direction in it.
pub type P2 = Point;

pub const NAMES: [&str; 4] = ["I-SECTION", "TURNED I-SECTION", "Z-SECTION", "SLOTTED TUBE"];
/// The turn of each I- and Z-section in its plane, in degrees.
pub const TURNS: [f64; 3] = [0.0, 30.0, 0.0];
/// The height between the flanges' midlines, the flanges' width, and the walls' thickness.
pub const HEIGHT: f64 = 0.92;
pub const FLANGE: f64 = 0.6;
pub const THICKNESS: f64 = 0.08;
/// The tube's midline radius, its wall's thickness, the opening of its slot in degrees, and
/// the walls along it.
pub const TUBE_RADIUS: f64 = 0.46;
pub const TUBE_WALL: f64 = 0.1;
pub const SLOT_ANGLE: f64 = 60.0;
pub const ARC_SEGMENTS: usize = 48;

/// A thin-walled cross-section: the nodes of its midline, its walls as pairs of nodes (an open
/// section: the walls make a tree), and each wall's thickness.
#[derive(Clone, Debug)]
pub struct Section {
    pub nodes: Vec<P2>,
    pub walls: Vec<[usize; 2]>,
    pub thickness: Vec<f64>,
}

impl Section {
    /// Each wall's two ends.
    pub fn ends(&self) -> impl Iterator<Item = [P2; 2]> + '_ {
        self.walls.iter().map(|w| w.map(|k| self.nodes[k]))
    }
}

/// A section from the coordinates of its nodes, across and up, turned about the origin by
/// `degrees`, its walls all of one thickness.
fn walled(coordinates: &[[f64; 2]], walls: &[[usize; 2]], degrees: f64, thickness: f64) -> Section {
    let turn = Motor::rotation(Point::origin(), degrees.to_radians());
    Section {
        nodes: coordinates
            .iter()
            .map(|[x, y]| turn >> Point::xy(*x, *y))
            .collect(),
        walls: walls.to_vec(),
        thickness: vec![thickness; walls.len()],
    }
}

/// An I-section: the web between the flanges' middles, and each flange out to both sides.
fn i_section(degrees: f64) -> Section {
    let (rise, half) = (HEIGHT / 2.0, FLANGE / 2.0);
    let nodes = [
        [-half, -rise],
        [0.0, -rise],
        [half, -rise],
        [-half, rise],
        [0.0, rise],
        [half, rise],
    ];
    let walls = [[1, 4], [1, 0], [1, 2], [4, 3], [4, 5]];
    walled(&nodes, &walls, degrees, THICKNESS)
}

/// A Z-section: the bottom flange out to the left, the web, and the top flange out to the right.
fn z_section(degrees: f64) -> Section {
    let rise = HEIGHT / 2.0;
    let nodes = [[-FLANGE, -rise], [0.0, -rise], [0.0, rise], [FLANGE, rise]];
    walled(&nodes, &[[0, 1], [1, 2], [2, 3]], degrees, THICKNESS)
}

/// The nodes of an arc of the midline circle, by turns of a point on the `x` axis, and its walls
/// between neighbours; the slot faces right.
fn slotted_tube() -> Section {
    let opening = SLOT_ANGLE.to_radians();
    let sweep = core::f64::consts::TAU - opening;
    let nodes = (0..=ARC_SEGMENTS)
        .map(|k| {
            let angle = opening / 2.0 + sweep * k as f64 / ARC_SEGMENTS as f64;
            Motor::rotation(Point::origin(), angle) >> Point::xy(TUBE_RADIUS, 0.0)
        })
        .collect();
    Section {
        nodes,
        walls: (0..ARC_SEGMENTS).map(|k| [k, k + 1]).collect(),
        thickness: vec![TUBE_WALL; ARC_SEGMENTS],
    }
}

/// The four sections, in the order of [`NAMES`].
pub fn sections() -> [Section; 4] {
    [
        i_section(TURNS[0]),
        i_section(TURNS[1]),
        z_section(TURNS[2]),
        slotted_tube(),
    ]
}
