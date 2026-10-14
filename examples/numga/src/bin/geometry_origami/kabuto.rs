//! numga's origami `scenarios.py`: the eleven-fold Kabuto helmet sequence of the origami
//! example of ganja.js by Steven De Keninck.

use super::paper::{L, P, Paper, Pl, line_bisector, point_bisector};
use gax::pga3d::{Motor, Plane, Point};
use std::f64::consts::PI;

/// The square's half side.
pub const SIDE: f64 = 2.0;
/// The distance to a crease plane within which a point counts as on it.
pub const CREASE_TOLERANCE: f64 = SIDE * 1e-10;
/// The faces each fold turns, by their index after the stage before. Each corner flap carries
/// both layers joined along the folded edge; the horns turn those paired layers together, and
/// the two brim folds use the front sheet.
pub const SELECTIONS: [&[usize]; 11] = [
    &[0],
    &[0, 1],
    &[1, 3],
    &[0, 1, 3, 4],
    &[2, 7],
    &[0, 6],
    &[0, 7],
    &[4, 12],
    &[8],
    &[8, 9],
    &[20],
];
/// The turn each fold passes through, and where it ends: ordinary folds pass halfway, stages
/// five and six reopen to leave a crease, and the last fold turns the other way.
pub const PEAKS: [f64; 11] = [0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, -0.5];
pub const ENDS: [f64; 11] = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, -1.0];

/// The paper's plane, `z = 0`.
pub fn sheet() -> Pl {
    Plane::new(0.0, 0.0, 1.0, 0.0)
}

/// The point halfway between two finite points.
fn midpoint(a: P, b: P) -> P {
    (a + b).gp(0.5)
}

/// The square sheet, and the eleven crease planes: point coincidences set the main flaps, line
/// coincidences open the two horns, and the last two folds turn opposite layers about the same
/// brim crease.
pub fn kabuto() -> (Paper, [Pl; 11]) {
    let corner = Point::xyz(-SIDE, SIDE, 0.0);
    let corners: [P; 4] = core::array::from_fn(|k| {
        Motor::rotation_about(0.0, 0.0, 1.0, PI / 2.0 * k as f64) >> corner
    });
    let [upper_left, lower_left, lower_right, upper_right] = corners;
    let centre = corners
        .iter()
        .fold(Point::direction(0.0, 0.0, 0.0), |sum, p| sum + *p)
        .gp(0.25);
    let top = midpoint(upper_right, upper_left);
    let left = midpoint(upper_left, lower_left);
    let quarter = midpoint(upper_left, centre);
    let upper_centre = midpoint(centre, top);
    let left_centre = midpoint(centre, left);
    let eighth = midpoint(centre, quarter);
    let brim = (sheet() | (left & top)).normalized().into_inner();
    let planes = [
        point_bisector(upper_left, lower_right),
        point_bisector(upper_right, lower_right),
        point_bisector(lower_left, lower_right),
        point_bisector(centre, upper_left),
        point_bisector(top, centre),
        point_bisector(left, centre),
        line_bisector(left_centre & quarter, quarter & centre, sheet()),
        line_bisector(quarter & upper_centre, centre & quarter, sheet()),
        point_bisector(eighth, upper_left),
        brim,
        brim,
    ];
    let paper = Paper {
        points: corners.to_vec(),
        counts: vec![4],
        offsets: vec![Point::direction(0.0, 0.0, 0.0)],
        sheet: sheet(),
    };
    (paper, planes)
}

/// One fold of the recipe: the paper cut along its crease, which faces turn, the hinge, and
/// the turns it passes through and ends at.
pub struct Stage {
    pub split: Paper,
    pub turning: Vec<bool>,
    pub hinge: L,
    pub peak: f64,
    pub end: f64,
}

impl Stage {
    /// The paper turned by `angle` about the hinge.
    pub fn at(&self, angle: f64) -> Paper {
        self.split.folded(self.hinge, &self.turning, angle)
    }

    /// The turn a fraction of the way through the fold: out to the peak and on to the end, each
    /// half eased in and out (numga's smoothstep).
    pub fn angle(&self, fraction: f64) -> f64 {
        let ease = |s: f64| s * s * (3.0 - 2.0 * s);
        if fraction < 0.5 {
            self.peak * ease(2.0 * fraction)
        } else {
            self.peak + (self.end - self.peak) * ease(2.0 * fraction - 1.0)
        }
    }
}

/// The eleven folds, each cutting the paper where the one before left it.
pub fn stages() -> Vec<Stage> {
    let (mut paper, planes) = kabuto();
    (0..planes.len())
        .map(|k| {
            let (split, turning, _) = paper.split(planes[k], SELECTIONS[k], CREASE_TOLERANCE);
            let hinge = planes[k] ^ paper.sheet;
            let stage = Stage {
                split,
                turning,
                hinge,
                peak: PI * PEAKS[k],
                end: PI * ENDS[k],
            };
            paper = stage.at(stage.end);
            stage
        })
        .collect()
}
