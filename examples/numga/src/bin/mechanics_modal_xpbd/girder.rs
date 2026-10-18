//! The girder: a cross-braced truss of bars, reduced to its lowest vibration modes, the
//! eigenfields of its sparse stiffness against its masses.

use crate::lie::Lie;
use gax::field::{Field, SparseMap};
use gax::motions::Pga2d;
use gax::pga2d::{Direction, Forque, Motor, Point, Scalar, Twist};

pub type P = Point<(), f64>;
/// A displacement, or a force (held as the direction it pulls in).
pub type D = Direction<(), f64>;
pub type S = Scalar<(), f64>;
/// A map on displacements: a bar's stiffness, a vertex's mass.
pub type DMap = Direction<(Direction,), f64>;
/// Rate to momentum: twist (an extrinsic point) to forque (an intrinsic line).
pub type Inertia = Forque<(Twist,), f64>;

/// The work a force does along a displacement: the force's polar, numga's force (a line through
/// the origin, the undual of the direction it pulls in), joined with the displacement. On the
/// directions' coefficients it is their dot product.
pub fn work() -> Scalar<(Direction, Direction), f64> {
    Direction::slot().undual() & Direction::slot()
}

/// The origin of the plane.
pub fn origin() -> P {
    Point::xy(0.0, 0.0)
}

/// A truss: its points, and its bars by their ends' indices.
#[derive(Clone, Debug)]
pub struct Truss {
    pub positions: Field<P>,
    pub edges: Vec<[usize; 2]>,
}

/// The girder's points and bars: two chords of `cells` cells, vertex `2 s + level` at station
/// `s`, joined by longitudinal bars, both diagonals of every cell, and an upright at every
/// station.
pub fn truss(cells: usize, length: f64, height: f64) -> Truss {
    let positions = Field::from_fn(2 * (cells + 1), |v| {
        let (station, level) = ((v / 2) as f64, (v % 2) as f64 - 0.5);
        Point::xy(station * length / cells as f64, level * height)
    });
    let along = |offsets: [usize; 2]| (0..cells).map(move |s| offsets.map(|o| 2 * s + o));
    let edges = along([0, 2])
        .chain(along([1, 3]))
        .chain(along([0, 3]))
        .chain(along([1, 2]))
        .chain((0..=cells).map(|s| [2 * s, 2 * s + 1]))
        .collect();
    Truss { positions, edges }
}

/// Each bar's tail taken from its head: the incidence map from vertices to bars.
fn incidence(truss: &Truss) -> SparseMap<S> {
    let edges = &truss.edges;
    let cells = edges
        .iter()
        .flat_map(|_| [S::new(-1.0), S::new(1.0)])
        .collect();
    let rows = (0..edges.len()).flat_map(|e| [e, e]).collect();
    let cols = edges.iter().flat_map(|e| *e).collect();
    SparseMap::new(cells, rows, cols, (edges.len(), truss.positions.len()))
}

/// A girder at rest and its retained vibrations.
#[derive(Clone, Debug)]
pub struct Shape {
    /// The rest points, centred on the centre of mass.
    pub rest: Field<P>,
    /// Each mode's displacement of every vertex, of unit mass.
    pub modes: Vec<Field<D>>,
    /// Each mode's angular frequency, and its compliance, the inverse squared frequency.
    pub frequencies: Vec<f64>,
    pub compliance: Vec<f64>,
    pub masses: Field<S>,
    pub edges: Vec<[usize; 2]>,
    pub inertia: Inertia,
}

impl Shape {
    /// The total mass.
    pub fn mass(&self) -> f64 {
        self.masses.sum().s()
    }

    /// The vertices of a girder deformed by its modes, in its own frame.
    pub fn deformed(&self, amplitudes: impl Fn(usize) -> f64) -> Field<P> {
        Field::from_fn(self.rest.len(), |v| {
            let offset: D = (0..self.modes.len())
                .map(|m| self.modes[m].site(v) * amplitudes(m))
                .sum();
            self.rest.site(v) + offset.cast::<Point>()
        })
    }
}

/// The stiffness of the truss's bars, a field map on the vertices' displacements: each bar's
/// stiffness acts along its direction, `d k / l (d · ·)`, carried to its ends by the incidence.
pub fn stiffness(truss: &Truss, stiffness: f64) -> SparseMap<DMap> {
    let ends = &incidence(truss);
    let differences = ends * &truss.positions;
    let bars = differences.map(|difference| {
        let direction = difference.cast::<Direction>();
        let length = difference.ideal_norm();
        let unit = direction / length;
        (unit * (stiffness / length) * work().of(unit)).cast::<Direction>()
    });
    let spread = (ends * Direction::slot()).map_cells(|c| c.cast::<Direction>());
    spread
        .transpose()
        .compose(&SparseMap::from_diagonal(&bars).compose(&spread))
}

/// The rigid motions of the plane, two translations and a turn: the girder's lowest
/// eigenfields, of no stiffness, which the motor carries.
const RIGID: usize = 3;

/// A cross-braced girder, reduced to its lowest vibration modes: the stiffness's eigenfields
/// against the masses, after the rigid motions.
#[allow(clippy::disallowed_methods)] // a mode's frequency, the root of its eigenvalue
pub fn girder(
    cells: usize,
    length: f64,
    height: f64,
    bar_stiffness: f64,
    density: f64,
    modes: usize,
) -> Shape {
    let truss = truss(cells, length, height);
    let vertices = truss.positions.len();
    let vertex_mass = density * length * height / vertices as f64;
    let masses = Field::from_fn(vertices, |_| S::new(vertex_mass));
    let weights = SparseMap::from_diagonal(&masses.map(|m| Direction::slot() * m.s()));
    let (values, fields) = stiffness(&truss, bar_stiffness).eigh_with(&weights, RIGID + modes);
    // Centred on the mass: the points' mean, as the masses are equal.
    let centre = truss.positions.mean();
    let to_centre = Motor::between(centre, origin());
    let rest = truss.positions.map(|p| to_centre >> p);
    let rest_masses = vec![vertex_mass; vertices];
    let (inertia, _) = Pga2d::inertia_of(rest.sites(), &rest_masses);
    Shape {
        rest,
        modes: fields[RIGID..].to_vec(),
        frequencies: values[RIGID..].iter().map(|v| v.sqrt()).collect(),
        compliance: values[RIGID..].iter().map(|v| 1.0 / v).collect(),
        masses,
        edges: truss.edges,
        inertia,
    }
}
