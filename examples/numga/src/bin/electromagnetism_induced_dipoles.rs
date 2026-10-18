//! numga's `electromagnetism/induced_dipoles`: the dipoles an applied field induces in a cluster
//! of atoms, all found at once from one field map, in VGA3D.
//!
//! An atom in a field carries a dipole, its polarizability times the field at it, and in a
//! cluster that field is the applied one plus the fields of all the other atoms' dipoles. A dipole
//! `p` at separation `r` makes the field `(r² p + 3 r p r) / (2 |r|⁵)`: the sandwich by the plane
//! square to the separation, its dual, turns the dipole a half turn about `r`, keeping its part
//! along `r` and reversing its part across, so the field is twice the part along and minus the part across, falling off
//! with the cube of the distance. As a map `Vector <- Vector` for every pair of atoms, these are
//! the couplings of a field map over the atoms; with one over the polarizability where an atom
//! meets itself, less the couplings, it is the response, the map from the dipoles to the applied
//! field that holds them. One solve gives every dipole (Silberstein's model of 1917, and the
//! discrete dipole approximation of Purcell and Pennypacker).
//!
//! Seven atoms one spacing apart: a chain along the field polarizes more than its atoms apart
//! (the neighbours' fields add to the applied one), a chain across it less, and a ring in
//! between. numga's notebook has the field along `x`; the animation turns it once, so the chain
//! goes from along to across, and the ring, symmetric under a seventh of a turn, keeps its gain.

use gax::field::Field;
use gax_numga_examples::mesh::direction;
use gax_numga_examples::{
    Align, Anim, Axes, Canvas, Light, Marker, Point2, Rect, backdrop, caption, font, from_above,
    palette, reach3, run,
};

#[path = "../shared/electromagnetism_dipoles.rs"]
mod pairs;

/// The atoms' dipoles, in units in which four pi times the permittivity of the vacuum is one.
mod dipoles {
    use super::pairs::between_others;
    use gax::field::{Field, SparseMap};
    use gax::vga3d::Vector;
    use gax_numga_examples::mesh::direction;
    use gax_numga_examples::signal::wave;

    pub type V = Vector<(), f64>;
    /// The field one atom's dipole makes at another, `Vector <- Vector`.
    pub type Coupling = Vector<(Vector,), f64>;
    /// A field map over the atoms with coupling cells.
    pub type FieldMap = SparseMap<Coupling>;

    /// The distance between neighbouring atoms.
    pub const SPACING: f64 = 1.0;
    /// The polarizability of each atom, a tenth of the spacing cubed: well below where the
    /// cluster would polarize itself without an applied field.
    pub const ALPHA: f64 = 0.1 * SPACING * SPACING * SPACING;
    pub const FIELD: f64 = 1.0;
    pub const ATOMS: usize = 7;

    /// The open vector.
    pub fn open() -> Coupling {
        Vector::slot()
    }

    /// The atoms one spacing apart along `axis`, centred on the origin.
    pub fn chain(axis: V) -> Field<V> {
        let offset = |i: usize| SPACING * (i as f64 - (ATOMS - 1) as f64 / 2.0);
        Field::from_fn(ATOMS, |i| axis * offset(i))
    }

    /// The atoms around a circle, one spacing from their neighbours: each turned from `x` by
    /// its share of a full turn, at the radius where a seventh of a turn spans one spacing.
    pub fn ring() -> Field<V> {
        let radius = SPACING / (2.0 * wave(core::f64::consts::PI / ATOMS as f64));
        let turn = |i: usize| core::f64::consts::TAU * i as f64 / ATOMS as f64;
        Field::from_fn(ATOMS, |i| direction(turn(i)) * radius)
    }

    /// The three clusters: a chain along `x`, the chain a quarter turn across, and the ring.
    pub fn clusters() -> [(&'static str, Field<V>); 3] {
        let quarter = core::f64::consts::FRAC_PI_2;
        [
            ("CHAIN ALONG X", chain(direction(0.0))),
            ("CHAIN ACROSS X", chain(direction(quarter))),
            ("RING", ring()),
        ]
    }

    /// The field a dipole makes at separation `r` from it, as a map from the dipole: the dipole
    /// plus three times the dipole turned a half turn about `r` (in the plane its dual makes of
    /// it), over twice the cube of the distance.
    pub fn dipole_field(r: V) -> Coupling {
        let squared = r.norm_squared();
        let turned = gax::vga3d::intrinsic::Rotor::from(r.dual()) >> open();
        (turned * 3.0 + open() * squared) * (1.0 / (2.0 * squared * squared * r.norm()))
    }

    /// The fields every atom's dipole makes at every other atom, output atom first: an atom
    /// makes no field at itself.
    pub fn coupling(positions: &Field<V>) -> FieldMap {
        between_others(positions.len(), |i, j| {
            dipole_field(positions.site(i) - positions.site(j))
        })
    }

    /// One over the polarizability at every atom: the field that holds an isolated atom's dipole.
    pub fn restoring(n: usize) -> FieldMap {
        SparseMap::from_diagonal(&Field::from_fn(n, |_| open() * (1.0 / ALPHA)))
    }

    /// The map from the dipoles to the applied field at every atom that holds them: each
    /// atom's restoring field less the fields of the others.
    pub fn response(positions: &Field<V>) -> FieldMap {
        let others = coupling(positions).map_cells(|c| -c);
        &restoring(positions.len()) + &others
    }

    /// The dipoles of all the atoms in the same applied field, found together.
    pub fn dipoles(response: &FieldMap, applied: V) -> Field<V> {
        let fields = Field::from_fn(response.shape().0, |_| applied);
        response.solve_direct(&fields)
    }

    /// The mean dipole per atom along the applied field, over the dipole of an isolated atom in
    /// the same field.
    pub fn gain(dipoles: &Field<V>, applied: V) -> f64 {
        let isolated = applied * ALPHA;
        (dipoles.mean() | isolated.inverse()).s()
    }
}

use dipoles::*;

/// One turn of the applied field.
const SECONDS: f32 = 12.0;
/// The length of the isolated atom's dipole arrow, in units of the plane; every arrow to this
/// scale.
const ARROW: f64 = 0.6;
/// The plane each cluster panel shows about its centre, up and down.
const REACH: f32 = 3.8;
/// Field angles sampled for the gain curves.
const GAIN_SAMPLES: usize = 73;

/// A point of the plane, as drawn.
fn planar(v: V) -> Point2 {
    from_above(reach3(v))
}

/// The applied field at `angle` from `x`.
fn applied(angle: f64) -> V {
    direction(angle) * FIELD
}

/// A cluster as drawn: its atoms, its response, and its gain over the field's angle in degrees.
struct Cluster {
    name: &'static str,
    positions: Field<V>,
    response: FieldMap,
    gains: Vec<Point2>,
}

impl Cluster {
    fn new(name: &'static str, positions: Field<V>) -> Cluster {
        let response = response(&positions);
        let mut cluster = Cluster {
            name,
            positions,
            response,
            gains: Vec::new(),
        };
        let turn = |k: usize| core::f64::consts::TAU * k as f64 / (GAIN_SAMPLES - 1) as f64;
        cluster.gains = (0..GAIN_SAMPLES)
            .map(|k| Point2::xy(turn(k).to_degrees() as f32, cluster.gain(turn(k)) as f32))
            .collect();
        cluster
    }

    /// The dipoles in the applied field at `angle`.
    fn dipoles(&self, angle: f64) -> Field<V> {
        dipoles(&self.response, applied(angle))
    }

    /// The gain in the applied field at `angle`.
    fn gain(&self, angle: f64) -> f64 {
        gain(&self.dipoles(angle), applied(angle))
    }
}

/// The clusters, built once.
fn scene() -> &'static [Cluster; 3] {
    static SCENE: std::sync::OnceLock<[Cluster; 3]> = std::sync::OnceLock::new();
    SCENE.get_or_init(|| clusters().map(|(name, positions)| Cluster::new(name, positions)))
}

/// A cluster: each atom with its dipole as an arrow centred on it, over the dipole an isolated
/// atom would carry, with its name and gain above.
fn cluster(c: &mut Canvas, rect: Rect, cluster: &Cluster, angle: f64) {
    let ax = Axes::equal(rect, Point2::xy(0.0, 0.0), REACH);
    let field = applied(angle);
    let scale = ARROW / (field * ALPHA).norm();
    let isolated = field * (ALPHA * scale);
    let dipoles = cluster.dipoles(angle);
    for (site, dipole) in cluster.positions.iter().zip(dipoles.iter()) {
        let faint = palette::blue().faded(0.3);
        centred_arrow(c, &ax, site, isolated, [5.0, 10.0], faint);
        centred_arrow(c, &ax, site, dipole * scale, [2.2, 8.0], palette::orange());
    }
    let atoms: Vec<Point2> = cluster.positions.iter().map(planar).collect();
    ax.scatter(c, &atoms, Marker::Dot, 7.0, palette::ink());
    let label = format!("{}: GAIN {:.3}", cluster.name, gain(&dipoles, field));
    let size = (rect.height() / 20.0).clamp(6.0, 12.0);
    let size = size.min(0.9 * rect.width() / font::width(&label, 1.0));
    c.text(
        &label,
        rect.top_middle(),
        size,
        palette::ink(),
        Align::Center,
    );
}

/// An arrow centred on `centre`, its stroke's width and its head's size `[width, head]`.
fn centred_arrow(
    c: &mut Canvas,
    ax: &Axes,
    centre: V,
    arrow: V,
    [width, head]: [f32; 2],
    light: Light,
) {
    let half = arrow * 0.5;
    ax.arrow(
        c,
        planar(centre - half),
        planar(centre + half),
        width,
        head,
        light,
    );
}

/// The applied field, an arrow through the middle of its box.
fn field_arrow(c: &mut Canvas, rect: Rect, angle: f64) {
    let ax = Axes::equal(rect, Point2::xy(0.0, 0.0), 1.0);
    // Seven tenths of the box's half width either way.
    let arrow = direction(angle) * 1.4;
    centred_arrow(c, &ax, V::zero(), arrow, [3.0, 10.0], palette::yellow());
    let label_at = ax.at(0.5, 0.0);
    ax.text(
        c,
        label_at,
        "APPLIED FIELD",
        9.0,
        palette::ink(),
        Align::Center,
    );
}

/// Each cluster's gain over the field's angle, with a cursor at the angle now.
fn gains(c: &mut Canvas, rect: Rect, angle: f64) {
    let ax = Axes::new(rect, [0.0, 360.0], [0.7, 1.8]);
    ax.frame(c, "GAIN OVER THE FIELD'S ANGLE", "DEGREES FROM X", "");
    let one = [Point2::xy(0.0, 1.0), Point2::xy(360.0, 1.0)];
    ax.dashed(c, &one, 1.0, 5.0, palette::grid().faded(3.0));
    let mut legend: Vec<(&str, Light)> = Vec::new();
    for (i, cluster) in scene().iter().enumerate() {
        let colour = palette::series(i + 1);
        ax.polyline(c, &cluster.gains, 1.8, colour);
        let cursor = [Point2::xy(
            angle.to_degrees() as f32,
            cluster.gain(angle) as f32,
        )];
        ax.scatter(c, &cursor, Marker::Dot, 8.0, colour);
        legend.push((cluster.name, colour));
    }
    ax.legend(c, &legend);
}

fn draw(c: &mut Canvas, t: f32) {
    backdrop(c);
    let angle = core::f64::consts::TAU * f64::from(t / SECONDS).rem_euclid(1.0);
    let screen = c.rect();
    let (w, h) = (screen.width(), screen.height());
    let top = (h / 30.0).clamp(10.0, 22.0) * 3.6;
    let split = top + (h - top) * 0.56;
    let row = Rect::new(0.0, top, w, split);
    for (i, each) in scene().iter().enumerate() {
        cluster(c, row.column(i, 3).inset(6.0, 10.0, 6.0, 4.0), each, angle);
    }
    let lower = Rect::new(0.0, split, w, h);
    let arrow_box = lower.part(0.0, 0.0, 0.16, 1.0).inset(8.0, 20.0, 8.0, 20.0);
    field_arrow(c, arrow_box, angle);
    let plot = lower.part(0.2, 0.0, 0.97, 1.0).inset(30.0, 22.0, 4.0, 34.0);
    gains(c, plot, angle);
    caption(
        c,
        "INDUCED DIPOLES: ONE SOLVE OF A FIELD MAP OVER THE ATOMS",
        "COUPLING (R2 P + 3 R P R) / 2|R|5 FOR EVERY PAIR; RESPONSE P / ALPHA - COUPLING = APPLIED FIELD (VGA3D)",
    );
}

fn main() {
    run(Anim::new("induced_dipoles", SECONDS).size(960, 540), draw);
}

#[cfg(test)]
mod tests {
    use super::dipoles::*;
    use super::pairs::is_self_adjoint;
    use gax::field::Field;

    /// The largest coefficient of a field.
    fn largest(field: &Field<V>) -> f64 {
        field
            .iter()
            .flat_map(|v| v.c)
            .fold(0.0, |m, x| m.max(x.abs()))
    }

    #[test]
    fn the_chain_along_the_field_polarizes_most_and_the_chain_across_least() {
        let applied = V::new(FIELD, 0.0, 0.0);
        let [along, across, ring] = clusters().map(|(_, positions)| {
            let response = response(&positions);
            let dipoles = dipoles(&response, applied);
            // Each atom's dipole is its polarizability times the field at it: the applied field
            // plus the coupled fields of all the other atoms' dipoles.
            let coupled = coupling(&positions).of(&dipoles);
            let local = coupled.map(|f| (f + applied) * ALPHA);
            assert!(largest(&(&dipoles - &local)) < 1e-13);
            assert!(is_self_adjoint(&response, 1e-13));
            // Far from the polarization catastrophe: every eigenvalue of the response keeps more
            // than half of the one over the polarizability an isolated atom has.
            let identity = restoring(positions.len()).map_cells(|c| c * ALPHA);
            let (lowest, _) = response.eigh_with(&identity, 1);
            assert!(ALPHA * lowest[0] > 0.5, "{}", lowest[0]);
            gain(&dipoles, applied)
        });
        // numga: 1.66839, 0.83817, 1.17403.
        assert!((along - 1.668_391_05).abs() < 1e-8, "{along}");
        assert!((across - 0.838_172_9).abs() < 1e-8, "{across}");
        assert!((ring - 1.174_032_86).abs() < 1e-8, "{ring}");
        assert!(along > ring && ring > 1.0 && 1.0 > across);
    }

    #[test]
    fn a_frame_draws() {
        gax_numga_examples::app::assert_draws(super::draw, 1.0);
    }
}
