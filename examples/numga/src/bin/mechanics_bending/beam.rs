//! numga's `bending/core.py` and `scenarios.py`: the section's stiffness, the strains under a
//! load, and the frames of the bent beam.

use crate::sections::{P2, Section, orientation};
use crate::simplex;
use gax::{Unit, pga2d, pga3d};

/// A strain: a line of the section, its neutral axis. The line at infinity is a pure stretch.
pub type Strain = pga2d::Line<(), f64>;
/// A strain's resultant pull: a point of the section weighted by the force (a direction for a
/// couple).
pub type Resultant = P2;
/// Strain to the fibres' resultant, `Point <- Line`: the section's polarity.
pub type Stiffness = pga2d::Point<(pga2d::Line,), f64>;
/// A point of space.
pub type P3 = pga3d::Point<(), f64>;
/// A line of space: a load, or a twist.
pub type L3 = pga3d::Line<(), f64>;
/// A cross-section's frame along the beam.
pub type Frame = Unit<pga3d::Motor<(), f64>>;

/// The material's modulus, the beam's length, the pieces it is cut into and its tip load.
pub const MODULUS: f64 = 1.0;
pub const LENGTH: f64 = 10.0;
pub const PIECES: usize = 40;
pub const LOAD: f64 = 4e-5;
/// The pushes along the beam: their size, and the points they go through, one inside every
/// section's kern and one outside.
pub const PUSH: f64 = 1e-3;
pub const PUSH_POINTS: [[f64; 2]; 2] = [[0.03, 0.04], [0.1, 0.3]];

/// The plane of the cross-section at the clamp, `z = 0`.
pub fn section_plane() -> pga3d::Plane<(), f64> {
    pga3d::Plane::new(0.0, 0.0, 1.0, 0.0)
}

/// The pure stretch: the section's line at infinity, under which every fibre stretches by one.
pub fn slide() -> Strain {
    Strain::new(0.0, 0.0, 1.0)
}

/// A point of the section as the point of space in the clamp's section plane: the line along
/// the beam through it (gax's `pga2d` to `pga3d` homomorphism takes a point of the plane to the
/// line along `z` through it) met with the plane.
pub fn in_space(p: P2) -> P3 {
    pga3d::Line::from(p) ^ section_plane()
}

/// A strain as a line of space: the plane along the beam through it (the homomorphism) met with
/// the section plane, oriented so that the twist about it stretches the fibres the strain does.
pub fn in_space_line(strain: Strain) -> L3 {
    section_plane() ^ pga3d::Plane::from(strain)
}

/// Space seen along the beam: a point's place in the section, its position along the beam
/// dropped.
pub fn chart() -> pga2d::Point<(pga3d::Point,), f64> {
    pga2d::Point::from_images([
        P2::new(1.0, 0.0, 0.0),
        P2::new(0.0, 1.0, 0.0),
        P2::new(0.0, 0.0, 0.0),
        P2::new(0.0, 0.0, 1.0),
    ])
}

/// A triangle's area, from the join of its corners.
pub fn area(t: &[P2; 3]) -> f64 {
    orientation(t[0], t[1], t[2]).abs() / 2.0
}

/// The centroid: each triangle's corners' mean, weighted by its area.
pub fn centroid(s: &Section) -> P2 {
    let weighted: P2 = s
        .triangles
        .iter()
        .map(|t| (t[0] + t[1] + t[2]) * (area(t) / 3.0))
        .sum();
    weighted.unitized()
}

/// The stiffness: each fibre, through a point of the section, stretches by the strain's
/// pairing with that point, `strain & p`, and pulls along the beam through the point with the
/// modulus times its stretch. Summed over the section, exactly by each triangle's lumped mass
/// points (the simplex's), it holds the area and the first and second moments of area, with no
/// centroid and no arms.
pub fn stiffness(s: &Section) -> Stiffness {
    let lumped = simplex::weights(3);
    let mut k = Stiffness::zero();
    for t in &s.triangles {
        let (points, masses) = simplex::samples(&lumped, t);
        for (p, m) in points.iter().zip(masses) {
            k += (*p * (pga2d::Line::slot() & *p)) * (MODULUS * area(t) * m);
        }
    }
    k
}

/// The stress along the beam at a point of the section under a strain.
pub fn stress(strain: Strain, p: P2) -> f64 {
    MODULUS * (strain & p).s()
}

/// The strain of a unit pull along the beam through a point of the section: its neutral axis,
/// the point's polar line under the stiffness.
pub fn pull(stiffness: Stiffness, p: P2) -> Strain {
    stiffness.solve(p)
}

/// The principal bending axes, weakest first: generalized eigenvectors of the metric on lines
/// against the energy form `Line & stiffness`. The metric does not see the line at infinity,
/// the pure stretch, whose eigenvalue is zero and comes first; the axes' eigenvalues are their
/// inverse bending stiffnesses, so the weakest comes last. (numga's metric on these lines, in
/// PGA3D, is negative, and its order the reverse.)
pub fn principal_axes(stiffness: Stiffness) -> [Strain; 2] {
    let metric = pga2d::Line::slot() | pga2d::Line::slot();
    let energy = pga2d::Line::slot() & stiffness;
    let (_, axes) = metric.eigh_with(energy);
    [axes[2], axes[1]]
}

/// What the fibres at `station` along the beam carry of a load (a line): the load seen from the
/// station's cross-section, met with its plane, in the section's chart. A load across the beam
/// meets it at infinity, a couple; a push along the beam meets it at its point.
pub fn carried(load: L3, station: f64) -> Resultant {
    let shift = (in_space_line(slide()) * (station / 2.0)).exp();
    chart().of((shift << load) ^ section_plane())
}

/// The frames of the cross-sections from the clamp to the tip: each piece's twist is the
/// slide plus its strain, and acts in the frame of the cross-section before it.
pub fn bend(strains: &[Strain], step: f64) -> Vec<Frame> {
    // The clamp's frame: no twist at all.
    let mut frames = vec![L3::zero().exp()];
    for strain in strains {
        let twist = (in_space_line(slide() + *strain) * (step / 2.0)).exp();
        let last = *frames.last().expect("the clamp");
        frames.push(last * twist);
    }
    frames
}

/// A section's stiffness, centroid and principal axes.
#[derive(Clone, Debug)]
pub struct Body {
    pub section: Section,
    pub stiffness: Stiffness,
    pub centroid: P2,
    pub principal: [Strain; 2],
}

pub fn body(section: Section) -> Body {
    let stiffness = stiffness(&section);
    Body {
        centroid: centroid(&section),
        principal: principal_axes(stiffness),
        stiffness,
        section,
    }
}

/// A cantilever under a tip load: the strain of each piece, the frames of the cross-sections and
/// their centres.
#[derive(Clone, Debug)]
pub struct Beam {
    pub load: P3,
    pub strains: Vec<Strain>,
    pub frames: Vec<Frame>,
    pub centres: Vec<P3>,
}

/// The tip of the straight beam, over the centroid.
pub fn tip(body: &Body) -> P3 {
    (in_space_line(slide()) * (LENGTH / 2.0)).exp() >> in_space(body.centroid)
}

/// A cantilever with its tip load (a direction of space, as long as the force) through the
/// centroid. The lever arms are measured on the straight beam, at the middle of each piece: a
/// small-deflection load with a finite-motion reconstruction.
pub fn cantilever(body: &Body, load: P3) -> Beam {
    let step = LENGTH / PIECES as f64;
    let line = tip(body) & load;
    let strains: Vec<Strain> = (0..PIECES)
        .map(|k| body.stiffness.solve(carried(line, (k as f64 + 0.5) * step)))
        .collect();
    let frames = bend(&strains, step);
    let centre = in_space(body.centroid);
    Beam {
        load,
        centres: frames.iter().map(|f| *f >> centre).collect(),
        strains,
        frames,
    }
}

/// A tip load straight down.
pub fn downward() -> P3 {
    P3::direction(0.0, -LOAD, 0.0)
}

/// The load turned by `angle` from straight down, in the plane of the section.
pub fn turned_load(angle: f64) -> P3 {
    pga3d::Motor::rotation_about(0.0, 0.0, 1.0, angle) >> downward()
}

/// Where a point of space is seen along the beam, across and up.
pub fn flat(p: P3) -> [f64; 2] {
    chart().of(p).to_euclidean()
}

/// The tip's displacement across the beam, seen along it.
pub fn deflection(body: &Body, beam: &Beam) -> [f64; 2] {
    let [x, y] = flat(*beam.centres.last().expect("the tip"));
    let [cx, cy] = body.centroid.to_euclidean();
    [x - cx, y - cy]
}

/// The kern: the points whose push along the beam compresses every fibre. A pull through `p`
/// stretches the fibre at `q` by `pull(p) & q`, which is `pull(q) & p` (the polarity is
/// symmetric), so the kern is where `p` lies on the positive side of every boundary point's
/// polar line: a square larger than the section, cut by each of those half-planes.
pub fn kern(body: &Body) -> Vec<P2> {
    let reach = 1.0;
    let mut kern: Vec<P2> = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]
        .iter()
        .map(|(x, y)| P2::xy(x * reach, y * reach))
        .collect();
    for q in &body.section.boundary {
        kern = cut(&kern, pull(body.stiffness, *q));
    }
    kern
}

/// The part of a convex polygon on the positive side of a line.
fn cut(polygon: &[P2], line: Strain) -> Vec<P2> {
    let side = |p: P2| (line & p).s();
    let mut kept = Vec::new();
    for (k, a) in polygon.iter().enumerate() {
        let b = polygon[(k + 1) % polygon.len()];
        if side(*a) >= 0.0 {
            kept.push(*a);
        }
        if side(*a) * side(b) < 0.0 {
            kept.push(((*a & b) ^ line).unitized());
        }
    }
    kept
}

/// The pushes along the beam through [`PUSH_POINTS`]: their points and strains.
pub fn pushes(body: &Body) -> [(P2, Strain); 2] {
    PUSH_POINTS.map(|[x, y]| {
        let p = P2::xy(x, y);
        (p, pull(body.stiffness, p) * -PUSH)
    })
}
