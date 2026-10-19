//! numga's `bending/core.py` and `scenarios.py`: a thin-walled section's bending stiffness,
//! warping, shear centre and whole stiffness, the strains under a load, and the frames of the
//! bent and twisted beam.

use crate::sections::{P2, Section};
use crate::simplex;
use gax::{Unit, pga2d, pga3d};

/// A bending strain: a line of the section, its neutral axis. The line at infinity is a pure
/// stretch.
pub type Strain2 = pga2d::Line;
/// A bending strain to the fibres' resultant pull, `Point <- Line`: the section's polarity. The
/// resultant is a point of the section weighted by the force (a direction, for a couple).
pub type Bending = pga2d::Point<(pga2d::Line,)>;
/// A point of space.
pub type P3 = pga3d::Point;
/// Any strain: how one cross-section moves relative to the next per unit length, a line of
/// space as a motion (a meet of planes, extrinsic).
pub type Strain = pga3d::Line;
/// A load: a line of force, the join of a point with a direction, or a couple (intrinsic).
pub type Load = pga3d::Forque;
/// Any strain to the resultant line of everything the section exerts, `Forque <- Line`.
pub type Stiffness = pga3d::Forque<(pga3d::Line,)>;
/// A cross-section's frame along the beam.
pub type Frame = Unit<pga3d::Motor>;

/// The material's modulus, Poisson's ratio and shear modulus; the beam's length, the pieces it
/// is cut into, and the load at its tip.
pub const MODULUS: f64 = 1.0;
pub const POISSON: f64 = 0.3;
pub const SHEAR_MODULUS: f64 = MODULUS / (2.0 * (1.0 + POISSON));
pub const LENGTH: f64 = 10.0;
pub const PIECES: usize = 40;
pub const LOAD: f64 = 1e-5;
/// The ends of the slotted tube's load's move across, across and up.
pub const ACROSS: [[f64; 2]; 2] = [[0.4, 0.0], [-1.15, 0.0]];
/// The stubby beam's length and load, and a shear modulus a million times the material's: the
/// slides held practically rigid, as Euler-Bernoulli theory takes them.
pub const STUBBY_LENGTH: f64 = 2.0;
pub const STUBBY_LOAD: f64 = 1.5e-3;
pub const RIGID_SHEAR: f64 = 1e6 * SHEAR_MODULUS;
/// The pushes along the beam: their size, and the points they go through, one inside every
/// section's kern and one outside.
pub const PUSH: f64 = 1e-3;
pub const PUSH_POINTS: [[f64; 2]; 2] = [[0.03, 0.04], [0.1, 0.3]];

/// The plane of the cross-section at the clamp, `z = 0`.
pub fn section_plane() -> pga3d::Plane {
    pga3d::Plane::new(0.0, 0.0, 1.0, 0.0)
}

/// The shift along the beam: the line where the section's plane meets the plane at infinity, a
/// turn about which carries a section along the beam.
pub fn shift_along_beam() -> Strain {
    section_plane() ^ pga3d::Plane::infinity()
}

/// The same line as a load: the couple about the beam, the join of the directions across and
/// up, which turns a section counterclockwise seen from beyond the tip. Paired with any strain
/// it gives the strain's rate of twist.
pub fn couple_about_beam() -> Load {
    P3::direction(1.0, 0.0, 0.0) & P3::direction(0.0, 1.0, 0.0)
}

/// The point at infinity along the beam.
pub fn beam_direction() -> P3 {
    P3::direction(0.0, 0.0, 1.0)
}

/// Points of the section as points of space in the clamp's section plane: gax's `pga2d` to
/// `pga3d` homomorphism takes a point of the plane to the line along `z` through it, met here
/// with the plane.
pub fn in_space(p: P2) -> P3 {
    pga3d::intrinsic::Line::from(p) ^ section_plane()
}

/// The fibre through a point of the section: the line along the beam through it, the join of
/// the point at infinity along the beam with the point.
pub fn fibre(p: P2) -> Load {
    beam_direction() & in_space(p)
}

/// Space seen along the beam: a point's place in the section, its position along the beam
/// dropped.
pub fn chart() -> pga2d::Point<(pga3d::Point,)> {
    pga2d::Point::from_images([
        P2::new(1.0, 0.0, 0.0),
        P2::new(0.0, 1.0, 0.0),
        P2::new(0.0, 0.0, 0.0),
        P2::new(0.0, 0.0, 1.0),
    ])
}

/// Where a point of space is seen along the beam, across and up.
#[cfg_attr(not(test), allow(dead_code))] // numga's checks of the tip
pub fn flat(p: P3) -> [f64; 2] {
    chart().of(p).to_euclidean()
}

/// A wall's length: the size of the line through its two ends.
pub fn length([a, b]: [P2; 2]) -> f64 {
    (a & b).norm()
}

/// Each wall's ends and its area, its length times its thickness.
fn walls(s: &Section) -> impl Iterator<Item = ([P2; 2], f64)> + '_ {
    s.ends().zip(&s.thickness).map(|(e, t)| (e, length(e) * t))
}

/// Two points along a wall that integrate any quadratic over it exactly (the simplex's lumped
/// mass points), with their shares of the wall's area.
fn samples(ends: [P2; 2], area: f64) -> impl Iterator<Item = (P2, f64)> {
    let (points, masses) = simplex::samples(&simplex::weights(2), &ends);
    points
        .into_iter()
        .zip(masses.into_iter().map(move |m| m * area))
}

/// The section's area.
#[cfg_attr(not(test), allow(dead_code))] // numga's check of the thin arc
pub fn area(s: &Section) -> f64 {
    walls(s).map(|(_, a)| a).sum()
}

/// The centroid of the whole section: the middle of each wall, weighted by the wall's area.
pub fn centroid(s: &Section) -> P2 {
    let weighted: P2 = walls(s).map(|([a, b], area)| (a + b) * (area / 2.0)).sum();
    weighted.unitized()
}

/// The bending stiffness: each fibre, through a point of the section, stretches by the strain's
/// pairing with that point, `strain & p`, and pulls along the beam through the point with the
/// modulus times its stretch. Summed over the walls, exactly by each wall's two sample points,
/// it holds the area and the first and second moments of area, with no centroid and no arms.
pub fn bending(s: &Section) -> Bending {
    let mut k = Bending::zero();
    for (ends, area) in walls(s) {
        for (p, share) in samples(ends, area) {
            k += (p * (pga2d::Line::slot() & p)) * (MODULUS * share);
        }
    }
    k
}

/// The warping of the walls seen from a pole, as a map from the pole: the fibres stretched in
/// proportion to the area that the line from the pole sweeps along the walls to them, summed
/// over the walls. The area swept along a wall is the join of the pole with its two ends, a
/// signed area, so the warped pull has the other orientation from the fibres' own. Added up
/// along the walls from node to node, less its mean over the nodes (numga's pseudo-inverse of
/// the walls' incidence: the walls of an open section are a tree), it is the warping at each
/// node, linear along each wall.
pub fn warping(s: &Section) -> pga2d::extrinsic::Point<(P2,)> {
    let pole = P2::slot();
    let swept: Vec<_> = s.ends().map(|[a, b]| pole & a & b).collect();
    let mut at = vec![None; s.nodes.len()];
    at[0] = Some(swept[0] * 0.0);
    while at.iter().any(Option::is_none) {
        for (w, [a, b]) in s.walls.iter().enumerate() {
            match (at[*a], at[*b]) {
                (Some(start), None) => at[*b] = Some(start + swept[w]),
                (None, Some(end)) => at[*a] = Some(end - swept[w]),
                _ => {}
            }
        }
    }
    let at: Vec<_> = at.into_iter().flatten().collect();
    let mean = at.iter().copied().sum::<pga2d::intrinsic::Scalar<(P2,)>>() / at.len() as f64;
    let mut warped = pga2d::extrinsic::Point::<(P2,)>::zero();
    for ((ends, area), [a, b]) in walls(s).zip(&s.walls) {
        let (along, _) = simplex::samples(&simplex::weights(2), &[at[*a] - mean, at[*b] - mean]);
        for ((p, share), w) in samples(ends, area).zip(along) {
            warped += w * p * share;
        }
    }
    warped
}

/// The shear centre: the pole whose warping no bending strain sees, its warped fibres pulling
/// with no couple across the beam, so that the pull runs through the origin and its join with
/// the origin vanishes. The warping is linear in the pole, so this is one solve, with the
/// pole's weight, the origin's, beside it.
pub fn shear_centre(s: &Section) -> P2 {
    let (origin, horizon) = (P2::origin(), pga2d::Line::infinity());
    let couples = warping(s) & origin;
    let weight = horizon * (horizon & P2::slot());
    (couples + weight).solve(horizon * (horizon & origin))
}

/// The torque per rate of twist of the open walls: the shear modulus times a third of each
/// wall's length times the cube of its thickness.
pub fn twisting(s: &Section, shear_modulus: f64) -> f64 {
    let walls = s.ends().zip(&s.thickness);
    shear_modulus * walls.map(|(e, t)| length(e) * t.powi(3) / 3.0).sum::<f64>()
}

/// The whole stiffness of the section, a sum of dyads of lines: the fibres' pulls for the
/// stretch and the bending; the couple about the beam for the twist; and for each wall, a force
/// along it through the shear centre for a slide along it, by the shear modulus times the
/// wall's area. A twist about the line along the beam through the shear centre meets every
/// slide's force and pairs with none: the twisting axis is nowhere written in, it comes out of
/// the solve.
pub fn stiffness(s: &Section, shear: P2, shear_modulus: f64) -> Stiffness {
    let slot = Strain::slot();
    let mut k = Stiffness::zero();
    for (ends, area) in walls(s) {
        for (p, share) in samples(ends, area) {
            let f = fibre(p);
            k += f * (slot & f) * (MODULUS * share);
        }
        let [a, b] = ends.map(in_space);
        let slide = in_space(shear) & ((b - a) / length(ends));
        k += slide * (slide & slot) * (shear_modulus * area);
    }
    let couple = couple_about_beam();
    k + couple * (couple & slot) * twisting(s, shear_modulus)
}

/// The bending part of a strain, as a line of the section: the line whose strain in space, the
/// meet of its vertical plane with the section's plane, stretches each point's fibre as the
/// strain does (numga's cast of the strain to the lines in the section's plane).
pub fn neutral_axis(strain: Strain) -> Strain2 {
    let points = pga3d::intrinsic::Line::from(P2::slot()) ^ section_plane();
    let fibres = beam_direction() & points;
    let pairing = (section_plane() ^ pga3d::Plane::from(pga2d::Line::slot())) & fibres;
    pairing.solve(strain & fibres)
}

/// The stress along the beam at a point of the section under a strain: the modulus times the
/// stretch of its fibre.
pub fn stress(strain: Strain, p: P2) -> f64 {
    MODULUS * (strain & fibre(p)).s()
}

/// The stress at a point under a bending strain alone.
pub fn bending_stress(strain: Strain2, p: P2) -> f64 {
    MODULUS * (strain & p).s()
}

/// The strain of a unit pull along the beam through a point of the section: its neutral axis,
/// the point's polar line under the bending stiffness.
pub fn pull(bending: Bending, p: P2) -> Strain2 {
    bending.solve(p)
}

/// The principal bending axes, weakest first: generalized eigenvectors of the metric on lines
/// against the energy form `Line & bending`. The metric does not see the line at infinity,
/// the pure stretch, whose eigenvalue is zero and comes first; the axes' eigenvalues are their
/// inverse bending stiffnesses, so the weakest comes last. (numga's metric on these lines, in
/// PGA3D, is negative, and its order the reverse.)
pub fn principal_axes(bending: Bending) -> [Strain2; 2] {
    let metric = pga2d::Line::slot() | pga2d::Line::slot();
    let energy = pga2d::Line::slot() & bending;
    let (_, axes) = metric.eigh(energy);
    [axes[2], axes[1]]
}

/// The loads seen from a station along the beam, as if the station were the clamp.
pub fn load_at(load: Load, station: f64) -> Load {
    (shift_along_beam() * (station / 2.0)).exp() << load
}

/// The frames of the cross-sections from the clamp to the tip: each piece's twist is the
/// shift along the beam plus its strain, and acts in the frame of the cross-section before it.
pub fn bend(strains: &[Strain], step: f64) -> Vec<Frame> {
    // The clamp's frame: no twist at all.
    let mut frames = vec![Frame::identity()];
    for strain in strains {
        let twist = ((shift_along_beam() + *strain) * (step / 2.0)).exp();
        let last = *frames.last().expect("the clamp");
        frames.push(last * twist);
    }
    frames
}

/// A section with what the scenes ask of it: its bending stiffness, centroid, shear centre,
/// principal axes, and its whole stiffness for a shear modulus.
#[derive(Clone, Debug)]
pub struct Body {
    pub section: Section,
    pub bending: Bending,
    pub centroid: P2,
    pub shear_centre: P2,
    pub principal: [Strain2; 2],
    pub stiffness: Stiffness,
}

pub fn body(section: Section, shear_modulus: f64) -> Body {
    let bending = bending(&section);
    let shear_centre = shear_centre(&section);
    Body {
        centroid: centroid(&section),
        principal: principal_axes(bending),
        stiffness: stiffness(&section, shear_centre, shear_modulus),
        bending,
        shear_centre,
        section,
    }
}

/// A cantilever under a load at its tip: the strain of each piece `[PIECES]` and the frame of
/// each cross-section `[PIECES + 1]` (numga's `cantilever` returns these with the principal
/// axes and the stress at the clamp, which here are the body's and [`stress`]).
#[derive(Clone, Debug)]
pub struct Beam {
    pub strains: Vec<Strain>,
    pub frames: Vec<Frame>,
}

/// A point of the section carried to the tip of the unbent beam of a length.
pub fn tip(p: P2, length: f64) -> P3 {
    (shift_along_beam() * (length / 2.0)).exp() >> in_space(p)
}

/// A cantilever of a length under a load line at its tip. The strain is solved at the middle of
/// each piece, with the lever arms measured on the straight beam: a small-deflection load with
/// a finite-motion reconstruction. A strain moves the frames by `exp(strain step / 2)`,
/// against its own sense, so it is solved against the load turned in sign.
pub fn cantilever(body: &Body, load: Load, length: f64) -> Beam {
    let step = length / PIECES as f64;
    let strains: Vec<Strain> = (0..PIECES)
        .map(|k| {
            body.stiffness
                .solve(-load_at(load, (k as f64 + 0.5) * step))
        })
        .collect();
    let frames = bend(&strains, step);
    Beam { strains, frames }
}

/// The point at infinity up the section.
pub fn up_direction() -> P3 {
    P3::direction(0.0, 1.0, 0.0)
}

/// A load straight down of a size, through a point of the section at the tip.
pub fn downward(through: P2, size: f64, length: f64) -> Load {
    tip(through, length) & (up_direction() * -size)
}

/// The tip load turned by `angle` from straight down, in the plane of the section, through the
/// centroid.
pub fn turned_load(body: &Body, angle: f64) -> Load {
    let down = up_direction() * -LOAD;
    let turn = pga3d::Motor::rotation_about(P3::direction(0.0, 0.0, 1.0), angle);
    tip(body.centroid, LENGTH) & (turn >> down)
}

/// The point the slotted tube's load goes through, a share of the way across.
pub fn across(fraction: f64) -> P2 {
    let [[x0, y0], [x1, y1]] = ACROSS;
    P2::xy(x0, y0) * (1.0 - fraction) + P2::xy(x1, y1) * fraction
}

/// The tip's displacement across the beam, seen along it.
#[cfg_attr(not(test), allow(dead_code))] // numga's checks of the tips
pub fn deflection(body: &Body, beam: &Beam) -> [f64; 2] {
    let tip = *beam.frames.last().expect("the tip") >> in_space(body.centroid);
    let [x, y] = flat(tip);
    let [cx, cy] = body.centroid.to_euclidean();
    [x - cx, y - cy]
}

/// How fast a strain turns the cross-sections about the beam, per unit length, counterclockwise
/// seen from beyond the tip: its pairing with the couple about the beam, turned in sign, since a
/// strain moves the frames against its own sense.
#[cfg_attr(not(test), allow(dead_code))] // numga's check of the tube's twist
pub fn twist_rate(strain: Strain) -> f64 {
    -(strain & couple_about_beam()).s()
}

/// How far a frame turns a cross-section about the beam, in radians, counterclockwise seen from
/// beyond the tip: the turn of the direction across the section.
pub fn twist(frame: Frame) -> f64 {
    let across = P3::direction(1.0, 0.0, 0.0);
    P2::direction(1.0, 0.0).angle_to(chart().of(frame >> across))
}

/// The kern: the points whose push along the beam squeezes every fibre. A pull through `p`
/// stretches the fibre at `q` by `pull(p) & q`, which is `pull(q) & p` (the polarity is
/// symmetric), so the kern is where `p` lies on the centroid's side of every node's polar line:
/// a square larger than the section, cut by each of those half-planes.
pub fn kern(body: &Body, reach: f64) -> Vec<P2> {
    let square: Vec<P2> = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]
        .iter()
        .map(|(x, y)| P2::xy(x * reach, y * reach))
        .collect();
    let cut_by = |kern: Vec<P2>, q: &P2| {
        let polar = pull(body.bending, *q);
        let towards = polar * (polar & body.centroid).s().signum();
        cut(&kern, towards)
    };
    body.section.nodes.iter().fold(square, cut_by)
}

/// The part of a convex polygon on the positive side of a line.
fn cut(polygon: &[P2], line: Strain2) -> Vec<P2> {
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
pub fn pushes(body: &Body) -> [(P2, Strain2); 2] {
    PUSH_POINTS.map(|[x, y]| {
        let p = P2::xy(x, y);
        (p, pull(body.bending, p) * -PUSH)
    })
}
