//! numga's `mechanics/shells`: thin elastic shells as triangle meshes that stretch along their
//! edges and bend at their hinges (PGA3D), after Grinspun, Hirani, Desbrun and Schröder,
//! "Discrete shells" (2003).
//!
//! A shell resists stretching, which changes lengths along the surface, and bending, which
//! changes how the surface curves. Every edge is a spring along the line joining its two ends,
//! whose length is the line's norm. Every hinge, an edge with a face on either side, holds the
//! bending: each face's plane joins the hinge's line with the corner across it, and the motor
//! that folds one face onto the other is a turn about the hinge, the square root of the
//! planes' product; its logarithm is half the bending angle times the hinge's unit line. A
//! hinge stores energy as the square of how far its angle is from its rest angle.
//!
//! A hinge bent away from its rest angle twists its two faces about its line, equally and
//! oppositely: each face pushes the corner across the hinge square to its plane, by the twist
//! over the corner's distance from the hinge, and the hinge's ends take the opposite push,
//! shared by the lever rule. The forces are the exact downhill direction of the energy, and as
//! internal forces they balance: the lines they act along, the joins of each vertex with its
//! force, add up to no force and no moment.
//!
//! A sheet's rest state need not fit in space. Rest lengths that grow towards the rim of a disc
//! leave too much rim for a flat disc, which buckles into a saddle; rest angles from a natural
//! curvature that no unstretched surface can take (a saddle, as in the two layers of a seed pod
//! curving across each other) leave a strip to bend partly both ways, and it twists into a
//! ribbon.
//!
//! The animation shows both relaxing from a flat, slightly rough start, damped, lit by their
//! faces' planes, with the energy each sheds.

use gax_numga_examples::signal::frame;
use gax_numga_examples::{
    Anim, Axes, Camera, Canvas, Lens, Marker, Point2, Point3, Rect, backdrop, caption, label_above,
    palette, run_with, scene3,
};

#[path = "../shared/simplicial.rs"]
mod simplicial;

mod shells {
    use gax::field::{Field, SparseMap};
    use gax::mesh::Topology;
    use gax::pga3d::{self, Direction, Forque, Motor, Plane, Point};
    use gax_numga_examples::rng::{Draw, rng};

    use crate::simplicial;

    /// A scalar.
    pub type Scalar<S = ()> = pga3d::Scalar<S>;
    /// A line joining points: an edge, a hinge, the line a force acts along (numga's
    /// `Antibivector-`).
    pub type Line = Forque;
    /// The form of a natural curvature, on unit lines in the sheet: the curvature across them.
    pub type Curvature = Scalar<(Line, Line)>;

    /// A triangle mesh of the plane's points: its faces counter-clockwise, its topology.
    pub struct Mesh {
        pub faces: Vec<[usize; 3]>,
        pub topology: Topology<3>,
    }

    /// A shell's rest state on a mesh: the maps reading every hinge's two ends and the corners
    /// across it, every edge's rest length and every hinge's rest angle, the stiffness of both,
    /// and the mass at every vertex.
    pub struct Sheet {
        pub mesh: Mesh,
        pub corners: [SparseMap<Scalar>; 4],
        pub lengths: Vec<f64>,
        pub angles: Vec<f64>,
        pub stretching: f64,
        pub bending: Vec<f64>,
        pub masses: Vec<f64>,
    }

    /// The plane at infinity, `e0` (numga's `mv.w`): a line's meet with it is its direction.
    fn infinity() -> Plane {
        Plane::new(0.0, 0.0, 0.0, 1.0)
    }

    /// Each edge's two ends, the lower vertex first.
    fn ends(mesh: &Mesh, points: &Field<Point>) -> (Vec<Point>, Vec<Point>) {
        simplicial::edges(&mesh.topology)
            .iter()
            .map(|&[a, b]| (points.site(a), points.site(b)))
            .unzip()
    }

    /// The hinges' readings: their ends, tips, the corners across them in the face on their
    /// left and on their right.
    fn hinges(sheet: &Sheet, points: &Field<Point>) -> [Field<Point>; 4] {
        sheet.corners.each_ref().map(|c| c * points)
    }

    /// A hinge's line and its two faces' planes, both wound as the first face, so that a flat
    /// hinge folds by nothing.
    fn hinge_planes(
        end: Point,
        tip: Point,
        across: Point,
        opposite: Point,
    ) -> (Line, Plane, Plane) {
        let line = end & tip;
        // A join is intrinsic; read as the plane facing the side from which the face turns
        // counter-clockwise, extrinsic, as numga's `Plane+`.
        let face = |corner: Point| -> Plane { (line & corner).with_orientation() };
        (line, face(across), -face(opposite))
    }

    /// The angle a hinge bends by: the product of two unit planes is twice the turn about their
    /// common line, its square root folds one onto the other, and twice its logarithm, along
    /// the unit line, is the angle.
    fn bending_angle(line: Line, first: Plane, second: Plane) -> f64 {
        let motor: gax::Unit<Motor> = (second.normalized() * first.normalized()).sqrt();
        -2.0 * (motor.log() | line.normalized()).s()
    }

    /// The elastic energy of the sheet at the given points.
    pub fn energy(sheet: &Sheet, points: &Field<Point>) -> f64 {
        let (tails, heads) = ends(&sheet.mesh, points);
        let stretching: f64 = tails
            .iter()
            .zip(&heads)
            .zip(&sheet.lengths)
            .map(|((t, h), rest)| {
                let stretched = (*t & *h).norm() - rest;
                sheet.stretching * stretched * stretched / rest
            })
            .sum();
        let [ends, tips, across, opposite] = hinges(sheet, points);
        let bending: f64 = (0..ends.len())
            .map(|i| {
                let (line, first, second) =
                    hinge_planes(ends.site(i), tips.site(i), across.site(i), opposite.site(i));
                let bent = bending_angle(line, first, second) - sheet.angles[i];
                sheet.bending[i] * bent * bent
            })
            .sum();
        (stretching + bending) / 2.0
    }

    /// The force at every vertex, the energy's downhill direction: the edges' springs pulling
    /// their ends together or apart, and the hinges twisting their faces back towards their
    /// rest angles.
    pub fn forces(sheet: &Sheet, points: &Field<Point>) -> Field<Direction> {
        let (tails, heads) = ends(&sheet.mesh, points);
        let pull: Field<Direction> = (0..tails.len())
            .map(|e| {
                let (t, h, rest) = (tails[e], heads[e], sheet.lengths[e]);
                let length = (t & h).norm();
                (h - t).cast::<Direction>() * (sheet.stretching * (length - rest) / (rest * length))
            })
            .collect();

        let [ends, tips, across, opposite] = hinges(sheet, points);
        let n = ends.len();
        let mut pushes: [Vec<Direction>; 4] = core::array::from_fn(|_| Vec::with_capacity(n));
        for i in 0..n {
            let corners = [across.site(i), opposite.site(i)];
            let (line, first, second) =
                hinge_planes(ends.site(i), tips.site(i), corners[0], corners[1]);
            let moment = line.norm()
                * sheet.bending[i]
                * (bending_angle(line, first, second) - sheet.angles[i]);
            // Each face pushes the corner across the hinge square to its plane, along the
            // direction of the line through the corner perpendicular to it, by the moment over
            // the corner's distance from the hinge.
            let push = |plane: Plane, corner: Point| -> Direction {
                ((plane | corner) ^ infinity()).cast::<Direction>()
                    * (moment / plane.norm_squared())
            };
            let face = [push(first, corners[0]), push(second, corners[1])];
            // The hinge's ends take the opposite pushes, shared by the lever rule: by where each
            // corner lies along the hinge.
            let share = |corner: Point| ((ends.site(i) & corner) | line).s() / line.dot(line).s();
            let at_tip = -(face[0] * share(corners[0]) + face[1] * share(corners[1]));
            let at_end = -(face[0] + face[1]) - at_tip;
            for (k, p) in [at_end, at_tip, face[0], face[1]].into_iter().enumerate() {
                pushes[k].push(p);
            }
        }
        // The incidence's reverse sums every edge's pull at its two ends, and each reading
        // map's reverse sums the hinges' pushes at their corners.
        let incidence = sheet.mesh.topology.d::<Scalar>(0);
        let gather = |map: &SparseMap<Scalar>, values: Vec<Direction>| -> Field<Direction> {
            (&map.reverse() * &Field::new(values)).map(|p| p.cast::<Direction>())
        };
        let mut total = -gather(&incidence, pull.into_sites());
        for (map, push) in sheet.corners.iter().zip(pushes) {
            total = &total + &gather(map, push);
        }
        total
    }

    /// The sheet moving under its own forces from the given points, from rest, losing a share
    /// `damping` of its velocity every step until it settles: the points at every frame, so
    /// many steps apart, and at the end.
    pub fn relax(
        sheet: &Sheet,
        points: Field<Point>,
        step: f64,
        damping: f64,
        frames: usize,
        steps_per_frame: usize,
    ) -> Vec<Field<Point>> {
        let mut points = points;
        let mut velocity = Field::<Direction>::zeros(points.len());
        let mut states = Vec::with_capacity(frames + 1);
        for _ in 0..frames {
            states.push(points.clone());
            for _ in 0..steps_per_frame {
                let pushed = forces(sheet, &points);
                velocity = velocity
                    .iter()
                    .zip(pushed.iter())
                    .zip(&sheet.masses)
                    .map(|((v, f), m)| (v + f * (step / m)) * (1.0 - damping))
                    .collect();
                points = points.zip_with(&velocity, |p, v| p + (v * step).cast::<Point>());
            }
        }
        states.push(points);
        states
    }

    /// The rest state of a sheet of the given thickness, of unit stiffness and density, at the
    /// mesh's points, with the given rest lengths and the rest angles of the given natural
    /// curvature, which bends towards the given top side of the sheet.
    pub fn sheet(
        mesh: Mesh,
        points: &Field<Point>,
        lengths: Vec<f64>,
        curvature: Curvature,
        top: Plane,
        thickness: f64,
    ) -> Sheet {
        // Every hinge's two ends, as the face on its left winds them, then the corner across it
        // in that face and in the other.
        let edges = simplicial::edges(&mesh.topology);
        let sides = simplicial::sides(&mesh.topology);
        let n = points.len();
        let column = |k: usize| -> Vec<usize> {
            sides
                .iter()
                .map(|s| match k {
                    0 | 1 => edges[s.edge][k],
                    _ => s.across[k - 2],
                })
                .collect()
        };
        let corners: [SparseMap<Scalar>; 4] =
            core::array::from_fn(|k| simplicial::reading(&column(k), n));
        let [ends, tips, across, opposite] = corners.each_ref().map(|c| c * points);
        let mut angles = Vec::with_capacity(sides.len());
        let mut bending = Vec::with_capacity(sides.len());
        for i in 0..sides.len() {
            let (line, first, second) =
                hinge_planes(ends.site(i), tips.site(i), across.site(i), opposite.site(i));
            // The curvature across each hinge, times the distance between its faces' centres, a
            // third of the sum of the corners' distances from the hinge, is the angle the hinge
            // bends by on a surface of that curvature.
            let unit = line.normalized();
            let spread = first.norm() + second.norm();
            angles.push(curvature.of(unit).of(unit).s() * spread / (3.0 * line.norm()));
            bending.push(thickness * thickness * thickness / 2.0 * -line.dot(line).s() / spread);
        }
        // The plane through each face's corners, wound as the face, has twice the face's area
        // as its norm; read against the top side, it says which way the bending angles count.
        let planes: Vec<Plane> = mesh
            .faces
            .iter()
            .map(|f| {
                let [a, b, c] = f.map(|v| points.site(v));
                (a & b & c).with_orientation()
            })
            .collect();
        let facing: f64 = planes.iter().map(|p| (*p | top).s()).sum();
        let sense = facing.signum();
        // Each vertex carries a third of the mass of every face at it.
        let mut masses = vec![0.0; n];
        for (f, plane) in mesh.faces.iter().zip(&planes) {
            for &v in f {
                masses[v] += plane.norm() * thickness / 6.0;
            }
        }
        Sheet {
            mesh,
            corners,
            lengths,
            angles: angles.into_iter().map(|a| a * sense).collect(),
            stretching: thickness,
            bending,
            masses,
        }
    }

    /// The point at `(x, y)` of the plane `z = 0`, lifted by `z`.
    pub fn point(x: f64, y: f64, z: f64) -> Point {
        Point::xyz(x, y, z)
    }

    /// The Delaunay triangulation of points in the xy plane, its faces counter-clockwise,
    /// without the slivers along straight rows of the boundary, and its points.
    pub fn flat_mesh(coords: &[[f64; 2]], spacing: f64) -> (Mesh, Field<Point>) {
        let vectors = coords
            .iter()
            .map(|&[x, y]| gax::vga3d::Vector::new(x, y, 0.0))
            .collect();
        let delaunay = gax_numga_examples::mesh::Mesh::triangulate(Field::new(vectors));
        let points: Field<Point> = coords.iter().map(|&[x, y]| point(x, y, 0.0)).collect();
        // The plane through a face's corners has twice the face's area as its norm, all but
        // nothing for a sliver.
        let faces: Vec<[usize; 3]> = delaunay
            .faces
            .into_iter()
            .filter(|f| {
                let [a, b, c] = f.map(|v| points.site(v));
                (a & b & c).norm() > 1e-6 * spacing * spacing
            })
            .collect();
        let topology = Topology::new(coords.len(), &faces).expect("a triangle mesh");
        (Mesh { faces, topology }, points)
    }

    /// The points of a triangular lattice of the given spacing, covering the given radius
    /// around the origin.
    pub fn lattice(spacing: f64, radius: f64) -> Vec<[f64; 2]> {
        let count = (radius / spacing) as i64 + 2;
        // A row up: the direction at sixty degrees.
        let up = gax::vga2d::Vector::from_angle(core::f64::consts::PI / 3.0);
        let mut points = Vec::new();
        for row in -2 * count..=2 * count {
            for column in -count - 1..=count + 1 {
                let (r, c) = (row as f64, column as f64);
                points.push([(r + c * up.e1()) * spacing, c * up.e2() * spacing]);
            }
        }
        points
    }

    // --- scenes

    /// The disc's radius, its lattice spacing, the share by which its rest lengths grow at the
    /// rim, from none at the centre, its thickness, and the height of the bumps it starts from.
    pub const DISC_RADIUS: f64 = 1.0;
    pub const DISC_SPACING: f64 = 0.1;
    pub const GROWTH: f64 = 0.3;
    pub const DISC_THICKNESS: f64 = 0.03;
    pub const ROUGHNESS: f64 = 1e-3;
    /// The strip's length and width, its lattice spacing and thickness, and the natural
    /// curvature of the pod's two layers, one along each of two directions crossing the strip
    /// at the given angle.
    pub const STRIP_LENGTH: f64 = 2.0;
    pub const STRIP_WIDTH: f64 = 0.3;
    pub const STRIP_SPACING: f64 = 0.08;
    pub const STRIP_THICKNESS: f64 = 0.05;
    pub const NATURAL_CURVATURE: f64 = 6.0;
    pub const POD_ANGLE: f64 = core::f64::consts::FRAC_PI_4;
    /// The relaxations: the step on either lattice, the share of velocity lost every step, the
    /// frames, and the steps per frame.
    pub const DISC_STEP: f64 = 0.02;
    pub const STRIP_STEP: f64 = 0.012;
    pub const DAMPING: f64 = 0.01;
    pub const FRAMES: usize = 100;
    pub const STEPS_PER_FRAME: usize = 20;
    pub const SEED: u64 = 0;

    /// The upward side, the plane `z = 0` facing `+z`.
    pub fn top() -> Plane {
        Plane::new(0.0, 0.0, 1.0, 0.0)
    }

    /// The flat points lifted by small random heights, to let them leave their plane.
    fn rough(flat: &[[f64; 2]]) -> Field<Point> {
        let mut r = rng(SEED);
        flat.iter()
            .map(|&[x, y]| point(x, y, ROUGHNESS * r.normal()))
            .collect()
    }

    /// The zero curvature form.
    pub fn no_curvature() -> Curvature {
        (Line::slot() | Line::slot()) * 0.0
    }

    /// The growing disc and its points at every frame.
    pub fn disc() -> (Sheet, Vec<Field<Point>>) {
        let flat: Vec<[f64; 2]> = lattice(DISC_SPACING, DISC_RADIUS)
            .into_iter()
            .filter(|&[x, y]| point(x, y, 0.0).distance(Point::origin()) <= DISC_RADIUS + 1e-9)
            .collect();
        let (mesh, points) = flat_mesh(&flat, DISC_SPACING);
        // Each edge grows with the square of its middle's distance from the centre, in disc
        // radii.
        let (tails, heads) = ends(&mesh, &points);
        let lengths = tails
            .iter()
            .zip(&heads)
            .map(|(t, h)| {
                let middle = (*t + *h) * 0.5;
                let distance = (Point::origin() & middle).norm() / DISC_RADIUS;
                (*t & *h).norm() * (1.0 + GROWTH * distance * distance)
            })
            .collect();
        let sheet = sheet(
            mesh,
            &points,
            lengths,
            no_curvature(),
            top(),
            DISC_THICKNESS,
        );
        let frames = relax(
            &sheet,
            rough(&flat),
            DISC_STEP,
            DAMPING,
            FRAMES,
            STEPS_PER_FRAME,
        );
        (sheet, frames)
    }

    /// The natural curvature of a pod's two layers: along lines through the centre, turned
    /// either way from the strip's axis, one curving one way and the other the other.
    pub fn pod_curvature() -> Curvature {
        let axis = Point::origin() & Point::xyz(1.0, 0.0, 0.0);
        let turn = |sense: f64| {
            Motor::rotation_about(Point::direction(0.0, 0.0, 1.0), sense * POD_ANGLE) >> axis
        };
        let (along, across) = (turn(1.0), turn(-1.0));
        let reading = |l: Line| l | Line::slot();
        (reading(across) * reading(across) - reading(along) * reading(along)) * NATURAL_CURVATURE
    }

    /// The pod strip and its points at every frame.
    pub fn strip() -> (Sheet, Vec<Field<Point>>) {
        let flat: Vec<[f64; 2]> = lattice(STRIP_SPACING, STRIP_LENGTH)
            .into_iter()
            .filter(|&[x, y]| x.abs() <= STRIP_LENGTH / 2.0 && y.abs() <= STRIP_WIDTH / 2.0)
            .collect();
        let (mesh, points) = flat_mesh(&flat, STRIP_SPACING);
        let (tails, heads) = ends(&mesh, &points);
        let lengths = tails
            .iter()
            .zip(&heads)
            .map(|(t, h)| (*t & *h).norm())
            .collect();
        let sheet = sheet(
            mesh,
            &points,
            lengths,
            pod_curvature(),
            top(),
            STRIP_THICKNESS,
        );
        let frames = relax(
            &sheet,
            rough(&flat),
            STRIP_STEP,
            DAMPING,
            FRAMES,
            STEPS_PER_FRAME,
        );
        (sheet, frames)
    }

    #[cfg(test)]
    /// The sum of the force lines, the joins of every vertex with its force: no force and no
    /// moment for internal forces.
    pub fn net(points: &Field<Point>, forces: &Field<Direction>) -> Line {
        points.iter().zip(forces.iter()).map(|(p, f)| p & f).sum()
    }
}

use gax::field::Field;
use gax::pga3d::Point;
use shells::*;

/// The relaxations at numga's 40 ms a frame, and a second's rest on the settled sheets.
const RELAXING: f32 = 4.04;
const SECONDS: f32 = 5.0;
/// The camera: its elevation and azimuth (numga's, azimuth zero looking along `+y`).
const ELEVATION: f32 = 24.0 * core::f32::consts::PI / 180.0;
const AZIMUTH: f32 = 40.0 * core::f32::consts::PI / 180.0;

/// A relaxing sheet, its frames and the energy at each.
struct Relaxing {
    sheet: Sheet,
    frames: Vec<Field<Point>>,
    energies: Vec<f64>,
}

fn relaxing((sheet, frames): (Sheet, Vec<Field<Point>>)) -> Relaxing {
    let energies = frames.iter().map(|p| energy(&sheet, p)).collect();
    Relaxing {
        sheet,
        frames,
        energies,
    }
}

/// Both sheets, relaxed once.
struct Scenes {
    disc: Relaxing,
    strip: Relaxing,
}

fn scenes() -> Scenes {
    Scenes {
        disc: relaxing(disc()),
        strip: relaxing(strip()),
    }
}

/// A point to draw.
fn at(p: Point) -> Point3 {
    let [x, y, z] = p.to_euclidean();
    Point3::xyz(x as f32, y as f32, z as f32)
}

/// A sheet's triangles at a frame, lit by their planes, seen about its first shape's mean,
/// all frames to the scale of the first with room to spread.
fn draw_sheet(c: &mut Canvas, rect: Rect, r: &Relaxing, k: usize) {
    let first = &r.frames[0];
    let mean = first
        .iter()
        .fold(Point3::xyz(0.0, 0.0, 0.0) * 0.0, |s, p| s + at(p));
    let centre = mean.unitized();
    let reach = first
        .iter()
        .map(|p| at(p).distance(centre))
        .fold(0.0, f32::max);
    // The half height that shows the sheet's reach across the panel's width, and room above.
    let half = 1.15 * reach * (rect.height() / rect.width()).max(0.62);
    let cam = Camera::orbit(
        rect,
        centre,
        10.0,
        AZIMUTH - core::f32::consts::FRAC_PI_2,
        ELEVATION,
        Lens::Parallel(half),
    );
    let points = &r.frames[k];
    scene3::panel3(c, rect, cam, |s| {
        let sheet = palette::ORANGE.faded(0.45);
        for f in &r.sheet.mesh.faces {
            let [a, b, cc] = f.map(|v| at(points.site(v)));
            let light = s.lit(a, b, cc, sheet);
            s.tri(a, b, cc, light, 1.0);
        }
        for [a, b] in simplicial::edges(&r.sheet.mesh.topology) {
            s.seg(
                at(points.site(a)),
                at(points.site(b)),
                0.6,
                palette::YELLOW.faded(0.12),
            );
        }
    });
}

/// The energy each sheet holds as it relaxes, on a logarithmic scale.
fn draw_energy(c: &mut Canvas, rect: Rect, s: &Scenes, k: usize) {
    let curves = [&s.disc.energies, &s.strip.energies];
    let all = curves.iter().flat_map(|e| e.iter().copied());
    let (lo, hi) = all.fold((f64::MAX, 0.0f64), |(lo, hi), e| (lo.min(e), hi.max(e)));
    let n = s.disc.energies.len() as f32;
    let ax = Axes::new(rect, [0.0, n - 1.0], [(lo * 0.5) as f32, (hi * 2.0) as f32]).log_y();
    let lights = [palette::SKY, palette::GREEN];
    for (e, light) in curves.iter().zip(lights) {
        let curve: Vec<Point2> = e
            .iter()
            .enumerate()
            .map(|(i, v)| Point2::xy(i as f32, *v as f32))
            .collect();
        ax.polyline(c, &curve, 1.0, light.faded(0.35));
        ax.polyline(c, &curve[..=k], 1.6, light);
        ax.scatter(c, &[curve[k]], Marker::Dot, 6.0, light.whitened(0.3));
    }
    ax.legend(c, &[("DISC", lights[0]), ("STRIP", lights[1])]);
    ax.frame(c, "ELASTIC ENERGY AS THEY SETTLE", "FRAME", "");
}

fn draw(c: &mut Canvas, s: &Scenes, t: f32) {
    backdrop(c);
    let time = t.rem_euclid(SECONDS);
    let k = frame((time / RELAXING).min(1.0), s.disc.frames.len());
    let screen = c.rect();
    let (w, h) = (screen.width(), screen.height());
    let size = (h / 50.0).clamp(6.0, 11.0);
    let row = Rect::new(0.0, h * 0.15, w, h * 0.72);
    let left = row.part(0.0, 0.0, 0.5, 1.0).inset(8.0, 16.0, 4.0, 4.0);
    let right = row.part(0.5, 0.0, 1.0, 1.0).inset(4.0, 16.0, 8.0, 4.0);
    draw_sheet(c, left, &s.disc, k);
    label_above(c, left, "RIM GROWN 30%: A SADDLE", size, 2.0);
    draw_sheet(c, right, &s.strip, k);
    label_above(c, right, "CROSSED NATURAL CURVATURE: A RIBBON", size, 2.0);
    let bottom = Rect::new(0.0, h * 0.72, w, h).inset(70.0, 22.0, 40.0, 30.0);
    draw_energy(c, bottom, s, k);
    caption(
        c,
        "DISCRETE SHELLS: STRETCHING AND BENDING",
        "EDGE LENGTH = |TAIL & HEAD|; BENDING ANGLE = -2 LOG(SQRT(B A)) | HINGE (PGA3D)",
    );
}

fn main() {
    run_with(Anim::new("shells", SECONDS).size(960, 540), scenes, draw);
}

#[cfg(test)]
mod tests {
    use super::shells::*;
    use gax::field::Field;
    use gax::pga3d::{Plane, Point};
    use gax_numga_examples::rng::{Draw, rng};

    /// The elastic energy at points given by their coordinates.
    fn energy_at(sheet: &Sheet, coords: &[[f64; 3]]) -> f64 {
        energy(
            sheet,
            &coords
                .iter()
                .map(|&[x, y, z]| Point::xyz(x, y, z))
                .collect(),
        )
    }

    /// On a crumpled patch with natural curvature, every force matches minus the energy's
    /// derivative by central differences, and the force lines add up to nothing.
    #[test]
    fn forces_are_the_energy_gradient_and_balance() {
        let mut r = rng(1);
        let flat: Vec<[f64; 2]> = lattice(0.3, 0.7)
            .into_iter()
            .filter(|&[x, y]| Point::xyz(x, y, 0.0).distance(Point::origin()) <= 0.7)
            .collect();
        let (mesh, points) = flat_mesh(&flat, 0.3);
        let lengths = simplicial_lengths(&mesh, &points, 1.1);
        let origin = Point::origin();
        let along = origin & Point::direction(1.0, 0.0, 0.0);
        let across = origin & Point::direction(0.0, 1.0, 0.0);
        let reading = |l: Line| l | Line::slot();
        let curvature = (reading(along) * reading(along) - reading(across) * reading(across)) * 2.0;
        let sheet = sheet(mesh, &points, lengths, curvature, top(), 0.1);
        let bent: Vec<[f64; 3]> = flat
            .iter()
            .map(|&[x, y]| {
                [
                    x + 0.05 * r.normal(),
                    y + 0.05 * r.normal(),
                    0.05 * r.normal(),
                ]
            })
            .collect();
        let at: Field<Point> = bent.iter().map(|&[x, y, z]| Point::xyz(x, y, z)).collect();
        let pushes = forces(&sheet, &at);
        let axes = [
            Plane::new(1.0, 0.0, 0.0, 0.0),
            Plane::new(0.0, 1.0, 0.0, 0.0),
            Plane::new(0.0, 0.0, 1.0, 0.0),
        ];
        let shift = 1e-6;
        for vertex in [0, bent.len() / 2] {
            for (axis, plane) in axes.iter().enumerate() {
                let (mut up, mut down) = (bent.clone(), bent.clone());
                up[vertex][axis] += shift;
                down[vertex][axis] -= shift;
                let slope = (energy_at(&sheet, &up) - energy_at(&sheet, &down)) / (2.0 * shift);
                let component = (*plane & pushes.site(vertex)).s();
                assert!(
                    (component + slope).abs() <= 1e-9 + 1e-6 * slope.abs(),
                    "{vertex} {axis}: {component} {slope}"
                );
            }
        }
        let total = net(&at, &pushes);
        assert!(total.c.iter().all(|c| c.abs() < 1e-12), "{total:?}");
    }

    /// numga's own numbers on a smoothly bent patch with natural curvature (numga 35797de):
    /// the energy, the first rest angles, and a vertex's force.
    #[test]
    fn the_energy_and_forces_are_numgas() {
        let flat: Vec<[f64; 2]> = lattice(0.3, 0.7)
            .into_iter()
            .filter(|&[x, y]| Point::xyz(x, y, 0.0).distance(Point::origin()) <= 0.7)
            .collect();
        let (mesh, points) = flat_mesh(&flat, 0.3);
        assert_eq!((flat.len(), mesh.faces.len()), (19, 24));
        let lengths = simplicial_lengths(&mesh, &points, 1.1);
        let origin = Point::origin();
        let along = origin & Point::direction(1.0, 0.0, 0.0);
        let across = origin & Point::direction(0.0, 1.0, 0.0);
        let reading = |l: Line| l | Line::slot();
        let curvature = (reading(along) * reading(along) - reading(across) * reading(across)) * 2.0;
        let sheet = sheet(mesh, &points, lengths, curvature, top(), 0.1);
        let bent: Vec<[f64; 3]> = flat
            .iter()
            .map(|&[x, y]| {
                [
                    x + 0.03 * y * y,
                    y - 0.02 * x,
                    0.2 * x * x - 0.1 * y * y + 0.05 * x * y,
                ]
            })
            .collect();
        let close = |a: f64, b: f64| (a - b).abs() <= 1e-12 * b.abs().max(1.0);
        assert!(close(energy_at(&sheet, &bent), 0.005732052705237218));
        for (a, b) in sheet
            .angles
            .iter()
            .zip([0.34641016, -0.17320508, 0.34641016, -0.17320508])
        {
            assert!((a - b).abs() < 1e-8, "{a} {b}");
        }
        let at: Field<Point> = bent.iter().map(|&[x, y, z]| Point::xyz(x, y, z)).collect();
        let f = forces(&sheet, &at).site(0);
        assert!(close(
            (Plane::new(1.0, 0.0, 0.0, 0.0) & f).s(),
            -0.01571293087479768
        ));
        assert!(close(
            (Plane::new(0.0, 0.0, 1.0, 0.0) & f).s(),
            0.002424073082586886
        ));
    }

    /// Each edge's length on the mesh, times `growth`.
    fn simplicial_lengths(mesh: &Mesh, points: &Field<Point>, growth: f64) -> Vec<f64> {
        super::simplicial::edges(&mesh.topology)
            .iter()
            .map(|&[a, b]| (points.site(a) & points.site(b)).norm() * growth)
            .collect()
    }

    /// numga's scenario checks: the disc has left its plane and settled close to its grown
    /// lengths; the strip has left its plane too, neither layer's curvature winning along its
    /// whole length; in both the forces inside the sheet balance, their lines adding up to no
    /// force and no moment.
    #[test]
    fn the_sheets_buckle_and_their_forces_balance() {
        let s = super::scenes();
        for (r, rise) in [(&s.disc, 0.3), (&s.strip, 0.1)] {
            let settled = &r.frames[r.frames.len() - 1];
            let total = net(settled, &forces(&r.sheet, settled));
            assert!(
                total.c.iter().map(|c| c * c).sum::<f64>() < 1e-30,
                "{total:?}"
            );
            let heights: Vec<f64> = settled.iter().map(|p| (top() & p).s()).collect();
            let spread = heights.iter().copied().fold(f64::MIN, f64::max)
                - heights.iter().copied().fold(f64::MAX, f64::min);
            assert!(spread > rise, "{spread}");
        }
        let settled = &s.disc.frames[s.disc.frames.len() - 1];
        let mut strain: Vec<f64> = simplicial_lengths(&s.disc.sheet.mesh, settled, 1.0)
            .iter()
            .zip(&s.disc.sheet.lengths)
            .map(|(l, rest)| (l / rest - 1.0).abs())
            .collect();
        strain.sort_by(f64::total_cmp);
        let median = strain[strain.len() / 2];
        assert!(median < 0.02, "{median}");
    }

    #[test]
    fn a_frame_draws() {
        let s = super::scenes();
        for t in [0.5, 4.5] {
            gax_numga_examples::app::assert_draws_with(super::draw, &s, t);
        }
    }
}
