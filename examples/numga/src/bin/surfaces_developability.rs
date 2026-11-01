//! numga's `surfaces/developability`: a surface flowed towards developable patches, which could
//! each be cut from a flat sheet, joined along sharp creases (Stein, Grinspun and Crane,
//! "Developability of triangle meshes", 2018), in VGA3D.
//!
//! A smooth surface is developable where its normals, all around a point, lie on one great
//! circle: it bends in one direction only, about its rulings. At a vertex of a mesh, the normals
//! of the faces around it lie on one great circle exactly when one direction meets all of them at
//! a right angle. Their spread, the sum of the dyads `normal.hodge() (normal & x)` over the faces
//! around the vertex, each weighted by the face's angle there, is a map whose least eigenvalue
//! measures how far they are from that, and the energy is the sum of the least eigenvalues. The
//! angles make the spread a property of the surface rather than of its triangles: a face split in
//! two keeps its angle at the vertex.
//!
//! The flow moves every vertex down the energy's gradient, holding each vertex's least direction
//! fixed. Two edges meet in their face's normal, twice its area long; moving a corner turns the
//! normal by the part of the step across the opposite edge, over twice the area, and opens the
//! corners' angles as the neighbours move square to the edges between them. The gradient is a
//! vector, the partner a vertex's step pairs with; the flow turns it into a step by the dual, and
//! over the energy's stiffness at each vertex every vertex moves by the same share of what its
//! faces can take.
//!
//! The animation shows numga's: the Stanford bunny (as decimated in libigl's tutorial data,
//! `models/bunny.obj`, built into the binary, re-wound coherently and outward as numga's
//! `Mesh.closed` does), each face shaded flat by its own normal, creasing into patches as the
//! flow runs, with the energy falling beside it. The corner angles are gax's `angle_between`.

use gax_numga_examples::plot::Axes;
use gax_numga_examples::scene3::panel3;
use gax_numga_examples::shared::simplicial;
use gax_numga_examples::signal::frame;
use gax_numga_examples::{
    Anim, Camera, Canvas, Lens, Marker, ORIGIN3, Point2, Point3, Rect, backdrop, caption,
    label_under, palette, run_with,
};

mod developability {
    use gax::field::Field;
    use gax::mesh::Topology;
    use gax::mesh::corners as local_corners;
    use gax::vga3d;

    pub type Scalar = vga3d::Scalar;
    /// A gradient, the partner of a step (numga's `Vector+`).
    pub type Vector = vga3d::Vector;
    /// A point or a step: an intrinsic bivector, its coefficients the coordinates.
    pub type Point<S = ()> = vga3d::intrinsic::Bivector<S>;
    /// A face's normal, the join of two edges (numga's `Vector-`).
    pub type Normal = vga3d::intrinsic::Vector;
    /// The spread of the normals at a vertex: a map on steps.
    pub type Spread = Point<(Point,)>;

    /// The point at `(x, y, z)`.
    #[cfg_attr(not(test), allow(dead_code))] // numga's, the tests use it
    pub fn point(x: f64, y: f64, z: f64) -> Point {
        Vector::new(x, y, z).hodge()
    }

    /// A point's coordinates.
    pub fn coordinates(p: Point) -> [f64; 3] {
        [
            Vector::new(1.0, 0.0, 0.0),
            Vector::new(0.0, 1.0, 0.0),
            Vector::new(0.0, 0.0, 1.0),
        ]
        .map(|s| (s & p).s())
    }

    /// The edge facing each corner of each face, and the angle at each corner between the two
    /// edges meeting there: between the edge after it reversed and the edge before it.
    pub fn corner_angles(
        mesh: &Topology<3>,
        vertices: &Field<Point>,
    ) -> ([Field<Point>; 3], [Field<Scalar>; 3]) {
        let edges = mesh.facets(2).read_all(&(&mesh.d::<Scalar>(0) * vertices));
        let edges: [Field<Point>; 3] = [edges[0].clone(), edges[1].clone(), edges[2].clone()];
        let ends = local_corners(2, 1);
        let angles = core::array::from_fn(|k| {
            let (after, before) = (&edges[ends[k][0]], &edges[ends[k][1]]);
            after.zip_with(before, |a, b| {
                Scalar::new((-a).undual().angle_between(b.undual()))
            })
        });
        (edges, angles)
    }

    /// The unit normal of each face, and the spread of the normals around each vertex: the sum
    /// of their dyads, each weighted by the face's angle at the vertex.
    pub fn spread(
        mesh: &Topology<3>,
        edges: &[Field<Point>; 3],
        angles: &[Field<Scalar>; 3],
    ) -> (Field<Normal>, Vec<Spread>) {
        let normals = edges[0].zip_with(&edges[1], |a, b| (a & b).normalized());
        let corners = mesh.incidence(2, 0);
        let mut spreads = vec![Point::slot() * 0.0; mesh.len(0)];
        for f in 0..mesh.len(2) {
            let n = normals.site(f);
            let dyad = n.hodge() * (n & Point::slot());
            for (&v, angle) in corners.sites_of(f).iter().zip(angles) {
                spreads[v] += dyad * angle.site(f).s();
            }
        }
        (normals, spreads)
    }

    /// Each vertex's least eigenvalue and its direction.
    pub fn least(spreads: &[Spread]) -> (Vec<f64>, Vec<Point>) {
        spreads
            .iter()
            .map(|s| {
                let (values, vectors) = s.eigh();
                (values[0], vectors[0])
            })
            .unzip()
    }

    /// The energy's gradient at each vertex, each vertex's least direction held fixed: as the
    /// normals turn, for each face the angle-weighted dyads of its corners' directions pulled
    /// onto its normal, the tangential part twice, joined with the edge facing each corner over
    /// twice the face's area; as the angles open, each corner's angle moves with its two
    /// neighbours along the face square to the edges from the corner, over those edges' lengths.
    pub fn gradient(
        mesh: &Topology<3>,
        edges: &[Field<Point>; 3],
        angles: &[Field<Scalar>; 3],
        normals: &Field<Normal>,
        directions: &[Point],
    ) -> Vec<Vector> {
        let corners = mesh.incidence(2, 0);
        let ends = local_corners(2, 1);
        let mut out = vec![Vector::new(0.0, 0.0, 0.0); mesh.len(0)];
        for f in 0..mesh.len(2) {
            let n = normals.site(f);
            let e: [Point; 3] = core::array::from_fn(|k| edges[k].site(f));
            let double_area = (e[0] & e[1]).norm();
            let vertex = corners.sites_of(f).to_vec();
            let d: [Point; 3] = core::array::from_fn(|k| directions[vertex[k]]);
            // A direction's pairing with the face's normal: its part along the normal.
            let along: [vga3d::intrinsic::Scalar; 3] = core::array::from_fn(|k| n & d[k]);
            let pulled = (0..3)
                .map(|k| d[k] * along[k] * angles[k].site(f).s())
                .reduce(|a, b| a + b)
                .expect("three corners");
            let across = (pulled - n.hodge() * (n & pulled)) * 2.0;
            for k in 0..3 {
                // The direction along the face joined with the edge facing each corner.
                let turning = (across & e[k]) * double_area.recip();
                out[vertex[k]] += turning;
                // Moving an edge's ends opposite ways along its in-plane normal opens one
                // corner's angle and closes the other's, by the difference of their squared
                // normal parts.
                let (t, h) = (ends[k][0], ends[k][1]);
                let opening = (e[k] | n)
                    * ((along[h].s() * along[h].s() - along[t].s() * along[t].s())
                        / e[k].norm_squared());
                out[vertex[h]] += opening;
                out[vertex[t]] -= opening;
            }
        }
        out
    }

    /// How sharply the energy curves at each vertex: each face around it adds its angle there
    /// over the square of the corner's height above the opposite edge.
    pub fn stiffness(
        mesh: &Topology<3>,
        edges: &[Field<Point>; 3],
        angles: &[Field<Scalar>; 3],
    ) -> Vec<f64> {
        let corners = mesh.incidence(2, 0);
        let mut out = vec![0.0; mesh.len(0)];
        for f in 0..mesh.len(2) {
            let e: [Point; 3] = core::array::from_fn(|k| edges[k].site(f));
            let double_area = (e[0] & e[1]).norm();
            for k in 0..3 {
                out[corners.site(f, k)] +=
                    angles[k].site(f).s() * e[k].norm_squared() / (double_area * double_area);
            }
        }
        out
    }

    /// The least eigenvalues of the spreads of a surface: the energy's terms.
    #[cfg_attr(not(test), allow(dead_code))] // numga's, the tests use it
    pub fn energies(mesh: &Topology<3>, vertices: &Field<Point>) -> Vec<f64> {
        let (edges, angles) = corner_angles(mesh, vertices);
        let (_, spreads) = spread(mesh, &edges, &angles);
        least(&spreads).0
    }

    /// Gradient descent on the developability energy, each vertex's step the gradient over the
    /// stiffness there, times `step`, turned into a step by the dual: the vertices and the
    /// energy at every `stride`-th state, the first included, through `steps` steps.
    pub fn flow(
        mesh: &Topology<3>,
        start: &Field<Point>,
        step: f64,
        steps: usize,
        stride: usize,
    ) -> Vec<(Field<Point>, f64)> {
        let mut vertices = start.clone();
        let mut states = Vec::new();
        for k in 0..=steps {
            let (edges, angles) = corner_angles(mesh, &vertices);
            let (normals, spreads) = spread(mesh, &edges, &angles);
            let (values, directions) = least(&spreads);
            if k % stride == 0 {
                states.push((vertices.clone(), values.iter().sum()));
            }
            if k == steps {
                break;
            }
            let g = gradient(mesh, &edges, &angles, &normals, &directions);
            let stiff = stiffness(mesh, &edges, &angles);
            vertices = Field::from_fn(vertices.len(), |v| {
                vertices.site(v) - (g[v] * (step / stiff[v])).hodge()
            });
        }
        states
    }

    /// The Stanford bunny: the decimated model's triangles wound coherently and outward, its
    /// vertices centred in a box `SIZE` across and stood up (its `y` turned onto `z`).
    pub fn bunny() -> (Topology<3>, Field<Point>) {
        const BUNNY: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/models/bunny.obj"));
        let model = gax_numga_examples::mesh::Mesh::from_obj(BUNNY, SIZE);
        let wound = model.closed().faces;
        let upright = vga3d::Bivector::new(-core::f64::consts::FRAC_PI_4, 0.0, 0.0).exp();
        let vertices = model.vertices.map(|v| (upright >> v).hodge());
        (
            Topology::new(vertices.len(), &wound).expect("the bunny"),
            vertices,
        )
    }

    pub const SIZE: f64 = 2.0;
    pub const STEP: f64 = 0.3;
    pub const STEPS: usize = 600;
    pub const STRIDE: usize = 5;
}

use developability::*;

/// The loop: numga's 121 frames of 40 ms, slowed.
const SECONDS: f32 = 8.0;
/// The view: numga's elevation and azimuth (radians), its distance and its field of view.
const ELEVATION: f32 = 0.419;
const AZIMUTH: f32 = 0.942;
const DISTANCE: f32 = 8.0;
const FIELD: f32 = 0.337;

struct Shown {
    mesh: gax::mesh::Topology<3>,
    states: Vec<(gax::field::Field<Point>, f64)>,
}

fn shown() -> Shown {
    let (mesh, vertices) = bunny();
    let states = flow(&mesh, &vertices, STEP, STEPS, STRIDE);
    Shown { mesh, states }
}

/// A point as drawn.
fn drawn(p: Point) -> Point3 {
    let [x, y, z] = coordinates(p).map(|c| c as f32);
    Point3::xyz(x, y, z)
}

fn draw(c: &mut Canvas, s: &Shown, t: f32) {
    backdrop(c);
    let screen = c.rect();
    let index = frame((t / SECONDS).rem_euclid(1.0), s.states.len());
    let (vertices, energy) = &s.states[index];
    let left = Rect::new(
        0.0,
        screen.height() * 0.12,
        screen.width() * 0.62,
        screen.height(),
    );
    let cam = Camera::orbit(
        left,
        ORIGIN3,
        DISTANCE,
        AZIMUTH,
        ELEVATION,
        Lens::Perspective(FIELD),
    );
    panel3(c, left, cam, |scene| {
        // Each face flat, lit by its own normal.
        for face in simplicial::faces(&s.mesh) {
            let [a, b, d] = face.map(|v| drawn(vertices.site(v)));
            let lit = scene.lit(a, b, d, palette::ORANGE.faded(0.45));
            scene.tri(a, b, d, lit, 1.0);
        }
    });
    let right = Rect::new(
        screen.width() * 0.64,
        screen.height() * 0.25,
        screen.width() * 0.96,
        screen.height() * 0.8,
    );
    let steps = (s.states.len() - 1) * STRIDE;
    let ax = Axes::new(right, [0.0, steps as f32], [1e-2, 1.0]).log_y();
    ax.frame(c, "ENERGY OVER THE FIRST", "STEP", "");
    let first = s.states[0].1;
    let curve: Vec<Point2> = s
        .states
        .iter()
        .enumerate()
        .map(|(k, (_, e))| Point2::xy((k * STRIDE) as f32, (e / first) as f32))
        .collect();
    ax.polyline(c, &curve, 1.6, palette::SKY);
    ax.scatter(c, &curve[index..=index], Marker::Dot, 8.0, palette::ORANGE);
    label_under(
        c,
        left,
        &format!("STEP {} OF {steps}, ENERGY {energy:.2}", index * STRIDE),
        22.0,
        0.6,
    );
    caption(
        c,
        "DEVELOPABILITY: A BUNNY FLOWED TOWARDS FLAT PATCHES",
        "AT EACH VERTEX THE LEAST EIGENVALUE OF THE ANGLE-WEIGHTED NORMAL DYADS; DOWN ITS GRADIENT (VGA3D)",
    );
}

fn main() {
    run_with(
        Anim::new("developability", SECONDS).size(960, 540),
        shown,
        draw,
    );
}

#[cfg(test)]
mod tests {
    use super::developability::*;
    use gax::field::Field;
    use gax::mesh::Topology;
    use gax_numga_examples::rng::{Draw, rng};

    /// An open cylinder of unit radius about `z`: `rings` circles of `sectors` points.
    fn cylinder(rings: usize, sectors: usize) -> (Topology<3>, Field<Point>) {
        let vertices = (0..rings)
            .flat_map(|r| {
                let z = -1.0 + 2.0 * r as f64 / (rings - 1) as f64;
                (0..sectors).map(move |s| {
                    let d = gax::vga2d::Vector::from_angle(
                        core::f64::consts::TAU * s as f64 / sectors as f64,
                    );
                    point(d.e1(), d.e2(), z)
                })
            })
            .collect();
        let at = |r: usize, s: usize| r * sectors + s % sectors;
        let cells = || (0..rings - 1).flat_map(|r| (0..sectors).map(move |s| (r, s)));
        let first = cells().map(|(r, s)| [at(r, s), at(r, s + 1), at(r + 1, s + 1)]);
        let second = cells().map(|(r, s)| [at(r, s), at(r + 1, s + 1), at(r + 1, s)]);
        let faces: Vec<[usize; 3]> = first.chain(second).collect();
        (Topology::new(rings * sectors, &faces).unwrap(), vertices)
    }

    #[test]
    fn a_cylinder_is_developable() {
        // Around every vertex of a cylinder the face normals lie on one great circle.
        let (mesh, vertices) = cylinder(6, 24);
        assert!(energies(&mesh, &vertices).iter().all(|e| e.abs() < 1e-13));
    }

    #[test]
    fn the_gradient_is_the_energys_derivative() {
        let (mesh, vertices) = bunny();
        let (edges, angles) = corner_angles(&mesh, &vertices);
        let (normals, spreads) = spread(&mesh, &edges, &angles);
        let (_, directions) = least(&spreads);
        let g = gradient(&mesh, &edges, &angles, &normals, &directions);
        let energy = |v: &Field<Point>| energies(&mesh, v).iter().sum::<f64>();
        let epsilon = 1e-6;
        // numga draws six vertices and axes from numpy's stream; here from the examples' own,
        // seeded.
        let mut draw = rng(0);
        for _ in 0..6 {
            let vertex = (draw.uniform() * vertices.len() as f64) as usize;
            let axis = (draw.uniform() * 3.0) as usize;
            let mut unit = [0.0; 3];
            unit[axis] = 1.0;
            let nudge = point(unit[0], unit[1], unit[2]) * epsilon;
            let moved = |sign: f64| {
                Field::from_fn(vertices.len(), |v| {
                    vertices.site(v)
                        + if v == vertex {
                            nudge * sign
                        } else {
                            nudge * 0.0
                        }
                })
            };
            let difference = (energy(&moved(1.0)) - energy(&moved(-1.0))) / (2.0 * epsilon);
            // The gradient paired with a unit step along the axis.
            let along = (g[vertex] & point(unit[0], unit[1], unit[2])).s();
            assert!(
                (difference - along).abs() <= 1e-8 + 1e-6 * along.abs(),
                "{difference} {along}"
            );
        }
    }

    #[test]
    fn the_bunny_flow_takes_away_nearly_all_its_energy() {
        let (mesh, vertices) = bunny();
        let states = flow(&mesh, &vertices, STEP, 100, 25);
        let energies: Vec<f64> = states.iter().map(|s| s.1).collect();
        assert!(energies.windows(2).all(|w| w[1] < w[0]), "{energies:?}");
        assert!(energies[energies.len() - 1] < 0.2 * energies[0]);
    }

    /// numga's scenario check: through all 600 steps every kept state has less energy than the
    /// one before, and the flow takes away nearly all of it.
    #[test]
    fn the_scene_passes_numgas_checks() {
        let shown = super::shown();
        let energies: Vec<f64> = shown.states.iter().map(|s| s.1).collect();
        assert!(energies.windows(2).all(|w| w[1] < w[0]));
        assert!(energies[energies.len() - 1] < 0.1 * energies[0]);
        gax_numga_examples::app::assert_draws_with(super::draw, &shown, 1.0);
    }
}
