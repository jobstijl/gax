//! The math: a triangle's projective readings, written once for the sphere (VGA3D, whose points
//! are rays) and for the plane (PGA2D, whose points are weighted points), steady diffusion on
//! the fixed unit sphere under the interpolation that the vertices' weights choose, and
//! numga's two scenes. No drawing.
//!
//! In both algebras a point is an antivector, intrinsic (numga's `Point-`), a plane a vector,
//! extrinsic (`Plane+`), and the join of three points a number (`Volume-`). The side facing a
//! corner, the join of the other two, over the join of all three, is the corner's reciprocal
//! plane: it reads one on that corner as weighted and nothing on the other two. Their sum is
//! the triangle's section, and a point's hat of a corner is the corner's reading over the
//! section's: a quotient of linear forms, so it does not change when the point is scaled, and
//! the hats of any point sum to one. The weights the corners carry choose the section, so they
//! choose which rational interpolation the same triangle carries.

use gax::field::{Field, SparseMap};
use gax::mesh::Topology;

use crate::simplicial::{icosphere, sides};

/// numga's core, instantiated for an algebra whose points are its grade-2 antivectors: its
/// readings of one triangle.
macro_rules! readings {
    ($(#[$doc:meta])* $module:ident: $ga:ident, $plane:ident, $point:ty, $join:ty) => {
        $(#[$doc])*
        pub mod $module {
            #![allow(dead_code)] // each scene uses part of it

            use gax::mesh::corners as local_corners;

            /// A number (numga's `Scalar+`).
            pub type Scalar = gax::$ga::Scalar;
            /// A plane: a vector, extrinsic (numga's `Plane+`).
            pub type Plane = gax::$ga::$plane;
            /// A point: an antivector, intrinsic (numga's `Point-`).
            pub type Point = $point;
            /// The join of three points (numga's `Volume-`).
            pub type Volume = gax::$ga::intrinsic::Scalar;
            /// The join of two points.
            pub type Join = $join;

            /// The join of a triangle's three corners.
            pub fn volume(corners: [Point; 3]) -> Volume {
                corners[0] & corners[1] & corners[2]
            }

            /// The reciprocal planes of a triangle's corners as they are weighted: the side
            /// facing each corner, the join of the next two, over the join of all three. Each
            /// reads one on its own corner and nothing on the other two (numga's
            /// `boundary[2, 1] * (tails & heads) / (c₀ & c₁ & c₂)`).
            pub fn reciprocals(corners: [Point; 3]) -> [Plane; 3] {
                let over = volume(corners).inverse();
                [0, 1, 2].map(|k| (corners[(k + 1) % 3] & corners[(k + 2) % 3]) * over)
            }

            /// The section: the sum of the reciprocal planes, which reads one on every corner.
            pub fn section(reciprocals: [Plane; 3]) -> Plane {
                reciprocals[0] + reciprocals[1] + reciprocals[2]
            }

            /// The corner hats at a point and their gradients: each corner's reading over the
            /// section's, and its derivative, the reciprocal less the hat times the section over
            /// the section's reading, which annihilates the point itself (the radial direction).
            pub fn hats(reciprocals: [Plane; 3], p: Point) -> ([Scalar; 3], [Plane; 3]) {
                let section = section(reciprocals);
                let over = (section & p).inverse();
                let hats = reciprocals.map(|r| (r & p) * over);
                let gradients = [0, 1, 2].map(|k| (reciprocals[k] - section * hats[k]) * over);
                (hats, gradients)
            }

            /// The edge forms at a point, one for the edge facing each corner, from its local
            /// tail to its head (numga's `corners[2, 1]`): the meet of the ends' reciprocal
            /// planes read at the point, over the square of the section's reading. Each is the
            /// Whitney 1-form `λₜ dλₕ − λₕ dλₜ` of the hats.
            pub fn edge_forms(reciprocals: [Plane; 3], p: Point) -> [Plane; 3] {
                let section = section(reciprocals);
                let reading = section & p;
                let over = (reading * reading).inverse();
                let ends = local_corners(2, 1);
                [0, 1, 2].map(|e| {
                    let (tail, head) = (ends[e][0], ends[e][1]);
                    ((reciprocals[tail] ^ reciprocals[head]) & p) * over
                })
            }
        }
    };
}

readings! {
    /// The sphere's readings: VGA3D, whose points are rays, the intrinsic bivectors `yz zx xy`
    /// whose coefficients are the coordinates of a point on the ray.
    sphere: vga3d, Vector, gax::vga3d::intrinsic::Bivector, gax::vga3d::intrinsic::Vector
}

readings! {
    /// The plane's readings: PGA2D, whose points are its weighted points; only the finite
    /// affine slice is read, and the metric, which is degenerate there, never inverted.
    plane: pga2d, Line, gax::pga2d::Point, gax::pga2d::intrinsic::Line
}

use sphere::{Plane, Point, Scalar};

/// What a conductivity makes of a gradient: a current, a point (numga's `Point- <- Plane+`).
pub type Conductivity = gax::vga3d::intrinsic::Bivector<(Plane,)>;
/// The sites' amounts: the dual of a number (numga's `Amount-`).
pub type Amount = gax::vga3d::intrinsic::Pseudoscalar;

/// A point's coordinates, read with the planes across the axes.
pub fn coordinates(p: Point) -> [f64; 3] {
    [
        (Plane::new(1.0, 0.0, 0.0) & p).s(),
        (Plane::new(0.0, 1.0, 0.0) & p).s(),
        (Plane::new(0.0, 0.0, 1.0) & p).s(),
    ]
}

/// The isotropic unit conductivity: a gradient to its dual (numga's `Plane.dual()`).
pub fn isotropic() -> Conductivity {
    Plane::slot().hodge()
}

/// The quadrature on each face: its samples, their measures, and the source there; the same
/// for every lift, since they belong to the physical sphere.
pub struct Quadrature {
    /// `samples[s]`: the `s`-th sample of each face, on the unit sphere.
    pub samples: Vec<Field<Point>>,
    /// The share of the sphere's area each sample stands for.
    pub measures: Vec<Field<Scalar>>,
    /// The source there.
    pub source: Vec<Field<Scalar>>,
}

/// The diffusion's results: the vertex potentials, and on each face the numerator and the
/// section of the rational interpolation, whose readings' ratio is the potential across it.
pub struct Diffusion {
    #[cfg_attr(not(test), allow(dead_code))]
    // the frames draw the interpolation; the tests read it
    pub potential: Field<Scalar>,
    pub numerator: Field<Plane>,
    pub section: Field<Plane>,
}

/// Each face's corners as the vertices are lifted.
pub fn face_corners(mesh: &Topology<3>, vertices: &Field<Point>) -> [Field<Point>; 3] {
    let at = mesh.incidence(2, 0);
    [0, 1, 2].map(|k| at.pick(k, vertices))
}

/// Each face's reciprocal planes, the side facing each corner over the corners' join: the
/// joins of the edges, read through the faces' signed boundary (numga's
/// `boundary[2, 1] * (tails & heads) / volume`).
pub fn face_reciprocals(mesh: &Topology<3>, vertices: &Field<Point>) -> [Field<Plane>; 3] {
    let joins: Field<sphere::Join> = (0..mesh.len(1))
        .map(|e| {
            let ends = mesh.simplex(1, e);
            vertices.site(ends[0]) & vertices.site(ends[1])
        })
        .collect();
    let corners = face_corners(mesh, vertices);
    let volumes = Field::from_fn(mesh.len(2), |f| {
        sphere::volume([0, 1, 2].map(|k| corners[k].site(f)))
    });
    let facing = mesh.facets(2);
    [0, 1, 2].map(|k| {
        facing
            .read(k, &joins)
            .zip_with(&volumes, |j, v| j * v.inverse())
    })
}

/// The sum of a field of amounts, read as a number.
fn total(amounts: &Field<Amount>) -> f64 {
    amounts.iter().map(|a| a.undual().s()).sum()
}

/// Steady diffusion on a closed surface under a prescribed source: the vertex potential, and the
/// face planes that interpolate it. The samples and their positive measures belong to the
/// physical surface; only the interpolation depends on how the vertices are lifted.
///
/// The hats and their gradients at each sample assemble, corner by corner, the maps from the
/// vertex potentials to the values and gradients there. The gradient's adjugate, its transpose
/// through the join, pulls the material's response back to the vertices: the stiffness. The
/// interpolation's adjugate takes the source to the vertex loads, and the measures to the vertex
/// masses. A closed surface has a solution only for a balanced source, so the quadrature mean
/// of the load is taken out, one vertex pinned, and the potential's mean taken out after.
pub fn diffusion(
    mesh: &Topology<3>,
    vertices: &Field<Point>,
    quadrature: &Quadrature,
    conductivity: Conductivity,
) -> Diffusion {
    let reciprocals = face_reciprocals(mesh, vertices);
    let section = &(&reciprocals[0] + &reciprocals[1]) + &reciprocals[2];
    let at_corners = mesh.incidence(2, 0);
    let slot = Scalar::slot();
    let n = mesh.len(0);
    let mut stiffness = SparseMap::from_triples(Vec::new(), (n, n));
    let (mut load, mut mass) = (Field::<Amount>::zeros(n), Field::<Amount>::zeros(n));
    for ((samples, measures), source) in quadrature
        .samples
        .iter()
        .zip(&quadrature.measures)
        .zip(&quadrature.source)
    {
        let readings: Vec<([Scalar; 3], [Plane; 3])> = (0..mesh.len(2))
            .map(|f| sphere::hats([0, 1, 2].map(|k| reciprocals[k].site(f)), samples.site(f)))
            .collect();
        let interpolation = at_corners.assemble(|f, k| readings[f].0[k] * slot);
        let gradient = at_corners.assemble(|f, k| readings[f].1[k] * slot);
        let response = SparseMap::from_diagonal(&measures.map(|m| conductivity * m.s()));
        stiffness = &stiffness + &gradient.adjugate().compose(&response.compose(&gradient));
        let weighted = measures.zip_with(source, |m, s| (m * s).hodge());
        load = &load + &interpolation.adjugate().of(&weighted);
        mass = &mass + &interpolation.adjugate().of(&measures.map(|m| m.hodge()));
    }
    let balance = total(&load) / total(&mass);
    let load = &load - &mass.map(|m| m * balance);
    let pin = SparseMap::from_triples(vec![(0, 0, slot.hodge())], (n, n));
    let potential = (&stiffness + &pin).solve_direct(&load);
    let mean = potential
        .iter()
        .zip(mass.iter())
        .map(|(p, m)| (p & m).s())
        .sum::<f64>()
        / total(&mass);
    let potential = potential.map(|p| p - Scalar::new(mean));
    let corners = at_corners.read_all(&potential);
    let numerator = Field::from_fn(mesh.len(2), |f| {
        (0..3)
            .map(|k| reciprocals[k].site(f) * corners[k].site(f))
            .fold(Plane::zero(), |a, b| a + b)
    });
    Diffusion {
        potential,
        numerator,
        section,
    }
}

// The scenes.

/// The icosphere's subdivisions.
pub const LEVELS: usize = 1;
/// The lattices of the quadrature and of the display on each face.
pub const QUADRATURE_DIVISIONS: usize = 5;
pub const DISPLAY_DIVISIONS: usize = 9;
/// The frames of one loop of the lifted vertex.
pub const FRAMES: usize = 90;
/// The swing of the lifted vertex's log radius.
pub const LOG_RADIUS_AMPLITUDE: f64 = 0.65;
/// The planar triangle's corners, their weights and their values.
pub const PLANAR_CORNERS: [[f64; 2]; 3] = [[-0.9, -0.65], [0.9, -0.5], [-0.1, 0.95]];
pub const CORNER_SCALES: [f64; 3] = [0.45, 1.7, 1.0];
pub const CORNER_VALUES: [f64; 3] = [0.0, 1.0, 0.25];

/// The lattice of `n` steps across the reference triangle: its small triangles, the upward ones
/// first, and each node's barycentric weights, independent of any triangle in space (numga's
/// `triangle_grid`, nodes in its order).
pub fn triangle_grid(n: usize) -> (Topology<3>, Vec<[f64; 3]>) {
    let mut lookup = vec![vec![usize::MAX; n + 1]; n + 1];
    let mut weights = Vec::new();
    for (first, row) in lookup.iter_mut().enumerate() {
        for (second, slot) in row.iter_mut().enumerate().take(n + 1 - first) {
            *slot = weights.len();
            weights.push([n - first - second, first, second].map(|k| k as f64 / n as f64));
        }
    }
    let cells = || (0..n).flat_map(|i| (0..n).map(move |j| (i, j)));
    let upward = cells()
        .filter(|&(i, j)| i + j < n)
        .map(|(i, j)| [lookup[i][j], lookup[i + 1][j], lookup[i][j + 1]]);
    let downward = cells()
        .filter(|&(i, j)| i + j + 1 < n)
        .map(|(i, j)| [lookup[i + 1][j], lookup[i + 1][j + 1], lookup[i][j + 1]]);
    let faces: Vec<[usize; 3]> = upward.chain(downward).collect();
    let grid = Topology::new(weights.len(), &faces).expect("the lattice's triangles");
    (grid, weights)
}

/// The barycentric weights of the centroids of the lattice's small triangles: each samples its
/// share of the face (numga's quadrature).
pub fn centroid_weights(n: usize) -> Vec<[f64; 3]> {
    let (grid, weights) = triangle_grid(n);
    grid.simplices(2)
        .map(|t| [0, 1, 2].map(|k| t.iter().map(|&v| weights[v][k]).sum::<f64>() / 3.0))
        .collect()
}

/// The point of each face at the given barycentric weights of its corners.
fn at_weights(corners: &[Field<Point>; 3], w: [f64; 3]) -> Field<Point> {
    Field::from_fn(corners[0].len(), |f| {
        corners[0].site(f) * w[0] + corners[1].site(f) * w[1] + corners[2].site(f) * w[2]
    })
}

/// The sphere's quadrature on the flat icosphere's faces, pushed out radially: each sample is a
/// chord point normalized, and stands for its share of the face's area, expanded by the
/// inverse cube of the chord point's distance; the source is twice the height, so that the
/// height, an eigenfunction of the sphere's Laplacian, is the solution.
pub fn quadrature(mesh: &Topology<3>, vertices: &Field<Point>, divisions: usize) -> Quadrature {
    let corners = face_corners(mesh, vertices);
    let volumes = Field::from_fn(mesh.len(2), |f| {
        sphere::volume([0, 1, 2].map(|k| corners[k].site(f)))
    });
    let weights = centroid_weights(divisions);
    let count = weights.len() as f64;
    let up = Plane::new(0.0, 0.0, 1.0);
    let (mut samples, mut measures, mut source) = (Vec::new(), Vec::new(), Vec::new());
    for &w in &weights {
        let chords = at_weights(&corners, w);
        let distance = chords.map(|c| Scalar::new(c.norm()));
        measures.push(volumes.zip_with(&distance, |v, d| {
            Scalar::new(v.norm()) * (d * d * d * (2.0 * count)).inverse()
        }));
        let on_sphere = chords.map(|c| c.normalized());
        source.push(on_sphere.map(|p| (up & p) * 2.0));
        samples.push(on_sphere);
    }
    Quadrature {
        samples,
        measures,
        source,
    }
}

/// One frame of the sphere: the lifted vertices, the dual's points (the faces' sections turned
/// into points, on the sphere), and the interpolation on each face, its numerator and section.
pub struct Frame {
    pub lifted: Field<Point>,
    pub poles: Field<Point>,
    pub numerator: Field<Plane>,
    pub section: Field<Plane>,
}

impl Frame {
    /// The potential at a point of face `f`: the numerator's reading over the section's, the
    /// same anywhere on the point's ray.
    pub fn value(&self, f: usize, p: Point) -> f64 {
        let (n, d) = (self.numerator.site(f), self.section.site(f));
        ((n & p) * (d & p).inverse()).s()
    }
}

/// The sphere's scene: the icosphere, its quadrature, the display samples of each face, the
/// lifted vertex, and the frames.
pub struct SphereScene {
    pub mesh: Topology<3>,
    pub vertices: Field<Point>,
    pub quadrature: Quadrature,
    /// `display[f][s]`: numga's display lattice on each face, on the sphere, which the tests
    /// read the frames at (the drawing reads them at every pixel).
    #[cfg_attr(not(test), allow(dead_code))]
    pub display: Vec<Vec<Point>>,
    /// The vertex that moves along its ray: the one farthest along `x`.
    pub chosen: usize,
    /// The interior edges and the faces beside each, for the dual's arcs.
    pub beside: Vec<[usize; 2]>,
    /// The faces' reciprocal planes on the sphere, unweighted: a point lies in a face's cone
    /// where it reads none of them negative.
    pub cones: Vec<[Plane; 3]>,
}

impl SphereScene {
    pub fn new() -> Self {
        let (mesh, vectors) = icosphere(LEVELS);
        let vertices = vectors.map(|v| v.hodge());
        let quadrature = quadrature(&mesh, &vertices, QUADRATURE_DIVISIONS);
        let (_, weights) = triangle_grid(DISPLAY_DIVISIONS);
        let corners = face_corners(&mesh, &vertices);
        let display = (0..mesh.len(2))
            .map(|f| {
                weights
                    .iter()
                    .map(|w| {
                        let chord = corners[0].site(f) * w[0]
                            + corners[1].site(f) * w[1]
                            + corners[2].site(f) * w[2];
                        chord.normalized()
                    })
                    .collect()
            })
            .collect();
        let along = Plane::new(1.0, 0.0, 0.0);
        let chosen = (0..vertices.len())
            .max_by(|&a, &b| {
                (along & vertices.site(a))
                    .s()
                    .total_cmp(&(along & vertices.site(b)).s())
            })
            .expect("vertices");
        let beside = sides(&mesh).into_iter().map(|s| s.faces).collect();
        let reciprocals = face_reciprocals(&mesh, &vertices);
        let cones = (0..mesh.len(2))
            .map(|f| [0, 1, 2].map(|k| reciprocals[k].site(f)))
            .collect();
        SphereScene {
            mesh,
            vertices,
            quadrature,
            display,
            chosen,
            beside,
            cones,
        }
    }

    /// The face whose cone holds a point.
    pub fn face_of(&self, p: Point) -> Option<usize> {
        self.cones
            .iter()
            .position(|r| r.iter().all(|&r| (r & p).s() >= -1e-12))
    }

    /// The vertex weights at a phase of the loop: the chosen vertex's radius swings by
    /// `exp(0.65 sin φ)`, the others stay on the sphere.
    pub fn scales(&self, phase: f64) -> Vec<f64> {
        let swing = gax_numga_examples::signal::wave(phase);
        let radius = (LOG_RADIUS_AMPLITUDE * swing).exp();
        (0..self.vertices.len())
            .map(|v| if v == self.chosen { radius } else { 1.0 })
            .collect()
    }

    /// The same problem solved again under the interpolation that these weights choose.
    pub fn frame(&self, scales: &[f64]) -> Frame {
        let lifted = Field::from_fn(self.vertices.len(), |v| self.vertices.site(v) * scales[v]);
        let solved = diffusion(&self.mesh, &lifted, &self.quadrature, isotropic());
        Frame {
            lifted,
            poles: solved.section.map(|s| s.hodge().normalized()),
            numerator: solved.numerator,
            section: solved.section,
        }
    }
}

/// The affine slice: PGA2D's triangle with weighted corners, its display lattice, and the
/// potential and current at each node.
pub struct Planar {
    pub grid: Topology<3>,
    pub points: Vec<plane::Point>,
    /// The hats and their gradients at each node, which the scenario checks.
    #[cfg_attr(not(test), allow(dead_code))]
    pub hats: Vec<[plane::Scalar; 3]>,
    #[cfg_attr(not(test), allow(dead_code))]
    pub gradients: Vec<[plane::Plane; 3]>,
    pub potential: Vec<f64>,
    pub current: Vec<plane::Point>,
}

/// A PGA2D point at `(x, y)` of weight one.
pub fn planar_point([x, y]: [f64; 2]) -> plane::Point {
    plane::Point::xy(x, y)
}

/// The material of the affine slice: it pairs spatial directions only, and never inverts the
/// null metric: the sum over the directions `d` of `x` and `y` of `d (plane & d)`.
pub fn planar_conductivity() -> gax::pga2d::Point<(plane::Plane,)> {
    let directions = [
        plane::Point::direction(1.0, 0.0),
        plane::Point::direction(0.0, 1.0),
    ];
    let dyad = |d: plane::Point| d * (plane::Plane::slot() & d);
    dyad(directions[0]) + dyad(directions[1])
}

impl Planar {
    pub fn new() -> Self {
        let corners = PLANAR_CORNERS.map(planar_point);
        let (grid, weights) = triangle_grid(DISPLAY_DIVISIONS);
        let points: Vec<plane::Point> = weights
            .iter()
            .map(|w| corners[0] * w[0] + corners[1] * w[1] + corners[2] * w[2])
            .collect();
        let lifted = [0, 1, 2].map(|k| corners[k] * CORNER_SCALES[k]);
        let reciprocals = plane::reciprocals(lifted);
        let (hats, gradients): (Vec<_>, Vec<_>) =
            points.iter().map(|&p| plane::hats(reciprocals, p)).unzip();
        let potential = hats
            .iter()
            .map(|h| (0..3).map(|k| h[k].s() * CORNER_VALUES[k]).sum())
            .collect();
        let conductivity = planar_conductivity();
        let current = gradients
            .iter()
            .map(|g| {
                let gradient = (0..3)
                    .map(|k| g[k] * CORNER_VALUES[k])
                    .fold(plane::Plane::zero(), |a, b| a + b);
                -conductivity.of(gradient)
            })
            .collect();
        Planar {
            grid,
            points,
            hats,
            gradients,
            potential,
            current,
        }
    }
}
