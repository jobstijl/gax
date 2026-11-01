//! numga's scenes: holonomy around latitude circles on the sphere; vector heat and the smoothest
//! tangent fields on a torus; parallel fields of trivial connections on spheres and tori; stripes
//! along them and across cross fields.

use super::connections::*;
use crate::simplicial::{icosphere, torus, vertex_faces};
use gax::field::Field;
use gax::mesh::Topology;

/// The sphere and its latitude circles, 30, 60 and 90 degrees from the north pole.
pub const SPHERE_LEVELS: usize = 3;
pub const COLATITUDES: [f64; 3] = [
    core::f64::consts::FRAC_PI_6,
    core::f64::consts::FRAC_PI_3,
    core::f64::consts::FRAC_PI_2,
];
pub const RING_STEPS: usize = 210;
/// The torus of every scene.
pub const MAJOR_RADIUS: f64 = 2.0;
pub const MINOR_RADIUS: f64 = 0.75;
/// The torus of vector heat and the smoothest fields, the tangent bump's width, the heat steps,
/// and the modes.
pub const HEAT_MAJOR: usize = 28;
pub const HEAT_MINOR: usize = 14;
pub const SOURCE_WIDTH: f64 = 0.55;
pub const STEP_TIME: f64 = 0.1;
pub const HEAT_STEPS: usize = 36;
pub const MODES: usize = 4;
/// For each scene on the sphere, the icosahedron's vertices whose first faces carry a
/// singularity, and their indices; vertices 9 and 10 are opposite.
pub const SPHERE_CORNERS: [&[usize]; 3] = [&[9, 10], &[9], &[9, 4, 8, 3]];
pub const SPHERE_INDICES: [&[f64]; 3] = [&[1.0, 1.0], &[2.0], &[1.0, 1.0, 1.0, -1.0]];
/// The torus of the designed fields, and for each scene its singular faces (by their place
/// around the axis and the tube) and their indices.
pub const FIELD_MAJOR: usize = 40;
pub const FIELD_MINOR: usize = 20;
pub const FIELD_TORUS_PLACES: [&[[usize; 2]]; 2] = [&[], &[[32, 2], [0, 2]]];
pub const TORUS_INDICES: [&[f64]; 2] = [&[], &[1.0, -1.0]];
/// The stripes: the sphere and its frequency, the torus and its singular places and frequency.
pub const STRIPE_SPHERE_LEVELS: usize = 4;
pub const SPHERE_FREQUENCY: f64 = 20.0;
pub const STRIPE_MAJOR: usize = 80;
pub const STRIPE_MINOR: usize = 40;
pub const STRIPE_TORUS_PLACES: [&[[usize; 2]]; 2] = [&[], &[[64, 4], [0, 4]]];
pub const TORUS_FREQUENCY: f64 = 12.0;
/// The cross fields' singularities: the cube's corners, each turning by a quarter, and the
/// corners of a tetrahedron in it, each by a half; and their frequency.
pub const CUBE: [[f64; 3]; 8] = [
    [1.0, 1.0, 1.0],
    [1.0, 1.0, -1.0],
    [1.0, -1.0, 1.0],
    [1.0, -1.0, -1.0],
    [-1.0, 1.0, 1.0],
    [-1.0, 1.0, -1.0],
    [-1.0, -1.0, 1.0],
    [-1.0, -1.0, -1.0],
];
pub const TETRAHEDRON: [[f64; 3]; 4] = [
    [1.0, 1.0, 1.0],
    [1.0, -1.0, -1.0],
    [-1.0, 1.0, -1.0],
    [-1.0, -1.0, 1.0],
];
pub const CROSS_INDICES: [f64; 2] = [0.25, 0.5];
pub const CROSS_FREQUENCY: f64 = 20.0;
/// The weight keeping fields out of the normal bundle.
pub const NORMAL_STIFFNESS: f64 = 100.0;

/// A surface with what its connections need: normals, vertex areas, cotangent weights.
pub struct Surface {
    pub mesh: Topology<3>,
    pub vertices: Field<Point>,
    pub normals: Field<Normal>,
    pub areas: Field<Scalar>,
    pub weights: Field<Scalar>,
}

impl Surface {
    pub fn new(mesh: Topology<3>, coordinates: &Field<gax::vga3d::Vector>) -> Surface {
        let vertices = coordinates.map(|v| v.hodge());
        let normals = vertex_normals(&mesh, &vertices);
        let areas = vertex_areas(&mesh, &vertices);
        let weights = cotangent_weights(&mesh, &vertices, &vec![1.0; mesh.len(2)]);
        Surface {
            mesh,
            vertices,
            normals,
            areas,
            weights,
        }
    }

    pub fn sphere(levels: usize) -> Surface {
        let (mesh, vertices) = icosphere(levels);
        Surface::new(mesh, &vertices)
    }

    pub fn torus(major: usize, minor: usize) -> Surface {
        let (mesh, vertices) = torus(MAJOR_RADIUS, MINOR_RADIUS, major, minor);
        Surface::new(mesh, &vertices)
    }

    /// The torus's two generators: once around the tube, and once around the axis along the
    /// outer equator.
    pub fn torus_generators(major: usize, minor: usize) -> Vec<Vec<usize>> {
        vec![
            (0..minor).collect(),
            (0..major).map(|u| u * minor).collect(),
        ]
    }
}

/// The holonomy scene: the latitude circles' normals and their starting vectors.
pub fn holonomy() -> (Vec<Vec<Normal>>, Vec<Point>) {
    latitude_rings(&COLATITUDES, RING_STEPS)
}

/// The vectors carried around each circle `[ring][step]`.
pub fn carried(rings: &[Vec<Normal>], starts: &[Point]) -> Vec<Vec<Point>> {
    (0..starts.len())
        .map(|r| {
            let along: Vec<Normal> = rings.iter().map(|step| step[r]).collect();
            transported(&along, starts[r])
        })
        .collect()
}

/// Vector heat and the smoothest tangent fields on the torus.
#[cfg_attr(not(test), allow(dead_code))] // numga's scenes, checked by the tests
pub struct TorusFields {
    pub surface: Surface,
    pub waves: Vec<Field<Point>>,
    pub transported: Field<Point>,
    pub modes: Vec<Field<Point>>,
    pub energies: Vec<f64>,
    pub covariant: gax::field::SparseMap<PointMap>,
}

pub fn torus_fields() -> TorusFields {
    let surface = Surface::torus(HEAT_MAJOR, HEAT_MINOR);
    let centre = point(MAJOR_RADIUS + MINOR_RADIUS, 0.0, 0.0);
    let direction = point(0.0, 1.0, 1.0).normalized();
    let falloff = -0.5 / (SOURCE_WIDTH * SOURCE_WIDTH);
    // A bump of the direction less its part along each normal.
    let source = surface.vertices.zip_with(&surface.normals, |v, n| {
        let weight = (falloff * (v - centre).norm_squared()).exp();
        (direction - along_normal(n).of(direction)) * weight
    });
    let no_turns = vec![0.0; surface.mesh.len(1)];
    let forward = adjusted_transport(&surface.mesh, &surface.normals, &no_turns);
    let (covariant, laplacian) = connection(&surface.mesh, &forward, &surface.weights);
    let heat = heat_transport(&laplacian, &surface.areas, &source, STEP_TIME, HEAT_STEPS);
    let (energies, modes) = low_modes(
        &laplacian,
        &surface.normals,
        &surface.areas,
        NORMAL_STIFFNESS,
        MODES,
    );
    let transported = heat.last().expect("steps").1.clone();
    TorusFields {
        waves: heat.into_iter().map(|(field, _)| field).collect(),
        transported,
        modes,
        energies,
        covariant,
        surface,
    }
}

/// A design: the trivial connection's extra turns, its unit parallel field, the energies of its
/// two parallel fields, and the circulation's residual.
#[cfg_attr(not(test), allow(dead_code))] // numga's scenes, checked by the tests
pub struct Design {
    pub angles: Vec<f64>,
    pub field: Field<Point>,
    pub energies: Vec<f64>,
    pub residual: Vec<f64>,
}

pub fn design(s: &Surface, indices: &[f64], generators: &[Vec<usize>]) -> Design {
    let (angles, circulation, target) =
        trivial_connection(&s.mesh, &s.normals, indices, generators);
    let forward = adjusted_transport(&s.mesh, &s.normals, &angles);
    let (_, laplacian) = connection(&s.mesh, &forward, &s.weights);
    let (energies, fields) = low_modes(&laplacian, &s.normals, &s.areas, NORMAL_STIFFNESS, 2);
    let field = turned_to_x(s, &fields[0]).map(|p| p.normalized());
    let residual = residual(&circulation, &angles, &target);
    Design {
        angles,
        field,
        energies,
        residual,
    }
}

/// A parallel field turned, by one angle everywhere, to point along x's tangent part at the
/// first vertex: a trivial connection's parallel fields are a plane of them (each with its
/// quarter turn), any of which an eigensolver may return, and this picks one regardless of the
/// solver (numga keeps scipy's).
pub fn turned_to_x(s: &Surface, field: &Field<Point>) -> Field<Point> {
    let quarter = field.zip_with(&s.normals, |p, n| p.commutator(plane(n)));
    let x = point(1.0, 0.0, 0.0);
    let (f, q) = (field.site(0), quarter.site(0));
    let turn = gax::vga2d::Vector::new(dot(x, f), dot(x, q)).normalized();
    field.zip_with(&quarter, |f, q| f * turn.e1() + q * turn.e2())
}

/// Every face's index: the given ones on the singular faces, none elsewhere.
pub fn face_indices(s: &Surface, singular: &[usize], indices: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; s.mesh.len(2)];
    for (&f, &k) in singular.iter().zip(indices) {
        out[f] = k;
    }
    out
}

/// The singular faces of the sphere's scenes: the first face at each chosen vertex.
pub fn sphere_singulars(s: &Surface) -> Vec<Vec<usize>> {
    let first = vertex_faces(&s.mesh);
    SPHERE_CORNERS
        .iter()
        .map(|c| c.iter().map(|&v| first[v]).collect())
        .collect()
}

/// The singular faces of a torus's scenes, from their places.
pub fn torus_singulars(places: &[&[[usize; 2]]], minor: usize) -> Vec<Vec<usize>> {
    places
        .iter()
        .map(|p| p.iter().map(|[u, v]| u * minor + v).collect())
        .collect()
}

/// The designed fields on the sphere and their singular faces.
pub fn field_spheres() -> (Surface, Vec<(Design, Vec<usize>)>) {
    let s = Surface::sphere(SPHERE_LEVELS);
    let designs = sphere_singulars(&s)
        .into_iter()
        .zip(SPHERE_INDICES)
        .map(|(singular, indices)| {
            (
                design(&s, &face_indices(&s, &singular, indices), &[]),
                singular,
            )
        })
        .collect();
    (s, designs)
}

/// The designed fields on the torus and their singular faces.
pub fn field_tori() -> (Surface, Vec<(Design, Vec<usize>)>) {
    let s = Surface::torus(FIELD_MAJOR, FIELD_MINOR);
    let generators = Surface::torus_generators(FIELD_MAJOR, FIELD_MINOR);
    let designs = torus_singulars(&FIELD_TORUS_PLACES, FIELD_MINOR)
        .into_iter()
        .zip(TORUS_INDICES)
        .map(|(singular, indices)| {
            (
                design(&s, &face_indices(&s, &singular, indices), &generators),
                singular,
            )
        })
        .collect();
    (s, designs)
}

/// A field turned about every normal by `angle`: still parallel.
pub fn turned(s: &Surface, field: &Field<Point>, angle: f64) -> Field<Point> {
    field.zip_with(&s.normals, |p, n| {
        (plane(n) * (-angle / 2.0)).exp().into_inner() >> p
    })
}

/// A stripe pattern along a field: its line, its unit stripe field, and the edges' targets.
#[cfg_attr(not(test), allow(dead_code))] // numga's scenes, checked by the tests
pub struct Stripes {
    pub line: Field<Point>,
    pub stripes: Field<Point>,
    pub advance: Vec<f64>,
    pub energies: Vec<f64>,
}

/// The stripes along a direction field parallel for the connection of the given turns: the
/// smoothest tangent field of that connection with each edge's phase advance added, the faces
/// where the direction field turns left out of the weights.
pub fn along(
    s: &Surface,
    angles: &[f64],
    field: &Field<Point>,
    regular: &[f64],
    frequency: f64,
) -> Stripes {
    let steps = &s.mesh.d::<Scalar>(0) * &s.vertices;
    let means = &s.mesh.means::<Scalar>(1) * field;
    let advance: Vec<f64> = means
        .iter()
        .zip(steps.iter())
        .map(|(m, e)| dot(m, e) * frequency)
        .collect();
    let total: Vec<f64> = angles.iter().zip(&advance).map(|(a, b)| a + b).collect();
    let forward = adjusted_transport(&s.mesh, &s.normals, &total);
    let weights = cotangent_weights(&s.mesh, &s.vertices, regular);
    let (_, laplacian) = connection(&s.mesh, &forward, &weights);
    let (energies, fields) = low_modes(&laplacian, &s.normals, &s.areas, NORMAL_STIFFNESS, 2);
    Stripes {
        line: field.clone(),
        stripes: fields[0].map(|p| p.normalized()),
        advance,
        energies,
    }
}

/// The stripes along a design's field and along its quarter turn.
pub fn pattern(
    s: &Surface,
    indices: &[f64],
    generators: &[Vec<usize>],
    frequency: f64,
) -> [Stripes; 2] {
    let d = design(s, indices, generators);
    let regular: Vec<f64> = indices
        .iter()
        .map(|&k| if k == 0.0 { 1.0 } else { 0.0 })
        .collect();
    let quarter = d.field.zip_with(&s.normals, |p, n| p.commutator(plane(n)));
    [d.field.clone(), quarter].map(|f| along(s, &d.angles, &f, &regular, frequency))
}

pub fn stripe_spheres() -> (Surface, Vec<[Stripes; 2]>) {
    let s = Surface::sphere(STRIPE_SPHERE_LEVELS);
    let patterns = sphere_singulars(&s)
        .into_iter()
        .zip(SPHERE_INDICES)
        .map(|(singular, indices)| {
            pattern(
                &s,
                &face_indices(&s, &singular, indices),
                &[],
                SPHERE_FREQUENCY,
            )
        })
        .collect();
    (s, patterns)
}

pub fn stripe_tori() -> (Surface, Vec<[Stripes; 2]>) {
    let s = Surface::torus(STRIPE_MAJOR, STRIPE_MINOR);
    let generators = Surface::torus_generators(STRIPE_MAJOR, STRIPE_MINOR);
    let patterns = torus_singulars(&STRIPE_TORUS_PLACES, STRIPE_MINOR)
        .into_iter()
        .zip(TORUS_INDICES)
        .map(|(singular, indices)| {
            pattern(
                &s,
                &face_indices(&s, &singular, indices),
                &generators,
                TORUS_FREQUENCY,
            )
        })
        .collect();
    (s, patterns)
}

/// The face whose centre lies furthest along each direction.
pub fn nearest_faces(s: &Surface, directions: &[[f64; 3]]) -> Vec<usize> {
    let centres = &s.mesh.means::<Scalar>(2) * &s.vertices;
    directions
        .iter()
        .map(|&[x, y, z]| {
            let d = gax::vga3d::Vector::new(x, y, z);
            (0..centres.len())
                .max_by(|&a, &b| {
                    (d & centres.site(a))
                        .s()
                        .total_cmp(&(d & centres.site(b)).s())
                })
                .expect("faces")
        })
        .collect()
}

/// The cross fields: for each set of corners, its extra turns and its two families of stripes.
pub fn crosses() -> (Surface, Vec<(Vec<f64>, Cross)>) {
    let s = Surface::sphere(STRIPE_SPHERE_LEVELS);
    let corners: [&[[f64; 3]]; 2] = [&CUBE, &TETRAHEDRON];
    let scenes = corners
        .iter()
        .zip(CROSS_INDICES)
        .map(|(c, index)| {
            let singular = nearest_faces(&s, c);
            let indices = face_indices(&s, &singular, &vec![index; singular.len()]);
            let (angles, _, _) = trivial_connection(&s.mesh, &s.normals, &indices, &[]);
            let cross = carried_out(&s.mesh, &s.vertices, &s.normals, &angles);
            let regular: Vec<f64> = indices
                .iter()
                .map(|&k| if k == 0.0 { 1.0 } else { 0.0 })
                .collect();
            let pattern = cross_stripes(
                &s.mesh,
                &s.vertices,
                &s.normals,
                &s.areas,
                &angles,
                &cross,
                &regular,
                CROSS_FREQUENCY,
                NORMAL_STIFFNESS,
            );
            (angles, pattern)
        })
        .collect();
    (s, scenes)
}

/// The phase of a stripe field along its line at each vertex: the angle from the stripe field
/// to the line about the normal.
#[cfg(test)]
pub fn phases(s: &Surface, line: &Field<Point>, stripes: &Field<Point>) -> Vec<f64> {
    (0..s.vertices.len())
        .map(|v| {
            let (l, t, n) = (line.site(v), stripes.site(v), s.normals.site(v));
            let sine = (l.commutator(t) | plane(n)).s();
            let cosine = dot(t, l);
            gax::vga2d::Vector::new(1.0, 0.0).angle_to(gax::vga2d::Vector::new(cosine, sine))
        })
        .collect()
}

/// How far the stripe phase across every edge misses its target, an angle in `[0, π]`.
#[cfg(test)]
pub fn stripe_mismatch(
    s: &Surface,
    line: &Field<Point>,
    stripes: &Field<Point>,
    advance: &[f64],
) -> Vec<f64> {
    let phase = phases(s, line, stripes);
    crate::simplicial::edges(&s.mesh)
        .iter()
        .zip(advance)
        .map(|(&[tail, head], &target)| {
            let step = phase[head] - phase[tail] - target;
            let wrapped = (step + core::f64::consts::PI).rem_euclid(core::f64::consts::TAU)
                - core::f64::consts::PI;
            wrapped.abs()
        })
        .collect()
}

/// The median of numbers.
#[cfg(test)]
pub fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    let n = values.len();
    if n % 2 == 1 {
        values[n / 2]
    } else {
        0.5 * (values[n / 2 - 1] + values[n / 2])
    }
}
