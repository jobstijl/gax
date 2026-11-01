//! numga's scenes: a scalene triangle and its sample lattice, a diamond of two triangles, a
//! scalene tetrahedron and a sheared one, disks of nearly equilateral triangles, an obtuse
//! diamond, and the two materials.

use super::whitney::*;
use crate::simplicial::hex_disk;
use gax::field::Field;
use gax::mesh::Topology;

pub const TRIANGLE_SUBDIVISIONS: usize = 8;
pub const STAR_DIVISIONS: usize = 2;
pub const TET_SUBDIVISIONS: usize = 6;
pub const DISK_DIVISIONS: usize = 9;
pub const DISK_RADIUS: f64 = 1.0;
/// The electrodes: the rim vertices farther than this from the `y` axis.
pub const ELECTRODE_THRESHOLD: f64 = 0.8;
/// The fibre's angle from `x`, and the principal conductivities along and across it of the two
/// materials: isotropic, and ten times as conductive along the fibre.
pub const FIBRE_ANGLE: f64 = core::f64::consts::FRAC_PI_6;
pub const PRINCIPAL: [[f64; 2]; 2] = [[1.0, 1.0], [10.0, 1.0]];

/// The interior points of the barycentric lattice of `n` steps across a triangle.
pub fn triangle_samples(corners: [Point; 3], n: usize) -> Vec<Point> {
    let mut out = Vec::new();
    for first in 1..n {
        for second in 1..n {
            if first + second < n {
                let w = [n - first - second, first, second].map(|k| k as f64 / n as f64);
                out.push(corners[0] * w[0] + corners[1] * w[1] + corners[2] * w[2]);
            }
        }
    }
    out
}

/// A scalene triangle and its interior samples.
pub fn single_triangle(n: usize) -> ([Point; 3], Vec<Point>) {
    let corners = [point(0.0, 0.0), point(1.2, 0.12), point(0.38, 0.96)];
    (corners, triangle_samples(corners, n))
}

/// Two triangles sharing a vertical interior edge, and the interior samples of each.
pub fn edge_diamond(n: usize) -> (Topology<3>, Field<Point>, Vec<Vec<Point>>) {
    let vertices: Field<Point> = [
        point(0.0, -0.72),
        point(0.0, 0.72),
        point(-0.95, 0.14),
        point(0.92, -0.12),
    ]
    .into_iter()
    .collect();
    let mesh = Topology::new(4, &[[0, 1, 2], [1, 0, 3]]).expect("the diamond");
    let samples = mesh
        .simplices(2)
        .map(|f| triangle_samples([f[0], f[1], f[2]].map(|v| vertices.site(v)), n))
        .collect();
    (mesh, vertices, samples)
}

/// The interior points of the barycentric lattice of `n` steps across a tetrahedron.
pub fn tetrahedron_samples(corners: [SpacePoint; 4], n: usize) -> Vec<SpacePoint> {
    let mut out = Vec::new();
    for first in 1..n {
        for second in 1..n {
            for third in 1..n {
                if first + second + third < n {
                    let w = [n - first - second - third, first, second, third]
                        .map(|k| k as f64 / n as f64);
                    out.push(
                        (0..4)
                            .map(|k| corners[k] * w[k])
                            .reduce(|a, b| a + b)
                            .expect("four corners"),
                    );
                }
            }
        }
    }
    out
}

/// A mesh of one positively oriented scalene tetrahedron, its vertices, and its interior
/// samples.
pub fn single_tetrahedron(n: usize) -> (Topology<4>, Field<SpacePoint>, Vec<SpacePoint>) {
    let vertices: Field<SpacePoint> = [
        space_point(-0.56, -0.38, -0.34),
        space_point(0.62, -0.28, -0.34),
        space_point(-0.05, 0.68, -0.34),
        space_point(0.03, 0.02, 0.66),
    ]
    .into_iter()
    .collect();
    let mesh = Topology::new(4, &[[0, 1, 2, 3]]).expect("the tetrahedron");
    let corners = core::array::from_fn(|k| vertices.site(k));
    (mesh, vertices, tetrahedron_samples(corners, n))
}

/// The vertices of a strongly sheared, stretched and shifted tetrahedron of the same sense.
#[cfg(test)]
pub fn deformed_tetrahedron() -> Field<SpacePoint> {
    [
        space_point(0.40, -1.10, 0.25),
        space_point(2.30, -0.35, -0.15),
        space_point(0.85, 1.65, 0.55),
        space_point(1.15, 0.20, 2.45),
    ]
    .into_iter()
    .collect()
}

/// A disk of nearly equilateral triangles with its first vertex at the centre.
pub fn disk(radius: f64, divisions: usize) -> (Topology<3>, Field<Point>) {
    let (mesh, coordinates) = hex_disk(radius, divisions);
    (mesh, coordinates.map(|v| v.hodge()))
}

/// Two obtuse triangles sharing a long horizontal edge, where circumcentric DEC's Hodge weight
/// is negative.
#[cfg(test)]
pub fn obtuse_diamond() -> (Topology<3>, Field<Point>) {
    let vertices = [
        point(0.0, 0.0),
        point(2.0, 0.0),
        point(1.0, 0.4),
        point(1.0, -0.4),
    ];
    let mesh = Topology::new(4, &[[0, 1, 2], [1, 0, 3]]).expect("the obtuse diamond");
    (mesh, vertices.into_iter().collect())
}

/// The fibre: the unit point at `FIBRE_ANGLE` from `x`, turned there by a rotor.
pub fn fibre() -> Point {
    Bivector::new(-FIBRE_ANGLE / 2.0).exp() >> point(1.0, 0.0)
}

/// The two materials: isotropic, and ten times as conductive along the fibre.
pub fn conductivities() -> [Conductivity; 2] {
    PRINCIPAL.map(|[along, across]| anisotropic(fibre(), along, across))
}

/// The electrodes, the rim vertices farther than `ELECTRODE_THRESHOLD` from the `y` axis, and
/// the potential that holds them at their `x` (zero elsewhere).
pub fn electrodes(mesh: &Topology<3>, vertices: &Field<Point>) -> (Vec<usize>, Field<Scalar>) {
    let rim = mesh.exterior(0);
    let x = |v: usize| coordinates(vertices.site(v))[0];
    let held: Vec<usize> = rim
        .into_iter()
        .filter(|&v| x(v).abs() > ELECTRODE_THRESHOLD)
        .collect();
    let prescribed = Field::from_fn(vertices.len(), |v| {
        Scalar::new(if held.contains(&v) { x(v) } else { 0.0 })
    });
    (held, prescribed)
}
