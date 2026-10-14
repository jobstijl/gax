//! The flat disks of [`Mesh`]: concentric rings, rings of nearly equilateral cells and a
//! hexagonal lattice mapped onto a circle, the last two Delaunay triangulations of their vertices.

use super::{Mesh, Vector3, delaunay, direction};
use gax::field::Field;

/// The fewest vertices on a ring of [`Mesh::concentric_disk`].
const MIN_RING: usize = 6;
/// The steps of edge springs that even out [`Mesh::triangular_disk`], and how far each moves.
const RELAXATION_STEPS: usize = 20;
const RELAXATION_RATE: f64 = 0.1;

/// The side of an equilateral triangle of the area: `area = side² sin(60°) / 2`.
fn equilateral_side(area: f64) -> f64 {
    let sine = direction(core::f64::consts::TAU / 6.0).e2();
    #[allow(clippy::disallowed_methods)] // a length from an area
    (2.0 * area / sine).sqrt()
}

impl Mesh<Vector3> {
    /// The Delaunay triangulation of vertices in the xy plane, its faces counter-clockwise.
    #[must_use]
    pub fn triangulate(vertices: Field<Vector3>) -> Self {
        let points: Vec<delaunay::Point> = vertices
            .iter()
            .map(|v| delaunay::Point::xy(v.e1(), v.e2()))
            .collect();
        let faces = delaunay::triangulate(&points);
        Mesh::new(vertices, faces)
    }

    /// A flat disk of the radius, with concentric rings about a centre vertex, each of `sectors`
    /// vertices, and a counter-clockwise triangular boundary.
    #[must_use]
    pub fn disk(radius: f64, rings: usize, sectors: usize) -> Self {
        let directions: Vec<Vector3> = (0..sectors)
            .map(|s| direction(core::f64::consts::TAU * s as f64 / sectors as f64))
            .collect();
        let ring_radius = |r: usize| radius * (r + 1) as f64 / rings as f64;
        let positions =
            (0..rings).flat_map(|r| directions.iter().map(move |&d| d * ring_radius(r)));
        let vertices = core::iter::once(Vector3::new(0.0, 0.0, 0.0))
            .chain(positions)
            .collect();

        // Each sector starts with a centre triangle; successive rings add two triangles.
        let at = |ring: usize, sector: usize| 1 + ring * sectors + sector % sectors;
        let fan = (0..sectors).map(|s| [0, at(0, s), at(0, s + 1)]);
        let quads = |rule: fn(usize, usize, usize, usize) -> [usize; 3]| {
            (1..rings).flat_map(move |r| {
                (0..sectors)
                    .map(move |s| rule(at(r - 1, s), at(r, s), at(r, s + 1), at(r - 1, s + 1)))
            })
        };
        let first = quads(|inner, outer, outer_next, _| [inner, outer, outer_next]);
        let second = quads(|inner, _, outer_next, inner_next| [inner, outer_next, inner_next]);
        Mesh::new(vertices, fan.chain(first).chain(second).collect())
    }

    /// Staggered circular rings with nearly equilateral cells: `boundaries` are increasing
    /// interface radii, ending at the disk's rim, each a ring of the mesh; `divisions` the
    /// number of radial intervals across the disk, roughly.
    #[must_use]
    pub fn concentric_disk(boundaries: &[f64], divisions: usize) -> Self {
        let rim = *boundaries.last().expect("the disk's rim");
        let spacing = rim / divisions as f64;
        let mut radii = Vec::new();
        let mut start = 0.0;
        for &end in boundaries {
            let width = end - start;
            let count = ((width / spacing).round_ties_even() as usize).max(1);
            radii.extend((1..=count).map(|i| start + width * i as f64 / count as f64));
            start = end;
        }
        // The circumference sets the population, so the centre's cells are the size of the
        // rim's: the side of an equilateral triangle whose height is the spacing.
        let edge_length = spacing / direction(core::f64::consts::TAU / 6.0).e2();
        let mut vertices = vec![Vector3::new(0.0, 0.0, 0.0)];
        for (ring, &r) in radii.iter().enumerate() {
            let population =
                MIN_RING.max((core::f64::consts::TAU * r / edge_length).round_ties_even() as usize);
            let stagger = (ring % 2) as f64 / 2.0;
            vertices.extend((0..population).map(|i| {
                let turn = core::f64::consts::TAU * (i as f64 + stagger) / population as f64;
                direction(turn) * r
            }));
        }
        Mesh::triangulate(Field::new(vertices))
    }

    /// A circular disk of nearly equilateral triangles whose interior vertices have six
    /// neighbours: the hexagonal rings of a triangular lattice mapped to circles, then evened out
    /// by edge springs.
    #[must_use]
    pub fn triangular_disk(radius: f64, divisions: usize) -> Self {
        let spacing = radius / divisions as f64;
        let d = divisions as i64;
        let axes = [direction(0.0), direction(core::f64::consts::TAU / 6.0)];
        let mut vertices = vec![Vector3::new(0.0, 0.0, 0.0)];
        let mut boundary = Vec::new();
        for column in -d..=d {
            for row in -d..=d {
                // The hexagonal ring of the lattice point.
                let layer = column.abs().max(row.abs()).max((column + row).abs());
                if layer == 0 || layer > d {
                    continue;
                }
                if layer == d {
                    boundary.push(vertices.len());
                }
                let at = axes[0] * column as f64 + axes[1] * row as f64;
                vertices.push(at.normalized().into_inner() * (spacing * layer as f64));
            }
        }
        let mut mesh = Mesh::triangulate(Field::new(vertices));

        // Equal edge springs even out the circular map's distortion. Rim vertices slide along
        // the circle; the incidence adjoint sums the edge forces at each vertex.
        let target = equilateral_side(mesh.triangle_areas().sum().s() / mesh.faces.len() as f64);
        for _ in 0..RELAXATION_STEPS {
            let force = mesh.edge_vectors().map(|e| e * (1.0 - target / e.norm()));
            let pull = &mesh.d0().reverse() * &force;
            let mut positions = &mesh.vertices - &pull.scaled(RELAXATION_RATE);
            for &b in &boundary {
                let p = &mut positions.sites_mut()[b];
                *p = p.normalized().into_inner() * radius;
            }
            mesh = Mesh::triangulate(positions);
        }
        mesh
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::assert_flat;
    use super::*;

    /// The cosine of an angle in degrees, from a rotor.
    fn cosine(degrees: f64) -> f64 {
        direction(degrees.to_radians()).e1()
    }

    /// The mean and variance of numbers.
    fn mean_variance(xs: &[f64]) -> (f64, f64) {
        let n = xs.len() as f64;
        let mean = xs.iter().sum::<f64>() / n;
        (mean, xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n)
    }

    /// Whether the spread (standard deviation) of numbers is under `share` of their mean.
    fn spread_under(xs: &[f64], share: f64) -> bool {
        let (mean, variance) = mean_variance(xs);
        variance < (share * mean).powi(2)
    }

    /// Every corner's cosine.
    fn cosines(mesh: &Mesh) -> Vec<f64> {
        mesh.corner_cosines()
            .iter()
            .flat_map(|c| c.iter().map(|c| c.s()))
            .collect()
    }

    /// Every corner's angle between `least` and `most` degrees.
    fn assert_angles_within(mesh: &Mesh, least: f64, most: f64) {
        let cosines = cosines(mesh);
        assert!(
            cosines
                .iter()
                .all(|&c| c < cosine(least) && c > cosine(most))
        );
    }

    #[test]
    fn triangular_disk_has_regular_interior_and_resolved_circular_boundary() {
        let radius = 0.1;
        for divisions in [6, 10, 16, 24, 32] {
            let mesh = Mesh::triangular_disk(radius, divisions);
            let boundary = mesh.boundary_vertices();
            let valences = mesh.valences();
            let lengths: Vec<f64> = mesh.edge_vectors().iter().map(|e| e.norm()).collect();
            let areas: Vec<f64> = mesh.triangle_areas().iter().map(|a| a.s()).collect();
            let rim_areas: Vec<f64> = mesh
                .face_centers()
                .iter()
                .zip(&areas)
                .filter(|(c, _)| c.norm() > 0.75 * radius)
                .map(|(_, a)| *a)
                .collect();

            // Every interior vertex has six neighbours; the boundary follows the circle.
            for v in 0..valences.len() {
                if boundary[v] {
                    assert!((mesh.vertices.site(v).norm() - radius).abs() < 1e-12);
                } else {
                    assert_eq!(valences[v], 6, "{divisions}");
                }
            }
            assert_flat(&mesh, 1e-12);
            // Edge lengths and areas stay uniform through the rim, without thin triangles.
            assert!(spread_under(&lengths, 0.1) && spread_under(&areas, 0.1));
            assert!(spread_under(&rim_areas, 0.15));
            assert_angles_within(&mesh, 35.0, 95.0);
        }
    }

    #[test]
    fn concentric_disk_follows_material_interfaces_without_thin_cells() {
        let boundaries = [0.02, 0.085, 0.1];
        let tolerance = 1e-12;
        for divisions in [10, 24, 32] {
            let mesh = Mesh::concentric_disk(&boundaries, divisions);
            let radii: Vec<f64> = mesh.vertices.iter().map(|v| v.norm()).collect();
            // Every prescribed circle is an unbroken chain of mesh edges; no face straddles it.
            for radius in boundaries {
                let on_ring: Vec<bool> = radii
                    .iter()
                    .map(|r| (r - radius).abs() < tolerance)
                    .collect();
                assert!(on_ring.iter().filter(|&&on| on).count() >= MIN_RING);
                let mut degrees = vec![0; radii.len()];
                for &[a, b] in &mesh.edges {
                    if on_ring[a] && on_ring[b] {
                        degrees[a] += 1;
                        degrees[b] += 1;
                    }
                    let (low, high) = (radii[a].min(radii[b]), radii[a].max(radii[b]));
                    assert!(!(low < radius - tolerance && high > radius + tolerance));
                }
                assert!(
                    (0..radii.len())
                        .filter(|&v| on_ring[v])
                        .all(|v| degrees[v] == 2)
                );
            }
            assert_flat(&mesh, tolerance);
            assert_angles_within(&mesh, 30.0, 105.0);
            let areas: Vec<f64> = mesh.triangle_areas().iter().map(|a| a.s()).collect();
            assert!(spread_under(&areas, 0.2));
        }
    }
}
