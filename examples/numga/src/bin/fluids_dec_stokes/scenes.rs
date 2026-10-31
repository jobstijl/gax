//! numga's scene: a channel six long and two high, a jittered triangular lattice inside it
//! between evenly spaced walls, an offset circular obstacle ringed so that Delaunay keeps its
//! sides, and the Delaunay triangulation with the obstacle cut out.

use super::stokes::*;
use gax::field::Field;
use gax::mesh::Topology;
use gax_numga_examples::rng::{Draw, rng};
use gax_numga_examples::signal::wave;

pub const LENGTH: f64 = 6.0;
pub const HEIGHT: f64 = 2.0;
pub const OBSTACLE_CENTRE: [f64; 2] = [2.5, 0.2];
pub const OBSTACLE_RADIUS: f64 = 0.45;
pub const DIVISIONS: usize = 18;
pub const VISCOSITY: f64 = 1.0;
pub const MEAN_SPEED: f64 = 1.0;
/// numga's seed, for `rand`'s generator: numpy's draw cannot be reproduced (README), and this
/// jitter's Poiseuille errors are numga's to a few percent (velocity, pressure, vorticity:
/// 0.069, 0.146, 0.074 at 8 divisions and 0.036, 0.043, 0.023 at 16, against numga's 0.070,
/// 0.144, 0.062 and 0.036, 0.042, 0.024), within its bounds by the same margins.
pub const MESH_SEED: u64 = 8294;

/// A channel's mesh: the triangles, the vertices, and the edges of the inlet, the outlet and the
/// walls (the obstacle's included).
pub struct Channel {
    pub mesh: Topology<3>,
    pub vertices: Field<Point>,
    pub inlet: Vec<usize>,
    pub outlet: Vec<usize>,
    #[cfg_attr(not(test), allow(dead_code))] // the tests check that no flow crosses them
    pub walls: Vec<usize>,
}

/// The rectangle's boundary vertices (the bottom wall, the top wall, then the inner vertices of
/// the inlet and of the outlet), a slightly jittered triangular lattice inside, and the
/// boundary vertices of the inlet and of the outlet.
fn channel_vertices(divisions: usize) -> (Vec<Point>, Vec<Point>, Vec<usize>, Vec<usize>) {
    let spacing = HEIGHT / divisions as f64;
    let n_length = (LENGTH / spacing).ceil() as usize;
    let along: Vec<f64> = (0..=n_length)
        .map(|k| LENGTH * k as f64 / n_length as f64)
        .collect();
    let across: Vec<f64> = (0..=divisions)
        .map(|k| -HEIGHT / 2.0 + HEIGHT * k as f64 / divisions as f64)
        .collect();
    let mut boundary: Vec<Point> = along.iter().map(|&x| point(x, -HEIGHT / 2.0)).collect();
    boundary.extend(along.iter().map(|&x| point(x, HEIGHT / 2.0)));
    boundary.extend(across[1..divisions].iter().map(|&y| point(0.0, y)));
    boundary.extend(across[1..divisions].iter().map(|&y| point(LENGTH, y)));
    let left = 2 * (n_length + 1);
    let right = left + divisions - 1;
    let inlet = [0, n_length + 1].into_iter().chain(left..right).collect();
    let outlet = [n_length, 2 * n_length + 1]
        .into_iter()
        .chain(right..right + divisions - 1)
        .collect();

    // A gap beside the straight walls keeps each boundary edge's circumcentric dual inside.
    let (wall_spacing, jitter) = (0.85, 0.08);
    let row_step = spacing * wave(core::f64::consts::FRAC_PI_3);
    let rows: Vec<f64> = (0..)
        .map(|k| wall_spacing * spacing + k as f64 * row_step)
        .take_while(|&y| y < HEIGHT - 0.7 * spacing)
        .collect();
    let columns: Vec<f64> = (0..)
        .map(|k| wall_spacing * spacing + k as f64 * spacing)
        .take_while(|&x| x < LENGTH - 0.7 * spacing)
        .collect();
    let mut random = rng(MESH_SEED);
    let mut interior = Vec::new();
    for (r, &y) in rows.iter().enumerate() {
        for &x in &columns {
            let x = x + if r % 2 == 1 { spacing / 2.0 } else { 0.0 };
            let x = x + random.range(-jitter, jitter) * spacing;
            let y = y + random.range(-jitter, jitter) * spacing;
            if x < LENGTH - 0.7 * spacing {
                interior.push(point(x, y - HEIGHT / 2.0));
            }
        }
    }
    (boundary, interior, inlet, outlet)
}

/// The Delaunay triangulation of the vertices, the triangles with every corner in `hole` left
/// out, and the boundary edges sorted into the inlet's, the outlet's and the walls'.
fn triangulate(vertices: Vec<Point>, hole: &[usize], inlet: &[usize], outlet: &[usize]) -> Channel {
    let placed: Field<gax::vga3d::Vector> = vertices
        .iter()
        .map(|&p| {
            let [x, y] = coordinates(p);
            gax::vga3d::Vector::new(x, y, 0.0)
        })
        .collect();
    let delaunay = gax_numga_examples::mesh::Mesh::triangulate(placed);
    let faces: Vec<[usize; 3]> = delaunay
        .faces
        .into_iter()
        .filter(|f| !f.iter().all(|v| hole.contains(v)))
        .collect();
    let mesh = Topology::new(vertices.len(), &faces).expect("the channel");
    let rim = boundary_edges(&mesh);
    let within = |set: &[usize]| -> Vec<usize> {
        rim.iter()
            .copied()
            .filter(|&e| mesh.simplex(1, e).iter().all(|v| set.contains(v)))
            .collect()
    };
    let (inlet_edges, outlet_edges) = (within(inlet), within(outlet));
    let walls = rim
        .iter()
        .filter(|e| !inlet_edges.contains(e) && !outlet_edges.contains(e))
        .copied()
        .collect();
    Channel {
        mesh,
        vertices: vertices.into_iter().collect(),
        inlet: inlet_edges,
        outlet: outlet_edges,
        walls,
    }
}

/// The channel around the obstacle: a polygon of vertices on its circle, a ring just outside so
/// that each side's diameter circle stays empty and Delaunay keeps it (its radius varying, so
/// that no four vertices share a circle), and the lattice cleared around it.
pub fn channel(divisions: usize) -> Channel {
    let (boundary, interior, inlet, outlet) = channel_vertices(divisions);
    let spacing = HEIGHT / divisions as f64;
    let centre = point(OBSTACLE_CENTRE[0], OBSTACLE_CENTRE[1]);
    let count = (core::f64::consts::TAU * OBSTACLE_RADIUS / spacing).ceil() as usize;
    let turn =
        |angle: f64| (gax::vga2d::Bivector::new(1.0) * (-angle / 2.0)).exp() >> point(1.0, 0.0);
    let angles: Vec<f64> = (0..count)
        .map(|k| core::f64::consts::TAU * k as f64 / count as f64)
        .collect();
    let obstacle = angles.iter().map(|&a| centre + turn(a) * OBSTACLE_RADIUS);
    let ring = angles.iter().map(|&a| {
        let radius = OBSTACLE_RADIUS + spacing * (0.85 + 0.06 * wave(3.0 * a + 0.3));
        centre + turn(a + core::f64::consts::PI / count as f64) * radius
    });
    let interior = interior
        .into_iter()
        .filter(|&p| (p - centre).norm() > OBSTACLE_RADIUS + 1.55 * spacing);
    let hole: Vec<usize> = (boundary.len()..boundary.len() + count).collect();
    let vertices: Vec<Point> = boundary
        .into_iter()
        .chain(obstacle)
        .chain(ring)
        .chain(interior)
        .collect();
    triangulate(vertices, &hole, &inlet, &outlet)
}

/// The same channel without the obstacle, for its parabolic flow.
#[cfg(test)]
pub fn plain_channel(divisions: usize) -> Channel {
    let (boundary, interior, inlet, outlet) = channel_vertices(divisions);
    let vertices = boundary.into_iter().chain(interior).collect();
    triangulate(vertices, &[], &inlet, &outlet)
}

/// The parabolic inflow and outflow of mean speed `MEAN_SPEED`: its streamfunction's
/// differences along the inlet's and the outlet's edges, nothing elsewhere.
pub fn prescribed_flux(operators: &Operators, channel: &Channel) -> Field<Scalar> {
    let stream = channel.vertices.map(|p| {
        let [_, y] = coordinates(p);
        Scalar::new(1.5 * MEAN_SPEED * (y - 4.0 * y * y * y / (3.0 * HEIGHT * HEIGHT)))
    });
    let differences = operators.d0.of(&stream);
    let ends: Vec<usize> = channel
        .inlet
        .iter()
        .chain(&channel.outlet)
        .copied()
        .collect();
    Field::from_fn(differences.len(), |e| {
        if ends.contains(&e) {
            differences.site(e)
        } else {
            Scalar::new(0.0)
        }
    })
}

/// The flow through a channel, its edges oriented by `signs`.
pub fn flow(channel: &Channel, signs: &[f64]) -> (Operators, Field<Scalar>, Flow) {
    let operators = Operators::new(&channel.mesh, &channel.vertices, signs);
    let prescribed = prescribed_flux(&operators, channel);
    let circulation = operators.dual_patches.map(|p| p * 0.0);
    let fixed = boundary_edges(&channel.mesh);
    let result = stokes(&operators, VISCOSITY, &prescribed, &fixed, &circulation);
    (operators, prescribed, result)
}
