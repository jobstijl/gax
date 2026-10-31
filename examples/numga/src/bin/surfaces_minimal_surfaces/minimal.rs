//! The math: triangles' corner gradients, the cotangent Laplacian assembled from them, the
//! damped harmonic relaxation, the harmonic conjugate and Bonnet's family; and the scenes'
//! surfaces. Points are VGA3D antivectors, intrinsic bivectors (numga's `Point`).

use gax::field::{Field, SparseMap};
use gax::vga3d::{Bivector, Scalar, Vector, intrinsic};
use gax_numga_examples::mesh::Mesh;
use gax_numga_examples::signal::phasor;

/// A point: the antivector of a direction from the origin.
pub type Point = intrinsic::Bivector;
/// A triangle's patch, the join of two of its sides: the direction square to it, as long as
/// twice its area (numga's `Patch`).
pub type Patch = intrinsic::Vector;
/// A corner's hat gradient on its triangle: a plane (numga's `Plane+`).
pub type Gradient = Vector;

/// The point a vector reaches: numga's `dual`, the metric one.
pub fn point(v: Vector) -> Point {
    v.hodge()
}

/// The unit points along `x` and `z`.
pub fn x_point() -> Point {
    point(Vector::new(1.0, 0.0, 0.0))
}
pub fn z_point() -> Point {
    point(Vector::new(0.0, 0.0, 1.0))
}

/// One triangle: the side facing each corner, in the face's sense, its patch (twice its area
/// along its normal), its area, and each corner's hat gradient, which rises from zero along the
/// facing side to one at the corner.
#[derive(Clone, Copy, Debug)]
pub struct Triangle {
    pub sides: [Point; 3],
    pub patch: Patch,
    pub area: f64,
    pub gradients: [Gradient; 3],
}

impl Triangle {
    /// The triangle of three corners, counter-clockwise.
    pub fn new(corners: [Point; 3]) -> Triangle {
        let sides = [0, 1, 2].map(|k| corners[(k + 2) % 3] - corners[(k + 1) % 3]);
        let patch = (sides[0] & sides[1]) * 0.5;
        let inverse = patch.inverse();
        Triangle {
            sides,
            patch,
            area: patch.norm(),
            gradients: sides.map(|s| (s | inverse) * 0.5),
        }
    }

    /// The unit normal.
    pub fn normal(&self) -> Patch {
        self.patch * (1.0 / self.area)
    }

    /// The sides turned a quarter turn in the triangle's plane: the commutator with the plane.
    pub fn turned_sides(&self) -> [Point; 3] {
        let plane = self.normal().hodge();
        self.sides.map(|s| s.commutator(plane))
    }

    /// Each corner's hat at a point of the triangle's plane: one plus the gradient's pairing with
    /// the step from the corner.
    pub fn hats(&self, corners: [Point; 3], p: Point) -> [f64; 3] {
        [0, 1, 2].map(|k| 1.0 + (self.gradients[k] & (p - corners[k])).s())
    }
}

/// The triangles of the faces on the vertices.
pub fn triangles(faces: &[[usize; 3]], vertices: &[Point]) -> Vec<Triangle> {
    faces
        .iter()
        .map(|f| Triangle::new(f.map(|v| vertices[v])))
        .collect()
}

/// The cotangent Laplacian `[V, V]`: each triangle's corner gradients paired with one another
/// over its area, pulled back to the vertices through both corners of each pair.
pub fn laplacian(faces: &[[usize; 3]], triangles: &[Triangle], n: usize) -> SparseMap<Scalar> {
    let triples = faces.iter().zip(triangles).flat_map(|(f, t)| {
        (0..9).map(move |k| {
            let (i, j) = (k / 3, k % 3);
            let cell = (t.gradients[i] | t.gradients[j]) * t.area;
            (f[i], f[j], cell)
        })
    });
    SparseMap::from_triples(triples, (n, n))
}

/// A value at each corner of each face, summed at the vertices.
pub fn at_vertices<V: Copy + core::ops::Add<Output = V>>(
    faces: &[[usize; 3]],
    corners: impl Iterator<Item = [V; 3]>,
    zero: V,
    n: usize,
) -> Vec<V> {
    let mut sums = vec![zero; n];
    for (f, values) in faces.iter().zip(corners) {
        for (v, value) in f.iter().zip(values) {
            sums[*v] = sums[*v] + value;
        }
    }
    sums
}

/// The area pull at each vertex: the quarter-turned facing sides summed, halved, which is the
/// gradient of the total area.
pub fn area_pull(faces: &[[usize; 3]], triangles: &[Triangle], n: usize) -> Vec<Point> {
    let turned = triangles.iter().map(|t| t.turned_sides().map(|s| s * 0.5));
    at_vertices(faces, turned, Point::zero(), n)
}

/// A relaxation step's surface: its vertices, total area, and mean curvature at the vertices
/// (zero on the wire), measured against the triangles' own winding.
#[derive(Clone, Debug)]
pub struct State {
    pub vertices: Vec<Point>,
    pub area: f64,
    pub curvature: Vec<f64>,
}

/// The interior vertices and those on the wire.
pub fn interior_and_wire<V: gax::Extensor<Slots = (), Coef = f64>>(
    mesh: &Mesh<V>,
) -> (Vec<usize>, Vec<usize>) {
    let rim = mesh.boundary_vertices();
    (0..rim.len()).partition(|v| !rim[*v])
}

/// The surface at each damped harmonic step towards least area, the wire held: each step moves
/// part of the way to the harmonic surface of its own triangles, keeping the triangles, and with
/// them the Laplacian, from changing too fast.
pub fn relax_plateau<V: gax::Extensor<Slots = (), Coef = f64>>(
    mesh: &Mesh<V>,
    initial: &[Point],
    step_size: f64,
    steps: usize,
) -> Vec<State> {
    let n = initial.len();
    let (interior, wire) = interior_and_wire(mesh);
    let mut slot = vec![None; n];
    for (k, v) in interior.iter().enumerate() {
        slot[*v] = Some(k);
    }
    let mut vertices = initial.to_vec();
    let mut states = Vec::new();
    for step in 0..=steps {
        let triangles = triangles(&mesh.faces, &vertices);
        let pull = area_pull(&mesh.faces, &triangles, n);
        // A vertex's area pull is twice its mean curvature times its third of the patches.
        let thirds = triangles.iter().map(|t| [t.patch * (1.0 / 3.0); 3]);
        let patches = at_vertices(&mesh.faces, thirds, Patch::zero(), n);
        let curvature = (0..n)
            .map(|v| match slot[v] {
                Some(_) => (pull[v] & patches[v].inverse()).s() / 2.0,
                None => 0.0,
            })
            .collect();
        let area = triangles.iter().map(|t| t.area).sum();
        states.push(State {
            vertices: vertices.clone(),
            area,
            curvature,
        });
        if step == steps {
            break;
        }
        // The harmonic surface of this step's triangles: the interior where their pull vanishes.
        let laplacian = laplacian(&mesh.faces, &triangles, n);
        let mut inner = Vec::new();
        let mut wire_pull = vec![Point::zero(); interior.len()];
        for (row, column, cell) in laplacian.couplings() {
            match (slot[row], slot[column]) {
                (Some(i), Some(j)) => inner.push((i, j, cell)),
                (Some(i), None) => wire_pull[i] += cell * initial[column],
                _ => {}
            }
        }
        let inner = SparseMap::from_triples(inner, (interior.len(), interior.len()));
        let goal = Field::new(wire_pull.iter().map(|p| -*p).collect());
        let harmonic = (&inner * Point::slot()).solve(&goal);
        let mut target = vertices.clone();
        for (v, p) in interior.iter().zip(harmonic.iter()) {
            target[*v] = p;
        }
        for v in &wire {
            target[*v] = initial[*v];
        }
        vertices = vertices
            .iter()
            .zip(&target)
            .map(|(a, b)| *a * (1.0 - step_size) + *b * step_size)
            .collect();
    }
    states
}

/// The harmonic conjugate of a minimal surface, up to a translation, and the sums of the sides
/// facing each vertex it is solved from: the corner gradients read the quarter-turned
/// differential over each triangle as half the bare facing side, and those sums cancel at
/// every interior vertex, so the conjugate is solved from the boundary alone (Pinkall and
/// Polthier, 1993).
pub fn conjugate_minimal_surface(
    faces: &[[usize; 3]],
    vertices: &[Point],
) -> (Vec<Point>, Vec<Point>) {
    let n = vertices.len();
    let triangles = triangles(faces, vertices);
    let laplacian = laplacian(faces, &triangles, n);
    let halves = triangles.iter().map(|t| t.sides.map(|s| s * 0.5));
    let flux = at_vertices(faces, halves, Point::zero(), n);
    // The Laplacian leaves translations free; the least-squares solve picks one conjugate.
    let conjugate = (&laplacian * Point::slot()).lstsq(&Field::new(flux.clone()));
    (conjugate.iter().collect(), flux)
}

/// The surface of Bonnet's family at an angle between a minimal surface and its conjugate,
/// with its unit normals at the vertices, the patches around each vertex summed.
pub fn bonnet(
    faces: &[[usize; 3]],
    vertices: &[Point],
    conjugate: &[Point],
    angle: f64,
) -> (Vec<Point>, Vec<Patch>) {
    let turn = phasor(angle);
    let (cosine, sine) = (turn.e20(), turn.e01());
    let surface: Vec<Point> = vertices
        .iter()
        .zip(conjugate)
        .map(|(v, c)| *v * cosine + *c * sine)
        .collect();
    let patches = triangles(faces, &surface).into_iter().map(|t| [t.patch; 3]);
    let normals = at_vertices(faces, patches, Patch::zero(), surface.len());
    (
        surface,
        normals.into_iter().map(|n| n.normalized()).collect(),
    )
}

/// The mean of points.
pub fn mean(points: &[Point]) -> Point {
    points.iter().copied().sum::<Point>() * (1.0 / points.len() as f64)
}

/// Points moved by their mean to the origin.
pub fn centred(points: &[Point]) -> Vec<Point> {
    let m = mean(points);
    points.iter().map(|p| *p - m).collect()
}

// --- scenes

pub const DISK_RADIUS: f64 = 1.0;
pub const DISK_DIVISIONS: usize = 10;
pub const ARCH_HEIGHT: f64 = 0.35;
pub const DOME_HEIGHT: f64 = 0.55;
pub const RELAXATION_STEP: f64 = 0.14;
pub const RELAXATION_STEPS: usize = 40;
pub const ENNEPER_RADIUS: f64 = 0.8;
pub const ENNEPER_DIVISIONS: usize = 10;
pub const CATENOID_HEIGHT: f64 = 1.15;
pub const CATENOID_SPAN: f64 = 0.85 * core::f64::consts::PI;
pub const CATENOID_HEIGHTS: usize = 22;
pub const CATENOID_AZIMUTHS: usize = 44;
pub const BONNET_FRAMES: usize = 36;

/// A flat triangulated disk, and its vertices as points.
pub fn disk(radius: f64, divisions: usize) -> (Mesh<Point>, Vec<Point>) {
    let mesh = Mesh::triangular_disk(radius, divisions);
    let flat: Vec<Point> = mesh.vertices.iter().map(point).collect();
    (mesh.with_vertices(Field::new(flat.clone())), flat)
}

/// A point of the plane `z = 0` as a planar spinor, `1` and `xy`: the point times the unit
/// `x` direction, whose powers carry the plane's harmonic polynomials.
fn planar(p: Point) -> gax::vga3d::Rotor {
    -(p * x_point())
}

/// A triangulated disk and a domed surface spanning a four-arch wire: the scalar part of the
/// planar spinor's fourth power lifts the rim in four arches, and a dome lifts the interior.
pub fn four_arch_dome(
    radius: f64,
    divisions: usize,
    arch: f64,
    dome: f64,
) -> (Mesh<Point>, Vec<Point>, Vec<Point>) {
    let (mesh, flat) = disk(radius, divisions);
    let initial = flat
        .iter()
        .map(|p| {
            let z = planar(*p);
            let fourth = (z * z) * (z * z);
            let envelope = 1.0 - p.norm_squared() / (radius * radius);
            *p + z_point() * (arch * fourth.s() + dome * envelope)
        })
        .collect();
    (mesh, flat, initial)
}

/// A triangulated disk, Enneper's minimal surface over it, and its exact conjugate, by the
/// Weierstrass–Enneper formulas in the planar spinor.
pub fn enneper_disk(
    radius: f64,
    divisions: usize,
) -> (Mesh<Point>, Vec<Point>, Vec<Point>, Vec<Point>) {
    let (mesh, flat) = disk(radius, divisions);
    let xy = Bivector::new(0.0, 0.0, 1.0);
    let (mut surface, mut conjugate) = (Vec::new(), Vec::new());
    for p in &flat {
        let z = planar(*p);
        let (squared, cubed) = (z * z, (z * z) * z);
        let along = (z.reverse() - cubed * (1.0 / 3.0)) * x_point();
        surface.push(along.grade::<2>() + z_point() * squared.s());
        let across = -((xy * (z.reverse() + cubed * (1.0 / 3.0))) * x_point());
        conjugate.push(across.grade::<2>() + z_point() * (xy * squared).s());
    }
    (mesh, flat, surface, conjugate)
}

/// A slit catenoid: a hyperbolic cosine profile turned about the z axis, its exact helicoid
/// conjugate, and its rings `[heights][azimuths]` as vertex indices.
pub struct Catenoid {
    pub mesh: Mesh<Point>,
    pub catenoid: Vec<Point>,
    #[cfg_attr(not(test), allow(dead_code))] // numga's check of the recovered conjugate
    pub helicoid: Vec<Point>,
    pub grid: Vec<Vec<usize>>,
}

pub fn slit_catenoid(height: f64, span: f64, heights: usize, azimuths: usize) -> Catenoid {
    let (hs, azs) = (
        gax_numga_examples::signal::linspace(-height, height, heights),
        gax_numga_examples::signal::linspace(-span, span, azimuths),
    );
    let xy = Bivector::new(0.0, 0.0, 1.0);
    let radial: Vec<Point> = azs
        .iter()
        .map(|a| (xy * (-a / 2.0)).exp() >> x_point())
        .collect();
    let quarter = (xy * (-core::f64::consts::PI / 4.0)).exp();
    let (mut catenoid, mut helicoid) = (Vec::new(), Vec::new());
    for h in &hs {
        for (r, a) in radial.iter().zip(&azs) {
            catenoid.push(*r * h.cosh() + z_point() * *h);
            helicoid.push((quarter >> *r) * h.sinh() - z_point() * *a);
        }
    }
    let grid: Vec<Vec<usize>> = (0..heights)
        .map(|i| (0..azimuths).map(|j| i * azimuths + j).collect())
        .collect();
    // Each quad's two triangles, counter-clockwise seen from outside.
    let mut faces = Vec::new();
    for i in 0..heights - 1 {
        for j in 0..azimuths - 1 {
            faces.push([grid[i][j], grid[i + 1][j + 1], grid[i + 1][j]]);
        }
    }
    for i in 0..heights - 1 {
        for j in 0..azimuths - 1 {
            faces.push([grid[i][j], grid[i][j + 1], grid[i + 1][j + 1]]);
        }
    }
    Catenoid {
        mesh: Mesh::new(Field::new(catenoid.clone()), faces),
        catenoid,
        helicoid,
        grid,
    }
}
