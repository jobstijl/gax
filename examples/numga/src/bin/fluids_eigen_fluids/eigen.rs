//! The math: a closed surface's corner gradients, its Laplace–Beltrami modes against the lumped
//! vertex mass, the modes' Lie–Poisson structure constants, and the flow in mode space, stepped
//! by classical Runge–Kutta. No drawing.
//!
//! In VGA3D a point is an antivector, intrinsic, an intrinsic bivector whose coefficients are its
//! coordinates (numga's `Point-`); the join of two edges is a triangle's patch, an intrinsic vector
//! normal to it whose size is twice its area (`Patch-`); a gradient is a vector (`Plane+`). The
//! side facing a corner, contracted with the inverse patch, is that corner's hat gradient. A
//! streamfunction's gradient met with the unit patch is the tangent flow it drives, and a
//! gradient joined with a flow is the rate at which the flow carries the field.

use gax::field::{Field, SparseMap};
use gax::mesh::Topology;
use gax::vga3d;
use gax_numga_examples::par_map;

/// A number (numga's `Scalar+`).
pub type Scalar = vga3d::Scalar;
/// A gradient (numga's `Plane+`).
pub type Plane = vga3d::Vector;
/// A point, an edge, a tangent velocity (numga's `Point-`).
pub type Point = vga3d::intrinsic::Bivector;
/// A triangle's area element, the join of two of its sides (numga's `Patch-`).
pub type Patch = vga3d::intrinsic::Vector;
/// A field's cell from vertex potentials to face gradients (numga's `Plane+ <- Scalar+`).
pub type GradientMap = vga3d::Vector<(Scalar,)>;
/// A scalar map: the cells of the stiffness and the mass once read as numbers.
pub type ScalarMap = vga3d::Scalar<(Scalar,)>;

/// The point at `(x, y, z)`: the metric dual of its vector.
pub fn point(x: f64, y: f64, z: f64) -> Point {
    Plane::new(x, y, z).hodge()
}

/// A point's coordinates, read with the planes across the axes.
pub fn coordinates(p: Point) -> [f64; 3] {
    [
        (Plane::new(1.0, 0.0, 0.0) & p).s(),
        (Plane::new(0.0, 1.0, 0.0) & p).s(),
        (Plane::new(0.0, 0.0, 1.0) & p).s(),
    ]
}

/// A closed triangle mesh with what its flows need: the corner gradients of each face, the
/// gradient map, and the stiffness and lumped mass of the vertices.
pub struct Surface {
    pub mesh: Topology<3>,
    pub vertices: Field<Point>,
    pub patches: Field<Patch>,
    pub areas: Field<Scalar>,
    /// The unit patches.
    pub normals: Field<Patch>,
    /// The hat gradient of each corner on each face: the side facing it contracted with the
    /// inverse patch, halved.
    pub corner_gradients: [Field<Plane>; 3],
    /// Vertex values to the constant gradient on each face.
    pub gradient: SparseMap<GradientMap>,
    /// A third of the area of every face around each vertex.
    pub dual_areas: Field<Scalar>,
}

impl Surface {
    pub fn new(mesh: Topology<3>, vertices: Field<Point>) -> Self {
        let edges = &mesh.d::<Scalar>(0) * &vertices;
        let sides = mesh.facets(2).read_all(&edges);
        let patches = sides[0].zip_with(&sides[1], |a, b| (a & b) * 0.5);
        let areas = patches.map(|p| Scalar::new(p.norm()));
        let normals = patches.zip_with(&areas, |p, a| p * a.inverse());
        let corner_gradients: [Field<Plane>; 3] =
            [0, 1, 2].map(|k| sides[k].zip_with(&patches, |s, p| (s | p.inverse()) * 0.5));
        let gradient = mesh
            .incidence(2, 0)
            .assemble(|f, k| corner_gradients[k].site(f) * Scalar::slot());
        let mut dual_areas = Field::<Scalar>::zeros(mesh.len(0));
        for f in 0..mesh.len(2) {
            for &v in mesh.simplex(2, f) {
                dual_areas.sites_mut()[v] += areas.site(f) * (1.0 / 3.0);
            }
        }
        Surface {
            mesh,
            vertices,
            patches,
            areas,
            normals,
            corner_gradients,
            gradient,
            dual_areas,
        }
    }

    /// The stiffness: the area-weighted isotropic material, a gradient to its metric dual,
    /// pulled back to the vertices through the gradient's adjugate (its transpose through the
    /// join), its cells read as numbers.
    pub fn stiffness(&self) -> SparseMap<ScalarMap> {
        let material = Plane::slot().hodge();
        let response = SparseMap::from_diagonal(&self.areas.map(|a| material * a.s()));
        let pulled = self
            .gradient
            .adjugate()
            .compose(&response.compose(&self.gradient));
        pulled.map_cells(|c| c.undual())
    }

    /// The lumped vertex mass.
    pub fn mass(&self) -> SparseMap<ScalarMap> {
        SparseMap::from_diagonal(&self.dual_areas.map(|a| Scalar::slot() * a.s()))
    }

    /// The `count` lowest Laplace–Beltrami modes after the constant one: their eigenvalues and
    /// the modes, orthonormal in the mass. A larger Lanczos `block` finds the copies of a value
    /// that a symmetric mesh repeats (the sphere's bands).
    pub fn modes(&self, count: usize, block: usize) -> (Vec<f64>, Vec<Field<Scalar>>) {
        let (values, fields) = self.stiffness().eigh_block(&self.mass(), count + 1, block);
        (values[1..].to_vec(), fields[1..].to_vec())
    }

    /// The tangent flow a streamfunction drives on each face: its gradient met with the unit
    /// patch.
    #[cfg(test)]
    pub fn velocity(&self, stream: &Field<Scalar>) -> Field<Point> {
        self.gradient
            .of(stream)
            .zip_with(&self.normals, |g, n| g ^ n)
    }

    /// A vertex field's projection on the modes through the mass: `⟨φᵢ, M f⟩`.
    pub fn project(&self, modes: &[Field<Scalar>], field: &Field<Scalar>) -> Field<Scalar> {
        let weighted = field.zip_with(&self.dual_areas, |f, a| f * a);
        modes
            .iter()
            .map(|m| Scalar::new(m.dot(&weighted)))
            .collect()
    }
}

/// The surface field of modal amplitudes: the modes they weigh, superposed (numga's
/// `superposition`, the adjoint of the modes' batch).
pub fn superpose(modes: &[Field<Scalar>], amplitudes: &Field<Scalar>) -> Field<Scalar> {
    let mut out = Field::<Scalar>::zeros(modes[0].len());
    for (m, a) in modes.iter().zip(amplitudes.iter()) {
        out = &out + &m.map(|x| x * a);
    }
    out
}

/// The modes' Lie–Poisson structure constants, a bracket `[modes <- (modes, modes)]`: entry
/// `(a, b, c)` integrates mode `c` against the Jacobian of modes `a` and `b`, the wedge of their
/// gradients read on the surface. Bound to a streamfunction and a vorticity it gives the advection
/// rate of every mode. (gax's field maps take one input; the bracket is a dense array of its
/// `n³` numbers.)
pub struct Bracket {
    pub n: usize,
    /// `c[(a n + b) n + c]`.
    pub c: Vec<f64>,
}

impl Bracket {
    /// The constants of the modes on a surface (numga's `structure_constants`).
    ///
    /// On each face the gradients are constant and tangent, and the Jacobian pairing
    /// `gₐ & (g_b ^ P)` is an antisymmetric form on the face's plane: in an orthonormal frame
    /// `e₁, e₂` of the face, `J(e₁, e₂) (uₐ v_b − vₐ u_b)` with `u`, `v` the gradients' readings
    /// on the frame. Against a Jacobian constant on the face, a mode integrates through its mean
    /// over the corners. So the constants are `Z(a, b, c) − Z(b, a, c)`, with
    /// `Z(a, b, c) = Σ_f J_f uₐ v_b m_c`: per face a rank-one product, summed one `a` at a time
    /// across the cores, `n³ F` products in all, held in `n³` numbers.
    pub fn new(surface: &Surface, modes: &[Field<Scalar>]) -> Self {
        let n = modes.len();
        let faces = surface.mesh.len(2);
        // Each face's orthonormal frame, the pairing's value on it, and each corner gradient's
        // readings on the frame.
        let mut readings = vec![[[0.0; 2]; 3]; faces];
        let mut pairing = vec![0.0; faces];
        for f in 0..faces {
            let g = [0, 1, 2].map(|k| surface.corner_gradients[k].site(f));
            let e1 = g[0].normalized();
            let e2 = (g[1] - e1 * (g[1] | e1).s()).normalized();
            pairing[f] = (e1 & (e2 ^ surface.patches.site(f))).s();
            for k in 0..3 {
                readings[f][k] = [(g[k] | e1).s(), (g[k] | e2).s()];
            }
        }
        // `u`, `v` and the corner means of every mode on every face, face by face.
        let mut u = vec![0.0; faces * n];
        let mut v = vec![0.0; faces * n];
        let mut m = vec![0.0; faces * n];
        for (i, mode) in modes.iter().enumerate() {
            for f in 0..faces {
                let corners = surface.mesh.simplex(2, f);
                for (k, &c) in corners.iter().enumerate() {
                    let value = mode.site(c).s();
                    u[f * n + i] += readings[f][k][0] * value;
                    v[f * n + i] += readings[f][k][1] * value;
                    m[f * n + i] += value / 3.0;
                }
            }
        }
        let rows: Vec<usize> = (0..n).collect();
        let z: Vec<Vec<f64>> = par_map(&rows, |&a| {
            let mut out = vec![0.0; n * n];
            for f in 0..faces {
                let x = pairing[f] * u[f * n + a];
                if x == 0.0 {
                    continue;
                }
                let (vf, mf) = (&v[f * n..(f + 1) * n], &m[f * n..(f + 1) * n]);
                for (b, row) in out.chunks_exact_mut(n).enumerate() {
                    let y = x * vf[b];
                    for (o, &mc) in row.iter_mut().zip(mf) {
                        *o += y * mc;
                    }
                }
            }
            out
        });
        let mut c = vec![0.0; n * n * n];
        for a in 0..n {
            for b in 0..n {
                for k in 0..n {
                    c[(a * n + b) * n + k] = z[a][b * n + k] - z[b][a * n + k];
                }
            }
        }
        Bracket { n, c }
    }

    /// The constant `(a, b, c)`.
    #[cfg(test)]
    pub fn at(&self, a: usize, b: usize, c: usize) -> f64 {
        self.c[(a * self.n + b) * self.n + c]
    }

    /// The bracket bound to a streamfunction and a vorticity: `Σ C(a, b, c) s_b ω_c` for every
    /// `a`, each row of constants read against the vorticity and the results against the
    /// streamfunction.
    pub fn of(&self, stream: &Field<Scalar>, vorticity: &Field<Scalar>) -> Field<Scalar> {
        let n = self.n;
        let s: Vec<f64> = stream.iter().map(|x| x.s()).collect();
        let w: Vec<f64> = vorticity.iter().map(|x| x.s()).collect();
        (0..n)
            .map(|a| {
                let block = &self.c[a * n * n..(a + 1) * n * n];
                let sum = block
                    .chunks_exact(n)
                    .zip(&s)
                    .map(|(row, sb)| sb * dot(row, &w))
                    .sum();
                Scalar::new(sum)
            })
            .collect()
    }
}

/// The dot product of two rows, summed in eight lanes so that it runs on the vector units.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    let mut lanes = [0.0; 8];
    let ((ca, ra), (cb, rb)) = (a.as_chunks::<8>(), b.as_chunks::<8>());
    let tail: f64 = ra.iter().zip(rb).map(|(x, y)| x * y).sum();
    for (x, y) in ca.iter().zip(cb) {
        for k in 0..8 {
            lanes[k] += x[k] * y[k];
        }
    }
    lanes.iter().sum::<f64>() + tail
}

/// The rate of the modal vorticity: the Lie–Poisson advection bracket of streamfunction and
/// vorticity, less viscous diffusion.
pub fn vorticity_rate(
    vorticity: &Field<Scalar>,
    bracket: &Bracket,
    eigenvalues: &Field<Scalar>,
    viscosity: f64,
) -> Field<Scalar> {
    let stream = vorticity.zip_with(eigenvalues, |w, l| w * l.inverse());
    let advection = bracket.of(&stream, vorticity);
    let diffusion = vorticity.zip_with(eigenvalues, |w, l| w * l * viscosity);
    &advection - &diffusion
}

/// The modal vorticity after each of `steps` classical fourth-order Runge–Kutta steps, the
/// start first.
pub fn advect(
    start: &Field<Scalar>,
    bracket: &Bracket,
    eigenvalues: &Field<Scalar>,
    viscosity: f64,
    dt: f64,
    steps: usize,
) -> Vec<Field<Scalar>> {
    let rate = |w: &Field<Scalar>| vorticity_rate(w, bracket, eigenvalues, viscosity);
    let mut states = vec![start.clone()];
    let mut w = start.clone();
    for _ in 0..steps {
        let first = rate(&w);
        let second = rate(&(&w + &first.map(|x| x * (dt / 2.0))));
        let third = rate(&(&w + &second.map(|x| x * (dt / 2.0))));
        let fourth = rate(&(&w + &third.map(|x| x * dt)));
        let sum = &(&first + &second.map(|x| x * 2.0)) + &(&third.map(|x| x * 2.0) + &fourth);
        w = &w + &sum.map(|x| x * (dt / 6.0));
        states.push(w.clone());
    }
    states
}

/// The kinetic energy of a modal vorticity, `Σ ω² / λ / 2`, and its enstrophy, `Σ ω² / 2`.
pub fn invariants(vorticity: &Field<Scalar>, eigenvalues: &Field<Scalar>) -> (f64, f64) {
    let energy = vorticity
        .iter()
        .zip(eigenvalues.iter())
        .map(|(w, l)| (w * w * l.inverse()).s())
        .sum::<f64>();
    (energy / 2.0, vorticity.dot(vorticity) / 2.0)
}

/// Opposite vorticity bands at `±latitude` with a ripple along them, of zero total circulation:
/// the bands' heights shifted by the product of three readings along meridians a third of a turn
/// apart, which vanishes smoothly at the poles.
pub fn shear_layers(
    vertices: &Field<Point>,
    dual_areas: &Field<Scalar>,
    latitude: f64,
    width: f64,
    strength: f64,
    wave_amplitude: f64,
) -> Field<Scalar> {
    let up = Plane::new(0.0, 0.0, 1.0);
    let meridians = [0.0, 1.0, 2.0].map(|k| {
        let angle = core::f64::consts::PI * k / 3.0;
        (vga3d::Bivector::new(0.0, 0.0, 1.0) * (-angle / 2.0)).exp() >> Plane::new(1.0, 0.0, 0.0)
    });
    let band = gax_numga_examples::signal::wave(latitude);
    let field = vertices.map(|p| {
        let height = (up & p).s();
        let readings = meridians.map(|m| (m & p).s());
        let shift = -4.0 * wave_amplitude * readings[0] * readings[1] * readings[2];
        let north = (height - band - shift) / width;
        let south = (height + band - shift) / width;
        Scalar::new(strength * ((-0.5 * north * north).exp() - (-0.5 * south * south).exp()))
    });
    let weighted: f64 = field
        .zip_with(dual_areas, |f, a| f * a)
        .iter()
        .map(|x| x.s())
        .sum();
    let total: f64 = dual_areas.iter().map(|a| a.s()).sum();
    field.map(|f| f - Scalar::new(weighted / total))
}
